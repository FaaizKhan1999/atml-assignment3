"""Task 3, Step 4: parallel sampling vs a single longer response vs sequential refinement (GPU).

    python -m task3.live --push

The first 50 held-out MATH questions (splits/task3/math_split.json, "live"), with
Qwen/Qwen2.5-3B-Instruct at the README's pinned revision, FP16, no quantization, batch size 1,
temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0, native chat template, system message
"You are a helpful assistant.", and the problem followed by two newlines and the fixed instruction.
Same 3,072-token output allowance per question:
  single      one response, up to 3,072 new tokens
  parallel    three independent responses, up to 1,024 each; self-consistency picks the answer
              (answer groups by the same Math-Verify equivalence, NO_ANSWER included; ties -> earliest)
  refinement  the FIRST parallel response is the initial response; it is appended as an assistant
              message followed by the fixed review instruction; one critique-and-revision response up to
              2,048 tokens; the revised answer is returned (no gold answers or correctness feedback)
Every call is independently seeded (seed derived from 6304, question, strategy and call index).
The last \\boxed{...} is extracted with the balanced-brace rule from the data README (absent, empty or
unclosed -> no answer). Grading uses the Math-Verify equivalence of the offline experiments.

Recorded per call: prompt, response, seed, input/output tokens, truncation (cap reached without an
end token), extracted answer, wall time (CUDA-synchronized, model loading excluded). Costs: the shared
initial response is charged to BOTH parallel and refinement; refinement also pays the longer revision
prompt (input tokens) and every critique token. Resumable per question.
Outputs: results/task3/live/{calls.jsonl, per_question.json, summary.json}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch

from common.config import REPO_ROOT, add_config_args, load_config
from common.logging_utils import get_logger, read_json, run_metadata, write_json
from common.publish import try_push
from task3.answers import NO_ANSWER, MathEquivalence
from task3.data import SPLITS, find_folder

log = get_logger()
OUT = REPO_ROOT / "results" / "task3" / "live"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}
CAPS = {"single": 3072, "parallel": 1024, "revision": 2048}


def last_boxed(text: str) -> str | None:
    """Last literal \\boxed{ with balanced braces; absent, empty or unclosed -> None."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    start = j
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                inner = text[start:j].strip()
                return inner or None
        j += 1
    return None


def call_seed(base: int, qid: str, strategy: str, k: int) -> int:
    return int(hashlib.sha256(f"{base}|{qid}|{strategy}|{k}".encode()).hexdigest()[:8], 16)


class Generator:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        from common.model_utils import _dtype_kwarg
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, attn_implementation="sdpa",
                                                          **{_dtype_kwarg(): torch.float16}).cuda().eval()
        self.load_seconds = time.perf_counter() - t0
        eos = self.model.generation_config.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]
        self.GenerationConfig = GenerationConfig
        self.commit = getattr(self.model.config, "_commit_hash", None)

    def prompt(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, messages: list[dict], max_new: int, seed: int) -> dict:
        text = self.prompt(messages)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        gc = self.GenerationConfig(max_new_tokens=max_new, eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id
                                   or self.eos[0], **SAMPLING)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**enc, generation_config=gc)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        gen = out[0, enc["input_ids"].shape[1]:].tolist()
        hit_eos = any(t in self.eos for t in gen)
        if hit_eos:
            gen = gen[: next(i for i, t in enumerate(gen) if t in self.eos) + 1]
        n_out = len(gen)
        response = self.tok.decode([t for t in gen if t not in self.eos], skip_special_tokens=True)
        return {"prompt": text, "input_tokens": int(enc["input_ids"].shape[1]), "output_tokens": n_out,
                "truncated": (not hit_eos) and n_out >= max_new, "response": response,
                "extracted": last_boxed(response), "seconds": secs, "seed": seed, "max_new_tokens": max_new}


def self_consistency(answers: list[str | None], meq: MathEquivalence) -> tuple[str | None, int]:
    """Vote over equivalence groups (NO_ANSWER is a group); ties -> earliest supporting response."""
    reps, groups = [], []
    for a in answers:
        if a is None:
            groups.append(NO_ANSWER)
            continue
        gid = next((g for g, r in reps if meq.equivalent(a, r)), None)
        if gid is None:
            gid = len(reps) + 1
            reps.append((gid, a))
        groups.append(gid)
    counts, first = {}, {}
    for pos, g in enumerate(groups):
        counts[g] = counts.get(g, 0) + 1
        first.setdefault(g, pos)
    best = max(counts, key=lambda g: (counts[g], -first[g]))
    return (None if best == NO_ANSWER else dict(reps)[best]), first[best]


def load_questions(cfg, live_ids: list[str]) -> dict:
    import pyarrow.parquet as pq

    folder = find_folder(cfg.data.folders["math"])
    t = pq.read_table(folder / "generations.parquet", columns=["sample_id", "rollout_id", "question_text", "gold_answer"])
    out = {}
    for s, r, q, g in zip(*(t.column(c).to_pylist() for c in ("sample_id", "rollout_id", "question_text", "gold_answer"))):
        if str(s) in live_ids and str(s) not in out:
            out[str(s)] = {"question": q, "gold": g}
    missing = [q for q in live_ids if q not in out]
    if missing:
        raise KeyError(f"live questions not found: {missing[:3]}")
    return out


def summarize(per_q: dict, meq: MathEquivalence, n_boot: int, seed: int, extra: dict) -> dict:
    qids = sorted(per_q)
    strat = ["single", "parallel", "refinement"]
    acc = {s: np.array([per_q[q][s]["correct"] for q in qids], dtype=float) for s in strat}
    acc["initial"] = np.array([per_q[q]["initial_correct"] for q in qids], dtype=float)
    out = {"n_questions": len(qids), "strategies": {}}
    for s in strat:
        calls = [c for q in qids for c in per_q[q][s]["calls"]]
        out["strategies"][s] = {
            "accuracy": float(acc[s].mean()),
            "mean_input_tokens": float(np.mean([per_q[q][s]["input_tokens"] for q in qids])),
            "mean_output_tokens": float(np.mean([per_q[q][s]["output_tokens"] for q in qids])),
            "mean_total_tokens": float(np.mean([per_q[q][s]["input_tokens"] + per_q[q][s]["output_tokens"] for q in qids])),
            "mean_seconds": float(np.mean([per_q[q][s]["seconds"] for q in qids])),
            "total_seconds": float(np.sum([per_q[q][s]["seconds"] for q in qids])),
            "call_truncation_rate": float(np.mean([c["truncated"] for c in calls])),
            "call_extraction_failure_rate": float(np.mean([c["extracted"] is None for c in calls])),
            "final_no_answer_rate": float(np.mean([per_q[q][s]["answer"] is None for q in qids])),
            "n_calls": len(calls)}
    ini, rev = acc["initial"], acc["refinement"]
    out["refinement_transitions"] = {"wrong_to_correct": int(((ini == 0) & (rev == 1)).sum()),
                                     "correct_to_wrong": int(((ini == 1) & (rev == 0)).sum()),
                                     "correct_to_correct": int(((ini == 1) & (rev == 1)).sum()),
                                     "wrong_to_wrong": int(((ini == 0) & (rev == 0)).sum()),
                                     "answer_changed": int(sum(per_q[q]["refinement"]["answer_changed"] for q in qids)),
                                     "initial_accuracy": float(ini.mean())}
    idx = np.random.default_rng(seed).integers(0, len(qids), size=(n_boot, len(qids)))
    out["paired_bootstrap"] = {}
    for a, b in (("parallel", "single"), ("refinement", "single"), ("refinement", "parallel"), ("refinement", "initial")):
        d = acc[a] - acc[b]
        s = d[idx].mean(1)
        out["paired_bootstrap"][f"{a}_minus_{b}"] = {"diff": float(d.mean()),
                                                     "ci95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]}
    out.update(extra)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/task3/base.yaml")
    parser.add_argument("--limit", type=int, default=None, help="first K questions only (smoke test; not pushed)")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Live generation needs a GPU.")
    cfg = load_config(args.config, args.set)
    base_seed = int(cfg.analysis.seed)
    live = read_json(SPLITS / "math_split.json")["live"]
    if args.limit:
        live = live[: args.limit]
    out_dir = OUT if not args.limit else OUT.parent / "live_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    push = args.push and not args.limit
    qs = load_questions(cfg, live)
    meq = MathEquivalence()
    gen = Generator()
    torch.cuda.reset_peak_memory_stats()
    per_path = out_dir / "per_question.json"
    per_q = read_json(per_path) if per_path.exists() else {}
    calls_fh = open(out_dir / "calls.jsonl", "a", encoding="utf-8")

    for n, qid in enumerate(live, 1):
        if qid in per_q:
            continue
        q = qs[qid]
        user = q["question"] + "\n\n" + INSTRUCTION
        base_msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        rec = {}

        def run(strategy, k, msgs, cap):
            c = gen.generate(msgs, cap, call_seed(base_seed, qid, strategy, k))
            c.update({"question": qid, "strategy": strategy, "call": k})
            calls_fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            calls_fh.flush()
            return c

        single = run("single", 0, base_msgs, CAPS["single"])
        par = [run("parallel", k, base_msgs, CAPS["parallel"]) for k in range(3)]
        initial = par[0]  # shared initial response for refinement
        rev_msgs = base_msgs + [{"role": "assistant", "content": initial["response"]}, {"role": "user", "content": REVIEW}]
        revision = run("refinement", 1, rev_msgs, CAPS["revision"])

        grade = lambda a: a is not None and meq.equivalent(str(q["gold"]).strip(), a)  # noqa: E731
        sc_answer, _ = self_consistency([c["extracted"] for c in par], meq)
        brief = lambda c: {k: c[k] for k in ("call", "seed", "input_tokens", "output_tokens", "truncated", "extracted", "seconds")}  # noqa: E731
        rec["single"] = {"answer": single["extracted"], "correct": grade(single["extracted"]), "calls": [brief(single)],
                         "input_tokens": single["input_tokens"], "output_tokens": single["output_tokens"],
                         "seconds": single["seconds"]}
        rec["parallel"] = {"answer": sc_answer, "correct": grade(sc_answer), "calls": [brief(c) for c in par],
                           "input_tokens": sum(c["input_tokens"] for c in par),
                           "output_tokens": sum(c["output_tokens"] for c in par),
                           "seconds": sum(c["seconds"] for c in par)}
        changed = not (initial["extracted"] is not None and revision["extracted"] is not None
                       and meq.equivalent(initial["extracted"], revision["extracted"])) and \
            not (initial["extracted"] is None and revision["extracted"] is None)
        rec["refinement"] = {"answer": revision["extracted"], "correct": grade(revision["extracted"]),
                             "calls": [brief(initial), brief(revision)],
                             "input_tokens": initial["input_tokens"] + revision["input_tokens"],
                             "output_tokens": initial["output_tokens"] + revision["output_tokens"],
                             "seconds": initial["seconds"] + revision["seconds"], "answer_changed": bool(changed)}
        rec["initial_answer"] = initial["extracted"]
        rec["initial_correct"] = grade(initial["extracted"])
        rec["gold"] = q["gold"]
        per_q[qid] = rec
        write_json(per_path, per_q)
        log.info("[%d/%d] %s: single %s | parallel %s | initial %s -> revised %s", n, len(live), qid,
                 rec["single"]["correct"], rec["parallel"]["correct"], rec["initial_correct"], rec["refinement"]["correct"])
        if push and n % 10 == 0:
            try_push([out_dir], f"task3/live: {len(per_q)} questions")
    calls_fh.close()

    extra = {"model": MODEL, "revision": REVISION, "model_commit": gen.commit, "dtype": "float16", "batch_size": 1,
             "sampling": SAMPLING, "caps": CAPS, "system": SYSTEM, "instruction": INSTRUCTION, "review_instruction": REVIEW,
             "model_load_seconds": gen.load_seconds, "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
             "gpu": torch.cuda.get_device_name(0), "scorer_math_verify_timeouts": meq.timeout_report(),
             "environment": run_metadata()}
    summary = summarize(per_q, meq, int(cfg.analysis.bootstrap), base_seed, extra)
    write_json(out_dir / "summary.json", summary)
    print("\n" + "=" * 92)
    print(f"{'strategy':<12}{'acc':>7}{'in tok':>9}{'out tok':>9}{'total':>9}{'sec/q':>8}{'trunc':>8}{'no-extr':>9}")
    for s, d in summary["strategies"].items():
        print(f"{s:<12}{d['accuracy']:>7.3f}{d['mean_input_tokens']:>9.0f}{d['mean_output_tokens']:>9.0f}"
              f"{d['mean_total_tokens']:>9.0f}{d['mean_seconds']:>8.1f}{d['call_truncation_rate']:>8.3f}"
              f"{d['call_extraction_failure_rate']:>9.3f}")
    t = summary["refinement_transitions"]
    print(f"refinement: initial acc {t['initial_accuracy']:.3f}; wrong->correct {t['wrong_to_correct']}, "
          f"correct->wrong {t['correct_to_wrong']}, answer changed {t['answer_changed']}")
    for k, v in summary["paired_bootstrap"].items():
        print(f"{k:<28} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print("=" * 92)
    if push:
        try_push([out_dir], "task3/live: finished")


if __name__ == "__main__":
    main()
