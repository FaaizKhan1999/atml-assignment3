"""lm-evaluation-harness runs for Task 1.

GSM8K: ``gsm8k_cot`` zero-shot, Qwen3 chat template with thinking disabled,
greedy, 1,024 new tokens, stock stop sequences and answer filters
(primary: flexible-extract exact match; unextractable answers are wrong).
HellaSwag / BoolQ / ARC-Easy: stock likelihood scoring and completion prompts,
no chat template.

lm-eval's HF backend applies chat templates without template kwargs, which
would leave Qwen3 in thinking mode, so ``NoThinkHFLM`` overrides
``apply_chat_template`` to pass ``enable_thinking=False``. Every GSM8K run
verifies from the logged prompts that the empty think block is present.

Per-example outputs are parsed from lm-eval's logged samples into compact
JSONL (extracted answers, truncation flags, per-choice log-likelihoods). The
log format differs slightly across lm-eval versions; the parsers accept both
the list and the ``gen_args_N``/``arg_N`` dict layouts.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

from common.logging_utils import get_logger, package_versions, write_json

log = get_logger()
EMPTY_THINK = "<think>\n\n</think>\n\n"


# ----------------------------------------------------------------------------- model wrapper

def make_lm(model_dir: str | Path, tokenizer_name: str, batch_size: int, max_gen_toks: int):
    """HF-backend LM with the original Hub tokenizer (never a re-saved copy).

    ``max_gen_toks`` is not an HFLM constructor argument in every lm-eval version
    (newer ones forward unknown kwargs to the model), so it is set on the
    instance afterwards; the GSM8K gen_kwargs also set it explicitly.
    """
    from lm_eval.models.huggingface import HFLM

    class NoThinkHFLM(HFLM):
        def apply_chat_template(self, chat_history, add_generation_prompt: bool = True, *args, **kwargs) -> str:
            kw = {"tokenize": False, "add_generation_prompt": add_generation_prompt, "enable_thinking": False}
            if not add_generation_prompt:
                kw["continue_final_message"] = True
            return self.tokenizer.apply_chat_template(chat_history, **kw)

    lm = NoThinkHFLM(pretrained=str(model_dir), tokenizer=tokenizer_name, dtype="float16", batch_size=batch_size,
                     device="cuda", trust_remote_code=False)
    if hasattr(lm, "_max_gen_toks"):
        lm._max_gen_toks = max_gen_toks
    return lm


# ----------------------------------------------------------------------------- parsing helpers

def _arguments(entry: dict) -> list[tuple]:
    a = entry.get("arguments") or []
    if isinstance(a, dict):  # {"gen_args_0": {"arg_0": ..., "arg_1": ...}, ...}
        out = []
        for _, v in sorted(a.items(), key=lambda kv: int(str(kv[0]).rsplit("_", 1)[-1])):
            out.append(tuple(v[k] for k in sorted(v, key=lambda s: int(str(s).rsplit("_", 1)[-1]))))
        return out
    return [tuple(x) if isinstance(x, (list, tuple)) else (x,) for x in a]


def _first_str(x):
    while isinstance(x, (list, tuple)):
        if not x:
            return None
        x = x[0]
    return x


def _loglik(x) -> float:
    while isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]
    if isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def _metric(v):
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    return None if v is None else float(v)


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def read_jsonl_gz(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_generation(samples: list[dict], tokenizer, max_gen_toks: int) -> list[dict]:
    rows: dict = {}
    for e in samples:
        did = int(e["doc_id"])
        filt = str(e.get("filter", "none")).replace("-", "_")
        r = rows.get(did)
        if r is None:
            args = _arguments(e)
            response = _first_str(e.get("resps")) or ""
            n_tok = len(tokenizer(response, add_special_tokens=False)["input_ids"])
            r = rows[did] = {"doc_id": did, "target": e.get("target"), "prompt": args[0][0] if args else None,
                             "response": response, "n_response_tokens": n_tok,
                             # re-tokenizing decoded text can differ by a token or two from the generated ids
                             "truncated": n_tok >= max_gen_toks - 2}
        r[f"extracted_{filt}"] = _first_str(e.get("filtered_resps"))
        r[f"em_{filt}"] = _metric(e.get("exact_match"))
    return [rows[k] for k in sorted(rows)]


def parse_multiple_choice(samples: list[dict]) -> list[dict]:
    rows = []
    for e in sorted(samples, key=lambda x: int(x["doc_id"])):
        args = _arguments(e)
        lls = [_loglik(x) for x in (e.get("filtered_resps") or e.get("resps") or [])]
        rows.append({"doc_id": int(e["doc_id"]), "gold": e.get("target"),
                     "prompt": args[0][0] if args else None,
                     "continuations": [a[1] for a in args if len(a) > 1],
                     "loglik": lls, "pred_raw": int(max(range(len(lls)), key=lls.__getitem__)) if lls else None,
                     "acc": _metric(e.get("acc")), "acc_norm": _metric(e.get("acc_norm"))})
    return rows


# ----------------------------------------------------------------------------- running a benchmark

def run_benchmark(lm, name: str, task: str, *, generation: bool, chat: bool, cfg, out_dir: Path,
                  tokenizer, limit: int | None = None) -> dict:
    import lm_eval
    import torch
    from lm_eval.tasks import TaskManager

    ev = cfg.eval
    kwargs = dict(model=lm, tasks=[task], num_fewshot=0, log_samples=True, apply_chat_template=chat,
                  limit=limit, bootstrap_iters=ev.bootstrap_iters, random_seed=ev.seed, numpy_random_seed=ev.seed,
                  torch_random_seed=ev.seed, fewshot_random_seed=ev.seed, task_manager=TaskManager())
    if generation:
        kwargs["gen_kwargs"] = f"max_gen_toks={ev.gsm8k.max_gen_toks},do_sample=False"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = lm_eval.simple_evaluate(**kwargs)
    seconds = time.perf_counter() - t0
    samples = res["samples"][task]
    agg = {k: v for k, v in res["results"][task].items() if k != "alias"}

    summary = {"benchmark": name, "task": task, "limit": limit, "seconds": seconds,
               "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
               "batch_size": ev.batch_size, "chat_template": chat, "num_fewshot": 0,
               "lm_eval_version": package_versions(["lm_eval"])["lm_eval"],
               "n_configs": res.get("n-samples", {}).get(task), "metrics": agg}
    if generation:
        rows = parse_generation(samples, tokenizer, ev.gsm8k.max_gen_toks)
        pf = ev.gsm8k.primary_filter.replace("-", "_")
        prompts_ok = [bool(r["prompt"]) and r["prompt"].endswith(EMPTY_THINK) for r in rows]
        if not all(prompts_ok):
            bad = rows[prompts_ok.index(False)]["prompt"]
            raise RuntimeError(f"GSM8K prompt does not end with the empty think block:\n{bad[-300:]!r}")
        n = len(rows)
        summary.update({
            "max_gen_toks": ev.gsm8k.max_gen_toks,
            "n_examples": n,
            "primary": {"metric": f"exact_match,{ev.gsm8k.primary_filter}",
                        "value": agg.get(f"exact_match,{ev.gsm8k.primary_filter}")},
            "strict_match": agg.get("exact_match,strict-match"),
            "unextractable_rate": sum(r.get(f"extracted_{pf}") in (None, "", "[invalid]") for r in rows) / n,
            "truncation_rate": sum(r["truncated"] for r in rows) / n,
            "mean_response_tokens": sum(r["n_response_tokens"] for r in rows) / n,
            "truncated_and_wrong": sum(r["truncated"] and not r.get(f"em_{pf}") for r in rows),
            "prompt_example": rows[0]["prompt"],
        })
        _write_jsonl_gz(out_dir / f"{name}_samples.jsonl.gz", rows)
    else:
        rows = parse_multiple_choice(samples)
        metrics = list(cfg.eval.likelihood_tasks[name].metrics)
        summary.update({"n_examples": len(rows),
                        "primary": {"metric": f"{metrics[0]},none", "value": agg.get(f"{metrics[0]},none")},
                        "raw_acc": agg.get("acc,none"),
                        "prompt_example": rows[0]["prompt"] if rows else None})
        _write_jsonl_gz(out_dir / f"{name}_samples.jsonl.gz", rows)
    write_json(out_dir / f"{name}.json", summary)
    log.info("%s: %s = %s (%d examples, %.1f min)", name, summary["primary"]["metric"], summary["primary"]["value"],
             summary["n_examples"], seconds / 60)
    return summary
