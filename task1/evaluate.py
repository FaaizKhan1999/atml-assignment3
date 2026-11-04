"""Evaluate the Task 1 conditions: lm-eval benchmarks and qualitative generations.

    python -m task1.evaluate                                   # all 7 conditions, all parts
    python -m task1.evaluate --conditions baseline full_ft_final
    python -m task1.evaluate --parallel                        # split conditions across all visible GPUs
    python -m task1.evaluate --conditions baseline --limit 20  # smoke test (scratch output, no push)

Outputs per condition (results/task1/eval/<condition>/):
  <bench>.json               aggregate metrics, timing, settings, lm-eval version
  <bench>_samples.jsonl.gz   per-example outputs (GSM8K: response, extracted answers,
                             exact match per filter, truncation flag; others: per-choice
                             log-likelihoods, acc/acc_norm)
and results/task1/qualitative/<condition>.jsonl (24 greedy responses + automatic flags).

Resumable: finished benchmarks/parts are skipped, so rerunning after a
disconnect continues where it stopped. Results are pushed to GitHub after
every benchmark; a failed push only logs a warning.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading

import torch

from common.config import REPO_ROOT, add_config_args, load_config
from common.logging_utils import get_logger, run_metadata, write_json
from common.model_utils import load_causal_lm, load_tokenizer
from common.paths import get_paths
from common.publish import try_push
from common.seed import set_seed
from task1.conditions import CONDITIONS, materialize, remove_materialized
from task1.lmeval_runner import make_lm, run_benchmark
from task1.qualitative import generate, load_prompts

log = get_logger()
BENCHMARKS = ["gsm8k", "hellaswag", "boolq", "arc_easy"]


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_condition(cond: str, cfg, args) -> None:
    smoke = args.limit is not None
    eval_root = (get_paths().scratch / "eval_smoke") if smoke else (REPO_ROOT / "results" / "task1" / "eval")
    qual_root = (get_paths().scratch / "eval_smoke" / "qualitative") if smoke else (REPO_ROOT / "results" / "task1" / "qualitative")
    out_dir = eval_root / cond
    out_dir.mkdir(parents=True, exist_ok=True)
    push = not (args.no_push or smoke)

    benches = [b for b in args.benchmarks if args.force or not (out_dir / f"{b}.json").exists()] \
        if "bench" in args.parts else []
    qual_file = qual_root / f"{cond}.jsonl"
    do_qual = "qual" in args.parts and (args.force or not qual_file.exists())
    if not benches and not do_qual:
        log.info("%s: everything already evaluated; skipping", cond)
        return

    model_dir = materialize(cond, cfg)
    tok = load_tokenizer(str(model_dir))
    if benches:
        lm = make_lm(model_dir, batch_size=cfg.eval.batch_size, max_gen_toks=cfg.eval.gsm8k.max_gen_toks)
        for b in benches:
            log.info("=== %s / %s ===", cond, b)
            if b == "gsm8k":
                run_benchmark(lm, b, cfg.eval.gsm8k.task, generation=True, chat=True, cfg=cfg, out_dir=out_dir,
                              tokenizer=tok, limit=args.limit)
            else:
                run_benchmark(lm, b, cfg.eval.likelihood_tasks[b].task, generation=False, chat=False, cfg=cfg,
                              out_dir=out_dir, tokenizer=tok, limit=args.limit)
            write_json(out_dir / "environment.json", run_metadata())
            try_push([out_dir], f"task1/eval: {cond} {b}", enabled=push)
        del lm
        _free()

    if do_qual:
        log.info("=== %s / qualitative ===", cond)
        model = load_causal_lm(str(model_dir), dtype=torch.float16)
        prompts = load_prompts(cfg.qualitative.prompts_file)
        rows, info = generate(model, tok, prompts, cfg.qualitative.max_new_tokens)
        qual_root.mkdir(parents=True, exist_ok=True)
        with open(qual_file, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"condition": cond, **r}, ensure_ascii=False) + "\n")
        write_json(qual_root / f"{cond}_info.json", info)
        log.info("qualitative: %d responses, %d truncated, %.1fs", len(rows), sum(r["truncated"] for r in rows),
                 info["seconds"])
        del model
        _free()
        try_push([qual_root], f"task1/qualitative: {cond}", enabled=push)

    if not args.keep_models:
        remove_materialized(cond)


def run_parallel(args, argv: list[str]) -> None:
    n = torch.cuda.device_count()
    if n < 2:
        raise SystemExit("--parallel needs at least 2 GPUs")
    groups = [args.conditions[i::n] for i in range(n)]
    base = [a for a in argv if a != "--parallel"]
    if "--conditions" in base:  # drop the original condition list
        i = base.index("--conditions")
        j = i + 1
        while j < len(base) and not base[j].startswith("--"):
            j += 1
        base = base[:i] + base[j:]
    lock = threading.Lock()
    codes = {}

    def worker(gpu: int, conds: list[str]) -> None:
        if not conds:
            return
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
               "ATML_PUBLISH_TAG": f"eval_gpu{gpu}", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        cmd = [sys.executable, "-m", "task1.evaluate", *base, "--conditions", *conds]
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            with lock:
                print(f"[gpu{gpu}] {line}", end="", flush=True)
        codes[gpu] = proc.wait()

    threads = [threading.Thread(target=worker, args=(g, groups[g])) for g in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for g, c in sorted(codes.items()):
        print(f"gpu{g} ({', '.join(groups[g])}): {'ok' if c == 0 else f'FAILED (exit {c})'}")
    if any(codes.values()):
        sys.exit(1)


def main() -> None:
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/task1/eval.yaml")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--parts", nargs="+", default=["bench", "qual"], choices=["bench", "qual"])
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    parser.add_argument("--limit", type=int, default=None, help="examples per benchmark (smoke test; not pushed)")
    parser.add_argument("--parallel", action="store_true", help="one subprocess per visible GPU")
    parser.add_argument("--force", action="store_true", help="re-evaluate even if results exist")
    parser.add_argument("--keep-models", action="store_true", help="keep materialized FP16 models in scratch")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("Evaluation needs a GPU.")
    if args.parallel:
        run_parallel(args, argv)
        return
    cfg = load_config(args.config, args.set)
    set_seed(cfg.eval.seed)
    failed = []
    for cond in args.conditions:
        try:
            evaluate_condition(cond, cfg, args)
        except Exception:
            log.exception("condition %s failed; continuing with the next one", cond)
            failed.append(cond)
            _free()
    if failed:
        raise SystemExit(f"failed conditions: {failed} (rerun the same command to retry them)")


if __name__ == "__main__":
    main()
