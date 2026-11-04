"""Run queues of Task 1 configs, one queue per GPU (e.g. Kaggle T4x2).

    python -m task1.launch --queue full_ft l2sp lora16 --queue oplora16 lora4

Each queue runs its configs one after another on its own GPU
(CUDA_VISIBLE_DEVICES), and queues run in parallel. Output is streamed with a
[gpuN:run] prefix and also written to <scratch>/launch_logs/. Every run
records which runs shared the machine (ATML_LAUNCH_INFO), because parallel
jobs share CPU, host memory and PCIe, which affects measured training time.
A failed run is reported and its queue moves on; rerunning the same command
resumes unfinished runs and skips finished work quickly (they resume at the
final update).

Extra arguments after `--` are passed to every task1.train call, e.g.
    python -m task1.launch --queue lora4 -- --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

from common.config import REPO_ROOT
from common.paths import get_paths


def run_queue(gpu: int, queue: list[str], all_queues: list[list[str]], extra: list[str], results: dict,
              lock: threading.Lock) -> None:
    log_dir = get_paths().scratch / "launch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in queue:
        cfg = REPO_ROOT / "configs" / "task1" / f"{name}.yaml"
        tag = f"gpu{gpu}:{name}"
        if not cfg.exists():
            with lock:
                print(f"[{tag}] config {cfg} not found; skipping", flush=True)
                results[name] = "missing config"
            continue
        concurrent = [q for i, q in enumerate(all_queues) if i != gpu]
        env = {**os.environ,
               "CUDA_VISIBLE_DEVICES": str(gpu),
               "PYTHONUNBUFFERED": "1",
               "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "TOKENIZERS_PARALLELISM": "false",
               "ATML_PUBLISH_TAG": f"{name}_gpu{gpu}",
               "ATML_LAUNCH_INFO": json.dumps({"gpu": gpu, "queue": queue, "other_queues": concurrent,
                                               "n_parallel_queues": len(all_queues)})}
        cmd = [sys.executable, "-m", "task1.train", "--config", str(cfg.relative_to(REPO_ROOT)), *extra]
        start = time.time()
        with open(log_dir / f"{name}.log", "a", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                fh.write(line)
                with lock:
                    print(f"[{tag}] {line}", end="", flush=True)
            code = proc.wait()
        hours = (time.time() - start) / 3600
        with lock:
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"[{tag}] finished: {status} after {hours:.2f} h", flush=True)
            results[name] = status


def main() -> None:
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", nargs="+", action="append", required=True,
                        help="config names (configs/task1/<name>.yaml) for one GPU; repeat per GPU")
    args = parser.parse_args(argv)

    import torch
    n_gpu = torch.cuda.device_count()
    if len(args.queue) > n_gpu:
        raise SystemExit(f"{len(args.queue)} queues but only {n_gpu} GPU(s) visible")
    results: dict = {}
    lock = threading.Lock()
    threads = [threading.Thread(target=run_queue, args=(g, q, args.queue, extra, results, lock))
               for g, q in enumerate(args.queue)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("\n=== launch summary ===")
    for name, status in results.items():
        print(f"{name:<14} {status}")
    if any(s != "ok" for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
