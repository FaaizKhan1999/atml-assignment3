"""Does vLLM run Qwen3-0.6B in FP16 on this GPU, how fast, and does greedy decoding match HF?

Run in a fresh session after `pip install vllm` (it may replace the preinstalled
torch, so don't install it in a session you train in):
    pip install -q vllm && python -m scripts.check_vllm

Writes results/env/vllm_<platform>_<gpu>.json. Uses hand-written questions only
(no benchmark data).
"""
from __future__ import annotations

import argparse
import gc
import time

from common.chat_format import generation_prompt
from common.config import REPO_ROOT
from common.logging_utils import get_logger, gpu_slug, package_versions, write_json
from common.paths import detect_platform

log = get_logger()

QUESTIONS = [
    "A shop sells pencils at 3 for $1. How much do 27 pencils cost?",
    "Tom has twice as many marbles as Ann. Together they have 36. How many does Ann have?",
    "Write one sentence explaining why the sky looks blue.",
    "List three prime numbers greater than 50.",
    "A train travels 180 km in 2.5 hours. What is its average speed in km/h?",
    "Give a short JSON object with keys 'city' and 'country' for the capital of Japan.",
    "If a rectangle has perimeter 30 and width 5, what is its area?",
    "Explain in two sentences what photosynthesis does.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    out = {"versions": package_versions(), "model": args.model}
    path = REPO_ROOT / "results" / "env" / f"vllm_{detect_platform()}_{gpu_slug()}.json"

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        out["vllm_import"] = f"failed: {exc}"
        write_json(path, out)
        raise SystemExit(f"vLLM not importable: {exc}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [generation_prompt(tok, q) for q in QUESTIONS]

    try:
        t0 = time.perf_counter()
        llm = LLM(model=args.model, dtype="float16", seed=6304, gpu_memory_utilization=0.80, max_model_len=4096)
        out["vllm_load_seconds"] = time.perf_counter() - t0
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        t0 = time.perf_counter()
        gens = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        v_ids = [list(g.outputs[0].token_ids) for g in gens]
        out["vllm_ok"] = True
        out["vllm_generated_tokens"] = sum(len(x) for x in v_ids)
        out["vllm_tokens_per_s_batched"] = out["vllm_generated_tokens"] / dt
        del llm
    except Exception as exc:
        out["vllm_ok"] = False
        out["vllm_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        write_json(path, out)
        raise SystemExit(f"vLLM failed on this GPU: {out['vllm_error']}")

    gc.collect()
    import torch
    torch.cuda.empty_cache()
    from common.model_utils import load_causal_lm
    model = load_causal_lm(args.model, dtype=torch.float16).eval()
    agree, first_div = [], []
    t0 = time.perf_counter()
    hf_tokens = 0
    for p, vid in zip(prompts, v_ids):
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens)
        hid = g[0, enc["input_ids"].shape[1]:].tolist()
        hf_tokens += len(hid)
        n = min(len(hid), len(vid))
        div = next((i for i in range(n) if hid[i] != vid[i]), n)
        first_div.append(div)
        agree.append(hid[:n] == vid[:n] and len(hid) == len(vid))
    out["hf_tokens_per_s_unbatched"] = hf_tokens / (time.perf_counter() - t0)
    out["greedy_exact_agreement"] = sum(agree) / len(agree)
    out["first_divergence_token"] = first_div
    write_json(path, out)
    log.info("vLLM OK: %.0f tok/s batched vs HF %.0f tok/s unbatched; exact greedy agreement %.2f; "
             "first divergence positions %s", out["vllm_tokens_per_s_batched"], out["hf_tokens_per_s_unbatched"],
             out["greedy_exact_agreement"], first_div)
    log.info("wrote %s (commit it with: python -c \"from common.sync import push_results; "
             "push_results(['results/env'], 'env: vllm check')\")", path)


if __name__ == "__main__":
    main()
