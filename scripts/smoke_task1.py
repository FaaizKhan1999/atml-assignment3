"""Phase 0 smoke test for Task 1 on the current GPU.

Runs a few real optimizer updates of each fine-tuning method on the first
MetaMathQA examples and reports:
  * chat-template checks (non-thinking prompt, train/inference token match)
  * MetaMathQA length statistics and the implied token budget for 40k examples
  * FP16 safety: weight cast overflow/underflow, per-layer activation maxima
  * per method: tokens/s -> estimated hours for the full run, peak GPU memory
    (including a worst-case 2x2048-token window), trainable parameters,
    skipped updates, checkpoint save/load sizes and time
  * adapters: zero-init equivalence to the base model, merged-vs-unmerged
    logits, and (OPLoRA) numerical projection constraints

Usage (from the repo root):
    python -m scripts.smoke_task1                      # all methods
    python -m scripts.smoke_task1 --methods lora16 oplora16
    python -m scripts.smoke_task1 --push               # commit results JSON to GitHub
"""
from __future__ import annotations

import argparse
import gc
import shutil
import statistics
import time
from functools import partial

import torch

from common.chat_format import build_sft_example, check_template
from common.checkpoint import load_checkpoint, save_checkpoint
from common.config import REPO_ROOT, add_config_args, load_config
from common.data_utils import SequentialWindows, collate_causal, load_metamath_rows, to_device
from common.logging_utils import get_logger, gpu_slug, run_metadata, write_json
from common.losses import TokenNormalizedObjective
from common.model_utils import (activation_range_check, build_param_groups, count_parameters,
                                enable_gradient_checkpointing, fp16_cast_report, load_causal_lm, load_tokenizer)
from common.paths import get_paths
from common.profiling import peak_memory, reset_peak_memory
from common.seed import set_seed
from common.train_loop import L2SPRegularizer, Trainer
from task1.adapters import export_adapters, inject_adapters, merge_exported_into, projection_residuals

log = get_logger()


# ----------------------------------------------------------------------------- helpers

def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _logits(model, batch: dict) -> torch.Tensor:
    was = model.training
    model.eval()
    b = to_device(batch)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits
    model.train(was)
    return out[b["attention_mask"].bool()].float()


def _compare_logits(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a - b).abs()
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "max_abs_logit": float(a.abs().max()),
            "argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean())}


def _length_stats(examples: list[dict], microbatch: int, budget: int) -> dict:
    lens = [len(e["input_ids"]) for e in examples]
    s = sorted(lens)
    pairs = [lens[i:i + microbatch] for i in range(0, len(lens) - microbatch + 1, microbatch)]
    padded = sum(max(p) * len(p) for p in pairs)
    real = sum(sum(p) for p in pairs)
    mean = statistics.fmean(lens)
    return {
        "n_sampled": len(lens), "mean": mean, "p50": s[len(s) // 2], "p95": s[int(0.95 * (len(s) - 1))],
        "p99": s[int(0.99 * (len(s) - 1))], "max": s[-1], "n_over_2048": sum(l > 2048 for l in lens),
        "mean_response_tokens": statistics.fmean(e["n_response"] for e in examples),
        "padding_overhead_fraction": (padded - real) / max(padded, 1),
        "est_total_input_tokens": mean * budget,
    }


def _stress_examples(template: dict, n: int, seq_len: int) -> list[dict]:
    """Synthetic worst-case examples: real tokens tiled to seq_len, last 75% supervised."""
    ids = (template["input_ids"] * (seq_len // len(template["input_ids"]) + 1))[:seq_len]
    cut = seq_len // 4
    labels = [-100] * cut + ids[cut:]
    return [{"input_ids": ids, "labels": labels, "n_prompt": cut, "n_response": seq_len - cut} for _ in range(n)]


# ----------------------------------------------------------------------------- one method

def run_method(name: str, spec, cfg, tok, train_examples, stress_examples, paths,
               est_total_tokens: float) -> dict:
    sm, tr = cfg.smoke, cfg.training
    _free()
    reset_peak_memory()
    is_full = spec.kind == "full"
    res: dict = {"kind": spec.kind, "lr": spec.lr}

    t0 = time.perf_counter()
    model = load_causal_lm(cfg.model.name, revision=cfg.model.get("revision"),
                           dtype=torch.float32 if is_full else torch.float16)
    enable_gradient_checkpointing(model)
    res["load_seconds"] = time.perf_counter() - t0

    probe = collate_causal(train_examples[:2], tok.pad_token_id)
    modules = None
    if not is_full:
        base_logits = _logits(model, probe)
        modules, info = inject_adapters(model, targets=tr.lora_targets, r=spec.r, alpha=spec.alpha,
                                        dropout=tr.lora_dropout, seed=cfg.project.seed,
                                        protect_rank=spec.get("protect_rank"))
        res["adapter_info"] = info
        res["zero_init_check"] = _compare_logits(base_logits, _logits(model, probe))
        del base_logits

    res["params"] = count_parameters(model)
    groups = build_param_groups([(n, p) for n, p in model.named_parameters() if p.requires_grad],
                                lr=spec.lr, weight_decay=tr.weight_decay)
    l2sp = None
    if spec.get("l2sp_lambda"):
        t0 = time.perf_counter()
        l2sp = L2SPRegularizer(model, lam=spec.l2sp_lambda, ref_device=spec.get("ref_device", "cuda"))
        res["l2sp"] = {"ref_device": l2sp.ref_device, "ref_bytes": l2sp.ref_bytes, "n_params": l2sp.n_params,
                       "setup_seconds": time.perf_counter() - t0}

    data = SequentialWindows(train_examples + stress_examples, microbatch=sm.microbatch, accum=sm.accum,
                             collate_fn=partial(collate_causal, pad_id=tok.pad_token_id))
    run_dir = paths.scratch / "smoke" / name
    shutil.rmtree(run_dir, ignore_errors=True)
    trainer = Trainer(model=model, param_groups=groups, objective=TokenNormalizedObjective(), data=data,
                      total_updates=data.num_windows, warmup_updates=0, min_lr_ratio=1.0,
                      grad_clip=tr.grad_clip, betas=tr.betas, eps=tr.eps, fused_adam=tr.fused_adam, l2sp=l2sp,
                      ckpt_dir=run_dir / "ckpt", log_path=run_dir / "log.jsonl", checkpoint_final=False)
    summary = trainer.train()
    recs = [r for r in trainer.logger.read() if r.get("event") == "update"]
    timed = recs[sm.skip_first:sm.timed_updates]
    tok_per_s = sum(r["input_tokens"] for r in timed) / sum(r["update_seconds"] for r in timed)
    res["updates"] = [{k: r.get(k) for k in ("update", "loss", "grad_norm", "loss_scale", "skipped",
                                             "input_tokens", "update_seconds", "omega", "l2sp_seconds")}
                      for r in recs]
    res["throughput_tokens_per_s"] = tok_per_s
    res["est_full_run_hours"] = est_total_tokens / tok_per_s / 3600
    if l2sp is not None:
        l2s = [r["l2sp_seconds"] for r in timed]
        res["l2sp"]["seconds_per_update"] = statistics.fmean(l2s)
        res["l2sp"]["displacement_after_smoke"] = l2sp.displacement()
    res["trainer_summary"] = summary
    res["peak_memory_incl_stress"] = peak_memory()

    if sm.test_checkpoint:
        t0 = time.perf_counter()
        path, sizes = save_checkpoint(run_dir / "ckpt", trainer.update, model, trainer.optimizer, trainer.scaler,
                                      data.state_dict(), {"smoke": True})
        save_s = time.perf_counter() - t0
        probe_param = min((p for p in model.parameters() if p.requires_grad), key=lambda p: p.numel())
        before = probe_param.detach().clone()
        with torch.no_grad():
            probe_param.add_(1.0)  # perturb, so an exact match proves the load restored it
        t0 = time.perf_counter()
        load_checkpoint(path, model, trainer.optimizer, trainer.scaler)
        after = probe_param
        res["checkpoint"] = {**sizes, "save_seconds": save_s, "load_seconds": time.perf_counter() - t0,
                             "roundtrip_exact": bool(torch.equal(before, after))}
        shutil.rmtree(path, ignore_errors=True)

    if modules is not None:
        res["projection_check"] = projection_residuals(modules)
        factors, meta = export_adapters(modules)
        res["export_bytes"] = sum(t.numel() * t.element_size() for t in factors.values())
        adapter_logits = _logits(model, probe)
        del trainer, model, modules, groups, data
        _free()
        fresh = load_causal_lm(cfg.model.name, revision=cfg.model.get("revision"), dtype=torch.float16)
        merge_exported_into(fresh, factors, meta["scaling"])
        res["merge_check"] = _compare_logits(adapter_logits, _logits(fresh, probe))
        del fresh
    _free()
    return res


# ----------------------------------------------------------------------------- main

def print_summary(results: dict) -> None:
    print("\n" + "=" * 96)
    L = results["lengths"]
    print(f"MetaMathQA sample: mean {L['mean']:.0f} tok, p95 {L['p95']}, max {L['max']}, "
          f"padding overhead {100 * L['padding_overhead_fraction']:.1f}%, "
          f"est. {L['est_total_input_tokens'] / 1e6:.1f}M input tokens for the full budget")
    T = results["template"]
    print(f"Template: empty-think prompt={T['prompt_ends_with_empty_think']}, "
          f"render match={T['full_render_match_rate']:.2f}, token match={T['split_vs_joint_tokenization_match_rate']:.2f}")
    print(f"FP16: weights inf={results['fp16']['weights']['n_overflow_to_inf']}, "
          f"max|hidden| fp32-weights={results['fp16']['activations_fp32_weights']['max_abs_hidden']:.0f}, "
          f"fp16-weights={results['fp16']['activations_fp16_weights']['max_abs_hidden']:.0f} (limit 65504)")
    print("-" * 96)
    print(f"{'method':<12}{'trainable':>14}{'tok/s':>9}{'est. h':>8}{'peak GB':>9}{'skipped':>9}{'ckpt GB':>9}  checks")
    for name, r in results["methods"].items():
        if r.get("oom") or r.get("error"):
            print(f"{name:<12}  FAILED: {r.get('error', 'OOM')[:70]}")
            continue
        checks = []
        if "zero_init_check" in r:
            checks.append(f"init dlogit {r['zero_init_check']['max_abs_diff']:.1e}")
        if "merge_check" in r:
            checks.append(f"merge dlogit {r['merge_check']['max_abs_diff']:.2e}")
        if r.get("projection_check"):
            checks.append(f"proj {r['projection_check']['max_rel_UkT_B']:.1e}/{r['projection_check']['max_rel_A_Vk']:.1e}")
        ck = r.get("checkpoint", {}).get("total_bytes")
        print(f"{name:<12}{r['params']['trainable']:>14,}{r['throughput_tokens_per_s']:>9.0f}"
              f"{r['est_full_run_hours']:>8.2f}{r['peak_memory_incl_stress']['peak_allocated_gb']:>9.2f}"
              f"{r['trainer_summary']['skipped_updates']:>9}{(ck or 0) / 2**30:>9.2f}  " + ", ".join(checks))
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/smoke/task1_smoke.yaml")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--push", action="store_true", help="commit the results JSON to GitHub")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("The smoke test needs a GPU.")
    cfg = load_config(args.config, args.set)
    sm = cfg.smoke
    paths = get_paths()
    set_seed(cfg.project.seed)
    methods = args.methods or list(sm.methods.keys())

    tok = load_tokenizer(cfg.model.name, revision=cfg.model.get("revision"))
    log.info("loading %d MetaMathQA rows", sm.length_sample)
    rows = load_metamath_rows(sm.length_sample)
    results: dict = {"meta": run_metadata(), "config": cfg.to_dict()}
    results["template"] = check_template(tok, [(r["query"], r["response"]) for r in rows[:200]])
    examples = [build_sft_example(tok, r["query"], r["response"]) for r in rows]
    results["lengths"] = _length_stats(examples, sm.microbatch, sm.full_budget_examples)

    n_train = sm.timed_updates * sm.microbatch * sm.accum
    train_examples = examples[:n_train]
    stress = _stress_examples(examples[0], sm.microbatch * sm.accum, sm.stress_seq_len)

    log.info("FP16 safety checks")
    model = load_causal_lm(cfg.model.name, revision=cfg.model.get("revision"), dtype=torch.float32)
    probe = to_device(collate_causal(examples[:4], tok.pad_token_id))
    fp16 = {"weights": fp16_cast_report(model), "activations_fp32_weights": activation_range_check(model, probe)}
    model.half()
    fp16["activations_fp16_weights"] = activation_range_check(model, probe)
    results["fp16"] = fp16
    del model, probe
    _free()

    results["methods"] = {}
    for name in methods:
        log.info("=== %s ===", name)
        try:
            results["methods"][name] = run_method(name, sm.methods[name], cfg, tok, train_examples, stress, paths,
                                                  results["lengths"]["est_total_input_tokens"])
        except torch.cuda.OutOfMemoryError as exc:
            results["methods"][name] = {"oom": True, "error": f"CUDA OOM: {str(exc)[:300]}"}
        except Exception as exc:  # keep going so one failure doesn't hide the others
            log.exception("method %s failed", name)
            results["methods"][name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        _free()

    out = REPO_ROOT / "results" / "smoke" / f"task1_{paths.platform}_{gpu_slug()}.json"
    write_json(out, results)
    print_summary(results)
    log.info("wrote %s", out)
    if args.push:
        from common.sync import push_results
        push_results([out], f"smoke: task1 on {paths.platform} ({gpu_slug()})")


if __name__ == "__main__":
    main()
