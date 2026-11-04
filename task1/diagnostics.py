"""Task 1 weight-space and activation diagnostics (FP32; runs on CPU or GPU).

For down_proj in layers 0, 13, 27 and each trained condition
(full_ft_midpoint, and the five final models):
  rho_k     = ||U_k U_k^T dW||_F^2 / ||dW||_F^2   (output subspace of W0, k = 128)
  rho_k_in  = ||dW V_k V_k^T||_F^2 / ||dW||_F^2   (input subspace; extra, cheap)
  eta       = ||dW||_F / ||W0||_F
with dW = W_trained - W0 for full fine-tuning and (alpha/r) B~ A~ (projected
factors for OPLoRA) for adapters. One set of bases, from the FP32 starting
weights (an exact upcast of the released BF16 checkpoint), is used for every
method. OPLoRA's own training bases (from its FP16 backbone) are also used as
a numerical check: rho_k with them should be ~0. For a direction-agnostic
reference, a random update has E[rho_k] = k / d_out = 128 / 1024.

Activations: the starting model's down_proj inputs z at the last prompt token
are cached for the probe set (32 GSM8K + 32 HellaSwag doc IDs from
splits/task1/probe_set.json), using the exact prompts lm-eval used for the
baseline (read from the baseline per-example outputs, so run the baseline
evaluation first). For each condition: R(z) = ||dW z|| / (||W0 z|| + 1e-8).

Usage: python -m task1.diagnostics [--push]
"""
from __future__ import annotations

import argparse
import csv
import statistics

import torch

from common.config import REPO_ROOT, add_config_args, load_config
from common.logging_utils import get_logger, read_json, write_json
from common.model_utils import load_causal_lm, load_tokenizer
from common.publish import try_push
from task1.adapters import leading_singular_bases
from task1.conditions import TRAINED_CONDITIONS, fetch_stage
from task1.data import PROBE_FILE
from task1.lmeval_runner import read_jsonl_gz

log = get_logger()
OUT = REPO_ROOT / "results" / "task1" / "diagnostics"
EVAL = REPO_ROOT / "results" / "task1" / "eval" / "baseline"


def _key(layer: int) -> str:
    return f"model.layers.{layer}.mlp.down_proj"


def load_delta(condition: str, layer: int, W0: torch.Tensor) -> tuple[torch.Tensor, dict]:
    from safetensors import safe_open

    folder = fetch_stage(condition)
    meta = read_json(folder / "meta.json")
    extra = {}
    if meta["method"] == "full":
        with safe_open(str(folder / "weights.safetensors"), framework="pt") as f:
            W = f.get_tensor(_key(layer) + ".weight").float().to(W0.device)
        return W - W0, extra
    with safe_open(str(folder / "adapter.safetensors"), framework="pt") as f:
        A = f.get_tensor(_key(layer) + ".lora_A").float().to(W0.device)
        B = f.get_tensor(_key(layer) + ".lora_B").float().to(W0.device)
    if (folder / "bases.safetensors").exists():
        with safe_open(str(folder / "bases.safetensors"), framework="pt") as f:
            extra["own_U_k"] = f.get_tensor(_key(layer) + ".U_k").float().to(W0.device)
            extra["own_V_k"] = f.get_tensor(_key(layer) + ".V_k").float().to(W0.device)
    return float(meta["scaling"]) * (B @ A), extra


def probe_prompts() -> list[dict]:
    probes = read_json(PROBE_FILE)
    sets = [("math", "gsm8k", probes["gsm8k_test_doc_ids"]), ("nonmath", "hellaswag", probes["hellaswag_validation_doc_ids"])]
    out = []
    for label, bench, ids in sets:
        path = EVAL / f"{bench}_samples.jsonl.gz"
        if not path.exists():
            raise SystemExit(f"{path.relative_to(REPO_ROOT)} missing: run the baseline evaluation first")
        rows = {r["doc_id"]: r for r in read_jsonl_gz(path)}
        expected = {"gsm8k": 1319, "hellaswag": 10042}[bench]
        if len(rows) != expected:
            raise SystemExit(f"baseline {bench} has {len(rows)} docs, expected {expected}")
        out += [{"set": label, "benchmark": bench, "doc_id": i, "prompt": rows[i]["prompt"]} for i in ids]
    return out


@torch.no_grad()
def cache_inputs(model, tok, prompts: list[dict], layers: list[int]) -> dict[int, torch.Tensor]:
    """down_proj input at the last prompt token, one prompt at a time (no padding)."""
    store: dict[int, list] = {l: [] for l in layers}
    current: dict[int, torch.Tensor] = {}
    handles = []
    for l in layers:
        def hook(_m, inp, _out, l=l):
            current[l] = inp[0][0, -1].detach().float().clone()
        handles.append(model.model.layers[l].mlp.down_proj.register_forward_hook(hook))
    try:
        for p in prompts:
            ids = tok(p["prompt"], add_special_tokens=False, return_tensors="pt").to(model.device)
            model(**ids, use_cache=False)
            for l in layers:
                store[l].append(current[l])
    finally:
        for h in handles:
            h.remove()
    return {l: torch.stack(v) for l, v in store.items()}


def _fraction(num: torch.Tensor, den: torch.Tensor) -> float:
    return float(num / den) if den > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/task1/eval.yaml")
    parser.add_argument("--conditions", nargs="+", default=TRAINED_CONDITIONS, choices=TRAINED_CONDITIONS)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg.diagnostics
    layers, k, eps = list(d.layers), int(d.protect_rank), float(d.eps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("diagnostics on %s", device)

    model = load_causal_lm(cfg.model.name, dtype=torch.float32, revision=cfg.model.get("revision"), device=device)
    model.eval()
    tok = load_tokenizer(cfg.model.name, revision=cfg.model.get("revision"))
    W0 = {l: model.model.layers[l].mlp.down_proj.weight.detach().float().clone() for l in layers}
    bases = {}
    for l in layers:
        U, V, st = leading_singular_bases(W0[l].cpu(), k)  # CPU LAPACK, FP32
        bases[l] = (U.to(device), V.to(device), st)
    prompts = probe_prompts()
    Z = cache_inputs(model, tok, prompts, layers)
    base_out = {l: torch.linalg.vector_norm(Z[l] @ W0[l].T, dim=1) for l in layers}
    del model

    weight_rows, act_rows = [], []
    for cond in args.conditions:
        for l in layers:
            dW, extra = load_delta(cond, l, W0[l])
            U, V, _ = bases[l]
            fro2 = torch.sum(dW * dW)
            row = {"condition": cond, "layer": l, "d_out": dW.shape[0], "d_in": dW.shape[1], "k": k,
                   "rho_k": _fraction(torch.sum((U.T @ dW) ** 2), fro2),
                   "rho_k_in": _fraction(torch.sum((dW @ V) ** 2), fro2),
                   "eta": float(torch.sqrt(fro2) / torch.linalg.matrix_norm(W0[l])),
                   "dW_fro": float(torch.sqrt(fro2)), "W0_fro": float(torch.linalg.matrix_norm(W0[l])),
                   "random_direction_rho": k / dW.shape[0]}
            if "own_U_k" in extra:
                row["rho_k_own_bases"] = _fraction(torch.sum((extra["own_U_k"].T @ dW) ** 2), fro2)
                row["rho_k_in_own_bases"] = _fraction(torch.sum((dW @ extra["own_V_k"]) ** 2), fro2)
            weight_rows.append(row)
            R = torch.linalg.vector_norm(Z[l] @ dW.T, dim=1) / (base_out[l] + eps)
            for p, r in zip(prompts, R.tolist()):
                act_rows.append({"condition": cond, "layer": l, "set": p["set"], "benchmark": p["benchmark"],
                                 "doc_id": p["doc_id"], "R": r})
            log.info("%s layer %d: rho_k %.4f eta %.4g", cond, l, row["rho_k"], row["eta"])
            del dW, extra

    OUT.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in weight_rows:
        cols += [c for c in r if c not in cols]
    with open(OUT / "weight_space.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(weight_rows)
    with open(OUT / "activation_R.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(act_rows[0]))
        w.writeheader()
        w.writerows(act_rows)
    summary = []
    for cond in args.conditions:
        for l in layers:
            entry = {"condition": cond, "layer": l}
            for s in ("math", "nonmath"):
                vals = sorted(r["R"] for r in act_rows if r["condition"] == cond and r["layer"] == l and r["set"] == s)
                entry[f"R_{s}_median"] = statistics.median(vals)
                entry[f"R_{s}_mean"] = statistics.fmean(vals)
                entry[f"R_{s}_q25"] = vals[len(vals) // 4]
                entry[f"R_{s}_q75"] = vals[(3 * len(vals)) // 4]
            entry["R_median_ratio_math_over_nonmath"] = entry["R_math_median"] / entry["R_nonmath_median"]
            summary.append(entry)
    write_json(OUT / "summary.json", {
        "layers": layers, "k": k, "eps": eps, "device": device,
        "bases": {str(l): bases[l][2] for l in layers},
        "probe_counts": {"math": sum(p["set"] == "math" for p in prompts),
                         "nonmath": sum(p["set"] == "nonmath" for p in prompts)},
        "weight_space": weight_rows, "activation_summary": summary})
    print(f"\n{'condition':<18}{'layer':>6}{'rho_k':>9}{'eta':>11}{'R math':>10}{'R nonmath':>11}")
    for wr, s in zip(weight_rows, summary):
        print(f"{wr['condition']:<18}{wr['layer']:>6}{wr['rho_k']:>9.4f}{wr['eta']:>11.3e}"
              f"{s['R_math_median']:>10.3e}{s['R_nonmath_median']:>11.3e}")
    if args.push:
        try_push([OUT], "task1/diagnostics: weight-space and activation measurements")


if __name__ == "__main__":
    main()
