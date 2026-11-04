"""The seven Task 1 evaluation conditions and how each model is built.

* baseline          - the unchanged Qwen3-0.6B, loaded in FP16
* full methods      - FP32 trained weights from the HF weights repo, cast to FP16
* adapter methods   - a fresh FP16 backbone with the exported effective factors
                      merged in (scaling * B @ A computed in FP32, then cast);
                      for OPLoRA these are the projected factors

``materialize`` saves the FP16 model (weights + config only) to scratch so
lm-eval's HF backend can load it like any checkpoint. The tokenizer is always
loaded from the original Hub repo, never from a re-saved copy, so tokenization
is identical to training.
"""
from __future__ import annotations

import gc
import shutil
from pathlib import Path

from common.checkpoint import load_tensors, load_trainable_weights
from common.logging_utils import get_logger, read_json, write_json
from common.model_utils import load_causal_lm
from common.paths import get_paths
from common.sync import hf_download_dir, hf_repo

log = get_logger()

CONDITIONS: dict[str, tuple[str, str] | None] = {
    "baseline": None,
    "full_ft_midpoint": ("full_ft", "midpoint"),
    "full_ft_final": ("full_ft", "final"),
    "l2sp_final": ("l2sp", "final"),
    "lora4_final": ("lora4", "final"),
    "lora16_final": ("lora16", "final"),
    "oplora16_final": ("oplora16", "final"),
}
TRAINED_CONDITIONS = [c for c, v in CONDITIONS.items() if v is not None]
N_ADAPTER_MODULES = 28 * 7


def fetch_stage(condition: str) -> Path:
    """Local folder with meta.json and weights for a trained condition (downloaded once)."""
    run, stage = CONDITIONS[condition]
    root = get_paths().downloads
    local = root / "task1" / run / stage
    if (local / "meta.json").exists():
        return local
    log.info("downloading task1/%s/%s from Hugging Face", run, stage)
    return hf_download_dir(hf_repo("weights"), f"task1/{run}/{stage}", root)


def build_fp16_model(condition: str, model_name: str, revision: str | None = None, device: str = "cuda"):
    """Returns (FP16 model in eval mode, info dict)."""
    import torch

    from task1.adapters import merge_exported_into

    if CONDITIONS[condition] is None:
        model = load_causal_lm(model_name, dtype=torch.float16, revision=revision, device=device)
        info = {"condition": condition, "source": "unchanged base model"}
    else:
        folder = fetch_stage(condition)
        meta = read_json(folder / "meta.json")
        if meta["method"] == "full":
            model = load_causal_lm(model_name, dtype=torch.float32, revision=revision, device=device)
            load_trainable_weights(model, folder / "weights.safetensors", strict=True)
            model = model.half()
            source = "full FP32 trained weights cast to FP16"
        else:
            model = load_causal_lm(model_name, dtype=torch.float16, revision=revision, device=device)
            factors = load_tensors(folder / "adapter.safetensors")
            n = merge_exported_into(model, factors, float(meta["scaling"]))
            if n != N_ADAPTER_MODULES:
                raise RuntimeError(f"merged {n} modules, expected {N_ADAPTER_MODULES}")
            source = f"adapter factors merged into a fresh FP16 backbone ({n} modules, scaling {meta['scaling']})"
        commit = getattr(model.config, "_commit_hash", None)
        if meta.get("model_commit") and commit and meta["model_commit"] != commit:
            log.warning("base model commit %s differs from training commit %s", commit, meta["model_commit"])
        info = {"condition": condition, "source": source, "stage_meta": meta}
    model.eval()
    return model, info


def materialize(condition: str, cfg) -> Path:
    """Save the condition's FP16 model to scratch (reused if already there)."""
    import torch

    out = get_paths().artifacts / "task1_eval_models" / condition
    if (out / "MATERIALIZED").exists():
        return out
    shutil.rmtree(out, ignore_errors=True)
    model, info = build_fp16_model(condition, cfg.model.name, cfg.model.get("revision"))
    model.save_pretrained(out, safe_serialization=True)
    write_json(out / "materialize_info.json", info)
    (out / "MATERIALIZED").write_text("ok")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log.info("materialized %s -> %s", condition, out)
    return out


def remove_materialized(condition: str) -> None:
    shutil.rmtree(get_paths().artifacts / "task1_eval_models" / condition, ignore_errors=True)
