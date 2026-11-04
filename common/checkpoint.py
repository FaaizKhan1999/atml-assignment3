"""Resumable training checkpoints.

A checkpoint is a folder ``update_XXXXX`` containing
* ``trainable.safetensors`` - every trainable parameter in its training dtype
  (FP32 master weights for full fine-tuning, FP32 adapters otherwise); tied
  parameters are stored once because named_parameters() de-duplicates them
* ``trainer_state.pt``      - optimizer, grad scaler, RNG, data position, extras
* ``COMPLETE``              - written last; folders without it are ignored

Folders are written to ``*.tmp`` and renamed, so a crash mid-save never leaves
a checkpoint that looks complete.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from .profiling import dir_size_bytes
from .seed import capture_rng_state, restore_rng_state

WEIGHTS_FILE = "trainable.safetensors"
STATE_FILE = "trainer_state.pt"
DONE_FILE = "COMPLETE"
_PATTERN = re.compile(r"^update_(\d+)$")


def trainable_state(model) -> dict:
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def save_tensors(tensors: dict, path: str | Path, metadata: dict | None = None) -> None:
    from safetensors.torch import save_file
    save_file({k: v.detach().contiguous().cpu() for k, v in tensors.items()}, str(path),
              metadata={k: str(v) for k, v in (metadata or {}).items()})


def load_tensors(path: str | Path, device: str = "cpu") -> dict:
    from safetensors.torch import load_file
    return load_file(str(path), device=device)


def load_trainable_weights(model, path: str | Path, strict: bool = True) -> None:
    import torch

    saved = load_tensors(path)
    params = dict(model.named_parameters())
    trainable = {n for n, p in params.items() if p.requires_grad}
    unexpected = set(saved) - set(params)
    missing = trainable - set(saved)
    if strict and (unexpected or missing):
        raise KeyError(f"checkpoint mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
    with torch.no_grad():
        for name, tensor in saved.items():
            if name in params:
                params[name].copy_(tensor.to(device=params[name].device, dtype=params[name].dtype))


def save_checkpoint(ckpt_dir: str | Path, update: int, model, optimizer, scaler, data_state: dict,
                    extra: dict | None = None) -> tuple[Path, dict]:
    import torch

    ckpt_dir = Path(ckpt_dir)
    final = ckpt_dir / f"update_{update:05d}"
    tmp = ckpt_dir / f"update_{update:05d}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_tensors(trainable_state(model), tmp / WEIGHTS_FILE)
    torch.save({
        "update": update,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng_state(),
        "data": data_state,
        "extra": extra or {},
    }, tmp / STATE_FILE)
    (tmp / DONE_FILE).write_text("ok")
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    sizes = {
        "weights_bytes": dir_size_bytes(final / WEIGHTS_FILE),
        "state_bytes": dir_size_bytes(final / STATE_FILE),
        "total_bytes": dir_size_bytes(final),
    }
    return final, sizes


def find_latest(ckpt_dir: str | Path) -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    found = []
    for p in ckpt_dir.iterdir():
        m = _PATTERN.match(p.name)
        if m and p.is_dir() and (p / DONE_FILE).exists():
            found.append((int(m.group(1)), p))
    return max(found)[1] if found else None


def load_checkpoint(path: str | Path, model, optimizer, scaler) -> dict:
    import torch

    path = Path(path)
    load_trainable_weights(model, path / WEIGHTS_FILE)
    state = torch.load(path / STATE_FILE, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng_state(state["rng"])
    return state


def prune_checkpoints(ckpt_dir: str | Path, keep_last: int = 1) -> None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return
    found = sorted((int(m.group(1)), p) for p in ckpt_dir.iterdir()
                   if (m := _PATTERN.match(p.name)) and p.is_dir())
    for _, p in found[:-keep_last] if keep_last > 0 else found:
        shutil.rmtree(p, ignore_errors=True)
