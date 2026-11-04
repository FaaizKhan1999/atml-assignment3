"""Timing, GPU memory, and disk-size helpers."""
from __future__ import annotations

import time
from pathlib import Path


def cuda_sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"peak_allocated_gb": None, "peak_reserved_gb": None}
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


class Timer:
    """Accumulating wall-clock timer; synchronizes CUDA on entry/exit when sync=True."""

    def __init__(self, sync: bool = True):
        self.sync = sync
        self.total = 0.0
        self._start = None

    def __enter__(self) -> "Timer":
        if self.sync:
            cuda_sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.sync:
            cuda_sync()
        self.total += time.perf_counter() - self._start
        self._start = None


def dir_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)
