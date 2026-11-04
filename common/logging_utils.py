"""Console logging, append-only JSONL metric logs, and run metadata."""
from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from .config import REPO_ROOT


def get_logger(name: str = "atml") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _jsonable(value: Any) -> Any:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.tolist()
        if isinstance(value, torch.dtype):
            return str(value)
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_jsonable)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class JsonlLogger:
    """Append-only JSONL log; safe to reopen after a resume."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        rec = {"time": time.time(), **record}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=_jsonable) + "\n")
            fh.flush()

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def truncate_after(self, key: str, value: float) -> None:
        """Drop records with record[key] > value (used when resuming from a checkpoint)."""
        kept = [r for r in self.read() if key not in r or r[key] <= value]
        with open(self.path, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, default=_jsonable) + "\n")


def git_commit() -> dict:
    def run(args: list[str]) -> str:
        res = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else ""

    return {"commit": run(["rev-parse", "HEAD"]), "dirty": bool(run(["status", "--porcelain", "--untracked-files=no"]))}


def package_versions(names: Iterable[str] = ("torch", "transformers", "accelerate", "datasets", "huggingface_hub",
                                             "safetensors", "numpy", "lm_eval", "vllm", "math-verify")) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_metadata() -> dict:
    from .paths import detect_platform

    meta: dict[str, Any] = {
        "platform": detect_platform(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "packages": package_versions(),
        "git": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import torch
        meta["cuda_available"] = torch.cuda.is_available()
        meta["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            meta["gpus"] = [
                {"name": torch.cuda.get_device_name(i),
                 "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
                 "capability": ".".join(map(str, torch.cuda.get_device_capability(i)))}
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        meta["cuda_available"] = False
    return meta


def gpu_slug() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            return "".join(c if c.isalnum() else "_" for c in name).strip("_")
    except ImportError:
        pass
    return "cpu"
