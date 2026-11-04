"""Platform detection and directory layout.

Small, committed artifacts (logs, metrics, per-example outputs) live in
``<repo>/results``. Large or temporary artifacts (datasets, caches, checkpoints,
exported weights awaiting upload) live in a per-platform scratch directory that
is never committed:

* Kaggle: ``/tmp/atml`` (keeps /kaggle/working, which is saved as notebook
  output and capped in size, free of multi-GB files)
* Colab:  ``/content/atml_scratch``
* local:  ``<repo>/_scratch``

Set ``ATML_SCRATCH`` to override.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from .config import REPO_ROOT


def detect_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if Path("/content").exists() and importlib.util.find_spec("google.colab") is not None:
        return "colab"
    return "local"


_DEFAULT_SCRATCH = {
    "kaggle": Path("/tmp/atml"),
    "colab": Path("/content/atml_scratch"),
    "local": REPO_ROOT / "_scratch",
}


@dataclass(frozen=True)
class Paths:
    platform: str
    repo: Path
    results: Path
    scratch: Path

    @property
    def data(self) -> Path:
        return self.scratch / "data"

    @property
    def ckpt(self) -> Path:
        return self.scratch / "ckpt"

    @property
    def artifacts(self) -> Path:
        return self.scratch / "artifacts"

    @property
    def downloads(self) -> Path:
        return self.scratch / "downloads"

    def ensure(self) -> "Paths":
        for p in (self.results, self.scratch, self.data, self.ckpt, self.artifacts, self.downloads):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    platform = detect_platform()
    scratch = Path(os.environ.get("ATML_SCRATCH", _DEFAULT_SCRATCH[platform]))
    return Paths(platform=platform, repo=REPO_ROOT, results=REPO_ROOT / "results", scratch=scratch).ensure()
