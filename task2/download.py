"""Downloads for Task 2: COCO 2014 annotations, VQAv2 questions/annotations/complementary pairs, COCO images.

Every download is retried, written to a temporary file and renamed, so an
interrupted session never leaves a truncated file that looks complete. If a
source is unreachable, the error names the file to obtain manually; put it in
a folder and pass ``--local-dir`` (JSON files or the original zips both work).
"""
from __future__ import annotations

import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common.logging_utils import get_logger

log = get_logger()
UA = {"User-Agent": "Mozilla/5.0 (atml-pa3 data preparation)"}


def fetch(url: str, dest: Path, retries: int = 4, timeout: int = 120) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 20)
            tmp.rename(dest)
            return dest
        except Exception as exc:  # network errors, HTTP errors
            last = exc
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"could not download {url}: {last}. Download it manually and pass --local-dir.")


def extract_json(zip_path: Path, out_dir: Path) -> list[Path]:
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                target = out_dir / Path(name).name
                if not target.exists():
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
                out.append(target)
    return out


def find_local(local_dir: Path | None, pattern: str) -> Path | None:
    """Look for an already-downloaded JSON (or a zip containing it) in local_dir."""
    if not local_dir:
        return None
    for p in sorted(Path(local_dir).rglob(pattern)):
        return p
    for z in sorted(Path(local_dir).rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            if any(Path(n).match(pattern) for n in zf.namelist()):
                return z
    return None


def get_json(pattern: str, url: str, raw_dir: Path, local_dir: Path | None) -> Path:
    """Path to the JSON matching ``pattern``: from local_dir if present, else downloaded from ``url``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    hits = sorted(raw_dir.glob(pattern))
    if hits:
        return hits[0]
    src = find_local(local_dir, pattern)
    if src is not None and src.suffix == ".json":
        return src
    if src is None:
        log.info("downloading %s", url)
        src = fetch(url, raw_dir / Path(url).name)
    extract_json(src, raw_dir)
    hits = sorted(raw_dir.glob(pattern))
    if not hits:
        raise RuntimeError(f"{src} does not contain a file matching {pattern}")
    return hits[0]


def image_path(root: Path, split: str, image_id: int) -> Path:
    return root / split / f"COCO_{split}_{image_id:012d}.jpg"


def download_images(items: list[tuple[str, int]], root: Path, url_template: str, workers: int = 32) -> dict:
    """Download COCO images (split, image_id) that are not already present. Returns timing stats."""
    todo = [(s, i) for s, i in items if not image_path(root, s, i).exists()]
    t0 = time.perf_counter()
    failed = []
    if todo:
        log.info("downloading %d images (%d already present)", len(todo), len(items) - len(todo))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch, url_template.format(split=s, image_id=i), image_path(root, s, i)): (s, i)
                    for s, i in todo}
            for n, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except Exception as exc:
                    failed.append((*futs[f], str(exc)[:200]))
                if n % 1000 == 0:
                    log.info("  %d / %d images", n, len(todo))
    if failed:
        raise RuntimeError(f"{len(failed)} images failed, e.g. {failed[:3]}")
    return {"n_images": len(items), "n_downloaded": len(todo), "seconds": time.perf_counter() - t0}
