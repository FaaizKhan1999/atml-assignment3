"""Verify tokens and both sync paths, and record this platform's environment.

1. Writes results/env/<platform>.json (versions, GPU, git commit).
2. Uploads a tiny file to both Hugging Face repos, lists it, deletes it.
3. Commits results/env/ (including the pip freeze from setup.sh) and pushes to GitHub.

Usage: python -m scripts.check_sync
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from common.config import REPO_ROOT
from common.logging_utils import get_logger, run_metadata, write_json
from common.paths import detect_platform
from common.sync import hf_delete_dir, hf_list, hf_repo, hf_upload_dir, push_results

log = get_logger()


def main() -> None:
    platform = detect_platform()
    env_dir = REPO_ROOT / "results" / "env"
    write_json(env_dir / f"{platform}.json", run_metadata())

    for kind in ("weights", "ckpt"):
        repo = hf_repo(kind)
        path = f"_sync_test/{platform}_{int(time.time())}"
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "hello.txt").write_text("sync test")
            hf_upload_dir(tmp, path, repo_id=repo, message="sync test")
        ok = any(f.startswith(path) for f in hf_list(repo, "_sync_test"))
        hf_delete_dir(repo, "_sync_test", message="remove sync test")
        log.info("HF %-7s repo %s: upload/list/delete %s", kind, repo, "OK" if ok else "FAILED")
        if not ok:
            raise SystemExit(f"HF sync check failed for {repo}")

    pushed = push_results([env_dir], f"env: record {platform} environment")
    log.info("GitHub push: %s", "OK" if pushed else "nothing new to push (already up to date)")


if __name__ == "__main__":
    main()
