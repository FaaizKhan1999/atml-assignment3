"""Push small results to GitHub and large artifacts to the Hugging Face Hub.

GitHub: the GH_TOKEN is passed per command as an HTTP header (``git -c``), so
it is never written to .git/config (important on Kaggle, where
/kaggle/working is saved as notebook output). Commits are made in a private
shallow clone per process (see ``_publisher``), so parallel jobs never race on
the shared working copy; each job only touches its own results directory.

Hugging Face: two private model repos from configs/project.yaml:
* ``hf_repo_id``      - weights needed for evaluation/analysis (kept)
* ``hf_ckpt_repo_id`` - resumable recovery checkpoints (only the latest per run
                        is kept; the whole repo can be deleted at the end)
"""
from __future__ import annotations

import base64
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

import subprocess

from .config import REPO_ROOT, load_config
from .logging_utils import get_logger
from .secrets import get_secret

log = get_logger()


# ----------------------------------------------------------------------------- helpers

def _project():
    return load_config().get("project", {})


def _require(value, key: str) -> str:
    if not value or "YOUR_" in str(value):
        raise ValueError(f"Set project.{key} in configs/project.yaml before syncing.")
    return str(value)


# ----------------------------------------------------------------------------- GitHub

def _auth_args(token: str) -> list[str]:
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]


def git(args: Iterable[str], *, token: str | None = None, check: bool = True,
        cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    args = list(args)
    cmd = ["git", *(_auth_args(token) if token else []), *args]
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and res.returncode != 0:
        err = res.stderr.strip()
        if token:
            err = err.replace(token, "***")
        raise RuntimeError(f"git {' '.join(args)} failed ({res.returncode}): {err}")
    return res


def _set_identity(cwd: Path) -> None:
    proj = _project()
    git(["config", "user.email", _require(proj.get("git_user_email"), "git_user_email")], cwd=cwd)
    git(["config", "user.name", _require(proj.get("git_user_name"), "git_user_name")], cwd=cwd)


def current_branch() -> str:
    return git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def _publisher(token: str) -> tuple[Path, str]:
    """A private shallow clone used only for committing and pushing.

    Several jobs can run from the same working copy (e.g. one per GPU on Kaggle
    T4x2) while training processes keep appending to their log files. Doing git
    operations in that shared working copy would race (index locks, autostash
    conflicts with files being written). Instead each process copies its result
    files into its own clone, commits there, and pushes.
    """
    from .paths import get_paths

    tag = os.environ.get("ATML_PUBLISH_TAG") or f"pid{os.getpid()}"
    d = get_paths().scratch / "git_publish" / tag
    branch = current_branch()
    if not (d / ".git").exists():
        shutil.rmtree(d, ignore_errors=True)
        d.parent.mkdir(parents=True, exist_ok=True)
        url = git(["remote", "get-url", "origin"]).stdout.strip()
        git(["clone", "--depth", "1", "--branch", branch, url, str(d)], token=token, cwd=d.parent)
        _set_identity(d)
    return d, branch


def push_results(paths: Iterable[str | Path], message: str, retries: int = 5) -> bool:
    """Copy the given repo paths into a private clone, commit, push. False if nothing changed.

    Each job only writes its own results directory, so re-copying onto the latest
    remote state on every attempt never conflicts with other jobs' pushes.
    """
    token = get_secret("GH_TOKEN")
    rel = []
    for p in paths:
        p = Path(p)
        p = p if p.is_absolute() else REPO_ROOT / p
        if p.exists():
            rel.append(p.resolve().relative_to(REPO_ROOT))
    if not rel:
        log.warning("push_results: none of the given paths exist; nothing to push")
        return False
    pub, branch = _publisher(token)
    for attempt in range(1, retries + 1):
        git(["fetch", "--depth", "1", "origin", branch], token=token, cwd=pub)
        git(["reset", "--hard", f"origin/{branch}"], cwd=pub)
        for r in rel:
            src, dst = REPO_ROOT / r, pub / r
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        git(["add", "--", *map(str, rel)], cwd=pub)
        if git(["diff", "--cached", "--quiet"], check=False, cwd=pub).returncode == 0:
            log.info("push_results: no changes to commit")
            return False
        git(["commit", "-m", message], cwd=pub)
        push = git(["push", "origin", f"HEAD:{branch}"], token=token, check=False, cwd=pub)
        if push.returncode == 0:
            log.info("pushed to GitHub: %s", message)
            return True
        log.warning("git push failed (attempt %d): %s", attempt, push.stderr.strip()[-300:])
        time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    raise RuntimeError("Could not push results to GitHub after retries; the files are still on local disk.")


# ----------------------------------------------------------------------------- Hugging Face

def _hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=get_secret("HF_TOKEN"))


def hf_repo(kind: str = "weights") -> str:
    key = {"weights": "hf_repo_id", "ckpt": "hf_ckpt_repo_id"}[kind]
    return _require(_project().get(key), key)


def hf_ensure_repo(repo_id: str) -> None:
    _hf_api().create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)


def hf_upload_dir(local_dir: str | Path, path_in_repo: str, *, repo_id: str, message: str,
                  retries: int = 3) -> None:
    api = _hf_api()
    hf_ensure_repo(repo_id)
    for attempt in range(1, retries + 1):
        try:
            api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(local_dir),
                              path_in_repo=path_in_repo, commit_message=message)
            log.info("uploaded %s -> %s/%s", local_dir, repo_id, path_in_repo)
            return
        except Exception as exc:  # network hiccups are common on free tiers
            if attempt == retries:
                raise
            log.warning("HF upload failed (attempt %d): %s", attempt, exc)
            time.sleep(10 * attempt)


def hf_list(repo_id: str, prefix: str) -> list[str]:
    try:
        files = _hf_api().list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception:
        return []
    prefix = prefix.rstrip("/") + "/"
    return [f for f in files if f.startswith(prefix)]


def hf_delete_dir(repo_id: str, path_in_repo: str, message: str) -> None:
    if hf_list(repo_id, path_in_repo):
        _hf_api().delete_folder(path_in_repo=path_in_repo, repo_id=repo_id, repo_type="model",
                                commit_message=message)


def hf_download_dir(repo_id: str, path_in_repo: str, local_root: str | Path) -> Path:
    from huggingface_hub import snapshot_download

    path_in_repo = path_in_repo.rstrip("/")
    snapshot_download(repo_id=repo_id, repo_type="model", allow_patterns=[f"{path_in_repo}/*"],
                      local_dir=str(local_root), token=get_secret("HF_TOKEN"))
    return Path(local_root) / path_in_repo


# ----------------------------------------------------------------------------- recovery checkpoints

def push_recovery_checkpoint(local_ckpt: str | Path, run_key: str) -> None:
    """Upload a checkpoint folder (update_XXXXX) and delete older ones for this run."""
    local_ckpt = Path(local_ckpt)
    repo = hf_repo("ckpt")
    prefix = f"{run_key}/recovery"
    hf_upload_dir(local_ckpt, f"{prefix}/{local_ckpt.name}", repo_id=repo,
                  message=f"{run_key}: recovery {local_ckpt.name}")
    names = {f[len(prefix) + 1:].split("/")[0] for f in hf_list(repo, prefix)}
    for name in sorted(names - {local_ckpt.name}):
        hf_delete_dir(repo, f"{prefix}/{name}", message=f"{run_key}: drop old recovery {name}")


def fetch_latest_recovery_checkpoint(run_key: str, local_root: str | Path) -> Path | None:
    """Download the newest complete recovery checkpoint for a run, or return None."""
    try:
        repo = hf_repo("ckpt")
    except ValueError:
        return None
    prefix = f"{run_key}/recovery"
    names = sorted({f[len(prefix) + 1:].split("/")[0] for f in hf_list(repo, prefix)
                    if f[len(prefix) + 1:].startswith("update_")})
    if not names:
        return None
    path = hf_download_dir(repo, f"{prefix}/{names[-1]}", local_root)
    return path if (path / "COMPLETE").exists() else None
