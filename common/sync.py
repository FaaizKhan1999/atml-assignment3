"""Push small results to GitHub and large artifacts to the Hugging Face Hub.

GitHub: the GH_TOKEN is passed per command as an HTTP header (``git -c``), so
it is never written to .git/config (important on Kaggle, where
/kaggle/working is saved as notebook output). Each job commits only its own
results directory, so parallel jobs don't conflict; pushes rebase onto the
remote and retry.

Hugging Face: two private model repos from configs/project.yaml:
* ``hf_repo_id``      - weights needed for evaluation/analysis (kept)
* ``hf_ckpt_repo_id`` - resumable recovery checkpoints (only the latest per run
                        is kept; the whole repo can be deleted at the end)
"""
from __future__ import annotations

import base64
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


def ensure_git_identity() -> None:
    proj = _project()
    if not git(["config", "user.email"], check=False).stdout.strip():
        git(["config", "user.email", _require(proj.get("git_user_email"), "git_user_email")])
    if not git(["config", "user.name"], check=False).stdout.strip():
        git(["config", "user.name", _require(proj.get("git_user_name"), "git_user_name")])


def current_branch() -> str:
    return git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def push_results(paths: Iterable[str | Path], message: str, retries: int = 5) -> bool:
    """Commit the given repo paths and push. Returns False if there was nothing new."""
    token = get_secret("GH_TOKEN")
    ensure_git_identity()
    rel = []
    for p in paths:
        p = Path(p)
        p = p if p.is_absolute() else REPO_ROOT / p
        if p.exists():
            rel.append(str(p.resolve().relative_to(REPO_ROOT)))
    if not rel:
        log.warning("push_results: none of the given paths exist; nothing to push")
        return False
    git(["add", "--", *rel])
    if git(["diff", "--cached", "--quiet"], check=False).returncode == 0:
        log.info("push_results: no changes to commit")
        return False
    git(["commit", "-m", message])
    branch = current_branch()
    for attempt in range(1, retries + 1):
        pull = git(["pull", "--rebase", "--autostash", "origin", branch], token=token, check=False)
        if pull.returncode != 0:
            git(["rebase", "--abort"], check=False)
            log.warning("git pull --rebase failed (attempt %d): %s", attempt, pull.stderr.strip()[-300:])
        else:
            push = git(["push", "origin", f"HEAD:{branch}"], token=token, check=False)
            if push.returncode == 0:
                log.info("pushed to GitHub: %s", message)
                return True
            log.warning("git push failed (attempt %d): %s", attempt, push.stderr.strip()[-300:])
        time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    raise RuntimeError("Could not push results; the commit is kept locally. Rerun push_results or push manually.")


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
