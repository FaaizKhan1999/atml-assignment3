"""Non-fatal GitHub publishing for long jobs.

``push_results`` raises after its retries are exhausted. During evaluation (and
any new long job) a temporary GitHub outage must not kill the job: results are
written cumulatively to disk, so a skipped push is simply retried by the next
call. Use ``try_push`` there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .logging_utils import get_logger

log = get_logger()


def try_push(paths: Iterable[str | Path], message: str, enabled: bool = True) -> bool:
    if not enabled:
        return False
    from .sync import push_results

    try:
        return push_results(list(paths), message)
    except Exception as exc:  # network / GitHub outage: keep going, next push includes these files
        log.warning("GitHub push failed (%s); continuing. The files stay on disk and go out with the next push.",
                    str(exc)[:300])
        return False
