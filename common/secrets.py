"""Token lookup that works the same on Kaggle, Colab and locally.

Order: environment variable -> Kaggle Secrets -> Colab secrets. Colab's
``userdata`` only works inside the notebook kernel, not in ``!python``
subprocesses, so run ``export_secrets()`` in a notebook cell first; it copies
the tokens into ``os.environ`` where every subprocess inherits them.
"""
from __future__ import annotations

import os

DEFAULT_SECRETS = ("HF_TOKEN", "GH_TOKEN")


def _from_kaggle(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def _from_colab(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name)
    except Exception:
        return None


def get_secret(name: str, required: bool = True) -> str | None:
    value = os.environ.get(name) or _from_kaggle(name) or _from_colab(name)
    if value:
        os.environ[name] = value
        return value
    if required:
        raise RuntimeError(
            f"Secret '{name}' not found. Add it to Kaggle Secrets / Colab secrets, then run in a notebook cell:\n"
            "    from common.secrets import export_secrets; export_secrets()"
        )
    return None


def export_secrets(names=DEFAULT_SECRETS) -> dict[str, bool]:
    """Copy secrets into os.environ (call from a notebook cell). Never prints values."""
    found = {n: get_secret(n, required=False) is not None for n in names}
    for n, ok in found.items():
        print(f"{n}: {'found' if ok else 'MISSING'}")
    return found
