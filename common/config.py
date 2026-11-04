"""Configuration loading.

Every entry point is driven by a YAML file. Configs may inherit from other
configs through a ``_base_`` key (a path or list of paths relative to the
file), and every config is merged on top of ``configs/project.yaml``.
Individual values can be overridden from the command line with
``--set key.subkey=value`` (values are parsed as YAML, so numbers, booleans
and lists work).
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
PROJECT_CONFIG = CONFIG_DIR / "project.yaml"


class Config(dict):
    """A dict with attribute access; nested dicts are wrapped recursively."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = _wrap(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"config has no key '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict where ``override`` is merged recursively into ``base``."""
    out = copy.deepcopy(_unwrap(base))
    for key, value in _unwrap(override).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file, resolving ``_base_`` inheritance."""
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base = data.pop("_base_", None)
    if base:
        merged: dict = {}
        for b in base if isinstance(base, list) else [base]:
            merged = deep_merge(merged, load_yaml(path.parent / b))
        data = deep_merge(merged, data)
    return data


def set_by_path(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path: str | Path | None = None, overrides: Iterable[str] = (),
                include_project: bool = True) -> Config:
    data: dict = load_yaml(PROJECT_CONFIG) if include_project and PROJECT_CONFIG.exists() else {}
    if path:
        data = deep_merge(data, load_yaml(path))
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key.sub=value")
        key, raw = item.split("=", 1)
        set_by_path(data, key.strip(), yaml.safe_load(raw))
    return Config(data)


def add_config_args(parser: argparse.ArgumentParser, default: str | None = None) -> None:
    parser.add_argument("--config", default=default, help="YAML config path (relative to repo root or absolute)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set training.lr=1e-4")


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_unwrap(cfg), fh, sort_keys=False)
