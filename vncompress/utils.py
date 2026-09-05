"""
Small, genuinely-shared utilities.
====================================
Reproducibility-critical helpers (seed, environment/config snapshotting) live
in config.py alongside ExperimentConfig, since they're only ever used
together. This module holds the handful of things everything else in the
repo needs: JSON I/O and a repo-relative path helper. Resist the urge to add
anything here that only one script uses -- see refactor.md section 12.
"""

from __future__ import annotations

import json
import os
from typing import Any


def save_json(path: str, data: Any) -> None:
    """Write `data` as pretty-printed, UTF-8 (non-ASCII-escaped) JSON,
    creating parent directories as needed."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def repo_root() -> str:
    """Absolute path to the repository root (the parent of this package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
