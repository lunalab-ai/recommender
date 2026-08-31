"""Dataset path helpers for local and Colab teaching environments."""

from __future__ import annotations

import os
from pathlib import Path


def course_root() -> Path:
    """Return the repository root when running from an editable checkout."""
    return Path(__file__).resolve().parents[2]


def data_path(filename: str, *, local_only: bool = False) -> Path:
    """Resolve a course data file without hard-coding a user's home directory.

    The ``RECOMMENDER_DATA_DIR`` environment variable takes precedence. Otherwise,
    local-only files are searched under ``data/local`` and bundled samples under
    ``data/sample``.
    """
    custom_dir = os.getenv("RECOMMENDER_DATA_DIR")
    if custom_dir:
        candidate = Path(custom_dir).expanduser() / filename
    else:
        subdir = "local" if local_only else "sample"
        candidate = course_root() / "data" / subdir / filename
    if not candidate.exists():
        raise FileNotFoundError(
            f"Data file not found: {candidate}. See data/README.md and data/registry.yml."
        )
    return candidate
