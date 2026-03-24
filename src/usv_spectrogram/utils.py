"""Miscellaneous helpers shared across modules."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
