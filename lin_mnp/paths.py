"""Portable repository paths configured through environment variables."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def environment_path(name: str, default: Path) -> Path:
    """Return an absolute path from an environment variable or local default."""
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


DATA_ROOT = environment_path("LIN_MNP_DATA_ROOT", REPOSITORY_ROOT / "data")
CACHE_ROOT = environment_path("LIN_MNP_CACHE_ROOT", DATA_ROOT / "cache")
RESULTS_ROOT = environment_path("LIN_MNP_RESULTS_ROOT", REPOSITORY_ROOT / "outputs")
ARCHIVE_ROOT = environment_path("LIN_MNP_ARCHIVE_ROOT", DATA_ROOT / "archive")
MODEL_ROOT = environment_path("LIN_MNP_MODEL_ROOT", DATA_ROOT / "models")
