"""Utility helpers."""

import hashlib
from pathlib import Path

from .config_yaml import flatten_run_config_defaults, load_yaml_file


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "flatten_run_config_defaults",
    "load_yaml_file",
    "sha256_file",
]
