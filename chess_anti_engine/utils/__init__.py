"""Utility helpers."""

import hashlib
from pathlib import Path

from .config_yaml import flatten_run_config_defaults, load_yaml_file


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


__all__ = [
    "flatten_run_config_defaults",
    "load_yaml_file",
    "sha256_file",
]
