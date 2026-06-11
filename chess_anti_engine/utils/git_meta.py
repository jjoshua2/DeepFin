"""Repo metadata helpers shared by eval/audit scripts."""
from __future__ import annotations

import subprocess


def git_sha(*, short: bool = False) -> str:
    """Current git SHA, or "unknown" outside a repo / without git."""
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
