"""Canonical Ray trial-dir / result.json discovery (stdlib only).

The diagnose_* scripts and status.py each grew a private copy of "find the
newest train_trial_* dir and read the last result.json line". This is the one
home for that logic; the copies should import from here.

Kept dependency-free (pathlib/json/os) so lightweight ops tools (loop_health)
can use it without pulling numpy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def default_run_dir() -> Path:
    return Path(os.environ.get("TRAIN_WORK_DIR", "runs/pbt2_small"))


def _trial_sort_key(trial_dir: Path) -> float:
    """Order trials by result.json mtime (actual iteration progress), falling
    back to the dir mtime — a trial dir's mtime bumps on any file write
    (checkpoint prune), so result.json is the truer recency signal."""
    result_path = trial_dir / "result.json"
    try:
        if result_path.exists():
            return result_path.stat().st_mtime
        return trial_dir.stat().st_mtime
    except OSError:
        return 0.0


def latest_trial_dir(run_dir: Path | None = None, *, required: bool = False) -> Path | None:
    """Newest ``train_trial_*`` dir under ``run_dir/tune`` (rglob, so a
    session/experiment-nested Ray layout still resolves).

    Returns None when none exist unless ``required`` (then raises
    FileNotFoundError) — the two call styles the existing copies used.
    """
    run_dir = run_dir or default_run_dir()
    tune = run_dir / "tune"
    # A trial dir is one that contains result.json (rglob catches nesting);
    # de-dup to the dirs themselves.
    trials = {p.parent for p in tune.rglob("train_trial_*/result.json")}
    trials |= {p for p in tune.rglob("train_trial_*") if p.is_dir()}
    if not trials:
        if required:
            raise FileNotFoundError(f"No Ray trial directories under {tune}")
        return None
    return max(trials, key=_trial_sort_key)


def latest_result(trial_dir: Path | None) -> dict[str, Any]:
    """The last parseable JSON line of ``trial_dir/result.json`` ({} if absent).

    Tolerates a torn trailing line from a live Ray append (keeps the last line
    that parsed) rather than crashing.
    """
    if trial_dir is None:
        return {}
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return {}
    latest: dict[str, Any] = {}
    with result_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn/partial line (live append) — keep the last good one
    return latest


def latest_result_path(run_dir: Path | None = None, *, required: bool = False) -> Path | None:
    """Path to the newest trial's result.json (for readers that stream lines)."""
    trial = latest_trial_dir(run_dir, required=required)
    if trial is None:
        return None
    result_path = trial / "result.json"
    if required and not result_path.exists():
        raise FileNotFoundError(f"No result.json in {trial}")
    return result_path
