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
from typing import Any, Literal, overload


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


def _candidate_trial_dirs(tune: Path) -> list[Path]:
    """The ``train_trial_*`` dirs under ``tune``, flat layout preferred.

    Production nests trials directly (``tune/train_trial_*``); a flat glob
    matches exactly those and cannot pick up a stray backup nested deeper.
    Only when the flat layout is empty do we fall back to a recursive search,
    so a session/experiment-nested Ray layout still resolves without the flat
    case ever seeing unrelated nested dirs.
    """
    flat = [p for p in tune.glob("train_trial_*") if p.is_dir()]
    if flat:
        return flat
    nested = {p.parent for p in tune.rglob("train_trial_*/result.json")}
    nested |= {p for p in tune.rglob("train_trial_*") if p.is_dir()}
    return list(nested)


@overload
def latest_trial_dir(run_dir: Path | None = ..., *, required: Literal[True]) -> Path: ...
@overload
def latest_trial_dir(run_dir: Path | None = ..., *, required: bool = ...) -> Path | None: ...
def latest_trial_dir(run_dir: Path | None = None, *, required: bool = False) -> Path | None:
    """Newest ``train_trial_*`` dir under ``run_dir/tune``.

    Ranked by ``result.json`` mtime (see ``_trial_sort_key``); the trial-name
    is a secondary key so an mtime tie resolves deterministically rather than
    by set/glob iteration order. Returns None when none exist unless
    ``required`` (then raises FileNotFoundError).
    """
    run_dir = run_dir or default_run_dir()
    tune = run_dir / "tune"
    trials = _candidate_trial_dirs(tune)
    if not trials:
        if required:
            raise FileNotFoundError(f"No Ray trial directories under {tune}")
        return None
    return max(trials, key=lambda p: (_trial_sort_key(p), p.name))


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
