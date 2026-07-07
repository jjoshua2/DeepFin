"""Canonical Ray trial-dir / result.json discovery (stdlib only).

The home for "find the newest train_trial_* dir under run_dir/tune and read the
last result.json line". The diagnose_* scripts import from here; status.py keeps
its own finder deliberately — it discovers by live Ray SESSION artifacts
(``/tmp/ray/session_*/artifacts``), a different root and concern from this
run_dir/tune scan, so it is not a duplicate of this logic.

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


def _trial_sort_key(trial_dir: Path) -> tuple[int, float, str]:
    """Rank key for a trial dir: (has_result.json, mtime, name).

    A dir WITH result.json always outranks one without — so a freshly-created
    empty trial dir (its own mtime is 'now' right after a restart) never beats
    an older but populated trial, which would otherwise make callers read an
    empty dir. Within each group, order by result.json mtime (the truer recency
    signal — a trial dir's mtime bumps on any file write, e.g. checkpoint
    pruning), falling back to dir mtime; the name is a final tiebreak so an
    mtime tie resolves deterministically instead of by glob/set order."""
    result_path = trial_dir / "result.json"
    try:
        if result_path.exists():
            return (1, result_path.stat().st_mtime, trial_dir.name)
        return (0, trial_dir.stat().st_mtime, trial_dir.name)
    except OSError:
        return (0, 0.0, trial_dir.name)


def _candidate_trial_dirs(tune: Path) -> list[Path]:
    """The ``train_trial_*`` dirs under ``tune``, flat layout preferred.

    Production nests trials directly (``tune/train_trial_*``); a flat glob
    matches exactly those and cannot pick up a stray backup nested deeper.
    Only when the flat layout is empty do we fall back to a single recursive
    search, so a session/experiment-nested Ray layout still resolves without
    the flat case ever seeing unrelated nested dirs.
    """
    flat = [p for p in tune.glob("train_trial_*") if p.is_dir()]
    if flat:
        return flat
    return [p for p in tune.rglob("train_trial_*") if p.is_dir()]


@overload
def latest_trial_dir(run_dir: Path | None = ..., *, required: Literal[True]) -> Path: ...
@overload
def latest_trial_dir(run_dir: Path | None = ..., *, required: bool = ...) -> Path | None: ...
def latest_trial_dir(run_dir: Path | None = None, *, required: bool = False) -> Path | None:
    """Newest ``train_trial_*`` dir under ``run_dir/tune``.

    Ranked by ``_trial_sort_key`` — a populated trial (has result.json) beats an
    empty one, then by result.json mtime, then by name for a deterministic tie.
    Returns None when none exist unless ``required`` (then raises
    FileNotFoundError).
    """
    run_dir = run_dir or default_run_dir()
    tune = run_dir / "tune"
    trials = _candidate_trial_dirs(tune)
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
