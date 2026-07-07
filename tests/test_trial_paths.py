"""Canonical trial-dir / result.json discovery (scripts/trial_paths.py)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import trial_paths


def _trial(run: Path, name: str, iters: list[int]) -> Path:
    d = run / "tune" / name
    d.mkdir(parents=True)
    with (d / "result.json").open("w", encoding="utf-8") as f:
        for it in iters:
            f.write(json.dumps({"training_iteration": it}) + "\n")
    return d


def test_latest_trial_dir_none_when_empty(tmp_path: Path) -> None:
    (tmp_path / "tune").mkdir()
    assert trial_paths.latest_trial_dir(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        trial_paths.latest_trial_dir(tmp_path, required=True)


def test_latest_trial_dir_ranks_by_result_mtime_over_dir_mtime(tmp_path: Path) -> None:
    # The load-bearing invariant: rank by result.json mtime, NOT dir mtime.
    # A dir-mtime implementation would pick 'stale' here (its DIR is newer, as
    # checkpoint pruning bumps a defunct trial's dir), so the conflict is
    # deliberate — this test must distinguish the two sort strategies.
    stale = _trial(tmp_path, "train_trial_aaaa", [1, 2])
    active = _trial(tmp_path, "train_trial_bbbb", [1, 2, 3])
    os.utime(stale / "result.json", (1_000, 1_000))
    os.utime(active / "result.json", (2_000, 2_000))
    os.utime(stale, (9_000, 9_000))   # defunct trial's DIR mtime bumped newer
    os.utime(active, (1_000, 1_000))  # active trial's DIR mtime older
    assert trial_paths.latest_trial_dir(tmp_path) == active


def test_latest_trial_dir_skips_empty_new_dir(tmp_path: Path) -> None:
    # A freshly-created post-restart trial dir has no result.json yet and its
    # own mtime is 'now'; it must NOT outrank the older populated trial, else
    # latest_result_path(required=True) would raise on an empty dir.
    populated = _trial(tmp_path, "train_trial_old", [1, 2])
    empty = tmp_path / "tune" / "train_trial_new"
    empty.mkdir(parents=True)
    os.utime(populated / "result.json", (1_000, 1_000))
    os.utime(empty, (9_000, 9_000))  # new empty dir mtime is newer
    assert trial_paths.latest_trial_dir(tmp_path) == populated
    assert trial_paths.latest_result_path(tmp_path, required=True) == populated / "result.json"


def test_latest_trial_dir_tie_is_deterministic(tmp_path: Path) -> None:
    # Equal sort keys must not resolve by PYTHONHASHSEED-salted set order.
    a = _trial(tmp_path, "train_trial_aaaa", [1])
    b = _trial(tmp_path, "train_trial_bbbb", [1])
    for d in (a, b):
        os.utime(d / "result.json", (5_000, 5_000))
    # Secondary key is the path name, so the larger name wins, every run.
    assert trial_paths.latest_trial_dir(tmp_path) == b


def test_latest_trial_dir_prefers_flat_over_nested_stray(tmp_path: Path) -> None:
    # A flat production trial must win over a stray nested train_trial_* backup
    # even if the backup's result.json is newer — the flat layout is preferred
    # and the recursive search is only a fallback when flat is empty.
    live = _trial(tmp_path, "train_trial_live", [1, 2])
    stray = _trial(tmp_path, "train_trial_live/backup/train_trial_stray", [1])
    os.utime(live / "result.json", (1_000, 1_000))
    os.utime(stray / "result.json", (9_000, 9_000))
    assert trial_paths.latest_trial_dir(tmp_path) == live


def test_latest_trial_dir_resolves_session_nested_layout_when_no_flat(tmp_path: Path) -> None:
    # No flat trial -> fall back to the recursive search for nested layouts.
    d = _trial(tmp_path, "session_x/artifacts/exp/train_trial_cccc", [5])
    assert trial_paths.latest_trial_dir(tmp_path) == d


def test_latest_result_reads_last_line(tmp_path: Path) -> None:
    d = _trial(tmp_path, "train_trial_dddd", [7, 8, 9])
    assert trial_paths.latest_result(d) == {"training_iteration": 9}


def test_latest_result_tolerates_torn_trailing_line(tmp_path: Path) -> None:
    d = tmp_path / "tune" / "train_trial_eeee"
    d.mkdir(parents=True)
    (d / "result.json").write_text(
        json.dumps({"training_iteration": 1}) + "\n" + '{"training_iter',
        encoding="utf-8",
    )
    assert trial_paths.latest_result(d) == {"training_iteration": 1}


def test_latest_result_missing_returns_empty(tmp_path: Path) -> None:
    assert trial_paths.latest_result(None) == {}
    assert trial_paths.latest_result(tmp_path / "nope") == {}
