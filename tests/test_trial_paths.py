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


def test_latest_trial_dir_picks_newest_by_result_mtime(tmp_path: Path) -> None:
    old = _trial(tmp_path, "train_trial_aaaa", [1, 2])
    new = _trial(tmp_path, "train_trial_bbbb", [1, 2, 3])
    # Make 'new' the more recently progressed trial regardless of dir mtime.
    os.utime(old / "result.json", (1_000, 1_000))
    os.utime(new / "result.json", (2_000, 2_000))
    assert trial_paths.latest_trial_dir(tmp_path) == new


def test_latest_trial_dir_resolves_session_nested_layout(tmp_path: Path) -> None:
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
