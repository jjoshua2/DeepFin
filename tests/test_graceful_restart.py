from __future__ import annotations

import json
from pathlib import Path

from scripts.graceful_restart import _active_trials, _required_paused_count


def _write_progress(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text("a,b\n" + "\n".join("1,2" for _ in range(80)) + "\n")


def test_active_trials_uses_latest_experiment_state(tmp_path: Path) -> None:
    live = tmp_path / "train_trial_live_00000_0"
    old = tmp_path / "train_trial_old_00000_0"
    _write_progress(live / "progress.csv")
    _write_progress(old / "progress.csv")
    state = {
        "trial_data": [
            [
                json.dumps(
                    {
                        "trial_id": "live_00000",
                        "status": "RUNNING",
                        "relative_logdir": live.name,
                    }
                )
            ],
            [
                json.dumps(
                    {
                        "trial_id": "old_00000",
                        "status": "TERMINATED",
                        "relative_logdir": old.name,
                    }
                )
            ],
        ]
    }
    (tmp_path / "experiment_state-2026-05-29_00-00-00.json").write_text(json.dumps(state))

    assert _active_trials(tmp_path) == [live / "progress.csv"]


def test_active_trials_ignores_scheduler_paused_trials(tmp_path: Path) -> None:
    running = tmp_path / "train_trial_running_00000_0"
    paused = tmp_path / "train_trial_paused_00000_0"
    _write_progress(running / "progress.csv")
    _write_progress(paused / "progress.csv")
    state = {
        "trial_data": [
            [
                json.dumps(
                    {
                        "trial_id": "running_00000",
                        "status": "RUNNING",
                        "relative_logdir": running.name,
                    }
                )
            ],
            [
                json.dumps(
                    {
                        "trial_id": "paused_00000",
                        "status": "PAUSED",
                        "relative_logdir": paused.name,
                    }
                )
            ],
        ]
    }
    (tmp_path / "experiment_state-2026-05-29_00-00-00.json").write_text(json.dumps(state))

    assert _active_trials(tmp_path) == [running / "progress.csv"]


def test_required_paused_count_accepts_deprecated_wait_arg() -> None:
    assert _required_paused_count(4, 0) == 4
    assert _required_paused_count(4, 2) == 2
    assert _required_paused_count(1, 3) == 1
