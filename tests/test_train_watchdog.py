"""Unit tests for scripts/train_watchdog.py pure decision logic (no live PIDs)."""
from __future__ import annotations

import json
from pathlib import Path

from tests.script_loading import load_script_module

wd = load_script_module("train_watchdog.py", "train_watchdog")


def _snap(
    *,
    pid: int | None = 1234,
    pid_alive: bool = True,
    pause_txt: str | None = None,
    rows: int = 10,
    rows_prev: int | None = 9,
    minutes_flat: float = 0.0,
    trial_dir: str | None = "train_trial_abc_00000",
    progress_file: str | None = None,
    intentional_stop: bool = False,
):
    return wd.ProgressSnapshot(
        pid=pid,
        pid_alive=pid_alive,
        pause_txt=pause_txt,
        rows=rows,
        rows_prev=rows_prev,
        minutes_flat=minutes_flat,
        trial_dir=trial_dir,
        progress_file=progress_file,
        intentional_stop=intentional_stop,
    )


# --- decide() state machine -------------------------------------------------


def test_crashed_when_pidfile_missing_and_no_marker() -> None:
    """No PID and no intentional-stop marker == it died. 2026-08-08 regression."""
    v = wd.decide(_snap(pid=None, pid_alive=False), stall_minutes=90.0)
    assert v.state == wd.STATE_CRASHED
    assert v.exit_code == wd.EXIT_CRASHED
    assert "watchdog: CRASHED" in v.format_line()
    assert "pid=none" in v.format_line()


def test_crashed_when_pid_dead_and_no_marker() -> None:
    v = wd.decide(_snap(pid=99999, pid_alive=False), stall_minutes=90.0)
    assert v.state == wd.STATE_CRASHED
    assert v.exit_code == wd.EXIT_CRASHED
    assert "pid=99999" in v.format_line()


def test_stopped_when_pidfile_missing_and_marker_present() -> None:
    """`train.sh stop` touches the marker before killing -> deliberate, stays quiet."""
    v = wd.decide(_snap(pid=None, pid_alive=False, intentional_stop=True), stall_minutes=90.0)
    assert v.state == wd.STATE_STOPPED
    assert v.exit_code == wd.EXIT_STOPPED
    assert "watchdog: STOPPED" in v.format_line()


def test_stopped_when_pid_dead_and_marker_present() -> None:
    v = wd.decide(_snap(pid=99999, pid_alive=False, intentional_stop=True), stall_minutes=90.0)
    assert v.state == wd.STATE_STOPPED
    assert v.exit_code == wd.EXIT_STOPPED


def test_crashed_and_stopped_differ_only_by_the_marker() -> None:
    """The NEGATIVE CONTROL: identical snapshots, marker is the only difference.

    Without this, a change that made `intentional_stop` unreachable (always
    False, or always True) would still pass every other test in this file --
    the exact 'a value is accepted and then silently ignored' failure this
    codebase keeps producing.
    """
    crashed = wd.decide(
        _snap(
            pid=4242, pid_alive=False, rows=249, rows_prev=249,
            minutes_flat=280.0, intentional_stop=False,
        ),
        stall_minutes=90.0,
    )
    stopped = wd.decide(
        _snap(
            pid=4242, pid_alive=False, rows=249, rows_prev=249,
            minutes_flat=280.0, intentional_stop=True,
        ),
        stall_minutes=90.0,
    )
    assert crashed.state == wd.STATE_CRASHED
    assert stopped.state == wd.STATE_STOPPED
    assert crashed.exit_code != stopped.exit_code
    # Everything else about the two verdicts is identical.
    assert crashed.details == stopped.details


def test_paused_held_when_pause_and_flat() -> None:
    v = wd.decide(
        _snap(
            pause_txt="runs/pbt2_small/tune/pause.txt",
            rows=42,
            rows_prev=42,
            minutes_flat=15.0,
        ),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_PAUSED_HELD
    assert v.exit_code == wd.EXIT_PAUSED_HELD
    line = v.format_line()
    assert "pause_txt=runs/pbt2_small/tune/pause.txt" in line
    assert "minutes_flat=15.0" in line


def test_paused_held_even_under_stall_threshold() -> None:
    """Boundary hold: no new rows, pause present — PAUSED-HELD, not OK."""
    v = wd.decide(
        _snap(pause_txt="/tmp/pause.txt", rows=5, rows_prev=5, minutes_flat=1.0),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_PAUSED_HELD


def test_ok_when_pause_but_rows_grew() -> None:
    """Pause marker exists but iteration still finished a row — not yet held."""
    v = wd.decide(
        _snap(pause_txt="/tmp/pause.txt", rows=11, rows_prev=10, minutes_flat=0.0),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_OK
    assert v.exit_code == wd.EXIT_OK


def test_stalled_when_flat_beyond_threshold() -> None:
    v = wd.decide(
        _snap(pause_txt=None, rows=20, rows_prev=20, minutes_flat=91.0),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_STALLED
    assert v.exit_code == wd.EXIT_STALLED
    assert "minutes_flat=91.0" in v.format_line()


def test_ok_when_flat_within_stall_window() -> None:
    """Normal mid-iteration: flat for 40 min, stall threshold 90 → OK."""
    v = wd.decide(
        _snap(pause_txt=None, rows=20, rows_prev=20, minutes_flat=40.0),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_OK
    assert v.exit_code == wd.EXIT_OK


def test_ok_when_rows_grew() -> None:
    v = wd.decide(
        _snap(rows=21, rows_prev=20, minutes_flat=0.0),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_OK
    line = v.format_line()
    assert line.startswith("watchdog: OK")
    assert "rows=21" in line
    assert "rows_prev=20" in line


def test_ok_on_first_observation_no_prev() -> None:
    v = wd.decide(
        _snap(rows=7, rows_prev=None, minutes_flat=0.0),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_OK
    assert "rows_prev=none" in v.format_line()


def test_dead_pid_takes_priority_over_pause_and_stall() -> None:
    """A dead PID outranks pause/stall; the marker only picks WHICH dead state."""
    crashed = wd.decide(
        _snap(
            pid=1, pid_alive=False, pause_txt="/x/pause.txt",
            rows=1, rows_prev=1, minutes_flat=999.0,
        ),
        stall_minutes=90.0,
    )
    stopped = wd.decide(
        _snap(
            pid=1, pid_alive=False, pause_txt="/x/pause.txt",
            rows=1, rows_prev=1, minutes_flat=999.0, intentional_stop=True,
        ),
        stall_minutes=90.0,
    )
    assert crashed.state == wd.STATE_CRASHED
    assert stopped.state == wd.STATE_STOPPED


def test_paused_held_takes_priority_over_stalled() -> None:
    """Flat past stall threshold WITH pause → PAUSED-HELD, not STALLED."""
    v = wd.decide(
        _snap(
            pause_txt="/x/pause.txt",
            rows=3,
            rows_prev=3,
            minutes_flat=200.0,
        ),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_PAUSED_HELD


# --- compute_flatness() -----------------------------------------------------


def test_flatness_first_sight_arms_clock() -> None:
    rows_prev, minutes_flat, new_state, grew = wd.compute_flatness(
        rows=10, trial="t1", now=1000.0, prev=None,
    )
    assert rows_prev is None
    assert minutes_flat == 0.0
    assert not grew
    assert new_state.rows == 10
    assert new_state.wall_time == 1000.0


def test_flatness_growth_resets_clock() -> None:
    prev = wd.PersistedState(rows=10, wall_time=1000.0, trial="t1")
    rows_prev, minutes_flat, new_state, grew = wd.compute_flatness(
        rows=12, trial="t1", now=4000.0, prev=prev,
    )
    assert rows_prev == 10
    assert minutes_flat == 0.0
    assert grew
    assert new_state.rows == 12
    assert new_state.wall_time == 4000.0


def test_flatness_same_rows_accumulates_minutes() -> None:
    prev = wd.PersistedState(rows=10, wall_time=1000.0, trial="t1")
    # 90 minutes later
    rows_prev, minutes_flat, new_state, grew = wd.compute_flatness(
        rows=10, trial="t1", now=1000.0 + 90 * 60, prev=prev,
    )
    assert rows_prev == 10
    assert minutes_flat == 90.0
    assert not grew
    # Wall time stays at first flat observation so the next call accumulates.
    assert new_state.wall_time == 1000.0
    assert new_state is prev


def test_flatness_row_decrease_resets() -> None:
    prev = wd.PersistedState(rows=50, wall_time=1000.0, trial="t1")
    rows_prev, minutes_flat, new_state, grew = wd.compute_flatness(
        rows=2, trial="t1", now=5000.0, prev=prev,
    )
    assert rows_prev == 50
    assert minutes_flat == 0.0
    assert grew
    assert new_state.rows == 2


def test_flatness_trial_switch_resets() -> None:
    prev = wd.PersistedState(rows=50, wall_time=1000.0, trial="old_trial")
    _rows_prev, minutes_flat, new_state, grew = wd.compute_flatness(
        rows=50, trial="new_trial", now=5000.0, prev=prev,
    )
    assert grew
    assert minutes_flat == 0.0
    assert new_state.trial == "new_trial"


# --- filesystem helpers (tmp_path, no live processes) -----------------------


def test_read_pid_missing(tmp_path: Path) -> None:
    assert wd.read_pid(tmp_path / "missing.pid") is None


def test_read_pid_valid(tmp_path: Path) -> None:
    p = tmp_path / "chess_training.pid"
    p.write_text("4242\n")
    assert wd.read_pid(p) == 4242


def test_read_pid_garbage(tmp_path: Path) -> None:
    p = tmp_path / "chess_training.pid"
    p.write_text("not-a-pid\n")
    assert wd.read_pid(p) is None


def test_find_pause_txt_root_and_nested(tmp_path: Path) -> None:
    tune = tmp_path / "tune"
    tune.mkdir()
    assert wd.find_pause_txt(tune) is None
    (tune / "pause.txt").write_text("hold\n")
    assert wd.find_pause_txt(tune) == tune / "pause.txt"

    # Nested per-trial marker when root is absent.
    (tune / "pause.txt").unlink()
    trial = tune / "train_trial_x_00000"
    trial.mkdir()
    marker = trial / "pause.txt"
    marker.write_text("hold\n")
    assert wd.find_pause_txt(tune) == marker


def test_newest_trial_and_progress_csv_rows(tmp_path: Path) -> None:
    tune = tmp_path / "tune"
    old = tune / "train_trial_old_00000"
    new = tune / "train_trial_new_00000"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "progress.csv").write_text("a,b\n1,2\n3,4\n")
    # newer mtime
    import time as _time
    _time.sleep(0.01)
    (new / "progress.csv").write_text("a,b\n1,2\n3,4\n5,6\n")

    trial = wd.newest_trial_dir(tune)
    assert trial is not None
    assert trial.name == "train_trial_new_00000"
    rows, src = wd.count_progress_rows(trial)
    assert rows == 3  # header excluded
    assert src is not None
    assert src.endswith("progress.csv")


def test_result_json_fallback_line_count(tmp_path: Path) -> None:
    trial = tmp_path / "train_trial_x"
    trial.mkdir()
    (trial / "result.json").write_text(
        '{"training_iteration": 1}\n{"training_iteration": 2}\n\n'
    )
    rows, src = wd.count_progress_rows(trial)
    assert rows == 2
    assert src is not None
    assert src.endswith("result.json")


def test_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "wd.json"
    state = wd.PersistedState(rows=7, wall_time=123.5, trial="t")
    wd.save_state(path, state)
    loaded = wd.load_state(path)
    assert loaded is not None
    assert loaded.rows == 7
    assert loaded.wall_time == 123.5
    assert loaded.trial == "t"


def test_build_snapshot_end_to_end(tmp_path: Path) -> None:
    """Wire PID + tune layout + state through build_snapshot with injected liveness."""
    pidfile = tmp_path / "t.pid"
    pidfile.write_text("555\n")
    work = tmp_path / "run"
    tune = work / "tune"
    trial = tune / "train_trial_live_00000_0"
    trial.mkdir(parents=True)
    (trial / "progress.csv").write_text("a,b\n1,2\n3,4\n")
    state_path = tmp_path / "state.json"
    marker = tmp_path / "intentional_stop"  # absent unless a case creates it

    # First check: arms clock, OK.
    snap1, new1 = wd.build_snapshot(
        pidfile=pidfile,
        work_dir=work,
        state_path=state_path,
        stop_marker=marker,
        now=1_000.0,
        pid_alive_fn=lambda _pid: True,
    )
    assert snap1.intentional_stop is False
    assert snap1.pid == 555
    assert snap1.pid_alive
    assert snap1.rows == 2
    assert snap1.rows_prev is None
    v1 = wd.decide(snap1, stall_minutes=90.0)
    assert v1.state == wd.STATE_OK
    wd.save_state(state_path, new1)

    # Second check, same rows, 100 min later, no pause → STALLED.
    snap2, new2 = wd.build_snapshot(
        pidfile=pidfile,
        work_dir=work,
        state_path=state_path,
        stop_marker=marker,
        now=1_000.0 + 100 * 60,
        pid_alive_fn=lambda _pid: True,
    )
    assert snap2.rows == 2
    assert snap2.rows_prev == 2
    assert snap2.minutes_flat == 100.0
    assert snap2.pause_txt is None
    v2 = wd.decide(snap2, stall_minutes=90.0)
    assert v2.state == wd.STATE_STALLED
    wd.save_state(state_path, new2)

    # With pause.txt → PAUSED-HELD (priority over STALLED).
    (tune / "pause.txt").write_text("graceful\n")
    snap3, _ = wd.build_snapshot(
        pidfile=pidfile,
        work_dir=work,
        state_path=state_path,
        stop_marker=marker,
        now=1_000.0 + 100 * 60,
        pid_alive_fn=lambda _pid: True,
    )
    assert snap3.pause_txt is not None
    v3 = wd.decide(snap3, stall_minutes=90.0)
    assert v3.state == wd.STATE_PAUSED_HELD

    # Dead PID, marker ABSENT → CRASHED.
    snap4, _ = wd.build_snapshot(
        pidfile=pidfile,
        work_dir=work,
        state_path=state_path,
        stop_marker=marker,
        now=1_000.0 + 100 * 60,
        pid_alive_fn=lambda _pid: False,
    )
    assert snap4.intentional_stop is False
    assert wd.decide(snap4, stall_minutes=90.0).state == wd.STATE_CRASHED

    # Same dead PID, marker PRESENT on disk → STOPPED. Proves build_snapshot
    # actually stats the file rather than hardcoding the flag.
    marker.write_text("")
    snap5, _ = wd.build_snapshot(
        pidfile=pidfile,
        work_dir=work,
        state_path=state_path,
        stop_marker=marker,
        now=1_000.0 + 100 * 60,
        pid_alive_fn=lambda _pid: False,
    )
    assert snap5.intentional_stop is True
    assert wd.decide(snap5, stall_minutes=90.0).state == wd.STATE_STOPPED


def test_main_cli_crashed(tmp_path: Path, capsys) -> None:
    """No PID, no marker: the CLI must report CRASHED, exit 5."""
    pidfile = tmp_path / "missing.pid"
    code = wd.main([
        "--pidfile", str(pidfile),
        "--work-dir", str(tmp_path / "run"),
        "--state", str(tmp_path / "state.json"),
        "--stop-marker", str(tmp_path / "no_such_marker"),
        "--stall-minutes", "90",
    ])
    assert code == wd.EXIT_CRASHED
    out = capsys.readouterr().out.strip()
    assert out.startswith("watchdog: CRASHED")
    # Exactly one line.
    assert "\n" not in out


def test_main_cli_ok_growth(tmp_path: Path, capsys) -> None:
    pidfile = tmp_path / "t.pid"
    pidfile.write_text(f"{os_getpid()}\n")
    work = tmp_path / "run"
    trial = work / "tune" / "train_trial_x_00000"
    trial.mkdir(parents=True)
    (trial / "progress.csv").write_text("a,b\n1,2\n")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"rows": 0, "wall_time": 0.0, "trial": "train_trial_x_00000"}))

    code = wd.main([
        "--pidfile", str(pidfile),
        "--work-dir", str(work),
        "--state", str(state),
        "--stall-minutes", "90",
    ])
    assert code == wd.EXIT_OK
    out = capsys.readouterr().out.strip()
    assert out.startswith("watchdog: OK")
    assert "rows=1" in out


def os_getpid() -> int:
    import os
    return os.getpid()


def test_main_never_raises_on_bad_stall(capsys) -> None:
    code = wd.main(["--stall-minutes", "0"])
    assert code == wd.EXIT_ERROR
    out = capsys.readouterr().out.strip()
    assert out.startswith("watchdog: ERROR")


def test_notify_cmd_failsoft(tmp_path: Path, capsys) -> None:
    """Non-OK triggers notify; a broken notify command must not crash the watchdog."""
    pidfile = tmp_path / "missing.pid"
    code = wd.main([
        "--pidfile", str(pidfile),
        "--work-dir", str(tmp_path / "run"),
        "--state", str(tmp_path / "state.json"),
        "--stop-marker", str(tmp_path / "no_such_marker"),
        "--notify-cmd", "false",  # exits non-zero; must be fail-soft
    ])
    assert code == wd.EXIT_CRASHED
    assert capsys.readouterr().out.strip().startswith("watchdog: CRASHED")
