"""Unit tests for scripts/train_watchdog.py pure decision logic (no live PIDs)."""
from __future__ import annotations

import json
import os
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
    pause_owner_pid: int | None = None,
    pause_owner_alive: bool | None = None,
    pause_age_minutes: float | None = None,
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
        pause_owner_pid=pause_owner_pid,
        pause_owner_alive=pause_owner_alive,
        pause_age_minutes=pause_age_minutes,
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


def test_main_cli_stopped_when_the_marker_exists(tmp_path: Path, capsys) -> None:
    """--stop-marker must REACH the verdict, not merely be parsed and dropped.

    Paired with test_main_cli_crashed: identical arguments except that this
    marker file EXISTS. Without the pair, `main()` could hardcode the default
    marker path and every CLI test would still pass — the flag would be
    accepted and silently ignored, which is this codebase's signature defect.
    """
    marker = tmp_path / "intentional_stop"
    marker.write_text("")
    pidfile = tmp_path / "missing.pid"
    code = wd.main([
        "--pidfile", str(pidfile),
        "--work-dir", str(tmp_path / "run"),
        "--state", str(tmp_path / "state.json"),
        "--stop-marker", str(marker),
        "--stall-minutes", "90",
    ])
    assert code == wd.EXIT_STOPPED
    out = capsys.readouterr().out.strip()
    assert out.startswith("watchdog: STOPPED")
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


# ── PAUSE-ABANDONED: the state that makes auto-recovery reachable ────────────
# ⚑ WHY THIS EXISTS. `decide()` returns PAUSED-HELD whenever the loop is flat
# AND a pause.txt is present, and STALLED requires `pause_txt is None`;
# `watchdog_loop.sh` recovers only on STALLED. So while ANY marker is held the
# recovery branch -- and `recover_stall.sh`'s `rm -f ... pause.txt` with it --
# is unreachable BY CONSTRUCTION. scripts/pause_window.sh makes a held marker a
# nightly event, so a window that dies holding one parks production until a
# human notices. The marker discriminates: pause_window.sh writes `pid=`,
# graceful_restart.py writes prose.

_OWNED = "pause_window.sh pid=4242 started=2026-08-09T05:03:54-04:00\njob=bash x\n"
_OPERATOR = "graceful restart in progress\n"


def _paused_snap(
    *,
    pid: int | None = 1234,
    pid_alive: bool = True,
    intentional_stop: bool = False,
    rows: int = 42,
    rows_prev: int | None = 42,
    minutes_flat: float = 15.0,
    pause_owner_pid: int | None = None,
    pause_owner_alive: bool | None = None,
    pause_age_minutes: float | None = None,
):
    """A flat loop with a marker present -- the PAUSED-HELD baseline.

    Spelled out rather than `**kw` onto `_snap`: a dict of mixed value types
    erases every parameter type on the way through, which is 11 basedpyright
    errors and, worse, means a typo'd kwarg name would be accepted here.
    """
    return _snap(
        pid=pid,
        pid_alive=pid_alive,
        intentional_stop=intentional_stop,
        pause_txt="runs/pbt2_small/tune/pause.txt",
        rows=rows,
        rows_prev=rows_prev,
        minutes_flat=minutes_flat,
        pause_owner_pid=pause_owner_pid,
        pause_owner_alive=pause_owner_alive,
        pause_age_minutes=pause_age_minutes,
    )


def test_a_held_marker_whose_owner_is_gone_is_recoverable() -> None:
    """The failure this state exists for: the wrapper died holding the marker."""
    v = wd.decide(
        _paused_snap(pause_owner_pid=4242, pause_owner_alive=False, pause_age_minutes=3.0),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_PAUSE_ABANDONED
    assert v.exit_code == wd.EXIT_PAUSE_ABANDONED
    assert "pause_abandoned=owner_pid_4242_is_gone" in v.format_line()


def test_a_held_marker_whose_owner_is_ALIVE_is_left_alone() -> None:
    """⚑ THE NEGATIVE CONTROL. The nightly window is exactly this state, and
    clearing it would resume training into the arena it is running."""
    v = wd.decide(
        _paused_snap(pause_owner_pid=4242, pause_owner_alive=True, pause_age_minutes=12.0),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_PAUSED_HELD
    assert v.exit_code == wd.EXIT_PAUSED_HELD


def test_an_owned_marker_held_past_the_bound_is_recoverable() -> None:
    """The owner can be alive and wedged. One window is bounded by the ack wait
    (30 min) plus BUDGET_MIN=90, so 180 is 50% headroom, not a guess."""
    v = wd.decide(
        _paused_snap(pause_owner_pid=4242, pause_owner_alive=True, pause_age_minutes=181.0),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_PAUSE_ABANDONED
    assert "pause_abandoned=held_181min_over_180" in v.format_line()


def test_an_OPERATORS_marker_is_never_abandoned_however_old() -> None:
    """⚑ THE ONE THAT MUST NOT REGRESS. `graceful_restart.py` writes no `pid=`,
    and an operator's pause is allowed to outlast any bound we could pick --
    clearing it resumes the run they deliberately parked. The gate is therefore
    "the marker names its owner", never the age.
    """
    v = wd.decide(
        _paused_snap(pause_owner_pid=None, pause_owner_alive=None, pause_age_minutes=10_000.0),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_PAUSED_HELD, (
        "an unowned marker was judged abandoned: the watchdog would delete an "
        "operator's pause"
    )


def test_an_abandoned_marker_over_a_GROWING_loop_is_still_OK() -> None:
    """Flatness is the whole reason PAUSED-HELD is not reported at every poll:
    a marker set moments ago over a loop still writing rows is not a problem."""
    v = wd.decide(
        _paused_snap(rows=43, rows_prev=42, pause_owner_pid=4242, pause_owner_alive=False),
        stall_minutes=90.0,
    )
    assert v.state == wd.STATE_OK


def test_a_dead_pid_still_outranks_an_abandoned_pause() -> None:
    """The dead-PID split (#371) runs FIRST: a crashed trainer with a stale
    pause marker is CRASHED, not a pause to be tidied up. Clearing the marker
    there would resume nothing and hide the outage behind a housekeeping line.
    """
    crashed = wd.decide(
        _paused_snap(pid=None, pid_alive=False, pause_owner_pid=1, pause_owner_alive=False),
        stall_minutes=90.0,
    )
    stopped = wd.decide(
        _paused_snap(
            pid=None, pid_alive=False, intentional_stop=True,
            pause_owner_pid=1, pause_owner_alive=False,
        ),
        stall_minutes=90.0,
    )
    assert crashed.state == wd.STATE_CRASHED
    assert stopped.state == wd.STATE_STOPPED


def test_parse_pause_marker_reads_pause_window_sh_format() -> None:
    """Against the literal bytes scripts/pause_window.sh writes."""
    started = "2026-08-09T05:03:54"
    import datetime as dt
    now = dt.datetime.fromisoformat(started).timestamp() + 600.0
    pid, age = wd.parse_pause_marker(
        f"pause_window.sh pid=4242 started={started}\njob=bash scripts/daily_gate_ratchet.sh\n",
        mtime=None, now=now,
    )
    assert pid == 4242
    assert age is not None
    assert abs(age - 10.0) < 0.1


def test_parse_pause_marker_rejects_the_operators_marker() -> None:
    pid, age = wd.parse_pause_marker(_OPERATOR, mtime=1000.0, now=99_000.0)
    assert pid is None, "an operator's marker was given an owner"
    assert age is None, (
        "an unowned marker was given an age; nothing downstream may act on it, "
        "so reporting one is an invitation to"
    )


def test_parse_pause_marker_falls_back_to_mtime_without_started() -> None:
    pid, age = wd.parse_pause_marker("pause_window.sh pid=7\n", mtime=1000.0, now=1600.0)
    assert pid == 7
    assert age == 10.0


def test_parse_pause_marker_survives_a_corrupt_started_field() -> None:
    pid, age = wd.parse_pause_marker(
        "pause_window.sh pid=7 started=NOT-A-DATE\n", mtime=1000.0, now=1600.0,
    )
    assert (pid, age) == (7, 10.0)


def test_build_snapshot_reads_the_marker_and_checks_its_owner(tmp_path: Path) -> None:
    """End to end from real files: the parse, the liveness check, and the
    verdict, so a snapshot that never populates the fields cannot pass."""
    work = tmp_path / "runs" / "x"
    tune = work / "tune"
    (tune / "train_trial_a_00000").mkdir(parents=True)
    (tune / "train_trial_a_00000" / "progress.csv").write_text("h\n1\n2\n")
    # A pid that cannot be alive: our own children are reaped, so use a pid the
    # kernel will not have (kill -0 raises ProcessLookupError).
    dead = 4_000_000
    (tune / "pause.txt").write_text(f"pause_window.sh pid={dead} started=2026-08-09T05:03:54\n")
    pidfile = tmp_path / "pid"
    pidfile.write_text(f"{os_getpid()}\n")
    state = tmp_path / "state.json"

    snap, new_state = wd.build_snapshot(
        pidfile=pidfile, work_dir=work, state_path=state,
        stop_marker=tmp_path / 'no_marker',
    )
    assert snap.pause_owner_pid == dead
    assert snap.pause_owner_alive is False
    wd.save_state(state, new_state)
    # Second pass: rows are unchanged, so the loop is flat and the verdict lands.
    snap2, _ = wd.build_snapshot(
        pidfile=pidfile, work_dir=work, state_path=state,
        stop_marker=tmp_path / 'no_marker',
    )
    v = wd.decide(snap2, stall_minutes=90.0)
    assert v.state == wd.STATE_PAUSE_ABANDONED, v.format_line()


def test_build_snapshot_leaves_an_unreadable_marker_as_paused_held(tmp_path: Path) -> None:
    """Fail-soft: a marker we cannot parse is reported, never cleared."""
    work = tmp_path / "runs" / "x"
    tune = work / "tune"
    (tune / "train_trial_a_00000").mkdir(parents=True)
    (tune / "train_trial_a_00000" / "progress.csv").write_text("h\n1\n")
    (tune / "pause.txt").write_text(_OPERATOR)
    pidfile = tmp_path / "pid"
    pidfile.write_text(f"{os_getpid()}\n")
    state = tmp_path / "state.json"

    _, new_state = wd.build_snapshot(
        pidfile=pidfile, work_dir=work, state_path=state,
        stop_marker=tmp_path / 'no_marker',
    )
    wd.save_state(state, new_state)
    snap2, _ = wd.build_snapshot(
        pidfile=pidfile, work_dir=work, state_path=state,
        stop_marker=tmp_path / 'no_marker',
    )
    assert snap2.pause_owner_pid is None
    assert wd.decide(snap2, stall_minutes=90.0).state == wd.STATE_PAUSED_HELD


# ── watchdog_loop.sh: the branch that actually CLEARS the marker ─────────────
# The verdict above is only worth having if something acts on it, and "acts on
# it" is a shell branch. These drive the real loop against a sandbox using the
# seams #371 already established (WATCHDOG_ROOT / WATCHDOG_MAX_ITERS /
# WATCHDOG_MARKER / WATCHDOG_STATEF / WATCHDOG_LOGF / WATCHDOG_ALERTF /
# WATCHDOG_LAST_ALERT_F / WATCHDOG_RECOVER_STAMP) plus TRAIN_PIDFILE, so this
# can run at all without reading or writing the LIVE run's /tmp state.

LOOP_SH = Path(__file__).resolve().parents[1] / "scripts" / "watchdog_loop.sh"
WATCHDOG_PY = Path(__file__).resolve().parents[1] / "scripts" / "train_watchdog.py"

# A pid the kernel will not have. `kill -0` on it raises ProcessLookupError,
# which is what makes the marker's owner "gone".
DEAD_PID = 4_000_000


def _loop_sandbox(tmp_path: Path, marker_text: str):
    import shutil

    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(LOOP_SH, root / "scripts" / LOOP_SH.name)
    shutil.copy(WATCHDOG_PY, root / "scripts" / WATCHDOG_PY.name)
    # Present so a wrong branch is a RECORDED wrong branch rather than a 127.
    recovered = root / "recover_stall_ran"
    stall = root / "scripts" / "recover_stall.sh"
    stall.write_text(f"#!/bin/bash\ntouch {recovered}\n")
    stall.chmod(0o755)

    tune = root / "runs" / "pbt2_small" / "tune"
    (tune / "train_trial_a_00000").mkdir(parents=True)
    (tune / "train_trial_a_00000" / "progress.csv").write_text("h\n1\n2\n")
    marker = tune / "pause.txt"
    marker.write_text(marker_text)
    return root, marker, recovered


def _run_loop_once(root: Path, tmp_path: Path, **env_extra: str):
    """Two passes: the first arms the flatness clock, the second is flat."""
    import subprocess

    alive = subprocess.Popen(["sleep", "600"])
    pidfile = root / "trainer.pid"
    pidfile.write_text(f"{alive.pid}\n")
    logf = tmp_path / "watchdog.log"
    alertf = tmp_path / "watchdog_alerts.log"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "WATCHDOG_ROOT": str(root),
        "WATCHDOG_MAX_ITERS": "1",
        "WATCHDOG_EVERY": "1",
        "WATCHDOG_STATEF": str(tmp_path / "wd_state.json"),
        "WATCHDOG_MARKER": str(tmp_path / "no_such_stop_marker"),
        "WATCHDOG_LOGF": str(logf),
        "WATCHDOG_ALERTF": str(alertf),
        "WATCHDOG_LAST_ALERT_F": str(tmp_path / "last_alert"),
        "WATCHDOG_LAST_ESCALATE_F": str(tmp_path / "last_escalate"),
        "WATCHDOG_RECOVER_STAMP": str(tmp_path / "recover_stamp"),
        "TRAIN_PIDFILE": str(pidfile),
        "PYTHONDONTWRITEBYTECODE": "1",
        **env_extra,
    }
    try:
        out = ""
        for _ in range(2):
            r = subprocess.run(
                ["bash", "scripts/watchdog_loop.sh"],
                cwd=str(root), capture_output=True, text=True,
                check=False, timeout=120, env=env,
            )
            out += r.stdout + r.stderr
    finally:
        alive.terminate()
        alive.wait()
    return (
        out
        + (logf.read_text() if logf.exists() else "")
        + (alertf.read_text() if alertf.exists() else "")
    )


def test_the_loop_clears_a_marker_whose_owner_is_gone(tmp_path: Path) -> None:
    """⚑ THE WHOLE POINT. Before this branch existed, a held marker made the
    STALLED verdict unreachable, so nothing in the stack could ever remove one
    -- `recover_stall.sh:31` is the only line that does, and it runs only on
    exit 3, which a held marker forbids.
    """
    root, marker, recovered = _loop_sandbox(
        tmp_path, f"pause_window.sh pid={DEAD_PID} started=2026-08-09T05:03:54\njob=x\n",
    )
    out = _run_loop_once(root, tmp_path)

    assert not marker.exists(), f"the abandoned marker survived:\n{out}"
    assert "CLEARING ABANDONED PAUSE MARKER" in out, out
    assert not recovered.exists(), (
        "it force-recovered a healthy stack: nothing was wedged, one file needed "
        "deleting, and recover_stall.sh SIGKILLs the whole run"
    )


def test_the_loop_does_NOT_clear_an_operators_marker(tmp_path: Path) -> None:
    """⚑ THE NEGATIVE CONTROL, and the one that must never regress:
    `graceful_restart.py` writes prose with no `pid=`, and deleting it resumes
    a run the operator deliberately parked."""
    root, marker, recovered = _loop_sandbox(tmp_path, "graceful restart in progress\n")
    out = _run_loop_once(root, tmp_path)

    assert marker.exists(), f"an operator's pause was deleted by the watchdog:\n{out}"
    assert "PAUSED-HELD" in out, out
    assert "CLEARING" not in out, out
    assert not recovered.exists()


def test_the_loop_does_NOT_clear_a_marker_whose_owner_is_alive(tmp_path: Path) -> None:
    """The nightly window itself. Clearing it resumes training into the arena.

    `started=` is NOW, not a literal: the first draft of this test hardcoded a
    timestamp and the age bound fired on it instead, which would have passed
    the "marker survives" assertion for the wrong reason on the day it was
    written and failed the day after.
    """
    import datetime as dt

    now = dt.datetime.now().isoformat(timespec="seconds")
    root, marker, _ = _loop_sandbox(
        tmp_path, f"pause_window.sh pid={os_getpid()} started={now}\n",
    )
    out = _run_loop_once(root, tmp_path)

    assert marker.exists(), f"the LIVE pause window's marker was deleted:\n{out}"
    assert "PAUSED-HELD" in out, out


def test_auto_recover_off_disables_the_clearing_too(tmp_path: Path) -> None:
    """One switch, both recoveries: an operator who turned recovery off must not
    find a different recovery still running."""
    root, marker, _ = _loop_sandbox(
        tmp_path, f"pause_window.sh pid={DEAD_PID} started=2026-08-09T05:03:54\n",
    )
    out = _run_loop_once(root, tmp_path, WATCHDOG_AUTO_RECOVER="0")

    assert marker.exists(), f"WATCHDOG_AUTO_RECOVER=0 did not disable it:\n{out}"
    assert "PAUSE-ABANDONED" in out, f"the verdict must still be REPORTED:\n{out}"


def test_the_intentional_stop_marker_suppresses_the_clearing(tmp_path: Path) -> None:
    root, marker, _ = _loop_sandbox(
        tmp_path, f"pause_window.sh pid={DEAD_PID} started=2026-08-09T05:03:54\n",
    )
    stop = tmp_path / "stop_marker"
    stop.write_text("")
    out = _run_loop_once(root, tmp_path, WATCHDOG_MARKER=str(stop))

    assert marker.exists(), f"a deliberate stop did not suppress the clearing:\n{out}"


def test_a_CRASHED_trainer_does_not_get_its_pause_marker_tidied_away(tmp_path: Path) -> None:
    """⚑ THE EXIT-CODE COLLISION, pinned where it would actually have hurt.

    This branch first numbered PAUSE-ABANDONED 5, which #371 had already taken
    for CRASHED. With both at 5, a trainer that DIED while a pause window held
    the marker would reach the clearing branch: the marker would be deleted,
    the alert would read like routine housekeeping, and the outage — the exact
    thing #371 exists to make legible — would be hidden behind it.

    Nothing about the numbers is asserted here. The observable is: dead
    trainer + an owned marker => the marker is still there afterwards.
    """
    root, marker, recovered = _loop_sandbox(
        tmp_path, f"pause_window.sh pid={DEAD_PID} started=2026-08-09T05:03:54\n",
    )
    # A pidfile naming a pid that is not alive: the trainer crashed, and the
    # intentional-stop marker is absent (`_run_loop_once` points it at a path
    # that does not exist), so the verdict is CRASHED rather than STOPPED.
    import subprocess

    pidfile = root / "trainer.pid"
    pidfile.write_text(f"{DEAD_PID}\n")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "WATCHDOG_ROOT": str(root),
        "WATCHDOG_MAX_ITERS": "1",
        "WATCHDOG_EVERY": "1",
        "WATCHDOG_STATEF": str(tmp_path / "wd_state.json"),
        "WATCHDOG_MARKER": str(tmp_path / "no_such_stop_marker"),
        "WATCHDOG_LOGF": str(tmp_path / "watchdog.log"),
        "WATCHDOG_ALERTF": str(tmp_path / "watchdog_alerts.log"),
        "WATCHDOG_LAST_ALERT_F": str(tmp_path / "last_alert"),
        "WATCHDOG_LAST_ESCALATE_F": str(tmp_path / "last_escalate"),
        "WATCHDOG_RECOVER_STAMP": str(tmp_path / "recover_stamp"),
        "TRAIN_PIDFILE": str(pidfile),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    out = ""
    for _ in range(2):
        r = subprocess.run(
            ["bash", "scripts/watchdog_loop.sh"], cwd=str(root),
            capture_output=True, text=True, check=False, timeout=120, env=env,
        )
        out += r.stdout + r.stderr
    out += (tmp_path / "watchdog.log").read_text() if (tmp_path / "watchdog.log").exists() else ""

    assert "CRASHED" in out, f"the dead trainer must report CRASHED:\n{out}"
    assert marker.exists(), (
        f"a CRASHED trainer's pause marker was cleared as though the window had "
        f"been abandoned; the outage now reads as housekeeping:\n{out}"
    )
    assert "CLEARING ABANDONED PAUSE MARKER" not in out, out
    assert not recovered.exists()
