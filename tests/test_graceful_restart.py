from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from chess_anti_engine.tune.trainable_config_ops import (
    _clear_pause_acks,
    _pause_ack_name,
    _wait_if_paused,
    _write_pause_acks,
)
from scripts.graceful_restart import (
    _active_trials,
    _pause_ack_files,
    _required_paused_count,
    _trial_is_acked,
)


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
    assert _required_paused_count(4, 1) == 1
    assert _required_paused_count(4, 2) == 2
    assert _required_paused_count(1, 3) == 1


def test_pause_ack_detected_without_progress_row_growth(tmp_path: Path) -> None:
    """The boundary-hold blind spot: the trial holds before any new progress.csv
    row, so the row heuristic can't fire — the ack it drops in the tune root must
    still be detected (this is the bug that left graceful_restart idle for ~1.5h)."""
    trial = tmp_path / "train_trial_5fac4_00000_0_lr=0.0003"
    _write_progress(trial / "progress.csv")
    csv = trial / "progress.csv"
    # Trial dropped its ack next to the persistent tune-root pause marker.
    (tmp_path / _pause_ack_name("5fac4_00000")).write_text("trial=5fac4_00000 next_iter=383\n")
    acks = _pause_ack_files(tmp_path, [csv], since_ts=0.0)
    assert acks, "ack in the tune root must be discovered"
    assert _trial_is_acked(csv, acks)


def test_pause_ack_in_trial_dir_detected(tmp_path: Path) -> None:
    trial = tmp_path / "train_trial_abc_00000_0"
    _write_progress(trial / "progress.csv")
    csv = trial / "progress.csv"
    (trial / _pause_ack_name("abc_00000")).write_text("x")
    assert _trial_is_acked(csv, _pause_ack_files(tmp_path, [csv], since_ts=0.0))


def test_pause_ack_generic_trial_id_does_not_match_all(tmp_path: Path) -> None:
    """A degenerate trial_id fallback ('trial', from `_ctx.get_trial_id() or "trial"`)
    must NOT match every `train_trial_*` dir via a loose substring."""
    trial = tmp_path / "train_trial_5fac4_00000_0_lr=0.0003"
    _write_progress(trial / "progress.csv")
    csv = trial / "progress.csv"
    (tmp_path / _pause_ack_name("trial")).write_text("x")
    assert not _trial_is_acked(csv, _pause_ack_files(tmp_path, [csv], since_ts=0.0))


def test_pause_ack_no_sibling_prefix_collision(tmp_path: Path) -> None:
    """One trial's ack must not be attributed to a sibling whose id it prefixes."""
    a = tmp_path / "train_trial_run_1_0"
    b = tmp_path / "train_trial_run_10_0"
    _write_progress(a / "progress.csv")
    _write_progress(b / "progress.csv")
    csv_a, csv_b = a / "progress.csv", b / "progress.csv"
    (tmp_path / _pause_ack_name("run_1")).write_text("x")  # only trial A paused
    acks = _pause_ack_files(tmp_path, [csv_a, csv_b], since_ts=0.0)
    assert _trial_is_acked(csv_a, acks)
    assert not _trial_is_acked(csv_b, acks)


def test_pause_ack_stale_is_ignored(tmp_path: Path) -> None:
    """An ack left by a crashed prior run (older than the pause request) must not
    be mistaken for a fresh pause."""
    trial = tmp_path / "train_trial_xyz_00000_0"
    _write_progress(trial / "progress.csv")
    csv = trial / "progress.csv"
    (tmp_path / _pause_ack_name("xyz_00000")).write_text("old")
    assert _pause_ack_files(tmp_path, [csv], since_ts=time.time() + 10_000) == []


def test_write_and_clear_pause_acks_roundtrip(tmp_path: Path) -> None:
    marker = tmp_path / "pause.txt"
    marker.write_text("pause")
    written = _write_pause_acks([marker], trial_id="t_0", iteration=5)
    assert written == [tmp_path / _pause_ack_name("t_0")]
    assert written[0].exists()
    _clear_pause_acks(written)
    assert not written[0].exists()


def test_wait_if_paused_writes_ack_while_held_and_clears_on_resume(tmp_path: Path) -> None:
    marker = tmp_path / "pause.txt"
    marker.write_text("pause")
    ack = tmp_path / _pause_ack_name("trial_held")
    done = threading.Event()

    def run() -> None:
        _wait_if_paused(
            pause_marker_paths=[marker], poll_seconds=1, trial_id="trial_held", iteration=7,
        )
        done.set()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    for _ in range(50):  # ack should appear promptly while the marker is present
        if ack.exists():
            break
        time.sleep(0.1)
    assert ack.exists(), "ack must be written while holding"
    assert not done.is_set(), "must still be holding while the marker is present"
    marker.unlink()  # resume
    th.join(timeout=5)
    assert done.is_set(), "trial must resume once the marker is gone"
    assert not ack.exists(), "ack must be cleared on resume"


def test_wait_if_paused_releases_the_cuda_cache_on_park(
    tmp_path: Path, monkeypatch,
) -> None:
    """⚑ MEASURED 2026-08-20: a parked trial held 32,059 of 32,607 MiB and the
    pause window's GPU job had 132 MiB to build in. The park exists so a
    window can run GPU work while the trial sleeps, so parking must hand back
    the allocator cache. Release happens exactly once, at park -- a poll that
    never sees a marker must not touch the allocator."""
    import torch

    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("ec"))
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: (30 * 2**30, 32 * 2**30),
    )

    marker = tmp_path / "pause.txt"
    marker.write_text("pause")
    done = threading.Event()

    def run() -> None:
        _wait_if_paused(
            pause_marker_paths=[marker], poll_seconds=1,
            trial_id="trial_cache", iteration=7,
        )
        done.set()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    for _ in range(50):
        if calls:
            break
        time.sleep(0.1)
    marker.unlink()
    th.join(timeout=5)
    assert done.is_set(), "trial must resume once the marker is gone"
    assert calls == ["ec"], (
        f"cache must be released exactly once, at park; saw {calls}"
    )

    calls.clear()
    _wait_if_paused(
        pause_marker_paths=[tmp_path / "absent.txt"], poll_seconds=1,
        trial_id="trial_cache", iteration=8,
    )
    assert calls == [], "the no-marker fast path must not touch the allocator"


def test_resume_preflight_reports_blockers_from_the_checker(monkeypatch) -> None:
    """A failing freshness check must surface its lines, not just an exit code."""
    import subprocess as _sp

    from scripts import graceful_restart as gr

    def fake_run(*args, **_kwargs):
        return _sp.CompletedProcess(
            args=args, returncode=1,
            stdout="C extension freshness check failed:\n  - _mcts_tree is older than _mcts_tree.c\n",
            stderr="",
        )

    monkeypatch.delenv("TRAIN_SKIP_C_EXT_CHECK", raising=False)
    monkeypatch.setattr(gr.subprocess, "run", fake_run)
    blockers = gr._resume_preflight()
    assert blockers, "a failing check must produce blockers"
    assert any("_mcts_tree" in b for b in blockers)


def test_resume_preflight_clean_when_check_passes(monkeypatch) -> None:
    import subprocess as _sp

    from scripts import graceful_restart as gr

    monkeypatch.delenv("TRAIN_SKIP_C_EXT_CHECK", raising=False)
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **_k: _sp.CompletedProcess(args=a, returncode=0, stdout="", stderr=""),
    )
    assert gr._resume_preflight() == []


def test_resume_preflight_honors_the_train_sh_skip_env(monkeypatch) -> None:
    """train.sh has an escape hatch; preflight must not be stricter than it."""
    from scripts import graceful_restart as gr

    monkeypatch.setenv("TRAIN_SKIP_C_EXT_CHECK", "1")
    monkeypatch.setattr(gr.subprocess, "run", lambda *_a, **_k: pytest.fail("must not run"))
    assert gr._resume_preflight() == []


def test_run_resume_retries_once_then_exits_nonzero(monkeypatch, capsys) -> None:
    """A failed resume means training is DOWN — it must be loud, not a traceback."""
    import subprocess as _sp

    from scripts import graceful_restart as gr

    calls: list[str] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return _sp.CompletedProcess(args=cmd, returncode=1)

    monkeypatch.setattr(gr.subprocess, "run", fake_run)
    monkeypatch.setattr(gr.time, "sleep", lambda _s: None)
    monkeypatch.setattr(gr, "_tuner_is_running", lambda: False)
    with pytest.raises(SystemExit) as exc:
        gr._run_resume("./scripts/train.sh restart", pause_targets=[Path("/tmp/pause.txt")])
    assert exc.value.code == 1
    assert len(calls) == 2, "must retry exactly once"
    out = capsys.readouterr().out
    assert "TRAINING IS DOWN" in out
    assert "/tmp/pause.txt" in out, "recovery steps must name the pause markers"


def test_run_resume_returns_on_success_without_retrying(monkeypatch) -> None:
    import subprocess as _sp

    from scripts import graceful_restart as gr

    calls: list[str] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return _sp.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(gr.subprocess, "run", fake_run)
    monkeypatch.setattr(gr.time, "sleep", lambda _s: None)
    monkeypatch.setattr(gr, "_tuner_is_running", lambda: True)
    gr._run_resume("./scripts/train.sh restart", pause_targets=[])
    assert len(calls) == 1


def test_preflight_argv_matches_train_sh() -> None:
    """The preflight must invoke the checker exactly as train.sh does.

    This is the test that would have caught the branch being cut from a stale
    main, where the preflight passed flags the checked-out checker did not yet
    accept — argparse exited 2 and every restart aborted.
    """
    from scripts.graceful_restart import _C_EXT_CHECK_ARGV, _REPO_ROOT

    train_sh = (_REPO_ROOT / "scripts" / "train.sh").read_text()
    invocation = next(
        (ln for ln in train_sh.splitlines() if "check_c_extensions_fresh.py" in ln), None
    )
    assert invocation is not None, "train.sh must still invoke the freshness checker"
    # train.sh line-continues the flags; join the block for comparison.
    idx = train_sh.splitlines().index(invocation)
    block = " ".join(
        ln.strip().rstrip("\\").strip() for ln in train_sh.splitlines()[idx:idx + 2]
    )
    for token in _C_EXT_CHECK_ARGV:
        assert token in block, f"{token!r} is in the preflight but not in train.sh"
    for flag in ("--min-gcc-major", "--require-production-recipe", "--quiet"):
        if flag in block:
            assert flag in _C_EXT_CHECK_ARGV, f"{flag!r} is in train.sh but not the preflight"


def test_preflight_runs_for_real_without_argparse_errors() -> None:
    """A real subprocess run: catches flag drift even if argv strings match."""
    from scripts.graceful_restart import _resume_preflight

    blockers = _resume_preflight()
    joined = " ".join(blockers).lower()
    assert "unrecognized arguments" not in joined, blockers
    assert "usage:" not in joined, blockers


def test_preflight_failure_leaves_the_live_run_untouched(tmp_path: Path, monkeypatch) -> None:
    """The whole point of the PR: abort BEFORE any pause marker is written."""
    from scripts import graceful_restart as gr

    trial = tmp_path / "train_trial_x_00000_0"
    _write_progress(trial / "progress.csv")

    monkeypatch.setattr(gr, "_resume_preflight", lambda: ["stale extension"])
    monkeypatch.setattr(gr.sys, "argv", ["graceful_restart.py", "--tune-dir", str(tmp_path)])
    killed: list[int] = []
    monkeypatch.setattr(gr.os, "kill", lambda pid, _sig: killed.append(pid))

    with pytest.raises(SystemExit) as exc:
        gr.main()

    assert exc.value.code == 1
    assert not (tmp_path / "pause.txt").exists(), "must not pause the live trial"
    assert not (trial / "pause.txt").exists(), "must not pause the live trial"
    assert killed == [], "must not signal the tuner"


def test_resume_that_exits_zero_but_dies_is_treated_as_failure(monkeypatch, capsys) -> None:
    """train.sh start returns 0 once the PID file is written; if the trainer
    then dies, exiting 0 here would leave training silently down."""
    import subprocess as _sp

    from scripts import graceful_restart as gr

    calls: list[str] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return _sp.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(gr.subprocess, "run", fake_run)
    monkeypatch.setattr(gr.time, "sleep", lambda _s: None)
    monkeypatch.setattr(gr, "_tuner_is_running", lambda: False)  # died after start
    with pytest.raises(SystemExit) as exc:
        gr._run_resume("./scripts/train.sh restart", pause_targets=[])
    assert exc.value.code == 1
    assert "TRAINING IS DOWN" in capsys.readouterr().out


def test_retry_is_skipped_when_a_trainer_is_already_running(monkeypatch) -> None:
    """Never race a second trainer onto the same tune dir and GPU."""
    import subprocess as _sp

    from scripts import graceful_restart as gr

    calls: list[str] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return _sp.CompletedProcess(args=cmd, returncode=1)

    # Attempt 1 exits nonzero but left a trainer alive (start spawned it, then
    # failed before writing the PID file) — the retry gate must see that.
    monkeypatch.setattr(gr.subprocess, "run", fake_run)
    monkeypatch.setattr(gr.time, "sleep", lambda _s: None)
    monkeypatch.setattr(gr, "_tuner_is_running", lambda: True)
    gr._run_resume("./scripts/train.sh restart", pause_targets=[])
    assert len(calls) == 1, "must not spawn a second trainer"
