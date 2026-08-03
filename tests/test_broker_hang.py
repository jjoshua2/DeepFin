"""Unit tests for broker hang self-abort (torch-free; no GPU / no os._exit).

The arming rule changed on 2026-08-03 (inference audit I3). It used to be
"inert until the first forward SUCCEEDS", which made the watchdog unable to fire
on the scenario its own docstring names: a broker booting into an already-wedged
CUDA/WSL2-dxg context hangs on forward #1, never arms, never aborts. Arming now
only selects WHICH threshold applies — a longer ``boot_threshold_s`` before the
first success, the normal one after — so the cold-compile window still cannot
false-fire but "forever" is no longer an option.
"""
from __future__ import annotations

import pytest

from chess_anti_engine.broker_hang import (
    DEFAULT_BOOT_HANG_ABORT_S,
    HANG_ABORT_EXIT_CODE,
    BrokerHangWatchdog,
    resolve_boot_hang_abort_seconds,
    resolve_hang_abort_seconds,
    should_hang_abort,
)


def test_resolve_boot_hang_abort_seconds_env_overrides_default() -> None:
    assert resolve_boot_hang_abort_seconds(300.0, env={}) == DEFAULT_BOOT_HANG_ABORT_S
    assert resolve_boot_hang_abort_seconds(300.0, 900.0, env={}) == 900.0
    assert resolve_boot_hang_abort_seconds(
        300.0, env={"CAE_BROKER_BOOT_HANG_ABORT_S": "60"},
    ) == 60.0
    # Empty / whitespace does not override, matching the steady-state resolver.
    assert resolve_boot_hang_abort_seconds(
        300.0, 900.0, env={"CAE_BROKER_BOOT_HANG_ABORT_S": "  "},
    ) == 900.0
    # 0 disables the cold-start window explicitly.
    assert resolve_boot_hang_abort_seconds(
        300.0, env={"CAE_BROKER_BOOT_HANG_ABORT_S": "0"},
    ) == 0.0


def test_disabling_the_watchdog_also_disables_the_cold_start_window() -> None:
    """`--hang-abort-seconds 0` must turn the whole feature off (PR #322 review).

    Arming from process start (audit I3) resolved the boot window independently,
    so a broker told NOT to hang-abort still started the thread and still
    os._exit(42)-ed at 1800s while unarmed. That is the escape hatch someone
    reaches for precisely when the watchdog is misfiring.
    """
    for env in ({}, {"CAE_BROKER_BOOT_HANG_ABORT_S": "60"}):
        assert resolve_boot_hang_abort_seconds(0.0, env=env) == 0.0
        assert resolve_boot_hang_abort_seconds(-1.0, env=env) == 0.0
    # A disabled pair really is inert on the object, not merely zero-valued.
    clock = [0.0]
    exits: list[int] = []
    wd = BrokerHangWatchdog(
        threshold_s=0.0,
        boot_threshold_s=resolve_boot_hang_abort_seconds(0.0, env={}),
        exit_fn=exits.append,
        clock=lambda: clock[0],
    )
    wd.start()
    assert wd._thread is None, "start() must be a no-op when both windows are off"
    wd.mark_forward_start(1)
    clock[0] = 10_000.0
    assert not wd.check_once()
    assert exits == []


def test_should_hang_abort_decision_matrix() -> None:
    # No in-flight work.
    assert not should_hang_abort(
        armed=True, oldest_inflight_age_s=None, threshold_s=300.0,
    )
    # Under threshold.
    assert not should_hang_abort(
        armed=True, oldest_inflight_age_s=299.9, threshold_s=300.0,
    )
    # At / over threshold while armed + in flight.
    assert should_hang_abort(
        armed=True, oldest_inflight_age_s=300.0, threshold_s=300.0,
    )
    assert should_hang_abort(
        armed=True, oldest_inflight_age_s=301.0, threshold_s=300.0,
    )
    # Disabled (threshold 0) — feature off.
    assert not should_hang_abort(
        armed=True, oldest_inflight_age_s=9999.0, threshold_s=0.0,
    )
    assert not should_hang_abort(
        armed=True, oldest_inflight_age_s=9999.0, threshold_s=-1.0,
    )


def test_unarmed_uses_the_boot_threshold_and_can_still_abort() -> None:
    """Audit I3-H1: the boot-into-wedged-dxg case must be reachable.

    Pre-fix this whole family returned False no matter the age.
    """
    # Inside the cold-start window: still no abort, so a slow first compile is
    # as safe as it was before.
    assert not should_hang_abort(
        armed=False, oldest_inflight_age_s=600.0,
        threshold_s=300.0, boot_threshold_s=1800.0,
    )
    # Past the cold-start window: abort, even though no forward ever succeeded.
    assert should_hang_abort(
        armed=False, oldest_inflight_age_s=1800.0,
        threshold_s=300.0, boot_threshold_s=1800.0,
    )
    # A disabled boot window keeps the old "never fire before arming" shape.
    assert not should_hang_abort(
        armed=False, oldest_inflight_age_s=9e9,
        threshold_s=300.0, boot_threshold_s=0.0,
    )
    # Once armed the boot threshold is irrelevant.
    assert should_hang_abort(
        armed=True, oldest_inflight_age_s=301.0,
        threshold_s=300.0, boot_threshold_s=1800.0,
    )


def test_completed_batch_clears_inflight_and_arms() -> None:
    clock = [100.0]
    exits: list[int] = []
    wd = BrokerHangWatchdog(
        threshold_s=10.0,
        boot_threshold_s=1000.0,
        poll_interval_s=1000.0,
        exit_fn=exits.append,
        clock=lambda: clock[0],
    )
    assert not wd.armed
    assert wd.oldest_inflight_age_s() is None

    wd.mark_forward_start(batch_size=64)
    clock[0] = 105.0
    assert wd.oldest_inflight_age_s() == pytest.approx(5.0)
    # Unarmed and inside the cold window: no abort even though age > threshold.
    clock[0] = 200.0
    assert not wd.check_once()
    assert exits == []

    wd.mark_forward_done(success=True)
    assert wd.armed
    assert wd.oldest_inflight_age_s() is None
    # Idle after completion: still no abort.
    clock[0] = 500.0
    assert not wd.check_once()
    assert exits == []


def test_watchdog_aborts_a_wedged_first_forward_past_the_boot_threshold() -> None:
    """The end-to-end I3-H1 shape on the object, not just the pure function."""
    clock = [0.0]
    exits: list[int] = []
    wd = BrokerHangWatchdog(
        threshold_s=10.0,
        boot_threshold_s=100.0,
        poll_interval_s=1000.0,
        exit_fn=exits.append,
        clock=lambda: clock[0],
    )
    wd.mark_forward_start(batch_size=512)
    clock[0] = 99.0
    assert not wd.check_once()
    clock[0] = 100.0
    assert wd.check_once()
    assert exits == [HANG_ABORT_EXIT_CODE]
    assert not wd.armed


def test_stage_is_watched_and_does_not_arm() -> None:
    """Audit I3, second hole: model load / compile now sit inside the window.

    ``_ensure_model`` (torch.load, .to(device), AOT load_constants,
    torch.compile) used to run entirely outside the instrumented region.
    """
    clock = [0.0]
    exits: list[int] = []
    wd = BrokerHangWatchdog(
        threshold_s=10.0,
        boot_threshold_s=50.0,
        poll_interval_s=1000.0,
        exit_fn=exits.append,
        clock=lambda: clock[0],
    )
    with wd.stage("ensure_model"):
        clock[0] = 49.0
        assert wd.oldest_inflight_age_s() == pytest.approx(49.0)
        assert not wd.check_once()
        clock[0] = 50.0
        assert wd.check_once()
    assert exits == [HANG_ABORT_EXIT_CODE]
    # A stage completing proves nothing about the CUDA context, so it must not
    # arm the shorter steady-state threshold.
    assert not wd.armed
    assert wd.oldest_inflight_age_s() is None


def test_oldest_inflight_is_actually_the_oldest() -> None:
    """Audit I3, third hole: completing a NEWER item erased an OLDER one's age.

    Not reachable from today's single-threaded serve loops; this pins it before
    anyone pipelines the broker forward.
    """
    clock = [100.0]
    exits: list[int] = []
    wd = BrokerHangWatchdog(
        threshold_s=10.0,
        boot_threshold_s=10.0,
        poll_interval_s=1000.0,
        exit_fn=exits.append,
        clock=lambda: clock[0],
    )
    tok_a = wd.mark_forward_start(batch_size=512)
    clock[0] = 101.0
    tok_b = wd.mark_forward_start(batch_size=256)
    clock[0] = 102.0
    wd.mark_forward_done(success=True, token=tok_b)
    # A still in flight since t=100 -> age 2.0, not None.
    assert wd.oldest_inflight_age_s() == pytest.approx(2.0)
    clock[0] = 100_000.0
    assert wd.check_once()
    assert exits == [HANG_ABORT_EXIT_CODE]
    wd.mark_forward_done(success=True, token=tok_a)


def test_mark_forward_done_without_a_token_clears_the_oldest_forward() -> None:
    """The serial serve loops still call done() with no token — keep that sane."""
    clock = [0.0]
    wd = BrokerHangWatchdog(
        threshold_s=10.0, poll_interval_s=1000.0,
        exit_fn=lambda _c: None, clock=lambda: clock[0],
    )
    wd.mark_forward_start(1)
    clock[0] = 5.0
    wd.mark_forward_done(success=True)
    assert wd.oldest_inflight_age_s() is None


def test_hang_watchdog_aborts_when_armed_inflight_exceeds_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = [0.0]
    exits: list[int] = []
    wd = BrokerHangWatchdog(
        threshold_s=10.0,
        poll_interval_s=1000.0,
        exit_fn=exits.append,
        clock=lambda: clock[0],
    )
    # Arm via a successful batch.
    wd.mark_forward_start(1)
    clock[0] = 1.0
    wd.mark_forward_done(success=True)
    assert wd.armed

    # Next batch hangs past threshold.
    wd.mark_forward_start(batch_size=128)
    clock[0] = 1.0 + 10.0
    with caplog.at_level("CRITICAL", logger="chess_anti_engine.broker_hang"):
        assert wd.check_once()
    assert exits == [HANG_ABORT_EXIT_CODE]
    # Exactly one critical line with the required hint.
    critical = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert len(critical) == 1
    msg = critical[0].getMessage()
    assert "batch_size=128" in msg
    assert "10.0s" in msg or "10.0" in msg
    assert "GPU context likely dead" in msg
    assert "WSL2 dxg vmbus" in msg
    # Second check must not re-log / re-exit (exit_fn is injectable mock).
    assert wd.check_once()
    assert exits == [HANG_ABORT_EXIT_CODE]
    assert len([r for r in caplog.records if r.levelname == "CRITICAL"]) == 1


def test_hang_watchdog_failed_batch_clears_inflight_without_arming() -> None:
    clock = [0.0]
    exits: list[int] = []
    wd = BrokerHangWatchdog(
        threshold_s=5.0,
        boot_threshold_s=1000.0,
        exit_fn=exits.append,
        clock=lambda: clock[0],
    )
    wd.mark_forward_start(4)
    clock[0] = 1.0
    wd.mark_forward_done(success=False)
    assert not wd.armed
    assert wd.oldest_inflight_age_s() is None
    # A hang before arming is judged against the boot threshold, not the
    # steady-state one: 100s is a long forward but a normal cold start.
    wd.mark_forward_start(4)
    clock[0] = 100.0
    assert not wd.check_once()
    assert exits == []


def test_hang_watchdog_disabled_threshold_never_aborts() -> None:
    clock = [0.0]
    exits: list[int] = []
    wd = BrokerHangWatchdog(
        threshold_s=0.0,
        exit_fn=exits.append,
        clock=lambda: clock[0],
    )
    # start() is a no-op when disabled; exercise the path anyway.
    wd.start()
    wd.mark_forward_start(1)
    clock[0] = 1.0
    wd.mark_forward_done(success=True)
    wd.mark_forward_start(8)
    clock[0] = 10_000.0
    assert not wd.check_once()
    assert exits == []
    wd.stop()


def test_resolve_hang_abort_seconds_env_overrides_cli() -> None:
    assert resolve_hang_abort_seconds(300.0, env={}) == 300.0
    assert resolve_hang_abort_seconds(120.0, env={}) == 120.0
    assert resolve_hang_abort_seconds(300.0, env={"CAE_BROKER_HANG_ABORT_S": "90"}) == 90.0
    assert resolve_hang_abort_seconds(300.0, env={"CAE_BROKER_HANG_ABORT_S": "0"}) == 0.0
    # Empty / whitespace env value does not override.
    assert resolve_hang_abort_seconds(300.0, env={"CAE_BROKER_HANG_ABORT_S": ""}) == 300.0
    assert resolve_hang_abort_seconds(300.0, env={"CAE_BROKER_HANG_ABORT_S": "  "}) == 300.0
