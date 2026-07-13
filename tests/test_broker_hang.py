"""Unit tests for broker hang self-abort (torch-free; no GPU / no os._exit)."""
from __future__ import annotations

import pytest

from chess_anti_engine.broker_hang import (
    HANG_ABORT_EXIT_CODE,
    BrokerHangWatchdog,
    resolve_hang_abort_seconds,
    should_hang_abort,
)


def test_should_hang_abort_decision_matrix() -> None:
    # Unarmed (cold compile window): never abort, even if "old".
    assert not should_hang_abort(
        armed=False, oldest_inflight_age_s=9999.0, threshold_s=300.0,
    )
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


def test_completed_batch_clears_inflight_and_arms() -> None:
    clock = [100.0]
    exits: list[int] = []
    wd = BrokerHangWatchdog(
        threshold_s=10.0,
        poll_interval_s=1000.0,
        exit_fn=exits.append,
        clock=lambda: clock[0],
    )
    assert not wd.armed
    assert wd.oldest_inflight_age_s() is None

    wd.mark_forward_start(batch_size=64)
    clock[0] = 105.0
    assert wd.oldest_inflight_age_s() == pytest.approx(5.0)
    # Still unarmed: first batch not complete → no abort even if age > threshold.
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
        exit_fn=exits.append,
        clock=lambda: clock[0],
    )
    wd.mark_forward_start(4)
    clock[0] = 1.0
    wd.mark_forward_done(success=False)
    assert not wd.armed
    assert wd.oldest_inflight_age_s() is None
    # A long hang before arming must not fire.
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
