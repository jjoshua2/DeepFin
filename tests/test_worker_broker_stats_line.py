"""The worker's `broker client stats:` line must be able to show a wedged slot.

SHARED_BROKER_AUDIT B6's real severity is diagnosis, not correctness. The only
production surface for the multi-slot client is this 60-second line, and of the
per-slot arrays in ``stats`` it read exactly one: ``slot_requests``. On the
measured wedge that array was ``[3, 3, 3, 3]`` — the dead slot indistinguishable
from the three healthy ones — while ``slot_roundtrip_s`` was
``[0.02, 0.0, 0.0, 2.43]``, a 100x separation that was computed, exported, and
read by nothing.

Meanwhile ``pos_s`` and ``rows_per_req`` came from ``lifetime_positions``, which
counted rows no model ever evaluated, so a broker stall reported full
throughput.
"""
from __future__ import annotations

import threading
from typing import Any, cast
from unittest.mock import Mock

from chess_anti_engine import worker as worker_mod
from chess_anti_engine.worker import WorkerSession


class _FakeBroker:
    def __init__(self, stats: dict[str, Any]) -> None:
        self.stats = stats


def _wedge_stats(**over: Any) -> dict[str, Any]:
    """3 healthy slots at ~5 ms, 1 wedged at ~30 s, quarantined after 2 strikes."""
    stats: dict[str, Any] = {
        "slots": 4,
        "max_inflight": 4,
        "available_slots": 3,
        "lifetime_requests": 92,
        "lifetime_served_requests": 90,
        "lifetime_failed_requests": 2,
        "lifetime_failed_positions": 16,
        "lifetime_positions": 720,
        "lifetime_legal_requests": 90,
        "lifetime_legal_positions": 720,
        "lifetime_wait_s": 0.2,
        "lifetime_roundtrip_s": 60.4,
        "slot_requests": [30, 30, 30, 2],
        "slot_served": [30, 30, 30, 0],
        "slot_failures": [0, 0, 0, 2],
        "slot_quarantines": [0, 0, 0, 1],
        "slots_quarantined": 1,
        "slot_positions": [240, 240, 240, 0],
        "slot_wait_s": [0.05, 0.05, 0.05, 0.05],
        "slot_roundtrip_s": [0.15, 0.15, 0.15, 60.0],
        "stale_responses_rejected": 0,
    }
    stats.update(over)
    return stats


def _session(stats: dict[str, Any], monkeypatch: Any) -> tuple[Any, Mock]:
    monkeypatch.setattr(worker_mod, "MultiSlotInferenceClient", _FakeBroker)
    session = object.__new__(WorkerSession)
    s = cast("Any", session)
    s.inference_client = _FakeBroker(stats)
    s._last_broker_client_stats_log_s = 0.0
    s._last_broker_client_stats_snapshot = {}
    s.log = Mock()
    s._completion_telemetry_lock = threading.Lock()
    s._completion_games = 0
    s._completion_positions = 0
    s._completion_callback_s = 0.0
    s._completion_upload_s = 0.0
    s._completion_by_thread = {}
    s._last_completion_stats_snapshot = (0, 0, 0.0, 0.0, {})
    s._maybe_log_selfplay_phase_stats = Mock()
    return session, s.log


def _rendered(log: Mock) -> str:
    log.info.assert_called_once()
    args = log.info.call_args[0]
    return args[0] % tuple(args[1:])


def test_the_line_identifies_the_wedged_slot(monkeypatch: Any) -> None:
    session, log = _session(_wedge_stats(), monkeypatch)
    WorkerSession._maybe_log_broker_client_stats(session, 60.0)
    line = _rendered(log)

    # slots_active must come from SERVED requests: the wedged slot took
    # attempts, so an attempts-based count still reads 4/4.
    assert "slots_active=3/4" in line, line
    # The roundtrip array, read for the first time. 60 s over 2 attempts.
    assert "slot_rt_ms_max=30000" in line, line
    assert "slots_quarantined=1" in line, line
    assert "failed_req=+2" in line, line


def test_throughput_excludes_the_rows_that_were_never_evaluated(
    monkeypatch: Any,
) -> None:
    session, log = _session(_wedge_stats(), monkeypatch)
    WorkerSession._maybe_log_broker_client_stats(session, 60.0)
    line = _rendered(log)
    # 720 served rows over 90 served requests, NOT over 92 attempts.
    assert "rows_per_req=8.0" in line, line
    assert "pos_s=12" in line, line


def test_a_window_where_every_request_failed_still_logs(monkeypatch: Any) -> None:
    """The stall this line exists to show must not silence it.

    With served-only totals and the old ``if delta_requests <= 0: return``,
    a window in which nothing succeeded would print nothing at all.
    """
    stats = _wedge_stats(
        lifetime_requests=8, lifetime_served_requests=0, lifetime_failed_requests=8,
        lifetime_positions=0, lifetime_legal_requests=0, lifetime_legal_positions=0,
        slot_requests=[2, 2, 2, 2], slot_served=[0, 0, 0, 0],
        slot_failures=[2, 2, 2, 2], slots_quarantined=4,
        slot_roundtrip_s=[60.0, 60.0, 60.0, 60.0],
    )
    session, log = _session(stats, monkeypatch)
    WorkerSession._maybe_log_broker_client_stats(session, 60.0)
    line = _rendered(log)
    assert "slots_active=0/4" in line, line
    assert "pos_s=0" in line, line
    assert "failed_req=+8" in line, line
    assert "slots_quarantined=4" in line, line


def test_a_healthy_window_carries_no_failure_tokens(monkeypatch: Any) -> None:
    """Negative control: the healthy line stays greppable-clean, so any
    ``failed_req``/``slots_quarantined`` hit is a real event."""
    stats = _wedge_stats(
        lifetime_requests=90, lifetime_served_requests=90,
        lifetime_failed_requests=0, lifetime_failed_positions=0,
        slot_requests=[30, 30, 30, 30], slot_served=[30, 30, 30, 30],
        slot_failures=[0, 0, 0, 0], slot_quarantines=[0, 0, 0, 0],
        slots_quarantined=0, slot_roundtrip_s=[0.15, 0.15, 0.15, 0.15],
        available_slots=4,
    )
    session, log = _session(stats, monkeypatch)
    WorkerSession._maybe_log_broker_client_stats(session, 60.0)
    line = _rendered(log)
    assert "failed_req" not in line, line
    assert "slots_quarantined" not in line, line
    assert "slots_active=4/4" in line, line
    assert "slot_rt_ms_max=5" in line, line


def test_nothing_happened_at_all_still_logs_nothing(monkeypatch: Any) -> None:
    stats = _wedge_stats(
        lifetime_requests=0, lifetime_served_requests=0, lifetime_failed_requests=0,
        lifetime_positions=0, slot_requests=[0, 0, 0, 0], slot_served=[0, 0, 0, 0],
        slot_failures=[0, 0, 0, 0], slots_quarantined=0,
        slot_roundtrip_s=[0.0, 0.0, 0.0, 0.0],
    )
    session, log = _session(stats, monkeypatch)
    WorkerSession._maybe_log_broker_client_stats(session, 60.0)
    log.info.assert_not_called()
