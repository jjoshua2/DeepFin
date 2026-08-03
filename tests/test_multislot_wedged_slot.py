"""A wedged broker slot must be evicted, re-probed, and never counted as throughput.

SHARED_BROKER_AUDIT B6, reproduced at 3 live slots + 1 whose shared memory never
exists (``sb_multislot_wedge_accounting.py``). Two defects, both on the
production path (``distributed_inference_slots_per_worker: 4``):

1. ``_release_client`` put the slot straight back on the availability queue no
   matter how the request ended. There was no per-slot failure counter and no
   eviction, so a permanently dead slot was handed out again forever — one slot
   in four costing a full ``request_timeout_s`` (30 s in production) each turn.
2. Every failed request incremented ``lifetime_requests``/``lifetime_positions``
   and ``slot_requests[idx]``, so the worker's 60-second ``broker client stats:``
   line reported rows/s for positions no model ever evaluated, and the per-slot
   array it reads was ``[3, 3, 3, 3]`` — the dead slot exactly indistinguishable
   from the three healthy ones.

These tests use a fake underlying client rather than real shared memory: the
subject is the multiplexer's bookkeeping and rotation policy, and a fake makes
"this request failed" exact rather than timing-dependent.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.inference import MultiSlotInferenceClient

PLANES = 8


class _FakeSlotClient:
    """One broker slot. ``alive`` flips to simulate a wedge and a recovery."""

    def __init__(self, name: str, *, alive: bool = True) -> None:
        self.slot_name = name
        self.alive = alive
        self.calls = 0
        self.stale_responses_rejected = 0
        self.closed = False

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        if not self.alive:
            raise TimeoutError("inference broker timed out after 0.010s")
        n = int(x.shape[0])
        return np.zeros((n, 8), dtype=np.float32), np.zeros((n, 3), dtype=np.float32)

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert legal_flat.size >= 0
        assert legal_counts.size >= 0
        return self.evaluate_encoded(x)

    def close(self) -> None:
        self.closed = True


def _client(
    n_slots: int = 4, *, dead: tuple[int, ...] = (), **kwargs: Any,
) -> tuple[MultiSlotInferenceClient, list[_FakeSlotClient]]:
    mc = MultiSlotInferenceClient(
        slot_names=[f"slot-{i}" for i in range(n_slots)],
        max_batch=4,
        request_timeout_s=kwargs.pop("request_timeout_s", 0.05),
        input_planes=PLANES,
        **kwargs,
    )
    fakes = [
        _FakeSlotClient(f"slot-{i}", alive=i not in dead) for i in range(n_slots)
    ]
    mc._clients = fakes  # pyright: ignore[reportAttributeAccessIssue]
    # Rebuild the availability queue over the fakes, preserving order.
    while not mc._available_clients.empty():
        mc._available_clients.get_nowait()
    for idx, fake in enumerate(fakes):
        mc._available_clients.put((idx, fake))  # pyright: ignore[reportArgumentType]
    return mc, fakes


def _drive(mc: MultiSlotInferenceClient, n: int) -> tuple[int, int]:
    x = np.full((2, PLANES, 8, 8), 0.5, dtype=np.float32)
    ok = fail = 0
    for _ in range(n):
        try:
            mc.evaluate_encoded(x)
            ok += 1
        except Exception:  # the point is that it raised
            fail += 1
    return ok, fail


def test_a_dead_slot_is_quarantined_instead_of_being_handed_out_forever() -> None:
    """The audit's shape: 4 slots, 1 dead, sequential requests."""
    mc, fakes = _client(4, dead=(3,), slot_failure_threshold=2, slot_quarantine_s=30.0)
    try:
        ok, fail = _drive(mc, 24)
        assert ok + fail == 24
        # Threshold is 2 consecutive failures, so the dead slot is hit exactly
        # twice and then removed from rotation for the whole run.
        assert fakes[3].calls == 2, (
            f"dead slot kept being handed out: {fakes[3].calls} calls "
            "(pre-fix it took a request every 4th rotation, forever)"
        )
        assert fail == 2
        st = mc.stats
        assert st["slots_quarantined"] == 1
        assert st["slot_quarantines"][3] == 1
        assert st["slot_failures"] == [0, 0, 0, 2]
        assert st["slot_served"][3] == 0
        # The array the worker reads must now separate them.
        assert min(st["slot_served"][:3]) > 0
    finally:
        mc.close()


def test_failed_requests_are_not_counted_as_served_throughput() -> None:
    mc, _ = _client(4, dead=(3,), slot_failure_threshold=2, slot_quarantine_s=30.0)
    try:
        ok, fail = _drive(mc, 24)
        st = mc.stats
        assert fail > 0, "test would be vacuous with no failures"
        assert st["lifetime_positions"] == ok * 2, (
            "lifetime_positions must count only rows a model evaluated; "
            f"got {st['lifetime_positions']} for {ok} served requests"
        )
        assert st["lifetime_served_requests"] == ok
        assert st["lifetime_failed_requests"] == fail
        assert st["lifetime_failed_positions"] == fail * 2
        # attempts = served + failed, so nothing was lost either.
        assert st["lifetime_requests"] == ok + fail
        assert st["avg_rows_per_request"] == pytest.approx(2.0)
    finally:
        mc.close()


def test_stats_name_the_failures_at_all() -> None:
    """Pre-fix, no key in ``stats`` contained 'error', 'fail' or 'timeout'."""
    mc, _ = _client(2, dead=(1,), slot_failure_threshold=1, slot_quarantine_s=30.0)
    try:
        _drive(mc, 4)
        st = mc.stats
        named = [k for k in st if any(w in k for w in ("error", "fail", "quarant"))]
        assert named, f"nothing in stats names a failure: {sorted(st)}"
    finally:
        mc.close()


def test_quarantine_is_a_probe_not_an_execution() -> None:
    """A transient stall must not permanently shrink the pool.

    The slot comes back after the backoff, and a success clears its record, so
    a broker restart costs one quarantine window rather than a slot.
    """
    mc, fakes = _client(2, dead=(1,), slot_failure_threshold=1, slot_quarantine_s=0.05)
    try:
        _drive(mc, 4)
        assert mc.stats["slot_quarantines"][1] >= 1
        fakes[1].alive = True  # the broker came back
        time.sleep(0.2)
        calls_before = fakes[1].calls
        _drive(mc, 12)
        assert fakes[1].calls > calls_before, (
            "a recovered slot never returned to rotation — quarantine became "
            "a permanent eviction"
        )
        st = mc.stats
        assert st["slots_quarantined"] == 0
        assert st["slot_served"][1] > 0
    finally:
        mc.close()


def test_backoff_grows_and_a_success_resets_it() -> None:
    mc, fakes = _client(1, dead=(0,), slot_failure_threshold=1, slot_quarantine_s=0.02,
                        slot_quarantine_max_s=1.0)
    try:
        _drive(mc, 1)
        first = mc._slot_quarantine_backoff_s[0]
        time.sleep(0.05)
        _drive(mc, 1)
        second = mc._slot_quarantine_backoff_s[0]
        assert second > first, (first, second)
        fakes[0].alive = True
        time.sleep(0.2)
        _drive(mc, 1)
        assert mc._slot_quarantine_backoff_s[0] == pytest.approx(0.02), (
            "a served request must reset the backoff, or a slot that recovers "
            "still carries a minute-long penalty from a stall an hour ago"
        )
    finally:
        mc.close()


def test_every_slot_dead_raises_rather_than_hanging() -> None:
    """All-quarantined must surface as a loud, promptly-raised error.

    The worker recovers by resetting the client (``_reset_inference_client``),
    which it does for TimeoutError and for RuntimeErrors naming a slot/broker —
    so the type and the wording here are load-bearing.
    """
    mc, _ = _client(2, dead=(0, 1), slot_failure_threshold=1, slot_quarantine_s=5.0,
                    request_timeout_s=0.05)
    try:
        _drive(mc, 2)  # quarantine both
        assert mc.stats["slots_quarantined"] == 2
        t0 = time.perf_counter()
        with pytest.raises(TimeoutError, match="quarantined"):
            mc.evaluate_encoded(np.zeros((1, PLANES, 8, 8), dtype=np.float32))
        assert time.perf_counter() - t0 < 2.0
    finally:
        mc.close()


def test_a_healthy_pool_is_unchanged() -> None:
    """Negative control: nothing quarantines, nothing is counted as failed."""
    mc, fakes = _client(4)
    try:
        ok, fail = _drive(mc, 40)
        assert (ok, fail) == (40, 0)
        st = mc.stats
        assert st["slots_quarantined"] == 0
        assert st["slot_quarantines"] == [0, 0, 0, 0]
        assert st["lifetime_failed_requests"] == 0
        assert st["lifetime_positions"] == 80
        assert sum(st["slot_served"]) == 40
        assert st["available_slots"] == 4
        assert all(f.calls > 0 for f in fakes), "rotation stopped covering slots"
    finally:
        mc.close()


def test_concurrent_callers_still_see_consistent_totals() -> None:
    mc, _ = _client(4, dead=(2,), slot_failure_threshold=3, slot_quarantine_s=0.02)
    errs: list[BaseException] = []

    def worker() -> None:
        try:
            _drive(mc, 20)
        except BaseException as exc:
            errs.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
        assert not errs, errs
        st = mc.stats
        assert st["lifetime_requests"] == (
            st["lifetime_served_requests"] + st["lifetime_failed_requests"]
        )
        assert st["lifetime_positions"] == st["lifetime_served_requests"] * 2
        assert st["inflight"] == 0
    finally:
        mc.close()


def test_the_all_quarantined_window_reaches_a_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The periodic `broker client stats:` line cannot show this window.

    Nothing completes while every slot is quarantined, so that line's delta gate
    returns before printing and `slots_quarantined` never reaches an operator
    through it. The count has to leave on the one event that DOES happen — the
    failed acquire — or the worst state the eviction can reach is the one state
    it cannot report.
    """
    mc, _ = _client(2, dead=(0, 1), slot_failure_threshold=1, slot_quarantine_s=5.0,
                    request_timeout_s=0.05)
    try:
        _drive(mc, 2)  # quarantine both
        caplog.clear()
        with (
            caplog.at_level(logging.WARNING, logger="chess_anti_engine.inference"),
            pytest.raises(TimeoutError),
        ):
            mc.evaluate_encoded(np.zeros((1, PLANES, 8, 8), dtype=np.float32))
        hits = [r for r in caplog.records if "slot acquire failed" in r.message]
        assert hits, "the all-quarantined window left no log line"
        assert "2/2 slot(s) quarantined" in hits[0].getMessage()
    finally:
        mc.close()
