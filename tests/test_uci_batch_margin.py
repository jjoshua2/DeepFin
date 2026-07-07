"""Batch-time-variance clock margin (chess_anti_engine/uci/search.py).

A search chunk (GPU batch) can't be interrupted once launched, and the deadline
is only checked between chunks, so a chunk slower than the running average can
overrun a tight clock. These tests pin the self-calibrating margin that reserves
one worst-case batch before the hard deadline.
"""
from __future__ import annotations

import threading

from chess_anti_engine.uci.search import (
    _BATCH_MARGIN_MIN_SAMPLES,
    _MAX_BATCH_MARGIN_FRACTION,
    SearchWorker,
)
from chess_anti_engine.uci.time_manager import Deadline


def _bare_worker(sigmas: float = 2.0) -> SearchWorker:
    """A SearchWorker with only the fields the margin/stop logic touches — no
    evaluator, tree, or C state (heavy __init__ side effects avoided)."""
    w = SearchWorker.__new__(SearchWorker)
    w._batch_margin_sigmas = sigmas
    w._reset_batch_timing()
    w._max_tree_bytes = 0
    w._tree = None
    return w


def test_margin_zero_until_enough_samples() -> None:
    w = _bare_worker()
    assert w._batch_margin_ms() == 0.0
    w._record_batch_ms(100.0)
    # One sample is not enough to estimate variance.
    assert _BATCH_MARGIN_MIN_SAMPLES == 2
    assert w._batch_margin_ms() == 0.0
    w._record_batch_ms(100.0)
    assert w._batch_margin_ms() > 0.0


def test_margin_is_mean_plus_k_sigma() -> None:
    w = _bare_worker(sigmas=2.0)
    for ms in (80.0, 120.0):  # mean 100, sample std 28.284...
        w._record_batch_ms(ms)
    # mean + 2*std = 100 + 2*28.2843 = 156.5685
    assert abs(w._batch_margin_ms() - 156.5685) < 1e-3


def test_sigmas_zero_disables_margin() -> None:
    w = _bare_worker(sigmas=0.0)
    w._record_batch_ms(50.0)
    w._record_batch_ms(90.0)
    assert w._batch_margin_ms() == 0.0


def test_reset_clears_stats() -> None:
    w = _bare_worker()
    w._record_batch_ms(100.0)
    w._record_batch_ms(140.0)
    assert w._batch_margin_ms() > 0.0
    w._reset_batch_timing()
    assert w._batch_n == 0
    assert w._batch_margin_ms() == 0.0


def test_stop_fires_when_remaining_below_margin() -> None:
    # _should_stop_search reads the real monotonic clock, so build deadlines
    # against it (now=None) rather than a fake start.
    w = _bare_worker()
    ev = threading.Event()
    # Plenty of time (~100s remaining) vs a 200ms margin -> keep searching.
    healthy = Deadline(100_000, now=None)
    assert w._should_stop_search(
        stop_event=ev, deadline=healthy, max_nodes=None, max_depth=None,
        total_nodes=10, pv_len=1, time_margin_ms=200.0,
    ) is None
    # ~50ms remaining < 200ms margin -> stop one batch early, before the deadline.
    tight = Deadline(50, now=None)
    assert w._should_stop_search(
        stop_event=ev, deadline=tight, max_nodes=None, max_depth=None,
        total_nodes=10, pv_len=1, time_margin_ms=200.0,
    ) == "time_margin"


def test_margin_cap_fraction_is_sane() -> None:
    # The cap must leave the search at least half the budget in the worst case.
    assert 0.0 < _MAX_BATCH_MARGIN_FRACTION <= 0.5


def test_clock_margin_is_zero_off_clock() -> None:
    # movetime/nodes/depth searches (optimum_ms None) never reserve the margin.
    w = _bare_worker()
    w._record_batch_ms(80.0)
    w._record_batch_ms(120.0)
    assert w._clock_time_margin_ms(None, 100_000) == 0.0
    # Clock game with the same samples returns the (uncapped-here) raw margin.
    assert w._clock_time_margin_ms(optimum_ms=500, start_remaining_ms=100_000) > 0.0


def test_clock_margin_cap_clamps_a_runaway_estimate() -> None:
    # Two very slow chunks -> a large raw margin that would reserve away a small
    # budget; the cap keeps the search from stopping after one chunk on a bad
    # early estimate.
    w = _bare_worker(sigmas=2.0)
    w._record_batch_ms(500.0)
    w._record_batch_ms(900.0)
    raw = w._batch_margin_ms()
    assert raw > 700.0  # mean 700 + 2*std(~283) ~ 1266ms
    capped = w._clock_time_margin_ms(optimum_ms=500, start_remaining_ms=400)
    assert capped == _MAX_BATCH_MARGIN_FRACTION * 400  # cap binds (200ms)
    assert capped < raw


def test_clock_margin_no_cap_when_budget_unknown() -> None:
    # If the deadline can't report a starting budget (None), fall back to the
    # raw margin rather than crashing.
    w = _bare_worker()
    w._record_batch_ms(80.0)
    w._record_batch_ms(120.0)
    assert w._clock_time_margin_ms(optimum_ms=500, start_remaining_ms=None) == w._batch_margin_ms()


def test_expired_outranks_margin_and_no_margin_is_noop() -> None:
    w = _bare_worker()
    ev = threading.Event()
    expired = Deadline(0, now=None)
    # Hard deadline already passed -> "external", regardless of margin.
    assert w._should_stop_search(
        stop_event=ev, deadline=expired, max_nodes=None, max_depth=None,
        total_nodes=1, pv_len=1, time_margin_ms=200.0,
    ) == "external"
    # With margin 0 (movetime/nodes searches), a healthy deadline never stops here.
    healthy = Deadline(100000, now=None)
    assert w._should_stop_search(
        stop_event=ev, deadline=healthy, max_nodes=None, max_depth=None,
        total_nodes=1, pv_len=1, time_margin_ms=0.0,
    ) is None
