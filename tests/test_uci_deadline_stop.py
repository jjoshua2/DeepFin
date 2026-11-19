"""A finite UCI deadline must reach workers blocked inside a search chunk."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.search import DEFAULT_LEAF_GATHER, SearchResult, SearchWorker
from chess_anti_engine.uci.time_manager import SearchLimits


def _result() -> SearchResult:
    return SearchResult(
        bestmove_uci="e2e4", ponder_uci=None, nodes=1, pv=("e2e4",), score_cp=0,
    )


class _SlowEvaluator:
    def evaluate_encoded(self, x, relations=None):
        del relations
        # Model latency releases the GIL; each submitted batch must finish and
        # unwind its virtual loss before the pool acknowledges the deadline.
        time.sleep(0.01)
        return np.zeros((len(x), 4672), np.float32), np.zeros((len(x), 3), np.float32)


def test_deadline_interrupts_real_walker_chunk_without_fallback(capsys):
    worker = SearchWorker(
        # Gather 1 pins this test's original walker batch: the node bound
        # below is a stop-latency bound, and the default gather is now 32.
        _SlowEvaluator(), device="cpu", n_walkers=2, walker_gather=1, chunk_sims=2048,
        gumbel_cfg=GumbelConfig(input_extra_features="v1", add_noise=False),
    )
    engine = Engine(worker)
    try:
        started = time.monotonic()
        result = engine._run_one_phase(
            SearchLimits(deadline_ms=100), is_ponder=False, board=chess.Board(),
        )
        elapsed = time.monotonic() - started
        assert elapsed < 1.5  # Scheduling tolerance, not a real-time guarantee.
        assert 0 < result.nodes < 128
        assert chess.Move.from_uci(result.bestmove_uci) in chess.Board().legal_moves
        assert engine.bestmove_fallback_used == 0
        assert "search error:" not in capsys.readouterr().out
        assert worker._tree is not None
        for nid in range(worker._tree.node_count()):
            assert worker._tree.get_virtual_loss(nid) == 0

        # A later, node-limited go gets a fresh event and can use the same tree.
        engine._stop_event = threading.Event()
        later = engine._run_one_phase(
            SearchLimits(max_nodes=8), is_ponder=False, board=chess.Board(),
        )
        assert later.nodes == 8
        assert not engine._stop_event.is_set()
        assert engine.bestmove_fallback_used == 0
    finally:
        engine.close()


class _BatchCostEvaluator:
    """Every call costs ``batch_s`` whatever its size (model latency releases
    the GIL), and records when it started."""

    def __init__(self, batch_s: float) -> None:
        self.batch_s = batch_s
        self.starts: list[float] = []
        self._lock = threading.Lock()

    def evaluate_encoded(self, x, relations=None):
        del relations
        with self._lock:
            self.starts.append(time.monotonic())
        time.sleep(self.batch_s)
        return np.zeros((len(x), 4672), np.float32), np.zeros((len(x), 3), np.float32)


def _walker_engine(evaluator, **kw) -> tuple[SearchWorker, Engine]:
    worker = SearchWorker(
        evaluator, device="cpu", n_walkers=2, chunk_sims=2048,
        gumbel_cfg=GumbelConfig(input_extra_features="v1", add_noise=False), **kw,
    )
    assert worker._walker_gather == DEFAULT_LEAF_GATHER
    return worker, Engine(worker)


def test_deadline_interrupts_walker_chunk_at_the_default_gather(capsys):
    """A walker chunk that ignored the stop would run its whole time-capped
    first chunk (2048 sims = 32 batches a walker at 0.1 s each, > 3 s); a
    working stop ends it within one batch of the 600 ms deadline."""
    _worker, engine = _walker_engine(_BatchCostEvaluator(0.1))
    try:
        started = time.monotonic()
        result = engine._run_one_phase(
            SearchLimits(deadline_ms=600), is_ponder=False, board=chess.Board(),
        )
        elapsed = time.monotonic() - started
        assert elapsed < 1.5  # Scheduling tolerance, not a real-time guarantee.
        assert 0 < result.nodes < 2048
        assert engine.bestmove_fallback_used == 0
        assert "search error:" not in capsys.readouterr().out
    finally:
        engine.close()


def test_walker_batches_in_flight_at_the_stop_end_by_the_deadline(capsys):
    """The stop reaches a walker pool one measured walker batch before the
    deadline: a walker that has submitted a batch only sees the stop after it,
    and at gather 32 on CPU a stop AT the deadline answered late enough for
    the watchdog to play an unsearched move. With every batch costing 150 ms,
    no batch may START later than 150 ms before the deadline. The walkers'
    batches start in step every 150 ms from the chunk start, so the deadline
    sits half a batch off that grid: a stop at the deadline itself then starts
    a batch about 65 ms before it, 85 ms past the bound (measured: last start
    910 ms with the reserve removed, 757-760 ms with it)."""
    batch_s, deadline_ms = 0.15, 975
    evaluator = _BatchCostEvaluator(batch_s)
    worker, engine = _walker_engine(evaluator)
    try:
        # The pool measures its batch time on any search (startup calibrates).
        engine._run_one_phase(SearchLimits(max_nodes=128), is_ponder=False, board=chess.Board())
        assert worker._walker_pool is not None
        assert worker._walker_pool.batch_ms_hi() >= batch_s * 1000.0
        for _ in range(3):
            worker.reset_tree()
            engine._stop_event = threading.Event()
            evaluator.starts.clear()
            started = time.monotonic()
            result = engine._run_one_phase(
                SearchLimits(deadline_ms=deadline_ms), is_ponder=False, board=chess.Board(),
            )
            elapsed_ms = (time.monotonic() - started) * 1000.0
            last_start_ms = (max(evaluator.starts) - started) * 1000.0
            assert last_start_ms <= deadline_ms - batch_s * 1000.0 + 25.0, last_start_ms
            assert elapsed_ms < deadline_ms + 100.0, elapsed_ms
            assert result.nodes > 0
            # Ended by the reserve between chunks, not by spinning empty
            # chunks until the deadline fired.
            assert worker.last_stop_reason == "time_margin", worker.last_stop_reason
        assert engine.bestmove_fallback_used == 0
        assert "search error:" not in capsys.readouterr().out
    finally:
        engine.close()


@pytest.mark.parametrize("raises", [False, True])
def test_timer_is_joined_after_success_or_failure_and_captures_phase(monkeypatch, raises):
    timers = []
    timer_type = threading.Timer

    class RecordingTimer(timer_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            timers.append(self)

    monkeypatch.setattr("chess_anti_engine.uci.engine.threading.Timer", RecordingTimer)
    worker = MagicMock()
    engine = Engine(worker)
    old_event = engine._stop_event
    new_event = threading.Event()

    def run(*_args, **kwargs):
        assert kwargs["stop_event"] is old_event
        engine._stop_event = new_event
        if raises:
            raise RuntimeError("injected evaluator fault")
        return _result()

    worker.run = run
    try:
        engine._run_one_phase(
            SearchLimits(deadline_ms=10_000), is_ponder=False, board=chess.Board(),
        )
        assert len(timers) == 1
        assert not timers[0].is_alive()
        assert timers[0].finished.is_set()
        assert not old_event.is_set()
        assert not new_event.is_set()
        assert engine.bestmove_fallback_used == int(raises)
        # Even an already-dispatched callback belongs solely to its old phase.
        timers[0].function(*timers[0].args, **timers[0].kwargs)
        assert old_event.is_set()
        assert not new_event.is_set()
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("limits", "is_ponder", "finite"),
    [
        (SearchLimits(max_nodes=8), False, False),
        (SearchLimits(max_depth=8), False, False),
        (SearchLimits(infinite=True), False, False),
        (SearchLimits(deadline_ms=10_000, ponder=True), True, False),
        (SearchLimits(deadline_ms=10_000), False, True),
        (SearchLimits(deadline_ms=10_000, max_nodes=8), False, True),
    ],
)
def test_timer_respects_phase_deadline(monkeypatch, limits, is_ponder, finite):
    timers = []
    timer_type = threading.Timer

    def make_timer(*args, **kwargs):
        timer = timer_type(*args, **kwargs)
        timers.append(timer)
        return timer

    monkeypatch.setattr("chess_anti_engine.uci.engine.threading.Timer", make_timer)
    worker = MagicMock()
    worker.run.return_value = _result()
    engine = Engine(worker)
    try:
        engine._run_one_phase(
            limits, is_ponder=is_ponder, board=chess.Board(),
            started_at=time.monotonic() - 2.0 if finite else None,
        )
        assert len(timers) == int(finite)
        for timer in timers:
            assert 0 < timer.interval <= 8.0  # Charge elapsed transition time.
            assert not timer.is_alive()
        assert not engine._stop_event.is_set()
    finally:
        engine.close()
