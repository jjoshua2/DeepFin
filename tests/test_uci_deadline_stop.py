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
from chess_anti_engine.uci.search import SearchResult, SearchWorker
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
        _SlowEvaluator(), device="cpu", n_walkers=2, chunk_sims=2048,
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
