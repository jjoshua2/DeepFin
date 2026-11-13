"""Forced roots and real node counts in ``SearchWorker.run``.

Found by the TCEC-readiness audit: with exactly one legal move on a 60 s clock
the engine searched the whole 4.7 s soft target, and the classic Gumbel path
reported ~100M "nodes" at ~20M nps for that search because every chunk counted
the REQUESTED simulations although the C path returns a forced root's sole
action without simulating at all.
"""
from __future__ import annotations

import hashlib
import threading
import time

import chess
import numpy as np

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.uci.search import SearchResult, SearchWorker
from chess_anti_engine.uci.time_manager import Deadline

FORCED = "7k/8/8/8/8/8/6q1/7K w - - 0 1"  # Kxg2 is the only legal move


class _SpreadEvaluator:
    def __init__(self) -> None:
        self.rows = 0

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        b = int(x.shape[0])
        self.rows += b
        rows = np.ascontiguousarray(x).reshape(b, -1)
        pol = np.zeros((b, POLICY_SIZE), dtype=np.float32)
        wdl = np.zeros((b, 3), dtype=np.float32)
        for i in range(b):
            digest = hashlib.blake2b(rows[i].tobytes(), digest_size=8).digest()
            r = np.random.default_rng(int.from_bytes(digest, "little"))
            pol[i, :] = r.standard_normal(POLICY_SIZE, dtype=np.float32)
            wdl[i, :] = r.standard_normal(3, dtype=np.float32)
        return pol, wdl


def _worker(chunk: int = 256) -> SearchWorker:
    return SearchWorker(
        _SpreadEvaluator(), device="cpu", chunk_sims=chunk,
        gumbel_cfg=GumbelConfig(simulations=chunk, topk=16, temperature=0.0, add_noise=False),
    )


def test_forced_root_on_a_clock_returns_immediately() -> None:
    w = _worker()
    t0 = time.monotonic()
    result = w.run(
        chess.Board(FORCED), stop_event=threading.Event(), deadline=Deadline(20_000),
        max_nodes=None, optimum_ms=2_500,
    )
    elapsed = time.monotonic() - t0
    assert result.bestmove_uci == "h1g2"
    assert elapsed < 1.0, elapsed
    assert result.pv
    assert result.pv[0] == "h1g2"


def test_single_searchmove_on_a_clock_returns_immediately() -> None:
    w = _worker()
    t0 = time.monotonic()
    result = w.run(
        chess.Board(), stop_event=threading.Event(), deadline=Deadline(20_000),
        max_nodes=None, optimum_ms=2_500, root_moves=("e2e4",),
    )
    assert result.bestmove_uci == "e2e4"
    assert time.monotonic() - t0 < 1.0


def test_forced_root_without_optimum_waits_for_stop() -> None:
    """ponder / infinite run with optimum_ms=None and must not return early."""
    w = _worker()
    stop = threading.Event()
    out: list[SearchResult] = []
    th = threading.Thread(target=lambda: out.append(w.run(
        chess.Board(FORCED), stop_event=stop, deadline=Deadline(None), max_nodes=None,
        optimum_ms=None,
    )))
    th.start()
    time.sleep(0.6)
    assert th.is_alive()
    assert not out
    stop.set()
    th.join(timeout=10)
    assert not th.is_alive()
    assert out
    assert out[0].bestmove_uci == "h1g2"


def test_forced_root_reports_real_node_count_and_clean_batch_stats() -> None:
    w = _worker()
    result = w.run(
        chess.Board(FORCED), stop_event=threading.Event(), deadline=Deadline(400),
        max_nodes=None, optimum_ms=None,
    )
    # The Gumbel path returns the sole action without simulating: 0 nodes,
    # not hundreds of thousands of requested-but-never-run simulations.
    assert result.nodes == 0
    assert w._batch_n == 0


def test_forced_root_under_node_limit_terminates() -> None:
    """A chunk that completes nothing can never reach a node budget; with no
    deadline the search must still end instead of idling until `stop`."""
    w = _worker()
    stop = threading.Event()
    out: list[SearchResult] = []
    th = threading.Thread(target=lambda: out.append(w.run(
        chess.Board(FORCED), stop_event=stop, deadline=Deadline(None), max_nodes=1000,
    )), daemon=True)
    th.start()
    th.join(timeout=10)
    hung = th.is_alive()
    stop.set()
    th.join(timeout=10)
    assert not hung
    assert out
    assert out[0].bestmove_uci == "h1g2"


def test_node_count_is_root_visits() -> None:
    w = _worker(chunk=256)
    result = w.run(
        chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=1024,
    )
    assert w._tree is not None
    assert w._root_id is not None
    visits = int(w._tree.get_children_visits(w._root_id)[1].sum())
    assert result.nodes == visits
    assert 0 < result.nodes <= 1024


def test_forced_root_on_walker_pool_spends_one_small_chunk() -> None:
    """The PUCT walker pool does simulate the forced child; a clocked search
    banks after one capped chunk instead of a full chunk_sims chunk."""
    w = SearchWorker(
        _SpreadEvaluator(), device="cpu", chunk_sims=2048, n_walkers=2,
        gumbel_cfg=GumbelConfig(simulations=2048, topk=16, temperature=0.0, add_noise=False),
    )
    try:
        result = w.run(
            chess.Board(FORCED), stop_event=threading.Event(), deadline=Deadline(20_000),
            max_nodes=None, optimum_ms=2_500,
        )
    finally:
        w.close()
    assert result.bestmove_uci == "h1g2"
    assert 0 < result.nodes <= 64, result.nodes
