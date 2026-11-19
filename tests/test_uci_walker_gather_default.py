"""The walker-pool leaf gather: one default for every consumer, and what a
larger (explicit) gather does.

The default stays 1: a walker batch cannot be interrupted, and at full
concurrency on a slow device a 32-leaf batch can outlast a short move's budget
(raising the default is a separate change). The classic path (one walker) never
reads the gather, which the identity check below demonstrates on a fixed-node
search.
"""
from __future__ import annotations

import threading
import time

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.__main__ import _build_parser
from chess_anti_engine.uci.engine import DEFAULT_LEAF_GATHER, EngineOptions
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


# An explicit larger gather (`--walker-gather 32`), as an operator may set it.
LARGE_GATHER = 32


def test_walker_gather_default_is_one_everywhere() -> None:
    assert DEFAULT_LEAF_GATHER == 1
    assert EngineOptions().leaf_gather == DEFAULT_LEAF_GATHER
    assert _build_parser().parse_args(["--checkpoint", "x"]).walker_gather == DEFAULT_LEAF_GATHER


def _fixed_node_search(walker_gather: int) -> tuple[str, int, tuple[str, ...], list[int]]:
    torch.manual_seed(1234)
    torch_model = build_model(ModelConfig(
        input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
    ))
    torch_model.eval()
    evaluator = ThreadSafeGPUDispatcher(DirectGPUEvaluator(
        torch_model, device="cpu", max_batch=128, use_amp=False, n_slots=2,
    ))
    worker = SearchWorker(
        evaluator, device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=128, add_noise=False),
        chunk_sims=128, n_walkers=1, walker_gather=walker_gather,
    )
    try:
        result = worker.run(
            chess.Board(), stop_event=threading.Event(),
            deadline=Deadline(60_000), max_nodes=384,
        )
        assert worker._tree is not None
        assert worker._root_id is not None
        _actions, visits = worker._tree.get_children_visits(worker._root_id)
        return result.bestmove_uci, result.nodes, result.pv, np.asarray(visits).tolist()
    finally:
        worker.close()


def test_leaf_gather_is_inert_on_the_single_walker_path() -> None:
    assert _fixed_node_search(1) == _fixed_node_search(LARGE_GATHER)


def test_every_gather_default_is_the_one_constant() -> None:
    """`SearchWorker` and the evaluator warm-up must not keep their own 1."""
    import inspect

    from chess_anti_engine.uci.__main__ import _warmup_evaluator

    for fn in (SearchWorker.__init__, _warmup_evaluator):
        assert inspect.signature(fn).parameters["walker_gather"].default == DEFAULT_LEAF_GATHER, fn


class _StopAfterPolls(threading.Event):
    """A stop event that reads set from its ``polls``-th `is_set` on."""

    def __init__(self, polls: int) -> None:
        super().__init__()
        self._left = polls
        self._lock = threading.Lock()

    def is_set(self) -> bool:
        with self._lock:
            self._left -= 1
            return self._left < 0 or super().is_set()


class _ZeroEvaluator:
    def __init__(self) -> None:
        self.batches: list[int] = []

    def evaluate_encoded(self, x, relations=None):
        del relations
        self.batches.append(len(x))
        return np.zeros((len(x), 4672), np.float32), np.zeros((len(x), 3), np.float32)


def test_a_walker_stops_mid_gather_and_submits_what_it_has() -> None:
    """At gather 32 (set explicitly) a stop must not wait for a full gather: the walker polls
    the stop before every leaf, submits the partial batch and leaves. (A
    walker that polled only between batches gathered 32 more leaves, and one
    that never saw the job's event ran the whole budget.)"""
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.mcts._mcts_tree import MCTSTree
    from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig

    tree = MCTSTree()
    tree.reserve(4096, 65536)
    root_cb = CBoard.from_board(chess.Board())
    rid = tree.add_root(0, 0.0)
    legal = root_cb.legal_move_indices().astype(np.int32)
    tree.expand(rid, legal, np.full(legal.size, 1.0 / legal.size, dtype=np.float64))
    evaluator = _ZeroEvaluator()
    pool = WalkerPool(
        WalkerPoolConfig(n_walkers=1, c_puct=1.5, fpu_at_root=0.0, fpu_reduction=0.33,
                         vloss_weight=3, gather=LARGE_GATHER),
        evaluator,
    )
    try:
        # Polls: one before the gather, then one per leaf -- set on the 5th.
        completed = pool.run(
            tree=tree, root_id=rid, root_cboard=root_cb,
            target_sims=1000, stop_event=_StopAfterPolls(4),
        )
    finally:
        pool.close()
    assert completed == 3, (completed, evaluator.batches)
    assert evaluator.batches == [3]
    assert pool.batch_ms_hi() == 0.0  # a partial batch is not timed


class _ScheduledEvaluator:
    """A full batch costs ``full_s``; a partial one costs nothing."""

    def __init__(self, gather: int) -> None:
        self.gather = gather
        self.full_s = 0.0

    def evaluate_encoded(self, x, relations=None):
        del relations
        if len(x) == self.gather:
            time.sleep(self.full_s)
        return np.zeros((len(x), 4672), np.float32), np.zeros((len(x), 3), np.float32)


def _single_walker_pool(evaluator, gather: int):
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.mcts._mcts_tree import MCTSTree
    from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig

    tree = MCTSTree()
    tree.reserve(4096, 65536)
    root_cb = CBoard.from_board(chess.Board())
    rid = tree.add_root(0, 0.0)
    legal = root_cb.legal_move_indices().astype(np.int32)
    tree.expand(rid, legal, np.full(legal.size, 1.0 / legal.size, dtype=np.float64))
    pool = WalkerPool(
        WalkerPoolConfig(n_walkers=1, c_puct=1.5, fpu_at_root=0.0, fpu_reduction=0.33,
                         vloss_weight=3, gather=gather),
        evaluator,
    )

    def run(sims: int) -> int:
        return pool.run(tree=tree, root_id=rid, root_cboard=root_cb,
                        target_sims=sims, stop_event=threading.Event())

    return pool, run


def test_batch_reserve_is_a_decayed_max_of_full_batches_only() -> None:
    """`batch_ms_hi` is the stop reserve of a walker pool: the slowest recent
    FULL batch. A fast full batch after a slow one decays it by one step (it is
    not the last value), and partial batches -- a chunk smaller than the
    gather -- neither lower nor decay it (with eight walkers they used to drive
    it to a fraction of a full batch, and the stall watchdog then answered)."""
    from chess_anti_engine.uci import walker_pool as wp

    gather = 4
    evaluator = _ScheduledEvaluator(gather)
    pool, run = _single_walker_pool(evaluator, gather)
    try:
        evaluator.full_s = 0.12
        run(gather)                       # one slow full batch
        slow = pool.batch_ms_hi()
        assert slow >= 120.0
        evaluator.full_s = 0.0
        run(gather)                       # one fast full batch: one decay step
        assert pool.batch_ms_hi() == pytest.approx(slow * wp._BATCH_HI_DECAY, rel=1e-6)
        kept = pool.batch_ms_hi()
        for _ in range(60):               # sixty partial batches
            run(gather - 1)
        assert pool.batch_ms_hi() == kept
    finally:
        pool.close()
