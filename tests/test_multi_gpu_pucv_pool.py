"""MultiGpuPucvPool tests.

Covers:
  - basic pool spawn + run + close
  - shared-tree visit accumulation across N=1 and N=2 worker pools
    (using two evaluator instances on CPU as a stand-in for two GPUs)
  - SearchWorker.install_multi_gpu_pucv routes through the pool
  - clear_multi_gpu_pucv reverts to single-evaluator search
  - vloss is fully unwound after run(), even with concurrent workers
"""
from __future__ import annotations

import threading
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.multi_gpu_pucv_pool import (
    MultiGpuPucvConfig,
    MultiGpuPucvPool,
)
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


def _make_evaluator(max_batch: int = 64) -> Any:
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    inner = DirectGPUEvaluator(
        model, device="cpu", max_batch=max_batch, use_amp=False, n_slots=2,
    )
    return ThreadSafeGPUDispatcher(inner)


def _seed_tree() -> tuple[MCTSTree, int, CBoard]:
    tree = MCTSTree()
    tree.reserve(1024, 8192)
    cb = CBoard.from_board(chess.Board())
    rid = tree.add_root(0, 0.0)
    legal = cb.legal_move_indices().astype(np.int32)
    priors = np.full(legal.size, 1.0 / legal.size, dtype=np.float64)
    tree.expand(rid, legal, priors)
    return tree, rid, cb


def test_pool_n1_accumulates_visits() -> None:
    """N=1 pool == single PucvChunker functionally; sims target reached."""
    ev = _make_evaluator()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=1, gather=8, vloss_weight=3),
        evaluators=[ev],
    )
    try:
        tree, rid, cb = _seed_tree()
        target = 32
        pool.run(tree=tree, root_id=rid, root_cboard=cb,
                 target_sims=target, stop_event=threading.Event())
        _, visits = tree.get_children_visits(rid)
        assert int(visits.sum()) >= target * 3 // 4
    finally:
        pool.close()


def test_pool_n2_accumulates_visits_no_vloss_leak() -> None:
    """N=2 workers on shared tree split the budget; total visits ≈ target.
    All vloss must be removed at end (atomicity check across workers)."""
    ev0 = _make_evaluator()
    ev1 = _make_evaluator()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, vloss_weight=3),
        evaluators=[ev0, ev1],
    )
    try:
        tree, rid, cb = _seed_tree()
        target = 64
        pool.run(tree=tree, root_id=rid, root_cboard=cb,
                 target_sims=target, stop_event=threading.Event())
        _, visits = tree.get_children_visits(rid)
        total = int(visits.sum())
        assert target * 3 // 4 <= total <= target

        for nid in range(tree.node_count()):
            assert tree.get_virtual_loss(nid) == 0, f"vl leaked on {nid}"
    finally:
        pool.close()


def test_pool_zero_target_is_noop() -> None:
    ev = _make_evaluator()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=1, gather=8),
        evaluators=[ev],
    )
    try:
        tree, rid, cb = _seed_tree()
        pre = tree.node_count()
        pool.run(tree=tree, root_id=rid, root_cboard=cb,
                 target_sims=0, stop_event=threading.Event())
        assert tree.node_count() == pre
    finally:
        pool.close()


def test_pool_rejects_single_slot_evaluator() -> None:
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    bad = DirectGPUEvaluator(
        model, device="cpu", max_batch=8, use_amp=False, n_slots=1,
    )
    with pytest.raises(ValueError, match="n_slots"):
        MultiGpuPucvPool(
            MultiGpuPucvConfig(n_gpus=1, gather=8),
            evaluators=[ThreadSafeGPUDispatcher(bad)],
        )


def test_pool_rejects_evaluator_count_mismatch() -> None:
    ev = _make_evaluator()
    with pytest.raises(ValueError, match="need 2 evaluators"):
        MultiGpuPucvPool(
            MultiGpuPucvConfig(n_gpus=2, gather=8),
            evaluators=[ev],
        )


def test_searchworker_install_multi_gpu_pucv_produces_bestmove() -> None:
    """Smoke: SearchWorker.install_multi_gpu_pucv routes search through the
    pool. Bestmove is non-empty and visit count meaningful."""
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
  # Two pool evaluators (independent of `primary`, used only for root-eval).
    p0 = _make_evaluator(max_batch=64)
    p1 = _make_evaluator(max_batch=64)
    worker.install_multi_gpu_pucv([p0, p1], gather=8, as_factories=False)
    deadline = Deadline(2_000)
    result = worker.run(chess.Board(), stop_event=threading.Event(),
                        deadline=deadline, max_nodes=64)
    assert len(result.bestmove_uci) >= 4
    assert result.nodes >= 32
    worker.close()


def test_searchworker_clear_multi_gpu_pucv_reverts() -> None:
    """install_multi_gpu_pucv → clear_multi_gpu_pucv must drop the pool and
    leave subsequent searches running through the gumbel/walker path."""
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    p0 = _make_evaluator(max_batch=64)
    worker.install_multi_gpu_pucv([p0], gather=8, as_factories=False)
    assert worker._pucv_pool is not None
    worker.clear_multi_gpu_pucv()
    assert worker._pucv_pool is None
    deadline = Deadline(2_000)
    result = worker.run(chess.Board(), stop_event=threading.Event(),
                        deadline=deadline, max_nodes=64)
    assert result.bestmove_uci
    worker.close()
