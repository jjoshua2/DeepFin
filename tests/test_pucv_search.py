"""PucvChunker integration smoke + SearchWorker --use-pucv wiring.

The bench in /tmp/bench_pucv_async.py validated raw throughput; these
tests confirm correctness:
  - pucv.run accumulates the requested visits at root
  - vloss is fully unwound after run (no leaked virtual losses)
  - n_slots<2 evaluator raises (preflight)
  - SearchWorker.set_use_pucv(True) routes through pucv
"""
from __future__ import annotations

import threading

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.puct_vl import PucvChunker
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


def _make_dispatcher(max_batch: int = 64):
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    inner = DirectGPUEvaluator(
        model, device="cpu", max_batch=max_batch, use_amp=False, n_slots=2,
    )
    return ThreadSafeGPUDispatcher(inner)


def _seed_tree(dispatcher, gather: int = 16) -> tuple[MCTSTree, int, CBoard, PucvChunker]:
    tree = MCTSTree()
    tree.reserve(1024, 8192)
    board = chess.Board()
    cb = CBoard.from_board(board)
  # Seed root with uniform priors so the chunker can descend.
    rid = tree.add_root(0, 0.0)
    legal = cb.legal_move_indices().astype(np.int32)
    priors = np.full(legal.size, 1.0 / legal.size, dtype=np.float64)
    tree.expand(rid, legal, priors)
    chunker = PucvChunker(dispatcher, gather=gather, c_puct=1.4)
    return tree, rid, cb, chunker


def test_pucv_accumulates_visits_at_root() -> None:
    """After run(target_sims=N), root visits should approach N (some may be
    spent on terminal leaves which don't take the integrate path)."""
    dispatcher = _make_dispatcher()
    tree, rid, cb, chunker = _seed_tree(dispatcher, gather=8)

    target = 32
    chunker.run(tree=tree, root_id=rid, root_cboard=cb, target_sims=target)

    actions, visits = tree.get_children_visits(rid)
    assert actions.size > 0
    total = int(visits.sum())
    assert total >= target * 3 // 4, (
        f"expected ≥{target * 3 // 4} child visits, got {total}"
    )


def test_pucv_no_vloss_leak() -> None:
    """Every batch_descend_puct that applies vloss must have a matching
    batch_integrate_leaves that removes it. After run(), all nodes must
    have virtual_loss == 0."""
    dispatcher = _make_dispatcher()
    tree, rid, cb, chunker = _seed_tree(dispatcher, gather=8)

    chunker.run(tree=tree, root_id=rid, root_cboard=cb, target_sims=64)

    for nid in range(tree.node_count()):
        assert tree.get_virtual_loss(nid) == 0, f"vl leaked on node {nid}"


def test_pucv_rejects_single_slot_evaluator() -> None:
    """n_slots=1 evaluators can't pipeline — preflight catch."""
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    inner = DirectGPUEvaluator(
        model, device="cpu", max_batch=16, use_amp=False, n_slots=1,
    )
    with pytest.raises(ValueError, match="n_slots"):
        PucvChunker(inner, gather=8)


def test_pucv_zero_target_is_noop() -> None:
    """target_sims=0 must return immediately without touching the tree."""
    dispatcher = _make_dispatcher()
    tree, rid, cb, chunker = _seed_tree(dispatcher, gather=8)
    pre_count = tree.node_count()

    chunker.run(tree=tree, root_id=rid, root_cboard=cb, target_sims=0)

    assert tree.node_count() == pre_count


def test_searchworker_use_pucv_produces_bestmove() -> None:
    """SearchWorker.set_use_pucv(True) routes through PucvChunker and still
    emits a sensible bestmove. Smoke-level — full nps measured separately."""
    dispatcher = _make_dispatcher(max_batch=128)
    worker = SearchWorker(
        dispatcher,
        device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64,
        n_walkers=1,
    )
    worker.set_use_pucv(True, gather=16)
    stop = threading.Event()
    deadline = Deadline(2_000)
    result = worker.run(chess.Board(), stop_event=stop, deadline=deadline,
                        max_nodes=64)
    assert result.bestmove_uci, "no bestmove emitted"
    assert len(result.bestmove_uci) >= 4
    assert result.nodes >= 32
    worker.close()


def test_searchworker_use_pucv_silently_disables_with_threads_gt_1() -> None:
    """Threads>1 + UseVL=on → pucv silently inactive (returns None from
    _build_pucv). Walker pool path runs instead. No crash, no error."""
    dispatcher = _make_dispatcher(max_batch=128)
    worker = SearchWorker(
        dispatcher,
        device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64,
        n_walkers=2,  # walker pool engaged
    )
    worker.set_use_pucv(True, gather=16)
    assert worker._pucv is None, "pucv must be disabled when n_walkers != 1"
    stop = threading.Event()
    deadline = Deadline(2_000)
    result = worker.run(chess.Board(), stop_event=stop, deadline=deadline,
                        max_nodes=64)
    assert result.bestmove_uci
    worker.close()
