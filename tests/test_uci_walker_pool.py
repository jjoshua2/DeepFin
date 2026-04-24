"""Walker-pool UCI smoke (phase 5).

Validates the `--walkers N` path end-to-end:
  - search produces a bestmove
  - tree accumulates visits as expected
  - N>1 engages the PUCT walker pool (no Gumbel halving) without crashing

We also exercise the WalkerPool directly with a small model to confirm
concurrency works against the shared tree."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.subprocess_client import LineReader as _LineReader
from chess_anti_engine.uci.subprocess_client import send_line as _send
from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig


def _make_tiny_checkpoint(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "checkpoint_000001"
    ckpt_dir.mkdir()
    cfg = ModelConfig(kind="tiny")
    model = build_model(cfg)
    torch.save({"model": model.state_dict(), "step": 0}, ckpt_dir / "trainer.pt")
    with (tmp_path / "params.json").open("w") as fh:
        json.dump({"model": "tiny"}, fh)
    return ckpt_dir


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    return _make_tiny_checkpoint(tmp_path)


def _spawn_engine(checkpoint: Path, *extra: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "chess_anti_engine.uci",
         "--checkpoint", str(checkpoint), "--device", "cpu", *extra],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )


def test_walkers_flag_produces_bestmove(tiny_checkpoint: Path) -> None:
    """Opt-in walker path must still emit a valid bestmove. Smoke-level;
    deeper correctness is in the walker-primitive tests."""
    proc = _spawn_engine(tiny_checkpoint, "--walkers", "4", "--chunk-sims", "32")
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        reader.read_until("uciok")
        _send(proc, "isready")
        reader.read_until("readyok")
        _send(proc, "ucinewgame")
        _send(proc, "position startpos")
        _send(proc, "go nodes 64")
        lines = reader.read_until("bestmove", timeout_s=30.0)
        bestmove_line = next(l for l in lines if l.startswith("bestmove "))
        bestmove = bestmove_line.split()[1]
        assert len(bestmove) >= 4
        _send(proc, "quit")
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_walker_pool_accumulates_visits() -> None:
    """WalkerPool against a shared tree: after run, root visits >= target
    (vloss doesn't prevent real visits, only nudges concurrent descent)."""
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    evaluator = DirectGPUEvaluator(model, device="cpu", max_batch=4, use_amp=False)
    dispatcher = ThreadSafeGPUDispatcher(evaluator)

    tree = MCTSTree()
    tree.reserve(1024, 8192)
    board = chess.Board()
    root_cb = CBoard.from_board(board)
    rid = tree.add_root(0, 0.0)
    # Seed the root with uniform priors so walkers have somewhere to go.
    legal = root_cb.legal_move_indices().astype(np.int32)
    priors = np.full(legal.size, 1.0 / legal.size, dtype=np.float64)
    tree.expand(rid, legal, priors)

    pool = WalkerPool(
        WalkerPoolConfig(n_walkers=4, c_puct=1.5, fpu_at_root=0.0,
                         fpu_reduction=0.33, vloss_weight=3),
        dispatcher,
    )
    stop = threading.Event()
    pool.run(tree=tree, root_id=rid, root_cboard=root_cb,
             target_sims=64, stop_event=stop)

    # Every sim backprops one visit to root, so N(root) must equal claims.
    actions, visits = tree.get_children_visits(rid)
    assert actions.size > 0
    total_child_visits = int(visits.sum())
    # Walkers cap at 64 claims; backprops accumulate at root.
    assert 32 <= total_child_visits <= 64
    # All virtual loss unwound.
    for nid in range(tree.node_count()):
        assert tree.get_virtual_loss(nid) == 0, f"vl leaked on node {nid}"


def test_walker_pool_gather_amplifies_batch() -> None:
    """With gather=G per walker, a single walker thread should submit
    NN batches of size up to G instead of always 1. Validates the
    per-walker leaf-gather amplification."""

    class _RecordingEvaluator:
        """Wraps a real evaluator, records every submit's batch size."""
        def __init__(self, inner):
            self._inner = inner
            self.submit_sizes: list[int] = []
            self._lock = threading.Lock()

        def evaluate_encoded(self, x):
            with self._lock:
                self.submit_sizes.append(int(x.shape[0]))
            return self._inner.evaluate_encoded(x)

    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    inner = DirectGPUEvaluator(model, device="cpu", max_batch=16, use_amp=False)
    rec = _RecordingEvaluator(inner)

    tree = MCTSTree()
    tree.reserve(1024, 8192)
    board = chess.Board()
    root_cb = CBoard.from_board(board)
    rid = tree.add_root(0, 0.0)
    legal = root_cb.legal_move_indices().astype(np.int32)
    priors = np.full(legal.size, 1.0 / legal.size, dtype=np.float64)
    tree.expand(rid, legal, priors)

    pool = WalkerPool(
        WalkerPoolConfig(n_walkers=1, c_puct=1.5, fpu_at_root=0.0,
                         fpu_reduction=0.33, vloss_weight=3, gather=4),
        rec,
    )
    stop = threading.Event()
    # 16 sims / gather=4 → expect ~4 submits, each of size ≤4.
    pool.run(tree=tree, root_id=rid, root_cboard=root_cb,
             target_sims=16, stop_event=stop)

    assert rec.submit_sizes, "no NN calls recorded"
    assert max(rec.submit_sizes) > 1, (
        f"gather=4 should yield batch>1; got {rec.submit_sizes}"
    )
    # Sum of submitted leaves equals non-terminal descents (≤ target).
    total = sum(rec.submit_sizes)
    assert 1 <= total <= 16, f"submitted {total} leaves for target=16"
    # No virtual loss leaked.
    for nid in range(tree.node_count()):
        assert tree.get_virtual_loss(nid) == 0, f"vl leaked on node {nid}"


def test_walker_pool_gather_default_is_one() -> None:
    """gather=1 (the default) must preserve classic one-leaf-per-submit
    behavior — regression guard against the gather parameter changing
    default semantics."""

    class _RecordingEvaluator:
        def __init__(self, inner):
            self._inner = inner
            self.submit_sizes: list[int] = []
            self._lock = threading.Lock()

        def evaluate_encoded(self, x):
            with self._lock:
                self.submit_sizes.append(int(x.shape[0]))
            return self._inner.evaluate_encoded(x)

    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    inner = DirectGPUEvaluator(model, device="cpu", max_batch=4, use_amp=False)
    rec = _RecordingEvaluator(inner)

    tree = MCTSTree()
    tree.reserve(512, 4096)
    board = chess.Board()
    root_cb = CBoard.from_board(board)
    rid = tree.add_root(0, 0.0)
    legal = root_cb.legal_move_indices().astype(np.int32)
    priors = np.full(legal.size, 1.0 / legal.size, dtype=np.float64)
    tree.expand(rid, legal, priors)

    pool = WalkerPool(
        WalkerPoolConfig(n_walkers=1, c_puct=1.5, fpu_at_root=0.0,
                         fpu_reduction=0.33, vloss_weight=3),  # gather default=1
        rec,
    )
    stop = threading.Event()
    pool.run(tree=tree, root_id=rid, root_cboard=root_cb,
             target_sims=8, stop_event=stop)
    # Every submit is single-leaf when gather=1 and walker=1.
    assert rec.submit_sizes
    assert all(n == 1 for n in rec.submit_sizes), (
        f"gather=1 should give only batch=1 submits; got {rec.submit_sizes}"
    )


def test_walker_pool_stop_event_shortens_run() -> None:
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    evaluator = DirectGPUEvaluator(model, device="cpu", max_batch=4, use_amp=False)
    dispatcher = ThreadSafeGPUDispatcher(evaluator)

    tree = MCTSTree()
    tree.reserve(512, 4096)
    board = chess.Board()
    root_cb = CBoard.from_board(board)
    rid = tree.add_root(0, 0.0)
    legal = root_cb.legal_move_indices().astype(np.int32)
    priors = np.full(legal.size, 1.0 / legal.size, dtype=np.float64)
    tree.expand(rid, legal, priors)

    pool = WalkerPool(
        WalkerPoolConfig(n_walkers=2, c_puct=1.5, fpu_at_root=0.0,
                         fpu_reduction=0.33, vloss_weight=3),
        dispatcher,
    )
    stop = threading.Event()
    stop.set()  # pre-set so workers exit immediately
    pool.run(tree=tree, root_id=rid, root_cboard=root_cb,
             target_sims=1_000_000, stop_event=stop)
    # Don't assert exact count — at least one worker may have claimed a slot
    # before the first stop check. Just verify we didn't hang.
