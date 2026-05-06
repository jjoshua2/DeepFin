"""Test ThreadedBatchEvaluator with concurrent selfplay threads."""
from __future__ import annotations

import threading

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.inference import ThreadedBatchEvaluator
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model

try:
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c as _gumbel_fn
except ImportError:
    from chess_anti_engine.mcts.gumbel import run_gumbel_root_many as _gumbel_fn


@pytest.fixture(scope="module")
def evaluator():
    model = build_model(ModelConfig(embed_dim=384, num_layers=9, num_heads=6))
    if torch.cuda.is_available():
        model = model.cuda().eval()
        device = "cuda"
    else:
        model = model.eval()
        device = "cpu"
    ev = ThreadedBatchEvaluator(model, device=device, max_batch=2048, min_batch=32)
    yield ev
    ev.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_thread(evaluator):
    """Basic sanity: one thread produces valid actions."""
    boards = [chess.Board() for _ in range(4)]
    rng = np.random.default_rng(42)
    _probs, actions, _values, _masks, *_ = _gumbel_fn(
        None, boards, device="cuda", evaluator=evaluator,
        cfg=GumbelConfig(simulations=10, topk=8), rng=rng,
    )
    assert len(actions) == 4
    assert all(a is not None for a in actions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_concurrent_threads(evaluator):
    """Multiple threads produce valid, independent results."""
    n_threads = 4
    results: list[list[object | None] | None] = [None] * n_threads
    errors: list[str | None] = [None] * n_threads

    def run_thread(tid: int) -> None:
        try:
            boards = [chess.Board() for _ in range(4)]
            rng = np.random.default_rng(100 + tid)
            _probs, actions, _values, _masks, *_ = _gumbel_fn(
                None, boards, device="cuda", evaluator=evaluator,
                cfg=GumbelConfig(simulations=10, topk=8), rng=rng,
            )
            results[tid] = list(actions)
        except Exception as e:
            errors[tid] = str(e)

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    for i in range(n_threads):
        assert errors[i] is None, f"Thread {i} error: {errors[i]}"
        result = results[i]
        assert result is not None, f"Thread {i} produced no result"
        assert len(result) == 4
        assert all(a is not None for a in result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_model_update(evaluator):
    """Model can be swapped without hanging or crashing."""
    model2 = build_model(ModelConfig(embed_dim=384, num_layers=9, num_heads=6)).cuda().eval()
    evaluator.update_model(model2)
    # Verify inference still works after swap
    boards = [chess.Board() for _ in range(2)]
    rng = np.random.default_rng(99)
    _probs, actions, _values, _masks, *_ = _gumbel_fn(
        None, boards, device="cuda", evaluator=evaluator,
        cfg=GumbelConfig(simulations=5, topk=4), rng=rng,
    )
    assert len(actions) == 2
