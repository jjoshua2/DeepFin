"""Thread-safe GPU dispatcher (Phase 1 of walker-pool plan).

Verifies the ``ThreadSafeGPUDispatcher`` lock wrapper: multiple callers
concurrently submitting batches must not corrupt shared pinned buffers and
results must match the sequential-call baseline.
"""
from __future__ import annotations

import threading

import numpy as np
import torch

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.model import ModelConfig, build_model


def _make_evaluator() -> DirectGPUEvaluator:
    cfg = ModelConfig(embed_dim=32, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    return DirectGPUEvaluator(model, device="cpu", max_batch=16, use_amp=False)


def test_dispatcher_proxies_evaluate_encoded():
    evaluator = _make_evaluator()
    dispatcher = ThreadSafeGPUDispatcher(evaluator)

    x = np.random.default_rng(0).standard_normal((2, 146, 8, 8), dtype=np.float32)
    pol, wdl = dispatcher.evaluate_encoded(x)

    assert pol.shape == (2, 4672)
    assert wdl.shape == (2, 3)


def test_dispatcher_concurrent_callers_get_correct_results():
    evaluator = _make_evaluator()
    dispatcher = ThreadSafeGPUDispatcher(evaluator)

    # Per-thread deterministic inputs that differ, so we can detect buffer
    # corruption: if two threads share a pinned buffer without the lock,
    # their outputs would get swapped or mangled.
    rng = np.random.default_rng(42)
    inputs = [rng.standard_normal((4, 146, 8, 8), dtype=np.float32) for _ in range(8)]

    # Baseline: serial eval, one at a time.
    with torch.no_grad():
        baseline = [dispatcher.evaluate_encoded(x) for x in inputs]

    # Concurrent: 8 threads each call the dispatcher N times with the same input.
    results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(inputs)
    barrier = threading.Barrier(len(inputs))

    def worker(i: int) -> None:
        barrier.wait()
        for _ in range(5):
            pol, wdl = dispatcher.evaluate_encoded(inputs[i])
            # Each iteration must match the baseline — lock ensures no cross-
            # thread buffer contamination.
            np.testing.assert_allclose(pol, baseline[i][0], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(wdl, baseline[i][1], rtol=1e-5, atol=1e-5)
        results[i] = (pol, wdl)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(inputs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for r in results:
        assert r is not None


def test_dispatcher_hides_async_attribute():
    # gumbel_c.py gates its pipelined path on hasattr(eval, 'evaluate_encoded_async').
    # Dispatchers deliberately don't expose it (no __getattr__ proxy) so the
    # pipelined path never activates through a wrapped evaluator — that path
    # would bypass our serialization/coalescing logic.
    evaluator = _make_evaluator()
    dispatcher = ThreadSafeGPUDispatcher(evaluator)
    assert not hasattr(dispatcher, "evaluate_encoded_async")
    assert hasattr(dispatcher, "evaluate_encoded")


def test_dispatcher_exposes_max_batch():
    evaluator = _make_evaluator()
    dispatcher = ThreadSafeGPUDispatcher(evaluator)
    assert dispatcher.max_batch == 16
