"""Parity + behavior tests for ThreadedDispatcher.

Cudagraph-validity (single GPU consumer thread) requires a real GPU + compile,
so it's covered by integration smoke (worker subprocess + ``frames_ok``), not
here. These tests pin numerical parity and the public API contract.
"""
from __future__ import annotations

import concurrent.futures
import threading

import numpy as np
import pytest
import torch

from chess_anti_engine.inference import DirectGPUEvaluator, ThreadedBatchEvaluator
from chess_anti_engine.inference_threaded import ThreadedDispatcher, _next_bucket
from chess_anti_engine.model import ModelConfig, build_model


def _make_model() -> torch.nn.Module:
    cfg = ModelConfig(embed_dim=32, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    return model


def _rand_batch(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal((n, 146, 8, 8), dtype=np.float32)


def test_next_bucket_picks_smallest_fit():
    assert _next_bucket(1) == 128
    assert _next_bucket(128) == 128
    assert _next_bucket(129) == 256
    assert _next_bucket(700) == 768
    # Oversize falls back to the max bucket; caller is expected to split.
    assert _next_bucket(99999) == 4096


def test_dispatcher_matches_direct_evaluator():
    model = _make_model()
    direct = DirectGPUEvaluator(model, device="cpu", max_batch=512, use_amp=False)
    dispatcher = ThreadedDispatcher(model, device="cpu", max_batch=512, batch_wait_ms=0.0)
    try:
        rng = np.random.default_rng(0)
        for n in (1, 5, 64, 200):
            x = _rand_batch(rng, n)
            pol_d, wdl_d = direct.evaluate_encoded(x)
            pol_t, wdl_t = dispatcher.evaluate_encoded(x)
            np.testing.assert_allclose(pol_d, pol_t, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(wdl_d, wdl_t, rtol=1e-5, atol=1e-5)
    finally:
        dispatcher.shutdown()


def test_dispatcher_matches_threaded_batch_evaluator():
    model = _make_model()
    tbe = ThreadedBatchEvaluator(model, device="cpu", max_batch=512, min_batch=1)
    dispatcher = ThreadedDispatcher(model, device="cpu", max_batch=512, batch_wait_ms=0.0)
    try:
        rng = np.random.default_rng(7)
        for n in (1, 32, 100):
            x = _rand_batch(rng, n)
            pol_a, wdl_a = tbe.evaluate_encoded(x)
            pol_b, wdl_b = dispatcher.evaluate_encoded(x)
            np.testing.assert_allclose(pol_a, pol_b, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(wdl_a, wdl_b, rtol=1e-4, atol=1e-4)
    finally:
        tbe.shutdown()
        dispatcher.shutdown()


def test_dispatcher_concurrent_producers_correct_per_caller():
    model = _make_model()
    direct_baseline = DirectGPUEvaluator(model, device="cpu", max_batch=512, use_amp=False)
    dispatcher = ThreadedDispatcher(model, device="cpu", max_batch=512, batch_wait_ms=2.0)
    try:
        rng = np.random.default_rng(42)
        n_threads = 8
        inputs = [_rand_batch(rng, 17 + i) for i in range(n_threads)]
        baselines = [direct_baseline.evaluate_encoded(x) for x in inputs]

        results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n_threads
        # Barrier so producers fire together — otherwise the dispatcher drains
        # them one-at-a-time and we don't actually exercise cross-thread batching.
        barrier = threading.Barrier(n_threads)

        def producer(i: int) -> None:
            barrier.wait()
            results[i] = dispatcher.evaluate_encoded(inputs[i])

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, r in enumerate(results):
            assert r is not None
            np.testing.assert_allclose(r[0], baselines[i][0], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(r[1], baselines[i][1], rtol=1e-5, atol=1e-5)

        assert dispatcher.stats["lifetime_batches"] >= 1
        assert dispatcher.stats["avg_batch_size"] > 0
    finally:
        dispatcher.shutdown()


def test_dispatcher_oversize_submission_raises():
    model = _make_model()
    dispatcher = ThreadedDispatcher(model, device="cpu", max_batch=128, batch_wait_ms=0.0)
    try:
        rng = np.random.default_rng(0)
        x = _rand_batch(rng, 200)
        with pytest.raises(ValueError, match="max_batch"):
            dispatcher.evaluate(x)
    finally:
        dispatcher.shutdown()


def test_dispatcher_update_model_swaps_weights():
    model_a = _make_model()
    model_b = _make_model()
    with torch.no_grad():
        for p in model_b.parameters():
            p.mul_(0.5)

    dispatcher = ThreadedDispatcher(model_a, device="cpu", max_batch=128, batch_wait_ms=0.0)
    try:
        rng = np.random.default_rng(1)
        x = _rand_batch(rng, 4)
        pol_a, _ = dispatcher.evaluate_encoded(x)
        dispatcher.update_model(model_b)
        pol_b, _ = dispatcher.evaluate_encoded(x)
        assert not np.allclose(pol_a, pol_b, rtol=1e-3, atol=1e-3)
    finally:
        dispatcher.shutdown()


def test_dispatcher_update_model_preserves_compiled_model(monkeypatch):
    calls: list[tuple[torch.nn.Module, str, str]] = []

    def fake_compile(model: torch.nn.Module, *, mode: str) -> torch.nn.Module:
        calls.append((model, mode, threading.current_thread().name))
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)

    model_a = _make_model()
    model_b = _make_model()
    dispatcher = ThreadedDispatcher(
        model_a,
        device="cpu",
        max_batch=128,
        batch_wait_ms=0.0,
        compile_mode="reduce-overhead",
    )
    try:
        rng = np.random.default_rng(1)
        dispatcher.evaluate_encoded(_rand_batch(rng, 4))
        dispatcher.update_model(model_b)

        assert calls == [(model_a, "reduce-overhead", "ThreadedDispatcher")]
        for actual, expected in zip(model_a.parameters(), model_b.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
    finally:
        dispatcher.shutdown()


def test_dispatcher_evaluate_returns_future():
    model = _make_model()
    dispatcher = ThreadedDispatcher(model, device="cpu", max_batch=128, batch_wait_ms=0.0)
    try:
        rng = np.random.default_rng(0)
        x = _rand_batch(rng, 4)
        fut = dispatcher.evaluate(x)
        assert isinstance(fut, concurrent.futures.Future)
        pol, wdl = fut.result(timeout=10.0)
        assert pol.shape == (4, 4672)
        assert wdl.shape == (4, 3)
    finally:
        dispatcher.shutdown()
