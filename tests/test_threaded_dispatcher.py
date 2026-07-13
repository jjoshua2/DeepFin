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
    cfg = ModelConfig(
        embed_dim=32, num_layers=1, num_heads=2, ffn_mult=2.0,
        input_extra_features="v1",
    )
    model = build_model(cfg)
    model.eval()
    return model


def _rand_batch(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal((n, 146, 8, 8), dtype=np.float32)


def test_next_bucket_picks_smallest_fit():
    assert _next_bucket(1) == 128
    assert _next_bucket(128) == 128
    assert _next_bucket(129) == 170
    assert _next_bucket(170) == 170
    assert _next_bucket(171) == 256
    assert _next_bucket(340) == 340
    assert _next_bucket(681) == 768
    assert _next_bucket(700) == 768
    assert _next_bucket(1190) == 1190
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
        assert dispatcher.stats["lifetime_requests"] == n_threads
        assert dispatcher.stats["avg_requests_per_batch"] > 0
        assert dispatcher.stats["avg_rows_per_request"] > 0
        assert dispatcher.stats["avg_queue_depth"] > 0
        assert dispatcher.stats["queue_depth_max"] > 0
        assert dispatcher.stats["avg_drain_ms"] >= 0
        assert dispatcher.stats["avg_pack_ms"] >= 0
        assert dispatcher.stats["avg_submit_ms"] >= 0
        assert dispatcher.stats["avg_wait_ms"] >= 0
        assert dispatcher.stats["avg_scatter_ms"] >= 0
    finally:
        dispatcher.shutdown()


def test_dispatcher_target_batch_limits_normal_drain():
    model = _make_model()
    dispatcher = ThreadedDispatcher(
        model,
        device="cpu",
        max_batch=512,
        target_batch=170,
        batch_wait_ms=2.0,
    )
    try:
        rng = np.random.default_rng(123)
        n_threads = 8
        inputs = [_rand_batch(rng, 80) for _ in range(n_threads)]
        barrier = threading.Barrier(n_threads)

        def producer(i: int) -> tuple[np.ndarray, np.ndarray]:
            barrier.wait()
            return dispatcher.evaluate_encoded(inputs[i])

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
            results = list(pool.map(producer, range(n_threads)))

        assert len(results) == n_threads
        assert dispatcher.stats["max_batch"] == 512
        assert dispatcher.stats["target_batch"] == 170
        assert dispatcher.stats["lifetime_batches"] >= 4
        assert dispatcher.stats["avg_batch_size"] <= 170
    finally:
        dispatcher.shutdown()


def test_dispatcher_legal_bf16_matches_dense_legal_logits():
    model = _make_model()
    direct = DirectGPUEvaluator(model, device="cpu", max_batch=128, use_amp=False)
    dispatcher = ThreadedDispatcher(model, device="cpu", max_batch=128, batch_wait_ms=0.0)
    try:
        rng = np.random.default_rng(99)
        x = _rand_batch(rng, 5)
        legal_counts = np.array([2, 3, 1, 4, 2], dtype=np.int32)
        legal_flat = np.array([0, 7, 1, 64, 200, 42, 3, 5, 8, 13, 21, 34], dtype=np.int32)

        dense_pol, dense_wdl = direct.evaluate_encoded(x)
        rows = np.repeat(np.arange(x.shape[0]), legal_counts)
        expected = torch.from_numpy(dense_pol[rows, legal_flat]).to(torch.bfloat16).view(torch.uint16).numpy()

        compact_pol, compact_wdl = dispatcher.evaluate_legal_bf16(x, legal_flat, legal_counts)

        assert compact_pol.dtype == np.uint16
        np.testing.assert_array_equal(compact_pol, expected)
        np.testing.assert_allclose(compact_wdl, dense_wdl, rtol=1e-5, atol=1e-5)
    finally:
        dispatcher.shutdown()


def test_dispatcher_coalesced_legal_bf16_preserves_request_slices():
    model = _make_model()
    direct = DirectGPUEvaluator(model, device="cpu", max_batch=128, use_amp=False)
    dispatcher = ThreadedDispatcher(model, device="cpu", max_batch=128, batch_wait_ms=20.0)
    try:
        rng = np.random.default_rng(20260712)
        row_counts = (2, 3, 5, 7)
        inputs = [_rand_batch(rng, rows) for rows in row_counts]
        legal_counts = [
            rng.integers(1, 9, size=rows, dtype=np.int32) for rows in row_counts
        ]
        legal_flat = [
            rng.integers(0, 4672, size=int(counts.sum()), dtype=np.int32)
            for counts in legal_counts
        ]
        expected: list[tuple[np.ndarray, np.ndarray]] = []
        for x, flat, counts in zip(inputs, legal_flat, legal_counts, strict=True):
            dense_pol, dense_wdl = direct.evaluate_encoded(x)
            rows = np.repeat(np.arange(x.shape[0]), counts)
            compact_pol = (
                torch.from_numpy(dense_pol[rows, flat])
                .to(torch.bfloat16)
                .view(torch.uint16)
                .numpy()
            )
            expected.append((compact_pol, dense_wdl))

        barrier = threading.Barrier(len(inputs))

        def producer(index: int) -> tuple[np.ndarray, np.ndarray]:
            barrier.wait()
            return dispatcher.evaluate_legal_bf16(
                inputs[index], legal_flat[index], legal_counts[index],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(inputs)) as pool:
            actual = list(pool.map(producer, range(len(inputs))))

        for result, baseline in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(result[0], baseline[0])
            np.testing.assert_allclose(result[1], baseline[1], rtol=1e-5, atol=1e-5)
        assert dispatcher.stats["avg_requests_per_batch"] > 1
    finally:
        dispatcher.shutdown()


def test_dispatcher_decodes_bf16_input_when_full_input_bf16_disabled():
    model = _make_model()
    direct = DirectGPUEvaluator(model, device="cpu", max_batch=128, use_amp=False)
    dispatcher = ThreadedDispatcher(model, device="cpu", max_batch=128, batch_wait_ms=0.0)
    try:
        rng = np.random.default_rng(100)
        x = _rand_batch(rng, 6)
        # Simulate a C encoder that has already rounded planes to bfloat16.
        x_bf16_bits = torch.from_numpy(x).to(torch.bfloat16).view(torch.uint16).numpy()
        x_decoded = (x_bf16_bits.astype(np.uint32) << 16).view(np.float32)

        expected_pol, expected_wdl = direct.evaluate_encoded(x_decoded)
        actual_pol, actual_wdl = dispatcher.evaluate_encoded(x_bf16_bits)

        np.testing.assert_allclose(actual_pol, expected_pol, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual_wdl, expected_wdl, rtol=1e-5, atol=1e-5)
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
