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

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_threaded import ThreadedDispatcher, _next_bucket
from chess_anti_engine.model import ModelConfig, build_model


def _make_model(seed: int = 0) -> torch.nn.Module:
    cfg = ModelConfig(
        embed_dim=32, num_layers=1, num_heads=2, ffn_mult=2.0,
        input_extra_features="v1",
    )
    # Seed the init draw, forked so this file neither depends on nor perturbs
    # the process-global torch RNG. Unseeded, the weights are whatever state
    # earlier tests in the suite happened to leave behind, which makes every
    # numeric assertion below a lottery re-rolled by test ORDER — a test that
    # passes alone and fails in the full run, with no local change to blame.
    # Callers that need two DIFFERENT nets must pass different seeds.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_model(cfg)
    model.eval()
    return model


def _rand_batch(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal((n, 146, 8, 8), dtype=np.float32)


def test_next_bucket_picks_smallest_fit():
    assert _next_bucket(1) == 16
    assert _next_bucket(16) == 16
    assert _next_bucket(17) == 32
    assert _next_bucket(32) == 32
    assert _next_bucket(33) == 64
    assert _next_bucket(64) == 64
    assert _next_bucket(65) == 96
    assert _next_bucket(96) == 96
    assert _next_bucket(97) == 128
    assert _next_bucket(128) == 128
    assert _next_bucket(129) == 170
    assert _next_bucket(170) == 170
    assert _next_bucket(171) == 256
    assert _next_bucket(340) == 340
    assert _next_bucket(513) == 576
    assert _next_bucket(576) == 576
    assert _next_bucket(577) == 680
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
        # Baseline: the SAME compact path, one request per submit. It must NOT
        # be the dense evaluate_encoded gather. Every compact submit pads the
        # forward to a bucket (>= 16 rows) while evaluate_encoded runs it at
        # the exact row count, and the two shapes select different CPU GEMM
        # kernels — real rows come out ~1e-7 apart (padding CONTENT is proven
        # irrelevant: NaN/Inf in the pad rows leaves real rows bit-identical).
        # That gap is far below any tolerance worth having, but bf16
        # quantisation turns it into a 1-ULP bit flip on near-zero logits, and
        # exact bit equality reports that as a failure — on ~0.7% of weight
        # draws. Compact-vs-dense agreement is the sibling test's job
        # (test_dispatcher_legal_bf16_matches_dense_legal_logits); this test's
        # job is that a COALESCED batch hands each caller its own slice, so the
        # single-request compact result is the right thing to compare against.
        expected = [
            direct.evaluate_legal_bf16(x, flat, counts)
            for x, flat, counts in zip(inputs, legal_flat, legal_counts, strict=True)
        ]

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
    model_a = _make_model(seed=0)
    model_b = _make_model(seed=1)
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

    # Distinct seeds: the post-update parameter comparison below is only a gate
    # if the two nets started out different.
    model_a = _make_model(seed=0)
    model_b = _make_model(seed=1)
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


def test_bucket_ladder_clamped_to_off_ladder_max_batch():
    dispatcher = ThreadedDispatcher(_make_model(), device="cpu", max_batch=1000)
    try:
        assert dispatcher._buckets[-1] == 1000
        assert all(b <= 1000 for b in dispatcher._buckets)
        # (768, 1000] must round to 1000, not the unclamped 1020 bucket.
        assert _next_bucket(800, dispatcher._buckets) == 1000
        assert _next_bucket(1000, dispatcher._buckets) == 1000
        assert _next_bucket(768, dispatcher._buckets) == 768
    finally:
        dispatcher.shutdown()


def test_bucket_ladder_on_ladder_max_batch_unchanged():
    dispatcher = ThreadedDispatcher(_make_model(), device="cpu", max_batch=576)
    try:
        assert dispatcher._buckets[-1] == 576
        assert _next_bucket(513, dispatcher._buckets) == 576
        assert _next_bucket(576, dispatcher._buckets) == 576
        assert _next_bucket(1, dispatcher._buckets) == 16
    finally:
        dispatcher.shutdown()


def test_dispatcher_forward_padded_to_clamped_bucket():
    # max_batch=20 is off-ladder: an 18-row submit must pad to 20 (the
    # appended max_batch bucket), never round up past max_batch.
    dispatcher = ThreadedDispatcher(_make_model(), device="cpu", max_batch=20)
    try:
        rng = np.random.default_rng(0)
        pol, wdl = dispatcher.evaluate_encoded(_rand_batch(rng, 18))
        assert pol.shape == (18, 4672)
        assert wdl.shape == (18, 3)
        assert dispatcher.stats["lifetime_forward_rows"] == 20
    finally:
        dispatcher.shutdown()


def test_dispatcher_fatal_startup_error_fails_producers_fast():
    class _BoomDispatcher(ThreadedDispatcher):
        def _compile_model_on_dispatcher(
            self, model: torch.nn.Module,
        ) -> torch.nn.Module:
            del model
            raise RuntimeError("compile boom")

    dispatcher = _BoomDispatcher(_make_model(), device="cpu", max_batch=64)
    dispatcher._thread.join(timeout=10.0)
    assert not dispatcher._thread.is_alive()
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="thread died"):
        dispatcher.evaluate_encoded(_rand_batch(rng, 2))
    with pytest.raises(RuntimeError, match="thread died"):
        dispatcher.update_model(_make_model())
    assert isinstance(dispatcher._fatal_exc, RuntimeError)
    assert "compile boom" in str(dispatcher._fatal_exc)


def test_dispatcher_poison_rejects_new_submits():
    dispatcher = ThreadedDispatcher(_make_model(), device="cpu", max_batch=64)
    try:
        rng = np.random.default_rng(0)
        fut = dispatcher.evaluate(_rand_batch(rng, 2))
        fut.result(timeout=10.0)
        dispatcher._poison(RuntimeError("late fatal"))
        # After poison, new submits raise immediately instead of queueing
        # onto a dispatcher that will never serve them.
        with pytest.raises(RuntimeError, match="ThreadedDispatcher thread died"):
            dispatcher.evaluate(_rand_batch(rng, 2))
    finally:
        dispatcher.shutdown()
