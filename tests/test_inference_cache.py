from __future__ import annotations

import numpy as np

from chess_anti_engine.inference_cache import EncodedEvalCache, PucvEvalCache


class _EchoEvaluator:
    def __init__(self) -> None:
        self.calls = 0
        self.rows_submitted: list[int] = []

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations  # interface conformance for BatchEvaluator
        self.calls += 1
        self.rows_submitted.append(int(x.shape[0]))
        n = int(x.shape[0])
        pol = np.zeros((n, 4), dtype=np.float32)
        wdl = np.zeros((n, 3), dtype=np.float32)
        tags = x[:, 0, 0, 0].astype(np.float32)
        pol[:, 0] = tags
        wdl[:, 0] = tags + 10.0
        return pol, wdl


def _row(tag: float) -> np.ndarray:
    x = np.zeros((146, 8, 8), dtype=np.float32)
    x[0, 0, 0] = tag
    return x


def test_cache_deduplicates_repeated_rows_in_one_batch() -> None:
    inner = _EchoEvaluator()
    cache = EncodedEvalCache(inner, max_entries=8)
    batch = np.stack([_row(1.0), _row(2.0), _row(1.0)], axis=0)

    pol, wdl = cache.evaluate_encoded(batch)

    assert inner.calls == 1
    assert inner.rows_submitted == [2]
    np.testing.assert_array_equal(pol[:, 0], np.array([1.0, 2.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(wdl[:, 0], np.array([11.0, 12.0, 11.0], dtype=np.float32))
    stats = cache.stats()
    assert stats.hits == 0
    assert stats.misses == 3
    assert stats.rows_submitted == 2
    assert stats.inner_calls == 1


def test_cache_reuses_rows_across_batches() -> None:
    inner = _EchoEvaluator()
    cache = EncodedEvalCache(inner, max_entries=8)

    cache.evaluate_encoded(np.stack([_row(1.0), _row(2.0)], axis=0))
    pol, wdl = cache.evaluate_encoded(np.stack([_row(2.0), _row(1.0)], axis=0))

    assert inner.calls == 1
    assert inner.rows_submitted == [2]
    np.testing.assert_array_equal(pol[:, 0], np.array([2.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(wdl[:, 0], np.array([12.0, 11.0], dtype=np.float32))
    stats = cache.stats()
    assert stats.hits == 2
    assert stats.misses == 2
    assert stats.hit_rate == 0.5


def test_cache_eviction_lru() -> None:
    inner = _EchoEvaluator()
    cache = EncodedEvalCache(inner, max_entries=1)

    cache.evaluate_encoded(np.stack([_row(1.0)], axis=0))
    cache.evaluate_encoded(np.stack([_row(2.0)], axis=0))
    cache.evaluate_encoded(np.stack([_row(1.0)], axis=0))

    assert inner.calls == 3
    assert inner.rows_submitted == [1, 1, 1]
    stats = cache.stats()
    assert stats.entries == 1
    assert stats.evictions == 2
    assert stats.hits == 0
    assert stats.misses == 3


def test_cache_clear_resets_entries_and_stats() -> None:
    inner = _EchoEvaluator()
    cache = EncodedEvalCache(inner, max_entries=8)
    cache.evaluate_encoded(np.stack([_row(1.0)], axis=0))

    cache.clear()

    stats = cache.stats()
    assert stats.entries == 0
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.inner_calls == 0
    assert stats.rows_submitted == 0


def test_cache_rejects_non_batch_input() -> None:
    cache = EncodedEvalCache(_EchoEvaluator(), max_entries=8)

    try:
        cache.evaluate_encoded(np.zeros((146, 8, 8), dtype=np.float32))
    except ValueError as exc:
        assert "expected encoded batch shape" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_pucv_cache_verifies_encoded_digest_on_fingerprint_hit() -> None:
    cache = PucvEvalCache(max_entries=8)
    key = np.array([123, 456], dtype=np.uint64)
    row_a = _row(1.0)
    row_b = _row(2.0)
    pol = np.full(4, 7.0, dtype=np.float32)
    wdl = np.full(3, 0.25, dtype=np.float32)

    cache.put(key, row_a, pol, wdl)

    hit = cache.get(key, row_a)
    assert hit is not None
    np.testing.assert_array_equal(hit[0], pol)
    assert cache.get(key, row_b) is None
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
