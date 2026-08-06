from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.inference_cache import EncodedEvalCache, PucvEvalCache


class _EchoEvaluator:
    # A real evaluator reaches its model through the wrapper chain; this one
    # declares the encoding itself, which is what resolve_encoding_source
    # looks for.
    input_extra_features = "v1"
    input_history_encoding = "legacy"
    policy_encoding = "lc0_1858"
    use_dynamic_relations = False

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

    with pytest.raises(ValueError, match="expected encoded batch shape"):
        cache.evaluate_encoded(np.zeros((146, 8, 8), dtype=np.float32))


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


# --- audit I5: the compact transport, and encoding-versioned keys ------------


class _CompactEvaluator(_EchoEvaluator):
    """Dense + compact transports, each counted separately.

    ``evaluate_legal_bf16`` returns one bf16-bits policy entry per legal move,
    concatenated over rows — the layout ``run_gumbel_root_many_c`` feeds
    straight into ``continue_gumbel_sims_legal_bf16``.
    """

    supports_legal_bf16 = True

    def __init__(self) -> None:
        super().__init__()
        self.compact_calls = 0
        self.compact_rows: list[int] = []

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.compact_calls += 1
        n = int(x.shape[0])
        self.compact_rows.append(n)
        counts = np.asarray(legal_counts, dtype=np.int32)
        flat = np.asarray(legal_flat, dtype=np.int32)
        tags = np.repeat(x[:, 0, 0, 0].astype(np.int64), counts.astype(np.int64))
        pol = ((tags * 1000 + flat.astype(np.int64)) % 65535).astype(np.uint16)
        wdl = np.zeros((n, 3), dtype=np.float32)
        wdl[:, 0] = x[:, 0, 0, 0].astype(np.float32) + 10.0
        return pol, wdl


def _legal(*idx: int) -> np.ndarray:
    return np.array(idx, dtype=np.int32)


def test_compact_transport_is_cached_rather_than_forwarded() -> None:
    """The defect: ``__getattr__`` handed this call to the inner evaluator.

    The cache was constructed, accepted its capacity, and reported a clean
    0 hits / 0 misses while every row went straight through.
    """
    inner = _CompactEvaluator()
    cache = EncodedEvalCache(inner, max_entries=64)
    batch = np.stack([_row(1.0), _row(2.0), _row(1.0)], axis=0)
    flat = np.concatenate([_legal(3, 9), _legal(4), _legal(3, 9)])
    counts = np.array([2, 1, 2], dtype=np.int32)

    pol, wdl = cache.evaluate_legal_bf16(batch, flat, counts)
    # Duplicate row deduped inside the batch: 2 distinct rows submitted.
    assert inner.compact_rows == [2]
    pol2, wdl2 = cache.evaluate_legal_bf16(batch, flat, counts)
    assert inner.compact_calls == 1, "second identical batch re-entered the model"
    np.testing.assert_array_equal(pol, pol2)
    np.testing.assert_array_equal(wdl, wdl2)

    stats = cache.stats()
    assert stats.hits == 3
    assert stats.misses == 3
    assert stats.rows_submitted == 2


def test_compact_cache_matches_the_uncached_result() -> None:
    """Layout and ordering, against the inner evaluator's own answer."""
    reference = _CompactEvaluator()
    inner = _CompactEvaluator()
    cache = EncodedEvalCache(inner, max_entries=64)
    batch = np.stack([_row(5.0), _row(6.0), _row(5.0), _row(7.0)], axis=0)
    flat = np.concatenate([_legal(1, 2, 3), _legal(8), _legal(1, 2, 3), _legal(4, 5)])
    counts = np.array([3, 1, 3, 2], dtype=np.int32)

    want_pol, want_wdl = reference.evaluate_legal_bf16(batch, flat, counts)
    got_pol, got_wdl = cache.evaluate_legal_bf16(batch, flat, counts)

    np.testing.assert_array_equal(got_pol, want_pol)
    np.testing.assert_array_equal(got_wdl, want_wdl)
    assert got_pol.dtype == want_pol.dtype
    # And again, now entirely from cache.
    hit_pol, hit_wdl = cache.evaluate_legal_bf16(batch, flat, counts)
    np.testing.assert_array_equal(hit_pol, want_pol)
    np.testing.assert_array_equal(hit_wdl, want_wdl)


def test_changed_encoding_version_misses_the_cache() -> None:
    """A cache built under one encoding can never serve entries under another.

    ``input_history_encoding`` is the sharp case: same plane count, different
    meaning, so neither the row shape nor the row digest moves. The key's
    encoding namespace is what makes the old entries unreachable.
    """
    inner = _CompactEvaluator()
    cache = EncodedEvalCache(inner, max_entries=64)
    batch = np.stack([_row(1.0)], axis=0)
    flat = _legal(3, 9)
    counts = np.array([2], dtype=np.int32)

    cache.evaluate_encoded(batch)
    cache.evaluate_legal_bf16(batch, flat, counts)
    assert inner.calls == 1
    assert inner.compact_calls == 1
    before = cache.encoding_identity

    inner.input_history_encoding = "lc0_8ply"

    cache.evaluate_encoded(batch)
    cache.evaluate_legal_bf16(batch, flat, counts)
    assert cache.encoding_identity != before
    assert inner.calls == 2, "dense hit an entry from the previous encoding"
    assert inner.compact_calls == 2, "compact hit an entry from the previous encoding"

    # The pre-change entries are unreachable, not overwritten: switching back
    # finds them again.
    inner.input_history_encoding = "legacy"
    cache.evaluate_encoded(batch)
    assert inner.calls == 2


def test_same_row_with_different_legal_indices_does_not_share_an_entry() -> None:
    """The compact value is the row's legal slice in the caller's order."""
    inner = _CompactEvaluator()
    cache = EncodedEvalCache(inner, max_entries=64)
    batch = np.stack([_row(1.0)], axis=0)

    a, _ = cache.evaluate_legal_bf16(batch, _legal(3, 9), np.array([2], dtype=np.int32))
    b, _ = cache.evaluate_legal_bf16(batch, _legal(9, 3), np.array([2], dtype=np.int32))

    assert inner.compact_calls == 2
    np.testing.assert_array_equal(a, b[::-1])


def test_dense_and_compact_entries_do_not_collide() -> None:
    inner = _CompactEvaluator()
    cache = EncodedEvalCache(inner, max_entries=64)
    batch = np.stack([_row(1.0)], axis=0)

    dense_pol, _ = cache.evaluate_encoded(batch)
    compact_pol, _ = cache.evaluate_legal_bf16(
        batch, _legal(3, 9), np.array([2], dtype=np.int32),
    )

    assert inner.calls == 1
    assert inner.compact_calls == 1
    assert dense_pol.shape == (1, 4)
    assert compact_pol.shape == (2,)


def test_relations_mode_is_part_of_the_key() -> None:
    """A with-relations row is a different computation from a bare one."""
    inner = _CompactEvaluator()
    cache = EncodedEvalCache(inner, max_entries=64)
    batch = np.stack([_row(1.0)], axis=0)
    rels = np.zeros((1, 2, 4, 4), dtype=np.uint8)

    cache.evaluate_encoded(batch)
    cache.evaluate_encoded(batch, relations=rels)

    assert inner.calls == 2


def test_supports_legal_bf16_follows_the_inner_evaluator() -> None:
    """A dense-only evaluator must not advertise compact through the cache.

    The search's transport test is ``hasattr(evaluator, "evaluate_legal_bf16")
    and evaluator.supports_legal_bf16``; the cache now defines the method
    unconditionally, so the flag has to carry the whole answer.
    """
    dense_only = EncodedEvalCache(_EchoEvaluator(), max_entries=8)
    assert dense_only.supports_legal_bf16 is False
    with pytest.raises(AttributeError, match="evaluate_legal_bf16"):
        dense_only.evaluate_legal_bf16(
            np.stack([_row(1.0)], axis=0), _legal(1), np.array([1], dtype=np.int32),
        )

    compact = EncodedEvalCache(_CompactEvaluator(), max_entries=8)
    assert compact.supports_legal_bf16 is True

    opted_out = _CompactEvaluator()
    opted_out.supports_legal_bf16 = False
    assert EncodedEvalCache(opted_out, max_entries=8).supports_legal_bf16 is False


def test_cache_refuses_an_evaluator_that_cannot_name_its_encoding() -> None:
    class _Opaque:
        def evaluate_encoded(
            self, x: np.ndarray, relations: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            del relations
            n = int(x.shape[0])
            return np.zeros((n, 4), dtype=np.float32), np.zeros((n, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="cannot resolve an encoding identity"):
        EncodedEvalCache(_Opaque(), max_entries=8)


def test_encoding_source_resolves_through_the_production_wrapper_chain() -> None:
    """ThreadSafeGPUDispatcher -> DirectGPUEvaluator -> model, as UCI builds it."""
    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
    from chess_anti_engine.model import ModelConfig, build_model

    model = build_model(ModelConfig(
        input_extra_features="v2_threats", embed_dim=16, num_layers=1,
        num_heads=2, ffn_mult=2.0,
    ))
    model.eval()
    evaluator = ThreadSafeGPUDispatcher(
        DirectGPUEvaluator(model, device="cpu", max_batch=4, use_amp=False),
    )
    cache = EncodedEvalCache(evaluator, max_entries=8)
    assert "input_extra_features='v2_threats'" in cache.encoding_identity
    assert "policy_encoding='lc0_1858'" in cache.encoding_identity


def test_cache_binds_on_the_compact_gumbel_search_path() -> None:
    """End-to-end: the transport UCI match play runs, through a real cache.

    ``run_gumbel_root_many_c`` selects the compact transport from
    ``hasattr(eval, "evaluate_legal_bf16") and eval.supports_legal_bf16``,
    which the cache satisfied by FORWARDING both to the inner evaluator — so
    the search ran while the cache saw nothing and reported 0/0. The
    observation that the fix binds is the cache's own request counter moving
    during a search it did not otherwise participate in.
    """
    import chess

    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
    from chess_anti_engine.model import ModelConfig, build_model

    model = build_model(ModelConfig(
        input_extra_features="v2_threats", embed_dim=16, num_layers=1,
        num_heads=2, ffn_mult=2.0,
    ))
    model.eval()
    cache = EncodedEvalCache(
        DirectGPUEvaluator(
            model, device="cpu", max_batch=64, use_amp=False, n_slots=2,
            legal_bf16=True,
        ),
        max_entries=4096,
    )
    assert cache.supports_legal_bf16 is True
    cfg = GumbelConfig(
        input_extra_features="v2_threats", simulations=32, topk=8,
        add_noise=False, temperature=0.0,
    )

    def _search() -> None:
        run_gumbel_root_many_c(
            None, [chess.Board()], device="cpu",
            rng=np.random.default_rng(0), cfg=cfg, evaluator=cache,
            vloss_weight=3,
        )

    _search()
    first = cache.stats()
    assert first.requests > 0, "the compact search bypassed the cache entirely"
    assert first.inner_calls > 0

    _search()
    second = cache.stats()
    assert second.hits > first.hits, "cached leaf evals were not reused"
