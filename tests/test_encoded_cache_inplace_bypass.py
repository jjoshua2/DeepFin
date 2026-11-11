"""`--eval-cache-entries` must take effect on the single-walker UCI chain.

Single-walker, non-coalesced UCI wraps ``ThreadSafeGPUDispatcher`` (which has
the pinned slot API) in ``EncodedEvalCache``. The cache used to forward
``supports_inplace_api`` through ``__getattr__``, so Gumbel took the in-place
path on the inner dispatcher and the cache saw zero requests while reporting a
clean 0/0. These tests drive a real tiny CPU net through the real factory and
the real C search.
"""
from __future__ import annotations

import random
from typing import Any, cast

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.inference_cache import EncodedEvalCache
from chess_anti_engine.inference_dispatcher import (
    ThreadSafeGPUDispatcher,
    supports_inplace_api,
)
from chess_anti_engine.mcts.gumbel import GumbelConfig
import chess_anti_engine.mcts.gumbel_c as gumbel_c_mod
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.__main__ import _make_evaluator_factory

_BOARDS = (
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
)


@pytest.fixture(scope="module")
def tiny_model() -> torch.nn.Module:
    torch.manual_seed(1234)
    return build_model(ModelConfig(
        input_extra_features="v2_threats", embed_dim=16, num_layers=1,
        num_heads=2, ffn_mult=2.0,
    )).eval()


def _single_walker_evaluator(
    model: torch.nn.Module, cache_entries: int, max_batch: int = 64,
) -> Any:
    # coalesce=False, n_walkers=1: the chain that used to bypass the cache.
    factory = _make_evaluator_factory([model], ["cpu"], False, 1, 1)
    return factory(max_batch, eval_cache_entries=cache_entries)


def _search(evaluator: Any, boards: Any = _BOARDS, *, sims: int = 16, topk: int = 4) -> list[Any]:
    out = run_gumbel_root_many_c(
        None,
        [chess.Board(fen) if isinstance(fen, str) else fen.copy() for fen in boards],
        device="cpu",
        rng=np.random.default_rng(7),
        cfg=GumbelConfig(
            input_extra_features="v2_threats", simulations=sims, topk=topk,
            add_noise=False, temperature=0.0,
        ),
        evaluator=evaluator,
    )
    return list(out[:4])


class _RowRecorder:
    """Wraps the cache's inner evaluator and records every row it is sent."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.rows: list[np.ndarray] = []
        self.batch_sizes: list[int] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def evaluate_encoded(self, x: np.ndarray, relations: Any = None) -> Any:
        self.rows.extend(np.array(r, copy=True) for r in x)
        self.batch_sizes.append(int(x.shape[0]))
        return self._inner.evaluate_encoded(x, relations=relations)


def test_single_walker_search_goes_through_the_cache(tiny_model: torch.nn.Module) -> None:
    uncached = _single_walker_evaluator(tiny_model, 0)
    # Default path untouched: no wrapper, and the pinned in-place API stays on.
    assert type(uncached) is ThreadSafeGPUDispatcher
    assert supports_inplace_api(uncached) is True

    cached = _single_walker_evaluator(tiny_model, 256)
    assert isinstance(cached, EncodedEvalCache)
    assert type(cached.async_backend) is ThreadSafeGPUDispatcher
    assert supports_inplace_api(cached) is False
    recorder = _RowRecorder(cached._inner)
    cached._inner = recorder

    _search(cached)
    after_first = cached.stats()
    assert after_first.misses > 0
    assert after_first.inner_calls == len(recorder.batch_sizes) > 0
    assert after_first.rows_submitted == len(recorder.rows)
    assert after_first.entries == len(recorder.rows)

    _search(cached)
    after_second = cached.stats()
    # The repeat search re-evaluates the same positions: all served from cache.
    assert after_second.hits > after_first.hits
    assert after_second.inner_calls == after_first.inner_calls

    # Cached outputs belong to their rows: every row the inner evaluated, served
    # back from the cache, matches a direct uncached evaluation of that row.
    # (Search OUTPUT equality cache-on vs cache-off is not guaranteed: dedup
    # changes batch composition, and float noise is amplified by search.)
    rows = np.stack(recorder.rows, axis=0)
    hits_before = cached.stats().hits
    pol_c, wdl_c = cached.evaluate_encoded(rows)
    assert cached.stats().hits == hits_before + len(rows)
    assert cached.stats().inner_calls == after_first.inner_calls
    pol_d, wdl_d = uncached.evaluate_encoded(rows)
    np.testing.assert_allclose(pol_c, pol_d, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(wdl_c, wdl_d, rtol=1e-4, atol=1e-4)


def _random_boards(n: int, seed: int) -> list[chess.Board]:
    rnd = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < n:
        b = chess.Board()
        for _ in range(rnd.randint(4, 20)):
            if b.is_game_over():
                break
            b.push(rnd.choice(list(b.legal_moves)))
        if not b.is_game_over():
            boards.append(b)
    return boards


def test_cache_keeps_the_max_batch_leaf_cap(
    tiny_model: torch.nn.Module, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MaxBatch 64 (the UCI minimum), 16 boards x topk 16: the search wants
    512-row leaf waves. Cache-off caps them at 64 on the in-place path and
    absorbs the rest; cache-on must cap identically instead of sending 256
    rows to an inner that raises above 64."""
    boards = _random_boards(16, seed=3)
    leaf_batches: dict[str, list[int]] = {}
    real_override = gumbel_c_mod._tb_override

    def run(label: str, evaluator: Any) -> None:
        seen: list[int] = []

        def spy(tree: Any, probe: Any, wdl: np.ndarray) -> None:
            seen.append(len(wdl))
            real_override(tree, probe, wdl)

        monkeypatch.setattr(gumbel_c_mod, "_tb_override", spy)
        _search(evaluator, boards, sims=64, topk=16)
        leaf_batches[label] = seen

    run("off", _single_walker_evaluator(tiny_model, 0))
    cached = _single_walker_evaluator(tiny_model, 100_000)
    assert cached.search_leaf_row_cap == 64
    recorder = _RowRecorder(cached._inner)
    cached._inner = recorder
    run("on", cached)

    assert max(leaf_batches["off"]) == 64
    # Same buffer cap, so the same saturated first wave and the same ceiling.
    assert max(leaf_batches["on"]) == 64
    assert leaf_batches["on"][0] == leaf_batches["off"][0] == 64
    assert max(recorder.batch_sizes) <= 64


def test_cache_receives_only_live_leaf_rows(
    tiny_model: torch.nn.Module, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows past n_leaves are stale/uninitialised padding: never hashed or stored."""
    cached = _single_walker_evaluator(tiny_model, 100_000, max_batch=1024)
    seen: list[np.ndarray] = []
    real_eval = EncodedEvalCache.evaluate_encoded

    def spy(self: Any, x: np.ndarray, relations: Any = None) -> Any:
        seen.append(np.array(x, copy=True))
        return real_eval(self, x, relations=relations)

    monkeypatch.setattr(cached, "evaluate_encoded", spy.__get__(cached))
    leaf_sizes: list[int] = []
    real_override = gumbel_c_mod._tb_override

    def override_spy(tree: Any, probe: Any, wdl: np.ndarray) -> None:
        leaf_sizes.append(len(wdl))
        real_override(tree, probe, wdl)

    monkeypatch.setattr(gumbel_c_mod, "_tb_override", override_spy)
    _search(cached, _random_boards(3, seed=11), sims=24, topk=4)

    # First call is the root batch; each later call is one leaf wave.
    assert [len(x) for x in seen[1:]] == leaf_sizes
    # At least one wave is off the bucket ladder, so padding would have shown.
    assert any(n not in (16, 32, 64, 96, 128, 170, 256, 340, 384, 512) for n in leaf_sizes)
    all_rows = np.concatenate(seen, axis=0)
    unique = {r.tobytes() for r in all_rows}
    stats = cached.stats()
    assert stats.requests == len(all_rows)
    assert stats.entries == len(unique)
    assert stats.evictions == 0


def test_cache_hides_uncached_dense_entry_points(tiny_model: torch.nn.Module) -> None:
    cached = _single_walker_evaluator(tiny_model, 64)
    inner = cached.async_backend
    assert hasattr(inner, "evaluate_encoded_async")
    # Gumbel's 2-group pipeline keys on this hasattr; forwarding it bypassed
    # the cache for every batch of >= 64 boards.
    assert not hasattr(cached, "evaluate_encoded_async")
    assert not hasattr(cached, "evaluate_encoded_with_volatility")
    # PUCV searchers still need the forwarded slot API (they cache through
    # their own PucvEvalCache).
    assert hasattr(cached, "evaluate_inplace_async")
    assert cached.n_slots == inner.n_slots


def test_large_batch_pipeline_goes_through_the_cache(tiny_model: torch.nn.Module) -> None:
    """>= 64 boards selects Gumbel's async pipeline when the hasattr succeeds."""
    cached = _single_walker_evaluator(tiny_model, 4096)
    boards = [chess.Board() for _ in range(64)]
    run_gumbel_root_many_c(
        None, boards, device="cpu", rng=np.random.default_rng(3),
        cfg=GumbelConfig(
            input_extra_features="v2_threats", simulations=4, topk=2,
            add_noise=False, temperature=0.0,
        ),
        evaluator=cached,
    )
    stats = cached.stats()
    # 64 identical roots dedupe within each batch: many requests, few rows.
    assert stats.misses >= 64
    assert 0 < stats.rows_submitted < stats.misses


class _Encoding:
    input_extra_features = "v2_threats"
    input_history_encoding = "legacy"
    policy_encoding = "lc0_1858"
    use_dynamic_relations = False


class _VolatilityInner:
    """An inner with every uncached dense entry point the cache must not forward."""

    max_batch = 8

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((len(x), 4672), np.float32), np.zeros((len(x), 3), np.float32)

    def evaluate_encoded_async(self, x: np.ndarray) -> object:
        del x
        raise AssertionError("cache forwarded evaluate_encoded_async")

    def evaluate_encoded_with_volatility(self, x: np.ndarray) -> object:
        del x
        raise AssertionError("cache forwarded evaluate_encoded_with_volatility")


@pytest.mark.parametrize("name", ["evaluate_encoded_async", "evaluate_encoded_with_volatility"])
def test_uncached_entry_points_are_hidden_even_when_inner_has_them(name: str) -> None:
    cache = EncodedEvalCache(
        cast(Any, _VolatilityInner()), max_entries=8, encoding_source=_Encoding(),
    )
    assert hasattr(cache._inner, name)
    assert not hasattr(cache, name)
    with pytest.raises(AttributeError, match="bypass the cache"):
        getattr(cache, name)
    # Ordinary attributes still forward.
    assert cache.max_batch == 8


class _CappedInner:
    """Rejects batches above its max, like DirectGPUEvaluator."""

    _max_batch = 4

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.legal_sizes: list[int] = []
        # One reused output buffer, as pinned-memory evaluators have with
        # copy_out=False: a chunk result must be copied before the next call.
        self._pol = np.zeros((self._max_batch, 4672), np.float32)
        self._wdl = np.zeros((self._max_batch, 3), np.float32)

    def evaluate_encoded(self, x: np.ndarray, relations: Any = None) -> Any:
        del relations
        if len(x) > self._max_batch:
            raise ValueError(f"batch {len(x)} > max {self._max_batch}")
        self.sizes.append(len(x))
        # Output encodes the row so a mis-stitched chunk is visible.
        tag = x.reshape(len(x), -1)[:, 0].astype(np.float32)
        self._pol[: len(x)] = tag[:, None]
        self._wdl[: len(x)] = tag[:, None]
        return self._pol[: len(x)], self._wdl[: len(x)]

    def evaluate_legal_bf16(self, x: np.ndarray, flat: np.ndarray, counts: np.ndarray) -> Any:
        if len(x) > self._max_batch:
            raise ValueError(f"batch {len(x)} > max {self._max_batch}")
        self.legal_sizes.append(len(x))
        tag = x.reshape(len(x), -1)[:, 0].astype(np.uint16)
        return np.repeat(tag, counts).astype(np.uint16) + flat.astype(np.uint16), np.repeat(
            tag[:, None].astype(np.float32), 3, axis=1,
        )


def test_no_leaf_cap_is_advertised_for_an_inner_without_the_inplace_path() -> None:
    # ``_CappedInner`` has ``_max_batch`` but no slot API: without this cache
    # Gumbel would have taken the copying branch, which is uncapped, so the
    # cache must not introduce a cap of its own.
    cache = EncodedEvalCache(cast(Any, _CappedInner()), max_entries=8, encoding_source=_Encoding())
    assert cache.search_leaf_row_cap is None


def test_cache_splits_miss_batches_at_the_inner_max() -> None:
    inner = _CappedInner()
    cache = EncodedEvalCache(cast(Any, inner), max_entries=64, encoding_source=_Encoding())
    x = np.zeros((10, 175, 8, 8), np.float32)
    x[:, 0, 0, 0] = np.arange(10)
    pol, wdl = cache.evaluate_encoded(x)
    assert inner.sizes == [4, 4, 2]
    np.testing.assert_array_equal(wdl[:, 0], np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(pol[:, 0], np.arange(10, dtype=np.float32))
    stats = cache.stats()
    assert (stats.inner_calls, stats.rows_submitted, stats.entries) == (3, 10, 10)


def test_cache_splits_compact_miss_batches_at_the_inner_max() -> None:
    inner = _CappedInner()
    cache = EncodedEvalCache(cast(Any, inner), max_entries=64, encoding_source=_Encoding())
    x = np.zeros((6, 175, 8, 8), np.uint16)
    x[:, 0, 0, 0] = np.arange(6)
    counts = np.array([1, 2, 3, 1, 2, 3], np.int32)
    flat = np.arange(int(counts.sum()), dtype=np.int32)
    pol, wdl = cache.evaluate_legal_bf16(x, flat, counts)
    assert inner.legal_sizes == [4, 2]
    want = np.repeat(np.arange(6), counts).astype(np.uint16) + flat.astype(np.uint16)
    np.testing.assert_array_equal(pol, want)
    np.testing.assert_array_equal(wdl[:, 0], np.arange(6, dtype=np.float32))
