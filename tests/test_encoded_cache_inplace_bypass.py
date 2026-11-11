"""`--eval-cache-entries` must take effect on the single-walker UCI chain.

Single-walker, non-coalesced UCI wraps ``ThreadSafeGPUDispatcher`` (which has
the pinned slot API) in ``EncodedEvalCache``. The cache used to forward
``supports_inplace_api`` through ``__getattr__``, so Gumbel took the in-place
path on the inner dispatcher and the cache saw zero requests while reporting a
clean 0/0. These tests drive a real tiny CPU net through the real factory and
the real C search.
"""
from __future__ import annotations

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


def _single_walker_evaluator(model: torch.nn.Module, cache_entries: int) -> Any:
    # coalesce=False, n_walkers=1: the chain that used to bypass the cache.
    factory = _make_evaluator_factory([model], ["cpu"], False, 1, 1)
    return factory(64, eval_cache_entries=cache_entries)


def _search(evaluator: Any) -> list[Any]:
    out = run_gumbel_root_many_c(
        None,
        [chess.Board(fen) for fen in _BOARDS],
        device="cpu",
        rng=np.random.default_rng(7),
        cfg=GumbelConfig(
            input_extra_features="v2_threats", simulations=16, topk=4,
            add_noise=False, temperature=0.0,
        ),
        evaluator=evaluator,
    )
    return list(out[:4])


def _assert_same_search(a: list[Any], b: list[Any]) -> None:
    assert len(a) == len(b)
    for got, want in zip(a, b):
        assert len(got) == len(want)
        for g, w in zip(got, want):
            np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_single_walker_search_goes_through_the_cache(tiny_model: torch.nn.Module) -> None:
    uncached = _single_walker_evaluator(tiny_model, 0)
    # Default path untouched: no wrapper, and the pinned in-place API stays on.
    assert type(uncached) is ThreadSafeGPUDispatcher
    assert supports_inplace_api(uncached) is True

    cached = _single_walker_evaluator(tiny_model, 256)
    assert isinstance(cached, EncodedEvalCache)
    assert type(cached.async_backend) is ThreadSafeGPUDispatcher
    assert supports_inplace_api(cached) is False

    baseline = _search(uncached)
    first = _search(cached)
    after_first = cached.stats()
    assert after_first.misses > 0
    assert after_first.inner_calls > 0
    assert after_first.entries > 0

    second = _search(cached)
    after_second = cached.stats()
    # The repeat search re-evaluates the same positions: all served from cache.
    assert after_second.hits > after_first.hits
    assert after_second.inner_calls == after_first.inner_calls

    _assert_same_search(first, baseline)
    _assert_same_search(second, baseline)


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
