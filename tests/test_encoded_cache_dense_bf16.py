from __future__ import annotations

import numpy as np
import pytest
import torch

from chess_anti_engine.inference_cache import EncodedEvalCache


class _BitEvaluator:
    input_extra_features = "v1"
    input_history_encoding = "legacy"
    policy_encoding = "lc0_1858"
    use_dynamic_relations = False
    supports_input_bf16_bits = True

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        self.inputs.append(x.copy())
        tensor = torch.from_numpy(x)
        if x.dtype == np.uint16:
            tensor = tensor.view(torch.bfloat16)
        tags = tensor.float().numpy()[:, 0, 0, 0]
        return np.repeat(tags[:, None], 4, axis=1), np.repeat(tags[:, None], 3, axis=1)


def _bits(*values: float) -> np.ndarray:
    rows = np.zeros((len(values), 1, 8, 8), dtype=np.float32)
    rows[:, 0, 0, 0] = values
    return torch.from_numpy(rows).to(torch.bfloat16).view(torch.uint16).numpy()


@pytest.mark.parametrize("strided", [False, True])
def test_dense_cache_preserves_bf16_bits_for_misses_and_hits(strided: bool) -> None:
    inner = _BitEvaluator()
    cache = EncodedEvalCache(inner, max_entries=8)
    rows = _bits(1.0, -2.0, 1.0)
    if strided:
        padded = np.zeros((3, 1, 8, 16), dtype=np.uint16)
        padded[..., ::2] = rows
        rows = padded[..., ::2]
        assert not rows.flags.c_contiguous
    expected = _BitEvaluator().evaluate_encoded(np.ascontiguousarray(rows))
    for _ in range(2):
        got = cache.evaluate_encoded(rows)
        for actual, want in zip(got, expected, strict=True):
            np.testing.assert_array_equal(actual, want)
    assert len(inner.inputs) == 1
    assert inner.inputs[0].dtype == np.uint16
    np.testing.assert_array_equal(inner.inputs[0], rows[:2])
    assert cache.stats().hits == 3
    assert cache.stats().rows_submitted == 2


def test_dense_cache_keeps_bit_patterns_distinct_from_numeric_float_values() -> None:
    inner = _BitEvaluator()
    cache = EncodedEvalCache(inner, max_entries=8)
    bits = _bits(1.0)
    # The same integers as float values mean 16256.0, not BF16 1.0.
    numeric = bits.astype(np.float32)
    actual_numeric, _ = cache.evaluate_encoded(numeric)
    actual_bits, _ = cache.evaluate_encoded(bits)
    np.testing.assert_array_equal(actual_numeric, np.full((1, 4), 16256.0))
    np.testing.assert_array_equal(actual_bits, np.ones((1, 4)))
    assert len(inner.inputs) == 2
    assert cache.stats().hits == 0


def test_dense_cache_mixed_hit_and_miss_preserves_bf16() -> None:
    cache = EncodedEvalCache(_BitEvaluator(), max_entries=8)
    cache.evaluate_encoded(_bits(1.0))
    pol, wdl = cache.evaluate_encoded(_bits(-2.0, 1.0, 3.0))
    np.testing.assert_array_equal(pol[:, 0], [-2.0, 1.0, 3.0])
    np.testing.assert_array_equal(wdl[:, 0], [-2.0, 1.0, 3.0])
    assert cache.stats().hits == 1


def test_empty_dense_bf16_batch_keeps_transport_dtype() -> None:
    inner = _BitEvaluator()
    cache = EncodedEvalCache(inner, max_entries=8)
    pol, wdl = cache.evaluate_encoded(_bits())
    assert pol.shape == (0, 4)
    assert wdl.shape == (0, 3)
    assert inner.inputs[0].dtype == np.uint16


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_dense_cache_retains_numeric_float32_coercion(dtype) -> None:
    inner = _BitEvaluator()
    cache = EncodedEvalCache(inner, max_entries=8)
    rows = np.full((1, 1, 8, 8), -2, dtype=dtype)
    pol, _ = cache.evaluate_encoded(rows)
    np.testing.assert_array_equal(pol, np.full((1, 4), -2.0))
    assert inner.inputs[0].dtype == np.float32


@pytest.mark.parametrize("cache_entries", [0, 64])
def test_uci_coalesced_gumbel_root_preserves_bf16_model_input(
    monkeypatch, cache_entries: int,
) -> None:
    import chess

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.inference_dispatcher import BatchCoalescingDispatcher
    from chess_anti_engine.mcts._mcts_tree import batch_encode_146_bf16
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.uci.__main__ import _make_evaluator_factory

    monkeypatch.setenv("CAE_UCI_COMPACT_BF16", "1")
    # Exercise the GPU-advertised transport through its real CPU decoder;
    # this does not claim CUDA execution or target-device qualification.
    monkeypatch.setattr(DirectGPUEvaluator, "supports_input_bf16_bits",
                        property(lambda _self: True))
    model = build_model(ModelConfig(
        input_extra_features="v2_threats", embed_dim=16, num_layers=1,
        num_heads=2, ffn_mult=2.0,
    )).eval()
    factory = _make_evaluator_factory([model], ["cpu"], True, 2, 1)
    evaluator = factory(64, eval_cache_entries=cache_entries)
    coalescer = evaluator.async_backend if isinstance(evaluator, EncodedEvalCache) else evaluator
    assert isinstance(coalescer, BatchCoalescingDispatcher)
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        captured.append(inputs[0].detach().float().clone())

    hook = model.register_forward_pre_hook(capture)
    board = chess.Board()
    try:
        run_gumbel_root_many_c(
            None, [board], device="cpu", rng=np.random.default_rng(0),
            cfg=GumbelConfig(input_extra_features="v2_threats", simulations=1,
                             topk=1, add_noise=False, temperature=0.0),
            evaluator=evaluator,
        )
        expected_bits = np.empty((1, 175, 8, 8), dtype=np.uint16)
        batch_encode_146_bf16([CBoard.from_board(board)], expected_bits)
        expected = torch.from_numpy(expected_bits).view(torch.bfloat16).float()
        assert captured
        torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
        if cache_entries:
            assert isinstance(evaluator, EncodedEvalCache)
            assert evaluator.stats().inner_calls > 0
    finally:
        hook.remove()
        coalescer.close()
