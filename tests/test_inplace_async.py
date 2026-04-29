"""Slot-pool inplace async eval — parity with the legacy memcpy path.

`DirectGPUEvaluator(n_slots>1)` lets the caller write encodes directly into
pinned host memory and skip the per-call input memcpy. Output buffers are
slot-indexed too, so two in-flight async calls on different slots don't
overwrite each other's pinned outputs (which the gumbel pipeline relies on
to drop a defensive .copy()).
"""
from __future__ import annotations

import numpy as np

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.model import ModelConfig, build_model


def _make_evaluator(n_slots: int = 1) -> DirectGPUEvaluator:
    cfg = ModelConfig(embed_dim=32, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    return DirectGPUEvaluator(
        model, device="cpu", max_batch=16, use_amp=False, n_slots=n_slots,
    )


def test_n_slots_default_is_one():
    ev = _make_evaluator()
    assert ev.n_slots == 1


def test_get_input_buffer_returns_distinct_slots():
    ev = _make_evaluator(n_slots=2)
    b0 = ev.get_input_buffer(4, slot=0)
    b1 = ev.get_input_buffer(4, slot=1)
    assert b0.ctypes.data != b1.ctypes.data
    assert b0.shape == b1.shape == (4, 146, 8, 8)


def test_inplace_matches_evaluate_encoded():
    """Writing into the pinned slot then evaluate_inplace == evaluate_encoded(copy)."""
    ev = _make_evaluator(n_slots=2)
    x = np.random.default_rng(0).standard_normal((4, 146, 8, 8), dtype=np.float32)

    buf = ev.get_input_buffer(4, slot=0)
    buf[:] = x
    pol_a, wdl_a = ev.evaluate_inplace(4, slot=0)

    pol_b, wdl_b = ev.evaluate_encoded(x)
    assert np.allclose(pol_a, pol_b, atol=1e-5)
    assert np.allclose(wdl_a, wdl_b, atol=1e-5)


def test_inplace_async_independent_slots():
    """Two concurrent slots produce results consistent with sync evaluate_encoded."""
    ev = _make_evaluator(n_slots=2)
    x0 = np.random.default_rng(0).standard_normal((3, 146, 8, 8), dtype=np.float32)
    x1 = np.random.default_rng(1).standard_normal((5, 146, 8, 8), dtype=np.float32)

    ev.get_input_buffer(3, slot=0)[:] = x0
    ev.get_input_buffer(5, slot=1)[:] = x1

    pol0, wdl0, ev0 = ev.evaluate_inplace_async(3, slot=0)
    pol1, wdl1, ev1 = ev.evaluate_inplace_async(5, slot=1)
    if ev0 is not None:
        ev0.synchronize()
    if ev1 is not None:
        ev1.synchronize()

    ref0_pol, ref0_wdl = ev.evaluate_encoded(x0)
    ref1_pol, ref1_wdl = ev.evaluate_encoded(x1)

    assert np.allclose(pol0.numpy(), ref0_pol, atol=1e-5)
    assert np.allclose(wdl0.numpy(), ref0_wdl, atol=1e-5)
    assert np.allclose(pol1.numpy(), ref1_pol, atol=1e-5)
    assert np.allclose(wdl1.numpy(), ref1_wdl, atol=1e-5)


def test_slot_out_of_range_rejected():
    ev = _make_evaluator(n_slots=2)
    try:
        ev.get_input_buffer(4, slot=2)
    except ValueError as e:
        assert "slot" in str(e)
    else:  # pragma: no cover — fail
        raise AssertionError("expected ValueError on slot=2 with n_slots=2")
