from __future__ import annotations

import numpy as np
import torch

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.dataset import collate, collate_arrays


def _make_sample(*, with_optionals: bool = False) -> ReplaySample:
    x = np.random.randn(146, 8, 8).astype(np.float32)
    pol = np.random.rand(POLICY_SIZE).astype(np.float32)
    pol /= pol.sum()
    s = ReplaySample(x=x, policy_target=pol, wdl_target=1, priority=1.0, has_policy=True)
    if with_optionals:
        s.sf_wdl = np.array([0.3, 0.4, 0.3], dtype=np.float32)
        s.sf_move_index = 42
        s.moves_left = 0.5
        s.is_network_turn = True
        s.categorical_target = np.ones(32, dtype=np.float32) / 32.0
        s.policy_soft_target = pol.copy()
        s.future_policy_target = pol.copy()
        s.volatility_target = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        s.sf_volatility_target = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        s.has_future = True
        s.has_volatility = True
        s.has_sf_volatility = True
        s.legal_mask = np.ones(POLICY_SIZE, dtype=np.float32)
    return s


def _present_array(arr: np.ndarray | None) -> np.ndarray:
    assert arr is not None
    return arr


# ── collate (from ReplaySample list) ─────────────────────────────────


def test_collate_minimal_shapes():
    samples = [_make_sample() for _ in range(3)]
    batch = collate(samples, device="cpu")

    assert batch["x"].shape == (3, 146, 8, 8)
    assert batch["policy_t"].shape == (3, POLICY_SIZE)
    assert batch["wdl_t"].shape == (3,)
    assert batch["wdl_t"].dtype == torch.int64
    assert batch["has_policy"].shape == (3,)


def test_collate_optional_flags_when_absent():
    samples = [_make_sample(with_optionals=False) for _ in range(2)]
    batch = collate(samples, device="cpu")

    assert batch["has_sf_wdl"].sum().item() == 0.0
    assert batch["has_sf_move"].sum().item() == 0.0
    assert batch["has_future"].sum().item() == 0.0
    assert batch["has_volatility"].sum().item() == 0.0
    assert batch["has_sf_volatility"].sum().item() == 0.0
    assert batch["has_legal_mask"].sum().item() == 0.0


def test_collate_optional_flags_when_present():
    samples = [_make_sample(with_optionals=True) for _ in range(2)]
    batch = collate(samples, device="cpu")

    assert batch["has_sf_wdl"].sum().item() == 2.0
    assert batch["has_sf_move"].sum().item() == 2.0
    assert batch["has_future"].sum().item() == 2.0
    assert batch["has_volatility"].sum().item() == 2.0
    assert batch["has_sf_volatility"].sum().item() == 2.0
    assert batch["has_legal_mask"].sum().item() == 2.0


def test_collate_preserves_values():
    s = _make_sample(with_optionals=True)
    batch = collate([s], device="cpu")

    np.testing.assert_allclose(batch["x"][0].numpy(), s.x, atol=1e-6)
    np.testing.assert_allclose(batch["policy_t"][0].numpy(), s.policy_target, atol=1e-6)
    assert batch["wdl_t"][0].item() == s.wdl_target
    assert batch["sf_move_index"][0].item() == 42


def test_collate_mixed_optional_presence():
    s1 = _make_sample(with_optionals=True)
    s2 = _make_sample(with_optionals=False)
    batch = collate([s1, s2], device="cpu")

    assert batch["has_sf_wdl"][0].item() == 1.0
    assert batch["has_sf_wdl"][1].item() == 0.0
    assert batch["sf_wdl"][1].sum().item() == 0.0


def test_collate_respects_explicit_false_optional_flags():
    s = _make_sample(with_optionals=True)
    s.has_future = False
    s.has_volatility = False
    s.has_sf_volatility = False

    batch = collate([s], device="cpu")

    assert batch["has_future"][0].item() == 0.0
    assert batch["future_policy_t"][0].sum().item() == 0.0
    assert batch["has_volatility"][0].item() == 0.0
    assert batch["volatility_t"][0].sum().item() == 0.0
    assert batch["has_sf_volatility"][0].item() == 0.0
    assert batch["sf_volatility_t"][0].sum().item() == 0.0


# ── collate_arrays (from numpy dict) ─────────────────────────────────


def test_collate_arrays_minimal():
    arrs = {
        "x": np.random.randn(4, 146, 8, 8).astype(np.float32),
        "policy_target": np.random.rand(4, POLICY_SIZE).astype(np.float32),
        "wdl_target": np.array([0, 1, 2, 0], dtype=np.int8),
    }
    batch = collate_arrays(arrs, device="cpu")

    assert batch["x"].shape == (4, 146, 8, 8)
    assert batch["policy_t"].shape == (4, POLICY_SIZE)
    assert batch["wdl_t"].dtype == torch.int64


def test_collate_arrays_with_optionals():
    n = 3
    arrs = {
        "x": np.random.randn(n, 146, 8, 8).astype(np.float32),
        "policy_target": np.random.rand(n, POLICY_SIZE).astype(np.float32),
        "wdl_target": np.array([0, 1, 2], dtype=np.int8),
        "sf_wdl": np.random.rand(n, 3).astype(np.float32),
        "has_sf_wdl": np.ones(n, dtype=np.float32),
        "sf_move_index": np.array([10, 20, 30], dtype=np.int64),
        "has_sf_move": np.ones(n, dtype=np.float32),
        "categorical_target": np.ones((n, 32), dtype=np.float32) / 32.0,
        "has_categorical": np.ones(n, dtype=np.float32),
        "volatility_target": np.random.rand(n, 3).astype(np.float32),
        "has_volatility": np.ones(n, dtype=np.float32),
    }
    batch = collate_arrays(arrs, device="cpu")

    assert "sf_wdl" in batch
    assert "categorical_t" in batch
    assert "volatility_t" in batch
    assert batch["sf_move_index"].dtype == torch.int64


def test_collate_arrays_missing_optionals_not_in_output():
    arrs = {
        "x": np.random.randn(2, 146, 8, 8).astype(np.float32),
        "policy_target": np.random.rand(2, POLICY_SIZE).astype(np.float32),
        "wdl_target": np.array([0, 1], dtype=np.int8),
    }
    batch = collate_arrays(arrs, device="cpu")

    assert "sf_wdl" not in batch
    assert "has_sf_wdl" not in batch
    assert "volatility_t" not in batch


# ── Consistency between collate and collate_arrays ────────────────────


def test_collate_vs_collate_arrays_key_parity():
    """Both collation paths should produce the same set of output keys
    when given equivalent full-featured inputs."""
    samples = [_make_sample(with_optionals=True) for _ in range(2)]
    batch_from_samples = collate(samples, device="cpu")

    arrs = {
        "x": np.stack([s.x for s in samples]),
        "policy_target": np.stack([s.policy_target for s in samples]),
        "wdl_target": np.array([s.wdl_target for s in samples], dtype=np.int8),
        "has_policy": np.ones(2, dtype=np.float32),
        "sf_wdl": np.stack([_present_array(s.sf_wdl) for s in samples]),
        "has_sf_wdl": np.ones(2, dtype=np.float32),
        "sf_move_index": np.array([s.sf_move_index for s in samples], dtype=np.int64),
        "has_sf_move": np.ones(2, dtype=np.float32),
        "moves_left": np.array([s.moves_left for s in samples], dtype=np.float32),
        "has_moves_left": np.ones(2, dtype=np.float32),
        "is_network_turn": np.array([True, True]),
        "categorical_target": np.stack([_present_array(s.categorical_target) for s in samples]),
        "has_categorical": np.ones(2, dtype=np.float32),
        "policy_soft_target": np.stack([_present_array(s.policy_soft_target) for s in samples]),
        "has_policy_soft": np.ones(2, dtype=np.float32),
        "future_policy_target": np.stack([_present_array(s.future_policy_target) for s in samples]),
        "has_future": np.ones(2, dtype=np.float32),
        "volatility_target": np.stack([_present_array(s.volatility_target) for s in samples]),
        "has_volatility": np.ones(2, dtype=np.float32),
        "sf_volatility_target": np.stack([_present_array(s.sf_volatility_target) for s in samples]),
        "has_sf_volatility": np.ones(2, dtype=np.float32),
        "legal_mask": np.stack([_present_array(s.legal_mask) for s in samples]),
        "has_legal_mask": np.ones(2, dtype=np.float32),
    }
    batch_from_arrs = collate_arrays(arrs, device="cpu")

    # collate always produces all keys; collate_arrays only includes present optionals
    # Check that collate_arrays keys are a subset of collate keys
    for key in batch_from_arrs:
        assert key in batch_from_samples, f"key {key!r} missing from collate output"
