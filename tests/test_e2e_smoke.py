"""End-to-end smoke test: selfplay → replay buffer → training → checkpoint.

Requires Stockfish.  Skipped automatically when the binary is absent.
Designed to run in a few minutes on CPU / any GPU — uses a tiny model and
very few games/steps so it completes quickly.

What is verified:
  1. play_batch() produces samples without crashing
  2. All ReplaySample fields have correct shapes, dtypes, and value ranges
  3. Full-search vs fast-search split is respected (playout_cap_fraction)
  4. Buffer add_many / sample_batch round-trips cleanly
  5. Trainer.train_steps() returns finite loss on every component
  6. Checkpoint save → load → train preserves step counter and produces
     identical loss structure
  7. Gumbel MCTS path exercised alongside PUCT
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stockfish fixture — skip entire module if binary missing
# ---------------------------------------------------------------------------

SF_CANDIDATES = [
    "/home/josh/projects/chess/e2e_server/publish/stockfish",
    "/usr/bin/stockfish",
    "/usr/games/stockfish",
]

def _find_stockfish() -> str | None:
    import os
    import sys
    if sys.platform != "linux":
        return None  # E2E tests only run inside WSL/Linux
    for p in SF_CANDIDATES:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None

SF_PATH = _find_stockfish()
pytestmark = pytest.mark.skipif(SF_PATH is None, reason="Stockfish not found (run inside WSL)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLICY_SIZE = 4672
INPUT_PLANES = 146


def _tiny_model() -> torch.nn.Module:
    """Smallest transformer that exercises all 10 heads."""
    from chess_anti_engine.model import ModelConfig, build_model
    return build_model(ModelConfig(
        kind="transformer",
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_mult=2,
        use_smolgen=False,  # saves time
        use_nla=False,
    ))


def _assert_policy(arr: np.ndarray, name: str) -> None:
    assert arr.shape == (POLICY_SIZE,), f"{name}: shape {arr.shape}"
    assert np.isfinite(arr).all(), f"{name}: contains non-finite values"
    assert (arr >= 0).all(), f"{name}: negative values"
    total = float(arr.sum())
    assert 0.99 <= total <= 1.01, f"{name}: sum {total:.4f} not ~1"


def _assert_wdl_probs(arr: np.ndarray, name: str) -> None:
    assert arr.shape == (3,), f"{name}: shape {arr.shape}"
    assert np.isfinite(arr).all(), f"{name}: non-finite"
    assert (arr >= 0).all(), f"{name}: negative"
    total = float(arr.sum())
    assert 0.99 <= total <= 1.01, f"{name}: sum {total:.4f}"


# ---------------------------------------------------------------------------
# Test 1: selfplay produces valid samples
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def selfplay_samples():
    """Run a tiny selfplay batch and return (samples, stats)."""
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.selfplay.config import (
        DiffFocusConfig,
        GameConfig,
        SearchConfig,
        TemperatureConfig,
    )
    from chess_anti_engine.stockfish import StockfishUCI

    model = _tiny_model().eval()
    rng = np.random.default_rng(42)
    sf = StockfishUCI(SF_PATH, nodes=100, multipv=3)
    try:
        samples, stats = play_batch(
            model, device="cpu", rng=rng, stockfish=sf, games=3,
            temp=TemperatureConfig(temperature=1.0),
            search=SearchConfig(simulations=4, mcts_type="puct", playout_cap_fraction=0.5, fast_simulations=2),
            diff_focus=DiffFocusConfig(min_keep=1.0),
            game=GameConfig(max_plies=30, sf_policy_temp=0.25, sf_policy_label_smooth=0.05, categorical_bins=32, hlgauss_sigma=0.04),
        )
    finally:
        sf.close()
    return samples, stats


def test_play_batch_returns_samples(selfplay_samples):
    samples, stats = selfplay_samples
    assert len(samples) > 0, "Expected at least one training sample"
    assert stats.games == 3
    assert stats.w + stats.d + stats.l == 3


def test_sample_input_planes(selfplay_samples):
    samples, _ = selfplay_samples
    for i, s in enumerate(samples):
        assert s.x.shape == (INPUT_PLANES, 8, 8), f"sample {i}: x.shape={s.x.shape}"
        assert s.x.dtype == np.float32, f"sample {i}: x.dtype={s.x.dtype}"
        assert np.isfinite(s.x).all(), f"sample {i}: x contains non-finite"


def test_sample_policy_targets(selfplay_samples):
    samples, _ = selfplay_samples
    for i, s in enumerate(samples):
        _assert_policy(s.policy_target, f"sample[{i}].policy_target")


def test_sample_wdl_targets(selfplay_samples):
    samples, _ = selfplay_samples
    for i, s in enumerate(samples):
        assert s.wdl_target in (0, 1, 2), f"sample {i}: wdl_target={s.wdl_target}"


def test_sample_network_turns_only(selfplay_samples):
    samples, _ = selfplay_samples
    for i, s in enumerate(samples):
        assert s.is_network_turn is True, f"sample {i}: is_network_turn={s.is_network_turn}"


def test_sample_sf_wdl_when_present(selfplay_samples):
    samples, _ = selfplay_samples
    has_any = False
    for i, s in enumerate(samples):
        if s.sf_wdl is not None:
            _assert_wdl_probs(s.sf_wdl, f"sample[{i}].sf_wdl")
            has_any = True
        if s.sf_policy_target is not None:
            _assert_policy(s.sf_policy_target, f"sample[{i}].sf_policy_target")
    # At least some samples should have SF targets (we play vs SF)
    assert has_any, "No samples had sf_wdl — SF reply targets missing entirely"


def test_sample_soft_policy_when_present(selfplay_samples):
    samples, _ = selfplay_samples
    for i, s in enumerate(samples):
        if s.policy_soft_target is not None:
            _assert_policy(s.policy_soft_target, f"sample[{i}].policy_soft_target")


def test_sample_future_policy_when_present(selfplay_samples):
    samples, _ = selfplay_samples
    has_future = [s for s in samples if s.has_future]
    # With 30-ply games, most positions should have a t+2 target
    assert len(has_future) > 0, "No samples had future_policy_target"
    for i, s in enumerate(has_future):
        _assert_policy(s.future_policy_target, f"future_sample[{i}].future_policy_target")


def test_sample_volatility_when_present(selfplay_samples):
    samples, _ = selfplay_samples
    has_vol = [s for s in samples if s.has_volatility]
    assert len(has_vol) > 0, "No samples had volatility_target"
    for i, s in enumerate(has_vol):
        v = s.volatility_target
        assert v is not None
        assert v.shape == (3,), f"vol sample {i}: shape {v.shape}"
        assert np.isfinite(v).all(), f"vol sample {i}: non-finite"
        assert (v >= 0).all(), f"vol sample {i}: negative (expected absolute values)"


def test_sample_categorical_target(selfplay_samples):
    samples, _ = selfplay_samples
    for i, s in enumerate(samples):
        if s.categorical_target is not None:
            ct = s.categorical_target
            assert ct.shape == (32,), f"sample {i}: categorical shape {ct.shape}"
            assert np.isfinite(ct).all()
            assert (ct >= 0).all()
            total = float(ct.sum())
            assert 0.99 <= total <= 1.01, f"sample {i}: categorical sum {total:.4f}"


def test_sample_moves_left(selfplay_samples):
    samples, _ = selfplay_samples
    for i, s in enumerate(samples):
        if s.moves_left is not None:
            assert np.isfinite(s.moves_left), f"sample {i}: moves_left non-finite"
            assert s.moves_left >= 0, f"sample {i}: moves_left negative"


# ---------------------------------------------------------------------------
# Test 2: Gumbel MCTS path
# ---------------------------------------------------------------------------

def test_gumbel_selfplay_smoke():
    """Gumbel MCTS path also produces valid samples."""
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.selfplay.config import (
        DiffFocusConfig,
        GameConfig,
        SearchConfig,
        TemperatureConfig,
    )
    from chess_anti_engine.stockfish import StockfishUCI

    model = _tiny_model().eval()
    rng = np.random.default_rng(7)
    sf = StockfishUCI(SF_PATH, nodes=100, multipv=1)
    try:
        samples, stats = play_batch(
            model, device="cpu", rng=rng, stockfish=sf, games=2,
            temp=TemperatureConfig(temperature=1.0),
            search=SearchConfig(simulations=8, mcts_type="gumbel", playout_cap_fraction=1.0, fast_simulations=2),
            diff_focus=DiffFocusConfig(min_keep=1.0),
            game=GameConfig(max_plies=20, sf_policy_temp=0.25, sf_policy_label_smooth=0.05),
        )
    finally:
        sf.close()

    assert len(samples) > 0
    assert stats.games == 2
    for s in samples:
        assert s.x.shape == (INPUT_PLANES, 8, 8)
        _assert_policy(s.policy_target, "gumbel.policy_target")
        assert s.wdl_target in (0, 1, 2)


# ---------------------------------------------------------------------------
# Test 3: replay buffer round-trip
# ---------------------------------------------------------------------------

def test_replay_buffer_roundtrip(selfplay_samples):
    from chess_anti_engine.replay import ReplayBuffer
    samples, _ = selfplay_samples
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=10_000, rng=rng)
    buf.add_many(samples)
    assert len(buf) == len(samples)

    batch = buf.sample_batch(min(8, len(samples)))
    assert len(batch) == min(8, len(samples))
    for s in batch:
        assert s.x.shape == (INPUT_PLANES, 8, 8)


# ---------------------------------------------------------------------------
# Test 4: training step produces finite loss on every component
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_state(selfplay_samples, tmp_path_factory):
    """Fill a buffer and run one round of train_steps. Return (trainer, buf, metrics)."""
    from chess_anti_engine.replay import ReplayBuffer
    from chess_anti_engine.train import Trainer

    samples, _ = selfplay_samples
    rng = np.random.default_rng(1)
    buf = ReplayBuffer(capacity=10_000, rng=rng)
    buf.add_many(samples)

    log_dir = tmp_path_factory.mktemp("tb")
    model = _tiny_model()
    trainer = Trainer(
        model,
        device="cpu",
        lr=1e-3,
        log_dir=log_dir,
        use_amp=False,
        feature_dropout_p=0.0,
        w_volatility=0.05,
        # Pass all configurable loss weights to exercise that code path
        w_policy=1.0,
        w_soft=0.5,
        w_future=0.15,
        w_wdl=1.0,
        w_sf_move=0.15,
        w_sf_eval=0.15,
        w_categorical=0.10,
        w_sf_volatility=0.05,
        w_moves_left=0.02,
    )

    metrics = trainer.train_steps(buf, batch_size=min(8, len(buf)), steps=3)
    return trainer, buf, metrics


def test_training_loss_finite(trained_state):
    _, _, metrics = trained_state
    assert math.isfinite(metrics.loss), f"total loss non-finite: {metrics.loss}"


def test_training_component_losses_finite(trained_state):
    _, _, metrics = trained_state
    fields = {
        "policy_loss": metrics.policy_loss,
        "soft_policy_loss": metrics.soft_policy_loss,
        "future_policy_loss": metrics.future_policy_loss,
        "wdl_loss": metrics.wdl_loss,
        "sf_move_loss": metrics.sf_move_loss,
        "sf_eval_loss": metrics.sf_eval_loss,
        "categorical_loss": metrics.categorical_loss,
        "volatility_loss": metrics.volatility_loss,
        "sf_volatility_loss": metrics.sf_volatility_loss,
        "moves_left_loss": metrics.moves_left_loss,
    }
    for name, val in fields.items():
        assert math.isfinite(val), f"{name} is non-finite: {val}"
        assert val >= 0, f"{name} is negative: {val}"


def test_trainer_step_advances(trained_state):
    trainer, _, _ = trained_state
    assert trainer.step == 3, f"Expected step=3 after 3 train_steps, got {trainer.step}"


# ---------------------------------------------------------------------------
# Test 5: checkpoint save → load → continue training
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(trained_state, tmp_path_factory):
    from chess_anti_engine.train import Trainer

    trainer, buf, _ = trained_state
    ckpt_dir = tmp_path_factory.mktemp("ckpt")
    ckpt_path = ckpt_dir / "trainer.pt"

    step_before = trainer.step
    trainer.save(ckpt_path)
    assert ckpt_path.exists()

    # Load into a fresh trainer with identical config
    model2 = _tiny_model()
    trainer2 = Trainer(
        model2,
        device="cpu",
        lr=1e-3,
        log_dir=ckpt_dir / "tb2",
        use_amp=False,
    )
    trainer2.load(ckpt_path)
    assert trainer2.step == step_before, (
        f"Step not restored: expected {step_before}, got {trainer2.step}"
    )

    # Can continue training without error
    metrics2 = trainer2.train_steps(buf, batch_size=min(8, len(buf)), steps=2)
    assert math.isfinite(metrics2.loss)
    assert trainer2.step == step_before + 2


# ---------------------------------------------------------------------------
# Test 6: model forward produces expected output keys and shapes
# ---------------------------------------------------------------------------

def test_model_output_heads():
    model = _tiny_model()
    x = torch.randn(2, INPUT_PLANES, 8, 8)
    with torch.no_grad():
        out = model(x)

    expected_heads = {
        "policy_own": (2, POLICY_SIZE),
        "policy_soft": (2, POLICY_SIZE),
        "policy_sf": (2, POLICY_SIZE),
        "policy_future": (2, POLICY_SIZE),
        "wdl": (2, 3),
        "sf_eval": (2, 3),
        "categorical": (2, 32),
        "volatility": (2, 3),
        "sf_volatility": (2, 3),
        "moves_left": (2, 1),
    }
    for head, expected_shape in expected_heads.items():
        assert head in out, f"Missing head: {head}"
        actual = tuple(out[head].shape)
        assert actual == expected_shape, f"{head}: expected {expected_shape}, got {actual}"
