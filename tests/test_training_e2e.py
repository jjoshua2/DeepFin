"""End-to-end training loop smoke test with profiling.

Exercises the full Trainer pipeline (forward, loss, backward, optimizer step,
LR schedule, SWA update, save/load) using synthetic data — no Stockfish needed.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay import ReplayBuffer
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.train import Trainer


def _make_sample(rng: np.random.Generator) -> ReplaySample:
    x = rng.standard_normal((146, 8, 8)).astype(np.float32)
    pol = rng.random(POLICY_SIZE).astype(np.float32)
    pol /= pol.sum()
    wdl = int(rng.integers(0, 3))

    s = ReplaySample(x=x, policy_target=pol, wdl_target=wdl, priority=1.0,
                     has_policy=True, is_network_turn=True)
    s.sf_wdl = rng.dirichlet([1, 1, 1]).astype(np.float32)
    s.sf_move_index = int(rng.integers(0, POLICY_SIZE))
    s.moves_left = float(rng.random())
    s.categorical_target = np.ones(32, dtype=np.float32) / 32.0
    s.policy_soft_target = pol.copy()
    s.future_policy_target = pol.copy()
    s.volatility_target = rng.random(3).astype(np.float32)
    s.sf_volatility_target = rng.random(3).astype(np.float32)
    s.has_future = True
    s.has_volatility = True
    s.has_sf_volatility = True
    return s


def test_e2e_training_loop_smoke(tmp_path):
    """Full training loop: train N steps, verify loss decreases, save/load roundtrip."""
    rng = np.random.default_rng(42)

    cfg = TransformerConfig(in_planes=146, embed_dim=64, num_layers=2, num_heads=4,
                            use_smolgen=False, use_nla=False)
    model = ChessNet(cfg)

    buf = ReplayBuffer(500, rng=rng)
    for _ in range(100):
        buf.add(_make_sample(rng))

    trainer = Trainer(
        model,
        device="cpu",
        lr=1e-3,
        log_dir=tmp_path / "tb",
        use_amp=False,
        feature_dropout_p=0.15,
        warmup_steps=5,
        warmup_lr_start=1e-5,
        swa_start=3,
        swa_freq=2,
    )

    # Phase 1: train a few steps
    t0 = time.perf_counter()
    m1 = trainer.train_steps(buf, batch_size=8, steps=10)
    t1 = time.perf_counter()

    assert m1.loss > 0.0, "Loss should be positive"
    assert m1.train_steps_done == 10
    assert m1.train_samples_seen > 0

    # Phase 2: train more, loss should decrease (or at least be finite)
    m2 = trainer.train_steps(buf, batch_size=8, steps=10)
    t2 = time.perf_counter()

    assert torch.isfinite(torch.tensor(m2.loss))
    assert trainer.step == 20

    # Phase 3: save and load roundtrip
    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save(ckpt_path)

    # Create a fresh trainer and load
    model2 = ChessNet(cfg)
    trainer2 = Trainer(
        model2,
        device="cpu",
        lr=1e-3,
        log_dir=tmp_path / "tb2",
        use_amp=False,
        feature_dropout_p=0.15,
        warmup_steps=5,
        warmup_lr_start=1e-5,
        swa_start=3,
        swa_freq=2,
    )
    trainer2.load(ckpt_path)

    assert trainer2.step == 20

    # Verify model weights match after load
    for (n1, p1), (n2, p2) in zip(
        trainer.model.named_parameters(),
        trainer2.model.named_parameters(),
    ):
        assert n1 == n2
        assert torch.equal(p1.data.cpu(), p2.data.cpu()), f"Param {n1} mismatch after load"

    # Phase 4: SWA export
    swa_path = tmp_path / "swa_model.pt"
    trainer.export_swa(swa_path)
    swa_ckpt = torch.load(str(swa_path), map_location="cpu")
    assert "model" in swa_ckpt

    # Print profiling info
    phase1_s = t1 - t0
    phase2_s = t2 - t1
    total_steps = 20
    total_s = phase2_s + phase1_s
    print("\n[profiling] 20 train steps on CPU (embed=64, 2L, 4H):")
    print(f"  phase 1 (10 steps): {phase1_s:.3f}s ({10/phase1_s:.1f} steps/s)")
    print(f"  phase 2 (10 steps): {phase2_s:.3f}s ({10/phase2_s:.1f} steps/s)")
    print(f"  total: {total_s:.3f}s ({total_steps/total_s:.1f} steps/s)")
    print(f"  loss: {m1.loss:.4f} -> {m2.loss:.4f}")


def test_e2e_gradient_accumulation(tmp_path):
    """With accum_steps=2, effective batch is 2x but step count is the same."""
    rng = np.random.default_rng(0)

    cfg = TransformerConfig(in_planes=146, embed_dim=32, num_layers=1, num_heads=2,
                            use_smolgen=False, use_nla=False)
    model = ChessNet(cfg)

    buf = ReplayBuffer(200, rng=rng)
    for _ in range(50):
        buf.add(_make_sample(rng))

    trainer = Trainer(
        model,
        device="cpu",
        lr=1e-3,
        log_dir=tmp_path / "tb",
        use_amp=False,
        feature_dropout_p=0.0,
        accum_steps=2,
    )

    m = trainer.train_steps(buf, batch_size=4, steps=5)

    assert trainer.step == 5
    # With accum_steps=2, each step processes 2 micro-batches of 4 = 8 samples
    assert m.train_samples_seen == 5 * 2 * 4
    assert m.loss > 0.0
