"""End-to-end training loop smoke tests.

Exercises the full Trainer pipeline (forward, loss, backward, optimizer step,
LR schedule, SWA update, save/load) using synthetic data — no Stockfish needed.
"""
from __future__ import annotations

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
    """Cross warmup, average multiple SWA states, then resume real training."""
    rng = np.random.default_rng(42)

    cfg = TransformerConfig(in_planes=146, embed_dim=64, num_layers=2, num_heads=4,
                            use_smolgen=False, use_nla=False)
    model = ChessNet(cfg)

    buf = ReplayBuffer(32, rng=rng)
    for _ in range(16):
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

    m1 = trainer.train_steps(buf, batch_size=8, steps=4)
    assert m1.loss > 0.0, "Loss should be positive"
    assert m1.train_steps_done == 4
    assert m1.train_samples_seen == 32
    warmup_lrs = [group["lr"] for group in trainer.opt.param_groups]

    # Eight steps cross the five-step warmup and observe SWA at steps 4 and 6.
    m2 = trainer.train_steps(buf, batch_size=8, steps=4)
    assert torch.isfinite(torch.tensor(m2.loss))
    assert m2.train_steps_done == 4
    assert m2.train_samples_seen == 32
    assert trainer.step == 8
    assert all(
        group["lr"] > warmup_lr
        for group, warmup_lr in zip(trainer.opt.param_groups, warmup_lrs, strict=True)
    )
    assert trainer._swa_model is not None
    assert int(trainer._swa_model.n_averaged) == 2

    # Save and load weights, optimizer moments, schedule, and the SWA average.
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

    assert trainer2.step == 8
    optimizer_state = trainer.opt.state_dict()
    assert optimizer_state["state"]
    torch.testing.assert_close(trainer2.opt.state_dict(), optimizer_state, rtol=0, atol=0)
    assert trainer2._swa_model is not None
    torch.testing.assert_close(
        trainer2._swa_model.state_dict(), trainer._swa_model.state_dict(), rtol=0, atol=0,
    )

    # Verify model weights match after load
    for (n1, p1), (n2, p2) in zip(
        trainer.model.named_parameters(),
        trainer2.model.named_parameters(),
    ):
        assert n1 == n2
        assert torch.equal(p1.data.cpu(), p2.data.cpu()), f"Param {n1} mismatch after load"

    resumed = trainer2.train_steps(buf, batch_size=8, steps=1)
    assert resumed.train_steps_done == 1
    assert resumed.train_samples_seen == 8
    assert torch.isfinite(torch.tensor(resumed.loss))
    assert trainer2.step == 9
    assert all(int(state["step"]) == 9 for state in trainer2.opt.state_dict()["state"].values())
    assert int(trainer2._swa_model.n_averaged) == 3
    assert any(
        not torch.equal(before, after)
        for before, after in zip(trainer.model.parameters(), trainer2.model.parameters(), strict=True)
    )

    # Export the restored average after its next observation.
    swa_path = tmp_path / "swa_model.pt"
    trainer2.export_swa(swa_path)
    swa_ckpt = torch.load(str(swa_path), map_location="cpu")
    assert "model" in swa_ckpt

    torch.testing.assert_close(swa_ckpt["model"], trainer2._swa_model.module.state_dict(), rtol=0, atol=0)


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
