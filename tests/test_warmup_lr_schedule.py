from __future__ import annotations

import numpy as np
import torch

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay import ReplayBuffer
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from chess_anti_engine.train import Trainer


def _make_sample() -> ReplaySample:
    x = np.random.randn(146, 8, 8).astype(np.float32)
    pol = np.random.rand(POLICY_SIZE).astype(np.float32)
    pol /= pol.sum()
    return ReplaySample(
        x=x, policy_target=pol, wdl_target=1, priority=1.0,
        has_policy=True, is_network_turn=True,
    )


def _make_trainer(tmp_path, *, optimizer: str = "adamw", lr: float = 1e-3,
                  warmup_steps: int = 10, warmup_lr_start: float = 1e-5) -> Trainer:
    cfg = TransformerConfig(in_planes=146, embed_dim=32, num_layers=1, num_heads=2,
                            use_smolgen=False, use_nla=False)
    model = ChessNet(cfg)
    return Trainer(
        model,
        device="cpu",
        lr=lr,
        log_dir=tmp_path / "tb",
        use_amp=False,
        feature_dropout_p=0.0,
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        warmup_lr_start=warmup_lr_start,
    )


def test_warmup_starts_at_warmup_lr_start(tmp_path):
    """At step 0, all param groups should be at their proportional warmup start LR."""
    trainer = _make_trainer(tmp_path, lr=1e-3, warmup_steps=10, warmup_lr_start=1e-5)

    # Step 0: should be at warmup start
    for pg in trainer.opt.param_groups:
        assert abs(pg["lr"] - 1e-5) < 1e-9


def test_warmup_reaches_peak_at_end(tmp_path):
    """After warmup_steps training steps, LR should reach peak_lr."""
    lr = 1e-3
    warmup = 5
    trainer = _make_trainer(tmp_path, lr=lr, warmup_steps=warmup, warmup_lr_start=1e-5)

    rng = np.random.default_rng(0)
    buf = ReplayBuffer(100, rng=rng)
    for _ in range(20):
        buf.add(_make_sample())

    trainer.train_steps(buf, batch_size=4, steps=warmup)

    # After warmup_steps, LR should be at or very near peak
    for pg in trainer.opt.param_groups:
        assert abs(pg["lr"] - lr) < 1e-6, \
            f"Expected LR ~{lr} after warmup, got {pg['lr']}"


def test_warmup_is_monotonically_increasing(tmp_path):
    """LR should monotonically increase during warmup."""
    trainer = _make_trainer(tmp_path, lr=1e-3, warmup_steps=8, warmup_lr_start=1e-5)

    rng = np.random.default_rng(0)
    buf = ReplayBuffer(100, rng=rng)
    for _ in range(20):
        buf.add(_make_sample())

    prev_lr = trainer.opt.param_groups[0]["lr"]
    for _ in range(8):
        trainer.train_steps(buf, batch_size=4, steps=1)
        cur_lr = trainer.opt.param_groups[0]["lr"]
        assert cur_lr >= prev_lr, f"LR decreased: {prev_lr} -> {cur_lr}"
        prev_lr = cur_lr


def test_muon_per_group_proportional_warmup(tmp_path):
    """Muon trunk groups (20x LR) should warm up proportionally, not flat."""
    lr = 1e-3
    warmup = 10
    warmup_start = 1e-5

    cfg = TransformerConfig(in_planes=146, embed_dim=32, num_layers=1, num_heads=2,
                            use_smolgen=False, use_nla=False)
    model = ChessNet(cfg)
    trainer = Trainer(
        model,
        device="cpu",
        lr=lr,
        log_dir=tmp_path / "tb",
        use_amp=False,
        feature_dropout_p=0.0,
        optimizer="muon",
        warmup_steps=warmup,
        warmup_lr_start=warmup_start,
    )

    # At step 0, Muon trunk groups (use_muon=True) should have 20x the warmup start
    muon_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_muon", False)]
    aux_groups = [pg for pg in trainer.opt.param_groups if not pg.get("use_muon", False)]

    assert len(muon_groups) > 0, "Expected at least one Muon param group"
    assert len(aux_groups) > 0, "Expected at least one aux param group"

    # At step 0: Muon trunk should be at warmup_start * 20, aux at warmup_start
    for pg in muon_groups:
        expected = warmup_start * 20.0
        assert abs(pg["lr"] - expected) < 1e-9, \
            f"Muon group LR at step 0: expected {expected}, got {pg['lr']}"

    for pg in aux_groups:
        assert abs(pg["lr"] - warmup_start) < 1e-9, \
            f"Aux group LR at step 0: expected {warmup_start}, got {pg['lr']}"

    # Train through warmup
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(100, rng=rng)
    for _ in range(30):
        buf.add(_make_sample())

    trainer.train_steps(buf, batch_size=4, steps=warmup)

    # After warmup: Muon trunk should be at 20*lr, aux at lr
    for pg in muon_groups:
        expected = lr * 20.0
        assert abs(pg["lr"] - expected) < 1e-6, \
            f"Muon group LR after warmup: expected {expected}, got {pg['lr']}"

    for pg in aux_groups:
        assert abs(pg["lr"] - lr) < 1e-6, \
            f"Aux group LR after warmup: expected {lr}, got {pg['lr']}"


def test_no_warmup_when_zero_steps(tmp_path):
    """With warmup_steps=0, LR should start at peak immediately."""
    lr = 1e-3
    trainer = _make_trainer(tmp_path, lr=lr, warmup_steps=0, warmup_lr_start=1e-5)

    for pg in trainer.opt.param_groups:
        assert abs(pg["lr"] - lr) < 1e-9
