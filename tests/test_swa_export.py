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


def test_swa_export_differs_from_raw_model(tmp_path):
    """After training with SWA enabled, the exported SWA weights should differ
    from the raw model weights (since they are a running average)."""
    cfg = TransformerConfig(in_planes=146, embed_dim=32, num_layers=1, num_heads=2,
                            use_smolgen=False, use_nla=False)
    model = ChessNet(cfg)

    rng = np.random.default_rng(42)
    buf = ReplayBuffer(100, rng=rng)
    for _ in range(20):
        buf.add(_make_sample())

    trainer = Trainer(
        model,
        device="cpu",
        lr=1e-2,
        log_dir=tmp_path / "tb",
        use_amp=False,
        feature_dropout_p=0.0,
        swa_start=1,
        swa_freq=1,
    )

    # Train enough steps to get SWA divergence from raw model
    trainer.train_steps(buf, batch_size=4, steps=10)

    raw_path = tmp_path / "raw.pt"
    swa_path = tmp_path / "swa.pt"

    trainer.save(raw_path)
    trainer.export_swa(swa_path)

    raw_ckpt = torch.load(str(raw_path), map_location="cpu")
    swa_ckpt = torch.load(str(swa_path), map_location="cpu")

    # Both should have model keys
    assert "model" in raw_ckpt
    assert "model" in swa_ckpt

    # At least one parameter should differ between raw and SWA
    any_diff = False
    for key in raw_ckpt["model"]:
        if key in swa_ckpt["model"]:
            if not torch.equal(raw_ckpt["model"][key], swa_ckpt["model"][key]):
                any_diff = True
                break
    assert any_diff, "SWA weights should differ from raw model after training"


def test_swa_export_without_swa_returns_raw_model(tmp_path):
    """When SWA is disabled (swa_start=0), export_swa should save raw model weights."""
    cfg = TransformerConfig(in_planes=146, embed_dim=32, num_layers=1, num_heads=2,
                            use_smolgen=False, use_nla=False)
    model = ChessNet(cfg)

    rng = np.random.default_rng(42)
    buf = ReplayBuffer(100, rng=rng)
    for _ in range(10):
        buf.add(_make_sample())

    trainer = Trainer(
        model,
        device="cpu",
        lr=1e-3,
        log_dir=tmp_path / "tb",
        use_amp=False,
        feature_dropout_p=0.0,
        swa_start=0,
    )

    trainer.train_steps(buf, batch_size=4, steps=3)

    raw_path = tmp_path / "raw.pt"
    swa_path = tmp_path / "swa.pt"

    trainer.save(raw_path)
    trainer.export_swa(swa_path)

    raw_ckpt = torch.load(str(raw_path), map_location="cpu")
    swa_ckpt = torch.load(str(swa_path), map_location="cpu")

    for key in raw_ckpt["model"]:
        assert torch.equal(raw_ckpt["model"][key], swa_ckpt["model"][key]), \
            f"Without SWA, exported weights should match raw model for {key}"
