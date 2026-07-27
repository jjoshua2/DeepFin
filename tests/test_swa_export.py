from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay import ReplayBuffer
from chess_anti_engine.replay.buffer import ReplaySample
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
        if key in swa_ckpt["model"] and not torch.equal(raw_ckpt["model"][key], swa_ckpt["model"][key]):
            any_diff = True
            break
    assert any_diff, "SWA weights should differ from raw model after training"


def test_swa_export_without_swa_returns_raw_model(tmp_path):
    """When SWA is disabled (swa_start=-1), export_swa should save raw model weights."""
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
        swa_start=-1,  # negative = disabled
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


class _OrigModWrapper(torch.nn.Module):
    """Stand-in for ``torch.compile``'s OptimizedModule wrapper.

    All that matters for the key-convention invariant is that the trained
    module hangs off an ``_orig_mod`` attribute, which is what puts the
    ``_orig_mod.`` prefix on every ``state_dict`` key.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = module

    def forward(self, *args: object, **kwargs: object) -> object:
        return self._orig_mod(*args, **kwargs)


def _compiled_trainer(tmp_path: Path, **kwargs: object) -> Trainer:
    cfg = TransformerConfig(in_planes=146, embed_dim=32, num_layers=1, num_heads=2,
                            use_smolgen=False, use_nla=False)
    trainer = Trainer(
        ChessNet(cfg),
        device="cpu",
        lr=1e-3,
        log_dir=tmp_path / "tb",
        use_amp=False,
        feature_dropout_p=0.0,
        **kwargs,
    )
    trainer.model = _OrigModWrapper(trainer.model)
    return trainer


def test_publish_and_checkpoint_agree_on_key_convention(tmp_path):
    """Audit J9: ``export_swa`` (publish) must strip ``_orig_mod.`` like ``save``.

    Measured live before the fix: 496/496 published keys carried the prefix
    against 0/496 in the checkpoint. Every in-tree consumer happened to route
    through ``load_state_dict_tolerant``, so nothing broke -- but a plain
    ``load_state_dict(pub, strict=False)`` returned missing=86, unexpected=86
    and left 0/86 tensors correct without raising.
    """
    trainer = _compiled_trainer(tmp_path, swa_start=-1)

    assert any(k.startswith("_orig_mod.") for k in trainer.model.state_dict()), \
        "wrapper must actually produce prefixed keys or the test proves nothing"

    ckpt_path = tmp_path / "trainer.pt"
    pub_path = tmp_path / "latest_model.pt"
    trainer.save(ckpt_path)
    trainer.export_swa(pub_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")["model"]
    pub = torch.load(str(pub_path), map_location="cpu")["model"]

    assert [k for k in pub if k.startswith("_orig_mod.")] == []
    assert [k for k in ckpt if k.startswith("_orig_mod.")] == []
    assert set(pub) == set(ckpt), "publish and checkpoint key sets must agree"

    # The consequence the invariant is really about: a plain, strict load of the
    # published file into an unwrapped model succeeds and lands real tensors.
    cfg = TransformerConfig(in_planes=146, embed_dim=32, num_layers=1, num_heads=2,
                            use_smolgen=False, use_nla=False)
    fresh = ChessNet(cfg)
    missing, unexpected = fresh.load_state_dict(pub, strict=False)
    assert not missing
    assert not unexpected
    for key, value in fresh.state_dict().items():
        assert torch.equal(value, pub[key]), f"{key} did not actually load"


def test_publish_strips_prefix_with_swa_enabled(tmp_path):
    """The SWA branch of the publish path strips the prefix too."""
    trainer = _compiled_trainer(tmp_path, swa_start=-1)
    trainer._swa_model = torch.optim.swa_utils.AveragedModel(trainer.model)

    pub_path = tmp_path / "latest_model.pt"
    trainer.export_swa(pub_path)

    pub = torch.load(str(pub_path), map_location="cpu")["model"]
    assert [k for k in pub if k.startswith("_orig_mod.")] == []


def test_export_swa_warns_when_swa_diverges_from_checkpoint(tmp_path, caplog):
    """Audit J10: with SWA on, publish and checkpoint hold DIFFERENT nets.

    ``save`` must keep the raw model under ``"model"`` -- resume has to
    continue the real trajectory, not an average -- while the publish path
    ships the average to workers. The ratchet arena reads the checkpoint, so
    enabling SWA silently points the strength ruler at a net nobody plays.
    The two artifacts cannot be reconciled without breaking resume, so the
    publish path refuses to let the divergence be quiet.
    """
    trainer = _compiled_trainer(tmp_path, swa_start=-1)
    # Attach a diverged average directly rather than training one: the running
    # average is torch's, not ours, and what this pins is the export/save
    # asymmetry, not how the average is computed.
    trainer._swa_model = torch.optim.swa_utils.AveragedModel(trainer.model)
    with torch.no_grad():
        for param in trainer._swa_model.module.parameters():
            param.add_(1.0)

    ckpt_path = tmp_path / "trainer.pt"
    pub_path = tmp_path / "latest_model.pt"
    trainer.save(ckpt_path)
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.train.trainer"):
        trainer.export_swa(pub_path)

    assert any("J10" in r.getMessage() for r in caplog.records), \
        f"expected a loud SWA divergence warning, got {[r.getMessage() for r in caplog.records]}"

    ckpt = torch.load(str(ckpt_path), map_location="cpu")["model"]
    pub = torch.load(str(pub_path), map_location="cpu")["model"]
    assert set(ckpt) == set(pub)
    differing = [k for k in ckpt if not torch.equal(ckpt[k], pub[k])]
    assert differing, "the warning must only fire where the nets really can differ"


def test_export_swa_is_quiet_when_swa_is_off(tmp_path, caplog):
    """Production runs ``swa_start: -1`` -- no warning, no behaviour change."""
    trainer = _compiled_trainer(tmp_path, swa_start=-1)
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.train.trainer"):
        trainer.export_swa(tmp_path / "latest_model.pt")
    assert [r.getMessage() for r in caplog.records if "J10" in r.getMessage()] == []
