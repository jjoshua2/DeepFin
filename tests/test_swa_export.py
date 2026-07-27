from __future__ import annotations

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


# ---------------------------------------------------------------------------
# rl_loop_audit J9: the PUBLISH path must use the same key convention as save().
#
# `save()` stripped torch.compile's `_orig_mod.` segment; `export_swa()` did
# not. With `use_compile: true` (production since 2026-04-27) that made every
# published `latest_model.pt` carry the prefix on all 496 keys while the
# sibling checkpoint carried it on none -- two conventions for the same
# weights. It never surfaced because every in-tree consumer routes through
# `load_state_dict_tolerant`, which normalizes either direction, so the tests
# could not see it either. The checks below use a plain `load_state_dict` on
# purpose: the non-tolerant loader is the one that fails silently, and only it
# can express the defect.
#
# `apply_compile` is a no-op off CUDA, so the wrap is applied directly here.
# The wrap is applied AFTER training and no forward is ever run through it:
# `torch.compile` is lazy, so dynamo and inductor never fire (a CPU inductor
# compile of ChessNet costs minutes) while the key renaming -- the only thing
# under test -- is reproduced exactly.
# ---------------------------------------------------------------------------

def _tiny_net() -> ChessNet:
    return ChessNet(TransformerConfig(in_planes=146, embed_dim=32, num_layers=1,
                                      num_heads=2, use_smolgen=False, use_nla=False))


def _trained_trainer(tmp_path, *, swa: bool, steps: int = 6) -> Trainer:
    rng = np.random.default_rng(42)
    buf = ReplayBuffer(100, rng=rng)
    for _ in range(20):
        buf.add(_make_sample())
    trainer = Trainer(
        _tiny_net(), device="cpu", lr=1e-2, log_dir=tmp_path / "tb", use_amp=False,
        feature_dropout_p=0.0, swa_start=1 if swa else -1, swa_freq=1,
    )
    trainer.train_steps(buf, batch_size=4, steps=steps)
    return trainer


def _wrap_in_compile(trainer: Trainer) -> None:
    """Move a trained trainer into the production layout: compile, then SWA.

    ``Trainer.__init__`` compiles before ``_init_swa`` on CUDA, so the
    ``AveragedModel`` wraps an ``OptimizedModule`` and its keys read
    ``module._orig_mod.*``. Rebuilding SWA here resets the average, so the
    already-computed one is copied back into the inner module -- by hand rather
    than via the production re-keying helper, which is itself under test.
    """
    swa = trainer._swa_model
    inner = None if swa is None else {k: v.clone() for k, v in swa.module.state_dict().items()}
    n_averaged = None if swa is None else swa.n_averaged.clone()
    trainer.model = torch.compile(trainer.model)
    assert hasattr(trainer.model, "_orig_mod"), "test needs a real OptimizedModule wrap"
    trainer._init_swa()
    if inner is not None:
        trainer._swa_model.module._orig_mod.load_state_dict(inner)
        trainer._swa_model.n_averaged.copy_(n_averaged)


def _unprefixed(sd: dict) -> dict:
    return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}


def test_export_swa_keys_match_checkpoint_keys_under_compile(tmp_path):
    """Published model and checkpoint must agree key-for-key, both prefix-free."""
    trainer = _trained_trainer(tmp_path, swa=True)
    _wrap_in_compile(trainer)

    trainer.save(tmp_path / "ckpt.pt")
    trainer.export_swa(tmp_path / "published.pt")
    ckpt = torch.load(str(tmp_path / "ckpt.pt"), map_location="cpu", weights_only=False)
    pub = torch.load(str(tmp_path / "published.pt"), map_location="cpu", weights_only=False)

    prefixed = [k for k in pub["model"] if "_orig_mod." in k]
    assert not prefixed, f"published model still carries the compile prefix: {prefixed[:3]}"
    assert set(pub["model"]) == set(ckpt["model"]), \
        "publish and checkpoint disagree on key convention"
    # save() claimed wrap-agnosticism for the SWA entry too, but removeprefix
    # could not deliver it: AveragedModel nests the wrap under `module.`.
    assert not [k for k in ckpt["swa_model"] if "_orig_mod." in k]


def test_published_model_loads_strictly_into_an_unwrapped_net(tmp_path):
    """A non-tolerant consumer must get the weights, not a silent fresh-init.

    Before the fix, ``strict=False`` on the published file reported every key
    missing AND unexpected -- no exception, no warning, every tensor left at its
    init value.
    """
    trainer = _trained_trainer(tmp_path, swa=False, steps=4)
    _wrap_in_compile(trainer)
    trainer.export_swa(tmp_path / "published.pt")
    pub = torch.load(str(tmp_path / "published.pt"), map_location="cpu", weights_only=False)

    fresh = _tiny_net()
    incompatible = fresh.load_state_dict(pub["model"], strict=False)
    assert not incompatible.missing_keys, incompatible.missing_keys[:3]
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys[:3]

    trained = _unprefixed(trainer.model.state_dict())
    assert any(not torch.equal(v, _tiny_net().state_dict()[k]) for k, v in trained.items()), \
        "the donor must differ from a fresh init or this test proves nothing"
    for key, value in fresh.state_dict().items():
        assert torch.equal(value, trained[key]), f"{key} did not actually load"


def test_swa_average_survives_a_compile_toggle_across_restart(tmp_path):
    """A wrap-agnostic checkpoint must restore into a COMPILED trainer too.

    Stripping the prefix on write is only half the fix: a compiled
    ``AveragedModel`` expects ``module._orig_mod.*``, so without the load-side
    realignment the resume drops into the 'SWA model state incompatible,
    reinitialising' branch and silently restarts the running average from the
    current weights. That branch logs and carries on, so the caller sees a
    trainer that looks fine.
    """
    donor = _trained_trainer(tmp_path / "donor", swa=True)
    _wrap_in_compile(donor)
    donor.save(tmp_path / "ckpt.pt")
    donor_swa = _unprefixed({k: v.clone() for k, v in donor._swa_model.state_dict().items()})

    for compiled in (True, False):
        resumed = _trained_trainer(tmp_path / f"r{compiled}", swa=True, steps=2)
        if compiled:
            _wrap_in_compile(resumed)
        assert any(
            not torch.equal(_unprefixed(resumed._swa_model.state_dict())[k], v)
            for k, v in donor_swa.items() if v.is_floating_point()
        ), "resumed trainer must start from a DIFFERENT average or the check is vacuous"

        resumed.load(tmp_path / "ckpt.pt")
        restored = _unprefixed(resumed._swa_model.state_dict())
        assert set(restored) == set(donor_swa)
        for key, value in donor_swa.items():
            assert torch.equal(restored[key], value), \
                f"compiled={compiled}: SWA entry {key} was not restored"
