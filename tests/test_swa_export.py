from __future__ import annotations

import logging

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


# ---------------------------------------------------------------------------
# rl_loop_audit J10: the published model is the trained model, not an average.
#
# With SWA on, `export_swa` ships `_swa_model.module` to the workers while
# `save` keeps the RAW model under "model" -- resume has to continue the real
# training trajectory, not an average. The ratchet arena reads that checkpoint
# key, so enabling SWA silently points the strength ruler at a net nobody
# plays (measured in repro: all 86/86 tensors differ).
#
# Deliberately NOT "fixed" by aligning the two: there is no version of aligning
# them that does not break either resume or publish. The rejected alternative
# -- making `load_model_from_checkpoint` prefer `ckpt["swa_model"]` -- would
# silently change which weights EVERY eval tool reads, on a code path with zero
# live coverage. What was available is refusing to let the divergence be quiet.
# ---------------------------------------------------------------------------

def test_export_swa_warns_when_swa_diverges_from_the_checkpoint(tmp_path, caplog):
    """The publish path must say so, loudly, on every publish while SWA is on."""
    trainer = _trained_trainer(tmp_path, swa=True)
    _wrap_in_compile(trainer)

    ckpt_path = tmp_path / "ckpt.pt"
    pub_path = tmp_path / "published.pt"
    trainer.save(ckpt_path)
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.train.trainer"):
        trainer.export_swa(pub_path)

    assert any("J10" in r.getMessage() for r in caplog.records), \
        f"expected a loud SWA divergence warning, got {[r.getMessage() for r in caplog.records]}"

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["model"]
    pub = torch.load(str(pub_path), map_location="cpu", weights_only=False)["model"]
    assert set(ckpt) == set(pub)
    differing = [k for k in ckpt if not torch.equal(ckpt[k], pub[k])]
    assert differing, "the warning must only fire where the nets really can differ"


def test_export_swa_is_quiet_when_swa_is_off(tmp_path, caplog):
    """Production runs ``swa_start: -1`` -- no warning, no behaviour change."""
    trainer = _trained_trainer(tmp_path, swa=False, steps=4)
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.train.trainer"):
        trainer.export_swa(tmp_path / "published.pt")

    assert [r.getMessage() for r in caplog.records if "J10" in r.getMessage()] == []


def test_export_swa_still_strips_the_compile_prefix_under_swa(tmp_path):
    """The warning must not have displaced the J9 strip on the SWA branch."""
    trainer = _trained_trainer(tmp_path, swa=True)
    _wrap_in_compile(trainer)

    trainer.export_swa(tmp_path / "published.pt")
    pub = torch.load(str(tmp_path / "published.pt"), map_location="cpu", weights_only=False)

    assert not [k for k in pub["model"] if "_orig_mod." in k]


# ---------------------------------------------------------------------------
# rl_loop_audit J10 / backlog #57: the publish must SAY which weights it shipped.
#
# J10 records that `export_swa` emits `self.model` today (production runs
# `swa_start: -1`) but would emit `_swa_model.module` with SWA on, while the
# ratchet arena reads the raw model out of `checkpoint_*/trainer.pt` -- measured
# in repro, those differ in 86/86 tensors. The verdict was reached by READING
# the branch. Nothing in the artifact said which side of it ran, so the fact had
# to be re-derived from the config every time anyone asked, and a future change
# to the branch could invalidate it silently.
#
# The line closes that: source, step, tensor count, unique-storage parameter
# count and a content digest, printed on every publish. The digest is what makes
# it an IDENTITY claim rather than a label -- it is recomputable from the
# published file by anyone holding it, which is exactly the observation that
# proves the line describes the bytes that shipped.
#
# ⚑ ASSERTED ON STDOUT, NOT ON caplog, AND THAT IS THE POINT. The trial actor
# installs no logging handler (`_set_log_level` sets a level and nothing else),
# so an INFO record falls to `logging.lastResort` at WARNING+ and is discarded.
# `caplog` installs the handler production lacks, so a caplog assertion proves
# the record was EMITTED and says nothing about whether anyone can SEE it --
# which is the exact "accepted then silently ignored" shape this PR exists to
# close, and the first version of this test had it. `capsys` observes the line
# the way an operator reading the trial's stdout does.
# ---------------------------------------------------------------------------


def _export_log_fields(captured_stdout: str) -> dict[str, str]:
    """Parse the `k=v` tail of the single export_swa provenance line."""
    hits = [
        line for line in captured_stdout.splitlines()
        if line.startswith("[trial] export_swa: wrote ")
    ]
    assert len(hits) == 1, f"expected exactly one provenance line, got {hits}"
    fields = {}
    for part in hits[0].split():
        if "=" in part:
            key, _, value = part.partition("=")
            fields[key] = value
    return fields


def test_the_provenance_line_survives_a_process_with_no_logging_handler() -> None:
    """The production configuration, reproduced: level set, handler absent.

    This is the regression that motivated the print(). `_set_log_level` sets a
    LEVEL on the `chess_anti_engine` logger and installs nothing; no other
    package code or Ray hook attaches a handler to that process. Under exactly
    that setup an INFO record is dropped by `logging.lastResort` (WARNING+)
    while a WARNING gets through -- so a logger-based provenance line would be
    invisible on the very path it was written for.
    """
    import io
    import logging as _logging
    from contextlib import redirect_stderr

    root, pkg = _logging.getLogger(), _logging.getLogger("chess_anti_engine")
    saved = (root.handlers[:], pkg.handlers[:], pkg.level)
    try:
        root.handlers, pkg.handlers = [], []
        pkg.setLevel(_logging.INFO)  # what _set_log_level does, and all it does
        buf = io.StringIO()
        with redirect_stderr(buf):
            _logging.getLogger("chess_anti_engine.train.trainer").info("INFO-PROBE")
            _logging.getLogger("chess_anti_engine.train.trainer").warning("WARN-PROBE")
        assert "WARN-PROBE" in buf.getvalue(), "lastResort should pass WARNING"
        assert "INFO-PROBE" not in buf.getvalue(), (
            "INFO reached output, so this test no longer describes production "
            "-- a handler has appeared; re-check whether print() is still needed"
        )
    finally:
        root.handlers, pkg.handlers, pkg.level = saved[0], saved[1], saved[2]


def test_export_swa_logs_a_digest_that_matches_the_published_file(tmp_path, capsys):
    """The claim must be checkable against the artifact, not merely printed."""
    from chess_anti_engine.train.trainer import (
        state_dict_digest,
        state_dict_unique_param_count,
    )

    trainer = _trained_trainer(tmp_path, swa=False, steps=4)
    pub_path = tmp_path / "published.pt"
    trainer.export_swa(pub_path)

    fields = _export_log_fields(capsys.readouterr().out)
    published = torch.load(str(pub_path), map_location="cpu", weights_only=False)["model"]

    assert fields["source"] == "model"
    assert fields["swa_enabled"] == "False"
    assert fields["digest"] == state_dict_digest(published)
    assert int(fields["tensors"]) == len(published)
    assert int(fields["params"]) == state_dict_unique_param_count(published)
    assert int(fields["step"]) == int(trainer.step)


def test_export_swa_log_names_the_swa_branch_and_its_digest_differs(tmp_path, capsys):
    """The line must distinguish the two sources, which is the whole of J10.

    Both halves in one test on purpose: `source=swa_model.module` is only worth
    anything if the digest ALSO moves, otherwise the label could be right while
    the branch shipped the other object.
    """
    from chess_anti_engine.train.trainer import state_dict_digest

    trainer = _trained_trainer(tmp_path, swa=True)
    # Force the two apart rather than hoping a few steps of SGD did it: with a
    # tiny net and a tiny buffer the running average can land on the current
    # weights, and a test that only sometimes has two distinct objects to tell
    # apart is a test that only sometimes checks anything.
    with torch.no_grad():
        for param in trainer.model.parameters():
            param.add_(1.0)
    raw_digest = state_dict_digest(_unprefixed(trainer.model.state_dict()))
    swa_digest = state_dict_digest(_unprefixed(trainer._swa_model.module.state_dict()))
    assert raw_digest != swa_digest, "test setup failed to separate the two nets"

    pub_path = tmp_path / "published.pt"
    trainer.export_swa(pub_path)

    fields = _export_log_fields(capsys.readouterr().out)
    assert fields["source"] == "swa_model.module"
    assert fields["swa_enabled"] == "True"
    assert fields["digest"] == swa_digest
    assert fields["digest"] != raw_digest, (
        "SWA export digest equals the raw model's -- the log would be labelling "
        "a branch it did not take"
    )
    published = torch.load(str(pub_path), map_location="cpu", weights_only=False)["model"]
    assert fields["digest"] == state_dict_digest(published)


def test_export_swa_provenance_line_fires_on_the_real_publish_path(tmp_path, capsys):
    """The production caller, not a direct `export_swa` call.

    `_publish_distributed_trial_state` is what writes `publish/latest_model.pt`
    for the selfplay fleet. A test that only calls `export_swa` directly proves
    the function logs; it does not prove the line appears where an operator
    would look for it.
    """
    from chess_anti_engine.model import ModelConfig
    from chess_anti_engine.train.trainer import state_dict_digest
    from chess_anti_engine.tune.distributed_runtime import _publish_distributed_trial_state

    trainer = _trained_trainer(tmp_path, swa=False, steps=2)
    model_cfg = ModelConfig(
        kind="transformer", embed_dim=32, num_layers=1, num_heads=2, ffn_mult=2,
        use_smolgen=False, use_nla=False, use_qk_rmsnorm=False,
        use_gradient_checkpointing=False,
    )
    _publish_distributed_trial_state(
        trainer=trainer,
        config={"selfplay_batch": 16, "max_plies": 240, "mcts": "gumbel"},
        model_cfg=model_cfg,
        server_root=tmp_path / "server",
        trial_id="trial_00000",
        training_iteration=3,
        trainer_step=int(trainer.step),
        sf_nodes=1000,
        mcts_simulations=64,
    )

    out = capsys.readouterr().out
    fields = _export_log_fields(out)
    published_path = tmp_path / "server" / "trials" / "trial_00000" / "publish" / "latest_model.pt"
    published = torch.load(str(published_path), map_location="cpu", weights_only=False)["model"]
    assert fields["digest"] == state_dict_digest(published)
    # The PATH must be the published artifact, not some other export: the line
    # is only useful if it names the file the workers will download.
    assert f"[trial] export_swa: wrote {published_path} " in out


def test_unique_param_count_does_not_double_count_tied_weights():
    """The count must obey CLAUDE.md's rule, not `sum(numel())`.

    The production net shares ONE smolgen generator across 16 `state_dict`
    keys, so the naive sum reads 78,812,768 against a true 63,084,128. A
    provenance line reporting the naive number would be a count that does not
    mean what its name says, in the one place this line exists to make
    unambiguous.
    """
    from chess_anti_engine.train.trainer import state_dict_unique_param_count

    shared = torch.zeros(4, 5)
    sd = {"a": shared, "b": shared, "c": torch.zeros(3)}
    assert sum(v.numel() for v in sd.values()) == 43
    assert state_dict_unique_param_count(sd) == 23

    # ...and the dedup must key on STORAGE, not on object identity. Two views
    # of one buffer are distinct objects sharing one allocation, which is what
    # CLAUDE.md's "count unique untyped_storage().data_ptr()" rule is for; an
    # `id()`-keyed version passes the case above and fails here.
    base = torch.zeros(10)
    viewed = {"a": base, "b": base.view(2, 5)}
    assert len({id(v) for v in viewed.values()}) == 2
    assert state_dict_unique_param_count(viewed) == 10


def test_digest_is_key_order_independent_but_value_sensitive():
    """It must survive re-keying and must not survive a weight change.

    Both directions, because a digest that changes on re-ordering would flag
    every harmless dict rebuild, and one that ignores values could not tell the
    SWA average from the raw model at all.
    """
    from chess_anti_engine.train.trainer import state_dict_digest

    a = {"x": torch.ones(2, 2), "y": torch.zeros(3)}
    reordered = {"y": a["y"], "x": a["x"]}
    assert state_dict_digest(a) == state_dict_digest(reordered)

    changed = {"x": torch.ones(2, 2), "y": torch.zeros(3)}
    changed["y"][0] = 1.0
    assert state_dict_digest(a) != state_dict_digest(changed)
