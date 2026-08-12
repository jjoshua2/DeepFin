"""`policy_embedding_shared` — the lc0-style shared policy representation, and
`enable_policy_sf_head` — dropping the permanently-dead `policy_sf` head.

The experiment: lc0 builds ONE policy embedding and hands it to every policy
head, which keeps only its own Q/K. Ours had four fully independent
`AttentionPolicyHead`s projecting straight off the trunk, so the auxiliary policy
losses could only reach `policy_own` through the global trunk -- a representation
that must simultaneously serve the value heads. The shared embedding gives
`soft_policy_ce` a policy-SPECIFIC representation that `policy_own` necessarily
reads.

Two claims these tests exist to defend, both of which have failed silently in
this repo before:

* **The flag reaches production.** yaml -> `TrialConfig` -> `ModelConfig` ->
  model, survives resume, and rebuilds from a saved checkpoint's `arch`.
* **EVERY policy path reads it.** `policy_own` is the search prior, and selfplay
  and UCI reach it through `forward_legal_policy`, `forward_legal_policy_rows`
  and the `_inference_only` branch -- not through the training `forward`. A path
  that skipped the embedding would deploy a different policy than the one that
  was trained, and nothing else would notice.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from chess_anti_engine.model import (
    ARCH_SCHEMA_VERSION,
    ModelConfig,
    build_model,
    model_config_from_manifest_dict,
    model_config_to_manifest_dict,
    resume_model_config_from_arch,
)
from chess_anti_engine.model.transformer import ChessNet
from chess_anti_engine.train.trainer import Trainer
from chess_anti_engine.tune.trainable import _build_trial_model_config
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

_PLANES = 175


def _cfg(
    *, policy_embedding_shared: bool = False, enable_policy_sf_head: bool = True,
) -> ModelConfig:
    return ModelConfig(
        kind="transformer",
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        use_smolgen=False,
        policy_embedding_shared=policy_embedding_shared,
        enable_policy_sf_head=enable_policy_sf_head,
    )


def _build(
    *, policy_embedding_shared: bool = False, enable_policy_sf_head: bool = True,
) -> ChessNet:
    model = build_model(
        _cfg(
            policy_embedding_shared=policy_embedding_shared,
            enable_policy_sf_head=enable_policy_sf_head,
        )
    )
    assert isinstance(model, ChessNet)
    return model.eval()


def _planes(batch: int = 3, seed: int = 1234) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, _PLANES, 8, 8)


def _train_the_embedding(model: ChessNet, scale: float = 0.05) -> ChessNet:
    """Move the embedding off its zero init, the way training would.

    Every "does this path read it" test needs a NON-zero embedding: at init the
    layer is exactly function-preserving by design, so a path that skipped it
    would be indistinguishable.
    """
    emb = model.policy_embedding
    assert emb is not None
    torch.manual_seed(7)
    with torch.no_grad():
        emb.weight.copy_(torch.randn_like(emb.weight) * scale)
        emb.bias.copy_(torch.randn_like(emb.bias) * scale)
    return model


# --------------------------------------------------------------------------
# mechanism
# --------------------------------------------------------------------------


def test_the_shared_embedding_is_exactly_function_preserving_at_init() -> None:
    """⚑ Zero-init + `mish(0) == 0` means the enabled net starts as a bit-identical
    copy of the disabled one, so a warm start from a checkpoint without the layer
    costs nothing. If this drifted, the arm would begin with a free boot shock and
    the readout would measure the shock."""
    off, on = _build(), _build(policy_embedding_shared=True)
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert sorted(missing) == ["policy_embedding.bias", "policy_embedding.weight"]
    assert not unexpected

    x = _planes()
    with torch.no_grad():
        a, b = off(x), on(x)
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_the_shared_embedding_is_genuinely_NONLINEAR() -> None:
    """⚑ The reason the branch is `t + mish(W t + b)` and not an identity-init
    `nn.Linear`.

    A linear shared layer composes with each head's linear Q/K into another
    linear map: it adds ZERO representational content and would test only
    gradient coupling -- the mechanism the 2026-08-12 grad-share probe already
    rejected (soft 94.1% vs own 94.7% trunk share).

    ⚑ Plain homogeneity (`f(2t) == 2 f(t)`) is NOT the discriminator, and a mutant
    that drops `mish` survives it: `t + W t + b` is AFFINE, so its bias already
    breaks homogeneity while the map stays exactly as absorbable as an identity.
    Subtracting `f(0)` removes the bias, and the residual is linear for the affine
    mutant and still curved here.
    """
    model = _train_the_embedding(_build(policy_embedding_shared=True), scale=0.5)
    t = torch.randn(2, 64, 64)
    with torch.no_grad():
        at_zero = model._policy_tokens(torch.zeros_like(t))
        once = model._policy_tokens(t) - at_zero
        twice = model._policy_tokens(2 * t) - at_zero
    assert not torch.allclose(twice, 2 * once, atol=1e-4)


def test_disabled_the_embedding_is_not_built_and_the_tokens_pass_through() -> None:
    model = _build()
    assert model.policy_embedding is None
    t = torch.randn(2, 64, 64)
    assert model._policy_tokens(t) is t


def _grad_norm_on_embedding(model: ChessNet, output_key: str) -> float:
    emb = model.policy_embedding
    assert emb is not None
    out = model(_planes())
    # `materialize_grads` turns "this output does not depend on the parameter"
    # into an explicit zero tensor rather than None, so the value-head case below
    # reads as a measured 0.0 instead of a missing measurement.
    (grad,) = torch.autograd.grad(
        out[output_key].square().sum(), emb.weight,
        allow_unused=True, materialize_grads=True,
    )
    return float(grad.norm())


_POLICY_OUTPUT_KEYS = ("policy_own", "policy_soft", "policy_sf", "policy_future")
_VALUE_OUTPUT_KEYS = (
    "wdl", "sf_eval", "categorical", "volatility", "sf_volatility", "moves_left",
)


@pytest.mark.parametrize("key", _POLICY_OUTPUT_KEYS)
def test_EVERY_policy_head_gradient_REACHES_the_shared_representation(key: str) -> None:
    """⚑ THE EXPERIMENT'S WHOLE HYPOTHESIS, stated as a measurement.

    Every policy head's loss must land on a parameter that `policy_own` -- the
    head MCTS reads as its search prior -- also reads. Positive norms on the SAME
    tensor from ALL of them is exactly the claim "one shared policy
    representation for EVERY policy head".

    ⚑ Parametrized over the heads, not just the four CALL PATHS. Review found
    that path coverage alone let `policy_future` and `policy_sf` silently revert
    to the raw trunk with all 31 tests still passing -- and a head detached from
    the shared layer makes the arm measure something other than sharing.
    """
    model = _train_the_embedding(_build(policy_embedding_shared=True))
    assert _grad_norm_on_embedding(model, key) > 0.0


@pytest.mark.parametrize("key", _VALUE_OUTPUT_KEYS)
def test_NO_value_head_reads_the_shared_policy_embedding(key: str) -> None:
    """The other half of the hypothesis: the representation is policy-SPECIFIC.
    If any value output read it, this would be another trunk layer and the arm
    would be testing depth, not sharing. Parametrized over ALL of them for the
    same reason as above -- checking only `wdl` and `sf_eval` left four
    unpinned."""
    model = _train_the_embedding(_build(policy_embedding_shared=True))
    assert _grad_norm_on_embedding(model, key) == 0.0


# --------------------------------------------------------------------------
# ⚑ every policy path
# --------------------------------------------------------------------------


def _policy_from_path(model: ChessNet, path: str, x: torch.Tensor) -> torch.Tensor:
    if path == "forward":
        return model(x)["policy_own"]
    if path == "inference_only":
        model._inference_only = True
        try:
            return model(x)["policy_own"]
        finally:
            model._inference_only = False
    n = x.shape[0]
    legal_flat = torch.arange(4 * n, dtype=torch.long) % 1858
    if path == "forward_legal_policy":
        counts = torch.full((n,), 4, dtype=torch.long)
        return model.forward_legal_policy(x, legal_flat, counts)["policy_own"]
    if path == "forward_legal_policy_rows":
        rows = torch.repeat_interleave(torch.arange(n, dtype=torch.long), 4)
        return model.forward_legal_policy_rows(x, legal_flat, rows)["policy_own"]
    raise AssertionError(path)


@pytest.mark.parametrize(
    "path",
    ["forward", "inference_only", "forward_legal_policy", "forward_legal_policy_rows"],
)
def test_EVERY_policy_path_reads_the_shared_embedding(path: str) -> None:
    """⚑⚑ THE DEPLOYMENT TEST. `policy_own` is the search prior, and selfplay/UCI
    do NOT take the training `forward`: they take the compact-legal paths and the
    `_inference_only` branch. A path that kept projecting off the raw trunk would
    make the deployed prior diverge from the trained one -- silently, with every
    loss curve and every unit test still green.

    Measured by zeroing the embedding back to its identity and requiring the
    path's output to CHANGE. A path that never called it is unaffected.
    """
    x = _planes()
    trained = _train_the_embedding(_build(policy_embedding_shared=True))
    with torch.no_grad():
        with_embedding = _policy_from_path(trained, path, x).clone()
        emb = trained.policy_embedding
        assert emb is not None
        emb.weight.zero_()
        emb.bias.zero_()
        without = _policy_from_path(trained, path, x)
    assert not torch.allclose(with_embedding, without, atol=1e-6)


def test_the_compact_legal_paths_agree_with_the_full_forward() -> None:
    """The four paths must be the SAME policy, not merely all non-trivial: a
    fresh `_policy_tokens` call per path is only correct if it is the same map."""
    model = _train_the_embedding(_build(policy_embedding_shared=True))
    x = _planes(batch=2)
    with torch.no_grad():
        full = model(x)["policy_own"]
        model._inference_only = True
        inf = model(x)["policy_own"]
        model._inference_only = False
        legal_flat = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        counts = torch.full((2,), 4, dtype=torch.long)
        compact = model.forward_legal_policy(x, legal_flat, counts)["policy_own"]
    assert torch.allclose(full, inf, atol=1e-6)
    gathered = torch.stack([full[0, :4], full[1, 4:8]])
    assert torch.allclose(compact.reshape(2, 4), gathered, atol=1e-5)


# --------------------------------------------------------------------------
# policy_sf: conditional construction
# --------------------------------------------------------------------------


def test_disabling_policy_sf_removes_the_head_its_output_and_its_weights() -> None:
    """~530k parameters that receive zero gradient at `w_sf_move: 0.0` and are
    still built, checkpointed and published every iteration."""
    on, off = _build(), _build(enable_policy_sf_head=False)
    assert on.policy_sf is not None
    assert off.policy_sf is None
    assert "policy_sf" in on(_planes())
    assert "policy_sf" not in off(_planes())
    assert not [k for k in off.state_dict() if k.startswith("policy_sf.")]
    assert sum(p.numel() for p in on.parameters()) > sum(
        p.numel() for p in off.parameters()
    )


def _make_trainer(model: ChessNet, log_dir: Path, w_sf_move: float) -> Trainer:
    return Trainer(
        model, device="cpu", lr=1e-3, optimizer="adamw", warmup_steps=10,
        warmup_lr_start=1e-5, use_amp=False, log_dir=log_dir,
        tb_log_interval=1000, prefetch_batches=False, w_sf_move=w_sf_move,
    )


def test_a_model_that_never_had_the_head_is_NOT_refused(tmp_path: Path) -> None:
    """⚑ REGRESSION for the review's first ship-blocker. Triggering on
    `policy_sf is None` refused to construct 107 existing tests' models -- every
    test double and partial-model rig lacks the attribute. The guard must key on
    the DELIBERATE `enable_policy_sf_head=False` choice, which only a `ChessNet`
    records, so a model that never made the choice is left alone at the default
    `w_sf_move=0.15`.
    """
    class _Double(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"policy_own": self.lin(x)}

    trainer = Trainer(
        _Double(), device="cpu", lr=1e-3, optimizer="adamw", warmup_steps=10,
        warmup_lr_start=1e-5, use_amp=False, log_dir=tmp_path / "d",
        tb_log_interval=1000, prefetch_batches=False,
    )
    assert float(trainer._loss_kwargs["w_sf_move"]) == 0.15


def test_RAISING_w_sf_move_AFTER_construction_still_fails_closed(
    tmp_path: Path,
) -> None:
    """⚑⚑ REGRESSION for the review's second ship-blocker: a construction-time
    check CANNOT FIRE for the scenario the guard exists for.

    `w_sf_move` is in `TRAINER_WEIGHT_KEYS`, and `_sync_trainer_weights` does
    `setattr(trainer, wk, float(config[wk]))` EVERY ITERATION so live-yaml and
    PB2 changes take effect immediately; `_apply_donor_config_overlay` is a
    second post-construction writer. Measured before the fix: construct at 0.0,
    sync to 0.15, `sf_move_ce` 0.0, no exception -- a gate that cannot fail.

    So the check lives in `_loss_kwargs`, which is rebuilt on every step. This
    test performs the setattr the sync loop performs.
    """
    trainer = _make_trainer(_build(enable_policy_sf_head=False), tmp_path / "s", 0.0)
    trainer.w_sf_move = 0.15  # exactly what `_sync_trainer_weights` does
    with pytest.raises(ValueError, match="w_sf_move"):
        _ = trainer._loss_kwargs


def test_a_positive_w_sf_move_on_a_net_without_the_head_FAILS_CLOSED(
    tmp_path: Path,
) -> None:
    """⚑ This repo's signature defect is a value accepted and then silently
    ignored. `compute_loss` contributes zero when `policy_sf` is absent -- correct
    for partial-model rigs, but with conditional construction it would let someone
    turn the teacher on in the live yaml against a net that has no head to train,
    see a clean run, and read the whole readout window as if it had trained.

    Checked ONCE at Trainer construction, so it costs no compute and fires before
    any is spent. `w_sf_own` / `w_sf_own_regret` are deliberately NOT part of it:
    they train `policy_own` via the sf_p0 terms, not this head.
    """
    with pytest.raises(ValueError, match="w_sf_move"):
        _make_trainer(_build(enable_policy_sf_head=False), tmp_path / "a", 0.15)


def test_the_guard_does_not_fire_when_the_head_exists_or_the_weight_is_zero(
    tmp_path: Path,
) -> None:
    """Negative control, both directions: a guard that fires unconditionally is
    not a guard, and one that cannot fire is not one either."""
    assert _make_trainer(_build(), tmp_path / "b", 0.15) is not None
    assert _make_trainer(_build(enable_policy_sf_head=False), tmp_path / "c", 0.0) is not None


def test_compute_loss_still_tolerates_the_absent_head() -> None:
    """The tolerance the guard replaces must stay: partial-model rigs depend on
    `absent optional head -> zero loss`."""
    from chess_anti_engine.train.losses import compute_loss

    model = _build(enable_policy_sf_head=False)
    x = _planes(batch=4)
    outputs = model(x)
    batch: dict[str, torch.Tensor] = {
        "x": x,
        "policy_t": torch.full((4, 1858), 1.0 / 1858),
        "wdl_t": torch.zeros(4, dtype=torch.long),
    }
    assert float(compute_loss(outputs, batch, w_sf_move=0.15)["sf_move_ce"]) == 0.0


# --------------------------------------------------------------------------
# plumbing: the flags must reach production and survive every hop
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["policy_embedding_shared", "enable_policy_sf_head"])
def test_flags_survive_the_manifest_round_trip(field: str) -> None:
    for want in (True, False):
        manifest = model_config_to_manifest_dict(dataclasses.replace(_cfg(), **{field: want}))
        assert manifest[field] is want
        assert getattr(model_config_from_manifest_dict(manifest), field) is want


@pytest.mark.parametrize("field", ["policy_embedding_shared", "enable_policy_sf_head"])
@pytest.mark.parametrize("value", [True, False])
def test_flags_reach_the_model_config_production_builds(field: str, value: bool) -> None:
    """⚑ yaml -> TrialConfig -> ModelConfig. `model_config_from_flat_config` is
    NOT this path -- it is only reachable from `scripts/`, so a key that stops at
    the flat dict is dead in training while looking wired (measured on
    `categorical_head_coupled`, PR #397)."""
    flat = flatten_run_config_defaults({"model": {field: value}, "train": {}})
    built = _build_trial_model_config(TrialConfig.from_dict(flat))
    assert getattr(built, field) is value


@pytest.mark.parametrize("field", ["policy_embedding_shared", "enable_policy_sf_head"])
def test_flags_are_accepted_by_the_yaml_schema(field: str) -> None:
    """Category-(a): a live-yaml key absent from the schema makes
    `flatten_run_config_defaults` raise, and it runs before the argument parser
    and outside any try -- the process would not boot."""
    assert flatten_run_config_defaults({"model": {field: True}, "train": {}})[field] is True
    with pytest.raises(ValueError, match=f"{field}_typo"):
        flatten_run_config_defaults({"model": {f"{field}_typo": True}, "train": {}})


@pytest.mark.parametrize("field", ["policy_embedding_shared", "enable_policy_sf_head"])
@pytest.mark.parametrize(("checkpoint_value", "config_value"), [(False, True), (True, False)])
def test_topology_migration_survives_resume(
    field: str, checkpoint_value: bool, config_value: bool,
) -> None:
    """⚑ Resume takes topology from the checkpoint's `arch`, so a flag outside
    `_RESUME_CONFIG_OWNED_ENCODING_KEYS` reverts to the donor on EVERY resume --
    silently, because the tolerant loader is happy either way."""
    donor = dataclasses.replace(_cfg(), **{field: checkpoint_value})
    arch = dataclasses.asdict(donor)
    arch["_schema_version"] = ARCH_SCHEMA_VERSION
    run_cfg = dataclasses.replace(donor, **{field: config_value})
    assert getattr(resume_model_config_from_arch(arch, run_cfg), field) is config_value


def test_resume_still_takes_real_topology_from_the_checkpoint() -> None:
    """Negative control: widening the config-owned list must not let a SHAPE key
    escape the checkpoint."""
    donor = dataclasses.replace(_cfg(policy_embedding_shared=True), num_layers=1)
    arch = dataclasses.asdict(donor)
    arch["_schema_version"] = ARCH_SCHEMA_VERSION
    run_cfg = dataclasses.replace(donor, num_layers=2)
    assert resume_model_config_from_arch(arch, run_cfg).num_layers == 1


@pytest.mark.parametrize("field", ["policy_embedding_shared", "enable_policy_sf_head"])
def test_the_flags_are_REFUSED_by_a_live_yaml_reload(field: str) -> None:
    """⚑⚑ HOW YOU TURN THESE ON, AND HOW YOU CANNOT. Every `ModelConfig` field is
    construction-bound, so editing the live yaml and restarting with `--resume`
    logs a WARNING, comes up healthy, and trains the OLD topology for the whole
    readout window. ⇒ needs a FRESH (non-resumed) trial."""
    from chess_anti_engine.tune.trainable_config_ops import _RESUME_CONSTRUCTION_BOUND_KEYS

    assert field in _RESUME_CONSTRUCTION_BOUND_KEYS


def test_the_arena_loader_rebuilds_both_flags_from_a_saved_checkpoint(
    tmp_path: Path,
) -> None:
    """⚑ THE VERDICT PATH. `scripts/arena_standard.py` and UCI both go through
    `load_model_from_checkpoint`, which builds topology from the checkpoint's
    embedded `arch` and never from a yaml. Defaulted, the arena would score a net
    whose search prior is read straight off the trunk and report it as the
    shared-embedding arm -- a wrong verdict, not a crash.

    Written against a real `Trainer.save` payload, because the claim is about
    what production writes.
    """
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    cfg = _cfg(policy_embedding_shared=True, enable_policy_sf_head=False)
    # ⚑ `w_sf_move=0.0` is REQUIRED here, not incidental: the Trainer default is
    # 0.15, and the fail-closed guard rejects a positive weight against a net with
    # no `policy_sf` head. Production's live yaml already sets 0.0, but any fresh
    # trial that turns `enable_policy_sf_head` off must set it explicitly.
    trainer = Trainer(
        build_model(cfg), device="cpu", lr=1e-3, optimizer="adamw", warmup_steps=10,
        warmup_lr_start=1e-5, use_amp=False, log_dir=tmp_path / "log",
        tb_log_interval=1000, prefetch_batches=False, model_config=cfg, w_sf_move=0.0,
    )
    ckpt = tmp_path / "shared.pt"
    trainer.save(ckpt)

    loaded = load_model_from_checkpoint(ckpt, device="cpu")
    assert isinstance(loaded, ChessNet)
    assert loaded.policy_embedding is not None
    assert loaded.policy_sf is None


def test_enabling_the_embedding_PRESERVES_optimizer_state(tmp_path: Path) -> None:
    """⚑ The confound this removes, and why it is worth a test.

    Without `policy_embedding.*` in `_FRESH_PARAM_NAME_SUFFIXES`, warm-starting
    with the flag on falls into `Trainer.load`'s reinit path, which resets EVERY
    moment, re-enters LR warmup, AND skips `load_zclip_state` (so the arm starts
    in the fresh hard-cap-only zclip regime). The control arm would then have to
    absorb an identical boot shock just to stay comparable, and the readout would
    be measuring the shock. Spliced, the donor moments survive.

    Measured 86 -> 86 with the splice, 86 -> 0 without.
    """
    donor_cfg = _cfg()
    donor = Trainer(
        build_model(donor_cfg), device="cpu", lr=1e-3, optimizer="adamw",
        warmup_steps=10, warmup_lr_start=1e-5, use_amp=False,
        log_dir=tmp_path / "donor", tb_log_interval=1000, prefetch_batches=False,
        model_config=donor_cfg,
    )
    for param in donor.model.parameters():
        param.grad = torch.randn_like(param)
    donor.opt.step()
    ckpt = tmp_path / "donor.pt"
    donor.save(ckpt)
    banked = len(donor.opt.state)
    assert banked > 0

    arm_cfg = _cfg(policy_embedding_shared=True)
    arm = Trainer(
        build_model(arm_cfg), device="cpu", lr=1e-3, optimizer="adamw",
        warmup_steps=10, warmup_lr_start=1e-5, use_amp=False,
        log_dir=tmp_path / "arm", tb_log_interval=1000, prefetch_batches=False,
        model_config=arm_cfg,
    )
    arm.load(ckpt)
    assert len(arm.opt.state) >= banked, "optimizer moments were reset, not spliced"


def test_the_AOT_path_REFUSES_to_serve_a_prior_without_the_adapter(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE FIFTH POLICY PATH, which the PR originally missed.

    The live yaml sets `distributed_inference_aot_dir: data/aot_models_512`. With
    it set, the broker serves the prior from a pre-compiled AOTInductor graph and
    never enters `ChessNet.forward`, so the shared adapter is bypassed --
    silently, because `build_aot_constants` drops model constants the package did
    not ask for, and `--verify` builds its reference from the same yaml so package
    and reference lack the layer together.

    Zero-init makes it worse, not better: selfplay would start in EXACT agreement
    with training and diverge as the layer trains, and AOT covers only exact batch
    buckets so uncovered sizes fall through to the eager path WITH the adapter --
    a served policy that is a mixture keyed on batch size.
    """
    from chess_anti_engine.tune.distributed_runtime import _launch_inference_broker

    with pytest.raises(ValueError, match="distributed_inference_aot_dir"):
        _launch_inference_broker(
            config={
                "policy_embedding_shared": True,
                "distributed_inference_aot_dir": "data/aot_models_512",
                "distributed_server_root": str(tmp_path / "srv"),
            },
            trial_id="t",
            publish_dir=tmp_path / "pub",
            trial_dir=tmp_path / "trial",
        )


@pytest.mark.parametrize(
    ("shared", "aot_dir"), [(False, "data/aot_models_512"), (True, "")],
)
def test_the_AOT_refusal_does_not_fire_on_either_safe_combination(
    shared: bool, aot_dir: str, tmp_path: Path,
) -> None:
    """Negative control. A guard that refuses every launch is not a guard --
    production runs AOT today with the adapter off, and the adapter is safe when
    AOT is not configured. Only the CONJUNCTION is the hazard.

    Asserted by the absence of the AOT ValueError specifically: the call fails
    later for unrelated reasons (no real server root), which is fine.
    """
    from chess_anti_engine.tune.distributed_runtime import _launch_inference_broker

    raised = ""
    try:
        _launch_inference_broker(
            config={
                "policy_embedding_shared": shared,
                "distributed_inference_aot_dir": aot_dir,
                "distributed_server_root": str(tmp_path / "srv"),
            },
            trial_id="t",
            publish_dir=tmp_path / "pub",
            trial_dir=tmp_path / "trial",
        )
    except Exception as exc:  # the launch fails for unrelated reasons; only the AOT text matters
        raised = f"{type(exc).__name__}: {exc}"
    assert "distributed_inference_aot_dir" not in raised, raised


def test_arch_schema_version_was_bumped_for_these_fields() -> None:
    """Defaulting either field builds a different architecture, which is exactly
    what the constant documents itself for."""
    assert ARCH_SCHEMA_VERSION >= 19
