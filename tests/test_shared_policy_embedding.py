"""`policy_embedding_mode` — the shared policy adapter, and
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
import subprocess
from pathlib import Path
from typing import Any

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


# The two adapter modes that actually build a layer. Both are function-preserving at
# init; they differ in WHICH mechanism the arm tests -- see `_policy_tokens`.
_ADAPTER_MODES = ("linear", "residual_mish")


def _cfg(
    *, policy_embedding_mode: str = "off", enable_policy_sf_head: bool = True,
) -> ModelConfig:
    return ModelConfig(
        kind="transformer",
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        use_smolgen=False,
        policy_embedding_mode=policy_embedding_mode,
        enable_policy_sf_head=enable_policy_sf_head,
    )


def _build(
    *, policy_embedding_mode: str = "off", enable_policy_sf_head: bool = True,
) -> ChessNet:
    model = build_model(
        _cfg(
            policy_embedding_mode=policy_embedding_mode,
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


@pytest.mark.parametrize("mode", _ADAPTER_MODES)
def test_the_shared_embedding_is_exactly_function_preserving_at_init(mode: str) -> None:
    """⚑ Zero-init + `mish(0) == 0` means the enabled net starts as a bit-identical
    copy of the disabled one, so a warm start from a checkpoint without the layer
    costs nothing. If this drifted, the arm would begin with a free boot shock and
    the readout would measure the shock."""
    off, on = _build(), _build(policy_embedding_mode=mode)
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert sorted(missing) == ["policy_embedding.bias", "policy_embedding.weight"]
    assert not unexpected

    x = _planes()
    with torch.no_grad():
        a, b = off(x), on(x)
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="production dtype path is CUDA-only")
@pytest.mark.parametrize("mode", _ADAPTER_MODES)
def test_function_preservation_HOLDS_IN_THE_PRODUCTION_DTYPE(mode: str) -> None:
    """⚑ The CPU/FP32 `torch.equal` above is not the deployed arithmetic.

    Production runs under CUDA + bf16 autocast (`Trainer._amp_context` pins
    bf16). `linear` mode inserts an actual identity GEMM where there was no GEMM
    at all, so "W = I is mathematically a no-op" is an argument about real
    numbers, not about a bf16 tensor core accumulating 512 products. If the
    insertion cost even one extra rounding of `t`, the arm would open with a free
    numerical step against its control -- small, but attributed to the
    architecture change by construction.

    `residual_mish` gets the same check because `mish(0) == 0` is likewise exact
    in fp32 and worth confirming rather than assuming in bf16.
    """
    off = _build().cuda()
    on = _build(policy_embedding_mode=mode).cuda()
    on.load_state_dict(off.state_dict(), strict=False)

    x = _planes().cuda()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        a, b = off(x), on(x)
    for key in a:
        assert torch.equal(a[key], b[key]), (
            f"{key} differs under CUDA/bf16 by "
            f"{(a[key].float() - b[key].float()).abs().max().item():.3e} "
            "-- the adapter is NOT function-preserving in the deployed dtype"
        )


def test_the_shared_embedding_is_genuinely_NONLINEAR() -> None:
    """⚑ The reason the branch is `t + mish(W t + b)` and not an identity-init
    `nn.Linear`.

    A linear shared layer composes with each head's linear Q/K into another
    linear map, so it adds ZERO representational CAPACITY. ⚑ That does not make
    it inert, and an earlier version of this docstring wrongly said it "would
    test only gradient coupling -- the mechanism the grad-share probe already
    rejected". Equal function class is not equal optimization: `A_h S` and `A'_h`
    reach different points under the same gradient steps. `linear` is the arm
    that isolates that reparameterization geometry; see the ISOLATION-arm test
    below. Only the CAPACITY mechanism is unique to `residual_mish`.

    ⚑ Plain homogeneity (`f(2t) == 2 f(t)`) is NOT the discriminator, and a mutant
    that drops `mish` survives it: `t + W t + b` is AFFINE, so its bias already
    breaks homogeneity while the map stays exactly as absorbable as an identity.
    Subtracting `f(0)` removes the bias, and the residual is linear for the affine
    mutant and still curved here.
    """
    model = _train_the_embedding(_build(policy_embedding_mode="residual_mish"), scale=0.5)
    t = torch.randn(2, 64, 64)
    with torch.no_grad():
        at_zero = model._policy_tokens(torch.zeros_like(t))
        once = model._policy_tokens(t) - at_zero
        twice = model._policy_tokens(2 * t) - at_zero
    assert not torch.allclose(twice, 2 * once, atol=1e-4)


def test_LINEAR_mode_is_exactly_affine_and_therefore_the_ISOLATION_arm() -> None:
    """⚑ The complement of the nonlinearity test, and the reason both modes exist.

    `linear` is `W t + b`. It spans the same function class as each head's Q/K on
    the raw trunk, so it adds NO representational capacity -- which is precisely
    what makes it the clean arm: it isolates **reparameterization geometry** (a
    policy-only factorization `A_h S` trains on a different trajectory than
    `A'_h`, even at equal function class) from the **capacity** that
    `residual_mish` also adds.

    ⚑ Affine, not linear: the bias means `f(2t) != 2 f(t)`. Subtracting `f(0)`
    removes it, and THAT residual must be exactly homogeneous here -- the same
    discriminator the nonlinearity test uses, read in the opposite direction.
    """
    model = _train_the_embedding(_build(policy_embedding_mode="linear"), scale=0.5)
    t = torch.randn(2, 64, 64)
    with torch.no_grad():
        at_zero = model._policy_tokens(torch.zeros_like(t))
        once = model._policy_tokens(t) - at_zero
        twice = model._policy_tokens(2 * t) - at_zero
    assert torch.allclose(twice, 2 * once, atol=1e-4)


def test_an_unknown_mode_RAISES_rather_than_falling_back_to_off() -> None:
    """⚑ A silent fallback to "off" is this codebase's signature defect: the arm
    would come up healthy, train the CONTROL topology, and the whole readout
    window would be recorded against the wrong architecture."""
    with pytest.raises(ValueError, match="policy_embedding_mode"):
        _build(policy_embedding_mode="residual_gelu")
    with pytest.raises(ValueError, match="policy_embedding_mode"):
        _build(policy_embedding_mode="shared")


def test_the_two_modes_build_DIFFERENT_functions_once_trained() -> None:
    """Same layer shape, same weights, different map -- otherwise the A/B/C
    comparison would be measuring nothing."""
    lin = _train_the_embedding(_build(policy_embedding_mode="linear"))
    res = _train_the_embedding(_build(policy_embedding_mode="residual_mish"))
    res.load_state_dict(lin.state_dict(), strict=False)
    x = _planes()
    with torch.no_grad():
        assert not torch.allclose(lin(x)["policy_own"], res(x)["policy_own"], atol=1e-5)


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


@pytest.mark.parametrize("mode", _ADAPTER_MODES)
@pytest.mark.parametrize("key", _POLICY_OUTPUT_KEYS)
def test_EVERY_policy_head_gradient_REACHES_the_shared_representation(
    key: str, mode: str,
) -> None:
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
    model = _train_the_embedding(_build(policy_embedding_mode=mode))
    assert _grad_norm_on_embedding(model, key) > 0.0


@pytest.mark.parametrize("mode", _ADAPTER_MODES)
@pytest.mark.parametrize("key", _VALUE_OUTPUT_KEYS)
def test_NO_value_head_reads_the_shared_policy_embedding(key: str, mode: str) -> None:
    """The other half of the hypothesis: the representation is policy-SPECIFIC.
    If any value output read it, this would be another trunk layer and the arm
    would be testing depth, not sharing. Parametrized over ALL of them for the
    same reason as above -- checking only `wdl` and `sf_eval` left four
    unpinned."""
    model = _train_the_embedding(_build(policy_embedding_mode=mode))
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


@pytest.mark.parametrize("mode", _ADAPTER_MODES)
@pytest.mark.parametrize(
    "path",
    ["forward", "inference_only", "forward_legal_policy", "forward_legal_policy_rows"],
)
def test_EVERY_policy_path_reads_the_shared_embedding(path: str, mode: str) -> None:
    """⚑⚑ THE DEPLOYMENT TEST. `policy_own` is the search prior, and selfplay/UCI
    do NOT take the training `forward`: they take the compact-legal paths and the
    `_inference_only` branch. A path that kept projecting off the raw trunk would
    make the deployed prior diverge from the trained one -- silently, with every
    loss curve and every unit test still green.

    Measured by zeroing the embedding back to its identity and requiring the
    path's output to CHANGE. A path that never called it is unaffected.
    """
    x = _planes()
    trained = _train_the_embedding(_build(policy_embedding_mode=mode))
    with torch.no_grad():
        with_embedding = _policy_from_path(trained, path, x).clone()
        emb = trained.policy_embedding
        assert emb is not None
        emb.weight.zero_()
        emb.bias.zero_()
        without = _policy_from_path(trained, path, x)
    assert not torch.allclose(with_embedding, without, atol=1e-6)


@pytest.mark.parametrize("mode", _ADAPTER_MODES)
def test_the_compact_legal_paths_agree_with_the_full_forward(mode: str) -> None:
    """The four paths must be the SAME policy, not merely all non-trivial: a
    fresh `_policy_tokens` call per path is only correct if it is the same map."""
    model = _train_the_embedding(_build(policy_embedding_mode=mode))
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


def _empty_buffer() -> Any:
    """Minimal ReplayBuffer stand-in: the guard must fire BEFORE any batch is
    drawn, so this never needs to yield one. Typed `Any` because the point is
    precisely that `train_steps` must not get far enough to use it."""

    class _EmptyBuffer:
        def __len__(self) -> int:
            return 0

        def sample(self, *_a: object, **_kw: object) -> None:  # pragma: no cover
            raise AssertionError("the guard must fire before any batch is drawn")

    return _EmptyBuffer()


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

    ⚑ The check does NOT live in `_loss_kwargs`, though that was the first fix:
    `eval_ruler_id_for` derives the holdout ruler's identity from
    `call_closure(_compute_metrics)`, which reaches `_loss_kwargs`, so putting a
    frame there CHANGED THE RULER ID -- invalidating every best-model comparison
    across it. It lives at the top of `train_steps` instead: same every-iteration
    coverage, training-only call graph. This test performs the setattr the sync
    loop performs and drives the real entry point.
    """
    trainer = _make_trainer(_build(enable_policy_sf_head=False), tmp_path / "s", 0.0)
    trainer.w_sf_move = 0.15  # exactly what `_sync_trainer_weights` does
    with pytest.raises(ValueError, match="w_sf_move"):
        trainer.train_steps(_empty_buffer(), batch_size=2, steps=1)


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


_FIELD_VALUES = (
    ("policy_embedding_mode", ("off", "linear", "residual_mish")),
    ("enable_policy_sf_head", (True, False)),
)


@pytest.mark.parametrize(("field", "values"), _FIELD_VALUES)
def test_flags_survive_the_manifest_round_trip(field: str, values: tuple) -> None:
    for want in values:
        manifest = model_config_to_manifest_dict(dataclasses.replace(_cfg(), **{field: want}))
        assert manifest[field] == want
        assert getattr(model_config_from_manifest_dict(manifest), field) == want


@pytest.mark.parametrize(("field", "values"), _FIELD_VALUES)
def test_flags_reach_the_model_config_production_builds(field: str, values: tuple) -> None:
    """⚑ yaml -> TrialConfig -> ModelConfig. `model_config_from_flat_config` is
    NOT this path -- it is only reachable from `scripts/`, so a key that stops at
    the flat dict is dead in training while looking wired (measured on
    `categorical_head_coupled`, PR #397)."""
    for value in values:
        flat = flatten_run_config_defaults({"model": {field: value}, "train": {}})
        built = _build_trial_model_config(TrialConfig.from_dict(flat))
        assert getattr(built, field) == value


@pytest.mark.parametrize("field", ["policy_embedding_mode", "enable_policy_sf_head"])
def test_flags_are_accepted_by_the_yaml_schema(field: str) -> None:
    """Category-(a): a live-yaml key absent from the schema makes
    `flatten_run_config_defaults` raise, and it runs before the argument parser
    and outside any try -- the process would not boot."""
    assert field in flatten_run_config_defaults({"model": {field: True}, "train": {}})
    with pytest.raises(ValueError, match=f"{field}_typo"):
        flatten_run_config_defaults({"model": {f"{field}_typo": True}, "train": {}})


@pytest.mark.parametrize(
    ("field", "checkpoint_value", "config_value"),
    [
        ("policy_embedding_mode", "off", "residual_mish"),
        ("policy_embedding_mode", "residual_mish", "off"),
        ("policy_embedding_mode", "linear", "residual_mish"),
        ("enable_policy_sf_head", False, True),
        ("enable_policy_sf_head", True, False),
    ],
)
def test_topology_migration_survives_resume(
    field: str, checkpoint_value: object, config_value: object,
) -> None:
    """⚑ Resume takes topology from the checkpoint's `arch`, so a flag outside
    `_RESUME_CONFIG_OWNED_ENCODING_KEYS` reverts to the donor on EVERY resume --
    silently, because the tolerant loader is happy either way."""
    donor = dataclasses.replace(_cfg(), **{field: checkpoint_value})
    arch = dataclasses.asdict(donor)
    arch["_schema_version"] = ARCH_SCHEMA_VERSION
    run_cfg = dataclasses.replace(donor, **{field: config_value})
    assert getattr(resume_model_config_from_arch(arch, run_cfg), field) == config_value


@pytest.mark.parametrize("mode", _ADAPTER_MODES)
def test_resume_still_takes_real_topology_from_the_checkpoint(mode: str) -> None:
    """Negative control: widening the config-owned list must not let a SHAPE key
    escape the checkpoint."""
    donor = dataclasses.replace(_cfg(policy_embedding_mode=mode), num_layers=1)
    arch = dataclasses.asdict(donor)
    arch["_schema_version"] = ARCH_SCHEMA_VERSION
    run_cfg = dataclasses.replace(donor, num_layers=2)
    assert resume_model_config_from_arch(arch, run_cfg).num_layers == 1


@pytest.mark.parametrize("field", ["policy_embedding_mode", "enable_policy_sf_head"])
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

    cfg = _cfg(policy_embedding_mode="residual_mish", enable_policy_sf_head=False)
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


@pytest.mark.parametrize("mode", _ADAPTER_MODES)
@pytest.mark.parametrize(
    ("optimizer", "scope"),
    [
        ("adamw", "default"),
        # ⚑ PRODUCTION's layout, and the only one where the splice's per-group
        # positional remap can actually go wrong. `matrix_optimizer_scope:
        # mlp_out` puts block FFNs and attention out-projections in the Aurora
        # group and everything else in AdamW, so there are TWO groups to remap
        # independently -- and `policy_embedding.weight` matches NEITHER
        # `.ffn.` NOR `.out_proj.` inside `blocks.`, so the new parameter lands
        # in the AdamW group while the Aurora group's positions must survive
        # untouched. A one-group test cannot observe that at all.
        ("aurora", "mlp_out"),
    ],
)
def test_the_splice_reattaches_moments_to_the_SAME_NAMED_parameters(
    mode: str, optimizer: str, scope: str, tmp_path: Path,
) -> None:
    """⚑ Counting state entries is NOT enough, and this repo is exactly where that
    matters.

    `_remap_optimizer_state_for_new_params` splices by POSITION inside each param
    group and its own docstring warns the remap "relies on every unchanged
    parameter retaining the same relative order". If the ordering assumption ever
    broke, donor moments would land on the WRONG parameter -- accepted, non-empty,
    same COUNT, and silently wrong. That is the failure class this codebase is
    built around, so the assertion has to be per-NAME, not per-total.

    Method: give every parameter's every state tensor a distinguishable value,
    save, warm-start the adapter arm, then require each (name, state_key) to
    carry back the value that was banked under THAT pair.

    ⚑ The fingerprint is DETERMINISTIC ENUMERATION (`idx + 0.5`), not
    `abs(hash(name)) % 100_000`. Hash-derived values can collide, and a collision
    is exactly the case where a mis-splice would pass -- two parameters swapping
    moments is invisible if both carry the same number. Enumeration makes every
    value distinct by construction. `+ 0.5` keeps it non-integral so a zeroed or
    freshly-initialised slot can never accidentally equal one.

    ⚑ And the state key is NOT hardcoded to `exp_avg`. Under production's
    `aurora` + `mlp_out` the optimizer is an `AuroraWithAuxAdam` over FOUR param
    groups, and the Aurora group's state is `momentum_buffer`, not
    `exp_avg`/`exp_avg_sq`. An `exp_avg`-only assertion silently SKIPS all six
    matrix-group tensors -- it would pass while the group whose remap is hardest
    went entirely unchecked.
    """
    donor_cfg = _cfg()
    donor = Trainer(
        build_model(donor_cfg), device="cpu", lr=1e-3, optimizer=optimizer,
        matrix_optimizer_scope=scope,
        warmup_steps=10, warmup_lr_start=1e-5, use_amp=False,
        log_dir=tmp_path / "donor", tb_log_interval=1000, prefetch_batches=False,
        model_config=donor_cfg,
    )
    for param in donor.model.parameters():
        param.grad = torch.randn_like(param)
    donor.opt.step()

    # Stamp a unique, order-sensitive fingerprint into every state tensor.
    donor_names = [n for n, p in donor.model.named_parameters() if p.requires_grad]
    by_id = {id(p): n for n, p in donor.model.named_parameters()}
    banked: dict[tuple[str, str], float] = {}
    stamp = 0
    for group in donor.opt.param_groups:
        for param in group["params"]:
            state = donor.opt.state.get(param)
            if not state:
                continue
            name = by_id[id(param)]
            for key, value_t in state.items():
                if not torch.is_tensor(value_t) or not value_t.is_floating_point():
                    continue
                stamp += 1
                value = float(stamp) + 0.5
                state[key] = torch.full_like(value_t, value)
                banked[name, key] = value
    # Distinct by construction -- the property the hash version only assumed.
    assert len(set(banked.values())) == len(banked)
    # Every trainable parameter carries state, and the matrix group is present.
    assert {n for n, _ in banked} >= set(donor_names) - {"_embed_gate_mul_raw"}
    if optimizer == "aurora":
        assert any(k == "momentum_buffer" for _, k in banked), (
            "the Aurora matrix group contributed no state -- this arm is not "
            "exercising the multi-group remap it exists for"
        )
    ckpt = tmp_path / "donor.pt"
    donor.save(ckpt)

    arm_cfg = _cfg(policy_embedding_mode=mode)
    arm = Trainer(
        build_model(arm_cfg), device="cpu", lr=1e-3, optimizer=optimizer,
        matrix_optimizer_scope=scope,
        warmup_steps=10, warmup_lr_start=1e-5, use_amp=False,
        log_dir=tmp_path / "arm", tb_log_interval=1000, prefetch_batches=False,
        model_config=arm_cfg,
    )
    arm.load(ckpt)

    arm_by_id = {id(p): n for n, p in arm.model.named_parameters()}
    seen: dict[tuple[str, str], float] = {}
    for group in arm.opt.param_groups:
        for param in group["params"]:
            state = arm.opt.state.get(param)
            if not state:
                continue
            name = arm_by_id[id(param)]
            for key, value_t in state.items():
                if torch.is_tensor(value_t) and value_t.is_floating_point():
                    seen[name, key] = float(value_t.flatten()[0])

    # The adapter itself is NEW: it must get a fresh (zero) slot, not a donor one.
    for (name, key), value in seen.items():
        if name.startswith("policy_embedding"):
            assert value == 0.0, f"{name}.{key} inherited a donor moment {value}"
            continue
        assert (name, key) in banked, (
            f"{name}.{key} carries state that no donor parameter banked"
        )
        assert value == banked[name, key], (
            f"{name}.{key} got the moment banked for a DIFFERENT parameter: "
            f"{value} != {banked[name, key]}"
        )
    # ...and every donor parameter that banked a moment got it back.
    assert set(banked) <= set(seen), sorted(set(banked) - set(seen))


@pytest.mark.parametrize("mode", _ADAPTER_MODES)
def test_SWA_is_NOT_preserved_across_the_topology_change(
    mode: str, tmp_path: Path,
) -> None:
    """⚑ PINNING A KNOWN LIMITATION, not claiming it away.

    The splice covers the OPTIMIZER. SWA is restored through a separate STRICT
    `self._swa_model.load_state_dict(...)`, which the two new `policy_embedding.*`
    keys make incompatible -- so the averaged weights and `n_averaged` are
    REINITIALISED. An earlier version of this PR described the splice as removing
    "the warm-start reset", full stop; that was true of the optimizer and false of
    SWA, and review was right to reject the claim as unestablished.

    Left as-is deliberately: production runs `swa_start: -1`, so `export_swa` never
    publishes the averaged net and the reset carries no measurement. If SWA is ever
    turned on for an arm using the adapter, this test is the thing that has to
    change first.
    """
    donor_cfg = _cfg()
    donor = Trainer(
        build_model(donor_cfg), device="cpu", lr=1e-3, optimizer="adamw",
        warmup_steps=10, warmup_lr_start=1e-5, use_amp=False,
        log_dir=tmp_path / "sd", tb_log_interval=1000, prefetch_batches=False,
        model_config=donor_cfg,
    )
    if getattr(donor, "_swa_model", None) is None:
        pytest.skip("SWA is not constructed in this configuration")
    donor.save(tmp_path / "sd.pt")

    arm_cfg = _cfg(policy_embedding_mode=mode)
    arm = Trainer(
        build_model(arm_cfg), device="cpu", lr=1e-3, optimizer="adamw",
        warmup_steps=10, warmup_lr_start=1e-5, use_amp=False,
        log_dir=tmp_path / "sa", tb_log_interval=1000, prefetch_batches=False,
        model_config=arm_cfg,
    )
    arm.load(tmp_path / "sd.pt")
    swa = getattr(arm, "_swa_model", None)
    assert swa is not None
    assert int(swa.n_averaged.item()) == 0, (
        "SWA now survives the topology change -- update this test AND the PR's "
        "claim, which currently says it does not"
    )


@pytest.mark.parametrize("mode", _ADAPTER_MODES)
@pytest.mark.parametrize(
    "aot_key", ["distributed_inference_aot_dir", "distributed_worker_aot_dir"],
)
def test_EVERY_AOT_route_REFUSES_to_serve_a_prior_without_the_adapter(
    aot_key: str, mode: str,
) -> None:
    """⚑⚑ THERE ARE TWO AOT ROUTES, and the first fix only covered one.

    `distributed_inference_aot_dir` reaches the BROKER; `distributed_worker_aot_dir`
    becomes `worker --aot-dir` and builds the same `AOTEvaluator` inside the worker.
    Both replace `ChessNet.forward` with a graph frozen at package-build time and
    rebind only the constants the PACKAGE asks for -- so a package built before the
    adapter existed simply never receives `policy_embedding.*`, and nothing raises,
    because `build_aot_constants` checks only the package->model direction.

    Parametrized over the key AND the mode so neither a new route nor a new mode
    can be added without a home here.
    """
    from chess_anti_engine.tune.distributed_runtime import (
        assert_no_aot_route_bypasses_the_policy_adapter,
    )

    with pytest.raises(ValueError, match=aot_key):
        assert_no_aot_route_bypasses_the_policy_adapter(
            {"policy_embedding_mode": mode, aot_key: "data/aot_models_512"},
        )


@pytest.mark.parametrize(
    ("launcher", "aot_key"),
    [
        ("_launch_inference_broker", "distributed_inference_aot_dir"),
        ("_launch_distributed_worker", "distributed_worker_aot_dir"),
    ],
)
def test_BOTH_LAUNCHERS_call_the_refusal(
    launcher: str, aot_key: str, tmp_path: Path,
) -> None:
    """⚑ Drives the real entry points, not the helper.

    The previous version of this test called
    `assert_no_aot_route_bypasses_the_policy_adapter` directly -- and a mutant that
    deleted the CALL from `_launch_distributed_worker` while leaving the function
    defined SURVIVED it. Testing the helper instead of the wiring is the trap this
    repo punishes; both launchers get driven here.
    """
    import chess_anti_engine.tune.distributed_runtime as dr

    fn = getattr(dr, launcher)
    kwargs: dict[str, object] = {
        "config": {
            "policy_embedding_mode": "residual_mish",
            aot_key: "data/aot_models_512",
            "distributed_server_root": str(tmp_path / "srv"),
        },
        "trial_id": "t",
        "trial_dir": tmp_path / "trial",
    }
    if launcher == "_launch_inference_broker":
        kwargs["publish_dir"] = tmp_path / "pub"
    else:
        kwargs["worker_index"] = 0

    with pytest.raises(ValueError, match=aot_key):
        fn(**kwargs)


def test_the_AOT_refusal_names_a_remedy_that_actually_works() -> None:
    """⚑ The message must not promise an impossible fix.

    The guard's condition is `adapter on AND an AOT dir configured` -- it does NOT
    inspect package architecture, so a correctly REBUILT package is refused too.
    An earlier message told the operator to "rebuild the packages", which this
    guard would then reject anyway. Saying so explicitly is the difference between
    a guard an operator can satisfy and one that reads as a bug.
    """
    from chess_anti_engine.tune.distributed_runtime import (
        assert_no_aot_route_bypasses_the_policy_adapter,
    )

    with pytest.raises(ValueError, match="policy_embedding_mode is not off") as excinfo:
        assert_no_aot_route_bypasses_the_policy_adapter(
            {"policy_embedding_mode": "linear",
             "distributed_inference_aot_dir": "data/aot_models_512"},
        )
    message = str(excinfo.value)
    assert "Clear the AOT" in message
    assert "rebuilding them will not satisfy it" in message


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
                "policy_embedding_mode": True,
                "distributed_inference_aot_dir": "data/aot_models_512",
                "distributed_server_root": str(tmp_path / "srv"),
            },
            trial_id="t",
            publish_dir=tmp_path / "pub",
            trial_dir=tmp_path / "trial",
        )


def _terminate_and_reap(proc: subprocess.Popen[bytes] | None) -> None:
    """Stop a broker by HANDLE and block until the kernel has reaped it.

    ⚑ By handle, never by name pattern. `pkill -f` / `pgrep -f` self-match the
    caller's own cmdline and have caused a 4h46m production outage in this repo
    (2026-08-10), so a "kill any stray brokers" sweep is not an option here even
    in a test.
    """
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


@pytest.mark.parametrize(
    ("mode", "aot_dir"),
    [("off", "data/aot_models_512"), ("residual_mish", ""), ("linear", "")],
)
def test_the_AOT_refusal_does_not_fire_on_either_safe_combination(
    mode: str, aot_dir: str, tmp_path: Path,
) -> None:
    """Negative control. A guard that refuses every launch is not a guard --
    production runs AOT today with the adapter off, and the adapter is safe when
    AOT is not configured. Only the CONJUNCTION is the hazard.

    Asserted by the absence of the AOT ValueError specifically.

    ⚑ The docstring used to continue "the call fails later for unrelated reasons
    (no real server root), which is fine." THAT IS FALSE, and believing it is
    what leaked the brokers: on both safe combinations the launch SUCCEEDS and
    returns a live `subprocess.Popen`. Nothing downstream fails, so nothing tore
    it down. The two safe rows are therefore the two rows that MUST clean up.
    """
    from chess_anti_engine.tune.distributed_runtime import _launch_inference_broker

    raised = ""
    proc = None
    try:
        proc = _launch_inference_broker(
            config={
                "policy_embedding_mode": mode,
                "distributed_inference_aot_dir": aot_dir,
                "distributed_server_root": str(tmp_path / "srv"),
            },
            trial_id="t",
            publish_dir=tmp_path / "pub",
            trial_dir=tmp_path / "trial",
        )
    except (NameError, AttributeError, TypeError):
        # ⚑ RE-RAISE programming errors. An earlier revision of this test had a
        # NameError in the config dict, and the broad `except` swallowed it -- the
        # assert then ran against an empty string and PASSED. A negative control
        # that cannot fail is worse than no control; only environment failures
        # (no real server root) may be tolerated here.
        raise
    except Exception as exc:  # only the AOT message is under test
        raised = f"{type(exc).__name__}: {exc}"
    finally:
        # ⚑⚑ THE SAFE COMBINATIONS ACTUALLY LAUNCH A BROKER, AND AN EARLIER
        # REVISION OF THIS TEST LEAKED IT. `_launch_inference_broker` returns a
        # live `subprocess.Popen` — that is the whole point of the negative
        # control, since the guard is being asserted NOT to fire. The result was
        # 4 orphaned `python -m chess_anti_engine.inference` processes per test
        # run (measured: 52 alive, oldest ~3h, after a handful of local runs),
        # each holding a model and surviving the session that spawned it.
        #
        # The docstring used to claim "the call fails later for unrelated
        # reasons (no real server root)". It does NOT fail — it succeeds and the
        # broker idles forever. Terminate by the HANDLE we were given; never by
        # name pattern (`pkill -f` self-matches and has caused a 4h46m outage
        # here).
        _terminate_and_reap(proc)
    # ⚑⚑ THIS ASSERTION IS WHAT MAKES THE CLEANUP ABOVE MUTATION-VISIBLE, AND IT
    # HAS TO LIVE IN *THIS* TEST. The first version of this fix pinned the
    # teardown with a separate standalone test that launched its own broker and
    # reaped it -- which proves a property of `_launch_inference_broker`, not
    # that THIS test cleans up. Deleting the `finally:` above left that
    # standalone test perfectly green while this test leaked again: a regression
    # test for a leak, that cannot observe the leak.
    #
    # Mutation-verified 2026-08-12: with the `finally:` removed, this line fails
    # on both safe rows.
    assert proc is None or proc.poll() is not None, (
        f"mode={mode!r} aot_dir={aot_dir!r}: the negative control launched a "
        f"broker (pid {proc.pid if proc else None}) and did not reap it"
    )
    assert "distributed_inference_aot_dir" not in raised, raised


def test_the_launcher_returns_a_handle_that_can_actually_be_reaped(
    tmp_path: Path,
) -> None:
    """Pins the CONTRACT the teardown depends on: `_launch_inference_broker`
    hands back a real `subprocess.Popen` whose child dies on `terminate()`.

    ⚑ This is NOT the regression test for the leak, and an earlier revision of
    this PR wrongly presented it as one. It launches its own broker and reaps
    it, so it stays green no matter what
    `test_the_AOT_refusal_does_not_fire_on_either_safe_combination` does -- the
    assertion that pins THAT test's cleanup has to live inside THAT test, and
    now does. What this one buys is the other half: if the launcher ever starts
    returning None, or wrapping the broker in a shell so the handle addresses a
    parent that dies without its child, the teardown would silently stop working
    and this test is what notices.
    """
    from chess_anti_engine.tune.distributed_runtime import _launch_inference_broker

    proc = _launch_inference_broker(
        config={
            "policy_embedding_mode": "off",
            "distributed_inference_aot_dir": "",
            "distributed_server_root": str(tmp_path / "srv"),
        },
        trial_id="t",
        publish_dir=tmp_path / "pub",
        trial_dir=tmp_path / "trial",
    )
    assert proc is not None, "launcher returned no handle -- teardown is impossible"
    _terminate_and_reap(proc)
    assert proc.poll() is not None, "broker survived terminate() -- it would leak"


def test_arch_schema_version_was_bumped_for_these_fields() -> None:
    """Defaulting either field builds a different architecture, which is exactly
    what the constant documents itself for."""
    assert ARCH_SCHEMA_VERSION == 19
