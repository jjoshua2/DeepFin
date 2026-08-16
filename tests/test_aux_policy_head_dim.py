"""`aux_policy_head_dim` — narrowing the AUXILIARY policy heads' projection.

`AttentionPolicyHead` has always accepted a `policy_dim`, and no caller ever
passed it: all four heads projected at full trunk width. This key lets
`policy_soft` and `policy_sf` — the two auxiliary SF-teacher heads — use a
narrower Q/K projection. `None` (the default, and what `configs/pbt2_small.yaml`
leaves it at) means trunk width, i.e. today's exact behaviour, bit-identical.

Two claims these tests exist to defend:

* **The aux-only restriction is a SAFETY property, not a preference.**
  `scripts/build_aot_packages.py` sets `model._inference_only = True`, and the
  `_inference_only` branch of `ChessNet.forward` traces only `policy_own` + `wdl`
  — so `policy_own`'s weights ARE in the exported constants and `policy_soft` /
  `policy_sf` are not. Narrowing `policy_own` would silently re-shape the tensors
  every AOT package runs search on. `AUX_POLICY_HEAD_DIM_HEADS` is pinned below
  so widening the scope cannot happen as a one-word edit at a construction site.

* **The width reaches production AND the workers.** ⚑ The maximally silent
  failure is the manifest: omit it from `model_config_to_manifest_dict` and the
  trainer builds narrow aux heads while every worker rebuilds them at trunk
  width, and `load_state_dict_tolerant` drops the mismatched tensors with only a
  `print()`. ⚑ It does NOT make selfplay serve a random `policy_soft` — that
  cannot happen. No search evaluator READS `policy_soft` / `policy_sf`; they take
  `policy_own` (or `policy`) and `wdl` and nothing else. Selfplay does not even
  COMPUTE them: `worker.py:1700`, `inference.py:1987` and `:3677` set
  `_inference_only = True`, and that branch of `ChessNet.forward`, like
  `forward_legal_policy` / `forward_legal_policy_rows`, emits only `policy_own` +
  `wdl`. What the omission really costs is (a) a served state_dict that is no
  longer the published one, dropped with a bare `print()`, and (b) `model_config_identity_key`
  — computed off the ModelConfig rebuilt FROM the manifest — hashing two
  genuinely different architectures equal, so the shared broker's "different
  model config, skipping" guard cannot fail on this field.

* **A width migration must survive the OPTIMIZER, not just the loader.** The
  shapes change while the parameter COUNT does not, so `Trainer.load`'s
  count-based reinit fallback never fires and `Optimizer.load_state_dict`
  restores donor moments by index without validating shape. `opt.step()` is the
  only thing that notices, and it is outside every `except`.
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
    model_config_from_flat_config,
    model_config_from_manifest_dict,
    model_config_to_manifest_dict,
    reset_mismatched_optimizer_state,
    resume_model_config_from_arch,
)
from chess_anti_engine.model.transformer import (
    AUX_POLICY_HEAD_DIM_HEADS,
    AttentionPolicyHead,
    ChessNet,
)
from chess_anti_engine.train.aurora import AuroraWithAuxAdam
from chess_anti_engine.train.trainer import Trainer
from chess_anti_engine.tune.trainable import _build_trial_model_config
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils.architecture import normalize_aux_policy_head_dim
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

_PLANES = 175
_TRUNK = 64
_NARROW = 16

# The four heads, and whether this key is allowed to narrow each one.
_HEADS_AND_WHETHER_NARROWED = (
    ("policy_own", False),
    ("policy_soft", True),
    ("policy_sf", True),
    ("policy_future", False),
)


def _cfg(aux_policy_head_dim: int | None = None, **kwargs: object) -> ModelConfig:
    return dataclasses.replace(
        ModelConfig(
            kind="transformer",
            embed_dim=_TRUNK,
            num_layers=2,
            num_heads=4,
            use_smolgen=False,
        ),
        aux_policy_head_dim=aux_policy_head_dim,
        **kwargs,
    )


def _build(aux_policy_head_dim: int | None = None, **kwargs: object) -> ChessNet:
    model = build_model(_cfg(aux_policy_head_dim, **kwargs))
    assert isinstance(model, ChessNet)
    return model


def _planes(batch: int = 2) -> torch.Tensor:
    return torch.zeros(batch, _PLANES, 8, 8)


def _head(model: ChessNet, name: str) -> AttentionPolicyHead:
    head = getattr(model, name)
    assert isinstance(head, AttentionPolicyHead), f"{name} was not built"
    return head


# --------------------------------------------------------------------------
# the safety scope: policy_own is the search prior and is NEVER narrowed
# --------------------------------------------------------------------------


def test_the_scope_tuple_names_only_the_auxiliary_heads() -> None:
    """⚑⚑ THE SAFETY PIN. `policy_own` is the search prior and the ONLY policy
    head an AOT package traces (`scripts/build_aot_packages.py` sets
    `_inference_only`, whose forward branch emits `policy_own` + `wdl`), so its
    weights are baked into `get_constant_fqns()` while the aux heads' are not.
    Adding it here would turn a training-side experiment into a silent re-shape
    of the tensors production search runs on.
    """
    assert AUX_POLICY_HEAD_DIM_HEADS == ("policy_soft", "policy_sf")
    assert "policy_own" not in AUX_POLICY_HEAD_DIM_HEADS
    assert "policy_future" not in AUX_POLICY_HEAD_DIM_HEADS


@pytest.mark.parametrize(("head", "narrowed"), _HEADS_AND_WHETHER_NARROWED)
def test_only_the_auxiliary_heads_narrow(head: str, narrowed: bool) -> None:
    """The scope tuple is a comment unless the construction sites obey it."""
    model = _build(_NARROW)
    want = _NARROW if narrowed else _TRUNK
    for proj in ("q", "k"):
        weight = getattr(_head(model, head), proj).weight
        assert tuple(weight.shape) == (want, _TRUNK), f"{head}.{proj}"
    # `underpromo` reads the trunk token directly and is not part of the Q/K
    # projection, so it stays trunk-wide on every head.
    assert tuple(_head(model, head).underpromo.weight.shape) == (9, _TRUNK)


def test_narrowing_the_aux_heads_actually_removes_parameters() -> None:
    """A knob that changes no parameter count is a knob that did not take."""
    assert sum(p.numel() for p in _build(_NARROW).parameters()) < sum(
        p.numel() for p in _build(None).parameters()
    )


# --------------------------------------------------------------------------
# the default is today's behaviour, bit-identical
# --------------------------------------------------------------------------


def test_default_none_is_bit_identical_to_the_trunk_width_model() -> None:
    """`None` must mean "trunk width", not "some default width". Same seed, same
    tensors, key for key — this is what keeps `configs/pbt2_small.yaml` (which
    does NOT set the key) at 63,084,128 params.
    """
    torch.manual_seed(0)
    default = _build(None).state_dict()
    torch.manual_seed(0)
    explicit = _build(_TRUNK).state_dict()
    assert default.keys() == explicit.keys()
    for key, value in default.items():
        assert torch.equal(value, explicit[key]), key


def test_the_narrowed_head_still_emits_the_compact_policy() -> None:
    """The projection width is internal to Q/K; the output contract is 1858
    compact logits regardless, and every head must still produce them."""
    outputs = _build(_NARROW)(_planes())
    for name in ("policy_own", "policy_soft", "policy_sf", "policy_future"):
        assert outputs[name].shape == (2, 1858), name


# --------------------------------------------------------------------------
# plumbing: the width must reach production and survive every hop
# --------------------------------------------------------------------------


_WIDTHS: tuple[int | None, ...] = (None, _NARROW)


@pytest.mark.parametrize("want", _WIDTHS)
def test_width_survives_the_manifest_round_trip(want: int | None) -> None:
    """⚑⚑ THE ONE THAT MATTERS MOST — `model_config_to_manifest_dict` (write) and
    `model_config_from_manifest_dict` (read) are how the TRAINER tells every
    WORKER what architecture to rebuild.

    Asserted in both directions and read back through the reader, because
    checking only the write side lets a reader hardcoded to `None` pass, and the
    reader is the half that runs on the worker. The built SHAPE is asserted too:
    the round-trip carrying the number is worth nothing if the number does not
    reach `nn.Linear`, and a shape mismatch here is the exact silent failure —
    `load_state_dict_tolerant` drops `policy_soft.q/k` with only a `print()`, so
    the worker's state_dict stops being the published one and the broker's
    architecture-identity guard stops being able to tell the two apart. (It does
    NOT corrupt the served policy: see the module docstring — nothing on a
    serving path evaluates `policy_soft`.)
    """
    manifest = model_config_to_manifest_dict(_cfg(want))
    assert manifest["aux_policy_head_dim"] == want
    restored = model_config_from_manifest_dict(manifest)
    assert restored.aux_policy_head_dim == want

    expect = _TRUNK if want is None else want
    rebuilt = build_model(restored)
    assert isinstance(rebuilt, ChessNet)
    assert tuple(rebuilt.policy_soft.q.weight.shape) == (expect, _TRUNK)
    assert _head(rebuilt, "policy_sf").q.weight.shape[0] == expect
    # ...and the untouched heads are untouched on the far side of the wire too.
    assert tuple(rebuilt.policy_own.q.weight.shape) == (_TRUNK, _TRUNK)
    assert tuple(rebuilt.policy_future.q.weight.shape) == (_TRUNK, _TRUNK)


def test_EVERY_model_config_field_reaches_the_manifest() -> None:
    """⚑⚑ The generalisation of the test above, and the reason it is not enough.

    `model_config_to_manifest_dict` is HAND-ENUMERATED. The test above pins one
    key, and this module's docstring calls a manifest omission "the maximally
    silent failure" — but a per-key test only ever protects the key someone
    remembered to write it for. Measured by independent review of PR #439:
    deleting `"input_square_embedding"` (a real architecture field) from the
    writer passes 15 test files, EXIT 0, and 30 of the 44 `ModelConfig` fields
    are asserted against the manifest nowhere at all.

    ⚑ The neighbouring guard already learned this. `model_config_identity_key`'s
    docstring records that its hand-written tuple "listed 35 of 41 fields and
    had silently drifted", omitting five SHAPE-CHANGING ones, and it was
    rewritten to derive from `dataclasses.fields` precisely so that a newly
    added field defaults to BEING carried. This asserts the same property for
    the manifest, which is the wire format the identity key is computed off.

    Deriving it means the next `ModelConfig` field fails HERE — one line naming
    the field — instead of shipping a worker that silently rebuilds the wrong
    architecture. The single rename is spelled out rather than tolerated by a
    fuzzy match, so a second rename also has to be declared.
    """
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    # The one deliberate rename between the dataclass and the wire format.
    expected = (fields - {"use_gradient_checkpointing"}) | {"gradient_checkpointing"}
    manifest = set(model_config_to_manifest_dict(_cfg(None)))

    assert not (expected - manifest), (
        "ModelConfig field(s) missing from the manifest: "
        f"{sorted(expected - manifest)}. A field absent here is not sent to the "
        "workers, so they rebuild a DIFFERENT architecture and "
        "load_state_dict_tolerant drops the mismatched tensors with only a "
        "print(); model_config_identity_key is computed off the config rebuilt "
        "from this manifest, so the broker's guard cannot tell them apart "
        "either. Add the field to model_config_to_manifest_dict."
    )
    assert not (manifest - expected), (
        f"manifest carries key(s) with no ModelConfig field: "
        f"{sorted(manifest - expected)}. Either add the field or declare the "
        "rename in this test."
    )


@pytest.mark.parametrize("want", _WIDTHS)
def test_width_reaches_the_model_config_production_builds(want: int | None) -> None:
    """⚑ yaml -> TrialConfig -> ModelConfig. `model_config_from_flat_config` is
    NOT this path -- it is only reachable from `scripts/`, so a key that stops at
    the flat dict is dead in training while looking wired (measured on
    `categorical_head_coupled`, PR #397)."""
    flat = flatten_run_config_defaults({"model": {"aux_policy_head_dim": want}, "train": {}})
    built = _build_trial_model_config(TrialConfig.from_dict(flat))
    assert built.aux_policy_head_dim == want


@pytest.mark.parametrize("want", _WIDTHS)
def test_width_reaches_the_flat_config_script_path(want: int | None) -> None:
    """`model_config_from_flat_config` is the offline diagnostic / bootstrap
    path. Not production, but a script that rebuilds the net at the wrong aux
    width scores a different architecture than the one that trained."""
    assert model_config_from_flat_config({"aux_policy_head_dim": want}).aux_policy_head_dim == want


def test_the_key_is_accepted_by_the_yaml_schema() -> None:
    """Category-(a): a live-yaml key absent from the schema makes
    `flatten_run_config_defaults` raise, and it runs before the argument parser
    and outside any try -- the process would not boot."""
    assert "aux_policy_head_dim" in flatten_run_config_defaults(
        {"model": {"aux_policy_head_dim": _NARROW}, "train": {}}
    )
    with pytest.raises(ValueError, match="aux_policy_head_dim_typo"):
        flatten_run_config_defaults({"model": {"aux_policy_head_dim_typo": 1}, "train": {}})


@pytest.mark.parametrize(
    ("checkpoint_value", "config_value"),
    [(None, _NARROW), (_NARROW, None), (_NARROW, 32)],
)
def test_the_config_width_wins_over_the_checkpoints_on_resume(
    checkpoint_value: int | None, config_value: int | None,
) -> None:
    """⚑ Resume takes topology from the checkpoint's `arch`, so a key outside
    `_RESUME_CONFIG_OWNED_ENCODING_KEYS` reverts to the donor on EVERY resume --
    silently, because the tolerant loader is happy either way. Without this the
    yaml edit would look applied (it is in `config`) while the trial kept
    building the donor's width for the whole readout window.

    ⚑ This is a DICT-LEVEL check only. It says the rebuilt config carries the
    new width; it says nothing about whether the resulting resume RUNS -- see
    `test_width_migration_survives_a_real_trainer_resume`, which is the test
    that can actually fail on the optimizer.
    """
    donor = _cfg(checkpoint_value)
    arch = dataclasses.asdict(donor)
    arch["_schema_version"] = ARCH_SCHEMA_VERSION
    run_cfg = dataclasses.replace(donor, aux_policy_head_dim=config_value)
    assert resume_model_config_from_arch(arch, run_cfg).aux_policy_head_dim == config_value


def _trainer(cfg: ModelConfig, log_dir: Path) -> Trainer:
    model = build_model(cfg)
    return Trainer(
        model,
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        matrix_optimizer_scope="mlp_out",
        use_amp=False,
        warmup_steps=0,
        log_dir=log_dir,
        tb_log_interval=1000,
        prefetch_batches=False,
        model_config=cfg,
    )


def _one_optimizer_step(trainer: Trainer) -> None:
    """A real backward + `opt.step()`. The shape crash this guards lives HERE —
    `Optimizer.load_state_dict` restores moments by index and never checks."""
    trainer.opt.zero_grad(set_to_none=True)
    out = trainer.model(_planes())
    torch.stack([v.float().sum() for v in out.values()]).sum().backward()
    trainer.opt.step()


@pytest.mark.parametrize(
    ("checkpoint_value", "config_value"),
    [(None, _NARROW), (_NARROW, None), (_NARROW, 32)],
)
def test_width_migration_survives_a_real_trainer_resume(
    tmp_path: Path, checkpoint_value: int | None, config_value: int | None,
) -> None:
    """⚑⚑ THE DECIDING OBSERVATION for this key, and the one the dict-level
    resume test structurally cannot make.

    The migration rebuilds `policy_soft`/`policy_sf` `q`/`k` at the config's
    width against a donor at the other width. The parameter COUNT is unchanged,
    so `Trainer.load`'s `n_ckpt_params == n_model_params` branch skips the
    remap, `opt.load_state_dict` SUCCEEDS (torch restores moments by param index
    with no shape validation), `optimizer_state_loaded` stays True, and the
    `except (ValueError, KeyError, RuntimeError)` never fires. The failure lands
    at the first `opt.step()`, OUTSIDE that try — which is why this test steps
    rather than inspecting state.

    Two steps, not one: the first proves the restored moment is usable, the
    second proves whatever the first wrote back is still the right shape.
    """
    donor_cfg = _cfg(checkpoint_value)
    donor = _trainer(donor_cfg, tmp_path / "donor")
    # The donor must have MOMENTS to restore — an unstepped optimizer has an
    # empty `state` and every shape mismatch below would be vacuous.
    _one_optimizer_step(donor)
    assert donor.opt.state, "donor optimizer carries no state to migrate"
    ckpt = tmp_path / "donor.pt"
    donor.save(ckpt)

    arch = dataclasses.asdict(donor_cfg)
    arch["_schema_version"] = ARCH_SCHEMA_VERSION
    resumed_cfg = resume_model_config_from_arch(
        arch, dataclasses.replace(donor_cfg, aux_policy_head_dim=config_value)
    )
    resumed = _trainer(resumed_cfg, tmp_path / "resumed")
    resumed.load(ckpt)

    # The migration actually happened: the rebuilt heads are at the NEW width.
    expect = _TRUNK if config_value is None else config_value
    resumed_model = resumed.model
    assert isinstance(resumed_model, ChessNet)
    assert tuple(_head(resumed_model, "policy_soft").q.weight.shape) == (expect, _TRUNK)

    _one_optimizer_step(resumed)
    _one_optimizer_step(resumed)

    # ...and every surviving moment now matches the parameter it sits under.
    for group in resumed.opt.param_groups:
        for param in group["params"]:
            for key, value in resumed.opt.state.get(param, {}).items():
                if torch.is_tensor(value) and value.dim() == param.dim():
                    assert tuple(value.shape) == tuple(param.shape), key


def test_the_migration_reports_what_optimizer_state_it_dropped(tmp_path: Path) -> None:
    """Silence is how this class of bug survives: the drop must be nameable."""
    donor_cfg = _cfg(_NARROW)
    donor = _trainer(donor_cfg, tmp_path / "donor")
    _one_optimizer_step(donor)

    resumed = _trainer(_cfg(None), tmp_path / "resumed")
    resumed.opt.load_state_dict(donor.opt.state_dict())
    dropped = reset_mismatched_optimizer_state(
        resumed.opt,
        param_names={id(p): n for n, p in resumed.model.named_parameters()},
    )
    assert dropped, "the width change dropped nothing — the sweep did not fire"
    assert any("policy_soft.q.weight" in line for line in dropped), dropped
    # `policy_own` is outside AUX_POLICY_HEAD_DIM_HEADS and must be untouched.
    assert not any("policy_own" in line for line in dropped), dropped


def test_resume_still_takes_real_topology_from_the_checkpoint() -> None:
    """Negative control: widening the config-owned list must not let a genuine
    SHAPE key escape the checkpoint."""
    donor = dataclasses.replace(_cfg(_NARROW), num_layers=1)
    arch = dataclasses.asdict(donor)
    arch["_schema_version"] = ARCH_SCHEMA_VERSION
    run_cfg = dataclasses.replace(donor, num_layers=2)
    assert resume_model_config_from_arch(arch, run_cfg).num_layers == 1


def test_the_key_is_REFUSED_by_a_live_yaml_reload() -> None:
    """⚑⚑ HOW YOU SET THIS, AND HOW YOU CANNOT. Every `ModelConfig` field is
    construction-bound, so editing the live yaml and restarting with `--resume`
    logs a WARNING, comes up healthy, and trains the OLD widths for the whole
    readout window. ⇒ needs a FRESH (non-resumed) trial, or a salvage warm-start
    whose launch config already carries the key."""
    from chess_anti_engine.tune.trainable_config_ops import _RESUME_CONSTRUCTION_BOUND_KEYS

    assert "aux_policy_head_dim" in _RESUME_CONSTRUCTION_BOUND_KEYS


def test_arch_schema_version_was_bumped_for_this_field() -> None:
    """⚑ Defaulting this field rebuilds `policy_soft` / `policy_sf` at trunk
    width, so their q/k tensors stop matching the checkpoint's and the tolerant
    loader drops them — exactly what the constant documents itself for. The
    EXACT pin lives with the newest field, so the next field to be added has to
    come here and bump it.
    """
    assert ARCH_SCHEMA_VERSION == 20


# --------------------------------------------------------------------------
# validation: a bad value must not be accepted and then silently ignored
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "none", "null"])
def test_absent_and_null_spellings_all_mean_trunk_width(value: object) -> None:
    assert normalize_aux_policy_head_dim(value) is None


@pytest.mark.parametrize("value", [False, "off", "false", "Off", "FALSE", " off "])
def test_the_off_spellings_end_the_experiment_instead_of_killing_the_trial(
    value: object,
) -> None:
    """⚑ HOW AN OPERATOR TURNS THIS OFF. `aux_policy_head_dim: off` in the live
    yaml parses as the BOOLEAN `False`, and a bare `false` likewise. Rejecting
    those is not a safe default: the key IS read by `TrialConfig.from_dict`, so
    the rejection is a category-(b) failure that raises inside `train_trial`'s
    iteration-loop `try:` (which has a `finally:` and zero `except`) and KILLS
    the trial mid-iteration. The sibling normalizers
    (`normalize_ffn_mult_by_layer`, `normalize_embed_dim_by_layer`) already
    spell "off" this way; matching them makes the revert edit inert instead of
    lethal.

    ⚑ `True` deliberately still raises — "on" is not a width.
    """
    assert normalize_aux_policy_head_dim(value) is None
    assert TrialConfig.from_dict(
        flatten_run_config_defaults({"model": {"aux_policy_head_dim": value}, "train": {}})
    ).aux_policy_head_dim is None


@pytest.mark.parametrize("value", [128, "128", 128.0])
def test_a_width_is_coerced_to_int(value: object) -> None:
    assert normalize_aux_policy_head_dim(value) == 128


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "wide", [64]])
def test_a_bad_width_raises_rather_than_reaching_nn_linear(value: object) -> None:
    """Category-(c) is this repo's signature defect: accepted, never validated,
    silently wrong. `aux_policy_head_dim: 0` must not build a zero-width head."""
    with pytest.raises(ValueError, match="aux_policy_head_dim"):
        normalize_aux_policy_head_dim(value)


def test_a_bad_width_raises_at_model_build() -> None:
    """The validator has to be ON the build path, not merely importable."""
    with pytest.raises(ValueError, match="aux_policy_head_dim"):
        build_model(_cfg(0))


# --------------------------------------------------------------------------
# composition with the other policy-head keys
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["linear", "residual_mish"])
def test_it_composes_with_the_shared_policy_embedding(mode: str) -> None:
    """The shared embedding is a trunk-wide adapter applied BEFORE the heads, so
    it is orthogonal to the heads' Q/K width — but "orthogonal" is a hypothesis
    until built."""
    model = _build(_NARROW, policy_embedding_mode=mode)
    assert tuple(model.policy_soft.q.weight.shape) == (_NARROW, _TRUNK)
    assert tuple(model.policy_own.q.weight.shape) == (_TRUNK, _TRUNK)
    assert model(_planes())["policy_soft"].shape == (2, 1858)


def test_it_composes_with_the_policy_sf_head_being_off() -> None:
    """`policy_sf` is one of the two heads this key narrows and is also
    switchable off; the width must not resurrect it."""
    model = _build(_NARROW, enable_policy_sf_head=False)
    assert model.policy_sf is None
    assert tuple(model.policy_soft.q.weight.shape) == (_NARROW, _TRUNK)


def test_the_built_width_is_recorded_not_the_raw_config_value() -> None:
    """`ChessNet.aux_policy_head_dim` is a record of what was BUILT, like its
    siblings (`enable_policy_sf_head`, `policy_embedding_mode`). Storing
    `cfg.aux_policy_head_dim` raw let it read `"16"` / `16.0` / `None` while the
    heads were `16` — an attribute that disagrees with the tensors under it."""
    assert _build(None).aux_policy_head_dim == _TRUNK
    for raw in (_NARROW, str(_NARROW), float(_NARROW)):
        model = _build(raw)  # pyright: ignore[reportArgumentType]
        assert model.aux_policy_head_dim == _NARROW
        assert isinstance(model.aux_policy_head_dim, int)
        assert tuple(model.policy_soft.q.weight.shape) == (_NARROW, _TRUNK)


# --------------------------------------------------------------------------
# the optimizer-state sweep the migration depends on, at BOTH state shapes
# --------------------------------------------------------------------------


def _aurora_pair(
    matrix_shape: tuple[int, int], aux_shape: tuple[int, int],
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, AuroraWithAuxAdam]:
    """An `AuroraWithAuxAdam` over one Aurora-group and one AdamW-group tensor.

    Production's optimizer keeps TWO state shapes, and a sweep that handles only
    one of them is half a fix: the Aurora matrix group stores `momentum_buffer`
    and NO step, the AdamW aux groups store `exp_avg`/`exp_avg_sq`/`step` — and
    that `step` is a python int, not a tensor.
    """
    matrix = torch.nn.Parameter(torch.randn(*matrix_shape))
    aux = torch.nn.Parameter(torch.randn(*aux_shape))
    opt = AuroraWithAuxAdam(
        [
            {"params": [matrix], "use_aurora": True, "lr": 1e-3, "weight_decay": 0.0},
            {"params": [aux], "use_aurora": False, "lr": 1e-3, "weight_decay": 0.0},
        ]
    )
    for param in (matrix, aux):
        param.grad = torch.randn_like(param)
    opt.step()
    return matrix, aux, opt


def test_the_sweep_covers_both_of_the_production_optimizers_state_shapes() -> None:
    """Pins the premise the sweep is written against — if Aurora ever grows a
    `step`, or the aux group loses one, this test says so before the sweep does
    the wrong thing quietly."""
    matrix, aux, opt = _aurora_pair((8, 8), (8, 8))
    assert set(opt.state[matrix]) == {"momentum_buffer"}
    assert set(opt.state[aux]) == {"exp_avg", "exp_avg_sq", "step"}
    assert isinstance(opt.state[aux]["step"], int)


def test_the_sweep_clears_stale_state_in_both_groups_and_leaves_step_usable() -> None:
    _, _, donor = _aurora_pair((8, 8), (8, 8))
    matrix, aux, opt = _aurora_pair((4, 8), (4, 8))
    opt.load_state_dict(donor.state_dict())  # torch accepts this silently
    assert tuple(opt.state[matrix]["momentum_buffer"].shape) == (8, 8)

    dropped = reset_mismatched_optimizer_state(
        opt, param_names={id(matrix): "matrix", id(aux): "aux"},
    )
    assert sorted(line.split()[0] for line in dropped) == ["aux", "matrix"]
    assert not opt.state[matrix]
    assert not opt.state[aux]

    for param in (matrix, aux):
        param.grad = torch.randn_like(param)
    opt.step()
    opt.step()  # the second proves what the first wrote back is right-shaped


def test_the_sweep_leaves_matching_state_alone() -> None:
    """The negative control. A sweep that fires on a clean resume would throw
    away every moment in the run — silently, and it would still 'pass'."""
    _, _, donor = _aurora_pair((8, 8), (8, 8))
    matrix, aux, opt = _aurora_pair((8, 8), (8, 8))
    opt.load_state_dict(donor.state_dict())
    before = opt.state[matrix]["momentum_buffer"].clone()

    assert reset_mismatched_optimizer_state(opt) == []
    assert torch.equal(opt.state[matrix]["momentum_buffer"], before)
    assert opt.state[aux]["step"] == 1


def test_the_sweep_ignores_state_of_a_different_rank_than_its_parameter() -> None:
    """Rank-0 step counters and Adafactor-style factored row/column statistics
    are param-state that legitimately does NOT have the parameter's shape.
    Treating them as stale would reset a healthy optimizer on every resume."""
    _, aux, opt = _aurora_pair((8, 8), (8, 8))
    opt.state[aux]["step_tensor"] = torch.tensor(3.0)
    opt.state[aux]["row_stat"] = torch.zeros(8)

    assert reset_mismatched_optimizer_state(opt) == []
    assert "step_tensor" in opt.state[aux]
    assert "row_stat" in opt.state[aux]
