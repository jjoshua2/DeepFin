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
  width, `load_state_dict_tolerant` drops the mismatched tensors with only a
  `print()`, and selfplay serves a RANDOM-INITIALISED `policy_soft`. Nothing
  crashes, and because both are auxiliary heads no metric need move.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from chess_anti_engine.model import (
    ARCH_SCHEMA_VERSION,
    ModelConfig,
    build_model,
    model_config_from_flat_config,
    model_config_from_manifest_dict,
    model_config_to_manifest_dict,
    resume_model_config_from_arch,
)
from chess_anti_engine.model.transformer import (
    AUX_POLICY_HEAD_DIM_HEADS,
    AttentionPolicyHead,
    ChessNet,
)
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
    `load_state_dict_tolerant` drops `policy_soft.q/k` with only a `print()` and
    selfplay serves a randomly-initialised head.
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
def test_width_migration_survives_resume(
    checkpoint_value: int | None, config_value: int | None,
) -> None:
    """⚑ Resume takes topology from the checkpoint's `arch`, so a key outside
    `_RESUME_CONFIG_OWNED_ENCODING_KEYS` reverts to the donor on EVERY resume --
    silently, because the tolerant loader is happy either way. Without this the
    yaml edit would look applied (it is in `config`) while the trial kept
    building the donor's width for the whole readout window.
    """
    donor = _cfg(checkpoint_value)
    arch = dataclasses.asdict(donor)
    arch["_schema_version"] = ARCH_SCHEMA_VERSION
    run_cfg = dataclasses.replace(donor, aux_policy_head_dim=config_value)
    assert resume_model_config_from_arch(arch, run_cfg).aux_policy_head_dim == config_value


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
