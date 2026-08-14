"""Resuming into a model with FEWER parameters must keep the donor's moments.

The defect these tests close: ``Trainer.load`` reached
``_remap_optimizer_state_for_new_params`` only when
``n_ckpt_params < n_model_params``. That splice inserts fresh slots by POSITION,
so it is defined only for additions — and the gate meant a warm start that
REMOVED parameters skipped it entirely, ``Optimizer.load_state_dict`` raised on
the group size, and the handler installed a FRESH optimizer. Measured on arm B
(``policy_embedding`` +2, ``value_categorical`` −6,
``value_categorical_coupled`` +2, 481 → 479): every donor moment in a 63M-param
run discarded at WARNING level inside a Ray actor.

⚑ The hard part is CORRESPONDENCE, and it is why these tests assert on VALUES.
``Optimizer.state_dict()`` keys its ``state`` by integer index into the
flattened ``param_groups`` order, so deleting one parameter shifts every later
index. A repair that maps positionally would produce a fully populated,
plausible optimizer state in which one parameter carries another's moments —
strictly worse than the clean wipe it replaced, and invisible to any
"is the state non-empty" check. Every assertion below is therefore per
``(parameter name, state key)`` against a value banked under that exact pair,
following ``tests/test_shared_policy_embedding.py``'s method.

Both layouts are exercised. Production is ``aurora`` + ``matrix_optimizer_scope:
mlp_out`` — four param groups, where the Aurora matrix group stores
``momentum_buffer`` only and the AdamW aux groups store
``exp_avg``/``exp_avg_sq``/``step``. A single-group ``adamw`` test cannot observe
a parameter moving between groups, and an ``exp_avg``-only assertion would skip
the matrix group entirely.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import pytest
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.train.trainer import Trainer

# (optimizer, matrix_optimizer_scope). The second is PRODUCTION's layout.
_LAYOUTS = [("adamw", "default"), ("aurora", "mlp_out")]


def _cfg(
    *,
    coupled: bool = False,
    policy_embedding_mode: str = "off",
    enable_policy_sf_head: bool = True,
) -> ModelConfig:
    return ModelConfig(
        kind="transformer",
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        use_smolgen=False,
        categorical_head_coupled=coupled,
        policy_embedding_mode=policy_embedding_mode,
        enable_policy_sf_head=enable_policy_sf_head,
    )


def _trainer(cfg: ModelConfig, log_dir: Path, optimizer: str, scope: str) -> Trainer:
    return Trainer(
        build_model(cfg), device="cpu", lr=1e-3, optimizer=optimizer,
        matrix_optimizer_scope=scope, warmup_steps=10, warmup_lr_start=1e-5,
        use_amp=False, log_dir=log_dir, tb_log_interval=1000,
        prefetch_batches=False, model_config=cfg,
  # Required by the `enable_policy_sf_head=False` arm: the trainer refuses to
  # weight a `policy_sf` loss for a model with no such head. Gradients here are
  # set by hand, so no loss weight influences what these tests measure.
        w_sf_move=0.0,
    )


def _bank_fingerprints(trainer: Trainer) -> dict[tuple[str, str], float]:
    """Step for real, then stamp a DISTINCT value into every state tensor.

    ⚑ An unstepped donor makes the whole test vacuous: with no moments to
    preserve, "the donor's moments survived" is true of a fresh optimizer too.
    Deterministic enumeration rather than a hash, because a hash COLLISION is
    precisely the case a mis-mapping would survive.
    """
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param)
    trainer.opt.step()
    by_id = {id(p): n for n, p in trainer.model.named_parameters()}
    banked: dict[tuple[str, str], float] = {}
    stamp = 0
    for group in trainer.opt.param_groups:
        for param in group["params"]:
            state = trainer.opt.state.get(param)
            if not state:
                continue
            for key, value in state.items():
                if not torch.is_tensor(value) or not value.is_floating_point():
                    continue
                stamp += 1
                state[key] = torch.full_like(value, float(stamp) + 0.5)
                banked[by_id[id(param)], key] = float(stamp) + 0.5
    assert banked, "donor carries no optimizer moments — the test would be vacuous"
    assert len(set(banked.values())) == len(banked)
    return banked


def _read_state(trainer: Trainer) -> dict[tuple[str, str], float]:
    by_id = {id(p): n for n, p in trainer.model.named_parameters()}
    seen: dict[tuple[str, str], float] = {}
    for group in trainer.opt.param_groups:
        for param in group["params"]:
            state = trainer.opt.state.get(param)
            if not state:
                continue
            for key, value in state.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    seen[by_id[id(param)], key] = float(value.flatten()[0])
    return seen


def _two_real_steps(trainer: Trainer) -> None:
    for _ in range(2):
        for param in trainer.model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param) * 1e-3
        trainer.opt.step()


def _strip_manifest(ckpt_path: Path) -> None:
    """Drop ``opt_param_names``, forcing the reconstruction-from-``model`` path.

    Every checkpoint that exists TODAY — including the arm B donor this defect
    was measured on — predates the manifest, so the reconstruction is the path
    production actually takes on the first resume after this change. A suite that
    only ever exercised the manifest would leave it untested.
    """
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    payload.pop("opt_param_names", None)
    torch.save(payload, str(ckpt_path))


def _warm_start(
    donor_cfg: ModelConfig,
    arm_cfg: ModelConfig,
    tmp_path: Path,
    optimizer: str,
    scope: str,
    *,
    manifest: str,
    caplog: pytest.LogCaptureFixture,
) -> tuple[Trainer, dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    donor = _trainer(donor_cfg, tmp_path / "donor", optimizer, scope)
    banked = _bank_fingerprints(donor)
    ckpt = tmp_path / "donor.pt"
    donor.save(ckpt)
    if manifest == "reconstructed":
        _strip_manifest(ckpt)

    arm = _trainer(arm_cfg, tmp_path / "arm", optimizer, scope)
    with caplog.at_level(logging.WARNING):
        arm.load(ckpt)
    assert not any("reinitialising optimizer" in r.message for r in caplog.records), (
        "the whole-optimizer reinit fired — the donor state was discarded"
    )
    return arm, banked, _read_state(arm)


@pytest.mark.parametrize("manifest", ["recorded", "reconstructed"])
@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_add_and_remove_in_ONE_load_keeps_every_surviving_moment(
    manifest: str, optimizer: str, scope: str, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑ ARM B's ACTUAL CASE: +2 / −6 / +2 simultaneously, a NET DECREASE.

    ``policy_embedding_mode`` adds two parameters, ``categorical_head_coupled``
    removes the standalone ``ValueHead``'s six and adds the coupled ``Linear``'s
    two. The donor is LONGER, so the pre-existing positional splice was
    unreachable by construction; and the six removals sit in the MIDDLE of the
    flattened order, so every later index shifts.
    """
    donor_cfg = _cfg()
    arm_cfg = _cfg(coupled=True, policy_embedding_mode="linear")
    arm, banked, seen = _warm_start(
        donor_cfg, arm_cfg, tmp_path, optimizer, scope, manifest=manifest, caplog=caplog,
    )

    donor_params = sum(1 for p in build_model(donor_cfg).parameters() if p.requires_grad)
    arm_params = sum(1 for p in arm.model.parameters() if p.requires_grad)
    assert arm_params < donor_params, "this arm must be a NET DECREASE"

    added = {"policy_embedding", "value_categorical_coupled"}
    removed_prefix = "value_categorical."
    for (name, key), value in seen.items():
        if name.split(".")[0] in added:
            pytest.fail(f"{name}.{key} carries donor state for a NEW parameter: {value}")
        assert (name, key) in banked, (
            f"{name}.{key} carries state no donor parameter banked — the mapping "
            "re-associated moments across parameters"
        )
        assert value == banked[name, key], (
            f"{name}.{key} got the moment banked for a DIFFERENT parameter: "
            f"{value} != {banked[name, key]}"
        )
    # ...and nothing that survived was dropped.
    survivors = {
        (name, key) for (name, key) in banked if not name.startswith(removed_prefix)
    }
    assert survivors <= set(seen), sorted(survivors - set(seen))
    # The removed head banked real moments, and none of them are still installed.
    assert any(name.startswith(removed_prefix) for name, _ in banked)
    assert not any(name.startswith(removed_prefix) for name, _ in seen)

    _two_real_steps(arm)


@pytest.mark.parametrize("manifest", ["recorded", "reconstructed"])
@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_a_PURE_decrease_keeps_every_surviving_moment(
    manifest: str, optimizer: str, scope: str, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dropping the dead ``policy_sf`` head: removals only, nothing added.

    ``_FRESH_PARAM_NAME_SUFFIXES``' comment named this direction as the one that
    "still resets"; it no longer does.
    """
    donor_cfg = _cfg()
    arm_cfg = _cfg(enable_policy_sf_head=False)
    arm, banked, seen = _warm_start(
        donor_cfg, arm_cfg, tmp_path, optimizer, scope, manifest=manifest, caplog=caplog,
    )

    assert any(name.startswith("policy_sf.") for name, _ in banked)
    assert not any(name.startswith("policy_sf.") for name, _ in seen)
    for (name, key), value in seen.items():
        assert (name, key) in banked, f"{name}.{key} carries state no donor banked"
        assert value == banked[name, key], (
            f"{name}.{key} got a DIFFERENT parameter's moment: "
            f"{value} != {banked[name, key]}"
        )
    survivors = {(name, key) for (name, key) in banked if not name.startswith("policy_sf.")}
    assert survivors <= set(seen), sorted(survivors - set(seen))

    _two_real_steps(arm)


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_the_INCREASE_path_is_not_regressed(
    optimizer: str, scope: str, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The pre-existing control: adding ``policy_embedding`` alone still warm-starts."""
    arm, banked, seen = _warm_start(
        _cfg(), _cfg(policy_embedding_mode="linear"), tmp_path, optimizer, scope,
        manifest="recorded", caplog=caplog,
    )
    for (name, key), value in seen.items():
        if name.startswith("policy_embedding"):
            pytest.fail(f"{name}.{key} inherited donor state {value}")
        assert value == banked[name, key], f"{name}.{key} moved: {value}"
    assert set(banked) <= set(seen), sorted(set(banked) - set(seen))
    _two_real_steps(arm)


def test_the_positional_splice_still_covers_a_donor_without_recoverable_names(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Name recovery declining must fall back to the splice, not to a reset.

    ``input_global_embedding: bt4_board`` re-homes the ``global_board_*``
    parameters into extra param groups by NAME, which ``decay_bucket_index``
    alone cannot reproduce, so ``_decay_group_layout`` is withheld — a real
    config, not a monkeypatched one. With the manifest stripped as well (every
    checkpoint written before this change), nothing can recover the donor's slot
    names and the increase-only positional splice is the remaining repair: the
    branch that prints ``Spliced N fresh parameter slot(s)``.
    """
    branch = {"input_global_embedding": "bt4_board", "input_global_embedding_channels": 4}
    donor_cfg = dataclasses.replace(_cfg(), **branch)
    arm_cfg = dataclasses.replace(_cfg(policy_embedding_mode="linear"), **branch)
    donor = _trainer(donor_cfg, tmp_path / "donor", "aurora", "mlp_out")
    assert donor._decay_group_layout is None, "this config must produce branch groups"
    banked = _bank_fingerprints(donor)
    ckpt = tmp_path / "donor.pt"
    donor.save(ckpt)
    _strip_manifest(ckpt)

    arm = _trainer(arm_cfg, tmp_path / "arm", "aurora", "mlp_out")
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert arm._donor_optimizer_param_names(payload) is None
    with caplog.at_level(logging.WARNING):
        arm.load(ckpt)
    assert not any("reinitialising optimizer" in r.message for r in caplog.records)
    seen = _read_state(arm)
    assert seen, "the fallback installed no state at all"
    for (name, key), value in seen.items():
        if name.startswith("policy_embedding"):
            continue
        assert value == banked[name, key], f"{name}.{key}: {value}"


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_the_reconstruction_reproduces_the_recorded_manifest_exactly(
    optimizer: str, scope: str, tmp_path: Path,
) -> None:
    """The inference and the record must agree, name for name and slot for slot.

    ``_donor_optimizer_param_names`` reconstructs the donor's slot order from
    ``state_dict`` key order + storage identity + ``decay_bucket_index``. This is
    the direct check on that inference: it must reproduce what the optimizer
    itself reported, in order, for every slot.
    """
    trainer = _trainer(_cfg(), tmp_path / "t", optimizer, scope)
    _bank_fingerprints(trainer)
    ckpt = tmp_path / "t.pt"
    trainer.save(ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    recorded = payload["opt_param_names"]
    assert recorded == trainer._optimizer_param_names()

    payload.pop("opt_param_names")
    assert trainer._donor_optimizer_param_names(payload) == recorded


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_the_reconstruction_survives_TIED_weights(
    optimizer: str, scope: str, tmp_path: Path,
) -> None:
    """⚑ Production's `state_dict` has MORE keys than it has parameters.

    `smolgen_mode: per_layer` shares ONE `gen_weight` across every layer, so the
    saved `state_dict` carries a key per layer while `named_parameters()` — and
    therefore the optimizer — holds the tensor once. Production: 496 keys, 481
    parameters, and the 15 extra keys are exactly why the model is 63.08M and not
    78.81M. A reconstruction that counted keys would place every later slot one
    name too far along; here that is a 1-key shift on a 2-layer model, which the
    slot-for-slot comparison against the optimizer's own manifest catches.

    The other tests in this file build `use_smolgen=False` models, where no
    tensor is tied and this branch never runs.
    """
    cfg = dataclasses.replace(_cfg(), use_smolgen=True, smolgen_mode="per_layer")
    trainer = _trainer(cfg, tmp_path / "t", optimizer, scope)
    n_keys = len(trainer.model.state_dict())
    n_params = sum(1 for _ in trainer.model.named_parameters())
    assert n_keys > n_params, "this config ties nothing — the test would be vacuous"

    _bank_fingerprints(trainer)
    ckpt = tmp_path / "t.pt"
    trainer.save(ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    recorded = payload.pop("opt_param_names")
    assert trainer._donor_optimizer_param_names(payload) == recorded


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_the_remap_DECLINES_rather_than_guess_when_its_check_fails(
    optimizer: str, scope: str, tmp_path: Path,
) -> None:
    """A mapping that cannot be verified must not be acted on.

    ⚑ The failure this guards is not "no repair"; it is a repair that installs a
    complete, non-empty, WRONG optimizer state. Declining returns the caller to
    its previous behaviour, which is recoverable; a silent mis-association is not.
    """
    trainer = _trainer(_cfg(), tmp_path / "t", optimizer, scope)
    _bank_fingerprints(trainer)
    ckpt = tmp_path / "t.pt"
    trainer.save(ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    payload.pop("opt_param_names")

    # Donor group sizes that the replayed bucketing cannot produce.
    bad = {
        "state": payload["opt"]["state"],
        "param_groups": [
            {**g, "params": list(g["params"])} for g in payload["opt"]["param_groups"]
        ],
    }
    moved = bad["param_groups"][-1]["params"].pop()
    bad["param_groups"][0]["params"].append(moved)
    assert trainer._donor_optimizer_param_names({**payload, "opt": bad}) is None

    # A donor moment whose shape disagrees with the tensor its recovered name
    # points at means the recovery is wrong, whatever the counts say. Rotating
    # the names by one is the shape an off-by-N slot walk has.
    #
    # ⚑ Defence in depth, NOT the correctness argument: a shift that happens to
    # land every moment on a same-shaped parameter passes this check, which is
    # why `test_the_correspondence_is_IDENTITY_not_POSITION` asserts the mapping
    # itself and the integration tests assert per-parameter VALUES.
    names = trainer._donor_optimizer_param_names(payload)
    assert names is not None
    rotated = [*names[1:], names[0]]
    assert trainer._remap_optimizer_state_by_param_name(
        payload["opt"], rotated, payload["model"],
    ) is None


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_the_correspondence_is_IDENTITY_not_POSITION(
    optimizer: str, scope: str, tmp_path: Path,
) -> None:
    """⚑ THE INDEX-SHIFT HAZARD, asserted directly on the mapping.

    Two donor slots carrying identically shaped tensors are given each other's
    names. Nothing about the shapes, the counts or the group sizes changes, so
    every structural check still passes and a positional mapping would sail
    through — leaving each parameter holding the other's moments: non-empty,
    correctly shaped, steppable, and wrong. The only thing that distinguishes
    the two mappings is which NAME each slot answers to, so that is what this
    asserts.
    """
    trainer = _trainer(_cfg(), tmp_path / "t", optimizer, scope)
    _bank_fingerprints(trainer)
    ckpt = tmp_path / "t.pt"
    trainer.save(ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    live_names = payload["opt_param_names"]
    model_state = payload["model"]

    group_of = {
        slot: index
        for index, group in enumerate(payload["opt"]["param_groups"])
        for slot in group["params"]
    }
    pair = next(
        (i, j)
        for i in range(len(live_names))
        for j in range(i + 1, len(live_names))
        if group_of[i] == group_of[j]
        and model_state[live_names[i]].shape == model_state[live_names[j]].shape
    )
    i, j = pair
    donor_names = list(live_names)
    donor_names[i], donor_names[j] = donor_names[j], donor_names[i]

    result = trainer._remap_optimizer_state_by_param_name(
        payload["opt"], donor_names, model_state,
    )
    assert result is not None, "the swap is a legal relayout; declining hides it"
    remapped, _ = result
    donor_state = payload["opt"]["state"]
    assert remapped["state"][j] is donor_state[i], (
        f"slot {i} carries the name now at live index {j}, so its moments must "
        f"land there — a positional mapping would have left them at {i}"
    )
    assert remapped["state"][i] is donor_state[j]
    # Everything else is untouched, so a mapping that shifted wholesale fails here.
    for slot in range(len(live_names)):
        if slot not in (i, j) and slot in donor_state:
            assert remapped["state"][slot] is donor_state[slot]


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_an_ordinary_resume_is_left_byte_identical(
    optimizer: str, scope: str, tmp_path: Path,
) -> None:
    """No layout change ⇒ the name path declines and the untouched code runs.

    Resume is the hottest path in the trainer; a repair that rewrote its
    optimizer state on every restart would put every run at the mercy of this
    code. Declining on the identity mapping keeps that blast radius at zero.
    """
    trainer = _trainer(_cfg(), tmp_path / "t", optimizer, scope)
    _bank_fingerprints(trainer)
    ckpt = tmp_path / "t.pt"
    trainer.save(ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)

    twin = _trainer(_cfg(), tmp_path / "twin", optimizer, scope)
    names = twin._donor_optimizer_param_names(payload)
    assert names is not None
    assert twin._remap_optimizer_state_by_param_name(
        payload["opt"], names, payload["model"],
    ) is None

    twin.load(ckpt)
    assert _read_state(twin) == _read_state(trainer)
