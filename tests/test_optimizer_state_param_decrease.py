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
import hashlib
import logging
from pathlib import Path
import pytest
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.train.trainer import Trainer, UntrustedOptimizerStateError

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


def _read_state_exact(trainer: Trainer) -> dict[tuple[str, str], object]:
    """Every optimizer-state value in full, plus the group hyperparameters.

    ⚑ `_read_state` above is a cheap FINGERPRINT: one element of each floating
    tensor. That is enough to tell two arms apart, and nowhere near enough to
    support the phrase "byte identical" -- it never reads `step` (a python int
    here), never reads element 1 onward, and never reads the group
    hyperparameters at all. A test whose NAME claims byte identity has to
    actually compare the bytes, so this hashes each tensor whole and keeps
    non-tensor state verbatim.
    """
    by_id = {id(p): n for n, p in trainer.model.named_parameters()}
    out: dict[tuple[str, str], object] = {}
    for g_index, group in enumerate(trainer.opt.param_groups):
        for key, value in sorted(group.items()):
            if key != "params":
                out[f"group{g_index}", key] = value
        for param in group["params"]:
            state = trainer.opt.state.get(param)
            if not state:
                continue
            for key, value in state.items():
                slot = (by_id[id(param)], key)
                if torch.is_tensor(value):
                    out[slot] = hashlib.sha256(
                        value.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                else:
                    out[slot] = value
    return out


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
    with pytest.raises(UntrustedOptimizerStateError):
        trainer._remap_optimizer_state_by_param_name(
            payload["opt"], rotated, payload["model"],
        )


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
    remapped, _report, _kept, _changed = result
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
  # SHA-256 of every state tensor whole, `step` verbatim, and all four groups'
  # hyperparameters -- not the one-element-per-tensor fingerprint `_read_state`
  # takes, which this assertion USED to make while its name claimed otherwise.
    assert _read_state_exact(twin) == _read_state_exact(trainer)


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_a_manifest_name_absent_from_the_donor_payload_declines(
    optimizer: str, scope: str, tmp_path: Path,
) -> None:
    """⚑⚑ A recovered name that is not a donor model key must DECLINE, not skip.

    Found by independent review 2026-08-14, after the fix this file exists for
    had already shipped. ``opt_param_names`` and ``ckpt["model"]`` are written by
    the same ``save``, so the manifest's names ARE that payload's keys. If a
    lookup misses, the mapping is wrong — not "this one slot is unverifiable".

    The original code did ``continue`` on a miss. A STALE manifest (a module
    renamed between save and load) misses on EVERY slot, so the shape guard
    inspected nothing, ``state_remap`` came out empty, and an EMPTY optimizer
    state installed *successfully*: ``0 kept, N dropped, N fresh``, no WARNING,
    and ``optimizer_state_loaded`` still True — so the scheduler and zclip were
    restored on top of a cold optimizer. That is the exact wipe this module
    repairs, moved into the quiet channel one level down.

    Two arms, because they fail differently: a WHOLE stale manifest (the loud
    case, which produced an empty state) and a SINGLE renamed parameter (the
    quiet case, which silently dropped one parameter's moments while reporting a
    healthy-looking count).
    """
    donor_cfg = _cfg()
    trainer = _trainer(donor_cfg, tmp_path / "t", optimizer, scope)
    _bank_fingerprints(trainer)
    ckpt = tmp_path / "t.pt"
    trainer.save(ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert isinstance(payload.get("opt_param_names"), list), "manifest must exist"

    # The arm: a real layout change, so the name path is genuinely engaged.
    arm_cfg = _cfg(coupled=True, policy_embedding_mode="linear")
    arm = _trainer(arm_cfg, tmp_path / "arm", optimizer, scope)

    # CONTROL — with the manifest intact this same load is ACCEPTED. Without
    # this the test could pass by declining for any unrelated reason.
    names = arm._donor_optimizer_param_names(payload)
    assert names is not None
    accepted = arm._remap_optimizer_state_by_param_name(
        payload["opt"], names, payload["model"],
    )
    assert accepted is not None, "control: an intact manifest must be accepted"
    assert accepted[2] > 0, "control: the accepted mapping must keep real state"

    # ARM 1 — the whole manifest is stale (every name renamed).
    stale = dict(payload)
    stale["opt_param_names"] = [f"renamed.{n}" for n in payload["opt_param_names"]]
    stale_names = arm._donor_optimizer_param_names(stale)
    assert stale_names is not None, "a stale manifest is still well-FORMED"
    with pytest.raises(UntrustedOptimizerStateError):
        arm._remap_optimizer_state_by_param_name(
            stale["opt"], stale_names, stale["model"],
        )

    # ARM 2 — exactly one parameter renamed. The quiet case: pre-fix this
    # returned a plausible mapping that silently dropped that parameter's state.
    one = dict(payload)
    renamed = list(payload["opt_param_names"])
    victim = next(
        i for i, n in enumerate(renamed)
        if n in payload["model"] and torch.is_tensor(payload["model"][n])
    )
    renamed[victim] = f"renamed.{renamed[victim]}"
    one["opt_param_names"] = renamed
    one_names = arm._donor_optimizer_param_names(one)
    assert one_names is not None
    with pytest.raises(UntrustedOptimizerStateError):
        arm._remap_optimizer_state_by_param_name(
            one["opt"], one_names, one["model"],
        )


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_duplicate_donor_slot_ids_decline(
    optimizer: str, scope: str, tmp_path: Path,
) -> None:
    """Two slots sharing an id would collapse under ``dict(zip(...))``.

    ``strict=True`` cannot catch it — the sequences are the same LENGTH — so one
    id would silently take whichever name came last, and every state entry keyed
    under the other would be dropped or mis-placed.
    """
    trainer = _trainer(_cfg(), tmp_path / "t", optimizer, scope)
    _bank_fingerprints(trainer)
    ckpt = tmp_path / "t.pt"
    trainer.save(ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)

    arm = _trainer(
        _cfg(coupled=True, policy_embedding_mode="linear"),
        tmp_path / "arm", optimizer, scope,
    )
    names = arm._donor_optimizer_param_names(payload)
    assert names is not None
    assert arm._remap_optimizer_state_by_param_name(
        payload["opt"], names, payload["model"],
    ) is not None, "control: the un-corrupted donor is accepted"

    opt_state = dict(payload["opt"])
    groups = [dict(g) for g in opt_state["param_groups"]]
    flat = [pid for g in groups for pid in g["params"]]
    assert len(flat) >= 2

  # ⚑⚑ THE FIXTURE HAS TO DEFEAT EVERY OTHER GUARD, OR IT TESTS THE WRONG ONE.
  # The first version of this test corrupted an arbitrary slot, and the mutation
  # run caught it: removing the duplicate-id check did NOT make it fail, because
  # orphaning a slot id also orphans a STATE key, and the name-resolution guard
  # (`name is None -> return None`) declines first. Passing for the wrong reason
  # is the same defect as not running.
  #
  # So: pick two parameters of IDENTICAL shape in the SAME group (the shape guard
  # then cannot fire), point the second's slot at the first's id, and DELETE the
  # now-orphaned state entry (the name guard then cannot fire). What is left is a
  # donor whose every check passes while two slots share one id -- so one
  # parameter's moments land on ANOTHER parameter. Only the duplicate-id check
  # stands between that and the optimizer.
    shape_of_name = {
        n: tuple(payload["model"][n].shape)
        for n in names
        if torch.is_tensor(payload["model"].get(n))
    }
    pair = None
    for group in groups:
        by_shape: dict[tuple[int, ...], list[int]] = {}
        for pid in group["params"]:
            name = names[pid] if pid < len(names) else None
            if name is None or name not in shape_of_name:
                continue
            by_shape.setdefault(shape_of_name[name], []).append(pid)
        for same in by_shape.values():
            if len(same) >= 2:
                pair = (same[0], same[1], group)
                break
        if pair:
            break
    if pair is None:
        pytest.skip("no two same-shape parameters share a group in this layout")
    keep_id, dup_id, group = pair

    group["params"] = [keep_id if pid == dup_id else pid for pid in group["params"]]
    opt_state["param_groups"] = groups
    state = {k: v for k, v in opt_state["state"].items() if k != dup_id}
    opt_state["state"] = state

  # Prove the fixture really is invisible to the other guards: every surviving
  # state key still resolves to a name, and to a tensor of the right shape.
    surviving = dict(zip([p for g in groups for p in g["params"]], names, strict=True))
    for sid in state:
        assert sid in surviving, "fixture would trip the NAME guard, not the id guard"
        assert surviving[sid] in shape_of_name, "fixture would trip the name guard"

    with pytest.raises(UntrustedOptimizerStateError):
        arm._remap_optimizer_state_by_param_name(
            opt_state, names, payload["model"],
        )


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_an_ndim_mismatch_declines(
    optimizer: str, scope: str, tmp_path: Path,
) -> None:
    """A moment whose ndim disagrees with its parameter is the STRONGEST signal.

    The original guard compared shapes only when ``value.dim() ==
    donor_tensor.dim()``, so it waved through precisely the case that proves the
    mapping wrong. Only 0-dim tensors (step counters, which carry no shape
    relationship to the parameter) are legitimately exempt.
    """
    trainer = _trainer(_cfg(), tmp_path / "t", optimizer, scope)
    _bank_fingerprints(trainer)
    ckpt = tmp_path / "t.pt"
    trainer.save(ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)

    arm = _trainer(
        _cfg(coupled=True, policy_embedding_mode="linear"),
        tmp_path / "arm", optimizer, scope,
    )
    names = arm._donor_optimizer_param_names(payload)
    assert names is not None
    assert arm._remap_optimizer_state_by_param_name(
        payload["opt"], names, payload["model"],
    ) is not None, "control: the un-corrupted donor is accepted"

    opt_state = dict(payload["opt"])
    state = {k: dict(v) for k, v in opt_state["state"].items()}
    slot_of_name = {n: i for i, n in enumerate(names)}
    victim = next(
        n for n in names
        if slot_of_name[n] in state
        and torch.is_tensor(payload["model"].get(n))
        and payload["model"][n].dim() >= 2
    )
    entry = state[slot_of_name[victim]]
    key = next(k for k, v in entry.items() if torch.is_tensor(v) and v.dim() >= 2)
    # Same NUMEL, different ndim — invisible to a shape check gated on equal dim.
    entry[key] = entry[key].flatten()
    opt_state["state"] = state
    with pytest.raises(UntrustedOptimizerStateError):
        arm._remap_optimizer_state_by_param_name(
            opt_state, names, payload["model"],
        )


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_a_parameter_that_moves_between_groups_does_not_crash(
    optimizer: str, scope: str, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """New reachable state: the name path can move a survivor between groups.

    The positional splice structurally could not — it preserved group order. The
    name path can, so a parameter may arrive in the Aurora matrix group carrying
    only ``exp_avg``/``exp_avg_sq``, or in an AdamW aux group carrying only
    ``momentum_buffer``.

    This is benign ONLY because ``AuroraWithAuxAdam`` and ``MuonWithAuxAdam``
    both probe per-KEY with ``.get()`` rather than gating on ``len(state) == 0``.
    That assumption is load-bearing and was untested; torch's own AdamW would
    ``KeyError``. Pinning it here so a future optimizer edit that switches to a
    ``len(state)`` probe fails loudly instead of at the first ``opt.step()`` of a
    live warm start.
    """
    trainer = _trainer(_cfg(), tmp_path / "t", optimizer, scope)
    _bank_fingerprints(trainer)
    ckpt = tmp_path / "t.pt"
    trainer.save(ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)

  # ⚑ The arm changes `matrix_optimizer_scope`, and that is the POINT. Adding a
  # parameter does not move any SURVIVOR between groups -- an existing
  # parameter's bucket is decided by its own name/shape, which a new sibling does
  # not touch -- so an add-only arm leaves this test asserting nothing (it skipped
  # in BOTH layouts before this line existed). Widening the matrix scope
  # re-buckets real parameters, which is exactly the reachable construction the
  # independent review used to demonstrate cross-group movement.
    arm_scope = "block_all" if scope == "mlp_out" else scope
    arm_cfg = _cfg(coupled=True, policy_embedding_mode="linear")
    arm = _trainer(arm_cfg, tmp_path / "arm", optimizer, arm_scope)

  # ⚑ ASSERT THE PREMISE FIRST. Loading and stepping proves nothing about
  # cross-group movement unless a surviving parameter ACTUALLY changed group --
  # if none does, this test passes while exercising none of the behaviour it
  # names, which is the exact defect the F9 mutant exposed one test above.
  # `adamw`/`default` is a single group and cannot express the property, so it
  # SKIPS rather than banking a free pass.
    def _group_of_name(names: list[str], sizes: list[int]) -> dict[str, int]:
        out: dict[str, int] = {}
        cursor = 0
        for index, size in enumerate(sizes):
            for name in names[cursor:cursor + size]:
                out[name] = index
            cursor += size
        return out

    donor_names = arm._donor_optimizer_param_names(payload)
    assert donor_names is not None
    donor_sizes = [len(g["params"]) for g in payload["opt"]["param_groups"]]
    live_names = arm._optimizer_param_names()
    assert live_names is not None
    live_sizes = [len(g["params"]) for g in arm.opt.param_groups]
    if len(donor_sizes) < 2:
        pytest.skip("single-group layout cannot express a cross-group move")
    donor_group = _group_of_name(list(donor_names), donor_sizes)
    live_group = _group_of_name(list(live_names), live_sizes)
    moved = [
        n for n in set(donor_group) & set(live_group)
        if donor_group[n] != live_group[n]
    ]
    if not moved:
        pytest.skip(
            "no surviving parameter changes group in this arm — the property "
            "under test is not reachable here, so passing would prove nothing"
        )

    # ⚑ ASSERT THE NAME PATH ACTUALLY RAN, not merely that loading survived.
    # `load` treats a `None` remap as "not applicable" and falls through to the
    # INDEX-keyed `load_state_dict`, which -- with equal counts -- produces state
    # that is non-empty, correctly shaped, steppable and finite. So every
    # assertion below this line is satisfied by the positional path too, and a
    # mutant that returns `None` from the remap passed this test unchanged. The
    # direct call pins the mechanism; the caplog check pins that `load` used it
    # rather than silently reinitialising.
    accepted = arm._remap_optimizer_state_by_param_name(
        payload["opt"], donor_names, payload["model"],
    )
    assert accepted is not None, (
        "the name path must ACCEPT this donor -- if it declines, the load below "
        "falls through to the positional splice and tests nothing about "
        "cross-group movement"
    )
    assert accepted[2] > 0, "the accepted mapping must carry real donor state"

    with caplog.at_level(logging.WARNING):
        arm.load(ckpt)
    assert not any("reinitialising optimizer" in r.message for r in caplog.records), (
        "the donor state must survive the group move, not be thrown away"
    )
    # Two real steps: initialisation of any missing moment happens on step 1, and
    # step 2 exercises the state it just built.
    _two_real_steps(arm)
    for group in arm.opt.param_groups:
        for param in group["params"]:
            for value in arm.opt.state.get(param, {}).values():
                if torch.is_tensor(value):
                    assert torch.isfinite(value).all()


@pytest.mark.parametrize(("optimizer", "scope"), _LAYOUTS)
def test_load_REFUSES_the_positional_fallback_when_the_name_map_is_untrusted(
    optimizer: str, scope: str, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑⚑ THE DECLINE MUST NOT LAND IN A POSITIONAL LOAD.

    Every other test here calls `_remap_optimizer_state_by_param_name` DIRECTLY,
    so none of them can see what `load` does with a `None`. Independent review
    2026-08-14 executed that path and found the answer: `load` skipped both
    warnings and fell through to `self.opt.load_state_dict(opt_state)`, which is
    keyed by INDEX. With the donor and live counts equal -- the normal case --
    every parameter received some OTHER parameter's moments: non-empty,
    correctly shaped, steppable, wrong, and completely unlogged, with
    `optimizer_state_loaded` left True so the scheduler and zclip were restored
    on top. `reset_mismatched_optimizer_state` cannot catch it because the
    shapes match by construction.

    So the guard against re-keying by position had, as its failure mode,
    re-keying by position.

    The fixture stales exactly ONE manifest name. That is enough to make the map
    untrusted while leaving the counts equal, which is precisely the shape that
    used to fall through silently.
    """
    trainer = _trainer(_cfg(), tmp_path / "t", optimizer, scope)
    banked = _bank_fingerprints(trainer)
    ckpt = tmp_path / "t.pt"
    trainer.save(ckpt)

    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    names = list(payload["opt_param_names"])
    assert len(names) >= 2, "fixture needs a real manifest"
    names[0] = f"{names[0]}.renamed_by_a_refactor"
    payload["opt_param_names"] = names
    torch.save(payload, str(ckpt))

    twin = _trainer(_cfg(), tmp_path / "twin", optimizer, scope)
    with caplog.at_level(logging.WARNING):
        twin.load(ckpt)

    assert any("REFUSING" in r.getMessage() for r in caplog.records), (
        "an untrusted name map was absorbed silently; the operator has no signal "
        f"that the optimizer state was not restored. records={[r.getMessage() for r in caplog.records]}"
    )
  # ⚑ The VALUE read, not the log read. `_bank_fingerprints` stamps a distinct
  # float into every donor moment, so ANY overlap here means donor moments were
  # installed -- which, given the map was rejected, could only have happened by
  # position.
    survived = set(_read_state(twin).values())
    assert not (survived & set(banked.values())), (
        "donor moments reached the live optimizer even though the name mapping "
        "was rejected -- they can only have been placed POSITIONALLY"
    )
