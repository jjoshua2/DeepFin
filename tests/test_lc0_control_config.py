"""The control config must be the production stack, minus exactly one change.

⚑ The prereg's premise is "our EXACT net and trainer, on someone else's data".
If the architecture or the optimizer drifts from production, the arm stops
being able to answer the question it was built for — and the drift would be
invisible, because both configs would still train.

So this pins the DIFF against `configs/pbt2_small.yaml` rather than pinning
absolute numbers. A pinned constant goes stale the moment production moves and
gets "fixed" by copying the new number across, which is how the two silently
diverge. A diff cannot be fixed that way: if production changes its
architecture, this test fails until someone decides whether the arm follows.

⚑⚑ "PRODUCTION" IS TWO DIFFERENT FILES HERE, AND WHICH ONE THIS TREE HOLDS
DEPENDS ON THE BRANCH. The live run reads `configs/pbt2_small.yaml` in the LIVE
working tree, on the live branch. A tree on `main` carries a copy that `main` has
committed nothing to since the live branch diverged; a tree ON the live branch
carries the live file itself. Measured 2026-08-16 from `main`: the two `model:`
sections differ by the whole bt4heads bundle (`aux_policy_head_dim: 128`,
`categorical_head_coupled: true`, `policy_embedding_mode: linear`) — 61,444,448
trainable params live against 63,084,128 in tree, and 17 `trainer_kwargs_from_config`
entries apart (re-measured 2026-08-29).

⚑⚑ SO THE REFERENCE DECISION IS A TWO-WORLD CONTRACT. Both of these are
legitimate states of this tree, and the guards below accept exactly these two:

* **STATE MAIN** — the in-tree config lacks every bt4heads `model:` key and its
  trainer kwargs drift from `LIVE_TRAINER_PIN` on the whole bt4heads trainer
  half. The pins are the reference BECAUSE the file next door is behind.
* **STATE LIVE** — the in-tree config carries every key the pin describes, at
  pin-equal values, and realizes an EMPTY trainer drift. The file next door IS
  the live file; the pin is a faithful second copy, and the reference decision
  is moot rather than wrong.

Anything between them FAILS: a half-adopted bundle is the one state in which
"the pin is the reference" reads as true and is not. ⚑ The pins stay the
arbiter in both worlds — STATE LIVE is accepted only because the file EQUALS the
pin's world, never because a mismatch was waved through as progress.

⇒ THE REFERENCE DECISION, per axis:

* ARCHITECTURE — judged against `LIVE_ARCH_PIN`, the recorded LIVE `model:`
  section, overlaid on the in-tree production config so every model-affecting
  key OUTSIDE `model:` still comes from a real production yaml. NOT against the
  in-tree `model:` section, which is stale in STATE MAIN and equal to the pin in
  STATE LIVE — either way the pin is the right reference, and WHICH world this
  tree is in is itself asserted, by
  `test_lc0_control_arch.py::test_the_pin_names_the_bt4heads_keys_the_in_tree_config_lacks`,
  so this file's reference cannot quietly become the wrong one. The pin is
  regenerable from the live file and its freshness has its own gate; the
  verdict here never depends on `$CHESS_LIVE_PRODUCTION_CONFIG`, because a
  guard that answers differently in the operator's shell than in CI is the
  defect, not the fix (review F6).
* TRAINER — judged against `LIVE_TRAINER_PIN`, the recorded LIVE
  `trainer_kwargs_from_config` signature, for the same reason and by the same
  construction. ⚑⚑ THIS GAP WAS OPEN UNTIL LATER ON 2026-08-16 AND THE
  DOCSTRING RECORDING IT IS KEPT: the live yaml's `train:`/`selfplay:` sections
  differed from `main`'s in THIRTEEN trainer kwargs (`w_sf_own_regret` 0.7 vs
  0.0, `w_categorical` 0.3 vs 1.0, `sf_wdl_frac` 0.5 vs 0.69, `lr_T0`,
  `warmup_steps`, `sf_target_params.sf_policy_score_mode`, ...), so the control
  was `main`'s committed trainer minus one change, not the live trainer minus
  one change. The consequence was not cosmetic: `search_wdl_frac: 0.70` was
  chosen to preserve "production's `game_frac` 0.30", and live's `game_frac`
  is **0.00** (`sf 0.69 + search 0.31 = 1.00`), so the number being preserved
  did not exist. The control now carries live's recipe, `search_wdl_frac` is
  1.0, and the allow-list below is down from thirteen-plus-two to two.

⚑ Neither axis's verdict depends on `$CHESS_LIVE_PRODUCTION_CONFIG`: a guard
that answers differently in the operator's shell than in CI is the defect, not
the fix (review F6). `scripts/lc0_control_arch_pin.py --check` is the mode that
reads the live file, and `test_lc0_control_arch.py` /
`test_lc0_control_trainer.py` re-derive each pin from it when it IS set.
"""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pytest

from chess_anti_engine.eval import lc0_control_replay
from chess_anti_engine.eval.lc0_control_arch import LIVE_ARCH_PIN, model_section
from chess_anti_engine.eval.lc0_control_trainer import (
    LC0_TRAINER_DEVIATIONS,
    LIVE_TRAINER_PIN,
    ControlTrainerDrift,
    assert_control_matches_live_trainer,
    trainer_kwargs_drift,
    trainer_kwargs_signature,
)
from chess_anti_engine.model import model_config_from_flat_config
from chess_anti_engine.train.trainer import trainer_kwargs_from_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
# ⚑ IMPORTED, NOT RESTATED. Both modules ask the same question of the same drift,
# and two copies of "what counts as stale" is exactly the divergence this file's
# `ALLOWED_TRAINER_DIFFS` comment argues against one paragraph up.
from tests.test_lc0_control_trainer import two_world_trainer_reference_problems

REPO = Path(__file__).resolve().parent.parent
PRODUCTION = REPO / "configs" / "pbt2_small.yaml"
CONTROL = REPO / "configs" / "lc0_positive_control.yaml"

# ⚑ The allow-list is `LC0_TRAINER_DEVIATIONS`, imported rather than restated:
# it is a mapping from kwarg NAME to the REASON it may differ, and it is what
# the launch guard uses. A second copy here would be free to drift from the one
# that gates the run — which is the whole failure mode this file exists for.
ALLOWED_TRAINER_DIFFS = set(LC0_TRAINER_DEVIATIONS)


@pytest.fixture(scope="module")
def configs() -> tuple[dict, dict]:
    return (
        flatten_run_config_defaults(load_yaml_file(str(PRODUCTION))),
        flatten_run_config_defaults(load_yaml_file(str(CONTROL))),
    )


@pytest.fixture(scope="module")
def live_architecture_reference() -> dict:
    """Production's config with the LIVE `model:` section overlaid.

    ⚑ The overlay, not the pin alone: several `ModelConfig` fields are read
    from keys that do not live under `model:` at all, and a reference built
    from `LIVE_ARCH_PIN["model"]` by itself would leave every one of them at
    its `model_config_from_flat_config` DEFAULT. The architecture diff would
    then be comparing the control against library defaults on those fields and
    passing only because the control also happens to use them — a check that
    agrees with a reference that says nothing, which is exactly the failure
    `assert_control_matches_live_architecture` refuses an empty `model:`
    section for.
    """
    raw = copy.deepcopy(load_yaml_file(str(PRODUCTION)))
    dropped = sorted(set(model_section(raw)) - set(LIVE_ARCH_PIN["model"]))
    assert dropped == [], (
        f"the in-tree production config has model: keys the pin does not "
        f"describe ({dropped}); the overlay would silently DROP them from the "
        "reference, which would make the architecture diff blind to exactly "
        "those keys. Regenerate the pin with scripts/lc0_control_arch_pin.py."
    )
    raw["model"] = dict(LIVE_ARCH_PIN["model"])
    return flatten_run_config_defaults(raw)


def test_control_config_loads_through_the_production_schema() -> None:
    """Category (a) from CLAUDE.md: an unknown key is FATAL AT LAUNCH.

    `flatten_run_config_defaults` is what `run.py` calls before its argument
    parser exists and outside any try, so a key that is not in
    `utils/config_yaml.py`'s schema means the process never starts. Loading the
    file here is the same check, run before anyone spends GPU time on it.
    """
    flat = flatten_run_config_defaults(load_yaml_file(str(CONTROL)))
    assert flat["model"] == "transformer"


def test_architecture_is_identical_to_production(
    configs: tuple[dict, dict], live_architecture_reference: dict,
) -> None:
    """⚑⚑ THE REFERENCE IS THE LIVE ARCHITECTURE, NOT THE FILE NEXT DOOR.

    Until 2026-08-16 this diffed against the in-tree `configs/pbt2_small.yaml`
    and passed while the control trained a net production does not run: the
    in-tree copy was stale by the whole bt4heads bundle, so the reference
    agreed with the control precisely because BOTH were missing the same three
    keys. Two things called "production" disagreed and this test was pointed at
    the wrong one.

    It now diffs the full `ModelConfig` against the pinned LIVE `model:`
    section overlaid on the production yaml (see `live_architecture_reference`).
    That keeps the shape the file's header argues for — a DIFF, not pinned
    constants, so "production moved" cannot be fixed by copying a number — and
    moves the reference onto the file production actually reads.

    ⚑ It is not the same check as
    `test_lc0_control_arch.py::test_the_control_config_matches_the_recorded_pin`,
    which compares raw `model:` MAPPINGS. This one compares built `ModelConfig`
    DATACLASSES, so it also covers every model-affecting key that does not live
    under `model:` and every normalisation the builder applies on the way — the
    axis on which the two configs could still diverge with identical `model:`
    sections.
    """
    _production, control = configs
    prod_model = model_config_from_flat_config(live_architecture_reference)
    ctrl_model = model_config_from_flat_config(control)
    differing = [
        field.name
        for field in dataclasses.fields(prod_model)
        if getattr(prod_model, field.name) != getattr(ctrl_model, field.name)
    ]
    assert differing == [], (
        f"the control's architecture drifted from production in {differing}. "
        "Any architecture change voids the arm — decide whether the control "
        "should follow production before updating this test."
    )


def test_only_the_value_blend_differs_from_the_live_trainer(
    configs: tuple[dict, dict],
) -> None:
    """⚑⚑ NAMED FOR ITS REFERENCE, BECAUSE THE REFERENCE IS THE WHOLE CLAIM.

    Until 2026-08-16 this diffed against the IN-TREE `configs/pbt2_small.yaml`
    and was named `..._from_the_in_tree_trainer` to say so. It passed while the
    control trained with a recipe production does not use: `main`'s copy is 13
    `trainer_kwargs_from_config` entries behind the live file, and the test
    agreed with the control precisely because both were behind by the same 13.
    Exactly the failure `test_architecture_is_identical_to_production` had, one
    axis over, and it outlived that fix by half a day.

    It now diffs against `LIVE_TRAINER_PIN`. The shape is unchanged — a DIFF
    with a named allow-list, not pinned constants, so "production moved" cannot
    be fixed by copying a number across — and the reference is the recipe
    production actually runs.
    """
    _production, control = configs
    reference = LIVE_TRAINER_PIN["kwargs"]
    drift = trainer_kwargs_drift(trainer_kwargs_signature(control), reference)
    assert set(drift) == ALLOWED_TRAINER_DIFFS, (
        f"trainer kwargs differ from LIVE in {sorted(drift)}, expected only "
        f"{sorted(ALLOWED_TRAINER_DIFFS)}. Differences: "
        + "; ".join(f"{k}: ctrl={c!r} live={live!r}" for k, (c, live) in drift.items())
        + ". Unexpected extras are usually a live key the control forgot to "
        "copy; unexpected absences mean production moved onto the control's "
        "value and the deviation's recorded reason now describes nothing."
    )


def test_the_in_tree_production_config_is_still_the_wrong_trainer_reference(
    configs: tuple[dict, dict],
) -> None:
    """⚑ WHICH WORLD this file's reference decision is being made in, ASSERTED.

    `test_only_the_value_blend_differs_from_the_live_trainer` judges against
    `LIVE_TRAINER_PIN` instead of the production yaml sitting next to it. That
    choice is right in both of the header's two worlds — because the file is
    behind (STATE MAIN), or because the file and the pin are the same recipe
    (STATE LIVE) — and it is WRONG in between, which is the state this asserts
    the absence of. Same construction, and the same shared predicate, as
    `test_lc0_control_trainer.py::test_the_pin_records_a_recipe_the_in_tree_config_does_not`.

    ⚑ The obvious relaxation is refused on purpose. `assert drift` said "the
    situation improving breaks the test", which was the right guard while only
    STATE MAIN existed; simply deleting it once STATE LIVE arrived would leave
    the reference decision unasserted, and widening it to "empty or non-empty"
    is a tautology. So the accepted states are named, and a half-adopted recipe
    still fails with the kwarg that moved.
    """
    production, _control = configs
    drift = trainer_kwargs_drift(
        trainer_kwargs_signature(production), LIVE_TRAINER_PIN["kwargs"],
    )
    problems = two_world_trainer_reference_problems(drift)
    assert not problems, (
        "the in-tree configs/pbt2_small.yaml is in NEITHER known world:\n  "
        + "\n  ".join(problems)
        + "\nIt has adopted part of LIVE's trainer recipe and not the rest, so "
        "this module's reference decision describes a file that no longer "
        "exists: re-read the header, and either finish the sync (drift goes "
        "empty, STATE LIVE) or re-record why the pin stays."
    )


def test_the_deviation_allow_list_states_a_reason_for_every_entry() -> None:
    """A named allow-list is only better than a set if the names carry weight.

    ⚑ The point of `LC0_TRAINER_DEVIATIONS` being a mapping is that an entry
    cannot be added without writing down why the arm may differ from production
    on that kwarg. A blank or placeholder reason turns it back into a set, and
    the launch guard's error message — which quotes the reason — into noise.
    """
    for key, reason in LC0_TRAINER_DEVIATIONS.items():
        assert len(reason) >= 80, (
            f"the deviation {key!r} is allowed with a {len(reason)}-character "
            "reason. Every entry has to say what breaks on this corpus if the "
            "arm carries production's value; that sentence is the only thing "
            "stopping the list from growing."
        )
        assert "LIVE" in reason, (
            f"the deviation {key!r} does not name the LIVE value it departs "
            "from. A reason that does not quote the number it is deviating "
            "from cannot go stale visibly when production moves."
        )


def test_the_trainer_guard_refuses_an_undeclared_deviation(
    configs: tuple[dict, dict],
) -> None:
    """⚑ A NEW TEST IS VACUOUS UNTIL IT IS SHOWN TO FAIL.

    The diff above asserts a state; this asserts the GUARD the driver actually
    launches behind can reach a failing one. Mutating a single loss weight —
    the smallest realistic drift, and the one that would be invisible in a
    training curve — must refuse the launch by name.
    """
    _production, control = configs
    drifted = copy.deepcopy(control)
    drifted["w_moves_left"] = float(drifted["w_moves_left"]) * 2.0
    with pytest.raises(ControlTrainerDrift, match="w_moves_left"):
        assert_control_matches_live_trainer(drifted, context="unit")


def test_the_trainer_guard_refuses_a_deviation_that_stopped_deviating(
    configs: tuple[dict, dict],
) -> None:
    """⚑ THE OTHER DIRECTION, WHICH IS THE ONE THAT ROTS QUIETLY.

    If production adopted `search_wdl_frac: 1.0`, the arm's recorded reason —
    "the freed 0.69 SF share, handed to lc0's own best_q/best_d" — would be
    describing a difference that no longer exists, and a guard that only looks
    for unexpected differences would keep passing while its own documentation
    became false.
    """
    _production, control = configs
    pin = {
        **LIVE_TRAINER_PIN,
        "kwargs": {
            **LIVE_TRAINER_PIN["kwargs"],
            "search_wdl_frac": float(control["search_wdl_frac"]),
        },
    }
    with pytest.raises(ControlTrainerDrift, match="does NOT differ"):
        assert_control_matches_live_trainer(control, pin=pin, context="unit")


def test_optimizer_is_the_production_one(configs: tuple[dict, dict]) -> None:
    """Stated separately because it is the claim people actually check."""
    _production, control = configs
    kwargs = trainer_kwargs_from_config(control)
    assert kwargs["optimizer"] == "aurora"
    assert kwargs["matrix_optimizer_scope"] == "mlp_out"


def test_the_value_blend_override_is_present_and_total(configs: tuple[dict, dict]) -> None:
    """Both halves of the override, and the share it was handed to.

    ⚑⚑ THE `game_frac` BAR MOVED 0.30 -> 0.00 ON 2026-08-16 AND IT WAS NEVER A
    TUNING CHOICE. This test used to assert `1 - blend == 0.30` "to match
    production's 0.30". That 0.30 was `1 - 0.50 - 0.20` off `main`'s committed
    yaml. Live runs `sf 0.69 + search 0.31 = 1.00`, so production's `game_frac`
    is ZERO — the assertion was pinning the control to a recipe production had
    already left, and the `search_wdl_frac: 0.70` it certified was introducing
    a 0.30 outcome share rather than preserving one.
    """
    _production, control = configs
    assert float(control["sf_wdl_frac"]) == 0.0
    assert float(control["sf_wdl_frac_floor"]) == 0.0
    assert float(control["search_wdl_frac"]) > 0.0, (
        "with sf_wdl_frac at 0 and search_wdl_frac at 0, the ENTIRE value "
        "target collapses onto the raw game outcome — the same defect the "
        "override exists to avoid, reached from the other side."
    )
    blend = float(control["sf_wdl_frac"]) + float(control["search_wdl_frac"])
    assert blend <= 1.0
    live = LIVE_TRAINER_PIN["kwargs"]
    live_game_frac = 1.0 - float(live["sf_wdl_frac"]) - float(live["search_wdl_frac"])
    assert 1.0 - blend == pytest.approx(live_game_frac), (
        "game_frac must match the LIVE file's, which is "
        f"{live_game_frac:.4f}; that invariance is the justification for "
        "moving the SF share to search rather than picking a new value "
        "recipe. ⚑ Derived from LIVE_TRAINER_PIN, not written as a literal — "
        "a literal here is what let the stale 0.30 survive."
    )


def test_moves_left_normalisation_matches_the_converter(configs: tuple[dict, dict]) -> None:
    """`moves_left` was divided by the converter's 450.0 when the rows were built."""
    _production, control = configs
    assert trainer_kwargs_from_config(control)["moves_left_max_plies"] == 450


def test_no_selfplay_or_search_machinery_is_configured() -> None:
    """The arm is supervised. A selfplay/PID knob here would be a false claim.

    ⚑ `tune:` is no longer forbidden OUTRIGHT, and the reason is a defect this
    blanket ban helped hide: production's replay-buffer knobs live in `tune:`
    and in `selfplay:`, `trainer_kwargs_from_config` reads none of them, and a
    control config with no `tune:` section took `DiskReplayBuffer`'s
    CONSTRUCTOR DEFAULTS — a 20,000-row hot pool against production's 100,000 —
    while every pin passed. So the ban is now on the machinery (`gate_*`,
    `pbt_*`, `distributed_*`, the PID) rather than on the section, and the keys
    that ARE allowed there are exactly the ones `lc0_control_replay` maps.
    """
    raw = load_yaml_file(str(CONTROL))
    assert "stockfish" not in raw
  # Every `tune:` key must be one the replay guard reads. A key that is not is
  # either PBT/distributed machinery this arm does not have, or a buffer knob
  # nobody wired into the guard — both are failures, and this cannot be
  # satisfied by adding a key to the yaml alone.
    replay_attributes = set(lc0_control_replay.CONFIG_KWARGS.values())
    assert set(raw.get("tune", {})) - replay_attributes == set()
    assert set(raw.get("tune", {})), "the replay-buffer keys must be PRESENT"
    forbidden = {
        key for key in raw.get("selfplay", {})
        if key.startswith(("gumbel_", "mcts_", "sf_pid_"))
        or (key.startswith("diff_focus_") and key not in replay_attributes)
    }
    assert forbidden == set()
