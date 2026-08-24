"""The control must train with the RECIPE PRODUCTION IS RUNNING.

⚑⚑ THE SECOND HALF OF "OUR EXACT NET AND TRAINER", AND IT WAS THE UNCHECKED
HALF. `tests/test_lc0_control_arch.py` moved the ARCHITECTURE reference off the
in-tree `configs/pbt2_small.yaml` on 2026-08-16. The TRAINER reference stayed on
it for the rest of that day, `main` being 13 `trainer_kwargs_from_config`
entries behind the live file, and the arm's own config test SAID SO in its
docstring while continuing to assert against the stale copy. Naming a gap is
not closing one.

The consequence was load-bearing, not cosmetic: the control's
`search_wdl_frac: 0.70` was justified as "keeps `game_frac` at production's
0.30", and the live blend is `sf 0.69 + search 0.31 = 1.00`, i.e. `game_frac`
**0.00**. A value chosen to preserve a number production does not have.

So this file tests the other instrument: `LIVE_TRAINER_PIN`, the drift check,
and the fact that the check prefers the live file over any in-tree copy — the
same three propositions `test_lc0_control_arch.py` tests one axis over.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.eval.lc0_control_arch import (
    LIVE_CONFIG_ENV,
    live_production_config_path,
)
from chess_anti_engine.eval.lc0_control_trainer import (
    LC0_TRAINER_DEVIATIONS,
    LIVE_TRAINER_PIN,
    ControlTrainerDrift,
    assert_control_matches_live_trainer,
    trainer_kwargs_drift,
    trainer_kwargs_signature,
)
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

REPO = Path(__file__).resolve().parent.parent
PRODUCTION = REPO / "configs" / "pbt2_small.yaml"
CONTROL = REPO / "configs" / "lc0_positive_control.yaml"


def _flat(path: Path) -> dict[str, Any]:
    return flatten_run_config_defaults(load_yaml_file(str(path)))


def test_the_recorded_pin_still_matches_the_live_file_when_one_is_named() -> None:
    """⚑⚑ THE PIN IS A COMMITTED COPY TOO — THIS IS WHAT KEEPS IT HONEST.

    Same construction, and same limitation, as the architecture pin's freshness
    test: it can only run where the live file exists, so when it cannot it says
    what was NOT checked rather than reporting a pass. And it moves no other
    test's verdict — every other gate here judges against the pin whether or
    not `$CHESS_LIVE_PRODUCTION_CONFIG` is set (review F6).
    """
    live = live_production_config_path()
    if live is None:
        pytest.skip(
            f"${LIVE_CONFIG_ENV} is not set — LIVE_TRAINER_PIN's freshness "
            "against the LIVE production yaml was NOT checked by this run",
        )
    realized = trainer_kwargs_signature(_flat(live))
    pinned = dict(LIVE_TRAINER_PIN["kwargs"])
    drift = [
        f"{key}: live={realized.get(key, '<absent>')!r} "
        f"pin={pinned.get(key, '<absent>')!r}"
        for key in sorted(set(realized) | set(pinned))
        if realized.get(key, "<absent>") != pinned.get(key, "<absent>")
    ]
    assert drift == [], (
        f"LIVE_TRAINER_PIN is STALE against {live}:\n  " + "\n  ".join(drift) + "\n"
        "Regenerate it with scripts/lc0_control_arch_pin.py --axis trainer "
        "--live-config <live yaml>, then decide whether the arm follows "
        "production. ⚑ That decision is training-affecting and needs a ledger "
        "entry, not a pin edit."
    )


def test_the_pin_records_a_recipe_the_in_tree_config_does_not(
) -> None:
    """⚑ The staleness the whole module rests on, ASSERTED rather than assumed.

    Judging against a pin instead of the yaml sitting next to it is only
    justified while the two disagree. Mirrors
    `test_lc0_control_arch.py::test_the_pin_names_the_bt4heads_keys_the_in_tree_config_lacks`:
    the situation improving must BREAK this, so nobody discovers it silently.
    """
    drift = trainer_kwargs_drift(
        trainer_kwargs_signature(_flat(PRODUCTION)), LIVE_TRAINER_PIN["kwargs"],
    )
  # The bt4heads bundle's trainer half is the load-bearing part of the gap:
  # `main` trains the categorical head at 0.0 and live at 1.0, with the rebuild
  # and its 0.69/0.31 blend on. Named individually so a partial catch-up is
  # visible instead of being absorbed by a bare "the sets differ".
    for key in ("w_categorical", "rebuild_categorical_target", "warmup_steps"):
        assert key in drift, (
            f"the in-tree production config now realizes LIVE's {key!r}. If "
            "`main` has caught up on every kwarg, this module's reference "
            "decision should be re-read rather than the assertion relaxed."
        )


def test_the_shipped_control_matches_the_pin_apart_from_the_declared_deviations() -> None:
    """The gate must be capable of PASSING, or it is a constant, not a check."""
    provenance = assert_control_matches_live_trainer(_flat(CONTROL), context="unit")
    assert "recorded trainer pin" in provenance
    assert "THE LIVE FILE WAS NOT READ" in provenance


def test_a_control_matching_the_stale_in_tree_trainer_is_refused() -> None:
    """⚑⚑ THE POINT OF THE MODULE, AS A REGRESSION.

    The control as shipped BEFORE 2026-08-16 was `main`'s trainer minus the
    value blend, and every test then in the tree passed on it. Reconstruct that
    config and require the guard to refuse it — by name, on the kwargs that
    differ — so "we re-pointed the trainer" cannot silently un-happen.
    """
    stale = _flat(CONTROL)
    stale.update({
        "warmup_steps": 72, "lr_T0": 999999, "lr_T_mult": 1,
        "w_sf_move": 0.02, "w_sf_own": 0.1, "w_sf_own_regret": 0.7,
        "w_sf_eval": 0.1, "w_categorical": 0.3,
        "rebuild_categorical_target": False,
        "categorical_blend_frac": 0.0, "categorical_search_blend_frac": 0.0,
        "sf_policy_score_mode": "wdl",
        "search_wdl_frac": 0.7,
    })
    with pytest.raises(ControlTrainerDrift) as exc:
        assert_control_matches_live_trainer(stale, context="unit")
    message = str(exc.value)
    for key in (
        "warmup_steps", "lr_T0", "lr_T_mult", "w_sf_move", "w_sf_own",
        "w_sf_own_regret", "w_sf_eval", "w_categorical",
        "rebuild_categorical_target", "categorical_target_params",
        "sf_target_params",
    ):
        assert key in message, f"{key} drifted but the refusal did not name it"


def test_a_caller_supplied_pin_is_the_one_judged_against() -> None:
    """⚑ REVIEW F4's defect, pre-empted on this axis: `pin=` accepted then
    ignored would make every caller-supplied reference decorative."""
    pin = {
        **LIVE_TRAINER_PIN,
        "kwargs": {**LIVE_TRAINER_PIN["kwargs"], "optimizer": "nadamw"},
    }
    with pytest.raises(ControlTrainerDrift, match="optimizer"):
        assert_control_matches_live_trainer(_flat(CONTROL), pin=pin, context="unit")


def test_an_empty_reference_is_refused_not_agreed_with() -> None:
    """An empty signature agrees with every control. Refuse it."""
    with pytest.raises(ControlTrainerDrift, match="EMPTY"):
        assert_control_matches_live_trainer(
            _flat(CONTROL), pin={**LIVE_TRAINER_PIN, "kwargs": {}}, context="unit",
        )


def test_the_live_file_beats_the_pin_when_both_are_available(tmp_path: Path) -> None:
    """⚑ The whole reason the module exists: the reference is a file OUTSIDE
    this tree. A control that matches the committed pin exactly must still fail
    against a live file that disagrees."""
    live = tmp_path / "live_pbt2_small.yaml"
    raw = copy.deepcopy(load_yaml_file(str(CONTROL)))
    raw["train"]["lr"] = 0.0001
    raw["train"]["sf_wdl_frac"] = float(LIVE_TRAINER_PIN["kwargs"]["sf_wdl_frac"])
    raw["train"]["search_wdl_frac"] = float(
        LIVE_TRAINER_PIN["kwargs"]["search_wdl_frac"],
    )
    live.write_text(
        __import__("yaml").safe_dump(raw), encoding="utf-8",
    )
    assert_control_matches_live_trainer(_flat(CONTROL), context="pin mode")
    with pytest.raises(ControlTrainerDrift, match="lr"):
        assert_control_matches_live_trainer(
            _flat(CONTROL), live_config=live, context="live mode",
        )


def test_the_signature_expands_dataclass_kwargs_by_field() -> None:
    """`sf_target_params` / `categorical_target_params` must diff by FIELD.

    ⚑ A repr comparison would report a whole dataclass as one opaque string, so
    a one-field change (`sf_policy_score_mode` wdl -> cp — one of the original
    thirteen) would be unreadable in the refusal.
    """
    signature = trainer_kwargs_signature(_flat(CONTROL))
    assert signature["sf_target_params"]["sf_policy_score_mode"] == "cp"
    assert signature["categorical_target_params"]["blend_frac"] == pytest.approx(0.69)


def test_the_dead_cosine_knobs_are_pinned_at_the_realized_defaults() -> None:
    """⚑ NEITHER CONFIG SETS `lr_T0`/`lr_T_mult`, AND BOTH REALIZE 5000/2.

    They are read only by the `CosineAnnealingWarmRestarts` branch, which
    `lr_schedule: sqrt_release` never constructs — the live file deletes them
    with exactly that comment, and the control now does too. The pin carries
    the DEFAULTS both sides realize rather than omitting the keys, because
    `trainer_kwargs_signature` asks what `Trainer(...)` is constructed with, and
    a knob that is dead today is one `lr_schedule` edit from being live.
    """
    control = trainer_kwargs_signature(_flat(CONTROL))
    assert control["lr_schedule"] == "sqrt_release"
    assert control["lr_T0"] == LIVE_TRAINER_PIN["kwargs"]["lr_T0"] == 5000
    assert control["lr_T_mult"] == LIVE_TRAINER_PIN["kwargs"]["lr_T_mult"] == 2
    raw = load_yaml_file(str(CONTROL))
    assert "lr_T0" not in raw["train"], (
        "the control sets lr_T0 explicitly again. The live file deletes it; "
        "carrying a value here reintroduces a live-vs-control diff in a knob "
        "neither schedule reads."
    )


def test_every_declared_deviation_is_a_real_trainer_kwarg() -> None:
    """An allow-list entry that names nothing excuses nothing — and would never
    be noticed, because a key absent from both sides never appears in a diff."""
    reference = LIVE_TRAINER_PIN["kwargs"]
    unknown = sorted(set(LC0_TRAINER_DEVIATIONS) - set(reference))
    assert unknown == [], (
        f"LC0_TRAINER_DEVIATIONS names {unknown}, which are not "
        "trainer_kwargs_from_config keys at all."
    )
