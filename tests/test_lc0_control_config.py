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

⚑⚑ AND IT IS STILL BLIND, WHICH IS WHY IT IS NOT THE ONLY INSTRUMENT.
`configs/pbt2_small.yaml` *in this tree* is not the file the live run reads —
that one lives in the live working tree, on the live branch, and moved to the
bt4heads bundle on 2026-08-15 while this branch did not. Everything here can
pass with a control that trains an architecture production no longer runs.
`tests/test_lc0_control_arch.py` owns that question and judges against the
LIVE file; read the two together.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from chess_anti_engine.model import model_config_from_flat_config
from chess_anti_engine.train.trainer import trainer_kwargs_from_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

REPO = Path(__file__).resolve().parent.parent
PRODUCTION = REPO / "configs" / "pbt2_small.yaml"
CONTROL = REPO / "configs" / "lc0_positive_control.yaml"

# The ONLY trainer kwargs allowed to differ, and why:
#   sf_wdl_frac     lc0 rows carry no SF label; losses.py would silently put
#                   this share on the raw game outcome (value_blend_guard).
#   search_wdl_frac the freed share, re-pointed at lc0's own best_q/best_d so
#                   game_frac stays at production's 0.30.
ALLOWED_TRAINER_DIFFS = {"sf_wdl_frac", "search_wdl_frac"}


@pytest.fixture(scope="module")
def configs() -> tuple[dict, dict]:
    return (
        flatten_run_config_defaults(load_yaml_file(str(PRODUCTION))),
        flatten_run_config_defaults(load_yaml_file(str(CONTROL))),
    )


def test_control_config_loads_through_the_production_schema() -> None:
    """Category (a) from CLAUDE.md: an unknown key is FATAL AT LAUNCH.

    `flatten_run_config_defaults` is what `run.py` calls before its argument
    parser exists and outside any try, so a key that is not in
    `utils/config_yaml.py`'s schema means the process never starts. Loading the
    file here is the same check, run before anyone spends GPU time on it.
    """
    flat = flatten_run_config_defaults(load_yaml_file(str(CONTROL)))
    assert flat["model"] == "transformer"


def test_architecture_is_identical_to_production(configs: tuple[dict, dict]) -> None:
    production, control = configs
    prod_model = model_config_from_flat_config(production)
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


def test_only_the_value_blend_differs_in_the_trainer(configs: tuple[dict, dict]) -> None:
    production, control = configs
    prod_kwargs = trainer_kwargs_from_config(production)
    ctrl_kwargs = trainer_kwargs_from_config(control)
    differing = {
        key
        for key in set(prod_kwargs) | set(ctrl_kwargs)
        if prod_kwargs.get(key) != ctrl_kwargs.get(key)
    }
    assert differing == ALLOWED_TRAINER_DIFFS, (
        f"trainer kwargs differ in {sorted(differing)}, expected only "
        f"{sorted(ALLOWED_TRAINER_DIFFS)}. Unexpected extras are usually a "
        "production key the control forgot to copy; unexpected absences mean "
        "production moved onto the control's value."
    )


def test_optimizer_is_the_production_one(configs: tuple[dict, dict]) -> None:
    """Stated separately because it is the claim people actually check."""
    _production, control = configs
    kwargs = trainer_kwargs_from_config(control)
    assert kwargs["optimizer"] == "aurora"
    assert kwargs["matrix_optimizer_scope"] == "mlp_out"


def test_the_value_blend_override_is_present_and_total(configs: tuple[dict, dict]) -> None:
    """Both halves of the override, and the share it was handed to."""
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
    assert 1.0 - blend == pytest.approx(0.30), (
        "game_frac must match production's 0.30; that invariance is the "
        "justification for moving the SF share to search rather than picking "
        "a new value recipe."
    )


def test_moves_left_normalisation_matches_the_converter(configs: tuple[dict, dict]) -> None:
    """`moves_left` was divided by the converter's 450.0 when the rows were built."""
    _production, control = configs
    assert trainer_kwargs_from_config(control)["moves_left_max_plies"] == 450


def test_no_selfplay_or_search_machinery_is_configured() -> None:
    """The arm is supervised. A selfplay/PID knob here would be a false claim."""
    raw = load_yaml_file(str(CONTROL))
    assert "stockfish" not in raw
    assert "tune" not in raw
    forbidden = {
        key for key in raw.get("selfplay", {})
        if key.startswith(("gumbel_", "mcts_", "sf_pid_", "diff_focus_"))
    }
    assert forbidden == set()
