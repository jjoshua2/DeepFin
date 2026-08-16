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

⚑⚑ "PRODUCTION" IS TWO DIFFERENT FILES HERE, AND THEY DISAGREE. The live run
reads `configs/pbt2_small.yaml` in the LIVE working tree, on the live branch;
this tree carries `main`'s copy, and `main` has committed nothing to that file
since the live branch diverged. Measured 2026-08-16: the two `model:` sections
differ by the whole bt4heads bundle (`aux_policy_head_dim: 128`,
`categorical_head_coupled: true`, `policy_embedding_mode: linear`) — 61,444,448
trainable params live against 63,084,128 in tree.

⇒ THE REFERENCE DECISION, per axis:

* ARCHITECTURE — judged against `LIVE_ARCH_PIN`, the recorded LIVE `model:`
  section, overlaid on the in-tree production config so every model-affecting
  key OUTSIDE `model:` still comes from a real production yaml. NOT against the
  in-tree `model:` section, which is known-stale — the staleness is itself
  asserted, by
  `test_lc0_control_arch.py::test_the_pin_names_the_bt4heads_keys_the_in_tree_config_lacks`,
  so this file's reference cannot quietly become the wrong one. The pin is
  regenerable from the live file and its freshness has its own gate; the
  verdict here never depends on `$CHESS_LIVE_PRODUCTION_CONFIG`, because a
  guard that answers differently in the operator's shell than in CI is the
  defect, not the fix (review F6).
* TRAINER — still judged against the IN-TREE production config, and the test
  is NAMED for that. ⚑⚑ THIS IS A KNOWN GAP, NOT A CLAIM ABOUT PRODUCTION:
  measured 2026-08-16, the live yaml's `train:` section differs from `main`'s
  in THIRTEEN trainer kwargs (`w_sf_own_regret` 0.7 vs 0.0, `w_categorical`
  0.3 vs 1.0, `sf_wdl_frac` 0.5 vs 0.69, `lr_T0`, `warmup_steps`,
  `sf_target_params.sf_policy_score_mode`, ...), so the control is `main`'s
  committed trainer minus one change, not the live trainer minus one change.
  Closing it means re-pointing the arm's training recipe, which is a
  training-affecting decision that needs its own ledger entry — it is not a
  test fix and is deliberately NOT made here.
"""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pytest

from chess_anti_engine.eval.lc0_control_arch import LIVE_ARCH_PIN, model_section
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


def test_only_the_value_blend_differs_from_the_in_tree_trainer(
    configs: tuple[dict, dict],
) -> None:
    """⚑⚑ NAMED FOR ITS REFERENCE, BECAUSE THE REFERENCE IS THE WEAK PART.

    The architecture axis moved onto the live file's architecture (see
    `test_architecture_is_identical_to_production`); this axis did NOT, and the
    old name — `test_only_the_value_blend_differs_in_the_trainer` — read as a
    claim about production's trainer. It is a claim about `main`'s committed
    one. Measured 2026-08-16, the live `train:` section differs from `main`'s in
    13 trainer kwargs (see this file's header), so the control is one change
    away from the config in this tree and thirteen-plus-one away from the
    trainer production is running.

    Kept as-is deliberately: re-pointing the arm at the live recipe changes what
    the arm trains and needs a ledger entry with a pre-committed yardstick, not
    a test edit.
    """
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
