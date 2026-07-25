"""The target audit must score training targets with the TRAINING search.

`scripts/audit_targets.py` exists to price training-target candidates against
the frozen deep-SF audit set, and its headline puts the net's search regret
next to the SF MultiPV soft target to decide whether SF's CPU bill is still
worth paying. Until 2026-07-25 it built ONE search from the UCI/TCEC
`PLAY_SEARCH_DEFAULTS` and labelled the resulting row "production training
target" — so the number driving that decision came from a search no training
row is ever built with.

The two searches are deliberately different and separately tuned: RL selfplay
keeps c_scale 0.1 with the legacy LINEAR root, UCI play uses 0.025 with the
LOG root, and at the 256-sim selfplay budget the RL value measured 0.688
puzzle accuracy against 0.598 for the play value. These tests pin that the
profiles stay distinct and that the training ones follow the live yaml.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS, GumbelConfig


def _load_audit_targets() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_targets.py"
    spec = importlib.util.spec_from_file_location("audit_targets_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves __module__ through sys.modules, so a module loaded
    # by path has to be registered before exec or @dataclass raises.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit_targets() -> ModuleType:
    return _load_audit_targets()


@pytest.fixture
def flat() -> dict[str, object]:
    """A config shaped like the production one, with distinctive values."""
    return {
        "gumbel_c_scale": 0.1,
        "gumbel_topk": 16,
        "mcts_simulations": 256,
        "fast_simulations": 32,
        "playout_cap_fraction": 0.25,
    }


def test_the_training_profile_uses_the_rl_c_scale_not_the_play_one(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    profiles = audit_targets.build_search_profiles(flat, play_sims=256, play_topk=16)

    assert profiles["train"].c_scale == 0.1
    assert profiles["train_fast"].c_scale == 0.1
    # The bug this pins: the training rows silently carrying the play value.
    assert profiles["train"].c_scale != PLAY_SEARCH_DEFAULTS["c_scale"]


def test_the_training_profile_keeps_the_legacy_linear_root(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    """Training/RL is bit-identical to the legacy root transform (PR #84).

    The sentinels are what select it, so assert against a real GumbelConfig
    rather than against the literals -- if the library ever changes which
    values mean "legacy", this test fails instead of silently passing.
    """
    legacy = GumbelConfig()
    for name in ("train", "train_fast"):
        prof = audit_targets.build_search_profiles(
            flat, play_sims=256, play_topk=16,
        )[name]
        assert prof.c_visit_root == legacy.c_visit_root
        assert prof.c_scale_root == legacy.c_scale_root
        assert prof.q_visit_exp_root == legacy.q_visit_exp_root
        assert prof.c_scale == legacy.c_scale


def test_the_play_profile_keeps_the_log_root(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    play = audit_targets.build_search_profiles(
        flat, play_sims=256, play_topk=16,
    )["search"]

    assert play.c_scale == PLAY_SEARCH_DEFAULTS["c_scale"]
    assert play.c_scale_root == PLAY_SEARCH_DEFAULTS["c_scale_root"]
    # q_visit_exp_root < 0 is the LOG-root sentinel.
    assert play.q_visit_exp_root < 0


def test_the_two_search_shapes_are_not_the_same(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    profiles = audit_targets.build_search_profiles(flat, play_sims=256, play_topk=16)

    assert profiles["search"] != profiles["train"]


def test_the_training_sims_come_from_the_config_not_the_cli(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    """--sims retunes the PLAY row only; the training rows track the yaml.

    Scoring the stored target at a budget selfplay never runs would recreate
    the same class of error in a new place.
    """
    flat["mcts_simulations"] = 128
    flat["fast_simulations"] = 8
    profiles = audit_targets.build_search_profiles(flat, play_sims=999, play_topk=16)

    assert profiles["search"].sims == 999
    assert profiles["train"].sims == 128
    assert profiles["train_fast"].sims == 8


def test_a_retuned_rl_c_scale_reaches_the_training_profile(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    flat["gumbel_c_scale"] = 0.042
    profiles = audit_targets.build_search_profiles(flat, play_sims=256, play_topk=16)

    assert profiles["train"].c_scale == 0.042
    assert profiles["train_fast"].c_scale == 0.042


def test_the_fast_profile_is_scored_at_the_fast_budget(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    profiles = audit_targets.build_search_profiles(flat, play_sims=256, play_topk=16)

    assert profiles["train"].sims != profiles["train_fast"].sims
    assert {"train", "train_fast"} <= set(profiles)


def test_the_fast_row_is_not_labelled_a_production_policy_target(
    audit_targets: ModuleType,
) -> None:
    """Playout-capped plies carry NO policy target, so (e) must not claim to.

    `finalize.py` drops fast rows outright by default, and with
    `record_fast_ply_value` they become value-only rows whose MAIN policy head
    is masked -- KataGo's playout-cap design, where cheap plies buy game length
    and value coverage but never policy supervision. A first pass at this fix
    blended (d) and (e) by playout_cap_fraction into the headline, which
    invented a corpus nothing stores and understated the target by ~9cp: the
    very "two incomparable scales" error the rest of this file exists to stop.
    """
    fast_label = audit_targets._CANDIDATE_NAMES["train_fast"]

    assert "NOT a policy target" in fast_label
    assert "production training target" not in fast_label


def test_every_profile_is_named_in_the_report_legend(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    """A profile with no candidate name would be scored but never printed."""
    profiles = audit_targets.build_search_profiles(flat, play_sims=256, play_topk=16)

    assert set(profiles) <= set(audit_targets._CANDIDATE_NAMES)


def test_the_candidate_legend_no_longer_calls_the_play_row_the_training_target(
    audit_targets: ModuleType,
) -> None:
    names = audit_targets._CANDIDATE_NAMES

    assert "training target" not in names["search"]
    assert "training target" in names["train"]


# --- Review findings on PR #246 ------------------------------------------


def test_the_play_profile_carries_the_descent_knobs_too(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    """c_puct / cpuct_factor / fpu_reduction differ between PLAY and training.

    Taking only the root-transform subset of PLAY_SEARCH_DEFAULTS left row (b)
    a hybrid: play root transform, training descent. Those fields act on tree
    descent, so it measured a search neither path runs.
    """
    play = audit_targets.build_search_profiles(
        flat, play_sims=256, play_topk=None,
    )["search"]

    assert play.c_puct == PLAY_SEARCH_DEFAULTS["c_puct"]
    assert play.cpuct_factor == PLAY_SEARCH_DEFAULTS["cpuct_factor"]
    assert play.cpuct_base == PLAY_SEARCH_DEFAULTS["cpuct_base"]
    assert play.fpu_reduction == PLAY_SEARCH_DEFAULTS["fpu_reduction"]


def test_the_play_profile_defaults_to_the_play_topk(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    profiles = audit_targets.build_search_profiles(
        flat, play_sims=256, play_topk=None,
    )

    assert profiles["search"].topk == PLAY_SEARCH_DEFAULTS["topk"]
    # ...while the training rows keep selfplay's own value from the config.
    assert profiles["train"].topk == 16


def test_the_topk_override_does_not_reach_the_training_rows(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    """--gumbel-topk retunes the PLAY row only.

    Overriding the training target's own topk would score a search selfplay
    never runs -- the same error as scoring it with play c_scale.
    """
    profiles = audit_targets.build_search_profiles(
        flat, play_sims=256, play_topk=48,
    )

    assert profiles["search"].topk == 48
    assert profiles["train"].topk == 16
    assert profiles["train_fast"].topk == 16


def test_the_training_profiles_carry_volatility_search_settings(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    """Volatility search is default-off but audit-first still has to judge it.

    Without propagation the audit runs the baseline search and reports it as
    the configured training target, so the pre-training gate is structurally
    unable to see the one flag family it was asked about.
    """
    flat["volatility_q_scale"] = 0.5
    flat["volatility_fpu"] = 0.25
    profiles = audit_targets.build_search_profiles(
        flat, play_sims=256, play_topk=None,
    )

    assert profiles["train"].volatility_q_scale == 0.5
    assert profiles["train"].volatility_fpu == 0.25
    assert profiles["train_fast"].volatility_q_scale == 0.5


def test_volatility_is_off_by_default(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    profiles = audit_targets.build_search_profiles(
        flat, play_sims=256, play_topk=None,
    )

    assert profiles["train"].volatility_q_scale == 0.0
    assert profiles["train"].volatility_fpu == 0.0


# --- Search WDL must be rebuilt the way selfplay stores it ---------------


def test_search_wdl_preserves_the_root_networks_draw_mass(
    audit_targets: ModuleType,
) -> None:
    """`network_turn.py` keeps d_raw and splits only the remaining mass.

    `_q_to_wdl` invents D = 1 - |q|, a different distribution whenever the net
    predicts any other draw mass -- i.e. almost always.
    """
    net_wdl = np.array([0.25, 0.60, 0.15])
    wdl = audit_targets._search_wdl_like_selfplay(0.2, net_wdl)

    assert wdl[1] == pytest.approx(0.60)
    rem = 1.0 - 0.60
    assert wdl[0] == pytest.approx(0.5 * (rem + 0.2))
    assert wdl[2] == pytest.approx(rem - wdl[0])
    assert wdl.sum() == pytest.approx(1.0)
    # The old formula would have said D = 1 - 0.2 = 0.8.
    assert wdl[1] != pytest.approx(audit_targets._q_to_wdl(0.2)[1])


def test_search_wdl_clamps_q_into_the_non_draw_mass(
    audit_targets: ModuleType,
) -> None:
    """A high-confidence Q against a large draw mass must not go negative."""
    net_wdl = np.array([0.05, 0.90, 0.05])
    wdl = audit_targets._search_wdl_like_selfplay(0.99, net_wdl)

    assert min(wdl) >= 0.0
    assert wdl.sum() == pytest.approx(1.0)
    assert wdl[1] == pytest.approx(0.90)


def test_search_wdl_falls_back_to_a_draw_on_a_non_finite_net_wdl(
    audit_targets: ModuleType,
) -> None:
    wdl = audit_targets._search_wdl_like_selfplay(
        0.2, np.array([np.nan, np.nan, np.nan]),
    )

    assert wdl.tolist() == [0.0, 1.0, 0.0]


def test_the_value_candidate_uses_the_selfplay_wdl_reconstruction(
    audit_targets: ModuleType,
) -> None:
    """Having the right formula is worthless if the call site skips it.

    Sabotaging the call site back to `_q_to_wdl` left every direct test of
    `_search_wdl_like_selfplay` passing, so this pins the wiring itself.
    """
    import inspect

    src = inspect.getsource(audit_targets.main)

    assert "_search_wdl_like_selfplay(root_q[i], root_wdl[i])" in src
    assert "search_root = _q_to_wdl(" not in src


def test_the_search_wdl_is_built_from_the_rl_root_not_the_play_root(
    audit_targets: ModuleType,
) -> None:
    """The blend's search component comes from the RL search."""
    import inspect

    src = inspect.getsource(audit_targets.main)

    assert 'root_q = root_q_by_profile["train"]' in src
