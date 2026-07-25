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


def test_the_playout_cap_split_is_represented_by_two_profiles(
    audit_targets: ModuleType, flat: dict[str, object],
) -> None:
    """Production stores a MIXTURE: 25% full-sim rows, 75% fast-sim rows.

    Scoring only the full-sim search would describe a quarter of the corpus
    and call it the training target.
    """
    profiles = audit_targets.build_search_profiles(flat, play_sims=256, play_topk=16)

    assert profiles["train"].sims != profiles["train_fast"].sims
    assert {"train", "train_fast"} <= set(profiles)


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
