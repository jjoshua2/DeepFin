"""Selfplay's Gumbel ``c_scale`` is 0.1 and must NOT be unified with play's 0.025.

The two values are separately measured optima for two different sim budgets:

* **Selfplay/RL, 256 sims: 0.1.** 2026-06-16 puzzle screen, n=1000 puzzles —
  0.1 -> 0.688 accuracy, 0.05 -> 0.652, 0.025 -> 0.598. Lowering it HURTS the
  RL regime by ~9 accuracy points.
* **UCI/match, 8000 sims: 0.025.** Measured +301 Elo, adopted for high-sim play.

The optimal q_scale keys off the TOTAL sim budget rather than max_visit, so no
single setting is sim-invariant; per-deployment values are the design, not an
oversight. Until this change the selfplay 0.1 existed only as repeated literals
in five plumbing sites and appeared in no config file, so a refactor that made
``GumbelConfig.c_scale`` agree with ``PLAY_SEARCH_DEFAULTS["c_scale"]`` a few
lines below it would have cost the RL regime those 9 points with no config diff
to notice. These tests fail if that happens.

Assertions are behavioural — resolved config objects and the real validator —
never source text.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import pytest

from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_DEFAULTS,
    SELFPLAY_GUMBEL_C_SCALE,
    GumbelConfig,
)
from chess_anti_engine.selfplay.config import SearchConfig
from chess_anti_engine.tune.trainable_config_ops import _play_batch_kwargs
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils.config_yaml import (
    flatten_run_config_defaults,
    load_yaml_file,
)
from chess_anti_engine.worker import WorkerSession

PRODUCTION_CONFIG = "configs/pbt2_small.yaml"


# ---------------------------------------------------------------------------
# The divergence itself.
# ---------------------------------------------------------------------------


def test_the_selfplay_and_play_scales_are_deliberately_different() -> None:
    """FAILING HERE PROBABLY MEANS SOMEONE "TIDIED" THE TWO INTO AGREEMENT.

    Read the module docstring before changing either number. 0.1 is the
    measured 256-sim selfplay optimum; 0.025 is the measured 8000-sim play
    optimum. Retuning one is a ledger experiment, not a cleanup.
    """
    assert SELFPLAY_GUMBEL_C_SCALE == 0.1
    assert PLAY_SEARCH_DEFAULTS["c_scale"] == 0.025
    assert PLAY_SEARCH_DEFAULTS["c_scale"] != SELFPLAY_GUMBEL_C_SCALE


def test_the_library_default_is_the_selfplay_value_not_the_play_value() -> None:
    """``GumbelConfig()`` is used as "the RL shape by construction".

    ``scripts/audit_targets.py`` builds its training profiles from a bare
    ``GumbelConfig()``, so the dataclass default drifting toward the play value
    would silently price training targets with a search no training row uses.
    """
    assert GumbelConfig().c_scale == SELFPLAY_GUMBEL_C_SCALE
    assert GumbelConfig().c_scale != PLAY_SEARCH_DEFAULTS["c_scale"]


# ---------------------------------------------------------------------------
# Every plumbing default resolves to the one constant.
# ---------------------------------------------------------------------------


def test_the_selfplay_search_config_default_is_the_selfplay_scale() -> None:
    assert SearchConfig().gumbel_c_scale == SELFPLAY_GUMBEL_C_SCALE


def test_the_trial_config_default_is_the_selfplay_scale() -> None:
    assert TrialConfig().gumbel_c_scale == SELFPLAY_GUMBEL_C_SCALE
    assert TrialConfig.from_dict({}).gumbel_c_scale == SELFPLAY_GUMBEL_C_SCALE


def test_a_retuned_scale_still_overrides_the_default() -> None:
    """The pin must not become a hard-code: the yaml stays the operator's lever."""
    assert TrialConfig.from_dict({"gumbel_c_scale": 0.042}).gumbel_c_scale == 0.042


def _bare_session() -> WorkerSession:
    """Minimal ``WorkerSession`` for ``_build_selfplay_configs``.

    That method reads only ``args`` plus the three opening-asset paths; a full
    session would need a broker, a model and a Stockfish binary.
    """
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.selfplay_gumbel_c_scale")
    session.args = SimpleNamespace()
    session.opening_book_path = None
    session.opening_book_path_2 = None
    session.opening_fen_list_path = None
    session._dole_lock = threading.Lock()
    return session


def test_a_manifest_without_the_key_still_gives_the_worker_the_selfplay_scale() -> None:
    """The distributed worker's fallback, exercised through the real builder."""
    cfgs, _sf_args = WorkerSession._build_selfplay_configs(
        _bare_session(), {"sf_nodes": 5000},
    )

    assert cfgs["search"].gumbel_c_scale == SELFPLAY_GUMBEL_C_SCALE


def test_a_published_scale_reaches_the_worker_search_config() -> None:
    cfgs, _sf_args = WorkerSession._build_selfplay_configs(
        _bare_session(), {"sf_nodes": 5000, "gumbel_c_scale": 0.042},
    )

    assert cfgs["search"].gumbel_c_scale == 0.042


def test_a_change_to_the_scale_restarts_the_worker_selfplay_session() -> None:
    """A live-applied c_scale would leave workers on the old search silently."""
    assert "gumbel_c_scale" in WorkerSession._RECO_RESTART_KEYS


# ---------------------------------------------------------------------------
# The yaml pin: validator behaviour, not source text.
# ---------------------------------------------------------------------------


def test_the_key_survives_validation_and_flattening_in_the_selfplay_section() -> None:
    """Assert the validator's BEHAVIOUR, not that the key appears in a list.

    ``gumbel_c_scale`` lives in the ``selfplay:`` section, so it has to be in
    ``_SELFPLAY_KEYS`` specifically — the all-or-nothing validator rejects the
    WHOLE reload on one unknown key, and a test that merely greps config_yaml.py
    would stay green with the key sitting in the wrong allowlist.
    """
    flat = flatten_run_config_defaults({"selfplay": {"gumbel_c_scale": 0.042}})

    assert flat["gumbel_c_scale"] == 0.042


def test_an_unknown_selfplay_key_is_still_rejected() -> None:
    """The test above must not pass by the validator having gone permissive."""
    with pytest.raises(ValueError, match="Unknown keys in yaml 'selfplay:'"):
        flatten_run_config_defaults({"selfplay": {"definitely_not_a_real_key": 1}})


def test_the_shipped_production_config_pins_the_selfplay_scale() -> None:
    """The whole point: the value is in the config file, not only in code."""
    flat = flatten_run_config_defaults(load_yaml_file(PRODUCTION_CONFIG))

    assert flat["gumbel_c_scale"] == SELFPLAY_GUMBEL_C_SCALE


def test_the_shipped_production_config_round_trips_into_the_search_config() -> None:
    """End to end: yaml -> validator -> TrialConfig -> the search selfplay runs."""
    flat = flatten_run_config_defaults(load_yaml_file(PRODUCTION_CONFIG))
    tc = TrialConfig.from_dict(flat)
    search = _play_batch_kwargs(tc)["search"]

    assert search.gumbel_c_scale == SELFPLAY_GUMBEL_C_SCALE
    assert search.gumbel_c_scale != PLAY_SEARCH_DEFAULTS["c_scale"]
