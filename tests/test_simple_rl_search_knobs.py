"""Every search/target knob the simple RL loop passes through changes the
STORED target on the compiled (C) path, reached through the production
SearchConfig -> GumbelConfig mapping; and the stored target is reproduced from
the exported root statistics with the search's own formula.

This file exists because a value accepted and then silently ignored is the
repo's signature defect: each knob is OBSERVED to change the row, not read off
source text.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel_c import rebuild_improved_policy, run_gumbel_root_many_c
from chess_anti_engine.selfplay.config import GameConfig, SearchConfig
from chess_anti_engine.selfplay.network_turn import build_selfplay_gumbel_config, build_selfplay_search_shape
from chess_anti_engine.simple_rl.collect import CollectSettings, build_configs
from tests.simple_rl_helpers import tiny_model

_FENS = [
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/pp2bppp/2n1bn2/3p4/3P4/2NBBN2/PP3PPP/R2Q1RK1 w - - 0 11",
    "8/5pk1/6p1/3R4/5P2/6P1/r5K1/8 b - - 1 40",
]
_BASE = SearchConfig(simulations=24, fast_simulations=24, playout_cap_fraction=1.0, gumbel_topk=16,
                     gumbel_c_scale=0.1, gumbel_vloss_weight=1)
_MODEL: dict[str, Any] = {}


def _model() -> Any:
    if "m" not in _MODEL:
        _MODEL["m"] = tiny_model(seed=3)
    return _MODEL["m"]


def _search(search: SearchConfig) -> tuple[list[np.ndarray], list[Any], Any]:
    shape = build_selfplay_search_shape(search=search, game=GameConfig(), simulations=int(search.simulations))
    boards = [chess.Board(f) for f in _FENS]
    stats: list[Any] = []
    out = run_gumbel_root_many_c(
        _model(), boards, device="cpu", rng=np.random.default_rng(11),
        per_game_add_noise=[False] * len(boards), root_stats_out=stats, **shape.runner_kwargs(),
    )
    return list(out[0]), stats, shape.cfg


KNOBS = {
    "gumbel_c_scale": 0.4,
    "gumbel_c_visit": 4.0,
    "gumbel_target_q_rescale": False,
    "gumbel_policy_temp": 1.6,
    "gumbel_topk": 4,
}


def test_the_baseline_is_deterministic() -> None:
    a, _, _ = _search(_BASE)
    b, _, _ = _search(_BASE)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("knob", sorted(KNOBS))
def test_each_knob_changes_the_stored_target(knob: str) -> None:
    base, _, base_cfg = _search(_BASE)
    probs, _, cfg = _search(replace(_BASE, **{knob: KNOBS[knob]}))
    assert base_cfg != cfg, f"{knob} did not reach the GumbelConfig"
    moved = [float(np.abs(p - b).max()) for p, b in zip(probs, base, strict=True)]
    assert max(moved) > 1e-4, f"{knob}={KNOBS[knob]} left every stored target unchanged: {moved}"


@pytest.mark.parametrize("knob", ["baseline", *sorted(KNOBS), "untempered"])
def test_stored_target_is_rebuilt_from_root_stats(knob: str) -> None:
    if knob == "baseline":
        search = _BASE
    elif knob == "untempered":
        search = replace(_BASE, gumbel_policy_temp=1.5, gumbel_target_untempered_prior=True)
    else:
        search = replace(_BASE, **{knob: KNOBS[knob]})
    probs, stats, cfg = _search(search)
    assert all(s is not None for s in stats)
    for p, s in zip(probs, stats, strict=True):
        rebuilt = rebuild_improved_policy(log_prior=s.log_prior, completed_q=s.completed_q, visits=s.visits,
                                          net_value=s.net_value, cfg=cfg)
        np.testing.assert_allclose(p[s.legal], rebuilt, atol=1e-6)
        assert float(p.sum()) == pytest.approx(float(p[s.legal].sum()), abs=1e-6)
        assert int(s.survivor) in {int(a) for a in s.legal}
        j = int(np.searchsorted(s.legal, s.survivor))
        assert s.survivor_q == s.completed_q[j]
        assert s.net_value == pytest.approx(float(s.net_wdl[0] - s.net_wdl[2]), abs=1e-6)
        assert s.start_visits == 0  # no tree handed in: every root is fresh


def test_root_stats_are_off_by_default() -> None:
    assert GameConfig().record_root_stats is False
    shape = build_selfplay_search_shape(search=_BASE, game=GameConfig(), simulations=24)
    out = run_gumbel_root_many_c(_model(), [chess.Board()], device="cpu", rng=np.random.default_rng(1),
                                 per_game_add_noise=[False], **shape.runner_kwargs())
    assert len(out) == 6  # no extra output appears for existing callers


def test_collector_settings_reach_the_search_config(tmp_path) -> None:
    settings = CollectSettings(games=1, sims=12, topk=5, c_scale=0.3, c_visit=7.0, target_q_rescale=False,
                               policy_temp=1.3, target_untempered_prior=True, vloss_weight=2,
                               temp_start=0.7, temp_floor=0.05, temp_decay_plies=6,
                               root_noise_scale=0.8, root_noise_after=0.2, root_noise_decay_start_move=9,
                               root_noise_decay_moves=4)
    arch = {"input_history_encoding": "lc0_root_legacy_meta", "input_extra_features": "v2_threats",
            "policy_encoding": "lc0_1858", "history_rep_fix": True}
    cfg = build_configs(settings, arch, tmp_path / "starts.txt")
    gumbel = build_selfplay_gumbel_config(search=cfg["search"], game=cfg["game"], simulations=cfg["search"].simulations)
    assert (gumbel.simulations, gumbel.topk, gumbel.c_scale, gumbel.c_visit, gumbel.target_q_rescale,
            gumbel.policy_temp, gumbel.target_untempered_prior) == (12, 5, 0.3, 7.0, False, 1.3, True)
    s, t, g = cfg["search"], cfg["temp"], cfg["game"]
    assert (s.gumbel_vloss_weight, s.playout_cap_fraction, s.fast_simulations) == (2, 1.0, 12)
    assert (s.gumbel_scale, s.gumbel_scale_after, s.gumbel_scale_decay_start_move, s.gumbel_scale_decay_moves) == (0.8, 0.2, 9, 4)
    assert (t.selfplay_temperature, t.selfplay_endgame, t.selfplay_decay_moves, t.schedule_plies_from_start) == (0.7, 0.05, 6, True)
    assert (g.selfplay_fraction, g.record_root_stats, g.policy_row_retention, g.history_rep_fix,
            g.input_history_encoding) == (1.0, True, "all_eligible", True, "lc0_root_legacy_meta")


def test_target_q_rescale_moves_the_target_but_not_the_survivor() -> None:
    """Coordinator calibration: same search, target_q_rescale off moves the
    stored target (mean TV ~0.09 there) and leaves the survivor unchanged."""
    base, base_stats, _ = _search(_BASE)
    off, off_stats, _ = _search(replace(_BASE, gumbel_target_q_rescale=False))
    tv = [0.5 * float(np.abs(a - b).sum()) for a, b in zip(base, off, strict=True)]
    assert float(np.mean(tv)) > 0.01, tv
    assert [s.survivor for s in base_stats] == [s.survivor for s in off_stats]
    for a, b in zip(base_stats, off_stats, strict=True):
        np.testing.assert_array_equal(a.visits, b.visits)  # store-arm only: the search is identical
