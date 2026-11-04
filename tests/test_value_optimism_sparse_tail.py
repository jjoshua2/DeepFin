"""Undefined tail means must not become observations in a clustered CI."""
from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.eval.value_optimism import OptimismRows, tail_asymmetry_ci


def _rows(cp: list[float], games: list[int], differences: list[float]) -> OptimismRows:
    ruler = np.full(len(cp), 0.5)
    return OptimismRows(
        sf_cp=np.array(cp), sf_ruler_score=ruler,
        net_score=ruler + np.array(differences), target_score=ruler,
        outcome_score=ruler, target_sf_score=ruler, search_score=ruler,
        game_id=np.array(games), piece_count=np.full(len(cp), 32),
    )


@pytest.mark.parametrize(("cp", "games"), [
    ([-500.0, -500.0, 500.0], [0, 1, 1]),  # only the high tail can be empty
    ([-500.0, 500.0, 500.0], [0, 0, 1]),  # only the low tail can be empty
    ([-500.0, 500.0], [0, 1]),  # either tail can be empty
    ([-500.0, 500.0, 0.0], [0, 1, 2]),  # both tails can be empty
])
def test_tail_ci_refuses_undefined_resamples(cp: list[float], games: list[int]) -> None:
    rows = _rows(cp, games, [0.1] * len(cp))
    assert tail_asymmetry_ci(rows, n_boot=64, seed=0) is None


def test_tail_ci_retains_defined_zero_means() -> None:
    rows = _rows([-500.0, 500.0, -500.0, 500.0], [0, 0, 1, 1], [0.0] * 4)
    assert tail_asymmetry_ci(rows, n_boot=64, seed=0) == (0.0, 0.0)


def test_tail_ci_matches_complete_game_resampling() -> None:
    rows = _rows(
        [-500.0, 500.0, -500.0, -500.0, 500.0, 0.0],
        [0, 0, 1, 1, 1, 1], [0.1, -0.2, 0.2, 0.1, -0.1, 0.05],
    )
    # Independent row-level oracle expands each sampled GAME in full. Unequal
    # tail row counts detect game-mean averaging or independent tail draws.
    estimates = []
    for draw in np.random.default_rng(7).integers(0, 2, size=(64, 2)):
        indices = np.concatenate([np.flatnonzero(rows.game_id == game) for game in draw])
        cp = rows.sf_cp[indices]
        diff = (rows.net_score - rows.sf_ruler_score)[indices]
        estimates.append(float(diff[cp <= -300.0].mean() + diff[cp > 300.0].mean()))
    expected = np.quantile(estimates, [0.025, 0.975])
    actual = tail_asymmetry_ci(rows, n_boot=64, seed=7)
    assert actual is not None
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize(("cp", "games", "n_boot"), [
    ([-500.0, -500.0], [0, 1], 64),  # the observed sample has no high tail
    ([500.0, 500.0], [0, 1], 64),  # the observed sample has no low tail
    ([-500.0, 500.0], [0, 0], 64),  # only one independent game
    ([-500.0, 500.0], [0, 1], 0),  # bootstrap disabled
])
def test_tail_ci_preserves_unavailable_contract(
    cp: list[float], games: list[int], n_boot: int,
) -> None:
    assert tail_asymmetry_ci(_rows(cp, games, [0.0] * len(cp)), n_boot=n_boot) is None


def test_tail_ci_does_not_refuse_a_defined_requested_draw() -> None:
    rows = _rows([-500.0, 500.0], [0, 1], [0.3, -0.1])
    # Seed 1's sole draw contains each game once. Availability is determined
    # by the requested batch, not by an arbitrary minimum tail-support cutoff.
    actual = tail_asymmetry_ci(rows, n_boot=1, seed=1)
    assert actual is not None
    assert actual == pytest.approx((0.2, 0.2))
