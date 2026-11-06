"""Do not condition loss-stratum uncertainty on bootstrap coverage."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.value_loss_scorer import (
    CONTRAST_COLUMNS,
    RowSet,
    _per_row_errors,
    build_report,
    contrast,
    render,
)


def _rows(games: list[int], *, lost: bool) -> RowSet:
    n = len(games)
    return RowSet(
        x=np.zeros((n, 1, 1, 1)),
        p_sf=np.tile([0.1, 0.0, 0.9] if lost else [0.9, 0.0, 0.1], (n, 1)),
        outcome=np.full(n, -1), game_id=np.asarray(games),
        ply=np.arange(n), is_selfplay=np.zeros(n, dtype=bool),
        source="synthetic-contrast-coverage",
    )


def _predictions(n: int, *, lost: bool) -> np.ndarray:
    return np.tile([0.8, 0.0, 0.2] if lost else [0.2, 0.0, 0.8], (n, 1))


@pytest.mark.parametrize("column", CONTRAST_COLUMNS)
def test_disjoint_game_strata_do_not_discard_undefined_resamples(column: str) -> None:
    lost, rest = _rows([0], lost=True), _rows([1], lost=False)
    result = contrast(
        lost, rest, _predictions(1, lost=True), _predictions(1, lost=False),
        column=column, slope=0.01, draw_width_cp=60.0, n_boot=256, seed=0,
    )
    draws = np.random.default_rng(0).integers(0, 2, size=(256, 2))
    covered = np.any(draws == 0, axis=1) & np.any(draws == 1, axis=1)
    assert 2 < int(covered.sum()) < len(draws)
    # Half the possible two-game draws omit a stratum. Keeping only [0, 1]
    # and [1, 0] manufactured a zero-width interval on these same data.
    assert np.isfinite(result.delta)
    assert np.isnan(result.ci).all()
    assert not result.excludes_zero


@pytest.mark.parametrize("column", CONTRAST_COLUMNS)
def test_one_shared_game_does_not_estimate_between_game_uncertainty(column: str) -> None:
    lost, rest = _rows([7, 7], lost=True), _rows([7], lost=False)
    result = contrast(
        lost, rest, _predictions(2, lost=True), _predictions(1, lost=False),
        column=column, slope=0.01, draw_width_cp=60.0, n_boot=256, seed=0,
    )
    assert np.isfinite(result.delta)
    assert np.isnan(result.ci).all()
    assert not result.excludes_zero


@pytest.mark.parametrize("sparse_side", ["lost", "rest"])
def test_partial_game_overlap_still_requires_each_draw_to_cover_both_strata(
    sparse_side: str,
) -> None:
    lost_games = [0] if sparse_side == "lost" else [0, 1]
    rest_games = [0] if sparse_side == "rest" else [0, 1]
    lost, rest = _rows(lost_games, lost=True), _rows(rest_games, lost=False)
    result = contrast(
        lost, rest, _predictions(len(lost), lost=True), _predictions(len(rest), lost=False),
        column="net_score", slope=0.01, draw_width_cp=60.0, n_boot=256, seed=0,
    )
    assert result.delta == pytest.approx(0.6)
    assert np.isnan(result.ci).all()


@pytest.mark.parametrize("seed", [0, 11, 42])
def test_fully_covered_draws_keep_paired_row_weighting_and_seed(seed: int) -> None:
    lost, rest = _rows([0, 1, 1, 2], lost=True), _rows([0, 0, 1, 2, 2], lost=False)
    q_lost = np.array([[0.8, 0.0, 0.2], [0.2, 0.4, 0.4], [0.1, 0.2, 0.7], [0.4, 0.2, 0.4]])
    q_rest = np.array([[0.7, 0.2, 0.1], [0.2, 0.2, 0.6], [0.5, 0.2, 0.3], [0.1, 0.4, 0.5], [0.4, 0.4, 0.2]])
    n_boot = 61
    draws = np.random.default_rng(seed).integers(0, 3, size=(n_boot, 3))
    a, _, _ = _per_row_errors(lost, q_lost, slope=0.01, draw_width_cp=60.0)
    b, _, _ = _per_row_errors(rest, q_rest, slope=0.01, draw_width_cp=60.0)
    # Independent row-expansion oracle preserves duplicates and unequal
    # per-game row counts, instead of repeating the sum/count implementation.
    for column in CONTRAST_COLUMNS:
        boot = [
            np.concatenate([a[column][lost.game_id == game] for game in draw]).mean()
            - np.concatenate([b[column][rest.game_id == game] for game in draw]).mean()
            for draw in draws
        ]
        result = contrast(
            lost, rest, q_lost, q_rest, column=column, slope=0.01,
            draw_width_cp=60.0, n_boot=n_boot, seed=seed,
        )
        assert result.delta == pytest.approx(a[column].mean() - b[column].mean())
        np.testing.assert_allclose(result.ci, np.quantile(boot, [0.025, 0.975]), rtol=1e-12)


@pytest.mark.parametrize("same_game", [False, True])
def test_real_report_withholds_unsupported_ci_but_preserves_point_estimate(same_game: bool) -> None:
    rows = _rows([0, 0 if same_game else 1], lost=True)
    rows.p_sf[1] = [0.9, 0.0, 0.1]
    q = np.array([[0.8, 0.0, 0.2], [0.2, 0.0, 0.8]])
    report, mask = build_report(
        rows, q, loss_prob_min=0.75, sf_cp_max=None, slope=0.01,
        draw_width_cp=60.0, n_boot=256, seed=0,
    )
    assert mask.tolist() == [True, False]
    assert report.contrasts[0].delta == pytest.approx(0.6)
    text = render(report)
    for result, line in zip(report.contrasts, text.splitlines()[-len(CONTRAST_COLUMNS):]):
        assert np.isfinite(result.delta)
        assert np.isnan(result.ci).all()
        assert line.endswith("CI unavailable")
        assert "excludes 0" not in line
