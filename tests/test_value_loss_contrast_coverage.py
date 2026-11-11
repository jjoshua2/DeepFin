"""Do not condition loss-stratum uncertainty on bootstrap coverage."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.value_loss_scorer import (
    CONTRAST_COLUMNS,
    CONTRAST_MAX_P_UNDEFINED,
    RowSet,
    _per_row_errors,
    build_report,
    contrast,
    render,
    undefined_draw_probability,
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


# --- The withhold decision must not depend on the seed or on n_boot. --------

def _sparse_lost_fixture(n_lost_games: int, n_games: int = 200):
    """``n_games`` games all in the rest stratum; the lost stratum in a few."""
    rng = np.random.default_rng(1)
    lost_games = sorted(rng.choice(n_games, n_lost_games, replace=False).tolist())
    lost_ids = [g for g in lost_games for _ in range(5)]
    rest_ids = [g for g in range(n_games) for _ in range(3)]
    lost, rest = _rows(lost_ids, lost=True), _rows(rest_ids, lost=False)
    q_lost = np.clip(_predictions(len(lost_ids), lost=True)
                     + rng.normal(0.0, 0.05, (len(lost_ids), 3)), 0.0, 1.0)
    q_rest = np.clip(_predictions(len(rest_ids), lost=False)
                     + rng.normal(0.0, 0.05, (len(rest_ids), 3)), 0.0, 1.0)
    return lost, rest, q_lost, q_rest


@pytest.mark.parametrize(
    ("n_lost_games", "reported"),
    # p_undef = (1 - k/200)^200: k=8 -> 2.9e-4 (below alpha/20 = 2.5e-3),
    # k=5 -> 6.3e-3 (above). Under an any-draw-undefined rule both of these
    # flip between "CI" and "no CI" across seeds and n_boot.
    [(8, True), (5, False)],
)
def test_withhold_decision_is_invariant_to_seed_and_n_boot(
    n_lost_games: int, reported: bool,
) -> None:
    lost, rest, q_lost, q_rest = _sparse_lost_fixture(n_lost_games)
    expected_p = (1.0 - n_lost_games / 200) ** 200
    assert (expected_p <= CONTRAST_MAX_P_UNDEFINED) is reported
    saw_undefined_draw = False
    for n_boot in (400, 2000, 10000):
        for seed in range(20):
            result = contrast(
                lost, rest, q_lost, q_rest, column="net_score", slope=0.01,
                draw_width_cp=60.0, n_boot=n_boot, seed=seed,
            )
            assert result.p_undefined == pytest.approx(expected_p, rel=1e-9)
            assert bool(np.isfinite(result.ci).all()) is reported, (n_boot, seed)
            if reported:
                assert result.ci[0] <= result.ci[1]
                assert 0.0 <= result.undefined_frac < 0.01
                saw_undefined_draw |= result.undefined_frac > 0.0
            else:
                assert not result.excludes_zero
                assert np.isnan(result.undefined_frac)  # decided before drawing
    if reported:
        # The fixture really exercises conditioning on defined draws.
        assert saw_undefined_draw


def test_undefined_draw_probability_matches_exact_enumeration() -> None:
    # 3-game pool: stratum a in games {0}, stratum b in games {0, 1, 2}.
    # Enumerate all 27 draws: a is absent in (2/3)^3 = 8/27 of them.
    games = np.array([0, 1, 2])
    draws = np.array(np.meshgrid(games, games, games)).reshape(3, -1).T
    absent = np.mean([not np.any(d == 0) for d in draws])
    assert undefined_draw_probability(games, np.array([0]), games) == pytest.approx(absent)
    assert undefined_draw_probability(games, games, games) == 0.0
    # Disjoint two-game case: each stratum absent in 1/4 of draws, never both.
    two = np.array([0, 1])
    assert undefined_draw_probability(two, np.array([0]), np.array([1])) == pytest.approx(0.5)


def test_populated_contrast_equals_all_draw_percentile_interval() -> None:
    # 40 games, lost in 30 of them: p_undef = 0.25**40 ~ 1e-24, so no draw is
    # undefined and the interval must be the plain all-draw percentile one
    # (the rule main used whenever every draw was defined).
    lost_ids = [g for g in range(30) for _ in range(2)]
    rest_ids = [g for g in range(40) for _ in range(3)]
    lost, rest = _rows(lost_ids, lost=True), _rows(rest_ids, lost=False)
    rng = np.random.default_rng(5)
    q_lost = np.clip(_predictions(len(lost_ids), lost=True)
                     + rng.normal(0.0, 0.1, (len(lost_ids), 3)), 0.0, 1.0)
    q_rest = np.clip(_predictions(len(rest_ids), lost=False)
                     + rng.normal(0.0, 0.1, (len(rest_ids), 3)), 0.0, 1.0)
    a, _, _ = _per_row_errors(lost, q_lost, slope=0.01, draw_width_cp=60.0)
    b, _, _ = _per_row_errors(rest, q_rest, slope=0.01, draw_width_cp=60.0)
    n_boot, seed = 500, 3
    draws = np.random.default_rng(seed).integers(0, 40, size=(n_boot, 40))
    lost_g, rest_g = np.asarray(lost_ids), np.asarray(rest_ids)
    boot = [
        np.concatenate([a["d_score"][lost_g == g] for g in d]).mean()
        - np.concatenate([b["d_score"][rest_g == g] for g in d]).mean()
        for d in draws
    ]
    result = contrast(
        lost, rest, q_lost, q_rest, column="d_score", slope=0.01,
        draw_width_cp=60.0, n_boot=n_boot, seed=seed,
    )
    assert result.undefined_frac == 0.0
    np.testing.assert_allclose(result.ci, np.quantile(boot, [0.025, 0.975]), rtol=1e-12)


def test_render_reports_analytic_and_realized_undefined_rates() -> None:
    lost, rest, q_lost, q_rest = _sparse_lost_fixture(8)
    report, _ = build_report(
        _rows([], lost=True), np.zeros((0, 3)), loss_prob_min=0.75, sf_cp_max=None,
        slope=0.01, draw_width_cp=60.0, n_boot=4, seed=0,
    )
    shown = contrast(
        lost, rest, q_lost, q_rest, column="net_score", slope=0.01,
        draw_width_cp=60.0, n_boot=2000, seed=0,
    )
    withheld = contrast(
        *_sparse_lost_fixture(5), column="net_score", slope=0.01,
        draw_width_cp=60.0, n_boot=2000, seed=0,
    )
    report.contrasts[:] = [shown, withheld]
    line_shown, line_withheld = render(report).splitlines()[-2:]
    assert f"undefined draws p={shown.p_undefined:.2e} realized={shown.undefined_frac:.4f}" \
        in line_shown
    assert line_shown.endswith(("excludes 0", "CONTAINS 0"))
    assert f"undefined draws p={withheld.p_undefined:.2e} realized=n/a" in line_withheld
    assert line_withheld.endswith("CI unavailable")
