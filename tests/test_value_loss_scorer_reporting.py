"""Missing contrast uncertainty must not be reported as measured inclusion."""
from __future__ import annotations

import numpy as np

from scripts.value_loss_scorer import (
    Contrast,
    ErrorStat,
    Report,
    RowSet,
    contrast,
    render,
)


def _report(contrasts: list[Contrast]) -> Report:
    def stratum(name: str) -> ErrorStat:
        return ErrorStat(
            name=name, n=0, n_games=0, n_outcome=0,
            oracle_loss_prob=float("nan"), oracle_score=float("nan"),
            net_score=float("nan"), means={}, cis={}, quantiles={}, n_used={},
            cp_clamped_frac=float("nan"),
        )

    return Report(lost=stratum("lost"), rest=stratum("rest"), contrasts=contrasts)


def test_finite_contrast_labels_and_format_are_unchanged() -> None:
    cases = [
        ((1.0, 2.0), True),
        ((-2.0, -1.0), True),
        ((-1.0, 1.0), False),
        ((0.0, 1.0), False),
        ((-1.0, 0.0), False),
        ((0.0, 0.0), False),
    ]
    contrasts = [Contrast(f"column_{i}", 0.25, ci) for i, (ci, _) in enumerate(cases)]
    lines = render(_report(contrasts)).splitlines()[-len(cases):]
    for c, line, (_, excludes) in zip(contrasts, lines, cases):
        assert c.excludes_zero is excludes
        flag = "excludes 0" if excludes else "CONTAINS 0"
        assert line == (
            f"{c.column:>12}    +0.2500  95% CI [{c.ci[0]:+.4f}, {c.ci[1]:+.4f}]  {flag}"
        )


def test_nonfinite_contrast_bounds_are_unavailable_not_contains_zero() -> None:
    for bound in (float("nan"), float("inf"), -float("inf")):
        for ci in ((bound, 1.0), (-1.0, bound), (bound, bound)):
            c = Contrast("net_score", 0.25, ci)
            line = render(_report([c])).splitlines()[-1]
            assert c.excludes_zero is False  # Preserve the existing boolean API.
            assert line.endswith("  CI unavailable")
            assert "CONTAINS 0" not in line
            assert "excludes 0" not in line


def _rows(n: int) -> RowSet:
    return RowSet(
        x=np.zeros((n, 1, 1, 1)), p_sf=np.tile([0.2, 0.3, 0.5], (n, 1)),
        outcome=np.full(n, -1), game_id=np.arange(n), ply=np.zeros(n),
        is_selfplay=np.zeros(n, dtype=bool), source="synthetic-reporting-test",
    )


def test_missing_data_and_insufficient_bootstrap_reports_are_unavailable() -> None:
    full, empty = _rows(2), _rows(0)
    for lost, rest, n_boot in ((empty, full, 4), (full, empty, 4), (full, full, 1)):
        c = contrast(
            lost, rest, lost.p_sf, rest.p_sf, column="net_score", slope=0.01,
            draw_width_cp=60.0, n_boot=n_boot, seed=0,
        )
        assert np.isnan(c.ci).all()
        line = render(_report([c])).splitlines()[-1]
        assert line.endswith("  CI unavailable")
        assert "CONTAINS 0" not in line
    # The estimator still emits a finite delta when only its CI is unavailable.
    assert c.delta == 0.0
