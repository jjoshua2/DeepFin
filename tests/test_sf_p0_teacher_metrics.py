"""Observability for the sf_p0 policy teacher (`w_sf_own` / `w_sf_own_regret`).

The teacher was fully off for about a month in July 2026 and nobody noticed,
because nothing in `progress.csv` moved when it died: both terms are masked to
eligible rows, so "no eligible rows" and "eligible rows at zero loss" are the
same number. These tests pin BOTH halves of the fix — the masked means and the
eligible-row fractions that disambiguate them — plus the row-weighted
aggregation those fractions are only meaningful under.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from chess_anti_engine.train.losses import compute_loss
from chess_anti_engine.train.trainer import (
    _RATIO_METRIC_FIELDS,
    _TRAIN_METRICS_FIELDS,
    TrainMetrics,
    Trainer,
    _loss_sums_to_metric_kwargs,
)
from chess_anti_engine.tune.trainable_report import (
    _TRAIN_METRIC_DEFAULTS,
    _train_metrics_dict,
)

SF_P0_COLUMNS = (
    "m_sf_own", "m_sf_own_regret", "has_sf_p0_frac", "has_sf_p0_regret_frac",
)

# Fixture move space: index 2 is SF's recommendation (zero regret), index 0 is
# maximally bad, index 1 is halfway.
_ACTIONS = 3
_SF_BEST = 2
_REGRET_VEC = (1.0, 0.5, 0.0)

Batch = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


def _losses(batch: Batch) -> dict[str, torch.Tensor]:
    """compute_loss with both sf_p0 teacher terms switched on."""
    outputs, inputs = batch
    return compute_loss(
        outputs, inputs, w_wdl=0.0, w_sf_own=0.5, w_sf_own_regret=0.5,
    )


def _make_batch(
    logits: list[list[float]],
    *,
    eligible: list[float],
    regret_eligible: list[float],
) -> Batch:
    """(outputs, batch) with per-row control of both sf_p0 eligibility masks."""
    n = len(logits)
    assert len(eligible) == n
    assert len(regret_eligible) == n
    policy_t = torch.zeros((n, _ACTIONS), dtype=torch.float32)
    policy_t[:, 0] = 1.0
    sf_p0_target = torch.zeros((n, _ACTIONS), dtype=torch.float32)
    sf_p0_target[:, _SF_BEST] = 1.0
    batch = {
        "x": torch.zeros((n, 1, 1, 1), dtype=torch.float32),
        "policy_t": policy_t,
        "wdl_t": torch.zeros((n,), dtype=torch.long),
        "has_policy": torch.ones((n,), dtype=torch.float32),
        "is_network_turn": torch.ones((n,), dtype=torch.float32),
        "sf_p0_policy_t": sf_p0_target,
        "has_sf_p0": torch.tensor(eligible, dtype=torch.float32),
        "sf_p0_regret_t": torch.tensor([list(_REGRET_VEC)] * n, dtype=torch.float32),
        "has_sf_p0_regret": torch.tensor(regret_eligible, dtype=torch.float32),
    }
    outputs = {
        "policy": torch.tensor(logits, dtype=torch.float32),
        "wdl": torch.zeros((n, 3), dtype=torch.float32),
    }
    return outputs, batch


def _accumulate(*batches: Batch) -> dict[str, float]:
    """Mirror the trainer's per-microbatch accumulation of compute_loss scalars."""
    sums: dict[str, float] = {}
    for batch in batches:
        losses = _losses(batch)
        for key, value in Trainer._extract_loss_scalars(losses).items():
            sums[key] = sums.get(key, 0.0) + value
    return sums


def _metrics_from(*batches: Batch) -> dict[str, float]:
    """The kwargs TrainMetrics is built from, for an iteration of ``batches``."""
    return _loss_sums_to_metric_kwargs(_accumulate(*batches), float(len(batches)))


# --- Independent reference implementations (no reuse of the code under test) ---

def _row_ce(logit_row: list[float]) -> float:
    """Soft CE of one row against the one-hot SF recommendation."""
    return float(-torch.log_softmax(torch.tensor(logit_row), dim=-1)[_SF_BEST])


def _row_regret(logit_row: list[float]) -> float:
    """E_p[regret] of one row under the fixture regret vector."""
    probs = torch.softmax(torch.tensor(logit_row), dim=-1)
    return float((probs * torch.tensor(_REGRET_VEC)).sum())


def _row_weighted(
    batches: tuple[Batch, ...], mask_key: str, per_row,
) -> float:
    """Sum over eligible rows / count of eligible rows, pooled over batches."""
    total, n = 0.0, 0.0
    for outputs, batch in batches:
        for i, keep in enumerate(batch[mask_key].tolist()):
            if keep:
                total += per_row(outputs["policy"][i].tolist())
                n += 1.0
    return total / n if n else 0.0


def _mean_of_means(
    batches: tuple[Batch, ...], mask_key: str, per_row,
) -> float:
    """The WRONG estimator: per-batch masked means averaged with equal weight."""
    return sum(_row_weighted((b,), mask_key, per_row) for b in batches) / len(batches)


# --- The fixtures: deliberately ragged, and ragged differently per mask ---
#
# Batch A has 4 rows, 1 eligible for the CE term and 2 for the regret term.
# Batch B has 6 rows, 4 eligible for the CE term and 1 for the regret term.
# Every count differs from every other count, so no two of the four columns can
# be swapped, and the row-weighted and mean-of-means estimators disagree on all
# four. The ineligible rows carry loss values far from the eligible ones, so
# dropping the mask also lands nowhere near the right answer.
_FLAT = [0.0, 0.0, 0.0]          # CE = log 3, E[regret] = 0.5
_AGAINST_SF = [5.0, 0.0, -5.0]   # near-certain on the worst move: big CE, regret ~1
_WITH_SF = [-5.0, 0.0, 5.0]      # near-certain on SF's move: tiny CE, regret ~0


def _batch_a() -> Batch:
    return _make_batch(
        [_FLAT, _AGAINST_SF, _AGAINST_SF, _AGAINST_SF],
        eligible=[1.0, 0.0, 0.0, 0.0],
        regret_eligible=[1.0, 1.0, 0.0, 0.0],
    )


def _batch_b() -> Batch:
    return _make_batch(
        [_WITH_SF, _WITH_SF, _WITH_SF, _WITH_SF, _FLAT, _FLAT],
        eligible=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        regret_eligible=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_sf_p0_columns_exist_end_to_end() -> None:
    """Every ratio field is a real TrainMetrics field and every input key it
    needs is actually produced by compute_loss — otherwise the metric is
    silently dropped, which is the failure mode this PR exists to fix."""
    losses = _losses(_batch_a())
    for field, (num_key, den_key) in _RATIO_METRIC_FIELDS.items():
        assert field in _TRAIN_METRICS_FIELDS, f"{field} would be dropped"
        assert num_key in losses, f"{field} numerator {num_key} not reported"
        assert den_key in losses, f"{field} denominator {den_key} not reported"
    assert set(SF_P0_COLUMNS) <= set(_RATIO_METRIC_FIELDS)


def test_fractions_are_zero_when_no_row_is_eligible() -> None:
    """THE regression test for the month-long outage: when the workers record
    no sf_p0 labels, both fraction columns read exactly 0.0."""
    dead = _make_batch(
        [_FLAT, _AGAINST_SF],
        eligible=[0.0, 0.0],
        regret_eligible=[0.0, 0.0],
    )
    got = _metrics_from(dead, dead)
    assert got["has_sf_p0_frac"] == 0.0
    assert got["has_sf_p0_regret_frac"] == 0.0


def test_fractions_report_the_eligible_share_when_rows_are_eligible() -> None:
    """5 of 10 rows eligible for the CE term, 3 of 10 for the regret term."""
    got = _metrics_from(_batch_a(), _batch_b())
    assert got["has_sf_p0_frac"] == pytest.approx(0.5)
    assert got["has_sf_p0_regret_frac"] == pytest.approx(0.3)


def test_no_eligible_rows_is_distinguishable_from_zero_loss_on_eligible_rows() -> None:
    """The ambiguity that hid the outage: a masked mean of 0.0 means nothing on
    its own. The fraction column has to separate the two worlds."""
    dead = _make_batch(
        [_WITH_SF, _WITH_SF], eligible=[0.0, 0.0], regret_eligible=[0.0, 0.0],
    )
    # Eligible rows whose loss is (numerically) zero: policy_own already puts
    # all its mass on SF's move, so both terms sit at ~0 with rows present.
    alive_at_zero = _make_batch(
        [[-50.0, -50.0, 50.0]] * 2, eligible=[1.0, 1.0], regret_eligible=[1.0, 1.0],
    )

    dead_m = _metrics_from(dead)
    alive_m = _metrics_from(alive_at_zero)

    assert dead_m["m_sf_own"] == pytest.approx(0.0, abs=1e-9)
    assert alive_m["m_sf_own"] == pytest.approx(0.0, abs=1e-9)
    assert dead_m["m_sf_own_regret"] == pytest.approx(0.0, abs=1e-9)
    assert alive_m["m_sf_own_regret"] == pytest.approx(0.0, abs=1e-9)
    # ...and yet the reported rows are NOT identical.
    assert [dead_m[c] for c in SF_P0_COLUMNS] != [alive_m[c] for c in SF_P0_COLUMNS]
    assert dead_m["has_sf_p0_frac"] == 0.0
    assert alive_m["has_sf_p0_frac"] == pytest.approx(1.0)
    assert dead_m["has_sf_p0_regret_frac"] == 0.0
    assert alive_m["has_sf_p0_regret_frac"] == pytest.approx(1.0)


def test_ragged_eligibility_aggregates_by_eligible_rows_not_by_batch() -> None:
    """Batches with DIFFERENT eligible counts must pool by row.

    Written to FAIL against the `/steps` mean-of-means estimator: each assert
    checks the row-weighted value AND that the mean-of-means value is far
    enough away that no rounding could confuse them.
    """
    batches = (_batch_a(), _batch_b())
    got = _metrics_from(*batches)

    expected = {
        "m_sf_own": _row_weighted(batches, "has_sf_p0", _row_ce),
        "m_sf_own_regret": _row_weighted(batches, "has_sf_p0_regret", _row_regret),
        "has_sf_p0_frac": 5.0 / 10.0,
        "has_sf_p0_regret_frac": 3.0 / 10.0,
    }
    wrong = {
        "m_sf_own": _mean_of_means(batches, "has_sf_p0", _row_ce),
        "m_sf_own_regret": _mean_of_means(batches, "has_sf_p0_regret", _row_regret),
        "has_sf_p0_frac": (1.0 / 4.0 + 4.0 / 6.0) / 2.0,
        "has_sf_p0_regret_frac": (2.0 / 4.0 + 1.0 / 6.0) / 2.0,
    }

    for column in SF_P0_COLUMNS:
        assert abs(expected[column] - wrong[column]) > 0.02, (
            f"{column}: fixture no longer separates the two estimators"
        )
        assert got[column] == pytest.approx(expected[column], rel=1e-5)


def test_masked_means_ignore_ineligible_rows() -> None:
    """Ineligible rows carry a wildly different loss here, so a mask-free mean
    lands nowhere near the reported value."""
    batches = (_batch_a(), _batch_b())
    got = _metrics_from(*batches)

    unmasked_ce = sum(
        _row_ce(o["policy"][i].tolist())
        for o, b in batches
        for i in range(int(b["x"].shape[0]))
    ) / 10.0
    assert abs(unmasked_ce - got["m_sf_own"]) > 1.0


def _blank_metrics(**overrides: float) -> TrainMetrics:
    base: dict[str, Any] = dict.fromkeys(
        (
            "loss", "policy_loss", "soft_policy_loss", "future_policy_loss",
            "wdl_loss", "sf_move_loss", "sf_move_acc", "sf_eval_loss",
            "categorical_loss", "volatility_loss", "sf_volatility_loss",
            "moves_left_loss",
        ),
        0.0,
    )
    base.update(overrides)
    return TrainMetrics(**base)


def test_sf_p0_columns_reach_the_progress_row() -> None:
    """TrainMetrics -> progress.csv. Ray's CSV logger fixes the header on row 1,
    so the no-train-phase fallback must carry the same keys (asserted by
    tests/test_trainable_report.py against _TRAIN_METRIC_DEFAULTS)."""
    got = _train_metrics_dict(
        _blank_metrics(
            m_sf_own=1.234, m_sf_own_regret=0.0567,
            has_sf_p0_frac=0.152, has_sf_p0_regret_frac=0.141,
        )
    )
    assert got["m_sf_own"] == 1.234
    assert got["m_sf_own_regret"] == 0.0567
    assert got["has_sf_p0_frac"] == 0.152
    assert got["has_sf_p0_regret_frac"] == 0.141
    for column in SF_P0_COLUMNS:
        assert column in _TRAIN_METRIC_DEFAULTS
