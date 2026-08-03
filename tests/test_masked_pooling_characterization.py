"""CHARACTERIZATION of a KNOWN DEFECT. These tests pin CURRENT behaviour.

They are not an endorsement. They exist so that the fix -- which CHANGES the
value of ~18 reported loss columns and therefore invalidates every recorded
number in those columns -- cannot land silently. See the 2026-08-03 train/data
audit C3 and the restart-gated entry at the end of docs/experiment_ledger.md.

Two defects, one root:

1. `masked_mean` clamps its denominator to 1.0, so a bucket with ZERO eligible
   rows in a batch publishes 0.0 -- the best possible loss -- and that 0.0 is
   averaged in with real values. Realized per-row coverage on live shards runs
   from 0.096 (`has_sf_p0_regret`) to 0.978 (`has_sf_policy`), and the
   selfplay/curriculum split swings 1-35 %, so empty buckets are routine, not
   hypothetical.

2. Per-batch `masked_mean`s are pooled by dividing by the STEP COUNT (train
   path) or by weighting with `n_rows` (eval path). Neither is the mask count,
   so the published number is a mean of ratios, not the ratio of pooled sums --
   a different estimator as soon as the mask count varies between batches,
   which `losses.masked_sum_and_count`'s own docstring says. That correction
   was applied to 2 of ~20 masked heads (`_RATIO_METRIC_FIELDS`).

When either is fixed, these tests fail. That failure is the reminder to
invalidate the old records, not a regression.
"""

from __future__ import annotations

import torch

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.train.losses import compute_loss, masked_mean, masked_sum_and_count


def test_masked_mean_publishes_zero_for_an_empty_bucket() -> None:
    x = torch.tensor([3.0, 4.0, 5.0])
    empty = torch.zeros(3)

    assert float(masked_mean(x, empty)) == 0.0  # "no rows" is indistinguishable from "perfect"

    # The honest pair is already available and already used by 2 of ~20 heads.
    total, count = masked_sum_and_count(x, empty)
    assert float(total) == 0.0
    assert float(count) == 0.0  # THIS is what tells the two states apart


def _batch(b: int, *, covered: bool) -> dict[str, torch.Tensor]:
    policy = torch.ones((b, POLICY_SIZE))
    return {
        "x": torch.randn((b, 146, 8, 8)),
        "policy_t": policy / policy.sum(dim=-1, keepdim=True),
        "wdl_t": torch.zeros((b,), dtype=torch.int64),
        "has_policy": torch.ones((b,)),
        # Target present in BOTH cases; only the eligibility flag differs, so
        # what is isolated here is the empty bucket and not a missing target.
        "sf_volatility_t": torch.full((b, 3), 0.9),
        "has_sf_volatility": torch.ones((b,)) if covered else torch.zeros((b,)),
    }


def _outputs(b: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "policy_own": torch.randn((b, POLICY_SIZE)),
        "wdl": torch.randn((b, 3)),
        "sf_volatility": torch.zeros((b, 3)),
    }


def test_uncovered_head_reports_zero_loss_not_absence() -> None:
    covered = compute_loss(_outputs(4), _batch(4, covered=True))
    uncovered = compute_loss(_outputs(4), _batch(4, covered=False))

    assert float(covered["sf_volatility"]) > 0.0
    # Same predictions, same targets, zero eligible rows -> the BEST possible
    # reading. Averaged across an iteration this drags the column toward 0
    # in proportion to how often the bucket is empty.
    assert float(uncovered["sf_volatility"]) == 0.0


def test_mean_of_per_batch_means_is_not_the_pooled_mean() -> None:
    """The estimator the train path actually publishes, against the honest one."""
    x_a, mask_a = torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, 1.0, 1.0])
    x_b, mask_b = torch.tensor([5.0, 0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0, 0.0])

    published = (float(masked_mean(x_a, mask_a)) + float(masked_mean(x_b, mask_b))) / 2.0

    sum_a, n_a = masked_sum_and_count(x_a, mask_a)
    sum_b, n_b = masked_sum_and_count(x_b, mask_b)
    pooled = float(sum_a + sum_b) / float(n_a + n_b)

    assert published == 3.0  # (1.0 + 5.0) / 2 -- each batch weighted equally
    assert pooled == 1.8     # 9 / 5 -- each ROW weighted equally
