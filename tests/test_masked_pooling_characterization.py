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
invalidate the old records, not a regression. Defect 1 is covered by
`test_masked_mean_publishes_zero_for_an_empty_bucket` and
`test_uncovered_head_reports_zero_loss_not_absence`; defect 2 by
`test_train_path_pools_a_masked_head_by_step_count_not_by_row_count` and
`test_only_the_two_sf_p0_heads_are_pooled_as_ratios_of_sums`, which assert on
the trainer helpers that actually publish the pooled number -- a demonstration
over `masked_mean` alone would stay green through the pooling fix.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.train.losses import (
    EXACT_OBJECTIVE_NAMES,
    compute_loss,
    masked_mean,
    masked_sum_and_count,
)
from chess_anti_engine.train.trainer import (
    Trainer,
    _EXACT_MASKED_METRIC_FIELDS,
    _RATIO_METRIC_FIELDS,
    _exact_masked_metric_kwargs,
    _loss_sums_to_metric_kwargs,
)


def test_exact_objective_census_matches_all_optimizer_masks() -> None:
    arrays = {
        "x": np.zeros((3, 1, 1, 1), dtype=np.float32),
        "is_network_turn": np.asarray([1, 1, 0], dtype=bool),
        "has_policy": np.asarray([1, 0, 1], dtype=np.float32),
        "has_policy_soft": np.asarray([1, 1, 1], dtype=np.float32),
        "has_future": np.asarray([0, 1, 1], dtype=np.float32),
        "has_sf_p0": np.asarray([1, 0, 1], dtype=np.float32),
        "sf_p0_policy_target": np.zeros((3, 2), dtype=np.float32),
        "has_sf_p0_regret": np.asarray([0, 1, 1], dtype=np.float32),
        "sf_p0_regret": np.zeros((3, 2), dtype=np.float32),
        "has_sf_policy": np.asarray([1, 1, 1], dtype=np.float32),
        "has_sf_wdl": np.asarray([1, 0, 1], dtype=np.float32),
        "sf_wdl": np.full((3, 3), 1.0 / 3.0, dtype=np.float32),
        "wdl_target": np.asarray([0, 1, 2], dtype=np.int64),
        "has_categorical": np.asarray([0, 1, 1], dtype=np.float32),
        "has_volatility": np.asarray([1, 0, 1], dtype=np.float32),
        "has_sf_volatility": np.asarray([0, 1, 1], dtype=np.float32),
        "has_moves_left": np.asarray([1, 1, 0], dtype=np.float32),
    }
    trainer = SimpleNamespace(
        model=SimpleNamespace(policy_size=2),
        soft_policy_min_tv=0.0,
        policy_target_temp=1.0,
        sf_policy_sparse_ce=False,
        rebuild_sf_targets=False,
        sf_wdl_conf_power=0.0,
        sf_wdl_draw_scale=1.0,
        sf_wdl_temperature=1.0,
    )

    got = Trainer.exact_objective_mask_counter(cast(Any, trainer), arrays)

    assert tuple(got) == EXACT_OBJECTIVE_NAMES
    assert got == {
        "policy": 1.0,
        "soft_policy": 2.0,
        "future_policy": 1.0,
        "sf_own": 1.0,
        "sf_own_regret": 1.0,
        "wdl": 2.0,
        "sf_move": 2.0,
        "sf_eval": 1.0,
        "categorical": 1.0,
        "volatility": 1.0,
        "sf_volatility": 1.0,
        "moves_left": 2.0,
        "sf_policy_floor": 1.0,
        "sf_shape": 1.0,
    }


def test_exact_objective_census_uses_the_real_soft_tv_population() -> None:
    arrays = {
        "x": np.zeros((2, 1, 1, 1), dtype=np.float32),
        "policy_target": np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "policy_soft_target": np.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32,
        ),
        "has_policy_soft": np.ones(2, dtype=np.float32),
        "wdl_target": np.zeros(2, dtype=np.int64),
    }
    trainer = SimpleNamespace(
        model=SimpleNamespace(policy_size=2),
        soft_policy_min_tv=0.5,
        policy_target_temp=1.0,
        sf_policy_sparse_ce=False,
        rebuild_sf_targets=False,
        sf_wdl_conf_power=0.0,
        sf_wdl_draw_scale=1.0,
        sf_wdl_temperature=1.0,
    )

    got = Trainer.exact_objective_mask_counter(cast(Any, trainer), arrays)

    assert got["soft_policy"] == 1.0


def test_exact_objective_census_applies_fractional_draw_scaling() -> None:
    arrays = {
        "x": np.zeros((2, 1, 1, 1), dtype=np.float32),
        "is_network_turn": np.ones(2, dtype=bool),
        "has_sf_wdl": np.ones(2, dtype=np.float32),
        "sf_wdl": np.full((2, 3), 1.0 / 3.0, dtype=np.float32),
        "wdl_target": np.asarray([0, 1], dtype=np.int64),
    }
    trainer = SimpleNamespace(
        model=SimpleNamespace(policy_size=2),
        soft_policy_min_tv=0.0,
        policy_target_temp=1.0,
        sf_policy_sparse_ce=False,
        rebuild_sf_targets=False,
        sf_wdl_conf_power=0.0,
        sf_wdl_draw_scale=0.5,
        sf_wdl_temperature=1.0,
    )

    got = Trainer.exact_objective_mask_counter(cast(Any, trainer), arrays)

    assert got["sf_eval"] == 1.5


def test_exact_sf_move_mask_excludes_targetless_mixed_schema_rows() -> None:
    trainer = SimpleNamespace(
        model=SimpleNamespace(policy_size=POLICY_SIZE),
        soft_policy_min_tv=0.0,
        policy_target_temp=1.0,
        sf_policy_sparse_ce=False,
        rebuild_sf_targets=False,
        sf_wdl_conf_power=0.0,
        sf_wdl_draw_scale=1.0,
        sf_wdl_temperature=1.0,
    )
    shard_censuses = [
        Trainer.exact_objective_mask_counter(
            cast(Any, trainer),
            {
                "x": np.zeros((1, 1, 1, 1), dtype=np.float32),
                "sf_policy_target": np.eye(1, POLICY_SIZE, dtype=np.float32),
                "has_sf_policy": np.ones(1, dtype=np.float32),
            },
        ),
        Trainer.exact_objective_mask_counter(
            cast(Any, trainer),
            {
                "x": np.zeros((1, 1, 1, 1), dtype=np.float32),
                "sf_move_index": np.zeros(1, dtype=np.int32),
                "has_sf_move": np.ones(1, dtype=np.float32),
            },
        ),
    ]
    census = {
        name: sum(shard[name] for shard in shard_censuses)
        for name in EXACT_OBJECTIVE_NAMES
    }
    # Union concatenation materializes a zero has_sf_policy value and a zero
    # dense target for the sparse-only row. Its best-move flag alone must not
    # enlarge the dense objective's denominator when sparse CE is disabled.
    batch = _batch(2, covered=False)
    batch["has_sf_policy"] = torch.tensor([1.0, 0.0])
    batch["has_sf_move"] = torch.ones(2)
    batch["sf_policy_t"] = torch.stack(
        (batch["policy_t"][0], torch.zeros_like(batch["policy_t"][1])),
    )
    outputs = _outputs(2)
    outputs["policy_sf"] = torch.randn((2, POLICY_SIZE))

    losses = compute_loss(
        outputs,
        batch,
        report_exact_masked_sums=True,
        exact_corpus_rows=2,
        exact_objective_mask_weights=census,
    )
    _, weight_key = _EXACT_MASKED_METRIC_FIELDS["sf_move_loss"]

    assert census["sf_move"] == 1.0
    assert float(losses[weight_key]) == 1.0


def test_exact_sf_move_mask_stays_stable_through_mixed_schema_rebuild() -> None:
    from chess_anti_engine.replay.disk_buffer import _concat_sparse_batches
    from chess_anti_engine.replay.shard import SF_CP_SENTINEL, SF_MULTIPV_RAW_MAX
    from chess_anti_engine.train.target_builder import (
        SfTargetParams,
        rebuild_sf_targets_in_arrays,
    )

    params = SfTargetParams(sf_policy_temp=0.012)
    trainer = SimpleNamespace(
        model=SimpleNamespace(policy_size=POLICY_SIZE),
        soft_policy_min_tv=0.0,
        policy_target_temp=1.0,
        sf_policy_sparse_ce=False,
        rebuild_sf_targets=True,
        sf_target_params=params,
        sf_wdl_conf_power=0.0,
        sf_wdl_draw_scale=1.0,
        sf_wdl_temperature=1.0,
    )
    raw = np.full((1, SF_MULTIPV_RAW_MAX, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[0, 0] = (3, 40, 0, 700, 200)
    common = {
        "x": np.zeros((1, 1, 1, 1), np.float16),
        "policy_target": np.eye(1, POLICY_SIZE, dtype=np.float16),
        "wdl_target": np.zeros(1, np.int8),
        "priority": np.ones(1, np.float32),
        "has_policy": np.ones(1, np.uint8),
        "sf_multipv_raw": raw,
        "has_sf_multipv_raw": np.ones(1, np.uint8),
        "sf_move_index": np.array([3], np.int32),
        "has_sf_move": np.ones(1, np.uint8),
    }
    dense_target = np.zeros((1, POLICY_SIZE), np.float16)
    dense_target[0, 5] = 1.0
    dense = {
        **common,
        "sf_policy_target": dense_target,
        "has_sf_policy": np.ones(1, np.uint8),
    }
    sparse_only = {
        key: np.array(value, copy=True) for key, value in common.items()
    }
    shard_censuses = [
        Trainer.exact_objective_mask_counter(cast(Any, trainer), shard)
        for shard in (dense, sparse_only)
    ]
    census = {
        name: sum(shard[name] for shard in shard_censuses)
        for name in EXACT_OBJECTIVE_NAMES
    }
    merged = _concat_sparse_batches([dense, sparse_only])
    rebuilt, coverage = rebuild_sf_targets_in_arrays(merged, params=params)

    batch = _batch(2, covered=False)
    batch["has_sf_policy"] = torch.as_tensor(rebuilt["has_sf_policy"])
    batch["has_sf_move"] = torch.as_tensor(rebuilt["has_sf_move"])
    batch["sf_policy_t"] = torch.as_tensor(rebuilt["sf_policy_target"])
    outputs = _outputs(2)
    outputs["policy_sf"] = torch.randn((2, POLICY_SIZE))
    losses = compute_loss(
        outputs,
        batch,
        report_exact_masked_sums=True,
        exact_corpus_rows=2,
        exact_objective_mask_weights=census,
    )
    _, weight_key = _EXACT_MASKED_METRIC_FIELDS["sf_move_loss"]

    assert coverage.policy_rebuilt == 1
    assert census["sf_move"] == 1.0
    assert float(rebuilt["sf_policy_target"][1].sum()) == 0.0
    assert float(losses[weight_key]) == 1.0


def test_exact_sf_move_mask_keeps_legacy_dense_target_fallback() -> None:
    trainer = SimpleNamespace(
        model=SimpleNamespace(policy_size=POLICY_SIZE),
        soft_policy_min_tv=0.0,
        policy_target_temp=1.0,
        sf_policy_sparse_ce=False,
        rebuild_sf_targets=False,
        sf_wdl_conf_power=0.0,
        sf_wdl_draw_scale=1.0,
        sf_wdl_temperature=1.0,
    )
    target = np.eye(1, POLICY_SIZE, dtype=np.float32)
    census = Trainer.exact_objective_mask_counter(
        cast(Any, trainer),
        {
            "x": np.zeros((1, 1, 1, 1), dtype=np.float32),
            "sf_policy_target": target,
            "has_sf_move": np.ones(1, dtype=np.float32),
        },
    )
    batch = _batch(1, covered=False)
    batch.pop("has_sf_policy", None)
    batch["has_sf_move"] = torch.ones(1)
    batch["sf_policy_t"] = torch.as_tensor(target)
    outputs = _outputs(1)
    outputs["policy_sf"] = torch.randn((1, POLICY_SIZE))

    losses = compute_loss(
        outputs,
        batch,
        report_exact_masked_sums=True,
        exact_corpus_rows=1,
        exact_objective_mask_weights=census,
    )
    _, weight_key = _EXACT_MASKED_METRIC_FIELDS["sf_move_loss"]

    assert census["sf_move"] == 1.0
    assert float(losses[weight_key]) == 1.0


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


# --- the pooling half, asserted on the instrument that publishes it ---------
# The three tests above cover `masked_mean`'s denominator clamp only. The
# POOLING estimator lives in the trainer, not in losses.py, and a guard must
# share the criterion's instrument: without the two tests below, B1's pooling
# fix could land while `compute_loss` still publishes per-batch means and this
# file would stay GREEN -- the reminder to invalidate the records never firing.


def test_train_path_pools_a_masked_head_by_step_count_not_by_row_count() -> None:
    """`sf_volatility_loss` is a MEAN OF PER-BATCH RATIOS over the iteration.

    Two microbatches whose masked means are 1.0 and 5.0 publish 3.0 regardless
    of how many rows each contributed. When the fix lands, `sf_volatility` moves
    into a (sum, count) pair and this division by the STEP COUNT disappears.
    """
    n_micro = 2.0
    sums = {"sf_volatility": 1.0 + 5.0}  # accumulated per-batch masked means

    out = _loss_sums_to_metric_kwargs(sums, n_micro)

    assert out["sf_volatility_loss"] == 3.0
    # No row count travels with it, so nothing published can contradict the 3.0.
    assert "sf_volatility_rows" not in out
    assert not any(k.startswith("sf_volatility") and k.endswith("_rows") for k in out)


def test_only_the_sf_p0_heads_are_pooled_as_ratios_of_sums() -> None:
    """The row-weighted correction exists and is applied to 3 loss heads.

    `_RATIO_METRIC_FIELDS` also carries `*_frac` coverage columns and the
    desync-contamination detector; the LOSS entries are the ones this
    characterization is about. Extending them is exactly the B1 fix, so this
    assertion is the tripwire: it fails the moment another head is corrected.

    ⚑ WENT 2 -> 3 with `m_sf_policy_floor` (the SF-approved-move floor), then
    3 -> 4 with `m_sf_shape` (the SF-shape conditional KL). Each is the tripwire
    doing its job, not a regression: both new heads are masked by the SAME
    `sf_p0_regret_base` tensor as `m_sf_own_regret` and divide by the SAME
    `sf_own_regret_rows` count, so they are row-weighted by construction and were
    never on the per-batch-mean path this file characterizes. Nothing that
    existed before moved: the two original names below are still here.
    """
    loss_heads = {k for k in _RATIO_METRIC_FIELDS if k.startswith("m_")}
    assert loss_heads == {
        "m_sf_own", "m_sf_own_regret", "m_sf_policy_floor", "m_sf_shape",
    }
    # The heads this file says are STILL wrong (per-batch means) must stay wrong
    # here, or the characterization has silently stopped characterizing.
    assert "m_sf_volatility" not in _RATIO_METRIC_FIELDS


def test_exact_epoch_masked_metrics_pool_by_eligible_rows_only() -> None:
    sum_key, weight_key = _EXACT_MASKED_METRIC_FIELDS["sf_volatility_loss"]
    sums = {
        # Batch A: mean 5 over one eligible row. Batch B: mean 1 over two.
        sum_key: 5.0 * 1.0 + 1.0 * 2.0,
        weight_key: 1.0 + 2.0,
        # The legacy scalar remains present and deliberately keeps its old
        # estimator; the exact-only override replaces just the exact row.
        "sf_volatility": 5.0 + 1.0,
    }

    assert _loss_sums_to_metric_kwargs(sums, 2.0)["sf_volatility_loss"] == 3.0
    assert _exact_masked_metric_kwargs(sums)["sf_volatility_loss"] == 7.0 / 3.0


def test_exact_epoch_emits_masked_pairs_without_changing_legacy_outputs() -> None:
    normalization = dict.fromkeys(EXACT_OBJECTIVE_NAMES, 4.0)
    legacy = compute_loss(_outputs(4), _batch(4, covered=True))
    exact = compute_loss(
        _outputs(4),
        _batch(4, covered=True),
        report_exact_masked_sums=True,
        exact_corpus_rows=4,
        exact_objective_mask_weights=normalization,
    )
    missing = compute_loss(
        _outputs(4),
        _batch(4, covered=False),
        report_exact_masked_sums=True,
        exact_corpus_rows=4,
        exact_objective_mask_weights=normalization,
    )
    sum_key, weight_key = _EXACT_MASKED_METRIC_FIELDS["sf_volatility_loss"]

    assert not any(key.startswith("_exact_") for key in legacy)
    assert float(exact[weight_key]) == 4.0
    assert float(exact[sum_key]) == float(exact["sf_volatility"]) * 4.0
    assert float(missing[weight_key]) == 0.0
    assert float(missing[sum_key]) == 0.0


def test_exact_epoch_reconstructs_fractional_mask_numerator_before_pooling() -> None:
    batch = _batch(1, covered=True)
    batch["sf_wdl"] = torch.tensor([[0.125, 0.75, 0.125]])
    batch["has_sf_wdl"] = torch.ones(1)
    outputs = _outputs(1)
    outputs["sf_eval"] = torch.tensor([[1.0, -0.5, 0.25]])
    losses = compute_loss(
        outputs,
        batch,
        w_policy=0.0,
        w_soft=0.0,
        w_future=0.0,
        w_wdl=0.0,
        w_sf_move=0.0,
        w_sf_eval=1.0,
        w_categorical=0.0,
        w_volatility=0.0,
        w_sf_volatility=0.0,
        w_moves_left=0.0,
        sf_wdl_conf_power=1.0,
        report_exact_masked_sums=True,
        exact_corpus_rows=1,
        exact_objective_mask_weights={
            **dict.fromkeys(EXACT_OBJECTIVE_NAMES, 1.0),
            "sf_eval": 0.25,
        },
    )
    sum_key, weight_key = _EXACT_MASKED_METRIC_FIELDS["sf_eval_loss"]
    sums = {
        sum_key: float(losses[sum_key]),
        weight_key: float(losses[weight_key]),
    }

    assert float(losses[weight_key]) == pytest.approx(0.25)
    # masked_mean clamps a sub-unit denominator to 1.0, so the returned mean
    # already is the numerator. Re-multiplying by 0.25 would attenuate twice.
    assert float(losses[sum_key]) == pytest.approx(float(losses["sf_eval_ce"]))
    assert float(losses["total"]) == pytest.approx(
        float(losses[sum_key]) / 0.25,
    )
    assert _exact_masked_metric_kwargs(sums)["sf_eval_loss"] == pytest.approx(
        float(losses["sf_eval_ce"]) / 0.25,
    )


def test_exact_epoch_ragged_batches_keep_eligible_gradient_weight_constant() -> None:
    requested_batch_rows = 2

    def eligible_gradient(rows: int, *, exact: bool) -> float:
        logits = torch.zeros((rows, POLICY_SIZE), requires_grad=True)
        policy_t = torch.zeros((rows, POLICY_SIZE))
        policy_t[:, 0] = 1.0
        batch = {
            "x": torch.zeros((rows, 1, 8, 8)),
            "policy_t": policy_t,
            "wdl_t": torch.zeros((rows,), dtype=torch.int64),
            "has_policy": torch.tensor([1.0, *([0.0] * (rows - 1))]),
        }
        losses = compute_loss(
            {"policy_own": logits, "wdl": torch.zeros((rows, 3))},
            batch,
            w_soft=0.0,
            w_future=0.0,
            w_wdl=0.0,
            w_sf_move=0.0,
            w_sf_eval=0.0,
            w_categorical=0.0,
            w_volatility=0.0,
            w_sf_volatility=0.0,
            w_moves_left=0.0,
            report_exact_masked_sums=exact,
            exact_corpus_rows=3 if exact else None,
            exact_objective_mask_weights=(
                {
                    **dict.fromkeys(EXACT_OBJECTIVE_NAMES, 3.0),
                    "policy": 2.0,
                }
                if exact else None
            ),
        )
        (losses["total"] * rows / requested_batch_rows).backward()
        assert logits.grad is not None
        return float(logits.grad[0, 0])

    full = eligible_gradient(2, exact=True)
    ragged = eligible_gradient(1, exact=True)
    legacy_full = eligible_gradient(2, exact=False)
    legacy_ragged = eligible_gradient(1, exact=False)

    assert full == pytest.approx(ragged)
    assert legacy_full == pytest.approx(2.0 * legacy_ragged)


def test_exact_epoch_preserves_each_heads_global_masked_mean_scale() -> None:
    policy_logits = torch.tensor(
        [[2.0, -1.0], [0.5, -0.5], [-1.0, 1.5]],
    )
    future_logits = torch.tensor(
        [[-0.5, 1.0], [1.0, -0.5], [0.25, -0.25]],
    )
    policy_t = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    future_t = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    has_future = torch.tensor([1.0, 0.0, 0.0])

    def run(row_slice: slice, *, exact: bool) -> torch.Tensor:
        batch = {
            "x": torch.zeros((policy_logits[row_slice].shape[0], 1, 1, 1)),
            "policy_t": policy_t[row_slice],
            "wdl_t": torch.zeros(
                policy_logits[row_slice].shape[0], dtype=torch.int64,
            ),
            "has_policy": torch.ones(policy_logits[row_slice].shape[0]),
            "future_policy_t": future_t[row_slice],
            "has_future": has_future[row_slice],
        }
        losses = compute_loss(
            {
                "policy_own": policy_logits[row_slice],
                "policy_future": future_logits[row_slice],
                "wdl": torch.zeros((policy_logits[row_slice].shape[0], 3)),
            },
            batch,
            w_policy=1.0,
            w_soft=0.0,
            w_future=1.0,
            w_wdl=0.0,
            w_sf_move=0.0,
            w_sf_eval=0.0,
            w_categorical=0.0,
            w_volatility=0.0,
            w_sf_volatility=0.0,
            w_moves_left=0.0,
            report_exact_masked_sums=exact,
            exact_corpus_rows=3 if exact else None,
            exact_objective_mask_weights=(
                {
                    **dict.fromkeys(EXACT_OBJECTIVE_NAMES, 3.0),
                    "future_policy": 1.0,
                }
                if exact else None
            ),
        )
        return losses["total"]

    configured_full_corpus_objective = run(slice(0, 3), exact=False)
    ragged_epoch_objective = (
        run(slice(0, 2), exact=True) * (2.0 / 3.0)
        + run(slice(2, 3), exact=True) * (1.0 / 3.0)
    )

    torch.testing.assert_close(
        ragged_epoch_objective, configured_full_corpus_objective,
    )
