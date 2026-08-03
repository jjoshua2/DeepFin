"""The always-on SF-label contamination column, VALUE half.

`sf_wdl` carries realized `sf_wdl_frac` 0.45 of the trained value target and,
until 2026-08-03, had no detector in EITHER direction: the always-on tripwire
(`sf_labelled_no_multipv_frac`) reads only whether the POLICY block survived,
and `sf_wdl` is attached unconditionally from `res.cp`. On the 122 shards the
2026-08-01 quarantine removed, 99.99 % of the rows that tripwire flagged still
carried a well-formed `has_sf_wdl = 1` value label (SELFPLAY_AUDIT P2).

What these tests are for, in order of what they would catch:

1. `sf_eval_pv_orphan_flags` FIRES on real poisoned rows and reads EXACTLY zero
   on real clean rows. Both fixtures below are verbatim excerpts of the two
   columns off disk (`shard_033949.zarr` from the quarantine directory,
   `shard_033130.zarr` from the pre-episode range), copied in rather than read
   from a live path so the test does not depend on a window that no longer
   exists in that state.
2. THE NEGATIVE CONTROL. A permutation that keeps each eval with its own PV
   block must leave the rate at exactly 0; a permutation that breaks the
   pairing must make it fire. A detector that only passes the first is
   `return 0.0`; one that only passes the second fires on anything.
3. It reaches `TrainMetrics` through the PRODUCTION path, including the payload
   prune that removes `sf_multipv_raw` — the flags are derived on the host
   BEFORE that prune, which is the only reason the column is not gated behind
   `sf_policy_sparse_ce` (default False, in no config file).
4. Its denominator is rows carrying all three blocks — NOT the labelled rows,
   which would let the no-PV rows the policy detector already reported dilute
   it — and a batch that cannot be measured reads UNMEASURED, not clean.
5. The live selfplay counter and the array-level function agree row for row, so
   the two sites are one measurement rather than two descriptions of one.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.replay.shard import (
    SF_EVAL_PV_CHECKED_FIELD,
    SF_EVAL_PV_ORPHAN_FIELD,
    SF_MULTIPV_RAW_COLS,
    SF_MULTIPV_RAW_MAX,
    sf_eval_pv_orphan_flags,
)
from chess_anti_engine.selfplay import stockfish_turn
from chess_anti_engine.train.losses import (
    sf_eval_pv_orphan_counts,
    sf_wdl_health_counts,
)
from chess_anti_engine.train.trainer import TrainMetrics, Trainer

POLICY_SIZE = 1858

# (cp, mate) of the top surviving MultiPV line, and of the record-level eval,
# for 48 labelled rows of `shard_033949.zarr` — one of the 122 shards the
# 2026-08-01 quarantine removed. 12 of the 48 disagree. The -32768 entries are
# `SF_CP_SENTINEL` on mate-scored rows and are MATCHES, which is why the
# comparison is on the (cp, mate) PAIR and not on cp alone.
POISONED_TOP = [
    [124, 0], [-422, 0], [-140, 0], [-133, 0], [258, 0], [-270, 0], [-38, 0],
    [-188, 0], [-182, 0], [-24, 0], [-523, 0], [0, 0], [-483, 0], [-206, 0],
    [-57, 0], [-384, 0], [-277, 0], [-52, 0], [46, 0], [-46, 0], [66, 0],
    [-43, 0], [-64, 0], [-53, 0], [75, 0], [79, 0], [53, 0], [14, 0], [-28, 0],
    [-217, 0], [-103, 0], [-375, 0], [-32768, 4], [116, 0], [-53, 0], [42, 0],
    [113, 0], [-19996, 0], [19993, 0], [149, 0], [138, 0], [-32768, -1],
    [158, 0], [730, 0], [-158, 0], [0, 0], [706, 0], [80, 0],
]
POISONED_META = [
    [271, 0], [456, 0], [41, 0], [-18, 0], [417, 0], [-237, 0], [205, 0],
    [-49, 0], [37, 0], [173, 0], [-523, 0], [0, 0], [-340, 0], [-159, 0],
    [-57, 0], [-384, 0], [-277, 0], [-52, 0], [46, 0], [-46, 0], [66, 0],
    [-43, 0], [-64, 0], [-53, 0], [75, 0], [79, 0], [53, 0], [14, 0], [-28, 0],
    [-217, 0], [-103, 0], [-375, 0], [-32768, 4], [116, 0], [-53, 0], [42, 0],
    [113, 0], [-19996, 0], [19993, 0], [149, 0], [138, 0], [-32768, -1],
    [158, 0], [730, 0], [-158, 0], [0, 0], [706, 0], [80, 0],
]
POISONED_ORPHANS = 12

# The same two columns for 48 labelled rows of `shard_033130.zarr`, which sits
# in the 33118-33387 pre-episode range — entirely before the first quarantined
# id (33388). Zero disagreements, which is the structural floor, not a small
# number that happens to round down.
CLEAN_TOP = [
    [-32768, 1], [-19, 0], [-24, 0], [-16, 0], [-106, 0], [-191, 0], [257, 0],
    [235, 0], [184, 0], [-23, 0], [125, 0], [206, 0], [-52, 0], [217, 0],
    [387, 0], [-465, 0], [-621, 0], [-652, 0], [-706, 0], [-32768, -8],
    [-32768, -7], [-32768, 4], [-30, 0], [-70, 0], [-72, 0], [-83, 0],
    [-117, 0], [-84, 0], [15, 0], [113, 0], [121, 0], [221, 0], [209, 0],
    [157, 0], [132, 0], [115, 0], [171, 0], [159, 0], [152, 0], [396, 0],
    [73, 0], [51, 0], [-16, 0], [0, 0], [0, 0], [-32, 0], [9, 0], [-21, 0],
]
CLEAN_META = list(CLEAN_TOP)


def _arrs(
    top: list[list[int]],
    meta_cp_mate: list[list[int]],
    *,
    has_raw: list[int] | None = None,
    has_wdl: list[int] | None = None,
    with_meta: bool = True,
) -> dict[str, np.ndarray]:
    """A host-array batch carrying the two blocks in their on-disk shapes."""
    n = len(top)
    raw = np.zeros((n, SF_MULTIPV_RAW_MAX, SF_MULTIPV_RAW_COLS), np.int16)
    raw[:, :, 0] = -1
    raw[:, 0, 0] = np.arange(n) % POLICY_SIZE
    raw[:, 0, 1] = [t[0] for t in top]
    raw[:, 0, 2] = [t[1] for t in top]
    meta = np.zeros((n, 6), np.int32)
    meta[:, 0] = 698289
    meta[:, 1] = 12
    meta[:, 2] = [m[0] for m in meta_cp_mate]
    meta[:, 3] = [m[1] for m in meta_cp_mate]
    out: dict[str, np.ndarray] = {
        "x": np.zeros((n, 175, 8, 8), np.float16),
        "policy_target": np.zeros((n, POLICY_SIZE), np.float16),
        "wdl_target": np.zeros((n,), np.int8),
        "sf_wdl": np.tile(np.asarray([0.4, 0.3, 0.3], np.float32), (n, 1)),
        "has_sf_wdl": np.asarray(
            has_wdl if has_wdl is not None else [1] * n, np.uint8,
        ),
        "sf_multipv_raw": raw,
        "has_sf_multipv_raw": np.asarray(
            has_raw if has_raw is not None else [1] * n, np.uint8,
        ),
    }
    if with_meta:
        out["sf_label_meta"] = meta
        out["has_sf_label_meta"] = np.ones((n,), np.uint8)
    return out


def _rate(arrs: dict[str, np.ndarray]) -> tuple[float, float]:
    orphan, checked = sf_eval_pv_orphan_flags(arrs)
    return float(orphan.sum()), float(checked.sum())


def _tiny_trainer(**kwargs: Any) -> Trainer:
    from chess_anti_engine.model import ModelConfig, build_model

    torch.manual_seed(0)
    model = build_model(
        ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False),
    )
    return Trainer(model, device="cpu", lr=1e-3, **kwargs)


# --------------------------------------------------------------------------
# 1. It fires on real poisoned rows and reads exactly zero on real clean rows.
# --------------------------------------------------------------------------


def test_it_fires_on_the_real_quarantined_excerpt():
    orphan, checked = _rate(_arrs(POISONED_TOP, POISONED_META))
    assert checked == len(POISONED_TOP)
    assert orphan == POISONED_ORPHANS


def test_it_reads_exactly_zero_on_the_real_pre_episode_excerpt():
    """EXACTLY zero, asserted with `==` and not a tolerance.

    The floor is structural: `res.cp` IS `pvs[0].cp` in
    `_SearchInfoAccumulator.result`, so the two stored numbers are the same
    number unless rank 1's move was dropped as illegal. A tolerance here would
    accept the "small non-zero background" reading that PR #304 had to
    retract on the policy half.
    """
    orphan, checked = _rate(_arrs(CLEAN_TOP, CLEAN_META))
    assert checked == len(CLEAN_TOP)
    assert orphan == 0.0


# --------------------------------------------------------------------------
# 2. The negative control, both directions.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_a_pairing_PRESERVING_permutation_leaves_the_clean_rate_at_zero(seed):
    """Reorder whole rows and nothing changes — the detector is not reading
    row ORDER, which is the artifact a shuffle test can manufacture."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(CLEAN_TOP))
    arrs = _arrs(
        [CLEAN_TOP[i] for i in perm], [CLEAN_META[i] for i in perm],
    )
    assert _rate(arrs) == (0.0, float(len(CLEAN_TOP)))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_a_pairing_BREAKING_shuffle_makes_the_clean_arm_fire(seed):
    """THE control this detector has to survive.

    Shuffle the record-level evals against the PV blocks — exactly what a
    desynced engine does, an eval filed against another position's candidate
    list — and the rate must move off its structural zero. A detector that
    stays at 0.0 here is measuring nothing; that it reads 0.0 on the
    un-shuffled arm and non-zero here is what separates the two.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(CLEAN_META))
    shuffled = [CLEAN_META[i] for i in perm]
    orphan, checked = _rate(_arrs(CLEAN_TOP, shuffled))
    n_moved = sum(1 for a, b in zip(CLEAN_TOP, shuffled, strict=True) if a != b)
    assert checked == len(CLEAN_TOP)
    assert orphan == n_moved, (
        "the detector must flag exactly the rows whose eval was reassigned; "
        f"{n_moved} of {len(CLEAN_TOP)} moved, it flagged {orphan}"
    )
    assert orphan > 0.5 * len(CLEAN_TOP), (
        "a random derangement of 48 evals should move most of them; a small "
        "number here means the fixture is too degenerate to control anything"
    )


# --------------------------------------------------------------------------
# 3. It survives the production path, prune included.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sparse_ce", [False, True])
def test_the_column_reaches_TrainMetrics_through_the_production_path(sparse_ce):
    """`sf_policy_sparse_ce` False is production and DROPS `sf_multipv_raw`.

    The flags are derived in `_prepare_host_arrays` BEFORE that prune, so the
    reading must be identical either way. If it were computed after — or in
    `compute_loss` off the raw block — the column would read 0.0 in production
    and non-zero only under a flag that is in no config file, which is the
    exact defect `sf_rebuild_policy_frac` has.
    """
    trainer = _tiny_trainer(sf_policy_sparse_ce=sparse_ce)
    arrs = _arrs(POISONED_TOP, POISONED_META)
    prepared = trainer._prepare_host_arrays(
        arrs, rng=np.random.default_rng(0), mirror_prob=0.0,
    )
    assert ("sf_multipv_raw" in prepared) is sparse_ce, (
        "the fixture is not exercising the prune it claims to"
    )
    assert SF_EVAL_PV_ORPHAN_FIELD in prepared
    assert SF_EVAL_PV_CHECKED_FIELD in prepared
    batch = collate_arrays(prepared, device="cpu")
    orphan, checked = sf_eval_pv_orphan_counts(batch)
    assert float(orphan) == POISONED_ORPHANS
    assert float(checked) == len(POISONED_TOP)


def test_the_metric_field_is_wired_to_the_published_dataclass():
    """The ratio actually lands on `TrainMetrics.sf_eval_pv_orphan_frac`.

    Deriving the flags and summing them is worth nothing if the pair never
    reaches the dataclass the reporter publishes — the "accepted and then
    silently ignored" shape. Drives the real reduction and reads the field.
    """
    from chess_anti_engine.train.trainer import (
        _RATIO_METRIC_FIELDS,
        _RAW_SUM_LOSS_KEYS,
        _loss_sums_to_metric_kwargs,
        _ratio_metric_kwargs,
    )

    assert _RATIO_METRIC_FIELDS["sf_eval_pv_orphan_frac"] == (
        "sf_eval_pv_orphan_rows", "sf_eval_pv_checked_rows",
    )
    # ⚑ The four numerators/denominators must be in `_RAW_SUM_LOSS_KEYS`, or the
    # accumulator weights them by batch row count before the division and the
    # ratio silently becomes a different estimator on ragged batches.
    for pair in (
        _RATIO_METRIC_FIELDS["sf_eval_pv_orphan_frac"],
        _RATIO_METRIC_FIELDS["sf_wdl_orphaned_frac"],
        _RATIO_METRIC_FIELDS["sf_wdl_degenerate_frac"],
    ):
        for key in pair:
            assert key in _RAW_SUM_LOSS_KEYS, f"{key} is not accumulated as a raw sum"

    sums = {
        "sf_eval_pv_orphan_rows": float(POISONED_ORPHANS),
        "sf_eval_pv_checked_rows": float(len(POISONED_TOP)),
        "sf_wdl_degenerate_rows": 0.0,
        "sf_wdl_orphaned_rows": 3.0,
        "sf_wdl_rows": float(len(POISONED_TOP)),
        "batch_rows": float(len(POISONED_TOP)),
    }
    required = dict.fromkeys(
        (
            "loss", "policy_loss", "soft_policy_loss", "future_policy_loss",
            "wdl_loss", "sf_move_loss", "sf_move_acc", "sf_eval_loss",
            "categorical_loss", "volatility_loss", "sf_volatility_loss",
            "moves_left_loss",
        ),
        0.0,
    )
    kwargs = _loss_sums_to_metric_kwargs(sums, 1.0)
    kwargs.update(_ratio_metric_kwargs(sums))
    metrics = TrainMetrics(**cast(Any, {**required, **kwargs}))
    assert metrics.sf_eval_pv_orphan_frac == pytest.approx(
        POISONED_ORPHANS / len(POISONED_TOP),
    )
    assert metrics.sf_eval_pv_checked_frac == pytest.approx(1.0)
    assert metrics.sf_wdl_orphaned_frac == pytest.approx(3.0 / len(POISONED_TOP))


# --------------------------------------------------------------------------
# 4. Denominators, and unmeasured != clean.
# --------------------------------------------------------------------------


def test_no_pv_rows_are_excluded_from_the_denominator_not_scored_as_clean():
    """The two detectors must not hide inside each other's denominator.

    A row with no MultiPV block is the POLICY detector's numerator and cannot
    be compared here at all. Counting it as checked-and-clean would damp this
    rate by exactly the contamination the other column already reported.
    """
    n = len(POISONED_TOP)
    has_raw = [0] * (n // 2) + [1] * (n - n // 2)
    orphan, checked = _rate(_arrs(POISONED_TOP, POISONED_META, has_raw=has_raw))
    assert checked == n - n // 2
    expected = sum(
        1 for i in range(n // 2, n) if POISONED_TOP[i] != POISONED_META[i]
    )
    assert orphan == expected


def test_a_batch_without_sf_label_meta_reads_unmeasured_not_clean():
    """`sf_label_meta` is an OPTIONAL shard field and is in no collate spec.

    A batch that never carried it must publish checked = 0 — the blind
    instrument state — rather than a perfect zero rate over a full denominator,
    which reads exactly like health.
    """
    orphan, checked = _rate(
        _arrs(POISONED_TOP, POISONED_META, with_meta=False),
    )
    assert (orphan, checked) == (0.0, 0.0)


def test_the_flag_vectors_are_batch_shaped_even_with_no_sf_fields():
    """A short column would silently reduce over a different population."""
    n = 7
    arrs = {"x": np.zeros((n, 175, 8, 8), np.float16)}
    orphan, checked = sf_eval_pv_orphan_flags(arrs)
    assert orphan.shape == (n,)
    assert checked.shape == (n,)
    assert not orphan.any()
    assert not checked.any()


def test_the_value_side_rates_do_not_go_blind_when_the_policy_field_vanishes():
    """`sf_wdl_*_frac` divide by `sf_wdl_rows`, not `sf_multipv_checked_rows`.

    Borrowing the policy pair's denominator would zero both value-side rates
    the moment `has_sf_multipv_raw` went missing — the one circumstance in
    which they are the only reading left.
    """
    arrs = _arrs(CLEAN_TOP, CLEAN_META)
    arrs["sf_wdl"][3] = np.asarray([1 / 3, 1 / 3, 1 / 3], np.float32)
    del arrs["has_sf_multipv_raw"]
    del arrs["sf_multipv_raw"]
    batch = collate_arrays(arrs, device="cpu")
    degenerate, orphaned, wdl_rows = sf_wdl_health_counts(
        batch, has_sf_wdl=batch["has_sf_wdl"],
    )
    assert float(wdl_rows) == len(CLEAN_TOP)
    assert float(degenerate) == 1.0
    assert float(orphaned) == 0.0, (
        "with no policy flag there is nothing to be orphaned FROM"
    )


def test_the_blind_spot_itself_is_counted():
    """P2: policy-flagged rows still carrying a well-formed value label."""
    n = len(CLEAN_TOP)
    arrs = _arrs(CLEAN_TOP, CLEAN_META, has_raw=[0] * 5 + [1] * (n - 5))
    batch = collate_arrays(arrs, device="cpu")
    degenerate, orphaned, wdl_rows = sf_wdl_health_counts(
        batch, has_sf_wdl=batch["has_sf_wdl"],
    )
    assert (float(degenerate), float(orphaned), float(wdl_rows)) == (0.0, 5.0, float(n))


# --------------------------------------------------------------------------
# 5. The live counter and the array function are ONE measurement.
# --------------------------------------------------------------------------


def test_the_live_counter_agrees_with_the_array_function_row_for_row():
    """Two sites, one predicate — pinned, not asserted in a docstring.

    `selfplay/stockfish_turn.py::_sf_eval_pv_orphaned` reads a live record's
    arrays and `replay/shard.py::sf_eval_pv_orphan_flags` reads a batch's; a
    drift between them would make the live health line and the training column
    disagree about the same rows, and each would look self-consistent.
    """
    arrs = _arrs(POISONED_TOP, POISONED_META)
    orphan, _checked = sf_eval_pv_orphan_flags(arrs)
    for i in range(len(POISONED_TOP)):
        rec = SimpleNamespace(
            sf_multipv_raw=arrs["sf_multipv_raw"][i],
            sf_label_meta=arrs["sf_label_meta"][i],
            sf_wdl=arrs["sf_wdl"][i],
        )
        live = stockfish_turn._sf_value_half_counts(rec)
        assert live["eval_pv_orphan"] == int(orphan[i]), (
            f"row {i}: live counter and array function disagree"
        )
        assert live["eval_pv_checked"] == 1


def test_the_two_wellformedness_predicates_use_the_same_tolerances():
    """`losses.sf_wdl_wellformed` and `stockfish_turn._sf_wdl_is_wellformed`
    are stated twice because they run on different objects. Pin them equal on
    the boundary cases, or a future edit to one silently splits the metric."""
    from chess_anti_engine.train.losses import sf_wdl_wellformed

    cases = [
        (0.4, 0.3, 0.3), (1 / 3, 1 / 3, 1 / 3), (0.34, 0.33, 0.33),
        (0.0, 0.0, 1.0), (float("nan"), 0.5, 0.5), (1.5, -0.3, -0.2),
        (0.2, 0.2, 0.2), (0.5, 0.5, 0.5),
    ]
    tensor = torch.tensor(cases, dtype=torch.float32)
    batched = sf_wdl_wellformed(tensor)
    assert batched is not None
    for i, case in enumerate(cases):
        live = stockfish_turn._sf_wdl_is_wellformed(np.asarray(case, np.float32))
        assert live == bool(batched[i] > 0.5), f"{case} disagrees between the two"
