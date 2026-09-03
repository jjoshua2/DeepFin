"""The always-on SF-label contamination column.

`sf_labelled_no_multipv_frac` is the share of SF-LABELLED rows in the trained
batch that carry no `sf_multipv_raw` block — the Stockfish UCI desync
fingerprint, whose value on healthy data is exactly 0.000000. A former
dense-only heuristic compared the policy and WDL rebuild fractions, but
supported sparse-policy rows legitimately separate them. This detector is
unconditional and remains valid for both dense and sparse policy formats.

What these tests are for, in order of what they would catch:

1. The column FIRES on poisoned data. A detector only ever exercised on clean
   input is indistinguishable from `return 0.0`.
2. Its denominator is the SF-LABELLED rows, exact to the row. Asserting a
   fraction against `pytest.approx` on a batch where several denominators agree
   pins the disjunction, not the term, so the fixtures below are built so that
   all-rows / network-turn-rows / labelled-rows give three DIFFERENT answers and
   only one of them passes.
3. It survives the production payload prune. `sf_policy_sparse_ce` defaults
   False and drops the raw candidate block from the H2D payload; if it dropped
   the presence flag with it the column would be flag-gated, which is the whole
   defect.
4. A batch with no presence field at all reads UNMEASURED, not clean.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.train.losses import sf_multipv_presence_counts
from chess_anti_engine.train.trainer import (
    _RATIO_METRIC_FIELDS,
    _RAW_SUM_LOSS_KEYS,
    TrainMetrics,
    Trainer,
    _loss_sums_to_metric_kwargs,
)

POLICY_SIZE = 1858


def _arrs(
    *, has_sf_wdl: list[int], has_raw: list[int], is_network_turn: list[int] | None = None,
) -> dict[str, np.ndarray]:
    n = len(has_sf_wdl)
    out: dict[str, np.ndarray] = {
        "x": np.zeros((n, 175, 8, 8), np.float16),
        "policy_target": np.zeros((n, POLICY_SIZE), np.float16),
        "wdl_target": np.zeros((n,), np.int8),
        "has_sf_wdl": np.asarray(has_sf_wdl, np.uint8),
        "has_sf_multipv_raw": np.asarray(has_raw, np.uint8),
    }
    if is_network_turn is not None:
        out["is_network_turn"] = np.asarray(is_network_turn, np.uint8)
    return out


def _counts(**kwargs) -> tuple[float, float]:
    batch = collate_arrays(_arrs(**kwargs), device="cpu")
    no_pv, checked = sf_multipv_presence_counts(batch, has_sf_wdl=batch["has_sf_wdl"])
    return float(no_pv), float(checked)


def _tiny_trainer(**kwargs) -> Trainer:
    from chess_anti_engine.model import ModelConfig, build_model

    torch.manual_seed(0)
    model = build_model(ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False))
    return Trainer(model, device="cpu", lr=1e-3, **kwargs)


# --------------------------------------------------------------------------
# 1. It fires, and it fires at the size the poisoned data actually has.
# --------------------------------------------------------------------------

def test_the_column_fires_on_poisoned_rows_at_the_exact_count():
    """8 labelled rows, 3 of them stripped of their MultiPV block.

    The exact counts are asserted, not just "> 0": a numerator that counted
    every stripped row (including the two UNLABELLED ones, rows 8-9) would give
    5/8 and still be non-zero, and a denominator of all rows would give 3/10.
    Both are wrong and both pass a `> 0` assertion.
    """
    no_pv, checked = _counts(
        has_sf_wdl=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        has_raw=[1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
    )
    assert (no_pv, checked) == (3.0, 8.0)
    assert no_pv / checked == pytest.approx(0.375)


def test_healthy_rows_read_exactly_zero_not_approximately_zero():
    """The floor is EXACTLY 0.000000, which is what licenses "any non-zero is
    an incident" as an alert rule with no threshold to argue about."""
    no_pv, checked = _counts(has_sf_wdl=[1] * 16, has_raw=[1] * 16)
    assert no_pv == 0.0
    assert checked == 16.0


def test_the_quarantined_shard_ratio_is_reproduced_from_its_row_counts():
    """Anchor to the real 2026-08-01 quarantine set: 43 413 no-PV rows out of
    209 259 labelled, over 226 141 total rows.

    The two published denominators are 0.207461 (labelled) and 0.191973 (all
    rows). This column reports the FIRST. Pinning both here is what stops the
    sibling defect — `sf_rebuild_*_frac` carries those two numbers four lines
    apart in one docstring, describing different populations.
    """
    total, labelled, no_pv = 226_141, 209_259, 43_413
    has_sf_wdl = [1] * labelled + [0] * (total - labelled)
    # Distribute the stripped rows over labelled rows only.
    has_raw = [0] * no_pv + [1] * (labelled - no_pv) + [1] * (total - labelled)
    counted_no_pv = sum(1 for lab, raw in zip(has_sf_wdl, has_raw, strict=True) if lab and not raw)
    assert counted_no_pv == no_pv
    assert counted_no_pv / labelled == pytest.approx(0.207461, abs=1e-6)
    assert counted_no_pv / total == pytest.approx(0.191973, abs=1e-6)


# --------------------------------------------------------------------------
# 2. The denominator is the SF-LABELLED rows, and nothing else.
# --------------------------------------------------------------------------

def test_denominator_is_labelled_rows_not_network_turn_rows_and_not_all_rows():
    """One fixture built so that every plausible denominator gives a DIFFERENT
    answer, because a fixture where two of them collide pins nothing.

    12 rows. Rows 0-7 are SF-labelled; rows 0, 1, 2 and 8 are stripped of their
    MultiPV block; `is_network_turn` is clear on rows 6 and 7.

      * labelled rows                  -> 3 /  8 = 0.3750  (correct)
      * network-turn AND labelled      -> 3 /  6 = 0.5000
      * all rows                       -> 3 / 12 = 0.2500
      * numerator ignoring the label   -> 4 /  8 = 0.5000

    Matching `eval/value_optimism.py::sf_multipv_missing_rate` matters
    operationally: it is the gate that selected the quarantined shards, so the
    live column and the offline per-shard reading must divide by the same
    population or they cannot be compared without rescaling.
    """
    has_sf_wdl = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    has_raw = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    is_network_turn = [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    no_pv, checked = _counts(
        has_sf_wdl=has_sf_wdl, has_raw=has_raw, is_network_turn=is_network_turn,
    )
    assert (no_pv, checked) == (3.0, 8.0)
    assert no_pv / checked == 0.375

    triples = list(zip(has_sf_wdl, has_raw, is_network_turn, strict=True))
    net_labelled = [(lab, raw) for lab, raw, net in triples if lab and net]
    assert len(net_labelled) == 6
    assert sum(1 for _lab, raw in net_labelled if not raw) / 6 == 0.5
    assert no_pv / len(triples) == 0.25
    assert sum(1 for _lab, raw, _net in triples if not raw) == 4


def test_unlabelled_rows_without_a_multipv_block_are_not_contamination():
    """A row with no SF label is not supposed to carry a candidate block. If it
    entered the numerator the column would report a large steady rate on a
    perfectly healthy window and be muted within a day."""
    no_pv, checked = _counts(has_sf_wdl=[0] * 8, has_raw=[0] * 8)
    assert (no_pv, checked) == (0.0, 0.0)


# --------------------------------------------------------------------------
# 3. Unmeasured must not look like clean.
# --------------------------------------------------------------------------

def test_a_batch_without_the_presence_field_reads_unmeasured_not_clean():
    """`has_sf_multipv_raw` is an OPTIONAL shard field. With it absent both
    counts are zero, so `sf_multipv_checked_frac` goes to 0.0 and says the rate
    measured nothing — as opposed to measured-and-found-clean."""
    arrs = _arrs(has_sf_wdl=[1] * 8, has_raw=[1] * 8)
    del arrs["has_sf_multipv_raw"]
    batch = collate_arrays(arrs, device="cpu")
    assert "has_sf_multipv_raw" not in batch
    no_pv, checked = sf_multipv_presence_counts(batch, has_sf_wdl=batch["has_sf_wdl"])
    assert (float(no_pv), float(checked)) == (0.0, 0.0)

    metrics = _loss_sums_to_metric_kwargs(
        {"sf_no_multipv_rows": 0.0, "sf_multipv_checked_rows": 0.0, "batch_rows": 8.0}, 1.0,
    )
    assert metrics["sf_labelled_no_multipv_frac"] == 0.0
    assert metrics["sf_multipv_checked_frac"] == 0.0


def test_checked_frac_reports_the_labelled_share_of_the_batch():
    """The companion column's own denominator is ALL rows of the batch, so on
    the production window it sits at the SF-labelled share (~0.99), and a drop
    to 0.0 is the unambiguous "this column is blind" signal."""
    metrics = _loss_sums_to_metric_kwargs(
        {"sf_no_multipv_rows": 3.0, "sf_multipv_checked_rows": 8.0, "batch_rows": 10.0}, 1.0,
    )
    assert metrics["sf_labelled_no_multipv_frac"] == pytest.approx(0.375)
    assert metrics["sf_multipv_checked_frac"] == pytest.approx(0.8)


# --------------------------------------------------------------------------
# 4. It reaches the production path, unconditionally.
# --------------------------------------------------------------------------

def test_the_payload_prune_keeps_the_presence_flag_with_sparse_ce_off():
    """`sf_policy_sparse_ce` defaults False and prunes the raw candidate block
    from the H2D payload. It must NOT prune the presence flag with it: that
    would gate the detector behind a flag that is in no config file, which is
    exactly the `rebuild_sf_targets` defect being fixed."""
    t = _tiny_trainer()
    assert t.sf_policy_sparse_ce is False
    arrs = _arrs(has_sf_wdl=[1, 1, 1, 1], has_raw=[1, 0, 1, 1])
    arrs["sf_multipv_raw"] = np.zeros((4, 40, 4), np.int32)
    out = t._prepare_host_arrays(arrs, rng=np.random.default_rng(0), mirror_prob=0.0)
    assert "sf_multipv_raw" not in out, "the big block should still be pruned"
    assert "has_sf_multipv_raw" in out, "the detector's input must survive the prune"
    np.testing.assert_array_equal(out["has_sf_multipv_raw"], [1, 0, 1, 1])


def test_adding_the_presence_flag_does_not_move_the_loss():
    """The holdout is a FROZEN ruler, and this change edits a frame the ruler
    id covers, so the id moves and an operator will see one best-model
    handover. That is only acceptable if the MEASUREMENT did not move.

    Proved here rather than argued: `compute_loss` is run on the same batch
    with and without `has_sf_multipv_raw`, and every scalar it returns — every
    loss term, not just `total` — must be bitwise equal. `has_sf_multipv_raw`
    reaches exactly one loss term, `sparse_sf_policy_ce`, which needs
    `sf_multipv_raw` as well (still pruned) and is reached only when
    `sf_policy_sparse_ce` is on (a config where nothing was pruned anyway).
    """
    from chess_anti_engine.train import losses as losses_mod

    n = 6
    arrs = _arrs(has_sf_wdl=[1, 1, 1, 1, 0, 0], has_raw=[1, 0, 1, 0, 1, 1])
    rng = np.random.default_rng(7)
    arrs["policy_target"] = rng.random((n, POLICY_SIZE)).astype(np.float16)
    arrs["wdl_target"] = np.array([0, 1, 2, 0, 1, 2], np.int8)
    arrs["sf_wdl"] = rng.random((n, 3)).astype(np.float16)

    torch.manual_seed(3)
    outputs = {
        "policy": torch.randn(n, POLICY_SIZE),
        "wdl": torch.randn(n, 3),
        "sf_eval": torch.randn(n, 3),
    }
    with_flag = losses_mod.compute_loss(outputs, collate_arrays(arrs, device="cpu"))
    stripped = {k: v for k, v in arrs.items() if k != "has_sf_multipv_raw"}
    without = losses_mod.compute_loss(outputs, collate_arrays(stripped, device="cpu"))

    moved = {
        k for k in with_flag
        if k not in ("sf_no_multipv_rows", "sf_multipv_checked_rows")
        and float(with_flag[k]) != float(without[k])
    }
    assert not moved, f"the presence flag moved these loss scalars: {sorted(moved)}"
    # ...and the detector itself is the only thing that changed.
    assert float(with_flag["sf_no_multipv_rows"]) == 2.0
    assert float(without["sf_no_multipv_rows"]) == 0.0


def test_the_metric_is_not_gated_on_rebuild_sf_targets():
    """Label health must read identically with the rebuild flag off and on."""
    arrs = _arrs(has_sf_wdl=[1, 1, 1, 1, 1, 1, 1, 1], has_raw=[1, 0, 0, 1, 1, 1, 1, 1])
    readings = []
    for enabled in (False, True):
        t = _tiny_trainer()
        if enabled:
            from chess_anti_engine.train.target_builder import SfTargetParams

            t.set_sf_target_rebuild(enabled=True, params=SfTargetParams())
        prepared = t._prepare_host_arrays(
            {k: np.array(v, copy=True) for k, v in arrs.items()},
            rng=np.random.default_rng(0), mirror_prob=0.0, rebuild_sf_targets=True,
        )
        batch = collate_arrays(prepared, device="cpu")
        no_pv, checked = sf_multipv_presence_counts(batch, has_sf_wdl=batch["has_sf_wdl"])
        readings.append((float(no_pv), float(checked)))
    assert readings == [(2.0, 8.0), (2.0, 8.0)]


def test_full_pass_publishes_the_columns_on_the_test_row():
    """End to end through the ruler path that produces the `test_*` twins:
    a poisoned holdout set must show up on the holdout's own row.

    ⚑ This is the ONLY fixture in this file that reaches `compute_loss` through
    the real path, so it is also the only place `sf_multipv_checked_frac`'s
    DENOMINATOR (`batch_rows`, all rows of the microbatch) is pinned end to end.
    It therefore carries an explicit `is_network_turn` so that all-rows,
    network-turn-rows and labelled-rows are three DIFFERENT numbers:

      * all rows          -> 8   (the correct denominator)
      * network-turn rows -> 4   (rows 0, 1, 4, 7)
      * labelled rows     -> 7   (row 7 is not labelled)
      * labelled AND net  -> 3   (rows 0, 1, 4)

    Without it every row is a network turn and all-rows collides with
    network-turn-rows, so swapping the denominator to `net_mask.sum()` — the
    live shards ARE all `is_network_turn`, which is exactly why the collision is
    easy to miss — left `checked_frac` at 0.875 and the whole suite green. With
    it that swap reads 7/4 = 1.75, an impossible "fraction", and this assertion
    fails naming `sf_multipv_checked_frac`.
    """

    class _SliceBuf:
        rng = np.random.default_rng(0)

        def __init__(self, arrs: dict[str, np.ndarray], n: int) -> None:
            self._arrs, self._n = arrs, n

        def __len__(self) -> int:
            return self._n

        def batch_row_bounds(self, bs: int) -> list[tuple[int, int]]:
            return [(i, min(i + bs, self._n)) for i in range(0, self._n, bs)]

        def rows_slice_arrays(self, start: int, stop: int) -> dict[str, np.ndarray]:
            return {k: np.array(v[start:stop], copy=True) for k, v in self._arrs.items()}

    t = _tiny_trainer()
    has_sf_wdl = [1, 1, 1, 1, 1, 1, 1, 0]
    has_raw = [1, 0, 1, 0, 1, 1, 1, 1]
    is_network_turn = [1, 1, 0, 0, 1, 0, 0, 1]
    arrs = _arrs(
        has_sf_wdl=has_sf_wdl, has_raw=has_raw, is_network_turn=is_network_turn,
    )
    metrics = t.eval_full_pass(cast(Any, _SliceBuf(arrs, 8)), batch_size=4)
    assert metrics.eval_rows == 8
    # 7 labelled rows (row 7 is not), 2 of them stripped (rows 1 and 3).
    assert metrics.sf_labelled_no_multipv_frac == pytest.approx(2 / 7)
    assert metrics.sf_multipv_checked_frac == pytest.approx(7 / 8)

    # The four candidate denominators, spelled out so a future edit to the
    # fixture that lets two of them collide again fails here rather than
    # silently un-pinning the assertion above.
    triples = list(zip(has_sf_wdl, has_raw, is_network_turn, strict=True))
    assert len(triples) == 8                                            # all rows
    assert sum(is_network_turn) == 4                                    # network-turn rows
    assert sum(has_sf_wdl) == 7                                         # labelled rows
    assert sum(1 for lab, _raw, net in triples if lab and net) == 3     # labelled AND net
    assert len({8, sum(is_network_turn), sum(has_sf_wdl)}) == 3


def test_the_columns_are_row_weighted_across_ragged_batches():
    """Pooled as a ratio of SUMS, not a mean of per-batch rates. The full pass
    ends on a ragged batch and the labelled count swings batch to batch, so the
    two estimators differ: here 1/1 and 1/7 average to 0.571 while the correct
    row-weighted answer is 2/8 = 0.25."""
    assert "sf_no_multipv_rows" in _RAW_SUM_LOSS_KEYS
    assert "sf_multipv_checked_rows" in _RAW_SUM_LOSS_KEYS
    assert "batch_rows" in _RAW_SUM_LOSS_KEYS
    pooled = _loss_sums_to_metric_kwargs(
        {"sf_no_multipv_rows": 1.0 + 1.0, "sf_multipv_checked_rows": 1.0 + 7.0,
         "batch_rows": 2.0 + 8.0},
        2.0,
    )
    assert pooled["sf_labelled_no_multipv_frac"] == pytest.approx(0.25)
    assert pooled["sf_labelled_no_multipv_frac"] != pytest.approx((1 / 1 + 1 / 7) / 2)


# --------------------------------------------------------------------------
# 5. It reaches progress.csv, on both rows.
# --------------------------------------------------------------------------

def test_both_columns_reach_the_train_and_test_report_rows():
    """A metric that stops at `TrainMetrics` dies with the TensorBoard event
    files at every Ray session boundary — the reason the grad-norm columns were
    promoted here. Both must appear in the reported dict, on both rows, with
    the SAME key set as the defaults so Ray's CSV header cannot drift."""
    from chess_anti_engine.tune.trainable_report import (
        _TEST_METRIC_KEYS,
        _TRAIN_METRIC_DEFAULTS,
        _test_and_drift_dict,
        _train_metrics_dict,
    )
    from chess_anti_engine.tune.trial_config import DriftMetrics, TrainingResult

    names = ("sf_labelled_no_multipv_frac", "sf_multipv_checked_frac")
    for name in names:
        assert name in _TRAIN_METRIC_DEFAULTS
        assert f"test_{name}" in _TEST_METRIC_KEYS

    # Present (as NaN) even when no eval ran, so Ray locks the column on row 1.
    empty = _test_and_drift_dict(
        tr=TrainingResult(), drift=DriftMetrics(),
        holdout_frozen=False, holdout_generation=0,
    )
    for name in names:
        assert f"test_{name}" in empty

    tm = TrainMetrics(
        loss=0.0, policy_loss=0.0, soft_policy_loss=0.0, future_policy_loss=0.0,
        wdl_loss=0.0, sf_move_loss=0.0, sf_move_acc=0.0, sf_eval_loss=0.0,
        categorical_loss=0.0, volatility_loss=0.0, sf_volatility_loss=0.0,
        moves_left_loss=0.0,
        sf_labelled_no_multipv_frac=0.5, sf_multipv_checked_frac=0.25,
    )
    scored = _test_and_drift_dict(
        tr=TrainingResult(test_metrics=tm), drift=DriftMetrics(),
        holdout_frozen=True, holdout_generation=1,
    )
    assert scored["test_sf_labelled_no_multipv_frac"] == pytest.approx(0.5)
    assert scored["test_sf_multipv_checked_frac"] == pytest.approx(0.25)

    metrics = TrainMetrics(
        loss=0.0, policy_loss=0.0, soft_policy_loss=0.0, future_policy_loss=0.0,
        wdl_loss=0.0, sf_move_loss=0.0, sf_move_acc=0.0, sf_eval_loss=0.0,
        categorical_loss=0.0, volatility_loss=0.0, sf_volatility_loss=0.0,
        moves_left_loss=0.0,
        sf_labelled_no_multipv_frac=0.207461, sf_multipv_checked_frac=0.9253,
    )
    reported = _train_metrics_dict(metrics)
    assert reported["sf_labelled_no_multipv_frac"] == pytest.approx(0.207461)
    assert reported["sf_multipv_checked_frac"] == pytest.approx(0.9253)
    # The "no train phase ran" fallback must carry the identical key set, or
    # the CSV header shifts on the first skipped iteration.
    assert set(_train_metrics_dict(None)) == set(reported)


def test_the_ratio_fields_are_declared_against_the_keys_compute_loss_emits():
    """`_RATIO_METRIC_FIELDS` is a claim about `compute_loss`'s output. If a
    numerator or denominator key were misspelled the column would silently read
    0.0 forever — the failure mode of every metric this file exists to replace."""
    from chess_anti_engine.train import losses as losses_mod

    emitted = _fake_compute_loss_keys(losses_mod)
    for field in ("sf_labelled_no_multipv_frac", "sf_multipv_checked_frac"):
        num, den = _RATIO_METRIC_FIELDS[field]
        assert num in emitted, f"{field}: numerator {num!r} is not emitted by compute_loss"
        assert den in emitted, f"{field}: denominator {den!r} is not emitted by compute_loss"
        assert field in {f.name for f in __import__("dataclasses").fields(TrainMetrics)}


# --------------------------------------------------------------------------
# 6. The probe that reads all of the above off real shards.
# --------------------------------------------------------------------------

def test_the_probe_pools_arms_by_rows_not_by_shard():
    """`scripts/sf_no_multipv_probe.py` reconstructs per-shard COUNTS out of the
    two published fractions and pools those. A mean of per-shard rates would
    weight a 40-row shard like a 2000-row one, and the quarantined set's
    per-shard rate ranges 0.013-0.31, so the two answers are far apart.

    Reproduced against the real numbers: this pooling is what turned 122
    per-shard readings into 0.207461, matching a direct count off the zarrs
    (43 413 / 209 259) to six decimals.
    """
    from tests.script_loading import load_script_module

    probe = load_script_module("sf_no_multipv_probe.py")
    arm = probe.ArmReading()
    arm = arm.add(rows=2000, no_multipv_frac=0.3, checked_frac=1.0)
    arm = arm.add(rows=100, no_multipv_frac=0.0, checked_frac=1.0)
    assert arm.no_multipv_frac == pytest.approx(600.0 / 2100.0)
    assert arm.no_multipv_frac != pytest.approx(0.15)   # the per-shard mean
    assert arm.checked_frac == pytest.approx(1.0)

    # An arm that inspected nothing must not read as clean.
    empty = probe.ArmReading().add(rows=500, no_multipv_frac=0.0, checked_frac=0.0)
    assert (empty.no_multipv_frac, empty.checked_frac) == (0.0, 0.0)


def test_the_probes_pass_bars_match_the_pre_registered_thresholds():
    """The bars live in the script so the run enforces what it was launched
    under, rather than the reader eyeballing the printout afterwards."""
    from tests.script_loading import load_script_module

    probe = load_script_module("sf_no_multipv_probe.py")
    # Pre-registered in docs/experiment_ledger.md, 2026-08-01.
    assert (probe.POISONED_MIN, probe.POISONED_MAX) == (0.19, 0.23)
    assert probe.CHECKED_MIN == 0.5
    # 0.207461 is the whole-quarantine-set truth the band was drawn around.
    assert probe.POISONED_MIN < 0.207461 < probe.POISONED_MAX


def _fake_compute_loss_keys(losses_mod) -> set[str]:
    """The key set `compute_loss` actually returns, from a real call."""
    n = 4
    batch = collate_arrays(_arrs(has_sf_wdl=[1] * n, has_raw=[1, 0, 1, 1]), device="cpu")
    outputs = {
        "policy": torch.zeros((n, POLICY_SIZE)),
        "wdl": torch.zeros((n, 3)),
    }
    return set(losses_mod.compute_loss(outputs, batch))
