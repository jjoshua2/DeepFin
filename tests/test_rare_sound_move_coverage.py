"""Tests for scripts/rare_sound_move_coverage.py.

The negative control is shipped here as an assertion rather than left in a
session transcript: a metric that keeps scoring well after its input
association is destroyed is measuring something structural, and the only way
that stays known is if CI re-runs it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rare_sound_move_coverage.py"
# ⚑ TRACKED TEST FIXTURES, and they live under `tests/data/` for a reason.
# They were originally committed to `data/rare_sound_move_coverage/`, which
# `.gitignore` excludes wholesale (`/data/`) as runtime output -- so they had
# to be force-added, and they sat in a directory the repo treats as
# uncommittable. Anything a test READS is a fixture, not runtime output.
# The local absolute paths their provenance carried (`/home/josh/...`, a
# `/tmp/claude-1000/<session-id>/...` scratchpad) were scrubbed to repo-relative
# form before landing: this repository is PUBLIC.
_BANK_DIR = Path(__file__).resolve().parent / "data" / "rare_sound_move_coverage"
_PRESENCE_KEYS = ("x", "legal_mask", "policy_target", "sf_p0_regret")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rare_sound_move_coverage", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rsmc = _load()


# ---------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------


def _row(
    prior: list[float],
    target: list[float],
    regret_cp: list[float],
    scored: list[bool] | None = None,
):
    n = len(prior)
    return rsmc.RowVectors(
        prior=np.asarray(prior, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        regret_cp=np.asarray(regret_cp, dtype=np.float64),
        scored=np.asarray([True] * n if scored is None else scored, dtype=bool),
        key="row0",
    )


def test_coverage_counts_exactly_the_sound_and_rare_moves() -> None:
    # 4 legal moves. Only index 1 and 2 are both sound (<=25cp) and rare
    # (prior < 0.01); index 1 is funded above the floor, index 2 is not.
    row = _row(
        prior=[0.90, 0.005, 0.004, 0.091],
        target=[0.60, 0.010, 1e-9, 0.39],
        regret_cp=[0.0, 12.0, 20.0, 300.0],
    )
    cells = rsmc.coverage_cells([row], taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,))
    assert len(cells) == 1
    assert cells[0].n_pairs == 2
    assert cells[0].n_covered == 1
    assert cells[0].coverage == pytest.approx(0.5)


def test_unscored_moves_are_never_sound() -> None:
    """A move Stockfish never scored is unknown-quality, not good-quality."""
    row = _row(
        prior=[0.90, 0.005, 0.005],
        target=[0.60, 0.20, 0.20],
        regret_cp=[0.0, 5.0, 5.0],
        scored=[True, True, False],
    )
    cells = rsmc.coverage_cells([row], taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,))
    assert cells[0].n_pairs == 1


def test_rowvectors_rejects_ragged_input() -> None:
    with pytest.raises(ValueError, match="entries"):
        rsmc.RowVectors(
            prior=np.zeros(3), target=np.zeros(2),
            regret_cp=np.zeros(3), scored=np.zeros(3, dtype=bool),
        )


def test_base_rate_is_over_all_legal_moves() -> None:
    row = _row(
        prior=[0.7, 0.1, 0.1, 0.1],
        target=[0.7, 0.3, 1e-9, 1e-9],
        regret_cp=[0.0, 1.0, 1.0, 1.0],
    )
    cells = rsmc.coverage_cells([row], taus_cp=(25.0,), rhos=(1.0,), phis=(1e-3,))
    assert cells[0].base_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Negative controls -- shipped as assertions
# ---------------------------------------------------------------------------


def _synthetic_rows(n_rows: int = 300, seed: int = 7) -> list:
    """Rows built so that BOTH shuffles have somewhere to fall.

    Twelve legal moves in four blocks:

      idx 0      sound, common (prior 0.60), FUNDED  -- the prior's own move
      idx 1,2    sound, RARE   (prior 0.005), FUNDED -- what coverage detects
      idx 3,4,5  sound, common (prior 0.10),  starved
      idx 6..11  unsound, RARE (prior 0.005),        starved

    The third block is what makes the prior shuffle informative: without a
    sound-but-unfunded move anywhere, every sound move is funded and no
    permutation of the prior can move the statistic.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        prior = np.concatenate([[0.60], np.full(2, 0.005), np.full(3, 0.10),
                                np.full(6, 0.005)])
        prior = prior / prior.sum()
        target = np.full(12, 1e-9)
        target[0] = 0.90
        target[1:3] = 0.045
        regret = np.concatenate([[0.0], [5.0, 8.0], [10.0, 12.0, 15.0],
                                 rng.uniform(400.0, 900.0, size=6)])
        rows.append(rsmc.RowVectors(
            prior=prior, target=target, regret_cp=regret,
            scored=np.ones(12, dtype=bool), key=f"row{len(rows)}",
        ))
    return rows


def test_shuffling_the_target_collapses_coverage_towards_chance() -> None:
    """THE negative control. Permuting the target within the row destroys its
    association with soundness and rarity, so coverage must fall towards
    chance. The NULL is the shuffle's own distribution (``shuffled_mean`` /
    ``shuffled_sd``); ``base_rate`` pools differently and is indicative only."""
    rows = _synthetic_rows()
    real = rsmc.coverage_cells(rows, taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,))[0]
    shuffled = rsmc.coverage_cells(
        rsmc.shuffled_rows(rows, what="target", seed=11),
        taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,),
    )[0]

    assert real.n_pairs == shuffled.n_pairs, "the shuffle must not move the denominator"
    assert real.coverage > 0.95
    # The NULL is the shuffle's own distribution, not `base_rate`. `base_rate`
    # pools over all rows while shuffled coverage is pairs-weighted over the
    # rows that HAVE a sound-and-rare move, so on real data the two differ by
    # more than any tolerance worth asserting (measured live: 0.3893 vs 0.3137,
    # a 0.0756 pooling gap that an earlier 0.06 tolerance here silently blessed).
    # Here the synthetic rows are homogeneous, so the two agree loosely -- and
    # even that is asserted only as an order-of-magnitude sanity check.
    assert shuffled.base_rate == pytest.approx(0.25)
    assert abs(shuffled.coverage - shuffled.base_rate) < 0.10
    assert real.coverage - shuffled.coverage > 0.5


def test_shuffling_the_prior_destroys_the_rarity_association() -> None:
    """``rare`` becomes a size-matched random subset of legal moves, so the
    sound-and-rare population stops being enriched for funded moves."""
    rows = _synthetic_rows()
    real = rsmc.coverage_cells(rows, taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,))[0]
    shuffled = rsmc.coverage_cells(
        rsmc.shuffled_rows(rows, what="prior", seed=13),
        taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,),
    )[0]
    # Six moves are sound; only two of them are funded. Rarity is now a coin
    # flip, so the sound-and-rare pool stops being the funded pair and coverage
    # falls towards the funded share of sound moves (3 of 6).
    assert real.coverage > 0.95
    assert shuffled.coverage < 0.75
    assert shuffled.n_pairs != real.n_pairs


def test_shuffled_rows_rejects_an_unknown_target() -> None:
    with pytest.raises(ValueError, match="must be 'target' or 'prior'"):
        rsmc.shuffled_rows(_synthetic_rows(4), what="regret", seed=1)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_point_estimate_and_widens_with_fewer_rows() -> None:
    rows = _synthetic_rows(n_rows=400, seed=3)
    # Make coverage genuinely uncertain: starve the sound rare moves in half
    # the rows, so the statistic is ~0.5 rather than pinned at 1.0.
    for i, row in enumerate(rows):
        if i % 2 == 0:
            row.target[1:3] = 1e-9
    kw = {"tau_cp": 25.0, "rho": 0.01, "phi": 1e-3}
    lo_big, hi_big, _sd = rsmc.bootstrap_ci(rows, resamples=800, seed=5, **kw)
    point = rsmc.coverage_cells(
        rows, taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,),
    )[0].coverage
    assert lo_big <= point <= hi_big
    lo_small, hi_small, _sd2 = rsmc.bootstrap_ci(rows[:40], resamples=800, seed=5, **kw)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_bootstrap_clusters_on_positions_not_on_moves() -> None:
    """⚑ The resolution claim rests on CLUSTERING, and this is its only guard.

    Replacing the cluster bootstrap with a move-level i.i.d. binomial resample
    passed all the other tests, because "the CI brackets the point estimate and
    widens with fewer rows" is true of any sane bootstrap. This one is built to
    separate them: every row is INTERNALLY PERFECTLY CORRELATED (all its
    sound-and-rare moves are funded, or none are), and half the rows are funded.
    A position-clustered bootstrap then sees an effective n of 40 rows; a
    move-level bootstrap sees 40 * 12 = 480 independent draws and reports a CI
    roughly sqrt(12) times too narrow.
    """
    rng = np.random.default_rng(17)
    rows = []
    for i in range(40):
        n = 13
        prior = np.concatenate([[0.90], np.full(n - 1, 0.10 / (n - 1))])
        funded = i % 2 == 0
        target = np.full(n, 1e-9)
        target[0] = 0.9
        if funded:
            target[1:] = 0.1 / (n - 1)   # every rare move funded, together
        regret = np.concatenate([[0.0], rng.uniform(0.0, 20.0, size=n - 1)])
        rows.append(rsmc.RowVectors(
            prior=prior, target=target, regret_cp=regret,
            scored=np.ones(n, dtype=bool), key=f"row{i}",
        ))
    kw = {"tau_cp": 25.0, "rho": 0.01, "phi": 1e-3}
    lo, hi, _sd = rsmc.bootstrap_ci(rows, resamples=4000, seed=5, **kw)
    clustered_width = hi - lo

    cell = rsmc.coverage_cells(rows, taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,))[0]
    # The i.i.d. move-level 95% interval on the SAME pooled counts.
    p = cell.coverage
    iid_width = 2 * 1.96 * math.sqrt(p * (1.0 - p) / cell.n_pairs)

    assert cell.n_pairs == 40 * 12
    assert clustered_width > 2.0 * iid_width, (
        f"clustered CI width {clustered_width:.4f} is not materially wider than "
        f"the i.i.d. move-level width {iid_width:.4f} -- the bootstrap has "
        "stopped clustering by position"
    )


def test_paired_delta_is_zero_and_tight_for_identical_arms() -> None:
    rows = _synthetic_rows(n_rows=200, seed=9)
    delta, lo, hi = rsmc.paired_delta_ci(
        rows, rows, tau_cp=25.0, rho=0.01, phi=1e-3, resamples=500, seed=4,
    )
    assert delta == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_paired_delta_rejects_unequal_arms() -> None:
    rows = _synthetic_rows(n_rows=8)
    with pytest.raises(ValueError, match="equal-length"):
        rsmc.paired_delta_ci(
            rows, rows[:4], tau_cp=25.0, rho=0.01, phi=1e-3, resamples=10, seed=1,
        )


def test_paired_delta_refuses_arms_that_describe_different_positions() -> None:
    """The promise that two arms are checked for identical ordering was, in the
    first version of this script, only a docstring: nothing read the keys."""
    rows = _synthetic_rows(n_rows=6)
    other = list(reversed(rows))
    with pytest.raises(ValueError, match="different positions"):
        rsmc.paired_delta_ci(
            rows, other, tau_cp=25.0, rho=0.01, phi=1e-3, resamples=10, seed=1,
        )


def test_paired_delta_refuses_arms_with_unset_keys() -> None:
    rows = _synthetic_rows(n_rows=4)
    keyless = [
        rsmc.RowVectors(prior=r.prior, target=r.target, regret_cp=r.regret_cp,
                        scored=r.scored)
        for r in rows
    ]
    with pytest.raises(ValueError, match="unset row keys"):
        rsmc.paired_delta_ci(
            keyless, keyless, tau_cp=25.0, rho=0.01, phi=1e-3, resamples=10, seed=1,
        )


# ---------------------------------------------------------------------------
# Shard selection and the aborts
# ---------------------------------------------------------------------------


def test_select_shards_follows_mtime_not_filename(tmp_path: Path) -> None:
    """The dead-directory trap, as a test.

    Filenames are made to sort OPPOSITE to write order, which is exactly the
    situation that made a sorted-name (or shard-index) selector read a
    four-month-dead directory and publish three wrong conclusions.
    """
    names = ["shard_000900.zarr", "shard_000500.zarr", "shard_000100.zarr"]
    for age_rank, name in enumerate(names):
        p = tmp_path / name
        p.mkdir()
        # shard_000900 oldest ... shard_000100 newest.
        os.utime(p, (1_700_000_000 + age_rank, 1_700_000_000 + age_rank))
    sel = rsmc.select_shards(str(tmp_path), 2)
    assert sel.newest_basename == "shard_000100.zarr"
    assert [os.path.basename(p) for p in sel.paths] == [
        "shard_000500.zarr", "shard_000100.zarr",
    ]
    assert sel.newest_mtime > sel.oldest_mtime
    assert "shard_000100.zarr" in sel.describe()


def test_select_shards_raises_on_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no shard_"):
        rsmc.select_shards(str(tmp_path), 4)


def test_assert_required_fields_aborts_on_a_dropped_field() -> None:
    complete = {k: object() for k in rsmc.REQUIRED_SHARD_FIELDS}
    rsmc.assert_required_fields(complete, "shard_000001.zarr")
    del complete["sf_p0_regret"]
    with pytest.raises(SystemExit, match="sf_p0_regret"):
        rsmc.assert_required_fields(complete, "shard_000001.zarr")


def test_assert_encodings_aborts_on_a_mismatch() -> None:
    attrs = {
        "input_history_encoding": "lc0_root_legacy_meta",
        "policy_encoding": "lc0_1858",
    }
    rsmc.assert_encodings(
        attrs, ck_hist="lc0_root_legacy_meta", ck_pol="lc0_1858", path="s.zarr",
    )
    with pytest.raises(SystemExit, match="input_history_encoding"):
        rsmc.assert_encodings(attrs, ck_hist="legacy", ck_pol="lc0_1858", path="s.zarr")
    with pytest.raises(SystemExit, match="policy_encoding"):
        rsmc.assert_encodings(
            attrs, ck_hist="lc0_root_legacy_meta", ck_pol="full_4672", path="s.zarr",
        )


def test_assert_population_aborts_when_the_sf_label_rate_collapses() -> None:
    healthy = rsmc.ShardReadStats(
        rows_total=1000, rows_net_policy=1000, rows_with_sf_p0=228, rows_used=228,
        field_present=dict.fromkeys(_PRESENCE_KEYS, 228),
    )
    rsmc.assert_population(healthy, min_sf_p0_rate=0.05)
    starved = rsmc.ShardReadStats(
        rows_total=1000, rows_net_policy=1000, rows_with_sf_p0=20, rows_used=20,
        field_present=dict.fromkeys(_PRESENCE_KEYS, 20),
    )
    with pytest.raises(SystemExit, match="sf_p0_regret"):
        rsmc.assert_population(starved, min_sf_p0_rate=0.05)


def test_assert_population_aborts_when_a_field_is_non_finite_on_1_percent() -> None:
    stats = rsmc.ShardReadStats(
        rows_total=1000, rows_net_policy=1000, rows_with_sf_p0=1000, rows_used=1000,
        field_present={
            "x": 1000, "legal_mask": 1000, "policy_target": 985, "sf_p0_regret": 1000,
        },
    )
    with pytest.raises(SystemExit, match="policy_target"):
        rsmc.assert_population(stats, min_sf_p0_rate=0.05)


# ---------------------------------------------------------------------------
# The SF-label conventions this script depends on
# ---------------------------------------------------------------------------


def test_scored_mask_matches_the_finalize_fill_convention() -> None:
    """``_build_sf_p0_regret_vector`` fills with ``(worst_covered + 1)/2`` and
    then overwrites the MultiPV moves, so the fill is the vector maximum."""
    from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY
    from chess_anti_engine.selfplay.finalize import _build_sf_p0_regret_vector

    # MultiPV move ids are FULL-policy ids; the stored vector is the shard's
    # compact encoding, so the three moves are named by their compact slots.
    compact = (5, 100, 900)
    full = [int(COMPACT_TO_FULL_POLICY[c]) for c in compact]
    rows = np.array(
        [[full[0], 40, 0, 0, 0], [full[1], 10, 0, 0, 0], [full[2], -260, 0, 0, 0]],
        dtype=np.int16,
    )
    vec = _build_sf_p0_regret_vector(rows, policy_encoding="lc0_1858")
    assert vec is not None
    cp = np.asarray(vec, dtype=np.float64) * rsmc.SF_OWN_REGRET_CAP_CP
    scored = rsmc._scored_mask_from_regret(np.asarray(vec, dtype=np.float64))
    assert sorted(np.nonzero(scored)[0].tolist()) == sorted(compact)
    # Best line is +40cp, so the three land at 0 / 30 / 300 cp behind it.
    assert cp[compact[0]] == pytest.approx(0.0)
    assert cp[compact[1]] == pytest.approx(30.0)
    assert cp[compact[2]] == pytest.approx(300.0)
    # The fill for everything else is (0.3 + 1)/2 = 0.65 -> 650cp, above every
    # threshold this script sweeps.
    assert cp[7] == pytest.approx(650.0)


def test_scored_mask_is_empty_when_every_move_hit_the_cap() -> None:
    reg = np.full(16, 1.0, dtype=np.float64)
    assert not rsmc._scored_mask_from_regret(reg).any()


def test_sim_shape_resolves_the_root_sentinels() -> None:
    """The sentinel trio must fall back to the DESCENT knobs (linear root),
    which is what live selfplay actually runs."""
    live = rsmc.SimShape(c_scale=0.025, policy_temp=1.0)
    assert live.root_sigma_span(59.0) == pytest.approx(0.025 * (50.0 + 59.0))
    louder = rsmc.SimShape(c_scale=0.1, policy_temp=1.0)
    assert louder.root_sigma_span(59.0) == pytest.approx(0.1 * (50.0 + 59.0))
    play = rsmc.SimShape(
        c_scale=0.025, policy_temp=1.0,
        c_visit_root=900.0, c_scale_root=7.0, q_visit_exp_root=-1.0,
    )
    assert play.root_sigma_span(59.0) == pytest.approx(7.0 * np.log1p(900.0 + 59.0))


def test_parser_requires_a_replay_dir_for_shard_mode() -> None:
    with pytest.raises(SystemExit, match="requires --replay-dir"):
        rsmc.main(["--mode", "shards", "--checkpoint", "x.pt"])


# ---------------------------------------------------------------------------
# The control is attached to EVERY cell, and cells can INVERT
# ---------------------------------------------------------------------------


def test_attach_controls_marks_a_good_cell_pass_and_an_inverted_cell_inverted() -> None:
    """⚑ The failure that produced this test: the control was run at one cell
    and the pin was placed at another.

    The synthetic rows fund the sound-rare moves at 0.045, so a floor BELOW
    that finds them (PASS) and a floor ABOVE it cannot (the shuffle, which can
    land the row's 0.90 mass on a rare move, then scores higher -- INVERTED).
    """
    rows = _synthetic_rows(n_rows=200, seed=5)
    cells = rsmc.coverage_cells(
        rows, taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3, 1e-1),
    )
    rsmc.attach_controls(rows, cells, seeds=8, seed0=3, resamples=300)

    good, bad = cells[0], cells[1]
    assert good.control_seeds == 8
    assert good.shuffled_sd > 0.0
    assert good.passes_control()
    assert good.verdict() == "PASS"
    assert good.control_margin > 1.0
    assert not bad.passes_control()
    assert bad.verdict() == "INVERTED"
    assert bad.control_margin < -1.0, (
        "the high floor must INVERT: the metric scores below its own shuffle"
    )
    for cell in (good, bad):
        assert cell.margin_ci_lo < cell.control_margin < cell.margin_ci_hi, (
            "the margin's own interval must bracket the point it is read from"
        )


def test_a_control_margin_without_an_interval_is_never_a_pass() -> None:
    """⚑ THE VERDICT MUST HAVE RESOLUTION. A reviewer row-resampled the cell the
    ledger had pinned and got margin +2.06 +/- 1.39 with 22% of draws FAILING
    the >= +1.0 gate it had been stamped PASS on. Absence of an interval is
    ``no-res`` -- unknown -- and must never read as a pass.
    """
    rows = _synthetic_rows(n_rows=200, seed=5)
    cells = rsmc.coverage_cells(rows, taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,))
    rsmc.attach_controls(rows, cells, seeds=8, seed0=3, resamples=0)
    cell = cells[0]
    assert cell.control_margin > 1.0, "the POINT estimate clears the old gate"
    assert not math.isfinite(cell.margin_ci_lo)
    assert cell.verdict() == "no-res"
    assert not cell.passes_control()


def _mixed_rows(n_rows: int, frac_covered: float, seed: int) -> list:
    """Like ``_synthetic_rows`` but only ``frac_covered`` of rows fund the rare
    sound moves, so coverage VARIES across positions and the control margin
    acquires a real sampling distribution."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_rows):
        prior = np.concatenate([[0.60], np.full(2, 0.005), np.full(3, 0.10),
                                np.full(6, 0.005)])
        prior = prior / prior.sum()
        target = np.full(12, 1e-9)
        target[0] = 0.90
        if rng.random() < frac_covered:
            target[1:3] = 0.045
        else:
            target[3:5] = 0.045
        regret = np.concatenate([[0.0], [5.0, 8.0], [10.0, 12.0, 15.0],
                                 rng.uniform(400.0, 900.0, size=6)])
        out.append(rsmc.RowVectors(
            prior=prior, target=target, regret_cp=regret,
            scored=np.ones(12, dtype=bool), key=f"mix{i}",
        ))
    return out


def test_a_point_margin_over_the_gate_is_null_when_the_interval_straddles_it() -> None:
    """⚑ THE EXACT SHAPE OF THE DEFECT THIS REPLACES.

    On the live rig the pinned cell's margin was +2.06 with a row-resampled 95%
    interval of [-0.06, +5.18]; it was stamped PASS off the point. Here 12 rows
    give point margin ~+1.8 -- clearing the old `>= 1.0` gate outright -- with
    an interval that runs NEGATIVE. The verdict must be `null`, not `PASS`.
    """
    rows = _mixed_rows(12, 0.45, seed=3)
    cells = rsmc.coverage_cells(rows, taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,))
    rsmc.attach_controls(rows, cells, seeds=8, seed0=3, resamples=800)
    cell = cells[0]
    assert cell.control_margin > 1.0, "the point estimate clears the old gate"
    assert cell.margin_ci_lo < 1.0 < cell.margin_ci_hi
    assert cell.verdict() == "null"
    assert not cell.passes_control()


def test_attach_controls_is_a_no_op_at_zero_seeds() -> None:
    rows = _synthetic_rows(n_rows=10)
    cells = rsmc.coverage_cells(rows, taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,))
    rsmc.attach_controls(rows, cells, seeds=0, seed0=1)
    assert cells[0].control_seeds == 0
    assert not cells[0].passes_control()
    assert not math.isfinite(cells[0].control_margin)


# ---------------------------------------------------------------------------
# diff_focus: the population moves with the knob
# ---------------------------------------------------------------------------


def test_read_diff_focus_returns_the_last_row_with_the_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "progress.csv"
    csv_path.write_text(
        "training_iteration,diff_focus_keep_rate,diff_focus_keep_limited_frac,"
        "diff_focus_keep_prob_mean\n"
        "700,0.7962,0.3874,0.7959\n"
        "701,0.8100,0.3500,0.8090\n",
        encoding="utf-8",
    )
    got = rsmc.read_diff_focus(str(csv_path))
    assert got["training_iteration"] == pytest.approx(701.0)
    assert got["diff_focus_keep_rate"] == pytest.approx(0.8100)
    assert got["diff_focus_keep_limited_frac"] == pytest.approx(0.3500)


def test_read_diff_focus_reports_unmeasured_rather_than_guessing() -> None:
    assert rsmc.read_diff_focus("/nonexistent/progress.csv") == {}


# ---------------------------------------------------------------------------
# CLI guards
# ---------------------------------------------------------------------------


def test_main_rejects_a_tau_at_or_above_the_regret_cap() -> None:
    """At tau >= the 1000cp cap the `scored` mask is wrong, because a capped
    move is indistinguishable from the unscored fill."""
    with pytest.raises(SystemExit, match="SF_OWN_REGRET_CAP_CP"):
        rsmc.main([
            "--mode", "shards", "--checkpoint", "x.pt",
            "--replay-dir", "/tmp/nope", "--taus", "25,1000",
        ])


def test_main_requires_a_replay_dir_for_research_mode() -> None:
    with pytest.raises(SystemExit, match="requires --replay-dir"):
        rsmc.main(["--mode", "research", "--checkpoint", "x.pt"])


def test_load_dump_round_trips_keys(tmp_path: Path) -> None:
    rows = _synthetic_rows(n_rows=3)
    payload = {
        "provenance": {"checkpoint": "c.pt"},
        "per_row": [
            {
                "key": r.key, "prior": r.prior.tolist(), "target": r.target.tolist(),
                "regret_cp": r.regret_cp.tolist(),
                "scored": r.scored.astype(int).tolist(),
            }
            for r in rows
        ],
    }
    p = tmp_path / "arm.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    back, prov = rsmc.load_dump(p)
    assert [r.key for r in back] == [r.key for r in rows]
    assert prov["checkpoint"] == "c.pt"
    rsmc.assert_paired(rows, back)


# ---------------------------------------------------------------------------
# The fidelity gate ABORTS. It is not a printout.
# ---------------------------------------------------------------------------


def _fid(n: int, argmax_rate: float, tv: float):
    return rsmc.ResearchFidelity(
        n=n, argmax_agree=round(argmax_rate * n), tv_sum=tv * n,
    )


def test_fidelity_gate_passes_the_measured_production_shape() -> None:
    """Arm A of `live_arms_20260809.json`: 488/600 argmax, TV 85.361/600."""
    _fid(600, 488 / 600, 85.36143494759395 / 600).assert_within(
        rsmc.FidelityTolerance(), is_production_shape=True,
    )


def test_fidelity_gate_passes_the_widest_legitimate_arm_at_the_floor() -> None:
    """Bundle arm E (c_scale 0.1, T 1.5) is the widest shape anyone runs:
    446/600 argmax, TV 125.633/600. It must clear the FLOOR tier and it must
    NOT be held to the production tier -- its disagreement is the arm."""
    e = _fid(600, 446 / 600, 125.63256437134343 / 600)
    e.assert_within(rsmc.FidelityTolerance(), is_production_shape=False)
    with pytest.raises(SystemExit, match="FIDELITY GATE FAILED"):
        e.assert_within(rsmc.FidelityTolerance(), is_production_shape=True)


def test_fidelity_gate_fires_on_the_broken_shape() -> None:
    """⚑ THE MUTANT THIS GATE EXISTS FOR. At `sims 2 / topk 2 / c_scale 5.0 /
    T 8.0 / gumbel_scale 4.0` the harness measured argmax 0.5400 / TV 0.7245
    and the old code printed `PASS +19.57` and exited 0 -- the broken shape
    beat the honest one's +1.56, because the shuffle control is blind to
    whether the harness is searching production's search at all.
    """
    broken = _fid(200, 0.5400, 0.7245)
    with pytest.raises(SystemExit) as exc:
        broken.assert_within(rsmc.FidelityTolerance(), is_production_shape=False)
    msg = str(exc.value)
    assert "FIDELITY GATE FAILED" in msg
    assert "argmax agreement 0.5400" in msg
    assert "mean TV 0.7245" in msg


def test_fidelity_gate_fires_when_nothing_was_scored() -> None:
    with pytest.raises(SystemExit, match="no rows were scored"):
        rsmc.ResearchFidelity().assert_within(
            rsmc.FidelityTolerance(), is_production_shape=True,
        )


def test_the_in_repo_config_is_not_the_live_config_so_it_cannot_pin_the_constant() -> None:
    """⚑ THE TEST THIS REPLACES WOULD HAVE PINNED THE WRONG SEARCH.

    The obvious gate for `PRODUCTION_SEARCH_SHAPE` going stale is to read
    `configs/pbt2_small.yaml`. It is wrong here: the live yaml and main have
    diverged (608 of 968 keys), and the in-repo file says
    `gumbel_topk 16 / gumbel_c_scale 0.1 / gumbel_scale_after 0.0` against the
    live tree's `32 / 0.025 / 0.5`. Pinning the constant to the repo would have
    tiered every calibration run against a search nobody runs, so the
    divergence is asserted instead -- when someone reconciles the two files
    this test goes red and the constant's provenance must be revisited.
    """
    import yaml

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "pbt2_small.yaml"
    sp = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["selfplay"]
    repo_shape = rsmc.SimShape(
        c_scale=float(sp["gumbel_c_scale"]),
        policy_temp=float(sp.get("gumbel_policy_temp", 1.0)),
        topk=int(sp["gumbel_topk"]),
        gumbel_scale=float(sp["gumbel_scale_after"]),
    )
    assert not rsmc.is_production_shape(repo_shape, int(sp["mcts_simulations"])), (
        "the in-repo config now matches PRODUCTION_SEARCH_SHAPE -- either the "
        "yamls were reconciled (good: pin the constant to the file) or the "
        "constant drifted to the repo's values instead of the live tree's"
    )


def test_the_production_tier_catches_a_bundle_sized_shape_mismatch() -> None:
    """⚑ THE ONLY IN-BAND DETECTOR THAT PRODUCTION MOVED.

    Nothing in a shard records the search that produced it, so
    `PRODUCTION_SEARCH_SHAPE` going stale cannot be caught from the data. What
    catches it is the production tier: if production moves and the constant
    does not, the "calibration" run is really searching at the wrong shape, and
    its disagreement with the stored target is then the size of an ARM's.

    ⚑ IT CATCHES TWO OF THE THREE, AND THE THIRD IS A STATED BLIND SPOT.
    Banked arms, all re-searched against a target stored at c_scale 0.025 /
    T 1.0:

        B  c_scale 0.1   argmax 0.7900  TV 0.2001   caught, but only just
        E  bundle        argmax 0.7433  TV 0.2094   caught on both criteria
        D  policy_temp   argmax 0.8133  TV 0.1751   NOT CAUGHT

    A `policy_temp`-only move of production would slip past the tier. Do not
    tighten `prod_max_tv` to 0.17 to catch it on this evidence: the calibration
    itself measured 0.1423 over 600 rows and 0.1221 over an independent 200,
    so 0.17 has no headroom over the honest run and would fail on window noise.
    """
    tol = rsmc.FidelityTolerance()
    for _label, agree, tv_sum in (
        ("B: c_scale 0.1", 474, 120.08872336197832),
        ("E: bundle", 446, 125.63256437134343),
    ):
        with pytest.raises(SystemExit, match="FIDELITY GATE FAILED"):
            _fid(600, agree / 600, tv_sum / 600).assert_within(
                tol, is_production_shape=True,
            )
    # The blind spot, asserted so it stays known and cannot be quietly claimed.
    _fid(600, 488 / 600, 105.04055490960401 / 600).assert_within(
        tol, is_production_shape=True,
    )


def test_a_moved_production_shape_can_be_declared_on_the_cli() -> None:
    bundle = rsmc.SimShape(c_scale=0.1, policy_temp=1.5, topk=32, gumbel_scale=0.5)
    assert not rsmc.is_production_shape(bundle, 256)
    declared = rsmc.parse_production_shape("0.1,1.5,32,0.5,256")
    assert rsmc.is_production_shape(bundle, 256, declared=declared)
    assert not rsmc.is_production_shape(
        rsmc.SimShape(c_scale=0.025, policy_temp=1.0, topk=32, gumbel_scale=0.5),
        256, declared=declared,
    )
    with pytest.raises(SystemExit, match="exactly 5 numbers"):
        rsmc.parse_production_shape("0.1,1.5,32")


def test_is_production_shape_rejects_a_single_moved_knob() -> None:
    prod = rsmc.SimShape(c_scale=0.025, policy_temp=1.0, topk=32, gumbel_scale=0.5)
    assert rsmc.is_production_shape(prod, 256)
    assert not rsmc.is_production_shape(prod, 128)
    for field_name, value in (
        ("c_scale", 0.1), ("policy_temp", 1.5), ("topk", 16), ("gumbel_scale", 1.0),
    ):
        moved = dataclasses.replace(prod, **{field_name: value})
        assert not rsmc.is_production_shape(moved, 256), field_name


def test_an_off_production_arm_needs_a_calibration() -> None:
    arm = rsmc.SimShape(c_scale=0.1, policy_temp=1.5, topk=32, gumbel_scale=0.5)
    with pytest.raises(SystemExit, match="no calibration"):
        rsmc.assert_calibrated(
            this_shape=arm, this_sims=256, other=None, other_label="none",
        )
    # An arm paired against ANOTHER off-production arm is still uncalibrated.
    with pytest.raises(SystemExit, match="neither arm is the production shape"):
        rsmc.assert_calibrated(
            this_shape=arm, this_sims=256, other_label="armD.json",
            other={
                "shape": dataclasses.asdict(
                    rsmc.SimShape(c_scale=0.025, policy_temp=1.5, topk=32,
                                  gumbel_scale=0.5)),
                "sims": 256,
                "fidelity": {"n": 600, "argmax_agree": 488, "tv_sum": 85.0},
            },
        )
    # Production-shape arm A with a passing banked fidelity certifies it.
    rsmc.assert_calibrated(
        this_shape=arm, this_sims=256, other_label="armA.json",
        other={
            "shape": dataclasses.asdict(
                rsmc.SimShape(c_scale=0.025, policy_temp=1.0, topk=32,
                              gumbel_scale=0.5)),
            "sims": 256,
            "fidelity": {"n": 600, "argmax_agree": 488,
                         "tv_sum": 85.36143494759395},
        },
    )


def test_a_production_shape_arm_with_no_banked_fidelity_cannot_certify() -> None:
    arm = rsmc.SimShape(c_scale=0.1, policy_temp=1.0, topk=32, gumbel_scale=0.5)
    with pytest.raises(SystemExit, match="banked no fidelity"):
        rsmc.assert_calibrated(
            this_shape=arm, this_sims=256, other_label="armA.json",
            other={
                "shape": dataclasses.asdict(
                    rsmc.SimShape(c_scale=0.025, policy_temp=1.0, topk=32,
                                  gumbel_scale=0.5)),
                "sims": 256,
            },
        )


# ---------------------------------------------------------------------------
# Producing-net provenance: ABSENT must not read as "fine"
# ---------------------------------------------------------------------------


def test_absent_producing_net_provenance_refuses_the_comparison() -> None:
    """⚑ Measured on the live shards 2026-08-09: `model_step` and
    `model_sha256` are None on every one, because
    `DiskReplayBuffer._flush_shard_arrays` writes no `ShardMeta`. The previous
    guard read `model_steps == []` and passed silently."""
    stats = rsmc.ShardReadStats(model_steps=[])
    with pytest.raises(SystemExit, match="producing net is UNKNOWN"):
        rsmc.assert_same_producing_net(
            stats, {"read_stats": {"model_steps": []}}, allow_missing=False,
        )


def test_absent_provenance_can_be_overridden_only_explicitly(capsys) -> None:
    stats = rsmc.ShardReadStats(model_steps=[])
    rsmc.assert_same_producing_net(
        stats, {"read_stats": {"model_steps": []}}, allow_missing=True,
    )
    assert "UNVERIFIABLE" in capsys.readouterr().out


def test_different_producing_nets_are_refused() -> None:
    with pytest.raises(SystemExit, match="different nets"):
        rsmc.assert_same_producing_net(
            rsmc.ShardReadStats(model_steps=[41, 41]),
            {"read_stats": {"model_steps": [40]}}, allow_missing=False,
        )


def test_the_same_producing_net_is_accepted() -> None:
    rsmc.assert_same_producing_net(
        rsmc.ShardReadStats(model_steps=[42, 42]),
        {"read_stats": {"model_steps": [42]}}, allow_missing=False,
    )


# ---------------------------------------------------------------------------
# The attribution scan: the criterion is in code, not in prose
# ---------------------------------------------------------------------------


def _scan_cell(tau, rho, phi, *, cov, sh_mean, sh_sd, pairs=100):
    return rsmc.CoverageCell(
        tau_cp=tau, rho=rho, phi=phi, n_pairs=pairs, n_covered=int(cov * pairs),
        n_rows=pairs, coverage=cov, base_rate=0.3, shuffled_mean=sh_mean,
        shuffled_sd=sh_sd, control_seeds=8,
    )


def test_attribution_scan_counts_by_the_stated_criterion() -> None:
    cells = [
        # passes control (margin +5), quiet (|d_c| = 0.1 * |d_T|)  -> BOTH
        _scan_cell(25.0, 0.01, 1e-3, cov=0.60, sh_mean=0.50, sh_sd=0.02),
        # passes control, NOT quiet (ratio 1.0)
        _scan_cell(25.0, 0.01, 1e-2, cov=0.60, sh_mean=0.50, sh_sd=0.02),
        # INVERTED, quiet
        _scan_cell(50.0, 0.01, 1e-2, cov=0.40, sh_mean=0.50, sh_sd=0.02),
    ]
    deltas_c = {(25.0, 0.01, 1e-3): -0.02, (25.0, 0.01, 1e-2): -0.20,
                (50.0, 0.01, 1e-2): 0.01}
    deltas_t = {(25.0, 0.01, 1e-3): 0.20, (25.0, 0.01, 1e-2): 0.20,
                (50.0, 0.01, 1e-2): 0.20}
    scan = rsmc.attribution_scan(cells, deltas_c=deltas_c, deltas_t=deltas_t)
    assert scan.n_cells == 3
    assert scan.n_pass_point == 2
    assert scan.n_quiet == 2
    assert scan.n_both_point == 1
    assert scan.both_point == [(25.0, 0.01, 1e-3)]
    # No margin intervals were attached, so nothing can be stamped PASS.
    assert scan.n_pass_ci == 0
    assert scan.n_both_ci == 0


def test_attribution_scan_criterion_moves_the_answer() -> None:
    """The reviewer's point: five plausible definitions gave 9/37/39/64/94
    quiet cells. The ratio is a PARAMETER, and the count moves with it."""
    cells = [_scan_cell(25.0, 0.01, 1e-3, cov=0.60, sh_mean=0.50, sh_sd=0.02)]
    d_c = {(25.0, 0.01, 1e-3): -0.08}
    d_t = {(25.0, 0.01, 1e-3): 0.20}
    tight = rsmc.attribution_scan(cells, deltas_c=d_c, deltas_t=d_t, quiet_ratio=0.25)
    loose = rsmc.attribution_scan(cells, deltas_c=d_c, deltas_t=d_t, quiet_ratio=0.50)
    assert tight.n_quiet == 0
    assert loose.n_quiet == 1


def test_scan_bank_reproduces_the_banked_headline() -> None:
    """⚑ THE RETRACTED HEADLINE, RECOMPUTED FROM THE ARTIFACT.

    "81 pass the control, 39 are c_scale-quiet, 2 are both" was prose. This is
    the command. `corr = +0.785` between the two selection criteria is why the
    2 survivors were never a regime: the criteria are one variable.
    """
    bank = _BANK_DIR / "live_arms_20260809.json"
    scan = rsmc.scan_bank(
        bank, ref_arm="A", c_arm="B", t_arm="D", quiet_ratio=0.25, min_pairs=50,
    )
    assert scan.n_cells == 128
    assert scan.n_pass_point == 81
    assert scan.n_quiet == 39
    assert scan.n_both_point == 2
    assert scan.both_point == [(25.0, 0.05, 0.02), (50.0, 0.05, 0.02)]
    assert scan.corr_margin_ratio == pytest.approx(0.785, abs=0.005)
    # The bank predates the control interval, so it carries NO resolution and
    # not one of its 81 "passes" can be reproduced as a pass.
    assert scan.n_pass_ci == 0


# ---------------------------------------------------------------------------
# Pinned row keys: a banked arm stays recomputable after the window rolls
# ---------------------------------------------------------------------------


def test_load_row_keys_accepts_both_bank_layouts(tmp_path: Path) -> None:
    a = tmp_path / "keys.json"
    a.write_text(json.dumps({"row_keys": ["s1.zarr:3", "s1.zarr:9"]}), encoding="utf-8")
    assert rsmc.load_row_keys(a) == ["s1.zarr:3", "s1.zarr:9"]
    b = tmp_path / "dump.json"
    b.write_text(
        json.dumps({"per_row": [{"key": "s2.zarr:0"}, {"key": "s2.zarr:1"}]}),
        encoding="utf-8",
    )
    assert rsmc.load_row_keys(b) == ["s2.zarr:0", "s2.zarr:1"]


def test_load_row_keys_refuses_a_dump_with_no_keys(tmp_path: Path) -> None:
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"cells": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="carries no row keys"):
        rsmc.load_row_keys(p)


def test_the_banked_arms_carry_reproducible_row_keys() -> None:
    bank = _BANK_DIR / "live_arms_20260809.json"
    keys = rsmc.load_row_keys(bank)
    assert len(keys) == 600
    assert len(set(keys)) == 600
    assert all(":" in k and k.endswith(tuple("0123456789")) for k in keys)


# ---------------------------------------------------------------------------
# The banked calibration is recomputable, not just readable
# ---------------------------------------------------------------------------




def test_the_banked_calibration_recomputes_from_its_own_per_row() -> None:
    """⚑ "reproducible from the banked keys" has to be TRUE, not claimed.

    The 600-row arm bank carries only summary cells: a reviewer could re-read
    its arithmetic and nothing more. This dump carries `per_row`, so every cell
    it reports is recomputed here from the vectors and must match exactly.
    """
    dump = json.loads(
        (_BANK_DIR / "calibration_20260809_prod_shape.json").read_text(encoding="utf-8")
    )
    rows, prov = rsmc.load_dump(_BANK_DIR / "calibration_20260809_prod_shape.json")
    assert len(rows) == dump["n_rows"] == 200
    assert [r.key for r in rows] == dump["row_keys"]
    assert prov["is_production_shape"] is True
    for banked in dump["cells"]:
        again = rsmc.coverage_cells(
            rows, taus_cp=(banked["tau_cp"],), rhos=(banked["rho"],),
            phis=(banked["phi"],),
        )[0]
        assert again.n_pairs == banked["n_pairs"]
        assert again.n_covered == banked["n_covered"]
        assert again.coverage == pytest.approx(banked["coverage"])


def test_the_calibration_dump_cleared_the_production_fidelity_tier() -> None:
    dump = json.loads(
        (_BANK_DIR / "calibration_20260809_prod_shape.json").read_text(encoding="utf-8")
    )
    fid = rsmc.ResearchFidelity(**dump["provenance"]["fidelity"])
    assert fid.n == 200
    assert fid.argmax_rate == pytest.approx(0.8200)
    assert fid.mean_tv == pytest.approx(0.1500, abs=5e-5)
    fid.assert_within(rsmc.FidelityTolerance(), is_production_shape=True)


def test_gumbel_scale_is_inert_on_the_frozen_pinned_window() -> None:
    """⚑ B5 WAS WIRED CORRECTLY AND CHANGES NOTHING -- recorded so nobody
    re-does it. Same 8-shard window, the SAME 200 rows pinned by `--row-keys`,
    `gumbel_scale` 0.5 vs 1.0."""
    a = json.loads(
        (_BANK_DIR / "calibration_20260809_prod_shape.json").read_text(encoding="utf-8")
    )
    b = json.loads(
        (_BANK_DIR / "gumbel_scale_1p0_20260809.json").read_text(encoding="utf-8")
    )
    assert a["row_keys"] == b["row_keys"], "the two arms must be the same rows"
    assert a["provenance"]["shape"]["gumbel_scale"] == 0.5
    assert b["provenance"]["shape"]["gumbel_scale"] == 1.0
    deltas = {(r["tau_cp"], r["rho"], r["phi"]): r["delta"] for r in b["paired"]}
    assert len(deltas) == 16
    assert max(abs(v) for v in deltas.values()) <= 0.0123 + 1e-9
    assert deltas[(50.0, 0.05, 0.02)] == 0.0, (
        "the cell the retracted rule pinned moves by exactly nothing"
    )


def test_the_retracted_survivor_cell_is_null_under_the_shipped_verdict() -> None:
    """⚑ THE RETRACTION, ASSERTED. The one-sided rule was pinned at
    `tau=50 rho=0.05 phi=2e-2` on a PASS taken from a point margin. With the
    interval the instrument now ships, that cell is `null` -- and no cell of
    the fresh run is both control-PASS and `c_scale`-quiet."""
    dump = json.loads(
        (_BANK_DIR / "calibration_20260809_prod_shape.json").read_text(encoding="utf-8")
    )
    cells = [
        rsmc.CoverageCell(**{k: v for k, v in c.items()
                             if k in rsmc.CoverageCell.__annotations__})
        for c in dump["cells"]
    ]
    pinned = next(
        c for c in cells
        if (c.tau_cp, c.rho, c.phi) == (50.0, 0.05, 0.02)
    )
    assert pinned.control_margin > 1.0, "the point estimate still clears the old gate"
    assert pinned.margin_ci_lo < 1.0
    assert pinned.verdict() == "null"

    bank = json.loads((_BANK_DIR / "live_arms_20260809.json").read_text(encoding="utf-8"))

    def _d(label: str) -> dict:
        return {
            (float(r["tau_cp"]), float(r["rho"]), float(r["phi"])): float(r["delta"])
            for r in bank["paired_vs_A"][label]
        }

    scan = rsmc.attribution_scan(cells, deltas_c=_d("B"), deltas_t=_d("D"))
    assert scan.n_both_point == 2, "the old point-estimate gate found two"
    assert scan.n_both_ci == 0, "the shipped interval gate finds none"
    assert scan.corr_margin_ratio > 0.7, (
        "the two selection criteria remain one variable on a fresh window"
    )


# ---------------------------------------------------------------------------
# The hole that produced the inverted pin
# ---------------------------------------------------------------------------


# No return annotation, matching `_row` above: `rsmc` is loaded from a path at
# import time, so `rsmc.RowVectors` is not a resolvable type expression.
def _graded_rows(n: int = 240, n_legal: int = 28):
    """Rows whose target funds sound moves, so coverage has real signal."""
    rng = np.random.default_rng(20260810)
    rows = []
    for i in range(n):
        prior = rng.dirichlet(np.full(n_legal, 0.35))
        regret = np.abs(rng.normal(0.0, 60.0, n_legal))
        regret[int(rng.integers(0, n_legal))] = 0.0
        target = np.exp(-regret / 50.0) * rng.random(n_legal)
        rows.append(rsmc.RowVectors(
            prior=prior, target=target / target.sum(), regret_cp=regret,
            scored=rng.random(n_legal) < 0.9, key=f"k{i}",
        ))
    return rows


def test_the_control_is_attached_to_every_cell_of_the_grid() -> None:
    """⚑ THE HOLE THAT PRODUCED THE UNSAFE PIN, CLOSED AS A TEST.

    The `phi = 1e-2` cell was pinned as a PASS while its own control INVERTED,
    because the control had only ever been evaluated at `phi = 1e-3` and assumed
    to hold across the sweep. It does not: the association inverts as the mass
    floor rises. So the invariant is not "a control exists somewhere" but "every
    cell that gets printed carries its OWN null and its OWN interval", and a
    cell that has neither can never read PASS.
    """
    rows = _graded_rows()
    taus, rhos = (25.0, 50.0), (0.01, 0.05)
    phis = (1e-4, 1e-3, 3e-3, 1e-2, 2e-2)
    cells = rsmc.coverage_cells(rows, taus_cp=taus, rhos=rhos, phis=phis)
    assert len(cells) == len(taus) * len(rhos) * len(phis)

    rsmc.attach_controls(rows, cells, seeds=8, seed0=5, resamples=250)
    for cell in cells:
        where = f"tau={cell.tau_cp} rho={cell.rho} phi={cell.phi}"
        assert cell.control_seeds > 0, f"{where} was printed with NO null"
        assert math.isfinite(cell.shuffled_mean), f"{where} has no shuffled mean"
        assert math.isfinite(cell.margin_ci_lo), f"{where} has no control interval"
        assert math.isfinite(cell.margin_ci_hi), f"{where} has no control interval"
        assert cell.verdict() not in ("no-ctrl", "no-res"), where

    # Every phi in the sweep is represented, so a control read at one floor can
    # never stand in for another.
    assert {c.phi for c in cells} == set(phis)


def test_a_cell_never_reads_pass_without_its_own_resolved_control() -> None:
    """A cell with no interval is `no-res`, and `no-res` is not a PASS.

    Complements the test above: the grid-wide attachment is only protective if
    an unattached cell is also unusable.
    """
    rows = _graded_rows(n=120)
    cells = rsmc.coverage_cells(rows, taus_cp=(50.0,), rhos=(0.05,),
                                phis=(1e-4, 1e-3, 1e-2))
    rsmc.attach_controls(rows, cells, seeds=8, seed0=5, resamples=0)
    for cell in cells:
        assert not math.isfinite(cell.margin_ci_lo)
        assert cell.verdict() == "no-res"
        assert not cell.passes_control()


def test_shuffle_refuses_to_be_differenced_against_an_unshuffled_arm(
    tmp_path: Path,
) -> None:
    """⚑ A shuffled run is a NULL, not an arm.

    `--shuffle` replaced `rows` but left `stored_rows` and any `--compare-to`
    bank unshuffled, so the harness-bias table -- printed under the heading
    "PURE HARNESS ERROR" -- would have priced the PERMUTATION, and a paired
    delta would have priced it and called it a knob effect. Refused now, because
    a mislabelled number in a banked table outlives the caveat beside it.
    """
    bank = tmp_path / "arm_a.json"
    bank.write_text(json.dumps({"provenance": {"checkpoint": "c.pt"}, "per_row": []}),
                    encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        rsmc.main([
            "--mode", "shards", "--checkpoint", "c.pt",
            "--replay-dir", str(tmp_path), "--shuffle", "target",
            "--compare-to", str(bank),
        ])
    assert "--shuffle" in str(exc.value)
    assert "permutation" in str(exc.value)


def test_the_unsafe_pinned_cell_is_named_as_unsafe_in_the_docstring() -> None:
    """The `rho=0.01, phi=1e-2` inversion must be stated where it is read.

    It is a production-safety item, not a style point: the ledger pinned that
    cell once already. Asserted against the module docstring so a rewrite that
    drops the warning fails here.
    """
    doc = rsmc.__doc__ or ""
    head = doc[:2000]
    assert "RETRACTED" in head, "the retraction must precede the claim it retracts"
    assert "UNSAFE" in head
    assert "INVERT" in head.upper()
