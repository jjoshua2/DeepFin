"""Tests for scripts/rare_sound_move_coverage.py.

The negative control is shipped here as an assertion rather than left in a
session transcript: a metric that keeps scoring well after its input
association is destroyed is measuring something structural, and the only way
that stays known is if CI re-runs it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rare_sound_move_coverage.py"
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
            scored=np.ones(12, dtype=bool),
        ))
    return rows


def test_shuffling_the_target_collapses_coverage_to_the_base_rate() -> None:
    """THE negative control. Permuting the target within the row destroys its
    association with soundness and rarity; coverage must fall to the chance
    level ``P(target >= phi)`` over legal moves."""
    rows = _synthetic_rows()
    real = rsmc.coverage_cells(rows, taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,))[0]
    shuffled = rsmc.coverage_cells(
        rsmc.shuffled_rows(rows, what="target", seed=11),
        taus_cp=(25.0,), rhos=(0.01,), phis=(1e-3,),
    )[0]

    assert real.n_pairs == shuffled.n_pairs, "the shuffle must not move the denominator"
    assert real.coverage > 0.95
    # 3 of 12 legal moves clear the floor, so chance is 0.25.
    assert shuffled.base_rate == pytest.approx(0.25)
    assert abs(shuffled.coverage - shuffled.base_rate) < 0.06
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
    with pytest.raises(ValueError, match="equal-length arms"):
        rsmc.paired_delta_ci(
            rows, rows[:4], tau_cp=25.0, rho=0.01, phi=1e-3, resamples=10, seed=1,
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
