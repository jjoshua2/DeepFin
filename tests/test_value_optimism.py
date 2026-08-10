"""Tests for the stratified value-optimism scorer (eval/value_optimism.py).

The two that matter most are the controls. A scorer that cannot fail is not a
scorer, and a scorer that only passes its NEGATIVE control has shown nothing —
so both directions are pinned here:

- ``test_shuffle_control_kills_bucket_structure``: destroy the position <->
  net-evaluation association and the bucket structure must collapse.
- ``test_tail_asymmetry_separates_artifact_from_defect``: a symmetric
  compression must read as ~0 EXCESS OVER THE MEASURED NULL, while an injected
  one-sided optimism must not. Without this, the negative control would happily
  pass on an instrument that reports the artifact as a finding.

An earlier version of that test asserted the raw asymmetry was ~0, on a rig
whose ``sf_cp`` came from a symmetric uniform and whose outcome equalled the
ruler — which forces the null to zero by construction, so the test could not
have caught the thing it was for. ``_noisy_ruler_rows`` replaces it with an
off-centre latent truth behind a noisy ruler, where the null really is non-zero
(``test_perfect_head_reads_nonzero_on_the_ruler_axis_and_the_null_catches_it``).
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from chess_anti_engine.eval.value_optimism import (
    SF_AXIS_MIN_ROWS,
    SF_EVAL_BUCKET_EDGES,
    SF_EVAL_BUCKET_NAMES,
    SF_LABEL_ATTACHMENT_MIN,
    SF_MULTIPV_MISS_MAX,
    OptimismRows,
    bucket_names_for,
    bucket_net_score_spread,
    cp_to_expected_score,
    expected_score,
    expected_score_to_cp,
    material_balance_from_planes,
    outcome_calibration,
    perfect_head_tail_asymmetry,
    rank_corr,
    score_buckets,
    sf_eval_bucket,
    sf_eval_bucket_array,
    sf_bestmove_is_first_legal_rate,
    sf_label_attachment_corr,
    sf_multipv_missing_rate,
    tail_asymmetry,
    tail_asymmetry_ci,
)
from chess_anti_engine.stockfish.wdl import cp_to_wdl

SLOPE = 0.0060
DRAW_WIDTH = 120.0


# ---------------------------------------------------------------------------
# The cp <-> score map must be production's, not a lookalike
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp", [-2000.0, -900.0, -300.0, -120.0, 0.0, 55.0, 300.0, 1500.0])
def test_cp_to_expected_score_matches_production_cp_to_wdl(cp: float) -> None:
    want = float(expected_score(cp_to_wdl(cp, None, slope=SLOPE, draw_width_cp=DRAW_WIDTH)))
    got = float(cp_to_expected_score(np.array(cp), slope=SLOPE, draw_width_cp=DRAW_WIDTH))
    assert got == pytest.approx(want, abs=1e-6)


def test_expected_score_to_cp_round_trips() -> None:
    cp = np.array([-1400.0, -640.0, -301.0, -80.0, 0.0, 77.0, 410.0, 1400.0])
    score = cp_to_expected_score(cp, slope=SLOPE, draw_width_cp=DRAW_WIDTH)
    back, clamped = expected_score_to_cp(score, slope=SLOPE, draw_width_cp=DRAW_WIDTH)
    assert not clamped.any()
    assert back == pytest.approx(cp, abs=0.5)


def test_expected_score_to_cp_reports_its_clamp() -> None:
    """A pinned tail must be reported, never silently folded into the mean."""
    score = np.array([0.0, 1.0, 0.5])
    cp, clamped = expected_score_to_cp(score, slope=SLOPE, draw_width_cp=DRAW_WIDTH)
    assert clamped.tolist() == [True, True, False]
    assert cp[0] == pytest.approx(-1500.0)
    assert cp[1] == pytest.approx(1500.0)


def test_cp_to_expected_score_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="slope>0"):
        cp_to_expected_score(np.array(0.0), slope=0.0, draw_width_cp=DRAW_WIDTH)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def test_scalar_and_vector_bucketing_agree() -> None:
    cp = np.array([-9000.0, -300.0, -299.9, -100.0, 0.0, 100.0, 100.1, 300.0, 4000.0])
    vec = sf_eval_bucket_array(cp)
    assert vec.tolist() == [sf_eval_bucket(float(v)) for v in cp]
    # Edges are inclusive on the LOW side: exactly -300 is "lost", not "losing".
    assert sf_eval_bucket(-300.0) == 0
    assert sf_eval_bucket(-299.9) == 1
    assert len(SF_EVAL_BUCKET_NAMES) == len(SF_EVAL_BUCKET_EDGES) + 1


def test_bucket_names_track_custom_edges() -> None:
    assert bucket_names_for(SF_EVAL_BUCKET_EDGES) == SF_EVAL_BUCKET_NAMES
    names = bucket_names_for((-200.0, 0.0, 200.0))
    assert len(names) == 4
    assert names[0].startswith("<=")
    assert names[-1].startswith(">")


# ---------------------------------------------------------------------------
# Synthetic row builders
# ---------------------------------------------------------------------------


def _rows(
    *, n: int = 6000, seed: int = 7, optimism_when_losing: float = 0.0,
    compression: float = 0.0, rows_per_game: int = 6,
) -> OptimismRows:
    """Synthetic rows with a known ground truth.

    ``compression`` pulls the net's score toward 0.5 SYMMETRICALLY — the
    artifact a noisy ruler manufactures even for a perfect head.
    ``optimism_when_losing`` adds a ONE-SIDED shift on losing rows only — a
    real directional defect.
    """
    rng = np.random.default_rng(seed)
    sf_cp = rng.uniform(-1200.0, 1200.0, size=n)
    ruler = cp_to_expected_score(sf_cp, slope=SLOPE, draw_width_cp=DRAW_WIDTH)
    net = ruler + compression * (0.5 - ruler)
    net = np.where(sf_cp < 0.0, net + optimism_when_losing, net)
    net = np.clip(net + rng.normal(0.0, 0.01, size=n), 0.0, 1.0)
    game_id = np.arange(n) // int(rows_per_game)
    return OptimismRows(
        sf_cp=sf_cp,
        sf_ruler_score=ruler,
        net_score=net,
        target_score=ruler.copy(),
        outcome_score=(ruler > 0.5).astype(np.float64),
        target_sf_score=ruler.copy(),
        search_score=net.copy(),
        game_id=game_id,
        piece_count=np.full(n, 20, dtype=np.int64),
    )


def _stat(stats: list, name: str):
    return next(s for s in stats if s.name == name)


# ---------------------------------------------------------------------------
# THE NEGATIVE CONTROL
# ---------------------------------------------------------------------------


def test_shuffle_control_kills_bucket_structure() -> None:
    """Break position <-> net-eval association; the scorer must go quiet.

    The control statistic is the SPREAD of the bucket mean net score, not the
    per-bucket ``net - SF_ruler`` difference. That distinction is the whole
    point: under the shuffle the net predicts the global mean everywhere, so
    ``net - SF_ruler`` stays hugely positive in the lost bucket (the ruler
    moved, the net did not) and would read as a finding. Measured on live
    rows, the shuffled run still reports "optimistic in 94.3% of lost
    positions" — HIGHER than the real 87.2%. An optimistic-fraction headline
    is therefore not evidence of anything, and this test pins the statistic
    that is.
    """
    rows = _rows(n=20000)
    real = score_buckets(rows, slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200)
    shuffled = score_buckets(
        rows.with_shuffled_net(np.random.default_rng(0)),
        slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200,
    )

    real_spread = bucket_net_score_spread(real)
    shuffled_spread = bucket_net_score_spread(shuffled)
    assert real_spread > 0.5
    assert shuffled_spread < real_spread / 20.0

    # Bound the residual against SAMPLING NOISE rather than a magic constant:
    # under the shuffle each bucket mean is an independent draw, so the spread
    # can only be a few standard errors of the smallest bucket.
    smallest = min(s.n for s in shuffled)
    noise = 6.0 * float(rows.net_score.std()) / np.sqrt(smallest)
    assert shuffled_spread < noise

    # Every bucket's mean net score collapses onto the global mean.
    global_mean = float(rows.net_score.mean())
    for s in shuffled:
        assert s.net_score == pytest.approx(global_mean, abs=noise)

    # And the trap this test exists to document: the naive statistic does NOT
    # collapse, so it must never be the deciding one.
    assert _stat(shuffled, SF_EVAL_BUCKET_NAMES[0]).net_minus_sf > 0.3


# ---------------------------------------------------------------------------
# THE POSITIVE CONTROL — a passing negative control proves nothing alone
# ---------------------------------------------------------------------------


def _noisy_ruler_rows(
    *, n: int = 30000, seed: int = 13, optimism_when_losing: float = 0.0,
    compression: float = 0.0, target_equals_net: bool = True,
) -> OptimismRows:
    """A rig where the tail-asymmetry null is genuinely NOT zero.

    The earlier builder drew ``sf_cp`` from a symmetric uniform and set the
    outcome equal to the ruler, which forces the null to zero by construction —
    so a test written against it cannot catch a non-zero null, which is exactly
    the mistake this rig exists to prevent.

    Here the ruler is a NOISY proxy for a latent truth and the truth's mean is
    off-centre, so bucketing on the ruler really does regress each tail toward
    the mean, by DIFFERENT amounts in the two tails. ``outcome_score`` is the
    latent truth (an unbiased draw of it), which is what makes
    ``perfect_head_tail_asymmetry`` the correct null.
    """
    rng = np.random.default_rng(seed)
    true_cp = rng.normal(-80.0, 500.0, size=n)
    ruler_cp = true_cp + rng.normal(0.0, 250.0, size=n)
    true_score = cp_to_expected_score(true_cp, slope=SLOPE, draw_width_cp=DRAW_WIDTH)
    net = true_score + compression * (0.5 - true_score)
    net = np.where(ruler_cp < 0.0, net + optimism_when_losing, net)
    net = np.clip(net, 0.0, 1.0)
    target = net.copy() if target_equals_net else true_score.copy()
    return OptimismRows(
        sf_cp=ruler_cp,
        sf_ruler_score=cp_to_expected_score(ruler_cp, slope=SLOPE, draw_width_cp=DRAW_WIDTH),
        net_score=net,
        target_score=target,
        outcome_score=true_score,
        target_sf_score=true_score,
        search_score=net.copy(),
        game_id=np.arange(n) // 6,
        piece_count=np.full(n, 20, dtype=np.int64),
    )


def test_perfect_head_reads_nonzero_on_the_ruler_axis_and_the_null_catches_it() -> None:
    """The artifact is real; the null is what makes it readable as one.

    A head that predicts the truth exactly still reads non-zero against a NOISY
    ruler, and the two tails do not cancel when the truth is off-centre. The
    whole point of `perfect_head_tail_asymmetry` is to supply that offset, so
    `tail_asymmetry` is never judged against zero.
    """
    stats = score_buckets(
        _noisy_ruler_rows(), slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200,
    )
    asym = tail_asymmetry(stats)
    null = perfect_head_tail_asymmetry(stats)
    assert asym is not None
    assert null is not None
    # The null is materially away from zero — a test that assumed zero here
    # would read the artifact as a finding.
    assert abs(null) > 0.01
    # And a perfect head's measured asymmetry IS that null.
    assert asym == pytest.approx(null, abs=0.005)


def test_tail_asymmetry_separates_artifact_from_defect() -> None:
    """Judged as an EXCESS over the measured null, never as a raw level."""
    symmetric = score_buckets(
        _noisy_ruler_rows(compression=0.30), slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200,
    )
    directional = score_buckets(
        _noisy_ruler_rows(compression=0.30, optimism_when_losing=0.10),
        slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200,
    )

    # A purely symmetric compression is large in each tail...
    assert _stat(symmetric, SF_EVAL_BUCKET_NAMES[0]).net_minus_sf > 0.05
    assert _stat(symmetric, SF_EVAL_BUCKET_NAMES[-1]).net_minus_sf < -0.05

    sym = tail_asymmetry(symmetric)
    sym_null = perfect_head_tail_asymmetry(symmetric)
    assert sym is not None
    assert sym_null is not None
    # ...and the RAW level is NOT near zero, so the old assertion would have
    # been reading the null. Only the excess over the null is near zero.
    assert abs(sym - sym_null) < 0.02

    directional_asym = tail_asymmetry(directional)
    directional_null = perfect_head_tail_asymmetry(directional)
    assert directional_asym is not None
    assert directional_null is not None
    assert directional_asym - directional_null > 0.05


def test_net_minus_target_is_free_of_the_bucketing_artifact() -> None:
    """The primary axis stays unbiased where the ruler-relative one does not.

    `target_equals_net` would make this pass trivially — the difference would be
    identically zero and the test would pin the field's identity rather than the
    claim. Instead the head is a NOISY but UNBIASED estimate of its target, so
    `net_minus_target` is a real random variable whose per-bucket mean must
    still be zero, on the same rows where `net_minus_sf` is driven far from zero
    by bucketing on a noisy ruler.
    """
    rng = np.random.default_rng(31)
    base = _noisy_ruler_rows(n=40000, compression=0.30, target_equals_net=True)
    noise = rng.normal(0.0, 0.05, size=base.net_score.size)
    rows = OptimismRows(
        sf_cp=base.sf_cp, sf_ruler_score=base.sf_ruler_score,
        net_score=np.clip(base.target_score + noise, 0.0, 1.0),
        target_score=base.target_score, outcome_score=base.outcome_score,
        target_sf_score=base.target_sf_score, search_score=base.search_score,
        game_id=base.game_id, piece_count=base.piece_count,
    )
    stats = score_buckets(rows, slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=400)
    # The ruler-relative axis IS driven away from zero in the tails...
    assert abs(_stat(stats, SF_EVAL_BUCKET_NAMES[0]).net_minus_sf) > 0.05
    # ...while the primary axis stays at zero in EVERY bucket, and its CI says so.
    for s in stats:
        assert abs(s.net_minus_target) < 0.01
        assert s.net_minus_target_ci[0] <= 0.0 <= s.net_minus_target_ci[1]


def test_tail_asymmetry_ci_covers_the_point_and_narrows_with_data() -> None:
    small = _noisy_ruler_rows(n=3000, seed=21)
    large = _noisy_ruler_rows(n=30000, seed=21)
    for rows in (small, large):
        stats = score_buckets(rows, slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=400)
        point = tail_asymmetry(stats)
        ci = tail_asymmetry_ci(rows, n_boot=400, seed=1)
        assert point is not None
        assert ci is not None
        assert ci[0] <= point <= ci[1]
    wide = tail_asymmetry_ci(small, n_boot=400, seed=1)
    narrow = tail_asymmetry_ci(large, n_boot=400, seed=1)
    assert wide is not None
    assert narrow is not None
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_scorer_attributes_error_to_head_versus_target() -> None:
    """A head that matches an optimistic target must not be blamed for it."""
    rows = _rows()
    bad_target = OptimismRows(
        sf_cp=rows.sf_cp, sf_ruler_score=rows.sf_ruler_score,
        net_score=np.clip(rows.sf_ruler_score + 0.20, 0.0, 1.0),
        target_score=np.clip(rows.sf_ruler_score + 0.20, 0.0, 1.0),
        outcome_score=rows.outcome_score, target_sf_score=rows.target_sf_score,
        search_score=rows.search_score, game_id=rows.game_id,
        piece_count=rows.piece_count,
    )
    stats = score_buckets(bad_target, slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200)
    lost = _stat(stats, SF_EVAL_BUCKET_NAMES[0])
    # The head fits its target exactly; the target is what is optimistic.
    assert abs(lost.net_minus_target) < 0.01
    assert lost.target_minus_sf > 0.1


def test_outcome_calibration_detects_a_weak_opponent() -> None:
    """The arm that carries the handicap conclusion must be able to show it.

    Two populations over the SAME rulers: one where results follow the eval, one
    where the loser escapes half the time from lost positions — the shape a
    handicapped opponent produces. Only the second may light up.
    """
    rng = np.random.default_rng(4)
    cp = rng.uniform(-1200.0, 1200.0, size=8000)
    ruler = cp_to_expected_score(cp, slope=SLOPE, draw_width_cp=DRAW_WIDTH)
    game_id = np.arange(cp.size) // 5
    honest = outcome_calibration(
        ruler_cp=cp, outcome_score=ruler, game_id=game_id,
        slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200,
    )
    escaped = np.where(cp <= -300.0, np.minimum(ruler + 0.25, 1.0), ruler)
    handicapped = outcome_calibration(
        ruler_cp=cp, outcome_score=escaped, game_id=game_id,
        slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200,
    )
    lost = SF_EVAL_BUCKET_NAMES[0]
    honest_lost = next(c for c in honest if c.name == lost)
    hand_lost = next(c for c in handicapped if c.name == lost)
    assert abs(honest_lost.delta) < 1e-9
    assert honest_lost.ci[0] <= 0.0 <= honest_lost.ci[1]
    assert hand_lost.delta > 0.2
    assert hand_lost.ci[0] > 0.15
    # And it must stay quiet outside the buckets where the escape was injected.
    hand_won = next(c for c in handicapped if c.name == SF_EVAL_BUCKET_NAMES[-1])
    assert abs(hand_won.delta) < 1e-9


def test_bootstrap_ci_respects_game_clustering() -> None:
    """Rows inside a game are correlated; the CI must not pretend otherwise."""
    many_games = score_buckets(
        _rows(rows_per_game=1), slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=400, seed=3,
    )
    few_games = score_buckets(
        _rows(rows_per_game=300), slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=400, seed=3,
    )
    lo_a, hi_a = _stat(many_games, SF_EVAL_BUCKET_NAMES[0]).net_minus_sf_ci
    lo_b, hi_b = _stat(few_games, SF_EVAL_BUCKET_NAMES[0]).net_minus_sf_ci
    assert (hi_b - lo_b) > (hi_a - lo_a)


# ---------------------------------------------------------------------------
# The shard-integrity gate — it must actually be able to fail
# ---------------------------------------------------------------------------


def _planes_and_labels(n: int = 800, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Boards with a random material edge plus the SF label that matches it."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 12, 8, 8), dtype=np.float32)
    material = np.zeros(n, dtype=np.float64)
    values = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)
    for i in range(n):
        for plane in range(12):
            count = int(rng.integers(0, 4))
            if count:
                cells = rng.choice(64, size=count, replace=False)
                x[i, plane, cells // 8, cells % 8] = 1.0
            v = values[plane % 6] * count
            material[i] += v if plane < 6 else -v
    cp = material * 100.0
    wdl = np.stack([
        cp_to_wdl(float(c), None, slope=SLOPE, draw_width_cp=DRAW_WIDTH) for c in cp
    ]).astype(np.float64)
    return x, wdl


def test_attachment_gate_passes_on_attached_labels() -> None:
    x, wdl = _planes_and_labels()
    assert material_balance_from_planes(x).std() > 0
    reading = sf_label_attachment_corr(x, wdl)
    assert reading.usable
    assert reading.value > 0.9


def test_attachment_gate_catches_detached_labels() -> None:
    """The 2026-07-31 defect: labels internally sound, on the wrong rows.

    The labels stay perfectly self-consistent under this permutation — every
    check that reads the label alone still passes — so only a check that joins
    the label to its own position can see it.
    """
    x, wdl = _planes_and_labels()
    scrambled = wdl[np.random.default_rng(5).permutation(wdl.shape[0])]
    reading = sf_label_attachment_corr(x, scrambled)
    assert reading.usable
    assert abs(reading.value) < 0.15
    assert reading.value < SF_LABEL_ATTACHMENT_MIN


def test_rank_corr_matches_a_known_case() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rank_corr(a, a) == pytest.approx(1.0)
    assert rank_corr(a, -a) == pytest.approx(-1.0)
    # Monotone but non-linear: Spearman is 1, Pearson would not be.
    assert rank_corr(a, a ** 3) == pytest.approx(1.0)
    # Ties are averaged rather than crashing.
    assert np.isnan(rank_corr(np.ones(5), a))


def test_optimism_rows_rejects_misaligned_inputs() -> None:
    rows = _rows(n=60, rows_per_game=3)
    with pytest.raises(ValueError, match="net_score has"):
        OptimismRows(
            sf_cp=rows.sf_cp, sf_ruler_score=rows.sf_ruler_score,
            net_score=rows.net_score[:-1], target_score=rows.target_score,
            outcome_score=rows.outcome_score, target_sf_score=rows.target_sf_score,
            search_score=rows.search_score, game_id=rows.game_id,
            piece_count=rows.piece_count,
        )


# ---------------------------------------------------------------------------
# Axis statuses: "cannot evaluate" must never be indistinguishable from "fine"
# ---------------------------------------------------------------------------


def test_every_axis_reports_too_few_rows_rather_than_a_noisy_value() -> None:
    """Below the row floor an axis must decline, not guess.

    A rate over a handful of rows is noise, and noise that lands on the pass
    side is worse than no reading. The floor is pinned here because a silent
    change to it would turn small shards into automatic passes.
    """
    assert SF_AXIS_MIN_ROWS == 30
    n = SF_AXIS_MIN_ROWS - 1
    x, wdl = _planes_and_labels(n=n)
    assert sf_label_attachment_corr(x, wdl).status == "too_few_rows"
    assert sf_multipv_missing_rate(np.ones(n, dtype=bool)).status == "too_few_rows"
    assert sf_bestmove_is_first_legal_rate(
        np.zeros(n, dtype=np.int64), np.ones((n, 8), dtype=np.uint8),
    ).status == "too_few_rows"
    # One row over the floor, every axis reports a value.
    m = SF_AXIS_MIN_ROWS
    x2, wdl2 = _planes_and_labels(n=m)
    assert sf_label_attachment_corr(x2, wdl2).usable
    assert sf_multipv_missing_rate(np.ones(m, dtype=bool)).usable


def test_degenerate_attachment_input_is_not_a_pass() -> None:
    """A constant-material shard yields NaN correlation — that is not evidence.

    Without a status of its own it would arrive as a bare NaN and, on any
    comparison written as `corr >= threshold`, evaluate False-y in a way that is
    easy to invert by accident.
    """
    n = 60
    x = np.zeros((n, 12, 8, 8), dtype=np.float32)
    x[:, 0, 0, 0] = 1.0                       # identical material on every row
    wdl = np.tile(np.array([0.3, 0.4, 0.3]), (n, 1))
    reading = sf_label_attachment_corr(x, wdl)
    assert reading.status == "degenerate"
    assert not reading.usable


def test_multipv_axis_separates_where_the_desync_rate_cannot() -> None:
    """The reason the enforced axis was switched: threshold headroom.

    Measured over all 834 shards of the live trial, sound shards reach 0.008032
    on this axis and corrupt shards start at 0.010511, so the 0.01 default sits
    in a real gap. The bestmove-is-first-legal rate has a sound max of 0.1496
    against a corrupt min of 0.1505 — no honest threshold exists on it, which is
    why it is a diagnostic and this is the gate.
    """
    n = 400
    sound = np.ones(n, dtype=bool)
    sound[:2] = False                                   # 0.005, inside the gap
    assert sf_multipv_missing_rate(sound).value <= SF_MULTIPV_MISS_MAX
    corrupt = np.ones(n, dtype=bool)
    corrupt[:60] = False                                # 0.15
    assert sf_multipv_missing_rate(corrupt).value > SF_MULTIPV_MISS_MAX
    # The rate is taken over LABELLED rows only, so unlabelled rows cannot
    # dilute a corrupt shard back under the threshold.
    labelled = np.zeros(n, dtype=bool)
    labelled[:100] = True
    diluted = np.ones(n, dtype=bool)
    diluted[:60] = False
    assert sf_multipv_missing_rate(diluted, labelled).value == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# The SCRIPT's config gate, on the SHIPPED config.
# ---------------------------------------------------------------------------
# Everything above tests `chess_anti_engine/eval/value_optimism.py`, the library.
# `scripts/value_optimism.py` — the entry point an operator actually runs — has
# its own hard startup gate, `resolve_blend_knobs`, and NOTHING reached it. That
# is a gate that cannot fail in the exact sense this repo keeps rediscovering:
# a config-only PR deleted `sf_search_dampen_sf_low` / `_high`, the script would
# have raised `SystemExit` on `--config configs/pbt2_small.yaml` before touching
# a model or a shard, and this whole file stayed green.
#
# So these two run the real function against the real production config. They
# read `_CROSSCHECK_YAML_VS_PARAMS` from the module rather than restating it, so
# adding a key to that tuple extends the coverage automatically.

def _fabricated_run_dir(tmp_path, flat: dict, keys: tuple[str, ...]):
    """A run dir with the two artifacts `resolve_blend_knobs` cross-checks."""
    import json

    trial = tmp_path / "tune" / "train_trial_00000"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text('{"training_iteration": 1}\n', encoding="utf-8")
    (trial / "progress.csv").write_text(
        "training_iteration,sf_wdl_frac,sf_wdl_temperature\n1,0.45,1.0\n", encoding="utf-8",
    )
    (trial / "params.json").write_text(
        json.dumps({k: flat[k] for k in keys if k in flat}), encoding="utf-8",
    )
    return tmp_path


def _production_flat() -> dict:
    from pathlib import Path

    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file

    repo_root = Path(__file__).resolve().parents[1]
    return flatten_run_config_defaults(load_yaml_file(str(repo_root / "configs" / "pbt2_small.yaml")))


def test_the_shipped_config_still_satisfies_the_scripts_blend_knob_gate(tmp_path) -> None:
    """`scripts/value_optimism.py --config configs/pbt2_small.yaml` must start.

    `resolve_blend_knobs` runs before any model or data work and REQUIRES every
    key of `_CROSSCHECK_YAML_VS_PARAMS` to be present in the flattened config —
    deliberately, because its docstring refuses to let an absent key fall back to
    the neutral value. So deleting one of those keys from the production yaml is
    not a documentation defect, it is the script failing to run at all.
    """
    from scripts.value_optimism import _CROSSCHECK_YAML_VS_PARAMS, resolve_blend_knobs

    flat = _production_flat()
    missing = [k for k in _CROSSCHECK_YAML_VS_PARAMS if k not in flat]
    assert not missing, (
        f"configs/pbt2_small.yaml no longer sets {missing}, which "
        f"scripts/value_optimism.py requires to be present. Restore the key(s) — "
        f"the script raises SystemExit at startup without them."
    )
    resolved = resolve_blend_knobs(flat, _fabricated_run_dir(tmp_path, flat, _CROSSCHECK_YAML_VS_PARAMS))
    for key in _CROSSCHECK_YAML_VS_PARAMS:
        assert key in resolved


def test_the_blend_knob_gate_actually_fails_when_a_required_key_is_absent(tmp_path) -> None:
    """The positive control for the test above: prove the gate can go red.

    Without this, `..._still_satisfies_...` would pass on a `resolve_blend_knobs`
    that had quietly been softened into a `.get(key, neutral)` — which is the
    `reco_diff misses absent keys` shape the function exists to refuse.
    """
    from scripts.value_optimism import _CROSSCHECK_YAML_VS_PARAMS, resolve_blend_knobs

    flat = _production_flat()
    run_dir = _fabricated_run_dir(tmp_path, flat, _CROSSCHECK_YAML_VS_PARAMS)
    for key in _CROSSCHECK_YAML_VS_PARAMS:
        without = {k: v for k, v in flat.items() if k != key}
        with pytest.raises(SystemExit, match=rf"{re.escape(key)} is absent from the config"):
            resolve_blend_knobs(without, run_dir)
