"""Tests for the stratified value-optimism scorer (eval/value_optimism.py).

The two that matter most are the controls. A scorer that cannot fail is not a
scorer, and a scorer that only passes its NEGATIVE control has shown nothing —
so both directions are pinned here:

- ``test_shuffle_control_kills_bucket_structure``: destroy the position <->
  net-evaluation association and the bucket structure must collapse.
- ``test_tail_asymmetry_separates_artifact_from_defect``: a perfectly
  SYMMETRIC compression (what a noisy ruler produces from a flawless head)
  must read as ~0 asymmetry, while an injected one-sided optimism must not.
  Without this, the negative control would happily pass on an instrument that
  reports the artifact as a finding.
"""
from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.eval.value_optimism import (
    SF_EVAL_BUCKET_EDGES,
    SF_EVAL_BUCKET_NAMES,
    SF_LABEL_ATTACHMENT_MIN,
    OptimismRows,
    bucket_names_for,
    bucket_net_score_spread,
    cp_to_expected_score,
    expected_score,
    expected_score_to_cp,
    material_balance_from_planes,
    rank_corr,
    score_buckets,
    sf_eval_bucket,
    sf_eval_bucket_array,
    sf_label_attachment_corr,
    tail_asymmetry,
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


def test_tail_asymmetry_separates_artifact_from_defect() -> None:
    symmetric = score_buckets(
        _rows(compression=0.30), slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200,
    )
    directional = score_buckets(
        _rows(compression=0.30, optimism_when_losing=0.10),
        slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200,
    )

    # A purely symmetric compression is large in each tail...
    assert _stat(symmetric, SF_EVAL_BUCKET_NAMES[0]).net_minus_sf > 0.05
    assert _stat(symmetric, SF_EVAL_BUCKET_NAMES[-1]).net_minus_sf < -0.05
    # ...yet cancels in the asymmetry, which is what makes it readable as an
    # artifact rather than a finding.
    sym = tail_asymmetry(symmetric)
    assert sym is not None
    assert abs(sym) < 0.01

    # An injected one-sided optimism survives the cancellation.
    directional_asym = tail_asymmetry(directional)
    assert directional_asym is not None
    assert directional_asym > 0.08


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
    assert sf_label_attachment_corr(x, wdl) > 0.9


def test_attachment_gate_catches_detached_labels() -> None:
    """The 2026-07-31 defect: labels internally sound, on the wrong rows.

    The labels stay perfectly self-consistent under this permutation — every
    check that reads the label alone still passes — so only a check that joins
    the label to its own position can see it.
    """
    x, wdl = _planes_and_labels()
    scrambled = wdl[np.random.default_rng(5).permutation(wdl.shape[0])]
    corr = sf_label_attachment_corr(x, scrambled)
    assert abs(corr) < 0.15
    assert corr < SF_LABEL_ATTACHMENT_MIN


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
