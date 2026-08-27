"""Pentanomial GSPRT: the math, the stop rule, and the wiring into all three loops.

Three layers, because they fail differently:

* the LLR itself, checked against an INDEPENDENT reference (fishtest's own
  normal approximation, ``LLRcalc.LLR_alt2``, reimplemented here from the
  published formula) plus exact invariants a wrong sign or a swapped hypothesis
  cannot satisfy;
* ``SprtMonitor``, the stop rule and its resume semantics;
* the three PRODUCTION play paths — rolling, chunked, matched_time — driven with
  a mocked, CPU-only game stream. The question those answer is the one this repo
  keeps getting wrong: does the boundary reach the loop and CONTROL it, or is it
  accepted and then ignored? Every one of them is paired with a MUTATED boundary
  on the same game stream, where the stop must NOT fire.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest

import chess_anti_engine.selfplay.match as match_mod
import scripts.arena_standard as arena
from chess_anti_engine.eval.sprt import (
    PAIR_OUTCOME_VALUES,
    SprtMonitor,
    SprtSpec,
    constrained_mle,
    gsprt_llr,
    gsprt_llr_elo,
    logistic_score,
    pentanomial_ascending,
    regularize,
)
from chess_anti_engine.utils.game_log import read_game_log
from scripts.arena_standard import (
    SideSearch,
    play_paired_games_matched_sims_rolling,
    play_paired_games_matched_time,
)

# The preregistered boundary these tests exercise. alpha = beta = 0.05 gives
# symmetric Wald bounds at +/- log(0.95/0.05) = +/- 2.9444.
SPEC = SprtSpec(elo0=0.0, elo1=20.0, alpha=0.05, beta=0.05)

# Two MUTANTS of that boundary, used everywhere the real one is:
#   WIDE  cannot be reached by any stream these tests play -> the stop must NOT
#         fire, which is what distinguishes "the boundary controls the loop"
#         from "the loop always stops after N pairs for some other reason".
#   TIGHT is reachable much sooner -> the stop must fire EARLIER on the very
#         same games, which is what distinguishes "the boundary controls the
#         loop" from "the loop ignores the value and uses a hardcoded one".
SPEC_WIDE = SprtSpec(elo0=0.0, elo1=20.0, alpha=1e-9, beta=1e-9)
SPEC_TIGHT = SprtSpec(elo0=0.0, elo1=20.0, alpha=0.3, beta=0.3)

# Crossing points, stated as constants rather than read back off the monitor.
#
# ⚑ A crossing point is a property of the stream AND of the LOOK SCHEDULE. The
# LLR of an all-draw stream passes -2.9444 at 50 pairs and -0.8473 at 15; a loop
# that retires TWO pairs per look (rolling at pool_size=4, chunked at
# --max-concurrent-games 4) cannot stop at 15 and stops at 16 instead. That is
# not slop to be papered over with an inequality: the difference between 15 and
# 16 is exactly the evidence that the look schedule is what the code says it is.
ALL_DRAW_CROSS_AT_PAIRS = 50           # even: identical under both schedules
ALL_DRAW_CROSS_TIGHT_ONE_AT_A_TIME = 15
ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME = 16

# A realistic candidate-favouring stream: mean pair score 1.5 (normalized 0.75),
# spread over three bins rather than one. Crosses the H1 bound at 53 pairs when
# looked at after every pair, which is what the matched_time loop does.
STRONG_CYCLE = (2.0, 1.5, 1.5, 1.0)
STRONG_CROSS_AT_PAIRS = 53


# ---------------------------------------------------------------------------
# The LLR math
# ---------------------------------------------------------------------------

def llr_normal_approximation(counts: Sequence[int], s0: float, s1: float) -> float:
    """fishtest ``LLRcalc.LLR_alt2`` — an INDEPENDENT check on the exact GSPRT.

        (N/2) * log( (var + (mu - s0)^2) / (var + (mu - s1)^2) )

    Derived from the normal approximation to the same likelihood ratio rather
    than from the multinomial MLE, so it shares no code path with
    ``gsprt_llr``: agreement is evidence about the formula, not about the
    implementation talking to itself. It is only an approximation, and it is a
    BAD one when the empirical variance collapses (a single-bin sample), which
    is why the cases below are all multi-bin.
    """
    reg = regularize(counts)
    n = sum(reg)
    p = [c / n for c in reg]
    mu = sum(pi * ai for pi, ai in zip(p, PAIR_OUTCOME_VALUES))
    var = sum(pi * (ai - mu) ** 2 for pi, ai in zip(p, PAIR_OUTCOME_VALUES))
    return (n / 2.0) * math.log((var + (mu - s0) ** 2) / (var + (mu - s1) ** 2))


@pytest.mark.parametrize(
    ("counts", "tolerance"),
    [
        # The approximation is an expansion in (mu - s), so its accuracy is a
        # property of how far the SAMPLE sits from the hypotheses, not of the
        # implementation. Near the hypotheses it pins the exact value to 2%; the
        # far sample below (mean score 0.35 against hypotheses at 0.50) is where
        # it starts to be the worse of the two, and 10% there is still far
        # tighter than any sign or scale error could survive.
        ((10, 20, 40, 20, 10), 0.02),    # symmetric, zero Elo
        ((5, 15, 40, 25, 15), 0.02),     # candidate slightly ahead
        ((40, 60, 100, 60, 40), 0.02),   # same shape as the first, 4x the pairs
        ((25, 35, 30, 8, 2), 0.10),      # candidate far behind both hypotheses
    ],
)
@pytest.mark.parametrize(("elo0", "elo1"), [(0.0, 5.0), (0.0, 20.0), (-3.0, 3.0)])
def test_llr_agrees_with_the_published_normal_approximation(
    counts: tuple[int, ...], tolerance: float, elo0: float, elo1: float,
) -> None:
    s0, s1 = logistic_score(elo0), logistic_score(elo1)
    exact = gsprt_llr(counts, s0=s0, s1=s1)
    approx = llr_normal_approximation(counts, s0, s1)
    assert exact == pytest.approx(approx, rel=tolerance, abs=1e-3), (
        f"exact GSPRT {exact} and fishtest's normal approximation {approx} "
        f"disagree by more than {tolerance:.0%} on {counts} at elo0={elo0}, "
        f"elo1={elo1}; one of the two is not the log-likelihood ratio it "
        "claims to be"
    )


def test_llr_is_antisymmetric_in_the_two_hypotheses() -> None:
    """Swapping s0 and s1 must negate the LLR — a sign error cannot survive this."""
    counts = (5, 15, 40, 25, 15)
    s0, s1 = logistic_score(0.0), logistic_score(20.0)
    assert gsprt_llr(counts, s0=s0, s1=s1) == pytest.approx(
        -gsprt_llr(counts, s0=s1, s1=s0), abs=1e-12,
    )


def test_llr_is_invariant_under_flipping_the_point_of_view() -> None:
    """Mirror the sample AND the hypotheses; the evidence is the same evidence.

    Reversing the counts is the candidate/reference POV swap (pair score x
    becomes 1 - x), which sends a mean of s to 1 - s, i.e. an Elo of e to -e.
    An implementation that reversed one and not the other passes every
    magnitude check and reports the wrong side as winning.
    """
    counts = (5, 15, 40, 25, 15)
    assert gsprt_llr_elo(counts, elo0=0.0, elo1=20.0) == pytest.approx(
        gsprt_llr_elo(tuple(reversed(counts)), elo0=0.0, elo1=-20.0), abs=1e-12,
    )


def test_llr_scales_with_the_number_of_pairs() -> None:
    """Same empirical distribution, k times the pairs => k times the LLR."""
    base = (5, 15, 40, 25, 15)
    one = gsprt_llr_elo(base, elo0=0.0, elo1=20.0)
    for k in (2, 5, 10):
        scaled = gsprt_llr_elo(tuple(k * c for c in base), elo0=0.0, elo1=20.0)
        assert scaled == pytest.approx(k * one, rel=0.01)


@pytest.mark.parametrize("elo", [-200.0, -20.0, 0.0, 20.0, 200.0])
def test_constrained_mle_is_a_distribution_with_the_requested_mean(elo: float) -> None:
    counts = (5, 15, 40, 25, 15)
    reg = regularize(counts)
    n = sum(reg)
    phat = [c / n for c in reg]
    s = logistic_score(elo)
    p = constrained_mle(phat, s)
    assert sum(p) == pytest.approx(1.0, abs=1e-12)
    assert sum(pi * ai for pi, ai in zip(p, PAIR_OUTCOME_VALUES)) == pytest.approx(
        s, abs=1e-12,
    )
    assert all(pi > 0.0 for pi in p)


def test_a_candidate_favouring_sample_accepts_h1() -> None:
    """~+190 Elo over 60 pairs is far past the H1 bound at alpha=beta=0.05."""
    counts = pentanomial_ascending([STRONG_CYCLE[i % 4] for i in range(60)])
    llr = gsprt_llr(counts, s0=SPEC.s0, s1=SPEC.s1)
    assert llr >= SPEC.bound_h1, f"LLR {llr} did not reach {SPEC.bound_h1}"


def test_an_all_draws_sample_accepts_h0() -> None:
    """Every pair split: no evidence of +20 Elo, and eventually proof against it."""
    llr = gsprt_llr((0, 0, 100, 0, 0), s0=SPEC.s0, s1=SPEC.s1)
    assert llr <= SPEC.bound_h0, f"LLR {llr} did not reach {SPEC.bound_h0}"


def test_the_ascending_binning_is_the_exact_reverse_of_the_arena_binning() -> None:
    """Pin the orientation the whole test hinges on.

    ``arena_standard.PAIR_SCORES`` runs BEST-first on the 0..2 point scale;
    ``PAIR_OUTCOME_VALUES`` runs WORST-first on the 0..1 scale. Confusing the two
    inverts the test silently — a winning candidate would accept H0 — and no
    amount of arithmetic checking catches it, so it is pinned by name here.
    """
    assert tuple(s / 2.0 for s in reversed(arena.PAIR_SCORES)) == PAIR_OUTCOME_VALUES
    scores = [2.0, 1.5, 1.5, 1.0, 0.5, 0.0, 0.0]
    assert pentanomial_ascending(scores) == tuple(
        reversed(arena.pentanomial_counts(scores))
    )
    assert pentanomial_ascending([2.0]) == (0, 0, 0, 0, 1)
    assert pentanomial_ascending([0.0]) == (1, 0, 0, 0, 0)


def test_pentanomial_ascending_refuses_a_score_that_is_not_a_pair_score() -> None:
    with pytest.raises(ValueError, match="pair score must be"):
        pentanomial_ascending([1.25])


# ---------------------------------------------------------------------------
# The spec: an unstated hypothesis is not a hypothesis
# ---------------------------------------------------------------------------

def test_spec_bounds_are_walds() -> None:
    spec = SprtSpec(elo0=0.0, elo1=5.0, alpha=0.05, beta=0.10)
    assert spec.bound_h1 == pytest.approx(math.log((1 - 0.10) / 0.05))
    assert spec.bound_h0 == pytest.approx(math.log(0.10 / (1 - 0.05)))
    assert spec.bound_h0 < 0.0 < spec.bound_h1


def test_spec_round_trips_from_the_cli_string() -> None:
    spec = SprtSpec.from_cli("elo0=-1.5,elo1=4.5,alpha=0.05,beta=0.1")
    assert (spec.elo0, spec.elo1, spec.alpha, spec.beta) == (-1.5, 4.5, 0.05, 0.1)
    assert spec.s0 == pytest.approx(logistic_score(-1.5))


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("elo0=0,elo1=5,alpha=0.05", "missing beta"),
        ("elo1=5,alpha=0.05,beta=0.05", "missing elo0"),
        ("", "missing elo0, elo1, alpha, beta"),
        ("elo0=0,elo1=5,alpha=0.05,beta=0.05,gamma=1", "unknown key 'gamma'"),
        ("elo0=0,elo1=5,alpha=0.05,beta=0.05,elo0=1", "given more than once"),
        ("elo0=0,elo1=5,alpha=nope,beta=0.05", "is not a number"),
        ("elo0,elo1=5,alpha=0.05,beta=0.05", "is not k=v"),
        ("elo0=0 elo1=5,alpha=0.05,beta=0.05", "is not a number"),
        ("elo0=5,elo1=0,alpha=0.05,beta=0.05", "needs elo0 < elo1"),
        ("elo0=0,elo1=5,alpha=0,beta=0.05", "alpha must be in"),
        ("elo0=0,elo1=5,alpha=0.05,beta=1", "beta must be in"),
        ("elo0=0,elo1=5,alpha=0.6,beta=0.6", "alpha \\+ beta < 1"),
        ("elo0=0,elo1=99999,alpha=0.05,beta=0.05", "beyond"),
        ("elo0=nan,elo1=5,alpha=0.05,beta=0.05", "must be finite"),
    ],
)
def test_spec_refuses_an_incomplete_or_impossible_hypothesis(
    spec: str, match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SprtSpec.from_cli(spec)


def test_a_boundaryless_spec_would_decide_on_the_first_look() -> None:
    """alpha + beta >= 1 is refused because the test could not fail.

    The direct demonstration: at alpha = beta = 0.5 the two Wald bounds coincide
    at 0, so an LLR of exactly zero — zero pairs, no evidence at all — sits on
    both. A gate that cannot fail is this repo's signature defect, so the spec
    refuses to construct rather than the monitor refusing to fire.
    """
    with pytest.raises(ValueError, match="alpha \\+ beta < 1"):
        SprtSpec(elo0=0.0, elo1=5.0, alpha=0.5, beta=0.5)


# ---------------------------------------------------------------------------
# The monitor: stop rule, cap, resume
# ---------------------------------------------------------------------------

def _feed(
    spec: SprtSpec, scores: Sequence[float], *, step: int = 1, cap: int = 400,
) -> SprtMonitor:
    monitor = SprtMonitor(spec, pairs_cap=cap, granularity="pair")
    seen: list[float] = []
    for i in range(0, len(scores), step):
        seen.extend(scores[i:i + step])
        if monitor.update(seen) is not None:
            break
    return monitor


def test_monitor_stops_at_the_boundary_and_not_one_pair_before() -> None:
    monitor = _feed(SPEC, [1.0] * 200)
    assert monitor.verdict == "H0"
    assert monitor.pairs == ALL_DRAW_CROSS_AT_PAIRS
    assert monitor.llr <= SPEC.bound_h0
    # The look one pair earlier must have been strictly inside the bounds,
    # or the "not before" half of the claim is vacuous.
    _, prior_llr = monitor.trajectory[-2]
    assert SPEC.bound_h0 < prior_llr < SPEC.bound_h1


def test_a_mutated_boundary_moves_the_stop() -> None:
    """The same games, three boundaries, three different stopping points.

    This is the take-effect proof for the monitor: an implementation that
    ignored the spec and stopped on some fixed schedule would give the same
    answer three times.
    """
    stream = [1.0] * 400
    assert _feed(SPEC_TIGHT, stream).pairs == ALL_DRAW_CROSS_TIGHT_ONE_AT_A_TIME
    assert _feed(SPEC, stream).pairs == ALL_DRAW_CROSS_AT_PAIRS
    wide = _feed(SPEC_WIDE, stream[:200], cap=200)
    assert wide.verdict is None, (
        f"the widened boundary {SPEC_WIDE.bound_h0} is unreachable in 200 "
        f"all-draw pairs (LLR reached {wide.llr}), so the stop must NOT fire"
    )


def test_monitor_accepts_h1_on_a_candidate_favouring_stream() -> None:
    monitor = _feed(SPEC, [STRONG_CYCLE[i % 4] for i in range(200)])
    assert monitor.verdict == "H1"
    assert monitor.pairs == STRONG_CROSS_AT_PAIRS
    assert monitor.llr >= SPEC.bound_h1


def test_reaching_the_cap_without_crossing_is_inconclusive() -> None:
    """Never silently converted into a fixed-N verdict."""
    monitor = SprtMonitor(SPEC_WIDE, pairs_cap=40, granularity="pair")
    monitor.update([1.0] * 40)
    assert monitor.verdict is None
    assert monitor.finalize(stop_reason="cap") == "INCONCLUSIVE"
    record = monitor.as_record()
    assert record["verdict"] == "INCONCLUSIVE"
    assert record["stop_reason"] == "cap"
    assert record["stopped_early"] is False
    assert "INCONCLUSIVE" in monitor.verdict_line()
    assert "NOT a fixed-N verdict" in monitor.verdict_line()


def test_finalize_never_overwrites_a_crossed_verdict() -> None:
    monitor = _feed(SPEC, [1.0] * 200)
    assert monitor.finalize(stop_reason="cap") == "H0"
    assert monitor.as_record()["stop_reason"] == "boundary"
    assert monitor.as_record()["stopped_early"] is True


def test_the_resumed_pairs_are_part_of_the_sample() -> None:
    """loaded + new must give the SAME statistic as one uninterrupted run.

    The GSPRT re-fits the constrained MLE over the whole empirical
    distribution, so an incremental accumulator would drift; and a monitor that
    only saw this process's own pairs would restart the test after every crash.
    """
    stream = [STRONG_CYCLE[i % 4] for i in range(40)]
    whole = SprtMonitor(SPEC, pairs_cap=200, granularity="pair")
    whole.update(stream)

    split = SprtMonitor(
        SPEC, prior_pair_scores=stream[:17], pairs_cap=200, granularity="pair",
    )
    split.update(stream[17:])

    assert split.pairs == whole.pairs == 40
    assert split.counts == whole.counts
    assert split.llr == pytest.approx(whole.llr, abs=1e-12)
    assert split.as_record()["resumed_pairs"] == 17


def test_a_monitor_built_on_already_crossed_prior_pairs_starts_crossed() -> None:
    monitor = SprtMonitor(
        SPEC, prior_pair_scores=[1.0] * 60, pairs_cap=200, granularity="pair",
    )
    assert monitor.crossed()
    assert monitor.verdict == "H0"
    assert monitor.llr_first == pytest.approx(monitor.llr)


def test_the_trajectory_banks_one_point_per_distinct_pair_count() -> None:
    """A rolling loop looks every ply; the bank must not grow per ply."""
    monitor = SprtMonitor(SPEC_WIDE, pairs_cap=200, granularity="pair")
    scores: list[float] = []
    for i in range(10):
        scores.append(1.0)
        for _ in range(5):       # five looks, one new pair
            monitor.update(scores)
        assert monitor.pairs == i + 1
    assert monitor.looks == 51   # 1 at construction + 10 * 5
    assert [p for p, _ in monitor.trajectory] == list(range(11))
    # 51 consultations of the boundary, but only 11 different samples: the
    # repeats carry no extra multiplicity and the record must not imply they do.
    assert monitor.as_record()["distinct_samples"] == 11


def test_record_carries_the_hypothesis_and_the_bias_caveat() -> None:
    monitor = _feed(SPEC, [1.0] * 200)
    record = monitor.as_record()
    assert record["elo0"] == 0.0
    assert record["elo1"] == 20.0
    assert record["alpha"] == 0.05
    assert record["beta"] == 0.05
    assert record["bound_h0"] == pytest.approx(SPEC.bound_h0)
    assert record["check_granularity"] == "pair"
    assert record["pairs"] == ALL_DRAW_CROSS_AT_PAIRS
    assert record["games"] == 2 * ALL_DRAW_CROSS_AT_PAIRS
    assert record["pentanomial_ascending"]["DD_WL"] == ALL_DRAW_CROSS_AT_PAIRS
    assert record["elo_estimate_biased_away_from_zero"] is True
    assert "BIASED AWAY FROM ZERO" in str(record["caveat"])
    trajectory = record["llr_trajectory"]
    assert isinstance(trajectory, list)
    assert trajectory[-1][0] == ALL_DRAW_CROSS_AT_PAIRS


# ---------------------------------------------------------------------------
# The production play loops
# ---------------------------------------------------------------------------

def _search() -> SideSearch:
    return SideSearch(shape="test", source="test", gumbel={}, vloss_weight=0,
                      target_batch=0)


@pytest.fixture
def scripted_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every game runs out its ply budget and is adjudicated a draw.

    Deliberately the DEGENERATE stream: what these tests measure is whether the
    boundary reaches the loop and controls it, and a stream whose stopping point
    is exactly computable from the formula is the sharpest instrument for that.
    The LLR's behaviour on realistic multi-bin samples is covered above, and by
    the matched_time loop below, which can produce asymmetric pairs.
    """
    def fake_pick(_model: object, sub_boards: list[chess.Board],
                  **_kwargs: object) -> list[int]:
        return [0] * len(sub_boards)

    def fake_apply(boards: list[chess.Board], idxs: list[int],
                   _actions: list[int], *, strict: bool) -> None:
        assert strict, "the arena must decode actions strictly"
        for i in idxs:
            boards[i].push(next(iter(boards[i].legal_moves)))

    monkeypatch.setattr(match_mod, "pick_moves_for_boards", fake_pick)
    monkeypatch.setattr(match_mod, "apply_actions_to_boards", fake_apply)


def _play_rolling(
    openings: list[chess.Board], *, sprt: SprtMonitor | None,
) -> list[float]:
    return play_paired_games_matched_sims_rolling(
        None, None, openings,
        device="cpu", rng=np.random.default_rng(7),
        sims_candidate=1, sims_reference=1, max_plies=1,
        temperature=0.1, gumbel_add_noise=False,
        search_candidate=_search(), search_reference=_search(),
        pool_size=4, report_every=10_000, sprt=sprt,
    )


@pytest.mark.usefixtures("scripted_moves")
def test_rolling_loop_stops_on_the_boundary_at_pair_granularity() -> None:
    """The stop fires in the REAL rolling loop, and only on complete pairs.

    ``pool_size=4`` retires two whole pairs per reap, so a stop at 50 pairs is
    also the observation that no look ever landed on a half-played pair: an
    odd count would mean one coloring had been scored on its own.
    """
    monitor = SprtMonitor(SPEC, pairs_cap=200, granularity="pair")
    scores = _play_rolling([chess.Board() for _ in range(200)], sprt=monitor)
    assert monitor.verdict == "H0"
    assert len(scores) == ALL_DRAW_CROSS_AT_PAIRS
    assert monitor.pairs == ALL_DRAW_CROSS_AT_PAIRS
    assert set(scores) == {1.0}
    assert all(p % 2 == 0 for p, _ in monitor.trajectory), (
        "a look landed on an odd number of pairs, which at pool_size=4 can "
        "only mean a half-played pair reached the statistic"
    )


@pytest.mark.usefixtures("scripted_moves")
def test_rolling_loop_plays_the_whole_schedule_when_the_boundary_moves() -> None:
    """MUTANT: widen the boundary out of reach; the same games must not stop.

    Without this the previous test proves nothing about the boundary — a loop
    that stopped after 50 pairs for any other reason would pass it.
    """
    monitor = SprtMonitor(SPEC_WIDE, pairs_cap=60, granularity="pair")
    scores = _play_rolling([chess.Board() for _ in range(60)], sprt=monitor)
    assert monitor.verdict is None
    assert len(scores) == 60, "the widened boundary must not stop the loop"
    assert monitor.finalize(stop_reason="cap") == "INCONCLUSIVE"


@pytest.mark.usefixtures("scripted_moves")
def test_rolling_loop_stops_earlier_on_a_tighter_boundary() -> None:
    """MUTANT: tighten the boundary; the same games must stop sooner."""
    monitor = SprtMonitor(SPEC_TIGHT, pairs_cap=200, granularity="pair")
    scores = _play_rolling([chess.Board() for _ in range(200)], sprt=monitor)
    assert monitor.verdict == "H0"
    assert len(scores) == ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME


@pytest.mark.usefixtures("scripted_moves")
def test_rolling_loop_without_sprt_is_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default path: no monitor, no stop, and not one line of SPRT output."""
    scores = _play_rolling([chess.Board() for _ in range(20)], sprt=None)
    assert len(scores) == 20
    assert "SPRT" not in capsys.readouterr().out


@pytest.mark.usefixtures("scripted_moves")
def test_the_rolling_loop_announces_the_boundary_it_received(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Printed from the monitor the LOOP holds, not from the CLI string.

    A flag that is parsed, echoed at startup and then dropped on the floor looks
    identical on the console to one that works — which is exactly this repo's
    signature defect. This line is emitted inside the loop body, so it cannot
    appear unless the object reached it.
    """
    monitor = SprtMonitor(SPEC_TIGHT, pairs_cap=40, granularity="pair")
    _play_rolling([chess.Board() for _ in range(40)], sprt=monitor)
    out = capsys.readouterr().out
    assert "SPRT ARMED in the rolling loop" in out
    assert f"{SPEC_TIGHT.bound_h0:+.4f}" in out
    assert "SPRT boundary CROSSED in the rolling loop" in out


# ---- the half-pair guard, on a stream that can tell the difference ---------
#
# ⚑ THE TESTS ABOVE CANNOT SEE THE GUARD. They play a lockstep ALL-DRAW stream,
# where a pair with one coloring finished scores 0.5 and imputing its missing
# partner as a draw scores 0.5 + 0.5 = 1.0 — exactly what the pair scores once
# it completes. So `complete_pair_scores` and an imputing substitute produce the
# SAME sample on the SAME games, and a mutant that swaps one for the other
# survives all of them. Measured, not argued: that mutant survived the 71 tests
# this file shipped with.
#
# What kills it is a HALF-PLAYED PAIR WHOSE FINISHED COLORING IS DECISIVE. Here
# every game is decided in one ply — the candidate WINS its White half and LOSES
# its Black half — except pair 0's Black half, which never finishes at all. So a
# complete pair scores 1.0 (win + loss) and pair 0 sits at a finished 1.0 with
# nothing beside it: imputed as a draw it would enter the sample as 1.5, a bin no
# complete pair in this stream can occupy, and it would drag the LLR toward H1
# and past the stopping point.
_HALF_TAG = "arena_sprt_test_coloring"   # (opening fen, candidate plays White)

# Identity is load-bearing: the loop hands each side's boards to that side's
# model, which is how a board learns which coloring it is.
_CANDIDATE_MODEL = "candidate-model"
_REFERENCE_MODEL = "reference-model"

# Complete pairs all land in the middle bin, so this is the all-draw LLR path at
# a tighter boundary — but on the schedule this stagger produces. One slot of the
# pool of 4 is occupied forever by the hung game, so the other three retire pairs
# at 1, 2, 4, 5, 7, ... and the look that would have landed on 15 lands on 16.
STAGGERED_CROSS_AT_PAIRS = 16


@pytest.fixture
def staggered_decisive_games(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[chess.Board], None]:
    """One-ply decisive games, and a registrable opening whose Black half hangs.

    Both halves are production paths, not stubs of the thing under test:

    * the ply-0 pick tags each board with its coloring. At ply 0 White is to
      move, so the model the loop asks IS the side the candidate plays there —
      that partition (``split_active_by_side_to_move``) is the loop's own.
    * results come from the syzygy adjudication hook, consulted on every reap,
      so a game ends exactly when this fixture says it does. Every finished game
      is ``1-0``: a WIN for the candidate's White half, a LOSS for its Black
      half, hence 1.0 for the pair and a DECISIVE 1.0 for a lone half.

    Returns the registration function: ``hang(opening)`` makes that opening's
    Black half never finish, which is the half-played pair the guard has to drop.
    """
    hung: set[str] = set()

    def fake_pick(model: object, sub_boards: list[chess.Board],
                  **_kwargs: object) -> list[int]:
        for board in sub_boards:
            if board.move_stack:
                continue
            assert board.turn == chess.WHITE, (
                "the tagging rests on White being to move at ply 0"
            )
            setattr(board, _HALF_TAG, (board.fen(), model is _CANDIDATE_MODEL))
        return [0] * len(sub_boards)

    def fake_apply(boards: list[chess.Board], idxs: list[int],
                   _actions: list[int], *, strict: bool) -> None:
        assert strict, "the arena must decode actions strictly"
        for i in idxs:
            boards[i].push(next(iter(boards[i].legal_moves)))

    def fake_adjudicate(board: chess.Board, _tablebase: object, *,
                        max_pieces: int) -> str | None:
        assert max_pieces > 0
        tag: tuple[str, bool] | None = getattr(board, _HALF_TAG, None)
        if tag is None or not board.move_stack:
            return None                      # nothing played yet
        opening_fen, cand_is_white = tag
        if opening_fen in hung and not cand_is_white:
            return None                      # the coloring that never finishes
        return "1-0"                         # White wins, always, after one ply

    import scripts.match_vs_uci as uci_mod

    monkeypatch.setattr(match_mod, "pick_moves_for_boards", fake_pick)
    monkeypatch.setattr(match_mod, "apply_actions_to_boards", fake_apply)
    monkeypatch.setattr(uci_mod, "_tb_adjudicate_result", fake_adjudicate)

    def hang(opening: chess.Board) -> None:
        hung.add(opening.fen())

    return hang


class RecordingMonitor(SprtMonitor):
    """Banks the EXACT sample each look was handed, and what had finished by then.

    The stop point is one observation of the guard; the sample is the other, and
    it is the direct one — it says what the boundary was TESTED AGAINST rather
    than what the loop did afterwards. The finished-game count is banked with it
    so each sample can be checked against the truth AT THAT LOOK, which is the
    only way the claim survives a mutant that changes when the loop stops.
    """

    def __init__(
        self, *args: Any, finished: list[dict[str, Any]], **kwargs: Any,
    ) -> None:
        # Both before super(): SprtMonitor.__init__ takes the first look itself.
        self.samples: list[list[float]] = []
        self.finished_at_look: list[int] = []
        self._finished = finished
        super().__init__(*args, **kwargs)

    def update(self, new_pair_scores: Sequence[float]) -> str | None:
        self.samples.append([float(s) for s in new_pair_scores])
        self.finished_at_look.append(len(self._finished))
        return super().update(new_pair_scores)


def _pairs_from_finished(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[float], dict[int, float]]:
    """``(complete pair scores, {pair_id: the lone coloring's score})``.

    Rebuilt from the production game sink rather than from the test's intent, so
    "one coloring finished, and it was decisive" is an OBSERVATION of the run.
    """
    by_pair: dict[int, dict[int, float]] = {}
    for row in rows:
        by_pair.setdefault(int(row["pair_id"]), {})[int(row["half"])] = (
            arena.score_from_result(
                str(row["result"]), a_is_white=bool(row["a_is_white"]),
            )
        )
    complete = [sum(h.values()) for _, h in sorted(by_pair.items()) if len(h) == 2]
    lone = {p: next(iter(h.values())) for p, h in by_pair.items() if len(h) == 1}
    return complete, lone


def _play_rolling_staggered(
    openings: list[chess.Board], *, sprt: SprtMonitor | None,
    finished: list[dict[str, Any]],
) -> list[float]:
    def _sink(**row: Any) -> None:
        finished.append(row)

    return play_paired_games_matched_sims_rolling(
        _CANDIDATE_MODEL, _REFERENCE_MODEL, openings,
        device="cpu", rng=np.random.default_rng(7),
        sims_candidate=1, sims_reference=1,
        # High enough that the hung game is never adjudicated at the cap before
        # the boundary stops the loop: max_plies would score it 0.5 and complete
        # the very pair this test needs to stay half played.
        max_plies=200,
        temperature=0.1, gumbel_add_noise=False,
        search_candidate=_search(), search_reference=_search(),
        syzygy_tablebase=object(), tb_max_pieces=6,
        pool_size=4, report_every=10_000, sprt=sprt, pgn_sink=_sink,
    )


def test_the_rolling_look_never_sees_a_half_played_pair(
    staggered_decisive_games: Callable[[chess.Board], None],
) -> None:
    """The guard, on a stream where imputing the missing coloring changes it.

    MUTANT, run: replace the look's input with a version that imputes a
    half-played pair's missing coloring as a 0.5 draw. Pair 0 then enters every
    sample as 1.5, the LLR is dragged toward H1, and the H0 crossing moves.
    Measured across this file plus test_arena_standard.py and test_arena.py:
    **1 failed, 101 passed** — this test is the only one that can see it.
    """
    openings = [chess.Board(fen) for fen in _distinct_openings(40)]
    staggered_decisive_games(openings[0])
    finished: list[dict[str, Any]] = []
    monitor = RecordingMonitor(
        SPEC_TIGHT, pairs_cap=40, granularity="pair", finished=finished,
    )
    scores = _play_rolling_staggered(openings, sprt=monitor, finished=finished)

    # (1) THE GUARD, look by look and against the truth AT THAT LOOK: the sample
    # IS the complete pairs. A lone coloring is absent from it, not folded in at
    # 0.5 — and this is checked at every look, so it does not depend on where the
    # loop happened to stop.
    decisive_lone_looks = 0
    for sample, n_finished in zip(monitor.samples, monitor.finished_at_look):
        complete, lone = _pairs_from_finished(finished[:n_finished])
        assert sample == complete
        if any(score in (0.0, 1.0) for score in lone.values()):
            decisive_lone_looks += 1

    # (2) ... and the case that makes (1) non-vacuous actually occurred: a look
    # landed while a pair had exactly ONE coloring finished and that coloring
    # was DECISIVE. Imputing its partner as a draw would have entered it as 1.5,
    # a bin no complete pair in this stream can occupy.
    assert decisive_lone_looks, (
        "no look landed on a half-played pair with a DECISIVE lone coloring, so "
        "imputing that partner as a draw would change nothing and this test "
        "cannot see the guard"
    )

    # (3) And the stop point, which is what the imputed sample would move.
    assert monitor.verdict == "H0"
    assert monitor.pairs == STAGGERED_CROSS_AT_PAIRS
    assert len(scores) == STAGGERED_CROSS_AT_PAIRS
    assert set(scores) == {1.0}


# ---- matched_time ---------------------------------------------------------

def _result_for(score: float, *, a_is_white: bool) -> str:
    if score == 0.5:
        return "1/2-1/2"
    return "1-0" if ((score == 1.0) == a_is_white) else "0-1"


_PAIR_HALVES: dict[float, tuple[float, float]] = {
    2.0: (1.0, 1.0), 1.5: (1.0, 0.5), 1.0: (0.5, 0.5),
    0.5: (0.5, 0.0), 0.0: (0.0, 0.0),
}


@pytest.fixture
def scripted_uci(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the two engine helpers the matched_time loop imports.

    The loop itself — the pairing, the per-pair scoring, the SPRT look — is
    production code. The plan gives the candidate a genuinely asymmetric,
    multi-bin pair distribution, which the paired matched_sims stubs cannot.
    """
    import scripts.match_vs_uci as uci_mod

    def fake_open_engine(_cmd: str, **_kw: object) -> Any:
        return SimpleNamespace(quit=lambda: None)

    def fake_play_one_game(
        _eng_w: Any, _eng_b: Any, *, start_board: chess.Board,
        game: tuple[int, bool], **_kw: Any,
    ) -> Any:
        pair_idx, a_is_white = game
        half = 0 if a_is_white else 1
        score = _PAIR_HALVES[STRONG_CYCLE[pair_idx % 4]][half]
        return SimpleNamespace(
            result=_result_for(score, a_is_white=a_is_white),
            start_board=start_board, moves=(), termination="rules", plies=12,
        )

    monkeypatch.setattr(uci_mod, "_open_engine", fake_open_engine)
    monkeypatch.setattr(uci_mod, "play_one_game", fake_play_one_game)


@pytest.mark.usefixtures("scripted_uci")
def test_matched_time_loop_stops_on_the_boundary_after_a_completed_pair() -> None:
    monitor = SprtMonitor(SPEC, pairs_cap=200, granularity="pair")
    scores = play_paired_games_matched_time(
        "cand.pt", "ref.pt", [chess.Board() for _ in range(200)],
        device="cpu", ms_per_move=1, max_plies=20, uci_args="", sprt=monitor,
    )
    assert monitor.verdict == "H1"
    assert len(scores) == STRONG_CROSS_AT_PAIRS
    assert monitor.pairs == STRONG_CROSS_AT_PAIRS
    assert scores == [STRONG_CYCLE[i % 4] for i in range(STRONG_CROSS_AT_PAIRS)]


@pytest.mark.usefixtures("scripted_uci")
def test_matched_time_loop_plays_on_when_the_boundary_is_moved() -> None:
    """MUTANT: the same 60 pairs against an unreachable boundary."""
    monitor = SprtMonitor(SPEC_WIDE, pairs_cap=60, granularity="pair")
    scores = play_paired_games_matched_time(
        "cand.pt", "ref.pt", [chess.Board() for _ in range(60)],
        device="cpu", ms_per_move=1, max_plies=20, uci_args="", sprt=monitor,
    )
    assert monitor.verdict is None
    assert len(scores) == 60


@pytest.mark.usefixtures("scripted_uci")
def test_matched_time_loop_without_sprt_is_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    scores = play_paired_games_matched_time(
        "cand.pt", "ref.pt", [chess.Board() for _ in range(6)],
        device="cpu", ms_per_move=1, max_plies=20, uci_args="",
    )
    assert len(scores) == 6
    assert "SPRT" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_arena: the chunked loop, the record, and resume
# ---------------------------------------------------------------------------

def _distinct_openings(n: int) -> list[str]:
    """``n`` distinct, legal, non-terminal two-ply positions."""
    out: list[str] = []
    root = chess.Board()
    for m1 in root.legal_moves:
        after_first = root.copy()
        after_first.push(m1)
        for m2 in after_first.legal_moves:
            board = after_first.copy()
            board.push(m2)
            out.append(board.fen())
            if len(out) == n:
                return out
    raise AssertionError(f"could only build {len(out)} openings, wanted {n}")


def _openings_file(tmp_path: Path, n: int) -> Path:
    path = tmp_path / f"openings{n}.fen"
    path.write_text("\n".join(_distinct_openings(n)) + "\n")
    return path


def _fake_chunk_play(calls: list[int]) -> Callable[..., list[float]]:
    """Stand-in for one CHUNK of matched_sims: every game a draw, all emitted.

    The chunk function is stubbed and ``run_arena``'s between-chunk stop check
    is not: that check is the thing under test, and it lives in ``run_arena``
    precisely because a chunk cannot be stopped part-way without imputing its
    unfinished games.
    """
    def _play(
        _cand: object, _ref: object, openings: list[chess.Board], *,
        pgn_sink: Any = None, pair_ids: Sequence[int] | None = None,
        chunk: int | None = None, **_kw: Any,
    ) -> list[float]:
        calls.append(len(openings))
        ids = list(range(len(openings))) if pair_ids is None else list(pair_ids)
        for k, opening in enumerate(openings):
            for half, a_is_white in ((0, True), (1, False)):
                if pgn_sink is not None:
                    pgn_sink(
                        pair_id=ids[k], half=half, a_is_white=a_is_white,
                        start_fen=opening.fen(), moves=(), result="1/2-1/2",
                        termination="rules", plies=11, duration_s=0.1,
                        chunk=chunk, loop="chunked",
                    )
        return [1.0] * len(openings)
    return _play


def _run_chunked_arena(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    sprt: SprtSpec | None,
    n_pairs: int,
    calls: list[int],
    log_path: Path | None = None,
    resume: bool = False,
) -> dict:
    import chess_anti_engine.uci.model_loader as loader

    monkeypatch.setattr(
        loader, "load_model_from_checkpoint", lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(arena, "play_paired_games_matched_sims", _fake_chunk_play(calls))
    return arena.run_arena(
        candidate="cand.pt", reference="ref.pt",
        games=2 * n_pairs,
        openings_path=None, openings_fen=_openings_file(tmp_path, n_pairs),
        opening_plies=16, mode="matched_sims",
        sims_candidate=1, sims_reference=1, ms_per_move=100, max_plies=20,
        temperature=0.1, gumbel_add_noise=False, device="cpu", seed=11,
        out_path=None,
        game_log_path=log_path or (tmp_path / "chunked.games.jsonl"),
        resume=resume,
        max_concurrent_games=4, eval_max_batch=0, compile_models=False,
        rolling=False,
        search_candidate=_search(), search_reference=_search(),
        sprt=sprt,
    )


def _run_loop_arena(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    sprt: SprtSpec | None,
    n_pairs: int,
    mode: str = "matched_sims",
    log_name: str = "loop.games.jsonl",
) -> dict:
    """``run_arena`` driving a REAL play loop — rolling or matched_time.

    ⚑ This exists because of a mutant that SURVIVED: deleting
    ``sprt=sprt_monitor`` from ``run_arena``'s call to the rolling loop broke
    nothing, since every other loop test calls the loop function directly and
    every other ``run_arena`` test goes down the chunked path. The call site was
    the one link in the chain nothing observed — a knob that never reaches the
    consumer, which is exactly the failure this whole feature is checked for.
    Nothing here stubs the loop; only the model loader and the move source.
    """
    import chess_anti_engine.uci.model_loader as loader

    monkeypatch.setattr(
        loader, "load_model_from_checkpoint", lambda *_a, **_k: object(),
    )
    matched_sims = mode == "matched_sims"
    return arena.run_arena(
        candidate="cand.pt", reference="ref.pt",
        games=2 * n_pairs,
        openings_path=None, openings_fen=_openings_file(tmp_path, n_pairs),
        opening_plies=16, mode=mode,
        sims_candidate=1, sims_reference=1, ms_per_move=1,
        # matched_sims: one ply, then every game is adjudicated a draw.
        max_plies=1 if matched_sims else 20,
        temperature=0.1, gumbel_add_noise=False, device="cpu", seed=13,
        out_path=None, game_log_path=tmp_path / log_name,
        max_concurrent_games=4, eval_max_batch=0, compile_models=False,
        report_every=10_000, rolling=True,
        search_candidate=_search() if matched_sims else None,
        search_reference=_search() if matched_sims else None,
        sprt=sprt,
    )


@pytest.mark.usefixtures("scripted_moves")
def test_run_arena_hands_the_boundary_to_the_rolling_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    record = _run_loop_arena(monkeypatch, tmp_path, sprt=SPEC, n_pairs=200)
    assert record["sprt"]["verdict"] == "H0"
    assert record["sprt"]["check_granularity"] == "pair"
    assert record["sprt"]["pairs"] == ALL_DRAW_CROSS_AT_PAIRS
    assert record["pairs"] == ALL_DRAW_CROSS_AT_PAIRS
    assert record["truncated"] is True
    assert record["game_log_agrees"] is True


@pytest.mark.usefixtures("scripted_moves")
def test_run_arena_rolling_plays_the_schedule_when_the_boundary_moves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    record = _run_loop_arena(monkeypatch, tmp_path, sprt=SPEC_WIDE, n_pairs=20)
    assert record["sprt"]["verdict"] == "INCONCLUSIVE"
    assert record["pairs"] == 20
    assert record["truncated"] is False


@pytest.mark.usefixtures("scripted_uci")
def test_run_arena_hands_the_boundary_to_the_matched_time_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    record = _run_loop_arena(
        monkeypatch, tmp_path, sprt=SPEC, n_pairs=200, mode="matched_time",
        log_name="mt.games.jsonl",
    )
    assert record["sprt"]["verdict"] == "H1"
    assert record["sprt"]["check_granularity"] == "pair"
    assert record["sprt"]["pairs"] == STRONG_CROSS_AT_PAIRS
    assert record["pairs"] == STRONG_CROSS_AT_PAIRS


def test_chunked_loop_stops_between_chunks_and_records_that_granularity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[int] = []
    record = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC, n_pairs=60, calls=calls,
    )
    sprt_record = record["sprt"]
    assert sprt_record["verdict"] == "H0"
    assert sprt_record["check_granularity"] == "chunk"
    assert sprt_record["pairs"] == ALL_DRAW_CROSS_AT_PAIRS
    assert sprt_record["stopped_early"] is True
    # --max-concurrent-games 4 => 2 pairs per chunk => 25 chunks, not 30.
    assert calls == [2] * (ALL_DRAW_CROSS_AT_PAIRS // 2)
    assert record["pairs"] == ALL_DRAW_CROSS_AT_PAIRS
    assert record["games_requested"] == 120
    assert record["truncated"] is True
    assert record["game_log_agrees"] is True


def test_chunked_loop_plays_every_chunk_when_the_boundary_moves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """MUTANT: unreachable boundary, same stubbed games, all 30 chunks played."""
    calls: list[int] = []
    record = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC_WIDE, n_pairs=60, calls=calls,
    )
    assert record["sprt"]["verdict"] == "INCONCLUSIVE"
    assert record["sprt"]["stop_reason"] == "cap"
    assert record["sprt"]["stopped_early"] is False
    assert calls == [2] * 30
    assert record["pairs"] == 60
    assert record["truncated"] is False


def test_without_sprt_the_record_has_no_sprt_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default is byte-comparable with every row already in the aggregate."""
    calls: list[int] = []
    record = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=None, n_pairs=6, calls=calls,
    )
    assert "sprt" not in record, (
        "a fixed-N run must not grow a key: runs/arena_results.jsonl is a "
        "shared append-only aggregate with years of rows in it"
    )
    assert calls == [2, 2, 2]
    assert "SPRT" not in capsys.readouterr().out


def test_the_sprt_key_sits_between_the_elo_and_the_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Adding the block must not reorder the fields around it."""
    calls: list[int] = []
    plain = _run_chunked_arena(monkeypatch, tmp_path, sprt=None, n_pairs=6, calls=calls)
    calls2: list[int] = []
    seq = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC_WIDE, n_pairs=6, calls=calls2,
        log_path=tmp_path / "seq.games.jsonl",
    )
    assert [k for k in seq if k != "sprt"] == list(plain)
    keys = list(seq)
    assert keys[keys.index("sprt") - 1] == "elo_ci95"
    assert keys[keys.index("sprt") + 1] == "duration_s"


def test_a_resumed_arena_recomputes_the_llr_over_loaded_and_new_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Segment 1 stops at a tight boundary; segment 2 resumes and re-decides.

    Three claims in one run, because they are one mechanism: the resumed pairs
    reach the statistic (segment 2 starts at 16, not 0), a run that already
    crossed does not spend GPU time re-proving it (zero chunks played), and the
    same log resumed WITHOUT --sprt still plays out the remainder (so the skip
    is the boundary's doing and not a resume bug).
    """
    log_path = tmp_path / "resumed.games.jsonl"
    first: list[int] = []
    seg1 = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC_TIGHT, n_pairs=60, calls=first,
        log_path=log_path,
    )
    assert seg1["sprt"]["pairs"] == ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME
    assert seg1["sprt"]["resumed_pairs"] == 0
    assert sum(first) == ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME

    second: list[int] = []
    seg2 = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC_TIGHT, n_pairs=60, calls=second,
        log_path=log_path, resume=True,
    )
    assert second == [], "the boundary was already crossed; play nothing"
    assert seg2["sprt"]["resumed_pairs"] == ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME
    assert seg2["sprt"]["pairs"] == ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME
    assert seg2["sprt"]["verdict"] == "H0"
    assert seg2["sprt"]["llr"] == pytest.approx(seg1["sprt"]["llr"], abs=1e-12)
    assert seg2["resumed_pairs"] == ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME

    third: list[int] = []
    seg3 = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=None, n_pairs=60, calls=third,
        log_path=log_path, resume=True,
    )
    assert sum(third) == 60 - ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME
    assert seg3["pairs"] == 60
    assert "sprt" not in seg3


def test_a_resumed_arena_continues_toward_a_boundary_it_has_not_reached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The other half of resume: 16 banked pairs are evidence, not a restart.

    Segment 1 crosses the TIGHT boundary at 16 pairs. Segment 2 resumes the same
    log under the standard boundary, which needs 50 — and must play exactly the
    34 further pairs, not another 50.
    """
    log_path = tmp_path / "carryover.games.jsonl"
    first: list[int] = []
    _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC_TIGHT, n_pairs=60, calls=first,
        log_path=log_path,
    )
    second: list[int] = []
    seg2 = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC, n_pairs=60, calls=second,
        log_path=log_path, resume=True,
    )
    assert seg2["sprt"]["verdict"] == "H0"
    assert seg2["sprt"]["pairs"] == ALL_DRAW_CROSS_AT_PAIRS
    assert seg2["sprt"]["resumed_pairs"] == ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME
    assert sum(second) == ALL_DRAW_CROSS_AT_PAIRS - ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME


# ---------------------------------------------------------------------------
# The spec in the game-log header: recorded, not fingerprinted
# ---------------------------------------------------------------------------
#
# A verdict is unreadable a month later without the hypothesis it was decided
# against, and the JSONL result row is not always what survives (a crashed run
# leaves only the game log). So the spec is banked in the log header — and
# banked BESIDE the fingerprinted settings, never in them: the fingerprint is a
# hash of the whole settings dict, so a spec inside it would refuse to resume
# every pre-branch log and would refuse the deliberate "resume a crashed fixed-N
# arena as a sequential test".

def _log_header(path: Path) -> dict[str, Any]:
    return read_game_log(path).header


def test_the_log_header_records_the_spec_beside_the_fingerprinted_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """MUTANT, run: pass ``info=None`` to ``GameLogWriter`` unconditionally, so
    the spec is accepted and then never written. 3 failed, 99 passed.
    """
    calls: list[int] = []
    _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC_WIDE, n_pairs=6, calls=calls,
        log_path=tmp_path / "withspec.games.jsonl",
    )
    header = _log_header(tmp_path / "withspec.games.jsonl")
    assert header["info"]["sprt"]["elo0"] == SPEC_WIDE.elo0
    assert header["info"]["sprt"]["elo1"] == SPEC_WIDE.elo1
    assert header["info"]["sprt"]["alpha"] == SPEC_WIDE.alpha
    assert header["info"]["sprt"]["beta"] == SPEC_WIDE.beta
    assert "sprt" not in header["settings"], (
        "everything under settings is hashed into the resume fingerprint"
    )

    # The same run without --sprt: no info block at all (the pre-branch header
    # shape), and — the load-bearing half — the SAME fingerprint, so the two
    # logs are resumable from each other.
    plain_path = tmp_path / "nospec.games.jsonl"
    _run_chunked_arena(
        monkeypatch, tmp_path, sprt=None, n_pairs=6, calls=[], log_path=plain_path,
    )
    plain = _log_header(plain_path)
    assert "info" not in plain
    assert header["fingerprint"] == plain["fingerprint"]


def test_a_resume_under_a_different_spec_warns_and_still_carries_the_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Warned about, never refused — and the warning names BOTH hypotheses.

    MUTANT, run: make ``sprt_spec_carryover_warning`` return None
    unconditionally. 6 failed, 96 passed — this test, its present-vs-absent
    twin, and four of the parametrized cases below.
    """
    log_path = tmp_path / "respec.games.jsonl"
    _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC_TIGHT, n_pairs=60, calls=[],
        log_path=log_path,
    )
    capsys.readouterr()
    seg2 = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC, n_pairs=60, calls=[],
        log_path=log_path, resume=True,
    )
    err = capsys.readouterr().err
    assert "DIFFERENT SPRT hypothesis" in err
    assert f"alpha={SPEC_TIGHT.alpha}" in err, "the RECORDED spec must be named"
    assert f"alpha={SPEC.alpha}" in err, "this invocation's spec must be named too"
    # A warning, not a refusal: the resumed pairs still carry over.
    assert seg2["sprt"]["resumed_pairs"] == ALL_DRAW_CROSS_TIGHT_TWO_AT_A_TIME


def test_a_resume_that_drops_the_flag_is_warned_about_as_present_vs_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other direction: the log has a spec, this invocation has none."""
    log_path = tmp_path / "dropped.games.jsonl"
    _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC_TIGHT, n_pairs=60, calls=[],
        log_path=log_path,
    )
    capsys.readouterr()
    seg2 = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=None, n_pairs=60, calls=[],
        log_path=log_path, resume=True,
    )
    err = capsys.readouterr().err
    assert "DIFFERENT SPRT hypothesis" in err
    assert f"alpha={SPEC_TIGHT.alpha}" in err
    assert "fixed-N" in err
    assert "sprt" not in seg2


def test_resuming_a_log_that_records_no_spec_neither_warns_nor_crashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pre-branch log shape, and the fixed-N -> sequential resume.

    A fixed-N run writes a header with no info block — byte-identical to every
    log written before this feature existed. Resuming one under a boundary is
    deliberate and there is no earlier hypothesis to contradict, so it must be
    silent: a warning on every legitimate first sequential resume is a warning
    nobody reads by the third time they see it.
    """
    log_path = tmp_path / "prebranch.games.jsonl"
    _run_chunked_arena(
        monkeypatch, tmp_path, sprt=None, n_pairs=60, calls=[], log_path=log_path,
    )
    assert "info" not in _log_header(log_path)
    capsys.readouterr()
    seg2 = _run_chunked_arena(
        monkeypatch, tmp_path, sprt=SPEC, n_pairs=60, calls=[],
        log_path=log_path, resume=True,
    )
    err = capsys.readouterr().err
    assert "SPRT" not in err
    assert seg2["sprt"]["resumed_pairs"] == 60
    assert seg2["sprt"]["verdict"] == "H0"


@pytest.mark.parametrize(
    ("recorded", "current", "expected"),
    [
        (None, None, False),
        (None, SPEC, False),                       # fixed-N log resumed as SPRT
        (SPEC.as_record(), SPEC, False),           # the same hypothesis
        (SPEC.as_record(), None, True),
        (SPEC.as_record(), SPEC_TIGHT, True),
        ({}, SPEC, True),                          # an info block with no numbers
        ({**SPEC.as_record(), "alpha": "0.05"}, SPEC, True),   # a string, not 0.05
    ],
)
def test_the_carryover_warning_fires_on_exactly_the_differences(
    recorded: dict[str, Any] | None, current: SprtSpec | None, expected: bool,
) -> None:
    """Including the one that is not a number: a header can be hand-edited."""
    fired = arena.sprt_spec_carryover_warning(recorded, current) is not None
    assert fired is expected


# ---------------------------------------------------------------------------
# The CLI: the last link, argv -> run_arena
# ---------------------------------------------------------------------------

def _drive_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str],
) -> dict[str, Any]:
    """Run ``main()`` with a captured ``run_arena`` and return its kwargs."""
    seen: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> dict[str, object]:
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(arena, "run_arena", _capture)
    monkeypatch.setattr("sys.argv", [
        "arena_standard.py",
        "--candidate", "cand.pt", "--reference", "ref.pt",
        "--mode", "matched_time",
        "--openings-fen", str(_openings_file(tmp_path, 4)),
        "--games", "8",
        *extra,
    ])
    arena.main()
    return seen


def test_the_cli_string_reaches_run_arena_as_a_parsed_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    seen = _drive_main(
        monkeypatch, tmp_path, ["--sprt", "elo0=-1,elo1=6,alpha=0.04,beta=0.09"],
    )
    assert seen["sprt"] == SprtSpec(elo0=-1.0, elo1=6.0, alpha=0.04, beta=0.09)


def test_without_the_flag_run_arena_is_called_with_no_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    assert _drive_main(monkeypatch, tmp_path, [])["sprt"] is None


def test_a_malformed_sprt_string_exits_before_any_game_is_played(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="missing beta"):
        _drive_main(monkeypatch, tmp_path, ["--sprt", "elo0=0,elo1=5,alpha=0.05"])


def test_the_flag_is_not_offered_on_the_shared_arena_parser() -> None:
    """``scripts/elo_vs_sims.py`` also builds from ``add_common_args``.

    It constructs its own ``run_arena`` calls and would not forward ``--sprt``,
    so putting the flag in the shared helper would create one that parses,
    prints and then decides nothing — the accepted-and-ignored defect, newly
    introduced by the fix for it. The flag is declared in ``main()`` instead,
    and this pins that.
    """
    import argparse

    parser = argparse.ArgumentParser()
    arena.add_common_args(parser)
    flags = {opt for action in parser._actions for opt in action.option_strings}
    assert "--sprt" not in flags
    assert "--games" in flags  # the helper really was populated
