"""`--blunder-taus`: the conjugacy-neutral blunder-mass ruler on audit_targets.

Two things are pinned here. First that the statistic is right. Second, and the
reason this flag is shaped the way it is, that **passing nothing changes
nothing**: `scripts/audit_targets.py` is a frozen ruler whose numbers are quoted
across months of ledger entries, so an additive flag that perturbed the default
report would silently invalidate every one of them.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from scripts import audit_targets as at

from chess_anti_engine.eval.audit import (
    expected_and_top1_regret,
    expected_blunder_rates,
)

TAUS = (50.0, 100.0, 200.0)


# --------------------------------------------------------------------------
# the statistic
# --------------------------------------------------------------------------


def test_rates_match_hand_computation() -> None:
    probs = np.array([0.5, 0.2, 0.2, 0.1])
    regrets = np.array([0.0, 60.0, 150.0, 500.0])
    #      >50 : 0.2 + 0.2 + 0.1 = 0.5
    #     >100 :       0.2 + 0.1 = 0.3
    #     >200 :             0.1 = 0.1
    assert expected_blunder_rates(probs, regrets, TAUS) == pytest.approx(
        (0.5, 0.3, 0.1),
    )


def test_empty_taus_returns_empty_tuple() -> None:
    """The value the whole default-path guarantee is built on."""
    probs = np.array([0.7, 0.3])
    regrets = np.array([0.0, 900.0])
    assert expected_blunder_rates(probs, regrets, ()) == ()


def test_strictly_at_threshold_is_not_a_blunder() -> None:
    """`> tau`, not `>= tau` — a move losing exactly tau is on the good side."""
    probs = np.array([0.5, 0.5])
    regrets = np.array([0.0, 100.0])
    assert expected_blunder_rates(probs, regrets, (100.0,)) == pytest.approx((0.0,))
    assert expected_blunder_rates(probs, regrets, (99.0,)) == pytest.approx((0.5,))


def test_unnormalized_probs_are_normalized() -> None:
    probs = np.array([5.0, 5.0])  # sums to 10, not 1
    regrets = np.array([0.0, 900.0])
    assert expected_blunder_rates(probs, regrets, (100.0,)) == pytest.approx((0.5,))


def test_degenerate_distribution_matches_the_sibling_metric_convention() -> None:
    """A non-positive total is treated as uniform — the same convention
    `expected_and_top1_regret` uses, so the two never disagree about what a
    row's distribution was."""
    probs = np.zeros(4)
    regrets = np.array([0.0, 0.0, 300.0, 300.0])
    assert expected_blunder_rates(probs, regrets, (100.0,)) == pytest.approx((0.5,))
    # sibling metric agrees the distribution was uniform
    exp_r, _ = expected_and_top1_regret(probs, regrets)
    assert exp_r == pytest.approx(150.0)


def test_rates_are_monotone_non_increasing_in_tau() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 30))
        probs = rng.random(n)
        regrets = rng.random(n) * 600.0
        rates = expected_blunder_rates(probs, regrets, TAUS)
        assert all(a >= b - 1e-12 for a, b in itertools.pairwise(rates))
        assert all(0.0 <= r <= 1.0 for r in rates)


def test_sharpening_onto_the_best_move_lowers_blunder_mass() -> None:
    """POSITIVE CONTROL: the ruler must resolve a known-better distribution."""
    regrets = np.array([0.0, 400.0, 400.0, 400.0])
    flat = np.full(4, 0.25)
    sharp = np.array([0.97, 0.01, 0.01, 0.01])
    assert expected_blunder_rates(sharp, regrets, (100.0,))[0] < (
        expected_blunder_rates(flat, regrets, (100.0,))[0]
    )


def test_shuffled_labels_destroy_discrimination() -> None:
    """NEGATIVE CONTROL, shipped as a test (repo standing rule).

    A distribution that concentrates on the genuinely best moves must lose its
    advantage once the regret vector is permuted relative to it. If the metric
    still separated them, it would be reading the distribution's SHAPE rather
    than its attribution of mass to good moves.
    """
    rng = np.random.default_rng(7)
    n = 24
    regrets = np.sort(rng.random(n) * 800.0)
    informed = np.zeros(n)
    informed[:4] = 0.25  # all mass on the four lowest-regret moves
    real = expected_blunder_rates(informed, regrets, (100.0,))[0]

    shuffled = [
        expected_blunder_rates(informed, rng.permutation(regrets), (100.0,))[0]
        for _ in range(200)
    ]
    uninformed = float(np.mean([r > 100.0 for r in regrets]))
    assert real < uninformed - 0.1
    # Destroying the attribution collapses the statistic onto the uninformed
    # mass; the informed reading does not survive the shuffle.
    assert float(np.mean(shuffled)) == pytest.approx(uninformed, abs=0.08)


# --------------------------------------------------------------------------
# flag parsing
# --------------------------------------------------------------------------


def test_parse_absent_flag_is_off() -> None:
    assert at._parse_blunder_taus(None) == ()


def test_parse_sorts_and_deduplicates() -> None:
    assert at._parse_blunder_taus("200,50,100,50") == (50.0, 100.0, 200.0)


def test_parse_tolerates_whitespace_and_empty_chunks() -> None:
    assert at._parse_blunder_taus(" 100 , , 50 ") == (50.0, 100.0)
    assert at._parse_blunder_taus("") == ()


@pytest.mark.parametrize("spec", ["abc", "50,x", "nan", "-10", "inf"])
def test_parse_rejects_garbage_and_invalid_thresholds(spec: str) -> None:
    with pytest.raises(SystemExit):
        at._parse_blunder_taus(spec)


@pytest.mark.parametrize("spec", ["12.5", "50,100.25", "0.5"])
def test_parse_rejects_fractional_thresholds(spec: str) -> None:
    """A fractional tau names a key like `blunder12.5`, and the dot is a
    separator in `paired_compare.py`'s dotted `--field` path — so the column
    could never be addressed and every paired row would be unusable. Rejected
    at parse time rather than after a full GPU scoring pass."""
    with pytest.raises(SystemExit, match="whole number of centipawns"):
        at._parse_blunder_taus(spec)


def test_parse_accepts_whole_numbers_written_with_a_decimal_point() -> None:
    assert at._parse_blunder_taus("50.0,100") == (50.0, 100.0)


def test_dump_key_is_the_flat_dotted_path_the_paired_reader_uses() -> None:
    """`paired_compare.py --field cand.raw.blunder100` resolves a dotted path,
    so the key must be flat inside the per-candidate dict, not nested — and,
    for the same reason, must never contain a dot itself."""
    assert at._blunder_key(100.0) == "blunder100"
    assert at._blunder_key(50.0) == "blunder50"
    for tau in at._parse_blunder_taus("50,100,200"):
        assert "." not in at._blunder_key(tau)


# --------------------------------------------------------------------------
# DEFAULT-PATH INVARIANCE — the reason the flag is shaped this way
# --------------------------------------------------------------------------

GROUPS = ["overall", "endgame", "middlegame", "opening"]
LABELS = {"raw": "(a) raw", "search": "(b) search"}


def _rows() -> list[dict]:
    return [
        {"cand": "raw", "phase": 0, "source": 0, "rates": (0.4, 0.2, 0.1)},
        {"cand": "raw", "phase": 1, "source": 0, "rates": (0.2, 0.1, 0.0)},
        {"cand": "search", "phase": 0, "source": 1, "rates": (0.1, 0.05, 0.0)},
    ]


def test_section_is_empty_string_without_the_flag() -> None:
    """The report is built by unconditional concatenation, so "" here is
    exactly what makes the default report byte-identical."""
    assert at._blunder_section(_rows(), GROUPS, LABELS, ()) == ""


def test_section_is_empty_string_when_no_rows_were_collected() -> None:
    assert at._blunder_section([], GROUPS, LABELS, TAUS) == ""


def test_section_is_non_empty_and_labelled_when_the_flag_is_passed() -> None:
    out = at._blunder_section(_rows(), GROUPS, LABELS, TAUS)
    assert out.startswith("\n## Blunder mass")
    # one row per (candidate, tau)
    for label in LABELS.values():
        for tau in TAUS:
            assert f"| {label} — >{tau:g} cp |" in out
    # and it says why it exists, so a reader cannot mistake it for the
    # conjugate E[regret] table above it
    assert "CONJUGACY-NEUTRAL" in out


def test_aggregate_blunders_means_and_counts() -> None:
    agg = at._aggregate_blunders(_rows(), 3)
    # 'raw' overall = mean of the two raw rows
    rates, n = agg[("overall", "raw")]
    assert n == 2
    assert rates == pytest.approx((0.3, 0.15, 0.05))
    # phase buckets keep their own rows
    assert agg[("endgame", "raw")][1] == 1
    assert agg[("middlegame", "raw")][1] == 1


def test_aggregate_blunders_is_disjoint_from_the_frozen_aggregate() -> None:
    """`_aggregate` still returns the 3-tuple every existing caller unpacks;
    the new statistic does not widen it."""
    policy_rows = [
        {"cand": "raw", "phase": 0, "source": 0, "expected": 10.0, "top1": 5.0},
    ]
    v = at._aggregate(policy_rows, "cand")[("overall", "raw")]
    assert len(v) == 3
    assert v == (10.0, 5.0, 1)
