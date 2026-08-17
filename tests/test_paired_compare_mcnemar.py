"""McNemar's exact test and the pre-committed-n join guard.

Both exist for one reason: the #440 prereg retired its `>300cp` gate (which
could KILL but arithmetically could not PASS) in favour of "McNemar on
top1_match" — and `grep -rniE mcnemar` over the whole repo returned ZERO hits.
Retiring a gate that cannot pass in favour of a gate that has no command moves
"cannot fire" from arithmetic to absence; it does not fix it. These tests pin
the command that closes it.

⚑ The load-bearing assertions here are the two that a plausible-looking wrong
implementation would fail:

  * `half_width` must be computed from the MEASURED discordance. The defect it
    replaces was a half-width quoted from an ASSUMED d=0.30, which is a 3.2x
    spread across the plausible range — so a version that hard-codes any
    constant produces a number of exactly the right shape and the wrong size.
  * zero discordant pairs must read VOID, not "no difference". That is the
    input shape a served cache produces, and it is the single way this gate
    would silently certify an experiment that never measured anything.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.paired_compare import (
    McNemar,
    exact_binomial_two_sided,
    mcnemar,
    report_mcnemar,
    require_paired_n,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_exact_binomial_matches_hand_computed_values() -> None:
    # n=10, k=1: 2 * (C(10,0)+C(10,1)) / 2^10 = 2 * 11 / 1024
    assert exact_binomial_two_sided(1, 10) == pytest.approx(22 / 1024)
    # Symmetric in k: the tail is taken at min(k, n-k).
    assert exact_binomial_two_sided(9, 10) == pytest.approx(22 / 1024)
    # A perfectly split table is maximally uninformative.
    assert exact_binomial_two_sided(5, 10) == pytest.approx(1.0)
    # 0 of 8 one way: 2 * 1/256.
    assert exact_binomial_two_sided(0, 8) == pytest.approx(2 / 256)


def test_no_discordant_pairs_is_p_one_not_p_zero() -> None:
    """⚑ The empty table must not read as overwhelming evidence.

    `min(k, n-k)` with n=0 and an unguarded `2*tail/2**0` would return 2.0,
    clipped to 1.0 by luck rather than by intent. State it.
    """
    assert exact_binomial_two_sided(0, 0) == 1.0


def test_cells_follow_the_predicate_and_the_sign_convention() -> None:
    # A: 3 hits (<=0), B: 1 hit. b (A-only) = 2, c (B-only) = 0.
    va = np.array([0.0, 0.0, 0.0, 50.0])
    vb = np.array([0.0, 10.0, 20.0, 50.0])
    m = mcnemar(va, vb, at=0.0)
    assert (m.n, m.both, m.b, m.c, m.neither) == (4, 1, 2, 0, 1)
    assert m.discordance == pytest.approx(0.5)


def test_the_threshold_is_read_not_hard_coded() -> None:
    """`--mcnemar-at 0` and `--mcnemar-at 25` must disagree on the same input."""
    va = np.array([0.0, 10.0, 30.0, 30.0])
    vb = np.array([0.0, 30.0, 10.0, 30.0])
    assert mcnemar(va, vb, at=0.0).b == 0
    at25 = mcnemar(va, vb, at=25.0)
    assert (at25.b, at25.c) == (1, 1)


def test_half_width_tracks_the_measured_discordance() -> None:
    """⚑⚑ THE ASSUMED-d DEFECT, PINNED.

    Two inputs at the same n and different discordance must give different
    half-widths, in the ratio sqrt(d1/d2). Any implementation that quotes a
    constant d (the 0.30 the amendment assumed) passes every other test in this
    file and fails this one.
    """
    n = 400
    lo_a = np.zeros(n)
    lo_b = np.zeros(n)
    lo_b[:20] = 100.0                      # d = 0.05
    hi_a = np.zeros(n)
    hi_b = np.zeros(n)
    hi_b[:200] = 100.0                     # d = 0.50

    lo = mcnemar(lo_a, lo_b, at=0.0)
    hi = mcnemar(hi_a, hi_b, at=0.0)
    assert lo.discordance == pytest.approx(0.05)
    assert hi.discordance == pytest.approx(0.50)
    assert lo.half_width == pytest.approx(1.96 * (0.05 / n) ** 0.5)
    assert hi.half_width / lo.half_width == pytest.approx((0.50 / 0.05) ** 0.5)


def test_zero_discordance_is_reported_VOID_and_never_as_agreement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑⚑ The served-cache shape. This is the #440 B5 failure, in the readout.

    Two byte-identical dumps have b=c=0. A gate that prints "NOT significant"
    there hands back a clean null for an experiment in which the second arm was
    never measured. The word the operator must see is VOID.
    """
    v = np.array([0.0, 5.0, 500.0, 0.0])
    report_mcnemar(mcnemar(v, v.copy(), at=0.0), at=0.0, label_a="A", label_b="B")
    out = capsys.readouterr().out
    assert "VOID" in out
    assert "significant" not in out
    assert "0 discordant pairs" in out


def test_a_real_difference_is_reported_significant(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The complement: the VOID branch must not have eaten the live one."""
    va = np.zeros(100)
    vb = np.zeros(100)
    vb[:20] = 100.0      # 20 discordant, all one way -> p ~ 2e-6
    report_mcnemar(mcnemar(va, vb, at=0.0), at=0.0, label_a="A", label_b="B")
    out = capsys.readouterr().out
    assert "VOID" not in out
    assert "significant at 95%" in out
    assert "d_obs 0.2000" in out


def test_require_paired_n_refuses_a_short_join() -> None:
    """⚑ A truncated dump joins, prints, and is quotable. Make it fatal."""
    with pytest.raises(SystemExit) as exc:
        require_paired_n(500, 4000, label_a="A", label_b="B")
    assert "n=4000" in str(exc.value)
    assert "500" in str(exc.value)


def test_require_paired_n_is_opt_in_and_passes_on_an_exact_match() -> None:
    require_paired_n(4000, None, label_a="A", label_b="B")   # exploratory: no n
    require_paired_n(4000, 4000, label_a="A", label_b="B")


def _dump(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/paired_compare.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )


@pytest.fixture
def arm_pair(tmp_path: Path) -> tuple[str, str]:
    """40 positions; B loses top1_match on 6 of them and gains it on 1."""
    a_rows, b_rows = [], []
    for i in range(40):
        a_val = 0.0 if i < 30 else 40.0
        b_val = a_val
        if 0 <= i < 6:
            b_val = 40.0        # A-only match
        if i == 30:
            b_val = 0.0         # B-only match
        a_rows.append({"key": f"k{i}", "phase": 1, "value": a_val})
        b_rows.append({"key": f"k{i}", "phase": 1, "value": b_val})
    return (
        _dump(tmp_path / "a.jsonl", a_rows),
        _dump(tmp_path / "b.jsonl", b_rows),
    )


def test_cli_reports_mcnemar_only_when_asked(arm_pair: tuple[str, str]) -> None:
    a, b = arm_pair
    off = _run([a, b, "--join-key", "key", "--n-boot", "200"])
    assert off.returncode == 0, off.stderr
    assert "McNemar" not in off.stdout

    on = _run([a, b, "--join-key", "key", "--n-boot", "200", "--mcnemar-at", "0"])
    assert on.returncode == 0, on.stderr
    assert "A-only 6   B-only 1" in on.stdout, on.stdout
    assert "discordant 7" in on.stdout


def test_cli_require_n_exits_nonzero_on_the_wrong_n(arm_pair: tuple[str, str]) -> None:
    """⚑ Exit CODE, not just a printed warning — this gates a launch."""
    a, b = arm_pair
    bad = _run([a, b, "--join-key", "key", "--n-boot", "200", "--require-n", "4000"])
    assert bad.returncode != 0
    assert "--require-n 4000" in bad.stderr
    good = _run([a, b, "--join-key", "key", "--n-boot", "200", "--require-n", "40"])
    assert good.returncode == 0, good.stderr


def test_require_n_is_checked_before_any_statistic_is_printed(
    arm_pair: tuple[str, str],
) -> None:
    """A refused run must not also emit a verdict a reader could quote."""
    a, b = arm_pair
    r = _run([a, b, "--join-key", "key", "--n-boot", "200", "--require-n", "39",
              "--mcnemar-at", "0"])
    assert r.returncode != 0
    assert "verdict at 95%" not in r.stdout
    assert "McNemar" not in r.stdout


def test_mcnemar_namedtuple_carries_the_numbers_the_prereg_must_publish() -> None:
    """`d_obs` and `half_width` are outputs, not derivations the reader redoes."""
    m = mcnemar(np.zeros(100), np.r_[np.ones(10) * 99, np.zeros(90)], at=0.0)
    assert isinstance(m, McNemar)
    assert m.discordance == pytest.approx(0.10)
    assert m.half_width == pytest.approx(1.96 * (0.10 / 100) ** 0.5)


def test_the_test_survives_a_realistic_discordant_count() -> None:
    """⚑⚑ n=4000 is the registered size, not a stress case.

    `2.0**n` raises OverflowError above n=1023, so a float denominator makes
    the primary gate CRASH on the exact run it was written for while passing
    every small-table test above. Predict the regime before trusting the
    arithmetic. [[compute_instrument_resolution_before_the_threshold]]
    """
    p = exact_binomial_two_sided(900, 2000)
    assert 0.0 < p < 1e-4

    va = np.zeros(4000)
    vb = np.zeros(4000)
    vb[:1500] = 100.0
    m = mcnemar(va, vb, at=0.0)
    assert m.b + m.c == 1500
    assert m.p_two_sided == pytest.approx(0.0, abs=1e-12)
    assert m.half_width == pytest.approx(1.96 * (0.375 / 4000) ** 0.5)


def test_an_even_split_names_no_winner(capsys: pytest.CaptureFixture[str]) -> None:
    """b == c > 0 is discordance without direction; a winner would be invented."""
    va = np.array([0.0, 0.0, 9.0, 9.0])
    vb = np.array([0.0, 9.0, 0.0, 9.0])
    report_mcnemar(mcnemar(va, vb, at=0.0), at=0.0, label_a="A", label_b="B")
    out = capsys.readouterr().out
    assert "exactly tied" in out
    assert "favours" not in out
    assert "VOID" not in out


# --------------------------------------------------------------------------
# PR #446 independent review. Every test below closes a mutant that SURVIVED
# the first suite, and each survivor had the same shape: the production code
# was correct and the number it produced was pinned by nothing. On a gate that
# decides a launch that is the same exposure as a wrong implementation.
# --------------------------------------------------------------------------


def test_the_p_value_conditions_on_the_DISCORDANT_pairs_not_on_n() -> None:
    """⚑⚑ M19: `exact_binomial_two_sided(b, b + c)` -> `(b, n)` passed everything.

    Conditioning on the discordant pairs IS McNemar's test; the paired n is the
    wrong denominator and it manufactures significance. Measured at the
    registered n=4000: b=25/c=20 is p=0.551 (n.s.) correctly and p=0 under the
    mutant, and b=c=19 is p=1.0 correctly and p=0 under the mutant — a genuine
    null printed as `significant at 95%`.

    Nothing asserted `mcnemar(...).p_two_sided` at all before this: the
    function was tested in isolation and its ONE call site was not.
    """
    # b=6, c=1 -> 7 discordant, lower tail at 1: 2*(C(7,0)+C(7,1))/2^7 = 16/128
    va = np.r_[np.zeros(6), np.full(1, 9.0), np.zeros(50)]
    vb = np.r_[np.full(6, 9.0), np.zeros(1), np.zeros(50)]
    m = mcnemar(va, vb, at=0.0)
    assert (m.b, m.c, m.n) == (6, 1, 57)
    assert m.p_two_sided == pytest.approx(0.125)
    assert m.p_two_sided != pytest.approx(exact_binomial_two_sided(m.b, m.n))


def test_a_balanced_discordant_table_is_not_significant_at_the_registered_n() -> None:
    """The mutant's headline failure, stated as its own case."""
    va = np.r_[np.zeros(19), np.full(19, 9.0), np.zeros(3962)]
    vb = np.r_[np.full(19, 9.0), np.zeros(19), np.zeros(3962)]
    m = mcnemar(va, vb, at=0.0)
    assert (m.b, m.c, m.n) == (19, 19, 4000)
    assert m.p_two_sided == pytest.approx(1.0)


def test_discordance_counts_BOTH_directions(capsys: pytest.CaptureFixture[str]) -> None:
    """⚑ M17: `d = b / n` passed every test, because every fixture had c = 0.

    `d_obs` and the half-width derived from it are the two numbers the prereg
    must publish BEFORE choosing an effect size, so understating them is the
    same defect as the assumed d=0.30 this file was written to prevent — one
    level down. At b=19/c=19 the mutant understates the half-width by 1.41x.
    """
    va = np.r_[np.zeros(30), np.full(8, 9.0), np.zeros(3962)]
    vb = np.r_[np.full(30, 9.0), np.zeros(8), np.zeros(3962)]
    m = mcnemar(va, vb, at=0.0)
    assert (m.b, m.c) == (30, 8)
    assert m.discordance == pytest.approx(38 / 4000)
    assert m.half_width == pytest.approx(1.96 * (38 / 4000 / 4000) ** 0.5)
    report_mcnemar(m, at=0.0, label_a="A", label_b="B")
    assert "d_obs 0.0095" in capsys.readouterr().out


def test_every_printed_number_in_the_readout_is_pinned(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑⚑ FOUR MUTANTS SURVIVED 171 TESTS, AND ALL FOUR WERE PRINT-ONLY.

    An independent review of this file's own delta found that `d_obs` and the
    CI bounds were pinned and NOTHING ELSE ON THE READOUT WAS. Each of these
    passed the entire `paired_compare` surface:

      * the half-width line printing `m.discordance` instead of `m.half_width`
      * the p line printing `m.discordance` instead of `m.p_two_sided`
      * `100 * m.half_width` -> `m.half_width` (the pp scale silently dropped)
      * the 2x2 line swapping `both` and `neither`

    The half-width is the worst of the four: it is exactly the quantity the
    prereg's step 2 owes, this delta RENAMED that line, and the test it added
    for the rename asserted only the label substrings. One assertion away.

    The standard this file already states for the production code — "the number
    it produced was pinned by nothing, and on a gate that decides a launch that
    is the same exposure as a wrong implementation" — applies to the numbers a
    human reads, not only to the dataclass. So: one fixture, every field
    hand-computable, asserted as PRINTED STRINGS.

        b=30, c=8, n=4000        (the registered n, a plausible discordance)
        d_obs      = 38/4000                       = 0.0095
        p          = 2 * sum_{i<=8} C(38,i) / 2^38 = 0.000472
        diff       = (30-8)/4000                   = +0.00550  = +0.550pp
        se         = sqrt((38 - 22^2/4000)/4000^2) = 0.00154...
        CI         = 0.0055 +/- 1.96*se            = +0.00248 .. +0.00852
        half-width = 1.96*sqrt(0.0095/4000)        = 0.00302   = 0.302pp
    """
    va = np.r_[np.zeros(30), np.full(8, 9.0), np.zeros(3962)]
    vb = np.r_[np.full(30, 9.0), np.zeros(8), np.zeros(3962)]
    m = mcnemar(va, vb, at=0.0)
    report_mcnemar(m, at=0.0, label_a="arm_old", label_b="arm_new")
    out = capsys.readouterr().out

    # The 2x2 itself. `both` and `neither` are the two cells a swap would move,
    # and they are the two that say whether the predicate fired at all. This
    # fixture deliberately makes them UNEQUAL and neither of them zero-and-zero,
    # so a swap is visible: the concordant mass sits in `both`.
    assert "both 3962   A-only 30   B-only 8   neither 0" in out
    # The p-value, from its OWN field — not the discordance printed beside it.
    assert "discordant 38  (d_obs 0.0095)   exact two-sided p = 0.000472" in out
    # The rate difference, its CI, and the pp form, at the 5 dp that makes the
    # paired correction visible at the registered n.
    assert (
        "paired rate diff (A-B) +0.00550 "
        "[95% CI +0.00248 .. +0.00852]   (+0.550pp)"
    ) in out
    # The half-width and its pp conversion. 0.00302 vs 0.0095 also distinguishes
    # it from the discordance one line up, which is what the mutant printed.
    assert "null-case half-width at this n: ±0.00302 (±0.302pp)" in out


def test_the_wald_ci_keeps_its_correlation_correction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑⚑ M16, AND MY FIRST VERSION OF THIS TEST WAS VACUOUS.

    Dropping `-(b-c)**2/n` from the variance survived the whole suite. My first
    attempt to close it computed BOTH formulas inside the test and asserted
    that one was smaller than the other — true arithmetic, touching no
    production code, passing with the mutant in. That is the failure this repo
    names "a new test is vacuous until mutated", committed inside the fix for
    a mutant. So this reads the CI the tool actually PRINTS.

    The correction is what makes the interval PAIRED; without it the interval
    is too wide. ⚑ THE TABLE HERE IS CHOSEN SO THE DIFFERENCE IS VISIBLE AT THE
    PRINTED PRECISION, and that is not a detail: at the registered n=4000 with
    38 discordant, corrected and uncorrected agree to 4 decimals (0.0048 vs
    0.0048), so a test written on the realistic table CANNOT see the mutant
    however it is asserted. The correction scales with (b-c)^2/n, so it only
    bites when the table is lopsided relative to n.
    """
    b, c, n = 60, 0, 100
    va = np.zeros(n)
    vb = np.zeros(n)
    vb[:b] = 9.0
    m = mcnemar(va, vb, at=0.0)
    assert (m.b, m.c, m.n) == (b, c, n)

    diff = (b - c) / n
    se_corrected = ((b + c - (b - c) ** 2 / n) / n**2) ** 0.5
    se_uncorrected = ((b + c) / n**2) ** 0.5
    assert se_corrected < se_uncorrected          # the mutant widens it

    report_mcnemar(m, at=0.0, label_a="A", label_b="B")
    line = next(
        ln for ln in capsys.readouterr().out.splitlines() if "paired rate diff" in ln
    )
    lo, hi = (float(x) for x in re.findall(r"[-+]\d+\.\d+", line)[1:3])
    assert lo == pytest.approx(diff - 1.96 * se_corrected, abs=5e-5), line
    assert hi == pytest.approx(diff + 1.96 * se_corrected, abs=5e-5), line
    assert lo != pytest.approx(diff - 1.96 * se_uncorrected, abs=5e-5)


def test_the_winner_line_names_the_arm_with_more_matches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ M27: swapping label_a/label_b in the verdict line passed 89 tests.

    This is the sentence the ledger quotes. `b` is A-only on a lower-is-better
    predicate, so more A-only matches must read as A. The 2x2 above it was
    asserted; the conclusion drawn from it was not.
    """
    va = np.r_[np.zeros(20), np.zeros(80)]
    vb = np.r_[np.full(20, 9.0), np.zeros(80)]
    report_mcnemar(mcnemar(va, vb, at=0.0), at=0.0, label_a="ARM_OLD", label_b="ARM_NEW")
    out = capsys.readouterr().out
    assert "A-only 20   B-only 0" in out
    assert "favours ARM_OLD" in out, out
    assert "favours ARM_NEW" not in out


def test_the_exact_p_matches_scipy_across_a_random_sweep() -> None:
    """⚑ The arithmetic changed from big-int to log-space; pin it to a REFERENCE.

    The int form was exact and O(n^2) — 17.6 s at n=10000 and ~4 min at
    n=20000, i.e. indistinguishable from a hang on a large dump. The log-space
    form is O(k) and flat in n, and "exact" now means the exact binomial TEST
    computed in floats. That is a claim about accuracy, so it is checked
    against scipy rather than asserted in a comment.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(300):
        n = int(rng.integers(1, 600))
        k = int(rng.integers(0, n + 1))
        ref = scipy_stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue
        worst = max(worst, abs(exact_binomial_two_sided(k, n) - ref) / max(ref, 1e-300))
    assert worst < 1e-9, worst


def test_the_p_value_is_fast_at_a_size_that_used_to_hang() -> None:
    """n=10000 took 17.6 s under the big-int form. Pin the regime, not the clock."""
    import time

    t0 = time.monotonic()
    p = exact_binomial_two_sided(4500, 10000)
    assert 0.0 < p < 1.0
    assert time.monotonic() - t0 < 2.0


def test_a_nan_threshold_is_refused_rather_than_printed_as_VOID() -> None:
    """⚑ `value <= nan` is False everywhere, so a typo reads as the served-cache
    shape — the one output this gate exists to make unmistakable."""
    with pytest.raises(SystemExit, match="nan"):
        mcnemar(np.zeros(4), np.zeros(4), at=float("nan"))


@pytest.mark.parametrize("spelling", ["inf", "-inf", "1e999", "-1e999"])
def test_an_INFINITE_threshold_is_refused_for_the_same_reason(
    spelling: str,
) -> None:
    """⚑ THE NaN GUARD ABOVE READ AS IF IT COVERED "not a usable threshold".

    It did not, and the infinities are the likelier typo: argparse's
    `type=float` accepts `inf`, `-inf` and overflowing spellings like `1e999`,
    and either makes the predicate CONSTANT over every finite row — `+inf`
    selects all of them, `-inf` selects none. Both give `b + c == 0`, which is
    the SAME `VOID` this file's other tests treat as the served-cache
    fingerprint. So a malformed threshold typed at a live keyboard was
    indistinguishable from the defect the gate exists to catch, while the
    guarded spelling (`nan`) is the one nobody types.

    Both directions are asserted deliberately: `+inf` produces an all-`both`
    table and `-inf` an all-`neither` one, so a guard that only handled the
    empty-selection case would let half of this through.
    """
    at = float(spelling)
    assert not math.isfinite(at)
    with pytest.raises(SystemExit, match="CONSTANT"):
        mcnemar(np.zeros(4), np.zeros(4), at=at)


def test_a_boolean_field_is_refused_because_it_inverts_the_verdict(
    tmp_path: Path,
) -> None:
    """⚑⚑ `audit_targets` writes `top1_agree` EXPRESSLY for this tool to join.

    Scored as a number under the documented `--mcnemar-at 0`, `agree <= 0`
    selects DISagreement: same n, same d_obs, same p, winner swapped, and
    nothing in the output looks wrong. Measured on the #440 arms as
    `favours arm_old` vs `favours arm_new` at p = 0.000472 both ways.
    """
    from scripts.paired_compare import load_dump

    p = _dump(tmp_path / "b.jsonl", [
        {"key": "k0", "phase": 1, "cand": {"sf_soft": {"top1_agree": True}}},
        {"key": "k1", "phase": 1, "cand": {"sf_soft": {"top1_agree": False}}},
    ])
    with pytest.raises(SystemExit, match="BOOLEAN"):
        load_dump(p, join_key="key", field="cand.sf_soft.top1_agree",
                  mcnemar_at=0.0)


def test_a_boolean_field_is_ALLOWED_when_no_threshold_gate_is_in_play(
    tmp_path: Path,
) -> None:
    """⚑ THE GUARD ABOVE COST MORE THAN THE HAZARD UNTIL IT WAS SCOPED.

    Found by review: the refusal fired unconditionally inside `load_dump`,
    including with no `--mcnemar-at` — a path where no threshold predicate
    exists, so the inversion it describes cannot occur. That broke the use
    these fields were WRITTEN for: `audit_targets`' `--dump-per-position` help
    calls `top1_agree`/`out_of_top10` "the paired per-position booleans ... for
    offline slicing and PAIRED statistics", and the paired bootstrap over a
    rate is exactly that.

    ⚑ And the refusal's remedy has NO ANALOGUE for `out_of_top10`: there is no
    dumped cp field whose zero means "outside SF's top-10 set". So an
    unconditional refusal made the only tool built to difference that field
    unreachable, while telling the operator to use a substitute that does not
    exist. A guard whose escape hatch is fictional is worse than no guard —
    it looks like the question was answered.

    Python's bool is an int, so both values must survive as 1.0/0.0.
    """
    from scripts.paired_compare import load_dump

    p = _dump(tmp_path / "b.jsonl", [
        {"key": "k0", "phase": 1, "cand": {"sf_soft": {"out_of_top10": True}}},
        {"key": "k1", "phase": 1, "cand": {"sf_soft": {"out_of_top10": False}}},
    ])
    dump = load_dump(p, join_key="key", field="cand.sf_soft.out_of_top10")
    assert {k: v for k, (v, _) in dump.rows.items()} == {"k0": 1.0, "k1": 0.0}
    assert dump.unusable == 0, "a bool must not be dropped as a non-number"


def test_the_boolean_refusal_names_the_field_with_no_cp_substitute() -> None:
    """The message must not send an `out_of_top10` operator to a fiction.

    `cand.sf_soft.top1` genuinely is `top1_agree`'s cp form (top1 is the argmax
    move's regret, so `top1 == 0` IS agreement). Nothing plays that role for
    `out_of_top10`, and the first version of this message implied one did.
    """
    import inspect

    from scripts import paired_compare

    src = inspect.getsource(paired_compare.load_dump)
    assert "no such cp substitute for `out_of_top10`" in src
    assert "the bootstrap is the route, not a different --field" in src


def test_the_two_sign_conventions_are_annotated_where_they_collide(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A cp delta (lower better) and a rate diff (higher better) print adjacent,
    both labelled `(A-B)`. Neither is wrong; unannotated they are a trap."""
    va = np.r_[np.zeros(20), np.zeros(80)]
    vb = np.r_[np.full(20, 9.0), np.zeros(80)]
    report_mcnemar(mcnemar(va, vb, at=0.0), at=0.0, label_a="A", label_b="B")
    out = capsys.readouterr().out
    assert "higher = A better" in out
    assert "null-case half-width" in out
    assert "upper bound on the CI above" in out
