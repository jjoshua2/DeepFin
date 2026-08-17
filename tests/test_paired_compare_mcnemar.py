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
