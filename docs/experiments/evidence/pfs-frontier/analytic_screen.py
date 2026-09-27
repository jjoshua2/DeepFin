#!/usr/bin/env python3
"""Zero-inference arithmetic examples for PR #755; NOT DeepFin engine tests.

Stdlib only. The width/first-round formulas are transcribed from the pinned
repository helpers named below. This file neither imports nor runs DeepFin.
Run: python analytic_screen.py --self-test --out analytic_screen.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import unittest
from fractions import Fraction
from pathlib import Path

REVISION = "c7b88eaf9da6f2c8a23675b52cac60275f726e36"
SOURCES = {
    "width": "chess_anti_engine/mcts/gumbel_c.py::_realized_candidate_width",
    "round": "chess_anti_engine/mcts/gumbel.py::halving_visits_per_action",
    "temperature": "chess_anti_engine/mcts/gumbel.py::apply_policy_temp",
    "normalization": "chess_anti_engine/mcts/gumbel.py::_completed_q_transform",
}


def first_round(budget: int, topk: int, legal: int = 40) -> dict[str, int]:
    """Ordinary multi-action root; no terminal shortcuts, carry or TT reuse."""
    if budget < 2 or topk < 2 or legal < 2:
        raise ValueError("This example requires budget, topk and legal >= 2")
    width = max(2, min(topk, legal, max(2, (budget + 1) // 2)))
    rounds, remaining = 0, width
    while remaining > 1:
        rounds += 1
        remaining = (remaining + 1) // 2
    visits = max(1, budget // (width * rounds))
    return {
        "budget": budget, "requested_topk": topk, "legal": legal,
        "width": width, "visits_per_candidate": visits,
        "scheduled_round_work": width * visits,
        "survivors": (width + 1) // 2,
        "interior_opportunities_upper_bound_per_candidate": visits - 1,
    }


def quota_hit(total: int, marked: int, opportunities: int) -> Fraction:
    """Exact hit probability for uniform marks and a FIXED opportunity set."""
    if total < 1 or not 0 <= marked <= total or not 0 <= opportunities <= total:
        raise ValueError("Invalid finite population")
    missed = math.comb(total - opportunities, marked) if marked <= total - opportunities else 0
    return 1 - Fraction(missed, math.comb(total, marked))


def ranked(logits: list[float], temperature: float) -> list[int]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return sorted(range(len(logits)), key=lambda i: (-logits[i] / temperature, i))


def normalized_scores(q: list[float], log_prior: list[float], scale: float) -> list[float]:
    """All-visited fixed snapshot; no mixed-Q, pending visits or solved states."""
    if not q or len(q) != len(log_prior):
        raise ValueError("Expected equal, nonempty vectors")
    lower, upper = min(q), max(q)
    denominator = max(upper - lower, 1e-8)
    return [p + scale * (v - lower) / denominator for p, v in zip(log_prior, q)]


def backup_mean(extra: int) -> float:
    """Nine previous +0.6 samples, then extra -0.9 samples; same root POV."""
    return (9 * 0.6 - extra * 0.9) / (9 + extra)


class ArithmeticChecks(unittest.TestCase):
    def test_first_round_examples(self) -> None:
        for budget, topk, width, visits in ((25, 32, 13, 1), (100, 32, 32, 1),
                                          (100, 16, 16, 1), (100, 8, 8, 4),
                                          (400, 32, 32, 2), (400, 16, 16, 6)):
            result = first_round(budget, topk)
            self.assertEqual((result["width"], result["visits_per_candidate"]), (width, visits))

    def test_quota_matches_exhaustive_subsets(self) -> None:
        for total in range(1, 9):
            for marked in range(total + 1):
                subsets = list(itertools.combinations(range(total), marked))
                for count in range(total + 1):
                    hits = sum(any(i < count for i in subset) for subset in subsets)
                    self.assertEqual(quota_hit(total, marked, count), Fraction(hits, len(subsets)))

    def test_temperature_preserves_strict_order(self) -> None:
        logits = [-float(i) / 7 for i in range(40)]
        for temperature in (0.5, 0.7, 1.0, 1.2, 2.0):
            self.assertEqual(ranked(logits, temperature), list(range(40)))

    def test_bad_outlier_changes_other_pair(self) -> None:
        prior = [0.0, -1.0, -20.0]
        before = normalized_scores([0.20, 0.21, 0.19], prior, 48.0)
        after = normalized_scores([0.20, 0.21, -0.90], prior, 48.0)
        self.assertGreater(before[1], before[0])
        self.assertGreater(after[0], after[1])
        # Hold the old range fixed: the A/B difference then stays unchanged.
        old_denominator = 0.21 - 0.19
        self.assertAlmostEqual((0.0 - -1.0) + 48 * (0.20 - 0.21) / old_denominator,
                               before[0] - before[1])

    def test_mean_requires_repeated_evidence(self) -> None:
        self.assertAlmostEqual(backup_mean(1), 0.45)
        self.assertGreater(backup_mean(3), 0.20)
        self.assertLess(backup_mean(4), 0.20)

    def test_first_eligible_shadows_deeper_level(self) -> None:
        eligible = [1, 2]
        first_counts = [sum(eligible[0] == level for _ in range(100)) for level in eligible]
        alternating = [sum(eligible[i % 2] == level for i in range(100)) for level in eligible]
        self.assertEqual(first_counts, [100, 0])
        self.assertEqual(alternating, [50, 50])

    def test_conditional_admission_regret_floor(self) -> None:
        scores = [100, 90, 80, 300]
        admitted = [0, 1, 2]
        floor = max(scores) - max(scores[i] for i in admitted)
        self.assertEqual(floor, 200)
        self.assertTrue(all(max(scores) - scores[i] >= floor for i in admitted))

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            first_round(1, 32)
        with self.assertRaises(ValueError):
            quota_hit(100, 101, 3)
        with self.assertRaises(ValueError):
            ranked([1.0], 0.0)
        with self.assertRaises(ValueError):
            normalized_scores([], [], 48.0)


def report(tests_run: int | None) -> dict[str, object]:
    prior = [0.0, -1.0, -20.0]
    before = normalized_scores([0.20, 0.21, 0.19], prior, 48.0)
    after = normalized_scores([0.20, 0.21, -0.90], prior, 48.0)
    return {
        "evidence_kind": "analytic_and_synthetic_only",
        "limits": "No DeepFin import/build, chess search, checkpoint, corpus, GPU, or strength test. "
                  "Opportunity sets are fixed, not counterfactual search trajectories. "
                  "First-round opportunities are upper bounds requiring fresh feedback; "
                  "terminal/carry/transposition/batching paths are not simulated.",
        "reviewed_repository_revision": REVISION,
        "source_helpers": SOURCES,
        "python": platform.python_version(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arithmetic_tests_passed": tests_run,
        "first_rounds": [first_round(b, k) for b, k in
                         ((25, 32), (100, 32), (100, 16), (100, 8), (400, 32), (400, 16))],
        "quota_examples": [{"B": 100, "R": 5, "fixed_opportunities": n,
                            "hit_probability": float(quota_hit(100, 5, n))} for n in (0, 1, 3, 10)],
        "temperature_example": {str(t): ranked([-i / 7 for i in range(40)], t)[:8]
                                for t in (0.5, 1.0, 1.2, 2.0)},
        "normalizer_example": {"q_before": [0.20, 0.21, 0.19], "q_after": [0.20, 0.21, -0.90],
                               "log_prior_up_to_constant": prior, "scale": 48.0,
                               "scores_before": before, "scores_after": after,
                               "winner_before": "B", "winner_after": "A"},
        "backup_dilution": {"old_count": 9, "old_q": 0.6, "new_q": -0.9,
                            "competitor_q": 0.2,
                            "means_after_k_bad_backups": {str(k): backup_mean(k) for k in range(1, 5)}},
        "depth_shadow_example": {"first_eligible": [100, 0], "alternating_depth": [50, 50],
                                 "assumption": "Both levels eligible on every opportunity"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    tests_run = None
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ArithmeticChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        tests_run = result.testsRun
    text = json.dumps(report(tests_run), indent=2, allow_nan=False) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        # Refuse to replace an existing bank; explicit new outputs only.
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
