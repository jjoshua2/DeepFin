#!/usr/bin/env python3
"""Paired comparison of two per-position yardstick dumps, with bootstrap CIs.

Every kill/hold decision in docs/experiment_ledger.md compares two checkpoint
reads on the same frozen positions. Comparing the two MEANS throws away the
pairing; this tool joins the dumps position-by-position and reports the paired
mean delta with a bootstrap confidence interval — typically several times
tighter than the naive two-means comparison, and it makes the ledger's cp
thresholds statistically meaningful.

Inputs: two JSONL files from `scripts/value_regret.py --dump-per-position`
(or any dump with ``fen``/``value`` and optional ``phase`` fields). Rows are
joined on FEN; rows missing from either side or with null values are dropped
(counted in the report).

Sign convention: delta = A - B per position. For regret-style metrics (lower
is better), a NEGATIVE mean delta means A is better.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

PHASE_NAMES = ("endgame", "middlegame", "opening")


def paired_bootstrap_ci(
    deltas: np.ndarray, *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired deltas."""
    rng = np.random.default_rng(seed)
    n = deltas.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def load_dump(path: str) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("value") is None:
                continue
            out[r["fen"]] = (float(r["value"]), int(r.get("phase", -1)))
    return out


def report(a: dict, b: dict, *, label_a: str, label_b: str, n_boot: int) -> None:
    common = sorted(set(a) & set(b))
    dropped = (len(a) - len(common)) + (len(b) - len(common))
    va = np.array([a[k][0] for k in common])
    vb = np.array([b[k][0] for k in common])
    ph = np.array([a[k][1] for k in common])
    d = va - vb

    lo, hi = paired_bootstrap_ci(d, n_boot=n_boot)
    frac_a_better = float((d < 0).mean())
    print(f"paired positions: {len(common)} (dropped {dropped} unmatched/null)")
    print(f"A = {label_a}: mean {va.mean():.2f}")
    print(f"B = {label_b}: mean {vb.mean():.2f}")
    print(f"paired delta (A-B): {d.mean():+.2f}  [95% CI {lo:+.2f} .. {hi:+.2f}]")
    verdict = "A better" if hi < 0 else ("B better" if lo > 0 else "NOT significant")
    print(f"verdict at 95%: {verdict}   (A better on {frac_a_better:.1%} of positions)")
    for p in sorted(set(ph)):
        m = ph == p
        if m.sum() < 30:
            continue
        plo, phi = paired_bootstrap_ci(d[m], n_boot=n_boot)
        name = PHASE_NAMES[p] if 0 <= p < len(PHASE_NAMES) else f"phase{p}"
        print(f"  {name:11s} n={int(m.sum()):5d} delta {d[m].mean():+.2f} "
              f"[{plo:+.2f} .. {phi:+.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump_a", help="per-position JSONL for checkpoint/candidate A")
    ap.add_argument("dump_b", help="per-position JSONL for checkpoint/candidate B")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    report(
        load_dump(args.dump_a), load_dump(args.dump_b),
        label_a=args.label_a or args.dump_a, label_b=args.label_b or args.dump_b,
        n_boot=args.n_boot,
    )


if __name__ == "__main__":
    main()
