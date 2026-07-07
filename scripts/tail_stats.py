"""Tail readout over per-position dumps (value_regret / audit_targets).

Means on the audit set are tail-dominated (median value regret ~10cp vs mean
~75): this prints the tail view — P90, frac>100cp, frac>300cp — plus paired
>300cp flip counts between two dumps (new blowups vs fixed ones), which is the
readout that tracks the Cheese single-collapse failure mode.

Usage:
  PYTHONPATH=. python3 scratchpad/policy_ci/tail_stats.py A.jsonl B.jsonl
  PYTHONPATH=. python3 scratchpad/policy_ci/tail_stats.py A.jsonl B.jsonl --raw-top1
(--raw-top1 reads audit dumps' cand.raw.top1; default reads value dumps.)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from chess_anti_engine.eval.audit import PHASE_NAMES


def load(path: str, raw_top1: bool) -> dict[str, tuple[float, str]]:
    rows: dict[str, tuple[float, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            v = r.get("cand", {}).get("raw", {}).get("top1") if raw_top1 else r.get("value")
            if isinstance(v, (int, float)):
                rows[r.get("fen") or r.get("key")] = (float(v), PHASE_NAMES[r["phase"]])
    return rows


def report(name: str, d: dict, phase: str | None = None) -> None:
    vs = np.array([v for v, p in d.values() if phase is None or p == phase])
    print(f"{name:30s} {phase or 'all':11s} n={len(vs):5d} mean={vs.mean():7.1f} "
          f"med={np.median(vs):6.1f} P90={np.percentile(vs, 90):7.1f} "
          f">100cp={100 * (vs > 100).mean():5.1f}% >300cp={100 * (vs > 300).mean():5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump_a")
    ap.add_argument("dump_b")
    ap.add_argument("--raw-top1", action="store_true")
    ap.add_argument("--tail-cp", type=float, default=300.0)
    args = ap.parse_args()

    a = load(args.dump_a, args.raw_top1)
    b = load(args.dump_b, args.raw_top1)
    for name, d in ((args.dump_a, a), (args.dump_b, b)):
        for ph in (None, "endgame", "middlegame"):
            report(name.split("/")[-1], d, ph)
    common = set(a) & set(b)
    t = args.tail_cp
    both = sum(1 for k in common if a[k][0] > t and b[k][0] > t)
    only_b = sum(1 for k in common if b[k][0] > t >= a[k][0])
    only_a = sum(1 for k in common if a[k][0] > t >= b[k][0])
    print(f"\n>{t:.0f}cp tail, paired (n={len(common)}): both {both}, "
          f"new-in-B {only_b}, fixed-in-B {only_a}  (net {'+' if only_b >= only_a else ''}{only_b - only_a} blowups)")


if __name__ == "__main__":
    main()
