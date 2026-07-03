#!/usr/bin/env python3
"""Paired comparison of two per-position yardstick dumps, with bootstrap CIs.

Every kill/hold decision in docs/experiment_ledger.md compares two checkpoint
reads on the same frozen positions. Comparing the two MEANS throws away the
pairing; this tool joins the dumps position-by-position and reports the paired
mean delta with a bootstrap confidence interval — typically several times
tighter than the naive two-means comparison, and it makes the ledger's cp
thresholds statistically meaningful.

Inputs: two JSONL per-position dumps. Supported sources:

  scripts/value_regret.py --dump-per-position   (defaults: join on ``fen``,
    compare the ``value`` field)
  scripts/audit_targets.py --dump-per-position  (join on ``key``; pick the
    metric with a dotted --field path), e.g.:
      --join-key key --field cand.search.exp   # net+search E[regret]
      --join-key key --field cand.raw.top1     # raw net top-1 regret
    (the raw net candidate is deterministic per checkpoint; the search
    candidate re-runs Gumbel search each audit, so its paired delta still
    carries search-seed noise on top of position pairing)

Rows missing from either side or with null values are dropped (counted in the
report). ``phase`` (int index or string) groups the per-phase breakdown.

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


def get_field(rec: dict, path: str) -> object | None:
    """Resolve a dotted path (``cand.search.exp``) inside a dump record."""
    cur: object = rec
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def phase_label(p: object) -> str:
    if isinstance(p, int) and not isinstance(p, bool) and 0 <= p < len(PHASE_NAMES):
        return PHASE_NAMES[p]
    return str(p)


def load_dump(
    path: str, *, join_key: str = "fen", field: str = "value",
) -> dict[str, tuple[float, str]]:
    out: dict[str, tuple[float, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = r.get(join_key)
            v = get_field(r, field)
            if k is None or not isinstance(v, (int, float)):
                continue
            out[str(k)] = (float(v), phase_label(r.get("phase", "?")))
    return out


def report(a: dict, b: dict, *, label_a: str, label_b: str, n_boot: int) -> None:
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit(
            f"no joinable rows (A has {len(a)}, B has {len(b)}) — "
            "check --join-key/--field against the dump schema",
        )
    dropped = (len(a) - len(common)) + (len(b) - len(common))
    va = np.array([a[k][0] for k in common])
    vb = np.array([b[k][0] for k in common])
    ph = np.array([a[k][1] for k in common])
    d = va - vb

    lo, hi = paired_bootstrap_ci(d, n_boot=n_boot)
    frac_a = float((d < 0).mean())
    frac_b = float((d > 0).mean())
    print(f"paired positions: {len(common)} (dropped {dropped} unmatched/null)")
    print(f"A = {label_a}: mean {va.mean():.2f}")
    print(f"B = {label_b}: mean {vb.mean():.2f}")
    print(f"paired delta (A-B): {d.mean():+.2f}  [95% CI {lo:+.2f} .. {hi:+.2f}]")
    verdict = "A better" if hi < 0 else ("B better" if lo > 0 else "NOT significant")
    print(f"verdict at 95%: {verdict}   "
          f"(A better {frac_a:.1%} / B better {frac_b:.1%} / tied {1 - frac_a - frac_b:.1%})")
    for name in sorted(set(ph)):
        m = ph == name
        if m.sum() < 30:
            continue
        plo, phi = paired_bootstrap_ci(d[m], n_boot=n_boot)
        print(f"  {name:11s} n={int(m.sum()):5d} delta {d[m].mean():+.2f} "
              f"[{plo:+.2f} .. {phi:+.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dump_a", help="per-position JSONL for checkpoint/candidate A")
    ap.add_argument("dump_b", help="per-position JSONL for checkpoint/candidate B")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--join-key", default="fen",
                    help="record field to join on (audit_targets dumps: 'key')")
    ap.add_argument("--field", default="value",
                    help="dotted path to the compared metric "
                         "(audit_targets dumps: e.g. 'cand.search.exp')")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    report(
        load_dump(args.dump_a, join_key=args.join_key, field=args.field),
        load_dump(args.dump_b, join_key=args.join_key, field=args.field),
        label_a=args.label_a or args.dump_a, label_b=args.label_b or args.dump_b,
        n_boot=args.n_boot,
    )


if __name__ == "__main__":
    main()
