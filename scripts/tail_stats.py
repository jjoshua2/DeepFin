"""Tail readout over per-position dumps (value_regret / audit_targets).

Means on the audit set are tail-dominated (median value regret ~10cp vs mean
~75): this prints the tail view — P90, frac>100cp, frac>300cp — plus paired
>300cp flip counts between two dumps (new blowups vs fixed ones), which is the
readout that tracks the Cheese single-collapse failure mode.

⚑ WHICH QUANTITY IS READ IS A ``--field`` DECISION, AND IT IS THE WHOLE POINT.
``audit_targets.py --dump-per-position`` writes FIVE candidates per position
under ``cand`` (a=``raw``, b=``search``, c=``sf_soft``, d=``train``,
e=``train_fast``), and they answer different questions. Until 2026-08-16 this
script could address exactly one of them (``cand.raw.top1``, via
``--raw-top1``), so a readout pre-committed on any other candidate silently
scored candidate (a) instead — and (a) is read off the checkpoint with zero
dependence on ``--stockfish``, i.e. it is IDENTICAL between two arms that
differ only in the SF binary. The statistic is then identically 0 and the
verdict is INCONCLUSIVE by construction. Name the field you pre-committed on.

Usage:
  PYTHONPATH=. python3 scripts/tail_stats.py A.jsonl B.jsonl
  PYTHONPATH=. python3 scripts/tail_stats.py A.jsonl B.jsonl --raw-top1
  PYTHONPATH=. python3 scripts/tail_stats.py A.jsonl B.jsonl --field cand.sf_soft.top1

``--field`` takes a dotted path into the JSONL row. Default ``value`` (the
value_regret dump shape). ``--raw-top1`` is a documented alias for
``--field cand.raw.top1`` and is kept so existing invocations keep working.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from chess_anti_engine.eval.audit import PHASE_NAMES

DEFAULT_FIELD = "value"
RAW_TOP1_FIELD = "cand.raw.top1"


def resolve_field(row: dict, field: str) -> Any:
    """Walk a dotted path into ``row``; return None if any hop is missing.

    Deliberately dict-only: a path that runs into a list or a scalar is a
    mis-specified field, not something to index numerically.
    """
    cur: Any = row
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def load(path: str, field: str = DEFAULT_FIELD) -> dict[str, tuple[float, str]]:
    """Load ``field`` per row, keyed by fen/key, with the row's phase name.

    ⚑ Raises if a non-empty dump yields ZERO numeric values for ``field``.
    Silently returning {} is how a mis-addressed readout reports "no
    difference" instead of "I did not measure the thing you named": the paired
    flip count over an empty join is 0, which is indistinguishable from a real
    null. This is the defect this script had, so it fails loudly instead.
    """
    rows: dict[str, tuple[float, str]] = {}
    n_rows = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_rows += 1
            v = resolve_field(r, field)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            phase = r.get("phase")
            name = (
                PHASE_NAMES[phase]
                if isinstance(phase, int) and 0 <= phase < len(PHASE_NAMES)
                else str(phase)
            )
            rows[r.get("fen") or r.get("key")] = (float(v), name)
    if n_rows and not rows:
        raise SystemExit(
            f"{path}: field {field!r} resolved to a number on 0 of {n_rows} rows. "
            "Check the field path against the dump (audit_targets writes "
            "cand.<raw|search|sf_soft|train|train_fast>.<exp|top1>; value_regret "
            "writes a top-level 'value')."
        )
    return rows


def report(name: str, d: dict, phase: str | None = None) -> None:
    vs = np.array([v for v, p in d.values() if phase is None or p == phase])
    if vs.size == 0:
        # np.percentile on an empty array raises IndexError. The deciding
        # readout must not be able to die on its own input.
        print(f"{name:30s} {phase or 'all':11s} n=    0 (no rows in this phase)")
        return
    print(f"{name:30s} {phase or 'all':11s} n={len(vs):5d} mean={vs.mean():7.1f} "
          f"med={np.median(vs):6.1f} P90={np.percentile(vs, 90):7.1f} "
          f">100cp={100 * (vs > 100).mean():5.1f}% >300cp={100 * (vs > 300).mean():5.1f}%")


def resolve_field_arg(field: str | None, raw_top1: bool) -> str:
    """Reconcile ``--field`` and the ``--raw-top1`` alias into one field path."""
    if raw_top1 and field is not None and field != RAW_TOP1_FIELD:
        raise SystemExit(
            f"--raw-top1 is an alias for --field {RAW_TOP1_FIELD}; it conflicts "
            f"with --field {field}. Pass one."
        )
    if raw_top1:
        return RAW_TOP1_FIELD
    return DEFAULT_FIELD if field is None else field


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump_a")
    ap.add_argument("dump_b")
    ap.add_argument("--field", default=None,
                    help="dotted path to the scored quantity in each JSONL row, "
                         f"e.g. cand.sf_soft.top1. Default {DEFAULT_FIELD!r}.")
    ap.add_argument("--raw-top1", action="store_true",
                    help=f"alias for --field {RAW_TOP1_FIELD}")
    ap.add_argument("--tail-cp", type=float, default=300.0)
    ap.add_argument("--phase", action="append", default=None,
                    help="phase to report (repeatable). Default: all, endgame, "
                         "middlegame.")
    args = ap.parse_args()

    field = resolve_field_arg(args.field, args.raw_top1)
    print(f"[field] {field}")
    a = load(args.dump_a, field)
    b = load(args.dump_b, field)
    phases: list[str | None] = (
        [None, "endgame", "middlegame"] if args.phase is None else list(args.phase)
    )
    for name, d in ((args.dump_a, a), (args.dump_b, b)):
        for ph in phases:
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
