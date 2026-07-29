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

Rows missing from either side, and rows whose metric is null / missing /
non-finite, are dropped. The report accounts for both PER SIDE — read the
header as ``rows = unusable + indexed`` and ``indexed = paired + unmatched``,
and check it against the input files before trusting a verdict. Duplicate join
keys are refused outright — see ``load_dump``. ``phase`` (int index or string)
groups the per-phase breakdown.

Sign convention: delta = A - B per position. For regret-style metrics (lower
is better), a NEGATIVE mean delta means A is better.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import NamedTuple

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


class Dump(NamedTuple):
    """One per-position dump, indexed for the join.

    ``unusable`` counts the rows that were read but could not enter the index
    — no join key, or a null / missing / non-finite metric. It rides along
    because ``report`` cannot recover it from ``rows`` and used to print a
    ``dropped`` figure that silently excluded it.
    """

    rows: dict[str, tuple[float, str]]
    unusable: int


def load_dump(
    path: str, *, join_key: str = "fen", field: str = "value",
) -> Dump:
    """Index one per-position dump by its join key.

    **A metric must be finite, not merely numeric.** ``isinstance(v, (int,
    float))`` alone admits NaN, and one NaN row poisons the entire comparison:
    numpy's ``mean`` and ``percentile`` both propagate it, so the delta and
    both CI bounds print as ``nan``, and since ``nan < 0`` and ``nan > 0`` are
    both False the verdict falls through to "NOT significant". Demonstrated on
    a 50-row pair: a single NaN turned a clean −5.0 delta into "NOT
    significant" — silently converting a KILL into a HOLD, which is the worst
    direction for a tool every ledger verdict is read off. Non-finite rows are
    now dropped and counted like nulls.

    **Duplicate join keys are refused.** A dump is one deterministic read of
    one checkpoint over a frozen position set, so a repeated key means the file
    is not what the join assumes — two runs concatenated, a re-run appended, or
    the wrong ``--join-key``. Before this check the dict build made duplicates
    last-win and the losers were invisible: not in ``common``, not in the
    reported ``dropped`` either, so the caller read a clean join over a
    silently smaller and silently biased sample (audit invariant L14). There is
    no principled winner between two rows claiming the same position, so the
    tool stops instead of guessing. Rows dropped as unusable never enter the
    index and so cannot trip it.
    """
    rows: dict[str, tuple[float, str]] = {}
    duplicates: list[str] = []
    unusable = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = r.get(join_key)
            v = get_field(r, field)
            if k is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                unusable += 1
                continue
            key = str(k)
            if key in rows:
                duplicates.append(key)
                continue
            rows[key] = (float(v), phase_label(r.get("phase", "?")))
    if duplicates:
        unique_dupes = sorted(set(duplicates))
        raise SystemExit(
            f"{path}: {len(duplicates)} duplicate rows across "
            f"{len(unique_dupes)} repeated '{join_key}' values, e.g. "
            f"{unique_dupes[:3]}. A paired comparison cannot join an ambiguous "
            f"key — de-duplicate the dump (or pass the right --join-key) and "
            f"re-run. Refusing rather than silently dropping them."
        )
    return Dump(rows, unusable)


def report(a: Dump, b: Dump, *, label_a: str, label_b: str, n_boot: int) -> None:
    common = sorted(set(a.rows) & set(b.rows))
    if not common:
  # Report `unusable` here too, in the same per-side shape as the success
  # path. Total scorer failure on one side is the EXTREME of the defect the
  # rest of this function fixes: "A has 50, B has 0" alone is indistinguishable
  # from an empty file or a wrong --field, so the operator goes hunting for a
  # schema bug when in fact every position scored and every score was
  # null/NaN. The count is already sitting in `Dump.unusable`; withholding it
  # from the one message the operator sees is how a data failure gets read as
  # a config typo.
  # Name the side that indexed nothing rather than saying "a side": the
  # accounting printed immediately before it already labels A and B, so an
  # unlabelled "A side indexed nothing" reads as the literal side A and
  # contradicts the numbers on the same line.
        empty = [name for name, d in (("A", a), ("B", b)) if not d.rows]
        raise SystemExit(
            "no joinable rows — "
            f"A: {len(a.rows) + a.unusable} rows, {a.unusable} unusable, "
            f"{len(a.rows)} indexed"
            f"   B: {len(b.rows) + b.unusable} rows, {b.unusable} unusable, "
            f"{len(b.rows)} indexed. "
            + (
                "Both sides indexed rows but share no key — check "
                "--join-key/--field against the dump schema."
                if not empty
                else f"{' and '.join(empty)} indexed nothing: with rows read but "
                "unusable the scorer failed on them (non-finite or null "
                "--field); with no rows at all the dump is empty. Otherwise "
                "check --join-key/--field against the dump schema."
            ),
        )
    va = np.array([a.rows[k][0] for k in common])
    vb = np.array([b.rows[k][0] for k in common])
    ph = np.array([a.rows[k][1] for k in common])
    d = va - vb

    lo, hi = paired_bootstrap_ci(d, n_boot=n_boot)
    frac_a = float((d < 0).mean())
    frac_b = float((d > 0).mean())
  # Per side, and never as one summed `dropped`. The old single number was
  # computed from the two INDEXES, so a row the scorer failed on -- absent from
  # both -- was invisible: three null rows per side printed "dropped 0". Summing
  # the two categories instead would misreport the opposite way, since a row
  # unusable on B is also unmatched from A and would be counted twice for one
  # lost position. Per side, every figure is checkable against the input files
  # and `rows = unusable + indexed`, `indexed = paired + unmatched`.
    print(f"paired positions: {len(common)}")
    print(f"  A: {len(a.rows) + a.unusable} rows, {a.unusable} unusable, "
          f"{len(a.rows) - len(common)} unmatched"
          f"   B: {len(b.rows) + b.unusable} rows, {b.unusable} unusable, "
          f"{len(b.rows) - len(common)} unmatched")
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
