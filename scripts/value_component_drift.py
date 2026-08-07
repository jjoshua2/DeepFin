#!/usr/bin/env python3
"""Track search-value vs SF-label divergence across shard history, by phase.

The wdl target is a blend (docs/model_heads.md): sf_wdl_frac x SF cp-logistic
+ search_wdl_frac x own search-root WDL (+ game_frac x outcome). The SF term is
the stationary anchor; the search term is self-referential — the net's own
search values feed the net's own value target. A bad feedback loop would show
up FIRST as the stored ``search_wdl`` drifting away from the stored ``sf_wdl``
on the same rows, before any frozen-ruler milestone can see it. This script is
that leading indicator (first built 2026-08-07; the day-one baseline was flat:
E|d|~0.11, signed +0.04, corr ~0.96 over 150 iters).

Read the output as TREND, not level. A constant gap is structure (search plays
positions out; SF scores them statically — search runs ~4 win% optimistic).
The signature to act on is the gap RISING over shard-time, fastest wherever
SF's label is least trustworthy (middlegame / closed positions) — which is
also exactly the anti-SF territory the project wants, so the response to a
rise is a ledger entry and a look, never an automatic knob.

Phase is BOARD phase by piece count from input planes 0-11 (validated: sums in
[2, 32], corr vs ply_index ~ -0.8), NOT plies — conflating those is the N1
mistake (docs/experiment_ledger.md). Buckets: opening >= 25 pieces,
middlegame 13-24, endgame <= 12.

Usage (defaults target the live trial's compacted shards):

    PYTHONPATH=. python3 scripts/value_component_drift.py
    PYTHONPATH=. python3 scripts/value_component_drift.py \
        --shard-glob 'runs/pbt2_small/server/trials/<trial>/*/_compacted/*.zarr' \
        --time-buckets 6 --rows-per-bucket 6000 --csv-out data/value_drift.csv

``--csv-out`` appends one row per (bucket x phase) stamped with the newest
shard's mtime, so repeated runs accumulate a longitudinal record.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

_DEFAULT_GLOB = "runs/pbt2_small/server/trials/*/*/_compacted/*.zarr"
_PHASES = (("opening", 25, 33), ("middlegame", 13, 25), ("endgame", 0, 13))


@dataclass
class _Cell:
    n: int
    abs_mean: float
    signed_mean: float
    p90: float
    corr: float


def _expectation(wdl: np.ndarray) -> np.ndarray:
    """E[score] in [-1, 1] from a (n, 3) W/D/L probability array."""
    return wdl[:, 0] - wdl[:, 2]


def _collect_bucket(paths: list[str], rows_cap: int) -> dict[str, np.ndarray]:
    e_search, e_sf, pieces = [], [], []
    total = 0
    for p in paths:
        z = zarr.open(p, mode="r")
        if not isinstance(z, zarr.Group):
            continue
        keys = set(z.array_keys())
        if not {"search_wdl", "sf_wdl", "has_search_wdl", "has_sf_wdl", "x"} <= keys:
            continue
        sel = np.asarray(z["has_search_wdl"]).astype(bool) & np.asarray(z["has_sf_wdl"]).astype(bool)
        if not sel.any():
            continue
        idx = np.where(sel)[0].tolist()
        e_search.append(_expectation(np.asarray(z["search_wdl"][idx]).astype(np.float64)))
        e_sf.append(_expectation(np.asarray(z["sf_wdl"][idx]).astype(np.float64)))
        # Board phase from the current-position piece planes (0-11). Validated
        # against ply_index; see module docstring.
        pieces.append(np.asarray(z["x"][idx])[:, :12].sum(axis=(1, 2, 3)))
        total += len(idx)
        if total >= rows_cap:
            break
    if not e_search:
        return {}
    return {
        "e_search": np.concatenate(e_search),
        "e_sf": np.concatenate(e_sf),
        "pieces": np.concatenate(pieces),
    }


def _cell(e_search: np.ndarray, e_sf: np.ndarray) -> _Cell:
    d = e_search - e_sf
    corr = float(np.corrcoef(e_search, e_sf)[0, 1]) if len(d) >= 3 else float("nan")
    return _Cell(
        n=len(d),
        abs_mean=float(np.abs(d).mean()),
        signed_mean=float(d.mean()),
        p90=float(np.percentile(np.abs(d), 90)),
        corr=corr,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Track search-value vs SF-label divergence across shard history, by phase.")
    ap.add_argument("--shard-glob", default=_DEFAULT_GLOB,
                    help="glob for shard .zarr dirs; sorted by mtime for bucketing")
    ap.add_argument("--time-buckets", type=int, default=3,
                    help="number of equal shard-count buckets across history")
    ap.add_argument("--rows-per-bucket", type=int, default=6000,
                    help="stop reading a bucket once this many joined rows are in")
    ap.add_argument("--shards-per-bucket", type=int, default=8,
                    help="max shards sampled from the start of each bucket")
    ap.add_argument("--csv-out", type=Path, default=None,
                    help="append per-cell rows here for longitudinal tracking")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shard_glob), key=os.path.getmtime)
    if not paths:
        print(f"no shards match {args.shard_glob}", file=sys.stderr)
        return 1
    print(f"{len(paths)} shards; {args.time_buckets} buckets; "
          f"sampling <= {args.shards_per_bucket} shards / {args.rows_per_bucket} rows each")

    newest_mtime = datetime.fromtimestamp(os.path.getmtime(paths[-1]), tz=timezone.utc)
    csv_rows: list[dict[str, object]] = []
    header = f"{'bucket':>8} {'phase':<11} {'n':>6} {'E|d|':>7} {'signed':>8} {'p90':>7} {'corr':>7}"
    print(header)
    for b in range(args.time_buckets):
        lo = b * len(paths) // args.time_buckets
        hi = (b + 1) * len(paths) // args.time_buckets
        bucket = _collect_bucket(paths[lo:hi][: args.shards_per_bucket], args.rows_per_bucket)
        if not bucket:
            print(f"{b:>8} (no joinable rows)")
            continue
        for phase, pmin, pmax in _PHASES:
            m = (bucket["pieces"] >= pmin) & (bucket["pieces"] < pmax)
            if m.sum() < 30:
                continue
            c = _cell(bucket["e_search"][m], bucket["e_sf"][m])
            print(f"{b:>8} {phase:<11} {c.n:>6} {c.abs_mean:>7.4f} {c.signed_mean:>+8.4f} "
                  f"{c.p90:>7.4f} {c.corr:>7.4f}")
            csv_rows.append({
                "run_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                "newest_shard_mtime": newest_mtime.isoformat(timespec="seconds"),
                "bucket": b, "phase": phase, "n": c.n,
                "abs_mean": round(c.abs_mean, 5), "signed_mean": round(c.signed_mean, 5),
                "p90": round(c.p90, 5), "corr": round(c.corr, 5),
            })

    if args.csv_out and csv_rows:
        exists = args.csv_out.exists()
        with args.csv_out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            if not exists:
                w.writeheader()
            w.writerows(csv_rows)
        print(f"appended {len(csv_rows)} rows to {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
