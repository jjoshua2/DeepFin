#!/usr/bin/env python3
"""Fit the value-target blend that best matches deep-SF ground truth.

For the audit positions (data/audit_set_v1.jsonl, deep-SF @ ~1M nodes) we have
the deep-SF expected score. The blend SOURCES — game outcome (wdl_target),
shallow-SF eval (sf_wdl), and the net's own search (search_wdl) — are stored in
the replay shards. We join by position_key (reconstructed from the planes, same
as build_audit_set), then grid-search the convex blend
``a*outcome + b*sf + c*search`` (a+b+c=1) minimizing MSE vs the deep-SF expected
score, and score the candidate blends we've been screening. GPU-free.

Usage: PYTHONPATH=. python scripts/fit_value_blend_to_deepsf.py [--replay-dir D] [--audit-set F]
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from chess_anti_engine.eval.audit import decode_board_from_planes, position_key
from chess_anti_engine.replay.shard import load_shard_arrays


def expected(wdl) -> float | None:
    a = np.asarray(wdl, dtype=np.float64).reshape(-1)
    if a.shape != (3,):
        return None
    s = float(a.sum())
    return (float(a[0]) - float(a[2])) / s if s > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", default="runs/parallel_candidate_replay_snapshots/current_live_20260602_202037")
    ap.add_argument("--audit-set", default="data/audit_set_v1.jsonl")
    args = ap.parse_args()

    deep = {json.loads(l)["key"]: json.loads(l)["wdl"] for l in open(args.audit_set)}
    print(f"audit positions: {len(deep)}")
    paths = sorted(glob.glob(os.path.join(args.replay_dir, "shard_*.zarr")),
                   key=os.path.getmtime, reverse=True)
    rows = {}  # key -> (outcome_e, sf_e, search_e, deep_e)
    scanned = 0
    for sh in paths:
        if len(rows) >= len(deep):
            break
        a, _ = load_shard_arrays(sh, lazy=False)
        if "x" not in a:
            continue
        x = np.asarray(a["x"])
        hist = str(np.asarray(a["_input_history_encoding"]).reshape(-1)[0]) if "_input_history_encoding" in a else "legacy"
        wdlt = np.asarray(a["wdl_target"]).reshape(-1)
        sfw = a.get("sf_wdl"); has_sf = np.asarray(a.get("has_sf_wdl", np.zeros(len(wdlt)))).reshape(-1)
        sw = a.get("search_wdl"); has_se = np.asarray(a.get("has_search_wdl", np.zeros(len(wdlt)))).reshape(-1)
        for i in range(x.shape[0]):
            scanned += 1
            k = position_key(decode_board_from_planes(x[i], input_history_encoding=hist))
            if k not in deep or k in rows:
                continue
            oe = {0: 1.0, 1: 0.0, 2: -1.0}.get(int(wdlt[i]))
            se = expected(sfw[i]) if (sfw is not None and bool(has_sf[i])) else None
            ce = expected(sw[i]) if (sw is not None and bool(has_se[i])) else None
            de = expected(deep[k])
            if None in (oe, se, ce, de):
                continue
            rows[k] = (oe, se, ce, de)
        print(f"  scanned {scanned} rows, matched {len(rows)}/{len(deep)}", flush=True)

    M = np.array(list(rows.values()), dtype=np.float64)
    if len(M) < 50:
        raise SystemExit(f"too few matched positions ({len(M)})")
    O, S, C, D = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
    print(f"\nfit set: {len(M)} positions (all of outcome/sf/search/deep present)")

    def mse(a, b, c):
        return float(np.mean((a * O + b * S + c * C - D) ** 2))

    # grid search the simplex (step 0.05)
    best = None
    for ai in range(21):
        a = ai / 20
        for bi in range(21 - ai):
            b = bi / 20
            c = 1.0 - a - b
            e = mse(a, b, c)
            if best is None or e < best[0]:
                best = (e, a, b, c)
    print(f"\n=== best-fit convex blend vs deep-SF ===")
    print(f"  outcome={best[1]:.2f}  sf={best[2]:.2f}  search={best[3]:.2f}   MSE={best[0]:.5f}")

    # score the candidate blends we've been using/screening (a=outcome, b=sf, c=search)
    cands = {
        "current categorical (pure outcome)": (1.0, 0.0, 0.0),
        "sf-only 0.7 (cat_sf07)":              (0.3, 0.7, 0.0),
        "search-only 0.7 (cat_search07)":      (0.3, 0.0, 0.7),
        "3-way 0.35/0.35 (cat_sfsearch)":      (0.30, 0.35, 0.35),
        "pure sf (1.0)":                       (0.0, 1.0, 0.0),
        "pure search (1.0)":                   (0.0, 0.0, 1.0),
    }
    print(f"\n=== candidate blends: MSE vs deep-SF (lower = closer to objective truth) ===")
    scored = [(mse(*w), name, w) for name, w in cands.items()]
    scored.append((best[0], "** best-fit **", (best[1], best[2], best[3])))
    for e, name, w in sorted(scored):
        print(f"  MSE={e:.5f}  (out={w[0]:.2f} sf={w[1]:.2f} search={w[2]:.2f})  {name}")
    # also report each pure source's correlation with deep-SF
    print(f"\n=== source correlation with deep-SF expected score ===")
    for name, v in (("outcome", O), ("shallow-sf", S), ("own-search", C)):
        print(f"  corr({name}, deep-SF) = {np.corrcoef(v, D)[0,1]:.3f}   MSE-alone={float(np.mean((v-D)**2)):.5f}")


if __name__ == "__main__":
    main()
