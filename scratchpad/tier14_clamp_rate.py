"""Tier-14 step 1: how often does q_clamped = clamp(best_q, +/-(1-d_raw)) BIND?

Clamp active <=> min(w_search, l_search) ~= 0 while the row is non-degenerate
(w = 0.5*(rem+q), l = 0.5*(rem-q); q hitting +/-rem zeroes one side).
Split by d_raw quartile (d_raw = search_wdl[:,1], the net's raw root draw).
CPU-only, newest N shards of the live arm.
"""
from __future__ import annotations
import sys, glob, os
import numpy as np
import zarr

shard_dir, n_shards = sys.argv[1], int(sys.argv[2])
paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.zarr")), key=os.path.getmtime)[-n_shards:]
sw_all, ok_all = [], []
for p in paths:
    g = zarr.open(p, mode="r")
    if "search_wdl" not in g or "has_search_wdl" not in g:
        continue
    sw = np.asarray(g["search_wdl"], dtype=np.float32)
    has = np.asarray(g["has_search_wdl"]).astype(bool)
    sw_all.append(sw); ok_all.append(has)
sw = np.concatenate(sw_all); ok = np.concatenate(ok_all)
w, d, l = sw[:, 0], sw[:, 1], sw[:, 2]
# valid: a real distribution, not the isfinite fallback (0,1,0)
valid = ok & (np.abs(w + d + l - 1.0) < 1e-3) & (d < 0.9999)
w, d, l = w[valid], d[valid], l[valid]
n = len(w)
TOL = 1e-6
active = np.minimum(w, l) <= TOL
print(f"shards={len(paths)} rows_valid={n} (of {len(sw)})")
print(f"OVERALL clamp-active: {active.mean():.4f} ({active.sum()}/{n})")
qs = np.quantile(d, [0.25, 0.5, 0.75])
print(f"d_raw quartile edges: {qs.round(4).tolist()}, d_raw mean {d.mean():.4f}")
lo = 0.0
for i, hi in enumerate([*qs, 1.01]):
    m = (d >= lo) & (d < hi)
    print(f"  Q{i+1} d_raw[{lo:.3f},{hi if hi<=1 else 1:.3f}): n={m.sum():6d} clamp-active={active[m].mean():.4f}  mean|q|={np.abs(w[m]-l[m]).mean():.4f}")
    lo = hi
# how big is the clipped mass when active? |q_raw| unknown, but rem tells the ceiling
rem = 1.0 - d[active]
print(f"when active: mean rem (= |q| ceiling) {rem.mean():.4f}, median {np.median(rem):.4f}")
# near-active (within 5% of the bound) as a softer read
near = (np.minimum(w, l) / np.maximum(1e-9, 1.0 - d)) <= 0.05
print(f"near-bound (min(w,l) <= 5% of rem): {near.mean():.4f}")
