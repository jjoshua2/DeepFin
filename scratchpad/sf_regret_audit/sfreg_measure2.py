"""Corrected measurement. The P2/P3 misses are an ESTIMATOR degeneracy, not data.

Hypothesis being tested here (registered before the run):
  H1  every P2 miss with d < 0.999 has L <= 6, i.e. SF's MultiPV 6 covered ALL
      legal moves, no entry was fabricated, and `d = max(v)` is then a REAL
      covered regret rather than the default -> my K estimator cannot see a
      default that does not exist.  Expect 1429/1429.
  H2  restricted to UNAMBIGUOUS rows (L > 6 and d < 0.999) K == 6 exactly on
      100% of rows, confirming MultiPV 6 empirically from the shard content.
  H3  on those rows, worst_covered == 2d-1 within float16 tolerance on >99%.
"""
from __future__ import annotations

import os

import numpy as np
import zarr

BASE = (
    "/home/josh/projects/chess/runs/pbt2_small/replay/"
    "train_trial_5ce02_00000_0_lr=0.0000_2026-08-11_04-19-24/replay_shards"
)


def main() -> None:
    names = sorted(os.listdir(BASE))[-40:]
    h1_hit = h1_tot = 0
    rows_all = 0
    rows_le6 = 0
    rows_sat = 0
    K6 = 0
    unamb = 0
    p3_ok = 0
    fab_frac: list[float] = []
    share_pol: list[float] = []
    share_soft: list[float] = []
    d_list: list[float] = []
    worst: list[float] = []
    for nm in names:
        g = zarr.open(os.path.join(BASE, nm), mode="r")
        sel = (np.asarray(g["has_sf_p0_regret"][:]).astype(bool)
               & np.asarray(g["has_legal_mask"][:]).astype(bool))
        if not sel.any():
            continue
        reg = np.asarray(g["sf_p0_regret"][:]).astype(np.float32)[sel]
        lm = np.asarray(g["legal_mask"][:]).astype(bool)[sel]
        pol = np.asarray(g["policy_target"][:]).astype(np.float32)[sel]
        has_soft = np.asarray(g["has_policy_soft"][:]).astype(bool)[sel]
        soft = np.asarray(g["policy_soft_target"][:]).astype(np.float32)[sel]
        for i in range(reg.shape[0]):
            m = lm[i]
            L = int(m.sum())
            if L <= 1:
                continue
            rows_all += 1
            v = reg[i][m]
            d = float(v.max())
            K = L - int((v >= d - 1e-6).sum())
            if K != min(6, L) and d < 0.999:
                h1_tot += 1
                if L <= 6:
                    h1_hit += 1
            if L <= 6:
                rows_le6 += 1
                continue
            if d >= 0.999:
                rows_sat += 1
                continue
            unamb += 1
            if K == 6:
                K6 += 1
            cov = v[v < d - 1e-6]
            if abs(float(cov.max()) - (2.0 * d - 1.0)) < 2e-3:
                p3_ok += 1
            d_list.append(d)
            worst.append(float(cov.max()))
            fab = v >= d - 1e-6
            fab_frac.append(float(fab.mean()))
            for src, out in ((pol[i], share_pol),
                             (soft[i] if has_soft[i] else None, share_soft)):
                if src is None:
                    continue
                p = src[m]
                s = p.sum()
                if s <= 0:
                    continue
                p = p / s
                den = float((p * v).sum())
                if den > 0:
                    out.append(float((p * v * fab).sum()) / den)
    a = np.asarray
    print(f"[H1] P2 misses with d<0.999 that have L<=6: {h1_hit}/{h1_tot}")
    print(f"rows_all={rows_all} L<=6 (fully covered, NO fabrication)={rows_le6} "
          f"({rows_le6 / rows_all:.4f})  d saturated 1.0={rows_sat} "
          f"({rows_sat / rows_all:.4f})  unambiguous={unamb} "
          f"({unamb / rows_all:.4f})")
    print(f"[H2] K==6 on unambiguous rows: {K6}/{unamb} = {K6 / unamb:.5f}")
    print(f"[H3] worst_covered == 2d-1: {p3_ok}/{unamb} = {p3_ok / unamb:.5f}")
    D, W, F = a(d_list), a(worst), a(fab_frac)
    print(f"default d: mean={D.mean():.4f} ({D.mean() * 1000:.0f}cp)  "
          f"worst covered: mean={W.mean():.4f} ({W.mean() * 1000:.0f}cp)  "
          f"inflation d/worst={D.mean() / W.mean():.2f}x")
    print(f"fabricated fraction of LEGAL moves: mean={F.mean():.4f}")
    sp, ss = a(share_pol), a(share_soft)
    print(f"E_p[regret] fabricated mass share, p=policy_target: "
          f"mean={sp.mean():.4f} median={np.median(sp):.4f} n={sp.size}")
    if ss.size:
        print(f"E_p[regret] fabricated mass share, p=policy_soft_target: "
              f"mean={ss.mean():.4f} median={np.median(ss):.4f} n={ss.size}")


if __name__ == "__main__":
    main()
