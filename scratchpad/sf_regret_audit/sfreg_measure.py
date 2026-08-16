"""Measure the sf_p0_regret fabrication on LIVE replay shards. CPU / numpy only.

PREDICTIONS registered BEFORE the first run (mismatch owes an explanation):

P1  eligible-row fraction (has_sf_p0_regret==1) ~= 0.22, matching the live
    progress.csv column has_sf_p0_regret_frac (mean 0.2204 over 218 iters).
P2  covered count K := L - count(v == max_legal(v)) equals min(6, L) on >=95%
    of eligible rows.  sf_multipv: 6 in configs/pbt2_small.yaml.
P3  for rows with K < L and d := max_legal(v) < 1.0, the worst COVERED regret
    max(v[v<d]) == 2d - 1 exactly (float16 tol 2e-3), because the builder sets
    the uncovered default to (worst + 1) / 2.
P4  mean d ~= 0.57 (the banked "~570cp invented"), cap = 1000cp.
P5  fabricated legal fraction (L-K)/L: banked claim says 0.68.  With L ~ 30 and
    K = 6 the arithmetic gives 0.80, so I expect ~0.80 and the banked 0.68
    to be a row population difference.  Flagged in advance.
"""
from __future__ import annotations

import sys

import numpy as np
import zarr

BASE = (
    "/home/josh/projects/chess/runs/pbt2_small/replay/"
    "train_trial_5ce02_00000_0_lr=0.0000_2026-08-11_04-19-24/replay_shards"
)


def main(n_shards: int = 40) -> None:
    import os

    names = sorted(os.listdir(BASE))[-n_shards:]
    tot_rows = 0
    tot_elig = 0
    Ls: list[int] = []
    Ks: list[int] = []
    ds: list[float] = []
    worst_cov: list[float] = []
    p3_ok = 0
    p3_n = 0
    fab_share_pol: list[float] = []
    fab_share_uni: list[float] = []
    true_mean_cov: list[float] = []
    for nm in names:
        g = zarr.open(os.path.join(BASE, nm), mode="r")
        has = np.asarray(g["has_sf_p0_regret"][:]).astype(bool)
        haslm = np.asarray(g["has_legal_mask"][:]).astype(bool)
        tot_rows += has.size
        tot_elig += int(has.sum())
        sel = has & haslm
        if not sel.any():
            continue
        reg = np.asarray(g["sf_p0_regret"][:]).astype(np.float32)[sel]
        lm = np.asarray(g["legal_mask"][:]).astype(bool)[sel]
        pol = np.asarray(g["policy_target"][:]).astype(np.float32)[sel]
        for i in range(reg.shape[0]):
            m = lm[i]
            L = int(m.sum())
            if L <= 1:
                continue
            v = reg[i][m]
            d = float(v.max())
            n_at_max = int((v >= d - 1e-6).sum())
            K = L - n_at_max
            Ls.append(L)
            Ks.append(K)
            ds.append(d)
            if K > 0 and d < 0.999:
                cov = v[v < d - 1e-6]
                w = float(cov.max())
                worst_cov.append(w)
                p3_n += 1
                if abs(w - (2.0 * d - 1.0)) < 2e-3:
                    p3_ok += 1
                true_mean_cov.append(float(cov.mean()))
            # mass share of E_p[regret] carried by fabricated entries
            fab = v >= d - 1e-6
            p = pol[i][m]
            s = p.sum()
            if s > 0:
                p = p / s
                num = float((p * v * fab).sum())
                den = float((p * v).sum())
                if den > 0:
                    fab_share_pol.append(num / den)
            u = np.full(L, 1.0 / L, dtype=np.float32)
            den_u = float((u * v).sum())
            if den_u > 0:
                fab_share_uni.append(float((u * v * fab).sum()) / den_u)

    a = np.asarray
    print(f"shards={len(names)} rows={tot_rows} eligible={tot_elig} "
          f"frac={tot_elig / max(1, tot_rows):.4f}   [P1: expect ~0.22]")
    L, K, D = a(Ls), a(Ks), a(ds)
    print(f"n_measured_rows={L.size}")
    print(f"legal L: mean={L.mean():.2f} median={np.median(L):.0f} "
          f"p10={np.percentile(L, 10):.0f} p90={np.percentile(L, 90):.0f}")
    print(f"covered K: mean={K.mean():.3f} "
          f"hist={np.bincount(np.clip(K, 0, 10)).tolist()}")
    exp = np.minimum(6, L)
    print(f"[P2] K == min(6,L): {float((K == exp).mean()):.4f}   "
          f"K>6: {float((K > 6).mean()):.5f}")
    print(f"[P3] worst_covered == 2d-1: {p3_ok}/{p3_n} = "
          f"{p3_ok / max(1, p3_n):.4f}")
    print(f"[P4] default d: mean={D.mean():.4f} -> {D.mean() * 1000:.0f}cp  "
          f"median={np.median(D):.4f} p10={np.percentile(D, 10):.3f} "
          f"p90={np.percentile(D, 90):.3f}")
    wc = a(worst_cov)
    tc = a(true_mean_cov)
    print(f"worst COVERED regret: mean={wc.mean():.4f} -> {wc.mean() * 1000:.0f}cp")
    print(f"mean COVERED regret:  mean={tc.mean():.4f} -> {tc.mean() * 1000:.0f}cp"
          f"   ratio d/mean_cov={D.mean() / max(1e-9, tc.mean()):.2f}x")
    fab_frac = (L - K) / L
    print(f"[P5] fabricated legal fraction: mean={fab_frac.mean():.4f}")
    fp, fu = a(fab_share_pol), a(fab_share_uni)
    print(f"E_p[regret] mass share FABRICATED, p=policy_target(MCTS): "
          f"mean={fp.mean():.4f} median={np.median(fp):.4f} n={fp.size}")
    print(f"E_p[regret] mass share FABRICATED, p=uniform-over-legal: "
          f"mean={fu.mean():.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
