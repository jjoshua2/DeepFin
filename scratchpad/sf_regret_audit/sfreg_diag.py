"""Explain the P2/P3 misses in sfreg_measure.py. CPU/numpy only."""
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
    p2_miss_d1 = 0
    p2_miss_other = 0
    p2_miss_total = 0
    p2_miss_raw_short = 0
    p3_miss = 0
    p3_miss_d1adj = 0
    p3_err: list[float] = []
    n = 0
    raw_scored_hist = np.zeros(12, dtype=np.int64)
    for nm in names:
        g = zarr.open(os.path.join(BASE, nm), mode="r")
        has = np.asarray(g["has_sf_p0_regret"][:]).astype(bool)
        haslm = np.asarray(g["has_legal_mask"][:]).astype(bool)
        sel = has & haslm
        if not sel.any():
            continue
        reg = np.asarray(g["sf_p0_regret"][:]).astype(np.float32)[sel]
        lm = np.asarray(g["legal_mask"][:]).astype(bool)[sel]
        for i in range(reg.shape[0]):
            m = lm[i]
            L = int(m.sum())
            if L <= 1:
                continue
            n += 1
            v = reg[i][m]
            d = float(v.max())
            K = L - int((v >= d - 1e-6).sum())
            if K != min(6, L):
                p2_miss_total += 1
                if d >= 0.999:
                    p2_miss_d1 += 1
                else:
                    p2_miss_other += 1
            if K > 0 and d < 0.999:
                cov = v[v < d - 1e-6]
                w = float(cov.max())
                e = abs(w - (2.0 * d - 1.0))
                if e >= 2e-3:
                    p3_miss += 1
                    p3_err.append(e)
                    # is the row consistent with d being a COVERED value
                    # (i.e. every legal move scored, no fabricated entry)?
                    if K == min(6, L):
                        p3_miss_d1adj += 1
    print(f"rows={n}")
    print(f"P2 misses total={p2_miss_total} ({p2_miss_total / n:.4f})")
    print(f"  of which d>=0.999 (cap saturated, K unrecoverable)={p2_miss_d1} "
          f"({p2_miss_d1 / max(1, p2_miss_total):.3f})")
    print(f"  other={p2_miss_other}")
    print(f"P3 misses={p3_miss}  median_err={np.median(p3_err) if p3_err else 0:.5f} "
          f"max_err={max(p3_err) if p3_err else 0:.5f}")
    print(f"  P3 misses where K==min(6,L)={p3_miss_d1adj}")
    _ = raw_scored_hist


if __name__ == "__main__":
    main()
