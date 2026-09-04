"""Structural audit of the FROZEN era-probe sets' sf_p0_regret vectors.

Question: how much of each set's regret vector is the fabricated
`(worst_covered + 1)/2` default, as opposed to a real Stockfish evaluation?

The signature of a fabricated block: >= 2 legal moves sharing one exact value d,
with d == (worst_real + 1)/2 where worst_real is the largest legal value < d.
The signature of cap saturation (a REAL evaluation clipped at
SF_OWN_REGRET_CAP_CP = 1000cp): several legal moves at exactly 1.0.
The two are distinguishable except when worst_real itself hits the cap.
"""
from __future__ import annotations

import numpy as np

SETS = {
    "era": "/home/josh/projects/chess/data/era_probe/era_20260804.npz",
    "inwindow": "/home/josh/projects/chess/data/era_probe/inwindow_20260804.npz",
}


def main() -> None:
    for label, path in SETS.items():
        z = np.load(path)
        print(f"== {label}  keys={sorted(z.keys())}")
        reg = np.asarray(z["sf_p0_regret"]).astype(np.float64)
        lm = np.asarray(z["legal_mask"]).astype(bool)
        has = np.asarray(z["has_sf_p0_regret"]).astype(bool)
        n = int(has.sum())
        L = lm.sum(-1)
        print(f"   rows={reg.shape[0]} with-regret={n} width={reg.shape[1]}")
        print(f"   legal moves: mean={L.mean():.2f} max={L.max()} "
              f"frac(L>40)={float((L > 40).mean()):.4f} frac(L>6)={float((L > 6).mean()):.4f}")
        n_at_cap = np.where(lm, reg >= 1.0 - 1e-9, False).sum(-1)
        print(f"   legal moves at exactly 1.0 (cap): mean={n_at_cap.mean():.2f} "
              f"frac of legal={float(n_at_cap.sum() / L.sum()):.4f}")
        n_zero = np.where(lm, reg <= 1e-9, False).sum(-1)
        print(f"   legal moves at exactly 0.0 (SF best): mean={n_zero.mean():.3f}")

        rl = np.where(lm, reg, -1.0)
        dmax = rl.max(-1)
        n_at_max = (np.isclose(rl, dmax[:, None], atol=1e-9) & lm).sum(-1)
        below = lm & (reg < dmax[:, None] - 1e-6)
        s = np.where(below, reg, -1.0).max(-1)
        has_s = below.any(-1)
        ident = has_s & (np.abs(dmax - (s + 1.0) / 2.0) <= 1e-4)
        print(f"   ties at the max value: mean={n_at_max.mean():.2f} "
              f"frac(>=2 tied)={float((n_at_max >= 2).mean()):.4f}")
        print(f"   rows matching the (worst+1)/2 default identity: "
              f"{float(ident.mean()):.4f}  (of these, mean tied={n_at_max[ident].mean() if ident.any() else float('nan'):.2f})")
        print(f"   rows with dmax == 1.0 exactly: {float((dmax >= 1.0 - 1e-9).mean()):.4f}")
        # distinct-value count over legal moves: MultiPV-K data with all moves
        # covered has many distinct values; a fabricated block collapses them.
        nd = np.array([len(np.unique(np.round(reg[i][lm[i]], 6))) for i in range(reg.shape[0])])
        print(f"   distinct legal regret values per row: mean={nd.mean():.2f} "
              f"median={np.median(nd):.1f} frac(<=7)={float((nd <= 7).mean()):.4f}")


if __name__ == "__main__":
    main()
