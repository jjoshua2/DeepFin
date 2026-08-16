"""tb4_supp.py -- SUPPLEMENTARY analyses. Not pre-registered deciders.

Three questions the pre-registered set does not answer:
  R1  RESOLUTION. BT4's Q saturates near +/-1 and the Q->cp map clips there, so
      on positions BT4 already considers decided both candidate moves land on
      the same clipped cp and the paired difference carries no information.
      How much of the sample is that, and what does the decider read on the
      subset where the ruler still has resolution?
  R2  Is the "C falls as the target gets more confident" trend the legal-move
      count in disguise? High top-1 mass concentrates in forced positions.
  R3  How big is |dQ| when it is non-zero -- is C=0.60 built out of material
      differences or float noise on decided positions?
"""
from __future__ import annotations

import json
import os

import numpy as np

OUT = "/home/josh/projects/chess/scratchpad/target_vs_bt4"
RNG = np.random.default_rng(9001)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def boot(a, reps=10000):
    a = np.asarray(a, float)
    if a.size == 0:
        return (float("nan"),) * 3
    m = a[RNG.integers(0, a.size, size=(reps, a.size))].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell(qt, qs):
    n = qt.size
    if n == 0:
        return {"n": 0}
    k = int((qs > qt).sum())
    m, lo, hi = boot(qt - qs)
    return {"n": n, "C": k / n, "C_ci95": wilson(k, n),
            "mean_dQ": m, "dQ_ci95": [lo, hi]}


def main() -> None:
    rows = np.load(os.path.join(OUT, "tb4_rows.npz"), allow_pickle=True)
    Q = np.load(os.path.join(OUT, "tb4_q_winner.npz"))["Q"]
    n = Q.shape[0]
    agree = rows["agree"][:n].astype(bool)
    dis = ~agree
    top1 = rows["top1_mass"][:n]
    nleg = rows["n_legal"][:n].astype(float)
    qt, qs = Q[:, 0], Q[:, 1]
    d = qt - qs
    R: dict[str, object] = {}

    # ---- R1 resolution ----------------------------------------------------
    decided = np.abs(qs) > 0.95
    R["R1_resolution"] = {
        "frac_rows_BT4_calls_decided_absQsf_gt_0.95": float(decided.mean()),
        "frac_disagreement_rows_decided": float(decided[dis].mean()),
        "decider_on_UNDECIDED_rows": cell(qt[dis & ~decided], qs[dis & ~decided]),
        "decider_on_DECIDED_rows": cell(qt[dis & decided], qs[dis & decided]),
        "abs_dQ_quantiles_disagreement": {
            f"p{p}": float(np.percentile(np.abs(d[dis]), p))
            for p in (10, 25, 50, 75, 90, 95, 99)},
        "frac_absdQ_lt_1e-3": float((np.abs(d[dis]) < 1e-3).mean()),
        "frac_absdQ_lt_1e-2": float((np.abs(d[dis]) < 1e-2).mean()),
    }
    for eps in (0.01, 0.05, 0.10):
        m = dis & (np.abs(d) >= eps)
        R["R1_resolution"][f"decider_material_absdQ_ge_{eps}"] = {
            "frac_of_disagreement_rows": float(m.sum() / max(dis.sum(), 1)),
            **cell(qt[m], qs[m]),
        }

    # ---- R2 confound: top-1 mass vs legal-move count ----------------------
    tb = [(0.0, 0.5), (0.5, 0.9), (0.9, 1.01)]
    nb = [(0, 15), (16, 30), (31, 999)]
    R["R2_top1_x_nlegal"] = [
        {"top1_bin": f"[{a},{b})", "nlegal_bin": f"{c}..{e}",
         "mean_nlegal": float(nleg[m].mean()) if m.any() else None,
         **cell(qt[m], qs[m])}
        for a, b in tb for c, e in nb
        for m in [dis & (top1 >= a) & (top1 < b) & (nleg >= c) & (nleg <= e)]
    ]
    R["R2_corr_top1_vs_nlegal"] = float(np.corrcoef(top1[dis], nleg[dis])[0, 1])

    # ---- R3 who wins the tail --------------------------------------------
    big = dis & (np.abs(d) >= 0.10)
    R["R3_material_tail"] = {
        "n": int(big.sum()),
        "frac_of_disagreement": float(big.sum() / max(dis.sum(), 1)),
        "frac_of_tail_where_TARGET_better": float((d[big] > 0).mean()),
        "mean_dQ_when_target_worse": float(d[big & (d < 0)].mean()),
        "mean_dQ_when_target_better": float(d[big & (d > 0)].mean()),
        "net_mean_dQ_over_tail": float(d[big].mean()),
        "contribution_of_tail_to_overall_mean_dQ":
            float(d[big].sum() / max(dis.sum(), 1)),
        "overall_mean_dQ": float(d[dis].mean()),
    }

    # ---- ruler-robustness arm if present ----------------------------------
    p = os.path.join(OUT, "tb4_q_q.npz")
    if os.path.exists(p):
        Qq = np.load(p)["Q"]
        m = Qq.shape[0]
        dq = dis[:m]
        R["ruler_robustness_vanilla_q"] = {
            "n": int(dq.sum()),
            "vanilla_q": cell(Qq[:m, 0][dq], Qq[:m, 1][dq]),
            "vanilla_winner_same_rows": cell(qt[:m][dq], qs[:m][dq]),
            "corr_dQ_between_heads": float(np.corrcoef(
                (Qq[:m, 0] - Qq[:m, 1])[dq], d[:m][dq])[0, 1]),
        }

    with open(os.path.join(OUT, "tb4_supp_results.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=float)
    print(json.dumps(R, indent=2, default=float))


if __name__ == "__main__":
    main()
