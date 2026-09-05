"""tb4_analyse.py -- the verdict pass. Pure arithmetic on the banked arrays."""
from __future__ import annotations

import json
import os

import numpy as np

OUT = "/home/josh/projects/chess/scratchpad/target_vs_bt4"
CAND = ["tgt", "sf", "foreign", "rand"]
RNG = np.random.default_rng(31337)


def q_to_cp(q: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=np.float64), -0.9995, 0.9995)
    return np.clip(111.714640912 * np.tan(1.5620688421 * q), -1500.0, 1500.0)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def boot_mean(a: np.ndarray, reps: int = 10000) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return (float("nan"),) * 3
    idx = RNG.integers(0, a.size, size=(reps, a.size))
    m = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def contrast(qa: np.ndarray, qb: np.ndarray, label: str) -> dict:
    """BT4's opinion of move A relative to move B, paired per row."""
    d = qa - qb
    dcp = q_to_cp(qa) - q_to_cp(qb)
    wins_b = int((qb > qa).sum())
    ties = int((qb == qa).sum())
    n = int(d.size)
    m, lo, hi = boot_mean(d)
    mc, loc, hic = boot_mean(dcp)
    return {
        "label": label, "n": n,
        "C_b_strictly_better": wins_b / n if n else None,
        "C_ci95": wilson(wins_b, n),
        "tie_rate": ties / n if n else None,
        "C_ties_half": (wins_b + 0.5 * ties) / n if n else None,
        "mean_dQ": m, "dQ_ci95": [lo, hi],
        "median_dQ": float(np.median(d)) if n else None,
        "mean_dcp": mc, "dcp_ci95": [loc, hic],
    }


def main() -> None:
    rows = np.load(os.path.join(OUT, "tb4_rows.npz"), allow_pickle=True)
    qz = np.load(os.path.join(OUT, "tb4_q_winner.npz"), allow_pickle=True)
    Q = qz["Q"]
    n = Q.shape[0]
    agree = rows["agree"][:n].astype(bool)
    listed = rows["tgt_listed"][:n].astype(bool)
    top1 = rows["top1_mass"][:n]
    nleg = rows["n_legal"][:n]
    sfcp = rows["sf_cp_regret_tgt"][:n]
    qt, qs, qf, qr = (Q[:, i] for i in range(4))

    R: dict[str, object] = {"n_rows": int(n), "frac_agree": float(agree.mean())}

    # ---------------- POSITIVE CONTROL ------------------------------------
    R["P_positive_control"] = {
        "P1_mean_Q_sfbest_minus_random": boot_mean(qs - qr)[0],
        "P1_ci95": boot_mean(qs - qr)[1:],
        "P1_mean_cp": boot_mean(q_to_cp(qs) - q_to_cp(qr))[0],
        "P2_frac_sfbest_better_than_random": float((qs > qr).mean()),
        "P2_ci95": wilson(int((qs > qr).sum()), n),
        "P2_tie_rate": float((qs == qr).mean()),
    }

    # ---------------- DECIDING MEASUREMENT --------------------------------
    dis = ~agree
    R["n_disagreement_rows"] = int(dis.sum())
    R["DECIDER_target_vs_sfbest_disagreement_rows"] = contrast(
        qt[dis], qs[dis], "Q(target argmax) - Q(sf best), rows where they differ")
    R["all_rows_target_vs_sfbest"] = contrast(qt, qs, "all rows incl. agreements")

    # ---------------- SHUFFLE CONTROL -------------------------------------
    R["S_shuffle_control"] = {
        "S2_foreign_vs_sfbest": contrast(qf[dis], qs[dis], "foreign-target argmax"),
        "S2_random_vs_sfbest": contrast(qr[dis], qs[dis], "uniform random legal"),
    }

    # ---------------- SPLITS ----------------------------------------------
    R["split_listed"] = {
        "listed": contrast(qt[dis & listed], qs[dis & listed], "argmax IS SF-listed"),
        "not_listed": contrast(qt[dis & ~listed], qs[dis & ~listed],
                               "argmax NOT SF-listed (SF cp is fabricated here)"),
    }
    bins = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)]
    R["split_top1_mass"] = [
        {"bin": f"[{lo},{hi})", "mean_top1": float(top1[m].mean()) if m.any() else None,
         **contrast(qt[m], qs[m], f"top1 in [{lo},{hi})")}
        for lo, hi in bins
        for m in [dis & (top1 >= lo) & (top1 < hi)]
    ]
    R["split_n_legal"] = [
        {"bin": lbl, **contrast(qt[m], qs[m], lbl)}
        for lbl, m in (
            ("nlegal<=15", dis & (nleg <= 15)),
            ("16..30", dis & (nleg > 15) & (nleg <= 30)),
            (">30", dis & (nleg > 30)),
        )
    ]

    # shape of the deficit: a uniform small tax (exploit-compatible) or a
    # blunder tail (not)?
    dcp_t = (q_to_cp(qt) - q_to_cp(qs))[dis]
    dcp_r = (q_to_cp(qr) - q_to_cp(qs))[dis]
    R["deficit_shape_disagreement_rows"] = {
        "target": {
            "frac_within_20cp": float((np.abs(dcp_t) <= 20).mean()),
            "frac_better_than_sf": float((dcp_t > 0).mean()),
            "frac_worse_50cp": float((dcp_t < -50).mean()),
            "frac_worse_100cp": float((dcp_t < -100).mean()),
            "frac_worse_300cp": float((dcp_t < -300).mean()),
            **{f"p{p}": float(np.percentile(dcp_t, p)) for p in (5, 25, 50, 75, 95)},
        },
        "random_legal_same_rows": {
            "frac_within_20cp": float((np.abs(dcp_r) <= 20).mean()),
            "frac_better_than_sf": float((dcp_r > 0).mean()),
            "frac_worse_100cp": float((dcp_r < -100).mean()),
            **{f"p{p}": float(np.percentile(dcp_r, p)) for p in (5, 25, 50, 75, 95)},
        },
    }

    # correspondence between the SF cp ruler and the BT4 ruler, listed rows only
    ml = dis & listed
    R["sf_vs_bt4_correspondence_listed"] = {
        "n": int(ml.sum()),
        "mean_sf_cp_regret_of_target_argmax": float(sfcp[ml].mean()),
        "mean_bt4_dcp": float((q_to_cp(qt) - q_to_cp(qs))[ml].mean()),
        "spearman_sfcp_vs_bt4dQ": float(np.corrcoef(
            np.argsort(np.argsort(sfcp[ml])),
            np.argsort(np.argsort((qt - qs)[ml])))[0, 1]),
    }

    # ---------------- POLICY RULER ----------------------------------------
    pz = np.load(os.path.join(OUT, "tb4_policy_winner.npz"), allow_pickle=True)
    prob, rank = pz["prob"][:n], pz["rank"][:n]
    top1_uci = np.array([str(x) for x in pz["top1_uci"][:n]])
    cand = np.array([[str(x) for x in r] for r in pz["cand_uci"][:n]])
    a_sf = top1_uci == cand[:, 1]
    a_tg = top1_uci == cand[:, 0]
    R["policy_ruler"] = {
        "A_sf_bt4top1_eq_sfbest": float(a_sf.mean()), "A_sf_ci95": wilson(int(a_sf.sum()), n),
        "A_tgt_bt4top1_eq_target_argmax": float(a_tg.mean()),
        "A_tgt_ci95": wilson(int(a_tg.sum()), n),
        "A_sf_minus_A_tgt": float(a_sf.mean() - a_tg.mean()),
        "mean_bt4_prob": {c: float(prob[:, i].mean()) for i, c in enumerate(CAND)},
        "mean_bt4_rank": {c: float(rank[:, i].mean()) for i, c in enumerate(CAND)},
        "median_bt4_rank": {c: float(np.median(rank[:, i])) for i, c in enumerate(CAND)},
        "disagreement_rows_only": {
            "A_sf": float(a_sf[dis].mean()), "A_tgt": float(a_tg[dis].mean()),
            "mean_bt4_prob": {c: float(prob[dis, i].mean()) for i, c in enumerate(CAND)},
            "mean_bt4_rank": {c: float(rank[dis, i].mean()) for i, c in enumerate(CAND)},
        },
    }

    # ---------------- A4 real-history arm ---------------------------------
    p = os.path.join(OUT, "tb4_realhist.npz")
    if os.path.exists(p):
        rh = np.load(p, allow_pickle=True)
        t1 = np.array([str(x) for x in rh["top1_uci"][:n]])
        R["A4_real_history"] = {
            "A_sf": float((t1 == cand[:, 1]).mean()),
            "A_tgt": float((t1 == cand[:, 0]).mean()),
            "delta_A_sf_vs_repeatfill": float((t1 == cand[:, 1]).mean() - a_sf.mean()),
            "delta_A_tgt_vs_repeatfill": float((t1 == cand[:, 0]).mean() - a_tg.mean()),
            "top1_same_move_as_repeatfill": float((t1 == top1_uci).mean()),
        }

    # ---------------- ruler-robustness: vanilla-q -------------------------
    p = os.path.join(OUT, "tb4_q_q.npz")
    if os.path.exists(p):
        Qq = np.load(p)["Q"]
        m = Qq.shape[0]
        dq = ~agree[:m]
        R["ruler_robustness_vanilla_q"] = {
            "n": int(m),
            "decider": contrast(Qq[:m, 0][dq], Qq[:m, 1][dq], "vanilla-q head"),
            "winner_head_same_subset": contrast(qt[:m][dq], qs[:m][dq], "winner head"),
        }

    with open(os.path.join(OUT, "tb4_results.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=float)
    print(json.dumps(R, indent=2, default=float))


if __name__ == "__main__":
    main()
