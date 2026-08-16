"""Out-of-fold AUC for the best label-time screen, against the pre-committed 0.65 bar.

The single-feature pass showed my two guessed features split: `sf_cp_regret_tgt` and
`n_legal` carry signal, `tgt_listed` and `top1_mass` carry none. Combining the two that
worked is legitimate against a FIXED bar, but selecting features by their measured AUC
and then scoring on the same rows is in-sample optimism, so every number here is
OUT-OF-FOLD ([[never_condition_a_control_on_its_own_outcome]]).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "target_vs_bt4"
BAD_CUT = -0.10
BAR = 0.65


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    sv = vals[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 400, lr: float = 0.1) -> np.ndarray:
    """Plain gradient-descent logistic fit. No sklearn dependency in scratch rigs."""
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        w -= lr * (Xb.T @ (p - y)) / max(y.size, 1)
    return w


def main() -> None:
    rows = np.load(SRC / "tb4_rows.npz", allow_pickle=True)
    Q = np.load(SRC / "tb4_q_winner.npz")["Q"]
    trained = ~np.load(SRC / "tb4_postckpt_mask.npy")

    keep = trained & np.isfinite(Q[:, 0]) & np.isfinite(Q[:, 1]) & ~rows["agree"].astype(bool)
    d = (Q[:, 0] - Q[:, 1])[keep]
    bad = d <= BAD_CUT
    assert bad.sum() == 153 and keep.sum() == 1007, f"population drift: {bad.sum()}/{keep.sum()}"

    n_legal = rows["n_legal"].astype(np.float64)[keep]
    sf_reg = rows["sf_cp_regret_tgt"].astype(np.float64)[keep]
    top1 = rows["top1_mass"].astype(np.float64)[keep]
    listed = rows["tgt_listed"].astype(bool)[keep]

    # standardised, with the two heavy-tailed features log-compressed
    F = np.column_stack(
        [
            np.log1p(n_legal),
            np.log1p(np.maximum(sf_reg, 0.0)),
            top1,
            (~listed).astype(np.float64),
        ]
    )
    F = (F - F.mean(axis=0)) / np.maximum(F.std(axis=0), 1e-9)
    names = ["log_n_legal", "log_sf_regret", "top1_mass", "not_listed"]

    rng = np.random.default_rng(20260816)
    fold = rng.permutation(np.arange(bad.size) % 5)
    oof = np.empty(bad.size)
    for f in range(5):
        te = fold == f
        w = fit_logistic(F[~te], bad[~te].astype(np.float64))
        oof[te] = np.hstack([np.ones((te.sum(), 1)), F[te]]) @ w

    a_full = auc(oof, bad)

    # the two-feature model (only the features that carried signal), also out-of-fold
    oof2 = np.empty(bad.size)
    F2 = F[:, :2]
    for f in range(5):
        te = fold == f
        w = fit_logistic(F2[~te], bad[~te].astype(np.float64))
        oof2[te] = np.hstack([np.ones((te.sum(), 1)), F2[te]]) @ w
    a_two = auc(oof2, bad)

    def boot(score: np.ndarray) -> list[float]:
        r = np.random.default_rng(7)
        out = [auc(score[k], bad[k]) for k in (r.integers(0, bad.size, bad.size) for _ in range(2000))]
        out = [v for v in out if np.isfinite(v)]
        return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]

    # ---- negative control: shuffle the labels. A screen that "works" on shuffled
    # labels is measuring the fitting procedure, not the data.
    sh = bad.copy()
    np.random.default_rng(99).shuffle(sh)
    oof_sh = np.empty(bad.size)
    for f in range(5):
        te = fold == f
        w = fit_logistic(F[~te], sh[~te].astype(np.float64))
        oof_sh[te] = np.hstack([np.ones((te.sum(), 1)), F[te]]) @ w
    a_shuf = auc(oof_sh, sh)

    # ---- the operating point that matters: cost vs deficit captured
    #
    # ⚑⚑ COST AXIS CORRECTED. My first pass costed this as `1 + f*6`, reading
    # [[sf_multipv_width_costs_7x]] as "width costs 7x". It says the OPPOSITE: at a
    # fixed `go nodes N` budget both settings spend N nodes, so MultiPV width costs
    # ZERO CPU ([[sf_cpu_cost_split]]: "sf_multipv is not a CPU lever at all"). The 7x
    # is the SAVING from NARROWING — MultiPV 3 reaches production's depth at ~100k
    # nodes instead of ~698k. Width is paid for in DEPTH, and depth is bought with
    # NODES. So the real escalation cost is the node ratio, not a width ratio.
    #
    # Production label budget 150k-200k (median ~175k, [[sf_label_nodes_are_not_sf_nodes]]);
    # escalation target 500k, the budget at which the unconverged-label study saw
    # SF settle. Labels are ~95% of loop cost.
    prod_nodes, escalate_nodes, label_share = 175_000.0, 500_000.0, 0.95
    node_ratio = escalate_nodes / prod_nodes

    order = np.argsort(-oof, kind="mergesort")
    deficit_all = float(-d[bad].sum())
    curve = []
    for frac in (0.10, 0.15, 0.20, 0.30, 0.50):
        k = int(round(frac * bad.size))
        sel = order[:k]
        label_mult = 1.0 + frac * (node_ratio - 1.0)
        curve.append(
            {
                "select_frac": frac,
                "bad_captured": round(float(bad[sel].sum() / bad.sum()), 4),
                "random_baseline": frac,
                "lift": round(float(bad[sel].sum() / bad.sum()) / frac, 3),
                "deficit_captured": round(float(-d[sel][bad[sel]].sum()) / deficit_all, 4),
                "label_cost_mult": round(label_mult, 3),
                "loop_cost_mult": round(1.0 + label_share * (label_mult - 1.0), 3),
            }
        )
    blanket_label = node_ratio
    curve.append(
        {
            "select_frac": 1.0,
            "bad_captured": 1.0,
            "random_baseline": 1.0,
            "lift": 1.0,
            "deficit_captured": 1.0,
            "label_cost_mult": round(blanket_label, 3),
            "loop_cost_mult": round(1.0 + label_share * (blanket_label - 1.0), 3),
            "note": "BLANKET re-label at 500k - the thing the screen has to beat",
        }
    )

    # ---- the residual that bounds every screen: rows the SHALLOW teacher rates as
    # nearly-fine (bottom quartile of its own regret) that are nonetheless BAD.
    q25 = float(np.percentile(sf_reg, 25))
    conf = sf_reg <= q25
    resid = {
        "shallow_regret_q25_cp": round(q25, 2),
        "n_bad_in_confident_quartile": int((conf & bad).sum()),
        "share_of_all_bad": round(float((conf & bad).sum() / bad.sum()), 4),
        "bad_rate_in_confident_quartile": round(float(bad[conf].mean()), 4),
        "bad_rate_elsewhere": round(float(bad[~conf].mean()), 4),
        "mean_dQ_of_those_rows": round(float(d[conf & bad].mean()), 4),
    }

    out = {
        "n_pop": int(bad.size),
        "n_bad": int(bad.sum()),
        "base_rate": round(float(bad.mean()), 4),
        "PRECOMMITTED_BAR_auc": BAR,
        "oof_auc_all4": {"auc": round(a_full, 4), "ci95": boot(oof), "features": names},
        "oof_auc_two_signal_feats": {"auc": round(a_two, 4), "ci95": boot(oof2), "features": names[:2]},
        "negative_control_shuffled_labels_auc": round(a_shuf, 4),
        "curve_oof_all4": curve,
        "residual_invisible_to_shallow_teacher": resid,
    }
    # ⚑⚑ BOTH pre-committed clauses are evaluated, not just the one that reads well.
    # The prereg defined SCREENABLE by P6 (>=50% of BAD rows in the top 15%) and
    # NOT-SCREENABLE by AUC < 0.65. Those two clauses do not tile the outcome space,
    # and this result lands in the gap they leave. Reporting only the clause that
    # passes would be exactly the one-way gate I have been rejecting in review.
    best_auc = max(a_full, a_two)
    p6 = next(c for c in curve if c["select_frac"] == 0.15)
    p6_bad = float(p6["bad_captured"])
    clause_screenable = p6_bad >= 0.50
    clause_not_screenable = best_auc < BAR
    if clause_screenable and not clause_not_screenable:
        verdict = "SCREENABLE"
    elif clause_not_screenable and not clause_screenable:
        verdict = "NOT SCREENABLE"
    else:
        verdict = "UNCLASSIFIED BY THE PREREG - the two clauses do not tile the space"
    out["clause_SCREENABLE_p6_bad_capture_at_15pct"] = {
        "value": round(p6_bad, 4),
        "threshold": 0.50,
        "passes": clause_screenable,
    }
    out["clause_NOT_SCREENABLE_auc_below_bar"] = {
        "value": round(best_auc, 4),
        "threshold": BAR,
        "passes": clause_not_screenable,
    }
    out["VERDICT"] = verdict
    out["VERDICT_note"] = (
        f"best out-of-fold AUC {best_auc:.4f} clears the {BAR} bar, but the top-15% "
        f"capture is {p6_bad:.3f} against a pre-committed 0.50. The screen is real and "
        "well above chance; it is weaker than the operating point I pre-registered."
    )

    Path(__file__).resolve().parent.joinpath("tailscreen_cv_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
