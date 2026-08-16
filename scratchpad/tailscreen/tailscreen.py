"""Can the bad-tail policy-target rows be SCREENED at label time?

Reads only banked artifacts from scratchpad/target_vs_bt4/. No SF, no GPU, no net.

The bad tail is `dQ = Q_target - Q_sf <= -0.10` measured against a deep full-width Q.
The question is whether membership is predictable from quantities ALREADY STORED on
every replay row, which would make a targeted full-width re-label affordable.

Predictions are pre-registered in PREDICTIONS_tailscreen.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "target_vs_bt4"
BAD_CUT = -0.10
GOOD_CUT = 0.10


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC of `score` for predicting `label`, ties at 0.5."""
    pos = score[label]
    neg = score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    # average ranks over ties so a constant feature scores exactly 0.5
    vals = np.concatenate([pos, neg])
    sv = vals[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def boot_ci(score: np.ndarray, label: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(20260816)
    n = score.size
    out = np.empty(n_boot)
    for b in range(n_boot):
        k = rng.integers(0, n, n)
        out[b] = auc(score[k], label[k])
    out = out[np.isfinite(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    rows = np.load(SRC / "tb4_rows.npz", allow_pickle=True)
    # ⚑ tb4_phase2.py reads tb4_q_WINNER.npz, not tb4_q_q.npz, and restricts to the
    # rows the checkpoint had actually trained on (~postckpt_mask). Using the wrong
    # Q file and skipping the mask gave 176/82/900 against the ledger's 153/66/788 —
    # caught only because the counts were pre-registered.
    qq = np.load(SRC / "tb4_q_winner.npz", allow_pickle=True)
    post = np.load(SRC / "tb4_postckpt_mask.npy")
    trained = ~post

    Q = qq["Q"]
    term = qq["terminal_ref"] if "terminal_ref" in qq.files else np.zeros_like(Q, dtype=np.int64)

    valid = trained & np.isfinite(Q[:, 0]) & np.isfinite(Q[:, 1])
    dQ = Q[:, 0] - Q[:, 1]

    n_legal = rows["n_legal"].astype(np.float64)
    tgt_listed = rows["tgt_listed"].astype(bool)
    top1_mass = rows["top1_mass"].astype(np.float64)
    sf_reg = rows["sf_cp_regret_tgt"].astype(np.float64)
    agree = rows["agree"].astype(bool)

    report: dict[str, object] = {}

    # ---- P1: reproduce the ledger's 153 / 66 / 788 -------------------------
    # The ledger's split is over DISAGREEMENTS (a row where target == sf has dQ == 0
    # by construction and cannot be in either tail).
    for name, keep in (
        ("valid_only", valid),
        ("valid_and_disagree", valid & ~agree),
    ):
        d = dQ[keep]
        report[f"split_{name}"] = {
            "n": int(keep.sum()),
            "bad": int((d <= BAD_CUT).sum()),
            "good": int((d >= GOOD_CUT).sum()),
            "nontail": int(((d > BAD_CUT) & (d < GOOD_CUT)).sum()),
            "bad_mean_dQ": float(d[d <= BAD_CUT].mean()) if (d <= BAD_CUT).any() else None,
        }

    # Lock onto whichever filter reproduces the ledger counts.
    keep = valid & ~agree
    tgt = report["split_valid_and_disagree"]
    assert isinstance(tgt, dict)
    if not (tgt["bad"] == 153 and tgt["good"] == 66 and tgt["nontail"] == 788):
        report["P1"] = "MISMATCH - population differs from the ledger; screen numbers VOID"
    else:
        report["P1"] = "EXACT match to ledger 153/66/788"

    d = dQ[keep]
    bad = d <= BAD_CUT
    report["n_pop"] = int(keep.sum())
    report["n_bad"] = int(bad.sum())
    report["base_rate"] = float(bad.mean())

    # ---- P2-P5: single-feature AUCs ---------------------------------------
    # Sign convention: every score is oriented so LARGER = more likely BAD.
    feats = {
        "sf_cp_regret_tgt": sf_reg[keep],
        "not_listed": (~tgt_listed[keep]).astype(np.float64),
        "n_legal": n_legal[keep],
        "neg_top1_mass": -top1_mass[keep],
    }
    report["auc"] = {}
    for k, v in feats.items():
        a = auc(v, bad)
        lo, hi = boot_ci(v, bad)
        report["auc"][k] = {"auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)]}

    # ---- P6/P7: the cost curve --------------------------------------------
    # A screen is only useful if it beats selecting rows at random, so the random
    # baseline (== the base rate) is reported at every operating point.
    def curve(score: np.ndarray) -> list[dict[str, float]]:
        order = np.argsort(-score, kind="mergesort")
        out = []
        for frac in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
            k = int(round(frac * score.size))
            sel = order[:k]
            capt = float(bad[sel].sum() / max(bad.sum(), 1))
            # deficit captured: how much of the bad tail's total dQ shortfall is in there
            deficit_all = float(-d[bad].sum())
            deficit_sel = float(-d[sel][bad[sel]].sum())
            out.append(
                {
                    "select_frac": frac,
                    "n_selected": k,
                    "bad_captured": round(capt, 4),
                    "random_baseline": round(frac, 4),
                    "lift": round(capt / frac, 3) if frac else None,
                    "deficit_captured": round(deficit_sel / deficit_all, 4) if deficit_all else None,
                    "cost_multiplier": round(1.0 + frac * 6.0, 3),
                }
            )
        return out

    report["curve"] = {k: curve(v) for k, v in feats.items()}

    # A trivially-combined screen: rank-average of the two features that can
    # plausibly carry independent signal (shallow regret, and coverage).
    def ranknorm(v: np.ndarray) -> np.ndarray:
        o = np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort")
        return o.astype(np.float64) / max(v.size - 1, 1)

    combo = ranknorm(feats["sf_cp_regret_tgt"]) + ranknorm(feats["not_listed"])
    a = auc(combo, bad)
    lo, hi = boot_ci(combo, bad)
    report["auc"]["combo_regret_plus_notlisted"] = {"auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    report["curve"]["combo_regret_plus_notlisted"] = curve(combo)

    # ---- the irreducible residual -----------------------------------------
    # Rows the shallow teacher is CONFIDENT about and wrong about. No screen built
    # from the shallow teacher can ever see these; they bound every number above.
    conf = sf_reg[keep] <= 1.0  # <=1cp shallow regret == teacher says "this move is fine"
    report["blind_spot"] = {
        "definition": "sf_cp_regret_tgt <= 1cp AND dQ <= -0.10",
        "n": int((conf & bad).sum()),
        "share_of_bad": round(float((conf & bad).sum() / max(bad.sum(), 1)), 4),
        "mean_dQ": float(d[conf & bad].mean()) if (conf & bad).any() else None,
    }

    # sanity: terminal references should not dominate either tail
    report["terminal_in_bad"] = int((term[keep][bad][:, :2] != 0).any(axis=1).sum())

    out = Path(__file__).resolve().parent / "tailscreen_results.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
