"""xtab.py -- BT4 cascade gate cross-tab (task #247).

Answers ONE question: inside the not-listed rows, does the target's own `top1_mass`
still separate BT4-corroborated error from BT4-endorsed divergence -- i.e. does the
cascade's second (free) gate add anything over the first?

Reads ONLY banked artifacts. Zero GPU, zero Stockfish, no live change. Predictions in
PREDICTIONS_bt4cascade.md were committed before this ran.

Population, declared: DISAGREEMENT rows, `trained = ~postckpt_mask` applied, finite Q
on cand 0 (tgt) and 1 (sf). The no-mask variant is printed as a robustness row only --
the banked C table (n=1158) and the banked dQ tables (n~1007) are DIFFERENT
populations and conflating them is how 176/82/900 got reported for 153/66/788.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC = Path("/home/josh/projects/chess/scratchpad/target_vs_bt4")
OUT = Path("/home/josh/projects/chess/scratchpad/bt4cascade")
RNG = np.random.default_rng(0)
N_BOOT = 10_000
N_FLOOR = 30  # pre-registered resolution floor on the smaller compared cell
BANDS: list[tuple[str, float, float]] = [
    ("[0,0.5)", 0.0, 0.5),
    ("[0.5,0.9)", 0.5, 0.9),
    ("[0.9,0.99)", 0.9, 0.99),
    ("[0.99,1.01)", 0.99, 1.01),
]


def boot_ci(x: np.ndarray) -> tuple[float, float, float]:
    """Bootstrap mean + 95% CI over rows. Returns (mean, lo, hi)."""
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, x.size, size=(N_BOOT, x.size))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_diff_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Bootstrap CI on mean(a) - mean(b), resampling each cell independently."""
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    ia = RNG.integers(0, a.size, size=(N_BOOT, a.size))
    ib = RNG.integers(0, b.size, size=(N_BOOT, b.size))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    rows = np.load(SRC / "tb4_rows.npz", allow_pickle=True)
    qq = np.load(SRC / "tb4_q_winner.npz", allow_pickle=True)
    post = np.load(SRC / "tb4_postckpt_mask.npy")

    cand = [str(c) for c in qq["cand"]]
    assert cand[0] == "tgt" and cand[1] == "sf", f"candidate order changed: {cand}"

    Q = qq["Q"]
    dQ = Q[:, 0] - Q[:, 1]          # target minus SF, under BT4
    prefers_sf = dQ < 0.0            # C = P(BT4 prefers SF's move)
    tie = dQ == 0.0

    listed = rows["tgt_listed"].astype(bool)
    top1 = rows["top1_mass"].astype(np.float64)
    agree = rows["agree"].astype(bool)
    trained = ~post
    finite = np.isfinite(Q[:, 0]) & np.isfinite(Q[:, 1])

    rep: dict[str, object] = {}
    rep["candidate_order"] = cand
    rep["tie_rate_all_finite"] = round(float(tie[finite].mean()), 5)

    # ---- P1/P2 REPRODUCTION CONTROLS, on the banked population (NO trained mask) ----
    # The banked C table is over disagreement rows with no trained mask; reproduce it
    # on ITS OWN population or the rig is wrong.
    base_nomask = (~agree) & finite
    c_nl, lo, hi = boot_ci(prefers_sf[base_nomask & ~listed].astype(np.float64))
    rep["P1_C_not_listed_nomask"] = {
        "n": int((base_nomask & ~listed).sum()),
        "C": round(c_nl, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "banked": 0.7815,
    }
    c_l, lo, hi = boot_ci(prefers_sf[base_nomask & listed].astype(np.float64))
    rep["C_listed_nomask"] = {
        "n": int((base_nomask & listed).sum()),
        "C": round(c_l, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "banked": 0.534,
    }
    rep["P2_confidence_bins_nomask"] = []
    for name, lo_b, hi_b in BANDS:
        m = base_nomask & (top1 >= lo_b) & (top1 < hi_b)
        c, l, h = boot_ci(prefers_sf[m].astype(np.float64))
        rep["P2_confidence_bins_nomask"].append({
            "band": name, "n": int(m.sum()), "C": round(c, 4),
            "ci95": [round(l, 4), round(h, 4)],
            "mean_dQ": round(float(dQ[m].mean()) if m.sum() else float("nan"), 4),
        })

    # ---- PRIMARY population: disagreement AND trained AND finite --------------------
    base = (~agree) & trained & finite
    rep["primary_population"] = {
        "n_disagree_trained_finite": int(base.sum()),
        "n_not_listed": int((base & ~listed).sum()),
        "n_listed": int((base & listed).sum()),
    }

    # ---- P4: do not-listed rows skew low confidence? --------------------------------
    rep["P4_median_top1_mass"] = {
        "not_listed": round(float(np.median(top1[base & ~listed])), 4),
        "listed": round(float(np.median(top1[base & listed])), 4),
    }

    # ---- THE CROSS-TAB: tgt_listed x top1_mass band --------------------------------
    xtab = []
    for name, lo_b, hi_b in BANDS:
        band = (top1 >= lo_b) & (top1 < hi_b)
        row: dict[str, object] = {"band": name}
        for label, sel in (("not_listed", base & ~listed & band), ("listed", base & listed & band)):
            c, l, h = boot_ci(prefers_sf[sel].astype(np.float64))
            row[label] = {
                "n": int(sel.sum()), "C": round(c, 4), "ci95": [round(l, 4), round(h, 4)],
                "mean_dQ": round(float(dQ[sel].mean()) if sel.sum() else float("nan"), 4),
            }
        xtab.append(row)
    rep["cross_tab"] = xtab

    # ---- DECIDING STATISTIC: dC inside the not-listed rows --------------------------
    lowc = base & ~listed & (top1 < 0.5)
    highc = base & ~listed & (top1 >= 0.9)
    n_small = int(min(lowc.sum(), highc.sum()))
    dc, dlo, dhi = boot_diff_ci(
        prefers_sf[lowc].astype(np.float64), prefers_sf[highc].astype(np.float64)
    )
    # ⚑ RESOLUTION BEFORE THRESHOLD. The floor is checked BEFORE any verdict is read,
    # and a shortfall is a STOP -- not a prompt to soften the bar.
    if n_small < N_FLOOR:
        verdict = f"NO RESOLUTION (smaller cell n={n_small} < {N_FLOOR})"
    elif dlo > 0.10:
        verdict = "GATE 2 EARNS ITS PLACE"
    elif dlo > 0.0:
        verdict = "WEAK"
    elif dhi < 0.0:
        verdict = "INVERTED -- STOP, re-derive"
    else:
        verdict = "REDUNDANT"
    rep["deciding_statistic"] = {
        "definition": "C(not_listed & top1<0.5) - C(not_listed & top1>=0.9)",
        "n_low": int(lowc.sum()), "n_high": int(highc.sum()), "n_smaller_cell": n_small,
        "resolution_floor": N_FLOOR,
        "dC": round(dc, 4), "ci95": [round(dlo, 4), round(dhi, 4)],
        "VERDICT": verdict,
    }

    # ---- P7: are BT4-endorsed not-listed rows the keep-worthy ones? -----------------
    keep = base & ~listed & ~prefers_sf
    drop = base & ~listed & prefers_sf
    m_keep, kl, kh = boot_ci(dQ[keep])
    m_drop, dl2, dh2 = boot_ci(dQ[drop])
    rep["P7_signed_split_not_listed"] = {
        "bt4_prefers_ours": {"n": int(keep.sum()), "mean_dQ": round(m_keep, 4),
                             "ci95": [round(kl, 4), round(kh, 4)]},
        "bt4_prefers_sf": {"n": int(drop.sum()), "mean_dQ": round(m_drop, 4),
                           "ci95": [round(dl2, 4), round(dh2, 4)]},
    }

    # ---- P8: what share of ALL rows would the cascade downweight? -------------------
    all_finite = finite & trained
    rep["P8_downweight_share_of_all_rows"] = {
        "cascade_not_listed_and_bt4_corroborates": round(
            float((base & ~listed & prefers_sf).sum() / all_finite.sum()), 4),
        "blanket_not_listed_drop": round(float((base & ~listed).sum() / all_finite.sum()), 4),
        "denominator_n": int(all_finite.sum()),
    }

    (OUT / "xtab_results.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
