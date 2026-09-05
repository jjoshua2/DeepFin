"""tgt_probe2.py -- follow-up: sharpness-vs-correctness calibration of the stored
policy target, KL distribution repair, and an independent cp-scale cross-check
of the sf_p0_regret decoding against sf_multipv_raw.

CPU only, read-only.
"""
from __future__ import annotations

import json
import os

import numpy as np
import zarr

REPLAY = "/home/josh/projects/chess/runs/pbt2_small/replay"
ERAS = [
    ("B_jul30", "train_trial_13a9f_00000_0_lr=0.0000_2026-07-26_06-02-14"),
    ("C_aug09", "train_trial_379f6_00000_0_lr=0.0000_2026-08-06_23-51-06"),
    ("D_aug12", "train_trial_b384d_00000_0_lr=0.0000_2026-08-12_17-56-41"),
    ("E_aug15", "train_trial_1d175_00000_0_lr=0.0000_2026-08-14_13-53-53"),
]
N_SHARDS = 16
CAP = 1000.0  # SF_OWN_REGRET_CAP_CP
RNG = np.random.default_rng(4242)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def load(path, n):
    sd = os.path.join(path, "replay_shards")
    names = sorted(x for x in os.listdir(sd) if x.endswith(".zarr"))[-n:]
    acc = {}
    for nm in names:
        z = zarr.open(os.path.join(sd, nm), mode="r")
        for k in ("policy_target", "legal_mask", "has_policy", "sf_p0_regret",
                  "has_sf_p0_regret", "priority_policy_kl", "sf_multipv_raw",
                  "has_sf_multipv_raw", "game_id", "ply_index"):
            if k in z:
                acc.setdefault(k, []).append(np.asarray(z[k][:]))
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def run(label, path):
    D = load(path, N_SHARDS)
    hp = D["has_policy"].astype(bool)
    hs = D["has_sf_p0_regret"].astype(bool) & hp
    res = {"era": label, "n_rows": int(hp.size), "n_sf": int(hs.sum())}

    # ---------- KL distribution, repaired -----------------------------------
    kl = D["priority_policy_kl"].astype(np.float64)[hp]
    res["KL_quantiles"] = {f"p{p}": float(np.percentile(kl, p))
                           for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    res["KL_mean"] = float(kl.mean())
    res["KL_frac_NEGATIVE"] = float((kl < 0).mean())
    res["KL_min"] = float(kl.min())
    res["KL_max"] = float(kl.max())

    pol = D["policy_target"][hs].astype(np.float64)
    lm = D["legal_mask"][hs].astype(bool)
    P = pol * lm
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    R = D["sf_p0_regret"][hs].astype(np.float64)
    nl = lm.sum(axis=1)

    fill = np.array([R[i][~lm[i]][0] if (~lm[i]).any() else np.nan
                     for i in range(R.shape[0])])
    ok = np.isfinite(fill)
    covered = lm & (R != fill[:, None])

    sf_best = np.argmin(np.where(lm, R, np.inf), axis=1)
    tgt_best = np.argmax(P, axis=1)
    top1 = P[np.arange(P.shape[0]), tgt_best]
    agree = (sf_best == tgt_best)

    # entropy for the binning
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(P * np.where(P > 0, np.log(P), 0.0)).sum(axis=1)

    # ---------- CALIBRATION: is the target's confidence earned? -------------
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95),
            (0.95, 0.99), (0.99, 1.01)]
    cal = []
    for lo, hi in bins:
        m = ok & (top1 >= lo) & (top1 < hi)
        n = int(m.sum())
        k = int(agree[m].sum())
        c = wilson(k, n)
        cal.append({
            "bin": f"[{lo},{hi})", "n": n,
            "mean_top1_mass": float(top1[m].mean()) if n else None,
            "agree_with_sf_best": (k / n) if n else None,
            "ci95": c,
            "mean_H": float(H[m].mean()) if n else None,
        })
    res["calibration"] = cal
    k_all, n_all = int(agree[ok].sum()), int(ok.sum())
    res["agree_overall"] = k_all / n_all
    res["agree_overall_ci95"] = wilson(k_all, n_all)
    res["mean_top1_mass"] = float(top1[ok].mean())
    res["OVERCONFIDENCE_top1mass_minus_agreement"] = float(top1[ok].mean()) - k_all / n_all

    # ---------- cp-scale quality of the target ------------------------------
    # regret of the target's OWN argmax (only meaningful when it is SF-listed)
    tb_cov = covered[np.arange(P.shape[0]), tgt_best]
    r_argmax = R[np.arange(P.shape[0]), tgt_best] * CAP
    m = ok & tb_cov
    res["n_argmax_listed"] = int(m.sum())
    res["frac_argmax_listed"] = float(tb_cov[ok].mean())
    res["cp_regret_of_target_argmax_LISTED"] = {
        "mean": float(r_argmax[m].mean()),
        **{f"p{p}": float(np.percentile(r_argmax[m], p)) for p in (50, 75, 90, 95, 99)},
        "frac_gt_50cp": float((r_argmax[m] > 50).mean()),
        "frac_gt_100cp": float((r_argmax[m] > 100).mean()),
        "frac_eq_0": float((r_argmax[m] == 0).mean()),
    }
    # target-weighted expected regret restricted to SF-listed moves
    mc = (P * covered).sum(axis=1)
    e_listed = (P * covered * R).sum(axis=1) / np.maximum(mc, 1e-12) * CAP
    res["E_cp_regret_given_listed"] = {
        "mean": float(e_listed[ok].mean()),
        **{f"p{p}": float(np.percentile(e_listed[ok], p)) for p in (50, 75, 90, 95)},
    }
    # how much of the target's LISTED mass sits on moves >= 50 / 100 cp worse
    bad50 = (P * covered * (R * CAP >= 50)).sum(axis=1) / np.maximum(mc, 1e-12)
    bad100 = (P * covered * (R * CAP >= 100)).sum(axis=1) / np.maximum(mc, 1e-12)
    res["listed_mass_on_moves_ge_50cp_worse"] = float(bad50[ok].mean())
    res["listed_mass_on_moves_ge_100cp_worse"] = float(bad100[ok].mean())

    # ---------- CONTROL: shuffle the SF rows --------------------------------
    perm = RNG.permutation(P.shape[0])
    res["CONTROL_shuffled_agree"] = float((sf_best[perm] == tgt_best)[ok].mean())
    calc = []
    for lo, hi in bins:
        m2 = ok & (top1 >= lo) & (top1 < hi)
        n = int(m2.sum())
        calc.append({"bin": f"[{lo},{hi})", "n": n,
                     "shuffled_agree": float((sf_best[perm] == tgt_best)[m2].mean()) if n else None})
    res["CONTROL_calibration"] = calc

    # ---------- independent cp cross-check via sf_multipv_raw --------------
    # row i's sf_p0_regret is built from the PREVIOUS net record's MultiPV rows
    # (SF analyses at P1 = the position after that record's move = THIS row's
    # position). Find i-1 in the same game with ply_index strictly smaller.
    if "sf_multipv_raw" in D and "game_id" in D and "ply_index" in D:
        gid = D["game_id"]
        ply = D["ply_index"]
        raw = D["sf_multipv_raw"]
        idx_hs = np.flatnonzero(hs)
        checks, matches = 0, 0
        maxabs = []
        for pos, i in enumerate(idx_hs):
            if i == 0:
                continue
            j = i - 1
            if gid[j] != gid[i] or ply[j] >= ply[i]:
                continue
            rows = raw[j]
            sel = rows[:, 0] >= 0
            if not sel.any():
                continue
            mv = rows[sel, 0].astype(np.int64)
            cp = rows[sel, 1].astype(np.float64)
            mate = rows[sel, 2].astype(np.float64)
            # _sf_move_score: mate dominates; approximate with cp only when mate==0
            if (mate != 0).any():
                continue
            best = cp.max()
            reg = np.clip(best - cp, 0, CAP) / CAP
            got = R[pos][mv]
            checks += 1
            d = np.abs(got - reg).max()
            maxabs.append(float(d))
            if d < 2e-3:  # f16 storage tolerance
                matches += 1
        res["xcheck_sf_multipv_raw"] = {
            "n_checked": checks, "n_matching": matches,
            "match_rate": (matches / checks) if checks else None,
            "max_abs_dev_p95": float(np.percentile(maxabs, 95)) if maxabs else None,
        }
    return res


def main():
    out = []
    for label, d in ERAS:
        print("===", label, flush=True)
        out.append(run(label, os.path.join(REPLAY, d)))
    dst = "/home/josh/projects/chess/scratchpad/target_audit/tgt_probe2_results.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", dst)


if __name__ == "__main__":
    main()
