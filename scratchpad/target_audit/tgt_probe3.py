"""tgt_probe3.py -- (a) explain the probe2 x-check shortfall, (b) redo the headline
statistics under the PRIORITY weighting the sampler actually uses.

PREDICTIONS (registered before running):
  (a) restricting the x-check to rows where ply_index[i] - ply_index[i-1] == 1
      lifts match_rate above 0.98 in every era -> the D/E shortfall is a PAIRING
      failure of my probe, not a decoding failure.
  (b) priority-weighted agreement is LOWER than unweighted by 2-6 pp and
      priority-weighted entropy is HIGHER, because priority = 6*|q_delta| +
      3.5*KL oversamples rows where search disagreed with the net.

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
CAP = 1000.0


def load(path, n):
    sd = os.path.join(path, "replay_shards")
    names = sorted(x for x in os.listdir(sd) if x.endswith(".zarr"))[-n:]
    acc = {}
    for nm in names:
        z = zarr.open(os.path.join(sd, nm), mode="r")
        for k in ("policy_target", "legal_mask", "has_policy", "sf_p0_regret",
                  "has_sf_p0_regret", "priority_policy_kl", "sf_multipv_raw",
                  "game_id", "ply_index", "priority", "priority_q_delta"):
            if k in z:
                acc.setdefault(k, []).append(np.asarray(z[k][:]))
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def wq(x, w, ps):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return {f"p{p}": float(np.interp(p / 100.0, c, x)) for p in ps}


def run(label, path):
    D = load(path, N_SHARDS)
    hp = D["has_policy"].astype(bool)
    res = {"era": label}

    pol = D["policy_target"][hp].astype(np.float64)
    lm = D["legal_mask"][hp].astype(bool)
    P = pol * lm
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(P * np.where(P > 0, np.log(P), 0.0)).sum(axis=1)
    top1 = P.max(axis=1)
    pri = D["priority"].astype(np.float64)[hp]
    kl = D["priority_policy_kl"].astype(np.float64)[hp]

    res["priority"] = {"mean": float(pri.mean()),
                       **{f"p{p}": float(np.percentile(pri, p)) for p in (5, 50, 95)},
                       "top_decile_share_of_total": float(
                           np.sort(pri)[-int(0.1 * pri.size):].sum() / pri.sum())}
    res["entropy_unweighted_mean"] = float(H.mean())
    res["entropy_PRIORITY_weighted_mean"] = float((H * pri).sum() / pri.sum())
    res["top1_unweighted_mean"] = float(top1.mean())
    res["top1_PRIORITY_weighted_mean"] = float((top1 * pri).sum() / pri.sum())
    res["entropy_quantiles"] = {f"p{p}": float(np.percentile(H, p))
                                for p in (5, 10, 25, 50, 75, 90, 95)}
    res["frac_target_top1_gt_0.99"] = float((top1 > 0.99).mean())
    res["frac_target_top1_gt_0.95"] = float((top1 > 0.95).mean())
    res["KL_PRIORITY_weighted_mean"] = float((kl * pri).sum() / pri.sum())
    res["KL_unweighted_mean"] = float(kl.mean())

    # ---- SF subset, weighted and unweighted --------------------------------
    hs = D["has_sf_p0_regret"].astype(bool) & hp
    sub = hs[hp]
    Ps, Ls, pris = P[sub], lm[sub], pri[sub]
    R = D["sf_p0_regret"][hs].astype(np.float64)
    fill = np.array([R[i][~Ls[i]][0] if (~Ls[i]).any() else np.nan
                     for i in range(R.shape[0])])
    ok = np.isfinite(fill)
    covered = Ls & (R != fill[:, None])
    sf_best = np.argmin(np.where(Ls, R, np.inf), axis=1)
    tgt_best = np.argmax(Ps, axis=1)
    agree = (sf_best == tgt_best).astype(np.float64)
    res["agree_unweighted"] = float(agree[ok].mean())
    res["agree_PRIORITY_weighted"] = float((agree[ok] * pris[ok]).sum() / pris[ok].sum())
    res["top1_sf_subset_unweighted"] = float(Ps.max(axis=1)[ok].mean())
    res["top1_sf_subset_PRIORITY_weighted"] = float(
        (Ps.max(axis=1)[ok] * pris[ok]).sum() / pris[ok].sum())
    mc = (Ps * covered).sum(axis=1)
    e_listed = (Ps * covered * R).sum(axis=1) / np.maximum(mc, 1e-12) * CAP
    res["E_cp_regret_listed_unweighted"] = float(e_listed[ok].mean())
    res["E_cp_regret_listed_PRIORITY_weighted"] = float(
        (e_listed[ok] * pris[ok]).sum() / pris[ok].sum())

    # ---- (a) x-check with the pairing repaired -----------------------------
    gid, ply, raw = D["game_id"], D["ply_index"], D["sf_multipv_raw"]
    idx_hs = np.flatnonzero(hs)
    stats = {}
    for gap in (1, 2, None):
        checks = matches = 0
        for pos, i in enumerate(idx_hs):
            if i == 0:
                continue
            j = i - 1
            if gid[j] != gid[i] or ply[j] >= ply[i]:
                continue
            if gap is not None and int(ply[i]) - int(ply[j]) != gap:
                continue
            rows = raw[j]
            sel = rows[:, 0] >= 0
            if not sel.any():
                continue
            if (rows[sel, 2] != 0).any():
                continue
            mv = rows[sel, 0].astype(np.int64)
            cp = rows[sel, 1].astype(np.float64)
            reg = np.clip(cp.max() - cp, 0, CAP) / CAP
            checks += 1
            if np.abs(R[pos][mv] - reg).max() < 2e-3:
                matches += 1
        stats[str(gap)] = {"n": checks, "match_rate": (matches / checks) if checks else None}
    res["xcheck_by_ply_gap"] = stats
    # distribution of the ply gap on SF rows
    gaps = []
    for i in idx_hs:
        if i == 0 or gid[i - 1] != gid[i]:
            continue
        gaps.append(int(ply[i]) - int(ply[i - 1]))
    g = np.array(gaps)
    res["ply_gap_hist"] = {str(k): int((g == k).sum()) for k in sorted(set(g.tolist()))[:8]}
    res["ply_gap_frac_eq_1"] = float((g == 1).mean())
    return res


def main():
    out = []
    for label, d in ERAS:
        print("===", label, flush=True)
        out.append(run(label, os.path.join(REPLAY, d)))
    dst = "/home/josh/projects/chess/scratchpad/target_audit/tgt_probe3_results.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
