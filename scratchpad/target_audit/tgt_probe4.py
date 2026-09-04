"""tgt_probe4.py -- does the search MOVEMENT buy target quality?

Bin rows by the stored KL(prior || target) and read SF agreement + cp regret of
the target's argmax in each bin. Also: how sharp is the target on the rows where
search moved it least (KL < 0.01) -- those are the rows where the target is,
numerically, the net's own prior.

PREDICTION (registered before running): agreement will FALL with KL, because
high-KL rows are harder positions where both search and SF disagree with the
net.  The informative reading is the LOW-KL end: if the near-zero-KL rows show
agreement no better than the rest, then the ~1/3 of the window where search
changed essentially nothing is training the net on its own opinion.

CPU only, read-only.
"""
from __future__ import annotations

import json
import os

import numpy as np
import zarr

REPLAY = "/home/josh/projects/chess/runs/pbt2_small/replay"
ERAS = [
    ("C_aug09", "train_trial_379f6_00000_0_lr=0.0000_2026-08-06_23-51-06"),
    ("E_aug15", "train_trial_1d175_00000_0_lr=0.0000_2026-08-14_13-53-53"),
]
N_SHARDS = 24
CAP = 1000.0
RNG = np.random.default_rng(99)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def run(label, path):
    sd = os.path.join(path, "replay_shards")
    names = sorted(x for x in os.listdir(sd) if x.endswith(".zarr"))[-N_SHARDS:]
    acc = {}
    for nm in names:
        z = zarr.open(os.path.join(sd, nm), mode="r")
        for k in ("policy_target", "legal_mask", "has_policy", "sf_p0_regret",
                  "has_sf_p0_regret", "priority_policy_kl"):
            acc.setdefault(k, []).append(np.asarray(z[k][:]))
    D = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    hp = D["has_policy"].astype(bool)
    hs = D["has_sf_p0_regret"].astype(bool) & hp

    P = D["policy_target"][hs].astype(np.float64)
    L = D["legal_mask"][hs].astype(bool)
    P = P * L
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    R = D["sf_p0_regret"][hs].astype(np.float64)
    kl = D["priority_policy_kl"].astype(np.float64)[hs]
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(P * np.where(P > 0, np.log(P), 0.0)).sum(axis=1)
    top1 = P.max(axis=1)

    fill = np.array([R[i][~L[i]][0] if (~L[i]).any() else np.nan for i in range(R.shape[0])])
    ok = np.isfinite(fill)
    covered = L & (R != fill[:, None])
    sf_best = np.argmin(np.where(L, R, np.inf), axis=1)
    tb = np.argmax(P, axis=1)
    agree = sf_best == tb
    r_argmax = R[np.arange(P.shape[0]), tb] * CAP
    listed = covered[np.arange(P.shape[0]), tb]
    perm = RNG.permutation(P.shape[0])
    agree_sh = sf_best[perm] == tb

    edges = [-1e9, 0.005, 0.02, 0.05, 0.15, 0.4, 1.0, 1e9]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (kl >= lo) & (kl < hi)
        n = int(m.sum())
        if n == 0:
            continue
        mm = m & listed
        rows.append({
            "kl_bin": f"[{lo:g},{hi:g})", "n": n, "frac_of_rows": round(n / int(ok.sum()), 4),
            "mean_kl": round(float(kl[m].mean()), 4),
            "mean_top1": round(float(top1[m].mean()), 4),
            "mean_H": round(float(H[m].mean()), 4),
            "agree": round(float(agree[m].mean()), 4),
            "ci95": wilson(int(agree[m].sum()), n),
            "CONTROL_shuffled_agree": round(float(agree_sh[m].mean()), 5),
            "cp_regret_argmax_listed": round(float(r_argmax[mm].mean()), 2) if mm.sum() else None,
            "n_listed": int(mm.sum()),
        })
    return {"era": label, "n_sf_rows": int(ok.sum()), "bins": rows}


def main():
    out = [run(l, os.path.join(REPLAY, d)) for l, d in ERAS]
    dst = "/home/josh/projects/chess/scratchpad/target_audit/tgt_probe4_results.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    for e in out:
        print("=" * 100)
        print(e["era"], "n_sf_rows", e["n_sf_rows"])
        print(f"{'kl_bin':16s}{'n':>7s}{'frac':>7s}{'meanKL':>9s}{'top1':>7s}{'H':>7s}"
              f"{'agree':>8s}{'ci95':>19s}{'shuf':>8s}{'cp_reg':>9s}")
        for b in e["bins"]:
            print(f"{b['kl_bin']:16s}{b['n']:>7d}{b['frac_of_rows']:>7.3f}{b['mean_kl']:>9.4f}"
                  f"{b['mean_top1']:>7.3f}{b['mean_H']:>7.3f}{b['agree']:>8.3f}"
                  f"{str(b['ci95']):>19s}{b['CONTROL_shuffled_agree']:>8.4f}"
                  f"{(b['cp_regret_argmax_listed'] or 0):>9.1f}")


if __name__ == "__main__":
    main()
