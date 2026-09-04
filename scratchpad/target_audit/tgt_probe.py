"""tgt_probe.py -- CPU-only forensic characterisation of the stored policy target.

Reads production replay shards (zarr) and characterises `policy_target`:
  1. entropy / support / top-k mass
  2. agreement with the SF-implied best move (from `sf_p0_regret`, P0, this position)
  3. how far SEARCH moved the target off the raw net prior
     (`priority_policy_kl` = KL(prior || target) in nats, computed in
      mcts/_mcts_tree.c over the legal set at record time -- RAW, unnormalised)
  4. drift across eras (one trial directory per era)

Negative controls:
  C1 shuffle: pair policy_target[i] with sf_p0_regret[perm[i]] -> agreement must
     collapse to the chance baseline mean(1/n_legal).
  C2 alignment: mass of policy_target[i] under legal_mask[perm[i]] -> must
     collapse from ~1.0.

NO GPU. Read-only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import zarr

REPLAY = "/home/josh/projects/chess/runs/pbt2_small/replay"

ERAS = [
    ("A_jun07", "train_trial_7cc7c_00000_0_lr=0.0003_2026-06-04_16-53-49"),
    ("B_jul30", "train_trial_13a9f_00000_0_lr=0.0000_2026-07-26_06-02-14"),
    ("C_aug09", "train_trial_379f6_00000_0_lr=0.0000_2026-08-06_23-51-06"),
    ("D_aug12", "train_trial_b384d_00000_0_lr=0.0000_2026-08-12_17-56-41"),
    ("E_aug15", "train_trial_1d175_00000_0_lr=0.0000_2026-08-14_13-53-53"),
]

N_SHARDS = 8
RNG = np.random.default_rng(20260815)


def q(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {}
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def summarise(name, a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    d = {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0}
    d.update(q(a))
    return d


def load_era(path, n_shards, tail=True):
    sd = os.path.join(path, "replay_shards")
    names = sorted(os.listdir(sd))
    names = [n for n in names if n.endswith(".zarr")]
    # take the NEWEST n_shards by index (tail) -- most recent data in that era
    pick = names[-n_shards:] if tail else names[:n_shards]
    return [os.path.join(sd, n) for n in pick]


MISSING: dict[str, list[str]] = {}

FIELDS = ("policy_target", "legal_mask", "has_policy", "sf_p0_regret",
          "has_sf_p0_regret", "priority_policy_kl", "has_priority_policy_kl",
          "priority_q_delta", "is_selfplay", "ply_index", "priority")


def probe_shard(p):
    z = zarr.open(p, mode="r")
    out = {"__attrs__": dict(z.attrs)}
    n = int(z["policy_target"].shape[0])
    a = int(z["policy_target"].shape[1])
    for k in FIELDS:
        if k in z:
            out[k] = np.asarray(z[k][:])
        elif k.startswith("has_"):
            out[k] = np.zeros((n,), dtype=np.uint8)
        elif k in ("sf_p0_regret", "legal_mask"):
            out[k] = np.zeros((n, a), dtype=np.float16)
        else:
            out[k] = np.zeros((n,), dtype=np.float32)
        if k not in z:
            MISSING.setdefault(os.path.basename(p), []).append(k)
    return out





def run_era(label, path):
    shard_paths = load_era(path, N_SHARDS)
    rows = {}
    attrs0 = None
    for sp in shard_paths:
        d = probe_shard(sp)
        if attrs0 is None:
            attrs0 = d["__attrs__"]
        d.pop("__attrs__")
        for k, v in d.items():
            rows.setdefault(k, []).append(v)
    D = {k: np.concatenate(v, axis=0) for k, v in rows.items()}

    n_total = D["policy_target"].shape[0]
    hp = D["has_policy"].astype(bool)
    n_hp = int(hp.sum())

    pol = D["policy_target"][hp].astype(np.float64)
    lm = D["legal_mask"][hp].astype(bool)
    n_legal = lm.sum(axis=1).astype(np.float64)

    res = {
        "era": label,
        "dir": os.path.basename(path),
        "shards": [os.path.basename(s) for s in shard_paths],
        "policy_encoding": (attrs0 or {}).get("policy_encoding"),
        "policy_size": int(D["policy_target"].shape[1]),
        "n_rows_total": n_total,
        "n_rows_has_policy": n_hp,
        "frac_has_policy": n_hp / n_total,
    }

    # ---- sanity: sums and illegal mass -------------------------------------
    s = pol.sum(axis=1)
    illegal_mass = (pol * (~lm)).sum(axis=1)
    res["target_row_sum"] = summarise("sum", s)
    res["target_mass_on_ILLEGAL"] = summarise("illegal", illegal_mass)

    # renormalise over legal (f16 storage -> sums are ~1 but not exactly)
    polm = pol * lm
    z = polm.sum(axis=1, keepdims=True)
    z[z <= 0] = 1.0
    P = polm / z

    # ---- 1. entropy / support ----------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(P > 0, np.log(P), 0.0)
    H = -(P * lp).sum(axis=1)
    eff = np.exp(H)
    srt = -np.sort(-P, axis=1)
    top1 = srt[:, 0]
    top3 = srt[:, :3].sum(axis=1)
    top8 = srt[:, :8].sum(axis=1)
    top16 = srt[:, :16].sum(axis=1)
    supp_1e3 = (P > 1e-3).sum(axis=1).astype(np.float64)
    supp_1e2 = (P > 1e-2).sum(axis=1).astype(np.float64)
    supp_nz = (P > 0).sum(axis=1).astype(np.float64)

    res["n_legal"] = summarise("nl", n_legal)
    res["entropy_nats"] = summarise("H", H)
    res["eff_support_expH"] = summarise("eff", eff)
    res["top1_mass"] = summarise("t1", top1)
    res["top3_mass"] = summarise("t3", top3)
    res["top8_mass"] = summarise("t8", top8)
    res["top16_mass"] = summarise("t16", top16)
    res["support_gt_1e-3"] = summarise("s3", supp_1e3)
    res["support_gt_1e-2"] = summarise("s2", supp_1e2)
    res["support_nonzero"] = summarise("snz", supp_nz)
    # uniform-over-legal reference: H_max = log(n_legal)
    res["entropy_frac_of_log_nlegal"] = summarise("hf", H / np.log(np.maximum(n_legal, 2)))

    # ---- 3. how much did SEARCH move the target ----------------------------
    hk = D["has_priority_policy_kl"].astype(bool) & hp
    kl_all = D["priority_policy_kl"].astype(np.float64)
    kl = kl_all[hk]
    res["n_rows_has_kl"] = int(hk.sum())
    res["KL_prior_to_target_nats"] = summarise("kl", kl)
    # ratio against that row's own target entropy
    idx_hp = np.flatnonzero(hp)
    pos_in_hp = {int(v): i for i, v in enumerate(idx_hp)}
    sel = np.array([pos_in_hp[int(i)] for i in np.flatnonzero(hk)], dtype=np.int64)
    Hk = H[sel]
    res["KL_over_H"] = summarise("r", kl / np.maximum(Hk, 1e-9))
    # Pinsker TV bound: TV <= sqrt(KL/2)
    res["TV_bound_sqrt_KL_over_2"] = summarise("tv", np.sqrt(np.maximum(kl, 0) / 2.0))
    res["frac_KL_lt_0.01"] = float((kl < 0.01).mean())
    res["frac_KL_lt_0.05"] = float((kl < 0.05).mean())
    res["frac_KL_lt_0.10"] = float((kl < 0.10).mean())
    res["frac_KL_exactly_0"] = float((kl == 0).mean())
    qd = np.abs(D["priority_q_delta"].astype(np.float64)[hp])
    res["abs_q_delta"] = summarise("qd", qd)

    # ---- 2. SF alignment ---------------------------------------------------
    hs = D["has_sf_p0_regret"].astype(bool) & hp
    res["n_rows_has_sf_p0_regret"] = int(hs.sum())
    res["frac_has_sf_p0_regret_of_haspolicy"] = float((hs & hp).sum() / max(n_hp, 1))
    if hs.sum() > 0:
        sel2 = np.array([pos_in_hp[int(i)] for i in np.flatnonzero(hs)], dtype=np.int64)
        Ps = P[sel2]
        Ls = lm[sel2]
        R = D["sf_p0_regret"][hs].astype(np.float64)
        nl_s = n_legal[sel2]

        # fill value = the value the builder wrote at every uncovered index,
        # including every ILLEGAL index -> read it off an illegal index.
        illeg = ~Ls
        fill = np.full(R.shape[0], np.nan)
        any_illeg = illeg.any(axis=1)
        # take the max-count value among illegal entries (they are all identical)
        for i in np.flatnonzero(any_illeg):
            vals = R[i][illeg[i]]
            fill[i] = vals[0]
        ok = np.isfinite(fill)
        covered = Ls & (R != fill[:, None])
        n_cov = covered.sum(axis=1).astype(np.float64)

        # SF best move = argmin regret over legal
        Rmask = np.where(Ls, R, np.inf)
        sf_best = np.argmin(Rmask, axis=1)
        tgt_best = np.argmax(Ps, axis=1)
        agree = (sf_best == tgt_best)
        mass_on_sfbest = Ps[np.arange(Ps.shape[0]), sf_best]
        # rank of sf_best under the target
        order = np.argsort(-Ps, axis=1)
        rank = np.argmax(order == sf_best[:, None], axis=1) + 1

        mass_on_covered = (Ps * covered).sum(axis=1)
        # E[regret | listed] : renormalise the target over the MultiPV-covered set
        denom = np.maximum(mass_on_covered, 1e-12)
        e_reg_listed = (Ps * covered * R).sum(axis=1) / denom
        # naive E[regret] over all legal (74% fabricated -- reported for contrast only)
        e_reg_all = (Ps * Ls * np.where(Ls, R, 0.0)).sum(axis=1)

        res["sf_multipv_covered_count"] = summarise("nc", n_cov[ok])
        res["sf_fill_value"] = summarise("fv", fill[ok])
        res["AGREE_target_argmax_eq_sf_best"] = float(agree[ok].mean())
        res["CHANCE_mean_1_over_nlegal"] = float((1.0 / nl_s[ok]).mean())
        res["mass_on_sf_best"] = summarise("m", mass_on_sfbest[ok])
        res["rank_of_sf_best_in_target"] = summarise("rk", rank[ok])
        res["frac_sf_best_in_target_top3"] = float((rank[ok] <= 3).mean())
        res["frac_sf_best_in_target_top8"] = float((rank[ok] <= 8).mean())
        res["mass_on_sf_multipv_covered_set"] = summarise("mc", mass_on_covered[ok])
        res["E_regret_given_listed"] = summarise("erl", e_reg_listed[ok])
        res["E_regret_all_legal_FABRICATED_TAIL"] = summarise("era", e_reg_all[ok])
        res["frac_target_argmax_is_covered"] = float(
            covered[np.arange(Ps.shape[0]), tgt_best][ok].mean())

        # ---- CONTROL C1: shuffle sf_p0_regret rows --------------------------
        perm = RNG.permutation(Ps.shape[0])
        sf_best_sh = sf_best[perm]
        # only meaningful where the shuffled best index is legal here; keep raw
        res["CONTROL_shuffled_agreement"] = float((sf_best_sh == tgt_best)[ok].mean())
        res["CONTROL_shuffled_mass_on_sf_best"] = summarise(
            "m", Ps[np.arange(Ps.shape[0]), sf_best_sh][ok])

    # ---- CONTROL C2: cross-row legal mask ----------------------------------
    perm2 = RNG.permutation(P.shape[0])
    res["CONTROL_mass_under_shuffled_legal_mask"] = summarise(
        "cm", (pol * lm[perm2]).sum(axis=1))
    res["mass_under_own_legal_mask"] = summarise("om", (pol * lm).sum(axis=1))

    return res


def main():
    out = []
    for label, d in ERAS:
        p = os.path.join(REPLAY, d)
        if not os.path.isdir(p):
            print("MISSING", p, file=sys.stderr)
            continue
        print("=== era", label, flush=True)
        out.append(run_era(label, p))
    dst = "/home/josh/projects/chess/scratchpad/target_audit/tgt_probe_results.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", dst)
    if MISSING:
        print("MISSING FIELDS:", json.dumps({k: v for k, v in list(MISSING.items())[:20]}, indent=1))
        print("n shards with missing fields:", len(MISSING))


if __name__ == "__main__":
    main()
