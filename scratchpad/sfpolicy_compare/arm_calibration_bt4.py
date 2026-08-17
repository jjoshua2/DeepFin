"""Arm calibration: which reduction of SF's MultiPV points our policy the right way?

Prereg: ledger 564e6035c, amended cca357141.

Scores each candidate ``w_sf_own_regret`` formulation as a GRADIENT DIRECTION on
our policy logits, and asks how well that direction agrees with a direction that
points at a stronger policy (BT4, and deep Stockfish).

    d_arm  = -d(arm_term)/d(logits)          evaluated at OUR net's logits
    d_bt4  =  p_bt4  - p_ours                descent dir of CE(p_bt4  || p_ours)
    d_deep =  p_deep - p_ours                descent dir of CE(p_deep || p_ours)

metric = per-row cosine(d_arm, d_ruler) over the LEGAL support.

⚑ The two negative controls are the point of the rig, not decoration. Every arm
has the shape d_j = -p_j (r_j - rbar), which pushes mass off our top move for ANY
r that is larger there. Our policy is over-confident relative to both rulers, so a
pure entropy regulariser scores POSITIVE here while carrying zero SF information.
N2 (anti-prior) measures exactly that, and an arm that does not beat it is an
entropy regulariser wearing an SF costume.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.selfplay.finalize import SF_OWN_REGRET_CAP_CP, _sf_move_score

PROD_MULTIPV = 6          # production's MultiPV width
TOP2_FLOOR = 0.10         # arm C's floor; binds 23.02% of production rows (probe 08-17)


def _load_jsonl(path: Path) -> dict[str, dict]:
    """key -> record (last wins). The audit caches are one record per position."""
    out: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def _load_shallow(path: Path, tier: int) -> dict[str, dict]:
    """key -> the record at the requested node tier (the file holds several)."""
    out: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r.get("nodes_requested", 0)) == tier:
                out[r["key"]] = r
    return out


def _probs_over(topk: list, ucis: list[str]) -> np.ndarray | None:
    """Gather a UCI-keyed top-k into a dense vector over ``ucis``, renormalised.

    Returns None if the cache put no mass on this position's legal moves, which
    would make the row's cosine undefined rather than zero.
    """
    m = {u: float(p) for u, p in topk}
    v = np.array([m.get(u, 0.0) for u in ucis], dtype=np.float64)
    s = v.sum()
    if not np.isfinite(s) or s <= 0.0:
        return None
    return v / s


def _cos(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0 or not np.isfinite(na) or not np.isfinite(nb):
        return None
    return float(np.dot(a, b) / (na * nb))


def _softmax_dir(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """-d/dlogits of sum_m p_m r_m  =  -p*(r - <p,r>)."""
    return -p * (r - float(np.dot(p, r)))


def _arm_directions(
    p_ours: np.ndarray,
    ucis: list[str],
    pvs: list[dict],
    rng: random.Random,
) -> dict[str, np.ndarray] | None:
    """Every arm's descent direction on this position, or None if unscoreable."""
    n = len(ucis)
    idx_of = {u: i for i, u in enumerate(ucis)}

    # --- production-shaped surfaced set: MultiPV 6, scored, legal -------------
    scored: list[tuple[int, float, float]] = []   # (idx, cp_score, Q in [0,1])
    for pv in pvs[:PROD_MULTIPV]:
        i = idx_of.get(str(pv.get("move", "")))
        if i is None:
            continue
        cp = pv.get("cp")
        mate = int(pv.get("mate") or 0)
        # SF_CP_SENTINEL stands in for "no cp"; _sf_move_score returns None then.
        score = _sf_move_score(-32768 if cp is None else int(cp), mate)
        if score is None:
            continue
        w, d, _l = (list(pv.get("wdl") or [0.0, 0.0, 0.0]) + [0.0, 0.0, 0.0])[:3]
        scored.append((i, float(score), float(w) + float(d) / 2.0))
    if len(scored) < 2:
        return None                      # no contrast to measure on this row

    surfaced = np.zeros(n, dtype=bool)
    for i, _s, _q in scored:
        surfaced[i] = True

    best_cp = max(s for _i, s, _q in scored)
    best_q = max(q for _i, _s, q in scored)

    # --- arm A: production. cp-space, ALL legal, constant fill on the tail ----
    r_cp = np.zeros(n, dtype=np.float64)
    worst = 0.0
    for i, s, _q in scored:
        v = min(max(best_cp - s, 0.0), SF_OWN_REGRET_CAP_CP) / SF_OWN_REGRET_CAP_CP
        r_cp[i] = v
        worst = max(worst, v)
    fill = (worst + 1.0) / 2.0
    r_A = np.where(surfaced, r_cp, fill)

    # --- arm B: #447's gate. same cp regret, surfaced support only -----------
    # loss = sum_{m in S} p_m r_m  =>  d_j = -p_j(r_j*1[j in S] - <p_S, r>)
    mean_B = float(np.dot(p_ours * surfaced, r_cp))
    d_B = -p_ours * (np.where(surfaced, r_cp, 0.0) - mean_B)

    # --- arm C: one-sided floor on SF's top-1/top-2 (always genuinely surfaced)
    order = sorted(scored, key=lambda t: -t[1])
    floor_idx = [i for i, _s, _q in order[:2]]
    # loss = sum_{m in F} relu(tau - p_m); dL/dp_m = -1 where it binds
    d_C = np.zeros(n, dtype=np.float64)
    for m in floor_idx:
        if p_ours[m] < TOP2_FLOOR:
            e = np.zeros(n, dtype=np.float64)
            e[m] = 1.0
            d_C += p_ours[m] * (e - p_ours)

    # --- arm D: WDL-space regret from SF's OWN w/d, surfaced support ---------
    r_q = np.zeros(n, dtype=np.float64)
    for i, _s, q in scored:
        r_q[i] = max(0.0, best_q - q)
    mean_D = float(np.dot(p_ours * surfaced, r_q))
    d_D = -p_ours * (np.where(surfaced, r_q, 0.0) - mean_D)

    # --- N1: shuffled regret. destroys the SF signal, keeps the magnitudes ---
    vals = [r_cp[i] for i, _s, _q in scored]
    rng.shuffle(vals)
    r_N1 = np.zeros(n, dtype=np.float64)
    for (i, _s, _q), v in zip(scored, vals, strict=True):
        r_N1[i] = v
    mean_N1 = float(np.dot(p_ours * surfaced, r_N1))
    d_N1 = -p_ours * (np.where(surfaced, r_N1, 0.0) - mean_N1)

    # --- N2: anti-prior. pure confidence flattening, zero SF input -----------
    d_N2 = _softmax_dir(p_ours, p_ours.copy())

    return {
        "A_prod_cp_fill": _softmax_dir(p_ours, r_A),
        "B_gated_cp": d_B,
        "C_top2_floor": d_C,
        "D_wdl_space": d_D,
        "N1_shuffled": d_N1,
        "N2_antiprior": d_N2,
    }


def _paired_bootstrap(vals: list[float], iters: int, rng: np.random.Generator) -> tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    means = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(iters)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--bt4", type=Path, required=True)
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--shallow", type=Path,
                    default=Path("data/audit_set_v1.jsonl.shallow_sf.jsonl"))
    ap.add_argument("--tier", type=int, default=50000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ours = _load_jsonl(args.ours)
    bt4 = _load_jsonl(args.bt4)
    deep = _load_jsonl(args.audit_set)
    shallow = _load_shallow(args.shallow, args.tier)
    print(f"[rig] ours={len(ours)} bt4={len(bt4)} deep={len(deep)} "
          f"shallow@{args.tier}={len(shallow)}")

    keys = sorted(set(ours) & set(bt4) & set(deep) & set(shallow))
    print(f"[rig] intersection = {len(keys)} positions")

    rng = random.Random(args.seed)
    # key-indexed so every contrast is computed on rows where BOTH arms are
    # defined. An arm is UNdefined on a row when its direction is identically
    # zero (C's floor does not bind; D's WDL regret is flat because every
    # surfaced move is equally won/lost) -- and those subsets differ by >1200
    # rows, so unpaired means compare different populations.
    cos: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    ocos: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    skipped = defaultdict(int)

    for key in keys:
        rec_deep = deep[key]
        board = chess.Board(rec_deep["fen"])
        ucis = [m.uci() for m in board.legal_moves]
        if len(ucis) < 2:
            skipped["<2 legal"] += 1
            continue

        p_ours = _probs_over(ours[key]["topk"], ucis)
        p_bt4 = _probs_over(bt4[key]["topk"], ucis)
        if p_ours is None or p_bt4 is None:
            skipped["no policy mass"] += 1
            continue

        # deep-SF ruler: softmax over cp-rank of the DEEP multipv, production's
        # transform is a rank/cp map; here a plain cp-logistic over surfaced
        # moves, uniform-zero elsewhere, is enough to define a direction.
        idx_of = {u: i for i, u in enumerate(ucis)}
        dv = np.zeros(len(ucis), dtype=np.float64)
        got = 0
        for pv in rec_deep.get("multipv", []):
            i = idx_of.get(str(pv.get("move", "")))
            if i is None:
                continue
            cp = pv.get("cp")
            s = _sf_move_score(-32768 if cp is None else int(cp), int(pv.get("mate") or 0))
            if s is None:
                continue
            dv[i] = s
            got += 1
        if got < 2:
            skipped["deep <2 scored"] += 1
            continue
        mask = dv != 0.0
        z = np.where(mask, dv, dv[mask].min() if mask.any() else 0.0)
        z = (z - z.max()) / 100.0                    # cp -> logit, 100cp scale
        p_deep = np.exp(z)
        p_deep /= p_deep.sum()

        dirs = _arm_directions(p_ours, ucis, shallow[key].get("pvs", []), rng)
        if dirs is None:
            skipped["shallow <2 scored"] += 1
            continue

        d_bt4 = p_bt4 - p_ours
        d_deep = p_deep - p_ours

        # ⚑ The SHARPNESS axis dominates the raw cosine: our policy is
        # over-confident against both rulers, so ANY direction that flattens it
        # scores positive with zero information (that is what N2 measures). To
        # separate "re-rank the moves" from "be less confident", project the
        # flattening direction out of both the arm and the ruler, and score the
        # residuals. This is the ranking-only read.
        f = dirs["N2_antiprior"]
        fn2 = float(np.dot(f, f))

        def _orth(v: np.ndarray) -> np.ndarray:
            return v if fn2 <= 0.0 else v - (float(np.dot(v, f)) / fn2) * f

        for ruler, dr in (("bt4", d_bt4), ("deep", d_deep)):
            dr_o = _orth(dr)
            for name, d in dirs.items():
                c = _cos(d, dr)
                if c is not None:
                    cos[ruler][name][key] = c
                if name == "N2_antiprior":
                    continue          # orthogonal to itself by construction
                co = _cos(_orth(d), dr_o)
                if co is not None:
                    ocos[ruler][name][key] = co

    print(f"[rig] skipped: {dict(skipped)}")
    print()

    boot = np.random.default_rng(args.seed)
    order = ("A_prod_cp_fill", "B_gated_cp", "C_top2_floor", "D_wdl_space",
             "N1_shuffled", "N2_antiprior")
    contrasts = (("D_wdl_space", "A_prod_cp_fill"),
                 ("D_wdl_space", "B_gated_cp"),
                 ("B_gated_cp", "A_prod_cp_fill"),
                 ("C_top2_floor", "N2_antiprior"),
                 ("A_prod_cp_fill", "N2_antiprior"),
                 ("B_gated_cp", "N2_antiprior"),
                 ("D_wdl_space", "N2_antiprior"),
                 ("A_prod_cp_fill", "N1_shuffled"),
                 ("D_wdl_space", "N1_shuffled"),
                 ("C_top2_floor", "N1_shuffled"))

    report: dict = {"tier": args.tier, "n_keys": len(keys),
                    "skipped": dict(skipped), "views": {}}

    for label, table in (("RAW (sharpness + ranking)", cos),
                         ("ORTHOGONALISED (ranking only)", ocos)):
        for ruler in ("bt4", "deep"):
            print(f"═══ {label} — ruler: {ruler} ═══")
            print(f"{'arm':<18} {'n':>6} {'mean cos':>10} {'95% CI':>22}")
            rows: dict = {}
            for name in order:
                d = table[ruler].get(name) or {}
                if not d:
                    continue
                v = list(d.values())
                lo, hi = _paired_bootstrap(v, args.bootstrap, boot)
                print(f"{name:<18} {len(v):>6} {np.mean(v):>10.4f}   "
                      f"[{lo:>+7.4f}, {hi:>+7.4f}]")
                rows[name] = {"n": len(v), "mean": float(np.mean(v)), "ci": [lo, hi]}
            print()
            print("  paired contrasts (rows where BOTH arms are defined):")
            cc: dict = {}
            for a, b in contrasts:
                da, db = table[ruler].get(a) or {}, table[ruler].get(b) or {}
                shared = sorted(set(da) & set(db))
                if len(shared) < 30:
                    continue
                diff = np.array([da[k] - db[k] for k in shared])
                lo, hi = _paired_bootstrap(list(diff), args.bootstrap, boot)
                sig = "" if (lo <= 0.0 <= hi) else "  ***"
                print(f"    {a:<16} - {b:<16} n={len(shared):>5} "
                      f"{diff.mean():>+8.4f}  [{lo:>+7.4f}, {hi:>+7.4f}]{sig}")
                cc[f"{a}-{b}"] = {"n": len(shared), "mean": float(diff.mean()),
                                  "ci": [lo, hi], "significant": not (lo <= 0.0 <= hi)}
            rows["_contrasts"] = cc
            report["views"].setdefault(label, {})[ruler] = rows
            print()

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[rig] wrote {args.out}")


if __name__ == "__main__":
    main()
