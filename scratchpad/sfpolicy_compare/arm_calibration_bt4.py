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

    def _floor_dir(idxs: list[int], tau: float) -> np.ndarray:
        """-d/dlogits of sum_{m in idxs} relu(tau - p_m); dL/dp_m = -1 where it binds."""
        out = np.zeros(n, dtype=np.float64)
        for m in idxs:
            if p_ours[m] < tau:
                e = np.zeros(n, dtype=np.float64)
                e[m] = 1.0
                out += p_ours[m] * (e - p_ours)
        return out

    d_C = _floor_dir(floor_idx, TOP2_FLOOR)

    # ⚑ C_rand — THE control C actually needs. Same mechanism, same tau, same
    # probability regime; only WHICH moves are floored differs. N2/N1 cannot test
    # C: N2 is a ceiling artefact (it prefers any under-weighted move) and N1
    # permutes magnitudes, which C never reads. If C >> C_rand then "the moves SF
    # NAMES" is the information, which is the whole claim.
    surf_idx = [i for i, _s, _q in scored]
    d_Crand = _floor_dir(rng.sample(surf_idx, min(2, len(surf_idx))), TOP2_FLOOR)

    # top-1 only: is the second floored move carrying weight or diluting?
    d_C1 = _floor_dir(floor_idx[:1], TOP2_FLOOR)

    # --- arm E: MEMBERSHIP ONLY. Floor every surfaced move, no rank read at all.
    # If ~77% of C's pull really is "floor a surfaced move", E captures it with
    # strictly less information, and shipping C's rank machinery would be waste.
    d_E = _floor_dir(surf_idx, TOP2_FLOOR)

    # --- arm F: Josh's adaptive floor -------------------------------------
    #   * SF's top-1 ALWAYS,
    #   * plus any surfaced move within `delta` cp of best,
    #   * but ONLY moves scoring strictly better than the move WE currently pick.
    # The last clause is what C lacks: it makes the term go QUIET as our argmax
    # improves, instead of permanently dragging mass onto SF's choice. A move we
    # already prefer that SF also likes needs no floor.
    our_i = int(np.argmax(p_ours))
    score_of = {i: s for i, s, _q in scored}
    our_score = score_of.get(our_i)

    def _adaptive_idx(delta_cp: float) -> list[int]:
        keep = [floor_idx[0]]                      # top-1 unconditionally
        for i, s, _q in scored:
            if i == floor_idx[0]:
                continue
            if best_cp - s > delta_cp:
                continue                            # not within delta of best
            if our_score is not None and s <= our_score:
                continue                            # not better than our pick
            keep.append(i)
        return keep


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
    # ⚑ arm G: the EXISTING `w_sf_own` knob -- plain CE toward the label's top-1.
    # loss = -log p[top1]  =>  d/dlogits = p - e_top1  =>  descent dir = e_top1 - p.
    # UNBOUNDED: it keeps pushing after the move is already searched, unlike the
    # one-sided floor which stops at tau. This is the "more extreme SL" arm and it
    # was never scored -- A/B/D were regret-WEIGHTED EXPECTATIONS, a different shape.
    e_top1 = np.zeros(n, dtype=np.float64)
    e_top1[floor_idx[0]] = 1.0
    d_G = e_top1 - p_ours

    d_N2 = _softmax_dir(p_ours, p_ours.copy())

    return {
        # sentinel: the PRODUCTION MultiPV-6 label's own top-1, popped by the
        # caller. Needed to split do-no-harm by whether the shallow label agrees
        # with the deep ruler -- an arm cannot be blamed for LABEL error.
        "__meta_label_best": np.array([float(floor_idx[0])]),
        "A_prod_cp_fill": _softmax_dir(p_ours, r_A),
        "B_gated_cp": d_B,
        "C_top2_floor": d_C,
        "C_top1_only": d_C1,
        "C_RANDOM_ctl": d_Crand,
        "E_membership_all": d_E,
        "F_adapt20_t0.35": _floor_dir(_adaptive_idx(20.0), 0.35),
        "F_adapt50_t0.35": _floor_dir(_adaptive_idx(50.0), 0.35),
        "F_adapt100_t0.35": _floor_dir(_adaptive_idx(100.0), 0.35),
        "F_adapt20_t0.15": _floor_dir(_adaptive_idx(20.0), 0.15),
        "C_tau0.02": _floor_dir(floor_idx, 0.02),
        "C_tau0.05": _floor_dir(floor_idx, 0.05),
        "C_tau0.20": _floor_dir(floor_idx, 0.20),
        "C_tau0.35": _floor_dir(floor_idx, 0.35),
        "C_tau0.50": _floor_dir(floor_idx, 0.50),
        "D_wdl_space": d_D,
        "G_sf_own_ce": d_G,
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
    # argmax pull is ruler-free: the deep-SF best move IS the target
    pull: dict[str, dict[str, float]] = defaultdict(dict)
    # ⚑ DO-NO-HARM: the COMPLEMENT population -- rows where our argmax ALREADY
    # is SF's best move. A good term is quiet there. d[our_argmax] < 0 means the
    # arm drags mass OFF a move we had right. Josh's clause 3 exists for this.
    harm: dict[str, dict[str, float]] = defaultdict(dict)
    cpm: dict[str, dict[str, float]] = defaultdict(dict)
    # Split at the historical 1/topk probability threshold (topk=16).
    # This is a shared population split, not a claim that any arm is inactive:
    # each arm's floor can still bind above 0.0625 when its own tau is larger.
    pull_hi: dict[str, dict[str, float]] = defaultdict(dict)
    pull_lo: dict[str, dict[str, float]] = defaultdict(dict)
    # ⚑ ORACLE CEILING for a "when to trust SF" gate. Split the pull population by
    # whether the SHALLOW label's top-1 equals the DEEP ruler's best. A confidence
    # signal can at BEST recover the difference between these two; if it is small,
    # no predictor is worth building.
    pull_agree: dict[str, dict[str, float]] = defaultdict(dict)
    pull_dis: dict[str, dict[str, float]] = defaultdict(dict)
    harm_ag: dict[str, dict[str, float]] = defaultdict(dict)
    harm_dis: dict[str, dict[str, float]] = defaultdict(dict)
    label_agree: dict[str, float] = {}
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
        label_best = int(dirs.pop("__meta_label_best")[0]) if dirs else -1
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

        # ⚑⚑ ARGMAX PULL — the metric that matches where our deficit actually is.
        # Our top-1 regret is 46.9 cp against BT4's 20.1 while our TAIL is BETTER
        # than theirs (67.1 vs 81.2), so the entire gap is WHICH MOVE IS #1.
        # Cosine scores calibration and ranking together; this scores ranking only.
        # d[sf_best] - d[our_argmax] > 0 means the arm raises mass on the deep-SF
        # best move faster than on the move we currently pick. Argmax-invariant
        # interventions (e.g. temperature) score exactly 0, which is correct.
        # ⚑⚑ THE CP METRIC. argmax pull scores TOP-1 IDENTITY and gives ZERO credit
        # for promoting a move 5cp from best -- which penalises any arm that follows a
        # teacher whose disagreements are SMALL. Our shallow label's top-1 costs 12.4cp
        # vs our own 46.9cp, so its "errors" are near-misses. This scores the arms on
        # the quantity that matters: the first-order change in EXPECTED DEEP CP LOSS.
        #   p(logits + eps*d) => dE[r]/deps = sum(p*r*d) - (sum p*r)(sum p*d)
        # Reported NEGATED so higher = reduces expected cp loss = better.
        dscored: list[tuple[int, float]] = []
        for pv in (rec_deep.get("multipv") or []):
            j = idx_of.get(str(pv.get("move", "")))
            if j is None:
                continue
            sc = _sf_move_score(-32768 if pv.get("cp") is None else int(pv["cp"]),
                                int(pv.get("mate") or 0))
            if sc is not None:
                dscored.append((j, float(sc)))
        if len(dscored) >= 2:
            dbest = max(v for _j, v in dscored)
            dworst = min(v for _j, v in dscored)
            # unlisted legal moves: worst-listed regret as a FLOOR (audit convention,
            # a lower bound -- MultiPV >= 10 at >= 1M nodes)
            rdeep = np.full(int(p_ours.shape[0]),
                            min(max(dbest - dworst, 0.0), 1000.0), dtype=np.float64)
            for j, v in dscored:
                rdeep[j] = min(max(dbest - v, 0.0), 1000.0)
            pr_ = float(np.dot(p_ours, rdeep))
            for name, d in dirs.items():
                nrm = float(np.linalg.norm(d))
                if nrm > 0.0:
                    rate = float(np.dot(p_ours * rdeep, d)) - pr_ * float(np.dot(p_ours, d))
                    cpm[name][key] = -rate / nrm

        sf_best_uci = str(rec_deep.get("bestmove", ""))
        i_sf = idx_of.get(sf_best_uci)
        i_ours = int(np.argmax(p_ours))
        if i_sf is not None and i_sf == i_ours:
            agree = (label_best == i_sf)
            label_agree[key] = 1.0 if agree else 0.0
            for name, d in dirs.items():
                nrm = float(np.linalg.norm(d))
                v = (float(d[i_ours]) / nrm) if nrm > 0.0 else 0.0
                harm[name][key] = v
                (harm_ag if agree else harm_dis)[name][key] = v
        if i_sf is not None and i_sf != i_ours:
            # normalise by |d| so arms with different natural scales compare
            hi = float(p_ours[i_sf]) >= 0.0625
            lab_ok = (label_best == i_sf)
            for name, d in dirs.items():
                nrm = float(np.linalg.norm(d))
                if nrm > 0.0:
                    v = float(d[i_sf] - d[i_ours]) / nrm
                    pull[name][key] = v
                    (pull_hi if hi else pull_lo)[name][key] = v
                    (pull_agree if lab_ok else pull_dis)[name][key] = v

    print(f"[rig] skipped: {dict(skipped)}")
    print()

    boot = np.random.default_rng(args.seed)
    order = ("A_prod_cp_fill", "B_gated_cp", "C_top2_floor", "D_wdl_space",
             "C_top1_only", "G_sf_own_ce", "E_membership_all",
             "F_adapt20_t0.15", "F_adapt20_t0.35", "F_adapt50_t0.35", "F_adapt100_t0.35",
             "C_tau0.02", "C_tau0.05", "C_tau0.20", "C_tau0.35",
             "C_tau0.50", "C_RANDOM_ctl", "N1_shuffled", "N2_antiprior")
    contrasts = (("D_wdl_space", "A_prod_cp_fill"),
                 ("D_wdl_space", "B_gated_cp"),
                 ("B_gated_cp", "A_prod_cp_fill"),
                 ("C_top2_floor", "N2_antiprior"),
                 ("A_prod_cp_fill", "N2_antiprior"),
                 ("B_gated_cp", "N2_antiprior"),
                 ("D_wdl_space", "N2_antiprior"),
                 ("A_prod_cp_fill", "N1_shuffled"),
                 ("D_wdl_space", "N1_shuffled"),
                 ("C_top2_floor", "N1_shuffled"),
                 # ⚑ the contrasts that actually TEST C: same mechanism both sides
                 ("C_top2_floor", "C_RANDOM_ctl"),
                 ("C_top1_only", "C_RANDOM_ctl"),
                 ("C_top1_only", "C_top2_floor"),
                 ("C_tau0.05", "C_top2_floor"),
                 ("C_tau0.20", "C_top2_floor"),
                 ("C_tau0.35", "C_top2_floor"),
                 ("C_tau0.50", "C_top2_floor"),
                 ("C_tau0.35", "C_RANDOM_ctl"),
                 ("F_adapt20_t0.35", "C_RANDOM_ctl"),
                 ("F_adapt20_t0.35", "C_tau0.35"),
                 ("F_adapt50_t0.35", "F_adapt20_t0.35"),
                 ("F_adapt100_t0.35", "F_adapt20_t0.35"),
                 ("F_adapt20_t0.15", "F_adapt20_t0.35"),
                 ("E_membership_all", "C_RANDOM_ctl"),
                 ("E_membership_all", "C_top2_floor"),
                 ("G_sf_own_ce", "C_RANDOM_ctl"),
                 ("G_sf_own_ce", "F_adapt20_t0.35"),
                 ("G_sf_own_ce", "N2_antiprior"))

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

    print("═══ ARGMAX PULL — d[sf_best] - d[our_argmax], |d|-normalised ═══")
    print("  (rows where SF's best != our argmax; >0 = arm moves us toward SF's move)")
    print(f"{'arm':<18} {'n':>6} {'mean pull':>11} {'95% CI':>24} {'frac>0':>8}")
    prow: dict = {}
    for name in order:
        d = pull.get(name) or {}
        if not d:
            continue
        v = list(d.values())
        lo, hi = _paired_bootstrap(v, args.bootstrap, boot)
        fp = float(np.mean([x > 0 for x in v]))
        print(f"{name:<18} {len(v):>6} {np.mean(v):>+11.5f}   "
              f"[{lo:>+9.5f}, {hi:>+9.5f}] {100*fp:>6.1f}%")
        prow[name] = {"n": len(v), "mean": float(np.mean(v)), "ci": [lo, hi], "frac_pos": fp}
    print()
    print("  paired contrasts:")
    pc: dict = {}
    for a, b in contrasts:
        da, db = pull.get(a) or {}, pull.get(b) or {}
        shared = sorted(set(da) & set(db))
        if len(shared) < 30:
            continue
        diff = np.array([da[k] - db[k] for k in shared])
        lo, hi = _paired_bootstrap(list(diff), args.bootstrap, boot)
        sig = "" if (lo <= 0.0 <= hi) else "  ***"
        print(f"    {a:<16} - {b:<16} n={len(shared):>5} {diff.mean():>+10.5f}  "
              f"[{lo:>+9.5f}, {hi:>+9.5f}]{sig}")
        pc[f"{a}-{b}"] = {"n": len(shared), "mean": float(diff.mean()), "ci": [lo, hi],
                          "significant": not (lo <= 0.0 <= hi)}
    prow["_contrasts"] = pc
    report["argmax_pull"] = prow
    print()

    for lbl, acc in (("HIGH PRIOR  p_ours[sf_best] >= 0.0625", pull_hi),
                     ("LOW PRIOR   p_ours[sf_best] <  0.0625", pull_lo)):
        print(f"═══ ARGMAX PULL — {lbl} ═══")
        print(f"{'arm':<20} {'n':>6} {'mean pull':>11} {'frac>0':>8}   "
              f"{'vs C_RANDOM (paired)':>26}")
        base = acc.get("C_RANDOM_ctl") or {}
        for name in order:
            d = acc.get(name) or {}
            if not d:
                continue
            v = np.array(list(d.values()))
            shared = sorted(set(d) & set(base))
            if shared and name != "C_RANDOM_ctl":
                diff = np.array([d[k] - base[k] for k in shared])
                lo, hi2 = _paired_bootstrap(list(diff), args.bootstrap, boot)
                star = "***" if (lo > 0 or hi2 < 0) else "   "
                con = f"{diff.mean():>+9.4f} [{lo:>+7.4f},{hi2:>+7.4f}] {star}"
            else:
                con = ""
            print(f"{name:<20} {v.size:>6} {v.mean():>+11.5f} {100*(v>0).mean():>7.1f}%   {con}")
        print()

    keep = ("A_prod_cp_fill", "G_sf_own_ce", "C_top2_floor", "C_tau0.35",
            "F_adapt20_t0.15", "F_adapt20_t0.35", "C_RANDOM_ctl")
    print("═══ ΔE[deep cp loss] — cp reduced per unit logit step (PARTIAL CREDIT) ═══")
    print("  the metric argmax-pull cannot see: promoting a move 5cp from best COUNTS.")
    print(f"{'arm':<20} {'n':>6} {'mean cp/step':>13} {'frac>0':>8}   {'vs C_RANDOM (paired)':>26}")
    cbase = cpm.get("C_RANDOM_ctl") or {}
    for name in order:
        d = cpm.get(name) or {}
        if not d:
            continue
        v = np.array(list(d.values()))
        sh = sorted(set(d) & set(cbase))
        if sh and name != "C_RANDOM_ctl":
            diff = np.array([d[k] - cbase[k] for k in sh])
            lo, hi2 = _paired_bootstrap(list(diff), args.bootstrap, boot)
            star = "***" if (lo > 0 or hi2 < 0) else "   "
            con = f"{diff.mean():>+9.4f} [{lo:>+7.4f},{hi2:>+7.4f}] {star}"
        else:
            con = ""
        print(f"{name:<20} {v.size:>6} {v.mean():>+13.5f} {100*(v>0).mean():>7.1f}%   {con}")
    print()

    print("═══ ORACLE CEILING — pull split by whether the SHALLOW label == DEEP best ═══")
    print("  a perfect 'trust SF here' gate zeroes the DISAGREE column; the ceiling is")
    print("  the gap between 'all rows' and 'agree only, disagree contributing 0'.")
    print(f"{'arm':<20} {'agree n':>8} {'agree':>9} {'dis n':>7} {'disagree':>10} "
          f"{'ALL':>9} {'GATED (dis->0)':>15} {'ceiling':>9}")
    for name in keep:
        da, dd = pull_agree.get(name) or {}, pull_dis.get(name) or {}
        if not da and not dd:
            continue
        a = np.array(list(da.values())) if da else np.array([])
        b = np.array(list(dd.values())) if dd else np.array([])
        tot = a.size + b.size
        allm = (a.sum() + b.sum()) / max(tot, 1)
        gated = a.sum() / max(tot, 1)      # disagree rows contribute exactly 0
        print(f"{name:<20} {a.size:>8} {a.mean() if a.size else 0:>+9.4f} {b.size:>7} "
              f"{b.mean() if b.size else 0:>+10.4f} {allm:>+9.4f} {gated:>+15.4f} "
              f"{gated-allm:>+9.4f}")
    print()

    print("═══ DO-NO-HARM — d[our_argmax] on rows where our move IS SF's best ═══")
    print("  (>=0 good: term leaves a correct move alone; <0 = drags mass off it)")
    print(f"{'arm':<20} {'n':>6} {'mean d[ours]':>13} {'frac<0':>8} {'frac==0 (silent)':>18}")
    hrow: dict = {}
    for name in order:
        d = harm.get(name) or {}
        if not d:
            continue
        v = np.array(list(d.values()))
        hrow[name] = {"n": int(v.size), "mean": float(v.mean()),
                      "frac_neg": float((v < 0).mean()), "frac_zero": float((v == 0).mean())}
        print(f"{name:<20} {v.size:>6} {v.mean():>+13.5f} {100*(v<0).mean():>7.1f}% "
              f"{100*(v==0).mean():>17.1f}%")
    report["do_no_harm"] = hrow
    print()

    # ⚑ SPLIT: an arm floors the PRODUCTION MultiPV-6 label's top-1, but this
    # population is defined by the DEEP ruler's best. Where the two disagree the
    # arm is faithfully executing a wrong label -- that is label error, not an
    # arm defect. Where they AGREE the arm has no excuse.
    ag = np.array(list(label_agree.values()))
    agreement_pct = f"{100*ag.mean():.1f}%" if ag.size else "n/a"
    print(f"═══ DO-NO-HARM SPLIT — shallow label agrees with deep ruler on "
          f"{agreement_pct} of n={ag.size} ═══")
    print(f"{'arm':<20} {'mean|AGREE':>11} {'frac<0':>8} {'silent':>8}   "
          f"{'mean|DISAGREE':>14} {'frac<0':>8}")
    srow: dict = {}
    for name in order:
        da, dd = harm_ag.get(name) or {}, harm_dis.get(name) or {}
        if not da and not dd:
            continue
        a = np.array(list(da.values()))
        b = np.array(list(dd.values()))
        srow[name] = {
            "n_agree": int(a.size),
            "mean_agree": float(a.mean()) if a.size else None,
            "frac_neg_agree": float((a < 0).mean()) if a.size else None,
            "frac_zero_agree": float((a == 0).mean()) if a.size else None,
            "n_dis": int(b.size),
            "mean_dis": float(b.mean()) if b.size else None,
            "frac_neg_dis": float((b < 0).mean()) if b.size else None,
        }
        agree_text = (
            f"{a.mean():>+11.5f} {100*(a<0).mean():>7.1f}% {100*(a==0).mean():>7.1f}%"
            if a.size else f"{'n/a':>11} {'n/a':>8} {'n/a':>8}"
        )
        disagree_text = (
            f"{b.mean():>+14.5f} {100*(b<0).mean():>7.1f}%"
            if b.size else f"{'n/a':>14} {'n/a':>8}"
        )
        print(f"{name:<20} {agree_text}   {disagree_text}")
    report["do_no_harm_split"] = srow
    print()

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[rig] wrote {args.out}")


if __name__ == "__main__":
    main()
