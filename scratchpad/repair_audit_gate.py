"""AUDIT-FIRST GATE for SF target repair. Prereg: ledger `039fe82dd` (+ amendment).

Scores repaired training targets against the FROZEN DEEP audit ruler.

THREE TIERS, all on the SAME 4000 frozen positions:
  DEEP RULER      the audit set's own MultiPV (>=1M nodes, MPV>=10) -- scores everything
  STRONG teacher  500k nodes / MPV40  -- cached, NOT shippable (MPV40 alone ~7x)
  PROD teacher    75k nodes / MPV6    -- what the loop actually labels with

⚑ Uses the audit's OWN scorer (`move_regrets`, `expected_and_top1_regret`),
imported not re-derived; reproduction verified to max 9.8e-03 cp.
⚑ LP/transplant copied verbatim from `repair_variant_screen.py` (`c1cc6e9b2`) so
the gate and the screen share one mechanism. Deliberate change: the candidate set
is the TEACHER'S LISTED MOVES, not the 2-3 argmax union, because that is what a
production repair sees.

Edge convention: for i, j with q_i > q_j, ordinary CE pushes the SF-correct way
iff g_ij = (p_i - p_j) - (t_i - t_j) < 0; VIOLATED when g_ij >= 0.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/home/josh/projects/chess")
from chess_anti_engine.eval.audit import (
    expected_and_top1_regret,
    load_audit_set,
    move_regrets,
)
from chess_anti_engine.stockfish.wdl import cp_to_wdl

EPS = 1e-3
SLOPE, DRAW_W = 0.006, 120.0
RNG = np.random.default_rng(20260819)
DUMP = "scratchpad/repairgate_prodteacher.jsonl"
CACHE = "data/audit_set_v1.jsonl.shallow_sf.jsonl"

ARMS = ["unchanged", "rho=0", "rho=1", "transplant", "blend", "shuffled",
        "strong_oracle", "screened"]

# ⚑ STRONG_ORACLE IS THE CEILING ARM AND IS **NOT SHIPPABLE**. rho=1 against the
# 500k/MPV40 teacher. Its job is the question the weak-teacher result cannot
# answer: if the repair wins with a strong teacher and loses with the production
# one, the TEACHER is the bottleneck and a screen is worth building; if it loses
# with BOTH, the repair idea is dead independent of teacher quality. Free -- the
# labels were already cached.
#
# SCREENED applies rho=1 with the PRODUCTION teacher, but only where an
# out-of-fold screen predicts the weak teacher is reliable. The screen is FIT
# against the strong teacher (offline, once) and READS ONLY production-observable
# features, so it is deployable. Scored on the DEEP ruler -- a third instrument
# neither the screen nor the repair was fit to, which is what breaks circularity.


def q_of(e: dict) -> float | None:
    cp, mate = e.get("cp"), e.get("mate")
    if cp is None and not mate:
        return None
    v = cp_to_wdl(None if cp is None else float(cp),
                  int(mate) if mate else None, slope=SLOPE, draw_width_cp=DRAW_W)
    return float(v[0]) - float(v[2])


def load_teachers() -> tuple[dict, dict]:
    prod, strong = {}, {}
    for line in open(CACHE):
        if not line.strip():
            continue
        d = json.loads(line)
        mp = {e["move"]: e for e in (d.get("pvs") or [])}
        if int(d.get("multipv", 0)) == 6:
            prod[d["key"]] = mp
        elif int(d.get("multipv", 0)) == 40:
            strong[d["key"]] = mp
    return prod, strong


def edge_list(q: np.ndarray) -> list[tuple[int, int]]:
    n = len(q)
    return [(i, j) for i in range(n) for j in range(n) if i != j and q[i] > q[j]]


def lp_repair(t, p, edges, rho: float):
    n = len(t)
    A, b = [], []
    for i, j in edges:
        g = (p[i] - p[j]) - (t[i] - t[j])
        rhs = (p[i] - p[j]) + rho * g + EPS
        row = np.zeros(2 * n)
        row[i], row[j] = -1.0, 1.0
        A.append(row)
        b.append(-rhs)
    for k in range(n):
        r1 = np.zeros(2 * n)
        r1[k], r1[n + k] = 1.0, -1.0
        A.append(r1)
        b.append(float(t[k]))
        r2 = np.zeros(2 * n)
        r2[k], r2[n + k] = -1.0, -1.0
        A.append(r2)
        b.append(float(-t[k]))
    Aeq = np.zeros((1, 2 * n))
    Aeq[0, :n] = 1.0
    res = linprog(np.concatenate([np.zeros(n), np.ones(n)]),
                  A_ub=np.array(A), b_ub=np.array(b),
                  A_eq=Aeq, b_eq=[float(t.sum())],
                  bounds=[(0.0, None)] * (2 * n), method="highs")
    return (np.asarray(res.x[:n]), True) if res.success else (t.copy(), False)


def transplant(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = np.empty_like(t)
    out[np.argsort(-q)] = np.sort(t)[::-1]
    return out


def kendall_agree(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n < 2:
        return 1.0
    tot = ok = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] == a[j] or b[i] == b[j]:
                continue
            tot += 1
            ok += int((a[i] > a[j]) == (b[i] > b[j]))
    return ok / tot if tot else 1.0


def main() -> None:
    aset = {p.key: p for p in load_audit_set("data/audit_set_v1.jsonl")}
    rows = [json.loads(x) for x in open(DUMP) if x.strip()]
    prod_t, strong_t = load_teachers()
    print(f"audit {len(aset)}  dump {len(rows)}  prod-teacher {len(prod_t)}  "
          f"strong-teacher {len(strong_t)}")

    recs = []
    for r in rows:
        pos = aset.get(r["key"])
        mpP = prod_t.get(r["key"])
        if pos is None or not mpP:
            continue
        c = r["cand"]
        t_all, p_all = dict(c["train"]["probs"]), dict(c["raw"]["probs"])
        legal = list(t_all)
        listed = [m for m in legal if m in mpP and q_of(mpP[m]) is not None]
        if len(listed) < 2:
            continue
        qP = np.array([q_of(mpP[m]) for m in listed], float)
        if len(np.unique(qP)) < 2:
            continue
        t = np.array([t_all.get(m, 0.0) for m in listed], float)
        p = np.array([p_all.get(m, 1e-9) for m in listed], float)
        if t.sum() <= 0:
            continue
        mpS = strong_t.get(r["key"]) or {}
        qS = np.array([(q_of(mpS[m]) if m in mpS and q_of(mpS[m]) is not None else np.nan)
                       for m in listed], float)
        recs.append({"key": r["key"], "pos": pos, "legal": legal, "listed": listed,
                     "qP": qP, "qS": qS, "t": t, "p": p,
                     "base": np.array([t_all[m] for m in legal], float),
                     "idx": [legal.index(m) for m in listed]})
    print(f"rows usable: {len(recs)}")

    # ---- the deployable screen: fit against the STRONG teacher, read PROD features
    feats, labs = [], []
    for rec in recs:
        qP, qS = rec["qP"], rec["qS"]
        o = np.argsort(-qP)
        s = np.sort(qP)[::-1]
        margin = float(s[0] - s[1]) if len(s) > 1 else 0.0
        w = np.exp((qP - qP.max()) * 4.0)
        w /= w.sum()
        feats.append([len(qP), margin, float(s[0] - s[-1]),
                      float(-(w * np.log(w + 1e-12)).sum()), len(rec["legal"])])
        ok = np.isfinite(qS)
        labs.append(int(ok.sum() >= 2 and kendall_agree(qP[ok], qS[ok]) >= 0.8
                        and int(np.argmax(qP)) == int(np.argmax(np.where(ok, qS, -np.inf)))))
    X, y = np.array(feats, float), np.array(labs, int)
    print(f"screen label base rate (prod teacher agrees with strong): {y.mean():.3f}")
    pred = np.zeros(len(y))
    if 0 < y.mean() < 1:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        from sklearn.metrics import roc_auc_score
        print(f"screen OOF AUC: {roc_auc_score(y, pred):.3f}")
    thr = float(np.median(pred))

    per_arm = {a: [] for a in ARMS}
    infeas = {a: 0 for a in ARMS}
    l1 = {a: [] for a in ARMS}
    n_viol = 0
    for k, rec in enumerate(recs):
        qP, qS, t, p = rec["qP"], rec["qS"], rec["t"], rec["p"]
        edges = edge_list(qP)
        if any((p[i] - p[j]) - (t[i] - t[j]) >= 0 for i, j in edges):
            n_viol += 1
        regs = move_regrets(rec["pos"], rec["legal"])
        for arm in ARMS:
            ok = True
            if arm == "unchanged":
                tp = t.copy()
            elif arm == "rho=0":
                tp, ok = lp_repair(t, p, edges, 0.0)
            elif arm == "rho=1":
                tp, ok = lp_repair(t, p, edges, 1.0)
            elif arm == "transplant":
                tp = transplant(t, qP)
            elif arm == "shuffled":
                tp, ok = lp_repair(t, p, edge_list(qP[RNG.permutation(len(qP))]), 1.0)
            elif arm == "blend":
                w = 0.25
                z = np.exp((qP - qP.max()) * 4.0)
                tp = (1.0 - w) * t + w * (z / z.sum() * t.sum())
            elif arm == "strong_oracle":
                m = np.isfinite(qS)
                if m.sum() < 2 or len(np.unique(qS[m])) < 2:
                    tp = t.copy()
                else:
                    sub, okk = lp_repair(t[m], p[m], edge_list(qS[m]), 1.0)
                    tp = t.copy()
                    tp[m] = sub
                    tp *= t.sum() / max(tp.sum(), 1e-12)
                    ok = okk
            else:  # screened
                if pred[k] >= thr:
                    tp, ok = lp_repair(t, p, edges, 1.0)
                else:
                    tp = t.copy()
            if not ok:
                infeas[arm] += 1
            l1[arm].append(float(np.abs(tp - t).sum()))
            new = rec["base"].copy()
            new[rec["idx"]] = tp
            per_arm[arm].append(expected_and_top1_regret(new, regs)[0])

    print(f"violated before repair: {n_viol}/{len(recs)} = {n_viol / max(len(recs), 1):.3f}")
    base_arr = np.array(per_arm["unchanged"])
    print(f"\n{'arm':>14s} {'E[deep regret]':>15s} {'delta':>10s} {'95% CI':>22s} "
          f"{'meanL1':>8s} {'infeas':>7s}")
    out = {}
    for arm in ARMS:
        a = np.array(per_arm[arm])
        if arm == "unchanged":
            print(f"{arm:>14s} {a.mean():15.3f} {'--':>10s} {'--':>22s} "
                  f"{np.mean(l1[arm]):8.4f} {infeas[arm]:7d}")
            continue
        d = base_arr - a  # positive = repair BETTER (less regret)
        n = len(d)
        boot = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[arm] = (float(d.mean()), float(lo), float(hi))
        print(f"{arm:>14s} {a.mean():15.3f} {d.mean():+10.3f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22s} {np.mean(l1[arm]):8.4f} {infeas[arm]:7d}")

    print("\n--- PRE-COMMITTED RULE (ledger 039fe82dd) ---")
    ship = {k: v for k, v in out.items()
            if k not in ("blend", "shuffled", "strong_oracle")}
    best = max(ship, key=lambda k: ship[k][0])
    d, lo, hi = ship[best]
    sd, slo, shi = out["shuffled"]
    checks = {
        "best shippable arm >= 3.0 cp": d >= 3.0,
        "its 95% CI excludes 0": lo > 0.0,
        "beats naive blend": d > out["blend"][0],
        "shuffled control shows no gain": slo <= 0.0,
    }
    print(f"best SHIPPABLE arm: {best}  {d:+.3f} cp  [{lo:+.3f}, {hi:+.3f}]")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\nVERDICT:", "PASS -> write a training prereg" if all(checks.values())
          else "KILL -> no training compute")
    od, olo, ohi = out["strong_oracle"]
    print(f"\nCEILING (not shippable) strong_oracle: {od:+.3f} [{olo:+.3f}, {ohi:+.3f}]")
    print("  -> " + ("teacher-limited: a strong teacher DOES buy target quality, so a "
                     "screen/correction is worth building"
                     if od >= 3.0 and olo > 0 else
                     "NOT teacher-limited: the repair fails even with a strong teacher, "
                     "so better labels cannot rescue it"))
    json.dump({"deltas": out, "n": len(recs), "n_viol": n_viol,
               "screen_base_rate": float(y.mean())},
              open("scratchpad/repair_audit_gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
