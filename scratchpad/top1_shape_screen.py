"""TOP-1 PROMOTION SHAPE SCREEN. Prereg: ledger `6e576e8a5`.

Donor m = argmax t_search, receiver s = SF top-1. Every other target entry is
left untouched. E[regret] is LINEAR in t, so transferring d from m to s changes
regret by exactly d*(r_s - r_m) -- which is why any variant that is a fixed
multiple of another cannot change the teacher-correct/teacher-wrong ratio.

⚑ DEV HALF ONLY unless --seal is passed. The sealed half is touched ONCE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, "/home/josh/projects/chess")
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from chess_anti_engine.eval.audit import load_audit_set, move_regrets
from chess_anti_engine.stockfish.wdl import cp_to_wdl

SLOPE, DRAW_W, EPS = 0.006, 120.0, 1e-6
RNG = np.random.default_rng(20260819)


def q_of(e: dict) -> float | None:
    cp, mate = e.get("cp"), e.get("mate")
    if cp is None and not mate:
        return None
    v = cp_to_wdl(None if cp is None else float(cp),
                  int(mate) if mate else None, slope=SLOPE, draw_width_cp=DRAW_W)
    return float(v[0]) - float(v[2])


def is_dev(key: str) -> bool:
    return int(hashlib.sha1(key.encode()).hexdigest()[0], 16) < 8


def load_rows():
    prod = {}
    for line in open("data/audit_set_v1.jsonl.shallow_sf.jsonl"):
        d = json.loads(line)
        if int(d.get("multipv", 0)) == 6:
            prod[d["key"]] = {e["move"]: e for e in (d.get("pvs") or [])}
    aset = {p.key: p for p in load_audit_set("data/audit_set_v1.jsonl")}
    out = []
    for line in open("scratchpad/repairgate_prodteacher.jsonl"):
        r = json.loads(line)
        pos, mpP = aset.get(r["key"]), prod.get(r["key"])
        if pos is None or not mpP:
            continue
        t_all = dict(r["cand"]["train"]["probs"])
        p_all = dict(r["cand"]["raw"]["probs"])
        legal = list(t_all)
        listed = [m for m in legal if m in mpP and q_of(mpP[m]) is not None]
        if len(listed) < 2:
            continue
        q = np.array([q_of(mpP[m]) for m in listed], float)
        if len(np.unique(q)) < 2:
            continue
        t = np.array([t_all.get(m, 0.0) for m in listed], float)
        p = np.array([p_all.get(m, 1e-9) for m in listed], float)
        if t.sum() <= 0:
            continue
        regs = np.asarray(move_regrets(pos, legal), float)
        rl = {mv: regs[i] for i, mv in enumerate(legal)}
        si = int(np.argmax(q))
        mi = int(np.argmax(t))
        s_mv, m_mv = listed[si], listed[mi]
        out.append({
            "key": r["key"], "dev": is_dev(r["key"]),
            "t_m": float(t[mi]), "t_s": float(t[si]),
            "p_m": float(p[mi]), "p_s": float(p[si]),
            # gain per unit mass transferred m -> s (linearity)
            "grad": float(rl[m_mv] - rl[s_mv]),
            "agree": int(s_mv == legal[int(np.argmin(regs))]),
            "n_listed": len(q), "n_legal": len(legal),
            "margin": float(np.sort(q)[::-1][0] - np.sort(q)[::-1][1]),
            "spread": float(q.max() - q.min()),
            "same": int(si == mi),
            # a random OTHER listed move, for the negative control
            "grad_rand": float(rl[m_mv] - rl[listed[int(RNG.integers(len(listed)))]]),
        })
    return out


def ci(x: np.ndarray, n_boot: int = 8000):
    r = np.random.default_rng(11)
    b = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(x.mean()), *np.percentile(b, [2.5, 97.5])


def report(name, d, agree):
    m, lo, hi = ci(d)
    a, b = d[agree == 1], d[agree == 0]
    print(f"{name:>16s} {m:+7.2f} [{lo:+6.2f},{hi:+6.2f}]   "
          f"correct {a.mean():+7.2f}   WRONG {b.mean():+7.2f}")
    return m, lo, hi, float(b.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--arm", default=None, help="on --seal, the ONE selected arm")
    ap.add_argument("--cap", type=float, default=0.10)
    args = ap.parse_args()

    rows = load_rows()
    half = [r for r in rows if r["dev"] != args.seal]
    print(f"total usable {len(rows)}   {'SEALED' if args.seal else 'DEV'} half: {len(half)}")
    t_m = np.array([r["t_m"] for r in half])
    t_s = np.array([r["t_s"] for r in half])
    p_m = np.array([r["p_m"] for r in half])
    p_s = np.array([r["p_s"] for r in half])
    grad = np.array([r["grad"] for r in half])
    grad_r = np.array([r["grad_rand"] for r in half])
    agree = np.array([r["agree"] for r in half])
    same = np.array([r["same"] for r in half])
    print(f"teacher-correct rate {agree.mean():.3f}   "
          f"rows where donor IS receiver (no-op) {same.mean():.3f}")

    d_swap = np.maximum(0.0, t_m - t_s)
    g = (p_s - p_m) - (t_s - t_m)
    d_veto = np.maximum(0.0, g / 2.0)
    arms = {
        "top1_swap": d_swap,
        "tie": d_swap / 2.0,
        "veto": np.minimum(d_veto, d_swap),
        f"capped({args.cap})": np.minimum(d_swap, args.cap * t_m),
    }
    for c in (0.02, 0.05, 0.25, 0.50):
        arms[f"capped({c})"] = np.minimum(d_swap, c * t_m)

    print(f"\n{'arm':>16s} {'aggregate cp':>9s} {'95% CI':>16s}   "
          f"{'teacher-correct':>16s}   {'teacher-WRONG':>14s}")
    res = {}
    for name, d in arms.items():
        res[name] = report(name, d * grad, agree)
    # negative control: promote a RANDOM listed move by the same swap mass
    report("shuffled_top1", d_swap * grad_r, agree)

    # ⚑ RIG TEST, predicted in the prereg BEFORE running: tie must be EXACTLY
    # half of top1_swap on both subsets, because d_tie == d_swap/2 identically.
    sw, ti = arms["top1_swap"] * grad, arms["tie"] * grad
    err = float(np.abs(ti - 0.5 * sw).max())
    print(f"\nRIG TEST  max|tie - 0.5*swap| = {err:.3e}  -> "
          f"{'CONFIRMED (tie is a global 0.5, as predicted)' if err < 1e-12 else '⚑ RIG WRONG'}")

    if not args.seal:
        # regcap: predict the SIGNED benefit from production-observable features
        X = np.column_stack([
            t_m, t_s, t_m - t_s, p_m, p_s, g,
            [r["n_listed"] for r in half], [r["n_legal"] for r in half],
            [r["margin"] for r in half], [r["spread"] for r in half],
        ])
        y = d_swap * grad
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            pred[te] = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
        for k in (2.0, 8.0):
            frac = 1.0 / (1.0 + np.exp(-k * pred))
            report(f"regcap(k={k})", d_swap * frac * grad, agree)
        print(f"  regcap OOF corr(pred, actual) = {np.corrcoef(pred, y)[0, 1]:.3f}")
        print("\n--- DEV SELECTION against the pre-committed rule ---")
        print("   need: teacher-WRONG >= -1.0 cp  AND  aggregate > +3.0 cp, CI excluding 0")
        for name, (m, lo, hi, wrong) in res.items():
            ok = wrong >= -1.0 and m > 3.0 and lo > 0.0
            print(f"   {'CANDIDATE' if ok else 'no       '}  {name:>14s}  "
                  f"agg {m:+6.2f} [{lo:+6.2f}] wrong {wrong:+6.2f}")
    else:
        name = args.arm
        m, lo, hi, wrong = res[name]
        print(f"\n=== SEALED-HALF GO/NO-GO on `{name}` ===")
        c1, c2 = wrong >= -1.0, (m > 3.0 and lo > 0.0)
        print(f"  {'PASS' if c1 else 'FAIL'}  teacher-WRONG {wrong:+.2f} >= -1.0")
        print(f"  {'PASS' if c2 else 'FAIL'}  aggregate {m:+.2f} > +3.0 with CI [{lo:+.2f},{hi:+.2f}] excluding 0")
        print("VERDICT:", "PASS -> small live arm justified" if (c1 and c2)
              else "FAIL -> close SF target manipulation for this run")


if __name__ == "__main__":
    main()
