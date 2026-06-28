#!/usr/bin/env python3
"""Analyse the marginal-value-of-search table from backtest_time_value.py.

Answers three questions for the time-management gate:

1. How much regret is AVOIDABLE by allocating search well, and how close does a
   feature-based policy get to the oracle (the lift)? -> "within X Elo of optimal".
2. WHICH features best identify the positions that deserve more search
   (capture@top-k vs an oracle that ranks by the true marginal value)?
3. Are those features / the threshold STATIC across search depth? The same lift
   is computed per budget STEP (512->1024, 1024->2048, ...), so drift is visible.

The unit is WDL expected-score regret (flat in decided positions). "Avoidable"
between budgets b<B is regret(b)-regret(B): the score recovered by more search.
A feature policy spends the extra search on its top-ranked positions; the lift is
the fraction of the oracle's avoidable regret it captures. score->Elo is a rough
per-move surrogate (~695 Elo per unit score near 50%); the real number needs SPRT.

Usage:
  PYTHONPATH=. python3 scripts/analyze_time_value.py \\
      --in runs/backtest/time_value.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

_ELO_PER_SCORE = 400.0 / math.log(10) / 0.25  # ~695, d(Elo)/d(score) at score=0.5

# feature -> "deserves more search" score (higher = spend more here), read at the
# LOWER budget of each step (the decision point). NaN/inf q_gap = decisive -> low.
_FEATURES = {
    "q_gap_top2": lambda r: -_finite(r.get("q_gap_top2"), default=1e9),
    "n_gap_top2": lambda r: -float(r.get("n_gap_top2", 1.0)),
    "visit_entropy": lambda r: float(r.get("visit_entropy", 0.0)),
    "q_drift": lambda r: float(r.get("q_drift") or 0.0),
    "bestmove_flip": lambda r: 1.0 if r.get("bestmove_flip") else 0.0,
    "mid_pieces": lambda r: -abs(int(r.get("piece_count", 16)) - 20),
    # the shipped gate: small Q gap OR just-flipped = keep searching.
    "gate(qgap+flip)": lambda r: (
        (1.0 if (_finite(r.get("q_gap_top2"), 1e9) < 0.10) else 0.0)
        + (1.0 if r.get("bestmove_flip") else 0.0)
    ),
}


def _finite(x, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _capture_at(order: list[int], avoidable: np.ndarray, fracs: tuple[float, ...]) -> dict:
    """Cumulative avoidable regret captured by spending extra search on the
    positions in ``order`` (best first), as a fraction of the total, at each
    top-k fraction. The oracle order is argsort(-avoidable)."""
    total = float(avoidable.sum())
    if total <= 0:
        return {f: float("nan") for f in fracs}
    cum = np.cumsum(avoidable[order])
    n = len(order)
    return {f: float(cum[min(n - 1, int(f * n))] / total) for f in fracs}


def _step_table(rows_by_key: dict, lo: int, hi: int, fracs: tuple[float, ...]) -> dict | None:
    """One budget step lo->hi: avoidable per position + capture@k per feature."""
    keys, avoid, feat_scores = [], [], defaultdict(list)
    for k, byb in rows_by_key.items():
        if lo not in byb or hi not in byb:
            continue
        a = float(byb[lo]["regret_score"]) - float(byb[hi]["regret_score"])
        keys.append(k)
        avoid.append(max(0.0, a))  # clamp: more search making it worse isn't "avoidable here"
        for name, fn in _FEATURES.items():
            feat_scores[name].append(fn(byb[lo]))
    if not keys:
        return None
    avoidable = np.asarray(avoid, dtype=np.float64)
    oracle_order = list(np.argsort(-avoidable))
    rng = np.random.default_rng(0)
    rand_order = list(rng.permutation(len(keys)))
    out: dict = {
        "lo": lo, "hi": hi, "n": len(keys),
        "avoidable_score_mean": float(avoidable.mean()),
        "avoidable_elo_mean": float(avoidable.mean() * _ELO_PER_SCORE),
        "oracle": _capture_at(oracle_order, avoidable, fracs),
        "random": _capture_at(rand_order, avoidable, fracs),
        "features": {},
    }
    for name, scores in feat_scores.items():
        order = list(np.argsort(-np.asarray(scores, dtype=np.float64)))
        out["features"][name] = _capture_at(order, avoidable, fracs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="analyze_time_value")
    ap.add_argument("--in", dest="inp", type=Path, default=Path("runs/backtest/time_value.jsonl"))
    ap.add_argument("--fracs", default="0.1,0.2,0.3,0.5")
    args = ap.parse_args()
    fracs = tuple(float(x) for x in str(args.fracs).split(","))

    rows_by_key: dict[str, dict[int, dict]] = defaultdict(dict)
    for line in args.inp.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows_by_key[r["key"]][int(r["sims"])] = r
    budgets = sorted({b for byb in rows_by_key.values() for b in byb})
    print(f"[analyze] {len(rows_by_key)} positions, budgets {budgets}")

    steps = [(budgets[i], budgets[i + 1]) for i in range(len(budgets) - 1)]
    print(f"\n{'step':>13} {'n':>4} {'avoid_elo':>9} {'best feature @20%':>22} {'capture':>8} {'oracle@20':>9}")
    print("-" * 72)
    per_step = []
    for lo, hi in steps:
        t = _step_table(rows_by_key, lo, hi, fracs)
        if t is None:
            continue
        per_step.append(t)
        i20 = fracs.index(0.2) if 0.2 in fracs else 0
        ranked = sorted(t["features"].items(), key=lambda kv: -(_nan0(list(kv[1].values())[i20])))
        bf = ranked[0][0]
        bc = _nan0(list(ranked[0][1].values())[i20])
        oracle20 = list(t["oracle"].values())[i20]
        print(f"{lo:>5}->{hi:<6} {t['n']:>4} {t['avoidable_elo_mean']:>9.2f} "
              f"{bf:>22} {bc:>8.2f} {oracle20:>9.2f}")

    # Pooled (min->max budget): the headline avoidable + per-feature capture curve.
    pooled = _step_table(rows_by_key, budgets[0], budgets[-1], fracs)
    if pooled is not None:
        print(f"\n=== pooled {budgets[0]}->{budgets[-1]} (headline) ===")
        print(f"avoidable: {pooled['avoidable_score_mean']:.4f} score/move "
              f"(~{pooled['avoidable_elo_mean']:.1f} Elo/move surrogate), n={pooled['n']}")
        hdr = "  ".join(f"@{int(f*100)}%" for f in fracs)
        print(f"\n{'policy':>18}  {hdr}")
        print(f"{'ORACLE':>18}  " + "  ".join(f"{v:>4.2f}" for v in pooled["oracle"].values()))
        for name, cap in sorted(pooled["features"].items(),
                                key=lambda kv: -_nan0(list(kv[1].values())[0])):
            print(f"{name:>18}  " + "  ".join(f"{v:>4.2f}" for v in cap.values()))
        print(f"{'random':>18}  " + "  ".join(f"{v:>4.2f}" for v in pooled["random"].values()))

    # Staticness: is the best feature's capture@20% stable across steps?
    if len(per_step) >= 2:
        print("\n=== staticness (capture@20% per feature, per budget step) ===")
        i20 = fracs.index(0.2) if 0.2 in fracs else 0
        names = list(_FEATURES)
        print(f"{'feature':>18}  " + "  ".join(f"{s['lo']}->{s['hi']}" for s in per_step))
        for name in names:
            vals = [list(s["features"][name].values())[i20] for s in per_step]
            drift = max(vals) - min(vals)
            print(f"{name:>18}  " + "  ".join(f"{v:>8.2f}" for v in vals)
                  + f"   (drift {drift:.2f})")
        print("\nStable across steps (low drift) => the feature/threshold transfers "
              "across search depth; large drift => budget-conditional tuning needed.")


def _nan0(x: float) -> float:
    return 0.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else x


if __name__ == "__main__":
    main()
