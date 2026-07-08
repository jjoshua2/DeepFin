"""Calibrate the cheap continuation pre-filter against deep-SF ground truth.

For each look-ahead horizon h (plies), classify the seed cheaply from the
seed-side sf_q at +h and compare to the deep-SF verdict. Reports, per h:
  * agree     — cheap verdict == deep verdict (raw agreement)
  * FP-caught — of deep-FINE seeds, how many the cheap rule would DROP (good)
  * REAL-drop — of deep-LOST seeds, how many the cheap rule would wrongly DROP (bad)

The pre-filter only ever DROPS (never admits), so the operating point is the
shortest h that still drops ZERO real blind spots while catching most FPs.

Ground truth source: either the scaling jsonl (per-seed ``verdict`` dict keyed by
node budget — the deepest budget is used) or the single-budget calib jsonl
(``deep`` field).
"""
from __future__ import annotations

import argparse
import glob
import json

from scripts.blindspot_continuation import (
    classify,
    default_replay_dir,
    load_game_rows,
    parse_seeds,
)
from scripts.blindspot_deepsf_calibrate import deep_verdict


def load_ground_truth(path: str, budget: int | None) -> dict[tuple[int, float], str]:
    """(game_id, round(sq,2)) -> deep verdict, from scaling or calib jsonl."""
    gt: dict[tuple[int, float], str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["game"], round(r["sq"], 2))
            if isinstance(r.get("verdict"), dict) and r["verdict"]:
                budgets = sorted(int(k) for k in r["verdict"])
                use = budget if (budget in budgets) else budgets[-1]
                dv = r["verdict"][str(use)]
            else:
                dv = r.get("deep")
            if dv in ("LOST", "MID", "FINE"):
                gt[key] = dv
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-jsonl", default="scratchpad/harvest_fp/deepsf_scaling.jsonl")
    ap.add_argument("--deep-budget", type=int, default=0, help="node budget to use as truth (0=deepest)")
    ap.add_argument("--severe-glob", default="data/harvest/blindspot_live.severe.p*.txt")
    ap.add_argument("--horizons", default="2,4,6,8,10,12,16,20,24,32")
    ap.add_argument("--recover-to", type=float, default=-0.2)
    ap.add_argument("--still-lost", type=float, default=-0.5)
    args = ap.parse_args()
    hs = [int(x) for x in args.horizons.split(",") if x.strip()]

    gt = load_ground_truth(args.deep_jsonl, args.deep_budget or None)
    seeds = parse_seeds(sorted(glob.glob(args.severe_glob)))
    games = load_game_rows(default_replay_dir(), {s.game_id for s in seeds})
    print(f"[prefilter] ground truth: {len(gt)} seeds from {args.deep_jsonl} "
          f"(budget={'deepest' if not args.deep_budget else args.deep_budget})")

    print(f"\n  {'h(ply)':>6} {'n':>4} {'agree':>6}  {'FP-caught':>10}  {'REAL-dropped':>12}")
    best = (-1.0, -1)
    for h in hs:
        agree = n = fp_tot = fp_caught = real_tot = real_dropped = 0
        for s in seeds:
            d = gt.get((s.game_id, round(s.sq, 2)))
            if d is None:
                continue
            v = classify(s, games.get(s.game_id), recover_to=args.recover_to,
                         still_lost=args.still_lost, confirm_h=h, horizons=hs, tol=0.03)
            q = v.profile.get(h)
            if q is None:
                continue
            cheap = deep_verdict(q, args.still_lost, args.recover_to)
            n += 1
            agree += (cheap == d)
            if d == "FINE":
                fp_tot += 1
                fp_caught += (cheap == "FINE")
            if d == "LOST":
                real_tot += 1
                real_dropped += (cheap == "FINE")
        if n:
            ag = 100 * agree / n
            if real_dropped == 0 and ag > best[0]:
                best = (ag, h)
            print(f"  {h:>6} {n:>4} {ag:5.0f}%  {fp_caught:>4}/{fp_tot:<4} FINE  "
                  f"{real_dropped:>3}/{real_tot:<3} LOST")
    if best[1] > 0:
        print(f"\n  best SAFE horizon (0 real dropped, max agreement): {best[1]} plies ({best[0]:.0f}%)")


if __name__ == "__main__":
    main()
