"""Depth-scaling check for the deep-SF calibration ground truth.

8M nodes is not an oracle — on horizon/fortress positions SF's eval can keep
moving with depth, and if it does we can't treat any single budget as truth. This
re-evaluates the SAME calibration sample at several node budgets (TB on) and
reports, per budget, the LOST/MID/FINE split plus how many seeds FLIP verdict
between budgets. A verdict stable from 8M to 16M => converged (trust it); lots of
flips => the position is genuinely SF-unstable (an anti-engine target, and a spot
where deep SF is least reliable as ground truth).
"""
from __future__ import annotations

import argparse
import json
import time

from chess_anti_engine.selfplay.opening import seed_board_from_line
from chess_anti_engine.stockfish.uci import StockfishUCI
from scripts.blindspot_deepsf_calibrate import (
    _DEFAULT_SF,
    deep_verdict,
    ensure_parent,
    parse_all,
    resolve_syzygy,
    select,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default="scratchpad/harvest_fp/all_severe.txt")
    ap.add_argument("--stockfish", default=_DEFAULT_SF)
    ap.add_argument("--nodes-list", default="4000000,8000000,16000000")
    ap.add_argument("--hash-mb", type=int, default=2048)
    ap.add_argument("--control-n", type=int, default=24)
    ap.add_argument("--deep-lost", type=float, default=-0.5)
    ap.add_argument("--deep-fine", type=float, default=-0.2)
    ap.add_argument("--syzygy-path", default="data/syzygy_3-4-5-6")
    ap.add_argument("--out", default="scratchpad/harvest_fp/deepsf_scaling.jsonl")
    ap.add_argument("--nice", type=int, default=15)
    args = ap.parse_args()
    ensure_parent(args.out)
    syzygy = resolve_syzygy(args.syzygy_path)  # fail-fast before the long search
    budgets = [int(x) for x in args.nodes_list.split(",") if x.strip()]

    rows = parse_all(args.all)
    sel = select(rows, args.control_n)
    print(f"[scaling] rechecking {len(sel)} seeds at budgets {budgets} (syzygy={syzygy})")

    def make_engine() -> StockfishUCI:
        return StockfishUCI(args.stockfish, nodes=budgets[-1], hash_mb=int(args.hash_mb),
                            nice=int(args.nice), syzygy_path=syzygy)

    eng = make_engine()
    results: list[dict] = []
    t0 = time.time()
    try:
        for i, r in enumerate(sel):
            try:
                fen = seed_board_from_line(r["line"]).fen()
                sqs = {}
                for n in budgets:
                    eng.new_game()  # cold TT per budget — independent convergence check (Codex #125)
                    res = eng.search(fen, nodes=int(n))
                    sqs[n] = None if res.wdl is None else round(float(res.wdl[0] - res.wdl[2]), 3)
                r2 = {**r, "fen": fen, "deep_sq": sqs,
                      "verdict": {n: (deep_verdict(sqs[n], args.deep_lost, args.deep_fine)
                                      if sqs[n] is not None else "UNKNOWN") for n in budgets}}
            except Exception as e:  # one bad seed must not kill the sweep
                # A raised/timed-out search leaves SF calculating -> desyncs later
                # searches; recreate the engine (Codex #125).
                r2 = {**r, "fen": "", "deep_sq": {}, "verdict": {}, "err": type(e).__name__}
                try:
                    eng.close()
                except Exception:
                    pass
                eng = make_engine()
            results.append(r2)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(sel)}  ({time.time()-t0:.0f}s)")
    finally:
        eng.close()

    with open(args.out, "w", encoding="utf-8") as fh:
        for r2 in results:
            fh.write(json.dumps(r2) + "\n")

    graded = [r for r in results if r.get("verdict") and all(v != "UNKNOWN" for v in r["verdict"].values())]
    print(f"\n=== per-budget verdict split (n={len(graded)}) ===")
    print(f"  {'nodes':>10}  {'LOST':>5} {'MID':>4} {'FINE':>5}")
    for n in budgets:
        lost = sum(1 for r in graded if r["verdict"][n] == "LOST")
        mid = sum(1 for r in graded if r["verdict"][n] == "MID")
        fine = sum(1 for r in graded if r["verdict"][n] == "FINE")
        print(f"  {n:>10,}  {lost:>5} {mid:>4} {fine:>5}")

    # Stability: does the verdict move as depth grows?
    stable = sum(1 for r in graded if len(set(r["verdict"].values())) == 1)
    hi = budgets[-1]
    agree_hi = sum(1 for r in graded if r["verdict"][hi] == r["verdict"][budgets[-2]])
    flips = [r for r in graded if len(set(r["verdict"].values())) > 1]
    print(f"\n  verdict identical across ALL budgets : {stable}/{len(graded)} "
          f"= {100*stable/len(graded):.0f}%")
    print(f"  agree {budgets[-2]:,} vs {hi:,}          : {agree_hi}/{len(graded)} "
          f"= {100*agree_hi/len(graded):.0f}%  (converged if high)")
    if flips:
        print(f"\n  {len(flips)} seed(s) FLIP verdict with depth (SF-unstable):")
        for r in flips[:12]:
            seq = " ".join(f"{n//1_000_000}M:{r['verdict'][n][0]}({r['deep_sq'][n]:+.2f})"
                           for n in budgets)
            print(f"    {r['bucket'][:6]:6s} sq={r['sq']:+.2f} {r['fen'][:34]}  {seq}")


if __name__ == "__main__":
    main()
