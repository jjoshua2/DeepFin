"""Deep-SF admission gate for harvested blind-spot seeds (batch / manual form).

The raw severe band is ~70% false positives (deep SF agrees with the net, not the
~700k in-loop label), so it must be gated before feeding. This dedups the banked
severe seeds, runs a static unhandicapped deep Stockfish (4M nodes — converged;
4M=8M=16M in calibration — with the 6-man syzygy TB on), and keeps only the
seeds deep SF still calls LOST. Those are the real, deep-confirmed value blind
spots safe to seed selfplay from.

Outputs:
  --out-list  : vetted seed lines (opening_fen_list_path grammar; comment stripped
                by the loader) — the auto-feed candidate list.
  --out-jsonl : per-seed audit (fen, captured sq/nq, deep_sq, verdict).

This is the same gate an automated loop would run; run it on the backlog now,
review, then feed (training-affecting -> ledger + yardstick + one-change/window).
"""
from __future__ import annotations

import argparse
import glob
import json
import time

from chess_anti_engine.selfplay.opening import seed_board_from_line
from chess_anti_engine.stockfish.uci import StockfishUCI
from scripts.blindspot_continuation import parse_seeds
from scripts.blindspot_deepsf_calibrate import deep_verdict, ensure_parent, resolve_syzygy

_DEFAULT_SF = "/home/josh/projects/chess/e2e_server/publish/stockfish"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--severe-glob", default="data/harvest/blindspot_live.severe.p*.txt")
    ap.add_argument("--stockfish", default=_DEFAULT_SF)
    ap.add_argument("--nodes", type=int, default=4_000_000,
                    help="deep-SF budget (4M = converged per the scaling check)")
    ap.add_argument("--hash-mb", type=int, default=2048)
    ap.add_argument("--syzygy-path", default="data/syzygy_3-4-5-6")
    ap.add_argument("--deep-lost", type=float, default=-0.5)
    ap.add_argument("--deep-fine", type=float, default=-0.2)
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--out-list", default="data/harvest/vetted_blindspots.txt")
    ap.add_argument("--out-jsonl", default="scratchpad/harvest_fp/gate_audit.jsonl")
    args = ap.parse_args()
    ensure_parent(args.out_list)
    ensure_parent(args.out_jsonl)
    syzygy = resolve_syzygy(args.syzygy_path)  # fail-fast before the long search

    seeds = parse_seeds(sorted(glob.glob(args.severe_glob)))
    seen: set[str] = set()
    uniq = []
    for s in seeds:
        if s.line in seen:
            continue
        seen.add(s.line)
        uniq.append(s)
    print(f"[gate] {len(uniq)} unique severe seeds; deep SF nodes={args.nodes:,} "
          f"syzygy={args.syzygy_path}")

    eng = StockfishUCI(args.stockfish, nodes=int(args.nodes), hash_mb=int(args.hash_mb),
                       nice=int(args.nice), syzygy_path=syzygy)
    audit = []
    vetted = []
    t0 = time.time()
    try:
        for i, s in enumerate(uniq):
            try:
                fen = seed_board_from_line(s.line).fen()
                res = eng.search(fen, nodes=int(args.nodes))
                if res.wdl is None:
                    verdict, dsq = "UNKNOWN", None
                else:
                    dsq = round(float(res.wdl[0] - res.wdl[2]), 3)
                    verdict = deep_verdict(dsq, args.deep_lost, args.deep_fine)
            except Exception as e:  # one bad seed must not kill the batch
                fen, verdict, dsq = "", f"ERR:{type(e).__name__}", None
            audit.append({"line": s.line, "fen": fen, "nq": s.nq, "sq": s.sq,
                          "deep_sq": dsq, "deep": verdict, "game": s.game_id})
            if verdict == "LOST":
                vetted.append((s, dsq))
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(uniq)}  kept={len(vetted)}  ({time.time()-t0:.0f}s)")
    finally:
        eng.close()

    with open(args.out_jsonl, "w", encoding="utf-8") as fh:
        for a in audit:
            fh.write(json.dumps(a) + "\n")
    with open(args.out_list, "w", encoding="utf-8") as fh:
        fh.write("# deep-SF-vetted blind spots (deep-LOST @ 4M nodes + 6-man TB)\n")
        for s, dsq in vetted:
            fh.write(f"{s.line}  # deep_sq={dsq} sq={s.sq:.2f} nq={s.nq:.2f} game={s.game_id}\n")

    graded = [a for a in audit if a["deep"] in ("LOST", "MID", "FINE")]
    lost = sum(1 for a in graded if a["deep"] == "LOST")
    fine = sum(1 for a in graded if a["deep"] == "FINE")
    mid = sum(1 for a in graded if a["deep"] == "MID")
    print(f"\n=== gate result ({len(graded)} graded) ===")
    print(f"  deep-LOST (VETTED, kept): {lost}  ({100*lost/len(graded):.0f}%)")
    print(f"  deep-FINE (false pos)   : {fine}  ({100*fine/len(graded):.0f}%)")
    print(f"  deep-MID                : {mid}")
    print(f"  vetted list -> {args.out_list} ({len(vetted)} seeds)")
    print(f"  audit       -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
