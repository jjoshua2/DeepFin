#!/usr/bin/env python3
"""How good is the SF LABEL itself, on the ruler that grades our value head?

We TRAIN the value head on SF at 150k-200k nodes / MultiPV 6, and GRADE it with
``scripts/value_regret.py`` against the frozen audit set at >=1M nodes / MultiPV >=10.
Those are different depths, so part of the head's measured regret is the LABEL's own
error against the ruler -- a floor no amount of fitting can beat.

This measures that floor, and the MultiPV component of it, by running the SAME metric
``value_regret`` uses with the evaluator swapped from our net to Stockfish:

    top1_regret[pos] = move_regrets(pos, legal)[index of the evaluator's chosen move]

For the value head the chosen move is argmax over child WDL. Here it is simply SF's
own ``bestmove`` at the root. Same positions, same filters, same scorer, same phase
split -- so the arms are directly comparable to the head's number.

⚑ BIAS DIRECTION, STATED UP FRONT. ``move_regrets`` gives a move the deep MultiPV never
listed the regret of the WORST listed line, an OPTIMISTIC floor. A strong evaluator
almost always picks a listed move and is scored exactly; a weak one picks unlisted moves
more often and is flattered. So this comparison favours OUR HEAD, and a verdict of "the
label floor is far below the head" is therefore CONSERVATIVE. The unlisted rate is
reported per arm so the size of the effect is visible rather than assumed.

CPU only, one Stockfish thread by default, so it can share the box with a clock match
without starving it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.eval.audit import load_audit_set, move_regrets
from chess_anti_engine.stockfish.uci import StockfishUCI

_PHASE_NAME = {0: "endgame", 1: "middlegame", 2: "opening"}


def _piece_count(fen: str) -> int:
    """Identical to scripts/value_regret.py::_piece_count -- keep the filters aligned."""
    board = fen.split(" ", 1)[0]
    return sum(1 for ch in board if ch.isalpha())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", default="/home/josh/projects/chess/e2e_server/publish/stockfish")
    ap.add_argument("--audit-set", default="data/audit_set_v1.jsonl")
    ap.add_argument("--nodes", type=int, default=200000,
                    help="production label depth: sf_label_nodes_cap")
    ap.add_argument("--multipv", type=int, required=True,
                    help="6 = what production ships; 1 = all nodes into the best line")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash-mb", type=int, default=17, help="matches sf_hash_mb in production")
    ap.add_argument("--max-positions", type=int, default=2000,
                    help="the canonical v1-2k subset, sliced BEFORE the piece filter")
    ap.add_argument("--min-pieces", type=int, default=8,
                    help="value_regret's TB exclusion; 2000 -> 1723")
    ap.add_argument("--nice", type=int, default=19)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    positions = load_audit_set(args.audit_set)
    if args.max_positions > 0:
        positions = positions[: args.max_positions]
    if args.min_pieces > 0:
        n_before = len(positions)
        positions = [p for p in positions if _piece_count(p.fen) >= args.min_pieces]
        print(f"[label-floor] min-pieces>={args.min_pieces}: dropped "
              f"{n_before - len(positions)} TB-range positions ({len(positions)} kept)",
              flush=True)

    tag = f"[nodes={args.nodes} multipv={args.multipv} threads={args.threads}]"
    print(f"[label-floor] {tag} {len(positions)} positions from {args.audit_set}", flush=True)

    eng = StockfishUCI(
        args.stockfish, nodes=args.nodes, multipv=args.multipv,
        hash_mb=args.hash_mb, nice=args.nice,
        # 60s default is fine for 200k nodes single-threaded (~0.2s), but a
        # cold first search plus nice-19 scheduling under a live clock match
        # can spike; give it room rather than poison the engine on one stall.
        read_timeout_s=300.0,
    )
    # Threads is not a StockfishUCI constructor arg -- it hardcodes Threads=1 at
    # handshake. Assert rather than assume, so a future default change cannot silently
    # turn this into a multi-threaded run that steals the clock match's cores.
    if args.threads != 1:
        raise SystemExit("this rig only supports --threads 1 (the wrapper pins Threads=1)")

    top1 = np.full(len(positions), np.nan, dtype=np.float64)
    phases = np.array([p.phase for p in positions], dtype=np.int64)
    unlisted = 0
    scored = 0
    t0 = time.monotonic()

    try:
        for i, pos in enumerate(positions):
            board = chess.Board(pos.fen)
            legal = [m.uci() for m in board.legal_moves]
            if not legal:
                continue  # terminal root -- value_regret leaves these NaN too
            # `fresh=True`: a warm TT from the previous position would let one search
            # steer the next and overstate agreement with the ruler.
            res = eng.search(pos.fen, fresh=True)
            # `bestmove_uci` is typed str, so the only real failure mode is a
            # bestmove that is not legal here (desync / "(none)" on a dead position).
            bm = res.bestmove_uci
            if bm not in legal:
                continue
            regrets = move_regrets(pos, legal)
            top1[i] = float(regrets[legal.index(bm)])
            if pos.move_cp.get(bm) is None:
                unlisted += 1
            scored += 1
            if (i + 1) % 200 == 0:
                el = time.monotonic() - t0
                print(f"[label-floor] {i + 1}/{len(positions)}  "
                      f"running mean {np.nanmean(top1[:i + 1]):.1f} cp  ({el:.0f}s)",
                      flush=True)
    finally:
        eng.close()

    overall = float(np.nanmean(top1))
    print(f"\n=== SF LABEL 1-ply deep-SF regret {tag} ===", flush=True)
    print(f"  {tag} OVERALL {overall:.1f} cp (n={scored})")
    for ph in range(3):
        sel = phases == ph
        if sel.any() and not np.all(np.isnan(top1[sel])):
            print(f"  {tag} {_PHASE_NAME.get(ph, ph):11s} {float(np.nanmean(top1[sel])):.1f} cp")
    frac = (unlisted / scored) if scored else float("nan")
    print(f"  {tag} chose a move the RULER never listed: {unlisted}/{scored} = {frac:.3%}")
    print(f"  {tag} ^ those get the optimistic worst-listed FLOOR, so this arm is "
          f"flattered by exactly that much")
    print(f"  {tag} wall {time.monotonic() - t0:.0f}s")

    if args.dump:
        with open(args.dump, "w") as f:
            for pos, r in zip(positions, top1, strict=True):
                if np.isnan(r):
                    continue
                f.write(json.dumps({"fen": pos.fen, "key": pos.key,
                                    "phase": int(pos.phase), "value": float(r)}) + "\n")
        print(f"  {tag} per-position dump -> {args.dump}")


if __name__ == "__main__":
    main()
