#!/usr/bin/env python3
"""Score an external UCI engine's moves against the frozen deep-SF audit labels.

Drives e.g. Cheese over the same audit positions our net is scored on, at a grid of
node budgets, and reports its mean deep-SF regret (cp). Put next to our net's
regret-vs-sims curve (from backtest_time_value.py) it answers, with no games and no
SPRT: how many of OUR sims it takes to match Cheese-at-N-nodes' move quality.

Uses the SAME stratified position subset as backtest_time_value (import _stratified)
so the regret numbers are directly comparable.

Usage:
  PYTHONPATH=. python3 scripts/audit_external_engine.py \\
    --engine "bash -c 'cd /home/josh/local_engines/cheese && exec ./cheese-321-linux-pext'" \\
    --nodes 1000000,5000000 --max-positions 250
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import chess
import chess.engine
import numpy as np

from chess_anti_engine.eval.audit import legal_full_indices, load_audit_set, move_regrets
from scripts.backtest_time_value import _stratified


def _move_at(snaps: list[tuple[int, str]], cp: int) -> str | None:
    """The engine's current bestmove after <= ``cp`` nodes of one search (the last
    iterative-deepening update at or below the checkpoint; the first update if the
    search blew past cp before reporting)."""
    mv = None
    for n, u in snaps:
        if n <= cp:
            mv = u
        else:
            break
    if mv is None and snaps:
        return snaps[0][1]
    return mv


def main() -> None:
    ap = argparse.ArgumentParser(prog="audit_external_engine")
    ap.add_argument("--engine", required=True, help="UCI engine command (shell string)")
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--nodes", default="50000,100000,250000,500000,1000000,2000000,5000000",
                    help="node checkpoints. ONE search to the max is run per position and its "
                         "bestmove is snapshotted at each checkpoint (iterative-deepening "
                         "scaling curve from a single search — no re-searching).")
    ap.add_argument("--max-positions", type=int, default=250)
    ap.add_argument("--out", type=Path, default=Path("runs/backtest/cheese_audit.jsonl"),
                    help="per-(position,checkpoint) rows, written incrementally (partial-safe)")
    args = ap.parse_args()

    positions = _stratified(load_audit_set(args.audit_set), int(args.max_positions))
    boards = [chess.Board(p.fen) for p in positions]
    legal = [legal_full_indices(b)[0] for b in boards]
    regrets = [move_regrets(p, lu) for p, lu in zip(positions, legal, strict=True)]
    checkpoints = sorted({int(n) for n in str(args.nodes).split(",") if n.strip()})
    max_nodes = checkpoints[-1]
    cmd = shlex.split(args.engine)
    print(f"[ext] {len(positions)} positions, engine={cmd[0]}, one search to {max_nodes} "
          f"nodes/pos, snapshot at {checkpoints}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_cp: dict[int, list[float]] = {c: [] for c in checkpoints}
    eng = chess.engine.SimpleEngine.popen_uci(cmd)
    n_done = 0
    with args.out.open("w") as fh:
        try:
            for board, lu, rg, pos in zip(boards, legal, regrets, positions, strict=True):
                snaps: list[tuple[int, str]] = []
                with eng.analysis(board, chess.engine.Limit(nodes=max_nodes)) as an:
                    for info in an:
                        pv = info.get("pv")
                        nodes = info.get("nodes")
                        if pv and nodes is not None:
                            snaps.append((int(nodes), pv[0].uci()))
                for cp in checkpoints:
                    mv = _move_at(snaps, cp)
                    if mv is None:
                        continue
                    regret = float(rg[lu.index(mv)]) if mv in lu else float(rg.max())
                    per_cp[cp].append(regret)
                    fh.write(json.dumps({
                        "key": pos.key, "phase": pos.phase, "source": pos.source,
                        "piece_count": chess.popcount(board.occupied),
                        "nodes": cp, "move": mv, "regret_cp": regret,
                        "listed": mv in pos.move_cp,
                    }) + "\n")
                n_done += 1
                if n_done % 25 == 0:
                    print(f"[ext] {n_done}/{len(positions)}", flush=True)
        finally:
            eng.quit()

    print(f"\n{'nodes':>9}  {'mean_regret_cp':>14}  {'median_cp':>10}  {'n':>4}")
    print("-" * 44)
    for cp in checkpoints:
        a = np.asarray(per_cp[cp])
        if a.size:
            print(f"{cp:>9}  {a.mean():>14.1f}  {np.median(a):>10.1f}  {a.size:>4}")
    print(f"\nrows -> {args.out}")


if __name__ == "__main__":
    main()
