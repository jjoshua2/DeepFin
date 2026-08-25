#!/usr/bin/env python3
"""Benchmark incremental NNUE state propagation against the full-refresh oracle.

This measures the evaluator change inside the same qsearch, not two different
searches.  ``nnue-qsearch`` is the production incremental provider;
``nnue-qsearch-refresh`` is the retained full-refresh implementation.  Exact
value equality and identical search-work counters are required before a speed
ratio is printed.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import chess

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext


def sample_boards(*, games: int, every: int, seed: int) -> list[CBoard]:
    rng = random.Random(seed)
    out: list[CBoard] = []
    for _ in range(games):
        board = chess.Board()
        ply = 0
        while not board.is_game_over(claim_draw=False) and ply < 180:
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            ply += 1
            if ply % every == 0:
                out.append(CBoard.from_board(board))
    if not out:
        raise RuntimeError("sampler produced no positions")
    return out


def run(handle: object, boards: list[CBoard], repeats: int) -> tuple[list[int], float, dict[str, int]]:
    values: list[int] = []
    start = time.perf_counter()
    for _ in range(repeats):
        values = _nnue_ext.arm_handle_eval(handle, boards)
    elapsed = time.perf_counter() - start
    return values, elapsed, _nnue_ext.arm_stats(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--every", type=int, default=4, help="sample every N plies")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resolver-depth", type=int, default=12)
    parser.add_argument("--qply", type=int, default=4)
    parser.add_argument("--check-plies", type=int, default=0)
    args = parser.parse_args()

    if args.games < 1 or args.every < 1 or args.repeats < 1:
        parser.error("--games, --every and --repeats must be >= 1")

    boards = sample_boards(games=args.games, every=args.every, seed=args.seed)
    _nnue_ext.set_arm_config(args.resolver_depth, args.qply, args.check_plies)

    inc = _nnue_ext.arm_open("nnue-qsearch", str(args.pack))
    refresh = _nnue_ext.arm_open("nnue-qsearch-refresh", str(args.pack))

    # Populate mmap/page/cache state before either timed region.
    _nnue_ext.arm_handle_eval(inc, boards[: min(16, len(boards))])
    _nnue_ext.arm_handle_eval(refresh, boards[: min(16, len(boards))])

    inc_values, inc_s, inc_stats = run(inc, boards, args.repeats)
    ref_values, ref_s, ref_stats = run(refresh, boards, args.repeats)

    if inc_values != ref_values:
        mismatch = next(i for i, (a, b) in enumerate(zip(inc_values, ref_values, strict=True)) if a != b)
        raise SystemExit(
            f"VALUE MISMATCH at row {mismatch}: incremental={inc_values[mismatch]} "
            f"refresh={ref_values[mismatch]}"
        )

    search_keys = (
        "calls",
        "calls_in_check",
        "nodes",
        "resolved_leaves",
        "terminal_mate",
        "terminal_draw",
        "depth_cutoffs",
        "max_depth_seen",
        "qnodes",
        "qterminal_draw",
        "qply_cutoffs",
        "qmax_ply_seen",
    )
    diffs = {k: (inc_stats[k], ref_stats[k]) for k in search_keys if inc_stats[k] != ref_stats[k]}
    if diffs:
        raise SystemExit(f"SEARCH-WORK MISMATCH: {diffs}")

    evals = len(boards) * args.repeats
    speedup = ref_s / inc_s
    print(
        f"positions={len(boards)} repeats={args.repeats} qply={args.qply} "
        f"check_plies={args.check_plies}"
    )
    print(f"incremental: {evals / inc_s:,.0f} roots/s  wall={inc_s:.3f}s")
    print(f"refresh:     {evals / ref_s:,.0f} roots/s  wall={ref_s:.3f}s")
    print(f"speedup:     {speedup:.3f}x")
    print(f"qnodes:      {inc_stats['qnodes']:,}  values: EXACT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
