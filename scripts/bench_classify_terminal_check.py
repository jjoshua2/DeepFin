#!/usr/bin/env python3
"""Benchmark redundant terminal rechecks in native selfplay classification."""
from __future__ import annotations

import argparse
import statistics
import time

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import classify_games


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=384)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    boards = [CBoard.from_board(chess.Board()) for _ in range(args.games)]
    net_color = np.arange(args.games, dtype=np.int8) & 1
    done = np.zeros(args.games, dtype=np.int8)
    finalized = np.zeros(args.games, dtype=np.int8)
    selfplay = np.arange(args.games, dtype=np.int8) & 1
    starting_ply = np.zeros(args.games, dtype=np.int32)

    def run(*, check_terminal: bool) -> int:
        checksum = 0
        for _ in range(args.iterations):
            groups = classify_games(
                boards,
                net_color,
                done,
                finalized,
                selfplay,
                starting_ply,
                450,
                check_terminal,
            )
            checksum += sum(item.size for item in groups)
        return checksum

    timings: dict[str, list[float]] = {"checked": [], "authoritative_done": []}
    checksums: dict[str, int] = {}
    for round_index in range(args.rounds):
        arms = (("checked", True), ("authoritative_done", False))
        for name, check_terminal in (arms if round_index % 2 == 0 else reversed(arms)):
            start = time.perf_counter()
            checksums[name] = run(check_terminal=check_terminal)
            timings[name].append(time.perf_counter() - start)
    if checksums["checked"] != checksums["authoritative_done"]:
        raise AssertionError(f"classification checksum mismatch: {checksums}")
    checked_s = statistics.median(timings["checked"])
    authoritative_s = statistics.median(timings["authoritative_done"])
    print(f"checked_s={checked_s:.6f}")
    print(f"authoritative_done_s={authoritative_s:.6f}")
    print(f"speedup={checked_s / authoritative_s:.6f}x")
    print(f"checksum={checksums['checked']}")


if __name__ == "__main__":
    main()
