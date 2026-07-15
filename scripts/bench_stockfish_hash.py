#!/usr/bin/env python3
"""Benchmark Stockfish hash sizes on deterministic production-shaped searches."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import chess

from chess_anti_engine.stockfish import StockfishUCI


def _positions(*, count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    positions: list[str] = []
    while len(positions) < count:
        board = chess.Board()
        target_ply = rng.randint(12, 70)
        for _ in range(target_ply):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over() and board.fullmove_number >= 7:
            positions.append(board.fen())
    return positions


def _result_key(result: object) -> str:
    bestmove = str(getattr(result, "bestmove_uci"))
    cp = getattr(result, "cp")
    mate = getattr(result, "mate")
    depth = int(getattr(result, "depth"))
    return f"{bestmove}|{cp}|{mate}|{depth}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=700_000)
    parser.add_argument("--multipv", type=int, default=40)
    parser.add_argument("--hash-mb", default="8,16,32,64,128")
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    hashes = [int(value) for value in str(args.hash_mb).split(",")]
    fens = _positions(count=int(args.positions), seed=int(args.seed))
    timings: dict[int, list[float]] = {value: [] for value in hashes}
    keys: dict[int, list[list[str]]] = {value: [] for value in hashes}

    for round_idx in range(int(args.rounds)):
        order = hashes[round_idx % len(hashes):] + hashes[:round_idx % len(hashes)]
        for hash_mb in order:
            engine = StockfishUCI(
                str(args.stockfish), nodes=int(args.nodes), multipv=int(args.multipv),
                hash_mb=hash_mb, nice=19,
            )
            try:
                started = time.perf_counter()
                round_keys = [
                    _result_key(engine.search(fen, nodes=int(args.nodes))) for fen in fens
                ]
                timings[hash_mb].append(time.perf_counter() - started)
                keys[hash_mb].append(round_keys)
            finally:
                engine.close()

    baseline = statistics.median(timings[16])
    baseline_moves = [key.split("|", 1)[0] for key in keys[16][0]]
    summary: dict[str, object] = {}
    for hash_mb in hashes:
        median_s = statistics.median(timings[hash_mb])
        moves = [key.split("|", 1)[0] for key in keys[hash_mb][0]]
        agreement = sum(a == b for a, b in zip(moves, baseline_moves, strict=True)) / len(moves)
        digest = hashlib.sha256("\n".join(keys[hash_mb][0]).encode()).hexdigest()[:16]
        summary[str(hash_mb)] = {
            "times_s": timings[hash_mb],
            "median_s": median_s,
            "ratio_vs_16": median_s / baseline,
            "bestmove_agreement_vs_16": agreement,
            "first_round_result_hash": digest,
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
