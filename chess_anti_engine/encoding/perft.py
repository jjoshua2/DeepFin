"""Native CBoard perft API and CLI; see docs/perft.md.

Example: python -m chess_anti_engine.encoding.perft 5 --expect 4865609
Python parses the root and formats results; all tree expansion is native C.
"""

from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter

import chess

from ._lc0_ext import CBoard
from ._perft_ext import MAX_DEPTH, SLIDER_BACKEND, perft, perft_divide

__all__ = ["perft", "perft_divide"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count legal move sequences with DeepFin's CBoard core.")
    parser.add_argument("depth", type=int, help=f"remaining plies (0..{MAX_DEPTH}; divide requires >= 1)")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="standard-chess root FEN; defaults to startpos")
    parser.add_argument("--moves", nargs="*", default=[], help="legal UCI moves to apply before counting")
    parser.add_argument("--divide", action="store_true", help="print the count under each root UCI move")
    parser.add_argument("--expect", type=int, help="return exit status 1 if the total differs from this count")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args(argv)
    if not 0 <= args.depth <= MAX_DEPTH or (args.divide and args.depth == 0):
        parser.error(f"depth must be {'1' if args.divide else '0'}..{MAX_DEPTH}")
    if args.expect is not None and args.expect < 0:
        parser.error("--expect must be nonnegative")
    try:
        board = chess.Board(args.fen)
        if not board.is_valid():
            parser.error(f"invalid standard-chess position (status={board.status()})")
        for move in args.moves:
            board.push_uci(move)
    except ValueError as exc:
        parser.error(str(exc))
    native = CBoard.from_board(board)
    started = perf_counter()
    try:
        divide = perft_divide(native, args.depth) if args.divide else None
        nodes = sum(divide.values()) if divide is not None else perft(native, args.depth)
    except (ValueError, OverflowError) as exc:
        parser.error(str(exc))
    elapsed = perf_counter() - started
    nps = nodes / elapsed if elapsed > 0 else 0.0
    matches = args.expect is None or nodes == args.expect
    result: dict[str, object] = {
        "fen": board.fen(en_passant="fen"),
        "depth": args.depth,
        "nodes": nodes,
        "seconds": elapsed,
        "nodes_per_second": nps,
        "slider_backend": SLIDER_BACKEND,
    }
    if divide is not None:
        result["divide"] = dict(sorted(divide.items()))
    if args.expect is not None:
        result["expected"] = args.expect
        result["matches_expected"] = matches
    if args.json:
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        print(f"FEN: {result['fen']}")
        print(f"Depth: {args.depth} | sliders: {SLIDER_BACKEND}")
        if divide is not None:
            for move, count in sorted(divide.items()):
                print(f"{move}: {count}")
        print(f"Nodes: {nodes}")
        print(f"Time: {elapsed:.6f} s | Nodes/s: {nps:.0f}")
    if not matches:
        print(f"Perft mismatch: expected {args.expect}, got {nodes}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
