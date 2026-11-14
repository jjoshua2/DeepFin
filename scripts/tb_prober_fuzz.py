#!/usr/bin/env python3
"""Crash fuzz for the C Syzygy prober (``chess_anti_engine.tb._tb_ext``).

Feeds deterministic random and mutated positions -- impossible en-passant
squares (including a king standing on ep-8/ep+8), stale castling flags, 0-3
kings, odd clocks, corrupted FEN text -- to every entry point: the FEN and
CBoard paths of probe_wdl / probe_dtz / probe_root. A probe may return a value
or None, or raise ValueError; anything else (a crash, an abort, another
exception) is a failure.

Run it in a SUBPROCESS: a segfault must fail the caller with the offending
input, not kill it. Before each input the current FEN is written to stdout and
flushed, so the last ``FEN `` line of a crashed run names the input. With
faulthandler enabled the native stack is printed to stderr as well.

    PYTHONPATH=. python3 -X faulthandler scripts/tb_prober_fuzz.py --n 100000 \\
        --syzygy-path data/syzygy_3-4-5:data/syzygy_6
"""
from __future__ import annotations

import argparse
import random
import sys

import chess

_PIECES = "QRBNP"


def _random_board(rng: random.Random) -> chess.Board:
    board = chess.Board(None)
    free = list(range(64))
    rng.shuffle(free)
    n_kings = rng.choices([2, 1, 0, 3], weights=[90, 4, 3, 3])[0]
    for i in range(n_kings):
        board.set_piece_at(free.pop(), chess.Piece(chess.KING, i % 2 == 0))
    for _ in range(rng.randint(0, 5)):
        letter = rng.choice(_PIECES)
        sq = free.pop()
        board.set_piece_at(sq, chess.Piece(chess.PIECE_SYMBOLS.index(letter.lower()), rng.random() < 0.5))
    board.turn = rng.random() < 0.5
    board.halfmove_clock = rng.choice([0, 0, 1, 50, 98, 99, 100, 101, 150, 255, 1000])
    return board


def _king_on_ep_victim(rng: random.Random) -> chess.Board:
    """Side-to-move pawn beside the opponent KING, ep square behind the king."""
    board = chess.Board(None)
    stm = rng.random() < 0.5
    rank = 4 if stm else 3
    f = rng.randrange(8)
    king = chess.square(f, rank)
    board.set_piece_at(king, chess.Piece(chess.KING, not stm))
    side = [x for x in (f - 1, f + 1) if 0 <= x < 8]
    board.set_piece_at(chess.square(rng.choice(side), rank), chess.Piece(chess.PAWN, stm))
    free = [sq for sq in range(64) if board.piece_at(sq) is None]
    rng.shuffle(free)
    board.set_piece_at(free.pop(), chess.Piece(chess.KING, stm))
    for _ in range(rng.randint(0, 3)):
        board.set_piece_at(free.pop(), chess.Piece(rng.choice([1, 2, 3, 4, 5]), rng.random() < 0.5))
    board.turn = stm
    board.ep_square = king + (8 if stm else -8)
    return board


def _mutate_text(rng: random.Random, fen: str) -> str:
    chars = list(fen)
    for _ in range(rng.randint(1, 3)):
        i = rng.randrange(len(chars))
        chars[i] = rng.choice("pnbrqkPNBRQK12345678/ wb-KQkqabcdefgh0123456789x")
    return "".join(chars)


def inputs(seed: int, n: int):
    rng = random.Random(seed)
    for _ in range(n):
        u = rng.random()
        board = _king_on_ep_victim(rng) if u < 0.25 else _random_board(rng)
        if u >= 0.25 and rng.random() < 0.6:
            board.ep_square = rng.randrange(0, 64) or None
        if rng.random() < 0.2:
            board.castling_rights = rng.choice([chess.BB_CORNERS, chess.BB_H1, chess.BB_A8 | chess.BB_H1, 0])
        fen = board.fen(en_passant="fen")
        if rng.random() < 0.1:
            fen = _mutate_text(rng, fen)
        yield fen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--syzygy-path", default="")
    args = ap.parse_args()

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.tb import _tb_ext

    largest = _tb_ext.init(args.syzygy_path) if args.syzygy_path else 0
    out = sys.stdout
    counts = {"inputs": 0, "value": 0, "none": 0, "value_error": 0, "board_unparsable": 0}

    def call(fn, arg) -> None:
        try:
            r = fn(arg)
        except ValueError:
            counts["value_error"] += 1
            return
        counts["none" if r is None else "value"] += 1

    for fen in inputs(args.seed, args.n):
        counts["inputs"] += 1
        out.write(f"FEN {fen}\n")
        out.flush()
        for fn in (_tb_ext.probe_wdl_fen, _tb_ext.probe_dtz_fen, _tb_ext.probe_root_fen):
            call(fn, fen)
        try:
            board = chess.Board(fen)
        except ValueError:
            counts["board_unparsable"] += 1
            continue
        cb = CBoard.from_board(board)
        for fn in (_tb_ext.probe_wdl, _tb_ext.probe_dtz, _tb_ext.probe_root):
            call(fn, cb)
    out.write(f"DONE largest={largest} {counts}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
