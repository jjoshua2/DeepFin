"""Differential fuzzer: CBoard C extension vs python-chess oracle.

Plays random games in lockstep on a ``chess.Board`` and a ``CBoard`` and
asserts at every ply that the two representations agree on:

  - the legal move set (policy indices)
  - piece bitboards, side to move, halfmove clock
  - terminal flags (checkmate / stalemate / game-over)
  - (opt-in: ``--encode-every N``) the full encoded input planes:
    ``encode_cboard(cb)`` must be bit-identical to ``encode_position(b)``
    for each history encoding — the selfplay/training parity contract.
    Off by default: the C encoder has a known repetition-plane divergence
    from the python-chess oracle (the gated ``history_rep_fix`` candidate),
    so the encode oracle fails on default seeds until that lands.

Run under the sanitized extension build for memory/UB coverage:

    CAE_EXT_SANITIZE=address,undefined python setup.py build_ext --inplace --force
    LD_PRELOAD=$(gcc -print-file-name=libasan.so) ASAN_OPTIONS=detect_leaks=0 \
        PYTHONPATH=. python scripts/fuzz_cboard_diff.py --games 500

Exits non-zero with the FEN + UCI move list needed to reproduce the first
divergence. See scripts/fuzz/run_fuzz.sh for the full fuzzing entry point.
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field

import chess
import numpy as np

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.moves import move_to_index

# Production history modes only. C and Python encoders are verified
# bit-identical for these (the live config uses lc0_root_legacy_meta). The
# deprecated "legacy"/None mode is deliberately excluded: there the C encoder
# zeros the repetition planes (see _cboard_impl.h "103-110: repetitions (all
# 0)") while the Python path populates them — a known, non-active-path
# divergence, not an invariant this oracle should assert.
HISTORY_ENCODINGS: tuple[str | None, ...] = ("lc0_root", "lc0_root_legacy_meta")


@dataclass
class Failure:
    context: str
    detail: str
    moves: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        repro = " ".join(self.moves)
        return f"{self.context}: {self.detail}\n  repro (UCI from startpos): {repro}"


def _check_state(cb: CBoard, b: chess.Board, ctx: str, moves: list[str]) -> Failure | None:
    want = sorted(move_to_index(m, b) for m in b.legal_moves)
    got = sorted(int(i) for i in cb.legal_move_indices())
    if want != got:
        return Failure(ctx, (
            f"legal move set mismatch fen={b.fen()} "
            f"only_py={set(want) - set(got)} only_c={set(got) - set(want)}"
        ), moves)
    py_bb = {
        "pawns": int(b.pawns), "knights": int(b.knights), "bishops": int(b.bishops),
        "rooks": int(b.rooks), "queens": int(b.queens), "kings": int(b.kings),
        "occ_white": int(b.occupied_co[chess.WHITE]),
        "occ_black": int(b.occupied_co[chess.BLACK]),
    }
    for name, val in py_bb.items():
        if int(getattr(cb, name)) != val:
            return Failure(ctx, f"bitboard {name} mismatch fen={b.fen()}", moves)
    if bool(cb.turn) != bool(b.turn):
        return Failure(ctx, f"turn mismatch fen={b.fen()}", moves)
    if int(cb.halfmove_clock) != int(b.halfmove_clock):
        return Failure(ctx, (
            f"halfmove_clock mismatch fen={b.fen()} "
            f"py={b.halfmove_clock} c={cb.halfmove_clock}"
        ), moves)
    if bool(cb.is_checkmate()) != b.is_checkmate() or bool(cb.is_stalemate()) != b.is_stalemate():
        return Failure(ctx, f"terminal flag mismatch fen={b.fen()}", moves)
    return None


def _check_encode(cb: CBoard, b: chess.Board, ctx: str, moves: list[str]) -> Failure | None:
    for hist in HISTORY_ENCODINGS:
        c_planes = encode_cboard(cb, input_history_encoding=hist, input_extra_features="v1")
        py_planes = encode_position(b, input_history_encoding=hist, input_extra_features="v1")
        if c_planes.shape != py_planes.shape:
            return Failure(ctx, (
                f"encode shape mismatch hist={hist!r} "
                f"c={c_planes.shape} py={py_planes.shape}"
            ), moves)
        if not np.array_equal(c_planes, py_planes):
            bad = np.argwhere(c_planes != py_planes)
            plane = int(bad[0][0])
            return Failure(ctx, (
                f"encode planes diverge hist={hist!r} fen={b.fen()} "
                f"first diff plane={plane} ({len(bad)} cells differ)"
            ), moves)
    return None


def run(*, games: int, seed: int, max_plies: int, encode_every: int) -> Failure | None:
    """Play ``games`` random lockstep games; return the first divergence."""
    rng = random.Random(seed)
    for g in range(games):
        b = chess.Board()
        cb = CBoard.from_board(b)
        moves: list[str] = []
        fail = _check_state(cb, b, f"game{g} ply0", moves)
        if fail is None and encode_every:
            fail = _check_encode(cb, b, f"game{g} ply0", moves)
        if fail is not None:
            return fail
        for ply in range(1, rng.randrange(2, max_plies + 1)):
            legal = list(b.legal_moves)
            if not legal:
                if not bool(cb.is_game_over()):
                    return Failure(
                        f"game{g} ply{ply}",
                        f"python game over, C board is not. fen={b.fen()}",
                        moves,
                    )
                break
            m = rng.choice(legal)
            cb.push_index(move_to_index(m, b))
            b.push(m)
            moves.append(m.uci())
            ctx = f"game{g} ply{ply}"
            fail = _check_state(cb, b, ctx, moves)
            if fail is None and encode_every and ply % encode_every == 0:
                fail = _check_encode(cb, b, ctx, moves)
            if fail is not None:
                return fail
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0xDEEFF15)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument(
        "--encode-every", type=int, default=0,
        help=(
            "compare encoded planes every N plies (0 = off, the default: the "
            "encode oracle is opt-in while the known repetition-plane "
            "divergence is unresolved — see the history_rep_fix candidate)"
        ),
    )
    args = parser.parse_args()
    fail = run(
        games=args.games, seed=args.seed,
        max_plies=args.max_plies, encode_every=args.encode_every,
    )
    if fail is not None:
        print(f"DIVERGENCE FOUND\n{fail}", file=sys.stderr)
        return 1
    oracle = (
        f"encode oracle every {args.encode_every} plies"
        if args.encode_every
        else "encode oracle off"
    )
    print(
        f"OK: {args.games} lockstep games (seed={args.seed:#x}, {oracle}) "
        "— no divergence"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
