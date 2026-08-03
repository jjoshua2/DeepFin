"""Differential fuzzer: CBoard C extension vs python-chess oracle.

Plays random games in lockstep on a ``chess.Board`` and a ``CBoard`` and
asserts at every ply that the two representations agree on:

  - the legal move set (policy indices)
  - piece bitboards, side to move, halfmove clock
  - terminal flags (checkmate / stalemate / game-over)
  - (``--encode-every N``, ON by default) the full encoded input planes:
    ``encode_cboard(cb)`` must be bit-identical to ``encode_position(b)``
    for each history encoding — the selfplay/training parity contract.

The encode oracle runs in the PRODUCTION regime by default: ``v2_threats``
(175 planes) with ``history_rep_fix`` on, matching ``configs/pbt2_small.yaml``.
It used to default to off, hardcoded to ``v1``/146 planes and never applying
``history_rep_fix``, on the strength of a docstring claiming it "fails on
default seeds until [history_rep_fix] lands". That landed 2026-06-17 and the
gate was never re-enabled, so the only regime carrying production traffic was
the one regime nothing checked (encoding audit E1). It passes: 1,654 real-game
positions and 60 fuzz games, 0 divergences at 175 planes with the fix on. It
still FAILS with ``--no-history-rep-fix`` (plane 90, a slot-6 repetition
plane), which is the pre-fix behaviour this oracle is meant to catch.

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

from chess_anti_engine.encoding import encode_position, input_plane_count, rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.moves import move_to_index

# Production encoding, from configs/pbt2_small.yaml (:71 / :197). The oracle
# defaults here so the checked regime is the shipped one; both are overridable.
PROD_EXTRA_FEATURES = "v2_threats"
PROD_HISTORY_REP_FIX = True
# Plies between encode comparisons when the oracle is on. 4 keeps the encode
# cost a small fraction of the per-ply state check while still sampling every
# history slot as games grow past 8 plies.
DEFAULT_ENCODE_EVERY = 4

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
    # Full game-over parity, not just mate/stalemate: the C board treats
    # reached claimable draws (50-move counter at 100, third repetition on the
    # board, insufficient material) as terminal for search/selfplay. NOT
    # b.is_game_over(claim_draw=True): that includes python's claim-by-move
    # lookahead (a legal move that WOULD create the third repetition), which
    # the C terminal deliberately lacks — only realized state counts.
    c_over = bool(cb.is_game_over())
    py_over = b.is_game_over() or b.is_repetition(3) or b.is_fifty_moves()
    if c_over != py_over:
        return Failure(ctx, (
            f"is_game_over mismatch fen={b.fen()} c={c_over} py={py_over} "
            f"(py = is_game_over or is_repetition(3) or is_fifty_moves)"
        ), moves)
    return None


def _check_encode(
    cb: CBoard, b: chess.Board, ctx: str, moves: list[str],
    *, input_extra_features: str, expect_planes: int,
) -> Failure | None:
    """Compare C vs Python planes, and pin WHICH planes were compared.

    ``expect_planes`` is computed by the caller from the same
    ``input_extra_features`` and is asserted against the realized arrays. Without
    it the feature version is a value that can be accepted and then ignored: a
    single line pinning ``input_extra_features`` to ``"v1"`` anywhere in this
    function silently drops the 29 v2_threats planes out of the only C-vs-Python
    differential in the repo while every test stays green and the CLI still
    prints ``v=v2_threats`` (PR #321 review F1). That is audit E1 verbatim, and
    it survived six weeks the first time.
    """
    for hist in HISTORY_ENCODINGS:
        c_planes = encode_cboard(
            cb, input_history_encoding=hist,
            input_extra_features=input_extra_features,
        )
        py_planes = encode_position(
            b, input_history_encoding=hist,
            input_extra_features=input_extra_features,
        )
        for side, planes in (("C", c_planes), ("python", py_planes)):
            if int(planes.shape[0]) != int(expect_planes):
                return Failure(ctx, (
                    f"oracle compared {int(planes.shape[0])} {side} planes, not "
                    f"{int(expect_planes)}: the requested feature version "
                    f"({input_extra_features!r}) did not reach the encoder"
                ), moves)
        if c_planes.shape != py_planes.shape:
            return Failure(ctx, (
                f"encode shape mismatch hist={hist!r} v={input_extra_features} "
                f"c={c_planes.shape} py={py_planes.shape}"
            ), moves)
        if not np.array_equal(c_planes, py_planes):
            bad = np.argwhere(c_planes != py_planes)
            plane = int(bad[0][0])
            return Failure(ctx, (
                f"encode planes diverge hist={hist!r} v={input_extra_features} "
                f"fen={b.fen()} first diff plane={plane} "
                f"({len(bad)} cells differ)"
            ), moves)
    return None


def run(
    *, games: int, seed: int, max_plies: int, encode_every: int,
    input_extra_features: str = PROD_EXTRA_FEATURES,
    history_rep_fix: bool = PROD_HISTORY_REP_FIX,
) -> Failure | None:
    """Play ``games`` random lockstep games; return the first divergence.

    ``history_rep_fix`` is process-global in the C encoders, so it is applied
    here and RESTORED on the way out — in-process callers (the pytest gate) must
    not inherit it.
    """
    prior = rep_fix.current()
    rep_fix.apply(bool(history_rep_fix), boards_discarded=True)
    try:
        return _run(
            games=games, seed=seed, max_plies=max_plies,
            encode_every=encode_every,
            input_extra_features=input_extra_features,
        )
    finally:
        if prior is not None:
            rep_fix.apply(prior, boards_discarded=True)


def _run(
    *, games: int, seed: int, max_plies: int, encode_every: int,
    input_extra_features: str,
) -> Failure | None:
    # Computed ONCE, here, far from the encoder calls: _check_encode asserts the
    # arrays it actually compared are this wide (review F1).
    expect_planes = int(input_plane_count(input_extra_features))
    rng = random.Random(seed)
    for g in range(games):
        b = chess.Board()
        cb = CBoard.from_board(b)
        moves: list[str] = []
        fail = _check_state(cb, b, f"game{g} ply0", moves)
        if fail is None and encode_every:
            fail = _check_encode(
                cb, b, f"game{g} ply0", moves,
                input_extra_features=input_extra_features,
                expect_planes=expect_planes,
            )
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
                fail = _check_encode(
                    cb, b, ctx, moves,
                    input_extra_features=input_extra_features,
                    expect_planes=expect_planes,
                )
            if fail is not None:
                return fail
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI, as a function so tests can assert on its realized defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0xDEEFF15)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument(
        "--encode-every", type=int, default=DEFAULT_ENCODE_EVERY,
        help=(
            "compare encoded planes every N plies (0 = off). Default "
            f"{DEFAULT_ENCODE_EVERY}: the encode oracle is the only C-vs-Python "
            "plane differential in the repo and passes in the production regime"
        ),
    )
    parser.add_argument(
        "--extra-features", type=str, default=PROD_EXTRA_FEATURES,
        help=(
            "input_extra_features for the encode oracle "
            f"(default {PROD_EXTRA_FEATURES} = 175 planes, production)"
        ),
    )
    parser.add_argument(
        "--history-rep-fix", dest="history_rep_fix",
        action=argparse.BooleanOptionalAction, default=PROD_HISTORY_REP_FIX,
        help=(
            "apply the process-global repetition-plane fix before playing "
            f"(default {PROD_HISTORY_REP_FIX}, production). --no-history-rep-fix "
            "reproduces the pre-fix divergence the oracle is meant to catch"
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    fail = run(
        games=args.games, seed=args.seed,
        max_plies=args.max_plies, encode_every=args.encode_every,
        input_extra_features=str(args.extra_features),
        history_rep_fix=bool(args.history_rep_fix),
    )
    if fail is not None:
        print(f"DIVERGENCE FOUND\n{fail}", file=sys.stderr)
        return 1
    oracle = (
        f"encode oracle every {args.encode_every} plies "
        f"(v={args.extra_features}, history_rep_fix={bool(args.history_rep_fix)})"
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
