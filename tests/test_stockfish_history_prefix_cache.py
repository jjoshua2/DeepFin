"""The validated-prefix cache must not change what history validation accepts.

``_validated_position_history`` reuses previously validated (root, moves)
prefixes so that the same game is not replayed with python-chess legality
checks on every Stockfish query. These tests compare it with an uncached
oracle (the pre-cache definition) on growing game histories, including errors
that occur only in the part a cache hit does not cover.
"""
from __future__ import annotations

import random

import chess
import pytest

from chess_anti_engine.stockfish import uci as sf_uci
from chess_anti_engine.stockfish.uci import _validated_position_history


def _oracle(fen: str, root_fen: str, moves: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    board = chess.Board(root_fen)
    requested = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("history root is not a valid standard-chess position")
    root = board.fen(en_passant="fen")
    for token in moves:
        if not sf_uci._UCI_MOVE_RE.fullmatch(token):
            raise ValueError(f"history contains a malformed UCI move: {token!r}")
        move = chess.Move.from_uci(token)
        if not board.is_legal(move):
            raise ValueError(f"history contains an illegal move: {token!r}")
        board.push(move)
    if board.fen() != requested.fen():
        raise ValueError("history does not reproduce the requested final FEN")
    return root, moves


@pytest.fixture(autouse=True)
def _empty_cache():
    sf_uci._prefix_cache.clear()
    yield
    sf_uci._prefix_cache.clear()


def _outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except ValueError as exc:
        return ("err", str(exc))


@pytest.mark.parametrize("seed", range(6))
def test_growing_game_histories_match_the_uncached_oracle(seed: int) -> None:
    rng = random.Random(seed)
    root = chess.STARTING_FEN if seed % 2 == 0 else "r3k2r/pp3ppp/8/8/8/8/PP3PPP/R3K2R w KQkq - 0 1"
    board = chess.Board(root)
    moves: list[str] = []
    for _ply in range(80):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(rng.choice(legal))
        moves.append(board.move_stack[-1].uci())
        snap = tuple(moves)
        fen = board.fen()
        # Twice per ply, like pool.submit followed by StockfishUCI.search.
        for _ in range(2):
            got = _outcome(lambda f, r, m: _validated_position_history(f, root_fen=r, root_moves=m),
                           fen, root, snap)
            assert got == _outcome(_oracle, fen, root, snap)
        # Errors past a cached prefix: illegal and malformed appended tokens,
        # and a final FEN the history does not reach.
        illegal = next((m.uci() for m in chess.Board(root).legal_moves), "a2a3")
        bad_cases = [
            (fen, (*snap, "a1a1")),
            (fen, (*snap, "zz")),
            (chess.STARTING_FEN if fen != chess.STARTING_FEN else root, snap),
        ]
        if board.piece_at(chess.parse_square(illegal[:2])) is None:
            bad_cases.append((fen, (*snap, illegal)))
        for case_fen, case_moves in bad_cases:
            got = _outcome(lambda f, r, m: _validated_position_history(f, root_fen=r, root_moves=m),
                           case_fen, root, case_moves)
            assert got == _outcome(_oracle, case_fen, root, case_moves)
            assert got[0] == "err"


def test_cache_hit_still_checks_the_moves_it_does_not_cover() -> None:
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3"]
    for m in moves:
        board.push_uci(m)
    root = chess.STARTING_FEN
    first = _validated_position_history(board.fen(), root_fen=root, root_moves=moves)
    assert first is not None
    assert first[1] == tuple(moves)
    # e7e5 is no longer available to Black: illegal right after the cached prefix.
    after = board.copy()
    with pytest.raises(ValueError, match="illegal move"):
        _validated_position_history(after.fen(), root_fen=root, root_moves=[*moves, "e7e5"])
    # The exact history is a hit and returns the identical value.
    assert _validated_position_history(board.fen(), root_fen=root, root_moves=tuple(moves)) == (
        chess.Board(root).fen(en_passant="fen"), tuple(moves))


def test_cache_is_bounded() -> None:
    board = chess.Board()
    root = chess.STARTING_FEN
    rng = random.Random(7)
    moves: list[str] = []
    for _ in range(sf_uci._PREFIX_CACHE_MAX + 50):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over():
            board = chess.Board()
            moves = []
            legal = list(board.legal_moves)
        board.push(rng.choice(legal))
        moves.append(board.move_stack[-1].uci())
        _validated_position_history(board.fen(), root_fen=root, root_moves=tuple(moves))
    assert len(sf_uci._prefix_cache) <= sf_uci._PREFIX_CACHE_MAX
