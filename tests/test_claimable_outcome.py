"""``outcome_claim_draw`` must equal ``board.outcome(claim_draw=True)`` everywhere.

The helper skips python-chess's legal-move sweep when no position repeats in
the reversible window. These tests compare it with the library on every
position of many games biased towards repetitions (knight shuffles), long
reversible stretches (fifty-move claims) and the claim boundaries, and check
that the board is left exactly as it was.
"""
from __future__ import annotations

import random

import chess
import pytest

from chess_anti_engine.utils.claimable_outcome import (
    _reversible_window_repeats,
    is_game_over_claim_draw,
    outcome_claim_draw,
    result_claim_draw,
)


def _assert_same(board: chess.Board) -> None:
    before_fen = board.fen()
    before_stack = list(board.move_stack)
    want = board.outcome(claim_draw=True)
    got = outcome_claim_draw(board)
    assert got == want, (board.fen(), [m.uci() for m in board.move_stack], got, want)
    assert is_game_over_claim_draw(board) == board.is_game_over(claim_draw=True)
    assert result_claim_draw(board) == board.result(claim_draw=True)
    assert board.fen() == before_fen
    assert board.move_stack == before_stack


def _shuffle_biased_game(rng: random.Random, plies: int, board: chess.Board) -> None:
    for _ in range(plies):
        _assert_same(board)
        if board.is_game_over(claim_draw=False):
            return
        moves = list(board.legal_moves)
        quiet = [m for m in moves if not board.is_zeroing(m)]
        if board.move_stack and rng.random() < 0.55:
            # Undo-like reply: move a piece back where it came from when legal.
            last = board.move_stack[-2] if len(board.move_stack) >= 2 else None
            back = [m for m in quiet if last is not None
                    and m.from_square == last.to_square and m.to_square == last.from_square]
            if back:
                board.push(back[0])
                continue
        pool = quiet if quiet and rng.random() < 0.9 else moves
        board.push(rng.choice(pool))
    _assert_same(board)


@pytest.mark.parametrize("seed", range(40))
def test_matches_library_on_repetition_heavy_games(seed: int) -> None:
    rng = random.Random(seed)
    _shuffle_biased_game(rng, 220, chess.Board())


@pytest.mark.parametrize("seed", range(10))
def test_matches_library_from_reversible_endgames(seed: int) -> None:
    # Bare-ish endgames reach the fifty-move claim boundary (clock 99/100).
    rng = random.Random(1000 + seed)
    for fen in ("8/8/4k3/8/8/2R5/4K3/8 w - - 90 1", "8/3k4/8/8/2N5/8/3KB3/8 w - - 0 1",
                "r3k3/8/8/8/8/8/8/4K2R w Kq - 0 1"):
        _shuffle_biased_game(rng, 140, chess.Board(fen))


def test_threefold_claim_and_next_move_claim_both_detected() -> None:
    board = chess.Board()
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in cycle + cycle[:3]:
        board.push_uci(uci)
    # The next move (f6g8) would make the start position's third occurrence.
    assert board.can_claim_threefold_repetition()
    assert outcome_claim_draw(board) == chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    board.push_uci(cycle[3])
    _assert_same(board)


def test_fifty_move_claim_by_next_move_without_any_repeat() -> None:
    # Clock 99 with no history: no repeat can exist, so the fast branch must
    # still find the claim reachable by one quiet move.
    board = chess.Board("8/8/4k3/8/8/2R5/4K3/8 w - - 99 80")
    assert not _reversible_window_repeats(board)
    assert board.outcome(claim_draw=True) == chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    _assert_same(board)
    board = chess.Board("8/8/4k3/8/8/2R5/4K3/8 w - - 98 80")
    _assert_same(board)


def test_window_stops_at_zeroing_move_not_earlier_irreversible_one() -> None:
    # Castling-right loss is irreversible but not zeroing: the scan continues
    # past it (wider than python-chess's window), which may only add repeats.
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    for uci in ["a1b1", "a8b8", "b1a1", "b8a8", "a1b1", "a8b8"]:
        board.push_uci(uci)
        _assert_same(board)
    assert not _reversible_window_repeats(chess.Board())
