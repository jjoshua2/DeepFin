from __future__ import annotations

import inspect

import chess
import numpy as np

from chess_anti_engine.moves import (
    POLICY_SIZE,
    index_to_move,
    legal_move_mask,
    move_to_index,
)


def test_policy_size_is_4672():
    assert POLICY_SIZE == 64 * 73


def test_move_to_index_signature_includes_board():
    sig = inspect.signature(move_to_index)
    assert list(sig.parameters.keys()) == ["move", "board"]


def test_legal_move_mask_marks_all_legal_moves_startpos():
    b = chess.Board()
    mask = legal_move_mask(b)
    assert mask.shape == (POLICY_SIZE,)

    for m in b.legal_moves:
        idx = move_to_index(m, b)
        assert 0 <= idx < POLICY_SIZE
        assert bool(mask[int(idx)])


def test_roundtrip_index_to_move_for_legal_moves_random_walk():
    rng = np.random.default_rng(0)
    b = chess.Board()

    for _ in range(60):
        # Check roundtrip for all legal moves in this position
        for m in list(b.legal_moves):
            idx = move_to_index(m, b)
            m2 = index_to_move(int(idx), b)
            assert m2 == m

        if b.is_game_over():
            break

        moves = list(b.legal_moves)
        b.push(moves[int(rng.integers(0, len(moves)))])