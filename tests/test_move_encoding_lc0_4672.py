from __future__ import annotations

import inspect

import chess
import numpy as np

from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    FULL_TO_COMPACT_POLICY,
    POLICY_SIZE,
    index_to_move,
    index_to_move_for_encoding,
    legal_move_mask_for_encoding,
    move_to_index_for_encoding,
    legal_move_mask,
    move_to_index,
    policy_batch_to_encoding,
    policy_vector_to_encoding,
)


def test_policy_size_is_4672():
    assert POLICY_SIZE == 64 * 73
    assert COMPACT_POLICY_SIZE == 1858
    assert COMPACT_TO_FULL_POLICY.shape == (COMPACT_POLICY_SIZE,)
    assert int((FULL_TO_COMPACT_POLICY >= 0).sum()) == COMPACT_POLICY_SIZE


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


def test_compact_roundtrip_for_legal_moves_random_walk():
    rng = np.random.default_rng(1)
    b = chess.Board()

    for _ in range(60):
        mask = legal_move_mask_for_encoding(b, policy_encoding="lc0_1858")
        assert mask.shape == (COMPACT_POLICY_SIZE,)

        for m in list(b.legal_moves):
            idx = move_to_index_for_encoding(m, b, policy_encoding="lc0_1858")
            assert 0 <= idx < COMPACT_POLICY_SIZE
            assert bool(mask[int(idx)])
            m2 = index_to_move_for_encoding(int(idx), b, policy_encoding="lc0_1858")
            assert m2 == m

        if b.is_game_over():
            break

        moves = list(b.legal_moves)
        b.push(moves[int(rng.integers(0, len(moves)))])


def test_full_to_compact_policy_conversion_renormalizes():
    p = np.zeros((POLICY_SIZE,), dtype=np.float32)
    p[int(COMPACT_TO_FULL_POLICY[0])] = 0.25
    p[int(np.flatnonzero(FULL_TO_COMPACT_POLICY < 0)[0])] = 0.75

    compact = policy_vector_to_encoding(p, policy_encoding="lc0_1858")

    assert compact.shape == (COMPACT_POLICY_SIZE,)
    assert np.isclose(float(compact.sum(dtype=np.float32)), 1.0)
    assert np.isclose(float(compact[0]), 1.0)


def test_full_to_compact_policy_batch_conversion_renormalizes_rows():
    p = np.zeros((2, POLICY_SIZE), dtype=np.float32)
    p[0, int(COMPACT_TO_FULL_POLICY[0])] = 0.25
    p[0, int(np.flatnonzero(FULL_TO_COMPACT_POLICY < 0)[0])] = 0.75
    p[1, int(COMPACT_TO_FULL_POLICY[1])] = 0.5

    compact = policy_batch_to_encoding(p, policy_encoding="lc0_1858")

    assert compact.shape == (2, COMPACT_POLICY_SIZE)
    np.testing.assert_allclose(compact.sum(axis=1), np.ones((2,), dtype=np.float32))
