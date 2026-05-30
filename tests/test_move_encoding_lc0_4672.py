from __future__ import annotations

import inspect

import chess
import numpy as np
import pytest

from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    FULL_TO_COMPACT_POLICY,
    POLICY_SIZE,
    index_to_move,
    index_to_move_for_encoding,
    legal_move_mask_for_encoding,
    move_to_index_for_encoding,
    normalize_policy_encoding,
    legal_move_mask,
    move_to_index,
    policy_batch_to_encoding,
    policy_batch_to_full,
    policy_batch_to_full_if_needed,
    policy_vector_to_encoding,
    policy_vector_to_full,
)


def test_policy_size_is_4672():
    assert POLICY_SIZE == 64 * 73
    assert COMPACT_POLICY_SIZE == 1858
    assert COMPACT_TO_FULL_POLICY.shape == (COMPACT_POLICY_SIZE,)
    assert int((FULL_TO_COMPACT_POLICY >= 0).sum()) == COMPACT_POLICY_SIZE


def test_lc0_4672_alias_is_legacy_full_policy_encoding():
    assert normalize_policy_encoding("lc0_4672") == "az_4672"


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


def test_compact_to_full_policy_conversion_fills_invalid_slots():
    compact = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float32)
    compact[0] = 3.0

    full = policy_vector_to_full(compact, policy_encoding="lc0_1858", fill_value=-1e9)

    assert full.shape == (POLICY_SIZE,)
    assert float(full[int(COMPACT_TO_FULL_POLICY[0])]) == 3.0
    assert float(full[int(np.flatnonzero(FULL_TO_COMPACT_POLICY < 0)[0])]) == -1e9


def test_compact_to_full_policy_batch_conversion_fills_invalid_slots():
    compact = np.zeros((2, COMPACT_POLICY_SIZE), dtype=np.float32)
    compact[0, 0] = 3.0
    compact[1, 1] = 4.0

    full = policy_batch_to_full(compact, policy_encoding="lc0_1858", fill_value=-1e9)

    assert full.shape == (2, POLICY_SIZE)
    assert float(full[0, int(COMPACT_TO_FULL_POLICY[0])]) == 3.0
    assert float(full[1, int(COMPACT_TO_FULL_POLICY[1])]) == 4.0
    invalid = int(np.flatnonzero(FULL_TO_COMPACT_POLICY < 0)[0])
    assert float(full[0, invalid]) == -1e9


def test_policy_batch_to_full_if_needed_uses_declared_encoding():
    full = np.zeros((2, POLICY_SIZE), dtype=np.float32)
    assert policy_batch_to_full_if_needed(full, policy_encoding="az_4672") is full

    compact = np.zeros((2, COMPACT_POLICY_SIZE), dtype=np.float32)
    compact[0, 0] = 5.0
    expanded = policy_batch_to_full_if_needed(compact, policy_encoding="lc0_1858", fill_value=-1e9)
    assert expanded.shape == (2, POLICY_SIZE)
    assert float(expanded[0, int(COMPACT_TO_FULL_POLICY[0])]) == 5.0

    with pytest.raises(ValueError, match="policies must be"):
        policy_batch_to_full_if_needed(compact, policy_encoding="az_4672", fill_value=-1e9)

    wrong = np.zeros((2, COMPACT_POLICY_SIZE - 1), dtype=np.float32)
    with pytest.raises(ValueError, match="compact policies"):
        policy_batch_to_full_if_needed(wrong, policy_encoding="lc0_1858", fill_value=-1e9)
