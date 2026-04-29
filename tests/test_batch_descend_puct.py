"""Parity tests for batch_descend_puct + batch_integrate_leaves.

The batched primitive must produce the same per-leaf descent results as
calling walker_descend_puct N times in a single thread (no virtual loss
race — vloss is applied in-order in both cases). Visit counts after a
full descend+integrate cycle must match the single-leaf-loop equivalent.
"""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree

_MAX_PATH = 512  # MCTS_MAX_PATH in the C extension
_C_PUCT = 1.4
_FPU_ROOT = 0.0
_FPU_RED = 0.2
_VLOSS = 3
_POL_SIZE = 4672


def _root(board: chess.Board) -> tuple[MCTSTree, int, CBoard, np.ndarray]:
    tree = MCTSTree()
    cb = CBoard.from_board(board)
    legal = cb.legal_move_indices()
    rid = tree.add_root(1, 0.0)
    priors = np.full(legal.size, 1.0 / max(1, legal.size), dtype=np.float64)
    tree.expand(rid, legal.astype(np.int32), priors)
    return tree, rid, cb, legal


def _fake_pol_wdl(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    pol = rng.standard_normal((n, _POL_SIZE), dtype=np.float32)
    wdl = rng.standard_normal((n, 3), dtype=np.float32)
    return pol, wdl


@pytest.mark.skip(
    reason="walker_descend_puct calls try_forced_collapse (extends path "
    "through forced reply chains); batch_descend_puct deliberately omits "
    "it (see _mcts_tree.c comment at MCTSTree_batch_descend_puct). Bit-"
    "equal leaf parity isn't expected by design — root visit-count parity "
    "is covered by test_batch_integrate_matches_walker_integrate_loop."
)
def test_batch_descend_matches_walker_descend_loop():
    """batch_descend_puct(N) ≡ N× walker_descend_puct in a single thread."""
    n = 16
    board = chess.Board()

    # Path A: batched primitive
    tree_a, rid_a, cb_a, _ = _root(board)
    enc_a = np.empty((n, 146, 8, 8), dtype=np.float32)
    leaf_ids_a = np.empty(n, dtype=np.int32)
    path_buf_a = np.empty(n * _MAX_PATH, dtype=np.int32)
    path_lens_a = np.empty(n, dtype=np.int32)
    legal_buf_a = np.empty(n * 256, dtype=np.int32)
    legal_lens_a = np.empty(n, dtype=np.int32)
    term_qs_a = np.empty(n, dtype=np.float64)
    is_term_a = np.empty(n, dtype=np.int8)
    got = tree_a.batch_descend_puct(
        rid_a, cb_a, n, _C_PUCT, _FPU_ROOT, _FPU_RED, _VLOSS,
        enc_a, leaf_ids_a, path_buf_a, path_lens_a,
        legal_buf_a, legal_lens_a, term_qs_a, is_term_a,
    )
    assert got == n

    # Path B: single-leaf loop
    tree_b, rid_b, cb_b, _ = _root(board)
    enc_b = np.empty((n, 146, 8, 8), dtype=np.float32)
    leaf_ids_b = np.empty(n, dtype=np.int32)
    path_lens_b = np.empty(n, dtype=np.int32)
    paths_b = []
    legals_b = []
    is_term_b = np.zeros(n, dtype=np.int8)
    for i in range(n):
        leaf_id, path, legal, term_q = tree_b.walker_descend_puct(
            rid_b, cb_b, _C_PUCT, _FPU_ROOT, _FPU_RED, _VLOSS,
            enc_b[i:i+1],
        )
        leaf_ids_b[i] = leaf_id
        path_lens_b[i] = path.size
        paths_b.append(path)
        legals_b.append(legal)
        if term_q is not None:
            is_term_b[i] = 1
            tree_b.backprop(path, float(term_q))

    # Verify per-leaf parity.
    np.testing.assert_array_equal(leaf_ids_a, leaf_ids_b)
    np.testing.assert_array_equal(path_lens_a, path_lens_b)
    np.testing.assert_array_equal(is_term_a, is_term_b)
    for i in range(n):
        plen = int(path_lens_a[i])
        np.testing.assert_array_equal(
            path_buf_a[i * _MAX_PATH:i * _MAX_PATH + plen], paths_b[i],
        )
        nlegal = int(legal_lens_a[i])
        # walker_descend_puct returns sorted-on-callsite, batch returns
        # unsorted (sorted=0 in both, so they match).
        if not is_term_a[i]:
            np.testing.assert_array_equal(
                legal_buf_a[i * 256:i * 256 + nlegal], legals_b[i],
            )
            np.testing.assert_array_equal(enc_a[i], enc_b[i])


def test_batch_integrate_matches_walker_integrate_loop():
    """After descend+integrate, root visit counts match the single-leaf path."""
    n = 32
    board = chess.Board()
    rng = np.random.default_rng(42)
    pol, wdl = _fake_pol_wdl(rng, n)

    # Path A: batch
    tree_a, rid_a, cb_a, _ = _root(board)
    enc_a = np.empty((n, 146, 8, 8), dtype=np.float32)
    leaf_ids_a = np.empty(n, dtype=np.int32)
    path_buf_a = np.empty(n * _MAX_PATH, dtype=np.int32)
    path_lens_a = np.empty(n, dtype=np.int32)
    legal_buf_a = np.empty(n * 256, dtype=np.int32)
    legal_lens_a = np.empty(n, dtype=np.int32)
    term_qs_a = np.empty(n, dtype=np.float64)
    is_term_a = np.empty(n, dtype=np.int8)
    tree_a.batch_descend_puct(
        rid_a, cb_a, n, _C_PUCT, _FPU_ROOT, _FPU_RED, _VLOSS,
        enc_a, leaf_ids_a, path_buf_a, path_lens_a,
        legal_buf_a, legal_lens_a, term_qs_a, is_term_a,
    )
    tree_a.batch_integrate_leaves(
        n, path_buf_a, path_lens_a, legal_buf_a, legal_lens_a,
        is_term_a, pol, wdl, _VLOSS,
    )
    actions_a, visits_a = tree_a.get_children_visits(rid_a)

    # Path B: descend N (vloss accumulated, no integration), then integrate
    # N. Mirrors the batched primitive's two-phase structure — interleaving
    # would give a different (sequential-vloss) algorithm.
    tree_b, rid_b, cb_b, _ = _root(board)
    enc_b = np.empty((1, 146, 8, 8), dtype=np.float32)
    descents = []
    for i in range(n):
        _leaf_id, path, legal, term_q = tree_b.walker_descend_puct(
            rid_b, cb_b, _C_PUCT, _FPU_ROOT, _FPU_RED, _VLOSS, enc_b,
        )
        descents.append((path, legal, term_q))
        if term_q is not None:
            tree_b.backprop(path, float(term_q))
    for i, (path, legal, term_q) in enumerate(descents):
        if term_q is not None:
            continue
        tree_b.walker_integrate_leaf(path, legal, pol[i], wdl[i], _VLOSS)
    actions_b, visits_b = tree_b.get_children_visits(rid_b)

    # Both methods explore the tree identically when run single-threaded.
    np.testing.assert_array_equal(actions_a, actions_b)
    np.testing.assert_array_equal(visits_a, visits_b)


def test_batch_descend_zero_n():
    """N=0 is a no-op that returns 0."""
    tree, rid, cb, _ = _root(chess.Board())
    enc = np.empty((1, 146, 8, 8), dtype=np.float32)
    got = tree.batch_descend_puct(
        rid, cb, 0, _C_PUCT, _FPU_ROOT, _FPU_RED, _VLOSS,
        enc, np.empty(1, np.int32), np.empty(_MAX_PATH, np.int32),
        np.empty(1, np.int32), np.empty(256, np.int32),
        np.empty(1, np.int32), np.empty(1, np.float64), np.empty(1, np.int8),
    )
    assert got == 0


def test_batch_descend_undersized_buffer_raises():
    """ValueError when any buffer is smaller than n_leaves needs."""
    tree, rid, cb, _ = _root(chess.Board())
    n = 4
    enc = np.empty((n, 146, 8, 8), dtype=np.float32)
    # Path buffer too small: needs n*_MAX_PATH, give n*16
    too_small_path = np.empty(n * 16, dtype=np.int32)
    try:
        tree.batch_descend_puct(
            rid, cb, n, _C_PUCT, _FPU_ROOT, _FPU_RED, _VLOSS,
            enc, np.empty(n, np.int32), too_small_path,
            np.empty(n, np.int32), np.empty(n * 256, np.int32),
            np.empty(n, np.int32), np.empty(n, np.float64), np.empty(n, np.int8),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on undersized path buffer")
