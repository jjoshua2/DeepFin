"""Tests for C-accelerated MCTS tree."""
from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import encode_positions_batch
from chess_anti_engine.inference import _policy_output
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.puct import MCTSConfig, run_mcts_many
from chess_anti_engine.mcts.puct_c import run_mcts_many_c
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import legal_move_mask

try:
    from chess_anti_engine.encoding._lc0_ext import CBoard
except ImportError:  # pragma: no cover - extension absent
    CBoard = None

try:
    from chess_anti_engine.encoding._features_ext import (
        compute_extra_features as _c_compute,
    )
except ImportError:  # pragma: no cover - extension absent
    _c_compute = None


# ── Low-level tree tests ─────────────────────────────────────────────────


def test_tree_add_root():
    t = MCTSTree()
    root = t.add_root(1, 0.5)
    assert root == 0
    assert t.node_count() == 1
    assert abs(t.node_q(root) - 0.5) < 1e-9


def test_tree_expand_and_select():
    t = MCTSTree()
    root = t.add_root(1, 0.0)

    # Expand root with 3 children
    actions = np.array([10, 20, 30], dtype=np.int32)
    priors = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    t.expand(root, actions, priors)

    assert t.is_expanded(root)
    assert t.node_count() == 4  # root + 3 children

    # Select should pick a leaf (one of the children)
    root_ids = np.array([root], dtype=np.int32)
    leaves = t.select_leaves(root_ids, 2.5, 1.0, 1.2)
    assert len(leaves) == 1
    leaf_id, action_path, node_path, is_exp = leaves[0]
    assert leaf_id in [1, 2, 3]
    assert not is_exp  # children are unexpanded
    assert len(action_path) == 1
    assert int(action_path[0]) in [10, 20, 30]
    assert len(node_path) == 2
    assert int(node_path[0]) == root


def test_tree_backprop():
    t = MCTSTree()
    root = t.add_root(1, 0.0)

    actions = np.array([10, 20], dtype=np.int32)
    priors = np.array([0.5, 0.5], dtype=np.float64)
    t.expand(root, actions, priors)

    # Backprop through path [root, child_0]
    node_path = np.array([0, 1], dtype=np.int32)
    t.backprop(node_path, 1.0)  # value=1.0 from leaf perspective

    # Child should have N=1, W=1.0 (value from its perspective)
    # Root should have N=2 (1 initial + 1 backprop), W = 0.0 + (-1.0) = -1.0
    acts, visits = t.get_children_visits(root)
    assert visits[0] == 1


def test_tree_backprop_many():
    t = MCTSTree()
    root = t.add_root(1, 0.0)
    actions = np.array([10, 20], dtype=np.int32)
    priors = np.array([0.5, 0.5], dtype=np.float64)
    t.expand(root, actions, priors)

    paths = [
        np.array([0, 1], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
    ]
    values = [1.0, -0.5]
    t.backprop_many(paths, values)

    acts, visits = t.get_children_visits(root)
    assert int(visits[0]) == 1
    assert int(visits[1]) == 1


def test_tree_select_uses_parent_perspective_for_visited_children():
    t = MCTSTree()
    root = t.add_root(1, 0.0)

    actions = np.array([10, 20], dtype=np.int32)
    priors = np.array([0.5, 0.5], dtype=np.float64)
    t.expand(root, actions, priors)

    # Child 10 gets a positive value from its own perspective, so it should
    # be worse for the root than child 20 after sign flip into parent frame.
    t.backprop(np.array([root, 1], dtype=np.int32), 1.0)
    t.backprop(np.array([root, 2], dtype=np.int32), -1.0)

    root_ids = np.array([root], dtype=np.int32)
    leaves = t.select_leaves(root_ids, 0.0, 1.0, 1.2)
    assert len(leaves) == 1
    _leaf_id, action_path, _node_path, _is_exp = leaves[0]
    assert int(action_path[0]) == 20


def test_tree_get_children_visits():
    t = MCTSTree()
    root = t.add_root(1, 0.0)
    actions = np.array([42, 99, 7], dtype=np.int32)
    priors = np.array([0.3, 0.5, 0.2], dtype=np.float64)
    t.expand(root, actions, priors)

    acts, visits = t.get_children_visits(root)
    assert len(acts) == 3
    assert set(acts.tolist()) == {42, 99, 7}
    assert all(v == 0 for v in visits)


def test_tree_reset():
    t = MCTSTree()
    t.add_root(1, 0.0)
    assert t.node_count() == 1
    t.reset()
    assert t.node_count() == 0


def test_tree_reset_compact_releases_high_water_buffers():
    t = MCTSTree()
    baseline = t.memory_bytes()

    t.reserve(16_384, 262_144)
    grown = t.memory_bytes()
    assert grown > baseline

    t.reset()
    assert t.memory_bytes() == grown

    t.reset_compact()
    assert t.node_count() == 0
    assert t.memory_bytes() < grown
    assert t.memory_bytes() <= baseline


# ── Integration: C tree vs Python tree produce same structure ────────────


@pytest.fixture(scope="module")
def tiny_model():
    m = build_model(ModelConfig(kind="tiny"))
    m.eval()
    return m


def test_run_mcts_many_c_basic(tiny_model):
    """C MCTS produces valid output (correct shapes, legal actions)."""
    boards = [chess.Board()]
    rng = np.random.default_rng(42)
    cfg = MCTSConfig(simulations=16, temperature=1.0)

    probs, actions, values, masks = run_mcts_many_c(
        tiny_model, boards, device="cpu", rng=rng, cfg=cfg,
    )

    assert len(probs) == 1
    assert probs[0].shape == (4672,)
    assert abs(float(probs[0].sum()) - 1.0) < 0.01
    assert 0 <= actions[0] < 4672
    assert masks[0].shape == (4672,)
    assert masks[0].any()
    # Selected action should be legal
    assert masks[0][actions[0]]
    assert np.array_equal(masks[0], legal_move_mask(boards[0]))
    assert np.count_nonzero(probs[0][~masks[0]]) == 0


def test_run_mcts_many_c_multi_board(tiny_model):
    """C MCTS with multiple boards."""
    rng = np.random.default_rng(0)
    boards = [chess.Board()]
    # Add a board a few moves in
    b2 = chess.Board()
    b2.push_san("e4")
    b2.push_san("e5")
    boards.append(b2)

    cfg = MCTSConfig(simulations=8, temperature=1.0)
    probs, actions, values, masks = run_mcts_many_c(
        tiny_model, boards, device="cpu", rng=rng, cfg=cfg,
    )

    assert len(probs) == 2
    for i in range(2):
        assert probs[i].shape == (4672,)
        assert masks[i][actions[i]]
        assert np.array_equal(masks[i], legal_move_mask(boards[i]))
        assert np.count_nonzero(probs[i][~masks[i]]) == 0


def test_run_mcts_many_c_terminal():
    """C MCTS handles terminal positions (scholars mate)."""
    model = build_model(ModelConfig(kind="tiny"))
    model.eval()

    # Fool's mate
    b = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4"]:
        b.push_san(san)
    assert b.is_checkmate()

    rng = np.random.default_rng(0)
    cfg = MCTSConfig(simulations=8)
    # Should not crash on terminal position
    probs, actions, values, masks = run_mcts_many_c(
        model, [b], device="cpu", rng=rng, cfg=cfg,
    )
    assert len(probs) == 1
    # No legal moves, so probs should be all zeros
    assert float(probs[0].sum()) < 0.01


def test_run_mcts_many_draw_terminal_root_returns_zero_policy(tiny_model):
    bare_kings = chess.Board("8/8/8/8/8/8/4k3/4K3 w - - 0 1")
    assert bare_kings.is_game_over()
    assert bare_kings.result(claim_draw=True) == "1/2-1/2"

    cfg = MCTSConfig(simulations=8, temperature=0.0)
    for fn in (run_mcts_many, run_mcts_many_c):
        probs, actions, values, masks = fn(
            tiny_model,
            [bare_kings],
            device="cpu",
            rng=np.random.default_rng(0),
            cfg=cfg,
        )
        assert actions == [0]
        assert values == [0.0]
        assert float(probs[0].sum()) == 0.0
        assert not masks[0].any()


def test_c_vs_python_same_root_q(tiny_model):
    """C and Python MCTS should produce the same root Q value (deterministic, no Dirichlet)."""
    board = chess.Board()
    cfg = MCTSConfig(simulations=32, dirichlet_eps=0.0, temperature=0.0)

    rng_py = np.random.default_rng(123)
    rng_c = np.random.default_rng(123)

    _, _, values_py, _ = run_mcts_many(
        tiny_model, [board], device="cpu", rng=rng_py, cfg=cfg,
    )
    _, _, values_c, _ = run_mcts_many_c(
        tiny_model, [board], device="cpu", rng=rng_c, cfg=cfg,
    )

    # Values should be close (not exact due to different traversal order)
    assert abs(values_py[0] - values_c[0]) < 0.15, \
        f"Python Q={values_py[0]:.4f} vs C Q={values_c[0]:.4f}"


def test_c_vs_python_match_on_history_rich_position(tiny_model):
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]:
        board.push(chess.Move.from_uci(uci))

    cfg = MCTSConfig(simulations=16, dirichlet_eps=0.0, temperature=0.0)
    probs_py, actions_py, values_py, masks_py = run_mcts_many(
        tiny_model,
        [board],
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=cfg,
    )
    probs_c, actions_c, values_c, masks_c = run_mcts_many_c(
        tiny_model,
        [board],
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=cfg,
    )

    assert actions_py == actions_c
    np.testing.assert_allclose(values_py, values_c, atol=1e-3)
    np.testing.assert_allclose(probs_py[0], probs_c[0], atol=1e-3)
    assert np.array_equal(masks_py[0], masks_c[0])


def test_run_mcts_many_precomputed_logits_matches_direct_path(tiny_model):
    boards = [chess.Board()]
    b2 = chess.Board()
    for san in ["e4", "c5", "Nf3", "d6", "d4"]:
        b2.push_san(san)
    boards.append(b2)

    cfg = MCTSConfig(simulations=12, dirichlet_eps=0.0, temperature=0.0)

    for board in boards:
        xs = encode_positions_batch([board], add_features=True)
        with torch.no_grad():
            out = tiny_model(torch.from_numpy(xs))
        policy_out = _policy_output(out)
        pre_pol = policy_out.detach().float().cpu().numpy()
        pre_wdl = out["wdl"].detach().float().cpu().numpy()

        direct = run_mcts_many(
            tiny_model,
            [board],
            device="cpu",
            rng=np.random.default_rng(321),
            cfg=cfg,
        )
        precomputed = run_mcts_many(
            tiny_model,
            [board],
            device="cpu",
            rng=np.random.default_rng(321),
            cfg=cfg,
            pre_pol_logits=pre_pol,
            pre_wdl_logits=pre_wdl,
        )

        probs_direct, actions_direct, values_direct, masks_direct = direct
        probs_pre, actions_pre, values_pre, masks_pre = precomputed

        assert actions_direct == actions_pre
        np.testing.assert_allclose(values_direct, values_pre, atol=5e-5)
        np.testing.assert_allclose(probs_direct[0], probs_pre[0], atol=1e-7)
        assert np.array_equal(masks_direct[0], masks_pre[0])
        assert np.array_equal(masks_direct[0], legal_move_mask(board))
        assert np.count_nonzero(probs_direct[0][~masks_direct[0]]) == 0


@pytest.mark.skipif(CBoard is None or _c_compute is None, reason="CBoard/features extension not available")
def test_cboard_encode_146_matches_encode_position():
    boards = [
        chess.Board(),
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        chess.Board("8/8/4k3/8/2p5/8/B2K4/8 w - - 0 1"),
    ]

    hist_board = chess.Board()
    for san in ["e4", "c5", "Nf3", "d6", "d4", "cxd4"]:
        hist_board.push_san(san)
    boards.append(hist_board)

    for board in boards:
        cb = CBoard.from_board(board)
        fused = cb.encode_146()
        turn = cb.turn
        occ_w = cb.occ_white
        occ_b = cb.occ_black
        if turn:
            us_occ, them_occ = occ_w, occ_b
        else:
            us_occ, them_occ = occ_b, occ_w
        pieces_us = np.array(
            [
                cb.pawns & us_occ,
                cb.knights & us_occ,
                cb.bishops & us_occ,
                cb.rooks & us_occ,
                cb.queens & us_occ,
                cb.kings & us_occ,
            ],
            dtype=np.uint64,
        )
        pieces_them = np.array(
            [
                cb.pawns & them_occ,
                cb.knights & them_occ,
                cb.bishops & them_occ,
                cb.rooks & them_occ,
                cb.queens & them_occ,
                cb.kings & them_occ,
            ],
            dtype=np.uint64,
        )
        occupied = int(occ_w | occ_b)
        us_kings = cb.kings & us_occ
        them_kings = cb.kings & them_occ
        king_sq_us = int(us_kings).bit_length() - 1 if us_kings else -1
        king_sq_them = int(them_kings).bit_length() - 1 if them_kings else -1
        ep_square = -1 if cb.ep_square is None else cb.ep_square
        feat = _c_compute(pieces_us, pieces_them, occupied, king_sq_us, king_sq_them, turn, ep_square)
        ref = np.concatenate([cb.encode_planes(), feat], axis=0)

        assert fused.shape == (146, 8, 8)
        assert fused.dtype == np.float32
        np.testing.assert_allclose(fused, ref, atol=1e-6)
