"""Test batch_process_ply C function against Python reference implementation."""
from __future__ import annotations

import math

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
from chess_anti_engine.moves import legal_move_mask, move_to_index, index_to_move, POLICY_SIZE

try:
    from chess_anti_engine.mcts._mcts_tree import batch_process_ply
    HAS_C = True
except ImportError:
    HAS_C = False


def _python_process_ply(board, pol_logits, wdl_logits, action, value, mcts_probs,
                        df_enabled, df_q_w, df_pol_s, df_min, df_slope):
    """Python reference implementation matching manager.py lines 760-834."""
    mask = legal_move_mask(board)
    ply = len(board.move_stack)
    pov = board.turn

    # Masked softmax
    raw = pol_logits.copy()
    raw[~mask] = -1e9
    raw -= raw.max()
    raw = np.exp(raw)
    raw[~mask] = 0.0
    s = raw.sum()
    if s > 0:
        raw /= s
    else:
        raw = mask.astype(np.float32) / mask.sum()

    # KL divergence
    imp = np.maximum(mcts_probs, 1e-12)
    raw_c = np.maximum(raw, 1e-12)
    kl = float(np.sum(raw_c * (np.log(raw_c) - np.log(imp))))

    # WDL softmax
    w, d, l = wdl_logits
    mx = max(w, d, l)
    ew, ed, el = math.exp(w - mx), math.exp(d - mx), math.exp(l - mx)
    ws = ew + ed + el
    wdl_net = np.array([ew/ws, ed/ws, el/ws], dtype=np.float32) if ws > 0 else np.array([0, 1, 0], dtype=np.float32)

    orig_q = float(wdl_net[0] - wdl_net[2])
    best_q = float(value)
    q_surprise = abs(best_q - orig_q)

    difficulty = q_surprise * df_q_w + kl * df_pol_s
    if not math.isfinite(difficulty):
        difficulty = 1.0
    keep_prob = 1.0
    if df_enabled:
        keep_prob = max(df_min, min(1.0, difficulty * df_slope))

    # Search WDL
    d_raw = wdl_net[1]
    rem = max(0.0, 1.0 - d_raw)
    q_cl = max(-rem, min(rem, best_q))
    w_search = 0.5 * (rem + q_cl)
    wdl_search = np.array([w_search, d_raw, rem - w_search], dtype=np.float32)

    # Push move
    move = index_to_move(int(action), board)
    board.push(move)
    game_over = board.is_game_over()

    return {
        'mask': mask, 'ply': ply, 'pov': pov,
        'wdl_net': wdl_net, 'wdl_search': wdl_search,
        'priority': difficulty, 'keep_prob': keep_prob,
        'game_over': game_over,
    }


@pytest.mark.skipif(not HAS_C, reason="C extension not available")
def test_single_game_parity():
    """C function produces same results as Python reference for one game."""
    board = chess.Board()
    cb = cboard_from_board_fast(board)
    mask = legal_move_mask(board)
    legal = np.flatnonzero(mask)
    action = int(legal[5])  # Pick an arbitrary legal move

    pol = np.random.RandomState(42).randn(POLICY_SIZE).astype(np.float32)
    wdl = np.array([0.3, 0.4, 0.3], dtype=np.float32)
    value = 0.05
    probs = np.abs(np.random.RandomState(43).randn(POLICY_SIZE).astype(np.float32))
    probs /= probs.sum()

    df_q_w, df_pol_s, df_min, df_slope = 4.8, 3.8, 0.09, 1.0

    # Python reference
    py = _python_process_ply(
        board.copy(), pol, wdl, action, value, probs,
        True, df_q_w, df_pol_s, df_min, df_slope,
    )

    # C function
    (c_x, c_probs, c_wdl_net, c_wdl_search, c_priority,
     c_keep, c_mask, c_ply, c_pov, c_over) = batch_process_ply(
        [cb],
        pol.reshape(1, -1), wdl.reshape(1, -1),
        np.array([action], dtype=np.int32),
        np.array([value], dtype=np.float64),
        probs.reshape(1, -1),
        np.array([1], dtype=np.int32),
        np.array([1.0], dtype=np.float64),
        1, df_q_w, df_pol_s, df_min, df_slope,
    )

    assert c_ply[0] == py['ply']
    assert c_pov[0] == (1 if py['pov'] else 0)
    assert c_over[0] == (1 if py['game_over'] else 0)
    assert int(c_mask[0].sum()) == int(py['mask'].sum())
    np.testing.assert_allclose(c_wdl_net[0], py['wdl_net'], atol=1e-5)
    np.testing.assert_allclose(c_wdl_search[0], py['wdl_search'], atol=1e-5)
    assert abs(float(c_priority[0]) - py['priority']) < 0.01
    assert abs(float(c_keep[0]) - py['keep_prob']) < 0.01


@pytest.mark.skipif(not HAS_C, reason="C extension not available")
def test_multi_game():
    """C function processes multiple games correctly."""
    n = 8
    boards = [chess.Board() for _ in range(n)]
    cboards = [cboard_from_board_fast(b) for b in boards]
    legal = cboards[0].legal_move_indices()

    pol = np.random.randn(n, POLICY_SIZE).astype(np.float32)
    wdl = np.random.randn(n, 3).astype(np.float32)
    actions = np.array([int(legal[i % len(legal)]) for i in range(n)], dtype=np.int32)
    values = np.random.randn(n).astype(np.float64) * 0.1
    probs = np.abs(np.random.randn(n, POLICY_SIZE).astype(np.float32))
    probs /= probs.sum(axis=1, keepdims=True)

    result = batch_process_ply(
        cboards, pol, wdl, actions, values, probs,
        np.ones(n, dtype=np.int32), np.ones(n, dtype=np.float64),
        1, 4.8, 3.8, 0.09, 1.0,
    )
    x, p, wn, ws, pri, keep, mask, ply, pov, over = result

    assert x.shape == (n, 146, 8, 8)
    assert p.shape == (n, POLICY_SIZE)
    assert mask.shape == (n, POLICY_SIZE)
    assert ply.shape == (n,)
    assert all(mask[i].sum() == 20 for i in range(n))  # Starting position has 20 legal moves
    assert all(ply[i] == 0 for i in range(n))
    assert all(pov[i] == 1 for i in range(n))  # White to move


@pytest.mark.skipif(not HAS_C, reason="C extension not available")
def test_game_over_detection():
    """C function detects checkmate."""
    # Scholar's mate position: 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6"]:
        board.push(chess.Move.from_uci(uci))

    cb = cboard_from_board_fast(board)
    # Qxf7# is the checkmate move
    action = int(move_to_index(chess.Move.from_uci("h5f7"), board))

    pol = np.zeros((1, POLICY_SIZE), dtype=np.float32)
    wdl = np.array([[0.9, 0.05, 0.05]], dtype=np.float32)
    probs = np.zeros((1, POLICY_SIZE), dtype=np.float32)
    probs[0, action] = 1.0

    result = batch_process_ply(
        [cb], pol, wdl,
        np.array([action], dtype=np.int32),
        np.array([0.9], dtype=np.float64),
        probs,
        np.array([1], dtype=np.int32), np.array([1.0], dtype=np.float64),
        0, 4.8, 3.8, 0.09, 1.0,
    )
    assert result[9][0] == 1  # game_over = True
