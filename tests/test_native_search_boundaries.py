"""Supported wrapper calls at legal-support, solved-output, and root-seed boundaries."""
from __future__ import annotations

import threading

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig, PLAY_PUCT_DEFAULTS, run_gumbel_root_many
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.mcts.puct import _value_scalar_from_wdl_logits
from chess_anti_engine.mcts.root_tactics import immediate_mate_move, immediate_terminal_draw_indices
from chess_anti_engine.moves.encode import move_to_index
from chess_anti_engine.uci.search import (
    SearchWorker,
    _best_move_and_pv,
    _multipv_lines,
    _pv_from_root_action,
    _reply_at_child,
)
from chess_anti_engine.uci.time_manager import Deadline


class _ZeroPriorEvaluator:
    def __init__(self):
        bad = chess.Board()
        bad.push_uci('e2e4')
        self.bad = encode_cboard(CBoard.from_board(bad), input_history_encoding='lc0_root_legacy_meta')[:12]

    def evaluate_encoded(self, x, relations=None):
        del relations
        wdl = np.empty((len(x), 3), np.float32)
        for i, row in enumerate(x):
            wdl[i] = ([8, 0, -8] if np.array_equal(row[:12], self.bad)
                      else ([-8, 0, 8] if row[108, 0, 0] else [8, 0, -8]))
        return np.zeros((len(x), 4672), np.float32), wdl


@pytest.mark.parametrize(('native', 'restricted'), [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize(('gap', 'policy_temp'), [(50.0, 0.05), (1000.0, 1.0)])
@pytest.mark.parametrize('temperature', [0.0, 1.0])
def test_zero_prior_legal_actions_keep_searched_policy_and_value(native, restricted, gap, policy_temp, temperature):
    board = chess.Board()
    bad = int(move_to_index(chess.Move.from_uci('e2e4'), board))
    good = int(move_to_index(chess.Move.from_uci('e2e3'), board))
    pol = np.full((1, 4672), -gap, np.float32)
    pol[0, bad] = 0.0
    cfg = GumbelConfig(
        simulations=8 if restricted else 50, topk=2 if restricted else 20,
        add_noise=False, temperature=temperature, policy_temp=policy_temp,
        c_scale=1.0, input_history_encoding='lc0_root_legacy_meta',
    )
    if native:
        result = run_gumbel_root_many_c(
            model=None, boards=[board], device='cpu', rng=np.random.default_rng(7),
            cfg=cfg, evaluator=_ZeroPriorEvaluator(), pre_pol_logits=pol,
            pre_wdl_logits=np.zeros((1, 3), np.float32),
            allowed_root_indices_batch=[{bad, good}] if restricted else None,
            target_batch=1, vloss_weight=1,
        )
    else:
        result = run_gumbel_root_many(
            model=None, boards=[board], device='cpu', rng=np.random.default_rng(7),
            cfg=cfg, evaluator=_ZeroPriorEvaluator(), pre_pol_logits=pol,
            pre_wdl_logits=np.zeros((1, 3), np.float32),
        )
    probs, actions, values, masks = result[:4]
    action = int(actions[0])
    assert masks[0][action]
    assert action != bad
    assert probs[0][action] > 0.0
    assert values[0] > 0.9
    assert probs[0].sum() == pytest.approx(1.0)
    if restricted:
        assert action == good
        assert set(np.flatnonzero(probs[0])) <= {bad, good}


@pytest.mark.parametrize('native', [False, True])
def test_explicit_gumbel_support_keeps_terminal_draw_pruning(native):
    board = chess.Board('7k/8/5K2/5Q2/8/8/8/8 w - - 0 1')
    draws = immediate_terminal_draw_indices(board)
    assert int(move_to_index(chess.Move.from_uci('f5g6'), board)) in draws
    cfg = GumbelConfig(simulations=8, topk=2, add_noise=False, temperature=0.0)
    run_root = run_gumbel_root_many_c if native else run_gumbel_root_many
    result = run_root(
        model=None, boards=[board], device='cpu', rng=np.random.default_rng(7),
        cfg=cfg, evaluator=_ZeroPriorEvaluator(),
        pre_pol_logits=np.zeros((1, 4672), np.float32),
        pre_wdl_logits=np.array([[3, 0, -3]], np.float32),
    )
    assert int(result[1][0]) not in draws
    assert not result[0][0][list(draws)].any()


class _MateEvaluator:
    def __init__(self, board):
        self.root = encode_cboard(CBoard.from_board(board))[:12]
        self.pol = np.full(4672, -100.0, np.float32)
        self.pol[int(move_to_index(chess.Move.from_uci('f5g6'), board))] = 0.0
        self.pol[int(move_to_index(chess.Move.from_uci('f5g5'), board))] = -2.2
        after = board.copy(stack=True)
        after.push_uci('f5g5')
        after.push_uci('h8h7')
        self.pol[int(move_to_index(chess.Move.from_uci('g5g7'), after))] = 6.0

    def evaluate_encoded(self, x, relations=None):
        del relations
        wdl = np.zeros((len(x), 3), np.float32)
        for i, row in enumerate(x):
            if not np.array_equal(row[:12], self.root) and row[4].sum() > 0:
                wdl[i] = [4, 0, -4]
        return np.tile(self.pol, (len(x), 1)), wdl


def _puct_config():
    return GumbelConfig(
        c_puct=PLAY_PUCT_DEFAULTS['c_puct'],
        cpuct_factor=PLAY_PUCT_DEFAULTS['cpuct_factor'],
        cpuct_base=PLAY_PUCT_DEFAULTS['cpuct_base'],
        fpu_reduction=PLAY_PUCT_DEFAULTS['fpu_reduction'],
    )


def test_puct_cutoff_preserves_a_late_real_mate_proof():
    board = chess.Board('7k/8/5K2/5Q2/8/8/8/8 w - - 0 1')
    assert board.is_valid()
    assert immediate_mate_move(board) is None
    stalemate = board.copy(stack=True)
    stalemate.push_uci('f5g6')
    assert stalemate.is_stalemate()
    worker = SearchWorker(_MateEvaluator(board), device='cpu',
                          gumbel_cfg=_puct_config(),
                          chunk_sims=1, n_walkers=2, vloss_weight=1)
    try:
        result = worker.run(board, stop_event=threading.Event(), deadline=Deadline(None),
                            max_nodes=11, include_ponder=True)
        tree, rid = worker._tree, worker._root_id
        assert tree is not None
        assert rid is not None
        assert tree.get_solved_status(rid) == 1
        proof = int(move_to_index(chess.Move.from_uci('f5g5'), board))
        draw = int(move_to_index(chess.Move.from_uci('f5g6'), board))
        a, visits = tree.get_children_visits(rid)
        assert visits[np.where(a == draw)[0][0]] > visits[np.where(a == proof)[0][0]]
        assert result.nodes == 11
        assert result.bestmove_uci == 'f5g5'
        assert result.pv == ('f5g5', 'h8h7', 'g5g7')
        assert result.ponder_uci == 'h8h7'
        assert worker._emitted_action(a, visits, None) == proof
        assert _multipv_lines(tree, rid, 2, 0.0)[0][2][0] == proof
    finally:
        worker.close()


def _output_tree(status):
    tree = MCTSTree()
    rid = tree.add_root(1, 0.0)
    tree.expand(rid, np.array([1, 2, 3], np.int32), np.array([0.5, 0.3, 0.2], np.float64))
    for action, visits in [(1, 9), (2, 2), (3, 1)]:
        child = tree.find_child(rid, action)
        for _ in range(visits):
            tree.backprop(np.array([rid, child], np.int32), 0.0)
    if status == 1:
        for action in [2, 3]:
            tree.mark_solved_path(np.array([rid, tree.find_child(rid, action)], np.int32), -1)
    elif status == 2:
        for action in [1, 2, 3]:
            tree.mark_solved_path(np.array([rid, tree.find_child(rid, action)], np.int32), 1 if action == 1 else 2)
    elif status == -1:
        for action in [1, 2, 3]:
            tree.mark_solved_path(np.array([rid, tree.find_child(rid, action)], np.int32), 1)
    assert tree.get_solved_status(rid) == status
    return tree, rid


@pytest.mark.parametrize('status', [1, 2])
def test_proof_preserving_output_respects_allowed_actions_and_safe_gumbel(status):
    tree, rid = _output_tree(status)
    worker = SearchWorker(_ZeroPriorEvaluator(), device='cpu')
    worker._tree, worker._root_id = tree, rid
    actions, visits = tree.get_children_visits(rid)
    try:
        assert worker._emitted_action(actions, visits, None) == 2
        assert _best_move_and_pv(tree, rid)[0] == 2
        assert _multipv_lines(tree, rid, 3, 0.0)[0][2][0] == 2
        worker._last_gumbel_action_idx = 1
        assert worker._emitted_action(actions, visits, None) == 2
        worker._last_gumbel_action_idx = 3
        assert worker._emitted_action(actions, visits, None) == 3
        assert _best_move_and_pv(tree, rid, {1, 3})[0] == 3
        assert _best_move_and_pv(tree, rid, {1})[0] == 1
        assert worker._current_best_root_action({1, 3}) == 3
    finally:
        worker.close()


@pytest.mark.parametrize('status', [0, -1])
def test_unknown_and_lost_output_keep_existing_visit_and_gumbel_order(status):
    tree, rid = _output_tree(status)
    worker = SearchWorker(_ZeroPriorEvaluator(), device='cpu')
    worker._tree, worker._root_id = tree, rid
    actions, visits = tree.get_children_visits(rid)
    try:
        assert worker._emitted_action(actions, visits, None) == 1
        assert _best_move_and_pv(tree, rid)[0] == 1
        worker._last_gumbel_action_idx = 3
        assert worker._emitted_action(actions, visits, None) == 3
    finally:
        worker.close()


def test_pv_and_ponder_continuations_preserve_a_draw_proof():
    tree = MCTSTree()
    rid = tree.add_root(1, 0.0)
    tree.expand(rid, np.array([1], np.int32), np.ones(1, np.float64))
    child = tree.find_child(rid, 1)
    tree.expand(child, np.array([2, 3], np.int32), np.array([0.8, 0.2], np.float64))
    for action, visits in [(2, 9), (3, 1)]:
        leaf = tree.find_child(child, action)
        for _ in range(visits):
            tree.backprop(np.array([rid, child, leaf], np.int32), 0.0)
        tree.mark_solved_path(np.array([rid, child, leaf], np.int32), 1 if action == 2 else 2)
    assert tree.get_solved_status(child) == 2
    assert _pv_from_root_action(tree, rid, 1) == [1, 3]
    assert _multipv_lines(tree, rid, 1, 0.0)[0][2] == [1, 3]
    assert _reply_at_child(tree, rid, 1) == 3


@pytest.mark.parametrize('status', [0, -1])
def test_unknown_and_lost_continuation_ties_keep_the_first_visit_maximum(status):
    tree = MCTSTree()
    rid = tree.add_root(1, 0.0)
    tree.expand(rid, np.array([1], np.int32), np.ones(1, np.float64))
    child = tree.find_child(rid, 1)
    actions = np.arange(2, 66, dtype=np.int32)
    tree.expand(child, actions, np.full(actions.size, 1.0 / actions.size, np.float64))
    for i, action in enumerate(actions):
        leaf = tree.find_child(child, int(action))
        for _ in range(1 if i % 3 == 0 else 2):
            tree.backprop(np.array([rid, child, leaf], np.int32), 0.0)
        if status == -1:
            tree.mark_solved_path(np.array([rid, child, leaf], np.int32), 1)
    assert tree.get_solved_status(child) == status
    assert _pv_from_root_action(tree, rid, 1) == [1, 3]
    assert _multipv_lines(tree, rid, 1, 0.0)[0][2] == [1, 3]
    assert _reply_at_child(tree, rid, 1) == 3


class _RootSeedEvaluator:
    def evaluate_encoded(self, x, relations=None):
        del relations
        wdl = np.array([[-3, 0, 3] if row[108, 0, 0] else [3, 0, -3] for row in x], np.float32)
        return np.zeros((len(x), 4672), np.float32), wdl


@pytest.mark.parametrize('budget', [1, 2, 8])
def test_fresh_puct_root_value_seed_has_a_counted_visit(budget):
    cfg = _puct_config()
    cfg.input_history_encoding = 'lc0_root_legacy_meta'
    worker = SearchWorker(_RootSeedEvaluator(), device='cpu', chunk_sims=1, n_walkers=2,
                          gumbel_cfg=cfg)
    try:
        result = worker.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(None), max_nodes=budget)
        tree, rid = worker._tree, worker._root_id
        assert tree is not None
        assert rid is not None
        assert tree.get_solved_status(rid) == 0
        expected = _value_scalar_from_wdl_logits(np.array([3, 0, -3], np.float32))
        assert tree.node_q(rid) == pytest.approx(expected)
        assert result.nodes == budget
        _, visits = tree.get_children_visits(rid)
        assert int(visits.sum()) == budget
    finally:
        worker.close()
