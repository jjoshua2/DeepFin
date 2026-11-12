"""Actual walker descendants, with controlled tables and no network construction."""
from __future__ import annotations

import threading

import chess
import chess.syzygy
import numpy as np
import pytest

from chess_anti_engine import tablebase as tbmod
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.moves.encode import move_to_index
from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig


class RawEvaluator:
    def __init__(self):
        self.raw = np.zeros((1, 3), np.float32)
        self.raw.flags.writeable = False

    def evaluate_encoded(self, x, relations=None):
        assert relations is None
        return np.zeros((len(x), 4672), np.float32), np.broadcast_to(self.raw, (len(x), 3))


def branch(board, move):
    tree = MCTSTree()
    tree.reserve(512, 4096)
    root = tree.add_root(0, 0.0)
    action = move_to_index(chess.Move.from_uci(move), board)
    tree.expand(root, np.array([action], np.int32), np.array([1.0], np.float64))
    return tree, root


class Tables:
    def __init__(self, value):
        self.wdl = {"KPvK": object()}
        self.dtz = {}
        self.value = value
        self.positions = []

    def probe_wdl(self, board):
        self.positions.append(board.fen())
        if self.value is None:
            raise chess.syzygy.MissingTableError("controlled missing material")
        return self.value


@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize(("wdl", "status"), [(2, 1), (-2, -1), (0, 2), (1, 2), (-1, 2)])
def test_actual_descendant_override_sign_and_solved_state(monkeypatch, turn, wdl, status):
    board = chess.Board(
        "7k/8/8/8/8/8/P7/K7 w - - 0 1" if turn else
        "k7/p7/8/8/8/8/8/7K b - - 0 1"
    )
    move = "a2a3" if turn else "a7a6"
    expected = board.copy(stack=True)
    expected.push_uci(move)
    tables = Tables(wdl)
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: tables)
    probe = tbmod.SyzygyProbe("controlled", max_pieces=3, cursed_as_draw=True)
    evaluator = RawEvaluator()
    tree, root = branch(board, move)
    with WalkerPool(WalkerPoolConfig(1, 1.5, 0.0, 0.33), evaluator) as pool:
        assert pool.run(tree=tree, root_id=root, root_cboard=CBoard.from_board(board),
                        target_sims=1, stop_event=threading.Event(), tb_probe=probe) == 1
    # No root probe exists in this direct route; the observed board is the child.
    assert tables.positions == [expected.fen()]
    assert (probe.probes, probe.hits) == (1, 1)
    assert tree.get_solved_status(1) == status
    assert tree.get_solved_status(root) == (-status if status != 2 else 2)
    assert tree.node_q(root) == pytest.approx(-wdl / abs(wdl) if abs(wdl) == 2 else 0, abs=1e-6)
    assert np.array_equal(evaluator.raw, np.zeros((1, 3), np.float32))
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


@pytest.mark.parametrize(("move", "missing"), [("a1b1", False), ("a2a3", True)])
def test_history_sensitive_or_missing_leaf_keeps_raw_value(monkeypatch, move, missing):
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    tables = Tables(None if missing else 2)
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: tables)
    probe = tbmod.SyzygyProbe("controlled", max_pieces=3)
    tree, root = branch(board, move)
    with WalkerPool(WalkerPoolConfig(1, 1.5, 0.0, 0.33), RawEvaluator()) as pool:
        pool.run(tree=tree, root_id=root, root_cboard=CBoard.from_board(board),
                 target_sims=1, stop_event=threading.Event(), tb_probe=probe)
    assert len(tables.positions) == int(missing)
    assert probe.hits == 0
    assert tree.get_solved_status(1) == 0
    assert tree.node_q(root) == pytest.approx(0)
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


def test_native_board_is_after_forced_chain_and_old_tuple_is_unchanged():
    board = chess.Board("k7/8/2K5/8/8/8/8/3Q4 w - - 0 1")
    expected = board.copy(stack=True)
    expected.push_uci("d1a1")
    expected.push_uci("a8b8")
    tree, root = branch(board, "d1a1")
    enc = np.empty((1, 146, 8, 8), np.float32)
    leaf, path, legal, terminal, cb = tree.walker_descend_puct_tb(
        root, CBoard.from_board(board), 1.5, 0.0, 0.33, 0, enc, 3,
    )
    assert terminal is None
    assert len(path) == 3
    assert path[-1] == leaf
    assert cb.fen() == expected.fen()
    assert set(legal) == set(CBoard.from_board(expected).legal_move_indices())
    assert len(tree.walker_descend_puct(root, CBoard.from_board(board),
                                      1.5, 0.0, 0.33, 0, enc)) == 4


def test_probe_failure_unwinds_actual_walker_virtual_loss(monkeypatch):
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: Tables(2))
    probe = tbmod.SyzygyProbe("controlled", max_pieces=3)

    def fail(*_args, **_kwargs):
        raise RuntimeError("controlled probe failure")

    monkeypatch.setattr(probe, "apply", fail)
    tree, root = branch(board, "a2a3")
    with (
        WalkerPool(WalkerPoolConfig(2, 1.5, 0.0, 0.33), RawEvaluator()) as pool,
        pytest.raises(RuntimeError, match="controlled probe failure"),
    ):
        pool.run(tree=tree, root_id=root, root_cboard=CBoard.from_board(board),
                 target_sims=2, stop_event=threading.Event(), tb_probe=probe)
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


def test_gather_keeps_sparse_leaf_row_and_raw_cache_separate(monkeypatch):
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    tables = Tables(-2)
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: tables)
    probe = tbmod.SyzygyProbe("controlled", max_pieces=3)
    tree = MCTSTree()
    tree.reserve(512, 4096)
    root = tree.add_root(0, 0.0)
    actions = np.array([move_to_index(chess.Move.from_uci(m), board)
                        for m in ("a2a3", "a1b1")], np.int32)
    tree.expand(root, actions, np.array([0.5, 0.5], np.float64))
    evaluator = RawEvaluator()
    with WalkerPool(WalkerPoolConfig(1, 1.5, 0.0, 0.33, gather=2), evaluator) as pool:
        assert pool.run(tree=tree, root_id=root, root_cboard=CBoard.from_board(board),
                        target_sims=2, stop_event=threading.Event(), tb_probe=probe) == 2
    assert len(tables.positions) == 1
    assert tree.get_solved_status(1) == -1
    assert tree.get_solved_status(2) == 0
    # One proven losing child for its STM proves a winning root immediately;
    # node_q exposes that solved value, rather than the visit-average 0.5.
    assert tree.node_q(1) == pytest.approx(-1, abs=1e-6)
    assert tree.node_q(2) == pytest.approx(0)
    assert tree.get_solved_status(root) == 1
    assert tree.node_q(root) == pytest.approx(1)
    assert np.array_equal(evaluator.raw, np.zeros((1, 3), np.float32))
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


@pytest.mark.parametrize(("fen", "move", "capacity", "terminal"), [
    (chess.STARTING_FEN, "e2e4", 3, False),
    ("4k3/8/8/8/8/8/P7/4K2R w K - 0 1", "a2a3", 4, False),
    ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "f7g7", 3, True),
])
def test_native_filter_and_terminal_tuple(fen, move, capacity, terminal):
    board = chess.Board(fen)
    tree, root = branch(board, move)
    result = tree.walker_descend_puct_tb(
        root, CBoard.from_board(board), 1.5, 0.0, 0.33, 3,
        np.empty((1, 146, 8, 8), np.float32), capacity,
    )
    assert len(result) == 5
    assert result[4] is None
    assert (result[3] is not None) == terminal
    if not terminal:
        tree.remove_vloss_path(result[1])
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))
