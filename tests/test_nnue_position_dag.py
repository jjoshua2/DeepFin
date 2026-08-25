"""Correctness gates for the reusable structural-position DAG + NNUE payload.

The DAG is intentionally a different abstraction from ``MCTSTree``:

* one structural chess position is one node;
* nodes have no parent pointer and can have multiple incoming edges;
* path-specific halfmove/repetition/history state is NOT node identity;
* each new NNUE node owns one incremental accumulator state and is evaluated at
  most once; a transposition hit reuses that state/value without make/evaluate.

The dense synthetic PSQT pack makes accumulator mistakes visible without needing
the real 100+ MB net in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.nnue import _nnue_ext
from scripts import nnue_parse
from tests.test_nnue_native_eval import write_synthetic_pack


@pytest.fixture(scope="module")
def dag_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(20260825)
    halfka = rng.integers(
        -32,
        33,
        size=nnue_parse.HALFKA_DIMS * nnue_parse.PSQT_BUCKETS,
        dtype=np.int32,
    )
    threats = rng.integers(
        -32,
        33,
        size=nnue_parse.THREAT_DIMS * nnue_parse.PSQT_BUCKETS,
        dtype=np.int32,
    )
    path = tmp_path_factory.mktemp("nnue-dag") / "dense-psqt.pack"
    write_synthetic_pack(
        path,
        blobs={
            "ft_psqt": [(0, halfka)],
            "threat_psqt": [(0, threats)],
        },
    )
    return path


def _cboard(board: chess.Board) -> CBoard:
    return CBoard.from_board(board)


def _push(
    dag: object,
    weights: object,
    parent_id: int,
    board: chess.Board,
    uci: str,
) -> tuple[int, chess.Board, bool]:
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    action = move_to_index(move, board)
    child = board.copy(stack=True)
    child.push(move)
    child_cb = _cboard(child)
    node_id, value, created = _nnue_ext.dag_intern_child(
        dag,
        parent_id,
        action,
        child_cb,
    )
    if not child.is_check():
        # ``child.is_check()`` asks whether the NEW side to move is in check,
        # exactly the condition under which the static evaluator refuses.
        assert value == _nnue_ext.evaluate(weights, child_cb)
    else:
        assert value is None
    return node_id, child, created


def test_transposed_move_orders_share_one_real_node_and_one_evaluation(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights, 16)

    start = chess.Board()
    root, root_value, created = _nnue_ext.dag_intern_root(dag, _cboard(start))
    assert created is True
    assert root_value == _nnue_ext.evaluate(weights, _cboard(start))

    # Same final structural position through two independent move orders.
    seq_a = ("g1f3", "g8f6", "g2g3", "g7g6")
    seq_b = ("g2g3", "g7g6", "g1f3", "g8f6")

    board_a = start.copy(stack=True)
    node_a = root
    penultimate_a = -1
    for i, uci in enumerate(seq_a):
        if i == len(seq_a) - 1:
            penultimate_a = node_a
        node_a, board_a, made = _push(dag, weights, node_a, board_a, uci)
        assert made is True

    board_b = start.copy(stack=True)
    node_b = root
    penultimate_b = -1
    last_created = True
    for i, uci in enumerate(seq_b):
        if i == len(seq_b) - 1:
            penultimate_b = node_b
        node_b, board_b, last_created = _push(dag, weights, node_b, board_b, uci)

    assert board_a.board_fen() == board_b.board_fen()
    assert board_a.turn == board_b.turn
    assert board_a.castling_rights == board_b.castling_rights
    assert node_b == node_a
    assert last_created is False

    # This is the defining difference from the current MCTSTree transposition
    # helper: two parents point to ONE child id; there is no cloned recipient.
    children_a = dict(_nnue_ext.dag_children(dag, penultimate_a))
    children_b = dict(_nnue_ext.dag_children(dag, penultimate_b))
    assert node_a in children_a.values()
    assert node_a in children_b.values()

    stats = _nnue_ext.dag_stats(dag)
    # root + 4 nodes on A + only 3 new nodes on B; B's fourth is the transposition.
    assert stats["node_count"] == 8
    assert stats["edge_count"] == 8
    assert stats["state_inits"] == 1
    assert stats["state_makes"] == 7
    assert stats["nnue_evals"] == 8
    assert stats["node_reuses"] >= 1
    assert stats["hits"] >= 1
    assert stats["nnue_evals"] == stats["node_count"]


def test_structural_identity_excludes_draw_history_but_includes_usable_ep(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights)

    ordinary = chess.Board()
    draw_adjacent = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99 50",
    )
    a, _, a_created = _nnue_ext.dag_intern_root(dag, _cboard(ordinary))
    b, _, b_created = _nnue_ext.dag_intern_root(dag, _cboard(draw_adjacent))
    assert a_created is True
    assert b_created is False
    assert a == b

    # But an actually exercisable EP right changes legal structure and therefore
    # must be a different canonical node.
    ep = chess.Board(
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    )
    no_ep = chess.Board(
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",
    )
    assert ep.has_legal_en_passant()
    ep_id, _, _ = _nnue_ext.dag_intern_root(dag, _cboard(ep))
    no_ep_id, _, _ = _nnue_ext.dag_intern_root(dag, _cboard(no_ep))
    assert ep_id != no_ep_id


def test_action_child_mismatch_is_rejected_before_graph_or_nnue_mutates(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights)
    board = chess.Board()
    root, _, _ = _nnue_ext.dag_intern_root(dag, _cboard(board))
    before = _nnue_ext.dag_stats(dag)

    asked_move = chess.Move.from_uci("e2e4")
    wrong_child = board.copy(stack=True)
    wrong_child.push(chess.Move.from_uci("d2d4"))
    action = move_to_index(asked_move, board)

    with pytest.raises(ValueError, match="does not produce the supplied child"):
        _nnue_ext.dag_intern_child(dag, root, action, _cboard(wrong_child))

    after = _nnue_ext.dag_stats(dag)
    assert after["node_count"] == before["node_count"]
    assert after["edge_count"] == before["edge_count"]
    assert after["state_makes"] == before["state_makes"]
    assert after["nnue_evals"] == before["nnue_evals"]


def test_in_check_node_has_state_but_no_fake_static_value(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights)

    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
    assert board.is_check()
    root, value, created = _nnue_ext.dag_intern_root(dag, _cboard(board))
    assert created is True
    assert value is None
    assert _nnue_ext.dag_value(dag, root) is None
    stats = _nnue_ext.dag_stats(dag)
    assert stats["state_inits"] == 1
    assert stats["nnue_evals"] == 0

    # The accumulator state is still useful: a legal evasion derives its child
    # incrementally and the resolved non-check child gets a real static value.
    move = next(iter(board.legal_moves))
    action = move_to_index(move, board)
    child = board.copy(stack=True)
    child.push(move)
    child_cb = _cboard(child)
    child_id, child_value, child_created = _nnue_ext.dag_intern_child(
        dag,
        root,
        action,
        child_cb,
    )
    assert child_created is True
    assert child_value == _nnue_ext.evaluate(weights, child_cb)
    assert _nnue_ext.dag_value(dag, child_id) == child_value
    stats = _nnue_ext.dag_stats(dag)
    assert stats["state_makes"] == 1
    assert stats["nnue_evals"] == 1


def test_reroot_and_reset_preserve_allocations_but_clear_graph_semantics(dag_pack: Path) -> None:
    weights = _nnue_ext.load(str(dag_pack))
    dag = _nnue_ext.dag_open(weights, 16)
    board = chess.Board()
    root, _, _ = _nnue_ext.dag_intern_root(dag, _cboard(board))
    child, child_board, _ = _push(dag, weights, root, board, "e2e4")

    _nnue_ext.dag_set_root(dag, child)
    before = _nnue_ext.dag_stats(dag)
    assert before["root_id"] == child
    allocated = before["memory_bytes"]

    _nnue_ext.dag_reset(dag)
    cleared = _nnue_ext.dag_stats(dag)
    assert cleared["root_id"] == -1
    assert cleared["node_count"] == 0
    assert cleared["edge_count"] == 0
    assert cleared["state_inits"] == 0
    assert cleared["state_makes"] == 0
    assert cleared["nnue_evals"] == 0
    assert cleared["memory_bytes"] == allocated

    new_root, new_value, created = _nnue_ext.dag_intern_root(dag, _cboard(child_board))
    assert new_root == 0
    assert created is True
    assert new_value == _nnue_ext.evaluate(weights, _cboard(child_board))


@pytest.mark.skipif(not os.environ.get("CAE_NNUE_TEST_PACK"), reason="needs real NNUE pack")
def test_real_net_dag_incremental_values_match_full_refresh() -> None:
    weights = _nnue_ext.load(os.environ["CAE_NNUE_TEST_PACK"])
    dag = _nnue_ext.dag_open(weights)
    board = chess.Board()
    node, value, _ = _nnue_ext.dag_intern_root(dag, _cboard(board))
    assert value == _nnue_ext.evaluate(weights, _cboard(board))
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"):
        node, board, _ = _push(dag, weights, node, board, uci)
        assert _nnue_ext.dag_value(dag, node) == _nnue_ext.evaluate(weights, _cboard(board))
