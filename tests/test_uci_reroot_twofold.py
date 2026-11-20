"""Tree reuse never adopts a root whose solved status the kept subtree does
not re-derive, nor an unexpanded one.

The compiled search marks a position that repeats one from the game history
(or the tree path) DRAW and never expands it. `SearchWorker.advance_root`
used to make such a node the root with its mark: the PUCT pool and walker
roots then stayed DRAW through `go nodes 2000` with almost no child proven,
and a hand-marked expanded node made the classic runner play the repeating
move with `score cp 0`. Self-play reuse is gated in `gumbel_c.py`
(`is_expanded` + root context); `_reroot_refusal` mirrors it for UCI.
"""
from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import chess
import numpy as np
import pytest

import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchWorker, _reroot_refusal
from chess_anti_engine.uci.time_manager import Deadline
from tests.test_uci_solved_root_stops import _make_worker, _mark
from tests.test_uci_time_forfeits import _Out

# (start FEN, game moves before the search, the move onto the natural twofold).
# The searched side's move repeats a position from the game history.
_NATURAL = {
    # 1.Nf3 Nf6 2.Ne5 Ng8: 3.Nf3 repeats the position after 1.Nf3.
    "knight_shuffle": (chess.STARTING_FEN, ["g1f3", "g8f6", "f3e5", "f6g8"], "e5f3"),
    # KR vs K: Ra2 Kd8 Ra1 Ke8, and Ra2 repeats the position after the first Ra2.
    "kr_vs_k": ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", ["a1a2", "e8d8", "a2a1", "d8e8"], "a1a2"),
}


def _searched(worker: SearchWorker, board: chess.Board, nodes: int = 2000) -> None:
    worker.run(board.copy(), stop_event=threading.Event(), deadline=Deadline(60_000),
               max_nodes=nodes, max_depth=None)


def _reroot_like_the_engine(worker: SearchWorker, board: chess.Board,
                            moves: list[chess.Move]) -> bool:
    reused = worker.advance_root(board.copy(), moves)
    if not reused:
        worker.reset_tree()  # what `Engine._sync_tree_root` does
    return reused


def _assert_root_not_stale(worker: SearchWorker) -> None:
    tree, rid = worker._tree, worker._root_id
    assert tree is not None
    assert rid is not None
    status = int(tree.get_solved_status(rid))
    assert status == 0 or _reroot_refusal(tree, rid) is None, (
        f"root reads solved {status} with its children not proving it")


@pytest.mark.parametrize("case", sorted(_NATURAL))
@pytest.mark.parametrize("path", ["walkers", "pool"])
def test_a_natural_twofold_leaf_is_not_adopted_as_the_root(path: str, case: str) -> None:
    fen, history, repeat = _NATURAL[case]
    worker = _make_worker(path, seed=0)
    try:
        board = chess.Board(fen)
        for uci in history:
            board.push_uci(uci)
        _searched(worker, board)
        tree, rid = worker._tree, worker._root_id
        assert tree is not None
        assert rid is not None
        move = chess.Move.from_uci(repeat)
        leaf = int(tree.find_child(rid, int(move_to_index(move, board))))
        assert leaf >= 0, "fixture: the repeating move was searched"
        assert int(tree.get_solved_status(leaf)) == 2, "fixture: marked as a twofold draw"
        assert not tree.is_expanded(leaf), "fixture: a natural twofold is never expanded"
        after = board.copy()
        after.push(move)
        assert after.is_repetition(2)
        reused = _reroot_like_the_engine(worker, board, [move])
        _searched(worker, after)
        _assert_root_not_stale(worker)
        assert not reused
        assert worker.reroots_refused == {"unexpanded": 1}
    finally:
        worker.close()


def _reroot_onto_a_marked_twofold_leaf(worker: SearchWorker) -> tuple[bool, chess.Board]:
    """1.Nf3 Nf6, search, mark the Ng1-Ng8 grandchild DRAW and give it children
    if it has none (an expanded node with a stale mark), then reroot onto it."""
    b0 = chess.Board()
    for u in ("g1f3", "g8f6"):
        b0.push_uci(u)
    _searched(worker, b0)
    tree, rid = worker._tree, worker._root_id
    assert tree is not None
    assert rid is not None
    m1, m2 = chess.Move.from_uci("f3g1"), chess.Move.from_uci("f6g8")
    b1 = b0.copy()
    b1.push(m1)
    child = int(tree.find_child(rid, int(move_to_index(m1, b0))))
    assert child >= 0, "fixture: Ng1 was searched"
    if len(tree.get_children_visits(child)[0]) == 0:
        # The classic search visits Ng1 once and leaves it unexpanded.
        tree.expand(child, np.array([move_to_index(m2, b1)], dtype=np.int32), np.array([1.0]))
    leaf = int(tree.find_child(child, int(move_to_index(m2, b1))))
    assert leaf >= 0, "fixture: Ng8 is in the tree"
    b2 = b1.copy()
    b2.push(m2)
    if len(tree.get_children_visits(leaf)[0]) == 0:
        replies = [move_to_index(m, b2) for m in b2.legal_moves]
        tree.expand(leaf, np.array(replies, dtype=np.int32),
                    np.full(len(replies), 1.0 / len(replies)))
    _mark(tree, leaf, 2)
    return _reroot_like_the_engine(worker, b0, [m1, m2]), b2


@pytest.mark.parametrize("path", ["classic", "walkers", "pool"])
def test_an_expanded_root_with_a_stale_draw_mark_is_not_reused(path: str) -> None:
    worker = _make_worker(path, seed=0)
    try:
        reused, b2 = _reroot_onto_a_marked_twofold_leaf(worker)
        worker.run(b2, stop_event=threading.Event(), deadline=Deadline(10_000), max_nodes=256)
        _assert_root_not_stale(worker)
        assert not reused
        assert worker.reroots_refused == {"stale_status": 1}
    finally:
        worker.close()


def _one_ply_tree(child_statuses: list[int], root_status: int) -> tuple[MCTSTree, int]:
    board = chess.Board()
    moves = list(board.legal_moves)[:len(child_statuses)]
    tree = MCTSTree()
    tree.reserve(64, 640)
    rid = int(tree.add_root(1, 0.0))
    idx = [move_to_index(m, board) for m in moves]
    tree.expand(rid, np.array(idx, dtype=np.int32), np.full(len(idx), 1.0 / len(idx)))
    for a, status in zip(idx, child_statuses):
        if status:
            _mark(tree, int(tree.find_child(rid, int(a))), status)
    if root_status:
        _mark(tree, rid, root_status)
    return tree, rid


@pytest.mark.parametrize(("root", "children", "expected"), [
    (0, [0, 0], None),
    (2, [2, 1], None),                # drawn: a drawn child, the rest won by the opponent
    (2, [2, 2], None),                # drawn: every child drawn
    (2, [2, 0], "stale_status"),      # drawn with an unproven child
    (2, [2, -1], "stale_status"),     # mis-marked: a child proven lost makes it WON
    (2, [1, 1], "stale_status"),      # mis-marked: no drawn child makes it LOST
    (1, [0, -1], None),               # won: a child proven lost
    (1, [0, 2], "stale_status"),      # won with no losing child
    (-1, [1, 1], None),               # lost: every child proven won
    (-1, [1, 0], "stale_status"),     # lost with an unproven child
    (-1, [1, 2], "stale_status"),     # mis-marked: a drawn child makes it DRAWN
    (1, [2, 2], "stale_status"),      # mis-marked: a drawn node
])
def test_a_solved_root_is_reused_only_when_its_children_re_derive_it(
    root: int, children: list[int], expected: str | None,
) -> None:
    tree, rid = _one_ply_tree(children, root)
    assert _reroot_refusal(tree, rid) == expected


def test_an_unexpanded_node_is_refused() -> None:
    tree = MCTSTree()
    tree.reserve(8, 8)
    assert _reroot_refusal(tree, int(tree.add_root(1, 0.0))) == "unexpanded"


def test_a_genuinely_proven_draw_root_is_still_reused() -> None:
    """A node drawn on its children's proofs (one reply drawn, the other won
    by the replying side, as `tree_resolve_from_children` derives DRAW) is
    re-derived from the kept subtree: the reroot is kept."""
    worker = _make_worker("classic", seed=0)
    try:
        b0 = chess.Board()
        tree = MCTSTree()
        tree.reserve(64, 640)
        rid = int(tree.add_root(1, 0.0))
        e4 = chess.Move.from_uci("e2e4")
        tree.expand(rid, np.array([move_to_index(e4, b0)], dtype=np.int32), np.array([1.0]))
        child = int(tree.find_child(rid, int(move_to_index(e4, b0))))
        b1 = b0.copy()
        b1.push(e4)
        replies = [move_to_index(m, b1) for m in list(b1.legal_moves)[:2]]
        tree.expand(child, np.array(replies, dtype=np.int32), np.array([0.5, 0.5]))
        _mark(tree, int(tree.find_child(child, int(replies[0]))), 2)
        _mark(tree, int(tree.find_child(child, int(replies[1]))), 1)
        _mark(tree, child, 2)
        worker._tree, worker._root_id, worker._tree_fen = tree, rid, b0.fen()
        assert worker.advance_root(b0, [e4])
        assert worker._root_id == child
        assert worker.reroots_refused == {}
    finally:
        worker.close()


# -- engine level: the callers reset the tree on a refusal ------------------------


@pytest.fixture
def out(monkeypatch) -> _Out:
    capture = _Out()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=capture, stderr=sys.stderr))
    return capture


def _count_resets(worker: SearchWorker) -> list[int]:
    calls = [0]
    real = worker.reset_tree

    def counted() -> None:
        calls[0] += 1
        real()

    worker.reset_tree = counted  # type: ignore[method-assign]
    return calls


def _twofold_leaf(worker: SearchWorker, board: chess.Board, move: chess.Move) -> int:
    tree, rid = worker._tree, worker._root_id
    if tree is None or rid is None:
        return -1
    return int(tree.find_child(rid, int(move_to_index(move, board))))


def _assert_natural_twofold(worker: SearchWorker, board: chess.Board, move: chess.Move) -> None:
    tree = worker._tree
    leaf = _twofold_leaf(worker, board, move)
    assert tree is not None
    assert leaf >= 0, "fixture: the repeating move was searched"
    assert int(tree.get_solved_status(leaf)) == 2, "fixture: marked as a twofold draw"
    assert not tree.is_expanded(leaf), "fixture: a natural twofold is never expanded"


def test_position_onto_a_natural_twofold_resets_the_tree(out: _Out) -> None:
    """`position` + `go`, then `position` extended by the move onto a natural
    twofold + `go`: `_sync_tree_root` gets the refusal and resets the tree;
    the search then runs on a fresh root and answers a legal move."""
    fen, history, repeat = _NATURAL["knight_shuffle"]
    worker = _make_worker("walkers", seed=0)
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command(f"position fen {fen} moves {' '.join(history)}"))
        engine.dispatch(parse_command("go nodes 2000"))
        assert len(out.wait_bestmoves(1, timeout=120)) == 1, out.text()
        board = chess.Board(fen)
        for uci in history:
            board.push_uci(uci)
        move = chess.Move.from_uci(repeat)
        _assert_natural_twofold(worker, board, move)
        resets = _count_resets(worker)
        engine.dispatch(parse_command(
            f"position fen {fen} moves {' '.join([*history, repeat])}"))
        engine.dispatch(parse_command("go nodes 256"))
        answers = out.wait_bestmoves(2, timeout=120)
        assert len(answers) == 2, out.text()
        after = board.copy()
        after.push(move)
        assert chess.Move.from_uci(answers[1]) in after.legal_moves
        assert worker.reroots_refused == {"unexpanded": 1}
        assert resets[0] == 1, "the refused reroot must reset the tree"
        assert worker._tree_fen == after.fen()
        _assert_root_not_stale(worker)
    finally:
        engine.close()


def test_ponderhit_onto_a_natural_twofold_resets_the_tree(out: _Out) -> None:
    """Pondering on the opponent's repeating reply (KR vs K: ... Kd8-e8 repeats
    the start), then `ponderhit`: the ponderhit caller gets the refusal and
    resets the tree before searching the hit position."""
    fen = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    history = ["a1a2", "e8d8", "a2a1"]
    predicted = chess.Move.from_uci("d8e8")
    worker = _make_worker("walkers", seed=0)
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command(
            f"position fen {fen} moves {' '.join([*history, predicted.uci()])}"))
        engine.dispatch(parse_command("go ponder wtime 4000 btime 4000"))
        board = chess.Board(fen)
        for uci in history:
            board.push_uci(uci)
        tree_seen = None
        end = time.monotonic() + 60
        while time.monotonic() < end:  # until the ponder search marks the repeat
            tree_seen = worker._tree
            leaf = _twofold_leaf(worker, board, predicted)
            if tree_seen is not None and leaf >= 0 and int(tree_seen.get_solved_status(leaf)) == 2:
                break
            time.sleep(0.05)
        _assert_natural_twofold(worker, board, predicted)
        resets = _count_resets(worker)
        engine.dispatch(parse_command("ponderhit"))
        answers = out.wait_bestmoves(1, timeout=120)
        assert len(answers) == 1, out.text()
        after = board.copy()
        after.push(predicted)
        assert chess.Move.from_uci(answers[0]) in after.legal_moves
        assert worker.reroots_refused == {"unexpanded": 1}
        assert resets[0] == 1, "the refused reroot must reset the tree"
        assert worker._tree_fen == after.fen()
    finally:
        engine.close()
