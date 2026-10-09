"""A carried root must retain one evaluation and repetition-history identity."""
from __future__ import annotations

import chess
import numpy as np
import pytest
from numpy.typing import NDArray

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.moves.encode import move_to_index

_POLICY = 4672


def _fixture() -> tuple[chess.Board, CBoard, chess.Board, CBoard, int, int, NDArray[np.float32]]:
    board = chess.Board()
    for uci in ("g1f3", "g8f6"):
        board.push_uci(uci)
    cb = CBoard.from_board(board)
    bare = chess.Board(board.fen())
    bare_cb = CBoard.from_board(bare)
    first = int(move_to_index(chess.Move.from_uci("f3g1"), board))
    child_board = board.copy()
    child_board.push_uci("f3g1")
    second = int(move_to_index(chess.Move.from_uci("f6g8"), child_board))
    logits = np.full(_POLICY, -50.0, dtype=np.float32)
    logits[second] = 20.0
    return board, cb, bare, bare_cb, first, second, logits


def _new_root(tree: MCTSTree, cb: CBoard) -> int:
    rid = tree.add_root(1, 0.1)
    legal = cb.legal_move_indices().astype(np.int32)
    tree.expand(rid, legal, np.full(legal.size, 1.0 / legal.size))
    return rid


def _start(
    tree: MCTSTree, cbs: list[CBoard], roots: list[int], actions: list[int], *, vloss: int = 0,
) -> int | None:
    priors = []
    for cb in cbs:
        pri = np.zeros(_POLICY, dtype=np.float64)
        legal = cb.legal_move_indices()
        pri[legal] = 1.0 / legal.size
        priors.append(pri)
    n = len(cbs)
    return tree.start_gumbel_sims(
        cbs, np.asarray(roots, dtype=np.int32),
        [[action] for action in actions],
        [np.zeros(_POLICY, dtype=np.float64) for _ in cbs], priors,
        np.ones(n, dtype=np.int32), np.full(n, 0.1, dtype=np.float64),
        0.1, 50.0, 2.5, 1.2, True,
        np.empty((32, 146, 8, 8), dtype=np.float32), vloss, 1, 0,
    )


def _visit(
    tree: MCTSTree, cb: CBoard, root: int, action: int, logits: NDArray[np.float32],
) -> None:
    pending = _start(tree, [cb], [root], [action])
    while pending is not None:
        n = int(pending)
        pending = tree.continue_gumbel_sims(
            np.repeat(logits[None, :], n, axis=0),
            np.zeros((n, 3), dtype=np.float32),
        )


@pytest.mark.parametrize("old_visits", [1, 2])
def test_native_history_replacement_refuses_cached_or_solved_descendants(old_visits: int) -> None:
    _, cb, _, bare_cb, first, second, logits = _fixture()
    tree = MCTSTree()
    root = _new_root(tree, cb)
    for _ in range(old_visits):
        _visit(tree, cb, root, first, logits)
    child = tree.find_child(root, first)
    grandchild = tree.find_child(child, second)
    assert grandchild >= 0
    expected = 0 if old_visits == 1 else 2
    assert tree.get_solved_status(grandchild) == expected
    old_q = tree.node_q(root)
    old_count = tree.node_count()

    with pytest.raises(ValueError, match="different board/history context"):
        _start(tree, [bare_cb], [root], [first])

    assert tree.get_solved_status(grandchild) == expected
    assert tree.node_q(root) == old_q
    assert tree.node_count() == old_count
    # Rejection leaves the old tree usable under the history it was built for.
    _visit(tree, cb, root, first, logits)
    assert tree.get_solved_status(grandchild) == 2


def test_repetition_result_depends_on_the_root_history() -> None:
    _, cb, _, bare_cb, first, second, logits = _fixture()
    statuses = []
    for root_cb in (cb, bare_cb):
        tree = MCTSTree()
        root = _new_root(tree, root_cb)
        for _ in range(2):
            _visit(tree, root_cb, root, first, logits)
        child = tree.find_child(root, first)
        statuses.append(tree.get_solved_status(tree.find_child(child, second)))
    assert statuses == [2, 0], "real 2-fold is solved; history-free twin is not"


class _Evaluator:
    def __init__(self, first: int, second: int) -> None:
        self.logits = np.full(_POLICY, -50.0, dtype=np.float32)
        self.logits[[first, second]] = 20.0

    def evaluate_encoded(
        self, x: NDArray[np.float32], relations: NDArray[np.uint8] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        del relations
        n = x.shape[0]
        return (
            np.repeat(self.logits[None, :], n, axis=0),
            np.zeros((n, 3), dtype=np.float32),
        )


def _run(tree: MCTSTree, board: chess.Board, root: int, first: int, second: int):
    return run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(7),
        cfg=GumbelConfig(simulations=2, topk=1, temperature=0.0, add_noise=False),
        evaluator=_Evaluator(first, second), tree=tree, root_node_ids=[root],
        target_batch=1, vloss_weight=1, allow_terminal_root_shortcuts=False,
    )


@pytest.mark.parametrize("old_visits", [1, 2])
def test_python_history_replacement_builds_a_fresh_root(old_visits: int) -> None:
    _, cb, bare, bare_cb, first, second, logits = _fixture()
    tree = MCTSTree()
    old_root = _new_root(tree, cb)
    for _ in range(old_visits):
        _visit(tree, cb, old_root, first, logits)
    old_child = tree.find_child(old_root, first)
    old_grand = tree.find_child(old_child, second)
    old_status = tree.get_solved_status(old_grand)

    result = _run(tree, bare, old_root, first, second)
    root = int(result[5][0])
    assert root != old_root
    assert tree.get_solved_status(old_grand) == old_status
    child = tree.find_child(root, first)
    grandchild = tree.find_child(child, second)
    assert grandchild >= 0
    assert tree.get_solved_status(grandchild) == 0
    assert tree.root_context_matches(root, bare_cb)

    repeated = _run(tree, bare, root, first, second)
    assert repeated[5][0] == root, "same-history searches still carry their root"


def test_advanced_root_with_matching_game_history_keeps_solved_descendants() -> None:
    board, cb, _, _, first, second, logits = _fixture()
    tree = MCTSTree()
    root = _new_root(tree, cb)
    for _ in range(2):
        _visit(tree, cb, root, first, logits)
    child = tree.find_child(root, first)
    grandchild = tree.find_child(child, second)
    assert tree.get_solved_status(grandchild) == 2
    board.push_uci("f3g1")
    next_cb = CBoard.from_board(board)
    assert tree.root_context_matches(child, next_cb)
    _visit(tree, next_cb, child, second, logits)
    assert tree.get_solved_status(grandchild) == 2


@pytest.mark.parametrize("move", ["e2e4", "g1f3"])
def test_imported_and_replayed_root_history_have_the_same_context(move: str) -> None:
    board = chess.Board()
    pushed = CBoard.from_board(board)
    pushed.push_index(int(move_to_index(chess.Move.from_uci(move), board)))
    board.push_uci(move)
    imported = CBoard.from_board(board)
    tree = MCTSTree()
    root = tree.add_root(1, 0.1, pushed)
    assert tree.root_context_matches(root, imported), (
        "root carry must accept the harmless extra pre-zeroing import hash"
    )


def test_mismatched_second_root_does_not_mutate_first_root_or_clear_virtual_loss() -> None:
    _, cb, _, bare_cb, first, _, logits = _fixture()
    tree = MCTSTree()
    roots = [_new_root(tree, cb) for _ in range(2)]
    for root in roots:
        _visit(tree, cb, root, first, logits)
    child = tree.find_child(roots[0], first)
    tree.apply_vloss_path(np.asarray([roots[0], child], dtype=np.int32))
    before = tree.get_children_q(roots[0], 99.0)
    before_q = tree.node_q(roots[0])
    before_count = tree.node_count()

    with pytest.raises(ValueError, match=r"root_id\[1\].*different board/history context"):
        _start(tree, [cb, bare_cb], roots, [first, first], vloss=1)

    assert tree.node_count() == before_count
    assert tree.node_q(roots[0]) == before_q
    assert tree.get_virtual_loss(child) == 1
    for old, new in zip(before, tree.get_children_q(roots[0], 99.0), strict=True):
        np.testing.assert_array_equal(old, new)
    assert tree.root_context_matches(roots[0], cb)
    assert tree.root_context_matches(roots[1], cb)


def test_root_context_is_bound_before_a_terminal_shortcut() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    tree = MCTSTree()
    result = run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(7),
        cfg=GumbelConfig(simulations=2, topk=1, temperature=0.0, add_noise=False),
        evaluator=_Evaluator(0, 1), tree=tree,
    )
    root = int(result[5][0])
    assert root >= 0
    assert tree.root_context_matches(root, CBoard.from_board(board))
    changed = chess.Board(board.fen())
    changed.halfmove_clock = 1
    assert not tree.root_context_matches(root, CBoard.from_board(changed)), (
        "mate shortcut skipped native start, but must still bind the root context"
    )


def test_forced_collapse_descendant_derives_context_from_bound_ancestor() -> None:
    board = chess.Board("k7/8/2K5/8/8/8/8/3Q4 w - - 0 1")
    cb = CBoard.from_board(board)
    first = int(move_to_index(chess.Move.from_uci("d1a1"), board))
    tree = MCTSTree()
    root = _new_root(tree, cb)
    logits = np.zeros(_POLICY, dtype=np.float32)
    _visit(tree, cb, root, first, logits)

    board.push_uci("d1a1")
    assert len(list(board.legal_moves)) == 1
    child = tree.find_child(root, first)
    assert tree.is_expanded(child), "native forced collapse expanded the reply"
    second = int(move_to_index(next(iter(board.legal_moves)), board))
    grandchild = tree.find_child(child, second)
    assert grandchild >= 0
    assert tree.is_expanded(grandchild)
    # Collapse caches the evaluated grandchild, but not the expanded child.
    history_cb = CBoard.from_board(board)
    bare = chess.Board(board.fen())
    bare_cb = CBoard.from_board(bare)
    assert tree.root_context_matches(child, history_cb)
    assert not tree.root_context_matches(child, bare_cb)
    with pytest.raises(ValueError, match="different board/history context"):
        _start(tree, [bare_cb], [child], [second])

    result = _run(tree, bare, child, second, second)
    assert result[5][0] != child, "single-legal shortcut must rebuild changed history"
    _visit(tree, history_cb, child, second, logits)
    assert tree.root_context_matches(child, history_cb)


def test_refused_history_replacement_preserves_the_pending_continuation() -> None:
    _, cb, _, bare_cb, first, _, logits = _fixture()
    tree = MCTSTree()
    root = _new_root(tree, cb)
    pending = _start(tree, [cb], [root], [first], vloss=1)
    assert pending is not None
    child = tree.find_child(root, first)
    assert tree.get_virtual_loss(child) == 1
    before_q = tree.node_q(root)

    with pytest.raises(ValueError, match="different board/history context"):
        _start(tree, [bare_cb], [root], [first], vloss=1)

    assert tree.get_virtual_loss(child) == 1
    assert tree.node_q(root) == before_q
    n = int(pending)
    completed = tree.continue_gumbel_sims(
        np.repeat(logits[None, :], n, axis=0),
        np.zeros((n, 3), dtype=np.float32),
    )
    assert completed is None
    assert tree.get_virtual_loss(child) == 0
    _, visits = tree.get_children_visits(root)
    assert int(np.sum(visits)) == 1, "the original pending visit completed once"


def test_unbound_duplicate_root_requires_one_history_context() -> None:
    _, cb, _, bare_cb, first, _, logits = _fixture()
    tree = MCTSTree()
    root = _new_root(tree, cb)
    before_count = tree.node_count()
    before_q = tree.node_q(root)

    with pytest.raises(ValueError, match="different board/history context"):
        _start(tree, [cb, bare_cb], [root, root], [first, first])

    assert tree.node_count() == before_count
    assert tree.node_q(root) == before_q
    _, visits = tree.get_children_visits(root)
    assert int(np.sum(visits)) == 0

    # Compatible duplicate roots remain supported by the raw batched API.
    pending = _start(tree, [cb, cb.copy()], [root, root], [first, first])
    while pending is not None:
        n = int(pending)
        pending = tree.continue_gumbel_sims(
            np.repeat(logits[None, :], n, axis=0),
            np.zeros((n, 3), dtype=np.float32),
        )
    _, visits = tree.get_children_visits(root)
    assert int(np.sum(visits)) == 2
    assert tree.root_context_matches(root, cb)
