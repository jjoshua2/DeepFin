"""The opt-in census describes cached occupancy without changing real search.

These counts are deliberately not evaluation-request or simulation counts:
bound roots and TT recipients occupy the same cache as evaluated leaves.
"""
from __future__ import annotations

import chess
import numpy as np
import pytest
from numpy.typing import NDArray

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts import _mcts_tree
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.moves.encode import move_to_index

_POLICY_SIZE = 4672
_KEYS = {
    "scope", "complete", "encoded_dtype", "input_history_mode", "n_extra",
    "max_cached_boards", "cached_boards", "unique_structural_positions",
    "structural_duplicate_nodes", "unique_search_contexts",
    "same_context_duplicate_nodes", "unique_encoded_inputs",
    "same_input_duplicate_nodes", "positions_with_multiple_contexts",
    "max_contexts_per_position",
}


def _census(tree: MCTSTree, **overrides: int) -> dict[str, int | str | bool]:
    result = tree.transposition_census(
        **{"input_history_mode": 2, "n_extra": 63, **overrides},
    )
    # Pins every Py_BuildValue entry, including the trailing maxima. A missing
    # format specifier must not silently drop one of the deciding measurements.
    assert set(result) == _KEYS
    assert result["scope"] == "all_cached_boards"
    assert result["complete"] is True
    assert result["encoded_dtype"] == "reconstructed_float32_planes"
    n = int(result["cached_boards"])
    for unique, duplicate in (
        ("unique_structural_positions", "structural_duplicate_nodes"),
        ("unique_search_contexts", "same_context_duplicate_nodes"),
        ("unique_encoded_inputs", "same_input_duplicate_nodes"),
    ):
        assert int(result[unique]) + int(result[duplicate]) == n
    return result


def _tree_with(*boards: chess.Board | CBoard) -> MCTSTree:
    tree = MCTSTree()
    for board in boards:
        cb = board if isinstance(board, CBoard) else CBoard.from_board(board)
        tree.add_root(1, 0.0, cb)
    return tree


def _after(*moves: str) -> chess.Board:
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return board


def test_different_move_orders_share_structure_but_not_history() -> None:
    first = _after("g1f3", "g8f6", "g2g3", "g7g6")
    second = _after("g2g3", "g7g6", "g1f3", "g8f6")
    assert first.board_fen() == second.board_fen()
    tree = _tree_with(first, first.copy(stack=True), second)
    stats = _census(tree)
    assert stats["cached_boards"] == 3
    assert stats["unique_structural_positions"] == 1
    assert stats["unique_search_contexts"] == 2
    assert stats["unique_encoded_inputs"] == 2
    assert stats["same_context_duplicate_nodes"] == 1
    assert stats["positions_with_multiple_contexts"] == 1
    assert stats["max_contexts_per_position"] == 2


def test_unusable_ep_is_structural_noise_but_the_whole_encoded_row_is_counted() -> None:
    with_ep = chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    without_ep = chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )
    tree = _tree_with(with_ep, without_ep)
    legacy_meta = _census(tree, input_history_mode=2)
    lc0_root = _census(tree, input_history_mode=1)
    assert legacy_meta["unique_structural_positions"] == 1
    assert legacy_meta["unique_search_contexts"] == 2
    assert legacy_meta["unique_encoded_inputs"] == 2
    # Root encoding has no raw-EP metadata plane in its first 112 planes, but
    # extra feature plane 141 STILL carries EP. Hashing only the LC0 prefix
    # would falsely report a duplicate model input here.
    first = CBoard.from_board(with_ep).encode_full(1, 63)
    second = CBoard.from_board(without_ep).encode_full(1, 63)
    np.testing.assert_array_equal(first[:112], second[:112])
    assert np.any(first[112:] != second[112:])
    assert lc0_root["unique_search_contexts"] == 2
    assert lc0_root["unique_encoded_inputs"] == 2


@pytest.mark.parametrize("field", ["turn", "castling", "usable_ep"])
def test_move_relevant_state_is_not_merged(field: str) -> None:
    base = chess.Board(
        "rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3"
    )
    other = base.copy(stack=True)
    if field == "turn":
        other.turn = not other.turn
    elif field == "castling":
        other.castling_rights = 0
    else:
        assert base.has_legal_en_passant()
        other.ep_square = None
    stats = _census(_tree_with(base, other))
    assert stats["unique_structural_positions"] == 2
    assert stats["unique_search_contexts"] == 2


def test_fullmove_bookkeeping_does_not_split_identical_contexts() -> None:
    first = chess.Board()
    second = first.copy(stack=True)
    second.fullmove_number = 30
    assert CBoard.from_board(first).ply != CBoard.from_board(second).ply
    tree = _tree_with(first, second)
    stats = _census(tree)
    assert stats["unique_structural_positions"] == 1
    assert stats["unique_search_contexts"] == 1
    assert stats["unique_encoded_inputs"] == 1


def test_halfmove_context_is_not_erased_by_encoder_clipping() -> None:
    first = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 100 70")
    second = first.copy(stack=True)
    second.halfmove_clock = 101
    tree = _tree_with(first, second)
    stats = _census(tree)
    assert stats["unique_structural_positions"] == 1
    assert stats["unique_search_contexts"] == 2
    assert stats["unique_encoded_inputs"] == 1  # legacy metadata clips at 100
    assert _census(tree, input_history_mode=1)["unique_encoded_inputs"] == 2


def test_limit_refuses_a_partial_census_and_leaves_the_tree_unchanged() -> None:
    tree = _tree_with(chess.Board(), chess.Board())
    before = tree.memory_stats(), tree.tt_entry_stats(), _mcts_tree.tt_stats()
    with pytest.raises(ValueError, match=r"all 2 cached boards.*no partial"):
        _census(tree, max_cached_boards=1)
    assert (tree.memory_stats(), tree.tt_entry_stats(), _mcts_tree.tt_stats()) == before
    assert _census(tree, max_cached_boards=2)["cached_boards"] == 2


@pytest.mark.parametrize("reset", ["reset", "reset_compact"])
def test_empty_and_reset_cache_is_a_complete_empty_census(reset: str) -> None:
    tree = _tree_with(chess.Board(), chess.Board())
    assert _census(tree)["same_context_duplicate_nodes"] == 1
    getattr(tree, reset)()
    stats = _census(tree)
    assert stats["cached_boards"] == 0
    assert stats["unique_structural_positions"] == 0
    assert stats["unique_search_contexts"] == 0
    assert stats["unique_encoded_inputs"] == 0
    assert stats["positions_with_multiple_contexts"] == 0
    assert stats["max_contexts_per_position"] == 0


def test_scope_includes_discarded_roots_until_compaction_removes_them() -> None:
    tree = _tree_with(chess.Board(), chess.Board())
    # Merely choosing the second root does not discard the first cached node.
    assert _census(tree)["same_context_duplicate_nodes"] == 1
    assert tree.compact_to_subtree(1) == 0
    stats = _census(tree)
    assert stats["cached_boards"] == 1
    assert stats["same_context_duplicate_nodes"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_history_mode": -1}, "input_history_mode"),
        ({"input_history_mode": 3}, "input_history_mode"),
        ({"n_extra": 35}, "registered feature width"),
        ({"max_cached_boards": 0}, "must be positive"),
    ],
)
def test_invalid_census_parameters_are_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _census(MCTSTree(), **kwargs)


def test_encoder_mode_and_width_must_be_supplied() -> None:
    with pytest.raises(TypeError, match="required argument"):
        MCTSTree().transposition_census()  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError, match="required argument"):
        MCTSTree().transposition_census(input_history_mode=2)  # pyright: ignore[reportCallIssue]


def test_pending_native_search_is_refused_and_can_still_finish() -> None:
    cb = CBoard.from_board(chess.Board())
    legal = cb.legal_move_indices().astype(np.int32)
    tree = _tree_with(cb)
    priors = np.full(legal.size, 1.0 / legal.size, dtype=np.float64)
    tree.expand(0, legal, priors)
    full_prior = np.zeros(_POLICY_SIZE, dtype=np.float64)
    full_prior[legal] = priors
    enc_buf = np.empty((32, 175, 8, 8), dtype=np.float32)
    pending = tree.start_gumbel_sims(
        [cb], np.array([0], dtype=np.int32), [[int(legal[0]), int(legal[1])]],
        [np.zeros(_POLICY_SIZE, dtype=np.float64)], [full_prior],
        np.array([8], dtype=np.int32), np.array([0.0], dtype=np.float64),
        0.1, 50.0, 2.5, 1.2, True, enc_buf, 1, 1, 2,
    )
    assert pending is not None
    assert pending > 0
    before = tree.memory_stats(), tree.tt_entry_stats(), _mcts_tree.tt_stats()
    with pytest.raises(RuntimeError, match="completed search; batch pending"):
        _census(tree)
    assert (tree.memory_stats(), tree.tt_entry_stats(), _mcts_tree.tt_stats()) == before
    while pending is not None:
        n = int(pending)
        pending = tree.continue_gumbel_sims(
            np.zeros((n, _POLICY_SIZE), dtype=np.float32),
            np.zeros((n, 3), dtype=np.float32),
        )
    assert int(_census(tree)["cached_boards"]) > 1


class _Evaluator:
    """Deterministic, batch-independent outputs; record batching for identity."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def evaluate_encoded(
        self, x: NDArray[np.float32], relations: NDArray[np.uint8] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        del relations
        n = int(x.shape[0])
        self.batches.append(n)
        pol = np.repeat(np.linspace(-1.0, 1.0, _POLICY_SIZE, dtype=np.float32)[None], n, axis=0)
        wdl = np.zeros((n, 3), dtype=np.float32)
        wdl[:, 0] = x[:, :12].sum(axis=(1, 2, 3)) * 0.01
        wdl[:, 2] = 0.3
        return pol, wdl


def _run(tree: MCTSTree, board: chess.Board, evaluator: _Evaluator, root: int | None = None):
    return run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(73),
        cfg=GumbelConfig(
            simulations=64, topk=8, c_scale=0.1, add_noise=False, temperature=0.0,
            input_history_encoding="lc0_root_legacy_meta", input_extra_features="v2_threats",
        ),
        evaluator=evaluator, tree=tree,
        root_node_ids=None if root is None else [root],
        target_batch=8, vloss_weight=1,
    )


def _tree_signature(tree: MCTSTree) -> list[tuple]:
    return [
        (
            tree.node_q(i), tree.get_solved_status(i), tree.is_expanded(i),
            tree.get_virtual_loss(i),
            tuple(tuple(array.tolist()) for array in tree.get_children_q(i, 0.0)),
        )
        for i in range(tree.node_count())
    ]


def test_census_preserves_real_gumbel_state_batching_and_carried_root_continuation() -> None:
    outputs = []
    batch_shapes = []
    signatures = []
    for inspect in (False, True):
        board = _after("g1f3", "g8f6", "g2g3", "g7g6")
        tree, evaluator = MCTSTree(), _Evaluator()
        first = _run(tree, board, evaluator)
        root = int(first[5][0])
        if inspect:
            signature = _tree_signature(tree)
            before = tree.memory_stats(), tree.tt_entry_stats(), _mcts_tree.tt_stats()
            stats = _census(tree)
            assert int(stats["cached_boards"]) > 10, "fixture did not exercise real C leaves"
            assert int(stats["cached_boards"]) < tree.node_count(), "cache is not all tree nodes"
            assert _tree_signature(tree) == signature
            assert (tree.memory_stats(), tree.tt_entry_stats(), _mcts_tree.tt_stats()) == before
        action = int(first[1][0])
        move = next(m for m in board.legal_moves if int(move_to_index(m, board)) == action)
        carried_root = tree.find_child(root, action)
        assert carried_root >= 0
        board.push(move)
        second = _run(tree, board, evaluator, carried_root)
        outputs.append((first, second))
        batch_shapes.append(evaluator.batches)
        signatures.append(_tree_signature(tree))
    for before, after in zip(outputs[0], outputs[1], strict=True):
        for index in (0, 1, 2, 3, 5):
            np.testing.assert_array_equal(before[index], after[index])
    assert batch_shapes[0] == batch_shapes[1]
    assert signatures[0] == signatures[1]
