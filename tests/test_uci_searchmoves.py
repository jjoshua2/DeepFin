from __future__ import annotations

import threading
from unittest.mock import MagicMock

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.moves import POLICY_SIZE, move_to_index
from chess_anti_engine.uci import search as uci_search
from chess_anti_engine.uci.search import (
    _allowed_root_indices,
    _board_after,
    _best_move_and_pv,
    SearchWorker,
)
from chess_anti_engine.uci.time_manager import Deadline


def test_allowed_root_indices_ignores_invalid_searchmoves() -> None:
    board = chess.Board()

    allowed = _allowed_root_indices(board, ("e2e4", "notamove", "e7e5"))

    assert allowed == {int(move_to_index(chess.Move.from_uci("e2e4"), board))}


def test_best_move_is_restricted_to_searchmoves() -> None:
    board = chess.Board()
    e2e4 = int(move_to_index(chess.Move.from_uci("e2e4"), board))
    d2d4 = int(move_to_index(chess.Move.from_uci("d2d4"), board))
    tree = MCTSTree()
    root = tree.add_root(0, 0.0)
    tree.expand(
        root,
        np.array([e2e4, d2d4], dtype=np.int32),
        np.array([0.5, 0.5], dtype=np.float64),
    )
    tree.backprop(np.array([root, tree.find_child(root, d2d4)], dtype=np.int32), 1.0)

    best, pv = _best_move_and_pv(tree, root, allowed_root_indices={e2e4})

    assert best == e2e4
    assert pv == [e2e4]


def test_reused_root_info_is_restricted_to_searchmoves() -> None:
    board = chess.Board()
    e2e4 = int(move_to_index(chess.Move.from_uci("e2e4"), board))
    d2d4 = int(move_to_index(chess.Move.from_uci("d2d4"), board))
    tree = MCTSTree()
    root = tree.add_root(0, 0.0)
    tree.expand(
        root,
        np.array([e2e4, d2d4], dtype=np.int32),
        np.array([0.5, 0.5], dtype=np.float64),
    )
    tree.backprop(np.array([root, tree.find_child(root, d2d4)], dtype=np.int32), 1.0)

    worker = SearchWorker(MagicMock(), device="cpu", n_walkers=1)
    worker._tree = tree  # noqa: SLF001
    worker._root_id = root  # noqa: SLF001
    seen: list[tuple[str, ...]] = []

    def _info_cb(**kwargs) -> None:
        seen.append(kwargs["pv"])

    pv_indices, _, _ = worker._maybe_emit_pv_info(  # noqa: SLF001
        board=board,
        deadline=Deadline(None),
        last_value=0.0,
        total_nodes=1,
        info_cb=_info_cb,
        max_depth=1,
        last_info_ms=-10_000,
        tb_probe=None,
        allowed_root_indices={e2e4},
    )

    assert pv_indices == [e2e4]
    assert seen == [("e2e4",)]


def test_result_ponder_failure_does_not_poison_bestmove(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If converting the ponder move raises, the bestmove must still be
    returned instead of letting the exception propagate and turning the
    whole search into ``bestmove 0000``."""
    worker = SearchWorker(MagicMock(), device="cpu", n_walkers=1)
    worker._walker_pool = None  # force single-walker Gumbel path  # noqa: SLF001
    board = chess.Board()
    e2e4 = int(move_to_index(chess.Move.from_uci("e2e4"), board))
    e7e5 = int(move_to_index(chess.Move.from_uci("e7e5"), board))

    tree = MCTSTree()
    root = tree.add_root(0, 0.0)
    tree.expand(
        root,
        np.array([e2e4, e7e5], dtype=np.int32),
        np.array([0.5, 0.5], dtype=np.float64),
    )
    worker._tree = tree  # noqa: SLF001
    worker._root_id = root  # noqa: SLF001
    worker._last_gumbel_action_idx = e2e4  # noqa: SLF001

    calls = 0
    original = uci_search._index_to_uci

    def _raising(b: chess.Board, idx: int) -> str:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("ponder crash")
        return original(b, idx)

    monkeypatch.setattr(uci_search, "_index_to_uci", _raising)

    result = worker._build_final_search_result(  # noqa: SLF001
        board=board,
        total_nodes=1,
        last_value=0.0,
        tb_probe=None,
        include_ponder=True,
    )

    assert result.bestmove_uci == "e2e4"
    assert result.ponder_uci is None


def test_board_after_returns_none_when_bestmove_index_cannot_be_pushed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = chess.Board()

    def _bad_index_to_move(*_args: object, **_kwargs: object) -> chess.Move:
        raise ValueError("bad policy index")

    monkeypatch.setattr(uci_search, "index_to_move", _bad_index_to_move)

    assert _board_after(board, 0) is None


def test_ponder_is_not_computed_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = SearchWorker(MagicMock(), device="cpu", n_walkers=1)
    worker._walker_pool = None  # force single-walker Gumbel path  # noqa: SLF001
    board = chess.Board()
    e2e4 = int(move_to_index(chess.Move.from_uci("e2e4"), board))

    tree = MCTSTree()
    root = tree.add_root(0, 0.0)
    tree.expand(
        root,
        np.array([e2e4], dtype=np.int32),
        np.array([1.0], dtype=np.float64),
    )
    worker._tree = tree  # noqa: SLF001
    worker._root_id = root  # noqa: SLF001
    worker._last_gumbel_action_idx = e2e4  # noqa: SLF001

    monkeypatch.setattr(
        uci_search,
        "_reply_at_child",
        lambda *_args, **_kwargs: pytest.fail("ponder reply should not be computed"),
    )

    result = worker._build_final_search_result(  # noqa: SLF001
        board=board,
        total_nodes=1,
        last_value=0.0,
        tb_probe=None,
        include_ponder=False,
    )

    assert result.bestmove_uci == "e2e4"
    assert result.ponder_uci is None


def test_ponder_aligns_with_gumbel_bestmove_not_most_visited_root() -> None:
    """When the Gumbel survivor differs from the most-visited root child,
    the ponder reply must come from the Gumbel survivor's child node, not
    from the most-visited root child's reply."""
    worker = SearchWorker(MagicMock(), device="cpu", n_walkers=1)
    worker._walker_pool = None  # force single-walker Gumbel path  # noqa: SLF001
    board = chess.Board()
    e2e4 = int(move_to_index(chess.Move.from_uci("e2e4"), board))
    d2d4 = int(move_to_index(chess.Move.from_uci("d2d4"), board))
    # Policy indices are perspective-dependent; encode replies on the child board.
    after_e2e4 = board.copy(stack=False)
    after_e2e4.push(chess.Move.from_uci("e2e4"))
    e7e5 = int(move_to_index(chess.Move.from_uci("e7e5"), after_e2e4))

    after_d2d4 = board.copy(stack=False)
    after_d2d4.push(chess.Move.from_uci("d2d4"))
    d7d5 = int(move_to_index(chess.Move.from_uci("d7d5"), after_d2d4))

    tree = MCTSTree()
    root = tree.add_root(0, 0.0)
    tree.expand(
        root,
        np.array([e2e4, d2d4], dtype=np.int32),
        np.array([0.3, 0.7], dtype=np.float64),
    )
    # d2d4 is most-visited; expand its child with d7d5
    child_d2d4 = tree.find_child(root, d2d4)
    tree.expand(
        child_d2d4,
        np.array([d7d5], dtype=np.int32),
        np.array([1.0], dtype=np.float64),
    )
    # e2e4 is the Gumbel survivor; expand its child with e7e5
    child_e2e4 = tree.find_child(root, e2e4)
    tree.expand(
        child_e2e4,
        np.array([e7e5], dtype=np.int32),
        np.array([1.0], dtype=np.float64),
    )

    worker._tree = tree  # noqa: SLF001
    worker._root_id = root  # noqa: SLF001
    worker._last_gumbel_action_idx = e2e4  # Gumbel survivor, NOT most visited  # noqa: SLF001

    result = worker._build_final_search_result(  # noqa: SLF001
        board=board,
        total_nodes=1,
        last_value=0.0,
        tb_probe=None,
        include_ponder=True,
    )

    assert result.bestmove_uci == "e2e4"
    # Ponder must be e7e5 (reply after e2e4), NOT d7d5 (reply after d2d4)
    assert result.ponder_uci == "e7e5"


def test_sampled_multi_ply_mate_pv_does_not_emit_proven_mate_score() -> None:
    """A Gumbel/MCTS PV can end in mate without proving mate at the root.

    Only direct root proofs such as the terminal shortcut's mate case should
    emit UCI ``score mate``. Otherwise sampled PVs can look like definite mate
    scores in match logs and then disappear on the next root.
    """
    worker = SearchWorker(MagicMock(), device="cpu", n_walkers=1)
    worker._walker_pool = None  # force single-walker Gumbel path  # noqa: SLF001

    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    g2g4_move = chess.Move.from_uci("g2g4")
    g2g4 = int(move_to_index(g2g4_move, board))

    after_g2g4 = board.copy(stack=False)
    after_g2g4.push(g2g4_move)
    qd8h4 = int(move_to_index(chess.Move.from_uci("d8h4"), after_g2g4))
    after_mate = after_g2g4.copy(stack=False)
    after_mate.push(chess.Move.from_uci("d8h4"))
    assert after_mate.is_checkmate()

    tree = MCTSTree()
    root = tree.add_root(0, 0.0)
    tree.expand(root, np.array([g2g4], dtype=np.int32), np.array([1.0], dtype=np.float64))
    child_g2g4 = tree.find_child(root, g2g4)
    tree.expand(child_g2g4, np.array([qd8h4], dtype=np.int32), np.array([1.0], dtype=np.float64))

    worker._tree = tree  # noqa: SLF001
    worker._root_id = root  # noqa: SLF001
    worker._last_gumbel_action_idx = g2g4  # noqa: SLF001

    result = worker._build_final_search_result(  # noqa: SLF001
        board=board, total_nodes=2, last_value=0.0, tb_probe=None,
    )

    assert result.bestmove_uci == "g2g4"
    assert result.pv == ("g2g4", "d8h4")
    assert result.score_mate is None


def test_immediate_mate_beats_high_prior_stalemate_root_move() -> None:
    """Mate-in-1 must bypass Gumbel root top-k pruning.

    Regression for a scaling-match game where the net had a huge advantage
    but selected Bf7 stalemate while Rb8# was legal.
    """

    class StalemateBiasedEvaluator:
        def __init__(self, board: chess.Board) -> None:
            self.calls = 0
            self.stalemate_idx = int(move_to_index(chess.Move.from_uci("g6f7"), board))

        def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            self.calls += 1
            batch = int(x.shape[0])
            pol = np.full((batch, POLICY_SIZE), -1000.0, dtype=np.float32)
            pol[:, self.stalemate_idx] = 1000.0
            wdl = np.tile(
                np.array([[10.0, -10.0, -10.0]], dtype=np.float32),
                (batch, 1),
            )
            return pol, wdl

    board = chess.Board("5k2/1R6/P4BB1/P7/2P5/8/3K3P/8 w - - 5 70")
    assert board.san(chess.Move.from_uci("b7b8")) == "Rb8#"
    after_stalemate = board.copy(stack=False)
    after_stalemate.push(chess.Move.from_uci("g6f7"))
    assert after_stalemate.is_stalemate()

    evaluator = StalemateBiasedEvaluator(board)
    worker = SearchWorker(
        evaluator,
        device="cpu",
        chunk_sims=1,
        gumbel_cfg=GumbelConfig(simulations=1, topk=1, temperature=0.0, add_noise=False),
    )

    result = worker.run(
        board,
        stop_event=threading.Event(),
        deadline=Deadline(None),
        max_nodes=1,
    )

    assert result.bestmove_uci == "b7b8"
    assert result.pv == ("b7b8",)
    assert result.score_mate == 1
    assert evaluator.calls == 0


def test_searchmoves_filter_reaches_c_mate_shortcut() -> None:
    """An out-of-list mate-in-1 must not poison the C-path result value.

    The root has Rb8# available, but UCI restricts the search to Bf7, a
    stalemating move. The Python pre-check respects ``searchmoves``; the C root
    shortcut must see the same restricted root set instead of returning the
    out-of-list mate value.
    """

    class NeutralEvaluator:
        def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            batch = int(x.shape[0])
            return (
                np.zeros((batch, POLICY_SIZE), dtype=np.float32),
                np.zeros((batch, 3), dtype=np.float32),
            )

    board = chess.Board("5k2/1R6/P4BB1/P7/2P5/8/3K3P/8 w - - 5 70")
    assert board.san(chess.Move.from_uci("b7b8")) == "Rb8#"
    after_allowed = board.copy(stack=False)
    after_allowed.push(chess.Move.from_uci("g6f7"))
    assert after_allowed.is_stalemate()

    worker = SearchWorker(
        NeutralEvaluator(),
        device="cpu",
        chunk_sims=1,
        gumbel_cfg=GumbelConfig(simulations=1, topk=1, temperature=0.0, add_noise=False),
    )

    result = worker.run(
        board,
        stop_event=threading.Event(),
        deadline=Deadline(None),
        max_nodes=1,
        root_moves=("g6f7",),
    )

    assert result.bestmove_uci == "g6f7"
    assert result.score_mate is None
    assert result.score_cp == 0
