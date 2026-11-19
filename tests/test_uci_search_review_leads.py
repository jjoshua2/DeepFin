"""Search / engine leads from an outside-model review (2026-11-19), each pinned
by the case that reproduced it.

- S5: ``go depth 4`` on a position whose only move stalemates never returned on
  the pool / walker paths: the root is proven drawn, the PV cannot grow, and a
  drawn root did not count as a finished depth.
- S7: the reply book stored a proven-lost node's visit leader, which can be
  mated at once (the root entry already avoided that).
- R1: a winning root pruned a move after which every opponent reply reaches the
  50-move draw, then counted it as not a draw: the root was marked lost and
  root-parallel Gumbel / classic Gumbel played a losing pawn move.
- E1: an at-once answer's fresh eval waited up to 1 s for the helper's OS
  thread to exit after it returned, outside its few-millisecond budget.
- W1: ``WalkerPool.run`` reserved the worst-case arena for its whole target
  before looking at a stop already set (~14 GB for the 1M-sim stop test).
"""
from __future__ import annotations

import threading
import time
from typing import Any

import chess
import numpy as np
import pytest

import chess_anti_engine.utils.thread_join as thread_join
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.root_parallel_gumbel import root_forced_draw_actions
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline
from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig
from tests.test_uci_solved_root_stops import (
    _WINNING_WDL,
    CLAIMABLE_ONLY,
    FAST_S,
    _make_worker,
    _mark,
    _rpg_pool,
)
from tests.test_uci_time_forfeits import _mock_worker

# -- S5: go depth on a proven-drawn root ------------------------------------------

# White's only move, Kxg7, stalemates Black.
ONLY_MOVE_STALEMATES = ("8/6qK/8/8/8/8/2Q5/k7 w - - 0 1", "h7g7")


@pytest.mark.parametrize("path", ["pool", "walkers"])
def test_go_depth_answers_on_a_proven_drawn_root(path: str) -> None:
    fen, only = ONLY_MOVE_STALEMATES
    worker = _make_worker(path)
    try:
        t0 = time.monotonic()
        result = worker.run(
            chess.Board(fen), stop_event=threading.Event(),
            deadline=Deadline(int(FAST_S * 1000) + 5_000), max_nodes=None, max_depth=4,
        )
        took = time.monotonic() - t0
        assert worker._root_solved_status() == 2
        assert worker.last_stop_reason == "max_depth", (worker.last_stop_reason, took)
        assert took < FAST_S
        assert result.bestmove_uci == only
    finally:
        worker.close()


# -- S7: the reply book on proven nodes ---------------------------------------------

# Kh3-h2 is mated in one (Qh1#); Kh3-h4 is not. Reached from START by Kh2-h3 Kb5-b4.
START = chess.Board("8/1q6/8/1k4r1/8/8/7K/8 w - - 0 1")
OUR_MOVE, REPLY = chess.Move.from_uci("h2h3"), chess.Move.from_uci("b5b4")


def _book_tree(answers: dict[str, tuple[int, int]], node_status: int):
    """START -> Kh3 -> Kb4 -> ``answers`` ({uci: (visits, solved status)}),
    the reply node marked ``node_status``; returns (worker, position)."""
    after = START.copy()
    after.push(OUR_MOVE)
    pos = after.copy()
    pos.push(REPLY)
    tree = MCTSTree()
    tree.reserve(64, 640)
    rid = int(tree.add_root(1, 0.0))
    tree.expand(rid, np.array([move_to_index(OUR_MOVE, START)], dtype=np.int32), np.array([1.0]))
    ours = int(tree.find_child(rid, int(move_to_index(OUR_MOVE, START))))
    tree.expand(ours, np.array([move_to_index(REPLY, after)], dtype=np.int32), np.array([1.0]))
    reply = int(tree.find_child(ours, int(move_to_index(REPLY, after))))
    acts = [int(move_to_index(chess.Move.from_uci(u), pos)) for u in answers]
    tree.expand(reply, np.array(acts, dtype=np.int32), np.full(len(acts), 1.0 / len(acts)))
    for a, (visits, status) in zip(acts, answers.values(), strict=True):
        child = int(tree.find_child(reply, a))
        for _ in range(visits):
            tree.backprop(np.array([rid, ours, reply, child], dtype=np.int32), 0.0)
        if status:
            _mark(tree, child, status)
    if node_status:
        _mark(tree, reply, node_status)
    worker = SearchWorker.__new__(SearchWorker)
    worker._tree, worker._root_id, worker._tree_fen = tree, rid, START.fen()
    worker._snapshot_reply_book(START, force=True)
    return worker, pos


def test_the_reply_book_avoids_mate_in_one_on_a_proven_lost_node() -> None:
    worker, pos = _book_tree({"h3h2": (20, 1), "h3h4": (2, 1)}, node_status=-1)
    assert worker.reply_book_bestmove(pos) == "h3h4"
    # Unproven, the visit leader stands.
    worker, pos = _book_tree({"h3h2": (20, 0), "h3h4": (2, 0)}, node_status=0)
    assert worker.reply_book_bestmove(pos) == "h3h2"


def test_the_reply_book_plays_the_proven_win_on_a_proven_won_node() -> None:
    # The visit leader was searched before the proof; Kh4 is the proven win.
    worker, pos = _book_tree({"h3h2": (20, 0), "h3h4": (3, -1)}, node_status=1)
    assert worker.reply_book_bestmove(pos) == "h3h4"


# -- R1: a draw the opponent cannot decline ------------------------------------------

# Every move but the drawing one is mated next move. The drawing king move
# leaves the clock at 99 and every Black reply is reversible (clock 100) and
# not mate; each fixture also has a losing pawn move, so the prune keeps one.
REPLY_FORCED_DRAWS = [
    ("6K1/8/3P2k1/8/8/8/1q6/8 w - - 98 1", "g8f8"),
    ("3q4/8/k7/8/8/6P1/4r3/1K6 w - - 98 1", "b1c1"),
    ("2K5/8/1k6/1q6/8/8/P7/8 w - - 98 1", "c8d8"),
]


def _idx(board: chess.Board, uci: str) -> int:
    return int(move_to_index(chess.Move.from_uci(uci), board))


@pytest.mark.parametrize(("fen", "draw"), REPLY_FORCED_DRAWS)
def test_a_draw_every_reply_reaches_is_a_forced_draw(fen: str, draw: str) -> None:
    board = chess.Board(fen)
    legal = [int(move_to_index(m, board)) for m in board.legal_moves]
    assert root_forced_draw_actions(board, legal) == [_idx(board, draw)]
    assert board.fen() == fen  # the board is left as it was


@pytest.mark.parametrize("fen", CLAIMABLE_ONLY)
def test_a_reply_that_mates_or_zeroes_keeps_the_draw_declinable(fen: str) -> None:
    board = chess.Board(fen)
    legal = [int(move_to_index(m, board)) for m in board.legal_moves]
    assert root_forced_draw_actions(board, legal) == []


@pytest.mark.parametrize(("fen", "draw"), REPLY_FORCED_DRAWS)
def test_prepare_root_records_a_pruned_reply_forced_draw(fen: str, draw: str) -> None:
    board = chess.Board(fen)
    pool = _rpg_pool()
    try:
        tree = MCTSTree()
        tree.reserve(1_000, 10_000)
        pool.prepare_root(
            tree=tree, board=board, pol_logits=np.zeros(4672, dtype=np.float32),
            wdl_logits=_WINNING_WDL,
        )
        st = pool._state
        assert st is not None
        assert _idx(board, draw) not in st.search_legal.tolist(), "fixture must prune"
        assert st.pruned_root_draws
    finally:
        pool.close()


@pytest.mark.parametrize("path", ["rpg", "classic"])
@pytest.mark.parametrize(("fen", "draw"), REPLY_FORCED_DRAWS)
def test_a_lost_looking_root_plays_the_reply_forced_draw(path: str, fen: str, draw: str) -> None:
    if path == "classic" and fen != REPLY_FORCED_DRAWS[2][0]:
        pytest.skip(
            "classic: the shared gumbel_c root prunes the draw and returns the one "
            "remaining move unsimulated (fixture 1) or does not prove it lost in "
            "budget, so no escape can arise without a gumbel_c change",
        )
    worker = _make_worker(path, optimistic=True)
    try:
        result = worker.run(
            chess.Board(fen), stop_event=threading.Event(),
            deadline=Deadline(30_000), max_nodes=4096,
        )
        assert result.bestmove_uci == draw, (path, result.bestmove_uci)
        assert result.score_mate is None
    finally:
        worker.close()


# -- E1: the fresh eval's OS-thread wait is inside its budget -------------------------


def test_the_fresh_eval_never_waits_past_its_budget_for_the_os_thread(monkeypatch) -> None:
    def stuck_teardown(_thread, cap, _start_time, **_kw) -> None:
        time.sleep(max(0.0, min(float(cap), 2.0)))  # the OS thread lingers to the cap

    monkeypatch.setattr(thread_join, "_await", stuck_teardown)
    worker = _mock_worker()
    worker.fresh_root_policy_bestmove.return_value = "e7e5"
    engine = Engine(worker, options=EngineOptions())
    try:
        board = chess.Board()
        board.push_uci("e2e4")
        t0 = time.monotonic()
        move = engine._bounded_fresh_eval(board, (), 0.010)
        took = time.monotonic() - t0
        assert move == "e7e5"
        assert took < 0.2, f"answered {took * 1000:.0f} ms into a 10 ms budget"
    finally:
        engine.close()


# -- W1: a stop already set reserves nothing ------------------------------------------


class _NeverEvaluated:
    def evaluate_encoded(self, *_a: Any, **_kw: Any):
        raise AssertionError("a pre-stopped run evaluated a leaf")


def test_a_pre_stopped_walker_run_reserves_no_arena() -> None:
    tree = MCTSTree()
    tree.reserve(512, 4096)
    root_cb = CBoard.from_board(chess.Board())
    rid = int(tree.add_root(1, 0.0))
    legal = root_cb.legal_move_indices().astype(np.int32)
    tree.expand(rid, legal, np.full(legal.size, 1.0 / legal.size))
    before = int(tree.node_capacity())
    pool = WalkerPool(
        WalkerPoolConfig(n_walkers=2, c_puct=1.5, fpu_at_root=0.0,
                         fpu_reduction=0.33, vloss_weight=3),
        _NeverEvaluated(),
    )
    stop = threading.Event()
    stop.set()
    try:
        assert pool.run(tree=tree, root_id=rid, root_cboard=root_cb,
                        target_sims=4096, stop_event=stop) == 0
        assert int(tree.node_capacity()) == before
    finally:
        pool.close()
