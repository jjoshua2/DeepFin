"""Search / engine leads from an outside-model review (2026-11-19), each pinned
by the case that reproduced it.

- S5: ``go depth 4`` on a position whose only move stalemates never returned on
  the pool / walker paths: the root is proven drawn, the PV cannot grow, and a
  drawn root did not count as a finished depth. It counts now only when every
  root child is proven: a reroot onto a twofold leaf leaves a stale draw.
- S7: the reply book stored a proven-lost node's visit leader, which can be
  mated at once (the root entry already avoided that); proven-won nodes and
  roots now give the proven win.
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


@pytest.mark.parametrize("path", ["walkers", "pool"])
def test_go_depth_does_not_stop_on_a_rerooted_stale_draw(path: str) -> None:
    """Review repro: the C tree marks an in-tree twofold LEAF drawn, and
    `advance_root` reroots onto it with that status, so the new root reads
    drawn with unproven children. `go depth 4` stopped after one chunk on an
    arbitrary move; it must keep searching (here: to the deadline)."""
    worker = _make_worker(path, seed=0)
    try:
        b0 = chess.Board()
        for u in ("g1f3", "g8f6"):
            b0.push_uci(u)
        worker.run(b0.copy(), stop_event=threading.Event(), deadline=Deadline(20_000),
                   max_nodes=2000, max_depth=None)
        tree, rid = worker._tree, worker._root_id
        assert tree is not None
        assert rid is not None
        m1, m2 = chess.Move.from_uci("f3g1"), chess.Move.from_uci("f6g8")
        b1 = b0.copy()
        b1.push(m1)
        child = int(tree.find_child(rid, int(move_to_index(m1, b0))))
        assert child >= 0, "fixture: Ng1 was searched"
        leaf = int(tree.find_child(child, int(move_to_index(m2, b1))))
        assert leaf >= 0, "fixture: Ng8 was searched"
        assert len(tree.get_children_visits(leaf)[0]) == 0, "fixture: a leaf"
        _mark(tree, leaf, 2)  # what an in-tree twofold leaf carries
        b2 = b1.copy()
        b2.push(m2)
        # `advance_root` now refuses this root (test_a_stale_draw_root_is_not_reused);
        # adopt it by hand to keep pinning the search-side defence.
        assert not worker.advance_root(b0.copy(), [m1, m2])
        worker._root_id, worker._tree_fen = leaf, b2.fen()
        worker._invalidate_root_caches()
        t0 = time.monotonic()
        result = worker.run(b2, stop_event=threading.Event(), deadline=Deadline(3_000),
                            max_nodes=None, max_depth=4)
        took = time.monotonic() - t0
        assert worker._root_solved_status() == 2, "fixture: the stale draw survives the reroot"
        assert worker.last_stop_reason != "max_depth" or len(result.pv) >= 4, (
            worker.last_stop_reason, result.pv, result.nodes, took)
    finally:
        worker.close()


# -- follow-up: a stale DRAW root is not reused --------------------------------------


def _reroot_onto_a_marked_twofold_leaf(worker: SearchWorker) -> tuple[bool, chess.Board]:
    """1.Nf3 Nf6, search, mark the Ng1-Ng8 leaf DRAW (what an in-tree twofold
    leaf carries), then advance the root onto it as the engine would; on a
    refusal reset the tree, as `Engine._sync_tree_root` does."""
    b0 = chess.Board()
    for u in ("g1f3", "g8f6"):
        b0.push_uci(u)
    worker.run(b0.copy(), stop_event=threading.Event(), deadline=Deadline(20_000),
               max_nodes=2000, max_depth=None)
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
    assert len(tree.get_children_visits(leaf)[0]) == 0, "fixture: a leaf"
    _mark(tree, leaf, 2)
    reused = worker.advance_root(b0.copy(), [m1, m2])
    if not reused:
        worker.reset_tree()
    b2 = b1.copy()
    b2.push(m2)
    return reused, b2


@pytest.mark.parametrize("path", ["classic", "walkers", "pool"])
def test_a_stale_draw_root_is_not_reused(path: str) -> None:
    """Reused, the stale mark made the classic runner play the repeating Ng1-f3
    with `score cp 0` and the pool / walker roots read solved. Refused, the
    search starts fresh: the root is unsolved."""
    worker = _make_worker(path, seed=0)
    try:
        reused, b2 = _reroot_onto_a_marked_twofold_leaf(worker)
        assert not reused
        assert worker.stale_draw_reroots_refused == 1
        worker.run(b2, stop_event=threading.Event(), deadline=Deadline(10_000),
                   max_nodes=256)
        assert worker._root_solved_status() == 0
    finally:
        worker.close()


def test_a_genuinely_proven_draw_root_is_still_reused() -> None:
    """A node drawn on its children's proofs (every child proven) is
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
        _mark(tree, int(tree.find_child(child, int(replies[1]))), -1)
        _mark(tree, child, 2)
        worker._tree, worker._root_id, worker._tree_fen = tree, rid, b0.fen()
        assert worker.advance_root(b0, [e4])
        assert worker._root_id == child
        assert worker.stale_draw_reroots_refused == 0
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


def test_the_tree_answer_on_a_proven_won_root_plays_the_proven_win() -> None:
    """`tree_root_bestmove` (the book's root entry and the at-once answer)
    follows the same rule as `_book_answer`: the proven win, not the visit
    leader searched before the proof."""
    after = START.copy()
    after.push(OUR_MOVE)
    pos = after.copy()
    pos.push(REPLY)
    tree = MCTSTree()
    tree.reserve(64, 640)
    rid = int(tree.add_root(1, 0.0))
    acts = [int(move_to_index(chess.Move.from_uci(u), pos)) for u in ("h3h2", "h3h4")]
    tree.expand(rid, np.array(acts, dtype=np.int32), np.array([0.5, 0.5]))
    for a, n in zip(acts, (20, 3), strict=True):
        child = int(tree.find_child(rid, a))
        for _ in range(n):
            tree.backprop(np.array([rid, child], dtype=np.int32), 0.0)
    worker = SearchWorker.__new__(SearchWorker)
    worker._tree, worker._root_id, worker._tree_fen = tree, rid, pos.fen()
    assert worker.tree_root_bestmove(pos) == "h3h2"  # unproven: the visit leader
    _mark(tree, int(tree.find_child(rid, acts[1])), -1)
    _mark(tree, rid, 1)
    assert worker.tree_root_bestmove(pos) == "h3h4"
    # `searchmoves` without the win: the old rule.
    assert worker.tree_root_bestmove(pos, searchmoves=("h3h2",)) == "h3h2"


# -- R1: a draw the opponent cannot decline ------------------------------------------

# Every move but the drawing one is mated next move. The drawing king move
# leaves the clock at 99 and every Black reply is reversible (clock 100) and
# not mate; each fixture has at least two losing pawn moves, so the prune
# (which since #1410 keeps at least two moves) still happens.
REPLY_FORCED_DRAWS = [
    ("K7/8/k6q/8/8/8/4P3/8 w - - 98 1", "a8b8"),
    ("8/P7/8/8/8/2k4q/8/2K5 w - - 98 1", "c1b1"),
    ("2K5/8/1k6/1q6/8/8/P7/8 w - - 98 1", "c8d8"),
]
# Two- and three-move roots with the same shape (one losing pawn move: no prune).
SMALL_REPLY_FORCED_DRAWS = [
    ("6K1/8/3P2k1/8/8/8/1q6/8 w - - 98 1", "g8f8"),
    ("3q4/8/k7/8/8/6P1/4r3/1K6 w - - 98 1", "b1c1"),
]
# Rf7+ leaves the clock at 99, but Black's Kxf7 (a capture into K+N vs K+B,
# not a dead position) declines the draw: not forced.
DECLINED_BY_A_CAPTURE = ("5k2/3R4/8/8/b7/8/5K2/7N w - - 98 1", "d7f7")


def _idx(board: chess.Board, uci: str) -> int:
    return int(move_to_index(chess.Move.from_uci(uci), board))


@pytest.mark.parametrize(("fen", "draw"), REPLY_FORCED_DRAWS + SMALL_REPLY_FORCED_DRAWS)
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


def test_a_capture_reply_keeps_the_draw_declinable() -> None:
    fen, move = DECLINED_BY_A_CAPTURE
    board = chess.Board(fen)
    legal = [int(move_to_index(m, board)) for m in board.legal_moves]
    assert _idx(board, move) not in root_forced_draw_actions(board, legal)


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
    worker = _make_worker(path, optimistic=True, seed=0)
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
