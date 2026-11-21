"""Search-termination on proven roots across the shared-tree search paths.

S1: root-parallel Gumbel descends each candidate from the candidate node, so a
proof found inside a candidate subtree never reached the real root: its solved
status stayed 0 and both `go mate` stops (`_proven_mate_within`,
`mate_unreachable`) were unreachable -- `go mate 2` ran to its 60 s cap.

S2: on the pool and root-parallel paths a proven root's PV ends on a proven
terminal and cannot reach `go depth N`, while every chunk keeps re-visiting the
proven lines (completed > 0), so `go depth N` only answered after `stop`.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.uci.root_parallel_gumbel import (
    RootParallelGumbelConfig,
    RootParallelGumbelPool,
    _SearchState,
    root_forced_draw_actions,
)
import chess_anti_engine.uci.search as search_module
from chess_anti_engine.uci.search import _FORCED_ROOT_MATE_SIMS_PER_MOVE, SearchWorker
from chess_anti_engine.uci.time_manager import Deadline

# Black to move, mated in one whatever it plays (a6/a5, then Qg7# / Qf8#).
MATED_IN_ONE = "7k/p4Q2/6K1/8/8/8/8/8 b - - 0 1"
# A deadline the fixed search must never come near; a mutant runs into it.
# One root-parallel chunk is a whole halving schedule (~4 s on two capped CPU
# threads), so the fixed search answers after one or two chunks.
DEADLINE_MS = 30_000
FAST_S = 15.0


def _make_evaluator(*, optimistic: bool = False, seed: int | None = None) -> Any:
    if seed is not None:
        torch.manual_seed(seed)
    cfg = ModelConfig(
        input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
    )
    torch_model = build_model(cfg)
    torch_model.eval()
    if optimistic:
        # Every position looks won for the side to move: a root with
        # root_q > 0 prunes its terminal-draw moves on every search path.
        params = dict(torch_model.named_parameters())
        with torch.no_grad():
            params["value_wdl.net.2.weight"].zero_()
            params["value_wdl.net.2.bias"].copy_(torch.tensor([8.0, 0.0, -8.0]))
    return ThreadSafeGPUDispatcher(DirectGPUEvaluator(
        torch_model, device="cpu", max_batch=64, use_amp=False, n_slots=2,
    ))


def _make_worker(
    path: str, *, optimistic: bool = False, seed: int | None = None,
) -> SearchWorker:
    worker = SearchWorker(
        _make_evaluator(optimistic=optimistic, seed=seed), device="cpu",
        gumbel_cfg=GumbelConfig(
            input_extra_features="v1", simulations=64, add_noise=False, temperature=0.0,
        ),
        chunk_sims=64, n_walkers=2 if path == "walkers" else 1, walker_gather=8,
    )
    if path == "walkers":
        return worker
    if path == "classic":
        return worker
    pair = [
        _make_evaluator(optimistic=optimistic, seed=None if seed is None else seed + k)
        for k in (1, 2)
    ]
    if path == "rpg":
        worker.install_root_parallel_gumbel(pair, gather=8, as_factories=False)
    elif path == "pool":
        worker.install_multi_gpu_pucv(pair, gather=8, as_factories=False)
    else:
        raise AssertionError(path)
    return worker


def _timed_run(worker: SearchWorker, fen: str, **limits: Any) -> tuple[Any, float]:
    t0 = time.monotonic()
    result = worker.run(
        chess.Board(fen), stop_event=threading.Event(),
        deadline=Deadline(DEADLINE_MS), max_nodes=None, **limits,
    )
    return result, time.monotonic() - t0


# --- S1: the real root resolves from its candidates' proofs ------------------


def test_rpg_resolves_real_root_from_candidate_proofs() -> None:
    evaluators = [_make_evaluator(), _make_evaluator()]
    pool = RootParallelGumbelPool(
        RootParallelGumbelConfig(n_groups=2, gather=8),
        GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False, temperature=0.0),
        evaluators=evaluators, rng=np.random.default_rng(0),
    )
    try:
        board = chess.Board(MATED_IN_ONE)
        tree = MCTSTree()
        tree.reserve(20_000, 200_000)
        rid = pool.prepare_root(
            tree=tree, board=board,
            pol_logits=np.zeros(4672, dtype=np.float32),
            wdl_logits=np.zeros(3, dtype=np.float32),
        )
        for _ in range(8):
            pool.run(target_sims=512, stop_event=threading.Event())
            if tree.get_solved_status(rid) != 0:
                break
        assert tree.get_solved_status(rid) == -1, (
            "every reply is mated in one: the real root must be proven lost"
        )
    finally:
        pool.close()


def test_rpg_terminal_candidate_counts_as_a_proven_child() -> None:
    """A candidate that ends the game is never searched (it backprops its
    terminal value), so unless its expansion records it as proven the root
    cannot resolve from it. Ponder / analysis searches disable the root mate
    shortcut, which is when the mating candidate reaches the arenas."""
    pool = RootParallelGumbelPool(
        RootParallelGumbelConfig(n_groups=2, gather=8),
        GumbelConfig(
            input_extra_features="v1", simulations=64, topk=64,
            add_noise=False, temperature=0.0,
        ),
        evaluators=[_make_evaluator(), _make_evaluator()], rng=np.random.default_rng(0),
    )
    try:
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")  # Re8#
        tree = MCTSTree()
        tree.reserve(20_000, 200_000)
        rid = pool.prepare_root(
            tree=tree, board=board,
            pol_logits=np.zeros(4672, dtype=np.float32),
            wdl_logits=np.zeros(3, dtype=np.float32),
            allow_terminal_shortcuts=False,
        )
        assert pool._state is not None
        assert pool._state.finished_action is None
        pool.run(target_sims=512, stop_event=threading.Event())
        assert tree.get_solved_status(rid) == 1
    finally:
        pool.close()


def _state_with_children(statuses: list[int], *, pruned: bool) -> _SearchState:
    tree = MCTSTree()
    tree.reserve(64, 640)
    rid = int(tree.add_root(1, 0.0))
    actions = np.arange(len(statuses), dtype=np.int32)
    tree.expand(rid, actions, np.full(len(statuses), 1.0 / len(statuses)))
    for action, status in zip(actions.tolist(), statuses):
        if status:
            # One-node path: marks the child itself, no propagation.
            cid = int(tree.find_child(rid, action))
            tree.mark_solved_path(np.array([cid], dtype=np.int32), status)
    return _SearchState(
        tree=tree, root_id=rid, root_cb=None, fen="", pri=np.zeros(4672),
        search_legal=actions.astype(np.int64), root_q_init=0.0,
        pruned_root_draws=pruned,
    )


@pytest.mark.parametrize(
    ("children", "pruned", "expected"),
    [
        ([1, 0, 1], False, 0),     # an unproven child blocks a loss/draw verdict
        ([1, 0, -1], False, 1),    # one reply lost for the opponent: root won
        ([1, 1, 1], False, -1),    # every reply wins for the opponent: lost
        ([1, 2, 1], False, 2),     # a drawn reply: draw
        ([1, 1, 1], True, 2),      # pruned terminal draws stay available: draw
        ([2, -1], False, 1),       # a draw scanned before a win must not end the scan
        ([1, 2, 0], False, 0),     # draw + unproven: not yet a draw
    ],
)
def test_resolve_root_solved_rule(children: list[int], pruned: bool, expected: int) -> None:
    st = _state_with_children(children, pruned=pruned)
    pool = RootParallelGumbelPool.__new__(RootParallelGumbelPool)
    assert pool._resolve_root_solved(st) == expected
    assert st.tree.get_solved_status(st.root_id) == expected


def test_rpg_go_mate_stops_on_proven_lost_root() -> None:
    worker = _make_worker("rpg")
    try:
        result, took = _timed_run(worker, MATED_IN_ONE, mate_moves=2)
        assert worker.last_stop_reason == "mate_unreachable", worker.last_stop_reason
        assert took < FAST_S, f"go mate on a lost root took {took:.1f}s"
        assert result.score_mate == -1
    finally:
        worker.close()


# --- S2: `go depth N` answers on a proven root ---------------------------------


@pytest.mark.parametrize("path", ["rpg", "pool"])
def test_go_depth_answers_on_proven_root(path: str) -> None:
    worker = _make_worker(path)
    try:
        result, took = _timed_run(worker, MATED_IN_ONE, max_depth=4)
        assert worker.last_stop_reason == "max_depth", worker.last_stop_reason
        assert took < FAST_S, f"{path} go depth 4 on a proven root took {took:.1f}s"
        assert result.score_mate == -1
    finally:
        worker.close()


@pytest.mark.parametrize("path", ["rpg", "pool"])
def test_go_infinite_on_proven_root_still_waits_for_stop(path: str) -> None:
    worker = _make_worker(path)
    stop = threading.Event()
    timer = threading.Timer(1.0, stop.set)
    try:
        timer.start()
        t0 = time.monotonic()
        worker.run(
            chess.Board(MATED_IN_ONE), stop_event=stop,
            deadline=Deadline(None), max_nodes=None,
        )
        took = time.monotonic() - t0
        assert worker.last_stop_reason == "external"
        assert took >= 0.9, f"infinite search ended before stop ({took:.2f}s)"
    finally:
        timer.cancel()
        worker.close()


# --- `go mate N` on a forced root ---------------------------------------------

# White is in check with one legal move (Kg2) and no mate behind it.
FORCED_NO_MATE = "7k/8/8/8/8/8/5P1P/r5K1 w - - 0 1"


@pytest.mark.parametrize("path", ["pool", "walkers", "rpg"])
def test_go_mate_on_forced_root_answers_within_its_budget(path: str) -> None:
    """The pool / walker paths simulate a forced root (completed > 0), so the
    old completed == 0 forced-root stop never fired and `go mate N` ran to its
    60 s cap. The move is decided; the mate search behind it is bounded."""
    worker = _make_worker(path)
    try:
        result, took = _timed_run(worker, FORCED_NO_MATE, mate_moves=2)
        assert result.bestmove_uci == "g1g2"
        # Root-parallel Gumbel finishes a single-legal root without a phase
        # (`root_finished`), before any stop reason is consulted.
        expected = None if path == "rpg" else "mate_forced_root"
        assert worker.last_stop_reason == expected, worker.last_stop_reason
        assert took < FAST_S, f"{path} go mate 2 on a forced root took {took:.1f}s"
        assert result.nodes <= 2 * _FORCED_ROOT_MATE_SIMS_PER_MOVE + 4096, result.nodes
    finally:
        worker.close()


@pytest.mark.parametrize("path", ["pool", "walkers"])
def test_go_mate_on_forced_root_still_finds_the_mate_behind_it(path: str) -> None:
    """The budget bounds the mate search behind the forced move; it must not
    pre-empt a mate the search can prove (Kxf1, then mate next move)."""
    worker = _make_worker(path)
    try:
        result, _took = _timed_run(
            worker, "3k4/1pR5/p7/2p1Q3/8/8/6PP/5rK1 w - - 2 13", mate_moves=2,
        )
        assert worker.last_stop_reason == "mate", worker.last_stop_reason
        assert result.bestmove_uci == "g1f1"
        assert result.score_mate == 2
    finally:
        worker.close()


# --- review items: pruned-draw rule, TB path, depth on a proven PV end ---------

TB_3MAN = str(Path(__file__).parent / "fixtures" / "syzygy_3man")


def _rpg_pool(*, topk: int = 64) -> RootParallelGumbelPool:
    return RootParallelGumbelPool(
        RootParallelGumbelConfig(n_groups=2, gather=8),
        GumbelConfig(
            input_extra_features="v1", simulations=64, topk=topk,
            add_noise=False, temperature=0.0,
        ),
        evaluators=[_make_evaluator(), _make_evaluator()], rng=np.random.default_rng(0),
    )


_WINNING_WDL = np.array([8.0, 0.0, -8.0], dtype=np.float32)


def test_prepare_root_records_a_pruned_stalemate_as_an_available_draw() -> None:
    """bxa7 stalemates: a winning root prunes it, and the root keeps the draw."""
    board = chess.Board("8/q7/1P6/8/8/3Q4/8/K1k5 w - - 0 1")
    stalemate = chess.Move.from_uci("b6a7")
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
        assert int(move_to_index(stalemate, board)) not in st.search_legal.tolist()
        assert st.pruned_root_draws
    finally:
        pool.close()


def test_prepare_root_keeps_the_draws_when_pruning_would_leave_one_move() -> None:
    """Kxd7 stalemates, Kf8 is mated: pruning would leave one move, so the
    drawing move stays in the search (the classic `gumbel_c` rule)."""
    board = chess.Board("4K2k/3q1P2/6Q1/8/8/8/8/8 w - - 0 1")
    expected = {int(move_to_index(m, board)) for m in board.legal_moves}
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
        assert set(st.search_legal.tolist()) == expected
        assert not st.pruned_root_draws
        assert st.finished_action is None
    finally:
        pool.close()


# Mated in one after every move; some moves let the OPPONENT claim a 50-move
# draw on its reply (clock 99 + reversible reply). The opponent can decline.
CLAIMABLE_ONLY = [
    "8/8/8/3P4/6q1/P4k2/7K/8 w - - 98 1",
    "8/8/8/2q5/8/3P2P1/8/5k1K w - - 98 1",
    "K7/2k5/8/8/P2P4/8/3q4/8 w - - 98 1",
]


@pytest.mark.parametrize("fen", CLAIMABLE_ONLY)
def test_a_claimable_draw_is_not_a_root_draw(fen: str) -> None:
    board = chess.Board(fen)
    pool = _rpg_pool()
    try:
        tree = MCTSTree()
        tree.reserve(50_000, 500_000)
        rid = pool.prepare_root(
            tree=tree, board=board, pol_logits=np.zeros(4672, dtype=np.float32),
            wdl_logits=_WINNING_WDL,
        )
        st = pool._state
        assert st is not None
        assert st.search_legal.size < board.legal_moves.count(), "fixture must prune"
        assert not st.pruned_root_draws
        for _ in range(8):
            pool.run(target_sims=512, stop_event=threading.Event())
            if tree.get_solved_status(rid) != 0:
                break
        assert tree.get_solved_status(rid) == -1, "mated in one after every move"
    finally:
        pool.close()


def test_tb_proofs_leave_the_root_to_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tablebase proofs (at candidate expansion and inside candidate searches)
    stop at the candidate: only `_resolve_root_solved`, which applies the
    pruned-draw rule, may mark the real root."""
    from chess_anti_engine.tablebase import SyzygyProbe

    monkeypatch.setattr(RootParallelGumbelPool, "_resolve_root_solved", lambda self, st: 0)
    # 4 men: Kxb2 / Rxb2-type captures reach 3-man tablebase positions.
    board = chess.Board("k7/8/8/8/8/8/1r6/K6R w - - 0 1")
    pool = _rpg_pool()
    try:
        tree = MCTSTree()
        tree.reserve(50_000, 500_000)
        rid = pool.prepare_root(
            tree=tree, board=board, pol_logits=np.zeros(4672, dtype=np.float32),
            wdl_logits=np.zeros(3, dtype=np.float32),
        )
        probe = SyzygyProbe(TB_3MAN)
        for _ in range(3):
            pool.run(target_sims=1024, stop_event=threading.Event(), tb_probe=probe)
        st = pool._state
        assert st is not None
        proven = [
            int(tree.get_solved_status(int(tree.find_child(rid, int(a)))))
            for a in st.search_legal.tolist()
        ]
        assert any(p != 0 for p in proven), "fixture must produce tablebase proofs"
        assert tree.get_solved_status(rid) == 0
    finally:
        pool.close()


def _depth_tree() -> tuple[SearchWorker, MCTSTree, int, int, int]:
    """root -> child(10) -> leaf(20), sibling 11 unexpanded."""
    worker = SearchWorker.__new__(SearchWorker)
    tree = MCTSTree()
    tree.reserve(64, 640)
    rid = int(tree.add_root(1, 0.0))
    tree.expand(rid, np.array([10, 11], dtype=np.int32), np.array([0.5, 0.5]))
    child = int(tree.find_child(rid, 10))
    tree.expand(child, np.array([20], dtype=np.int32), np.array([1.0]))
    leaf = int(tree.find_child(child, 20))
    worker._tree, worker._root_id = tree, rid
    return worker, tree, rid, child, leaf


def _mark(tree: MCTSTree, node: int, status: int) -> None:
    tree.mark_solved_path(np.array([node], dtype=np.int32), status)  # this node only


def test_depth_pv_end_counts_only_when_nothing_better_can_exist() -> None:
    worker, tree, rid, child, leaf = _depth_tree()
    assert worker._depth_pv_len([10, 20], 4) == 2   # unproven end: keep deepening
    _mark(tree, leaf, 2)                             # the leader's line ends drawn ...
    assert worker._depth_pv_len([10, 20], 4) == 2   # ... but sibling 11 is unproven
    assert worker._depth_pv_len([], 4) == 0
    assert worker._depth_pv_len([10, 20], None) == 2

    worker, tree, rid, child, leaf = _depth_tree()
    _mark(tree, leaf, 1)
    _mark(tree, child, -1)                           # first move: a proven win
    assert worker._depth_pv_len([10, 20], 4) == 4

    for first_status in (2, 1):                      # first move drawn / lost: not decided
        worker, tree, rid, child, leaf = _depth_tree()
        _mark(tree, leaf, 2 if first_status == 2 else -1)
        _mark(tree, child, first_status)
        assert worker._depth_pv_len([10, 20], 4) == 2, first_status

    for root_status, complete in ((1, True), (-1, True), (2, False)):
        worker, tree, rid, child, leaf = _depth_tree()
        _mark(tree, leaf, 2)
        # A DRAW root with an unproven child (11) does not count: the C tree
        # marks an in-tree twofold LEAF drawn, and a reroot onto it leaves a
        # drawn root with unproven children (review of lead S5,
        # test_uci_search_review_leads.py).
        _mark(tree, rid, root_status)
        assert worker._depth_pv_len([10, 20], 4) == (4 if complete else 2), root_status

    # ... but with EVERY child proven it does: no further search can reach an
    # unproven sibling or un-prove a child, so waiting cannot change the move.
    worker, tree, rid, child, leaf = _depth_tree()
    _mark(tree, leaf, 2)
    _mark(tree, child, 2)
    _mark(tree, int(tree.find_child(rid, 11)), 1)
    _mark(tree, rid, 2)
    assert worker._depth_pv_len([10, 20], 4) == 4


# White wins, and Qg6 stalemates at once: a drawn terminal the visit leader
# can sit on (review: pool played it in 19/19 such positions under go depth).
TEMPTING_STALEMATE = ("7k/2K5/8/8/8/8/2Q5/8 w - - 0 1", "c2g6")


@pytest.mark.parametrize("path", ["pool", "walkers"])
def test_go_depth_never_settles_for_the_stalemate_in_a_won_position(path: str) -> None:
    """The drawn PV end must not complete `go depth`: the root is unproven and
    a mate exists. (What an UNPROVEN pool search plays when it is cut off is
    the search's own choice and identical to main -- a random net can rank the
    exact 0 of the stalemate above its own guesses -- so only the stop is
    pinned here, plus the move on the walker path, which never picks it.)"""
    fen, stalemate = TEMPTING_STALEMATE
    worker = _make_worker(path)
    try:
        result = worker.run(
            chess.Board(fen), stop_event=threading.Event(),
            deadline=Deadline(5_000), max_nodes=None, max_depth=10,
        )
        if worker.last_stop_reason == "max_depth":
            assert worker._root_solved_status() == 1 or len(result.pv) >= 10, result.pv
        else:
            # The deadline ended it (walkers: one walker batch early).
            assert worker.last_stop_reason in ("external", "time_margin"), worker.last_stop_reason
        if path == "walkers":
            assert result.bestmove_uci != stalemate
    finally:
        worker.close()


# --- review item 6: a proven-lost search with a pruned drawing move ------------

# Every move but the drawing one lets the opponent mate next move.
DRAW_ESCAPES = [
    ("8/q7/1P6/8/8/3Q4/8/K1k5 w - - 0 1", "b6a7"),   # bxa7 stalemates
    ("4K2k/3q1P2/6Q1/8/8/8/8/8 w - - 0 1", "e8d7"),  # Kxd7 stalemates
]


@pytest.mark.parametrize("path", ["classic", "pool", "rpg", "walkers"])
@pytest.mark.parametrize(("fen", "draw"), DRAW_ESCAPES)
def test_a_lost_root_plays_the_pruned_drawing_move(path: str, fen: str, draw: str) -> None:
    if path == "classic" and fen == DRAW_ESCAPES[1][0]:
        # Pruning no longer leaves Kf8 alone: both moves are searched. But
        # under this unseeded optimistic random net Black's reply search
        # settles on Qe7+ Kxe7 (a stalemate, value 0) and does not reliably
        # try Qd8#, so Kf8 also searches at ~0 and the tie goes to the prior
        # (seed-dependent). That is search strength on a degenerate net, not
        # the pruning defect; the deterministic classic cases are
        # tests/test_root_draw_pruning.py and
        # test_uci_classic_searches_a_root_with_one_nondrawing_move.
        pytest.skip("classic: Kf8 vs Kxd7 is a value tie under this optimistic fixture")
    worker = _make_worker(path, optimistic=True)
    infos: list[dict[str, Any]] = []
    try:
        result = worker.run(
            chess.Board(fen), stop_event=threading.Event(),
            deadline=Deadline(DEADLINE_MS), max_nodes=4096,
            info_cb=lambda **f: infos.append(f),
        )
        assert result.bestmove_uci == draw, (path, result.bestmove_uci)
        assert result.score_cp == 0
        assert result.score_mate is None
        assert infos
        assert infos[-1]["pv"][0] == draw, infos[-1:]
        assert infos[-1]["score_cp"] == 0, infos[-1:]
        assert infos[-1]["score_mate"] is None, infos[-1:]
    finally:
        worker.close()


# --- review item 7: on a proven-lost root, delay mate longest ------------------

LONGEST_LOSS_SEED = 1
LONGEST_LOSS = [
    ("8/1q6/8/6r1/1k6/7K/8/8 w - - 0 1", "h3h4"),  # Kh2 is mated in 1
    ("7q/8/2r5/8/8/8/8/4k2K w - - 0 1", "h1g2"),   # Kg1 is mated in 1
]


@pytest.mark.parametrize("path", ["classic", "pool", "rpg", "walkers"])
@pytest.mark.parametrize(("fen", "longest"), LONGEST_LOSS)
def test_a_proven_lost_root_plays_the_longest_loss(path: str, fen: str, longest: str) -> None:
    if path == "classic" and fen == LONGEST_LOSS[1][0]:
        pytest.skip("classic Gumbel does not prove this root lost inside the deadline")
    # Seeded: whether a random net's search proves the root lost in time
    # depends on its weights.
    worker = _make_worker(path, seed=LONGEST_LOSS_SEED)
    try:
        # `go mate` stops once the root is proven lost (`mate_unreachable`).
        result, _took = _timed_run(worker, fen, mate_moves=3)
        assert worker._root_solved_status() == -1, "fixture: the root is proven lost"
        assert result.bestmove_uci == longest, (path, result.bestmove_uci, result.score_mate)
    finally:
        worker.close()


# --- review round 2: targeted pins -------------------------------------------


def test_only_a_third_occurrence_is_a_forced_repetition_draw() -> None:
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)
    back = int(move_to_index(chess.Move.from_uci("f6g8"), board))
    assert root_forced_draw_actions(board, [back]) == []  # second occurrence
    for uci in ("f6g8", "g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)
    back = int(move_to_index(chess.Move.from_uci("f6g8"), board))
    assert root_forced_draw_actions(board, [back]) == [back]  # third occurrence


def test_go_mate_budget_applies_to_forced_roots_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal root under `go mate N` is bounded by its deadline, not by the
    forced-root simulation budget (shrunk here so a slow CI box passes it)."""
    monkeypatch.setattr(search_module, "_FORCED_ROOT_MATE_SIMS_PER_MOVE", 16)
    for fen, expected in ((FORCED_NO_MATE, "mate_forced_root"), (chess.STARTING_FEN, "external")):
        worker = _make_worker("pool")
        try:
            worker.run(
                chess.Board(fen), stop_event=threading.Event(),
                deadline=Deadline(2_000), max_nodes=None, mate_moves=1,
            )
            assert worker.last_stop_reason == expected, (fen, worker.last_stop_reason)
        finally:
            worker.close()


def _lost_root(board: chess.Board, children: list[chess.Move]) -> tuple[SearchWorker, MCTSTree, int]:
    """A root proven lost: every child marked won for the opponent, no subtrees."""
    worker = SearchWorker.__new__(SearchWorker)
    tree = MCTSTree()
    tree.reserve(64, 640)
    rid = int(tree.add_root(1, 0.0))
    actions = np.array([move_to_index(m, board) for m in children], dtype=np.int32)
    tree.expand(rid, actions, np.full(len(children), 1.0 / len(children)))
    for a in actions.tolist():
        _mark(tree, int(tree.find_child(rid, int(a))), 1)
    _mark(tree, rid, -1)
    worker._tree, worker._root_id = tree, rid
    return worker, tree, rid


def test_longest_loss_prefers_a_move_with_no_proven_mate_distance() -> None:
    board = chess.Board(LONGEST_LOSS[0][0])  # Kh2 is mated in 1 (a board fact)
    short, unmeasured = chess.Move.from_uci("h3h2"), chess.Move.from_uci("h3h4")
    worker, _tree, _rid = _lost_root(board, [short, unmeasured])
    actions = np.array([move_to_index(short, board), move_to_index(unmeasured, board)])
    chosen = worker._longest_proven_loss(board, actions, np.array([9, 1]), int(actions[0]))
    assert chosen == int(actions[1])


def test_drawing_escape_honours_searchmoves() -> None:
    fen, draw = DRAW_ESCAPES[0]
    board = chess.Board(fen)
    draw_move = chess.Move.from_uci(draw)
    others = [m for m in board.legal_moves if m != draw_move]
    worker, _tree, _rid = _lost_root(board, others)
    draw_idx = int(move_to_index(draw_move, board))
    assert worker._drawing_escape_action(board, None) == draw_idx
    allowed = {int(move_to_index(m, board)) for m in others}
    assert worker._drawing_escape_action(board, allowed) is None


def test_drawing_escape_honours_only_a_classic_survivor_that_draws() -> None:
    """The classic runner's below-0 fallback returns a pruned draw that is not a
    tree child; that survivor is played. A non-child survivor that does NOT end
    the game drawn (a stale index from another search) is never played."""
    board = chess.Board("7k/8/6KP/8/8/8/8/8 w - - 0 1")
    stalemate = chess.Move.from_uci("h6h7")
    walk = chess.Move.from_uci("g6f6")
    searched = [m for m in board.legal_moves if m not in (stalemate, walk)]
    worker = SearchWorker.__new__(SearchWorker)
    tree = MCTSTree()
    tree.reserve(64, 640)
    rid = int(tree.add_root(1, 0.0))
    tree.expand(rid, np.array([move_to_index(m, board) for m in searched], dtype=np.int32),
                np.full(len(searched), 1.0 / len(searched)))
    worker._tree, worker._root_id = tree, rid
    worker._last_gumbel_action_idx = int(move_to_index(stalemate, board))
    assert worker._drawing_escape_action(board, None) == int(move_to_index(stalemate, board))
    assert worker._drawing_escape_action(board, {int(move_to_index(walk, board))}) is None
    worker._last_gumbel_action_idx = int(move_to_index(walk, board))
    assert worker._drawing_escape_action(board, None) is None


# --- review round 3: the longest loss under a starved mate walk ---------------

# White to move, lost. Kh2 and a4 are mated in one (board facts); Kh4 and a3 are not.
LOSS_FAN = "8/1q6/8/6r1/1k6/7K/P7/8 w - - 0 1"


def _idx(board: chess.Board, *ucis: str) -> np.ndarray:
    return np.array([move_to_index(chess.Move.from_uci(u), board) for u in ucis])


def test_longest_loss_gives_every_child_its_own_share_of_the_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One node of walk per child: a3 spends its share, and Kh4 must still be
    measured (no proven distance), not read as exhausted -- so the `chosen`
    tie-break between the two unmeasured moves decides, not walk order."""
    monkeypatch.setattr(search_module, "_MATE_WALK_NODE_LIMIT", 3)
    board = chess.Board(LOSS_FAN)
    moves = [chess.Move.from_uci(u) for u in ("a2a3", "h3h4", "h3h2")]
    worker, _tree, _rid = _lost_root(board, moves)
    actions = _idx(board, "a2a3", "h3h4", "h3h2")
    chosen = worker._longest_proven_loss(board, actions, np.array([1, 1, 9]), int(actions[1]))
    assert chosen == int(actions[1])


def _lost_root_with_a_deep_child(board: chess.Board, deep: str, others: list[str]):
    """`_lost_root` where ``deep`` has a proven reply under it, so measuring it
    takes a second walk node."""
    worker, tree, rid = _lost_root(board, [chess.Move.from_uci(u) for u in (deep, *others)])
    after = board.copy()
    after.push_uci(deep)
    reply = next(iter(after.legal_moves))
    node = int(tree.find_child(rid, int(move_to_index(chess.Move.from_uci(deep), board))))
    tree.expand(node, np.array([move_to_index(reply, after)], dtype=np.int32), np.array([1.0]))
    _mark(tree, int(tree.find_child(node, int(move_to_index(reply, after)))), -1)
    return worker


def test_an_exhausted_walk_ranks_above_mate_in_one_and_below_no_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(search_module, "_MATE_WALK_NODE_LIMIT", 2)  # one node a child
    board = chess.Board(LOSS_FAN)
    # a3's walk runs out; Kh2 is mated in one even with no walk left.
    worker = _lost_root_with_a_deep_child(board, "a2a3", ["h3h2"])
    actions = _idx(board, "a2a3", "h3h2")
    assert worker._longest_proven_loss(
        board, actions, np.array([1, 9]), int(actions[1])) == int(actions[0])
    # ... but a move with no proven mate distance beats the exhausted walk.
    worker = _lost_root_with_a_deep_child(board, "a2a3", ["h3h4"])
    actions = _idx(board, "a2a3", "h3h4")
    assert worker._longest_proven_loss(
        board, actions, np.array([9, 1]), int(actions[0])) == int(actions[1])


def test_longest_loss_never_plays_mate_in_one_on_a_tiny_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review demo: with the walk budget at one node, Kh4's walk used it and
    Kh2 (mated in one) was then read as unmeasured -- the best rank."""
    monkeypatch.setattr(search_module, "_MATE_WALK_NODE_LIMIT", 1)
    board = chess.Board(LONGEST_LOSS[0][0])
    moves = [chess.Move.from_uci(u) for u in ("h3h4", "h3h2")]
    worker, _tree, _rid = _lost_root(board, moves)
    actions = _idx(board, "h3h4", "h3h2")
    chosen = worker._longest_proven_loss(board, actions, np.array([1, 9]), int(actions[1]))
    assert chosen == int(actions[0])


# --- review round 3: the at-once answer from the reused tree ------------------


def _tree_answer_worker(board: chess.Board, moves: list[chess.Move], visits: list[int]):
    worker, tree, rid = _lost_root(board, moves)
    for move, n in zip(moves, visits, strict=True):
        node = int(tree.find_child(rid, int(move_to_index(move, board))))
        for _ in range(n):
            tree.backprop(np.array([rid, node], dtype=np.int32), 0.0)
    worker._tree_fen = board.fen()
    return worker


def test_tree_answer_on_a_lost_root_plays_the_drawing_escape() -> None:
    fen, draw = DRAW_ESCAPES[0]
    board = chess.Board(fen)
    others = [m for m in board.legal_moves if m.uci() != draw]
    worker = _tree_answer_worker(board, others, [20] + [1] * (len(others) - 1))
    assert worker.tree_root_bestmove(board) == draw
    # `searchmoves` without the drawing move: no escape.
    assert worker.tree_root_bestmove(board, searchmoves=(others[0].uci(),)) == others[0].uci()


def test_tree_answer_on_a_lost_root_avoids_mate_in_one() -> None:
    board = chess.Board(LONGEST_LOSS[0][0])  # Kh2 is mated in 1
    moves = [chess.Move.from_uci(u) for u in ("h3h2", "h3h4")]
    worker = _tree_answer_worker(board, moves, [20, 2])  # Kh4 under the min visits
    assert worker.tree_root_bestmove(board) == "h3h4"


def test_walker_time_stop_is_named_when_the_batch_reserve_decays(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI flake (`...stalemate_in_a_won_position[walkers]`: last_stop_reason None).

    A walker chunk's early stop is computed from `batch_ms_hi`, a decayed
    maximum, when the chunk launches; the loop re-read it after the chunk, by
    when it could have decayed. A chunk launched inside its own reserve stops
    before any sim, the re-read reserve said "not yet", and the go ended through
    the zero-progress break with no stop reason. Here the reserve reads 1.5 s at
    each chunk launch and 0 at each re-check: the go must end as a time stop."""
    worker = _make_worker("walkers")
    calls = [0]

    def decaying_reserve(*_args: Any) -> float:
        calls[0] += 1
        return 1500.0 if calls[0] % 2 == 1 else 0.0  # launch, then re-check

    monkeypatch.setattr(worker, "_walker_stop_reserve_ms", decaying_reserve)
    try:
        t0 = time.monotonic()
        worker.run(
            chess.Board(), stop_event=threading.Event(),
            deadline=Deadline(3_000), max_nodes=None, max_depth=50,
        )
        took = time.monotonic() - t0
        assert worker.last_stop_reason == "time_margin", worker.last_stop_reason
        assert 1.0 < took < 2.5, took  # ended at the 1.5 s reserve, not the deadline
        assert calls[0] >= 2
    finally:
        worker.close()


def test_a_zero_progress_bounded_search_names_its_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk that completes nothing cannot move a node or depth bound, so the
    search stops at once -- and must say why, not leave last_stop_reason None."""
    worker = _make_worker("walkers")
    monkeypatch.setattr(worker, "_run_one_chunk", lambda *_a, **_k: (0.0, 0))
    try:
        worker.run(
            chess.Board(), stop_event=threading.Event(),
            deadline=Deadline(30_000), max_nodes=None, max_depth=50,
        )
        assert worker.last_stop_reason == "no_progress", worker.last_stop_reason
    finally:
        worker.close()
