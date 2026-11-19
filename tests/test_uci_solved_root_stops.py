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
)
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


def test_a_pv_ending_on_a_proven_node_completes_the_depth_bound() -> None:
    worker = SearchWorker.__new__(SearchWorker)
    tree = MCTSTree()
    tree.reserve(64, 640)
    rid = int(tree.add_root(1, 0.0))
    tree.expand(rid, np.array([10, 11], dtype=np.int32), np.array([0.5, 0.5]))
    child = int(tree.find_child(rid, 10))
    tree.expand(child, np.array([20], dtype=np.int32), np.array([1.0]))
    leaf = int(tree.find_child(child, 20))
    worker._tree, worker._root_id = tree, rid
    assert worker._depth_pv_len([10, 20], 4) == 2  # unproven leaf: keep deepening
    tree.mark_solved_path(np.array([leaf], dtype=np.int32), 2)  # a drawn terminal
    assert worker._depth_pv_len([10, 20], 4) == 4
    assert worker._depth_pv_len([10, 20], None) == 2
    assert worker._depth_pv_len([], 4) == 0


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
        pytest.skip(
            "classic: pruning leaves one move, which the shared gumbel_c root "
            "returns without simulating, so no proof (and no escape) can arise",
        )
    worker = _make_worker(path, optimistic=True)
    infos: list[dict[str, Any]] = []
    try:
        result = worker.run(
            chess.Board(fen), stop_event=threading.Event(),
            deadline=Deadline(DEADLINE_MS), max_nodes=4096,
            info_cb=lambda **f: infos.append(f),
        )
        assert result.bestmove_uci == draw, (path, result.bestmove_uci)
        assert result.score_mate is None
        assert infos
        assert infos[-1]["pv"][0] == draw, infos[-1:]
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
