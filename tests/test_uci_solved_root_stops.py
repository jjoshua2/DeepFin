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
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.root_parallel_gumbel import (
    RootParallelGumbelConfig,
    RootParallelGumbelPool,
    _SearchState,
)
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline

# Black to move, mated in one whatever it plays (a6/a5, then Qg7# / Qf8#).
MATED_IN_ONE = "7k/p4Q2/6K1/8/8/8/8/8 b - - 0 1"
# A deadline the fixed search must never come near; a mutant runs into it.
# One root-parallel chunk is a whole halving schedule (~4 s on two capped CPU
# threads), so the fixed search answers after one or two chunks.
DEADLINE_MS = 30_000
FAST_S = 15.0


def _make_evaluator() -> Any:
    cfg = ModelConfig(
        input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
    )
    torch_model = build_model(cfg)
    torch_model.eval()
    return ThreadSafeGPUDispatcher(DirectGPUEvaluator(
        torch_model, device="cpu", max_batch=64, use_amp=False, n_slots=2,
    ))


def _make_worker(path: str) -> SearchWorker:
    worker = SearchWorker(
        _make_evaluator(), device="cpu",
        gumbel_cfg=GumbelConfig(
            input_extra_features="v1", simulations=64, add_noise=False, temperature=0.0,
        ),
        chunk_sims=64, n_walkers=1,
    )
    pair = [_make_evaluator(), _make_evaluator()]
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
        assert worker.last_stop_reason == "depth_solved_root", worker.last_stop_reason
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
