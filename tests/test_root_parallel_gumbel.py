"""Root-parallel Gumbel pool tests (design_multi_gpu_search.md §4, CPU evaluators).

Covers:
  - halving schedule/budget arithmetic pinned against the classic C
    implementation's (`gss_begin_round` / `gss_score_and_halve` in
    mcts/_mcts_tree.c)
  - semantics under parallelism: g in {2, 4} produce IDENTICAL phase
    schedules, halving decisions, final move, and root child visits to the
    serial reference (the same orchestrator forced to g=1) with a
    deterministic pure-function evaluator stub
  - ownership invariant: no candidate arena is touched by two groups
    concurrently (test-only touch hook)
  - stop behavior: stop_event mid-phase returns promptly with a selection
    from completed work, and the pool stays usable
  - degenerate dispatch: single device / SearchParallel pucv never construct
    the module
  - SearchWorker / Engine integration smoke + teardown
"""
from __future__ import annotations

import threading
import time
import zlib
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.root_parallel_gumbel import (
    RootParallelGumbelConfig,
    RootParallelGumbelPool,
    halving_keep_count,
    halving_schedule,
    halving_visits_per_action,
)
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


def _make_evaluator(max_batch: int = 64) -> Any:
    cfg = ModelConfig(input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    inner = DirectGPUEvaluator(
        model, device="cpu", max_batch=max_batch, use_amp=False, n_slots=2,
    )
    return ThreadSafeGPUDispatcher(inner)


class _DeterministicStubEvaluator:
    """Pure-function evaluator: pol/wdl logits are a deterministic hash of the
    encoded input row, so the same position gets the same eval no matter which
    instance/thread/group computes it. That makes per-candidate simulation
    outcomes identical across group counts — exactly the premise under which
    the design's semantics contract promises identical root decisions."""

    n_slots = 2

    def __init__(
        self, max_batch: int = 64, planes: int = 146, sleep_s: float = 0.0,
    ) -> None:
        self.calls = 0
        self._sleep_s = float(sleep_s)
        self._lock = threading.Lock()
        self._bufs = [
            np.zeros((max_batch, planes, 8, 8), dtype=np.float32),
            np.zeros((max_batch, planes, 8, 8), dtype=np.float32),
        ]

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        return self._bufs[slot][:bsz]

    def evaluate_inplace_async(
        self, bsz: int, *, slot: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        with self._lock:
            self.calls += 1
        if self._sleep_s > 0.0:
            time.sleep(self._sleep_s)
        pol = np.empty((bsz, 4672), dtype=np.float32)
        wdl = np.empty((bsz, 3), dtype=np.float32)
        for i in range(bsz):
            seed = zlib.crc32(self._bufs[slot][i].tobytes()) & 0xFFFFFFFF
            r = np.random.default_rng(seed)
            pol[i] = r.standard_normal(4672).astype(np.float32)
            wdl[i] = r.standard_normal(3).astype(np.float32)
        return pol, wdl, None


def _root_eval(seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    pol = r.standard_normal(4672).astype(np.float32)
    wdl = np.array([0.4, 0.1, -0.3], dtype=np.float32)
    return pol, wdl


def _make_pool(
    g: int,
    *,
    topk: int = 8,
    gather: int = 8,
    rng_seed: int = 7,
    sleep_s: float = 0.0,
) -> RootParallelGumbelPool:
    evaluators = [_DeterministicStubEvaluator(sleep_s=sleep_s) for _ in range(g)]
    cfg = RootParallelGumbelConfig(n_groups=g, gather=gather)
    gcfg = GumbelConfig(input_extra_features="v1",
        simulations=64, topk=topk, add_noise=True, gumbel_scale=1.0,
        temperature=0.0,
    )
    return RootParallelGumbelPool(
        cfg, gcfg, evaluators=evaluators, rng=np.random.default_rng(rng_seed),
    )


def _run_search(
    pool: RootParallelGumbelPool, *, target_sims: int,
) -> tuple[float, int, Any, np.ndarray, np.ndarray]:
    board = chess.Board()
    pol, wdl = _root_eval()
    tree = MCTSTree()
    tree.reserve(50_000, 500_000)
    rid = pool.prepare_root(tree=tree, board=board, pol_logits=pol, wdl_logits=wdl)
    value, action = pool.run(
        target_sims=target_sims, stop_event=threading.Event(),
    )
    actions, visits = tree.get_children_visits(rid)
    return value, action, pool.last_stats(), actions, visits


# --- schedule/budget arithmetic vs the classic C implementation ---------------


def _c_reference_schedule(
    n_cands: int, budget: int, div: int = 2,
) -> list[tuple[int, int]]:
    """Line-for-line mirror of the classic C schedule (mcts/_mcts_tree.c):

    - gss_begin_round: rem<=1 -> vpa=budget; else rounds_left counts
      ceil-divisions by div down to 1 and vpa = max(1, budget/(rem*rounds));
    - gss_score_and_halve: budget -= vpa*rem (floored at 0), then
      keep = clamp(ceil(rem/div), 1, rem-1) when rem > 1.
    """
    phases: list[tuple[int, int]] = []
    rem, b = int(n_cands), int(budget)
    while b > 0 and rem > 0:
        if rem <= 1:
            vpa = b
        else:
            rounds_left, tmp = 0, rem
            while tmp > 1:
                rounds_left += 1
                tmp = (tmp + div - 1) // div
            vpa = max(1, b // (rem * rounds_left))
        phases.append((rem, vpa))
        b = max(0, b - vpa * rem)
        if rem > 1:
            keep = (rem + div - 1) // div
            rem = max(1, min(keep, rem - 1))
    return phases


def test_halving_schedule_matches_classic_c_arithmetic() -> None:
    for div in (2, 3):
        for n_cands in (1, 2, 3, 4, 5, 8, 12, 16, 32):
            for budget in (1, 2, 7, 16, 50, 64, 200, 800, 2048):
                assert halving_schedule(n_cands, budget, div) == (
                    _c_reference_schedule(n_cands, budget, div)
                ), f"schedule mismatch at n={n_cands} b={budget} div={div}"


def test_halving_helpers_edge_cases() -> None:
  # Lone survivor absorbs the whole remaining budget (gss_begin_round).
    assert halving_visits_per_action(1, 37) == 37
  # keep is clamped to guarantee progress: 2 candidates always halve to 1.
    assert halving_keep_count(2) == 1
    assert halving_keep_count(3) == 2
    assert halving_keep_count(2, halving_div=4) == 1


# --- semantics under parallelism ----------------------------------------------


def test_parallel_groups_identical_to_serial_reference() -> None:
    """g in {2, 4} must reproduce the serial (g=1 forced through the same
    orchestrator) phase schedule, per-phase survivors, final move, value, and
    root child visits exactly — the deterministic stub makes per-candidate
    outcomes a pure function of the position, so any deviation is a
    scheduling/ownership bug."""
    target = 192
    ref_pool = _make_pool(1)
    try:
        ref = _run_search(ref_pool, target_sims=target)
    finally:
        ref_pool.close()
    ref_value, ref_action, ref_stats, ref_actions, ref_visits = ref

    assert ref_stats.phases, "serial reference ran no phases"
    for g in (2, 4):
        pool = _make_pool(g)
        try:
            assert pool.n_groups == g
            value, action, stats, actions, visits = _run_search(
                pool, target_sims=target,
            )
        finally:
            pool.close()
        assert action == ref_action, f"final move diverged at g={g}"
        assert value == ref_value, f"root value diverged at g={g}"
        schedule = [
            (p.candidates, p.visits_per_action, p.survivor_actions)
            for p in stats.phases
        ]
        ref_schedule = [
            (p.candidates, p.visits_per_action, p.survivor_actions)
            for p in ref_stats.phases
        ]
        assert schedule == ref_schedule, f"halving decisions diverged at g={g}"
        np.testing.assert_array_equal(actions, ref_actions)
        np.testing.assert_array_equal(visits, ref_visits)


def test_run_schedule_matches_pure_helper() -> None:
    """The live run's (candidates, vpa) sequence must equal the pure
    ``halving_schedule`` for the sampled candidate count."""
    pool = _make_pool(2, topk=8)
    try:
        _, _, stats, _, _ = _run_search(pool, target_sims=128)
    finally:
        pool.close()
    n0 = stats.phases[0].candidates
    assert [
        (p.candidates, p.visits_per_action) for p in stats.phases
    ] == halving_schedule(n0, 128, 2)
  # All budgeted work landed (no stop): each phase ran cands * vpa sims.
    for p in stats.phases:
        assert p.sims_completed == p.candidates * p.visits_per_action
    assert stats.target_sims == 128
    assert stats.elapsed_seconds >= 0.0
    assert sum(stats.group_sims) == sum(
        p.sims_completed for p in stats.phases
    )


# --- ownership invariant --------------------------------------------------------


def test_ownership_invariant_no_concurrent_touch() -> None:
    pool = _make_pool(4, topk=16)
    active: dict[int, int] = {}
    violations: list[tuple[str, int, int, int | None]] = []
    hlock = threading.Lock()

    def hook(event: str, action: int, group: int) -> None:
        with hlock:
            if event == "claim":
                if action in active:
                    violations.append((event, action, group, active[action]))
                active[action] = group
            elif event == "touch":
                if active.get(action) != group:
                    violations.append((event, action, group, active.get(action)))
            elif event == "release":
                active.pop(action, None)

    pool.touch_hook = hook
    try:
        _run_search(pool, target_sims=256)
    finally:
        pool.close()
    assert not violations, f"arena ownership violated: {violations[:5]}"
  # Every candidate was claimed at most once per phase.
    seen: set[tuple[int, int]] = set()
    for phase, action, _group in pool.owner_history:
        assert (phase, action) not in seen
        seen.add((phase, action))
    assert seen, "no work was dispatched"


# --- stop behavior ---------------------------------------------------------------


def test_stop_mid_phase_returns_promptly_and_pool_survives() -> None:
    pool = _make_pool(2, topk=16, gather=4, sleep_s=0.005)
    try:
        board = chess.Board()
        pol, wdl = _root_eval()
        tree = MCTSTree()
        tree.reserve(50_000, 500_000)
        rid = pool.prepare_root(
            tree=tree, board=board, pol_logits=pol, wdl_logits=wdl,
        )
        assert pool._state is not None
        legal = set(np.nonzero(pool._state.pri > 0)[0].astype(int).tolist())
        stop = threading.Event()
        timer = threading.Timer(0.3, stop.set)
        timer.start()
        t0 = time.monotonic()
        value, action = pool.run(target_sims=200_000, stop_event=stop)
        elapsed = time.monotonic() - t0
        timer.cancel()
        assert elapsed < 10.0, "stop_event did not interrupt the phase"
        assert action in legal
        assert np.isfinite(value)
  # Selection came from completed work: the survivor has real visits
  # (at least the phase-1 budget that finished before the stop).
        _actions, visits = tree.get_children_visits(rid)
        assert int(visits.sum()) > 0
  # Pool remains usable for the next chunk.
        value2, action2 = pool.run(
            target_sims=32, stop_event=threading.Event(),
        )
        assert action2 in legal
        assert np.isfinite(value2)
    finally:
        pool.close()


# --- degenerate roots -------------------------------------------------------------


def test_single_legal_move_finishes_without_search() -> None:
  # Black king a8, white queen b2 (controls the whole b-file), white king a1:
  # black's only legal move is Ka8-a7. Classic finishes such roots without
  # search; the pool must reproduce that (no group evals at all).
    board = chess.Board("k7/8/8/8/8/8/1Q6/K7 b - - 0 1")
    legal = list(board.legal_moves)
    assert len(legal) == 1
    pool = _make_pool(2)
    try:
        pol, wdl = _root_eval()
        tree = MCTSTree()
        tree.reserve(10_000, 100_000)
        pool.prepare_root(tree=tree, board=board, pol_logits=pol, wdl_logits=wdl)
        evals_before = sum(
            ev.calls for ev in pool._evals  # stub evaluators
        )
        _value, action = pool.run(target_sims=64, stop_event=threading.Event())
        evals_after = sum(ev.calls for ev in pool._evals)
        assert evals_after == evals_before, "finished root must not search"
        assert action == int(move_to_index(legal[0], board))
        # Finished roots must still report completed sims so SearchWorker
        # advances total_nodes (go nodes / infinite would otherwise spin).
        stats = pool.last_stats()
        assert sum(p.sims_completed for p in stats.phases) == 64
    finally:
        pool.close()


def test_finished_root_nodes_advance_under_searchworker() -> None:
    """SearchWorker node accounting must advance on RPG finished roots."""
    board = chess.Board("k7/8/8/8/8/8/1Q6/K7 b - - 0 1")
    assert len(list(board.legal_moves)) == 1
    worker = _make_worker(chunk_sims=32)
    worker.install_root_parallel_gumbel(
        [_make_evaluator(max_batch=64), _make_evaluator(max_batch=64)],
        gather=8, as_factories=False,
    )
    try:
        result = worker.run(
            board, stop_event=threading.Event(),
            deadline=Deadline(5_000), max_nodes=32,
        )
        assert result.bestmove_uci
        assert result.nodes >= 32, (
            f"finished-root RPG must count the chunk, got nodes={result.nodes}"
        )
    finally:
        worker.close()


def test_prepare_root_honors_terminal_shortcut_gate() -> None:
    """allow_terminal_shortcuts=False must skip mate-in-1 finish (ponder/analysis)."""
    # Back-rank mate in 1: Re8#.
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    legal = list(board.legal_moves)
    mate_uci = None
    for m in legal:
        board.push(m)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            mate_uci = m.uci()
            break
    assert mate_uci is not None, "fixture must contain a mate-in-1"
    pol, wdl = _root_eval()
    # Winning root eval so mate shortcut is eligible under classic rules.
    wdl = np.array([5.0, 0.0, -5.0], dtype=np.float32)

    pool = _make_pool(1)
    try:
        tree = MCTSTree()
        tree.reserve(10_000, 100_000)
        pool.prepare_root(
            tree=tree, board=board, pol_logits=pol, wdl_logits=wdl,
            allow_terminal_shortcuts=True,
        )
        assert pool._state is not None
        assert pool._state.finished_action is not None, "mate shortcut should fire"

        tree2 = MCTSTree()
        tree2.reserve(10_000, 100_000)
        pool.prepare_root(
            tree=tree2, board=board, pol_logits=pol, wdl_logits=wdl,
            allow_terminal_shortcuts=False,
        )
        assert pool._state is not None
        assert pool._state.finished_action is None, (
            "ponder/analysis must not mate-shortcut when gate is off"
        )
        assert pool._state.search_legal.size > 1
    finally:
        pool.close()


# --- SearchWorker integration -------------------------------------------------------


def _make_worker(chunk_sims: int = 64) -> SearchWorker:
    return SearchWorker(
        _make_evaluator(max_batch=64), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1",
            simulations=chunk_sims, add_noise=False, temperature=0.0,
        ),
        chunk_sims=chunk_sims, n_walkers=1,
    )


def test_searchworker_install_root_parallel_gumbel_produces_bestmove() -> None:
    worker = _make_worker()
    p0 = _make_evaluator(max_batch=64)
    p1 = _make_evaluator(max_batch=64)
    worker.install_root_parallel_gumbel([p0, p1], gather=8, as_factories=False)
    try:
        result = worker.run(
            chess.Board(), stop_event=threading.Event(),
            deadline=Deadline(10_000), max_nodes=128,
        )
        assert len(result.bestmove_uci) >= 4
        assert result.nodes >= 64
        stats = worker.last_root_parallel_gumbel_stats()
        assert stats is not None
        assert stats.phases
        assert stats.phases[0].candidates >= 2
  # The played move is the halving survivor, classic-Gumbel style.
        assert worker._last_gumbel_action_idx is not None
    finally:
        worker.close()


def test_searchworker_rpg_refuses_cross_move_tree_reuse() -> None:
    """v1: fresh per-candidate arenas each move — advance_root must refuse."""
    worker = _make_worker()
    p0 = _make_evaluator(max_batch=64)
    worker.install_root_parallel_gumbel([p0], gather=8, as_factories=False)
    try:
        board = chess.Board()
        worker.run(
            board, stop_event=threading.Event(),
            deadline=Deadline(10_000), max_nodes=64,
        )
        assert worker._tree is not None
        move = next(iter(board.legal_moves))
        assert worker.advance_root(board, [move]) is False
    finally:
        worker.close()


def test_searchworker_clear_root_parallel_gumbel_reverts() -> None:
    worker = _make_worker()
    p0 = _make_evaluator(max_batch=64)
    worker.install_root_parallel_gumbel([p0], gather=8, as_factories=False)
    assert worker._rpg_pool is not None
    assert worker._is_shared_tree_path(None)
    worker.clear_root_parallel_gumbel()
    assert worker._rpg_pool is None
    assert not worker._is_shared_tree_path(None)
    result = worker.run(
        chess.Board(), stop_event=threading.Event(),
        deadline=Deadline(5_000), max_nodes=64,
    )
    assert result.bestmove_uci
    worker.close()


def test_install_multi_gpu_pucv_clears_rpg_and_vice_versa() -> None:
    worker = _make_worker()
    try:
        worker.install_root_parallel_gumbel(
            [_make_evaluator()], gather=8, as_factories=False,
        )
        assert worker._rpg_pool is not None
        worker.install_multi_gpu_pucv(
            [_make_evaluator()], gather=8, as_factories=False,
        )
        assert worker._rpg_pool is None
        assert worker._pucv_pool is not None
        worker.install_root_parallel_gumbel(
            [_make_evaluator()], gather=8, as_factories=False,
        )
        assert worker._pucv_pool is None
        assert worker._rpg_pool is not None
    finally:
        worker.close()


# --- degenerate dispatch (Engine option surface) --------------------------------------


def test_engine_search_parallel_refused_below_two_devices() -> None:
    """SearchParallel gumbel with < 2 device factories must never construct
    the pool and must revert the option to pucv."""
    worker = _make_worker()
    engine = Engine(
        worker,
        rebuild_multi_gpu_pucv_factories=lambda _mb, _g: [_make_evaluator],
        options=EngineOptions(),
    )
    try:
        engine._set_search_parallel("gumbel")
        assert worker._rpg_pool is None
        assert engine._options.search_parallel == "pucv"
    finally:
        engine.close()


def test_engine_search_parallel_no_factories_wired() -> None:
    worker = _make_worker()
    engine = Engine(worker, options=EngineOptions())
    try:
        engine._set_search_parallel("gumbel")
        assert worker._rpg_pool is None
        assert engine._options.search_parallel == "pucv"
  # pucv value is always accepted and never constructs the module.
        engine._set_search_parallel("pucv")
        assert worker._rpg_pool is None
    finally:
        engine.close()


def test_engine_search_parallel_installs_and_reverts() -> None:
    worker = _make_worker()

    def factories(_mb: int, _g: int) -> list[Any]:
        return [_make_evaluator, _make_evaluator]

    engine = Engine(
        worker,
        rebuild_multi_gpu_pucv_factories=factories,
        options=EngineOptions(),
    )
    try:
        engine._set_search_parallel("gumbel")
        assert worker._rpg_pool is not None
        assert engine._options.search_parallel == "gumbel"
        result = worker.run(
            chess.Board(), stop_event=threading.Event(),
            deadline=Deadline(10_000), max_nodes=64,
        )
        assert result.bestmove_uci
        engine._set_search_parallel("pucv")
        assert worker._rpg_pool is None
        assert engine._options.search_parallel == "pucv"
    finally:
        engine.close()


def test_engine_max_batch_reinstalls_gumbel() -> None:
    """MaxBatch rebuild must reinstall SearchParallel=gumbel, not drop it."""
    worker = _make_worker()
    builds: list[tuple[int, int]] = []

    def factories(mb: int, g: int) -> list[Any]:
        builds.append((mb, g))
        return [_make_evaluator, _make_evaluator]

    def rebuild_eval(mb: int, _cache: int = 0, *, n_walkers: int | None = None) -> Any:
        del n_walkers
        return _make_evaluator(max_batch=mb)

    engine = Engine(
        worker,
        rebuild_evaluator=rebuild_eval,
        rebuild_multi_gpu_pucv_factories=factories,
        options=EngineOptions(max_batch=64, eval_cache_entries=0),
    )
    try:
        engine._set_search_parallel("gumbel")
        assert worker._rpg_pool is not None
        n_before = len(builds)
        engine._set_max_batch("128")
        assert engine._options.search_parallel == "gumbel"
        assert worker._rpg_pool is not None, "MaxBatch must reinstall RPG pool"
        assert len(builds) > n_before
        assert builds[-1][0] == 128
    finally:
        engine.close()


def test_engine_leaving_gumbel_restores_use_vl() -> None:
    """SearchParallel pucv after gumbel must restore UseVL when multi-GPU is off."""
    worker = _make_worker()
    # Primary evaluator is 2-slot async so UseVL can install.
    worker = SearchWorker(
        _make_evaluator(max_batch=64), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False, temperature=0.0),
        chunk_sims=64, n_walkers=1,
    )
    worker.set_use_pucv(True, gather=32)
    assert worker._pucv is not None

    def factories(_mb: int, _g: int) -> list[Any]:
        return [_make_evaluator, _make_evaluator]

    engine = Engine(
        worker,
        rebuild_multi_gpu_pucv_factories=factories,
        options=EngineOptions(use_vl=True, vl_gather=32, use_multi_gpu_pucv=False),
    )
    try:
        engine._set_search_parallel("gumbel")
        assert worker._rpg_pool is not None
        assert worker._pucv is None  # install tore down single-thread PUCV
        engine._set_search_parallel("pucv")
        assert worker._rpg_pool is None
        assert worker._pucv is not None, "UseVL path must be restored after leaving Gumbel"
    finally:
        engine.close()


def test_clear_rpg_invalidates_tree_for_classic() -> None:
    """Clearing RPG must drop the persistent tree so classic does not score
    with a stale root N/W left by candidate-only backprop."""
    worker = _make_worker()
    worker.install_root_parallel_gumbel(
        [_make_evaluator(max_batch=64), _make_evaluator(max_batch=64)],
        gather=8, as_factories=False,
    )
    try:
        worker.run(
            chess.Board(), stop_event=threading.Event(),
            deadline=Deadline(10_000), max_nodes=64,
        )
        assert worker._tree is not None
        worker.clear_root_parallel_gumbel()
        assert worker._rpg_pool is None
        assert worker._tree is None, "RPG tree must not be reused by classic"
    finally:
        worker.close()


def test_searchmoves_fallback_resets_rpg_tree() -> None:
    """searchmoves forces classic Gumbel; any RPG-shaped tree must be dropped."""
    worker = _make_worker()
    worker.install_root_parallel_gumbel(
        [_make_evaluator(max_batch=64), _make_evaluator(max_batch=64)],
        gather=8, as_factories=False,
    )
    try:
        board = chess.Board()
        worker.run(
            board, stop_event=threading.Event(),
            deadline=Deadline(10_000), max_nodes=64,
        )
        assert worker._tree is not None
        dirty = worker._tree
        # Same position with searchmoves — classic fallback must not reuse dirty.
        result = worker.run(
            board, stop_event=threading.Event(),
            deadline=Deadline(5_000), max_nodes=32,
            root_moves=("e2e4",),
        )
        assert result.bestmove_uci == "e2e4"
        assert worker._tree is not dirty, "classic fallback must not reuse RPG tree"
    finally:
        worker.close()


def test_use_multi_gpu_pucv_forces_search_parallel_pucv() -> None:
    """Enabling UseMultiGpuPUCV must not leave search_parallel=gumbel sticky
    (MaxBatch would silently reinstall RPG)."""
    worker = _make_worker()

    def factories(_mb: int, _g: int) -> list[Any]:
        return [_make_evaluator, _make_evaluator]

    def rebuild_eval(mb: int, _cache: int = 0, *, n_walkers: int | None = None) -> Any:
        del n_walkers
        return _make_evaluator(max_batch=mb)

    engine = Engine(
        worker,
        rebuild_evaluator=rebuild_eval,
        rebuild_multi_gpu_pucv_factories=factories,
        options=EngineOptions(max_batch=64, eval_cache_entries=0),
    )
    try:
        engine._set_search_parallel("gumbel")
        assert worker._rpg_pool is not None
        assert engine._options.search_parallel == "gumbel"
        engine._set_use_multi_gpu_pucv("true")
        assert worker._rpg_pool is None
        assert worker._pucv_pool is not None
        assert engine._options.search_parallel == "pucv"
        assert engine._options.use_multi_gpu_pucv is True
        # MaxBatch must keep PUCV, not flip back to Gumbel.
        engine._set_max_batch("128")
        assert engine._options.search_parallel == "pucv"
        assert worker._pucv_pool is not None
        assert worker._rpg_pool is None
    finally:
        engine.close()


def test_threads_under_gumbel_does_not_materialize_walker() -> None:
    """Threads setoption while SearchParallel=gumbel must not spawn a live
    walker pool beside RPG; the count is stored for leave-gumbel restore."""
    worker = _make_worker()

    def factories(_mb: int, _g: int) -> list[Any]:
        return [_make_evaluator, _make_evaluator]

    engine = Engine(
        worker,
        rebuild_multi_gpu_pucv_factories=factories,
        options=EngineOptions(threads=1),
    )
    try:
        engine._set_search_parallel("gumbel")
        assert worker._rpg_pool is not None
        assert worker._walker_pool is None
        engine._set_threads("4")
        assert engine._options.threads == 4
        assert worker._n_walkers == 4
        assert worker._walker_pool is None, "must not allocate walker beside RPG"
        assert worker._rpg_pool is not None
        engine._set_search_parallel("pucv")
        assert worker._rpg_pool is None
        assert worker._walker_pool is not None, "leave-gumbel must build walker pool"
        assert worker._n_walkers == 4
    finally:
        engine.close()


def test_install_rpg_wires_eval_cache_entries() -> None:
    """EvalCacheEntries must reach per-group PucvChunkers (not a silent no-op)."""
    worker = SearchWorker(
        _make_evaluator(max_batch=64), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False, temperature=0.0),
        chunk_sims=64, n_walkers=1, eval_cache_entries=128,
    )
    try:
        worker.install_root_parallel_gumbel(
            [_make_evaluator(max_batch=64), _make_evaluator(max_batch=64)],
            gather=8, as_factories=False,
        )
        assert worker._rpg_pool is not None
        assert worker._rpg_pool._cfg.eval_cache_entries == 128
        for ch in worker._rpg_pool._chunkers:
            assert ch is not None
            assert ch.cache_stats() is not None, "chunker must own an eval cache"
    finally:
        worker.close()


def test_option_matrix_multi_gpu_transitions() -> None:
    """Focused SearchParallel × UseMultiGpuPUCV × MaxBatch/VLGather/Threads/UseVL
    matrix: advertised options and live path must stay consistent across
    transitions (the surface both review passes kept finding bugs on)."""
    def factories(_mb: int, _g: int) -> list[Any]:
        return [_make_evaluator, _make_evaluator]

    def rebuild_eval(mb: int, _cache: int = 0, *, n_walkers: int | None = None) -> Any:
        del n_walkers
        return _make_evaluator(max_batch=mb)

    def live_path(worker: SearchWorker) -> str:
        if worker._rpg_pool is not None:
            return "rpg"
        if worker._pucv_pool is not None:
            return "pucv_pool"
        if worker._walker_pool is not None:
            return "walker"
        if worker._pucv is not None:
            return "use_vl"
        return "classic"

    # A: gumbel → multi-pucv (forces sp=pucv) → MaxBatch stays pucv → off
    # restores UseVL (regression found in option-matrix recheck).
    worker = SearchWorker(
        _make_evaluator(max_batch=64), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=32, add_noise=False, temperature=0.0),
        chunk_sims=32, n_walkers=1,
    )
    worker.set_use_pucv(True, gather=32)
    engine = Engine(
        worker,
        rebuild_evaluator=rebuild_eval,
        rebuild_multi_gpu_pucv_factories=factories,
        options=EngineOptions(
            max_batch=64, use_vl=True, vl_gather=32, threads=1,
        ),
    )
    try:
        engine._set_search_parallel("gumbel")
        assert live_path(worker) == "rpg"
        engine._set_use_multi_gpu_pucv("true")
        assert live_path(worker) == "pucv_pool"
        assert engine._options.search_parallel == "pucv"
        engine._set_max_batch("128")
        assert live_path(worker) == "pucv_pool"
        assert engine._options.search_parallel == "pucv"
        engine._set_use_multi_gpu_pucv("false")
        assert engine._options.use_multi_gpu_pucv is False
        assert live_path(worker) == "use_vl", (
            f"UseVL must be restored after multi-pucv off, got {live_path(worker)}"
        )
    finally:
        engine.close()

    # B: multi-pucv → gumbel (flag preserved) → leave restores multi-pucv
    worker = _make_worker(chunk_sims=32)
    engine = Engine(
        worker,
        rebuild_multi_gpu_pucv_factories=factories,
        options=EngineOptions(max_batch=64),
    )
    try:
        engine._set_use_multi_gpu_pucv("true")
        assert live_path(worker) == "pucv_pool"
        engine._set_search_parallel("gumbel")
        assert live_path(worker) == "rpg"
        assert engine._options.use_multi_gpu_pucv is True
        engine._set_search_parallel("pucv")
        assert live_path(worker) == "pucv_pool"
    finally:
        engine.close()

    # C: gumbel → Threads/UseVL deferred → leave materialises both intents
    worker = _make_worker(chunk_sims=32)
    engine = Engine(
        worker,
        rebuild_multi_gpu_pucv_factories=factories,
        options=EngineOptions(vl_gather=32),
    )
    try:
        engine._set_search_parallel("gumbel")
        engine._set_threads("4")
        engine._set_use_vl("true")
        assert worker._walker_pool is None
        assert worker._pucv is None
        assert worker._n_walkers == 4
        assert worker._use_pucv is True
        engine._set_search_parallel("pucv")
        # Threads=4 wins over UseVL (pucv requires n_walkers==1).
        assert live_path(worker) == "walker"
        assert worker._n_walkers == 4
    finally:
        engine.close()

    # D: multi-pucv → Threads deferred (no dual walker) → off builds walker
    worker = _make_worker(chunk_sims=32)
    engine = Engine(
        worker,
        rebuild_multi_gpu_pucv_factories=factories,
        options=EngineOptions(),
    )
    try:
        engine._set_use_multi_gpu_pucv("true")
        engine._set_threads("4")
        assert worker._pucv_pool is not None
        assert worker._walker_pool is None, "must not dual-allocate walker beside multi-pucv"
        assert worker._n_walkers == 4
        engine._set_use_multi_gpu_pucv("false")
        assert live_path(worker) == "walker"
        assert worker._n_walkers == 4
    finally:
        engine.close()

    # E: gumbel → VLGather / EvalCacheEntries keep RPG and apply knobs
    worker = SearchWorker(
        _make_evaluator(max_batch=64), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=32, add_noise=False, temperature=0.0),
        chunk_sims=32, n_walkers=1, eval_cache_entries=0,
    )
    engine = Engine(
        worker,
        rebuild_evaluator=rebuild_eval,
        rebuild_multi_gpu_pucv_factories=factories,
        options=EngineOptions(max_batch=256, vl_gather=64, eval_cache_entries=0),
    )
    try:
        engine._set_search_parallel("gumbel")
        engine._set_vl_gather("128")
        assert live_path(worker) == "rpg"
        assert worker._rpg_pool is not None
        assert worker._rpg_pool._cfg.gather == 128
        engine._set_eval_cache_entries("64")
        assert live_path(worker) == "rpg"
        assert worker._rpg_pool is not None
        assert worker._rpg_pool._cfg.eval_cache_entries == 64
    finally:
        engine.close()


def test_terminal_candidate_respects_stop_event() -> None:
    """Terminal forced-sim backprop must poll stop between chunks."""
    from chess_anti_engine.uci.root_parallel_gumbel import (
        _CandidateArena,
        _WorkItem,
    )

    pool = _make_pool(1, gather=8)
    try:
        board = chess.Board()
        pol, wdl = _root_eval()
        tree = MCTSTree()
        tree.reserve(50_000, 500_000)
        rid = pool.prepare_root(
            tree=tree, board=board, pol_logits=pol, wdl_logits=wdl,
        )
        assert pool._state is not None
        # Expand one legal as a pre-marked terminal child so _run_item takes
        # the forced-sim path without an NN expand.
        legal = pool._state.search_legal
        assert legal.size > 0
        action = int(legal[0])
        cid = int(tree.find_child(rid, action))
        assert cid >= 0
        arena = _CandidateArena(
            action=action, cid=cid, terminal_value=1.0, expanded=True,
        )
        chunker = pool._chunkers[0]
        assert chunker is not None
        # Unstopped: full residual budget is applied (chunked, but complete).
        done_full = pool._run_item(
            0, pool._evals[0], chunker,
            _WorkItem(arena=arena, budget=100, phase_index=0),
            None,
        )
        assert done_full == 100

        # Already stopped: must not apply any residual.
        stop = threading.Event()
        stop.set()
        done_stopped = pool._run_item(
            0, pool._evals[0], chunker,
            _WorkItem(arena=arena, budget=50_000, phase_index=0),
            stop,
        )
        assert done_stopped == 0, (
            f"stopped terminal item must complete 0 sims, got {done_stopped}"
        )
    finally:
        pool.close()


# --- construction validation ------------------------------------------------------------


def test_pool_rejects_bad_construction() -> None:
    cfg = RootParallelGumbelConfig(n_groups=2)
    gcfg = GumbelConfig(input_extra_features="v1", )
    with pytest.raises(ValueError, match="need 2 evaluators"):
        RootParallelGumbelPool(
            cfg, gcfg, evaluators=[_DeterministicStubEvaluator()],
        )
    with pytest.raises(ValueError, match="exactly one"):
        RootParallelGumbelPool(cfg, gcfg)
    pool = _make_pool(1)
    try:
        with pytest.raises(RuntimeError, match="prepare_root"):
            pool.run(target_sims=8, stop_event=threading.Event())
    finally:
        pool.close()
