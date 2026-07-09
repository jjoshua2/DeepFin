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
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
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
    gcfg = GumbelConfig(
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
    finally:
        pool.close()


# --- SearchWorker integration -------------------------------------------------------


def _make_worker(chunk_sims: int = 64) -> SearchWorker:
    return SearchWorker(
        _make_evaluator(max_batch=64), device="cpu",
        gumbel_cfg=GumbelConfig(
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


# --- construction validation ------------------------------------------------------------


def test_pool_rejects_bad_construction() -> None:
    cfg = RootParallelGumbelConfig(n_groups=2)
    gcfg = GumbelConfig()
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
