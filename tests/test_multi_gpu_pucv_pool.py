"""MultiGpuPucvPool tests.

Covers:
  - basic pool spawn + run + close
  - shared-tree visit accumulation across N=1 and N=2 worker pools
    (using two evaluator instances on CPU as a stand-in for two GPUs)
  - SearchWorker.install_multi_gpu_pucv routes through the pool
  - clear_multi_gpu_pucv reverts to single-evaluator search
  - vloss is fully unwound after run(), even with concurrent workers
"""
from __future__ import annotations

import threading
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.multi_gpu_pucv_pool import (
    MultiGpuPucvConfig,
    MultiGpuPucvPool,
)
from chess_anti_engine.uci.search import SearchWorker, _pucv_stats_string
from chess_anti_engine.uci.time_manager import Deadline


def _make_evaluator(max_batch: int = 64) -> Any:
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    inner = DirectGPUEvaluator(
        model, device="cpu", max_batch=max_batch, use_amp=False, n_slots=2,
    )
    return ThreadSafeGPUDispatcher(inner)


def _seed_tree() -> tuple[MCTSTree, int, CBoard]:
    tree = MCTSTree()
    tree.reserve(1024, 8192)
    cb = CBoard.from_board(chess.Board())
    rid = tree.add_root(0, 0.0)
    legal = cb.legal_move_indices().astype(np.int32)
    priors = np.full(legal.size, 1.0 / legal.size, dtype=np.float64)
    tree.expand(rid, legal, priors)
    return tree, rid, cb


class _CountingInplaceEvaluator:
    n_slots = 2

    def __init__(self, max_batch: int = 8) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []
        self._lock = threading.Lock()
        self._bufs = [
            np.zeros((max_batch, 146, 8, 8), dtype=np.float32),
            np.zeros((max_batch, 146, 8, 8), dtype=np.float32),
        ]

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        return self._bufs[slot][:bsz]

    def evaluate_inplace_async(
        self,
        bsz: int,
        *,
        slot: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        del slot
        with self._lock:
            self.calls += 1
            self.batch_sizes.append(int(bsz))
        return (
            np.zeros((bsz, 4672), dtype=np.float32),
            np.zeros((bsz, 3), dtype=np.float32),
            None,
        )


def test_pool_n1_accumulates_visits() -> None:
    """N=1 pool == single PucvChunker functionally; sims target reached."""
    ev = _make_evaluator()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=1, gather=8, vloss_weight=3),
        evaluators=[ev],
    )
    try:
        tree, rid, cb = _seed_tree()
        target = 32
        pool.run(tree=tree, root_id=rid, root_cboard=cb,
                 target_sims=target, stop_event=threading.Event())
        _, visits = tree.get_children_visits(rid)
        assert int(visits.sum()) >= target * 3 // 4
    finally:
        pool.close()


def test_pool_n2_accumulates_visits_no_vloss_leak() -> None:
    """N=2 workers on shared tree split the budget; total visits ≈ target.
    All vloss must be removed at end (atomicity check across workers)."""
    ev0 = _make_evaluator()
    ev1 = _make_evaluator()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, vloss_weight=3),
        evaluators=[ev0, ev1],
    )
    try:
        tree, rid, cb = _seed_tree()
        target = 64
        pool.run(tree=tree, root_id=rid, root_cboard=cb,
                 target_sims=target, stop_event=threading.Event())
        _, visits = tree.get_children_visits(rid)
        total = int(visits.sum())
        assert target * 3 // 4 <= total <= target
        stats = pool.last_stats()
        assert stats.target_sims == target
        assert stats.leaves == target
        assert stats.batches >= 1
        assert stats.avg_batch > 0.0
        assert len(stats.workers) == 2
        assert sum(w.leaves for w in stats.workers) == target
        assert max(w.max_batch for w in stats.workers) <= 8

        for nid in range(tree.node_count()):
            assert tree.get_virtual_loss(nid) == 0, f"vl leaked on {nid}"
    finally:
        pool.close()


def test_pool_run_returns_completed_not_target_on_early_stop() -> None:
    """run() must report the sims actually completed: a pre-set stop_event
    means workers acquire no budget tokens, so the return is 0, not the
    requested target (which would inflate the caller's nps/node math)."""
    ev = _make_evaluator()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=1, gather=8, vloss_weight=3),
        evaluators=[ev],
    )
    try:
        tree, rid, cb = _seed_tree()
        stopped = threading.Event()
        stopped.set()
        completed = pool.run(tree=tree, root_id=rid, root_cboard=cb,
                             target_sims=64, stop_event=stopped)
        assert completed == 0
        _, visits = tree.get_children_visits(rid)
        assert int(visits.sum()) == 0

  # Uninterrupted run consumes the whole budget and reports it.
        completed = pool.run(tree=tree, root_id=rid, root_cboard=cb,
                             target_sims=64, stop_event=threading.Event())
        assert completed == 64
    finally:
        pool.close()


def test_pool_terminal_only_batches_skip_evaluator() -> None:
    """With the eval cache off, batches that gathered ONLY terminal leaves
    must not submit (zeroed) inputs to the evaluator — descend already
    backpropped the terminal values inline. A stalemate root makes every
    descent terminal, so the evaluator must never be called."""
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    ev = _CountingInplaceEvaluator()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=1, gather=4),
        evaluators=[ev],
    )
    try:
        tree = MCTSTree()
        tree.reserve(1024, 8192)
        cb = CBoard.from_board(board)
        rid = tree.add_root(0, 0.0)
        target = 16
        completed = pool.run(tree=tree, root_id=rid, root_cboard=cb,
                             target_sims=target, stop_event=threading.Event())
        assert completed == target
        assert ev.calls == 0
        stats = pool.last_stats()
        assert stats.terminal_leaves == target
    finally:
        pool.close()


def test_searchworker_pool_nodes_reflect_completed_sims() -> None:
    """SearchWorker must account pool chunks by their completed count: with a
    pre-set stop_event the pool completes nothing, so result.nodes == 0
    (previously it reported the full requested chunk)."""
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    p0 = _make_evaluator(max_batch=64)
    p1 = _make_evaluator(max_batch=64)
    worker.install_multi_gpu_pucv([p0, p1], gather=8, as_factories=False)
    stopped = threading.Event()
    stopped.set()
    result = worker.run(chess.Board(), stop_event=stopped,
                        deadline=Deadline(2_000), max_nodes=64)
    assert result.nodes == 0
    assert len(result.bestmove_uci) >= 4  # bestmove still emitted (root priors)
    worker.close()


def test_pool_zero_target_is_noop() -> None:
    ev = _make_evaluator()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=1, gather=8),
        evaluators=[ev],
    )
    try:
        tree, rid, cb = _seed_tree()
        pre = tree.node_count()
        pool.run(tree=tree, root_id=rid, root_cboard=cb,
                 target_sims=0, stop_event=threading.Event())
        assert tree.node_count() == pre
    finally:
        pool.close()


def test_pool_eval_cache_reuses_identical_fresh_tree_leaf() -> None:
    ev = _CountingInplaceEvaluator()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=1, gather=1, eval_cache_entries=8),
        evaluators=[ev],
    )
    try:
        tree_a, rid_a, cb_a = _seed_tree()
        pool.run(tree=tree_a, root_id=rid_a, root_cboard=cb_a,
                 target_sims=1, stop_event=threading.Event())

        tree_b, rid_b, cb_b = _seed_tree()
        pool.run(tree=tree_b, root_id=rid_b, root_cboard=cb_b,
                 target_sims=1, stop_event=threading.Event())

        assert ev.calls == 1
        assert ev.batch_sizes == [1]
        stats = pool.last_stats()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 0
        for nid in range(tree_b.node_count()):
            assert tree_b.get_virtual_loss(nid) == 0, f"vl leaked on {nid}"
    finally:
        pool.close()


def test_pool_rejects_single_slot_evaluator() -> None:
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    bad = DirectGPUEvaluator(
        model, device="cpu", max_batch=8, use_amp=False, n_slots=1,
    )
    with pytest.raises(ValueError, match="n_slots"):
        MultiGpuPucvPool(
            MultiGpuPucvConfig(n_gpus=1, gather=8),
            evaluators=[ThreadSafeGPUDispatcher(bad)],
        )


def test_pool_rejects_evaluator_count_mismatch() -> None:
    ev = _make_evaluator()
    with pytest.raises(ValueError, match="need 2 evaluators"):
        MultiGpuPucvPool(
            MultiGpuPucvConfig(n_gpus=2, gather=8),
            evaluators=[ev],
        )


def test_searchworker_install_multi_gpu_pucv_produces_bestmove() -> None:
    """Smoke: SearchWorker.install_multi_gpu_pucv routes search through the
    pool. Bestmove is non-empty and visit count meaningful."""
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
  # Two pool evaluators (independent of `primary`, used only for root-eval).
    p0 = _make_evaluator(max_batch=64)
    p1 = _make_evaluator(max_batch=64)
    worker.install_multi_gpu_pucv([p0, p1], gather=8, as_factories=False)
    deadline = Deadline(2_000)
    result = worker.run(chess.Board(), stop_event=threading.Event(),
                        deadline=deadline, max_nodes=64)
    assert len(result.bestmove_uci) >= 4
    assert result.nodes >= 32
    stats = worker.last_multi_gpu_pucv_stats()
    assert stats is not None
    assert stats.batches >= 1
    assert stats.leaves >= 32
    stats_line = _pucv_stats_string(stats, pending_mode=1)
    assert stats_line is not None
    assert "pending=virtual-mean" in stats_line
    worker.close()


def test_shared_tree_headroom_pregrows_arena_past_fixed_reserve() -> None:
    """_ensure_shared_tree_headroom must pre-grow the arena for the next chunk's
    worst-case node growth (chunk * branching), well past the old fixed
    reserve(50_000, ...). This is what keeps a concurrent chunk from triggering
    a tree_grow_* realloc mid-descent (use-after-free per _mcts_tree.c)."""
    ev = _make_evaluator()
    worker = SearchWorker(
        ev, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=512, n_walkers=1,
    )
    try:
        tree = MCTSTree()
        tree.reserve(64, 256)  # tiny, like a freshly created tree
        tree.add_root(0, 0.0)
        worker._tree = tree
        before = tree.memory_bytes()
        worker._ensure_shared_tree_headroom(512)
        after = tree.memory_bytes()
        assert after > before
  # 512 sims * 256 max branching = 131072 nodes of headroom; per-node node +
  # child arrays are ~50 bytes, so capacity now far exceeds the old 50k reserve.
        assert after >= 512 * 256 * 50
    finally:
        worker.close()


def test_shared_tree_headroom_respects_hash_cap() -> None:
    """The pre-grow must not push memory_bytes() past the Hash cap: it returns
    the sims that fit and 0 (no growth) once the tree is at the cap. A fresh tree
    per case keeps node_count ~ node_cap (as it is in a live search, where each
    chunk fills the prior reservation before the next cap check)."""
    ev = _make_evaluator()
    worker = SearchWorker(
        ev, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=512, n_walkers=1,
    )

    def _fresh() -> MCTSTree:
        tree, rid, _ = _seed_tree()
        worker._tree = tree
        worker._root_id = rid
        return tree

    try:
  # Cap disabled -> no bounding, full budget granted (and arena grows).
        _fresh()
        worker.set_max_tree_mb(0)
        assert worker._ensure_shared_tree_headroom(512) == 512
  # Generous cap -> still the full budget.
        _fresh()
        worker.set_max_tree_mb(100_000)
        assert worker._ensure_shared_tree_headroom(512) == 512
  # Cap already below current usage -> 0 sims and no further growth.
  # Cap below current usage: only already-reserved free slots are usable, so
  # the chunk shrinks to them but must NOT grow memory past the cap (and must
  # not starve to 0 — that would play an unvisited root).
        tree = _fresh()
        worker.set_max_tree_mb(1)
        before = tree.memory_bytes()
        sims = worker._ensure_shared_tree_headroom(512)
        assert 0 < sims <= 512
        assert tree.memory_bytes() == before
    finally:
        worker.close()


def test_shared_tree_headroom_uses_free_slots_at_tiny_hash() -> None:
    """A Hash cap below the already-reserved arena must not starve the chunk to
    0 (which would play an unvisited root): the free reserved slots are usable
    without growing memory, so the search still runs."""
    ev = _make_evaluator()
    worker = SearchWorker(
        ev, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=512, n_walkers=1,
    )
    try:
  # Production-sized initial reserve (lots of free slots), then a Hash cap
  # below current usage so `growable` is 0 and only free slots remain.
        tree = MCTSTree()
        tree.reserve(50_000, 500_000)
        tree.add_root(0, 0.0)
        worker._tree = tree
        worker._root_id = int(tree.node_count()) - 1
        worker.set_max_tree_mb(1)
        sims = worker._ensure_shared_tree_headroom(512)
        assert sims > 0
    finally:
        worker.close()


def test_chunk_budget_scales_by_gather_and_devices() -> None:
    """When VLGather exceeds --chunk-sims, the per-device share must be at least
    the gather size or workers starve again."""
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    p0 = _make_evaluator(max_batch=64)
    p1 = _make_evaluator(max_batch=64)
  # gather 128 > chunk_sims 64 -> per-device budget must use the gather.
    worker.install_multi_gpu_pucv([p0, p1], gather=128, as_factories=False)
    assert worker._chunk_budget(None) == 128 * 2
    worker.close()


def test_current_best_root_action_tracks_emitted_move() -> None:
    """Soft-stop stability must judge the move final selection will emit: the
    Gumbel survivor when set+legal, else the visit leader (walker/pucv paths)."""
    ev = _make_evaluator()
    worker = SearchWorker(
        ev, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    try:
        tree, rid, cb = _seed_tree()
        worker._tree = tree
        worker._root_id = rid
        legal = cb.legal_move_indices().astype(np.int32)
        survivor = int(legal[1])
  # Gumbel survivor set + legal -> returned even though visits are all zero.
        worker._last_gumbel_action_idx = survivor
        assert worker._current_best_root_action(None) == survivor
  # searchmoves excludes the survivor -> fall back to a visit-leader in-set.
        assert worker._current_best_root_action({int(legal[0])}) == int(legal[0])
  # No survivor (walker / pucv paths) -> a legal visit-leader.
        worker._last_gumbel_action_idx = None
        assert worker._current_best_root_action(None) in set(legal.tolist())
    finally:
        worker.close()


def test_searchworker_multi_gpu_pucv_multi_chunk_search() -> None:
    """A pool search spanning several chunks exercises the per-chunk headroom
    growth in run()'s loop and must complete with a legal bestmove. The budget
    is scaled by device count, so node growth crosses the initial reserve."""
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    p0 = _make_evaluator(max_batch=64)
    p1 = _make_evaluator(max_batch=64)
    worker.install_multi_gpu_pucv([p0, p1], gather=8, as_factories=False)
  # 2 devices -> chunk budget 128; max_nodes 400 -> several chunks.
    result = worker.run(chess.Board(), stop_event=threading.Event(),
                        deadline=Deadline(5_000), max_nodes=400)
    assert len(result.bestmove_uci) >= 4
    assert result.nodes >= 256
    worker.close()


def test_search_emit_info_reports_hashfull_and_seldepth() -> None:
    """Info lines must carry hashfull (tree fill vs Hash cap) and a climbing
    seldepth — both expected by GUIs/TCEC spectators and previously never set."""
    ev = _make_evaluator()
    worker = SearchWorker(
        ev, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    try:
        worker.set_max_tree_mb(16)
        tree, rid, _ = _seed_tree()
        worker._tree = tree
        worker._root_id = rid
        captured: list[dict[str, Any]] = []
        worker._emit_pv_info(
            lambda **kw: captured.append(kw),
            chess.Board(), 0.0, 64, 100, None,
        )
        assert captured
        kw = captured[0]
        assert kw["seldepth"] is not None
        assert kw["seldepth"] >= 1
        assert kw["hashfull"] is not None
        assert 0 <= kw["hashfull"] <= 1000
  # Disabling the Hash cap drops hashfull rather than dividing by zero.
        worker.set_max_tree_mb(0)
        assert worker._hashfull_permille() is None
    finally:
        worker.close()


def test_root_visit_lead_returns_emitted_move_margin() -> None:
    """_root_visit_lead reports the emitted move and a real visit margin."""
    ev = _make_evaluator()
    worker = SearchWorker(
        ev, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=32, add_noise=False),
        chunk_sims=32, n_walkers=1,
    )
    try:
        worker.run(chess.Board(), stop_event=threading.Event(),
                   deadline=Deadline(5_000), max_nodes=256)
        best, lead = worker._root_visit_lead(None)
        assert best >= 0
        assert isinstance(lead, int)
    finally:
        worker.close()


def test_abort_ready_branches() -> None:
    """_abort_ready, with the visit margin controlled directly:
      - no optimum                -> never aborts;
      - past optimum + settled    -> banks;
      - past optimum + unsettled  -> extends (keeps searching to the deadline);
      - past optimum + no move    -> extends;
      - before optimum + settled  -> waits under the provable factor, banks once
        the lead exceeds the remaining budget (and aggressive factor banks
        sooner)."""
    import time as _time

    ev = _make_evaluator()
    worker = SearchWorker(
        ev, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=32, add_noise=False),
        chunk_sims=32, n_walkers=1,
    )
    try:
        past = Deadline(deadline_ms=60_000, now=_time.monotonic() - 5.0)  # elapsed ~5s
        worker._root_visit_lead = lambda allowed_root_indices: (5, 100)
        assert worker._abort_ready(None, past, 256, None, 1.0) is False
  # Complexity gate: a move that leads on visits but has not yet been stable for
  # _ABORT_MIN_STABLE_CHUNKS does NOT bank even past the optimum — it reads as a
  # still-moving (hard) position and keeps searching. It banks once it stabilises.
        worker._abort_last_best = -1
        worker._abort_stable_chunks = 0
        assert worker._abort_ready(1, past, 256, None, 1.0) is False
        assert worker._abort_ready(1, past, 256, None, 1.0) is False
        assert worker._abort_ready(1, past, 256, None, 1.0) is True
  # Unsettled / no move past the optimum -> extend toward the hard deadline.
        worker._root_visit_lead = lambda allowed_root_indices: (5, 0)
        assert worker._abort_ready(1, past, 256, None, 1.0) is False
        worker._root_visit_lead = lambda allowed_root_indices: (-1, 0)
        assert worker._abort_ready(1, past, 256, None, 1.0) is False
  # Before the optimum: elapsed ~500ms of a 1000ms optimum, nps ~2 -> ~1000
  # remaining sims. lead 2000 clears the provable factor but not factor 3.
        mid = Deadline(deadline_ms=10_000, now=_time.monotonic() - 0.5)
        worker._root_visit_lead = lambda allowed_root_indices: (5, 2000)
        assert worker._abort_ready(1000, mid, 1000, None, 1.0) is True
        assert worker._abort_ready(1000, mid, 1000, None, 3.0) is False
    finally:
        worker.close()


def test_search_abort_terminates_with_legal_move() -> None:
    """A tiny optimum + bounded hard deadline must return a legal move promptly:
    banking when the move is settled, extending to the deadline when it is not."""
    import time as _time

    ev = _make_evaluator()
    worker = SearchWorker(
        ev, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    try:
        t0 = _time.monotonic()
        result = worker.run(
            chess.Board(), stop_event=threading.Event(),
            deadline=Deadline(3_000), max_nodes=None, optimum_ms=1,
        )
        elapsed = _time.monotonic() - t0
        assert len(result.bestmove_uci) >= 4
  # Bounded by the 3s hard deadline (plus a little warmup slack).
        assert elapsed < 8.0
    finally:
        worker.close()


def test_time_capped_chunk_shrinks_to_remaining_time() -> None:
    """A scaled chunk is capped to the sims that fit in the remaining deadline,
    floored at the base chunk; open-ended / first-chunk searches are uncapped."""
    import time as _time

    ev = _make_evaluator()
    worker = SearchWorker(
        ev, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    try:
  # elapsed ~1000ms, nps = 1000/1000 = 1; ~500ms remaining -> cap ~500.
        d = Deadline(deadline_ms=1500, now=_time.monotonic() - 1.0)
        capped = worker._time_capped_chunk(4096, d, total_nodes=1000)
        assert 64 <= capped < 4096
  # Almost no time left -> floor at the base chunk (its overrun is negligible).
        d2 = Deadline(deadline_ms=1001, now=_time.monotonic() - 1.0)
        assert worker._time_capped_chunk(4096, d2, total_nodes=1000) == 64
  # Open-ended deadline (no clock) is uncapped even with an nps estimate.
        assert worker._time_capped_chunk(4096, Deadline(None), 1000) == 4096
  # First timed chunk (deadline set, no nps yet): bound to the base chunk so a
  # scaled multi-GPU chunk can't run unbounded past the deadline (move-1 forfeit).
        assert worker._time_capped_chunk(4096, d, total_nodes=0) == 64
    finally:
        worker.close()


def test_searchworker_clear_multi_gpu_pucv_reverts() -> None:
    """install_multi_gpu_pucv → clear_multi_gpu_pucv must drop the pool and
    leave subsequent searches running through the gumbel/walker path."""
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    p0 = _make_evaluator(max_batch=64)
    worker.install_multi_gpu_pucv([p0], gather=8, as_factories=False)
    assert worker._pucv_pool is not None
    worker.clear_multi_gpu_pucv()
    assert worker._pucv_pool is None
    deadline = Deadline(2_000)
    result = worker.run(chess.Board(), stop_event=threading.Event(),
                        deadline=deadline, max_nodes=64)
    assert result.bestmove_uci
    worker.close()


def test_clear_multi_gpu_pucv_restores_walker_pool() -> None:
    """UCI opt-out after multi-GPU install must restore Threads>1 walkers.

    install_multi_gpu_pucv closes the walker pool. Without a rebuild on
    clear, the default Threads=2 path silently falls back to single-thread
    Gumbel (Codex review on PR #138).
    """
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=2,
    )
    try:
        assert worker._walker_pool is not None
        p0 = _make_evaluator(max_batch=64)
        p1 = _make_evaluator(max_batch=64)
        worker.install_multi_gpu_pucv([p0, p1], gather=8, as_factories=False)
        assert worker._pucv_pool is not None
        assert worker._walker_pool is None
        worker.clear_multi_gpu_pucv()
        assert worker._pucv_pool is None
        assert worker._walker_pool is not None
        deadline = Deadline(2_000)
        result = worker.run(
            chess.Board(), stop_event=threading.Event(),
            deadline=deadline, max_nodes=64,
        )
        assert result.bestmove_uci
    finally:
        worker.close()


def test_clear_multi_gpu_pucv_restores_use_pucv() -> None:
    """clear must rebuild single-thread PUCV when UseVL was on under the pool."""
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    try:
        worker.set_use_pucv(True, gather=8)
        assert worker._pucv is not None
        p0 = _make_evaluator(max_batch=64)
        p1 = _make_evaluator(max_batch=64)
        worker.install_multi_gpu_pucv([p0, p1], gather=8, as_factories=False)
        assert worker._pucv is None
        assert worker._use_pucv is True
        worker.clear_multi_gpu_pucv()
        assert worker._pucv_pool is None
        assert worker._pucv is not None
    finally:
        worker.close()


def test_clear_multi_gpu_pucv_restore_fallback_false_skips_walker() -> None:
    """install/set_evaluator pass restore_fallback=False to avoid thrash."""
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=2,
    )
    try:
        p0 = _make_evaluator(max_batch=64)
        p1 = _make_evaluator(max_batch=64)
        worker.install_multi_gpu_pucv([p0, p1], gather=8, as_factories=False)
        assert worker._walker_pool is None
        worker.clear_multi_gpu_pucv(restore_fallback=False)
        assert worker._pucv_pool is None
        assert worker._walker_pool is None  # not rebuilt
    finally:
        worker.close()


def test_set_num_threads_under_pucv_pool_does_not_spawn_walkers() -> None:
    primary = _make_evaluator(max_batch=64)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    try:
        p0 = _make_evaluator(max_batch=64)
        p1 = _make_evaluator(max_batch=64)
        worker.install_multi_gpu_pucv([p0, p1], gather=8, as_factories=False)
        assert worker._walker_pool is None
        worker.set_num_threads(4)
        assert worker._n_walkers == 4
        assert worker._walker_pool is None
        worker.clear_multi_gpu_pucv()
        assert worker._walker_pool is not None  # restore uses stored n=4
    finally:
        worker.close()
