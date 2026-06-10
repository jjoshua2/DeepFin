"""Multi-GPU pucv pool: one worker thread per GPU sharing a single tree.

Each worker:
  - Owns a dedicated GPU evaluator (its own DirectGPUEvaluator on its own
    device, with its own torch.compile + cudagraph state).
  - Runs the 2-slot async pipeline (the same one ``PucvChunker`` uses)
    against the SHARED MCTSTree. Tree mutations are atomic + mutex-
    protected in the C extension (same guarantees walker_pool relies on).
  - Pulls work from a shared atomic budget (Semaphore, like walker_pool)
    so faster GPUs naturally take more leaves.

This is the pool-level multi-GPU primitive. The single-GPU PucvChunker
stays as a leaner code path for `Threads=1` UCI; this kicks in when the
user wires multiple devices.

Why a new class instead of extending WalkerPool: walker_pool's hot path
is ``walker_descend_puct`` (single leaf → ONE submit per gather) on a
shared evaluator. Multi-GPU pucv runs ``batch_descend_puct`` (G leaves →
ONE submit) on a per-worker evaluator with 2-slot pipelining. The shapes
are different enough that unifying them would muddy both.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard
from chess_anti_engine.inference_cache import PucvEvalCache, PucvEvalCacheStats
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.puct_vl import _PLANES, _alloc_buffers

_log = logging.getLogger(__name__)


@dataclass
class MultiGpuPucvConfig:
    n_gpus: int
    gather: int = 384
    c_puct: float = 1.4
    fpu_at_root: float = 0.0
    fpu_reduction: float = 0.2
    vloss_weight: int = 3
    vloss_mode: int = 0
    eval_cache_entries: int = 0
  # Input channel count (146 v1 / 175 v2_threats); sized from the model.
    input_planes: int = _PLANES
  # Transport dynamic board-relation matrices per leaf (model.use_dynamic_relations).
    compute_relations: bool = False


@dataclass(frozen=True)
class MultiGpuPucvWorkerStats:
    worker_index: int
    batches: int = 0
    leaves: int = 0
    terminal_leaves: int = 0
    max_batch: int = 0
    wait_seconds: float = 0.0
    wall_seconds: float = 0.0

    @property
    def avg_batch(self) -> float:
        return self.leaves / self.batches if self.batches else 0.0


@dataclass(frozen=True)
class MultiGpuPucvStats:
    target_sims: int = 0
    elapsed_seconds: float = 0.0
    workers: tuple[MultiGpuPucvWorkerStats, ...] = ()
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def batches(self) -> int:
        return sum(w.batches for w in self.workers)

    @property
    def leaves(self) -> int:
        return sum(w.leaves for w in self.workers)

    @property
    def terminal_leaves(self) -> int:
        return sum(w.terminal_leaves for w in self.workers)

    @property
    def avg_batch(self) -> float:
        return self.leaves / self.batches if self.batches else 0.0

    @property
    def cache_requests(self) -> int:
        return self.cache_hits + self.cache_misses

    @property
    def cache_hit_rate(self) -> float:
        requests = self.cache_requests
        if requests <= 0:
            return 0.0
        return self.cache_hits / requests


@dataclass
class _Job:
    tree: MCTSTree
    root_id: int
    root_cboard: _CBoard
    budget: threading.Semaphore
    stop_event: threading.Event


class MultiGpuPucvPool:
    """Run a single shared MCTS search across N GPU evaluators in parallel.

    ``evaluators[k]`` must be a DirectGPUEvaluator-shaped object exposing
    ``n_slots >= 2``, ``get_input_buffer(bsz, slot)``, and
    ``evaluate_inplace_async(bsz, slot)``. Each worker uses slots 0/1 of
    its own evaluator; slots are not shared across evaluators.
    """

    def __init__(
        self,
        cfg: MultiGpuPucvConfig,
        evaluator_factories: Sequence[Callable[[], Any]] | None = None,
        *,
        evaluators: Sequence[Any] | None = None,
    ) -> None:
        """Two construction modes:

        - **Factories** (recommended): each worker thread invokes its own
          factory at startup, so the resulting evaluator's compiled model
          + cudagraph state lives on the SAME thread that will replay it.
          This is required for ``torch.compile + cudagraphs`` because
          cudagraph_trees uses thread-local state — a compiled model
          built on thread A cannot be evaluated on thread B (TLS-key
          missing assertion in cudagraph_trees.py).

        - **Pre-built evaluators**: only safe when the evaluator does NOT
          use cudagraphs (e.g. CPU/eager paths in tests). Validated for
          the test path; production runs with compile=on must use factories.
        """
        if cfg.n_gpus < 1:
            raise ValueError(f"n_gpus must be >= 1, got {cfg.n_gpus}")
        if (evaluator_factories is None) == (evaluators is None):
            raise ValueError(
                "specify exactly one of evaluator_factories / evaluators",
            )
        if evaluator_factories is not None:
            if len(evaluator_factories) != cfg.n_gpus:
                raise ValueError(
                    f"need {cfg.n_gpus} factories, got {len(evaluator_factories)}",
                )
            self._factories: list[Callable[[], Any]] | None = list(evaluator_factories)
            self._evals: list[Any] = []  # populated per-thread
        else:
            assert evaluators is not None
            if len(evaluators) != cfg.n_gpus:
                raise ValueError(
                    f"need {cfg.n_gpus} evaluators, got {len(evaluators)}",
                )
            for k, ev in enumerate(evaluators):
                if not hasattr(ev, "evaluate_inplace_async"):
                    raise TypeError(
                        f"evaluator[{k}] missing evaluate_inplace_async",
                    )
                if getattr(ev, "n_slots", 1) < 2:
                    raise ValueError(
                        f"evaluator[{k}] needs n_slots >= 2 for pipelining",
                    )
            self._factories = None
            self._evals = list(evaluators)
        self._cfg = cfg

  # Same sync structure as WalkerPool: per-worker wake event + Barrier
  # for completion. Persistent threads, no per-chunk spawn cost. With
  # factories, an extra Barrier(n+1) gates run() until every worker has
  # built its evaluator (compile + cudagraph capture happen here, can
  # take 5-30s on first cold run depending on compile_mode).
        self._job_lock = threading.Lock()
        self._job: _Job | None = None
        self._wakes = [threading.Event() for _ in range(cfg.n_gpus)]
        self._done = threading.Barrier(cfg.n_gpus + 1)
        self._init_done = threading.Barrier(cfg.n_gpus + 1)
        self._shutdown = threading.Event()
        self._errors: list[BaseException] = []
        self._init_errors: list[BaseException] = []
        self._stats_lock = threading.Lock()
        self._run_started = 0.0
        self._cache = (
            PucvEvalCache(max_entries=int(cfg.eval_cache_entries))
            if cfg.eval_cache_entries > 0
            else None
        )
        self._cache_run_start = PucvEvalCacheStats(
            entries=0, hits=0, misses=0, stores=0, evictions=0,
        )
        self._last_stats = MultiGpuPucvStats(
            workers=tuple(
                MultiGpuPucvWorkerStats(worker_index=k)
                for k in range(cfg.n_gpus)
            ),
        )
        self._worker_stats = [
            MultiGpuPucvWorkerStats(worker_index=k)
            for k in range(cfg.n_gpus)
        ]
  # Pre-size the per-worker evaluator slot list when factories are used,
  # so worker threads can write their evaluator at index `k` safely.
        if self._factories is not None:
            self._evals = [None] * cfg.n_gpus
        self._threads = [
            threading.Thread(
                target=self._worker_loop, args=(k,),
                name=f"pucv-gpu-{k}", daemon=True,
            )
            for k in range(cfg.n_gpus)
        ]
        for th in self._threads:
            th.start()
        self._init_done.wait()
        if self._init_errors:
            self.close()
            raise self._init_errors[0]

    def __enter__(self) -> "MultiGpuPucvPool":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        for ev in self._wakes:
            ev.set()
        for th in self._threads:
            th.join(timeout=2.0)

    def last_stats(self) -> MultiGpuPucvStats:
        """Return stats from the most recent completed run.

        The pool mutates counters only from worker threads and snapshots them
        after the completion barrier, so callers get a stable immutable view.
        """
        with self._stats_lock:
            return self._last_stats

    def run(
        self,
        *,
        tree: MCTSTree,
        root_id: int,
        root_cboard: _CBoard,
        target_sims: int,
        stop_event: threading.Event,
    ) -> int:
        if target_sims <= 0:
            return 0
        if self._shutdown.is_set():
            raise RuntimeError("MultiGpuPucvPool is closed")

        with self._job_lock:
            self._errors = []
            self._run_started = time.perf_counter()
            self._worker_stats = [
                MultiGpuPucvWorkerStats(worker_index=k)
                for k in range(self._cfg.n_gpus)
            ]
            if self._cache is not None:
                self._cache_run_start = self._cache.stats()
            self._job = _Job(
                tree=tree, root_id=root_id, root_cboard=root_cboard,
                budget=threading.Semaphore(target_sims),
                stop_event=stop_event,
            )
        for ev in self._wakes:
            ev.set()
        self._done.wait()

        if self._errors:
            raise self._errors[0]
        elapsed = time.perf_counter() - self._run_started
        cache_hits = 0
        cache_misses = 0
        if self._cache is not None:
            cache_after = self._cache.stats()
            cache_hits = cache_after.hits - self._cache_run_start.hits
            cache_misses = cache_after.misses - self._cache_run_start.misses
        with self._stats_lock:
            self._last_stats = MultiGpuPucvStats(
                target_sims=target_sims,
                elapsed_seconds=elapsed,
                workers=tuple(self._worker_stats),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )
        _log.debug(
            "multi_gpu_pucv target=%d leaves=%d batches=%d avg_batch=%.1f "
            "wall=%.3fs worker_leaves=%s",
            target_sims,
            self._last_stats.leaves,
            self._last_stats.batches,
            self._last_stats.avg_batch,
            elapsed,
            [w.leaves for w in self._last_stats.workers],
        )
        return target_sims

    def _worker_loop(self, idx: int) -> None:
        cfg = self._cfg
        gather = max(1, int(cfg.gather))

        try:
            if self._factories is not None:
  # Build evaluator on THIS thread so torch.compile + cudagraph
  # state lives where it'll be replayed. Crucial: compile builds
  # tree_manager_containers in TLS on first forward; calling from
  # another thread asserts in cudagraph_trees.py:get_obj.
                evaluator = self._factories[idx]()
                if not hasattr(evaluator, "evaluate_inplace_async"):
                    raise TypeError(
                        f"factory[{idx}] returned an evaluator missing "
                        "evaluate_inplace_async",
                    )
                if getattr(evaluator, "n_slots", 1) < 2:
                    raise ValueError(
                        f"factory[{idx}] returned an evaluator with "
                        "n_slots < 2",
                    )
                self._evals[idx] = evaluator
            else:
                evaluator = self._evals[idx]
        except BaseException as exc:
            self._init_errors.append(exc)
            self._init_done.wait()
            return
        self._init_done.wait()

  # Per-worker buffer ping-pong (one set per slot). Encoding lands
  # directly in the evaluator's pinned input via get_input_buffer.
        bufs = [
            _alloc_buffers(
                gather, int(cfg.input_planes),
                with_relations=bool(cfg.compute_relations),
            ),
            _alloc_buffers(
                gather, int(cfg.input_planes),
                with_relations=bool(cfg.compute_relations),
            ),
        ]
        my_wake = self._wakes[idx]

        while True:
            my_wake.wait()
            my_wake.clear()
            if self._shutdown.is_set():
                return
            job = self._job
            if job is None:
                continue
            try:
                self._pipeline_until_done(idx, job, evaluator, bufs, gather)
            except Exception as exc:
                _log.exception("pucv-gpu-%d raised; requesting pool stop", idx)
                self._errors.append(exc)
                job.stop_event.set()
            self._done.wait()

    def _pipeline_until_done(
        self,
        idx: int,
        job: _Job,
        evaluator: Any,
        bufs: list[dict[str, np.ndarray]],
        gather: int,
    ) -> None:
        cfg = self._cfg
        c_puct = cfg.c_puct
        fpu_root = cfg.fpu_at_root
        fpu_red = cfg.fpu_reduction
        vloss = cfg.vloss_weight
        vloss_mode = cfg.vloss_mode
        tree = job.tree
        root_id = job.root_id
        root_cb = job.root_cboard
        budget = job.budget
        stop_event = job.stop_event

        pending: Any = None
        batches = 0
        leaves = 0
        terminal_leaves = 0
        max_batch = 0
        wait_seconds = 0.0
        wall_start = time.perf_counter()
        while not stop_event.is_set() or pending is not None:
  # Greedy budget grab: take up to ``gather`` tokens. Faster GPUs
  # naturally take more leaves over the search.
            n = 0
            if not stop_event.is_set():
                while n < gather and budget.acquire(blocking=False):
                    n += 1
            if n == 0 and pending is None:
                break

            if n > 0:
                next_local = 0 if pending is None else 1 - int(pending[1])
                next_slot = next_local  # slot 0/1 on this worker's evaluator
                inp = evaluator.get_input_buffer(n, slot=next_slot)
                inp_np = inp.numpy() if hasattr(inp, "numpy") else inp
                enc_view = np.asarray(inp_np).reshape(n, int(cfg.input_planes), 8, 8)
                b = bufs[next_local]
                rel = b["rel"] if cfg.compute_relations else None
                tree.batch_descend_puct(
                    root_id, root_cb, n, c_puct, fpu_root, fpu_red, vloss,
                    enc_view,
                    b["leaf_ids"], b["path_buf"], b["path_lens"],
                    b["legal_buf"], b["legal_lens"],
                    b["term_qs"], b["is_term"], vloss_mode,
                    b["cache_keys"] if self._cache is not None else None,
                    rel,
                )
                batches += 1
                leaves += n
                max_batch = max(max_batch, n)
                terminal_leaves += int(b["is_term"][:n].sum())
                next_handle = self._prepare_submit(
                    evaluator, next_slot, next_local, n, enc_view, b, rel,
                )
            else:
                next_handle = None

            if pending is not None:
                b_cur, p_n, pol, wdl, submit_t = self._finish_pending(
                    pending, bufs,
                )
                if submit_t > 0.0:
                    wait_seconds += time.perf_counter() - submit_t
                tree.batch_integrate_leaves(
                    p_n,
                    b_cur["path_buf"], b_cur["path_lens"],
                    b_cur["legal_buf"], b_cur["legal_lens"],
                    b_cur["is_term"], pol, wdl, vloss,
                )

            pending = next_handle

        self._worker_stats[idx] = MultiGpuPucvWorkerStats(
            worker_index=idx,
            batches=batches,
            leaves=leaves,
            terminal_leaves=terminal_leaves,
            max_batch=max_batch,
            wait_seconds=wait_seconds,
            wall_seconds=time.perf_counter() - wall_start,
        )

    def _prepare_submit(
        self,
        evaluator: Any,
        slot: int,
        buf_idx: int,
        n: int,
        enc_view: np.ndarray,
        b: dict[str, np.ndarray],
        rel: np.ndarray | None,
    ) -> tuple[Any, ...]:
        cache = self._cache
        if cache is None:
            if rel is None:
                handle = evaluator.evaluate_inplace_async(n, slot=slot)
            else:
                handle = evaluator.evaluate_inplace_async(
                    n, slot=slot, relations=rel[:n],
                )
            return ("gpu_full", buf_idx, n, handle, time.perf_counter())

        hit_values: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        miss_indices: list[int] = []
        for i in range(n):
            if int(b["is_term"][i]) != 0:
                continue
            hit = cache.get(b["cache_keys"][i], enc_view[i])
            if hit is None:
                miss_indices.append(i)
            else:
                hit_values[i] = hit

        if not miss_indices:
            return ("cache_only", buf_idx, n, hit_values, 0.0)

        miss_idx = np.asarray(miss_indices, dtype=np.int32)
        miss_n = int(miss_idx.size)
        b["enc_rows"][:n] = enc_view[:n]
        enc_view[:miss_n] = enc_view[miss_idx]
        if rel is None:
            handle = evaluator.evaluate_inplace_async(miss_n, slot=slot)
        else:
            # Compact relation rows alongside the encoding rows. Cache hits
            # stay valid: relations are a pure function of the position, so
            # position-keyed cached outputs already include the bias effect.
            rel[:miss_n] = rel[miss_idx]
            handle = evaluator.evaluate_inplace_async(
                miss_n, slot=slot, relations=rel[:miss_n],
            )
        return (
            "gpu_miss", buf_idx, n, miss_idx, hit_values,
            handle,
            time.perf_counter(),
        )

    def _finish_pending(
        self,
        pending: tuple[Any, ...],
        bufs: list[dict[str, np.ndarray]],
    ) -> tuple[dict[str, np.ndarray], int, np.ndarray, np.ndarray, float]:
        kind = pending[0]
        if kind == "gpu_full":
            _, buf_idx, n, handle, submit_t = pending
            pol_t, wdl_t, evt = handle
            if evt is not None:
                evt.synchronize()
            pol = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
            wdl = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
            return bufs[buf_idx], int(n), pol, wdl, float(submit_t)

        if kind == "cache_only":
            _, buf_idx, n, hit_values, submit_t = pending
            pol, wdl = self._materialize_cached_outputs(int(n), hit_values, None, None)
            return bufs[buf_idx], int(n), pol, wdl, float(submit_t)

        _, buf_idx, n, miss_idx, hit_values, handle, submit_t = pending
        pol_t, wdl_t, evt = handle
        if evt is not None:
            evt.synchronize()
        pol_m = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
        wdl_m = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
        b = bufs[buf_idx]
        cache = self._cache
        assert cache is not None
        for j, original_i in enumerate(miss_idx):
            i = int(original_i)
            cache.put(b["cache_keys"][i], b["enc_rows"][i], pol_m[j], wdl_m[j])
        pol, wdl = self._materialize_cached_outputs(
            int(n), hit_values, miss_idx, (pol_m, wdl_m),
        )
        return b, int(n), pol, wdl, float(submit_t)

    def _materialize_cached_outputs(
        self,
        n: int,
        hit_values: dict[int, tuple[np.ndarray, np.ndarray]],
        miss_idx: np.ndarray | None,
        miss_outputs: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        first_value = next(iter(hit_values.values()), None)
        if first_value is None and miss_outputs is not None:
            first_value = (miss_outputs[0][0], miss_outputs[1][0])
        if first_value is None:
            return (
                np.zeros((n, 4672), dtype=np.float32),
                np.zeros((n, 3), dtype=np.float32),
            )
        pol = np.zeros((n, *first_value[0].shape), dtype=first_value[0].dtype)
        wdl = np.zeros((n, *first_value[1].shape), dtype=first_value[1].dtype)
        for i, (pol_row, wdl_row) in hit_values.items():
            pol[i] = pol_row
            wdl[i] = wdl_row
        if miss_idx is not None and miss_outputs is not None:
            pol_m, wdl_m = miss_outputs
            for j, original_i in enumerate(miss_idx):
                i = int(original_i)
                pol[i] = pol_m[j]
                wdl[i] = wdl_m[j]
        return pol, wdl
