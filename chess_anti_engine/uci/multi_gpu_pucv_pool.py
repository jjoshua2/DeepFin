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
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree

_log = logging.getLogger(__name__)

_MAX_PATH = 512
_MAX_LEGAL = 256
_PLANES = 146


@dataclass
class MultiGpuPucvConfig:
    n_gpus: int
    gather: int = 384
    c_puct: float = 1.4
    fpu_at_root: float = 0.0
    fpu_reduction: float = 0.2
    vloss_weight: int = 3


@dataclass
class _Job:
    tree: MCTSTree
    root_id: int
    root_cboard: _CBoard
    budget: threading.Semaphore
    stop_event: threading.Event


def _alloc_buffers(gather: int) -> dict[str, np.ndarray]:
    return {
        "leaf_ids": np.empty(gather, dtype=np.int32),
        "path_buf": np.empty(gather * _MAX_PATH, dtype=np.int32),
        "path_lens": np.empty(gather, dtype=np.int32),
        "legal_buf": np.empty(gather * _MAX_LEGAL, dtype=np.int32),
        "legal_lens": np.empty(gather, dtype=np.int32),
        "term_qs": np.empty(gather, dtype=np.float64),
        "is_term": np.empty(gather, dtype=np.int8),
    }


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
        bufs = [_alloc_buffers(gather), _alloc_buffers(gather)]
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
                self._pipeline_until_done(job, evaluator, bufs, gather)
            except Exception as exc:
                _log.exception("pucv-gpu-%d raised; requesting pool stop", idx)
                self._errors.append(exc)
                job.stop_event.set()
            self._done.wait()

    def _pipeline_until_done(
        self,
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
        tree = job.tree
        root_id = job.root_id
        root_cb = job.root_cboard
        budget = job.budget
        stop_event = job.stop_event

        pending: Any = None
        while not stop_event.is_set() or pending is not None:
  # Greedy budget grab: take up to ``gather`` tokens. Faster GPUs
  # naturally take more leaves over the search.
            n = 0
            if not stop_event.is_set():
                while n < gather and budget.acquire(blocking=False):
                    n += 1
            if n == 0 and pending is None:
                return

            if n > 0:
                next_local = 0 if pending is None else 1 - pending[0]
                next_slot = next_local  # slot 0/1 on this worker's evaluator
                inp = evaluator.get_input_buffer(n, slot=next_slot)
                inp_np = inp.numpy() if hasattr(inp, "numpy") else inp
                enc_view = np.asarray(inp_np).reshape(n, _PLANES, 8, 8)
                b = bufs[next_local]
                tree.batch_descend_puct(
                    root_id, root_cb, n, c_puct, fpu_root, fpu_red, vloss,
                    enc_view,
                    b["leaf_ids"], b["path_buf"], b["path_lens"],
                    b["legal_buf"], b["legal_lens"],
                    b["term_qs"], b["is_term"],
                )
                pol_t, wdl_t, evt = evaluator.evaluate_inplace_async(
                    n, slot=next_slot,
                )
                next_handle = (next_local, n, pol_t, wdl_t, evt)
            else:
                next_handle = None

            if pending is not None:
                p_local, p_n, pol_t, wdl_t, evt = pending
                if evt is not None:
                    evt.synchronize()
                pol = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
                wdl = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
                b_cur = bufs[p_local]
                tree.batch_integrate_leaves(
                    p_n,
                    b_cur["path_buf"], b_cur["path_lens"],
                    b_cur["legal_buf"], b_cur["legal_lens"],
                    b_cur["is_term"], pol, wdl, vloss,
                )

            pending = next_handle
