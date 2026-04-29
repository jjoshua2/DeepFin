"""Walker-thread pool: concurrent PUCT descent with virtual loss.

Each walker runs a tight loop:

    leaf_id, path, legal, term_q = tree.walker_descend_puct(...)
    if term_q is not None:
        tree.backprop(path, term_q)  # terminal leaf
    else:
        pol, wdl = evaluator.evaluate_encoded(enc)  # releases the GIL
        tree.walker_integrate_leaf(path, legal, pol[0], wdl[0], vloss)

Virtual loss in ``walker_descend_puct`` keeps concurrent walkers off each
other's in-flight paths. Evaluator must be thread-safe. Tree must have
had ``reserve()`` called so concurrent descents cannot trigger a realloc.

Threads are persistent. ``__init__`` spawns N daemon walker threads that
park on a per-walker Event. ``run()`` publishes the new (tree, root_id,
root_cboard, target_sims) job, sets each Event to wake the walkers, and
waits on a Barrier for completion. Profile (2026-04-28) of the previous
spawn-and-join-per-chunk implementation showed 97% of pool wall-clock in
``Thread.join`` for chunk_sims=512 — that's now zero.

Trade-off vs. the Gumbel+halving path: drops sequential halving, so the
policy distribution is visit-count from PUCT rather than Gumbel's top-m
survivors. For single-game UCI that's fine; for training-style runs,
keep walkers=1 and use the classic path.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree

_log = logging.getLogger(__name__)


class _Evaluator(Protocol):
    def evaluate_encoded(
        self, x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class WalkerPoolConfig:
    n_walkers: int
    c_puct: float
    fpu_at_root: float
    fpu_reduction: float
    vloss_weight: int = 3
  # Each walker does up to `gather` descents before submitting one NN
  # batch — virtual loss diversifies within a gather the same way it
  # diversifies across walkers. Lc0's canonical default is 8; we keep
  # 1 to preserve the classic one-leaf-per-call shape.
    gather: int = 1


@dataclass
class _Job:
    """Per-chunk parameters published to all walkers under ``_job_lock``."""
    tree: MCTSTree
    root_id: int
    root_cboard: _CBoard
    budget: threading.Semaphore
    stop_event: threading.Event


class WalkerPool:
    """Run ``target_sims`` simulations on a shared tree with N persistent threads.

    Threads spin up at ``__init__`` time and park between ``run()`` calls.
    Caller must pre-expand the root before each ``run()``; otherwise all
    walkers race on the same unexpanded leaf and waste N-1 NN evals.

    ``close()`` shuts the threads down cleanly. The class is also a context
    manager. Threads are daemons so a forgotten ``close()`` doesn't hang
    interpreter exit.
    """

    def __init__(self, cfg: WalkerPoolConfig, evaluator: _Evaluator) -> None:
        self._cfg = cfg
        self._evaluator = evaluator

  # Synchronization:
  # - _job_lock + _job: single-producer (caller of run()) → N consumers.
  #   Caller mutates _job under _job_lock then sets each walker's _wake.
  # - _wakes[i]: per-walker Event. Cleared by walker after it sees the
  #   job; set by caller. Per-walker (rather than one shared) so each
  #   walker resets its own without racing the next-chunk publish.
  # - _done: Barrier(n_walkers + 1). All walkers + the caller must call
  #   wait() to release; the caller's wait blocks until every walker has
  #   drained its budget. Cheaper than N joins.
  # - _shutdown: separate flag + wake to break walkers out of park.
        self._job_lock = threading.Lock()
        self._job: _Job | None = None
        self._wakes = [threading.Event() for _ in range(cfg.n_walkers)]
        self._done = threading.Barrier(cfg.n_walkers + 1)
        self._shutdown = threading.Event()
        self._errors: list[BaseException] = []

        self._threads = [
            threading.Thread(
                target=self._worker_loop,
                args=(i,),
                name=f"walker-{i}",
                daemon=True,
            )
            for i in range(cfg.n_walkers)
        ]
        for th in self._threads:
            th.start()

    def __enter__(self) -> "WalkerPool":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        """Wake walkers with shutdown=True; let them exit. Idempotent."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        for ev in self._wakes:
            ev.set()
  # Walkers won't hit the barrier in shutdown mode, so don't wait on it.
        for th in self._threads:
  # daemon=True; bounded join just ensures clean teardown for tests.
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
            raise RuntimeError("WalkerPool is closed")

  # Publish the job and wake every walker. Walkers signal completion
  # via the barrier; we participate too, so the caller blocks here
  # until budget is fully consumed (or stop_event fires).
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
  # A dead walker means the tree is underfilled; any bestmove would
  # be based on partial data, so re-raise instead of returning early.
            raise self._errors[0]
  # Semaphore counter isn't portably readable; best-effort.
        return target_sims

    def _worker_loop(self, idx: int) -> None:
        cfg = self._cfg
        c_puct = cfg.c_puct
        fpu_root = cfg.fpu_at_root
        fpu_red = cfg.fpu_reduction
        vloss = cfg.vloss_weight
        gather = max(1, int(cfg.gather))
        enc = np.empty((gather, 146, 8, 8), dtype=np.float32)
        my_wake = self._wakes[idx]

        while True:
            my_wake.wait()
            my_wake.clear()
            if self._shutdown.is_set():
                return
            job = self._job
            if job is None:  # spurious wake (shouldn't happen, defensive)
                continue
            try:
                self._descend_until_done(
                    job, enc, gather, c_puct, fpu_root, fpu_red, vloss,
                )
            except Exception as exc:
                _log.exception("walker thread raised; requesting pool stop")
                self._errors.append(exc)
                job.stop_event.set()
            self._done.wait()

    def _descend_until_done(
        self, job: _Job, enc: np.ndarray, gather: int,
        c_puct: float, fpu_root: float, fpu_red: float, vloss: int,
    ) -> None:
        tree = job.tree
        root_id = job.root_id
        root_cb = job.root_cboard
        budget = job.budget
        stop_event = job.stop_event
        evaluator = self._evaluator
        while not stop_event.is_set():
  # Budget is acquired per-leaf (not per-gather) so a partially-drained
  # budget doesn't leave sims on the table. Terminal leaves backprop
  # inline; non-terminals are queued for one batched NN call.
            pending_slots: list[int] = []
            pending_paths = []
            pending_legals = []
            acquired = 0
            for i in range(gather):
                if stop_event.is_set() or not budget.acquire(blocking=False):
                    break
                acquired += 1
                _, path, legal, term_q = tree.walker_descend_puct(
                    root_id, root_cb, c_puct, fpu_root, fpu_red, vloss,
                    enc[i:i+1],
                )
                if term_q is not None:
                    tree.backprop(path, float(term_q))
                else:
                    pending_slots.append(i)
                    pending_paths.append(path)
                    pending_legals.append(legal)
            if acquired == 0:
                return
            if not pending_paths:
                continue
  # Hot path is full-gather with no terminals → contiguous slice. Fancy
  # indexing copies, so reserve it for the partial-gather case.
            n_pending = len(pending_slots)
            if n_pending == gather:
                xs = enc[:n_pending]
            else:
                xs = enc[pending_slots]
            pol, wdl_arr = evaluator.evaluate_encoded(xs)
            for k in range(n_pending):
                tree.walker_integrate_leaf(
                    pending_paths[k], pending_legals[k],
                    pol[k], wdl_arr[k], vloss,
                )
