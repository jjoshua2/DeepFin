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

Trade-off vs. the Gumbel+halving path: drops sequential halving, so the
policy distribution is visit-count from PUCT rather than Gumbel's top-m
survivors. For single-game UCI that's fine; for training-style runs,
keep walkers=1 and use the classic path.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree


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
  # Per-walker leaf gather: each walker does up to `gather` descents
  # before submitting one NN batch. gather=1 is the classic "one leaf
  # per NN call" shape; gather=G amplifies the effective submit batch
  # to N_walkers×G. Virtual loss diversifies descents within the gather
  # the same way it diversifies across walkers. Default 1 preserves
  # existing behavior; Lc0's canonical leaf-gather is 8.
    gather: int = 1


class WalkerPool:
    """Run ``target_sims`` simulations on a shared tree with N threads.

    The pool is stateless between ``run`` calls — pass the tree, root node
    id, and root CBoard each time. The caller must pre-expand the root;
    otherwise all N walkers race on the same unexpanded leaf and waste
    N-1 NN evals on the first sim.
    """

    def __init__(self, cfg: WalkerPoolConfig, evaluator: _Evaluator) -> None:
        self._cfg = cfg
        self._evaluator = evaluator

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

        cfg = self._cfg
  # Semaphore models N remaining claims; workers exit when acquire
  # returns False. Also serves as the "done" counter via target - value.
        budget = threading.Semaphore(target_sims)

  # Per-pool exception capture. First worker to fail pushes here and
  # sets ``stop_event`` so siblings exit early instead of running a
  # full sim budget against a broken evaluator / tree. CPython
  # list.append is GIL-atomic so no explicit lock needed.
        errors: list[BaseException] = []

        threads = [
            threading.Thread(
                target=_worker_loop,
                args=(tree, root_id, root_cboard, self._evaluator,
                      cfg, budget, stop_event, errors),
                name=f"walker-{i}",
                daemon=True,
            )
            for i in range(cfg.n_walkers)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        if errors:
  # Re-raise the first one; a dead walker means the search
  # tree is underfilled and any bestmove would be based on
  # partial data. Caller (Engine._run_one_phase) catches and
  # surfaces via ``info string search error``.
            raise errors[0]
  # Semaphore's internal counter is target - claimed; we cannot read
  # it portably, so return target_sims as the best-effort count. The
  # caller only uses this for progress logging.
        return target_sims


def _worker_loop(
    tree: MCTSTree,
    root_id: int,
    root_cboard: _CBoard,
    evaluator: _Evaluator,
    cfg: WalkerPoolConfig,
    budget: threading.Semaphore,
    stop_event: threading.Event,
    errors: list[BaseException],
) -> None:
    import logging as _logging
    _log = _logging.getLogger(__name__)
    c_puct = cfg.c_puct
    fpu_root = cfg.fpu_at_root
    fpu_red = cfg.fpu_reduction
    vloss = cfg.vloss_weight
    gather = max(1, int(cfg.gather))
  # Preallocate once; each iteration writes into enc[0:k] for the k
  # leaves gathered this round. Slicing `enc[i:i+1]` gives descend a
  # writable 1-row view without another allocation.
    enc = np.empty((gather, 146, 8, 8), dtype=np.float32)

    try:
        while not stop_event.is_set():
  # Gather up to `gather` leaves. Track (slot, path, legal) for
  # non-terminal leaves so we can submit one batch to the NN
  # and re-integrate in order. Terminal leaves backprop inline —
  # no NN eval needed. Budget acquired per-leaf so a partially-
  # drained budget doesn't leave sims on the table.
            pending_slots: list[int] = []
            pending_paths = []
            pending_legals = []
            acquired = 0
            for i in range(gather):
                if stop_event.is_set() or not budget.acquire(blocking=False):
                    break
                acquired += 1
                _, path, legal, term_q = tree.walker_descend_puct(
                    root_id, root_cboard, c_puct, fpu_root, fpu_red, vloss,
                    enc[i:i+1],
                )
                if term_q is not None:
                    tree.backprop(path, float(term_q))
                else:
                    pending_slots.append(i)
                    pending_paths.append(path)
                    pending_legals.append(legal)
            if acquired == 0:
  # Budget drained or stop signalled; exit.
                return
            if not pending_paths:
  # All gathered sims were terminal — no NN batch to submit.
                continue
  # Compact non-terminal rows into a contiguous array only when
  # needed. Full-gather with no terminals is the hot path.
            n_pending = len(pending_slots)
            if n_pending == gather and pending_slots[-1] == gather - 1:
                xs = enc[:n_pending]
            else:
                xs = enc[pending_slots]  # fancy-index → contiguous copy
            pol, wdl_arr = evaluator.evaluate_encoded(xs)
            for k in range(n_pending):
                tree.walker_integrate_leaf(
                    pending_paths[k], pending_legals[k],
                    pol[k], wdl_arr[k], vloss,
                )
    except Exception as exc:
  # Record the failure so ``WalkerPool.run`` can re-raise after join,
  # and signal siblings to stop — running a full sim budget against
  # a broken evaluator would just produce N identical failures and
  # delay the error surfacing. ``KeyboardInterrupt`` / ``SystemExit``
  # deliberately propagate out of the thread (daemon, so Python
  # lets them die without join).
        _log.exception("walker thread raised; requesting pool stop")
        errors.append(exc)
        stop_event.set()
