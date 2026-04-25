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
  # Each walker does up to `gather` descents before submitting one NN
  # batch — virtual loss diversifies within a gather the same way it
  # diversifies across walkers. Lc0's canonical default is 8; we keep
  # 1 to preserve the classic one-leaf-per-call shape.
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
  # Semaphore models N remaining sim claims; workers exit when
  # ``acquire(blocking=False)`` returns False.
        budget = threading.Semaphore(target_sims)

  # Per-pool exception capture. CPython ``list.append`` is GIL-atomic
  # so no explicit lock; workers also set ``stop_event`` so siblings
  # exit instead of burning the rest of the budget on the same fault.
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
  # A dead walker means the tree is underfilled; any bestmove would
  # be based on partial data, so re-raise instead of returning early.
            raise errors[0]
  # Semaphore counter isn't portably readable; best-effort.
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
    enc = np.empty((gather, 146, 8, 8), dtype=np.float32)

    try:
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
    except Exception as exc:
  # KeyboardInterrupt / SystemExit deliberately propagate (daemon
  # threads, Python lets them die without join).
        _log.exception("walker thread raised; requesting pool stop")
        errors.append(exc)
        stop_event.set()
