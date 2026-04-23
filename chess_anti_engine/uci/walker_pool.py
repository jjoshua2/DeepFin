"""Walker-thread pool: concurrent PUCT descent with virtual loss.

Each walker runs a tight loop:

    leaf_id, path, legal, term_q = tree.walker_descend_puct(...)
    if term_q is not None:
        tree.backprop(path, term_q)              # terminal leaf
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

        threads = [
            threading.Thread(
                target=_worker_loop,
                args=(tree, root_id, root_cboard, self._evaluator,
                      cfg, budget, stop_event),
                name=f"walker-{i}",
                daemon=True,
            )
            for i in range(cfg.n_walkers)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
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
) -> None:
    import logging as _logging
    _log = _logging.getLogger(__name__)
    enc = np.empty((1, 146, 8, 8), dtype=np.float32)
    c_puct = cfg.c_puct
    fpu_root = cfg.fpu_at_root
    fpu_red = cfg.fpu_reduction
    vloss = cfg.vloss_weight

    try:
        while not stop_event.is_set():
            if not budget.acquire(blocking=False):
                return
            _, path, legal, term_q = tree.walker_descend_puct(
                root_id, root_cboard, c_puct, fpu_root, fpu_red, vloss, enc,
            )
            if term_q is not None:
                tree.backprop(path, float(term_q))
                continue
            pol, wdl = evaluator.evaluate_encoded(enc)
            tree.walker_integrate_leaf(path, legal, pol[0], wdl[0], vloss)
    except BaseException:
        # Log and swallow so the join() in WalkerPool.run doesn't see a
        # dangling thread — Python's threading re-raises nothing to the
        # parent anyway, but the silent-exit path corrupted state in PyTorch's
        # C++ CUDA layer on release (manifesting as ``terminate called
        # without an active exception`` on shutdown). Explicit trap here
        # surfaces the root cause and keeps cleanup deterministic.
        _log.exception("walker thread raised; exiting worker loop")
