"""Walker-thread pool driving PUCT descent with virtual loss.

Phase 5 of the UCI scaling plan. Opt-in via ``--walkers N`` (N > 1). Each
walker runs a tight loop:

    leaf_id, path, legal, term_q = tree.walker_descend_puct(...)
    if term_q is not None:
        tree.backprop(path, term_q)              # terminal leaf
    else:
        pol, wdl = evaluator.evaluate_encoded(enc)  # releases the GIL
        tree.walker_integrate_leaf(path, legal, pol[0], wdl[0], vloss)

Virtual loss inside ``walker_descend_puct`` keeps concurrent walkers off
each other's in-flight paths. The evaluator must be thread-safe — pair
with ``ThreadSafeGPUDispatcher`` from phase 1. The tree must have had
``reserve()`` called so concurrent descents cannot trigger a realloc.

Trade-off vs. the existing Gumbel+halving path: we drop sequential
halving semantics — walkers do plain PUCT, so policy distribution is
visit-count from PUCT rather than Gumbel's top-m survivor set. In
practice for single-game UCI this is fine; for training or Gumbel-style
exploration, keep walkers=1 and use the classic path.
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
    """Run ``target_sims`` simulations on a shared tree with N worker threads.

    The pool is stateless between ``run`` calls — pass the tree, root node
    id, and root CBoard each time. The tree is expected to be pre-expanded
    at the root (walker_descend_puct would otherwise race on root
    expansion and waste N-1 NN evals on the first sim). Callers obtain
    root policy + wdl once and pass the expanded root in.
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
        """Run up to ``target_sims`` simulations across the pool. Returns the
        actual number completed (may be < target if ``stop_event`` fired).

        Safe to call repeatedly within a game with the same tree — the tree
        accumulates visits across calls."""
        if target_sims <= 0:
            return 0

        cfg = self._cfg
        # Shared progress counter. `target` is the ceiling; workers exit as
        # soon as any of {stop_event, counter >= target} becomes true.
        counter = _SharedCounter(target_sims)

        threads = [
            threading.Thread(
                target=_worker_loop,
                args=(tree, root_id, root_cboard, self._evaluator,
                      cfg, counter, stop_event),
                name=f"walker-{i}",
                daemon=True,
            )
            for i in range(cfg.n_walkers)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        return counter.done


class _SharedCounter:
    """Atomic done/target counter. Single mutex; only taken for O(1) work
    per simulation — contention is low relative to the NN-eval wait."""

    __slots__ = ("_lock", "target", "done")

    def __init__(self, target: int) -> None:
        self._lock = threading.Lock()
        self.target = int(target)
        self.done = 0

    def claim(self) -> bool:
        """Reserve one sim slot. Returns False if target already reached."""
        with self._lock:
            if self.done >= self.target:
                return False
            self.done += 1
            return True


def _worker_loop(
    tree: MCTSTree,
    root_id: int,
    root_cboard: _CBoard,
    evaluator: _Evaluator,
    cfg: WalkerPoolConfig,
    counter: _SharedCounter,
    stop_event: threading.Event,
) -> None:
    """One walker's body. Owns its encoding buffer; the tree + evaluator
    are shared. Exits on stop_event or when the target is reached."""
    enc = np.empty((1, 146, 8, 8), dtype=np.float32)
    c_puct = cfg.c_puct
    fpu_root = cfg.fpu_at_root
    fpu_red = cfg.fpu_reduction
    vloss = cfg.vloss_weight

    while not stop_event.is_set():
        if not counter.claim():
            return
        _, path, legal, term_q = tree.walker_descend_puct(
            root_id, root_cboard, c_puct, fpu_root, fpu_red, vloss, enc,
        )
        if term_q is not None:
            tree.backprop(path, float(term_q))
            continue
        pol, wdl = evaluator.evaluate_encoded(enc)
        pol_arr = np.ascontiguousarray(pol[0], dtype=np.float32)
        wdl_arr = np.ascontiguousarray(wdl[0], dtype=np.float32)
        tree.walker_integrate_leaf(path, legal, pol_arr, wdl_arr, vloss)
