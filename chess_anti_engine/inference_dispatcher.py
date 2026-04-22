"""Thread-safe dispatcher around one or more GPU evaluators.

Wraps ``DirectGPUEvaluator`` (or any evaluator matching the ``BatchEvaluator``
protocol) behind a lock so multiple walker threads can submit ``evaluate_encoded``
concurrently without corrupting the shared pinned buffers. Calls serialize on
the lock — the dispatcher does NOT batch across callers (walkers already
produce halving-round-sized batches of ~256+ leaves on their own).

Multi-GPU (phase 7): ``MultiGPUDispatcher`` owns N evaluators (one per
device) and routes each call to the device with the smallest in-flight
call count. Ties break round-robin. Callers keep the same
``evaluate_encoded`` entry point.

The dispatcher deliberately does NOT expose ``evaluate_encoded_async``.
``gumbel_c.py`` gates its pipelined path on ``hasattr(..., 'evaluate_encoded_async')``,
and that path is for cross-game pipelining at ``n_boards >= 64`` — irrelevant
for single-game UCI. Forcing the sync path also makes the lock contract
straightforward: one caller's result is safely copied out before the next
caller's buffer write begins.
"""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from chess_anti_engine.inference import BatchEvaluator


class ThreadSafeGPUDispatcher:
    def __init__(self, evaluator: BatchEvaluator) -> None:
        self._eval = evaluator
        self._lock = threading.Lock()

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._eval.evaluate_encoded(x)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        # Proxy read-only attrs (device, _max_batch, model, etc.) to the
        # underlying evaluator. Only attribute lookup — mutations still go
        # through the wrapped object directly if the caller holds a reference.
        return getattr(self._eval, name)


class MultiGPUDispatcher:
    """Route ``evaluate_encoded`` across N device-local evaluators.

    Each evaluator lives on its own CUDA device (or CPU, for testing) with
    its own pinned buffers, compiled graph, and lock. On every call, the
    dispatcher picks the evaluator with the fewest in-flight calls (ties
    broken by round-robin) and serializes that device's buffer access on
    its lock. With walker pool = N * #walkers threads, this gives near-
    linear scaling up to the point where a single device's CUDA kernels
    saturate its SMs.

    For #evaluators = 1 this is equivalent to ``ThreadSafeGPUDispatcher``
    but with an extra ``_inflight`` counter. Overhead is ~50ns per call;
    safe to use as a uniform wrapper.
    """

    def __init__(self, evaluators: Sequence[BatchEvaluator]) -> None:
        if not evaluators:
            raise ValueError("MultiGPUDispatcher requires at least one evaluator")
        self._evals = list(evaluators)
        self._locks = [threading.Lock() for _ in self._evals]
        self._inflight = [0] * len(self._evals)
        self._select_lock = threading.Lock()
        self._rr = 0  # round-robin tiebreaker

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = self._pick_device()
        try:
            with self._locks[idx]:
                return self._evals[idx].evaluate_encoded(x)
        finally:
            with self._select_lock:
                self._inflight[idx] -= 1

    def _pick_device(self) -> int:
        with self._select_lock:
            # Scan for min. Ties go to the round-robin position if it qualifies,
            # otherwise first-seen wins. N is small (1–8) so linear scan is fine.
            best = 0
            best_load = self._inflight[0]
            for i in range(1, len(self._inflight)):
                if self._inflight[i] < best_load:
                    best = i
                    best_load = self._inflight[i]
            # Tiebreak: if multiple devices share best_load, prefer round-robin.
            ties = [i for i, n in enumerate(self._inflight) if n == best_load]
            if len(ties) > 1:
                start = self._rr % len(ties)
                best = ties[start]
                self._rr = (self._rr + 1) % len(self._evals)
            self._inflight[best] += 1
            return best

    @property
    def n_devices(self) -> int:
        return len(self._evals)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        # Proxy to device 0 for consistency — max_batch, etc. should match
        # across devices since they were constructed with the same config.
        return getattr(self._evals[0], name)
