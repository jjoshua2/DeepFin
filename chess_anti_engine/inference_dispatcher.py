"""Dispatchers that wrap one or more GPU evaluators for concurrent access.

- ``ThreadSafeGPUDispatcher``: single evaluator + lock (minimum safety for
  multi-threaded callers sharing pinned buffers).
- ``MultiGPUDispatcher``: N evaluators, route each call to the device with
  the smallest in-flight call count (ties broken round-robin).
- ``BatchCoalescingDispatcher``: merge concurrent batch=1 calls into one
  ``np.concatenate``-d submit so the GPU sees batch=N.

None of these expose ``evaluate_encoded_async``. ``gumbel_c.py`` gates its
pipelined path on ``hasattr(eval, 'evaluate_encoded_async')``, and the
pipelined path requires ``n_boards >= 64`` — irrelevant for single-game
UCI and would bypass the dispatch/coalesce logic if silently proxied. The
explicit ``max_batch`` property is the only attribute surface we promise.
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

    @property
    def max_batch(self) -> int:
        return int(getattr(self._eval, "_max_batch", getattr(self._eval, "max_batch", 0)))


class BatchCoalescingDispatcher:
    """Merge concurrent ``evaluate_encoded`` calls into one GPU submit.

    With a walker pool, N threads submit batch=1 in a tight loop. Any
    caller finding no submit in flight becomes the submitter: drains the
    pending queue into one ``np.concatenate``, submits to the inner
    dispatcher, and distributes result slices back via per-caller Events.
    Callers arriving during a submit accumulate in pending for the next
    iteration. No artificial timer — the submit window IS the time the
    current submitter spends preparing and executing.

    Observed coalescing factor equals the number of walker threads whose
    CPU descend completes within one GPU call's wall time. At 4 walkers
    on a ~5ms call that's typically 3-4.
    """

    def __init__(self, inner, max_batch: int = 128) -> None:
        self._inner = inner
        self._max_batch = int(max_batch)
        self._lock = threading.Lock()
        self._pending: list[tuple[np.ndarray, threading.Event, list]] = []
        self._busy = False

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        done = threading.Event()
        result: list[tuple[np.ndarray, np.ndarray] | BaseException | None] = [None]
        claim: list[tuple[np.ndarray, threading.Event, list]] | None = None
        with self._lock:
            self._pending.append((x, done, result))
            if not self._busy:
                self._busy = True
                claim = self._pending[:self._max_batch]
                self._pending = self._pending[self._max_batch:]

        if claim is not None:
            self._run_submitter_loop(claim)
        done.wait()
        got = result[0]
        if isinstance(got, BaseException):
            raise got
        assert got is not None
        return got

    def _run_submitter_loop(
        self,
        batch: list[tuple[np.ndarray, threading.Event, list]],
    ) -> None:
        while batch:
            xs = np.concatenate([entry[0] for entry in batch], axis=0)
            try:
                pol, wdl = self._inner.evaluate_encoded(xs)
            except BaseException as exc:
                # Propagate to every waiter so no walker hangs on its Event.
                # Also drain pending: callers that arrived after our claim
                # but before this failure are stuck on done.wait() too.
                for _, ev, res in batch:
                    res[0] = exc
                    ev.set()
                with self._lock:
                    pending = self._pending
                    self._pending = []
                    self._busy = False
                for _, ev, res in pending:
                    res[0] = exc
                    ev.set()
                raise
            offset = 0
            for x, ev, res in batch:
                n = x.shape[0]
                res[0] = (pol[offset:offset + n], wdl[offset:offset + n])
                ev.set()
                offset += n
            with self._lock:
                if not self._pending:
                    self._busy = False
                    batch = []
                else:
                    batch = self._pending[:self._max_batch]
                    self._pending = self._pending[self._max_batch:]

    @property
    def max_batch(self) -> int:
        return self._max_batch


class MultiGPUDispatcher:
    """Route ``evaluate_encoded`` across N device-local evaluators.

    Each evaluator lives on its own CUDA device (or CPU, for testing)
    with its own pinned buffers, compiled graph, and lock. On every call
    the dispatcher picks the evaluator with the fewest in-flight calls
    (ties broken round-robin) and serializes that device's buffer access
    on its lock. Scales ~linearly with walkers × devices until each
    device's SMs saturate.
    """

    def __init__(self, evaluators: Sequence[BatchEvaluator]) -> None:
        if not evaluators:
            raise ValueError("MultiGPUDispatcher requires at least one evaluator")
        self._evals = list(evaluators)
        self._locks = [threading.Lock() for _ in self._evals]
        self._inflight = [0] * len(self._evals)
        self._select_lock = threading.Lock()
        self._rr = 0  # round-robin tiebreaker cursor

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
            best = 0
            best_load = self._inflight[0]
            for i in range(1, len(self._inflight)):
                if self._inflight[i] < best_load:
                    best = i
                    best_load = self._inflight[i]
            ties = [i for i, n in enumerate(self._inflight) if n == best_load]
            if len(ties) > 1:
                best = ties[self._rr % len(ties)]
                self._rr = (self._rr + 1) % len(self._evals)
            self._inflight[best] += 1
            return best

    @property
    def n_devices(self) -> int:
        return len(self._evals)

    @property
    def max_batch(self) -> int:
        return int(getattr(self._evals[0], "_max_batch",
                           getattr(self._evals[0], "max_batch", 0)))
