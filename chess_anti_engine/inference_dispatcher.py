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


class BatchCoalescingDispatcher:
    """Coalesce concurrent ``evaluate_encoded`` calls into one GPU submit.

    With a walker pool, N threads each submit batch=1 calls in a tight
    descend → evaluate → integrate loop. At batch=1 each call saturates
    dispatch overhead but uses <1% of transformer compute capacity — the
    GPU is lying idle most of the time on a single call's timeline.

    This wrapper coalesces. Any caller finding no active submit becomes
    the submitter: they drain the pending queue into one ``np.concatenate``,
    submit to the inner dispatcher, and distribute result slices back via
    per-caller Events. While the submitter is mid-GPU, other walkers'
    calls accumulate in pending — the NEXT submit picks them up as one
    batch.

    No artificial wait/timer window. Observed coalescing factor matches
    the number of walkers whose CPU descend completes within one GPU
    call's wall time. For 4 walkers on a ~5ms GPU call with ~1ms CPU
    descend, this is typically 3–4.

    Inner dispatcher must be thread-safe (only one submitter at a time
    from this layer, but mixing coalescing with non-coalescing callers
    would need outer serialization — don't).
    """

    def __init__(self, inner, max_batch: int = 128) -> None:
        self._inner = inner
        self._max_batch = int(max_batch)
        self._lock = threading.Lock()
        # Each pending entry: (x_array, done_event, result_box: list[...|None]).
        self._pending: list[tuple[np.ndarray, threading.Event, list]] = []
        self._busy = False  # True while someone is mid-submit

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        done = threading.Event()
        result: list[tuple[np.ndarray, np.ndarray] | None] = [None]
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
        """Drain pending until empty. Any caller arriving during a submit
        queues up and is picked up on the next iteration."""
        while batch:
            xs = np.concatenate([entry[0] for entry in batch], axis=0)
            try:
                pol, wdl = self._inner.evaluate_encoded(xs)
            except BaseException as exc:
                # Propagate to every waiter — don't leave walkers hung on the
                # event. Each caller re-raises out of its own evaluate_encoded.
                for _, ev, res in batch:
                    res[0] = exc  # type: ignore[assignment]
                    ev.set()
                with self._lock:
                    self._busy = False
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

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._inner, name)


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
