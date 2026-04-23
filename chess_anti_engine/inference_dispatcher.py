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

    With a walker pool, N threads submit batch=1 in a tight loop. Callers
    push their x into a pending queue + wait on a per-call Event. A single
    persistent **submitter thread** drains the queue, coalesces into one
    ``np.concatenate``, submits to the inner dispatcher, and distributes
    result slices back via Events. Callers arriving during a submit
    accumulate in pending for the next round.

    Why a dedicated thread (rather than whichever caller wins the "I'll
    submit" race): torch.compile in reduce-overhead mode captures CUDA
    graphs that are thread-local in practice (stream context + cached
    autograd state). When submits come from varying threads, the capture
    state drifts, and the mismatched cleanup on interpreter shutdown
    shows up as ``terminate called without an active exception``.
    Pinning the submit to one thread makes torch.compile's internal
    state deterministic.

    Observed coalescing factor equals the number of walker threads whose
    CPU descend completes within one GPU call's wall time. At 4 walkers
    on a ~5ms call that's typically 3-4.
    """

    def __init__(self, inner, max_batch: int = 128) -> None:
        self._inner = inner
        self._max_batch = int(max_batch)
        self._lock = threading.Lock()
        self._pending: list[tuple[np.ndarray, threading.Event, list]] = []
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        # Daemon so tests / scripts that forget ``close()`` don't hang at
        # interpreter shutdown. The UCI main loop calls ``close()`` in a
        # ``finally`` before returning, so the CUDA-owning path still drains
        # deterministically before Python tears down torch's C++ context
        # (which was the original ``terminate called without an active
        # exception`` failure mode).
        self._submitter = threading.Thread(
            target=self._submitter_loop,
            name="coalesce-submitter",
            daemon=True,
        )
        self._submitter.start()

    def close(self) -> None:
        """Stop the submitter thread and fail any in-flight or pending
        submits with a ``RuntimeError``. Idempotent.

        Shutdown is atomic: under ``_lock`` we flip ``_shutdown`` and snapshot
        ``_pending`` into a local. Every waiter (in the snapshot) has its
        result set to the shutdown error and its Event released before we
        signal the submitter to exit. That removes the race where a caller
        appends right before close and then blocks forever on an already-
        exited submitter. After ``close()`` returns, no new submits are
        accepted (see ``evaluate_encoded``).
        """
        with self._lock:
            if self._shutdown.is_set():
                return
            self._shutdown.set()
            stranded = self._pending
            self._pending = []
        err = RuntimeError("BatchCoalescingDispatcher is closed")
        for _, ev, res in stranded:
            res[0] = err
            ev.set()
        self._wake.set()
        if self._submitter.is_alive():
            self._submitter.join(timeout=30.0)

    def __del__(self) -> None:
        # Best-effort close on GC, in case caller forgets.
        try:
            self.close()
        except Exception:
            pass

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        done = threading.Event()
        result: list[tuple[np.ndarray, np.ndarray] | BaseException | None] = [None]
        with self._lock:
            # Reject new work after close() rather than silently waiting on
            # a submitter thread that's about to exit.
            if self._shutdown.is_set():
                raise RuntimeError("BatchCoalescingDispatcher is closed")
            self._pending.append((x, done, result))
        self._wake.set()
        done.wait()
        got = result[0]
        if isinstance(got, BaseException):
            raise got
        assert got is not None
        return got

    def _submitter_loop(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            # Drain any work that landed before shutdown AND while the
            # submitter was busy on the prior batch — so a caller that
            # appended in the window between our wake.clear and our lock
            # acquire isn't left stranded (close() also drains, but only
            # sees items at shutdown time).
            while True:
                with self._lock:
                    if not self._pending:
                        break
                    batch = self._pending[:self._max_batch]
                    self._pending = self._pending[self._max_batch:]
                xs = np.concatenate([entry[0] for entry in batch], axis=0)
                try:
                    pol, wdl = self._inner.evaluate_encoded(xs)
                except BaseException as exc:
                    # Wake every waiter with the exception so no walker
                    # hangs, then drain anything that arrived during the
                    # failed submit too.
                    for _, ev, res in batch:
                        res[0] = exc
                        ev.set()
                    with self._lock:
                        pending = self._pending
                        self._pending = []
                    for _, ev, res in pending:
                        res[0] = exc
                        ev.set()
                    continue
                offset = 0
                for x, ev, res in batch:
                    n = x.shape[0]
                    res[0] = (pol[offset:offset + n], wdl[offset:offset + n])
                    ev.set()
                    offset += n
            # Exit only after pending is fully drained — ``close()`` already
            # emptied pending into stranded-err waiters, so at this point
            # there's nothing the submitter can do except leave.
            if self._shutdown.is_set():
                return

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
