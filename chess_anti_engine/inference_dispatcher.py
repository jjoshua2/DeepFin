"""Dispatchers that wrap one or more GPU evaluators for concurrent access.

- ``ThreadSafeGPUDispatcher``: single evaluator + lock (minimum safety for
  multi-threaded callers sharing pinned buffers).
- ``MultiGPUDispatcher``: N evaluators, route each call to the device with
  the smallest in-flight call count (ties broken round-robin).
- ``BatchCoalescingDispatcher``: merge concurrent batch=1 calls into one
  ``np.concatenate``-d submit so the GPU sees batch=N.

``ThreadSafeGPUDispatcher`` forwards the slot-pool inplace API
(``get_input_buffer`` / ``evaluate_inplace`` / ``evaluate_inplace_async``)
when the wrapped evaluator implements it. UCI single-game searches
(``n_walkers=1``) thus skip the input + output memcpys.

Slot-pool API contract under the dispatcher: callers must hold a slot for
its full write→submit→read cycle. The lock serializes submits but does
not prevent two threads picking the same slot — that's the caller's job.
gumbel_c uses one Python thread (with two slots for its own pipeline), so
the constraint is satisfied; walker-pool paths don't go through gumbel_c.

``BatchCoalescingDispatcher`` and ``MultiGPUDispatcher`` deliberately do
NOT forward the inplace API: their value is in the dispatch / coalesce
logic, which slot-aliasing would route around.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np
import torch

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

  # Inplace API forwarders. Defined unconditionally so hasattr() probes
  # see them; calls fall through to AttributeError on the inner if it
  # doesn't implement them — the caller's hasattr() will fail then.
    @property
    def n_slots(self) -> int:
        return int(getattr(self._eval, "n_slots", 1))

    @property
    def _max_batch(self) -> int:
        return self.max_batch

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        return self._eval.get_input_buffer(bsz, slot=slot)  # pyright: ignore[reportAttributeAccessIssue]

    def evaluate_inplace(
        self, bsz: int, *, copy_out: bool = True, slot: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._eval.evaluate_inplace(  # pyright: ignore[reportAttributeAccessIssue]
                bsz, copy_out=copy_out, slot=slot,
            )

    def evaluate_inplace_async(
        self, bsz: int, *, slot: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        with self._lock:
            return self._eval.evaluate_inplace_async(bsz, slot=slot)  # pyright: ignore[reportAttributeAccessIssue]

    def evaluate_encoded_async(
        self, x: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        with self._lock:
            return self._eval.evaluate_encoded_async(x)  # pyright: ignore[reportAttributeAccessIssue]


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
  # Fail fast on single requests that can never be dispatched. Without
  # this, an oversize lone request would sit in pending and eventually
  # be forwarded to the inner evaluator (which raises), which is
  # harder to debug — the caller blocked on done.wait() gets a
  # generic error from another thread's context.
        n = int(x.shape[0])
        if n > self._max_batch:
            raise ValueError(
                f"request batch {n} > coalescer max {self._max_batch}")
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
  # Pack by cumulative rows, not request count. Each
  # request is x.shape[0] rows (leaf-gather > 1 with the
  # walker pool), and the inner evaluator caps at
  # _max_batch rows. Always include the head so a lone
  # oversize request (already rejected in evaluate_encoded,
  # but defensive) surfaces an error instead of stalling.
                    rows = 0
                    cut = 0
                    for entry in self._pending:
                        n = entry[0].shape[0]
                        if cut > 0 and rows + n > self._max_batch:
                            break
                        rows += n
                        cut += 1
                    batch = self._pending[:cut]
                    self._pending = self._pending[cut:]
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
