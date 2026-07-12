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
from dataclasses import dataclass

import numpy as np
import torch

from chess_anti_engine.inference import BatchEvaluator
import contextlib


@dataclass(slots=True)
class _CoalesceRequest:
    x: np.ndarray
    relations: np.ndarray | None
    legal_flat: np.ndarray | None
    legal_counts: np.ndarray | None
    done: threading.Event
    result: list


class ThreadSafeGPUDispatcher:
    def __init__(self, evaluator: BatchEvaluator) -> None:
        self._eval = evaluator
        self._lock = threading.Lock()

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if relations is not None:
                return self._eval.evaluate_encoded(x, relations=relations)
            return self._eval.evaluate_encoded(x)

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._eval.evaluate_legal_bf16(  # pyright: ignore[reportAttributeAccessIssue]
                x, legal_flat, legal_counts,
            )

    @property
    def supports_input_bf16_bits(self) -> bool:
        return bool(getattr(self._eval, "supports_input_bf16_bits", False))

    @property
    def supports_legal_bf16(self) -> bool:
        # Honor the inner evaluator's opt-out (CAE_UCI_COMPACT_BF16 gate) —
        # a bare hasattr would re-enable the compact path the gate disabled.
        return bool(getattr(
            self._eval, "supports_legal_bf16",
            hasattr(self._eval, "evaluate_legal_bf16"),
        ))

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
        relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if relations is not None:
                return self._eval.evaluate_inplace(  # pyright: ignore[reportAttributeAccessIssue]
                    bsz, copy_out=copy_out, slot=slot, relations=relations,
                )
            return self._eval.evaluate_inplace(  # pyright: ignore[reportAttributeAccessIssue]
                bsz, copy_out=copy_out, slot=slot,
            )

    def evaluate_inplace_async(
        self, bsz: int, *, slot: int = 0, relations: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        with self._lock:
            if relations is not None:
                return self._eval.evaluate_inplace_async(bsz, slot=slot, relations=relations)  # pyright: ignore[reportAttributeAccessIssue]
            return self._eval.evaluate_inplace_async(bsz, slot=slot)  # pyright: ignore[reportAttributeAccessIssue]

    def evaluate_encoded_async(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        with self._lock:
            if relations is not None:
                return self._eval.evaluate_encoded_async(x, relations=relations)  # pyright: ignore[reportAttributeAccessIssue]
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
        self._pending: list[_CoalesceRequest] = []
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
        for request in stranded:
            request.result[0] = err
            request.done.set()
        self._wake.set()
        if self._submitter.is_alive():
            self._submitter.join(timeout=30.0)

    def __del__(self) -> None:
  # Best-effort close on GC, in case caller forgets.
        with contextlib.suppress(Exception):
            self.close()

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
  # Fail fast on single requests that can never be dispatched. Without
  # this, an oversize lone request would sit in pending and eventually
  # be forwarded to the inner evaluator (which raises), which is
  # harder to debug — the caller blocked on done.wait() gets a
  # generic error from another thread's context.
        n = int(x.shape[0])
        if n > self._max_batch:
            raise ValueError(
                f"request batch {n} > coalescer max {self._max_batch}")
        return self._enqueue(x, relations=relations)

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        counts = np.asarray(legal_counts, dtype=np.int32)
        flat = np.asarray(legal_flat, dtype=np.int32)
        if counts.ndim != 1 or counts.shape[0] != int(x.shape[0]):
            raise ValueError(
                f"legal_counts must be shape ({x.shape[0]},), got {counts.shape}"
            )
        if flat.ndim != 1 or int(counts.sum()) != int(flat.shape[0]):
            raise ValueError("legal_flat must be 1D with len == sum(legal_counts)")
        return self._enqueue(x, legal_flat=flat, legal_counts=counts)

    def _enqueue(
        self,
        x: np.ndarray,
        *,
        relations: np.ndarray | None = None,
        legal_flat: np.ndarray | None = None,
        legal_counts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(x.shape[0])
        if n > self._max_batch:
            raise ValueError(f"request batch {n} > coalescer max {self._max_batch}")
        done = threading.Event()
        result: list[tuple[np.ndarray, np.ndarray] | BaseException | None] = [None]
        with self._lock:
  # Reject new work after close() rather than silently waiting on
  # a submitter thread that's about to exit.
            if self._shutdown.is_set():
                raise RuntimeError("BatchCoalescingDispatcher is closed")
            self._pending.append(_CoalesceRequest(
                x, relations, legal_flat, legal_counts, done, result,
            ))
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
                    first_legal = self._pending[0].legal_counts is not None
                    for entry in self._pending:
                        if (entry.legal_counts is not None) != first_legal:
                            break
                        n = entry.x.shape[0]
                        if cut > 0 and rows + n > self._max_batch:
                            break
                        rows += n
                        cut += 1
                    batch = self._pending[:cut]
                    self._pending = self._pending[cut:]
                singleton = len(batch) == 1
                xs = (
                    batch[0].x if singleton
                    else np.concatenate([entry.x for entry in batch], axis=0)
                )
  # Relations coalesce exactly like x: all-or-nothing per submit.
  # Mixed submits zero-fill the missing rows (zero matrices == zero
  # bias, identical to the absent-tensor semantics).
                rels = batch[0].relations if singleton else None
                if not singleton and any(entry.relations is not None for entry in batch):
                    rels = np.zeros((xs.shape[0], 5, 64, 64), dtype=np.uint8)
                    off = 0
                    for entry in batch:
                        n_rows = entry.x.shape[0]
                        if entry.relations is not None:
                            rels[off:off + n_rows] = entry.relations
                        off += n_rows
                try:
                    if first_legal:
                        if singleton:
                            legal_flat = batch[0].legal_flat
                            legal_counts = batch[0].legal_counts
                            assert legal_flat is not None
                            assert legal_counts is not None
                        else:
                            legal_flat = np.concatenate([
                                entry.legal_flat for entry in batch
                                if entry.legal_flat is not None
                            ])
                            legal_counts = np.concatenate([
                                entry.legal_counts for entry in batch
                                if entry.legal_counts is not None
                            ])
                        pol, wdl = self._inner.evaluate_legal_bf16(
                            xs, legal_flat, legal_counts,
                        )
                    else:
                        pol, wdl = (
                            self._inner.evaluate_encoded(xs, relations=rels)
                            if rels is not None
                            else self._inner.evaluate_encoded(xs)
                        )
                except BaseException as exc:
  # Wake every waiter with the exception so no walker
  # hangs, then drain anything that arrived during the
  # failed submit too.
                    for request in batch:
                        request.result[0] = exc
                        request.done.set()
                    with self._lock:
                        pending = self._pending
                        self._pending = []
                    for request in pending:
                        request.result[0] = exc
                        request.done.set()
                    continue
                row_offset = 0
                pol_offset = 0
                for request in batch:
                    n = request.x.shape[0]
                    n_pol = (
                        int(request.legal_counts.sum())
                        if request.legal_counts is not None else n
                    )
                    request.result[0] = (
                        pol[pol_offset:pol_offset + n_pol],
                        wdl[row_offset:row_offset + n],
                    )
                    request.done.set()
                    row_offset += n
                    pol_offset += n_pol
  # Exit only after pending is fully drained — ``close()`` already
  # emptied pending into stranded-err waiters, so at this point
  # there's nothing the submitter can do except leave.
            if self._shutdown.is_set():
                return

    @property
    def max_batch(self) -> int:
        return self._max_batch

    @property
    def supports_input_bf16_bits(self) -> bool:
        return bool(getattr(self._inner, "supports_input_bf16_bits", False))

    @property
    def supports_legal_bf16(self) -> bool:
        return bool(getattr(
            self._inner, "supports_legal_bf16",
            hasattr(self._inner, "evaluate_legal_bf16"),
        ))


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

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = self._pick_device()
        try:
            with self._locks[idx]:
                if relations is not None:
                    return self._evals[idx].evaluate_encoded(x, relations=relations)
                return self._evals[idx].evaluate_encoded(x)
        finally:
            with self._select_lock:
                self._inflight[idx] -= 1

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = self._pick_device()
        try:
            with self._locks[idx]:
                return self._evals[idx].evaluate_legal_bf16(  # pyright: ignore[reportAttributeAccessIssue]
                    x, legal_flat, legal_counts,
                )
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

    @property
    def supports_input_bf16_bits(self) -> bool:
        return all(
            bool(getattr(evaluator, "supports_input_bf16_bits", False))
            for evaluator in self._evals
        )

    @property
    def supports_legal_bf16(self) -> bool:
        return all(
            getattr(e, "supports_legal_bf16", hasattr(e, "evaluate_legal_bf16"))
            for e in self._evals
        )
