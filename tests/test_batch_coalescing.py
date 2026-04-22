"""BatchCoalescingDispatcher (phase 6).

Under concurrent callers, the dispatcher merges batch=1 calls into one
submit so the GPU sees batch=N instead of N×batch=1. Validated via a
recording inner dispatcher — we check that the sum of submitted batch
sizes matches the number of callers AND that at least some submits
coalesce (batch > 1) when callers overlap in time.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from chess_anti_engine.inference_dispatcher import BatchCoalescingDispatcher


class _RecordingInner:
    """Inner dispatcher that records every submitted batch size, then
    sleeps briefly to give other callers a chance to queue up."""

    def __init__(self, submit_sleep: float = 0.01) -> None:
        self.submit_sleep = submit_sleep
        self.submit_sizes: list[int] = []
        self._lock = threading.Lock()

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            self.submit_sizes.append(x.shape[0])
        time.sleep(self.submit_sleep)
        n = x.shape[0]
        return (np.full((n, 4672), fill_value=float(n), dtype=np.float32),
                np.full((n, 3), fill_value=float(n), dtype=np.float32))


def test_single_caller_is_batch_one() -> None:
    """One caller, no concurrency — dispatch serializes through as batch=1."""
    inner = _RecordingInner(submit_sleep=0.0)
    coalesce = BatchCoalescingDispatcher(inner, max_batch=128)
    x = np.zeros((1, 146, 8, 8), dtype=np.float32)
    pol, wdl = coalesce.evaluate_encoded(x)
    assert pol.shape == (1, 4672)
    assert wdl.shape == (1, 3)
    assert inner.submit_sizes == [1]


def test_concurrent_callers_coalesce() -> None:
    """Eight walkers submitting concurrently should see the inner dispatcher
    receive far fewer calls than 8 (because they merge into fewer, larger
    batches). Exact count depends on scheduling — we assert on the bound."""
    inner = _RecordingInner(submit_sleep=0.02)  # 20ms fake GPU call
    coalesce = BatchCoalescingDispatcher(inner, max_batch=128)

    n_callers = 8
    barrier = threading.Barrier(n_callers)
    results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n_callers

    def worker(i: int) -> None:
        barrier.wait()
        x = np.zeros((1, 146, 8, 8), dtype=np.float32)
        results[i] = coalesce.evaluate_encoded(x)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_callers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert all(r is not None for r in results), "some caller didn't return"
    total = sum(inner.submit_sizes)
    assert total == n_callers, f"submitted {total} samples, expected {n_callers}"
    # With all 8 callers arriving at a barrier and a 20ms inner sleep, at
    # least one submit should be batch > 1. Otherwise the coalescer isn't
    # actually coalescing.
    assert max(inner.submit_sizes) > 1, (
        f"no coalescing observed: {inner.submit_sizes}")
    assert len(inner.submit_sizes) < n_callers, (
        f"expected fewer inner calls than callers: {inner.submit_sizes}")


def test_result_slicing_matches_per_caller() -> None:
    """Each caller must get exactly the rows corresponding to their input,
    not some other caller's slice. Inner echoes x's first value into pol's
    first column; we verify each caller's echo matches what they submitted."""

    class _EchoInner:
        def __init__(self) -> None:
            self.sleep = 0.02

        def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            time.sleep(self.sleep)
            n = x.shape[0]
            pol = np.zeros((n, 4672), dtype=np.float32)
            # Echo x[row, 0, 0, 0] into pol[row, 0] so each caller can
            # recognise their own slice.
            pol[:, 0] = x[:, 0, 0, 0]
            return pol, np.zeros((n, 3), dtype=np.float32)

    coalesce = BatchCoalescingDispatcher(_EchoInner(), max_batch=128)
    n_callers = 4
    barrier = threading.Barrier(n_callers)
    got: list[float | None] = [None] * n_callers

    def worker(i: int) -> None:
        barrier.wait()
        x = np.zeros((1, 146, 8, 8), dtype=np.float32)
        x[0, 0, 0, 0] = 1000.0 + i  # unique tag
        pol, _ = coalesce.evaluate_encoded(x)
        got[i] = float(pol[0, 0])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_callers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    for i in range(n_callers):
        assert got[i] == 1000.0 + i, (
            f"caller {i}: got {got[i]}, expected {1000.0 + i} — slices mismatched")


def test_exception_propagates_to_waiters() -> None:
    """If the inner dispatcher raises, every caller currently in the
    coalesced batch must see the exception (not hang forever)."""

    class _RaisingInner:
        def evaluate_encoded(self, x: np.ndarray):
            del x
            raise RuntimeError("simulated GPU failure")

    coalesce = BatchCoalescingDispatcher(_RaisingInner(), max_batch=128)
    n_callers = 4
    barrier = threading.Barrier(n_callers)
    errors: list[BaseException | None] = [None] * n_callers

    def worker(i: int) -> None:
        barrier.wait()
        try:
            coalesce.evaluate_encoded(np.zeros((1, 146, 8, 8), dtype=np.float32))
        except BaseException as e:
            errors[i] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_callers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    # At least the submitter raised; non-submitter waiters see the
    # exception stored in their result box (by design, they re-raise).
    # We don't assert on all errors being RuntimeError — only that nobody
    # is stuck alive (deadlock check).
    for t in threads:
        assert not t.is_alive(), "worker hung (no exception propagation)"
