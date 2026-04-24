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


def test_close_rejects_new_submits() -> None:
    """After close(), evaluate_encoded raises immediately rather than
    hanging on an exited submitter."""
    inner = _RecordingInner(submit_sleep=0.0)
    coalesce = BatchCoalescingDispatcher(inner, max_batch=128)
    coalesce.close()
    import pytest
    with pytest.raises(RuntimeError, match="closed"):
        coalesce.evaluate_encoded(np.zeros((1, 146, 8, 8), dtype=np.float32))


def test_close_unblocks_pending_waiters() -> None:
    """If close() fires while a caller has work pending (the race that
    previously stranded the caller on done.wait() forever), the waiter
    must be released with a shutdown error instead."""

    class _SlowInner:
        """Blocks forever on first call — simulates a submitter stuck in
        the middle of a slow forward when close() is invoked."""
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def evaluate_encoded(self, x):
            del x
            self.started.set()
            # Block until test allows — by the time we unblock, close()
            # should have run. Return a valid result for the first caller
            # (which was in-flight when close() ran) so we exercise the
            # post-forward path, not the error path.
            self.release.wait(timeout=5.0)
            return (np.zeros((1, 4672), dtype=np.float32),
                    np.zeros((1, 3), dtype=np.float32))

    inner = _SlowInner()
    coalesce = BatchCoalescingDispatcher(inner, max_batch=128)

    # Caller A: kicks off the submit, will block on inner.
    errors_a: list[BaseException] = []

    def caller_a() -> None:
        try:
            coalesce.evaluate_encoded(np.zeros((1, 146, 8, 8), dtype=np.float32))
        except BaseException as e:
            errors_a.append(e)

    ta = threading.Thread(target=caller_a)
    ta.start()
    assert inner.started.wait(timeout=2.0), "submitter never started first forward"

    # Caller B: arrives while A's forward is running — its item sits in
    # pending. This is the race that used to strand: if close() fires now,
    # B's done.wait() should return with an error, not hang.
    errors_b: list[BaseException] = []

    def caller_b() -> None:
        try:
            coalesce.evaluate_encoded(np.zeros((1, 146, 8, 8), dtype=np.float32))
        except BaseException as e:
            errors_b.append(e)

    tb = threading.Thread(target=caller_b)
    tb.start()
    time.sleep(0.05)  # give B time to enqueue

    # Fire close() while A is mid-forward and B is pending.
    inner.release.set()  # let A's forward complete (A returns normally)
    coalesce.close()

    ta.join(timeout=5.0)
    tb.join(timeout=5.0)
    assert not ta.is_alive(), "caller A hung across close"
    assert not tb.is_alive(), "caller B hung across close — the close-race bug"
    # A completed normally (its forward returned before close drained).
    # B was stranded in pending when close ran; must see RuntimeError.
    assert len(errors_b) == 1 and isinstance(errors_b[0], RuntimeError), (
        f"caller B should get RuntimeError from close(), got {errors_b}")


def test_close_is_idempotent() -> None:
    """Calling close() a second time is a no-op, not a deadlock."""
    coalesce = BatchCoalescingDispatcher(_RecordingInner(), max_batch=128)
    coalesce.close()
    coalesce.close()  # must not hang or raise


def test_row_aware_packing_never_exceeds_max_batch() -> None:
    """Walker-pool scenario (Codex adversarial finding): each caller submits
    leaf_gather > 1 rows. Coalescer must pack by cumulative rows, not by
    request count, or it concatenates two legal requests into an oversize
    tensor the inner evaluator rejects."""
    inner = _RecordingInner(submit_sleep=0.02)
    max_batch = 64
    coalesce = BatchCoalescingDispatcher(inner, max_batch=max_batch)

    # Two concurrent callers each sending leaf_gather=64 rows. Under the
    # old slice-by-request behavior, both got merged into a 128-row submit
    # and raised ValueError from the inner evaluator.
    n_callers = 2
    gather = 64
    barrier = threading.Barrier(n_callers)
    results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n_callers

    def worker(i: int) -> None:
        barrier.wait()
        x = np.zeros((gather, 146, 8, 8), dtype=np.float32)
        results[i] = coalesce.evaluate_encoded(x)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_callers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert all(r is not None for r in results)
    assert all(size <= max_batch for size in inner.submit_sizes), (
        f"submit exceeded max_batch: {inner.submit_sizes}")
    assert sum(inner.submit_sizes) == n_callers * gather


def test_rejects_single_request_larger_than_max_batch() -> None:
    """A single caller with x.shape[0] > max_batch can never be dispatched.
    Reject at entry so the error carries the right traceback instead of
    surfacing from the submitter thread."""
    coalesce = BatchCoalescingDispatcher(_RecordingInner(submit_sleep=0.0), max_batch=64)
    import pytest
    with pytest.raises(ValueError, match="coalescer max 64"):
        coalesce.evaluate_encoded(np.zeros((128, 146, 8, 8), dtype=np.float32))


def test_cumulative_packing_splits_across_rounds() -> None:
    """Requests of heterogeneous sizes must be packed cumulatively up to
    max_batch. FIFO order preserved (no reordering that could starve
    large requests)."""
    inner = _RecordingInner(submit_sleep=0.05)
    coalesce = BatchCoalescingDispatcher(inner, max_batch=64)

    # Sizes 40, 30, 20: first round fits 40 only (40+30=70 > 64);
    # second round fits 30+20=50; total two submits.
    sizes = [40, 30, 20]
    barrier = threading.Barrier(len(sizes))
    results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(sizes)

    def worker(i: int, n: int) -> None:
        barrier.wait()
        # Stagger submission slightly so FIFO order in pending reflects
        # sizes[] order (threads arriving out of order would give a
        # different packing; we assert the correctness invariant, not a
        # specific pack pattern, for robustness against scheduler jitter).
        time.sleep(0.001 * i)
        x = np.zeros((n, 146, 8, 8), dtype=np.float32)
        results[i] = coalesce.evaluate_encoded(x)

    threads = [threading.Thread(target=worker, args=(i, n)) for i, n in enumerate(sizes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert all(r is not None for r in results)
    assert all(size <= 64 for size in inner.submit_sizes), (
        f"submit exceeded max_batch: {inner.submit_sizes}")
    assert sum(inner.submit_sizes) == sum(sizes)
