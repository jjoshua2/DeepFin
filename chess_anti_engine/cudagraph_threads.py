"""CUDA-graph captures on a thread other than the one that warmed the shape.

Inductor's CUDA graphs (``torch.compile`` modes ``reduce-overhead`` /
``max-autotune``) are keyed per *compiled graph*, not per caller. Every
``torch.compile`` of the same module class on the same device with the same
mode reuses one dynamo cache entry, and that entry's ``cudagraphify`` closure
binds each batch shape to the cudagraph tree manager of whichever thread ran
the shape FIRST (``cudagraph_trees.cudagraphify_impl.fn_cache``). A later
evaluator on another thread -- a rebuilt UCI evaluator's new owner thread, a
second pool worker or routing-chain device on the same GPU, a ``SearchWalkers``
change that swaps the owner dispatcher for the coalescer -- therefore replays
and RECORDS through the first thread's manager. Measured on torch 2.10 / RTX
5090 (real 61M net and a two-layer toy model alike):

1. The first cuBLAS call of a thread allocates its handle (``cublasCreate``).
   A thread that records a graph warmed up by another (still running) thread
   makes that call inside the stream capture, which fails with
   ``CUBLAS_STATUS_NOT_INITIALIZED`` and invalidates the capture
   (``cudaErrorStreamCaptureInvalidated``). :func:`prepare_thread_for_capture`
   makes the thread's cuBLAS / cuBLASLt calls once, outside any capture,
   before its first compiled forward.
2. ``CUDAGraph.capture_end`` raises before it ends the caching allocator's
   "allocate to pool" scope, so that manager's memory pool stays marked as
   capturing and every later warm-up or record through it -- the PREVIOUS
   evaluator's included -- fails with ``beginAllocateToPool: already recording
   to mempool_id``. :func:`release_failed_capture` finds the failed capture's
   pool in the exception's traceback and ends that scope, so the manager (and
   every evaluator replaying through it) keeps working.

The managers themselves stay per thread, as Inductor keeps them. (Pointing
every thread at one process-wide dict was tried: it extended the managers'
lifetime past their threads and the process segfaulted at ``quit`` on the
routing chain, 2 of 2 runs, 0 of 1 without it.) What this cannot make safe is
CONCURRENT use of one manager from two threads (two evaluators on the same GPU
replaying at once share its memory pool): a device listed twice gets a
no-graph compile mode for its repeats (UCI ``__main__``).
"""
from __future__ import annotations

import contextlib
import gc
import logging
import threading
from collections.abc import Generator
from typing import Any

_log = logging.getLogger(__name__)

# Per thread: device indices whose cuBLAS/cuBLASLt handles this thread created.
_tls = threading.local()
# Every successful ``release_failed_capture`` (device index), for tests/diagnostics.
released_captures: list[int] = []
# Set by a release: the failed capture's tensors (allocated in the graph pool)
# are still referenced by the exception's traceback, usually through a
# reference cycle, when the scope is ended. They must be gone before the next
# record through that manager, which otherwise refuses with "live storage data
# ptrs are in the cudagraph pool but not accounted for": the next compiled
# forward collects them first.
_collect_before_next_forward = threading.Event()


def _cuda_index(device: Any) -> int | None:
    import torch

    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    index: int | None = getattr(dev, "index", None)
    return int(torch.cuda.current_device()) if index is None else int(index)


def prepare_thread_for_capture(device: Any) -> None:
    """Create this thread's cuBLAS / cuBLASLt handles on ``device`` (once per thread).

    Runs a few tiny matmuls (the ops the compiled net lowers to: ``mm``,
    ``addmm`` with a bias, ``bmm``; fp32/bf16/fp16) on the current stream and
    waits for them, so no handle is first created inside a graph capture.
    """
    idx = _cuda_index(device)
    if idx is None:
        return
    ready = getattr(_tls, "ready", None)
    if ready is None:
        ready = set()
        _tls.ready = ready
    if idx in ready:
        return
    import torch

    with torch.no_grad(), torch.cuda.device(idx):
        base = torch.ones((16, 16), device=f"cuda:{idx}", dtype=torch.float32)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            m = base.to(dtype)
            torch.mm(m, m)
            torch.addmm(m[0], m, m)
            torch.bmm(m.unsqueeze(0), m.unsqueeze(0))
        torch.cuda.current_stream(idx).synchronize()
    ready.add(idx)


_CUDAGRAPH_TREES_MODULE = "torch._inductor.cudagraph_trees"


def _capture_pools(exc: BaseException) -> list[tuple[int, Any]]:
    """(device index, mempool id) of every cudagraph tree manager / node on the
    stack of ``exc`` or the exceptions it chains (``__cause__``/``__context__``).

    Only objects of ``torch._inductor.cudagraph_trees`` classes are read: a
    ``CUDAGraphTreeManager`` (``device_index``, ``cuda_graphs_thread_pool``)
    or a ``CUDAGraphNode`` (``device``, ``cuda_graphs_pool``).
    """
    found: list[tuple[int, Any]] = []
    seen_exc: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        e = pending.pop()
        if e is None or id(e) in seen_exc:
            continue
        seen_exc.add(id(e))
        tb = e.__traceback__
        while tb is not None:
            obj = tb.tb_frame.f_locals.get("self")
            if type(obj).__module__ == _CUDAGRAPH_TREES_MODULE:
                pool = getattr(obj, "cuda_graphs_thread_pool", None)
                if pool is None:
                    pool = getattr(obj, "cuda_graphs_pool", None)
                dev = getattr(obj, "device_index", None)
                if dev is None:
                    dev = getattr(obj, "device", None)
                if pool is not None and isinstance(dev, int) and (dev, pool) not in found:
                    found.append((dev, pool))
            tb = tb.tb_next
        pending.append(e.__cause__)
        pending.append(e.__context__)
    return found


def release_failed_capture(exc: BaseException) -> bool:
    """End the allocator scope a failed graph capture left open.

    ``exc`` is the exception that escaped a compiled forward. True when a pool
    on its stack was still marked as capturing (a capture died inside
    ``capture_end``) and is now usable again; False when nothing was left open
    (the usual case: the failure was not a capture, or its scope was closed).
    """
    pools = _capture_pools(exc)
    if not pools:
        return False
    import torch

    end = getattr(torch._C, "_cuda_endAllocateToPool", None)
    if end is None:
        return False
    released = False
    for idx, pool in pools:
        try:
            end(idx, pool)
        except RuntimeError:
            continue  # "not currently recording": nothing was left open
        released = True
        _collect_before_next_forward.set()
        released_captures.append(idx)
        _log.warning(
            "cuda:%d: a CUDA-graph capture failed and left its memory pool marked as "
            "capturing; released it so the device's compiled graphs keep working", idx,
        )
    return released


@contextlib.contextmanager
def compiled_forward_scope(device: Any) -> Generator[None]:
    """Wrap one compiled forward: handles first, failed-capture cleanup after."""
    if _collect_before_next_forward.is_set():
        _collect_before_next_forward.clear()
        gc.collect()
    prepare_thread_for_capture(device)
    try:
        yield
    except BaseException as exc:
        release_failed_capture(exc)
        raise
