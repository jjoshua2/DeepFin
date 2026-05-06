"""Single-process queue-fed GPU dispatcher.

Worker threads enqueue (encoded, Future) tuples; one dispatcher thread drains
the queue, runs a batched forward, and scatters results back. Only the
dispatcher thread touches CUDA, so torch.compile cudagraphs stay valid.
"""
from __future__ import annotations

import collections
import concurrent.futures
import logging
import threading
import time
from typing import Any, Literal, TypeGuard, cast

import numpy as np
import torch

from chess_anti_engine.inference import (
    DirectGPUEvaluator,
    _COMPILED_BATCH_BUCKETS,
)

logger = logging.getLogger(__name__)

EvalItem = tuple[np.ndarray, concurrent.futures.Future]
ModelUpdateItem = tuple[Literal["_update_model"], torch.nn.Module, threading.Event, dict[str, BaseException]]
QueueItem = EvalItem | ModelUpdateItem
DispatchHandle = tuple[list[EvalItem], int, Any, Any, Any]


def _next_bucket(n: int, buckets: tuple[int, ...] = _COMPILED_BATCH_BUCKETS) -> int:
    for b in buckets:
        if b >= n:
            return b
    return buckets[-1]


def _snapshot_dynamo_counters() -> dict[str, dict[str, int]] | None:
    """Snapshot torch._dynamo.utils.counters; returns None if unavailable."""
    try:
        from torch._dynamo.utils import counters
    except Exception:  # noqa: BLE001
        return None
    return {cat: dict(d) for cat, d in counters.items()}


def _delta_dynamo_counters(
    baseline: dict[str, dict[str, int]] | None,
) -> dict[str, int] | None:
    """Compute (now - baseline) for the cudagraph-relevant counters.

    Keys returned: frames_ok, recompiles, graph_breaks, cudagraph_skips,
    cudagraphify_called. ``cudagraph_skips`` > 0 means inductor compiled
    the graph but the cudagraph layer rejected it (dynamic shapes,
    cross-thread TLS miss, etc) and silently fell back to eager.
    """
    if baseline is None:
        return None
    try:
        from torch._dynamo.utils import counters
    except Exception:  # noqa: BLE001
        return None

    def _delta(cat: str, key: str) -> int:
        return max(0, int(counters.get(cat, {}).get(key, 0)) - int(baseline.get(cat, {}).get(key, 0)))

    graph_break_total = sum(
        _delta("graph_break", k) for k in counters.get("graph_break", {})
    )
    return {
        "frames_ok": _delta("frames", "ok"),
        "recompiles": _delta("recompiles", "recompile"),
        "graph_breaks": graph_break_total,
        "cudagraph_skips": _delta("inductor", "cudagraph_skips"),
        "cudagraphify_called": _delta("inductor", "cudagraphify_called"),
    }


class ThreadedDispatcher:
    """Queue-batched GPU dispatcher with a single CUDA-owning thread.

    If ``compile_mode`` is set, the dispatcher thread does ``torch.compile`` on
    the model itself before its first forward. This is load-bearing for
    cudagraphs: torch.compile + cudagraph_trees use thread-local CUDA state, so
    the dynamo trace + cudagraph capture must happen on the same thread that
    replays. Compiling on the caller's thread and dispatching to a worker
    thread trips `assert torch._C._is_key_in_tls(...)` on the first forward.

    Compile is lazy — it runs on the dispatcher thread when the first user
    request arrives. The constructor returns immediately so worker startup
    is not blocked by a multi-minute autotune pass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str = "cuda",
        max_batch: int = 4096,
        batch_wait_ms: float = 1.0,
        compile_mode: str | None = None,
    ) -> None:
  # n_slots=2 is the heart of the async pipeline: while the GPU runs
  # forward K, the dispatcher fills slot 1-K and submits forward K+1.
  # Without this the CPU descend/scatter and GPU forward serialize and
  # the dispatcher's effective throughput drops by ~50%.
        self._evaluator = DirectGPUEvaluator(
            model,
            device=device,
            max_batch=max_batch,
            n_slots=2,
        )
        self._max_batch = int(max_batch)
        self._batch_wait_s = float(batch_wait_ms) / 1000.0
        self._compile_mode = compile_mode
        self._device = device

        self._queue: collections.deque[QueueItem] = collections.deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = False

        self._lifetime_batches = 0
        self._lifetime_positions = 0
        self._lifetime_full_drains = 0
        self._lifetime_forward_s = 0.0

  # Dynamo/cudagraph counters: snapshotted on the dispatcher thread before
  # first compile, then re-read after warmup so the delta tells us whether
  # cudagraphs actually fired (frames_ok > 0, cudagraph_skips == 0) or
  # silently fell back to eager. See compile_probe.py for the same idea.
        self._compile_baseline: dict[str, dict[str, int]] | None = None
        self._compile_probe_logged = False
        self._compile_probe_after_batches = 50

        self._thread = threading.Thread(
            target=self._dispatch_loop,
            name="ThreadedDispatcher",
            daemon=True,
        )
        self._thread.start()

    def evaluate(self, x: np.ndarray) -> concurrent.futures.Future:
        if x.shape[0] > self._max_batch:
            raise ValueError(
                f"single submission of {x.shape[0]} > max_batch={self._max_batch}; "
                "callers must split before submission"
            )
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._cond:
            self._queue.append((x, future))
            self._cond.notify()
        return future

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.evaluate(x).result()

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=timeout)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "lifetime_batches": self._lifetime_batches,
            "lifetime_positions": self._lifetime_positions,
            "lifetime_full_drains": self._lifetime_full_drains,
            "lifetime_forward_s": self._lifetime_forward_s,
            "avg_batch_size": (
                self._lifetime_positions / self._lifetime_batches
                if self._lifetime_batches > 0 else 0.0
            ),
            "avg_forward_ms": (
                1000.0 * self._lifetime_forward_s / self._lifetime_batches
                if self._lifetime_batches > 0 else 0.0
            ),
        }

    def update_model(self, model: torch.nn.Module) -> None:
        if not self._thread.is_alive():
            raise RuntimeError("dispatcher thread died")
        done = threading.Event()
        result: dict[str, BaseException] = {}
        with self._cond:
            self._queue.append(("_update_model", model, done, result))
            self._cond.notify()
        while not done.wait(timeout=5.0):
            if not self._thread.is_alive():
                raise RuntimeError("dispatcher thread died during model update")
        if "error" in result:
            raise RuntimeError("dispatcher model update failed") from result["error"]

    @staticmethod
    def _is_model_update(item: QueueItem) -> TypeGuard[ModelUpdateItem]:
        return len(item) == 4 and item[0] == "_update_model"

    def _drain_batch(
        self, *, blocking: bool = True,
    ) -> list[QueueItem]:
        """Pop up to max_batch leaves from the queue.

        ``blocking=True`` waits indefinitely for the first item to arrive
        (used at idle, when no pipeline submit is in flight). ``blocking=
        False`` returns immediately with whatever's already queued — used
        when ``pending`` is alive: blocking would deadlock the pipeline
        because the dispatcher couldn't return to scatter the in-flight
        forward whose future the producer is awaiting.
        """
        with self._cond:
            if blocking:
                while not self._queue and not self._stop:
                    self._cond.wait(timeout=0.01)
                if self._stop and not self._queue:
                    return []
                if self._batch_wait_s > 0 and self._queue:
                    self._cond.wait(timeout=self._batch_wait_s)
            elif not self._queue:
                return []

            first = self._queue[0]
            if self._is_model_update(first):
                return [self._queue.popleft()]

            items: list[EvalItem] = []
            total = 0
            while self._queue:
                item = self._queue[0]
                if self._is_model_update(item):
                    break
                eval_item = cast(EvalItem, item)
                enc = eval_item[0]
                if total + enc.shape[0] > self._max_batch:
                    break
                items.append(cast(EvalItem, self._queue.popleft()))
                total += enc.shape[0]
            return cast(list[QueueItem], items)

    def _compile_model_on_dispatcher(self, model: torch.nn.Module) -> torch.nn.Module:
        model.eval()
        if self._compile_mode is None:
            return model
        # set_device wants an index or "cuda:N", not bare "cuda".
        dev = torch.device(self._device)
        if dev.type == "cuda":
            torch.cuda.set_device(dev.index or 0)
        self._compile_baseline = _snapshot_dynamo_counters()
        self._compile_probe_logged = False
        self._compile_probe_after_batches = self._lifetime_batches + 50
        return cast(torch.nn.Module, torch.compile(
            model, mode=self._compile_mode,
        ))

    def _apply_model_update(self, item: ModelUpdateItem) -> None:
        _, model, done, result = item
        try:
            if self._compile_mode is not None:
                load_target = getattr(self._evaluator.model, "_orig_mod", self._evaluator.model)
                load_target.load_state_dict(model.state_dict(), strict=True)
                load_target.eval()
            else:
                self._evaluator.model = model
            self._evaluator.model.eval()
        except BaseException:  # noqa: BLE001
            try:
                self._evaluator.model = self._compile_model_on_dispatcher(model)
                self._evaluator.model.eval()
            except BaseException as fallback_exc:  # noqa: BLE001
                result["error"] = fallback_exc
        finally:
            done.set()

    def _dispatch_loop(self) -> None:
        self._evaluator.model = self._compile_model_on_dispatcher(self._evaluator.model)

  # Two-slot async pipeline. While the GPU runs forward K (slot s),
  # the dispatcher drains the queue and writes batch K+1 into slot
  # 1-s's pinned input. Forward K+1 fires before K finishes, so CPU
  # batch-build / scatter and GPU compute fully overlap. Mirrors
  # PucvChunker's pattern from chess_anti_engine/mcts/puct_vl.py.
        ev = self._evaluator
        slot = 0
        pending: DispatchHandle | None = None  # (items, real_n, pol_t, wdl_t, evt)

        while not self._stop or self._queue or pending is not None:
  # Block (cond-wait) only when the pipeline is idle. With pending
  # in flight, the dispatcher MUST return promptly so we can scatter
  # the result whose future the producer is awaiting; otherwise the
  # producer's evaluate_encoded() call deadlocks.
            blocking = pending is None
            items = self._drain_batch(blocking=blocking)

            next_handle: DispatchHandle | None = None
            model_update = items[0] if len(items) == 1 and self._is_model_update(items[0]) else None
            if items and model_update is None:
                eval_items = cast(list[EvalItem], items)
                try:
                    next_handle = self._submit_batch(eval_items, slot, ev)
                    slot = 1 - slot
                except Exception as exc:  # noqa: BLE001
                    logger.exception("dispatcher submit failed: %s", exc)
                    for _, fut in eval_items:
                        if not fut.done():
                            fut.set_exception(exc)

  # One-shot cudagraph probe. After the warmup window (50 batches) has
  # passed compile + capture, the dynamo counters' delta tells us
  # whether cudagraphs are actually firing on this thread. The
  # alternative is silent eager fallback, in which case the
  # ``set_device`` workaround claim is folklore.
            if (
                not self._compile_probe_logged
                and self._compile_baseline is not None
                and self._lifetime_batches >= self._compile_probe_after_batches
            ):
                d = _delta_dynamo_counters(self._compile_baseline)
                if d is not None:
                    logger.info(
                        "dispatcher cudagraph probe: frames_ok=%d cudagraphify_called=%d "
                        "cudagraph_skips=%d recompiles=%d graph_breaks=%d",
                        d["frames_ok"], d["cudagraphify_called"],
                        d["cudagraph_skips"], d["recompiles"], d["graph_breaks"],
                    )
                self._compile_probe_logged = True

            if pending is not None:
                try:
                    self._scatter_pending(pending)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("dispatcher scatter failed: %s", exc)
                    p_items = pending[0]
                    for _, fut in p_items:
                        if not fut.done():
                            fut.set_exception(exc)

            if model_update is not None:
                self._apply_model_update(model_update)

            pending = next_handle

            if self._stop and not self._queue and pending is None:
                return

    def _submit_batch(
        self, items: list[EvalItem],
        slot: int, ev: DirectGPUEvaluator,
    ) -> DispatchHandle:
        """Pack items into slot's pinned input and fire async forward.

        Returns (items, real_n, pol_t, wdl_t, evt). Caller waits on evt
        before reading pol_t/wdl_t and scattering to futures.
        """
        real_n = sum(enc.shape[0] for enc, _ in items)
        bucket = _next_bucket(real_n)
        inp: Any = ev.get_input_buffer(bucket, slot=slot)
  # get_input_buffer returns the pinned tensor; convert to numpy view
  # to write directly via numpy slice copy (matches gumbel_c.py pattern).
        inp_np = inp.numpy() if hasattr(inp, "numpy") else inp
        offset = 0
        for enc, _ in items:
            n = enc.shape[0]
            inp_np[offset : offset + n] = enc
            offset += n
  # Bucket padding zone is left untouched — model sees stale data in
  # the padded slots, but caller only reads results[:real_n]. This
  # matches the pre-refactor behavior (pad with zeros) functionally;
  # leaving it untouched saves a memset.
        t0 = time.perf_counter()
        pol_t, wdl_t, evt = ev.evaluate_inplace_async(bucket, slot=slot)
        self._lifetime_forward_s += time.perf_counter() - t0
        self._lifetime_batches += 1
        self._lifetime_positions += real_n
        if real_n >= self._max_batch:
            self._lifetime_full_drains += 1
        return (items, real_n, pol_t, wdl_t, evt)

    def _scatter_pending(self, pending: DispatchHandle) -> None:
        items, pol_t, wdl_t, evt = pending[0], pending[2], pending[3], pending[4]
        if evt is not None:
            evt.synchronize()
        pol_any: Any = pol_t
        wdl_any: Any = wdl_t
        pol = pol_any.numpy() if hasattr(pol_any, "numpy") else np.asarray(pol_any)
        wdl = wdl_any.numpy() if hasattr(wdl_any, "numpy") else np.asarray(wdl_any)
  # CRITICAL: pol/wdl are views of the slot's pinned output. The next
  # submit to this same slot (one ping-pong away) overwrites these
  # views in-place. Callers might read the future result asynchronously
  # — sometimes after the next-next submit lands. So copy each slice
  # into caller-owned numpy memory before set_result. Same constraint
  # as gumbel_c.py's clone-before-resubmit comment.
        offset = 0
        for enc, fut in items:
            n = enc.shape[0]
            fut.set_result((
                pol[offset : offset + n].copy(),
                wdl[offset : offset + n].copy(),
            ))
            offset += n
