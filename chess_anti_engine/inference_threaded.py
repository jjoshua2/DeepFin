"""Single-process queue-fed GPU dispatcher.

Worker threads enqueue evaluation requests; one dispatcher thread drains the
queue, runs a batched forward, and scatters results back. Only the
dispatcher thread touches CUDA, so torch.compile cudagraphs stay valid.
"""
from __future__ import annotations

import collections
import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from chess_anti_engine.inference import (
    DirectGPUEvaluator,
    _COMPILED_BATCH_BUCKETS,
    _bf16_bits_to_float32_np,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _EvalRequest:
    encoded: np.ndarray
    future: concurrent.futures.Future
    legal_flat: np.ndarray | None = None
    legal_counts: np.ndarray | None = None


@dataclass(slots=True)
class _ModelUpdateRequest:
    model: torch.nn.Module
    done: threading.Event
    result: dict[str, BaseException]


@dataclass(slots=True)
class _DispatchHandle:
    items: list[_EvalRequest]
    real_n: int
    real_pol_n: int
    compact: bool
    pol_t: Any
    wdl_t: Any
    event: Any


QueueItem = _EvalRequest | _ModelUpdateRequest


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
        target_batch: int | None = None,
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
        self._target_batch = (
            self._max_batch
            if target_batch is None or int(target_batch) <= 0
            else min(int(target_batch), self._max_batch)
        )
        self._batch_wait_s = float(batch_wait_ms) / 1000.0
        self._compile_mode = compile_mode
        self._device = device

        self._queue: collections.deque[QueueItem] = collections.deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = False

        self._lifetime_batches = 0
        self._lifetime_positions = 0
        self._lifetime_forward_rows = 0
        self._lifetime_requests = 0
        self._lifetime_legal_bf16_batches = 0
        self._lifetime_legal_bf16_positions = 0
        self._lifetime_full_drains = 0
        self._lifetime_drain_s = 0.0
        self._lifetime_pack_s = 0.0
        self._lifetime_submit_s = 0.0
        self._lifetime_wait_s = 0.0
        self._lifetime_scatter_s = 0.0
        self._lifetime_forward_s = 0.0
        self._lifetime_queue_depth_samples = 0
        self._lifetime_queue_depth_sum = 0
        self._lifetime_queue_depth_max = 0

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
            self._queue.append(_EvalRequest(x, future))
            self._cond.notify()
        return future

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.evaluate(x).result()

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if x.shape[0] > self._max_batch:
            raise ValueError(
                f"single submission of {x.shape[0]} > max_batch={self._max_batch}; "
                "callers must split before submission"
            )
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._cond:
            self._queue.append(_EvalRequest(
                x, future,
                np.asarray(legal_flat, dtype=np.int32),
                np.asarray(legal_counts, dtype=np.int32),
            ))
            self._cond.notify()
        return future.result()

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
            "lifetime_forward_rows": self._lifetime_forward_rows,
            "lifetime_requests": self._lifetime_requests,
            "lifetime_legal_bf16_batches": self._lifetime_legal_bf16_batches,
            "lifetime_legal_bf16_positions": self._lifetime_legal_bf16_positions,
            "lifetime_full_drains": self._lifetime_full_drains,
            "max_batch": self._max_batch,
            "target_batch": self._target_batch,
            "lifetime_drain_s": self._lifetime_drain_s,
            "lifetime_pack_s": self._lifetime_pack_s,
            "lifetime_submit_s": self._lifetime_submit_s,
            "lifetime_wait_s": self._lifetime_wait_s,
            "lifetime_scatter_s": self._lifetime_scatter_s,
            "lifetime_forward_s": self._lifetime_forward_s,
            "queue_depth_max": self._lifetime_queue_depth_max,
            "avg_batch_size": (
                self._lifetime_positions / self._lifetime_batches
                if self._lifetime_batches > 0 else 0.0
            ),
            "avg_requests_per_batch": (
                self._lifetime_requests / self._lifetime_batches
                if self._lifetime_batches > 0 else 0.0
            ),
            "avg_rows_per_request": (
                self._lifetime_positions / self._lifetime_requests
                if self._lifetime_requests > 0 else 0.0
            ),
            "avg_queue_depth": (
                self._lifetime_queue_depth_sum / self._lifetime_queue_depth_samples
                if self._lifetime_queue_depth_samples > 0 else 0.0
            ),
            "avg_drain_ms": (
                1000.0 * self._lifetime_drain_s / self._lifetime_batches
                if self._lifetime_batches > 0 else 0.0
            ),
            "avg_pack_ms": (
                1000.0 * self._lifetime_pack_s / self._lifetime_batches
                if self._lifetime_batches > 0 else 0.0
            ),
            "avg_submit_ms": (
                1000.0 * self._lifetime_submit_s / self._lifetime_batches
                if self._lifetime_batches > 0 else 0.0
            ),
            "avg_wait_ms": (
                1000.0 * self._lifetime_wait_s / self._lifetime_batches
                if self._lifetime_batches > 0 else 0.0
            ),
            "avg_scatter_ms": (
                1000.0 * self._lifetime_scatter_s / self._lifetime_batches
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
            self._queue.append(_ModelUpdateRequest(model, done, result))
            self._cond.notify()
        while not done.wait(timeout=5.0):
            if not self._thread.is_alive():
                raise RuntimeError("dispatcher thread died during model update")
        if "error" in result:
            raise RuntimeError("dispatcher model update failed") from result["error"]

    @staticmethod
    def _event_ready(event: Any) -> bool:
        if event is None:
            return True
        query = getattr(event, "query", None)
        if query is None:
            return False
        try:
            return bool(query())
        except Exception:  # noqa: BLE001
            return False

    def _drain_batch(
        self, *, blocking: bool = True, coalesce_wait: bool = False,
    ) -> list[QueueItem]:
        """Pop up to max_batch leaves from the queue.

        ``blocking=True`` waits indefinitely for the first item to arrive
        (used at idle, when no pipeline submit is in flight). ``blocking=
        False`` normally returns promptly with whatever's already queued.
        When an in-flight GPU forward is known not to be ready yet,
        ``coalesce_wait`` can wait up to ``batch_wait_ms`` to gather a
        larger next batch without delaying scatter.
        """
        t0 = time.perf_counter()
        with self._cond:
            if blocking:
                while not self._queue and not self._stop:
                    self._cond.wait(timeout=0.01)
                if self._stop and not self._queue:
                    return []
                if self._batch_wait_s > 0 and self._queue:
                    self._cond.wait(timeout=self._batch_wait_s)
            else:
                if coalesce_wait and self._batch_wait_s > 0:
                    self._cond.wait(timeout=self._batch_wait_s)
                if not self._queue:
                    return []

            first = self._queue[0]
            if isinstance(first, _ModelUpdateRequest):
                self._lifetime_drain_s += time.perf_counter() - t0
                return [self._queue.popleft()]

            queue_depth = len(self._queue)
            self._lifetime_queue_depth_samples += 1
            self._lifetime_queue_depth_sum += queue_depth
            self._lifetime_queue_depth_max = max(self._lifetime_queue_depth_max, queue_depth)
            items: list[QueueItem] = []
            total = 0
            first_compact = (
                first.legal_flat is not None
                and first.legal_counts is not None
            )
            first_dtype = first.encoded.dtype
            while self._queue:
                item = self._queue[0]
                if isinstance(item, _ModelUpdateRequest):
                    break
                item_compact = item.legal_flat is not None and item.legal_counts is not None
                if item_compact != first_compact:
                    break
                if item.encoded.dtype != first_dtype:
                    break
                n_rows = int(item.encoded.shape[0])
                if total + n_rows > self._target_batch and items:
                    break
                if n_rows > self._max_batch:
                    raise ValueError(
                        f"queued item has {n_rows} rows > max_batch={self._max_batch}"
                    )
                items.append(self._queue.popleft())
                total += n_rows
            self._lifetime_drain_s += time.perf_counter() - t0
            return items

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

    def _apply_model_update(self, item: _ModelUpdateRequest) -> None:
        try:
            if self._compile_mode is not None:
                load_target = getattr(self._evaluator.model, "_orig_mod", self._evaluator.model)
                load_target.load_state_dict(item.model.state_dict(), strict=True)
                load_target.eval()
            else:
                self._evaluator.model = item.model
            self._evaluator.model.eval()
        except BaseException:  # noqa: BLE001
            try:
                self._evaluator.model = self._compile_model_on_dispatcher(item.model)
                self._evaluator.model.eval()
            except BaseException as fallback_exc:  # noqa: BLE001
                item.result["error"] = fallback_exc
        finally:
            item.done.set()

    def _dispatch_loop(self) -> None:
        self._evaluator.model = self._compile_model_on_dispatcher(self._evaluator.model)

  # Two-slot async pipeline. While the GPU runs forward K (slot s),
  # the dispatcher drains the queue and writes batch K+1 into slot
  # 1-s's pinned input. Forward K+1 fires before K finishes, so CPU
  # batch-build / scatter and GPU compute fully overlap. Mirrors
  # PucvChunker's pattern from chess_anti_engine/mcts/puct_vl.py.
        ev = self._evaluator
        slot = 0
        pending: _DispatchHandle | None = None

        while not self._stop or self._queue or pending is not None:
  # Block (cond-wait) only when the pipeline is idle. With pending
  # in flight, the dispatcher MUST return promptly so we can scatter
  # the result whose future the producer is awaiting; otherwise the
  # producer's evaluate_encoded() call deadlocks.
            blocking = pending is None
            coalesce_wait = (
                pending is not None
                and not self._event_ready(pending.event)
            )
            items = self._drain_batch(
                blocking=blocking,
                coalesce_wait=coalesce_wait,
            )

            next_handle: _DispatchHandle | None = None
            model_update = items[0] if len(items) == 1 and isinstance(items[0], _ModelUpdateRequest) else None
            if items and model_update is None:
                eval_items = [item for item in items if isinstance(item, _EvalRequest)]
                try:
                    next_handle = self._submit_batch(eval_items, slot, ev)
                    slot = 1 - slot
                except Exception as exc:  # noqa: BLE001
                    logger.exception("dispatcher submit failed: %s", exc)
                    for item in eval_items:
                        if not item.future.done():
                            item.future.set_exception(exc)

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
                    for item in pending.items:
                        if not item.future.done():
                            item.future.set_exception(exc)

            if model_update is not None:
                self._apply_model_update(model_update)

            pending = next_handle

            if self._stop and not self._queue and pending is None:
                return

    @property
    def supports_input_bf16_bits(self) -> bool:
        return bool(getattr(self._evaluator, "supports_input_bf16_bits", False))

    def _submit_batch(
        self, items: list[_EvalRequest],
        slot: int, ev: DirectGPUEvaluator,
    ) -> _DispatchHandle:
        """Pack items into slot's pinned input and fire async forward.

        Caller waits on the returned event before reading pol_t/wdl_t and
        scattering to futures.
        """
        real_n = sum(item.encoded.shape[0] for item in items)
        bucket = _next_bucket(real_n)
        compact = items[0].legal_flat is not None and items[0].legal_counts is not None
        input_bf16 = items[0].encoded.dtype == np.uint16 and ev.supports_input_bf16_bits
        real_pol_n = real_n
        t_pack0 = time.perf_counter()
        if input_bf16:
            inp = ev.get_input_buffer_bf16_bits(bucket, slot=slot)
        else:
            inp = ev.get_input_buffer(bucket, slot=slot)
  # get_input_buffer returns the pinned tensor; convert to numpy view
  # to write directly via numpy slice copy (matches gumbel_c.py pattern).
        inp_np = inp
        offset = 0
        for item in items:
            n = item.encoded.shape[0]
            if item.encoded.dtype == np.uint16 and not input_bf16:
                inp_np[offset : offset + n] = _bf16_bits_to_float32_np(item.encoded)
            else:
                inp_np[offset : offset + n] = item.encoded
            offset += n
  # Bucket padding zone is left untouched — model sees stale data in
  # the padded slots, but caller only reads results[:real_n]. This
  # matches the pre-refactor behavior (pad with zeros) functionally;
  # leaving it untouched saves a memset.
        self._lifetime_pack_s += time.perf_counter() - t_pack0
        t0 = time.perf_counter()
        if compact:
            legal_flat = np.concatenate([
                np.asarray(item.legal_flat, dtype=np.int32) for item in items
            ])
            legal_counts = np.concatenate([
                np.asarray(item.legal_counts, dtype=np.int32) for item in items
            ])
            real_pol_n = int(legal_counts.sum())
            if bucket > real_n:
                legal_counts = np.pad(legal_counts, (0, bucket - real_n))
            pol_t, wdl_t, evt = ev.evaluate_inplace_legal_bf16_async(
                bucket, legal_flat, legal_counts, slot=slot,
            )
        else:
            pol_t, wdl_t, evt = ev.evaluate_inplace_async(bucket, slot=slot)
        self._lifetime_submit_s += time.perf_counter() - t0
        self._lifetime_batches += 1
        self._lifetime_positions += real_n
        self._lifetime_forward_rows += bucket
        self._lifetime_requests += len(items)
        if compact:
            self._lifetime_legal_bf16_batches += 1
            self._lifetime_legal_bf16_positions += real_n
        if real_n >= self._max_batch:
            self._lifetime_full_drains += 1
        return _DispatchHandle(items, real_n, real_pol_n, compact, pol_t, wdl_t, evt)

    def _scatter_pending(self, pending: _DispatchHandle) -> None:
        if pending.event is not None:
            t0 = time.perf_counter()
            pending.event.synchronize()
            self._lifetime_wait_s += time.perf_counter() - t0
        t_scatter0 = time.perf_counter()
        pol_any: Any = pending.pol_t
        wdl_any: Any = pending.wdl_t
        pol = pol_any.numpy() if hasattr(pol_any, "numpy") else np.asarray(pol_any)
        wdl = wdl_any.numpy() if hasattr(wdl_any, "numpy") else np.asarray(wdl_any)
  # CRITICAL: pol/wdl are views of the slot's pinned output. The next
  # submit to this same slot (one ping-pong away) overwrites these
  # views in-place. Callers might read the future result asynchronously,
  # so copy the whole completed batch once and hand callers slices backed
  # by that batch-owned copy.
        pol_copy = pol[:pending.real_pol_n].copy()
        wdl_copy = wdl[:pending.real_n].copy()
        row_offset = 0
        pol_offset = 0
        for item in pending.items:
            n = item.encoded.shape[0]
            if item.legal_counts is not None:
                n_pol = int(np.asarray(item.legal_counts, dtype=np.int32).sum())
                item.future.set_result((
                    pol_copy[pol_offset : pol_offset + n_pol],
                    wdl_copy[row_offset : row_offset + n],
                ))
                pol_offset += n_pol
            else:
                item.future.set_result((
                    pol_copy[row_offset : row_offset + n],
                    wdl_copy[row_offset : row_offset + n],
                ))
            row_offset += n
        self._lifetime_scatter_s += time.perf_counter() - t_scatter0
        self._lifetime_forward_s = (
            self._lifetime_pack_s
            + self._lifetime_submit_s
            + self._lifetime_wait_s
            + self._lifetime_scatter_s
        )
