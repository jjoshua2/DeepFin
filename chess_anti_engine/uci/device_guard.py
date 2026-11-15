"""Stall-watched device calls, and the degrading multi-device routing chain.

``MultiGpuPucvPool`` (the default multi-GPU search) survives a device that
raises or stops answering. Two other UCI paths call a device synchronously on
the search thread, where one stuck call used to hang the whole search:

- the ``--no-multi-gpu-pucv`` routing chain, whose ``MultiGPUDispatcher``
  routes each batch to one device: :class:`DegradingMultiGPUDispatcher`;
- the PRIMARY (root-eval / ``searchmoves``) evaluator beside the pool:
  :class:`DeviceCallRunner` used by ``SearchWorker``.

Both run each device's calls on that device's own persistent thread
(:class:`DeviceCallRunner`), so the caller can stop waiting for a stuck call:
the call is abandoned, the device dropped and announced, and the work redone
on a survivor. Thresholds are the pool's (``multi_gpu_pucv_pool``): while
another device can take the batch, a call blocked past the IDLE-SURVIVOR
threshold (1 s) is abandoned -- nothing else progresses while a synchronous
caller waits, exactly the pool's idle-survivor case. With one device left,
the mid-search threshold (at least 5 s) applies, as on the pool. Before the
chain is armed (``arm``, after its warm-up), a call may take up to the device
startup timeout (``startup_progress``) instead: warm-up calls compile.

A device that RAISED stays dropped. A device dropped for a STALL whose call
later comes back without error is re-admitted before the next call, unless it
stalled ``DEVICE_FLAP_LIMIT`` times in the current search (then at the next
``go``). Every drop / restore / hold is announced once through
``drain_notes``; ``degraded_summary`` is the per-``go`` line.
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from chess_anti_engine.inference_dispatcher import MultiGPUDispatcher, bootstrap_cudagraph_tls
from chess_anti_engine.uci import multi_gpu_pucv_pool as _pool
from chess_anti_engine.uci import startup_progress
from chess_anti_engine.uci import wedge

_log = __import__("logging").getLogger(__name__)


# How long a new call thread may take to set itself up before its first call
# is accepted anyway (its setup then counts against that call).
_RUNNER_SETUP_WAIT_S = 30.0


class DeviceStalled(RuntimeError):
    """A device call outlived its stall threshold and was abandoned."""


def _one_line(exc: BaseException, limit: int = 160) -> str:
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def mid_search_threshold_s(slowest: float) -> float:
    """The pool's mid-search stuck-call threshold for a device whose slowest
    completed call took ``slowest`` seconds (``MultiGpuPucvPool.stall_threshold_s``)."""
    return max(
        _pool._STALL_FLOOR_S,
        min(_pool._STALL_CEILING_S, _pool._STALL_SLOWEST_CALL_FACTOR * slowest),
    )


def idle_survivor_threshold_s(slowest: float) -> float:
    """The pool's idle-survivor threshold (``idle_stall_threshold_s``): 1 s for
    any healthy device; used while a survivor could take the batch."""
    stop = min(
        _pool._STOP_STALL_CEILING_S,
        mid_search_threshold_s(slowest),
        max(_pool._STOP_STALL_FLOOR_S, _pool._STOP_STALL_SLOWEST_CALL_FACTOR * slowest),
    )
    return max(_pool._IDLE_SURVIVOR_STALL_FLOOR_S, stop)


class DeviceCallRunner:
    """Run one device's blocking calls on its own persistent thread.

    ``call(fn, limit_s)`` runs ``fn`` there and waits at most ``limit_s``. A
    call past that is ABANDONED: ``DeviceStalled`` is raised to the caller,
    the call's eventual result is discarded, and every later ``call`` raises
    ``DeviceStalled`` at once until ``restore``. If the abandoned call comes
    back WITHOUT error, the thread parks and ``restorable`` turns True; if it
    raises, the thread ends (the device is not restorable) -- or, while any
    device thread is wedged, parks for good instead (``wedge``).
    """

    def __init__(self, label: str, *, device: str | None = None, bootstrap: bool = True) -> None:
        self.label = label
        self._device = device
        # ``bootstrap``: set up Inductor's cudagraph thread-local state on the
        # call thread (needed when compiled forwards run ON it; skipped for a
        # thread that only forwards to an eager / owner-thread evaluator).
        self._bootstrap = bootstrap
        self._q: queue.SimpleQueue[tuple[Callable[[], Any], list[Any], threading.Event] | None] = (
            queue.SimpleQueue()
        )
        self._lock = threading.Lock()
        self._abandoned = False
        self._late_ok = False
        self._stuck = False  # abandoned AND still inside that call
        self._closed = False
        self.slowest_call_s = 0.0
        self.timing = False  # record ``slowest_call_s`` (after warm-up only)
        self.warmed = False  # one measured call has run (``SearchWorker.warm_primary_runner``)
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=wedge.parking(self._loop), name=f"device-call-{label}", daemon=True,
        )
        self._thread.start()
        # Thread setup (cudagraph TLS bootstrap imports Inductor: about a
        # second, once) must not count against the first call's stall limit.
        self._ready.wait(_RUNNER_SETUP_WAIT_S)

    def _loop(self) -> None:
        try:
            if self._bootstrap:
                bootstrap_cudagraph_tls()
            dev = self._device
            if dev is not None and dev.startswith("cuda"):
                try:
                    torch.cuda.set_device(torch.device(dev).index or 0)
                except Exception:
                    _log.debug("set_device(%s) failed on its call thread", dev, exc_info=True)
        finally:
            self._ready.set()
        while True:
            item = self._q.get()
            if item is None:
                return
            fn, box, done = item
            ok = True
            try:
                box[0] = fn()
            except BaseException as exc:
                box[1] = exc
                ok = False
            with self._lock:
                late = self._abandoned
                if late:
                    self._late_ok = ok
                    self._stuck = False
                    wedge.clear_wedged(threading.current_thread())
                done.set()
            if late and not ok:
                return  # an abandoned call that then raised: not restorable

    def call(self, fn: Callable[[], Any], *, limit_s: float | None) -> Any:
        with self._lock:
            if self._abandoned:
                raise DeviceStalled(f"device {self.label} is dropped after a stall")
            if self._closed:
                raise RuntimeError(f"device {self.label} call runner is closed")
        done = threading.Event()
        box: list[Any] = [None, None]
        started = time.monotonic()
        self._q.put((fn, box, done))
        if limit_s is None:
            done.wait()
        elif not done.wait(limit_s):
            with self._lock:
                if not done.is_set():
                    self._abandoned = True
                    self._late_ok = False
                    self._stuck = True
                    wedge.mark_wedged(self._thread, f"device {self.label} call", device=self.label)
                    raise DeviceStalled(
                        f"one device call blocked for {time.monotonic() - started:.1f}s "
                        f"> {limit_s:.1f}s"
                    )
        if box[1] is not None:
            raise box[1]
        if self.timing:
            took = time.monotonic() - started
            if took > self.slowest_call_s:
                self.slowest_call_s = took
        return box[0]

    @property
    def abandoned(self) -> bool:
        return self._abandoned

    @property
    def restorable(self) -> bool:
        return self._abandoned and self._late_ok and self._thread.is_alive()

    def restore(self) -> None:
        with self._lock:
            self._abandoned = False
            self._late_ok = False

    def thread_alive(self) -> bool:
        return self._thread.is_alive()

    def wedged(self) -> bool:
        """Abandoned inside a call that has not returned (thread still in it)."""
        return self._stuck and self._thread.is_alive()

    def close(self, timeout: float) -> None:
        with self._lock:
            self._closed = True
        self._q.put(None)
        wedge.join_unless_wedged(self._thread, max(0.0, timeout))


class DegradingMultiGPUDispatcher(MultiGPUDispatcher):
    """``MultiGPUDispatcher`` that survives a device that raises or stalls.

    The UCI ``--no-multi-gpu-pucv`` multi-device chain. Each device's calls
    run on its own :class:`DeviceCallRunner` thread; a device that raises or
    blocks past its threshold is dropped (announced once), and the same batch
    is redone on a surviving device, so the search never sees the failure.
    Only when every device has failed does a call raise
    (``ALL_DEVICES_FAILED``); the engine's fallback then answers.
    Routing among live devices is the parent's: fewest in-flight calls, ties
    round-robin.
    """

    def __init__(
        self,
        evaluators: Sequence[Any],
        *,
        labels: Sequence[str] | None = None,
        devices: Sequence[str] | None = None,
        startup_timeout_s: float | None = None,
    ) -> None:
        super().__init__(evaluators)
        n = len(self._evals)
        self._labels = tuple(
            str(labels[k]) if labels is not None and k < len(labels) else f"#{k}"
            for k in range(n)
        )
        self._runners = [
            DeviceCallRunner(
                self._labels[k],
                device=str(devices[k]) if devices is not None and k < len(devices) else None,
            )
            for k in range(n)
        ]
        self._live = [True] * n
        self._raised = [False] * n
        self._drop_reason: list[str | None] = [None] * n
        self._stall_drops_this_search = [0] * n
        self._held_announced = [False] * n
        self._notes: list[str] = []
        self._state_lock = threading.Lock()
        self._armed = False
        self._startup_timeout_s = startup_timeout_s
        self._closed = False

    # -- arming / thresholds ------------------------------------------------

    def arm(self) -> None:
        """Warm-up is done: from now on stalls use the search thresholds and
        each device's call times feed its threshold."""
        self._armed = True
        for r in self._runners:
            r.timing = True

    @property
    def armed(self) -> bool:
        return self._armed

    def _limit_s(self, idx: int) -> float | None:
        if not self._armed:
            return self._startup_timeout_s
        slowest = self._runners[idx].slowest_call_s
        others = any(alive for k, alive in enumerate(self._live) if k != idx)
        return idle_survivor_threshold_s(slowest) if others else mid_search_threshold_s(slowest)

    # -- degrade bookkeeping -------------------------------------------------

    def _drop_locked(self, idx: int, what: str, reason: str, *, stalled: bool) -> None:
        if not self._live[idx]:
            return
        wedge.note_device_fault(f"device {self._labels[idx]} {what}")
        self._live[idx] = False
        self._raised[idx] = self._raised[idx] or not stalled
        if stalled:
            self._stall_drops_this_search[idx] += 1
        if not self._armed:
            what = "failed at startup" if not stalled else "stopped answering at startup"
        self._drop_reason[idx] = f"{what} ({reason})"
        msg = _pool.DEVICE_DROPPED_NOTE.format(
            label=self._labels[idx], what=what, reason=reason,
            live=sum(self._live), total=len(self._evals),
        )
        _log.error("%s", msg)
        self._notes.append(msg)

    def begin_search(self) -> None:
        with self._state_lock:
            n = len(self._evals)
            self._stall_drops_this_search = [0] * n
            self._held_announced = [False] * n

    def restore_recovered_devices(self) -> tuple[str, ...]:
        if not any(r.restorable for r in self._runners):
            return ()
        restored: list[int] = []
        with self._state_lock:
            if self._closed:
                return ()
            for k, r in enumerate(self._runners):
                if self._live[k] or self._raised[k] or not r.restorable:
                    continue
                if self._stall_drops_this_search[k] >= _pool.DEVICE_FLAP_LIMIT:
                    if not self._held_announced[k]:
                        self._held_announced[k] = True
                        msg = _pool.DEVICE_HELD_NOTE.format(
                            label=self._labels[k], count=self._stall_drops_this_search[k],
                            live=sum(self._live), total=len(self._evals),
                        )
                        _log.warning("%s", msg)
                        self._notes.append(msg)
                    continue
                r.restore()
                self._live[k] = True
                self._drop_reason[k] = None
                restored.append(k)
            for k in restored:
                msg = _pool.DEVICE_RESTORED_NOTE.format(
                    label=self._labels[k], live=sum(self._live), total=len(self._evals),
                )
                _log.warning("%s", msg)
                self._notes.append(msg)
        return tuple(self._labels[k] for k in restored)

    def drain_notes(self) -> list[str]:
        with self._state_lock:
            notes, self._notes = self._notes, []
        return notes

    def failed_devices(self) -> tuple[str, ...]:
        with self._state_lock:
            return tuple(self._labels[k] for k, alive in enumerate(self._live) if not alive)

    def degraded_summary(self) -> str | None:
        with self._state_lock:
            failed = [
                f"{self._labels[k]} failed: {self._drop_reason[k] or 'unknown'}"
                for k, alive in enumerate(self._live) if not alive
            ]
            live = sum(self._live)
        if not failed:
            return None
        return f"multi-GPU degraded: {live} of {len(self._evals)} devices ({'; '.join(failed)})"

    def live_evaluators(self) -> list[Any]:
        """The devices still in service (the warm-up's strict check covers these)."""
        with self._state_lock:
            return [ev for ev, alive in zip(self._evals, self._live) if alive]

    def wedged_threads(self) -> bool:
        return any(r.wedged() for r in self._runners)

    @property
    def n_live_devices(self) -> int:
        return sum(self._live)

    # -- routing ------------------------------------------------------------

    def _pick_live(self, tried: set[int]) -> int:
        with self._select_lock:
            live = [k for k, alive in enumerate(self._live) if alive and k not in tried]
            if not live:
                return -1
            best_load = min(self._inflight[k] for k in live)
            ties = [k for k in live if self._inflight[k] == best_load]
            best = ties[self._rr % len(ties)] if len(ties) > 1 else ties[0]
            if len(ties) > 1:
                self._rr = (self._rr + 1) % len(self._evals)
            self._inflight[best] += 1
            return best

    def _dispatch(self, op: Callable[[Any], Any]) -> Any:
        self.restore_recovered_devices()
        tried: set[int] = set()
        last: BaseException | None = None
        while True:
            idx = self._pick_live(tried)
            if idx < 0:
                break
            tried.add(idx)
            ev = self._evals[idx]
            try:
                result = self._runners[idx].call(lambda ev=ev: op(ev), limit_s=self._limit_s(idx))
            except DeviceStalled as exc:
                last = exc
                with self._state_lock:
                    self._drop_locked(idx, "stopped answering", str(exc), stalled=True)
                continue
            except Exception as exc:
                last = exc
                _log.error("device %s raised", self._labels[idx], exc_info=True)
                with self._state_lock:
                    self._drop_locked(idx, "failed mid-search", _one_line(exc), stalled=False)
                continue
            finally:
                with self._select_lock:
                    self._inflight[idx] -= 1
            if not self._armed:
                startup_progress.beat()
            return result
        raise RuntimeError(
            f"{_pool.ALL_DEVICES_FAILED} ({', '.join(self.failed_devices())})"
        ) from last

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if relations is not None:
            return self._dispatch(lambda ev: ev.evaluate_encoded(x, relations=relations))
        return self._dispatch(lambda ev: ev.evaluate_encoded(x))

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._dispatch(lambda ev: ev.evaluate_legal_bf16(x, legal_flat, legal_counts))

    @property
    def search_leaf_row_cap(self) -> int | None:
        caps = [
            getattr(e, "_max_batch", getattr(e, "max_batch", None)) for e in self.live_evaluators()
        ]
        ints = [int(c) for c in caps if isinstance(c, int) and not isinstance(c, bool) and c > 0]
        if not caps or len(ints) != len(caps):
            return None
        return min(ints)

    def close(self) -> None:
        """Stop every device thread; a thread wedged in a call is left (see
        ``wedged_threads``) after a short shared wait."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        deadline = time.monotonic() + _pool._CLOSE_ABANDONED_JOIN_S
        for r in self._runners:
            if r.wedged():
                r.close(deadline - time.monotonic())
            else:
                r.close(2.0)


def find_degrade_source(chain: Any) -> DegradingMultiGPUDispatcher | None:
    """The :class:`DegradingMultiGPUDispatcher` inside a UCI evaluator chain."""
    seen: set[int] = set()
    stack = [chain]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, DegradingMultiGPUDispatcher):
            return node
        d = getattr(node, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for attr in ("_inner", "_eval"):
            child = d.get(attr)
            if child is not None:
                stack.append(child)
    return None


class GuardedPrimaryEvaluator:
    """The primary evaluator seen through ``SearchWorker._primary_call``.

    Handed to the classic Gumbel C path for a ``searchmoves`` search while a
    multi-GPU pool is live, so a primary that raises or stalls mid-search is
    caught (``failure``) instead of failing or hanging the search; the caller
    then reruns the chunk on the pool (``PoolRowsEvaluator``). It exposes only
    the synchronous ``evaluate_encoded`` (every call is one guarded device
    call) and advertises the primary's own batch cap as
    ``search_leaf_row_cap`` -- the cap the in-place path applies -- so the
    search shape (which leaves are evaluated vs absorbed) is unchanged.
    """

    def __init__(self, inner: Any, call: Callable[[Callable[[], Any]], Any]) -> None:
        self._inner = inner
        self._call = call
        self.failure: BaseException | None = None

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        inner = self._inner
        try:
            if relations is not None:
                return self._call(lambda: inner.evaluate_encoded(x, relations=relations))
            return self._call(lambda: inner.evaluate_encoded(x))
        except BaseException as exc:
            if self.failure is None:
                self.failure = exc
            raise

    @property
    def search_leaf_row_cap(self) -> int | None:
        cap = getattr(self._inner, "search_leaf_row_cap", None)
        if isinstance(cap, int) and not isinstance(cap, bool):
            return cap
        mb = getattr(self._inner, "_max_batch", getattr(self._inner, "max_batch", None))
        if isinstance(mb, int) and not isinstance(mb, bool) and mb > 0:
            return mb
        return None


class PoolRowsEvaluator:
    """Classic-Gumbel leaf evaluations served by the multi-GPU pool's workers.

    The ``searchmoves`` path once the primary evaluator has failed: each batch
    runs on a surviving pool worker's own evaluator and thread
    (``MultiGpuPucvPool.evaluate_rows``), so a worker that raises or stalls is
    dropped and the next one takes the batch. Leaf batches are capped at the
    rows every worker warmed (its gather), so no unwarmed shape is compiled on
    the clock; that cap makes this degraded search narrower per batch.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._pool.evaluate_rows(x, relations, idle_limit=True)

    @property
    def search_leaf_row_cap(self) -> int:
        return int(self._pool.worker_row_cap())

    @property
    def _max_batch(self) -> int:
        return int(self._pool.worker_row_cap())
