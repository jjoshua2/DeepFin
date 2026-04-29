"""1-iter-lagged background test eval.

Holdout eval used to run inline at the end of train phase (~30-50s).
This module spawns it on a snapshot of the post-train model so it can
run during the next iter's selfplay phase (the trainer process is
mostly idle waiting for shards). The result is reported in the *next*
iter's metrics row, with ``test_iter`` set to the iter whose model was
actually evaluated — visualisation should key plots on ``test_iter``,
not ``training_iteration``, when async eval is enabled.

**Why a snapshot.** The eval thread reads the model parameters; the
next iter's ``train_steps`` mutates them in place. Without isolation
the eval would read a mix of pre- and mid-train weights. The snapshot
is a freshly-built ``ChessNet`` instance loaded from a CPU copy of the
post-train state_dict.

**Why one long-lived eval thread.** cudagraph_trees keeps thread-local
state and asserts when a compiled forward replays on a thread that
didn't capture its tree (observed crashes 2026-04-29 with per-iter
spawned threads). The fix is to use a single persistent eval thread
that captures its cudagraph tree on the first eval and replays it on
every subsequent iter — same pattern the trainer and ThreadedDispatcher
use. Each iter we just ``load_state_dict`` into the long-lived snapshot
in-place; the cudagraph keys on graph topology, not weight values, so
it stays valid across weight updates.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.train.compile_probe import apply_compile

log = logging.getLogger(__name__)


class _Work:
    """Per-iter work item handed to the eval thread."""

    __slots__ = ("snap_state", "trainer", "buf", "batch_size", "steps", "source_iter")

    def __init__(
        self,
        *,
        snap_state: dict[str, torch.Tensor],
        trainer: Any,
        buf: Any,
        batch_size: int,
        steps: int,
        source_iter: int,
    ) -> None:
        self.snap_state = snap_state
        self.trainer = trainer
        self.buf = buf
        self.batch_size = batch_size
        self.steps = steps
        self.source_iter = source_iter


class AsyncTestEval:
    """Long-lived eval thread + reusable snapshot model.

    The snapshot is built and compiled once on first start(); each
    subsequent start() pushes a new state_dict copy that the worker
    thread loads in-place. cudagraph capture happens once on the worker
    thread's first forward and replays on every iter after.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result_event = threading.Event()
        self._work_q: queue.Queue[_Work | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._init_args: dict[str, Any] | None = None
        self._inflight_iter: int = -1
        self._result: Any = None
        self._exc: BaseException | None = None
        self._source_iter: int = -1

    def start(
        self,
        *,
        trainer: Any,
        model_cfg: ModelConfig,
        holdout_buf: Any,
        batch_size: int,
        steps: int,
        device: str,
        source_iter: int,
        compile_mode: str = "off",
    ) -> None:
        """Snapshot weights to CPU and hand them to the eval thread."""
  # torch.compile prefixes parameter keys with `_orig_mod.`; strip so
  # the snapshot (uncompiled before apply_compile wraps it) loads them.
        snap_state: dict[str, torch.Tensor] = {
            k.removeprefix("_orig_mod."): v.detach().to("cpu").clone()
            for k, v in trainer.model.state_dict().items()
        }
        work = _Work(
            snap_state=snap_state, trainer=trainer, buf=holdout_buf,
            batch_size=batch_size, steps=steps, source_iter=int(source_iter),
        )

        with self._lock:
            if self._thread is None:
  # Lazy init on first call so we can capture device/compile_mode/
  # model_cfg from the trainer's runtime config without forcing the
  # caller to pass them at construction time.
                self._init_args = {
                    "model_cfg": model_cfg,
                    "device": device,
                    "compile_mode": compile_mode,
                }
                self._thread = threading.Thread(
                    target=self._loop, name="AsyncTestEval", daemon=True,
                )
                self._thread.start()

            if self._inflight_iter >= 0 and not self._result_event.is_set():
                log.warning(
                    "AsyncTestEval.start called while previous eval (iter %d) still "
                    "running; abandoning prior result", self._inflight_iter,
                )
  # Drain any stale work item the worker hasn't picked up yet. If the
  # worker has already pulled it and is mid-eval, its result will be
  # written under its work.source_iter (set in _loop), not the new
  # source_iter — so collect() always returns a correctly-labeled
  # result, just possibly an older one.
                try:
                    self._work_q.get_nowait()
                except queue.Empty:
                    pass
            self._inflight_iter = int(source_iter)
            self._result = None
            self._exc = None
            self._source_iter = -1
            self._result_event.clear()

        self._work_q.put(work)

    def _loop(self) -> None:
        """Worker thread main loop. Builds the snapshot once; reuses it forever."""
        if self._init_args is None:
            log.error("AsyncTestEval._loop entered before init args were set")
            return
        try:
            snap = build_model(self._init_args["model_cfg"]).to(self._init_args["device"])
            snap.eval()
            snap = apply_compile(
                snap, mode=self._init_args["compile_mode"],
                device=self._init_args["device"],
            )
        except BaseException as exc:  # noqa: BLE001
            log.exception("AsyncTestEval init failed")
            with self._lock:
                self._exc = exc
            self._result_event.set()
            return

  # apply_compile wraps the model in OptimizedModule whose state_dict
  # carries an ``_orig_mod.`` prefix; the caller strips that prefix
  # so we must load into the underlying module to get matching keys.
  # Falls through to ``snap`` when compile is off.
        load_target = getattr(snap, "_orig_mod", snap)

        while True:
            work = self._work_q.get()
            if work is None:
                return
            try:
  # In-place state_dict load preserves the compiled callable + its
  # cudagraph tree (which key on graph topology, not parameter
  # values) so the next forward just replays the captured graph.
                load_target.load_state_dict(work.snap_state, strict=True)
                metrics = work.trainer._compute_metrics(  # noqa: SLF001
                    buf=work.buf,
                    batch_size=work.batch_size,
                    steps=work.steps,
                    tag="eval",
                    model_override=snap,
                )
                with self._lock:
                    self._result = metrics
                    self._source_iter = work.source_iter
            except BaseException as exc:  # noqa: BLE001
                log.exception("async test eval failed")
                with self._lock:
                    self._exc = exc
            self._result_event.set()

    def collect(self, timeout: float = 120.0) -> tuple[Any, int]:
        """Wait for the in-flight eval and return ``(metrics, source_iter)``.

        ``(None, -1)`` when no eval was started, the eval raised, or the
        wait timed out (the thread keeps the snapshot alive for the next
        iter's start()).
        """
        if self._thread is None:
            return None, -1
        if not self._result_event.wait(timeout=timeout):
            log.warning(
                "async test eval did not finish within %.1fs; dropping iter %d's metrics",
                timeout, self._inflight_iter,
            )
            return None, -1
        with self._lock:
            r = self._result
            it = self._source_iter
            exc = self._exc
            self._result = None
            self._exc = None
            self._source_iter = -1
            self._inflight_iter = -1
            self._result_event.clear()
        if exc is not None:
            return None, -1
        return r, it

    def shutdown(self, timeout: float = 10.0) -> None:
        """Tell the worker thread to exit and join it."""
        if self._thread is None:
            return
        self._work_q.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None
