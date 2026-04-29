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
post-train state_dict, moved to GPU at thread start.

**Why eager (not compiled).** Compiling the snapshot was tempting (it
would hit the shared inductor cache) but cudagraph_trees keeps
thread-local state and asserts when a compiled forward replays on a
non-main thread (observed crashes 2026-04-29). Eval is once per iter
on ~200 batches, so the 2-3x compile speedup isn't worth the
thread-affinity hazard.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import torch

from chess_anti_engine.model import ModelConfig, build_model

log = logging.getLogger(__name__)


class AsyncTestEval:
    """Owns the snapshot model + thread + result holder for one trial."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: Any = None
        self._exc: BaseException | None = None
        self._source_iter: int = -1

    def start(
        self,
        *,
        trainer,
        model_cfg: ModelConfig,
        holdout_buf,
        batch_size: int,
        steps: int,
        device: str,
        source_iter: int,
        compile_mode: str = "off",
    ) -> None:
        """Snapshot weights to CPU + spawn eval thread.

        State_dict cloning happens synchronously here, so this returns
        once the snapshot is independent of ``trainer.model``. The
        caller has just finished ``train_steps``, so no concurrent
        optimizer is running — the CPU detour is purely to release the
        GPU references before the next iter's train phase grows them.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                log.warning(
                    "AsyncTestEval.start called while previous eval still running; "
                    "abandoning prior thread (it will finish in the background)",
                )
            self._result = None
            self._exc = None
            self._source_iter = int(source_iter)

  # torch.compile prefixes parameter keys with `_orig_mod.`; strip so
  # the snapshot (uncompiled) can load them via load_state_dict.
        snap_state: dict[str, torch.Tensor] = {
            k.removeprefix("_orig_mod."): v.detach().to("cpu").clone()
            for k, v in trainer.model.state_dict().items()
        }

        def _run() -> None:
            try:
                snap = build_model(model_cfg).to(device)
                snap.load_state_dict(snap_state, strict=False)
                snap.eval()
  # Skip torch.compile: cudagraph_trees keeps thread-local state and
  # asserts in get_obj when a compiled forward runs on a non-main
  # thread (observed iter 26-27 — every async eval crashed with
  # AssertionError in cudagraph_trees.py). Eval is one shot per iter
  # on ~200 batches, so the inductor speedup isn't worth the
  # thread-affinity hazard. Eager forward runs ~2-3x slower than
  # compiled but completes in ~30-50s, well inside the 120s timeout.
                _ = compile_mode  # accepted for API parity; intentionally unused
                metrics = trainer._compute_metrics(  # noqa: SLF001
                    buf=holdout_buf,
                    batch_size=batch_size,
                    steps=steps,
                    tag="eval",
                    model_override=snap,
                )
                with self._lock:
                    self._result = metrics
            except BaseException as exc:  # noqa: BLE001
                log.exception("async test eval failed")
                with self._lock:
                    self._exc = exc

        self._thread = threading.Thread(
            target=_run, name="AsyncTestEval", daemon=True,
        )
        self._thread.start()

    def collect(self, timeout: float = 120.0) -> tuple[Any, int]:
        """Join the running eval and return ``(metrics, source_iter)``.

        ``(None, -1)`` when no eval ran, or the join timed out (the
        thread keeps running and ``start()`` will warn-and-orphan it on
        the next iter rather than crash).
        """
        if self._thread is None:
            return None, -1
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            log.warning(
                "async test eval did not finish within %.1fs; dropping iter %d's metrics",
                timeout, self._source_iter,
            )
  # Clear the handle so the next start() doesn't see this as "still
  # running" — the thread is daemon, it'll exit when the trial does.
            self._thread = None
            self._source_iter = -1
            return None, -1
        with self._lock:
            r = self._result
            it = self._source_iter
            exc = self._exc
            self._result = None
            self._exc = None
            self._source_iter = -1
        self._thread = None
        if exc is not None:
            return None, -1
        return r, it
