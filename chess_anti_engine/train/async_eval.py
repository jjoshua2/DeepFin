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
post-train state_dict, moved to GPU at thread start. The snapshot is
re-wrapped with the same ``apply_compile`` mode as the trainer so the
shared TORCHINDUCTOR/Triton caches hit on graph + kernel keys, skipping
the autotune phase. cudagraph capture still happens once on the eval
thread (cudagraph state is thread-local per the ThreadedDispatcher
comments) but replays cheaply on subsequent iters.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.train.compile_probe import apply_compile

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
  # Wrap in the trainer's compile mode so the shared inductor +
  # Triton caches hit (shared via TORCHINDUCTOR_CACHE_DIR /
  # TRITON_CACHE_DIR). Compile happens on this thread so cudagraph
  # TLS lines up with the forward call below.
                snap = apply_compile(snap, mode=compile_mode, device=device)
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
