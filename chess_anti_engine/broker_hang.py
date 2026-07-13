"""Hang self-abort for the inference broker (torch-free).

A wedged CUDA / WSL2 dxg vmbus context blocks forever inside a CUDA call with
no exception. The broker process stays alive, so the per-iteration supervisor
sees a live PID and never respawns. This module tracks in-flight forwards and
hard-exits (``os._exit(42)``) so the supervisor can restart a healthy process.

Kept free of torch/CUDA imports so unit tests can exercise the decision logic
even when the GPU stack is wedged.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping

log = logging.getLogger(__name__)

DEFAULT_HANG_ABORT_S = 300.0
HANG_ABORT_ENV = "CAE_BROKER_HANG_ABORT_S"
HANG_ABORT_EXIT_CODE = 42
HANG_ABORT_POLL_S = 10.0


def resolve_hang_abort_seconds(
    cli_seconds: float,
    *,
    env: Mapping[str, str] | None = None,
) -> float:
    """Resolve hang-abort threshold: env ``CAE_BROKER_HANG_ABORT_S`` overrides CLI."""
    env_map = os.environ if env is None else env
    raw = env_map.get(HANG_ABORT_ENV)
    if raw is not None and str(raw).strip() != "":
        return float(raw)
    return float(cli_seconds)


def should_hang_abort(
    *,
    armed: bool,
    oldest_inflight_age_s: float | None,
    threshold_s: float,
) -> bool:
    """Pure decision: abort when armed, a forward is in flight, and age >= threshold.

    Inert until the first successful batch completes (``armed``). ``threshold_s <= 0``
    disables the feature. ``oldest_inflight_age_s is None`` means nothing is in flight.
    """
    if threshold_s <= 0.0 or not armed or oldest_inflight_age_s is None:
        return False
    return float(oldest_inflight_age_s) >= float(threshold_s)


class BrokerHangWatchdog:
    """Daemon thread that hard-exits if a GPU forward wedges past a threshold.

    Tracks the monotonic start of the oldest currently in-flight batch. Arms only
    after the first successful completion so cold ``torch.compile`` max-autotune
    can take many minutes without false-firing. Exit uses ``os._exit`` (injectable)
    because a dead CUDA context can hang normal teardown.
    """

    def __init__(
        self,
        *,
        threshold_s: float,
        poll_interval_s: float = HANG_ABORT_POLL_S,
        exit_fn: Callable[[int], None] | None = None,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._threshold_s = float(threshold_s)
        self._poll_interval_s = float(poll_interval_s)
        self._exit_fn: Callable[[int], None] = exit_fn if exit_fn is not None else os._exit
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._sleep: Callable[[float], None] = sleep_fn if sleep_fn is not None else time.sleep
        self._lock = threading.Lock()
        self._armed = False
        self._inflight_start_s: float | None = None
        self._inflight_batch_size = 0
        self._stop = False
        self._aborted = False
        self._thread: threading.Thread | None = None

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    @property
    def threshold_s(self) -> float:
        return self._threshold_s

    def oldest_inflight_age_s(self, now: float | None = None) -> float | None:
        """Age of the oldest in-flight forward, or None if idle."""
        with self._lock:
            start = self._inflight_start_s
        if start is None:
            return None
        t = self._clock() if now is None else float(now)
        return max(0.0, t - start)

    def mark_forward_start(self, batch_size: int) -> None:
        """Record start of a GPU batch. Keeps the oldest start if already in flight."""
        with self._lock:
            if self._inflight_start_s is None:
                self._inflight_start_s = self._clock()
                self._inflight_batch_size = int(batch_size)

    def mark_forward_done(self, *, success: bool = True) -> None:
        """Clear in-flight marker; arm the detector after a successful completion."""
        with self._lock:
            self._inflight_start_s = None
            self._inflight_batch_size = 0
            if success:
                self._armed = True

    def start(self) -> None:
        """Start the daemon poll loop. No-op when threshold is disabled (<= 0)."""
        if self._threshold_s <= 0.0:
            return
        if self._thread is not None:
            return
        self._stop = False
        self._thread = threading.Thread(
            target=self._run,
            name="broker-hang-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def check_once(self) -> bool:
        """Evaluate once (for tests). Returns True if abort was triggered."""
        return self._maybe_abort()

    def _run(self) -> None:
        while not self._stop:
            try:
                if self._maybe_abort():
                    return
            except Exception:
                # A broken watchdog must not kill a healthy broker.
                log.exception("broker hang watchdog tick failed (ignored)")
            try:
                self._sleep(self._poll_interval_s)
            except Exception:
                log.exception("broker hang watchdog sleep failed (ignored)")

    def _maybe_abort(self) -> bool:
        with self._lock:
            if self._aborted:
                return True
            armed = self._armed
            start = self._inflight_start_s
            batch_size = self._inflight_batch_size
            threshold = self._threshold_s
        age_s = None if start is None else max(0.0, self._clock() - start)
        if not should_hang_abort(
            armed=armed,
            oldest_inflight_age_s=age_s,
            threshold_s=threshold,
        ):
            return False
        with self._lock:
            if self._aborted:
                return True
            self._aborted = True
        # ONE critical line then hard-exit — no cleanup (CUDA context is dead).
        log.critical(
            "broker hang abort: forward in flight for %.1fs (batch_size=%d) — "
            "GPU context likely dead — see the WSL2 dxg vmbus wedge; supervisor will respawn",
            float(age_s if age_s is not None else 0.0),
            int(batch_size),
        )
        self._exit_fn(HANG_ABORT_EXIT_CODE)
        return True
