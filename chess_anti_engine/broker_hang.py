"""Hang self-abort for GPU-touching processes (torch-free).

A wedged CUDA / WSL2 dxg vmbus context blocks forever inside a CUDA call with
no exception. The process stays alive, so the per-iteration supervisor sees a
live PID and never respawns. This module tracks in-flight work and hard-exits
so the supervisor can restart a healthy process.

Two callers, one implementation: the inference **broker** (exit 42) covers its
forwards and boot stages, and the selfplay **worker** (exit 43) covers the init
span between taking a server lease and having a live evaluator (audit R1). They
differ only in the name in the log line and the code in the exit status.

Two things about the pre-2026-08-03 version made it unable to fire on the
scenario in the paragraph above (audit I3), and both are fixed here:

* it was **inert until the first successful forward** (``armed``), so a broker
  that booted straight into a wedged context hung on forward #1, was never
  armed, and never aborted. Arming now happens at construction; the pre-first-
  success window simply uses a longer ``boot_threshold_s`` so a cold
  ``torch.compile`` max-autotune pass still cannot false-fire.
* the instrumented window covered only H2D + forward + device sync, while the
  calls most likely to wedge a cold bridge — ``torch.load``, ``.to(device)``,
  the AOT ``load_constants`` loop, ``torch.compile`` — ran outside it. Callers
  now open a named stage with :meth:`stage` around any of those.

In-flight work is tracked as a dict keyed by an opaque token rather than one
slot, so "oldest" means oldest: previously, completing a NEWER item cleared an
OLDER item's start time and a wedged batch could age forever without aborting.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import threading
import time
from collections.abc import Callable, Generator, Mapping, MutableMapping

log = logging.getLogger(__name__)

DEFAULT_HANG_ABORT_S = 300.0
# Cold start (model load, AOT package load/replay, first compile) is legitimately
# slow and highly variable, so the pre-first-success window gets its own, much
# larger threshold instead of the old "never abort at all". 1800s is far above
# any observed broker boot and far below "forever", which is what the wedge does.
DEFAULT_BOOT_HANG_ABORT_S = 1800.0
HANG_ABORT_ENV = "CAE_BROKER_HANG_ABORT_S"
BOOT_HANG_ABORT_ENV = "CAE_BROKER_BOOT_HANG_ABORT_S"
HANG_ABORT_EXIT_CODE = 42
HANG_ABORT_POLL_S = 10.0

# The worker runs the same watchdog over its own init span. A DISTINCT code so
# a supervisor (and a human reading `ps`/exit status) can tell "the broker's
# CUDA context wedged" from "a worker's did" without parsing logs.
WORKER_HANG_ABORT_EXIT_CODE = 43
WORKER_HANG_ABORT_ENV = "CAE_WORKER_BOOT_HANG_ABORT_S"
# Same reasoning as DEFAULT_BOOT_HANG_ABORT_S: the span this covers is a model
# load + optional torch.compile, which is legitimately minutes on a cold cache.
DEFAULT_WORKER_BOOT_HANG_ABORT_S = 1800.0

NVML_CUDA_CHECK_ENV = "PYTORCH_NVML_BASED_CUDA_CHECK"


def pin_nvml_cuda_check(env: MutableMapping[str, str] | None = None) -> bool:
    """Route ``torch.cuda.is_available()`` around ``cuInit``. Returns True if set here.

    ⚑ LOAD-BEARING, AND IT USED TO BE INHERITED BY ACCIDENT (audit R3). torch's
    ``is_available()`` has two implementations::

        if _nvml_based_avail():          # PYTORCH_NVML_BASED_CUDA_CHECK == "1"
            return device_count() > 0    #   NVML — does not touch the driver
        else:
            return torch._C._cuda_getDeviceCount() > 0

    and torch's own comment on the second branch says it "uses the CUDA Runtime
    API ``cudaGetDeviceCount`` which in turn initializes the CUDA Driver API via
    ``cuInit``". ``cuInit`` is exactly the call that never returns on a wedged
    WSL2 dxg bridge, so on the default branch a *probe for whether a GPU exists*
    is itself a hang point — before any watchdog stage is open.

    The live run had this set, which is why the probe has never hung in practice.
    But `git grep` found it in **no tracked file**: it came from the interactive
    shell that started Ray. A run launched from systemd, cron, or a volunteer's
    machine would silently lose it. Pinning it here makes the property the code's
    rather than the environment's.

    ``setdefault`` semantics: an operator who deliberately exports ``0`` to force
    the driver-based check keeps it. torch reads this with ``os.getenv`` at CALL
    time (not at import), so setting it any time before the first
    ``torch.cuda.is_available()`` is effective — including after ``import torch``.

    Not a complete fix: torch's NVML branch falls back to the ``cuInit`` path when
    NVML discovery itself fails, which a sufficiently wedged bridge may cause.
    That residual is what the watchdog in this module covers.
    """
    target = os.environ if env is None else env
    if target.get(NVML_CUDA_CHECK_ENV):
        return False
    target[NVML_CUDA_CHECK_ENV] = "1"
    return True


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


def resolve_boot_hang_abort_seconds(
    hang_abort_seconds: float,
    default_seconds: float = DEFAULT_BOOT_HANG_ABORT_S,
    *,
    env: Mapping[str, str] | None = None,
) -> float:
    """Resolve the cold-start threshold; ``CAE_BROKER_BOOT_HANG_ABORT_S`` overrides.

    **Returns 0.0 (disabled) whenever the steady-state threshold is disabled.**
    ``--hang-abort-seconds 0`` / ``CAE_BROKER_HANG_ABORT_S=0`` is the documented
    escape hatch for "do not hang-abort this broker", and it is what someone
    reaches for precisely when the watchdog is misfiring. Arming from process
    start (audit I3) turned that off by accident: the boot window was resolved
    independently, so a broker told not to hang-abort still started the thread
    and still ``os._exit(42)``-ed at 1800 s while unarmed.

    The gate lives HERE rather than at each construction site on purpose --
    there are two brokers, and a rule duplicated across call sites is how one of
    them ends up without it. ``hang_abort_seconds`` is positional so a new
    caller cannot omit it.
    """
    if float(hang_abort_seconds) <= 0.0:
        return 0.0
    env_map = os.environ if env is None else env
    raw = env_map.get(BOOT_HANG_ABORT_ENV)
    if raw is not None and str(raw).strip() != "":
        return float(raw)
    return float(default_seconds)


def resolve_worker_boot_hang_abort_seconds(
    *,
    env: Mapping[str, str] | None = None,
    default_seconds: float = DEFAULT_WORKER_BOOT_HANG_ABORT_S,
) -> float:
    """Worker init-span threshold; ``CAE_WORKER_BOOT_HANG_ABORT_S`` overrides.

    ``0`` (or any non-positive value) disables the watchdog entirely, matching
    the broker's documented escape hatch and ``CAE_WORKER_STALL_TIMEOUT_S=0``
    for the worker's existing in-session stall watchdog. A malformed value is
    NOT silently treated as "disabled" -- that is the one reading an operator
    would never intend -- it falls back to the default.
    """
    env_map = os.environ if env is None else env
    raw = env_map.get(WORKER_HANG_ABORT_ENV)
    if raw is None or str(raw).strip() == "":
        return float(default_seconds)
    try:
        return float(raw)
    except ValueError:
        log.warning(
            "%s=%r is not a number; using the %.0fs default",
            WORKER_HANG_ABORT_ENV, raw, float(default_seconds),
        )
        return float(default_seconds)


def should_hang_abort(
    *,
    armed: bool,
    oldest_inflight_age_s: float | None,
    threshold_s: float,
    boot_threshold_s: float | None = None,
) -> bool:
    """Pure decision: abort when work is in flight and older than the threshold.

    ``armed`` (first forward has succeeded) no longer gates the decision — it
    only selects WHICH threshold applies. Before the first success the window is
    ``boot_threshold_s`` (defaults to ``threshold_s`` when not supplied, which
    keeps single-threshold callers behaving sensibly); after it, ``threshold_s``.
    A threshold of <= 0 disables that window. ``oldest_inflight_age_s is None``
    means nothing is in flight.
    """
    if oldest_inflight_age_s is None:
        return False
    if armed or boot_threshold_s is None:
        effective = float(threshold_s)
    else:
        effective = float(boot_threshold_s)
    if effective <= 0.0:
        return False
    return float(oldest_inflight_age_s) >= effective


class BrokerHangWatchdog:
    """Daemon thread that hard-exits if broker work wedges past a threshold.

    Tracks the monotonic start of every in-flight item (forward or named boot
    stage) and aborts on the oldest. Live from construction: the cold window is
    governed by ``boot_threshold_s`` rather than by refusing to look. Exit uses
    ``os._exit`` (injectable) because a dead CUDA context can hang normal
    teardown.
    """

    def __init__(
        self,
        *,
        threshold_s: float,
        boot_threshold_s: float | None = None,
        poll_interval_s: float = HANG_ABORT_POLL_S,
        exit_fn: Callable[[int], None] | None = None,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        component: str = "broker",
        exit_code: int = HANG_ABORT_EXIT_CODE,
    ) -> None:
        # `component` and `exit_code` exist so the WORKER can run this same
        # detector over its own init span (audit R1) without a second copy of
        # the logic. A duplicated watchdog is how one of them silently stops
        # matching the other; what differs between the two callers is the name
        # in the log line and the code in the exit status, nothing else.
        self._component = str(component)
        self._exit_code = int(exit_code)
        self._threshold_s = float(threshold_s)
        self._boot_threshold_s = (
            float(threshold_s) if boot_threshold_s is None else float(boot_threshold_s)
        )
        self._poll_interval_s = float(poll_interval_s)
        self._exit_fn: Callable[[int], None] = exit_fn if exit_fn is not None else os._exit
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._sleep: Callable[[float], None] = sleep_fn if sleep_fn is not None else time.sleep
        self._lock = threading.Lock()
        self._armed = False
        # token -> (start_s, label, batch_size). A dict rather than one slot so
        # completing a newer item cannot erase an older one's age.
        self._inflight: dict[int, tuple[float, str, int]] = {}
        self._tokens = itertools.count(1)
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

    @property
    def boot_threshold_s(self) -> float:
        return self._boot_threshold_s

    def oldest_inflight_age_s(self, now: float | None = None) -> float | None:
        """Age of the oldest in-flight item, or None if idle."""
        with self._lock:
            if not self._inflight:
                return None
            start = min(s for s, _, _ in self._inflight.values())
        t = self._clock() if now is None else float(now)
        return max(0.0, t - start)

    def _begin(self, label: str, batch_size: int) -> int:
        token = next(self._tokens)
        with self._lock:
            self._inflight[token] = (self._clock(), str(label), int(batch_size))
        return token

    def _end(self, token: int, *, success: bool, arm: bool) -> None:
        with self._lock:
            self._inflight.pop(token, None)
            if success and arm:
                self._armed = True

    def mark_forward_start(self, batch_size: int) -> int:
        """Record start of a GPU batch. Returns a token for ``mark_forward_done``."""
        return self._begin("forward", batch_size)

    def mark_forward_done(self, *, success: bool = True, token: int | None = None) -> None:
        """Clear an in-flight forward; arm the detector after a success.

        ``token`` is what ``mark_forward_start`` returned. It stays optional so
        the single-threaded serve loops read the same as before; when omitted the
        oldest forward is cleared, which is exactly right for a serial loop and
        is never reached by the (currently non-existent) pipelined caller.
        """
        with self._lock:
            if token is None:
                forwards = [
                    (start, tok) for tok, (start, label, _) in self._inflight.items()
                    if label == "forward"
                ]
                token = min(forwards)[1] if forwards else None
            if token is not None:
                self._inflight.pop(token, None)
            if success:
                self._armed = True

    @contextlib.contextmanager
    def stage(self, label: str) -> Generator[None]:
        """Cover a non-forward blocking section (model load, compile, AOT load).

        Does NOT arm the detector: arming means "a real forward has completed on
        this CUDA context", and a stage completing proves nothing about that.
        """
        token = self._begin(label, 0)
        try:
            yield
        finally:
            self._end(token, success=False, arm=False)

    def start(self) -> None:
        """Start the daemon poll loop. No-op when both thresholds are disabled."""
        if self._threshold_s <= 0.0 and self._boot_threshold_s <= 0.0:
            return
        if self._thread is not None:
            return
        self._stop = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self._component}-hang-watchdog",
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
                log.exception("%s hang watchdog tick failed (ignored)", self._component)
            try:
                self._sleep(self._poll_interval_s)
            except Exception:
                log.exception("%s hang watchdog sleep failed (ignored)", self._component)

    def _maybe_abort(self) -> bool:
        with self._lock:
            if self._aborted:
                return True
            armed = self._armed
            oldest = min(self._inflight.values()) if self._inflight else None
            threshold = self._threshold_s
            boot_threshold = self._boot_threshold_s
        if oldest is None:
            age_s = None
            label = ""
            batch_size = 0
        else:
            start, label, batch_size = oldest
            age_s = max(0.0, self._clock() - start)
        if not should_hang_abort(
            armed=armed,
            oldest_inflight_age_s=age_s,
            threshold_s=threshold,
            boot_threshold_s=boot_threshold,
        ):
            return False
        with self._lock:
            if self._aborted:
                return True
            self._aborted = True
        # ONE critical line then hard-exit — no cleanup (CUDA context is dead).
        log.critical(
            "%s hang abort: %s in flight for %.1fs (batch_size=%d, armed=%s, "
            "threshold=%.1fs) — GPU context likely dead — see the WSL2 dxg vmbus "
            "wedge; supervisor will respawn",
            self._component,
            label or "work",
            float(age_s if age_s is not None else 0.0),
            int(batch_size),
            armed,
            float(threshold if armed else boot_threshold),
        )
        self._exit_fn(self._exit_code)
        return True
