"""Multi-GPU pucv pool: one worker thread per GPU sharing a single tree.

Each worker:
  - Owns a dedicated GPU evaluator (its own DirectGPUEvaluator on its own
    device, with its own torch.compile + cudagraph state).
  - Runs the 2-slot async pipeline (the same one ``PucvChunker`` uses)
    against the SHARED MCTSTree. Tree mutations are atomic + mutex-
    protected in the C extension (same guarantees walker_pool relies on).
  - Pulls work from a shared atomic budget (Semaphore, like walker_pool)
    so faster GPUs naturally take more leaves.

This is the pool-level multi-GPU primitive. The single-GPU PucvChunker
stays as a leaner code path for `Threads=1` UCI; this kicks in when the
user wires multiple devices.

Degrade path (one device failing must not cost the move or the game):
  - A worker whose evaluator RAISES mid-search is dropped for the rest of the
    pool's life. Its in-flight leaves have their virtual loss released (the
    same ``_release_outstanding`` cleanup as before), their budget tokens go
    back to the shared semaphore so the surviving workers search them, and the
    run returns normally. Only when no worker is left does ``run`` raise, which
    the engine answers with its existing fallback bestmove.
  - A worker stuck inside ONE device call (submit or completion wait) for
    longer than ``stall_threshold_s()`` is ABANDONED: the run thread releases
    its in-flight virtual loss under the pool lock and stops waiting for it.
    The stuck thread re-checks under the same lock when (if ever) the call
    returns and never touches the tree or the eval cache again with that
    result, so the release happens exactly once.
  - RESTORE: a device abandoned for a STALL whose call later returns without
    error (and whose other in-flight receipts then drain cleanly) is healthy;
    it parks, and ``restore_recovered_devices()`` -- called at the start of
    every UCI search -- re-admits it with a fresh quarantine. A device that
    RAISED stays dropped for the pool's life: a raise is evidence of a fault,
    a stall alone is not.
  - With ``degrade_on_startup_failure`` (set by the UCI engine) a worker that
    fails at startup leaves the pool running on the others (DEGRADED); only
    when every worker fails does construction raise. The class default stays
    strict: any startup failure raises and retires every owned evaluator.
  - A quiesced failure (evaluator still usable) on the LAST live worker is
    not a drop: there is nothing to degrade onto, so the run re-raises and
    the next run reuses the device, exactly as before the degrade path.
  - Every drop is announced once through ``drain_notes()``; the failed
    evaluator stays quarantined (``async_safety``) so nothing reuses it.

Why a new class instead of extending WalkerPool: walker_pool's hot path
is ``walker_descend_puct`` (single leaf → ONE submit per gather) on a
shared evaluator. Multi-GPU pucv runs ``batch_descend_puct`` (G leaves →
ONE submit) on a per-worker evaluator with 2-slot pipelining. The shapes
are different enough that unifying them would muddy both.
"""
from __future__ import annotations

import sys
import logging
from contextlib import contextmanager
import threading
import time
from dataclasses import dataclass, replace
from collections.abc import Sequence
from typing import Any, cast
from collections.abc import Callable

import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard
from chess_anti_engine.inference_cache import PucvEvalCache, PucvEvalCacheStats
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.headroom import reserve_concurrent_headroom
from chess_anti_engine.mcts.puct_history import puct_history_args
from chess_anti_engine.mcts.puct_tb import LeafSnapshots, descend_batch, mark_batch_solved, override_leaf_values
from chess_anti_engine.tablebase import SyzygyProbe
from chess_anti_engine.async_safety import (
    drain_failed_receipts,
    ensure_usable,
    note_cleanup,
    pending_receipt,
    poison,
    release_quarantine,
    submit_async,
)
from chess_anti_engine.mcts.puct_vl import _PLANES, _alloc_buffers, _release_outstanding
from chess_anti_engine.uci import wedge
from chess_anti_engine.uci.startup_progress import StartupContext, startup_context

_log = logging.getLogger(__name__)

# Floor of the MID-SEARCH stuck-device threshold. The threshold used for device
# k is the larger of this and ``_STALL_SLOWEST_CALL_FACTOR`` x the slowest call
# device k has completed, seeded from its steady-state (second-pass) warmup
# timings (``warm_steady_call_s``). Why 5 s, deliberately conservative: WHILE THE
# OTHER DEVICES STILL HAVE WORK, a stuck device costs only throughput -- the
# survivors keep searching -- whereas a FALSE drop costs a whole GPU until it is
# restored at the next ``go``. Healthy one-off stalls (a CUDA-graph record on a
# shape's first real call, a GC pause, a driver/OS hiccup) are well under a
# second. Once the survivors have drained the job's budget and sit idle, the
# patience buys nothing (the run cannot return while a worker is stuck), so the
# STOP threshold applies at once instead (see ``_abandon_stuck_locked``).
_STALL_FLOOR_S = 5.0
_STALL_SLOWEST_CALL_FACTOR = 10.0
# Hard ceiling of the mid-search threshold. The slowest-call baseline only ever
# grows (one healthy 4 s call would otherwise mean 40 s of patience for the
# rest of the pool's life); no decay machinery, just a cap.
_STALL_CEILING_S = 30.0
# Once the job's stop event is set (``stop``, the deadline watchdog) nobody waits
# for search results any more, and a search still blocked on one device would
# hold the worker past the next ``go`` (it would be parked and that move
# answered unsearched). Under a stop the threshold drops to the larger of this
# and ``_STOP_STALL_SLOWEST_CALL_FACTOR`` x the slowest completed call.
_STOP_STALL_FLOOR_S = 0.25
_STOP_STALL_SLOWEST_CALL_FACTOR = 5.0
# Hard ceiling of the stop-time threshold: however slow a call the device once
# completed, a wedged device can never hold a stopped search (and so the next
# ``go``) for more than this.
_STOP_STALL_CEILING_S = 1.0
# Floor of the IDLE-SURVIVOR threshold (the others have finished the job and
# wait on one worker; see ``_abandon_stuck_locked``). A chunk is only a few
# batches per device, so the others go idle almost at once; at the bare stop
# threshold (0.25 s) any healthy hiccup over a quarter second would drop a GPU.
# One second keeps ordinary hiccups and still bounds the wasted time per chunk;
# a stall past it is dropped and re-admitted at the next chunk once it answers.
_IDLE_SURVIVOR_STALL_FLOOR_S = _STOP_STALL_CEILING_S
# ``close`` waits at most this long (shared) for abandoned workers to exit.
_CLOSE_ABANDONED_JOIN_S = 1.0
# How often ``run`` re-checks for a stuck worker while it waits; a normal
# completion wakes it at once through the condition variable.
_RUN_POLL_S = 0.05
# How often the constructor re-checks the startup no-progress clocks.
_STARTUP_POLL_S = 0.5

# The text every device drop is announced with. "DEGRADED" is one of the
# selfcheck's failure markers, so ``--selfcheck`` reports a degraded start.
DEVICE_DROPPED_NOTE = (
    "WARNING multi-GPU DEGRADED: device {label} {what} ({reason}); "
    "searching on {live} of {total} devices"
)
ALL_DEVICES_FAILED = "multi-GPU pool: every device has failed"
DEVICE_RESTORED_NOTE = (
    "multi-GPU RESTORED: device {label} answered again after its stall; "
    "searching on {live} of {total} devices"
)
# Flap limit: a device dropped for a stall this many times within ONE search
# (one ``go``) is not re-admitted again until the next ``go``. Without it, a
# device whose calls keep taking just over the idle threshold prints a
# DEGRADED / RESTORED pair every chunk. Two keeps the "a false drop costs one
# chunk" recovery for a one-off hiccup, and caps the chatter at two drops, one
# restore and one hold line per move.
DEVICE_FLAP_LIMIT = 2
DEVICE_HELD_NOTE = (
    "WARNING multi-GPU device {label} stalled {count} times during this search; "
    "kept out until the next go ({live} of {total} devices)"
)


class _Abandoned(Exception):
    """Raised inside a worker whose stuck device call outlived its abandonment."""


class _WorkerQuarantine:
    """Per-worker stand-in for the pool in ``async_safety`` poisoning.

    A failed worker's evaluator and buffers are quarantined on this object, not
    on the pool, so the other workers stay usable."""

    def __init__(self, idx: int) -> None:
        self.idx = idx


def _is_poisoned(owner: Any) -> bool:
    try:
        ensure_usable(owner)
    except RuntimeError:
        return True
    return False


def _one_line(exc: BaseException, limit: int = 160) -> str:
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class MultiGpuPucvConfig:
    n_gpus: int
    gather: int = 384
    c_puct: float = 1.4
    # Lc0-style visit-dependent Cpuct (factor<=0 = fixed c_puct).
    cpuct_factor: float = 0.0
    cpuct_base: float = 38739.0
    fpu_at_root: float = 0.0
    fpu_reduction: float = 0.2
    vloss_weight: int = 3
    vloss_mode: int = 0
    eval_cache_entries: int = 0
  # Input channel count (146 v1 / 175 v2_threats); sized from the model.
    input_planes: int = _PLANES
  # Transport dynamic board-relation matrices per leaf (model.use_dynamic_relations).
    compute_relations: bool = False
    input_history_encoding: str = "legacy"
  # Floor (seconds) of the MID-SEARCH stuck-device threshold; see
  # ``stall_threshold_s``. None = the module's ``_STALL_FLOOR_S``.
    stall_floor_s: float | None = None
  # Names used in the drop announcements (e.g. "cuda:1"); default "#k".
    device_labels: tuple[str, ...] = ()
  # False (default): any worker failing at startup fails construction and
  # every owned evaluator is retired. True: start DEGRADED on the workers that
  # built (construction still raises if every worker failed). The UCI engine
  # sets True; see SearchWorker.install_multi_gpu_pucv.
    degrade_on_startup_failure: bool = False
  # Seconds one worker's evaluator factory may go WITHOUT PROGRESS (a
  # ``startup_progress.beat()``: one warm-up shape done) before the worker is
  # timed out as a startup failure. None (default) = wait forever, the
  # historical behaviour. Time spent waiting for the compile turn another
  # worker holds does not count (``startup_progress.StartupTurn``).
    startup_timeout_s: float | None = None


_NO_ROWS = np.empty(0, dtype=np.int32)


def _distinct_leaf_rows(
    leaf_ids: np.ndarray, is_term: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of a descended batch to evaluate, and its duplicate selections.

    Returns ``(eval_rows, dup_rows, dup_first)``: ``eval_rows`` (ascending) is
    the first row selecting each distinct non-terminal leaf; ``dup_rows[k]``
    selected the same leaf as the earlier row ``dup_first[k]``. Terminal rows
    are in neither. Same leaf => same path => same position and history, so
    the same network input.
    """
    live = np.flatnonzero(is_term[:n] == 0).astype(np.int32)
    if live.size == 0:
        return live, _NO_ROWS, _NO_ROWS
    _, first_pos, inverse = np.unique(leaf_ids[live], return_index=True, return_inverse=True)
    if first_pos.size == live.size:
        return live, _NO_ROWS, _NO_ROWS
    first_row = live[first_pos]
    keep = np.zeros(live.size, dtype=bool)
    keep[first_pos] = True
    eval_rows = live[keep]
    return eval_rows, live[~keep], first_row[inverse[~keep]].astype(np.int32)


def _forwarded_rows(pending: tuple[Any, ...]) -> int:
    """Rows a prepared batch sends to the device."""
    kind = pending[0]
    if kind == "gpu_full":
        return int(pending[2])
    if kind == "gpu_miss":
        return len(pending[3])
    return 0


def _fan_out_duplicates(b: dict[str, np.ndarray], pol: np.ndarray, wdl: np.ndarray) -> None:
    """Give each duplicate selection the outputs of the leaf's first row."""
    dup_rows, dup_first = b["dup_rows"], b["dup_first"]
    if dup_rows.size:
        pol[dup_rows] = pol[dup_first]
        wdl[dup_rows] = wdl[dup_first]


@dataclass(frozen=True)
class MultiGpuPucvWorkerStats:
    worker_index: int
    batches: int = 0
    leaves: int = 0
    terminal_leaves: int = 0
    max_batch: int = 0
    wait_seconds: float = 0.0
    wall_seconds: float = 0.0
    # Rows actually sent to the device (distinct non-terminal leaves that
    # missed the eval cache) and duplicate selections answered by fan-out.
    evaluated_rows: int = 0
    duplicate_leaves: int = 0

    @property
    def avg_batch(self) -> float:
        return self.leaves / self.batches if self.batches else 0.0


@dataclass(frozen=True)
class MultiGpuPucvStats:
    target_sims: int = 0
    elapsed_seconds: float = 0.0
    workers: tuple[MultiGpuPucvWorkerStats, ...] = ()
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def batches(self) -> int:
        return sum(w.batches for w in self.workers)

    @property
    def leaves(self) -> int:
        return sum(w.leaves for w in self.workers)

    @property
    def terminal_leaves(self) -> int:
        return sum(w.terminal_leaves for w in self.workers)

    @property
    def evaluated_rows(self) -> int:
        return sum(w.evaluated_rows for w in self.workers)

    @property
    def duplicate_leaves(self) -> int:
        return sum(w.duplicate_leaves for w in self.workers)

    @property
    def avg_batch(self) -> float:
        return self.leaves / self.batches if self.batches else 0.0

    @property
    def cache_requests(self) -> int:
        return self.cache_hits + self.cache_misses

    @property
    def cache_hit_rate(self) -> float:
        requests = self.cache_requests
        if requests <= 0:
            return 0.0
        return self.cache_hits / requests


@dataclass
class _Job:
    tree: MCTSTree
    root_id: int
    root_cboard: _CBoard
    budget: threading.Semaphore
    stop_event: threading.Event
    tb_probe: SyzygyProbe | None = None


@dataclass
class _EvalJob:
    """One root evaluation routed to a single live worker's own evaluator."""
    worker: int
    xs: np.ndarray
    relations: np.ndarray | None
    done: threading.Event
    result: tuple[np.ndarray, np.ndarray] | None = None
    error: BaseException | None = None


class MultiGpuPucvPool:
    """Run a single shared MCTS search across N GPU evaluators in parallel.

    ``evaluators[k]`` must be a DirectGPUEvaluator-shaped object exposing
    ``n_slots >= 2``, ``get_input_buffer(bsz, slot)``, and
    ``evaluate_inplace_async(bsz, slot)``. Each worker uses slots 0/1 of
    its own evaluator; slots are not shared across evaluators.
    """

    def __init__(
        self,
        cfg: MultiGpuPucvConfig,
        evaluator_factories: Sequence[Callable[[], Any]] | None = None,
        *,
        evaluators: Sequence[Any] | None = None,
    ) -> None:
        """Two construction modes:

        - **Factories** (recommended): each worker thread invokes its own
          factory at startup, so the resulting evaluator's compiled model
          + cudagraph state lives on the SAME thread that will replay it.
          This is required for ``torch.compile + cudagraphs`` because
          cudagraph_trees uses thread-local state — a compiled model
          built on thread A cannot be evaluated on thread B (TLS-key
          missing assertion in cudagraph_trees.py). CUDA factories must
          also call ``torch.cuda.set_device(<their device>)`` first:
          worker threads inherit current-device ``cuda:0``, and cudagraph
          capture on devices 1..N-1 must not rely on stream contexts
          alone (``__main__.make_one`` does this; see also
          inference_threaded / train.async_eval for the same pattern).

        - **Pre-built evaluators**: only safe when the evaluator does NOT
          use cudagraphs (e.g. CPU/eager paths in tests). Validated for
          the test path; production runs with compile=on must use factories.
        """
        self._input_history_args = puct_history_args(cfg.input_history_encoding)
        if cfg.n_gpus < 1:
            raise ValueError(f"n_gpus must be >= 1, got {cfg.n_gpus}")
        if (evaluator_factories is None) == (evaluators is None):
            raise ValueError(
                "specify exactly one of evaluator_factories / evaluators",
            )
        if evaluator_factories is not None:
            if len(evaluator_factories) != cfg.n_gpus:
                raise ValueError(
                    f"need {cfg.n_gpus} factories, got {len(evaluator_factories)}",
                )
            self._factories: list[Callable[[], Any]] | None = list(evaluator_factories)
            self._evals: list[Any] = []  # populated per-thread
        else:
            assert evaluators is not None
            if len(evaluators) != cfg.n_gpus:
                raise ValueError(
                    f"need {cfg.n_gpus} evaluators, got {len(evaluators)}",
                )
            for k, ev in enumerate(evaluators):
                if not hasattr(ev, "evaluate_inplace_async"):
                    raise TypeError(
                        f"evaluator[{k}] missing evaluate_inplace_async",
                    )
                if getattr(ev, "n_slots", 1) < 2:
                    raise ValueError(
                        f"evaluator[{k}] needs n_slots >= 2 for pipelining",
                    )
            self._factories = None
            self._evals = list(evaluators)
        self._cfg = cfg

  # Per-worker wake event; completion is a condition variable over the set
  # of workers still running the current job, so a dead or stuck worker
  # can be taken out of the wait (a Barrier cannot lose a party). With
  # factories, a Barrier(n+1) gates construction until every worker has
  # built its evaluator (compile + cudagraph capture happen here, can
  # take 5-30s on first cold run depending on compile_mode).
        n = cfg.n_gpus
        self._job_lock = threading.Lock()
        self._job: _Job | None = None
        self._eval_job: _EvalJob | None = None
        self._wakes = [threading.Event() for _ in range(n)]
        self._cv = threading.Condition()
        self._running: set[int] = set()
        self._live = [True] * n
        self._abandoned = [False] * n
        self._busy_since: list[float | None] = [None] * n
        self._outstanding: list[dict[int, int]] = [{} for _ in range(n)]
        self._worker_bufs: list[list[dict[str, np.ndarray]]] = [[] for _ in range(n)]
        self._integrated = [0] * n
        self._quarantine = [_WorkerQuarantine(k) for k in range(n)]
        # Per-device slowest completed call; seeded from warmup (second pass).
        self._slowest_call_s = [0.0] * n
        # Abandoned call came back without error (set under the lock) ...
        self._late_ok = [False] * n
        # ... and its other receipts drained: the parked worker can be restored.
        self._restorable = [False] * n
        self._drop_reason: list[str | None] = [None] * n
        # Stall drops of each device within the current search (``begin_search``
        # resets them); at ``DEVICE_FLAP_LIMIT`` the device is held out.
        self._stall_drops_this_search = [0] * n
        self._held_announced = [False] * n
        self._notes: list[str] = []
        self._labels = tuple(
            cfg.device_labels[k] if k < len(cfg.device_labels) else f"#{k}"
            for k in range(n)
        )
        # Startup handshake: each worker sets its ``_built`` event once its
        # factory returned or raised; the constructor waits for all of them
        # (bounded by ``startup_timeout_s`` of NO PROGRESS per worker), decides
        # which workers start, then releases them all via ``_init_decided``.
        # Not a Barrier: a worker hung inside its factory never arrives.
        self._built = [threading.Event() for _ in range(n)]
        self._init_decided = threading.Event()
        self._startup_ctx = [StartupContext(self._labels[k]) for k in range(n)]
        self._startup_timed_out: set[int] = set()
        self._shutdown = threading.Event()
        self._errors: list[BaseException] = []
        self._retained_error: BaseException | None = None
        self._init_errors: list[BaseException] = []
        self._init_error_by_worker: dict[int, BaseException] = {}
        self._stats_lock = threading.Lock()
        self._run_started = 0.0
        self._cache = (
            PucvEvalCache(max_entries=int(cfg.eval_cache_entries))
            if cfg.eval_cache_entries > 0
            else None
        )
        self._cache_run_start = PucvEvalCacheStats(
            entries=0, hits=0, misses=0, stores=0, evictions=0,
        )
        self._last_stats = MultiGpuPucvStats(
            workers=tuple(
                MultiGpuPucvWorkerStats(worker_index=k)
                for k in range(cfg.n_gpus)
            ),
        )
        self._worker_stats = [
            MultiGpuPucvWorkerStats(worker_index=k)
            for k in range(cfg.n_gpus)
        ]
  # Pre-size the per-worker evaluator slot list when factories are used,
  # so worker threads can write their evaluator at index `k` safely.
        if self._factories is not None:
            self._evals = [None] * cfg.n_gpus
        self._threads = [
            threading.Thread(
                target=wedge.parking(self._worker_loop), args=(k,),
                name=f"pucv-gpu-{k}", daemon=True,
            )
            for k in range(cfg.n_gpus)
        ]
        for th in self._threads:
            th.start()
        self._await_startup()
        if self._init_errors:
            failed = sorted(self._init_error_by_worker)
            if not cfg.degrade_on_startup_failure or len(failed) >= n or not failed:
                # Strict construction (the class default), every device failed,
                # or a failure not tied to one worker: refuse exactly as before.
                # Every worker has already retired its own evaluator.
                self.close()
                raise self._init_errors[0]
            for k in failed:
                self._live[k] = False
            for k in failed:
                self._note_drop(k, "failed at startup", _one_line(self._init_error_by_worker[k]))

    def _await_startup(self) -> None:
        """Wait until every worker has built (or failed to build) its evaluator.

        A worker whose factory makes no progress for ``startup_timeout_s`` is
        recorded as a startup failure ("made no progress ..."), its compile
        turn is handed on, and it is abandoned: if its factory ever returns,
        the worker retires what it built and exits; until then
        ``live_worker_threads`` reports it, so the UCI process hard-exits at
        quit instead of finalizing under it. Then every worker is released."""
        timeout = self._cfg.startup_timeout_s
        try:
            while True:
                pending = [k for k, ev in enumerate(self._built) if not ev.is_set()]
                if not pending:
                    return
                if timeout is None:
                    self._built[pending[0]].wait()
                    continue
                now = time.monotonic()
                for k in pending:
                    idle = self._startup_ctx[k].idle_for(now)
                    if idle <= timeout:
                        continue
                    with self._cv:
                        if self._built[k].is_set():
                            continue  # finished just now: not a timeout
                        exc = TimeoutError(
                            f"evaluator build/warm-up made no progress for {idle:.0f}s "
                            f"(device startup timeout {timeout:g}s)"
                        )
                        self._startup_timed_out.add(k)
                        self._abandoned[k] = True
                        wedge.mark_wedged(self._threads[k], f"device {self._labels[k]} startup",
                                          device=self._labels[k])
                        self._init_errors.append(exc)
                        self._init_error_by_worker[k] = exc
                        self._built[k].set()
                    self._startup_ctx[k].revoke_turns()
                    _log.error("multi-GPU device %s timed out at startup: %s", self._labels[k], exc)
                self._built[pending[0]].wait(min(_STARTUP_POLL_S, max(0.01, timeout / 4)))
        finally:
            self._init_decided.set()

    def __enter__(self) -> MultiGpuPucvPool:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @property
    def n_devices(self) -> int:
        """Number of LIVE GPU worker evaluators driving the shared tree."""
        return sum(self._live)

    @property
    def n_configured_devices(self) -> int:
        """Workers the pool was built with, live or not."""
        return self._cfg.n_gpus

    def failed_devices(self) -> tuple[str, ...]:
        """Labels of the workers this pool has dropped."""
        return tuple(self._labels[k] for k, alive in enumerate(self._live) if not alive)

    def degraded_summary(self) -> str | None:
        """One line naming every dropped device and why; None when all are live."""
        with self._cv:
            failed = [
                f"{self._labels[k]} failed: {self._drop_reason[k] or 'unknown'}"
                for k, alive in enumerate(self._live) if not alive
            ]
            live = sum(self._live)
        if not failed:
            return None
        return (
            f"multi-GPU degraded: {live} of {self._cfg.n_gpus} devices "
            f"({'; '.join(failed)})"
        )

    def begin_search(self) -> None:
        """A new search (``go``) starts: reset the per-search flap counters."""
        with self._cv:
            self._stall_drops_this_search = [0] * self._cfg.n_gpus
            self._held_announced = [False] * self._cfg.n_gpus

    def restore_recovered_devices(self) -> tuple[str, ...]:
        """Re-admit every device dropped for a stall whose call came back clean.

        Call only between runs (the UCI engine calls it at the start of each
        search and before every chunk). The parked worker gets a fresh
        quarantine object; its late result was already discarded. Each restore
        is announced once. A device that stalled ``DEVICE_FLAP_LIMIT`` times
        in the current search stays out until ``begin_search`` (announced
        once)."""
        if not any(self._restorable):
            return ()  # healthy path: no lock, no allocation beyond the scan
        restored: list[int] = []
        with self._cv:
            if self._shutdown.is_set():
                return ()
            for k in range(self._cfg.n_gpus):
                if not (self._restorable[k] and not self._live[k]):
                    continue
                if not self._threads[k].is_alive():
                    continue
                if self._stall_drops_this_search[k] >= DEVICE_FLAP_LIMIT:
                    if not self._held_announced[k]:
                        self._held_announced[k] = True
                        msg = DEVICE_HELD_NOTE.format(
                            label=self._labels[k], count=self._stall_drops_this_search[k],
                            live=sum(self._live), total=self._cfg.n_gpus,
                        )
                        _log.warning("%s", msg)
                        self._notes.append(msg)
                    continue
                self._restorable[k] = False
                self._late_ok[k] = False
                self._abandoned[k] = False
                self._busy_since[k] = None
                self._outstanding[k].clear()
                # The drain succeeded: release (not replace) this worker's
                # quarantine, so repeated stall/restore cycles do not pile up
                # entries in the process-wide quarantine list.
                release_quarantine(self._quarantine[k])
                self._drop_reason[k] = None
                self._live[k] = True
                restored.append(k)
            for k in restored:
                msg = DEVICE_RESTORED_NOTE.format(
                    label=self._labels[k], live=sum(self._live), total=self._cfg.n_gpus,
                )
                _log.warning("%s", msg)
                self._notes.append(msg)
        return tuple(self._labels[k] for k in restored)

    def drain_notes(self) -> list[str]:
        """Pop the pending device-drop announcements (each is produced once)."""
        with self._cv:
            notes, self._notes = self._notes, []
        return notes

    def _slowest(self, idx: int | None) -> float:
        return max(self._slowest_call_s) if idx is None else self._slowest_call_s[idx]

    def stall_threshold_s(self, idx: int | None = None) -> float:
        """Seconds one device call may block mid-search before its worker is
        abandoned (device ``idx``; None = the most patient device's value)."""
        floor = self._cfg.stall_floor_s
        return max(
            float(_STALL_FLOOR_S if floor is None else floor),
            min(_STALL_CEILING_S, _STALL_SLOWEST_CALL_FACTOR * self._slowest(idx)),
        )

    def stop_stall_threshold_s(self, idx: int | None = None) -> float:
        """The shorter threshold once nothing waits on the stuck worker (the job
        is stopped, or every other device is idle); never above
        ``_STOP_STALL_CEILING_S``."""
        return min(
            _STOP_STALL_CEILING_S,
            self.stall_threshold_s(idx),
            max(_STOP_STALL_FLOOR_S, _STOP_STALL_SLOWEST_CALL_FACTOR * self._slowest(idx)),
        )

    def idle_stall_threshold_s(self, idx: int | None = None) -> float:
        """Threshold while every other live device is idle on this job."""
        return max(_IDLE_SURVIVOR_STALL_FLOOR_S, self.stop_stall_threshold_s(idx))

    @property
    def gather(self) -> int:
        """Per-worker leaf gather: the most tokens a worker grabs per batch."""
        return max(1, int(self._cfg.gather))

    def close(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        for ev in self._wakes:
            ev.set()
        abandoned_deadline = time.monotonic() + _CLOSE_ABANDONED_JOIN_S
        stuck = [k for k in range(len(self._threads))
                 if self._abandoned[k] and not self._restorable[k]]
        for k in stuck:
            # An abandoned worker means a device fault this session, after
            # which ending workers park instead of exiting (``wedge``): there
            # is nothing to join. (Without the latch -- tests -- a call that
            # returns shortly lets it end within a short shared budget.)
            wedge.join_unless_wedged(
                self._threads[k], max(0.0, abandoned_deadline - time.monotonic()))
        for k, th in enumerate(self._threads):
            if k not in stuck:
                # After a device fault, ending workers park: no join.
                wedge.join_unless_wedged(th, 2.0)

    def worker_row_cap(self) -> int:
        """The most rows ``evaluate_rows`` may submit: the gather, which every
        worker warmed (and armed its shape guard for), capped by each live
        worker's ``max_batch``."""
        cap = self.gather
        for k, alive in enumerate(self._live):
            if not alive or k >= len(self._evals) or self._evals[k] is None:
                continue
            mb = getattr(self._evals[k], "max_batch", getattr(self._evals[k], "_max_batch", None))
            if isinstance(mb, int) and not isinstance(mb, bool) and mb > 0:
                cap = min(cap, mb)
        return max(1, cap)

    def live_worker_threads(self) -> list[threading.Thread]:
        """Worker threads still alive (after ``close``: wedged in a device call).

        The UCI process hard-exits after its normal shutdown when any remain,
        instead of finalizing the interpreter under a thread inside torch."""
        return [th for th in self._threads if th.is_alive()]

    # -- failure bookkeeping (callers hold no lock unless named *_locked) --

    def _note_drop(self, idx: int, what: str, reason: str) -> None:
        wedge.note_device_fault(f"device {self._labels[idx]} {what}")
        msg = DEVICE_DROPPED_NOTE.format(
            label=self._labels[idx], what=what, reason=reason,
            live=sum(self._live), total=self._cfg.n_gpus,
        )
        _log.error("%s", msg)
        with self._cv:
            self._drop_reason[idx] = f"{what} ({reason})"
            self._notes.append(msg)

    def _device_call_begin(self, idx: int) -> None:
        with self._cv:
            if self._abandoned[idx]:
                raise _Abandoned()
            self._busy_since[idx] = time.monotonic()

    def _device_call_end(self, idx: int, *, failed: bool) -> None:
        """Close a device call; raise ``_Abandoned`` if the pool gave up on it.

        Runs under the same lock as the abandonment, so either this worker
        resumes (and owns its outstanding cleanup) or the run thread already
        released that cleanup and this worker never touches the tree again.
        An abandoned call that came back WITHOUT error (``failed=False``) is
        recorded as a candidate for restore."""
        with self._cv:
            since = self._busy_since[idx]
            self._busy_since[idx] = None
            if self._abandoned[idx]:
                wedge.clear_wedged(threading.current_thread())  # the call came back
                if not failed:
                    self._late_ok[idx] = True
                raise _Abandoned()
            if since is not None and not failed:
                took = time.monotonic() - since
                if took > self._slowest_call_s[idx]:
                    self._slowest_call_s[idx] = took

    @contextmanager
    def _device_call(self, idx: int):
        """Bracket ONE blocking device call for the stuck-device watch."""
        self._device_call_begin(idx)
        try:
            yield
        except BaseException:
            self._device_call_end(idx, failed=True)
            raise
        self._device_call_end(idx, failed=False)

    def _drain_failed(
        self, idx: int, evaluator: Any, receipts: list[Any], error: BaseException,
    ) -> None:
        """Keep failure cleanup visible to the same stuck-device watchdog.

        A completion can raise and its failure drain can then hang. The normal
        call bracket has already ended at that point. This bracket always ends
        as FAILED: even a clean late drain must not restore a device that raised.
        """
        try:
            self._device_call_begin(idx)
        except _Abandoned:
            poison(self._quarantine[idx], evaluator, receipts, error)
            raise
        try:
            drain_failed_receipts(evaluator, self._quarantine[idx], receipts, error)
        finally:
            self._device_call_end(idx, failed=True)

    def _wake_idle_for_budget_locked(self, job: _Job) -> None:
        """Re-enrol survivors after a dropped worker returns the final tokens.

        Workers use nonblocking grabs and may already be asleep when a failed
        peer releases its budget. The coordinator calls this under ``_cv``
        before declaring completion, including after abandonment. Peeking takes
        and immediately returns one token; only workers actually consume it.
        """
        if (self._job is not job or self._retained_error is not None
                or job.stop_event.is_set() or self._shutdown.is_set()):
            return
        idle = [k for k, alive in enumerate(self._live) if alive and k not in self._running]
        if not idle or not job.budget.acquire(blocking=False):
            return
        job.budget.release()
        for k in idle:
            self._running.add(k)
            self._wakes[k].set()

    def _park_recovered(self, idx: int) -> bool:
        """After ``_Abandoned``: park this worker for restore if its late call
        came back clean (True = keep the thread, False = it exits)."""
        with self._cv:
            if self._late_ok[idx] and not self._shutdown.is_set():
                self._restorable[idx] = True
                return True
            return False

    def _drain_for_restore(self, idx: int, receipts: list[Any]) -> None:
        """A late-returned abandoned worker must prove the rest of its in-flight
        work finished before it can ever be restored; any failure keeps it out."""
        try:
            for receipt in receipts:
                if receipt is None:
                    continue
                event = receipt[2]
                if event is not None:
                    event.synchronize()
        except BaseException:
            _log.warning("abandoned device %s failed to drain; it stays dropped",
                         self._labels[idx], exc_info=True)
            with self._cv:
                self._late_ok[idx] = False

    def _abandon_stuck_locked(self, job: _Job | None) -> None:
        """Abandon every running worker stuck in one device call too long.

        Three thresholds:
          - job stopped: ``stop_stall_threshold_s`` (<= 1 s);
          - IDLE SURVIVOR -- some live device has finished this job and sits
            idle, and EVERY worker still running is inside a call older than
            ``idle_stall_threshold_s`` (>= 1 s): ``run`` cannot return while a
            worker is stuck, so mid-search patience would only burn the move;
          - otherwise (someone is still making progress): the mid-search
            ``stall_threshold_s`` (>= 5 s).
        The survivors cannot instead take over a stuck worker's leaves while it
        might still answer -- its batch is under virtual loss in the tree and
        its late answer would integrate twice -- so abandoning it (which
        releases that virtual loss under this lock and hands its tokens back)
        IS the take-over. A healthy call slower than 1 s that lands when the
        others are idle is dropped this way too, and re-admitted at the next
        chunk (``restore_recovered_devices``) once it answers. With no idle
        survivor (one device left) the long threshold applies: dropping it
        gains nothing."""
        stopped = job is not None and job.stop_event.is_set()
        now = time.monotonic()
        idle_survivor = any(
            alive and j not in self._running for j, alive in enumerate(self._live)
        )
        nobody_progressing = idle_survivor and bool(self._running) and all(
            self._busy_since[k] is not None
            and now - cast(float, self._busy_since[k]) > self.idle_stall_threshold_s(k)
            for k in self._running
        )
        for k in sorted(self._running):
            since = self._busy_since[k]
            if stopped:
                limit = self.stop_stall_threshold_s(k)
            elif nobody_progressing:
                limit = self.idle_stall_threshold_s(k)
            else:
                limit = self.stall_threshold_s(k)
            if since is None or now - since <= limit:
                continue
            self._abandoned[k] = True
            self._late_ok[k] = False
            self._live[k] = False
            self._stall_drops_this_search[k] += 1
            wedge.mark_wedged(self._threads[k], f"device {self._labels[k]} call",
                              device=self._labels[k])
            self._running.discard(k)
            outstanding = self._outstanding[k]
            if job is not None and outstanding:
                in_flight = sum(outstanding.values())
                try:
                    _release_outstanding(
                        job.tree, self._worker_bufs[k], outstanding,
                        self._cfg.vloss_weight, None,
                    )
                except BaseException:
                    _log.exception("releasing a stuck worker's virtual loss failed")
                outstanding.clear()
                if not job.stop_event.is_set():
                    job.budget.release(in_flight)
            evaluator = self._evals[k] if k < len(self._evals) else None
            poison(self._quarantine[k], evaluator, self._worker_bufs[k])
            self._worker_stats[k] = replace(self._worker_stats[k], leaves=self._integrated[k])
            reason = f"one device call blocked for {now - since:.1f}s > {limit:.1f}s"
            self._drop_reason[k] = f"stopped answering ({reason})"
            msg = DEVICE_DROPPED_NOTE.format(
                label=self._labels[k], what="stopped answering", reason=reason,
                live=sum(self._live), total=self._cfg.n_gpus,
            )
            _log.error("%s", msg)
            self._notes.append(msg)

    def _fail_worker(self, idx: int, exc: BaseException, job: _Job | None) -> bool:
        """A worker's search raised: drop it, keep the pool going on the rest.

        Returns True when the worker is RETAINED instead: its failure was
        quiesced (its evaluator is still usable) and no other worker is live.
        Then there is nothing to degrade onto, so the run re-raises the error
        and the next run reuses the device -- the pre-degrade semantics."""
        with self._cv:
            if self._abandoned[idx]:
                return False
            wedge.note_device_fault(f"device {self._labels[idx]} raised")
            self._errors.append(exc)
            in_flight = sum(self._outstanding[idx].values())
            self._outstanding[idx].clear()
            others = any(alive for k, alive in enumerate(self._live) if k != idx)
            if not others and not _is_poisoned(self._quarantine[idx]):
                self._retained_error = exc
                return True
            self._live[idx] = False
        if job is not None and in_flight and not job.stop_event.is_set():
            job.budget.release(in_flight)
        if _is_poisoned(self._quarantine[idx]) and not any(self._live):
            # The last worker died with an unquiesced evaluator: the pool as a
            # whole is now unusable, exactly as before the degrade path.
            poison(self, self._quarantine[idx], exc)
        self._note_drop(idx, "failed mid-search", _one_line(exc))
        return False

    def last_stats(self) -> MultiGpuPucvStats:
        """Return stats from the most recent completed run.

        The pool mutates counters only from worker threads and snapshots them
        after the completion barrier, so callers get a stable immutable view.
        """
        with self._stats_lock:
            return self._last_stats

    def run(
        self,
        *,
        tree: MCTSTree,
        root_id: int,
        root_cboard: _CBoard,
        target_sims: int,
        stop_event: threading.Event,
        tb_probe: SyzygyProbe | None = None,
    ) -> int:
        """Run up to ``target_sims`` sims; returns the count actually
        COMPLETED (== target unless ``stop_event`` fired mid-run). Callers
        use the return for node/nps accounting, so returning the target on
        an early stop would inflate the nps estimate that sizes the next
        time-capped chunk."""
        ensure_usable(self)
        live = [k for k, alive in enumerate(self._live) if alive]
        if not live:
            raise RuntimeError(
                f"{ALL_DEVICES_FAILED} ({', '.join(self.failed_devices())})"
            )
        for k in live:
            ensure_usable(self._evals[k])
        if target_sims <= 0:
            return 0
        if self._shutdown.is_set():
            raise RuntimeError("MultiGpuPucvPool is closed")
  # Workers descend lock-free while others expand under tree_lock: an arena
  # realloc mid-run is a use-after-free in the descending thread.
        reserve_concurrent_headroom(tree, target_sims)

        with self._job_lock:
            self._errors = []
            self._retained_error = None
            self._run_started = time.perf_counter()
            self._worker_stats = [
                MultiGpuPucvWorkerStats(worker_index=k)
                for k in range(self._cfg.n_gpus)
            ]
            self._integrated = [0] * self._cfg.n_gpus
            if self._cache is not None:
                self._cache_run_start = self._cache.stats()
            job = _Job(
                tree=tree, root_id=root_id, root_cboard=root_cboard,
                budget=threading.Semaphore(target_sims),
                stop_event=stop_event, tb_probe=tb_probe,
            )
            self._job = job
            with self._cv:
                self._running = set(live)
        for k in live:
            self._wakes[k].set()
        with self._cv:
            while True:
                self._wake_idle_for_budget_locked(job)
                if not self._running:
                    break
                self._cv.wait(timeout=_RUN_POLL_S)
                self._abandon_stuck_locked(job)

        if self._retained_error is not None:
            raise self._retained_error
        if not any(self._live):
            if self._errors:
                raise self._errors[0]
            raise RuntimeError(
                f"{ALL_DEVICES_FAILED} ({', '.join(self.failed_devices())})"
            )
        elapsed = time.perf_counter() - self._run_started
        cache_hits = 0
        cache_misses = 0
        if self._cache is not None:
            cache_after = self._cache.stats()
            cache_hits = cache_after.hits - self._cache_run_start.hits
            cache_misses = cache_after.misses - self._cache_run_start.misses
        with self._stats_lock:
            self._last_stats = MultiGpuPucvStats(
                target_sims=target_sims,
                elapsed_seconds=elapsed,
                workers=tuple(self._worker_stats),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )
        _log.debug(
            "multi_gpu_pucv target=%d leaves=%d batches=%d avg_batch=%.1f "
            "wall=%.3fs worker_leaves=%s",
            target_sims,
            self._last_stats.leaves,
            self._last_stats.batches,
            self._last_stats.avg_batch,
            elapsed,
            [w.leaves for w in self._last_stats.workers],
        )
  # Every budget token a worker acquired became exactly one descended +
  # integrated leaf (terminal or GPU-evaluated), so the summed worker
  # leaves ARE the completed sims. Tokens never acquired (early stop)
  # are correctly excluded.
        return self._last_stats.leaves

    def evaluate_root(
        self, xs: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a (1, C, 8, 8) root on a surviving worker's own evaluator.

        The degrade path for the PRIMARY evaluator (root evals run on
        ``devices[0]`` outside the pool): when it fails, the search can still
        get its root prior from any live device. Runs on the worker's own
        thread (cudagraph thread affinity). Call only between ``run`` calls.
        A worker that raises or stays stuck past ``stall_threshold_s`` is
        dropped and the next live one is tried; raises once none is left.
        """
        return self.evaluate_rows(xs, relations)

    def evaluate_rows(
        self, xs: np.ndarray, relations: np.ndarray | None = None, *,
        idle_limit: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``evaluate_root`` for any (B, C, 8, 8) batch, B <= ``gather``.

        Used by the classic-Gumbel ``searchmoves`` search once the primary
        evaluator has failed (``PoolRowsEvaluator``). ``idle_limit``: nothing
        else progresses while the caller waits, so a stuck worker is dropped at
        the idle-survivor threshold (1 s) when another worker is live, as in a
        search; otherwise the mid-search threshold, as for root evaluations.
        """
        ensure_usable(self)
        last: BaseException | None = None
        for k in range(self._cfg.n_gpus):
            if not self._live[k]:
                continue
            job = _EvalJob(worker=k, xs=xs, relations=relations, done=threading.Event())
            self._eval_job = job
            stuck = False
            try:
                self._wakes[k].set()
                others = any(alive for j, alive in enumerate(self._live) if j != k)
                limit = (
                    self.idle_stall_threshold_s(k) if idle_limit and others
                    else self.stall_threshold_s(k)
                )
                while not job.done.wait(_RUN_POLL_S):
                    if not self._threads[k].is_alive():
                        self._live[k] = False
                        last = RuntimeError(f"device {self._labels[k]} worker exited")
                        break
                    with self._cv:
                        since = self._busy_since[k]
                        # Same rule as a search: only a worker still INSIDE its
                        # device call is abandoned, under the lock it re-checks.
                        if (since is not None and not job.done.is_set()
                                and time.monotonic() - since > limit):
                            self._abandoned[k] = True
                            self._late_ok[k] = False
                            self._live[k] = False
                            wedge.mark_wedged(self._threads[k], f"device {self._labels[k]} call",
                              device=self._labels[k])
                            stuck = True
                            break
            finally:
                self._eval_job = None
            if stuck:
                poison(self._quarantine[k], self._evals[k])
                self._note_drop(
                    k, "stopped answering",
                    f"evaluation blocked for more than {limit:.1f}s",
                )
                last = TimeoutError(f"device {self._labels[k]} root evaluation stuck")
                continue
            if not job.done.is_set():
                continue
            if job.error is not None:
                last = job.error
                continue
            assert job.result is not None
            return job.result
        raise RuntimeError(
            f"{ALL_DEVICES_FAILED} ({', '.join(self.failed_devices())})"
        ) from last

    def _run_eval_job(self, idx: int, job: _EvalJob, evaluator: Any) -> bool:
        """Run ``job`` on this worker. False = this worker is now retired."""
        receipt = None
        try:
            bsz = int(job.xs.shape[0])
            buf = evaluator.get_input_buffer(bsz, slot=0)
            buf_np = buf.numpy() if hasattr(buf, "numpy") else np.asarray(buf)
            buf_np.reshape(job.xs.shape)[...] = job.xs
            with self._device_call(idx):
                if job.relations is None:
                    receipt = submit_async(evaluator, bsz, slot=0)
                else:
                    receipt = submit_async(evaluator, bsz, slot=0, relations=job.relations)
                pol_t, wdl_t, event = receipt
                if event is not None:
                    event.synchronize()
            pol = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
            wdl = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
            job.result = (
                np.array(pol[:bsz], dtype=np.float32), np.array(wdl[:bsz], dtype=np.float32),
            )
            return True
        except _Abandoned:
            # The result is discarded (never handed to the caller); a clean
            # late answer parks the worker for restore.
            return self._park_recovered(idx)
        except BaseException as exc:
            if receipt is not None:
                try:
                    self._drain_failed(idx, evaluator, [receipt], exc)
                except _Abandoned:
                    return False
            with self._cv:
                self._live[idx] = False
            self._note_drop(idx, "failed an evaluation", _one_line(exc))
            job.error = exc
            return False
        finally:
            job.done.set()

    def _worker_loop(self, idx: int) -> None:
        cfg = self._cfg
        gather = 1
        my_wake = self._wakes[idx]

        evaluator: Any = None
        bufs: list[dict[str, np.ndarray]] = []
        try:
            gather = max(1, int(cfg.gather))
            if self._factories is not None:
                earlier = wedge.wedged_earlier(self._labels[idx])
                if earlier is not None:
                    # A rebuild must not build on a device that already wedged
                    # this session (its build would likely hang the whole
                    # startup timeout before readyok): it is simply left out.
                    raise RuntimeError(f"left out: it wedged earlier this session ({earlier})")
  # Build evaluator on THIS thread so torch.compile + cudagraph
  # state lives where it'll be replayed. Crucial: compile builds
  # tree_manager_containers in TLS on first forward; calling from
  # another thread asserts in cudagraph_trees.py:get_obj.
  # The startup context lets the factory's warm-up report progress (and
  # wait for the compile turn without its no-progress clock running).
                with startup_context(self._startup_ctx[idx]):
                    evaluator = self._factories[idx]()
                if not hasattr(evaluator, "evaluate_inplace_async"):
                    raise TypeError(
                        f"factory[{idx}] returned an evaluator missing "
                        "evaluate_inplace_async",
                    )
                if getattr(evaluator, "n_slots", 1) < 2:
                    raise ValueError(
                        f"factory[{idx}] returned an evaluator with "
                        "n_slots < 2",
                    )
                self._evals[idx] = evaluator
            else:
                evaluator = self._evals[idx]
            seed = getattr(evaluator, "warm_steady_call_s", None)
            if isinstance(seed, (int, float)) and not isinstance(seed, bool) and seed > 0:
                # Steady-state warmup timing (second pass only; see
                # ``_warmup_pucv_evaluator``) -- the baseline the stall check
                # scales, so the first real calls are not judged against 0.
                # Seeded BEFORE the construction barrier: a run may follow it.
                with self._cv:
                    self._slowest_call_s[idx] = max(self._slowest_call_s[idx], float(seed))
      # Per-worker buffer ping-pong (one set per slot). Encoding lands
      # directly in the evaluator's pinned input via get_input_buffer.
            bufs = [
                _alloc_buffers(
                    gather, int(cfg.input_planes),
                    with_relations=bool(cfg.compute_relations),
                ),
                _alloc_buffers(
                    gather, int(cfg.input_planes),
                    with_relations=bool(cfg.compute_relations),
                ),
            ]
        except BaseException as exc:
            with self._cv:
                if idx not in self._startup_timed_out:
                    self._init_errors.append(exc)
                    self._init_error_by_worker[idx] = exc
        # A build timed out as hung that came back is no longer wedged (if it
        # retires below while another device is wedged, it parks: ``wedge``).
        wedge.clear_wedged(threading.current_thread())
        with self._cv:
            self._built[idx].set()
        self._init_decided.wait()
        # Strict construction retires every worker on any failure. With
        # ``degrade_on_startup_failure`` only THIS worker's failure retires it
        # and the others keep running (construction raises only if all failed).
        # A worker timed out at startup retires even if its factory came back.
        with self._cv:
            retire = idx in self._init_error_by_worker or (
                bool(self._init_errors) and not cfg.degrade_on_startup_failure
            )
        if retire:
            if self._factories is not None:
                self._evals[idx] = None
                close = getattr(evaluator, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException as cleanup:
                        note_cleanup(self._init_errors[0], f"startup evaluator cleanup failed: {cleanup!r}")
                evaluator = None
            bufs.clear()
            return
        self._worker_bufs[idx] = bufs

        while True:
            my_wake.wait()
            my_wake.clear()
            if self._shutdown.is_set():
                return
            if not self._live[idx]:
                if self._restorable[idx]:
                    continue  # parked after a clean late answer; awaiting restore
                return
            eval_job = self._eval_job
            if eval_job is not None and eval_job.worker == idx:
                if not self._run_eval_job(idx, eval_job, evaluator):
                    return  # abandoned or failed: this worker is retired
                continue
            job = self._job
            if job is None:
                continue
            try:
                self._pipeline_until_done(idx, job, evaluator, bufs, gather)
            except _Abandoned:
                # The run thread already released this worker's virtual loss
                # and stopped waiting for it; touch nothing. A clean late
                # answer parks the thread for restore, otherwise it leaves.
                if self._park_recovered(idx):
                    continue
                return
            except BaseException as exc:
                _log.exception("pucv-gpu-%d raised", idx)
                retained = self._fail_worker(idx, exc, job)
                with self._cv:
                    self._running.discard(idx)
                    self._cv.notify_all()
                if not retained:
                    return
                continue
            with self._cv:
                self._running.discard(idx)
                self._cv.notify_all()

    def _pipeline_until_done(
        self,
        idx: int,
        job: _Job,
        evaluator: Any,
        bufs: list[dict[str, np.ndarray]],
        gather: int,
    ) -> None:
        ensure_usable(self)
        ensure_usable(evaluator)
        cfg = self._cfg
        c_puct = cfg.c_puct
        fpu_root = cfg.fpu_at_root
        fpu_red = cfg.fpu_reduction
        vloss = cfg.vloss_weight
        vloss_mode = cfg.vloss_mode
        tree = job.tree
        root_id = job.root_id
        root_cb = job.root_cboard
        budget = job.budget
        stop_event = job.stop_event
        tree.set_cpuct_scaling(float(cfg.cpuct_factor), float(cfg.cpuct_base))

        pending: Any = None
        batches = 0
        terminal_leaves = 0
        evaluated_rows = 0
        duplicate_leaves = 0
        max_batch = 0
        wait_seconds = 0.0
        wall_start = time.perf_counter()
  # buf index -> row count for a batch whose vloss is applied and not yet
  # integrated. batch_descend_puct applies it, batch_integrate_leaves
  # removes it, and the evaluator between them can raise; the tree outlives
  # this call, so an unguarded escape biases it permanently
  # (SHARED_BROKER_AUDIT B7 -- reproduced on the WalkerPool twin).
  # Shared with the run thread, which releases it if this worker is abandoned
  # inside a stuck device call (see ``_abandon_stuck_locked``).
        outstanding: dict[int, int] = self._outstanding[idx]
        outstanding.clear()
        receipts: dict[int, Any] = {}
        tb_batches: dict[int, LeafSnapshots | None] = {}
        # Own the exact async result before pending metadata or registration can fail.
        unregistered_receipt: list[Any] = [None]
        next_local = -1
        try:
            while not stop_event.is_set() or pending is not None:
      # Greedy budget grab: take up to ``gather`` tokens. Faster GPUs
      # naturally take more leaves over the search.
                n = 0
                if not stop_event.is_set():
                    while n < gather and budget.acquire(blocking=False):
                        n += 1
                if n == 0 and pending is None:
                    break

                next_local = -1
                if n > 0:
                    next_local = 0 if pending is None else 1 - int(pending[1])
                    next_slot = next_local  # slot 0/1 on this worker's evaluator
                    inp = evaluator.get_input_buffer(n, slot=next_slot)
                    inp_np = inp.numpy() if hasattr(inp, "numpy") else inp
                    enc_view = np.asarray(inp_np).reshape(n, int(cfg.input_planes), 8, 8)
                    b = bufs[next_local]
                    rel = b["rel"] if cfg.compute_relations else None
                    snapshots = descend_batch(
                        tree, root_id, root_cb, n, c_puct, fpu_root, fpu_red, vloss,
                        enc_view, b, vloss_mode,
                        b["cache_keys"] if self._cache is not None else None,
                        rel, self._input_history_args, job.tb_probe,
                    )
                    outstanding[next_local] = n
                    tb_batches[next_local] = snapshots
                    batches += 1
                    max_batch = max(max_batch, n)
                    n_term = int(b["is_term"][:n].sum())
                    terminal_leaves += n_term
                    with self._device_call(idx):
                        next_handle = self._prepare_submit(
                            evaluator, next_slot, next_local, n, enc_view, b, rel,
                            n_term=n_term, receipt_cell=unregistered_receipt,
                        )
                else:
                    next_handle = None

                if next_handle is not None:
                    evaluated_rows += _forwarded_rows(next_handle)
                    duplicate_leaves += int(bufs[next_local]["dup_rows"].size)
                if next_handle is not None and next_handle[0] != "cache_only":
                    receipts[next_local] = pending_receipt(next_handle)
                    unregistered_receipt[0] = None

                if pending is not None:
                    # Only the WAIT is the device call. Everything that uses
                    # the result -- the eval-cache put, the tree integration
                    # -- runs after ``_device_call`` has confirmed the pool did
                    # not abandon this worker, so a late answer from a dying
                    # device can never land in the shared cache or the tree.
                    with self._device_call(idx):
                        self._await_pending(pending)
                    b_cur, p_n, pol, wdl, submit_t = self._finish_pending(
                        pending, bufs,
                    )
                    receipts.pop(int(pending[1]), None)
                    if submit_t > 0.0:
                        wait_seconds += time.perf_counter() - submit_t
                    wdl, solved = override_leaf_values(
                        job.tb_probe, tb_batches.get(int(pending[1])), wdl,
                    )
                    tree.batch_integrate_leaves(
                        p_n,
                        b_cur["path_buf"], b_cur["path_lens"],
                        b_cur["legal_buf"], b_cur["legal_lens"],
                        b_cur["is_term"], pol, wdl, vloss,
                    )
                    outstanding.pop(int(pending[1]), None)
                    self._integrated[idx] += int(p_n)
                    tb_batches.pop(int(pending[1]), None)
                    mark_batch_solved(tree, b_cur, solved)

                pending = next_handle
        except _Abandoned:
            # Stuck device: the run thread already quarantined the evaluator
            # and released the virtual loss (``outstanding`` is empty now).
            # Only a call that came back clean drains its other receipts (the
            # device just answered); a still-stuck one is never drained here.
            if self._late_ok[idx]:
                self._drain_for_restore(idx, [*receipts.values(), unregistered_receipt[0]])
            raise
        except BaseException as error:
            failed_receipts: list[Any] = []
            if unregistered_receipt[0] is not None and receipts.get(next_local) is not unregistered_receipt[0]:
                failed_receipts.append(unregistered_receipt[0])
            failed_receipts.extend(receipts.values())
            self._drain_failed(idx, evaluator, failed_receipts, error)
            raise
        finally:
            _release_outstanding(tree, bufs, outstanding, vloss, sys.exc_info()[1])
            if not self._abandoned[idx]:
                # A failed worker reports what it INTEGRATED: its in-flight
                # leaves never landed and their tokens went back to the budget.
                previous = self._worker_stats[idx]
                self._worker_stats[idx] = MultiGpuPucvWorkerStats(
                    worker_index=idx,
                    batches=previous.batches + batches,
                    leaves=self._integrated[idx],
                    terminal_leaves=previous.terminal_leaves + terminal_leaves,
                    evaluated_rows=previous.evaluated_rows + evaluated_rows,
                    duplicate_leaves=previous.duplicate_leaves + duplicate_leaves,
                    max_batch=max(previous.max_batch, max_batch),
                    wait_seconds=previous.wait_seconds + wait_seconds,
                    wall_seconds=previous.wall_seconds + time.perf_counter() - wall_start,
                )

    def _prepare_submit(
        self,
        evaluator: Any,
        slot: int,
        buf_idx: int,
        n: int,
        enc_view: np.ndarray,
        b: dict[str, np.ndarray],
        rel: np.ndarray | None,
        *,
        n_term: int,
        receipt_cell: list[Any] | None = None,
    ) -> tuple[Any, ...]:
  # All-terminal batches have nothing to evaluate: descend already
  # backpropped the terminal leaves inline, so submitting would run the
  # GPU on zeroed rows whose outputs are discarded. The cache_only
  # pending shape with no hits materializes zero outputs, and
  # batch_integrate_leaves skips terminal rows regardless. (The cache-on
  # path below reaches the same result via its empty miss list.)
        if n_term >= n:
            b["dup_rows"], b["dup_first"] = _NO_ROWS, _NO_ROWS
            return ("cache_only", buf_idx, n, {}, 0.0)
        # Only the first selection of each distinct leaf is evaluated: terminal
        # rows (already backpropped by the descent) are left out of the forward,
        # and a leaf several descents selected in this batch (virtual loss does
        # not prevent it) is evaluated once, its output copied to every later
        # row (``_fan_out_duplicates``) -- the integration is unchanged.
        eval_rows, b["dup_rows"], b["dup_first"] = _distinct_leaf_rows(
            b["leaf_ids"], b["is_term"], n,
        )
        cache = self._cache
        if cache is None and eval_rows.size < n:
            m = int(eval_rows.size)
            enc_view[:m] = enc_view[eval_rows]
            if rel is None:
                handle = submit_async(evaluator, m, slot=slot)
            else:
                rel[:m] = rel[eval_rows]
                handle = submit_async(evaluator, m, slot=slot, relations=rel[:m])
            if receipt_cell is not None:
                receipt_cell[0] = handle
            return ("gpu_miss", buf_idx, n, eval_rows, {}, handle, time.perf_counter())
        if cache is None:
            if rel is None:
                handle = submit_async(evaluator, n, slot=slot)
            else:
                handle = submit_async(evaluator,
                    n, slot=slot, relations=rel[:n],
                )
            if receipt_cell is not None:
                receipt_cell[0] = handle
            return ("gpu_full", buf_idx, n, handle, time.perf_counter())

        hit_values: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        miss_indices: list[int] = []
        for i in eval_rows.tolist():
            hit = cache.get(b["cache_keys"][i], enc_view[i])
            if hit is None:
                miss_indices.append(i)
            else:
                hit_values[i] = hit

        if not miss_indices:
            return ("cache_only", buf_idx, n, hit_values, 0.0)

        miss_idx = np.asarray(miss_indices, dtype=np.int32)
        miss_n = int(miss_idx.size)
        b["enc_rows"][:n] = enc_view[:n]
        enc_view[:miss_n] = enc_view[miss_idx]
        if rel is None:
            handle = submit_async(evaluator, miss_n, slot=slot)
        else:
            # Compact relation rows alongside the encoding rows. Cache hits
            # stay valid: relations are a pure function of the position, so
            # position-keyed cached outputs already include the bias effect.
            rel[:miss_n] = rel[miss_idx]
            handle = submit_async(evaluator,
                miss_n, slot=slot, relations=rel[:miss_n],
            )
        if receipt_cell is not None:
            receipt_cell[0] = handle
        return (
            "gpu_miss", buf_idx, n, miss_idx, hit_values,
            handle,
            time.perf_counter(),
        )

    @staticmethod
    def _await_pending(pending: tuple[Any, ...]) -> None:
        """Block until ``pending``'s device work is done (the device call)."""
        if pending[0] == "cache_only":
            return
        evt = pending_receipt(pending)[2]
        if evt is not None:
            evt.synchronize()

    def _finish_pending(
        self,
        pending: tuple[Any, ...],
        bufs: list[dict[str, np.ndarray]],
    ) -> tuple[dict[str, np.ndarray], int, np.ndarray, np.ndarray, float]:
        """Read back an ALREADY-AWAITED pending batch (``_await_pending``)."""
        kind = pending[0]
        if kind == "gpu_full":
            _, buf_idx, n, handle, submit_t = pending
            pol_t, wdl_t, _evt = handle
            pol = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
            wdl = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
            return bufs[buf_idx], int(n), pol, wdl, float(submit_t)

        if kind == "cache_only":
            _, buf_idx, n, hit_values, submit_t = pending
            pol, wdl = self._materialize_cached_outputs(int(n), hit_values, None, None)
            _fan_out_duplicates(bufs[buf_idx], pol, wdl)
            return bufs[buf_idx], int(n), pol, wdl, float(submit_t)

        _, buf_idx, n, miss_idx, hit_values, handle, submit_t = pending
        pol_t, wdl_t, _evt = handle
        pol_m = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
        wdl_m = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
        b = bufs[buf_idx]
        cache = self._cache
        if cache is not None:
            for j, original_i in enumerate(miss_idx):
                i = int(original_i)
                cache.put(b["cache_keys"][i], b["enc_rows"][i], pol_m[j], wdl_m[j])
        pol, wdl = self._materialize_cached_outputs(
            int(n), hit_values, miss_idx, (pol_m, wdl_m),
        )
        _fan_out_duplicates(b, pol, wdl)
        return b, int(n), pol, wdl, float(submit_t)

    def _materialize_cached_outputs(
        self,
        n: int,
        hit_values: dict[int, tuple[np.ndarray, np.ndarray]],
        miss_idx: np.ndarray | None,
        miss_outputs: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        first_value = next(iter(hit_values.values()), None)
        if first_value is None and miss_outputs is not None:
            first_value = (miss_outputs[0][0], miss_outputs[1][0])
        if first_value is None:
            return (
                np.zeros((n, 4672), dtype=np.float32),
                np.zeros((n, 3), dtype=np.float32),
            )
        pol = np.zeros((n, *first_value[0].shape), dtype=first_value[0].dtype)
        wdl = np.zeros((n, *first_value[1].shape), dtype=first_value[1].dtype)
        for i, (pol_row, wdl_row) in hit_values.items():
            pol[i] = pol_row
            wdl[i] = wdl_row
        if miss_idx is not None and miss_outputs is not None:
            pol_m, wdl_m = miss_outputs
            for j, original_i in enumerate(miss_idx):
                i = int(original_i)
                pol[i] = pol_m[j]
                wdl[i] = wdl_m[j]
        return pol, wdl
