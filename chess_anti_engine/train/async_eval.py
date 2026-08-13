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

**Cudagraphs are NOT supported on this thread.** PyTorch's
``cudagraph_trees`` module stashes ``tree_manager_containers`` in
TLS at import time on the main thread only — user-spawned threads
have neither the Python ``threading.local`` attr nor the C++ TLS
key, so the first compiled forward asserts in ``get_obj``. The
caller (trainable_phases) maps ``reduce-overhead → default`` and
``max-autotune → max-autotune-no-cudagraphs`` before invoking
``start()``. ``torch.cuda.set_device`` here pins the thread's
default CUDA device for allocations; it does NOT bootstrap the
cudagraph TLS (a 2026-04-29 attempt to use it as a "real fix" was
reverted after the assertion kept firing).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.train.compile_probe import apply_compile
import contextlib

log = logging.getLogger(__name__)


class _Work:
    """Per-iter work item handed to the eval thread."""

    __slots__ = ("batch_size", "buf", "full_pass", "snap_state", "source_iter", "steps", "trainer")

    def __init__(
        self,
        *,
        snap_state: dict[str, torch.Tensor],
        trainer: Any,
        buf: Any,
        batch_size: int,
        steps: int,
        source_iter: int,
        full_pass: bool = False,
    ) -> None:
        self.snap_state = snap_state
        self.trainer = trainer
        self.buf = buf
        self.batch_size = batch_size
        self.steps = steps
        self.source_iter = source_iter
        self.full_pass = full_pass


class AsyncTestEval:
    """Long-lived eval thread + reusable snapshot model.

    The snapshot is built and compiled once on first start(); each
    subsequent start() pushes a new state_dict copy that the worker
    thread loads in-place. cudagraph capture happens once on the worker
    thread's first forward and replays on every iter after.
    """

    def __init__(self) -> None:
  # ⚑ EXACTLY ONE WORKER THREAD PER OBJECT, for the object's whole life.
  # ``shutdown()`` is TERMINAL (it sets ``_shutdown``, and ``start()`` refuses
  # afterwards) precisely so that this holds, which is what lets the fields
  # below be sorted into three categories with no state to reset between
  # workers:
  #   per-object   ``_lock``, ``_result_event``, ``_work_q``, ``_shutdown``
  #   per-WORKER   ``_thread``, ``_init_args``, ``_init_failed``,
  #                ``_compiled_shape_keys`` -- each describes the worker's own
  #                thread, init and dynamo cache, so under "one worker" they
  #                are also per-object and are never reset
  #   per-EVAL     ``_inflight_iter``, ``_result``, ``_exc``, ``_source_iter``
  #                -- reset by every ``start()``
  # The category matters because mixing them is silent: a second worker
  # inheriting per-worker state from a dead first one skips the compile
  # barrier this class exists for (review finding P1/P1b, PR #405 -- measured:
  # ``build_model`` called twice with ``start()`` returning in 0.00s). Adding
  # a field means placing it in one of these three categories; a per-worker
  # field is only safe while the invariant above holds.
        self._lock = threading.Lock()
        self._result_event = threading.Event()
        self._work_q: queue.Queue[_Work | None] = queue.Queue(maxsize=1)
        self._shutdown: bool = False
        self._thread: threading.Thread | None = None
        self._init_args: dict[str, Any] | None = None
        self._inflight_iter: int = -1
        self._result: Any = None
        self._exc: BaseException | None = None
        self._source_iter: int = -1
  # The worker thread died during init and will never dequeue work, so no
  # eval can ever identify itself by source_iter. The barrier below releases
  # on this INSTEAD of on "any exception": a per-eval exception belongs to
  # exactly one eval and is matched by source_iter like a success, whereas
  # this one means nobody is left to signal (review finding P2-1, PR #405).
        self._init_failed: bool = False
  # Batch-shape keys whose compiles this thread has already completed under
  # the barrier (see start()). A NEW key means the next eval will trigger a
  # dynamo (re)compile on the eval thread, so that start() must be a barrier
  # again. Keyed on (len(holdout_buf), batch_size): the full-pass shapes are
  # exactly the full batches plus the ragged tail those two determine. Both
  # can change mid-run — a holdout drift reset refreezes at a different row
  # count, and batch_size is absent from every declined-reload set — so
  # "first start only" would be a gate that protects exactly once (review
  # finding S1, PR #405).
        self._compiled_shape_keys: set[tuple[int, int]] = set()

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
        full_pass: bool = False,
    ) -> None:
        """Snapshot weights to CPU and hand them to the eval thread.

        ``full_pass`` scores every row of ``holdout_buf`` exactly once and
        ignores ``steps``; the holdout ruler uses it (docs/rl_loop_audit.md
        G14). It has to be plumbed through here rather than decided inside the
        thread because this path calls ``_compute_metrics`` directly, so the
        sync and async holdout evals would otherwise measure different things.

        A call whose batch shapes ``(len(holdout_buf), batch_size)`` this
        thread has not yet compiled is SYNCHRONOUS: it blocks until that eval
        completes, because the eval triggers a ``torch.compile`` trace on the
        eval thread and a concurrent main-thread compiled forward during the
        trace is fatal to the trial (see the compile-barrier comment below).
        In steady state — same shapes every iteration — only the first call
        blocks and every later call returns immediately as before.

        A call that arrives after ``shutdown()``, or once the worker has died
        at init, is REFUSED: it logs an error and returns without queueing, and
        ``collect()`` reports ``(None, -1)``. It does not raise (see
        ``_refuse_locked``) and it does not spawn a replacement worker (see
        ``shutdown``).
        """
  # torch.compile prefixes parameter keys with `_orig_mod.`; strip so
  # the snapshot (uncompiled before apply_compile wraps it) loads them.
        snap_state: dict[str, torch.Tensor] = {
            k.removeprefix("_orig_mod."): v.detach().to("cpu").clone()
            for k, v in trainer.model.state_dict().items()
        }
        work = _Work(
            snap_state=snap_state, trainer=trainer, buf=holdout_buf,
            batch_size=batch_size, steps=steps, source_iter=int(source_iter),
            full_pass=bool(full_pass),
        )

        shape_key = (len(holdout_buf), int(batch_size))
        with self._lock:
  # Two unrecoverable states, one contract: there is no usable worker and
  # none will be created, so the eval is REFUSED rather than queued.
  #
  # (1) ``shutdown()`` ran. This object is single-use — see the invariant
  # in __init__ — and a respawn would be a fresh worker inheriting the
  # dead one's per-worker state: it would skip the compile barrier
  # (``_compiled_shape_keys``) and dequeue shutdown()'s own leftover
  # ``None`` sentinel from the maxsize=1 queue and exit before running
  # anything, leaving the unbounded barrier waiting FOREVER (both
  # measured, review finding P1/P1b, PR #405).
  #
  # (2) The worker died during init. It never drains the maxsize=1 queue,
  # so the put below would block forever once the queue holds the
  # never-dequeued prior item — and with the unbounded barrier the wait
  # after it would too (review follow-up, PR #405).
            if self._shutdown:
                self._refuse_locked(
                    source_iter=int(source_iter),
                    why="after shutdown(); an AsyncTestEval is single-use and "
                        "shutdown is terminal, so a new instance is required",
                )
                return
            if self._thread is not None and not self._thread.is_alive():
                self._refuse_locked(
                    source_iter=int(source_iter),
                    why="with a dead worker thread (init failure?), whose "
                        "maxsize=1 queue nothing drains",
                )
                return
            needs_barrier = shape_key not in self._compiled_shape_keys
            if self._thread is None:
  # Lazy init on first call so we can capture device/compile_mode/
  # model_cfg from the trainer's runtime config without forcing the
  # caller to pass them at construction time. Runs at most ONCE per
  # object: the only writer of ``_thread = None`` is shutdown(), and a
  # start() after shutdown() is refused above. That is what makes the
  # per-worker fields (``_init_failed``, ``_compiled_shape_keys``,
  # ``_init_args``) safe without a reset here — an earlier revision reset
  # ``_init_failed`` and not ``_compiled_shape_keys``, which is the
  # asymmetry that produced an un-barriered compile on the respawn path.
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
                with contextlib.suppress(queue.Empty):
                    self._work_q.get_nowait()
            self._inflight_iter = int(source_iter)
            self._result = None
            self._exc = None
            self._source_iter = -1
            self._result_event.clear()

        self._work_q.put(work)

  # ⚑ COMPILE BARRIER: an eval whose batch shapes this thread has not yet
  # compiled is synchronous. torch.compile's trace phase (make_fx /
  # proxy_tensor) sets `torch.fx._symbolic_trace`'s PROCESS-GLOBAL tracing
  # flag and patches `nn.Module.__call__` process-wide, so while the eval
  # thread lazily compiles its snapshot, ANY dynamo-optimized call on the
  # main thread — the training step, the era probe, a gate eval — raises
  #   RuntimeError: Detected that you are using FX to symbolically trace a
  #   dynamo-optimized function
  # and kills the trial (observed 2026-08-12 16:39, trial d76cc, iteration 2;
  # docs/experiment_ledger.md "arm A CRASHED at iter 2"). Blocking here until
  # that eval finishes means a compile on this thread cannot overlap
  # main-thread work.
  #
  # The barrier re-arms whenever (len(holdout_buf), batch_size) is a pair it
  # has not seen: those two determine every shape in a full pass, and both
  # can change mid-run (holdout drift reset; live batch_size edit). While
  # they are stable — the common case, since the L2 unfrozen-holdout skip
  # keeps the buffer frozen across evals — every start() after the first is
  # fully asynchronous, and weight reloads never invalidate dynamo guards.
  #
  # The wait is UNBOUNDED, with loud progress prints: the one measured cold
  # compile took >1800s (the trainer's, 2063s, same boot), so any finite
  # budget small enough to matter disengages the barrier exactly on the cold
  # boots it exists for (review finding B1, PR #405). Both eval-thread
  # failure paths set ``_result_event``, so this only blocks forever if the
  # compile itself is wedged — in which case the alternative was the crash.
  #
  # The barrier waits on ``_result_event`` WITHOUT consuming the result:
  # ``collect()`` still performs the read + bookkeeping. The result of the
  # barriered eval is charged to the iteration that started it, exactly as
  # before — only the wall clock moves (the compile cost lands here instead
  # of racing the next iteration).
        if needs_barrier:
            t0 = time.monotonic()
            while True:
                if not self._result_event.wait(timeout=300.0):
                    log.warning(
                        "AsyncTestEval compile barrier still waiting after %.0fs "
                        "(shapes %s; a cold max-autotune eval compile can exceed "
                        "30 min — this is the barrier working, not a hang)",
                        time.monotonic() - t0, shape_key,
                    )
                    continue
                with self._lock:
                    if self._init_failed or self._source_iter == int(source_iter):
                        barrier_ok = self._exc is None
                        if barrier_ok:
                            self._compiled_shape_keys.add(shape_key)
                        break
  # A STALE eval set the event: one started before this call and
  # abandoned by the drain above, completing mid-barrier. Its
  # compile ran under the OLD shapes, so releasing here would mark
  # the NEW key compiled without its compile ever having happened
  # (review finding S3, PR #405). Consume the wakeup and keep
  # waiting for OUR eval — identified by source_iter, which _loop
  # writes under this same lock on BOTH its success and its failure
  # path. Testing ``self._exc is not None`` here instead would apply
  # the ownership test to only half the outcomes: a stale eval that
  # RAISES releases this barrier, reopening the same hole S3 closed
  # for the success path (review finding P2-1, PR #405 — reproduced:
  # a stale eval raising at t=0.4s returned start() at 0.40s with the
  # new-shape compile still to come).
  #
  # ⚑ DO NOT fold ``_init_failed`` into the identity check above —
  # it is NOT a redundant special case, it is the termination
  # condition. A worker that dies during init returns before its
  # dequeue loop, so it never writes a source_iter for ANY eval;
  # gating its release on identity would leave this wait unsatisfied
  # forever, and the wait is now UNBOUNDED (review finding B1), so
  # that is a HANG, not a late failure. It is the one exception that
  # must release every waiter, precisely because nobody is left to
  # signal. It needs no reset: there is exactly one worker per object
  # (see __init__), so it can only ever describe that worker.
                    self._result_event.clear()
            log.info(
                "AsyncTestEval compile barrier released after %.1fs "
                "(shapes %s, eval %s); later evals with these shapes run "
                "asynchronously",
                time.monotonic() - t0, shape_key,
                "succeeded" if barrier_ok else "FAILED (will re-barrier)",
            )

    def _refuse_locked(self, *, source_iter: int, why: str) -> None:
        """Fail this eval instead of queueing it. The caller must hold ``_lock``.

        Same contract as an eval that raised: a ``log.error``, an ``_exc`` for
        ``collect()`` to report as ``(None, -1)``, and ``_result_event`` set so
        nothing is left waiting on the (unbounded) barrier.

        It deliberately does NOT raise. ``start()`` is called from
        ``train_trial``'s iteration loop, which has a ``finally:`` and zero
        ``except``, so raising here would turn a missing holdout metric — the
        whole cost of a refused eval — into a dead trial.
        """
        log.error("AsyncTestEval.start called %s; dropping the iter %d eval", why, source_iter)
        if self._exc is None:
            self._exc = RuntimeError(f"AsyncTestEval.start called {why}")
        self._result_event.set()

    def _loop(self) -> None:
        """Worker thread main loop. Builds the snapshot once; reuses it forever."""
        if self._init_args is None:
            log.error("AsyncTestEval._loop entered before init args were set")
  # Release any barrier/collect waiter: with an unbounded barrier wait,
  # returning without signalling would hang start() forever (review
  # nit, PR #405 — same contract as the two failure paths below).
            with self._lock:
                self._exc = RuntimeError("AsyncTestEval init args were never set")
                self._init_failed = True
            self._result_event.set()
            return
        try:
  # Pin the thread's default CUDA device for tensor allocations made
  # via ``.to("cuda")`` without an explicit index. Does NOT bootstrap
  # cudagraph_trees TLS (that key only ever lands in TLS on the
  # import thread or on autograd-spawned threads); the caller strips
  # cudagraphs from compile_mode for this thread to compensate.
            dev = torch.device(self._init_args["device"])
            if dev.type == "cuda":
  # bare "cuda" yields dev.index=None; set_device needs an int.
  # ``or 0`` collapses None and 0 to 0 (both denote default device).
                torch.cuda.set_device(dev.index or 0)
            snap = build_model(self._init_args["model_cfg"]).to(self._init_args["device"])
            snap.eval()
            snap = apply_compile(
                snap, mode=self._init_args["compile_mode"],
                device=self._init_args["device"],
            )
        except BaseException as exc:
            log.exception("AsyncTestEval init failed")
            with self._lock:
                self._exc = exc
                self._init_failed = True
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
                metrics = work.trainer._compute_metrics(
                    buf=work.buf,
                    batch_size=work.batch_size,
                    steps=work.steps,
                    tag="eval",
                    model_override=snap,
                    full_pass=work.full_pass,
                )
                with self._lock:
                    self._result = metrics
                    self._source_iter = work.source_iter
            except BaseException as exc:
                log.exception("async test eval failed")
                with self._lock:
                    self._exc = exc
  # Record WHOSE eval failed, exactly as the success path does. The
  # compile barrier identifies its own eval by source_iter, so a
  # failure with no source_iter would be indistinguishable from any
  # other eval's failure and would release a barrier belonging to a
  # different shape key (review finding P2-1, PR #405). collect() is
  # unaffected: it returns (None, -1) whenever _exc is set.
                    self._source_iter = work.source_iter
            self._result_event.set()

    def has_inflight(self) -> bool:
        """True when a ``start()`` is outstanding and ``collect()`` has work.

        ``collect()`` blocks on ``_result_event`` for its FULL timeout when
        nothing was started, so a caller that conditionally skips ``start()``
        must not blindly call it on the next iteration -- 120s of dead wall
        clock per skipped iteration, charged to the training loop. The
        conditions that skip a start already exist (a holdout below
        ``batch_size``, and the unfrozen-holdout skip added for audit L2), so
        this predicate is what makes skipping cheap rather than expensive.

        The very first skip is harmless without it -- ``_thread`` is still
        None because it is created lazily inside ``start()``, and ``collect()``
        short-circuits on that -- but any skip AFTER a successful start is not.
        """
        with self._lock:
            return self._inflight_iter >= 0

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
        """Tell the worker thread to exit and join it. TERMINAL and idempotent.

        After this the object is spent: ``start()`` refuses every later eval
        (loudly, via ``_refuse_locked``) instead of spawning a second worker.
        That is a deliberate choice over making respawn work, because the
        respawn path was reachable ONLY through here — ``shutdown()`` is the
        only writer of ``_thread = None``, so even a worker that dies at init
        cannot be replaced without it — and production never takes it:
        ``AsyncTestEval`` is built once per trial by
        ``_lazy_construct_iter_helpers`` and shut down once by
        ``_cleanup_trial_resources``, which then discards it. Supporting a
        second worker would mean resetting every per-worker field correctly
        (see __init__) on a path no caller uses, and getting that partially
        right is what produced the un-barriered compile in review finding
        P1b, PR #405.
        """
        with self._lock:
            self._shutdown = True
            thread = self._thread
        if thread is None:
            return
  # Drain any queued work so the sentinel put doesn't block on a
  # full queue (maxsize=1). Worker may still be mid-eval on a
  # previously dequeued item; that's fine — it'll see None next.
  # A worker that is ALREADY dead never dequeues this sentinel, so it
  # outlives the join; harmless now that nothing else is ever put.
        with contextlib.suppress(queue.Empty):
            self._work_q.get_nowait()
        self._work_q.put(None)
        thread.join(timeout=timeout)
        with self._lock:
            self._thread = None
