"""Tests for AsyncTestEval lifecycle and snapshot isolation."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

from chess_anti_engine.train.async_eval import AsyncTestEval


def _start_bounded(aer: AsyncTestEval, *, bound_s: float = 30.0, **kwargs) -> float:
    """Run ``AsyncTestEval.start()`` on a helper thread; FAIL if it doesn't return.

    ⚑ Every ``start()`` in this file goes through here, and that is not
    ceremony. The compile barrier's wait is UNBOUNDED by design (a finite
    budget disengages it exactly on the cold boots it exists for — review
    finding B1, PR #405), so a regression that leaves the wait unsatisfied does
    not fail a test that calls ``start()`` on the main thread: it hangs the
    whole session until CI's job timeout, with no attribution. Measured on this
    file before this helper: the mutant that folds ``_init_failed`` into the
    barrier's identity check, and the one that drops the failure path's
    ``_source_iter`` write, were both real detections rendered undiagnosable —
    each burned 240s and exited 124 instead of failing (review finding P2,
    PR #405). There is no ``pytest-timeout`` in this repo's test deps, and a
    wall-clock bound here needs none.

    Returns elapsed seconds, so a caller can additionally bound the latency.
    """
    done = threading.Event()

    def _run() -> None:
        aer.start(**kwargs)
        done.set()

    t = threading.Thread(target=_run, name="test-start-bounded", daemon=True)
    t0 = time.monotonic()
    t.start()
    t.join(timeout=bound_s)
    elapsed = time.monotonic() - t0
    assert done.is_set(), (
        f"start(source_iter={kwargs.get('source_iter')}) did not return within "
        f"{bound_s:.1f}s -- the compile barrier's wait is unbounded, so this is "
        "a hang, not slowness"
    )
    return elapsed


class _StubChessNet(torch.nn.Module):
    """Minimal model with a state_dict — enough to exercise snapshot path."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)


@pytest.fixture
def cfg_and_builder(monkeypatch):
    """Patch ``build_model`` so the test doesn't need a real ChessNet config."""
    fake_cfg = MagicMock(name="ModelConfig")

    def _fake_build_model(_cfg):
        return _StubChessNet(dim=4)

    monkeypatch.setattr("chess_anti_engine.train.async_eval.build_model", _fake_build_model)
    return fake_cfg


def _make_trainer_stub(model: torch.nn.Module, eval_payload):
    """Build a minimal trainer-like object that AsyncTestEval can drive."""
    trainer = MagicMock()
    trainer.model = model

    captured = {}

    def _compute_metrics(*, buf, batch_size, steps, tag, model_override=None, full_pass=False):
        del buf, tag  # consumed only via captured below
        captured["model_override"] = model_override
        captured["batch_size"] = batch_size
        captured["steps"] = steps
        captured["full_pass"] = full_pass
        return eval_payload

    trainer._compute_metrics = _compute_metrics
    return trainer, captured


def test_collect_returns_started_eval(cfg_and_builder):
    model = _StubChessNet(dim=4)
    trainer, captured = _make_trainer_stub(model, eval_payload="EVAL_RESULT_OK")

    aer = AsyncTestEval()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="STUB_BUF",
        batch_size=4, steps=2, device="cpu", source_iter=7,
    )
    metrics, src = aer.collect(timeout=5.0)
    assert metrics == "EVAL_RESULT_OK"
    assert src == 7
    assert captured["model_override"] is not trainer.model, (
        "eval should run on snapshot, not on trainer.model"
    )


def test_collect_returns_none_when_no_eval_started():
    aer = AsyncTestEval()
    metrics, src = aer.collect(timeout=0.1)
    assert metrics is None
    assert src == -1


def test_start_during_inflight_orphans_prior(cfg_and_builder, caplog):
    """If start() is called while a prior eval is still running, it must
    warn + accept the new work rather than raise. With the long-lived
    eval thread the prior eval's result gets dropped (collect after the
    new start sees only the new iter's metrics), but the warning is the
    user-visible signal that something was abandoned.

    The gated (still-running) eval is the SECOND one, not the first: the
    first start() is now a synchronous compile barrier, so gating the
    first eval would simply block that start() instead of producing an
    in-flight overlap. Steady-state overlap — the thing this test pins —
    only ever happens from the second start() onward."""
    import threading
    model = _StubChessNet(dim=4)
    block = threading.Event()
    calls = {"n": 0}

    def _first_fast_then_slow(**_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return "FIRST"
        block.wait(timeout=5.0)
        return "X"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _first_fast_then_slow

    aer = AsyncTestEval()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == ("FIRST", 1)
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=2,
    )
    with caplog.at_level("WARNING"):
        _start_bounded(aer,
            trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
            batch_size=4, steps=2, device="cpu", source_iter=3,
        )
    assert any("previous eval" in r.getMessage() and "still running" in r.getMessage() for r in caplog.records)
    block.set()
    aer.collect(timeout=5.0)


def test_first_start_is_a_compile_barrier(cfg_and_builder):
    """The FIRST start() must not return until the first eval has finished.

    That first eval is where the thread's only ``torch.compile`` trace
    happens, and the trace sets a PROCESS-GLOBAL fx-tracing flag: a
    concurrent main-thread compiled forward then dies with "Detected that
    you are using FX to symbolically trace a dynamo-optimized function"
    (killed Tier-13 arm A, 2026-08-12, trial d76cc iteration 2). The
    barrier is the fix, so this test is its regression pin: it fails if
    the wait in start() is removed.
    """
    import threading
    model = _StubChessNet(dim=4)
    release = threading.Event()
    eval_finished = threading.Event()
    start_returned = threading.Event()
    overlap_seen: dict[str, bool | None] = {"value": None}

    def _gated_eval(**_kw):
        release.wait(timeout=10.0)
        eval_finished.set()
        return "GATED"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _gated_eval

    release_time: dict[str, float] = {}

    def _observer():
        # Hold the gate for 1.5s. While the eval is gated, start() must still
        # be blocked. Record the observation, then release the eval so
        # start() can return. The 1.5s hold plus the post-release latency
        # bound below is what makes a fixed-sleep mutant fail: a sleep
        # shorter than the hold returns before the eval finished, a sleep
        # meaningfully longer than the hold blows the latency bound. Only a
        # sleep landing within ~1s of the hold escapes both, and the real
        # mechanism (waiting on the eval's completion event) satisfies both
        # for ANY hold duration (review finding S2, PR #405).
        time.sleep(1.5)
        overlap_seen["value"] = start_returned.is_set()
        release_time["t"] = time.monotonic()
        release.set()

    obs = threading.Thread(target=_observer)
    obs.start()
    aer = AsyncTestEval()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    t_returned = time.monotonic()
    start_returned.set()
    obs.join(timeout=10.0)
    # The eval must have completed BEFORE start() returned...
    assert eval_finished.is_set(), "first start() returned before the first eval ran"
    # ...while the eval was still gated, start() had not yet returned...
    assert overlap_seen["value"] is False, (
        "first start() returned while the first eval (the compile) was still "
        "running -- the compile barrier is gone and the FX-trace race is back"
    )
    # ...and start() returned BECAUSE the eval finished, not after some fixed
    # sleep: it must unblock promptly once the gate opens.
    latency = t_returned - release_time["t"]
    assert latency < 1.0, (
        f"start() returned {latency:.2f}s after the eval was released -- it is "
        "not waiting on the eval's completion event"
    )
    assert aer.collect(timeout=5.0) == ("GATED", 1)


def test_barrier_rearms_when_batch_shapes_change(cfg_and_builder):
    """The barrier must re-engage when (len(holdout_buf), batch_size) is a
    pair the eval thread has not compiled yet.

    Both components can change mid-run — a holdout drift reset refreezes at
    a different row count, and batch_size is live-reloadable — and a changed
    pair changes the full-pass batch shapes, which triggers a dynamo
    recompile ON THE EVAL THREAD. A first-start-only barrier would protect
    exactly once and then silently stop protecting (review finding S1,
    PR #405).
    """
    import threading
    model = _StubChessNet(dim=4)
    gate = threading.Event()
    eval_done = threading.Event()
    calls = {"n": 0}

    def _eval(**_kw):
        calls["n"] += 1
        if calls["n"] < 3:
            return f"FAST{calls['n']}"
        gate.wait(timeout=10.0)
        eval_done.set()
        return "REARMED"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _eval

    aer = AsyncTestEval()
    # Eval 1: first ever -> barrier (shapes (1, 4) now registered).
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == ("FAST1", 1)
    # Eval 2: same shapes -> asynchronous (returns while nothing blocks it).
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=2,
    )
    assert aer.collect(timeout=5.0) == ("FAST2", 2)
    # Eval 3: batch_size changed -> NEW shape pair -> must barrier again.
    def _release_later():
        time.sleep(0.5)
        gate.set()
    rel = threading.Thread(target=_release_later)
    rel.start()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=8, steps=2, device="cpu", source_iter=3,
    )
    # Observe AT THE MOMENT start() RETURNS, before any join: an un-barriered
    # start returns in microseconds, long before the 0.5s-gated eval has run,
    # so sampling later (after join) would see the eval finished either way
    # and the test could not observe the regression it exists for.
    done_at_return = eval_done.is_set()
    rel.join(timeout=10.0)
    assert done_at_return, (
        "start() with a NEW batch-shape pair returned before its eval ran -- "
        "the barrier did not re-arm on a shape change"
    )
    assert aer.collect(timeout=5.0) == ("REARMED", 3)


def test_second_start_stays_asynchronous(cfg_and_builder):
    """Only the FIRST start() is synchronous; the second must return while
    its eval is still running (the async design is the point of the module,
    and an over-broad barrier would silently turn every eval synchronous)."""
    import threading
    model = _StubChessNet(dim=4)
    block = threading.Event()
    calls = {"n": 0}

    def _first_fast_then_gated(**_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return "FIRST"
        block.wait(timeout=10.0)
        return "SECOND"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _first_fast_then_gated

    aer = AsyncTestEval()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == ("FIRST", 1)

    t0 = time.monotonic()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=2,
    )
    elapsed = time.monotonic() - t0
    # The gated eval holds for up to 10s; an asynchronous start returns in
    # milliseconds. 2s is three orders of magnitude of slack, not a race.
    assert elapsed < 2.0, f"second start() blocked for {elapsed:.1f}s -- barrier over-applied"
    assert not block.is_set()
    block.set()
    assert aer.collect(timeout=5.0) == ("SECOND", 2)


def test_snapshot_isolated_from_trainer_model(cfg_and_builder):
    """Mutating trainer.model after start() must not affect snapshot's eval."""
    model = _StubChessNet(dim=4)
    snap_seen = {}

    def _compute_metrics_capturing(*, buf, batch_size, steps, tag, model_override=None, full_pass=False):
        del buf, batch_size, steps, tag, full_pass  # only model_override matters here
  # Sleep so the test has time to mutate trainer.model before the
  # eval thread reads it. AsyncTestEval should have already snapped
  # the state_dict before kicking the thread, so this mutation
  # should NOT change the snapshot's params.
        time.sleep(0.1)
        assert model_override is not None, "AsyncTestEval should pass a snapshot"
        snap_seen["weight_id"] = id(model_override.linear.weight)
        snap_seen["weight_data"] = model_override.linear.weight.detach().clone()
        return "OK"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _compute_metrics_capturing

    aer = AsyncTestEval()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=42,
    )
  # Mutate trainer.model in place — snapshot must already be detached.
    model.linear.weight.data.fill_(99.0)
    aer.collect(timeout=5.0)
    assert snap_seen["weight_id"] != id(model.linear.weight), (
        "snapshot should be a separate parameter tensor"
    )
    snap_w = snap_seen["weight_data"]
    assert not torch.allclose(snap_w, torch.full_like(snap_w, 99.0)), (
        "snapshot must not see post-start mutations to trainer.model"
    )


def test_collect_handles_exception_in_thread(cfg_and_builder):
    model = _StubChessNet(dim=4)
    trainer = MagicMock()
    trainer.model = model

    def _boom(**_kw):
        raise RuntimeError("intentional")
    trainer._compute_metrics = _boom

    aer = AsyncTestEval()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=99,
    )
    metrics, src = aer.collect(timeout=5.0)
    assert metrics is None
    assert src == -1


def test_load_state_dict_works_through_compile_wrapper(cfg_and_builder, monkeypatch):
    """When apply_compile wraps the snap, the state_dict load must reach the
    underlying params. Previously strict=False against an OptimizedModule's
    prefixed keys silently no-op'd, leaving the snap at its build-time random
    init."""
    model = _StubChessNet(dim=4)
    model.linear.weight.data.fill_(7.0)
    model.linear.bias.data.fill_(11.0)

    captured: dict = {}

    class _FakeOptimizedModule(torch.nn.Module):
        """Mimic torch.compile's OptimizedModule wrapping convention: the
        wrapped model is stored as an attribute named ``_orig_mod`` and the
        state_dict picks up an ``_orig_mod.`` prefix."""
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = inner

        def forward(self, *args, **kwargs):
            return self._orig_mod(*args, **kwargs)

    def _fake_apply_compile(m, *, mode, device):
        del mode, device
        return _FakeOptimizedModule(m)

    monkeypatch.setattr("chess_anti_engine.train.async_eval.apply_compile", _fake_apply_compile)

    def _capture(*, buf, batch_size, steps, tag, model_override=None, full_pass=False):
        del buf, batch_size, steps, tag, full_pass
  # Reach into the compiled snap and read the loaded params.
        assert model_override is not None
        captured["w"] = model_override._orig_mod.linear.weight.detach().clone()
        captured["b"] = model_override._orig_mod.linear.bias.detach().clone()
        return "OK"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _capture

    aer = AsyncTestEval()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=5, compile_mode="reduce-overhead",
    )
    metrics, src = aer.collect(timeout=5.0)
    assert metrics == "OK"
    assert src == 5
    assert torch.allclose(captured["w"], torch.full_like(captured["w"], 7.0)), (
        "snap params should equal the trainer's pushed weights, not the build-time init"
    )
    assert torch.allclose(captured["b"], torch.full_like(captured["b"], 11.0))


def test_second_iter_reuses_snap_with_new_weights(cfg_and_builder):
    """Persistent eval thread + in-place state_dict load: iter 2 must see iter
    2's weights, not iter 1's, even though the snap model instance is reused."""
    model = _StubChessNet(dim=4)
    seen_weights: list[torch.Tensor] = []

    def _capture(*, buf, batch_size, steps, tag, model_override=None, full_pass=False):
        del buf, batch_size, steps, tag, full_pass
        assert model_override is not None
        seen_weights.append(model_override.linear.weight.detach().clone())
        return f"iter_{len(seen_weights)}"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _capture

    aer = AsyncTestEval()
    model.linear.weight.data.fill_(1.0)
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=10,
    )
    m1, s1 = aer.collect(timeout=5.0)

    model.linear.weight.data.fill_(2.0)
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=11,
    )
    m2, s2 = aer.collect(timeout=5.0)

    assert (m1, s1) == ("iter_1", 10)
    assert (m2, s2) == ("iter_2", 11)
    assert len(seen_weights) == 2
    assert torch.allclose(seen_weights[0], torch.ones_like(seen_weights[0]))
    assert torch.allclose(seen_weights[1], torch.full_like(seen_weights[1], 2.0))
    aer.shutdown(timeout=5.0)


def test_a_stale_evals_completion_does_not_release_a_rearmed_barrier(cfg_and_builder):
    """A re-armed barrier must wait for ITS OWN eval, not any eval.

    Scenario (review finding S3, PR #405, demonstrated against the previous
    revision): eval N is still in flight when start() is called with NEW
    batch shapes. The stale eval completes mid-barrier and sets
    ``_result_event`` — releasing a barrier keyed to shapes whose compile
    has not run, and marking the new shape key compiled. The barrier must
    consume that stale wakeup and keep waiting for the eval it started,
    identified by source_iter.
    """
    import threading
    model = _StubChessNet(dim=4)
    entered2 = threading.Event()  # the STALE eval has been DEQUEUED and is running
    g2 = threading.Event()   # gates the STALE eval (iter 2, old shapes)
    g3 = threading.Event()   # gates OUR eval (iter 3, new shapes)
    done3 = threading.Event()
    calls = {"n": 0}

    def _eval(**_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return "FIRST"
        if calls["n"] == 2:
            entered2.set()
            g2.wait(timeout=10.0)
            return "STALE"
        g3.wait(timeout=10.0)
        done3.set()
        return "OURS"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _eval

    start_returned = threading.Event()
    stale_released_barrier: dict[str, bool | None] = {"value": None}

    def _releaser():
        # Complete the STALE eval first; the barrier must NOT release on it.
        time.sleep(0.3)
        g2.set()
        time.sleep(0.7)
        # If start() returned before OUR eval was even ungated, the stale
        # completion released the barrier.
        stale_released_barrier["value"] = start_returned.is_set()
        g3.set()

    aer = AsyncTestEval()
    # Eval 1: registers shapes (1, 4) under the first barrier.
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == ("FIRST", 1)
    # Eval 2: same shapes -> asynchronous; leave it IN FLIGHT (gated on g2).
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=2,
    )
    # ⚑ Wait for the worker to actually DEQUEUE eval 2 before starting eval 3.
    # Without this the test is flaky (measured 3/25 whole-file runs): start()'s
    # drain does `_work_q.get_nowait()`, so whenever the worker had not yet
    # taken work2 it is removed and eval 2 NEVER RUNS. Eval 3 then becomes the
    # SECOND _compute_metrics call, takes the g2 branch, and is released at
    # t=0.3s -- the barrier returning early for a legitimate reason, which the
    # assertions below misread as "the stale eval released it". A sleep would
    # only make that window smaller; this makes the precondition true.
    assert entered2.wait(timeout=10.0), "the stale eval was never dequeued"
    rel = threading.Thread(target=_releaser)
    rel.start()
    # Eval 3: NEW shapes (1, 8) -> re-armed barrier, while eval 2 is in flight.
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=8, steps=2, device="cpu", source_iter=3,
    )
    done3_at_return = done3.is_set()
    start_returned.set()
    rel.join(timeout=10.0)
    assert stale_released_barrier["value"] is False, (
        "the STALE eval's completion released the re-armed barrier -- the new "
        "shape key was marked compiled without its compile having run"
    )
    assert done3_at_return, (
        "start() returned before its own eval completed"
    )
    assert aer.collect(timeout=5.0) == ("OURS", 3)


def test_a_stale_evals_FAILURE_does_not_release_a_rearmed_barrier(cfg_and_builder):
    """The ownership test must cover the FAILURE path, not just the success one.

    Review finding P2-1 (PR #405). The release condition was
    ``self._exc is not None or self._source_iter == int(source_iter)``: the
    second half asks "was this MY eval?", the first half does not — so ANY
    eval's exception released a re-armed barrier, reopening for failures
    exactly the hole S3 closed for successes. Reproduced against that
    revision: a stale eval raising at t=0.4s returned ``start()`` at 0.40s,
    leaving the NEW-shape compile to run concurrently with main-thread work.

    Reachability is S3's, which this PR already chose to fix: a shape change
    plus an in-flight stale eval that raises. ``trainable.py``'s drift-reset
    path predicts that exact pair ("Expect either a ValueError from the eval
    thread"); it is disarmed today only by ``reset_holdout_on_drift: false``.
    """
    import threading
    model = _StubChessNet(dim=4)
    entered2 = threading.Event()  # the STALE eval has been DEQUEUED and is running
    g2 = threading.Event()   # gates the STALE eval's RAISE (iter 2, old shapes)
    g3 = threading.Event()   # gates OUR eval (iter 3, new shapes)
    done3 = threading.Event()
    calls = {"n": 0}

    def _eval(**_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return "FIRST"
        if calls["n"] == 2:
            entered2.set()
            g2.wait(timeout=10.0)
            raise ValueError("stale eval raised (holdout cleared under it)")
        g3.wait(timeout=10.0)
        done3.set()
        return "OURS"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _eval

    start_returned = threading.Event()
    stale_released_barrier: dict[str, bool | None] = {"value": None}

    def _releaser():
        # Make the STALE eval RAISE first; the barrier must NOT release on it.
        time.sleep(0.3)
        g2.set()
        time.sleep(0.7)
        stale_released_barrier["value"] = start_returned.is_set()
        g3.set()

    aer = AsyncTestEval()
    # Eval 1: registers shapes (1, 4) under the first barrier.
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == ("FIRST", 1)
    # Eval 2: same shapes -> asynchronous; leave it IN FLIGHT (gated on g2).
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=2,
    )
    assert entered2.wait(timeout=10.0), "the stale eval was never dequeued"
    rel = threading.Thread(target=_releaser)
    rel.start()
    # Eval 3: NEW shapes (1, 8) -> re-armed barrier, while eval 2 is in flight.
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=8, steps=2, device="cpu", source_iter=3,
    )
    done3_at_return = done3.is_set()
    start_returned.set()
    rel.join(timeout=10.0)
    assert stale_released_barrier["value"] is False, (
        "the STALE eval's EXCEPTION released the re-armed barrier -- the "
        "new-shape compile is now racing main-thread work"
    )
    assert done3_at_return, "start() returned before its own eval completed"
    # The stale exception is still charged to whatever collect() reads next --
    # pre-existing behaviour of the abandoned-prior-result path (start()'s
    # drain comment), unchanged here and deliberately out of this fix's scope.
    # The barrier's correctness does not depend on it: a lingering _exc only
    # withholds the shape key, so the next start() with these shapes takes the
    # barrier again. That is the fail-safe direction.
    assert aer.collect(timeout=5.0) == (None, -1)
    aer.shutdown(timeout=5.0)


def test_barrier_rearms_when_the_holdout_row_count_changes(cfg_and_builder):
    """Pins the OTHER half of the shape key: ``len(holdout_buf)``.

    Review finding P2-3 (PR #405): mutating the key to ``(0, int(batch_size))``
    survived 6/6 whole-file runs, because every existing test varies only
    batch_size. Yet the row-count half is the one gated by a config key -- a
    holdout drift reset refreezes at a DIFFERENT row count with batch_size
    unchanged, which changes the full pass's ragged tail and forces an
    eval-thread dynamo recompile exactly as a batch_size change does.
    """
    import threading
    model = _StubChessNet(dim=4)
    gate = threading.Event()
    eval_done = threading.Event()
    calls = {"n": 0}

    def _eval(**_kw):
        calls["n"] += 1
        if calls["n"] < 2:
            return "FAST"
        gate.wait(timeout=10.0)
        eval_done.set()
        return "REARMED"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _eval

    aer = AsyncTestEval()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == ("FAST", 1)

    def _release_later():
        time.sleep(0.5)
        gate.set()

    rel = threading.Thread(target=_release_later)
    rel.start()
    # SAME batch_size, DIFFERENT holdout row count (len 1 -> len 3).
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="BBB",
        batch_size=4, steps=2, device="cpu", source_iter=2,
    )
    # Sampled at the moment start() returns, before any join: an un-barriered
    # return takes microseconds and is cleanly distinguishable from the
    # 0.5s-gated eval.
    done_at_return = eval_done.is_set()
    rel.join(timeout=10.0)
    assert done_at_return, (
        "start() with a NEW holdout row count returned before its eval ran -- "
        "the barrier did not re-arm on a row-count change"
    )
    assert aer.collect(timeout=5.0) == ("REARMED", 2)
    aer.shutdown(timeout=5.0)


def test_a_dead_worker_thread_fails_evals_instead_of_hanging(monkeypatch):
    """If the eval thread died at init, later start() calls must fail loudly
    and promptly — not block forever on the maxsize=1 queue nothing drains
    (pre-existing hang surfaced by the PR #405 review; the unbounded barrier
    made fixing it mandatory)."""
    import threading

    def _boom(_cfg):
        raise RuntimeError("synthetic init failure")

    monkeypatch.setattr("chess_anti_engine.train.async_eval.build_model", _boom)
    model = _StubChessNet(dim=4)
    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = MagicMock()

    aer = AsyncTestEval()
    # First start: init fails on the worker thread; the failure paths set
    # _exc + _result_event, so the barrier releases with the error rather
    # than hanging, and collect reports the failure as (None, -1).
    _start_bounded(aer,
        trainer=trainer, model_cfg=MagicMock(), holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == (None, -1)

    # Second start: the thread is dead and its queue still holds the first,
    # never-dequeued work item. Run in a helper thread so a regression hangs
    # THIS join instead of the whole test session.
    finished = threading.Event()

    def _second_start():
        aer.start(
            trainer=trainer, model_cfg=MagicMock(), holdout_buf="B",
            batch_size=4, steps=2, device="cpu", source_iter=2,
        )
        finished.set()

    t = threading.Thread(target=_second_start, daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert finished.is_set(), (
        "start() on a dead worker thread blocked (the maxsize=1 queue put) "
        "instead of failing the eval loudly"
    )
    # ⚑ Bound the wall clock, don't just read the value. (None, -1) is ALSO
    # what a timed-out collect() returns, so the value alone cannot see the
    # guard's `_result_event.set()` disappear -- that mutant survived 6/6
    # whole-file runs and was visible only as 10.2s vs 5.7s of elapsed time
    # (review finding P3-4, PR #405). The assertion is that collect() returned
    # because the guard SIGNALLED, not because it gave up.
    t0 = time.monotonic()
    assert aer.collect(timeout=5.0) == (None, -1)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, (
        f"collect() took {elapsed:.2f}s -- it timed out rather than being "
        "released by the dead-worker guard's _result_event.set()"
    )


def test_shutdown_joins_thread(cfg_and_builder):
    model = _StubChessNet(dim=4)
    trainer, _ = _make_trainer_stub(model, eval_payload="OK")
    aer = AsyncTestEval()
    _start_bounded(aer,
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    aer.collect(timeout=5.0)
    aer.shutdown(timeout=5.0)
    assert aer._thread is None


def test_start_after_shutdown_is_refused_and_never_respawns_a_worker(monkeypatch, caplog):
    """``shutdown()`` is TERMINAL: a later ``start()`` refuses, loudly.

    Review finding P1/P1b (PR #405). A respawn after shutdown was not a
    working path — it was two different silent breakages, one per piece of
    per-worker state that survived ``shutdown()``:

    * ``_compiled_shape_keys`` survived, so the fresh worker inherited keys
      for compiles it had never run: ``needs_barrier`` was False and its
      genuinely-fresh ``torch.compile`` ran with NO barrier at all. Measured
      at the previous head: ``start()`` returned in 0.00s with ``build_model``
      called TWICE. That is precisely the FX-trace race this module exists to
      close, reached through a field the fix did not cover.
    * ``shutdown()``'s ``None`` sentinel stayed in the maxsize=1 queue (see
      the sibling test below).

    Both are closed at the root rather than field-by-field: one worker per
    object, so per-worker state cannot be inherited by anybody. The assertions
    are therefore "no second worker exists", not "the second worker was reset
    correctly".
    """
    builds = {"n": 0}

    def _counting_build(_cfg):
        builds["n"] += 1
        return _StubChessNet(dim=4)

    monkeypatch.setattr("chess_anti_engine.train.async_eval.build_model", _counting_build)
    calls = {"n": 0}

    def _eval(**_kw):
        calls["n"] += 1
        return "OK"

    trainer = MagicMock()
    trainer.model = _StubChessNet(dim=4)
    trainer._compute_metrics = _eval

    aer = AsyncTestEval()
    _start_bounded(aer,
        trainer=trainer, model_cfg=MagicMock(), holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == ("OK", 1)
    assert (builds["n"], calls["n"]) == (1, 1)
    aer.shutdown(timeout=5.0)
    assert aer._compiled_shape_keys == {(1, 4)}, (
        "precondition: the dead worker's compiled-shape keys are still on the "
        "object -- the state a respawn would silently inherit"
    )

    with caplog.at_level("ERROR"):
        elapsed = _start_bounded(aer, bound_s=5.0,
            trainer=trainer, model_cfg=MagicMock(), holdout_buf="B",
            batch_size=4, steps=2, device="cpu", source_iter=2,
        )
    assert elapsed < 2.0, f"the refused start() took {elapsed:.2f}s"
    assert aer._thread is None, "shutdown() must be terminal -- no second worker"
    assert builds["n"] == 1, (
        "a second worker was spawned after shutdown(): it built a fresh "
        "snapshot and compiled it while start() returned, inheriting "
        "_compiled_shape_keys so no barrier ever engaged (review finding P1b)"
    )
    assert calls["n"] == 1, "a refused start() must not run an eval"
    # Refused, not silently dropped: an error in the log, and collect() reports
    # the failure the same way it reports an eval that raised.
    assert any(
        "after shutdown()" in r.getMessage() for r in caplog.records
    ), "the refusal must be audible -- a silently ignored start() is worse"
    assert aer.collect(timeout=5.0) == (None, -1)
    aer.shutdown(timeout=5.0)  # idempotent


def test_start_after_shutting_down_a_dead_worker_is_refused_instead_of_hanging(monkeypatch):
    """The OTHER respawn case: worker died at init, then shutdown, then start.

    Review finding P1 (PR #405), measured at the previous head: ``start()``
    had not returned after 10s, the queue held 1 item and no thread was alive
    — a PERMANENT hang, because ``shutdown()`` leaves a ``None`` sentinel in
    the maxsize=1 queue that a dead worker never dequeued, the respawned
    worker takes that sentinel and exits before running anything, and the
    barrier's wait is unbounded.

    This is the case the deleted ``_init_failed = False`` reset was written
    for, and it is reachable ONLY through ``shutdown()``: that is the only
    writer of ``_thread = None``, so a worker that dies at init is otherwise
    never replaced (``start()`` refuses on the dead thread instead). Refusing
    after shutdown therefore covers both respawn cases, which is why the reset
    could go rather than being made correct.
    """
    boom = {"on": True}
    builds = {"n": 0}

    def _maybe_boom(_cfg):
        builds["n"] += 1
        if boom["on"]:
            raise RuntimeError("synthetic init failure")
        return _StubChessNet(dim=4)

    monkeypatch.setattr("chess_anti_engine.train.async_eval.build_model", _maybe_boom)
    trainer = MagicMock()
    trainer.model = _StubChessNet(dim=4)
    trainer._compute_metrics = MagicMock(return_value="OK")

    aer = AsyncTestEval()
    _start_bounded(aer, bound_s=10.0,
        trainer=trainer, model_cfg=MagicMock(), holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == (None, -1)
    assert aer._init_failed is True

    aer.shutdown(timeout=5.0)
    assert aer._work_q.qsize() == 1, (
        "precondition: shutdown()'s sentinel is still queued because the dead "
        "worker never dequeued it -- what a respawned worker would consume "
        "instead of its own work"
    )

    boom["on"] = False  # a fresh worker WOULD now succeed; it must not exist
    elapsed = _start_bounded(aer, bound_s=5.0,
        trainer=trainer, model_cfg=MagicMock(), holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=2,
    )
    assert elapsed < 2.0, f"the refused start() took {elapsed:.2f}s"
    assert aer._thread is None, "shutdown() must be terminal -- no second worker"
    assert builds["n"] == 1, "a second worker was spawned onto a poisoned queue"
    trainer._compute_metrics.assert_not_called()
    assert aer.collect(timeout=5.0) == (None, -1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
