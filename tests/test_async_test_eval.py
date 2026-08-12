"""Tests for AsyncTestEval lifecycle and snapshot isolation."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
import torch

from chess_anti_engine.train.async_eval import AsyncTestEval


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
    aer.start(
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
    aer.start(
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == ("FIRST", 1)
    aer.start(
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=2,
    )
    with caplog.at_level("WARNING"):
        aer.start(
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

    def _observer():
        # While the eval is gated, start() must still be blocked. Record the
        # observation, then release the eval so start() can return.
        time.sleep(0.3)
        overlap_seen["value"] = start_returned.is_set()
        release.set()

    obs = threading.Thread(target=_observer)
    obs.start()
    aer = AsyncTestEval()
    aer.start(
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    start_returned.set()
    obs.join(timeout=10.0)
    # The eval must have completed BEFORE start() returned...
    assert eval_finished.is_set(), "first start() returned before the first eval ran"
    # ...and while the eval was still gated, start() had not yet returned.
    assert overlap_seen["value"] is False, (
        "first start() returned while the first eval (the compile) was still "
        "running -- the compile barrier is gone and the FX-trace race is back"
    )
    assert aer.collect(timeout=5.0) == ("GATED", 1)


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
    aer.start(
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    assert aer.collect(timeout=5.0) == ("FIRST", 1)

    t0 = time.monotonic()
    aer.start(
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
    aer.start(
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
    aer.start(
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
    aer.start(
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
    aer.start(
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=10,
    )
    m1, s1 = aer.collect(timeout=5.0)

    model.linear.weight.data.fill_(2.0)
    aer.start(
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


def test_shutdown_joins_thread(cfg_and_builder):
    model = _StubChessNet(dim=4)
    trainer, _ = _make_trainer_stub(model, eval_payload="OK")
    aer = AsyncTestEval()
    aer.start(
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    aer.collect(timeout=5.0)
    aer.shutdown(timeout=5.0)
    assert aer._thread is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
