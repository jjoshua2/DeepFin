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

    def _compute_metrics(*, buf, batch_size, steps, tag, model_override=None):
        del buf, tag  # consumed only via captured below
        captured["model_override"] = model_override
        captured["batch_size"] = batch_size
        captured["steps"] = steps
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
    """If a prior eval is still running, start() warns + spawns a new one
    rather than raising. The orphaned thread is daemon — it'll exit on
    trial teardown. This avoids a latent crash when an eval times out
    in collect()."""
    import threading
    model = _StubChessNet(dim=4)
    block = threading.Event()

    def _slow_eval(**_kw):
        block.wait(timeout=5.0)
        return "X"

    trainer = MagicMock()
    trainer.model = model
    trainer._compute_metrics = _slow_eval

    aer = AsyncTestEval()
    aer.start(
        trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
        batch_size=4, steps=2, device="cpu", source_iter=1,
    )
    with caplog.at_level("WARNING"):
        aer.start(
            trainer=trainer, model_cfg=cfg_and_builder, holdout_buf="B",
            batch_size=4, steps=2, device="cpu", source_iter=2,
        )
    assert any("previous eval" in r.getMessage() and "still running" in r.getMessage() for r in caplog.records)
    block.set()
    aer.collect(timeout=5.0)


def test_snapshot_isolated_from_trainer_model(cfg_and_builder):
    """Mutating trainer.model after start() must not affect snapshot's eval."""
    model = _StubChessNet(dim=4)
    snap_seen = {}

    def _compute_metrics_capturing(*, buf, batch_size, steps, tag, model_override=None):
        del buf, batch_size, steps, tag  # only model_override matters here
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
    assert metrics is None and src == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
