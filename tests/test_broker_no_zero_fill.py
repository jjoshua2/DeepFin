"""A broker with no model must not answer with a fabricated all-zero policy.

``SlotBroker._release_slots_for_retry`` states the rule outright — *"Deliberately
NOT a zero-filled response: the clients feed selfplay, and a fabricated all-zero
policy would be recorded as training data rather than raising"* — but two other
paths in the same file did exactly that. ``_ensure_model`` returns silently when
the publish manifest is still missing at its 30s deadline or carries no model
sha, and both brokers then filled zeros and marked the slot ``_STATE_RESPONSE``.

That is worse than a crash. An all-zero policy+WDL is indistinguishable from a
real answer, so it reaches MCTS and is recorded as training data; and because
the per-trial path *returned normally*, ``_process_batch`` scored it a success
and reset ``_consecutive_batch_failures``, so a broker that never got a model
served zeros forever with no log line and no escalation.

Supersedes ``test_slot_broker_zeroes_outputs_when_model_unavailable``, which
asserted the old behaviour with no stated rationale.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.inference import (
    _MAX_CONSECUTIVE_BATCH_FAILURES,
    _MODE_DENSE_F32,
    _STATE_IDLE,
    _STATE_REQUEST,
    _STATE_RESPONSE,
    BrokerModelUnavailable,
    SharedSlotBroker,
    SlotBroker,
)

_SENTINEL = 1234.5


def _make_modelless_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> SlotBroker:
    """A broker whose model never loads, without paying _ensure_model's 30s wait.

    Patching ``_ensure_model`` to a no-op reproduces both of its silent-return
    paths (missing manifest, manifest without a sha) in the only way that
    matters here: ``self._model`` stays ``None``.
    """
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    broker = SlotBroker(
        publish_dir=publish_dir,
        num_slots=1,
        max_batch_per_slot=8,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-nozero-{uuid.uuid4().hex}",
    )
    monkeypatch.setattr(SlotBroker, "_ensure_model", lambda _self: None)
    assert broker._model is None
    return broker


def _arm_slot(broker: SlotBroker) -> object:
    slot = broker._slots[0]
    slot.batch_size = 2
    slot.request_mode = _MODE_DENSE_F32
    slot.state = _STATE_REQUEST
  # Poison the response buffers: if anything zero-fills them we can tell the
  # difference between "left alone" and "answered with zeros".
    slot.policy[:2].fill(_SENTINEL)
    slot.wdl[:2].fill(_SENTINEL)
    return slot


def test_modelless_batch_releases_slots_instead_of_answering_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _make_modelless_broker(tmp_path, monkeypatch)
    try:
        slot = _arm_slot(broker)

        broker._process_batch([slot])  # pyright: ignore[reportArgumentType]

        assert slot.state == _STATE_IDLE, (  # pyright: ignore[reportAttributeAccessIssue]
            "a model-less batch must be released for retry, not answered"
        )
        assert slot.state != _STATE_RESPONSE  # pyright: ignore[reportAttributeAccessIssue]
      # The response buffers must be untouched — no fabricated zeros.
        assert np.allclose(slot.policy[:2], _SENTINEL)  # pyright: ignore[reportAttributeAccessIssue]
        assert np.allclose(slot.wdl[:2], _SENTINEL)  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        broker.shutdown()


def test_modelless_batch_counts_toward_the_failure_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The old path reset the counter, so a model-less broker never escalated."""
    broker = _make_modelless_broker(tmp_path, monkeypatch)
    try:
        slot = _arm_slot(broker)
        broker._process_batch([slot])  # pyright: ignore[reportArgumentType]

        assert broker._consecutive_batch_failures == 1, (
            "a model-less batch must count as a failure; resetting the counter "
            "is what let the broker serve zeros indefinitely"
        )
    finally:
        broker.shutdown()


def test_persistently_modelless_broker_eventually_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Escalation must actually fire, so the next phase relaunches a clean broker."""
    broker = _make_modelless_broker(tmp_path, monkeypatch)
    try:
        slot = _arm_slot(broker)
        for _ in range(_MAX_CONSECUTIVE_BATCH_FAILURES - 1):
            slot.state = _STATE_REQUEST  # pyright: ignore[reportAttributeAccessIssue]
            broker._process_batch([slot])  # pyright: ignore[reportArgumentType]

        slot.state = _STATE_REQUEST  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(BrokerModelUnavailable, match="no model loaded"):
            broker._process_batch([slot])  # pyright: ignore[reportArgumentType]
    finally:
        broker.shutdown()


class _StubSlot:
    """Only the attributes SharedSlotBroker's no-model branch touches."""

    def __init__(self) -> None:
        self.state = _STATE_REQUEST


def _shared_broker_skeleton(models: dict[str, object]) -> SharedSlotBroker:
    """A SharedSlotBroker with just enough state to drive _process_parallel.

    Built with ``object.__new__`` deliberately: the real constructor stands up
    shared memory, CUDA streams and a manifest watcher, none of which the
    no-model branch reaches. Keeping the fixture this thin is what makes the
    test worth having -- it pins the release policy, not the plumbing.
    """
    broker = object.__new__(SharedSlotBroker)
    broker._trial_models = models  # pyright: ignore[reportAttributeAccessIssue]
    broker._trial_no_model_warned_at = {}
    broker.device = "cpu"
    return broker


def test_shared_broker_releases_slots_for_a_trial_with_no_model() -> None:
    broker = _shared_broker_skeleton({})
    slots = [_StubSlot(), _StubSlot()]

    broker._process_parallel({"trial-a": slots})  # pyright: ignore[reportArgumentType]

    assert [s.state for s in slots] == [_STATE_IDLE, _STATE_IDLE], (
        "a trial with no model must have its slots released, not answered"
    )
    assert all(s.state != _STATE_RESPONSE for s in slots)


def test_shared_broker_no_model_warning_is_throttled() -> None:
    """Released slots come straight back round the serve loop.

    Without the throttle this warns on every pass, thousands of lines a
    second, burying whatever the operator actually needs to see.
    """
    broker = _shared_broker_skeleton({})
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Capture()
    inference_log = logging.getLogger("chess_anti_engine.inference")
    inference_log.addHandler(handler)
    try:
        for _ in range(25):
            broker._process_parallel(
                {"trial-a": [_StubSlot()]},  # pyright: ignore[reportArgumentType]
            )
    finally:
        inference_log.removeHandler(handler)

    hits = [r for r in records if "no model for trial" in r]
    assert len(hits) == 1, f"expected one throttled warning, got {len(hits)}"
