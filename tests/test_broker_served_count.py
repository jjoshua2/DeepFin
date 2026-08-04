"""The production broker must not report unanswered rows as served throughput.

Task #142, the third and last of the three known instances of the family PR
#325 named: *a serve loop that counts work at DISPATCH time plus a callee that
can decline it*. B6 fixed it in ``MultiSlotInferenceClient`` (a timed-out
request transported nothing yet counted as ``lifetime_positions``) and B6(c) in
``SharedSlotBroker._process_parallel`` (rows refused for an unimplemented
``request_mode``). This is the PRODUCTION broker, which the other two are only
siblings of.

``SlotBroker.serve_forever`` does::

    metrics["positions"] += sum(s.batch_size for s in ready)
    self._process_batch(ready)

and ``_process_batch`` hands slots back UNANSWERED on three paths, every one of
them a real production failure mode:

1. ``BrokerModelUnavailable`` -- the publish manifest is missing past its 30s
   deadline or carries no sha. This is the cold-boot failure, and it repeats
   every serve loop until a model appears.
2. A generic batch failure -- the broker deliberately survives one bad batch
   rather than taking inference away from the whole fleet.
3. Malformed compact-legal metadata -- a per-SLOT release inside
   ``_process_batch_mode``, so a batch can be partly served and partly refused.

On all three the slots go to ``_STATE_IDLE`` and the client re-submits, so no
model ever evaluated those rows -- but the broker's own ``pos/s`` line counted
them. A broker answering nothing at all reported full throughput, which is the
number an operator uses to decide whether the fleet is healthy.

The fix mirrors B6(c) exactly: the callee returns the unanswered ROW count and
the serve loop subtracts it. ``batches`` and ``slots`` stay dispatch counts on
purpose -- they answer "how much did the loop try to batch", which a refusal
does not change.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.inference import (
    _MODE_DENSE_F32,
    _MODE_LEGAL_BF16,
    _POLICY_SIZE,
    _STATE_REQUEST,
    _InferenceSlot,
    BrokerModelUnavailable,
    SlotBroker,
)


class _TinyPolicy(torch.nn.Module):
    """A model that answers, so a SERVED row can be told from a refused one.

    Every test that asserts "this many rows were served" needs a forward that
    completes; without one, dropping the accumulation entirely would pass
    every refusal test.
    """

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = int(x.shape[0])
        return {
            "policy": torch.zeros(
                (bsz, _POLICY_SIZE), dtype=torch.float32, device=x.device,
            ),
            "wdl": torch.full((bsz, 3), 1.0 / 3.0, dtype=torch.float32, device=x.device),
        }


def _make_broker(tmp_path: Path, *, num_slots: int = 1) -> SlotBroker:
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    return SlotBroker(
        publish_dir=publish_dir,
        num_slots=num_slots,
        max_batch_per_slot=8,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-served-{uuid.uuid4().hex}",
    )


def _modelless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, num_slots: int = 1,
) -> SlotBroker:
    """A broker whose model never loads, without paying _ensure_model's 30s wait.

    Patching ``_ensure_model`` to a no-op reproduces both of its silent-return
    paths in the only way that matters here: ``self._model`` stays ``None``, so
    ``_process_batch_mode`` raises ``BrokerModelUnavailable``.
    """
    broker = _make_broker(tmp_path, num_slots=num_slots)
    monkeypatch.setattr(SlotBroker, "_ensure_model", lambda _self: None)
    assert broker._model is None
    return broker


def _serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, num_slots: int = 1,
) -> SlotBroker:
    """A broker with a working model, so served rows are genuinely served."""
    broker = _make_broker(tmp_path, num_slots=num_slots)
    monkeypatch.setattr(SlotBroker, "_ensure_model", lambda _self: None)
    broker._model = _TinyPolicy().eval()
    broker._model_sha = "test"
    return broker


def _arm_dense(broker: SlotBroker, idx: int = 0, *, batch_size: int = 2) -> _InferenceSlot:
    slot = broker._slots[idx]
    slot.batch_size = batch_size
    slot.request_mode = _MODE_DENSE_F32
    slot.request_id = 11 + idx
    slot.state = _STATE_REQUEST
    return slot


def _arm_legal(
    broker: SlotBroker, idx: int = 0, *, malformed: bool, batch_size: int = 2,
) -> _InferenceSlot:
    """Arm a compact-legal slot, optionally with metadata that fails validation.

    The malformation is a header/counts disagreement -- the exact shape of the
    torn read this rejection exists for (the client rewrote the slot while the
    broker was mid-read).
    """
    slot = broker._slots[idx]
    counts = np.full((batch_size,), 2, dtype=np.int32)
    n_legal = int(counts.sum())
    flat = np.arange(n_legal, dtype=np.int32)
    header = n_legal - 1 if malformed else n_legal
    slot.policy_i32[0] = header
    slot.policy_i32[1:1 + batch_size] = counts
    slot.policy_i32[1 + batch_size:1 + batch_size + n_legal] = flat
    slot.batch_size = batch_size
    slot.request_mode = _MODE_LEGAL_BF16
    slot.request_id = 21 + idx
    slot.state = _STATE_REQUEST
    return slot


def _run_one_serve_iteration(
    broker: SlotBroker, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Drive the REAL serve loop for exactly one batch and return its metrics.

    ``serve_forever`` publishes its metrics dict on ``self._timing_metrics``, so
    the accumulator the pos/s line is computed from can be read directly rather
    than scraped out of stdout. The 10s report interval has not elapsed, so
    nothing resets it.

    The wrapper stops the loop and otherwise forwards the return value
    untouched -- the subtraction under test is the loop's own.
    """
    original = SlotBroker._process_batch

    def _once(self: SlotBroker, ready: list[_InferenceSlot]) -> int:
        try:
            return original(self, ready)
        finally:
            self._stop = True

    monkeypatch.setattr(SlotBroker, "_process_batch", _once)
    broker.serve_forever()
    metrics = broker._timing_metrics
    assert metrics is not None, "the serve loop never dispatched a batch"
    return metrics


# ---------------------------------------------------------------------------
# Path 1 of 3: BrokerModelUnavailable
# ---------------------------------------------------------------------------


def test_modelless_batch_reports_its_rows_unanswered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _modelless(tmp_path, monkeypatch)
    try:
        slot = _arm_dense(broker, batch_size=5)

        unanswered = broker._process_batch([slot])

        assert unanswered == 5, (
            "the model-less path releases every row unanswered; the serve loop "
            "counted them at dispatch and has no other way to learn that"
        )
    finally:
        broker.shutdown()


def test_modelless_rows_are_counted_before_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering is load-bearing, not incidental.

    ``_release_slots_for_retry`` sets ``_STATE_IDLE``, at which point the client
    may re-submit into the same shared slot with a different ``batch_size``.
    Reading the field after the release would subtract a row count belonging to
    the NEXT request. Simulated here by a client that re-submits the instant the
    slot is released.
    """
    broker = _modelless(tmp_path, monkeypatch)
    try:
        slot = _arm_dense(broker, batch_size=5)
        original_release = SlotBroker._release_slots_for_retry

        def _release_then_resubmit(
            self: SlotBroker, slots: list[_InferenceSlot],
        ) -> None:
            original_release(self, slots)
            for s in slots:
                s.batch_size = 1  # the next request, not this one

        monkeypatch.setattr(
            SlotBroker, "_release_slots_for_retry", _release_then_resubmit,
        )

        assert broker._process_batch([slot]) == 5, (
            "the count must come from the rows this batch was dispatched with"
        )
    finally:
        broker.shutdown()


# ---------------------------------------------------------------------------
# Path 2 of 3: a generic batch failure
# ---------------------------------------------------------------------------


def test_failed_batch_reports_its_rows_unanswered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _serving(tmp_path, monkeypatch)
    try:
        slot = _arm_dense(broker, batch_size=3)

        def _boom(
            _self: SlotBroker,
            _ready: list[_InferenceSlot],
            *,
            mode: int,
            request_ids: list[int],
        ) -> int:
            raise ValueError(f"boom mode={mode} ids={request_ids}")

        monkeypatch.setattr(SlotBroker, "_process_batch_mode", _boom)

        unanswered = broker._process_batch([slot])

        assert unanswered == 3, (
            "one bad batch is survivable by design, but its rows were never "
            "evaluated and must not read as throughput"
        )
        assert broker._consecutive_batch_failures == 1, (
            "sanity: the generic-failure branch is the one that ran"
        )
    finally:
        broker.shutdown()


# ---------------------------------------------------------------------------
# Path 3 of 3: malformed compact-legal metadata
# ---------------------------------------------------------------------------


def test_malformed_legal_metadata_reports_its_rows_unanswered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _serving(tmp_path, monkeypatch)
    try:
        slot = _arm_legal(broker, malformed=True, batch_size=4)

        unanswered = broker._process_batch([slot])

        assert broker._malformed_legal_meta_total == 1, (
            "sanity: the malformed-metadata branch is the one that ran"
        )
        assert unanswered == 4, (
            "a per-slot release inside _process_batch_mode still owes the "
            "serve loop its rows"
        )
    finally:
        broker.shutdown()


def test_a_partly_malformed_batch_reports_only_the_bad_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The discriminating case: same batch, one slot refused and one served.

    A fix that subtracted the whole batch, or none of it, passes the
    single-slot tests and fails this one.
    """
    broker = _serving(tmp_path, monkeypatch, num_slots=2)
    try:
        bad = _arm_legal(broker, 0, malformed=True, batch_size=4)
        good = _arm_legal(broker, 1, malformed=False, batch_size=3)

        unanswered = broker._process_batch([bad, good])

        assert broker._malformed_legal_meta_total == 1
        assert unanswered == 4, (
            "only the refused slot's rows are unanswered; the other slot was "
            "evaluated and IS throughput"
        )
    finally:
        broker.shutdown()


# ---------------------------------------------------------------------------
# The happy path must stay counted (the accumulation, not just the subtraction)
# ---------------------------------------------------------------------------


def test_a_served_batch_reports_nothing_unanswered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _serving(tmp_path, monkeypatch)
    try:
        slot = _arm_dense(broker, batch_size=6)

        assert broker._process_batch([slot]) == 0, (
            "a healthy batch must subtract nothing; a pessimistic counter is "
            "as wrong as the optimistic one it replaced"
        )
    finally:
        broker.shutdown()


# ---------------------------------------------------------------------------
# The serve loop itself: accumulation AND subtraction
# ---------------------------------------------------------------------------


def test_the_serve_loop_counts_served_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control. Deleting the dispatch-time accumulation turns this red.

    Without it every other test here would still pass while ``pos/s`` read
    zero forever, which is the mirror-image defect.
    """
    broker = _serving(tmp_path, monkeypatch)
    try:
        _arm_dense(broker, batch_size=6)

        metrics = _run_one_serve_iteration(broker, monkeypatch)

        assert metrics["batches"] == 1
        assert metrics["positions"] == 6, (
            "rows a model evaluated must be counted as served"
        )
    finally:
        broker.shutdown()


def test_the_serve_loop_does_not_count_unanswered_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The red-on-main assertion: pre-fix this reads 6, not 0.

    End-to-end through the real ``serve_forever`` accumulator, so it fails if
    the subtraction is dropped from the loop even while ``_process_batch``
    still returns the right number.
    """
    broker = _modelless(tmp_path, monkeypatch)
    try:
        _arm_dense(broker, batch_size=6)

        metrics = _run_one_serve_iteration(broker, monkeypatch)

        assert metrics["batches"] == 1, (
            "sanity: the batch WAS dispatched, so the rows were counted"
        )
        assert metrics["positions"] == 0, (
            "a broker with no model served nothing; reporting full pos/s is "
            "the failure-reads-as-throughput defect (task #142)"
        )
    finally:
        broker.shutdown()


def test_the_serve_loop_counts_only_the_served_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both directions in one loop iteration, end to end."""
    broker = _serving(tmp_path, monkeypatch, num_slots=2)
    try:
        _arm_legal(broker, 0, malformed=True, batch_size=4)
        _arm_legal(broker, 1, malformed=False, batch_size=3)

        metrics = _run_one_serve_iteration(broker, monkeypatch)

        assert metrics["batches"] == 1
        assert metrics["slots"] == 2, (
            "slots stays a DISPATCH count on purpose: it answers how much the "
            "loop tried to batch"
        )
        assert metrics["positions"] == 3, (
            "4 refused + 3 served must report 3, not 7 and not 0"
        )
    finally:
        broker.shutdown()


def test_the_escalating_failure_still_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning a count must not swallow the exit path.

    ``_process_batch`` re-raises once ``_MAX_CONSECUTIVE_BATCH_FAILURES`` is
    reached so the next selfplay phase relaunches a clean broker. A refactor
    that turned every failure into a return value would take that away.
    """
    from chess_anti_engine.inference import _MAX_CONSECUTIVE_BATCH_FAILURES

    broker = _modelless(tmp_path, monkeypatch)
    try:
        slot = _arm_dense(broker, batch_size=2)
        for _ in range(_MAX_CONSECUTIVE_BATCH_FAILURES - 1):
            slot.state = _STATE_REQUEST
            assert broker._process_batch([slot]) == 2
        slot.state = _STATE_REQUEST
        with pytest.raises(BrokerModelUnavailable, match="no model loaded"):
            broker._process_batch([slot])
    finally:
        broker.shutdown()
