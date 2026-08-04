"""A broker must not report unanswered rows as served throughput.

Task #142 and its re-review. The family PR #325 named is *a serve loop that
counts work at DISPATCH time plus a callee that can decline it*. B6 fixed it in
``MultiSlotInferenceClient`` (a timed-out request transported nothing yet
counted as ``lifetime_positions``) and B6(c) in
``SharedSlotBroker._process_parallel`` (rows refused for an unimplemented
``request_mode``).

⚑ **"Three known instances" was WRONG, and the way it was wrong is the lesson.**
This file's first version closed the family on a grep for counter INCREMENTS
(``+= 1``, ``+= sum``, ...). But the defect IS THE ABSENCE OF AN INCREMENT, so
that instrument cannot see it -- it can only find sites that already count. The
enumerating instrument is the sites that RELEASE work
(``_release_slots_for_retry(``, ``state = _STATE_IDLE``), each checked against
whether a dispatch-time counter already counted those rows. Re-enumerated that
way, ``SharedSlotBroker._process_parallel``'s NO-MODEL branch turned out to
release every one of a trial's slots with no increment at all: instance four,
found by the reviewer, fixed and tested at the bottom of this file.

The production ``SlotBroker`` is the instance this file was opened for.

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
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.inference import (
    _MODE_DENSE_F32,
    _MODE_LEGAL_BF16,
    _POLICY_SIZE,
    _STATE_IDLE,
    _STATE_REQUEST,
    _InferenceSlot,
    BrokerModelUnavailable,
    SharedSlotBroker,
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


# ---------------------------------------------------------------------------
# The FOURTH instance, found by the re-review: SharedSlotBroker's NO-MODEL trial
#
# Method lesson, recorded here because this file is where the next reader lands:
# the first enumeration grepped for counter INCREMENTS (`+= 1`, `+= sum`, ...)
# and concluded the family was closed. But this defect class IS THE ABSENCE OF
# AN INCREMENT, so that instrument is blind to it by construction. The
# enumerating instrument is the sites that RELEASE work --
# `_release_slots_for_retry(` and `state = _STATE_IDLE` -- each checked for
# whether a dispatch-time counter already counted those rows.
# ---------------------------------------------------------------------------


class _SharedStubSlot:
    """Enough of a slot to drive SharedSlotBroker's no-model branch.

    The policy buffer is full width because the served-trial control really
    does run a forward and scatter the result back; a narrow buffer would fail
    on the shape rather than on the counting claim.
    """

    def __init__(self, batch_size: int = 2) -> None:
        self.state = _STATE_REQUEST
        self.request_mode = _MODE_DENSE_F32
        self.request_id = 7
        self.batch_size = batch_size
        self.policy = np.zeros((batch_size, _POLICY_SIZE), dtype=np.float32)
        self.wdl = np.zeros((batch_size, 3), dtype=np.float32)
        self.input = np.zeros((batch_size, 4), dtype=np.float32)


class _SharedWatchdog:
    """The forward-hang instrument, stubbed to assert it is still driven."""

    def mark_forward_start(self, batch: int) -> int:
        assert batch >= 0
        return 1

    def mark_forward_done(self, *, success: bool, token: int) -> None:
        assert token == 1
        assert success in (True, False)


def _shared_broker(*, with_model: bool) -> SharedSlotBroker:
    broker = object.__new__(SharedSlotBroker)
    broker._trial_models = {"t0": _TinyPolicy().eval()} if with_model else {}
    broker._trial_no_model_warned_at = {}
    broker._trial_bad_mode_warned_at = {}
    broker._trial_streams = {}
    broker._first_inference_done = False
    broker._hang_watchdog = _SharedWatchdog()  # pyright: ignore[reportAttributeAccessIssue]
    broker.unsupported_mode_requests = 0
    broker.device = "cpu"
    return broker


def test_shared_broker_no_model_rows_are_not_counted_as_served() -> None:
    """RED on this PR's first head: `_process_parallel` returned 0 here.

    `SharedSlotBroker._process_parallel` sets every ready slot of a
    model-less trial to `_STATE_IDLE` and `continue`s, with no increment. The
    serve loop counted those rows at dispatch and subtracts only this return,
    so a trial whose weights had not loaded yet reported every row as served
    pos/s. Dormant today (`distributed_inference_shared_broker: false`) but on
    the #322 restart-gated deploy path, so it can go live at a restart.
    """
    broker = _shared_broker(with_model=False)
    slots = [_SharedStubSlot(batch_size=5), _SharedStubSlot(batch_size=3)]

    unanswered = broker._process_parallel({"t0": slots})  # pyright: ignore[reportArgumentType]

    assert [s.state for s in slots] == [_STATE_IDLE, _STATE_IDLE], (
        "sanity: the no-model branch IS what released these slots"
    )
    assert unanswered == 8, (
        "rows released for a model-less trial were never evaluated; returning "
        "0 here is the fourth instance of the dispatch-time counting family"
    )


def test_shared_broker_no_model_rows_are_counted_before_the_release() -> None:
    """Same read-before-release rule as the production broker's path 1."""
    broker = _shared_broker(with_model=False)

    class _ResubmitOnRelease(_SharedStubSlot):
        """A client that re-submits the instant its slot goes IDLE."""

        def __setattr__(self, name: str, value: Any) -> None:
            object.__setattr__(self, name, value)
            if name == "state" and value == _STATE_IDLE:
                object.__setattr__(self, "batch_size", 1)

    slots = [_ResubmitOnRelease(batch_size=5)]

    assert broker._process_parallel({"t0": slots}) == 5, (  # pyright: ignore[reportArgumentType]
        "the count must describe the rows this batch was dispatched with"
    )


def test_shared_broker_still_counts_a_served_trial_as_served() -> None:
    """Positive control: a trial WITH a model must subtract nothing."""
    broker = _shared_broker(with_model=True)
    slots = [_SharedStubSlot(batch_size=4)]

    assert broker._process_parallel(cast("Any", {"t0": slots})) == 0, (
        "a served trial declines no rows; a pessimistic counter is as wrong "
        "as the optimistic one it replaced"
    )


def test_shared_broker_counts_only_the_modelless_trial() -> None:
    """One trial's missing model must not cost another trial its throughput."""
    broker = _shared_broker(with_model=True)
    served = [_SharedStubSlot(batch_size=4)]
    starved = [_SharedStubSlot(batch_size=6)]

    unanswered = broker._process_parallel(
        cast("Any", {"t0": served, "no_model_trial": starved}),
    )

    assert unanswered == 6, "only the model-less trial's rows are unanswered"
    assert starved[0].state == _STATE_IDLE
    assert served[0].state != _STATE_IDLE


# ---------------------------------------------------------------------------
# The FIFTH instance by mechanism: the FIRST_BATCH_DONE boot marker
# ---------------------------------------------------------------------------


def test_first_batch_done_reports_rows_actually_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    broker = _serving(tmp_path, monkeypatch, num_slots=2)
    try:
        _arm_legal(broker, 0, malformed=True, batch_size=4)
        _arm_legal(broker, 1, malformed=False, batch_size=3)

        _run_one_serve_iteration(broker, monkeypatch)

        out = capsys.readouterr().out
        assert "FIRST_BATCH_DONE positions=3 " in out, (
            f"the boot marker must report the 3 rows served, not the 7 "
            f"dispatched; got:\n{out}"
        )
    finally:
        broker.shutdown()


def test_first_batch_done_does_not_fire_on_a_wholly_unanswered_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The marker is read as "this broker is serving". It must not lie.

    Pre-fix, a model-less broker -- the exact cold-boot failure the marker is
    watched for -- printed FIRST_BATCH_DONE with a full row count on its first
    batch, then never mentioned it again because the flag was already down.
    """
    broker = _modelless(tmp_path, monkeypatch)
    try:
        _arm_dense(broker, batch_size=6)

        metrics = _run_one_serve_iteration(broker, monkeypatch)

        out = capsys.readouterr().out
        assert metrics["batches"] == 1, "sanity: a batch WAS dispatched"
        assert "FIRST_BATCH_DONE" not in out, (
            f"a batch in which nothing was served is not a first batch; "
            f"got:\n{out}"
        )
        assert not broker._first_batch_logged, (
            "the flag must stay down so the marker can still fire on the "
            "first batch that really is served"
        )
    finally:
        broker.shutdown()


def test_first_batch_done_fires_on_the_first_genuinely_served_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Suppressing the marker must not DELETE it -- it moves to the real batch."""
    broker = _serving(tmp_path, monkeypatch)
    try:
        slot = _arm_dense(broker, batch_size=6)
        # Batch 1: the model is not there yet, so nothing is served.
        broker._model = None
        monkeypatch.setattr(SlotBroker, "_ensure_model", lambda _self: None)
        assert broker._process_batch([slot]) == 6
        assert not broker._first_batch_logged

        # Batch 2: the model arrived.
        broker._model = _TinyPolicy().eval()
        _arm_dense(broker, batch_size=6)
        _run_one_serve_iteration(broker, monkeypatch)

        out = capsys.readouterr().out
        assert "FIRST_BATCH_DONE positions=6 " in out, (
            f"the marker must fire once the broker genuinely serves; got:\n{out}"
        )
    finally:
        broker.shutdown()


# ---------------------------------------------------------------------------
# The clamp: dispatch counted rows the gather would never evaluate
# ---------------------------------------------------------------------------


def test_an_oversized_batch_size_is_not_counted_beyond_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gather clamps to ``max_batch``; the dispatch count must clamp too.

    Found by the release-site re-enumeration as the one site with a THIRD
    disposition: not a release at all, but still "counted at dispatch, never
    evaluated". A client that writes a batch_size above the layout capacity had
    the excess reported as served throughput. Only reachable through a
    client-side protocol violation, which is why it is small -- and exactly why
    it survived: nothing on the healthy path can show it.

    ``max_batch_per_slot`` is 8 here, so a slot claiming 12 rows can only ever
    contribute 8.
    """
    broker = _serving(tmp_path, monkeypatch)
    try:
        slot = _arm_dense(broker, batch_size=8)
        slot.batch_size = 12  # over capacity, past _arm_dense's own bound

        metrics = _run_one_serve_iteration(broker, monkeypatch)

        assert metrics["positions"] == 8, (
            "only the 8 rows the gather can evaluate may be counted; counting "
            "the claimed 12 reports 4 rows of throughput that never existed"
        )
    finally:
        broker.shutdown()
