"""The shared broker must refuse a transport it cannot serve, not fake one.

SHARED_BROKER_AUDIT B1. ``SlotBroker._process_batch`` snapshots
``slot.request_mode``, groups by it and dispatches all three modes.
``SharedSlotBroker._process_parallel`` never read the field at all: it read
``slot.input`` (float32) and wrote ``slot.policy`` (dense float32)
unconditionally.

Because ``_InferenceSlot`` ALIASES the regions — ``input``/``input_bf16_bits``
share ``input_offset``, ``policy``/``policy_i32``/``policy_u16`` share
``policy_offset`` — a bf16-bits payload was read as float32, the model ran on
garbage, and the client decoded the dense f32 answer as bf16. Measured
(``sb_shared_broker_ignores_mode.py``): an in-range plausible WDL and a policy
of alternating 3.85e-13 / 0.488 where every entry should have been ~0.51. No
exception, no counter, no log line.

Both production transports send the modes with no dispatch here
(``evaluate_encoded``'s bf16 path and ``evaluate_legal_bf16``;
``MultiSlotInferenceClient`` hardcodes ``supports_compact_root_policy`` and
``supports_input_bf16_bits`` to True), so dense f32 is the only mode that ever
worked and no production client ever sends it. The shared broker is OFF
(``distributed_inference_shared_broker: false``), so the fix is to refuse
loudly rather than to implement dispatch nothing exercises.

Note what the existing coverage did: every shared-broker test in
``test_broker_no_zero_fill.py`` sets ``request_mode = _MODE_DENSE_F32``, i.e.
the only mode that worked was the only mode tested.
"""
from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pytest

from chess_anti_engine.inference import (
    _MODE_DENSE_BF16,
    _MODE_DENSE_F32,
    _MODE_LEGAL_BF16,
    _STATE_IDLE,
    _STATE_REQUEST,
    _STATE_RESPONSE,
    SharedSlotBroker,
)

_SENTINEL = 1234.5


class _StubSlot:
    def __init__(self, mode: int, batch_size: int = 2) -> None:
        self.state = _STATE_REQUEST
        self.request_mode = int(mode)
        self.request_id = 7
        self.batch_size = batch_size
        self.policy = np.full((batch_size, 8), _SENTINEL, dtype=np.float32)
        self.wdl = np.full((batch_size, 3), _SENTINEL, dtype=np.float32)
        self.input = np.zeros((batch_size, 4), dtype=np.float32)


class _Model:
    """A model that announces the forward was reached.

    Present (rather than absent) so the no-model branch is not what does the
    releasing, and loud so "the slot got past the mode check" is an assertion
    rather than an inference from an incidental AttributeError.
    """

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(f"forward reached ({len(args)} args, {len(kwargs)} kw)")


class _Watchdog:
    def mark_forward_start(self, batch: int) -> int:
        assert batch >= 0
        return 1

    def mark_forward_done(self, *, success: bool, token: int) -> None:
        assert token == 1
        assert success in (True, False)


def _broker() -> SharedSlotBroker:
    broker = object.__new__(SharedSlotBroker)
    broker._trial_models = {"t0": _Model()}  # pyright: ignore[reportAttributeAccessIssue]
    broker._trial_no_model_warned_at = {}
    broker._trial_bad_mode_warned_at = {}
    broker._trial_streams = {}
    broker._first_inference_done = False
    broker._hang_watchdog = _Watchdog()  # pyright: ignore[reportAttributeAccessIssue]
    broker.unsupported_mode_requests = 0
    broker.device = "cpu"
    return broker


@pytest.mark.parametrize(
    ("mode", "name"),
    [(_MODE_DENSE_BF16, "dense bf16"), (_MODE_LEGAL_BF16, "compact legal bf16")],
)
def test_unsupported_mode_is_released_unanswered(mode: int, name: str) -> None:
    broker = _broker()
    slots = [_StubSlot(mode)]

    broker._process_parallel({"t0": slots})  # pyright: ignore[reportArgumentType]

    assert slots[0].state == _STATE_IDLE, (
        f"{name} must be released for retry, not answered"
    )
    assert slots[0].state != _STATE_RESPONSE
    # Untouched buffers: pre-fix these carried a dense f32 policy the client
    # then re-read as bf16.
    assert np.allclose(slots[0].policy, _SENTINEL)
    assert np.allclose(slots[0].wdl, _SENTINEL)
    assert broker.unsupported_mode_requests == 1


def test_the_refusal_names_the_missing_dispatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broker = _broker()
    with caplog.at_level(logging.ERROR, logger="chess_anti_engine.inference"):
        broker._process_parallel({"t0": [_StubSlot(_MODE_LEGAL_BF16)]})  # pyright: ignore[reportArgumentType]
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert messages, "an unimplemented transport must not be silent"
    joined = " ".join(messages)
    assert "_MODE_DENSE_F32" in joined
    assert "_MODE_LEGAL_BF16" in joined
    assert "SlotBroker" in joined, (
        "the message must name the broker that DOES implement the mode"
    )


def test_dense_f32_still_reaches_the_forward() -> None:
    """The one mode with a dispatch must not be swept up by the refusal.

    ``_Model`` raises on call, so reaching "forward reached" IS the assertion:
    the slot got past the mode check, into the packed batch, and through the
    hang watchdog into the forward. The mode counter stayed at zero.
    """
    broker = _broker()
    slots = [_StubSlot(_MODE_DENSE_F32)]
    with pytest.raises(RuntimeError, match="forward reached"):
        broker._process_parallel({"t0": slots})  # pyright: ignore[reportArgumentType]
    assert broker.unsupported_mode_requests == 0
    assert slots[0].state != _STATE_IDLE, "a servable mode must not be released"


def test_a_mixed_batch_refuses_only_the_unsupported_slots() -> None:
    """One client's compact request must not cost the dense clients their answer."""
    broker = _broker()
    good = _StubSlot(_MODE_DENSE_F32)
    bad = _StubSlot(_MODE_LEGAL_BF16)
    with pytest.raises(RuntimeError, match="forward reached"):
        broker._process_parallel({"t0": [good, bad]})  # pyright: ignore[reportArgumentType]
    assert bad.state == _STATE_IDLE
    assert good.state != _STATE_IDLE
    assert broker.unsupported_mode_requests == 1


def test_the_refusal_log_is_throttled() -> None:
    """A released slot comes straight back round the serve loop; an unthrottled
    line here writes thousands a second and buries the reason."""
    broker = _broker()
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Capture()
    log = logging.getLogger("chess_anti_engine.inference")
    log.addHandler(handler)
    try:
        for _ in range(25):
            broker._process_parallel({"t0": [_StubSlot(_MODE_LEGAL_BF16)]})  # pyright: ignore[reportArgumentType]
    finally:
        log.removeHandler(handler)
    hits = [m for m in records if "cannot serve request_mode" in m]
    assert len(hits) == 1, hits
    # Throttling the LOG must not throttle the COUNT.
    assert broker.unsupported_mode_requests == 25


def test_refused_rows_are_not_reported_as_served_throughput() -> None:
    """``_process_parallel`` returns the refused ROW count.

    The serve loop counts positions BEFORE calling it, so without the return the
    broker's own `pos/s` would include rows it declined to evaluate — the same
    "a failure reads as throughput" defect B6 fixed on the client side, in the
    sibling path. Reviewer finding: the counter also needs a reader, and a
    reader next to a lying number is not worth much.
    """
    broker = _broker()
    refused = broker._process_parallel(
        cast("Any", {"t0": [_StubSlot(_MODE_LEGAL_BF16, batch_size=5)]}),
    )
    assert refused == 5, "refused rows must be reported back to the serve loop"

    # A served batch refuses nothing (the dummy model's forward raises, which is
    # after the count is decided).
    broker = _broker()
    with pytest.raises(RuntimeError, match="forward reached"):
        broker._process_parallel({"t0": [_StubSlot(_MODE_DENSE_F32)]})  # pyright: ignore[reportArgumentType]


def test_a_mixed_batch_counts_only_the_unsupported_slot() -> None:
    """Counts SLOTS, not rows.

    The refused ROW count cannot be observed on a mixed batch here: the good
    slot reaches the forward, which raises on the stub model, so
    ``_process_parallel`` never returns. The row-count assertion is in
    ``test_refused_rows_are_not_reported_as_served_throughput``, on an
    all-refused batch that returns before the forward.
    """
    broker = _broker()
    good = _StubSlot(_MODE_DENSE_F32, batch_size=4)
    bad = _StubSlot(_MODE_LEGAL_BF16, batch_size=3)
    with pytest.raises(RuntimeError, match="forward reached"):
        broker._process_parallel({"t0": [good, bad]})  # pyright: ignore[reportArgumentType]
    assert broker.unsupported_mode_requests == 1
    assert bad.state == _STATE_IDLE
    assert good.state != _STATE_IDLE


def test_the_refusal_counter_has_a_reader() -> None:
    """A counter with no reader is the defect this guard exists to fix.

    ``unsupported_mode_requests`` shipped in the first draft of this PR with no
    reader at all — the signature defect inside the fix for the signature
    defect. Its reader is a print in ``serve_forever``, which is a serve loop
    rather than a unit-testable function, so this is a STRUCTURAL gate: it fails
    if the reader is deleted, which is the regression that matters. It cannot
    check the line's formatting, and does not claim to.
    """
    import inspect

    src = inspect.getsource(SharedSlotBroker.serve_forever)
    assert "unsupported_mode_requests" in src, (
        "the B1 refusal counter lost its only reader; a refusal nobody can "
        "count is indistinguishable from one that never fires"
    )
    # And the refused rows must not be left in the served-throughput total.
    assert "_process_parallel" in src
    assert "_total_positions -=" in src, (
        "refused rows are being counted as served pos/s again (B6(c) in the "
        "sibling path)"
    )
