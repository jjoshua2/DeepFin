"""Malformed compact-legal metadata must not be answered with a stale policy.

The third path in ``inference.py`` that fabricated a response instead of
refusing one, and the worst of the three. When the compact-legal metadata
snapshot failed validation the broker did::

    slot.policy_u16[:1] = 0
    slot.wdl[:bsz].fill(0.0)
    slot.request_mode = _MODE_DENSE_F32
    slot.state = _STATE_RESPONSE

which the client cannot tell apart from a real answer: it never reads
``request_mode`` back, and the success path sets ``_MODE_DENSE_F32`` too.

The damage is worse than the all-zero policy that ``_release_slots_for_retry``
was written to prevent. ``SlotInferenceClient.evaluate_legal_bf16`` reads
``slot.policy_u16[:n_legal]`` using its *own* ``n_legal``, so zeroing entry 0
left entries ``1..n_legal`` holding whatever bf16 logits the previous request
left in that shared buffer — a plausible-looking policy for a different
position, paired with an all-zero WDL, fed into MCTS and recorded as training
data. Nothing counted a failure and nothing logged.

The dominant trigger is benign and self-healing: the client rewrites the slot
while the broker is mid-read (see the SNAPSHOT comment in
``_process_batch_mode``), so releasing the slot makes the client re-submit and
the re-read succeeds. These tests pin the refusal, not the validation rules.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.inference import (
    _MODE_LEGAL_BF16,
    _POLICY_SIZE,
    _STATE_IDLE,
    _STATE_REQUEST,
    _STATE_RESPONSE,
    _InferenceSlot,
    SlotBroker,
)

_BSZ = 2
_SENTINEL_U16 = 0xBEEF
_SENTINEL_F32 = 1234.5


def _make_broker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SlotBroker:
    """A broker that believes it has a model, so validation is what we exercise.

    The model is a bare ``torch.nn.Module`` with no ``forward``, and is never
    called: the malformed slot is skipped before the gather, leaving
    ``total == 0``, so ``_process_batch_mode`` returns at its own empty-batch
    guard. Bare rather than a working stub on purpose -- if the rejection ever
    stops short-circuiting, the forward raises ``NotImplementedError`` instead
    of quietly returning plausible numbers and letting these tests pass for
    the wrong reason.
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
        slot_prefix=f"cae-badmeta-{uuid.uuid4().hex}",
    )

    def _fake_ensure_model(self: SlotBroker) -> None:
        self._model = torch.nn.Module()

    monkeypatch.setattr(SlotBroker, "_ensure_model", _fake_ensure_model)
    return broker


def _arm_malformed_slot(
    broker: SlotBroker, *, bad: str,
) -> tuple[_InferenceSlot, np.ndarray, np.ndarray]:
    """Arm slot 0 with legal metadata that fails validation in one specific way.

    Returns the slot plus pre-call copies of both response buffers, so a test
    can assert the broker wrote *nothing* rather than merely that it wrote
    something other than a valid answer.
    """
    slot = broker._slots[0]
  # policy_i32 and policy_u16 alias the same bytes, so poison first and write
  # the metadata second -- the reverse order would erase the metadata.
    slot.policy_u16.fill(_SENTINEL_U16)
    slot.wdl.fill(_SENTINEL_F32)

    counts = np.array([3, 4], dtype=np.int32)
    n_legal = int(counts.sum())
    flat = np.arange(n_legal, dtype=np.int32)
    if bad == "count_sum":
      # Header disagrees with the per-row counts: the exact shape of a torn
      # read, where the header came from one request and the counts another.
        n_legal_header = n_legal - 2
    elif bad == "out_of_range":
        n_legal_header = n_legal
        flat[-1] = _POLICY_SIZE
    else:  # pragma: no cover - guards a typo in a caller
        raise AssertionError(f"unknown malformation {bad!r}")

    slot.policy_i32[0] = n_legal_header
    slot.policy_i32[1:1 + _BSZ] = counts
    slot.policy_i32[1 + _BSZ:1 + _BSZ + n_legal] = flat
    slot.batch_size = _BSZ
    slot.request_mode = _MODE_LEGAL_BF16
    slot.state = _STATE_REQUEST
    return slot, slot.policy_u16.copy(), slot.wdl.copy()


@pytest.mark.parametrize("bad", ["count_sum", "out_of_range"])
def test_malformed_metadata_releases_the_slot_instead_of_answering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str,
) -> None:
    broker = _make_broker(tmp_path, monkeypatch)
    try:
        slot, _, _ = _arm_malformed_slot(broker, bad=bad)

        broker._process_batch([slot])

        assert slot.state == _STATE_IDLE, (
            "malformed metadata must hand the slot back for retry"
        )
        assert slot.state != _STATE_RESPONSE
    finally:
        broker.shutdown()


@pytest.mark.parametrize("bad", ["count_sum", "out_of_range"])
def test_malformed_metadata_writes_nothing_into_the_response_buffers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str,
) -> None:
    """The old path zeroed entry 0 and the WDL, leaving 1..n_legal stale."""
    broker = _make_broker(tmp_path, monkeypatch)
    try:
        slot, policy_before, wdl_before = _arm_malformed_slot(broker, bad=bad)

        broker._process_batch([slot])

        assert np.array_equal(slot.policy_u16, policy_before), (
            "broker must not write into the policy buffer it is refusing to fill"
        )
        assert np.array_equal(slot.wdl, wdl_before), (
            "the all-zero WDL was the fabricated value that reached MCTS"
        )
    finally:
        broker.shutdown()


def test_malformed_metadata_is_counted_but_does_not_kill_the_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counted for visibility; deliberately outside the exit ceiling.

    The consecutive-failure ceiling exits the broker, which removes inference
    for the whole fleet. A broker restart cannot fix a client-side protocol
    bug, so escalating there would trade one broken client for a fleet outage
    and then repeat. The client's own TimeoutError is the correctly scoped
    consequence.
    """
    broker = _make_broker(tmp_path, monkeypatch)
    try:
        for _ in range(5):
            slot, _, _ = _arm_malformed_slot(broker, bad="count_sum")
            broker._process_batch([slot])

        assert broker._malformed_legal_meta_total == 5
        assert broker._consecutive_batch_failures == 0, (
            "a malformed client request is not a broker failure"
        )
        assert not broker._stop
    finally:
        broker.shutdown()


def test_malformed_metadata_warning_is_throttled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A retrying client hits this every ~10ms; one line per 30s, not per hit."""
    broker = _make_broker(tmp_path, monkeypatch)
    try:
        with caplog.at_level(logging.WARNING, logger="chess_anti_engine.inference"):
            for _ in range(8):
                slot, _, _ = _arm_malformed_slot(broker, bad="count_sum")
                broker._process_batch([slot])

        warnings = [
            r for r in caplog.records if "malformed compact-legal metadata" in r.getMessage()
        ]
        assert len(warnings) == 1, f"expected one throttled warning, got {len(warnings)}"
      # The running total is in the line precisely so a throttled burst is
      # still legible from a single message.
        assert broker._malformed_legal_meta_total == 8
    finally:
        broker.shutdown()


def test_rejection_log_names_the_invariant_that_broke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A torn snapshot and a client encoding bug need different responses.

    The first is benign and self-heals on the client's re-submit; the second is
    a real bug in whoever built the request. A log line that says only
    "malformed" cannot tell them apart, which is the first thing worth knowing
    when the counter finally goes non-zero.
    """
    broker = _make_broker(tmp_path, monkeypatch)
    try:
        seen: dict[str, str] = {}
        for bad in ("count_sum", "out_of_range"):
          # Clear the 30s throttle so the second rejection also logs.
            broker._malformed_legal_meta_warned_at = 0.0
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="chess_anti_engine.inference"):
                slot, _, _ = _arm_malformed_slot(broker, bad=bad)
                broker._process_batch([slot])
            seen[bad] = "\n".join(
                r.getMessage() for r in caplog.records
                if "malformed compact-legal metadata" in r.getMessage()
            )

        assert "counts sum 7 != header n_legal 5" in seen["count_sum"]
        assert f"outside [0, {_POLICY_SIZE})" in seen["out_of_range"]
        assert seen["count_sum"] != seen["out_of_range"]
    finally:
        broker.shutdown()


def test_a_malformed_slot_does_not_corrupt_the_healthy_slots_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping a slot must not desync the compact gather bookkeeping.

    ``batch_sizes``, ``legal_counts_by_slot`` and ``legal_flat_by_slot`` are
    zipped ``strict=True`` and ``compact_offsets`` indexes the scatter, so a
    slot that is dropped after any of them is appended would shift every
    later slot's policy window -- silently handing slot B the logits computed
    for slot A. The rejection therefore has to happen before all four, which
    is what this pins.

    The model encodes BOTH coordinates that a desync could shift --
    ``input_sum * 1000 + row * 10000 + index`` -- because they fail
    independently. A model that ignores ``x`` (as the one in
    test_inference_broker.py does) cannot see an input-staging desync at all:
    the gather writes the healthy slot's planes at the wrong pinned offset,
    the forward reads the rejected slot's planes instead, and the answer is
    byte-identical because it never depended on the input. Verified by
    sabotage -- with an ``x``-independent model, bumping ``total`` for the
    rejected slot still passes.

    Each slot therefore gets a distinct one-hot input, so the healthy slot's
    answer names the row it occupied AND the planes that row held.
    """

    class _TinyPolicy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            bsz = x.shape[0]
            base = x.to(torch.float32).flatten(1).sum(1).unsqueeze(1) * 1000.0
            row = torch.arange(bsz, dtype=torch.float32).unsqueeze(1) * 10000.0
            return {
                "policy": base + row + torch.arange(4672, dtype=torch.float32).unsqueeze(0),
                "wdl": torch.arange(bsz * 3, dtype=torch.float32).reshape(bsz, 3),
            }

    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    broker = SlotBroker(
        publish_dir=publish_dir,
        num_slots=2,
        max_batch_per_slot=8,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-badmeta-mixed-{uuid.uuid4().hex}",
    )
    monkeypatch.setattr(SlotBroker, "_ensure_model", lambda _self: None)
    broker._model = _TinyPolicy().eval()
    broker._model_sha = "test"
    try:
        bad, _, _ = _arm_malformed_slot(broker, bad="count_sum")
      # bf16 bit patterns: 0x4000 == 2.0, 0x3F80 == 1.0. Distinct per slot so
      # a mis-staged input row changes the answer instead of hiding in zeros.
        bad.input_bf16_bits.fill(0)
        bad.input_bf16_bits[0, 0, 0, 0] = 0x4000

        good = broker._slots[1]
        good.input_bf16_bits.fill(0)
        good.input_bf16_bits[0, 0, 0, 0] = 0x3F80
        good.policy_i32[0] = 2
        good.policy_i32[1:2] = np.array([2], dtype=np.int32)
        good.policy_i32[2:4] = np.array([3, 7], dtype=np.int32)
        good.batch_size = 1
        good.request_mode = _MODE_LEGAL_BF16
        good.state = _STATE_REQUEST

        broker._process_batch([bad, good])

        assert bad.state == _STATE_IDLE
        assert good.state == _STATE_RESPONSE, (
            "one client's bad request must not cost every other client its answer"
        )
      # input_sum 1.0 * 1000 + row 0 * 10000 + index. A row desync reads
      # 11003/11007; an input desync reads 2003/2007.
        expected = (
            torch.tensor([1003.0, 1007.0]).to(torch.bfloat16).view(torch.uint16).numpy()
        )
        assert np.array_equal(good.policy_u16[:2], expected), (
            "healthy slot read the wrong batch row -- gather bookkeeping desynced"
        )
        assert broker._malformed_legal_meta_total == 1
    finally:
        broker.shutdown()


def test_metrics_line_reports_malformed_total_only_when_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Healthy runs keep the old line; any non-zero value is a grep hit.

    The metrics line only prints when a batch completed, and a rejected slot
    never reaches a batch -- which is why the counter is a lifetime total
    rather than a per-interval one.
    """
    broker = _make_broker(tmp_path, monkeypatch)
    try:
        metrics = {
            "batches": 4, "positions": 128, "slots": 8, "pack_s": 0.01,
            "forward_s": 0.04, "output_s": 0.01, "scatter_s": 0.01,
            "last_report": 0.0,
        }
        broker._maybe_print_broker_metrics(dict(metrics), now=10.0, report_interval=5.0)
        assert "malformed_legal_meta" not in capsys.readouterr().out

        broker._malformed_legal_meta_total = 3
        broker._maybe_print_broker_metrics(dict(metrics), now=10.0, report_interval=5.0)
        assert "malformed_legal_meta=3" in capsys.readouterr().out
    finally:
        broker.shutdown()
