"""Slot-protocol identity and layout guards (inference audit I1, I2).

I1 — a client that timed out and re-submitted into the same named slot was
served the PREVIOUS request's policy and WDL. The broker leaves the slot in
``_STATE_REQUEST`` while it works, so the sequence

    submit R1 -> timeout -> worker resets the client -> submit R2
              -> broker finishes R1 and marks the slot RESPONSE

handed the client waiting on R2 the answer to R1, with nothing raised, nothing
logged and no counter moved. R2 was never evaluated at all. On the compact-legal
transport (the production one) it is worse: the client slices
``policy_u16[:n_legal]`` with its OWN n_legal, so the tail of the returned row
is re-interpreted bytes of its own request metadata.

I2 — client and broker each computed ``_SlotLayout.compute(max_batch, planes)``
and never compared the results. When the client's plane count was the smaller of
the two, every numpy view still fitted inside the larger segment, the protocol
completed normally, and the client read its policy and WDL out of the middle of
the broker's INPUT region: all-zero policy, all-zero WDL, no exception. That is
the zero-fill poisoning ``test_broker_no_zero_fill.py`` exists to prevent,
reached through a channel that test does not cover.

Every test here is CPU-only with a synthetic marker model.
"""

from __future__ import annotations

import struct
import threading
import time
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from chess_anti_engine import inference as inf
from chess_anti_engine.inference import (
    _OFF_MAGIC,
    _SLOT_MAGIC,
    _InferenceSlot,
    _SlotLayout,
    SlotInferenceClient,
    SlotProtocolMismatch,
)

_PLANES = 8
_MAX_BATCH = 4


class _MarkerModel(torch.nn.Module):
    """policy/wdl are a pure function of a scalar marker in plane 0.

    Lets a test name which request produced an answer, which is the whole point:
    the bug returns a well-formed answer to the wrong question.
    """

    def __init__(self) -> None:
        super().__init__()
        self.delay_s = 0.0
        self.seen: list[float] = []

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        marker = x[:, 0, 0, 0].float()
        self.seen.append(float(marker[0]))
        if self.delay_s > 0.0:
            time.sleep(self.delay_s)
        n = x.shape[0]
        return {
            "policy": marker[:, None].expand(n, inf._POLICY_SIZE).contiguous(),
            "wdl": marker[:, None].expand(n, 3).contiguous(),
        }


def _row(marker: float, dtype: type = np.float32) -> np.ndarray:
    x = np.zeros((1, _PLANES, 8, 8), dtype=np.float32)
    x[0, 0, 0, 0] = marker
    if dtype is np.uint16:
        return (
            torch.from_numpy(x).to(torch.bfloat16).view(torch.uint16).numpy().copy()
        )
    return x


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return torch.from_numpy(bits.astype(np.uint16)).view(torch.bfloat16).float().numpy()


class _RunningBroker:
    """A real ``SlotBroker`` on CPU with a marker model, served in a thread."""

    def __init__(self, tmp_path: Path) -> None:
        prefix = f"cae-proto-{uuid.uuid4().hex[:12]}"
        self.broker = inf.SlotBroker(
            publish_dir=tmp_path,
            num_slots=1,
            max_batch_per_slot=_MAX_BATCH,
            device="cpu",
            compile_inference=False,
            batch_wait_ms=0.0,
            slot_prefix=prefix,
            input_planes=_PLANES,
            hang_abort_seconds=0.0,
        )
        self.model = _MarkerModel()
        self.broker._model = self.model
        self.broker._ensure_model = lambda: None  # type: ignore[method-assign]
        self.thread = threading.Thread(target=self.broker.serve_forever, daemon=True)
        self.thread.start()

    @property
    def slot_name(self) -> str:
        return self.broker.slot_names[0]

    def close(self) -> None:
        self.broker._stop = True
        self.thread.join(timeout=2.0)
        self.broker.shutdown()


@pytest.fixture
def running_broker(tmp_path: Path):
    rb = _RunningBroker(tmp_path)
    try:
        yield rb
    finally:
        rb.close()


def _client(slot_name: str, timeout_s: float) -> SlotInferenceClient:
    return SlotInferenceClient(
        slot_name=slot_name,
        max_batch=_MAX_BATCH,
        request_timeout_s=timeout_s,
        input_planes=_PLANES,
    )


# ---------------------------------------------------------------------------
# I1 — request identity
# ---------------------------------------------------------------------------


def test_timed_out_resubmit_is_not_served_the_previous_answer_dense(
    running_broker: _RunningBroker,
) -> None:
    """The dense-f32 half of the I1 race: R2 must get R2's answer, not R1's."""
    rb = running_broker

    warm = _client(rb.slot_name, 5.0)
    _, wdl = warm.evaluate_encoded(_row(1.0))
    assert float(wdl[0, 0]) == pytest.approx(1.0)
    warm.close()

    # A forward that outlives the client's request timeout.
    rb.model.delay_s = 2.0
    c1 = _client(rb.slot_name, 0.4)
    with pytest.raises(TimeoutError):
        c1.evaluate_encoded(_row(11.0))
    c1.close()

    # The worker's recovery (worker.py _reset_inference_client): fresh client,
    # SAME named slot, new request. A per-instance counter starting at zero
    # would hand this client R1's tag and reproduce the bug exactly, which is
    # why request ids are seeded randomly per instance.
    rb.model.delay_s = 0.0
    c2 = _client(rb.slot_name, 5.0)
    _, wdl2 = c2.evaluate_encoded(_row(22.0))

    assert float(wdl2[0, 0]) == pytest.approx(22.0), (
        "client was served a different position's WDL"
    )
    assert 22.0 in rb.model.seen, "position 22.0 was never evaluated at all"
    assert c2.stale_responses_rejected >= 1, (
        "the stale answer must be observed and counted, not silently absent"
    )
    c2.close()


def test_timed_out_resubmit_is_not_served_the_previous_answer_compact(
    running_broker: _RunningBroker,
) -> None:
    """The compact-legal (production) transport half of the I1 race.

    Here the corruption was worse than a wrong policy: the client reads
    ``policy_u16[:n_legal]`` with its own n_legal, so a 3-legal answer read as
    7-legal returned 3 foreign logits followed by 4 entries of re-interpreted
    request metadata.
    """
    rb = running_broker
    flat_a = np.array([0, 5, 4671], dtype=np.int32)
    counts_a = np.array([3], dtype=np.int32)
    flat_b = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    counts_b = np.array([7], dtype=np.int32)

    warm = _client(rb.slot_name, 5.0)
    pol, _ = warm.evaluate_legal_bf16(_row(7.0, np.uint16), flat_b, counts_b)
    assert np.allclose(_bf16_bits_to_f32(pol), 7.0)
    warm.close()

    rb.model.delay_s = 2.0
    c1 = _client(rb.slot_name, 0.4)
    with pytest.raises(TimeoutError):
        c1.evaluate_legal_bf16(_row(11.0, np.uint16), flat_a, counts_a)
    c1.close()

    rb.model.delay_s = 0.0
    c2 = _client(rb.slot_name, 5.0)
    pol2, wdl2 = c2.evaluate_legal_bf16(_row(22.0, np.uint16), flat_b, counts_b)

    logits = _bf16_bits_to_f32(pol2)
    assert logits.shape == (7,)
    assert np.allclose(logits, 22.0), f"stale/torn compact policy served: {logits}"
    assert np.allclose(wdl2[0], 22.0)
    assert c2.stale_responses_rejected >= 1
    c2.close()


def test_negative_control_stale_answer_is_accepted_without_the_id_check(
    running_broker: _RunningBroker,
) -> None:
    """Shuffle-the-labels control: the rig must FAIL when the guard is removed.

    Without this, a green I1 test proves only that the race did not happen on
    this run. Here the client's tag allocator is pinned to a constant, which is
    exactly the pre-fix state (no identity in the header), and the same
    sequence must then serve R1's answer to R2.
    """
    rb = running_broker
    warm = _client(rb.slot_name, 5.0)
    warm.evaluate_encoded(_row(1.0))
    warm.close()

    rb.model.delay_s = 2.0
    c1 = _client(rb.slot_name, 0.4)
    c1._alloc_request_id = lambda: 0xABCDEF  # type: ignore[method-assign]
    with pytest.raises(TimeoutError):
        c1.evaluate_encoded(_row(11.0))
    c1.close()

    rb.model.delay_s = 0.0
    c2 = _client(rb.slot_name, 5.0)
    c2._alloc_request_id = lambda: 0xABCDEF  # type: ignore[method-assign]
    _, wdl2 = c2.evaluate_encoded(_row(22.0))
    c2.close()

    assert float(wdl2[0, 0]) == pytest.approx(11.0), (
        "the control did not reproduce the pre-fix behaviour, so the paired "
        "positive tests prove nothing about the guard"
    )


def test_reset_inference_client_recovery_loop_stays_correct(
    running_broker: _RunningBroker,
) -> None:
    """End-to-end worker.py:_reset_inference_client shape, repeated.

    close() + rebuild on the same slot name, several times, each with a distinct
    marker. Every answer must be its own request's.
    """
    rb = running_broker
    for marker in (3.0, 4.0, 5.0, 6.0):
        client = _client(rb.slot_name, 5.0)
        try:
            _, wdl = client.evaluate_encoded(_row(marker))
            assert float(wdl[0, 0]) == pytest.approx(marker)
        finally:
            client.close()


def test_broker_echoes_the_tag_it_gathered_not_the_one_in_the_slot(
    running_broker: _RunningBroker,
) -> None:
    """Protocol-level: the echo comes from the gather snapshot.

    Overwriting the slot's tag while the broker is mid-forward (which is what a
    re-submitting client does) must not make the in-flight answer look fresh.
    """
    rb = running_broker
    rb.model.delay_s = 0.5
    client = _client(rb.slot_name, 5.0)
    slot = client._connect(deadline=time.monotonic() + 5.0)

    submitted = client._alloc_request_id()
    slot.input[:1] = _row(31.0)
    slot.request_mode = inf._MODE_DENSE_F32
    slot.batch_size = 1
    slot.request_id = submitted
    slot.state = inf._STATE_REQUEST

    # Let the broker gather, then clobber the tag the way a re-submit would.
    time.sleep(0.15)
    slot.request_id = (submitted + 999) & 0xFFFFFFFF

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and int(slot.state) != inf._STATE_RESPONSE:
        time.sleep(0.005)
    assert int(slot.state) == inf._STATE_RESPONSE
    assert int(slot.request_id) == submitted, (
        "broker echoed the client's post-gather tag; a re-submitted request "
        "would accept this answer"
    )
    client.close()


def test_the_tag_is_read_before_every_other_slot_field(tmp_path: Path) -> None:
    """The guarantee is "tag first", and `request_mode` is read in _process_batch.

    Reading mode first and the tag later leaves a window in which a
    re-submitting client writes mode -> batch_size -> tag between the two
    reads: the broker then decodes the NEW payload in the OLD mode and echoes
    the NEW tag, and the client accepts a mis-decoded answer as its own. This
    records the read order rather than the outcome, because the outcome is a
    microsecond race that a test cannot reliably provoke.
    """
    prefix = f"cae-order-{uuid.uuid4().hex[:12]}"
    broker = inf.SlotBroker(
        publish_dir=tmp_path,
        num_slots=1,
        max_batch_per_slot=2,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=prefix,
        input_planes=_PLANES,
        hang_abort_seconds=0.0,
    )
    try:
        slot = broker._slots[0]
        reads: list[str] = []

        class _Spy:
            def __init__(self, inner: _InferenceSlot) -> None:
                object.__setattr__(self, "_inner", inner)

            def __getattr__(self, name: str):
                if name in {"request_id", "request_mode", "batch_size"}:
                    reads.append(name)
                return getattr(object.__getattribute__(self, "_inner"), name)

        broker._process_batch_mode = (  # type: ignore[method-assign]
            lambda ready, *, mode, request_ids: None
        )
        # The spy deliberately duck-types _InferenceSlot rather than subclassing
        # it: __slots__ + numpy views over shm make a real subclass awkward, and
        # the point is only to observe the ORDER of header attribute reads.
        spy = cast("_InferenceSlot", cast("object", _Spy(slot)))
        broker._process_batch([spy])

        assert reads, "the spy observed no header reads at all"
        assert reads[0] == "request_id", (
            f"broker read {reads[0]!r} before the request tag; read order was {reads}"
        )
    finally:
        broker.shutdown()


def test_request_ids_are_unique_across_client_instances() -> None:
    """Two clients on the same slot must not hand out the same tag first."""
    name = f"cae-proto-{uuid.uuid4().hex[:12]}"
    a = SlotInferenceClient(slot_name=name, max_batch=_MAX_BATCH, input_planes=_PLANES)
    b = SlotInferenceClient(slot_name=name, max_batch=_MAX_BATCH, input_planes=_PLANES)
    assert a._alloc_request_id() != b._alloc_request_id()


def test_request_id_never_uses_the_reserved_zero() -> None:
    client = SlotInferenceClient(
        slot_name="unused", max_batch=_MAX_BATCH, input_planes=_PLANES,
    )
    client._next_request_id = inf._REQUEST_ID_MASK  # next wrap lands on 0
    assert client._alloc_request_id() == inf._REQUEST_ID_MASK
    assert client._alloc_request_id() != inf._REQUEST_ID_NONE


# ---------------------------------------------------------------------------
# I2 — layout identity
# ---------------------------------------------------------------------------


def test_layout_identity_separates_plane_counts_and_batch_sizes() -> None:
    base = _SlotLayout.compute(8, 175)
    assert base.identity() != _SlotLayout.compute(8, 146).identity()
    assert base.identity() != _SlotLayout.compute(16, 175).identity()
    assert base.identity() == _SlotLayout.compute(8, 175).identity()


def test_connect_refuses_a_plane_count_skew_instead_of_returning_zeros(
    tmp_path: Path,
) -> None:
    """The I2 repro: 175-plane broker, 146-plane client (the argparse default).

    Pre-fix this returned an all-zero policy and an all-zero WDL in ~1ms with no
    exception, because every client view still fitted inside the larger segment.
    """
    prefix = f"cae-skew-{uuid.uuid4().hex[:12]}"
    broker = inf.SlotBroker(
        publish_dir=tmp_path,
        num_slots=1,
        max_batch_per_slot=2,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=prefix,
        input_planes=175,
        hang_abort_seconds=0.0,
    )
    try:
        matched = SlotInferenceClient(
            slot_name=broker.slot_names[0], max_batch=2,
            request_timeout_s=1.0, input_planes=175,
        )
        matched._connect(deadline=time.monotonic() + 1.0)
        matched.close()

        skewed = SlotInferenceClient(
            slot_name=broker.slot_names[0], max_batch=2,
            request_timeout_s=1.0, input_planes=146,
        )
        with pytest.raises(SlotProtocolMismatch, match="was created for layout"):
            skewed._connect(deadline=time.monotonic() + 1.0)
        skewed.close()
    finally:
        broker.shutdown()


def test_connect_refuses_an_undersized_segment() -> None:
    """A segment smaller than the client's layout cannot be read at all."""
    name = f"cae-small-{uuid.uuid4().hex[:12]}"
    small = _SlotLayout.compute(1, _PLANES)
    shm = SharedMemory(name=name, create=True, size=small.total_bytes)
    try:
        _InferenceSlot(shm, small, owns=False).write_protocol_header()
        client = _client(name, 0.05)  # max_batch=4 -> needs a bigger segment
        with pytest.raises(SlotProtocolMismatch, match="bytes"):
            client._connect(deadline=time.monotonic() + 0.05)
        client.close()
    finally:
        shm.close()
        shm.unlink()


def test_connect_refuses_a_segment_with_no_protocol_magic() -> None:
    """A leftover pre-v2 segment (or any foreign shm) must not be accepted."""
    name = f"cae-nomagic-{uuid.uuid4().hex[:12]}"
    layout = _SlotLayout.compute(_MAX_BATCH, _PLANES)
    shm = SharedMemory(name=name, create=True, size=layout.total_bytes)
    try:
        client = _client(name, 0.05)
        with pytest.raises(SlotProtocolMismatch, match="magic"):
            client._connect(deadline=time.monotonic() + 0.05)
        client.close()
    finally:
        shm.close()
        shm.unlink()


def test_an_unstamped_segment_is_waited_for_not_rejected() -> None:
    """The guard must not turn the broker's own slot-creation window into a crash.

    A creating process does shm_open, then ftruncate, then stamps the header; a
    client attaching inside that window sees a zero-length or unstamped segment.
    That is "not ready yet", the same class as "does not exist yet" -- not a
    mismatch. Rejecting it outright would make the I2 guard a boot-time crash on
    a perfectly matched fleet.
    """
    name = f"cae-settling-{uuid.uuid4().hex[:12]}"
    layout = _SlotLayout.compute(_MAX_BATCH, _PLANES)
    shm = SharedMemory(name=name, create=True, size=layout.total_bytes)
    stamped = threading.Event()

    def _finish_creation() -> None:
        time.sleep(0.15)  # the ftruncate -> stamp gap, widened to be observable
        _InferenceSlot(shm, layout, owns=False).write_protocol_header()
        stamped.set()

    t = threading.Thread(target=_finish_creation, daemon=True)
    t.start()
    try:
        client = _client(name, 5.0)
        slot = client._connect(deadline=time.monotonic() + 5.0)
        assert stamped.is_set(), "connect returned before the header was stamped"
        assert slot is not None
        client.close()
    finally:
        t.join(timeout=2.0)
        shm.close()
        shm.unlink()


def test_layout_mismatch_is_fatal_immediately_not_waited_out() -> None:
    """A real mismatch must not be mistaken for a slot that is still settling."""
    name = f"cae-fatal-{uuid.uuid4().hex[:12]}"
    other = _SlotLayout.compute(_MAX_BATCH, _PLANES + 1)
    shm = SharedMemory(name=name, create=True, size=other.total_bytes)
    try:
        _InferenceSlot(shm, other, owns=False).write_protocol_header()
        client = _client(name, 30.0)
        t0 = time.monotonic()
        with pytest.raises(SlotProtocolMismatch, match="was created for layout"):
            client._connect(deadline=time.monotonic() + 30.0)
        assert time.monotonic() - t0 < 1.0, (
            "a layout mismatch must fail at once, not after the request timeout"
        )
        client.close()
    finally:
        shm.close()
        shm.unlink()


def test_broker_stamps_magic_and_layout_id_into_every_slot(tmp_path: Path) -> None:
    prefix = f"cae-stamp-{uuid.uuid4().hex[:12]}"
    broker = inf.SlotBroker(
        publish_dir=tmp_path,
        num_slots=2,
        max_batch_per_slot=2,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=prefix,
        input_planes=_PLANES,
        hang_abort_seconds=0.0,
    )
    try:
        want = _SlotLayout.compute(2, _PLANES).identity()
        for name in broker.slot_names:
            shm = SharedMemory(name=name, create=False)
            try:
                buf = shm.buf
                assert buf is not None
                assert struct.unpack_from("<I", buf, _OFF_MAGIC)[0] == _SLOT_MAGIC
                assert struct.unpack_from("<I", buf, inf._OFF_LAYOUT_ID)[0] == want
            finally:
                shm.close()
    finally:
        broker.shutdown()


# ---------------------------------------------------------------------------
# Backward-safety of the wire-format bump (PR #322 review, blocking finding)
# ---------------------------------------------------------------------------


def test_slot_name_carries_the_protocol_version() -> None:
    """The version must be in the NAME, not only in the header.

    An old client runs OLD code and has no validation to fail. The v2 header
    grew at the FRONT while state@0 / mode@1 / batch_size@4 kept their offsets,
    so an origin/main client on a v2 segment completes the handshake 16 bytes
    out of phase and returns an all-zero policy and WDL with no exception --
    measured, across two worktrees. A version bump cannot make an old peer
    refuse; it can only make the name it looks for not exist.
    """
    from chess_anti_engine.tune.distributed_runtime import _trial_slot_prefix

    prefix = _trial_slot_prefix(trial_id="abc123_00000")
    assert prefix.startswith(f"cae{inf.SLOT_PROTOCOL_VERSION}-"), prefix
    # The pre-bump name for the same trial must NOT be produced, or a stale
    # worker would still find a segment to map.
    from chess_anti_engine.tune._utils import stable_seed_u32

    legacy = f"cae-{stable_seed_u32('slot-prefix', 'abc123_00000'):08x}"
    assert prefix != legacy


def test_both_slot_prefix_derivations_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launcher tells workers one name and the shared broker creates another.

    If they drift, every worker times out against a slot nobody created --
    silent from the broker's side, since it just never sees a request.

    The first version of this test re-derived the formula by hand, so it was a
    THIRD copy: sabotaging either implementation left it green (PR #322
    reviewer). Both sites now delegate to ``inference.trial_slot_prefix`` and
    this test calls the REAL ones -- including the broker's
    ``_register_new_trial``, whose name is otherwise only observable through
    the shared-memory segments it creates.
    """
    from chess_anti_engine.tune.distributed_runtime import _trial_slot_prefix

    created: list[str] = []

    class _FakeShm:
        def __init__(self, *, name: str, create: bool = False, size: int = 0) -> None:
            if not create:
                raise FileNotFoundError(name)
            created.append(name)
            self.name = name
            self.buf = memoryview(bytearray(max(size, 1)))

        def close(self) -> None:
            pass

        def unlink(self) -> None:
            pass

    class _FakeSlot:
        def __init__(self, shm: object, layout: object, owns: bool = False) -> None:
            assert shm is not None
            assert layout is not None
            assert owns
            self.state = 0
            self.batch_size = 0

        def write_protocol_header(self) -> None:
            pass

    monkeypatch.setattr(inf, "SharedMemory", _FakeShm)
    monkeypatch.setattr(inf, "_InferenceSlot", _FakeSlot)

    for trial_id in ("13a9f_00000", "9c36d_00000", "x"):
        created.clear()
        broker = object.__new__(inf.SharedSlotBroker)
        broker.slots_per_trial = 2
        broker._layout = inf._SlotLayout.compute(8, 175)
        broker._trial_slots = {}
        broker._all_slots = []
        inf.SharedSlotBroker._register_new_trial(broker, trial_id)

        assert len(created) == 2, created
        # The broker's realized segment names, not a re-derivation of them.
        broker_prefix = created[0].rsplit("-", 1)[0]
        assert created == [f"{broker_prefix}-0", f"{broker_prefix}-1"]
        assert _trial_slot_prefix(trial_id=trial_id) == broker_prefix


def test_slot_prefix_agreement_test_is_sabotage_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating the shared helper must turn the agreement test RED.

    The failure this guards against is precise: a test that re-derives the
    formula agrees with itself no matter what either caller does. Patch the one
    helper to a wrong value and require the launcher side to follow it -- if it
    does not, the launcher is still carrying its own copy.
    """
    from chess_anti_engine.tune import distributed_runtime as dr

    monkeypatch.setattr(inf, "trial_slot_prefix", lambda *, trial_id: "SABOTAGED")
    monkeypatch.setattr(dr, "trial_slot_prefix", inf.trial_slot_prefix)
    assert dr._trial_slot_prefix(trial_id="13a9f_00000") == "SABOTAGED", (
        "distributed_runtime._trial_slot_prefix does not delegate to the shared "
        "helper -- it is re-deriving the name, which is how the two sides drift"
    )
