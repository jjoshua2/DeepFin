"""Bounded attachment recovery before a POSIX shared-memory slot is sized."""
from __future__ import annotations

import _posixshmem  # POSIX-only module, like the slot transport under test
import itertools
import os
import threading
import time
import uuid
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import pytest

from chess_anti_engine import inference as inf


def _client(name: str) -> inf.SlotInferenceClient:
    return inf.SlotInferenceClient(
        slot_name=name, max_batch=2, input_planes=8, request_timeout_s=1.0,
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX shm_open/ftruncate publication gap")
def test_real_zero_length_creation_gap_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pause real creation between shm_open and ftruncate, not a fake mmap error."""
    name = f"cae-connect-{uuid.uuid4().hex}"
    layout = inf._SlotLayout.compute(2, 8)
    opened = threading.Event()
    finish_sizing = threading.Event()
    owners: list[SharedMemory] = []
    failures: list[BaseException] = []
    attachment_errors: list[str] = []
    original_ftruncate = os.ftruncate

    def delay_sizing(fd: int, size: int) -> None:
        if threading.current_thread() is creator:
            opened.set()
            if not finish_sizing.wait(timeout=3.0):
                raise TimeoutError("test did not release its own creator")
        original_ftruncate(fd, size)

    def create_slot() -> None:
        try:
            shm = SharedMemory(name=name, create=True, size=layout.total_bytes)
            owners.append(shm)
            inf._InferenceSlot(shm, layout, owns=False).write_protocol_header()
        except BaseException as exc:
            failures.append(exc)

    def attach(**kwargs: Any) -> SharedMemory:
        try:
            return SharedMemory(**kwargs)
        except ValueError as exc:
            attachment_errors.append(str(exc))
            finish_sizing.set()
            raise

    creator = threading.Thread(target=create_slot, name="test-slot-creator", daemon=True)
    monkeypatch.setattr(os, "ftruncate", delay_sizing)
    monkeypatch.setattr(inf, "SharedMemory", attach)
    # Both creator and client live in this test process, sharing one tracker.
    monkeypatch.setattr(inf, "_detach_attached_shm_from_resource_tracker", lambda _shm: None)
    client = _client(name)
    creator.start()
    try:
        assert opened.wait(timeout=3.0), "creator did not reach the sizing gap"
        slot = client._connect(deadline=time.monotonic() + 2.0)
        creator.join(timeout=3.0)
        assert not creator.is_alive()
        assert not failures
        assert attachment_errors
        assert set(attachment_errors) == {"cannot mmap an empty file"}
        assert slot._layout.identity() == layout.identity()
        assert client._shm is not None
        assert client._shm.size == layout.total_bytes
    finally:
        finish_sizing.set()
        creator.join(timeout=3.0)
        client.close()
        for shm in owners:
            shm.close()
            shm.unlink()


@pytest.mark.parametrize("empty_attempts", [1, 3])
def test_repeated_empty_maps_retry_until_ready(
    monkeypatch: pytest.MonkeyPatch, empty_attempts: int,
) -> None:
    layout = inf._SlotLayout.compute(2, 8)
    shm = SharedMemory(create=True, size=layout.total_bytes)
    inf._InferenceSlot(shm, layout, owns=False).write_protocol_header()
    attempts = 0

    def attach(**kwargs: Any) -> SharedMemory:
        nonlocal attempts
        attempts += 1
        if attempts <= empty_attempts:
            raise ValueError("cannot mmap an empty file")
        return SharedMemory(**kwargs)

    monkeypatch.setattr(inf, "SharedMemory", attach)
    monkeypatch.setattr(inf, "_detach_attached_shm_from_resource_tracker", lambda _shm: None)
    client = _client(shm.name)
    try:
        slot = client._connect(deadline=time.monotonic() + 2.0)
        assert slot is client._slot
        assert attempts == empty_attempts + 1
    finally:
        client.close()
        shm.close()
        shm.unlink()


_ATTEMPT_CAP = 50  # a deadline-ignoring loop trips this instead of hanging


def _zero_length_segment(name: str) -> None:
    """A segment published by shm_open but never sized: a crashed creator's orphan."""
    fd = _posixshmem.shm_open(
        "/" + name, os.O_CREAT | os.O_EXCL | os.O_RDWR, mode=0o600,
    )
    os.close(fd)


def _unlink_quietly(name: str) -> None:
    try:
        _posixshmem.shm_unlink("/" + name)
    except FileNotFoundError:
        pass


def _advancing_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = itertools.count(9.0, 0.5)
    monkeypatch.setattr(inf.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(inf.time, "sleep", lambda _s: None)


@pytest.mark.skipif(os.name != "posix", reason="POSIX shm_open/ftruncate publication gap")
def test_permanently_empty_map_obeys_existing_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real zero-length segment past the deadline names the stuck creator."""
    name = f"cae-test-empty-{uuid.uuid4().hex}"
    attempts = 0
    real_shared_memory = inf.SharedMemory

    def attach(**kwargs: Any) -> SharedMemory:
        nonlocal attempts
        attempts += 1
        if attempts > _ATTEMPT_CAP:
            raise AssertionError("connect kept retrying past its deadline")
        return real_shared_memory(**kwargs)

    _zero_length_segment(name)
    try:
        monkeypatch.setattr(inf, "SharedMemory", attach)
        _advancing_clock(monkeypatch)
        client = _client(name)
        with pytest.raises(TimeoutError, match="still zero-length") as raised:
            client._connect(deadline=10.0)
        assert "never appeared" not in str(raised.value)
        assert isinstance(raised.value.__cause__, ValueError)
        # Clock reads 9.0, 9.5, 10.0: two retries, then the deadline fires.
        assert attempts == 3
        assert client._slot is None
        assert client._shm is None
    finally:
        _unlink_quietly(name)


def test_missing_slot_timeout_says_it_never_appeared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = f"cae-test-missing-{uuid.uuid4().hex}"
    attempts = 0
    real_shared_memory = inf.SharedMemory

    def attach(**kwargs: Any) -> SharedMemory:
        nonlocal attempts
        attempts += 1
        if attempts > _ATTEMPT_CAP:
            raise AssertionError("connect kept retrying past its deadline")
        return real_shared_memory(**kwargs)

    monkeypatch.setattr(inf, "SharedMemory", attach)
    _advancing_clock(monkeypatch)
    client = _client(name)
    with pytest.raises(TimeoutError, match="never appeared") as raised:
        client._connect(deadline=10.0)
    assert "zero-length" not in str(raised.value)
    assert isinstance(raised.value.__cause__, FileNotFoundError)
    assert attempts == 3


@pytest.mark.skipif(os.name != "posix", reason="POSIX shm_open/ftruncate publication gap")
def test_empty_segment_is_retried_whatever_the_error_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settling is judged by the segment's size, not CPython's message text."""
    name = f"cae-test-wording-{uuid.uuid4().hex}"
    attempts = 0
    real_shared_memory = inf.SharedMemory

    def attach(**kwargs: Any) -> SharedMemory:
        nonlocal attempts
        attempts += 1
        if attempts > _ATTEMPT_CAP:
            raise AssertionError("connect kept retrying past its deadline")
        try:
            return real_shared_memory(**kwargs)
        except ValueError as exc:
            raise ValueError("some future CPython wording") from exc

    _zero_length_segment(name)
    try:
        monkeypatch.setattr(inf, "SharedMemory", attach)
        _advancing_clock(monkeypatch)
        client = _client(name)
        with pytest.raises(TimeoutError, match="still zero-length"):
            client._connect(deadline=10.0)
        assert attempts == 3
    finally:
        _unlink_quietly(name)


def test_value_error_on_a_sized_segment_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = inf._SlotLayout.compute(2, 8)
    shm = SharedMemory(create=True, size=layout.total_bytes)
    error = ValueError("mmap length is greater than file size")
    attempts = 0

    def attach(**_kwargs: Any) -> SharedMemory:
        nonlocal attempts
        attempts += 1
        raise error

    monkeypatch.setattr(inf, "SharedMemory", attach)
    client = _client(shm.name)
    try:
        with pytest.raises(ValueError, match="mmap length is greater") as raised:
            client._connect(deadline=time.monotonic() + 10.0)
        assert raised.value is error
        assert attempts == 1
    finally:
        shm.close()
        shm.unlink()


@pytest.mark.skipif(os.name != "posix", reason="POSIX shm_open/ftruncate publication gap")
def test_stale_cleanup_unlinks_a_zero_length_orphan() -> None:
    name = f"cae-test-orphan-{uuid.uuid4().hex}"
    _zero_length_segment(name)
    try:
        assert inf._posix_shm_segment_size(name) == 0
        inf._unlink_stale_slot_segment(name)
        assert inf._posix_shm_segment_size(name) is None
    finally:
        _unlink_quietly(name)


@pytest.mark.skipif(os.name != "posix", reason="POSIX shm_open/ftruncate publication gap")
def test_shared_broker_registers_over_a_zero_length_orphan() -> None:
    """The broker's own slot-creation path survives its predecessor's crash."""
    trial_id = f"test-orphan-{uuid.uuid4().hex}"
    prefix = inf.trial_slot_prefix(trial_id=trial_id)
    names = [f"{prefix}-{i}" for i in range(2)]
    broker = object.__new__(inf.SharedSlotBroker)
    broker.slots_per_trial = 2
    broker._layout = inf._SlotLayout.compute(2, 8)
    broker._trial_slots = {}
    broker._all_slots = []
    _zero_length_segment(names[0])
    try:
        broker._register_new_trial(trial_id)
        slots = broker._trial_slots[trial_id]
        assert len(slots) == 2
        for name in names:
            assert inf._posix_shm_segment_size(name) == broker._layout.total_bytes
    finally:
        for slot in broker._trial_slots.get(trial_id, []):
            slot.close()  # owns=True: closes and unlinks
        for name in names:
            _unlink_quietly(name)


def test_stale_cleanup_propagates_value_error_for_a_sized_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = inf._SlotLayout.compute(2, 8)
    shm = SharedMemory(create=True, size=layout.total_bytes)
    error = ValueError("unrelated attach failure")

    def attach(**_kwargs: Any) -> SharedMemory:
        raise error

    monkeypatch.setattr(inf, "SharedMemory", attach)
    try:
        with pytest.raises(ValueError, match="unrelated attach failure") as raised:
            inf._unlink_stale_slot_segment(shm.name)
        assert raised.value is error
        assert inf._posix_shm_segment_size(shm.name) == layout.total_bytes
    finally:
        shm.close()
        shm.unlink()


@pytest.mark.parametrize("error", [
    ValueError("invalid shared-memory name"),
    ValueError("mmap length is greater than file size"),
    PermissionError("access denied"),
    OSError("resource exhaustion"),
])
def test_other_attachment_failures_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    attempts = 0

    def attach(**_kwargs: Any) -> SharedMemory:
        nonlocal attempts
        attempts += 1
        raise error

    monkeypatch.setattr(inf, "SharedMemory", attach)
    client = _client("synthetic-error")
    with pytest.raises(type(error)) as raised:
        client._connect(deadline=time.monotonic() + 10.0)
    assert raised.value is error
    assert attempts == 1
    assert client._slot is None
    assert client._shm is None
