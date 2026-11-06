"""Bounded attachment recovery before a POSIX shared-memory slot is sized."""
from __future__ import annotations

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


def test_permanently_empty_map_obeys_existing_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def attach(**_kwargs: Any) -> SharedMemory:
        nonlocal attempts
        attempts += 1
        raise ValueError("cannot mmap an empty file")

    monkeypatch.setattr(inf, "SharedMemory", attach)
    monkeypatch.setattr(inf.time, "monotonic", lambda: 10.0)
    client = _client("synthetic-unready")
    with pytest.raises(TimeoutError, match="was not available"):
        client._connect(deadline=9.0)
    assert attempts == 1
    assert client._slot is None
    assert client._shm is None


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
