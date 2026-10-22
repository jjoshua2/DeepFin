from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from chess_anti_engine.inference import MultiSlotInferenceClient, SlotInferenceClient


@pytest.mark.parametrize("legal", [False, True])
def test_scalar_is_rejected_before_pool_acquisition(legal: bool) -> None:
    client = MultiSlotInferenceClient(slot_names=["unused"], max_batch=4)
    try:
        x = np.array(0, dtype=np.uint16 if legal else np.float32)
        with pytest.raises(ValueError, match="expected encoded batch shape"):
            if legal:
                client.evaluate_legal_bf16(x, np.array([], dtype=np.int32), np.array([], dtype=np.int32))
            else:
                client.evaluate_encoded(x)
        assert client.stats["inflight"] == 0
        assert client.stats["available_slots"] == 1
        assert client.stats["lifetime_requests"] == 0
        assert client.stats["slot_quarantines"] == [0]
    finally:
        client.close()


@pytest.mark.parametrize("legal", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_lease_releases_once_with_captured_rows(
    monkeypatch: pytest.MonkeyPatch, legal: bool, fail: bool,
) -> None:
    client = MultiSlotInferenceClient(
        slot_names=["unused"], max_batch=4, slot_failure_threshold=1,
    )
    released: list[int] = []
    original = client._release_client

    def release(*args: Any, **kwargs: Any) -> None:
        released.append(kwargs["positions"])
        original(*args, **kwargs)

    def evaluate(_self: SlotInferenceClient, x: np.ndarray, *_args: Any) -> tuple[np.ndarray, np.ndarray]:
        # Even caller mutation after acquisition must not corrupt cleanup.
        x.shape = () if x.size == 1 else (x.size,)
        if fail:
            raise TimeoutError("submitted failure")
        return np.zeros((1, 1)), np.zeros((1, 3))

    monkeypatch.setattr(client, "_release_client", release)
    monkeypatch.setattr(SlotInferenceClient, "evaluate_legal_bf16" if legal else "evaluate_encoded", evaluate)
    x = np.zeros((1, 1, 1, 1), dtype=np.uint16 if legal else np.float32)

    def call() -> None:
        if legal:
            client.evaluate_legal_bf16(x, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32))
        else:
            client.evaluate_encoded(x)

    try:
        if fail:
            with pytest.raises(TimeoutError, match="submitted failure"):
                call()
        else:
            call()
        assert released == [1]
        stats = client.stats
        assert stats["inflight"] == 0
        assert stats["lifetime_requests"] == 1
        assert stats["lifetime_failed_positions"] == int(fail)
        assert stats["lifetime_positions"] == int(not fail)
        assert stats["slots_quarantined"] == int(fail)
        assert stats["available_slots"] == int(not fail)
    finally:
        client.close()


def test_negative_legal_count_rejected_before_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SlotInferenceClient(slot_name="unused", max_batch=4)
    published: list[bool] = []

    def publish(*_args: Any) -> tuple[np.ndarray, np.ndarray]:
        published.append(True)
        raise AssertionError("invalid metadata was published")

    monkeypatch.setattr(client, "_submit_and_wait_locked", publish)
    with pytest.raises(ValueError, match="negative counts"):
        client.evaluate_legal_bf16(
            np.zeros((2, 146, 8, 8), dtype=np.uint16),
            np.array([3], dtype=np.int32), np.array([-1, 2], dtype=np.int32),
        )
    assert not published
