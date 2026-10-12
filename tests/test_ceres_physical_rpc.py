"""CPU fake-session shape propagation through actual pipelined Ceres RPC."""
from __future__ import annotations

import json
import socket
import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.ceres_raw_backend import Root
from scripts.ceres_shared_game_service import bind_ceres_raw_backend, serve_ceres_stream


class FakeSession:
    def __init__(self) -> None:
        self.feeds: list[np.ndarray] = []

    def run(self, names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == ["policy", "value", "value2"]
        feed = inputs["squares_byte"]
        self.feeds.append(feed.copy())
        ids = feed[:, 0, 0].astype(np.int64) + 256 * feed[:, 0, 1].astype(np.int64)
        return [np.tile(ids[:, None], (1, 1858)).astype(np.float16),
                (ids[:, None] + np.arange(3)).astype(np.float16),
                -(ids[:, None] + np.arange(3, 6)).astype(np.float16)]


def roots_for(message: dict[str, Any]) -> tuple[Root, ...]:
    roots = []
    for index in range(300):
        identity = message["game_id"] * 1024 + index
        fen = f"CPU game{message['game_id']} root{index}"
        feed = np.zeros((64, 137), np.uint8)
        feed[0, :2] = identity % 256, identity // 256
        board = SimpleNamespace(fen=lambda value=fen: value, move_stack=[])
        roots.append(Root(identity, board, fen, (), ("CPU legal move",), (0,), (0,), feed))
    return tuple(roots)


@pytest.mark.parametrize("physical", [32, 256, 512])
def test_physical_shape_reaches_rpc_and_padding_never_becomes_labels(physical: int) -> None:
    session = FakeSession()
    accounting: dict[str, int] = {}
    infer = bind_ceres_raw_backend(
        session=session, physical_batch=physical,
        gather_context=lambda feed: (np.zeros(len(feed)), np.zeros(len(feed))),
        gather_indices=lambda pawn, _castle: np.tile(np.arange(1858), (len(pawn), 1)),
        accounting=accounting,
    )
    published: dict[int, tuple[Any, ...]] = {}
    def publish(message: dict[str, Any], values: tuple[Any, ...]) -> dict[str, Any]:
        game = message["game_id"]
        assert message["raw_sha256"] == ("a" if game == 0 else "b") * 64
        assert len(values) == 300
        for index, value in enumerate(values):
            assert value.slot_id == game * 1024 + index
            assert value.fen == f"CPU game{game} root{index}"
            assert np.array_equal(value.policy_logits, np.full(1858, value.slot_id, np.float16))
            assert np.array_equal(value.value_logits, (value.slot_id + np.arange(3)).astype(np.float16))
            assert np.array_equal(value.value2_logits, -(value.slot_id + np.arange(3, 6)).astype(np.float16))
            assert all(head.dtype == np.float16 for head in
                       (value.policy_logits, value.value_logits, value.value2_logits))
        published[game] = values
        return {"raw_sha256": message["raw_sha256"], "rows": len(values)}
    server, client = socket.socketpair()
    client.settimeout(10)
    result: list[Any] = []
    def run() -> None:
        try:
            with server.makefile("r") as incoming, server.makefile("w") as outgoing:
                result.append(serve_ceres_stream(
                    incoming, outgoing, total_games=2, max_games=2, target_rows=512,
                    max_rows=1024, batch_wait_ms=1000, poll_seconds=0.001,
                    deadline_seconds=10, load_roots=roots_for, infer=infer, publish=publish,
                ))
        except BaseException as exc:
            result.append(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        client.sendall((json.dumps({"op": "game", "game_id": 0, "raw_sha256": "a" * 64}) + "\n"
                        + json.dumps({"op": "game", "game_id": 1, "raw_sha256": "b" * 64}) + "\n"
                        + '{"op":"stop"}\n').encode())
        with client.makefile("r") as replies:
            assert json.loads(replies.readline())["state"] == "READY_FOR_GAME"
            assert {json.loads(replies.readline())["game_id"] for _ in range(2)} == {0, 1}
            assert json.loads(replies.readline())["games"] == 2
        thread.join(10)
        assert not thread.is_alive()
        assert result
        assert isinstance(result[0], dict)
        assert result[0]["logical_batch_histogram"] == {512: 1, 88: 1}
        calls = 512 // physical + (88 + physical - 1) // physical
        assert accounting == {"calls": calls, "real_rows": 600,
                              "padding_rows": calls * physical - 600, "physical_rows": calls * physical}
        assert all(feed.shape == (physical, 64, 137) for feed in session.feeds)
        last_real = 88 % physical or physical
        assert np.all(session.feeds[-1][last_real:] == session.feeds[-1][last_real - 1])
        assert set(published) == {0, 1}
    finally:
        server.close()
        client.close()


def test_physical_backend_invalid_shape_rejects_before_session_call() -> None:
    session = FakeSession()
    with pytest.raises(ValueError, match="physical batch"):
        bind_ceres_raw_backend(session=session, physical_batch=128,
                               gather_context=lambda feed: (feed, feed),
                               gather_indices=lambda a, _b: a, accounting={})
    assert not session.feeds


@pytest.mark.parametrize("failure", ["legal_oracle", "board_mutation", "raw_dtype"])
def test_bound_physical_backend_fails_closed_at_real_adapter(failure: str) -> None:
    class Session(FakeSession):
        def run(self, names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            values = super().run(names, inputs)
            if failure == "raw_dtype":
                values[2] = values[2].astype(np.float32)
            return values
    session = Session()
    roots = roots_for({"game_id": 0})[:1]
    if failure == "board_mutation":
        roots[0].board.move_stack.append("changed CPU history")
    def gather(pawn: np.ndarray, _castle: np.ndarray) -> np.ndarray:
        indices = np.tile(np.arange(1858), (len(pawn), 1))
        if failure == "legal_oracle":
            indices[:, 0] = 1
        return indices
    accounting: dict[str, int] = {}
    infer = bind_ceres_raw_backend(session=session, physical_batch=512,
                                   gather_context=lambda feed: (np.zeros(len(feed)), np.zeros(len(feed))),
                                   gather_indices=gather, accounting=accounting)
    with pytest.raises(ValueError, match="C3"):
        infer(roots)
    assert len(session.feeds) == (1 if failure == "raw_dtype" else 0)
    assert accounting == {"calls": 0, "real_rows": 0, "padding_rows": 0, "physical_rows": 0}


@pytest.mark.parametrize("indices", [(-1,), (1858,), (0, 0), ()])
def test_label_roots_validate_legal_indices_before_session(indices: tuple[int, ...]) -> None:
    session = FakeSession()
    root = replace(roots_for({"game_id": 0})[0], compact_indices=indices)
    infer = bind_ceres_raw_backend(session=session, physical_batch=256,
                                   gather_context=lambda feed: (feed, feed),
                                   gather_indices=lambda a, _b: a, accounting={})
    with pytest.raises(ValueError, match="C3 root legal indices"):
        infer((root,))
    assert not session.feeds
