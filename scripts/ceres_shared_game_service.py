"""Pipelined game RPC consumer for the shared teacher dispatcher.

Resource/session admission stays with the existing owner. Its immutable raw
history loader and atomic companion publisher are supplied directly; no second
model loader, root encoder, watchdog, or publication schema is introduced.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import json
import math
import os
import select
import time
from typing import Any, TextIO

import numpy as np

from chess_anti_engine.teacher_dispatch import TeacherDispatcher
from scripts.shared_teacher_generation import CompletedGameLabels


def checked_ceres_backend(
    infer: Callable[[Sequence[Any]], Sequence[Any]],
) -> Callable[[Sequence[Any]], Sequence[Any]]:
    """Keep the frozen C3 raw three-head contract and validate exact routing."""
    def evaluate(roots: Sequence[Any]) -> Sequence[Any]:
        identities = [(root.slot_id, root.fen, np.ascontiguousarray(root.feed).tobytes()) for root in roots]
        values = tuple(infer(roots))
        if len(values) != len(roots):
            raise ValueError("Ceres exact root coverage changed")
        for root, value, (root_id, fen, feed_bytes) in zip(roots, values, identities):
            digest = hashlib.sha256(feed_bytes).hexdigest()
            if (root.feed.dtype != np.uint8 or root.feed.shape != (64, 137)
                    or root.slot_id != root_id or root.fen != fen
                    or np.ascontiguousarray(root.feed).tobytes() != feed_bytes
                    or value.slot_id != root_id or value.fen != fen
                    or value.feed.dtype != np.uint8 or value.feed.shape != (64, 137)
                    or value.feed_sha256 != digest or not np.array_equal(value.feed, root.feed)):
                raise ValueError("Ceres stale/misrouted root or feed identity")
            for name, width in (("policy_logits", 1858), ("value_logits", 3), ("value2_logits", 3)):
                array = getattr(value, name)
                if array.dtype != np.float16 or array.shape != (width,) or not np.isfinite(array).all():
                    raise ValueError("Ceres raw FP16 head contract changed: " + name)
        return values
    return evaluate


def serve_ceres_stream(
    incoming: TextIO, outgoing: TextIO, *, total_games: int, max_games: int,
    target_rows: int, max_rows: int, batch_wait_ms: float,
    poll_seconds: float, deadline_seconds: float,
    load_roots: Callable[[dict[str, Any]], Sequence[Any]],
    infer: Callable[[Sequence[Any]], Sequence[Any]],
    publish: Callable[[dict[str, Any], tuple[Any, ...]], dict[str, Any]],
    control: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Actual bounded JSON-line RPC: pipeline games, route complete durable replies.

    The independent fill deadline always drains low volume. select's existing
    response-poll cadence remains separate from that deadline. A stop/EOF stops
    admission and drains accepted requests; the owner-supplied guard runs each
    controller poll and before backend/publication. One held input request plus
    max_games admitted games bounds decoded histories and result storage.
    """
    if (type(total_games) is not int or not 1 <= total_games <= 128
            or not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 1
            or not math.isfinite(deadline_seconds) or deadline_seconds <= 0):
        raise ValueError("finite explicit RPC poll and operation budget required")
    messages: dict[int, dict[str, Any]] = {}
    responses: dict[int, dict[str, Any]] = {}
    evaluate = checked_ceres_backend(infer)

    def guarded_infer(roots: Sequence[Any]) -> Sequence[Any]:
        control()
        return evaluate(roots)

    def durable(game_id: int, raw_sha: str, values: tuple[Any, ...]) -> None:
        control()
        message = messages[game_id]
        if message["raw_sha256"] != raw_sha:
            raise ValueError("Ceres raw-game provenance changed")
        responses[game_id] = publish(message, values)

    dispatcher = TeacherDispatcher(guarded_infer, target_rows=target_rows, max_rows=max_rows, batch_wait_ms=batch_wait_ms)
    try:
        labels = CompletedGameLabels(dispatcher, max_games=max_games, total_games=total_games,
                                    max_game_rows=400, publish=durable)
    except BaseException:
        dispatcher.close(timeout=min(deadline_seconds, 30))
        raise
    deadline = time.monotonic() + deadline_seconds
    held: tuple[dict[str, Any], Sequence[Any]] | None = None
    closing = False
    wire_buffer = b""

    def emit(message: dict[str, Any]) -> None:
        outgoing.write(json.dumps(message, sort_keys=True) + "\n")
        outgoing.flush()

    try:
        emit({"state": "READY_FOR_GAME", "gpu_session_opened": False})
        while not closing or labels.pending or held is not None:
            control()
            if time.monotonic() >= deadline:
                raise TimeoutError("finite Ceres RPC budget expired; owned state retained")
            for game_id in labels.drain():
                emit({**responses.pop(game_id), "state": "GAME_DURABLE", "game_id": game_id})
                del messages[game_id]
            if held is not None:
                message, roots = held
                try:
                    labels.submit(message["game_id"], message["raw_sha256"], roots)
                except BufferError:
                    pass
                else:
                    messages[message["game_id"]] = message
                    held = None
            if not closing and held is None:
                ready, _, _ = select.select([incoming], [], [], 0 if b"\n" in wire_buffer else poll_seconds)
                if ready or b"\n" in wire_buffer:
                    if b"\n" not in wire_buffer:
                        chunk = os.read(incoming.fileno(), 4096)
                        if not chunk:
                            if wire_buffer:
                                raise ValueError("truncated Ceres RPC message")
                            wire_buffer = b'{"op":"stop"}\n'
                        else:
                            wire_buffer += chunk
                            if len(wire_buffer) > 8192:
                                raise ValueError("Ceres RPC wire buffer exceeds bound")
                    if b"\n" not in wire_buffer:
                        continue
                    line, wire_buffer = wire_buffer.split(b"\n", 1)
                    message = json.loads(line)
                    if message["op"] == "stop":
                        closing = True
                        dispatcher.flush()
                    elif message["op"] == "game":
                        game_id = message["game_id"]
                        if (type(game_id) is not int or not 0 <= game_id < total_games
                                or game_id in labels.pending or game_id in labels.durable):
                            raise ValueError("distinct finite Ceres game RPC required")
                        held = message, load_roots(message)
                    else:
                        raise ValueError("unsupported Ceres game RPC operation")
            else:
                time.sleep(min(poll_seconds, 0.01))
        result = {"status": "C3_GAME_LABEL_SERVICE_COMPLETE_NOT_ADMISSION",
                  "games": len(labels.durable), "logical_batch_histogram": dispatcher.histogram}
        emit(result)
        return result
    finally:
        try:
            dispatcher.close(timeout=min(deadline_seconds, 30))
        finally:
            labels.close(timeout=min(deadline_seconds, 30))
