"""Bounded pipelining over the retained Companion's already owned child.

No subprocess/session/lease is opened here. Receipt checks preserve the retained
DeferredCompanion contract; wire IDs namespace local unit/game identities only.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
import hashlib
import json
import os
from pathlib import Path
import select
from typing import Any


class PipelinedCompanion:
    def __init__(self, client: Any, units: tuple[str, ...], *, max_pending: int, wire_bytes: int) -> None:
        self.validate_bounds(units, max_pending=max_pending, wire_bytes=wire_bytes)
        self.client = client
        self.units = units
        self.max_pending = max_pending
        self.wire_bytes = wire_bytes
        self.pending: dict[int, tuple[str, int, dict[str, Any], Future[dict[str, Any]]]] = {}
        self.submitted: set[int] = set()
        self._out: deque[bytes] = deque()
        self._out_bytes = 0
        self._in = b""
        os.set_blocking(client.child.stdin.fileno(), False)
        os.set_blocking(client.child.stdout.fileno(), False)

    @staticmethod
    def validate_bounds(units: tuple[str, ...], *, max_pending: int, wire_bytes: int) -> None:
        if (not 1 <= len(units) <= 4096 or len(set(units)) != len(units)
                or any(type(unit) is not str or not unit or len(unit.encode()) > 128 for unit in units)
                or type(max_pending) is not int or not 1 <= max_pending <= 128 * len(units)
                or type(wire_bytes) is not int or not 8192 <= wire_bytes <= 2 * 1024 ** 2):
            raise ValueError("finite explicit Ceres pipeline bounds required")

    def submit(self, unit: str, raw: dict[str, Any], game: int) -> Future[dict[str, Any]]:
        if unit not in self.units or type(game) is not int or not 0 <= game < 128:
            raise ValueError("exact finite local unit/game required")
        request = self.units.index(unit) * 128 + game
        if request in self.submitted:
            raise ValueError("duplicate companion request")
        if len(self.pending) >= self.max_pending:
            raise BufferError("bounded companion pipeline backpressure")
        self.client.guard()
        self.client.check_priority()
        message = json.dumps({"op": "game", "unit_id": unit, "game_id": game, "request_id": request,
                                  "raw_path": raw["path"], "raw_sha256": raw["sha256"]}).encode() + b"\n"
        if len(message) > 8192:
            raise ValueError("bounded companion wire message required")
        if self._out_bytes + len(message) > self.wire_bytes:
            raise BufferError("bounded companion wire backpressure")
        future: Future[dict[str, Any]] = Future()
        future.set_running_or_notify_cancel()
        self.pending[request] = unit, game, dict(raw), future
        self.submitted.add(request)
        self._out.append(message)
        self._out_bytes += len(message)
        return future

    def _receipt(self, reply: dict[str, Any]) -> None:
        request = reply.get("request_id")
        if type(request) is not int or request not in self.pending:
            raise ValueError("unknown/duplicate Ceres wire acknowledgment")
        unit, game, raw, future = self.pending[request]
        if (reply.get("state") != "GAME_DURABLE" or reply.get("unit_id") != unit
                or reply.get("game_id") != game or reply.get("real_rows") != raw["rows"]):
            raise ValueError("Ceres acknowledgment changed local raw identity")
        ref = reply["companion"]
        expected = Path(raw["path"]).parent.parent / "ceres_companions"
        path = Path(ref["path"])
        def sha(value: Path) -> str:
            return hashlib.sha256(value.read_bytes()).hexdigest()
        if path != expected / (Path(raw["path"]).stem + ".json") or sha(path) != ref["sha256"]:
            raise ValueError("Ceres durable companion binding changed")
        receipt = json.loads(path.read_text())
        labels = Path(receipt["labels"]["path"])
        if labels != expected / Path(raw["path"]).name or sha(labels) != receipt["labels"]["sha256"]:
            raise ValueError("Ceres durable labels binding changed")
        if (receipt["raw_sha256"] != raw["sha256"] or receipt["game_id"] != game
                or receipt["real_rows"] != raw["rows"] or receipt["unit_id"] != unit):
            raise ValueError("Ceres durable local unit/game binding changed")
        self.client.charge(path.stat().st_size + labels.stat().st_size)
        future.set_result({**raw, "ceres_companion": ref})
        del self.pending[request]

    def poll(self) -> None:
        """One bounded nonblocking wire turn; caller retains control/deadline polling."""
        c = self.client
        try:
            c.guard()
            c.check_priority()
            if c.child.poll() is not None:
                raise RuntimeError("owned Ceres service exited with pending publication")
            c.seen.extend(identity for identity in c.cleanup.members(c.child.pid) + c.cleanup.descendants(c.child.pid)
                          if identity not in c.seen)
            if self._out:
                try:
                    written = os.write(c.child.stdin.fileno(), self._out[0])
                except BlockingIOError:
                    written = 0
                if written:
                    self._out_bytes -= written
                    tail = self._out.popleft()[written:]
                    if tail:
                        self._out.appendleft(tail)
            if select.select([c.child.stdout], [], [], 0)[0]:
                chunk = os.read(c.child.stdout.fileno(), 4096)
                if not chunk:
                    raise RuntimeError("Ceres RPC EOF before owned completion")
                self._in += chunk
                if len(self._in) > self.wire_bytes:
                    raise ValueError("Ceres reply wire buffer exceeds bound")
            for _ in range(min(self.max_pending, 64)):
                if b"\n" not in self._in:
                    if len(self._in) > 8192:
                        raise ValueError("Ceres reply message exceeds bound")
                    break
                line, self._in = self._in.split(b"\n", 1)
                if len(line) > 8192:
                    raise ValueError("Ceres reply message exceeds bound")
                self._receipt(json.loads(line))
        except BaseException as exc:
            for _unit, _game, _raw, future in self.pending.values():
                if not future.done():
                    future.set_exception(exc)
            raise
