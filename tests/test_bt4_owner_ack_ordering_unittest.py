"""UNRUN component controls: real writer queue, retained verifier/SQLite ACK, refill.

No model, native stepper, engine, admission or canonical custody is constructed.
The explicit fixture seam is a zero-row discarded NPZ publisher and inert actor.
Production serialization/gameplay and run_owned_roster admission are not covered.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace
from typing import Any, cast
import unittest

import numpy as np

from chess_anti_engine.producer_executor import ProducerExecutor
from scripts.shared_teacher_generation import BT4GamePool, BT4GenerationDriver, _Game
from scripts.shared_teacher_owner import bind_owner_checkpoints

RETAINED_OWNERSHIP = []  # An outer finite process owner resolves unknown joins.


@contextmanager
def expected_error(kind, pattern):
    """Stdlib-only equivalent of the former expected type/regex checks."""
    caught = None
    try:
        yield
    except kind as error:
        caught = error
    else:
        raise AssertionError(f"{kind.__name__} was not raised")
    assert re.search(pattern, str(caught)) is not None, str(caught)


def retained_module(name: str, relative: str):
    """Load only the exact two retained reference bodies, after manifest checks."""
    fixtures = Path(__file__).parent / "fixtures" / "teacher_reference"
    manifest = json.loads((fixtures / "manifest.json").read_text())
    if manifest["schema"] != 2:
        raise ValueError("retained manifest schema")
    records = [item for item in manifest["files"] if item["relative"] == relative]
    if len(records) != 1:
        raise ValueError("one exact retained source required")
    record = records[0]
    raw = (fixtures / record["fixture"]).read_bytes()
    if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("retained source changed")
    # The checked .txt source is a normal source module; no AST extraction,
    # package replacement, private path relocation or source rewriting occurs.
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader(name, str(fixtures / record["fixture"]))
    spec = importlib.util.spec_from_loader(name, loader)
    if spec is None:
        raise ValueError("source module specification")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class OwnerAckOrdering(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # Resource-free defaults; actual resources still start in setUp only.
        self.temporary: tempfile.TemporaryDirectory[str] | None = None
        self._pool: BT4GamePool | None = None
        self.producers: ProducerExecutor | None = None
        self._driver: BT4GenerationDriver | None = None
        self._game: _Game | None = None
        self.root = self.raw = self.launch = Path()
        self.owner = self.attempt = 0
        self.events: list[tuple[str, int, int]] = []
        self.stopped = self.failure = self.mismatch = False
        self.entered: threading.Event | None = None
        self.release: threading.Event | None = None
        # Dynamic retained-reference modules/checkpoint are explicit fixture boundaries.
        self.checkpoint: Any = None
        self.recovery: Any = None

    @property
    def pool(self) -> BT4GamePool:
        assert self._pool is not None
        return self._pool

    @property
    def driver(self) -> BT4GenerationDriver:
        assert self._driver is not None
        return self._driver

    @property
    def game(self) -> _Game:
        assert self._game is not None
        return self._game

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bt4-owner-ack-component-")
        self.addCleanup(self.cleanup_owned)
        self.root = Path(self.temporary.name).resolve()
        self.owner = threading.get_ident()
        self.events = []
        self.stopped = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.failure = False
        self.mismatch = False
        checkpoint_module = retained_module("ack_checkpoint_component",
                                            "scripts/bt4_longblock_checkpoint_v2.py")
        self.recovery = retained_module("ack_recovery_component", "scripts/bt4_raw_recovery_v2.py")
        limits = {"max_attempts": 1, "unit_seconds": 10, "unit_io_bytes": 1 << 20,
                  "unit_output_bytes": 1 << 20, "total_seconds": 10,
                  "total_io_bytes": 1 << 20, "total_output_bytes": 1 << 20}
        # Explicit fake custody identity; only retained SQLite semantics are tested.
        self.checkpoint = checkpoint_module.Checkpoint(self.root / "checkpoint.sqlite",
                                                       {"fixture": "component-only"}, limits, self.guard)
        self.raw = self.root / "attempt" / "raw"
        self.attempt = self.checkpoint.begin(0, {"pid": 1, "start_ticks": 1,
                                               "boot_id": "synthetic-no-custody"}, str(self.raw))
        assert not self.raw.exists()
        self.raw.mkdir(parents=True)
        self.launch = self.raw / "launch.json"
        self.launch.write_text('{"fixture":"synthetic-zero-row"}')

        def verify_raw(unit, path, game):
            assert unit == "u0"
            return self.recovery.verified_game(path, game, self.checkpoint.binding,
                                              "0" * 64, self.guard, lambda _size: None)

        completed, acknowledge = bind_owner_checkpoints(
            units={"u0": (0, self.raw, self.checkpoint, self.attempt)},
            verify_raw=verify_raw, verify_completed=lambda _unit, _receipt: self.guard())

        def ack(game, receipt):
            self.events.append(("ack_enter", game, threading.get_ident()))
            result = acknowledge("u0", game, receipt)
            assert self.checkpoint.completed(0, lambda _receipt: None) == (game,)
            self.events.append(("ack_committed", game, threading.get_ident()))
            return result

        def make_actor(game: int, _rng: np.random.Generator) -> Any:
            self.guard()
            if game:
                assert self.checkpoint.completed(0, lambda _receipt: None) == (0,)
                assert self.events[-1][:2] == ("ack_committed", 0)
            self.events.append(("actor", game, threading.get_ident()))
            return SimpleNamespace(slot_id=game)  # no native or board computation

        self._pool = BT4GamePool(games=2, capacity=1, seed=0, temperature=0,
            make_stepper=make_actor, dispatcher=cast(Any, None), commit=self.publish, max_writes=1,
            completed=completed["u0"], namespace="u0", acknowledge=ack)
        # The single cleanup retains every dependency if a join is unknown.
        producers = ProducerExecutor(max_workers=1, capacity=1)
        self.producers = producers
        self._driver = BT4GenerationDriver({"u0": self.pool}, producers=producers,
            native_microbatch=1, max_live_games=1, retained_byte_budget=2,
            reservation_bytes={"u0": 1}, should_stop=lambda: self.stopped)
        assert self.driver._fill("u0")
        self._game = self.pool.active[0]
        self.game.finalized = cast(Any, SimpleNamespace(slot_id=0))
        self.driver._publications[("u0", 0)] = self.game

    def cleanup_owned(self):
        errors = []
        if self.release is not None:
            self.release.set()  # Fixture gate only; no production STOP mutation.
        for resource, method in ((self._pool, "close"), (self.producers, "finish")):
            if resource is not None:
                try:
                    getattr(resource, method)(timeout=2)
                except BaseException as error:
                    errors.append(error)
        if errors:
            RETAINED_OWNERSHIP.append(self)
            raise RuntimeError("component join unknown; SQLite/output retained") from errors[0]
        try:
            if self.checkpoint is not None:
                self.checkpoint.close()
            if self.temporary is not None:
                self.temporary.cleanup()
        except BaseException:
            RETAINED_OWNERSHIP.append(self)
            raise

    def guard(self):
        assert threading.get_ident() == self.owner

    def publish(self, finalized):
        """Fixture-only atomic storage, executed by the actual DurableWriter."""
        self.events.append(("writer_enter", finalized.slot_id, threading.get_ident()))
        assert self.entered is not None
        self.entered.set()
        assert self.release is not None
        if not self.release.wait(2):
            raise TimeoutError("bounded fixture release")
        if self.failure:
            raise OSError("injected storage failure before receipt")
        game = finalized.slot_id
        games = self.raw / "games"
        games.mkdir(exist_ok=True)
        path = games / f"game_{game:08d}.npz"
        partial = games / f"game_{game:08d}.partial"
        metadata = {"schema": "bt4_root_policy_games_v1", "game_id": game,
                    "status": "discarded", "rows": [], "discarded_rows": 1,
                    "termination": "synthetic-component-discard"}
        with partial.open("xb") as stream:
            np.savez(stream, metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
                     x=np.empty((0, 0, 8, 8), dtype=np.float32),
                     policy_t1=np.empty((0, 1858), dtype=np.float32),
                     wdl_raw=np.empty((0, 3), dtype=np.float32))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        fd = os.open(games, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        receipt = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "game_id": game, "status": "discarded", "rows": 0, "discarded_rows": 1,
                   "termination": metadata["termination"],
                   "launch_sha256": hashlib.sha256(self.launch.read_bytes()).hexdigest()}
        self.events.append(("writer_receipt", game, threading.get_ident()))
        if self.mismatch:
            receipt["sha256"] = "f" * 64
        return receipt

    def submit_write(self):
        self.driver._publish()
        future = self.game.writing
        assert future is not None
        assert self.entered is not None
        assert self.entered.wait(2)
        return future

    def assert_retained(self):
        assert self.pool.active[0] is self.game
        assert self.pool.next_id == 1
        assert self.pool.receipts == {}
        assert self.checkpoint.completed(0, lambda _receipt: None) == ()
        assert self.driver.live == 1
        assert self.driver.credits.used == 1
        assert not any(event[0] == "actor" and event[1] == 1 for event in self.events)

    def test_receipt_then_actual_checkpoint_ack_then_refill(self):
        future = self.submit_write()
        future.result(timeout=2)
        self.assert_retained()  # raw receipt alone has not earned an ACK
        self.driver._publish()
        assert 0 not in self.pool.active
        assert 1 in self.pool.active
        assert self.pool.next_id == 2
        assert self.checkpoint.completed(0, lambda _receipt: None) == (0,)
        names = [event[0] for event in self.events]
        assert names.index("writer_receipt") < names.index("ack_enter")
        assert names.index("ack_committed") < len(names) - 1
        assert self.events[-1][:2] == ("actor", 1)
        assert self.events[names.index("writer_receipt")][2] != self.owner
        assert self.events[names.index("ack_committed")][2] == self.owner
        with expected_error(ValueError, "full unit"):
            self.checkpoint.finish(self.attempt, True)

    def test_failed_writer_has_no_ack_or_replacement(self):
        self.failure = True
        future = self.submit_write()
        with expected_error(OSError, "injected storage"):
            future.result(timeout=2)
        with expected_error(OSError, "injected storage"):
            self.driver._publish()
        self.assert_retained()
        assert not any(event[0].startswith("ack_") for event in self.events)

    def test_mismatched_immutable_receipt_has_no_ack_or_replacement(self):
        self.mismatch = True
        self.submit_write().result(timeout=2)
        with expected_error(ValueError, "canonical owner raw receipt"):
            self.driver._publish()
        self.assert_retained()
        assert not any(event[0] == "ack_committed" for event in self.events)

    def test_stop_during_pending_writer_retains_slot_and_durable_raw(self):
        assert self.release is not None
        self.release.clear()
        future = self.submit_write()
        self.stopped = True
        assert self.driver._observe_stop()
        self.driver._publish()
        assert not future.done()
        self.assert_retained()
        self.release.set()
        future.result(timeout=2)
        self.driver._publish()
        assert self.game.raw_receipt is not None
        self.assert_retained()
        assert not any(event[0].startswith("ack_") for event in self.events)


if __name__ == "__main__":
    unittest.main()
