"""Low-dependency actual-component controls, independent of pytest/conftest.

No Syzygy files are read. The library's counted-reader, LRU and close methods
are real; only table decoding is replaced to expose a bounded lock schedule.
These controls establish neither real decoding nor opener/rule50 integration.
"""
from __future__ import annotations

import threading
import unittest
from collections.abc import Callable
from unittest.mock import patch

import chess
import chess.syzygy

from chess_anti_engine.strict_match_tablebase import StrictMatchTablebase


FIRST = chess.Board("7k/8/8/8/8/8/8/KQ6 w - - 0 1")
SECOND = chess.Board("7k/8/8/8/8/8/8/KR6 w - - 0 1")


class Gate:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.owner: int | None = None
        self.depth = 0
        self.contender = threading.Event()

    def __enter__(self) -> None:
        identity = threading.get_ident()
        if self.owner is not None and self.owner != identity:
            self.contender.set()
        self.lock.acquire()
        if self.owner is None:
            self.owner = identity
        self.depth += 1

    def __exit__(self, *_error: object) -> None:
        self.depth -= 1
        if self.depth == 0:
            self.owner = None
        self.lock.release()


class ObservedCondition(threading.Condition):
    def __init__(self, waiting: threading.Event) -> None:
        super().__init__()
        self.waiting = waiting

    def wait(self, timeout: float | None = None) -> bool:
        self.waiting.set()
        return super().wait(timeout)


class ScheduledTable(chess.syzygy.WdlTable):
    def __init__(self, name: str, entered: threading.Event, release: threading.Event,
                 eviction_wait: threading.Event) -> None:
        super().__init__(f"not-opened/{name}.rtbw")
        self.first = name == "KQvK"
        self.entered, self.release = entered, release
        self.read_condition = ObservedCondition(eviction_wait)

    def _probe_wdl_table(self, board: chess.Board) -> int:
        if self.first:
            self.entered.set()
            if not self.release.wait(2):
                raise TimeoutError("counted-reader release")
        if not self.write_lock.acquire(timeout=0.5):
            # Turn the actual reader/close cycle into a recoverable test error.
            raise TimeoutError("counted reader blocked by eviction write lock")
        try:
            return 0
        finally:
            self.write_lock.release()


def capture(errors: list[Exception], action: Callable[[], object]) -> None:
    try:
        action()
    except Exception as error:
        errors.append(error)


class StrictMatchComponentControls(unittest.TestCase):
    def schedule(self, serialized: bool) -> list[Exception]:
        handle = StrictMatchTablebase() if serialized else chess.syzygy.Tablebase()
        handle.max_fds = 1  # Small dependency schedule, not production geometry.
        entered, release, eviction_wait = (threading.Event() for _ in range(3))
        first = ScheduledTable("KQvK", entered, release, eviction_wait)
        second = ScheduledTable("KRvK", entered, release, eviction_wait)
        handle.wdl[chess.syzygy.calc_key(FIRST)] = first
        handle.wdl[chess.syzygy.calc_key(SECOND)] = second
        gate = Gate()
        if serialized:
            # Preserve real RLock semantics and observe contention deterministically.
            with patch.object(handle, "_match_probe_lock", gate):
                return self.run_schedule(handle, first, second, entered, release,
                                         eviction_wait, gate)
        return self.run_schedule(handle, first, second, entered, release, eviction_wait, None)

    def run_schedule(self, handle: chess.syzygy.Tablebase, first: ScheduledTable,
                     second: ScheduledTable, entered: threading.Event,
                     release: threading.Event, eviction_wait: threading.Event,
                     gate: Gate | None) -> list[Exception]:
        errors: list[Exception] = []
        a = threading.Thread(target=capture, args=(errors, lambda: handle.probe_wdl(FIRST.copy())))
        b = threading.Thread(target=capture, args=(errors, lambda: handle.probe_wdl(SECOND.copy())))
        try:
            a.start()
            self.assertTrue(entered.wait(2))
            b.start()
            self.assertTrue((gate.contender if gate else eviction_wait).wait(2))
            release.set()
        finally:
            release.set()
            a.join(3)
            if b.ident is not None:
                b.join(3)
        self.assertFalse(a.is_alive())
        self.assertFalse(b.is_alive())
        self.assertEqual(first.read_count, 0)
        self.assertEqual(second.read_count, 0)
        self.assertEqual(eviction_wait.is_set(), gate is None)
        handle.close()
        return errors

    def test_actual_reader_lru_original_negative_and_serialized_success(self) -> None:
        errors = self.schedule(False)
        self.assertEqual(len(errors), 1)
        self.assertEqual(str(errors[0]), "counted reader blocked by eviction write lock")
        self.assertEqual(self.schedule(True), [])

    def test_atomic_pair_reentrancy_and_competing_probe(self) -> None:
        handle, gate = StrictMatchTablebase(), Gate()
        entered, release = threading.Event(), threading.Event()
        trace: list[str] = []
        errors: list[Exception] = []

        def wdl(_handle: chess.syzygy.Tablebase, board: chess.Board) -> int:
            trace.append("first_wdl" if board.queens else "second_wdl")
            if board.queens:
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("pair release")
            return 2

        def dtz(_handle: chess.syzygy.Tablebase, _board: chess.Board) -> int:
            trace.append("first_dtz")
            return 1

        with (patch.object(handle, "_match_probe_lock", gate),
              patch.object(chess.syzygy.Tablebase, "probe_wdl", wdl),
              patch.object(chess.syzygy.Tablebase, "probe_dtz", dtz)):
            a = threading.Thread(target=capture, args=(errors, lambda: handle.probe_wdl_dtz(FIRST.copy())))
            b = threading.Thread(target=capture, args=(errors, lambda: handle.probe_wdl(SECOND.copy())))
            try:
                a.start()
                self.assertTrue(entered.wait(2))
                b.start()
                self.assertTrue(gate.contender.wait(2))
                release.set()
            finally:
                release.set()
                a.join(3)
                if b.ident is not None:
                    b.join(3)
        self.assertFalse(a.is_alive())
        self.assertFalse(b.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(trace, ["first_wdl", "first_dtz", "second_wdl"])

    def test_probe_exception_identity_and_gate_release(self) -> None:
        handle = StrictMatchTablebase()
        failure = chess.syzygy.MissingTableError("exact missing material")

        def wdl(_handle: chess.syzygy.Tablebase, _board: chess.Board) -> int:
            raise failure

        with (patch.object(chess.syzygy.Tablebase, "probe_wdl", wdl),
              patch.object(chess.syzygy.Tablebase, "probe_dtz", return_value=0)):
            with self.assertRaises(chess.syzygy.MissingTableError) as caught:
                handle.probe_wdl(FIRST.copy())
            self.assertIs(caught.exception, failure)
            errors: list[Exception] = []
            worker = threading.Thread(target=capture, args=(errors, lambda: handle.probe_dtz(FIRST.copy())))
            worker.start()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])

    def test_close_follows_pair_and_keeps_default_fd_bound(self) -> None:
        handle, gate = StrictMatchTablebase(), Gate()
        self.assertEqual(handle.max_fds, 128)
        entered, release = threading.Event(), threading.Event()
        trace: list[str] = []
        errors: list[Exception] = []

        def wdl(_handle: chess.syzygy.Tablebase, _board: chess.Board) -> int:
            entered.set()
            if not release.wait(2):
                raise TimeoutError("close release")
            trace.append("wdl")
            return 0

        def dtz(_handle: chess.syzygy.Tablebase, _board: chess.Board) -> int:
            trace.append("dtz")
            return 0

        def close(_handle: chess.syzygy.Tablebase) -> None:
            trace.append("close")

        with (patch.object(handle, "_match_probe_lock", gate),
              patch.object(chess.syzygy.Tablebase, "probe_wdl", wdl),
              patch.object(chess.syzygy.Tablebase, "probe_dtz", dtz),
              patch.object(chess.syzygy.Tablebase, "close", close)):
            a = threading.Thread(target=capture, args=(errors, lambda: handle.probe_wdl_dtz(FIRST.copy())))
            b = threading.Thread(target=capture, args=(errors, handle.close))
            try:
                a.start()
                self.assertTrue(entered.wait(2))
                b.start()
                self.assertTrue(gate.contender.wait(2))
                release.set()
            finally:
                release.set()
                a.join(3)
                if b.ident is not None:
                    b.join(3)
        self.assertFalse(a.is_alive())
        self.assertFalse(b.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(trace, ["wdl", "dtz", "close"])

    def test_serial_values_and_wdl_conversion_failure_precedes_dtz(self) -> None:
        handle = StrictMatchTablebase()
        with (patch.object(chess.syzygy.Tablebase, "probe_wdl", return_value=2),
              patch.object(chess.syzygy.Tablebase, "probe_dtz", return_value=1)):
            self.assertEqual(handle.probe_wdl_dtz(FIRST.copy()), (2, 1))
        with (patch.object(chess.syzygy.Tablebase, "probe_wdl", return_value="invalid"),
              patch.object(chess.syzygy.Tablebase, "probe_dtz", return_value=1) as dtz):
            with self.assertRaises(ValueError):
                handle.probe_wdl_dtz(FIRST.copy())
            dtz.assert_not_called()


if __name__ == "__main__":
    unittest.main()
