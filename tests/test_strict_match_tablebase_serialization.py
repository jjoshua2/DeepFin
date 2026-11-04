"""Actual handle/reader/LRU lock schedules; no table files, model or GPU.

The decoding seam is synthetic. The reader count, LRU eviction and close
methods are python-chess's actual methods, not copied lock algebra. These
controls do not qualify real Syzygy file decoding or native encoding.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import chess
import chess.syzygy
import pytest

from chess_anti_engine import tablebase as subject


FIRST = chess.Board("7k/8/8/8/8/8/8/KQ6 w - - 0 1")
SECOND = chess.Board("7k/8/8/8/8/8/8/KR6 w - - 0 1")


class ObservedCondition(threading.Condition):
    def __init__(self, waiting: threading.Event) -> None:
        super().__init__()
        self.waiting = waiting

    def wait(self, timeout: float | None = None) -> bool:
        self.waiting.set()
        return super().wait(timeout)


class ObservedGate:
    """Observe a second thread attempting the real reentrant handle gate."""

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

    def __exit__(self, *_exc: object) -> None:
        self.depth -= 1
        if self.depth == 0:
            self.owner = None
        self.lock.release()


class ScheduledWdlTable(chess.syzygy.WdlTable):
    def __init__(
        self, name: str, *, entered: threading.Event, release: threading.Event,
        eviction_wait: threading.Event,
    ) -> None:
        super().__init__(str(Path("not-opened") / f"{name}.rtbw"))
        self.entered = entered
        self.release = release
        self.first = name == "KQvK"
        self.read_condition = ObservedCondition(eviction_wait)

    def _probe_wdl_table(self, board: chess.Board) -> int:
        _ = board  # Preserve the dependency's keyword-compatible override.
        # This seam replaces file decoding, not the actual counted-reader
        # wrapper, eviction, close, or their acquisition order.
        if self.first:
            self.entered.set()
            if not self.release.wait(2):
                raise TimeoutError("test did not release counted reader")
        if not self.write_lock.acquire(timeout=0.5):
            # Bound the original cycle so every test thread can join normally.
            raise TimeoutError("counted reader blocked by eviction write lock")
        try:
            return 0
        finally:
            self.write_lock.release()


def capture(errors: list[Exception], action: Callable[[], object]) -> None:
    try:
        action()
    except Exception as exc:
        errors.append(exc)


def lock_schedule(
    monkeypatch: pytest.MonkeyPatch, *, serialized: bool,
) -> tuple[list[Exception], bool]:
    handle = subject.StrictMatchTablebase() if serialized else chess.syzygy.Tablebase()
    # Deliberately small dependency schedule, not production's max_fds=128.
    handle.max_fds = 1
    gate = ObservedGate()
    if serialized:
        # Same RLock semantics; the event makes attempted contention observable
        # without changing the library reader/close or using scheduler sleeps.
        monkeypatch.setattr(handle, "_match_probe_lock", gate)
    entered, release, eviction_wait = (threading.Event() for _ in range(3))
    first = ScheduledWdlTable("KQvK", entered=entered, release=release,
                              eviction_wait=eviction_wait)
    second = ScheduledWdlTable("KRvK", entered=entered, release=release,
                               eviction_wait=eviction_wait)
    handle.wdl[chess.syzygy.calc_key(FIRST)] = first
    handle.wdl[chess.syzygy.calc_key(SECOND)] = second
    errors: list[Exception] = []
    a = threading.Thread(target=capture, args=(errors, lambda: handle.probe_wdl(FIRST.copy())))
    b = threading.Thread(target=capture, args=(errors, lambda: handle.probe_wdl(SECOND.copy())))
    try:
        a.start()
        assert entered.wait(2)
        b.start()
        if serialized:
            assert gate.contender.wait(2)
        else:
            # Actual Table.close now holds write_lock while waiting for A's
            # actual read_count. A's timed acquisition exposes the inversion.
            assert eviction_wait.wait(2)
        release.set()
    finally:
        release.set()
        a.join(3)
        if b.ident is not None:
            b.join(3)
    assert not a.is_alive()
    assert not b.is_alive()
    assert first.read_count == 0
    assert second.read_count == 0
    handle.close()
    return errors, eviction_wait.is_set()


def test_original_shared_reader_eviction_lock_order_is_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    errors, waited = lock_schedule(monkeypatch, serialized=False)
    assert waited
    assert len(errors) == 1
    assert str(errors[0]) == "counted reader blocked by eviction write lock"


def test_strict_handle_serializes_actual_reader_and_lru_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    errors, waited = lock_schedule(monkeypatch, serialized=True)
    assert errors == []
    assert not waited


def test_match_wdl_dtz_pair_is_atomic_and_reentrant(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = subject.StrictMatchTablebase()
    gate = ObservedGate()
    monkeypatch.setattr(handle, "_match_probe_lock", gate)
    entered, release, competitor_started = (threading.Event() for _ in range(3))
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

    monkeypatch.setattr(chess.syzygy.Tablebase, "probe_wdl", wdl)
    monkeypatch.setattr(chess.syzygy.Tablebase, "probe_dtz", dtz)

    def pair() -> None:
        assert subject.rule50_match_status(FIRST.copy(), handle) == 1

    def competitor() -> None:
        competitor_started.set()
        assert handle.probe_wdl(SECOND.copy()) == 2

    a = threading.Thread(target=capture, args=(errors, pair))
    b = threading.Thread(target=capture, args=(errors, competitor))
    try:
        a.start()
        assert entered.wait(2)
        b.start()
        assert competitor_started.wait(2)
        assert gate.contender.wait(2)
        release.set()
    finally:
        release.set()
        a.join(3)
        if b.ident is not None:
            b.join(3)
    assert not a.is_alive()
    assert not b.is_alive()
    assert errors == []
    assert trace == ["first_wdl", "first_dtz", "second_wdl"]


def test_probe_exception_releases_shared_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = subject.StrictMatchTablebase()
    failure = chess.syzygy.MissingTableError("exact failure")

    def wdl(_handle: chess.syzygy.Tablebase, _board: chess.Board) -> int:
        raise failure

    def dtz(_handle: chess.syzygy.Tablebase, _board: chess.Board) -> int:
        return 0

    monkeypatch.setattr(chess.syzygy.Tablebase, "probe_wdl", wdl)
    monkeypatch.setattr(chess.syzygy.Tablebase, "probe_dtz", dtz)
    with pytest.raises(chess.syzygy.MissingTableError) as caught:
        handle.probe_wdl(FIRST.copy())
    assert caught.value is failure
    errors: list[Exception] = []
    worker = threading.Thread(target=capture, args=(errors, lambda: handle.probe_dtz(FIRST.copy())))
    worker.start()
    worker.join(3)
    assert not worker.is_alive()
    assert errors == []


def test_strict_open_failure_closes_serialized_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    opened: list[chess.syzygy.Tablebase] = []
    closed: list[chess.syzygy.Tablebase] = []

    def add(handle: chess.syzygy.Tablebase, _path: str) -> int:
        opened.append(handle)
        raise OSError("unavailable component")

    def close(handle: chess.syzygy.Tablebase) -> None:
        closed.append(handle)

    monkeypatch.setattr(chess.syzygy.Tablebase, "add_directory", add)
    monkeypatch.setattr(chess.syzygy.Tablebase, "close", close)
    with pytest.raises(subject.MatchTablebaseError, match="could not open match Syzygy component"):
        subject.open_strict_match_tablebase(str(tmp_path))
    assert len(opened) == 1
    assert isinstance(opened[0], subject.StrictMatchTablebase)
    assert closed == opened


def test_strict_open_success_retains_original_capacity_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def add(handle: chess.syzygy.Tablebase, _path: str) -> int:
        handle.wdl["KQBNRvK"] = chess.syzygy.WdlTable("not-opened/KQBNRvK.rtbw")
        handle.dtz["KQBNRvK"] = chess.syzygy.DtzTable("not-opened/KQBNRvK.rtbz")
        return 2

    monkeypatch.setattr(chess.syzygy.Tablebase, "add_directory", add)
    handle = subject.open_strict_match_tablebase(str(tmp_path))
    assert isinstance(handle, subject.StrictMatchTablebase)
    assert handle.max_fds == 128
    handle.close()
    assert handle.wdl == {}
    assert handle.dtz == {}


def test_close_waits_for_complete_pair_and_preserves_default_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = subject.StrictMatchTablebase()
    gate = ObservedGate()
    monkeypatch.setattr(handle, "_match_probe_lock", gate)
    assert handle.max_fds == 128
    entered, release, closing = (threading.Event() for _ in range(3))
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

    def close_after_started() -> None:
        closing.set()
        handle.close()

    monkeypatch.setattr(chess.syzygy.Tablebase, "probe_wdl", wdl)
    monkeypatch.setattr(chess.syzygy.Tablebase, "probe_dtz", dtz)
    monkeypatch.setattr(chess.syzygy.Tablebase, "close", close)
    a = threading.Thread(target=capture, args=(errors, lambda: handle.probe_wdl_dtz(FIRST.copy())))
    b = threading.Thread(target=capture, args=(errors, close_after_started))
    try:
        a.start()
        assert entered.wait(2)
        b.start()
        assert closing.wait(2)
        assert gate.contender.wait(2)
        release.set()
    finally:
        release.set()
        a.join(3)
        if b.ident is not None:
            b.join(3)
    assert not a.is_alive()
    assert not b.is_alive()
    assert errors == []
    assert trace == ["wdl", "dtz", "close"]
