"""The test-side thread tagger (`tests/_thread_origin.py`) must never break
`Thread.start`: an exception while recording a thread's origin is swallowed
and the real `start` still runs."""
from __future__ import annotations

import threading
import traceback

import pytest

from tests import _thread_origin

_thread_origin.install()


class _RaisingOrigins:
    def __setitem__(self, _key: object, _value: object) -> None:
        raise TypeError("recorder failure")

    def get(self, _key: object, default: str) -> str:
        return default


def test_a_recorder_failure_does_not_stop_the_thread_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_thread_origin, "_ORIGIN", _RaisingOrigins())
    ran = threading.Event()
    th = threading.Thread(target=ran.set, daemon=True)
    th.start()
    th.join(5)
    assert ran.is_set()


class _UnhashableThread(threading.Thread):
    __hash__ = None  # pyright: ignore[reportAssignmentType]


def test_an_unhashable_thread_subclass_reaches_the_real_start() -> None:
    """The reviewer's case. CPython's own `Thread.start` hashes the thread
    (`_limbo[self]`), and 3.10's `Thread.__init__` already does (`_dangling`),
    so the subclass cannot run on CPython at all; what the tagger owes is to get
    out of the way: `start` must reach the stdlib's own `start`."""
    try:
        th = _UnhashableThread(target=lambda: None, daemon=True)
    except TypeError:
        pytest.skip("this Python's Thread.__init__ already rejects an unhashable thread")
    try:
        th.start()
    except TypeError as exc:
        frames = traceback.extract_tb(exc.__traceback__)
        assert any(f.name == "start" and f.filename == threading.__file__ for f in frames), (
            "the tagger raised before the real Thread.start ran"
        )
    else:
        th.join(5)


def test_an_ordinary_thread_is_tagged_with_this_test() -> None:
    th = threading.Thread(target=lambda: None)
    th.start()
    th.join(5)
    assert "test_an_ordinary_thread_is_tagged_with_this_test" in _thread_origin.origin(th)
