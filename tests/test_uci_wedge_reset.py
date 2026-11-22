"""``wedge.reset_session_for_tests`` returns only once the parked threads it
released have ended (CI #1480: a previous test's released ``rpg-group-*``
threads were still dying when the next test scanned ``threading.enumerate()``
for helpers that had neither ended nor parked)."""
from __future__ import annotations

import threading
import time

import pytest

from chess_anti_engine.uci import wedge


def _parked_thread(after_release_s: float) -> threading.Thread:
    """A worker that parks at its end (a device fault is latched) and, once
    released, takes ``after_release_s`` to die: the slow tail of a real one."""
    def run() -> None:
        wedge.parking(lambda: None)()
        time.sleep(after_release_s)

    thread = threading.Thread(target=run, name="rpg-group-test", daemon=True)
    wedge.note_device_fault("test fault")
    thread.start()
    deadline = time.monotonic() + 5.0
    while thread not in wedge.parked_threads():
        assert time.monotonic() < deadline, "the worker never parked"
        time.sleep(0.01)
    return thread


def test_reset_returns_only_after_the_released_threads_have_ended() -> None:
    thread = _parked_thread(after_release_s=0.5)
    wedge.reset_session_for_tests()
    assert not thread.is_alive()
    assert wedge.device_fault() is None
    assert not wedge.parked_threads()


def test_a_released_thread_that_does_not_end_is_named() -> None:
    thread = _parked_thread(after_release_s=1.5)
    with pytest.raises(RuntimeError, match="rpg-group-test"):
        wedge.reset_session_for_tests(released_join_s=0.2)
    thread.join(5.0)
    assert not thread.is_alive()
