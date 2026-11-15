"""The shared OS-thread join (``chess_anti_engine.utils.thread_join``).

On CPython 3.10-3.12 ``Thread.join`` returns once the Python thread state is
gone while the OS thread still runs its C/C++ ``thread_local`` destructors
(torch's); a caller that then closes evaluators or exits races them (PR #1333).
These tests model the destructor with a pthread TLS key whose destructor is
libc ``usleep``: no torch needed, so the CI leg on an exposed interpreter can
run them with the bare package.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from chess_anti_engine.utils import thread_join

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/task")

_LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
_USLEEP = ctypes.cast(_LIBC.usleep, ctypes.c_void_p).value


@pytest.fixture
def keys() -> Iterator[list[Any]]:
    created: list[Any] = []
    yield created
    for key in created:
        _LIBC.pthread_key_delete(key)


def _slow_tls_exit(keys: list[Any], usec: int) -> None:
    """Thread body: register a TLS value whose destructor sleeps ``usec`` --
    run by the C runtime after Python considers the thread ended."""
    key = ctypes.c_uint()
    assert _LIBC.pthread_key_create(ctypes.byref(key), ctypes.c_void_p(_USLEEP)) == 0
    keys.append(key)
    assert _LIBC.pthread_setspecific(key, ctypes.c_void_p(usec)) == 0


def _task(thread: threading.Thread) -> str:
    return f"/proc/self/task/{thread.native_id}"


def _gone_within(path: str, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while os.path.exists(path):
        if time.monotonic() > end:
            return False
        time.sleep(0.001)
    return True


def test_join_thread_returns_only_after_the_os_thread_is_gone(keys) -> None:
    control = threading.Thread(target=_slow_tls_exit, args=(keys, 200_000), daemon=True)
    control.start()
    control.join(5)
    assert not control.is_alive()
    if sys.version_info < (3, 13):
        # The mechanism: a plain join returns under the destructor. (From 3.13
        # ``Thread.join`` joins the OS thread itself; only the check below runs.)
        assert os.path.exists(_task(control))
    assert _gone_within(_task(control), 5)

    worker = threading.Thread(target=_slow_tls_exit, args=(keys, 200_000), daemon=True)
    worker.start()
    thread_join.join_thread(worker, 5.0)
    assert not worker.is_alive()
    assert not os.path.exists(_task(worker))


def test_join_thread_keeps_the_join_timeout_for_a_thread_that_does_not_end() -> None:
    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    worker.start()
    try:
        started = time.monotonic()
        thread_join.join_thread(worker, 0.1)
        assert worker.is_alive()
        assert time.monotonic() - started < 1.0
    finally:
        release.set()
        worker.join(5)


def test_os_joined_waits_for_every_executor_worker(keys) -> None:
    with thread_join.os_joined(ThreadPoolExecutor(max_workers=2)) as pool:
        barrier = threading.Barrier(2)

        def body() -> threading.Thread:
            barrier.wait(5)  # both workers exist, so both run a slow exit
            _slow_tls_exit(keys, 200_000)
            return threading.current_thread()

        threads = [f.result() for f in [pool.submit(body), pool.submit(body)]]
    assert len({t.native_id for t in threads}) == 2
    for t in threads:
        assert not os.path.exists(_task(t))


def test_os_joined_waits_on_the_exception_path_too(keys) -> None:
    seen: list[threading.Thread] = []

    def body() -> None:
        seen.append(threading.current_thread())
        _slow_tls_exit(keys, 200_000)

    def run() -> None:
        with thread_join.os_joined(ThreadPoolExecutor(max_workers=1)) as pool:
            pool.submit(body).result()
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run()
    assert seen
    assert not os.path.exists(_task(seen[0]))


def test_a_cap_hit_logs_once_per_thread_and_carries_on(keys, monkeypatch, caplog) -> None:
    monkeypatch.setattr(thread_join, "OS_EXIT_WAIT_S", 0.05)
    caplog.set_level(logging.WARNING, logger=thread_join.__name__)
    slow = [threading.Thread(target=_slow_tls_exit, args=(keys, 600_000), daemon=True,
                             name=f"slow-exit-{k}") for k in range(2)]
    for t in slow:
        t.start()
        t.join(5)
    started = time.monotonic()
    assert thread_join.await_os_thread_exit(slow[0]) is False
    assert thread_join.await_os_thread_exit(slow[0]) is False  # again: no second warning
    thread_join.join_thread(slow[1])
    assert time.monotonic() - started < 0.5  # bounded by the cap, not the destructor
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage().split("'")[1] for r in warnings] == ["slow-exit-0", "slow-exit-1"], warnings
    for t in slow:
        assert _gone_within(_task(t), 5)


def test_wedge_join_uses_the_shared_wait_and_warns_on_a_cap_hit(keys, monkeypatch, caplog) -> None:
    """The UCI path (``wedge.join_unless_wedged``) goes through the same helper:
    a cap hit is a WARNING on the logging path (stderr), never stdout."""
    from chess_anti_engine.uci import wedge

    monkeypatch.setattr(thread_join, "OS_EXIT_WAIT_S", 0.05)
    caplog.set_level(logging.WARNING, logger=thread_join.__name__)
    worker = threading.Thread(target=_slow_tls_exit, args=(keys, 600_000), daemon=True,
                              name="rpg-group-7")
    worker.start()
    wedge.join_unless_wedged(worker, 5.0)
    assert not worker.is_alive()
    assert any("rpg-group-7" in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING), caplog.records
    assert _gone_within(_task(worker), 5)


def test_await_is_a_no_op_for_an_unstarted_or_running_thread() -> None:
    assert thread_join.await_os_thread_exit(threading.Thread(target=lambda: None)) is True
    release = threading.Event()
    running = threading.Thread(target=release.wait, daemon=True)
    running.start()
    try:
        started = time.monotonic()
        assert thread_join.await_os_thread_exit(running) is True
        assert time.monotonic() - started < 0.05
    finally:
        release.set()
        running.join(5)
