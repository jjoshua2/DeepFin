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
import textwrap
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


@pytest.fixture(autouse=True)
def _fresh_pool_warn_rate_limit(monkeypatch) -> None:
    """Each test starts with no pool warning issued recently."""
    monkeypatch.setitem(thread_join._pool_warn, "last", float("-inf"))
    monkeypatch.setitem(thread_join._pool_warn, "suppressed", 0)


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


@pytest.mark.skipif(sys.version_info >= (3, 13), reason=(
    "from 3.13 Thread.join already waits for the OS thread, so no cap can be hit after it; "
    "the join-race CI leg runs this on 3.10"))
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


@pytest.mark.skipif(sys.version_info >= (3, 13), reason=(
    "from 3.13 Thread.join already waits for the OS thread, so no cap can be hit after it; "
    "the join-race CI leg runs this on 3.10"))
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


class _EndedThread:
    """An ended thread whose tid now names another live task (a reused tid)."""

    def __init__(self, native_id: int, name: str = "ended") -> None:
        self.native_id = native_id
        self.name = name

    def is_alive(self) -> bool:
        return False


@pytest.fixture
def live_tid() -> Iterator[int]:
    release = threading.Event()
    other = threading.Thread(target=release.wait, daemon=True)
    other.start()
    assert other.native_id is not None
    yield other.native_id
    release.set()
    other.join(5)


def test_a_reused_tid_with_a_different_start_time_ends_the_wait_at_once(live_tid) -> None:
    ghost = _EndedThread(live_tid)
    real = thread_join.task_start_time(live_tid)
    assert real is not None
    started = time.monotonic()
    assert thread_join.await_os_thread_exit(ghost, start_time=real + 1) is True  # pyright: ignore[reportArgumentType]
    assert time.monotonic() - started < 0.05


def test_a_caller_budget_that_runs_out_is_neither_logged_nor_remembered(live_tid, caplog) -> None:
    """A tiny budget (``go movetime 1``) running out is not a cap hit: no
    warning, and a later call with more budget still waits."""
    caplog.set_level(logging.WARNING, logger=thread_join.__name__)
    ghost = _EndedThread(live_tid, name="ghost-budget")
    for _ in range(2):
        started = time.monotonic()
        assert thread_join.await_os_thread_exit(ghost, cap_s=0.05) is False  # pyright: ignore[reportArgumentType]
        elapsed = time.monotonic() - started
        assert 0.04 < elapsed < 0.5, elapsed  # waited the budget again: not memoised
    assert not [r for r in caplog.records if "ghost-budget" in r.getMessage()]


def test_the_wait_never_exceeds_the_cap_and_a_cap_hit_is_logged_once_and_remembered(
        live_tid, caplog) -> None:
    """A reused tid with no start time to compare: a 5 s caller budget still
    waits only the 1 s cap, logs once, and is never waited for again."""
    caplog.set_level(logging.WARNING, logger=thread_join.__name__)
    ghost = _EndedThread(live_tid, name="ghost-cap")
    started = time.monotonic()
    assert thread_join.await_os_thread_exit(ghost, cap_s=5.0) is False  # pyright: ignore[reportArgumentType]
    elapsed = time.monotonic() - started
    assert thread_join.OS_EXIT_WAIT_S - 0.05 < elapsed < thread_join.OS_EXIT_WAIT_S + 0.5, elapsed
    started = time.monotonic()
    assert thread_join.await_os_thread_exit(ghost, cap_s=5.0) is False  # pyright: ignore[reportArgumentType]
    assert time.monotonic() - started < 0.05
    assert sum("ghost-cap" in r.getMessage() for r in caplog.records) == 1


def test_task_start_time_is_the_threads_own_start_tick_and_stable() -> None:
    """Field 22 of /proc/self/task/<tid>/stat, in clock ticks since boot: a
    wrong field would make every wait end at once as "tid reused"."""
    release = threading.Event()
    t = threading.Thread(target=release.wait, daemon=True)
    t.start()
    try:
        assert t.native_id is not None
        first = thread_join.task_start_time(t.native_id)
        with open("/proc/uptime") as f:
            uptime_ticks = float(f.read().split()[0]) * os.sysconf("SC_CLK_TCK")
        assert first is not None
        assert abs(first - uptime_ticks) <= 10, (first, uptime_ticks)  # started just now
        time.sleep(0.05)
        assert thread_join.task_start_time(t.native_id) == first
    finally:
        release.set()
        t.join(5)


def test_a_start_time_noted_at_thread_start_is_used_after_it_ended(live_tid) -> None:
    ghost = _EndedThread(live_tid)
    real = thread_join.task_start_time(live_tid)
    assert real is not None
    thread_join._noted[ghost] = real + 1  # pyright: ignore[reportArgumentType]  # noted for the earlier task
    started = time.monotonic()
    assert thread_join.await_os_thread_exit(ghost, cap_s=5.0) is True  # pyright: ignore[reportArgumentType]
    assert time.monotonic() - started < 0.05


def test_a_thread_seen_gone_is_not_waited_for_again_even_if_its_tid_is_reused(live_tid) -> None:
    done = threading.Thread(target=lambda: None, daemon=True)
    done.start()
    done.join(5)
    assert done.native_id is not None
    assert _gone_within(_task(done), 5)
    ghost = _EndedThread(done.native_id)
    assert thread_join.await_os_thread_exit(ghost) is True  # pyright: ignore[reportArgumentType]
    ghost.native_id = live_tid  # the tid now names a live task
    started = time.monotonic()
    assert thread_join.await_os_thread_exit(ghost) is True  # pyright: ignore[reportArgumentType]
    assert time.monotonic() - started < 0.05


def test_join_sites_check_the_task_start_time_read_before_the_join(keys, monkeypatch) -> None:
    """``join_thread`` and ``os_joined`` read each live thread's start time before
    joining; if the tid then names a different task they do not wait."""
    real = thread_join.task_start_time
    after_join = threading.Event()
    def shifted(tid: int) -> int | None:
        seen = real(tid)
        return None if seen is None else seen + int(after_join.is_set())

    monkeypatch.setattr(thread_join, "task_start_time", shifted)
    go = threading.Event()  # keeps the worker alive until join_thread has read its start time

    def body() -> None:
        go.wait(5)
        _slow_tls_exit(keys, 600_000)

    worker = threading.Thread(target=body, daemon=True)
    real_join = worker.join

    def join(timeout: float | None = None) -> None:
        go.set()
        real_join(timeout)
        after_join.set()

    monkeypatch.setattr(worker, "join", join)
    worker.start()
    started = time.monotonic()
    thread_join.join_thread(worker, 5.0)
    # From 3.13 Thread.join itself waits out the destructor; only below can the
    # start-time check be what ends the wait early.
    exposed = sys.version_info < (3, 13)
    if exposed:
        assert time.monotonic() - started < 0.4  # the 0.6 s destructor was not waited for

    after_join.clear()
    pool = ThreadPoolExecutor(max_workers=1)
    real_shutdown = pool.shutdown

    def shutdown(wait: bool = True, **kw: Any) -> None:
        real_shutdown(wait=wait, **kw)
        after_join.set()

    monkeypatch.setattr(pool, "shutdown", shutdown)
    started = time.monotonic()
    with thread_join.os_joined(pool) as p:
        p.submit(_slow_tls_exit, keys, 600_000).result()
    if exposed:
        assert time.monotonic() - started < 0.4


@pytest.mark.skipif(sys.version_info >= (3, 13), reason=(
    "from 3.13 Thread.join already waits for the OS thread, so no cap can be hit after it; "
    "the join-race CI leg runs this on 3.10"))
def test_uci_cap_warning_goes_to_stderr_and_never_to_stdout(tmp_path) -> None:
    """With the engine's own logging setup, a cap hit on the UCI join path is one
    WARNING line on stderr and nothing on stdout (the protocol stream)."""
    import subprocess
    from pathlib import Path

    script = tmp_path / "uci_cap.py"
    script.write_text(textwrap.dedent('''
        import ctypes, ctypes.util, threading
        from chess_anti_engine.uci import __main__ as uci_main, wedge
        from chess_anti_engine.utils import thread_join
        uci_main._configure_logging("warning")
        thread_join.OS_EXIT_WAIT_S = 0.05
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        usleep = ctypes.cast(libc.usleep, ctypes.c_void_p).value
        def body():
            key = ctypes.c_uint()
            libc.pthread_key_create(ctypes.byref(key), ctypes.c_void_p(usleep))
            libc.pthread_setspecific(key, ctypes.c_void_p(600000))
        t = threading.Thread(target=body, name="rpg-group-3")
        t.start()
        wedge.join_unless_wedged(t, 5.0)
    '''))
    repo = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=repo)
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          env=env, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "", done.stdout
    warnings = [ln for ln in done.stderr.splitlines() if "WARNING" in ln and "rpg-group-3" in ln]
    assert len(warnings) == 1, done.stderr


def test_join_thread_deadline_bounds_the_os_thread_wait(keys, caplog) -> None:
    caplog.set_level(logging.WARNING, logger=thread_join.__name__)
    worker = threading.Thread(target=_slow_tls_exit, args=(keys, 600_000), daemon=True,
                              name="deadline-bound")
    worker.start()
    started = time.monotonic()
    thread_join.join_thread(worker, 5.0, deadline=started + 0.1)
    assert not worker.is_alive()
    if sys.version_info < (3, 13):  # from 3.13 the join itself waits out the destructor
        assert time.monotonic() - started < 0.45  # the deadline, not the 0.6 s destructor
    assert _gone_within(_task(worker), 5)


def test_os_joined_shares_one_cap_across_its_workers(keys, monkeypatch) -> None:
    monkeypatch.setattr(thread_join, "OS_EXIT_WAIT_S", 0.2)
    barrier = threading.Barrier(3)

    def body() -> None:
        barrier.wait(5)  # three workers exist, each with a 0.6 s teardown
        _slow_tls_exit(keys, 600_000)

    started = time.monotonic()
    with thread_join.os_joined(ThreadPoolExecutor(max_workers=3)) as pool:
        for f in [pool.submit(body) for _ in range(3)]:
            f.result()
    elapsed = time.monotonic() - started
    if sys.version_info < (3, 13):
        assert elapsed < 0.5, elapsed  # one 0.2 s cap, not three


def _stuck_pool_shutdown(keys: list[Any], usec: int, n: int = 2,
                         workers: list[threading.Thread] | None = None) -> float:
    barrier = threading.Barrier(n)

    def body() -> None:
        barrier.wait(5)
        if workers is not None:
            workers.append(threading.current_thread())
        _slow_tls_exit(keys, usec)

    started = time.monotonic()
    with thread_join.os_joined(ThreadPoolExecutor(max_workers=n)) as pool:
        for f in [pool.submit(body) for _ in range(n)]:
            f.result()
    return time.monotonic() - started


def _cap_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.name == thread_join.__name__]


@pytest.mark.skipif(sys.version_info >= (3, 13), reason=(
    "from 3.13 the join itself waits out the teardown; the join-race CI leg runs this on 3.10"))
def test_a_stuck_pool_teardown_costs_the_cap_once_and_logs_once(keys, caplog) -> None:
    """``os_joined`` sets its bound from the module cap: a teardown longer than
    the cap is a cap hit -- about 1 s, then exactly one warning naming the workers."""
    caplog.set_level(logging.WARNING, logger=thread_join.__name__)
    workers: list[threading.Thread] = []
    elapsed = _stuck_pool_shutdown(keys, 1_500_000, workers=workers)
    assert thread_join.OS_EXIT_WAIT_S - 0.05 < elapsed < thread_join.OS_EXIT_WAIT_S + 0.4, elapsed
    warnings = _cap_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert "2 worker thread(s)" in warnings[0], warnings
    # A cap hit is remembered like any other: never waited for again.
    started = time.monotonic()
    assert [thread_join.await_os_thread_exit(t) for t in workers] == [False, False]
    assert time.monotonic() - started < 0.05


@pytest.mark.skipif(sys.version_info >= (3, 13), reason=(
    "from 3.13 the join itself waits out the teardown; the join-race CI leg runs this on 3.10"))
def test_join_thread_with_a_full_cap_deadline_logs_a_stuck_teardown(keys, caplog) -> None:
    """A deadline that allowed the whole cap at the call: running out is a cap
    hit (warned once), even though the join itself used a little of it."""
    caplog.set_level(logging.WARNING, logger=thread_join.__name__)
    worker = threading.Thread(target=_slow_tls_exit, args=(keys, 1_500_000), daemon=True,
                              name="stuck-deadline")
    worker.start()
    started = time.monotonic()
    thread_join.join_thread(worker, 5.0, deadline=started + thread_join.OS_EXIT_WAIT_S)
    elapsed = time.monotonic() - started
    assert elapsed < thread_join.OS_EXIT_WAIT_S + 0.4, elapsed
    assert sum("stuck-deadline" in m for m in _cap_warnings(caplog)) == 1
    thread_join.join_thread(worker, 5.0, deadline=time.monotonic() + thread_join.OS_EXIT_WAIT_S)
    assert sum("stuck-deadline" in m for m in _cap_warnings(caplog)) == 1  # remembered


@pytest.mark.skipif(sys.version_info >= (3, 13), reason=(
    "from 3.13 the join itself waits out the teardown; the join-race CI leg runs this on 3.10"))
def test_pool_cap_hits_are_logged_at_a_bounded_rate(keys, monkeypatch, caplog) -> None:
    """A persistent stuck teardown in a per-batch pool: one warning per
    ``POOL_WARN_INTERVAL_S``, the next one counting the shutdowns not logged."""
    caplog.set_level(logging.WARNING, logger=thread_join.__name__)
    monkeypatch.setattr(thread_join, "OS_EXIT_WAIT_S", 0.1)
    for _ in range(3):
        _stuck_pool_shutdown(keys, 400_000)
    assert len(_cap_warnings(caplog)) == 1
    monkeypatch.setitem(thread_join._pool_warn, "last",
                        time.monotonic() - thread_join.POOL_WARN_INTERVAL_S - 1)
    _stuck_pool_shutdown(keys, 400_000)
    warnings = _cap_warnings(caplog)
    assert len(warnings) == 2, warnings
    assert "2 more pool shutdown(s)" in warnings[1], warnings
