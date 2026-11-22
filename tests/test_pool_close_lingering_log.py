"""A pool `close()` whose bounded 2 s join runs out says so, once.

Before, `wedge.join_unless_wedged` returned silently on timeout: a worker
still running after its pool closed (holding its evaluator, and an unjoined
thread at interpreter exit) was visible to no one. The close itself is
unchanged: the thread is still left running (daemon).

Each pool is built with ``object.__new__`` and only the fields its `close()`
reads, plus one real thread that ignores the shutdown for longer than the join.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from chess_anti_engine.uci import wedge
from chess_anti_engine.uci.multi_gpu_pucv_pool import MultiGpuPucvPool
from chess_anti_engine.uci.root_parallel_gumbel import RootParallelGumbelPool
from chess_anti_engine.uci.walker_pool import WalkerPool

_LINGER_S = 2.4  # beyond close()'s 2 s join


def _rpg(th: threading.Thread) -> Any:
    pool = object.__new__(RootParallelGumbelPool)
    pool._shutdown = threading.Event()
    pool._threads = [th]
    pool._started_threads = [th]
    pool._work_q = queue.SimpleQueue()
    pool._abandoned = set()
    pool._phase_done = threading.Event()
    return pool


def _walkers(th: threading.Thread) -> Any:
    pool = object.__new__(WalkerPool)
    pool._shutdown = threading.Event()
    pool._wakes = [threading.Event()]
    pool._started_threads = [th]
    return pool


def _multi_gpu(th: threading.Thread) -> Any:
    pool = object.__new__(MultiGpuPucvPool)
    pool._shutdown = threading.Event()
    pool._wakes = [threading.Event()]
    pool._threads = [th]
    pool._abandoned = [False]
    pool._restorable = [False]
    return pool


@pytest.mark.parametrize(("make", "name"), [
    (_rpg, "root-parallel Gumbel pool"),
    (_walkers, "walker pool"),
    (_multi_gpu, "multi-GPU PUCT pool"),
], ids=["rpg", "walkers", "multi_gpu"])
def test_a_worker_outliving_close_is_logged_once(
    caplog: pytest.LogCaptureFixture, make: Callable[[threading.Thread], Any], name: str,
) -> None:
    release = threading.Event()
    th = threading.Thread(target=release.wait, args=(_LINGER_S,), name="pool-worker-under-test",
                          daemon=True)
    th.start()
    pool = make(th)
    try:
        with caplog.at_level(logging.WARNING):
            pool.close()
            pool.close()  # idempotent: no second line
        assert th.is_alive()  # unchanged: left running, not killed
        lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert lines == [
            f"{name} close: pool-worker-under-test still running after the 2s join; left running"
        ]
    finally:
        release.set()
        th.join(5)


def test_a_worker_that_ends_in_time_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    th = threading.Thread(target=lambda: None, name="pool-worker-under-test", daemon=True)
    th.start()
    with caplog.at_level(logging.WARNING):
        _walkers(th).close()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_join_unless_wedged_reports_ended_parked_or_still_running() -> None:
    done = threading.Thread(target=lambda: None)
    done.start()
    assert wedge.join_unless_wedged(done, 1.0) is True
    release = threading.Event()
    busy = threading.Thread(target=release.wait, args=(5.0,), daemon=True)
    busy.start()
    try:
        assert wedge.join_unless_wedged(busy, 0.05) is False
    finally:
        release.set()
        busy.join(5)
    # After a device fault a worker that parks counts as retired.
    wedge.note_device_fault("test fault")
    parked = threading.Thread(target=wedge.parking(lambda: None), daemon=True)
    parked.start()
    assert wedge.join_unless_wedged(parked, 2.0) is True
    assert parked in wedge.parked_threads()


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_after_a_device_fault_a_worker_that_neither_parks_nor_ends_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The production case: an RPG startup failure notes a fault before close().
    wedge.note_device_fault("test fault before close")
    release = threading.Event()
    th = threading.Thread(target=release.wait, args=(_LINGER_S,), name="busy-worker",
                          daemon=True)  # not wrapped in wedge.parking: never parks
    th.start()
    try:
        assert wedge.join_unless_wedged(th, 0.05) is False
        with caplog.at_level(logging.WARNING):
            _walkers(th).close()
        assert _warnings(caplog) == [
            "walker pool close: busy-worker still running after the 2s join; left running"
        ]
    finally:
        release.set()
        th.join(5)


def test_two_lingering_workers_get_one_line_naming_both(
    caplog: pytest.LogCaptureFixture,
) -> None:
    release = threading.Event()
    threads = [threading.Thread(target=release.wait, args=(_LINGER_S + 2.0,), name=f"w-{k}",
                                daemon=True) for k in range(2)]
    for th in threads:
        th.start()
    pool = _walkers(threads[0])
    pool._wakes = [threading.Event(), threading.Event()]
    pool._started_threads = threads
    try:
        with caplog.at_level(logging.WARNING):
            pool.close()
        assert _warnings(caplog) == [
            "walker pool close: w-0, w-1 still running after the 2s join; left running"
        ]
    finally:
        release.set()
        for th in threads:
            th.join(5)


def test_a_worker_that_parked_during_a_no_fault_join_is_not_called_running(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # No fault when close() starts, so the join is a plain Thread.join; the
    # worker then latches a fault and parks. The join spends its 2 s and
    # returns False (alive), but the thread is parked, not running.
    def work() -> None:
        time.sleep(0.2)
        wedge.note_device_fault("fault latched during the join")

    th = threading.Thread(target=wedge.parking(work), name="parks-mid-join", daemon=True)
    th.start()
    with caplog.at_level(logging.WARNING):
        _walkers(th).close()
    assert th in wedge.parked_threads()
    assert _warnings(caplog) == []
