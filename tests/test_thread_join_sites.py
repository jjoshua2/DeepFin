"""Every join of a thread that ran torch work before teardown waits for the OS thread.

Each site below joins a thread that ran evaluator / torch work and is followed
by evaluator teardown or interpreter exit. ``slow_thread_exit`` makes every
thread started during the test outlive its Python state by ~0.15 s (a pthread
TLS destructor, the stand-in for torch's ``thread_local`` teardown), so on
CPython 3.10-3.12 a plain ``Thread.join`` at the site returns with
``/proc/self/task/<tid>`` still present and the assertion fails. On 3.13+
``Thread.join`` joins the OS thread itself and these pass either way: the
``join-race-py310`` CI leg is what guards them.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import pytest

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/task"),
    pytest.mark.usefixtures("slow_thread_exit"),
]

_LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
_USLEEP = ctypes.cast(_LIBC.usleep, ctypes.c_void_p).value


@pytest.fixture
def slow_thread_exit(monkeypatch) -> Iterator[list[threading.Thread]]:
    """Threads started during the test run a 0.15 s TLS destructor after their
    Python state is gone. Yields the list of those threads."""
    keys: list[Any] = []
    started: list[threading.Thread] = []
    real_run = threading.Thread.run

    def run(self: threading.Thread) -> None:
        started.append(self)
        try:
            real_run(self)
        finally:
            key = ctypes.c_uint()
            if _LIBC.pthread_key_create(ctypes.byref(key), ctypes.c_void_p(_USLEEP)) == 0:
                keys.append(key)
                _LIBC.pthread_setspecific(key, ctypes.c_void_p(150_000))

    monkeypatch.setattr(threading.Thread, "run", run)
    yield started
    monkeypatch.undo()
    end = time.monotonic() + 5
    while any(os.path.exists(f"/proc/self/task/{t.native_id}") for t in started if t.native_id):
        if time.monotonic() > end:
            break
        time.sleep(0.005)
    for key in keys:
        _LIBC.pthread_key_delete(key)


def _assert_os_thread_gone(thread: threading.Thread) -> None:
    assert thread.native_id is not None
    assert not thread.is_alive()
    assert not os.path.exists(f"/proc/self/task/{thread.native_id}"), (
        f"{thread.name}: join returned while its OS thread was still running")


def test_threaded_dispatcher_shutdown() -> None:
    import numpy as np

    from chess_anti_engine.inference_threaded import ThreadedDispatcher
    from chess_anti_engine.model import ModelConfig, build_model

    disp = ThreadedDispatcher(build_model(ModelConfig(kind="tiny")).eval(), device="cpu",
                              max_batch=8, target_batch=1, batch_wait_ms=0.0)
    disp.evaluate_encoded(np.zeros((1, 175, 8, 8), dtype=np.float32))
    disp.shutdown(5.0)
    _assert_os_thread_gone(disp._thread)


def test_teacher_dispatcher_close() -> None:
    from chess_anti_engine.teacher_dispatch import TeacherDispatcher

    disp = TeacherDispatcher[int, int](lambda rows: [r * 2 for r in rows], target_rows=1, max_rows=4,
                                       batch_wait_ms=0.0)
    assert disp.submit("k", (1, 2)).result(5) == (2, 4)
    disp.close(timeout=5.0)
    _assert_os_thread_gone(disp._thread)


def test_async_test_eval_shutdown() -> None:
    from chess_anti_engine.train.async_eval import AsyncTestEval

    ev = AsyncTestEval()
    thread = threading.Thread(target=ev._work_q.get, daemon=True)  # consumes the sentinel
    thread.start()
    ev._thread = thread
    ev.shutdown(timeout=5.0)
    _assert_os_thread_gone(thread)


def test_worker_model_watch_stop() -> None:
    from chess_anti_engine.worker import WorkerSession

    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, daemon=True)
    thread.start()
    fake = SimpleNamespace(_model_watch_stop=stop, _model_watch_lock=threading.Lock(),
                           _selfplay_session_active=True, _model_watch_thread=thread,
                           log=SimpleNamespace(warning=print))
    assert WorkerSession._stop_model_watch_thread(fake, timeout_s=5.0) is True  # pyright: ignore[reportArgumentType]
    _assert_os_thread_gone(thread)


def test_uci_drain_builder() -> None:
    from chess_anti_engine.uci import __main__ as uci_main

    builder = threading.Thread(target=time.sleep, args=(0.05,), daemon=True)
    builder.start()
    uci_main._drain_builder(builder, 0)
    _assert_os_thread_gone(builder)


def test_uci_drain_builder_that_already_ended() -> None:
    from chess_anti_engine.uci import __main__ as uci_main

    builder = threading.Thread(target=lambda: None, daemon=True)
    builder.start()
    while builder.is_alive():
        time.sleep(0.001)
    uci_main._drain_builder(builder, 0)
    _assert_os_thread_gone(builder)


def test_uci_parallel_model_load(slow_thread_exit, monkeypatch) -> None:
    from chess_anti_engine.uci import __main__ as uci_main

    monkeypatch.setattr(uci_main, "load_model_from_checkpoint", lambda ckpt, device: (ckpt, device))
    assert uci_main._load_models("ck", ["cpu", "cpu"]) == [("ck", "cpu"), ("ck", "cpu")]
    workers = [t for t in slow_thread_exit if t.name.startswith("ThreadPoolExecutor")]
    assert workers
    for t in workers:
        _assert_os_thread_gone(t)


def test_aot_package_load(slow_thread_exit, monkeypatch, tmp_path: Path) -> None:
    from chess_anti_engine import inference

    for b in (1, 2):
        (tmp_path / inference.aot_package_filename(b)).write_bytes(b"")
    monkeypatch.setattr(inference, "_aoti_load_package", lambda path: path)
    assert sorted(inference.load_aot_packages(tmp_path, buckets=(1, 2))) == [1, 2]
    workers = [t for t in slow_thread_exit if t.name.startswith("ThreadPoolExecutor")]
    assert workers
    for t in workers:
        _assert_os_thread_gone(t)


def _engine_like(**extra: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "_search_thread": None, "_cancel_event": threading.Event(), "_stop_event": threading.Event(),
        "_ponder_release": threading.Event(), "_warmup_done": threading.Event(),
        "_orphan_threads": [], "_state_lock": threading.Lock(), "_live_go": None}
    fields.update(extra)
    return SimpleNamespace(**fields)


def test_uci_search_thread_join_at_stop_or_quit() -> None:
    from chess_anti_engine.uci.engine import Engine

    fake = _engine_like()
    fake._warmup_done.set()
    search = threading.Thread(target=fake._stop_event.wait, daemon=True)
    search.start()
    fake._search_thread = search
    Engine._wait_for_search(fake, timeout_s=5.0)  # pyright: ignore[reportArgumentType]
    _assert_os_thread_gone(search)


def test_uci_search_thread_that_already_ended_before_quit() -> None:
    """``quit`` right after ``bestmove``: the search thread has just ended on its
    own, so no join runs, but its OS thread may still be tearing down torch."""
    from chess_anti_engine.uci.engine import Engine

    fake = _engine_like()
    fake._warmup_done.set()
    search = threading.Thread(target=lambda: None, daemon=True)
    search.start()
    while search.is_alive():
        time.sleep(0.0005)
    fake._search_thread = search
    Engine._wait_for_search(fake, timeout_s=5.0)  # pyright: ignore[reportArgumentType]
    _assert_os_thread_gone(search)


def test_uci_orphan_join() -> None:
    from chess_anti_engine.uci.engine import Engine

    release = threading.Event()
    orphan = threading.Thread(target=release.wait, daemon=True)
    orphan.start()
    fake = _engine_like(_orphan_threads=[orphan])
    fake._warmup_done.set()
    release.set()
    Engine._wait_for_search(fake, timeout_s=5.0)  # pyright: ignore[reportArgumentType]
    _assert_os_thread_gone(orphan)


def test_uci_fresh_eval_helper_join(slow_thread_exit) -> None:
    from chess_anti_engine.uci.engine import Engine

    fake = _engine_like(_worker=SimpleNamespace(
        fresh_root_policy_bestmove=lambda *_a, **_k: "e2e4"))
    assert Engine._bounded_fresh_eval(fake, chess.Board(), (), 5.0) == "e2e4"  # pyright: ignore[reportArgumentType]
    helpers = [t for t in slow_thread_exit if t.name == "uci-fresh-eval"]
    assert len(helpers) == 1
    _assert_os_thread_gone(helpers[0])


def test_uci_deadline_watchdog_close() -> None:
    from chess_anti_engine.uci.engine import _DeadlineWatchdog, _WatchdogEntry

    polled = threading.Event()

    def poll(_entry: _WatchdogEntry) -> None:
        polled.set()

    dog = _DeadlineWatchdog(poll)
    now = time.monotonic()
    dog.arm(_WatchdogEntry(gen=1, check_at=now, deadline_at=now + 1, last_moment_at=None,
                           budget_ms=None, board=chess.Board(), searchmoves=(),
                           thread=threading.current_thread(), stops=(), started_at=now))
    assert polled.wait(5)
    dog.close(5.0)
    assert dog._thread is not None
    _assert_os_thread_gone(dog._thread)
