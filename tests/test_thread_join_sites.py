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
        "_orphan_threads": [], "_exiting_helpers": [], "_state_lock": threading.Lock(),
        "_live_go": None}
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
    """The at-once answer does not wait out the helper's 0.15 s teardown past
    its 20 ms budget, and the still-exiting helper is recorded, keeps the
    worker busy, and is waited for by the next `_wait_for_search` (quit)."""
    from chess_anti_engine.uci.engine import Engine

    fake = _engine_like(_worker=SimpleNamespace(
        fresh_root_policy_bestmove=lambda *_a, **_k: "e2e4"))
    fake._warmup_done.set()
    t0 = time.monotonic()
    assert Engine._bounded_fresh_eval(fake, chess.Board(), (), 0.02) == "e2e4"  # pyright: ignore[reportArgumentType]
    elapsed = time.monotonic() - t0
    helpers = [t for t in slow_thread_exit if t.name == "uci-fresh-eval"]
    assert len(helpers) == 1
    assert elapsed < 0.12, f"waited {elapsed:.3f} s for the helper's teardown"
    assert [h.thread for h in fake._exiting_helpers] == helpers, (
        "a still-exiting helper must be recorded")
    assert not Engine._search_free_locked(fake)  # pyright: ignore[reportArgumentType]
    Engine._wait_for_search(fake, timeout_s=5.0)  # pyright: ignore[reportArgumentType]
    _assert_os_thread_gone(helpers[0])
    assert fake._exiting_helpers == []


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


def test_worker_selfplay_thread_pool(slow_thread_exit, monkeypatch) -> None:
    """``_run_selfplay_threaded``'s per-batch pool (``os_joined``): every
    selfplay thread's OS thread is gone when the session returns."""
    import numpy as np

    from chess_anti_engine import worker as worker_module
    from chess_anti_engine.worker import WorkerSession

    calls: list[str] = []
    both = threading.Barrier(2)  # one game per thread: the pool must start two

    def fake_play_batch(*_a: Any, **_k: Any) -> tuple[None, str]:
        both.wait(5)
        calls.append(threading.current_thread().name)
        return None, "stats"

    monkeypatch.setattr(worker_module, "play_batch", fake_play_batch)
    fake = SimpleNamespace(
        _selfplay_state_count=lambda games: 2, rng=np.random.default_rng(0), device="cpu",
        _stop_selfplay=False, log=SimpleNamespace(exception=print),
        _on_completed_game=None, _record_selfplay_phase_timing=None, _check_model_update=None,
        _register_live_state=None, _resume_inflight_enabled=False, _suspend_inflight_games=None,
        _stop_fn=None, _pause_fn=None, _clear_live_states=lambda: None,
        _aggregate_thread_stats=lambda stats: stats,
    )
    out = WorkerSession._run_selfplay_threaded(
        fake, games_per_batch=2, sf=None, eval_=None, cfgs={})  # pyright: ignore[reportArgumentType]
    assert out == ["stats", "stats"]
    workers = [t for t in slow_thread_exit if t.name.startswith("ThreadPoolExecutor")]
    assert len(workers) == 2
    assert sorted(calls) == sorted(t.name for t in workers)
    for t in workers:
        _assert_os_thread_gone(t)


def test_uci_ended_search_wait_with_a_zero_budget_does_not_give_up_for_good(caplog) -> None:
    """``go movetime 1``: the previous search's OS thread is still in its
    teardown and the go budget is ~0. The wait returns at once without a
    warning, and a later call with a real budget (quit) still waits it out."""
    import logging

    from chess_anti_engine.uci.engine import Engine
    from chess_anti_engine.utils import thread_join

    caplog.set_level(logging.WARNING, logger=thread_join.__name__)
    search = threading.Thread(target=lambda: None, daemon=True, name="search-zero-budget")
    search.start()
    while search.is_alive():
        time.sleep(0.0005)
    fake = _engine_like(_search_thread=search)
    fake._warmup_done.set()
    started = time.monotonic()
    Engine._wait_for_search(fake, timeout_s=0.0)  # pyright: ignore[reportArgumentType]
    assert time.monotonic() - started < 0.05
    if sys.version_info < (3, 13):
        assert os.path.exists(f"/proc/self/task/{search.native_id}")  # still tearing down
    assert not [r for r in caplog.records if "search-zero-budget" in r.getMessage()]
    fake._search_thread = search
    Engine._wait_for_search(fake, timeout_s=5.0)  # pyright: ignore[reportArgumentType]
    _assert_os_thread_gone(search)


def test_uci_go_notes_the_search_threads_start_time(monkeypatch) -> None:
    """The ``go`` path records the search thread's task start time, so the
    ended-search wait can tell a reused tid (costs nothing) from a linger."""
    from unittest.mock import MagicMock

    import chess_anti_engine.uci.engine as engine_module
    from chess_anti_engine.uci.engine import Engine, EngineOptions
    from chess_anti_engine.uci.protocol import parse_command
    from chess_anti_engine.uci.search import SearchResult
    from chess_anti_engine.utils import thread_join

    lines: list[str] = []
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(
        stdout=SimpleNamespace(write=lambda s: lines.append(s) or len(s), flush=lambda: None),
        stderr=sys.stderr))
    entered, release = threading.Event(), threading.Event()

    def run(board: chess.Board, **_kw: Any) -> SearchResult:
        entered.set()
        assert release.wait(10)
        move = next(iter(board.legal_moves)).uci()
        return SearchResult(move, None, 8, (move,), 0)

    worker = MagicMock()
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    worker.realized_search_path.return_value = "gumbel"
    worker.run.side_effect = run
    engine = Engine(worker, options=EngineOptions())
    try:
        engine.dispatch(parse_command("position startpos"))
        engine.dispatch(parse_command("go infinite"))
        assert entered.wait(5)
        thread = engine._search_thread
        assert thread is not None
        assert thread.native_id is not None
        assert thread_join._noted.get(thread) == thread_join.task_start_time(thread.native_id)
    finally:
        release.set()
        engine.dispatch(parse_command("stop"))
        engine.close()


class _GhostThread:
    """An ended search thread whose tid now names another live task."""

    def __init__(self, native_id: int) -> None:
        self.native_id = native_id
        self.name = "ghost-search"

    def is_alive(self) -> bool:
        return False


def test_uci_wait_for_an_ended_search_is_bounded_and_honours_a_noted_start_time() -> None:
    """``_wait_for_search`` runs on every go/stop/setoption: for an ended search
    thread whose tid was reused, the OS-thread wait spends at most the call's own
    budget, once; later calls do not wait again."""
    from chess_anti_engine.uci.engine import Engine

    release = threading.Event()
    other = threading.Thread(target=release.wait, daemon=True)
    other.start()
    try:
        assert other.native_id is not None
        ghost = _GhostThread(other.native_id)
        fake = _engine_like(_search_thread=ghost)
        fake._warmup_done.set()
        started = time.monotonic()
        Engine._wait_for_search(fake, timeout_s=0.1)  # pyright: ignore[reportArgumentType]
        assert time.monotonic() - started < 0.5  # the 0.1 s budget, not the 1 s cap
        fake._search_thread = ghost  # stop / setoption default to a 30 s budget
        started = time.monotonic()
        Engine._wait_for_search(fake, timeout_s=30.0)  # pyright: ignore[reportArgumentType]
        assert time.monotonic() - started < 1.5  # clamped to the 1 s cap
        fake._search_thread = ghost
        started = time.monotonic()
        Engine._wait_for_search(fake, timeout_s=30.0)  # pyright: ignore[reportArgumentType]
        assert time.monotonic() - started < 0.05  # a cap hit is not repeated

        # With the start time noted at its start (the go path), a reused tid costs nothing.
        from chess_anti_engine.utils import thread_join
        noted = _GhostThread(other.native_id)
        real = thread_join.task_start_time(other.native_id)
        assert real is not None
        thread_join._noted[noted] = real - 1  # pyright: ignore[reportArgumentType]
        fake._search_thread = noted
        started = time.monotonic()
        Engine._wait_for_search(fake, timeout_s=30.0)  # pyright: ignore[reportArgumentType]
        assert time.monotonic() - started < 0.05
    finally:
        release.set()
        other.join(5)


def _fresh_eval_fake() -> SimpleNamespace:
    fake = _engine_like(_worker=SimpleNamespace(
        fresh_root_policy_bestmove=lambda *_a, **_k: "e2e4"))
    fake._warmup_done.set()
    return fake


def test_uci_fresh_eval_wait_for_an_exiting_helper_keeps_its_budget(slow_thread_exit) -> None:
    """`_wait_for_search` polls a still-exiting helper only until its own
    budget: 2 ms against the helper's 0.15 s teardown (a Thread join would
    wait it out on CPython 3.13). `go` tolerates it (recorded, no raise);
    any other caller gets `SearchNotQuiescent`; a later wait with room joins it."""
    from chess_anti_engine.uci.engine import Engine, SearchNotQuiescent

    fake = _fresh_eval_fake()
    assert Engine._bounded_fresh_eval(fake, chess.Board(), (), 0.02) == "e2e4"  # pyright: ignore[reportArgumentType]
    helpers = [t for t in slow_thread_exit if t.name == "uci-fresh-eval"]
    assert [h.thread for h in fake._exiting_helpers] == helpers
    t0 = time.monotonic()
    Engine._wait_for_search(fake, timeout_s=0.002, tolerate_exiting_helpers=True)  # pyright: ignore[reportArgumentType]
    assert time.monotonic() - t0 < 0.08, "the go wait outlived its 2 ms budget"
    assert [h.thread for h in fake._exiting_helpers] == helpers, "still recorded"
    t0 = time.monotonic()
    with pytest.raises(SearchNotQuiescent, match="still exiting"):
        Engine._wait_for_search(fake, timeout_s=0.002)  # pyright: ignore[reportArgumentType]
    assert time.monotonic() - t0 < 0.08, "the raising wait outlived its 2 ms budget"
    assert [h.thread for h in fake._exiting_helpers] == helpers, "still recorded"
    Engine._wait_for_search(fake, timeout_s=5.0)  # pyright: ignore[reportArgumentType]
    _assert_os_thread_gone(helpers[0])
    assert fake._exiting_helpers == []


class _FinishesAfterTheWait(threading.Event):
    """The eval ends just after the timed wait gave up: `wait` reports a
    timeout while the flag is (by then) set."""

    def wait(self, timeout: float | None = None) -> bool:
        super().wait(timeout)
        return False


def test_uci_fresh_eval_a_helper_finishing_after_the_wait_is_orphaned(monkeypatch) -> None:
    """One read of the event decides: a helper whose eval ended after the wait
    timed out is treated as timed out (parked as an orphan, first-legal
    answer), never dropped between the two branches."""
    from chess_anti_engine.uci import engine as engine_module
    from chess_anti_engine.uci.engine import Engine

    # Only this module's `threading.Event` changes (`Thread` keeps the real one).
    monkeypatch.setattr(engine_module, "threading", SimpleNamespace(
        **{**vars(threading), "Event": _FinishesAfterTheWait}))
    fake = _fresh_eval_fake()
    assert Engine._bounded_fresh_eval(fake, chess.Board(), (), 0.05) is None  # pyright: ignore[reportArgumentType]
    monkeypatch.undo()
    helpers = [t for t in fake._orphan_threads if t.name == "uci-fresh-eval"]
    assert len(helpers) == 1, "the helper must be parked, not dropped"
    Engine._wait_for_search(fake, timeout_s=5.0)  # pyright: ignore[reportArgumentType]
    _assert_os_thread_gone(helpers[0])
