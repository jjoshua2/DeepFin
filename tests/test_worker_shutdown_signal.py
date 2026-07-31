"""SIGTERM must route into the SAME graceful teardown a reco restart-key change
performs, so a shutdown suspends in-flight games instead of discarding them.

Before this, `worker.py` installed no signal handler at all, so
`_suspend_inflight_games` was reachable ONLY through `_stop_selfplay`. Measured
on production 2026-07-31: `resumed games=0` on all four workers after an
operator pause, versus `resumed games=24 records=731` across a graceful session
restart. The discarded games are what makes the next window length-truncated,
which spikes winrate and drives the PID to tighten regret against a phantom.
"""
from __future__ import annotations

import logging
import os
import signal

import pytest

from chess_anti_engine.worker import WorkerSession


@pytest.fixture
def restore_handlers():
    """Leave the interpreter's handlers exactly as found — a leaked handler
    would silently change how the rest of the suite reacts to signals."""
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


def _bare_session() -> WorkerSession:
    """A WorkerSession with only the attributes the handler path touches.

    `__init__` needs a server, CLI args and a work dir; none of that is
    involved in signal handling, so constructing it would test the harness
    rather than the behaviour.
    """
    w = object.__new__(WorkerSession)
    w._stop_selfplay = False
    w._shutdown_requested = False
    w._shutdown_signal = 0
    w._hold_selfplay = False
    w.log = logging.getLogger("test_worker_shutdown")
    return w


@pytest.mark.usefixtures("restore_handlers")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_sets_the_flag_the_suspend_path_keys_on(sig: signal.Signals) -> None:
    w = _bare_session()
    assert w._stop_fn() is False

    w._install_shutdown_handlers()
    os.kill(os.getpid(), sig)

    # THE POINT: `_stop_fn` is what play_batch polls every ply, and a True here
    # is what makes it run `on_suspend` -> `_suspend_inflight_games`. Asserting
    # only `_shutdown_requested` would pass even if the suspend path were never
    # reached.
    assert w._stop_fn() is True
    assert w._shutdown_requested is True
    assert w._shutdown_signal == int(sig)


@pytest.mark.usefixtures("restore_handlers")
def test_pending_stop_beats_a_train_phase_hold() -> None:
    """A shutdown must not be parked by the train-phase pause gate: `_pause_fn`
    holds games in flight rather than tearing down, so if a hold outranked the
    signal the worker would sit there until SIGKILL and lose the games anyway."""
    w = _bare_session()
    w._hold_selfplay = True
    assert w._pause_fn() is True

    w._install_shutdown_handlers()
    os.kill(os.getpid(), signal.SIGTERM)

    assert w._pause_fn() is False
    assert w._stop_fn() is True


def test_run_installs_the_handlers_before_it_polls() -> None:
    """The handler only helps if the PRODUCTION entry point installs it. Every
    other test here calls `_install_shutdown_handlers` directly and so passes
    even with the call deleted from `run()` — verified by removing it. This is
    the test that fails.
    """
    order: list[str] = []

    class _StopLoop(Exception):
        pass

    def _poll() -> None:
        order.append("poll")
        raise _StopLoop

    w = _bare_session()
    w._install_shutdown_handlers = lambda: order.append("install")
    w._poll_manifest = _poll
    w._cleanup = lambda: order.append("cleanup")

    with pytest.raises(_StopLoop):
        w.run()

    # Order matters: a worker that polls first can be signalled during that
    # first poll and die unhandled.
    assert order == ["install", "poll", "cleanup"], order


@pytest.mark.usefixtures("restore_handlers")
def test_run_exits_at_the_loop_head_once_a_shutdown_is_requested() -> None:
    """After the suspend path returns, `run()` must exit rather than poll for
    more work — otherwise the drained games are followed by a fresh session
    that the imminent SIGKILL discards."""
    order: list[str] = []
    w = _bare_session()
    w._poll_manifest = lambda: order.append("poll")
    w._cleanup = lambda: order.append("cleanup")
    w._install_shutdown_handlers()
    os.kill(os.getpid(), signal.SIGTERM)

    w.run()   # must return, not spin

    assert "poll" not in order, order
    assert order == ["cleanup"], order


@pytest.mark.usefixtures("restore_handlers")
def test_session_setup_cannot_erase_a_pending_shutdown() -> None:
    """`_run_selfplay` clears `_stop_selfplay` at session start. That is the
    ONLY flag `play_batch` polls, so a SIGTERM arriving between `run()`'s
    loop-head check and that line used to be erased outright — and the worker
    then started a full 32-thread session with nothing left to stop it.

    The window is not exotic: the signal normally lands while the worker is
    blocked in `_poll_manifest`'s 30s GET, or syncing assets, or resuming
    banked games. Worse, that session would run `_resume_inflight_games` and
    un-bank the games an EARLIER teardown had saved, so they die too.
    """
    w = _bare_session()
    w._install_shutdown_handlers()
    os.kill(os.getpid(), signal.SIGTERM)
    assert w._stop_fn() is True

    # Exactly what session start does to the flag.
    w._stop_selfplay = False

    # THE POINT: the shutdown is terminal and must survive the reset.
    assert w._stop_fn() is True, "the flag play_batch polls was erased by session setup"
    assert w._pause_fn() is False


@pytest.mark.usefixtures("restore_handlers")
def test_run_selfplay_refuses_to_start_a_session_after_a_shutdown() -> None:
    """Belt to the `_stop_fn` braces: don't merely stop the new session on its
    first ply, don't start it. A bare session has no model, client or manifest,
    so reaching ANY real setup would raise — returning cleanly is the proof."""
    w = _bare_session()
    w._install_shutdown_handlers()
    os.kill(os.getpid(), signal.SIGTERM)

    w._run_selfplay({})   # must return, not raise and not set up a session

    assert w._stop_fn() is True


@pytest.mark.usefixtures("restore_handlers")
def test_handler_installation_is_not_fatal_off_the_main_thread(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Handlers can only be installed from the main thread. If that ever fails
    the worker must keep running (degraded to the old behaviour), not crash —
    losing in-flight games is bad, losing the worker is worse."""
    def _boom(*_a, **_k):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(signal, "signal", _boom)
    w = _bare_session()
    with caplog.at_level(logging.WARNING):
        w._install_shutdown_handlers()   # must not raise
    assert w._stop_fn() is False
