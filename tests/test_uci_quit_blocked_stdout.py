"""`quit` must exit within a fixed bound even when the GUI stopped reading stdout.

A GUI (or tournament manager) that stops draining our stdout leaves the search
thread blocked inside ``write`` while it holds the print lock. Before the
bounded quit path, `quit`'s own "search did not stop ... parked" line then
waited on that lock forever: the process never exited and kept its evaluator,
device context and threads (a leaked GPU holder after the manager's quit).

The rig owns the engine's stdout pipe: it shrinks it to one page and fills it
from its own non-blocking write end (a separate open file description, so the
engine's end stays blocking), then waits until ``/proc`` shows an engine
thread inside ``write(1, ...)``. Linux only; CPU only.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.test_uci_smoke import _make_tiny_checkpoint

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="needs /proc and F_SETPIPE_SZ",
)

_F_SETPIPE_SZ = 1031
# `quit` with a blocked stdout exits about 2 s after a writer blocks (the guard's
# 1.7 s stall window, one 0.05 s look, the 0.25 s-bounded stderr note): measured
# 1.87-2.09 s under two busy loops, about 2.4 s under heavier load. The block
# itself comes at most about 1 s after `quit` (stop latency), so the exit is
# within about 3 s of `quit` by construction. The tests allow `_SLACK_S` over
# the 2.0 s target for a loaded CI box (2.7 s, still far from "never exits":
# before the fix the process never exited).
QUIT_BOUND_S = 2.0
_SLACK_S = 0.7
# `_handle_stop` waits at most this (engine `_GO_WAIT_NO_CLOCK_S`) for the search,
# then parks it; a parked thread is silenced inside the print lock. So a writer
# that is going to block does so within about this long of `quit`.
_STOP_LATENCY_S = 1.0
# How long a pre-fix run is watched before it counts as "never exits".
_NEVER_S = 10.0


def _descendants(pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                stat = fh.read()
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(entry))
    out: set[int] = set()
    todo = [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            if child not in out:
                out.add(child)
                todo.append(child)
    return out


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


class _Engine:
    """The UCI engine on a stdout pipe this rig can stop draining."""

    def __init__(self, checkpoint: Path, stderr_path: Path, *,
                 launcher: Path | None = None, env_extra: dict[str, str] | None = None) -> None:
        r, w = os.pipe()
        fcntl.fcntl(r, _F_SETPIPE_SZ, 4096)
        # Own open file description: O_NONBLOCK here does not reach the engine.
        self._filler = os.open(f"/proc/self/fd/{w}", os.O_WRONLY | os.O_NONBLOCK)
        env = {**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": "", **(env_extra or {})}
        self._stderr = stderr_path.open("w")
        entry = [str(launcher)] if launcher is not None else ["-m", "chess_anti_engine.uci"]
        self.proc = subprocess.Popen(
            [sys.executable, "-u", *entry,
             "--checkpoint", str(checkpoint), "--device", "cpu", "--no-compile"],
            stdin=subprocess.PIPE, stdout=w, stderr=self._stderr, bufsize=0, env=env,
        )
        os.close(w)
        self._r = r
        self._buf = b""
        self.lines: list[str] = []
        self._filled = 0

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()

    def read_until(self, want: str, timeout_s: float = 120.0) -> list[str]:
        """Drain stdout until a line starts with ``want``; the lines read."""
        deadline = time.monotonic() + timeout_s
        start = len(self.lines)
        while time.monotonic() < deadline:
            if any(s.startswith(want) for s in self.lines[start:]):
                return self.lines[start:]
            self._drain_once(0.05)
        raise AssertionError(f"no {want!r} within {timeout_s}s: {self.lines[start:][-5:]}")

    def _drain_once(self, wait_s: float) -> None:
        r, _, _ = select.select([self._r], [], [], wait_s)
        if not r:
            return
        chunk = os.read(self._r, 65536)
        if self._filled:
            # The rig's own filler bytes come first: drop them.
            skip = min(self._filled, len(chunk))
            chunk, self._filled = chunk[skip:], self._filled - skip
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self.lines.append(line.decode(errors="replace"))

    def block_stdout(self) -> None:
        """Fill the pipe to the brim; any engine write now blocks."""
        assert not self._buf, "drain whole lines before blocking"
        while True:
            try:
                n = os.write(self._filler, b"#" * 512)
            except BlockingIOError:
                break
            self._filled += n

    def drop_stdout_reader(self) -> None:
        """Close every read end of the engine's stdout: its writes now fail
        with EPIPE (the GUI died)."""
        for fd in (self._r, self._filler):
            os.close(fd)
        self._r = os.open(os.devnull, os.O_RDONLY)
        self._filler = os.open(os.devnull, os.O_WRONLY)

    def blocked_in_write(self) -> bool:
        """An engine thread sits in write(1, ...) now (x86-64/arm64 nr 1/64)."""
        nr = {"x86_64": "1", "aarch64": "64"}.get(os.uname().machine, "1")
        try:
            tids = os.listdir(f"/proc/{self.proc.pid}/task")
        except OSError:
            return False
        for tid in tids:
            try:
                with open(f"/proc/{self.proc.pid}/task/{tid}/syscall") as fh:
                    fields = fh.read().split()
            except OSError:
                continue
            if fields[:2] == [nr, "0x1"]:
                return True
        return False

    def wait_blocked_in_write(self, timeout_s: float = 60.0) -> None:
        """Until an engine thread sits in write(1, ...)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.blocked_in_write():
                return
            time.sleep(0.05)
        raise AssertionError("no engine thread blocked in write(stdout)")

    def wait_exit(self, timeout_s: float) -> float | None:
        """Seconds until the engine exited, without draining stdout; None if not."""
        t0 = time.monotonic()
        try:
            self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None
        return time.monotonic() - t0

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(10)
        for fd in (self._r, self._filler):
            try:
                os.close(fd)
            except OSError:
                pass
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        self._stderr.close()


@pytest.fixture
def engine(tmp_path: Path):
    eng = _Engine(_make_tiny_checkpoint(tmp_path), tmp_path / "engine.stderr")
    eng.send("uci")
    eng.read_until("uciok")
    eng.send("isready")
    eng.read_until("readyok")
    yield eng
    eng.close()


def _stderr_text(tmp_path: Path) -> str:
    return (tmp_path / "engine.stderr").read_text(errors="replace")


def _assert_quits_within_bound(engine: _Engine, tmp_path: Path) -> None:
    helpers = _descendants(engine.proc.pid)
    engine.send("quit")
    took = engine.wait_exit(_NEVER_S)
    assert took is not None, (
        f"quit with stdout blocked did not exit within {_NEVER_S}s\n{_stderr_text(tmp_path)[-2000:]}"
    )
    assert took <= QUIT_BOUND_S + _SLACK_S, f"quit took {took:.2f}s > {QUIT_BOUND_S}s"
    # Exit status 1: the search could not be quiesced (as for any quit whose
    # search ignored stop), not a clean 0.
    assert engine.proc.returncode == 1, _stderr_text(tmp_path)[-2000:]
    assert "stdout is not being read" in _stderr_text(tmp_path)
    # Nothing the engine started outlives it (a helper still holding a device).
    deadline = time.monotonic() + 2.0
    while any(_alive(p) for p in helpers) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not [p for p in helpers if _alive(p)], f"helpers survived quit: {helpers}"


# Test-only faults, injected before `main()` runs (never product code):
# DEEPFIN_TEST_TEARDOWN_S: Engine.close sleeps that long, then the real close
#   runs and DEEPFIN_TEST_CLOSED_MARK is written (proof the cleanup completed).
# DEEPFIN_TEST_BUILD_SLEEP_S: the startup model load sleeps that long.
_LAUNCHER = """
import os, sys, time
sys.path.insert(0, os.getcwd())
td = os.environ.get("DEEPFIN_TEST_TEARDOWN_S")
if td:
    import chess_anti_engine.uci.engine as e
    real_close = e.Engine.close
    def close(self, *a, **k):
        time.sleep(float(td))
        out = real_close(self, *a, **k)
        with open(os.environ["DEEPFIN_TEST_CLOSED_MARK"], "w") as fh:
            fh.write("closed")
        return out
    e.Engine.close = close
bs = os.environ.get("DEEPFIN_TEST_BUILD_SLEEP_S")
if bs:
    import chess_anti_engine.uci.__main__ as m
    real_load = m._load_models
    def load(*a, **k):
        time.sleep(float(bs))
        return real_load(*a, **k)
    m._load_models = load
from chess_anti_engine.uci.cli import main
raise SystemExit(main())
"""


@pytest.fixture
def launcher(tmp_path: Path) -> Path:
    path = tmp_path / "launch_engine.py"
    path.write_text(_LAUNCHER)
    return path


def _ready(eng: _Engine) -> _Engine:
    eng.send("uci")
    eng.read_until("uciok")
    eng.send("isready")
    eng.read_until("readyok")
    return eng


def test_quit_exits_within_bound_while_the_search_is_blocked_writing(
    engine: _Engine, tmp_path: Path,
) -> None:
    engine.send("position startpos")
    engine.send("go infinite")
    engine.read_until("info")
    engine.block_stdout()
    engine.wait_blocked_in_write()
    _assert_quits_within_bound(engine, tmp_path)


def test_quit_during_a_search_sent_the_moment_stdout_fills(
    engine: _Engine, tmp_path: Path,
) -> None:
    # No wait for a writer to block: the quit races the search's next line,
    # which may block only after `quit` arrived (later still under load). The
    # bound runs from whichever came later: `quit`, or the writer blocking.
    engine.send("position startpos")
    engine.send("go infinite")
    engine.read_until("info")
    engine.block_stdout()
    helpers = _descendants(engine.proc.pid)
    engine.send("quit")
    t_quit = t_ref = time.monotonic()
    seen_block = False
    deadline = t_ref + _NEVER_S
    while engine.proc.poll() is None and time.monotonic() < deadline:
        if not seen_block and engine.blocked_in_write():
            seen_block, t_ref = True, max(t_ref, time.monotonic())
        time.sleep(0.01)
    assert engine.proc.poll() is not None, (
        f"no exit within {_NEVER_S}s\n{_stderr_text(tmp_path)[-2000:]}")
    took = time.monotonic() - t_ref
    assert took <= QUIT_BOUND_S + _SLACK_S, (
        f"exit took {took:.2f}s after {'the writer blocked' if seen_block else 'quit'}")
    # And from `quit` itself: the block comes within the stop latency, so a
    # regression that delays it (stop no longer prompt) fails here instead of
    # being absorbed by the moving reference above.
    from_quit = time.monotonic() - t_quit
    assert from_quit <= QUIT_BOUND_S + _STOP_LATENCY_S + _SLACK_S, (
        f"exit took {from_quit:.2f}s after quit")
    # 0 if the search stopped before its next write, 1 if that write blocked.
    assert engine.proc.returncode in (0, 1)
    assert not [p for p in helpers if _alive(p)]


def test_quit_after_stop_parked_a_search_blocked_writing(
    engine: _Engine, tmp_path: Path,
) -> None:
    # `stop` parks the blocked search after 1 s and answers from the fallback;
    # that bestmove itself waits for stdout. The quit behind it still exits.
    engine.send("position startpos")
    engine.send("go infinite")
    engine.read_until("info")
    engine.block_stdout()
    engine.wait_blocked_in_write()
    engine.send("stop")
    time.sleep(1.5)
    assert engine.proc.poll() is None
    _assert_quits_within_bound(engine, tmp_path)


def test_stdin_eof_with_stdout_blocked_exits_within_bound(
    engine: _Engine, tmp_path: Path,
) -> None:
    # The GUI closed our stdin but still holds (and does not drain) our stdout.
    # EOF ends the session like `quit`; the exit path's bounded print finds
    # stdout stuck and leaves.
    engine.send("position startpos")
    engine.send("go infinite")
    engine.read_until("info")
    engine.block_stdout()
    engine.wait_blocked_in_write()
    assert engine.proc.stdin is not None
    engine.proc.stdin.close()
    took = engine.wait_exit(_NEVER_S)
    assert took is not None, f"no exit within {_NEVER_S}s\n{_stderr_text(tmp_path)[-2000:]}"
    assert took <= QUIT_BOUND_S + _SLACK_S, f"exit took {took:.2f}s > {QUIT_BOUND_S}s"
    assert engine.proc.returncode == 1


def test_stdin_eof_after_the_gui_closed_our_stdout_exits_on_epipe(
    engine: _Engine, tmp_path: Path,
) -> None:
    # The GUI died: no reader on our stdout, then EOF. Writes fail with EPIPE
    # instead of blocking; the engine exits promptly (it does not spin), with
    # the status it had before this change (0: the search stopped and joined).
    engine.send("position startpos")
    engine.send("go infinite")
    engine.read_until("info")
    engine.drop_stdout_reader()
    time.sleep(1.0)
    assert engine.proc.stdin is not None
    engine.proc.stdin.close()
    took = engine.wait_exit(_NEVER_S)
    assert took is not None, f"no exit within {_NEVER_S}s\n{_stderr_text(tmp_path)[-2000:]}"
    assert took <= QUIT_BOUND_S + _SLACK_S, f"exit took {took:.2f}s > {QUIT_BOUND_S}s"
    assert engine.proc.returncode == 0


def test_a_normal_quit_is_unchanged(tmp_path: Path) -> None:
    # stdout drained throughout: a search to its bestmove, then quit. Exit 0
    # and nothing printed after the bestmove (no quit-path line, no guard).
    eng = _Engine(_make_tiny_checkpoint(tmp_path), tmp_path / "engine.stderr")
    try:
        eng.send("uci")
        eng.read_until("uciok")
        eng.send("isready")
        eng.read_until("readyok")
        eng.send("position startpos")
        eng.send("go nodes 8")
        eng.read_until("bestmove")
        eng.send("quit")
        t0 = time.monotonic()
        while eng.proc.poll() is None and time.monotonic() - t0 < 30:
            eng._drain_once(0.05)
        eng._drain_once(0.0)
        assert eng.proc.returncode == 0
        assert eng.lines[-1].startswith("bestmove")
        assert "stdout is not being read" not in _stderr_text(tmp_path)
    finally:
        eng.close()


@pytest.fixture
def pipe_stdout(monkeypatch: pytest.MonkeyPatch):
    """A real one-page pipe for sys.stdout; the stuck latch cleared after.
    The test installs it (`_install`): pytest's capture resumes between
    fixture setup and the call and would put its own stdout back."""
    from chess_anti_engine.uci import engine as engine_module

    r, w = os.pipe()
    fcntl.fcntl(r, _F_SETPIPE_SZ, 4096)
    out = open(w, "w", buffering=1, closefd=True)  # noqa: SIM115
    engine_module._STDOUT_STUCK.clear()
    try:
        yield r, w, lambda: monkeypatch.setattr(sys, "stdout", out)
    finally:
        engine_module._STDOUT_STUCK.clear()
        os.close(r)  # first: a writer blocked on the full pipe now gets EPIPE
        with contextlib.suppress(OSError):
            out.close()


def test_bounded_print_writes_what_println_writes(pipe_stdout) -> None:
    from chess_anti_engine.uci.engine import _println, _println_bounded, stdout_stuck

    r, _w, install = pipe_stdout
    install()
    assert _println("info string a") is True
    assert _println_bounded("info string a") is True
    assert os.read(r, 4096) == b"info string a\ninfo string a\n"
    assert not stdout_stuck()


def test_bounded_print_skips_the_line_when_the_print_lock_is_held(pipe_stdout) -> None:
    from chess_anti_engine.uci.engine import _PRINT_LOCK, _println_bounded, stdout_stuck

    r, _w, install = pipe_stdout
    install()
    held, release = threading.Event(), threading.Event()

    def writer_stuck_in_write() -> None:
        with _PRINT_LOCK:
            held.set()
            release.wait(10)

    th = threading.Thread(target=writer_stuck_in_write, daemon=True)
    th.start()
    assert held.wait(5)
    try:
        result: list[bool] = []
        t0 = time.monotonic()
        caller = threading.Thread(target=lambda: result.append(_println_bounded("info string x")),
                                  daemon=True)
        caller.start()
        caller.join(3.0)
        took = time.monotonic() - t0
        assert result == [False], "bounded print waited on the held print lock"
        assert took < 1.0
        assert stdout_stuck()
        # Each later quit-path line waits its own bounded try again.
        assert _println_bounded("info string y") is False
        r_ready, _, _ = select.select([r], [], [], 0)
        assert not r_ready  # nothing was written while the lock was held
    finally:
        release.set()
        th.join(5)
    # The lock is free again: the next bounded print gets it and clears the flag.
    assert _println_bounded("info string z") is True
    assert not stdout_stuck()
    assert os.read(r, 4096) == b"info string z\n"



# -- the guard's trigger: a writer actually blocked, never a late reader ---------


def test_quit_with_a_slow_teardown_and_stdout_drained_exits_cleanly(
    tmp_path: Path, launcher: Path,
) -> None:
    # Nothing is blocked: a 2 s teardown (longer than the guard's window) must
    # run to completion and exit 0. Kills "the guard fires without a blocked writer".
    mark = tmp_path / "closed"
    eng = _Engine(_make_tiny_checkpoint(tmp_path), tmp_path / "engine.stderr", launcher=launcher,
                  env_extra={"DEEPFIN_TEST_TEARDOWN_S": "2.0", "DEEPFIN_TEST_CLOSED_MARK": str(mark)})
    try:
        _ready(eng)
        eng.send("position startpos")
        eng.send("go nodes 8")
        eng.read_until("bestmove")
        eng.send("quit")
        t0 = time.monotonic()
        while eng.proc.poll() is None and time.monotonic() - t0 < 30:
            eng._drain_once(0.05)
        assert eng.proc.returncode == 0, _stderr_text(tmp_path)[-2000:]
        assert mark.exists(), "the real close did not complete"
    finally:
        eng.close()


def test_a_one_page_pipe_with_an_unread_bestmove_and_a_slow_teardown_exits_cleanly(
    tmp_path: Path, launcher: Path,
) -> None:
    # The final bestmove sits unread in a one-page pipe (Linux then reports the
    # pipe unwritable) while a 2 s teardown runs: no writer is blocked, so the
    # exit is the clean one.
    mark = tmp_path / "closed"
    eng = _Engine(_make_tiny_checkpoint(tmp_path), tmp_path / "engine.stderr", launcher=launcher,
                  env_extra={"DEEPFIN_TEST_TEARDOWN_S": "2.0", "DEEPFIN_TEST_CLOSED_MARK": str(mark)})
    try:
        _ready(eng)
        eng.send("position startpos")
        eng.send("go nodes 8")
        time.sleep(1.0)  # the search ends; its lines and bestmove stay unread
        eng.send("quit")
        assert eng.wait_exit(30) is not None
        assert eng.proc.returncode == 0, _stderr_text(tmp_path)[-2000:]
        assert mark.exists()
        eng.read_until("bestmove", 5.0)
    finally:
        eng.close()


def test_a_gui_that_stalls_1_5s_after_quit_then_reads_gets_the_clean_exit(
    engine: _Engine, tmp_path: Path,
) -> None:
    # A writer IS blocked, but the GUI resumes reading 1.5 s after `quit` --
    # inside the guard's 1.7 s window -- so the quit completes normally.
    engine.send("position startpos")
    engine.send("go infinite")
    engine.read_until("info")
    engine.block_stdout()
    engine.wait_blocked_in_write()
    engine.send("quit")
    time.sleep(1.5)
    t0 = time.monotonic()
    while engine.proc.poll() is None and time.monotonic() - t0 < 30:
        engine._drain_once(0.05)
    assert engine.proc.returncode == 0, _stderr_text(tmp_path)[-2000:]
    assert "stdout is not being read" not in _stderr_text(tmp_path)


def test_the_guard_does_not_fire_on_a_writer_that_keeps_completing_lines(
    pipe_stdout, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A slow but progressing writer, at its worst: it never lets go of the print
    # lock (so every look finds it held), yet completes a line every 0.3 s. Only
    # progress tells it apart from a writer blocked in `write`.
    from chess_anti_engine.uci import __main__ as uci_main
    from chess_anti_engine.uci.engine import _PRINT_LOCK, _println

    r, _w, install = pipe_stdout
    install()
    exits: list[int] = []
    monkeypatch.setattr(uci_main, "_hard_exit", exits.append)
    stop = threading.Event()

    def progressing_writer() -> None:
        with _PRINT_LOCK:
            while not stop.is_set():
                time.sleep(0.3)
                _println("info string progress")
                os.read(r, 4096)

    th = threading.Thread(target=progressing_writer, daemon=True)
    th.start()
    feed = uci_main._StdinFeed(iter(()))
    try:
        feed.arm_quit_guard()
        time.sleep(uci_main._STDOUT_STALL_S + 1.0)
        assert exits == []
    finally:
        feed.disarm_quit_guard()
        stop.set()
        th.join(5)


def test_the_guard_fires_when_the_lock_is_held_and_nothing_completes(
    pipe_stdout, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The case the lock-based trigger exists for, including the main thread
    # itself blocked inside a print while holding the lock.
    from chess_anti_engine.uci import __main__ as uci_main
    from chess_anti_engine.uci.engine import _PRINT_LOCK

    _r, _w, install = pipe_stdout
    install()
    exits: list[int] = []
    monkeypatch.setattr(uci_main, "_hard_exit", exits.append)
    release = threading.Event()

    def stuck_writer() -> None:
        with _PRINT_LOCK:
            release.wait(10)

    th = threading.Thread(target=stuck_writer, daemon=True)
    th.start()
    feed = uci_main._StdinFeed(iter(()))
    try:
        t0 = time.monotonic()
        feed.arm_quit_guard()
        while not exits and time.monotonic() - t0 < 5:
            time.sleep(0.02)
        took = time.monotonic() - t0
        assert exits[:1] == [1]
        assert uci_main._STDOUT_STALL_S <= took <= QUIT_BOUND_S + _SLACK_S
    finally:
        feed.disarm_quit_guard()
        release.set()
        th.join(5)


# -- stdin feed: stamps and arming -----------------------------------------------


def test_a_line_queued_behind_a_slow_command_is_stamped_when_the_loop_takes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The move clock's origin is when the main loop takes the line, as before
    # the reader thread existed -- not when the reader pulled it off the pipe.
    from chess_anti_engine.uci import __main__ as uci_main

    now = [1.0]
    monkeypatch.setattr(uci_main, "_receipt_clock", lambda: now[0])
    feed = uci_main._StdinFeed(iter(["isready\n", "go movetime 1000\n"]))
    deadline = time.monotonic() + 5
    while feed._lines.qsize() < 3 and time.monotonic() < deadline:  # two lines + EOF
        time.sleep(0.01)
    lines = uci_main._received_lines(feed)
    assert next(lines) == ("isready\n", 1.0)
    now[0] = 9.0  # the loop spent 8 s on `isready` while `go` waited queued
    assert next(lines) == ("go movetime 1000\n", 9.0)


@pytest.mark.parametrize(("line", "arms"), [
    ("quit\n", True),
    ("setoption name X value quit\n", False),
    ("position startpos moves e2e4 quit\n", False),
])
def test_only_the_quit_command_arms_the_guard(line: str, arms: bool) -> None:
    from chess_anti_engine.uci import __main__ as uci_main

    feed = uci_main._StdinFeed(iter([line]))
    try:
        assert next(iter(feed)) == line  # after `quit` the main loop stops reading
        feed._reader.join(5)
        assert not feed._reader.is_alive()
        assert (feed._guard is not None) is arms
    finally:
        feed.disarm_quit_guard()


def test_the_pre_readiness_burst_still_searches_its_movetime(
    tmp_path: Path, launcher: Path,
) -> None:
    # `uci / isready / position / go movetime 1000` sent at once while the model
    # loads for 3 s: `go` is stamped when the loop takes it (after readyok), so
    # it searches about its movetime instead of answering unsearched.
    eng = _Engine(_make_tiny_checkpoint(tmp_path), tmp_path / "engine.stderr", launcher=launcher,
                  env_extra={"DEEPFIN_TEST_BUILD_SLEEP_S": "3.0"})
    try:
        for line in ("uci", "isready", "position startpos", "go movetime 1000"):
            eng.send(line)
        eng.read_until("readyok", 120)
        t_ready = time.monotonic()
        lines = eng.read_until("bestmove", 30)
        searched = time.monotonic() - t_ready
        nodes = [int(s.split()[s.split().index("nodes") + 1]) for s in lines
                 if s.startswith("info") and " nodes " in s]
        assert searched >= 0.7, f"bestmove {searched:.2f}s after readyok for movetime 1000"
        if max(nodes, default=0) == 0:
            # An overloaded box can starve the search (seen on main too, about 1
            # run in 6): then the watchdog answers, and its note states how far
            # into the move it was -- the clock origin itself. Timed from
            # readiness that is about the movetime; charged for the 3 s model
            # load (the origin at arrival) it would be over 3000 ms.
            import re

            notes = [re.search(r"(\d+) ms into a clocked move", s) for s in lines]
            into = [int(m.group(1)) for m in notes if m]
            assert into, ("no nodes and no watchdog note", lines[-5:])
            assert 700 <= into[0] <= 2500, ("no nodes and a wrong clock origin", lines[-5:])
    finally:
        eng.close()


def test_stderr_note_is_skipped_when_stderr_cannot_be_polled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Windows: `select` on a pipe raises. An unknown-writability stderr gets no
    # (possibly unbounded) write; a non-file stand-in still gets the line.
    import io

    from chess_anti_engine.uci import __main__ as uci_main

    def no_select(*_a, **_k):
        raise OSError("select on a pipe is not supported")

    monkeypatch.setattr(uci_main.select, "select", no_select)
    with (tmp_path / "stderr").open("w") as real:
        monkeypatch.setattr(sys, "stderr", real)
        uci_main._stderr_note("deepfin: note")
    assert (tmp_path / "stderr").read_text() == ""
    fake = io.StringIO()
    monkeypatch.setattr(sys, "stderr", fake)
    uci_main._stderr_note("deepfin: note")
    assert fake.getvalue() == "deepfin: note\n"
