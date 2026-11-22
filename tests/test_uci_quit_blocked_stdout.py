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
# `quit` with a blocked stdout: the existing 1 s stop wait for the search, one
# 0.25 s bounded try for the print lock, then the hard exit. 2 s leaves margin
# for a loaded CI box; today the process never exits.
QUIT_BOUND_S = 2.0
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

    def __init__(self, checkpoint: Path, stderr_path: Path) -> None:
        r, w = os.pipe()
        fcntl.fcntl(r, _F_SETPIPE_SZ, 4096)
        # Own open file description: O_NONBLOCK here does not reach the engine.
        self._filler = os.open(f"/proc/self/fd/{w}", os.O_WRONLY | os.O_NONBLOCK)
        env = {**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": ""}
        self._stderr = stderr_path.open("w")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "chess_anti_engine.uci",
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

    def wait_blocked_in_write(self, timeout_s: float = 60.0) -> None:
        """Until an engine thread sits in write(1, ...) (x86-64/arm64 nr 1/64)."""
        nr = {"x86_64": "1", "aarch64": "64"}.get(os.uname().machine, "1")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for tid in os.listdir(f"/proc/{self.proc.pid}/task"):
                try:
                    with open(f"/proc/{self.proc.pid}/task/{tid}/syscall") as fh:
                        fields = fh.read().split()
                except OSError:
                    continue
                if fields[:2] == [nr, "0x1"]:
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
    assert took <= QUIT_BOUND_S, f"quit took {took:.2f}s > {QUIT_BOUND_S}s"
    # Exit status 1: the search could not be quiesced (as for any quit whose
    # search ignored stop), not a clean 0.
    assert engine.proc.returncode == 1, _stderr_text(tmp_path)[-2000:]
    assert "stdout is not being read" in _stderr_text(tmp_path)
    # Nothing the engine started outlives it (a helper still holding a device).
    deadline = time.monotonic() + 2.0
    while any(_alive(p) for p in helpers) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not [p for p in helpers if _alive(p)], f"helpers survived quit: {helpers}"


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
    # No wait for a writer to block: the quit races the search's next line.
    engine.send("position startpos")
    engine.send("go infinite")
    engine.read_until("info")
    engine.block_stdout()
    helpers = _descendants(engine.proc.pid)
    engine.send("quit")
    took = engine.wait_exit(_NEVER_S)
    assert took is not None, f"no exit within {_NEVER_S}s\n{_stderr_text(tmp_path)[-2000:]}"
    assert took <= QUIT_BOUND_S, f"exit took {took:.2f}s > {QUIT_BOUND_S}s"
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
    assert took <= QUIT_BOUND_S, f"exit took {took:.2f}s > {QUIT_BOUND_S}s"
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
    assert took <= QUIT_BOUND_S, f"exit took {took:.2f}s > {QUIT_BOUND_S}s"
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
        # Latched: later quit-path lines skip without waiting at all.
        assert _println_bounded("info string y") is False
    finally:
        release.set()
        th.join(5)
    r_ready, _, _ = select.select([r], [], [], 0)
    assert not r_ready  # nothing was written

