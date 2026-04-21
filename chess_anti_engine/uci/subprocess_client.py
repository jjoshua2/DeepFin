"""Client-side helper for talking to a UCI engine subprocess.

Shared by `tests/test_uci_smoke.py` and `scripts/bench_uci_engine.py`.
Both need to spawn the engine, pump stdout without deadlocking on pipe
buffering, and drive it through handshakes by sending lines and reading
until a needle. This module gives them one implementation.
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time


class LineReader:
    """Background thread pumping subprocess stdout into a queue.

    Avoids the select+readline pipe-buffering race where TextIOWrapper
    has buffered lines user-side but select on the fd says "nothing to
    read".
    """

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self._q: queue.Queue[str | None] = queue.Queue()
        self._proc = proc
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

    def _pump(self) -> None:
        assert self._proc.stdout is not None
        for line in iter(self._proc.stdout.readline, ""):
            self._q.put(line.rstrip("\n"))
        self._q.put(None)

    def read_until(self, needle: str, *, timeout_s: float = 60.0) -> list[str]:
        lines: list[str] = []
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out waiting for {needle!r}; got:\n" + "\n".join(lines[-20:])
                )
            try:
                line = self._q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(
                    f"engine exited before {needle!r}; got:\n" + "\n".join(lines[-20:])
                )
            lines.append(line)
            if needle in line:
                return lines


def send_line(proc: subprocess.Popen[str], line: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(line + "\n")
    proc.stdin.flush()
