from __future__ import annotations

import os
import pty
import select
import subprocess
import termios
import threading
import time
from dataclasses import dataclass

import numpy as np


# Default deadline for any single readline waiting on Stockfish stdout.
# 60s is generous enough for the slowest realistic search (~5k nodes per
# our PID config completes in <100ms; even 1M-node deep searches finish
# well under this). The ceiling exists to break a stalled subprocess /
# protocol deadlock that would otherwise hang selfplay/arena while
# holding the lock (F004).
_DEFAULT_READ_TIMEOUT_S = 60.0


class StockfishTimeoutError(RuntimeError):
    """Raised when Stockfish stdout doesn't deliver a line within the deadline."""


@dataclass
class StockfishPV:
    move_uci: str
    wdl: np.ndarray | None  # (3,) float32 or None
    cp: int | None = None    # raw centipawn score (None if mate-only / unknown)
    mate: int | None = None  # raw mate-in-N (None if cp / unknown)


@dataclass
class StockfishResult:
    bestmove_uci: str
    wdl: np.ndarray | None  # (3,) float32 or None (PV1)
    pvs: list[StockfishPV]
    cp: int | None = None    # PV1 raw centipawn score
    mate: int | None = None  # PV1 raw mate-in-N


class StockfishUCI:
    def __init__(
        self,
        path: str,
        *,
        nodes: int = 2000,
        multipv: int = 1,
        hash_mb: int | None = None,
        syzygy_path: str | None = None,
        read_timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
    ):
        self.path = path
        self.nodes = int(nodes)
        self.multipv = int(multipv)
        self.hash_mb = None if hash_mb is None else max(1, int(hash_mb))
        self.syzygy_path = syzygy_path or None
        self.read_timeout_s = float(read_timeout_s)
        self._lock = threading.Lock()

  # Stockfish's stdout switches to block-buffered when stdin is a pipe,
  # which causes the `uci` response (~1.5 KB) to never reach us — the
  # buffer never fills, so we hang at "id name" and time out at 60s. A
  # pty makes Stockfish's stdout line-buffered like a terminal and avoids
  # the deadlock. Disable ECHO on the slave so input commands don't get
  # echoed back into our read stream.
        master_fd, slave_fd = pty.openpty()
        attrs = termios.tcgetattr(slave_fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
        self._tty_fd = master_fd
        self._read_buf = b""
        self.proc = subprocess.Popen(  # pylint: disable=consider-using-with  # process outlives __init__ (closed in .close())
            [self.path],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=subprocess.DEVNULL,
        )
        os.close(slave_fd)

        self._send("uci")
        self._wait_for("uciok")
        self._send("setoption name UCI_ShowWDL value true")
        self._send("setoption name Threads value 1")
        if self.hash_mb is not None:
            self._send(f"setoption name Hash value {self.hash_mb}")
        if self.syzygy_path:
            self._send(f"setoption name SyzygyPath value {self.syzygy_path}")
        if self.multipv > 1:
            self._send(f"setoption name MultiPV value {self.multipv}")
        self._send("isready")
        self._wait_for("readyok")

    def close(self) -> None:
        with self._lock:
            try:
                self._send("quit")
            except (BrokenPipeError, OSError):
                pass  # stockfish already exited
            try:
                os.close(self._tty_fd)
            except OSError:
                pass  # already closed
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass  # already exited
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # kill didn't finish before timeout — leave the zombie

    def set_nodes(self, nodes: int) -> None:
  # nodes are passed via `go nodes X`, so changing the attribute is sufficient.
        with self._lock:
            self.nodes = int(nodes)

    def _send(self, cmd: str) -> None:
        os.write(self._tty_fd, (cmd + "\n").encode("utf-8"))

    def _readline_with_deadline(self, deadline: float) -> str:
        """Blocking readline that respects ``deadline`` (monotonic seconds).

        Raises ``StockfishTimeoutError`` if no line arrives in time and
        ``RuntimeError`` if the process closed stdout. Reads from the pty
        master fd, accumulating bytes until a newline appears or the
        deadline expires (F004).
        """
        while b"\n" not in self._read_buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StockfishTimeoutError("Stockfish read deadline expired before select")
            ready, _, _ = select.select([self._tty_fd], [], [], remaining)
            if not ready:
                raise StockfishTimeoutError(
                    f"Stockfish stdout silent for {self.read_timeout_s:.1f}s"
                )
            try:
                chunk = os.read(self._tty_fd, 4096)
            except OSError as exc:
                raise RuntimeError("Stockfish process exited") from exc
            if not chunk:
                raise RuntimeError("Stockfish process exited")
            self._read_buf += chunk
        nl = self._read_buf.index(b"\n")
        line = self._read_buf[: nl + 1].decode("utf-8", errors="replace")
        self._read_buf = self._read_buf[nl + 1 :]
        return line.replace("\r\n", "\n")

    def _wait_for(self, token: str) -> None:
        deadline = time.monotonic() + self.read_timeout_s
        while True:
            line = self._readline_with_deadline(deadline)
            if token in line:
                return

    def search(self, fen: str, *, nodes: int | None = None) -> StockfishResult:
        """Node-limited search from FEN.

        If MultiPV>1 is enabled, we attempt to collect (move, wdl) pairs for the
        top lines so the caller can build a soft "SF policy" target distribution.
        """
        with self._lock:
            self._send(f"position fen {fen}")
            n = int(self.nodes) if nodes is None else int(nodes)
            self._send(f"go nodes {n}")

            bestmove = None
            wdl_pv1 = None
            cp_pv1: int | None = None
            mate_pv1: int | None = None
            pvs: dict[int, StockfishPV] = {}
            deadline = time.monotonic() + self.read_timeout_s

            while True:
                line = self._readline_with_deadline(deadline).strip()

                if line.startswith("info"):
                    parts = line.split()

  # multipv index (default 1 if absent)
                    mpv = 1
                    if "multipv" in parts:
                        try:
                            mpv = int(parts[parts.index("multipv") + 1])
                        except Exception:
                            mpv = 1

  # parse score (cp / mate) if present
                    cp_val: int | None = None
                    mate_val: int | None = None
                    if "score" in parts:
                        try:
                            score_idx = parts.index("score")
                            score_kind = parts[score_idx + 1]
                            score_arg = parts[score_idx + 2]
                            if score_kind == "cp":
                                cp_val = int(score_arg)
                            elif score_kind == "mate":
                                mate_val = int(score_arg)
                        except (ValueError, IndexError):
                            cp_val = None
                            mate_val = None

  # parse WDL if present
                    wdl_vec = None
                    if "wdl" in parts:
                        try:
                            wdl_idx = parts.index("wdl")
                            w = int(parts[wdl_idx + 1])
                            d = int(parts[wdl_idx + 2])
                            l = int(parts[wdl_idx + 3])
                            vec = np.array([w, d, l], dtype=np.float32)
                            s = float(vec.sum())
                            if s > 0:
                                wdl_vec = vec / s
                        except Exception:
                            wdl_vec = None

  # parse PV first move if present
                    pv_move = None
                    if "pv" in parts:
                        try:
                            pv_idx = parts.index("pv")
                            if pv_idx + 1 < len(parts):
                                pv_move = parts[pv_idx + 1]
                        except Exception:
                            pv_move = None

                    if mpv == 1:
                        if wdl_vec is not None:
                            wdl_pv1 = wdl_vec
                        if cp_val is not None:
                            cp_pv1 = cp_val
                        if mate_val is not None:
                            mate_pv1 = mate_val

                    if pv_move is not None:
                        pvs[mpv] = StockfishPV(
                            move_uci=pv_move, wdl=wdl_vec, cp=cp_val, mate=mate_val
                        )

                if line.startswith("bestmove"):
                    toks = line.split()
                    bestmove = toks[1] if len(toks) > 1 else None
                    break

            pv_list = [pvs[k] for k in sorted(pvs.keys())]
            return StockfishResult(
                bestmove_uci=bestmove or "0000",
                wdl=wdl_pv1,
                pvs=pv_list,
                cp=cp_pv1,
                mate=mate_pv1,
            )
