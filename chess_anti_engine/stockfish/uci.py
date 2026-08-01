from __future__ import annotations

import os
import pty
import select
import subprocess
import termios
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


# Default deadline for any single readline waiting on Stockfish stdout.
# 60s is generous enough for the slowest realistic search (~5k nodes per
# our PID config completes in <100ms; even 1M-node deep searches finish
# well under this). The ceiling exists to break a stalled subprocess /
# protocol deadlock that would otherwise hang selfplay/arena while
# holding the lock (F004).
_DEFAULT_READ_TIMEOUT_S = 60.0


def _stockfish_child_nice(current_nice: int, configured_nice: int) -> int:
    """Return the absolute child nice value without raising process priority."""
    target_nice = min(19, max(0, int(configured_nice)))
    return max(int(current_nice), target_nice)


class StockfishTimeoutError(RuntimeError):
    """Raised when Stockfish stdout doesn't deliver a line within the deadline."""


class StockfishDesyncError(RuntimeError):
    """Raised when a protocol section was abandoned and the stream is unsafe.

    A search that raises part-way (deadline expiry is the realistic one) leaves
    the engine still calculating: its ``info``/``bestmove`` lines arrive AFTER
    we stopped reading and sit in the pty buffer. The next ``search`` sends its
    own ``position``/``go`` and then reads — and the first ``bestmove`` it sees
    is the ABANDONED query's. The engine is silently one search behind from
    then on, permanently, and every result it returns is a real Stockfish
    search of the WRONG position. That failure is invisible downstream: the
    label's own legality mask, node count and depth are all computed by the
    caller or by the engine's genuine work, so it reads as a sane label.

    So an abandoned protocol section poisons the engine: every later protocol
    call raises this instead of returning a stale result. Callers must replace
    the engine (``StockfishPool`` does this automatically). Wrong-and-silent is
    downgraded to absent-and-loud, which is the only safe direction here.
    """


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
    nodes: int | None = None  # nodes searched (last info line)
    depth: int | None = None  # depth reached (last info line)


def _normalize_wdl_counts(counts: tuple[int, int, int] | None) -> np.ndarray | None:
    if counts is None:
        return None
    vec = np.asarray(counts, dtype=np.float32)
    total = float(vec.sum())
    return vec / total if total > 0.0 else None


def _parse_info_fields(
    parts: list[str],
) -> tuple[
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    tuple[int, int, int] | None,
    str | None,
]:
    """Parse the UCI fields selfplay retains with one token scan.

    WDL stays as integer permille counts. Search normalizes only the final
    retained line for each MultiPV rank instead of allocating a NumPy vector
    at every intermediate depth.
    """
    mpv = nodes = depth = cp = mate = None
    wdl_counts: tuple[int, int, int] | None = None
    pv_move = None
    idx = 0
    n_parts = len(parts)
    while idx < n_parts:
        token = parts[idx]
        try:
            if token == "multipv" and idx + 1 < n_parts:
                mpv = int(parts[idx + 1])
                idx += 2
                continue
            if token == "nodes" and idx + 1 < n_parts:
                nodes = int(parts[idx + 1])
                idx += 2
                continue
            if token == "depth" and idx + 1 < n_parts:
                depth = int(parts[idx + 1])
                idx += 2
                continue
            if token == "score" and idx + 2 < n_parts:
                score_kind = parts[idx + 1]
                score_arg = int(parts[idx + 2])
                if score_kind == "cp":
                    cp = score_arg
                elif score_kind == "mate":
                    mate = score_arg
                idx += 3
                continue
            if token == "wdl" and idx + 3 < n_parts:
                wdl_counts = (
                    int(parts[idx + 1]),
                    int(parts[idx + 2]),
                    int(parts[idx + 3]),
                )
                idx += 4
                continue
            if token == "pv" and idx + 1 < n_parts:
                pv_move = parts[idx + 1]
                break
        except ValueError:
            # Preserve the tolerant legacy parser: one malformed optional
            # field does not discard other usable fields on the info line.
            pass
        idx += 1
    return mpv, nodes, depth, cp, mate, wdl_counts, pv_move


@dataclass
class _SearchInfoAccumulator:
    """Latest UCI search fields, retaining raw WDL until final materialization."""

    wdl_pv1_counts: tuple[int, int, int] | None = None
    cp_pv1: int | None = None
    mate_pv1: int | None = None
    nodes_seen: int | None = None
    depth_seen: int | None = None
    pvs: dict[
        int,
        tuple[str, tuple[int, int, int] | None, int | None, int | None],
    ] = field(default_factory=dict)

    def consume(self, parts: list[str]) -> None:
        mpv_val, nodes, depth, cp, mate, wdl_counts, pv_move = _parse_info_fields(parts)
        mpv = mpv_val or 1
        if nodes is not None:
            self.nodes_seen = nodes
        if depth is not None:
            self.depth_seen = max(self.depth_seen or 0, depth)
        if mpv == 1:
            if wdl_counts is not None:
                self.wdl_pv1_counts = wdl_counts
            if cp is not None:
                self.cp_pv1 = cp
                self.mate_pv1 = None
            if mate is not None:
                self.mate_pv1 = mate
                self.cp_pv1 = None
        if pv_move is not None:
            self.pvs[mpv] = (pv_move, wdl_counts, cp, mate)

    def result(self, bestmove: str | None) -> StockfishResult:
        pv_list = [
            StockfishPV(
                move_uci=self.pvs[k][0],
                wdl=_normalize_wdl_counts(self.pvs[k][1]),
                cp=self.pvs[k][2],
                mate=self.pvs[k][3],
            )
            for k in sorted(self.pvs)
        ]
        return StockfishResult(
            bestmove_uci=bestmove or "0000",
            wdl=_normalize_wdl_counts(self.wdl_pv1_counts),
            pvs=pv_list,
            cp=self.cp_pv1,
            mate=self.mate_pv1,
            nodes=self.nodes_seen,
            depth=self.depth_seen,
        )


class StockfishUCI:
    def __init__(
        self,
        path: str,
        *,
        nodes: int = 2000,
        multipv: int = 1,
        hash_mb: int | None = None,
        syzygy_path: str | None = None,
        nice: int = 0,
        read_timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
    ):
        self.path = path
        self.nodes = int(nodes)
        self.multipv = int(multipv)
        self.hash_mb = None if hash_mb is None else max(1, int(hash_mb))
        self.syzygy_path = syzygy_path or None
        self.nice = min(19, max(0, int(nice)))
        self.read_timeout_s = float(read_timeout_s)
        self._lock = threading.Lock()
        # Set by _protocol_section when a command was sent but its terminating
        # token was never consumed. Read without the lock (a single bool store)
        # so a poisoned engine is reported even while another thread holds it.
        self._desynced = False

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
        self.proc = subprocess.Popen(  # process outlives __init__ (closed in .close())
            [self.path],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=subprocess.DEVNULL,
        )
        if self.nice > 0:
            current_nice = os.getpriority(os.PRIO_PROCESS, self.proc.pid)
            target_nice = _stockfish_child_nice(current_nice, self.nice)
            if target_nice > current_nice:
                os.setpriority(os.PRIO_PROCESS, self.proc.pid, target_nice)
        os.close(slave_fd)

        try:
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
        except BaseException:
            self.close()
            raise

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

    @property
    def desynced(self) -> bool:
        """Whether a protocol exchange was abandoned and the stream is unsafe."""
        return self._desynced

    @contextmanager
    def _protocol_section(self) -> Generator[None]:
        """Guard one send-then-read-to-terminator exchange.

        Refuses to start on a poisoned engine, and poisons the engine if the
        exchange is abandoned part-way — see ``StockfishDesyncError``.

        ``BaseException``, not ``Exception``: a ``CancelledError`` (Python 3.8+
        derives it from ``BaseException``) or a ``KeyboardInterrupt`` leaves the
        very same unread ``bestmove`` behind as a deadline expiry does.

        A raise before the first byte is written (a malformed FEN rejected by
        ``_send``) also poisons, which is stricter than strictly necessary. That
        direction is deliberate: the cost is one engine restart, and the
        alternative is reasoning per call site about how far the exchange got.
        """
        if self._desynced:
            raise StockfishDesyncError(
                "this engine abandoned an earlier protocol exchange and is one "
                "search behind; replace the process instead of reusing it",
            )
        try:
            yield
        except BaseException:
            self._desynced = True
            raise

    def set_nodes(self, nodes: int) -> None:
  # nodes are passed via `go nodes X`, so changing the attribute is sufficient.
        with self._lock:
            self.nodes = int(nodes)

    def new_game(self) -> None:
        """Clear the transposition table (`ucinewgame`) so the next search is a
        COLD, independent search — needed when re-evaluating one position at
        several node budgets, else the earlier search's hash warm-starts the
        later ones and overstates their agreement (Codex #125)."""
        with self._lock, self._protocol_section():
            self._new_game_locked()

    def _new_game_locked(self) -> None:
        """`ucinewgame` + readyok handshake; caller must hold ``self._lock``."""
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def _set_syzygy_path_locked(self, syzygy_path: str) -> None:
        if str(syzygy_path) == str(self.syzygy_path or ""):
            return
        self._send(f"setoption name SyzygyPath value {syzygy_path}")
        self._send("isready")
        self._wait_for("readyok")
        self.syzygy_path = str(syzygy_path)

    def _send(self, cmd: str) -> None:
        if "\n" in cmd or "\r" in cmd:
            # A stray newline (e.g. from a corrupted FEN) would silently
            # desync the UCI session; fail loud instead.
            raise ValueError(f"UCI command contains a newline: {cmd!r}")
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
            # uciok/readyok are standalone engine->GUI messages in the UCI
            # protocol, so require the whole line: an `info string` line that
            # merely mentions the token must not satisfy the handshake.
            if line.strip() == token:
                return

    def search(
        self, fen: str, *, nodes: int | None = None,
        syzygy_path: str | None = None,
        fresh: bool = False,
    ) -> StockfishResult:
        """Node-limited search from FEN.

        If MultiPV>1 is enabled, we attempt to collect (move, wdl) pairs for the
        top lines so the caller can build a soft "SF policy" target distribution.

        ``fresh`` clears the transposition table (``ucinewgame``) first so this
        is a COLD, independent search. Required when re-querying a position the
        engine may have just searched at a shallower budget (label escalation):
        a warm TT would let the shallow pass steer the deep one and overstate
        their agreement — the same hygiene scripts/blindspot_deepsf_gate.py
        applies between admission verdicts.
        """
        with self._lock, self._protocol_section():
            if fresh:
                self._new_game_locked()
            if syzygy_path is not None:
                self._set_syzygy_path_locked(str(syzygy_path))
            self._send(f"position fen {fen}")
            n = int(self.nodes) if nodes is None else int(nodes)
            self._send(f"go nodes {n}")

            bestmove = None
            info = _SearchInfoAccumulator()
            deadline = time.monotonic() + self.read_timeout_s

            while True:
                line = self._readline_with_deadline(deadline).strip()

                if line.startswith("info"):
                    info.consume(line.split())

                if line.startswith("bestmove"):
                    toks = line.split()
                    bestmove = toks[1] if len(toks) > 1 else None
                    break

            return info.result(bestmove)
