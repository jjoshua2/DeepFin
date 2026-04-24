from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class StockfishPV:
    move_uci: str
    wdl: np.ndarray | None  # (3,) float32 or None


@dataclass
class StockfishResult:
    bestmove_uci: str
    wdl: np.ndarray | None  # (3,) float32 or None (PV1)
    pvs: list[StockfishPV]


class StockfishUCI:
    def __init__(
        self,
        path: str,
        *,
        nodes: int = 2000,
        multipv: int = 1,
        hash_mb: int | None = None,
        syzygy_path: str | None = None,
    ):
        self.path = path
        self.nodes = int(nodes)
        self.multipv = int(multipv)
        self.hash_mb = None if hash_mb is None else max(1, int(hash_mb))
        self.syzygy_path = syzygy_path or None
        self._lock = threading.Lock()

        self.proc = subprocess.Popen(  # pylint: disable=consider-using-with  # process outlives __init__ (closed in .close())
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

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
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream is None:
                    continue
                try:
                    stream.close()
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
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str) -> None:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish process exited")
            if token in line:
                return

    def search(self, fen: str, *, nodes: int | None = None) -> StockfishResult:
        """Node-limited search from FEN.

        If MultiPV>1 is enabled, we attempt to collect (move, wdl) pairs for the
        top lines so the caller can build a soft "SF policy" target distribution.
        """
        with self._lock:
            assert self.proc.stdout is not None
            self._send(f"position fen {fen}")
            n = int(self.nodes) if nodes is None else int(nodes)
            self._send(f"go nodes {n}")

            bestmove = None
            wdl_pv1 = None
            pvs: dict[int, StockfishPV] = {}

            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("Stockfish process exited")
                line = line.strip()

                if line.startswith("info"):
                    parts = line.split()

  # multipv index (default 1 if absent)
                    mpv = 1
                    if "multipv" in parts:
                        try:
                            mpv = int(parts[parts.index("multipv") + 1])
                        except Exception:
                            mpv = 1

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

                    if mpv == 1 and wdl_vec is not None:
                        wdl_pv1 = wdl_vec

                    if pv_move is not None:
                        pvs[mpv] = StockfishPV(move_uci=pv_move, wdl=wdl_vec)

                if line.startswith("bestmove"):
                    toks = line.split()
                    bestmove = toks[1] if len(toks) > 1 else None
                    break

            pv_list = [pvs[k] for k in sorted(pvs.keys())]
            return StockfishResult(bestmove_uci=bestmove or "0000", wdl=wdl_pv1, pvs=pv_list)
