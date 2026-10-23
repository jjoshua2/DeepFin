"""Owned strict Syzygy handles with serialized high-level probes."""
from __future__ import annotations

import threading

import chess
import chess.syzygy


class StrictMatchTablebase(chess.syzygy.Tablebase):
    """Keep complete shared-handle probes outside the reader/eviction cycle.

    python-chess counts a table reader before acquiring its initialization
    lock; an LRU close can hold that lock while waiting for that reader. The
    reentrant outer gate prevents different high-level probes from entering
    that cycle, and keeps a match decision's WDL/DTZ observations together.
    Opening remains owner-only. Close still belongs to the owner after all
    producer work has joined; the gate does not establish worker quiescence.
    """

    def __init__(self) -> None:
        super().__init__()
        self._match_probe_lock = threading.RLock()

    def probe_wdl(self, board: chess.Board) -> int:
        with self._match_probe_lock:
            return super().probe_wdl(board)

    def probe_dtz(self, board: chess.Board) -> int:
        with self._match_probe_lock:
            return super().probe_dtz(board)

    def probe_wdl_dtz(self, board: chess.Board) -> tuple[int, int]:
        """Preserve serial conversion/error order while holding one gate."""
        with self._match_probe_lock:
            return int(self.probe_wdl(board)), int(self.probe_dtz(board))

    def close(self) -> None:
        with self._match_probe_lock:
            super().close()
