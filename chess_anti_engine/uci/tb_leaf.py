"""UCI search-leaf and root-eval Syzygy WDL overrides on the C prober (Fathom).

The UCI engine's in-search probe used to be ``tablebase.SyzygyProbe``: per
eligible leaf it built ``chess.Board(cb.fen())`` and asked python-chess's
``Tablebase.probe_wdl``, on its own python-chess table handle beside the C one
the root shortcut uses. :class:`CLeafProbe` is the same probe -- same
interface, same eligibility, same value written into the leaf, same
``hits``/``probes`` counting -- answered by ``_tb_ext.probe_wdl_batch`` on the
ONE table set ``uci/tb_root.py`` loaded. Selfplay, arena and the worker still
construct the python-chess ``SyzygyProbe``; nothing here is imported by them.

What is unchanged, by construction (``apply`` mirrors ``SyzygyProbe.apply``
line for line, docs/tablebase_prober.md "UCI leaf probe"):

* eligibility: with ``indices`` the caller pre-filtered (the C tree's
  ``get_pending_tb_leaves``, or the walker / PUCV / RPG clock-0 rule in
  ``mcts/puct_tb.py`` and ``uci/walker_pool.py`` -- untouched); without them
  (the root eval) a leaf is probed when it has no castling rights and at most
  ``max_pieces`` men, exactly as before. ``max_pieces`` is the largest WDL
  table's man count, which is what ``SyzygyProbe`` derived from its keys;
* the value: WIN/DRAW/LOSS one-hot logits, cursed wins / blessed losses
  counted as draws under ``Syzygy50MoveRule`` (``cursed_as_draw``) and as
  decisive without it; the halfmove clock is not read by either prober;
* "no TB answer" (the leaf keeps the net's WDL and counts as a probe, not a
  hit): every case where python-chess raised (castling rights, a missing
  table, an illegal position) is a C ``None`` / refusal here. The cases where
  python-chess answered and C does not are listed in the doc; they become
  "no TB answer", never a fallback to python-chess.

Lifetime. ``probe_wdl_batch`` takes the generation of the table set this probe
was built for and refuses (returns None, nothing probed) once ``init``/``free``
has replaced it; the check and the probes run under the same shared lock, and
``init``/``free`` take that lock exclusively, so a leaf probe running on a pool
thread -- including a worker the engine abandoned after a device fault -- can
neither touch unmapped tables nor be answered from a set it was not built for.
A stale probe answers nothing and says so once; ``SearchWorker.run`` drops it
before the next search (``is_current``).
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.tablebase import (
    _DRAW_LOGITS,
    _LOSS_LOGITS,
    _WIN_LOGITS,
    SyzygyProbe,
)
from chess_anti_engine.tb import _tb_ext
from chess_anti_engine.uci.tb_root import PROBER_NAME, LoadedTables

_log = logging.getLogger(__name__)

#: ``probe_wdl_batch`` result codes above the WDL range (0..4 = WDL -2..2).
_CODE_MAX_WDL = 4


class CLeafProbe(SyzygyProbe):
    """``SyzygyProbe``'s interface, answered by the C prober's loaded set.

    A subclass only so every search path's ``SyzygyProbe`` annotation accepts
    it; ``SyzygyProbe.__init__`` is deliberately NOT called (it opens a
    python-chess table handle) and every method the search uses is defined
    here. ``rule50_aware`` (selfplay's strict mode) is not offered: the UCI
    engine never used it.
    """

    def __init__(self, tables: LoadedTables, *, cursed_as_draw: bool) -> None:  # pyright: ignore[reportMissingSuperCall]
        # The attributes SyzygyProbe's own methods and callers read.
        self._path = tables.configured
        self._cursed_as_draw = bool(cursed_as_draw)
        self._rule50_aware = False
        self._tablebase = None
        self.tables = tables
        self.n_wdl = tables.n_wdl
        self.n_dtz = tables.n_dtz
        self.max_pieces = tables.largest_wdl
        self.hits = 0
        self.probes = 0
        #: Leaf batches refused because the table set was replaced under us.
        self.stale_batches = 0
        # Walker / pool threads share one probe; the GIL is released inside
        # probe_wdl_batch, so the counters get a lock of their own.
        self._count_lock = threading.Lock()

    @property
    def prober(self) -> str:
        return PROBER_NAME

    def is_current(self) -> bool:
        """True while the C table set is still the one this probe was built for."""
        return int(_tb_ext.generation()) == self.tables.c_generation

    def reset_counts(self) -> None:
        with self._count_lock:
            self.hits = 0
            self.probes = 0

    def apply(
        self,
        leaf_cboards: list[CBoard],
        wdl: np.ndarray,
        indices: np.ndarray | None = None,
        solved_out: np.ndarray | None = None,
    ) -> int:
        """``SyzygyProbe.apply``: overwrite each answered leaf's wdl row.

        ``indices`` given: the leaves are pre-filtered and ``indices[j]`` is
        the row of ``wdl`` for ``leaf_cboards[j]``. ``indices`` None: each
        leaf is checked here (castling rights, more than ``max_pieces`` men)
        and written to row ``j``. ``solved_out[j]`` receives 1 / -1 / 2
        (SOLVED_WIN / LOSS / DRAW) for each answered leaf.
        """
        if not leaf_cboards:
            return 0
        if indices is None:
            max_pieces = self.max_pieces
            picked: Sequence[int] = [
                j for j, cb in enumerate(leaf_cboards)
                if (int(cb.occ_white) | int(cb.occ_black)).bit_count() <= max_pieces
                and int(cb.castling) == 0
            ]
            if not picked:
                return 0
            boards: Sequence[CBoard] = [leaf_cboards[j] for j in picked]
        else:
            picked = range(len(leaf_cboards))
            boards = leaf_cboards
        codes = _tb_ext.probe_wdl_batch(boards, self.tables.c_generation)
        if codes is None:
            with self._count_lock:
                self.probes += len(picked)
                self.stale_batches += 1
                first = self.stale_batches == 1
            if first:
                _log.warning(
                    "C Syzygy table set was re-initialised after this leaf probe was built "
                    "(%s); its leaves get no tablebase answer", self.tables.configured,
                )
            return 0
        # Decisive threshold: with the 50-move rule only +-2 is a result,
        # without it cursed wins / blessed losses (+-1) count too.
        win_thresh = 2 if self._cursed_as_draw else 1
        hits = 0
        for k, j in enumerate(picked):
            code = codes[k]
            if code > _CODE_MAX_WDL:
                continue  # None / refused: no TB answer, the net's WDL stays
            value = code - 2
            i = j if indices is None else int(indices[j])
            if value >= win_thresh:
                wdl[i] = _WIN_LOGITS
                if solved_out is not None:
                    solved_out[j] = 1  # SOLVED_WIN
            elif value <= -win_thresh:
                wdl[i] = _LOSS_LOGITS
                if solved_out is not None:
                    solved_out[j] = -1  # SOLVED_LOSS
            else:
                wdl[i] = _DRAW_LOGITS
                if solved_out is not None:
                    solved_out[j] = 2  # SOLVED_DRAW
            hits += 1
        with self._count_lock:
            self.probes += len(picked)
            self.hits += hits
        return hits
