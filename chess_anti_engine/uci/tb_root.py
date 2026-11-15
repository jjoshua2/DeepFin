"""UCI root Syzygy shortcut on the vendored C prober (Fathom, ``tb._tb_ext``).

The engine's root DTZ shortcut used to probe every child through python-chess
(``tablebase.try_tb_root_move``). It now asks Fathom's ``probe_root`` for the
per-move DTZ of every legal root move in one C call, then applies the repo's
own move-choice rules on top, unchanged:

* the claim rules of ``tablebase._dtz_of_played_move``: a mating move is
  distance 1, and a move into stalemate, insufficient material, threefold
  repetition (read from the board's move stack) or -- with
  ``Syzygy50MoveRule`` on -- a run-out 50-move counter is distance 0. Fathom
  knows none of these claims (docs/tablebase_prober.md, "the one root
  difference"), so they are decided here, from the python-chess board;
* the ranking of ``tablebase._select_dtz_move``: ``_rule50_dtz_key`` (or
  ``_plain_dtz_key`` with the rule off) over python-chess's legal-move order,
  first best move wins, so ties break exactly as before;
* the reported WDL of ``tablebase.try_tb_root_move``.

Only the table lookups moved. python-chess is still the move generator and the
claim oracle here; ``chess.syzygy`` is not used on this path.

Fathom's table set is one per process (``_tb_ext`` is a singleton). This module
owns it for the engine: :func:`load` (re)initialises it, :func:`unload` frees
it, and every :class:`RootTablebase` handle remembers the generation it was
built for, so a handle that outlived a re-init refuses to probe instead of
answering from somebody else's tables. ``_tb_ext.probe_root`` serialises
concurrent calls itself (a C mutex); the engine probes only from its single
search thread and changes options only between searches.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import chess

from chess_anti_engine.tablebase import (
    _plain_dtz_key,
    _rule50_dtz_key,
    _wdl_adjusted_for_rule50,
    is_tb_eligible,
)
from chess_anti_engine.tb import _tb_ext

_log = logging.getLogger(__name__)

#: What the engine's load line calls this prober (asserted by the tests).
PROBER_NAME = "C Fathom"


class RootTablebaseError(RuntimeError):
    """The C prober cannot serve the configured SyzygyPath."""


@dataclass(frozen=True)
class LoadedTables:
    """One successful ``_tb_ext.init``: what was asked for and what loaded."""

    configured: str
    resolved: str
    largest: int
    n_wdl: int
    n_dtz: int
    largest_dtz: int
    generation: int

    def describe(self) -> str:
        return (f"{PROBER_NAME} ({_tb_ext.FATHOM_COMMIT[:8]}): {self.n_wdl} WDL + "
                f"{self.n_dtz} DTZ tables, WDL up to {self.largest}-man, "
                f"DTZ up to {self.largest_dtz}-man")


_state_lock = threading.Lock()
_loaded: LoadedTables | None = None
_generation = 0


def normalise_path(path: str) -> str:
    """The component list ``tablebase.get_tablebase`` reads, joined for Fathom.

    Components are split on ``os.pathsep`` and stripped, and empty ones are
    dropped, exactly as the python-chess opener the leaf probes use does, so
    both probers are given the same directories.
    """
    return os.pathsep.join(p.strip() for p in path.split(os.pathsep) if p.strip())


def loaded() -> LoadedTables | None:
    with _state_lock:
        return _loaded


def load(path: str) -> LoadedTables:
    """(Re)initialise the process's C table set from ``path``.

    Raises :class:`RootTablebaseError` when a component is not a readable
    directory or nothing loads; the C table set is then empty (never a partial
    or stale set the caller did not ask for). Re-loading the path that is
    already loaded keeps the open tables.
    """
    global _loaded, _generation
    wanted = normalise_path(path)
    if not wanted:
        raise RootTablebaseError(f"SyzygyPath {path!r} names no directory")
    with _state_lock:
        if _loaded is not None and _loaded.configured == wanted and _tb_ext.path() == _loaded.resolved:
            return _loaded
        _generation += 1
        _loaded = None
        try:
            largest = int(_tb_ext.init(wanted))
        except (ValueError, RuntimeError) as exc:
            _tb_ext.free()
            raise RootTablebaseError(str(exc)) from exc
        resolved = _tb_ext.path()
        if resolved is None or largest <= 0:  # init raises first; belt and braces
            _tb_ext.free()
            raise RootTablebaseError(f"no Syzygy tables found under {wanted}")
        n_wdl, n_dtz, largest_dtz = _tb_ext.table_counts()
        _loaded = LoadedTables(
            configured=wanted, resolved=resolved, largest=largest,
            n_wdl=int(n_wdl), n_dtz=int(n_dtz), largest_dtz=int(largest_dtz),
            generation=_generation,
        )
        return _loaded


def unload() -> None:
    """Free the C table set (``SyzygyPath`` cleared, or the engine closing)."""
    global _loaded, _generation
    with _state_lock:
        _generation += 1
        _loaded = None
        _tb_ext.free()


def _current_generation() -> int:
    with _state_lock:
        return _generation


def _claim_dtz(child: chess.Board, *, rule50: bool) -> int | None:
    """``tablebase._dtz_of_played_move``'s short-circuits, or None for "ask the table"."""
    if child.is_checkmate():
        return 1
    if (
        child.is_stalemate()
        or child.is_insufficient_material()
        or (rule50 and child.is_fifty_moves())
        or child.is_repetition(3)
    ):
        return 0
    return None


class RootTablebase:
    """The root DTZ shortcut for one engine configuration.

    ``rule50`` is UCI ``Syzygy50MoveRule``; the engine builds a new handle
    whenever that or ``SyzygyPath`` changes.
    """

    def __init__(self, tables: LoadedTables, *, rule50: bool) -> None:
        self.tables = tables
        self.rule50 = bool(rule50)
        self.root_probes = 0  # C probe_root calls made through this handle

    @property
    def prober(self) -> str:
        return PROBER_NAME

    def _dtz_by_move(self, board: chess.Board) -> dict[str, int] | None:
        if self.tables.generation != _current_generation() or _tb_ext.path() != self.tables.resolved:
            raise RootTablebaseError(
                "the C Syzygy table set was re-initialised after this root handle was built"
            )
        self.root_probes += 1
        try:
            rows = _tb_ext.probe_root_fen(board.fen(), has_repeated=False, use_rule50=True)
        except ValueError as exc:
            # docs/tablebase_prober.md, wiring rule 1: an input the C prober
            # refuses (an illegal position or an impossible en-passant square a
            # GUI can still send) is "no TB" -- the search answers it.
            _log.debug("C root probe refused %s: %s", board.fen(), exc)
            return None
        if rows is None:
            return None
        return {uci: int(dtz) for uci, dtz, _rank, _score in rows}

    def move_dtz(self, board: chess.Board) -> list[tuple[chess.Move, int]] | None:
        """Every legal move with its DTZ from the mover's view, in python-chess
        legal-move order: ``tablebase._dtz_of_played_move`` per child, with the
        table distance from C. None when the root is not answerable (the same
        cases ``tablebase._select_dtz_move`` returns None for)."""
        if not is_tb_eligible(board):
            return None
        if board.is_game_over(claim_draw=False):
            return None
        if chess.popcount(board.occupied) > self.tables.largest:
            return None
        dtz_by_move = self._dtz_by_move(board)
        if dtz_by_move is None:
            return None
        legal = list(board.legal_moves)
        if len(legal) != len(dtz_by_move):
            _log.warning(
                "C root probe returned %d moves for %d legal moves at %s; no TB answer",
                len(dtz_by_move), len(legal), board.fen(),
            )
            return None
        out: list[tuple[chess.Move, int]] = []
        for move in legal:
            table_dtz = dtz_by_move.get(move.uci())
            if table_dtz is None:
                _log.warning("C root probe lacks legal move %s at %s; no TB answer",
                             move.uci(), board.fen())
                return None
            board.push(move)
            try:
                claimed = _claim_dtz(board, rule50=self.rule50)
            finally:
                board.pop()
            out.append((move, table_dtz if claimed is None else claimed))
        return out

    def select(self, board: chess.Board) -> tuple[chess.Move, int] | None:
        """``tablebase._select_dtz_move`` on C per-move DTZ: (best move, its DTZ)."""
        moves = self.move_dtz(board)
        if moves is None:
            return None
        halfmove = board.halfmove_clock
        best_move: chess.Move | None = None
        best_key: tuple[int, int] | None = None
        best_dtz = 0
        for move, dtz in moves:
            key = _rule50_dtz_key(dtz, halfmove) if self.rule50 else _plain_dtz_key(dtz)
            if best_key is None or key > best_key:
                best_key = key
                best_move = move
                best_dtz = dtz
        if best_move is None:
            return None
        return best_move, best_dtz

    def root_move(self, board: chess.Board) -> tuple[chess.Move, int] | None:
        """``tablebase.try_tb_root_move`` semantics: (DTZ-optimal move, WDL of it).

        WDL is from the side to move: with the 50-move rule, +2/-2 only while
        the chosen line still wins/loses at this counter and +1/-1 for a result
        the rule draws; without it, +2/-2 for any win/loss and 0 for a draw.
        """
        picked = self.select(board)
        if picked is None:
            return None
        move, dtz = picked
        if not self.rule50:
            return move, (2 if dtz > 0 else -2 if dtz < 0 else 0)
        child = board.copy(stack=False)
        child.push(move)
        counter = 0 if child.halfmove_clock == 0 or child.is_checkmate() else board.halfmove_clock
        return move, _wdl_adjusted_for_rule50(dtz, counter)
