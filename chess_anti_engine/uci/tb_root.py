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

Two documented differences from ``try_tb_root_move`` (both make the engine
SEARCH a root python-chess answered; neither changes an answered move):

* a root with more men than the loaded tables (e.g. a 7-man root with only
  6-man tables, ``8/8/8/2br4/3P2k1/4B3/2p5/4K3 w - - 95 40``) is not
  answered: python-chess sometimes resolved one through a capture into the
  tables, the C prober never answers above ``largest()``;
* a root whose material has a WDL table but no DTZ table (e.g. a 6-man root
  with only the first directory of the production pair,
  ``7n/8/4k3/8/6P1/5p1K/1p6/8 w - - 5 40``) is not answered, while
  python-chess could still rank it when every child resolved through zeroing
  moves. The engine warns at load when WDL tables lack DTZ.

Only the table lookups moved. python-chess is still the move generator and the
claim oracle here; ``chess.syzygy`` is not used on this path.

Fathom's table set is one per process (``_tb_ext`` is a singleton). This module
owns it for the engine: :func:`load` (re)initialises it, :func:`unload` frees
it. Engines sharing one ``SyzygyPath`` share one generation, so each engine
loads and releases as an OWNER (``owner=``): the set is freed only when its
last owner releases it, never under another engine still using it. Every :class:`RootTablebase` handle remembers the generation it was
built for, so a handle that outlived a re-init refuses to probe instead of
answering from somebody else's tables. ``_tb_ext.probe_root`` serialises
concurrent calls itself (a C mutex); the engine probes only from its single
search thread and changes options only between searches.
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
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
# ``id(owner)`` of every engine holding the loaded generation (see ``load``).
_owners: set[int] = set()


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


def _table_files(path: str) -> dict[str, list[str]]:
    """Every ``.rtbw`` / ``.rtbz`` file name in ``path``'s directories -> its full paths."""
    files: dict[str, list[str]] = {}
    for part in path.split(os.pathsep):
        try:
            names = sorted(os.listdir(part))
        except OSError:
            continue
        for name in names:
            if name.endswith((".rtbw", ".rtbz")):
                files.setdefault(name, []).append(os.path.join(part, name))
    return files


#: Incomplete-size files named before "and N more" (a truncated download can
#: drop hundreds; the fix for all of them is the same re-download).
_INCOMPLETE_SHOWN = 20


def _why_dropped(full: str) -> tuple[str, str]:
    """(category, reason) for a table file the C prober did not register."""
    import chess.syzygy

    name = os.path.basename(full)
    stem = name.rsplit(".", 1)[0]
    if os.path.islink(full) and not os.path.exists(full):
        return "stray", f"{full} (broken symlink -> {os.readlink(full)})"
    if not chess.syzygy.is_tablename(stem):
        try:
            canonical = chess.syzygy.normalize_tablename(stem)
        except (ValueError, KeyError, IndexError):
            canonical = None
        if canonical and canonical != stem and chess.syzygy.is_tablename(canonical):
            return "stray", f"{full} (mirrored name; the table is {canonical}{name[len(stem):]})"
        return "stray", f"{full} (not a Syzygy table name)"
    try:
        size = os.path.getsize(full)
    except OSError as exc:
        return "unreadable", f"{full} ({exc.strerror or exc})"
    if size % 64 != 16:
        return "incomplete", f"{full} ({size} bytes, not 16 mod 64: incomplete)"
    return "unreadable", f"{full} (not opened by the prober: unreadable?)"


def _refusal(what: str, names: list[str]) -> str:
    shown = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")
    return f"{len(names)} table file(s) {what}: {shown}"


def validate_loaded(
    path: str, progress: Callable[[str], None] | None = None,
) -> list[str]:
    """Initialise EVERY loaded table now and list what is wrong with the set.

    Fathom initialises a table on the first probe of its material, so a
    corrupt file used to fail (or crash) in the middle of a game. Here each
    WDL and DTZ file is parsed up front (``_tb_ext.validate_table``, with the
    bounds checks of the local Fathom patch). ``progress`` is called with each
    table's name before it is initialised, so a caller running this in a
    child process knows which table a crash happened on.

    Also lists table files present in the directories that Fathom did not
    register (it drops a file whose size is not 16 mod 64 with only an
    ``info string Incomplete tablebase file`` on stdout).
    """
    problems: list[str] = []
    names = _tb_ext.table_names()
    corrupt: list[str] = []
    for i, (name, _has_dtz) in enumerate(names):
        if progress is not None:
            progress(name)
        failed = _tb_ext.validate_table(i)
        if failed is not None:
            corrupt.append(failed)
    if corrupt:
        problems.append(_refusal("are corrupt (their headers do not parse)", corrupt))
    files = _table_files(path)
    loaded_wdl = {name for name, _ in names}
    loaded_dtz = {name for name, has_dtz in names if has_dtz}
    dropped: list[str] = []
    for fname, paths in sorted(files.items()):
        stem, ext = fname.rsplit(".", 1)
        if (ext == "rtbw" and stem not in loaded_wdl) or (
                ext == "rtbz" and stem in loaded_wdl and stem not in loaded_dtz):
            dropped += paths
        elif ext == "rtbz" and stem not in loaded_wdl:
            # A DTZ file with no WDL partner: Fathom never opens it. A stray
            # name or a broken link is still the operator's to fix.
            dropped += [p for p in paths if _why_dropped(p)[0] == "stray"]
    by_kind: dict[str, list[str]] = {}
    for full in dropped:
        kind, reason = _why_dropped(full)
        by_kind.setdefault(kind, []).append(reason)
    # Every stray file is named: the operator must act on each one.
    if by_kind.get("stray"):
        stray = by_kind["stray"]
        problems.append(
            f"{len(stray)} file(s) in the table directories are not loadable Syzygy tables; "
            f"remove or rename each one (all tablebases stay OFF until then): "
            + ", ".join(stray))
    if by_kind.get("incomplete"):
        bad = by_kind["incomplete"]
        shown = ", ".join(bad[:_INCOMPLETE_SHOWN]) + (
            f" and {len(bad) - _INCOMPLETE_SHOWN} more" if len(bad) > _INCOMPLETE_SHOWN else "")
        problems.append(f"{len(bad)} table file(s) were not loaded by the C prober (incomplete "
                        f"download; re-download or remove each): {shown}")
    if by_kind.get("unreadable"):
        bad = by_kind["unreadable"]
        problems.append(f"{len(bad)} table file(s) were not loaded by the C prober; fix the "
                        f"permissions, or remove or rename each: " + ", ".join(bad))
    return problems


def load(path: str, *, owner: object | None = None) -> LoadedTables:
    """(Re)initialise the process's C table set from ``path`` and validate it.

    Raises :class:`RootTablebaseError` when a component is not a readable
    directory, nothing loads, a table file is corrupt or a table file in the
    directories was not loaded (:func:`validate_loaded`); the C table set is
    then empty (never a partial or stale set the caller did not ask for).
    Re-loading the path that is already loaded keeps the open tables.

    ``owner`` (an engine) is recorded as holding the returned generation; it
    must hand the same object to :func:`unload`. Loading a DIFFERENT path
    re-initialises the one C table set: every earlier owner's handles go stale
    (they refuse to probe) and their releases become no-ops.
    """
    global _loaded, _generation, _owners
    wanted = normalise_path(path)
    if not wanted:
        raise RootTablebaseError(f"SyzygyPath {path!r} names no directory")
    with _state_lock:
        if _loaded is not None and _loaded.configured == wanted and _tb_ext.path() == _loaded.resolved:
            if owner is not None:
                _owners.add(id(owner))
            return _loaded
        _generation += 1
        _loaded = None
        _owners = set()
        try:
            largest = int(_tb_ext.init(wanted))
        except (ValueError, RuntimeError) as exc:
            _tb_ext.free()
            raise RootTablebaseError(str(exc)) from exc
        resolved = _tb_ext.path()
        if resolved is None or largest <= 0:  # init raises first; belt and braces
            _tb_ext.free()
            raise RootTablebaseError(f"no Syzygy tables found under {wanted}")
        problems = validate_loaded(wanted)
        if problems:
            _tb_ext.free()
            raise RootTablebaseError("; ".join(problems))
        n_wdl, n_dtz, largest_dtz = _tb_ext.table_counts()
        _loaded = LoadedTables(
            configured=wanted, resolved=resolved, largest=largest,
            n_wdl=int(n_wdl), n_dtz=int(n_dtz), largest_dtz=int(largest_dtz),
            generation=_generation,
        )
        if owner is not None:
            _owners.add(id(owner))
        return _loaded


def unload(generation: int | None = None, *, owner: object | None = None) -> bool:
    """Free the C table set (``SyzygyPath`` cleared, or the engine closing).

    With ``generation``, only when that is still the loaded table set: an
    engine releasing ITS tables must not free a set another engine in the same
    process loaded after it. With ``owner`` too, that owner is released and the
    set is freed only if no other owner still holds it (two engines on one
    ``SyzygyPath``: the first to close must not take the second's tables).
    Without ``generation`` the set is freed unconditionally. Returns whether
    anything was freed.
    """
    global _loaded, _generation, _owners
    with _state_lock:
        if generation is not None and (_loaded is None or _loaded.generation != generation):
            return False
        if generation is not None and owner is not None:
            _owners.discard(id(owner))
            if _owners:
                return False
        _owners = set()
        _generation += 1
        _loaded = None
        _tb_ext.free()
        return True


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


#: Line prefixes of :func:`_validate_main`'s protocol (read by uci/selfcheck.py).
VALIDATE_START = "TBVALIDATE table "
VALIDATE_OK = "TBVALIDATE ok "
VALIDATE_FAIL = "TBVALIDATE fail "


def _validate_main(argv: list[str]) -> int:
    """``python -m chess_anti_engine.uci.tb_root <SyzygyPath>``: validate every
    table, naming each before it is initialised, so a parent that sees this
    process die knows the last table attempted (``deepfin --selfcheck``)."""
    import sys

    path = normalise_path(argv[0]) if argv else ""
    try:
        _tb_ext.init(path)
    except (ValueError, RuntimeError) as exc:
        print(f"{VALIDATE_FAIL}{exc}", flush=True)
        return 1

    def progress(name: str) -> None:
        sys.stdout.write(f"{VALIDATE_START}{name}\n")
        sys.stdout.flush()

    problems = validate_loaded(path, progress)
    if problems:
        print(f"{VALIDATE_FAIL}{'; '.join(problems)}", flush=True)
        return 1
    n_wdl, n_dtz, _ = _tb_ext.table_counts()
    print(f"{VALIDATE_OK}{n_wdl} WDL + {n_dtz} DTZ tables initialised", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_validate_main(sys.argv[1:]))
