"""Shared Syzygy tablebase primitives.

Two consumers:
  - training (``selfplay/manager.py``): post-hoc game rescoring and
    DTZ-optimal policy-target rewriting.
  - UCI search (``uci/tablebase.py``): in-tree WDL override on MCTS leaves.

Both go through :func:`get_tablebase` so they share a single opened handle
per path. Eligibility, WDL semantics (for *labels*), and the DTZ move picker
live here too — anything that's tablebase-shaped rather than training- or
search-shaped belongs in this module.

Cursed-win / blessed-loss (±1) are treated as WIN/LOSS here to match the
game-outcome label used for value-head training. UCI search intentionally
diverges and treats them as draws (50-move rule in actual play); see
``SyzygyProbe`` in ``uci/tablebase.py``.
"""
from __future__ import annotations

import os

import chess
import chess.syzygy


def is_tb_eligible(board: chess.Board) -> bool:
    """True iff ``board`` is probable with Syzygy.

    Requires ≤7 pieces total AND no castling rights — Syzygy's index doesn't
    encode castling, so positions with it must be excluded.
    """
    return (
        chess.popcount(board.occupied) <= 7
        and not board.has_castling_rights(chess.WHITE)
        and not board.has_castling_rights(chess.BLACK)
    )


# Lazy, path-keyed cache. ``path`` is the raw string the caller passed in
# (OS-separated multi-dir supported), so two callers with the same path
# string share the opened Tablebase.
_tablebase: chess.syzygy.Tablebase | None = None
_tablebase_path: str | None = None


def get_tablebase(path: str) -> chess.syzygy.Tablebase | None:
    """Return an opened Tablebase for ``path``, caching across calls.

    ``path`` is an OS-separated list of directories (``:`` on POSIX, ``;``
    on Windows). Returns ``None`` on empty path or if none of the listed
    directories could be added.

    Swapping to a new path closes the previous instance; passing the same
    path returns the cached handle without reopening.
    """
    global _tablebase, _tablebase_path
    if _tablebase is not None and _tablebase_path == path:
        return _tablebase
    if _tablebase is not None:
        try:
            _tablebase.close()
        except Exception:
            pass
        _tablebase = None
        _tablebase_path = None
    if not path:
        return None
    dirs = [p.strip() for p in path.split(os.pathsep) if p.strip()]
    if not dirs:
        return None
    tb = chess.syzygy.Tablebase()
    added = 0
    for d in dirs:
        try:
            tb.add_directory(d)
            added += 1
        except (OSError, FileNotFoundError):
            continue
    if added == 0:
        try:
            tb.close()
        except Exception:
            pass
        return None
    _tablebase = tb
    _tablebase_path = path
    return tb


def probe_wdl(board: chess.Board, syzygy_path: str) -> int | None:
    """Training-label flavor: returns 0=loss, 1=draw, 2=win from STM's
    perspective, or None if not eligible / not found.

    Cursed-win (+1) → 2 and blessed-loss (-1) → 0, so the returned label
    matches the theoretical game outcome even under the 50-move rule.
    UCI search calls a different wrapper (``SyzygyProbe``) that treats
    them as draws.
    """
    if not is_tb_eligible(board):
        return None
    tb = get_tablebase(syzygy_path)
    if tb is None:
        return None
    try:
        wdl = tb.probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError):
        return None
    if wdl > 0:
        return 2
    if wdl < 0:
        return 0
    return 1


def probe_best_move(board: chess.Board, syzygy_path: str) -> chess.Move | None:
    """DTZ-optimal move that preserves the current WDL category.

    Returns ``None`` if the position isn't eligible, the probe fails, or
    no move preserves the WDL class (which shouldn't happen for a solved
    position but we defend against anyway).
    """
    if not is_tb_eligible(board):
        return None
    tb = get_tablebase(syzygy_path)
    if tb is None:
        return None
    try:
        root_wdl = tb.probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError):
        return None

    best_move: chess.Move | None = None
    best_score: int | None = None

    for move in board.legal_moves:
        board.push(move)
        try:
            child_dtz = tb.probe_dtz(board)
        except (KeyError, chess.syzygy.MissingTableError):
            board.pop()
            continue
        board.pop()

        # After our move the signs flip: child_dtz is from the opponent's
        # POV. We want the highest "score for us", which is -child_dtz.
        # Filter by WDL class so a winning move must still leave opp losing,
        # etc.
        if root_wdl > 0 and child_dtz >= 0:
            continue  # winning move must leave opponent at dtz<0
        if root_wdl == 0 and child_dtz != 0:
            continue  # drawing move must leave opponent at dtz=0
        if root_wdl < 0 and child_dtz <= 0:
            continue  # from a loss, only care about largest positive (longest survival)

        score = -child_dtz
        if best_score is None or score > best_score:
            best_score = score
            best_move = move

    return best_move


def rescore_game_samples(
    boards_history: list[chess.Board],
    syzygy_path: str,
) -> str | None:
    """Walk ``boards_history`` forward and relabel the whole game to the
    first TB-proven result we encounter (``"1-0"`` / ``"0-1"`` / ``"1/2-1/2"``).

    Returns ``None`` if no board in the history was TB-eligible.
    """
    for board in boards_history:
        wdl = probe_wdl(board, syzygy_path)
        if wdl is None:
            continue
        if wdl == 1:
            return "1/2-1/2"
        if wdl == 2:
            return "1-0" if board.turn == chess.WHITE else "0-1"
        return "0-1" if board.turn == chess.WHITE else "1-0"
    return None
