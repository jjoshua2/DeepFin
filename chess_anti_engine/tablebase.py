"""Shared Syzygy tablebase primitives.

Callers:
  - training / selfplay: game rescoring, DTZ-optimal policy targets,
    in-play TB adjudication (end the game at the first TB-eligible
    position), and optional in-search WDL overrides on MCTS leaves.
  - UCI search: in-search WDL overrides, DTZ-optimal root-move shortcut.

Both go through :func:`get_tablebase` so one opened handle is shared per
path. Eligibility, WDL label mapping, the DTZ move picker, the in-search
``SyzygyProbe`` class, and the small glue helpers used by UCI's root
shortcut and selfplay's adjudication path all live here — anything
tablebase-shaped rather than training- or search-shaped.

Cursed-win / blessed-loss (±1) handling differs by purpose:
  * training-label functions (:func:`probe_wdl`, :func:`rescore_game_samples`,
    :func:`tb_adjudicate_result`) treat cursed as WIN to match the
    theoretical game outcome the value head is supposed to predict.
  * in-search functions (:class:`SyzygyProbe`) treat cursed as DRAW to
    match actual play under the 50-move rule, where a cursed win really
    does draw.
"""
from __future__ import annotations

import logging
import os

import chess
import chess.syzygy
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard

_log = logging.getLogger(__name__)


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
        except OSError:
            pass  # syzygy already closed / file handle gone
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
    best_dtz: int | None = None

    for move in board.legal_moves:
        board.push(move)
        try:
            child_dtz = tb.probe_dtz(board)
        except (KeyError, chess.syzygy.MissingTableError):
            board.pop()
            continue
        board.pop()

  # After our move, it's opponent's turn. ``child_dtz`` is DTZ from
  # their perspective: negative if they're losing (we're winning),
  # positive if they're winning (we're losing), 0 on draw.
  #
  # We always MAXIMIZE child_dtz, regardless of WDL class:
  #   * winning: child_dtz ∈ (-∞, 0). Largest = least negative =
  #     smallest |dtz| = opponent converts fastest = we win fastest.
  #   * drawing: child_dtz == 0 (filter), trivially max.
  #   * losing: child_dtz ∈ (0, +∞). Largest = most plies before
  #     opponent can zero = we survive longest.
        if root_wdl > 0 and child_dtz >= 0:
            continue  # winning move must leave opponent in a lost position
        if root_wdl == 0 and child_dtz != 0:
            continue  # drawing move must leave opponent at dtz=0
        if root_wdl < 0 and child_dtz <= 0:
            continue  # from a loss, only care about longer-survival options

        if best_dtz is None or child_dtz > best_dtz:
            best_dtz = child_dtz
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
        result = tb_adjudicate_result(board, syzygy_path)
        if result is not None:
            return result
    return None


def tb_adjudicate_result(
    board: chess.Board, syzygy_path: str,
) -> str | None:
    """If ``board`` is TB-eligible and in a probable material, return the
    game result string ("1-0" / "0-1" / "1/2-1/2") from the TB's view.
    Cursed/blessed treated as decisive (training-label convention).

    Used by selfplay in-play adjudication (end the game here, save the
    remaining compute) and by :func:`rescore_game_samples` (relabel an
    already-finished game). Returns ``None`` if not TB-eligible or the
    probe fails.
    """
    wdl = probe_wdl(board, syzygy_path)
    if wdl is None:
        return None
    if wdl == 1:
        return "1/2-1/2"
    if wdl == 2:
        return "1-0" if board.turn == chess.WHITE else "0-1"
    return "0-1" if board.turn == chess.WHITE else "1-0"


def try_tb_root_move(
    board: chess.Board, syzygy_path: str,
) -> tuple[chess.Move, int] | None:
    """Return the DTZ-optimal move and raw ``probe_wdl`` value (-2..2 from
    STM's perspective) for ``board``, or None if the root isn't TB-eligible
    or the probe fails.

    Used by UCI's root shortcut to bypass MCTS entirely when the root is
    in TB range, and by selfplay to play TB-optimal moves in endgame
    sequences. Caller decides how to interpret the WDL value (UCI uses
    saturated cp, selfplay uses the training-label mapping).
    """
    if not is_tb_eligible(board):
        return None
    best = probe_best_move(board, syzygy_path)
    if best is None:
        return None
    tb = get_tablebase(syzygy_path)
    if tb is None:
        return None
    try:
        wdl_val = tb.probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError, IndexError):
        return None
    except Exception as exc:
        _log.debug("syzygy probe_wdl failed at root %s: %r", board.fen(), exc)
        return None
    return best, int(wdl_val)


# ---- In-search WDL overrides ------------------------------------------------
#
# Large-magnitude logits so softmax collapses to one-hot. 12.0 against zeros
# gives >0.999999 on the target bin — well inside what MCTS needs to pin Q.
_WIN_LOGITS = np.array([12.0, -12.0, -12.0], dtype=np.float32)
_LOSS_LOGITS = np.array([-12.0, -12.0, 12.0], dtype=np.float32)
_DRAW_LOGITS = np.array([-12.0, 12.0, -12.0], dtype=np.float32)


class SyzygyProbe:
    """Override NN wdl logits with TB truth for TB-eligible MCTS leaves.

    Consumes leaves as CBoard objects, writes directly into the wdl ndarray
    the evaluator returned. See ``gumbel_c.py:_tb_override`` for the hook
    points; both UCI and selfplay share this class.

    Cursed-win / blessed-loss treated as DRAWS here — matches 50-move-rule
    play. Training *labels* diverge and treat cursed as decisive; use
    :func:`tb_adjudicate_result` or :func:`probe_wdl` for that.

    Holds only the path string. The actual :class:`chess.syzygy.Tablebase`
    handle is fetched (and cached) via :func:`get_tablebase`, so multiple
    probes against the same path share one open instance.
    """

    def __init__(
        self,
        syzygy_path: str,
        max_pieces: int | None = None,
        *,
        cursed_as_draw: bool = True,
    ) -> None:
        """``max_pieces`` caps the filter piece-count; if None (default) we
        detect it from the opened tables' material keys so a user with only
        3–5-man files gets 5 and we don't waste probes on 6–7-piece leaves
        that would always miss. Explicit values clamp lower.

        ``cursed_as_draw``: when True (default), cursed-win (+1) /
        blessed-loss (-1) results are treated as draws — correct for
        50-move-rule play. When False, they count as decisive — useful
        for analysis / correspondence positions without the 50-move rule.
        """
        self._path = syzygy_path
        self._cursed_as_draw = bool(cursed_as_draw)
        tb = get_tablebase(syzygy_path)
        if tb is not None:
            self.n_wdl = len(tb.wdl)
            self.n_dtz = len(tb.dtz)
  # Material keys like "KQvK" / "KRPvKP" — piece letters are
  # uppercase, 'v' is lowercase; count uppercase only.
            available = max(
                (sum(c.isupper() for c in k) for k in tb.wdl.keys()),
                default=0,
            )
        else:
            self.n_wdl = 0
            self.n_dtz = 0
            available = 0
        self.max_pieces = (
            min(int(max_pieces), available) if max_pieces is not None else available
        )
        self.hits = 0
        self.probes = 0

    def reset_counts(self) -> None:
        self.hits = 0
        self.probes = 0

    def apply(
        self,
        leaf_cboards: list[CBoard],
        wdl: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> int:
        """Probe each leaf and overwrite its wdl row on a TB hit.

        When ``indices`` is provided, ``leaf_cboards`` are assumed to already
        satisfy :func:`is_tb_eligible` (the C-side ``get_pending_tb_leaves``
        path) and ``indices[j]`` is the row of ``wdl`` to write. When
        ``indices`` is None, each leaf is checked here — used at the root
        where leaves are a short explicit list, not pre-filtered.
        """
        if not leaf_cboards:
            return 0
        tb = get_tablebase(self._path)
        if tb is None:
            return 0
        hits = 0
        max_pieces = self.max_pieces
        for j, cb in enumerate(leaf_cboards):
            if indices is None:
                occ = int(cb.occ_white) | int(cb.occ_black)
                if occ.bit_count() > max_pieces or int(cb.castling) != 0:
                    continue
                i = j
            else:
                i = int(indices[j])
            self.probes += 1
            try:
                board = chess.Board(cb.fen())
                wdl_val = tb.probe_wdl(board)
            except (KeyError, chess.syzygy.MissingTableError, IndexError):
                continue
            except Exception as exc:
                _log.debug("syzygy probe failed on %s: %r", cb.fen(), exc)
                continue
  # Decisive threshold depends on 50-move-rule semantics:
  # cursed_as_draw=True (default) => only ±2 counts as decisive;
  # cursed_as_draw=False          => ±1 counts too.
            win_thresh = 2 if self._cursed_as_draw else 1
            loss_thresh = -win_thresh
            if wdl_val >= win_thresh:
                wdl[i] = _WIN_LOGITS
            elif wdl_val <= loss_thresh:
                wdl[i] = _LOSS_LOGITS
            else:
                wdl[i] = _DRAW_LOGITS
            hits += 1
        self.hits += hits
        return hits
