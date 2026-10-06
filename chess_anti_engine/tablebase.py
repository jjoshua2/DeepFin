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
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import chess
import chess.syzygy
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
import contextlib

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
# string share the opened Tablebase. We keep old path handles open instead of
# closing on swap: callers probe the returned handle after this function
# releases its lock, so closing the previous global handle can invalidate an
# in-flight probe from another thread.
_tablebases: dict[str, chess.syzygy.Tablebase] = {}
_tablebase_lock = threading.Lock()


class MatchTablebaseError(RuntimeError):
    """A required match tablebase component or eligible probe is unavailable."""


def _table_max_pieces(tables: Iterable[str]) -> int:
    return max((sum(ch.isupper() for ch in key) for key in tables), default=0)


def open_strict_match_tablebase(
    path: str, *, max_pieces: int = 6,
) -> chess.syzygy.Tablebase:
    """Open an owned match handle, refusing absent components and insufficient capacity.

    This checks *capacity*, not every material. A missing eligible material
    raises at its WDL/DTZ probe in :func:`rule50_match_status`. The caller must
    close the returned handle. Legacy cached/training openers are unchanged.
    """
    if type(max_pieces) is not int or not 3 <= max_pieces <= 7:
        raise ValueError("max_pieces must be an integer from 3 through 7")
    if not path or not path.strip():
        raise MatchTablebaseError("match Syzygy path is empty")
    parts = path.split(os.pathsep)
    if any(not part or part != part.strip() for part in parts):
        raise MatchTablebaseError("match Syzygy path has an empty or padded component")
    resolved: list[Path] = []
    for part in parts:
        directory = Path(part).resolve()
        if not directory.is_dir():
            raise MatchTablebaseError(f"match Syzygy directory is missing: {part}")
        if directory in resolved:
            raise MatchTablebaseError(f"duplicate match Syzygy directory: {part}")
        resolved.append(directory)
    tablebase = chess.syzygy.Tablebase()
    try:
        for directory in resolved:
            before = len(tablebase.wdl) + len(tablebase.dtz)
            try:
                tablebase.add_directory(str(directory))
            except (OSError, ValueError) as exc:
                raise MatchTablebaseError(
                    f"could not open match Syzygy component: {directory}"
                ) from exc
            if len(tablebase.wdl) + len(tablebase.dtz) == before:
                raise MatchTablebaseError(
                    f"match Syzygy component opened no new tables: {directory}"
                )
        wdl_max = _table_max_pieces(tablebase.wdl)
        dtz_max = _table_max_pieces(tablebase.dtz)
        if wdl_max < max_pieces or dtz_max < max_pieces:
            raise MatchTablebaseError(
                f"match Syzygy capacity below {max_pieces}-man: "
                f"WDL max={wdl_max}, DTZ max={dtz_max}"
            )
        return tablebase
    except BaseException:
        tablebase.close()
        raise


def rule50_match_status(
    board: chess.Board,
    tablebase: chess.syzygy.Tablebase,
    *,
    max_pieces: int = 6,
) -> int | None:
    """Return STM +1 win, -1 loss, 2 draw, or unresolved ``None``.

    Natural claimable outcomes take precedence. At a covered nonterminal root,
    ±1 WDL is a 50-move draw. ±2 WDL is certified decisive only at a reset
    clock. At a positive clock, even a short DTZ does not rule out a future
    threefold claim against earlier game history, so the result stays unknown.
    DTZ is still required and checked for coverage and WDL-class consistency.
    Missing eligible WDL or DTZ is an error, not neural fallback.
    """
    natural = board.outcome(claim_draw=True)
    if natural is not None:
        if natural.winner is None:
            return 2
        return 1 if natural.winner == board.turn else -1
    if chess.popcount(board.occupied) > max_pieces or board.castling_rights:
        return None
    try:
        wdl = int(tablebase.probe_wdl(board))
        dtz = int(tablebase.probe_dtz(board))
    except (KeyError, IndexError, chess.syzygy.MissingTableError) as exc:
        raise MatchTablebaseError(f"missing eligible match Syzygy probe: {board.fen()}") from exc
    if wdl not in (-2, -1, 0, 1, 2):
        raise MatchTablebaseError(f"invalid match Syzygy WDL {wdl}: {board.fen()}")
    if wdl == 0:
        if dtz != 0:
            raise MatchTablebaseError(f"WDL/DTZ mismatch: {board.fen()}")
        return 2
    if (wdl > 0) != (dtz > 0) or dtz == 0:
        raise MatchTablebaseError(f"WDL/DTZ sign mismatch: {board.fen()}")
    if (abs(wdl) == 2 and abs(dtz) > 100) or (abs(wdl) == 1 and abs(dtz) <= 100):
        raise MatchTablebaseError(f"WDL/DTZ magnitude mismatch: {board.fen()}")
    if abs(wdl) == 1:
        return 2
    if board.halfmove_clock == 0:
        return 1 if wdl > 0 else -1
    return None


def rule50_match_result(
    board: chess.Board,
    tablebase: chess.syzygy.Tablebase,
    *,
    max_pieces: int = 6,
) -> str | None:
    """White-relative game result from :func:`rule50_match_status`."""
    status = rule50_match_status(board, tablebase, max_pieces=max_pieces)
    if status is None:
        return None
    if status == 2:
        return "1/2-1/2"
    white_wins = (status == 1) == (board.turn == chess.WHITE)
    return "1-0" if white_wins else "0-1"


def get_tablebase(path: str) -> chess.syzygy.Tablebase | None:
    """Return an opened Tablebase for ``path``, caching across calls.

    ``path`` is an OS-separated list of directories (``:`` on POSIX, ``;``
    on Windows). Returns ``None`` on empty path or if none of the listed
    directories could be added.

    Passing the same path returns the cached handle without reopening.
    Different paths keep independent handles so an option change cannot close
    a tablebase while another thread is still probing it.
    """
    if not path:
        return None
    with _tablebase_lock:
        cached = _tablebases.get(path)
        if cached is not None:
            return cached
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
            with contextlib.suppress(OSError):
                tb.close()
            return None
        _tablebases[path] = tb
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


# DTZ of the move that just zeroed the counter, from the mover's point of
# view. Tables do not store a DTZ for the zeroing move itself; the next
# phase's DTZ is a different counter. Stockfish recovers it from the child
# WDL the same way (`dtz_before_zeroing`).
_DTZ_BEFORE_ZEROING = {2: 1, 1: 101, 0: 0, -1: -101, -2: -1}
_TB_PROBE_ERRORS = (KeyError, chess.syzygy.MissingTableError, IndexError)


def _dtz_of_played_move(child: chess.Board, tablebase: chess.syzygy.Tablebase) -> int:
    """DTZ of the move that produced ``child``, from the mover's perspective.

    ``child`` is the position after the move. A zeroing move (counter already
    0) is distance 1 or 101, not the length of the following phase.
    Non-zeroing moves negate the child's DTZ and add the ply just played.
    Checkmate outranks a 50-move claim on the same position; a claimable
    draw is distance 0 because the tables assume a fresh counter and do not
    know this game's history.
    """
    if child.is_checkmate():
        return 1
    # A claim the side to move could reach by playing on is not a draw yet:
    # can_claim_draw() is true at halfmove 99 whenever a quiet move exists.
    # Stockfish zeros DTZ only for a draw that has already happened
    # (is_draw / is_repetition at one ply).
    if (
        child.is_stalemate()
        or child.is_insufficient_material()
        or child.is_fifty_moves()
        or child.is_repetition(3)
    ):
        return 0
    if child.halfmove_clock == 0:
        our_wdl = -int(tablebase.probe_wdl(child))
        try:
            return _DTZ_BEFORE_ZEROING[our_wdl]
        except KeyError as exc:
            raise KeyError(f"invalid syzygy wdl {our_wdl}") from exc
    child_dtz = int(tablebase.probe_dtz(child))
    if child_dtz > 0:
        return -child_dtz - 1
    if child_dtz < 0:
        return -child_dtz + 1
    return 0


def _rule50_dtz_key(dtz: int, halfmove_clock: int) -> tuple[int, int]:
    """Higher is better. ``halfmove_clock`` is the counter before the move.

    These classes are Stockfish ``root_probe``'s rank buckets, not the
    score. A win with ``dtz + counter <= 99`` outranks every later win,
    including one that still wins at ``== 100``. A loss in the
    ``-dtz * 2 + counter < 100`` bucket outranks no softer loss; inside a
    losing bucket the longer distance is better. The score of the chosen
    move is :func:`_wdl_adjusted_for_rule50`.
    """
    if dtz > 0:
        strict = dtz + halfmove_clock <= 99
        return (2 if strict else 1, -dtz)
    if dtz < 0:
        hard_loss = (-dtz * 2 + halfmove_clock) < 100
        return (-2 if hard_loss else -1, -dtz)
    return (0, 0)


def _wdl_adjusted_for_rule50(dtz: int, halfmove_clock: int) -> int:
    """±2 when that result still holds at this counter.

    A win survives through ``dtz + counter == 100`` (the move that lands
    on the boundary still zeroed in time, or the opponent is not yet able
    to claim). A loss is certain through ``|dtz| + counter == 100``. The
    stricter ``<= 99`` and ``* 2`` tests are rank buckets only; using them
    here reports a real win or loss as a draw.
    """
    if dtz > 0:
        return 2 if dtz + halfmove_clock <= 100 else 1
    if dtz < 0:
        return -2 if -dtz + halfmove_clock <= 100 else -1
    return 0


def _select_dtz_move(
    board: chess.Board, syzygy_path: str,
) -> tuple[chess.Move, int] | None:
    """Best legal move and its DTZ from the mover's perspective."""
    if not is_tb_eligible(board):
        return None
    tablebase = get_tablebase(syzygy_path)
    if tablebase is None:
        return None
    halfmove = board.halfmove_clock
    best_move: chess.Move | None = None
    best_key: tuple[int, int] | None = None
    best_dtz = 0
    for move in board.legal_moves:
        board.push(move)
        dtz: int | None = None
        try:
            try:
                dtz = _dtz_of_played_move(board, tablebase)
            except _TB_PROBE_ERRORS:
                dtz = None
            except Exception as exc:
                _log.debug(
                    "syzygy move probe failed %s %s: %r", board.fen(), move, exc,
                )
                dtz = None
        finally:
            board.pop()
        if dtz is None:
            continue
        key = _rule50_dtz_key(dtz, halfmove)
        if best_key is None or key > best_key:
            best_key = key
            best_move = move
            best_dtz = dtz
    if best_move is None:
        return None
    return best_move, best_dtz


def probe_best_move(board: chess.Board, syzygy_path: str) -> chess.Move | None:
    """DTZ-optimal move that preserves a 50-move result when one is available.

    A capture or pawn move is ranked by the fact that it zeroed the counter,
    not by the next phase's DTZ. Non-zeroing moves add the root halfmove
    clock, so a short table distance that crosses the 50-move boundary loses
    to a zeroing move that keeps the win. Returns ``None`` if the position
    is not eligible or every legal probe fails.
    """
    picked = _select_dtz_move(board, syzygy_path)
    if picked is None:
        return None
    return picked[0]


def rescore_game_samples(
    boards_history: list[chess.Board],
    syzygy_path: str,
) -> str | None:
    """Walk ``boards_history`` and relabel the whole game to the LAST
    TB-proven result (``"1-0"`` / ``"0-1"`` / ``"1/2-1/2"``).

    Using the last result (not the first) correctly handles games that
    transition through a drawn endgame into a winning one — e.g., KRvKR
    (drawn) → rook captured → KRvK (won). The first-eligible result would
    be the drawn KRvKR position; the last is the won KRvK, which is correct.

    Returns ``None`` if no board in the history was TB-eligible.
    """
    last: str | None = None
    for board in boards_history:
        result = tb_adjudicate_result(board, syzygy_path)
        if result is not None:
            last = result
    return last


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
    """Return the DTZ-optimal move and the rule50-adjusted WDL of that move.

    The WDL is from the side to move. +2 means the chosen line still wins
    with this counter (``dtz + halfmove_clock <= 100``); -2 means it still
    loses (``|dtz| + halfmove_clock <= 100``). +1 and -1 are results the
    50-move rule draws, which UCI's root shortcut already maps to score 0.
    Raw ``probe_wdl`` assumes the counter is zero and is not what this returns.

    Used by UCI's root shortcut to bypass MCTS when the root is in TB range.
    """
    picked = _select_dtz_move(board, syzygy_path)
    if picked is None:
        return None
    move, dtz = picked
    return move, _wdl_adjusted_for_rule50(dtz, board.halfmove_clock)


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

    Opt-in ``rule50_aware=True`` instead uses a caller-owned strict handle and
    :func:`rule50_match_status`. CBoard FEN loses repetition history, so this
    mode abstains from decisive solved status at any positive halfmove clock;
    an immediate zeroing capture/pawn move still permits ±2 guidance. Even a
    full-board present claim check plus short DTZ cannot prove a future line
    will avoid repeating positions from earlier game history.
    """

    def __init__(
        self,
        syzygy_path: str,
        max_pieces: int | None = None,
        *,
        cursed_as_draw: bool = True,
        rule50_aware: bool = False,
        tablebase: chess.syzygy.Tablebase | None = None,
    ) -> None:
        """``max_pieces`` caps the filter piece-count; if None (default) we
        detect it from the opened tables' material keys so a user with only
        3–5-man files gets 5 and we don't waste probes on 6–7-piece leaves
        that would always miss. Explicit values clamp lower.

        ``cursed_as_draw``: when True (default), cursed-win (+1) /
        blessed-loss (-1) results are treated as draws — correct for
        50-move-rule play. When False, they count as decisive — useful
        for analysis / correspondence positions without the 50-move rule.

        ``rule50_aware`` requires ``cursed_as_draw=True`` and an explicitly
        opened strict ``tablebase``. Legacy construction remains unchanged.
        """
        if type(cast(object, rule50_aware)) is not bool:
            raise TypeError("rule50_aware must be a bool")
        if rule50_aware and not cursed_as_draw:
            raise ValueError("rule50-aware search requires cursed_as_draw=True")
        self._path = syzygy_path
        self._cursed_as_draw = bool(cursed_as_draw)
        self._rule50_aware = bool(rule50_aware)
        if self._rule50_aware and tablebase is None:
            raise MatchTablebaseError(
                "rule50-aware search requires an explicitly opened strict match tablebase"
            )
        self._tablebase = tablebase
        tb = tablebase if tablebase is not None else get_tablebase(syzygy_path)
        if tb is not None:
            self.n_wdl = len(tb.wdl)
            self.n_dtz = len(tb.dtz)
  # Material keys like "KQvK" / "KRPvKP" — piece letters are
  # uppercase, 'v' is lowercase; count uppercase only.
            available = max(
                (sum(c.isupper() for c in k) for k in tb.wdl),
                default=0,
            )
        else:
            self.n_wdl = 0
            self.n_dtz = 0
            available = 0
        self.max_pieces = (
            min(int(max_pieces), available) if max_pieces is not None else available
        )
        if self._rule50_aware:
            assert tb is not None
            if (
                max_pieces is None or self.max_pieces != int(max_pieces)
                or _table_max_pieces(tb.dtz) < int(max_pieces)
            ):
                raise MatchTablebaseError("rule50-aware search lacks requested WDL/DTZ capacity")
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
        solved_out: np.ndarray | None = None,
    ) -> int:
        """Probe each leaf and overwrite its wdl row on a TB hit.

        When ``indices`` is provided, ``leaf_cboards`` are assumed to already
        satisfy :func:`is_tb_eligible` (the C-side ``get_pending_tb_leaves``
        path) and ``indices[j]`` is the row of ``wdl`` to write. When
        ``indices`` is None, each leaf is checked here — used at the root
        where leaves are a short explicit list, not pre-filtered.

        ``solved_out``: optional int8 array, same length as ``leaf_cboards``,
        receives the solved status for each hit (1=WIN, -1=LOSS, 2=DRAW from
        STM perspective; 0=no hit / skipped). Caller passes it to
        ``MCTSTree.mark_tb_solved`` to propagate proven values up the tree.
        """
        if not leaf_cboards:
            return 0
        tb = self._tablebase if self._tablebase is not None else get_tablebase(self._path)
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
            if self._rule50_aware:
                board = chess.Board(cb.fen())
                status = rule50_match_status(board, tb, max_pieces=max_pieces)
                if status is None:
                    continue
                if status == 1:
                    wdl[i] = _WIN_LOGITS
                elif status == -1:
                    wdl[i] = _LOSS_LOGITS
                else:
                    wdl[i] = _DRAW_LOGITS
                if solved_out is not None:
                    solved_out[j] = status
                hits += 1
                continue
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
                if solved_out is not None:
                    solved_out[j] = 1  # SOLVED_WIN
            elif wdl_val <= loss_thresh:
                wdl[i] = _LOSS_LOGITS
                if solved_out is not None:
                    solved_out[j] = -1  # SOLVED_LOSS
            else:
                wdl[i] = _DRAW_LOGITS
                if solved_out is not None:
                    solved_out[j] = 2  # SOLVED_DRAW
            hits += 1
        self.hits += hits
        return hits
