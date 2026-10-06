"""Rule-50 ranking for the DTZ root move, with a generated fake tablebase.

The native extension is not required: these tests call the Python selector
UCI's root shortcut and selfplay's TB policy target both use. When the
extension is absent, a stand-in keeps ``tablebase.py`` importable. That
stand-in is never evaluated.
"""
from __future__ import annotations

import importlib
import sys
import types

import chess
import pytest


def _allow_import_without_native_extension() -> None:
    """``tablebase.py`` imports the native board type at module scope.

    This worktree has no built extension, and these tests never construct a
    CBoard. When the extension imports, CI uses that module unchanged.
    """
    name = "chess_anti_engine.encoding._lc0_ext"
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        stand_in = types.ModuleType(name)
        setattr(stand_in, "CBoard", type("CBoard", (), {}))
        sys.modules[name] = stand_in


_allow_import_without_native_extension()

from chess_anti_engine import tablebase as tbmod  # noqa: E402


class ScriptedTable:
    """Clock-0 WDL/DTZ answers. Zeroing moves are detected by the selector,
    so their next-phase DTZ is deliberately a worse number than the quiet
    line: a picker that ranks raw child DTZ chooses the quiet move."""

    def __init__(
        self,
        *,
        root_turn: chess.Color,
        pawn_square: chess.Square,
        quiet_child_dtz: int,
        zero_child_dtz: int,
        root_wdl: int = 2,
        child_wdl: int = -2,
    ) -> None:
        self.root_turn = root_turn
        self.pawn_square = pawn_square
        self.quiet_child_dtz = quiet_child_dtz
        self.zero_child_dtz = zero_child_dtz
        self.root_wdl = root_wdl
        self.child_wdl = child_wdl

    def probe_wdl(self, board: chess.Board) -> int:
        if board.turn == self.root_turn:
            return self.root_wdl
        return self.child_wdl

    def probe_dtz(self, board: chess.Board) -> int:
        if board.turn == self.root_turn:
            return 1
        if board.piece_at(self.pawn_square) is None:
            return self.zero_child_dtz
        return self.quiet_child_dtz


class MoveTable:
    """Non-zeroing DTZ keyed by the move just pushed. Unlisted moves use
    ``default``."""

    def __init__(
        self,
        *,
        root_turn: chess.Color,
        by_uci: dict[str, int],
        default: int,
        root_wdl: int = 2,
        child_wdl: int = -2,
    ) -> None:
        self.root_turn = root_turn
        self.by_uci = by_uci
        self.default = default
        self.root_wdl = root_wdl
        self.child_wdl = child_wdl

    def probe_wdl(self, board: chess.Board) -> int:
        if board.turn == self.root_turn:
            return self.root_wdl
        return self.child_wdl

    def probe_dtz(self, board: chess.Board) -> int:
        if board.turn == self.root_turn:
            return 1
        return self.by_uci.get(board.peek().uci(), self.default)


def _install(monkeypatch: pytest.MonkeyPatch, table: object) -> None:
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: table)


def test_high_clock_prefers_zeroing_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    board = chess.Board("4k3/8/8/8/8/8/p7/R3K3 w - - 96 40")
    capture = chess.Move.from_uci("a1a2")
    assert capture in board.legal_moves
    _install(monkeypatch, ScriptedTable(
        root_turn=chess.WHITE, pawn_square=chess.A2,
        quiet_child_dtz=-20, zero_child_dtz=-80,
    ))
    assert tbmod.probe_best_move(board, "generated") == capture
    picked = tbmod.try_tb_root_move(board, "generated")
    assert picked is not None
    assert picked[0] == capture
    assert picked[1] == 2


def test_black_high_clock_prefers_zeroing_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    board = chess.Board("r3k3/P7/8/8/8/8/8/4K3 b - - 96 40")
    capture = chess.Move.from_uci("a8a7")
    assert capture in board.legal_moves
    _install(monkeypatch, ScriptedTable(
        root_turn=chess.BLACK, pawn_square=chess.A7,
        quiet_child_dtz=-20, zero_child_dtz=-80,
    ))
    assert tbmod.probe_best_move(board, "generated") == capture


def test_unpreserved_table_win_is_not_reported_as_certain(monkeypatch: pytest.MonkeyPatch) -> None:
    board = chess.Board("7k/8/8/8/8/8/8/KQ6 w - - 96 40")
    assert not any(board.is_capture(move) for move in board.legal_moves)
    _install(monkeypatch, MoveTable(
        root_turn=chess.WHITE, by_uci={}, default=-30,
    ))
    picked = tbmod.try_tb_root_move(board, "generated")
    assert picked is not None
    assert picked[1] == 1


def test_shorter_non_zeroing_win_still_beats_a_longer_one(monkeypatch: pytest.MonkeyPatch) -> None:
    board = chess.Board("8/8/k7/8/8/8/8/4K2R w - - 0 1")
    quiet = []
    for move in board.legal_moves:
        board.push(move)
        terminal = board.is_checkmate() or board.halfmove_clock == 0
        board.pop()
        if not terminal:
            quiet.append(move)
    assert len(quiet) >= 2
    fast, slow = quiet[0], quiet[1]
    _install(monkeypatch, MoveTable(
        root_turn=chess.WHITE,
        by_uci={fast.uci(): -4, slow.uci(): -30},
        default=-30,
    ))
    assert tbmod.probe_best_move(board, "generated") == fast


def test_longer_loss_still_beats_a_faster_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    # No capture and no mate, so every move stays a scripted loss.
    board = chess.Board("8/8/k7/8/8/8/8/4K2R w - - 0 1")
    quiet = []
    for move in board.legal_moves:
        board.push(move)
        terminal = board.is_checkmate() or board.halfmove_clock == 0
        board.pop()
        if not terminal:
            quiet.append(move)
    assert len(quiet) >= 2
    fast_loss, slow_loss = quiet[0], quiet[1]
    _install(monkeypatch, MoveTable(
        root_turn=chess.WHITE,
        by_uci={fast_loss.uci(): 4, slow_loss.uci(): 40},
        default=4,
        root_wdl=-2,
        child_wdl=2,
    ))
    assert tbmod.probe_best_move(board, "generated") == slow_loss


def test_missing_table_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    board = chess.Board("4k3/8/8/8/8/8/p7/R3K3 w - - 0 1")
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: None)
    assert tbmod.probe_best_move(board, "generated") is None
    assert tbmod.try_tb_root_move(board, "generated") is None


def test_checkmate_at_high_clock_stays_a_certain_win(monkeypatch: pytest.MonkeyPatch) -> None:
    board = chess.Board("k7/8/1K6/8/8/8/8/6Q1 w - - 96 40")
    mate = chess.Move.from_uci("g1g8")
    board.push(mate)
    assert board.is_checkmate()
    board.pop()
    _install(monkeypatch, MoveTable(
        root_turn=chess.WHITE, by_uci={}, default=-4,
    ))
    picked = tbmod.try_tb_root_move(board, "generated")
    assert picked is not None
    assert picked[0] == mate
    assert picked[1] == 2


def test_probe_error_skips_that_move(monkeypatch: pytest.MonkeyPatch) -> None:
    board = chess.Board("8/8/k7/8/8/8/8/4K2R w - - 0 1")
    quiet = []
    for move in board.legal_moves:
        board.push(move)
        terminal = board.is_checkmate() or board.halfmove_clock == 0
        board.pop()
        if not terminal:
            quiet.append(move)
    assert len(quiet) >= 2
    broken, kept = quiet[0], quiet[1]

    class _Flaky(MoveTable):
        def probe_dtz(self, board: chess.Board) -> int:
            if board.peek().uci() == broken.uci():
                raise KeyError("generated missing")
            return super().probe_dtz(board)

    _install(monkeypatch, _Flaky(
        root_turn=chess.WHITE,
        by_uci={kept.uci(): -4},
        default=-30,
    ))
    assert tbmod.probe_best_move(board, "generated") == kept


def test_ineligible_castling_position_is_not_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def _boom(_path: str) -> None:
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(tbmod, "get_tablebase", _boom)
    board = chess.Board()
    assert tbmod.probe_best_move(board, "generated") is None
    assert called is False
