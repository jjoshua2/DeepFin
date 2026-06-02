from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts import root_tactics
from chess_anti_engine.moves import move_to_index

try:
    from chess_anti_engine.encoding._lc0_ext import CBoard
except ImportError:  # pragma: no cover - extension absent
    CBoard = None


def _require_cboard():
    assert CBoard is not None
    return CBoard


def test_root_tactics_import_does_not_require_cboard_encode() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCBoardEncode(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "chess_anti_engine.encoding.cboard_encode":
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockCBoardEncode())
        import chess_anti_engine.mcts.root_tactics as root_tactics
        assert root_tactics.immediate_mate_move is not None
        """
    )
    env = os.environ.copy()
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = (
        str(repo)
        if not env.get("PYTHONPATH")
        else f"{repo}{os.pathsep}{env['PYTHONPATH']}"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_immediate_terminal_draw_indices_include_fifty_move_claim() -> None:
    board = chess.Board("8/8/8/4k3/8/8/3R4/4K3 w - - 99 1")

    draws = root_tactics.immediate_terminal_draw_indices(board)

    expected = {int(move_to_index(move, board)) for move in board.legal_moves}
    assert draws == expected


def test_immediate_terminal_draw_indices_include_threefold_claim_after_move() -> None:
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        board.push(chess.Move.from_uci(move))
    repeated = chess.Move.from_uci("f6g8")
    after = board.copy(stack=True)
    after.push(repeated)
    assert after.is_game_over(claim_draw=True)

    draws = root_tactics.immediate_terminal_draw_indices(board)

    assert draws == {int(move_to_index(repeated, board))}


def test_immediate_terminal_draw_indices_include_threefold_claim_after_reply() -> None:
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]:
        board.push(chess.Move.from_uci(move))
    draw_move = chess.Move.from_uci("f3g1")
    after = board.copy(stack=True)
    after.push(draw_move)
    assert after.is_game_over(claim_draw=True)

    draws = root_tactics.immediate_terminal_draw_indices(board)

    assert draws == {int(move_to_index(draw_move, board))}


@pytest.mark.skipif(CBoard is None, reason="CBoard extension not available")
def test_immediate_terminal_cboard_policy_or_draws_include_fifty_move_claim() -> None:
    board = chess.Board("8/8/8/4k3/8/8/3R4/4K3 w - - 99 1")
    cboard_cls = _require_cboard()
    cb = cboard_cls.from_board(board)
    legal = cb.legal_move_indices()

    mate, draws = root_tactics.immediate_terminal_cboard_policy_or_draws(cb, legal)

    assert mate is None
    assert draws == set(np.asarray(legal, dtype=np.int32).astype(int).tolist())


@pytest.mark.skipif(CBoard is None, reason="CBoard extension not available")
def test_immediate_terminal_cboard_policy_or_draws_include_fifty_move_reply_claim() -> None:
    board = chess.Board("8/8/8/4k3/8/8/3R4/4K3 w - - 98 1")
    cboard_cls = _require_cboard()
    cb = cboard_cls.from_board(board)
    legal = cb.legal_move_indices()

    mate, draws = root_tactics.immediate_terminal_cboard_policy_or_draws(cb, legal)

    assert mate is None
    assert draws == set(np.asarray(legal, dtype=np.int32).astype(int).tolist())


@pytest.mark.skipif(CBoard is None, reason="CBoard extension not available")
def test_immediate_terminal_cboard_policy_or_draws_include_threefold_after_move() -> None:
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        board.push(chess.Move.from_uci(move))
    repeated = chess.Move.from_uci("f6g8")
    repeated_idx = int(move_to_index(repeated, board))
    cboard_cls = _require_cboard()
    cb = cboard_cls.from_board(board)

    mate, draws = root_tactics.immediate_terminal_cboard_policy_or_draws(
        cb, cb.legal_move_indices(),
    )

    assert mate is None
    assert draws == {repeated_idx}


@pytest.mark.skipif(CBoard is None, reason="CBoard extension not available")
def test_immediate_terminal_cboard_policy_or_draws_include_threefold_after_reply() -> None:
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]:
        board.push(chess.Move.from_uci(move))
    draw_move = chess.Move.from_uci("f3g1")
    draw_idx = int(move_to_index(draw_move, board))
    cboard_cls = _require_cboard()
    cb = cboard_cls.from_board(board)

    mate, draws = root_tactics.immediate_terminal_cboard_policy_or_draws(
        cb, cb.legal_move_indices(),
    )

    assert mate is None
    assert draws == {draw_idx}


def _py_root_terminals(board: chess.Board) -> tuple[set[int], set[int]]:
    """python-chess ground truth: ({mate actions}, {draw actions}) over legal root moves."""
    mates: set[int] = set()
    draws: set[int] = set()
    for move in board.legal_moves:
        idx = int(move_to_index(move, board))
        child = board.copy(stack=True)
        child.push(move)
        if child.is_checkmate():
            mates.add(idx)
        elif child.is_game_over(claim_draw=True):
            draws.add(idx)
    return mates, draws


# Each case must NOT depend on repetition history (covered separately above) so a
# plain FEN round-trips through CBoard.from_board. Covers every 1-move terminal the
# selfplay scan must never miss: mate (win), stalemate, fifty-move, insufficient material.
_PARITY_FENS = [
    pytest.param("5k2/1R6/P4BB1/P7/2P5/8/3K3P/8 w - - 5 70", id="mate-in-1"),
    pytest.param("8/8/8/8/8/8/5k2/5K1R w - - 5 1", id="stalemate-move-no-mate"),
    pytest.param("8/8/8/4k3/8/8/3R4/4K3 w - - 98 1", id="fifty-move-reply-claim"),
    pytest.param("8/8/8/4k3/8/8/3R4/4K3 w - - 99 1", id="fifty-move"),
    pytest.param("8/8/8/4k3/8/8/3bK3/8 w - - 0 1", id="insufficient-material-capture"),
    pytest.param("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", id="no-terminal"),
]


@pytest.mark.skipif(CBoard is None, reason="CBoard extension not available")
@pytest.mark.parametrize("fen", _PARITY_FENS)
def test_cboard_root_terminal_scan_matches_python_chess(fen: str) -> None:
    """The C terminal scan must agree with python-chess on every 1-move terminal.

    Mate has priority over draw adjudication (matching FIDE: a checkmating move is
    never reported as a draw), and the helper returns a single mate action with an
    empty draw set when any mate exists.
    """
    board = chess.Board(fen)
    py_mates, py_draws = _py_root_terminals(board)

    cboard_cls = _require_cboard()
    cb = cboard_cls.from_board(board)
    mate, draws = root_tactics.immediate_terminal_cboard_policy_or_draws(
        cb, cb.legal_move_indices(),
    )

    if py_mates:
        assert mate is not None, f"missed mate in {fen}"
        _, action, value = mate
        assert action in py_mates
        assert value == 1.0
        assert draws == set(), "mate must take priority over draw adjudication"
    else:
        assert mate is None
        assert draws == py_draws, f"draw-action mismatch in {fen}"
