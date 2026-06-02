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
