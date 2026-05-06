from __future__ import annotations

from unittest.mock import MagicMock

import chess

from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import CmdPosition


def test_invalid_position_fen_clears_pending_state() -> None:
    engine = Engine(worker=MagicMock())
    engine._handle_position(CmdPosition(fen=None, moves=("e2e4",)))  # noqa: SLF001

    assert engine._pending_moves == [chess.Move.from_uci("e2e4")]  # noqa: SLF001

    engine._handle_position(CmdPosition(fen="not a valid fen", moves=()))  # noqa: SLF001

    assert engine._board == chess.Board()  # noqa: SLF001
    assert engine._pending_fen is None  # noqa: SLF001
    assert engine._pending_moves == []  # noqa: SLF001
    assert engine._applied_fen is None  # noqa: SLF001
    assert engine._applied_moves == ()  # noqa: SLF001
    assert engine._popped_ponder_move is None  # noqa: SLF001
