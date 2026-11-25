import chess
import pytest

from chess_anti_engine.moves.encode import (
    PLANE_COUNT, POLICY_SIZE, UNDERPROMO_DFS, ActionDecodeError, move_to_index,
)
from chess_anti_engine.simple_rl.collect import _uci_moves


def test_collection_replay_rejects_invalid_actions_even_if_final_fen_matches():
    board = chess.Board()
    move = next(iter(board.legal_moves))
    final = board.copy()
    final.push(move)
    valid = move_to_index(move, board)
    assert _uci_moves(board.fen(), [valid], final.fen()) == [move.uci()]
    for index in (-1, valid - POLICY_SIZE, POLICY_SIZE):
        with pytest.raises(ValueError, match="outside"):
            _uci_moves(board.fen(), [index], final.fen())
    in_range_illegal = move_to_index(chess.Move.from_uci("a1a3"), board)
    with pytest.raises(ActionDecodeError):
        _uci_moves(board.fen(), [in_range_illegal], final.fen())
    with pytest.raises(ValueError, match="not the final"):
        _uci_moves(board.fen(), [valid], board.fen())


def test_collection_replay_rejects_noncanonical_underpromotion_alias():
    board = chess.Board("k7/8/8/8/4R3/8/8/4K3 w - - 0 1")
    move = board.parse_uci("e4e5")
    final = board.copy()
    final.push(move)
    assert _uci_moves(board.fen(), [move_to_index(move, board)], final.fen()) == ["e4e5"]
    alias = chess.E4 * PLANE_COUNT + 64 + UNDERPROMO_DFS.index(0)
    with pytest.raises(ValueError, match="noncanonical"):
        _uci_moves(board.fen(), [alias], final.fen())
