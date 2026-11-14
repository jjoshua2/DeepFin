"""A rejected FEN must never be searched: the next go answers 0000 without
touching the tree, and the next VALID position is served normally."""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves.encode import move_to_index
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchWorker

_OLD_FEN = "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2"
_INVALID_FENS = ["invalid w - - 0 1", "8/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1"]


class _Evaluator:
    def __init__(self):
        child = chess.Board(_OLD_FEN)
        child.push_uci("g1f3")
        self.black_action = move_to_index(chess.Move.from_uci("e5e4"), child)

    def evaluate_encoded(self, x, relations=None):
        del relations
        pol = np.zeros((len(x), 4672), np.float32)
        pol[:, self.black_action] = 12
        return pol, np.zeros((len(x), 3), np.float32)


def _join(engine):
    assert engine._search_thread is not None
    engine._search_thread.join(timeout=10)
    assert not engine._search_thread.is_alive()


def _searched_engine(mode):
    worker = SearchWorker(
        _Evaluator(), device="cpu", n_walkers=2, chunk_sims=16,
        gumbel_cfg=GumbelConfig(input_extra_features="v1", input_history_encoding=mode,
                                add_noise=False),
    )
    engine = Engine(worker)
    engine.dispatch(parse_command("position fen " + _OLD_FEN))
    engine.dispatch(parse_command("go nodes 64"))
    _join(engine)
    assert worker._tree is not None
    assert worker._root_id is not None
    child = worker._tree.find_child(
        worker._root_id, move_to_index(chess.Move.from_uci("g1f3"), chess.Board(_OLD_FEN)),
    )
    assert child >= 0
    assert worker._tree.is_expanded(child)
    black_board = chess.Board(_OLD_FEN)
    black_board.push_uci("g1f3")
    action = move_to_index(chess.Move.from_uci("e5e4"), black_board)
    assert worker._tree.find_child(child, action) >= 0
    actions, _ = worker._tree.get_children_visits(worker._root_id)
    assert len(actions) == 20
    assert set(map(int, actions)) == set(map(int, CBoard.from_board(chess.Board()).legal_move_indices()))
    return engine, worker


@pytest.mark.parametrize("mode", ["legacy", "lc0_root", "lc0_root_legacy_meta"])
@pytest.mark.parametrize("fen", _INVALID_FENS)
def test_rejected_fen_answers_0000_and_never_reaches_the_tree(mode, fen, monkeypatch, capsys):
    engine, worker = _searched_engine(mode)
    old_board = engine._board.copy()
    captured = []
    try:
        capsys.readouterr()
        engine.dispatch(parse_command("position fen " + fen))
        assert engine._board == old_board  # the previous position, kept
        assert engine._force_tree_reset is False
        assert "position: rejected" in capsys.readouterr().out
        assert worker._walker_pool is not None

        def capture(**kwargs):
            captured.append((kwargs["tree"], kwargs["root_cboard"].fen()))
            raise RuntimeError("intercept before native descent")

        monkeypatch.setattr(worker._walker_pool, "run", capture)
        engine.dispatch(parse_command("go nodes 1"))
        out = capsys.readouterr().out
        assert "go refused: the last position command was rejected" in out
        assert out.strip().splitlines()[-1] == "bestmove 0000"
        assert captured == []  # nothing was searched
        # The next valid position is served by a normal search again.
        engine.dispatch(parse_command("position startpos"))
        engine.dispatch(parse_command("go nodes 1"))
        _join(engine)
        assert len(captured) == 1
        assert captured[0][1] == CBoard.from_board(chess.Board()).fen()
    finally:
        engine.close()


def test_valid_same_fen_keeps_useful_tree_reuse(monkeypatch):
    engine, worker = _searched_engine("legacy")
    old_tree = worker._tree
    captured = []
    try:
        engine.dispatch(parse_command("position fen " + _OLD_FEN))
        assert worker._walker_pool is not None

        def capture(**kwargs):
            captured.append((kwargs["tree"], kwargs["root_cboard"].fen()))
            raise RuntimeError("intercept before native descent")

        monkeypatch.setattr(worker._walker_pool, "run", capture)
        engine.dispatch(parse_command("go nodes 1"))
        _join(engine)
        assert len(captured) == 1
        assert captured[0][0] is old_tree
        assert captured[0][1] == CBoard.from_board(chess.Board(_OLD_FEN)).fen()
    finally:
        engine.close()
