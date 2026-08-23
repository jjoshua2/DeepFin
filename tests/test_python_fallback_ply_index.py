from __future__ import annotations

from unittest.mock import Mock

import chess
import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.network_turn import _append_records_via_python
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import SelfplayState


def _state_from_fen(fen: str) -> SelfplayState:
    evaluator = Mock(spec=["evaluate_encoded"])
    stockfish = Mock(spec=["search", "nodes"])
    stockfish.nodes = 0
    state = SelfplayState.create(
        model=None,
        device="cpu",
        rng=np.random.default_rng(0),
        stockfish=stockfish,
        evaluator=evaluator,
        batch_size=1,
        continuous=False,
        target=1,
        opponent=OpponentConfig(),
        temp=TemperatureConfig(),
        search=SearchConfig(),
        opening=OpeningConfig(),
        diff_focus=DiffFocusConfig(),
        game=GameConfig(),
    )
    board = chess.Board(fen)
    state.boards[0] = board
    # Match SelfplayState.create/recycle_slot exactly: production uses
    # CBoard.from_board(), not the history-free from_raw fast constructor.
    state.cboards[0] = CBoard.from_board(board)
    state.starting_ply_arr[0] = board.ply()
    return state


def test_python_fallback_records_absolute_fen_ply_like_c_path() -> None:
    # A FEN-created board has no local move-stack history, but its fullmove
    # counter still places it at an absolute game ply. The production CBoard
    # preserves that absolute ply; the Python fallback must not reset it to 0.
    state = _state_from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 69",
    )
    board = state.boards[0]
    assert len(board.move_stack) == 0
    assert board.ply() == 136
    absolute_ply = int(state.cboards[0].ply)
    assert absolute_ply == board.ply()

    mask = legal_move_mask(board)
    action = int(np.flatnonzero(mask)[0])
    probs = np.zeros((POLICY_SIZE,), dtype=np.float32)
    probs[action] = 1.0
    planes = input_plane_count(state.game.input_extra_features)

    _append_records_via_python(
        state,
        [0],
        xs_batch=np.zeros((1, planes, 8, 8), dtype=np.float32),
        pol_logits=np.zeros((1, POLICY_SIZE), dtype=np.float32),
        wdl_est=np.array([[0.4, 0.2, 0.4]], dtype=np.float32),
        probs_list=[probs],
        actions=[action],
        values_list=[0.0],
        gumbel_diags=[None],
        masks_list=[mask],
        is_full=np.ones((1,), dtype=bool),
        sample_weights=[1.0],
        diff_focus=DiffFocusConfig(),
    )

    assert state.samples_per_game[0][0].ply_index == absolute_ply
