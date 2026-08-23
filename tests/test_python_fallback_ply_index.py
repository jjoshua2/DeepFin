from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask
from chess_anti_engine.selfplay.blindspot_harvest import pre_move_boards
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay import finalize as finalize_mod
from chess_anti_engine.selfplay.network_turn import _append_records_via_python
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.resume import RESUME_FORMAT_VERSION, should_resume_game
from chess_anti_engine.selfplay.state import SelfplayState


_FEN_PLY_136 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 69"


def _state_from_fen(fen: str, *, game: GameConfig | None = None) -> SelfplayState:
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
        game=GameConfig() if game is None else game,
    )
    board = chess.Board(fen)
    state.boards[0] = board
    # Match SelfplayState.create/recycle_slot exactly: production uses
    # CBoard.from_board(), not the history-free from_raw fast constructor.
    state.cboards[0] = CBoard.from_board(board)
    assert state.starting_boards is not None
    state.starting_boards[0] = board.copy()
    state.starting_ply_arr[0] = board.ply()
    return state


def test_python_fallback_records_absolute_fen_ply_like_c_path() -> None:
    # A FEN-created board has no local move-stack history, but its fullmove
    # counter still places it at an absolute game ply. The production CBoard
    # preserves that absolute ply; the Python fallback must not reset it to 0.
    state = _state_from_fen(_FEN_PLY_136)
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


def test_blindspot_reconstruction_matches_absolute_fen_ply() -> None:
    starting = chess.Board(_FEN_PLY_136)
    final = starting.copy()
    played_move = chess.Move.from_uci("e2e4")
    final.push(played_move)

    boards, played = pre_move_boards(
        starting,
        list(final.move_stack),
        [starting.ply()],
        opening_len=0,
    )

    assert boards[0] is not None
    assert boards[0].fen() == starting.fen()
    assert played[0] == played_move


def test_syzygy_policy_rescore_matches_absolute_fen_ply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = GameConfig(syzygy_path="/fake", syzygy_rescore_policy=True)
    state = _state_from_fen(_FEN_PLY_136, game=game)
    # Exercise the Python-fallback replay shape: the starting FEN has an empty
    # local move stack even though its absolute game ply is 136.
    state.has_c_ply = False
    starting = state.starting_boards[0] if state.starting_boards is not None else None
    assert starting is not None
    final = starting.copy()
    played_move = chess.Move.from_uci("e2e4")
    final.push(played_move)

    monkeypatch.setattr(finalize_mod, "rescore_game_samples", lambda *_args: None)
    monkeypatch.setattr(finalize_mod, "is_tb_eligible", lambda _board: True)
    monkeypatch.setattr(finalize_mod, "probe_best_move", lambda _board, _path: played_move)

    result, overrides = finalize_mod._rescore_with_syzygy(
        state,
        0,
        final,
        [SimpleNamespace(ply_index=starting.ply())],
        "1/2-1/2",
    )

    assert result == "1/2-1/2"
    assert 0 in overrides
    assert float(overrides[0].sum()) == pytest.approx(1.0)


def test_pre_absolute_ply_resume_format_is_rejected() -> None:
    # v2 Python-fallback records used move-stack-relative ply indices. They
    # must never be resumed into a process that records new absolute v3 plies.
    assert RESUME_FORMAT_VERSION >= 3
    assert should_resume_game(
        {"format_version": 2, "compat_fingerprint": "same"},
        compat_fingerprint="same",
    ) == "version_mismatch"
