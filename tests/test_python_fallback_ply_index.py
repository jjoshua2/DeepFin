from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask, move_to_index
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
from chess_anti_engine.selfplay.stockfish_turn import _push_curriculum_opponent_move


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


def _force_python_fallback(state: SelfplayState) -> None:
    """Mirror the coupled C-ply import fallback used by SelfplayState.create."""
    state.has_c_ply = False
    state.c_process_ply = None
    for name in (
        "batch_enc_146",
        "batch_enc_146_lc0_root",
        "batch_enc_146_lc0_root_legacy_meta",
        "batch_enc_146_bf16",
        "batch_enc_146_lc0_root_bf16",
        "batch_enc_146_lc0_root_legacy_meta_bf16",
    ):
        setattr(state, name, None)


def _append_python_net_move(
    state: SelfplayState, uci: str, *, mask: np.ndarray | None = None,
) -> Any:
    """Play one selected CBoard action through the Python fallback recorder."""
    cboard_position = chess.Board(state.cboards[0].fen())
    action = int(move_to_index(chess.Move.from_uci(uci), cboard_position))
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
    return state.samples_per_game[0][-1]


def test_python_fallback_records_absolute_fen_ply_like_c_path() -> None:
    # A FEN-created board has no local move-stack history, but its fullmove
    # counter still places it at an absolute game ply. The production CBoard
    # preserves that absolute ply; the Python fallback must not reset it to 0.
    state = _state_from_fen(_FEN_PLY_136)
    # The dispatcher in run_network_turn only reaches _append_records_via_python
    # when has_c_ply is False. In a built env the constructor sets it True, so
    # leaving it there tests a configuration production never produces.
    _force_python_fallback(state)
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

    rec = state.samples_per_game[0][0]
    assert rec.ply_index == absolute_ply
    # The pre-v3 formula, spelled out so a revert to it cannot pass quietly.
    assert rec.ply_index != len(board.move_stack)


def test_python_fallback_ply_survives_an_intervening_curriculum_sf_ply() -> None:
    """A fallback SF move stays synchronized before the following net move."""
    state = _state_from_fen(_FEN_PLY_136)
    _force_python_fallback(state)
    board = state.boards[0]

    sf_move = chess.Move.from_uci("e2e4")
    sf_move_idx = int(move_to_index(sf_move, board))
    _push_curriculum_opponent_move(
        state, 0,
        legal_indices=state.cboards[0].legal_move_indices(),
        cand_idxs=[sf_move_idx], cand_scores=[0.0],
        regret_limit=float("inf"),
    )
    assert int(state.move_idx_history[0][-1]) == sf_move_idx

    true_ply = int(state.cboards[0].ply)
    assert true_ply == 137
    assert board.ply() == true_ply
    assert len(board.move_stack) == 1
    assert board.fen() == state.cboards[0].fen()

    # The next action comes from the CBoard position and must be legal and
    # position-preserving when decoded and pushed by the Python fallback.
    rec = _append_python_net_move(state, "e7e5")
    assert rec.ply_index == true_ply
    assert board.ply() == true_ply + 1
    assert len(state.move_idx_history[0]) == 2
    assert board.fen() == state.cboards[0].fen()


def test_python_fallback_queen_check_mask_and_reply_match_cboard() -> None:
    fen = "4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1"
    state = _state_from_fen(fen)
    _force_python_fallback(state)
    sf_move = chess.Move.from_uci("e2e7")
    sf_idx = int(move_to_index(sf_move, state.boards[0]))
    _push_curriculum_opponent_move(
        state, 0,
        legal_indices=state.cboards[0].legal_move_indices(),
        cand_idxs=[sf_idx], cand_scores=[0.0], regret_limit=float("inf"),
    )

    assert state.boards[0].fen() == state.cboards[0].fen()
    c_mask = np.zeros((POLICY_SIZE,), dtype=bool)
    c_mask[state.cboards[0].legal_move_indices()] = True
    np.testing.assert_array_equal(legal_move_mask(state.boards[0]), c_mask)
    reply = chess.Move.from_uci("e8e7")
    reply_idx = int(move_to_index(reply, chess.Board(state.cboards[0].fen())))
    assert c_mask[reply_idx]

    rec = _append_python_net_move(state, "e8e7")
    assert rec.legal_mask[reply_idx]
    assert state.boards[0].fen() == state.cboards[0].fen()


def test_curriculum_sf_move_keeps_cboard_replay_authoritative_on_c_path() -> None:
    state = _state_from_fen(chess.STARTING_FEN)
    assert state.has_c_ply
    before = state.boards[0].fen()
    sf_move = chess.Move.from_uci("e2e4")
    sf_idx = int(move_to_index(sf_move, state.boards[0]))

    _push_curriculum_opponent_move(
        state, 0,
        legal_indices=state.cboards[0].legal_move_indices(),
        cand_idxs=[sf_idx], cand_scores=[0.0], regret_limit=float("inf"),
    )

    expected = chess.Board(before)
    expected.push(sf_move)
    assert state.boards[0].fen() == before
    assert state.cboards[0].fen() == expected.fen()
    assert state.move_idx_history[0] == [sf_idx]

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

    # The helper only reads ``ply_index`` from records in this branch. Make the
    # deliberate minimal test double explicit at the typed private API boundary.
    records = cast(Any, [SimpleNamespace(ply_index=starting.ply())])
    result, overrides = finalize_mod._rescore_with_syzygy(
        state,
        0,
        final,
        records,
        "1/2-1/2",
    )

    assert result == "1/2-1/2"
    assert 0 in overrides
    assert float(overrides[0].sum()) == pytest.approx(1.0)


def test_both_walkers_align_for_a_fen_root_with_pushed_opening_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production seed-line shape, untested until now.

    ``opening.seed_board_from_line`` builds a board from a FEN and then PUSHES
    the seed's continuation, so on the Python play path ``starting_board`` has
    BOTH a non-zero absolute ply and a non-empty local move stack. The two
    numbers now disagree three ways — absolute 138, stack length 2, opening_len
    2 — and both replay walkers must key on the absolute one.
    """
    starting = chess.Board(_FEN_PLY_136)
    starting.push(chess.Move.from_uci("e2e4"))
    starting.push(chess.Move.from_uci("e7e5"))
    assert len(starting.move_stack) == 2
    assert starting.ply() == 138
    assert int(CBoard.from_board(starting).ply) == 138

    final = starting.copy()
    net_move = chess.Move.from_uci("g1f3")
    final.push(net_move)
    opening_len = len(starting.move_stack)

    # 1. the blind-spot walker
    boards, played = pre_move_boards(
        starting,
        list(final.move_stack),
        [starting.ply()],
        opening_len=opening_len,
    )
    assert boards[0] is not None
    assert boards[0].fen() == starting.fen()
    assert played[0] == net_move

    # 2. the syzygy-rescore walker, over the same game
    game = GameConfig(syzygy_path="/fake", syzygy_rescore_policy=True)
    state = _state_from_fen(_FEN_PLY_136, game=game)
    state.has_c_ply = False
    state.boards[0] = starting.copy()
    state.cboards[0] = CBoard.from_board(starting)
    assert state.starting_boards is not None
    state.starting_boards[0] = starting.copy()
    state.starting_ply_arr[0] = starting.ply()

    monkeypatch.setattr(finalize_mod, "rescore_game_samples", lambda *_args: None)
    monkeypatch.setattr(finalize_mod, "is_tb_eligible", lambda _board: True)
    monkeypatch.setattr(finalize_mod, "probe_best_move", lambda _board, _path: net_move)

    records = cast(Any, [SimpleNamespace(ply_index=starting.ply())])
    result, overrides = finalize_mod._rescore_with_syzygy(
        state, 0, final, records, "1/2-1/2",
    )
    assert result == "1/2-1/2"
    assert 0 in overrides
    assert float(overrides[0].sum()) == pytest.approx(1.0)


def test_both_walkers_break_a_duplicate_ply_tie_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pre_move_boards`` documents itself as mirroring the syzygy-rescore
    walk, and they keyed the same ply→record map with OPPOSITE tie-breaks:
    ``setdefault`` (first wins) here, a last-wins dict comprehension there.
    Duplicates should not arise, so this only bites on already-corrupt input —
    but two different answers to one question is a silent divergence, not a
    design. Both are first-wins; this fails if either drifts."""
    starting = chess.Board(_FEN_PLY_136)
    final = starting.copy()
    net_move = chess.Move.from_uci("e2e4")
    final.push(net_move)
    duplicate_plies = [starting.ply(), starting.ply()]

    boards, _played = pre_move_boards(
        starting, list(final.move_stack), duplicate_plies, opening_len=0,
    )
    assert boards[0] is not None
    assert boards[1] is None

    game = GameConfig(syzygy_path="/fake", syzygy_rescore_policy=True)
    state = _state_from_fen(_FEN_PLY_136, game=game)
    state.has_c_ply = False
    monkeypatch.setattr(finalize_mod, "rescore_game_samples", lambda *_args: None)
    monkeypatch.setattr(finalize_mod, "is_tb_eligible", lambda _board: True)
    monkeypatch.setattr(finalize_mod, "probe_best_move", lambda _board, _path: net_move)

    records = cast(Any, [SimpleNamespace(ply_index=p) for p in duplicate_plies])
    _result, overrides = finalize_mod._rescore_with_syzygy(
        state, 0, final, records, "1/2-1/2",
    )
    assert set(overrides) == {0}


def test_pre_absolute_ply_resume_format_is_rejected() -> None:
    # v2 Python-fallback records used move-stack-relative ply indices. They
    # must never be resumed into a process that records new absolute v3 plies.
    assert RESUME_FORMAT_VERSION >= 3
    assert should_resume_game(
        {"format_version": 2, "compat_fingerprint": "same"},
        compat_fingerprint="same",
    ) == "version_mismatch"
