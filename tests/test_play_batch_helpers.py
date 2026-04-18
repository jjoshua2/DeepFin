"""Unit tests for the module-scope helpers lifted out of ``play_batch``.

These tests exercise the pieces that used to be nested closures inside
``play_batch`` and are now importable as plain module-scope functions
(``_sf_terminal_result``, ``_init_play_batch_state``). The more stateful
helpers (``_finalize_game``, ``_process_sf_results``, ``_run_network_turn``)
are covered end-to-end by the existing selfplay / continuous / smoke tests.
"""
from __future__ import annotations

import numpy as np

from chess_anti_engine.selfplay.config import GameConfig
from chess_anti_engine.selfplay.manager import (
    _PlayBatchState,
    _init_play_batch_state,
    _sf_terminal_result,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish.uci import StockfishResult


class _FakeStockfish:
    """Minimal Stockfish stand-in that only exposes the ``nodes`` attribute
    read by ``_init_play_batch_state``."""

    def __init__(self, nodes: int = 0) -> None:
        self.nodes = nodes


# ───────────────────────── _sf_terminal_result ──────────────────────────

def test_sf_terminal_result_returns_draw_when_result_is_none() -> None:
    """No Stockfish result ⇒ draw (conservative fallback)."""
    assert _sf_terminal_result(
        turn_is_white=True, sf_res=None, adjudication_threshold=0.6,
    ) == "1/2-1/2"


def test_sf_terminal_result_returns_draw_when_wdl_is_none() -> None:
    """Stockfish result without WDL ⇒ draw."""
    res = StockfishResult(bestmove_uci="e2e4", wdl=None, pvs=[])
    assert _sf_terminal_result(
        turn_is_white=True, sf_res=res, adjudication_threshold=0.6,
    ) == "1/2-1/2"


def test_sf_terminal_result_adjudicates_white_win_from_white_pov() -> None:
    """High W-from-white ⇒ 1-0 when side-to-move is white."""
    # WDL is reported from side-to-move POV; white-to-move means wdl[0]
    # (win) is already from white's POV.
    res = StockfishResult(
        bestmove_uci="e2e4", wdl=np.array([0.9, 0.05, 0.05], dtype=np.float32),
        pvs=[],
    )
    assert _sf_terminal_result(
        turn_is_white=True, sf_res=res, adjudication_threshold=0.6,
    ) == "1-0"


def test_sf_terminal_result_adjudicates_white_win_from_black_pov() -> None:
    """Black-to-move sees its own loss ⇒ white wins (POV flip)."""
    # wdl_stm[2] (loss for side-to-move) ≈ white win when it's black to move.
    res = StockfishResult(
        bestmove_uci="e7e5", wdl=np.array([0.05, 0.05, 0.9], dtype=np.float32),
        pvs=[],
    )
    assert _sf_terminal_result(
        turn_is_white=False, sf_res=res, adjudication_threshold=0.6,
    ) == "1-0"


def test_sf_terminal_result_adjudicates_black_win_from_white_pov() -> None:
    """White-to-move sees a loss ⇒ black wins."""
    res = StockfishResult(
        bestmove_uci="e2e4", wdl=np.array([0.05, 0.05, 0.9], dtype=np.float32),
        pvs=[],
    )
    assert _sf_terminal_result(
        turn_is_white=True, sf_res=res, adjudication_threshold=0.6,
    ) == "0-1"


def test_sf_terminal_result_returns_draw_when_no_side_crosses_threshold() -> None:
    """Neither win nor loss above threshold ⇒ draw."""
    res = StockfishResult(
        bestmove_uci="e2e4", wdl=np.array([0.45, 0.1, 0.45], dtype=np.float32),
        pvs=[],
    )
    assert _sf_terminal_result(
        turn_is_white=True, sf_res=res, adjudication_threshold=0.6,
    ) == "1/2-1/2"


# ───────────────────────── _init_play_batch_state ───────────────────────

def test_init_play_batch_state_creates_correctly_shaped_arrays() -> None:
    """Per-slot arrays should all have length ``batch_size``."""
    rng = np.random.default_rng(0)
    state = _init_play_batch_state(
        batch_size=4,
        continuous=False,
        target=8,
        rng=rng,
        stockfish=_FakeStockfish(nodes=100),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=2, selfplay_fraction=0.5),
    )

    assert isinstance(state, _PlayBatchState)
    assert state.batch_size == 4
    assert state.continuous is False
    assert state.target == 8

    # All per-slot arrays/lists have length batch_size.
    assert len(state.boards) == 4
    assert len(state.cboards) == 4
    assert len(state.move_idx_history) == 4
    assert state.done_arr.shape == (4,)
    assert state.finalized_arr.shape == (4,)
    assert state.net_color_arr.shape == (4,)
    assert state.selfplay_arr.shape == (4,)
    assert len(state.root_ids) == 4
    assert len(state.consecutive_low_winrate) == 4
    assert len(state.sf_resign_scale) == 4
    assert len(state.last_net_full) == 4
    assert len(state.samples_per_game) == 4


def test_init_play_batch_state_initial_flags_and_counters() -> None:
    """done / finalized start zero; games_started = batch_size; stats zero."""
    rng = np.random.default_rng(0)
    state = _init_play_batch_state(
        batch_size=3,
        continuous=True,
        target=0,
        rng=rng,
        stockfish=_FakeStockfish(nodes=100),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=2),
    )

    assert np.all(state.done_arr == 0)
    assert np.all(state.finalized_arr == 0)
    # games_started is primed to batch_size (the initial slots already exist).
    assert state.games_started == 3
    assert state.games_completed == 0
    # Stats all start zero.
    assert state.st_w == state.st_d == state.st_l == 0
    assert state.st_game_plies == 0
    assert state.st_adjudicated == 0
    assert state.st_draw == 0
    assert state.st_sf_d6_sum == 0.0
    assert state.st_sf_d6_n == 0
    assert state.st_checkmate == 0
    assert state.st_stalemate == 0
    # root_ids pre-fill with -1 (no cached subtree yet).
    assert state.root_ids == [-1, -1, -1]
    # soft-resign trackers start at zero / 1.0 / True.
    assert state.consecutive_low_winrate == [0, 0, 0]
    assert state.sf_resign_scale == [1.0, 1.0, 1.0]
    assert state.last_net_full == [True, True, True]
    # All-samples buffer starts empty.
    assert state.all_samples == []
    # Per-slot sample lists are independent objects (not aliased to each other).
    assert state.samples_per_game[0] is not state.samples_per_game[1]


def test_init_play_batch_state_net_color_alternates_even_odd() -> None:
    """Slot i gets network-plays-white for even i, black for odd i."""
    rng = np.random.default_rng(0)
    state = _init_play_batch_state(
        batch_size=6,
        continuous=False,
        target=6,
        rng=rng,
        stockfish=_FakeStockfish(nodes=100),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=2),
    )
    # Even slots ⇒ WHITE (1), odd ⇒ BLACK (0).
    expected = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
    np.testing.assert_array_equal(state.net_color_arr, expected)


def test_init_play_batch_state_selfplay_fraction_zero_has_no_selfplay_slots() -> None:
    """selfplay_fraction=0 ⇒ every slot is curriculum (selfplay_arr all zero)."""
    rng = np.random.default_rng(0)
    state = _init_play_batch_state(
        batch_size=8,
        continuous=False,
        target=8,
        rng=rng,
        stockfish=_FakeStockfish(nodes=100),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=2, selfplay_fraction=0.0),
    )
    assert int(state.selfplay_arr.sum()) == 0


def test_init_play_batch_state_selfplay_fraction_one_flags_all_slots_selfplay() -> None:
    """selfplay_fraction=1 ⇒ every slot is selfplay."""
    rng = np.random.default_rng(0)
    state = _init_play_batch_state(
        batch_size=8,
        continuous=False,
        target=8,
        rng=rng,
        stockfish=_FakeStockfish(nodes=100),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )
    assert int(state.selfplay_arr.sum()) == 8


def test_init_play_batch_state_terminal_eval_nodes_scales_with_base_nodes() -> None:
    """terminal_eval_nodes = 5 * stockfish.nodes when base_nodes > 0."""
    rng = np.random.default_rng(0)
    state = _init_play_batch_state(
        batch_size=2,
        continuous=False,
        target=2,
        rng=rng,
        stockfish=_FakeStockfish(nodes=200),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=2),
    )
    assert state.terminal_eval_nodes == 5 * 200


def test_init_play_batch_state_terminal_eval_nodes_defaults_when_base_is_zero() -> None:
    """With no SF node budget, fall back to 1000-node terminal eval."""
    rng = np.random.default_rng(0)
    state = _init_play_batch_state(
        batch_size=2,
        continuous=False,
        target=2,
        rng=rng,
        stockfish=_FakeStockfish(nodes=0),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=2),
    )
    assert state.terminal_eval_nodes == 1000


def test_init_play_batch_state_normalizes_volatility_source() -> None:
    """volatility_source values other than 'raw'/'search' are coerced to 'raw'."""
    rng = np.random.default_rng(0)
    state = _init_play_batch_state(
        batch_size=1,
        continuous=False,
        target=1,
        rng=rng,
        stockfish=_FakeStockfish(nodes=0),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=2, volatility_source="bogus"),
    )
    assert state.vs == "raw"


def test_init_play_batch_state_accepts_search_volatility_source() -> None:
    rng = np.random.default_rng(0)
    state = _init_play_batch_state(
        batch_size=1,
        continuous=False,
        target=1,
        rng=rng,
        stockfish=_FakeStockfish(nodes=0),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=2, volatility_source="search"),
    )
    assert state.vs == "search"


def test_init_play_batch_state_starting_boards_tracked_only_with_syzygy() -> None:
    """starting_boards snapshot is only kept when syzygy rescoring is enabled."""
    rng = np.random.default_rng(0)
    state_without = _init_play_batch_state(
        batch_size=2,
        continuous=False,
        target=2,
        rng=rng,
        stockfish=_FakeStockfish(nodes=0),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=2, syzygy_path=""),
    )
    assert state_without.starting_boards is None

    rng2 = np.random.default_rng(0)
    state_with = _init_play_batch_state(
        batch_size=2,
        continuous=False,
        target=2,
        rng=rng2,
        stockfish=_FakeStockfish(nodes=0),
        opening=OpeningConfig(random_start_plies=0),
        # Path doesn't need to exist — _init only checks truthiness.
        game=GameConfig(max_plies=2, syzygy_path="/tmp/syzygy-placeholder"),
    )
    assert state_with.starting_boards is not None
    assert len(state_with.starting_boards) == 2
