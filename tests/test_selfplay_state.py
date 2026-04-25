"""Unit tests for the ``SelfplayState`` + ``_StatsAcc`` helpers introduced
during the play_batch refactor.

These target the new surface area that isn't already covered by the
end-to-end integration tests:

* ``_StatsAcc.to_batch_stats`` — snapshot + SF-delta-6 averaging
* ``SelfplayState.classify_active_slots`` — Python fallback partitioning
  (the C fast path is exercised by ``test_play_batch_continuous`` and
  friends)
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import chess
import numpy as np
import pytest

from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import SelfplayState, _StatsAcc


def test_stats_acc_to_batch_stats_copies_counters():
    acc = _StatsAcc()
    acc.w = 3
    acc.d = 2
    acc.l = 1
    acc.total_game_plies = 100
    acc.adjudicated_games = 4
    acc.tb_adjudicated_games = 2
    acc.plies_win = 30
    acc.plies_draw = 20
    acc.plies_loss = 10
    acc.checkmate_games = 2
    acc.stalemate_games = 1

    stats = acc.to_batch_stats(games=6, positions=200, sf_nodes=42)

    assert stats.games == 6
    assert stats.positions == 200
    assert stats.w == 3
    assert stats.d == 2
    assert stats.l == 1
    assert stats.total_game_plies == 100
    assert stats.adjudicated_games == 4
    assert stats.tb_adjudicated_games == 2
    assert stats.plies_win == 30
    assert stats.plies_draw == 20
    assert stats.plies_loss == 10
    assert stats.checkmate_games == 2
    assert stats.stalemate_games == 1
    assert stats.sf_nodes == 42
    assert stats.sf_nodes_next is None
    assert stats.pid_ema_winrate is None


def test_stats_acc_sf_nodes_zero_becomes_none():
    """A zero or negative nodes value should round-trip as None (worker expects
    this to decide whether to attach PID diagnostics)."""
    stats = _StatsAcc().to_batch_stats(games=0, positions=0, sf_nodes=0)
    assert stats.sf_nodes is None

    stats2 = _StatsAcc().to_batch_stats(games=0, positions=0, sf_nodes=None)
    assert stats2.sf_nodes is None


def test_stats_acc_sf_delta6_mean():
    """sf_eval_delta6 is the mean of sf_d6_sum / sf_d6_n (0.0 when n==0)."""
    acc = _StatsAcc()
    acc.sf_d6_sum = 3.0
    acc.sf_d6_n = 6
    stats = acc.to_batch_stats(games=0, positions=0, sf_nodes=None)
    assert stats.sf_eval_delta6 == pytest.approx(0.5)
    assert stats.sf_eval_delta6_n == 6

    empty = _StatsAcc().to_batch_stats(games=0, positions=0, sf_nodes=None)
    assert empty.sf_eval_delta6 == 0.0
    assert empty.sf_eval_delta6_n == 0


def _make_state(batch_size: int = 4, **game_kwargs) -> SelfplayState:
    """Build a minimal SelfplayState for classification tests.

    Uses a mock evaluator so we don't need a model; C-classify probing
    is left on if the extension is built (it's covered elsewhere).
    """
    rng = np.random.default_rng(0)
    evaluator = Mock(spec=["evaluate_encoded"])
    stockfish = Mock(spec=["search", "nodes"])
    stockfish.nodes = 0
    cfg = GameConfig(selfplay_fraction=0.0, **game_kwargs)
    return SelfplayState.create(
        model=None,
        device="cpu",
        rng=rng,
        stockfish=stockfish,
        evaluator=evaluator,
        batch_size=batch_size,
        continuous=False,
        target=batch_size,
        opponent=OpponentConfig(),
        temp=TemperatureConfig(),
        search=SearchConfig(),
        opening=OpeningConfig(),
        diff_focus=DiffFocusConfig(),
        game=cfg,
    )


def test_classify_active_slots_partitions_by_turn():
    """With selfplay_fraction=0, all games are curriculum; on a fresh
    starting position the network plays WHITE on even slots, BLACK on odd."""
    state = _make_state(batch_size=4)
    # Force the classification: network plays WHITE on even slots, BLACK on odd.
    state.net_color_arr[:] = np.array([1, 0, 1, 0], dtype=np.int8)
    state.selfplay_arr[:] = 0  # all curriculum

    # Fresh boards → WHITE to move → even slots are network turns, odd are
    # curriculum.
    net_idxs, sp_idxs, cur_idxs, all_done = state.classify_active_slots()
    assert not all_done
    # Order is implementation-defined but the partitioning must be correct.
    assert set(net_idxs) == {0, 2}
    assert set(cur_idxs) == {1, 3}
    assert sp_idxs == []

    # Advance every slot's CBoard by one WHITE move → BLACK to move.
    for i in range(state.batch_size):
        legal = state.cboards[i].legal_move_indices()
        state.cboards[i].push_index(int(legal[0]))
    net_idxs, sp_idxs, cur_idxs, _ = state.classify_active_slots()
    # Now BLACK is to move, so the assignment flips.
    assert set(net_idxs) == {1, 3}
    assert set(cur_idxs) == {0, 2}


def test_classify_active_slots_breaks_when_all_finalized():
    state = _make_state(batch_size=3)
    state.finalized_arr[:] = 1  # pretend all games done
    net_idxs, sp_idxs, cur_idxs, all_done = state.classify_active_slots()
    assert all_done
    assert net_idxs == []
    assert sp_idxs == []
    assert cur_idxs == []


def test_classify_active_slots_marks_timed_out():
    """Slots past max_plies or game-over should get done_arr=1 without
    being scheduled for another turn."""
    # Short max_plies so we can force a timeout quickly.
    state = _make_state(batch_size=2, max_plies=1)
    # Push one move so each CBoard's ply == 1 >= max_plies.
    for i in range(2):
        legal = state.cboards[i].legal_move_indices()
        state.cboards[i].push_index(int(legal[0]))

    net_idxs, sp_idxs, cur_idxs, _ = state.classify_active_slots()
    # C path and Python fallback both mark timed-out slots done.
    assert all(state.done_arr[i] == 1 for i in range(2))
    # No live slots remain, so all three partitions are empty.
    assert net_idxs == [] and sp_idxs == [] and cur_idxs == []


def test_selfplay_state_net_color_accessor():
    state = _make_state(batch_size=2)
    state.net_color_arr[:] = np.array([1, 0], dtype=np.int8)
    assert state.net_color(0) == chess.WHITE
    assert state.net_color(1) == chess.BLACK


def test_recycle_slot_resets_state_and_rerolls_selfplay():
    """recycle_slot bumps games_started, resets per-slot counters, and
    re-rolls the selfplay flag from the configured fraction."""
    state = _make_state(batch_size=2)
    # Push a move so the slot has nonzero state to clear.
    legal = state.cboards[0].legal_move_indices()
    state.cboards[0].push_index(int(legal[0]))
    state.move_idx_history[0].append(int(legal[0]))
    state.consecutive_low_winrate[0] = 5
    state.last_net_full[0] = False
    state.done_arr[0] = 1
    state.finalized_arr[0] = 1
    started_before = state.games_started

    # Bump selfplay fraction so the re-roll is deterministic.
    state.game = replace(state.game, selfplay_fraction=1.0)
    state.recycle_slot(0)

    assert state.games_started == started_before + 1
    assert state.move_idx_history[0] == []
    assert state.consecutive_low_winrate[0] == 0
    assert state.last_net_full[0] is True
    assert state.done_arr[0] == 0
    assert state.finalized_arr[0] == 0
    assert state.selfplay_arr[0] == 1  # re-rolled with sp_frac=1.0
