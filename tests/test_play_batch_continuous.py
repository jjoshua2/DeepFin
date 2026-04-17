"""Tests for play_batch continuous mode, recycling, and on_game_complete callback.

These tests lock down behavior prior to the Phase 5 refactor of play_batch.
"""
from __future__ import annotations

import chess
import numpy as np
import torch

from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.manager import CompletedGameBatch
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish.uci import StockfishResult


class _UniformModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        return {
            "policy": torch.zeros((b, 4672), dtype=torch.float32, device=x.device),
            "wdl": torch.zeros((b, 3), dtype=torch.float32, device=x.device),
        }


class _FakeStockfish:
    def __init__(self, wdl: list[float]):
        self.nodes = 1
        self._wdl = np.asarray(wdl, dtype=np.float32)

    def search(self, fen: str, *, nodes: int | None = None) -> StockfishResult:  # noqa: ARG002
        board = chess.Board(fen)
        move = next(iter(board.legal_moves), chess.Move.null())
        return StockfishResult(bestmove_uci=move.uci(), wdl=self._wdl, pvs=[])


def test_continuous_mode_stops_on_stop_fn_and_delivers_via_callback():
    """Continuous mode: stop_fn terminates the loop; samples flow via callback."""
    model = _UniformModel().eval()
    rng = np.random.default_rng(0)

    completed: list[CompletedGameBatch] = []

    # Stop once we've seen at least 2 completed games.
    def _stop() -> bool:
        return len(completed) >= 2

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.0, 1.0, 0.0]),
        games=2,
        target_games=0,  # continuous mode
        stop_fn=_stop,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    # Continuous mode must NOT accumulate samples in the returned list
    # (they are delivered via on_game_complete to bound memory growth).
    assert samples == []
    # But stats should still reflect games played.
    assert stats.games >= 2
    # Callback should have received each completed game.
    assert len(completed) >= 2
    # Each CompletedGameBatch reports its own positions count.
    for cg in completed:
        assert isinstance(cg, CompletedGameBatch)
        assert cg.positions == len(cg.samples)


def test_continuous_mode_recycles_slots_beyond_initial_batch():
    """Continuous mode with small batch_size must recycle slots indefinitely."""
    model = _UniformModel().eval()
    rng = np.random.default_rng(1)

    completed: list[CompletedGameBatch] = []

    def _stop() -> bool:
        return len(completed) >= 4  # more than batch_size

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.0, 1.0, 0.0]),
        games=2,  # only 2 parallel slots
        target_games=0,
        stop_fn=_stop,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    # Slot recycling must produce more completed games than the batch size.
    assert stats.games >= 4
    assert len(completed) >= 4
    assert samples == []


def test_continuous_mode_stats_sum_matches_callback_sum():
    """Aggregated BatchStats must match the sum of individual CompletedGameBatch
    deliveries, with no double counting after slot recycling."""
    model = _UniformModel().eval()
    rng = np.random.default_rng(2)

    completed: list[CompletedGameBatch] = []

    def _stop() -> bool:
        return len(completed) >= 3

    _samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.0, 1.0, 0.0]),
        games=2,
        target_games=0,
        stop_fn=_stop,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        # All curriculum games with the fake SF's draw-only WDL.
        game=GameConfig(max_plies=2, selfplay_fraction=0.0),
    )

    # Continuous mode clears all_samples to bound memory, so stats.positions
    # is always 0; all position accounting is delivered via the callback.
    assert stats.positions == 0
    assert sum(cg.games for cg in completed) == stats.games
    assert sum(cg.w for cg in completed) == stats.w
    assert sum(cg.d for cg in completed) == stats.d
    assert sum(cg.l for cg in completed) == stats.l
    assert sum(cg.total_game_plies for cg in completed) == stats.total_game_plies
    assert sum(cg.adjudicated_games for cg in completed) == stats.adjudicated_games
    assert sum(cg.curriculum_games for cg in completed) == stats.curriculum_games
    assert sum(cg.selfplay_games for cg in completed) == stats.selfplay_games


def test_finite_mode_does_not_deliver_via_callback_when_none_given():
    """Finite mode without callback: samples are returned directly, no crash."""
    model = _UniformModel().eval()
    rng = np.random.default_rng(3)

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.0, 1.0, 0.0]),
        games=2, target_games=2,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=0.0),
    )

    assert stats.games == 2
    assert len(samples) == stats.positions


def test_finite_mode_with_callback_also_delivers_via_callback():
    """When both finite target_games and callback are set, both paths deliver."""
    model = _UniformModel().eval()
    rng = np.random.default_rng(4)

    completed: list[CompletedGameBatch] = []

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.0, 1.0, 0.0]),
        games=2, target_games=3,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=0.0),
    )

    # Finite mode returns samples AND calls callback.
    assert stats.games == 3
    assert len(completed) == 3
    # Sample counts match.
    assert sum(cg.positions for cg in completed) == len(samples)


def test_continuous_mode_respects_on_step_callback():
    """on_step should be invoked during the main loop in continuous mode."""
    model = _UniformModel().eval()
    rng = np.random.default_rng(5)

    step_calls: list[int] = []
    completed: list[CompletedGameBatch] = []

    def _on_step() -> None:
        step_calls.append(1)

    def _stop() -> bool:
        return len(completed) >= 2

    play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.0, 1.0, 0.0]),
        games=2,
        target_games=0,
        stop_fn=_stop,
        on_step=_on_step,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    # on_step should fire at least once per main-loop iteration.
    assert len(step_calls) > 0
