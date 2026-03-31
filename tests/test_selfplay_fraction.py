from __future__ import annotations

import chess
import numpy as np
import torch

from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import TemperatureConfig, SearchConfig, DiffFocusConfig, GameConfig
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


def test_full_selfplay_generates_both_side_samples_and_no_pid_wdl_stats():
    model = _UniformModel().eval()
    rng = np.random.default_rng(0)

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.6, 0.2, 0.2]),
        games=1,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    assert len(samples) >= 2
    assert stats.w == 0
    assert stats.d == 0
    assert stats.l == 0
    assert any(s.sf_wdl is not None for s in samples)


def test_target_games_recycles_slots_and_preserves_selfplay_accounting():
    model = _UniformModel().eval()
    rng = np.random.default_rng(1)
    target_games = 5

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.0, 1.0, 0.0]),
        games=2, target_games=target_games,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    assert stats.games == target_games
    assert stats.selfplay_games == target_games
    assert stats.curriculum_games == 0
    assert stats.selfplay_games + stats.curriculum_games == stats.games
    assert stats.adjudicated_games == target_games
    assert stats.selfplay_adjudicated_games == target_games
    assert stats.curriculum_adjudicated_games == 0
    assert stats.w == 0
    assert stats.d == 0
    assert stats.l == 0
    assert stats.positions == len(samples)
    assert len(samples) > 0


def test_target_games_recycles_slots_and_preserves_curriculum_accounting():
    model = _UniformModel().eval()
    rng = np.random.default_rng(2)
    target_games = 5

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.0, 1.0, 0.0]),
        games=2, target_games=target_games,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=0.0),
    )

    assert stats.games == target_games
    assert stats.selfplay_games == 0
    assert stats.curriculum_games == target_games
    assert stats.selfplay_games + stats.curriculum_games == stats.games
    assert stats.adjudicated_games == target_games
    assert stats.selfplay_adjudicated_games == 0
    assert stats.curriculum_adjudicated_games == target_games
    assert stats.w == 0
    assert stats.d == target_games
    assert stats.l == 0
    assert stats.positions == len(samples)
    assert len(samples) > 0


def test_target_games_smaller_than_batch_does_not_over_start_games():
    model = _UniformModel().eval()
    rng = np.random.default_rng(3)

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.0, 1.0, 0.0]),
        games=8, target_games=1,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    assert stats.games == 1
    assert stats.selfplay_games == 1
    assert stats.curriculum_games == 0
    assert stats.positions == len(samples)
    assert len(samples) > 0


def test_play_batch_can_return_zero_samples_when_no_full_playouts():
    model = _UniformModel().eval()
    rng = np.random.default_rng(4)

    samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([0.0, 1.0, 0.0]),
        games=2, target_games=2,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=0.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
    )

    assert stats.games == 2
    assert stats.positions == 0
    assert samples == []
