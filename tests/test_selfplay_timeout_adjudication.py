from __future__ import annotations

import numpy as np
import torch

from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import GameConfig, TemperatureConfig
from chess_anti_engine.selfplay.opening import OpeningConfig
from tests.selfplay_helpers import FakeStockfish


def test_timeout_adjudication_threshold_white_to_move_win():
    # Starting position is white to move. If SF says white has > threshold win prob,
    # the timeout is labeled as a white win.
    model = torch.nn.Linear(1, 1)
    rng = np.random.default_rng(0)

    _samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.95, 0.05, 0.00]),
        games=1,
        temp=TemperatureConfig(temperature=1.0),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=0, timeout_adjudication_threshold=0.90),
    )

    assert stats.w == 1
    assert stats.d == 0
    assert stats.l == 0


def test_timeout_adjudication_threshold_white_to_move_draw_below_threshold():
    model = torch.nn.Linear(1, 1)
    rng = np.random.default_rng(1)

    _samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.89, 0.11, 0.00]),
        games=1,
        temp=TemperatureConfig(temperature=1.0),
        opening=OpeningConfig(random_start_plies=0),
        game=GameConfig(max_plies=0, timeout_adjudication_threshold=0.90),
    )

    assert stats.w == 0
    assert stats.d == 1
    assert stats.l == 0


def test_timeout_adjudication_threshold_flip_when_black_to_move():
    # If it's black to move and SF says side-to-move has > threshold win prob, then
    # from white's POV it's a loss.
    model = torch.nn.Linear(1, 1)
    rng = np.random.default_rng(2)

    _samples, stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([0.95, 0.05, 0.00]),
        games=1,
        temp=TemperatureConfig(temperature=1.0),
        opening=OpeningConfig(random_start_plies=1),
        game=GameConfig(max_plies=0, timeout_adjudication_threshold=0.90),
    )

    assert stats.w == 0
    assert stats.d == 0
    assert stats.l == 1
