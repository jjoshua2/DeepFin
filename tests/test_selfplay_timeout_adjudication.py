import numpy as np
import torch

from chess_anti_engine.selfplay.manager import play_batch
from chess_anti_engine.stockfish.uci import StockfishResult


class _DummyModel(torch.nn.Module):
    def forward(self, x):
        raise AssertionError("Model should not be called when max_plies=0")


class _FakeStockfish:
    def __init__(self, wdl):
        self._wdl = np.asarray(wdl, dtype=np.float32)

    def search(self, fen: str, *, nodes: int | None = None):
        return StockfishResult(bestmove_uci="0000", wdl=self._wdl, pvs=[])


def test_timeout_adjudication_keeps_89pct_as_draw():
    samples, stats = play_batch(
        _DummyModel().eval(),
        device="cpu",
        rng=np.random.default_rng(0),
        stockfish=_FakeStockfish([0.89, 0.11, 0.0]),
        games=1,
        temperature=1.0,
        max_plies=0,
    )
    assert samples == []
    assert stats.games == 1
    assert stats.w == 0 and stats.d == 1 and stats.l == 0


def test_timeout_adjudication_marks_91pct_as_win():
    samples, stats = play_batch(
        _DummyModel().eval(),
        device="cpu",
        rng=np.random.default_rng(0),
        stockfish=_FakeStockfish([0.91, 0.09, 0.0]),
        games=1,
        temperature=1.0,
        max_plies=0,
    )
    assert samples == []
    assert stats.games == 1
    assert stats.w == 1 and stats.d == 0 and stats.l == 0


def test_timeout_adjudication_honors_custom_threshold():
    samples, stats = play_batch(
        _DummyModel().eval(),
        device="cpu",
        rng=np.random.default_rng(0),
        stockfish=_FakeStockfish([0.81, 0.19, 0.0]),
        games=1,
        temperature=1.0,
        max_plies=0,
        timeout_adjudication_threshold=0.80,
    )
    assert samples == []
    assert stats.games == 1
    assert stats.w == 1 and stats.d == 0 and stats.l == 0
