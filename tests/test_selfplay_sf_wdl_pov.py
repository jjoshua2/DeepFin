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

    def search(self, fen: str, *, nodes: int | None = None) -> StockfishResult:  # noqa: ARG002  # pylint: disable=unused-argument  # mock matches StockfishUCI.search signature
        board = chess.Board(fen)
        move = next(iter(board.legal_moves), chess.Move.null())
        return StockfishResult(bestmove_uci=move.uci(), wdl=self._wdl, pvs=[])


def test_sf_wdl_target_is_flipped_to_network_turn_pov_for_both_colors():
    """SF eval from opponent turn should be flipped when attached to network-turn sample."""
    model = _UniformModel().eval()
    rng = np.random.default_rng(0)

    samples, _stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=_FakeStockfish([1.0, 0.0, 0.0]),
        games=2,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=4),
    )

    sf_wdls = [s.sf_wdl for s in samples if s.sf_wdl is not None]
    assert sf_wdls, "Expected at least one sample with sf_wdl"

    # SF reports [1,0,0] for side-to-move at t+1 (opponent turn).
    # Attached target on sample t (network turn) must be flipped => [0,0,1].
    expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    assert any(np.allclose(wdl, expected) for wdl in sf_wdls)
