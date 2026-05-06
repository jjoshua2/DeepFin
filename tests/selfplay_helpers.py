from __future__ import annotations

import chess
import numpy as np
import torch

from chess_anti_engine.stockfish.uci import StockfishResult, StockfishUCI


class UniformPolicyValueModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        return {
            "policy": torch.zeros((batch_size, 4672), dtype=torch.float32, device=x.device),
            "wdl": torch.zeros((batch_size, 3), dtype=torch.float32, device=x.device),
        }


class FakeStockfish(StockfishUCI):
    def __init__(self, wdl: list[float]):  # pyright: ignore[reportMissingSuperCall]
        self.nodes = 1
        self._wdl = np.asarray(wdl, dtype=np.float32)

    def search(self, fen: str, *, nodes: int | None = None) -> StockfishResult:  # noqa: ARG002
        del nodes
        board = chess.Board(fen)
        move = next(iter(board.legal_moves), chess.Move.null())
        return StockfishResult(bestmove_uci=move.uci(), wdl=self._wdl, pvs=[])
