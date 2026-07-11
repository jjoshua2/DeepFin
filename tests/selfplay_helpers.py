from __future__ import annotations

import chess
import numpy as np
import torch

from chess_anti_engine.stockfish.uci import StockfishResult, StockfishUCI


class UniformPolicyValueModel(torch.nn.Module):
    """Stub net: compact 1858 policy (same width as ChessNet); search expands to 4672."""

    policy_encoding = "lc0_1858"

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        return {
            "policy": torch.zeros((batch_size, 1858), dtype=torch.float32, device=x.device),
            "wdl": torch.zeros((batch_size, 3), dtype=torch.float32, device=x.device),
        }


class FakeStockfish(StockfishUCI):
    def __init__(self, wdl: list[float]):  # pyright: ignore[reportMissingSuperCall]
        self.nodes = 1
        self._wdl = np.asarray(wdl, dtype=np.float32)

    def search(
        self,
        fen: str,
        *,
        nodes: int | None = None,
        syzygy_path: str | None = None,
        fresh: bool = False,
    ) -> StockfishResult:
        del syzygy_path
        del nodes
        del fresh
        board = chess.Board(fen)
        move = next(iter(board.legal_moves), chess.Move.null())
        return StockfishResult(bestmove_uci=move.uci(), wdl=self._wdl, pvs=[])
