from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import chess
import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.moves import (
    POLICY_SIZE,
    index_to_move,
    legal_move_mask,
    sample_move_from_logits,
    move_to_index,
)
from chess_anti_engine.mcts import MCTSConfig, run_mcts
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.stockfish.uci import StockfishUCI


@dataclass
class GameStats:
    result: str  # "1-0", "0-1", "1/2-1/2"
    plies: int


def _result_to_wdl(result: str, *, pov_white: bool) -> int:
    # 0=W,1=D,2=L from side-to-move perspective at that position.
    # python-chess returns "*" when a game is truncated before a terminal result
    # (e.g. max_plies reached). Treat this as a draw target so we don't inject
    # systematic "loss" labels into unfinished games.
    if result in {"1/2-1/2", "*"}:
        return 1
    white_won = result == "1-0"
    if pov_white:
        return 0 if white_won else 2
    return 0 if (not white_won) else 2


def play_game(
    model: torch.nn.Module,
    *,
    device: str,
    rng: np.random.Generator,
    stockfish: StockfishUCI,
    temperature: float,
    max_plies: int,
    mcts_simulations: int = 50,
) -> tuple[list[ReplaySample], GameStats]:
    b = chess.Board()

    # Network-turn samples only (one per full move pair).
    samples: list[ReplaySample] = []

    for _move in range(int(max_plies)):
        if b.is_game_over():
            break

        # Network plays white using MCTS (PUCT) for policy improvement
        if b.turn != chess.WHITE:
            # If an opening/random start puts black to move, let Stockfish move without recording.
            res0 = stockfish.search(b.fen())
            mv0 = chess.Move.from_uci(res0.bestmove_uci)
            if mv0 not in b.legal_moves:
                mv0 = next(iter(b.legal_moves))
            b.push(mv0)
            continue

        x = encode_position(b, add_features=True)
        probs, a, _v = run_mcts(
            model,
            b,
            device=device,
            rng=rng,
            cfg=MCTSConfig(simulations=int(mcts_simulations), temperature=float(temperature)),
        )
        mv_net = index_to_move(a, b)
        b.push(mv_net)

        s = ReplaySample(x=x, policy_target=probs, wdl_target=1, is_network_turn=True)

        # If game ended after network move, there is no SF reply target.
        if b.is_game_over():
            samples.append(s)
            break

        # Stockfish reply (black)
        res = stockfish.search(b.fen())
        mv_sf = chess.Move.from_uci(res.bestmove_uci)
        if mv_sf not in b.legal_moves:
            mv_sf = next(iter(b.legal_moves))

        a_sf = int(move_to_index(mv_sf, b))
        onehot = np.zeros((POLICY_SIZE,), dtype=np.float32)
        onehot[a_sf] = 1.0
        s.sf_move_index = a_sf
        s.sf_policy_target = onehot
        if res.wdl is not None:
            wdl = np.asarray(res.wdl, dtype=np.float32)
            if wdl.shape == (3,):
                s.sf_wdl = np.array([float(wdl[2]), float(wdl[1]), float(wdl[0])], dtype=np.float32)

        b.push(mv_sf)
        samples.append(s)

    result = b.result(claim_draw=True)
    for s in samples:
        s.wdl_target = int(_result_to_wdl(result, pov_white=True))

    return samples, GameStats(result=result, plies=len(samples) * 2)
