from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import chess
import torch

from chess_anti_engine.mcts import MCTSConfig, GumbelConfig
from chess_anti_engine.mcts.puct import run_mcts_many
from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
from chess_anti_engine.selfplay.opening import OpeningConfig, make_starting_board


@dataclass(frozen=True)
class MatchStats:
    games: int
    max_plies: int

    # From model_a perspective
    a_win: int
    a_draw: int
    a_loss: int

    a_as_white: int
    a_as_black: int


def play_match_batch(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    *,
    device: str,
    rng: np.random.Generator,
    games: int,
    max_plies: int,
    a_plays_white: list[bool] | None = None,
    mcts_type: str = "puct",
    mcts_simulations: int = 200,
    temperature: float = 0.1,
    c_puct: float = 2.5,
    opening_cfg: OpeningConfig | None = None,
) -> MatchStats:
    """Play model-vs-model matches.

    `a_plays_white[i]` controls which side model_a plays in game i.

    `opening_cfg` controls opening diversification — pass an OpeningConfig with
    a book path or random_start_plies so games don't all start from the same position.
    Defaults to 2 random start plies if not provided.
    """

    g = int(games)
    if g <= 0:
        raise ValueError("games must be > 0")

    if a_plays_white is None:
        a_plays_white = [True] * g
    if len(a_plays_white) != g:
        raise ValueError("a_plays_white length must match games")

    if opening_cfg is None:
        opening_cfg = OpeningConfig(random_start_plies=2)
    boards = [make_starting_board(rng=rng, cfg=opening_cfg) for _ in range(g)]
    done = [False] * g

    # Main play loop
    for _ply in range(int(max_plies)):
        active = [i for i in range(g) if not done[i]]
        if not active:
            break

        for i in active:
            if boards[i].is_game_over(claim_draw=True):
                done[i] = True

        active = [i for i in range(g) if not done[i]]
        if not active:
            break

        # Split by which model is to move (depends on side assignment + board.turn).
        a_to_move: list[int] = []
        b_to_move: list[int] = []
        for i in active:
            a_is_white = bool(a_plays_white[i])
            a_moves_now = (boards[i].turn == chess.WHITE and a_is_white) or (boards[i].turn == chess.BLACK and not a_is_white)
            (a_to_move if a_moves_now else b_to_move).append(i)

        def _pick_moves(model: torch.nn.Module, idxs: list[int]) -> list[int]:
            if not idxs:
                return []

            sub_boards = [boards[i] for i in idxs]
            if str(mcts_type) == "gumbel":
                probs, actions, _values = run_gumbel_root_many(
                    model,
                    sub_boards,
                    device=device,
                    rng=rng,
                    cfg=GumbelConfig(simulations=int(mcts_simulations), temperature=float(temperature)),
                )
            else:
                probs, actions, _values = run_mcts_many(
                    model,
                    sub_boards,
                    device=device,
                    rng=rng,
                    cfg=MCTSConfig(
                        simulations=int(mcts_simulations),
                        temperature=float(temperature),
                        c_puct=float(c_puct),
                        dirichlet_eps=0.0,
                    ),
                )
            return [int(a) for a in actions]

        a_actions = _pick_moves(model_a, a_to_move)
        b_actions = _pick_moves(model_b, b_to_move)

        # Apply chosen moves.
        from chess_anti_engine.moves import index_to_move

        for i, a in zip(a_to_move, a_actions, strict=True):
            mv = index_to_move(int(a), boards[i])
            if mv not in boards[i].legal_moves:
                mv = next(iter(boards[i].legal_moves))
            boards[i].push(mv)

        for i, a in zip(b_to_move, b_actions, strict=True):
            mv = index_to_move(int(a), boards[i])
            if mv not in boards[i].legal_moves:
                mv = next(iter(boards[i].legal_moves))
            boards[i].push(mv)

    # Score from model_a perspective.
    a_win = a_draw = a_loss = 0
    for i, b in enumerate(boards):
        res = b.result(claim_draw=True)
        if res == "1/2-1/2":
            a_draw += 1
            continue

        a_is_white = bool(a_plays_white[i])
        white_won = res == "1-0"
        a_won = (white_won and a_is_white) or ((not white_won) and (not a_is_white))
        if a_won:
            a_win += 1
        else:
            a_loss += 1

    return MatchStats(
        games=g,
        max_plies=int(max_plies),
        a_win=int(a_win),
        a_draw=int(a_draw),
        a_loss=int(a_loss),
        a_as_white=int(sum(1 for v in a_plays_white if bool(v))),
        a_as_black=int(sum(1 for v in a_plays_white if not bool(v))),
    )
