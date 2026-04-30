from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch

from chess_anti_engine.mcts import GumbelConfig, MCTSConfig
from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
from chess_anti_engine.mcts.puct import run_mcts_many
from chess_anti_engine.moves import index_to_move
from chess_anti_engine.selfplay.opening import OpeningConfig, make_starting_board

try:
    from chess_anti_engine.mcts.puct_c import run_mcts_many_c as _run_mcts_many_c
    _HAS_C_TREE = True
except ImportError:
    _HAS_C_TREE = False

try:
    from chess_anti_engine.mcts.gumbel_c import (
        run_gumbel_root_many_c as _run_gumbel_root_many_c,
    )
    _HAS_GUMBEL_C = True
except ImportError:
    _HAS_GUMBEL_C = False

if TYPE_CHECKING:
    from chess_anti_engine.mcts.gumbel_c import (
        run_gumbel_root_many_c as _run_gumbel_root_many_c,  # noqa: F401,F811
    )
    from chess_anti_engine.mcts.puct_c import (
        run_mcts_many_c as _run_mcts_many_c,  # noqa: F401,F811
    )


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


def _result_from_a_pov(result: str, *, a_is_white: bool) -> int:
    """Map game result to model-a outcome.

    Returns:
        1 for model-a win, 0 for draw, -1 for model-a loss.

    python-chess returns "*" when a game is truncated before reaching a
    terminal result (for example when max_plies is hit). Treat this as draw
    rather than a decisive result.
    """
    if result in {"1/2-1/2", "*"}:
        return 0

    white_won = result == "1-0"
    a_won = (white_won and a_is_white) or ((not white_won) and (not a_is_white))
    return 1 if a_won else -1


def _pick_moves_for_boards(
    model: torch.nn.Module, sub_boards: list[chess.Board], *,
    device: str, rng: np.random.Generator,
    mcts_type: str, mcts_simulations: int, temperature: float, c_puct: float,
) -> list[int]:
    """Run gumbel- or PUCT-MCTS for one model on a list of boards."""
    if str(mcts_type) == "gumbel":
        gumbel_fn = _run_gumbel_root_many_c if _HAS_GUMBEL_C else run_gumbel_root_many
        result = gumbel_fn(
            model, sub_boards, device=device, rng=rng,
            cfg=GumbelConfig(
                simulations=int(mcts_simulations), temperature=float(temperature),
            ),
        )
        _probs, actions, _values, _masks = result[:4]
    else:
        puct_fn = _run_mcts_many_c if _HAS_C_TREE else run_mcts_many
        _probs, actions, _values, _masks = puct_fn(
            model, sub_boards, device=device, rng=rng,
            cfg=MCTSConfig(
                simulations=int(mcts_simulations), temperature=float(temperature),
                c_puct=float(c_puct), dirichlet_eps=0.0,
            ),
        )
    return [int(a) for a in actions]


def _apply_actions_to_boards(
    boards: list[chess.Board], idxs: list[int], actions: list[int],
) -> None:
    """Push each chosen action onto its board (fall back to first legal if illegal)."""
    for i, a in zip(idxs, actions, strict=True):
        mv = index_to_move(int(a), boards[i])
        if mv not in boards[i].legal_moves:
            mv = next(iter(boards[i].legal_moves))
        boards[i].push(mv)


def _split_active_by_side_to_move(
    active: list[int], boards: list[chess.Board], a_plays_white: list[bool],
) -> tuple[list[int], list[int]]:
    """Partition active slots by which model (a/b) is to move."""
    a_to_move: list[int] = []
    b_to_move: list[int] = []
    for i in active:
        a_is_white = bool(a_plays_white[i])
        a_moves_now = (
            (boards[i].turn == chess.WHITE and a_is_white)
            or (boards[i].turn == chess.BLACK and not a_is_white)
        )
        (a_to_move if a_moves_now else b_to_move).append(i)
    return a_to_move, b_to_move


def _tally_match_results(
    boards: list[chess.Board], a_plays_white: list[bool],
) -> tuple[int, int, int]:
    """Count (a_win, a_draw, a_loss) over the finished boards."""
    a_win = a_draw = a_loss = 0
    for i, b in enumerate(boards):
        outcome = _result_from_a_pov(
            b.result(claim_draw=True), a_is_white=bool(a_plays_white[i]),
        )
        if outcome == 0:
            a_draw += 1
        elif outcome > 0:
            a_win += 1
        else:
            a_loss += 1
    return a_win, a_draw, a_loss


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

    def _pick(model: torch.nn.Module, idxs: list[int]) -> list[int]:
        if not idxs:
            return []
        return _pick_moves_for_boards(
            model, [boards[i] for i in idxs],
            device=device, rng=rng,
            mcts_type=mcts_type, mcts_simulations=mcts_simulations,
            temperature=temperature, c_puct=c_puct,
        )

    for _ply in range(int(max_plies)):
        for i in range(g):
            if not done[i] and boards[i].is_game_over(claim_draw=True):
                done[i] = True
        active = [i for i in range(g) if not done[i]]
        if not active:
            break

        a_to_move, b_to_move = _split_active_by_side_to_move(
            active, boards, a_plays_white,
        )
        _apply_actions_to_boards(boards, a_to_move, _pick(model_a, a_to_move))
        _apply_actions_to_boards(boards, b_to_move, _pick(model_b, b_to_move))

    a_win, a_draw, a_loss = _tally_match_results(boards, a_plays_white)

    return MatchStats(
        games=g,
        max_plies=int(max_plies),
        a_win=int(a_win),
        a_draw=int(a_draw),
        a_loss=int(a_loss),
        a_as_white=int(sum(1 for v in a_plays_white if bool(v))),
        a_as_black=int(sum(1 for v in a_plays_white if not bool(v))),
    )
