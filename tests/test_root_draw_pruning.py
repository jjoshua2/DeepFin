"""Winning-root draw pruning must never cost a draw the search can see.

A root the network thinks is winning (root_q > 0) leaves out the moves that
draw at once. Two shapes used to lose a drawn position:

* pruning left ONE move, which the runner returned as forced with zero
  simulations (``4K2k/3q1P2/6Q1/8/8/8/8/8 w``: Kf8 is mated by Qd8, Kxd7
  stalemates) -- now the draws stay in the search;
* pruning left several moves and every one of them searched below 0
  (``8/q7/1P6/8/8/3Q4/8/K1k5 w``: everything but bxa7 is mated next move) --
  now the pruned on-board draw is played.

Both run on the shared search (self-play, arena and UCI classic), the C runner
and the Python reference alike.
"""
from __future__ import annotations

from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig, run_gumbel_root_many
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.mcts.root_tactics import forced_draw_actions, forced_draw_actions_cboard
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask, move_to_index

KXD7_FEN = "4K2k/3q1P2/6Q1/8/8/8/8/8 w - - 0 1"        # Kxd7 stalemates, Kf8 Qd8#
BXA7_FEN = "8/q7/1P6/8/8/3Q4/8/K1k5 w - - 0 1"          # bxa7 stalemates, all else mated
SURVIVOR_WINS_FEN = "8/8/8/1Q1N4/2K5/k1q5/8/8 w - - 0 1"  # Nxc3 stalemates, Kxc3 mates in 2
# Mated in one after every move; some moves only let the OPPONENT claim a
# 50-move draw on its reply (clock 99 + a reversible reply). Not a forced draw.
CLAIMABLE_ONLY_FEN = "8/8/8/3P4/6q1/P4k2/7K/8 w - - 98 1"

_WINNING_ROOT = np.array([[8.0, 0.0, -8.0]], dtype=np.float32)


def _mirror(fen: str, *ucis: str) -> tuple[str, list[str]]:
    """The colour-flipped position (the other side to move) and its moves."""
    board = chess.Board(fen)
    mirrored = board.mirror()
    out = []
    for uci in ucis:
        move = chess.Move.from_uci(uci)
        out.append(chess.Move(
            chess.square_mirror(move.from_square), chess.square_mirror(move.to_square),
            move.promotion,
        ).uci())
    return mirrored.fen(), out


def _variants(fen: str, *ucis: str) -> list[Any]:
    black_fen, black_ucis = _mirror(fen, *ucis)
    return [
        pytest.param(fen, list(ucis), id="white"),
        pytest.param(black_fen, black_ucis, id="black"),
    ]


class _NeutralLeaves:
    """Uniform policy, value 0 at every leaf."""

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        n = int(x.shape[0])
        return np.zeros((n, POLICY_SIZE), np.float32), np.zeros((n, 3), np.float32)


class _SideToMoveWins:
    """Every leaf looks won for the side to move, so every root move searches
    near -1: the shape of a root whose winning look the search refutes."""

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        n = int(x.shape[0])
        return (
            np.zeros((n, POLICY_SIZE), np.float32),
            np.broadcast_to(_WINNING_ROOT, (n, 3)).copy(),
        )


def _idx(board: chess.Board, uci: str) -> int:
    return int(move_to_index(chess.Move.from_uci(uci), board))


def _run_c(
    board: chess.Board, evaluator: Any, *, sims: int = 64, topk: int = 16,
    use_cboard: bool = False, allowed: set[int] | None = None,
    pre_pol: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float, np.ndarray, MCTSTree, int]:
    cb = cboard_from_board_fast(board)
    probs, actions, values, masks, tree, roots = run_gumbel_root_many_c(
        None, cast(Any, [cb]) if use_cboard else [board], device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=sims, topk=topk,
                         temperature=0.0, add_noise=False),
        evaluator=evaluator,
        pre_pol_logits=(
            pre_pol if pre_pol is not None else np.zeros((1, POLICY_SIZE), np.float32)
        ),
        pre_wdl_logits=_WINNING_ROOT,
        cboards=[cb] if use_cboard else None, target_batch=1,
        allowed_root_indices_batch=[allowed] if allowed is not None else None,
    )[:6]
    return probs[0], int(actions[0]), float(values[0]), masks[0], tree, int(roots[0])


def _assert_legal_distribution(probs: np.ndarray, board: chess.Board) -> None:
    assert probs.shape == (POLICY_SIZE,)
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert not np.any(probs[~legal_move_mask(board)]), "policy mass on an illegal move"


# --- the forced-draw helpers -------------------------------------------------

@pytest.mark.parametrize(("fen", "draw"), [(KXD7_FEN, "e8d7"), (BXA7_FEN, "b6a7")])
def test_forced_draw_helpers_agree_on_stalemates(fen: str, draw: str) -> None:
    board = chess.Board(fen)
    legal = [int(move_to_index(m, board)) for m in board.legal_moves]
    assert forced_draw_actions(board, legal) == [_idx(board, draw)]
    assert forced_draw_actions_cboard(cboard_from_board_fast(board), legal) == [_idx(board, draw)]
    assert board.fen() == fen, "the helper must restore the board"


def test_forced_draw_helpers_exclude_reply_claims() -> None:
    board = chess.Board(CLAIMABLE_ONLY_FEN)
    legal = np.array([int(move_to_index(m, board)) for m in board.legal_moves], np.int32)
    _mate, claims = cboard_from_board_fast(board).root_terminal_actions(legal, True)
    assert claims.size > 0, "fixture: the draw scan reports reply-claim draws"
    assert forced_draw_actions(board, claims.tolist()) == []
    assert forced_draw_actions_cboard(cboard_from_board_fast(board), claims.tolist()) == []


# --- singleton: pruning no longer fakes a forced move ------------------------

@pytest.mark.parametrize("use_cboard", [False, True], ids=["board", "cboard"])
@pytest.mark.parametrize(("fen", "moves"), _variants(KXD7_FEN, "e8d7", "e8f8"))
def test_c_singleton_keeps_the_draw_and_plays_it(
    fen: str, moves: list[str], use_cboard: bool,
) -> None:
    board = chess.Board(fen)
    draw, loss = (_idx(board, u) for u in moves)
    probs, action, value, mask, tree, root = _run_c(
        board, _NeutralLeaves(), topk=2, use_cboard=use_cboard,
    )
    children, visits = tree.get_children_visits(root)
    assert set(children.tolist()) == {draw, loss}
    assert int(visits.sum()) > 0
    assert tree.get_solved_status(root) == 2, "Kf8 Qd8# is found, the root is a proven draw"
    assert action == draw
    assert value == 0.0
    assert probs[draw] == 1.0
    assert np.array_equal(mask, legal_move_mask(board))


@pytest.mark.parametrize(("fen", "moves"), _variants(KXD7_FEN, "e8d7", "e8f8"))
def test_c_singleton_stored_target_is_a_legal_distribution_over_both_moves(
    fen: str, moves: list[str],
) -> None:
    """Two simulations: each move once, nothing proven -- the stored target is
    the ordinary completed-Q policy over the two legal moves."""
    board = chess.Board(fen)
    draw, loss = (_idx(board, u) for u in moves)
    probs, _action, _value, _mask, tree, root = _run_c(board, _NeutralLeaves(), sims=2, topk=2)
    assert tree.get_solved_status(root) == 0
    _assert_legal_distribution(probs, board)
    assert probs[draw] > 0.0
    assert probs[loss] > 0.0


@pytest.mark.parametrize(("fen", "moves"), _variants(SURVIVOR_WINS_FEN, "c4c3", "d5c3"))
def test_c_singleton_survivor_that_wins_is_still_played(fen: str, moves: list[str]) -> None:
    board = chess.Board(fen)
    win, draw = (_idx(board, u) for u in moves)
    assert forced_draw_actions(board, [win, draw]) == [draw]
    probs, action, value, _mask, tree, root = _run_c(board, _NeutralLeaves(), topk=2)
    assert set(tree.get_children_visits(root)[0].tolist()) == {win, draw}
    assert action == win
    assert value > 0.0
    assert probs[win] > probs[draw]


def test_c_explicit_singleton_searchmove_is_still_returned_unsearched() -> None:
    board = chess.Board(KXD7_FEN)
    loss = _idx(board, "e8f8")
    probs, action, _value, _mask, tree, root = _run_c(
        board, cast(Any, object()), topk=2, allowed={loss},
    )
    assert action == loss
    assert probs[loss] == 1.0
    assert int(tree.get_children_visits(root)[1].sum()) == 0


# --- several survivors, all below 0: play the pruned draw ---------------------

@pytest.mark.parametrize("use_cboard", [False, True], ids=["board", "cboard"])
@pytest.mark.parametrize(("fen", "moves"), _variants(BXA7_FEN, "b6a7"))
def test_c_refuted_winning_root_plays_the_pruned_draw(
    fen: str, moves: list[str], use_cboard: bool,
) -> None:
    board = chess.Board(fen)
    draw = _idx(board, moves[0])
    probs, action, value, mask, tree, root = _run_c(
        board, _SideToMoveWins(), use_cboard=use_cboard,
    )
    children, visits = tree.get_children_visits(root)
    assert draw not in children.tolist(), "the draw is still pruned from the search"
    assert int(visits.sum()) > 0
    assert action == draw
    assert value == 0.0
    assert probs[draw] == 1.0
    assert np.count_nonzero(probs) == 1
    assert np.array_equal(mask, legal_move_mask(board))


def test_c_refuted_root_does_not_play_a_reply_claim() -> None:
    """Only a move that ENDS the game drawn is a fallback; one after which the
    opponent may claim is the opponent's option."""
    board = chess.Board(CLAIMABLE_ONLY_FEN)
    legal = np.array([int(move_to_index(m, board)) for m in board.legal_moves], np.int32)
    _mate, claims = cboard_from_board_fast(board).root_terminal_actions(legal, True)
    _probs, action, value, _mask, tree, root = _run_c(board, _SideToMoveWins())
    searched = set(tree.get_children_visits(root)[0].tolist())
    assert not searched & set(claims.tolist()), "fixture: the claims are pruned"
    assert action in searched
    assert value < 0.0


def test_c_fallback_respects_searchmoves() -> None:
    board = chess.Board(BXA7_FEN)
    draw = _idx(board, "b6a7")
    others = {int(move_to_index(m, board)) for m in board.legal_moves} - {draw}
    _probs, action, _value, _mask, _tree, _root = _run_c(
        board, _SideToMoveWins(), allowed=others,
    )
    assert action != draw, "an excluded move is never played"
    # Allowing the draw and ONE other move: pruning would leave a singleton,
    # so both are searched and the draw wins on its exact value.
    one = min(others)
    _probs, action, _value, _mask, tree, root = _run_c(
        board, _SideToMoveWins(), allowed={draw, one},
    )
    assert set(tree.get_children_visits(root)[0].tolist()) == {draw, one}
    assert action == draw


def test_c_fallback_needs_every_searched_move_below_zero() -> None:
    """A survivor at exactly 0 is not refuted: the pruned stalemate stays out."""
    board = chess.Board("7k/8/6KP/8/8/8/8/8 w - - 0 1")
    stalemate = _idx(board, "h6h7")
    pre_pol = np.full((1, POLICY_SIZE), -100.0, np.float32)
    pre_pol[0, stalemate] = 100.0
    probs, action, _value, _mask, _tree, _root = _run_c(
        board, _NeutralLeaves(), sims=1, topk=1, pre_pol=pre_pol,
    )
    assert action != stalemate
    assert probs[stalemate] == 0.0


# --- Python reference (self-play's volatility path) --------------------------

def _run_py(board: chess.Board, evaluator: Any, *, topk: int) -> tuple[np.ndarray, int, float]:
    probs, actions, values, _masks = run_gumbel_root_many(
        None, [board], device="cpu", rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=64, topk=topk,
                         temperature=0.0, add_noise=False),
        evaluator=evaluator,
        pre_pol_logits=np.zeros((1, POLICY_SIZE), np.float32),
        pre_wdl_logits=_WINNING_ROOT,
    )
    return probs[0], int(actions[0]), float(values[0])


@pytest.mark.parametrize(("fen", "moves"), _variants(KXD7_FEN, "e8d7", "e8f8"))
def test_python_singleton_searches_both_moves(fen: str, moves: list[str]) -> None:
    board = chess.Board(fen)
    draw, loss = (_idx(board, u) for u in moves)
    probs, action, _value = _run_py(board, _NeutralLeaves(), topk=2)
    _assert_legal_distribution(probs, board)
    assert probs[draw] > 0.0, "both moves were searched"
    assert probs[loss] > 0.0, "both moves were searched"
    assert action == draw


@pytest.mark.parametrize(("fen", "moves"), _variants(BXA7_FEN, "b6a7"))
def test_python_refuted_winning_root_plays_the_pruned_draw(fen: str, moves: list[str]) -> None:
    board = chess.Board(fen)
    draw = _idx(board, moves[0])
    probs, action, value = _run_py(board, _SideToMoveWins(), topk=16)
    assert action == draw
    assert value == 0.0
    assert probs[draw] == 1.0
    assert np.count_nonzero(probs) == 1
