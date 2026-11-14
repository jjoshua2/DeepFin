"""Python search-tree primitives shared by the Python Gumbel reference search.

``Node``, ``_expand_sparse``, ``_backprop`` and ``_terminal_value`` are the tree
``mcts/gumbel.py``'s ``run_gumbel_root_many`` (the parity oracle and the
volatility-arm search) is built on; ``_value_scalar_from_wdl_logits`` is also
read by the UCI engine. The pure-Python PUCT search that used to live here
(``run_mcts_many``/``run_mcts``/``MCTSConfig``) and its compiled driver
(``mcts/puct_c.py``) were retired on 2026-11-14: selfplay, arena and puzzle
eval have one search, compiled Gumbel. The UCI PUCT engines (walker pool,
pucv, multi-GPU pool, root-parallel Gumbel) drive ``_mcts_tree`` directly and
never used either.
"""
from __future__ import annotations

import math

import chess
import numpy as np

from chess_anti_engine.moves.encode import index_to_move_fast


def _value_scalar_from_wdl_logits(wdl_logits: np.ndarray) -> float:
  # Convert (3,) logits into a scalar v in [-1,1] from side-to-move perspective.
  # Pure Python math is faster than numpy for 3-element arrays.
    w, d, l = float(wdl_logits[0]), float(wdl_logits[1]), float(wdl_logits[2])
    mx = max(w, d, l)
    ew = math.exp(w - mx)
    ed = math.exp(d - mx)
    el = math.exp(l - mx)
    s = ew + ed + el
    return (ew - el) / s if s > 0 else 0.0


class Node:
    __slots__ = ("N", "W", "_action_idx", "_board", "_move", "children", "expanded", "parent", "prior", "to_play", "vol")

    _board: chess.Board | None
    _move: chess.Move | None
    _action_idx: int | None
    parent: Node | None
    prior: float
    N: int
    W: float
    children: dict[int, Node]
    expanded: bool
    to_play: chess.Color
  # Predicted position volatility at this node's NN eval (scalar summary of
  # the volatility head). 0.0 unless volatility-aware Gumbel search is on.
    vol: float

    def __init__(
        self,
        board: chess.Board | None,
        *,
        parent: Node | None,
        prior: float,
        move: chess.Move | None = None,
        action_idx: int | None = None,
    ):
        self._board = board
        self._move = move
        self._action_idx = action_idx
        self.parent = parent
        self.prior = float(prior)
        self.vol = 0.0
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.expanded = False
        if board is not None:
            self.to_play = board.turn
        elif parent is not None:
            self.to_play = not parent.to_play
        else:
            self.to_play = chess.WHITE

    @property
    def board(self) -> chess.Board:
        if self._board is None:
            assert self.parent is not None
            if self._move is None:
                assert self._action_idx is not None
                self._move = index_to_move_fast(self._action_idx, self.parent.board)
  # Preserve full history so search-time encodings keep LC0 history
  # and repetition planes consistent with python-chess semantics.
            self._board = self.parent.board.copy(stack=True)
            self._board.push(self._move)
        return self._board

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


def _expand_sparse(node: Node, legal_indices: np.ndarray, priors: np.ndarray) -> None:
    """Expand using pre-computed legal indices and their priors (avoids 4672-bool mask)."""
    node.expanded = True
    for i in range(len(legal_indices)):
        a = int(legal_indices[i])
        node.children[a] = Node(None, parent=node, prior=float(priors[i]), action_idx=a)


def _backprop(path: list[Node], value: float) -> None:
  # value is from perspective of the leaf's side-to-move.
    v = float(value)
    for n in reversed(path):
        n.N += 1
        n.W += v
        v = -v  # switch perspective each ply


def _terminal_value(board: chess.Board) -> float:
    res = board.result(claim_draw=True)
    if res == "1/2-1/2":
        return 0.0
    if res == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    return 1.0 if board.turn == chess.BLACK else -1.0
