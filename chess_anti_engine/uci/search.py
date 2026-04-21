"""Chunked MCTS search worker.

Runs ``run_gumbel_root_many_c`` in small sim-chunks so we can check a stop
event between calls. Threads ``tree`` + ``root_node_ids`` across chunks so
each chunk continues the previous tree rather than starting over.

The worker is deliberately oblivious to UCI state (pondering, time);
``Engine`` wraps it with the cooperation protocol. This keeps search pure
and makes the v2 multi-GPU swap a local change.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import chess
import numpy as np

from chess_anti_engine.encoding.cboard_encode import CBoard
from chess_anti_engine.inference import BatchEvaluator
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.moves import index_to_move

from .score import q_to_cp
from .time_manager import Deadline


# Keep chunks small enough that a ``stop`` arriving mid-search is answered
# within ~50 ms on a warm GPU, but large enough that per-call overhead is
# amortized. 32 sims/chunk works well as a starting point.
_DEFAULT_CHUNK_SIMS = 32

_PV_MAX_DEPTH = 12

# Safety ceiling for `go depth`: MCTS has no true depth — we terminate on
# PV length, but a shallow tree could stall forever, so cap total sims too.
_DEPTH_NODE_SAFETY_CAP = 200_000


@dataclass
class SearchResult:
    bestmove_uci: str
    ponder_uci: str | None
    nodes: int
    pv: tuple[str, ...]
    score_cp: int


class InfoCallback(Protocol):
    def __call__(self, *, nodes: int, elapsed_ms: int, score_cp: int, pv: tuple[str, ...]) -> None:
        ...


class SearchWorker:
    """Owns one MCTS tree + one evaluator. Single-search at a time."""

    def __init__(
        self,
        evaluator: BatchEvaluator,
        *,
        device: str,
        gumbel_cfg: GumbelConfig | None = None,
        chunk_sims: int = _DEFAULT_CHUNK_SIMS,
    ) -> None:
        self._evaluator = evaluator
        self._device = device
        self._cfg = gumbel_cfg or GumbelConfig(
            simulations=chunk_sims,
            add_noise=False,  # no exploration noise at test time
        )
        self._chunk_sims = int(chunk_sims)
        self._rng = np.random.default_rng()

        # Persistent tree across calls within a game. Reset on new position.
        self._tree: MCTSTree | None = None
        self._root_id: int | None = None
        self._tree_fen: str | None = None
        # Cache of the root's policy + WDL logits. Valid for as long as the
        # tree is valid (same position). Lets chunks after the first skip
        # the ~1ms root GPU call.
        self._root_pol_logits: np.ndarray | None = None
        self._root_wdl_logits: np.ndarray | None = None

    def reset_tree(self) -> None:
        self._tree = None
        self._root_id = None
        self._tree_fen = None
        self._root_pol_logits = None
        self._root_wdl_logits = None

    def run(
        self,
        board: chess.Board,
        *,
        stop_event: threading.Event,
        deadline: Deadline,
        max_nodes: int | None,
        max_depth: int | None = None,
        info_cb: InfoCallback | None = None,
    ) -> SearchResult:
        """Search until any of: stop_event set, deadline expired, max_nodes hit,
        PV length ≥ max_depth.

        Returns when at least one chunk has run (so bestmove is always
        backed by MCTS data, never a raw priors pick).
        """
        # UCI depth has no clean MCTS analog; we stop when the tree's PV
        # reaches that ply count, with a hard node ceiling so a shallow
        # tree can't stall forever.
        if max_depth is not None and max_nodes is None:
            max_nodes = _DEPTH_NODE_SAFETY_CAP
        fen = board.fen()
        if self._tree is None or self._tree_fen != fen:
            self._tree = None
            self._root_id = None
            self._tree_fen = fen
            self._root_pol_logits = None
            self._root_wdl_logits = None

        total_nodes = 0
        last_info_ms = -1
        last_value = 0.0
        pv_indices: list[int] = []

        # Root eval is the same every chunk (same position, same net). Do it
        # once here and pass pre_pol_logits/pre_wdl_logits into each chunk so
        # the C path skips its own root GPU call. Saves ~1ms × (chunks-1) per
        # search and lets us hand-share the encoding across chunks for free.
        if self._root_pol_logits is None or self._root_wdl_logits is None:
            xs = np.empty((1, 146, 8, 8), dtype=np.float32)
            xs[0] = CBoard.from_board(board).encode_146()
            pol, wdl = self._evaluator.evaluate_encoded(xs)
            self._root_pol_logits = np.asarray(pol, dtype=np.float32)
            self._root_wdl_logits = np.asarray(wdl, dtype=np.float32)

        while True:
            chunk = self._chunk_sims
            if max_nodes is not None:
                remaining = max_nodes - total_nodes
                if remaining <= 0:
                    break
                chunk = min(chunk, remaining)

            _, _, values, _, tree, root_ids = run_gumbel_root_many_c(
                model=None,
                boards=[board],
                device=self._device,
                rng=self._rng,
                cfg=GumbelConfig(
                    simulations=chunk,
                    topk=self._cfg.topk,
                    temperature=self._cfg.temperature,
                    c_visit=self._cfg.c_visit,
                    c_scale=self._cfg.c_scale,
                    c_puct=self._cfg.c_puct,
                    fpu_reduction=self._cfg.fpu_reduction,
                    full_tree=self._cfg.full_tree,
                    add_noise=False,
                ),
                evaluator=self._evaluator,
                pre_pol_logits=self._root_pol_logits,
                pre_wdl_logits=self._root_wdl_logits,
                tree=self._tree,
                root_node_ids=[self._root_id] if self._root_id is not None else None,
            )
            self._tree = tree
            self._root_id = int(root_ids[0])
            last_value = float(values[0])
            total_nodes += int(chunk)

            # PV extraction is only needed for info emission (rate-limited
            # to 5/sec) and for max_depth termination. Skip otherwise —
            # saves a handful of tree walks per second on chunk=512 at ~5 nps/chunk.
            elapsed = deadline.elapsed_ms() if info_cb is not None else 0
            need_pv = (
                (info_cb is not None and elapsed - last_info_ms >= 200)
                or max_depth is not None
            )
            if need_pv:
                _, pv_indices = _best_move_and_pv(tree, self._root_id)
                if info_cb is not None and elapsed - last_info_ms >= 200:
                    info_cb(
                        nodes=total_nodes,
                        elapsed_ms=elapsed,
                        score_cp=q_to_cp(0.5 * (last_value + 1.0)),
                        pv=_uci_pv(board, pv_indices),
                    )
                    last_info_ms = elapsed

            if stop_event.is_set() or deadline.expired():
                break
            if max_nodes is not None and total_nodes >= max_nodes:
                break
            if max_depth is not None and len(pv_indices) >= max_depth:
                break

        # Final snapshot using whatever the tree knows now.
        assert self._tree is not None and self._root_id is not None
        bestmove_idx, pv_indices = _best_move_and_pv(self._tree, self._root_id)
        ponder_idx = _predicted_opponent_reply(self._tree, self._root_id)
        bestmove = _index_to_uci(board, bestmove_idx)
        ponder = (
            _index_to_uci(_board_after(board, bestmove_idx), ponder_idx)
            if ponder_idx is not None else None
        )
        pv = _uci_pv(board, pv_indices)
        return SearchResult(
            bestmove_uci=bestmove,
            ponder_uci=ponder,
            nodes=total_nodes,
            pv=pv,
            score_cp=q_to_cp(0.5 * (last_value + 1.0)),
        )


# --- tree + move helpers -----------------------------------------------------


def _best_move_and_pv(tree: MCTSTree, root_id: int) -> tuple[int, list[int]]:
    actions, visits = tree.get_children_visits(root_id)
    if actions.size == 0:
        return -1, []
    best = int(actions[int(np.argmax(visits))])
    pv = [best]
    current_id = tree.find_child(root_id, best)
    depth = 1
    while current_id != -1 and depth < _PV_MAX_DEPTH:
        a, vs = tree.get_children_visits(current_id)
        if a.size == 0:
            break
        nxt = int(a[int(np.argmax(vs))])
        pv.append(nxt)
        current_id = tree.find_child(current_id, nxt)
        depth += 1
    return best, pv


def _predicted_opponent_reply(tree: MCTSTree, root_id: int) -> int | None:
    """Move index the opponent is predicted to play after OUR bestmove.

    This is what we ponder on, not our own alternative — we take the
    most-visited root child (our bestmove), descend to that node, and
    return ITS most-visited child (opponent's best reply at that node).

    Distinct from the "root's second-most-visited child" which would be
    our 2nd-best move from the current position — a different concept.
    """
    actions, visits = tree.get_children_visits(root_id)
    if actions.size == 0:
        return None
    best = int(actions[int(np.argmax(visits))])
    child_id = tree.find_child(root_id, best)
    if child_id == -1:
        return None
    a, vs = tree.get_children_visits(child_id)
    if a.size == 0:
        return None
    return int(a[int(np.argmax(vs))])


def _uci_pv(root_board: chess.Board, pv_indices: list[int]) -> tuple[str, ...]:
    b = root_board.copy(stack=False)
    out: list[str] = []
    for idx in pv_indices:
        try:
            mv = index_to_move(int(idx), b)
        except Exception:
            break
        if mv not in b.legal_moves:
            break
        out.append(mv.uci())
        b.push(mv)
    return tuple(out)


def _index_to_uci(board: chess.Board, idx: int) -> str:
    if idx < 0:
        # Fallback: any legal move. Should not happen except on game-ended positions.
        legal = list(board.legal_moves)
        return legal[0].uci() if legal else "0000"
    return index_to_move(int(idx), board).uci()


def _board_after(board: chess.Board, idx: int) -> chess.Board:
    b = board.copy(stack=False)
    try:
        b.push(index_to_move(int(idx), board))
    except Exception:
        pass
    return b


