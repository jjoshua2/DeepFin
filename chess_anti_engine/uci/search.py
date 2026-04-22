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
from chess_anti_engine.mcts.puct import _value_scalar_from_wdl_logits
from chess_anti_engine.moves import index_to_move, move_to_index
from chess_anti_engine.tablebase import SyzygyProbe, try_tb_root_move

from .score import q_to_cp
from .time_manager import Deadline
from .walker_pool import WalkerPool, WalkerPoolConfig


# Saturated cp for TB-decisive positions. Matches what the NN-backed path
# naturally emits when Q is pinned to ±1 by the SyzygyProbe's wdl override,
# so the two code paths report consistently.
_TB_WIN_CP = 41890
_TB_LOSS_CP = -41890


# Keep chunks small enough that a ``stop`` arriving mid-search is answered
# within ~50 ms on a warm GPU, but large enough that per-call overhead is
# amortized. 32 sims/chunk works well as a starting point.
_DEFAULT_CHUNK_SIMS = 32

# Info-line emission cadence. 2 Hz keeps GUI output readable without flooding
# the log, and PV extraction (a handful of tree walks) runs only at this rate.
_INFO_EMIT_INTERVAL_MS = 500


@dataclass
class SearchResult:
    bestmove_uci: str
    ponder_uci: str | None
    nodes: int
    pv: tuple[str, ...]
    score_cp: int
    tbhits: int = 0
    # When set, the PV terminates in checkmate; emit `score mate N` instead
    # of `score cp`. Sign: positive = root STM mates, negative = gets mated.
    # Units: UCI moves (ceil(plies/2) with sign).
    score_mate: int | None = None


class InfoCallback(Protocol):
    def __call__(
        self, *,
        nodes: int, elapsed_ms: int, score_cp: int, pv: tuple[str, ...],
        tbhits: int, score_mate: int | None,
    ) -> None:
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
        n_walkers: int = 1,
        vloss_weight: int = 3,
    ) -> None:
        self._evaluator = evaluator
        self._device = device
        self._cfg = gumbel_cfg or GumbelConfig(
            simulations=chunk_sims,
            add_noise=False,  # no exploration noise at test time
        )
        self._chunk_sims = int(chunk_sims)
        self._rng = np.random.default_rng()
        # n_walkers > 1 → PUCT pool with vloss, sequential halving dropped.
        # evaluator MUST be thread-safe (caller wraps with Thread/MultiGPU
        # dispatcher or BatchCoalescingDispatcher).
        self._vloss_weight = int(vloss_weight)
        self._n_walkers = max(1, int(n_walkers))
        self._walker_pool: WalkerPool | None = self._build_walker_pool(self._n_walkers)
        self._walker_cboard: CBoard | None = None

        # Persistent tree across calls within a game. Reset on new position.
        self._tree: MCTSTree | None = None
        self._root_id: int | None = None
        self._tree_fen: str | None = None
        # Cache of the root's policy + WDL logits. Valid for as long as the
        # tree is valid (same position). Lets chunks after the first skip
        # the ~1ms root GPU call.
        self._root_pol_logits: np.ndarray | None = None
        self._root_wdl_logits: np.ndarray | None = None
        # Optional Syzygy probe. When set, MCTS leaves in the TB range get
        # their NN wdl overridden with the TB-truth distribution.
        self._tb_probe = None
        # Soft memory cap: search halts between chunks if tree size exceeds
        # this. 0 / None = unbounded. Not a hash table — tree growth is all
        # or nothing, we stop adding rather than evicting.
        self._max_tree_bytes: int = 0

    def _build_walker_pool(self, n: int) -> WalkerPool | None:
        if n <= 1:
            return None
        return WalkerPool(
            WalkerPoolConfig(
                n_walkers=n,
                c_puct=float(self._cfg.c_puct),
                fpu_at_root=0.0,
                fpu_reduction=float(self._cfg.fpu_reduction),
                vloss_weight=self._vloss_weight,
            ),
            self._evaluator,
        )

    def set_num_threads(self, n: int) -> None:
        """Rebuild the walker pool at thread count ``n`` (1 = classic Gumbel
        path, no pool). Drops the persistent tree because walker-pool Q/N
        stats accumulate with vloss adjustments that depend on thread count.
        Caller should hold the search barrier — same pattern as
        ``set_tb_probe``.
        """
        n = max(1, int(n))
        if n == self._n_walkers:
            return
        self._n_walkers = n
        self._walker_pool = self._build_walker_pool(n)
        self.reset_tree()

    def reset_tree(self) -> None:
        self._tree = None
        self._root_id = None
        self._tree_fen = None
        self._invalidate_root_caches()

    def _invalidate_root_caches(self) -> None:
        self._root_pol_logits = None
        self._root_wdl_logits = None
        self._walker_cboard = None

    def set_max_tree_mb(self, mb: int) -> None:
        """Soft cap on tree memory; 0 disables. Checked between chunks."""
        self._max_tree_bytes = max(0, int(mb)) * 1024 * 1024

    def set_tb_probe(self, probe) -> None:
        """Install (or replace, or clear with None) the Syzygy probe used for
        leaf-batch WDL overrides.

        Changing the probe invalidates the persistent MCTS tree wholesale —
        Q/N stats along every path were computed under the old evaluation
        source (NN-only vs TB-corrected), so reusing the tree would mix
        regimes and could back-propagate stale values. Simpler and correct
        to reset and let the next search rebuild from scratch."""
        self._tb_probe = probe
        self.reset_tree()

    def advance_root(self, board: chess.Board, moves: list[chess.Move]) -> bool:
        """Descend the current tree by ``moves`` plies, making the last-reached
        node the new root. ``board`` is the position BEFORE the first move;
        we push each move onto a local copy to compute its policy index.

        Returns True if the whole walk succeeded (tree reusable), False if any
        step fell off the expanded tree (caller must call ``reset_tree``).
        """
        if self._tree is None or self._root_id is None or self._root_id < 0:
            return False
        b = board.copy(stack=False)
        rid = self._root_id
        for mv in moves:
            idx = move_to_index(mv, b)
            rid = self._tree.find_child(rid, int(idx))
            if rid < 0:
                return False
            b.push(mv)
        self._root_id = rid
        self._tree_fen = b.fen()
        self._invalidate_root_caches()
        return True

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
        fen = board.fen()
        if self._tree is None or self._tree_fen != fen:
            self._tree = None
            self._root_id = None
            self._tree_fen = fen
            self._invalidate_root_caches()

        total_nodes = 0
        last_info_ms = -1
        last_value = 0.0
        pv_indices: list[int] = []
        tb_probe = self._tb_probe
        if tb_probe is not None:
            tb_probe.reset_counts()

        # TB root shortcut: if the root position is TB-eligible, the tables
        # know the move exactly (DTZ-optimal for decisive; any for drawn).
        # MCTS adds nothing and, worse, picks by visits — which in a TB-win
        # with Q=1.0 everywhere reduces to "most-popular NN prior", yielding
        # a valid but DTZ-sub-optimal sequence. Skip straight to the answer.
        if tb_probe is not None:
            short = _try_tb_root_bestmove(board, tb_probe)
            if short is not None:
                if info_cb is not None:
                    info_cb(
                        nodes=short.nodes,
                        elapsed_ms=deadline.elapsed_ms(),
                        score_cp=short.score_cp,
                        pv=short.pv,
                        tbhits=short.tbhits,
                        score_mate=short.score_mate,
                    )
                return short

        # Root eval is the same every chunk (same position, same net). Do it
        # once here and pass pre_pol_logits/pre_wdl_logits into each chunk so
        # the C path skips its own root GPU call. Saves ~1ms × (chunks-1) per
        # search and lets us hand-share the encoding across chunks for free.
        if self._root_pol_logits is None or self._root_wdl_logits is None:
            xs = np.empty((1, 146, 8, 8), dtype=np.float32)
            root_cb = CBoard.from_board(board)
            xs[0] = root_cb.encode_146()
            pol, wdl = self._evaluator.evaluate_encoded(xs)
            pol_np = np.asarray(pol, dtype=np.float32)
            wdl_np = np.asarray(wdl, dtype=np.float32).copy()
            # Probe at root so score_cp reflects TB truth on the very first
            # chunk's info emission, before MCTS has back-propagated it.
            if tb_probe is not None:
                tb_probe.apply([root_cb], wdl_np)
            self._root_pol_logits = pol_np
            self._root_wdl_logits = wdl_np

        # Walker-pool path pre-expands the root from the cached logits so
        # the concurrent walkers don't race on it. The classic path does
        # this internally in run_gumbel_root_many_c.
        if self._walker_pool is not None:
            self._ensure_walker_root_expanded(board)

        while True:
            chunk = self._chunk_sims
            if max_nodes is not None:
                remaining = max_nodes - total_nodes
                if remaining <= 0:
                    break
                chunk = min(chunk, remaining)

            if self._walker_pool is not None:
                last_value = self._run_walker_chunk(chunk, stop_event)
            else:
                last_value = self._run_gumbel_chunk(chunk, board, tb_probe)
            total_nodes += int(chunk)

            # PV extraction is only needed for info emission (rate-limited)
            # and for max_depth termination. Skip otherwise — saves a handful
            # of tree walks per second on chunk=512 at ~5 nps/chunk.
            elapsed = deadline.elapsed_ms() if info_cb is not None else 0
            need_pv = (
                (info_cb is not None and elapsed - last_info_ms >= _INFO_EMIT_INTERVAL_MS)
                or max_depth is not None
            )
            if need_pv:
                assert self._tree is not None and self._root_id is not None
                _, pv_indices = _best_move_and_pv(self._tree, self._root_id)
                if info_cb is not None and elapsed - last_info_ms >= _INFO_EMIT_INTERVAL_MS:
                    info_cb(
                        nodes=total_nodes,
                        elapsed_ms=elapsed,
                        score_cp=q_to_cp(0.5 * (last_value + 1.0)),
                        pv=_uci_pv(board, pv_indices),
                        tbhits=tb_probe.hits if tb_probe is not None else 0,
                        score_mate=_pv_mate_moves(board, pv_indices),
                    )
                    last_info_ms = elapsed

            if stop_event.is_set() or deadline.expired():
                break
            if max_nodes is not None and total_nodes >= max_nodes:
                break
            if max_depth is not None and len(pv_indices) >= max_depth:
                break
            if (self._max_tree_bytes > 0
                    and self._tree is not None
                    and self._tree.memory_bytes() >= self._max_tree_bytes):
                # Don't try to grow into swap; halt with the best move we have.
                # Info-string so GUIs/logs surface the reason.
                if info_cb is not None:
                    info_cb(
                        nodes=total_nodes,
                        elapsed_ms=elapsed,
                        score_cp=q_to_cp(0.5 * (last_value + 1.0)),
                        pv=_uci_pv(board, pv_indices),
                        tbhits=tb_probe.hits if tb_probe is not None else 0,
                        score_mate=_pv_mate_moves(board, pv_indices),
                    )
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
            tbhits=tb_probe.hits if tb_probe is not None else 0,
            score_mate=_pv_mate_moves(board, pv_indices),
        )

    def _run_walker_chunk(
        self, chunk: int, stop_event: threading.Event,
    ) -> float:
        assert self._tree is not None and self._root_id is not None
        assert self._walker_pool is not None and self._walker_cboard is not None
        self._walker_pool.run(
            tree=self._tree,
            root_id=self._root_id,
            root_cboard=self._walker_cboard,
            target_sims=chunk,
            stop_event=stop_event,
        )
        return self._tree.node_q(self._root_id)

    def _run_gumbel_chunk(
        self, chunk: int, board: chess.Board, tb_probe,
    ) -> float:
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
            tb_probe=tb_probe,
        )
        self._tree = tree
        self._root_id = int(root_ids[0])
        return float(values[0])

    def _ensure_walker_root_expanded(self, board: chess.Board) -> None:
        """Walker path needs the root pre-expanded before workers start;
        otherwise all N walkers hit the same unexpanded leaf and waste
        N-1 NN evals on the first sim."""
        assert self._root_pol_logits is not None
        if self._tree is None:
            self._tree = MCTSTree()
            # Pre-size so concurrent descents can't trigger a realloc.
            self._tree.reserve(50_000, 500_000)
            self._root_id = None
        if self._root_id is None:
            assert self._root_wdl_logits is not None
            root_q = float(_value_scalar_from_wdl_logits(
                self._root_wdl_logits[0]))
            self._root_id = int(self._tree.add_root(0, root_q))
        if self._walker_cboard is None:
            self._walker_cboard = CBoard.from_board(board)
        if not self._tree.is_expanded(self._root_id):
            legal_idx = self._walker_cboard.legal_move_indices()
            if legal_idx.size > 0:
                self._tree.expand_from_logits(
                    self._root_id,
                    legal_idx.astype(np.int32),
                    self._root_pol_logits,
                )


# --- tree + move helpers -----------------------------------------------------


def _try_tb_root_bestmove(
    board: chess.Board, tb_probe: SyzygyProbe,
) -> SearchResult | None:
    """Return a SearchResult built from the TB's DTZ-optimal move at root,
    or None if the position isn't TB-eligible (or the probe fails)."""
    root = try_tb_root_move(board, tb_probe._path)  # pyright: ignore[reportPrivateUsage]
    if root is None:
        return None
    best, wdl_val = root

    # Count the root probe toward tbhits so downstream info emission shows
    # a non-zero hit count (MCTS path isn't run in the shortcut).
    tb_probe.probes += 1
    tb_probe.hits += 1

    if wdl_val >= 2:
        score_cp = _TB_WIN_CP
    elif wdl_val <= -2:
        score_cp = _TB_LOSS_CP
    else:
        score_cp = 0  # draw (includes cursed/blessed in our convention)

    # Ponder move: after our best, what's the opponent's DTZ-optimal reply?
    # Re-runs try_tb_root_move; still cheap (a few legal-move probes).
    board_after = board.copy(stack=False)
    board_after.push(best)
    ponder = try_tb_root_move(board_after, tb_probe._path)  # pyright: ignore[reportPrivateUsage]
    ponder_uci = ponder[0].uci() if ponder is not None else None

    return SearchResult(
        bestmove_uci=best.uci(),
        ponder_uci=ponder_uci,
        nodes=1,
        pv=(best.uci(),),
        score_cp=score_cp,
        tbhits=1,
        score_mate=None,
    )


def _pv_mate_moves(root_board: chess.Board, pv_indices: list[int]) -> int | None:
    """If the PV terminates in checkmate, return signed mate-in-N *moves*
    (UCI convention); otherwise None. Positive = root STM mates, negative
    = root STM gets mated.

    Walks the PV onto a local copy of the board — O(len(pv)). Only emits
    mate when the PV actually ends in a checkmate, so the number is real,
    not a Q-saturation artifact (Syzygy wins still go out as high cp)."""
    if not pv_indices:
        return None
    b = root_board.copy(stack=False)
    for idx in pv_indices:
        try:
            mv = index_to_move(int(idx), b)
        except Exception:
            return None
        if mv not in b.legal_moves:
            return None
        b.push(mv)
    if not b.is_checkmate():
        return None
    # After pushing all PV moves, b.turn is the side that has no legal moves
    # (the mated side). Mating side = opposite. Convert plies → UCI moves
    # with ceil(plies/2) so odd plies (STM delivers mate) round up correctly.
    plies = len(pv_indices)
    mating_side = not b.turn
    moves = (plies + 1) // 2
    return moves if mating_side == root_board.turn else -moves


def _best_move_and_pv(tree: MCTSTree, root_id: int) -> tuple[int, list[int]]:
    actions, visits = tree.get_children_visits(root_id)
    if actions.size == 0:
        return -1, []
    best = int(actions[int(np.argmax(visits))])
    pv = [best]
    current_id = tree.find_child(root_id, best)
    # Tree walks from root are acyclic by construction of MCTSTree, and the
    # first unexpanded node naturally terminates the descent. No depth cap.
    while current_id != -1:
        a, vs = tree.get_children_visits(current_id)
        if a.size == 0:
            break
        nxt = int(a[int(np.argmax(vs))])
        pv.append(nxt)
        current_id = tree.find_child(current_id, nxt)
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


