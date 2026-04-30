"""Chunked MCTS search worker.

Runs ``run_gumbel_root_many_c`` in small sim-chunks so we can check a stop
event between calls. Threads ``tree`` + ``root_node_ids`` across chunks so
each chunk continues the previous tree rather than starting over.

The worker is deliberately oblivious to UCI state (pondering, time);
``Engine`` wraps it with the cooperation protocol. This keeps search pure
and makes the v2 multi-GPU swap a local change.
"""
from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import chess
import numpy as np

from chess_anti_engine.encoding.cboard_encode import CBoard
from chess_anti_engine.inference import BatchEvaluator
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.mcts.puct import _value_scalar_from_wdl_logits
from chess_anti_engine.mcts.puct_vl import PucvChunker
from .multi_gpu_pucv_pool import MultiGpuPucvConfig, MultiGpuPucvPool
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

# Info-line emission cadence. 1 Hz matches Lc0/Stockfish real-world behavior
# and avoids flooding terminal GUIs (cutechess-cli, engine logs) — at
# MultiPV=N this expands to N lines per tick, so 2Hz × MultiPV 5 = 10 lines/s
# gets noisy fast. PV extraction (tree walks) runs only on the tick.
_INFO_EMIT_INTERVAL_MS = 1000


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
        tbhits: int, score_mate: int | None, multipv: int | None,
        wdl: tuple[int, int, int] | None,
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
        walker_gather: int = 1,
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
  # Per-walker leaf gather: each walker collects up to this many
  # descents before submitting one NN batch. 1 = classic batch=1
  # per walker; higher amplifies the effective submit batch without
  # spawning more threads. Set via UCI `LeafGather`.
        self._walker_gather = max(1, int(walker_gather))
        self._walker_pool: WalkerPool | None = self._build_walker_pool(self._n_walkers)
        self._walker_cboard: CBoard | None = None
  # Async-pipeline batched-VL PUCT (single-thread, 2-slot CPU/GPU
  # overlap via batch_descend_puct + batch_integrate_leaves). Active
  # only when n_walkers == 1, evaluator exposes evaluate_inplace_async
  # with n_slots >= 2, and ``set_use_pucv(True)`` was called. Bench
  # showed +112% over sync evaluate_encoded — same throughput as
  # gumbel walkers=1 with classic PUCT visit counts.
        self._use_pucv: bool = False
        self._pucv_gather: int = 512
        self._pucv: PucvChunker | None = None
        self._pucv_cboard: CBoard | None = None
  # Multi-GPU pucv pool: one worker thread per GPU, each running its
  # own 2-slot pipeline against the shared tree. Active when set via
  # ``install_multi_gpu_pucv`` — overrides both walker_pool and pucv.
  # Caller is responsible for building the per-GPU evaluators (typically
  # via ``_make_evaluator_factory`` repeated per device).
        self._pucv_pool: MultiGpuPucvPool | None = None
        self._pucv_pool_evals: list[Any] = []
        self._pucv_pool_cboard: CBoard | None = None

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
  # Minibatch target for the C gumbel state machine. 0 = use the
  # C-side GSS_GPU_BATCH default. Higher = better GPU util on large
  # batches; lower = faster stop latency + fresher tree state on
  # each leaf. Set via UCI `MinibatchSize`.
        self._minibatch_size: int = 0
  # MultiPV: emit this many top-ranked lines per info tick. 1 = one PV
  # (classic behavior). >1 triggers a loop that extracts each of the
  # top-N root children by visits, walks a most-visited PV from each,
  # and emits them all with ``multipv N`` fields.
        self._multi_pv: int = 1
  # Emit `wdl W D L` per-mille alongside score_cp. Derived per-line
  # from that line's Q plus a draw-rate estimate from the root NN
  # evaluation (all lines share the same draw rate — they're
  # different continuations of the same root position).
        self._show_wdl: bool = False

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
                gather=self._walker_gather,
            ),
            self._evaluator,
        )

    def set_show_wdl(self, enabled: bool) -> None:
        """Toggle WDL emission on info lines. Takes effect next emit."""
        self._show_wdl = bool(enabled)

    def set_multi_pv(self, n: int) -> None:
        """Number of top-ranked lines to emit per info tick. 1 = classic
        single-PV behavior. Takes effect on the next info emission —
        no rebuild, no tree reset."""
        self._multi_pv = max(1, int(n))

    def set_walker_gather(self, n: int) -> None:
        """Set per-walker leaf gather. Rebuilds the walker pool (now
        spawns N persistent threads, so tear the old pool down first) and
        resets the tree only when walker pool exists; at n_walkers=1 the
        option is accepted but has no runtime effect until threads are
        raised. Caller holds the search barrier."""
        n = max(1, int(n))
        if n == self._walker_gather:
            return
        self._walker_gather = n
        if self._walker_pool is not None:
            self._walker_pool.close()
            self._walker_pool = self._build_walker_pool(self._n_walkers)

    def install_multi_gpu_pucv(
        self,
        evaluators_or_factories: list[Any],
        *,
        gather: int = 384,
        as_factories: bool = False,
    ) -> None:
        """Install N per-GPU evaluators driven by ``MultiGpuPucvPool``.

        Two modes:
          - ``as_factories=False`` (default): pre-built evaluators. Only
            safe when evaluators don't use cudagraphs (CPU/eager/test
            paths). Each evaluator must expose ``evaluate_inplace_async``
            and ``n_slots >= 2``.
          - ``as_factories=True``: callables returning evaluators. Each
            factory is invoked on its own pool worker thread so any
            cudagraph state lives where it'll be replayed. **Required
            for production runs with torch.compile + cudagraphs** —
            building the compiled model on the main thread and replaying
            it from a worker thread crashes with a TLS-key assertion
            inside torch._inductor.cudagraph_trees.

        Replaces both ``_walker_pool`` and ``_pucv`` so this becomes the
        sole search path until ``clear_multi_gpu_pucv`` is called. Caller
        holds the search barrier (same as ``set_evaluator``).
        """
        if not evaluators_or_factories:
            raise ValueError("at least one evaluator/factory required")
        self.clear_multi_gpu_pucv()
        if self._walker_pool is not None:
            self._walker_pool.close()
            self._walker_pool = None
        self._pucv = None
        cfg = MultiGpuPucvConfig(
            n_gpus=len(evaluators_or_factories),
            gather=int(gather),
            c_puct=float(self._cfg.c_puct),
            fpu_at_root=0.0,
            fpu_reduction=float(self._cfg.fpu_reduction),
            vloss_weight=self._vloss_weight,
        )
        if as_factories:
            self._pucv_pool = MultiGpuPucvPool(
                cfg, evaluator_factories=evaluators_or_factories,
            )
  # Keep refs to evaluators built inside workers, for close() teardown.
            self._pucv_pool_evals = list(self._pucv_pool._evals)  # noqa: SLF001
        else:
            self._pucv_pool_evals = list(evaluators_or_factories)
            self._pucv_pool = MultiGpuPucvPool(
                cfg, evaluators=evaluators_or_factories,
            )
        self.reset_tree()

    def clear_multi_gpu_pucv(self) -> None:
        """Tear down the multi-GPU pool and revert to single-evaluator
        gumbel/walker/pucv routing. Idempotent."""
        if self._pucv_pool is not None:
            self._pucv_pool.close()
            self._pucv_pool = None
        self._pucv_pool_evals = []
        self._pucv_pool_cboard = None

    def set_use_pucv(self, enabled: bool, *, gather: int | None = None) -> None:
        """Enable async-pipeline batched-VL PUCT (single-thread, 2-slot
        overlap). Requires ``n_walkers == 1`` and an evaluator with
        ``evaluate_inplace_async`` + ``n_slots >= 2``. When the requirement
        isn't met or ``enabled`` is False, the classic gumbel path is used.
        Resets the tree because pucv accumulates vloss-adjusted Q/N stats
        that don't blend with gumbel's halving stats. Caller holds the
        search barrier."""
        enabled = bool(enabled)
        if gather is not None:
            self._pucv_gather = max(1, int(gather))
        if enabled == self._use_pucv and self._pucv is not None:
            return
        self._use_pucv = enabled
        self._pucv = self._build_pucv() if enabled else None
        self.reset_tree()

    def _build_pucv(self) -> PucvChunker | None:
        if self._n_walkers != 1:
            return None
        ev = self._evaluator
        if not hasattr(ev, "evaluate_inplace_async"):
            return None
        if getattr(ev, "n_slots", 1) < 2:
            return None
        return PucvChunker(
            ev,
            gather=self._pucv_gather,
            c_puct=float(self._cfg.c_puct),
            fpu_at_root=0.0,
            fpu_reduction=float(self._cfg.fpu_reduction),
            vloss_weight=self._vloss_weight,
        )

    def set_minibatch_size(self, n: int) -> None:
        """Set the minibatch accumulation target for the C gumbel state
        machine. 0 means fall back to the C-side default. Takes effect
        on the next ``run_gumbel_root_many_c`` call — no rebuild, no
        tree reset. Just read next time."""
        self._minibatch_size = max(0, int(n))

    def close(self) -> None:
        """Close the current evaluator. Safe to call multiple times; no-op
        if the evaluator has no ``close`` method. Used at process shutdown
        to drain ``BatchCoalescingDispatcher``'s submitter thread before
        Python tears down PyTorch's CUDA context. Also tears down the
        multi-GPU pucv pool's worker threads if active."""
        if self._pucv_pool is not None:
            self._pucv_pool.close()
            self._pucv_pool = None
  # Each per-GPU evaluator may also need close() (BatchCoalescingDispatcher
  # submitter etc.).
        for ev in self._pucv_pool_evals:
            close = getattr(ev, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self._pucv_pool_evals = []
        ev = self._evaluator
        close = getattr(ev, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def set_evaluator(self, evaluator: BatchEvaluator) -> None:
        """Swap in a freshly-built (and warmed-up) evaluator. Rebuilds the
        walker pool so it sees the new reference, resets the tree since
        cached Q values were computed through the old evaluator. Caller
        holds the search barrier (same pattern as ``set_num_threads`` /
        ``set_tb_probe``).

        Closes the outgoing evaluator to release its background threads
        (notably ``BatchCoalescingDispatcher``'s non-daemon submitter).
        Without this, ``setoption MaxBatch`` repeatedly would leak one
        submitter per rebuild and block process shutdown.
        """
        old = self._evaluator
        self._evaluator = evaluator
        if self._walker_pool is not None:
            self._walker_pool.close()
        self._walker_pool = self._build_walker_pool(self._n_walkers)
        if self._use_pucv:
            self._pucv = self._build_pucv()
        self.reset_tree()
        close = getattr(old, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

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
        if self._walker_pool is not None:
            self._walker_pool.close()
        self._walker_pool = self._build_walker_pool(n)
  # pucv requires single-thread; rebuild against the new walker count
  # (returns None when n != 1, which silently disables the path).
        if self._use_pucv:
            self._pucv = self._build_pucv()
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
        self._pucv_cboard = None
        self._pucv_pool_cboard = None

    def _emit_pv_info(
        self,
        info_cb: InfoCallback,
        board: chess.Board,
        root_q: float,
        total_nodes: int,
        elapsed_ms: int,
        tb_probe,
    ) -> None:
        """Emit one or more ``info`` lines, one per top-ranked PV.

        When ``self._multi_pv`` == 1 the ``multipv`` field is omitted
        (classic single-PV UCI). When > 1 each line gets ``multipv 1..N``.
        When ``self._show_wdl`` is on and the root NN eval has been cached,
        each line also gets a ``wdl W D L`` field derived from its Q plus
        the root's NN draw-rate estimate.
        """
        assert self._tree is not None and self._root_id is not None
        lines = _multipv_lines(self._tree, self._root_id, self._multi_pv, root_q)
        if not lines:
            return
        emit_multipv = self._multi_pv > 1
        tbhits = tb_probe.hits if tb_probe is not None else 0
        draw_rate = (
            _root_draw_rate(self._root_wdl_logits[0])
            if self._show_wdl and self._root_wdl_logits is not None
            else 0.0
        )
        for rank, q, pv_idx in lines:
            uci_pv = _uci_pv(board, pv_idx)
            wdl = _q_to_wdl_permille(q, draw_rate) if self._show_wdl else None
            info_cb(
                nodes=total_nodes,
                elapsed_ms=elapsed_ms,
                score_cp=q_to_cp(0.5 * (q + 1.0)),
                pv=uci_pv,
                tbhits=tbhits,
                score_mate=_pv_mate_moves(board, pv_idx),
                multipv=rank if emit_multipv else None,
                wdl=wdl,
            )

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

    def _try_tb_shortcut(
        self, board: chess.Board, tb_probe, deadline: Deadline, info_cb: InfoCallback | None,
    ) -> SearchResult | None:
        """If TB knows the answer at root, return DTZ-optimal move directly.

        MCTS picks by visits — in a TB-win with Q=1.0 everywhere that reduces
        to "most-popular NN prior" and yields valid but DTZ-suboptimal play.
        Bypassing MCTS here is a correctness win, not just a speed win.
        """
        if tb_probe is None:
            return None
        short = _try_tb_root_bestmove(board, tb_probe)
        if short is None:
            return None
        if info_cb is not None:
  # TB-decisive positions get saturated cp; no per-line q to derive
  # a wdl, so we omit the wdl field.
            info_cb(
                nodes=short.nodes,
                elapsed_ms=deadline.elapsed_ms(),
                score_cp=short.score_cp,
                pv=short.pv,
                tbhits=short.tbhits,
                score_mate=short.score_mate,
                multipv=1 if self._multi_pv > 1 else None,
                wdl=None,
            )
        return short

    def _ensure_root_eval_cached(self, board: chess.Board, tb_probe) -> None:
        """Cache root NN eval once per search.

        The root position is the same every chunk; doing the eval once and
        passing pre_pol_logits/pre_wdl_logits into each chunk skips a ~1ms GPU
        call per chunk and lets us share the encoding across chunks.
        """
        if self._root_pol_logits is not None and self._root_wdl_logits is not None:
            return
        xs = np.empty((1, 146, 8, 8), dtype=np.float32)
        root_cb = CBoard.from_board(board)
        xs[0] = root_cb.encode_146()
        pol, wdl = self._evaluator.evaluate_encoded(xs)
        pol_np = np.asarray(pol, dtype=np.float32)
        wdl_np = np.asarray(wdl, dtype=np.float32).copy()
  # Probe at root so score_cp reflects TB truth on the very first chunk's
  # info emission, before MCTS has back-propagated it.
        if tb_probe is not None:
            tb_probe.apply([root_cb], wdl_np)
        self._root_pol_logits = pol_np
        self._root_wdl_logits = wdl_np

    def _pre_expand_root_for_pool(self, board: chess.Board) -> None:
        """Pool paths race on the root's first descent so it must be expanded
        upfront. The classic gumbel path does this internally."""
        if self._pucv_pool is not None:
            self._ensure_pucv_pool_root_expanded(board)
        elif self._walker_pool is not None:
            self._ensure_walker_root_expanded(board)
        elif self._pucv is not None:
            self._ensure_pucv_root_expanded(board)

    def _run_one_chunk(
        self, chunk: int, board: chess.Board, stop_event: threading.Event, tb_probe,
    ) -> float:
        if self._pucv_pool is not None:
            return self._run_pucv_pool_chunk(chunk, stop_event)
        if self._walker_pool is not None:
            return self._run_walker_chunk(chunk, stop_event)
        if self._pucv is not None:
            return self._run_pucv_chunk(chunk)
        return self._run_gumbel_chunk(chunk, board, tb_probe)

    def _maybe_emit_pv_info(
        self,
        *,
        board: chess.Board, deadline: Deadline,
        last_value: float, total_nodes: int,
        info_cb: InfoCallback | None, max_depth: int | None,
        last_info_ms: int, tb_probe,
    ) -> tuple[list[int], int, int]:
        """Extract PV (only when needed) and rate-limited emit-info side effect.

        Returns (pv_indices, last_info_ms, elapsed). PV extraction is skipped
        unless info emission is due or max_depth termination needs it — saves
        a handful of tree walks per second at chunk=512.
        """
        elapsed = deadline.elapsed_ms() if info_cb is not None else 0
        info_due = info_cb is not None and elapsed - last_info_ms >= _INFO_EMIT_INTERVAL_MS
        need_pv = info_due or max_depth is not None
        if not need_pv:
            return [], last_info_ms, elapsed
        assert self._tree is not None and self._root_id is not None
        _, pv_indices = _best_move_and_pv(self._tree, self._root_id)
        if info_due:
            assert info_cb is not None
            self._emit_pv_info(
                info_cb, board, float(last_value), total_nodes, elapsed, tb_probe,
            )
            last_info_ms = elapsed
        return pv_indices, last_info_ms, elapsed

    def _build_final_search_result(
        self, *, board: chess.Board, total_nodes: int, last_value: float, tb_probe,
    ) -> SearchResult:
        """Final snapshot of the searched tree → SearchResult."""
        assert self._tree is not None and self._root_id is not None
        bestmove_idx, pv_indices = _best_move_and_pv(self._tree, self._root_id)
        ponder_idx = _predicted_opponent_reply(self._tree, self._root_id)
        bestmove = _index_to_uci(board, bestmove_idx)
        ponder = (
            _index_to_uci(_board_after(board, bestmove_idx), ponder_idx)
            if ponder_idx is not None else None
        )
        return SearchResult(
            bestmove_uci=bestmove,
            ponder_uci=ponder,
            nodes=total_nodes,
            pv=_uci_pv(board, pv_indices),
            score_cp=q_to_cp(0.5 * (last_value + 1.0)),
            tbhits=tb_probe.hits if tb_probe is not None else 0,
            score_mate=_pv_mate_moves(board, pv_indices),
        )

    def _should_stop_search(
        self,
        *,
        stop_event: threading.Event, deadline: Deadline,
        max_nodes: int | None, max_depth: int | None,
        total_nodes: int, pv_len: int,
    ) -> str | None:
        """Return the first-firing stop reason, or ``None`` to keep searching.

        Reasons are checked in priority order so the caller can match the
        original sequential-break semantics — ``"tree_bytes"`` only fires when
        no earlier budget/external/depth limit was already hit, which matters
        for the tree-bytes-only final info-cb emission in :meth:`run`.
        """
        if stop_event.is_set() or deadline.expired():
            return "external"
        if max_nodes is not None and total_nodes >= max_nodes:
            return "max_nodes"
        if max_depth is not None and pv_len >= max_depth:
            return "max_depth"
        if (
            self._max_tree_bytes > 0
            and self._tree is not None
            and self._tree.memory_bytes() >= self._max_tree_bytes
        ):
            return "tree_bytes"
        return None

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

        tb_probe = self._tb_probe
        if tb_probe is not None:
            tb_probe.reset_counts()

        short = self._try_tb_shortcut(board, tb_probe, deadline, info_cb)
        if short is not None:
            return short

        self._ensure_root_eval_cached(board, tb_probe)
        self._pre_expand_root_for_pool(board)

        total_nodes = 0
        last_info_ms = -1
        last_value = 0.0
        pv_indices: list[int] = []
        elapsed = 0
        while True:
            chunk = self._chunk_sims
            if max_nodes is not None:
                remaining = max_nodes - total_nodes
                if remaining <= 0:
                    break
                chunk = min(chunk, remaining)

            last_value = self._run_one_chunk(chunk, board, stop_event, tb_probe)
            total_nodes += int(chunk)

            pv_indices, last_info_ms, elapsed = self._maybe_emit_pv_info(
                board=board, deadline=deadline,
                last_value=last_value, total_nodes=total_nodes,
                info_cb=info_cb, max_depth=max_depth,
                last_info_ms=last_info_ms, tb_probe=tb_probe,
            )

            stop_reason = self._should_stop_search(
                stop_event=stop_event, deadline=deadline,
                max_nodes=max_nodes, max_depth=max_depth,
                total_nodes=total_nodes, pv_len=len(pv_indices),
            )
            if stop_reason is not None:
  # Tree-bytes-only stops emit a final info line; other stop reasons
  # either already emitted via _maybe_emit_pv_info above or want quiet exit.
                if stop_reason == "tree_bytes" and info_cb is not None:
                    assert self._root_id is not None
                    self._emit_pv_info(
                        info_cb, board, float(last_value),
                        total_nodes, elapsed, tb_probe,
                    )
                break

        return self._build_final_search_result(
            board=board, total_nodes=total_nodes,
            last_value=last_value, tb_probe=tb_probe,
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
            cfg=dataclasses.replace(self._cfg, simulations=chunk, add_noise=False),
            evaluator=self._evaluator,
            pre_pol_logits=self._root_pol_logits,
            pre_wdl_logits=self._root_wdl_logits,
            tree=self._tree,
            root_node_ids=[self._root_id] if self._root_id is not None else None,
            tb_probe=tb_probe,
            target_batch=self._minibatch_size,
        )
        self._tree = tree
        self._root_id = int(root_ids[0])
        return float(values[0])

    def _run_pucv_pool_chunk(
        self, chunk: int, stop_event: threading.Event,
    ) -> float:
        assert self._tree is not None and self._root_id is not None
        assert self._pucv_pool is not None and self._pucv_pool_cboard is not None
        self._pucv_pool.run(
            tree=self._tree,
            root_id=self._root_id,
            root_cboard=self._pucv_pool_cboard,
            target_sims=chunk,
            stop_event=stop_event,
        )
        return self._tree.node_q(self._root_id)

    def _ensure_pucv_pool_root_expanded(self, board: chess.Board) -> None:
        """Same root-prep contract as walker_pool / pucv: pool workers
        race on the root's first descent, so it must be expanded before
        run() returns control. Mirrors ``_ensure_walker_root_expanded``."""
        assert self._root_pol_logits is not None
        if self._tree is None:
            self._tree = MCTSTree()
            self._tree.reserve(50_000, 500_000)
            self._root_id = None
        if self._root_id is None:
            assert self._root_wdl_logits is not None
            root_q = float(_value_scalar_from_wdl_logits(self._root_wdl_logits[0]))
            self._root_id = int(self._tree.add_root(0, root_q))
        if self._pucv_pool_cboard is None:
            self._pucv_pool_cboard = CBoard.from_board(board)
        if not self._tree.is_expanded(self._root_id):
            legal_idx = self._pucv_pool_cboard.legal_move_indices()
            if legal_idx.size > 0:
                self._tree.expand_from_logits(
                    self._root_id,
                    legal_idx.astype(np.int32),
                    self._root_pol_logits[0],
                )

    def _run_pucv_chunk(self, chunk: int) -> float:
        assert self._tree is not None and self._root_id is not None
        assert self._pucv is not None and self._pucv_cboard is not None
        self._pucv.run(
            tree=self._tree,
            root_id=self._root_id,
            root_cboard=self._pucv_cboard,
            target_sims=chunk,
        )
        return self._tree.node_q(self._root_id)

    def _ensure_pucv_root_expanded(self, board: chess.Board) -> None:
        """pucv path needs the root pre-expanded (batch_descend_puct expects
        an expanded root) and a CBoard handle reused across calls. Mirrors
        ``_ensure_walker_root_expanded``."""
        assert self._root_pol_logits is not None
        if self._tree is None:
            self._tree = MCTSTree()
            self._tree.reserve(50_000, 500_000)
            self._root_id = None
        if self._root_id is None:
            assert self._root_wdl_logits is not None
            root_q = float(_value_scalar_from_wdl_logits(self._root_wdl_logits[0]))
            self._root_id = int(self._tree.add_root(0, root_q))
        if self._pucv_cboard is None:
            self._pucv_cboard = CBoard.from_board(board)
        if not self._tree.is_expanded(self._root_id):
            legal_idx = self._pucv_cboard.legal_move_indices()
            if legal_idx.size > 0:
                self._tree.expand_from_logits(
                    self._root_id,
                    legal_idx.astype(np.int32),
                    self._root_pol_logits[0],
                )

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
  # _root_pol_logits is shape (1, 4672) — we cache the
  # batched eval output. expand_from_logits wants 1D.
                self._tree.expand_from_logits(
                    self._root_id,
                    legal_idx.astype(np.int32),
                    self._root_pol_logits[0],
                )


# --- tree + move helpers -----------------------------------------------------


def _root_draw_rate(wdl_logits: np.ndarray) -> float:
    """Softmax the root's raw NN [w, d, l] logits and return the draw
    probability. Used by UCI_ShowWDL to derive per-line WDL from per-line
    Q — the draw rate is shared across all multipv lines since they're
    all continuations of the same root position."""
    z = np.asarray(wdl_logits, dtype=np.float64).ravel()
    z = z - z.max()
    e = np.exp(z)
    return float(e[1] / e.sum())


def _q_to_wdl_permille(q: float, draw_rate: float) -> tuple[int, int, int]:
    """Split Q ∈ [-1, 1] into (W, D, L) per-mille using a fixed draw rate.
    Q = W - L after expected draws are removed. Clamps draw so W and L
    stay non-negative."""
    q = max(-1.0, min(1.0, float(q)))
    d = max(0.0, min(1.0 - abs(q), float(draw_rate)))
    w = (q + 1.0 - d) / 2.0
    l = (1.0 - q - d) / 2.0
    return (
        int(round(w * 1000)),
        int(round(d * 1000)),
        int(round(l * 1000)),
    )


def _multipv_lines(
    tree: MCTSTree, root_id: int, n: int, root_q_default: float,
) -> list[tuple[int, float, list[int]]]:
    """Return up to ``n`` (rank, q, pv_indices) triples for the top-visited
    root children. Rank is 1-based (UCI convention: ``multipv 1`` = best).
    Each pv_indices walks the most-visited path from that root child.

    At n=1 this is equivalent to ``_best_move_and_pv`` plus a Q read,
    so callers can always route through this helper.
    """
    actions, visits, qs = tree.get_children_q(root_id, root_q_default)
    if actions.size == 0:
        return []
  # Sort descending by visits; ties tolerated — argsort is stable in numpy.
    order = np.argsort(-visits)[:max(1, int(n))]
    out: list[tuple[int, float, list[int]]] = []
    for rank, i in enumerate(order.tolist(), start=1):
        move = int(actions[i])
        q = float(qs[i])
        pv = [move]
        cid = tree.find_child(root_id, move)
        while cid != -1:
            a, vs = tree.get_children_visits(cid)
            if a.size == 0:
                break
            nxt = int(a[int(np.argmax(vs))])
            pv.append(nxt)
            cid = tree.find_child(cid, nxt)
        out.append((rank, q, pv))
    return out




def _try_tb_root_bestmove(
    board: chess.Board, tb_probe: SyzygyProbe,
) -> SearchResult | None:
    """Return a SearchResult built from the TB's DTZ-optimal move at root,
    or None if the position isn't TB-eligible (or the probe fails)."""
    root = try_tb_root_move(board, tb_probe._path)
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
    ponder = try_tb_root_move(board_after, tb_probe._path)
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
    """Most-visited root move and its PV. ``q`` is unused at the call sites
    that take this shape (so the third tuple element from ``_multipv_lines``
    is dropped here)."""
    lines = _multipv_lines(tree, root_id, root_q_default=0.0, n=1)
    if not lines:
        return -1, []
    _, _, pv = lines[0]
    return pv[0], pv


def _predicted_opponent_reply(tree: MCTSTree, root_id: int) -> int | None:
    """Move index the opponent is predicted to play after OUR bestmove.

    This is what we ponder on, not our own alternative — we take the
    most-visited root child (our bestmove), descend to that node, and
    return ITS most-visited child (opponent's best reply at that node).

    Distinct from the "root's second-most-visited child" which would be
    our 2nd-best move from the current position — a different concept.
    """
    lines = _multipv_lines(tree, root_id, root_q_default=0.0, n=1)
    if not lines or len(lines[0][2]) < 2:
        return None
    return lines[0][2][1]


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


