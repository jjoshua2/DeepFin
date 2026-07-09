"""Chunked MCTS search worker.

Runs ``run_gumbel_root_many_c`` in small sim-chunks so we can check a stop
event between calls. Threads ``tree`` + ``root_node_ids`` across chunks so
each chunk continues the previous tree rather than starting over.

The worker is deliberately oblivious to UCI time control and ponder protocol;
``Engine`` wraps it with the cooperation protocol. This keeps search pure
and makes the v2 multi-GPU swap a local change.
"""
from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Protocol

import chess
import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.inference import BatchEvaluator
from chess_anti_engine.mcts import _mcts_tree as _mcts_tree_ext
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import _REQUIRED_MCTS_ABI, run_gumbel_root_many_c
from chess_anti_engine.mcts.root_tactics import immediate_mate_move
from chess_anti_engine.mcts.puct import _value_scalar_from_wdl_logits
from chess_anti_engine.mcts.puct_vl import PucvChunker
from .multi_gpu_pucv_pool import (
    MultiGpuPucvConfig,
    MultiGpuPucvPool,
    MultiGpuPucvStats,
    MultiGpuPucvWorkerStats,
)
from .root_parallel_gumbel import (
    RootParallelGumbelConfig,
    RootParallelGumbelPool,
    RootParallelGumbelStats,
)
from chess_anti_engine.moves import index_to_move, move_to_index
from chess_anti_engine.tablebase import SyzygyProbe, try_tb_root_move

from .score import q_to_cp
from .time_manager import Deadline
from .walker_pool import WalkerPool, WalkerPoolConfig
import contextlib
import time

# _REQUIRED_MCTS_ABI is imported from gumbel_c (canonical owner of the C-path ABI
# contract), so the SearchWorker construction guard and the run_gumbel_root_many_c
# entry guard agree on one value.

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

# --- shared-tree concurrency: pre-grow to avoid mid-chunk realloc ------------
# The walker / multi-GPU pucv pools mutate one MCTSTree from N threads. The C
# extension reads node/child arrays lock-free during descent, and `_mcts_tree.c`
# documents that a `tree_grow_*` realloc mid-descent can hand a concurrent
# reader a stale (freed) base pointer — a use-after-free. Its contract: callers
# "MUST call MCTSTree.reserve(max_nodes) upfront so no realloc fires during
# concurrent descent." A real-game search has no node cap (it runs to the time
# deadline), so node_count climbs past any fixed reserve and the arena reallocs
# mid-chunk. We instead reserve just-in-time *between* chunks — a quiescent
# point where the pool has joined its barrier and no worker is descending —
# sized to the next chunk's worst-case growth. reserve() grows via the same
# internal doubling the mid-chunk realloc would have used, so node_cap /
# memory_bytes() (and thus the byte-cap and advance_root half-cap behavior) are
# unchanged: we only move the realloc to a safe point.
#
# Worst-case nodes a single concurrent sim adds to the arena, so a chunk of S
# sims adds at most S * _TREE_MAX_LEAF_BRANCHING nodes (and the same number of
# child edges, since #edges == #nodes-1 in a tree). Two contributions:
#   - the leaf expansion eagerly creates one child per legal move, up to
#     _MAX_LEGAL_MOVES (the absolute maximum in any chess position); plus
#   - on the walker path, walker_descend_puct first runs try_forced_collapse for
#     up to _FORCED_COLLAPSE_DEPTH plies, each expanding one forced single-child
#     node along the descent (the pucv pool's batch_descend_puct omits forced
#     collapse, so only the leaf expansion applies there).
# 256 covers the worst case (218 + 16 = 234) with margin.
_MAX_LEGAL_MOVES = 218
_FORCED_COLLAPSE_DEPTH = 16  # try_forced_collapse depth cap in walker_descend_puct
_TREE_MAX_LEAF_BRANCHING = 256
assert _TREE_MAX_LEAF_BRANCHING >= _MAX_LEGAL_MOVES + _FORCED_COLLAPSE_DEPTH

# Visit-margin early abort (Lc0 "smart pruning"). Once past the optimum budget
# the search stops as soon as the move it would play has any visit lead; before
# the optimum it can also stop early if that lead is large enough that the
# runner-up cannot overtake it within the remaining optimum budget. The default
# factor 1.0 is the (approximately) provable bound — the runner-up would need
# *every* remaining sim to catch up; factor < 1.0 is the aggressive Lc0-style
# bet that it won't, banking more time for harder moves. The move whose lead is
# measured is the one final selection emits (Gumbel survivor or visit leader),
# so on the Gumbel production path the abort tracks the played move.
_DEFAULT_ABORT_FACTOR = 1.0
# Complexity gate on the early abort (when to bank vs "use it all"). The aggressive
# Lc0-style allocation over-budgets and relies on the abort to reclaim the slack on
# EASY moves; these decide which moves count as easy. A move banks only when it is
# both VISIT-decisive (its share of root visits leads the best alternative by >= the
# margin) AND stable (the played move repeated for a few consecutive chunks). Hard
# positions (a near-tie in visit share, or a still-flipping choice) keep searching
# toward the hard deadline. Flag-safe: the allocation already bounds a single move's
# spend, so suppressing the abort only spends budget already deemed safe.
#
# The decisiveness feature is the VISIT-share gap, not the root Q gap: the time-value
# backtest (scripts/backtest_time_value.py) found visit-gap predicts "more search
# won't change the move" ~1.5-2x better than the Q gap across the calibrated search
# configs. 0.30 fired rarely in high-branching middlegames (the engine then searched
# to the deadline on most moves); 0.25 lets the abort bank a touch more readily while
# the consecutive-stable-chunks requirement still guards against premature banking.
# Heuristic — sweep with arena games (visit share is in [0, 1]).
_ABORT_VISIT_GAP_MARGIN = 0.25
_ABORT_MIN_STABLE_CHUNKS = 2

# First-chunk time bootstrap. Before any chunk has run there's no measured nps, so
# _time_capped_chunk can't size the first chunk to the budget. Cap it with a
# CONSERVATIVE nps (well under the ~17.6 sims/ms single-game ceiling, and below a
# cold first-forward rate) so a tight deadline (e.g. 100ms matched_time) shrinks the
# first chunk instead of overrunning, while a normal/long budget leaves it at the
# full _chunk_sims. Lower-bounding nps means we under-spend the budget, never over.
_BOOTSTRAP_NPS_PER_MS = 6.0
_MIN_FIRST_CHUNK = 64

# Batch-time-variance clock margin. The deadline is only checked BETWEEN chunks,
# and a chunk (one GPU batch) can't be interrupted once launched — so a chunk that
# runs slower than the running-average nps can overrun a tight deadline. That is the
# time-forfeit mode a fixed MoveOverhead can't cover: batch wall-time varies with the
# position, tree state, and GPU load, so the right reserve is data-driven, not fixed.
# Once a few chunks have been timed this search, stop launching new ones when the
# remaining time is below mean + K*std of measured chunk durations, banking one
# worst-case batch as margin. Clock games only (``optimum_ms`` set) — movetime / nodes
# / depth searches asked for exact work and must not stop short. K=2 ≈ a one-sided
# 97.7% batch-fits bound under a normal approximation; tune via UCI
# ``ClockBatchMarginSigmas`` (0 disables, restoring the pre-margin behavior).
_BATCH_MARGIN_SIGMAS = 2.0
_BATCH_MARGIN_MIN_SAMPLES = 2
# Cap the reserve at this fraction of the move's starting budget so an
# anomalously slow early chunk (a transient stall inflating mean+K*std) can't
# reserve away the whole move and stop the search after one or two chunks — the
# opposite failure (weak near-instant moves). At normal TCs budget >> chunk so
# this never binds; it only guards the pathological short-TC / high-variance case.
_MAX_BATCH_MARGIN_FRACTION = 0.5


def _visit_gap(actions: np.ndarray, visits: np.ndarray, best_action: int) -> float:
    """``best_action``'s share of the root visits minus the best alternative's share.
    A large gap means visits have concentrated on the played move (settled); a small
    gap means the search is still splitting them (hard). Single-move positions return
    the played move's full share; a best_action absent from ``actions`` returns 0."""
    total = float(visits.sum())
    if total <= 0.0 or actions.size == 0:
        return 0.0
    shares = visits.astype(np.float64) / total
    mask = actions == best_action
    if not bool(mask.any()):
        return 0.0
    best_share = float(shares[mask].max())
    others = shares[~mask]
    return best_share - (float(others.max()) if others.size else 0.0)


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
        wdl: tuple[int, int, int] | None, string: str | None = None,
        hashfull: int | None = None, seldepth: int | None = None,
    ) -> None:
        ...


def _pucv_stats_string(
    stats: MultiGpuPucvStats | None,
    *,
    pending_mode: int,
) -> str | None:
    if stats is None or stats.leaves <= 0:
        return None
    workers: tuple[MultiGpuPucvWorkerStats, ...] = stats.workers
    if not workers:
        return None
    worker_leaves = ",".join(str(w.leaves) for w in workers)
    max_batch = max((w.max_batch for w in workers), default=0)
    mode = "virtual-mean" if pending_mode == 1 else "legacy"
    cache_part = ""
    if stats.cache_requests > 0:
        cache_part = (
            f" cache={stats.cache_hits}/{stats.cache_requests}"
            f"({stats.cache_hit_rate:.1%})"
        )
    return (
        f"pucv leaves={stats.leaves} batches={stats.batches} "
        f"avg_batch={stats.avg_batch:.1f} max_batch={max_batch} "
        f"workers={worker_leaves} pending={mode}{cache_part}"
    )


def _single_pucv_cache_stats_string(
    stats: Any,
    *,
    pending_mode: int,
) -> str | None:
    if stats is None or stats.requests <= 0:
        return None
    mode = "virtual-mean" if pending_mode == 1 else "legacy"
    return (
        f"pucv pending={mode} "
        f"cache={stats.hits}/{stats.requests}({stats.hit_rate:.1%})"
    )


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
        pucv_vloss_mode: int = 0,
        eval_cache_entries: int = 0,
    ) -> None:
  # Fail fast on a stale compiled extension. Two ABI changes ship here that Python
  # relies on but can't detect by signature: `MCTSTree.node_capacity()` (used by
  # `_ensure_shared_tree_headroom`) and the new `start_gumbel_sims` root-scale args
  # (`c_scale_root`/`q_visit_exp_root`). An un-rebuilt `.so` would otherwise fail
  # mid-search (losing the game on time), and a node_capacity-only check wouldn't
  # catch a `.so` built between the two changes. The module ABI_VERSION marker covers
  # both — check it at construction and surface the rebuild command.
        _abi = getattr(_mcts_tree_ext, "ABI_VERSION", 0)
        if _abi < _REQUIRED_MCTS_ABI:
            raise RuntimeError(
                f"compiled _mcts_tree ABI_VERSION={_abi} < required {_REQUIRED_MCTS_ABI} "
                "(missing node_capacity and/or the start_gumbel_sims root-scale args); "
                "rebuild the C extension: python setup.py build_ext --inplace"
            )
        self._evaluator = evaluator
        self._device = device
        self._cfg = gumbel_cfg or GumbelConfig(
            simulations=chunk_sims,
            add_noise=False,  # no exploration noise at test time
            temperature=0.0,
        )
        self._chunk_sims = int(chunk_sims)
        self._rng = np.random.default_rng()
  # n_walkers > 1 → PUCT pool with vloss, sequential halving dropped.
  # evaluator MUST be thread-safe (caller wraps with Thread/MultiGPU
  # dispatcher or BatchCoalescingDispatcher).
        self._vloss_weight = int(vloss_weight)
        self._pucv_vloss_mode = 1 if int(pucv_vloss_mode) == 1 else 0
        self._eval_cache_entries = max(0, int(eval_cache_entries))
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
  # Root-parallel Gumbel pool (multi-GPU search v2, quality-preserving
  # alternative to the pucv pool): per-device evaluator groups run each
  # surviving Gumbel candidate's phase budget in its own subtree of the
  # shared tree; halving decisions happen at single-threaded phase
  # barriers. Installed via ``install_root_parallel_gumbel`` — mutually
  # exclusive with the pucv pool.
        self._rpg_pool: RootParallelGumbelPool | None = None
        self._rpg_pool_evals: list[Any] = []

  # Persistent tree across calls within a game. Reset on new position.
        self._tree: MCTSTree | None = None
        self._root_id: int | None = None
        self._tree_fen: str | None = None
  # Cache of the root's policy + WDL logits. Valid for as long as the
  # tree is valid (same position). Lets chunks after the first skip
  # the ~1ms root GPU call.
        self._root_pol_logits: np.ndarray | None = None
        self._root_wdl_logits: np.ndarray | None = None
        self._last_gumbel_action_idx: int | None = None
  # Best-move stability across chunks, for the complexity gate on the early abort
  # (see ``_abort_ready``). Reset at the start of every search.
        self._abort_last_best: int = -1
        self._abort_stable_chunks: int = 0
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
  # Clock-margin width in std-devs of measured chunk wall-time (see
  # `_BATCH_MARGIN_SIGMAS`). Set via UCI `ClockBatchMarginSigmas`; 0 disables.
        self._batch_margin_sigmas: float = _BATCH_MARGIN_SIGMAS
  # Per-search Welford accumulator of chunk durations (ms), reset in `run()`.
        self._batch_n: int = 0
        self._batch_mean_ms: float = 0.0
        self._batch_m2_ms: float = 0.0
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
  # Selective depth: the deepest principal-variation length observed during
  # the current search (monotonic max, reset per `run()`). A real, climbing
  # number that GUIs/TCEC spectators expect alongside `depth`.
        self._seldepth: int = 0

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
                input_planes=input_plane_count(self._cfg.input_extra_features),
                compute_relations=bool(self._cfg.compute_relations),
            ),
            self._evaluator,
        )

    def set_show_wdl(self, enabled: bool) -> None:
        """Toggle WDL emission on info lines. Takes effect next emit."""
        self._show_wdl = bool(enabled)

    def _hashfull_permille(self) -> int | None:
        """Tree-memory fill as per-mille of the soft cap (UCI ``hashfull``).

        The cap is ``set_max_tree_mb`` (UCI ``Hash``); ``memory_bytes()`` is the
        arena's allocated size, the same quantity the byte-cap stop checks. None
        when the cap is disabled (0) or no tree exists yet."""
        if self._max_tree_bytes <= 0 or self._tree is None:
            return None
        used = int(self._tree.memory_bytes())
        return max(0, min(1000, used * 1000 // self._max_tree_bytes))

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
        # Teardown only — do not rebuild walker/PUCV that we immediately close.
        self.clear_multi_gpu_pucv(restore_fallback=False)
        self.clear_root_parallel_gumbel()
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
            vloss_mode=self._pucv_vloss_mode,
            eval_cache_entries=self._eval_cache_entries,
            input_planes=input_plane_count(self._cfg.input_extra_features),
            compute_relations=bool(self._cfg.compute_relations),
        )
        if as_factories:
            self._pucv_pool = MultiGpuPucvPool(
                cfg, evaluator_factories=evaluators_or_factories,
            )
  # Keep refs to evaluators built inside workers, for close() teardown.
            self._pucv_pool_evals = list(self._pucv_pool._evals)
        else:
            self._pucv_pool_evals = list(evaluators_or_factories)
            self._pucv_pool = MultiGpuPucvPool(
                cfg, evaluators=evaluators_or_factories,
            )
        self.reset_tree()

    def clear_multi_gpu_pucv(self, *, restore_fallback: bool = True) -> None:
        """Tear down the multi-GPU pool; optionally restore walker/pucv routing.

        ``install_multi_gpu_pucv`` closes ``_walker_pool`` and nulls ``_pucv``
        so the pool is the sole search path. Dropping the pool without
        rebuilding those fallbacks leaves the advertised default
        ``Threads=2`` on the single-thread Gumbel path (and leaves
        ``UseVL`` inert even though ``_use_pucv`` is still set).

        Pass ``restore_fallback=False`` from callers that immediately rebuild
        the path they want (``install_multi_gpu_pucv``, ``set_evaluator``) so
        we do not spawn a walker pool only to tear it down again. Default
        ``True`` is for UCI opt-out. Idempotent.
        """
        had_pool = self._pucv_pool is not None or bool(self._pucv_pool_evals)
        if self._pucv_pool is not None:
            self._pucv_pool.close()
            self._pucv_pool = None
        for ev in self._pucv_pool_evals:
            close = getattr(ev, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        self._pucv_pool_evals = []
        self._pucv_pool_cboard = None
        if not had_pool or not restore_fallback:
            return
        # Restore the path install_multi_gpu_pucv tore down.
        if self._walker_pool is not None:
            self._walker_pool.close()
        self._walker_pool = self._build_walker_pool(self._n_walkers)
        if self._use_pucv:
            self._pucv = self._build_pucv()
        else:
            self._pucv = None
        self.reset_tree()

    def last_multi_gpu_pucv_stats(self) -> MultiGpuPucvStats | None:
        """Return stats from the most recent multi-GPU PUCV chunk."""
        if self._pucv_pool is None:
            return None
        return self._pucv_pool.last_stats()

    def last_single_pucv_cache_stats(self) -> Any | None:
        """Return cache stats for the active single-thread PUCV chunker."""
        if self._pucv is None:
            return None
        return self._pucv.cache_stats()

    def install_root_parallel_gumbel(
        self,
        evaluators_or_factories: list[Any],
        *,
        gather: int = 512,
        as_factories: bool = False,
        devices: list[str] | None = None,
        info_string_cb: Callable[[str], None] | None = None,
    ) -> None:
        """Install N per-device evaluator groups driven by
        ``RootParallelGumbelPool`` (root-parallel Gumbel, design §2).

        Same factory-vs-prebuilt contract as ``install_multi_gpu_pucv``
        (factories are REQUIRED for torch.compile + cudagraphs; each is
        invoked on its group's own thread, with ``torch.cuda.set_device``
        first when ``devices`` is provided). Replaces every other search
        path until ``clear_root_parallel_gumbel``. Caller holds the search
        barrier.
        """
        if not evaluators_or_factories:
            raise ValueError("at least one evaluator/factory required")
        self.clear_root_parallel_gumbel()
        # Teardown only — install immediately owns the path.
        self.clear_multi_gpu_pucv(restore_fallback=False)
        if self._walker_pool is not None:
            self._walker_pool.close()
            self._walker_pool = None
        self._pucv = None
        cfg = RootParallelGumbelConfig(
            n_groups=len(evaluators_or_factories),
            gather=int(gather),
            c_puct=float(self._cfg.c_puct),
            fpu_at_root=0.0,
            fpu_reduction=float(self._cfg.fpu_reduction),
            vloss_weight=self._vloss_weight,
  # vloss_mode stays at the config default (virtual-mean): the design pins
  # the intra-candidate regime; the pucv pool's PUCVPendingMode default
  # (legacy) deliberately does not leak in here.
            input_planes=input_plane_count(self._cfg.input_extra_features),
            compute_relations=bool(self._cfg.compute_relations),
        )
        if as_factories:
            self._rpg_pool = RootParallelGumbelPool(
                cfg, self._cfg,
                evaluator_factories=evaluators_or_factories,
                devices=devices,
                info_string_cb=info_string_cb,
            )
  # Keep refs to evaluators built inside workers, for close() teardown.
            self._rpg_pool_evals = list(self._rpg_pool._evals)
        else:
            self._rpg_pool_evals = list(evaluators_or_factories)
            self._rpg_pool = RootParallelGumbelPool(
                cfg, self._cfg,
                evaluators=evaluators_or_factories,
                devices=devices,
                info_string_cb=info_string_cb,
            )
        self.reset_tree()

    def clear_root_parallel_gumbel(self) -> None:
        """Tear down the root-parallel Gumbel pool and revert to the
        single-evaluator gumbel/walker/pucv routing. Idempotent."""
        if self._rpg_pool is not None:
            self._rpg_pool.close()
            self._rpg_pool = None
        for ev in self._rpg_pool_evals:
            close = getattr(ev, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        self._rpg_pool_evals = []

    def last_root_parallel_gumbel_stats(self) -> RootParallelGumbelStats | None:
        """Return stats from the most recent root-parallel Gumbel chunk."""
        if self._rpg_pool is None:
            return None
        return self._rpg_pool.last_stats()

    def set_pucv_vloss_mode(self, mode: int) -> None:
        """Set batched-PUCV pending accounting.

        ``0`` keeps legacy absolute virtual-loss scoring. ``1`` uses
        virtual-mean pending samples during batched PUCT selection. The
        walker-pool path is unchanged because it uses the single-leaf C API.
        """
        mode = 1 if int(mode) == 1 else 0
        if mode == self._pucv_vloss_mode:
            return
        self._pucv_vloss_mode = mode
        if self._pucv is not None:
            self._pucv = self._build_pucv()
        if self._pucv_pool is not None:
            self._pucv_pool._cfg.vloss_mode = mode
        self.reset_tree()

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
            vloss_mode=self._pucv_vloss_mode,
            eval_cache_entries=self._eval_cache_entries,
            input_planes=input_plane_count(self._cfg.input_extra_features),
            compute_relations=bool(self._cfg.compute_relations),
        )

    def set_eval_cache_entries(self, n: int) -> None:
        """Set eval-cache capacity for newly built search helpers.

        This is behavior-preserving, so changing it does not require a tree
        reset. If single-thread PUCV is active, rebuild the chunker so the
        capacity takes effect immediately.
        """
        n = max(0, int(n))
        if n == self._eval_cache_entries:
            return
        self._eval_cache_entries = n
        if self._pucv is not None:
            self._pucv = self._build_pucv()

    def set_minibatch_size(self, n: int) -> None:
        """Set the minibatch accumulation target for the C gumbel state
        machine. 0 means fall back to the C-side default. Takes effect
        on the next ``run_gumbel_root_many_c`` call — no rebuild, no
        tree reset. Just read next time."""
        self._minibatch_size = max(0, int(n))

    def set_batch_margin_sigmas(self, sigmas: float) -> None:
        """Set the batch-time clock-margin width in std-devs (0 disables).

        Read live per stop-check; no rebuild or tree reset. See
        ``_BATCH_MARGIN_SIGMAS`` for the semantics.
        """
        self._batch_margin_sigmas = max(0.0, float(sigmas))

    def _reset_batch_timing(self) -> None:
        """Clear the per-search chunk-duration accumulator (call in ``run()``)."""
        self._batch_n = 0
        self._batch_mean_ms = 0.0
        self._batch_m2_ms = 0.0

    def _record_batch_ms(self, dt_ms: float) -> None:
        """Fold one chunk's wall-time into the online mean/variance (Welford)."""
        self._batch_n += 1
        delta = dt_ms - self._batch_mean_ms
        self._batch_mean_ms += delta / self._batch_n
        self._batch_m2_ms += delta * (dt_ms - self._batch_mean_ms)

    def _batch_margin_ms(self) -> float:
        """Clock reserve = mean + K*std of measured chunk time.

        Returns 0 until ``_BATCH_MARGIN_MIN_SAMPLES`` chunks have run (no
        estimate yet) or when the margin is disabled (sigmas <= 0)."""
        if (
            self._batch_margin_sigmas <= 0.0
            or self._batch_n < _BATCH_MARGIN_MIN_SAMPLES
        ):
            return 0.0
        var = self._batch_m2_ms / (self._batch_n - 1)
        std = var ** 0.5 if var > 0.0 else 0.0
        return self._batch_mean_ms + self._batch_margin_sigmas * std

    def _clock_time_margin_ms(
        self, optimum_ms: int | None, start_remaining_ms: int | None,
    ) -> float:
        """Capped batch-time reserve passed to the stop check.

        Zero unless this is a clock game (``optimum_ms`` set) — ``movetime`` /
        ``nodes`` / ``depth`` searches run their exact requested work. Capped at
        ``_MAX_BATCH_MARGIN_FRACTION`` of the move's starting budget so a bad
        early chunk estimate can't reserve away the whole move.
        """
        if optimum_ms is None:
            return 0.0
        m = self._batch_margin_ms()
        if m > 0.0 and start_remaining_ms is not None:
            m = min(m, _MAX_BATCH_MARGIN_FRACTION * float(start_remaining_ms))
        return m

    def close(self) -> None:
        """Close the current evaluator. Safe to call multiple times; no-op
        if the evaluator has no ``close`` method. Used at process shutdown
        to drain ``BatchCoalescingDispatcher``'s submitter thread before
        Python tears down PyTorch's CUDA context. Also tears down the
        multi-GPU pucv pool's worker threads if active."""
        self.clear_root_parallel_gumbel()
        if self._pucv_pool is not None:
            self._pucv_pool.close()
            self._pucv_pool = None
  # Each per-GPU evaluator may also need close() (BatchCoalescingDispatcher
  # submitter etc.).
        for ev in self._pucv_pool_evals:
            close = getattr(ev, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        self._pucv_pool_evals = []
        ev = self._evaluator
        close = getattr(ev, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()

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
        # Teardown only — rebuild walker/pucv against the new evaluator below.
        # Always rebuild the fallback path here (even if a multi-GPU reinstall
        # will close the walker next): if install fails we must not leave the
        # worker with no search path.
        self.clear_multi_gpu_pucv(restore_fallback=False)
        self.clear_root_parallel_gumbel()
        if self._walker_pool is not None:
            self._walker_pool.close()
        self._walker_pool = self._build_walker_pool(self._n_walkers)
        if self._use_pucv:
            self._pucv = self._build_pucv()
        else:
            self._pucv = None
        self.reset_tree()
        close = getattr(old, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()

    def set_num_threads(self, n: int) -> None:
        """Rebuild the walker pool at thread count ``n`` (1 = classic Gumbel
        path, no pool). Drops the persistent tree because walker-pool Q/N
        stats accumulate with vloss adjustments that depend on thread count.
        Caller should hold the search barrier — same pattern as
        ``set_tb_probe``.

        When the multi-GPU PUCV pool is active it owns search; only store
        ``n`` for the next ``clear_multi_gpu_pucv`` restore and skip
        spawning an idle walker pool.
        """
        n = max(1, int(n))
        if n == self._n_walkers:
            return
        self._n_walkers = n
        if self._pucv_pool is not None:
            return
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
        self._last_gumbel_action_idx = None
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
        *,
        allowed_root_indices: set[int] | None = None,
    ) -> None:
        """Emit one or more ``info`` lines, one per top-ranked PV.

        When ``self._multi_pv`` == 1 the ``multipv`` field is omitted
        (classic single-PV UCI). When > 1 each line gets ``multipv 1..N``.
        When ``self._show_wdl`` is on and the root NN eval has been cached,
        each line also gets a ``wdl W D L`` field derived from its Q plus
        the root's NN draw-rate estimate.
        """
        assert self._tree is not None
        assert self._root_id is not None
        lines = _multipv_lines(
            self._tree,
            self._root_id,
            self._multi_pv,
            root_q,
            allowed_root_indices=allowed_root_indices,
        )
        if not lines:
            return
        emit_multipv = self._multi_pv > 1
        tbhits = tb_probe.hits if tb_probe is not None else 0
        draw_rate = (
            _root_draw_rate(self._root_wdl_logits[0])
            if self._show_wdl and self._root_wdl_logits is not None
            else 0.0
        )
        pucv_info = _pucv_stats_string(
            self.last_multi_gpu_pucv_stats(),
            pending_mode=self._pucv_vloss_mode,
        )
        if pucv_info is None:
            pucv_info = _single_pucv_cache_stats_string(
                self.last_single_pucv_cache_stats(),
                pending_mode=self._pucv_vloss_mode,
            )
        hashfull = self._hashfull_permille()
  # seldepth is the deepest PV any emitted line reaches, kept monotonic across
  # the search so it climbs like a real selective-depth counter.
        self._seldepth = max(
            self._seldepth, max((len(pv_idx) for _, _, pv_idx in lines), default=0),
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
                score_mate=None,
                multipv=rank if emit_multipv else None,
                wdl=wdl,
                string=pucv_info if rank == 1 else None,
                hashfull=hashfull,
                seldepth=self._seldepth,
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

        Reuse is also refused once the persistent tree grows past half the
        memory cap. Rerooting only repoints ``self._root_id`` — the C arena
        never reclaims the discarded sibling subtrees, so ``memory_bytes()``
        climbs monotonically across a game. Left unchecked it reaches
        ``_max_tree_bytes`` mid-game, after which every search halts after its
        first chunk (``stop_reason == "tree_bytes"``) with the clock still
        full. Forcing a rebuild here keeps each search bounded and fully
        budgeted, at the cost of dropping cross-move reuse near the cap.
        """
        if self._rpg_pool is not None:
  # Root-parallel Gumbel v1: no cross-move tree reuse — fresh per-candidate
  # arenas each move (design §2 "Tree reuse across moves", v1 scope).
            return False
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
        if self._max_tree_bytes > 0 and self._tree.memory_bytes() >= self._max_tree_bytes // 2:
            return False
        self._root_id = rid
        self._tree_fen = b.fen()
        self._invalidate_root_caches()
        return True

    def _try_tb_shortcut(
        self,
        board: chess.Board,
        tb_probe,
        deadline: Deadline,
        info_cb: InfoCallback | None,
        *,
        include_ponder: bool,
    ) -> SearchResult | None:
        """If TB knows the answer at root, return DTZ-optimal move directly.

        MCTS picks by visits — in a TB-win with Q=1.0 everywhere that reduces
        to "most-popular NN prior" and yields valid but DTZ-suboptimal play.
        Bypassing MCTS here is a correctness win, not just a speed win.
        """
        if tb_probe is None:
            return None
        short = _try_tb_root_bestmove(board, tb_probe, include_ponder=include_ponder)
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
  # TB shortcut bypasses MCTS, so there is no tree to report fill for.
                hashfull=None,
                seldepth=len(short.pv),
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
        xs = np.empty(
            (1, input_plane_count(self._cfg.input_extra_features), 8, 8),
            dtype=np.float32,
        )
        root_cb = CBoard.from_board(board)
        xs[0] = encode_cboard(
            root_cb,
            input_history_encoding=self._cfg.input_history_encoding,
            input_extra_features=self._cfg.input_extra_features,
        )
        if self._cfg.compute_relations:
            rels = root_cb.compute_relations()[None, ...]
            pol, wdl = self._evaluator.evaluate_encoded(xs, relations=rels)
        else:
            pol, wdl = self._evaluator.evaluate_encoded(xs)
        pol_np = np.asarray(pol, dtype=np.float32)
        wdl_np = np.asarray(wdl, dtype=np.float32).copy()
  # Probe at root so score_cp reflects TB truth on the very first chunk's
  # info emission, before MCTS has back-propagated it.
        if tb_probe is not None:
            tb_probe.apply([root_cb], wdl_np)
        self._root_pol_logits = pol_np
        self._root_wdl_logits = wdl_np

    def _pre_expand_root_for_pool(
        self,
        board: chess.Board,
        allowed_root_indices: set[int] | None,
    ) -> None:
        """Pool paths race on the root's first descent so it must be expanded
        upfront. The classic gumbel path does this internally."""
        if allowed_root_indices is not None:
            return
        if self._rpg_pool is not None:
            self._ensure_rpg_root_prepared(board)
        elif self._pucv_pool is not None:
            self._ensure_pucv_pool_root_expanded(board)
        elif self._walker_pool is not None:
            self._ensure_walker_root_expanded(board)
        elif self._pucv is not None:
            self._ensure_pucv_root_expanded(board)

    def _run_one_chunk(
        self,
        chunk: int,
        board: chess.Board,
        stop_event: threading.Event,
        tb_probe,
        allowed_root_indices: set[int] | None,
        allow_terminal_shortcuts: bool,
    ) -> tuple[float, int]:
        """Returns ``(root_value, completed_sims)``.

        Multi-GPU pool and walker pool report real completed counts (either
        can stop mid-chunk on ``stop_event``). Single-thread PUCV and the
        gumbel C path run the full requested chunk (no early stop), so they
        count ``chunk``.
        """
        if allowed_root_indices is not None:
            # `searchmoves` restriction is only threaded through the single-walker
            # gumbel C path (via allowed_root_indices_batch). The pool / pucv / walker
            # paths have no such filter, so routing here keeps `searchmoves` correct
            # even at Threads>1 — at the cost of single-threaded search for that move.
            # Do NOT "optimize" this back onto the pools without first teaching them to
            # honor allowed_root_indices, or the restriction is silently ignored.
            return self._run_gumbel_chunk(
                chunk, board, tb_probe, allowed_root_indices,
                allow_terminal_shortcuts=allow_terminal_shortcuts,
            ), chunk
        if self._rpg_pool is not None:
            return self._run_rpg_chunk(chunk, stop_event)
        if self._pucv_pool is not None:
            return self._run_pucv_pool_chunk(chunk, stop_event)
        if self._walker_pool is not None:
            return self._run_walker_chunk(chunk, stop_event)
        if self._pucv is not None:
            return self._run_pucv_chunk(chunk), chunk
        return self._run_gumbel_chunk(
            chunk, board, tb_probe, allowed_root_indices,
            allow_terminal_shortcuts=allow_terminal_shortcuts,
        ), chunk

    def _maybe_emit_pv_info(
        self,
        *,
        board: chess.Board, deadline: Deadline,
        last_value: float, total_nodes: int,
        info_cb: InfoCallback | None, max_depth: int | None,
        last_info_ms: int, tb_probe,
        allowed_root_indices: set[int] | None,
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
        assert self._tree is not None
        assert self._root_id is not None
        _, pv_indices = _best_move_and_pv(
            self._tree,
            self._root_id,
            allowed_root_indices=allowed_root_indices,
        )
        if info_due:
            assert info_cb is not None
            self._emit_pv_info(
                info_cb, board, float(last_value), total_nodes, elapsed, tb_probe,
                allowed_root_indices=allowed_root_indices,
            )
            last_info_ms = elapsed
        return pv_indices, last_info_ms, elapsed

    def _build_final_search_result(
        self, *,
        board: chess.Board,
        total_nodes: int,
        last_value: float,
        tb_probe,
        allowed_root_indices: set[int] | None = None,
        include_ponder: bool = False,
    ) -> SearchResult:
        """Final snapshot of the searched tree → SearchResult."""
        assert self._tree is not None
        assert self._root_id is not None
        bestmove_idx, pv_indices = _best_move_and_pv(
            self._tree, self._root_id,
            allowed_root_indices=allowed_root_indices,
        )
        if (
            self._last_gumbel_action_idx is not None
            and (
                allowed_root_indices is None
                or int(self._last_gumbel_action_idx) in allowed_root_indices
            )
        ):
            gumbel_pv = _pv_from_root_action(
                self._tree,
                self._root_id,
                int(self._last_gumbel_action_idx),
            )
            if gumbel_pv:
                bestmove_idx = int(self._last_gumbel_action_idx)
                pv_indices = gumbel_pv
        bestmove = _index_to_uci(board, bestmove_idx)
        ponder = None
        if include_ponder:
            # Ponder must be aligned with the actual bestmove, not with the
            # most-visited root child (they can differ in Gumbel paths).
            if len(pv_indices) >= 2:
                ponder_idx = int(pv_indices[1])
            else:
                ponder_idx = _reply_at_child(
                    self._tree, self._root_id, int(bestmove_idx),
                )
        else:
            ponder_idx = None
        if include_ponder and ponder_idx is not None:
            try:
                after_bestmove = _board_after(board, bestmove_idx)
                if after_bestmove is not None:
                    ponder = _index_to_uci(after_bestmove, ponder_idx)
            except Exception:
                ponder = None
        return SearchResult(
            bestmove_uci=bestmove,
            ponder_uci=ponder,
            nodes=total_nodes,
            pv=_uci_pv(board, pv_indices),
            score_cp=q_to_cp(0.5 * (last_value + 1.0)),
            tbhits=tb_probe.hits if tb_probe is not None else 0,
            score_mate=None,
        )

    def _should_stop_search(
        self,
        *,
        stop_event: threading.Event, deadline: Deadline,
        max_nodes: int | None, max_depth: int | None,
        total_nodes: int, pv_len: int,
        time_margin_ms: float = 0.0,
    ) -> str | None:
        """Return the first-firing stop reason, or ``None`` to keep searching.

        Reasons are checked in priority order so the caller can match the
        original sequential-break semantics — ``"tree_bytes"`` only fires when
        no earlier budget/external/depth limit was already hit, which matters
        for the tree-bytes-only final info-cb emission in :meth:`run`.

        ``time_margin_ms`` (clock games only) stops one worst-case batch BEFORE
        the hard deadline so an un-interruptible chunk can't overrun it — the
        batch-time-variance guard (see ``_batch_margin_ms``). It ranks with the
        hard-deadline "external" stop, ahead of the work bounds.
        """
        if stop_event.is_set() or deadline.expired():
            return "external"
        if time_margin_ms > 0.0:
            rem = deadline.remaining_ms()
            if rem is not None and rem < time_margin_ms:
                return "time_margin"
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

    def _is_shared_tree_path(self, allowed_root_indices: set[int] | None) -> bool:
        """True when the active chunk path mutates the shared tree from multiple
        threads (walker pool, multi-GPU pucv pool, or the root-parallel Gumbel
        pool — its groups own disjoint candidate subtrees but still allocate
        nodes from the one shared arena). ``searchmoves`` forces the
        single-thread gumbel path, so it is never shared-tree."""
        if allowed_root_indices is not None:
            return False
        return (
            self._pucv_pool is not None
            or self._walker_pool is not None
            or self._rpg_pool is not None
        )

    def _chunk_budget(self, allowed_root_indices: set[int] | None) -> int:
        """Sims to request from the active path for one chunk.

        The multi-GPU pucv pool splits a single ``run()``'s budget across all
        GPU workers via one shared semaphore. Each worker greedily grabs up to
        ``gather`` tokens, so the per-device share must be at least ``gather``;
        otherwise (e.g. ``VLGather`` > ``--chunk-sims``) a few workers drain the
        whole budget and the rest idle — the very starvation this scaling fixes.
        Use ``max(chunk_sims, gather)`` per device. The pool still polls
        ``stop_event`` between batches, so stop latency stays ~one batch rather
        than ~one (now larger) chunk.
        """
        if allowed_root_indices is None and self._pucv_pool is not None:
            per_device = max(self._chunk_sims, self._pucv_pool.gather)
            return per_device * max(1, self._pucv_pool.n_devices)
        return self._chunk_sims

    def _time_capped_chunk(
        self, chunk: int, deadline: Deadline, total_nodes: int,
    ) -> int:
        """Shrink ``chunk`` to the sims that fit in the remaining deadline.

        The hard deadline is only checked between chunks and the pucv pool isn't
        handed the deadline, so a big scaled multi-GPU chunk could run past the
        time budget. Once an nps estimate exists (after the first chunk), cap the
        chunk at ``nps * remaining_ms``; floor it at ``_chunk_sims`` so the base
        single-GPU granularity (whose overrun is ~one base chunk = negligible) is
        never reduced. No-op for open-ended (no deadline) searches."""
        remaining_ms = deadline.remaining_ms()
        if remaining_ms is None:
            return chunk
        if total_nodes <= 0:
  # First chunk: no nps estimate yet. A scaled multi-GPU chunk (per_device *
  # n_devices) could run unbounded past the deadline before the between-chunks
  # check — a time forfeit on move 1. Bound it to the base single-GPU granularity,
  # AND to a conservative bootstrap-nps time cap so a tight budget (e.g. 100ms
  # matched_time, where one full _chunk_sims=2048 chunk would itself overrun)
  # shrinks the first chunk. A normal/long budget leaves it at _chunk_sims; the
  # measured nps then caps every later chunk.
            boot_cap = int(_BOOTSTRAP_NPS_PER_MS * remaining_ms)
            return min(chunk, self._chunk_sims, max(_MIN_FIRST_CHUNK, boot_cap))
        elapsed = deadline.elapsed_ms()
        if elapsed <= 0:
            return chunk
        nps = total_nodes / elapsed
        time_cap = int(nps * remaining_ms)
        return min(chunk, max(self._chunk_sims, time_cap))

    def _ensure_shared_tree_headroom(self, upcoming_sims: int) -> int:
        """Pre-grow the shared tree for the next concurrent chunk and return the
        sim budget that actually fits under the Hash cap (0 = stop the search).

        Reserving the chunk's worst-case growth between chunks is what stops a
        ``tree_grow_*`` realloc mid-descent (use-after-free; see
        ``_TREE_MAX_LEAF_BRANCHING``). Two bounds:

        - Slots already reserved but unused (``node_capacity - node_count``) can
          be filled without any realloc and without changing ``memory_bytes()``,
          so they are always free to use — even when the tree already sits at the
          Hash cap. This is what keeps a small ``Hash`` from starving the very
          first chunk to an unvisited root.
        - Growing *beyond* the reservation raises ``memory_bytes()``, so that part
          is capped by the remaining Hash budget. Per-node cost is sized from
          *capacity* (``memory_bytes()`` is capacity-based); dividing by the live
          ``node_count`` would over-estimate hugely right after the initial
          reserve. ``reserve()`` still rounds capacity up by doubling, so the cap
          stays *soft*, but we never grow past it on purpose."""
        tree = self._tree
        if tree is None:
            return int(upcoming_sims)
        sims = max(0, int(upcoming_sims))
        node_count = tree.node_count()
        node_cap = tree.node_capacity()
        free = max(0, node_cap - node_count)
  # One sim expands one leaf into up to _TREE_MAX_LEAF_BRANCHING nodes (256 >
  # 218, the absolute max legal moves), so a chunk adds at most this many nodes.
        need = sims * _TREE_MAX_LEAF_BRANCHING
        if self._max_tree_bytes > 0 and need > free:
            used = int(tree.memory_bytes())
            remaining = self._max_tree_bytes - used
            bytes_per_cap_node = max(1, used // max(1, node_cap))
            growable = int(remaining // bytes_per_cap_node) if remaining > 0 else 0
  # Usable slots = already-reserved free slots (no memory growth) + what we can
  # still grow into under the cap. Free slots are usable even at/over the cap.
            budget = free + max(0, growable)
            sims = min(sims, budget // _TREE_MAX_LEAF_BRANCHING)
            if sims <= 0:
                return 0
            need = sims * _TREE_MAX_LEAF_BRANCHING
  # child edges grow 1:1 with nodes (#edges == #nodes-1), so the same target
  # bounds the child pool too. No-op when `need` already fits in `free`; the
  # doubling allocator supplies the slack that keeps re-reserves infrequent.
        tree.reserve(node_count + need, node_count + need)
        return sims

    def _filtered_root_visits(
        self, allowed_root_indices: set[int] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Root child ``(actions, visits)``, restricted to ``allowed_root_indices``
        when a ``searchmoves`` filter is active. Empty arrays when there is no
        tree/root or nothing survives the filter. One C read + one filter, shared
        by the best-action and visit-margin helpers (polled once per chunk)."""
        if self._tree is None or self._root_id is None:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty
        actions, visits = self._tree.get_children_visits(self._root_id)
        if actions.size == 0 or allowed_root_indices is None:
            return actions, visits
        keep = np.isin(actions, np.fromiter(allowed_root_indices, dtype=np.int32))
        return actions[keep], visits[keep]

    def _emitted_action(
        self,
        actions: np.ndarray,
        visits: np.ndarray,
        allowed_root_indices: set[int] | None,
    ) -> int:
        """The action final bestmove selection would emit given pre-fetched root
        ``(actions, visits)``, or -1 if none.

        Mirrors ``_build_final_search_result``: the Gumbel path plays the
        sequential-halving survivor (``_last_gumbel_action_idx``) when it is a
        legal root child, not the visit leader; other paths (walker / pucv pool)
        leave the survivor unset and select by visits."""
        if actions.size == 0:
            return -1
        gi = self._last_gumbel_action_idx
        if (
            gi is not None
            and self._tree is not None
            and self._root_id is not None
            and (allowed_root_indices is None or int(gi) in allowed_root_indices)
            and self._tree.find_child(self._root_id, int(gi)) != -1
        ):
            return int(gi)
        return int(actions[int(np.argmax(visits))])

    def _current_best_root_action(
        self, allowed_root_indices: set[int] | None,
    ) -> int:
        """The action final bestmove selection would emit right now, or -1 if
        none. Cheap (no full PV walk) so it can be polled once per chunk."""
        actions, visits = self._filtered_root_visits(allowed_root_indices)
        return self._emitted_action(actions, visits, allowed_root_indices)

    def _root_visit_lead(
        self, allowed_root_indices: set[int] | None,
    ) -> tuple[int, int]:
        """``(emitted_action, lead)`` where ``emitted_action`` is the move final
        selection would play right now and ``lead`` is its visit count minus the
        runner-up's. ``lead`` can be negative when the played move (a Gumbel
        survivor) is not the visit leader — the abort treats that as "not
        decided" and keeps searching. ``(-1, 0)`` when there is no move."""
        actions, visits = self._filtered_root_visits(allowed_root_indices)
        best = self._emitted_action(actions, visits, allowed_root_indices)
        if best < 0:
            return -1, 0
        best_mask = actions == best
        if not bool(best_mask.any()):
            return -1, 0
        best_visits = int(visits[best_mask].max())
        others = visits[~best_mask]
        runner_up = int(others.max()) if others.size else 0
        return best, best_visits - runner_up

    def _move_is_decided(
        self, best_action: int, allowed_root_indices: set[int] | None,
    ) -> bool:
        """Complexity gate deciding *easy vs hard*: the played move is "decided"
        (safe to bank) only when it is BOTH stable — the same move for
        ``_ABORT_MIN_STABLE_CHUNKS`` consecutive chunks — AND visit-decisive — its
        share of the root visits leads the best alternative by ``_ABORT_VISIT_GAP_MARGIN``.
        A still-flipping choice or a near-tie in visit share reads as a hard position,
        so the search keeps going and uses the (over-allocated) budget. Must be called
        once per chunk: it advances the stability counter."""
        if best_action == self._abort_last_best:
            self._abort_stable_chunks += 1
        else:
            self._abort_last_best = best_action
            self._abort_stable_chunks = 0
        if self._abort_stable_chunks < _ABORT_MIN_STABLE_CHUNKS:
            return False
        actions, visits = self._filtered_root_visits(allowed_root_indices)
        if actions.size < 2:
            return True
        return _visit_gap(actions, visits, best_action) >= _ABORT_VISIT_GAP_MARGIN

    def _abort_ready(
        self,
        optimum_ms: int | None,
        deadline: Deadline,
        total_nodes: int,
        allowed_root_indices: set[int] | None,
        abort_factor: float,
    ) -> bool:
        """Lc0-style early abort with a complexity gate. Banks the clock only when
        the move final selection would play is *settled* on three signals:

        - it leads the runner-up on visits (``_root_visit_lead``);
        - it is visit-decisive and stable (``_move_is_decided`` — visit-share gap +
          best-move stability across chunks); and
        - the optimum budget is spent, OR (before it) the visit lead is already
          insurmountable within the remaining optimum (``lead > factor * sims``).

        A hard position (still-flipping choice, near-tie in visit share, or trailing
        survivor) fails the gate and keeps searching toward the hard ``deadline``
        — "use it all" on the moves that matter, which is what makes the aggressive
        over-allocation pay off. Returns False when no optimum is set, so
        node/movetime/infinite searches and the hard ``deadline`` are unaffected."""
        if optimum_ms is None:
            return False
        best, lead = self._root_visit_lead(allowed_root_indices)
        if best < 0:
  # No legal/root move: nothing to bank, keep searching toward the hard deadline.
            return False
  # A forced position (exactly one legal/root move) is decided even with a zero
  # visit lead: the C Gumbel path returns the sole action WITHOUT visiting its
  # child, so `lead` is trivially 0 — bank it instead of spending the full
  # deadline. For the multi-move case a non-positive lead means the survivor isn't
  # the visit leader (hard/unsettled), so keep searching.
        if lead <= 0:
            actions, _ = self._filtered_root_visits(allowed_root_indices)
            if actions.size != 1:
  # Unsettled (the survivor isn't the visit leader). _move_is_decided — the only
  # updater of the per-chunk stability streak — is skipped on this early return, so
  # reset the streak here: a flicker (lead>0, then lead<=0, then lead>0 again) must
  # NOT count as consecutive stable chunks toward the stability-gated abort below.
                self._abort_stable_chunks = 0
                return False
        elapsed = deadline.elapsed_ms()
        remaining_ms = optimum_ms - elapsed
  # Provable bank: a visit lead the runner-up cannot overtake within the remaining
  # optimum (lead > factor * projected remaining sims) is decisive on its own — the
  # played move can't change — so bank it even before the optimum and regardless of
  # the stability streak.
        if remaining_ms > 0:
            nps = total_nodes / max(1, elapsed)
            if lead > abort_factor * nps * remaining_ms:
                return True
  # Not (yet) provable: require the complexity gate — visit-share gap + consecutive
  # best-move stability — and only bank once the optimum budget is actually spent.
        if not self._move_is_decided(best, allowed_root_indices):
            return False
        return remaining_ms <= 0

    def run(
        self,
        board: chess.Board,
        *,
        stop_event: threading.Event,
        deadline: Deadline,
        max_nodes: int | None,
        max_depth: int | None = None,
        optimum_ms: int | None = None,
        abort_factor: float = _DEFAULT_ABORT_FACTOR,
        root_moves: tuple[str, ...] = (),
        info_cb: InfoCallback | None = None,
        include_ponder: bool = False,
        allow_terminal_shortcuts: bool = True,
        on_chunk: Callable[[int], None] | None = None,
    ) -> SearchResult:
        """Search until any of: stop_event set, deadline expired, max_nodes hit,
        PV length ≥ max_depth, or (``optimum_ms`` set) the visit-margin abort
        fires (see ``_abort_ready``).

        Returns when at least one chunk has run (so bestmove is always
        backed by MCTS data, never a raw priors pick).
        """
        fen = board.fen()
        if self._tree is None or self._tree_fen != fen:
            self._tree = None
            self._root_id = None
            self._tree_fen = fen
            self._invalidate_root_caches()
  # Fresh best-move stability tracking for this move's abort gate (tree may be
  # reused across the game, so reset here rather than only on a new position).
        self._abort_last_best = -1
        self._abort_stable_chunks = 0
  # Fresh chunk-duration stats for this move's batch-time clock margin.
        self._reset_batch_timing()

        tb_probe = self._tb_probe
        if tb_probe is not None:
            tb_probe.reset_counts()

        allowed_root_indices = _allowed_root_indices(board, root_moves)

        if allow_terminal_shortcuts:
            mate = _try_immediate_checkmate(
                board, allowed_root_indices=allowed_root_indices,
            )
            if mate is not None:
                return mate

        short = (
            None
            if allowed_root_indices is not None
            else self._try_tb_shortcut(
                board,
                tb_probe,
                deadline,
                info_cb,
                include_ponder=include_ponder,
            )
        )
        if short is not None:
            return short

        if allowed_root_indices is not None and not allowed_root_indices:
            return SearchResult(
                bestmove_uci="0000",
                ponder_uci=None,
                nodes=0,
                pv=(),
                score_cp=0,
                tbhits=tb_probe.hits if tb_probe is not None else 0,
            )

        self._last_gumbel_action_idx = None
        self._seldepth = 0
        self._ensure_root_eval_cached(board, tb_probe)
        self._pre_expand_root_for_pool(board, allowed_root_indices)

        total_nodes = 0
        last_info_ms = -1
        last_value = 0.0
        pv_indices: list[int] = []
        elapsed = 0
  # Starting budget for this move — caps the batch-time margin so a bad early
  # estimate can't reserve away the whole move (see `_MAX_BATCH_MARGIN_FRACTION`).
        start_remaining_ms = deadline.remaining_ms()
        while True:
            chunk = self._chunk_budget(allowed_root_indices)
            if max_nodes is not None:
                remaining = max_nodes - total_nodes
                if remaining <= 0:
                    break
                chunk = min(chunk, remaining)
  # The deadline is only checked *between* chunks, and the pucv pool receives
  # stop_event but not the deadline, so a large scaled multi-GPU chunk could
  # otherwise run well past the time budget. Cap it to the sims that fit in the
  # remaining time (estimated from the running nps), but never below the base
  # chunk so single-GPU stop latency is unchanged.
            chunk = self._time_capped_chunk(chunk, deadline, total_nodes)

  # Shared-tree (walker / multi-GPU pucv) chunks descend lock-free from N
  # threads; grow the arena now, while quiescent, so the chunk can't realloc
  # mid-descent (use-after-free). The pre-grow is bounded by the Hash cap and
  # returns the sims that fit; 0 means we're at the cap, so stop (same final
  # info as the tree_bytes stop) rather than grow past the user's setting.
            if self._is_shared_tree_path(allowed_root_indices):
                chunk = self._ensure_shared_tree_headroom(chunk)
                if chunk <= 0:
                    if info_cb is not None and self._root_id is not None:
                        self._emit_pv_info(
                            info_cb, board, float(last_value), total_nodes,
                            deadline.elapsed_ms(), tb_probe,
                            allowed_root_indices=allowed_root_indices,
                        )
                    break

            _chunk_t0 = time.monotonic()
            last_value, completed = self._run_one_chunk(
                chunk, board, stop_event, tb_probe, allowed_root_indices,
                allow_terminal_shortcuts=allow_terminal_shortcuts,
            )
  # Time the (un-interruptible) chunk so the batch-time clock margin below can
  # reserve one worst-case batch before the hard deadline.
            self._record_batch_ms((time.monotonic() - _chunk_t0) * 1000.0)
  # Count the sims actually completed, not the requested chunk: a pool chunk
  # cut short by stop_event would otherwise inflate reported nodes AND the nps
  # estimate `_time_capped_chunk` sizes later chunks with.
            total_nodes += int(completed)
  # Per-chunk instrumentation hook (offline analysis of the accumulating tree —
  # the states the abort actually decides between). Fired after each chunk with
  # the cumulative node count; the callback reads worker root state itself.
            if on_chunk is not None:
                on_chunk(total_nodes)

            pv_indices, last_info_ms, elapsed = self._maybe_emit_pv_info(
                board=board, deadline=deadline,
                last_value=last_value, total_nodes=total_nodes,
                info_cb=info_cb, max_depth=max_depth,
                last_info_ms=last_info_ms, tb_probe=tb_probe,
                allowed_root_indices=allowed_root_indices,
            )

  # Batch-time-variance clock margin: clock games only, capped so a bad early
  # estimate can't reserve away the whole move (see `_clock_time_margin_ms`).
            time_margin_ms = self._clock_time_margin_ms(optimum_ms, start_remaining_ms)
            stop_reason = self._should_stop_search(
                stop_event=stop_event, deadline=deadline,
                max_nodes=max_nodes, max_depth=max_depth,
                total_nodes=total_nodes, pv_len=len(pv_indices),
                time_margin_ms=time_margin_ms,
            )
            if stop_reason is not None:
  # Tree-bytes-only stops emit a final info line; other stop reasons
  # either already emitted via _maybe_emit_pv_info above or want quiet exit.
                if stop_reason == "tree_bytes" and info_cb is not None:
                    assert self._root_id is not None
                    self._emit_pv_info(
                        info_cb, board, float(last_value),
                        total_nodes, elapsed, tb_probe,
                        allowed_root_indices=allowed_root_indices,
                    )
                break

  # Visit-margin abort: bank the clock once the move we would play is decided
  # for the remaining optimum budget. Checked after the hard stop reasons so
  # the deadline / node / depth bounds always win, and a no-op when no optimum
  # is set (node / movetime / infinite searches).
            if self._abort_ready(
                optimum_ms, deadline, total_nodes, allowed_root_indices, abort_factor,
            ):
                break

        return self._build_final_search_result(
            board=board, total_nodes=total_nodes,
            last_value=last_value, tb_probe=tb_probe,
            allowed_root_indices=allowed_root_indices,
            include_ponder=include_ponder,
        )

    def _run_walker_chunk(
        self, chunk: int, stop_event: threading.Event,
    ) -> tuple[float, int]:
        assert self._tree is not None
        assert self._root_id is not None
        assert self._walker_pool is not None
        assert self._walker_cboard is not None
        completed = self._walker_pool.run(
            tree=self._tree,
            root_id=self._root_id,
            root_cboard=self._walker_cboard,
            target_sims=chunk,
            stop_event=stop_event,
        )
        return self._tree.node_q(self._root_id), int(completed)

    def _run_gumbel_chunk(
        self,
        chunk: int,
        board: chess.Board,
        tb_probe,
        allowed_root_indices: set[int] | None = None,
        allow_terminal_shortcuts: bool = True,
    ) -> float:
        gumbel_result = run_gumbel_root_many_c(
            model=None,
            boards=[board],
            device=self._device,
            rng=self._rng,
            cfg=dataclasses.replace(
                self._cfg,
                simulations=chunk,
                add_noise=False,
                temperature=0.0,
            ),
            evaluator=self._evaluator,
            pre_pol_logits=self._root_pol_logits,
            pre_wdl_logits=self._root_wdl_logits,
            tree=self._tree,
            root_node_ids=[self._root_id] if self._root_id is not None else None,
            allowed_root_indices_batch=[allowed_root_indices],
            allow_terminal_root_shortcuts=allow_terminal_shortcuts,
            tb_probe=tb_probe,
            pre_wdl_logits_tb_probed=True,
            target_batch=self._minibatch_size,
        )
        _, actions, values, _, tree, root_ids = gumbel_result[:6]
        self._tree = tree
        self._root_id = int(root_ids[0])
        self._last_gumbel_action_idx = int(actions[0])
        return float(values[0])

    def _run_pucv_pool_chunk(
        self, chunk: int, stop_event: threading.Event,
    ) -> tuple[float, int]:
        assert self._tree is not None
        assert self._root_id is not None
        assert self._pucv_pool is not None
        assert self._pucv_pool_cboard is not None
        completed = self._pucv_pool.run(
            tree=self._tree,
            root_id=self._root_id,
            root_cboard=self._pucv_pool_cboard,
            target_sims=chunk,
            stop_event=stop_event,
        )
        return self._tree.node_q(self._root_id), int(completed)

    def _run_rpg_chunk(
        self, chunk: int, stop_event: threading.Event,
    ) -> float:
        """One root-parallel Gumbel chunk. The pool runs a full sequential-
        halving schedule over ``chunk`` sims (identical budget shape to the
        classic single-GPU chunk — the per-chunk budget is deliberately NOT
        scaled by group count, so root decisions at equal total sims match
        serial Gumbel) and returns the survivor + its completed value; the
        survivor feeds ``_last_gumbel_action_idx`` exactly like the classic
        Gumbel chunk so final selection semantics are unchanged."""
        assert self._rpg_pool is not None
        value, action = self._rpg_pool.run(
            target_sims=chunk, stop_event=stop_event,
        )
        if action >= 0:
            self._last_gumbel_action_idx = int(action)
        return float(value)

    def _ensure_rpg_root_prepared(self, board: chess.Board) -> None:
        """Root prep for the root-parallel Gumbel pool: create the shared
        tree, then let the pool install + expand the root (classic-Gumbel
        priors from the cached root eval) and sample-ready state. Mirrors
        ``_ensure_pucv_pool_root_expanded``'s contract; ``prepare_root`` is
        idempotent per (tree, position)."""
        assert self._rpg_pool is not None
        assert self._root_pol_logits is not None
        assert self._root_wdl_logits is not None
        if self._tree is None:
            self._tree = MCTSTree()
            self._tree.reserve(50_000, 500_000)
            self._root_id = None
        self._root_id = self._rpg_pool.prepare_root(
            tree=self._tree,
            board=board,
            pol_logits=self._root_pol_logits[0],
            wdl_logits=self._root_wdl_logits[0],
        )

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
        assert self._tree is not None
        assert self._root_id is not None
        assert self._pucv is not None
        assert self._pucv_cboard is not None
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
        round(w * 1000),
        round(d * 1000),
        round(l * 1000),
    )


def _multipv_lines(
    tree: MCTSTree,
    root_id: int,
    n: int,
    root_q_default: float,
    allowed_root_indices: set[int] | None = None,
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
    if allowed_root_indices is not None:
        keep = np.isin(actions, np.fromiter(allowed_root_indices, dtype=np.int32))
        actions = actions[keep]
        visits = visits[keep]
        qs = qs[keep]
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




def _allowed_root_indices(board: chess.Board, root_moves: tuple[str, ...]) -> set[int] | None:
    """Convert UCI ``searchmoves`` to legal root policy indices."""
    if not root_moves:
        return None
    allowed: set[int] = set()
    for uci in root_moves:
        try:
            move = chess.Move.from_uci(str(uci))
        except ValueError:
            continue
        if move not in board.legal_moves:
            continue
        allowed.add(int(move_to_index(move, board)))
    return allowed


def _try_tb_root_bestmove(
    board: chess.Board, tb_probe: SyzygyProbe, *, include_ponder: bool,
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

    ponder_uci = None
    if include_ponder:
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


def _try_immediate_checkmate(
    board: chess.Board,
    *,
    allowed_root_indices: set[int] | None = None,
) -> SearchResult | None:
    """Return an immediate mating move before root candidate pruning.

    Gumbel search only expands its top-k root candidates. If the policy puts
    a mate-in-1 outside that set, the tree may never discover it and can even
    prefer a high-prior terminal draw. A legal mate is tactically decisive and
    cheap to detect directly at the root.
    """
    mate = immediate_mate_move(board, allowed_root_indices=allowed_root_indices)
    if mate is None:
        return None
    move, _ = mate
    return SearchResult(
        bestmove_uci=move.uci(),
        ponder_uci=None,
        nodes=1,
        pv=(move.uci(),),
        score_cp=q_to_cp(1.0),
        tbhits=0,
        score_mate=1,
    )


def _best_move_and_pv(
    tree: MCTSTree,
    root_id: int,
    allowed_root_indices: set[int] | None = None,
) -> tuple[int, list[int]]:
    """Most-visited root move and its PV. ``q`` is unused at the call sites
    that take this shape (so the third tuple element from ``_multipv_lines``
    is dropped here)."""
    lines = _multipv_lines(
        tree,
        root_id,
        root_q_default=0.0,
        n=1,
        allowed_root_indices=allowed_root_indices,
    )
    if not lines:
        return -1, []
    _, _, pv = lines[0]
    return pv[0], pv


def _pv_from_root_action(
    tree: MCTSTree,
    root_id: int,
    action_idx: int,
) -> list[int]:
    """Build a PV starting from a known root action.

    Classic UCI Gumbel returns a sequential-halving survivor from the C search
    call. That action is not guaranteed to be the most-visited root child, so
    final bestmove must preserve it instead of rebuilding from visit counts.
    After the root action, the continuation can still follow most visits.
    """
    cid = tree.find_child(root_id, int(action_idx))
    if cid == -1:
        return []
    pv = [int(action_idx)]
    while cid != -1:
        a, vs = tree.get_children_visits(cid)
        if a.size == 0:
            break
        nxt = int(a[int(np.argmax(vs))])
        pv.append(nxt)
        cid = tree.find_child(cid, nxt)
    return pv


def _reply_at_child(
    tree: MCTSTree,
    root_id: int,
    action_idx: int,
) -> int | None:
    """Most-visited child at the node reached by ``action_idx`` from root.

    This is the opponent's predicted reply after OUR specific move,
    not after the globally most-visited root move.
    """
    cid = tree.find_child(root_id, int(action_idx))
    if cid == -1:
        return None
    a, vs = tree.get_children_visits(cid)
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


def _board_after(board: chess.Board, idx: int) -> chess.Board | None:
    b = board.copy(stack=False)
    try:
        move = index_to_move(int(idx), board)
    except Exception:
        return None
    if move not in b.legal_moves:
        return None
    b.push(move)
    return b
