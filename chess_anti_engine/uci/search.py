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
from typing import Any, Protocol, cast

import chess
import numpy as np

from chess_anti_engine.async_safety import note_cleanup
from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.inference import BatchEvaluator
from chess_anti_engine.mcts import _mcts_tree as _mcts_tree_ext
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_TARGET_BATCH,
    PLAY_SEARCH_VLOSS_WEIGHT,
    GumbelConfig,
    _policy_logits_to_full,
)
from chess_anti_engine.mcts.gumbel_c import (
    _REQUIRED_MCTS_ABI,
    _ROOT_SUPPORT_EQUAL,
    _ROOT_SUPPORT_MISSING_ACTION,
    _classify_expanded_root_support,
    SolvedRootWitnessUnavailable,
    _SOLVED_ROOT_TB_CHECK_LIMIT,
    run_gumbel_root_many_c,
)
from chess_anti_engine.mcts.search_options import MAX_CHUNK_SIMS, SEARCH_OPTIONS
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
    resolve_rpg_schedule_knobs,
)
from chess_anti_engine.moves import index_to_move, move_to_index
from chess_anti_engine.tablebase import SyzygyProbe, try_tb_root_move

from .score import q_to_cp
from .time_manager import Deadline
from .walker_pool import WalkerPool, WalkerPoolConfig
import contextlib
import logging as _logging
import time

_log = _logging.getLogger(__name__)

# _REQUIRED_MCTS_ABI is imported from gumbel_c (canonical owner of the C-path ABI
# contract), so the SearchWorker construction guard and the run_gumbel_root_many_c
# entry guard agree on one value. `_classify_expanded_root_support` and its two
# verdicts come from the same module for the same reason: the pool-side root
# guard below must decide "does this expanded root's child set match the support
# I am about to search?" by the SAME predicate the C Gumbel path uses, or the two
# can disagree about a root they both look at.

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

# First-chunk size for a time-managed search of a forced root (one legal or
# `searchmoves` move); the visit-margin abort banks the move after that chunk.
# Only the walker / pool paths simulate a forced root at all (the classic Gumbel
# path returns the sole action without a simulation); this keeps them from
# spending a full `chunk_sims` chunk on a decided move. `run` emits the info
# line (score from the root evaluation / the visited child, pv = the move).
_FORCED_ROOT_CHUNK_SIMS = 64
# Back-off after a chunk that completed no simulation (see `run`).
_IDLE_CHUNK_WAIT_S = 0.01

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
# internal growth the mid-chunk realloc would have used, so node_cap /
# footprint_bytes() (and thus the Hash cap and advance_root reuse decision) are
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
# Hash-capped arena growth: one node-array growth adds at most
# min(Hash/64, 32 MiB) of node slots (MCTSTree.set_growth_step). The `tree_bytes`
# stop reserves the previous chunk's growth, so the tree FOOTPRINT ends within
# one growth step of Hash (measured: 1032.5 MB at Hash 1024, 259 MB at 256).
# Process RSS is higher than the footprint by memory Hash does not cover:
# per-search scratch (encode buffers, evaluator tensors, ~60 MB), the eval
# cache (EvalCacheEntries), and glibc heap slack from tree-array reallocs
# (measured RSS - footprint: ~+120 MB at Hash 64-1024 on CPU; malloc_trim
# returns all but ~20-30 MB of it, so it is slack, not live tree). Operators
# should provision about Hash + 150 MB + the eval cache. _TREE_NODE_BYTES
# mirrors TREE_NODE_BYTES in _mcts_tree.c (pinned by a test against
# memory_stats()["node_bytes"]).
_TREE_GROWTH_STEP_DIV = 64
_TREE_NODE_BYTES = 51
_MIN_TREE_GROWTH_STEP_NODES = 4096
# ... and at most 32 MiB of node slots: growth zero-fills the new slots inside
# an un-interruptible chunk, measured at ~1 GB/s (16 MiB 8 ms, 32 MiB 20-46 ms,
# 64 MiB 33-53 ms, 1 GiB ~960 ms per step), so an uncapped Hash/64 step at a
# 64 GB Hash stalled the search ~1 s in a chunk the batch-time margin never saw.
_MAX_TREE_GROWTH_STEP_BYTES = 32 * 1024 * 1024
# Node and child ids are int32. Stop (reason "tree_ids") while this many slots
# remain, which covers one worst-case chunk of any path (2^25 nodes = 131k sims
# x 256) — the C pools refuse to grow past INT32_MAX - 1.
_TREE_ID_HEADROOM = 1 << 25
_TREE_NODE_ID_LIMIT = (1 << 31) - 1 - _TREE_ID_HEADROOM
assert MAX_CHUNK_SIMS * _TREE_MAX_LEAF_BRANCHING <= _TREE_ID_HEADROOM
# Deferred reuse decision (see `_resolve_pending_compaction`). Measured
# `compact_to_subtree` cost on CPU: 86-142 ms/GiB of arena (keep 8%-100%),
# up to ~300 ms/GiB in the independent review's runs; the estimate uses a
# conservative 400 ms/GiB.
_COMPACTION_MS_PER_GIB = 400.0
# With a move budget and a measured growth rate, keep / compact / rebuild is
# chosen by projected effective search time (see the resolver). Without a rate
# (the previous search ran fewer than `_REUSE_RATE_MIN_VISITS` visits) the
# fallback is: compact if the estimate fits in 5% of the budget, else keep the
# tree uncompacted below 3/4 of Hash and rebuild above that.
_REUSE_RATE_MIN_VISITS = 1024
_COMPACTION_BUDGET_FRACTION = 0.05
_KEEP_UNCOMPACTED_FRACTION = 0.75
# Ponder (before ponderhit) and `go infinite`: `stop` / `ponderhit` cannot
# interrupt the C compaction, so a ponder miss would delay `bestmove` — and the
# next `go`'s clock — by the whole compaction. Compact only while the estimate
# is at most this; otherwise use the band fallback above. (`go nodes` / `go
# depth` have no clock and are not capped; see the resolver.)
_STOP_SENSITIVE_COMPACTION_MAX_MS = 250.0
# Compaction allocates the new child arrays (2 x int32 per kept child) and a
# CBoard owner map (int32 per cached board) before freeing the old ones, so
# RSS peaks at footprint + that transient. Compaction is not chosen when the
# peak would exceed Hash by more than this (the transient is ~13% of the kept
# footprint on the walker path, ~2 GB at a 16 GB keep-all, so an unchecked
# compaction near the cap would break the "Hash + ~150 MB" provisioning rule).
_COMPACTION_TRANSIENT_MARGIN_BYTES = 32 * 1024 * 1024
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

# Where each PUCT path keeps the family it will actually descend on. All four
# snapshot the values at CONSTRUCTION, so these -- not the shared `GumbelConfig`
# a setter writes -- are the live objects `realized_search_values` must read.
# WalkerPoolConfig / MultiGpuPucvConfig / RootParallelGumbelConfig share the
# dataclass field names; PucvChunker keeps them in private attributes.
_PUCT_DESCENT_FIELDS: tuple[str, ...] = (
    "c_puct", "cpuct_factor", "cpuct_base", "fpu_reduction", "vloss_weight",
)
_PUCV_CHUNKER_ATTRS: dict[str, str] = {
    "c_puct": "_c_puct",
    "cpuct_factor": "_cpuct_factor",
    "cpuct_base": "_cpuct_base",
    "fpu_reduction": "_fpu_red",
    "vloss_weight": "_vloss",
}

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


def _checked_chunk_sims(n: int) -> int:
    """Reject a chunk the int32 tree-id headroom cannot cover (see
    ``MAX_CHUNK_SIMS``): a classic Gumbel chunk cannot be shrunk once started."""
    n = int(n)
    if n > MAX_CHUNK_SIMS:
        raise ValueError(f"chunk_sims {n} exceeds the maximum {MAX_CHUNK_SIMS}")
    return n


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
  # Set (to a human reason) when the C Gumbel path REFUSED to build a tree for
  # this root, so `bestmove` is the raw cached policy prior and NO simulation
  # ran. `nodes` is 0 on such a result. `Engine` turns this into an
  # `info string` + a counter: an unsearched move that looks like a searched one
  # in a match log is exactly the class of defect this file keeps producing.
    root_declined: str | None = None
    search_incomplete: str | None = None


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


# Pool-side root-support guard, observable. `_drop_pool_root_if_support_differs`
# rebuilds an expanded persistent root whose child set is not exactly the board's
# legal support. Both counters are process-cumulative so a run can be asked "did
# this ever fire?" — a guard nobody can observe is a guard nobody knows is dead
# (the same reason gumbel_c keeps `root_coverage_miss_count()` /
# `root_support_narrowed_count()`, whose naming these mirror).
#
# ⚑ The two verdicts mean the OPPOSITE things here that they mean in gumbel_c,
# because the pool asks the WIDEST question there is — the board's full legal
# support — while a Gumbel search can legitimately ask a narrower one:
#
#   * `_ROOT_SUPPORT_MISSING_ACTION` (the carried root lacks a legal move) is the
#     ROUTINE case here: it is exactly what a preceding `searchmoves` Gumbel
#     search leaves behind, and healing it is this guard's whole purpose. Counted
#     only, plus a debug line — warning on it would fire on ordinary UCI traffic.
#     In gumbel_c the same verdict is the W2 ALARM, because there the narrower
#     search should never want a move its carried root does not have.
#   * `_ROOT_SUPPORT_NARROWED` (every legal move is present and the root carries
#     something ELSE) is the ALARM here: the tree holds a root child that is not
#     a legal move of the position it claims to be, which no supported path
#     produces. Warn once, then count. In gumbel_c that verdict is the routine
#     one.
#
# `_POOL_ROOT_SUPPORT_CORRUPTIONS` counts a SUBSET of
# `_POOL_ROOT_SUPPORT_REBUILDS`: every rebuild bumps the rebuild counter, and the
# alarm-shaped ones bump both. Both verdicts still rebuild, so the counters are
# pure observation and change no behaviour.
_POOL_ROOT_SUPPORT_REBUILDS = 0
_POOL_ROOT_SUPPORT_CORRUPTIONS = 0
_POOL_ROOT_SUPPORT_CORRUPTION_WARNED = False


def _note_pool_root_support_rebuild(*, corrupt: bool) -> None:
    global _POOL_ROOT_SUPPORT_REBUILDS, _POOL_ROOT_SUPPORT_CORRUPTIONS
    global _POOL_ROOT_SUPPORT_CORRUPTION_WARNED
    _POOL_ROOT_SUPPORT_REBUILDS += 1
    if not corrupt:
        _log.debug(
            "uci: rebuilt an expanded pool root that was MISSING a legal move "
            "(routine searchmoves aftermath; the previous search narrowed the "
            "root and this one asks the full legal question). Rebuild %d.",
            _POOL_ROOT_SUPPORT_REBUILDS,
        )
        return
    _POOL_ROOT_SUPPORT_CORRUPTIONS += 1
    if not _POOL_ROOT_SUPPORT_CORRUPTION_WARNED:
        _POOL_ROOT_SUPPORT_CORRUPTION_WARNED = True
        _log.warning(
            "uci: rebuilt an expanded pool root that carried a child action "
            "which is NOT a legal move of this position. Every legal move was "
            "present, so this is not searchmoves aftermath — the tree disagrees "
            "with the rules and something upstream changed. Further occurrences "
            "are counted, not logged."
        )


def pool_root_support_rebuild_count() -> int:
    """Pool roots rebuilt for support mismatch since process start (all causes)."""
    return _POOL_ROOT_SUPPORT_REBUILDS


def pool_root_support_corruption_count() -> int:
    """Subset of the above whose root carried a NON-legal action (alarm)."""
    return _POOL_ROOT_SUPPORT_CORRUPTIONS


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
        vloss_weight: int = PLAY_SEARCH_VLOSS_WEIGHT,
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
        _required = _REQUIRED_MCTS_ABI
        if _abi < _required:
            raise RuntimeError(
                f"compiled _mcts_tree ABI_VERSION={_abi} < required {_required} "
                "(missing node_capacity, the start_gumbel_sims root-scale args, "
                "the audit-W1 transposition-key fix, batch_process_ply's "
                "search_wdl draw-mode args and/or the ABI-10 Hash footprint API: "
                "footprint_bytes / set_growth_step / compact_to_subtree); "
                "rebuild the C extension: "
                "python3 scripts/build_production_extensions.py"
            )
        self._evaluator = evaluator
        self._device = device
        self._cfg = gumbel_cfg or GumbelConfig(
            simulations=chunk_sims,
            add_noise=False,  # no exploration noise at test time
            temperature=0.0,
        )
        self._chunk_sims = _checked_chunk_sims(chunk_sims)
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
  # FEN of the position `_root_pol_logits` was evaluated on. The bestmove
  # fallback reads the cached prior; without the key a failure on a NEW
  # position (before its root eval) picked a move from the PREVIOUS root.
        self._root_pol_key: str | None = None
        self._last_gumbel_action_idx: int | None = None
  # Best-move stability across chunks, for the complexity gate on the early abort
  # (see ``_abort_ready``). Reset at the start of every search.
        self._abort_last_best: int = -1
        self._abort_stable_chunks: int = 0
  # Optional Syzygy probe. When set, MCTS leaves in the TB range get
  # their NN wdl overridden with the TB-truth distribution.
        self._tb_probe = None
  # Soft memory cap (UCI ``Hash``) on the tree's RESIDENT footprint
  # (``MCTSTree.footprint_bytes()``): the search halts between chunks once the
  # footprint reaches it, and says so with an ``info string``. 0 = unbounded.
  # Not a hash table — tree growth is all or nothing, we stop adding rather
  # than evicting.
        self._max_tree_bytes: int = 0
  # Why the last `run()` stopped (`_should_stop_search` reason, or "tree_bytes"
  # from the shared-tree headroom check); None = budget/abort/early return.
        self.last_stop_reason: str | None = None
  # `go mate N` bookkeeping (`_proven_mate_within`): the (tree, root, solved
  # status, node count) of the last mate walk, so an unchanged tree is not
  # re-walked; and the node visits that walk spent (test / diagnostics read).
        self._mate_check_key: tuple[int, int, int, int] | None = None
        self._mate_witness_action: int | None = None
        self.last_mate_walk_visits: int = 0
  # Set by `advance_root` on every in-place reroot under a Hash cap; resolved
  # (keep / compact / rebuild) at the start of the next `run()`, which knows
  # the move budget. `last_reuse_inputs` records the numbers the decision used.
        self._compaction_pending: bool = False
        self.last_reuse_decision: str | None = None
        self.last_reuse_inputs: dict[str, float] = {}
  # Growth rate measured by the last `run()` that did enough visits: visits per
  # ms of search and resident tree bytes added per visit. Feeds the reuse rule.
        self._recent_visits_per_ms: float = 0.0
        self._recent_bytes_per_visit: float = 0.0
  # Minibatch target for the C gumbel state machine. 0 = use the
  # C-side GSS_GPU_BATCH default. Higher = better GPU util on large
  # batches; lower = faster stop latency + fresher tree state on
  # each leaf. Set via UCI `MinibatchSize`.
        self._minibatch_size: int = PLAY_SEARCH_TARGET_BATCH
  # Clock-margin width in std-devs of measured chunk wall-time (see
  # `_BATCH_MARGIN_SIGMAS`). Set via UCI `ClockBatchMarginSigmas`; 0 disables.
        self._batch_margin_sigmas: float = _BATCH_MARGIN_SIGMAS
  # Root Gumbel-noise strength, UCI `GumbelScale`. 0 = deterministic search,
  # which is what every build before this option did (the chunk hard-coded
  # `add_noise=False`). See `set_root_noise_scale` for why this is a worker
  # field and not `GumbelConfig.gumbel_scale`.
        self._root_noise_scale: float = 0.0
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
                cpuct_factor=float(getattr(self._cfg, "cpuct_factor", 0.0) or 0.0),
                cpuct_base=float(getattr(self._cfg, "cpuct_base", 38739.0) or 38739.0),
                fpu_at_root=0.0,
                fpu_reduction=float(self._cfg.fpu_reduction),
                vloss_weight=self._vloss_weight,
                gather=self._walker_gather,
                input_planes=input_plane_count(self._cfg.input_extra_features),
                compute_relations=bool(self._cfg.compute_relations),
                input_history_encoding=self._cfg.input_history_encoding,
            ),
            self._evaluator,
        )

    def set_show_wdl(self, enabled: bool) -> None:
        """Toggle WDL emission on info lines. Takes effect next emit."""
        self._show_wdl = bool(enabled)

    def concurrency_profile(self) -> tuple[str, int]:
        """Current search mode and Python worker-thread count for telemetry."""
        if self._rpg_pool is not None:
            return "root_parallel_gumbel", max(1, len(self._rpg_pool_evals))
        if self._pucv_pool is not None:
            return "multi_gpu_pucv", max(1, len(self._pucv_pool_evals))
        if self._walker_pool is not None:
            return "walker_puct", max(1, int(self._n_walkers))
        if self._pucv is not None:
            return "pucv", 1
        return "gumbel", 1

    def _tree_footprint_bytes(self) -> int:
        """Bytes the tree really holds resident (``MCTSTree.footprint_bytes``).

        This — not the allocated capacity ``memory_bytes()`` — is what the Hash
        cap, ``hashfull`` and the ``advance_root`` reuse decision measure. The
        capacity figure counted a per-node-id CBoard cache that was ~25x larger
        than the boards it held and over-reported the tree ~6x against RSS, so
        a search stopped at ~1/8 of Hash with the clock still full."""
        return 0 if self._tree is None else int(self._tree.footprint_bytes())

    def _tree_node_count(self) -> int:
        return 0 if self._tree is None else int(self._tree.node_count())

    def _tree_growth_step_nodes(self) -> int:
        """Node slots one arena growth may add under the Hash cap (0 = doubling).

        Node arrays are zero-filled on growth, so capacity is resident. Plain
        doubling at a multi-GB arena would add GBs in one step and blow past
        Hash; capping each step at ``min(Hash/64, 32 MiB)`` bounds that
        overshoot and keeps each step's zero-fill stall to tens of ms."""
        if self._max_tree_bytes <= 0:
            return 0
        step_bytes = min(self._max_tree_bytes // _TREE_GROWTH_STEP_DIV, _MAX_TREE_GROWTH_STEP_BYTES)
        return max(_MIN_TREE_GROWTH_STEP_NODES, step_bytes // _TREE_NODE_BYTES)

    def _sync_tree_growth_step(self) -> None:
        if self._tree is not None:
            self._tree.set_growth_step(self._tree_growth_step_nodes())

    def _hashfull_permille(self) -> int | None:
        """Tree-memory fill as per-mille of the soft cap (UCI ``hashfull``).

        The cap is ``set_max_tree_mb`` (UCI ``Hash``); the fill is the tree's
        resident footprint, the same quantity the ``tree_bytes`` stop checks.
        None when the cap is disabled (0) or no tree exists yet."""
        if self._max_tree_bytes <= 0 or self._tree is None:
            return None
        used = self._tree_footprint_bytes()
        return max(0, min(1000, used * 1000 // self._max_tree_bytes))

    def _tree_full_note(self, elapsed_ms: int, deadline: Deadline, reason: str = "tree_bytes") -> str:
        """``info string`` text for a search that stops because the tree is full."""
        rem = deadline.remaining_ms()
        left = "no deadline" if rem is None else f"{rem} ms left on the hard deadline"
        if reason == "tree_ids":
            nodes = 0 if self._tree is None else int(self._tree.node_count())
            return (
                f"tree node-id space full: {nodes} nodes of the int32 id limit at footprint "
                f"{self._tree_footprint_bytes() // (1024 * 1024)} MB; Hash beyond this cannot "
                f"be used by one tree; stopping search at {elapsed_ms} ms ({left})"
            )
        return (
            f"tree memory full: footprint {self._tree_footprint_bytes() // (1024 * 1024)} MB "
            f"of Hash {self._max_tree_bytes // (1024 * 1024)} MB (next chunk would exceed it); "
            f"stopping search at {elapsed_ms} ms ({left}); raise Hash to search longer"
        )

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
        if self._walker_pool is None:
            self._walker_gather = n
            return
        previous = self._walker_gather
        self._walker_gather = n
        try:
            new_pool = self._build_walker_pool(self._n_walkers)
        except BaseException:
            self._walker_gather = previous  # old pool is untouched and live
            raise
        self._swap_walker_pool(new_pool)

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
            cpuct_factor=float(getattr(self._cfg, "cpuct_factor", 0.0) or 0.0),
            cpuct_base=float(getattr(self._cfg, "cpuct_base", 38739.0) or 38739.0),
            fpu_at_root=0.0,
            fpu_reduction=float(self._cfg.fpu_reduction),
            vloss_weight=self._vloss_weight,
            vloss_mode=self._pucv_vloss_mode,
            eval_cache_entries=self._eval_cache_entries,
            input_planes=input_plane_count(self._cfg.input_extra_features),
            compute_relations=bool(self._cfg.compute_relations),
            input_history_encoding=self._cfg.input_history_encoding,
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
        self.restore_classic_search_path()

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
        # UCI schedule knobs (-1 = auto when multi-GPU; 0 = off; >0 = fixed).
        # See ``resolve_rpg_schedule_knobs``.
        open_vpa: int = -1,
        min_vpa: int = -1,
        min_keep: int = -1,
        open_budget_frac: float = 0.50,
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
        n_groups = len(evaluators_or_factories)
        gath = max(1, int(gather))
        multi = n_groups > 1
        # Multi-GPU defaults (open/min_vpa = gather, min_keep = n_groups) keep
        # devices fed; UCI can force off (0) or a fixed value. Single-group
        # leaves schedule off for classic bit-identity.
        # See root_parallel_gumbel.RootParallelGumbelConfig / resolve_rpg_*.
        open_r, min_vpa_r, min_keep_r, frac_r = resolve_rpg_schedule_knobs(
            n_groups=n_groups,
            gather=gath,
            open_vpa=int(open_vpa),
            min_vpa=int(min_vpa),
            min_keep=int(min_keep),
            open_budget_frac=float(open_budget_frac),
        )
        cfg = RootParallelGumbelConfig(
            n_groups=n_groups,
            gather=gath,
            c_puct=float(self._cfg.c_puct),
            cpuct_factor=float(getattr(self._cfg, "cpuct_factor", 0.0) or 0.0),
            cpuct_base=float(getattr(self._cfg, "cpuct_base", 38739.0) or 38739.0),
            fpu_at_root=0.0,
            fpu_reduction=float(self._cfg.fpu_reduction),
            vloss_weight=self._vloss_weight,
            # Same PUCVPendingMode the pucv pool reads. RPG used to pin
            # virtual-mean here regardless, so an operator reading `legacy` off
            # the UCI option was reading a knob this pool never consulted —
            # while its descents re-walked the in-flight leaf (audit W3).
            vloss_mode=self._pucv_vloss_mode,
            split_idle_groups=multi,
            min_vpa=min_vpa_r,
            open_vpa=open_r,
            open_budget_frac=frac_r,
            min_keep=min_keep_r,
            eval_cache_entries=self._eval_cache_entries,
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
        single-evaluator gumbel/walker/pucv routing. Idempotent.

        Always resets the persistent tree: RPG updates only candidate-child
        N/W and never backprops into the root node (its scorer takes the
        FRESH root value), so it hands
        back a tree whose root ``W``/``N`` still sit at the initial eval while
        its children carry a full search. ``_run_*_chunk`` returns
        ``self._tree.node_q(self._root_id)`` as the REPORTED root value, so a
        carried RPG-shaped tree makes the reported score wrong.
        ⚑ That is now the only reason. The original one — classic
        ``gss_score_and_halve`` reading ``W[root]/N[root]`` — no longer holds:
        it takes the fresh ``root_qs``.
        """
        had_pool = self._rpg_pool is not None
        if self._rpg_pool is not None:
            self._rpg_pool.close()
            self._rpg_pool = None
        for ev in self._rpg_pool_evals:
            close = getattr(ev, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        self._rpg_pool_evals = []
        if had_pool:
            self.reset_tree()

    def restore_classic_search_path(self) -> None:
        """Rebuild walker-pool / single-thread PUCV after multi-GPU teardown.

        ``install_root_parallel_gumbel`` / ``install_multi_gpu_pucv`` close the
        walker pool and clear ``_pucv`` without touching ``_n_walkers`` /
        ``_use_pucv``, so ``set_num_threads`` / ``set_use_pucv`` early-return
        and leave the engine on the single-thread classic Gumbel path. Call
        this after ``clear_*`` when the configured non-multi-GPU path should
        come back.
        """
        # Build first, then swap: a failed build leaves the current path live.
        new_pool = self._build_walker_pool(self._n_walkers)
        try:
            new_pucv = self._build_pucv() if self._use_pucv else None
        except BaseException:
            if new_pool is not None:
                new_pool.close()
            raise
        self._swap_walker_pool(new_pool)
        self._pucv = new_pucv
        self.reset_tree()

    def _swap_walker_pool(self, new_pool: WalkerPool | None) -> None:
        """Install an already-built walker pool, then retire the old one."""
        old_pool, self._walker_pool = self._walker_pool, new_pool
        if old_pool is not None and old_pool is not new_pool:
            old_pool.close()

    def realized_rpg_vloss_mode(self) -> tuple[int, ...] | None:
        """Per-group pending accounting the installed RPG pool descends with.

        ``None`` when no pool is installed. Sourced from the live chunkers, so
        it answers "did PUCVPendingMode reach the search" rather than "what was
        the pool configured with".
        """
        if self._rpg_pool is None:
            return None
        return self._rpg_pool.realized_vloss_mode()

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

        Reaches all three batched consumers: the single-thread chunker (rebuilt),
        the multi-GPU pucv pool (reads ``_cfg`` per job) and the root-parallel
        Gumbel pool (whose per-group chunkers are built once at pool
        construction, so its config alone would not reach the descent).
        """
        mode = 1 if int(mode) == 1 else 0
        if mode == self._pucv_vloss_mode:
            return
        previous = self._pucv_vloss_mode
        self._pucv_vloss_mode = mode
        if self._pucv is not None:
            try:
                self._pucv = self._build_pucv()
            except BaseException:
                self._pucv_vloss_mode = previous  # old chunker untouched and live
                raise
        if self._pucv_pool is not None:
            self._pucv_pool._cfg.vloss_mode = mode
        if self._rpg_pool is not None:
            self._rpg_pool.set_vloss_mode(mode)
        self.reset_tree()

    def set_use_pucv(self, enabled: bool, *, gather: int | None = None) -> None:
        """Enable async-pipeline batched-VL PUCT (single-thread, 2-slot
        overlap). Requires ``n_walkers == 1`` and an evaluator with
        ``evaluate_inplace_async`` + ``n_slots >= 2``. When the requirement
        isn't met or ``enabled`` is False, the classic gumbel path is used.
        Resets the tree because pucv accumulates vloss-adjusted Q/N stats
        that don't blend with gumbel's halving stats. Caller holds the
        search barrier.

        When a multi-GPU pool (RPG or multi-GPU PUCV) is installed, only the
        intended flag/gather are updated — materializing a live single-thread
        chunker beside those pools would be inert (dispatch prefers the pool)
        but wasteful. Leave-multi-GPU reinstall builds the chunker then.
        """
        enabled = bool(enabled)
        previous_gather = self._pucv_gather
        new_gather = previous_gather if gather is None else max(1, int(gather))
        if self._rpg_pool is not None or self._pucv_pool is not None:
            self._pucv_gather = new_gather
            self._use_pucv = enabled
            self._pucv = None
            return
        if (
            enabled == self._use_pucv
            and (self._pucv is not None) == enabled
            and previous_gather == new_gather
        ):
            # Disabling with a chunker still installed is NOT a no-op: the
            # last-resort fallback relies on this clearing a stray `_pucv`.
            return
        # Build first, flags after: a failed build must not leave `_use_pucv`
        # set, or a later rollback rebuilds the PUCV path the options say is off.
        self._pucv_gather = new_gather
        try:
            new_pucv = self._build_pucv() if enabled else None
        except BaseException:
            self._pucv_gather = previous_gather
            raise
        self._use_pucv = enabled
        self._pucv = new_pucv
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
            cpuct_factor=float(getattr(self._cfg, "cpuct_factor", 0.0) or 0.0),
            cpuct_base=float(getattr(self._cfg, "cpuct_base", 38739.0) or 38739.0),
            eval_cache_entries=self._eval_cache_entries,
            input_planes=input_plane_count(self._cfg.input_extra_features),
            compute_relations=bool(self._cfg.compute_relations),
            input_history_encoding=self._cfg.input_history_encoding,
        )

  # ─── realized search configuration ─────────────────────────────────────────
  # The readback an operator uses to prove what the engine ran. Both methods
  # read the LIVE objects, never the requested options: `UseVL=true` against an
  # evaluator without the slot API is accepted and then falls through to
  # classic Gumbel, and an EngineOptions-derived answer would confidently
  # report a path that never ran.
  #
  # "Live object" has to mean the object the DESCENT dereferences, not the one
  # the setter writes. `realized_search_values` sourced the PUCT family from
  # `self._cfg` while every PUCT descent reads a construction-time snapshot in
  # its own pool/chunker config -- so the readback could not witness the very
  # defect `rebuild_puct_helpers` exists to fix, and printed `CPuct = 99.0
  # [LIVE]` byte-identically whether or not the rebuild ran. A readback that
  # re-reads the request is not evidence about the search. `_puct_descent_values`
  # below closes that; everything it does not cover is `self._cfg`, and that IS
  # the live source for those fields (see the docstring).

    # Worker-side search state a setoption can change. `Engine.dispatch`
    # snapshots it with `EngineOptions` so a failed setoption restores BOTH
    # (e.g. `set_gumbel_field` writes `_cfg.c_puct` before the helper rebuild
    # that can fail). The evaluator, pools and tablebase are not here: they
    # have their own build-then-swap ownership. Deliberately NOT here: the tree
    # and its reuse state (`_compaction_pending`, `last_reuse_*`) and the
    # measured growth rate (`_recent_visits_per_ms` / `_recent_bytes_per_visit`).
    # A setoption that drops them (via `reset_tree` / a Hash change) does not
    # get them back on rollback: the tree is gone either way, and a rate is
    # re-measured by the next search -- restoring it could resurrect a rate
    # measured under a configuration the failed setoption half-changed.
    _SEARCH_STATE_FIELDS: tuple[str, ...] = (
        "_n_walkers", "_walker_gather", "_use_pucv", "_pucv_gather",
        "_vloss_weight", "_eval_cache_entries", "_root_noise_scale",
        "_chunk_sims", "_pucv_vloss_mode", "_minibatch_size",
        "_batch_margin_sigmas", "_multi_pv", "_show_wdl", "_max_tree_bytes",
    )

    def search_state(self) -> tuple[dict[str, object], object]:
        """A restorable copy of the worker-side search state."""
        fields_ = {name: getattr(self, name) for name in self._SEARCH_STATE_FIELDS
                   if hasattr(self, name)}
        return fields_, dataclasses.replace(self._cfg)

    def restore_search_state(self, state: tuple[dict[str, object], object]) -> bool:
        """Put ``search_state()`` back; True if anything had changed.

        `_cfg` is restored field by field INTO the live object: the RPG pool
        reads it by reference, so replacing the object would detach it.
        """
        fields_, cfg = state
        changed = False
        for name, value in fields_.items():
            if getattr(self, name) != value:
                setattr(self, name, value)
                changed = True
        for f in dataclasses.fields(cfg):  # pyright: ignore[reportArgumentType]
            value = getattr(cfg, f.name)
            if getattr(self._cfg, f.name) != value:
                setattr(self._cfg, f.name, value)
                changed = True
        tree = self._tree
        if changed and tree is not None:
            tree.set_cpuct_scaling(
                float(getattr(self._cfg, "cpuct_factor", 0.0) or 0.0),
                float(getattr(self._cfg, "cpuct_base", 38739.0) or 38739.0),
            )
        if changed:
            # The live tree's growth step follows `_max_tree_bytes`
            # (`set_max_tree_mb`); a restored cap must re-sync it too.
            self._sync_tree_growth_step()
        return changed

    def classic_helpers_missing(self) -> bool:
        """The classic path the state names is not (or no longer) installed.

        A CLOSED walker pool counts as missing: `realized_search_path()` would
        report "walker" while every run raises "WalkerPool is closed".
        """
        pool = self._walker_pool
        if self._n_walkers > 1:
            return pool is None or pool.closed
        if pool is not None:
            return True  # a pool beside a single-walker state
        return self._use_pucv and self._pucv is None and self._build_pucv_supported()

    def _build_pucv_supported(self) -> bool:
        ev = self._evaluator
        return (self._n_walkers == 1 and hasattr(ev, "evaluate_inplace_async")
                and getattr(ev, "n_slots", 1) >= 2)

    def realized_search_path(self) -> str:
        """Which search path a `run()` right now would actually dispatch to.

        Mirrors `_run_one_chunk`'s branch order exactly. Note `searchmoves`
        routes that ONE move back onto the classic Gumbel path regardless of
        installed pools -- that is per-`go` and not part of the configuration,
        so it is not reported here.
        """
        if self._rpg_pool is not None:
            return "rpg"
        if self._pucv_pool is not None:
            return "pucv_pool"
        if self._walker_pool is not None:
            return "walker"
        if self._pucv is not None:
            return "pucv"
        return "gumbel"

    def _puct_descent_values(self) -> dict[str, float | int]:
        """The PUCT family as the LIVE descent object holds it; `{}` on gumbel.

        Branch order mirrors `realized_search_path` so the readback can never
        answer off a pool the dispatch would not reach.
        """
        pool = (
            self._rpg_pool if self._rpg_pool is not None else
            self._pucv_pool if self._pucv_pool is not None else
            self._walker_pool
        )
        if pool is not None:
            cfg = pool._cfg
            return {f: getattr(cfg, f) for f in _PUCT_DESCENT_FIELDS}
        if self._pucv is not None:
            return {
                field: getattr(self._pucv, attr)
                for field, attr in _PUCV_CHUNKER_ATTRS.items()
            }
        return {}

    def realized_search_values(self) -> dict[str, float | int | bool]:
        """Every registered search parameter's value as the search will read it.

        Keyed by `SearchOption.field`. The four worker-level entries are not
        `GumbelConfig` fields at all -- `vloss_weight` and `minibatch_size` are
        function ARGUMENTS of `run_gumbel_root_many_c`, which is exactly why no
        `dataclasses.replace`-based override surface ever reached them and why
        every arena before 2026-07-28 silently ran `vloss_weight=0`.

        "As the search will read it" is meant literally, and it costs the
        `_puct_descent_values` override to be true. Every PUCT path snapshots
        `c_puct` / `cpuct_factor` / `cpuct_base` / `fpu_reduction` /
        `vloss_weight` into its own config at construction, so answering those
        from `self._cfg` would be re-reading the REQUEST -- and the readback
        would then certify `CPuct = 99.0 [LIVE]` on a pool still descending on
        2.5, which is exactly the failure it is supposed to expose. It stayed
        wrong for a reachable reason too: `rebuild_puct_helpers` closes before
        it rebuilds, so a `_build_walker_pool` that raises leaves the CLOSED
        pool installed with the old config, and a `_cfg`-sourced readback would
        certify the new value while the next `go` errors out.

        Everything not in that override does come off `self._cfg`, and for
        those fields `_cfg` IS the live object, not a proxy for it: the classic
        Gumbel chunk `dataclasses.replace`s `_cfg` per call, `RootParallelGumbelPool`
        dereferences the shared config at candidate-selection time, and
        `chunk_sims` / `minibatch_size` / `root_noise_scale` are read off the
        worker fields returned here.
        """
        cfg = self._cfg
        out: dict[str, float | int | bool] = {
            "chunk_sims": int(self._chunk_sims),
            "vloss_weight": int(self._vloss_weight),
            "minibatch_size": int(self._minibatch_size),
            "root_noise_scale": float(self._root_noise_scale),
        }
        for opt in SEARCH_OPTIONS:
            if opt.field in out:
                continue
            value = getattr(cfg, opt.field)
            out[opt.field] = (
                bool(value) if isinstance(value, bool) else
                int(value) if isinstance(value, int) else float(value)
            )
        for field, value in self._puct_descent_values().items():
            out[field] = (
                int(value) if isinstance(value, int) else float(value)
            )
        return out

    def set_gumbel_field(self, field: str, value: float | int | bool) -> None:
        """Set one `GumbelConfig` field on the config the search reads.

        `self._cfg` is the object every path reads its shape off, and the
        classic Gumbel chunk `dataclasses.replace`s it per call, so a plain
        assignment IS live there. The PUCT helpers (walker pool / PucvChunker /
        multi-GPU pool / RPG) copy c_puct & friends at CONSTRUCTION time, so
        for those the assignment alone is inert and something must rebuild
        them: `Engine._sync_cpuct_to_worker` routes RPG to
        `_reinstall_rpg_if_active`, the multi-GPU pool to
        `_install_multi_gpu_pucv_pool`, and walker / single-thread PucvChunker
        to `rebuild_puct_helpers` below.
        """
        previous = getattr(self._cfg, field)
        setattr(self._cfg, field, value)
        if (
            field == "policy_temp"
            and previous != value
            and self.realized_search_path() in ("gumbel", "rpg")
        ):
            # Rebuild priors seeded at the preceding temperature. Plain PUCT
            # advertises this option inert and retains its existing tree.
            self.reset_tree()

    def rebuild_puct_helpers(self) -> None:
        """Re-materialize the helpers that snapshot c_puct & friends at build.

        The walker pool copies the family into its own `WalkerPoolConfig` and
        `PucvChunker` copies it into `_c_puct` / `_fpu_red` / `_cpuct_*`, both
        at CONSTRUCTION. Writing `_cfg.c_puct` therefore does not reach either
        descent, which is the "accepted, certified realized, and then ignored"
        failure this whole option surface exists to prevent — and it is worse
        than dead, because the stale value silently lands later on the next
        command that happens to rebuild (`VLossWeight`, `MaxBatch`, ...).

        Deliberately does NOT reset the tree, and that is a TRADE-OFF, not a
        clean win -- measured, not asserted. On pucv with the tree reused
        across two `go`s (`test_a_mid_tree_puct_change_keeps_the_tree_and_is_a_hybrid`):

            A  default throughout                     go2 = (1921, 14753)
            B  c_puct=99 from construction            go2 = (1921, 15144)
            C  default go -> setoption 99 -> go       go2 = (1921, 15094)

        `C` equals NEITHER. The accumulated visit distribution is itself a
        function of the old constant, so until the tree turns over the search
        is a hybrid of two regimes rather than the one the operator configured.
        Q and N stay individually meaningful; the SELECTION they feed does not
        become the new configuration's selection immediately.

        Kept anyway: discarding a tree mid-game (or mid-ponder) to honour a
        `setoption` is the worse failure, and at handshake time -- the
        TCEC/cutechess case this surface exists for -- the tree is empty and the
        question does not arise. Note the asymmetry with the row directly above
        it in `searchconfig`: `VLossWeight` DOES reset, because vloss-adjusted Q
        is not comparable across weights, whereas these only bias selection.

        Also deliberately does not set `Engine._warmup_dirty`. Re-warming exists
        for knobs that RESHAPE the batch (Threads, LeafGather, VLGather,
        MinibatchSize); a rebuilt chunker reallocates the same shapes against
        the same evaluator, so the captured graph stays valid.
        """
        self._rebuild_classic_helpers()

    def _rebuild_classic_helpers(self) -> None:
        """Rebuild the installed walker pool / PucvChunker from current state.

        BUILD, THEN SWAP: both replacements are constructed before either old
        helper is retired, so a raise leaves the old (live, unclosed) helpers in
        place and the caller reverts the field it changed. Closing first used to
        leave `_walker_pool` pointing at a CLOSED pool after a failed build, and
        every later move was a fallback.
        """
        new_pool = (
            self._build_walker_pool(self._n_walkers) if self._walker_pool is not None else None
        )
        try:
            new_pucv = self._build_pucv() if self._pucv is not None else None
        except BaseException:
            if new_pool is not None:
                new_pool.close()
            raise
        if new_pool is not None:
            self._swap_walker_pool(new_pool)
        if self._pucv is not None:
            self._pucv = new_pucv

    def set_root_noise_scale(self, scale: float) -> None:
        """Root Gumbel-noise strength for match play. 0 = deterministic.

        The worker field, not `GumbelConfig.gumbel_scale`, is the source of
        truth: the classic chunk has always forced `add_noise=False`, so the
        dataclass field was unreadable from here, and its library default of
        1.0 would silently switch noise ON for every existing SearchWorker
        caller the moment the gate moved.

        The mirror into `_cfg` is what makes the option reach the OTHER Gumbel
        path: `RootParallelGumbelPool` reads `add_noise`/`gumbel_scale` off the
        shared config by reference at candidate-selection time, so writing them
        here reaches an installed RPG pool live, with no reinstall. Without the
        mirror this option would be a textbook silent null on `rpg`.
        """
        self._root_noise_scale = max(0.0, float(scale))
        self._cfg.add_noise = self._root_noise_scale > 0.0
        self._cfg.gumbel_scale = self._root_noise_scale

    def set_chunk_sims(self, n: int) -> None:
        """Sims per search chunk. Read fresh by `_chunk_budget` each chunk."""
        self._chunk_sims = max(1, _checked_chunk_sims(n))

    def set_vloss_weight(self, n: int) -> None:
        """Virtual-loss weight. Live for the Gumbel chunk (passed per call);
        the pooled PUCT paths bake it into their config at construction, so
        rebuild whichever one is installed."""
        n = max(0, int(n))
        if n == self._vloss_weight:
            return
        previous = self._vloss_weight
        self._vloss_weight = n
        try:
            self._rebuild_classic_helpers()
        except BaseException:
            self._vloss_weight = previous  # old helpers untouched and live
            raise
        self.reset_tree()

    def set_eval_cache_entries(self, n: int) -> None:
        """Set eval-cache capacity for newly built search helpers.

        This is behavior-preserving, so changing it does not require a tree
        reset. If single-thread PUCV is active, rebuild the chunker so the
        capacity takes effect immediately.
        """
        n = max(0, int(n))
        if n == self._eval_cache_entries:
            return
        previous = self._eval_cache_entries
        self._eval_cache_entries = n
        if self._pucv is not None:
            try:
                self._pucv = self._build_pucv()
            except BaseException:
                self._eval_cache_entries = previous
                raise

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

    def _batch_margin_ms(self, min_samples: int = _BATCH_MARGIN_MIN_SAMPLES) -> float:
        """Clock reserve = mean + K*std of measured chunk time.

        Returns 0 until ``min_samples`` chunks have run (no estimate yet) or
        when the margin is disabled (sigmas <= 0). With one sample the reserve
        is that chunk's time (no spread estimate yet)."""
        if (
            self._batch_margin_sigmas <= 0.0
            or self._batch_n < max(1, min_samples)
        ):
            return 0.0
        if self._batch_n < 2:
            return self._batch_mean_ms
        var = self._batch_m2_ms / (self._batch_n - 1)
        std = var ** 0.5 if var > 0.0 else 0.0
        return self._batch_mean_ms + self._batch_margin_sigmas * std

    def _clock_time_margin_ms(
        self, optimum_ms: int | None, start_remaining_ms: int | None,
        *, fixed_time: bool = False,
    ) -> float:
        """Capped batch-time reserve passed to the stop check.

        Zero unless this is a clock game (``optimum_ms`` set) or ``fixed_time``
        (UCI ``movetime``, opted in by the engine) — ``nodes`` / ``depth`` and
        other callers' fixed-deadline searches run their exact requested work. Capped at
        ``_MAX_BATCH_MARGIN_FRACTION`` of the move's starting budget so a bad
        early chunk estimate can't reserve away the whole move.
        """
        if optimum_ms is None:
            if not (fixed_time and start_remaining_ms is not None):
                return 0.0
            # A short movetime may fit only one or two chunks: reserve from the
            # first measured one rather than overrun the deadline by the second.
            m = self._batch_margin_ms(min_samples=1)
        else:
            m = self._batch_margin_ms()
        if m > 0.0 and start_remaining_ms is not None:
            m = min(m, _MAX_BATCH_MARGIN_FRACTION * float(start_remaining_ms))
        return m

    def close(self, *, close_evaluator: bool = True) -> None:
        """Close the current evaluator. Safe to call multiple times; no-op
        if the evaluator has no ``close`` method. Used at process shutdown
        to drain ``BatchCoalescingDispatcher``'s submitter thread before
        Python tears down PyTorch's CUDA context. Also tears down the
        owned search-pool threads. ``close_evaluator=False`` retires helpers
        for a failed construction while the caller retains the primary."""
        if self._walker_pool is not None:
            self._walker_pool.close()
            self._walker_pool = None
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
        if close_evaluator:
            ev = self._evaluator
            close = getattr(ev, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

    def set_evaluator(self, evaluator: BatchEvaluator, *, keep_pools: bool = False) -> None:
        """Swap in a freshly-built (and warmed-up) evaluator. BUILD, THEN SWAP.

        The walker pool / single-thread PUCV chunker are rebuilt against the
        new evaluator BEFORE anything is torn down. If that build raises, the
        worker is exactly as it was -- old evaluator, old helpers, old pools,
        all live -- the replacement is closed (ownership passed to us with the
        call), and the error propagates. Only after a successful build are the
        multi-GPU pools cleared, the helpers swapped and the old evaluator
        closed. Before, a raise here left `_walker_pool` pointing at the
        CLOSED old pool and every later search failed ("WalkerPool is closed").

        ``keep_pools=True`` swaps only the primary under an already-installed
        multi-GPU pool (the pool owns search, so no walker / PUCV helper is
        built beside it). Caller holds the search barrier.

        Closes the outgoing evaluator to release its background threads
        (notably ``BatchCoalescingDispatcher``'s non-daemon submitter).
        Without this, ``setoption MaxBatch`` repeatedly would leak one
        submitter per rebuild and block process shutdown.
        """
        old = self._evaluator
        pools_kept = keep_pools and (self._rpg_pool is not None or self._pucv_pool is not None)
        self._evaluator = evaluator
        new_pool: WalkerPool | None = None
        new_pucv: PucvChunker | None = None
        try:
            if not pools_kept:
                new_pool = self._build_walker_pool(self._n_walkers)
                new_pucv = self._build_pucv() if self._use_pucv else None
        except BaseException as exc:
            self._evaluator = old
            if new_pool is not None:
                with contextlib.suppress(Exception):
                    new_pool.close()
            if evaluator is not old:
                try:
                    close = getattr(evaluator, "close", None)
                    if callable(close):
                        close()
                except BaseException as cleanup:
                    note_cleanup(exc, f"replacement evaluator close failed: {cleanup!r}")
            raise
        if not pools_kept:
            self.clear_multi_gpu_pucv(restore_fallback=False)
            self.clear_root_parallel_gumbel()
            self._swap_walker_pool(new_pool)
            self._pucv = new_pucv
        self.reset_tree()
        if old is not evaluator:
            try:
                close = getattr(old, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass  # the replacement is installed; a failed retire is not a failed swap

    def set_num_threads(self, n: int) -> None:
        """Rebuild the walker pool at thread count ``n`` (1 = classic Gumbel
        path, no pool). Drops the persistent tree because walker-pool Q/N
        stats accumulate with vloss adjustments that depend on thread count.
        Caller should hold the search barrier — same pattern as
        ``set_tb_probe``.

        When a multi-GPU pool (RPG or multi-GPU PUCV) is active it owns
        search; only store ``n`` for the next leave-multi-GPU restore and
        skip spawning an idle walker pool beside the live path.
        """
        n = max(1, int(n))
        if self._rpg_pool is not None or self._pucv_pool is not None:
            self._n_walkers = n
            if self._walker_pool is not None:
                self._walker_pool.close()
                self._walker_pool = None
            self._pucv = None
            return
        if n == self._n_walkers:
            return
        previous = self._n_walkers
        self._n_walkers = n
        # Build first, then swap: a failed build keeps the old count and pool.
        try:
            new_pool = self._build_walker_pool(n)
        except BaseException:
            self._n_walkers = previous
            raise
        self._swap_walker_pool(new_pool)
        # pucv requires single-thread; rebuild against the new walker count
        # (returns None when n != 1, which silently disables the path).
        if self._use_pucv:
            self._pucv = self._build_pucv()
        self.reset_tree()

    def reset_tree(self) -> None:
        self._compaction_pending = False
  # Every evaluator swap and search-topology change (MaxBatch / EvalCacheEntries
  # rebuild, Threads, walkers, VL, multi-GPU / RPG install or teardown) and
  # `ucinewgame` reset the tree, so a growth rate measured under the old
  # configuration is dropped here too. A rate measured on a slower setup
  # would inflate `carried` and keep a full arena.
        self._recent_visits_per_ms = 0.0
        self._recent_bytes_per_visit = 0.0
        self._tree = None
        self._root_id = None
        self._tree_fen = None
        self._invalidate_root_caches()

    def _invalidate_root_caches(self) -> None:
        self._root_pol_logits = None
        self._root_pol_key = None
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
        note: str | None = None,
    ) -> None:
        """Emit one or more ``info`` lines, one per top-ranked PV.

        ``note`` (e.g. the tree-full stop reason) is appended to the first
        line's ``string`` field, so it reaches the GUI log next to the PV.

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
        if note is not None:
            pucv_info = note if pucv_info is None else f"{pucv_info}; {note}"
        hashfull = self._hashfull_permille()
  # seldepth is the deepest PV any emitted line reaches, kept monotonic across
  # the search so it climbs like a real selective-depth counter.
        self._seldepth = max(
            self._seldepth, max((len(pv_idx) for _, _, pv_idx in lines), default=0),
        )
        mates = _proven_line_mates(
            self._tree, self._root_id, board, [pv_idx[0] if pv_idx else None for _, _, pv_idx in lines],
        )
        for (rank, q, pv_idx), score_mate in zip(lines, mates, strict=True):
            uci_pv = _uci_pv(board, pv_idx)
            wdl = _q_to_wdl_permille(q, draw_rate) if self._show_wdl else None
            info_cb(
                nodes=total_nodes,
                elapsed_ms=elapsed_ms,
                score_cp=q_to_cp(0.5 * (q + 1.0)),
                pv=uci_pv,
                tbhits=tbhits,
                score_mate=score_mate,
                multipv=rank if emit_multipv else None,
                wdl=wdl,
                string=pucv_info if rank == 1 else None,
                hashfull=hashfull,
                seldepth=self._seldepth,
            )

    def set_max_tree_mb(self, mb: int) -> None:
        """Soft cap on the tree's resident footprint; 0 disables. Checked
        between chunks; the live tree's growth step follows immediately."""
        new_cap = max(0, int(mb)) * 1024 * 1024
        if new_cap != self._max_tree_bytes:
            self._recent_visits_per_ms = 0.0
            self._recent_bytes_per_visit = 0.0
        self._max_tree_bytes = new_cap
        self._sync_tree_growth_step()

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

    def leaf_tb_inactive_mode(self) -> str | None:
        """The realized search path whose leaves never see the installed Syzygy probe.

        All five currently wired search paths receive ``tb_probe``. Keep an
        unknown future path visible until its leaf transport is implemented.
        ``None`` when no probe is installed or a leaf-enabled path is active.
        """
        if self._tb_probe is None:
            return None
        path = self.realized_search_path()
        return None if path in ("gumbel", "walker", "pucv", "pucv_pool", "rpg") else path

    def advance_root(self, board: chess.Board, moves: list[chess.Move]) -> bool:
        """Descend the current tree by ``moves`` plies, making the last-reached
        node the new root. ``board`` is the position BEFORE the first move;
        we push each move onto a local copy to compute its policy index.

        Returns True if the whole walk succeeded (tree reusable), False if any
        step fell off the expanded tree (caller must call ``reset_tree``).

        Rerooting in place only repoints ``self._root_id`` — the discarded
        sibling subtrees stay in the arena, so the footprint climbs across a
        game. Under a Hash cap the keep / compact / rebuild decision is
        scheduled for the next ``run()``, which knows the move budget, charges
        any compaction to the move clock and bounds its cost (see
        ``_resolve_pending_compaction``).
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
  # The reuse decision is deferred to `run()`, which knows the move's time
  # budget and runs under its clock (see `_resolve_pending_compaction`). Even a
  # half-empty arena can starve this move if it grows faster than the room left.
        self._compaction_pending = self._max_tree_bytes > 0
        self._sync_tree_growth_step()
        self._root_id = rid
        self._tree_fen = b.fen()
        self._invalidate_root_caches()
        return True

    def _resolve_pending_compaction(
        self, deadline: Deadline, optimum_ms: int | None, *,
        max_nodes: int | None = None, stop_sensitive: bool = False,
    ) -> str | None:
        """Carry out the keep / compact / rebuild decision ``advance_root``
        scheduled. Returns "kept" / "compacted" / "refused" / "rebuilt" (also
        ``last_reuse_decision``; inputs in ``last_reuse_inputs``), or None when
        nothing was pending.

        Inputs: arena footprint F, Hash H, compaction estimate
        ``est = F x _COMPACTION_MS_PER_GIB``, and the last search's growth
        rate under the current configuration: v visits/ms and b bytes/visit
        (r = v*b bytes/ms). The kept subtree holds N_k root visits, i.e.
        ``K = min(F, N_k * b)`` bytes after compaction.

        Time-budgeted search (``optimum_ms`` or a deadline). B = the budget
        (``optimum_ms``, else the deadline's remaining time); T = the horizon
        the search may really run for room purposes: the hard deadline's
        remaining time when there is one (a clocked search often runs past its
        optimum toward the hard deadline), else B. ``carried = min(B, N_k/v)``.
        Pick the largest projected effective search time; ties prefer keep,
        then compact:
        - keep:    carried + min(T, (H - F) / r)
        - compact: carried + min(T - est, (H - K) / r), only if est <= B and
                   the compaction transient fits (see below)
        - rebuild: min(T, H / r)
        No rate yet: compact if ``est <= 5% x B``.

        No time budget:
        - ponder / ``go infinite`` (``stop_sensitive``): compact only if
          ``est <= _STOP_SENSITIVE_COMPACTION_MAX_MS`` (stop cannot interrupt it);
        - ``go nodes N``: keep while ``F + N*b <= H`` (the move fits), else compact
          (no clock, so the cost only delays the start); without a rate keep
          below H/2;
        - ``go depth``: keep below H/2, else compact.

        A compaction is never chosen when ``F + transient > H + margin``
        (``_compaction_transient_bytes``). Whatever is not decided above (or a
        compaction ruled out) keeps the tree below 3/4 H and rebuilds above.
        """
        if not self._compaction_pending:
            return None
        self._compaction_pending = False
        tree, rid = self._tree, self._root_id
        if tree is None or rid is None or rid < 0 or self._max_tree_bytes <= 0:
            return None
        hash_bytes = float(self._max_tree_bytes)
        footprint = float(self._tree_footprint_bytes())
        est_ms = footprint / float(1 << 30) * _COMPACTION_MS_PER_GIB
        budget = optimum_ms if optimum_ms is not None else deadline.remaining_ms()
        hard = deadline.remaining_ms()
        horizon = float(max(budget, hard)) if budget is not None and hard is not None else (
            None if budget is None else float(budget)
        )
        v, b = self._recent_visits_per_ms, self._recent_bytes_per_visit
        carried_visits = float(self._kept_subtree_visits())
        kept_bytes = min(footprint, carried_visits * b) if b > 0.0 else footprint
        transient = self._compaction_transient_bytes(footprint, kept_bytes)
        can_compact = footprint + transient <= hash_bytes + _COMPACTION_TRANSIENT_MARGIN_BYTES
        inputs: dict[str, float] = {
            "footprint": footprint, "hash": hash_bytes, "est_ms": est_ms,
            "budget_ms": -1.0 if budget is None else float(budget),
            "horizon_ms": -1.0 if horizon is None else horizon,
            "visits_per_ms": v, "bytes_per_visit": b, "kept_bytes": kept_bytes,
            "transient_bytes": transient, "can_compact": float(can_compact),
        }
        decision = ""
        if budget is not None and horizon is not None and v > 0.0 and b > 0.0:
            rate = v * b
            carried = min(float(budget), carried_visits / v)
            options = [
                ("kept", carried + min(horizon, max(0.0, hash_bytes - footprint) / rate)),
                ("compacted", carried + min(max(0.0, horizon - est_ms),
                                            max(0.0, hash_bytes - kept_bytes) / rate)),
                ("rebuilt", min(horizon, hash_bytes / rate)),
            ]
            inputs["carried_ms"] = carried
            inputs.update({f"eff_{name}_ms": eff for name, eff in options})
            if not can_compact or est_ms > float(budget):
                options = [o for o in options if o[0] != "compacted"]
            choice = options[0]
            for option in options[1:]:
                if option[1] > choice[1] + 1.0:  # 1 ms tolerance: ties keep the earlier option
                    choice = option
            decision = choice[0]
        elif budget is not None:
            if est_ms <= _COMPACTION_BUDGET_FRACTION * float(budget):
                decision = "compacted"
        elif stop_sensitive:
            if est_ms <= _STOP_SENSITIVE_COMPACTION_MAX_MS:
                decision = "compacted"
        elif max_nodes is not None and b > 0.0:
            decision = "kept" if footprint + float(max_nodes) * b <= hash_bytes else "compacted"
        else:
            decision = "kept" if footprint < 0.5 * hash_bytes else "compacted"
        if decision == "compacted" and not can_compact:
            decision = ""
        if not decision:
            decision = "kept" if footprint < _KEEP_UNCOMPACTED_FRACTION * hash_bytes else "rebuilt"
        if decision == "compacted":
            new_rid = int(tree.compact_to_subtree(int(rid), int(self._max_tree_bytes)))
            if new_rid < 0:
                decision = "refused"
                self.reset_tree()
            else:
                self._root_id = new_rid
                self._invalidate_root_caches()
        elif decision == "rebuilt":
            self.reset_tree()
        self.last_reuse_decision = decision
        self.last_reuse_inputs = inputs
        return decision

    def _compaction_transient_bytes(self, footprint: float, kept_bytes: float) -> float:
        """Extra resident bytes while ``compact_to_subtree`` runs: the new child
        arrays (2 x int32 per kept child, pro rata of the arena's children) and
        the CBoard owner map (int32 per cached board), allocated before the old
        arrays are freed."""
        if self._tree is None or footprint <= 0.0:
            return 0.0
        stats = self._tree.memory_stats()
        kept_frac = min(1.0, kept_bytes / footprint)
        return 8.0 * float(stats["child_count"]) * kept_frac + 4.0 * float(stats["cb_pool_count"])

    def _kept_subtree_visits(self) -> int:
        """Root visits already searched under the current (rerooted) root."""
        if self._tree is None or self._root_id is None or self._root_id < 0:
            return 0
        _, visits = self._tree.get_children_visits(int(self._root_id))
        return int(np.asarray(visits, dtype=np.int64).sum())

    def _record_growth_rate(self, visits: int, nodes_before: int, search_ms: float) -> None:
        """Remember this search's visits/ms and resident bytes per visit for
        the next move's reuse decision (``_resolve_pending_compaction``).
        Bytes per visit = nodes added per visit x the arena's footprint per
        node, which is smooth where footprint growth itself is step-quantised."""
        if visits < _REUSE_RATE_MIN_VISITS or search_ms <= 0.0 or self._tree is None:
            return
        node_count = self._tree_node_count()
        added = node_count - nodes_before
        if added <= 0 or node_count <= 0:
            return
        self._recent_visits_per_ms = float(visits) / search_ms
        self._recent_bytes_per_visit = (
            float(added) / float(visits) * self._tree_footprint_bytes() / float(node_count)
        )

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
        self._root_pol_key = board.fen()

    def _root_policy_move(
        self, board: chess.Board, allowed_root_indices: set[int] | None,
    ) -> chess.Move | None:
        """Argmax of the CACHED root policy prior over ``board``'s legal moves.

        ``None`` when no root eval has been cached (the failure happened before
        or inside ``_ensure_root_eval_cached``), when the cache belongs to a
        different position than ``board``, or nothing survives the
        ``searchmoves`` filter. Pure bookkeeping — no evaluator call, no tree.
        """
        logits = self._root_pol_logits
        if logits is None or self._root_pol_key != board.fen():
            return None
  # The cached logits are the RAW head output (compact lc0_1858 on the
  # production net); the C path converts them exactly here before indexing by
  # policy index, so use the same converter rather than indexing raw.
        row = np.asarray(
            _policy_logits_to_full(np.asarray(logits, dtype=np.float32), cfg=self._cfg),
            dtype=np.float32,
        ).reshape(-1)
        best_move: chess.Move | None = None
        best_logit = -np.inf
        for move in board.legal_moves:
            idx = int(move_to_index(move, board))
            if allowed_root_indices is not None and idx not in allowed_root_indices:
                continue
            if idx < 0 or idx >= row.size:
                continue
            value = float(row[idx])
            if value > best_logit:
                best_logit = value
                best_move = move
        return best_move

    def root_policy_bestmove(
        self, board: chess.Board, *, searchmoves: tuple[str, ...] = (),
    ) -> str | None:
        """The net's own top root move from the cached prior, or ``None``.

        ``Engine`` calls this when a search raises: the root NN eval is cached
        once per search before the first chunk runs, so on essentially every
        in-search failure the prior that would name a decent move is already in
        memory. Returning it instead of ``next(iter(board.legal_moves))``
        (python-chess square order — effectively a random move) is the whole
        point; see the U2 entry in the wave-4 UCI audit.
        """
        allowed = _allowed_root_indices(board, searchmoves)
        if allowed is not None and not allowed:
            return None
        move = self._root_policy_move(board, allowed)
        return move.uci() if move is not None else None

    def fresh_root_policy_bestmove(
        self, board: chess.Board, *, searchmoves: tuple[str, ...] = (),
    ) -> str | None:
        """One root NN eval of ``board`` (no search), then its top legal move.

        The bestmove fallback's second resort, for when the cached prior is
        missing or belongs to another position. Bounded: one forward call, no
        tree. Raises whatever the evaluator raises; the caller catches.
        """
        allowed = _allowed_root_indices(board, searchmoves)
        if allowed is not None and not allowed:
            return None
        self._root_pol_logits = self._root_wdl_logits = None
        self._root_pol_key = None
        self._ensure_root_eval_cached(board, None)
        move = self._root_policy_move(board, allowed)
        return move.uci() if move is not None else None

    def _drop_pool_root_if_support_differs(self, board: chess.Board) -> None:
        """Drop an expanded root a PUCT pool cannot safely widen in place.

        A searchmoves search is routed through classic Gumbel because the PUCV
        and walker paths have no root filter. That Gumbel search can leave the
        persistent root expanded over only the requested actions. A later
        unrestricted pool search used to see an already-expanded root and adopt
        it verbatim, permanently hiding every excluded legal move.

        Pool roots are supposed to contain the board's full legal support. If
        an already-expanded root does not match it exactly, discard the tree
        and root and let ordinary pool prep rebuild from the cached root eval.
        The position is unchanged, so repeating the NN root call is unnecessary.

        The mismatch predicate is `gumbel_c._classify_expanded_root_support`, not
        a local copy: the C Gumbel path applies it to the same persistent root on
        alternating plies, and two hand-rolled comparisons drifting apart is how
        one of them silently stops rejecting. Its verdicts are read in the POOL
        direction — see the counter block above `SearchWorker` — and BOTH of them
        rebuild, so reusing the classifier changes nothing but observability.
        """
  # Exactly the paths `_run_one_chunk` dispatches an UNRESTRICTED search to that
  # adopt an already-expanded persistent root. `realized_search_path` is the
  # single source of truth for that dispatch (it mirrors `_run_one_chunk`'s
  # branch order); enumerating the pools by hand here is how this guard and the
  # dispatch would come to disagree about which one runs. "rpg" is excluded
  # because the RPG path never reuses a carried root — `_run_one_chunk` calls
  # `reset_tree()` before an RPG searchmoves chunk, and `advance_root` refuses
  # cross-move reuse outright — and "gumbel" because that path runs the
  # classifier itself.
        if self.realized_search_path() not in ("pucv_pool", "walker", "pucv"):
            return
        if self._tree is None or self._root_id is None or self._root_id < 0:
            return
        if not self._tree.is_expanded(self._root_id):
            return

  # ⚑ Chess960 would make this equality UNSATISFIABLE as the code stands:
  # `CBoard.legal_move_indices()` omits 960 castling while `move_to_index()`
  # emits it, so every 960 root would rebuild on every ply. MEASURED over all
  # 960 back-rank layouts: of the 696 that can castle, 696 mismatch (the review
  # that asked for this note measured 140/140 on its own sample). Unreachable
  # today — the engine exposes no `UCI_Chess960` option, and 960 rook placement
  # is not recoverable from the input planes at all
  # (`moves/leela_index.py:125-128`) — but a future 960 effort must revisit this
  # comparison before it revisits anything else here.
        legal_actions = CBoard.from_board(board).legal_move_indices().astype(
            np.int32, copy=False,
        )
        support = _classify_expanded_root_support(
            self._tree, self._root_id, legal_actions,
        )
        if support == _ROOT_SUPPORT_EQUAL:
            return
  # MISSING_ACTION here = the root lacks a legal move = searchmoves aftermath,
  # the routine case. Anything else = every legal move is present and the root
  # carries an extra, which is the corruption signal. ⚑ A root that is BOTH
  # missing a legal move AND carrying a non-legal one classifies as
  # MISSING_ACTION (the classifier ranks that verdict first), so the corruption
  # counter is a floor, not an exact census.
        _note_pool_root_support_rebuild(
            corrupt=support != _ROOT_SUPPORT_MISSING_ACTION,
        )

        self._tree = None
        self._root_id = None
        self._last_gumbel_action_idx = None
        self._walker_cboard = None
        self._pucv_cboard = None
        self._pucv_pool_cboard = None

    def _pre_expand_root_for_pool(
        self,
        board: chess.Board,
        allowed_root_indices: set[int] | None,
        *,
        allow_terminal_shortcuts: bool = True,
    ) -> None:
        """Pool paths race on the root's first descent so it must be expanded
        upfront. The classic gumbel path does this internally."""
        if allowed_root_indices is not None:
            return
        self._drop_pool_root_if_support_differs(board)
        if self._rpg_pool is not None:
            self._ensure_rpg_root_prepared(
                board, allow_terminal_shortcuts=allow_terminal_shortcuts,
            )
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
        witness_stop_fn: Callable[[], bool] | None = None,
    ) -> tuple[float, int]:
        """Returns ``(root_value, completed_sims)``.

        Multi-GPU pool and walker pool report real completed counts (either
        can stop mid-chunk on ``stop_event``). Single-thread PUCV and the
        gumbel C path do not report a count, so theirs is the root's visit
        growth over the chunk. Counting the REQUESTED ``chunk`` instead made a
        forced root (the Gumbel path returns its sole action without visiting
        it) print ~100M nodes at ~20M nps and fed that nps to
        ``_time_capped_chunk`` and the abort's provable-bank projection.
        """
        if allowed_root_indices is not None:
            # `searchmoves` restriction is only threaded through the single-walker
            # gumbel C path (via allowed_root_indices_batch). The pool / pucv / walker
            # paths have no such filter, so routing here keeps `searchmoves` correct
            # even at Threads>1 — at the cost of single-threaded search for that move.
            # Do NOT "optimize" this back onto the pools without first teaching them to
            # honor allowed_root_indices, or the restriction is silently ignored.
            #
            visits_before = self._root_visit_snapshot()
            value = self._run_gumbel_chunk(
                chunk, board, tb_probe, allowed_root_indices,
                allow_terminal_shortcuts=allow_terminal_shortcuts,
                witness_stop_fn=witness_stop_fn,
            )
            return value, self._root_visits_added(visits_before)
        tb_args = () if tb_probe is None else (tb_probe,)
        if self._rpg_pool is not None:
            return self._run_rpg_chunk(chunk, stop_event, *tb_args)
        if self._pucv_pool is not None:
            return self._run_pucv_pool_chunk(chunk, stop_event, *tb_args)
        if self._walker_pool is not None:
            return self._run_walker_chunk(chunk, stop_event, tb_probe)
        visits_before = self._root_visit_snapshot()
        if self._pucv is not None:
            value = self._run_pucv_chunk(chunk, *tb_args)
            return value, self._root_visits_added(visits_before)
        value = self._run_gumbel_chunk(
            chunk, board, tb_probe, allowed_root_indices,
            allow_terminal_shortcuts=allow_terminal_shortcuts,
            witness_stop_fn=witness_stop_fn,
        )
        return value, self._root_visits_added(visits_before)

    def _root_visit_total(self) -> int:
        """Visits summed over every root child (unfiltered); 0 without a root."""
        if self._tree is None or self._root_id is None or self._root_id < 0:
            return 0
        _actions, visits = self._tree.get_children_visits(self._root_id)
        return int(visits.sum())

    def _root_visit_snapshot(self) -> tuple[int, int | None, int]:
        """``(tree identity, root id, root visits)`` before a chunk."""
        return id(self._tree), self._root_id, self._root_visit_total()

    def _root_visits_added(self, before: tuple[int, int | None, int]) -> int:
        """Simulations a chunk really completed: the root's visit growth.

        Every completed simulation backs up through exactly one root child, so
        this is the chunk's real node count. When the chunk searched a
        different root than the one measured before it — a fresh tree, or a
        fresh root the C runner allocated because the carried one was narrowed
        (`searchmoves`, winning-root draw prune), lost a legal action, or no
        longer matched the root history context — the new root's visits all
        came from this chunk, so count them from 0."""
        tree_id, root_id, visits = before
        if id(self._tree) != tree_id or self._root_id != root_id:
            return self._root_visit_total()
        return max(0, self._root_visit_total() - int(visits))

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
        if self._tree is None or self._root_id is None:
  # No tree to read: the C Gumbel path declined this root (see the
  # `root_ids[0] < 0` note in `_run_gumbel_chunk`). The cached root prior is
  # still in memory and names a far better move than "first legal" — but the
  # move is UNSEARCHED, so it must not be reported as if it were searched.
  # `nodes` is 0 (the caller has already discarded the fabricated chunk count)
  # and `root_declined` carries the reason up to `Engine`, which prints it.
            policy_move = self._root_policy_move(board, allowed_root_indices)
            return SearchResult(
                bestmove_uci=policy_move.uci() if policy_move is not None else "0000",
                ponder_uci=None,
                nodes=total_nodes,
                pv=(policy_move.uci(),) if policy_move is not None else (),
                score_cp=q_to_cp(0.5 * (last_value + 1.0)),
                tbhits=tb_probe.hits if tb_probe is not None else 0,
                score_mate=None,
                root_declined=_declined_root_reason(board),
            )
        actions, visits = self._filtered_root_visits(allowed_root_indices)
        bestmove_idx = self._emitted_action(actions, visits, allowed_root_indices)
        pv_indices = (
            _pv_from_root_action(self._tree, self._root_id, bestmove_idx)
            if bestmove_idx >= 0 else []
        )
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
            score_mate=(
                _proven_line_mate(self._tree, self._root_id, board, bestmove_idx)
                if bestmove_idx >= 0 else None
            ),
        )

    def _root_solved_status(self) -> int:
        tree, root_id = self._tree, self._root_id
        if tree is None or root_id is None:
            return 0
        return int(tree.get_solved_status(root_id))

    def _proven_mate_within(
        self, board: chess.Board, mate_moves: int,
        allowed_root_indices: set[int] | None,
    ) -> int | None:
        """The shortest proven mate (in moves) at the root if it is <= ``mate_moves``,
        else None. Free unless the root itself is proven won; then one shared
        `_MateWalk` budget for all root actions, and no walk at all unless the
        root's solved status or the tree's size changed since the last check
        (an idle solved root -- nothing left to expand -- would otherwise re-walk
        every `_IDLE_CHUNK_WAIT_S`)."""
        self._mate_witness_action = None
        tree, root_id = self._tree, self._root_id
        if tree is None or root_id is None:
            return None
        status = int(tree.get_solved_status(root_id))
        key = (id(tree), int(root_id), status, int(tree.node_count()))
        if key == self._mate_check_key:
            return None
        self._mate_check_key = key
        if status != 1:
            return None
        actions, _ = tree.get_children_visits(root_id)
        walk = _MateWalk()
        best: int | None = None
        witness: int | None = None
        for action in actions.tolist():
            if allowed_root_indices is not None and int(action) not in allowed_root_indices:
                continue
            mate = _proven_line_mate(tree, root_id, board, int(action), walk)
            if mate is not None and mate > 0 and (best is None or mate < best):
                best = mate
                witness = int(action)
        self.last_mate_walk_visits = walk.visits
        if best is not None and best <= mate_moves:
            self._mate_witness_action = witness
            return best
        return None

    def _should_stop_search(
        self,
        *,
        stop_event: threading.Event, deadline: Deadline,
        max_nodes: int | None, max_depth: int | None,
        total_nodes: int, pv_len: int,
        time_margin_ms: float = 0.0,
        tree_growth_bytes: int = 0,
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

        ``tree_growth_bytes`` is the footprint the previous chunk added; the
        ``"tree_bytes"`` stop fires once the footprint PLUS that projected
        next-chunk growth would reach the Hash cap, so the resident tree stays
        under Hash instead of overshooting it by one chunk.
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
            and self._tree_footprint_bytes() + max(0, int(tree_growth_bytes))
            >= self._max_tree_bytes
        ):
            return "tree_bytes"
        if self._tree_node_count() >= _TREE_NODE_ID_LIMIT:
            return "tree_ids"
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
        if allowed_root_indices is None and self._rpg_pool is not None:
            # Root-parallel Gumbel re-runs a FULL sequential-halving schedule
            # per chunk. Tiny chunks (e.g. 512) force first-phase
            # visits_per_action ≈ 5 on 20 candidates — GPU batches of size 5
            # and near-zero multi-GPU speedup. Prefer one large schedule so
            # early-phase vpa fills gather-sized batches (at 8k sims, vpa≈80
            # on 20 cands). The run loop still min()s with remaining max_nodes
            # / time-capped chunk; workers poll stop_event between gather
            # batches so clock latency stays ~one batch, not ~one schedule.
            n_groups = max(1, int(self._rpg_pool.n_groups))
            gather = max(1, int(getattr(self._rpg_pool, "gather", 256) or 256))
            # Aim for first-phase vpa ≳ gather/2: budget ≳ n_cands * rounds * vpa
            # ≈ 20 * 5 * (gather/2). Use a generous floor; remaining-node cap
            # trims this for short go nodes searches.
            target = max(
                self._chunk_sims * max(8, n_groups * 4),
                20 * 5 * max(gather // 2, 64),
            )
            return int(target)
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
  #
  # Exception: root-parallel Gumbel *requires* a large first schedule or
  # early-phase vpa collapses (vpa≈5 at 512 sims → GPU batch starvation).
  # Cap only by bootstrap time, not by base _chunk_sims.
            boot_cap = int(_BOOTSTRAP_NPS_PER_MS * remaining_ms)
            time_bound = max(_MIN_FIRST_CHUNK, boot_cap)
            # getattr: unit tests may construct SearchWorker via
            # object.__new__ without running __init__.
            if getattr(self, "_rpg_pool", None) is not None:
                return min(chunk, time_bound)
            return min(chunk, self._chunk_sims, time_bound)
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

        - Slots already reserved but unused (``node_capacity - node_count``) are
          already resident (zero-filled on growth) and counted in
          ``footprint_bytes()``, so filling them costs only their child edges —
          they are always usable, even when the tree already sits at the Hash
          cap. This is what keeps a small ``Hash`` from starving the very first
          chunk to an unvisited root.
        - Growing *beyond* the reservation adds ``node_bytes`` (+ the CBoard
          slot index when that cache is on) per slot to the footprint, so that
          part is capped by the remaining Hash budget, with each slot priced
          at its full node + child-edge cost. ``reserve()`` grows in steps of at
          most ``_tree_growth_step_nodes``, so the cap stays soft by at most one
          step, but we never grow past it on purpose."""
        tree = self._tree
        if tree is None:
            return int(upcoming_sims)
        sims = max(0, int(upcoming_sims))
        node_count = tree.node_count()
        node_cap = tree.node_capacity()
  # int32 node/child ids: never reserve past the id limit (C refuses anyway).
        sims = min(sims, max(0, _TREE_NODE_ID_LIMIT - node_count) // _TREE_MAX_LEAF_BRANCHING)
        if sims <= 0:
            return 0
        free = max(0, node_cap - node_count)
  # One sim expands one leaf into up to _TREE_MAX_LEAF_BRANCHING nodes (256 >
  # 218, the absolute max legal moves), so a chunk adds at most this many nodes.
        need = sims * _TREE_MAX_LEAF_BRANCHING
        if self._max_tree_bytes > 0 and need > free:
            stats = tree.memory_stats()
            used = int(tree.footprint_bytes())
            remaining = self._max_tree_bytes - used
            bytes_per_new_node = int(stats["node_bytes"]) + int(stats["child_bytes"]) + (
                int(stats["cb_slot_bytes"]) if int(stats["cb_slot_cap"]) > 0 else 0
            )
            growable = int(remaining // bytes_per_new_node) if remaining > 0 else 0
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
  # `< 0` is defence in depth: `_run_gumbel_chunk` normalises the C sentinel to
  # None, and a negative id here would raise `node_id out of range` in C.
        if self._tree is None or self._root_id is None or self._root_id < 0:
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
        leave the survivor unset and select by visits. A known WIN/DRAW proof
        takes priority over a survivor or visit leader that would discard it."""
        if actions.size == 0:
            return -1
        proof_mask = (
            _proof_preserving_mask(self._tree, self._root_id, actions)
            if self._tree is not None and self._root_id is not None else None
        )
        gi = self._last_gumbel_action_idx
        if (
            gi is not None
            and self._tree is not None
            and self._root_id is not None
            and (allowed_root_indices is None or int(gi) in allowed_root_indices)
            and self._tree.find_child(self._root_id, int(gi)) != -1
            and (proof_mask is None or bool(np.any(actions[proof_mask] == int(gi))))
        ):
            return int(gi)
        if proof_mask is not None:
            actions, visits = actions[proof_mask], visits[proof_mask]
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
  # Exactly one legal (or `searchmoves`) move: nothing to decide, so bank now.
  # This used to fall through to `remaining_ms <= 0` and spend the whole soft
  # target (4.7 s of a 60 s clock) on a move that cannot change. `optimum_ms` is
  # None for ponder / infinite / movetime / nodes searches (checked above), so
  # those still run until stop / ponderhit / their own limit, as UCI requires.
        if self._filtered_root_visits(allowed_root_indices)[0].size == 1:
            return True
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
        stop_sensitive: bool = False,
        fixed_time_margin: bool = False,
        mate_moves: int | None = None,
    ) -> SearchResult:
        """Search until any of: stop_event set, deadline expired, max_nodes hit,
        PV length ≥ max_depth, or (``optimum_ms`` set) the visit-margin abort
        fires (see ``_abort_ready``).

        Returns when at least one chunk has run (so bestmove is always
        backed by MCTS data, never a raw priors pick).

        ``fixed_time_margin``: apply the batch-time stop margin to a deadline
        with no ``optimum_ms`` too (UCI ``movetime``), so the search stops one
        measured chunk BEFORE the deadline instead of overrunning it by up to a
        chunk. Off by default: other callers keep their exact behaviour.

        ``mate_moves`` (UCI ``go mate N``): also stop as soon as a mate in
        <= N moves is proven at the root (see ``_proven_line_mate``).
        """
        fen = board.fen()
  # A compaction `advance_root` scheduled runs here, on this move's clock.
  # `stop_sensitive`: ponder / `go infinite`, where an uninterruptible
  # compaction would delay the answer to `stop` / `ponderhit`.
        self._resolve_pending_compaction(
            deadline, optimum_ms, max_nodes=max_nodes, stop_sensitive=stop_sensitive,
        )
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

  # No legal move at the root: checkmate or stalemate. Answer immediately with
  # the UCI null move. Without this the classic-Gumbel path leaves the root id
  # at -1 and raises `node_id out of range`, while the walker pool descends
  # every walker straight into the terminal root until the deadline expires and
  # reports tens of thousands of "nodes" for zero tree growth.
        if not any(board.legal_moves):
            return SearchResult(
                bestmove_uci="0000",
                ponder_uci=None,
                nodes=0,
                pv=(),
                score_cp=0,
                tbhits=tb_probe.hits if tb_probe is not None else 0,
                score_mate=0 if board.is_checkmate() else None,
            )

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
        self.last_stop_reason = None
        self._mate_check_key = None
        self._mate_witness_action = None
        self._ensure_root_eval_cached(board, tb_probe)
        if allowed_root_indices is not None and self._rpg_pool is not None:
            # Convert an RPG-shaped tree once for this restricted search.
            # Later classic chunks must retain their own tree progress. Keep
            # the current root heads for the declined-root prior fallback.
            pol, wdl, key = self._root_pol_logits, self._root_wdl_logits, self._root_pol_key
            self.reset_tree()
            self._root_pol_logits, self._root_wdl_logits = pol, wdl
            self._root_pol_key = key
        _rate_nodes_before = self._tree_node_count()
        _rate_t0 = time.monotonic()
        self._pre_expand_root_for_pool(
            board,
            allowed_root_indices,
            allow_terminal_shortcuts=allow_terminal_shortcuts,
        )
        # Lc0-style visit-dependent Cpuct (no-op when factor<=0).
        if self._tree is not None:
            self._tree.set_cpuct_scaling(
                float(getattr(self._cfg, "cpuct_factor", 0.0) or 0.0),
                float(getattr(self._cfg, "cpuct_base", 38739.0) or 38739.0),
            )

        total_nodes = 0
        last_info_ms = -1
  # Track whether any info line went out (periodic tick, tree-full stop, ...).
        info_emitted = [False]
        if info_cb is not None:
            outer_info_cb = info_cb

            def _tracking_info_cb(**fields: Any) -> None:
                info_emitted[0] = True
                outer_info_cb(**fields)

            info_cb = cast("InfoCallback", _tracking_info_cb)
        last_value = 0.0
        pv_indices: list[int] = []
        elapsed = 0
  # Starting budget for this move — caps the batch-time margin so a bad early
  # estimate can't reserve away the whole move (see `_MAX_BATCH_MARGIN_FRACTION`).
        start_remaining_ms = deadline.remaining_ms()
        witness_checks_remaining = _SOLVED_ROOT_TB_CHECK_LIMIT
        witness_checks_total = 0
  # A time-managed search of a forced root banks after its first chunk (see
  # `_abort_ready`); keep that chunk small — it only has to produce the info
  # line's score/PV, not spend a full chunk of evaluations on one move.
        forced_root = optimum_ms is not None and _root_move_count(
            board, allowed_root_indices,
        ) == 1

        def witness_stop() -> bool:
            nonlocal witness_checks_remaining
            if stop_event.is_set() or deadline.expired() or witness_checks_remaining <= 0:
                return True
            witness_checks_remaining -= 1
            return False

        while True:
  # Hash-capped growth step on whatever tree the previous chunk left (the
  # classic Gumbel path creates its tree inside the first chunk).
            self._sync_tree_growth_step()
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
            if forced_root:
                chunk = min(chunk, _FORCED_ROOT_CHUNK_SIMS)

  # Shared-tree (walker / multi-GPU pucv) chunks descend lock-free from N
  # threads; grow the arena now, while quiescent, so the chunk can't realloc
  # mid-descent (use-after-free). The pre-grow is bounded by the Hash cap and
  # returns the sims that fit; 0 means we're at the cap, so stop (same final
  # info as the tree_bytes stop) rather than grow past the user's setting.
            if self._is_shared_tree_path(allowed_root_indices):
                chunk = self._ensure_shared_tree_headroom(chunk)
                if chunk <= 0:
                    tree_now = self._tree
                    self.last_stop_reason = (
                        "tree_ids"
                        if tree_now is not None
                        and int(tree_now.node_count()) + _TREE_MAX_LEAF_BRANCHING > _TREE_NODE_ID_LIMIT
                        else "tree_bytes"
                    )
                    if info_cb is not None and self._root_id is not None:
                        self._emit_pv_info(
                            info_cb, board, float(last_value), total_nodes,
                            deadline.elapsed_ms(), tb_probe,
                            allowed_root_indices=allowed_root_indices,
                            note=self._tree_full_note(
                                deadline.elapsed_ms(), deadline, self.last_stop_reason,
                            ),
                        )
                    break

            _chunk_t0 = time.monotonic()
            _footprint_before = self._tree_footprint_bytes()
            _visits_before = self._root_visit_snapshot()
            try:
                last_value, completed = self._run_one_chunk(
                    chunk, board, stop_event, tb_probe, allowed_root_indices,
                    allow_terminal_shortcuts=allow_terminal_shortcuts,
                    witness_stop_fn=witness_stop,
                )
            except SolvedRootWitnessUnavailable as exc:
                self._tree = exc.partial_result[4]
                rid = int(exc.partial_result[5][0])
                self._root_id = rid if rid >= 0 else None
                self._last_gumbel_action_idx = None
                # Count what the interrupted native chunk really simulated
                # (root visit growth), not the requested chunk.
                # An incomplete chunk is not yet an exhausted whole go.
                _witness_grown = self._root_visits_added(_visits_before)
                total_nodes += _witness_grown
                witness_checks_total += exc.child_checks.get(0, 0)
                exc.child_checks[0] = witness_checks_total
                if _witness_grown > 0:
                    self._record_batch_ms((time.monotonic() - _chunk_t0) * 1000.0)
                if on_chunk is not None:
                    on_chunk(total_nodes)
                pv_indices, _, _ = self._maybe_emit_pv_info(
                    board=board, deadline=deadline, last_value=0.0,
                    total_nodes=total_nodes, info_cb=None, max_depth=max_depth,
                    last_info_ms=last_info_ms, tb_probe=tb_probe,
                    allowed_root_indices=allowed_root_indices,
                )
                if self._should_stop_search(
                    stop_event=stop_event, deadline=deadline,
                    max_nodes=max_nodes, max_depth=max_depth,
                    total_nodes=total_nodes, pv_len=len(pv_indices),
                    time_margin_ms=self._clock_time_margin_ms(optimum_ms, start_remaining_ms, fixed_time=fixed_time_margin),
                ) is not None:
                    raise
                # Same zero-progress rule as a normal chunk (below): an
                # interrupted chunk that simulated nothing cannot move a node /
                # depth bound, so retrying would spin forever. Re-raise, which
                # ends the go with the engine's witness fallback move (the
                # pre-#1265 terminal behaviour); without a bound, idle on the
                # stop event instead of spinning a core.
                if _witness_grown == 0:
                    if max_nodes is not None or max_depth is not None:
                        raise
                    if not stop_event.is_set():
                        stop_event.wait(_IDLE_CHUNK_WAIT_S)
                # No preserving move exists yet to justify a visit-margin
                # abort or a proven-value PV. Continue within the original cap.
                continue
  # Time the (un-interruptible) chunk so the batch-time clock margin below can
  # reserve one worst-case batch before the hard deadline. A chunk that did no
  # work (declined/forced root) took ~0 ms and would drag the batch-time
  # statistics toward zero, shrinking the margin for the real chunks.
            if int(completed) > 0:
                self._record_batch_ms((time.monotonic() - _chunk_t0) * 1000.0)
  # Count the sims actually completed, not the requested chunk: a pool chunk
  # cut short by stop_event would otherwise inflate reported nodes AND the nps
  # estimate `_time_capped_chunk` sizes later chunks with.
            total_nodes += int(completed)
  # The C Gumbel path declines roots `CBoard.is_game_over()` reports (claimable
  # draws included) and returns no root id — see `_run_gumbel_chunk`. There is
  # nothing to search or read, and another chunk would decline identically, so
  # stop now and let `_build_final_search_result` answer from the cached root
  # prior instead of walking a -1 node id.
            if self._root_id is None:
  # A declined root ran no simulation; `_run_one_chunk` now reports the root's
  # real visit growth, but keep the reset explicit — 0 is the truth, and it is
  # what makes this move findable in a banked per-move CSV.
                total_nodes = 0
                break
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

            if (
                allowed_root_indices is None
                and self._rpg_pool is not None
                and self._rpg_pool.last_stats().root_finished
            ):
                # Exact root / single-action completion is independent of
                # simulations. Do not spin until a budget or deadline expires.
                break

  # Batch-time-variance clock margin: clock games only, capped so a bad early
  # estimate can't reserve away the whole move (see `_clock_time_margin_ms`).
            time_margin_ms = self._clock_time_margin_ms(optimum_ms, start_remaining_ms, fixed_time=fixed_time_margin)
            stop_reason = self._should_stop_search(
                stop_event=stop_event, deadline=deadline,
                max_nodes=max_nodes, max_depth=max_depth,
                total_nodes=total_nodes, pv_len=len(pv_indices),
                time_margin_ms=time_margin_ms,
                tree_growth_bytes=self._tree_footprint_bytes() - _footprint_before,
            )
            if stop_reason is None and mate_moves is not None:
                if self._proven_mate_within(board, mate_moves, allowed_root_indices) is not None:
                    stop_reason = "mate"
                    # Preserve the move whose proof satisfied this mate bound.
                    # Outcome-only selection can otherwise emit another WIN
                    # with a longer mate (or only a tablebase WDL proof).
                    assert self._mate_witness_action is not None
                    allowed_root_indices = {self._mate_witness_action}
                elif int(completed) == 0 and self._root_solved_status() != 0:
                    # `go mate N` on a root already proven (won with no mate in
                    # <= N found, drawn, or lost) that the search cannot grow:
                    # no later chunk can prove anything new, so idling to the
                    # implicit cap would only burn the clock.
                    stop_reason = "mate_solved_root"
            if stop_reason is not None:
                self.last_stop_reason = stop_reason
  # Tree-bytes stops emit a final info line whose `string` names the reason —
  # a memory stop before the time budget is spent must never be silent. Other
  # stop reasons either already emitted via _maybe_emit_pv_info or want quiet exit.
                if stop_reason in ("tree_bytes", "tree_ids") and info_cb is not None:
                    assert self._root_id is not None
                    self._emit_pv_info(
                        info_cb, board, float(last_value),
                        total_nodes, elapsed, tb_probe,
                        allowed_root_indices=allowed_root_indices,
                        note=self._tree_full_note(elapsed, deadline, stop_reason),
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
  # A chunk that completed nothing (forced root: the Gumbel path returns the
  # sole action without simulating) cannot make progress. Under a node or depth
  # bound that bound can then never be reached, so stop; otherwise (ponder /
  # infinite / movetime, which must run until stop or their deadline) idle
  # briefly on the stop event instead of spinning a core.
            if int(completed) == 0 and not stop_event.is_set():
                if max_nodes is not None or max_depth is not None:
                    break
                stop_event.wait(_IDLE_CHUNK_WAIT_S)

  # Every search that reached the tree reports at least one info line (score +
  # pv) before bestmove: a forced root banks after one chunk, long before the
  # first 1 s info tick, and GUIs / broadcasts expect a line per move.
        if info_cb is not None and not info_emitted[0] and self._root_id is not None:
            self._emit_pv_info(
                info_cb, board, float(last_value), total_nodes,
                deadline.elapsed_ms(), tb_probe,
                allowed_root_indices=allowed_root_indices,
            )

        self._record_growth_rate(
            total_nodes, _rate_nodes_before, (time.monotonic() - _rate_t0) * 1000.0,
        )
        return self._build_final_search_result(
            board=board, total_nodes=total_nodes,
            last_value=last_value, tb_probe=tb_probe,
            allowed_root_indices=allowed_root_indices,
            include_ponder=include_ponder,
        )

    def _run_walker_chunk(
        self, chunk: int, stop_event: threading.Event, tb_probe=None,
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
            tb_probe=tb_probe,
        )
        return self._tree.node_q(self._root_id), int(completed)

    def _run_gumbel_chunk(
        self,
        chunk: int,
        board: chess.Board,
        tb_probe,
        allowed_root_indices: set[int] | None = None,
        allow_terminal_shortcuts: bool = True,
        witness_stop_fn: Callable[[], bool] | None = None,
    ) -> float:
        gumbel_result = run_gumbel_root_many_c(
            model=None,
            boards=[board],
            device=self._device,
            rng=self._rng,
            cfg=dataclasses.replace(
                self._cfg,
                simulations=chunk,
  # `add_noise` was hard-coded False here, which made the whole root-noise
  # mechanism unreachable from the engine. Deriving the gate from the scale
  # keeps the default (scale 0.0) bit-identical to that while giving
  # `GumbelScale` somewhere to land. `temperature` stays 0: match play picks
  # the best-scored survivor, noise only perturbs which candidates get looked
  # at, not which of them is finally played.
                add_noise=self._root_noise_scale > 0.0,
                gumbel_scale=self._root_noise_scale,
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
            witness_stop_fn=witness_stop_fn,
            target_batch=self._minibatch_size,
  # C17: without this the classic-Gumbel UCI path (`--walkers 1`) runs the
  # PRE-virtual-loss search --- a later halving rep re-walks an unchanged tree
  # back onto a leaf already in the batch and back-propagates the same value
  # again. The four PUCT/walker/pool sibling call sites above all pass
  # `self._vloss_weight`; this one silently did not, so the knob (and its UCI
  # `--vloss-weight` flag) was dead on the only path that uses Gumbel.
            vloss_weight=self._vloss_weight,
        )
        _, actions, values, _, tree, root_ids = gumbel_result[:6]
        self._tree = tree
  # `run_gumbel_root_many_c` leaves the root id at its -1 sentinel whenever it
  # declines the root — which `CBoard.is_game_over()` reports for CLAIMABLE
  # draws (halfmove>=100, threefold, insufficient material) that still have
  # legal moves and must still be played. Storing -1 verbatim made every later
  # `get_children_visits(self._root_id)` raise `ValueError: node_id out of
  # range`, and the engine answered that raise with the first legal move.
  # Normalise to None ("no root"), which every consumer here already handles;
  # `run` stops on it and `_build_final_search_result` answers from the cached
  # root prior. The action is a placeholder 0 in that case, so drop it too.
        rid = int(root_ids[0])
        if rid < 0:
            self._root_id = None
            self._last_gumbel_action_idx = None
        else:
            self._root_id = rid
            self._last_gumbel_action_idx = int(actions[0])
        return float(values[0])

    def _run_pucv_pool_chunk(
        self, chunk: int, stop_event: threading.Event, tb_probe=None,
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
            stop_event=stop_event, **({"tb_probe": tb_probe} if tb_probe is not None else {}),
        )
        return self._tree.node_q(self._root_id), int(completed)

    def _run_rpg_chunk(
        self, chunk: int, stop_event: threading.Event, tb_probe=None,
    ) -> tuple[float, int]:
        """One root-parallel Gumbel chunk. The pool runs a full sequential-
        halving schedule over ``chunk`` sims (identical budget shape to the
        classic single-GPU chunk — the per-chunk budget is deliberately NOT
        scaled by group count, so root decisions at equal total sims match
        serial Gumbel) and returns ``(root_value, completed_sims)``; the
        survivor feeds ``_last_gumbel_action_idx`` exactly like the classic
        Gumbel chunk so final selection semantics are unchanged."""
        assert self._rpg_pool is not None
        value, action = self._rpg_pool.run(
            target_sims=chunk, stop_event=stop_event, **({"tb_probe": tb_probe} if tb_probe is not None else {}),
        )
        if action >= 0:
            self._last_gumbel_action_idx = int(action)
        # Completed sims come from the pool's per-phase stats (a stop_event
        # can cut the schedule short), keeping nodes/nps accounting honest.
        stats = self._rpg_pool.last_stats()
        if stats.root_finished and action < 0:
            # The oracle declined expansion. Keep its returned value, but
            # select a legal cached policy move and report the existing
            # prior-only diagnostic, as on the classic declined-root path.
            self._root_id = None
            self._last_gumbel_action_idx = None
        completed = sum(p.sims_completed for p in stats.phases)
        return float(value), int(completed)

    def _ensure_rpg_root_prepared(
        self,
        board: chess.Board,
        *,
        allow_terminal_shortcuts: bool = True,
    ) -> None:
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
            allow_terminal_shortcuts=allow_terminal_shortcuts,
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
            self._root_id = int(self._tree.add_root(1, root_q))
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

    def _run_pucv_chunk(self, chunk: int, tb_probe=None) -> float:
        assert self._tree is not None
        assert self._root_id is not None
        assert self._pucv is not None
        assert self._pucv_cboard is not None
        self._pucv.run(
            tree=self._tree,
            root_id=self._root_id,
            root_cboard=self._pucv_cboard,
            target_sims=chunk, **({"tb_probe": tb_probe} if tb_probe is not None else {}),
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
            self._root_id = int(self._tree.add_root(1, root_q))
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
            self._root_id = int(self._tree.add_root(1, root_q))
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


def _proof_preserving_mask(
    tree: MCTSTree, node_id: int, actions: np.ndarray,
) -> np.ndarray | None:
    """Prefer known witnesses within the supplied support of a WIN/DRAW node."""
    status = tree.get_solved_status(node_id)
    if status not in (1, 2):
        return None
    proof = -1 if status == 1 else 2
    keep = np.fromiter(
        (tree.get_solved_status(tree.find_child(node_id, int(action))) == proof
         for action in actions),
        dtype=np.bool_, count=actions.size,
    )
    return keep if bool(keep.any()) else None


# UCI `score mate N`. The tree stores proven OUTCOMES (the solved lattice:
# WIN / LOSS / DRAW per node, from checkmate / stalemate leaves and tablebase
# hits) but no distances, so a distance is only known where the proof can be
# walked down to checkmate leaves. A WIN node takes its shortest proven
# winning child; a LOSS node needs EVERY legal move expanded and proven, and
# takes the longest. Tablebase-solved leaves carry no distance (Syzygy has
# WDL/DTZ, not DTM), so a proof that rests on one gives no `score mate`. The
# result is the length of a forced mate the search has proven against every
# defence -- an upper bound on the true distance, exactly as an alpha-beta
# engine's mate score is. One exact shortcut: a WIN node whose side to move has
# a mate in one ON THE BOARD is distance 1 even if the search never expanded
# that edge (measured: without it a mated-in-1 root read "mate -3", the proof
# having found a longer mate first).
#
# Bounded: ONE `_MateWalk` budget of `_MATE_WALK_NODE_LIMIT` node visits is
# shared by everything one report needs (every multipv line of an info
# emission; every root action of a `go mate` check), so the cost on the search
# thread is bounded per report, not per line (per-line budgets measured 10 s on a
# fully expanded synthetic tree). Once the budget runs out the walk is
# EXHAUSTED: every line it has not already finished reports no distance (cp),
# never a distance from a partial walk. Past `_MATE_WALK_MAX_PLIES` a branch
# simply has no distance.
_MATE_WALK_NODE_LIMIT = 4_000
_MATE_WALK_MAX_PLIES = 128


class _MateWalk:
    """Memo + node-visit budget shared by every line of one report."""

    __slots__ = ("budget", "exhausted", "memo", "visits")

    def __init__(self, budget: int = _MATE_WALK_NODE_LIMIT) -> None:
        self.memo: dict[int, int | None] = {}
        self.budget = int(budget)
        self.visits = 0
        self.exhausted = False


def _proven_mate_plies(
    tree: MCTSTree, node_id: int, board: chess.Board,
    walk: _MateWalk, depth: int = 0,
) -> int | None:
    """Plies to checkmate from ``node_id`` inside its proven subtree, or None.

    ``board`` is the node's position (pushed and popped in place). Only
    SOLVED_WIN (1) / SOLVED_LOSS (-1) nodes have a distance; a leaf counts only
    if it really is checkmate. Returns None once ``walk`` is exhausted."""
    if walk.exhausted:
        return None
    if node_id in walk.memo:
        return walk.memo[node_id]
    status = tree.get_solved_status(node_id)
    if status not in (1, -1):
        return None
    if depth > _MATE_WALK_MAX_PLIES:
        return None
    if walk.visits >= walk.budget:
        walk.exhausted = True
        return None
    walk.visits += 1
    walk.memo[node_id] = None  # in progress: a transposition cycle proves nothing
    actions, _ = tree.get_children_visits(node_id)
    result: int | None
    if status == 1 and _has_mate_in_one(board):
        result = 1  # exact, expanded or not (a tablebase-solved leaf included)
    elif actions.size == 0:
        result = 0 if status == -1 and board.is_checkmate() else None
    elif status == 1:
        result = None
        for action in actions.tolist():
            child = tree.find_child(node_id, int(action))
            if child < 0 or tree.get_solved_status(child) != -1:
                continue
            board.push(index_to_move(int(action), board))
            try:
                d = _proven_mate_plies(tree, child, board, walk, depth + 1)
            finally:
                board.pop()
            if d is not None and (result is None or d + 1 < result):
                result = d + 1
    elif actions.size != board.legal_moves.count():
        result = None  # a LOSS proof must cover every legal move
    else:
        result = 0
        for action in actions.tolist():
            child = tree.find_child(node_id, int(action))
            if child < 0 or tree.get_solved_status(child) != 1:
                result = None
                break
            board.push(index_to_move(int(action), board))
            try:
                d = _proven_mate_plies(tree, child, board, walk, depth + 1)
            finally:
                board.pop()
            if d is None:
                result = None
                break
            result = max(result, d + 1)
    if walk.exhausted:
        # A partial walk proves nothing about this node; drop the marker so no
        # later lookup reads it as a finished answer.
        walk.memo.pop(node_id, None)
        return None
    walk.memo[node_id] = result
    return result


def _has_mate_in_one(board: chess.Board) -> bool:
    for move in board.legal_moves:
        if board.gives_check(move):
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return True
    return False


def _proven_line_mate(
    tree: MCTSTree, root_id: int, board: chess.Board, action: int,
    walk: _MateWalk | None = None,
) -> int | None:
    """UCI ``score mate`` value for the line that starts with root ``action``:
    +N when the side to move mates in N moves along a proven line, -N when it
    is mated in N, None when the search has proven no distance for it -- or
    when ``walk`` (shared by the caller's other lines) ran out of budget."""
    walk = _MateWalk() if walk is None else walk
    if walk.exhausted:
        return None
    child = tree.find_child(root_id, int(action))
    if child < 0:
        return None
    status = tree.get_solved_status(child)
    if status not in (1, -1):
        return None
    after = board.copy(stack=False)
    after.push(index_to_move(int(action), after))
    d = _proven_mate_plies(tree, child, after, walk)  # None whenever the walk ran out
    if d is None:
        return None
    plies = d + 1
    # The child is LOSS: we deliver mate on an odd ply. WIN: we are mated on an even one.
    return (plies + 1) // 2 if status == -1 else -(plies // 2)


def _proven_line_mates(
    tree: MCTSTree | None, root_id: int | None, board: chess.Board,
    first_actions: list[int | None],
) -> list[int | None]:
    """`_proven_line_mate` for each line of one info emission (None for a line
    with no first move), all under ONE shared `_MateWalk` budget."""
    if tree is None or root_id is None:
        return [None] * len(first_actions)
    walk = _MateWalk()
    return [
        None if action is None else _proven_line_mate(tree, root_id, board, int(action), walk)
        for action in first_actions
    ]


def _child_visit_order(
    tree: MCTSTree, node_id: int, actions: np.ndarray, visits: np.ndarray,
) -> np.ndarray:
    """Rank proof-preserving children first, retaining visit order within each group."""
    order = np.argsort(-visits)
    proof_mask = _proof_preserving_mask(tree, node_id, actions)
    if proof_mask is not None:
        order = np.concatenate((order[proof_mask[order]], order[~proof_mask[order]]))
    return order


def _best_child_action(
    tree: MCTSTree, node_id: int, actions: np.ndarray, visits: np.ndarray,
) -> int:
    """Choose a witness by visits, keeping the original first-maximum tie rule."""
    proof_mask = _proof_preserving_mask(tree, node_id, actions)
    if proof_mask is not None:
        actions, visits = actions[proof_mask], visits[proof_mask]
    return int(actions[int(np.argmax(visits))])


def _multipv_lines(
    tree: MCTSTree,
    root_id: int,
    n: int,
    root_q_default: float,
    allowed_root_indices: set[int] | None = None,
) -> list[tuple[int, float, list[int]]]:
    """Return up to ``n`` (rank, q, pv_indices) triples, ranking known WIN/DRAW
    witnesses before visits. Rank is 1-based (UCI convention: ``multipv 1`` = best).
    Each pv_indices preserves known WIN/DRAW proofs before following most visits.

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
    order = _child_visit_order(tree, root_id, actions, visits)[:max(1, int(n))]
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
            nxt = _best_child_action(tree, cid, a, vs)
            pv.append(nxt)
            cid = tree.find_child(cid, nxt)
        out.append((rank, q, pv))
    return out




def _root_move_count(board: chess.Board, allowed_root_indices: set[int] | None) -> int:
    """Moves the root may play: the ``searchmoves`` set when given, else legal moves."""
    if allowed_root_indices is not None:
        return len(allowed_root_indices)
    return board.legal_moves.count()


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
    # The probe carries UCI `Syzygy50MoveRule` (as `cursed_as_draw`), so the
    # root shortcut and the leaf probes of one search follow the same rule.
    rule50 = bool(tb_probe.cursed_as_draw)
    root = try_tb_root_move(board, tb_probe._path, rule50=rule50)
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
        score_cp = 0  # draw (cursed/blessed too, unless Syzygy50MoveRule is off)

    ponder_uci = None
    if include_ponder:
        # Ponder move: after our best, what's the opponent's DTZ-optimal reply?
        # Re-runs try_tb_root_move; still cheap (a few legal-move probes).
        board_after = board.copy(stack=False)
        board_after.push(best)
        ponder = try_tb_root_move(board_after, tb_probe._path, rule50=rule50)
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


def _declined_root_reason(board: chess.Board) -> str:
    """Why `CBoard.is_game_over()` refused to expand this root.

    Mirrors `cboard_is_game_over` (`_cboard_impl.h`), which returns true for
    **claimable** draws — positions that still have legal moves and must still
    be played. Only the reason string differs between these; the consequence
    (no tree, answer from the prior) is the same, and naming it is the point:
    "threefold repetition" tells an operator the engine is shuffling unsearched,
    which is a different problem from a 50-move counter running out.

    Threefold is reached through the MOVE STACK, so it fires for any position
    the game actually repeated into — not a rare class in real match play.
    """
    if board.is_repetition(3):
        return "threefold repetition"
    if board.halfmove_clock >= 100:
        return f"50-move counter at {board.halfmove_clock}"
    if board.is_insufficient_material():
        return "insufficient material"
    return "CBoard.is_game_over()"


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
    """Proof-preserving root move by visits and its PV. ``q`` is unused at the call sites
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
    After the root action, preserve known WIN/DRAW proofs before most visits.
    """
    cid = tree.find_child(root_id, int(action_idx))
    if cid == -1:
        return []
    pv = [int(action_idx)]
    while cid != -1:
        a, vs = tree.get_children_visits(cid)
        if a.size == 0:
            break
        nxt = _best_child_action(tree, cid, a, vs)
        pv.append(nxt)
        cid = tree.find_child(cid, nxt)
    return pv


def _reply_at_child(
    tree: MCTSTree,
    root_id: int,
    action_idx: int,
) -> int | None:
    """Proof-preserving child by visits at the node reached by ``action_idx``.

    This is the opponent's predicted reply after OUR specific move,
    not after the globally most-visited root move.
    """
    cid = tree.find_child(root_id, int(action_idx))
    if cid == -1:
        return None
    a, vs = tree.get_children_visits(cid)
    if a.size == 0:
        return None
    return _best_child_action(tree, cid, a, vs)


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
