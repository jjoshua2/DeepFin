"""Engine — holds board state, a SearchWorker, and the command dispatch.

Command handlers are called from the main UCI loop in response to parsed
commands. ``go`` kicks off a background thread that runs the search, so
the main loop can stay responsive for ``stop`` / ``ponderhit``.

v1 scope: no pondering yet. Implements the ``go`` / ``stop`` / ``quit``
path end-to-end so we can play a game.
"""
from __future__ import annotations

import gc
import logging
import math
import sys
import threading
import time
import weakref
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, ClassVar, Protocol
from collections.abc import Callable

import chess

from chess_anti_engine.mcts.solved_root import SolvedRootWitnessUnavailable
from chess_anti_engine.moves import index_to_move, move_to_index
from chess_anti_engine.mcts.search_options import (
    branch_note,
    OPTIONS_BY_NAME,
    PATH_DESCRIPTIONS,
    SEARCH_OPTIONS,
    SearchOption,
    inert_reason,
    realized_rows,
)
from chess_anti_engine.tablebase import SyzygyProbe, get_tablebase
from chess_anti_engine.utils.bitboards import unsearchable_king_reason
from chess_anti_engine.utils.gil_probe import GilContentionProbe

from .protocol import (
    CmdGo,
    CmdIsReady,
    CmdPonderHit,
    CmdPosition,
    CmdQuit,
    CmdSearchConfig,
    CmdSetOption,
    CmdStop,
    CmdUci,
    CmdUciNewGame,
    Command,
    GoArgs,
    InfoFields,
    format_bestmove,
    format_id_lines,
    format_info,
    format_readyok,
    format_uciok,
)
from .search import SearchResult, SearchWorker
from .time_manager import (
    GO_IMPLICIT_CAP_MS,
    _DEFAULT_MOVES_REMAINING,
    _OPTIMUM_FRACTION,
    Deadline,
    SearchLimits,
    limits_from_go,
)

_log = logging.getLogger(__name__)

_ENGINE_NAME = "DeepFin"
_ENGINE_AUTHOR = "jjosh"

if TYPE_CHECKING:
    from chess_anti_engine.inference import BatchEvaluator


class _DeepFinLogFileHandler(logging.FileHandler):
    """Marker subclass so repeat setoption can find and replace our handler."""


def _attach_log_file(path: str) -> None:
    """Tee stderr-bound logs into ``path``. Empty path is a no-op (any
    existing FileHandler stays attached — UCI spec has no 'clear' command
    for string options, so we treat empty as 'leave as-is'). Non-empty
    replaces any prior DeepFin file handler."""
    root = logging.getLogger()
  # Our handlers are the marker subclass so repeat setoption doesn't stack duplicates.
    for h in list(root.handlers):
        if isinstance(h, _DeepFinLogFileHandler):
            root.removeHandler(h)
            try:
                h.close()
            except OSError:
                pass  # log file already closed / gone
    if not path:
        return
    try:
        fh = _DeepFinLogFileHandler(path, mode="a")
    except OSError as exc:
        _println(f"info string LogFile could not open {path!r}: {exc!r}")
        return
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    ))
    root.addHandler(fh)
    _println(f"info string LogFile attached: {path!r}")


def emit_handshake(options: EngineOptions) -> None:
    """Print the UCI `uci` response — id, options, uciok — using ``options``
    defaults. Kept out of the Engine class so ``__main__`` can reply to the
    GUI before model load finishes (standard Lc0 pattern: the handshake is
    static, ``readyok`` still waits for real readiness)."""
    for line in format_id_lines(_ENGINE_NAME, _ENGINE_AUTHOR):
        _println(line)
    _println(
        f"option name Hash type spin default {options.hash_mb} {_spin_range('Hash')}"
    )
    _println(f"option name Threads type spin default {options.threads} {_spin_range('Threads')}")
    _println(
        f"option name SearchWalkers type spin default {options.search_walkers} "
        f"{_spin_range('SearchWalkers')}"
    )
    _println(
        f"option name LeafGather type spin default {options.leaf_gather} "
        f"{_spin_range('LeafGather')}"
    )
    _println(f"option name UseVL type check default {'true' if options.use_vl else 'false'}")
    _println(
        "option name UseMultiGpuPUCV type check default "
        f"{'true' if options.use_multi_gpu_pucv else 'false'}"
    )
    _println(
        "option name PUCVPendingMode type combo default "
        f"{options.pucv_pending_mode} var legacy var virtual-mean"
    )
    _println(
        "option name SearchParallel type combo default "
        f"{options.search_parallel} var pucv var gumbel"
    )
    # Root-parallel Gumbel multi-GPU schedule (ignored unless SearchParallel=gumbel).
    # -1 = auto (open/min_vpa→VLGather, min_keep→#devices); 0 = off; >0 = fixed.
    _println(
        f"option name RPGOpenVPA type spin default {options.rpg_open_vpa} "
        f"{_spin_range('RPGOpenVPA')}"
    )
    _println(
        f"option name RPGMinVPA type spin default {options.rpg_min_vpa} "
        f"{_spin_range('RPGMinVPA')}"
    )
    _println(
        f"option name RPGMinKeep type spin default {options.rpg_min_keep} "
        f"{_spin_range('RPGMinKeep')}"
    )
    _println(
        "option name RPGOpenBudgetFrac type string default "
        f"{options.rpg_open_budget_frac}"
    )
    _println(f"option name VLGather type spin default {options.vl_gather} {_spin_range('VLGather')}")
    _println(f"option name MaxBatch type spin default {options.max_batch} {_spin_range('MaxBatch')}")
    _println(
        f"option name EvalCacheEntries type spin default {options.eval_cache_entries} "
        f"{_spin_range('EvalCacheEntries')}"
    )
    _println(f"option name MultiPV type spin default {options.multi_pv} {_spin_range('MultiPV')}")
    _println(f"option name UCI_ShowWDL type check default {'true' if options.show_wdl else 'false'}")
    _println(
        f"option name MoveOverheadMs type spin default {options.move_overhead_ms} "
        f"{_spin_range('MoveOverheadMs')}"
    )
    _println(
        "option name ClockBatchMarginSigmas type string default "
        f"{options.clock_batch_margin_sigmas}",
    )
    _println("option name SyzygyPath type string default <empty>")
    _println(f"option name Syzygy50MoveRule type check default {'true' if options.syzygy_50_move_rule else 'false'}")
    _println("option name LogFile type string default <empty>")
    _println(f"option name Ponder type check default {'true' if options.ponder else 'false'}")
  # Search-parameter surface (mcts/search_options.py). Declared from the ONE
  # registry so the advertised name/type/default cannot drift from the handler
  # that parses it or from the `searchconfig` readback that reports it.
  # MinibatchSize / CPuct / CPuctFactor / CPuctBase are declared there too, so
  # they are deliberately absent from the hand-written block above.
    for opt in SEARCH_OPTIONS:
        _println(opt.declaration(options.search_value(opt.field)))
    _println(format_uciok())

# Tight (e.g. 5s) timeouts let a slow stop orphan a thread that emits a
# stale bestmove against the next board — a cold CUDA-graph compile on
# the first search is ~3-4s on its own.
_JOIN_TIMEOUT_S = 30.0
# A `go` waits for the worker at most this fraction of its own hard deadline
# (clamped to [_GO_WAIT_MIN_S, _JOIN_TIMEOUT_S]), or _GO_WAIT_NO_CLOCK_S with no
# clock, then answers from the busy-worker fallback. Normally the previous
# search has already exited and the wait is zero. On a very short clock the
# floor itself shrinks to _GO_WAIT_MAX_SHARE of the deadline, so the wait plus
# a fallback's fresh eval (<= a tenth of what is left) stays inside the clock.
_GO_WAIT_FRACTION = 0.1
_GO_WAIT_MIN_S = 0.1
_GO_WAIT_MAX_SHARE = 0.25
_GO_WAIT_NO_CLOCK_S = 1.0
# The one fresh root eval a fallback may run: at most this long (and at most a
# tenth of the remaining clock), then first-legal. A hung evaluator must not
# withhold the bestmove.
_FRESH_EVAL_TIMEOUT_S = 0.15
# Deadline watchdog (see `_DeadlineWatchdog`): it answers a clocked `go` the
# search is not going to answer in time, and leaves a search that is making
# progress alone. Two thresholds, both from the moment the clock started:
#   EARLY (needs no progress): past the search's own hard deadline AND no chunk
#     completed for S = clamp(K x chunk_ms, S_FLOOR, S_CEIL) ms. chunk_ms is the
#     slowest chunk this search has completed (inter-chunk wall time); before
#     its second chunk, recent healthy searches' slowest chunk; before its
#     FIRST chunk, recent healthy searches' phase-start-to-first-chunk time
#     (root eval, tree setup and compaction included, typically 2-4x a chunk).
#     "Recent" is a running max that decays x0.9 per search, so
#     one quick search does not shrink the scale. With no previous search,
#     S = _WATCHDOG_STALL_NO_ESTIMATE_MS. A
#     search the watchdog answered (silenced) never sets these. A healthy search
#     completes a chunk every ~chunk_ms and stops at the first chunk boundary
#     after the deadline, so it is at most one chunk from a heartbeat when it
#     answers.
#   LAST MOMENT (unconditional): the move's real budget (movetime, or the
#     remaining clock) minus M = _WATCHDOG_LAST_MOMENT_MS, never before the
#     search's own deadline. M is the watchdog's wake-up latency plus its
#     answer: wake latency measured during real CPU searches is p99 0.37 /
#     max 0.73 ms, and p99 2.2 ms beside a Python thread holding the GIL half
#     the time; the answer (snapshot or cached-prior read, two lines) is under
#     0.5 ms. A search that has not answered by then would answer in the last
#     M ms of its budget at best, and the watchdog's answer is out no later.
#     A hang is normally caught much earlier, by the EARLY rule.
_WATCHDOG_STALL_CHUNKS = 4.0
_WATCHDOG_STALL_FLOOR_MS = 300.0
_WATCHDOG_STALL_CEIL_MS = 2000.0
_WATCHDOG_STALL_NO_ESTIMATE_MS = 1000.0
_WATCHDOG_SCALE_DECAY = 0.9
_WATCHDOG_LAST_MOMENT_MS = 3.0
# `close()` waits this long for a search / helper; a thread still alive after
# that is left to `__main__`, which hard-exits instead of finalizing under it.
_SHUTDOWN_JOIN_S = 3.0


# (options snapshot, `SearchWorker.search_state()` or None for a test double)
_SetoptionSnapshot = tuple["EngineOptions", object]

# Consecutive fallback moves that mark the engine unhealthy (see
# `_note_fallback_phase`). Every published unsearched move counts: a raised
# search, a solved root without a witness, a ponder transition past its
# deadline, a reply-only answer for a `go` whose handler raised, and a
# busy-worker answer. Nothing is rebuilt mid-game; an unhealthy engine keeps
# answering every `go` and is rebuilt and re-checked at the next game boundary.
_UNHEALTHY_FALLBACK_STREAK = 3


class SearchNotQuiescent(TimeoutError):
    """The search still owns the worker; mutation and teardown must wait."""


def _one_line(exc: BaseException, limit: int = 300) -> str:
    """``Type: message`` on one line, for an ``info string``.

    A GUI parses stdout line by line; a multi-line exception message (torch's
    are) would put free text on lines that do not start with a UCI keyword.
    """
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= limit else text[: limit - 3] + "..."

# Status flags of an UNREACHABLE but playable position: legal moves exist and
# the king predicate holds, but the C move generator's assumptions are not
# guaranteed, so `_validated_position` accepts these only when the C board and
# python-chess agree on the legal-move set (`_movegen_disagreement`).
_TOLERATED_STATUS = (
    chess.STATUS_TOO_MANY_WHITE_PAWNS
    | chess.STATUS_TOO_MANY_BLACK_PAWNS
    | chess.STATUS_TOO_MANY_WHITE_PIECES
    | chess.STATUS_TOO_MANY_BLACK_PIECES
    | chess.STATUS_IMPOSSIBLE_CHECK
    | chess.STATUS_TOO_MANY_CHECKERS
)


def _status_names(status: int) -> str:
    return ", ".join(
        str(flag.name).lower().replace("_", " ")
        for flag in chess.Status
        if flag and flag & status
    )


def _movegen_disagreement(board: chess.Board) -> str | None:
    """None when ``CBoard`` and python-chess generate the same legal moves for
    ``board`` (as 4672 action ids), else a one-line description of the diff."""
    from chess_anti_engine.encoding._lc0_ext import CBoard

    try:
        c_ids = {int(i) for i in CBoard.from_board(board).legal_move_indices()}
    except Exception as exc:
        return f"the C board cannot load it ({_one_line(exc)})"
    py_ids = {move_to_index(mv, board) for mv in board.legal_moves}
    if c_ids == py_ids:
        return None
    return (
        f"the C move generator disagrees with python-chess "
        f"({len(c_ids)} vs {len(py_ids)} legal moves)"
    )


# Advertised (and now enforced) floor for the `Hash` UCI option, in MB. Keep this
# in step with the `min` field `emit_handshake` prints — the two disagreeing is
# exactly the defect this constant closes.
_MIN_HASH_MB = 1024

# The ONE table of hand-written spin-option bounds. `emit_handshake` prints these
# numbers as `min`/`max` and every setoption handler clamps to them, so what a GUI
# is told and what the engine accepts cannot drift. The maxima used to be printed
# and never enforced: `MaxBatch 50000000` reached the evaluator factory, tried to
# allocate 2.2 TB of pinned input and killed the process. Registry search options
# (mcts/search_options.py) carry their own lo/hi and are range-checked by
# `_parse_search_value`; they are deliberately not repeated here.
_SPIN_BOUNDS: dict[str, tuple[int, int]] = {
    "Hash": (_MIN_HASH_MB, 524288),
    "Threads": (1, 64),
    "SearchWalkers": (1, 64),
    "LeafGather": (1, 64),
    "RPGOpenVPA": (-1, 4096),
    "RPGMinVPA": (-1, 4096),
    "RPGMinKeep": (-1, 64),
    "VLGather": (32, 4096),
    "MaxBatch": (64, 8192),
    "EvalCacheEntries": (0, 1048576),
    "MultiPV": (1, 256),
    "MoveOverheadMs": (0, 5000),
}


def _spin_range(name: str) -> str:
    lo, hi = _SPIN_BOUNDS[name]
    return f"min {lo} max {hi}"

# The one registry field whose EngineOptions attribute does not match its name:
# `CPuct` predates the registry and stores into `cpuct`. Renaming the attribute
# would churn `__main__` and three tests for nothing.
_SEARCH_ATTR: dict[str, str] = {"c_puct": "cpuct"}


def _search_default(field: str) -> float:
    """The registry's default for ``field``, as an EngineOptions field default.

    Sourced rather than retyped on purpose: a retyped default silently drifts
    from the code default, and then the handshake advertises a number the
    engine is not running.
    """
    for opt in SEARCH_OPTIONS:
        if opt.field == field:
            return float(opt.default)
    raise KeyError(field)


@dataclass
class EngineOptions:
  # Soft cap on MCTS tree memory in MB. Allocated lazily — the tree only
  # grows to what a search actually needs, so this is a ceiling, not a
  # reservation. Tree reuse refuses to extend a tree past half this value
  # (see ``advance_root``), rebuilding fresh instead, so memory stays
  # bounded without the mid-game search collapse that hitting the hard cap
  # would cause. It is NOT a bounded transposition table; we rebuild rather
  # than evict.
    hash_mb: int = 16384
  # The GUI-facing UCI `Threads` value. Stored and echoed, NEVER read by the
  # search: a GUI sends it from its own CPU-thread setting, and letting it pick
  # the search algorithm moved tournament play off the validated Gumbel path.
    threads: int = 1
  # Number of MCTS walker threads. 1 = classic Gumbel path (the validated
  # tournament search); >1 = PUCT walker pool with virtual loss, where most
  # tuned Gumbel knobs are inert. Operator-only: `--walkers` / `SearchWalkers`.
    search_walkers: int = 1
  # Per-walker leaf gather: each walker does G descents → one NN batch
  # instead of one descent per submit. Amplifies effective batch to
  # N_walkers×G without spawning more threads. 1 = classic. Lc0's
  # canonical value is 8. Set via UCI `LeafGather`.
    leaf_gather: int = 1
  # Single-thread async-pipeline batched-VL PUCT. Active only at
  # `SearchWalkers=1` with a 2-slot inplace-async evaluator. Bench: +112%
  # over sync gumbel walkers=1 (54k vs 26k nps on a 10-layer model).
  # Set via UCI `UseVL`.
    use_vl: bool = False
  # Multi-GPU shared-tree PUCT. Active only when the executable supplied
  # per-device evaluator factories; otherwise the option is accepted but
  # cannot install a pool. Set via UCI `UseMultiGpuPUCV`.
    use_multi_gpu_pucv: bool = False
  # Pending accounting for batched PUCV paths. `legacy` keeps the existing
  # absolute virtual-loss scoring; `virtual-mean` treats in-flight leaves as
  # virtual-mean samples. Set via UCI `PUCVPendingMode`.
    pucv_pending_mode: str = "legacy"
  # Which multi-GPU search mode drives multi-device searches: `pucv` =
  # shared-tree PUCT+virtual-loss pool (throughput baseline), `gumbel` =
  # root-parallel Gumbel over evaluator groups (quality-preserving; the
  # validated sequential-halving root semantics). Requires >= 2 devices;
  # single-device configs always run the classic paths. Set via UCI
  # `SearchParallel`.
    search_parallel: str = "pucv"
  # Root-parallel Gumbel multi-GPU util schedule (SearchParallel=gumbel only).
  # UCI: RPGOpenVPA / RPGMinVPA / RPGMinKeep / RPGOpenBudgetFrac.
  # -1 = auto (open & min_vpa → VLGather, min_keep → n_groups); 0 = off;
  # >0 = fixed. Fat auto defaults close ~20% of the RPG↔PUCV nps gap vs thin SH.
    rpg_open_vpa: int = -1
    rpg_min_vpa: int = -1
    rpg_min_keep: int = -1
    rpg_open_budget_frac: float = 0.50
  # Lc0-style Cpuct scaling (PUCT descent). factor<=0 → fixed cpuct.
  # Live via UCI CPuct / CPuctFactor / CPuctBase; also set on SearchWorker._cfg.
    # Lc0 classic defaults (play).
    cpuct: float = 1.75
    cpuct_factor: float = 3.89
    cpuct_base: float = 38739.0
  # Sims per pipeline submit when `UseVL=true`. Sweet spot 384-768 for
  # the 384-dim 10-layer model on RTX 5090. Set via UCI `VLGather`.
    vl_gather: int = 512
  # Hardware-side max batch: the largest leaf-batch shape the evaluator
  # allocates buffers + CUDA-graph captures for. Changing it rebuilds
  # the evaluator + re-warmup (5-10s stall). Set via UCI `MaxBatch`.
    max_batch: int = 1024
  # Encoded eval-cache size. 0 = disabled. Covers plain evaluate_encoded
  # and the single-thread PUCV in-place path.
    eval_cache_entries: int = 0
  # Minibatch target the C gumbel state machine aims for before flushing
  # leaves to GPU eval. 0 = C-side default (GSS_GPU_BATCH = 1024). Live
  # update; takes effect on the next chunk.
    minibatch_size: int = 0
  # Number of top-ranked lines to emit per info tick. 1 = classic single
  # PV (no `multipv` field). >1 emits N lines each tagged `multipv k`.
    multi_pv: int = 1
  # Emit a `wdl W D L` field (per-mille) on each info line. Derived
  # per-line from that line's Q plus a shared draw-rate estimate from
  # the root NN evaluation. Off by default to keep info strings compact.
    show_wdl: bool = False
  # Milliseconds reserved per move for UCI/GUI overhead. Subtracted from
  # the computed deadline before search starts; keeps us off the clock
  # in fast games.
    move_overhead_ms: int = 30
  # Batch-time-variance clock margin, in std-devs of measured chunk wall-time.
  # A chunk (GPU batch) can't be stopped mid-flight, so a slow one can overrun a
  # tight deadline; the search reserves mean + this*std of chunk time before the
  # hard deadline on clock games. 0 disables (restores the pre-margin behavior).
  # See chess_anti_engine/uci/search.py:_BATCH_MARGIN_SIGMAS.
    clock_batch_margin_sigmas: float = 2.0
  # Syzygy semantics: when true, cursed-win/blessed-loss count as draws
  # (matches 50-move-rule play). When false, treat them as decisive
  # (theoretical result — useful for correspondence / analysis).
    syzygy_50_move_rule: bool = True
  # Optional log-file path. When set, stderr logs are mirrored to this
  # file so Windows GUIs (which swallow stderr) can surface diagnostics.
    log_file: str = ""
  # Syzygy tablebase directory path(s). Multiple paths separated by
  # OS-conventional separators (';' on Windows, ':' elsewhere) per the
  # de-facto UCI convention. Empty means disabled.
    syzygy_path: str = ""
  # Ponder gates whether we compute/emit a `ponder` suffix in `bestmove`.
  # If a GUI sends `go ponder` anyway, the engine still honors the command;
  # this option controls advertised ponder metadata, not command parsing.
  # Default off — safer for fixed-time match play.
    ponder: bool = False
  # Visit-margin early-abort aggressiveness (Lc0 smart-pruning factor). 1.0 is
  # the ~provable bound (stop only when the runner-up cannot catch up given
  # every remaining sim); < 1.0 stops earlier on the bet that it won't, banking
  # time. The clock backstop (deadline) is unaffected. Tuning knob — sweep it.
    abort_factor: float = 1.0
  # Per-move time-budget multiplier. > 1.0 schedules more time per move, relying
  # on the abort to bank it back on easy moves; the 50%-of-remaining ceiling
  # still caps each move. 1.0 keeps the conservative allocation. Off by default
  # until validated by a real time-control gauntlet.
    time_budget_scale: float = 1.0
  # Soft-target fraction of the hard clock budget at which a settled move banks
  # the rest (Lc0 optimum-time). 0.7 is the validated default; 0 turns the
  # visit-margin abort fully off (spend the whole deadline — the pre-time-mgmt
  # baseline). Clamped to (0, 1]. Tuning knob — sweep alongside abort_factor.
    optimum_fraction: float = _OPTIMUM_FRACTION
  # Upper cap on the pieces-driven moves estimate (conservatism guard; the
  # estimate tops out near 30, so this normally does not bind). Ignored when the
  # GUI sends movestogo. The allocation itself is driven by material, not this.
    moves_horizon: int = _DEFAULT_MOVES_REMAINING

  # ── search-parameter surface (mcts/search_options.py) ────────────────────
  # One attribute per registry entry, named after `SearchOption.field`, so the
  # handshake / handler / readback all address the same store. Defaults are
  # taken from the registry rather than retyped: a retyped default that drifts
  # from the code default is a handshake that lies about what the engine runs,
  # and `test_advertised_defaults_match_the_live_worker` fails on it.
  # `c_puct` maps to the pre-existing `cpuct` attribute (see _SEARCH_ATTR).
    policy_temp: float = _search_default("policy_temp")
    c_scale: float = _search_default("c_scale")
    c_visit: float = _search_default("c_visit")
    c_scale_root: float = _search_default("c_scale_root")
    c_visit_root: float = _search_default("c_visit_root")
    q_visit_exp_root: float = _search_default("q_visit_exp_root")
    halving_div: int = int(_search_default("halving_div"))
    topk: int = int(_search_default("topk"))
    root_noise_scale: float = _search_default("root_noise_scale")
    chunk_sims: int = int(_search_default("chunk_sims"))
    vloss_weight: int = int(_search_default("vloss_weight"))
    fpu_reduction: float = _search_default("fpu_reduction")

    def search_value(self, field: str) -> float | int | bool:
        """The configured value of one registry field."""
        return getattr(self, _SEARCH_ATTR.get(field, field))

    def set_search_value(self, field: str, value: float | int | bool) -> None:
        setattr(self, _SEARCH_ATTR.get(field, field), value)


class RebuildEvaluator(Protocol):
    """Evaluator factory: ``n_walkers`` (when given) re-tiers the dispatcher
    stack (CUDAOwner at 1 walker vs BatchCoalescing at >1) and is remembered
    by the factory for later rebuilds; ``None`` keeps the current count."""

    def __call__(
        self, max_batch: int, eval_cache_entries: int = 0, /,
        *, n_walkers: int | None = None,
    ) -> BatchEvaluator: ...


class Engine:
    def __init__(
        self,
        worker: SearchWorker,
        *,
        rebuild_evaluator: RebuildEvaluator | None = None,
        rebuild_multi_gpu_pucv_factories: (
            Callable[[int, int], list[Callable[[], BatchEvaluator]]] | None
        ) = None,
        search_devices: tuple[str, ...] | None = None,
        options: EngineOptions | None = None,
        gil_profile: bool = False,
    ) -> None:
        self._worker = worker
  # Factory that takes a max_batch and returns a warmed-up evaluator
  # (model + devices + coalesce flag are captured at construction).
  # When None, the MaxBatch setoption silently no-ops.
        self._rebuild_evaluator = rebuild_evaluator
        self._rebuild_multi_gpu_pucv_factories = rebuild_multi_gpu_pucv_factories
  # Device list backing the per-device factories, threaded through to the
  # root-parallel Gumbel pool so each group thread can set_device before
  # its factory compiles (cudagraph TLS). None when unknown (single device).
        self._search_devices = search_devices
        self._options = options if options is not None else EngineOptions()
        self._gil_probe = GilContentionProbe(interval_s=0.010) if gil_profile else None
        self._worker.set_max_tree_mb(self._options.hash_mb)
        self._leaf_tb_noted: str | None = None
  # Set by setoption handlers that reshape the search path / evaluator
  # (SearchWalkers/UseVL/UseMultiGpuPUCV/LeafGather/VLGather/MaxBatch/
  # EvalCacheEntries). The build-time ``warmup_search`` runs BEFORE any
  # setoption is applied, so such a change leaves its cudagraph stale;
  # ``_handle_isready`` re-warms the configured path before ``readyok``
  # so the first real ``go`` never pays cold capture on the clock.
        self._warmup_dirty = False
  # Search-phase exceptions that selected a bestmove fallback. A ponder
  # fallback may be replaced after ponderhit; its fault still counts. Every
  # increment emits its phase; see the `bestmove_fallback_used` property.
        self._bestmove_fallback_used = 0
  # Times the C search DECLINED the root and the move came from the raw prior
  # with no simulation. Deliberately separate from the fault counter above —
  # see `_report_declined_root`.
        self._prior_only_roots = 0
  # Search phases in a row that published a fallback move (reset by any phase
  # that returns a searched result). See `_note_fallback_phase`.
        self._consecutive_search_failures = 0
  # Solved roots whose proof named no move (`SolvedRootWitnessUnavailable`):
  # counted here, NOT in the streak above -- a property of the position.
        self._witness_unavailable_roots = 0
  # A ponder phase that fell back has not published anything yet: ponderhit
  # hands the move to a real phase, which then decides the streak. Only a
  # ponder reply that IS published (stop) counts. See `_run_search`.
        self._ponder_fallback_pending = False
  # Why the engine is unhealthy (None = healthy). Set after
  # `_UNHEALTHY_FALLBACK_STREAK` fallbacks in a row; cleared only by passing the
  # strict warmup at the next game boundary, after a synchronous evaluator
  # rebuild there. A failed re-check sets ``_exit_code`` and requests quit;
  # `__main__` exits with that code.
        self._unhealthy: str | None = None
        self._exit_code = 0
  # True from the first `go` until the next game boundary: `ucinewgame`, or a
  # `position` that starts a game (see `_handle_position`). The health re-check
  # and the exit it can trigger run only while this is False; the plain
  # re-warm after an option rebuild runs on ANY isready while the worker is free.
        self._in_game = False
        self._board = chess.Board()
  # Why the last `position` was rejected, or None. While set, every `go` answers
  # `bestmove 0000` (see `_handle_position`).
        self._position_error: str | None = None
        self._search_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cancel_event = threading.Event()
        self._ponderhit_at: float | None = None
        self._ponderhit_event = threading.Event()
  # Set only by a GUI command that ends ponder (stop, ponderhit, quit,
  # or a command that waits out the search). A pool fault sets
  # ``_stop_event`` and then raises; that is not one of those commands.
        self._ponder_release = threading.Event()
  # While the current search is a `go ponder`, hold the "real" limits
  # here so ponderhit can swap them in.
        self._pending_real_limits: SearchLimits | None = None
        self._quit_requested = False
        self._closed = False
  # Monotonic counter bumped on every new `go`. Stored on the search
  # thread's frame to guard result publication. Worker writes are protected
  # by quiescence before any command changes the worker or starts another go.
        self._search_gen = 0
        self._state_lock = threading.Lock()
        self._warmup_done = threading.Event()
        self._warmup_done.set()
  # `position` commands go here; the tree isn't advanced until `go` so
  # ponder mode can choose to back off one ply (see _handle_go).
        self._pending_fen: str | None = None
        self._pending_moves: list[chess.Move] = []
  # What we actually descended the tree through. On the next `position`+
  # `go` pair, if pending-moves extend applied-moves we descend the
  # extras instead of rebuilding. On ponder, applied-moves is one ply
  # short of pending-moves.
        self._applied_fen: str | None = None
        self._applied_moves: tuple[chess.Move, ...] = ()
  # Set on `go ponder`: the last move of the position command, which
  # the opponent will play if we predicted correctly. On ponderhit, the
  # search thread advances the tree root by this move before switching
  # to time-bounded search.
        self._popped_ponder_move: chess.Move | None = None
  # The reply-only thread `_answer_failed_go` starts, while it is the current
  # search thread; ponderhit releases it like stop.
        self._fallback_reply_thread: threading.Thread | None = None
  # Search threads that outlived their join timeout and were superseded by a
  # fallback reply. They may still own the worker, so the quiescence barrier
  # waits for them too.
        self._orphan_threads: list[threading.Thread] = []
  # (board, searchmoves) of the `go` the current search thread answers, so a
  # search parked as an orphan by `_wait_for_search` still gets its one bestmove.
        self._live_go: tuple[chess.Board, tuple[str, ...]] | None = None
        self._go_received_at: float | None = None
  # Who answers a go: the generation whose bestmove has been printed. Every
  # bestmove writer (the search's emit, the reply-only thread, the parked-search
  # answer at `stop`, the deadline watchdog) claims it under `_state_lock`
  # first (`_claim_answer`), so a go is answered exactly once under any
  # interleaving.
        self._claimed_gen = 0
  # The search's latest best move, published by the search thread as ONE
  # immutable ``(fen, uci)`` tuple swapped atomically; the watchdog reads it
  # from its own thread and uses it only for the position it is keyed to.
        self._pv_snapshot: tuple[str, str] | None = None
        self._watchdog_fires = 0
        self._watchdog = _DeadlineWatchdog(self._watchdog_poll)
  # Search heartbeat: ``(gen, monotonic time of the last completed chunk,
  # slowest inter-chunk time in ms or None)``, one tuple swapped by a single
  # assignment from the search thread's per-chunk callback; read lock-free by
  # the watchdog, which uses it only when ``gen`` is the armed go's.
        self._heartbeat: tuple[int, float, float | None] | None = None
  # The slowest chunk of the last search that completed normally: the stall
  # scale before the current search has timed a chunk of its own.
        self._prev_chunk_ms: float | None = None
  # The same for the stretch from phase start to the first chunk, which also
  # holds the root eval, tree setup and any compaction: the stall scale while
  # the current search has not completed a chunk at all.
        self._prev_first_chunk_ms: float | None = None
  # The real budget of the pending ponder go (movetime or remaining clock).
        self._pending_real_budget_ms: int | None = None
  # Set by `position` / `ucinewgame` when the tree must not be reused; consumed by
  # `_sync_tree_root` at the next `go`, when the worker is free. `position` never
  # touches the worker itself.
        self._force_tree_reset = False

  # -- main-thread command dispatch -----------------------------------------

    def dispatch(self, cmd: Command) -> None:
        if self._quit_requested or self._closed:
            _println("info string command rejected: engine is shutting down")
            return
        # A setoption is a transaction over BOTH the options and the worker-side
        # search state (`set_gumbel_field` writes `_cfg` before a helper rebuild
        # that can fail). On a raise both are restored and the matching path
        # reinstalled; on success OR rollback the realized path must be the one
        # the options name. See `_restore_after_failed_setoption`.
        snapshot: _SetoptionSnapshot | None = None
        if isinstance(cmd, CmdSetOption):
            snapshot = (replace(self._options), self._worker_search_state())
        try:
            self._dispatch(cmd)
            if snapshot is not None:
                self._enforce_path_invariant(after="setoption")
        except SearchNotQuiescent as exc:
            _println(f"info string command rejected ({type(cmd).__name__}): {exc}")
            if isinstance(cmd, CmdGo):
  # The previous search did not stop within the wait budget and still owns the
  # worker. This `go` must still be answered exactly once, from a fallback that
  # does not touch the busy worker's evaluator.
                self._answer_failed_go(cmd, worker_busy=True)
        except Exception as exc:
  # A handler bug or a resource failure must never end the process: a dead
  # engine forfeits every remaining game, a lost option costs nothing. One
  # line on stdout for the GUI, the traceback to stderr / LogFile for the
  # operator. KeyboardInterrupt / SystemExit are not Exceptions and still exit.
            _log.error("UCI command %r failed", cmd, exc_info=True)
            _println(
                f"info string error handling {type(cmd).__name__}: "
                f"{_one_line(exc)}; engine continues"
            )
            if isinstance(cmd, CmdGo):
                self._answer_failed_go(cmd)
            elif isinstance(cmd, CmdIsReady):
  # `readyok` is the handler's last statement, so it was not printed. A GUI
  # blocks on it; the engine is alive and answers `go`, so say ready.
                _println(format_readyok())
            elif snapshot is not None:
                self._restore_after_failed_setoption(snapshot)

    def _worker_search_state(self) -> object:
        snap = getattr(self._worker, "search_state", None)
        return snap() if callable(snap) else None

    def _restore_after_failed_setoption(
        self, snapshot: _SetoptionSnapshot, *, what: str = "setoption",
    ) -> None:
        """Put the engine back on a working configuration that matches its options.

        Preferred: the configuration before the failed setoption -- options AND
        worker-side search state from ``snapshot``, with the matching path
        reinstalled if the failure tore it down or the restored state no longer
        matches the installed helpers. If that cannot be done, or the realized
        path still disagrees with the options, fall back to the last-resort
        single-evaluator classic Gumbel path and make the options SAY so.
        One info string names the outcome and the realized path.
        """
        options, worker_state = snapshot
        for f in fields(EngineOptions):
            setattr(self._options, f.name, getattr(options, f.name))
        try:
            restore = getattr(self._worker, "restore_search_state", None)
            changed = bool(restore(worker_state)) if callable(restore) else False
            self._ensure_configured_path_installed(force=changed)
            mismatch = self._path_mismatch()
            if mismatch is not None:
                raise RuntimeError(mismatch)
            outcome = "previous configuration kept"
        except Exception as exc:
            _log.error("restoring the previous configuration failed", exc_info=True)
            outcome = (
                f"previous configuration could not be restored ({_one_line(exc)}); "
                + self._fall_back_to_single_evaluator_gumbel()
            )
        self._warmup_dirty = True
        _println(
            f"info string {what} failed; {outcome}: "
            f"path={self._worker.realized_search_path()}"
        )

    def _set_configured_walkers(self, n: int) -> None:
        self._options.search_walkers = n

    def _path_mismatch(self) -> str | None:
        """Why the realized search path is not the one the options name, or None.

        ``UseVL`` may legitimately realize classic Gumbel (an evaluator without
        the slot API); every other combination names exactly one path.
        """
        realized = getattr(self._worker, "realized_search_path")()
        if not isinstance(realized, str):  # a test double without a real worker
            return None
        allowed = self._paths_named_by(self._options)
        if realized in allowed:
            return None
        return f"realized path={realized} but the options name {'/'.join(allowed)}"

    def _paths_named_by(self, o: EngineOptions) -> tuple[str, ...]:
        """The realized search path(s) a set of options names."""
        if o.search_parallel == "gumbel":
            allowed: tuple[str, ...] = ("rpg",)
        elif o.use_multi_gpu_pucv:
            allowed = ("pucv_pool",)
        elif int(o.search_walkers) > 1:
            allowed = ("walker",)
        elif o.use_vl:
            allowed = ("pucv", "gumbel")
        else:
            allowed = ("gumbel",)
        return allowed

    def _enforce_path_invariant(self, *, after: str) -> None:
        mismatch = self._path_mismatch()
        if mismatch is None:
            return
        _log.error("%s left an inconsistent search path: %s", after, mismatch)
        outcome = self._fall_back_to_single_evaluator_gumbel()
        self._warmup_dirty = True
        _println(
            f"info string {after}: {mismatch}; {outcome}: "
            f"path={self._worker.realized_search_path()}"
        )

    def _fall_back_to_single_evaluator_gumbel(self) -> str:
        """Last resort: classic Gumbel on the current primary, options to match.

        Builds nothing: it only tears pools and helpers DOWN, so it cannot fail
        for lack of memory.
        """
        # Each option is written right AFTER the teardown it describes, so at
        # every point -- including after a raise part-way -- the options name
        # what is actually installed.
        o = self._options
        try:
            worker = self._worker
            worker.clear_root_parallel_gumbel()
            o.search_parallel = "pucv"
            worker.clear_multi_gpu_pucv(restore_fallback=False)
            o.use_multi_gpu_pucv = False
            worker.set_use_pucv(False)
            o.use_vl = False
            worker.set_num_threads(1)
            self._set_configured_walkers(1)
        except Exception as exc:
            _log.error("single-evaluator fallback failed", exc_info=True)
            return (
                f"no working search path could be installed ({_one_line(exc)}); "
                "moves will be unsearched fallbacks"
            )
        return ("fell back to the single-evaluator classic Gumbel path "
                "(multi-GPU pools OFF, UseVL off, 1 walker)")

    def _ensure_configured_path_installed(self, *, force: bool = False) -> None:
        """Install the path the options name if it is missing (or ``force``).

        ``force``: the worker-side search state was rolled back, so helpers
        built from the failed state must be rebuilt from the restored one.
        Raises if impossible.
        """
        worker = self._worker
        rpg = getattr(worker, "_rpg_pool", None)
        pucv_pool = getattr(worker, "_pucv_pool", None)
        if self._options.search_parallel == "gumbel":
            if (rpg is None or force) and not self._try_install_root_parallel_gumbel():
                raise RuntimeError("root-parallel Gumbel pool could not be reinstalled")
        elif self._options.use_multi_gpu_pucv:
            if (pucv_pool is None or force) and self._install_multi_gpu_pucv_pool() is None:
                raise RuntimeError("multi-GPU PUCV pool could not be reinstalled")
        else:
            # A pool install tears the walker pool / PUCV chunker down before it
            # builds; if it then failed, the classic helpers are missing -- and a
            # CLOSED pool is missing too (see `classic_helpers_missing`).
            missing = getattr(worker, "classic_helpers_missing", None)
            helpers_missing = bool(missing()) if callable(missing) else False
            if rpg is not None or pucv_pool is not None or helpers_missing or force:
                worker.clear_root_parallel_gumbel()
                worker.clear_multi_gpu_pucv(restore_fallback=False)
                worker.restore_classic_search_path()

    def _answer_failed_go(self, cmd: CmdGo, *, worker_busy: bool = False) -> None:
        """Guarantee exactly one bestmove for a ``go`` whose handler raised.

        If the handler got as far as starting the search thread, that thread owns
        the reply (and the generation gate already keeps it to one). Otherwise
        nothing would ever answer and the GUI would flag us on time, so start a
        reply-only thread under a NEW generation: it obeys the same UCI rules as a
        real search -- a ponder reply waits for ``ponderhit``/``stop``, an
        open-ended one for ``stop`` -- and the same gate drops it if a newer
        ``go`` or ``quit`` supersedes it.

        ``worker_busy``: the go was rejected because the PREVIOUS search did not
        stop. That thread is parked as an orphan (every later command still
        waits for it -- see `_wait_for_search`), the new generation silences its
        stale result, and the fallback reads only the cached prior, never the
        evaluator the orphan is still using.
        """
        thread = self._search_thread
        if worker_busy:
            if thread is not None and thread.is_alive():
                self._park_search_thread(thread, answer=False)
            self._search_thread = None
        elif thread is not None and thread.ident is not None:
            return
        if self._position_error is not None:
            # No fallback move either: it would be for a board the GUI did not send.
            self._refuse_go_for_rejected_position(cmd.args)
            return
        args = cmd.args
        try:
            open_ended = limits_from_go(
                args, side_to_move_is_white=self._board.turn == chess.WHITE,
            ).is_open_ended()
        except Exception:
            open_ended = args.infinite or args.ponder
        board = self._board.copy()
        self._stop_event = threading.Event()
        self._cancel_event = threading.Event()
        self._ponderhit_at = None
        self._ponderhit_event = threading.Event()
        self._ponder_release = threading.Event()
        self._pending_real_limits = None
        self._popped_ponder_move = None
        with self._state_lock:
            self._search_gen += 1
            gen = self._search_gen
        # Recorded like a search's, so a `stop` that has to park this thread
        # (stalled before it could answer) answers its go from the main thread.
        self._live_go = (board.copy(), tuple(args.searchmoves))
        self._bestmove_fallback_used += 1
        self._note_unsearched_reply()
        fresh_budget_s = self._failed_go_fresh_eval_budget_s(cmd)
        reply = threading.Thread(
            target=self._run_failed_go_reply,
            args=(gen, board, args.searchmoves, args.ponder, open_ended, not worker_busy,
                  fresh_budget_s),
            kwargs={"cancel_event": self._cancel_event, "ponder_release": self._ponder_release},
            daemon=True,
        )
        self._search_thread = reply
        self._fallback_reply_thread = reply
        try:
            reply.start()
        except Exception:
  # No thread to wait for UCI's permission: answer now rather than never.
            self._search_thread = None
            self._fallback_reply_thread = None
            self._run_failed_go_reply(
                gen, board, args.searchmoves, False, False, not worker_busy, fresh_budget_s,
            )

    def _run_failed_go_reply(
        self, gen: int, board: chess.Board, searchmoves: tuple[str, ...],
        ponder: bool, open_ended: bool, allow_fresh_eval: bool = True,
        fresh_budget_s: float | None = None,
        *, cancel_event: threading.Event | None = None,
        ponder_release: threading.Event | None = None,
    ) -> None:
        # Wait on THIS go's events, handed over at thread creation: `self._...`
        # is replaced by every later go, and a superseded thread waiting on the
        # new, never-set event would hold the worker forever.
        if ponder:
            (ponder_release or self._ponder_release).wait()
        elif open_ended:
            (cancel_event or self._cancel_event).wait()
        with self._state_lock:
            if gen != self._search_gen or self._quit_requested:
                return
        move, source = self._bestmove_fallback(
            board, searchmoves, allow_fresh_eval=allow_fresh_eval, budget_s=fresh_budget_s,
        )
        # Claimed only once the move is in hand: if this thread stalls inside
        # the fallback and `stop` parks it, park answers the go instead.
        if not self._claim_answer(gen):
            return
        with _answer_owner():
            _println(
                f"info string search not started; bestmove_fallback_used="
                f"{self._bestmove_fallback_used} source={source} move={move}"
            )
            _println(format_bestmove(move, ponder=None))

    def _dispatch(self, cmd: Command) -> None:
        if isinstance(cmd, CmdUci):
            self._handle_uci()
        elif isinstance(cmd, CmdIsReady):
            self._handle_isready()
        elif isinstance(cmd, CmdUciNewGame):
            self._handle_newgame()
        elif isinstance(cmd, CmdPosition):
            self._handle_position(cmd)
        elif isinstance(cmd, CmdGo):
            self._handle_go(cmd)
        elif isinstance(cmd, CmdStop):
            self._handle_stop()
        elif isinstance(cmd, CmdPonderHit):
            self._handle_ponderhit()
        elif isinstance(cmd, CmdSetOption):
            self._handle_setoption(cmd)
        elif isinstance(cmd, CmdSearchConfig):
            self._handle_searchconfig()
        elif isinstance(cmd, CmdQuit):
            self._quit_requested = True
            self._handle_stop()
        else:  # CmdUnknown
            head = cmd.raw.split()[0] if cmd.raw.split() else ""
            if head in ("debug", "register"):
  # UCI spec commands this engine does not implement. Unknown commands stay
  # silent (the spec says to ignore them, and a blank line is one).
                _println(f"info string {head}: not implemented by this engine; ignored")

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

  # -- handlers -------------------------------------------------------------

    def _handle_uci(self) -> None:
        emit_handshake(self._options)

    def _handle_isready(self) -> None:
  # A setoption that reshaped the search path / evaluator since the last warmup
  # left its cudagraph stale. Re-warm the CONFIGURED path here, BEFORE readyok, so
  # the first real ``go`` (the clock starts after readyok in the standard handshake)
  # never pays cold capture. ONLY when idle: warmup_search drives self._worker.run,
  # so re-warming while a ponder search is live on the same worker would race the
  # shared tree / eval caches. On an isready mid-ponder, leave _warmup_dirty set and
  # re-warm on the next idle isready rather than corrupting the live search.
  # Idempotent: the flag is cleared so we don't warm again until the next reshape.
        # Not free (a search or an orphan still inside the worker): answer at once
        # and touch nothing. Free: re-warm a dirty path on ANY isready, mid-game
        # too (no clock runs between `bestmove` and the next `go`). The health
        # re-check and the exit it can trigger run only at a game boundary.
        if self.worker_is_free():
            if not self._in_game and not self._game_boundary_checks():
                return  # exiting non-zero; no readyok over a broken search
            if self._warmup_dirty:
                try:
                    self.warmup_search()
                except Exception as exc:
                    # The path stays dirty (the next free isready retries) and
                    # the next game boundary rebuilds and strictly re-checks.
                    _log.error("search-path re-warm failed", exc_info=True)
                    _println(
                        f"info string search-path re-warm failed ({_one_line(exc)}); "
                        "the engine still answers every go"
                    )
                    self._mark_unhealthy("the search-path re-warm at isready failed")
                else:
                    self._warmup_dirty = False
        _println(format_readyok())

    def _handle_newgame(self) -> None:
        self._in_game = False
        try:
            self._wait_for_search()
        except SearchNotQuiescent as exc:
            # The Python-side game state still resets; the tree resets at the
            # first `go` that gets the worker.
            _println(f"info string ucinewgame: {exc}; tree reset deferred")
            self._force_tree_reset = True
        else:
            # Only opens the boundary: the health rebuild / re-check runs at the
            # next free isready, never here -- a GUI may send `position` + `go`
            # right after `ucinewgame` and would not see an exit until it wrote.
            self._worker.reset_tree()
        self._board = chess.Board()
        self._position_error = None
        self._pending_fen = None
        self._pending_moves = []
        self._applied_fen = None
        self._applied_moves = ()
        self._popped_ponder_move = None

    def _handle_position(self, cmd: CmdPosition) -> None:
  # Records Python-side board state only: never waits on, rejects for, or touches
  # the worker. The board every later reply is for is therefore always the latest
  # one the GUI sent, even while an orphaned search is still inside the worker.
  #
  # A `position` the engine cannot take EXACTLY as sent is rejected as a whole
  # (see `_validated_position`). The previous position is KEPT -- no state
  # changes -- and every `go` until the next valid `position` / `ucinewgame`
  # answers `bestmove 0000` with an `info string` naming the rejection
  # (`_refuse_go_for_rejected_position`). Never a move for a board the GUI did
  # not send: the old behaviour (the start position for a bad FEN, the legal
  # prefix for a bad move list) answered confidently for a different board.
        validated = self._validated_position(cmd)
        if isinstance(validated, str):
            self._position_error = validated
            _println(
                f"info string position: rejected ({validated}); keeping the previous "
                "position; go answers bestmove 0000 until a valid position arrives"
            )
            return
        new_board, parsed, root_fen, notes = validated
        for note in notes:
            _println(f"info string position: normalised {note}")
        self._position_error = None
        self._board = new_board
  # Game boundary for a GUI that never sends `ucinewgame`: the standard start
  # position with NO moves. Inside a game every `position` carries the moves
  # played so far, so this is never sent mid-game -- not after a ponder miss
  # (same move count), a takeback or an analysis jump (each still carries its
  # moves), nor a per-move `position fen <current>` (not the start position);
  # the one exception, a takeback all the way to the initial position, IS a
  # game start. It only opens the boundary: the checks run at the next free
  # isready, never here, because a `go` (and our clock) may follow at once.
        if not cmd.moves and new_board.board_fen() == chess.STARTING_BOARD_FEN \
                and new_board.fen() == chess.STARTING_FEN:
            self._in_game = False
  # Record intent; the tree is advanced in _handle_go so ponder mode
  # can choose to stop one ply short.
        self._pending_fen = root_fen
        self._pending_moves = parsed
  # `_popped_ponder_move` is left alone: a live ponder search reads it at
  # ponderhit, and every `go` sets it afresh.

    @staticmethod
    def _validated_position(
        cmd: CmdPosition,
    ) -> tuple[chess.Board, list[chess.Move], str | None, tuple[str, ...]] | str:
        """``(board, moves, root_fen, notes)`` for a `position` the engine can
        answer, else the reason it cannot.

        ``root_fen`` is the FEN every later rebuild of the root uses (tree sync,
        the ponder board, and through them the C board and the net's planes):
        ``cmd.fen`` unchanged for a ``Board.status()``-VALID root, the NORMALISED
        FEN otherwise, so python-chess, the C board and the planes all see the
        same castling rights and en-passant square. ``notes`` names each
        normalisation, for one `info string` apiece.

        Normalised (a legal answer exists; python-chess's own move generation
        already ignores these, the C board did not): castling rights without the
        king and rook on their squares -> ``clean_castling_rights()``; an
        en-passant square no double push could have made -> none.

        Rejected (no answer exists, or the C board cannot search it): a
        malformed command (``cmd.error``); an unparseable FEN; missing / extra
        kings, an empty board, pawns on the back rank, the side NOT to move in
        check (``unsearchable_king_reason`` too); any status flag this method
        does not know; and any move in ``moves`` that is not legal where it is
        played.

        Tolerated only when the C board agrees (too many pieces / pawns, an
        impossible check, too many checkers -- legal moves exist, but the
        position is unreachable, so the C move generator's assumptions are not
        guaranteed): accepted iff ``CBoard`` and python-chess produce the same
        legal-move set on the root and on every board the move list reaches,
        else rejected with that reason.
        """
        if cmd.error is not None:
            return cmd.error
        notes: list[str] = []
        root_fen: str | None = cmd.fen
        tolerated = False
        if cmd.fen is None:
            start = chess.Board()
        else:
            try:
                start = chess.Board(cmd.fen)
            except ValueError as exc:
                return f"FEN {cmd.fen!r}: {exc}"
            status = start.status()
            if status != chess.STATUS_VALID:
                if status & chess.STATUS_BAD_CASTLING_RIGHTS:
                    # The FEN's own field: `castling_xfen()` already prints
                    # the cleaned rights.
                    fields = cmd.fen.split()
                    before = fields[2] if len(fields) > 2 else "?"
                    start.castling_rights = start.clean_castling_rights()
                    notes.append(
                        f"castling rights {before} -> {start.castling_xfen()} "
                        "(a right without its king/rook on the home square)"
                    )
                if status & chess.STATUS_INVALID_EP_SQUARE and start.ep_square is not None:
                    notes.append(
                        f"en-passant square {chess.square_name(start.ep_square)} "
                        "is impossible -> none"
                    )
                    start.ep_square = None
                status = start.status()
                hard = status & ~_TOLERATED_STATUS
                if hard:
                    return f"FEN {cmd.fen!r} is not a legal position: {_status_names(hard)}"
                tolerated = bool(status)
                root_fen = start.fen(en_passant="fen")
            king = unsearchable_king_reason(start)
            if king is not None:
                return f"FEN {cmd.fen!r}: {king}"
        board = start.copy(stack=False)
        if tolerated:
            disagreement = _movegen_disagreement(board)
            if disagreement is not None:
                return (
                    f"FEN {cmd.fen!r} is unreachable ({_status_names(board.status())}) "
                    f"and {disagreement}"
                )
        parsed: list[chess.Move] = []
        for ply, uci in enumerate(cmd.moves, start=1):
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                return f"move {ply} {uci!r} is not valid UCI"
            if mv not in board.legal_moves:
                return f"move {ply} {uci!r} is illegal in {board.fen()}"
            board.push(mv)
  # Defence in depth. python-chess's legal_moves includes CAPTURING THE ENEMY
  # KING when the side not to move is already in check, which walked a legal-
  # looking FEN + move list into a kingless board (measured:
  # 4k2r/8/8/8/3p4/8/2P5/4R1K1 w - - 0 1 moves e1e8 h8h7 c2c4 gave a board the
  # Python and C halves disagree about -- the key-MERGING divergence
  # _cboard_impl.h's precondition (i) is about). OPPOSITE_CHECK roots are
  # rejected above, and legal play from a VALID position cannot reach one; this
  # stays because that argument is a property of the current rules, and an
  # unreachable (tolerated) root weakens it.
            post_reason = unsearchable_king_reason(board)
            if post_reason is not None:
                return f"move {ply} {uci!r} leaves an unsearchable position ({post_reason})"
            if tolerated:
                disagreement = _movegen_disagreement(board)
                if disagreement is not None:
                    return f"after move {ply} {uci!r}: {disagreement}"
            parsed.append(mv)
        return board, parsed, root_fen, tuple(notes)

    def _sync_tree_root(self, target_moves: list[chess.Move]) -> None:
        """Descend (or reset) the worker's tree so its root matches
        ``self._pending_fen`` plus ``target_moves``. Updates ``_applied_*`` to
        reflect the post-sync state."""
        reused = (
            not self._force_tree_reset
            and self._applied_fen == self._pending_fen
            and len(target_moves) >= len(self._applied_moves)
            and tuple(target_moves[:len(self._applied_moves)]) == self._applied_moves
        )
        if reused:
            if self._pending_fen is None:
                from_board = chess.Board()
            else:
                from_board = chess.Board(self._pending_fen)
            for mv in self._applied_moves:
                from_board.push(mv)
            extras = target_moves[len(self._applied_moves):]
            reused = self._worker.advance_root(from_board, extras)
        if not reused:
            self._worker.reset_tree()
        self._force_tree_reset = False
        self._applied_fen = self._pending_fen
        self._applied_moves = tuple(target_moves)

    def _handle_go(self, cmd: CmdGo) -> None:
        # First, before anything that can raise: every route that answers a
        # `go` -- a search, the reply-only answer for a handler fault, the
        # busy-worker answer when the wait below raises -- is a move of a game,
        # so a later isready must not run the boundary checks.
        self._in_game = True
        # The move clock starts at `go` receipt: the bounded wait for a live
        # search, tree sync (including any reuse work) and thread start-up
        # below are all charged to this move's deadline.
        go_received_at = time.monotonic()
        self._go_received_at = go_received_at
        # A timeout cannot transfer ownership of the still-live worker; it is
        # bounded by this move's own clock, then `dispatch` answers from the
        # busy-worker fallback for the latest board.
        self._wait_for_search(timeout_s=self._go_wait_budget_s(cmd))
        if self._position_error is not None:
            self._refuse_go_for_rejected_position(cmd.args)
            return
        self._report_go_args(cmd)
  # Ponder mode: search at the position BEFORE opponent's predicted
  # reply. That node's root expansion creates a child for every legal
  # opponent move, so whichever one they play — predicted or not — we
  # already have sims proportional to the Gumbel prior. Ponder-hit
  # advances the root by one ply; ponder-miss advances by the actually-
  # played move, which is still a root child, so it descends cleanly.
        if cmd.args.ponder and len(self._pending_moves) >= 1:
            target_moves = self._pending_moves[:-1]
            self._popped_ponder_move = self._pending_moves[-1]
            search_board = chess.Board() if self._pending_fen is None else chess.Board(self._pending_fen)
            for mv in target_moves:
                search_board.push(mv)
        else:
            target_moves = self._pending_moves
            self._popped_ponder_move = None
            search_board = self._board
        self._note_leaf_tb_inactive()
        self._sync_tree_root(target_moves)
        overhead = self._options.move_overhead_ms
        limits = limits_from_go(
            cmd.args,
            side_to_move_is_white=(search_board.turn == chess.WHITE),
            move_overhead_ms=overhead,
            time_budget_scale=self._options.time_budget_scale,
            optimum_fraction=self._options.optimum_fraction,
            moves_horizon=self._options.moves_horizon,
            pieces=chess.popcount(search_board.occupied),
        )
        self._stop_event = threading.Event()
        self._cancel_event = threading.Event()
        self._ponderhit_at = None
        self._ponderhit_event = threading.Event()
        self._ponder_release = threading.Event()
  # For `go ponder`, the ponder phase runs open-ended; ponderhit
  # converts to a real-deadline phase using the SAME underlying clock
  # args. We re-derive the "real" limits here by synthesizing a
  # non-ponder copy of the args — but the side-to-move must be
  # derived from the REAL-phase board (after the opponent's predicted
  # move was pushed), not search_board (which is their side). Using
  # search_board.turn here picks the opponent's wtime/btime and
  # causes the real phase to flag or undersearch at asymmetric
  # clocks. (Codex adversarial review.)
        if cmd.args.ponder:
            real_board = search_board.copy(stack=False)
            if self._popped_ponder_move is not None:
                real_board.push(self._popped_ponder_move)
            # Bare FEN/startpos already names the real board; it still needs
            # a clocked handoff on ponderhit without any root advancement.
            self._pending_real_limits = limits_from_go(
                replace(cmd.args, ponder=False),
                side_to_move_is_white=(real_board.turn == chess.WHITE),
                move_overhead_ms=overhead,
                time_budget_scale=self._options.time_budget_scale,
                optimum_fraction=self._options.optimum_fraction,
                moves_horizon=self._options.moves_horizon,
                pieces=chess.popcount(real_board.occupied),
            )
        else:
            self._pending_real_limits = None
        self._pending_real_budget_ms = (
            self._move_budget_ms(replace(cmd.args, ponder=False),
                                 white_to_move=self._board.turn == chess.WHITE)
            if cmd.args.ponder else None
        )
        with self._state_lock:
            self._search_gen += 1
            gen = self._search_gen
        self._pv_snapshot = None  # only this go's search may publish one
        self._live_go = (self._board.copy(), tuple(cmd.args.searchmoves))
        self._search_thread = threading.Thread(
            target=self._run_search,
            args=(limits, gen, search_board, go_received_at),
            kwargs={"cancel_event": self._cancel_event, "ponder_release": self._ponder_release},
            daemon=True,
        )
        if not cmd.args.ponder and not limits.is_open_ended():
            # Armed before the thread starts, so the search's disarm can never
            # precede it. Ponder arms at ponderhit, with the real limits; pure
            # `go nodes` / `go depth` have no deadline and are not armed.
            self._arm_watchdog(
                gen, started_at=go_received_at, deadline_ms=limits.deadline_ms,
                budget_ms=self._move_budget_ms(
                    cmd.args, white_to_move=self._board.turn == chess.WHITE,
                ),
                board=self._board, searchmoves=tuple(cmd.args.searchmoves),
                thread=self._search_thread, stops=(self._stop_event, self._cancel_event),
            )
        self._search_thread.start()

    def _refuse_go_for_rejected_position(self, args: GoArgs) -> None:
        """Answer this `go` with ``bestmove 0000``: the last `position` was
        rejected, and any move would be for a board the GUI did not send.

        Why not answer for the previous valid position instead: that board is
        the one BEFORE the GUI's current one (usually the opponent's turn or an
        earlier move), so its move is illegal in the real game -- or, worse,
        legal and silently wrong. ``0000`` is UCI's null move: the GUI rejects it
        at once and the game log shows the `info string` naming the rejection,
        rather than the engine flagging on time or playing an unrelated move.

        Timing obeys UCI like any search: `go ponder` holds the answer until
        ``ponderhit`` / ``stop``, `go infinite` until ``stop`` (a reply-only
        thread, released by the same events as `_run_failed_go_reply`); every
        other `go` answers at once. One writer, through the usual claim."""
        self._stop_event = threading.Event()
        self._cancel_event = threading.Event()
        self._ponderhit_at = None
        self._ponderhit_event = threading.Event()
        self._ponder_release = threading.Event()
        self._pending_real_limits = None
        self._popped_ponder_move = None
        with self._state_lock:
            self._search_gen += 1
            gen = self._search_gen
        self._live_go = None  # nothing to fall back to: no move is legal to send
        reason = self._position_error
        _println(
            f"info string go refused: the last position command was rejected "
            f"({reason}); send a valid position"
        )
        if not (args.ponder or args.infinite):
            self._print_refusal(gen)
            return
        hold = self._ponder_release if args.ponder else self._cancel_event
        reply = threading.Thread(
            target=self._held_refusal, args=(gen, hold), daemon=True,
        )
        self._search_thread = reply
        self._fallback_reply_thread = reply
        try:
            reply.start()
        except Exception:
            self._search_thread = None
            self._fallback_reply_thread = None
            self._print_refusal(gen)

    def _held_refusal(self, gen: int, hold: threading.Event) -> None:
        hold.wait()
        self._print_refusal(gen)

    def _print_refusal(self, gen: int) -> None:
        with self._state_lock:
            if gen != self._search_gen or self._quit_requested:
                return
        if not self._claim_answer(gen):
            return
        with _answer_owner():
            _println(format_bestmove("0000"))

    def _report_go_args(self, cmd: CmdGo) -> None:
        """Name every part of this `go` that will not be honoured as sent."""
        args = cmd.args
        if args.errors:
            _println(
                "info string go: ignored unusable value(s) "
                + ", ".join(repr(e) for e in args.errors)
                + (" (mate needs N >= 1)" if any(e.startswith("mate ") for e in args.errors) else "")
            )
        if args.truncated:
            _println(
                "info string go: non-integer value(s) rounded toward zero: "
                + ", ".join(args.truncated)
            )
        if args.searchmoves:
            illegal = []
            for uci in args.searchmoves:
                try:
                    ok = chess.Move.from_uci(uci) in self._board.legal_moves
                except ValueError:
                    ok = False
                if not ok:
                    illegal.append(uci)
            if illegal:
                kept = len(args.searchmoves) - len(illegal)
                _println(
                    f"info string go: searchmoves {' '.join(illegal)} not legal here, "
                    f"ignored ({kept} of {len(args.searchmoves)} kept"
                    + ("; nothing left to search, answering bestmove 0000)" if kept == 0 else ")")
                )
        if args.ponder or args.infinite:
            if args.mate is not None and args.infinite:
                _println(
                    f"info string go: mate {args.mate} with infinite: the mate stop is "
                    "not applied; searching until stop"
                )
            elif args.mate is not None:
                _println(
                    f"info string go: mate {args.mate} with ponder: pondering until "
                    f"ponderhit / stop; after ponderhit the search stops once a mate in "
                    f"<= {args.mate} is proven"
                )
            return
        try:
            limits = limits_from_go(
                args, side_to_move_is_white=self._board.turn == chess.WHITE,
                move_overhead_ms=self._options.move_overhead_ms,
            )
        except Exception:
            return
        if limits.implicit_cap:
            what = (
                f"mate {args.mate} with no other bound" if args.mate is not None
                else "every time/node bound was unparseable"
            )
            _println(
                f"info string go: {what} -- searching at most "
                f"{GO_IMPLICIT_CAP_MS} ms"
                + (", stopping as soon as a mate in <= "
                   f"{args.mate} is proven" if limits.mate_moves is not None else "")
            )

    @staticmethod
    def _move_budget_ms(args: GoArgs, *, white_to_move: bool) -> int | None:
        """What the GUI allows this move: ``movetime``, else the side to move's
        remaining clock, else None (no clock: no last-moment threshold)."""
        if args.movetime_ms is not None:
            return int(args.movetime_ms)
        remaining = args.wtime_ms if white_to_move else args.btime_ms
        return None if remaining is None else int(remaining)

    def _go_wait_budget_s(self, cmd: CmdGo) -> float:
        try:
            limits = limits_from_go(
                cmd.args, side_to_move_is_white=self._board.turn == chess.WHITE,
                move_overhead_ms=self._options.move_overhead_ms,
            )
        except Exception:
            return _GO_WAIT_NO_CLOCK_S
        if cmd.args.ponder or limits.deadline_ms is None:
            return _GO_WAIT_NO_CLOCK_S
        deadline_s = max(0.0, limits.deadline_ms / 1000.0)
        floor_s = min(_GO_WAIT_MIN_S, _GO_WAIT_MAX_SHARE * deadline_s)
        return min(_JOIN_TIMEOUT_S, max(floor_s, _GO_WAIT_FRACTION * deadline_s))

    def _failed_go_fresh_eval_budget_s(self, cmd: CmdGo) -> float | None:
        """A tenth of what is left of this `go`'s clock since receipt (None: no clock).

        The reply for a `go` whose handler raised runs after the bounded wait;
        its fresh eval must fit in what remains, not take a fixed 150 ms."""
        received = self._go_received_at
        if received is None or cmd.args.ponder:
            return None
        try:
            limits = limits_from_go(
                cmd.args, side_to_move_is_white=self._board.turn == chess.WHITE,
                move_overhead_ms=self._options.move_overhead_ms,
            )
        except Exception:
            return None
        if limits.deadline_ms is None:
            return None
        remaining_s = limits.deadline_ms / 1000.0 - (time.monotonic() - received)
        return max(0.0, remaining_s) * 0.1

    def _handle_stop(self) -> None:
        try:
            # The GUI is waiting for this bestmove: a search that has not stopped
            # within a second is parked and answered from the fallback.
            self._wait_for_search(
                timeout_s=min(_JOIN_TIMEOUT_S, _GO_WAIT_NO_CLOCK_S), answer_parked=True,
            )
        except SearchNotQuiescent as exc:
            # Stop remains requested. Keep the owner available for a later join.
            _println(f"info string {exc}")

    def _handle_ponderhit(self) -> None:
        """Opponent played our predicted move. Convert open-ended ponder
        search into a time-bounded real search, keeping the MCTS tree.
        """
        if self._search_thread is None or not self._search_thread.is_alive():
            return
        if self._search_thread is self._fallback_reply_thread:
            # A failed `go ponder` is answered by a reply-only thread; ponderhit
            # releases it exactly as `stop` would.
            self._ponder_release.set()
            return
        if self._pending_real_limits is None:
            return
  # Signal the search loop: next iteration, swap ponder-deadline
  # for real-deadline.
        with self._state_lock:
            if self._ponderhit_at is not None:
                return
            ponder_stop = self._stop_event
            self._ponderhit_at = time.monotonic()
            self._ponderhit_event.set()
            gen = self._search_gen
        real = self._pending_real_limits
        if not real.is_open_ended():
            # The real phase's clock starts at ponderhit; its stop event is the
            # go's cancel event (see `_run_search`).
            live = self._live_go
            self._arm_watchdog(
                gen, started_at=self._ponderhit_at, deadline_ms=real.deadline_ms,
                budget_ms=self._pending_real_budget_ms,
                board=self._board, searchmoves=live[1] if live is not None else (),
                thread=self._search_thread, stops=(self._cancel_event,),
            )
        ponder_stop.set()
        self._ponder_release.set()

    def _handle_setoption(self, cmd: CmdSetOption) -> None:
  # Same barrier as _handle_position / _handle_newgame / _handle_isready:
  # options must not mutate while a search thread is still reading them.
  # In particular SyzygyPath swaps the shared tablebase cache, which the
  # search thread probes through on every leaf batch.
        name = cmd.name.lower()
        try:
            self._wait_for_search()
        except SearchNotQuiescent:
  # A stuck search (orphan) still holds the worker. Reporting-only options write
  # one plain field the next info emission reads (no evaluator, tree or pool),
  # and the orphan's own output is silenced, so they apply now. Everything else
  # -- Hash resizes the growth step of the C tree the orphan is still searching
  # -- stays rejected (the previous value stays in force).
            if name not in self._REPORTING_ONLY_OPTIONS:
                raise
            _println(
                f"info string setoption {cmd.name}: applied while a stuck search still "
                "holds the worker (reporting option only)"
            )
  # A building option mid-game is applied here, synchronously, inside the
  # setoption transaction of `dispatch` (rolled back on failure). Nothing is
  # deferred: no background build exists that it could race.
  # SyzygyPath is special: empty value is meaningful (sentinel for unset),
  # all others bail when value is None.
        if name == "syzygypath":
            self._set_syzygy_path(cmd.value)
            return
        handler = self._SETOPTION_HANDLERS.get(name)
        search_opt = OPTIONS_BY_NAME.get(name)
        if handler is None and search_opt is None:
  # UCI says to ignore an unknown option; ignoring it SILENTLY is how a typo in
  # a tournament config (`Syzygy50MoveRul`) goes unnoticed for a whole event.
            _println(f"info string setoption {cmd.name!r}: unknown option, ignored")
            return
        if cmd.value is None:
            _println(f"info string setoption {cmd.name}: no value given, ignored")
            return
        if handler is not None:
            handler(self, cmd.value)
            return
        assert search_opt is not None
        self._set_search_option(search_opt, cmd.value)

  # ─── search-parameter options ─────────────────────────────────────────────

    def _set_search_option(self, opt: SearchOption, raw: str) -> None:
        """Parse, range-check, apply, and report one registry option.

        The reporting is the load-bearing half. UCI gives a GUI no way to see a
        rejected option, so a bad value that is quietly dropped and an inert
        option that is quietly accepted look identical to an operator: the
        engine plays on and the number in their config file never mattered.
        Every branch here says what happened on an `info string`.
        """
        parsed = self._parse_search_value(opt, raw)
        if parsed is None:
            return
        self._options.set_search_value(opt.field, parsed)
        self._apply_search_option(opt, parsed)
        path = self._worker.realized_search_path()
        values = self._worker.realized_search_values()
        why = inert_reason(opt, path, values)
        if why is None:
  # Reaches the search. It may still be pinned inside a branch whose members
  # are all equivalent — say so, but do NOT say "no effect": this setoption may
  # have just CROSSED a branch boundary and changed the played move. Claiming
  # otherwise is the same defect as a silently-ignored knob, sign-flipped.
            note = branch_note(opt, path, values)
            if note is None:
                _println(f"info string {opt.name} set to {parsed}")
            else:
                _println(
                    f"info string {opt.name} set to {parsed} — reaches the live "
                    f"search; {note}. Use `searchconfig` to see every realized "
                    "value."
                )
            return
  # Accepted (a GUI must not see an error for a legal option) but never
  # silent: this is the exact shape of the c_puct-under-Gumbel defect.
        _println(
            f"info string {opt.name} set to {parsed} — NO EFFECT on the live "
            f"search: {why}. Use `searchconfig` to see every realized value."
        )

    def _parse_search_value(
        self, opt: SearchOption, raw: str,
    ) -> float | int | bool | None:
        """Value for ``opt`` from a UCI token, or None (with a reason printed).

        Floats ride `type string` because UCI has no float type — lc0's
        PolicyTemperature convention — so parsing and range-checking are the
        engine's job, and an unchecked one would accept `PolicyTemperature 0`
        and divide the logits by zero.
        """
        text = raw.strip()
        if opt.kind == "check":
            lowered = text.lower()
            if lowered not in ("true", "false"):
                _println(
                    f"info string {opt.name}: expected true/false, got {text!r}; "
                    f"keeping {self._options.search_value(opt.field)}"
                )
                return None
            return lowered == "true"
        try:
            value: float | int = (
                int(text) if opt.kind == "spin" else float(text)
            )
        except ValueError:
            _println(
                f"info string {opt.name}: {text!r} is not a "
                f"{'integer' if opt.kind == 'spin' else 'number'}; "
                f"keeping {self._options.search_value(opt.field)}"
            )
            return None
        lo, hi = opt.lo, opt.hi
      # ⚑ The finiteness clause is load-bearing, not belt-and-braces. Every
      # comparison against `nan` is False, so the range pair alone waved
      # `PolicyTemperature nan` through: it was stored, `searchconfig` reported
      # it LIVE, and the search ran UNTEMPERED because `policy_temp_active(nan)`
      # is False. The startup half of this same guard
      # (`__main__._refuse_out_of_range_startup_value`) writes the range as the
      # chained `lo <= v <= hi`, which refuses nan -- two halves of one guard,
      # sharing bounds and disagreeing on the one value that is silently
      # dropped rather than loudly wrong.
        # Python integers are finite; converting a large spin to float can overflow.
        if (isinstance(value, float) and not math.isfinite(value)) or (
            (lo is not None and value < lo) or (hi is not None and value > hi)
        ):
            _println(
                f"info string {opt.name}: {value} is out of range "
                f"[{lo}, {hi}]; keeping {self._options.search_value(opt.field)}"
            )
            return None
        return value

    def _apply_search_option(
        self, opt: SearchOption, value: float | int | bool,
    ) -> None:
        """Push one option into the object the search actually reads.

        The three worker-level fields do not live on GumbelConfig at all, and
        the PUCT knobs are copied into the walker / PucvChunker / pool configs
        at CONSTRUCTION time — so for those, assigning the config field is
        precisely the "accepted then ignored" failure, and the reinstall is the
        part that makes the option real.
        """
        field = opt.field
        if field == "chunk_sims":
            self._worker.set_chunk_sims(int(value))
            return
        if field == "vloss_weight":
            self._worker.set_vloss_weight(int(value))
            self._reinstall_configured_search_path()
            return
        if field == "minibatch_size":
            self._worker.set_minibatch_size(int(value))
            self._warmup_dirty = True
            return
        if field == "root_noise_scale":
            self._worker.set_root_noise_scale(float(value))
            return
        self._worker.set_gumbel_field(field, value)
        if field in ("c_puct", "cpuct_factor", "cpuct_base", "fpu_reduction"):
  # `_sync_cpuct_to_worker` already owns the rebuild-the-PUCT-helpers dance
  # (live tree scaling + whichever pool cached the value at install).
            self._sync_cpuct_to_worker()

    def _handle_searchconfig(self) -> None:
        """`searchconfig`: dump every realized search parameter, LIVE or INERT.

        Not in the UCI spec — no GUI sends it — which is the point: it is for
        the operator setting up a TCEC/cutechess config, who needs to prove the
        engine used what the config said. Values are read off the live worker,
        so `UseVL=true` on an evaluator that cannot support it reports the path
        that actually ran rather than the one that was asked for.
        """
        self._wait_for_search()
        path = self._worker.realized_search_path()
        values = self._worker.realized_search_values()
        _println(
            f"info string searchconfig path={path} "
            f"({PATH_DESCRIPTIONS.get(path, path)})"
        )
        for name, value, status, why in realized_rows(path, values):
            suffix = f" — {why}" if why else ""
            _println(f"info string searchconfig {name} = {value} [{status}]{suffix}")
        rows = realized_rows(path, values)
        inert = sum(1 for _, _, s, _ in rows if s == "INERT")
  # `BRANCH` is counted apart from `INERT` on purpose: an operator proving
  # their config must not read "this parameter is not reaching the search"
  # (CPuct under Gumbel) off the same number as "you cannot vary it without
  # crossing its branch boundary" (QVisitExpRoot at its shipped -1.0).
        branch = sum(1 for _, _, s, _ in rows if s == "BRANCH")
        _println(
            f"info string searchconfig {len(rows) - inert - branch} live, "
            f"{branch} branch-pinned, {inert} inert on path={path}"
        )

  # ─── setoption handlers ────────────────────────────────────────────────────
  # Each takes the raw value string and is responsible for parsing + clamping.
  # Non-int values keep the prior value, and say so.
    @staticmethod
    def _parse_clamped_int(value: str, name: str) -> int | None:
        """Parse hand-written spin option ``name`` and clamp it to ``_SPIN_BOUNDS``.

        CLAMP, not reject, at both ends: clamping is what this engine always did
        at the low end (``Hash 256`` -> 1024 is pinned by a test), and a GUI that
        sends ``Threads 128`` or ``MaxBatch 50000000`` means "as much as you
        allow", which the advertised maximum answers. Every clamp is announced on
        an ``info string`` -- a silently moved value is this repo's signature
        defect. Registry search options keep their own reject-with-reason rule
        (``_parse_search_value``).
        """
        try:
            requested = int(value.strip())
        except ValueError:
            _println(f"info string {name}: {value.strip()!r} is not an integer; ignored")
            return None
        lo, hi = _SPIN_BOUNDS[name]
        n = min(hi, max(lo, requested))
        if requested < lo:
            _println(
                f"info string {name} {requested} is below the advertised minimum "
                f"{lo}; using {n}"
            )
        elif requested > hi:
            _println(
                f"info string {name} {requested} is above the advertised maximum "
                f"{hi}; using {n}"
            )
        return n

    @staticmethod
    def _parse_check(value: str, name: str, current: bool) -> bool | None:
        """``true`` / ``false`` (any case) for a check option, else None with an
        `info string`: anything else used to read as false, silently."""
        text = value.strip().lower()
        if text in ("true", "false"):
            return text == "true"
        _println(
            f"info string {name}: expected true/false, got {value.strip()!r}; "
            f"keeping {'true' if current else 'false'}"
        )
        return None

    def _path_note(self, applies_on: tuple[str, ...], when: str) -> str:
        """`` -- NO EFFECT ...`` when the live search path is not one of
        ``applies_on``; empty when the value reaches the live search."""
        path = self._worker.realized_search_path()
        if path in applies_on:
            return ""
        return f" — NO EFFECT on the live search (path={path}); used {when}"

    def _set_hash(self, value: str) -> None:
  # The handshake advertises `min 1024` but enforcement used to floor at 1, so a
  # GUI's conventional `Hash 256` was accepted and then quietly crippled the
  # engine: `advance_root` refuses cross-move reuse once the tree passes
  # `max_tree_bytes // 2`, and the initial reserve alone is ~8.6 MB — so any
  # Hash below ~17 MB kills reuse from move one and throttles every chunk.
  # Advertisement and enforcement now agree, and a clamp is announced.
        n = self._parse_clamped_int(value, "Hash")
        if n is None:
            return
        self._options.hash_mb = n
        self._worker.set_max_tree_mb(n)

    def _set_ponder(self, value: str) -> None:
        enabled = self._parse_check(value, "Ponder", self._options.ponder)
        if enabled is None:
            return
        self._options.ponder = enabled

    def _multi_gpu_path_active(self) -> bool:
        """True when a multi-GPU pool (RPG or PUCV) owns search dispatch."""
        return (
            self._options.search_parallel == "gumbel"
            or self._options.use_multi_gpu_pucv
        )

    def _set_threads(self, value: str) -> None:
        """`Threads`: accepted (GUIs send it) and reported as having no effect.

        DeepFin is a GPU engine; CPU threads are not its scaling axis, and the
        search algorithm must not follow a GUI's generic concurrency setting.
        Stored so the handshake echoes it back, and said out loud -- an accepted
        and silently ignored value is this repo's signature defect.
        """
        n = self._parse_clamped_int(value, "Threads")
        if n is None:
            return
        self._options.threads = n
        _println(
            f"info string Threads set to {n} — NO EFFECT on the search: the "
            f"search path is path={self._worker.realized_search_path()} "
            f"(SearchWalkers={self._options.search_walkers}); walker count is set "
            "only by SearchWalkers / --walkers"
        )

    def _set_search_walkers(self, value: str) -> None:
        """`SearchWalkers` (operator-only): 1 = classic Gumbel, >1 = PUCT walker pool.

        This is the ONLY setoption that selects between those two paths. It used
        to be `Threads`, which every GUI sends from its own concurrency setting,
        so a routine `Threads 8` silently moved a tournament engine off the
        validated Gumbel search onto a pool where most tuned knobs are inert.
        """
        n = self._parse_clamped_int(value, "SearchWalkers")
        if n is None:
            return
        old_walkers = self._options.search_walkers
        if self._multi_gpu_path_active():
            self._options.search_walkers = n
            self._worker.set_num_threads(n)
            # Worker stores n_walkers for leave-multi-GPU restore; live
            # dispatch still uses the RPG / multi-GPU PUCV pool.
            mode = (
                "SearchParallel=gumbel"
                if self._options.search_parallel == "gumbel"
                else "UseMultiGpuPUCV"
            )
            _println(
                f"info string SearchWalkers set to {n} "
                f"(deferred — {mode} is active)",
            )
            return
        # Crossing 1 <-> >1 changes which dispatcher tier the factory builds
        # (compiled: CUDAOwner at 1 walker vs BatchCoalescing at >1). The
        # walker-pool rebuild alone leaves the old wrapper in place — walkers
        # would then serialize through the owner queue (1->N) or lose the
        # pinned in-place path (N->1). Same-tier changes skip the rebuild.
        # The rebuild runs BEFORE any option or worker state moves, so a
        # failed one leaves the engine exactly as it was.
        new_eval = None
        if (
            (n == 1) != (old_walkers == 1)
            and self._rebuild_evaluator is not None
        ):
            _println(
                f"info string SearchWalkers {old_walkers}->{n} crosses dispatcher "
                "tier; rebuilding evaluator…"
            )
            new_eval = self._rebuild_or_keep(
                f"SearchWalkers {n}", self._options.max_batch,
                self._options.eval_cache_entries, n_walkers=n,
            )
            if new_eval is None:
                return
        # Both worker steps build before they swap, so a raise leaves the old
        # pool / evaluator live; undo the first if the second fails.
        try:
            self._worker.set_num_threads(n)
        except Exception:
            _close_quietly(new_eval)
            raise
        if new_eval is not None:
            try:
                self._worker.set_evaluator(new_eval)
            except Exception:
                self._worker.set_num_threads(old_walkers)
                raise
        self._options.search_walkers = n
        self._warmup_dirty = True
        if new_eval is not None:
            self._reinstall_configured_search_path()
        _println(
            f"info string SearchWalkers set to {n} "
            f"({'walker pool' if n > 1 else 'classic Gumbel path'})"
        )
        # The walker pool selects on raw Q and never reads c_scale/c_visit (Gumbel
        # sigma(q) only), so switching to SearchWalkers>1 silently drops the tuned
        # c_scale value. Surface it; SearchWalkers=1 restores the c_scale-tuned path.
        if n > 1:
            _println(
                "info string note: c_scale/c_visit are inactive on the walker pool; "
                "SearchWalkers=1 uses the c_scale-tuned classic Gumbel path"
            )

    def _set_leaf_gather(self, value: str) -> None:
        n = self._parse_clamped_int(value, "LeafGather")
        if n is None:
            return
        self._options.leaf_gather = n
        self._worker.set_walker_gather(n)
        self._warmup_dirty = True
        _println(
            f"info string LeafGather set to {n}"
            + self._path_note(("walker",), "by the SearchWalkers>1 walker pool")
        )

    def _set_use_vl(self, value: str) -> None:
        parsed = self._parse_check(value, "UseVL", self._options.use_vl)
        if parsed is None:
            return
        enabled = parsed
        self._options.use_vl = enabled
        self._worker.set_use_pucv(enabled, gather=self._options.vl_gather)
        if self._multi_gpu_path_active():
            mode = (
                "SearchParallel=gumbel"
                if self._options.search_parallel == "gumbel"
                else "UseMultiGpuPUCV"
            )
            _println(
                f"info string UseVL {'on' if enabled else 'off'} "
                f"(deferred — {mode} is active; "
                f"gather={self._options.vl_gather})",
            )
            return
        self._warmup_dirty = True
        _println(
            f"info string UseVL {'on' if enabled else 'off'} "
            f"(gather={self._options.vl_gather}; "
            f"requires SearchWalkers=1 + 2-slot async evaluator)"
        )

    def _set_use_multi_gpu_pucv(self, value: str) -> None:
        parsed = self._parse_check(value, "UseMultiGpuPUCV", self._options.use_multi_gpu_pucv)
        if parsed is None:
            return
        enabled = parsed
        if not enabled:
            had_pool = getattr(self._worker, "_pucv_pool", None) is not None
            # Size the primary for the remaining path: RPG still needs only
            # device 0; the routing/walker path uses all devices.
            # Build the replacement primary BEFORE tearing the pool down: a
            # failed build leaves pool, primary and options exactly as they were.
            new_eval = None
            if (
                self._rebuild_evaluator is not None
                and self._rebuild_multi_gpu_pucv_factories is not None
            ):
                previous = self._options.use_multi_gpu_pucv
                self._options.use_multi_gpu_pucv = False
                try:
                    new_eval = self._rebuild_or_keep(
                        "UseMultiGpuPUCV off", self._options.max_batch,
                        self._options.eval_cache_entries,
                    )
                finally:
                    self._options.use_multi_gpu_pucv = previous
                if new_eval is None:
                    return
            self._options.use_multi_gpu_pucv = False
            self._worker.clear_multi_gpu_pucv()
            if new_eval is not None:
                self._worker.set_evaluator(new_eval)
            self._warmup_dirty = True
            _println("info string UseMultiGpuPUCV off")
            # If SearchParallel still advertises gumbel, reinstall it — the
            # PUCV install tore the RPG pool down, and leaving search_parallel
            # sticky without a live pool would fall through to classic.
            if self._options.search_parallel == "gumbel":
                self._try_install_root_parallel_gumbel()
            elif had_pool:
                # install_multi_gpu_pucv tears down walker / UseVL; restore
                # them the same way leave-Gumbel does. Without this,
                # gumbel → UseMultiGpuPUCV on (forces sp=pucv) → off leaves
                # UseVL/SearchWalkers flags set but no live path materialised.
                self._reinstall_configured_search_path(after_leaving_gumbel=True)
            return
        n_devices = self._install_multi_gpu_pucv_pool(rebuild_primary=True)
        if n_devices is None:
            return
        # Mutual exclusion with SearchParallel=gumbel: the pool install clears
        # RPG, so the advertised mode must follow or the next MaxBatch
        # reinstall (which prefers search_parallel) would silently flip back.
        self._options.use_multi_gpu_pucv = True
        self._options.search_parallel = "pucv"
        self._warmup_dirty = True
        gather = min(self._options.vl_gather, self._options.max_batch)
        _println(
            f"info string UseMultiGpuPUCV on "
            f"(devices={n_devices} gather={gather})"
        )
        # Same inertness as the SearchWalkers>1 walker pool: the
        # pool selects with plain PUCT and never reads c_scale/c_visit, so the
        # tuned Gumbel values silently stop applying while it is installed.
        _println(
            "info string note: c_scale/c_visit are inactive on the multi-GPU "
            "PUCV pool (plain PUCT); a single device with SearchWalkers=1 restores "
            "the c_scale-tuned classic Gumbel path"
        )

    def _install_multi_gpu_pucv_pool(self, *, rebuild_primary: bool = False) -> int | None:
        """Install/reinstall the multi-GPU pool from current options.

        Returns device count on success, or None (and leaves
        ``use_multi_gpu_pucv`` False) when factories are missing or fewer
        than two devices. ``rebuild_primary`` stages a single-device primary
        after checking factory availability. Other reinstalls retain theirs.
        """
        if self._rebuild_multi_gpu_pucv_factories is None:
            self._options.use_multi_gpu_pucv = False
            _println("info string UseMultiGpuPUCV unavailable — no per-device factories wired")
            return None
        gather = min(self._options.vl_gather, self._options.max_batch)
        factories = self._rebuild_multi_gpu_pucv_factories(
            self._options.max_batch, gather,
        )
        if len(factories) < 2:
            self._options.use_multi_gpu_pucv = False
            _println("info string UseMultiGpuPUCV unavailable — need at least two devices")
            return None
        new_eval = None
        if rebuild_primary and self._rebuild_evaluator is not None:
            # The entry factory closes over these shared options. Construction
            # must see the intended topology, including when leaving RPG.
            # The setoption search barrier excludes readers; restore before
            # installing so only a successful transition commits the options.
            previous = self._options.use_multi_gpu_pucv, self._options.search_parallel
            self._options.use_multi_gpu_pucv = True
            self._options.search_parallel = "pucv"
            try:
                new_eval = self._rebuild_evaluator(
                    self._options.max_batch, self._options.eval_cache_entries,
                )
            finally:
                self._options.use_multi_gpu_pucv, self._options.search_parallel = previous
        # Pool first, primary swap last: if the pool install raises, the old
        # primary is still installed and the new one is retired here.
        try:
            self._worker.install_multi_gpu_pucv(
                factories, gather=gather, as_factories=True,
            )
        except BaseException:
            _close_quietly(new_eval)
            raise
        if new_eval is not None:
            self._worker.set_evaluator(new_eval, keep_pools=True)
        return len(factories)

    def _set_pucv_pending_mode(self, value: str) -> None:
        normalized = value.strip().lower().replace("_", "-")
        if normalized not in ("legacy", "virtual-mean"):
            return
        self._options.pucv_pending_mode = normalized
        self._worker.set_pucv_vloss_mode(1 if normalized == "virtual-mean" else 0)
        if self._options.search_parallel == "gumbel":
            # set_pucv_vloss_mode re-points the live RPG chunkers, so no
            # reinstall is needed — but don't let a stale use_multi_gpu_pucv
            # flag replace the Gumbel pool either. Report what the descent
            # will actually use, not what was requested.
            realized = self._worker.realized_rpg_vloss_mode()
            if realized is not None:
                _println(f"info string RPG vloss_mode={realized}")
        elif self._options.use_multi_gpu_pucv:
            # Pool config changed (vloss_mode); reinstall only — primary stays.
            self._install_multi_gpu_pucv_pool()
        _println(f"info string PUCVPendingMode set to {normalized}")

    def _set_search_parallel(self, value: str) -> None:
        """Switch the multi-GPU search mode: ``gumbel`` installs the
        root-parallel Gumbel pool (>= 2 device factories required, else the
        option reverts to ``pucv``); ``pucv`` tears it down and restores the
        UseMultiGpuPUCV / SearchWalkers / UseVL routing. Single-device configs never
        construct the new module — dispatch falls through to the classic
        paths untouched."""
        normalized = value.strip().lower()
        if normalized not in ("pucv", "gumbel"):
            return
        if normalized == "gumbel":
            if not self._try_install_root_parallel_gumbel(
                rebuild_primary=self._options.search_parallel != "gumbel",
            ):
                return
            return
        was_gumbel = self._options.search_parallel == "gumbel"
        # Build the replacement primary BEFORE clearing the RPG pool, under the
        # target topology: a failed build leaves the pool and options as they were.
        new_eval = None
        if (was_gumbel and not self._options.use_multi_gpu_pucv
                and self._rebuild_evaluator is not None):
            previous = self._options.search_parallel
            self._options.search_parallel = "pucv"
            try:
                new_eval = self._rebuild_or_keep(
                    "SearchParallel pucv", self._options.max_batch,
                    self._options.eval_cache_entries,
                )
            finally:
                self._options.search_parallel = previous
            if new_eval is None:
                return
        self._options.search_parallel = "pucv"
        self._worker.clear_root_parallel_gumbel()
        if new_eval is not None:
            self._worker.set_evaluator(new_eval)
        self._warmup_dirty = True
        _println("info string SearchParallel pucv")
        # Restore whichever non-RPG path is configured. install_root_parallel
        # gumbel closed the walker pool and cleared single-thread PUCV, so
        # leaving only clear_root_parallel_gumbel would fall through to
        # single-thread classic Gumbel even with SearchWalkers>1 / UseVL / PUCV on.
        self._reinstall_configured_search_path(after_leaving_gumbel=True)

    def _try_install_root_parallel_gumbel(self, *, rebuild_primary: bool = False) -> bool:
        """Install the RPG pool from current MaxBatch/VLGather. On success
        sets ``search_parallel=gumbel`` and suppresses the PUCV auto-enable
        flag so later MaxBatch rebuilds reinstall Gumbel, not PUCV. Returns
        False (and reverts the option to pucv) when factories are missing
        or fewer than two devices are available."""
        if self._rebuild_multi_gpu_pucv_factories is None:
            self._options.search_parallel = "pucv"
            _println(
                "info string SearchParallel gumbel unavailable — "
                "no per-device factories wired",
            )
            return False
        gather = min(self._options.vl_gather, self._options.max_batch)
        factories = self._rebuild_multi_gpu_pucv_factories(
            self._options.max_batch, gather,
        )
        if len(factories) < 2:
            self._options.search_parallel = "pucv"
            _println(
                "info string SearchParallel gumbel unavailable — "
                "need at least two devices",
            )
            return False
        new_eval = None
        if rebuild_primary and self._rebuild_evaluator is not None:
            # Preflight factories first, then let the primary factory observe
            # the intended topology, as in the multi-GPU PUCV transition.
            previous = self._options.search_parallel
            self._options.search_parallel = "gumbel"
            try:
                new_eval = self._rebuild_evaluator(
                    self._options.max_batch, self._options.eval_cache_entries,
                )
            finally:
                self._options.search_parallel = previous
        previous_mode = self._options.search_parallel
        self._options.search_parallel = "gumbel"
        # search_parallel is the source of truth while Gumbel is selected:
        # _reinstall_configured_search_path prefers it over use_multi_gpu_pucv
        # so MaxBatch/EvalCache/VLGather rebuilds reinstall RPG, not PUCV.
        # The use_multi_gpu_pucv flag is left intact so switching back to
        # SearchParallel=pucv can restore the multi-GPU PUCV pool.
        # Pool first, primary swap last (see _install_multi_gpu_pucv_pool).
        try:
            self._worker.install_root_parallel_gumbel(
                factories, gather=gather, as_factories=True,
                devices=list(self._search_devices) if self._search_devices else None,
                info_string_cb=_emit_info_string,
                open_vpa=int(self._options.rpg_open_vpa),
                min_vpa=int(self._options.rpg_min_vpa),
                min_keep=int(self._options.rpg_min_keep),
                open_budget_frac=float(self._options.rpg_open_budget_frac),
            )
        except BaseException:
            self._options.search_parallel = previous_mode
            _close_quietly(new_eval)
            raise
        if new_eval is not None:
            self._worker.set_evaluator(new_eval, keep_pools=True)
        self._warmup_dirty = True
        _println(
            f"info string SearchParallel gumbel on "
            f"(groups={len(factories)} gather={gather} "
            f"open_vpa={self._options.rpg_open_vpa} "
            f"min_vpa={self._options.rpg_min_vpa} "
            f"min_keep={self._options.rpg_min_keep} "
            f"vloss_mode={self._worker.realized_rpg_vloss_mode()})",
        )
        return True

    def _reinstall_rpg_if_active(self, *, reason: str) -> None:
        """Re-apply root-parallel Gumbel when schedule knobs change mid-run."""
        if self._options.search_parallel != "gumbel":
            return
        if not self._try_install_root_parallel_gumbel():
            return
        _println(f"info string RPG reinstalled after {reason}")

    def _set_rpg_open_vpa(self, value: str) -> None:
        n = self._parse_clamped_int(value, "RPGOpenVPA")
        if n is None:
            return
        self._options.rpg_open_vpa = n
        _println(
            f"info string RPGOpenVPA set to {self._options.rpg_open_vpa}"
            + self._path_note(("rpg",), "when SearchParallel=gumbel installs the root-parallel pool")
        )
        self._reinstall_rpg_if_active(reason="RPGOpenVPA")

    def _set_rpg_min_vpa(self, value: str) -> None:
        n = self._parse_clamped_int(value, "RPGMinVPA")
        if n is None:
            return
        self._options.rpg_min_vpa = n
        _println(
            f"info string RPGMinVPA set to {self._options.rpg_min_vpa}"
            + self._path_note(("rpg",), "when SearchParallel=gumbel installs the root-parallel pool")
        )
        self._reinstall_rpg_if_active(reason="RPGMinVPA")

    def _set_rpg_min_keep(self, value: str) -> None:
        n = self._parse_clamped_int(value, "RPGMinKeep")
        if n is None:
            return
        self._options.rpg_min_keep = n
        _println(
            f"info string RPGMinKeep set to {self._options.rpg_min_keep}"
            + self._path_note(("rpg",), "when SearchParallel=gumbel installs the root-parallel pool")
        )
        self._reinstall_rpg_if_active(reason="RPGMinKeep")

    def _set_rpg_open_budget_frac(self, value: str) -> None:
        try:
            frac = float(value.strip())
        except ValueError:
            frac = math.nan
        if not (0.0 < frac <= 1.0):
            _println(
                f"info string RPGOpenBudgetFrac: {value.strip()!r} is not a number in "
                f"(0, 1]; keeping {self._options.rpg_open_budget_frac}"
            )
            return
        self._options.rpg_open_budget_frac = frac
        _println(
            f"info string RPGOpenBudgetFrac set to {frac}"
            + self._path_note(("rpg",), "when SearchParallel=gumbel installs the root-parallel pool")
        )
        self._reinstall_rpg_if_active(reason="RPGOpenBudgetFrac")

    def _sync_cpuct_to_worker(self) -> None:
        """Push CPuct* options into SearchWorker GumbelConfig + live tree."""
        cfg = self._worker._cfg
        cfg.c_puct = float(self._options.cpuct)
        cfg.cpuct_factor = float(self._options.cpuct_factor)
        cfg.cpuct_base = float(self._options.cpuct_base)
        tree = getattr(self._worker, "_tree", None)
        if tree is not None:
            tree.set_cpuct_scaling(cfg.cpuct_factor, cfg.cpuct_base)
        # Rebuild every helper that CACHES the family at construction time.
        # One branch per realized path, in `realized_search_path()` order, so a
        # path cannot be reported [LIVE] with nothing here to make it so. The
        # walker pool and the single-thread PucvChunker used to fall off the
        # end of this chain -- `set_use_pucv(True)` early-returns when pucv is
        # already on, and there was no walker branch at all -- so on the
        # then-shipped default (Threads=2 -> walker) four options `searchconfig`
        # certifies as [LIVE] could not be moved by a `setoption`.
        if self._options.search_parallel == "gumbel":
            self._reinstall_rpg_if_active(reason="CPuct")
        elif self._options.use_multi_gpu_pucv:
            self._install_multi_gpu_pucv_pool()
        else:
            self._worker.rebuild_puct_helpers()

    def _reinstall_configured_search_path(
        self, *, after_leaving_gumbel: bool = False,
    ) -> None:
        """Reinstall the search path that ``search_parallel`` / multi-GPU /
        SearchWalkers / UseVL currently advertise.

        Used after evaluator rebuilds (MaxBatch, EvalCacheEntries) and after
        VLGather reshape so Gumbel mode is not silently dropped when
        ``set_evaluator`` clears ``_rpg_pool``. Also used when leaving Gumbel
        so walker/PUCV come back when multi-GPU PUCV is off.
        """
        if self._options.search_parallel == "gumbel":
            if not self._try_install_root_parallel_gumbel():
                # Install failed (e.g. factories went away) — fall through to
                # whatever non-Gumbel path is still configured.
                self._reinstall_configured_search_path(after_leaving_gumbel=True)
            return
        if self._options.use_multi_gpu_pucv:
            if after_leaving_gumbel:
                # Full enable (may shrink primary to device 0) when returning
                # from Gumbel to the multi-GPU PUCV pool.
                self._set_use_multi_gpu_pucv("true")
            else:
                # MaxBatch/VLGather already rebuilt the primary; reinstall pool only.
                self._install_multi_gpu_pucv_pool()
            return
        if after_leaving_gumbel:
            # Rebuild walker / single-thread PUCV from the options that
            # install_root_parallel_gumbel tore down (n_walkers / _use_pucv
            # flags survive install; only the live pool/chunker was cleared).
            self._worker.restore_classic_search_path()
            if self._options.use_vl:
                # Toggle so gather is applied even if _use_pucv was already True
                # (set_use_pucv early-returns when the chunker is non-None).
                self._worker.set_use_pucv(False)
                self._worker.set_use_pucv(True, gather=self._options.vl_gather)

    def _set_vl_gather(self, value: str) -> None:
        n = self._parse_clamped_int(value, "VLGather")
        if n is None:
            return
        self._options.vl_gather = n
        if self._options.search_parallel == "gumbel":
            self._reinstall_configured_search_path()
        else:
            if self._options.use_vl:
                self._worker.set_use_pucv(True, gather=n)
            if self._options.use_multi_gpu_pucv:
                # gather is a pool-construction param; reinstall without primary rebuild.
                self._install_multi_gpu_pucv_pool()
        # VLGather reshapes the pucv / multi-GPU / RPG per-worker batch, the
        # same way LeafGather reshapes the walker batch — re-warm the path.
        self._warmup_dirty = True
        _println(
            f"info string VLGather set to {n}"
            + self._path_note(
                ("pucv", "pucv_pool", "rpg"),
                "by UseVL / UseMultiGpuPUCV / SearchParallel=gumbel",
            )
        )

    def _set_multi_pv(self, value: str) -> None:
        n = self._parse_clamped_int(value, "MultiPV")
        if n is None:
            return
        self._options.multi_pv = n
        self._worker.set_multi_pv(n)

    def _set_show_wdl(self, value: str) -> None:
        enabled = self._parse_check(value, "UCI_ShowWDL", self._options.show_wdl)
        if enabled is None:
            return
        self._options.show_wdl = enabled
        self._worker.set_show_wdl(self._options.show_wdl)

    def _set_move_overhead_ms(self, value: str) -> None:
        n = self._parse_clamped_int(value, "MoveOverheadMs")
        if n is None:
            return
        self._options.move_overhead_ms = n

    def _set_clock_batch_margin_sigmas(self, value: str) -> None:
        try:
            s = float(value.strip())
        except ValueError:
            s = math.nan
        if not math.isfinite(s):
            _println(
                f"info string ClockBatchMarginSigmas: {value.strip()!r} is not a number; "
                f"keeping {self._options.clock_batch_margin_sigmas}"
            )
            return
        if s < 0.0:
            _println(f"info string ClockBatchMarginSigmas {s} is below 0; using 0.0")
        s = max(0.0, s)
        self._options.clock_batch_margin_sigmas = s
        self._worker.set_batch_margin_sigmas(s)

    def _set_syzygy_50_move_rule(self, value: str) -> None:
        enabled = self._parse_check(
            value, "Syzygy50MoveRule", self._options.syzygy_50_move_rule,
        )
        if enabled is None:
            return
        self._options.syzygy_50_move_rule = enabled
  # Re-install probe so the new semantics take effect for the
  # next search. Cheap — just wraps the same path with different
  # cursed/blessed handling. The probe carries the rule to BOTH tablebase
  # paths: the leaf probes and the root DTZ shortcut (`cursed_as_draw`).
        if self._options.syzygy_path:
            self._install_tablebase(self._options.syzygy_path)
        _println(
            f"info string Syzygy50MoveRule set to {'true' if enabled else 'false'}"
            + ("" if self._options.syzygy_path else "; takes effect once SyzygyPath is set")
        )

    def _set_log_file(self, value: str) -> None:
        path = value.strip()
        self._options.log_file = path
        _attach_log_file(path)

    def _set_max_batch(self, value: str) -> None:
        mb = self._parse_clamped_int(value, "MaxBatch")
        if mb is None:
            return
        if self._rebuild_evaluator is None:
            _println("info string MaxBatch ignored — no evaluator factory wired")
            return
        if mb == self._options.max_batch:
            return
  # Rebuild is 5-10s (CUDA graph recapture on first forward).
  # User sees the stall only if they poll isready soon after.
        _println(f"info string MaxBatch rebuilding evaluator at {mb}…")
        new_eval = self._rebuild_or_keep(
            f"MaxBatch {mb}", mb, self._options.eval_cache_entries,
        )
        if new_eval is None:
            return
        self._install_rebuilt_primary(new_eval, max_batch=mb)
        # The rebuilt evaluator's raw forward is warmed, but the SEARCH-path
        # cudagraph is captured separately — re-warm it before the clock.
        self._warmup_dirty = True
        _println(f"info string MaxBatch set to {mb}; evaluator rebuilt + warmed")

    def _set_eval_cache_entries(self, value: str) -> None:
        n = self._parse_clamped_int(value, "EvalCacheEntries")
        if n is None:
            return
        if n == self._options.eval_cache_entries:
            return
        if self._rebuild_evaluator is None:
            _println("info string EvalCacheEntries ignored — no evaluator factory wired")
            return
        _println(f"info string EvalCacheEntries rebuilding evaluator at {n}")
        new_eval = self._rebuild_or_keep(
            f"EvalCacheEntries {n}", self._options.max_batch, n,
        )
        if new_eval is None:
            return
        previous = self._options.eval_cache_entries
        self._worker.set_eval_cache_entries(n)
        try:
            self._install_rebuilt_primary(new_eval, eval_cache_entries=n)
        except BaseException:
            self._worker.set_eval_cache_entries(previous)
            raise
        # Same as MaxBatch: evaluator rebuilt cold, so the search-path
        # cudagraph must be re-warmed before the first real ``go``.
        self._warmup_dirty = True
        _println(f"info string EvalCacheEntries set to {n}; evaluator rebuilt + warmed")

    def _install_rebuilt_primary(self, new_eval: BatchEvaluator, **option_updates: int) -> None:
        """Commit ``option_updates`` and make ``new_eval`` the primary, all or nothing.

        Classic path: ``set_evaluator`` builds the walker / PUCV helpers before
        it swaps, so a raise leaves the old evaluator live. Multi-GPU path: the
        pool is rebuilt from the new options FIRST and the primary swapped in
        last with the pool kept; if the pool rebuild raises, the old primary is
        still installed and the setoption transaction in `dispatch` reinstalls
        the previous pool. Options are restored here on any raise.
        """
        previous = {k: getattr(self._options, k) for k in option_updates}
        for key, value in option_updates.items():
            setattr(self._options, key, value)
        handed_over = False
        try:
            if self._multi_gpu_path_active():
                self._reinstall_configured_search_path()
                handed_over = True
                self._worker.set_evaluator(new_eval, keep_pools=True)
            else:
                handed_over = True
                self._worker.set_evaluator(new_eval)
        except BaseException:
            for key, value in previous.items():
                setattr(self._options, key, value)
            if not handed_over:
                _close_quietly(new_eval)
            raise

    def _rebuild_or_keep(
        self, what: str, max_batch: int, eval_cache_entries: int,
        *, n_walkers: int | None = None,
    ) -> BatchEvaluator | None:
        """A replacement evaluator, or ``None`` with the current one kept.

        Every caller builds BEFORE it changes an option or calls
        ``set_evaluator``, so on failure the worker still holds the previous,
        working evaluator and every setting still describes it. A raise here
        used to escape the handler: the process died (``MaxBatch`` OOM) or, once
        dispatch is guarded, the options would claim a batch size nothing was
        built for. Say so instead and keep playing on the last good setup.
        """
        assert self._rebuild_evaluator is not None
        try:
            if n_walkers is None:
                return self._rebuild_evaluator(max_batch, eval_cache_entries)
            return self._rebuild_evaluator(
                max_batch, eval_cache_entries, n_walkers=n_walkers,
            )
        except Exception as exc:
            _log.error("evaluator rebuild for %s failed", what, exc_info=True)
            reason = _one_line(exc)
        # Outside the except block, so the traceback (and the half-built
        # evaluator's buffers it references) is already released.
        gc.collect()
        _println(
            f"info string {what}: evaluator rebuild failed ({reason}); "
            "keeping the previous evaluator and settings"
        )
        return None

    def _set_syzygy_path(self, value: str | None) -> None:
        v = (value or "").strip()
  # Conventional UCI sentinel for "unset".
        if v.lower() in ("", "<empty>"):
            v = ""
        self._options.syzygy_path = v
        self._install_tablebase(v)

    # Options whose handler writes only a reporting field read at the next info
    # emission; safe to apply while an orphan is still inside the worker. They
    # still pass through the setoption transaction (snapshot, restore on
    # failure), which is safe beside a live orphan only because these handlers
    # cannot raise and never change the search path. A handler that can raise
    # or rebuild anything must NOT be added here.
    _REPORTING_ONLY_OPTIONS: ClassVar[frozenset[str]] = frozenset({"multipv", "uci_showwdl"})

    _SETOPTION_HANDLERS: ClassVar[dict[str, Callable[[Engine, str], None]]] = {
        "hash": _set_hash,
        "ponder": _set_ponder,
        "threads": _set_threads,
        "searchwalkers": _set_search_walkers,
        "leafgather": _set_leaf_gather,
        "usevl": _set_use_vl,
        "usemultigpupucv": _set_use_multi_gpu_pucv,
        "pucvpendingmode": _set_pucv_pending_mode,
        "searchparallel": _set_search_parallel,
        "rpgopenvpa": _set_rpg_open_vpa,
        "rpgminvpa": _set_rpg_min_vpa,
        "rpgminkeep": _set_rpg_min_keep,
        "rpgopenbudgetfrac": _set_rpg_open_budget_frac,
        "vlgather": _set_vl_gather,
        "multipv": _set_multi_pv,
        "uci_showwdl": _set_show_wdl,
        "moveoverheadms": _set_move_overhead_ms,
        "clockbatchmarginsigmas": _set_clock_batch_margin_sigmas,
        "syzygy50moverule": _set_syzygy_50_move_rule,
        "logfile": _set_log_file,
        "maxbatch": _set_max_batch,
        "evalcacheentries": _set_eval_cache_entries,
    }

    def _install_tablebase(self, path: str) -> None:
        """Validate ``path`` by opening the tablebase once, then install a
        path-backed SyzygyProbe on the worker (or clear it if ``path`` is
        empty). The open handle is cached in ``chess_anti_engine.tablebase``
        and shared with training-time rescoring.

        Calling ``get_tablebase`` unconditionally — including for empty paths
        — is what closes any previously-cached handle when the user clears
        or swaps SyzygyPath.
        """
        tb = get_tablebase(path)
        if not path:
            self._worker.set_tb_probe(None)
            _println("info string SyzygyPath cleared; tablebase probing disabled")
            return
        if tb is None:
            self._worker.set_tb_probe(None)
            _println(f"info string SyzygyPath {path!r} did not open any tablebase directories")
            return
        probe = SyzygyProbe(
            path, cursed_as_draw=self._options.syzygy_50_move_rule,
        )
        self._worker.set_tb_probe(probe)
        _println(
            f"info string SyzygyPath loaded from {path!r}: "
            f"{probe.n_wdl} WDL + {probe.n_dtz} DTZ tables, "
            f"up to {probe.max_pieces}-piece positions"
        )
        self._leaf_tb_noted = None
        self._note_leaf_tb_inactive()

    def _note_leaf_tb_inactive(self) -> None:
        """Say once per mode that the active search path ignores SyzygyPath at leaves."""
        mode = self._worker.leaf_tb_inactive_mode()
        if mode is not None and mode != self._leaf_tb_noted:
            _println(
                f"info string note: leaf tablebase probing is inactive on search path {mode}; "
                "SyzygyPath reaches only the root move shortcut and root eval on this path"
            )
        self._leaf_tb_noted = mode

  # -- search thread body ---------------------------------------------------

    def warmup_search(self) -> None:
        """Keep public warmup custody separate from the background go thread."""
        with self._state_lock:
            if self._quit_requested or self._closed:
                return
            if not self._warmup_done.is_set():
                raise SearchNotQuiescent("warmup already owns the worker")
            if not self._search_free_locked():
                raise SearchNotQuiescent("a search owns the worker; warmup deferred")
            self._warmup_done.clear()
        try:
            self._warmup_search_owned()
        finally:
            self._warmup_done.set()

    def _warmup_search_owned(self) -> None:
        """Run one tiny real search on the start position at startup so the first
        actual ``go`` doesn't pay torch.compile + cudagraph capture mid-move.

        The forward warmup (``_warmup_evaluator``) only exercises the raw
        evaluator at batches {1, 128}; the gumbel *search* path submits evals
        through a different entry, so its cudagraph is captured on the first real
        search (~3-4s). At a short first-move budget that overruns the clock — a
        move-1 time forfeit (a real TCEC game's long base time would otherwise
        hide it). This runs through the same worker + dispatcher a real ``go``
        uses, so the search-path graphs are captured on the dispatcher's
        persistent submitter thread and every later move just replays them.
        One full chunk (the configured ``chunk_sims``) covers the real batch
        shapes. STRICT on every path: a failure raises. A best-effort warmup
        used to answer ``readyok`` over a broken search, and every move after
        that was the unsearched root-policy fallback.

        Multi-GPU PUCV caveat: that pool fans a single ``run`` budget across
        every GPU worker through one shared semaphore (each greedily grabs up
        to ``gather`` tokens). A lone ``max(256, chunk)`` chunk can drain
        before the later workers grab anything, so they never capture their
        own search-path cudagraph and the first real ``go`` pays cold capture
        on those devices. When the pool is active, size the warmup to the same
        per-device budget the real search uses (``search.py:_chunk_budget``:
        ``max(chunk, gather)`` per device) so every worker gets a full batch.

        Root-parallel Gumbel: work units are *candidates*, not sim tokens, so
        raising max_nodes cannot create more first-phase jobs than
        ``min(topk, n_legal)`` (20 on the start position). Production configs
        (topk≥16, ≤8 devices) fill every group on startpos; when groups outrun
        first-phase candidates some stay cold (factory-level
        ``_warmup_pucv_evaluator`` still covers raw eval shapes). We still size
        the budget so each active group can complete a full expand + gather
        batch during the first phase."""
        chunk = int(getattr(self._worker, "_chunk_sims", 512))
        max_nodes = max(256, chunk)
        pucv_pool = getattr(self._worker, "_pucv_pool", None)
        if pucv_pool is not None:
            n_devices = getattr(pucv_pool, "n_devices", None)
            gather = getattr(pucv_pool, "gather", None)
            if n_devices and gather:
                per_device = max(chunk, int(gather))
                max_nodes = max(max_nodes, per_device * int(n_devices))
        rpg_pool = getattr(self._worker, "_rpg_pool", None)
        if rpg_pool is not None:
            n_groups = int(getattr(rpg_pool, "n_groups", 0) or 0)
            rpg_cfg = getattr(rpg_pool, "_cfg", None)
            gather = int(getattr(rpg_cfg, "gather", 0) or 0)
            if n_groups > 0 and gather > 0:
                # Enough sims that first-phase vpa can cover expand + one
                # 2×gather batch for every candidate that gets a group.
                per_group = max(chunk, 1 + 2 * gather)
                max_nodes = max(max_nodes, per_group * n_groups)
        try:
            self._worker.run(
                chess.Board(),
                stop_event=threading.Event(),
                deadline=Deadline(deadline_ms=None),
                max_nodes=max_nodes,
                optimum_ms=None,
                abort_factor=self._options.abort_factor,
            )
        finally:
            self._worker.reset_tree()

    def _run_search(
        self, limits: SearchLimits, gen: int, board: chess.Board,
        go_received_at: float | None = None,
        *, cancel_event: threading.Event | None = None,
        ponder_release: threading.Event | None = None,
    ) -> None:
  # THIS go's events, handed over at thread creation. Every later `go` replaces
  # `self._cancel_event` / `self._ponder_release`; a search parked after `stop`
  # that returned later and then waited on the NEW, never-set event would stay
  # an orphan for good -- every later move a busy-worker fallback.
        cancel = cancel_event if cancel_event is not None else self._cancel_event
        release = ponder_release if ponder_release is not None else self._ponder_release
        # The predicted reply this ponder search stopped one ply short of; read
        # now, before ponderhit's handoff clears it.
        popped_at_start = self._popped_ponder_move if limits.ponder else None
  # Ponder search: no deadline yet; runs until ponderhit or stop.
        result = self._run_one_phase(
            limits, is_ponder=limits.ponder, board=board, clock_started_at=go_received_at,
            gen=gen,
        )
  # The worker can return before the GUI does. A mated root, a tablebase
  # hit, a declined draw, a full hash, or a search exception is not a
  # ponder release: this phase is still the position *before* the predicted
  # reply, and its bestmove belongs to the opponent. UCI stays in ponder
  # until ``stop`` or ``ponderhit``. Those commands set ``_ponder_release``.
  # ``_stop_event`` does not: a pool fault sets it and then raises, and
  # treating that as a release publishes the pre-reply fallback.
        if limits.ponder and not release.is_set():
            release.wait()
        ponderhit = self._ponderhit_event.is_set() and self._pending_real_limits is not None
        if self._ponder_fallback_pending and self._counts_for_streak():
            # Gated like the streak: a parked ponder search must neither print
            # this warning nor consume the flag the live search owns.
            self._ponder_fallback_pending = False
            if not ponderhit:  # stop: the ponder fallback IS the published move
                self._note_fallback_phase()
                _println(
                    "info string WARNING search failed on "
                    f"{self._consecutive_search_failures} consecutive move(s); this move "
                    "is UNSEARCHED (fallback)"
                )
        if ponderhit and self._pending_real_limits is not None:
  # Ponderhit: opponent played our predicted move. Advance root by
  # one ply (the popped move) so the real phase searches at the
  # actual current position, reusing sims the ponder accumulated
  # below that child.
            # Ponderhit stops the first phase; GUI cancellation survives the
            # phase switch through a separate event which is never cleared.
            self._stop_event = self._cancel_event
            started_at = self._ponderhit_at
            if started_at is None:
                started_at = time.monotonic()
            real_limits = self._pending_real_limits
            self._pending_real_limits = None
            self._ponderhit_event.clear()
            real_board = board.copy(stack=True)
            popped = self._popped_ponder_move
            self._popped_ponder_move = None
            if popped is not None:
                real_board.push(popped)
            transition_deadline = Deadline(real_limits.deadline_ms, now=started_at)
            if (popped is not None and not self._cancel_event.is_set()
                    and not transition_deadline.expired()):
                if not self._worker.advance_root(board, [popped]):
  # Defensive: ponder-at-M root-expanded the popped child, so
  # this shouldn't fail. Reset rather than search stale tree.
                    self._worker.reset_tree()
                self._applied_moves = (*self._applied_moves, popped)
            result = self._run_one_phase(
                real_limits, is_ponder=False, board=real_board, started_at=started_at,
                gen=gen,
            )
            limits = real_limits
        elif limits.ponder and popped_at_start is not None:
            # `stop` during ponder (a ponder miss): the search ran on the board
            # BEFORE the predicted reply, so its move is the opponent's. UCI
            # still owes a bestmove for the GUI's position (that board plus the
            # predicted reply) -- a GUI may check it is legal before discarding it.
            result = self._ponder_miss_result(result, board, popped_at_start)
        # A completed infinite search is still owned by the GUI until stop.
        # Pool failures can set _stop_event without cancelling that ownership.
        # After ponderhit, use the real phase's limits so finite searches emit.
        if not limits.ponder and limits.is_open_ended():
            cancel.wait()
  # Gate the emit: if a newer `go` has started, our result is stale
  # and the new search owns bestmove. Silently drop.
        if not self._claim_answer(gen):
            return  # a newer go, quit, `stop`'s parked answer or the watchdog owns it
        self._watchdog.disarm(gen)
        with _answer_owner():
            self._emit_bestmove(result)

  # -- unhealthy engine: streak, boundary rebuild, strict re-check ------------
  #
  # A search phase that cannot produce a searched move plays the fallback (never
  # lose on time) and extends the streak. Nothing is rebuilt mid-game: a rebuild
  # beside a live search would double VRAM exactly when the fault may have been an
  # OOM, and its safety rests on GPU behaviour no CPU test can check. After
  # `_UNHEALTHY_FALLBACK_STREAK` fallbacks in a row the engine is marked unhealthy
  # (one WARNING); at the next game boundary -- worker free, no clock running --
  # it rebuilds the evaluator synchronously and reruns the strict startup check,
  # exiting 1 only if that check fails.

    def _mark_unhealthy(self, reason: str) -> None:
        with self._state_lock:
            first = self._unhealthy is None
            if first:
                self._unhealthy = reason
        if first:
            _println(
                f"info string WARNING engine unhealthy: {reason}; every go is still "
                "answered; at the next game boundary the evaluator is rebuilt and the "
                "strict startup check rerun, and the engine exits if it fails"
            )

    def _game_boundary_checks(self) -> bool:
        """Between games, worker free. Unhealthy: rebuild, then strict re-check.

        Returns False when the re-check failed and the engine is exiting.
        """
        if self._unhealthy is None:
            return True
        reason = self._unhealthy
        _println(
            f"info string engine unhealthy ({reason}): rebuilding the evaluator and "
            "rerunning the strict startup check before the next game"
        )
        named_before = self._paths_named_by(replace(self._options))
        try:
            self._rebuild_for_health_check()
        except Exception as exc:
            # One outcome, reported once: the strict re-check below still decides.
            _log.error("health rebuild raised", exc_info=True)
            _println(f"info string health rebuild raised ({_one_line(exc)}); "
                     "the strict re-check decides")
        try:
            realized = self._worker.realized_search_path()
        except Exception:
            realized = None
        degraded = None
        if isinstance(realized, str) and realized not in named_before:
            degraded = (f"realized path={realized} but the options named "
                        f"{'/'.join(named_before)} before the rebuild")
        return self._recheck_health(degraded=degraded)

    def _rebuild_for_health_check(self) -> None:
        """Build a fresh evaluator for the configured path and install it.

        Build-then-swap: a failed build or install keeps the current evaluator,
        and the strict re-check that follows decides. The path invariant runs
        afterwards exactly as after a setoption.
        """
        if self._rebuild_evaluator is None:
            _println("info string health rebuild skipped: no evaluator factory wired")
            return
        new_eval = self._rebuild_or_keep(
            "health rebuild", self._options.max_batch, self._options.eval_cache_entries,
        )
        if new_eval is None:
            return
        # Transactional like a setoption: a failed install (which on a multi-GPU
        # path may already have torn the pool down) restores the previous
        # options, worker search state and pool via the same restore.
        snapshot = (replace(self._options), self._worker_search_state())
        try:
            self._install_rebuilt_primary(new_eval)
        except Exception as exc:
            _log.error("installing the health-rebuild evaluator failed", exc_info=True)
            _println(f"info string health rebuild: install failed ({_one_line(exc)})")
            self._restore_after_failed_setoption(snapshot, what="health rebuild")
            return
        self._enforce_path_invariant(after="health rebuild")
        self._warmup_dirty = True
        _println("info string health rebuild: evaluator rebuilt")

    def _note_unsearched_reply(self) -> None:
        # A reply-only or busy-worker fallback is a published unsearched move: it
        # extends the streak, so a `go` that always fails still goes unhealthy.
        self._note_fallback_phase()

    def _recheck_health(self, *, degraded: str | None = None) -> bool:
        """Rerun the strict startup warmup; False = failed, engine is exiting.

        Called only from `_game_boundary_checks` (worker free, between games).
        """
        reason = self._unhealthy
        try:
            self.warmup_search()
        except Exception as exc:
            _log.error("health re-check failed", exc_info=True)
            text = " ".join(f"{type(exc).__name__}: {exc}".split())[:500]
            _println(f"info string engine health check failed ({reason}): {text}; exiting")
            print(f"deepfin: engine health check failed: {text}", file=sys.stderr, flush=True)
            self._exit_code = 1
            with self._state_lock:
                self._quit_requested = True
                self._search_gen += 1
            return False
        self._warmup_dirty = False
        self._unhealthy = None
        self._consecutive_search_failures = 0
        if degraded is not None:
            _println(
                f"info string WARNING engine health check passed on a DEGRADED path "
                f"({degraded}); unhealthy state cleared"
            )
        else:
            _println("info string engine health check passed; unhealthy state cleared")
        return True

    @property
    def exit_code(self) -> int:
        """Process exit code the UCI loop should use (1 after a failed health re-check)."""
        return self._exit_code

    def _claim_answer(self, gen: int) -> bool:
        """Become the one writer of generation ``gen``'s bestmove, or False.

        The winner prints its bestmove inside `_answer_owner()`: a thread parked
        between its claim and its print is silenced, but the answer it owns
        must still reach the GUI, since nobody else will answer that go."""
        with self._state_lock:
            if self._quit_requested or gen != self._search_gen or self._claimed_gen == gen:
                return False
            self._claimed_gen = gen
            return True

    @staticmethod
    def _watchdog_last_moment_ms() -> float:
        return _WATCHDOG_LAST_MOMENT_MS

    def _arm_watchdog(
        self, gen: int, *, started_at: float, deadline_ms: int | None,
        budget_ms: int | None,
        board: chess.Board, searchmoves: tuple[str, ...],
        thread: threading.Thread | None, stops: tuple[threading.Event, ...],
    ) -> None:
        """Arm for a clocked phase; an unclocked one (infinite, ponder, nodes,
        depth: ``deadline_ms is None``) gets no watchdog."""
        if deadline_ms is None or thread is None:
            return
        deadline_at = started_at + float(deadline_ms) / 1000.0
        last_moment_at = None
        if budget_ms is not None:
            last_moment_at = max(
                deadline_at,
                started_at + (float(budget_ms) - self._watchdog_last_moment_ms()) / 1000.0,
            )
        self._watchdog.arm(_WatchdogEntry(
            gen=gen, check_at=deadline_at, deadline_at=deadline_at,
            last_moment_at=last_moment_at, budget_ms=budget_ms,
            board=board.copy(), searchmoves=searchmoves,
            thread=thread, stops=stops, started_at=started_at,
        ))

    def _watchdog_stall_ms(self, chunk_ms: float | None, *, any_chunk: bool) -> float:
        """How long without a completed chunk counts as a stall."""
        if chunk_ms is None:
            chunk_ms = self._prev_chunk_ms if any_chunk else self._prev_first_chunk_ms
        if chunk_ms is None:
            return _WATCHDOG_STALL_NO_ESTIMATE_MS
        return min(_WATCHDOG_STALL_CEIL_MS,
                   max(_WATCHDOG_STALL_FLOOR_MS, _WATCHDOG_STALL_CHUNKS * chunk_ms))

    def _watchdog_poll(self, entry: _WatchdogEntry) -> float | None:
        """Runs on the watchdog thread at ``entry.check_at``. Answers the go
        now (returns None), or returns when to look again."""
        with self._state_lock:
            if (self._quit_requested or entry.gen != self._search_gen
                    or self._claimed_gen == entry.gen):
                return None  # answered, superseded or quitting: drop the entry
        now = time.monotonic()
        if entry.last_moment_at is not None and now >= entry.last_moment_at:
            self._watchdog_fire(entry, f"no bestmove {self._watchdog_last_moment_ms():.0f} ms "
                                       f"before the {entry.budget_ms} ms budget")
            return None
        last_at, chunk_ms, any_chunk = entry.started_at, None, False
        beat = self._heartbeat
        if beat is not None and beat[0] == entry.gen and beat[1] >= entry.started_at:
            last_at, chunk_ms, any_chunk = beat[1], beat[2], True
        stall_s = self._watchdog_stall_ms(chunk_ms, any_chunk=any_chunk) / 1000.0
        if now >= entry.deadline_at and now - last_at >= stall_s:
            self._watchdog_fire(
                entry, f"no search progress for {int((now - last_at) * 1000)} ms "
                       f"(stall limit {int(stall_s * 1000)} ms) past the deadline",
            )
            return None
        next_at = max(entry.deadline_at, last_at + stall_s)
        if entry.last_moment_at is not None:
            next_at = min(next_at, entry.last_moment_at)
        return next_at

    def _watchdog_fire(self, entry: _WatchdogEntry, reason: str = "stalled") -> None:
        """Answer ``entry``'s go. Never touches the worker's evaluator."""
        if not self._claim_answer(entry.gen):
            return  # already answered (the search won, `stop`, a newer go, quit)
        # The stuck search's later output and streak bookkeeping are dropped;
        # the next command that needs the worker parks it (`_wait_for_search`).
        _SILENCED_THREADS.add(entry.thread)
        for stop in entry.stops:
            stop.set()
        snap = self._pv_snapshot
        if (snap is not None and snap[0] == entry.board.fen()
                and _is_playable_fallback(entry.board, snap[1], entry.searchmoves)):
            move, source = snap[1], "last_pv"
        else:
            move, source = self._bestmove_fallback(
                entry.board, entry.searchmoves, allow_fresh_eval=False,
            )
        self._watchdog_fires += 1
        self._bestmove_fallback_used += 1
        self._note_unsearched_reply()
        elapsed_ms = int((time.monotonic() - entry.started_at) * 1000)
        with _answer_owner():
            _println(
                f"info string WARNING search stalled: {reason}; {elapsed_ms} ms into a "
                f"clocked move; deadline watchdog answers unsearched source={source} "
                f"move={move} bestmove_fallback_used={self._bestmove_fallback_used}"
            )
            _println(format_bestmove(move, ponder=None))

    def _counts_for_streak(self) -> bool:
        """False on a parked search, or one the deadline watchdog answered: its
        result was discarded, so it says nothing about the moves actually played."""
        current = threading.current_thread()
        if current in _SILENCED_THREADS:
            return False
        with self._state_lock:
            return current not in self._orphan_threads

    def _note_searched_phase(self) -> None:
        if not self._counts_for_streak():
            return
        self._consecutive_search_failures = 0
        self._ponder_fallback_pending = False

    def _note_fallback_phase(self, *, is_ponder: bool = False) -> None:
        """Count a fallback move.

        Every published fallback extends the streak (an unsearched move is an
        unsearched move, whatever produced it). A ponder phase's fallback is held
        as pending until `_run_search` knows whether it is published (stop) or
        superseded (ponderhit). A parked search's result is not counted.
        """
        if not self._counts_for_streak():
            return
        if is_ponder:
            self._ponder_fallback_pending = True
            return
        self._consecutive_search_failures += 1
        if self._consecutive_search_failures >= _UNHEALTHY_FALLBACK_STREAK:
            self._mark_unhealthy(
                f"{self._consecutive_search_failures} consecutive moves were fallbacks"
            )

    def _run_one_phase(
        self, limits: SearchLimits, *, is_ponder: bool, board: chess.Board,
        started_at: float | None = None,
        clock_started_at: float | None = None,
        gen: int | None = None,
    ) -> SearchResult:
        """``gen``: the go this phase belongs to (`_run_search` passes its own,
        as it does its events); it stamps the watchdog heartbeat. Direct callers
        with no go of their own (warmup, tests) get the current generation."""
        deadline_ms = None if is_ponder else limits.deadline_ms
        max_nodes = None if is_ponder else limits.max_nodes
        max_depth = None if is_ponder else limits.max_depth
        optimum_ms = None if is_ponder else limits.optimum_ms
        deadline = Deadline(
            deadline_ms=deadline_ms,
            now=started_at if started_at is not None else clock_started_at,
        )
        if started_at is not None and (self._stop_event.is_set() or deadline.expired()):
            reason = ("ponder_transition_cancelled" if self._stop_event.is_set()
                      else "ponder_transition_deadline")
            move, source = self._bestmove_fallback(
                board, limits.searchmoves, budget_s=_fresh_eval_budget_s(deadline),
            )
            _println(f"info string search incomplete reason={reason} move={move} source={source}")
            self._note_fallback_phase()
            return SearchResult(
                bestmove_uci=move, ponder_uci=None, nodes=0, pv=(), score_cp=0,
                search_incomplete=reason,
            )
        last_emitted_nodes = [-1]
        profile_start = time.perf_counter()
        if self._gil_probe is not None:
            self._gil_probe.reset()

        board_fen = board.fen()
        beat_gen = self._search_gen if gen is None else gen
        # [last chunk end, chunks completed, slowest inter-chunk ms, phase start
        # to first chunk ms]. The first chunk's time includes root eval / tree
        # setup / compaction, so it is a heartbeat and its own sample, not a
        # chunk-time sample. Both samples are committed to the next search's
        # scale only when this phase finishes un-silenced (below).
        beat: list[float] = [time.monotonic(), 0.0, 0.0, 0.0]

        def _phase_on_chunk(_total_nodes: int) -> None:
            now = time.monotonic()
            if beat[1] >= 1.0:
                beat[2] = max(beat[2], (now - beat[0]) * 1000.0)
            else:
                beat[3] = (now - beat[0]) * 1000.0
            beat[0] = now
            beat[1] += 1.0
            # One tuple, one assignment: the watchdog reads it lock-free.
            self._heartbeat = (beat_gen, now, beat[2] if beat[1] >= 2.0 else None)

        def _phase_info_cb(
            *,
            nodes: int,
            elapsed_ms: int,
            score_cp: int,
            pv: tuple[str, ...],
            tbhits: int,
            score_mate: int | None,
            multipv: int | None,
            wdl: tuple[int, int, int] | None,
            string: str | None = None,
            hashfull: int | None = None,
            seldepth: int | None = None,
        ) -> None:
            last_emitted_nodes[0] = int(nodes)
            if (pv and multipv in (None, 1)
                    and threading.current_thread() not in _SILENCED_THREADS):
                # Keyed by the board this phase searches: a ponder phase's PV
                # starts with the opponent's move and must not answer after
                # ponderhit. One tuple, swapped by a single assignment.
                self._pv_snapshot = (board_fen, pv[0])
            self._emit_info(
                nodes=nodes,
                elapsed_ms=elapsed_ms,
                score_cp=score_cp,
                pv=pv,
                tbhits=tbhits,
                score_mate=score_mate,
                multipv=multipv,
                wdl=wdl,
                string=string,
                hashfull=hashfull,
                seldepth=seldepth,
            )

        # Pools poll this event between batches, whereas their caller only
        # checks Deadline between whole chunks. A slow model can take longer
        # than the entire clock to finish one chunk. Wake cooperative workers
        # at the actual deadline, without relying on a bootstrap nodes/s guess.
        # Capture this phase's event: a later go/ponder phase owns another one.
        phase_stop = self._stop_event
        deadline_timer: threading.Timer | None = None
        remaining_ms = deadline.remaining_ms()
        if remaining_ms is not None:
            deadline_timer = threading.Timer(remaining_ms / 1000.0, phase_stop.set)
            deadline_timer.name = "uci-phase-deadline"
            deadline_timer.daemon = True
            deadline_timer.start()
        try:
            result = self._worker.run(
                board,
                stop_event=phase_stop,
                deadline=deadline,
                max_nodes=max_nodes,
                max_depth=max_depth,
                optimum_ms=optimum_ms,
                abort_factor=self._options.abort_factor,
                root_moves=limits.searchmoves,
                info_cb=_phase_info_cb,
                include_ponder=self._options.ponder,
                allow_terminal_shortcuts=not is_ponder and not limits.is_open_ended(),
                stop_sensitive=is_ponder or limits.infinite,
                on_chunk=_phase_on_chunk,
                # `movetime`: stop one measured chunk before the deadline, as
                # clock games already do, instead of overrunning it by a chunk.
                fixed_time_margin=not is_ponder and limits.fixed_time,
                # `go mate N`: stop once a mate in <= N is proven at the root.
                mate_moves=None if is_ponder else limits.mate_moves,
            )
            if threading.current_thread() not in _SILENCED_THREADS:
                # The next search's stall scale -- never from a search the
                # watchdog answered: its stall would be learned as a chunk.
                if beat[1] >= 1.0:
                    self._prev_first_chunk_ms = _decayed_max(self._prev_first_chunk_ms, beat[3])
                if beat[1] >= 2.0:
                    self._prev_chunk_ms = _decayed_max(self._prev_chunk_ms, beat[2])
            if not is_ponder and last_emitted_nodes[0] != result.nodes:
                self._emit_info(
                    nodes=result.nodes,
                    elapsed_ms=deadline.elapsed_ms(),
                    score_cp=result.score_cp,
                    pv=result.pv,
                    tbhits=result.tbhits,
                    score_mate=result.score_mate,
                    multipv=None,
                    wdl=None,
                )
            self._emit_gil_profile(
                nodes=result.nodes,
                elapsed_s=time.perf_counter() - profile_start,
                is_ponder=is_ponder,
            )
            self._note_searched_phase()
            return result
        except SolvedRootWitnessUnavailable as exc:
            candidate = exc.fallback_actions.get(0)
            move = index_to_move(candidate, board).uci() if candidate is not None else None
            if move is not None and _is_playable_fallback(board, move, limits.searchmoves):
                source = "unknown_root_child"
            else:
                move, source = self._bestmove_fallback(
                    board, limits.searchmoves, budget_s=_fresh_eval_budget_s(deadline),
                )
            self._bestmove_fallback_used += 1
            self._witness_unavailable_roots += 1
            reason = "solved_root_without_move_witness"
            _println(
                f"info string search incomplete reason={reason} root_status={exc.outcomes.get(0)} "
                f"child_checks={exc.child_checks.get(0, 0)} move={move} "
                f"outcome_preserved=UNVERIFIED source={source} "
                f"bestmove_fallback_used={self._bestmove_fallback_used} "
                f"witness_unavailable_roots={self._witness_unavailable_roots}"
            )
  # A property of the POSITION (the root is proved solved, the proof names no
  # move), not of the engine: an evaluator or search fault raises a different
  # exception. It neither extends nor resets the unhealthy streak.
            return SearchResult(
                bestmove_uci=move, ponder_uci=None, nodes=0, pv=(), score_cp=0,
                search_incomplete=reason,
            )
        except Exception as exc:  # pragma: no cover — UCI crash-safety
            self._emit_gil_profile(
                nodes=0,
                elapsed_s=time.perf_counter() - profile_start,
                is_ponder=is_ponder,
            )
            _println(f"info string search error: {exc!r}")
            _log.error("search failed; playing the fallback move", exc_info=True)
            fallback, source = self._bestmove_fallback(
                board, limits.searchmoves, budget_s=_fresh_eval_budget_s(deadline),
            )
            self._bestmove_fallback_used += 1
            self._note_fallback_phase(is_ponder=is_ponder)
            _println(
                f"info string bestmove_fallback_used={self._bestmove_fallback_used} "
                f"source={source} move={fallback} exception={type(exc).__name__} "
                f"phase={'ponder' if is_ponder else 'main'}"
            )
            if is_ponder:
                _println(
                    "info string WARNING ponder search failed; the reply is UNSEARCHED "
                    "(fallback) unless the search after ponderhit succeeds"
                )
            else:
                _println(
                    "info string WARNING search failed on "
                    f"{self._consecutive_search_failures} consecutive move(s); this move "
                    "is UNSEARCHED (fallback)"
                )
            return SearchResult(
                bestmove_uci=fallback, ponder_uci=None, nodes=0, pv=(), score_cp=0, tbhits=0,
            )
        finally:
            # Join even after cancellation: an already-running callback must
            # finish before another search can reuse an event or the tree.
            if deadline_timer is not None:
                deadline_timer.cancel()
                deadline_timer.join()

    def _report_declined_root(self, result: SearchResult) -> None:
        """Announce and count a published bestmove the search REFUSED to produce.

        Distinct from `bestmove_fallback_used` on purpose. A raise is a *fault*
        — evaluator, walker, CUDA, OOM — and `bestmove_fallback_used` is a
        health signal whose only acceptable value is 0. A declined root is not a
        fault: `CBoard.is_game_over()` legitimately refuses claimable draws, and
        threefold is reached through the move stack, so in real match play this
        fires every time we shuffle into a repetition. Folding the two together
        would make the fault counter tick during ordinary play and destroy the
        one thing it is for. Two counters; both emit an `info string`, and both
        return `nodes=0` so the behavioural proxy sees either.
        """
        reason = getattr(result, "root_declined", None)
        if not isinstance(reason, str):
            return
        self._prior_only_roots += 1
        _println(
            f"info string prior_only_root={self._prior_only_roots} reason={reason} "
            f"move={result.bestmove_uci} nodes=0 "
            "(the C search declined this root; move is the raw policy prior, "
            "NOT searched)"
        )

    @property
    def prior_only_roots(self) -> int:
        """How many `go`s this process answered from an unsearched root prior.

        Session total, like `bestmove_fallback_used`. Unlike it, a non-zero
        value is not automatically a defect — it is the count of plies played
        without search, which is what an operator needs to know when the engine
        is shuffling in a repetition.
        """
        return self._prior_only_roots

    def _bestmove_fallback(
        self, board: chess.Board, searchmoves: tuple[str, ...],
        *, allow_fresh_eval: bool = True, budget_s: float | None = None,
    ) -> tuple[str, str]:
        """``(uci, source)`` for a bestmove the search could not produce.

        ``SearchWorker`` caches the root NN eval before the first chunk runs, so
        on essentially every in-search failure the net's own prior over root
        moves is already in memory and free to read. Prefer its argmax;
        ``next(iter(board.legal_moves))`` is python-chess square order —
        effectively a random move (the measured case hangs a knight), and it is
        the terminal handler for a claimable-draw root, a walker raise, a
        transient CUDA fault and an OOM alike.

        The worker's answer is re-validated here rather than trusted: a fallback
        that is illegal in the real position is a forfeit, which is strictly
        worse than the random move it replaces.
        """
        try:
            candidate = self._worker.root_policy_bestmove(board, searchmoves=searchmoves)
        except Exception as exc:
            _println(f"info string root-policy fallback unavailable: {exc!r}")
            candidate = None
        if isinstance(candidate, str) and _is_playable_fallback(board, candidate, searchmoves):
            return candidate, "root_policy"
  # The cache is empty or belongs to another position (a `go` that failed before
  # its own root eval). One forward pass on THIS board, no search, beats a
  # square-order move. Skipped when another thread may still own the evaluator.
        if allow_fresh_eval:
            fresh = self._bounded_fresh_eval(board, searchmoves, budget_s)
            if isinstance(fresh, str) and _is_playable_fallback(board, fresh, searchmoves):
                return fresh, "fresh_root_eval"
        return _legal_fallback_move(board, searchmoves), "first_legal"

    @property
    def bestmove_fallback_used(self) -> int:
        """How many search-phase exceptions selected a bestmove fallback.

        Monotonic for the life of the engine (a GUI plays many games in one
        process), so it is a session total, not a per-game one. Zero is the only
        healthy value. Each diagnostic identifies the phase: a ponder fallback
        may be replaced by a successful main phase after ponderhit, but that
        earlier search fault still counts.
        """
        return self._bestmove_fallback_used

    def _emit_gil_profile(self, *, nodes: int, elapsed_s: float, is_ponder: bool) -> None:
        probe = self._gil_probe
        if probe is None:
            return
        stats = probe.read_and_reset()
        mode, threads = self._worker.concurrency_profile()
        devices = max(1, len(self._search_devices or ()))
        _println(
            "info string gil_profile "
            f"phase={'ponder' if is_ponder else 'main'} mode={mode} "
            f"threads={threads} devices={devices} nodes={int(nodes)} "
            f"elapsed_ms={float(elapsed_s) * 1000.0:.1f} "
            f"samples={stats.samples} rate={stats.sample_rate:.1f}/s "
            f"delay_mean={stats.mean_ms:.3f}ms "
            f"p50<={stats.p50_ms:.2f}ms p95<={stats.p95_ms:.2f}ms "
            f"p99<={stats.p99_ms:.2f}ms max={stats.max_ms:.2f}ms "
            f"over1ms={stats.over_1ms_pct:.1f}% over5ms={stats.over_5ms_pct:.1f}%"
        )

    def _emit_info(
        self, *,
        nodes: int, elapsed_ms: int, score_cp: int, pv: tuple[str, ...],
        tbhits: int, score_mate: int | None, multipv: int | None,
        wdl: tuple[int, int, int] | None, string: str | None = None,
        hashfull: int | None = None, seldepth: int | None = None,
    ) -> None:
        nps = int(nodes * 1000 / max(1, elapsed_ms))
        _println(format_info(InfoFields(
            depth=len(pv),
            seldepth=seldepth,
            multipv=multipv,
            nodes=nodes,
            nps=nps,
            time_ms=elapsed_ms,
            score_cp=score_cp,
            score_mate=score_mate,
            pv=pv,
            hashfull_per_mille=hashfull,
            tbhits=tbhits,
            wdl=wdl,
            string=string,
        )))

    def _ponder_miss_result(
        self, result: SearchResult, ponder_board: chess.Board, popped: chess.Move,
    ) -> SearchResult:
        """``result`` re-targeted to the GUI's board after a `stop` in ponder.

        Our reply from the ponder PV when it starts with the predicted move;
        else the usual fallback (cached root prior / first legal), never a fresh
        eval -- the GUI discards this move, so no time is spent on it. The
        ponder root's `root_declined` is not carried over: the published move is
        not that root's raw prior, so it must not count as a prior-only root."""
        real_board = ponder_board.copy(stack=True)
        real_board.push(popped)
        pv = tuple(result.pv or ())
        if len(pv) >= 2 and pv[0] == popped.uci() and _is_playable_fallback(real_board, pv[1], ()):
            move, source, new_pv = pv[1], "ponder_pv", pv[1:]
        else:
            move, source = self._bestmove_fallback(real_board, (), allow_fresh_eval=False)
            new_pv = ()
        _println(
            f"info string ponder miss: stop before ponderhit; the ponder search was for "
            f"the position before {popped.uci()}, so bestmove is for the GUI's position "
            f"(source={source} move={move})"
        )
        return replace(
            result, bestmove_uci=move, ponder_uci=None, pv=new_pv, root_declined=None,
        )

    def _emit_bestmove(self, result: SearchResult) -> None:
        self._report_declined_root(result)
        ponder = result.ponder_uci if self._options.ponder else None
        _println(format_bestmove(result.bestmove_uci, ponder=ponder))

    def close(self) -> None:
        """Stop any running search and release the worker's evaluator. Call
        from the UCI main loop's finally block so ``BatchCoalescingDispatcher``
        (if active) drains its submitter before the interpreter
        tears down torch's CUDA context. A surviving search raises
        ``SearchNotQuiescent`` and retains worker ownership for a later join."""
        if self._closed:
            return
        # Record terminal intent before a bounded join can fail or emit.
        with self._state_lock:
            self._quit_requested = True
            self._search_gen += 1
        self._watchdog.close()
        self._wait_for_search(timeout_s=min(_SHUTDOWN_JOIN_S, _JOIN_TIMEOUT_S))
        if self._gil_probe is not None:
            self._gil_probe.close()
        self._worker.close()
        self._closed = True

  # -- helpers --------------------------------------------------------------

    def worker_is_free(self) -> bool:
        """No search thread, no live orphan, no warmup: the worker is ours.

        The single predicate every path that touches the worker, tree or
        evaluator outside a search uses (re-warm, the boundary
        health rebuild and re-check; setoption rebuilds go through `_wait_for_search`,
        which raises on the same condition).
        """
        with self._state_lock:
            return self._search_free_locked() and self._warmup_done.is_set()

    def _search_free_locked(self) -> bool:
        self._orphan_threads = [t for t in self._orphan_threads if t.is_alive()]
        thread = self._search_thread
        return (thread is None or not thread.is_alive()) and not self._orphan_threads

    def has_live_threads(self) -> bool:
        """A search, orphan or fallback helper is still running (e.g. in torch).

        `__main__` hard-exits after `close()` when this is True rather than
        finalizing the interpreter under a live thread.
        """
        with self._state_lock:
            return not self._search_free_locked()

    def _wait_for_search(
        self, *, timeout_s: float | None = None, answer_parked: bool = False,
    ) -> None:
        """Stop the current search and make sure nothing else owns the worker.

        Waits at most ``timeout_s`` (default `_JOIN_TIMEOUT_S`) IN TOTAL. A
        search that does not stop in time is parked as an orphan, ONCE, and its
        late result is silenced by a generation bump. Later commands only POLL
        orphans (no join on every command); `go` passes its clock-bounded budget.
        ``answer_parked`` (only `stop`, whose GUI is waiting for exactly that
        bestmove): answer the parked search's `go` from the fallback now. Any
        other command must NOT: a late bestmove after the GUI has moved on is
        read as the reply to its NEXT `go`, for the wrong board.
        Raises `SearchNotQuiescent` while anything still owns the worker.
        """
        budget = _JOIN_TIMEOUT_S if timeout_s is None else max(0.0, timeout_s)
        end = time.monotonic() + budget
        thread = self._search_thread
        if thread is not None and thread.is_alive():
            self._cancel_event.set()
            self._stop_event.set()
            self._ponder_release.set()
            thread.join(timeout=max(0.0, end - time.monotonic()))
            if thread.is_alive():
                self._park_search_thread(thread, answer=answer_parked)
                raise SearchNotQuiescent(
                    f"search did not stop within {budget:.1f}s; parked; worker unchanged"
                )
        if not self._warmup_done.wait(timeout=max(0.0, end - time.monotonic())):
            raise SearchNotQuiescent("warmup stop timed out; worker unchanged")
        for orphan in list(self._orphan_threads):
            wait = 0.0 if timeout_s is None else max(0.0, end - time.monotonic())
            orphan.join(timeout=wait)
            if orphan.is_alive():
                raise SearchNotQuiescent(
                    "an earlier search that missed its stop deadline still owns the worker"
                )
            with self._state_lock:
                if orphan in self._orphan_threads:
                    self._orphan_threads.remove(orphan)
        self._search_thread = None
        self._live_go = None

    def _park_search_thread(self, thread: threading.Thread, *, answer: bool) -> None:
        """Retire a search that missed its stop: orphan it, silence it, answer it.

        The reply-only thread is answered the same way: its go was already
        counted when the thread was started, so it is not counted again."""
        with self._state_lock:
            # Answer only a go nobody has claimed yet (the thread may have
            # claimed and be about to print; then it prints, as the owner; or the
            # deadline watchdog may already have answered).
            claim = answer and not self._quit_requested and self._claimed_gen != self._search_gen
            if claim:
                self._claimed_gen = self._search_gen
            self._search_gen += 1
        if thread not in self._orphan_threads:
            self._orphan_threads.append(thread)
        _SILENCED_THREADS.add(thread)  # its later info lines must not reach stdout
        self._search_thread = None
        live, self._live_go = self._live_go, None
        if not claim or live is None:
            return
        board, searchmoves = live
        move, source = self._bestmove_fallback(board, searchmoves, allow_fresh_eval=False)
        if thread is not self._fallback_reply_thread:
            self._bestmove_fallback_used += 1
            self._note_unsearched_reply()
        _println(
            "info string search did not stop in time; answering unsearched "
            f"bestmove_fallback_used={self._bestmove_fallback_used} source={source} move={move}"
        )
        _println(format_bestmove(move, ponder=None))

    def _bounded_fresh_eval(
        self, board: chess.Board, searchmoves: tuple[str, ...], budget_s: float | None,
    ) -> str | None:
        """One fresh root eval of ``board`` on a helper thread, time-bounded.

        A helper still running after the budget is still inside the evaluator, so
        it is parked with the orphans: nothing reuses the worker until it ends.
        """
        timeout = _FRESH_EVAL_TIMEOUT_S if budget_s is None else min(_FRESH_EVAL_TIMEOUT_S, budget_s)
        box: dict[str, object] = {}

        def run() -> None:
            try:
                box["move"] = self._worker.fresh_root_policy_bestmove(
                    board, searchmoves=searchmoves,
                )
            except Exception as exc:
                box["error"] = exc

        helper = threading.Thread(target=run, daemon=True, name="uci-fresh-eval")
        helper.start()
        helper.join(timeout=max(0.0, timeout))
        if helper.is_alive():
            with self._state_lock:
                self._orphan_threads.append(helper)
            _SILENCED_THREADS.add(helper)
            _println(
                f"info string fresh root-eval fallback timed out after {timeout * 1000:.0f} ms; "
                "using the first legal move"
            )
            return None
        error = box.get("error")
        if isinstance(error, BaseException):
            _println(f"info string fresh root-eval fallback unavailable: {_one_line(error)}")
            return None
        move = box.get("move")
        return move if isinstance(move, str) else None


def _fresh_eval_budget_s(deadline: Deadline) -> float | None:
    """A tenth of what is left of this phase's clock (None: no clock)."""
    remaining = deadline.remaining_ms()
    return None if remaining is None else max(0.0, remaining / 1000.0) * 0.1


def _close_quietly(evaluator: object) -> None:
    """Retire an evaluator that was built but never installed."""
    close = getattr(evaluator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            _log.warning("closing an uninstalled evaluator failed", exc_info=True)


def _is_playable_fallback(
    board: chess.Board, uci: str, searchmoves: tuple[str, ...],
) -> bool:
    """True when ``uci`` is legal here and honours a ``searchmoves`` filter."""
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return False
    if move not in board.legal_moves:
        return False
    return not searchmoves or uci in searchmoves


def _legal_fallback_move(board: chess.Board, searchmoves: tuple[str, ...]) -> str:
    """Return a legal fallback that respects UCI ``searchmoves`` constraints."""
    if searchmoves:
        for uci in searchmoves:
            try:
                move = chess.Move.from_uci(str(uci))
            except ValueError:
                continue
            if move in board.legal_moves:
                return move.uci()
        return "0000"
    try:
        return next(iter(board.legal_moves)).uci()
    except StopIteration:
        return "0000"


# UCI stdout is written from several threads: the main loop (handshake,
# readyok, bestmove), the background model-load thread in __main__ (startup
# `info string`s), and the search thread (`info` lines). A bare `print()`
# emits text and newline as separate writes, so without serialization a
# concurrent line can be spliced mid-line into another — observed as a startup
# `info string` swallowing the `id name` handshake line and breaking the UCI
# handshake. Hold a lock across the write+flush so every logical line is atomic
# on stdout; all UCI output (including __main__'s startup prints) routes here.
_PRINT_LOCK = threading.Lock()


def _decayed_max(previous: float | None, sample: float) -> float:
    """A running maximum that forgets slowly (x0.9 per search): one fast
    search must not shrink the stall scale below what recent searches needed."""
    return sample if previous is None else max(sample, _WATCHDOG_SCALE_DECAY * previous)


@dataclass(frozen=True)
class _WatchdogEntry:
    gen: int
    check_at: float  # when the watchdog next looks (see `Engine._watchdog_poll`)
    deadline_at: float  # the search's own hard deadline
    last_moment_at: float | None  # budget minus the answer margin; None: no clock
    budget_ms: int | None  # the move's real budget (movetime or remaining clock)
    board: chess.Board
    searchmoves: tuple[str, ...]
    thread: threading.Thread
    stops: tuple[threading.Event, ...]
    started_at: float


class _DeadlineWatchdog:
    """One long-lived daemon thread per engine, started at the first arm.

    Holds at most ONE armed entry (the current go's clocked phase; arming
    replaces it) and calls ``poll(entry)`` at ``entry.check_at``; ``poll``
    answers the go (returns None) or returns the next time to look. A timed
    wait on a Condition, not a thread per go: arming and disarming are
    notifications.
    ``close()`` stops and joins it; nothing else holds a reference.
    """

    def __init__(self, poll: Callable[[_WatchdogEntry], float | None]) -> None:
        self._poll = poll
        self._cond = threading.Condition()
        self._entry: _WatchdogEntry | None = None
        self._closed = False
        self._thread: threading.Thread | None = None

    @property
    def thread(self) -> threading.Thread | None:
        return self._thread

    @property
    def armed(self) -> _WatchdogEntry | None:
        return self._entry

    def arm(self, entry: _WatchdogEntry) -> None:
        with self._cond:
            if self._closed:
                return
            self._entry = entry
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._loop, name="uci-deadline-watchdog", daemon=True,
                )
                self._thread.start()
            self._cond.notify()

    def disarm(self, gen: int) -> None:
        with self._cond:
            if self._entry is not None and self._entry.gen == gen:
                self._entry = None
                self._cond.notify()

    def close(self, timeout_s: float = 1.0) -> None:
        with self._cond:
            self._closed = True
            self._entry = None
            self._cond.notify()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._closed:
                    entry = self._entry
                    if entry is None:
                        self._cond.wait()
                        continue
                    remaining = entry.check_at - time.monotonic()
                    if remaining > 0:
                        self._cond.wait(remaining)
                        continue
                    self._entry = None
                    break
                else:
                    return
            try:
                next_at = self._poll(entry)
            except Exception:
                _log.error("deadline watchdog failed", exc_info=True)
                next_at = None
            if next_at is not None:
                with self._cond:
                    if self._entry is None and not self._closed:  # not re-armed meanwhile
                        self._entry = replace(entry, check_at=next_at)


# Threads parked as orphans (a search that missed its stop, a hung fresh-eval
# helper). Once parked, their `go` has been answered (or will be by a fallback),
# so anything they print -- info lines, warnings -- would land in the middle of
# another move. `_println` drops it; logging to stderr is unaffected.
_SILENCED_THREADS: weakref.WeakSet[threading.Thread] = weakref.WeakSet()


# Set on a thread while it prints a bestmove it has claimed (`_claim_answer`).
_ANSWER_OWNER = threading.local()


class _answer_owner:  # lowercase: used like a function, `with _answer_owner():`
    def __enter__(self) -> None:
        _ANSWER_OWNER.active = True

    def __exit__(self, *_exc: object) -> None:
        _ANSWER_OWNER.active = False


def _println(s: str) -> None:
    if (threading.current_thread() in _SILENCED_THREADS
            and not getattr(_ANSWER_OWNER, "active", False)):
        return
    with _PRINT_LOCK:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()


def _emit_info_string(s: str) -> None:
    """`info string` emitter handed to search pools (root-parallel Gumbel
    per-phase lines). Runs on the search thread at phase barriers, protected
    by the same stdout lock as every other UCI line."""
    _println(f"info string {s}")
