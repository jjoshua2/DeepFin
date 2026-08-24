"""Engine — holds board state, a SearchWorker, and the command dispatch.

Command handlers are called from the main UCI loop in response to parsed
commands. ``go`` kicks off a background thread that runs the search, so
the main loop can stay responsive for ``stop`` / ``ponderhit``.

v1 scope: no pondering yet. Implements the ``go`` / ``stop`` / ``quit``
path end-to-end so we can play a game.
"""
from __future__ import annotations

import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Protocol
from collections.abc import Callable

import chess

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
    InfoFields,
    format_bestmove,
    format_id_lines,
    format_info,
    format_readyok,
    format_uciok,
)
from .search import SearchResult, SearchWorker
from .time_manager import (
    _DEFAULT_MOVES_REMAINING,
    _OPTIMUM_FRACTION,
    Deadline,
    SearchLimits,
    limits_from_go,
)

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
        f"option name Hash type spin default {options.hash_mb} "
        f"min {_MIN_HASH_MB} max 524288"
    )
    _println(f"option name Threads type spin default {options.threads} min 1 max 64")
    _println(f"option name LeafGather type spin default {options.leaf_gather} min 1 max 64")
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
        "min -1 max 4096"
    )
    _println(
        f"option name RPGMinVPA type spin default {options.rpg_min_vpa} "
        "min -1 max 4096"
    )
    _println(
        f"option name RPGMinKeep type spin default {options.rpg_min_keep} "
        "min -1 max 64"
    )
    _println(
        "option name RPGOpenBudgetFrac type string default "
        f"{options.rpg_open_budget_frac}"
    )
    _println(f"option name VLGather type spin default {options.vl_gather} min 32 max 4096")
    _println(f"option name MaxBatch type spin default {options.max_batch} min 64 max 8192")
    _println(f"option name EvalCacheEntries type spin default {options.eval_cache_entries} min 0 max 1048576")
    _println(f"option name MultiPV type spin default {options.multi_pv} min 1 max 256")
    _println(f"option name UCI_ShowWDL type check default {'true' if options.show_wdl else 'false'}")
    _println(f"option name MoveOverheadMs type spin default {options.move_overhead_ms} min 0 max 5000")
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

# Advertised (and now enforced) floor for the `Hash` UCI option, in MB. Keep this
# in step with the `min` field `emit_handshake` prints — the two disagreeing is
# exactly the defect this constant closes.
_MIN_HASH_MB = 1024

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
  # Number of MCTS walker threads. 1 = classic Gumbel path; >1 = PUCT
  # walker pool with virtual loss. Set via UCI `Threads`.
    threads: int = 2
  # Per-walker leaf gather: each walker does G descents → one NN batch
  # instead of one descent per submit. Amplifies effective batch to
  # N_walkers×G without spawning more threads. 1 = classic. Lc0's
  # canonical value is 8. Set via UCI `LeafGather`.
    leaf_gather: int = 1
  # Single-thread async-pipeline batched-VL PUCT. Active only at
  # `Threads=1` with a 2-slot inplace-async evaluator. Bench: +112%
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
  # Set by setoption handlers that reshape the search path / evaluator
  # (Threads/UseVL/UseMultiGpuPUCV/LeafGather/VLGather/MaxBatch/
  # EvalCacheEntries). The build-time ``warmup_search`` runs BEFORE any
  # setoption is applied, so such a change leaves its cudagraph stale;
  # ``_handle_isready`` re-warms the configured path before ``readyok``
  # so the first real ``go`` never pays cold capture on the clock.
        self._warmup_dirty = False
  # Times this process answered a `go` from the bestmove fallback instead of a
  # searched tree. Surfaced as an `info string` on every increment so a match
  # log carries the evidence; see the `bestmove_fallback_used` property.
        self._bestmove_fallback_used = 0
  # Times the C search DECLINED the root and the move came from the raw prior
  # with no simulation. Deliberately separate from the fault counter above —
  # see `_report_declined_root`.
        self._prior_only_roots = 0
        self._board = chess.Board()
        self._search_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ponderhit_event = threading.Event()
  # While the current search is a `go ponder`, hold the "real" limits
  # here so ponderhit can swap them in.
        self._pending_real_limits: SearchLimits | None = None
        self._quit_requested = False
  # Monotonic counter bumped on every new `go`. Stored on the search
  # thread's frame; a stale thread that outlives its join still sees
  # its own captured value, so emit-bestmove and tree-writeback both
  # gate on `gen == self._search_gen`. Prevents races after a slow stop.
        self._search_gen = 0
        self._state_lock = threading.Lock()
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

  # -- main-thread command dispatch -----------------------------------------

    def dispatch(self, cmd: Command) -> None:
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
        searching = self._search_thread is not None and self._search_thread.is_alive()
        if self._warmup_dirty and not searching:
            self._warmup_dirty = False
            self.warmup_search()
        _println(format_readyok())

    def _handle_newgame(self) -> None:
        self._wait_for_search()
        self._worker.reset_tree()
        self._board = chess.Board()
        self._pending_fen = None
        self._pending_moves = []
        self._applied_fen = None
        self._applied_moves = ()
        self._popped_ponder_move = None

    def _handle_position(self, cmd: CmdPosition) -> None:
        self._wait_for_search()
        if cmd.fen is None:
            start = chess.Board()
        else:
            try:
                start = chess.Board(cmd.fen)
            except ValueError as exc:
  # Falling back to the start position is a silent wrong-position answer: the
  # engine then returns a confidently-scored move for a board the caller never
  # asked about. Say so — EPD/puzzle/blind-spot drivers feed arbitrary FENs.
                _println(
                    f"info string position: rejected FEN {cmd.fen!r} ({exc}); "
                    "using the start position"
                )
                self._board = chess.Board()
                self._pending_fen = None
                self._pending_moves = []
                self._applied_fen = None
                self._applied_moves = ()
                self._popped_ponder_move = None
                return
        new_board = start.copy(stack=False)
        parsed: list[chess.Move] = []
        for uci in cmd.moves:
  # A bad move truncates the game and the engine then searches a position n
  # plies behind the real one, emitting a move that is very likely illegal
  # there — a forfeit, delivered silently. Diagnose every truncation.
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                self._warn_moves_truncated(cmd.moves, parsed, uci, "not valid UCI")
                break
            if mv not in new_board.legal_moves:
                self._warn_moves_truncated(
                    cmd.moves, parsed, uci, f"illegal in {new_board.fen()}",
                )
                break
            new_board.push(mv)
            parsed.append(mv)
        self._board = new_board
  # Record intent; the tree is advanced in _handle_go so ponder mode
  # can choose to stop one ply short.
        self._pending_fen = cmd.fen
        self._pending_moves = parsed
  # Any popped-ponder state from a prior `go ponder` is invalid now.
        self._popped_ponder_move = None

    @staticmethod
    def _warn_moves_truncated(
        sent: tuple[str, ...], parsed: list[chess.Move], offender: str, why: str,
    ) -> None:
        """Report that ``position ... moves`` stopped early at ``offender``."""
        _println(
            f"info string position: ignored {len(sent) - len(parsed)} of "
            f"{len(sent)} move(s) from {offender!r} onward ({why}); "
            "searching the truncated position"
        )

    def _sync_tree_root(self, target_moves: list[chess.Move]) -> None:
        """Descend (or reset) the worker's tree so its root matches
        ``self._pending_fen`` plus ``target_moves``. Updates ``_applied_*`` to
        reflect the post-sync state."""
        reused = (
            self._applied_fen == self._pending_fen
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
        self._applied_fen = self._pending_fen
        self._applied_moves = tuple(target_moves)

    def _handle_go(self, cmd: CmdGo) -> None:
        if self._search_thread is not None and self._search_thread.is_alive():
  # UCI says you should `stop` before issuing another `go`. Be
  # forgiving: stop the current one first.
            self._handle_stop()
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
        self._ponderhit_event = threading.Event()
  # For `go ponder`, the ponder phase runs open-ended; ponderhit
  # converts to a real-deadline phase using the SAME underlying clock
  # args. We re-derive the "real" limits here by synthesizing a
  # non-ponder copy of the args — but the side-to-move must be
  # derived from the REAL-phase board (after the opponent's predicted
  # move was pushed), not search_board (which is their side). Using
  # search_board.turn here picks the opponent's wtime/btime and
  # causes the real phase to flag or undersearch at asymmetric
  # clocks. (Codex adversarial review.)
        if cmd.args.ponder and self._popped_ponder_move is not None:
            real_board = search_board.copy(stack=False)
            real_board.push(self._popped_ponder_move)
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
        with self._state_lock:
            self._search_gen += 1
            gen = self._search_gen
        self._search_thread = threading.Thread(
            target=self._run_search,
            args=(limits, gen, search_board),
            daemon=True,
        )
        self._search_thread.start()

    def _handle_stop(self) -> None:
        if self._search_thread is not None:
            self._stop_event.set()
            self._search_thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._search_thread.is_alive():
  # Don't clear the handle: the thread may still be running a
  # C chunk that can't be interrupted. Bumping _search_gen
  # (in _handle_go) will invalidate any late bestmove. We
  # retry the cleanup on the next isready / go.
                _println("info string search stop timed out; thread still running")
            else:
                self._search_thread = None

    def _handle_ponderhit(self) -> None:
        """Opponent played our predicted move. Convert open-ended ponder
        search into a time-bounded real search, keeping the MCTS tree.
        """
        if self._search_thread is None or not self._search_thread.is_alive():
            return
  # Signal the search loop: next iteration, swap ponder-deadline
  # for real-deadline.
        self._ponderhit_event.set()
        self._stop_event.set()

    def _handle_setoption(self, cmd: CmdSetOption) -> None:
  # Same barrier as _handle_position / _handle_newgame / _handle_isready:
  # options must not mutate while a search thread is still reading them.
  # In particular SyzygyPath swaps the shared tablebase cache, which the
  # search thread probes through on every leaf batch.
        self._wait_for_search()
        name = cmd.name.lower()
  # SyzygyPath is special: empty value is meaningful (sentinel for unset),
  # all others bail when value is None.
        if name == "syzygypath":
            self._set_syzygy_path(cmd.value)
            return
        if cmd.value is None:
            return
        handler = self._SETOPTION_HANDLERS.get(name)
        if handler is not None:
            handler(self, cmd.value)
            return
        search_opt = OPTIONS_BY_NAME.get(name)
        if search_opt is not None:
            self._set_search_option(search_opt, cmd.value)
  # All other options silently accepted — we don't expose any yet.

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
        if not math.isfinite(value) or (
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
  # Non-int values silently keep the prior value (mirrors original behavior).
    @staticmethod
    def _parse_clamped_int(value: str, lo: int) -> int | None:
        try:
            return max(lo, int(value))
        except ValueError:
            return None

    def _set_hash(self, value: str) -> None:
        try:
            requested = int(value)
        except ValueError:
            return
  # The handshake advertises `min 1024` but enforcement used to floor at 1, so a
  # GUI's conventional `Hash 256` was accepted and then quietly crippled the
  # engine: `advance_root` refuses cross-move reuse once the tree passes
  # `max_tree_bytes // 2`, and the initial reserve alone is ~8.6 MB — so any
  # Hash below ~17 MB kills reuse from move one and throttles every chunk.
  # Advertisement and enforcement now agree, and a clamp is announced.
        n = max(_MIN_HASH_MB, requested)
        if n != requested:
            _println(
                f"info string Hash {requested} is below the advertised minimum "
                f"{_MIN_HASH_MB}; using {n}"
            )
        self._options.hash_mb = n
        self._worker.set_max_tree_mb(n)

    def _set_ponder(self, value: str) -> None:
        self._options.ponder = value.strip().lower() == "true"

    def _multi_gpu_path_active(self) -> bool:
        """True when a multi-GPU pool (RPG or PUCV) owns search dispatch."""
        return (
            self._options.search_parallel == "gumbel"
            or self._options.use_multi_gpu_pucv
        )

    def _set_threads(self, value: str) -> None:
        n = self._parse_clamped_int(value, lo=1)
        if n is None:
            return
        old_threads = self._options.threads
        self._options.threads = n
        self._worker.set_num_threads(n)
        if self._multi_gpu_path_active():
            # Worker stores n_walkers for leave-multi-GPU restore; live
            # dispatch still uses the RPG / multi-GPU PUCV pool.
            mode = (
                "SearchParallel=gumbel"
                if self._options.search_parallel == "gumbel"
                else "UseMultiGpuPUCV"
            )
            _println(
                f"info string Threads set to {n} "
                f"(deferred — {mode} is active)",
            )
            return
        self._warmup_dirty = True
        # Crossing 1 <-> >1 changes which dispatcher tier the factory builds
        # (compiled: CUDAOwner at 1 walker vs BatchCoalescing at >1). The
        # walker-pool rebuild alone leaves the old wrapper in place — walkers
        # would then serialize through the owner queue (1->N) or lose the
        # pinned in-place path (N->1). Same-tier changes skip the rebuild.
        if (
            (n == 1) != (old_threads == 1)
            and self._rebuild_evaluator is not None
        ):
            _println(
                f"info string Threads {old_threads}->{n} crosses dispatcher "
                "tier; rebuilding evaluator…"
            )
            new_eval = self._rebuild_evaluator(
                self._options.max_batch, self._options.eval_cache_entries,
                n_walkers=n,
            )
            self._worker.set_evaluator(new_eval)
            self._reinstall_configured_search_path()
        _println(
            f"info string Threads set to {n} "
            f"({'walker pool' if n > 1 else 'classic Gumbel path'})"
        )
        # The walker pool selects on raw Q and never reads c_scale/c_visit (Gumbel
        # sigma(q) only), so switching to Threads>1 silently drops the tuned
        # c_scale value. Surface it; Threads=1 restores the c_scale-tuned path.
        if n > 1:
            _println(
                "info string note: c_scale/c_visit are inactive on the walker pool; "
                "Threads=1 uses the c_scale-tuned classic Gumbel path"
            )

    def _set_leaf_gather(self, value: str) -> None:
        n = self._parse_clamped_int(value, lo=1)
        if n is None:
            return
        self._options.leaf_gather = n
        self._worker.set_walker_gather(n)
        self._warmup_dirty = True
        _println(f"info string LeafGather set to {n}")

    def _set_use_vl(self, value: str) -> None:
        enabled = value.strip().lower() == "true"
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
            f"requires Threads=1 + 2-slot async evaluator)"
        )

    def _set_use_multi_gpu_pucv(self, value: str) -> None:
        enabled = value.strip().lower() == "true"
        if not enabled:
            had_pool = getattr(self._worker, "_pucv_pool", None) is not None
            self._options.use_multi_gpu_pucv = False
            self._worker.clear_multi_gpu_pucv()
            # Expand primary back to all devices for the routing/walker path
            # (rebuild_evaluator closes over use_multi_gpu_pucv → primary_only=False).
            if (
                self._rebuild_evaluator is not None
                and self._rebuild_multi_gpu_pucv_factories is not None
            ):
                new_eval = self._rebuild_evaluator(
                    self._options.max_batch, self._options.eval_cache_entries,
                )
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
                # UseVL/Threads flags set but no live path materialised.
                self._reinstall_configured_search_path(after_leaving_gumbel=True)
            return
        # Shrink primary to device 0 before installing the pool so we don't
        # keep a full MultiGPUDispatcher stack alongside per-GPU pool evals.
        if self._rebuild_evaluator is not None:
            new_eval = self._rebuild_evaluator(
                self._options.max_batch, self._options.eval_cache_entries,
            )
            self._worker.set_evaluator(new_eval)
        n_devices = self._install_multi_gpu_pucv_pool()
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
        # Same inertness as the Threads>1 walker pool (see _set_threads): the
        # pool selects with plain PUCT and never reads c_scale/c_visit, so the
        # tuned Gumbel values silently stop applying while it is installed.
        _println(
            "info string note: c_scale/c_visit are inactive on the multi-GPU "
            "PUCV pool (plain PUCT); a single device with Threads=1 restores "
            "the c_scale-tuned classic Gumbel path"
        )

    def _install_multi_gpu_pucv_pool(self) -> int | None:
        """Install/reinstall the multi-GPU pool from current options.

        Returns device count on success, or None (and leaves
        ``use_multi_gpu_pucv`` False) when factories are missing or fewer
        than two devices. Does not rebuild the primary evaluator — callers
        that need a single-device primary do that first.
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
        self._worker.install_multi_gpu_pucv(
            factories, gather=gather, as_factories=True,
        )
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
        UseMultiGpuPUCV / Threads / UseVL routing. Single-device configs never
        construct the new module — dispatch falls through to the classic
        paths untouched."""
        normalized = value.strip().lower()
        if normalized not in ("pucv", "gumbel"):
            return
        if normalized == "gumbel":
            if not self._try_install_root_parallel_gumbel():
                return
            return
        self._options.search_parallel = "pucv"
        self._worker.clear_root_parallel_gumbel()
        self._warmup_dirty = True
        _println("info string SearchParallel pucv")
        # Restore whichever non-RPG path is configured. install_root_parallel
        # gumbel closed the walker pool and cleared single-thread PUCV, so
        # leaving only clear_root_parallel_gumbel would fall through to
        # single-thread classic Gumbel even with Threads>1 / UseVL / PUCV on.
        self._reinstall_configured_search_path(after_leaving_gumbel=True)

    def _try_install_root_parallel_gumbel(self) -> bool:
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
        self._options.search_parallel = "gumbel"
        # search_parallel is the source of truth while Gumbel is selected:
        # _reinstall_configured_search_path prefers it over use_multi_gpu_pucv
        # so MaxBatch/EvalCache/VLGather rebuilds reinstall RPG, not PUCV.
        # The use_multi_gpu_pucv flag is left intact so switching back to
        # SearchParallel=pucv can restore the multi-GPU PUCV pool.
        self._worker.install_root_parallel_gumbel(
            factories, gather=gather, as_factories=True,
            devices=list(self._search_devices) if self._search_devices else None,
            info_string_cb=_emit_info_string,
            open_vpa=int(self._options.rpg_open_vpa),
            min_vpa=int(self._options.rpg_min_vpa),
            min_keep=int(self._options.rpg_min_keep),
            open_budget_frac=float(self._options.rpg_open_budget_frac),
        )
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
        n = self._parse_clamped_int(value, lo=-1)
        if n is None:
            return
        self._options.rpg_open_vpa = min(4096, n)
        _println(f"info string RPGOpenVPA set to {self._options.rpg_open_vpa}")
        self._reinstall_rpg_if_active(reason="RPGOpenVPA")

    def _set_rpg_min_vpa(self, value: str) -> None:
        n = self._parse_clamped_int(value, lo=-1)
        if n is None:
            return
        self._options.rpg_min_vpa = min(4096, n)
        _println(f"info string RPGMinVPA set to {self._options.rpg_min_vpa}")
        self._reinstall_rpg_if_active(reason="RPGMinVPA")

    def _set_rpg_min_keep(self, value: str) -> None:
        n = self._parse_clamped_int(value, lo=-1)
        if n is None:
            return
        self._options.rpg_min_keep = min(64, n)
        _println(f"info string RPGMinKeep set to {self._options.rpg_min_keep}")
        self._reinstall_rpg_if_active(reason="RPGMinKeep")

    def _set_rpg_open_budget_frac(self, value: str) -> None:
        try:
            frac = float(value.strip())
        except ValueError:
            return
        if not (0.0 < frac <= 1.0):
            return
        self._options.rpg_open_budget_frac = frac
        _println(f"info string RPGOpenBudgetFrac set to {frac}")
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
        # SHIPPED default (Threads=2 -> walker) four options `searchconfig`
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
        Threads / UseVL currently advertise.

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
        n = self._parse_clamped_int(value, lo=32)
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
        _println(f"info string VLGather set to {n}")

    def _set_multi_pv(self, value: str) -> None:
        n = self._parse_clamped_int(value, lo=1)
        if n is None:
            return
        self._options.multi_pv = n
        self._worker.set_multi_pv(n)

    def _set_show_wdl(self, value: str) -> None:
        self._options.show_wdl = value.strip().lower() == "true"
        self._worker.set_show_wdl(self._options.show_wdl)

    def _set_move_overhead_ms(self, value: str) -> None:
        n = self._parse_clamped_int(value, lo=0)
        if n is None:
            return
        self._options.move_overhead_ms = n

    def _set_clock_batch_margin_sigmas(self, value: str) -> None:
        try:
            s = float(value.strip())
        except ValueError:
            return
        s = max(0.0, s)
        self._options.clock_batch_margin_sigmas = s
        self._worker.set_batch_margin_sigmas(s)

    def _set_syzygy_50_move_rule(self, value: str) -> None:
        self._options.syzygy_50_move_rule = value.strip().lower() == "true"
  # Re-install probe so the new semantics take effect for the
  # next search. Cheap — just wraps the same path with different
  # cursed/blessed handling.
        if self._options.syzygy_path:
            self._install_tablebase(self._options.syzygy_path)

    def _set_log_file(self, value: str) -> None:
        path = value.strip()
        self._options.log_file = path
        _attach_log_file(path)

    def _set_max_batch(self, value: str) -> None:
        mb = self._parse_clamped_int(value, lo=64)
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
        new_eval = self._rebuild_evaluator(mb, self._options.eval_cache_entries)
        self._options.max_batch = mb
        self._worker.set_evaluator(new_eval)
        # set_evaluator clears multi-GPU pools; reinstall from search_parallel
        # (Gumbel) or use_multi_gpu_pucv — not only the PUCV flag.
        self._reinstall_configured_search_path()
        # The rebuilt evaluator's raw forward is warmed, but the SEARCH-path
        # cudagraph is captured separately — re-warm it before the clock.
        self._warmup_dirty = True
        _println(f"info string MaxBatch set to {mb}; evaluator rebuilt + warmed")

    def _set_eval_cache_entries(self, value: str) -> None:
        n = self._parse_clamped_int(value, lo=0)
        if n is None:
            return
        if n == self._options.eval_cache_entries:
            return
        if self._rebuild_evaluator is None:
            _println("info string EvalCacheEntries ignored — no evaluator factory wired")
            return
        _println(f"info string EvalCacheEntries rebuilding evaluator at {n}")
        new_eval = self._rebuild_evaluator(self._options.max_batch, n)
        self._options.eval_cache_entries = n
        self._worker.set_eval_cache_entries(n)
        self._worker.set_evaluator(new_eval)
        self._reinstall_configured_search_path()
        # Same as MaxBatch: evaluator rebuilt cold, so the search-path
        # cudagraph must be re-warmed before the first real ``go``.
        self._warmup_dirty = True
        _println(f"info string EvalCacheEntries set to {n}; evaluator rebuilt + warmed")

    def _set_syzygy_path(self, value: str | None) -> None:
        v = (value or "").strip()
  # Conventional UCI sentinel for "unset".
        if v.lower() in ("", "<empty>"):
            v = ""
        self._options.syzygy_path = v
        self._install_tablebase(v)

    _SETOPTION_HANDLERS: ClassVar[dict[str, Callable[[Engine, str], None]]] = {
        "hash": _set_hash,
        "ponder": _set_ponder,
        "threads": _set_threads,
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

  # -- search thread body ---------------------------------------------------

    def warmup_search(self) -> None:
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
        shapes. Best-effort: a real ``go`` surfaces any genuine error.

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
        except Exception as exc:
  # Multi-GPU tournament path: a silent warmup failure leaves cold
  # workers that then crash on the first clocked go. Surface it so
  # isready fails loud before the GUI starts the clock. Single-GPU
  # keeps best-effort (a real go still reports the error).
            if self._multi_gpu_path_active():
                raise
            logging.getLogger(__name__).debug(
                "warmup_search best-effort failure: %r", exc,
            )
        finally:
            self._worker.reset_tree()

    def _run_search(self, limits: SearchLimits, gen: int, board: chess.Board) -> None:
  # Ponder search: no deadline yet; runs until ponderhit or stop.
        result = self._run_one_phase(limits, is_ponder=limits.ponder, board=board)
        if self._ponderhit_event.is_set() and self._pending_real_limits is not None:
  # Ponderhit: opponent played our predicted move. Advance root by
  # one ply (the popped move) so the real phase searches at the
  # actual current position, reusing sims the ponder accumulated
  # below that child.
            self._stop_event = threading.Event()
            real_limits = self._pending_real_limits
            self._pending_real_limits = None
            self._ponderhit_event.clear()
            real_board = board.copy(stack=False)
            popped = self._popped_ponder_move
            self._popped_ponder_move = None
            if popped is not None:
                if not self._worker.advance_root(board, [popped]):
  # Defensive: ponder-at-M root-expanded the popped child, so
  # this shouldn't fail. Reset rather than search stale tree.
                    self._worker.reset_tree()
                self._applied_moves = (*self._applied_moves, popped)
                real_board.push(popped)
            result = self._run_one_phase(real_limits, is_ponder=False, board=real_board)
  # Gate the emit: if a newer `go` has started, our result is stale
  # and the new search owns bestmove. Silently drop.
        with self._state_lock:
            if gen != self._search_gen:
                return
        self._emit_bestmove(result)

    def _run_one_phase(
        self, limits: SearchLimits, *, is_ponder: bool, board: chess.Board,
    ) -> SearchResult:
        deadline_ms = None if is_ponder else limits.deadline_ms
        max_nodes = None if is_ponder else limits.max_nodes
        max_depth = None if is_ponder else limits.max_depth
        optimum_ms = None if is_ponder else limits.optimum_ms
        deadline = Deadline(deadline_ms=deadline_ms)
        last_emitted_nodes = [-1]
        profile_start = time.perf_counter()
        if self._gil_probe is not None:
            self._gil_probe.reset()

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

        try:
            result = self._worker.run(
                board,
                stop_event=self._stop_event,
                deadline=deadline,
                max_nodes=max_nodes,
                max_depth=max_depth,
                optimum_ms=optimum_ms,
                abort_factor=self._options.abort_factor,
                root_moves=limits.searchmoves,
                info_cb=_phase_info_cb,
                include_ponder=self._options.ponder,
                allow_terminal_shortcuts=not is_ponder and not limits.is_open_ended(),
            )
            self._report_declined_root(result)
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
            return result
        except Exception as exc:  # pragma: no cover — UCI crash-safety
            self._emit_gil_profile(
                nodes=0,
                elapsed_s=time.perf_counter() - profile_start,
                is_ponder=is_ponder,
            )
            _println(f"info string search error: {exc!r}")
            fallback, source = self._bestmove_fallback(board, limits.searchmoves)
            self._bestmove_fallback_used += 1
            _println(
                f"info string bestmove_fallback_used={self._bestmove_fallback_used} "
                f"source={source} move={fallback} exception={type(exc).__name__}"
            )
            return SearchResult(
                bestmove_uci=fallback, ponder_uci=None, nodes=0, pv=(), score_cp=0, tbhits=0,
            )

    def _report_declined_root(self, result: SearchResult) -> None:
        """Announce and count a bestmove the search REFUSED to produce.

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
        return _legal_fallback_move(board, searchmoves), "first_legal"

    @property
    def bestmove_fallback_used(self) -> int:
        """How many times this process answered a `go` from the fallback path.

        Monotonic for the life of the engine (a GUI plays many games in one
        process), so it is a session total, not a per-game one. Zero is the only
        healthy value; every increment is one move the search did not produce.
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

    def _emit_bestmove(self, result: SearchResult) -> None:
        ponder = result.ponder_uci if self._options.ponder else None
        _println(format_bestmove(result.bestmove_uci, ponder=ponder))

    def close(self) -> None:
        """Stop any running search and release the worker's evaluator. Call
        from the UCI main loop's finally block so ``BatchCoalescingDispatcher``
        (if active) drains its non-daemon submitter before the interpreter
        tears down torch's CUDA context."""
        self._wait_for_search()
        if self._gil_probe is not None:
            self._gil_probe.close()
        self._worker.close()

  # -- helpers --------------------------------------------------------------

    def _wait_for_search(self) -> None:
        if self._search_thread is not None and self._search_thread.is_alive():
            self._stop_event.set()
            self._search_thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._search_thread.is_alive():
                _println("info string search stop timed out; thread still running")
            else:
                self._search_thread = None


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


def _println(s: str) -> None:
    with _PRINT_LOCK:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()


def _emit_info_string(s: str) -> None:
    """`info string` emitter handed to search pools (root-parallel Gumbel
    per-phase lines). Runs on the search thread at phase barriers, protected
    by the same stdout lock as every other UCI line."""
    _println(f"info string {s}")
