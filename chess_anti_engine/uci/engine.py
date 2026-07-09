"""Engine — holds board state, a SearchWorker, and the command dispatch.

Command handlers are called from the main UCI loop in response to parsed
commands. ``go`` kicks off a background thread that runs the search, so
the main loop can stay responsive for ``stop`` / ``ponderhit``.

v1 scope: no pondering yet. Implements the ``go`` / ``stop`` / ``quit``
path end-to-end so we can play a game.
"""
from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar
from collections.abc import Callable

import chess

from chess_anti_engine.tablebase import SyzygyProbe, get_tablebase

from .protocol import (
    CmdGo,
    CmdIsReady,
    CmdPonderHit,
    CmdPosition,
    CmdQuit,
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
    _println(f"option name Hash type spin default {options.hash_mb} min 1024 max 524288")
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
    _println(f"option name VLGather type spin default {options.vl_gather} min 32 max 4096")
    _println(f"option name MaxBatch type spin default {options.max_batch} min 64 max 8192")
    _println(f"option name EvalCacheEntries type spin default {options.eval_cache_entries} min 0 max 1048576")
    _println(f"option name MinibatchSize type spin default {options.minibatch_size} min 0 max 8192")
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
    _println(format_uciok())

# Tight (e.g. 5s) timeouts let a slow stop orphan a thread that emits a
# stale bestmove against the next board — a cold CUDA-graph compile on
# the first search is ~3-4s on its own.
_JOIN_TIMEOUT_S = 30.0


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


class Engine:
    def __init__(
        self,
        worker: SearchWorker,
        *,
        rebuild_evaluator: Callable[[int, int], BatchEvaluator] | None = None,
        rebuild_multi_gpu_pucv_factories: (
            Callable[[int, int], list[Callable[[], BatchEvaluator]]] | None
        ) = None,
        options: EngineOptions | None = None,
    ) -> None:
        self._worker = worker
  # Factory that takes a max_batch and returns a warmed-up evaluator
  # (model + devices + coalesce flag are captured at construction).
  # When None, the MaxBatch setoption silently no-ops.
        self._rebuild_evaluator = rebuild_evaluator
        self._rebuild_multi_gpu_pucv_factories = rebuild_multi_gpu_pucv_factories
        self._options = options if options is not None else EngineOptions()
        self._worker.set_max_tree_mb(self._options.hash_mb)
  # Set by setoption handlers that reshape the search path / evaluator
  # (Threads/UseVL/UseMultiGpuPUCV/LeafGather/VLGather/MaxBatch/
  # EvalCacheEntries). The build-time ``warmup_search`` runs BEFORE any
  # setoption is applied, so such a change leaves its cudagraph stale;
  # ``_handle_isready`` re-warms the configured path before ``readyok``
  # so the first real ``go`` never pays cold capture on the clock.
        self._warmup_dirty = False
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
            except ValueError:
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
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                break
            if mv not in new_board.legal_moves:
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
  # All other options silently accepted — we don't expose any yet.

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
        n = self._parse_clamped_int(value, lo=1)
        if n is None:
            return
        self._options.hash_mb = n
        self._worker.set_max_tree_mb(n)

    def _set_ponder(self, value: str) -> None:
        self._options.ponder = value.strip().lower() == "true"

    def _set_threads(self, value: str) -> None:
        n = self._parse_clamped_int(value, lo=1)
        if n is None:
            return
        self._options.threads = n
        self._worker.set_num_threads(n)
        self._warmup_dirty = True
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
        self._warmup_dirty = True
        _println(
            f"info string UseVL {'on' if enabled else 'off'} "
            f"(gather={self._options.vl_gather}; "
            f"requires Threads=1 + 2-slot async evaluator)"
        )

    def _set_use_multi_gpu_pucv(self, value: str) -> None:
        enabled = value.strip().lower() == "true"
        self._options.use_multi_gpu_pucv = enabled
        if not enabled:
            self._worker.clear_multi_gpu_pucv()
            self._warmup_dirty = True
            _println("info string UseMultiGpuPUCV off")
            return
        if self._rebuild_multi_gpu_pucv_factories is None:
            self._options.use_multi_gpu_pucv = False
            _println("info string UseMultiGpuPUCV unavailable — no per-device factories wired")
            return
        gather = min(self._options.vl_gather, self._options.max_batch)
        factories = self._rebuild_multi_gpu_pucv_factories(
            self._options.max_batch, gather,
        )
        if len(factories) < 2:
            self._options.use_multi_gpu_pucv = False
            _println("info string UseMultiGpuPUCV unavailable — need at least two devices")
            return
        self._worker.install_multi_gpu_pucv(
            factories, gather=gather, as_factories=True,
        )
        self._warmup_dirty = True
        _println(
            f"info string UseMultiGpuPUCV on "
            f"(devices={len(factories)} gather={gather})"
        )
  # Same inertness as the Threads>1 walker pool (see _set_threads): the
  # pool selects with plain PUCT and never reads c_scale/c_visit, so the
  # tuned Gumbel values silently stop applying while it is installed.
        _println(
            "info string note: c_scale/c_visit are inactive on the multi-GPU "
            "PUCV pool (plain PUCT); a single device with Threads=1 restores "
            "the c_scale-tuned classic Gumbel path"
        )

    def _set_pucv_pending_mode(self, value: str) -> None:
        normalized = value.strip().lower().replace("_", "-")
        if normalized not in ("legacy", "virtual-mean"):
            return
        self._options.pucv_pending_mode = normalized
        self._worker.set_pucv_vloss_mode(1 if normalized == "virtual-mean" else 0)
        if self._options.use_multi_gpu_pucv:
            self._set_use_multi_gpu_pucv("true")
        _println(f"info string PUCVPendingMode set to {normalized}")

    def _set_vl_gather(self, value: str) -> None:
        n = self._parse_clamped_int(value, lo=32)
        if n is None:
            return
        self._options.vl_gather = n
        if self._options.use_vl:
            self._worker.set_use_pucv(True, gather=n)
        if self._options.use_multi_gpu_pucv:
            self._set_use_multi_gpu_pucv("true")
  # VLGather reshapes the pucv / multi-GPU per-worker batch, the same way
  # LeafGather reshapes the walker batch — re-warm the configured path.
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

    def _set_minibatch_size(self, value: str) -> None:
        n = self._parse_clamped_int(value, lo=0)
        if n is None:
            return
        self._options.minibatch_size = n
        self._worker.set_minibatch_size(n)
  # minibatch_size feeds target_batch into start_gumbel_sims, so the search-path
  # cudagraph captured at build-time warmup is now the wrong shape — re-warm before
  # the clock or the first real `go` pays cold capture (same contract as MaxBatch).
        self._warmup_dirty = True
        _println(
            f"info string MinibatchSize set to {n} "
            f"({'default' if n == 0 else f'{n} leaves per GPU flush'})"
        )

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
        if self._options.use_multi_gpu_pucv:
            self._set_use_multi_gpu_pucv("true")
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
        if self._options.use_multi_gpu_pucv:
            self._set_use_multi_gpu_pucv("true")
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
        "vlgather": _set_vl_gather,
        "multipv": _set_multi_pv,
        "uci_showwdl": _set_show_wdl,
        "moveoverheadms": _set_move_overhead_ms,
        "clockbatchmarginsigmas": _set_clock_batch_margin_sigmas,
        "syzygy50moverule": _set_syzygy_50_move_rule,
        "logfile": _set_log_file,
        "minibatchsize": _set_minibatch_size,
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
        ``max(chunk, gather)`` per device) so every worker gets a full batch."""
        chunk = int(getattr(self._worker, "_chunk_sims", 512))
        max_nodes = max(256, chunk)
        pool = getattr(self._worker, "_pucv_pool", None)
        if pool is not None:
            n_devices = getattr(pool, "n_devices", None)
            gather = getattr(pool, "gather", None)
            if n_devices and gather:
                per_device = max(chunk, int(gather))
                max_nodes = max(max_nodes, per_device * int(n_devices))
        try:
            self._worker.run(
                chess.Board(),
                stop_event=threading.Event(),
                deadline=Deadline(deadline_ms=None),
                max_nodes=max_nodes,
                optimum_ms=None,
                abort_factor=self._options.abort_factor,
            )
        except Exception:
            pass
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
        emitted_info = False

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
            nonlocal emitted_info
            emitted_info = True
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
            if not is_ponder and not emitted_info:
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
            return result
        except Exception as exc:  # pragma: no cover — UCI crash-safety
            _println(f"info string search error: {exc!r}")
            fallback = _legal_fallback_move(board, limits.searchmoves)
            return SearchResult(
                bestmove_uci=fallback, ponder_uci=None, nodes=0, pv=(), score_cp=0, tbhits=0,
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
