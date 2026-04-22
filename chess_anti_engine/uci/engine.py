"""Engine — holds board state, a SearchWorker, and the command dispatch.

Command handlers are called from the main UCI loop in response to parsed
commands. ``go`` kicks off a background thread that runs the search, so
the main loop can stay responsive for ``stop`` / ``ponderhit``.

v1 scope: no pondering yet. Implements the ``go`` / ``stop`` / ``quit``
path end-to-end so we can play a game.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, replace
from typing import Callable

import chess

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
from chess_anti_engine.tablebase import SyzygyProbe, get_tablebase

from .search import SearchResult, SearchWorker
from .time_manager import Deadline, SearchLimits, limits_from_go


_ENGINE_NAME = "DeepFin"
_ENGINE_AUTHOR = "jjosh"


def emit_handshake(options: "EngineOptions") -> None:
    """Print the UCI `uci` response — id, options, uciok — using ``options``
    defaults. Kept out of the Engine class so ``__main__`` can reply to the
    GUI before model load finishes (standard Lc0 pattern: the handshake is
    static, ``readyok`` still waits for real readiness)."""
    for line in format_id_lines(_ENGINE_NAME, _ENGINE_AUTHOR):
        _println(line)
    _println(f"option name Hash type spin default {options.hash_mb} min 1024 max 524288")
    _println(f"option name Threads type spin default {options.threads} min 1 max 64")
    _println(f"option name MaxBatch type spin default {options.max_batch} min 64 max 8192")
    _println(f"option name MinibatchSize type spin default {options.minibatch_size} min 0 max 8192")
    _println(f"option name MultiPV type spin default {options.multi_pv} min 1 max 256")
    _println("option name SyzygyPath type string default <empty>")
    _println(f"option name Ponder type check default {'true' if options.ponder else 'false'}")
    _println(format_uciok())

# How long to wait for a search thread to honor `stop`. 5s was too tight:
# a cold CUDA-graph compile on the first search can take ~3-4s, which
# previously let a slow stop orphan a running thread that would later
# emit a stale bestmove against a new board.
_JOIN_TIMEOUT_S = 30.0


@dataclass
class EngineOptions:
    # Soft cap on MCTS tree memory in MB. Search halts between chunks when
    # the tree's own allocations exceed this — prevents runaway growth from
    # pushing the process into swap on long analysis. It is NOT a bounded
    # transposition table; we stop adding nodes rather than evicting.
    hash_mb: int = 4096
    # Number of MCTS walker threads. 1 = classic Gumbel path; >1 = PUCT
    # walker pool with virtual loss. Set via UCI `Threads`.
    threads: int = 2
    # Hardware-side max batch: the largest leaf-batch shape the evaluator
    # allocates buffers + CUDA-graph captures for. Changing it rebuilds
    # the evaluator + re-warmup (5-10s stall). Set via UCI `MaxBatch`.
    max_batch: int = 1024
    # Minibatch target the C gumbel state machine aims for before flushing
    # leaves to GPU eval. 0 = C-side default (GSS_GPU_BATCH = 1024). Live
    # update; takes effect on the next chunk.
    minibatch_size: int = 0
    # Number of top-ranked lines to emit per info tick. 1 = classic single
    # PV (no `multipv` field). >1 emits N lines each tagged `multipv k`.
    multi_pv: int = 1
    # Syzygy tablebase directory path(s). Multiple paths separated by
    # OS-conventional separators (';' on Windows, ':' elsewhere) per the
    # de-facto UCI convention. Empty means disabled.
    syzygy_path: str = ""
    # Ponder is a signal to the GUI about whether to issue `go ponder`
    # commands; the engine itself honors `go ponder` regardless, since
    # ignoring it would break cutechess/Arena if the user forgets to flip
    # the option. Default off — safer for fixed-time match play.
    ponder: bool = False


class Engine:
    def __init__(
        self,
        worker: SearchWorker,
        *,
        rebuild_evaluator: "Callable[[int], object] | None" = None,
    ) -> None:
        self._worker = worker
        # Factory handed in by __main__. Captures the model, devices, and
        # coalesce flag; takes a max_batch and returns a warmed-up
        # evaluator. When None, the MaxBatch setoption silently no-ops
        # (e.g., in unit-test harnesses that don't build a real evaluator).
        self._rebuild_evaluator = rebuild_evaluator
        self._options = EngineOptions()
        self._worker.set_max_tree_mb(self._options.hash_mb)
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
        # Block until any running search has responded to a stop. Simpler
        # than juggling readyok semantics during an in-flight search.
        self._wait_for_search()
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
        limits = limits_from_go(cmd.args, side_to_move_is_white=(search_board.turn == chess.WHITE))
        self._stop_event = threading.Event()
        self._ponderhit_event = threading.Event()
        # For `go ponder`, the ponder phase runs open-ended; ponderhit
        # converts to a real-deadline phase using the SAME underlying clock
        # args. We re-derive the "real" limits here by synthesizing a
        # non-ponder copy of the args.
        self._pending_real_limits = (
            limits_from_go(
                replace(cmd.args, ponder=False),
                side_to_move_is_white=(search_board.turn == chess.WHITE),
            )
            if cmd.args.ponder else None
        )
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
        if name == "hash" and cmd.value is not None:
            try:
                self._options.hash_mb = int(cmd.value)
                self._worker.set_max_tree_mb(self._options.hash_mb)
            except ValueError:
                pass
        elif name == "ponder" and cmd.value is not None:
            self._options.ponder = cmd.value.strip().lower() == "true"
        elif name == "threads" and cmd.value is not None:
            try:
                n = max(1, int(cmd.value))
            except ValueError:
                return
            self._options.threads = n
            self._worker.set_num_threads(n)
            _println(
                f"info string Threads set to {n} "
                f"({'walker pool' if n > 1 else 'classic Gumbel path'})"
            )
        elif name == "multipv" and cmd.value is not None:
            try:
                n = max(1, int(cmd.value))
            except ValueError:
                return
            self._options.multi_pv = n
            self._worker.set_multi_pv(n)
        elif name == "minibatchsize" and cmd.value is not None:
            try:
                n = max(0, int(cmd.value))
            except ValueError:
                return
            self._options.minibatch_size = n
            self._worker.set_minibatch_size(n)
            _println(
                f"info string MinibatchSize set to {n} "
                f"({'default' if n == 0 else f'{n} leaves per GPU flush'})"
            )
        elif name == "maxbatch" and cmd.value is not None:
            try:
                mb = max(64, int(cmd.value))
            except ValueError:
                return
            if self._rebuild_evaluator is None:
                _println("info string MaxBatch ignored — no evaluator factory wired")
                return
            if mb == self._options.max_batch:
                return
            # Rebuild is 5-10s (CUDA graph recapture on first forward).
            # User sees the stall only if they poll isready soon after.
            _println(f"info string MaxBatch rebuilding evaluator at {mb}…")
            new_eval = self._rebuild_evaluator(mb)
            self._options.max_batch = mb
            self._worker.set_evaluator(new_eval)
            _println(f"info string MaxBatch set to {mb}; evaluator rebuilt + warmed")
        elif name == "syzygypath":
            value = (cmd.value or "").strip()
            # Conventional UCI sentinel for "unset".
            if value.lower() in ("", "<empty>"):
                value = ""
            self._options.syzygy_path = value
            self._install_tablebase(value)
        # All other options silently accepted — we don't expose any yet.

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
        probe = SyzygyProbe(path)
        self._worker.set_tb_probe(probe)
        _println(
            f"info string SyzygyPath loaded from {path!r}: "
            f"{probe.n_wdl} WDL + {probe.n_dtz} DTZ tables, "
            f"up to {probe.max_pieces}-piece positions"
        )

    # -- search thread body ---------------------------------------------------

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
                self._applied_moves = self._applied_moves + (popped,)
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
        deadline = Deadline(deadline_ms=deadline_ms)
        try:
            return self._worker.run(
                board,
                stop_event=self._stop_event,
                deadline=deadline,
                max_nodes=max_nodes,
                max_depth=max_depth,
                info_cb=self._emit_info,
            )
        except Exception as exc:  # pragma: no cover — UCI crash-safety
            _println(f"info string search error: {exc!r}")
            return SearchResult(
                bestmove_uci="0000", ponder_uci=None, nodes=0, pv=(), score_cp=0, tbhits=0,
            )

    def _emit_info(
        self, *,
        nodes: int, elapsed_ms: int, score_cp: int, pv: tuple[str, ...],
        tbhits: int, score_mate: int | None, multipv: int | None,
    ) -> None:
        nps = int(nodes * 1000 / max(1, elapsed_ms))
        _println(format_info(InfoFields(
            depth=len(pv),
            multipv=multipv,
            nodes=nodes,
            nps=nps,
            time_ms=elapsed_ms,
            score_cp=score_cp,
            score_mate=score_mate,
            pv=pv,
            tbhits=tbhits,
        )))

    def _emit_bestmove(self, result: SearchResult) -> None:
        _println(format_bestmove(result.bestmove_uci, ponder=result.ponder_uci))

    # -- helpers --------------------------------------------------------------

    def _wait_for_search(self) -> None:
        if self._search_thread is not None and self._search_thread.is_alive():
            self._stop_event.set()
            self._search_thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._search_thread.is_alive():
                _println("info string search stop timed out; thread still running")
            else:
                self._search_thread = None


def _println(s: str) -> None:
    sys.stdout.write(s + "\n")
    sys.stdout.flush()
