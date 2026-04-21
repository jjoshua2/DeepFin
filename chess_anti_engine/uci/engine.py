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
from .search import SearchResult, SearchWorker
from .time_manager import Deadline, SearchLimits, limits_from_go


_ENGINE_NAME = "chess-anti-engine"
_ENGINE_AUTHOR = "josh + claude"

# How long to wait for a search thread to honor `stop`. 5s was too tight:
# a cold CUDA-graph compile on the first search can take ~3-4s, which
# previously let a slow stop orphan a running thread that would later
# emit a stale bestmove against a new board.
_JOIN_TIMEOUT_S = 30.0


@dataclass
class EngineOptions:
    # Declared UCI options are reported on `uci` but have no effect yet.
    # Hash is the obvious one GUIs poke; we ignore the value but accept it
    # cleanly so cutechess-cli doesn't complain.
    hash_mb: int = 256


class Engine:
    def __init__(self, worker: SearchWorker) -> None:
        self._worker = worker
        self._options = EngineOptions()
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
        # Anything else (CmdUnknown, CmdDebug) is silently ignored.

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    # -- handlers -------------------------------------------------------------

    def _handle_uci(self) -> None:
        for line in format_id_lines(_ENGINE_NAME, _ENGINE_AUTHOR):
            _println(line)
        _println(f"option name Hash type spin default {self._options.hash_mb} min 1 max 65536")
        _println(format_uciok())

    def _handle_isready(self) -> None:
        # Block until any running search has responded to a stop. Simpler
        # than juggling readyok semantics during an in-flight search.
        self._wait_for_search()
        _println(format_readyok())

    def _handle_newgame(self) -> None:
        self._wait_for_search()
        self._worker.reset_tree()
        self._board = chess.Board()

    def _handle_position(self, cmd: CmdPosition) -> None:
        self._wait_for_search()
        if cmd.fen is None:
            self._board = chess.Board()
        else:
            try:
                self._board = chess.Board(cmd.fen)
            except ValueError:
                self._board = chess.Board()
                return
        for uci in cmd.moves:
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                break
            if mv not in self._board.legal_moves:
                break
            self._board.push(mv)
        # Different position from the last search — tree must be rebuilt.
        self._worker.reset_tree()

    def _handle_go(self, cmd: CmdGo) -> None:
        if self._search_thread is not None and self._search_thread.is_alive():
            # UCI says you should `stop` before issuing another `go`. Be
            # forgiving: stop the current one first.
            self._handle_stop()
        limits = limits_from_go(cmd.args, side_to_move_is_white=(self._board.turn == chess.WHITE))
        self._stop_event = threading.Event()
        self._ponderhit_event = threading.Event()
        # For `go ponder`, the ponder phase runs open-ended; ponderhit
        # converts to a real-deadline phase using the SAME underlying clock
        # args. We re-derive the "real" limits here by synthesizing a
        # non-ponder copy of the args.
        self._pending_real_limits = (
            limits_from_go(
                replace(cmd.args, ponder=False),
                side_to_move_is_white=(self._board.turn == chess.WHITE),
            )
            if cmd.args.ponder else None
        )
        with self._state_lock:
            self._search_gen += 1
            gen = self._search_gen
        self._search_thread = threading.Thread(
            target=self._run_search,
            args=(limits, gen),
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
        if cmd.name.lower() == "hash" and cmd.value is not None:
            try:
                self._options.hash_mb = int(cmd.value)
            except ValueError:
                pass
        # All other options silently accepted — we don't expose any yet.

    # -- search thread body ---------------------------------------------------

    def _run_search(self, limits: SearchLimits, gen: int) -> None:
        # Ponder search: no deadline yet; runs until ponderhit or stop.
        result = self._run_one_phase(limits, is_ponder=limits.ponder)
        if self._ponderhit_event.is_set() and self._pending_real_limits is not None:
            # Ponderhit arrived: resume the same tree with real limits.
            self._stop_event = threading.Event()
            real_limits = self._pending_real_limits
            self._pending_real_limits = None
            self._ponderhit_event.clear()
            result = self._run_one_phase(real_limits, is_ponder=False)
        # Gate the emit: if a newer `go` has started, our result is stale
        # and the new search owns bestmove. Silently drop.
        with self._state_lock:
            if gen != self._search_gen:
                return
        self._emit_bestmove(result)

    def _run_one_phase(self, limits: SearchLimits, *, is_ponder: bool) -> SearchResult:
        deadline_ms = None if is_ponder else limits.deadline_ms
        max_nodes = None if is_ponder else limits.max_nodes
        max_depth = None if is_ponder else limits.max_depth
        deadline = Deadline(deadline_ms=deadline_ms)
        try:
            return self._worker.run(
                self._board,
                stop_event=self._stop_event,
                deadline=deadline,
                max_nodes=max_nodes,
                max_depth=max_depth,
                info_cb=self._emit_info,
            )
        except Exception as exc:  # pragma: no cover — UCI crash-safety
            _println(f"info string search error: {exc!r}")
            return SearchResult(
                bestmove_uci="0000", ponder_uci=None, nodes=0, pv=(), score_cp=0,
            )

    def _emit_info(self, *, nodes: int, elapsed_ms: int, score_cp: int, pv: tuple[str, ...]) -> None:
        nps = int(nodes * 1000 / max(1, elapsed_ms))
        _println(format_info(InfoFields(
            depth=len(pv),
            nodes=nodes,
            nps=nps,
            time_ms=elapsed_ms,
            score_cp=score_cp,
            pv=pv,
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
