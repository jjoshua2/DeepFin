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
from dataclasses import dataclass

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
        self._quit_requested = False

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
            # v1: no pondering — treat as stop (emits bestmove from whatever
            # search we have in flight, which is none, so no-op).
            self._handle_stop()
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
        self._search_thread = threading.Thread(
            target=self._run_search,
            args=(limits,),
            daemon=True,
        )
        self._search_thread.start()

    def _handle_stop(self) -> None:
        if self._search_thread is not None:
            self._stop_event.set()
            self._search_thread.join(timeout=5.0)
            self._search_thread = None

    def _handle_setoption(self, cmd: CmdSetOption) -> None:
        if cmd.name.lower() == "hash" and cmd.value is not None:
            try:
                self._options.hash_mb = int(cmd.value)
            except ValueError:
                pass
        # All other options silently accepted — we don't expose any yet.

    # -- search thread body ---------------------------------------------------

    def _run_search(self, limits: SearchLimits) -> None:
        deadline = Deadline(deadline_ms=limits.deadline_ms)
        try:
            result = self._worker.run(
                self._board,
                stop_event=self._stop_event,
                deadline=deadline,
                max_nodes=limits.max_nodes if not limits.is_open_ended() else None,
                info_cb=self._emit_info,
            )
        except Exception as exc:  # pragma: no cover — UCI crash-safety
            _println(f"info string search error: {exc!r}")
            result = SearchResult(
                bestmove_uci="0000", ponder_uci=None, nodes=0, pv=(), score_cp=0,
            )
        self._emit_bestmove(result)

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
        _println(format_bestmove(result.bestmove_uci, ponder=None))

    # -- helpers --------------------------------------------------------------

    def _wait_for_search(self) -> None:
        if self._search_thread is not None and self._search_thread.is_alive():
            self._stop_event.set()
            self._search_thread.join(timeout=5.0)
            self._search_thread = None


def _println(s: str) -> None:
    sys.stdout.write(s + "\n")
    sys.stdout.flush()
