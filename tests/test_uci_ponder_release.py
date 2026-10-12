"""Ponder must not publish bestmove until the GUI sends stop or ponderhit.

``SearchWorker.run`` returns immediately for a mated root, a tablebase hit,
a declined draw, a full hash, or an exception. Those are not UCI events.
A pool fault also sets the search stop event before it raises. That event
is not ``stop`` or ``ponderhit``. ``go ponder`` searches the position
*before* the predicted reply, so that early result is the opponent's move.
Publishing it lets the GUI play it after ``ponderhit``.
"""
from __future__ import annotations

import threading
from dataclasses import replace
import time
from typing import cast

import chess
import pytest

from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import (
    CmdGo,
    CmdIsReady,
    CmdPonderHit,
    CmdPosition,
    CmdQuit,
    CmdStop,
    GoArgs,
)
from chess_anti_engine.uci.search import SearchResult, SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


class _ImmediateWorker:
    """Stand-in for a worker that has already finished the root."""

    def __init__(self) -> None:
        self.turns: list[bool] = []
        self.moves: list[str] = []
        self.deadline_finite: list[bool] = []
        self.started = threading.Event()

    def set_max_tree_mb(self, _mb: int) -> None:
        return None

    def reset_tree(self) -> None:
        return None

    def advance_root(self, _board: chess.Board, _moves: list[chess.Move]) -> bool:
        return True

    def run(
        self,
        board: chess.Board,
        *,
        stop_event: threading.Event,
        deadline: Deadline,
        max_nodes: int | None = None,
        max_depth: int | None = None,
        optimum_ms: int | None = None,
        abort_factor: float = 0.0,
        root_moves: tuple[str, ...] = (),
        info_cb: object = None,
        include_ponder: bool = False,
        allow_terminal_shortcuts: bool = True,
    ) -> SearchResult:
        del (
            stop_event,
            max_nodes,
            max_depth,
            optimum_ms,
            abort_factor,
            root_moves,
            info_cb,
            include_ponder,
            allow_terminal_shortcuts,
        )
        move = next(iter(board.legal_moves)).uci()
        self.turns.append(board.turn)
        self.moves.append(move)
        self.deadline_finite.append(deadline.remaining_ms() is not None)
        self.started.set()
        return SearchResult(
            bestmove_uci=move,
            ponder_uci=None,
            nodes=1,
            pv=(move,),
            score_cp=0,
            tbhits=0,
        )


class _BlockingWorker(_ImmediateWorker):
    """First phase blocks until stop; the real phase returns at once."""

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def run(
        self,
        board: chess.Board,
        *,
        stop_event: threading.Event,
        deadline: Deadline,
        max_nodes: int | None = None,
        max_depth: int | None = None,
        optimum_ms: int | None = None,
        abort_factor: float = 0.0,
        root_moves: tuple[str, ...] = (),
        info_cb: object = None,
        include_ponder: bool = False,
        allow_terminal_shortcuts: bool = True,
    ) -> SearchResult:
        self._calls += 1
        if self._calls == 1:
            self.started.set()
            if not stop_event.wait(timeout=3.0):
                raise TimeoutError("ponder phase was not released")
        return super().run(
            board,
            stop_event=stop_event,
            deadline=deadline,
            max_nodes=max_nodes,
            max_depth=max_depth,
            optimum_ms=optimum_ms,
            abort_factor=abort_factor,
            root_moves=root_moves,
            info_cb=info_cb,
            include_ponder=include_ponder,
            allow_terminal_shortcuts=allow_terminal_shortcuts,
        )


class _FaultThenRun(_ImmediateWorker):
    """First phase sets the search stop event and raises, as a pool fault does."""

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def run(
        self,
        board: chess.Board,
        *,
        stop_event: threading.Event,
        deadline: Deadline,
        max_nodes: int | None = None,
        max_depth: int | None = None,
        optimum_ms: int | None = None,
        abort_factor: float = 0.0,
        root_moves: tuple[str, ...] = (),
        info_cb: object = None,
        include_ponder: bool = False,
        allow_terminal_shortcuts: bool = True,
    ) -> SearchResult:
        self._calls += 1
        if self._calls == 1:
            self.started.set()
            stop_event.set()
            raise RuntimeError("pool evaluator failed")
        return super().run(
            board,
            stop_event=stop_event,
            deadline=deadline,
            max_nodes=max_nodes,
            max_depth=max_depth,
            optimum_ms=optimum_ms,
            abort_factor=abort_factor,
            root_moves=root_moves,
            info_cb=info_cb,
            include_ponder=include_ponder,
            allow_terminal_shortcuts=allow_terminal_shortcuts,
        )


class _DeclineThenRun(_ImmediateWorker):
    """Return a declined prior in ponder, optionally again in the main phase."""

    def __init__(self, *, decline_main: bool = False) -> None:
        super().__init__()
        self._calls = 0
        self.decline_main = decline_main

    def run(
        self,
        board: chess.Board,
        *,
        stop_event: threading.Event,
        deadline: Deadline,
        max_nodes: int | None = None,
        max_depth: int | None = None,
        optimum_ms: int | None = None,
        abort_factor: float = 0.0,
        root_moves: tuple[str, ...] = (),
        info_cb: object = None,
        include_ponder: bool = False,
        allow_terminal_shortcuts: bool = True,
    ) -> SearchResult:
        self._calls += 1
        result = super().run(
            board,
            stop_event=stop_event,
            deadline=deadline,
            max_nodes=max_nodes,
            max_depth=max_depth,
            optimum_ms=optimum_ms,
            abort_factor=abort_factor,
            root_moves=root_moves,
            info_cb=info_cb,
            include_ponder=include_ponder,
            allow_terminal_shortcuts=allow_terminal_shortcuts,
        )

        if self._calls == 1 or self.decline_main:
            return replace(result, nodes=0, root_declined="threefold repetition")
        return result


def _engine(worker: _ImmediateWorker) -> Engine:
    return Engine(worker=cast(SearchWorker, cast(object, worker)))


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    lines: list[str] = []

    def _print(text: str) -> None:
        lines.append(text)

    monkeypatch.setattr("chess_anti_engine.uci.engine._println", _print)
    return lines


def _bestmoves(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("bestmove ")]


def _ponder_go() -> CmdGo:
    return CmdGo(
        args=GoArgs(
            ponder=True,
            wtime_ms=30_000,
            btime_ms=30_000,
            winc_ms=100,
            binc_ms=100,
            movestogo=20,
        ),
    )


def _start_ponder(engine: Engine, worker: _ImmediateWorker) -> None:
    engine.dispatch(CmdPosition(fen=None, moves=("e2e4", "e7e5")))
    engine.dispatch(_ponder_go())
    assert worker.started.wait(timeout=2.0), "ponder search did not start"


def _assert_quiet(lines: list[str], *, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        assert _bestmoves(lines) == []
        time.sleep(0.01)


def test_early_ponder_holds_bestmove_until_ponderhit(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = _capture(monkeypatch)
    worker = _ImmediateWorker()
    engine = _engine(worker)
    _start_ponder(engine, worker)
    engine.dispatch(CmdIsReady())
    assert "readyok" in lines
    _assert_quiet(lines, seconds=0.25)
    assert worker.turns == [chess.BLACK]
    assert worker.deadline_finite == [False]
    thread = engine._search_thread
    assert thread is not None
    assert thread.is_alive()

    engine.dispatch(CmdPonderHit())
    assert thread.join(timeout=2.0) is None
    assert not thread.is_alive()
    published = _bestmoves(lines)
    assert published == [f"bestmove {worker.moves[1]}"]
    assert worker.turns == [chess.BLACK, chess.WHITE]
    assert worker.deadline_finite == [False, True]
    assert worker.moves[0] != worker.moves[1]


def test_stop_after_early_ponder_emits_the_pre_reply_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _capture(monkeypatch)
    worker = _ImmediateWorker()
    engine = _engine(worker)
    _start_ponder(engine, worker)
    _assert_quiet(lines, seconds=0.15)
    engine.dispatch(CmdStop())
    thread = engine._search_thread
    assert thread is None or not thread.is_alive()
    assert _bestmoves(lines) == [f"bestmove {worker.moves[0]}"]
    assert worker.turns == [chess.BLACK]


def test_quit_during_held_ponder_does_not_hang(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = _capture(monkeypatch)
    worker = _ImmediateWorker()
    engine = _engine(worker)
    _start_ponder(engine, worker)
    _assert_quiet(lines, seconds=0.05)
    engine.dispatch(CmdQuit())
    thread = engine._search_thread
    assert engine.quit_requested
    assert thread is None or not thread.is_alive()
    assert len(_bestmoves(lines)) == 1


def test_blocking_ponder_still_hands_off_on_ponderhit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _capture(monkeypatch)
    worker = _BlockingWorker()
    engine = _engine(worker)
    _start_ponder(engine, worker)
    _assert_quiet(lines, seconds=0.05)
    assert worker.turns == []
    engine.dispatch(CmdPonderHit())
    thread = engine._search_thread
    assert thread is not None
    assert thread.join(timeout=2.0) is None
    assert not thread.is_alive()
    assert worker.turns == [chess.BLACK, chess.WHITE]
    assert worker.deadline_finite == [False, True]
    assert _bestmoves(lines) == [f"bestmove {worker.moves[1]}"]


def test_pool_fault_during_ponder_holds_until_ponderhit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _capture(monkeypatch)
    worker = _FaultThenRun()
    engine = _engine(worker)
    _start_ponder(engine, worker)
    _assert_quiet(lines, seconds=0.15)
    thread = engine._search_thread
    assert thread is not None
    assert thread.is_alive()
    assert worker.turns == []

    engine.dispatch(CmdPonderHit())
    assert thread.join(timeout=2.0) is None
    assert worker.turns == [chess.WHITE]
    assert worker.deadline_finite == [True]
    assert _bestmoves(lines) == [f"bestmove {worker.moves[0]}"]
    assert engine.bestmove_fallback_used == 1
    diagnostics = [line for line in lines if "bestmove_fallback_used=" in line]
    assert len(diagnostics) == 1
    assert "phase=ponder" in diagnostics[0]
    assert "move=g8h6" in diagnostics[0]
    assert worker.moves[0] == "g1h3"


def test_stop_after_pool_fault_emits_one_pre_reply_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _capture(monkeypatch)
    worker = _FaultThenRun()
    engine = _engine(worker)
    _start_ponder(engine, worker)
    _assert_quiet(lines, seconds=0.05)
    engine.dispatch(CmdStop())
    thread = engine._search_thread
    assert thread is None or not thread.is_alive()
    assert worker.turns == []
    pre_reply = chess.Board()
    pre_reply.push_uci("e2e4")
    assert _bestmoves(lines) == [f"bestmove {next(iter(pre_reply.legal_moves)).uci()}"]
    assert engine.bestmove_fallback_used == 1
    diagnostics = [line for line in lines if "bestmove_fallback_used=" in line]
    assert len(diagnostics) == 1
    assert "phase=ponder" in diagnostics[0]
    assert "move=g8h6" in diagnostics[0]


def test_non_ponder_search_still_publishes_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _capture(monkeypatch)
    worker = _ImmediateWorker()
    engine = _engine(worker)
    engine.dispatch(CmdPosition(fen=None, moves=()))
    started = time.perf_counter()
    engine.dispatch(CmdGo(args=GoArgs(wtime_ms=30_000, btime_ms=30_000)))
    thread = engine._search_thread
    assert thread is not None
    assert thread.join(timeout=2.0) is None
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5
    assert worker.turns == [chess.WHITE]
    assert _bestmoves(lines) == [f"bestmove {worker.moves[0]}"]


@pytest.mark.parametrize(
    ("stop", "decline_main", "expected_count", "expected_move"),
    [(False, False, 0, "g1h3"), (True, False, 1, "g8h6"), (False, True, 1, "g1h3")],
    ids=["ponderhit-searched", "stop-prior", "ponderhit-main-prior"],
)
def test_prior_only_counter_counts_only_the_published_result(
    monkeypatch: pytest.MonkeyPatch,
    stop: bool,
    decline_main: bool,
    expected_count: int,
    expected_move: str,
) -> None:
    lines = _capture(monkeypatch)
    worker = _DeclineThenRun(decline_main=decline_main)
    engine = _engine(worker)
    _start_ponder(engine, worker)
    _assert_quiet(lines, seconds=0.05)
    thread = engine._search_thread
    assert thread is not None
    try:
        assert engine.prior_only_roots == 0
        assert not any("prior_only_root=" in line for line in lines)
    finally:
        engine.dispatch(CmdStop() if stop else CmdPonderHit())
        thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert _bestmoves(lines) == [f"bestmove {expected_move}"]
    assert engine.prior_only_roots == expected_count
    assert engine.bestmove_fallback_used == 0
    diagnostics = [line for line in lines if "prior_only_root=" in line]
    assert len(diagnostics) == expected_count
    if diagnostics:
        assert f"move={expected_move} nodes=0" in diagnostics[0]


def test_main_declined_root_is_counted_when_published(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = _capture(monkeypatch)
    worker = _DeclineThenRun()
    engine = _engine(worker)
    engine.dispatch(CmdGo(args=GoArgs(wtime_ms=30_000, btime_ms=30_000)))
    thread = engine._search_thread
    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert _bestmoves(lines) == ["bestmove g1h3"]
    assert engine.prior_only_roots == 1
    assert engine.bestmove_fallback_used == 0
    assert len([line for line in lines if "prior_only_root=" in line]) == 1
