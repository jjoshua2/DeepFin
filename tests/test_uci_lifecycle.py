"""Commands must not reuse a live worker; ponderhit starts the real clock."""
from __future__ import annotations

import threading
from typing import Any, cast
from unittest.mock import MagicMock

import chess
import pytest

from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import (
    CmdGo, CmdPonderHit, CmdPosition, CmdQuit, CmdSetOption, CmdStop,
    CmdUciNewGame, GoArgs,
)
from chess_anti_engine.uci.search import SearchResult


def _result(board: chess.Board) -> SearchResult:
    move = next(iter(board.legal_moves)).uci()
    return SearchResult(move, None, 1, (move,), 0)


@pytest.fixture
def held_engine(monkeypatch):
    monkeypatch.setattr("chess_anti_engine.uci.engine._JOIN_TIMEOUT_S", 0.005)
    entered, release = threading.Event(), threading.Event()
    worker = MagicMock()
    threads: list[threading.Thread] = []

    def run(board, **_kwargs):
        threads.append(threading.current_thread())
        entered.set()
        assert release.wait(2), "test did not release its held worker"
        return _result(board)

    worker.run.side_effect = run
    worker.advance_root.return_value = True
    engine = Engine(worker, options=EngineOptions(hash_mb=1024))
    engine.dispatch(CmdGo(GoArgs(nodes=1)))
    assert entered.wait(1)
    worker.reset_tree.reset_mock()
    worker.advance_root.reset_mock()
    worker.set_max_tree_mb.reset_mock()
    try:
        yield engine, worker, release, threads
    finally:
        release.set()
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()


@pytest.mark.parametrize("command", ["position", "newgame", "setoption", "go"])
def test_timeout_rejects_mutation_and_second_run_until_quiescent(held_engine, command, capsys):
    engine, worker, release, threads = held_engine
    cmd = {
        "position": CmdPosition(None, ("e2e4",)),
        "newgame": CmdUciNewGame(),
        "setoption": CmdSetOption("Hash", "2048"),
        "go": CmdGo(GoArgs(nodes=1)),
    }[command]
    old_thread, old_stop = engine._search_thread, engine._stop_event
    old_fen, old_gen = engine._board.fen(), engine._search_gen
    engine.dispatch(cmd)
    assert engine._search_thread is old_thread
    assert engine._stop_event is old_stop
    assert engine._board.fen() == old_fen
    assert engine._search_gen == old_gen
    assert worker.run.call_count == 1
    worker.reset_tree.assert_not_called()
    worker.advance_root.assert_not_called()
    worker.set_max_tree_mb.assert_not_called()
    assert engine._options.hash_mb == 1024
    assert "rejected" in capsys.readouterr().out
    release.set()
    threads[0].join(2)
    engine.dispatch(cmd)
    if command == "position":
        assert engine._board.peek().uci() == "e2e4"
    elif command == "newgame":
        worker.reset_tree.assert_called_once()
    elif command == "setoption":
        assert engine._options.hash_mb == 2048
    else:
        thread = engine._search_thread
        assert thread is not None
        thread.join(2)
        assert not thread.is_alive()
        assert worker.run.call_count == 2


def test_close_refuses_live_worker_and_quit_is_terminal(held_engine, capsys):
    engine, worker, release, threads = held_engine
    engine.dispatch(CmdQuit())
    assert engine.quit_requested
    with pytest.raises(TimeoutError):
        engine.close()
    worker.close.assert_not_called()
    engine.dispatch(CmdGo(GoArgs(nodes=1)))
    assert worker.run.call_count == 1
    release.set()
    threads[0].join(2)
    assert "bestmove " not in capsys.readouterr().out
    engine.close()
    engine.close()
    worker.close.assert_called_once()


def _ponder_engine(monkeypatch, *, advance_ms=0):
    monkeypatch.setattr("chess_anti_engine.uci.engine._JOIN_TIMEOUT_S", 0.005)
    now = [100.0]
    monkeypatch.setattr("chess_anti_engine.uci.engine.time.monotonic", lambda: now[0])
    entered, release = threading.Event(), threading.Event()
    worker = MagicMock()
    calls: list[dict[str, Any]] = []

    def run(board, **kwargs):
        calls.append({"board": board.copy(), **kwargs})
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        return _result(board)

    def advance(_board, _moves):
        now[0] += advance_ms / 1000
        return True

    worker.run.side_effect = run
    worker.advance_root.side_effect = advance
    worker.root_policy_bestmove.return_value = None
    engine = Engine(worker, options=EngineOptions(move_overhead_ms=0))
    engine.dispatch(CmdPosition(None, ("e2e4", "e7e5")))
    engine.dispatch(CmdGo(GoArgs(ponder=True, movetime_ms=100)))
    assert entered.wait(1)
    assert engine._pending_real_limits is not None
    worker.advance_root.reset_mock()
    return engine, worker, now, release, calls


def _join(engine, release):
    release.set()
    thread = engine._search_thread
    assert thread is not None
    thread.join(2)
    assert not thread.is_alive()


def test_ponderhit_charges_drain_and_advance_time_from_first_hit(monkeypatch):
    engine, _worker, now, release, calls = _ponder_engine(monkeypatch, advance_ms=30)
    try:
        engine.dispatch(CmdPonderHit())
        now[0] += 0.01
        engine.dispatch(CmdPonderHit())  # Repeated hit must not restart the clock.
        now[0] += 0.03
    finally:
        _join(engine, release)
    assert len(calls) == 2
    deadline = calls[1]["deadline"]
    assert 28 <= deadline.remaining_ms() <= 30
    assert 69 <= deadline.elapsed_ms() <= 71
    assert not calls[1]["stop_event"].is_set()
    assert calls[1]["board"].turn == chess.WHITE


@pytest.mark.parametrize("reason", ["stop", "deadline"])
def test_ponder_transition_does_not_start_real_search_when_cancelled_or_expired(monkeypatch, reason, capsys):
    engine, worker, now, release, calls = _ponder_engine(monkeypatch)
    try:
        engine.dispatch(CmdPonderHit())
        if reason == "stop":
            engine.dispatch(CmdStop())
        else:
            now[0] += 0.15
    finally:
        _join(engine, release)
    assert len(calls) == 1
    worker.advance_root.assert_not_called()
    output = capsys.readouterr().out
    real_board = chess.Board()
    real_board.push_uci("e2e4")
    real_board.push_uci("e7e5")
    bestmoves = [line.split()[1] for line in output.splitlines() if line.startswith("bestmove ")]
    assert len(bestmoves) == 1
    assert chess.Move.from_uci(bestmoves[0]) in real_board.legal_moves
    assert "ponder_transition_" in output


def test_stop_during_root_advance_survives_phase_event_replacement(monkeypatch):
    engine, worker, _now, release, calls = _ponder_engine(monkeypatch)
    advancing, finish_advance = threading.Event(), threading.Event()

    def advance(_board, _moves):
        advancing.set()
        assert finish_advance.wait(2)
        return True

    worker.advance_root.side_effect = advance
    try:
        engine.dispatch(CmdPonderHit())
        release.set()
        assert advancing.wait(1)
        engine.dispatch(CmdStop())
    finally:
        finish_advance.set()
        _join(engine, release)
    assert len(calls) == 1


def test_root_advance_can_exhaust_ponderhit_budget(monkeypatch, capsys):
    engine, worker, _now, release, calls = _ponder_engine(monkeypatch, advance_ms=150)
    engine.dispatch(CmdPonderHit())
    _join(engine, release)
    assert len(calls) == 1
    worker.advance_root.assert_called_once()
    assert "ponder_transition_deadline" in capsys.readouterr().out


def test_main_reports_shutdown_failure_without_claiming_success(monkeypatch, capsys):
    from types import SimpleNamespace
    from chess_anti_engine.uci import __main__ as main
    from chess_anti_engine.uci.engine import SearchNotQuiescent

    engine = Engine(MagicMock())
    monkeypatch.setattr(engine, "warmup_search", lambda: None)
    close = MagicMock(side_effect=SearchNotQuiescent("held worker"))
    monkeypatch.setattr(engine, "close", close)
    monkeypatch.setattr(main, "_load_models", lambda *_a, **_kw: [SimpleNamespace()])
    monkeypatch.setattr(main, "_make_evaluator_factory", lambda *_a, **_kw: lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "_build_engine", lambda **_kw: engine)
    monkeypatch.setattr(main.sys, "argv", ["deepfin", "--checkpoint", "fake.pt", "--device", "cpu"])
    monkeypatch.setattr(main.sys, "stdin", iter(["isready\n", "quit\n"]))
    assert main.main() == 1
    assert engine.quit_requested
    assert "engine shutdown failed" in capsys.readouterr().out
    close.assert_called_once()


@pytest.mark.parametrize("warmup_failure", [False, True])
def test_quit_during_actual_warmup_keeps_builder_ownership_until_drain(monkeypatch, warmup_failure):
    from types import SimpleNamespace
    from chess_anti_engine.uci import __main__ as main

    entered, release, built, closed = [threading.Event() for _ in range(4)]
    worker = MagicMock()
    worker._chunk_sims = 1
    worker._pucv_pool = worker._rpg_pool = None
    closed_while_running = []
    build_threads: list[threading.Thread] = []

    def run(board, **_kwargs):
        build_threads.append(threading.current_thread())
        entered.set()
        assert release.wait(2)
        return _result(board)

    def close():
        closed_while_running.append(not release.is_set())
        closed.set()

    worker.run.side_effect, worker.close.side_effect = run, close
    engine = Engine(worker)
    original_warmup = engine.warmup_search

    def warmup():
        try:
            original_warmup()
            if warmup_failure:
                raise RuntimeError("warmup failed after drain")
        finally:
            built.set()

    def stdin():
        assert entered.wait(2)
        yield "quit\n"

    monkeypatch.setattr(engine, "warmup_search", warmup)
    monkeypatch.setattr(main, "_load_models", lambda *_a, **_kw: [SimpleNamespace()])
    monkeypatch.setattr(main, "_make_evaluator_factory", lambda *_a, **_kw: lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "_build_engine", lambda **_kw: engine)
    monkeypatch.setattr(main.sys, "argv", ["deepfin", "--checkpoint", "fake.pt", "--device", "cpu"])
    monkeypatch.setattr(main.sys, "stdin", stdin())
    try:
        assert main.main() == 0
        worker.close.assert_not_called()
    finally:
        release.set()
        assert built.wait(2)
        assert closed.wait(2)
        build_threads[0].join(2)
        assert not build_threads[0].is_alive()
    assert closed_while_running == [False]
    assert engine._closed
    worker.close.assert_called_once()


def test_repeated_ponderhit_cannot_signal_the_replacement_cancel_event(monkeypatch):
    engine, worker, _now, release, calls = _ponder_engine(monkeypatch)
    advancing, finish_advance = threading.Event(), threading.Event()

    def advance(_board, _moves):
        advancing.set()
        assert finish_advance.wait(2)
        return True

    engine.dispatch(CmdPonderHit())
    worker.advance_root.side_effect = advance
    original = engine._ponderhit_event

    class HandoffEvent:
        def is_set(self):
            return original.is_set()

        def clear(self):
            original.clear()

        def set(self):
            original.set()
            release.set()
            assert advancing.wait(2)

    engine._ponderhit_event = cast(threading.Event, cast(object, HandoffEvent()))
    try:
        engine.dispatch(CmdPonderHit())
        assert not engine._cancel_event.is_set()
    finally:
        finish_advance.set()
        _join(engine, release)
    assert len(calls) == 2


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("fen", [None, "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"])
def test_bare_position_ponderhit_releases_and_searches_same_real_board(monkeypatch, fen, held):
    lines = []
    monkeypatch.setattr("chess_anti_engine.uci.engine._println", lines.append)
    entered = threading.Event()
    calls = []
    worker = MagicMock()
    worker.advance_root.return_value = True

    def run(board, **kwargs):
        calls.append({"board": board.copy(), **kwargs})
        if len(calls) == 1:
            entered.set()
            if held:
                assert kwargs["stop_event"].wait(2)
        return _result(board)

    worker.run.side_effect = run
    engine = Engine(worker, options=EngineOptions(move_overhead_ms=0))
    engine.dispatch(CmdPosition(fen, ()))
    engine.dispatch(CmdGo(GoArgs(ponder=True, movetime_ms=5000)))
    assert entered.wait(1)
    thread = engine._search_thread
    assert thread is not None
    try:
        engine.dispatch(CmdPonderHit())
        thread.join(0.2)
        assert not thread.is_alive()
        assert len(calls) == 2
        expected = chess.Board(fen) if fen is not None else chess.Board()
        assert calls[1]["board"].fen() == expected.fen()
        assert calls[0]["deadline"].remaining_ms() is None
        assert calls[1]["deadline"].remaining_ms() is not None
        assert not calls[1]["stop_event"].is_set()
        assert len([v for v in lines if v.startswith("bestmove ")]) == 1
    finally:
        engine.dispatch(CmdStop())
        thread.join(2)
        assert not thread.is_alive()
