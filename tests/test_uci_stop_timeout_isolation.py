from __future__ import annotations

import threading
from unittest.mock import MagicMock

import chess

import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import CmdGo, CmdPosition, CmdSetOption, CmdUciNewGame, GoArgs
from chess_anti_engine.uci.search import SearchResult


def test_second_go_does_not_overlap_worker_when_stop_times_out(monkeypatch) -> None:
    monkeypatch.setattr(engine_module, "_JOIN_TIMEOUT_S", 0.01)
    worker = MagicMock()
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    running = 0
    peak_running = 0
    calls = 0

    def run(board: chess.Board, **_kwargs) -> SearchResult:
        nonlocal calls, running, peak_running
        with lock:
            calls += 1
            call = calls
            running += 1
            peak_running = max(peak_running, running)
        (first_entered if call == 1 else second_entered).set()
        release.wait(timeout=2)
        move = next(iter(board.legal_moves)).uci()
        with lock:
            running -= 1
        return SearchResult(
            bestmove_uci=move, ponder_uci=None, nodes=0, pv=(), score_cp=0, tbhits=0,
        )

    worker.run = run
    engine = Engine(worker=worker)
    engine.dispatch(CmdGo(GoArgs(infinite=True)))
    assert first_entered.wait(timeout=1)
    first_thread = engine._search_thread
    assert first_thread is not None

    engine.dispatch(CmdGo(GoArgs(depth=1)))
    second_thread = engine._search_thread
    original_board = engine._board.fen()
    tree_calls = (worker.reset_tree.call_count, worker.advance_root.call_count)
    engine.dispatch(CmdPosition(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"))
    engine.dispatch(CmdUciNewGame())
    old_overhead = engine._options.move_overhead_ms
    engine.dispatch(CmdSetOption(name="MoveOverheadMs", value="321"))
    engine._handle_searchconfig()
    close_done = threading.Event()

    def close_engine() -> None:
        engine.close()
        close_done.set()

    close_thread = threading.Thread(target=close_engine)
    close_thread.start()
    assert not close_done.wait(timeout=0.05)
    assert not worker.close.called
    assert engine._board.fen() == original_board
    assert engine._options.move_overhead_ms == old_overhead
    assert (worker.reset_tree.call_count, worker.advance_root.call_count) == tree_calls
    assert not worker.close.called
    try:
        assert not second_entered.wait(timeout=0.05), "second search overlapped the first"
    finally:
        release.set()
        first_thread.join(timeout=2)
        close_thread.join(timeout=2)
        if second_thread is not None and second_thread is not first_thread:
            second_thread.join(timeout=2)

    assert close_done.is_set()
    assert worker.close.called
    assert peak_running == 1
    assert calls == 1
