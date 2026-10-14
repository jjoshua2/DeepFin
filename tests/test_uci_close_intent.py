from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import chess
import pytest

import chess_anti_engine.uci.__main__ as uci_main
import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.uci.engine import Engine, SearchNotQuiescent
from chess_anti_engine.uci.protocol import CmdGo, CmdPonderHit, CmdPosition, CmdStop, GoArgs
from chess_anti_engine.uci.search import SearchResult


def _worker():
    worker = MagicMock()
    worker._chunk_sims = 8
    worker._pucv_pool = None
    worker._rpg_pool = None
    worker.advance_root.return_value = True
    return worker


def _held_ponder(monkeypatch):
    monkeypatch.setattr(engine_module, "_JOIN_TIMEOUT_S", 0.01)
    output: list[str] = []
    monkeypatch.setattr(engine_module, "_println", output.append)
    worker = _worker()
    entered = threading.Event()
    release = threading.Event()
    calls: list[threading.Event] = []

    def run(board, *, stop_event, **_kwargs):
        calls.append(stop_event)
        entered.set()
        assert release.wait(2), "test did not release the held search"
        move = next(iter(board.legal_moves)).uci()
        return SearchResult(bestmove_uci=move, ponder_uci=None, nodes=1,
                            pv=(move,), score_cp=0, tbhits=0)

    worker.run.side_effect = run
    engine = Engine(worker=worker)
    engine.dispatch(CmdPosition(fen=None, moves=("e2e4", "e7e5")))
    engine.dispatch(CmdGo(GoArgs(ponder=True, wtime_ms=10000, btime_ms=10000)))
    assert entered.wait(1)
    return engine, worker, release, calls, output


def test_direct_close_preserves_custody_and_suppresses_late_ponder_result(monkeypatch):
    engine, worker, release, calls, output = _held_ponder(monkeypatch)
    thread = engine._search_thread
    assert thread is not None
    advances = worker.advance_root.call_count
    try:
        engine.dispatch(CmdPonderHit())
        with pytest.raises(SearchNotQuiescent):
            engine.close()
        assert thread.is_alive()
        worker.close.assert_not_called()
        engine.dispatch(CmdGo(GoArgs(nodes=1)))
        assert len(calls) == 1
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert len(calls) == 1
    assert worker.advance_root.call_count == advances
    assert not any(line.startswith("bestmove ") for line in output)
    engine.close()
    engine.close()
    worker.close.assert_called_once()


def test_ordinary_stop_keeps_one_legal_reply_and_cancelled_handoff(monkeypatch):
    engine, _worker, release, calls, output = _held_ponder(monkeypatch)
    thread = engine._search_thread
    assert thread is not None
    try:
        engine.dispatch(CmdPonderHit())
        engine.dispatch(CmdStop())
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert len(calls) == 1, "cancelled real phase called the worker again"
    replies = [line.split()[1] for line in output if line.startswith("bestmove ")]
    assert len(replies) == 1
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert chess.Move.from_uci(replies[0]) in board.legal_moves
    engine.close()


def test_public_warmup_retains_worker_until_close_can_retry(monkeypatch):
    monkeypatch.setattr(engine_module, "_JOIN_TIMEOUT_S", 0.01)
    worker = _worker()
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def run(_board, **_kwargs):
        entered.set()
        assert release.wait(2), "test did not release warmup"

    worker.run.side_effect = run
    engine = Engine(worker=worker)

    def warm():
        try:
            engine.warmup_search()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=warm)
    thread.start()
    try:
        assert entered.wait(1)
        with pytest.raises(SearchNotQuiescent):
            engine.close()
        worker.close.assert_not_called()
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert not errors
    engine.close()
    worker.close.assert_called_once()
    engine.warmup_search()
    assert worker.run.call_count == 1, "shutdown admitted another warmup"


def test_prelaunch_warmup_error_releases_custody(monkeypatch):
    monkeypatch.setattr(engine_module, "_JOIN_TIMEOUT_S", 0.01)
    worker = _worker()
    worker._chunk_sims = "invalid"
    engine = Engine(worker=worker)
    with pytest.raises(
        ValueError, match=r"invalid literal for int\(\) with base 10: 'invalid'",
    ):
        engine.warmup_search()
    worker.run.assert_not_called()
    engine.close()
    worker.close.assert_called_once()


def test_cli_eof_suppresses_late_result_without_retiring_live_worker(monkeypatch):
    monkeypatch.setattr(engine_module, "_JOIN_TIMEOUT_S", 0.01)
    output: list[str] = []
    monkeypatch.setattr(engine_module, "_println", output.append)
    monkeypatch.setattr(uci_main, "_println", output.append)
    worker = _worker()
    entered = threading.Event()
    release = threading.Event()

    def run(board, **_kwargs):
        entered.set()
        assert release.wait(2), "test did not release the EOF search"
        move = next(iter(board.legal_moves)).uci()
        return SearchResult(bestmove_uci=move, ponder_uci=None, nodes=1,
                            pv=(move,), score_cp=0, tbhits=0)

    worker.run.side_effect = run
    engine = Engine(worker=worker)
    monkeypatch.setattr(engine, "warmup_search", lambda: None)

    class EofAfterSearch:
        sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if not self.sent:
                self.sent = True
                return "go nodes 1"
            assert entered.wait(1)
            raise StopIteration

    monkeypatch.setattr(sys, "argv", ["deepfin", "--checkpoint", "dummy"])
    monkeypatch.setattr(sys, "stdin", EofAfterSearch())
    monkeypatch.setattr(uci_main, "_pick_device", lambda _device: "cpu")
    monkeypatch.setattr(uci_main, "_resolve_multi_gpu_startup", lambda *_args: (False, False, False))
    monkeypatch.setattr(uci_main, "_startup_engine_options", lambda *_args, **_kwargs: SimpleNamespace(
        eval_cache_entries=0, max_batch=1, vl_gather=1,
        use_multi_gpu_pucv=False, search_parallel="pucv",
    ))
    monkeypatch.setattr(uci_main, "_load_models", lambda *_args: [SimpleNamespace()])
    monkeypatch.setattr(uci_main, "_make_evaluator_factory", lambda *_args, **_kwargs: lambda *_a, **_k: object())
    monkeypatch.setattr(uci_main, "_engine_search_kwargs", lambda _args: {})
    monkeypatch.setattr(uci_main, "_build_engine", lambda **_kwargs: engine)
    try:
        assert uci_main.main() == 1, "non-quiescent EOF must report shutdown failure"
        thread = engine._search_thread
        assert thread is not None
        assert thread.is_alive()
        worker.close.assert_not_called()
    finally:
        release.set()
        thread = engine._search_thread
        if thread is not None:
            thread.join(2)
    assert not any(line.startswith("bestmove ") for line in output)
    engine.close()
    worker.close.assert_called_once()
