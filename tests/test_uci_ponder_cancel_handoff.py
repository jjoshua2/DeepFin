from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import chess_anti_engine.uci.engine as engine_module
import chess_anti_engine.uci.__main__ as uci_main
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import CmdGo, CmdPonderHit, CmdPosition, GoArgs
from chess_anti_engine.uci.search import SearchResult


def _ponder_engine(monkeypatch):
    monkeypatch.setattr(engine_module, "_JOIN_TIMEOUT_S", 0.01)
    worker = MagicMock()
    worker.advance_root.return_value = True
    ponder_entered = threading.Event()
    release_ponder = threading.Event()
    real_entered = threading.Event()
    release_real = threading.Event()
    calls = []
    real_phase_stopped = []

    def run(board, *, stop_event, **_kwargs):
        calls.append(stop_event)
        if len(calls) == 1:
            ponder_entered.set()
            assert release_ponder.wait(2)
        else:
            real_entered.set()
            real_phase_stopped.append(stop_event.is_set())
            if not stop_event.is_set():
                assert release_real.wait(2)
        move = next(iter(board.legal_moves)).uci()
        return SearchResult(
            bestmove_uci=move, ponder_uci=None, nodes=1, pv=(move,),
            score_cp=0, tbhits=0,
        )

    worker.run = run
    engine = Engine(worker=worker)
    engine.dispatch(CmdPosition(fen=None, moves=("e2e4", "e7e5")))
    engine.dispatch(CmdGo(GoArgs(ponder=True, wtime_ms=10000, btime_ms=10000)))
    assert ponder_entered.wait(1)
    return engine, worker, calls, real_phase_stopped, release_ponder, real_entered, release_real


def test_stop_after_ponderhit_cancels_pending_real_phase(monkeypatch):
    output = []
    monkeypatch.setattr(engine_module, "_println", output.append)
    engine, _worker, calls, real_phase_stopped, release_ponder, real_entered, release_real = _ponder_engine(monkeypatch)
    search_thread = engine._search_thread
    try:
        engine.dispatch(CmdPonderHit())
        assert engine._handle_stop() is False  # bounded stop while fake ponder is held
        release_ponder.set()
        assert real_entered.wait(1), "ponderhit failed to hand off to a real-phase bestmove"
    finally:
        release_ponder.set()
        release_real.set()
        if search_thread is not None:
            search_thread.join(timeout=2)
    assert len(calls) == 2
    assert real_phase_stopped == [True], "real phase received a fresh, uncancelled stop Event"
    assert sum(line.startswith("bestmove ") for line in output) == 1


def test_close_after_ponderhit_does_not_start_another_worker_phase(monkeypatch):
    output = []
    monkeypatch.setattr(engine_module, "_println", output.append)
    engine, worker, calls, _stopped, release_ponder, real_entered, release_real = _ponder_engine(monkeypatch)
    search_thread = engine._search_thread
    close_done = threading.Event()
    close_entered = threading.Event()
    original_close = engine.close

    def close_engine():
        close_entered.set()
        original_close()
        close_done.set()

    engine.dispatch(CmdPonderHit())
    close_thread = threading.Thread(target=close_engine)
    close_thread.start()
    try:
        assert close_entered.wait(1)
        deadline = time.monotonic() + 1
        while not engine._closing and time.monotonic() < deadline:
            time.sleep(0.001)
        assert engine._closing, "close did not mark the generation before the phase handoff"
        release_ponder.set()
        assert not real_entered.wait(0.2), "close started another worker phase after the generation was closed"
        assert close_done.wait(1), "close did not finish after active ponder work exited"
        assert worker.close.called
    finally:
        release_ponder.set()
        release_real.set()
        close_thread.join(timeout=2)
        if search_thread is not None:
            search_thread.join(timeout=2)
    assert len(calls) == 1
    assert not any(line.startswith("bestmove ") for line in output)

class _StopDuringWarmupInput:
    def __init__(self, warmup_entered, raw):
        self.warmup_entered = warmup_entered
        self.raw = raw
        self.delivered = False

    def __iter__(self):
        return self

    def __next__(self):
        assert self.warmup_entered.wait(2), "main never began startup warmup"
        if self.delivered or self.raw is None:
            raise StopIteration
        self.delivered = True
        return self.raw


@pytest.mark.parametrize("raw", ["quit", None], ids=["quit", "eof"])
def test_startup_warmup_owns_worker_until_main_shutdown(monkeypatch, raw):

    warmup_entered = threading.Event()
    release_warmup = threading.Event()
    warmup_done = threading.Event()
    close_overlap = []
    worker = MagicMock()
    worker._chunk_sims = 8

    def run(_board, **_kwargs):
        warmup_entered.set()
        try:
            assert release_warmup.wait(2)
        finally:
            warmup_done.set()
        return None

    worker.run.side_effect = run
    worker.close.side_effect = lambda: close_overlap.append(not warmup_done.is_set())
    engine = Engine(worker=worker)
    monkeypatch.setattr(sys, "argv", ["deepfin", "--checkpoint", "dummy"])
    monkeypatch.setattr(sys, "stdin", _StopDuringWarmupInput(warmup_entered, raw))
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

    done = threading.Event()
    result = []

    def run_main():
        result.append(uci_main.main())
        done.set()

    main_thread = threading.Thread(target=run_main)
    main_thread.start()
    try:
        assert warmup_entered.wait(2)
        assert not done.wait(0.1), "main returned while startup warmup still owned the worker"
        assert not warmup_done.is_set()
        release_warmup.set()
        assert warmup_done.wait(2), "startup warmup did not return"
        assert done.wait(2), "main did not finish after startup warmup was released"
    finally:
        release_warmup.set()
        main_thread.join(timeout=2)

    assert result == [0]
    assert close_overlap == [False], "worker.close overlapped startup warmup"


def test_early_quit_waits_for_builder_and_closes_published_engine(monkeypatch):
    load_entered = threading.Event()
    release_load = threading.Event()
    quit_yielded = threading.Event()
    worker = MagicMock()
    worker._chunk_sims = 8
    engine = Engine(worker=worker)

    class QuitAfterLoadStarts:
        def __init__(self):
            self.sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.sent:
                raise StopIteration
            assert load_entered.wait(2), "background model load did not start"
            self.sent = True
            quit_yielded.set()
            return "quit"

    def load_models(*_args):
        load_entered.set()
        assert release_load.wait(2), "test did not release background model loading"
        return [SimpleNamespace()]

    monkeypatch.setattr(sys, "argv", ["deepfin", "--checkpoint", "dummy"])
    monkeypatch.setattr(sys, "stdin", QuitAfterLoadStarts())
    monkeypatch.setattr(uci_main, "_pick_device", lambda _device: "cpu")
    monkeypatch.setattr(uci_main, "_resolve_multi_gpu_startup", lambda *_args: (False, False, False))
    monkeypatch.setattr(uci_main, "_startup_engine_options", lambda *_args, **_kwargs: SimpleNamespace(
        eval_cache_entries=0, max_batch=1, vl_gather=1,
        use_multi_gpu_pucv=False, search_parallel="pucv",
    ))
    monkeypatch.setattr(uci_main, "_load_models", load_models)
    monkeypatch.setattr(uci_main, "_make_evaluator_factory", lambda *_args, **_kwargs: lambda *_a, **_k: object())
    monkeypatch.setattr(uci_main, "_engine_search_kwargs", lambda _args: {})
    monkeypatch.setattr(uci_main, "_build_engine", lambda **_kwargs: engine)

    done = threading.Event()
    result = []

    def run_main():
        result.append(uci_main.main())
        done.set()

    main_thread = threading.Thread(target=run_main)
    main_thread.start()
    try:
        assert quit_yielded.wait(2)
        assert not done.wait(0.1), "main returned while the background builder still owned startup"
        release_load.set()
        assert done.wait(2), "main did not finish after the builder published its engine"
    finally:
        release_load.set()
        main_thread.join(timeout=2)

    assert result == [0]
    worker.close.assert_called_once()
