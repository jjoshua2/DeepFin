"""Follow-ups from the final review of #1263: a parked search stays silent and
cannot touch the live search's ponder bookkeeping; reporting-only options apply
while it lives; `quit` says why it exits non-zero.
"""
from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import chess

import chess_anti_engine.uci.__main__ as uci_main
import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import CmdGo, GoArgs, parse_command
from chess_anti_engine.uci.search import SearchResult


def _bestmoves(out: str) -> list[str]:
    return [ln.split()[1] for ln in out.splitlines() if ln.startswith("bestmove ")]


def _worker() -> MagicMock:
    worker = MagicMock()
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    worker.realized_search_path.return_value = "gumbel"
    return worker


def _ok(board: chess.Board) -> SearchResult:
    move = next(iter(board.legal_moves)).uci()
    return SearchResult(move, None, 8, (move,), 0)


def _info(info_cb, marker: str) -> None:
    info_cb(nodes=7, elapsed_ms=1, score_cp=0, pv=(marker,), tbhits=0,
            score_mate=None, multipv=None, wdl=None, string=f"marker {marker}")


def _park_a_search(monkeypatch, worker: MagicMock, first_go: str, after_release=None):
    """Start ``first_go`` on a search that ignores stop, then `stop` parks it."""
    monkeypatch.setattr(engine_module, "_GO_WAIT_NO_CLOCK_S", 0.05)
    release, entered = threading.Event(), threading.Event()

    def hang(board, **kw):
        entered.set()
        release.wait(10)
        if after_release is not None:
            after_release(kw)
        return _ok(board)

    worker.run.side_effect = hang
    engine = Engine(worker, options=EngineOptions(ponder=True))
    engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
    engine.dispatch(parse_command(first_go))
    assert entered.wait(5)
    engine.dispatch(parse_command("stop"))  # parked, answered from the fallback
    (orphan,) = engine._orphan_threads
    return engine, release, orphan


def test_a_parked_searchs_info_lines_never_reach_stdout(monkeypatch, capsys) -> None:
    worker = _worker()
    engine, release, orphan = _park_a_search(
        monkeypatch, worker, "go infinite",
        after_release=lambda kw: _info(kw["info_cb"], "late"),
    )
    capsys.readouterr()
    release.set()
    orphan.join(5)
    assert not orphan.is_alive()
    assert "marker late" not in capsys.readouterr().out  # silenced
    # A live search's info lines still print.
    worker.run.side_effect = lambda board, **kw: (_info(kw["info_cb"], "live"), _ok(board))[1]
    engine.dispatch(CmdGo(GoArgs(nodes=8)))
    thread = engine._search_thread
    assert thread is not None
    thread.join(5)
    out = capsys.readouterr().out
    assert "marker live" in out
    assert len(_bestmoves(out)) == 1


def test_a_parked_ponder_search_does_not_consume_the_live_ponder_flag(monkeypatch, capsys) -> None:
    engine, release, orphan = _park_a_search(
        monkeypatch, _worker(), "go ponder wtime 5000 btime 5000",
    )
    # As if the live search's ponder phase had fallen back and is pending.
    engine._ponder_fallback_pending = True
    release.set()
    orphan.join(5)
    assert not orphan.is_alive()
    assert engine._ponder_fallback_pending is True  # still the live search's
    assert "WARNING search failed" not in capsys.readouterr().out


def test_reporting_options_apply_with_an_orphan_alive_and_hash_does_not(monkeypatch, capsys) -> None:
    worker = _worker()
    engine, release, orphan = _park_a_search(monkeypatch, worker, "go infinite")
    hash_before = engine._options.hash_mb
    hash_calls = worker.set_max_tree_mb.call_count  # the startup value
    capsys.readouterr()
    engine.dispatch(parse_command("setoption name MultiPV value 3"))
    engine.dispatch(parse_command("setoption name UCI_ShowWDL value true"))
    engine.dispatch(parse_command("setoption name Hash value 4096"))
    out = capsys.readouterr().out
    assert engine._options.multi_pv == 3
    worker.set_multi_pv.assert_called_with(3)
    assert engine._options.show_wdl is True
    worker.set_show_wdl.assert_called_with(True)
    assert out.count("applied while a stuck search still holds the worker") == 2
    assert engine._options.hash_mb == hash_before  # Hash: rejected
    assert worker.set_max_tree_mb.call_count == hash_calls
    assert "command rejected (CmdSetOption)" in out
    release.set()
    orphan.join(5)


def test_quit_with_a_stuck_search_says_why_and_with_what_status(monkeypatch) -> None:
    monkeypatch.setattr(engine_module, "_JOIN_TIMEOUT_S", 0.01)
    output: list[str] = []
    monkeypatch.setattr(engine_module, "_println", output.append)
    monkeypatch.setattr(uci_main, "_println", output.append)
    err: list[str] = []
    real_print = print

    def capture_print(*args, file=None, **kwargs):
        if file is sys.stderr:
            err.append(" ".join(str(a) for a in args))
        else:
            real_print(*args, file=file, **kwargs)

    monkeypatch.setattr(uci_main, "print", capture_print, raising=False)
    worker = _worker()
    entered, release = threading.Event(), threading.Event()

    def run(board, **_kw):
        entered.set()
        assert release.wait(5)
        return _ok(board)

    worker.run.side_effect = run
    engine = Engine(worker=worker)
    monkeypatch.setattr(engine, "warmup_search", lambda: None)

    class QuitAfterSearch:
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
    monkeypatch.setattr(sys, "stdin", QuitAfterSearch())
    monkeypatch.setattr(uci_main, "_pick_device", lambda _device: "cpu")
    monkeypatch.setattr(uci_main, "_resolve_multi_gpu_startup", lambda *_a: (False, False, False))
    monkeypatch.setattr(uci_main, "_startup_engine_options", lambda *_a, **_k: SimpleNamespace(
        eval_cache_entries=0, max_batch=1, vl_gather=1, use_multi_gpu_pucv=False,
        search_parallel="pucv", search_walkers=1, leaf_gather=1,
    ))
    monkeypatch.setattr(uci_main, "_load_models", lambda *_a: [SimpleNamespace()])
    monkeypatch.setattr(uci_main, "_make_evaluator_factory", lambda *_a, **_k: lambda *_x, **_y: object())
    monkeypatch.setattr(uci_main, "_engine_search_kwargs", lambda _args: {})
    monkeypatch.setattr(uci_main, "_build_engine", lambda **_k: engine)
    exits: list[int] = []
    monkeypatch.setattr(uci_main, "_hard_exit", exits.append)
    try:
        assert uci_main.main() == 1
        assert exits == [1]
        assert output[-1] == (
            "info string quit: a search that ignored stop, or a fallback eval stuck in "
            "the evaluator, is still running; exiting with status 1 without waiting for it"
        )
        assert err[-1].endswith("exiting with status 1 without waiting for it")
        assert err[-1].startswith("deepfin: a search that ignored stop")
    finally:
        release.set()
        for t in list(engine._orphan_threads):
            t.join(5)
