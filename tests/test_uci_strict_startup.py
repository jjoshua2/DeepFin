"""Startup is strict, in-play search failures are loud, and Gumbel is the default.

(a) A broken compile/warmup must not answer ``readyok``: on main the evaluator
    warmup swallowed its exception, single-GPU ``warmup_search`` logged at
    DEBUG, and every later move was the unsearched root-policy fallback. During
    play the fallback stays (never lose on time) but is counted, announced, and
    triggers one evaluator rebuild per failure streak.
(b) The validated tournament search is classic Gumbel. ``--walkers`` now
    defaults to 1, and the GUI's ``Threads`` option can no longer switch the
    search onto the PUCT walker pool; ``SearchWalkers`` is the operator control.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import chess
import pytest

from chess_anti_engine.uci import __main__ as main
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import CmdGo, GoArgs, parse_command
from chess_anti_engine.uci.search import SearchResult


def _bestmoves(out: str) -> list[str]:
    return [ln.split()[1] for ln in out.splitlines() if ln.startswith("bestmove ")]


def _join(engine: Engine) -> None:
    thread = engine._search_thread
    if thread is not None:
        thread.join(5)
        assert not thread.is_alive()


# --- (a) strict startup -------------------------------------------------------


class _BrokenEvaluator:
    def evaluate_encoded(self, *_a, **_kw):
        raise RuntimeError("cudagraph capture failed")


def test_evaluator_warmup_failure_raises() -> None:
    with pytest.raises(RuntimeError, match="evaluator warmup failed"):
        main._warmup_evaluator(_BrokenEvaluator(), input_extra_features="v2_threats")


class _CappedEvaluator:
    """Refuses more rows than its max_batch, like DirectGPUEvaluator."""

    def __init__(self, max_batch: int) -> None:
        self.max_batch = max_batch
        self.batches: list[int] = []

    def evaluate_encoded(self, x, relations=None):
        del relations
        if len(x) > self.max_batch:
            raise ValueError(f"batch {len(x)} > max {self.max_batch}")
        self.batches.append(len(x))


def test_strict_warmup_never_asks_for_more_than_max_batch() -> None:
    """MaxBatch 64 is a legal setting; strict warmup must not fail on it.

    The Gumbel warmup shape is 128. While warmup was best-effort that shape was
    silently skipped at MaxBatch 64; strict, it would refuse to start.
    """
    ev = _CappedEvaluator(64)
    main._warmup_evaluator(ev, input_extra_features="v2_threats", max_batch=64)
    assert ev.batches == [1, 64]


def test_single_gpu_search_warmup_failure_raises() -> None:
    worker = MagicMock()
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    worker.run.side_effect = RuntimeError("search-path capture failed")
    engine = Engine(worker, options=EngineOptions())
    assert not engine._multi_gpu_path_active()
    with pytest.raises(RuntimeError, match="search-path capture failed"):
        engine.warmup_search()
    worker.reset_tree.assert_called_once()  # tree is still cleaned up


def test_startup_warmup_failure_exits_nonzero_without_readyok(monkeypatch, capsys) -> None:
    worker = MagicMock()
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    worker.run.side_effect = RuntimeError("search-path capture failed")
    engine = Engine(worker)
    monkeypatch.setattr(main, "_load_models", lambda *_a, **_kw: [SimpleNamespace()])
    monkeypatch.setattr(main, "_make_evaluator_factory", lambda *_a, **_kw: lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "_build_engine", lambda **_kw: engine)
    monkeypatch.setattr(main.sys, "argv", ["deepfin", "--checkpoint", "fake.pt", "--device", "cpu"])
    monkeypatch.setattr(main.sys, "stdin", iter(["uci\n", "isready\n", "go nodes 1\n"]))

    assert main.main() == 1

    out = capsys.readouterr()
    assert "readyok" not in out.out.splitlines()
    assert "bestmove" not in out.out
    assert any(
        ln.startswith("info string engine startup failed:") and "search-path capture failed" in ln
        for ln in out.out.splitlines()
    )
    assert "deepfin: engine startup failed: RuntimeError: search-path capture failed" in out.err
    worker.close.assert_called_once()  # the half-built engine was released


# --- (a) loud in-play fallback + one rebuild per streak -----------------------


def _flaky_engine() -> tuple[Engine, MagicMock, list[int]]:
    worker = MagicMock()
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    rebuilds: list[int] = []

    def rebuild(max_batch, _cache=0, /, **_kw):
        rebuilds.append(max_batch)
        return MagicMock(name="rebuilt-evaluator")

    return Engine(worker, options=EngineOptions(), rebuild_evaluator=rebuild), worker, rebuilds


def _ok(board, **_kw):
    move = next(iter(board.legal_moves)).uci()
    return SearchResult(move, None, 8, (move,), 0)


def _go(engine: Engine) -> None:
    engine.dispatch(CmdGo(GoArgs(nodes=8)))
    _join(engine)


def test_consecutive_search_failures_are_counted_and_rebuild_once_per_streak(capsys) -> None:
    engine, worker, rebuilds = _flaky_engine()
    worker.run.side_effect = RuntimeError("CUDA error: illegal memory access")

    _go(engine)
    out = capsys.readouterr().out
    assert len(_bestmoves(out)) == 1  # never lose on time
    assert "WARNING search failed on 1 consecutive move(s)" in out
    assert rebuilds == [1024]
    assert "search-failure recovery: evaluator rebuilt" in out
    # The rebuild is paid after the bestmove is out, not on our clock.
    lines = out.splitlines()
    assert lines.index(next(ln for ln in lines if ln.startswith("bestmove "))) < lines.index(
        next(ln for ln in lines if "rebuilding the evaluator once" in ln)
    )
    worker.set_evaluator.assert_called_once()

    _go(engine)
    out = capsys.readouterr().out
    assert "WARNING search failed on 2 consecutive move(s)" in out
    assert rebuilds == [1024]  # bounded: still one rebuild for this streak

    worker.run.side_effect = _ok
    _go(engine)
    assert "WARNING" not in capsys.readouterr().out

    worker.run.side_effect = RuntimeError("again")
    _go(engine)
    out = capsys.readouterr().out
    assert "WARNING search failed on 1 consecutive move(s)" in out  # new streak
    assert rebuilds == [1024, 1024]


def test_failed_recovery_rebuild_keeps_playing(capsys) -> None:
    worker = MagicMock()
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    worker.run.side_effect = RuntimeError("broken")

    def rebuild(*_a, **_kw):
        raise RuntimeError("out of memory")

    engine = Engine(worker, options=EngineOptions(), rebuild_evaluator=rebuild)
    _go(engine)
    out = capsys.readouterr().out
    assert len(_bestmoves(out)) == 1
    assert "search-failure recovery: evaluator rebuild failed" in out
    worker.set_evaluator.assert_not_called()


# --- (b) Gumbel by default; Threads cannot change the search path -------------


def test_default_launch_selects_one_walker() -> None:
    args = main._build_parser().parse_args(["--checkpoint", "x.pt"])
    assert args.walkers == 1
    options = main._startup_engine_options(
        args, search_parallel="pucv", restore_multi_gpu_pucv=False,
    )
    assert options.search_walkers == 1
    assert EngineOptions().search_walkers == 1


def test_threads_is_accepted_reported_and_does_not_change_the_search(capsys) -> None:
    from tests.test_uci_search_options import _make_engine

    engine = _make_engine(threads=1)
    try:
        assert engine._worker.realized_search_path() == "gumbel"
        engine.dispatch(parse_command("setoption name Threads value 8"))
        out = capsys.readouterr().out
        assert engine._worker.realized_search_path() == "gumbel"
        assert engine._options.threads == 8
        notes = [ln for ln in out.splitlines() if ln.startswith("info string Threads set to 8")]
        assert len(notes) == 1
        assert "NO EFFECT on the search" in notes[0]
        assert "path=gumbel" in notes[0]
        # The operator control still works, and reaches the worker.
        engine.dispatch(parse_command("setoption name SearchWalkers value 2"))
        assert engine._worker.realized_search_path() == "walker"
        engine.dispatch(parse_command("setoption name SearchWalkers value 1"))
        assert engine._worker.realized_search_path() == "gumbel"
    finally:
        engine.close()


def test_pipe_default_launch_is_gumbel_and_threads_cannot_move_it(tmp_path: Path) -> None:
    from tests.test_uci_smoke import _LineReader, _make_tiny_checkpoint, _send

    ckpt = _make_tiny_checkpoint(tmp_path)
    env = {**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.Popen(  # no --walkers: the DEFAULT launch is under test
        [sys.executable, "-u", "-m", "chess_anti_engine.uci", "--checkpoint", str(ckpt),
         "--device", "cpu", "--no-compile"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    reader = _LineReader(proc)
    try:
        _send(proc, "uci")
        lines = reader.read_until("uciok")
        assert any(ln.startswith("option name Threads type spin default 1 ") for ln in lines)
        assert any(ln.startswith("option name SearchWalkers type spin default 1 ") for ln in lines)
        _send(proc, "isready")
        reader.read_until("readyok")
        _send(proc, "searchconfig")
        _send(proc, "isready")
        lines = reader.read_until("readyok")
        assert any(ln.startswith("info string searchconfig path=gumbel ") for ln in lines)
        _send(proc, "setoption name Threads value 8")
        _send(proc, "searchconfig")
        _send(proc, "isready")
        lines = reader.read_until("readyok")
        assert any("Threads set to 8 — NO EFFECT on the search" in ln for ln in lines)
        assert any(ln.startswith("info string searchconfig path=gumbel ") for ln in lines)
        _send(proc, "position startpos")
        _send(proc, "go nodes 16")
        lines = reader.read_until("bestmove", timeout_s=60.0)
        move = next(ln for ln in lines if ln.startswith("bestmove ")).split()[1]
        assert chess.Move.from_uci(move) in chess.Board().legal_moves
        _send(proc, "quit")
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)

