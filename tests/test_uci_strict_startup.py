"""Startup is strict, in-play search failures are loud, and Gumbel is the default.

(a) A broken compile/warmup must not answer ``readyok``: on main the evaluator
    warmup swallowed its exception, single-GPU ``warmup_search`` logged at
    DEBUG, and every later move was the unsearched root-policy fallback. During
    play the fallback stays (never lose on time) but is counted, announced, and
    triggers one BACKGROUND evaluator rebuild per failure streak (never on the
    search barrier); a failed recovery or 3 fallbacks in a row mark the engine
    unhealthy, and the next idle isready / ucinewgame reruns the strict check
    and exits non-zero if it fails.
(b) The validated tournament search is classic Gumbel. ``--walkers`` now
    defaults to 1, and the GUI's ``Threads`` option can no longer switch the
    search onto the PUCT walker pool; ``SearchWalkers`` is the operator control.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import chess
import pytest

from chess_anti_engine.mcts.solved_root import SolvedRootWitnessUnavailable
from chess_anti_engine.uci import __main__ as main
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import CmdGo, GoArgs, parse_command
from chess_anti_engine.uci.search import SearchResult
from chess_anti_engine.uci.time_manager import SearchLimits


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


# --- (a) loud in-play fallback; background recovery; unhealthy re-check -------


def _flaky_engine(rebuild=None) -> tuple[Engine, MagicMock, list[object]]:
    worker = MagicMock()
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    worker.fresh_root_policy_bestmove.return_value = None
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    rebuilt: list[object] = []

    def default_rebuild(_max_batch, _cache=0, /, **_kw):
        ev = MagicMock(name=f"rebuilt-evaluator-{len(rebuilt)}")
        rebuilt.append(ev)
        return ev

    engine = Engine(worker, options=EngineOptions(), rebuild_evaluator=rebuild or default_rebuild)
    return engine, worker, rebuilt


def _ok(board, **_kw):
    move = next(iter(board.legal_moves)).uci()
    return SearchResult(move, None, 8, (move,), 0)


def _go(engine: Engine) -> None:
    engine.dispatch(CmdGo(GoArgs(nodes=8)))
    _join(engine)


def _join_recovery(engine: Engine) -> None:
    thread = engine._recovery_thread
    if thread is not None:
        thread.join(5)
        assert not thread.is_alive()


def test_recovery_builds_in_background_and_goes_in_at_the_next_go(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = RuntimeError("CUDA error: illegal memory access")

    _go(engine)
    _join_recovery(engine)
    out = capsys.readouterr().out
    assert len(_bestmoves(out)) == 1  # never lose on time
    assert "WARNING search failed on 1 consecutive move(s)" in out
    assert "(attempt 1/2)" in out
    assert len(rebuilt) == 1
    # The search thread only STARTED the rebuild; nothing touched the worker.
    worker.set_evaluator.assert_not_called()
    assert engine._staged_evaluator is rebuilt[0]

    calls: list[str] = []
    worker.set_evaluator.side_effect = lambda *_a, **_kw: calls.append("swap")
    worker.run.side_effect = lambda board, **kw: (calls.append("search"), _ok(board))[1]
    _go(engine)
    worker.set_evaluator.assert_called_once_with(rebuilt[0], keep_pools=False)
    assert calls == ["swap", "search"]  # swapped in BEFORE the search
    out = capsys.readouterr().out
    assert "rebuilt evaluator installed" in out
    assert engine._staged_evaluator is None


def test_one_rebuild_per_streak_bounded_per_session_then_unhealthy(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = RuntimeError("broken")
    _go(engine)
    _join_recovery(engine)
    _go(engine)  # adopts the rebuilt evaluator, fails again: same streak
    _join_recovery(engine)
    assert len(rebuilt) == 1
    assert "WARNING search failed on 2 consecutive move(s)" in capsys.readouterr().out

    worker.run.side_effect = _ok
    _go(engine)  # streak reset
    worker.run.side_effect = RuntimeError("again")
    _go(engine)
    _join_recovery(engine)
    out = capsys.readouterr().out
    assert "WARNING search failed on 1 consecutive move(s)" in out
    assert "(attempt 2/2)" in out
    assert len(rebuilt) == 2

    worker.run.side_effect = _ok
    _go(engine)
    worker.run.side_effect = RuntimeError("third streak")
    _go(engine)
    _join_recovery(engine)
    out = capsys.readouterr().out
    assert len(rebuilt) == 2  # session budget spent: no third rebuild
    assert len(_bestmoves(out)) == 2  # both moves answered
    assert "WARNING engine unhealthy: search failed again after 2 evaluator rebuilds" in out
    assert engine._unhealthy is not None


def test_slow_rebuild_never_delays_position_or_answers_a_stale_board(monkeypatch, capsys) -> None:
    """The reviewer's sc_stale/sc_slow: on #1263's first cut the rebuild ran on
    the search thread, `position` timed out on the barrier and was rejected, and
    the next `go` answered the PREVIOUS board."""
    import chess_anti_engine.uci.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_JOIN_TIMEOUT_S", 0.2)
    release = threading.Event()
    rebuilt: list[object] = []

    def slow_rebuild(*_a, **_kw):
        release.wait(10)
        ev = MagicMock(name="slow-rebuilt")
        rebuilt.append(ev)
        return ev

    engine, worker, _ = _flaky_engine(slow_rebuild)
    worker.run.side_effect = RuntimeError("broken")
    engine.dispatch(parse_command("position startpos moves e2e4"))
    _go(engine)
    assert engine._recovery_thread is not None
    assert engine._recovery_thread.is_alive()

    searched: list[str] = []

    def record(board, **_kw):
        searched.append(board.fen())
        return _ok(board)

    worker.run.side_effect = record
    capsys.readouterr()
    engine.dispatch(parse_command("position startpos moves e2e4 e7e5 g1f3"))
    _go(engine)
    out = capsys.readouterr().out
    assert "rejected" not in out, out
    board = chess.Board()
    for mv in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(mv)
    assert searched == [board.fen()]  # the LATEST position, searched
    (move,) = _bestmoves(out)
    assert chess.Move.from_uci(move) in board.legal_moves
    worker.set_evaluator.assert_not_called()  # not built yet: current evaluator

    release.set()
    _join_recovery(engine)
    _go(engine)
    worker.set_evaluator.assert_called_once_with(rebuilt[0], keep_pools=False)


def test_never_two_rebuilds_at_once(capsys) -> None:
    release = threading.Event()
    started: list[int] = []

    def slow_rebuild(*_a, **_kw):
        started.append(1)
        release.wait(10)
        return MagicMock()

    engine, worker, _ = _flaky_engine(slow_rebuild)
    worker.run.side_effect = RuntimeError("broken")
    _go(engine)
    worker.run.side_effect = _ok
    _go(engine)  # streak over
    worker.run.side_effect = RuntimeError("new streak")
    _go(engine)  # owes a rebuild, but one is still building
    assert len(started) == 1
    assert engine._recovery_rebuilds == 1
    release.set()
    _join_recovery(engine)
    capsys.readouterr()


def test_setoption_retires_a_staged_evaluator_built_for_the_old_options(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = RuntimeError("broken")
    _go(engine)
    _join_recovery(engine)
    assert engine._staged_evaluator is rebuilt[0]
    engine.dispatch(parse_command("setoption name MoveOverhead value 50"))
    assert engine._staged_evaluator is None
    staged = rebuilt[0]
    assert isinstance(staged, MagicMock)
    staged.close.assert_called_once()
    assert "staged evaluator discarded (setoption)" in capsys.readouterr().out
    worker.run.side_effect = _ok
    _go(engine)
    worker.set_evaluator.assert_not_called()


def test_build_that_outlives_a_setoption_is_closed_not_staged(monkeypatch, capsys) -> None:
    import chess_anti_engine.uci.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_JOIN_TIMEOUT_S", 0.1)
    release = threading.Event()
    built: list[MagicMock] = []

    def slow_rebuild(*_a, **_kw):
        release.wait(10)
        built.append(MagicMock(name="obsolete"))
        return built[-1]

    engine, worker, _ = _flaky_engine(slow_rebuild)
    worker.run.side_effect = RuntimeError("broken")
    _go(engine)
    engine.dispatch(parse_command("setoption name MoveOverhead value 50"))  # build still running
    release.set()
    _join_recovery(engine)
    assert engine._staged_evaluator is None
    built[0].close.assert_called_once()
    capsys.readouterr()


def test_failed_recovery_marks_unhealthy_keeps_playing_and_isready_clears_it(capsys) -> None:
    def refuse(*_a, **_kw):
        raise RuntimeError("out of memory")

    engine, worker, _ = _flaky_engine(refuse)
    worker.run.side_effect = RuntimeError("broken")
    _go(engine)
    _join_recovery(engine)
    out = capsys.readouterr().out
    assert "search-failure recovery: evaluator rebuild failed" in out
    assert "WARNING engine unhealthy: the evaluator rebuild after a search failure failed" in out
    worker.set_evaluator.assert_not_called()

    worker.run.side_effect = RuntimeError("still broken")
    _go(engine)
    assert len(_bestmoves(capsys.readouterr().out)) == 1  # unhealthy still answers

    worker.run.reset_mock()
    worker.run.side_effect = _ok  # the fault was transient
    engine.dispatch(parse_command("isready"))
    out = capsys.readouterr().out
    assert worker.run.call_args.args[0] == chess.Board()  # the strict startup warmup
    assert "health check passed" in out
    assert out.splitlines()[-1] == "readyok"
    assert engine._unhealthy is None
    assert engine.exit_code == 0


def test_unhealthy_engine_reruns_the_check_at_ucinewgame(capsys) -> None:
    engine, worker, _ = _flaky_engine()
    engine._mark_unhealthy("test")
    worker.run.side_effect = _ok
    engine.dispatch(parse_command("ucinewgame"))
    assert "health check passed" in capsys.readouterr().out
    assert engine._unhealthy is None


def test_three_fallbacks_in_a_row_mark_unhealthy(capsys) -> None:
    engine, worker, _ = _flaky_engine()
    worker.run.side_effect = RuntimeError("broken")
    for _ in range(2):
        _go(engine)
        _join_recovery(engine)
    assert engine._unhealthy is None
    _go(engine)
    assert "WARNING engine unhealthy: 3 consecutive moves were fallbacks" in capsys.readouterr().out


def test_failed_health_recheck_exits_nonzero_before_readyok(monkeypatch, capsys) -> None:
    worker = MagicMock()
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    worker.fresh_root_policy_bestmove.return_value = None
    runs = [0]

    def run(board, **_kw):
        runs[0] += 1
        if runs[0] == 1:
            return _ok(board)  # startup warmup passes
        raise RuntimeError("device lost")

    worker.run.side_effect = run

    def refuse(*_a, **_kw):
        raise RuntimeError("out of memory")

    engine = Engine(worker, options=EngineOptions(), rebuild_evaluator=refuse)
    monkeypatch.setattr(main, "_load_models", lambda *_a, **_kw: [SimpleNamespace()])
    monkeypatch.setattr(main, "_make_evaluator_factory", lambda *_a, **_kw: lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "_build_engine", lambda **_kw: engine)
    monkeypatch.setattr(main.sys, "argv", ["deepfin", "--checkpoint", "fake.pt", "--device", "cpu"])

    def stdin():
        yield from ("uci\n", "isready\n", "position startpos\n", "go nodes 8\n")
        for _ in range(500):  # the go is answered on its own thread
            thread = engine._search_thread
            if engine._unhealthy is not None and (thread is None or not thread.is_alive()):
                break
            time.sleep(0.01)
        yield from ("isready\n", "go nodes 8\n")

    monkeypatch.setattr(main.sys, "stdin", stdin())
    assert main.main() == 1
    out = capsys.readouterr()
    lines = out.out.splitlines()
    assert lines.count("readyok") == 1  # the startup one only
    assert len(_bestmoves(out.out)) == 1  # the second go is never played
    assert any(ln.startswith("info string engine health check failed") for ln in lines)
    assert "deepfin: engine health check failed: RuntimeError: device lost" in out.err
    worker.close.assert_called_once()


def test_quit_does_not_sit_out_an_in_flight_rebuild(monkeypatch, capsys) -> None:
    """close() waits only briefly for a recovery build; one still in torch makes
    main hard-exit instead of finalizing under it (cf. the startup builder)."""
    import chess_anti_engine.uci.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_SHUTDOWN_RECOVERY_JOIN_S", 0.1)
    release = threading.Event()

    def stuck_rebuild(*_a, **_kw):
        release.wait(10)
        return MagicMock()

    engine, worker, _ = _flaky_engine(stuck_rebuild)
    runs = [0]

    def run(board, **_kw):
        runs[0] += 1
        if runs[0] == 1:
            return _ok(board)  # startup warmup
        raise RuntimeError("broken")

    worker.run.side_effect = run
    monkeypatch.setattr(main, "_load_models", lambda *_a, **_kw: [SimpleNamespace()])
    monkeypatch.setattr(main, "_make_evaluator_factory", lambda *_a, **_kw: lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "_build_engine", lambda **_kw: engine)
    monkeypatch.setattr(main.sys, "argv", ["deepfin", "--checkpoint", "fake.pt", "--device", "cpu"])
    exits: list[int] = []
    monkeypatch.setattr(main, "_hard_exit", exits.append)

    def stdin():
        yield from ("uci\n", "isready\n", "position startpos\n", "go nodes 8\n")
        for _ in range(500):
            if engine.background_build_running():
                break
            time.sleep(0.01)
        yield "quit\n"

    monkeypatch.setattr(main.sys, "stdin", stdin())
    t0 = time.monotonic()
    assert main.main() == 0
    assert time.monotonic() - t0 < 5.0
    assert exits == [0]
    assert "recovery build still running at shutdown" in capsys.readouterr().err
    release.set()
    _join_recovery(engine)


# --- (a) exact streak / recovery bookkeeping ----------------------------------


def test_quit_that_drops_the_emit_also_drops_the_owed_rebuild(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = RuntimeError("broken")
    engine.dispatch(parse_command("position startpos"))
    engine.dispatch(parse_command("go infinite"))
    for _ in range(500):
        if worker.run.called:
            break
        time.sleep(0.01)
    engine.dispatch(parse_command("quit"))
    assert _bestmoves(capsys.readouterr().out) == []
    assert engine._evaluator_recovery_due is False
    assert engine._recovery_thread is None
    assert rebuilt == []


def test_ponder_failure_then_searched_ponderhit_owes_no_rebuild(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    calls = [0]

    def run(board, **_kw):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("ponder phase fault")
        return _ok(board)

    worker.run.side_effect = run
    engine.dispatch(parse_command("setoption name Ponder value true"))
    engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
    engine.dispatch(parse_command("go ponder wtime 5000 btime 5000"))
    for _ in range(500):
        if calls[0] >= 1:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    engine.dispatch(parse_command("ponderhit"))
    _join(engine)
    assert calls[0] == 2
    assert len(_bestmoves(capsys.readouterr().out)) == 1
    assert engine._evaluator_recovery_due is False
    assert engine._consecutive_search_failures == 0
    assert engine._recovery_thread is None
    assert rebuilt == []


def test_a_go_ponder_counts_once_in_the_streak(capsys) -> None:
    """Superseded ponder fallback: not a move. Published one (stop): one move."""
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = RuntimeError("broken")
    engine.dispatch(parse_command("setoption name Ponder value true"))
    for release in ("ponderhit", "stop"):
        engine._consecutive_search_failures = 0
        engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
        engine.dispatch(parse_command("go ponder wtime 5000 btime 5000"))
        time.sleep(0.05)
        engine.dispatch(parse_command(release))
        _join(engine)
        out = capsys.readouterr().out
        assert len(_bestmoves(out)) == 1
        assert engine._consecutive_search_failures == 1, release
        assert "WARNING search failed on 1 consecutive move(s)" in out
        _join_recovery(engine)
    assert len(rebuilt) == 1  # one streak, one rebuild


def test_solved_root_and_ponder_transition_fallbacks_extend_the_streak_without_a_rebuild(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = SolvedRootWitnessUnavailable({}, (), {}, {})
    _go(engine)
    assert engine._consecutive_search_failures == 1
    out = capsys.readouterr().out
    assert "solved_root_without_move_witness" in out
    assert len(_bestmoves(out)) == 1

    result = engine._run_one_phase(
        SearchLimits(deadline_ms=10), is_ponder=False, board=chess.Board(),
        started_at=time.monotonic() - 1.0,
    )
    assert result.search_incomplete == "ponder_transition_deadline"
    assert engine._consecutive_search_failures == 2
    assert engine._evaluator_recovery_due is False
    assert rebuilt == []

    worker.run.side_effect = _ok
    _go(engine)
    assert engine._consecutive_search_failures == 0


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

