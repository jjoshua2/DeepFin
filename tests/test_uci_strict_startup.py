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


# --- (a) loud in-play fallback; streak; boundary rebuild + strict re-check ----


def _flaky_engine(rebuild=None) -> tuple[Engine, MagicMock, list[object]]:
    worker = MagicMock()
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    worker.fresh_root_policy_bestmove.return_value = None
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    worker.realized_search_path.return_value = "gumbel"
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


def _make_unhealthy(engine: Engine, worker: MagicMock) -> None:
    worker.run.side_effect = RuntimeError("broken")
    for _ in range(3):
        _go(engine)
    assert engine._unhealthy is not None


def test_a_failed_search_is_loud_and_nothing_is_rebuilt_mid_game(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = RuntimeError("CUDA error: illegal memory access")
    _go(engine)
    out = capsys.readouterr().out
    assert len(_bestmoves(out)) == 1  # never lose on time
    assert "WARNING search failed on 1 consecutive move(s)" in out
    assert rebuilt == []  # no background (or any) rebuild in play
    worker.set_evaluator.assert_not_called()


def test_three_fallbacks_in_a_row_mark_unhealthy_once(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = RuntimeError("broken")
    for _ in range(2):
        _go(engine)
    assert engine._unhealthy is None
    _go(engine)
    _go(engine)
    out = capsys.readouterr().out
    assert "WARNING engine unhealthy: 3 consecutive moves were fallbacks" in out
    assert out.count("WARNING engine unhealthy:") == 1  # announced once, not per move
    assert len(_bestmoves(out)) == 4  # unhealthy still answers every go
    assert rebuilt == []


def test_mid_game_isready_answers_at_once_even_when_unhealthy(capsys) -> None:
    """An isready between moves must never run the re-check or exit."""
    engine, worker, rebuilt = _flaky_engine()
    _make_unhealthy(engine, worker)
    capsys.readouterr()
    worker.run.reset_mock()
    t0 = time.monotonic()
    engine.dispatch(parse_command("isready"))
    assert time.monotonic() - t0 < 0.5
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == "readyok"
    assert "health check" not in out
    worker.run.assert_not_called()
    assert rebuilt == []
    assert engine.exit_code == 0
    assert not engine.quit_requested


def test_boundary_rebuilds_synchronously_announces_first_then_rechecks(capsys) -> None:
    calls: list[str] = []
    rebuilt: list[object] = []

    def rebuild(_max_batch, _cache=0, /, **_kw):
        calls.append("build")
        ev = MagicMock(name="health-rebuilt")
        rebuilt.append(ev)
        return ev

    engine, worker, _ = _flaky_engine(rebuild)
    _make_unhealthy(engine, worker)
    capsys.readouterr()
    worker.set_evaluator.side_effect = lambda *_a, **_kw: calls.append("install")

    def warm(board, **_kw):
        calls.append("warmup")
        return _ok(board)

    worker.run.side_effect = warm
    engine.dispatch(parse_command("ucinewgame"))
    assert calls == []  # ucinewgame only opens the boundary
    engine.dispatch(parse_command("isready"))
    out = capsys.readouterr().out
    assert calls == ["build", "install", "warmup"]  # synchronous, in order
    worker.set_evaluator.assert_called_once_with(rebuilt[0])
    announce = out.index("rebuilding the evaluator and rerunning the strict startup check")
    assert announce < out.index("health rebuild: evaluator rebuilt")
    assert announce < out.index("health check passed")
    assert engine._unhealthy is None
    assert engine._consecutive_search_failures == 0
    assert engine.exit_code == 0


def test_failed_boundary_rebuild_keeps_the_evaluator_and_the_recheck_decides(capsys) -> None:
    def refuse(*_a, **_kw):
        raise RuntimeError("out of memory")

    engine, worker, _ = _flaky_engine(refuse)
    _make_unhealthy(engine, worker)
    worker.run.side_effect = _ok  # the fault was transient
    engine.dispatch(parse_command("ucinewgame"))
    engine.dispatch(parse_command("isready"))
    out = capsys.readouterr().out
    assert "health rebuild: evaluator rebuild failed" in out
    worker.set_evaluator.assert_not_called()
    assert "health check passed" in out
    assert engine._unhealthy is None
    assert engine.exit_code == 0


def test_failed_boundary_install_is_contained_and_the_recheck_still_runs(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    _make_unhealthy(engine, worker)
    worker.set_evaluator.side_effect = RuntimeError("install failed: device busy")
    worker.run.side_effect = _ok
    engine.dispatch(parse_command("ucinewgame"))
    engine.dispatch(parse_command("isready"))
    out = capsys.readouterr().out
    assert len(rebuilt) == 1
    assert "health rebuild: install failed" in out
    assert "health rebuild failed; previous configuration kept" in out  # restored
    assert "health rebuild: evaluator rebuilt" not in out  # only on success
    assert "error handling" not in out  # contained here, not by the dispatch guard
    assert "health check passed" in out
    assert engine._unhealthy is None


def test_ucinewgame_only_opens_the_boundary_position_and_go_are_never_delayed(capsys) -> None:
    """A GUI that sends ucinewgame + position + go with no isready: nothing is
    rebuilt and nothing can exit inside ucinewgame; the go is answered."""
    engine, worker, rebuilt = _flaky_engine()
    _make_unhealthy(engine, worker)
    capsys.readouterr()
    worker.run.side_effect = _ok
    engine.dispatch(parse_command("ucinewgame"))
    assert rebuilt == []
    assert not engine.quit_requested
    engine.dispatch(parse_command("position startpos moves e2e4"))
    _go(engine)
    out = capsys.readouterr().out
    assert len(_bestmoves(out)) == 1
    assert "health check" not in out
    assert rebuilt == []
    assert engine._unhealthy is not None  # still owed: no isready between games


def test_a_failed_mid_game_rewarm_marks_unhealthy_and_keeps_the_path_dirty(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = _ok
    engine.dispatch(parse_command("position startpos moves e2e4"))
    _go(engine)  # in game
    engine._warmup_dirty = True
    worker.run.side_effect = RuntimeError("search-path capture failed")
    capsys.readouterr()
    engine.dispatch(parse_command("isready"))
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == "readyok"
    assert "search-path re-warm failed" in out
    assert "error handling" not in out
    assert engine._warmup_dirty is True  # retried at the next free isready
    assert engine._unhealthy == "the search-path re-warm at isready failed"
    assert rebuilt == []  # mid-game: the rebuild waits for the boundary
    assert not engine.quit_requested


def test_a_raising_health_rebuild_is_one_reported_outcome(monkeypatch, capsys) -> None:
    engine, worker, _ = _flaky_engine()
    _make_unhealthy(engine, worker)
    worker.run.side_effect = _ok

    def boom(*_a, **_kw):
        raise RuntimeError("invariant check exploded")

    monkeypatch.setattr(engine, "_enforce_path_invariant", boom)
    engine.dispatch(parse_command("ucinewgame"))
    capsys.readouterr()
    engine.dispatch(parse_command("isready"))
    out = capsys.readouterr().out
    assert out.count("health rebuild raised (RuntimeError: invariant check exploded)") == 1
    assert out.count("health check passed") == 1
    assert "error handling" not in out
    assert out.splitlines()[-1] == "readyok"
    assert engine._unhealthy is None


def test_first_isready_after_ucinewgame_reruns_the_check_before_the_first_go(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = _ok
    _go(engine)  # in game
    engine.dispatch(parse_command("ucinewgame"))  # healthy: nothing to re-check
    engine._mark_unhealthy("test")
    capsys.readouterr()
    engine.dispatch(parse_command("isready"))  # before this game's first go
    out = capsys.readouterr().out
    assert "health check passed" in out
    assert out.splitlines()[-1] == "readyok"
    assert engine._unhealthy is None
    assert len(rebuilt) == 1


def test_failed_health_recheck_exits_nonzero_before_readyok(monkeypatch, capsys) -> None:
    worker = MagicMock()
    worker._chunk_sims = 16
    worker._pucv_pool = worker._rpg_pool = None
    worker.advance_root.return_value = True
    worker.root_policy_bestmove.return_value = None
    worker.fresh_root_policy_bestmove.return_value = None
    worker.realized_search_path.return_value = "gumbel"
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
        yield from ("uci\n", "isready\n", "position startpos moves e2e4\n")
        for _ in range(3):
            yield "go nodes 8\n"
            for _ in range(500):  # each go is answered on its own thread
                thread = engine._search_thread
                if thread is None or not thread.is_alive():
                    break
                time.sleep(0.01)
        yield from ("ucinewgame\n", "isready\n", "go nodes 8\n")

    monkeypatch.setattr(main.sys, "stdin", stdin())
    assert main.main() == 1
    out = capsys.readouterr()
    lines = out.out.splitlines()
    assert lines.count("readyok") == 1  # the startup one only
    assert len(_bestmoves(out.out)) == 3  # the go after the failed check is never played
    assert any(ln.startswith("info string engine health check failed") for ln in lines)
    assert "deepfin: engine health check failed: RuntimeError: device lost" in out.err
    worker.close.assert_called_once()


def test_mid_game_building_setoption_applies_at_once_and_is_transactional(capsys) -> None:
    built: list[int] = []

    def rebuild(max_batch, _cache=0, /, **_kw):
        built.append(max_batch)
        if max_batch == 256:
            raise RuntimeError("out of memory")
        return MagicMock(name=f"built-{max_batch}")

    engine, worker, _ = _flaky_engine(rebuild)
    worker.run.side_effect = _ok
    _go(engine)  # in game
    engine.dispatch(parse_command("setoption name MaxBatch value 512"))
    assert built == [512]  # synchronously, right here -- nothing deferred
    assert engine._options.max_batch == 512
    worker.set_evaluator.assert_called_once()
    engine.dispatch(parse_command("setoption name MaxBatch value 256"))  # fails
    assert built == [512, 256]
    assert engine._options.max_batch == 512  # rolled back with the evaluator kept
    assert worker.set_evaluator.call_count == 1
    assert "evaluator rebuild failed" in capsys.readouterr().out


def test_boundary_checks_wait_for_a_free_worker(monkeypatch, capsys) -> None:
    """Unhealthy, between games, but a parked search still in the worker: the
    isready answers at once and neither rebuilds nor re-checks."""
    monkeypatch.setattr("chess_anti_engine.uci.engine._GO_WAIT_NO_CLOCK_S", 0.05)
    engine, worker, rebuilt = _flaky_engine()
    _make_unhealthy(engine, worker)
    release, entered = threading.Event(), threading.Event()

    def hang(board, **_kw):
        entered.set()
        release.wait(10)
        return _ok(board)

    worker.run.side_effect = hang
    engine.dispatch(parse_command("go infinite"))
    assert entered.wait(5)
    engine.dispatch(parse_command("stop"))  # parked: an orphan
    engine.dispatch(parse_command("ucinewgame"))
    assert not engine.worker_is_free()
    capsys.readouterr()
    engine.dispatch(parse_command("isready"))
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == "readyok"
    assert rebuilt == []
    assert "health check" not in out
    release.set()
    for t in list(engine._orphan_threads):
        t.join(5)


def test_busy_worker_answers_after_a_boundary_still_start_the_game(monkeypatch, capsys) -> None:
    """The reviewer's scenario: a search ignores stop and is parked, a new game
    starts, its first gos are busy-worker answers (unhealthy after 3), the stuck
    search ends -- and a mid-game isready must still not rebuild, re-check or
    exit. `_in_game` used to be set only after the wait, which raised here."""
    monkeypatch.setattr("chess_anti_engine.uci.engine._GO_WAIT_NO_CLOCK_S", 0.05)
    engine, worker, rebuilt = _flaky_engine()
    release, entered = threading.Event(), threading.Event()

    def hang(board, **_kw):
        entered.set()
        release.wait(10)
        return _ok(board)

    worker.run.side_effect = hang
    engine.dispatch(parse_command("go infinite"))
    assert entered.wait(5)
    engine.dispatch(parse_command("stop"))  # parked: an orphan
    for boundary in ("ucinewgame", "position startpos"):
        engine.dispatch(parse_command(boundary))
        assert not engine._in_game
    engine.dispatch(parse_command("position startpos moves e2e4"))
    for _ in range(3):
        engine.dispatch(CmdGo(GoArgs(nodes=8)))  # busy-worker answers
        _join(engine)
        assert engine._in_game  # each one is a move of this game
    assert engine._unhealthy is not None
    release.set()
    for t in list(engine._orphan_threads):
        t.join(5)
    assert engine.worker_is_free()
    capsys.readouterr()
    engine.dispatch(parse_command("isready"))  # mid-game
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == "readyok"
    assert "health check" not in out
    assert rebuilt == []
    assert not engine.quit_requested


@pytest.mark.parametrize("first_go", ["go infinite", "go ponder wtime 5000 btime 5000"])
def test_a_parked_search_released_after_a_later_go_still_ends(first_go: str, monkeypatch, capsys) -> None:
    """A search parked after `stop` that returns only after a LATER go must still
    end: it waits on its own go's cancel / ponder-release event, not on the one
    the later go installed (never set), which held the worker forever and made
    every later move a busy-worker fallback."""
    monkeypatch.setattr("chess_anti_engine.uci.engine._GO_WAIT_NO_CLOCK_S", 0.05)
    engine, worker, _ = _flaky_engine()
    engine.dispatch(parse_command("setoption name Ponder value true"))
    release, entered = threading.Event(), threading.Event()

    def hang(board, **_kw):
        entered.set()
        release.wait(10)
        return _ok(board)

    worker.run.side_effect = hang
    engine.dispatch(parse_command("position startpos moves e2e4 e7e5"))
    engine.dispatch(parse_command(first_go))
    assert entered.wait(5)
    engine.dispatch(parse_command("stop"))  # parked; answered from the fallback
    engine.dispatch(CmdGo(GoArgs(nodes=8)))  # busy-worker answer: new events
    _join(engine)
    release.set()
    for t in list(engine._orphan_threads):
        t.join(5)
    assert engine.worker_is_free()
    worker.run.side_effect = _ok
    capsys.readouterr()
    _go(engine)
    out = capsys.readouterr().out
    assert "search not started" not in out  # a real search again
    assert len(_bestmoves(out)) == 1


def test_a_parked_searchs_late_result_does_not_touch_the_streak(monkeypatch, capsys) -> None:
    monkeypatch.setattr("chess_anti_engine.uci.engine._GO_WAIT_NO_CLOCK_S", 0.05)
    engine, worker, _ = _flaky_engine()
    release, entered = threading.Event(), threading.Event()

    def hang(board, **_kw):
        entered.set()
        release.wait(10)
        return _ok(board)  # a SEARCHED result -- for a move nobody plays

    worker.run.side_effect = hang
    engine.dispatch(parse_command("go infinite"))
    assert entered.wait(5)
    engine.dispatch(parse_command("stop"))  # parked; answered from the fallback
    streak = engine._consecutive_search_failures
    engine._consecutive_search_failures = streak + 2
    release.set()
    for t in list(engine._orphan_threads):
        t.join(5)
    assert engine._consecutive_search_failures == streak + 2  # not reset by it
    capsys.readouterr()


# --- (a) no `ucinewgame` at all: re-warm and the new-game `position` rule -----


def test_isready_rewarms_mid_game_without_ucinewgame(capsys) -> None:
    engine, worker, _ = _flaky_engine()
    worker.run.side_effect = _ok
    engine.dispatch(parse_command("position startpos moves e2e4"))
    _go(engine)
    assert engine._in_game
    engine.dispatch(parse_command("setoption name MaxBatch value 512"))
    assert engine._warmup_dirty
    runs = worker.run.call_count
    engine.dispatch(parse_command("isready"))
    assert worker.run.call_count == runs + 1  # the search path was re-warmed
    assert worker.run.call_args.args[0] == chess.Board()
    assert not engine._warmup_dirty
    assert capsys.readouterr().out.splitlines()[-1] == "readyok"


_START_FEN = chess.STARTING_FEN


@pytest.mark.parametrize(("cmd", "opens"), [
    ("position startpos", True),
    (f"position fen {_START_FEN}", True),
    ("position startpos moves e2e4", False),
    ("position startpos moves e2e4 e7e5 g1f3 b8c6", False),  # a later move
    ("position startpos moves e2e4 e7e5", False),  # takeback to an earlier ply
    ("position startpos moves d2d4", False),  # an analysis jump elsewhere
    ("position fen rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", False),
    ("position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4", False),
    ("position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", False),  # no castling
])
def test_only_the_start_position_with_no_moves_opens_a_game_boundary(cmd: str, opens: bool) -> None:
    engine, worker, _ = _flaky_engine()
    worker.run.side_effect = _ok
    engine.dispatch(parse_command("position startpos moves e2e4 e7e5 g1f3"))
    _go(engine)
    assert engine._in_game
    engine.dispatch(parse_command(cmd))
    assert engine._in_game is (not opens)


def test_ponder_miss_sequence_never_opens_a_boundary() -> None:
    engine, worker, _ = _flaky_engine()
    worker.run.side_effect = _ok
    engine.dispatch(parse_command("position startpos moves e2e4"))
    _go(engine)
    for cmd in ("position startpos moves e2e4 e7e5 g1f3 b8c6",  # ponder on b8c6
                "position startpos moves e2e4 e7e5 g1f3 g8f6"):  # the miss
        engine.dispatch(parse_command(cmd))
        assert engine._in_game


def test_session_without_ucinewgame_reaches_the_health_check_at_a_new_game(capsys) -> None:
    engine, worker, rebuilt = _flaky_engine()
    engine.dispatch(parse_command("position startpos moves e2e4"))
    _make_unhealthy(engine, worker)
    worker.run.side_effect = _ok
    engine.dispatch(parse_command("isready"))  # mid-game: no check
    assert engine._unhealthy is not None
    assert rebuilt == []
    engine.dispatch(parse_command("position startpos"))  # the next game starts
    assert rebuilt == []  # opening the boundary does nothing on its own
    capsys.readouterr()
    engine.dispatch(parse_command("isready"))
    out = capsys.readouterr().out
    assert len(rebuilt) == 1
    assert "health check passed" in out
    assert out.splitlines()[-1] == "readyok"
    assert engine._unhealthy is None


# --- (a) exact streak bookkeeping ---------------------------------------------


def test_ponder_failure_then_searched_ponderhit_does_not_extend_the_streak(capsys) -> None:
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
    assert engine._consecutive_search_failures == 0
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
    assert rebuilt == []


def test_solved_root_without_witness_is_exempt_and_transition_extends_the_streak(capsys) -> None:
    """A solved root whose proof names no move is a property of the position:
    announced and counted on its own, never in the unhealthy streak (#1265's
    node-bound re-raise makes it common)."""
    engine, worker, rebuilt = _flaky_engine()
    worker.run.side_effect = SolvedRootWitnessUnavailable({}, (), {}, {})
    for n in (1, 2, 3, 4):
        _go(engine)
        assert engine._witness_unavailable_roots == n
    assert engine._consecutive_search_failures == 0
    assert engine._unhealthy is None
    out = capsys.readouterr().out
    assert "solved_root_without_move_witness" in out
    assert "witness_unavailable_roots=4" in out
    assert len(_bestmoves(out)) == 4
    # ...and it does not RESET a streak a real fault started either.
    worker.run.side_effect = RuntimeError("broken")
    _go(engine)
    worker.run.side_effect = SolvedRootWitnessUnavailable({}, (), {}, {})
    _go(engine)
    assert engine._consecutive_search_failures == 1

    result = engine._run_one_phase(
        SearchLimits(deadline_ms=10), is_ponder=False, board=chess.Board(),
        started_at=time.monotonic() - 1.0,
    )
    assert result.search_incomplete == "ponder_transition_deadline"
    assert engine._consecutive_search_failures == 2
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


def test_a_go_whose_handler_always_raises_goes_unhealthy(monkeypatch, capsys) -> None:
    engine, _, _ = _flaky_engine()

    def boom(*_a, **_kw):
        raise RuntimeError("handler fault")

    monkeypatch.setattr(engine, "_note_leaf_tb_inactive", boom)
    for _ in range(3):
        _go(engine)
    out = capsys.readouterr().out
    assert len(_bestmoves(out)) == 3
    assert "WARNING engine unhealthy: 3 consecutive moves were fallbacks" in out



def test_last_resort_fallback_options_follow_the_teardown_step_by_step() -> None:
    """Each option is written right after the teardown it describes: a raise
    part-way leaves the options naming what is still installed."""
    worker = MagicMock()
    worker.clear_multi_gpu_pucv.side_effect = RuntimeError("pool teardown failed")
    engine = Engine(worker, options=EngineOptions(
        search_parallel="gumbel", use_multi_gpu_pucv=True, use_vl=True,
    ))
    outcome = engine._fall_back_to_single_evaluator_gumbel()
    assert "no working search path could be installed" in outcome
    worker.clear_root_parallel_gumbel.assert_called_once()
    assert engine._options.search_parallel == "pucv"  # the RPG pool is gone
    assert engine._options.use_multi_gpu_pucv is True  # its pool is still there
    assert engine._options.use_vl is True
    worker.set_use_pucv.assert_not_called()

    worker.clear_multi_gpu_pucv.side_effect = None
    engine._fall_back_to_single_evaluator_gumbel()
    assert engine._options.use_multi_gpu_pucv is False
    assert engine._options.use_vl is False
    assert engine._options.search_walkers == 1


def test_fresh_eval_keeps_its_absolute_150ms_cap_on_a_long_clock(capsys) -> None:
    """On a two-minute move a tenth of the clock is 12 s; the fresh root-eval
    fallback must still give up after the absolute 150 ms cap."""
    hang = threading.Event()
    worker = MagicMock()
    worker.advance_root.side_effect = RuntimeError("tree allocation failed")  # handler fault
    worker.reset_tree.side_effect = RuntimeError("tree allocation failed")
    worker.root_policy_bestmove.return_value = None
    worker.fresh_root_policy_bestmove.side_effect = lambda *_a, **_kw: hang.wait(30)
    engine = Engine(worker, options=EngineOptions())
    seen: list[float | None] = []
    real = engine._bounded_fresh_eval

    def spy(board: chess.Board, searchmoves: tuple[str, ...], budget_s: float | None) -> str | None:
        seen.append(budget_s)
        return real(board, searchmoves, budget_s)

    engine._bounded_fresh_eval = spy  # type: ignore[method-assign]
    engine.dispatch(parse_command("position startpos moves e2e4"))
    t0 = time.monotonic()
    engine.dispatch(parse_command("go movetime 120000"))
    _join(engine)
    elapsed = time.monotonic() - t0
    out = capsys.readouterr().out
    (budget,) = seen
    assert budget is not None
    assert budget > 10.0  # the clock alone would allow ~12 s
    assert "fresh root-eval fallback timed out after 150 ms" in out
    assert elapsed < 1.0, elapsed
    assert len(_bestmoves(out)) == 1
    hang.set()
    for t in list(engine._orphan_threads):
        t.join(5)
