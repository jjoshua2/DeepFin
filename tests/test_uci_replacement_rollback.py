"""After ANY failed evaluator / pool replacement the engine still searches.

Each case injects a build failure into one replacement route and then checks
the two things a tournament operator would: the NEXT `go` runs a real search
(not the unsearched fallback), and `searchconfig` reports the path that is
actually installed, which in turn matches the options. Before build-then-swap a
failed `set_evaluator` left the worker pointing at a CLOSED walker pool (every
later move: "WalkerPool is closed" + fallback), and a failed multi-GPU
transition left options advertising a pool nobody had installed.

CPU only: real `SearchWorker`, tiny real models, two CPU "devices" for the
multi-GPU pools (the same fixtures `test_root_parallel_gumbel.py` uses).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import chess
import pytest

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchWorker
from tests.test_root_parallel_gumbel import _make_evaluator


def _engine(*, walkers: int = 1, multi: bool = True) -> Engine:
    worker = SearchWorker(
        _make_evaluator(max_batch=64), device="cpu",
        gumbel_cfg=GumbelConfig(
            input_extra_features="v1", simulations=32, add_noise=False, temperature=0.0,
        ),
        chunk_sims=32, n_walkers=walkers,
    )

    def rebuild(max_batch: int, _cache: int = 0, *, n_walkers: int | None = None) -> Any:
        del n_walkers
        return _make_evaluator(max_batch=max_batch)

    def factories(_mb: int, _gather: int) -> list[Callable[[], Any]]:
        return [_make_evaluator, _make_evaluator]

    return Engine(
        worker,
        rebuild_evaluator=rebuild,
        rebuild_multi_gpu_pucv_factories=factories if multi else None,
        options=EngineOptions(max_batch=64, search_walkers=walkers, vl_gather=32),
    )


def _expected_path(options: EngineOptions) -> str:
    if options.search_parallel == "gumbel":
        return "rpg"
    if options.use_multi_gpu_pucv:
        return "pucv_pool"
    return "walker" if options.search_walkers > 1 else "gumbel"


def _assert_searches_and_reports_truthfully(engine: Engine, capsys) -> str:
    capsys.readouterr()
    engine.dispatch(parse_command("position startpos moves e2e4"))
    engine.dispatch(parse_command("go nodes 48"))
    thread = engine._search_thread
    assert thread is not None
    thread.join(30)
    assert not thread.is_alive()
    out = capsys.readouterr().out
    moves = [ln.split()[1] for ln in out.splitlines() if ln.startswith("bestmove ")]
    board = chess.Board()
    board.push_uci("e2e4")
    assert len(moves) == 1, out
    assert chess.Move.from_uci(moves[0]) in board.legal_moves
    assert "search error" not in out, out
    assert "bestmove_fallback_used" not in out, out
    engine.dispatch(parse_command("searchconfig"))
    line = next(
        ln for ln in capsys.readouterr().out.splitlines()
        if ln.startswith("info string searchconfig path=")
    )
    reported = line.split("path=", 1)[1].split()[0]
    assert reported == engine._worker.realized_search_path()
    assert reported == _expected_path(engine._options), (reported, engine._options)
    return reported


def _fail(monkeypatch, name: str, *, times: int = 1, teardown_first: bool = False) -> list[int]:
    """Make ``SearchWorker.<name>`` raise ``times`` times, then behave normally.

    ``teardown_first`` mimics the real pool installers, which clear every other
    path BEFORE constructing the new pool, so a failure there loses the old one.
    """
    real = getattr(SearchWorker, name)
    calls = [0]

    def faulty(self: SearchWorker, *args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] <= times:
            if teardown_first:
                self.clear_multi_gpu_pucv(restore_fallback=False)
                self.clear_root_parallel_gumbel()
                if self._walker_pool is not None:
                    self._walker_pool.close()
                    self._walker_pool = None
                self._pucv = None
            raise MemoryError(f"injected {name} failure")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(SearchWorker, name, faulty)
    return calls


@pytest.fixture
def make_engine():
    engines: list[Engine] = []

    def make(**kwargs: Any) -> Engine:
        engine = _engine(**kwargs)
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.close()


def test_failed_max_batch_rebuild_keeps_the_live_walker_pool(make_engine, monkeypatch, capsys) -> None:
    """The reviewer's repro: walkers=2, MaxBatch, walker-pool build fails."""
    engine = make_engine(walkers=2)
    pool = engine._worker._walker_pool
    calls = _fail(monkeypatch, "_build_walker_pool")
    engine.dispatch(parse_command("setoption name MaxBatch value 128"))
    assert calls[0] == 1
    assert engine._worker._walker_pool is pool
    assert engine._options.max_batch == 64
    out = capsys.readouterr().out
    assert "previous configuration kept: path=walker" in out
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "walker"


def test_failed_eval_cache_rebuild_keeps_the_previous_setup(make_engine, monkeypatch, capsys) -> None:
    engine = make_engine(walkers=2)
    _fail(monkeypatch, "_build_walker_pool")
    engine.dispatch(parse_command("setoption name EvalCacheEntries value 128"))
    assert engine._options.eval_cache_entries == 0
    assert engine._worker._eval_cache_entries == 0
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "walker"


def test_failed_search_walkers_tier_cross_keeps_the_previous_setup(make_engine, monkeypatch, capsys) -> None:
    engine = make_engine(walkers=1)
    _fail(monkeypatch, "_build_walker_pool")
    engine.dispatch(parse_command("setoption name SearchWalkers value 2"))
    assert engine._options.search_walkers == 1
    assert engine._worker._n_walkers == 1
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "gumbel"


def test_failed_pucv_pool_install_keeps_the_old_primary(make_engine, monkeypatch, capsys) -> None:
    engine = make_engine(walkers=2)
    primary = engine._worker._evaluator
    _fail(monkeypatch, "install_multi_gpu_pucv", teardown_first=True)
    engine.dispatch(parse_command("setoption name UseMultiGpuPUCV value true"))
    assert engine._worker._evaluator is primary
    assert engine._options.use_multi_gpu_pucv is False
    assert "previous configuration kept: path=walker" in capsys.readouterr().out
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "walker"


def test_failed_rpg_install_restores_the_classic_path(make_engine, monkeypatch, capsys) -> None:
    engine = make_engine(walkers=2)
    primary = engine._worker._evaluator
    _fail(monkeypatch, "install_root_parallel_gumbel", teardown_first=True)
    engine.dispatch(parse_command("setoption name SearchParallel value gumbel"))
    assert engine._worker._evaluator is primary
    assert engine._options.search_parallel == "pucv"
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "walker"


def test_failed_primary_build_when_leaving_rpg_keeps_rpg(make_engine, capsys) -> None:
    engine = make_engine()
    engine.dispatch(parse_command("setoption name SearchParallel value gumbel"))
    assert engine._worker.realized_search_path() == "rpg"
    rpg = engine._worker._rpg_pool

    def refuse(*_a: Any, **_kw: Any) -> Any:
        raise MemoryError("injected primary build failure")

    engine._rebuild_evaluator = refuse
    engine.dispatch(parse_command("setoption name SearchParallel value pucv"))
    assert engine._worker._rpg_pool is rpg
    assert engine._options.search_parallel == "gumbel"
    assert "keeping the previous evaluator and settings" in capsys.readouterr().out
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "rpg"


def test_failed_primary_build_when_leaving_pucv_pool_keeps_the_pool(make_engine, capsys) -> None:
    engine = make_engine()
    engine.dispatch(parse_command("setoption name UseMultiGpuPUCV value true"))
    assert engine._worker.realized_search_path() == "pucv_pool"
    pool = engine._worker._pucv_pool

    def refuse(*_a: Any, **_kw: Any) -> Any:
        raise MemoryError("injected primary build failure")

    engine._rebuild_evaluator = refuse
    engine.dispatch(parse_command("setoption name UseMultiGpuPUCV value false"))
    assert engine._worker._pucv_pool is pool
    assert engine._options.use_multi_gpu_pucv is True
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "pucv_pool"


def test_failed_swap_when_leaving_pucv_pool_reinstalls_the_pool(make_engine, monkeypatch, capsys) -> None:
    """The primary built, but its helpers did not: options and pool come back."""
    engine = make_engine(walkers=2)
    engine.dispatch(parse_command("setoption name UseMultiGpuPUCV value true"))
    _fail(monkeypatch, "_build_walker_pool", times=2)  # restore + set_evaluator
    engine.dispatch(parse_command("setoption name UseMultiGpuPUCV value false"))
    assert engine._options.use_multi_gpu_pucv is True
    assert "previous configuration kept: path=pucv_pool" in capsys.readouterr().out
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "pucv_pool"


def test_failed_pool_rebuild_on_max_batch_under_rpg_reinstalls_rpg(make_engine, monkeypatch, capsys) -> None:
    engine = make_engine()
    engine.dispatch(parse_command("setoption name SearchParallel value gumbel"))
    primary = engine._worker._evaluator
    _fail(monkeypatch, "install_root_parallel_gumbel", teardown_first=True)
    engine.dispatch(parse_command("setoption name MaxBatch value 128"))
    assert engine._options.max_batch == 64
    assert engine._worker._evaluator is primary  # pool first, primary swap last
    assert "previous configuration kept: path=rpg" in capsys.readouterr().out
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "rpg"


def test_unrestorable_pool_falls_back_to_a_consistent_single_evaluator_path(make_engine, monkeypatch, capsys) -> None:
    """When the previous pool cannot be rebuilt either, options must stop
    advertising it: a pool in the options with none installed is the silent
    one-GPU configuration."""
    engine = make_engine()
    engine.dispatch(parse_command("setoption name UseMultiGpuPUCV value true"))
    _fail(monkeypatch, "install_root_parallel_gumbel", times=99, teardown_first=True)
    _fail(monkeypatch, "install_multi_gpu_pucv", times=99, teardown_first=True)
    engine.dispatch(parse_command("setoption name SearchParallel value gumbel"))
    out = capsys.readouterr().out
    assert "previous configuration could not be restored" in out
    assert engine._options.use_multi_gpu_pucv is False
    assert engine._options.search_parallel == "pucv"
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "gumbel"

