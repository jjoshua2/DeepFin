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

import threading
from collections.abc import Callable
from typing import Any

import chess
import pytest

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import Deadline, SearchWorker
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
        options=EngineOptions(max_batch=64, threads=walkers, vl_gather=32),
    )


def _expected_path(options: EngineOptions) -> str:
    if options.search_parallel == "gumbel":
        return "rpg"
    if options.use_multi_gpu_pucv:
        return "pucv_pool"
    if options.threads > 1:
        return "walker"
    return "pucv" if options.use_vl else "gumbel"


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


def test_failed_threads_tier_cross_keeps_the_previous_setup(make_engine, monkeypatch, capsys) -> None:
    engine = make_engine(walkers=1)
    _fail(monkeypatch, "_build_walker_pool")
    engine.dispatch(parse_command("setoption name Threads value 2"))
    assert engine._options.threads == 1
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



# --- PUCT-family / VLossWeight / UseVL / LeafGather routes (round 3) ----------
#
# These rebuilt the walker pool / PucvChunker close-then-build, and wrote the new
# value into worker state (`_cfg`, `_use_pucv`, `_vloss_weight`) BEFORE the build,
# outside the `EngineOptions` snapshot: a failed `setoption CPuct 2.5` left options
# at 1.75, the worker config at 2.5 and the walker pool CLOSED.


def _search_engine(*, walkers: int, use_vl: bool = False) -> Engine:
    engine = _engine(walkers=walkers, multi=False)
    if use_vl:
        engine.dispatch(parse_command("setoption name UseVL value true"))
        assert engine._worker.realized_search_path() == "pucv"
    return engine


@pytest.mark.parametrize(("option", "field", "value"), [
    ("CPuct", "c_puct", "2.5"),
    ("CPuctFactor", "cpuct_factor", "1.0"),
    ("CPuctBase", "cpuct_base", "20000"),
    ("FpuReduction", "fpu_reduction", "0.7"),
])
@pytest.mark.parametrize(("walkers", "use_vl", "builder"), [
    (2, False, "_build_walker_pool"),
    (1, True, "_build_pucv"),
])
def test_failed_puct_family_rebuild_keeps_options_worker_config_and_helpers(
    make_engine, monkeypatch, capsys, option, field, value, walkers, use_vl, builder,
) -> None:
    engine = make_engine(walkers=walkers, multi=False)
    if use_vl:
        engine.dispatch(parse_command("setoption name UseVL value true"))
    worker = engine._worker
    path = worker.realized_search_path()
    pool, pucv = worker._walker_pool, worker._pucv
    before_opt = engine._options.search_value(field)
    before_cfg = getattr(worker._cfg, field)
    calls = _fail(monkeypatch, builder)
    capsys.readouterr()
    engine.dispatch(parse_command(f"setoption name {option} value {value}"))
    assert calls[0] >= 1
    assert engine._options.search_value(field) == before_opt
    assert getattr(worker._cfg, field) == before_cfg  # worker state rolled back too
    # The live descent object (whatever the rollback left installed) carries the
    # restored value, and nothing installed is closed.
    assert worker._puct_descent_values()[field] == pytest.approx(before_cfg)
    assert (worker._walker_pool is None) == (pool is None)
    assert (worker._pucv is None) == (pucv is None)
    if worker._walker_pool is not None:
        assert not worker._walker_pool.closed
    assert f"previous configuration kept: path={path}" in capsys.readouterr().out
    assert _assert_searches_and_reports_truthfully(engine, capsys) == path


def test_failed_vloss_weight_rebuild_keeps_the_live_walker_pool(make_engine, monkeypatch, capsys) -> None:
    engine = make_engine(walkers=2, multi=False)
    pool = engine._worker._walker_pool
    before = engine._worker._vloss_weight
    _fail(monkeypatch, "_build_walker_pool")
    engine.dispatch(parse_command(f"setoption name VLossWeight value {before + 3}"))
    assert engine._worker._vloss_weight == before
    assert engine._worker._walker_pool is pool
    assert not pool.closed
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "walker"


def test_failed_use_vl_does_not_leave_the_pucv_flags_set(make_engine, monkeypatch, capsys) -> None:
    """Before: `_use_pucv` stayed True, the rollback rebuilt PUCV, and the game
    was searched with untuned PUCT while options said UseVL=false."""
    engine = make_engine(walkers=1, multi=False)
    _fail(monkeypatch, "_build_pucv")
    engine.dispatch(parse_command("setoption name UseVL value true"))
    assert engine._options.use_vl is False
    assert engine._worker._use_pucv is False
    assert "previous configuration kept: path=gumbel" in capsys.readouterr().out
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "gumbel"


def test_failed_leaf_gather_keeps_the_previous_gather(make_engine, monkeypatch, capsys) -> None:
    engine = make_engine(walkers=2, multi=False)
    gather = engine._worker._walker_gather
    pool = engine._worker._walker_pool
    _fail(monkeypatch, "_build_walker_pool")
    engine.dispatch(parse_command("setoption name LeafGather value 4"))
    assert engine._options.leaf_gather == 1
    assert engine._worker._walker_gather == gather
    assert engine._worker._walker_pool is pool
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "walker"


@pytest.mark.parametrize(("setter", "arg", "field"), [
    ("set_walker_gather", 4, "_walker_gather"),
    ("set_vloss_weight", 9, "_vloss_weight"),
    ("rebuild_puct_helpers", None, None),
])
def test_worker_setters_build_before_they_swap(monkeypatch, setter, arg, field) -> None:
    """Setter level, so the transaction's own restore cannot mask a setter that
    forgets to revert (the reviewer's surviving `set_walker_gather` mutant)."""
    engine = _engine(walkers=2, multi=False)
    try:
        worker = engine._worker
        pool = worker._walker_pool
        assert pool is not None
        before = getattr(worker, field) if field else None
        _fail(monkeypatch, "_build_walker_pool")
        with pytest.raises(MemoryError):
            getattr(worker, setter)(*(() if arg is None else (arg,)))
        assert worker._walker_pool is pool
        assert not pool.closed
        if field:
            assert getattr(worker, field) == before
    finally:
        engine.close()


def test_a_closed_walker_pool_counts_as_missing(make_engine) -> None:
    engine = make_engine(walkers=2, multi=False)
    worker = engine._worker
    assert worker.classic_helpers_missing() is False
    pool = worker._walker_pool
    assert pool is not None
    pool.close()
    assert worker.classic_helpers_missing() is True


def test_a_setoption_that_leaves_options_and_path_disagreeing_falls_back(make_engine, monkeypatch, capsys) -> None:
    """Invariant on SUCCESS too: a handler that 'succeeds' while the realized
    path differs from what the options name lands on the last resort, loudly."""
    engine = make_engine(walkers=1, multi=False)

    def desync(self, _value):
        self._worker.set_use_pucv(True)  # path changes, options do not

    monkeypatch.setitem(Engine._SETOPTION_HANDLERS, "usevl", desync)
    engine.dispatch(parse_command("setoption name UseVL value true"))
    out = capsys.readouterr().out
    assert "realized path=pucv but the options name gumbel" in out
    assert "fell back to the single-evaluator classic Gumbel path" in out
    assert engine._worker.realized_search_path() == "gumbel"
    assert engine._options.use_vl is False
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "gumbel"


def test_rollback_after_a_completed_helper_swap_rebuilds_from_restored_state(
    make_engine, monkeypatch, capsys,
) -> None:
    """VLossWeight rebuilt the walker pool (new weight baked in), THEN a later
    step raised: restoring the worker field alone would leave the live pool on
    the failed value. The transaction rebuilds helpers from the restored state."""
    engine = make_engine(walkers=2, multi=False)
    before = engine._worker._vloss_weight

    def boom(*_a, **_kw):
        raise MemoryError("injected after the helper swap")

    monkeypatch.setattr(engine, "_reinstall_configured_search_path", boom)
    engine.dispatch(parse_command(f"setoption name VLossWeight value {before + 5}"))
    worker = engine._worker
    assert worker._vloss_weight == before
    pool = worker._walker_pool
    assert pool is not None
    assert pool._cfg.vloss_weight == before  # the LIVE pool, not just the field
    assert "previous configuration kept: path=walker" in capsys.readouterr().out
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "walker"


def test_set_use_pucv_sets_its_flags_only_after_the_build(monkeypatch) -> None:
    engine = _engine(walkers=1, multi=False)
    try:
        worker = engine._worker
        _fail(monkeypatch, "_build_pucv")
        with pytest.raises(MemoryError):
            worker.set_use_pucv(True, gather=7)
        assert worker._use_pucv is False
        assert worker._pucv is None
        assert worker._pucv_gather != 7
    finally:
        engine.close()


def test_a_rollback_that_leaves_the_wrong_path_falls_back_loudly(make_engine, monkeypatch, capsys) -> None:
    """The invariant also gates the ROLLBACK: a restore that 'succeeds' while
    the realized path still disagrees with the options is not 'kept'."""
    engine = make_engine(walkers=1, multi=False)

    def desync_then_fail(self, _value):
        self._worker.set_use_pucv(True)
        raise MemoryError("injected after the path changed")

    monkeypatch.setitem(Engine._SETOPTION_HANDLERS, "usevl", desync_then_fail)
    monkeypatch.setattr(engine, "_ensure_configured_path_installed", lambda **_kw: None)
    engine.dispatch(parse_command("setoption name UseVL value true"))
    out = capsys.readouterr().out
    assert "previous configuration could not be restored" in out
    assert "realized path=pucv but the options name gumbel" in out
    assert "previous configuration kept" not in out
    assert engine._worker.realized_search_path() == "gumbel"
    assert _assert_searches_and_reports_truthfully(engine, capsys) == "gumbel"


def test_restored_hash_cap_resyncs_the_live_tree_growth_step() -> None:
    """A rolled-back `_max_tree_bytes` must reach the live tree's growth step,
    which `set_max_tree_mb` keeps in step with it; restoring the field alone
    would leave the tree growing under the abandoned cap."""
    from tests.test_uci_hash_accounting import _worker as _hash_worker

    w = _hash_worker()
    w.set_max_tree_mb(64)
    w.run(chess.Board(), stop_event=threading.Event(), deadline=Deadline(60_000),
          max_nodes=64, info_cb=lambda **_kw: None)
    assert w._tree is not None
    snapshot = w.search_state()
    expected = w._tree_growth_step_nodes()
    w.set_max_tree_mb(4096)
    assert w._tree_growth_step_nodes() != expected

    seen: list[int] = []

    class _Spy:
        def __init__(self, tree: Any) -> None:
            self._t = tree

        def set_growth_step(self, n: int) -> None:
            seen.append(n)
            self._t.set_growth_step(n)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._t, name)

    w._tree = _Spy(w._tree)  # pyright: ignore[reportAttributeAccessIssue]
    assert w.restore_search_state(snapshot) is True
    assert seen == [expected]
