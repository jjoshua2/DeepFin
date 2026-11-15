"""The degrade-path gaps #1315 left open, closed (in-process tests).

1. ``--no-multi-gpu-pucv`` routing chain: ``DegradingMultiGPUDispatcher``.
2. Root-parallel Gumbel: a device failure retires the pool and the same search
   continues on the classic path.
3. A device whose evaluator build / warm-up HANGS is timed out at startup.
4. ``go searchmoves`` keeps its restriction when the primary fails beside a pool.
5. The non-fast self-check requires the compiled warm-up's verification line.
6. A device that keeps stalling within one ``go`` is held out (flap limit).

Devices are CPU evaluators; faults are injected by wrappers, as in
``test_uci_multi_gpu_degrade.py`` (whose helpers are reused).
"""
from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest

import chess_anti_engine.uci.engine as engine_module
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.uci import __main__ as uci_main
from chess_anti_engine.uci import startup_progress, wedge
from chess_anti_engine.uci.device_guard import (
    DegradingMultiGPUDispatcher,
    DeviceCallRunner,
    DeviceStalled,
    GuardedPrimaryEvaluator,
)
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.multi_gpu_pucv_pool import (
    DEVICE_FLAP_LIMIT,
    MultiGpuPucvConfig,
    MultiGpuPucvPool,
)
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.root_parallel_gumbel import (
    RootParallelGumbelConfig,
    RootParallelGumbelPool,
    RPGDeviceFailure,
)
from chess_anti_engine.uci.search import SearchWorker
from tests.test_uci_multi_gpu_degrade import (
    _UNSEARCHED,
    _ended_or_parked,
    _engine_with_pool,
    _evaluator,
    _Faulty,
    _legal_after,
    _OnceSlow,
    _Out,
    _run,
    _wait_for,
)


@pytest.fixture
def out(monkeypatch) -> _Out:
    """Engine stdout capture (the same as ``test_uci_multi_gpu_degrade``'s)."""
    capture = _Out()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=capture, stderr=sys.stderr))
    return capture


def _model() -> Any:
    model = build_model(ModelConfig(
        input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
    ))
    model.eval()
    return model


def _direct(max_batch: int = 64, *, pad: bool = False) -> DirectGPUEvaluator:
    kw: dict[str, Any] = {"pad_to_compiled_buckets": True} if pad else {}
    return DirectGPUEvaluator(_model(), device="cpu", max_batch=max_batch, use_amp=False,
                              n_slots=2, **kw)


class _AllFaulty(_Faulty):
    """``_Faulty`` that gates EVERY ``evaluate*`` entry (the routing chain and
    classic Gumbel use ``evaluate_encoded`` / ``evaluate_legal_bf16``)."""

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if callable(attr) and name.startswith("evaluate"):
            def gated(*a: Any, **k: Any) -> Any:
                self._gate()
                return attr(*a, **k)
            return gated
        return attr

    evaluate_encoded = property(  # pyright: ignore[reportAssignmentType]
        lambda self: self.__getattr__("evaluate_encoded"),
    )
    evaluate_inplace_async = property(  # pyright: ignore[reportAssignmentType]
        lambda self: self.__getattr__("evaluate_inplace_async"),
    )


class _SlowEvery:
    """A device whose every ``evaluate*`` call takes ``delay`` s (a device that
    keeps answering just over the idle threshold)."""

    def __init__(self, inner: Any, delay: float) -> None:
        self._inner = inner
        self.delay = delay
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if callable(attr) and name.startswith("evaluate"):
            def slow(*a: Any, **k: Any) -> Any:
                self.calls += 1
                time.sleep(self.delay)
                return attr(*a, **k)
            return slow
        return attr


def _chain_engine(devices: list[Any]) -> tuple[Engine, SearchWorker, DegradingMultiGPUDispatcher]:
    chain = DegradingMultiGPUDispatcher(devices, labels=[f"cpu:{k}" for k in range(len(devices))])
    worker = SearchWorker(
        chain, device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    worker.configure_device_notes(engine_module.emit_info_string_unless_silenced, ())
    return Engine(worker, options=EngineOptions()), worker, chain


def _lines(out, needle: str) -> list[str]:
    return [ln for ln in out.text().splitlines() if needle in ln]


def _go(engine: Engine, out, moves: str, go: str, n: int, timeout: float = 25.0) -> tuple[str, float]:
    engine.dispatch(parse_command(f"position startpos moves {moves}" if moves else "position startpos"))
    started = time.monotonic()
    engine.dispatch(parse_command(go))
    got = out.wait_bestmoves(n, timeout=timeout)
    assert len(got) == n, out.text()
    return got[-1], time.monotonic() - started


# ── 1. the --no-multi-gpu-pucv routing chain ────────────────────────────────


def test_chain_redoes_a_raising_devices_batch_on_a_survivor() -> None:
    good, bad = _direct(), _AllFaulty(_direct(), mode="raise", after=0)
    chain = DegradingMultiGPUDispatcher([good, bad], labels=["g0", "g1"])
    chain.arm()
    try:
        x = np.random.default_rng(0).standard_normal((3, 146, 8, 8)).astype(np.float32)
        want = good.evaluate_encoded(x)
        for _ in range(3):  # round-robin would have hit g1 at least once
            pol, wdl = chain.evaluate_encoded(x)
            np.testing.assert_allclose(pol, want[0], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(wdl, want[1], rtol=1e-5, atol=1e-6)
        assert bad.entered.is_set()
        assert chain.failed_devices() == ("g1",)
        notes = chain.drain_notes()
        assert len(notes) == 1
        assert "DEGRADED: device g1 failed mid-search" in notes[0]
        assert "1 of 2 devices" in (chain.degraded_summary() or "")
    finally:
        chain.close()


def test_chain_abandons_a_stalled_call_restores_it_and_holds_a_flapping_device() -> None:
    good, stuck = _direct(), _AllFaulty(_direct(), mode="hang", after=0)
    chain = DegradingMultiGPUDispatcher([good, stuck], labels=["g0", "g1"])
    chain.arm()
    x = np.zeros((2, 146, 8, 8), dtype=np.float32)
    try:
        started = time.monotonic()
        for _ in range(2):
            chain.evaluate_encoded(x)
        took = time.monotonic() - started
        assert stuck.entered.is_set()
        assert 0.9 < took < 2.5  # the 1 s idle-survivor threshold, then g0 took it
        assert chain.failed_devices() == ("g1",)
        assert "stopped answering" in chain.drain_notes()[0]
        # Its call comes back clean: re-admitted before the next call.
        stuck.release.set()
        assert _wait_for(lambda: chain._runners[1].restorable)
        chain.evaluate_encoded(x)
        assert chain.failed_devices() == ()
        assert any("RESTORED: device g1" in n for n in chain.drain_notes())
        # Flap limit: it stalls again (2nd drop this search) -> held until begin_search.
        stuck.release.clear()
        stuck.entered.clear()
        for _ in range(3):
            chain.evaluate_encoded(x)
        assert chain.failed_devices() == ("g1",)
        assert chain._stall_drops_this_search[1] == DEVICE_FLAP_LIMIT
        stuck.release.set()
        assert _wait_for(lambda: chain._runners[1].restorable)
        chain.evaluate_encoded(x)
        chain.evaluate_encoded(x)
        assert chain.failed_devices() == ("g1",)  # held out for the rest of this go
        notes = chain.drain_notes()
        held = [n for n in notes if "kept out until the next go" in n]
        assert len(held) == 1, notes
        chain.begin_search()
        chain.evaluate_encoded(x)
        assert chain.failed_devices() == ()
    finally:
        stuck.release.set()
        chain.close()


def test_chain_before_arming_waits_out_slow_calls_up_to_the_startup_timeout() -> None:
    """Warm-up calls compile: until ``arm`` a call may take up to the startup
    timeout, and only one past it is dropped ("at startup")."""
    slow = _SlowEvery(_direct(), 1.3)
    chain = DegradingMultiGPUDispatcher([_direct(), slow], labels=["g0", "g1"],
                                        startup_timeout_s=3.0)
    x = np.zeros((1, 146, 8, 8), dtype=np.float32)
    try:
        for _ in range(2):
            chain.evaluate_encoded(x)
        assert slow.calls >= 1
        assert chain.failed_devices() == ()  # 1.3 s < 3 s startup timeout
        hung_dev = _AllFaulty(_direct(), mode="hang", after=0)
        stuck = DegradingMultiGPUDispatcher(
            [_direct(), hung_dev], labels=["g0", "g1"], startup_timeout_s=0.5,
        )
        try:
            for _ in range(2):
                stuck.evaluate_encoded(x)
            assert stuck.failed_devices() == ("g1",)
            assert "stopped answering at startup" in stuck.drain_notes()[0]
        finally:
            hung_dev.release.set()
            stuck.close()
    finally:
        chain.close()


def test_chain_compiled_warmup_drops_a_hung_device_and_verifies_the_survivor(capsys) -> None:
    """``_warmup_evaluator`` over a padding chain where one device hangs during
    warm-up: that device is dropped, every shape runs on the survivor, the
    strict check covers the survivor only and passes, and the chain is armed."""
    good = _direct(64, pad=True)
    hung = _AllFaulty(_direct(64, pad=True), mode="hang", after=0)
    chain = DegradingMultiGPUDispatcher([good, hung], labels=["g0", "g1"], startup_timeout_s=0.5)
    try:
        uci_main._warmup_evaluator(chain, n_devices=2, input_extra_features="v1", max_batch=64)
        assert chain.failed_devices() == ("g1",)
        assert chain.armed
        assert good.shape_ledger.armed
        out_text = capsys.readouterr().out
        verified = [ln for ln in out_text.splitlines() if ln.startswith("info string warmup verified:")]
        assert len(verified) == 1, out_text
        assert "first-seen-shape guard armed" in verified[0]
    finally:
        hung.release.set()
        chain.close()


def test_chain_engine_raise_mid_search_keeps_playing_on_the_survivor(out) -> None:
    bad = _AllFaulty(_direct(), mode="raise", after=3)
    engine, worker, _chain = _chain_engine([_direct(), bad])
    try:
        move, _ = _go(engine, out, "e2e4", "go movetime 1500", 1)
        assert bad.entered.is_set()
        assert _legal_after("e2e4", move)
        assert not any(m in out.text() for m in _UNSEARCHED), out.text()
        assert len(_lines(out, "multi-GPU DEGRADED")) == 1
        move2, _ = _go(engine, out, "e2e4 e7e5", "go nodes 300", 2)
        assert _legal_after("e2e4 e7e5", move2)
        assert not any(m in out.text() for m in _UNSEARCHED), out.text()
        assert len(_lines(out, "multi-GPU degraded: 1 of 2 devices")) >= 1  # per-go line
        assert worker.failed_pool_devices() == ("cpu:1",)
        engine.dispatch(parse_command("isready"))
        assert "multi-GPU degraded: 1 of 2 devices" in out.text().rsplit("readyok", 1)[0].splitlines()[-1]
    finally:
        engine.close()
        worker.close()


def test_chain_engine_stall_mid_search_is_searched_and_quits_clean(out) -> None:
    stuck = _AllFaulty(_direct(), mode="hang", after=3)
    engine, worker, _chain = _chain_engine([_direct(), stuck])
    try:
        move, took = _go(engine, out, "g1f3", "go movetime 2500", 1)
        assert stuck.entered.is_set()
        assert _legal_after("g1f3", move)
        assert took < 2.5 + 0.5
        assert not any(m in out.text() for m in _UNSEARCHED), out.text()
        assert "stopped answering" in _lines(out, "multi-GPU DEGRADED")[0]
        move2, _ = _go(engine, out, "g1f3 g8f6", "go nodes 200", 2)
        assert _legal_after("g1f3 g8f6", move2)
        assert worker.has_wedged_device_threads()  # quit hard-exits instead of finalizing
    finally:
        stuck.release.set()
        engine.close()
        worker.close()


def test_chain_engine_every_device_failing_still_answers_once(out) -> None:
    engine, worker, _chain = _chain_engine([
        _AllFaulty(_direct(), mode="raise", after=1), _AllFaulty(_direct(), mode="raise", after=1),
    ])
    try:
        move, _ = _go(engine, out, "d2d4", "go movetime 800", 1)
        assert _legal_after("d2d4", move)
        move2, _ = _go(engine, out, "d2d4 d7d5", "go nodes 100", 2)
        assert _legal_after("d2d4 d7d5", move2)
        assert out.text().count("bestmove ") == 2
        assert "bestmove_fallback_used" in out.text()
    finally:
        engine.close()
        worker.close()


# ── 2. root-parallel Gumbel ─────────────────────────────────────────────────


def _rpg_engine(groups: list[Any]) -> tuple[Engine, SearchWorker]:
    worker = SearchWorker(
        _evaluator(), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    worker.configure_device_notes(engine_module.emit_info_string_unless_silenced, ())
    worker.install_root_parallel_gumbel(groups, gather=8, as_factories=False)
    return Engine(worker, options=EngineOptions()), worker


@pytest.mark.parametrize("mode", ["raise", "hang"])
@pytest.mark.parametrize("go", ["go movetime 1500", "go nodes 2000"])
def test_rpg_device_failure_retires_rpg_and_the_move_is_searched(out, mode, go) -> None:
    bad = _Faulty(_evaluator(), mode=mode, after=3)
    engine, worker = _rpg_engine([_evaluator(), bad])
    try:
        assert worker.realized_search_path() == "rpg"
        move, _ = _go(engine, out, "e2e4", go, 1, timeout=20)
        assert bad.entered.is_set()
        assert _legal_after("e2e4", move)
        assert not any(m in out.text() for m in _UNSEARCHED), out.text()
        notes = _lines(out, "root-parallel Gumbel device #1")
        assert len(notes) == 1
        assert notes[0].startswith("info string WARNING multi-GPU DEGRADED")
        assert worker.realized_search_path() != "rpg"
        move2, _ = _go(engine, out, "e2e4 e7e5", "go nodes 200", 2)
        assert _legal_after("e2e4 e7e5", move2)
        assert not any(m in out.text() for m in _UNSEARCHED), out.text()
        assert _lines(out, "multi-GPU degraded: root-parallel Gumbel retired")  # per-go line
        assert "root-parallel Gumbel" in worker.failed_pool_devices()
    finally:
        bad.release.set()
        engine.close()
        worker.close()


def test_rpg_startup_hang_fails_construction_instead_of_hanging() -> None:
    def hang() -> Any:
        threading.Event().wait(30)
        return _evaluator()

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="made no progress"):
        RootParallelGumbelPool(
            RootParallelGumbelConfig(n_groups=2, gather=8),
            GumbelConfig(input_extra_features="v1"),
            evaluator_factories=[_evaluator, hang],
            startup_timeout_s=0.5,
        )
    assert time.monotonic() - started < 5.0


# ── 3. startup hang timeout ─────────────────────────────────────────────────


def test_pool_startup_hang_starts_degraded_and_the_hung_build_retires_later() -> None:
    release = threading.Event()
    built: list[Any] = []

    def hang() -> Any:
        release.wait(30)
        ev = _evaluator()
        built.append(ev)
        return ev

    started = time.monotonic()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1"),
                           degrade_on_startup_failure=True, startup_timeout_s=0.5),
        evaluator_factories=[_evaluator, hang],
    )
    try:
        assert time.monotonic() - started < 3.0
        assert pool.n_devices == 1
        assert pool.failed_devices() == ("g1",)
        note = pool.drain_notes()[0]
        assert "g1 failed at startup" in note
        assert "made no progress" in note
        assert _run(pool, 100)[0] == 100  # the survivor searches
        pool.close()
        assert pool._threads[1] in pool.live_worker_threads()  # still in its factory
        release.set()  # the hung build returns: it retires what it built and parks
        assert _wait_for(lambda: _ended_or_parked(pool._threads[1]), timeout=10)
        assert pool._evals[1] is None
    finally:
        release.set()
        pool.close()


def test_pool_startup_every_device_hung_raises() -> None:
    def hang() -> Any:
        threading.Event().wait(30)
        return _evaluator()

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="made no progress"):
        MultiGpuPucvPool(
            MultiGpuPucvConfig(n_gpus=2, gather=8, degrade_on_startup_failure=True,
                               startup_timeout_s=0.5),
            evaluator_factories=[hang, hang],
        )
    assert time.monotonic() - started < 4.0


def test_a_device_waiting_for_the_compile_turn_is_not_timed_out() -> None:
    """Device g1 holds the compile turn and hangs; g2 is queued behind it. Only
    g1 is dropped: its turn is revoked and g2 then builds. (Before the turn
    was revocable, every device queued behind a hung one was lost with it.)"""
    turn = startup_progress.StartupTurn()

    def hang_holding_turn() -> Any:
        with turn:
            threading.Event().wait(30)
        return _evaluator()

    def wait_for_turn() -> Any:
        time.sleep(0.2)  # g1 takes the turn first
        with turn:
            startup_progress.beat()
            return _evaluator()

    started = time.monotonic()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=3, gather=8, device_labels=("g0", "g1", "g2"),
                           degrade_on_startup_failure=True, startup_timeout_s=1.0),
        evaluator_factories=[_evaluator, hang_holding_turn, wait_for_turn],
    )
    try:
        # g2 got the turn when g1 was timed out, not when g1's 30 s hang ended.
        assert time.monotonic() - started < 10.0
        assert pool.failed_devices() == ("g1",)
        assert pool.n_devices == 2
    finally:
        pool.close()


def test_startup_timeout_flag_reaches_every_pool_install() -> None:
    """The consumer receives the value: ``--device-startup-timeout`` parses
    (0 = unbounded), ``_build_engine`` hands it to the worker, and the pool's
    own config carries it (read back from the installed pool)."""
    parser = uci_main._build_parser()
    assert parser.parse_args(["--checkpoint", "x"]).device_startup_timeout_s == pytest.approx(
        startup_progress.DEFAULT_DEVICE_STARTUP_TIMEOUT_S,
    )
    assert parser.parse_args(["--checkpoint", "x", "--device-startup-timeout", "0"]
                             ).device_startup_timeout_s is None
    assert parser.parse_args(["--checkpoint", "x", "--device-startup-timeout", "42.5"]
                             ).device_startup_timeout_s == pytest.approx(42.5)
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "x", "--device-startup-timeout", "-1"])
    engine = uci_main._build_engine(
        evaluator=_evaluator(), primary_device="cpu", chunk_sims=64, topk=8,
        n_walkers=1, vloss_weight=3, walker_gather=1, pucv_vloss_mode=0, max_batch=64,
        vl_gather=8, eval_cache_entries=0, use_multi_gpu_pucv=True, devices=("cpu", "cpu"),
        input_extra_features="v1",
        rebuild_multi_gpu_pucv_factories=lambda _mb, _g: [_evaluator, _evaluator],
        device_startup_timeout_s=42.5,
    )
    try:
        worker = engine._worker
        assert worker._pucv_pool is not None
        assert worker._pucv_pool._cfg.startup_timeout_s == pytest.approx(42.5)
    finally:
        engine.close()


def test_builder_watchdog_reports_a_stuck_startup() -> None:
    ready = threading.Event()
    started = time.monotonic()
    orig = startup_progress.BUILDER_TIMEOUT_GRACE_S
    startup_progress.BUILDER_TIMEOUT_GRACE_S = 0.0
    try:
        stalled = uci_main._await_builder(ready, 0.5)
    finally:
        startup_progress.BUILDER_TIMEOUT_GRACE_S = orig
    assert isinstance(stalled, TimeoutError)
    assert "made no progress" in str(stalled)
    assert time.monotonic() - started < 3.0
    ready.set()
    assert uci_main._await_builder(ready, 0.5) is None


# ── 4. searchmoves beside a pool ────────────────────────────────────────────


@pytest.mark.parametrize(("mode", "after"), [("raise", 1), ("raise", 0), ("hang", 1), ("hang", 0)])
def test_searchmoves_keeps_its_restriction_when_the_primary_fails(out, mode, after) -> None:
    primary = _AllFaulty(_evaluator(), mode=mode, after=after)
    engine, worker = _engine_with_pool([_evaluator(), _evaluator()], primary=primary)
    try:
        move, _ = _go(engine, out, "e2e4", "go movetime 2500 searchmoves a7a6 h7h6", 1)
        assert primary.entered.is_set()
        assert move in ("a7a6", "h7h6"), out.text()
        assert not any(m in out.text() for m in _UNSEARCHED), out.text()
        notes = _lines(out, "primary (root-eval) evaluator")
        assert notes, out.text()
        assert notes[0].startswith("info string WARNING multi-GPU DEGRADED"), out.text()
        move2, _ = _go(engine, out, "e2e4 a7a6", "go nodes 300 searchmoves a2a3 h2h3", 2)
        assert move2 in ("a2a3", "h2h3")
        assert not any(m in out.text() for m in _UNSEARCHED), out.text()
        assert len(_lines(out, "WARNING multi-GPU DEGRADED: the primary")) == 1
        assert "primary" in worker.failed_pool_devices()
    finally:
        primary.release.set()
        engine.close()
        worker.close()


def test_a_primary_stall_on_the_root_eval_moves_root_evals_to_the_pool(out) -> None:
    primary = _AllFaulty(_evaluator(), mode="hang", after=0)
    engine, worker = _engine_with_pool([_evaluator(), _evaluator()], primary=primary)
    try:
        move, _ = _go(engine, out, "d2d4", "go movetime 2500", 1)
        assert _legal_after("d2d4", move)
        assert not any(m in out.text() for m in _UNSEARCHED), out.text()
        assert "stopped answering" in _lines(out, "WARNING multi-GPU DEGRADED: the primary")[0]
    finally:
        primary.release.set()
        engine.close()
        worker.close()


def test_guarded_primary_searchmoves_is_identical_to_the_unguarded_search() -> None:
    """The healthy path: a ``searchmoves`` chunk through the guard (copying
    entry, primary's batch cap as the leaf cap) visits exactly as the direct
    primary does."""
    model = _model()

    def worker() -> SearchWorker:
        from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
        ev = ThreadSafeGPUDispatcher(DirectGPUEvaluator(
            model, device="cpu", max_batch=64, use_amp=False, n_slots=2))
        return SearchWorker(
            ev, device="cpu",
            gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=96, add_noise=False),
            chunk_sims=96, n_walkers=1,
        )

    board = chess.Board()
    allowed = {int(move_to_index(chess.Move.from_uci(u), board)) for u in ("e2e4", "d2d4", "g1f3")}
    results = []
    for guard in (False, True):
        w = worker()
        try:
            w._ensure_root_eval_cached(board, None)
            runner = DeviceCallRunner("primary", bootstrap=False)
            ev = GuardedPrimaryEvaluator(
                w._evaluator, lambda fn, r=runner: r.call(fn, limit_s=30.0),
            ) if guard else None
            w._run_gumbel_chunk(96, board, None, allowed, evaluator=ev)
            assert w._tree is not None
            assert w._root_id is not None
            actions, visits = w._tree.get_children_visits(w._root_id)
            results.append(sorted(zip(actions.tolist(), visits.tolist())))
            runner.close(1.0)
        finally:
            w.close()
    assert results[0] == results[1]
    assert sum(v for _, v in results[0]) > 0


# ── 5. self-check ───────────────────────────────────────────────────────────


def _fake_engine_command(lines: list[str]):
    import sys

    script = "import sys\n" + "".join(
        f"print({ln!r}, flush=True)\n" for ln in lines
    )
    body = (
        script
        + "for raw in sys.stdin:\n"
        "    cmd = raw.strip()\n"
        "    if cmd == 'uci': print('uciok', flush=True)\n"
        "    elif cmd == 'isready': print('readyok', flush=True)\n"
        "    elif cmd.startswith('go'):\n"
        "        print('bestmove e2e4', flush=True)\n"
        "    elif cmd == 'quit': break\n"
    )

    def command(_ckpt: str, _opts: Any, _cache: str) -> list[str]:
        return [sys.executable, "-c", body]

    return command


@pytest.mark.parametrize(("lines", "fast", "verdict"), [
    ([], False, "never printed"),
    (["info string warmup verified: cpu ran all 8 batch shapes; first-seen-shape guard armed"],
     False, None),
    (["info string warmup verified: cpu ran all 8 batch shapes; first-seen-shape guard armed",
      "info string WARNING first-seen batch shape after warmup: device=cpu entry=x batch=48"],
     False, "missed"),
    ([], True, None),
])
def test_compiled_selfcheck_requires_the_warmup_verification(lines, fast, verdict) -> None:
    from chess_anti_engine.uci.selfcheck import CheckFailed, SelfcheckOptions, check_uci

    opts = SelfcheckOptions(checkpoint="unused", fast=fast, uci_timeout_s=30.0)
    if verdict is None:
        detail = check_uci(opts, command=_fake_engine_command(lines))
        if not fast:
            assert "bucket warm-up verified on 1 evaluator(s)" in detail
        else:
            assert "run without --fast before an event" in detail
    else:
        with pytest.raises(CheckFailed, match=verdict):
            check_uci(opts, command=_fake_engine_command(lines))


# ── 6. flap limit on the pool ───────────────────────────────────────────────


def test_pool_flapping_device_is_held_out_until_the_next_search(out) -> None:
    """A device whose calls keep taking 1.3 s (over the 1 s idle floor): within
    one go it is dropped at most DEVICE_FLAP_LIMIT times, restored once, then
    held with one line; the next go re-admits it."""
    slow = _SlowEvery(_evaluator(), 1.3)
    engine, worker = _engine_with_pool([_evaluator(), slow])
    try:
        move, _ = _go(engine, out, "e2e4", "go movetime 6000", 1, timeout=30)
        assert _legal_after("e2e4", move)
        go1 = out.text()
        assert len(_lines(out, "multi-GPU DEGRADED")) == DEVICE_FLAP_LIMIT, go1
        assert len(_lines(out, "multi-GPU RESTORED")) == DEVICE_FLAP_LIMIT - 1, go1
        held = _lines(out, "kept out until the next go")
        assert len(held) == 1, go1
        assert go1.index(held[0]) < go1.index(f"bestmove {move}")
        _go(engine, out, "e2e4 e7e5", "go movetime 1500", 2, timeout=30)
        assert len(_lines(out, "multi-GPU RESTORED")) == DEVICE_FLAP_LIMIT  # back at the next go
    finally:
        engine.close()
        worker.close()


def test_pool_flap_counter_resets_per_search() -> None:
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[_evaluator(), _evaluator()],
    )
    try:
        with pool._cv:
            pool._live[1] = False
            pool._restorable[1] = True
            pool._stall_drops_this_search[1] = DEVICE_FLAP_LIMIT
        assert pool.restore_recovered_devices() == ()
        assert pool.restore_recovered_devices() == ()
        assert len([n for n in pool.drain_notes() if "kept out" in n]) == 1
        pool.begin_search()
        assert pool.restore_recovered_devices() == ("g1",)
    finally:
        pool.close()


def test_runner_reports_stalls_and_late_results() -> None:
    runner = DeviceCallRunner("x", bootstrap=False)
    gate = threading.Event()
    try:
        assert runner.call(lambda: 7, limit_s=1.0) == 7
        with pytest.raises(DeviceStalled):
            runner.call(gate.wait, limit_s=0.2)
        assert runner.wedged()
        assert not runner.restorable
        with pytest.raises(DeviceStalled):
            runner.call(lambda: 1, limit_s=1.0)
        gate.set()
        assert _wait_for(lambda: runner.restorable)
        runner.restore()
        assert runner.call(lambda: 3, limit_s=1.0) == 3
    finally:
        gate.set()
        runner.close(1.0)


def test_one_slow_call_under_a_survivor_is_kept_by_the_pool() -> None:
    """Regression guard for the flap counter: a single 0.6 s hiccup is not a drop."""
    slow = _OnceSlow(_evaluator(), k=2, delay=0.6)
    slow.warm_steady_call_s = 0.017
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[_evaluator(), slow],
    )
    try:
        assert _run(pool, 300)[0] == 300
        assert pool._stall_drops_this_search == [0, 0]
    finally:
        pool.close()



# ── review round 1: the wedge latch, RPG primary / GPU0, primary restore ───


def test_wedge_latch_parks_an_ending_worker_for_the_rest_of_the_session() -> None:
    """Once a thread has been marked wedged (a device fault), a worker
    wrapped by ``wedge.parking`` does not end: it parks, and stays parked
    after the wedge clears (only the test-session reset releases it)."""
    release = threading.Event()

    def stuck() -> None:
        release.wait(30)
        wedge.clear_wedged(threading.current_thread())

    hung = threading.Thread(target=stuck, daemon=True)
    hung.start()
    wedge.mark_wedged(hung, "test device call", device="t1")
    ended = threading.Event()
    worker = threading.Thread(target=wedge.parking(lambda: None), daemon=True)
    worker.start()
    try:
        time.sleep(0.3)
        assert worker.is_alive()  # parked, not ended
        assert worker in wedge.parked_threads()
        assert wedge.wedged_earlier("t1") == "test device call"
        release.set()
        assert _wait_for(lambda: not wedge.any_wedged(), timeout=5)
        time.sleep(1.2)  # past the latch's 1 s re-check
        assert worker in wedge.parked_threads()
        wedge.reset_session_for_tests()
        assert _wait_for(lambda: not worker.is_alive(), timeout=5)
        ended.set()
    finally:
        release.set()
    assert ended.is_set()


def test_a_raise_alone_makes_ending_workers_park_and_join_skip() -> None:
    """Round-2 BLOCKER: no wedge is needed. After a device fault that is only
    a raise, ending workers park and ``join_unless_wedged`` does not wait."""
    assert wedge.device_fault() is None
    first = threading.Thread(target=wedge.parking(lambda: None), daemon=True)
    first.start()
    first.join(5)
    assert not first.is_alive()  # before any fault, workers end normally
    wedge.note_device_fault("device g1 raised")
    assert wedge.device_fault() == "device g1 raised"
    worker = threading.Thread(target=wedge.parking(lambda: None), daemon=True)
    worker.start()
    assert _wait_for(lambda: worker in wedge.parked_threads(), timeout=5)
    started = time.monotonic()
    wedge.join_unless_wedged(worker, 3.0)
    assert time.monotonic() - started < 1.0
    assert worker.is_alive()


def test_pool_close_with_a_wedged_worker_parks_the_healthy_one() -> None:
    """``close`` while a worker is stuck inside a device call: the healthy
    worker parks (no join wait), the stuck one stays; once the call comes
    back it parks too (the session had a device fault)."""
    hung = _Faulty(_evaluator(), mode="hang", after=1)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[_evaluator(), hung],
    )
    try:
        _run(pool, 200)
        assert pool.failed_devices() == ("g1",)
        assert wedge.any_wedged()
        started = time.monotonic()
        pool.close()
        assert time.monotonic() - started < 1.9  # no 2 s join of a parking worker
        assert _wait_for(lambda: pool._threads[0] in wedge.parked_threads(), timeout=5)
        assert pool._threads[1].is_alive()
    finally:
        hung.release.set()
        pool.close()
    assert _wait_for(lambda: all(_ended_or_parked(th) for th in pool._threads), timeout=10)
    assert not wedge.any_wedged()


def test_guarded_primary_reports_the_inner_batch_cap() -> None:
    """D8: the guard advertises the primary's own cap (the in-place path's),
    so a guarded searchmoves search has the same leaf shape."""
    inner = _direct(48)
    guarded = GuardedPrimaryEvaluator(inner, lambda fn: fn())
    assert guarded.search_leaf_row_cap == 48

    class _Capped:
        search_leaf_row_cap = 24

    assert GuardedPrimaryEvaluator(_Capped(), lambda fn: fn()).search_leaf_row_cap == 24


def test_pool_evaluate_rows_with_idle_limit_drops_a_stuck_worker_in_about_a_second() -> None:
    """D9: ``evaluate_rows(idle_limit=True)`` (the failed-primary searchmoves
    batch) drops a worker stuck in that call at the 1 s idle-survivor
    threshold, not the >= 5 s mid-search one, and the next worker answers."""
    hung = _Faulty(_evaluator(), mode="hang", after=0)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[hung, _evaluator()],
    )
    try:
        x = np.zeros((2, 146, 8, 8), dtype=np.float32)
        started = time.monotonic()
        pol, wdl = pool.evaluate_rows(x, None, idle_limit=True)
        took = time.monotonic() - started
        assert pol.shape[0] == 2
        assert wdl.shape[0] == 2
        assert took < 3.5, took
        assert pool.failed_devices() == ("g0",)
    finally:
        hung.release.set()
        pool.close()


def test_rpg_abandoning_a_group_stops_the_job() -> None:
    """D10: abandoning a stuck group stops the running job, so the other
    groups leave the half-updated tree instead of working on under the
    caller's classic search."""
    pool = RootParallelGumbelPool(
        RootParallelGumbelConfig(n_groups=2, gather=8),
        GumbelConfig(input_extra_features="v1"),
        evaluator_factories=[_evaluator, _evaluator],
    )
    try:
        job = threading.Event()
        pool._job_stop = job  # pyright: ignore[reportAttributeAccessIssue]
        pool._abandon_group(1, 2.0, 1.0)
        assert job.is_set()
        assert 1 in pool._abandoned
        assert isinstance(pool._errors[0], RPGDeviceFailure)
    finally:
        pool.close()


class _RaiseAll:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def evaluate_encoded(self, *_a: Any, **_k: Any) -> Any:
        raise RuntimeError("CUDA error: device lost (injected)")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _HangEncoded:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.release = threading.Event()

    def evaluate_encoded(self, *a: Any, **k: Any) -> Any:
        self.release.wait(30)
        return self._inner.evaluate_encoded(*a, **k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.mark.parametrize("bad", ["raise", "hang"])
def test_rpg_evaluate_rows_runs_on_a_surviving_group(bad) -> None:
    """``RootParallelGumbelPool.evaluate_rows`` (the classic path after a
    GPU0 retirement, root evals after a primary failure) runs on a group's
    own thread; a group that raises is set aside, one that stalls is
    abandoned (and marked wedged), and the other group answers."""
    holder: list[Any] = []

    def bad_factory() -> Any:
        ev = _RaiseAll(_evaluator()) if bad == "raise" else _HangEncoded(_evaluator())
        holder.append(ev)
        return ev

    pool = RootParallelGumbelPool(
        RootParallelGumbelConfig(n_groups=2, gather=8),
        GumbelConfig(input_extra_features="v1"),
        evaluator_factories=[bad_factory, _evaluator],
        devices=["cpu", "cpu"],
    )
    try:
        x = np.zeros((3, 146, 8, 8), dtype=np.float32)
        # The shared queue may hand a call to either group: call until the bad
        # group has taken one (and been set aside); every call still answers.
        for _ in range(20):
            pol, wdl = pool.evaluate_rows(x, None, idle_limit=True)
            assert pol.shape[0] == 3
            assert wdl.shape[0] == 3
            if 0 not in pool.usable_groups():
                break
        assert pool.usable_groups() == [1]
        if bad == "hang":
            assert 0 in pool._abandoned
            assert wedge.wedged_earlier("cpu#0") is not None
        else:
            assert 0 in pool._rows_failed
        assert any("cpu#0" in note for note in pool.drain_rows_notes())
    finally:
        for ev in holder:
            if isinstance(ev, _HangEncoded):
                ev.release.set()
        pool.close()


class _SlowOnceEncoded:
    """A primary whose ``k``-th ``evaluate_encoded`` takes ``delay`` s, then
    answers cleanly (a stall that recovers)."""

    def __init__(self, inner: Any, *, k: int, delay: float) -> None:
        self._inner = inner
        self.k = k
        self.delay = delay
        self.calls = 0

    def evaluate_encoded(self, *a: Any, **k: Any) -> Any:
        self.calls += 1
        if self.calls == self.k:
            time.sleep(self.delay)
        return self._inner.evaluate_encoded(*a, **k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_primary_is_warmed_at_install_and_restored_after_a_clean_stall(out) -> None:
    """MINOR 3: the primary's call thread runs one measured call at install
    (``warm_primary_runner``, which the UCI install sites call before
    readyok), seeding its stall limit; a primary dropped for a stall whose
    late call comes back clean is re-admitted at the next go, announced."""
    primary = _SlowOnceEncoded(_evaluator(), k=2, delay=2.0)  # call 1 = the warm call
    engine, worker = _engine_with_pool([_evaluator(), _evaluator()], primary=primary)
    try:
        assert primary.calls == 0  # the library install itself makes no call
        worker.warm_primary_runner()
        runner = worker._primary_runner
        assert runner is not None
        assert runner.warmed
        assert primary.calls == 1
        assert runner.slowest_call_s > 0.0
        move, _ = _go(engine, out, "e2e4", "go movetime 1500", 1)
        assert _legal_after("e2e4", move)
        assert worker._primary_failed
        assert _wait_for(lambda: runner.restorable, timeout=5)
        move2, _ = _go(engine, out, "e2e4 e7e5", "go movetime 800", 2)
        assert _legal_after("e2e4 e7e5", move2)
        assert not worker._primary_failed
        assert _lines(out, "multi-GPU RESTORED: the primary")
    finally:
        engine.close()
        worker.close()


def test_a_device_that_wedged_earlier_is_left_out_of_a_rebuilt_pool() -> None:
    """MINOR 6: a pool rebuilt after a device wedged this session does not
    build on it (that build could hang until the startup timeout): the
    device is dropped at once, announced, and the rest start."""
    wedge.mark_wedged(None, "device g1 call", device="g1")
    built: list[int] = []

    def factory(k: int) -> Any:
        built.append(k)
        return _evaluator()

    started = time.monotonic()
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1"),
                           degrade_on_startup_failure=True, startup_timeout_s=1800.0),
        evaluator_factories=[lambda: factory(0), lambda: factory(1)],
    )
    try:
        assert time.monotonic() - started < 10
        assert built == [0]
        assert pool.failed_devices() == ("g1",)
        assert "left out: it wedged earlier this session" in pool.drain_notes()[0]
    finally:
        pool.close()


def test_rpg_reinstall_after_a_wedge_stays_retired() -> None:
    """A root-parallel Gumbel rebuild after one of its devices wedged this
    session keeps the classic path (announced) instead of building on it."""
    wedge.mark_wedged(None, "RPG device cpu#1 step", device="cpu#1")
    worker = SearchWorker(
        _evaluator(), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    try:
        worker.install_root_parallel_gumbel(
            [_evaluator, _evaluator], gather=8, as_factories=True, devices=["cpu", "cpu"],
        )
        assert worker._rpg_pool is None
        assert "stays retired" in (worker._rpg_retired_reason or "")
        assert "cpu#1" in (worker.pool_degraded_summary() or "")
    finally:
        worker.close()


def test_a_revoked_compile_turn_holder_still_building_blocks_compiling() -> None:
    """MINOR 5: after a holder of the compile turn is timed out and revoked,
    ``revoked_holder_running`` stays True while it is still inside its build
    (it may wake up and keep compiling), so later devices build eager; it
    turns False once that build exits."""
    turn = startup_progress.StartupTurn()
    ctx = startup_progress.StartupContext("g1")
    inside = threading.Event()
    leave = threading.Event()

    def holder() -> None:
        with startup_progress.startup_context(ctx), turn:
            inside.set()
            leave.wait(30)

    th = threading.Thread(target=holder, daemon=True)
    th.start()
    try:
        assert inside.wait(5)
        assert not turn.revoked_holder_running()
        ctx.revoke_turns()
        assert not turn.locked()  # handed on: the next device may proceed ...
        assert turn.revoked_holder_running()  # ... but must not compile
        leave.set()
        th.join(5)
        assert not turn.revoked_holder_running()
    finally:
        leave.set()


def test_prebuilt_rpg_evaluators_receive_the_startup_timeout() -> None:
    """MINOR 7: the ``evaluators=`` (prebuilt) root-parallel Gumbel install
    hands the device startup timeout to the pool, like the factory path."""
    worker = SearchWorker(
        _evaluator(), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    try:
        worker.set_device_startup_timeout(123.0)
        worker.install_root_parallel_gumbel([_evaluator(), _evaluator()], gather=8, as_factories=False)
        assert worker._rpg_pool is not None
        assert worker._rpg_pool._startup_timeout_s == 123.0
    finally:
        worker.close()


def test_a_stalled_call_that_returns_clears_its_wedge() -> None:
    """A worker abandoned in a slow call that then answers is no longer
    wedged (it stays alive, parked for restore): otherwise every ending
    worker would park and quit would hard-exit for nothing."""
    slow = _OnceSlow(_evaluator(), k=1, delay=1.5)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[_evaluator(), slow],
    )
    try:
        _run(pool, 100)
        assert pool.failed_devices() == ("g1",)
        assert _wait_for(lambda: pool._restorable[1], timeout=5)
        assert pool._threads[1].is_alive()
        assert not wedge.any_wedged()
    finally:
        pool.close()


@pytest.mark.parametrize("late", ["return", "raise"])
def test_a_device_call_runner_stall_that_ends_clears_its_wedge(late: str) -> None:
    """Round-2 N3: the chain's per-device call threads and the primary's call
    thread are both ``DeviceCallRunner``s. A call abandoned for a stall marks
    the thread wedged; once that call ends (late answer or late raise) the mark
    must clear, or every later ending worker parks and quit hard-exits for a
    device that is no longer stuck."""
    gate = threading.Event()

    def slow() -> int:
        gate.wait(5)
        if late == "raise":
            raise RuntimeError("late failure")
        return 7

    runner = DeviceCallRunner("chain-or-primary", bootstrap=False)
    try:
        with pytest.raises(DeviceStalled):
            runner.call(slow, limit_s=0.2)
        assert wedge.any_wedged()
        assert runner.wedged()
        gate.set()
        assert _wait_for(lambda: not runner.wedged(), timeout=5)
        assert not wedge.any_wedged()
        assert runner.restorable is (late == "return")
    finally:
        gate.set()
        runner.close(2.0)


class _RecordingEncoded:
    """Primary wrapper recording what ``evaluate_encoded`` is sent (the inner
    tiny net ignores relations)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.sent: list[tuple[np.ndarray, Any]] = []

    def evaluate_encoded(self, x: np.ndarray, relations: Any = None) -> Any:
        self.sent.append((np.array(x, copy=True), relations))
        return self._inner.evaluate_encoded(x)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.mark.parametrize("relations", [False, True])
def test_the_primary_warm_call_sends_real_root_inputs(relations: bool) -> None:
    """Round-2 item 2b: the warm call sends what a real root evaluation sends
    (startpos with this net's history encoding and extra planes, relations
    when the net uses them), so the warmed shape / graph is the real one."""
    import dataclasses

    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard

    primary = _RecordingEncoded(_evaluator())
    engine, worker = _engine_with_pool([_evaluator(), _evaluator()], primary=primary)
    try:
        worker._cfg = dataclasses.replace(worker._cfg, compute_relations=relations)
        worker.warm_primary_runner()
        assert len(primary.sent) == 1
        x, rels = primary.sent[0]
        cb = CBoard.from_board(chess.Board())
        want = encode_cboard(cb, input_history_encoding=worker._cfg.input_history_encoding,
                             input_extra_features=worker._cfg.input_extra_features)
        assert x.shape[0] == 1
        assert np.array_equal(x[0], want)
        if relations:
            assert rels is not None
            assert np.array_equal(rels, cb.compute_relations()[None, ...])
        else:
            assert rels is None
    finally:
        engine.close()
        worker.close()


class _RaiseWhenArmed:
    """Healthy until ``armed`` is set; then, after ``skip`` more calls (the
    move's root evaluation), every ``evaluate*`` raises -- inside the search
    chunk, not at the root."""

    def __init__(self, inner: Any, skip: int = 1) -> None:
        self._inner = inner
        self.armed = threading.Event()
        self.raised = 0
        self._skip = skip

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if callable(attr) and name.startswith("evaluate"):
            def gated(*a: Any, **k: Any) -> Any:
                if self.armed.is_set() and self._skip > 0:
                    self._skip -= 1
                elif self.armed.is_set():
                    self.raised += 1
                    raise RuntimeError("CUDA error: device lost (injected, second failure)")
                return attr(*a, **k)
            return gated
        return attr


def test_a_second_failure_after_rpg_retirement_is_announced_before_bestmove(out) -> None:
    """Round-2 item 5: root-parallel Gumbel retired after devices[0] failed,
    classic Gumbel carried by a surviving group; when that carrier fails too
    its DEGRADED line comes before that move's bestmove, and the per-go
    summary names the device that carries on now, not the dead one."""
    first = _Faulty(_evaluator(), mode="raise", after=3)
    second = _RaiseWhenArmed(_evaluator())
    engine, worker = _rpg_engine([first, second, _evaluator()])
    try:
        move, _ = _go(engine, out, "e2e4", "go nodes 2000", 1, timeout=20)
        assert _legal_after("e2e4", move)
        assert worker._rpg_rows_pool is not None  # carried by a surviving group
        assert "device #1" in (worker.pool_degraded_summary() or "")
        second.armed.set()
        move2, _ = _go(engine, out, "e2e4 e7e5", "go nodes 300", 2, timeout=20)
        assert _legal_after("e2e4 e7e5", move2)
        assert second.raised >= 1
        lines = out.text().splitlines()
        bestmoves = [i for i, ln in enumerate(lines) if ln.startswith("bestmove")]
        notes = [i for i, ln in enumerate(lines)
                 if "WARNING multi-GPU DEGRADED: root-parallel Gumbel device #1" in ln]
        assert notes, lines
        assert bestmoves[0] < notes[0] < bestmoves[1], lines
        summary = worker.pool_degraded_summary() or ""
        assert "now searching classic Gumbel on device #2" in summary, summary
    finally:
        first.release.set()
        engine.close()
        worker.close()


class _Closeable:
    """Evaluator wrapper that counts ``close`` calls."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_rpg_raise_retirement_parks_the_groups_and_keeps_their_evaluators(out) -> None:
    """Round-2 BLOCKER: a non-devices[0] group that RAISES (no wedge) retires
    root-parallel Gumbel onto the primary. The retired pool's group threads
    park instead of ending and their evaluators are kept referenced, never
    closed (ending them crashed the process)."""
    healthy = _Closeable(_evaluator())
    bad = _Faulty(_evaluator(), mode="raise", after=3)
    engine, worker = _rpg_engine([healthy, bad])
    try:
        pool = worker._rpg_pool
        assert pool is not None
        move, _ = _go(engine, out, "e2e4", "go nodes 2000", 1, timeout=20)
        assert _legal_after("e2e4", move)
        assert worker._rpg_pool is None
        assert worker._rpg_rows_pool is None  # carried by the primary
        fault = wedge.device_fault()
        assert fault is not None
        assert "#1" in fault
        assert _wait_for(lambda: all(th in wedge.parked_threads() for th in pool._threads), timeout=5)
        assert any(kept is pool for kept, _evals in worker._rpg_retired_keepalive)
        assert healthy.closed == 0
    finally:
        bad.release.set()
        engine.close()
        worker.close()


@pytest.mark.parametrize("path", ["pool", "chain", "primary"])
def test_every_device_drop_notes_a_device_fault(path: str) -> None:
    """Every place that drops a device notes the session's device fault, so
    the parking rule and the quit hard-exit apply after a raise alone."""
    assert wedge.device_fault() is None
    if path == "pool":
        pool = MultiGpuPucvPool(
            MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
            evaluators=[_evaluator(), _Faulty(_evaluator(), mode="raise", after=1)],
        )
        try:
            _run(pool, 100)
            assert pool.failed_devices() == ("g1",)
        finally:
            pool.close()
        assert "g1" in (wedge.device_fault() or "")
    elif path == "chain":
        chain = DegradingMultiGPUDispatcher(
            [_direct(), _AllFaulty(_direct(), mode="raise", after=0)], labels=["c0", "c1"],
        )
        try:
            chain.arm()
            x = np.zeros((3, 146, 8, 8), dtype=np.float32)
            for _ in range(3):
                chain.evaluate_encoded(x)
            assert chain.failed_devices() == ("c1",)
        finally:
            chain.close()
        assert "c1" in (wedge.device_fault() or "")
    else:
        engine, worker = _engine_with_pool([_evaluator(), _evaluator()],
                                           primary=_AllFaulty(_evaluator(), mode="raise", after=0))
        try:
            worker._note_primary_failed(RuntimeError("primary lost"))
            assert worker._primary_failed
        finally:
            engine.close()
            worker.close()
        assert wedge.device_fault() == "the primary evaluator failed"


def test_walker_and_owner_threads_park_after_a_device_fault() -> None:
    """Item 7: the persistent walker-pool and CUDA-owner threads run
    evaluator work, so after a device fault they park instead of ending."""
    from chess_anti_engine.inference_dispatcher import CUDAOwnerDispatcher
    from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig

    wedge.note_device_fault("device g1 raised")
    walkers = WalkerPool(WalkerPoolConfig(n_walkers=2, c_puct=1.0, fpu_at_root=0.0,
                                          fpu_reduction=0.0, input_planes=146), _direct())
    owner = CUDAOwnerDispatcher(_direct())
    started = time.monotonic()
    walkers.close()  # returns once each walker parked: no 2 s join per walker
    owner.close()  # no 30 s join of a parked owner
    # Plain joins would take >= 4 s (2 s per walker) and 30 s for the owner.
    assert time.monotonic() - started < 3.5
    threads = [*walkers._threads, owner._owner]
    assert _wait_for(lambda: all(th in wedge.parked_threads() for th in threads), timeout=5)
