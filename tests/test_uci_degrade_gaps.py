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
from chess_anti_engine.uci import startup_progress
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
)
from chess_anti_engine.uci.search import SearchWorker
from tests.test_uci_multi_gpu_degrade import (
    _UNSEARCHED,
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
        release.set()  # the hung build returns: it retires what it built and exits
        assert _wait_for(lambda: not pool._threads[1].is_alive(), timeout=10)
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

