"""One GPU failing in the multi-GPU PUCV pool must not cost the move or the game.

Devices are faked as several CPU evaluators (the pattern of
`test_multi_gpu_pucv_pool.py`); faults are injected by wrapping one evaluator:
it raises (CUDA OOM / device lost), or blocks forever inside a device call.
Pool-level tests pin the mechanics (survivors finish the budget, virtual loss
fully released, one announcement); engine-level tests pin the UCI contract
(a legal bestmove in time, the `info string` once, the next `go` searched).
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

import chess_anti_engine.async_safety as async_safety
import chess_anti_engine.uci.engine as engine_module
import chess_anti_engine.uci.multi_gpu_pucv_pool as pool_module
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.multi_gpu_pucv_pool import (
    DEVICE_DROPPED_NOTE,
    MultiGpuPucvConfig,
    MultiGpuPucvPool,
)
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.selfcheck import UCI_FAILURE_MARKERS, _failure_lines


def _evaluator(max_batch: int = 64) -> Any:
    model = build_model(ModelConfig(
        input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
    ))
    model.eval()
    return ThreadSafeGPUDispatcher(DirectGPUEvaluator(
        model, device="cpu", max_batch=max_batch, use_amp=False, n_slots=2,
    ))


class _Faulty:
    """An evaluator that turns bad on its ``after``-th batch submission.

    ``mode="raise"``: every submission from then on raises (device lost / OOM).
    ``mode="hang"``: that submission blocks until ``release`` is set (a wedged
    device), then answers normally -- the late answer must corrupt nothing.
    ``mode="hang_sync"``: the submission returns, but waiting for its completion
    blocks until ``release`` (a kernel that never finishes).
    ``mode="hang_raise"``: like ``hang``, but when released the call RAISES.
    """

    def __init__(self, inner: Any, *, mode: str, after: int = 0) -> None:
        self._inner = inner
        self.mode = mode
        self.after = after
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _gate(self) -> None:
        with self._lock:
            self.calls += 1
            bad = self.calls > self.after
        if not bad:
            return
        self.entered.set()
        if self.mode == "raise":
            raise RuntimeError("CUDA error: an illegal memory access was encountered (injected)")
        assert self.release.wait(30)
        if self.mode == "hang_raise":
            raise RuntimeError("CUDA error: device lost after a stall (injected)")

    def evaluate_inplace_async(self, *args: Any, **kwargs: Any) -> Any:
        if self.mode == "hang_sync":
            with self._lock:
                self.calls += 1
                bad = self.calls > self.after
            pol, wdl, event = self._inner.evaluate_inplace_async(*args, **kwargs)
            return (pol, wdl, _BlockingEvent(self)) if bad else (pol, wdl, event)
        self._gate()
        return self._inner.evaluate_inplace_async(*args, **kwargs)

    def evaluate_encoded(self, *args: Any, **kwargs: Any) -> Any:
        self._gate()
        return self._inner.evaluate_encoded(*args, **kwargs)


class _BlockingEvent:
    """A completion receipt whose wait never ends until the test releases it.

    ``owner`` is any fake device with ``entered`` / ``release`` events."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def synchronize(self) -> None:
        self._owner.entered.set()
        assert self._owner.release.wait(30)


def _seed_tree() -> tuple[MCTSTree, int, CBoard]:
    tree = MCTSTree()
    tree.reserve(4096, 32768)
    cb = CBoard.from_board(chess.Board())
    rid = tree.add_root(0, 0.0)
    legal = cb.legal_move_indices().astype(np.int32)
    tree.expand(rid, legal, np.full(legal.size, 1.0 / legal.size, dtype=np.float64))
    return tree, rid, cb


def _assert_no_vloss(tree: MCTSTree) -> None:
    for nid in range(tree.node_count()):
        assert tree.get_virtual_loss(nid) == 0, f"virtual loss leaked on node {nid}"


class _OrderedPool(MultiGpuPucvPool):
    """Force worker 0 to own work before worker 1 can drain the remaining budget."""

    def __init__(self, bad: Any, cfg: MultiGpuPucvConfig) -> None:
        self.bad = bad
        self.first_idle = threading.Event()
        self.passes = [0, 0]
        super().__init__(cfg, evaluators=[bad, _evaluator()])

    def _pipeline_until_done(
        self, idx: int, job: Any, evaluator: Any,
        bufs: list[dict[str, np.ndarray]], gather: int,
    ) -> None:
        if idx == 1 and self.passes[idx] == 0:
            assert self.bad.entered.wait(5)
        self.passes[idx] += 1
        super()._pipeline_until_done(idx, job, evaluator, bufs, gather)
        if idx == 1 and self.passes[idx] == 1:
            self.first_idle.set()


@pytest.mark.parametrize(("target", "fails", "finish_before_first_wait"), [
    pytest.param(1, False, False, id="healthy_single_token"),
    pytest.param(3, False, False, id="healthy_cumulative"),
    pytest.param(1, True, False, id="returned_single_token"),
    pytest.param(3, True, False, id="returned_token_cumulative"),
    pytest.param(1, True, True, id="both_initial_passes_finished_before_first_wait"),
])
def test_returned_final_tokens_rewake_idle_survivor_with_cumulative_stats(
    monkeypatch, target, fails, finish_before_first_wait,
) -> None:
    """The survivor is demonstrably asleep BEFORE the final token is returned.

    target=3 also gives it two completed leaves before re-entry, so resetting
    its per-job counters on the second pipeline pass cannot pass this test.
    The healthy control releases the original owner without returning tokens.
    """
    bad = _Faulty(_evaluator(), mode="hang_raise" if fails else "hang")
    pool = _OrderedPool(bad, MultiGpuPucvConfig(n_gpus=2, gather=1, stall_floor_s=30.0))
    tree, rid, cb = _seed_tree()
    stop = threading.Event()
    answers: list[int] = []
    errors: list[BaseException] = []
    coordinator_paused = threading.Event()
    release_coordinator = threading.Event()
    if finish_before_first_wait:
        original_set = pool._wakes[1].set
        first_wake = [True]

        def wake_then_pause() -> None:
            original_set()
            if first_wake[0]:
                first_wake[0] = False
                # run() has woken both actual workers but has not entered
                # its completion condition yet. Let both first passes end.
                coordinator_paused.set()
                assert release_coordinator.wait(5)

        monkeypatch.setattr(pool._wakes[1], "set", wake_then_pause)

    def search() -> None:
        try:
            answers.append(pool.run(tree=tree, root_id=rid, root_cboard=cb,
                                    target_sims=target, stop_event=stop))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=search, daemon=True)
    thread.start()
    try:
        if finish_before_first_wait:
            assert coordinator_paused.wait(5)
        assert pool.first_idle.wait(5)
        with pool._cv:
            assert pool._cv.wait_for(lambda: 1 not in pool._running, timeout=5)
        assert pool._worker_stats[1].leaves == target - 1
        bad.release.set()
        if finish_before_first_wait:
            with pool._cv:
                assert pool._cv.wait_for(lambda: not pool._running, timeout=5)
            assert pool.failed_devices() == ("#0",)
            assert not answers
            release_coordinator.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors
        assert answers == [target]
        assert not stop.is_set()
        stats = pool.last_stats()
        assert stats.target_sims == stats.leaves == target
        assert stats.workers[1].leaves == (target if fails else target - 1)
        assert stats.workers[1].batches == (target if fails else target - 1)
        assert pool.passes[1] == (2 if fails else 1)
        _assert_no_vloss(tree)
        if fails:
            assert pool.failed_devices() == ("#0",)
            assert pool.restore_recovered_devices() == ()
            assert not pool._late_ok[0]
            assert not pool._restorable[0]
        # The next job must reset the counters rather than inherit this job's
        # completed passes. A failed device must never be called again.
        calls = bad.calls
        assert pool.run(tree=tree, root_id=rid, root_cboard=cb,
                        target_sims=4, stop_event=threading.Event()) == 4
        assert pool.last_stats().leaves == 4
        if fails:
            assert pool.last_stats().workers[1].leaves == 4
            assert bad.calls == calls
        _assert_no_vloss(tree)
    finally:
        release_coordinator.set()
        bad.release.set()
        thread.join(timeout=5)
        pool.close()


class _FailureDrainEvent:
    """First wait raises; the failure-drain wait blocks until explicitly released."""

    def __init__(self, *, fails: bool) -> None:
        self.fails = fails
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def synchronize(self) -> None:
        self.calls += 1
        if self.fails and self.calls == 1:
            raise RuntimeError("injected completion failure before blocked cleanup")
        self.entered.set()
        assert self.release.wait(5)


def test_last_quiesced_worker_error_is_not_retried_within_the_job() -> None:
    error = ValueError("injected CPU buffer preparation failure")

    class OnceBadBuffer:
        def __init__(self) -> None:
            self.inner = _evaluator()
            self.calls = 0

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def get_input_buffer(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise error  # No asynchronous submission: the owner stays usable.
            return self.inner.get_input_buffer(*args, **kwargs)

    ev = OnceBadBuffer()
    pool = MultiGpuPucvPool(MultiGpuPucvConfig(n_gpus=1, gather=1), evaluators=[ev])
    tree, rid, cb = _seed_tree()
    try:
        with pytest.raises(ValueError, match="CPU buffer preparation") as caught:
            pool.run(tree=tree, root_id=rid, root_cboard=cb,
                     target_sims=4, stop_event=threading.Event())
        assert caught.value is error
        assert ev.calls == 1
        assert pool.n_devices == 1
        _assert_no_vloss(tree)
        assert pool.run(tree=tree, root_id=rid, root_cboard=cb,
                        target_sims=4, stop_event=threading.Event()) == 4
        assert pool.last_stats().leaves == 4
        _assert_no_vloss(tree)
    finally:
        pool.close()


class _FailureDrainEvaluator:
    def __init__(self, *, fails: bool) -> None:
        self.inner = _evaluator()
        self.event = _FailureDrainEvent(fails=fails)
        self.entered = self.event.entered
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def evaluate_inplace_async(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        policy, wdl, event = self.inner.evaluate_inplace_async(*args, **kwargs)
        if self.calls == 1:
            # Complete the real CPU evaluator before injecting its replacement
            # completion receipt; no model/tree/cache operation is mocked.
            if event is not None:
                event.synchronize()
            return policy, wdl, self.event
        return policy, wdl, event


@pytest.mark.parametrize("route", ["root", "pipeline"])
@pytest.mark.parametrize("fails", [False, True])
def test_failure_drain_is_watchdog_visible_and_never_restorable(monkeypatch, route, fails) -> None:
    """Exercise both real entry paths; a direct hang is the positive control.

    The pool's watchdog clock advances only after the receipt is blocked and
    the healthy pipeline is idle. This pins the 20ms threshold without racing
    a slow CI host's actual CPU forward against a wall-clock deadline.
    """
    now = [0.0]
    monkeypatch.setattr(pool_module, "time", SimpleNamespace(
        monotonic=lambda: now[0], perf_counter=time.perf_counter,
    ))
    monkeypatch.setattr(pool_module, "_RUN_POLL_S", 0.002)
    monkeypatch.setattr(pool_module, "_STOP_STALL_FLOOR_S", 0.02)
    monkeypatch.setattr(pool_module, "_IDLE_SURVIVOR_STALL_FLOOR_S", 0.02)
    bad = _FailureDrainEvaluator(fails=fails)
    pool = _OrderedPool(bad, MultiGpuPucvConfig(
        n_gpus=2, gather=1, stall_floor_s=0.02, eval_cache_entries=8,
    ))
    tree, rid, cb = _seed_tree()
    done = threading.Event()
    errors: list[BaseException] = []
    answers: list[Any] = []

    def search() -> None:
        try:
            if route == "root":
                answers.append(pool.evaluate_root(np.zeros((1, 146, 8, 8), dtype=np.float32)))
            else:
                answers.append(pool.run(tree=tree, root_id=rid, root_cboard=cb,
                                        target_sims=4, stop_event=threading.Event()))
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=search, daemon=True)
    thread.start()
    try:
        assert bad.entered.wait(5)
        if route == "pipeline":
            assert pool.first_idle.wait(5)
            with pool._cv:
                assert pool._cv.wait_for(lambda: 1 not in pool._running, timeout=5)
        assert bad.event.calls == (2 if fails else 1)
        # Advance beyond the configured threshold while keeping cleanup held.
        # Before the fix, only the direct-hang control has a busy marker.
        now[0] = 0.1
        with pool._cv:
            pool._cv.notify_all()
        assert done.wait(5), "failure cleanup escaped the device-call watchdog"
        assert not errors
        assert len(answers) == 1
        assert pool._abandoned[0]
        if route == "pipeline":
            assert answers == [4]
            assert pool.last_stats().leaves == 4
        else:
            assert answers[0][0].shape == (1, 4672)
            assert answers[0][1].shape == (1, 3)
        _assert_no_vloss(tree)
        before = tree.get_children_visits(rid)[1].copy()
        assert pool._cache is not None
        cache_before = pool._cache.stats()
        bad.event.release.set()
        if fails:
            pool._threads[0].join(timeout=5)
            assert not pool._threads[0].is_alive()
            assert not pool._late_ok[0]
            assert not pool._restorable[0]
            assert pool.restore_recovered_devices() == ()
        else:
            assert _wait_for(lambda: pool._restorable[0], timeout=5)
            assert pool.restore_recovered_devices() == ("#0",)
        # Neither late completion nor late cleanup may integrate or populate
        # shared cache, or release the same virtual loss a second time.
        np.testing.assert_array_equal(tree.get_children_visits(rid)[1], before)
        assert pool._cache.stats() == cache_before
        _assert_no_vloss(tree)
    finally:
        bad.event.release.set()
        thread.join(timeout=5)
        pool.close()


# -- pool mechanics -----------------------------------------------------------


def test_raising_device_is_dropped_and_survivors_finish_the_budget() -> None:
    bad = _Faulty(_evaluator(), mode="raise", after=2)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=3, gather=8, device_labels=("cpu:0", "cpu:1", "cpu:2")),
        evaluators=[_evaluator(), bad, _evaluator()],
    )
    try:
        tree, rid, cb = _seed_tree()
        target = 400
        done = pool.run(tree=tree, root_id=rid, root_cboard=cb,
                        target_sims=target, stop_event=threading.Event())
        assert bad.entered.is_set()
        # The dead device's in-flight leaves went back to the budget and the
        # survivors searched them: the chunk completes in full.
        assert done == target
        _, visits = tree.get_children_visits(rid)
        assert int(visits.sum()) >= target - 2 * 8
        _assert_no_vloss(tree)
        assert pool.n_devices == 2
        assert pool.failed_devices() == ("cpu:1",)
        notes = pool.drain_notes()
        assert len(notes) == 1
        assert "cpu:1" in notes[0]
        assert "2 of 3" in notes[0]
        assert pool.drain_notes() == []  # announced once
        # The next run uses only the survivors and completes in full.
        calls = bad.calls
        assert pool.run(tree=tree, root_id=rid, root_cboard=cb,
                        target_sims=200, stop_event=threading.Event()) == 200
        assert bad.calls == calls
        _assert_no_vloss(tree)
    finally:
        pool.close()


def test_every_device_failing_raises_instead_of_hanging() -> None:
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8),
        evaluators=[_Faulty(_evaluator(), mode="raise", after=1),
                    _Faulty(_evaluator(), mode="raise", after=1)],
    )
    try:
        tree, rid, cb = _seed_tree()
        with pytest.raises(RuntimeError, match="illegal memory access"):
            pool.run(tree=tree, root_id=rid, root_cboard=cb,
                     target_sims=400, stop_event=threading.Event())
        _assert_no_vloss(tree)
        assert pool.n_devices == 0
        # Unquiesced submit failures poison the last worker, and with it the
        # pool (as before the degrade path); either way: an immediate raise.
        with pytest.raises(RuntimeError, match=r"every device has failed|poisoned"):
            pool.run(tree=tree, root_id=rid, root_cboard=cb,
                     target_sims=10, stop_event=threading.Event())
    finally:
        pool.close()


@pytest.mark.parametrize("where", ["hang", "hang_sync"])
def test_stuck_device_is_abandoned_and_its_late_answer_corrupts_nothing(monkeypatch, where) -> None:
    """Stuck in the submit call, or in the wait for a submitted batch (with
    leaves already descended under virtual loss): either way the run returns,
    the stuck worker's virtual loss is released once, and its late answer is
    never integrated."""
    monkeypatch.setattr(pool_module, "_STALL_FLOOR_S", 0.3)
    stuck = _Faulty(_evaluator(), mode=where, after=1)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("cpu:0", "cpu:1")),
        evaluators=[_evaluator(), stuck],
    )
    try:
        tree, rid, cb = _seed_tree()
        started = time.monotonic()
        done = pool.run(tree=tree, root_id=rid, root_cboard=cb,
                        target_sims=300, stop_event=threading.Event())
        took = time.monotonic() - started
        assert stuck.entered.is_set()
        assert took < 10.0  # returned on its own, long before the 30 s release
        assert done > 0
        _assert_no_vloss(tree)  # the stuck worker's in-flight vloss was released
        assert pool.failed_devices() == ("cpu:1",)
        notes = pool.drain_notes()
        assert len(notes) == 1
        assert "stopped answering" in notes[0]
        visits_before = int(tree.get_children_visits(rid)[1].sum())
        # The wedged call now returns: the worker must leave without
        # integrating, double-releasing or otherwise touching the tree.
        stuck.release.set()
        time.sleep(0.3)
        _assert_no_vloss(tree)
        assert int(tree.get_children_visits(rid)[1].sum()) == visits_before
        assert pool.run(tree=tree, root_id=rid, root_cboard=cb,
                        target_sims=100, stop_event=threading.Event()) == 100
    finally:
        stuck.release.set()
        pool.close()


def test_one_factory_failing_at_startup_starts_degraded() -> None:
    good = [_evaluator(), _evaluator()]

    def broken() -> Any:
        raise RuntimeError("CUDA out of memory (injected at startup)")

    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=3, gather=8, device_labels=("cuda:0", "cuda:1", "cuda:2"),
                           degrade_on_startup_failure=True),
        evaluator_factories=[lambda: good[0], broken, lambda: good[1]],
    )
    try:
        assert pool.n_devices == 2
        notes = pool.drain_notes()
        assert len(notes) == 1
        assert "cuda:1 failed at startup" in notes[0]
        assert "DEGRADED" in notes[0]
        tree, rid, cb = _seed_tree()
        assert pool.run(tree=tree, root_id=rid, root_cboard=cb,
                        target_sims=64, stop_event=threading.Event()) == 64
    finally:
        pool.close()


def test_every_factory_failing_at_startup_still_refuses() -> None:
    def broken() -> Any:
        raise RuntimeError("no device")

    with pytest.raises(RuntimeError, match="no device"):
        MultiGpuPucvPool(MultiGpuPucvConfig(n_gpus=2, gather=8, degrade_on_startup_failure=True),
                         evaluator_factories=[broken, broken])


def test_a_note_a_silenced_thread_cannot_print_waits_for_the_next_search(out) -> None:
    """A parked (silenced) search thread must not swallow the one-time
    announcement: the emitter reports False and the worker keeps the note."""
    worker = SearchWorker(
        _evaluator(), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False),
        chunk_sims=64,
    )
    worker._device_note_cb = engine_module.emit_info_string_unless_silenced
    worker._device_notes = ["WARNING multi-GPU DEGRADED: device cpu:9 stopped answering"]
    seen: list[list[str]] = []

    def parked() -> None:
        engine_module._SILENCED_THREADS.add(threading.current_thread())
        try:
            worker._flush_device_notes()
            seen.append(list(worker._device_notes))
        finally:
            engine_module._SILENCED_THREADS.discard(threading.current_thread())

    thread = threading.Thread(target=parked)
    thread.start()
    thread.join(5)
    assert seen == [["WARNING multi-GPU DEGRADED: device cpu:9 stopped answering"]]
    assert "cpu:9" not in out.text()
    worker._flush_device_notes()  # the next search's thread prints it, once
    worker._flush_device_notes()
    assert out.text().count("device cpu:9 stopped answering") == 1
    worker.close()


def test_degraded_note_trips_the_selfcheck() -> None:
    line = "info string " + DEVICE_DROPPED_NOTE.format(
        label="cuda:3", what="failed at startup", reason="x", live=3, total=4,
    )
    assert any(marker in line for marker in UCI_FAILURE_MARKERS)
    assert _failure_lines(["readyok", line]) == [line]


# -- engine level: the UCI contract ---------------------------------------------


class _Out:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._cond = threading.Condition()

    def write(self, s: str) -> int:
        with self._cond:
            self.lines.extend(ln for ln in s.splitlines() if ln)
            self._cond.notify_all()
        return len(s)

    def flush(self) -> None:
        pass

    def bestmoves(self) -> list[str]:
        with self._cond:
            return [ln.split()[1] for ln in self.lines if ln.startswith("bestmove ")]

    def wait_bestmoves(self, n: int, timeout: float) -> list[str]:
        with self._cond:
            self._cond.wait_for(
                lambda: sum(ln.startswith("bestmove ") for ln in self.lines) >= n, timeout,
            )
        return self.bestmoves()

    def text(self) -> str:
        with self._cond:
            return "\n".join(self.lines)


@pytest.fixture
def out(monkeypatch) -> _Out:
    capture = _Out()
    monkeypatch.setattr(engine_module, "sys", SimpleNamespace(stdout=capture, stderr=sys.stderr))
    return capture


def _engine_with_pool(pool_evals: list[Any], primary: Any | None = None) -> tuple[Engine, SearchWorker]:
    worker = SearchWorker(
        primary if primary is not None else _evaluator(), device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=64, add_noise=False),
        chunk_sims=64, n_walkers=1,
    )
    worker.install_multi_gpu_pucv(
        pool_evals, gather=8, as_factories=False,
        device_labels=[f"cpu:{k}" for k in range(len(pool_evals))],
        info_string_cb=engine_module.emit_info_string_unless_silenced,
    )
    return Engine(worker, options=EngineOptions()), worker


def _legal_after(moves: str, uci: str) -> bool:
    board = chess.Board()
    for mv in moves.split():
        board.push_uci(mv)
    return chess.Move.from_uci(uci) in board.legal_moves


def _searched_go(engine: Engine, out: _Out, moves: str, movetime: int, n: int) -> tuple[str, float]:
    engine.dispatch(parse_command(f"position startpos moves {moves}"))
    started = time.monotonic()
    engine.dispatch(parse_command(f"go movetime {movetime}"))
    got = out.wait_bestmoves(n, timeout=movetime / 1000 + 20)
    assert len(got) == n
    return got[-1], time.monotonic() - started


def _degraded_lines(out: _Out) -> list[str]:
    return [ln for ln in out.text().splitlines() if "multi-GPU DEGRADED" in ln]


_UNSEARCHED = ("search error", "bestmove_fallback_used", "deadline watchdog", "UNSEARCHED")


def test_engine_device_raising_mid_search_keeps_playing_on_the_rest(out) -> None:
    bad = _Faulty(_evaluator(), mode="raise", after=3)
    engine, worker = _engine_with_pool([_evaluator(), bad, _evaluator()])
    try:
        move, took = _searched_go(engine, out, "e2e4", 1500, 1)
        assert bad.entered.is_set()
        assert _legal_after("e2e4", move)
        assert took < 1.5 + 1.0
        text = out.text()
        assert not any(marker in text for marker in _UNSEARCHED), text
        degraded = _degraded_lines(out)
        assert len(degraded) == 1
        assert degraded[0].startswith("info string WARNING")
        assert "cpu:1" in degraded[0]
        # Subsequent moves: searched on the survivors, no second announcement.
        move2, _ = _searched_go(engine, out, "e2e4 e7e5", 800, 2)
        assert _legal_after("e2e4 e7e5", move2)
        assert not any(marker in out.text() for marker in _UNSEARCHED)
        assert len(_degraded_lines(out)) == 1
        assert worker.failed_pool_devices() == ("cpu:1",)
    finally:
        engine.close()
        worker.close()


def test_engine_every_device_failing_still_answers(out) -> None:
    engine, worker = _engine_with_pool([
        _Faulty(_evaluator(), mode="raise", after=1),
        _Faulty(_evaluator(), mode="raise", after=1),
    ])
    try:
        move, took = _searched_go(engine, out, "d2d4", 1000, 1)
        assert _legal_after("d2d4", move)
        assert took < 1.0 + 1.0
        # A later go is answered too (the existing fallback path), never hangs.
        move2, _ = _searched_go(engine, out, "d2d4 d7d5", 500, 2)
        assert _legal_after("d2d4 d7d5", move2)
        assert "bestmove_fallback_used" in out.text()
    finally:
        engine.close()
        worker.close()


def test_engine_hung_device_answers_by_the_deadline_and_recovers(out, monkeypatch) -> None:
    monkeypatch.setattr(pool_module, "_STALL_FLOOR_S", 0.4)
    stuck = _Faulty(_evaluator(), mode="hang", after=2)
    engine, worker = _engine_with_pool([_evaluator(), stuck])
    try:
        move, took = _searched_go(engine, out, "g1f3", 2500, 1)
        assert stuck.entered.is_set()
        assert _legal_after("g1f3", move)
        assert took < 2.5 + 0.5  # by the deadline (plus scheduling slack)
        assert len(_degraded_lines(out)) == 1
        assert "stopped answering" in _degraded_lines(out)[0]
        # The search was not parked: the next go is searched on the survivor.
        move2, _ = _searched_go(engine, out, "g1f3 g8f6", 800, 2)
        assert _legal_after("g1f3 g8f6", move2)
        assert not any(marker in out.text() for marker in _UNSEARCHED)
    finally:
        stuck.release.set()
        engine.close()
        worker.close()


def test_engine_short_move_on_a_hung_device_does_not_park_the_next_one(out) -> None:
    """A move shorter than the stuck-device threshold: once its stop fires
    (the search deadline, or the deadline watchdog answering for it) the stuck
    device is released at once, so the search returns and the very next `go`
    is searched instead of being refused as 'parked' (before: that move and
    every later one unsearched)."""
    stuck = _Faulty(_evaluator(), mode="hang", after=2)
    engine, worker = _engine_with_pool([_evaluator(), stuck])
    try:
        move, took = _searched_go(engine, out, "e2e4", 300, 1)
        assert _legal_after("e2e4", move)
        assert took < 0.3 + 0.5
        time.sleep(0.5)  # the opponent's reply takes time
        move2, _ = _searched_go(engine, out, "e2e4 c7c5", 1000, 2)
        assert _legal_after("e2e4 c7c5", move2)
        tail = out.text().split(f"bestmove {move}", 1)[1]
        assert "parked" not in tail, tail
        assert "search not started" not in tail, tail
        assert "bestmove_fallback_used" not in tail, tail
        assert len(_degraded_lines(out)) == 1  # queued by the silenced thread, printed once
    finally:
        stuck.release.set()
        engine.close()
        worker.close()


def test_engine_primary_failure_moves_root_evals_to_the_pool(out) -> None:
    """devices[0] dying takes the primary (root-eval) evaluator with it; the
    root prior then comes from a surviving pool device, announced once."""
    primary = _Faulty(_evaluator(), mode="raise", after=0)
    engine, worker = _engine_with_pool([_evaluator(), _evaluator()], primary=primary)
    try:
        move, _ = _searched_go(engine, out, "c2c4", 800, 1)
        assert _legal_after("c2c4", move)
        assert primary.entered.is_set()
        assert not any(marker in out.text() for marker in _UNSEARCHED), out.text()
        move2, _ = _searched_go(engine, out, "c2c4 c7c5", 800, 2)
        assert _legal_after("c2c4 c7c5", move2)
        notes = [ln for ln in out.text().splitlines() if "primary (root-eval) evaluator" in ln]
        assert len(notes) == 1
        assert primary.calls == 1  # never retried once it failed
    finally:
        engine.close()
        worker.close()


# -- false drops, restore, visibility (review of the first cut) ------------------


class _OnceSlow:
    """A HEALTHY device whose ``k``-th submission takes ``delay`` s (a
    CUDA-graph record on a shape's first real call, a GC pause, an OS stall);
    every other call is normal."""

    def __init__(self, inner: Any, *, k: int, delay: float) -> None:
        self._inner = inner
        self.k = k
        self.delay = delay
        self.calls = 0
        # What the UCI pool warmup stamps on an evaluator (None = unseeded).
        self.warm_steady_call_s: float | None = None
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def evaluate_inplace_async(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self.calls += 1
            call = self.calls
        if call == self.k:
            time.sleep(self.delay)
        return self._inner.evaluate_inplace_async(*args, **kwargs)


def _run(pool: MultiGpuPucvPool, sims: int) -> tuple[int, MCTSTree]:
    tree, rid, cb = _seed_tree()
    done = pool.run(tree=tree, root_id=rid, root_cboard=cb, target_sims=sims,
                    stop_event=threading.Event())
    _assert_no_vloss(tree)
    return done, tree


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class _Paced:
    """A healthy device that takes ``per_call`` s on every submission, so it is
    still busy with the job while another device sits in a slow call."""

    def __init__(self, inner: Any, per_call: float) -> None:
        self._inner = inner
        self.per_call = per_call

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def evaluate_inplace_async(self, *args: Any, **kwargs: Any) -> Any:
        time.sleep(self.per_call)
        return self._inner.evaluate_inplace_async(*args, **kwargs)


def test_production_floor_keeps_a_device_with_one_slow_first_call() -> None:
    """The reviewer's repro at the shipped constants: a healthy device whose
    first call after warmup takes 1.2 s must not be dropped while the other
    device still has work (before: dropped for the life of the pool by the old
    1 s floor)."""
    slow = _OnceSlow(_evaluator(), k=1, delay=1.2)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[_Paced(_evaluator(), 0.06), slow],  # ~38 batches: busy > 2 s
    )
    try:
        assert pool.stall_threshold_s() >= 5.0
        done, _ = _run(pool, 300)
        assert slow.calls >= 1  # the slow call happened
        assert done == 300
        assert pool.failed_devices() == ()
        assert pool.drain_notes() == []
        calls = slow.calls
        assert _run(pool, 300)[0] == 300
        assert slow.calls > calls  # still live: it searches the next run too
    finally:
        pool.close()


def test_threshold_scales_with_the_seeded_steady_state_call(monkeypatch) -> None:
    """With a low floor, the threshold is carried by 10x the device's seeded
    (second-pass warmup) call time: a 1.2 s call on a device whose steady call
    is 0.5 s is slow-but-healthy, so it stays."""
    monkeypatch.setattr(pool_module, "_STALL_FLOOR_S", 0.3)
    slow = _OnceSlow(_evaluator(), k=1, delay=1.2)
    slow.warm_steady_call_s = 0.5
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[_Paced(_evaluator(), 0.06), slow],
    )
    try:
        assert pool._slowest_call_s[1] == pytest.approx(0.5)
        assert pool.stall_threshold_s(1) == pytest.approx(5.0)
        done, _ = _run(pool, 300)
        assert done == 300
        assert pool.failed_devices() == ()
    finally:
        pool.close()


@pytest.mark.parametrize("where", ["hang", "hang_sync"])
def test_stalled_device_is_restored_after_its_call_returns(monkeypatch, where) -> None:
    monkeypatch.setattr(pool_module, "_STALL_FLOOR_S", 0.3)
    stuck = _Faulty(_evaluator(), mode=where, after=1)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("cpu:0", "cpu:1")),
        evaluators=[_evaluator(), stuck],
    )
    try:
        quarantined = len(async_safety._quarantine)
        _run(pool, 300)
        assert pool.failed_devices() == ("cpu:1",)
        assert len(async_safety._quarantine) == quarantined + 1
        assert "stopped answering" in (pool.degraded_summary() or "")
        pool.drain_notes()
        assert pool.restore_recovered_devices() == ()  # still stuck: nothing to restore
        stuck.release.set()
        assert _wait_for(lambda: pool._restorable[1])
        assert pool.restore_recovered_devices() == ("cpu:1",)
        # Released, not replaced: a flapping device does not grow the list.
        assert len(async_safety._quarantine) == quarantined
        notes = pool.drain_notes()
        assert len(notes) == 1
        assert notes[0].startswith("multi-GPU RESTORED: device cpu:1")
        assert "2 of 2" in notes[0]
        assert pool.failed_devices() == ()
        assert pool.degraded_summary() is None
        calls = stuck.calls
        done, _ = _run(pool, 300)
        assert done == 300
        assert stuck.calls > calls  # the restored device searches again
        assert pool.failed_devices() == ()
    finally:
        stuck.release.set()
        pool.close()


def test_a_late_answer_never_reaches_the_shared_eval_cache(monkeypatch) -> None:
    """A device stuck waiting for its batch answers after it was abandoned:
    that result must not be stored in the eval cache the survivors share."""
    monkeypatch.setattr(pool_module, "_STALL_FLOOR_S", 0.3)
    stuck = _Faulty(_evaluator(), mode="hang_sync", after=1)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, eval_cache_entries=4096,
                           device_labels=("cpu:0", "cpu:1")),
        evaluators=[_evaluator(), stuck],
    )
    try:
        _run(pool, 300)
        assert pool.failed_devices() == ("cpu:1",)
        cache: Any = pool._cache
        late_puts: list[Any] = []
        real_put = cache.put
        monkeypatch.setattr(cache, "put", lambda *a, **k: (late_puts.append(a), real_put(*a, **k)))
        stuck.release.set()
        assert _wait_for(lambda: pool._restorable[1])
        assert late_puts == []
    finally:
        stuck.release.set()
        pool.close()


def test_a_device_that_raised_is_not_restored() -> None:
    bad = _Faulty(_evaluator(), mode="raise", after=2)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("cpu:0", "cpu:1")),
        evaluators=[_evaluator(), bad],
    )
    try:
        _run(pool, 300)
        assert pool.failed_devices() == ("cpu:1",)
        time.sleep(0.2)
        assert pool.restore_recovered_devices() == ()
        assert pool.failed_devices() == ("cpu:1",)
        assert "failed mid-search" in (pool.degraded_summary() or "")
    finally:
        pool.close()


def _degraded_go_lines(out: _Out) -> list[str]:
    return [ln for ln in out.text().splitlines() if ln.startswith("info string multi-GPU degraded:")]


def test_engine_says_degraded_on_every_go_isready_and_searchconfig(out) -> None:
    bad = _Faulty(_evaluator(), mode="raise", after=3)
    engine, worker = _engine_with_pool([_evaluator(), bad, _evaluator()])
    try:
        _searched_go(engine, out, "e2e4", 600, 1)
        assert worker.failed_pool_devices() == ("cpu:1",)
        assert _degraded_go_lines(out) == []  # the drop happened DURING that search
        _searched_go(engine, out, "e2e4 e7e5", 400, 2)
        _searched_go(engine, out, "e2e4 e7e5 g1f3", 400, 3)
        lines = _degraded_go_lines(out)
        assert len(lines) == 2  # one per later search
        assert "2 of 3 devices" in lines[0]
        assert "cpu:1 failed: failed mid-search" in lines[0]
        engine.dispatch(parse_command("isready"))
        assert len(_degraded_go_lines(out)) == 3
        engine.dispatch(parse_command("searchconfig"))
        assert any(
            ln.startswith("info string searchconfig multi-GPU degraded: 2 of 3 devices")
            for ln in out.text().splitlines()
        )
        assert "unhealthy" not in out.text().lower()
    finally:
        engine.close()
        worker.close()


def test_engine_restores_a_stalled_device_on_the_next_go(out, monkeypatch) -> None:
    monkeypatch.setattr(pool_module, "_STALL_FLOOR_S", 0.4)
    stuck = _Faulty(_evaluator(), mode="hang", after=2)
    engine, worker = _engine_with_pool([_evaluator(), stuck])
    try:
        _searched_go(engine, out, "g1f3", 2000, 1)
        assert worker.failed_pool_devices() == ("cpu:1",)
        stuck.release.set()
        pool: Any = worker._pucv_pool
        assert _wait_for(lambda: pool._restorable[1])
        calls = stuck.calls
        move2, _ = _searched_go(engine, out, "g1f3 g8f6", 800, 2)
        assert _legal_after("g1f3 g8f6", move2)
        restored = [ln for ln in out.text().splitlines() if "multi-GPU RESTORED" in ln]
        assert len(restored) == 1
        assert restored[0].startswith("info string multi-GPU RESTORED: device cpu:1")
        assert worker.failed_pool_devices() == ()
        assert stuck.calls > calls
        assert _degraded_go_lines(out) == []  # restored before the per-go report
    finally:
        stuck.release.set()
        engine.close()
        worker.close()


class _HangThenBadReceipt:
    """Submission ``after+1`` hangs in its completion wait; submission
    ``after+2`` (already in flight while the worker waits) can never be
    drained. The late answer to the first comes back clean, but the device
    did not quiesce, so it must NOT be restored."""

    def __init__(self, inner: Any, *, after: int) -> None:
        self._inner = inner
        self.after = after
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def evaluate_inplace_async(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self.calls += 1
            call = self.calls
        pol, wdl, event = self._inner.evaluate_inplace_async(*args, **kwargs)
        if call == self.after + 1:
            return pol, wdl, _BlockingEvent(self)
        if call == self.after + 2:
            return pol, wdl, _UndrainableEvent()
        return pol, wdl, event


class _UndrainableEvent:
    def synchronize(self) -> None:
        raise RuntimeError("CUDA error: unspecified launch failure (injected at drain)")


def test_a_late_clean_answer_with_an_undrainable_receipt_is_not_restored(monkeypatch) -> None:
    monkeypatch.setattr(pool_module, "_STALL_FLOOR_S", 0.3)
    dev = _HangThenBadReceipt(_evaluator(), after=1)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("cpu:0", "cpu:1")),
        evaluators=[_evaluator(), dev],
    )
    try:
        _run(pool, 300)
        assert dev.entered.is_set()
        assert dev.calls >= 3  # the undrainable submission really was in flight
        assert pool.failed_devices() == ("cpu:1",)
        dev.release.set()
        assert _wait_for(lambda: not pool._threads[1].is_alive(), timeout=5.0)
        assert not pool._restorable[1]
        assert pool.restore_recovered_devices() == ()
        assert pool.failed_devices() == ("cpu:1",)
    finally:
        dev.release.set()
        pool.close()


# -- round 2: idle survivors, caps, stall-then-raise ----------------------------


def test_a_stuck_device_does_not_cost_the_move_once_the_others_are_idle(out) -> None:
    """At the SHIPPED 5 s floor, device 1 hangs early in a 2 s move. The other
    device drains each chunk's budget and goes idle, so patience buys nothing:
    the stuck worker is abandoned at the stop threshold and the move is
    SEARCHED (before: the run waited 5 s and the watchdog answered unsearched
    from the root policy)."""
    stuck = _Faulty(_evaluator(), mode="hang", after=2)
    engine, worker = _engine_with_pool([_evaluator(), stuck])
    try:
        move, took = _searched_go(engine, out, "e2e4", 2000, 1)
        assert stuck.entered.is_set()
        assert _legal_after("e2e4", move)
        assert took < 2.0 + 0.5
        assert not any(marker in out.text() for marker in _UNSEARCHED), out.text()
        assert worker.failed_pool_devices() == ("cpu:1",)
        stuck.release.set()
        pool: Any = worker._pucv_pool
        assert _wait_for(lambda: pool._restorable[1])  # the stall was not a fault
    finally:
        stuck.release.set()
        engine.close()
        worker.close()


def test_a_healthy_slow_call_on_a_drained_budget_is_dropped_then_restored() -> None:
    """The accepted cost of the idle-survivor rule: a healthy call slower than
    the 1 s idle floor that lands when the other device has already finished
    the job is dropped, and restored (next chunk / search) once it answers."""
    slow = _OnceSlow(_evaluator(), k=1, delay=1.6)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[_evaluator(), slow],
    )
    try:
        assert pool.idle_stall_threshold_s(1) == pytest.approx(1.0)
        started = time.monotonic()
        _run(pool, 100)
        assert time.monotonic() - started < 1.5  # did not wait out the slow call
        assert pool.failed_devices() == ("g1",)
        assert _wait_for(lambda: pool._restorable[1])
        assert pool.restore_recovered_devices() == ("g1",)
        assert pool.failed_devices() == ()
    finally:
        pool.close()


def test_one_device_left_keeps_the_long_threshold() -> None:
    """With no idle survivor there is nothing to hand the work to: a single
    device's slow call is waited out, not dropped."""
    slow = _OnceSlow(_evaluator(), k=2, delay=1.2)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=1, gather=8, device_labels=("g0",)),
        evaluators=[slow],
    )
    try:
        assert _run(pool, 100)[0] == 100
        assert pool.failed_devices() == ()
    finally:
        pool.close()


def test_thresholds_are_capped_however_slow_a_call_once_was() -> None:
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[_evaluator(), _evaluator()],
    )
    try:
        pool._slowest_call_s[1] = 4.0  # one healthy 4 s call
        assert pool.stall_threshold_s(1) == pytest.approx(30.0)  # not 40 s
        assert pool.stop_stall_threshold_s(1) == pytest.approx(1.0)  # not 20 s
        pool._slowest_call_s[0] = 0.01
        assert pool.stall_threshold_s(0) == pytest.approx(5.0)
        assert pool.stop_stall_threshold_s(0) == pytest.approx(0.25)
    finally:
        pool.close()


def test_a_stalled_call_that_then_raises_is_not_restorable(monkeypatch) -> None:
    monkeypatch.setattr(pool_module, "_STALL_FLOOR_S", 0.3)
    bad = _Faulty(_evaluator(), mode="hang_raise", after=1)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("cpu:0", "cpu:1")),
        evaluators=[_Paced(_evaluator(), 0.05), bad],
    )
    try:
        _run(pool, 300)
        assert pool.failed_devices() == ("cpu:1",)
        bad.release.set()  # the stuck call now RAISES
        assert _wait_for(lambda: not pool._threads[1].is_alive(), timeout=5.0)
        assert not pool._restorable[1]
        assert pool.restore_recovered_devices() == ()
        assert pool.failed_devices() == ("cpu:1",)
    finally:
        bad.release.set()
        pool.close()



@pytest.mark.parametrize("delay", [0.3, 0.6])
def test_a_tail_hiccup_under_the_idle_floor_is_kept(delay) -> None:
    """Shipped constants, survivor already idle (small budget): a healthy
    0.3 s / 0.6 s hiccup is waited out, not dropped (the bare 0.25 s stop
    threshold used to drop it until the next search)."""
    slow = _OnceSlow(_evaluator(), k=2, delay=delay)
    slow.warm_steady_call_s = 0.017
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[_evaluator(), slow],
    )
    try:
        assert _run(pool, 300)[0] == 300
        assert pool.failed_devices() == ()
        assert pool.drain_notes() == []
    finally:
        pool.close()


def test_a_dropped_slow_device_is_back_at_the_next_chunk_of_the_same_go(out) -> None:
    """A healthy 1.2 s call with the survivor idle is dropped at the 1 s idle
    floor; once it answers, the NEXT CHUNK of the same search re-admits it, so
    within one `go` the device searches again. One DEGRADED and one RESTORED
    line; the move is searched."""
    slow = _OnceSlow(_evaluator(), k=3, delay=1.2)
    engine, worker = _engine_with_pool([_evaluator(), slow])
    try:
        move, _ = _searched_go(engine, out, "d2d4", 4000, 1)
        assert _legal_after("d2d4", move)
        text = out.text()
        assert not any(marker in text for marker in _UNSEARCHED), text
        assert len(_degraded_lines(out)) == 1
        restored = [ln for ln in text.splitlines() if "multi-GPU RESTORED" in ln]
        assert len(restored) == 1
        # It searched again within the SAME go: calls after the slow one, and
        # the restore was printed before that go's bestmove.
        assert slow.calls > 3 + 2
        assert text.index("multi-GPU RESTORED") < text.index(f"bestmove {move}")
        assert worker.failed_pool_devices() == ()
    finally:
        engine.close()
        worker.close()


def test_a_progressing_worker_keeps_the_mid_search_patience_for_a_stuck_one() -> None:
    """Three devices: g0 finished the job (idle), g1 is still running and
    making progress (inside a YOUNG call), g2 is stuck 2 s. The idle-survivor
    rule needs EVERY running worker stuck past the idle floor; with g1
    progressing, g2 keeps the 5 s mid-search patience."""
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=3, gather=8, device_labels=("g0", "g1", "g2")),
        evaluators=[_evaluator(), _evaluator(), _evaluator()],
    )
    try:
        now = time.monotonic()
        with pool._cv:
            pool._running = {1, 2}
            pool._busy_since[1] = now - 0.2  # progressing
            pool._busy_since[2] = now - 2.0  # stuck, under the 5 s patience
            pool._abandon_stuck_locked(None)
            assert not pool._abandoned[2]
            assert pool._live == [True, True, True]
            # Once g1 is ALSO stuck past the idle floor, nobody progresses.
            pool._busy_since[1] = now - 1.5
            pool._abandon_stuck_locked(None)
            assert pool._abandoned[2]
            pool._running = set()
            pool._busy_since = [None, None, None]
    finally:
        pool.close()


def test_close_joins_an_abandoned_worker_whose_call_returns_shortly() -> None:
    """A worker abandoned inside a call that returns ~0.5 s later is joined by
    close() (bounded), so no pool thread is still inside the evaluator when the
    interpreter finalizes; only a genuinely wedged one would be left alive."""
    slow = _OnceSlow(_evaluator(), k=1, delay=1.5)
    pool = MultiGpuPucvPool(
        MultiGpuPucvConfig(n_gpus=2, gather=8, device_labels=("g0", "g1")),
        evaluators=[_evaluator(), slow],
    )
    try:
        _run(pool, 100)
        assert pool.failed_devices() == ("g1",)
    finally:
        pool.close()
    assert pool.live_worker_threads() == []
