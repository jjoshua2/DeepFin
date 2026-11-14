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
    """A completion receipt whose wait never ends until the test releases it."""

    def __init__(self, owner: _Faulty) -> None:
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
        MultiGpuPucvConfig(n_gpus=3, gather=8, device_labels=("cuda:0", "cuda:1", "cuda:2")),
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
        MultiGpuPucvPool(MultiGpuPucvConfig(n_gpus=2, gather=8),
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
