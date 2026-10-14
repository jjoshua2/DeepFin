"""Bounded public RPG startup ownership and real async-loan failures, CPU only."""
from __future__ import annotations

import contextlib
from pathlib import Path
import threading
from types import FrameType
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine import async_safety, inference
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import CUDAOwnerDispatcher, ThreadSafeGPUDispatcher
from chess_anti_engine.mcts import gumbel_c, puct_vl
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.uci import __main__ as main
from chess_anti_engine.uci import root_parallel_gumbel as rpg
from tests.test_gumbel_pipeline_receipts import ZeroModel, run_pipeline_case


class Owner(CUDAOwnerDispatcher):
    def __init__(self, fail_close: bool = False):
        self.close_calls = 0
        self.closed_on: list[str] = []
        self.invalid_slots = False
        self.fail_close = fail_close
        super().__init__(DirectGPUEvaluator(ZeroModel(), device="cpu", max_batch=64, n_slots=2))

    @property
    def n_slots(self) -> int:
        return 1 if self.invalid_slots else 2

    def close(self):
        self.close_calls += 1
        self.closed_on.append(threading.current_thread().name)
        super().close()
        if self.fail_close:
            raise SystemExit("controlled secondary owner close failure")


def build(primary, factory):
    return main._build_engine(
        evaluator=primary, primary_device="cpu", chunk_sims=16, topk=4,
        n_walkers=2, vloss_weight=1, walker_gather=1, pucv_vloss_mode=0,
        max_batch=64, vl_gather=32, eval_cache_entries=0,
        use_multi_gpu_pucv=False, use_root_parallel_gumbel=True,
        devices=("cpu", "cpu"),
        rebuild_multi_gpu_pucv_factories=lambda *_a, **_kw: [factory, factory],
    )


def cleanup(owners):
    for owner in owners:
        owner.fail_close = False
        with contextlib.suppress(BaseException):
            owner.close()


def assert_pool_threads_retired():
    assert not any(t.name.startswith(("rpg-group-", "walker-")) for t in threading.enumerate())


@pytest.mark.parametrize("stage", ["buffers", "factory", "validation", "interrupt"])
@pytest.mark.parametrize("fail_close", [False, True])
def test_public_builder_retires_every_startup_owned_helper(monkeypatch, stage, fail_close):
    primary = Owner()
    created = []
    error = KeyboardInterrupt("controlled factory interrupt") if stage == "interrupt" else MemoryError("controlled startup failure")
    cause = ValueError("existing primary cause")
    error.__cause__ = cause
    worker_errors = []
    monkeypatch.setattr(threading, "excepthook", lambda args: worker_errors.append(args.exc_value))

    def factory():
        idx = int(threading.current_thread().name.rsplit("-", 1)[1])
        if stage in ("factory", "interrupt") and idx == 1:
            raise error
        owner = Owner(fail_close)
        created.append(owner)
        owner.invalid_slots = stage == "validation" and idx == 1
        return owner

    original = puct_vl._alloc_buffers

    def allocate(*args, **kwargs):
        if stage == "buffers" and threading.current_thread().name == "rpg-group-1":
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(puct_vl, "_alloc_buffers", allocate)
    engine = None
    try:
        with pytest.raises(ValueError if stage == "validation" else type(error)) as caught:
            engine = build(primary, factory)
        if stage != "validation":
            assert caught.value is error
            assert caught.value.__cause__ is cause
        assert primary.close_calls == 0  # Public builder retains caller's primary.
        assert created
        assert all(owner.close_calls == 1 and not owner._owner.is_alive() for owner in created)
        assert all(owner.closed_on[0].startswith("rpg-group-") for owner in created)
        assert not worker_errors
        assert_pool_threads_retired()
        if fail_close:
            assert any("secondary owner close failure" in note for note in getattr(caught.value, "__notes__", ()))
    finally:
        if engine is not None:
            engine.close()
        cleanup([primary, *created])


def test_successful_public_builder_transfers_helpers_for_single_close():
    primary = Owner()
    created = []

    def factory():
        owner = Owner()
        created.append(owner)
        return owner

    engine = None
    try:
        engine = build(primary, factory)
        assert all(owner.close_calls == 0 and owner._owner.is_alive() for owner in created)
        engine.close()
        assert primary.close_calls == 1
        assert not primary._owner.is_alive()
        assert all(owner.close_calls == 1 and not owner._owner.is_alive() for owner in created)
        assert_pool_threads_retired()
    finally:
        cleanup([primary, *created])


@pytest.mark.parametrize("fault", [False, True])
def test_borrowed_helpers_remain_caller_owned(monkeypatch, fault):
    owners = [Owner(), Owner()]
    original = puct_vl._alloc_buffers

    def allocate(*args, **kwargs):
        if fault and threading.current_thread().name == "rpg-group-1":
            raise MemoryError("controlled borrowed startup allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(puct_vl, "_alloc_buffers", allocate)
    pool = None
    try:
        cfg = rpg.RootParallelGumbelConfig(n_groups=2, gather=2, input_planes=146)
        gcfg = GumbelConfig(input_extra_features="v1", topk=2, add_noise=False)
        if fault:
            with pytest.raises(MemoryError):
                pool = rpg.RootParallelGumbelPool(cfg, gcfg, evaluators=owners)
        else:
            pool = rpg.RootParallelGumbelPool(cfg, gcfg, evaluators=owners)
            pool.prepare_root(tree=MCTSTree(), board=chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
                              pol_logits=np.zeros(4672), wdl_logits=np.zeros(3), allow_terminal_shortcuts=False)
            pool.run(target_sims=4, stop_event=threading.Event())
            assert sum(pool.last_stats().group_sims) == 4
            pool.close()
        assert all(owner.close_calls == 0 and owner._owner.is_alive() for owner in owners)
        assert_pool_threads_retired()
    finally:
        if pool is not None:
            pool.close()
        cleanup(owners)


@pytest.mark.parametrize(("fault", "cleanup_failure"), [(False, "clean"), (True, "clean"), (True, "event"), (True, "stream")])
def test_public_rpg_owns_valid_returned_receipt_before_cpu_work(monkeypatch, fault, cleanup_failure):
    primary = MemoryError("controlled CPU continuation failure after valid submit")
    cause = ValueError("existing continuation cause")
    primary.__cause__ = cause
    state: dict[str, Any] = {"faulted": False, "reached": False, "valid_return": False}
    lines = Path(rpg.__file__).read_text().splitlines()
    boundaries = {"receipt = (pol_t, wdl_t, event)", "pol_t, wdl_t, event = receipt"}
    boundary = next(i + 1 for i, line in enumerate(lines) if line.strip() in boundaries)

    def continuation(frame: FrameType, event: str, arg: Any):
        del arg
        if event == "line" and frame.f_code.co_name == "_expand_candidate" and frame.f_lineno == boundary:
            state["reached"] = True
            receipt = frame.f_locals.get("receipt")
            state["valid_return"] = (type(receipt) is tuple and len(receipt) == 3) or all(name in frame.f_locals for name in ("pol_t", "wdl_t", "event"))
            if fault and not state["faulted"]:
                state["faulted"] = True
                raise primary
        return continuation

    original_guard = inference.guarded_event

    def guard(owner: Any, event: Any, stream: Any, *retained: Any) -> Any:
        original_event_sync = event.synchronize
        original_stream_sync = stream.synchronize

        def event_sync():
            if state["faulted"] and cleanup_failure != "clean":
                event.calls += 1
                raise TimeoutError("controlled secondary completion failure")
            original_event_sync()

        def stream_sync():
            if state["faulted"] and cleanup_failure == "stream":
                stream.drains += 1
                raise RuntimeError("controlled secondary stream drain failure")
            original_stream_sync()

        event.synchronize = event_sync
        stream.synchronize = stream_sync
        return original_guard(owner, event, stream, *retained)

    def public_rpg(*args, **kwargs):
        pool = rpg.RootParallelGumbelPool(rpg.RootParallelGumbelConfig(n_groups=1, gather=2, input_planes=146),
                                        kwargs["cfg"], evaluators=[kwargs["evaluator"]])
        try:
            pool.prepare_root(tree=MCTSTree(), board=args[1][0], pol_logits=kwargs["pre_pol_logits"][0],
                              wdl_logits=kwargs["pre_wdl_logits"][0], allow_terminal_shortcuts=False)
            return pool.run(target_sims=4, stop_event=threading.Event())
        finally:
            pool.close()

    monkeypatch.setattr(inference, "guarded_event", guard)
    monkeypatch.setattr(gumbel_c, "run_gumbel_root_many_c", public_rpg)
    previous = threading.gettrace()
    try:
        threading.settrace(continuation)
        case = run_pipeline_case(monkeypatch, inplace=True)
    finally:
        threading.settrace(previous)
    try:
        assert state["reached"]
        assert state["valid_return"]
        if fault:
            assert case["error"] is primary
            assert case["error"].__cause__ is cause
            assert len(case["receipts"]) == 1
        else:
            assert case["error"] is None
            assert len(case["receipts"]) == 4
        assert all(receipt._event.calls == 1 for receipt in case["receipts"])
        if cleanup_failure == "stream":
            with pytest.raises(RuntimeError, match="poisoned"):
                async_safety.ensure_usable(case["evaluator"])
            with pytest.raises(RuntimeError, match="poisoned"):
                case["backend"].get_input_buffer(1)
            fresh = ThreadSafeGPUDispatcher(case["backend"])
            with pytest.raises(RuntimeError, match="poisoned"):
                fresh.evaluate_inplace_async(1)
            assert any(entry[0] is case["backend"] for entry in async_safety._quarantine)
        else:
            assert all(job["complete"] for job in case["jobs"])
            async_safety.ensure_usable(case["evaluator"])
        assert sum(stream.drains for stream in case["streams"]) == (0 if cleanup_failure == "clean" else 1)
        if fault and cleanup_failure != "clean":
            assert any("pending receipt drain failed" in note for note in getattr(primary, "__notes__", ()))
        assert_pool_threads_retired()
    finally:
        for stream in case["streams"]:
            stream.finish(len(stream.jobs))
