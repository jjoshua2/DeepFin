"""Real native/Direct async pipeline with CPU-only CUDA loan sentinels."""
from __future__ import annotations

from contextlib import contextmanager
import types
from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine import async_safety, inference
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts import gumbel_c
from chess_anti_engine.mcts.gumbel import GumbelConfig


class ZeroModel(torch.nn.Module):
    input_extra_features = "v1"

    def forward(self, x):
        return {"policy": x.new_zeros((len(x), 4672)), "wdl": x.new_zeros((len(x), 3))}


def run_pipeline_case(monkeypatch, *, stage=None, group=1, inplace=True, cleanup="clean") -> Any:
    jobs = []
    receipts = []
    streams = []
    primary = MemoryError(f"primary {stage} CPU consumer sentinel")
    secondary = TimeoutError("secondary completion sentinel")
    state = {"raised": False}

    class Stream:
        def __init__(self, device=None):
            del device
            self.jobs = []
            self.drains = 0
            streams.append(self)

        def wait_event(self, event):
            del event

        def synchronize(self):
            self.drains += 1
            if cleanup == "stream" and state["raised"]:
                raise RuntimeError("secondary stream drain sentinel")
            self.finish(len(self.jobs))

        def finish(self, count):
            for job in self.jobs[:count]:
                job["complete"] = True

    default = Stream()
    current = [default]

    @contextmanager
    def scope(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    class Event:
        def __init__(self):
            self.calls = 0
            self.stream = current[0]
            self.count = 0

        def record(self, stream=None):
            self.stream = stream or current[0]
            self.count = len(self.stream.jobs)

        def synchronize(self):
            self.calls += 1
            if cleanup != "clean" and state["raised"]:
                raise secondary
            self.stream.finish(self.count)

    ev: Any = inference.DirectGPUEvaluator(ZeroModel(), device="cpu", max_batch=512,
                                           n_slots=2, use_amp=False, legal_bf16=False)
    ev._use_cuda = True  # Production async body; every CUDA API below is mocked.

    def device_input(owner, bsz, *, slot):
        view = owner._pinned_inputs_np[slot][:bsz]
        job = {"slot": slot, "group": len(jobs) % 2, "view": view,
               "snapshot": view.copy(), "complete": False}
        jobs.append(job)
        current[0].jobs.append(job)
        return owner._pinned_inputs[slot][:bsz].clone()

    ev._device_input = types.MethodType(device_input, ev)
    original_guard = inference.guarded_event

    def guarded(owner, event, stream, *retained):
        receipt = original_guard(owner, event, stream, *retained)
        receipts.append(receipt)
        return receipt

    class LegacyEvaluator:
        supports_inplace_api = False

        def evaluate_encoded(self, x: np.ndarray, relations: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
            return ev.evaluate_encoded(x, relations=relations)

        @property
        def async_backend(self):
            return ev

        def __getattr__(self, name):
            return getattr(ev, name)

    evaluator = ev if inplace else LegacyEvaluator()
    if stage == "unpack":
        method = "evaluate_inplace_async" if inplace else "evaluate_encoded_async"
        original_submit = getattr(ev, method)

        class FailedUnpack(tuple):
            def __iter__(self):
                state["raised"] = True
                raise primary

        def submit(*args, **kwargs):
            return FailedUnpack(original_submit(*args, **kwargs))

        monkeypatch.setattr(ev, method, submit)

    def pending():
        return [job for job in jobs if not job["complete"]]

    def fail_cpu(cpu_group):
        held = pending()
        if held and cpu_group == group:
            assert held[0]["group"] != cpu_group
            state["raised"] = True
            raise primary

    class Probe:
        max_pieces = 7

        def apply(self, leaves, wdl, **kwargs):
            del leaves, wdl, kwargs
            if stage == "callback" and pending():
                fail_cpu(1 - pending()[0]["group"])

    if stage == "native":
        original_tree = gumbel_c.MCTSTree
        started = []

        class TreeProxy:
            def __init__(self):
                self.tree = original_tree()
                self.group = -1

            def __getattr__(self, name):
                return getattr(self.tree, name)

            def start_gumbel_sims(self, *args, **kwargs):
                self.group = len(started)
                started.append(self)
                return self.tree.start_gumbel_sims(*args, **kwargs)

            def continue_gumbel_sims(self, *args, **kwargs):
                fail_cpu(self.group)
                return self.tree.continue_gumbel_sims(*args, **kwargs)

        monkeypatch.setattr(gumbel_c, "MCTSTree", TreeProxy)
    if stage == "transform":
        transform = gumbel_c._policy_logits_to_full

        def fail_transform(*args, **kwargs):
            if receipts:
                assert not pending()  # Result transforms follow synchronization.
                state["raised"] = True
                raise primary
            return transform(*args, **kwargs)

        monkeypatch.setattr(gumbel_c, "_policy_logits_to_full", fail_transform)

    monkeypatch.setattr(inference.torch.cuda, "Stream", Stream)
    monkeypatch.setattr(inference.torch.cuda, "Event", Event)
    monkeypatch.setattr(inference.torch.cuda, "current_stream", lambda device=None: current[0])
    monkeypatch.setattr(inference.torch.cuda, "stream", scope)
    monkeypatch.setattr(inference, "guarded_event", guarded)
    result = None
    error = None
    try:
        result = gumbel_c.run_gumbel_root_many_c(
            None, [chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1") for _ in range(64)],
            device="cpu", rng=np.random.default_rng(7301),
            cfg=GumbelConfig(simulations=8, topk=2, add_noise=False, input_extra_features="v1"),
            evaluator=evaluator, pre_pol_logits=np.zeros((64, 4672), dtype=np.float32),
            pre_wdl_logits=np.zeros((64, 3), dtype=np.float32), pre_wdl_logits_tb_probed=True,
            allow_terminal_root_shortcuts=False, tb_probe=Probe(), target_batch=1,
        )
    except BaseException as exc:
        error = exc
    return {"result": result, "error": error, "primary": primary, "jobs": jobs,
            "receipts": receipts, "streams": streams, "backend": ev, "evaluator": evaluator}


@pytest.mark.parametrize("inplace", [False, True])
def test_healthy_gumbel_pipeline_finishes_each_receipt_once(monkeypatch, inplace):
    case = run_pipeline_case(monkeypatch, inplace=inplace)
    assert case["error"] is None
    assert len(case["result"][1]) == 64
    assert len(case["receipts"]) == 8
    assert all(receipt._event.calls == 1 for receipt in case["receipts"])
    assert all(job["complete"] for job in case["jobs"])
    assert all(stream.drains == 0 for stream in case["streams"])


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("stage", ["callback", "native"])
def test_cpu_consumer_failure_closes_other_groups_receipt(monkeypatch, inplace, group, stage):
    case = run_pipeline_case(monkeypatch, inplace=inplace, group=group, stage=stage)
    assert case["error"] is case["primary"]
    assert all(job["complete"] for job in case["jobs"])
    assert all(receipt._event.calls == 1 for receipt in case["receipts"])
    assert all(stream.drains == 0 for stream in case["streams"])
    async_safety.ensure_usable(case["evaluator"])
    last = case["jobs"][-1]
    view = case["backend"].get_input_buffer(1, slot=last["slot"])
    assert np.shares_memory(view, last["view"])
    view.fill(7)


@pytest.mark.parametrize("inplace", [False, True])
def test_completed_receipt_is_not_retried_for_policy_transform_failure(monkeypatch, inplace):
    case = run_pipeline_case(monkeypatch, inplace=inplace, stage="transform")
    assert case["error"] is case["primary"]
    assert all(job["complete"] for job in case["jobs"])
    assert all(receipt._event.calls == 1 for receipt in case["receipts"])
    assert all(stream.drains == 0 for stream in case["streams"])
    async_safety.ensure_usable(case["evaluator"])


@pytest.mark.parametrize("inplace", [False, True])
def test_returned_receipt_is_owned_before_cpu_unpack(monkeypatch, inplace):
    case = run_pipeline_case(monkeypatch, inplace=inplace, stage="unpack")
    assert case["error"] is case["primary"]
    assert len(case["receipts"]) == 1
    assert all(job["complete"] for job in case["jobs"])
    assert case["receipts"][0]._event.calls == 1
    async_safety.ensure_usable(case["evaluator"])


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("cleanup", ["event", "stream"])
def test_cpu_primary_survives_receipt_failure_with_existing_quarantine(monkeypatch, inplace, cleanup):
    case = run_pipeline_case(monkeypatch, inplace=inplace, stage="callback", cleanup=cleanup)
    assert case["error"] is case["primary"]
    assert any("pending receipt drain failed" in note for note in getattr(case["error"], "__notes__", ()))
    assert sum(stream.drains for stream in case["streams"]) == 1
    if cleanup == "stream":
        with pytest.raises(RuntimeError, match="poisoned"):
            async_safety.ensure_usable(case["evaluator"])
        with pytest.raises(RuntimeError, match="poisoned"):
            case["backend"].get_input_buffer(1)
        fresh = ThreadSafeGPUDispatcher(case["backend"])
        with pytest.raises(RuntimeError, match="poisoned"):
            async_safety.ensure_usable(fresh)
        with pytest.raises(RuntimeError, match="poisoned"):
            fresh.get_input_buffer(1)
        with pytest.raises(RuntimeError, match="poisoned"):
            fresh.evaluate_inplace_async(1)
        assert any(entry[0] is case["backend"] for entry in async_safety._quarantine)
        held = [job for job in case["jobs"] if not job["complete"]]
        assert len(held) == 1
        np.testing.assert_array_equal(held[0]["view"], held[0]["snapshot"])
    else:
        assert all(job["complete"] for job in case["jobs"])
        async_safety.ensure_usable(case["evaluator"])
