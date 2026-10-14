"""Unequal-budget native Gumbel tail ownership with real Direct CPU mocks."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from chess_anti_engine import async_safety
from chess_anti_engine.inference_dispatcher import ThreadSafeGPUDispatcher
from chess_anti_engine.mcts import gumbel_c


def run_tail_case(monkeypatch, *, inplace, budgets=(4, 12), fault=None, cleanup="clean") -> Any:
    # Reuse the pinned composition's existing real Direct/native loan fixture.
    spec = importlib.util.spec_from_file_location("tail_loan_fixture", Path(__file__).with_name("test_gumbel_pipeline_receipts.py"))
    assert spec is not None
    assert spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    actual_run = gumbel_c.run_gumbel_root_many_c
    primary = KeyboardInterrupt("tail CPU cancellation after valid tuple handoff") if fault == "interrupt" else MemoryError(f"tail {fault} consumer sentinel")
    observed: dict[str, Any] = {"entered": 0, "entry_loans": [], "raised": False,
                                "builtin_tuple": False, "owned_at_handoff": False}

    def unequal_run(*args, **kwargs):
        fixture_frame = sys._getframe(1)
        jobs = fixture_frame.f_locals["jobs"]
        failure_state = fixture_frame.f_locals["state"]
        backend = fixture_frame.f_locals["ev"]
        kwargs["per_game_simulations"] = [budgets[0]] * 32 + [budgets[1]] * 32

        def fail():
            observed["raised"] = True
            failure_state["raised"] = True
            raise primary

        method = "evaluate_inplace_async" if inplace else "evaluate_encoded_async"
        original_submit = getattr(backend, method)

        def submit(*values, **options):
            result = original_submit(*values, **options)
            if sys._getframe(1).f_code.co_name == "_drain_sequential":
                observed["builtin_tuple"] = type(result) is tuple
                observed["returned_tuple"] = result
            return result

        monkeypatch.setattr(backend, method, submit)

        class Callback:
            max_pieces = 7

            def apply(self, leaves, wdl, **options):
                del leaves, wdl, options
                if sys._getframe(2).f_code.co_name == "_drain_sequential":
                    fail()

        if fault == "callback":
            kwargs["tb_probe"] = Callback()
        if fault == "native":
            native_tree = gumbel_c.MCTSTree

            class TreeProxy:
                def __init__(self):
                    self.tree = native_tree()

                def __getattr__(self, name):
                    return getattr(self.tree, name)

                def continue_gumbel_sims(self, *values, **options):
                    if sys._getframe(1).f_code.co_name == "_drain_sequential":
                        fail()
                    return self.tree.continue_gumbel_sims(*values, **options)

            monkeypatch.setattr(gumbel_c, "MCTSTree", TreeProxy)

        def trace(frame, event, arg):
            del arg
            if frame.f_code.co_name == "_drain_sequential" and frame.f_code.co_filename == gumbel_c.__file__:
                if event == "call":
                    observed["entered"] += 1
                    observed["entry_loans"].append(sum(not job["complete"] for job in jobs))
                    assert frame.f_back.f_locals["_pending_receipt"] is None
                elif event == "line" and "returned_tuple" in observed and not observed["raised"]:
                    owned = frame.f_locals.get("_pending_receipt")
                    observed["owned_at_handoff"] = owned is observed["returned_tuple"]
                    if fault == "interrupt":
                        # First CPU statement after a real built-in tuple returns:
                        # before unpack in the fix, before sync in the old helper.
                        fail()
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            return actual_run(*args, **kwargs)
        finally:
            sys.settrace(previous)

    monkeypatch.setattr(gumbel_c, "run_gumbel_root_many_c", unequal_run)
    result = fixture.run_pipeline_case(monkeypatch, inplace=inplace, cleanup=cleanup)
    result["tail"] = observed
    result["tail_primary"] = primary
    return result


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("budgets", [(4, 12), (12, 4)])
def test_unequal_budget_tail_finishes_once_without_stream_drain(monkeypatch, inplace, budgets):
    case = run_tail_case(monkeypatch, inplace=inplace, budgets=budgets)
    assert case["error"] is None
    assert len(case["result"][1]) == 64
    assert case["tail"]["entered"] == 1
    assert case["tail"]["entry_loans"] == [0]
    assert case["tail"]["builtin_tuple"]
    assert len(case["receipts"]) == 8
    assert all(receipt._event.calls == 1 for receipt in case["receipts"])
    assert all(job["complete"] for job in case["jobs"])
    assert all(stream.drains == 0 for stream in case["streams"])


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("budgets", [(4, 12), (12, 4)])
def test_tail_interrupt_owns_builtin_tuple_before_unpack(monkeypatch, inplace, budgets):
    case = run_tail_case(monkeypatch, inplace=inplace, budgets=budgets, fault="interrupt")
    assert case["error"] is case["tail_primary"]
    assert case["tail"]["entry_loans"] == [0]
    assert case["tail"]["builtin_tuple"]
    assert case["tail"]["owned_at_handoff"]
    assert all(job["complete"] for job in case["jobs"])
    assert all(receipt._event.calls == 1 for receipt in case["receipts"])
    assert all(stream.drains == 0 for stream in case["streams"])
    async_safety.ensure_usable(case["evaluator"])


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("fault", ["callback", "native"])
def test_tail_consumer_error_does_not_retry_completed_receipt(monkeypatch, inplace, fault):
    case = run_tail_case(monkeypatch, inplace=inplace, fault=fault)
    assert case["error"] is case["tail_primary"]
    assert case["tail"]["entry_loans"] == [0]
    assert all(job["complete"] for job in case["jobs"])
    assert all(receipt._event.calls == 1 for receipt in case["receipts"])
    assert all(stream.drains == 0 for stream in case["streams"])
    async_safety.ensure_usable(case["evaluator"])


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("cleanup", ["event", "stream"])
def test_tail_cleanup_preserves_interrupt_and_backing_backend_quarantine(monkeypatch, inplace, cleanup):
    case = run_tail_case(monkeypatch, inplace=inplace, fault="interrupt", cleanup=cleanup)
    assert case["error"] is case["tail_primary"]
    assert any("pending receipt drain failed" in note for note in getattr(case["error"], "__notes__", ()))
    assert sum(stream.drains for stream in case["streams"]) == 1
    fresh = ThreadSafeGPUDispatcher(case["backend"])
    if cleanup == "stream":
        with pytest.raises(RuntimeError, match="poisoned"):
            async_safety.ensure_usable(case["evaluator"])
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
        async_safety.ensure_usable(fresh)
        fresh.get_input_buffer(1).fill(7)
