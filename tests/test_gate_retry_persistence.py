"""Retry persistence regressions; normal project/pytest execution required.

The caller fragment retains the production retry branch (including continue),
without starting a trainable. File writer is substituted: these tests assert the
snapshot/restore contract, not atomic-writer durability.
"""
from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

from chess_anti_engine.tune import trainable, trainable_phases
from chess_anti_engine.tune.promotion_gate import (
    AnchoredSample,
    GateConfig,
    GateHoldController,
    PromotionGate,
)
from chess_anti_engine.tune.trial_config import DifficultyState, RestoreResult, SelfplayResult


def _retry_branch() -> Callable[..., bool]:
    source = ast.parse(inspect.getsource(trainable))
    branch = next(
        node for node in ast.walk(source)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "should_retry"
    )
    wrapper = ast.parse(
        "def run(sp, gate_hold, gate, gate_match_idx, gate_state_path):\n"
        "    for _ in (0,):\n"
        "        pass\n"
        "        return False\n"
        "    return True\n"
    ).body[0]
    assert isinstance(wrapper, ast.FunctionDef)
    loop = wrapper.body[0]
    assert isinstance(loop, ast.For)
    loop.body[0] = branch
    namespace: dict[str, object] = {"_persist_gate_state": trainable_phases._persist_gate_state}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "<retry-branch>", "exec"), namespace)
    return cast(Callable[..., bool], namespace["run"])


@pytest.fixture
def writer(monkeypatch):
    calls = []

    def write(path, text):
        calls.append(path)
        path.write_text(text, encoding="utf-8")

    monkeypatch.setattr(trainable_phases, "atomic_write_text", write)
    return calls


def _gate(holds=1, active=True):
    gate = PromotionGate(cfg=GateConfig(mode="enforce", max_hold_iters=4))
    gate.holds, gate.hold_active = holds, active
    gate.observe(AnchoredSample(iteration=9, cur_w=11, prev_w=12))
    hold = GateHoldController(gate, anchor_refresh_failures=2, fallback_missing=3)
    return gate, hold


@pytest.mark.parametrize(("holds", "expected_holds", "active"), [(1, 2, True), (3, 0, False)])
def test_retry_persists_age_and_release(tmp_path, writer, holds, expected_holds, active):
    gate, hold = _gate(holds)
    hold.hold_path = tmp_path / "inert-anchor"
    path = tmp_path / "gate_state.json"
    trainable_phases._persist_gate_state(gate=gate, hold=hold, gate_match_idx=17, gate_state_path=path)
    before = json.loads(path.read_text())
    assert _retry_branch()(SimpleNamespace(should_retry=True), hold, gate, 17, path)
    state = trainable._startup_gate_state(gate_state_path=path, restore=RestoreResult(restored_gate_state={"matches": 999}))
    restored = PromotionGate(cfg=gate.cfg)
    restored.load_state_dict(state)
    assert (restored.holds, restored.hold_active) == (expected_holds, active)
    assert (hold.hold_path is not None) == active
    assert state["matches"] == 17
    assert state["samples"] == before["samples"]
    assert state["hold"] == before["hold"] == hold.state_dict()
    assert len(writer) == 2


@pytest.mark.parametrize(("active", "retry"), [(False, True), (True, False)])
def test_inactive_and_no_retry_do_not_write(tmp_path, writer, active, retry):
    gate, hold = _gate(holds=1 if active else 0, active=active)
    before = gate.state_dict()
    path = tmp_path / "gate_state.json"
    assert _retry_branch()(SimpleNamespace(should_retry=retry), hold, gate, 17, path) == retry
    assert gate.state_dict() == before
    assert not path.exists()
    assert not writer


@pytest.mark.parametrize("failure", ["write", "json"])
def test_retry_write_failure_is_counted_and_nonfatal(tmp_path, monkeypatch, failure):
    gate, hold = _gate(holds=3)
    attempted = []
    if failure == "write":
        def fail(path, text):
            attempted.append((path, json.loads(text)))
            raise OSError("injected state write failure")
        monkeypatch.setattr(trainable_phases, "atomic_write_text", fail)
    else:
        monkeypatch.setattr(hold, "state_dict", lambda: {"invalid": object()})
    assert _retry_branch()(SimpleNamespace(should_retry=True), hold, gate, 17, tmp_path / "gate_state.json")
    assert (gate.holds, gate.hold_active, hold.hold_path) == (0, False, None)
    assert gate.state_write_failures == 1
    if failure == "write":
        assert len(attempted) == 1
        assert attempted[0][0] == tmp_path / "gate_state.json"
        assert attempted[0][1]["holds"] == 0


@pytest.mark.parametrize("with_hold", [False, True])
def test_normal_decision_snapshot_order_and_optional_hold(tmp_path, writer, with_hold):
    gate, hold = _gate(holds=0, active=False)
    supplied_hold = hold if with_hold else None
    path = tmp_path / "gate_state.json"
    scalars = []

    def scalar(name, value, step):
        scalars.append((name, value, step))

    trainer = SimpleNamespace(writer=SimpleNamespace(add_scalar=scalar))
    decision, matches = trainable_phases._run_net_gating(
        trainer, gate=gate, sp=SelfplayResult(gate_sample_valid=False),
        ds=DifficultyState(wdl_regret=0.0, sf_nodes=0), gate_match_idx=17, gate_state_path=path,
        iteration_idx=10, hold=supplied_hold,
    )
    state = json.loads(path.read_text())
    assert matches == state["matches"] == 18
    assert state["holds"] == decision.holds == gate.holds
    assert len(state["samples"]) == 2
    assert state["hold"] == (hold.state_dict() if with_hold else {})
    # The normal phase still snapshots before caller on_decision; factoring
    # must not apply a verdict to the actuator or change its prior counters.
    assert hold.last_decision is None
    assert hold.anchor_refresh_failures == 2
    assert hold.fallback_missing == 3
    assert len(writer) == 1
    assert scalars
