"""Node-budget invariants through real PID construction/live/restore consumers."""
from __future__ import annotations

from typing import Any

import pytest

from chess_anti_engine.stockfish.pid import DifficultyPID, pid_from_config


def make_pid(**overrides: Any) -> DifficultyPID:
    args = {"initial_nodes": 500, "min_nodes": 100, "max_nodes": 1000}
    args.update(overrides)
    return DifficultyPID(**args)


@pytest.mark.parametrize("field", ["initial_nodes", "min_nodes", "max_nodes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan"), float("inf"), "1.5", 2**53 + 1])
def test_constructor_refuses_invalid_node_counts(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="positive finite integer"):
        make_pid(**{field: value})


@pytest.mark.parametrize("field", ["sf_nodes", "sf_pid_min_nodes", "sf_pid_max_nodes"])
@pytest.mark.parametrize("value", [1.5, False, 0, float("inf")])
def test_config_factory_does_not_erase_invalid_counts_with_int(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="positive finite integer"):
        pid_from_config({field: value})


def test_constructor_normalizes_inversion_using_existing_live_ceiling_policy() -> None:
    pid = make_pid(min_nodes=800, max_nodes=200)
    assert (pid.min_nodes, pid.nodes, pid.max_nodes) == (200, 200, 200)
    assert (pid.metric_min_nodes, pid.metric_max_nodes) == (200, 200)


def test_integral_config_values_and_clamping_remain_supported() -> None:
    pid = pid_from_config({"sf_nodes": "2000", "sf_pid_min_nodes": "100", "sf_pid_max_nodes": 1000.0})
    assert (pid.min_nodes, pid.nodes, pid.max_nodes) == (100, 1000, 1000)
    pid.nodes = 50
    assert pid.nodes == 100


@pytest.mark.parametrize(("field", "value"), [
    ("sf_pid_min_nodes", 0), ("sf_pid_max_nodes", -1),
    ("sf_pid_min_nodes", 4.5), ("sf_pid_max_nodes", "broken"),
])
def test_invalid_live_pair_changes_neither_bound_lever_nor_other_params(field: str, value: Any) -> None:
    pid = make_pid()
    before = (pid.state_dict(), pid.target, pid.min_nodes, pid.max_nodes,
              pid.nodes_lever.min_value, pid.nodes_lever.max_value)
    edit: dict[str, Any] = {"sf_pid_min_nodes": 900, "sf_pid_target_winrate": 0.99, field: value}
    with pytest.raises(ValueError, match="positive finite integer"):
        pid.refresh_live_params(edit)
    assert before == (pid.state_dict(), pid.target, pid.min_nodes, pid.max_nodes,
                      pid.nodes_lever.min_value, pid.nodes_lever.max_value)


def test_unchanged_live_bounds_preserve_fractional_accumulator_and_frozen_metric() -> None:
    pid = make_pid()
    pid.nodes_lever.value = 500.125
    pid.refresh_live_params({"sf_pid_min_nodes": 100, "sf_pid_max_nodes": 1000})
    assert pid.nodes_lever.value == 500.125
    pid.refresh_live_params({"sf_pid_min_nodes": 800})
    assert pid.nodes == 800
    pid.refresh_live_params({"sf_pid_max_nodes": 200})
    assert (pid.min_nodes, pid.nodes, pid.max_nodes) == (200, 200, 200)
    assert (pid.metric_min_nodes, pid.metric_max_nodes) == (100, 1000)


@pytest.mark.parametrize(("key", "value"), [("nodes", 0), ("metric_min_nodes", -1), ("metric_max_nodes", 1.5)])
def test_invalid_restored_node_fields_refuse_before_state_change(key: str, value: Any) -> None:
    pid = make_pid()
    before = pid.state_dict()
    state = {"nodes": 900, "metric_min_nodes": 200, "metric_max_nodes": 800, key: value}
    with pytest.raises(ValueError, match="positive finite integer"):
        pid.load_state_dict(state)
    assert pid.state_dict() == before


def test_restore_keeps_metric_bounds_separate_from_live_bounds() -> None:
    original = make_pid()
    original.refresh_live_params({"sf_pid_min_nodes": 800})
    restored = make_pid(min_nodes=800)
    restored.load_state_dict(original.state_dict())
    assert restored.nodes == 800
    assert (restored.min_nodes, restored.max_nodes) == (800, 1000)
    assert (restored.metric_min_nodes, restored.metric_max_nodes) == (100, 1000)
