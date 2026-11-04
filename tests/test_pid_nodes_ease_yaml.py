"""The YAML node-easing setting reaches the existing PID consumers."""
import pytest

from chess_anti_engine.stockfish.pid import pid_from_config
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults


@pytest.mark.parametrize("nested", [False, True])
def test_nodes_ease_yaml_reaches_construction_and_refresh(nested: bool) -> None:
    key = "sf_pid_nodes_ease_step_frac"
    values = {key: 0.25}
    config = flatten_run_config_defaults({"stockfish": values} if nested else values)
    pid = pid_from_config(config)
    assert pid.nodes_lever.ease_step_frac == 0.25

    values = {key: 0.4}
    fresh = flatten_run_config_defaults({"stockfish": values} if nested else values)
    pid.refresh_live_params(fresh)
    assert pid.nodes_lever.ease_step_frac == 0.4


def test_nested_nodes_ease_yaml_overrides_root_value() -> None:
    key = "sf_pid_nodes_ease_step_frac"
    config = flatten_run_config_defaults({key: 0.1, "stockfish": {key: 0.3}})
    assert pid_from_config(config).nodes_lever.ease_step_frac == 0.3


def test_nodes_ease_default_is_unchanged() -> None:
    assert pid_from_config(flatten_run_config_defaults({})).nodes_lever.ease_step_frac == 0.0
