"""PID node-bound invariants: loader range check, atomic live update, safe restore.

Adapted from #1110's test_pid_node_invariants.py with a different contract:
a bad node bound is REJECTED BY THE YAML LOADER (a live edit then rejects the
whole reload and the old config stays in effect; at launch it is fatal), while
the PID's own live/restore paths never raise and never half-apply -- a bad
value that arrives by another route keeps the previous bounds and warns.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from chess_anti_engine.stockfish.pid import DifficultyPID, pid_from_config, positive_node_count
from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

BAD_COUNTS: list[Any] = [0, -1, True, False, 1.5, float("nan"), float("inf"), "1.5", "broken", None, [1]]


def make_pid(**overrides: Any) -> DifficultyPID:
    args: dict[str, Any] = {"initial_nodes": 500, "min_nodes": 100, "max_nodes": 1000}
    args.update(overrides)
    return DifficultyPID(**args)


def _bounds(pid: DifficultyPID) -> tuple[int, int, float, float]:
    return (pid.min_nodes, pid.max_nodes, pid.nodes_lever.min_value, pid.nodes_lever.max_value)


# ---------------------------------------------------------------- the rule
@pytest.mark.parametrize("value", BAD_COUNTS)
def test_positive_node_count_refuses_what_int_would_truncate_or_zero(value: Any) -> None:
    assert positive_node_count(value) is None


@pytest.mark.parametrize(("value", "expected"), [(1, 1), (50000, 50000), (50000.0, 50000), ("75000", 75000)])
def test_positive_node_count_keeps_what_int_always_accepted(value: Any, expected: int) -> None:
    assert positive_node_count(value) == expected


# ---------------------------------------------------------------- the loader
@pytest.mark.parametrize("key", ["sf_pid_min_nodes", "sf_pid_max_nodes"])
@pytest.mark.parametrize("value", [v for v in BAD_COUNTS if v is not None])
def test_loader_rejects_bad_node_bounds(key: str, value: Any) -> None:
    """MUTATION: drop ``_check_pid_node_bounds`` from the loader -- every case passes the load."""
    cfg = {"stockfish": {"sf_pid_min_nodes": 100, "sf_pid_max_nodes": 1000, key: value}}
    with pytest.raises(ValueError, match=key):
        flatten_run_config_defaults(cfg)


def test_loader_rejects_an_inverted_pair_including_the_realized_default_ceiling() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        flatten_run_config_defaults({"stockfish": {"sf_pid_min_nodes": 2000, "sf_pid_max_nodes": 1000}})
    # max absent: pid_from_config realizes 50000, so min 60000 is still inverted.
    with pytest.raises(ValueError, match="must not exceed"):
        flatten_run_config_defaults({"stockfish": {"sf_pid_min_nodes": 60000}})


def test_loader_passes_valid_values_through_unchanged() -> None:
    flat = flatten_run_config_defaults({"stockfish": {
        "sf_pid_min_nodes": 50000, "sf_pid_max_nodes": 1000000.0, "sf_nodes": 75000,
    }})
    # Validation only: the raw value reaches the consumer exactly as written.
    assert flat["sf_pid_min_nodes"] == 50000
    assert flat["sf_pid_max_nodes"] == 1000000.0
    pid = pid_from_config(flat)
    assert (pid.min_nodes, pid.nodes, pid.max_nodes) == (50000, 75000, 1000000)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SHIPPED_CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))


def test_the_shipped_config_set_is_not_empty() -> None:
    """An empty parameter set would SKIP the gate below, not fail it."""
    assert len(SHIPPED_CONFIGS) >= 2
    assert any(p.parent.name == "snapshots" for p in SHIPPED_CONFIGS)


@pytest.mark.parametrize(
    "path", SHIPPED_CONFIGS, ids=lambda p: str(p.relative_to(CONFIG_DIR)),
)
def test_every_shipped_config_still_loads(path: Path) -> None:
    flatten_run_config_defaults(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_bad_live_edit_rejects_the_whole_reload_and_keeps_the_old_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Class (a) behaviour: the reloader keeps every old value, not just this key."""
    doc = {"stockfish": {"sf_pid_min_nodes": 50000, "sf_pid_max_nodes": 1000000,
                         "sf_pid_target_winrate": 0.5}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    config = dict(flatten_run_config_defaults(doc))
    before = dict(config)
    doc["stockfish"]["sf_pid_min_nodes"] = 0.5  # the decimal typo
    doc["stockfish"]["sf_pid_target_winrate"] = 0.9  # a sibling edit in the same save
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        _reload_yaml_into_config(config, str(path), live_reload=True)
    assert config == before
    assert any("YAML reload failed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- live refresh
@pytest.mark.parametrize(("field", "value"), [
    ("sf_pid_min_nodes", 0), ("sf_pid_max_nodes", -1), ("sf_pid_min_nodes", 4.5),
    ("sf_pid_max_nodes", "broken"), ("sf_pid_max_nodes", None),
])
def test_bad_live_bound_keeps_both_previous_bounds_and_warns(
    field: str, value: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    """Never raise, never half-apply; other live knobs still apply.

    MUTATION A: assign ``min`` before parsing ``max`` -- the min moves to 900.
    MUTATION B: raise instead of warning -- the call raises.
    """
    pid = make_pid()
    before = (_bounds(pid), pid.nodes, pid.nodes_lever.value)
    edit: dict[str, Any] = {"sf_pid_min_nodes": 900, "sf_pid_target_winrate": 0.99, field: value}
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.stockfish.pid"):
        pid.refresh_live_params(edit)
    assert (_bounds(pid), pid.nodes, pid.nodes_lever.value) == before
    assert pid.target == pytest.approx(0.99)
    assert any("PID node bounds NOT changed" in r.getMessage() for r in caplog.records)


def test_unchanged_live_bounds_preserve_fractional_accumulator() -> None:
    pid = make_pid()
    pid.nodes_lever.value = 500.125
    for _ in range(3):
        pid.refresh_live_params({"sf_pid_min_nodes": 100, "sf_pid_max_nodes": 1000})
    assert pid.nodes_lever.value == 500.125


def test_live_ratchet_and_inversion_follow_the_existing_policy() -> None:
    pid = make_pid()
    pid.refresh_live_params({"sf_pid_min_nodes": 800, "sf_pid_max_nodes": 1000})
    assert (pid.min_nodes, pid.nodes, pid.max_nodes) == (800, 800, 1000)
    # Inverted pair by a non-loader route: floor yields to the ceiling.
    pid.refresh_live_params({"sf_pid_min_nodes": 800, "sf_pid_max_nodes": 200})
    assert _bounds(pid) == (200, 200, 200.0, 200.0)
    assert pid.nodes == 200
    # The metric normalizer stays frozen at construction.
    assert (pid.metric_min_nodes, pid.metric_max_nodes) == (100, 1000)
    # Integral float / int string are the same bound, so nothing moves.
    pid.nodes_lever.value = 200.0
    pid.refresh_live_params({"sf_pid_min_nodes": 200.0, "sf_pid_max_nodes": "200"})
    assert _bounds(pid) == (200, 200, 200.0, 200.0)


# ---------------------------------------------------------------- restore
def test_restore_keeps_metric_bounds_separate_from_live_bounds() -> None:
    original = make_pid()
    original.refresh_live_params({"sf_pid_min_nodes": 800})
    restored = make_pid(min_nodes=800)
    restored.load_state_dict(original.state_dict())
    assert restored.nodes == 800
    assert (restored.min_nodes, restored.max_nodes) == (800, 1000)
    assert (restored.metric_min_nodes, restored.metric_max_nodes) == (100, 1000)


def test_legacy_state_with_odd_values_still_loads_everything_else(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bad node field keeps the built value; the rest of the state restores.

    MUTATION: let the nodes/metric fields raise again -- ``_init_pid`` would
    then drop the WHOLE state (winrate EMA, lever histories) for one field.
    """
    source = make_pid()
    source.ema_winrate = 0.37
    state = source.state_dict()
    state.update({"nodes": "not-a-number", "metric_min_nodes": 0, "metric_max_nodes": 1.5})
    pid = make_pid()
    built = (pid.nodes, pid.metric_min_nodes, pid.metric_max_nodes)
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.stockfish.pid"):
        pid.load_state_dict(state)
    assert (pid.nodes, pid.metric_min_nodes, pid.metric_max_nodes) == built
    assert pid.ema_winrate == pytest.approx(0.37)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "nodes='not-a-number'" in messages
    assert "metric_min_nodes=0" in messages


@pytest.mark.parametrize(("raw", "expected"), [(698289.0, 1000), (750.9, 750), (5, 100), ("640", 640)])
def test_numeric_legacy_nodes_restore_exactly_as_before(raw: Any, expected: int) -> None:
    """Finite numbers are truncated then clamped -- the pre-existing behaviour."""
    pid = make_pid()
    pid.load_state_dict({"nodes": raw})
    assert pid.nodes == expected


def test_state_without_metric_keys_falls_back_to_the_built_values() -> None:
    pid = make_pid(min_nodes=200, max_nodes=900)
    pid.load_state_dict({"nodes": 400})
    assert (pid.metric_min_nodes, pid.metric_max_nodes) == (200, 900)


# ---------------------------------------------------------------- observability
def test_a_live_bound_change_logs_old_and_new_bounds_and_nodes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MUTATION: drop the change log -- the realized value is invisible unless the clamp bites."""
    pid = make_pid()
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.stockfish.pid"):
        pid.refresh_live_params({"sf_pid_min_nodes": 100, "sf_pid_max_nodes": 2000})
        pid.refresh_live_params({"sf_pid_min_nodes": 100, "sf_pid_max_nodes": 2000})  # no change
        pid.refresh_live_params({"sf_pid_min_nodes": 800, "sf_pid_max_nodes": 2000})
    lines = [r.getMessage() for r in caplog.records if "PID node bounds [" in r.getMessage()]
    assert lines == [
        "PID node bounds [100,1000] -> [100,2000], nodes 500 -> 500",
        "PID node bounds [100,2000] -> [800,2000], nodes 500 -> 800",
    ]
    assert all(r.levelno == logging.WARNING for r in caplog.records if "PID node bounds [" in r.getMessage())


def _real_shaped_state() -> dict[str, Any]:
    pid = make_pid()
    for w, d, l in [(30, 10, 5), (12, 20, 18), (40, 2, 3)]:
        pid.observe(wins=w, draws=d, losses=l, force=True)
    return pid.state_dict()


# The keys pid_state.json carried before this change; the restore flag is NOT persisted.
PERSISTED_KEYS = {
    "nodes", "wdl_regret", "ema_winrate", "games_since_adjust", "held_wins",
    "held_draws", "held_losses", "regret_stage_complete", "last_regret_fit_slope",
    "regret_history", "nodes_history", "metric_min_nodes", "metric_max_nodes",
}


def test_clean_restore_reports_zero_and_round_trips_byte_identically() -> None:
    state = _real_shaped_state()
    assert set(state) == PERSISTED_KEYS
    pid = make_pid()
    pid.load_state_dict(json.loads(json.dumps(state)))
    assert pid.restore_fallbacks == 0
    assert json.dumps(pid.state_dict(), sort_keys=True) == json.dumps(state, sort_keys=True)


@pytest.mark.parametrize(("corrupt", "count"), [
    ({"nodes": "698289x"}, 1), ({"metric_min_nodes": 0}, 1),
    ({"nodes": None, "metric_max_nodes": "x"}, 2),
])
def test_corrupt_restore_sets_the_flag_and_logs_error(
    corrupt: dict[str, Any], count: int, caplog: pytest.LogCaptureFixture,
) -> None:
    """MUTATION: stop counting (or log at WARNING) -- the step at restart goes unseen."""
    state = {**_real_shaped_state(), **corrupt}
    pid = make_pid()
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.stockfish.pid"):
        pid.load_state_dict(state)
    assert pid.restore_fallbacks == count
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == count
    for key, value in corrupt.items():
        assert any(f"{key}={value!r}" in m for m in errors), errors
    # A later clean restore clears it.
    pid.load_state_dict(_real_shaped_state())
    assert pid.restore_fallbacks == 0


def test_restore_flag_reaches_the_iteration_result_on_the_production_path() -> None:
    """``_run_pid_and_eval`` -> ``PidResult`` -> ``_pid_report_dict`` (the result dict).

    MUTATION: drop the field from ``_run_pid_and_eval``'s PidResult -- the key reads 0.
    """
    from chess_anti_engine.tune.trainable_phases import _run_pid_and_eval
    from chess_anti_engine.tune.trainable_report import _pid_report_dict
    from chess_anti_engine.tune.trial_config import DifficultyState, SelfplayResult, TrialConfig

    tc = TrialConfig.from_dict({"device": "cpu", "sf_nodes": 500})
    for corrupt, expected in [({}, 0), ({"nodes": "bad"}, 1)]:
        pid = make_pid()
        pid.load_state_dict({**_real_shaped_state(), **corrupt})
        for total in (0, 40):  # with and without a PID update this iteration
            result = _run_pid_and_eval(
                tc=tc, config={}, pid=pid, sf=None,
                sp_result=SelfplayResult(
                    total_w=total, total_d=0, total_l=0,
                    matching_games=max(1, total), total_selfplay_games=1,
                ),
                opp_strength_ema=0.0, opp_ema_alpha=0.1,
                ds=DifficultyState(sf_nodes=pid.nodes, wdl_regret=-1.0),
            )
            assert _pid_report_dict(result)["pid_restore_fallbacks"] == expected


def test_init_pid_marks_a_restore_that_raised(caplog: pytest.LogCaptureFixture) -> None:
    from types import SimpleNamespace

    from chess_anti_engine.tune.trainable import _init_pid

    tc = SimpleNamespace(sf_pid_enabled=True)
    config = {"sf_nodes": 500, "sf_pid_min_nodes": 100, "sf_pid_max_nodes": 1000}
    bad = {**_real_shaped_state(), "ema_winrate": "not-a-float"}
    with caplog.at_level(logging.ERROR, logger="chess_anti_engine.tune.trainable"):
        pid = _init_pid(tc, config, bad, None)  # pyright: ignore[reportArgumentType]
    assert pid is not None
    assert pid.restore_fallbacks == 3
    assert any("PID state restore FAILED" in r.getMessage() and "nodes=" in r.getMessage()
               for r in caplog.records)
    clean = _init_pid(tc, config, _real_shaped_state(), None)  # pyright: ignore[reportArgumentType]
    assert clean is not None
    assert clean.restore_fallbacks == 0
