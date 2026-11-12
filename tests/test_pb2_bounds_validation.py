"""Loader-side validation of ``tune.pb2_bounds_*``.

``pb2_bounds_lr`` -- not ``train.lr`` -- sets the production learning rate:
``tune/harness.py:_build_mutation_param_space`` replaces ``lr`` with
``tune.uniform(*bounds)`` at launch. A malformed entry used to be either
dropped silently by ``_collect_mutation_bounds`` or left to fail inside Ray.
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import pytest

from chess_anti_engine.tune.harness import _collect_mutation_bounds
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "bounds",
    [
        [3e-5, 3e-5],          # production: a pinned (degenerate) interval
        [0.0001, 0.0003],
        (1, 2),                # ints and tuples are real-number pairs too
        [0.0, 0.0],            # zero is a legal LR endpoint
        [-0.0, 3e-5],          # -0.0 == 0, accepted like the scalar `lr`
        [0.3, 0.3],            # no magnitude cap here, by design
        ["3e-5", "3e-5"],      # YAML reads `3e-5` as a string; works on main
        ["0.1", 0.2],          # mixed string/number
        [" 1e-4 ", "2e-4"],    # float() tolerates surrounding whitespace
    ],
)
def test_valid_pb2_bounds_are_accepted_and_reach_the_harness(bounds):
    flat = flatten_run_config_defaults({"tune": {"pb2_bounds_lr": bounds}})
    assert flat["pb2_bounds_lr"] == bounds
    assert _collect_mutation_bounds(flat) == {"lr": [float(bounds[0]), float(bounds[1])]}


@pytest.mark.parametrize(
    ("bounds", "match"),
    [
        (3e-5, "two-element"),
        ("3e-5", "two-element"),
        ([3e-5], "two-element"),
        ([1e-5, 2e-5, 3e-5], "two-element"),
        ({"lo": 1e-5, "hi": 2e-5}, "two-element"),
        (["bad", 1.0], "real numbers"),
        (["", 1.0], "real numbers"),
        (["0,1", 1.0], "real numbers"),
        ([[0.1], 1.0], "real numbers"),
        ([True, 1.0], "real numbers"),
        ([0.0, False], "real numbers"),
        ([None, 1.0], "real numbers"),
        ([math.nan, 1.0], "finite"),
        ([0.0, math.inf], "finite"),
        ([-math.inf, 0.0], "finite"),
        (["1e400", 1.0], "finite"),           # float("1e400") is inf
        (["nan", 1.0], "finite"),
        (["-inf", "0"], "finite"),
        ([10**400, 1.0], "real numbers"),     # float() overflows on the int
        (["3e-4", "3e-5"], "lo <= hi"),       # ordered on the realized floats
        ([3e-4, 3e-5], "lo <= hi"),
        ([-1.0, 0.0], "must not be negative"),     # production draws its LR here
        ([-1e-3, 3e-5], "must not be negative"),
        (["-1e-3", "3e-5"], "must not be negative"),
    ],
)
def test_malformed_pb2_bounds_raise_naming_the_key(bounds, match):
    with pytest.raises(ValueError, match=match) as exc:
        flatten_run_config_defaults({"tune": {"pb2_bounds_lr": bounds}})
    assert "pb2_bounds_lr" in str(exc.value)


def test_negative_bounds_stay_legal_for_keys_other_than_lr():
    # The sign rule is a learning-rate rule; other searched keys may be signed.
    flat = flatten_run_config_defaults({"tune": {"pb2_bounds_w_value": [-1.0, 0.0]}})
    assert _collect_mutation_bounds(flat) == {"w_value": [-1.0, 0.0]}


def test_every_pb2_bounds_key_is_checked_not_only_lr():
    with pytest.raises(ValueError, match="pb2_bounds_w_value"):
        flatten_run_config_defaults(
            {"tune": {"pb2_bounds_lr": [3e-5, 3e-5], "pb2_bounds_w_value": [1.0, 0.5]}}
        )


def test_review_reproducer_inverted_and_explicit_null():
    # The review's evidence config stays loadable (no cap), so the bound is
    # visible to the harness rather than silently replaced.
    flat = flatten_run_config_defaults(
        {"train": {"lr": 3e-5}, "tune": {"pb2_bounds_lr": [0.3, 0.3]}}
    )
    assert _collect_mutation_bounds(flat) == {"lr": [0.3, 0.3]}
    # An explicit null is dropped by the loader before the check (not searched).
    assert "pb2_bounds_lr" not in flatten_run_config_defaults({"tune": {"pb2_bounds_lr": None}})


@pytest.mark.parametrize(
    "path", sorted(glob.glob(str(_REPO / "configs" / "*.yaml"))), ids=lambda p: Path(p).name
)
def test_checked_in_configs_still_load(path):
    flatten_run_config_defaults(load_yaml_file(path))


def test_mid_run_reload_with_bad_bounds_keeps_the_old_config(tmp_path):
    """Mid-run the raise rejects the WHOLE reload (category (a) behaviour):
    the other edit in the same file is not applied, and nothing crashes."""
    from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "train:\n  w_policy: 2.5\ntune:\n  pb2_bounds_lr: [0.0003, 0.00003]\n",
        encoding="utf-8",
    )
    config = {"w_policy": 1.0, "lr": 3e-5, "pb2_bounds_lr": [3e-5, 3e-5]}
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config == {"w_policy": 1.0, "lr": 3e-5, "pb2_bounds_lr": [3e-5, 3e-5]}

    yaml_path.write_text(
        "train:\n  w_policy: 2.5\ntune:\n  pb2_bounds_lr: [0.00003, 0.00003]\n",
        encoding="utf-8",
    )
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config["w_policy"] == 2.5


def test_numeric_string_bounds_from_yaml_reach_the_harness_as_floats(tmp_path):
    """End to end from YAML text: the unquoted `3e-5` form loads as strings,
    passes the check, and `_collect_mutation_bounds` realizes the SAME floats
    that a decimal-point spelling would."""
    path = tmp_path / "b.yaml"
    path.write_text("tune:\n  pb2_bounds_lr: [3e-5, 3e-4]\n", encoding="utf-8")
    flat = flatten_run_config_defaults(load_yaml_file(str(path)))
    assert flat["pb2_bounds_lr"] == ["3e-5", "3e-4"]
    assert _collect_mutation_bounds(flat) == {"lr": [3e-5, 3e-4]}


# Ported from draft PR #1213 (fix/yaml-pb2-bounds-20261111), which duplicated
# this check; adapted to this file. The `'0,1'`/`[bad, 1]` cases stay rejected
# under the numeric-string policy because they do not parse as floats.
@pytest.mark.parametrize("searched", [False, True])
@pytest.mark.parametrize("live_reload", [False, True])
@pytest.mark.parametrize(
    "bounds_yaml",
    ["0.1", "[]", "[0.1]", "[0, 1, 2]", "'0,1'", "[bad, 1]",
     "[null, 1]", "[.nan, 1]", "[0, .inf]", "[-.inf, 1]", "[2, 1]", "[1e400, 1]"],
)
def test_pb2_bounds_reload_rejects_atomically_and_recovers(
    tmp_path: Path, searched: bool, live_reload: bool, bounds_yaml: str,
) -> None:
    from chess_anti_engine.tune.trainable_config_ops import (
        _reload_yaml_into_config,
        last_reload_yaml_keys,
        reset_yaml_reload_key_tracking,
    )

    reset_yaml_reload_key_tracking()
    path = tmp_path / "pb2.yaml"
    config: dict = {"lr": 0.003, "batch_size": 32}
    if searched:
        config["pb2_bounds_lr"] = [0.001, 0.01]
    path.write_text("lr: 0.003\nbatch_size: 32\n", encoding="utf-8")
    _reload_yaml_into_config(config, str(path), live_reload=live_reload)
    before = dict(config)
    baseline = last_reload_yaml_keys(str(path))

    # A bad bound must not apply even unrelated eligible siblings or advance
    # the baseline. Include another valid bound so silently dropping the bad
    # one would still leave a nonempty PB2 search space.
    path.write_text(
        "lr: 0.009\nbatch_size: 99\nw_wdl: 0.8\n"
        f"tune:\n  pb2_bounds_lr: {bounds_yaml}\n"
        "  pb2_bounds_w_wdl: [0, 1]\n",
        encoding="utf-8",
    )
    _reload_yaml_into_config(config, str(path), live_reload=live_reload)
    assert config == before
    assert last_reload_yaml_keys(str(path)) == baseline

    path.write_text(
        "lr: 0.009\nbatch_size: 64\nw_wdl: 0.9\n"
        "tune:\n  pb2_bounds_lr: [0.002, 0.02]\n",
        encoding="utf-8",
    )
    _reload_yaml_into_config(config, str(path), live_reload=live_reload)
    assert config["batch_size"] == 64
    assert config["w_wdl"] == 0.9
    assert config["lr"] == (0.003 if searched else 0.009)
    # Bounds remain launch-time settings, including on startup/resume reload.
    if searched:
        assert config["pb2_bounds_lr"] == [0.001, 0.01]
    else:
        assert "pb2_bounds_lr" not in config
    assert "pb2_bounds_w_wdl" not in config
    assert last_reload_yaml_keys(str(path)) == frozenset(
        {"lr", "batch_size", "w_wdl", "pb2_bounds_lr"}
    )
