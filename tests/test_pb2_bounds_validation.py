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
        [-1.0, 0.0],           # no sign/magnitude policy at this layer
        [0.3, 0.3],            # no magnitude cap here, by design
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
        (["3e-5", "3e-5"], "real numbers"),   # YAML reads `3e-5` as a string
        ([True, 1.0], "real numbers"),
        ([0.0, False], "real numbers"),
        ([None, 1.0], "real numbers"),
        ([math.nan, 1.0], "finite"),
        ([0.0, math.inf], "finite"),
        ([-math.inf, 0.0], "finite"),
        ([3e-4, 3e-5], "lo <= hi"),
    ],
)
def test_malformed_pb2_bounds_raise_naming_the_key(bounds, match):
    with pytest.raises(ValueError, match=match) as exc:
        flatten_run_config_defaults({"tune": {"pb2_bounds_lr": bounds}})
    assert "pb2_bounds_lr" in str(exc.value)


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
