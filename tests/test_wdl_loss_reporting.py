"""The reported `wdl_loss` must be the loss the optimizer sees.

docs/rl_loop_audit.md I7: `wdl_loss` used to carry a hard one-hot CE that no
gradient came from, and everything derived from it (`test_wdl_loss`, the
per-source and per-phase splits) inherited the defect.
"""
from __future__ import annotations

import torch

from chess_anti_engine.train.losses import compute_loss
from chess_anti_engine.train.trainer import (
    _LOSS_KEY_TO_METRIC_FIELD,
    _TRAIN_METRICS_FIELDS,
    _loss_sums_to_metric_kwargs,
)
from chess_anti_engine.tune.trainable_report import (
    _TEST_METRIC_KEYS,
    _TRAIN_METRIC_DEFAULTS,
)


def test_trained_loss_maps_to_the_wdl_loss_column() -> None:
    assert _LOSS_KEY_TO_METRIC_FIELD["wdl_ce"] == "wdl_loss"
    assert _LOSS_KEY_TO_METRIC_FIELD["blended_wdl_ce"] == "blended_wdl_loss"
    assert _LOSS_KEY_TO_METRIC_FIELD["wdl_onehot_ce"] == "wdl_onehot_loss"
    for field in ("wdl_loss", "blended_wdl_loss", "wdl_onehot_loss"):
        assert field in _TRAIN_METRICS_FIELDS


def test_metric_kwargs_carry_the_blended_value_into_wdl_loss() -> None:
    sums = {"wdl_ce": 4.0, "blended_wdl_ce": 4.0, "wdl_onehot_ce": 2.0}
    kwargs = _loss_sums_to_metric_kwargs(sums, 2.0)
    assert kwargs["wdl_loss"] == 2.0
    assert kwargs["blended_wdl_loss"] == 2.0
    assert kwargs["wdl_onehot_loss"] == 1.0


def test_every_reported_loss_key_reaches_a_metric_field() -> None:
    """A compute_loss key that maps to nothing is silently dropped — catch that."""
    b = 4
    batch = {
        "x": torch.randn((b, 146, 8, 8)),
        "policy_t": torch.full((b, 1858), 1.0 / 1858.0),
        "wdl_t": torch.randint(0, 3, (b,)),
        "has_policy": torch.ones((b,)),
    }
    outputs = {
        "policy_own": torch.randn((b, 1858)),
        "wdl": torch.randn((b, 3)),
    }
    losses = compute_loss(outputs, batch)
    for key in ("wdl_ce", "blended_wdl_ce", "wdl_onehot_ce"):
        field = _LOSS_KEY_TO_METRIC_FIELD.get(key, key)
        assert field in _TRAIN_METRICS_FIELDS, f"{key} would be dropped"
        assert key in losses


def test_report_columns_exist_for_both_quantities() -> None:
    for key in ("wdl_loss", "blended_wdl_loss", "wdl_onehot_loss"):
        assert key in _TRAIN_METRIC_DEFAULTS
    assert "test_wdl_loss" in _TEST_METRIC_KEYS
    assert "test_wdl_onehot_loss" in _TEST_METRIC_KEYS
