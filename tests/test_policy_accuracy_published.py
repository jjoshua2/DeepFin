"""``policy_own_acc_top1`` and friends must actually REACH ``result.json``.

They were computed on every training iteration and every holdout eval, and
reached nothing -- so "is the policy head's move ordering improving?" got
answered off ``E[regret]``, which rewards SHARPNESS and therefore reported a
gain on 2026-08-08 while accuracy was flat (+0.7pp / -0.1pp, both ns) and ECE
got worse. A metric that is computed but not published is the same defect as a
knob that is read but not applied: it costs the compute and yields nothing.

⚑ ``sf_move_acc`` is NOT a substitute and never was. It scores ``policy_sf``,
the opponent-reply head, which production leaves untrained (``w_sf_move: 0.0``),
so its movement is an untrained head drifting under a moving trunk.
"""
from __future__ import annotations

from dataclasses import replace

from chess_anti_engine.train.trainer import TrainMetrics
from chess_anti_engine.tune.trainable_report import (
    _TEST_METRIC_KEYS,
    _TRAIN_METRIC_DEFAULTS,
    _train_metrics_dict,
    _test_and_drift_dict,
)
from chess_anti_engine.tune.trial_config import DriftMetrics, TrainingResult

ACC_FIELDS = (
    "policy_own_acc_top1",
    "policy_own_acc_top5",
    "policy_future_acc_top1",
    "policy_future_acc_top5",
)


_REQUIRED = dict.fromkeys(
    ("loss", "policy_loss", "soft_policy_loss", "future_policy_loss", "wdl_loss",
     "sf_move_loss", "sf_move_acc", "sf_eval_loss", "categorical_loss",
     "volatility_loss", "sf_volatility_loss", "moves_left_loss"), 0.0,
)


def _metrics(**kw: float) -> TrainMetrics:
    return replace(TrainMetrics(**_REQUIRED), **kw)


def _test_row(m: TrainMetrics) -> dict:
    return _test_and_drift_dict(
        tr=TrainingResult(metrics=m, test_metrics=m),
        drift=DriftMetrics(), holdout_frozen=True, holdout_generation=1,
    )


def test_the_train_row_carries_every_accuracy_field() -> None:
    m = _metrics(**{f: 0.25 + i / 100 for i, f in enumerate(ACC_FIELDS)})
    row = _train_metrics_dict(m)
    for i, f in enumerate(ACC_FIELDS):
        assert f in row, f"{f} is computed every iteration and never published"
        assert row[f] == 0.25 + i / 100, f"{f} published under the wrong source field"


def test_the_test_row_carries_every_accuracy_field() -> None:
    m = _metrics(**{f: 0.4 + i / 100 for i, f in enumerate(ACC_FIELDS)})
    row = _test_row(m)
    for i, f in enumerate(ACC_FIELDS):
        key = f"test_{f}"
        assert key in row, f"{key} missing from the holdout row"
        assert row[key] == 0.4 + i / 100


def test_each_published_key_is_declared_so_a_row_is_never_ragged() -> None:
    """Ray's result table is built from the FIRST row's keys. A key published
    without a declared zero default appears only on iterations where it happens
    to be emitted, and every earlier row reads as missing rather than 0."""
    for f in ACC_FIELDS:
        assert f in _TRAIN_METRIC_DEFAULTS, f"{f} published without a zero default"
        assert f"test_{f}" in _TEST_METRIC_KEYS, f"test_{f} not declared"


def test_the_defaults_do_not_invent_a_value() -> None:
    """A zero default must be ZERO. Defaulting an accuracy to anything else
    would make a dead head look alive on iterations that never wrote it."""
    for f in ACC_FIELDS:
        assert _TRAIN_METRIC_DEFAULTS[f] == 0.0


def test_an_unset_metric_publishes_zero_rather_than_being_dropped() -> None:
    row = _train_metrics_dict(_metrics())
    for f in ACC_FIELDS:
        assert row.get(f) == 0.0


def test_policy_own_is_distinguishable_from_sf_move_acc() -> None:
    """The two must not collapse onto one source. `sf_move_acc` scores an
    UNTRAINED head in production; reading it as "the net's accuracy" is the
    misreading this publish exists to end."""
    m = _metrics(sf_move_acc=0.99, policy_own_acc_top1=0.11)
    row = _train_metrics_dict(m)
    assert row["sf_move_acc"] == 0.99
    assert row["policy_own_acc_top1"] == 0.11
