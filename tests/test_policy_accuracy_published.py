"""``policy_own_acc_top1`` and friends must actually REACH ``result.json``.

They were computed on every training iteration and every holdout eval, and
reached nothing -- so "is the policy head's move ordering improving?" got
answered off ``E[regret]``, which rewards SHARPNESS and therefore reported a
gain on 2026-08-08 while accuracy was flat (+0.7pp / -0.1pp, both ns) and ECE
got worse. A metric that is computed but not published is the same defect as a
knob that is read but not applied: it costs the compute and yields nothing.

⚑ ``sf_move_acc`` is NOT a substitute. It scores ``policy_sf``, the
opponent-reply head, whose weight the live branch parks at 0.0, so its movement
is an untrained head drifting under a moving trunk.

⚑ WHAT THESE TESTS DELIBERATELY DO NOT CLAIM. Publishing is all this PR does.
That ``policy_own_acc_top1`` *rises* is not evidence the loop improved: on the
TRAIN row it scores agreement with search(net) over rows the net was just
fitted to, which saturates by construction at a self-referential fixed point.
The ``test_`` twin is the generalisation reading, and ``policy_future_acc_*``
additionally carries the ``rl_loop_audit`` D18 mask leak.
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

# The denominators each accuracy divided by. On an empty denominator the
# accuracy publishes 0.0, which for an accuracy is the WORST attainable value
# and not a null -- these are what tell the two apart.
ROW_FIELDS = ("policy_own_acc_rows", "policy_future_acc_rows")

PUBLISHED = ACC_FIELDS + ROW_FIELDS


# `sf_move_acc` is a REQUIRED TrainMetrics field but is NOT a loss sum --
# `_build_metrics` passes it explicitly, so leaving it in the `sums` dict
# collides ("got multiple values for keyword argument").
_LOSS_SUMS: dict[str, float] = dict.fromkeys(
    ("loss", "policy_loss", "soft_policy_loss", "future_policy_loss", "wdl_loss",
     "sf_move_loss", "sf_eval_loss", "categorical_loss",
     "volatility_loss", "sf_volatility_loss", "moves_left_loss"), 0.0,
)

_REQUIRED = dict.fromkeys(
    ("loss", "policy_loss", "soft_policy_loss", "future_policy_loss", "wdl_loss",
     "sf_move_loss", "sf_move_acc", "sf_eval_loss", "categorical_loss",
     "volatility_loss", "sf_volatility_loss", "moves_left_loss"), 0.0,
)


def _metrics(**kw: float) -> TrainMetrics:
    return replace(TrainMetrics(**_REQUIRED), **kw)


def _test_row(test: TrainMetrics, *, train: TrainMetrics | None = None) -> dict:
    """⚑ `train` and `test` must be DISTINCT objects, and are by default.

    The first version of this helper passed ONE object as both `metrics` and
    `test_metrics`. Every assertion below still passed -- and so did mutating
    the source to `tr.metrics`, because the two names referred to the same
    object. The holdout row's SOURCE was therefore unpinned by a file whose
    entire purpose is pinning where these numbers come from. A reviewer found
    it by making that substitution and watching the suite stay green.
    """
    return _test_and_drift_dict(
        tr=TrainingResult(metrics=train if train is not None else _metrics(), test_metrics=test),
        drift=DriftMetrics(), holdout_frozen=True, holdout_generation=1,
    )


def test_the_train_row_carries_every_accuracy_field() -> None:
    m = _metrics(**{f: 0.25 + i / 100 for i, f in enumerate(PUBLISHED)})
    row = _train_metrics_dict(m)
    for i, f in enumerate(PUBLISHED):
        assert f in row, f"{f} is computed every iteration and never published"
        assert row[f] == 0.25 + i / 100, f"{f} published under the wrong source field"


def test_the_test_row_carries_every_accuracy_field() -> None:
    m = _metrics(**{f: 0.4 + i / 100 for i, f in enumerate(PUBLISHED)})
    row = _test_row(m)
    for i, f in enumerate(PUBLISHED):
        key = f"test_{f}"
        assert key in row, f"{key} missing from the holdout row"
        assert row[key] == 0.4 + i / 100


def test_the_holdout_row_reads_test_metrics_and_never_the_train_metrics() -> None:
    """The defect this pins: `test_*` silently sourced from the TRAIN metrics.

    A holdout column fed by train numbers is worse than an absent one -- it
    reads as generalisation and moves with the fit. Distinct values on the two
    objects are the only thing that can tell them apart.
    """
    train = _metrics(**dict.fromkeys(PUBLISHED, 0.11))
    test = _metrics(**dict.fromkeys(PUBLISHED, 0.88))
    row = _test_row(test, train=train)
    for f in PUBLISHED:
        assert row[f"test_{f}"] == 0.88, (
            f"test_{f} did not come from test_metrics -- 0.11 is the TRAIN value"
        )


def test_an_absent_holdout_publishes_nan_rather_than_a_number() -> None:
    """No holdout eval ran => NaN, not 0.0. A 0.0 accuracy is a legible
    measurement ("ranks nothing correctly"); NaN is the only value that cannot
    be mistaken for one, and it is what the surrounding row already does."""
    row = _test_and_drift_dict(
        tr=TrainingResult(metrics=_metrics(), test_metrics=None),
        drift=DriftMetrics(), holdout_frozen=True, holdout_generation=1,
    )
    for f in PUBLISHED:
        val = row[f"test_{f}"]
        assert val != val, f"test_{f} defaulted to {val!r}, not NaN"


def test_each_published_key_is_declared_so_a_row_is_never_ragged() -> None:
    """Ray's result table is built from the FIRST row's keys. A key published
    without a declared zero default appears only on iterations where it happens
    to be emitted, and every earlier row reads as missing rather than 0."""
    for f in PUBLISHED:
        assert f in _TRAIN_METRIC_DEFAULTS, f"{f} published without a zero default"
        assert f"test_{f}" in _TEST_METRIC_KEYS, f"test_{f} not declared"


def test_the_defaults_do_not_invent_a_value() -> None:
    """A zero default must be ZERO. Defaulting an accuracy to anything else
    would make a dead head look alive on iterations that never wrote it."""
    for f in PUBLISHED:
        assert _TRAIN_METRIC_DEFAULTS[f] == 0.0


def test_an_unset_metric_publishes_zero_rather_than_being_dropped() -> None:
    row = _train_metrics_dict(_metrics())
    for f in PUBLISHED:
        assert row.get(f) == 0.0


def test_a_zero_accuracy_is_distinguishable_from_an_unmeasured_one() -> None:
    """The whole reason `_rows` ships alongside. `_acc` returns 0.0 on an empty
    denominator, so without the count "the head ranked nothing correctly" and
    "no row in this batch carried the target" are the same published number.
    `has_future` is false near the end of a game, so the empty case is real."""
    measured = _train_metrics_dict(
        _metrics(policy_future_acc_top1=0.0, policy_future_acc_rows=4096.0),
    )
    absent = _train_metrics_dict(
        _metrics(policy_future_acc_top1=0.0, policy_future_acc_rows=0.0),
    )
    assert measured["policy_future_acc_top1"] == absent["policy_future_acc_top1"] == 0.0
    assert measured["policy_future_acc_rows"] != absent["policy_future_acc_rows"], (
        "the denominator is the ONLY thing separating these two rows"
    )


def test_build_metrics_publishes_the_denominator_the_accuracy_divided_by() -> None:
    """⚑ Pins the wiring in `trainer.py`, not just the publish site.

    Everything above constructs `TrainMetrics` by hand, so all of it stays
    green if `_build_metrics` fills `policy_own_acc_rows` from the wrong head,
    from the NUMERATOR, or from nothing. `_build_metrics` is a staticmethod;
    call it. (The author-scoped-mutation trap: mutating only the file the test
    lives in cannot see a binding made in another one.)
    """
    import torch

    from chess_anti_engine.train.trainer import Trainer

    def pair(num: float, den: float) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(num), torch.tensor(den)

    m = Trainer._build_metrics(
        sums=dict(_LOSS_SUMS),
        acc_sums={
            "policy_own_acc_top1": pair(64.0, 256.0),
            "policy_own_acc_top5": pair(192.0, 256.0),
            "policy_future_acc_top1": pair(3.0, 12.0),
        },
        n=1.0,
    )
    assert m.policy_own_acc_top1 == 0.25
    assert m.policy_own_acc_rows == 256.0, "the own-policy denominator, not the numerator or another head"
    assert m.policy_future_acc_rows == 12.0


def test_build_metrics_reports_zero_rows_for_a_head_that_never_ran() -> None:
    """The case the field exists for: accuracy 0.0 AND rows 0.0 together are
    the null. Either alone is indistinguishable from a head ranking nothing
    correctly over a full batch."""
    from chess_anti_engine.train.trainer import Trainer

    m = Trainer._build_metrics(sums=dict(_LOSS_SUMS), acc_sums={}, n=1.0)
    assert m.policy_own_acc_top1 == 0.0
    assert m.policy_own_acc_rows == 0.0
    assert m.policy_future_acc_rows == 0.0


def test_policy_own_is_distinguishable_from_sf_move_acc() -> None:
    """The two must not collapse onto one source. `sf_move_acc` scores an
    UNTRAINED head in production; reading it as "the net's accuracy" is the
    misreading this publish exists to end."""
    m = _metrics(sf_move_acc=0.99, policy_own_acc_top1=0.11)
    row = _train_metrics_dict(m)
    assert row["sf_move_acc"] == 0.99
    assert row["policy_own_acc_top1"] == 0.11
