from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from chess_anti_engine.train.trainer import TrainMetrics
from chess_anti_engine.tune.trainable_report import (
    _STATUS_COLS,
    _TRAIN_METRIC_DEFAULTS,
    _pid_report_dict,
    _pid_step_diag_dict,
    _train_metrics_dict,
)


def _full_diag() -> SimpleNamespace:
    return SimpleNamespace(
        reason="fit", changed=True, value_before=0.05, value_after=0.04,
        applied_delta=-0.01, raw_delta=-0.02, cap=0.1, observation_se=0.03,
        raw_deadband=0.01, ema_deadband=0.01, history_len=5,
        tighten_gain_applied=1.0, crash_ease_applied=False,
        predicted_value=0.5, fit_slope=2.0,
    )


def test_pid_step_diag_dict_key_set_is_stable() -> None:
    """A lever that didn't run (diag=None) must emit the SAME keys as one that did,
    so Ray's first-row-fixes-the-header CSV logger never sees a varying schema."""
    assert set(_pid_step_diag_dict("pid_regret", None)) == set(
        _pid_step_diag_dict("pid_regret", _full_diag())
    )


def test_pid_report_dict_key_set_is_stable() -> None:
    """The PID report block has an identical key set whether or not an update ran
    and regardless of which levers were active."""
    pr_none = SimpleNamespace(pid_update=None)
    pr_regret_only = SimpleNamespace(pid_update=SimpleNamespace(
        raw_winrate=0.6, observation_se=0.03, regret_frozen=False, nodes_active=False,
        regret_diag=_full_diag(), nodes_diag=None,
    ))
    pr_both = SimpleNamespace(pid_update=SimpleNamespace(
        raw_winrate=0.6, observation_se=0.03, regret_frozen=True, nodes_active=True,
        regret_diag=_full_diag(), nodes_diag=_full_diag(),
    ))
    keys_none = set(_pid_report_dict(pr_none))  # pyright: ignore[reportArgumentType]
    assert keys_none == set(_pid_report_dict(pr_regret_only))  # pyright: ignore[reportArgumentType]
    assert keys_none == set(_pid_report_dict(pr_both))  # pyright: ignore[reportArgumentType]
    assert keys_none, "must emit the full key set, not an empty dict"


def _metrics(**overrides: Any) -> TrainMetrics:
    base: dict[str, Any] = dict.fromkeys(
        (
            "loss", "policy_loss", "soft_policy_loss", "future_policy_loss",
            "wdl_loss", "sf_move_loss", "sf_move_acc", "sf_eval_loss",
            "categorical_loss", "volatility_loss", "sf_volatility_loss",
            "moves_left_loss",
        ),
        0.0,
    )
    base.update(overrides)
    return TrainMetrics(**base)


def test_train_metrics_dict_key_set_is_stable_with_and_without_metrics() -> None:
    """The no-train-phase fallback must emit the SAME keys as a real iteration,
    or Ray's first-row-fixes-the-header CSV logger misaligns every later row."""
    assert set(_train_metrics_dict(None)) == set(_train_metrics_dict(_metrics()))
    assert set(_TRAIN_METRIC_DEFAULTS) == set(_train_metrics_dict(_metrics()))


def test_train_metrics_dict_promotes_grad_norm_clip_rate_and_operating_lr() -> None:
    """rl_loop_audit I9/I19: these were TensorBoard-only (and TB event files
    rotate per Ray session), so no ledger yardstick could cite them."""
    got = _train_metrics_dict(
        _metrics(
            grad_norm_median=4.244, grad_norm_p95=4.941, grad_norm_max=5.538,
            grad_norm_mean=4.2, grad_clip_rate=0.0515, grad_adaptive_clip_rate=0.0429,
            grad_hard_clip_rate=0.0129, grad_norm_samples=233,
            opt_lr_mean=5.306e-4, opt_lr_max=6e-4,
        )
    )

    assert got["grad_norm_median"] == 4.244
    assert got["grad_norm_p95"] == 4.941
    assert got["grad_norm_max"] == 5.538
    assert got["grad_clip_rate"] == 0.0515
    assert got["grad_adaptive_clip_rate"] == 0.0429
    assert got["grad_hard_clip_rate"] == 0.0129
    assert got["grad_norm_samples"] == 233
    assert got["opt_lr_mean"] == 5.306e-4
    assert got["opt_lr_max"] == 6e-4


def test_the_draw_provenance_counters_reach_the_result_row_not_just_tensorboard() -> None:
    """PR #373 added `batches_drawn` / `transient_cuda_retry_batches` to
    `TrainMetrics` and called them "worth having live" -- but
    `_train_metrics_dict` enumerates report columns BY NAME, so neither
    appeared in the Ray result row. Computed every step, read by nobody.

    `_log_metrics` does splat every field to TensorBoard, which is exactly the
    non-sink the grad-norm family was promoted out of: the event files rotate
    per Ray session, so no ledger yardstick can cite them.

    Asserting on the VALUES, not on membership: a column wired to a literal 0.0
    (or to the wrong metric) is the same defect wearing the right key. A retry
    count of 3 has to arrive as 3.
    """
    got = _train_metrics_dict(
        _metrics(batches_drawn=1603.0, transient_cuda_retry_batches=3.0)
    )

    assert got["batches_drawn"] == 1603.0
    assert got["transient_cuda_retry_batches"] == 3.0


def test_a_retry_free_iteration_publishes_zero_rather_than_omitting_the_column() -> None:
    """The healthy value is 0.0 and it must still be PUBLISHED. Ray fixes the
    CSV header from the first row, so a counter emitted only on the iterations
    that happened to retry makes every clean iteration read as missing rather
    than as "no retry" -- and the whole point of the counter is that a clean run
    is distinguishable from an unrecorded one."""
    clean = _train_metrics_dict(_metrics())

    assert clean["transient_cuda_retry_batches"] == 0.0
    assert _TRAIN_METRIC_DEFAULTS["transient_cuda_retry_batches"] == 0.0
    assert _TRAIN_METRIC_DEFAULTS["batches_drawn"] == 0.0


def test_status_csv_lr_column_is_named_for_the_mean_not_the_trough() -> None:
    """`lr` used to be the end-of-iteration sample, i.e. the sqrt_release
    trough (~9x below the LR the trunk trains at) — rl_loop_audit I19."""
    assert "lr" not in _STATUS_COLS
    assert "lr_mean" in _STATUS_COLS
