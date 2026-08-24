from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay import ReplayBuffer
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.train.trainer import TrainMetrics, Trainer
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


def test_the_value_blend_fallback_coverage_reaches_the_result_row() -> None:
    """⚑ REVIEW F9. The two columns were TensorBoard-only, and TB is not the
    path `scripts/loop_health.py` reads — so the state their field comment
    names ("the 2026-05 realized sf_wdl_frac 0.45 episode") could still not be
    alerted on.

    ⚑ Polarity: healthy is 1.0 here and 0.0 is a FULL leak, the opposite of
    every other detector in this block. The default must therefore NOT read as
    healthy, which is what the second assertion pins.
    """
    got = _train_metrics_dict(
        _metrics(sf_wdl_effective_frac=0.25, search_wdl_effective_frac=0.5),
    )
    assert got["sf_wdl_effective_frac"] == 0.25
    assert got["search_wdl_effective_frac"] == 0.5
    absent = _train_metrics_dict(None)
    assert absent["sf_wdl_effective_frac"] == 0.0
    assert absent["search_wdl_effective_frac"] == 0.0


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


_PLANES = 146


def _draw_provenance_sample() -> ReplaySample:
    x = np.zeros((_PLANES, 8, 8), dtype=np.float32)
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    policy[0] = 1.0
    return ReplaySample(
        x=x, policy_target=policy, wdl_target=1, priority=1.0,
        has_policy=True, is_network_turn=True,
    )


def _train_steps_on_a_tiny_cpu_buffer(
    tmp_path: Path, *, accum_steps: int, steps: int, batch_size: int,
) -> TrainMetrics:
    cfg = TransformerConfig(
        in_planes=_PLANES, embed_dim=32, num_layers=1, num_heads=2,
        use_smolgen=False, use_nla=False,
    )
    trainer = Trainer(
        ChessNet(cfg), device="cpu", lr=1e-4, log_dir=tmp_path / f"tb{accum_steps}",
        use_amp=False, feature_dropout_p=0.0, swa_start=-1, accum_steps=accum_steps,
    )
    buf = ReplayBuffer(64, rng=np.random.default_rng(0))
    for _ in range(32):
        buf.add(_draw_provenance_sample())
    return trainer.train_steps(buf, batch_size=batch_size, steps=steps)


@pytest.mark.parametrize("accum_steps", [1, 3])
def test_batches_drawn_counts_the_microbatches_train_steps_actually_pulled(
    tmp_path: Path, accum_steps: int,
) -> None:
    """The SOURCE, not the column. The two tests below pin that
    `_train_metrics_dict` publishes these; nothing pinned that
    `Trainer.train_steps` computes them, so replacing both with a literal 0.0
    at the `_build_metrics` call left every test in the repo green while the
    published column reported a constant forever -- the same defect one level
    up from the one this pair was added to fix.

    ⚑ BOTH `accum_steps` VALUES ARE LOAD-BEARING. At `accum_steps=1`,
    `batches_drawn` and `train_steps_done` are the same number (3), so a
    substitution of one for the other is invisible; at 3 they are 9 vs 3. The
    counter must be MICROBATCHES pulled from the buffer -- that is the quantity
    the replay RNG advances on, and therefore the one that says whether two
    paired arms drew the same rows.
    """
    metrics = _train_steps_on_a_tiny_cpu_buffer(
        tmp_path, accum_steps=accum_steps, steps=3, batch_size=4,
    )

    assert metrics.batches_drawn == float(3 * accum_steps)
    assert metrics.train_steps_done == 3
    if accum_steps > 1:
        assert metrics.batches_drawn != float(metrics.train_steps_done), (
            "batches_drawn collapsed onto the optimizer-step count -- it must "
            "count buffer draws, which is what the replay RNG advances on"
        )


def test_a_clean_cpu_run_reports_no_retries_and_publishes_the_measured_draws(
    tmp_path: Path,
) -> None:
    """End to end on the production path: `train_steps` -> `TrainMetrics` ->
    `_train_metrics_dict`. A retry only happens on a transient CUDA error, so a
    CPU run must report exactly 0.0 -- and 0.0 has to be a MEASUREMENT here,
    which is why the draw count beside it is asserted against the arithmetic
    rather than against itself."""
    metrics = _train_steps_on_a_tiny_cpu_buffer(
        tmp_path, accum_steps=2, steps=3, batch_size=4,
    )
    row = _train_metrics_dict(metrics)

    assert metrics.transient_cuda_retry_batches == 0.0
    assert row["transient_cuda_retry_batches"] == 0.0
    assert row["batches_drawn"] == metrics.batches_drawn == 6.0


def test_status_csv_lr_column_is_named_for_the_mean_not_the_trough() -> None:
    """`lr` used to be the end-of-iteration sample, i.e. the sqrt_release
    trough (~9x below the LR the trunk trains at) — rl_loop_audit I19."""
    assert "lr" not in _STATUS_COLS
    assert "lr_mean" in _STATUS_COLS


def test_the_realized_gate_keys_are_reported_from_the_trainer() -> None:
    """MUTATION: point either column at the config instead of `trainer.*`, or
    delete the pair from `_build_report_dict`.

    ⚑⚑ THE COLUMN IS THE ONLY DURABLE RECORD OF A SUBSTITUTION. The two
    fabricated-tail gate keys are sanitized at `Trainer` construction --
    non-finite falls back to OFF, finite out-of-range clamps into [0, 1] -- and
    `params.json` is NOT fixed up on the way, because it persists the Ray
    `config`, i.e. what the operator typed. So a run configured
    `sf_own_regret_listed_mass_min: 10` trains at `1.0` with its saved config
    still reading `10`, and `sf_own_regret_gated_frac` cannot separate the two
    cases -- it reads `0.0` for "deliberately off" and for "disabled by a
    non-finite value" alike. Without this row the one-time construction warning
    is the only evidence, and a warning that scrolled past three days ago is not
    evidence.

    ⚑ ASSERTED WITH A REALIZED VALUE THAT DIFFERS FROM ANY PLAUSIBLE TYPED ONE
    (0.75 / 0.25, neither a default nor an endpoint), so a column silently wired
    to a constant, to the identity defaults, or to a clamp of the typed value
    cannot pass by coincidence.
    """
    from types import SimpleNamespace

    from chess_anti_engine.tune.trainable_report import _build_report_dict
    from chess_anti_engine.tune.trial_config import (
        DriftMetrics,
        PidResult,
        RestoreResult,
        SelfplayResult,
        TrainingResult,
        TrialConfig,
    )

    trainer = SimpleNamespace(
        opt=SimpleNamespace(param_groups=[{"lr": 3e-4}]),
        w_wdl=1.0, w_soft=1.0, w_sf_move=1.0, w_categorical=1.0,
        sf_wdl_frac=0.5, sf_wdl_temperature=1.0, sf_wdl_draw_scale=1.0,
        sf_wdl_conf_power=1.0,
        mirror_prob=0.5,
      # The REALIZED pair, as `Trainer.__init__` would have stored it.
        sf_own_regret_listed_mass_min=0.75,
        sf_own_regret_unlisted_scale=0.25,
        _feature_group_dropout=[(f"g{i}", (), 0.0) for i in range(8)],
    )
    row = _build_report_dict(
        tc=TrialConfig(), trainer=trainer, pr=PidResult(), sp=SelfplayResult(),
        tr=TrainingResult(), drift=DriftMetrics(), eval_dict={}, puzzle_dict={},
        probe_dict=None,
        wdl_regret_used=0.07, sf_nodes_used=5000,
        pause_metrics={
            "paused_seconds": 0.0, "paused_fraction": 0.0, "paused_percent": 0.0,
        },
        restore=RestoreResult(), best_loss=1.0, iter_t0=0.0, iteration_idx=1,
        buf_size=10, holdout_buf_size=1, holdout_frozen=False,
        holdout_generation=0,
    )

    assert row["sf_own_regret_listed_mass_min"] == pytest.approx(0.75), (
        "the realized listed_mass_min did not reach the result row -- an operator "
        "reading params.json has no way to learn the value was substituted"
    )
    assert row["sf_own_regret_unlisted_scale"] == pytest.approx(0.25)
