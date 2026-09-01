"""The I11 zclip watch fires on the BINDING clip, and the per-group optimizer
scalars are range-checked at construction.

Two defects, one theme — a value that is accepted and then means something other
than what it says:

* the I11 grad-norm watch gated on the windowed median ALONE while its own
  comment defined the condition as "median past 4.75 (hard-clip rate ~10%)".
  On the production run that combination emitted 124 times at a measured
  hard-clip rate of 0.0% — `zclip_max_norm` was not the binding constraint on a
  single step, and the message told operators to re-set it anyway.
* `matrix_lr_multiplier` reached the Aurora matrix group raw. That group's
  update is scale-invariant and carries no adaptive denominator, so the
  multiplier is its ENTIRE step-size control and nothing downstream absorbs a
  misplaced decimal.

The zclip semantics these tests encode, measured against the installed library
(`zclip.zclip.ZClip`, clip_option="adaptive_scaling"): `_compute_clip_val`
returns a threshold ONLY when the z-score exceeds `z_thresh` — it is not
computed every step — and `_apply_clipping` then takes
`effective_clip = min(adaptive_threshold, max_grad_norm)`. So "the adaptive
threshold fired" and "the adaptive threshold bound" are different events, and
`test_adaptive_threshold_can_fire_while_the_hard_cap_binds` is the case that
tells them apart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from chess_anti_engine.train import trainer as trainer_mod
from chess_anti_engine.train.trainer import Trainer, TrainMetrics, trainer_kwargs_from_config
from chess_anti_engine.tune.trainable_report import _train_metrics_dict


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32, device=self.head.weight.device),
        }


def _make_trainer(tmp_path: Path, **kwargs: Any) -> Trainer:
    trainer_kwargs: dict[str, Any] = {
        "device": "cpu",
        "lr": 1e-3,
        "optimizer": "muon",
        "warmup_steps": 10,
        "warmup_lr_start": 1e-5,
        "use_amp": False,
        "log_dir": tmp_path,
        "tb_log_interval": 1000,
        "prefetch_batches": False,
    }
    trainer_kwargs.update(kwargs)
    return Trainer(_TinyModel(), **trainer_kwargs)


_LOSS_METRIC_KEYS = (
    "loss", "policy_loss", "soft_policy_loss", "future_policy_loss", "wdl_loss",
    "sf_move_loss", "sf_eval_loss", "categorical_loss", "volatility_loss",
    "sf_volatility_loss", "moves_left_loss",
)


def _metrics(**overrides: Any) -> TrainMetrics:
    base: dict[str, Any] = dict.fromkeys(_LOSS_METRIC_KEYS, 0.0)
    base["sf_move_acc"] = 0.0
    base.update(overrides)
    return TrainMetrics(**base)


# --------------------------------------------------------------------------
# (a) the I11 watch
# --------------------------------------------------------------------------


def test_watch_fires_when_median_and_hard_clip_rate_both_trip(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    trainer = _make_trainer(tmp_path, zclip_max_norm=6.5)
    metrics = _metrics(
        grad_norm_samples=200,
        grad_norm_median=trainer_mod.GRAD_NORM_MEDIAN_WATCH + 0.1,
        grad_clip_rate=0.62,
        grad_hard_clip_rate=0.60,
        grad_adaptive_bound_rate=0.02,
        grad_adaptive_clip_rate=0.05,
    )

    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(metrics)

    assert "watch threshold" in caplog.text
    assert "hard-clip rate 60.0%" in caplog.text


def test_watch_is_silent_on_a_high_median_with_zero_hard_clips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """THE REGRESSION THIS CHANGE EXISTS FOR — the live 124-emission false positive.

    Median past the watch, every step clipped, and the hard cap binding on none
    of them: `zclip_max_norm` is provably not the constraint doing the clipping,
    so a message telling an operator to re-set it is instructing a no-op.
    """
    trainer = _make_trainer(tmp_path, zclip_max_norm=6.5)
    metrics = _metrics(
        grad_norm_samples=233,
        grad_norm_median=5.31,
        grad_clip_rate=1.0,
        grad_hard_clip_rate=0.0,
        grad_adaptive_bound_rate=1.0,
        grad_adaptive_clip_rate=1.0,
    )
    assert metrics.grad_norm_median > trainer_mod.GRAD_NORM_MEDIAN_WATCH

    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(metrics)

    assert caplog.text == ""


def test_watch_gate_sits_exactly_at_the_hard_clip_rate_watch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Both halves of I11 are gates, and the hard-clip half has a stated value."""
    trainer = _make_trainer(tmp_path, zclip_max_norm=6.5)
    watch = trainer_mod.GRAD_HARD_CLIP_RATE_WATCH
    base = {
        "grad_norm_samples": 200,
        "grad_norm_median": trainer_mod.GRAD_NORM_MEDIAN_WATCH + 0.1,
        "grad_adaptive_bound_rate": 0.0,
    }

    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(
            _metrics(grad_hard_clip_rate=watch * 0.99, grad_clip_rate=watch, **base)
        )
    assert caplog.text == ""

    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(
            _metrics(grad_hard_clip_rate=watch, grad_clip_rate=watch, **base)
        )
    assert "watch threshold" in caplog.text


def test_watch_message_names_the_hard_cap_when_the_hard_cap_binds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    trainer = _make_trainer(tmp_path, zclip_max_norm=6.5)
    metrics = _metrics(
        grad_norm_samples=200,
        grad_norm_median=trainer_mod.GRAD_NORM_MEDIAN_WATCH + 0.1,
        grad_clip_rate=0.75,
        grad_hard_clip_rate=0.60,
        grad_adaptive_bound_rate=0.15,
        grad_adaptive_clip_rate=0.20,
    )

    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(metrics)

    assert "HARD cap zclip_max_norm=6.50 is the binding clip" in caplog.text
    assert "re-set zclip_max_norm" in caplog.text
    assert "ADAPTIVE z-score threshold is the binding clip" not in caplog.text


def test_watch_message_names_the_adaptive_threshold_when_it_binds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The knob the message names must be the knob that moves the number.

    Both clips are past the gate here, but the adaptive threshold is the `min()`
    winner four times as often. Telling the operator to re-set `zclip_max_norm`
    would send them to the smaller share.
    """
    trainer = _make_trainer(tmp_path, zclip_max_norm=6.5)
    metrics = _metrics(
        grad_norm_samples=200,
        grad_norm_median=trainer_mod.GRAD_NORM_MEDIAN_WATCH + 0.1,
        grad_clip_rate=0.75,
        grad_hard_clip_rate=0.15,
        grad_adaptive_bound_rate=0.60,
        grad_adaptive_clip_rate=0.80,
    )

    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(metrics)

    assert "ADAPTIVE z-score threshold is the binding clip" in caplog.text
    assert "zclip_z_thresh / zclip_alpha" in caplog.text
    assert "HARD cap zclip_max_norm=6.50 is the binding clip" not in caplog.text


def test_the_adaptive_branch_says_its_knobs_are_restart_gated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑ Naming a knob is not enough — say whether a live edit can reach it.

    `ZClip.__init__` reads `zclip_z_thresh` and `zclip_alpha` once, and nothing
    in `tune/` pushes either at a running trial: a live yaml edit to them is
    overlaid into the config and then silently ignored until restart. Only
    `zclip_max_norm` has a live path (`Trainer.set_grad_clip_max_norm`, pushed
    every iteration), and that setter exists precisely because editing the cap
    live used to be a no-op. A message that sends an operator to a restart-gated
    knob without saying so reproduces this repo's signature defect inside the
    very text written to stop it.
    """
    trainer = _make_trainer(tmp_path, zclip_max_norm=6.5)
    metrics = _metrics(
        grad_norm_samples=200,
        grad_norm_median=trainer_mod.GRAD_NORM_MEDIAN_WATCH + 0.1,
        grad_clip_rate=0.75,
        grad_hard_clip_rate=0.15,
        grad_adaptive_bound_rate=0.60,
        grad_adaptive_clip_rate=0.80,
    )

    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(metrics)

    assert "RESTART-GATED" in caplog.text
    assert "silently ignored until the next restart" in caplog.text
    assert "zclip_max_norm is the only clip knob that takes effect mid-run" in caplog.text
    # The old wording told operators to re-set them as if it were a live edit.
    assert "are the knobs on the binding one" not in caplog.text


def test_an_exact_tie_credits_the_hard_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Tie goes to the hard cap: it is the one an operator can actually re-set."""
    trainer = _make_trainer(tmp_path, zclip_max_norm=6.5)
    metrics = _metrics(
        grad_norm_samples=200,
        grad_norm_median=trainer_mod.GRAD_NORM_MEDIAN_WATCH + 0.1,
        grad_clip_rate=0.60,
        grad_hard_clip_rate=0.30,
        grad_adaptive_bound_rate=0.30,
        grad_adaptive_clip_rate=0.35,
    )

    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(metrics)

    assert "HARD cap zclip_max_norm=6.50 is the binding clip" in caplog.text
    assert "ADAPTIVE z-score threshold is the binding clip" not in caplog.text


def test_watch_still_declines_when_the_hard_cap_is_disabled(tmp_path: Path,
                                                            caplog: pytest.LogCaptureFixture) -> None:
    """`zclip_max_norm: null` leaves adaptive clipping on and no cap to re-set."""
    trainer = _make_trainer(tmp_path, zclip_max_norm=None)
    with caplog.at_level("WARNING", logger="chess_anti_engine.train.trainer"):
        trainer._warn_if_grad_norm_median_past_watch(
            _metrics(
                grad_norm_samples=200,
                grad_norm_median=trainer_mod.GRAD_NORM_MEDIAN_WATCH + 5.0,
                grad_clip_rate=1.0,
                grad_hard_clip_rate=1.0,
                grad_adaptive_bound_rate=1.0,
            )
        )
    assert caplog.text == ""


# --------------------------------------------------------------------------
# (a3) adaptive-vs-hard is separately observable
# --------------------------------------------------------------------------


def _drive_zclip(trainer: Trainer, *, mean: float, var: float, norm: float) -> dict[str, float]:
    """One `_zclip_step` against a hand-set EMA and an exact gradient norm."""
    trainer.zclip.initialized = True
    trainer.zclip.mean = mean
    trainer.zclip.var = var
    named = dict(trainer.model.named_parameters())
    for param in named.values():
        param.grad = None
    named["head.bias"].grad = torch.tensor([norm, 0.0, 0.0])
    _total, stats = trainer._zclip_step(collect_stats=True)
    assert stats is not None
    return stats


def _clip_trainer(tmp_path: Path, **kwargs: Any) -> Trainer:
    # adamw keeps the whole model inside the clip scope, so the norm this test
    # sets is the norm zclip sees.
    return _make_trainer(
        tmp_path, optimizer="adamw", zclip_max_norm=6.5, zclip_z_thresh=2.0,
        zclip_clip_factor=1.0, **kwargs,
    )


def test_adaptive_threshold_can_fire_while_the_hard_cap_binds(tmp_path: Path) -> None:
    """`adaptive_clip` is FIRING; `adaptive_bound`/`hard_clip` are BINDING.

    EMA mean 10.0, std 2.0, norm 20.0 => z = 5.0 > z_thresh, so the adaptive
    threshold fires at 10 + (2*2)/2.5 = 11.6. That is above the 6.5 cap, so
    `min()` picks the cap and the gradient is scaled to 6.5: the adaptive clip
    fired and did NOT bind. Reading `grad_adaptive_clip_rate` as "the adaptive
    clip was in charge" is wrong exactly here.
    """
    trainer = _clip_trainer(tmp_path)
    stats = _drive_zclip(trainer, mean=10.0, var=4.0, norm=20.0)

    assert stats["adaptive_clip"] == 1.0
    assert stats["hard_clip"] == 1.0
    assert stats["adaptive_bound"] == 0.0
    assert stats["clipped"] == 1.0
    assert stats["effective_clip"] == pytest.approx(6.5)


def test_adaptive_threshold_binds_when_it_sits_below_the_cap(tmp_path: Path) -> None:
    # mean 4.0, std 0.5, norm 10.0 => z = 12 => threshold 4 + (2*0.5)/6 = 4.167,
    # comfortably under the 6.5 cap, so the adaptive clip is the min() winner.
    trainer = _clip_trainer(tmp_path)
    stats = _drive_zclip(trainer, mean=4.0, var=0.25, norm=10.0)

    assert stats["adaptive_clip"] == 1.0
    assert stats["adaptive_bound"] == 1.0
    assert stats["hard_clip"] == 0.0
    assert stats["effective_clip"] == pytest.approx(4.0 + 1.0 / 6.0, rel=1e-3)


def test_hard_cap_binds_with_no_adaptive_firing(tmp_path: Path) -> None:
    # mean 8.0, std 2.0, norm 9.0 => z = 0.5, no spike, so zclip computes no
    # adaptive threshold at all and the cap is the only constraint.
    trainer = _clip_trainer(tmp_path)
    stats = _drive_zclip(trainer, mean=8.0, var=4.0, norm=9.0)

    assert stats["adaptive_clip"] == 0.0
    assert stats["adaptive_bound"] == 0.0
    assert stats["hard_clip"] == 1.0
    assert stats["effective_clip"] == pytest.approx(6.5)


def test_a_quiet_step_binds_neither_clip(tmp_path: Path) -> None:
    trainer = _clip_trainer(tmp_path)
    stats = _drive_zclip(trainer, mean=4.0, var=0.25, norm=4.25)

    assert stats["clipped"] == 0.0
    assert stats["adaptive_clip"] == 0.0
    assert stats["adaptive_bound"] == 0.0
    assert stats["hard_clip"] == 0.0


def test_bound_rates_partition_the_clip_rate() -> None:
    """`hard + adaptive_bound == clipped`, which is what makes the two readable."""
    kwargs = trainer_mod._grad_clip_metric_kwargs(
        [1.0, 5.0, 3.0, 4.0, 2.0],
        {"clipped": 4, "adaptive_clip": 3, "adaptive_bound": 1, "hard_clip": 3},
    )

    assert kwargs["grad_adaptive_bound_rate"] == pytest.approx(0.2)
    assert kwargs["grad_hard_clip_rate"] == pytest.approx(0.6)
    assert kwargs["grad_clip_rate"] == pytest.approx(0.8)
    assert (
        float(kwargs["grad_hard_clip_rate"]) + float(kwargs["grad_adaptive_bound_rate"])
        == pytest.approx(float(kwargs["grad_clip_rate"]))
    )
    # ... and the FIRING rate is genuinely a different number.
    assert kwargs["grad_adaptive_clip_rate"] == pytest.approx(0.6)


def test_train_steps_accumulates_adaptive_bound_over_the_whole_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-step flag has to survive the iteration's own tally.

    `train_steps` seeds `clip_counts` from a LITERAL key list and then does
    `for flag in clip_counts`, so a flag the seed omits is emitted by
    `_zclip_step`, ignored by the tally, and published as a flat 0.0 — present,
    plausible, and constant. That is this repo's signature defect exactly, and
    nothing else in this file would catch it: the `_zclip_step` cases below the
    fold pass, the report column exists, and the TensorBoard scalar is written.
    Three of four steps here bind adaptively, one binds at the hard cap.
    """
    trainer = _make_trainer(tmp_path, warmup_steps=0)
    steps = {"n": 0}

    def fake_run_optimizer_step(
        *,
        step_sums: trainer_mod._DeviceLossSums,
        step_acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        step_opt_stats: dict[str, float],
        buf: Any,
        batch_size: int,
        update_lr: bool = True,
        collect_optimizer_stats: bool = True,
        batch_iter: Any = None,
        timer: Any = None,
    ) -> tuple[int, float]:
        del timer, step_acc_sums, buf, batch_size, update_lr, collect_optimizer_stats, batch_iter
        steps["n"] += 1
        hard = steps["n"] == 4
        step_opt_stats["grad_norm"] = 8.0
        step_opt_stats["clipped"] = 1.0
        step_opt_stats["hard_clip"] = 1.0 if hard else 0.0
        step_opt_stats["adaptive_bound"] = 0.0 if hard else 1.0
        step_opt_stats["adaptive_clip"] = 1.0
        step_opt_stats["lr"] = 1e-3
        step_sums.add_losses(dict.fromkeys(_LOSS_METRIC_KEYS, torch.zeros(())))
        return 1, 0.0

    monkeypatch.setattr(trainer, "_run_optimizer_step", fake_run_optimizer_step)
    metrics = trainer.train_steps(None, batch_size=1, steps=4)  # pyright: ignore[reportArgumentType]

    assert metrics.grad_norm_samples == 4
    assert metrics.grad_adaptive_bound_rate == pytest.approx(0.75)
    assert metrics.grad_hard_clip_rate == pytest.approx(0.25)
    assert metrics.grad_clip_rate == pytest.approx(1.0)
    # The two bound rates still partition the clip rate after aggregation.
    assert metrics.grad_adaptive_bound_rate + metrics.grad_hard_clip_rate == pytest.approx(
        metrics.grad_clip_rate
    )


def test_adaptive_bound_rate_reaches_the_ray_result_row() -> None:
    """A TrainMetrics field absent from `_train_metrics_dict` is read by nobody."""
    row = _train_metrics_dict(
        _metrics(grad_adaptive_bound_rate=0.4271, grad_hard_clip_rate=0.0129),
    )

    assert row["grad_adaptive_bound_rate"] == pytest.approx(0.4271)
    assert row["grad_hard_clip_rate"] == pytest.approx(0.0129)


def test_adaptive_bound_rate_is_a_tensorboard_scalar(tmp_path: Path) -> None:
    """`_log_metrics` walks the dataclass, so the field must be a real float."""
    trainer = _make_trainer(tmp_path)
    logged: dict[str, float] = {}

    def _capture(tag: str, value: Any, _step: int = 0) -> None:
        logged[tag] = float(value)

    trainer.writer.add_scalar = _capture

    trainer._log_metrics(_metrics(grad_adaptive_bound_rate=0.375), "train_avg")

    assert logged["train_avg/grad_adaptive_bound_rate"] == pytest.approx(0.375)


# --------------------------------------------------------------------------
# (b) construction-site range validation
# --------------------------------------------------------------------------


PRODUCTION_MATRIX_LR_MULTIPLIER = 20.0
PRODUCTION_WEIGHT_DECAY = 1e-4


@pytest.mark.parametrize("multiplier", [PRODUCTION_MATRIX_LR_MULTIPLIER, 20, 12.0, 2.0, 0.2, 1e-3])
def test_validator_accepts_production_and_deliberate_multipliers(
    tmp_path: Path, multiplier: float,
) -> None:
    """Accepted AND still applied — `initial_lr` is the un-warmed base LR.

    Asserting acceptance alone would pass against a validator that swallowed
    the value, which is the defect class this repo keeps re-introducing.
    """
    trainer = _make_trainer(
        tmp_path, optimizer="aurora", matrix_lr_multiplier=multiplier, lr=1e-3,
    )
    groups = trainer.opt.param_groups
    assert groups[0]["initial_lr"] == pytest.approx(1e-3 * float(multiplier))
    assert groups[2]["initial_lr"] == pytest.approx(1e-3)


@pytest.mark.parametrize("weight_decay", [PRODUCTION_WEIGHT_DECAY, 0, 0.0, 3e-5, 1.0])
def test_validator_accepts_every_weight_decay_the_configs_use(
    tmp_path: Path, weight_decay: float,
) -> None:
    """Every value `configs/` sets (0 and 1e-4), plus the top of the band."""
    aux = float(weight_decay) / 2.0
    trainer = _make_trainer(
        tmp_path, optimizer="aurora",
        matrix_weight_decay=weight_decay, aux_weight_decay=aux,
    )
    groups = trainer.opt.param_groups
    # Group 0 is the Aurora matrix group, group 2 the decaying aux bucket.
    assert groups[0]["weight_decay"] == pytest.approx(float(weight_decay))
    assert groups[2]["weight_decay"] == pytest.approx(aux)


@pytest.mark.parametrize("bad", [200.0, 200, 100.0001, 0.0, 0, -1.0, -20.0])
def test_validator_rejects_an_out_of_range_matrix_lr_multiplier(
    tmp_path: Path, bad: float,
) -> None:
    with pytest.raises(ValueError, match="matrix_lr_multiplier"):
        _make_trainer(tmp_path, optimizer="aurora", matrix_lr_multiplier=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_validator_rejects_a_non_finite_matrix_lr_multiplier(
    tmp_path: Path, bad: float,
) -> None:
    """⚑ A CLAMP IS NOT A VALIDATOR: `min`/`max` propagate nan.

    `min(float('nan'), 100.0)` is `nan` on CPython, so a `min`/`max` "guard"
    hands nan straight through to `lr = lr * nan` and every weight becomes nan
    on the first step. The message must say *not finite*, not report a range —
    a rejection that blames the bounds sends the operator hunting for a decimal
    that is not there.
    """
    with pytest.raises(ValueError, match="is not finite"):
        _make_trainer(tmp_path, optimizer="aurora", matrix_lr_multiplier=bad)


def test_multiplier_rejection_quotes_the_recorded_lr_fact(tmp_path: Path) -> None:
    """The operator needs a replacement value, not just a refusal.

    And the two LR pairs must be labelled apart: production is lr 3e-5 x 20 =
    6e-4, while 3e-4 x 20 = 6e-3 is the 2026-07-11 FAILURE. Quoting the failing
    pair as the production one overstates the live matrix-group LR by 10x.
    """
    with pytest.raises(ValueError, match="matrix_lr_multiplier") as excinfo:
        _make_trainer(tmp_path, optimizer="aurora", matrix_lr_multiplier=200.0)

    message = str(excinfo.value)
    assert "6e-4" in message          # what production actually runs
    assert "3e-5" in message
    assert "6e-3" in message          # ... labelled as the historical failure
    assert "historical FAILING pair" in message
    assert "0.003" in message
    assert "Production multiplier is 20." in message


def test_a_near_boundary_rejection_does_not_print_as_the_boundary(
    tmp_path: Path,
) -> None:
    """⚑ A rejection that reads as self-contradictory teaches operators to distrust it.

    `f"{100.0001:g}"` is `"100"`, so a `:g`-formatted message says "100 is
    outside the accepted range (0 < matrix_lr_multiplier <= 100)". The value
    must print at full precision; the bounds stay exact literals.
    """
    with pytest.raises(ValueError, match="matrix_lr_multiplier") as excinfo:
        _make_trainer(tmp_path, optimizer="aurora", matrix_lr_multiplier=100.0001)

    message = str(excinfo.value)
    assert "100.0001" in message
    assert "matrix_lr_multiplier=100 " not in message
    assert "<= 100)" in message


@pytest.mark.parametrize("key", ["matrix_weight_decay", "aux_weight_decay"])
@pytest.mark.parametrize("bad", [-1.0, -1e-9, 2.0, float("nan"), float("inf")])
def test_validator_rejects_out_of_range_weight_decays(
    tmp_path: Path, key: str, bad: float,
) -> None:
    with pytest.raises(ValueError, match=key):
        _make_trainer(tmp_path, optimizer="aurora", **{key: bad})


def test_validation_is_not_gated_on_the_aurora_branch(tmp_path: Path) -> None:
    """A guard that only fires on the production optimizer is untested elsewhere."""
    with pytest.raises(ValueError, match="matrix_lr_multiplier"):
        _make_trainer(tmp_path, optimizer="adamw", matrix_lr_multiplier=200.0)


def test_a_yaml_decimal_typo_is_refused_on_the_config_route(tmp_path: Path) -> None:
    """The realistic trigger: `matrix_lr_multiplier: 200` in the yaml, at restart.

    `trainer_kwargs_from_config` is the route every real construction takes, so
    the guard has to hold there and not only for a hand-built kwarg.
    """
    config: dict[str, Any] = {
        "device": "cpu", "lr": 0.0003, "optimizer": "aurora",
        "matrix_optimizer_scope": "mlp_out", "matrix_lr_multiplier": 200,
    }
    kwargs = trainer_kwargs_from_config(config, log_dir=tmp_path)
    assert kwargs["matrix_lr_multiplier"] == 200.0  # accepted raw, as today

    with pytest.raises(ValueError, match="matrix_lr_multiplier=200"):
        Trainer(_TinyModel(), **kwargs)


def test_validator_helper_reports_a_non_numeric_value_as_such() -> None:
    with pytest.raises(ValueError, match="is not a number"):
        trainer_mod._validated_optimizer_scalar(
            "matrix_lr_multiplier", "twenty",
            minimum=0.0, maximum=100.0, minimum_inclusive=False,
            consequence="Production is 20.",
        )
