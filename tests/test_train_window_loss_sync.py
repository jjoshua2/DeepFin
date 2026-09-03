"""The training window's loss scalars leave the device ONCE, and unchanged.

`Trainer._extract_loss_scalars` ends in `.tolist()`, which on CUDA is a full
host sync. It was called once per MICROBATCH, immediately after
`loss.backward()` -- i.e. before zclip, the matrix grad norm and the optimizer
step had even been enqueued, so the training thread stopped at the earliest
possible point in every step. `_DeviceLossSums` keeps the same detached
scalars on the device and `train_steps` drains the whole window in one
`torch.stack(...).tolist()`.

Three things have to be true for that to be a performance change and not a
measurement change, and this file pins all three:

1. **The numbers are identical, not close.** Every assertion here is `==` on
   floats, never `approx`. The accumulation is float64 in the SAME ORDER the
   host loop used -- microbatches into a per-step accumulator, per-step
   accumulators into the window -- so IEEE-754 makes that reproducible rather
   than lucky.
2. **The per-step `train/loss` TensorBoard series survives.** Same values,
   same `global_step`s; only the WRITE moves to the window's single transfer.
   A scalar carries its step explicitly, so a late write is indistinguishable
   in the event file from an on-time one.
3. **⚑⚑ THE KEPT SYNC STAYS PER-STEP.** The non-finite gradient guard reads
   the two grad norms as host floats EVERY step and branches on them. That one
   is a safety check, not a metric: batching it to the end of the window would
   apply ~88 non-finite updates before anything noticed. There is a test below
   whose only job is to fail if a later refactor window-batches it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from chess_anti_engine.train import trainer as trainer_mod
from chess_anti_engine.train.trainer import (
    Trainer,
    _DeviceLossSums,
    _materialize_device_scalars,
)

#: Every `compute_loss` key that maps onto a REQUIRED `TrainMetrics` field.
#: `_build_metrics` splats the loss sums into the dataclass, so a fake loss
#: that omits one of these raises `TypeError` rather than failing an assert.
_ZERO_LOSS_KEYS = tuple(trainer_mod._LOSS_KEY_TO_METRIC_FIELD)


class _TinyModel(nn.Module):
    """Smallest model the Aurora split and the training loop both accept."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32, device=self.head.weight.device),
        }


class _RecordingWriter:
    """Captures `(tag, value, step)` instead of writing an event file."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: Any, step: Any = None) -> None:
        self.scalars.append((str(tag), float(value), int(step)))

    def close(self) -> None: ...


def _make_trainer(tmp_path: Path, **kwargs: Any) -> Trainer:
    trainer_kwargs: dict[str, Any] = {
        "device": "cpu",
        "lr": 1e-3,
        "optimizer": "aurora",
        "use_amp": False,
        "log_dir": tmp_path,
        "tb_log_interval": 1,
        "prefetch_batches": False,
    }
    trainer_kwargs.update(kwargs)
    return Trainer(_TinyModel(), **trainer_kwargs)


def _install_stepped_losses(
    trainer: Trainer,
    monkeypatch: pytest.MonkeyPatch,
    scales: list[float],
) -> list[torch.Tensor]:
    """Drive `compute_loss` through *scales*, and BANK what it returned.

    The reference the tests compare against is built from the totals the loss
    actually produced, not from a formula -- the parameters move under the
    optimizer, so a re-derived expectation would be measuring the optimizer.
    """
    param = next(trainer.model.parameters())
    produced: list[torch.Tensor] = []
    calls = {"n": 0}

    def fake_compute_loss(out: Any, batch: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del out, batch, kwargs
        scale = scales[calls["n"] % len(scales)]
        calls["n"] += 1
        total = (param * param).sum() * float(scale)
        produced.append(total.detach().clone())
        return {"total": total, **dict.fromkeys(_ZERO_LOSS_KEYS, torch.zeros(()))}

    monkeypatch.setattr(trainer_mod, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})
    monkeypatch.setattr(
        trainer,
        "_iter_prefetched_batches",
        lambda *_args, **_kwargs: iter([{"x": torch.zeros((1, 4, 8, 8))}] * 512),
    )
    return produced


def _neutralize_cuda_retry_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry path calls into `torch.cuda` and sleeps; neither works here."""
    monkeypatch.setattr(trainer_mod.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(trainer_mod.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(trainer_mod.time, "sleep", lambda _seconds: None)


# --- 1. the accumulator itself ---------------------------------------------


def _host_reference(
    window: list[list[dict[str, torch.Tensor]]], *, accum_steps: int,
) -> dict[str, float]:
    """The PRE-CHANGE path, rebuilt here so the comparison is against code.

    This is `_extract_loss_scalars` plus `_run_optimizer_step`'s and
    `train_steps`' two-level `dict.get(k, 0.0) + v` folds, transcribed. It is
    deliberately a transcription and not a call into the trainer: a test that
    compared the new path against itself would pass under every mutant.
    """
    sums: dict[str, float] = {}
    for step in window:
        step_sums: dict[str, float] = {}
        for losses in step:
            divided = losses["total"] / accum_steps
            keys = list(losses)
            stacked = torch.stack([
                (divided if k == "total" else losses[k]).detach() for k in keys
            ])
            values = stacked.tolist()
            for key, value in zip(keys, values, strict=True):
                name = "loss" if key == "total" else key
                scalar = float(value) * float(accum_steps) if key == "total" else float(value)
                step_sums[name] = step_sums.get(name, 0.0) + scalar
        for name, value_f in step_sums.items():
            sums[name] = sums.get(name, 0.0) + value_f
    return sums


def _device_sums(
    window: list[list[dict[str, torch.Tensor]]], *, accum_steps: int,
) -> dict[str, float]:
    total_sums = _DeviceLossSums()
    for step in window:
        step_sums = _DeviceLossSums()
        for losses in step:
            step_sums.add_losses(
                losses,
                total_override=losses["total"] / accum_steps,
                total_scale=float(accum_steps),
            )
        total_sums.merge(step_sums)
    items = total_sums.items()
    flat = _materialize_device_scalars([t for _, t in items])
    return {key: flat[i] for i, (key, _) in enumerate(items)}


@pytest.mark.parametrize("accum_steps", [1, 2, 4])
def test_the_device_accumulator_is_bit_identical_to_the_host_fold(accum_steps: int) -> None:
    """`==`, not `approx`. Values chosen to be inexact in binary so a changed
    summation order or a float32 accumulator would show up in the last bits."""
    gen = torch.Generator().manual_seed(20260901)
    keys = ("total", "policy_ce", "wdl_ce", "moves_left")
    window = [
        [
            {k: (torch.rand((), generator=gen) * 13.7 + 0.1) for k in keys}
            for _ in range(accum_steps)
        ]
        for _ in range(11)
    ]

    assert _device_sums(window, accum_steps=accum_steps) == _host_reference(
        window, accum_steps=accum_steps,
    )


def test_a_step_that_is_never_merged_contributes_nothing() -> None:
    """The retry semantics, at the unit level: `train_steps` builds a fresh
    per-step accumulator per ATTEMPT and merges only on success, so a
    discarded attempt must be invisible in the window."""
    window_sums = _DeviceLossSums()
    kept = _DeviceLossSums()
    kept.add_losses({"total": torch.tensor(3.0)})
    discarded = _DeviceLossSums()
    discarded.add_losses({"total": torch.tensor(100.0)})

    window_sums.merge(kept)

    assert _materialize_device_scalars([t for _, t in window_sums.items()]) == [3.0]


# --- 2. what `train_steps` publishes ---------------------------------------


def test_the_window_mean_matches_the_old_per_step_host_path_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    assert trainer.accum_steps == 1, "the reference below folds one microbatch per step"
    produced = _install_stepped_losses(trainer, monkeypatch, [1.0, 0.3, 7.0, 0.1, 2.5])

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=5)

    running = 0.0
    for total in produced:
        running = running + float(total)
    assert len(produced) == 5
    assert metrics.loss == running / float(len(produced))


def test_the_window_mean_is_a_mean_and_not_a_sum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stated separately because "report sums instead of means" is a one-word
    mutant that the equality above would still catch only by arithmetic
    accident at ``steps == 1``."""
    trainer = _make_trainer(tmp_path)
    produced = _install_stepped_losses(trainer, monkeypatch, [2.0])

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=6)

    banked = [float(t) for t in produced]
    assert len(banked) == 6
    assert metrics.loss == sum(banked) / 6.0
    assert metrics.loss != sum(banked)


def test_the_per_step_train_loss_series_keeps_its_values_and_its_step_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path, tb_log_interval=1)
    writer = _RecordingWriter()
    monkeypatch.setattr(trainer, "writer", writer)
    produced = _install_stepped_losses(trainer, monkeypatch, [1.0, 4.0, 0.25, 9.0])
    first_step = int(trainer.step)

    trainer.train_steps(cast(Any, None), batch_size=1, steps=4)

    series = [(value, step) for tag, value, step in writer.scalars if tag == "train/loss"]
    assert [step for _, step in series] == [first_step + i for i in range(4)]
    assert [value for value, _ in series] == [float(t) for t in produced]


def test_a_retried_step_is_not_double_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient CUDA failure re-runs the step. The failed attempt already
    accumulated its microbatch into the per-step accumulator, so a design that
    added straight into the window sum would count it twice."""
    trainer = _make_trainer(tmp_path)
    _neutralize_cuda_retry_calls(monkeypatch)
    produced = _install_stepped_losses(trainer, monkeypatch, [5.0, 1.0, 2.0])
    real_matrix_norm = trainer._matrix_grad_norm
    calls = {"n": 0}

    def flaky_matrix_norm() -> float:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA transient test failure")
        return real_matrix_norm()

    monkeypatch.setattr(trainer, "_matrix_grad_norm", flaky_matrix_norm)

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=2)

    assert len(produced) == 3, "one discarded attempt plus two committed steps"
    kept = [float(produced[1]), float(produced[2])]
    assert metrics.loss == (kept[0] + kept[1]) / 2.0
    assert metrics.transient_cuda_retry_batches == 1.0


# --- 3. the sync that was removed, and the one that was kept ---------------


def test_the_window_pays_exactly_one_host_transfer_for_its_loss_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the change, asserted as a COUNT rather than as a
    timing. `_extract_loss_scalars` must not be reached from the training path
    at all -- it is the eval path's function now."""
    trainer = _make_trainer(tmp_path)
    _install_stepped_losses(trainer, monkeypatch, [1.0])
    transfers: list[int] = []
    extractions: list[int] = []
    real_materialize = trainer_mod._materialize_device_scalars
    real_extract = Trainer._extract_loss_scalars

    def spy_materialize(tensors: Any) -> list[float]:
        transfers.append(len(tensors))
        return real_materialize(tensors)

    def spy_extract(*args: Any, **kwargs: Any) -> dict[str, float]:
        extractions.append(1)
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "_materialize_device_scalars", spy_materialize)
    monkeypatch.setattr(Trainer, "_extract_loss_scalars", staticmethod(spy_extract))

    trainer.train_steps(cast(Any, None), batch_size=1, steps=8)

    assert len(transfers) == 1, f"one transfer per window, got {len(transfers)}"
    assert extractions == [], "the training path must not materialize per microbatch"


def test_the_nonfinite_gradient_guard_is_still_read_every_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ THE KEPT-SYNC PIN.

    The guard is a per-step SAFETY CHECK, not a metric, so it keeps its
    per-step host read. This test makes step 2 of 3 report a non-finite matrix
    norm and asserts the optimizer skipped exactly that step. If the check were
    window-batched -- read once at the end alongside the loss scalars -- all
    three steps would take an update and `grad_nonfinite_skip_rate` would be
    whatever the last reading happened to be.
    """
    trainer = _make_trainer(tmp_path)
    _install_stepped_losses(trainer, monkeypatch, [1.0])
    real_matrix_norm = trainer._matrix_grad_norm
    readings: list[float] = []

    def spy_matrix_norm() -> float:
        value = float("nan") if len(readings) == 1 else real_matrix_norm()
        readings.append(value)
        return value

    monkeypatch.setattr(trainer, "_matrix_grad_norm", spy_matrix_norm)
    steps_taken: list[int] = []
    real_opt_step = trainer.opt.step

    def spy_opt_step(*args: Any, **kwargs: Any) -> Any:
        steps_taken.append(int(trainer.step))
        return real_opt_step(*args, **kwargs)

    monkeypatch.setattr(trainer.opt, "step", spy_opt_step)
    first_step = int(trainer.step)

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=3)

    assert len(readings) == 3, "the guard reads its norms once per optimizer step"
    assert steps_taken == [first_step, first_step + 2]
    assert metrics.grad_nonfinite_skip_rate == pytest.approx(1.0 / 3.0)


def test_the_guard_reads_host_floats_not_device_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the same pin: `math.isfinite` is what makes the check
    a check, and it needs a Python float. A refactor that handed it a 0-dim
    tensor would still "work" (`math.isfinite` accepts anything with
    `__float__`) while silently reintroducing the sync one line later."""
    trainer = _make_trainer(tmp_path)
    _install_stepped_losses(trainer, monkeypatch, [1.0])
    seen: list[tuple[type, type]] = []
    real_matrix_norm = trainer._matrix_grad_norm
    real_zclip_step = trainer._zclip_step

    def spy_matrix_norm() -> float:
        value = real_matrix_norm()
        seen.append((type(value), type(value)))
        return value

    def spy_zclip_step(*, collect_stats: bool) -> tuple[float, dict[str, float] | None]:
        norm, stats = real_zclip_step(collect_stats=collect_stats)
        seen[-1] = (seen[-1][0], type(norm))
        return norm, stats

    monkeypatch.setattr(trainer, "_matrix_grad_norm", spy_matrix_norm)
    monkeypatch.setattr(trainer, "_zclip_step", spy_zclip_step)

    trainer.train_steps(cast(Any, None), batch_size=1, steps=2)

    assert seen == [(float, float), (float, float)]


class _DispatchCounter(torch.overrides.TorchFunctionMode):
    """Every torch function dispatched inside the block, by name. Counts
    view ops and kernels alike -- on a launch-bound step both cost a
    Python round-trip, which is the thing being bounded."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_function__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.ops.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))


def _dispatches_for(n_keys: int, *, total_scale: float) -> list[str]:
    losses = {f"term_{i}": torch.tensor(float(i) + 0.125) for i in range(n_keys)}
    losses["total"] = torch.tensor(2.5)
    sums = _DeviceLossSums()
    sums.add_losses(losses, total_scale=total_scale)  # first microbatch: no running sum yet
    with _DispatchCounter() as counter:
        sums.add_losses(losses, total_scale=total_scale)  # second: the accumulate path too
    return counter.ops


@pytest.mark.parametrize("total_scale", [1.0, 2.0])
def test_the_dispatch_count_per_microbatch_does_not_grow_with_the_key_count(
    total_scale: float,
) -> None:
    """⚑ The Codex finding on PR #496: a per-key `.to(float64)` plus a per-key
    add is dozens of tiny dispatches per microbatch on a launch-bound step --
    the overhead the accumulator exists to remove. The count must be a small
    constant: one stack, one detach, one cast, one add (plus the scale's
    getitem/mul/setitem when accumulating), whatever the key count."""
    few = _dispatches_for(4, total_scale=total_scale)
    many = _dispatches_for(40, total_scale=total_scale)

    assert few == many, f"dispatch count grew with the key count:\n{few}\n{many}"
    assert len(many) <= 8, many


def test_a_key_that_appears_mid_window_still_matches_the_host_fold() -> None:
    """Optional terms (a channel-balance loss, a disarmed blend) come and go
    between steps. The vectorised accumulator must keep the old dict's
    first-seen order and per-key sums, digit for digit."""
    gen = torch.Generator().manual_seed(20260902)

    def draw() -> torch.Tensor:
        return torch.rand((), generator=gen) * 7.3 + 0.01

    window = [
        [{"total": draw(), "policy_ce": draw()}],
        [{"total": draw(), "policy_ce": draw(), "channel_balance": draw()}],
        [{"total": draw(), "channel_balance": draw(), "policy_ce": draw(), "wdl_ce": draw()}],
        [{"total": draw(), "policy_ce": draw()}],
    ]

    device = _device_sums(window, accum_steps=1)
    host = _host_reference(window, accum_steps=1)

    assert device == host
    assert list(device) == ["loss", "policy_ce", "channel_balance", "wdl_ce"]


def test_mixed_dtype_scalars_promote_exactly_as_the_old_stack_did() -> None:
    """Under autocast the total can be bf16 while diagnostics are fp32; the old
    path stacked them (promoting to fp32) before `.tolist()`. Same stack, same
    promotion, same bits."""
    window = [
        [{
            "total": torch.tensor(1.3, dtype=torch.bfloat16),
            "policy_ce": torch.tensor(0.7, dtype=torch.float32),
            "wdl_ce": torch.tensor(2.9, dtype=torch.float32),
        }]
        for _ in range(3)
    ]

    assert _device_sums(window, accum_steps=2) == _host_reference(window, accum_steps=2)


def test_a_negative_zero_first_value_keeps_the_host_folds_positive_zero() -> None:
    """The host fold began every key with `0.0 + v`, so a `-0.0` first value
    became `+0.0`. `==` cannot see the difference; the sign bit can."""
    import math

    sums = _DeviceLossSums()
    sums.add_losses({"total": torch.tensor(-0.0), "policy_ce": torch.tensor(1.5)})
    window = _DeviceLossSums()
    window.merge(sums)
    flat = _materialize_device_scalars([t for _, t in window.items()])

    assert flat[0] == 0.0
    assert math.copysign(1.0, flat[0]) == 1.0, "sign of zero differs from the host fold"


def test_exact_train_steps_pools_masked_metrics_by_eligible_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the exact callsite, raw accumulation, and post-build override."""
    sum_key, weight_key = trainer_mod._EXACT_MASKED_METRIC_FIELDS[
        "sf_volatility_loss"
    ]

    class ExactBuffer:
        exact_without_replacement = True

    def run(exact: bool) -> trainer_mod.TrainMetrics:
        trainer = _make_trainer(tmp_path / ("exact" if exact else "replacement"))
        param = next(trainer.model.parameters())
        calls = 0
        means = (5.0, 1.0)
        eligible = (1.0, 2.0)
        totals = (10.0, 2.0)
        batches = (
            {"x": torch.zeros((4, 4, 8, 8))},
            {"x": torch.zeros((2, 4, 8, 8))},
        )

        def fake_compute_loss(
            out: Any, batch: Any, **kwargs: Any,
        ) -> dict[str, torch.Tensor]:
            nonlocal calls
            del out, batch
            report_exact = bool(kwargs.pop("report_exact_masked_sums"))
            assert report_exact is exact
            idx = calls
            calls += 1
            zero = param.sum() * 0.0
            result = {
                "total": zero + totals[idx],
                **dict.fromkeys(_ZERO_LOSS_KEYS, zero),
                "sf_volatility": zero + means[idx],
            }
            if report_exact:
                result[sum_key] = zero + means[idx] * eligible[idx]
                result[weight_key] = zero + eligible[idx]
            return result

        monkeypatch.setattr(trainer_mod, "compute_loss", fake_compute_loss)
        monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})
        monkeypatch.setattr(
            trainer,
            "_iter_training_batches",
            lambda *_args, **_kwargs: iter(batches),
        )
        monkeypatch.setattr(trainer, "_matrix_grad_norm", lambda: 0.0)
        monkeypatch.setattr(trainer, "_zclip_step", lambda **_kwargs: (0.0, None))
        monkeypatch.setattr(trainer.opt, "step", lambda: None)

        return trainer.train_steps(
            cast(Any, ExactBuffer() if exact else None),
            batch_size=4,
            steps=2,
        )

    exact_metrics = run(True)
    replacement_metrics = run(False)

    assert exact_metrics.sf_volatility_loss == 7.0 / 3.0
    assert exact_metrics.loss == 44.0 / 6.0
    assert replacement_metrics.sf_volatility_loss == 3.0
    assert replacement_metrics.loss == 6.0


def test_the_pipeline_timing_keys_reach_the_ray_report_row() -> None:
    """A field that lands only in TensorBoard is this repo's signature defect;
    the Ray progress row is enumerated by hand, so pin every timing key there."""
    from chess_anti_engine.tune import trainable_report
    from chess_anti_engine.train.trainer import _PIPELINE_PHASE_GPU_KEY, _PIPELINE_RESIDUAL_KEY

    keys = [*_PIPELINE_PHASE_GPU_KEY, _PIPELINE_RESIDUAL_KEY,
            *(twin for twin in _PIPELINE_PHASE_GPU_KEY.values() if twin is not None)]
    defaults = trainable_report._train_metrics_dict(None)
    missing = [k for k in keys if k not in defaults]

    assert not missing, f"timing keys absent from the Ray report row: {missing}"


def test_the_pipeline_timing_values_round_trip_into_the_ray_report_row() -> None:
    """Membership is not a value read: `_train_metrics_dict` enumerates the
    row by hand, so a key wired to a literal `0.0` or to the wrong field
    passes the test above. Push ten distinct spans through a real
    `TrainMetrics` and read every one back off the row (Grok review, PR #496)."""
    import dataclasses

    from chess_anti_engine.tune import trainable_report
    from chess_anti_engine.train.trainer import _PIPELINE_PHASE_GPU_KEY, _PIPELINE_RESIDUAL_KEY, TrainMetrics

    keys = [*_PIPELINE_PHASE_GPU_KEY, _PIPELINE_RESIDUAL_KEY,
            *(twin for twin in _PIPELINE_PHASE_GPU_KEY.values() if twin is not None)]
    spans: dict[str, Any] = {key: 0.125 * (i + 1) for i, key in enumerate(keys)}
    required: dict[str, Any] = {
        f.name: 0.0
        for f in dataclasses.fields(TrainMetrics)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    row = trainable_report._train_metrics_dict(TrainMetrics(**required, **spans))

    wrong = {key: (row.get(key), spans[key]) for key in keys if row.get(key) != spans[key]}
    assert not wrong, f"Ray row does not carry the TrainMetrics span: {wrong}"
