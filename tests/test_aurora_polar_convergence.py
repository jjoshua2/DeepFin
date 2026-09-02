"""Polar convergence is an INSTRUMENT, so its calibration is the test.

`MODEL_OPT_AUDIT.md` M4-1: at production's `aurora_polar_steps: 8` the Polar
Express iteration does not converge on the Aurora matrix group's real momentum,
and NOTHING measured it for the life of the run. The A8 A/B (8 -> 12) reads the
metric these tests pin, so "the function returns a plausible number" is not the
bar. Four properties are:

1. **It measures the quantity the audit measured.** `polar_convergence` is
   pinned against Addendum II's B3 table -- SQUARE and RECT, `full` and `orth`,
   at 8 AND 12 polar steps -- on the real `checkpoint_000478` momentum spectra
   banked in `tests/data/aurora_polar_momentum_spectra.npz`. Rename the metric
   onto a different quantity and eight reference cells move.
2. **It reads the PRODUCTION update.** `_aurora_update`'s square branch and its
   trailing `sqrt(rows/cols)` rescale leave the reading alone, so the number on
   the row is the number in the reference table rather than a cousin of it.
3. **A negative control exists in both directions.** A random orthogonal matrix
   must read converged (1.0 / 0.0); a purpose-built ill-conditioned matrix at 8
   steps must read UNconverged, and must converge once given enough steps. A
   metric that cannot fail is not a metric.
4. **It reaches progress.csv off the production path.** The optimizer only
   samples when the trainer arms it, the trainer arms it once per iteration,
   and `train_steps` -> `TrainMetrics` -> `_train_metrics_dict` carries every
   key. Drop the splat anywhere in that chain and `test_train_steps_*` or
   `test_progress_report_*` fails.
5. **The column's own statistic is pinned separately from the group mean.**
   B3's table is a MEAN over 16 square tensors; the column samples ONE. Those
   differ by 1.95x on `orth_err`, and the first version of this PR's ledger
   gate quoted the mean at a one-tensor column -- a bar a correctly installed
   instrument would have failed for the wrong reason (PR #327 review). The
   designated tensor's own readings and the across-tensor spread are pinned
   here so that gate cannot be re-derived from the wrong population.

The fixture is spectra, not tensors, because the Polar Express iterate is
orthogonally equivariant (`scripts/extract_aurora_momentum_spectra.py` states
the derivation) -- so a matrix rebuilt from a banked spectrum reproduces the
reference readings exactly, at 83 kB instead of 23 MB. Verified across three
reconstruction seeds and both float32 and float64: all four decimals stable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import torch.nn as nn

from chess_anti_engine.train import aurora as aurora_mod
from chess_anti_engine.train import trainer as trainer_mod
from chess_anti_engine.train.aurora import (
    AuroraWithAuxAdam,
    _aurora_update,
    _polar_factor,
    polar_convergence,
)
from chess_anti_engine.train.trainer import Trainer
from chess_anti_engine.tune import trainable_report

FIXTURE = Path(__file__).parent / "data" / "aurora_polar_momentum_spectra.npz"

# MODEL_OPT_AUDIT.md Addendum II, table B3, fp32 rows, as MEANS over the group:
#   "SQ full" / "SQ orth" over the 16 square out_proj momentum buffers,
#   "RECT full" / "RE orth" over the 4 sampled rectangular ffn buffers.
# `full` is sigma_min/sigma_max; `orth` is ||QQ^T - I||_F / sqrt(n).
# Reproduced here to four decimals from the banked spectra.
_B3_REFERENCE: dict[tuple[str, int], tuple[float, float]] = {
    ("square", 8): (0.0209, 0.1082),
    ("square", 12): (0.2489, 0.0439),
    ("rect", 8): (0.0604, 0.0800),
    ("rect", 12): (0.6220, 0.0257),
}
# The reference cells are quoted to four decimals, so the tolerance is the
# quoting resolution and not a fudge factor: a real drift in what the metric
# computes moves these by whole percent, not by 5e-5.
_B3_TOLERANCE = 5e-4

# Production's realized Aurora group at checkpoint_000478, read off
# `trainer.pt`'s own param group by scripts/extract_aurora_momentum_spectra.py.
_PROD_POLAR_METHOD = "polar_express"
_PROD_POLAR_SAFETY = 1.01
_PROD_PP_ITERATIONS = 3
_PROD_PP_BETA = 0.25
_PROD_POLAR_STEPS = 8


def _load_fixture() -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int]]]:
    data = np.load(FIXTURE)
    shapes = {}
    for row in data["shapes"]:
        name, rows, cols = str(row).split(",")
        shapes[name] = (int(rows), int(cols))
    spectra = {name: data[f"s_{name}"] for name in shapes}
    return spectra, shapes


def _reconstruct(spectrum: np.ndarray, shape: tuple[int, int], seed: int) -> torch.Tensor:
    """``U diag(s) V^T`` for a seeded orthogonal ``U``/``V``.

    Equivariance makes the choice of basis irrelevant to the polar path; the
    seed is fixed only so a failure is reproducible.
    """
    rows, cols = shape
    k = min(rows, cols)
    gen = torch.Generator().manual_seed(seed)
    u, _ = torch.linalg.qr(torch.randn(rows, k, generator=gen, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(cols, k, generator=gen, dtype=torch.float64))
    svals = torch.from_numpy(np.asarray(spectrum, dtype=np.float64))
    return ((u * svals) @ v.transpose(0, 1)).to(torch.float32)


def _real_momentum(seed: int = 12345) -> dict[str, list[torch.Tensor]]:
    spectra, shapes = _load_fixture()
    out: dict[str, list[torch.Tensor]] = {"square": [], "rect": []}
    for name, shape in sorted(shapes.items()):
        key = "square" if shape[0] == shape[1] else "rect"
        out[key].append(_reconstruct(spectra[name], shape, seed))
    return out


def _polar_only(mat: torch.Tensor, steps: int) -> torch.Tensor:
    """The polar factor alone -- exactly what the B3 table measured."""
    return _polar_factor(
        mat,
        method=_PROD_POLAR_METHOD,
        steps=steps,
        eps=1e-7,
        safety=_PROD_POLAR_SAFETY,
        work_dtype=None,
    )


def _production_update(mat: torch.Tensor, steps: int) -> torch.Tensor:
    """The update `AuroraWithAuxAdam.step` applies, at production's group hparams."""
    return _aurora_update(
        mat,
        pp_iterations=_PROD_PP_ITERATIONS,
        pp_beta=_PROD_PP_BETA,
        polar_steps=steps,
        polar_method=_PROD_POLAR_METHOD,
        polar_dtype="fp16",
        polar_safety=_PROD_POLAR_SAFETY,
        eps=1e-8,
    )


def _group_mean(mats: list[torch.Tensor], steps: int, fn: Any) -> tuple[float, float]:
    readings = [polar_convergence(fn(m, steps)) for m in mats]
    return (
        float(np.mean([r[0] for r in readings])),
        float(np.mean([r[1] for r in readings])),
    )


# --- 1. calibration against the banked reference ----------------------------


@pytest.mark.parametrize(("shape_class", "steps"), sorted(_B3_REFERENCE))
def test_polar_convergence_reproduces_addendum_ii_b3(shape_class: str, steps: int) -> None:
    momentum = _real_momentum()[shape_class]
    assert momentum, f"fixture carries no {shape_class} tensors"

    sv_ratio, orth_err = _group_mean(momentum, steps, _polar_only)

    ref_ratio, ref_orth = _B3_REFERENCE[(shape_class, steps)]
    assert sv_ratio == pytest.approx(ref_ratio, abs=_B3_TOLERANCE)
    assert orth_err == pytest.approx(ref_orth, abs=_B3_TOLERANCE)


# --- 1b. the statistic the COLUMN computes, which is not the group mean ------

# The DESIGNATED square tensor -- group index 0, the one `step()` samples --
# through the production `_aurora_update`, on checkpoint_000478's momentum.
# These are the numbers a live row carries. They are NOT `_B3_REFERENCE`'s
# group means, and the gap is the whole reason this block exists.
_DESIGNATED_SQUARE: dict[int, tuple[float, float]] = {
    8: (0.0273, 0.2114),
    12: (0.3275, 0.0614),
}


def _designated_square() -> torch.Tensor:
    spectra, shapes = _load_fixture()
    square = sorted(name for name, shape in shapes.items() if shape[0] == shape[1])
    return _reconstruct(spectra[square[0]], shapes[square[0]], 12345)


@pytest.mark.parametrize("steps", sorted(_DESIGNATED_SQUARE))
def test_the_designated_square_tensor_reads_its_own_pinned_values(steps: int) -> None:
    # PR #327 review, blocking finding. The ledger gate first quoted B3's
    # 16-tensor MEANS (0.0209 / 0.1082 at 8 steps) as centres for a column that
    # samples ONE tensor. This tensor reads 0.2114 on `orth_err` at 8 steps --
    # 1.95x that centre -- so a correctly installed instrument on a correctly
    # applied arm would have failed the gate for the wrong reason.
    reading = polar_convergence(_production_update(_designated_square(), steps))

    expected = _DESIGNATED_SQUARE[steps]
    assert reading[0] == pytest.approx(expected[0], abs=_B3_TOLERANCE)
    assert reading[1] == pytest.approx(expected[1], abs=_B3_TOLERANCE)


def test_the_group_mean_is_not_a_bar_a_one_tensor_column_can_meet() -> None:
    # State the blocking finding as an assertion rather than as prose, so a
    # future reader cannot re-derive the gate from B3's means without this
    # failing first.
    at8 = _DESIGNATED_SQUARE[8]
    assert at8[1] > _B3_REFERENCE[("square", 8)][1] * 1.5

    squares = _real_momentum()["square"]
    orth_at8 = [polar_convergence(_production_update(m, 8))[1] for m in squares]
    ratio_at8 = [polar_convergence(_production_update(m, 8))[0] for m in squares]
    assert max(orth_at8) / min(orth_at8) > 2.0
    assert max(ratio_at8) / min(ratio_at8) > 50.0


def test_the_paired_arm_over_control_ratio_survives_the_choice_of_tensor() -> None:
    # Why the ledger gate's confirming leg is a RATIO on one tensor rather than
    # an absolute. Measured across all 16 squares: PE-12/PE-8 on `sv_ratio`
    # spans 11.40-12.36 (1.08x) while the PE-8 absolute spans 94x. The gate's
    # bar is >= 8.0, which the loosest tensor here clears by 1.4x.
    squares = _real_momentum()["square"]
    paired = [
        polar_convergence(_production_update(m, 12))[0]
        / polar_convergence(_production_update(m, 8))[0]
        for m in squares
    ]

    assert max(paired) / min(paired) < 1.20
    assert min(paired) > 8.0
    assert min(paired) == pytest.approx(11.40, abs=0.05)
    assert max(paired) == pytest.approx(12.36, abs=0.05)


def test_the_reference_actually_discriminates_eight_from_twelve_steps() -> None:
    # The calibration above is only worth something if the two step counts it
    # pins are far apart. If they were not, the A/B this metric adjudicates
    # would be unreadable no matter how the metric was wired.
    for shape_class in ("square", "rect"):
        at8 = _B3_REFERENCE[(shape_class, 8)]
        at12 = _B3_REFERENCE[(shape_class, 12)]
        assert at12[0] > at8[0] * 5.0
        assert at12[1] < at8[1] * 0.5


# --- 2. the production path reads the same number ---------------------------


def test_production_update_matches_the_polar_path_on_square_tensors() -> None:
    # The square branch of `_aurora_update` IS `_polar_factor` plus a positive
    # scalar, so the number the optimizer's own update yields must be the
    # reference number. This is what makes the row's value quotable against
    # the audit table rather than merely correlated with it.
    square = _real_momentum()["square"]
    for steps in (8, 12):
        production = _group_mean(square, steps, _production_update)
        reference = _B3_REFERENCE[("square", steps)]
        assert production[0] == pytest.approx(reference[0], abs=_B3_TOLERANCE)
        assert production[1] == pytest.approx(reference[1], abs=_B3_TOLERANCE)


def test_reading_is_invariant_to_the_updates_overall_scale() -> None:
    # `_aurora_update` multiplies by sqrt(rows/cols) and `aurora_uw_floor` can
    # multiply again. Neither may move the metric, or "measured on the applied
    # update" and "measured on the polar factor" would be different claims.
    mat = _real_momentum()["rect"][0]
    base = polar_convergence(_polar_only(mat, 8))
    for scale in (1e-3, 0.5, 7.0, 1e3):
  # float32 rescale roundoff only; the metric itself divides the scale out
  # exactly, so the residual is 1e-8-ish, not the 1e-3 a real dependence
  # on magnitude would produce.
        scaled = polar_convergence(_polar_only(mat, 8) * scale)
        assert scaled[0] == pytest.approx(base[0], rel=1e-6)
        assert scaled[1] == pytest.approx(base[1], rel=1e-6)


def test_the_polar_path_reading_depends_only_on_the_spectrum() -> None:
    # The premise the 83 kB fixture rests on. If a reconstruction basis could
    # move these numbers, the calibration above would be pinning an artifact of
    # one seed rather than production's conditioning.
    for shape_class in ("square", "rect"):
        first = _group_mean(_real_momentum(seed=1)[shape_class], 8, _polar_only)
        second = _group_mean(_real_momentum(seed=999)[shape_class], 8, _polar_only)
        assert first[0] == pytest.approx(second[0], abs=1e-4)
        assert first[1] == pytest.approx(second[1], abs=1e-4)


def test_rectangular_production_path_is_not_the_b3_rect_column() -> None:
    # A live trap for whoever reads the A/B, stated as a test. B3's RECT column
    # is the polar factor of the RAW momentum, but production feeds the
    # rectangular branch a ROW-NORMALISED matrix `pp_iterations` times, and
    # that loop is NOT orthogonally equivariant -- so unlike everything else
    # here it cannot be pinned from a spectrum. Measured directly on
    # checkpoint_000478's own rectangular momentum tensors, the production path
    # reads sv_ratio 0.3926 / orth 0.0374 at 8 steps and 1.0000 / 0.0000 at 12,
    # against B3's polar-only 0.0604 / 0.0800 and 0.6220 / 0.0257. Reading a
    # live `aurora_polar_sv_ratio_rect` against 0.0604 would therefore report a
    # large change where there is none.
    rect = _real_momentum()["rect"]
    polar_ratio, polar_orth = _group_mean(rect, 8, _polar_only)
    prod_ratio, prod_orth = _group_mean(rect, 8, _production_update)
    assert polar_ratio == pytest.approx(_B3_REFERENCE[("rect", 8)][0], abs=_B3_TOLERANCE)
    assert prod_ratio != pytest.approx(polar_ratio, rel=1e-2)
    assert prod_orth != pytest.approx(polar_orth, rel=1e-2)


# --- 3. negative controls ---------------------------------------------------


def test_a_random_orthogonal_matrix_reads_fully_converged() -> None:
    gen = torch.Generator().manual_seed(7)
    q, _ = torch.linalg.qr(torch.randn(64, 64, generator=gen, dtype=torch.float64))

    sv_ratio, orth_err = polar_convergence(q)

    assert sv_ratio == pytest.approx(1.0, abs=1e-9)
    assert orth_err == pytest.approx(0.0, abs=1e-9)


def test_an_ill_conditioned_matrix_reads_unconverged_at_eight_steps() -> None:
    # Prescribed spectrum, condition number 1e5 -- past the point where PE-8's
    # amplification budget (prod of the leading coefficients, 6363) can lift
    # sigma_min to sigma_max (Addendum A3). The metric MUST see that, and MUST
    # stop seeing it when the iteration is given enough steps.
    gen = torch.Generator().manual_seed(11)
    n = 128
    svals = torch.logspace(0.0, -5.0, n, dtype=torch.float64)
    u, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    mat = ((u * svals) @ v.transpose(0, 1)).to(torch.float32)

    at8 = polar_convergence(_polar_only(mat, 8))
    at32 = polar_convergence(_polar_only(mat, 32))

    assert at8[0] < 1.0
    assert at8[1] > 1e-3
    assert at32[0] == pytest.approx(1.0, abs=1e-4)
    assert at32[1] < at8[1]


def test_a_degenerate_update_raises_rather_than_reporting_a_ratio() -> None:
    with pytest.raises(ValueError, match="degenerate update spectrum"):
        polar_convergence(torch.zeros((8, 8), dtype=torch.float32))


# --- 4. it reaches progress.csv off the production path ---------------------


class _SquareAndRectModel(nn.Module):
    """Two Aurora-owned matrices, one of each shape class, plus AdamW tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 8, bias=False), nn.Linear(8, 12, bias=False)])
        self.head = nn.Linear(8, 3)

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
        "optimizer": "aurora",
        "use_amp": False,
        "log_dir": tmp_path,
        "tb_log_interval": 1000,
        "prefetch_batches": False,
        "aurora_polar_method": _PROD_POLAR_METHOD,
        "aurora_polar_steps": _PROD_POLAR_STEPS,
    }
    trainer_kwargs.update(kwargs)
  # Seeded so the two assertions below are about the metric and not about
  # which random 8x8 the model happened to draw.
    torch.manual_seed(4242)
    return Trainer(_SquareAndRectModel(), **trainer_kwargs)


_LOSS_KEYS = (
    "policy_ce", "soft_policy_ce", "future_policy_ce", "wdl_ce", "sf_move_ce",
    "sf_eval_ce", "categorical_ce", "volatility", "sf_volatility", "moves_left",
)


def _stub_losses(trainer: Trainer, monkeypatch: pytest.MonkeyPatch) -> None:
    targets = [p for p in trainer.model.parameters() if p.ndim == 2]

    def fake_compute_loss(out: Any, batch: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del out, batch, kwargs
        total = cast(torch.Tensor, sum((t * t).sum() for t in targets))
        losses: dict[str, torch.Tensor] = {"total": total}
        losses.update(dict.fromkeys(_LOSS_KEYS, total.detach()))
        return losses

    monkeypatch.setattr(trainer_mod, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})


def test_the_optimizer_samples_nothing_until_the_caller_arms_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The sampling counter is the thing that makes "the sampler never fired"
    # visible. Without it, an unarmed collector and a converged optimizer look
    # identical on the row: all zeros.
    trainer = _make_trainer(tmp_path)
    _stub_losses(trainer, monkeypatch)
    opt = cast(AuroraWithAuxAdam, trainer.opt)

    trainer._run_optimizer_step(
        step_sums=trainer_mod._DeviceLossSums(), step_acc_sums={}, step_opt_stats={},
        buf=cast(Any, None), batch_size=1,
        collect_optimizer_stats=False,
        batch_iter=iter([{"x": torch.zeros((1, 4, 8, 8))}] * trainer.accum_steps),
    )
    assert opt.last_polar_stats == {}

    trainer._run_optimizer_step(
        step_sums=trainer_mod._DeviceLossSums(), step_acc_sums={}, step_opt_stats={},
        buf=cast(Any, None), batch_size=1,
        collect_optimizer_stats=True,
        batch_iter=iter([{"x": torch.zeros((1, 4, 8, 8))}] * trainer.accum_steps),
    )
    stats = opt.last_polar_stats
    assert stats["aurora_polar_sv_samples"] == pytest.approx(2.0)
    assert stats["aurora_polar_sv_errors"] == pytest.approx(0.0)
    assert stats["aurora_polar_steps_configured"] == pytest.approx(float(_PROD_POLAR_STEPS))
  # These two tensors are small and well conditioned, so the APPLIED UPDATE is
  # a true orthogonal step and reads 1.0000 / 0.0000. The momentum buffer it
  # was built from reads 0.0102 / 0.6213 and 0.1945 / 0.6058. Pinning the
  # converged end therefore catches a metric that measured the gradient, the
  # momentum or the weight instead of the update Aurora applies.
    for shape_class in ("square", "rect"):
        assert stats[f"aurora_polar_sv_ratio_{shape_class}"] > 0.99
        assert stats[f"aurora_polar_orth_err_{shape_class}"] < 1e-2


def test_run_optimizer_step_arms_the_collector_from_its_own_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `collect_optimizer_stats` is the only gate; if `_run_optimizer_step`
    # stopped forwarding it the sampler would never fire in production, where
    # nothing else calls the setter.
    trainer = _make_trainer(tmp_path)
    _stub_losses(trainer, monkeypatch)
    opt = cast(AuroraWithAuxAdam, trainer.opt)

    seen: list[bool] = []
    real = opt.set_collect_polar_stats
    monkeypatch.setattr(
        opt, "set_collect_polar_stats",
        lambda collect: (seen.append(bool(collect)), real(collect))[1],
    )
    for flag in (False, True):
        trainer._run_optimizer_step(
            step_sums=trainer_mod._DeviceLossSums(), step_acc_sums={}, step_opt_stats={},
            buf=cast(Any, None), batch_size=1,
            collect_optimizer_stats=flag,
            batch_iter=iter([{"x": torch.zeros((1, 4, 8, 8))}] * trainer.accum_steps),
        )
    assert seen == [False, True]


def test_train_steps_carries_polar_stats_into_train_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole chain on the production path: train_steps arms the sampler on
    # one step, splats `last_polar_stats` into `_build_metrics`, and the fields
    # exist on TrainMetrics to receive them. Drop the splat and this fails with
    # zeros; drop a field and `_build_metrics` raises TypeError.
    trainer = _make_trainer(tmp_path)
    _stub_losses(trainer, monkeypatch)

    def fake_batches(buf: Any, **kwargs: Any):
        del buf, kwargs
        while True:
            yield {"x": torch.zeros((1, 4, 8, 8))}

    monkeypatch.setattr(trainer, "_iter_prefetched_batches", fake_batches)

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=3)

    assert metrics.train_steps_done == 3
    assert metrics.aurora_polar_sv_samples == pytest.approx(2.0)
    assert metrics.aurora_polar_steps_configured == pytest.approx(float(_PROD_POLAR_STEPS))
    assert metrics.aurora_polar_sv_ratio_square > 0.99
    assert metrics.aurora_polar_sv_ratio_rect > 0.99
    assert metrics.aurora_polar_orth_err_square < 1e-2
    # M4-2: the uw-effective pair is only interpretable against the LR it was
    # multiplied by, so that LR has to be on the same row.
    assert metrics.aurora_uw_lr > 0.0
    assert metrics.aurora_uw_count == pytest.approx(2.0)


def test_a_failed_sample_is_counted_and_named_but_an_oom_is_re_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    # PR #327 review, non-blocking note, taken. `RuntimeError` is both what
    # torch.linalg raises on a non-converged decomposition AND what an
    # exhausted allocator raises. Counting the first is right; counting the
    # second would turn "the run is out of memory" into an incremented integer
    # and let the step proceed. So OOM propagates, everything else is counted
    # WITH its exception class in the log -- a bare counter is not diagnosable.
    # It propagates THROUGH the trainer's optimizer-step boundary, so it
    # arrives as `OptimizerStepFailed` with the OOM as its cause and the
    # Aurora tensor named -- an optimizer-phase failure the CUDA retry must
    # not take (see `test_aurora_adamw_foreach.py`, P2-3).
    trainer = _make_trainer(tmp_path)
    _stub_losses(trainer, monkeypatch)
    opt = cast(AuroraWithAuxAdam, trainer.opt)

    def boom_linalg(update: torch.Tensor) -> tuple[float, float]:
        del update
        raise RuntimeError("linalg.svd: failed to converge")

    monkeypatch.setattr(aurora_mod, "polar_convergence", boom_linalg)
    with caplog.at_level("WARNING"):
        trainer._run_optimizer_step(
            step_sums=trainer_mod._DeviceLossSums(), step_acc_sums={}, step_opt_stats={},
            buf=cast(Any, None), batch_size=1,
            collect_optimizer_stats=True,
            batch_iter=iter([{"x": torch.zeros((1, 4, 8, 8))}] * trainer.accum_steps),
        )
    assert opt.last_polar_stats["aurora_polar_sv_errors"] == pytest.approx(2.0)
    assert opt.last_polar_stats["aurora_polar_sv_samples"] == pytest.approx(0.0)
    assert "RuntimeError" in caplog.text

    def boom_oom(update: torch.Tensor) -> tuple[float, float]:
        del update
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(aurora_mod, "polar_convergence", boom_oom)
    with pytest.raises(aurora_mod.OptimizerStepFailed) as excinfo:
        trainer._run_optimizer_step(
            step_sums=trainer_mod._DeviceLossSums(), step_acc_sums={}, step_opt_stats={},
            buf=cast(Any, None), batch_size=1,
            collect_optimizer_stats=True,
            batch_iter=iter([{"x": torch.zeros((1, 4, 8, 8))}] * trainer.accum_steps),
        )
    assert isinstance(excinfo.value.__cause__, torch.cuda.OutOfMemoryError)
    assert excinfo.value.location is not None
    assert excinfo.value.location.startswith("Aurora matrix parameter")


def test_progress_report_carries_the_polar_and_uw_columns() -> None:
    metrics = trainer_mod.TrainMetrics(
        **dict.fromkeys(
            (
                "loss", "policy_loss", "soft_policy_loss", "future_policy_loss",
                "wdl_loss", "sf_move_loss", "sf_move_acc", "sf_eval_loss",
                "categorical_loss", "volatility_loss", "sf_volatility_loss",
                "moves_left_loss",
            ),
            0.0,
        ),
        aurora_polar_steps_configured=12.0,
        aurora_polar_sv_samples=2.0,
        aurora_polar_sv_errors=0.0,
        aurora_polar_sv_ratio_square=0.2489,
        aurora_polar_sv_ratio_rect=0.6220,
        aurora_polar_orth_err_square=0.0439,
        aurora_polar_orth_err_rect=0.0257,
        aurora_uw_lr=6.0e-5,
        aurora_uw_count=48.0,
        aurora_uw_ratio_median=0.5,
        aurora_uw_effective_ratio_min=1.0e-6,
        aurora_uw_effective_ratio_median=2.0e-6,
    )

    row = trainable_report._train_metrics_dict(metrics)

    assert row["aurora_polar_steps_configured"] == pytest.approx(12.0)
    assert row["aurora_polar_sv_ratio_square"] == pytest.approx(0.2489)
    assert row["aurora_polar_sv_ratio_rect"] == pytest.approx(0.6220)
    assert row["aurora_polar_orth_err_square"] == pytest.approx(0.0439)
    assert row["aurora_polar_orth_err_rect"] == pytest.approx(0.0257)
    assert row["aurora_polar_sv_samples"] == pytest.approx(2.0)
    assert row["aurora_polar_sv_errors"] == pytest.approx(0.0)
    assert row["aurora_uw_lr"] == pytest.approx(6.0e-5)
    assert row["aurora_uw_count"] == pytest.approx(48.0)
    assert row["aurora_uw_effective_ratio_median"] == pytest.approx(2.0e-6)
    # The no-metrics fallback must carry the same key set, or a column appears
    # and disappears between rows and progress.csv rotates mid-run.
    assert set(trainable_report._train_metrics_dict(None)) == set(row)


def test_every_polar_stat_key_is_a_train_metrics_field() -> None:
    # `_build_metrics(**last_polar_stats)` is a keyword splat: a key the
    # dataclass does not declare is a TypeError mid-iteration, not a dropped
    # column. Derive the key set from the emitting function rather than
    # restating it.
    from chess_anti_engine.train.aurora import _polar_stats

    emitted = set(
        _polar_stats(
            {"square": (0.5, 0.1), "rect": (0.6, 0.2)}, polar_steps=8, errors=0,
        ),
    )
    fields = {f.name for f in trainer_mod.dataclasses.fields(trainer_mod.TrainMetrics)}
    assert emitted <= fields
    assert emitted <= set(trainable_report._train_metrics_dict(None))


def test_polar_sampling_is_deterministic_given_the_same_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No rng is consulted: the designated tensors are the first of each shape
    # class in group order. Two trainers seeded identically must agree bit for
    # bit, so a moved reading is a moved optimizer and never a resample.
    rows = []
    for _ in range(2):
        torch.manual_seed(4242)
        trainer = _make_trainer(tmp_path)
        _stub_losses(trainer, monkeypatch)
        opt = cast(AuroraWithAuxAdam, trainer.opt)
        trainer._run_optimizer_step(
            step_sums=trainer_mod._DeviceLossSums(), step_acc_sums={}, step_opt_stats={},
            buf=cast(Any, None), batch_size=1,
            collect_optimizer_stats=True,
            batch_iter=iter([{"x": torch.zeros((1, 4, 8, 8))}] * trainer.accum_steps),
        )
        rows.append(dict(opt.last_polar_stats))
        monkeypatch.undo()
    assert rows[0] == rows[1]
    assert not math.isnan(rows[0]["aurora_polar_sv_ratio_square"])
