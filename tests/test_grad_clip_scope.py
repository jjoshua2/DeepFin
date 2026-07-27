"""The gradient-norm clip must be computed over the params it can actually move.

Aurora's update is the polar factor of the (momentum-smoothed) gradient, taken
by Newton-Schulz. The polar factor is scale-invariant, so multiplying the
gradient by a constant leaves the update bit-for-bit alone -- measured on
`_polar_factor`: max|polar(cG) - polar(G)| = 1.6e-07 at c=0.90, exactly 0.0 at
c=0.50, 1.8e-07 at c=0.10. A global gradient-norm clip is therefore EXACTLY
INERT for the Aurora group, yet the group's norm was still folded into the
quantity the clip decides on.

Measured on `checkpoint_000122`, 8 real batches through the real loss:
||g|| aurora 3.648, adamw 12.137, global 12.673 -- the AdamW group carries
91.7% of the global norm^2 on 71.4% of the parameters, and the global norm sits
4.4% above the only quantity the clip can affect. That inflation is not fixed;
it is whatever the ratio between two independently drifting norms happens to be.

These tests pin three things: the scope is the AdamW params and nothing else,
the split is reported per iteration rather than reconstructed by an offline
probe, and the non-finite guard that the clip's removal from the Aurora group
makes necessary.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import zclip.zclip as zclip_mod

from chess_anti_engine.model import build_model, model_config_from_flat_config
from chess_anti_engine.train import trainer as trainer_mod
from chess_anti_engine.train.trainer import (
    Trainer,
    _GradClipScope,
    _optimizer_matrix_param_ids,
    split_matrix_and_clipped_params,
)
from chess_anti_engine.tune import trainable_report
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

_REPO = Path(__file__).resolve().parents[1]

# Measured 2026-07-27 on configs/pbt2_small.yaml (optimizer aurora,
# matrix_optimizer_scope mlp_out), deduped by storage. The same 63,084,128
# total that tests/test_param_count.py pins.
_PROD_MATRIX_TENSORS = 48
_PROD_MATRIX_PARAMS = 18_033_664
_PROD_CLIPPED_TENSORS = 433
_PROD_CLIPPED_PARAMS = 45_050_464


class _TinyMuonModel(nn.Module):
    """Smallest model with both a matrix group and an AdamW group.

    `blocks.0.weight` is the only tensor the `default` scope hands Aurora;
    `embed.weight` joins it under Muon. Everything else -- biases, the head --
    is AdamW, which is what the clip may touch.
    """

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


def _make_trainer(tmp_path: Path, **kwargs: Any) -> Trainer:
    trainer_kwargs: dict[str, Any] = {
        "device": "cpu",
        "lr": 1e-3,
        "optimizer": "aurora",
        "use_amp": False,
        "log_dir": tmp_path,
        "tb_log_interval": 1000,
        "prefetch_batches": False,
    }
    trainer_kwargs.update(kwargs)
    return Trainer(_TinyMuonModel(), **trainer_kwargs)


def _named(trainer: Trainer) -> dict[int, str]:
    return {id(p): n for n, p in trainer.model.named_parameters()}


def _clipped(trainer: Trainer) -> list[torch.nn.Parameter]:
    """Exactly the parameters zclip is handed."""
    return list(trainer._grad_clip_target.parameters())


def _grads(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    grads = [p.grad for p in params]
    assert all(g is not None for g in grads)
    return [g for g in grads if g is not None]


# --- the zclip contract the scope object stands on -------------------------


def _attrs_used_on_first_arg(func: Any) -> set[str]:
    """Attribute names zclip reads off the module it is handed."""
    tree = ast.parse(inspect.getsource(func).lstrip())
    fn = tree.body[0]
    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    args = fn.args.args
    target = args[1].arg if fn.name != "is_fsdp_model" else args[0].arg
    return {
        node.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == target
    }


# `clip_grad_norm_` and `trainer` are reached only inside `ZClip.step`'s
# `is_fsdp_model(model)` branch, which `_GradClipScope` forces False (asserted
# below). Everything else zclip reads off the module must be one of the two the
# scope implements.
_ZCLIP_FSDP_ONLY_ATTRS = {"clip_grad_norm_", "trainer"}


@pytest.mark.parametrize(
    "func",
    [
        zclip_mod.is_fsdp_model,
        zclip_mod.ZClip._compute_grad_norm,
        zclip_mod.ZClip.apply_in_place_clipping,
        zclip_mod.ZClip._apply_clipping,
        zclip_mod.ZClip.step,
    ],
)
def test_zclip_only_touches_parameters_and_modules(func: Any) -> None:
    # The whole design rests on this: if a future zclip reaches for anything
    # else off the module (`.named_parameters()`, a real `nn.Module` API),
    # `_GradClipScope` silently stops being a valid stand-in and this test is
    # the thing that says so.
    used = _attrs_used_on_first_arg(func)
    assert used <= {"parameters", "modules"} | _ZCLIP_FSDP_ONLY_ATTRS


def test_scope_forces_zclips_local_non_fsdp_path() -> None:
    scope = _GradClipScope([nn.Parameter(torch.zeros(2, 2))])

    assert zclip_mod.is_fsdp_model(scope) is False
    assert list(scope.modules()) == []
    assert len(list(scope.parameters())) == 1


# --- the split itself ------------------------------------------------------


def test_split_matches_the_optimizers_own_matrix_groups(tmp_path: Path) -> None:
    for optimizer in ("aurora", "muon"):
        trainer = _make_trainer(tmp_path / optimizer, optimizer=optimizer)
        names = _named(trainer)

        matrix = {names[id(p)] for p in trainer._matrix_clip_params}
        from_opt = {names[i] for i in _optimizer_matrix_param_ids(trainer.opt)}

        assert matrix == from_opt
        assert matrix  # a silently empty matrix group would make the fix a no-op
        # Muon claims the embedding under the legacy `default` scope, Aurora
        # does not: the split must track `include_embed_default`, not guess.
        assert ("embed.weight" in matrix) is (optimizer == "muon")


def test_clipped_group_is_the_complement_and_nothing_is_lost(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    clipped = _clipped(trainer)

    matrix_ids = {id(p) for p in trainer._matrix_clip_params}
    clipped_ids = {id(p) for p in clipped}
    all_ids = {id(p) for p in trainer.model.parameters() if p.requires_grad}

    assert matrix_ids & clipped_ids == set()
    assert matrix_ids | clipped_ids == all_ids


def test_optimizers_without_a_scale_invariant_group_keep_the_whole_model(
    tmp_path: Path,
) -> None:
    # Behaviour must be bit-identical to before for every optimizer that has
    # not been shown to have the scale-invariance property.
    for optimizer in ("adamw", "nadamw"):
        trainer = _make_trainer(tmp_path / optimizer, optimizer=optimizer)
        assert trainer._matrix_clip_params == []
        assert trainer._grad_clip_target is trainer.model


def test_a_scope_that_disagrees_with_the_optimizer_is_a_hard_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The predicate and the optimizer groups are built from the same function
    # today. If they are ever edited apart, the clip is silently mis-scoped --
    # the exact defect class this change removes -- so it must not be silent.
    real = trainer_mod._matrix_optimizer_filter
    calls = {"n": 0}

    def drifted(scope: str, *, include_embed_default: bool) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:  # the optimizer builds its groups first
            return real(scope, include_embed_default=include_embed_default)
        return lambda name, p: False  # ...and the clip scope then disagrees

    monkeypatch.setattr(trainer_mod, "_matrix_optimizer_filter", drifted)

    with pytest.raises(ValueError, match="disagrees with the optimizer"):
        _make_trainer(tmp_path)


def test_production_config_split_is_the_measured_one() -> None:
    flat = flatten_run_config_defaults(load_yaml_file(_REPO / "configs" / "pbt2_small.yaml"))
    model = build_model(model_config_from_flat_config(flat))

    matrix, clipped = split_matrix_and_clipped_params(
        model,
        optimizer=str(flat["optimizer"]),
        matrix_optimizer_scope=str(flat["matrix_optimizer_scope"]),
    )

    def unique_params(params: list[torch.nn.Parameter]) -> int:
        return sum({p.untyped_storage().data_ptr(): p.numel() for p in params}.values())

    assert (len(matrix), unique_params(matrix)) == (_PROD_MATRIX_TENSORS, _PROD_MATRIX_PARAMS)
    assert (len(clipped), unique_params(clipped)) == (_PROD_CLIPPED_TENSORS, _PROD_CLIPPED_PARAMS)


# --- what zclip actually measures and rescales ------------------------------


def _set_grads(trainer: Trainer, *, matrix: float, clipped: float) -> None:
    matrix_ids = {id(p) for p in trainer._matrix_clip_params}
    for param in trainer.model.parameters():
        fill = matrix if id(param) in matrix_ids else clipped
        param.grad = torch.full_like(param, fill)


def test_reported_norm_is_the_clipped_group_alone(tmp_path: Path) -> None:
    # The load-bearing assertion. A scope that silently falls back to all
    # params reports the global norm here, which is a different number.
    trainer = _make_trainer(tmp_path, zclip_max_norm=None)
    _set_grads(trainer, matrix=10.0, clipped=1.0)

    expected_clipped = math.sqrt(
        sum(float(p.numel()) for p in _clipped(trainer))
    )
    expected_matrix = 10.0 * math.sqrt(
        sum(float(p.numel()) for p in trainer._matrix_clip_params)
    )
    expected_global = math.hypot(expected_clipped, expected_matrix)

    total_norm, _stats = trainer._zclip_step(collect_stats=True)

    assert total_norm == pytest.approx(expected_clipped, rel=1e-5)
    assert total_norm != pytest.approx(expected_global, rel=1e-3)
    assert trainer._matrix_grad_norm() == pytest.approx(expected_matrix, rel=1e-5)


def test_clipping_never_rescales_the_matrix_groups_gradients(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, zclip_max_norm=0.5)
    _set_grads(trainer, matrix=10.0, clipped=1.0)
    matrix_before = [g.clone() for g in _grads(trainer._matrix_clip_params)]
    clipped_before = [g.clone() for g in _grads(_clipped(trainer))]

    _total_norm, stats = trainer._zclip_step(collect_stats=True)

    assert stats is not None
    assert stats["hard_clip"] == 1.0
    for grad, before in zip(_grads(trainer._matrix_clip_params), matrix_before, strict=True):
        assert torch.equal(grad, before)
    shrunk = [
        not torch.allclose(grad, before)
        for grad, before in zip(_grads(_clipped(trainer)), clipped_before, strict=True)
    ]
    assert all(shrunk)


def test_the_cap_binds_on_the_clipped_group_not_the_global_norm(tmp_path: Path) -> None:
    # An AdamW-only cap at 5.0 yields an effective AdamW norm of 5.0, where the
    # old global cap yielded 5.0 / sqrt(1 + (aurora/adamw)^2) < 5.0. Same
    # number in the yaml, a slightly LARGER effective allowance -- the point of
    # the change is that the allowance now means what it says.
    trainer = _make_trainer(tmp_path, zclip_max_norm=5.0)
    _set_grads(trainer, matrix=100.0, clipped=1.0)

    trainer._zclip_step(collect_stats=True)

    after = math.sqrt(sum(float(g.pow(2).sum()) for g in _grads(_clipped(trainer))))
    assert after == pytest.approx(5.0, rel=1e-4)


# --- per-group metrics ------------------------------------------------------


def test_grad_clip_metric_kwargs_reports_both_groups() -> None:
    kwargs = trainer_mod._grad_clip_metric_kwargs(
        [1.0, 3.0], {"clipped": 1, "nonfinite_grad": 1}, [4.0, 6.0],
    )

    assert kwargs["grad_norm_adamw"] == pytest.approx(2.0)
    assert kwargs["grad_norm_aurora"] == pytest.approx(5.0)
    # grad_norm_mean is the clipped group's, which is what makes the I11
    # median-vs-cap comparison a comparison of one quantity with itself.
    assert kwargs["grad_norm_mean"] == pytest.approx(kwargs["grad_norm_adamw"])
    assert kwargs["grad_nonfinite_skip_rate"] == pytest.approx(0.5)
    assert trainer_mod._grad_clip_metric_kwargs([1.0], {}, None)["grad_norm_aurora"] == 0.0


def _drive_one_step(
    trainer: Trainer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scale: float,
    on_matrix: bool,
) -> dict[str, float]:
    matrix_ids = {id(p) for p in trainer._matrix_clip_params}
    target = next(
        p
        for p in trainer.model.parameters()
        if (id(p) in matrix_ids) is on_matrix
    )

    def fake_compute_loss(out: Any, batch: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del out, batch, kwargs
        return {"total": (target * target).sum() * scale}

    monkeypatch.setattr(trainer_mod, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})

    step_opt_stats: dict[str, float] = {}
    trainer._run_optimizer_step(
        step_sums={},
        step_acc_sums={},
        step_opt_stats=step_opt_stats,
        buf=cast(Any, None),
        batch_size=1,
        batch_iter=iter([{"x": torch.zeros((1, 4, 8, 8))}] * trainer.accum_steps),
    )
    return step_opt_stats


def test_run_optimizer_step_reports_the_matrix_group_norm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path)

    stats = _drive_one_step(trainer, monkeypatch, scale=1.0, on_matrix=True)

    # The gradient lives entirely on a matrix param, so the clipped group's
    # norm is 0 and the matrix group's is not. Nothing else can produce this.
    assert stats["grad_norm_aurora"] > 0.0
    assert stats["grad_norm"] == pytest.approx(0.0)
    assert stats.get("nonfinite_grad", 0.0) == 0.0


def test_progress_report_carries_the_new_columns() -> None:
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
        grad_norm_aurora=3.648,
        grad_norm_adamw=12.137,
        grad_nonfinite_skip_rate=0.25,
    )

    row = trainable_report._train_metrics_dict(metrics)

    assert row["grad_norm_aurora"] == pytest.approx(3.648)
    assert row["grad_norm_adamw"] == pytest.approx(12.137)
    assert row["grad_nonfinite_skip_rate"] == pytest.approx(0.25)
    # The no-metrics fallback must carry the same key set, or the columns
    # appear and disappear between rows.
    assert set(trainable_report._train_metrics_dict(None)) >= {
        "grad_norm_aurora", "grad_norm_adamw", "grad_nonfinite_skip_rate",
    }


# --- the non-finite guard ---------------------------------------------------


def test_polar_factor_is_scale_invariant_but_not_nan_invariant() -> None:
    # Why the guard is needed at all, stated as a test rather than as prose:
    # scale-invariance is a statement about FINITE rescales. inf/nan is not a
    # rescale, and Aurora turns it into a hard RuntimeError mid-iteration.
    grad = torch.randn(8, 6, generator=torch.Generator().manual_seed(0))
    base = trainer_mod.AuroraWithAuxAdam.__module__
    aurora = __import__(base, fromlist=["_aurora_update"])

    assert torch.allclose(aurora._aurora_update(grad * 0.5), aurora._aurora_update(grad), atol=1e-5)
    with pytest.raises(RuntimeError, match="non-finite"):
        aurora._aurora_update(grad * float("nan"))


@pytest.mark.parametrize("on_matrix", [True, False])
def test_non_finite_gradients_skip_the_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, on_matrix: bool,
) -> None:
    trainer = _make_trainer(tmp_path)
    before = [p.detach().clone() for p in trainer.model.parameters()]

    stats = _drive_one_step(trainer, monkeypatch, scale=float("inf"), on_matrix=on_matrix)

    assert stats["nonfinite_grad"] == 1.0
    assert "lr" in stats  # the row must still be countable
    for param, snapshot in zip(trainer.model.parameters(), before, strict=True):
        assert torch.equal(param.detach(), snapshot)
    assert all(p.grad is None for p in trainer.model.parameters())


def test_finite_gradients_still_take_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    matrix_param = trainer._matrix_clip_params[0]
    before = matrix_param.detach().clone()

    stats = _drive_one_step(trainer, monkeypatch, scale=1.0, on_matrix=True)

    assert stats.get("nonfinite_grad", 0.0) == 0.0
    assert not torch.equal(matrix_param.detach(), before)


def test_train_steps_surfaces_the_skip_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    matrix_param = trainer._matrix_clip_params[0]

    def fake_compute_loss(out: Any, batch: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del out, batch, kwargs
        zero = torch.zeros(())
        return {
            "total": (matrix_param * matrix_param).sum() * float("inf"),
            **dict.fromkeys(trainer_mod._LOSS_KEY_TO_METRIC_FIELD, zero),
        }

    monkeypatch.setattr(trainer_mod, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})
    monkeypatch.setattr(
        trainer,
        "_iter_prefetched_batches",
        lambda *_args, **_kwargs: iter([{"x": torch.zeros((1, 4, 8, 8))}] * 64),
    )

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=2)

    assert metrics.grad_nonfinite_skip_rate == pytest.approx(1.0)
    assert metrics.grad_norm_samples == 2
