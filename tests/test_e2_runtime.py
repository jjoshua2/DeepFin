from dataclasses import asdict
from typing import Any
import hashlib
import pytest
import torch
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.train.e2_objective import freeze_objective
from chess_anti_engine.train.e2_runtime import (
    optimizer_contract,
    digest,
    trace_digest,
    run_verified_pair,
    state_digest,
)
from tests.test_e2_batch_adapter import qualified as qualified
from tests.test_e2_bank_gate import bank as bank
from chess_anti_engine.train.e2_batch_adapter import load_paired_batch


def setup(qualified: dict[str, Any]):
    qualified = dict(
        qualified,
        indices=(0, 1),
        expected_indices_sha256=hashlib.sha256(b"[0,1]").hexdigest(),
    )
    pair = load_paired_batch(**qualified)
    cfg = ModelConfig(
        kind="transformer",
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
        history_rep_fix=True,
    )
    model = build_model(cfg)
    overrides = {
        "device": "cpu",
        "lr": 3e-6,
        "optimizer": "aurora",
        "matrix_optimizer_scope": "mlp_out",
        "matrix_lr_multiplier": 20.0,
        "warmup_steps": 8,
        "lr_schedule": "sqrt_release",
        "lr_release_cycle_steps": 120,
        "warmup_lr_start": 1e-7,
        "use_amp": False,
        "use_compile": False,
        "feature_dropout_p": 0.0,
        "mirror_prob": 0.0,
        "resid_channel_dropout": 0.0,
        "resid_channel_balance_weight": 0.0,
        "prefetch_batches": False,
        "accum_steps": 1,
        "swa_start": 0,
        "rebuild_sf_targets": False,
        "rebuild_categorical_target": False,
        "aurora_cuda_graphs": False,
        "log_dir": None,
        "w_sf_move": 0.0,
        "w_sf_eval": 0.0,
    }
    plan = optimizer_contract(overrides)
    objective = freeze_objective(
        {"w_wdl": 1.0, "w_categorical": 1.0, "w_sf_move": 0.0, "w_sf_eval": 0.0},
        donor_sha256="a" * 64,
        architecture_sha256=digest(asdict(cfg)),
        optimizer_plan_sha256=digest(plan),
        batch_trace_sha256=trace_digest([pair, pair]),
    )
    return model, objective, pair, overrides


def test_real_transformer_actual_aurora_updates_shared_trace(qualified):
    model, c, p, kwargs = setup(qualified)
    before = state_digest(model)
    rng = torch.get_rng_state().clone()
    out = run_verified_pair(
        model, c, c.sha256, [p, p], trainer_overrides=kwargs, seed=17, synthetic=True
    )
    assert out["status"] == "SYNTHETIC_RUNTIME_ONLY"
    assert out["steps_per_arm"] == 2
    assert state_digest(model) == before
    assert torch.equal(torch.get_rng_state(), rng)
    for arm in out["arms"]:
        assert arm["final_model_sha256"] != before
        assert arm["optimizer_state_entries"] > 0
        assert any(g["use_aurora"] for g in arm["optimizer_group_membership"])
        assert arm["initial_trunk_sha256"] != arm["final_trunk_sha256"]
        assert [r["step"] for r in arm["steps"]] == [1, 2]
    assert out["arms"][0]["final_model_sha256"] != out["arms"][1]["final_model_sha256"]


@pytest.mark.parametrize("fault", ["lr", "trace", "tamper", "non64"])
def test_runtime_gates(qualified, fault):
    model, c, p, kwargs = setup(qualified)
    pairs = [p, p]
    if fault == "lr":
        kwargs["lr"] = 1e-4
    if fault == "trace":
        pairs = [p]
    if fault == "tamper":
        p.search_only["x"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match=r".+"):
        run_verified_pair(
            model,
            c,
            c.sha256,
            pairs,
            trainer_overrides=kwargs,
            seed=17,
            synthetic=fault != "non64",
        )


@pytest.mark.parametrize("cycle", [0, -1, 1, 64])
def test_reject_window_relative_or_wrong_cycle(qualified, cycle):
    _, _, _, kw = setup(qualified)
    kw["lr_release_cycle_steps"] = cycle
    with pytest.raises(ValueError, match="120-step"):
        optimizer_contract(kw)


def test_postwarmup_release_applied_lr_hook(qualified):
    model, c, p, kw = setup(qualified)
    kw.update(warmup_steps=1, lr_release_start_frac=0.0)
    payload = __import__("json").loads(c.payload_json)
    c = freeze_objective(
        payload["effective_shared_kwargs"],
        donor_sha256="a" * 64,
        architecture_sha256=payload["pins"]["architecture"],
        optimizer_plan_sha256=digest(optimizer_contract(kw)),
        batch_trace_sha256=trace_digest([p] * 4),
    )
    out = run_verified_pair(
        model, c, c.sha256, [p] * 4, trainer_overrides=kw, seed=17, synthetic=True
    )
    a, b = [v["steps"] for v in out["arms"]]
    assert [x["group_lr_applied"] for x in a] == [x["group_lr_applied"] for x in b]
    assert a[0]["group_lr_applied"][0] < a[1]["group_lr_applied"][0]
    assert a[3]["group_lr_applied"][0] < a[2]["group_lr_applied"][0]
    assert [x["cursor"] for x in a] == [1, 2, 3, 4]


def test_exact64_schedule_and_release_boundary_without_model_compute():
    import math
    from chess_anti_engine.train.trainer import _SqrtReleaseLRScheduler

    # Independent formula covers64 updates and the later global120-cycle boundary.
    traces = []
    for _ in range(2):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=3e-6)
        scheduler = _SqrtReleaseLRScheduler(
            opt, cycle_steps=120, release_start_frac=0.85, min_scale=0.1
        )
        applied = []
        for step in range(129):
            scheduler.step(step)
            phase = step % 120
            expected = (
                1.0
                if phase <= 102
                else 0.1 + 0.9 * (1.0 - math.sqrt((phase - 102) / 18))
            )
            assert opt.param_groups[0]["lr"] == pytest.approx(3e-6 * expected)
            applied.append(opt.param_groups[0]["lr"])
        traces.append(applied)
    assert traces[0] == traces[1]
    assert len(traces[0][:64]) == 64
    assert traces[0][119] < traces[0][120]


def test_arm1_applied_lr_mismatch_not_hidden_by_presampling(qualified, monkeypatch):
    from chess_anti_engine.train.e2_runtime import _TraceTrainer

    original = _TraceTrainer._iter_training_batches

    def changed(self, *args, **kwargs):
        if self.fixed_loss["search_wdl_frac"] == 0.8:
            for group in self.opt.param_groups:
                group["lr"] *= 0.5
        yield from original(self, *args, **kwargs)

    monkeypatch.setattr(_TraceTrainer, "_iter_training_batches", changed)
    model, c, p, kw = setup(qualified)
    with pytest.raises(ValueError, match="arm LR trace differs"):
        run_verified_pair(
            model, c, c.sha256, [p, p], trainer_overrides=kw, seed=17, synthetic=True
        )


def test_actual_trainer64_tiny_steps_both_arms(qualified):
    import json

    model, c, p, kw = setup(qualified)
    payload = json.loads(c.payload_json)
    # Global120 cycle, warmup8; release boundary moved ONLY for synthetic coverage.
    kw["lr_release_start_frac"] = 0.2
    c = freeze_objective(
        payload["effective_shared_kwargs"],
        donor_sha256="a" * 64,
        architecture_sha256=payload["pins"]["architecture"],
        optimizer_plan_sha256=digest(optimizer_contract(kw)),
        batch_trace_sha256=trace_digest([p] * 64),
    )
    out = run_verified_pair(
        model, c, c.sha256, [p] * 64, trainer_overrides=kw, seed=17, synthetic=True
    )
    a, b = [arm["steps"] for arm in out["arms"]]
    assert [v["step"] for v in a] == list(range(1, 65))
    assert [v["cursor"] for v in b] == list(range(1, 65))
    assert [v["group_lr_applied"] for v in a] == [v["group_lr_applied"] for v in b]
    assert a[0]["group_lr_applied"][0] < a[8]["group_lr_applied"][0]
    assert a[63]["group_lr_applied"][0] < a[32]["group_lr_applied"][0]
    assert out["batch_size"] == 2
    assert out["status"] == "SYNTHETIC_RUNTIME_ONLY"


def test_optimizer_membership_rejects_foreign_parameter():
    from chess_anti_engine.train.e2_runtime import group_membership
    from types import SimpleNamespace

    model = torch.nn.Linear(2, 2)
    fake = SimpleNamespace(
        param_groups=[{"params": [torch.nn.Parameter(torch.zeros(1))]}]
    )
    with pytest.raises(ValueError, match="foreign"):
        group_membership(model, fake)
