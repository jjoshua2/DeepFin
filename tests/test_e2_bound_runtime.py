import json
from dataclasses import asdict
import pytest
import torch
from chess_anti_engine.model import ModelConfig, ARCH_SCHEMA_VERSION, build_model
from chess_anti_engine.train.e2_objective import freeze_objective, canonical
from chess_anti_engine.train.e2_runtime import optimizer_contract, digest, trace_digest
from scripts.e2_bound_runtime import (
    load_trace,
    run_synthetic,
    sha,
    source_hashes,
    within,
)
from tests.test_e2_finite_bank import test_producer_to_bank_and_actual_loss as make_bank
from tests.test_e2_runtime import setup
from tests.test_e2_batch_adapter import qualified as qualified
from tests.test_e2_bank_gate import bank as bank


def test_strict_checkpoint_bank_trace_runtime(tmp_path, qualified):
    root = tmp_path / "production"
    root.mkdir()
    make_bank(root)
    marker = root / "natural/BANK-READBACK-COMMITTED.json"
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
    arch = {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)}
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({"arch": arch, "model": model.state_dict()}, checkpoint)
    _, _, _, overrides = setup(qualified)
    plan = {
        "synthetic": True,
        "bank_manifest": str(marker),
        "bank_sha256": sha(marker),
        "trace": [[{"ordinal": 0, "indices": [0, 0]}]] * 2,
        "trainer_overrides": overrides,
        "seed": 17,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha(checkpoint),
            "architecture_json": canonical(arch),
        },
        "source_sha256": source_hashes(),
    }
    pairs = load_trace(plan)
    objective = freeze_objective(
        {"w_wdl": 1.0, "w_categorical": 1.0, "w_sf_move": 0.0, "w_sf_eval": 0.0},
        donor_sha256=sha(checkpoint),
        architecture_sha256=digest(arch),
        optimizer_plan_sha256=digest(optimizer_contract(overrides)),
        batch_trace_sha256=trace_digest(pairs),
    )
    plan.update(
        objective_json=objective.payload_json, objective_sha256=objective.sha256
    )
    path = tmp_path / "plan.json"
    path.write_text(canonical(plan))
    output = run_synthetic(path, sha(path), tmp_path)
    result = json.loads((output / "SYNTHETIC-RUNTIME-COMMITTED.json").read_bytes())
    assert result["sha256"] == sha(checkpoint)
    assert result["diagnostic"]["steps_per_arm"] == 2
    assert result["diagnostic"]["status"] == "SYNTHETIC_RUNTIME_ONLY"
    assert result["synthetic_training_performed"]
    assert result["limits"]["cpu_seconds"] == 120


def test_real_route_and_plan_tamper_disabled(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text('{"synthetic":false}')
    with pytest.raises(ValueError, match="synthetic-only"):
        run_synthetic(path, sha(path), tmp_path)
    with pytest.raises(ValueError, match="identity"):
        run_synthetic(path, "0" * 64, tmp_path)


def test_bank_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="escape"):
        within(tmp_path, "../outside")


def test_owned_child_interrupt_is_killed_and_reaped(tmp_path, monkeypatch):
    import scripts.e2_bound_runtime as wrapper

    path = tmp_path / "plan.json"
    path.write_text('{"synthetic":true}')
    calls = []

    class Child:
        pid = 987654321

        def wait(self, timeout=None):
            if timeout is not None:
                raise KeyboardInterrupt()
            calls.append("reaped")

        def poll(self):
            return None

    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(wrapper.os, "killpg", lambda pid, sig: calls.append(pid))
    with pytest.raises(KeyboardInterrupt):
        run_synthetic(path, sha(path), tmp_path)
    assert calls == [987654321, "reaped"]
    assert not list(tmp_path.rglob("SYNTHETIC-RUNTIME-COMMITTED.json"))


def test_full_attempt_order_and_selection_rejected(tmp_path, monkeypatch):
    import scripts.e2_bound_runtime as wrapper

    bank = {
        "status": "FINITE_READBACK_BANK_NOT_EXPERIMENT_READY",
        "attempts": [
            {"ordinal": 0, "exclusion": None},
            {"ordinal": 1, "exclusion": None},
        ],
        "selected_start_ordinals": [1],
    }
    monkeypatch.setattr(wrapper, "read_json", lambda *a, **kw: bank)
    with pytest.raises(ValueError, match="selection"):
        load_trace(
            {
                "bank_manifest": str(tmp_path / "bank.json"),
                "bank_sha256": "0" * 64,
                "synthetic": True,
            }
        )


def test_helper_pin_fails_before_bank_or_checkpoint(tmp_path):
    from scripts.e2_bound_runtime import child

    with pytest.raises(ValueError, match="source identity"):
        child({"source_sha256": {}}, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()
