"""Synthetic trusted-admission fixtures; no claim of real outcome qualification."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model, infer_input_planes
from chess_anti_engine.moves import MODEL_POLICY_SIZE
from chess_anti_engine.replay.shard import save_local_shard_arrays
from chess_anti_engine.train import Trainer
from scripts import core_finite_learner as core
from scripts.bootstrap_checkpoint_resume import require_equal


def frozen(path: Path, value: Any) -> dict[str, str]:
    core.write_json(path, value)
    path.chmod(0o444)
    return {"path": str(path), "sha256": core.file_sha(path)}


def contract(tmp_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    cfg = ModelConfig(kind="tiny", input_extra_features="v1", history_rep_fix=True)
    arch = {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)}
    donor = tmp_path / "donor.pt"
    torch.save({"model": build_model(cfg).state_dict(), "arch": arch,
                "step": 9000, "opt": {"must_not_load": True}}, donor)
    donor.chmod(0o444)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=core.REPO,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"), text=True,
    ).strip()
    c = {"schema": "core_finite_learning_v1", "run_uuid": "11111111-1111-4111-8111-111111111111",
         "source_revision": revision,
         "source_files": {p: core.file_sha(core.REPO / p) for p in core.SOURCE_FILES},
         "architecture": arch, "donor": {"path": str(donor), "sha256": core.file_sha(donor)},
         "batch_size": 2, "games_per_generation": 1, "updates_per_generation": [2, 2],
         "replay_generations": 4, "seed": 45, "mirror_prob": 0.5,
         "runtime": {"device": "cpu", "use_amp": False, "torch_version": torch.__version__,
                     "numpy_version": np.__version__, "torch_threads": 2, "cuda_devices": 0},
         "lr": {"peak": 1e-4, "floor": 1e-6, "warmup_steps": 1,
                "warmup_start": 1e-5, "first_period": 10},
         "limits": {"max_rows": 100, "max_decoded_window_bytes": 10 << 20}}
    return c, frozen(tmp_path / "contract.json", c)


def generation(tmp_path: Path, c: dict[str, Any], n: int, actor: str,
               *, network_turn: bool = True) -> dict[str, str]:
    rows = 3
    shard = tmp_path / f"game-{n}.zarr"
    policy = np.zeros((rows, MODEL_POLICY_SIZE), dtype=np.float16)
    policy[:, 0] = 1
    arrays = {"x": np.ones((rows, infer_input_planes(c["architecture"]["input_extra_features"]), 8, 8), dtype=np.float16),
              "policy_target": policy, "wdl_target": np.array([0, 1, 2], dtype=np.int8),
              "priority": np.ones(rows, dtype=np.float32),
              "game_id": np.full(rows, n, dtype=np.int64), "ply_index": np.arange(rows, dtype=np.int32),
              "legal_mask": np.ones_like(policy, dtype=np.uint8),
              "_input_history_encoding": np.asarray(c["architecture"]["input_history_encoding"]),
              "_policy_encoding": np.asarray(c["architecture"]["policy_encoding"]),
              "_history_rep_fix": np.asarray("true")}
    for key in ("has_policy", "has_legal_mask", "is_network_turn", "has_is_network_turn",
                "is_selfplay", "has_is_selfplay", "has_game_id", "has_ply_index"):
        arrays[key] = np.ones(rows, dtype=np.uint8)
    arrays["is_network_turn"][:] = int(network_turn)
    save_local_shard_arrays(shard, arrs=arrays, meta={"games": 1, "positions": rows})
    core.seal(shard)
    inventory = [{"path": str(p.relative_to(shard)), "bytes": p.stat().st_size, "sha256": core.file_sha(p)}
                 for p in sorted(shard.rglob("*")) if p.is_file()]
    served = "a" * 64
    common = {"run_uuid": c["run_uuid"], "source_revision": c["source_revision"], "generation": n,
              "actor_sha256": actor, "physical_game_id": f"{c['run_uuid']}:{n}",
              "served_state_sha256": served, "game_id": n, "rows": rows}
    verification = {**common, "resolved": True, "retained_rows": rows, "scientific_class": "NATURAL"}
    outcome = {**common, "schema": "CORE_PRODUCING_OUTCOME_PROOF_V1", "scientific_class": "NATURAL",
               "verification": verification, "proof_sha256": core.digest(verification)}
    evidence = {"outcome_proof": frozen(tmp_path / f"outcome-{n}.json", outcome),
                "validation": frozen(tmp_path / f"validation-{n}.json", common),
                "provenance": frozen(tmp_path / f"provenance-{n}.json", common)}
    admission = {**common, "status": "VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1", "bank": "train",
                 "resolved": True, "full_history_verified": True, "raw_targets_verified": True,
                 "label_class": "NATURAL", "shard_inventory_sha256": core.digest(inventory), "evidence": evidence}
    game = {"physical_game_id": common["physical_game_id"], "game_id": n, "bank": "train", "rows": rows,
            "label_class": "NATURAL", "producer_proof": frozen(tmp_path / f"admission-{n}.json", admission),
            "shard": {"path": str(shard), "files": inventory}}
    serving = frozen(tmp_path / f"serving-{n}.json", {"actor_sha256": actor,
                     "pre_state_sha256": served, "post_state_sha256": served,
                     "checkpoint_model_state_sha256": served})
    return frozen(tmp_path / f"generation-{n}.json", {
        "schema": "ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1", "run_uuid": c["run_uuid"],
        "source_revision": c["source_revision"], "generation": n, "actor_sha256": actor,
        "actor_serving": serving, "games": [game],
    })


def stage(tmp_path: Path, cref: dict[str, str], refs: list[dict[str, str]],
          parent: dict[str, str] | None, label: str) -> dict[str, str]:
    return frozen(tmp_path / f"stage-{label}.json", {
        "schema": "core_finite_stage_v1", "contract_sha256": cref["sha256"],
        "generation": len(refs), "generations": refs, "parent": parent, "output": str(tmp_path / label),
    })


def test_real_trainer_two_generation_continuation_matches_direct_state_and_rng(tmp_path):
    from scripts.offline_epoch_resume import _restore_rng, _rng_state

    c, cref = contract(tmp_path)
    one = generation(tmp_path, c, 1, c["donor"]["sha256"])
    first = core.execute(cref, stage(tmp_path, cref, [one], None, "first"))
    assert first["step"] == first["applied_updates"] == 2
    assert first["drawn_rows"] == first["applied_views"] == 4
    parent = {"path": str(tmp_path / "first/boundary.json"),
              "sha256": core.file_sha(tmp_path / "first/boundary.json")}
    two = generation(tmp_path, c, 2, first["checkpoint"]["sha256"])
    # Reference directly continues the real Trainer state. The caller also
    # verifies parameter-name identity and restores its explicit RNG sidecar.
    reference = core.make_trainer(c, tmp_path / "reference-log")
    reference.load(Path(first["checkpoint"]["path"]), exact_resume=True)
    rng = np.random.default_rng()
    gens = [json.loads(Path(ref["path"]).read_bytes()) for ref in [one, two]]
    replay = core.load_replay(c, gens, rng, None)
    _restore_rng(first["rng"], rng, np.random)
    expected_metrics = reference.train_steps(replay, batch_size=2, steps=2)
    expected_rng = _rng_state(rng, np.random)
    expected_model = copy.deepcopy(reference.model.state_dict())
    expected_opt = copy.deepcopy(reference.opt.state_dict())
    expected_scheduler = copy.deepcopy(reference._scheduler.state_dict())
    reference.writer.close()
    second_ref = stage(tmp_path, cref, [one, two], parent, "second")
    second = core.execute(cref, second_ref)
    checkpoint = torch.load(second["checkpoint"]["path"], map_location="cpu", weights_only=False)
    require_equal(checkpoint["model"], expected_model, "model")
    require_equal(checkpoint["opt"], expected_opt, "optimizer")
    require_equal(checkpoint["scheduler"], expected_scheduler, "scheduler")
    assert second["rng"] == expected_rng
    assert second["step"] == 4 and second["metrics"]["lr"] == expected_metrics.lr
    assert second["final_step"] == second["step"]
    assert second["final_checkpoint"] == second["checkpoint"]
    assert len(second["membership"]) == 2
    # No replay or overwrite of a completed output.
    with pytest.raises(FileExistsError):
        core.execute(cref, second_ref)


@pytest.mark.parametrize("fault", ["hold", "unresolved", "original", "actor", "rows", "inventory"])
def test_data_refusals_precede_model_construction(tmp_path, monkeypatch, fault):
    c, cref = contract(tmp_path)
    gen_ref = generation(tmp_path, c, 1, c["donor"]["sha256"])
    gen = json.loads(Path(gen_ref["path"]).read_bytes())
    game = gen["games"][0]
    if fault == "hold":
        game["bank"] = "holdout"
    elif fault == "unresolved":
        game["label_class"] = "CENSORED_MAX_PLIES"
    elif fault == "original":
        proof = json.loads(Path(game["producer_proof"]["path"]).read_bytes())
        proof["evidence"] = {}
        game["producer_proof"] = frozen(tmp_path / "bad-proof.json", proof)
    elif fault == "actor":
        gen["actor_sha256"] = "b" * 64
    elif fault == "rows":
        game["rows"] += 1
    else:
        game["shard"]["files"][0]["sha256"] = "0" * 64
    bad = frozen(tmp_path / "bad-generation.json", gen)
    called = []
    monkeypatch.setattr(core, "make_trainer", lambda *_a: called.append(True))
    with pytest.raises(ValueError):
        core.execute(cref, stage(tmp_path, cref, [bad], None, "refused"))
    assert not called
    assert not (tmp_path / "refused").exists()


def test_no_retry_wrapper_stops_cuda_fault_and_nonfinite_before_next_update(tmp_path, monkeypatch):
    c, _ = contract(tmp_path)
    trainer = core.make_trainer(c, tmp_path / "log")
    calls = []
    def cuda_fault(_self, **_kwargs):
        calls.append(True)
        raise RuntimeError("CUDA synthetic failure")
    monkeypatch.setattr(Trainer, "_run_optimizer_step", cuda_fault)
    with pytest.raises(ValueError, match="retries forbidden"):
        trainer._run_optimizer_step(step_opt_stats={})
    assert len(calls) == 1
    monkeypatch.setattr(Trainer, "_run_optimizer_step", lambda _self, **_kw: (1, 0.0))
    with pytest.raises(ValueError, match="nonfinite update"):
        trainer._run_optimizer_step(step_opt_stats={"nonfinite_grad": 1.0})
    trainer.writer.close()


def test_array_predicate_refusal_precedes_any_update(tmp_path, monkeypatch):
    c, cref = contract(tmp_path)
    # The synthetic trusted wrapper lies; the actual stored row flags still
    # have to qualify before Trainer receives a training window.
    one = generation(tmp_path, c, 1, c["donor"]["sha256"], network_turn=False)
    calls = []
    monkeypatch.setattr(Trainer, "train_steps", lambda *_args, **_kwargs: calls.append(True))
    with pytest.raises(ValueError, match="active TRAIN identity"):
        core.execute(cref, stage(tmp_path, cref, [one], None, "bad-arrays"))
    assert not calls
    assert not (tmp_path / "bad-arrays/boundary.json").exists()


def test_contract_rejects_short_restart_and_inert_extra_knob(tmp_path):
    c, _ = contract(tmp_path)
    core.validate_contract(c)
    bad = copy.deepcopy(c)
    bad["lr"]["first_period"] = 1
    with pytest.raises(ValueError, match="restart"):
        core.validate_contract(bad)
    bad = copy.deepcopy(c)
    bad["aux_weight_decay"] = 0.5
    with pytest.raises(ValueError, match="contract fields"):
        core.validate_contract(bad)


def test_core_weights_and_coupled_value_body_are_not_frozen(tmp_path):
    c, _ = contract(tmp_path)
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, use_smolgen=False,
                      input_extra_features="v1", categorical_head_coupled=True,
                      aux_policy_head_dim=8)
    c["architecture"] = {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)}
    trainer = core.make_trainer(c, tmp_path / "log")
    assert trainer.grad_clip_mode == "none" and trainer._swa_model is None
    assert trainer._feature_group_dropout and all(item[2] == 0 for item in trainer._feature_group_dropout)
    assert trainer.w_policy == trainer.w_wdl == 1
    assert trainer.sf_wdl_frac == trainer.search_wdl_frac == 0
    assert all(value == 0 for key, value in core.CORE_KWARGS.items() if key.startswith("w_") and key not in {"w_policy", "w_wdl"})
    assert all(p.requires_grad for p in trainer.model.parameters())
    assert [g["weight_decay"] for g in trainer.opt.param_groups] == [1e-4, 0]
    outputs = trainer.model(torch.zeros(1, infer_input_planes(cfg.input_extra_features), 8, 8))
    outputs["wdl"].sum().backward()
    assert trainer.model.value_wdl.token_proj.weight.grad is not None
    assert trainer.model.value_wdl.net[0].weight.grad is not None
    assert trainer.model.value_categorical_coupled.weight.grad is None
    trainer.writer.close()


def test_failed_state_has_no_completed_boundary(tmp_path, monkeypatch):
    c, cref = contract(tmp_path)
    one = generation(tmp_path, c, 1, c["donor"]["sha256"])
    def fail(_self, *_args, **_kwargs):
        raise ValueError("synthetic terminal failure")
    monkeypatch.setattr(Trainer, "train_steps", fail)
    with pytest.raises(ValueError, match="terminal failure"):
        core.execute(cref, stage(tmp_path, cref, [one], None, "failed"))
    assert not (tmp_path / "failed/boundary.json").exists()
    assert json.loads((tmp_path / "failed/failure.json").read_text())["status"] == "FAILED_CORE_GENERATION_NO_RECOVERY"
    assert (tmp_path / "failed/draws.jsonl").stat().st_mode & 0o222 == 0
