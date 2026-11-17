"""Real Tiny union load/Trainer/checkpoint plumbing; fictional caller trust only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any

import pytest
import torch

from scripts import core_finite_learner as core
from scripts import core_selfplay_union_stored_row_once as entry
from scripts.bootstrap_checkpoint_resume import require_equal
from chess_anti_engine.replay.fixed_population import manifest_identity
from tests.test_core_finite_learner import contract as tiny_contract, frozen
from tests.test_core_selfplay_union_manifest import compiled, legacy, plan


def method(tmp_path: Path, c: dict[str, Any], name: str) -> dict[str, str]:
    n, b = c["population"]["rows"], c["batch_size"]
    return frozen(tmp_path / (name + ".method.json"), {
        "study_uuid": c["study_uuid"], "bootstrap": {"model": c["initial_model"], "architecture": c["architecture"],
            "donor_recorded_step": c["initial_model_step"], "student_start_step": 0},
        "requested_population": {"stored_sources": c["population"]["sources"], "stored_rows": n},
        "training": {"batch_size": b, "closed_segments_updates": c["updates_per_segment"],
            "expected_optimizer_updates": sum(c["updates_per_segment"]), "last_batch_real_rows": n % b or b,
            "seed": c["seed"], "order_seed": c["sampling"]["order_seed"], "mirror_probability": c["mirror_prob"],
            "learning_rate": c["lr"], "runtime": c["runtime"], "policy_anchor_coefficient": 0,
            "no_priority_diff_focus_skip_or_extra_pass": True, "max_rows": c["limits"]["max_rows"],
            "max_total_decoded_bytes": c["limits"]["max_decoded_bytes"]},
        "scope": "fictional caller-selected software fixture, not production authority"})


def contract(tmp_path: Path, *, split: bool = True) -> tuple[dict[str, Any], dict[str, str]]:
    old, _ = tiny_contract(tmp_path)
    branches = [{"kind": "legacy", "manifest": legacy(tmp_path, generation=8, collection=i, rows=rows,
                 marker=1.0, architecture=old["architecture"])} for i, rows in ((1, 3), (2, 2))]
    pref = plan(tmp_path, branches, expected_sources=2, expected_rows=5, architecture=old["architecture"])
    manifest, mref = compiled(tmp_path, pref)
    identity = manifest_identity(manifest, mref)
    c = {"schema": entry.CONTRACT, "study_uuid": "44444444-4444-4444-8444-444444444444", "run_uuid": old["run_uuid"],
         "source_revision": old["source_revision"], "source_files": {n: core.file_sha(core.REPO / n) for n in entry.SOURCE_FILES},
         "architecture": old["architecture"], "initial_model": old["donor"], "initial_model_step": 9000,
         "batch_size": 2, "updates_per_segment": [1, 2] if split else [3], "seed": 45,
         "runtime": old["runtime"], "lr": {"peak": 1e-4, "floor": 1e-4, "warmup_steps": 0,
             "warmup_start": 1e-4, "first_period": 10}, "mirror_prob": 0.5,
         "limits": {"max_rows": 5, "max_decoded_bytes": 32 << 20, "max_file_bytes": 8 << 20,
                    "max_total_metadata_bytes": 128 << 20}, "population": {**identity, "plan": pref},
         "sampling": {"schema": entry.SAMPLING, "order_seed": 1673, "expected_rows": 5,
             "presentations_per_row": 1, "mirror_shares_quota": True,
             "partial_batch": "EMIT_FINAL_REMAINDER_ONCE_NO_PADDING", "lineage_uuid": old["run_uuid"],
             "quota_ledger": str(tmp_path / "quota.jsonl"), "order": entry.ORDER,
             "permutation_sha256": core.digest(entry.permutation_for(5, 1673))}}
    c["method"] = method(tmp_path, c, "selected")
    return c, frozen(tmp_path / "union-contract.json", c)


def stage(tmp_path: Path, cref: dict[str, str], c: dict[str, Any], segment: int,
          parent: dict[str, Any] | None, name: str) -> dict[str, str]:
    return frozen(tmp_path / (name + ".stage.json"), {"schema": entry.STAGE,
        "contract_sha256": cref["sha256"], "segment": segment, "population": c["population"],
        "parent": parent, "output": str(tmp_path / name)})


def closed_parent(tmp_path: Path, cref: dict[str, str], boundary: dict[str, Any], *, proof: bool = False) -> dict[str, Any]:
    module = "scripts.core_selfplay_union_stored_row_once_proof" if proof else "scripts.core_selfplay_union_stored_row_once"
    command = [sys.executable, "-B", "-m", module, "--contract", cref["path"], "--contract-sha256", cref["sha256"],
               "--stage", boundary["stage"]["path"], "--stage-sha256", boundary["stage"]["sha256"]]
    terminal = frozen(tmp_path / f"{Path(boundary['checkpoint']['path']).parent.name}.terminal.json", {
        "status": "PASS", "returncode": 0, "owned_processes_closed": True,
        "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": "f" * 64, "command": command})
    path = Path(boundary["checkpoint"]["path"]).parent / "boundary.json"
    return {"boundary": {"path": str(path), "sha256": core.file_sha(path)},
            "owner_terminal": terminal, "owner_sha256": "f" * 64, "command": command}


def test_real_union_split_resume_and_tail_match_uninterrupted_model_rng_and_rows(tmp_path: Path) -> None:
    c, cref = contract(tmp_path)
    first = entry.execute(cref, stage(tmp_path, cref, c, 1, None, "first"))
    second = entry.execute(cref, stage(tmp_path, cref, c, 2, closed_parent(tmp_path, cref, first), "second"))
    assert first["pass_complete"] is False
    assert second["pass_complete"] is True
    assert second["quota_state"]["cursor"] == 5
    assert second["quota_state"]["batches"] == second["step"] == 3
    assert second["quota_state"]["last_batch_rows"] == 1
    assert second["applied_updates"] == 2
    assert second["applied_views"] == second["metrics"]["train_samples_seen"] == 3
    records = [json.loads(line) for ref in second["draw_journals"] for line in Path(ref["path"]).read_text().splitlines()]
    ids = [(r["generation"], r["source_sha256"], r["row_index"]) for record in records for r in record["rows"]]
    assert len(ids) == len(set(ids)) == 5
    assert [len(record["rows"]) for record in records] == [2, 2, 1]
    assert {r[0] for r in ids} == {8}  # Colliding branch generations remain genuine.
    assert len({r[1] for r in ids}) == 2  # Equal board content is not deduplicated.
    direct = copy.deepcopy(c)
    direct["updates_per_segment"] = [3]
    direct["sampling"]["quota_ledger"] = str(tmp_path / "direct-quota.jsonl")
    direct["method"] = method(tmp_path, direct, "direct")
    direct_ref = frozen(tmp_path / "direct.contract.json", direct)
    full = entry.execute(direct_ref, stage(tmp_path, direct_ref, direct, 1, None, "full"))
    a = torch.load(second["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(full["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler", "zclip", "step"):
        require_equal(a[key], b[key], "union split vs uninterrupted " + key)
    assert second["rng"] == full["rng"]
    assert second["quota_state"] == full["quota_state"]
    assert second["quota_ledger_checkpoint"]["sha256"] == full["quota_ledger_checkpoint"]["sha256"]
    assert a[entry.CHECKPOINT_KEY]["population_identity"] == second["population_identity"]
    assert a[entry.CHECKPOINT_KEY]["source_origin_binding"] == second["source_origin_binding"]


@pytest.mark.parametrize("fault", ["manifest", "origins", "dose", "quota", "ledger_output", "anchor"])
def test_union_identity_recipe_refusals_precede_model_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    c, _ = contract(tmp_path)
    if fault == "manifest":
        c["population"]["manifest"]["sha256"] = "0" * 64
    elif fault == "origins":
        c["population"]["origin_map_sha256"] = "0" * 64
    elif fault == "dose":
        c["updates_per_segment"] = [2]
    elif fault == "quota":
        c["sampling"]["presentations_per_row"] = 2
    elif fault == "ledger_output":
        c["sampling"]["quota_ledger"] = str(tmp_path / "refused/quota.jsonl")
    else:
        c["policy_anchor"] = {"coefficient": 1}
    cref = frozen(tmp_path / "invalid.contract.json", c)
    calls: list[bool] = []
    monkeypatch.setattr(core, "make_trainer", lambda *_args: calls.append(True))
    with pytest.raises(ValueError):
        entry.execute(cref, stage(tmp_path, cref, c, 1, None, "refused"))
    assert not calls
    assert not (tmp_path / "refused").exists()


def test_pure_closed_consumer_avoids_checkpoint_journal_and_shard_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, cref = contract(tmp_path, split=False)
    full = entry.execute(cref, stage(tmp_path, cref, c, 1, None, "full"))
    binding = closed_parent(tmp_path, cref, full)
    original = core.file_sha

    def metadata_only(path: Path) -> str:
        assert path.suffix == ".json"
        return original(path)

    monkeypatch.setattr(core, "file_sha", metadata_only)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("metadata consumer decoded checkpoint"))
    result = entry.validate_closed_segment(cref, binding)
    assert result["actual_views"] == 5
    assert result["pass_complete"] is True
    assert result["population_identity"] == full["population_identity"]
    assert "caller-trusted closed lineage metadata" in result["scope"]


def test_actual_resume_refuses_journal_tail_ahead_of_checkpoint_before_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, cref = contract(tmp_path)
    first = entry.execute(cref, stage(tmp_path, cref, c, 1, None, "first"))
    binding = closed_parent(tmp_path, cref, first)
    value = copy.deepcopy(first)
    value["quota_state"].update(cursor=0, batches=0, last_batch_rows=0, row_prefix_sha256=hashlib.sha256(b"").hexdigest())
    target = Path(binding["boundary"]["path"])
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o555
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    target.parent.chmod(0o755)
    try:
        target.chmod(0o644)
        target.unlink()
        binding["boundary"] = frozen(target, value)
    finally:
        target.chmod(0o444)
        target.parent.chmod(0o555)
    calls: list[bool] = []
    monkeypatch.setattr(core, "make_trainer", lambda *_args: calls.append(True))
    with pytest.raises(ValueError, match="checkpoint/journal cursor mismatch"):
        entry.execute(cref, stage(tmp_path, cref, c, 2, binding, "refused"))
    assert not calls
    assert not (tmp_path / "refused/learner.pt").exists()


@pytest.mark.parametrize("fault", ["membership", "step_start", "updates", "draws_sha256", "draws_path"])
def test_pure_closed_consumer_refuses_rehashed_boundary_alias_tampering(tmp_path: Path, fault: str) -> None:
    c, cref = contract(tmp_path, split=False)
    full = entry.execute(cref, stage(tmp_path, cref, c, 1, None, "full"))
    binding = closed_parent(tmp_path, cref, full)
    value = copy.deepcopy(full)
    if fault == "membership":
        value["membership"] = [[7, *value["membership"][0][1:]], *value["membership"][1:]]
    elif fault == "step_start":
        value["step_start"] = 1
        value["drawn_rows"] = value["applied_views"] = value["metrics"]["train_samples_seen"] = 3
    elif fault == "updates":
        value["applied_updates"] = value["metrics"]["train_steps_done"] = 4
    elif fault == "draws_sha256":
        value["draws_sha256"] = "0" * 64
    else:
        value["draw_journals"][-1]["path"] = str(tmp_path / "forged-draws.jsonl")
    target = Path(binding["boundary"]["path"])
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o555
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    target.parent.chmod(0o755)
    try:
        target.chmod(0o644)
        target.unlink()
        binding["boundary"] = frozen(target, value)
    finally:
        target.chmod(0o444)
        target.parent.chmod(0o555)
    with pytest.raises(ValueError, match="closed union membership/selected segment counters|current journal hash/path aliases"):
        entry.validate_closed_segment(cref, binding)
