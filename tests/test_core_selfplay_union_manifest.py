"""Fictional trusted pins, real metadata validation; no historical-data credit."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.replay.fixed_population import canonical, digest, manifest_identity
from chess_anti_engine.replay.shard import save_local_shard_arrays
from scripts import core_finite_learner as core
from scripts import core_selfplay_union_manifest as union

ARCH: dict[str, Any] = {"_schema_version": 20, "input_extra_features": "v1",
    "input_history_encoding": "lc0_root_legacy_meta", "policy_encoding": "lc0_1858", "history_rep_fix": True}


def frozen(path: Path, value: dict[str, Any]) -> dict[str, str]:
    raw = (canonical(value) + "\n").encode()
    path.write_bytes(raw)
    path.chmod(0o444)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def legacy(tmp_path: Path, *, generation: int = 8, collection: int = 1, game_id: int = 7,
           rows: int = 4, marker: float = 1.0, architecture: dict[str, Any] | None = None) -> dict[str, str]:
    """One genuine-shape synthetic legacy source; every compiler gate is real.

    Equal boards and game IDs across branches are deliberate. Distinct source
    metadata/physical namespaces remain distinct. No collector/history/native
    qualification is claimed by these fictional caller-trusted pins.
    """
    root = tmp_path / ("collection-" + str(collection))
    root.mkdir()
    from chess_anti_engine.model import infer_input_planes

    arch = ARCH if architecture is None else architecture
    collection_uuid = f"22222222-2222-4222-8222-{collection:012d}"
    run_uuid = "11111111-1111-4111-8111-111111111111"
    actor, served = "c" * 64, "d" * 64
    physical = collection_uuid + ":0"
    policy = np.zeros((rows, 1858), dtype=np.float16)
    policy[:, 0] = 1
    arrays = {"x": np.full((rows, infer_input_planes(arch["input_extra_features"]), 8, 8), marker, dtype=np.float16),
              "policy_target": policy, "legal_mask": np.ones_like(policy, dtype=np.uint8),
              "wdl_target": np.arange(rows, dtype=np.int8) % 3, "priority": np.ones(rows, dtype=np.float32),
              "game_id": np.full(rows, game_id, dtype=np.int64), "ply_index": np.arange(rows, dtype=np.int32),
              "_input_history_encoding": np.asarray(arch["input_history_encoding"]),
              "_policy_encoding": np.asarray(arch["policy_encoding"]),
              "_history_rep_fix": np.asarray(str(arch["history_rep_fix"]).lower())}
    for key in ("has_policy", "has_legal_mask", "is_network_turn", "has_is_network_turn",
                "is_selfplay", "has_is_selfplay", "has_game_id", "has_ply_index"):
        arrays[key] = np.ones(rows, dtype=np.uint8)
    shard_path = root / "shard.zarr"
    save_local_shard_arrays(shard_path, arrs=arrays,
        meta={"games": 1, "positions": rows, "username": "synthetic_collection_" + str(collection),
              "provenance_verified": False})
    core.seal(shard_path)
    files = [{"path": path.relative_to(shard_path).as_posix(), "bytes": path.stat().st_size,
              "sha256": core.file_sha(path)} for path in sorted(shard_path.rglob("*")) if path.is_file()]
    shard = {"path": str(shard_path), "files": files}
    producer = frozen(root / "producer.json", {"run_uuid": run_uuid, "collection_uuid": collection_uuid,
        "generation": generation, "source": {"revision": "a" * 40},
        "actor": {"sha256": actor, "expected_checkpoint_step": 9000, "architecture": arch},
        "output_root": str(root)})
    audit = frozen(root / "audit.json", {"schema": "CORE_FINITE_ADMISSION_V1", "producer_protocol": producer})
    serving = frozen(root / "actor-serving.json", {"actor_sha256": actor, "pre_state_sha256": served,
        "post_state_sha256": served, "checkpoint_model_state_sha256": served})
    common = {"run_uuid": run_uuid, "source_revision": "a" * 40, "generation": generation,
              "actor_sha256": actor, "physical_game_id": physical, "served_state_sha256": served,
              "game_id": game_id, "rows": rows}
    verification = {**common, "resolved": True, "retained_rows": rows, "scientific_class": "NATURAL"}
    evidence = {
        "outcome_proof": frozen(root / "outcome.json", {**common, "schema": "CORE_PRODUCING_OUTCOME_PROOF_V1",
            "scientific_class": "NATURAL", "verification": verification, "proof_sha256": digest(verification)}),
        "validation": frozen(root / "validation.json", common),
        "provenance": frozen(root / "provenance.json", common),
    }
    proof = frozen(root / "game.json", {**common, "status": "VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1",
        "bank": "train", "resolved": True, "full_history_verified": True, "raw_targets_verified": True,
        "label_class": "NATURAL", "shard_inventory_sha256": digest(files), "audit_protocol": audit, "evidence": evidence})
    return frozen(root / "generation.json", {"schema": "ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1",
        "run_uuid": run_uuid, "source_revision": "a" * 40, "generation": generation,
        "actor_sha256": actor, "actor_serving": serving, "audit_protocol": audit,
        "root_postclosure_promotion_required": True, "games": [{"physical_game_id": physical,
        "game_id": game_id, "bank": "train", "rows": rows, "label_class": "NATURAL",
        "shard": shard, "producer_proof": proof}]})


def plan(tmp_path: Path, branches: list[dict[str, Any]], *, expected_sources: int,
         expected_rows: int, copies: list[dict[str, Any]] | None = None,
         limits: dict[str, int] | None = None, architecture: dict[str, Any] | None = None) -> dict[str, str]:
    return frozen(tmp_path / "plan.json", {"schema": union.PLAN,
        "architecture": ARCH if architecture is None else architecture,
        "branches": branches, "copies": copies or [], "expected_sources": expected_sources,
        "expected_rows": expected_rows, "limits": limits or {"metadata_file_bytes": 8 << 20,
        "metadata_total_bytes": 128 << 20, "bank_metadata_bytes": 16 << 20,
        "max_games_per_branch": 8, "max_inventory_files": 10000,
        "max_inventory_bytes": 128 << 20, "max_manifest_bytes": 8 << 20}})


def compiled(tmp_path: Path, plan_ref: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    value = union.compile_union(plan_ref, max_file_bytes=8 << 20, max_total_bytes=128 << 20)
    return value, frozen(tmp_path / "union.json", value)


def test_nonconsecutive_branches_preserve_colliding_run_generation_game_ids(tmp_path: Path) -> None:
    branches = [{"kind": "legacy", "manifest": legacy(tmp_path, generation=g, collection=i)}
                for i, g in enumerate((1, 8, 8, 37, 2, 99), 1)]
    p = plan(tmp_path, branches, expected_sources=6, expected_rows=24)
    m, mref = compiled(tmp_path, p)
    assert [s["generation"] for s in m["sources"]] == [1, 8, 8, 37, 2, 99]
    assert len(m["origins"]) == 6
    assert {s["game_id"] for s in m["sources"]} == {7}
    assert len({s["source_sha256"] for s in m["sources"]}) == 6
    assert manifest_identity(m, mref)["rows"] == 24
    assert union.read_union_manifest(mref, p, max_file_bytes=8 << 20, max_total_bytes=128 << 20) == m


def test_same_manifest_copy_is_counted_once_and_keeps_alias(tmp_path: Path) -> None:
    g = legacy(tmp_path)
    p = plan(tmp_path, [{"kind": "legacy", "manifest": g}] * 2, expected_sources=1, expected_rows=4)
    m, _ = compiled(tmp_path, p)
    assert len(m["sources"]) == 1
    assert len(m["aliases"]) == 1
    assert m["aliases"][0]["canonical_slot"] == 0


@pytest.mark.parametrize("declare_copy", [False, True])
def test_copied_inventory_path_requires_explicit_same_origin_proof(tmp_path: Path, declare_copy: bool) -> None:
    original = legacy(tmp_path)
    value = json.loads(Path(original["path"]).read_bytes())
    copy_path = tmp_path / "copied-shard.zarr"
    shutil.copytree(value["games"][0]["shard"]["path"], copy_path)
    value["games"][0]["shard"]["path"] = str(copy_path)
    alias = frozen(tmp_path / "copied-generation.json", value)
    p = plan(tmp_path, [{"kind": "legacy", "manifest": original}, {"kind": "legacy", "manifest": alias}],
             expected_sources=1, expected_rows=4,
             copies=[{"canonical": original, "alias": alias}] if declare_copy else [])
    if not declare_copy:
        with pytest.raises(ValueError, match="unproven copied-source"):
            compiled(tmp_path, p)
    else:
        m, _ = compiled(tmp_path, p)
        assert len(m["origins"]) == 2
        assert m["aliases"][0]["shard"]["path"] == str(copy_path)


@pytest.mark.parametrize("fault", ["holdout", "unresolved", "wrong_rows", "wrong_proof", "wrong_history_claim"])
def test_rehashed_unqualified_metadata_is_refused(tmp_path: Path, fault: str) -> None:
    g = legacy(tmp_path)
    value = json.loads(Path(g["path"]).read_bytes())
    game = value["games"][0]
    if fault == "holdout":
        game["bank"] = "holdout"
    elif fault == "unresolved":
        game["label_class"] = "UNRESOLVED"
    elif fault == "wrong_rows":
        game["rows"] += 1
    else:
        proof = json.loads(Path(game["producer_proof"]["path"]).read_bytes())
        proof["full_history_verified" if fault == "wrong_history_claim" else "actor_sha256"] = (
            False if fault == "wrong_history_claim" else "0" * 64)
        game["producer_proof"] = frozen(tmp_path / "bad-proof.json", proof)
    bad = frozen(tmp_path / "bad-generation.json", value)
    p = plan(tmp_path, [{"kind": "legacy", "manifest": bad}], expected_sources=1, expected_rows=4)
    with pytest.raises(ValueError, match=r"protected or unresolved|legacy TRAIN proof"):
        compiled(tmp_path, p)


def test_existing_recent_accepted_bank_api_is_exercised_without_mocks(tmp_path: Path) -> None:
    from tests.test_core_generation_bank_learner import bank

    binding = bank(tmp_path, {"architecture": ARCH}, 7,
                   {"path": str(tmp_path / "not-opened-model.pt"), "sha256": "c" * 64}, 9000)
    root = json.loads(Path(binding["root_acceptance"]["path"]).read_bytes())
    candidate = json.loads(Path(root["candidate"]["path"]).read_bytes())
    p = plan(tmp_path, [{"kind": "bank", **binding}], expected_sources=len(candidate["games"]),
             expected_rows=sum(g["rows"] for g in candidate["games"]))
    m, _ = compiled(tmp_path, p)
    assert m["origins"][0]["acceptance"] == {"kind": "bank", **binding}
    assert m["origins"][0]["identity"]["generation"] == 7
    assert all(s["bank"] == "train" for s in m["sources"])


def test_metadata_budget_refuses_before_io_and_manifest_plan_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    g = legacy(tmp_path)
    meter = union.Metadata(8 << 20, 1)

    def forbidden_open(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("budget refusal must precede metadata open")

    with monkeypatch.context() as patch:
        patch.setattr(union.os, "open", forbidden_open)
        with pytest.raises(ValueError, match="before read"):
            meter.read(g)
    assert meter.files == {}
    p = plan(tmp_path, [{"kind": "legacy", "manifest": g}], expected_sources=1, expected_rows=4)
    m, _ = compiled(tmp_path, p)
    bad = copy.deepcopy(m)
    bad["origin_map_sha256"] = "0" * 64
    badref = frozen(tmp_path / "bad-union.json", bad)
    with pytest.raises(ValueError, match="origin map"):
        union.read_union_manifest(badref, p, max_file_bytes=8 << 20, max_total_bytes=128 << 20)


@pytest.mark.parametrize("fault", ["rows", "sources", "inventory_bytes"])
def test_fixed_roster_and_inventory_caps_are_effective(tmp_path: Path, fault: str) -> None:
    g = legacy(tmp_path)
    p = plan(tmp_path, [{"kind": "legacy", "manifest": g}], expected_sources=1, expected_rows=4)
    value = json.loads(Path(p["path"]).read_bytes())
    if fault == "rows":
        value["expected_rows"] = 3
    elif fault == "sources":
        value["expected_sources"] = 2
    else:
        value["limits"]["max_inventory_bytes"] = 1
    bad = frozen(tmp_path / "bad-plan.json", value)
    with pytest.raises(ValueError, match=r"union roster exceeds|source roster length|union inventory aggregate"):
        compiled(tmp_path, bad)
