"""Tiny actual Trainer and payload tests with fictional postclosure authorities.

No collector, real actor/bank, outcome qualification or scientific recipe credit.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.model import infer_input_planes
from chess_anti_engine.moves import MODEL_POLICY_SIZE
from chess_anti_engine.replay.shard import save_local_shard_arrays
from scripts import core_finite_learner as core
from scripts import core_generation_bank_acceptance as acceptance
from scripts import core_generation_bank_learner as learner
from scripts.bootstrap_checkpoint_resume import require_equal
from tests.test_core_finite_learner import contract as old_contract, frozen
from tests.test_core_generation_bank_acceptance import graph


def contract(tmp_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    old, _ = old_contract(tmp_path, max_games=2)
    c = {key: value for key, value in old.items() if key not in {"schema", "donor", "updates_per_generation"}}
    c.update(schema=learner.CONTRACT, run_uuid="33333333-3333-4333-8333-333333333333", initial_model=old["donor"], initial_actor_step=9000,
             first_generation=2, updates_per_segment=[2, 2],
             bank_limits={"max_file_bytes": 1 << 20, "max_total_bytes": 4 << 20, "max_games": 3})
    c["source_files"] = {name: core.file_sha(core.REPO / name) for name in learner.SOURCE_FILES}
    return c, frozen(tmp_path / "bank-contract.json", c)


def bank(tmp_path: Path, c: dict[str, Any], logical: int, actor: dict[str, str], actor_step: int,
         *, corruption: str = "") -> dict[str, Any]:
    """Adapt the real candidate-shape fixture; preserve statuses and closure joins."""
    root_dir = tmp_path / ("bank-" + str(logical))
    root_dir.mkdir()
    root_ref, review_ref, _ = graph(root_dir)
    records = {str(path): json.loads(path.read_bytes()) for path in root_dir.rglob("*.json")}
    original_collection = records[str(root_dir / "spec.json")]["collection_uuid"]
    collection = f"22222222-2222-4222-8222-{logical:012d}"

    def identities(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: (logical if key in {"generation", "logical_generation"} else
                          actor_step if key in {"actor_checkpoint_step", "expected_checkpoint_step", "checkpoint_step"} else
                          c["architecture"] if key == "architecture" else identities(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [identities(item) for item in value]
        if value == "c" * 64:
            return actor["sha256"]
        if isinstance(value, str) and value.startswith(original_collection):
            return collection + value[len(original_collection):]
        return value

    records = {path: identities(value) for path, value in records.items()}
    records[str(root_dir / "producer.json")]["actor"]["path"] = actor["path"]
    readout = records[str(root_dir / "audit/readout.json")]
    candidate = records[str(root_dir / "audit/generation.json")]
    for index, game in enumerate(candidate["games"]):
        rows = game["rows"]
        policy = np.zeros((rows, MODEL_POLICY_SIZE), dtype=np.float16)
        policy[:, 0] = 1
        arrays = {"x": np.full((rows, infer_input_planes(c["architecture"]["input_extra_features"]), 8, 8), logical + index, dtype=np.float16),
            "policy_target": policy, "wdl_target": np.arange(rows, dtype=np.int8), "priority": np.ones(rows, dtype=np.float32),
            "game_id": np.full(rows, game["game_id"], dtype=np.int64), "ply_index": np.arange(rows, dtype=np.int32),
            "legal_mask": np.ones_like(policy, dtype=np.uint8),
            "_input_history_encoding": np.asarray(c["architecture"]["input_history_encoding"]),
            "_policy_encoding": np.asarray(c["architecture"]["policy_encoding"]), "_history_rep_fix": np.asarray("true")}
        for key in ("has_policy", "has_legal_mask", "is_network_turn", "has_is_network_turn", "is_selfplay", "has_is_selfplay", "has_game_id", "has_ply_index"):
            arrays[key] = np.ones(rows, dtype=np.uint8)
        if index == 0:
            if corruption == "network":
                arrays["is_network_turn"][:] = 0
            elif corruption == "game_id":
                arrays["game_id"] = np.zeros(rows, dtype=np.float32)
            elif corruption == "ply_index":
                arrays["ply_index"] = np.arange(rows, dtype=np.float32)
            elif corruption == "duplicate_ply":
                arrays["ply_index"][:] = 0
            elif corruption == "illegal":
                arrays["legal_mask"][:, 0] = 0
        path = root_dir / f"payload-{index}.zarr"
        save_local_shard_arrays(path, arrs=arrays, meta={"games": 1, "positions": rows})
        core.seal(path)
        inventory = [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": core.file_sha(p)}
                     for p in sorted(path.rglob("*")) if p.is_file()]
        shard = {"path": str(path), "files": inventory}
        game["shard"] = shard
        readout["audited"][index]["shard"] = shard
        digest = core.digest(inventory)
        readout["audited"][index]["shard_inventory_sha256"] = digest
        records[str(root_dir / f"audit/game-{index}.json")]["shard_inventory_sha256"] = digest

    memo: dict[str, dict[str, str]] = {}

    def write(path: str) -> dict[str, str]:
        if path in memo:
            return memo[path]

        def rebind(value: Any) -> Any:
            if isinstance(value, dict):
                if set(value) == {"path", "sha256"} and value["path"] in records:
                    return write(value["path"])
                return {key: rebind(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rebind(item) for item in value]
            return value

        value = rebind(records[path])
        if path.endswith(("/terminal.json", "/admission.json", "/root.json")):
            # Actual child argv pins the updated protocol, not its stale digest.
            protocol_ref = write(str(root_dir / "protocol.json"))
            command = value["audit_owner"]["command"] if path.endswith("/root.json") else value["command"]
            command[-1] = protocol_ref["sha256"]
        target = Path(path)
        target.chmod(0o644)
        target.unlink()
        memo[path] = frozen(target, value)
        return memo[path]

    root_ref = write(root_ref["path"])
    review_ref = write(review_ref["path"])
    return {"root_acceptance": root_ref, "independent_review": review_ref}


def stage(tmp_path: Path, cref: dict[str, str], banks: list[dict[str, Any]], parent: dict[str, Any] | None,
          name: str) -> dict[str, str]:
    return frozen(tmp_path / (name + "-stage.json"), {"schema": learner.STAGE, "contract_sha256": cref["sha256"],
        "segment": len(banks), "banks": banks, "parent": parent, "output": str(tmp_path / name)})


def parent(tmp_path: Path, cref: dict[str, str], boundary: dict[str, Any]) -> dict[str, Any]:
    command = [sys.executable, "-B", "-m", "scripts.core_generation_bank_learner", "--contract", cref["path"],
               "--contract-sha256", cref["sha256"], "--stage", boundary["stage"]["path"], "--stage-sha256", boundary["stage"]["sha256"]]
    terminal = frozen(tmp_path / "learner-terminal.json", {"status": "PASS", "returncode": 0, "owned_processes_closed": True,
        "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": "f" * 64, "command": command,
        "source_revision": "distinct-owner-wrapper-revision"})
    output = Path(boundary["checkpoint"]["path"]).parent
    return {"boundary": {"path": str(output / "boundary.json"), "sha256": core.file_sha(output / "boundary.json")},
            "owner_terminal": terminal, "owner_sha256": "f" * 64, "command": command}


def test_real_g2_update_keeps_logical_segment_and_optimizer_counters_distinct(tmp_path: Path):
    c, cref = contract(tmp_path)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    result = learner.execute(cref, stage(tmp_path, cref, [binding], None, "first"))
    assert result["logical_generation"] == 2
    assert result["segment"] == 1
    assert result["initial_actor_step"] == 9000
    assert result["bank_identities"][0]["run_uuid"] != result["run_uuid"]
    assert result["bank_identities"][0]["source_revision"] != result["source_revision"]
    assert result["step_start"] == 0
    assert result["step"] == result["applied_updates"] == 2
    assert result["drawn_rows"] == result["applied_views"] == 4
    assert {row[0] for row in result["membership"]} == {2}
    journal = [json.loads(line) for line in (tmp_path / "first/draws.jsonl").read_text().splitlines()]
    assert len(journal) == 2
    assert all(row["generation"] == 2 for batch in journal for row in batch["rows"])
    state = torch.load(result["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert state["step"] == 2
    assert state[learner.CHECKPOINT_KEY] == learner.checkpoint_identity(c, cref, 1)
    assert state["fresh_rl_donor_provenance"]["sha256"] == c["initial_model"]["sha256"]
    _, root = core.read_ref(binding["root_acceptance"])
    _, candidate = core.read_ref(root["candidate"])
    assert candidate["schema"] == "CORE_TRAIN_GENERATION_DATA_CANDIDATE_V1"
    assert candidate["learning_admitted"] is False


def test_adjacent_resume_matches_direct_checkpoint_continuation_and_five_rng_streams(tmp_path: Path):
    from scripts.offline_epoch_resume import _restore_rng, _rng_state

    c, cref = contract(tmp_path)
    first_bank = bank(tmp_path, c, 2, c["initial_model"], 9000)
    first = learner.execute(cref, stage(tmp_path, cref, [first_bank], None, "first"))
    second_bank = bank(tmp_path, c, 3, first["checkpoint"], first["step"])
    banks = [first_bank, second_bank]
    second_stage_ref = stage(tmp_path, cref, banks, parent(tmp_path, cref, first), "second")
    _, second_stage = core.read_ref(second_stage_ref)
    generations, _, _ = learner.metadata(c, second_stage, cref)
    reference = core.make_trainer(learner.settings(c), tmp_path / "reference")
    reference.load(Path(first["checkpoint"]["path"]), exact_resume=True)
    rng = np.random.default_rng()
    replay = core.load_replay(learner.settings(c), generations, rng, None, first_generation=2)
    _restore_rng(first["rng"], rng, cast(Any, np.random))
    reference.train_steps(replay, batch_size=c["batch_size"], steps=2)
    expected = {"model": copy.deepcopy(reference.model.state_dict()), "opt": copy.deepcopy(reference.opt.state_dict()),
                "scheduler": copy.deepcopy(reference._scheduler.state_dict()), "zclip": copy.deepcopy(reference.zclip_state_dict())}
    expected_rng = _rng_state(rng, cast(Any, np.random))
    reference.writer.close()
    second = learner.execute(cref, second_stage_ref)
    saved = torch.load(second["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key, value in expected.items():
        require_equal(saved[key], value, key)
    assert second["rng"] == expected_rng
    assert second["step_start"] == 2
    assert second["step"] == 4
    assert second["logical_generation"] == 3
    assert second["segment"] == 2
    assert {item[0] for item in second["membership"]} == {2, 3}


@pytest.mark.parametrize("corruption", ["network", "game_id", "ply_index", "duplicate_ply", "illegal"])
def test_real_payload_refusals_precede_model_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str):
    c, cref = contract(tmp_path)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000, corruption=corruption)
    called = []
    monkeypatch.setattr(core, "make_trainer", lambda *_args: called.append(True))
    with pytest.raises(ValueError, match=r"identity|integer|duplicate|illegal"):
        learner.execute(cref, stage(tmp_path, cref, [binding], None, "bad"))
    assert called == []
    assert not (tmp_path / "bad/boundary.json").exists()


def test_frozen_bank_view_is_deeply_thawed_for_real_inventory_helper(tmp_path: Path):
    c, _ = contract(tmp_path)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    view = acceptance.validate_accepted_generation(binding["root_acceptance"], binding["independent_review"], **c["bank_limits"])
    assert isinstance(view.games[0]["shard"]["files"], tuple)
    games = learner.plain(view.games)
    assert isinstance(games[0]["shard"]["files"], list)
    core.verify_inventory(games[0]["shard"])
    assert isinstance(view.games[0]["shard"]["files"], tuple)
    rng = np.random.default_rng(11)
    before = copy.deepcopy(rng.bit_generator.state)
    replay = core.load_replay(learner.settings(c), [{"generation": 2, "games": games}], rng, None, first_generation=2)
    assert replay.rng is rng
    assert rng.bit_generator.state == before
    assert {item[0] for item in replay.membership} == {2}
    with pytest.raises(ValueError, match="consecutive"):
        core.load_replay(learner.settings(c), [{"generation": 2, "games": games}], rng, None)


@pytest.mark.parametrize("fault", ["gap", "repeat", "initial_actor", "old_contract", "missing_review"])
def test_new_family_contract_and_bank_seams_fail_closed(tmp_path: Path, fault: str):
    c, cref = contract(tmp_path)
    binding = bank(tmp_path, c, 4 if fault == "gap" else 2, c["initial_model"], 8999 if fault == "initial_actor" else 9000)
    banks = [binding]
    if fault == "repeat":
        banks.append(binding)
    elif fault == "missing_review":
        del binding["independent_review"]
    if fault == "old_contract":
        c["schema"] = "core_finite_learning_v1"
        cref = frozen(tmp_path / "wrong-family.json", c)
    with pytest.raises(ValueError, match=r"logical|actor provenance|contract|binding"):
        learner.execute(cref, stage(tmp_path, cref, banks, None, "refused"))
    assert not (tmp_path / "refused/boundary.json").exists()


@pytest.mark.parametrize("fault", ["failed_owner", "wrong_source", "old_family", "bad_command", "bad_step", "pending"])
def test_parent_refusals_precede_model_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str):
    c, cref = contract(tmp_path)
    first_bank = bank(tmp_path, c, 2, c["initial_model"], 9000)
    first = learner.execute(cref, stage(tmp_path, cref, [first_bank], None, "first"))
    second_bank = bank(tmp_path, c, 3, first["checkpoint"], first["step"])
    prior = parent(tmp_path, cref, first)
    if fault == "failed_owner":
        _, terminal = core.read_ref(prior["owner_terminal"])
        terminal["status"] = "FAILED"
        prior["owner_terminal"] = frozen(tmp_path / "failed-terminal.json", terminal)
    elif fault == "bad_command":
        prior["command"] = ["wrong-public-entry"]
    else:
        path = Path(prior["boundary"]["path"])
        path.parent.chmod(0o755)
        if fault == "pending":
            frozen(path.parent / "boundary.pending.json", first)
        else:
            changed = dict(first)
            if fault == "wrong_source":
                changed["source_revision"] = "f" * 40
            elif fault == "old_family":
                changed["status"] = "COMPLETED_CORE_GENERATION_V1"
            else:
                changed["step"] += 1
            path.unlink()
            prior["boundary"] = frozen(path, changed)
        path.parent.chmod(0o555)
    called = []
    monkeypatch.setattr(core, "make_trainer", lambda *_args: called.append(True))
    with pytest.raises(ValueError, match="parent"):
        learner.execute(cref, stage(tmp_path, cref, [first_bank, second_bank], prior, "refused"))
    assert called == []


def test_legacy_entry_does_not_accept_new_family_contract(tmp_path: Path):
    c, cref = contract(tmp_path)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    with pytest.raises(ValueError, match="unknown or missing contract fields"):
        core.execute(cref, stage(tmp_path, cref, [binding], None, "legacy-refused"))
    assert not (tmp_path / "legacy-refused").exists()


def test_actual_loader_first_generation_fifo_and_rng_defaults(tmp_path: Path):
    c, _ = contract(tmp_path)
    generations = []
    for logical in range(2, 7):
        binding = bank(tmp_path, c, logical, c["initial_model"], 9000)
        view = acceptance.validate_accepted_generation(binding["root_acceptance"], binding["independent_review"], **c["bank_limits"])
        generations.append({"generation": logical, "games": learner.plain(view.games)})
    rng = np.random.default_rng(91)
    before = copy.deepcopy(rng.bit_generator.state)
    replay = core.load_replay(learner.settings(c), generations, rng, None, first_generation=2)
    assert {item[0] for item in replay.membership} == {3, 4, 5, 6}
    assert len(replay) == 16
    assert replay.rng is rng
    assert rng.bit_generator.state == before
    # The unchanged default still admits an actual generation1 bank.
    one_binding = bank(tmp_path, c, 1, c["initial_model"], 9000)
    one_view = acceptance.validate_accepted_generation(one_binding["root_acceptance"], one_binding["independent_review"], **c["bank_limits"])
    one = {"generation": one_view.identity["generation"], "games": learner.plain(one_view.games)}
    legacy = core.load_replay(learner.settings(c), [one], rng, None)
    assert {item[0] for item in legacy.membership} == {1}
    assert rng.bit_generator.state == before
