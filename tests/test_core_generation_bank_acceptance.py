"""Synthetic postclosure JSON graphs; no native, model, replay or audit execution."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts import core_generation_bank_acceptance as acceptance


def graph(tmp_path: Path, *, fault: str = "", game_ids: tuple[int, int] = (0, 2**63 - 1)) -> tuple[dict[str, str], dict[str, str], Path]:
    def write(name: str, value: dict[str, Any]) -> dict[str, str]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(raw)
        path.chmod(0o444)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    run, collection, revision = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222", "a" * 40
    source = {"revision": revision, "checkout": "/historical-auditor", "source_files": dict.fromkeys(acceptance.SOURCE_FILES, "b" * 64)}
    spec: dict[str, Any] = {"schema": "core_generation_data_spec_v1", "run_uuid": run, "collection_uuid": collection,
        "generation": 2, "source_revision": revision, "source_files": source["source_files"], "architecture": {"schema": 20, "planes": 175},
        "actor_sha256": "c" * 64, "actor_checkpoint_step": 17, "games_per_generation": 1, "max_games_per_generation": 3}
    if fault == "recipe":
        spec["lr"] = 0.001
    elif fault == "source_pin":
        spec["source_files"] = {}
    elif fault == "bool_generation":
        spec["generation"] = True
    spec_ref = write("spec.json", spec)
    actor: dict[str, Any] = {"sha256": "c" * 64, "path": "/NEVER-OPEN/model.pt", "expected_checkpoint_step": 17,
                             "architecture": {"schema": 20, "planes": 175}}
    if fault == "actor_step":
        actor["expected_checkpoint_step"] = 18
    producer: dict[str, Any] = {"schema": "CORE_FINITE_GENERATION_V1", "run_uuid": run, "collection_uuid": collection,
        "generation": 2, "source": {"revision": revision, "checkout": "/historical-auditor", "sha256": dict.fromkeys(acceptance.SOURCE_FILES, "b" * 64)},
        "actor": actor, "attempts": 3, "output_root": str(tmp_path / "producer")}
    if fault in {"collection", "producer_source", "architecture"}:
        if fault == "collection":
            producer["collection_uuid"] = run
        elif fault == "producer_source":
            producer["source"]["revision"] = "f" * 40
        else:
            actor["architecture"] = {"planes": 174}
    producer_ref = write("producer.json", producer)
    serving: dict[str, Any] = {"actor_sha256": "c" * 64, "pre_state_sha256": "d" * 64,
        "post_state_sha256": "d" * 64, "checkpoint_model_state_sha256": "d" * 64, "state_tensors": 494,
        "model_build_count": 1, "checkpoint_step": 17, "logical_generation": 2, "forwards": {"calls": 5}}
    if fault == "serving":
        serving["post_state_sha256"] = "e" * 64
    serving_ref = write("producer/actor-serving.json", serving)
    owner_code = {"path": "/NEVER-OPEN/owner.py", "sha256": "e" * 64}
    protocol: dict[str, Any] = {"schema": "CORE_FINITE_DATA_ADMISSION_V1", "data_spec": spec_ref,
        "producer_protocol": producer_ref, "producer_readout": {"path": "/NEVER-OPEN/readout.json", "sha256": "1" * 64},
        "producer_inventory": {"path": "/NEVER-OPEN/inventory.json", "sha256": "2" * 64},
        "producer_owner": {"terminal": {"path": "/NEVER-OPEN/terminal.json", "sha256": "3" * 64},
                           "owner_sha256": "4" * 64, "command": ["historical producer command"]},
        "gates": {"attempts": 3, "min_resolved_games": 1, "min_total_rows": 4},
        "runtime": {"owner": owner_code, "seconds": 300, "inner_seconds": 270, "cpu_affinity": [20, 22], "python": "/python"},
        "output_root": str(tmp_path / "audit"), "limits": dict.fromkeys(("max_metadata_bytes", "max_inventory_files", "max_inventory_bytes",
            "max_raw_decoded_bytes_per_game", "max_shard_decoded_bytes_per_game", "max_total_decoded_bytes"), 10000)}
    if fault == "protocol":
        protocol["schema"] = "CORE_FINITE_ADMISSION_V1"
    elif fault == "row_gate":
        protocol["gates"]["min_total_rows"] = 5
    protocol_ref = write("protocol.json", protocol)
    common = {"data_spec": spec_ref, "audit_protocol": protocol_ref, "collection_uuid": collection, "logical_generation": 2,
        "actor_checkpoint_step": 17, "candidate_only": True, "learning_admitted": False, "root_postclosure_promotion_required": True}
    audited: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    for i, kind in enumerate(acceptance.CLASSES):
        if i == 2:
            audited.append({"eligible": False, "physical_game_id": collection + ":2", "scientific_class": kind,
                            "rows": 0, "proof_sha256": "5" * 64})
            continue
        item: dict[str, Any] = {"eligible": True, "run_uuid": run, "collection_uuid": collection, "generation": 2,
            "source_revision": revision, "actor_sha256": "c" * 64, "served_state_sha256": "d" * 64,
            "physical_game_id": collection + ":" + str(i), "bank": "train", "game_id": game_ids[i], "rows": 2,
            "scientific_class": kind, "label_class": "NATURAL" if i == 0 else "TB_ADJUDICATED",
            "shard": {"path": "/NEVER-OPEN/shard" + str(i), "files": [{"path": "model.npy", "bytes": 10, "sha256": "6" * 64}]},
            "evidence": {key: {"path": "/NEVER-OPEN/" + key + ".json", "sha256": "7" * 64}
                         for key in ("provenance", "validation", "outcome_proof", "raw_metadata")},
            "shard_inventory_sha256": "8" * 64, "proof_sha256": "9" * 64}
        item["shard_inventory_sha256"] = hashlib.sha256(json.dumps(item["shard"]["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if fault == "inventory" and i == 0:
            item["shard_inventory_sha256"] = "0" * 64
        if fault == "duplicate_game" and i == 1:
            item["game_id"] = game_ids[0]
        invalid_ids: dict[str, Any] = {"id_string": "0", "id_bool": False, "id_negative": -1,
                                       "id_range": 2**63, "id_float": 0.0}
        if i == 0 and fault in invalid_ids:
            item["game_id"] = invalid_ids[fault]
        wrapper = {key: item[key] for key in ("run_uuid", "generation", "source_revision", "actor_sha256", "served_state_sha256",
            "physical_game_id", "game_id", "bank", "rows", "label_class", "evidence", "shard_inventory_sha256")}
        wrapper.update(common, status="VERIFIED_RESOLVED_CORE_DATA_GAME_CANDIDATE_V1", resolved=True,
                       full_history_verified=True, raw_targets_verified=True)
        if fault == "wrapper" and i == 0:
            wrapper["rows"] = 3
        elif fault == "wrapper_alias" and i == 0:
            wrapper["status"] = "VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1"
        wrapper_ref = write(f"audit/game-{i}.json", wrapper)
        games.append({key: item[key] for key in ("physical_game_id", "game_id", "bank", "rows", "label_class", "shard")} |
                     {"producer_proof": wrapper_ref})
        audited.append(item)
    if fault == "missing_attempt":
        audited.pop()
    if fault == "missing_eligible":
        games.pop()
    candidate = common | {"schema": "CORE_TRAIN_GENERATION_DATA_CANDIDATE_V1", "run_uuid": run, "source_revision": revision,
        "generation": 2, "actor_sha256": "c" * 64, "actor_serving": serving_ref, "games": games}
    if fault == "candidate_alias":
        candidate["schema"] = "ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1"
    elif fault == "admitted":
        candidate["learning_admitted"] = True
    candidate_ref = write("audit/generation.json", candidate)
    readout = common | {"status": "AUDITED_CORE_DATA_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1",
        "generation_manifest": candidate_ref, "attempts": 3, "audited": audited, "eligible_games": 2, "eligible_rows": 4,
        "class_counts": dict.fromkeys(acceptance.CLASSES, 1), "decoded_bytes": 100,
        "models_constructed": 0, "forwards": 0, "optimizer_updates": 0,
        "side_effect_count_scope": "source call graph; not instrumented constructor/forward counters",
        "policy_limit": "original F32 observation plus legal/strictTB recheck; no independent Gumbel replay"}
    if fault == "totals":
        readout["eligible_rows"] = 5
    readout_ref = write("audit/readout.json", readout)
    command = ["/python", "-B", "-m", "scripts.core_generation_data_admit", "--protocol", protocol_ref["path"],
               "--protocol-sha256", protocol_ref["sha256"]]
    if fault == "command":
        command[-1] = "0" * 64
    terminal: dict[str, Any] = {"status": "PASS", "returncode": 0, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED",
        "resource": "cpu", "seconds": 300, "wall_seconds": 1.0, "supervisor_sha256": "e" * 64, "command": command,
        "source_revision": "f" * 40, "cwd": "/separate-owner-checkout", "known_processes": {"123": 456}, "supervisor_pid": 122}
    if fault == "failed":
        terminal["status"] = "FAILED"
    elif fault == "gpu":
        terminal["resource"] = "gpu"
    elif fault == "unclosed":
        terminal["owned_processes_closed"] = False
    terminal_ref = write("terminal.json", terminal)
    admission = {"status": acceptance.ADMISSION_STATUS, "audit_protocol": protocol_ref, "audit_source": source,
        "command": command, "owner": owner_code, "resource": "cpu", "seconds": 300}
    admission_ref = write("admission.json", admission)
    closure: dict[str, Any] = {"supported_eof": True, "exit_code": 0, "supervisor_pid": 122, "supervisor_closed": True,
                              "owned_births_closed": True, "known_processes": {"123": 456}}
    if fault in {"eof", "supervisor", "births"}:
        closure[{"eof": "supported_eof", "supervisor": "supervisor_closed", "births": "owned_births_closed"}[fault]] = False
    root = {"status": acceptance.ROOT_STATUS, "audit_protocol": protocol_ref, "data_spec": spec_ref, "readout": readout_ref,
        "candidate": candidate_ref, "audit_source": source, "learning_admitted": False,
        "audit_owner": {"terminal": terminal_ref, "admission": admission_ref, "command": command,
                        "source_revision": "f" * 40, "cwd": "/separate-owner-checkout", "closure": closure}}
    root_ref = write("root.json", root)
    review = {"status": acceptance.REVIEW_STATUS, "root_acceptance": root_ref, "findings": [],
              **{key: root[key] for key in ("audit_protocol", "data_spec", "readout", "candidate")}}
    if fault == "review":
        review["findings"] = ["unresolved"]
    elif fault == "wrong_root":
        review["root_acceptance"] = spec_ref
    review_ref = write("review.json", review)
    if fault == "failure_marker":
        write("audit.failed.json", {"status": "FAILED_CORE_DATA_ADMISSION_V1"})
    return root_ref, review_ref, Path(candidate_ref["path"])


def validate(root: dict[str, str], review: dict[str, str], **limits: int) -> acceptance.AcceptedGeneration:
    return acceptance.validate_accepted_generation(root, review, **({"max_file_bytes": 1 << 20,
        "max_total_bytes": 4 << 20, "max_games": 3} | limits))


def test_complete_postclosure_view_keeps_generation_and_all_eligible_games(tmp_path: Path):
    root, review, candidate = graph(tmp_path)
    before = candidate.read_bytes()
    view = validate(root, review)
    assert view.identity["generation"] == 2
    assert len(view.games) == 2  # Minimum1 does not filter out the second eligible game.
    assert tuple(game["game_id"] for game in view.games) == (0, 2**63 - 1)
    assert view.evidence["eligible_rows"] == 4
    assert view.evidence["learning_admitted"] is False
    assert view.evidence["audit_owner"]["source_revision"] != view.identity["source_revision"]
    assert candidate.read_bytes() == before
    assert json.loads(before)["schema"] == "CORE_TRAIN_GENERATION_DATA_CANDIDATE_V1"
    with pytest.raises(TypeError, match="does not support item assignment"):
        cast(Any, view.games[0])["rows"] = 999


@pytest.mark.parametrize("fault", ["recipe", "source_pin", "bool_generation", "actor_step", "collection", "producer_source",
    "architecture", "serving", "inventory", "protocol", "row_gate", "duplicate_game", "wrapper", "wrapper_alias", "missing_attempt",
    "missing_eligible", "candidate_alias", "admitted", "totals", "command", "failed", "gpu", "unclosed", "eof", "supervisor",
    "births", "review", "wrong_root", "failure_marker", "id_string", "id_bool", "id_negative", "id_range", "id_float"])
def test_rehashed_but_inconsistent_graph_is_refused(tmp_path: Path, fault: str):
    root, review, _ = graph(tmp_path, fault=fault)
    expected = {
        "recipe": "invalid data spec", "source_pin": "missing historical source pins", "bool_generation": "invalid generation",
        "actor_step": "producer identity", "collection": "producer identity", "producer_source": "producer identity",
        "architecture": "producer identity", "serving": "serving evidence", "inventory": "inventory descriptor digest",
        "protocol": "invalid data audit protocol", "row_gate": "row gate", "duplicate_game": "game membership",
        "wrapper": "wrapper membership", "wrapper_alias": "wrapper verification", "missing_attempt": "incomplete audit membership",
        "missing_eligible": "eligible/class counts", "candidate_alias": "candidate schema", "admitted": "candidate authority",
        "totals": "row gate", "command": "audit command", "failed": "closed CPU PASS", "gpu": "closed CPU PASS",
        "unclosed": "closed CPU PASS", "eof": "root closure", "supervisor": "root closure", "births": "root closure",
        "review": "independent acceptance", "wrong_root": "independent acceptance", "failure_marker": "failed audit output",
        "id_string": "invalid game ID", "id_bool": "invalid game ID", "id_negative": "invalid game ID",
        "id_range": "invalid game ID", "id_float": "invalid game ID",
    }
    with pytest.raises(ValueError, match=expected[fault]):
        validate(root, review)


@pytest.mark.parametrize("limits", [{"max_file_bytes": 1}, {"max_total_bytes": 1}, {"max_games": 2},
                                   {"max_games": True}, {"max_file_bytes": 129 << 20}])
def test_reader_and_membership_bounds_are_explicit(tmp_path: Path, limits: dict[str, int]):
    root, review, _ = graph(tmp_path)
    with pytest.raises(ValueError, match=r"limit|attempt/gate"):
        validate(root, review, **limits)


def test_writable_or_replaced_receipt_is_refused(tmp_path: Path):
    root, review, _ = graph(tmp_path)
    Path(review["path"]).chmod(0o644)
    with pytest.raises(ValueError, match="readonly"):
        validate(root, review)
    Path(review["path"]).chmod(0o444)
    with pytest.raises(ValueError, match="digest"):
        validate(root, {"path": review["path"], "sha256": "0" * 64})


def test_symlink_and_missing_receipt_refused(tmp_path: Path):
    root, review, _ = graph(tmp_path)
    alias = tmp_path / "alias.json"
    alias.symlink_to(review["path"])
    with pytest.raises(ValueError, match="noncanonical"):
        validate(root, {"path": str(alias), "sha256": review["sha256"]})
    Path(review["path"]).unlink()
    with pytest.raises(FileNotFoundError, match="No such file"):
        validate(root, review)


def test_final_custody_refuses_metadata_changed_after_join(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, review, candidate = graph(tmp_path)
    finish = acceptance._Metadata.finish

    def replace_then_finish(reader: acceptance._Metadata) -> None:
        candidate.chmod(0o644)
        candidate.write_bytes(b"{}")
        candidate.chmod(0o444)
        finish(reader)

    monkeypatch.setattr(acceptance._Metadata, "finish", replace_then_finish)
    with pytest.raises(ValueError, match="custody changed"):
        validate(root, review)


@pytest.mark.parametrize("raw", [b'{"status":1,"status":2}', b'{"value":NaN}', b'{"value":1e999}'])
def test_invalid_json_constants_and_duplicate_keys_refused(tmp_path: Path, raw: bytes):
    root, review, _ = graph(tmp_path)
    path = Path(root["path"])
    path.chmod(0o644)
    path.write_bytes(raw)
    path.chmod(0o444)
    root["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match=r"duplicate|nonfinite"):
        validate(root, review)


def test_actual_emitter_game_id_representation_is_accepted(tmp_path: Path):
    # Execute only this stdlib-only function's actual source, never import selfplay.
    source = Path(__file__).resolve().parents[1] / "chess_anti_engine/selfplay/finalize.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_stable_game_id")
    namespace: dict[str, Any] = {"hashlib": hashlib}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    game_id = namespace["_stable_game_id"](start_fen="synthetic-start", opening_source="synthetic",
        move_trace="synthetic-trace", result="1/2-1/2", total_plies_played=0)
    assert type(game_id) is int
    assert 0 <= game_id < 2**63
    root, review, _ = graph(tmp_path, game_ids=(game_id, (game_id + 1) % 2**63))
    view = validate(root, review)
    assert view.games[0]["game_id"] == game_id
