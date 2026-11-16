"""Fictional metadata only: no actual GPU run, saved payload, bank or authority.

The shape follows the versioned passive prover. Numeric model facts are receipt
fixtures, not a full-model reconstruction or new numerical qualification.
"""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

from scripts import core_generation_holdout_ruler as ruler

HISTORICAL_LEARNER = "b" * 40
OWNER_SOURCE = "f" * 40


def seal(path: Path, body: Any) -> dict[str, str]:
    raw = ruler.canonical(body) + b"\n"
    path.write_bytes(raw)
    path.chmod(0o444)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def gpu_graph(tmp_path: Path, fault: str = "", *, arm: str = "A", threaded: bool = False) -> tuple[dict[str, Any], ruler.bank.AcceptedGeneration]:
    """Seal every changed edge, so refusals exercise joins, not stale hashes."""
    architecture: dict[str, Any] = {"_schema_version": 20, "input_extra_features": "v2_threats", "fixture": "metadata-only full494 receipt"}
    initial_model = {"path": "/fiction/original-model.pt", "sha256": "a" * 64}
    anchor: dict[str, Any] = {"schema": "frozen_reference_policy_anchor_v1", "direction": "reference_to_student_forward_KL",
              "mask": "original_network_turn_has_policy_with_explicit_legal_mask_v1", "coefficient": 0.0, "reference": None}
    if arm == "C":
        anchor["coefficient"] = 0.1
        anchor["reference"] = {"checkpoint": initial_model, "architecture": architecture,
                               "checkpoint_step": 680276, "historical_source_revision": None}
    rate = 7.5e-7 if arm == "B" else 3e-6
    contract: dict[str, Any] = {"schema": ruler.ANCHOR_KEY, "run_uuid": str("ABC".index(arm) + 1) * 8 + "-1111-4111-8111-111111111111",
                "source_revision": HISTORICAL_LEARNER, "architecture": architecture, "first_generation": 1,
                "initial_model": initial_model, "initial_actor_step": 680276, "batch_size": 32,
                "updates_per_segment": [250], "policy_anchor": anchor, "seed": 2026111673, "mirror_prob": 0.5,
                "lr": {"peak": rate, "floor": rate, "warmup_start": rate, "warmup_steps": 0, "first_period": 2048}}
    cref = seal(tmp_path / "contract.json", contract)
    bank_binding = {"root_acceptance": {"path": "/fiction/bank-root.json", "sha256": "1" * 64},
                    "independent_review": {"path": "/fiction/bank-review.json", "sha256": "2" * 64}}
    output = tmp_path / "learner"
    sref = seal(tmp_path / "stage.json", {"schema": "core_generation_bank_policy_anchor_stage_v1", "contract_sha256": cref["sha256"],
        "segment": 1, "parent": None, "banks": [bank_binding], "output": str(output)})
    games: list[dict[str, Any]] = []
    for index in range(160):
        files = [{"path": "fixture.arrays", "bytes": 1, "sha256": hashlib.sha256(str(index).encode()).hexdigest()}]
        games.append({"physical_game_id": "fiction:" + str(index), "rows": 50,
                      "shard": {"path": "/fiction/shard-" + str(index), "files": files}})
    membership = [[1, hashlib.sha256(ruler.canonical(game["shard"]["files"])).hexdigest(), 50] for game in games]
    rng: dict[str, Any] = {"numpy": {"fixture": 1}, "mask": [], "torch_cpu": [], "torch_cuda": [[]], "python": []}
    backend: dict[str, Any] = {"fixture_backend": "eager BF16 source observation"}
    if threaded:
        backend.update(torch_threads=1 if fault == "thread_backend_intra" else 2,
                       interop_threads=1 if fault == "thread_backend_interop" else 2)
    checkpoint = {"path": str(output / "learner.pt"), "sha256": "c" * 64}
    boundary: dict[str, Any] = {"status": ruler.BOUNDARY, "contract_sha256": cref["sha256"], "stage": sref, "checkpoint": checkpoint,
        "source_revision": HISTORICAL_LEARNER, "run_uuid": contract["run_uuid"], "logical_generation": 1, "segment": 1,
        "initial_model": initial_model, "initial_actor_step": 680276, "parent": None, "step_start": 0,
        "step": 250, "applied_updates": 250, "applied_views": 8000, "drawn_rows": 8000, "policy_anchor": anchor,
        "membership": membership, "accepted_banks": [bank_binding], "bank_identities": [{"run_uuid": "44444444-4444-4444-8444-444444444444",
            "collection_uuid": "22222222-2222-4222-8222-222222222222", "generation": 1, "source_revision": "d" * 40,
            "architecture": architecture, "actor_sha256": initial_model["sha256"], "actor_checkpoint_step": 680276}],
        "rng": rng, "backend": backend, "draws_sha256": "d" * 64}
    bref = seal(tmp_path / "boundary.json", boundary)
    suffix = ["-m", "scripts.core_generation_bank_policy_anchor", "--contract", cref["path"], "--contract-sha256", cref["sha256"],
              "--stage", sref["path"], "--stage-sha256", sref["sha256"]]
    helper = {"path": "/fiction/source-prover/observe.py",
              "sha256": ruler.GPU_PROOF_THREAD2_HELPER_SHA256 if threaded else ruler.GPU_PROOF_HELPER_SHA256}
    if threaded and fault == "thread_helper_unknown":
        helper["sha256"] = "f" * 64
    if fault == "helper":
        helper["sha256"] = "0" * 64
    pins = seal(tmp_path / "source-pins.json", {"source_revision": "e" * 40 if fault == "historical_source" else HISTORICAL_LEARNER,
        "source_checkout": "/fiction/historical-learner", "files": dict.fromkeys((f"fixture-{i}.py" for i in range(219)), "e" * 64)})
    review_helper = {**helper, "sha256": ruler.GPU_PROOF_HELPER_SHA256} if threaded and fault == "thread_review_helper" else helper
    review_pins = {**pins, "sha256": "0" * 64} if threaded and fault == "thread_review_pins" else pins
    source_review = seal(tmp_path / "source-review.json", {"status": ruler.GPU_PROOF_SOURCE_REVIEW, "helper": review_helper,
        "source_pins": review_pins, "findings": ["unresolved"] if fault == "source_review" or (threaded and fault == "thread_review_findings") else []})
    proof_path = tmp_path / "gpu-proof.json"
    plan = seal(tmp_path / "observer-plan.json", {"schema": "SCIENTIFIC_ANCHORED_LEARNER_PASSIVE_PROOF_V1",
        "status": "BOUND_SCIENTIFIC_LEARNER_PASSIVE_PROOF_NOT_DELIVERY_AUTHORITY_V1", "helper": helper, "source_review": source_review,
        "source_pins": pins, "contract": cref, "stage": sref, "public_suffix": suffix, "learner_output": str(output),
        "observation": str(proof_path), "N": 8001 if fault == "N" else 8000, "K": 250,
        "limits": {"outer_seconds": 900, "inner_seconds": 870, "checkpoint_file_bytes": 8 << 30,
                   "journal_bytes": 16 << 20, "aggregate_bytes": 4 << 20}})
    admission_path = tmp_path / "phase-admission.json"
    normalized = ["/fiction/python", "-I", "-B", helper["path"], "--observer-plan", plan["path"], "--observer-plan-sha256", plan["sha256"],
                  "--admission", str(admission_path), "--admission-sha256", "SELF_ADMISSION_SHA256", "--", *suffix]
    cap = 901 if fault == "cap" else 900
    admission: dict[str, Any] = {"status": "wrong" if fault == "admission_status" else ruler.GPU_PROOF_ADMISSION,
        "execution_authorized": True, "observer_plan": plan, "helper": helper, "source_review": source_review, "source_pins": pins,
        "normalized_child": copy.deepcopy(normalized), "resource": "gpu", "seconds": cap, "owner": {"sha256": "e" * 64}}
    if fault == "normalized":
        admission["normalized_child"][-1] = "wrong"
    aref = seal(admission_path, admission)
    command = list(normalized)
    command[command.index("--admission-sha256") + 1] = "0" * 64 if fault == "admission_sha" else aref["sha256"]
    if fault == "observer_ref":
        command[command.index("--observer-plan-sha256") + 1] = "0" * 64
    terminal: dict[str, Any] = {"status": "PASS", "returncode": 0, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED",
        "resource": "cpu" if fault == "resource" else "gpu", "seconds": cap, "wall_seconds": 2.0, "command": command,
        "source_revision": OWNER_SOURCE, "cwd": "/fiction/wrapper-owner", "supervisor_sha256": "e" * 64,
        "supervisor_pid": 2147483645, "known_processes": {"2147483646": 7}}
    tref = seal(tmp_path / "terminal.json", terminal)
    owner: dict[str, Any] = {"terminal": tref, "admission": aref, "command": command, "source_revision": OWNER_SOURCE, "cwd": "/fiction/wrapper-owner",
        "closure": {"supported_eof": True, "exit_code": 0, "supervisor_pid": 2147483645, "supervisor_closed": True,
                    "owned_births_closed": fault != "closure", "known_processes": terminal["known_processes"]}}
    rows: list[dict[str, Any]] = []
    for index, game in enumerate(games):
        count, unique = (0, 0) if index == 0 else (100, 30) if index == 1 else (50, 25)
        rows.append({"physical_game_id": game["physical_game_id"], "population_rows": 50,
                     "draws": count, "unique_rows": unique, "unique_fraction": unique / 50})
    unique_total = sum(row["unique_rows"] for row in rows)
    coverage: dict[str, Any] = {"population_rows": 8000, "batches": 250, "views": 8000, "views_per_population_row": 1.0,
        "unique_physical_rows_drawn": unique_total, "unique_fraction": unique_total / 8000, "games": 160, "zero_draw_games": 1,
        "per_game_aggregate_counts": rows, "journal_sha256": boundary["draws_sha256"], "sampling": "uniform with replacement"}
    proof: dict[str, Any] = {"status": ruler.GPU_PAYLOAD_STATUS, "contract": cref, "stage": sref, "boundary": bref, "checkpoint": checkpoint,
        "source_revision": HISTORICAL_LEARNER, "source_checkout": "/fiction/historical-learner", "architecture": architecture, "step": 250,
        "public_suffix": suffix, "observer_plan": plan, "helper": helper, "source_review": source_review, "source_pins": pins,
        "outer_seconds": 900, "absolute_inner_seconds": 870, "model_state_sha256": "3" * 64,
        "reconciled": dict.fromkeys(("model", "optimizer", "scheduler", "zclip", "counters", "rng", "journal", "membership", "policy_anchor"), True),
        "own_live_state_fingerprint": "4" * 64, "save_before_equals_after": True, "reference_immutable": True,
        "additional_final_CPU_checkpoint_decodes": 1, "optimizer_named_parameters": 438, "active_optimizer_slots": 438,
        "initial_model_state_sha256": "5" * 64, "initial": {"model": {"state_entries": 494, "input_planes": 175,
            "unique_parameter_numel": 61444448, "naive_state_numel": 77173088, "finite": True, "state_sha256": "5" * 64},
            "membership": membership, "backend": backend}, "actual_backend": backend,
        "five_rng_snapshot_sha256": hashlib.sha256(ruler.canonical(rng)).hexdigest(),
        "membership_sha256": hashlib.sha256(ruler.canonical(membership)).hexdigest(), "draw_coverage": coverage}
    if threaded:
        proof.update(Torch_intra=2, Torch_interop=2)
        if fault == "thread_missing_intra":
            del proof["Torch_intra"]
        elif fault == "thread_missing_interop":
            del proof["Torch_interop"]
        elif fault == "thread_intra":
            proof["Torch_intra"] = 3
        elif fault == "thread_interop":
            proof["Torch_interop"] = 3
        elif fault == "thread_bool_intra":
            proof["Torch_intra"] = True
        elif fault == "thread_float_interop":
            proof["Torch_interop"] = 2.0
    if fault == "cpu_status":
        proof["status"] = "CORE_BANK_LEARNER_SAVED_STATE_AUDIT_PENDING_ROOT_CLOSURE_V1"
    elif fault == "checkpoint":
        proof["checkpoint"] = {**checkpoint, "sha256": "0" * 64}
    elif fault in {"rng", "membership"}:
        proof["five_rng_snapshot_sha256" if fault == "rng" else "membership_sha256"] = "0" * 64
    elif fault == "journal":
        coverage["journal_sha256"] = "0" * 64
    elif fault == "views":
        coverage["views"] -= 1
    elif fault in {"save", "reference"}:
        proof["save_before_equals_after" if fault == "save" else "reference_immutable"] = False
    elif fault == "decode_count":
        proof["additional_final_CPU_checkpoint_decodes"] = 2
    elif fault == "full_shape":
        proof["initial"]["model"]["input_planes"] = 174
    elif fault == "drop_game":
        rows.pop()
    elif fault == "reorder_game":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "duplicate_game":
        rows[1] = copy.deepcopy(rows[0])
    elif fault == "population":
        rows[0]["population_rows"] = 51
    elif fault == "unique_bound":
        rows[0]["unique_rows"] = 1
    elif fault == "fraction":
        rows[1]["unique_fraction"] = 0.0
    elif fault == "total_unique":
        coverage["unique_physical_rows_drawn"] -= 1
    elif fault == "zero_games":
        coverage["zero_draw_games"] = 0
    audit: dict[str, Any] = {"schema": "wrong" if fault == "schema" else ruler.GPU_PAYLOAD_SCHEMA,
             "readout": seal(proof_path, proof), "owner": copy.deepcopy(owner)}
    if fault == "missing_schema":
        del audit["schema"]
    elif fault == "different_owner":
        audit["owner"]["source_revision"] = "0" * 40
    root: dict[str, Any] = {"status": ruler.DELIVERY_ROOT, "contract": cref, "stage": sref, "boundary": bref, "checkpoint": checkpoint,
            "source_revision": HISTORICAL_LEARNER, "source_checkout": "/fiction/historical-learner", "learner_owner": owner, "payload_audit": audit}
    rref = seal(tmp_path / "delivery-root.json", root)
    review: dict[str, Any] = {"status": ruler.DELIVERY_REVIEW, "root_acceptance": rref, "findings": [],
              **{key: root[key] for key in ("contract", "stage", "boundary", "checkpoint", "payload_audit")}}
    binding = {"root": rref, "independent": seal(tmp_path / "delivery-review.json", review)}
    train = ruler.bank.AcceptedGeneration(identity=boundary["bank_identities"][0], games=tuple(games), evidence={})
    return binding, train


def test_same_gpu_delivery_uses_actual_versioned_metadata_and_distinct_sources(tmp_path: Path) -> None:
    binding, train = gpu_graph(tmp_path)
    metadata = ruler.Metadata(1 << 20, 8 << 20)
    model = ruler.model_delivery(metadata, binding, "A")
    ruler.same_gpu_game_coverage(metadata, binding, model, train)
    assert model["step"] == 250
    assert model["tag"]["source_revision"] == HISTORICAL_LEARNER != OWNER_SOURCE
    assert model["contract"]["source_revision"] != ruler.subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ruler.REPO, env={"GIT_OPTIONAL_LOCKS": "0"}, text=True).strip()
    metadata.finish()


@pytest.mark.parametrize(("fault", "message"), [
    ("schema", "invalid saved-state audit binding"), ("missing_schema", "saved-state audit incomplete"),
    ("cpu_status", "same-GPU saved-state proof incomplete"), ("different_owner", "payload proof must share complete learner owner"),
    ("resource", "owner not exact closed PASS"), ("cap", "owner cap/wall differs"), ("closure", "closure evidence differs"),
    ("admission_status", "same-GPU proof admission differs"), ("normalized", "same-GPU normalized admission differs"),
    ("admission_sha", "same-GPU observer command differs"), ("observer_ref", "same-GPU observer command differs"),
    ("helper", "unsupported same-GPU prover"), ("source_review", "same-GPU prover source acceptance differs"),
    ("historical_source", "same-GPU historical source identity differs"), ("checkpoint", "same-GPU saved-state proof incomplete"),
    ("rng", "same-GPU RNG/membership/backend differs"), ("membership", "same-GPU RNG/membership/backend differs"),
    ("journal", "same-GPU draw totals differ"), ("views", "same-GPU draw totals differ"), ("N", "same-GPU draw totals differ"),
    ("save", "same-GPU live/final proof incomplete"), ("reference", "same-GPU live/final proof incomplete"),
    ("decode_count", "same-GPU live/final proof incomplete"), ("full_shape", "same-GPU full model facts differ"),
])
def test_rehashed_same_gpu_proof_refusals(tmp_path: Path, fault: str, message: str) -> None:
    binding, _ = gpu_graph(tmp_path, fault)
    with pytest.raises(ValueError, match=rf"^{re.escape(message)}$"):
        ruler.model_delivery(ruler.Metadata(1 << 20, 8 << 20), binding, "A")


@pytest.mark.parametrize(("fault", "message"), [
    ("drop_game", "same-GPU coverage game roster differs"), ("reorder_game", "same-GPU coverage physical game differs"),
    ("duplicate_game", "same-GPU coverage physical game differs"), ("population", "same-GPU coverage physical game differs"),
    ("unique_bound", "same-GPU per-game coverage differs"), ("fraction", "same-GPU per-game coverage differs"),
    ("total_unique", "same-GPU aggregate coverage differs"), ("zero_games", "same-GPU aggregate coverage differs"),
])
def test_rehashed_same_gpu_coverage_refusals_after_population(tmp_path: Path, fault: str, message: str) -> None:
    binding, train = gpu_graph(tmp_path, fault)
    metadata = ruler.Metadata(1 << 20, 8 << 20)
    model = ruler.model_delivery(metadata, binding, "A")
    with pytest.raises(ValueError, match=rf"^{re.escape(message)}$"):
        ruler.same_gpu_game_coverage(metadata, binding, model, train)


def test_normalized_gpu_admission_cannot_be_used_as_cpu_audit(tmp_path: Path) -> None:
    binding, _ = gpu_graph(tmp_path, "resource")
    metadata = ruler.Metadata(1 << 20, 8 << 20)
    root = metadata.read(binding["root"])
    suffix = metadata.read(root["payload_audit"]["readout"])["public_suffix"]
    with pytest.raises(ValueError, match=r"^same-GPU proof admission differs$"):
        ruler.closed_owner(metadata, root["learner_owner"], resource="cpu", suffix=suffix,
                           seconds=900, _same_gpu_proof=True)


@pytest.mark.parametrize("corrupt", [False, True])
def test_shared_bindings_apply_new_coverage_after_training_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool) -> None:
    # Inert pre-existing bank/asset credentials isolate the NEW public delivery
    # validator and its actual bindings call site. Existing upstream credentials
    # are inert; the NEW owner/proof/coverage predicates execute without mocks.
    models: dict[str, Any] = {}
    train: ruler.bank.AcceptedGeneration | None = None
    for arm in "ABC":
        folder = tmp_path / arm
        folder.mkdir()
        models[arm], train = gpu_graph(folder, "drop_game" if corrupt and arm == "C" else "", arm=arm)
    assert train is not None
    architecture = dict(train.identity["architecture"])
    view = ruler.hold.AcceptedHoldout(identity={"bank": "holdout", "architecture": architecture, "collection_uuid": "HOLD-fixture"},
        games=tuple({"physical_game_id": "hold:" + str(i), "rows": 50} for i in range(48)), evidence={})
    donor = {"checkpoint": {"path": "/fiction/original-model.pt", "sha256": "a" * 64}, "architecture": architecture, "step": 680276}
    models["A39"] = donor
    monkeypatch.setattr(ruler.hold, "validate_accepted_holdout", lambda *_args, **_kwargs: view)
    monkeypatch.setattr(ruler, "parent_map", lambda *_args: {game["physical_game_id"]: str(i) for i, game in enumerate(view.games)})
    monkeypatch.setattr(ruler, "asset_identity", lambda _m, bound: {**bound, "label": "A39", "tag": None})
    monkeypatch.setattr(ruler, "training_population", lambda _m, _model: train)
    protocol: dict[str, Any] = {"holdout": {"root": {}, "independent": {}}, "models": models,
                "limits": {"metadata_file_bytes": 1 << 20, "metadata_total_bytes": 8 << 20}}
    metadata = ruler.Metadata(1 << 20, 8 << 20)
    if corrupt:
        with pytest.raises(ValueError, match=r"^same-GPU coverage game roster differs$"):
            ruler.bindings(metadata, protocol)
    else:
        returned, _, delivered = ruler.bindings(metadata, protocol)
        assert returned is view
        assert set(delivered) == {"A", "B", "C", "A39"}
        metadata.finish()
