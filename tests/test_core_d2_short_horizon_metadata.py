"""Fictional metadata graphs; no model, journal, book, GPU or study execution.

Only the preexisting accepted-bank token is substituted. New owner, delivery,
prefix, retained provenance and per-game predicates execute without gate mocks.
Source trust anchors are replaced by their explicit sealed fixture counterparts.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts import core_generation_d2_metadata as d2
from scripts import core_generation_holdout_ruler as ruler
from tests.test_core_k0_saved_endpoint_recovery import no_clip_corrective_graph
from tests.test_core_same_gpu_payload_delivery import gpu_graph, seal


def read(ref: dict[str, str]) -> dict[str, Any]:
    return json.loads(Path(ref["path"]).read_bytes())


def short_graph(tmp: Path, monkeypatch: pytest.MonkeyPatch, label: str, fault: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    long_folder = tmp / "long"
    long_folder.mkdir()
    if label == "S0":
        long_binding, accepted = no_clip_corrective_graph(long_folder, monkeypatch)
    else:
        long_binding, accepted = gpu_graph(long_folder, arm="K1", d1=True,
            design_ref={"path": str(long_folder / "unused-design.json"), "sha256": ruler.D1_DESIGN_SHA256})
    long_root = read(long_binding["root"])
    long_contract, long_boundary = read(long_root["contract"]), read(long_root["boundary"])
    long_stage = read(long_root["stage"])
    train = ruler.bank.AcceptedGeneration(identity=accepted.identity, games=accepted.games, evidence={"eligible_rows": 17568})
    monkeypatch.setattr(ruler.bank, "validate_accepted_generation", lambda *_a, **_k: train)
    long_label = "L0" if label == "S0" else "L1"
    method = seal(tmp / "D1-method.json", {"study_uuid": "8045699e-4088-4a41-ae9c-4013d4a55469",
        "foundations": {"TRAIN_bank": {"root": long_stage["banks"][0]["root_acceptance"],
                                     "independent": long_stage["banks"][0]["independent_review"]}}})
    monkeypatch.setattr(d2, "D1_DESIGN", method)
    donor: dict[str, Any] = {"checkpoint": long_contract["initial_model"], "architecture": long_contract["architecture"], "step": 680276,
                             "model_state_sha256": "9796079cd73bef73ba2f5d5f2ad0779704498832e34a7c2a880f0d0b7f305229"}
    monkeypatch.setattr(d2, "LONG_BINDINGS", {**d2.LONG_BINDINGS, long_label: long_binding})
    files = dict.fromkeys((f"fixture-{i}.py" for i in range(219)), "e" * 64)
    pins = seal(tmp / "pins.json", {"source_revision": d2.LEARNER_SOURCE, "source_checkout": "/fiction/learner", "files": files})
    monkeypatch.setattr(d2, "LEARNER_PINS", pins)
    monkeypatch.setattr(d2, "TRAIN_BANK", long_stage["banks"][0])
    contract: dict[str, Any] = {**long_contract, "run_uuid": d2.RUN_UUIDS[label], "source_files": files,
        "updates_per_segment": [512], "games_per_generation": 160, "max_games_per_generation": 192, "replay_generations": 4,
        "runtime": d2.RUNTIME, "lr": d2.LR, "limits": {"max_rows": 17568, "max_decoded_window_bytes": 16 << 30},
        "bank_limits": {"max_file_bytes": 128 << 20, "max_total_bytes": 4 << 30, "max_games": 192}}
    if fault in {"511", "513", "4096"}:
        contract["updates_per_segment"] = [int(fault)]
    elif fault == "source":
        contract["source_revision"] = "f" * 40
    elif fault == "reference":
        contract["policy_anchor"] = {**contract["policy_anchor"], "reference": None if label == "S1" else donor}
    elif fault == "loss_recipe":
        contract["sf_wdl_frac"] = 1.0
    cref = seal(tmp / "contract.json", contract)
    monkeypatch.setattr(d2, "SHORT_CONTRACT_TEMPLATES", {**d2.SHORT_CONTRACT_TEMPLATES, label: cref})
    output = tmp / "student"
    stage = {"schema": "core_generation_bank_policy_anchor_stage_v1", "contract_sha256": cref["sha256"],
             "segment": 1, "parent": None, "banks": [d2.TRAIN_BANK], "output": str(output)}
    sref = seal(tmp / "stage.json", stage)
    boundary = {**long_boundary, "contract_sha256": cref["sha256"], "stage": sref,
                "checkpoint": {"path": str(output / "learner.pt"), "sha256": "c" * 64}, "source_revision": contract["source_revision"],
                "run_uuid": contract["run_uuid"], "step": 512, "applied_updates": 512, "applied_views": 16383 if fault == "views" else 16384,
                "drawn_rows": 16384, "draws_sha256": "9" * 64, "policy_anchor": contract["policy_anchor"]}
    bref = seal(tmp / "boundary.json", boundary)
    suffix = ["-m", "scripts.core_generation_bank_policy_anchor", "--contract", cref["path"], "--contract-sha256", cref["sha256"],
              "--stage", sref["path"], "--stage-sha256", sref["sha256"]]
    helper = {"path": "/fiction/D2/observe.py", "sha256": "0" * 64 if fault == "helper" else "8" * 64}
    monkeypatch.setattr(d2, "SHORT_HELPER_SHA256", "8" * 64)
    source_review = seal(tmp / "source-review.json", {"status": d2.SHORT_REVIEW_STATUS, "helper": helper, "source_pins": pins, "findings": []})
    monkeypatch.setattr(d2, "SHORT_SOURCE_REVIEW", source_review)
    proof_path = tmp / "proof.json"
    plan: dict[str, Any] = {"schema": "SCIENTIFIC_D2_SHORT_HORIZON_PASSIVE_PROOF_V1",
        "status": "BOUND_D2_SHORT_HORIZON_PASSIVE_PROOF_NOT_DELIVERY_AUTHORITY_V1", "study_uuid": "wrong" if fault == "study" else d2.STUDY,
        "arm": label, "design": d2.DESIGN, "helper": helper, "source_review": source_review, "source_pins": pins,
        "qualification": {"fixture": "historical accepted token"}, "assets": {"fixture": "donor"}, "runtime": {"fixture": "managed proof"},
        "owner": {"sha256": "e" * 64}, "stage_name": "fiction-D2", "contract": cref, "contract_template": cref, "stage": sref,
        "accepted_bank": d2.TRAIN_BANK, "train_protection": {"fixture": "already accepted"}, "N": 17568, "K": 512,
        "learner_output": str(output), "observation": str(proof_path), "long_comparator": long_binding,
        "limits": {"outer_seconds": 3600, "inner_seconds": 3570, "checkpoint_file_bytes": 8 << 30, "journal_bytes": 64 << 20, "aggregate_bytes": 4 << 20},
        "issued_utc": "fixture", "deadline_utc": "fixture", "public_suffix": suffix}
    pref = seal(tmp / "observer-plan.json", plan)
    ad_path = tmp / "admission.json"
    normalized = ["/fiction/python", "-I", "-B", helper["path"], "--observer-plan", pref["path"], "--observer-plan-sha256", pref["sha256"],
                  "--admission", str(ad_path), "--admission-sha256", "SELF_ADMISSION_SHA256", "--", *suffix]
    admission = seal(ad_path, {"status": d2.SHORT_ADMISSION, "execution_authorized": True, "resource": "gpu", "seconds": 3600,
        "observer_plan": pref, "helper": helper, "source_review": source_review, "source_pins": pins,
        "owner": {"sha256": "e" * 64}, "normalized_child": normalized})
    command = copy.deepcopy(normalized)
    command[command.index("--admission-sha256") + 1] = admission["sha256"]
    terminal = seal(tmp / "terminal.json", {"status": "PASS", "returncode": 0, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED",
        "resource": "gpu", "seconds": 3600, "wall_seconds": 2.0, "command": command, "source_revision": "f" * 40, "cwd": "/fiction/owner",
        "supervisor_sha256": "e" * 64, "supervisor_pid": 2147483645, "known_processes": {"2147483646": 1}})
    owner: dict[str, Any] = {"terminal": terminal, "admission": admission, "command": command, "source_revision": "f" * 40, "cwd": "/fiction/owner",
        "closure": {"supported_eof": True, "exit_code": 0, "supervisor_pid": 2147483645, "supervisor_closed": True,
                    "owned_births_closed": True, "known_processes": {"2147483646": 1}}}
    per_game: list[dict[str, Any]] = []
    for index, game in enumerate(train.games):
        count = 0 if index == 0 else 234 if index == 1 else 85
        unique = min(count, game["rows"])
        per_game.append({"physical_game_id": game["physical_game_id"], "population_rows": game["rows"], "draws": count,
                         "unique_rows": unique, "unique_fraction": unique / game["rows"]})
    unique = sum(row["unique_rows"] for row in per_game)
    coverage: dict[str, Any] = {"population_rows": 17568, "batches": 512, "views": 16384, "views_per_population_row": 16384 / 17568,
        "games": 192, "unique_physical_rows_drawn": unique, "unique_fraction": unique / 17568, "zero_draw_games": 1,
        "per_game_aggregate_counts": per_game, "journal_sha256": boundary["draws_sha256"]}
    if fault == "drop_game":
        per_game.pop()
    backend = {"torch_threads": 2, "interop_threads": 2}
    boundary["backend"] = backend
    bref = seal(tmp / "boundary-v2.json", boundary)
    stamp = {"st_dev": 1, "st_ino": 2, "st_size": 1024, "st_mtime_ns": 3, "st_ctime_ns": 4, "st_mode": 0o100444}
    old_journal = {"path": str(Path(long_root["checkpoint"]["path"]).parent / "draws.jsonl"), "sha256": long_boundary["draws_sha256"]}
    observed = {"stamp": stamp, "bytes": 1024, "sha256": old_journal["sha256"]}
    prefix: dict[str, Any] = {"schema": "D2_PAIRED_SAMPLER_PREFIX_COMPARISON_V1", "long_comparator": long_binding,
        "long_boundary": long_root["boundary"], "long_journal": old_journal,
        "short_journal": {"path": str(output / "draws.jsonl"), "sha256": boundary["draws_sha256"]}, "batches": 512, "rows_per_batch": 32,
        "views": 16384, "long_batches": 4096, "long_views": 131072, "population_rows": 17568, "population_games": 192,
        "membership_sha256": hashlib.sha256(ruler.canonical(boundary["membership"])).hexdigest(),
        "coverage_sha256": hashlib.sha256(ruler.canonical(coverage)).hexdigest(), "long_prefix_sha256": "9" * 64, "short_prefix_sha256": "9" * 64,
        "equal": fault != "prefix", "long_stream_passes": 3, "long_raw_read_bytes": 3072, "short_stream_passes": 1, "short_raw_read_bytes": 512,
        "long_pre": observed, "long_post": copy.deepcopy(observed), "long_final": copy.deepcopy(observed),
        "short_observed": {"stamp": {**stamp, "st_ino": 7, "st_size": 512}, "bytes": 512, "sha256": boundary["draws_sha256"]},
        "all_reads_metered": True, "claims": "Physical-row sampler prefix equality only; no mirror/full-RNG/arithmetic identity claim."}
    proof: dict[str, Any] = {"status": d2.SHORT_PENDING, "payload_schema": d2.SHORT_SCHEMA, "study_uuid": plan["study_uuid"], "arm": label,
        "contract": cref, "stage": sref, "boundary": bref, "checkpoint": boundary["checkpoint"], "source_revision": contract["source_revision"],
        "source_checkout": "/fiction/learner", "architecture": contract["architecture"], "step": 512, "public_suffix": suffix,
        "observer_plan": pref, "helper": helper, "source_review": source_review, "source_pins": pins, "outer_seconds": 3600,
        "absolute_inner_seconds": 3570, "model_state_sha256": "3" * 64,
        "reconciled": dict.fromkeys(("model", "optimizer", "scheduler", "zclip", "counters", "rng", "journal", "membership", "policy_anchor"), True),
        "save_before_equals_after": True, "reference_immutable": True, "additional_final_CPU_checkpoint_decodes": 1,
        "own_live_state_fingerprint": "4" * 64, "initial_model_state_sha256": donor["model_state_sha256"],
        "optimizer_named_parameters": 438, "active_optimizer_slots": 438,
        "initial": {"model": {"state_entries": 494, "input_planes": 175, "unique_parameter_numel": 61444448,
                               "naive_state_numel": 77173088, "finite": True, "state_sha256": donor["model_state_sha256"]},
                    "actual_student_device": "cuda:0", "actual_Trainer_use_amp": True, "membership": boundary["membership"], "backend": backend},
        "actual_backend": backend, "five_rng_snapshot_sha256": hashlib.sha256(ruler.canonical(boundary["rng"])).hexdigest(),
        "membership_sha256": hashlib.sha256(ruler.canonical(boundary["membership"])).hexdigest(), "Torch_intra": 2, "Torch_interop": 2,
        "draw_coverage": coverage, "draw_prefix_comparison": prefix, "additional_wrapper_read_bytes": 4096}
    if fault == "integer_flag":
        proof["reconciled"]["zclip"] = 1
    audit: dict[str, Any] = {"schema": d2.SHORT_SCHEMA, "readout": seal(proof_path, proof), "owner": copy.deepcopy(owner)}
    if fault == "different_owner":
        audit["owner"]["source_revision"] = "0" * 40
    root: dict[str, Any] = {"status": ruler.K0_NO_CLIP_ROOT if fault == "recovery" else ruler.DELIVERY_ROOT,
        "contract": cref, "stage": sref, "boundary": bref, "checkpoint": boundary["checkpoint"], "source_revision": contract["source_revision"],
        "source_checkout": "/fiction/learner", "learner_owner": owner, "payload_audit": audit}
    rref = seal(tmp / "Root.json", root)
    independent: dict[str, Any] = {"status": ruler.DELIVERY_REVIEW, "root_acceptance": rref, "findings": [],
                   **{k: root[k] for k in ("contract", "stage", "boundary", "checkpoint", "payload_audit")}}
    return {"root": rref, "independent": seal(tmp / "independent.json", independent)}, donor


@pytest.mark.parametrize("label", ["L0", "L1"])
def test_retained_models_use_distinct_actual_receipt_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str) -> None:
    _short, donor = short_graph(tmp_path, monkeypatch, "S0" if label == "L0" else "S1")
    binding = d2.LONG_BINDINGS[label]
    root = read(binding["root"])
    saved = read(root["payload_audit"]["readout"])
    if label == "L0":
        assert root["payload_audit"]["schema"] == ruler.K0_NO_CLIP_SCHEMA
        assert "initial_model_state_sha256" not in saved
        assert set(saved["donor_provenance"]) == {"mode", "sha256", "architecture"}
        assert saved["own_live_state_reconstructed"] is False
        assert "own_live_state_fingerprint" not in saved
    else:
        assert root["payload_audit"]["schema"] == ruler.D1_GPU_SCHEMA
        assert saved["initial_model_state_sha256"] == donor["model_state_sha256"]
        assert saved["own_live_state_fingerprint"]
    metadata = ruler.Metadata(2 << 20, 64 << 20)
    model = d2.retained_model(metadata, binding, label, donor)
    assert model["label"] == {"L0": "K0", "L1": "K1"}[label]
    assert model["step"] == 4096 and model["boundary"]["applied_views"] == 131072
    metadata.finish()


def reseal_retained_proof(binding: dict[str, Any], saved: dict[str, Any], *, schema: str | None = None) -> dict[str, Any]:
    """Rehash fictional metadata after mutation so the real predicate is reached."""
    root = read(binding["root"])
    saved_path = Path(root["payload_audit"]["readout"]["path"])
    saved_path.chmod(0o644)
    root["payload_audit"]["readout"] = seal(saved_path, saved)
    if schema is not None:
        root["payload_audit"]["schema"] = schema
    root_path = Path(binding["root"]["path"])
    root_path.chmod(0o644)
    root_ref = seal(root_path, root)
    independent = read(binding["independent"])
    independent.update(root_acceptance=root_ref, payload_audit=root["payload_audit"])
    independent_path = Path(binding["independent"]["path"])
    independent_path.chmod(0o644)
    return {"root": root_ref, "independent": seal(independent_path, independent)}


@pytest.mark.parametrize(("label", "fault", "error", "message"), [
    ("L0", "mode", ValueError, "saved donor provenance"),
    ("L0", "sha", ValueError, "saved donor provenance"),
    ("L0", "architecture", ValueError, "saved donor provenance"),
    ("L0", "live", ValueError, "saved-only scope"),
    ("L0", "family", ValueError, "corrective saved-only family"),
    ("L1", "missing_initial", KeyError, "initial_model_state_sha256"),
    ("L1", "wrong_initial", ValueError, "initial model"),
])
def test_retained_receipt_refusals_keep_family_requirements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        label: str, fault: str, error: type[Exception], message: str) -> None:
    _short, donor = short_graph(tmp_path, monkeypatch, "S0" if label == "L0" else "S1")
    binding = d2.LONG_BINDINGS[label]
    saved = read(read(binding["root"])["payload_audit"]["readout"])
    if fault in {"mode", "sha", "architecture"}:
        key = "sha256" if fault == "sha" else fault
        saved["donor_provenance"][key] = {"different": True} if fault == "architecture" else "0" * 64 if fault == "sha" else "wrong"
    elif fault == "live":
        saved["own_live_state_reconstructed"] = True
    elif fault == "missing_initial":
        del saved["initial_model_state_sha256"]
    elif fault == "wrong_initial":
        saved["initial_model_state_sha256"] = "0" * 64
    binding = reseal_retained_proof(binding, saved, schema=ruler.D1_GPU_SCHEMA if fault == "family" else None)
    monkeypatch.setattr(d2, "LONG_BINDINGS", {**d2.LONG_BINDINGS, label: binding})
    with pytest.raises(error, match=message):
        d2.retained_model(ruler.Metadata(2 << 20, 64 << 20), binding, label, donor)


@pytest.mark.parametrize("label", ["L0", "L1"])
def test_retained_alias_refuses_the_other_long_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str) -> None:
    _short, donor = short_graph(tmp_path, monkeypatch, "S0" if label == "L0" else "S1")
    other = "L1" if label == "L0" else "L0"
    other_folder = tmp_path / "other"
    other_folder.mkdir()
    short_graph(other_folder, monkeypatch, "S0" if other == "L0" else "S1")
    with pytest.raises(ValueError, match="retained endpoint identity"):
        d2.retained_model(ruler.Metadata(2 << 20, 64 << 20), d2.LONG_BINDINGS[other], label, donor)


@pytest.mark.parametrize("label", ["S0", "S1"])
def test_short_delivery_reaches_real_retained_prefix_and_game_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str) -> None:
    binding, donor = short_graph(tmp_path, monkeypatch, label)
    metadata = ruler.Metadata(2 << 20, 64 << 20)
    model = d2.model_delivery(metadata, binding, label, donor)
    assert model["step"] == 512
    assert model["boundary"]["drawn_rows"] == 16384
    metadata.finish()


@pytest.mark.parametrize(("fault", "message"), [
    ("511", "short dose"), ("513", "short dose"), ("4096", "short dose"),
    ("views", "complete fresh endpoint"), ("source", "source/run"), ("study", "own-live saved proof"),
    ("reference", "frozen reference"), ("loss_recipe", "public contract shape"), ("helper", "observer source identity"),
    ("prefix", "sampler equality"), ("drop_game", "game roster"), ("integer_flag", "own-live saved proof"),
    ("different_owner", "complete learner owner"), ("recovery", "normal GPU delivery envelope"),
])
def test_rehashed_short_family_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, message: str) -> None:
    binding, donor = short_graph(tmp_path, monkeypatch, "S1", fault)
    with pytest.raises(ValueError, match=message):
        d2.model_delivery(ruler.Metadata(2 << 20, 64 << 20), binding, "S1", donor)


def slice_graph(tmp: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    games: list[dict[str, Any]] = [{"pair_id": i, "original_role_ordinal": i + 80, "parent_id": f"parent-{i}",
              "parent_member_ids": [f"member-{i}"], "source_parents": [{"path": "/fiction/parent.json", "sha256": "2" * 64}]} for i in range(80)]
    role: dict[str, Any] = {"source_role": "SCREEN_C_A", "original400_book": {"path": str(tmp / "original.epd"), "sha256": "1" * 64},
        "original_full_role": {"path": str(tmp / "original-role.json"), "sha256": "2" * 64},
        "book": {"path": str(tmp / "slice80_159.epd"), "sha256": "3" * 64}, "parent_map": {"path": str(tmp / "parents.json"), "sha256": "4" * 64},
        "selection": "exact original ordinals80..159; no rng/subsample/reorder", "prefix_start": 80, "pairs": 80, "source_count": 400,
        "complete_parents": 80, "actual_public_loader_count": 80, "no_skips_dedup_shrink": True,
        "parent_order_sha256": hashlib.sha256(ruler.canonical(games)).hexdigest(), "history_order_sha256": "5" * 64}
    projection: dict[str, Any] = {"schema": "D2_COMPLETE_FRESH_SLICE_PARENT_PROJECTION_V1", "study_uuid": d2.STUDY, "design": d2.DESIGN,
        "source_role": "SCREEN_C_A", "original_full_role": role["original_full_role"], "original400_book": role["original400_book"],
        "pairs": 80, "games": games, "scope": "Fictional complete-parent metadata; not an actual role qualification."}
    consumed: dict[str, Any] = {"book": role["book"], "pairs": 80, "actual_public_reads": 1, "full16_history_order_EPD_exact": True}
    return projection, role, consumed


def test_fresh_slice_uses_original_ordinals_and_epd_descriptor(tmp_path: Path) -> None:
    projection, role, consumed = slice_graph(tmp_path)
    d2.slice_projection(projection, role, consumed)


@pytest.mark.parametrize(("fault", "message"), [
    ("old_prefix", "fixed slice"), ("ordinal", "original parent ordinal"),
    ("history", "full-history consumption"), ("source_role", "fixed slice"), ("old_study", "parent authority"),
])
def test_fresh_slice_refuses_relabelled_or_incomplete_witnesses(tmp_path: Path, fault: str, message: str) -> None:
    projection, role, consumed = slice_graph(tmp_path)
    if fault == "old_prefix":
        role["prefix_start"] = 0
    elif fault == "ordinal":
        projection["games"][0]["original_role_ordinal"] = 0
        role["parent_order_sha256"] = hashlib.sha256(ruler.canonical(projection["games"])).hexdigest()
    elif fault == "history":
        consumed["full16_history_order_EPD_exact"] = False
    elif fault == "source_role":
        role["source_role"] = "SCREEN_C_B"
    else:
        projection["study_uuid"] = "8045699e-4088-4a41-ae9c-4013d4a55469"
    with pytest.raises(ValueError, match=message):
        d2.slice_projection(projection, role, consumed)


def role_authority_graph(tmp: Path, monkeypatch: pytest.MonkeyPatch, fault: str = "") -> dict[str, Any]:
    names = ("SCREEN_C_A", "SCREEN_C_B", "SCREEN_C_A39", "CONFIRM_C_A", "CONFIRM_C_B", "CONFIRM_C_A39")
    books = {name: {"path": str(tmp / (name + ".epd")), "sha256": "1" * 64} for name in names}
    roles = {name: {"path": str(tmp / (name + ".json")), "sha256": "2" * 64} for name in names}
    original = seal(tmp / "original-role-root.json", {"status": "ROOT_ACCEPTS_CLOSED_PROSPECTIVE_WHOLE_GAME_ROLES_V1", "books": books, "roles": roles})
    old_ind = seal(tmp / "original-role-ind.json", {"status": "ACCEPT_CLOSED_PROSPECTIVE_WHOLE_GAME_ROLES_NO_FINDINGS_V1", "root_acceptance": original, "findings": []})
    source_roles = {"root": original, "independent": old_ind}
    monkeypatch.setattr(d2, "SOURCE_ROLES", source_roles)
    d1 = seal(tmp / "old-D1-readout.json", {"all_seven_technically_complete": True})
    d1i = seal(tmp / "old-D1-review.json", {"readout": d1, "findings": []})
    monkeypatch.setattr(d2, "D1_NO_WINNER", d1)
    monkeypatch.setattr(d2, "D1_NO_WINNER_INDEPENDENT", d1i)
    usage: dict[str, Any] = {"TRAIN": [[0, 192]], "HOLD": [[0, 64]], "SCREEN_C_A": [[0, 80]], "SCREEN_C_B": [[0, 80]],
                             "SCREEN_C_A39": [], "CONFIRM_C_A": [], "CONFIRM_C_B": [], "CONFIRM_C_A39": []}
    registry: dict[str, Any] = {"status": "ROOT_SELECTS_KNOWN_EXPOSURE_HISTORY_BEFORE_D2_ROLE_QUALIFICATION_V1", "study_uuid": d2.STUDY,
        "design": d2.DESIGN, "source_roles": source_roles, "prior_closed_owner_count": 406, "known_usage": usage,
        "D1_readout": d1, "D1_review": d1i, "usage_evidence": [], "no_other_known_controlled_role_usage": True,
        "external_usage_claim": fault == "external", "scope": "Fictional known-controlled-exposure metadata only."}
    registry_ref = seal(tmp / "registry.json", registry)
    registry_ind = seal(tmp / "registry-review.json", {**registry,
        "status": "ACCEPT_ROOT_KNOWN_EXPOSURE_HISTORY_BEFORE_D2_ROLE_QUALIFICATION_NO_FINDINGS_V1", "root_acceptance": registry_ref, "findings": []})
    registry_pair = {"root": registry_ref, "independent": registry_ind}
    _projection, entry, consumed = slice_graph(tmp)
    entry["original400_book"] = books["SCREEN_C_A"]
    entry["original_full_role"] = roles["SCREEN_C_A"]
    if fault == "old_prefix":
        entry["prefix_start"] = 0
    helper = {"path": "/fiction/D2/qualify_slice.py", "sha256": d2.ROLE_HELPER_SHA256}
    pins = {"path": "/fiction/role-pins.json", "sha256": "7" * 64}
    controls = {"path": "/fiction/role-controls.py", "sha256": "8" * 64}
    source_review = seal(tmp / "role-source-review.json", {"status": "ACCEPT_KL_DEVELOPMENT_FIRST_D2_PREFIX_SOURCE_ONLY_NO_FINDINGS_V1",
        "helper": helper, "controls": controls, "source_pins": pins, "findings": []})
    monkeypatch.setattr(d2, "ROLE_SOURCE_REVIEW", source_review)
    protocol = seal(tmp / "role-protocol.json", {"schema": "D2_FRESH_SLICE_METADATA_CPU_PROTOCOL_V1", "status": "BOUND_D2_FRESH_SLICE_METADATA_ONLY_V1",
        "study_uuid": d2.STUDY, "design": d2.DESIGN, "role_root": original, "role_independent": old_ind, "exposure_registry": registry_pair,
        "helper": helper, "controls": controls, "source_review": source_review, "source_pins": pins, "limits": {"outer_seconds": 300, "inner_seconds": 270}})
    ad_path = tmp / "role-admission.json"
    normalized = ["/fiction/python", "-I", "-B", helper["path"], "--protocol", protocol["path"], "--protocol-sha256", protocol["sha256"],
                  "--admission", str(ad_path), "--admission-sha256", "SELF_ADMISSION_SHA256"]
    admission = seal(ad_path, {"status": "ROOT_ADMITS_KL_DEVELOPMENT_FIRST_D2_PREFIX_CPU_V1", "execution_authorized": True,
        "resource": "cpu", "seconds": 300, "protocol": protocol, "helper": helper, "source_review": source_review,
        "source_pins": pins, "owner": {"sha256": "e" * 64}, "normalized_child": normalized})
    command = copy.deepcopy(normalized)
    command[command.index("--admission-sha256") + 1] = admission["sha256"]
    terminal = seal(tmp / "role-terminal.json", {"status": "PASS", "returncode": 0, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED",
        "resource": "cpu", "seconds": 300, "wall_seconds": 2.0, "command": command, "source_revision": "f" * 40, "cwd": "/fiction/owner",
        "supervisor_sha256": "e" * 64, "supervisor_pid": 2147483645, "known_processes": {"2147483646": 1}})
    owner: dict[str, Any] = {"terminal": terminal, "admission": admission, "command": command, "source_revision": "f" * 40, "cwd": "/fiction/owner",
        "closure": {"supported_eof": True, "exit_code": 0, "supervisor_pid": 2147483645, "supervisor_closed": True,
                    "owned_births_closed": True, "known_processes": {"2147483646": 1}}}
    consumption = [{"book": book, "pairs": 400, "actual_public_reads": 1, "full16_history_order_EPD_exact": True} for book in books.values()] + [consumed]
    if fault == "integer_history":
        consumed["full16_history_order_EPD_exact"] = 1
    report = seal(tmp / "role-readout.json", {"status": "D2_FRESH_SLICE_METADATA_CANDIDATE_PENDING_ROOT_CLOSURE_V1", "protocol": protocol,
        "design": d2.DESIGN, "role_root": original, "role_independent": old_ind, "prefixes": {"SCREEN_C_A": entry},
        "original_six400_books": books, "exposure_registry": registry_pair, "complete_parent_descriptors_preserved": True, "controls_passed": 16,
        "all8role_domains_disjoint": {"group": 2656, "parent": 2657, "history": 2656, "epd": 2656, "history_sha": 2656, "input": 5312},
        "public_consumption": consumption, "known_exposure_check": {"registry": registry_pair, "prior_closed_owner_count": 406, "known_usage": usage,
            "conservatively_used_rows": 416, "selected_original_ordinals": [80, 160], "known_controlled_domains_disjoint": fault != "collision",
            "selected_domain_counts": {"group": 80, "parent": 81, "history": 80, "epd": 80, "history_sha": 80, "input": 160}, "scope": "Known use only."}})
    root: dict[str, Any] = {"status": "ROOT_ACCEPTS_CLOSED_D2_FRESH_SCREEN_A_80_159_METADATA_V1", "study_uuid": d2.STUDY, "design": d2.DESIGN,
        "source_roles": source_roles, "prefixes": {"SCREEN_C_A": entry}, "original_six400_books": books, "protocol": protocol, "readout": report,
        "owner": owner, "exposure_registry": registry_pair}
    root_ref = seal(tmp / "role-root.json", root)
    return {"root": root_ref, "independent": seal(tmp / "role-review.json", {**root,
        "status": "ACCEPT_CLOSED_D2_FRESH_SCREEN_A_80_159_METADATA_NO_FINDINGS_V1", "root_acceptance": root_ref, "findings": []})}


def test_closed_role_authority_reaches_source_owner_and_known_exposure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding = role_authority_graph(tmp_path, monkeypatch)
    metadata = ruler.Metadata(2 << 20, 32 << 20)
    entry = d2.role_authority(metadata, binding)
    assert entry["prefix_start"] == 80
    metadata.finish()


@pytest.mark.parametrize(("fault", "message"), [("old_prefix", "exact fresh slice"), ("collision", "actual known-exposure"),
                                               ("external", "reviewed known exposure"), ("integer_history", "seven actual public")])
def test_closed_role_authority_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, message: str) -> None:
    binding = role_authority_graph(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError, match=message):
        d2.role_authority(ruler.Metadata(2 << 20, 32 << 20), binding)


def portable_ast(value: Any) -> Any:
    """Version-neutral fields; ignore only the known empty generic parameter list."""
    if isinstance(value, ast.AST):
        fields = list(ast.iter_fields(value))
        if type(value).__name__ in ("FunctionDef", "AsyncFunctionDef", "ClassDef"):
            parameters = getattr(value, "type_params", [])
            if type(parameters) is not list:
                raise TypeError("generic AST parameters must be a list")
            if parameters and not any(name == "type_params" for name, _item in fields):
                fields.append(("type_params", parameters))
        return {"node": type(value).__name__, "fields": [
            [name, portable_ast(item)] for name, item in fields
            if not (name == "type_params" and type(value).__name__ in ("FunctionDef", "AsyncFunctionDef", "ClassDef")
                    and type(item) is list and not item)]}
    if isinstance(value, list):
        return [portable_ast(item) for item in value]
    if type(value) is bytes:
        return {"scalar_type": "bytes", "hex": value.hex()}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("unsupported portable AST scalar " + type(value).__name__)


def prefix_namespace() -> dict[str, Any]:
    """Compile the actual published declaration subset, never helper main."""
    folder = Path(__file__).resolve().parents[1] / "docs/experiments/source/2026-11-17-d2-short-horizon"
    path = folder / "prefix_audit.py.txt"
    raw = path.read_bytes()
    provenance = json.loads((folder / "prefix_audit.provenance.json").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == provenance["snapshot_sha256"]
    assert provenance["portable_AST_fingerprint"]["schema"] == "AST_ITER_FIELDS_PORTABLE_EMPTY_TYPE_PARAMS_V1"
    tree = ast.parse(raw)
    declarations = [node for node in tree.body if not isinstance(node, ast.Expr)]
    assert len(declarations) == len(provenance["declarations"])
    for node, identity in zip(declarations, provenance["declarations"], strict=True):
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef))
        segment = ast.get_source_segment(raw.decode(), node)
        assert segment is not None
        assert hashlib.sha256(segment.encode()).hexdigest() == identity["declaration_sha256"]
        encoded = json.dumps(portable_ast(node), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == identity["portable_AST_sha256"]
        assert identity["portable_AST_sha256"] == identity["intended_observer_portable_AST_sha256"]
        if isinstance(node, ast.Import):
            assert all(value.name in {"hashlib", "json", "math", "os", "re", "stat"} for value in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module == "pathlib"
    names = {node.name for node in declarations if isinstance(node, ast.FunctionDef)}
    assert names == {"require", "sha", "canonical", "digest", "stamp", "external", "pairs", "finite", "bad_constant",
                     "journal_stream", "compare_prefix", "finish_prefix"}
    namespace: dict[str, Any] = {"__name__": "d2_actual_prefix_declaration_fixture"}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class MeterStub:
    def __init__(self, limit: int = 4 << 30) -> None:
        self.used = 0
        self.limit = limit

    def charge(self, count: int) -> None:
        if type(count) is not int or count < 0 or self.used + count > self.limit:
            raise RuntimeError("fixture metered read budget")
        self.used += count


def journal_fixture(folder: Path, batches: int = 512, fault: str = "") -> tuple[dict[str, str], list[list[Any]]]:
    """Synthetic row IDs only; no actual bank or draw journal is read."""
    folder.mkdir()
    path = folder / "draws.jsonl"
    membership: list[list[Any]] = [[1, hashlib.sha256(str(index).encode()).hexdigest(), 92 if index < 96 else 91] for index in range(192)]
    if fault == "oversize":
        with path.open("wb") as stream:
            stream.truncate((64 << 20) + 1)
        path.chmod(0o444)
        return {"path": str(path), "sha256": "a" * 64}, membership
    with path.open("wb") as stream:
        for index in range(batches):
            rows: list[dict[str, Any]] = [{"generation": 1, "source_sha256": membership[(index * 32 + offset) % 192][1], "row_index": 0}
                                          for offset in range(32)]
            batch: dict[str, Any] = {"draw_batch": index, "rows": rows}
            if index == 0:
                if fault == "batch":
                    batch["draw_batch"] = 1
                elif fault == "rows":
                    rows.pop()
                elif fault == "generation":
                    rows[0]["generation"] = True
                elif fault == "row_type":
                    rows[0]["row_index"] = 0.0
                elif fault == "row_range":
                    rows[0]["row_index"] = 999
                elif fault == "source":
                    rows[0]["source_sha256"] = "0" * 64
                elif fault == "different_valid_row":
                    rows[0]["row_index"] = 1
            value = ruler.canonical(batch) + b"\n"
            if index == batches - 1 and fault == "truncated":
                value = value[: len(value) // 2]
            if index == 0 and fault == "duplicate_key":
                value = value.replace(b'{"draw_batch":0,', b'{"draw_batch":0,"draw_batch":0,', 1)
            stream.write(value)
    raw = path.read_bytes()
    path.chmod(0o444)
    return {"path": str(path), "sha256": "0" * 64 if fault == "hash" else hashlib.sha256(raw).hexdigest()}, membership


def test_actual_prefix_functions_positive_three_long_passes(tmp_path: Path) -> None:
    # Nonempty generic parameters remain semantic input to the fingerprint.
    plain = ast.parse("def marker():\n    pass\n").body[0]
    generic = copy.deepcopy(plain)
    setattr(generic, "type_params", [ast.Name(id="T", ctx=ast.Load())])
    assert portable_ast(plain) != portable_ast(generic)
    namespace = prefix_namespace()
    long_ref, membership = journal_fixture(tmp_path / "long", 4096)
    short_ref, _ = journal_fixture(tmp_path / "short")
    meter = MeterStub()
    before = namespace["journal_stream"](long_ref, 4096, membership, meter)
    comparator = {"root": {"path": "/fiction/long-root.json", "sha256": "1" * 64},
                  "independent": {"path": "/fiction/long-review.json", "sha256": "2" * 64}}
    context = {"journal": long_ref, "membership": membership, "before": before,
               "boundary": {"path": "/fiction/long-boundary.json", "sha256": "3" * 64}}
    boundary = {"membership": membership, "draws_sha256": short_ref["sha256"]}
    drawn = {"population_rows": 17568, "games": 192, "batches": 512, "views": 16384,
             "per_game_aggregate_counts": [{} for _ in range(192)]}
    plan = {"learner_output": str(Path(short_ref["path"]).parent), "long_comparator": comparator}
    report = namespace["compare_prefix"](plan, meter, context, boundary, drawn)
    namespace["finish_prefix"](plan, meter, context, {"draw_prefix_comparison": report})
    assert report["equal"] is True
    assert report["long_prefix_sha256"] == report["short_prefix_sha256"] == short_ref["sha256"]
    assert report["long_stream_passes"] == 3
    assert report["long_raw_read_bytes"] == 3 * Path(long_ref["path"]).stat().st_size
    assert report["short_raw_read_bytes"] == Path(short_ref["path"]).stat().st_size
    assert meter.used == report["long_raw_read_bytes"] + report["short_raw_read_bytes"]


@pytest.mark.parametrize("fault", ["batch", "rows", "generation", "row_type", "row_range", "source", "truncated", "duplicate_key", "hash", "oversize"])
def test_actual_prefix_stream_refuses_malformed_fixture(tmp_path: Path, fault: str) -> None:
    namespace = prefix_namespace()
    ref, membership = journal_fixture(tmp_path / "journal", fault=fault)
    with pytest.raises((RuntimeError, ValueError)):
        namespace["journal_stream"](ref, 512, membership, MeterStub())


@pytest.mark.parametrize(("actual", "expected"), [(511, 512), (513, 512), (4095, 4096), (512, 511)])
def test_actual_prefix_stream_refuses_wrong_horizon(tmp_path: Path, actual: int, expected: int) -> None:
    namespace = prefix_namespace()
    ref, membership = journal_fixture(tmp_path / "journal", actual)
    with pytest.raises(RuntimeError):
        namespace["journal_stream"](ref, expected, membership, MeterStub())


def test_actual_prefix_comparison_refuses_valid_different_row(tmp_path: Path) -> None:
    namespace = prefix_namespace()
    long_ref, membership = journal_fixture(tmp_path / "long", 4096)
    short_ref, _ = journal_fixture(tmp_path / "short", fault="different_valid_row")
    meter = MeterStub()
    before = namespace["journal_stream"](long_ref, 4096, membership, meter)
    context = {"journal": long_ref, "membership": membership, "before": before}
    boundary = {"membership": membership, "draws_sha256": short_ref["sha256"]}
    drawn = {"population_rows": 17568, "games": 192, "batches": 512, "views": 16384, "per_game_aggregate_counts": [{} for _ in range(192)]}
    with pytest.raises(RuntimeError, match="physical-row prefix differs"):
        namespace["compare_prefix"]({"learner_output": str(Path(short_ref["path"]).parent)}, meter, context, boundary, drawn)


def test_actual_prefix_stream_budget_checked_before_payload_read(tmp_path: Path) -> None:
    namespace = prefix_namespace()
    ref, membership = journal_fixture(tmp_path / "journal")
    reads: list[int] = []

    class TrackedStream:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def __enter__(self) -> TrackedStream:
            self.stream.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self.stream.__exit__(*args)

        def fileno(self) -> int:
            return self.stream.fileno()

        def readline(self, size: int) -> bytes:
            reads.append(size)
            return self.stream.readline(size)

        def read(self, size: int) -> bytes:
            reads.append(size)
            return self.stream.read(size)

    class TrackedOS:
        def __getattr__(self, name: str) -> Any:
            return getattr(os, name)

        def fdopen(self, *args: Any, **kwargs: Any) -> TrackedStream:
            return TrackedStream(os.fdopen(*args, **kwargs))

    namespace["os"] = TrackedOS()
    with pytest.raises(RuntimeError, match="metered read budget"):
        namespace["journal_stream"](ref, 512, membership, MeterStub(limit=1))
    assert reads == []


def test_actual_prefix_stream_detects_stamp_change(tmp_path: Path) -> None:
    namespace = prefix_namespace()
    ref, membership = journal_fixture(tmp_path / "journal")
    original_stamp = namespace["stamp"]
    calls = 0

    def changed(value: Any) -> dict[str, int]:
        nonlocal calls
        calls += 1
        result = original_stamp(value)
        if calls == 3:
            result["st_mtime_ns"] += 1
        return result

    namespace["stamp"] = changed
    with pytest.raises(RuntimeError, match="journal fd changed"):
        namespace["journal_stream"](ref, 512, membership, MeterStub())
