"""Fictional recovery authorities, real public metadata predicates; no payloads.

Only caller-trusted issuer/path anchors are redirected to fixture objects. The
failed-owner, CPU-owner, amendment, family, saved-proof and coverage predicates
run unmocked. No real checkpoint/journal/row/native source is opened.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import core_generation_holdout_ruler as ruler
from tests.test_core_same_gpu_payload_delivery import gpu_graph, seal


def read(ref: dict[str, str]) -> Any:
    return json.loads(Path(ref["path"]).read_bytes())


def recovery_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str = "") -> tuple[dict[str, Any], ruler.bank.AcceptedGeneration]:
    binding, train = gpu_graph(tmp_path, arm="K0", d1=True,
        design_ref={"path": str(tmp_path / "design.json"), "sha256": ruler.D1_DESIGN_SHA256})
    original = read(binding["root"])
    oldproof = read(original["payload_audit"]["readout"])
    oldplan_ref = oldproof["observer_plan"]
    oldplan = read(oldplan_ref)
    oldplan["arm"] = "K0"
    # The graph writer carries the original plan SHA into every original owner
    # command/admission when the extra arm field changes that genuine edge.
    oldplan_ref2 = seal(tmp_path / "original-plan.json", oldplan)
    failed = copy.deepcopy(original["learner_owner"])
    original_ad = read(failed["admission"])
    cmd = failed["command"]
    cmd[cmd.index("--observer-plan") + 1] = oldplan_ref2["path"]
    cmd[cmd.index("--observer-plan-sha256") + 1] = oldplan_ref2["sha256"]
    original_ad["observer_plan"] = oldplan_ref2
    cmd[cmd.index("--admission") + 1] = str(tmp_path / "original-failed-admission.json")
    normalized = list(cmd)
    normalized[normalized.index("--admission-sha256") + 1] = "SELF_ADMISSION_SHA256"
    original_ad["normalized_child"] = normalized
    failed["admission"] = seal(tmp_path / "original-failed-admission.json", original_ad)
    cmd[cmd.index("--admission-sha256") + 1] = failed["admission"]["sha256"]
    terminal = read(failed["terminal"])
    terminal.update(status="PASS" if fault == "failed_status" else "FAILED", returncode=1, command=cmd)
    failed["terminal"] = seal(tmp_path / "original-failed-terminal.json", terminal)
    failed["closure"]["exit_code"] = 0 if fault == "failed_exit" else 1
    historical_close = seal(tmp_path / "original-close.json", {
        "status": "ROOT_OBSERVES_SUPPORTED_EOF_CLOSED_CUSTODY_NO_RESULT_ACCEPTANCE_V1",
        "terminal": failed["terminal"], "admission": failed["admission"], "closure": failed["closure"],
        "command": seal(tmp_path / "original-command.json", {"child": cmd})})
    trigger: dict[str, Any] = {"terminal": failed["terminal"], "closure": historical_close,
        "public_boundary": original["boundary"], "checkpoint_REFERENCE_ONLY": original["checkpoint"],
        "helper": oldplan["helper"], "log": {"path": str(tmp_path / "failed-log.txt"), "sha256": "e" * 64},
        "owner_wall_seconds": terminal["wall_seconds"]}
    selected = seal(tmp_path / "selected-source.json", {"execution_authorized": False,
        "design": {"path": str(tmp_path / "design.json"), "sha256": ruler.D1_DESIGN_SHA256}, "trigger": trigger})
    selection_review = seal(tmp_path / "selection-review.json", {"findings": []})
    monkeypatch.setattr(ruler, "K0_RECOVERY_SELECTION", selected)
    monkeypatch.setattr(ruler, "K0_RECOVERY_SELECTION_REVIEW", selection_review)
    authority_path = str(tmp_path / "root-authority.json")
    amendment_path = str(tmp_path / "amendment-publication.json")
    monkeypatch.setattr(ruler, "K0_RECOVERY_AUTHORITY_PATH", authority_path)
    monkeypatch.setattr(ruler, "K0_RECOVERY_AMENDMENT_PATH", amendment_path)
    helper = {"path": "/fiction/K0-audit.py", "sha256": ruler.K0_RECOVERY_HELPER}
    source_review = seal(tmp_path / "audit-source-review.json", {
        "status": "ACCEPT_KL_D1_K0_SAVED_ENDPOINT_RECOVERY_SOURCE_ONLY_NO_FINDINGS_V1",
        "helper": helper, "source_pins": oldproof["source_pins"], "findings": []})
    monkeypatch.setattr(ruler, "K0_RECOVERY_SOURCE_REVIEW", source_review)
    amendment = seal(Path(amendment_path), {
        "status": "ROOT_RECORDS_PUBLISHED_D1_T1_MECHANICAL_RECOVERY_AMENDMENT_V1", "selection": selected,
        "design": read(selected)["design"], "main_revision": "f" * 40,
        "document_path": "docs/experiments/fictional-recovery.md", "document_sha256": "a" * 64,
        "independent_source_review": seal(tmp_path / "amendment-review.json", {"findings": []})})
    authority = seal(Path(authority_path), {
        "status": "ROOT_AUTHORIZES_KL_D1_T1_K0_SAVED_ENDPOINT_RECOVERY_V1", "selection": selected,
        "selection_review": selection_review, "amendment": amendment, "auditor": helper,
        "auditor_source_review": source_review, "trigger": trigger})
    failed_observation = seal(Path(oldplan["observation"] + ".failed.json"), {
        "status": "FAILED_KL_DEVELOPMENT_FIRST_D1_PASSIVE_PROOF_NO_DELIVERY_V1",
        "error": "RuntimeError('saved counters/architecture/reference')"})
    proof_path = tmp_path / "recovery-proof.json"
    plan: dict[str, Any] = {"schema": "KL_D1_K0_SAVED_ENDPOINT_RECOVERY_PLAN_V1",
        "status": "BOUND_KL_D1_K0_SAVED_ENDPOINT_RECOVERY_NOT_DELIVERY_AUTHORITY_V1",
        "selection": selected, "selection_review": selection_review, "authority": authority, "layout_basis": ruler.K0_LAYOUT_BASIS,
        "helper": helper, "source_review": source_review, "source_pins": oldproof["source_pins"],
        **{key: original[key] for key in ("contract", "stage", "boundary", "checkpoint")},
        "failed_gpu_owner": failed, "original_observer_plan": oldplan_ref2, "failed_observation": failed_observation,
        "observation": str(proof_path), "limits": {"outer_seconds": 600, "inner_seconds": 570,
            "checkpoint_file_bytes": 8 << 30, "journal_bytes": 64 << 20, "serialized_read_bytes": 32 << 30, "aggregate_bytes": 4 << 20}}
    planref = seal(tmp_path / "recovery-plan.json", plan)
    adpath = tmp_path / "CPU-admission.json"
    cpu_command = ["/fiction/python", "-I", "-B", helper["path"], "--recovery-plan", planref["path"],
        "--recovery-plan-sha256", planref["sha256"], "--admission", str(adpath), "--admission-sha256", "SELF_ADMISSION_SHA256"]
    cpu_ad: dict[str, Any] = {"status": "ROOT_ADMITS_KL_D1_T1_K0_SAVED_ENDPOINT_RECOVERY_CPU_V1", "execution_authorized": True,
        "recovery_plan": planref, "authority": authority, "resource": "cpu", "seconds": 600,
        "owner": {"sha256": "e" * 64}, "normalized_child": list(cpu_command), "helper": helper,
        "source_review": source_review, "source_pins": oldproof["source_pins"]}
    if fault == "cpu_authority":
        cpu_ad["authority"] = {**authority, "sha256": "0" * 64}
    elif fault == "cpu_normalized":
        cpu_ad["normalized_child"][0] = "/fiction/wrong-python"
    adref = seal(adpath, cpu_ad)
    cpu_command[cpu_command.index("--admission-sha256") + 1] = adref["sha256"]
    cpu_terminal: dict[str, Any] = {**terminal, "status": "PASS", "returncode": 0, "command": cpu_command,
        "resource": "cpu", "seconds": 600}
    if fault == "cpu_resource":
        cpu_terminal["resource"] = "gpu"
    elif fault == "cpu_cap":
        cpu_terminal["seconds"] = 601
    cpu_owner: dict[str, Any] = {"terminal": seal(tmp_path / "CPU-terminal.json", cpu_terminal), "admission": adref, "command": cpu_command,
        "source_revision": failed["source_revision"], "cwd": failed["cwd"], "closure": {**failed["closure"], "exit_code": 0}}
    if fault == "cpu_closure":
        cpu_owner["closure"]["owned_births_closed"] = False
    proof: dict[str, Any] = {"status": ruler.K0_RECOVERY_PENDING, "arm": "K0", "helper": helper, "source_review": source_review,
        "source_pins": oldproof["source_pins"], "source_revision": original["source_revision"], "source_checkout": original["source_checkout"],
        **{key: original[key] for key in ("contract", "stage", "boundary", "checkpoint")},
        "architecture": read(original["contract"])["architecture"], "step": 4096, "failed_gpu_owner": failed,
        "recovery_authority": authority, "original_failure_log": trigger["log"], "historical_closure": historical_close,
        "original_observer_plan": oldplan_ref2, "failed_observation": failed_observation, "recovery_plan": planref,
        "layout_basis": {**ruler.K0_LAYOUT_BASIS, "scope": "MODEL_LAYOUT_BASIS_ONLY"},
        "saved_checks": dict.fromkeys(("model", "optimizer", "scheduler", "zclip", "counters", "recorded_rng", "journal", "membership", "policy_anchor", "provenance"), True),
        "checkpoint_CPU_weights_only_decodes": 1, "own_live_state_reconstructed": False,
        "historical_final_brackets_reconstructed": False, "historical_reference_immutability_reconstructed": False,
        **dict.fromkeys(("model_state_sha256", "saved_payload_fingerprint", "key_shape_dtype_sha256", "scheduler_fingerprint", "zclip_fingerprint"), "3" * 64),
        "optimizer_named_parameters": 479, "active_optimizer_slots": 438, "model_changed_from_authenticated_A39": True,
        "five_rng_snapshot_sha256": oldproof["five_rng_snapshot_sha256"], "membership_sha256": oldproof["membership_sha256"],
        "outer_seconds": 600, "absolute_inner_seconds": 570, "elapsed_seconds": 2.0, "CUDA_initialized": False,
        "current_CPU_RNG_unchanged": True, "Torch_intra": 2, "Torch_interop": 2,
        "current_native_named_file_bracket": {"fixture": "current-only actual audit witness"},
        "serialized_read_bytes_total": 100, "serialized_read_bytes_limit": 32 << 30,
        "serialized_read_bytes_by_category": {"fixture_metadata": 100}, "serialized_read_bytes_by_path": {"fiction.json": 100},
        "draw_coverage": oldproof["draw_coverage"]}
    if fault == "old_pass":
        proof["status"] = ruler.D1_GPU_STATUS
    elif fault == "own_live":
        proof["own_live_state_fingerprint"] = "4" * 64
    elif fault in {"historical_final_brackets_reconstructed", "own_live_state_reconstructed", "historical_reference_immutability_reconstructed"}:
        proof[fault] = True
    elif fault in {"membership_sha256", "five_rng_snapshot_sha256"}:
        proof[fault] = "0" * 64
    elif fault == "helper":
        proof["helper"] = {**helper, "sha256": "0" * 64}
    elif fault == "threads":
        proof["Torch_interop"] = 1
    elif fault == "decode":
        proof["checkpoint_CPU_weights_only_decodes"] = 2
    elif fault == "saved_checks":
        proof["saved_checks"]["scheduler"] = False
    elif fault == "integer_check":
        proof["saved_checks"]["model"] = 1
    elif fault == "budget":
        proof["serialized_read_bytes_total"] += 1
    elif fault == "views":
        proof["draw_coverage"]["views"] -= 1
    elif fault == "drop_game":
        proof["draw_coverage"]["per_game_aggregate_counts"].pop()
    root: dict[str, Any] = {**original, "status": ruler.K0_RECOVERY_ROOT, "learner_owner": failed, "recovery_authority": authority,
        "payload_audit": {"schema": ruler.K0_RECOVERY_SCHEMA, "readout": seal(proof_path, proof), "owner": cpu_owner}}
    rref = seal(tmp_path / "recovery-root.json", root)
    review: dict[str, Any] = {**root, "status": ruler.K0_RECOVERY_REVIEW, "root_acceptance": rref, "findings": []}
    return {"root": rref, "independent": seal(tmp_path / "recovery-review.json", review)}, train


def test_specific_K0_recovery_real_delivery_and_full_game_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding, train = recovery_graph(tmp_path, monkeypatch)
    metadata = ruler.Metadata(1 << 20, 16 << 20)
    model = ruler.model_delivery(metadata, binding, "K0", _d1=True)
    ruler.same_gpu_game_coverage(metadata, binding, model, train, _d1=True)
    assert model["step"] == 4096
    assert len(train.games) == 192
    root = metadata.read(binding["root"])
    assert set(root) == {"status", "contract", "stage", "boundary", "checkpoint", "source_revision", "source_checkout", "learner_owner", "payload_audit", "recovery_authority"}
    assert metadata.observed(root["learner_owner"]["terminal"])["status"] == "FAILED"
    assert metadata.observed(root["payload_audit"]["owner"]["terminal"])["status"] == "PASS"
    metadata.finish()


@pytest.mark.parametrize(("fault", "message"), [
    ("old_pass", "K0 saved-endpoint proof identity differs"),
    ("failed_status", "original failed terminal/identity differs"),
    ("failed_exit", "original failed closure differs"),
    ("cpu_resource", "recovery CPU owner not exact closed PASS"),
    ("cpu_cap", "recovery CPU owner not exact closed PASS"),
    ("cpu_closure", "recovery CPU closure differs"),
    ("cpu_authority", "recovery CPU admission/authority differs"),
    ("cpu_normalized", "recovery CPU normalized admission differs"),
    ("helper", "unsupported K0 recovery auditor"),
    ("own_live", "K0 saved-only scope differs"),
    ("historical_final_brackets_reconstructed", "K0 saved-only scope differs"),
    ("own_live_state_reconstructed", "K0 saved-only scope differs"),
    ("historical_reference_immutability_reconstructed", "K0 saved-only scope differs"),
    ("saved_checks", "K0 saved-only scope differs"),
    ("integer_check", "K0 saved-only scope differs"),
    ("decode", "K0 saved-only scope differs"),
    ("membership_sha256", "K0 recorded RNG/membership differs"),
    ("five_rng_snapshot_sha256", "K0 recorded RNG/membership differs"),
    ("threads", "current CPU recovery custody incomplete"),
    ("budget", "K0 actual serialized-read accounting differs"),
    ("views", "K0 recovered draw totals differ"),
])
def test_rehashed_recovery_proof_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, message: str) -> None:
    binding, _ = recovery_graph(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError, match=message):
        ruler.model_delivery(ruler.Metadata(1 << 20, 16 << 20), binding, "K0", _d1=True)


@pytest.mark.parametrize(("label", "d1"), [("K001", True), ("K01", True), ("K1", True), ("K0", False), ("A", False)])
def test_recovery_never_enters_other_arm_or_legacy_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, d1: bool) -> None:
    binding, _ = recovery_graph(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="K0 recovery is D1 K0 only"):
        ruler.model_delivery(ruler.Metadata(1 << 20, 16 << 20), binding, label, _d1=d1)


def test_recovery_missing_game_refused_after_real_delivery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding, train = recovery_graph(tmp_path, monkeypatch, "drop_game")
    metadata = ruler.Metadata(1 << 20, 16 << 20)
    model = ruler.model_delivery(metadata, binding, "K0", _d1=True)
    with pytest.raises(ValueError, match="same-GPU coverage game roster differs"):
        ruler.same_gpu_game_coverage(metadata, binding, model, train, _d1=True)


@pytest.mark.parametrize(("variant", "message"), [
    ("corrected", ""),
    ("foreign_review", "corrected D1 prover review differs"),
    ("engineering", "unsupported same-GPU prover"),
])
def test_corrected_4096_prover_explicit_fingerprint_not_engineering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str, message: str) -> None:
    # Factory default selects a fingerprint; restore the actual production
    # allowlist BEFORE calling the real consumer. No predicate is patched.
    fingerprint = "acf8aa74c99787bb715538a8842810953bed5f0592fb1e9278c60fdecc2810e7" if variant == "engineering" else ruler.D1_T1_GPU_HELPER
    with monkeypatch.context() as factory:
        factory.setattr(ruler, "D1_GPU_HELPER", fingerprint)
        binding, train = gpu_graph(tmp_path, arm="K001", d1=True,
            design_ref={"path": str(tmp_path / "design.json"), "sha256": ruler.D1_DESIGN_SHA256})
    root = read(binding["root"])
    saved = read(root["payload_audit"]["readout"])
    if variant == "corrected":
        # Fictional independent issuer anchor; exact accepted-reference guard
        # and all review/plan/source/owner joins execute normally afterward.
        monkeypatch.setattr(ruler, "D1_T1_GPU_SOURCE_REVIEW", saved["source_review"])
    metadata = ruler.Metadata(1 << 20, 16 << 20)
    if message:
        with pytest.raises(ValueError, match=message):
            ruler.model_delivery(metadata, binding, "K001", _d1=True)
    else:
        model = ruler.model_delivery(metadata, binding, "K001", _d1=True)
        ruler.same_gpu_game_coverage(metadata, binding, model, train, _d1=True)
        assert model["step"] == 4096
        metadata.finish()


def no_clip_corrective_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str = "") -> tuple[dict[str, Any], ruler.bank.AcceptedGeneration]:
    """Fictional two-failure chain; real public delivery/coverage/decision guards."""
    old_binding, train = recovery_graph(tmp_path, monkeypatch)
    old_root = read(old_binding["root"])
    proof = read(old_root["payload_audit"]["readout"])
    old_plan = read(proof["recovery_plan"])
    previous_owner = old_root["payload_audit"]["owner"]
    previous_terminal = read(previous_owner["terminal"])
    previous_terminal.update(status="PASS" if fault == "previous_status" else "FAILED", returncode=1, wall_seconds=6.470659800921567)
    previous_terminal_ref = seal(tmp_path / "previous-failed-CPU-terminal.json", previous_terminal)
    previous_closure: dict[str, Any] = {**previous_owner["closure"], "supported_eof": True,
        "exit_code": 0 if fault == "previous_exit" else 1, "three_leases_free": True}
    previous: dict[str, Any] = {"terminal": previous_terminal_ref, "closure": seal(tmp_path / "previous-CPU-close.json", {
        "terminal": previous_terminal_ref, "admission": previous_owner["admission"], "recovery_plan": proof["recovery_plan"], "closure": previous_closure})}
    monkeypatch.setattr(ruler, "K0_NO_CLIP_PREVIOUS_CPU_FAILURE", previous)
    selection = seal(tmp_path / "manual-corrective-selection.json", {
        "schema": "ROOT_D1_T1_ONE_MANUAL_NO_CLIP_AUDITOR_CORRECTION_SOURCE_SELECTION_V1", "execution_authorized": False,
        "original_selection": ruler.K0_RECOVERY_SELECTION, "original_GPU_trigger_unchanged": read(ruler.K0_RECOVERY_SELECTION)["trigger"],
        "failed_CPU_attempt": {**previous, "wall_seconds": 6.470659800921567}})
    selection_review = seal(tmp_path / "manual-corrective-selection-review.json", {
        "status": "ACCEPT_KL_D1_T1_K0_NO_CLIP_CORRECTIVE_AUDIT_SOURCE_SELECTION_NO_FINDINGS_V1", "selection": selection,
        "findings": ["fictional refusal"] if fault == "manual_review" else []})
    monkeypatch.setattr(ruler, "K0_NO_CLIP_SELECTION", selection)
    monkeypatch.setattr(ruler, "K0_NO_CLIP_SELECTION_REVIEW", selection_review)
    authority_path = str(tmp_path / "corrective-authority.json")
    publication_path = str(tmp_path / "corrective-publication.json")
    monkeypatch.setattr(ruler, "K0_NO_CLIP_AUTHORITY_PATH", authority_path)
    monkeypatch.setattr(ruler, "K0_NO_CLIP_PUBLICATION_PATH", publication_path)
    helper: dict[str, Any] = {"path": "/fiction/no-clip-corrective-audit.py", "sha256": ruler.K0_NO_CLIP_HELPER}
    source_review = seal(tmp_path / "no-clip-source-review.json", {
        "status": "ACCEPT_KL_D1_K0_NO_CLIP_CORRECTIVE_AUDIT_SOURCE_ONLY_NO_FINDINGS_V1", "helper": helper,
        "source_pins": proof["source_pins"], "findings": []})
    monkeypatch.setattr(ruler, "K0_NO_CLIP_SOURCE_REVIEW", source_review)
    authority = read(old_root["recovery_authority"])
    authority.update(status="ROOT_AUTHORIZES_KL_D1_T1_K0_NO_CLIP_CORRECTIVE_AUDIT_V1", auditor=helper, auditor_source_review=source_review)
    authority_ref = seal(Path(authority_path), authority)
    publication = seal(Path(publication_path), {
        "status": "ROOT_RECORDS_PUBLIC_IMMUTABLE_NO_CLIP_CORRECTION_PR_V1", "corrective_selection": selection,
        "design": read(ruler.K0_RECOVERY_SELECTION)["design"], "pull_request": "https://github.com/jjoshua2/DeepFin/pull/9999",
        "head": "a" * 40, "tree": "b" * 40, "selection_review": selection_review, "execution_authorized": False,
        "document_path": "docs/experiments/fictional-manual-correction.md", "document_sha256": "e" * 64,
        "independent_source_review": seal(tmp_path / "manual-record-source-review.json", {"findings": ["fictional refusal"] if fault == "record_review" else []})})
    extra: dict[str, Any] = {"corrective_source_selection": selection, "corrective_source_selection_review": selection_review,
        "previous_CPU_failure": previous, "corrective_publication": publication}
    proof_path = tmp_path / "corrective-proof.json"
    plan = copy.deepcopy(old_plan)
    plan.update(schema="KL_D1_K0_SAVED_ENDPOINT_RECOVERY_PLAN_V2", status="BOUND_KL_D1_K0_SAVED_ENDPOINT_RECOVERY_NOT_DELIVERY_AUTHORITY_V2",
        authority=authority_ref, helper=helper, source_review=source_review, observation=str(proof_path),
        limits={**old_plan["limits"], "outer_seconds": 590, "inner_seconds": 560})
    plan_ref = seal(tmp_path / "corrective-plan.json", plan)
    ad_path = tmp_path / "corrective-CPU-admission.json"
    command = ["/fiction/python", "-I", "-B", helper["path"], "--recovery-plan", plan_ref["path"],
        "--recovery-plan-sha256", plan_ref["sha256"], "--admission", str(ad_path), "--admission-sha256", "SELF_ADMISSION_SHA256"]
    admission: dict[str, Any] = {"status": "ROOT_ADMITS_KL_D1_T1_K0_NO_CLIP_CORRECTIVE_AUDIT_CPU_V1", "execution_authorized": True,
        "recovery_plan": plan_ref, "authority": authority_ref, "resource": "cpu", "seconds": 590, "owner": {"sha256": "e" * 64},
        "normalized_child": list(command), "helper": helper, "source_review": source_review, "source_pins": proof["source_pins"], **extra}
    if fault == "manual_binding":
        admission["corrective_source_selection"] = {**selection, "sha256": "0" * 64}
    admission_ref = seal(ad_path, admission)
    command[command.index("--admission-sha256") + 1] = admission_ref["sha256"]
    terminal: dict[str, Any] = {**previous_terminal, "status": "PASS", "returncode": 0, "wall_seconds": 2.0,
        "resource": "cpu", "seconds": 600 if fault == "old600" else 590, "command": command}
    owner: dict[str, Any] = {"terminal": seal(tmp_path / "corrective-CPU-terminal.json", terminal), "admission": admission_ref,
        "command": command, "source_revision": previous_owner["source_revision"], "cwd": previous_owner["cwd"],
        "closure": {**previous_owner["closure"], "exit_code": 0}}
    proof.update(status=ruler.K0_NO_CLIP_PENDING, payload_schema=ruler.K0_NO_CLIP_SCHEMA, helper=helper, source_review=source_review,
        recovery_authority=authority_ref, recovery_plan=plan_ref, outer_seconds=590, absolute_inner_seconds=560,
        saved_grad_clip_mode="none", saved_zclip_state={"initialized": False, "mean": None, "var": None, "buffer": [], "warmup_steps": 25},
        donor_provenance={"mode": "fresh_rl_complete_v1", "sha256": read(old_root["contract"])["initial_model"]["sha256"],
            "architecture": read(old_root["contract"])["architecture"]}, **extra)
    if fault == "helper":
        proof["helper"] = {**helper, "sha256": ruler.K0_RECOVERY_HELPER}
    elif fault == "clip_active":
        proof["saved_zclip_state"]["initialized"] = True
    elif fault == "clip_mean":
        proof["saved_zclip_state"]["mean"] = 0.0
    elif fault == "clip_buffer":
        proof["saved_zclip_state"]["buffer"] = [0.0]
    elif fault == "clip_warmup":
        proof["saved_zclip_state"]["warmup_steps"] = 25.0
    elif fault == "clip_mode":
        proof["saved_grad_clip_mode"] = "zclip"
    root: dict[str, Any] = {**old_root, "status": ruler.K0_NO_CLIP_ROOT, "recovery_authority": authority_ref,
        "payload_audit": {"schema": ruler.K0_NO_CLIP_SCHEMA, "readout": seal(proof_path, proof), "owner": owner}}
    root_ref = seal(tmp_path / "corrective-root.json", root)
    review: dict[str, Any] = {**root, "status": ruler.K0_NO_CLIP_REVIEW, "root_acceptance": root_ref, "findings": []}
    return {"root": root_ref, "independent": seal(tmp_path / "corrective-review.json", review)}, train


def test_no_clip_corrective_recovery_real_delivery_and_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding, train = no_clip_corrective_graph(tmp_path, monkeypatch)
    metadata = ruler.Metadata(1 << 20, 32 << 20)
    model = ruler.model_delivery(metadata, binding, "K0", _d1=True)
    ruler.same_gpu_game_coverage(metadata, binding, model, train, _d1=True)
    assert model["step"] == 4096
    assert len(train.games) == 192
    root = metadata.read(binding["root"])
    assert metadata.observed(root["learner_owner"]["terminal"])["status"] == "FAILED"
    assert metadata.observed(ruler.K0_NO_CLIP_PREVIOUS_CPU_FAILURE["terminal"])["status"] == "FAILED"
    assert metadata.observed(root["payload_audit"]["owner"]["terminal"])["seconds"] == 590
    metadata.finish()


@pytest.mark.parametrize(("fault", "message"), [
    ("previous_status", "K0 previous failed CPU differs"),
    ("previous_exit", "K0 previous failed CPU differs"),
    ("manual_review", "K0 manual correction source selection differs"),
    ("manual_binding", "K0 corrective decision bindings differ"),
    ("record_review", "K0 correction record source review differs"),
    ("old600", "recovery CPU owner not exact closed PASS"),
    ("helper", "unsupported K0 recovery auditor"),
    ("clip_active", "K0 exact no-clip saved state differs"),
    ("clip_mean", "K0 exact no-clip saved state differs"),
    ("clip_buffer", "K0 exact no-clip saved state differs"),
    ("clip_warmup", "K0 exact no-clip saved state differs"),
    ("clip_mode", "K0 exact no-clip saved state differs"),
])
def test_rehashed_no_clip_corrective_metadata_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, message: str) -> None:
    binding, _ = no_clip_corrective_graph(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError, match=message):
        ruler.model_delivery(ruler.Metadata(1 << 20, 32 << 20), binding, "K0", _d1=True)
