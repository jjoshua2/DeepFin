"""Fictional trusted delivery metadata plus real synthetic tensor fingerprint.

No producer/actor/Root acceptance is manufactured outside this test fixture.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts import core_closed_once_actor as bridge
from scripts import core_finite_collect as collect
from scripts.core_selfplay_union_stored_row_once_proof import fingerprint
from tests.test_core_finite_learner import frozen


def actor_fixture(tmp_path: Path, *, family: str = "global_union", architecture: dict[str, Any] | None = None,
                  checkpoint_ref: dict[str, str] | None = None, full_state: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the actual Root9/ind11/tag shapes; no bare compatibility flag."""
    tmp_path.mkdir(exist_ok=True)
    state = ({"tensor." + str(i): torch.tensor([float(i)]) for i in range(494)} if full_state
             else {"left": torch.ones(2), "right": torch.zeros(1)})
    state_hash = fingerprint(state, torch)
    arch = architecture or {"kind": "tiny"}
    selected = bridge.FAMILIES[family]
    method = frozen(tmp_path / "method.json", {"fictional_test": True})
    c = {"schema": selected["contract"], "architecture": arch,
         "method": method, "study_uuid": "11111111-1111-4111-8111-111111111111",
         "run_uuid": "22222222-2222-4222-8222-222222222222", "source_revision": "a" * 40,
         "sampling": {"fictional_test": True}, "updates_per_segment": [2], "population": {"rows": 3}}
    if family == "policy_anchor":
        c["policy_anchor"] = {"schema": "frozen_reference_policy_anchor_v1", "coefficient": 1.0,
            "direction": "reference_to_student_forward_KL", "mask": "original_network_turn_has_policy_with_explicit_legal_mask_v1",
            "reference": {"checkpoint": {"path": str(tmp_path / "teacher.pt"), "sha256": "e" * 64},
                          "architecture": arch, "checkpoint_step": 9, "historical_source_revision": None}}
    cref = frozen(tmp_path / "contract.json", c)
    if checkpoint_ref is None:
        path = tmp_path / "learner.pt"
        path.write_bytes(b"inert owned synthetic actor; not a production model")
        path.chmod(0o444)
        checkpoint_ref = {"path": str(path), "sha256": collect.digest(path)}
    checkpoint = checkpoint_ref
    # The boundary must live beside the actual declared actor file.
    stage = frozen(tmp_path / "actual-stage.json", {"output": str(Path(checkpoint["path"]).parent),
                   "contract_sha256": cref["sha256"], "segment": 1, "population": c["population"]})
    certificate = {"schema": selected["certificate"], "contract": cref,
        "stage": stage, "checkpoint": checkpoint, "actual_calls": {"forward": 2, "optimizer_pre": 2,
        "optimizer_post": 2, "save": 1}, "save_before_equals_after": True, "extra_forwards": 0,
        "additional_final_CPU_checkpoint_decodes": 1, "saved_model_state_sha256": state_hash,
        "saved_model": {"finite": True, "model_state_sha256": state_hash}}
    b = {"status": selected["boundary"], "pass_complete": True,
         "contract_sha256": cref["sha256"], "stage": stage, "method": method, "study_uuid": c["study_uuid"],
         "run_uuid": c["run_uuid"], "source_revision": c["source_revision"], "segment": 1, "step": 2,
         "checkpoint": checkpoint, "own_live_certificate": certificate, "quota_state": {"cursor": 3, "tainted": False},
         "draw_journals": [], "quota_ledger_checkpoint": {}, "population_identity": {}, "source_origin_binding": {}}
    bpath = Path(checkpoint["path"]).parent / "boundary.json"
    if bpath.exists():
        bpath.unlink()  # This fixture's earlier fictional bank boundary only.
    bref = frozen(bpath, b)
    command = ["python", "-m", selected["module"], "--contract", cref["path"],
               "--contract-sha256", cref["sha256"], "--stage", stage["path"], "--stage-sha256", stage["sha256"]]
    terminal = frozen(tmp_path / "terminal.json", {"status": "PASS", "returncode": 0,
        "command": command, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": "d" * 64})
    binding = {"boundary": bref, "owner_terminal": terminal, "owner_sha256": "d" * 64, "command": command}
    owned: dict[str, Any] = {}
    source = frozen(tmp_path / "source.json", {"owned_source_files": owned})
    review = frozen(tmp_path / "review.json", {"status": ("ACCEPT_HISTORICAL_SELFPLAY_UNIFORM_ONCE_ENTRY_SOURCE_ONLY_NO_FINDINGS_V1"
                    if family == "global_union" else selected["source_status"]),
                    "source_manifest": source, "owned_source_files": owned, "findings": []})
    identity = {k: c[k] for k in ("run_uuid", "study_uuid", "source_revision", "sampling", "method")}
    identity.update(contract_sha256=cref["sha256"], segment=1)
    if family == "policy_anchor":
        identity["policy_anchor"] = c["policy_anchor"]
    tag = {"identity": identity, **{k: b[k] for k in ("quota_state", "draw_journals", "quota_ledger_checkpoint",
           "population_identity", "source_origin_binding")}}
    model = {"checkpoint": checkpoint, "architecture": arch, "step": 2, "model_state_sha256": state_hash,
             "model_state_hash_schema": "union_typed_tensor_tree_sha256_v1", "label": "fixture",
             "tag_key": selected["tag"], "tag": tag}
    root_status = next(iter(selected["deliveries"]))
    root = {"status": root_status, "study_uuid": c["study_uuid"],
            "method": method, "contract": cref, "boundary": bref, "closed_binding": binding,
            "entry_source": source, "entry_source_review": review, "model": model}
    rref = frozen(tmp_path / "root.json", root)
    independent = dict(root, status=selected["deliveries"][root_status],
                       root_acceptance=rref, findings=[])
    iref = frozen(tmp_path / "independent.json", independent)
    actor = {"kind": bridge.KIND, **checkpoint, "expected_checkpoint_step": 2, "architecture": arch,
             "predecessor": binding, "lineage": {"schema": bridge.SCHEMA, "family": family,
             "delivery": {"root": rref, "independent": iref}}}
    payload = {"model": state, selected["tag"]: tag, "arch": arch, "step": 2}
    if family == "policy_anchor":
        payload["frozen_reference_policy_anchor_v1"] = c["policy_anchor"]
    return actor, payload


def test_closed_actor_preserves_truthful_new_tag_and_real_tensor_fingerprint(tmp_path: Path) -> None:
    actor, payload = actor_fixture(tmp_path)
    joined = bridge.validate_actor(actor)
    assert joined["model"]["tag_key"] == "core_selfplay_union_stored_row_once_v1"
    assert joined["boundary"]["pass_complete"] is True
    bridge.validate_payload(actor, payload, torch)
    payload["model"]["left"][0] += 1
    with pytest.raises(ValueError, match="actual payload tag/typed state differs"):
        bridge.validate_payload(actor, payload, torch)


@pytest.mark.parametrize("fault", ["step", "checkpoint", "architecture", "parent", "family", "extra_field"])
def test_actor_alias_tampering_never_becomes_a_legacy_envelope(tmp_path: Path, fault: str) -> None:
    actor, _ = actor_fixture(tmp_path)
    if fault == "step":
        actor["expected_checkpoint_step"] += 1
    elif fault == "checkpoint":
        actor["sha256"] = "0" * 64
    elif fault == "architecture":
        actor["architecture"] = {"kind": "other"}
    elif fault == "parent":
        actor["predecessor"] = None
    elif fault == "family":
        actor["lineage"]["family"] = "legacy"
    else:
        actor["compatibility"] = True
    with pytest.raises(ValueError, match="closed once actor"):
        bridge.validate_actor(actor)


def test_same_shaped_but_unrelated_independent_receipt_is_refused(tmp_path: Path) -> None:
    actor, _ = actor_fixture(tmp_path)
    pair = actor["lineage"]["delivery"]
    independent = copy.deepcopy(bridge.read(pair["independent"]))
    independent["root_acceptance"]["sha256"] = "0" * 64
    pair["independent"] = frozen(tmp_path / "unrelated-independent.json", independent)
    with pytest.raises(ValueError, match="trusted delivery joins"):
        bridge.validate_actor(actor)


def test_actual_new_tag_cannot_hide_an_old_family_or_modified_tag(tmp_path: Path) -> None:
    actor, payload = actor_fixture(tmp_path)
    payload["core_generation_bank_learner_v1"] = {"fake": True}
    # Explicitly exclude older learner envelopes as well as other once tags.
    with pytest.raises(ValueError, match="actual payload tag/typed state differs"):
        bridge.validate_payload(actor, payload, torch)


@pytest.mark.parametrize("family", ["global_union", "policy_anchor"])
def test_real_collector_validator_dispatches_explicit_closed_actor_without_legacy_labels(tmp_path: Path, family: str) -> None:
    from tests.test_core_finite_collect import protocol, roster

    p = protocol(tmp_path)
    actor, _ = actor_fixture(tmp_path / "actor", family=family, architecture=p["actor"]["architecture"])
    p["actor"] = actor
    collect.validate_protocol(p, roster())
    assert p["generation"] == 1
    assert p["actor"]["kind"] == bridge.KIND
    assert p["actor"]["sha256"] != collect.INITIAL_DONOR_SHA256
    p["actor"]["expected_checkpoint_step"] += 1
    with pytest.raises(ValueError, match="selected checkpoint identity"):
        collect.validate_protocol(p, roster())


@pytest.mark.parametrize(("family", "fault"), [
    ("global_union", "none"), ("global_union", "tag"), ("global_union", "state"), ("global_union", "legacy"),
    ("policy_anchor", "none"), ("policy_anchor", "tag"), ("policy_anchor", "state"), ("policy_anchor", "reference"),
])
def test_actual_collector_single_load_hook_checks_new_tag_and_state_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, fault: str,
) -> None:
    from chess_anti_engine.uci import model_loader
    from tests.test_core_finite_collect import protocol, roster

    p = protocol(tmp_path)
    actor, payload = actor_fixture(tmp_path / "actor", family=family, architecture=p["actor"]["architecture"], full_state=True)
    p["actor"] = actor
    collect.validate_protocol(p, roster())
    if fault == "tag":
        payload[bridge.FAMILIES[family]["tag"]]["identity"]["segment"] = 2
    elif fault == "state":
        payload["model"]["tensor.0"].add_(1)
    elif fault == "legacy":
        payload[collect.BANK_CHECKPOINT_KEY] = {"unsupported": True}
    elif fault == "reference":
        payload["frozen_reference_policy_anchor_v1"] = {"wrong": True}
    loads: list[bool] = []
    builds: list[bool] = []
    def load(*_args: Any, **_kwargs: Any) -> Any:
        loads.append(True)
        return payload
    class InertActor:
        def state_dict(self) -> dict[str, Any]:
            return payload["model"]
        def parameters(self) -> list[Any]:
            return [SimpleNamespace(device="cuda:0")]
    def build(_configuration: Any) -> InertActor:
        builds.append(True)
        return InertActor()
    def public_loader(path: Path, **_kwargs: Any) -> Any:
        # The production collector interception is real; this CPU fixture makes
        # no real model/CUDA allocation and does not replace its validation.
        torch.load(path, map_location="cpu", weights_only=True)
        return model_loader.build_model(actor["architecture"])
    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(model_loader, "build_model", build)
    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", public_loader)
    if fault == "none":
        _, serving = collect.load_actor(p)
        assert serving["closed_once_actor_lineage"] == actor["lineage"]
        assert serving["closed_once_model_state_sha256"] == fingerprint(payload["model"], torch)
        assert serving["predecessor_boundary"] == actor["predecessor"]["boundary"]
        assert serving["pre_state_sha256"] == serving["checkpoint_model_state_sha256"]
        assert serving["model_build_count"] == 1
        assert builds == [True]
    else:
        with pytest.raises(ValueError, match=r"actual payload tag/typed state|actual frozen reference identity"):
            collect.load_actor(p)
        assert builds == []
    assert loads == [True]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(("family", "fault"), [
    ("global_union", "none"), ("policy_anchor", "none"),
    ("policy_anchor", "lineage"), ("policy_anchor", "typed_state"), ("policy_anchor", "source_pin"),
])
def test_actual_recipe_free_data_auditor_transports_closed_actor_lineage_into_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, fault: str,
) -> None:
    from scripts import core_finite_admit as audit
    from scripts import core_generation_data_admit as data
    from tests.test_core_bank_actor_lineage import audit_case, replace
    from tests.test_core_finite_admit import bind_data_case

    f = audit_case(tmp_path, monkeypatch)
    p, case, output = f.producer, f.case, f.producer_root
    old_actor = p["actor"]
    actor, _ = actor_fixture(tmp_path / "closed-once", family=family, architecture=old_actor["architecture"],
                            checkpoint_ref={"path": old_actor["path"], "sha256": old_actor["sha256"]})
    p["actor"] = actor
    helper = "scripts/core_closed_once_actor.py"
    case.spec["source_files"][helper] = audit.learning.file_sha(audit.learning.REPO / helper)
    p["source"]["sha256"][helper] = case.spec["source_files"][helper]
    if fault == "source_pin":
        p["source"]["sha256"][helper] = "0" * 64
    pref = frozen(tmp_path / "closed-once-producer.json", p)
    case.protocol["producer_protocol"] = pref
    serving = f.serving
    serving.pop("bank_learner_tag")
    selected = bridge.validate_actor(actor)["model"]
    serving.update(actor_kind=bridge.KIND, closed_once_actor_lineage=actor["lineage"],
        predecessor_boundary=actor["predecessor"]["boundary"], closed_once_model_state_sha256=selected["model_state_sha256"])
    if fault == "lineage":
        serving["closed_once_actor_lineage"] = {"unsupported": True}
    elif fault == "typed_state":
        serving["closed_once_model_state_sha256"] = "0" * 64
    _, readout = audit.learning.read_ref(case.protocol["producer_readout"])
    output.chmod(0o755)
    readout.update(protocol_sha256=pref["sha256"], actor_serving=replace(audit.ref(output / "actor-serving.json"), serving))
    case.protocol["producer_readout"] = replace(case.protocol["producer_readout"], readout)
    admission = frozen(tmp_path / "closed-once-producer-admission.json", {"status": "ROOT_ADMITS_CORE_GENERATION_V1",
        "protocol_sha256": pref["sha256"], "run_uuid": p["run_uuid"], "generation": p["generation"],
        "source_revision": p["source"]["revision"], "actor_sha256": actor["sha256"]})
    command = ["python", "-m", "scripts.core_finite_collect", "--execute", "--protocol", pref["path"],
        "--protocol-sha256", pref["sha256"], "--admission", admission["path"], "--admission-sha256", admission["sha256"]]
    _, terminal = audit.learning.read_ref(case.protocol["producer_owner"]["terminal"])
    terminal["command"] = command
    case.protocol["producer_owner"].update(terminal=frozen(tmp_path / "closed-once-producer-terminal.json", terminal), command=command)
    case.protocol["producer_inventory"] = frozen(tmp_path / "closed-once-inventory.json",
        {"path": str(output), "files": collect.inventory(output, seal=True)})
    aref = bind_data_case(tmp_path, case)
    if fault != "none":
        with pytest.raises(ValueError, match="closed once actor"):
            data.execute(aref)
        assert not (Path(case.protocol["output_root"]) / "generation.json").exists()
        return
    result = data.execute(aref)
    _, candidate = audit.learning.read_ref(result["generation_manifest"])
    assert result["status"] == "AUDITED_CORE_DATA_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1"
    assert candidate["actor_sha256"] == actor["sha256"]
    assert candidate["actor_checkpoint_step"] == 2
    assert candidate["actor_serving"] == readout["actor_serving"]
    assert candidate["learning_admitted"] is False
    # Continue through the actual accepted-bank consumer with explicitly
    # fictional postclosure authorities; do not promote the candidate itself.
    from scripts import core_generation_bank_acceptance as bank_api

    audit_command = ["/python", "-B", "-m", "scripts.core_generation_data_admit", "--protocol", aref["path"],
                     "--protocol-sha256", aref["sha256"]]
    source = {"revision": case.spec["source_revision"], "checkout": str(audit.learning.REPO),
              "source_files": case.spec["source_files"]}
    terminal = frozen(tmp_path / "fictional-audit-terminal.json", {"status": "PASS", "returncode": 0,
        "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED", "resource": "cpu", "seconds": 300,
        "wall_seconds": 1.0, "supervisor_sha256": "d" * 64, "command": audit_command, "source_revision": "a" * 40,
        "cwd": "/synthetic-owner-checkout", "known_processes": {"123": 456}, "supervisor_pid": 122})
    admission = frozen(tmp_path / "fictional-audit-admission.json", {"status": bank_api.ADMISSION_STATUS,
        "audit_protocol": aref, "audit_source": source, "command": audit_command,
        "owner": case.protocol["runtime"]["owner"], "resource": "cpu", "seconds": 300})
    root = {"status": bank_api.ROOT_STATUS, "audit_protocol": aref, "data_spec": case.protocol["data_spec"],
        "readout": audit.ref(Path(case.protocol["output_root"]) / "readout.json"), "candidate": result["generation_manifest"],
        "audit_source": source, "learning_admitted": False, "audit_owner": {"terminal": terminal, "admission": admission,
        "command": audit_command, "source_revision": "a" * 40, "cwd": "/synthetic-owner-checkout",
        "closure": {"supported_eof": True, "exit_code": 0, "supervisor_closed": True, "owned_births_closed": True,
                    "supervisor_pid": 122, "known_processes": {"123": 456}}}}
    rref = frozen(tmp_path / "fictional-bank-root.json", root)
    iref = frozen(tmp_path / "fictional-bank-independent.json", {"status": bank_api.REVIEW_STATUS, "root_acceptance": rref,
        "findings": [], **{k: root[k] for k in ("audit_protocol", "data_spec", "readout", "candidate")}})
    accepted = bank_api.validate_accepted_generation(rref, iref, max_file_bytes=4 << 20, max_total_bytes=16 << 20, max_games=1)
    assert accepted.identity["actor_sha256"] == actor["sha256"]
    assert accepted.identity["actor_checkpoint_step"] == 2
    assert accepted.identity["generation"] == p["generation"]
    assert len(accepted.games) == 1
    assert accepted.evidence["learning_admitted"] is False

