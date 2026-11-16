"""Fictional software fixtures, real shared raw/Zarr/history/TB audit bodies.

No real actor/bank, collection, research cohort or postclosure authority credit.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, cast

import pytest

from scripts import core_finite_admit as audit
from scripts import core_finite_collect as collect
from scripts import core_finite_holdout_collect as hold_collect
from scripts import core_generation_bank_acceptance as train_view
from scripts import core_generation_bank_learner as learner
from scripts import core_generation_data_admit as train_audit
from scripts import core_generation_holdout_acceptance as hold_view
from scripts import core_generation_holdout_admit as hold_audit
from tests.test_core_bank_actor_lineage import audit_case, bank_predecessor, replace
from tests.test_core_finite_admit import bind_data_case, entry_case, frozen
from tests.test_core_policy_anchor_actor import project
from tests.test_core_finite_collect import protocol, roster
from tests.test_core_generation_bank_acceptance import graph

LIMITS = {"max_file_bytes": 1 << 20, "max_total_bytes": 8 << 20, "max_games": 3}


def test_holdout_descriptor_has_separate_schema_bank_and_original_actor(tmp_path: Path):
    p, games = protocol(tmp_path), roster()
    p["schema"] = collect.HOLDOUT_PROTOCOL
    for game in games:
        game["bank"] = "holdout"
    collect.validate_holdout_protocol(p, games)
    assert p["actor"]["expected_checkpoint_step"] == 680276
    with pytest.raises(ValueError, match="wrong generation protocol schema"):
        collect.validate_protocol(p, games)
    p["schema"] = "CORE_FINITE_GENERATION_V1"
    with pytest.raises(ValueError, match="wrong generation protocol schema"):
        collect.validate_holdout_protocol(p, games)
    with pytest.raises(ValueError, match="fixed TRAIN"):
        collect.validate_protocol(p, games)


@pytest.mark.parametrize("fault", ["train_bank", "order", "duplicate_history", "actor", "physical_id"])
def test_holdout_does_not_relax_roster_or_actor_guards(tmp_path: Path, fault: str):
    p, games = protocol(tmp_path), roster()
    p["schema"] = collect.HOLDOUT_PROTOCOL
    for game in games:
        game["bank"] = "holdout"
    if fault == "train_bank":
        games[0]["bank"] = "train"
    elif fault == "order":
        games.reverse()
    elif fault == "duplicate_history":
        games[1]["opening_moves"] = games[0]["opening_moves"]
    elif fault == "actor":
        p["actor"]["sha256"] = "0" * 64
    else:
        games[0]["physical_game_id"] = "foreign:0"
    with pytest.raises(ValueError, match=r"fixed HOLDOUT|duplicate planned|immutable A39"):
        collect.validate_holdout_protocol(p, games)


@pytest.mark.parametrize("missing", ["base", "hold_entry"])
def test_holdout_source_guard_requires_both_old_closure_and_new_entry(tmp_path: Path, monkeypatch, missing: str):
    source = tmp_path / "source"
    monkeypatch.setattr(collect, "__file__", str(source / "scripts/core_finite_collect.py"))
    monkeypatch.setattr(collect.subprocess, "check_output", lambda argv, **_k: b"a" * 40 if "rev-parse" in argv else b"")
    p = {"schema": collect.HOLDOUT_PROTOCOL, "actor": {"kind": "donor"},
         "source": {"checkout": str(source), "revision": "a" * 40,
                    "sha256": {} if missing == "base" else dict.fromkeys(collect.REQUIRED_SOURCE_FILES, "b" * 64)}}
    with pytest.raises(ValueError, match=r"incomplete.*source fingerprint"):
        collect.holdout_source_guard(p)


def test_holdout_public_descriptor_entry_checks_without_actor_or_engine(tmp_path: Path, monkeypatch):
    case = entry_case(tmp_path, monkeypatch, holdout=True)
    producer = json.loads(Path(case.protocol["producer_protocol"]["path"]).read_text())
    producer["output_root"] = str(tmp_path / "new-uncreated-producer")
    pref = frozen(tmp_path / "check-producer.json", producer)
    report = tmp_path / "descriptor-report.json"
    monkeypatch.setattr(sys, "argv", ["hold-collect", "--check", "--protocol", pref["path"],
                                      "--protocol-sha256", pref["sha256"], "--report", str(report)])
    assert hold_collect.main() == 0
    result = json.loads(report.read_text())
    assert result["status"] == "CHECKED_CORE_HOLDOUT_GENERATION_DESCRIPTOR_V1"
    assert result["models_created"] is False
    assert result["engines_created"] is False
    assert result["execution_admitted"] is False
    assert not Path(producer["output_root"]).exists()


def accepted_audit(tmp_path: Path, case: Any, pref: dict[str, str], result: dict[str, Any]):
    """Explicit fictional postclosure owner/issuers around actual emitted audit."""
    p = case.protocol
    command = [p["runtime"]["python"], "-B", "-m", "scripts.core_generation_holdout_admit",
               "--protocol", pref["path"], "--protocol-sha256", pref["sha256"]]
    source = {"revision": case.spec["source_revision"], "checkout": str(audit.learning.REPO), "source_files": case.spec["source_files"]}
    terminal = frozen(tmp_path / "audit-terminal.json", {"status": "PASS", "returncode": 0, "owned_processes_closed": True,
        "custody": "OWNED_PROCESSES_CLOSED", "resource": "cpu", "seconds": 60, "wall_seconds": 1.0,
        "supervisor_sha256": "e" * 64, "command": command, "source_revision": "f" * 40, "cwd": "/fictional-owner-checkout",
        "known_processes": {"2147483646": 1}, "supervisor_pid": 2147483645})
    admission = frozen(tmp_path / "audit-admission.json", {"status": hold_view.ADMISSION_STATUS, "audit_protocol": pref,
        "audit_source": source, "command": command, "owner": p["runtime"]["owner"], "resource": "cpu", "seconds": 60})
    root = {"status": hold_view.ROOT_STATUS, "audit_protocol": pref, "data_spec": p["data_spec"],
        "readout": audit.ref(Path(p["output_root"]) / "readout.json"), "candidate": result["generation_manifest"],
        "audit_source": source, "learning_admitted": False, "evaluation_only": True,
        "audit_owner": {"terminal": terminal, "admission": admission, "command": command, "source_revision": "f" * 40,
            "cwd": "/fictional-owner-checkout", "closure": {"supported_eof": True, "exit_code": 0, "supervisor_pid": 2147483645,
                "supervisor_closed": True, "owned_births_closed": True, "known_processes": {"2147483646": 1}}}}
    rref = frozen(tmp_path / "root-HOLD.json", root)
    iref = frozen(tmp_path / "independent-HOLD.json", {"status": hold_view.REVIEW_STATUS, "root_acceptance": rref,
        "findings": [], "evaluation_only": True, **{key: root[key] for key in ("audit_protocol", "data_spec", "readout", "candidate")}})
    return rref, iref


def test_public_holdout_audit_real_payload_to_readonly_view_and_learner_refusal(tmp_path: Path, monkeypatch, capsys):
    case = entry_case(tmp_path, monkeypatch, holdout=True)
    case.protocol["runtime"] = {"owner": {"path": "/fictional-owner.py", "sha256": "e" * 64}, "seconds": 60,
        "inner_seconds": 30, "cpu_affinity": [20, 22], "python": "/python"}
    pref = bind_data_case(tmp_path, case)
    assert hold_audit.main(["--protocol", pref["path"], "--protocol-sha256", pref["sha256"]]) == 0
    output = Path(case.protocol["output_root"])
    result = json.loads((output / "readout.json").read_text())
    assert result["status"] == "AUDITED_CORE_HOLDOUT_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1"
    assert result["models_constructed"] == result["forwards"] == result["optimizer_updates"] == 0
    assert result["evaluation_only"] is True
    assert "HOLDOUT" in capsys.readouterr().out
    candidate_path = Path(result["generation_manifest"]["path"])
    before = candidate_path.read_bytes()
    candidate = json.loads(before)
    assert candidate["schema"] == audit.HOLDOUT_CANDIDATE
    assert candidate["candidate_only"] is True
    assert candidate["learning_admitted"] is False
    assert candidate["evaluation_only"] is True
    wrapper = json.loads(Path(candidate["games"][0]["producer_proof"]["path"]).read_text())
    assert wrapper["status"] == audit.HOLDOUT_GAME_CANDIDATE
    assert wrapper["bank"] == candidate["games"][0]["bank"] == result["audited"][0]["bank"] == "holdout"
    assert wrapper["evaluation_only"] is True
    root, review = accepted_audit(tmp_path, case, pref, result)
    view = hold_view.validate_accepted_holdout(root, review, **LIMITS)
    assert isinstance(view, hold_view.AcceptedHoldout)
    assert not isinstance(view, train_view.AcceptedGeneration)
    assert view.identity["bank"] == "holdout"
    assert view.evidence["evaluation_only"] is True
    assert view.identity["actor_checkpoint_step"] == 7
    assert view.games[0]["game_id"] == case.fixture.gid
    with pytest.raises(TypeError, match="does not support item assignment"):
        cast(Any, view.games[0])["bank"] = "train"
    assert candidate_path.read_bytes() == before
    with pytest.raises(ValueError, match="invalid root acceptance"):
        train_view.validate_accepted_generation(root, review, **LIMITS)
    with pytest.raises(ValueError, match="invalid root acceptance"):
        learner.metadata({"updates_per_segment": [1], "bank_limits": LIMITS},
            {"schema": learner.STAGE, "contract_sha256": "a" * 64, "segment": 1,
             "banks": [{"root_acceptance": root, "independent_review": review}], "parent": None, "output": "unused"},
            {"path": "/unused/contract.json", "sha256": "a" * 64})
    for entry in (audit.execute, train_audit.execute):
        with pytest.raises(ValueError, match="invalid audit protocol"):
            entry(pref)
    assert all(owner.closed for owner in case.fixture.owners)


@pytest.mark.parametrize("fault", ["input", "policy", "outcome"])
def test_holdout_uses_actual_raw_stored_and_outcome_verification(tmp_path: Path, monkeypatch, fault: str):
    case = entry_case(tmp_path, monkeypatch, holdout=True, corruption=fault)
    pref = bind_data_case(tmp_path, case)
    with pytest.raises(ValueError, match=r"differs|bytes|outcome|onehot"):
        hold_audit.execute(pref)
    assert audit.learning.failure_marker(Path(case.protocol["output_root"])).exists()


@pytest.mark.parametrize("fault", ["owner_unclosed", "readout_collection", "attempt_accounting", "serving_step",
                                  "producer_failure", "exclusions", "lifecycle", "producer_schema"])
def test_holdout_refuses_incomplete_or_crossed_producer_evidence(tmp_path: Path, monkeypatch, fault: str):
    case = entry_case(tmp_path, monkeypatch, holdout=True, fault=fault)
    with pytest.raises(ValueError, match=r"closed PASS|identity|account|serving|failed|partial|exposure|lifecycle|schema|closure"):
        hold_audit.execute(bind_data_case(tmp_path, case))


@pytest.mark.parametrize("field", ["run_uuid", "collection_uuid", "generation", "source_revision", "architecture",
                                  "actor_sha256", "actor_checkpoint_step"])
def test_holdout_spec_cannot_change_producer_identity(tmp_path: Path, monkeypatch, field: str):
    case = entry_case(tmp_path, monkeypatch, holdout=True)
    values = {"run_uuid": "33333333-3333-4333-8333-333333333333", "collection_uuid": "33333333-3333-4333-8333-333333333333",
              "generation": 3, "source_revision": "d" * 40, "architecture": {"different": True}, "actor_sha256": "e" * 64,
              "actor_checkpoint_step": 8}
    case.spec[field] = values[field]
    with pytest.raises(ValueError, match="specification/producer identity"):
        hold_audit.execute(bind_data_case(tmp_path, case))


@pytest.mark.parametrize("field", ["lr", "optimizer", "seed", "batch_size"])
def test_holdout_has_no_recipe_or_training_route(tmp_path: Path, monkeypatch, field: str):
    case = entry_case(tmp_path, monkeypatch, holdout=True)
    case.spec[field] = 1
    with pytest.raises(ValueError, match="data specification fields/schema"):
        hold_audit.execute(bind_data_case(tmp_path, case))


@pytest.mark.parametrize("fault", ["none", "wrong_family"])
def test_metadata_holdout_and_train_views_are_not_interchangeable(tmp_path: Path, fault: str):
    root, review, candidate = graph(tmp_path, holdout=fault == "none")
    if fault == "wrong_family":
        with pytest.raises(ValueError, match="invalid root acceptance"):
            hold_view.validate_accepted_holdout(root, review, **LIMITS)
        return
    before = candidate.read_bytes()
    view = hold_view.validate_accepted_holdout(root, review, **LIMITS)
    assert tuple(game["game_id"] for game in view.games) == (0, 2**63 - 1)
    assert all(game["bank"] == "holdout" for game in view.games)
    assert view.evidence["eligible_rows"] == 4
    assert candidate.read_bytes() == before


@pytest.mark.parametrize("fault", ["actor_step", "serving", "inventory", "row_gate", "duplicate_game", "wrapper", "wrapper_alias",
                                  "missing_attempt", "missing_eligible", "candidate_alias", "admitted", "totals", "command",
                                  "failed", "unclosed", "eof", "review", "id_string", "id_bool", "id_negative", "id_range"])
def test_rehashed_holdout_metadata_cannot_bypass_full_graph_checks(tmp_path: Path, fault: str):
    root, review, _ = graph(tmp_path, holdout=True, fault=fault)
    with pytest.raises(ValueError, match=r"differs?|invalid|changed|incomplete|candidate|owner|gate|row|bounds"):
        hold_view.validate_accepted_holdout(root, review, **LIMITS)


@pytest.mark.parametrize("anchored", [False, True])
@pytest.mark.parametrize("fault", ["none", "generation", "step", "old_kind"])
def test_holdout_keeps_exact_adjacent_actor_lineage(tmp_path: Path, anchored: bool, fault: str):
    p, games = protocol(tmp_path), roster()
    bank_predecessor(tmp_path, p)
    if anchored:
        project(p)
    p["schema"] = collect.HOLDOUT_PROTOCOL
    for game in games:
        game["bank"] = "holdout"
    assert p["source"]["revision"] != p["actor"]["lineage"]["source_revision"]
    if fault == "generation":
        p["generation"] = 4
    elif fault == "step":
        p["actor"]["expected_checkpoint_step"] = 3
    elif fault == "old_kind":
        p["actor"]["kind"] = "learner"
    if fault == "none":
        collect.validate_holdout_protocol(p, games)
        assert p["generation"] == 3
        assert p["actor"]["lineage"]["logical_generation"] == 2
        assert p["actor"]["lineage"]["segment"] == 1
        assert p["actor"]["expected_checkpoint_step"] == 2
    else:
        with pytest.raises(ValueError, match=r"adjacent|step|predecessor|previous-generation|closed.*parent|parent.*identity"):
            collect.validate_holdout_protocol(p, games)


@pytest.mark.parametrize("bad_serving", [False, True])
def test_holdout_actual_audit_preserves_distinct_producer_parent_source_and_steps(tmp_path: Path, monkeypatch, bad_serving: bool):
    context = audit_case(tmp_path, monkeypatch, holdout=True)
    case, producer = context.case, context.producer
    assert producer["schema"] == collect.HOLDOUT_PROTOCOL
    assert case.spec["actor_checkpoint_step"] == 2
    assert producer["generation"] == 3
    assert producer["actor"]["lineage"]["logical_generation"] == 2
    assert producer["source"]["revision"] != producer["actor"]["lineage"]["source_revision"]
    if bad_serving:
        _, serving = audit.learning.read_ref(audit.ref(context.producer_root / "actor-serving.json"))
        serving["actor_lineage"] = {**serving["actor_lineage"], "source_revision": "0" * 40}
        context.producer_root.chmod(0o755)
        serving_ref = replace(audit.ref(context.producer_root / "actor-serving.json"), serving)
        _, readout = audit.learning.read_ref(case.protocol["producer_readout"])
        readout["actor_serving"] = serving_ref
        case.protocol["producer_readout"] = replace(case.protocol["producer_readout"], readout)
        case.protocol["producer_inventory"] = frozen(tmp_path / "updated-bank-inventory.json",
            {"path": str(context.producer_root), "files": collect.inventory(context.producer_root, seal=True)})
        with pytest.raises(ValueError, match="bank actor serving lineage differs"):
            hold_audit.execute(bind_data_case(tmp_path, case))
    else:
        result = hold_audit.execute(bind_data_case(tmp_path, case))
        assert result["logical_generation"] == 3
        assert result["actor_checkpoint_step"] == 2
        assert result["audited"][0]["bank"] == "holdout"


@pytest.mark.parametrize("producer_holdout", [False, True])
def test_public_data_entries_refuse_crossed_protocol_before_audit(tmp_path: Path, monkeypatch, producer_holdout: bool):
    case = entry_case(tmp_path, monkeypatch, holdout=producer_holdout)
    pref = bind_data_case(tmp_path, case)
    entry = train_audit.execute if producer_holdout else hold_audit.execute
    with pytest.raises(ValueError, match="invalid audit protocol"):
        entry(pref)
    assert not Path(case.protocol["output_root"]).exists()


def test_rehashed_holdout_candidate_cannot_drop_evaluation_purpose(tmp_path: Path):
    root, review, _ = graph(tmp_path, holdout=True, fault="evaluation_purpose")
    with pytest.raises(ValueError, match="HOLDOUT purpose changed"):
        hold_view.validate_accepted_holdout(root, review, **LIMITS)
