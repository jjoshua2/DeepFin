"""Truthful metadata bridge from a closed once learner to finite collection.

The caller must pin the delivery pair and its independently reviewed source.
This does not select an actor or issue collection authority. Checkpoint tensors
are checked later, by the collector's existing single real load/build path.
"""
from __future__ import annotations

from typing import Any, cast

from scripts import core_finite_learner as core

KIND = "closed_once_learner"
SCHEMA = "core_closed_once_actor_lineage_v1"
FAMILIES: dict[str, dict[str, Any]] = {
    "capped_cycle": {"tag": "core_capped_on_policy_cycle_v1"},
    "global_union": {
        "contract": "core_selfplay_union_stored_row_once_v1",
        "boundary": "COMPLETED_CORE_SELFPLAY_UNION_STORED_ROW_ONCE_SEGMENT_V1",
        "module": "scripts.core_selfplay_union_stored_row_once_proof",
        "tag": "core_selfplay_union_stored_row_once_v1",
        "certificate": "core_selfplay_union_own_live_saved_payload_proof_v1",
        "source_status": "ACCEPT_UNIFORM_ONCE_ENTRY_SOURCE_ONLY_NO_FINDINGS_V1",
        "deliveries": {
            "ROOT_ACCEPTS_CLOSED_U1_FULL_PASS_MODEL_DELIVERY_V1": "ACCEPT_CLOSED_U1_FULL_PASS_MODEL_DELIVERY_NO_FINDINGS_V1",
            "ROOT_ACCEPTS_CLOSED_V10_FULL_PASS_MODEL_DELIVERY_V1": "ACCEPT_CLOSED_V10_FULL_PASS_MODEL_DELIVERY_NO_FINDINGS_V1",
        },
    },
    "policy_anchor": {
        "contract": "core_selfplay_union_policy_anchor_stored_row_once_v1",
        "boundary": "COMPLETED_CORE_SELFPLAY_UNION_POLICY_ANCHOR_STORED_ROW_ONCE_SEGMENT_V1",
        "module": "scripts.core_selfplay_union_policy_anchor_stored_row_once_proof",
        "tag": "core_selfplay_union_policy_anchor_stored_row_once_v1",
        "certificate": "core_selfplay_union_policy_anchor_own_live_saved_payload_proof_v1",
        "source_status": "ACCEPT_GLOBAL_UNION_POLICY_ANCHOR_INTEGRATION_SOURCE_ONLY_NO_FINDINGS_V1",
        "deliveries": {
            "ROOT_ACCEPTS_CLOSED_U2_FULL_PASS_MODEL_DELIVERY_V1": "ACCEPT_CLOSED_U2_FULL_PASS_MODEL_DELIVERY_NO_FINDINGS_V1",
        },
    },
}
ROOT_FIELDS = {"status", "study_uuid", "method", "contract", "boundary", "closed_binding",
               "entry_source", "entry_source_review", "model"}
MODEL_FIELDS = {"checkpoint", "architecture", "step", "model_state_sha256",
                "model_state_hash_schema", "label", "tag_key", "tag"}


def read(ref: dict[str, str]) -> dict[str, Any]:
    return core.read_ref(ref)[1]


def closed_delivery(lineage: dict[str, Any]) -> dict[str, Any]:
    """Join caller-trusted Root/ind metadata; never promote a bare status string.

    Deliberately no public virtual-population consumer call here: the delivery
    pair is the acceptance of that earlier owned invocation. The actual current
    collector still authenticates and decodes the checkpoint exactly once.
    """
    core.require(type(lineage) is dict and set(lineage) == {"schema", "family", "delivery"}
                 and lineage["schema"] == SCHEMA and lineage["family"] in FAMILIES,
                 "closed once actor lineage schema")
    if lineage["family"] == "capped_cycle":
        from scripts.core_capped_on_policy_cycle import accepted_delivery

        return accepted_delivery(lineage, resolve_delivery=closed_delivery)
    family = FAMILIES[lineage["family"]]
    pair = lineage["delivery"]
    core.require(type(pair) is dict and set(pair) == {"root", "independent"}, "closed once actor delivery pair")
    pair = cast(dict[str, Any], pair)
    root, independent = read(pair["root"]), read(pair["independent"])
    core.require(set(root) == ROOT_FIELDS and set(independent) == ROOT_FIELDS | {"root_acceptance", "findings"}
                 and root["status"] in family["deliveries"]
                 and independent["status"] == family["deliveries"][root["status"]]
                 and independent["root_acceptance"] == pair["root"] and independent["findings"] == []
                 and all(independent[k] == v for k, v in root.items() if k != "status"),
                 "closed once actor trusted delivery joins")
    contract, boundary = read(root["contract"]), read(root["boundary"])
    source, review = read(root["entry_source"]), read(root["entry_source_review"])
    core.require(review["source_manifest"] == root["entry_source"] and review["findings"] == []
                 and source["owned_source_files"] == review["owned_source_files"],
                 "closed once actor accepted source pair")
    # Historical global entry reviews predate the retained-search source status.
    allowed = {family["source_status"]}
    if lineage["family"] == "global_union":
        allowed |= {"ACCEPT_GLOBAL_UNION_RETAINED_SEARCH_SHARE_SOURCE_ONLY_NO_FINDINGS_V1",
                    "ACCEPT_HISTORICAL_SELFPLAY_UNIFORM_ONCE_ENTRY_SOURCE_ONLY_NO_FINDINGS_V1"}
    core.require(review["status"] in allowed and contract["schema"] == family["contract"]
                 and boundary["status"] == family["boundary"] and boundary["pass_complete"] is True,
                 "closed once actor source/contract/boundary family")
    stage = read(boundary["stage"])
    output = core.external(stage["output"])
    binding = root["closed_binding"]
    core.require(type(binding) is dict and set(binding) == {"boundary", "owner_terminal", "owner_sha256", "command"}
                 and binding["boundary"] == root["boundary"]
                 and core.external(root["boundary"]["path"]) == output / "boundary.json"
                 and not core.failure_marker(output).exists() and not (output / "boundary.pending.json").exists(),
                 "closed once actor closed boundary binding")
    binding = cast(dict[str, Any], binding)
    terminal = read(binding["owner_terminal"])
    suffix = ["-m", family["module"], "--contract", root["contract"]["path"],
              "--contract-sha256", root["contract"]["sha256"], "--stage", boundary["stage"]["path"],
              "--stage-sha256", boundary["stage"]["sha256"]]
    core.require(type(binding["command"]) is list and binding["command"][-10:] == suffix
                 and terminal["command"] == binding["command"] and terminal["status"] == "PASS"
                 and type(terminal["returncode"]) is int and terminal["returncode"] == 0
                 and terminal["owned_processes_closed"] is True and terminal["custody"] == "OWNED_PROCESSES_CLOSED"
                 and terminal["supervisor_sha256"] == binding["owner_sha256"], "closed once actor owner/command")
    core.require(boundary["contract_sha256"] == stage["contract_sha256"] == root["contract"]["sha256"]
                 and boundary["method"] == root["method"] == contract["method"]
                 and boundary["study_uuid"] == root["study_uuid"] == contract["study_uuid"]
                 and boundary["source_revision"] == contract["source_revision"]
                 and boundary["run_uuid"] == contract["run_uuid"]
                 and boundary["segment"] == stage["segment"] == len(contract["updates_per_segment"])
                 and boundary["step"] == sum(contract["updates_per_segment"])
                 and stage["population"] == contract["population"], "closed once actor original contract/stage")
    certificate = boundary["own_live_certificate"]
    facts = certificate["saved_model"]
    count = contract["updates_per_segment"][-1]
    core.require(certificate["schema"] == family["certificate"]
                 and certificate["contract"] == root["contract"] and certificate["stage"] == boundary["stage"]
                 and certificate["checkpoint"] == boundary["checkpoint"]
                 and certificate["actual_calls"] == {"forward": count, "optimizer_pre": count, "optimizer_post": count, "save": 1}
                 and certificate["save_before_equals_after"] is True and certificate["extra_forwards"] == 0
                 and certificate["additional_final_CPU_checkpoint_decodes"] == 1
                 and facts["finite"] is True, "closed once actor actual saved model certificate")
    model = root["model"]
    core.require(set(model) == MODEL_FIELDS and model["checkpoint"] == boundary["checkpoint"]
                 and core.external(model["checkpoint"]["path"]) == output / "learner.pt"
                 and model["architecture"] == contract["architecture"] and model["step"] == boundary["step"]
                 and model["model_state_hash_schema"] == "union_typed_tensor_tree_sha256_v1"
                 and model["model_state_sha256"] == certificate["saved_model_state_sha256"] == facts["model_state_sha256"]
                 and model["tag_key"] == family["tag"], "closed once actor model descriptor")
    identity = {k: contract[k] for k in ("run_uuid", "study_uuid", "source_revision", "sampling", "method")}
    identity.update(contract_sha256=root["contract"]["sha256"], segment=boundary["segment"])
    if lineage["family"] == "policy_anchor":
        identity["policy_anchor"] = contract["policy_anchor"]
    elif "retained_search_share" in contract:
        identity["retained_search_share"] = contract["retained_search_share"]
    tag = {"identity": identity, **{k: boundary[k] for k in ("quota_state", "draw_journals", "quota_ledger_checkpoint",
           "population_identity", "source_origin_binding")}}
    core.require(model["tag"] == tag and boundary["quota_state"]["tainted"] is False
                 and boundary["quota_state"]["cursor"] == contract["population"]["rows"],
                 "closed once actor final tag/cursor")
    return {"root": root, "boundary": boundary, "contract": contract, "model": model}


def validate_actor(actor: dict[str, Any]) -> dict[str, Any]:
    core.require(set(actor) == {"kind", "path", "sha256", "expected_checkpoint_step", "architecture", "predecessor", "lineage"}
                 and actor["kind"] == KIND, "closed once actor fields")
    delivery = closed_delivery(actor["lineage"])
    root, model = delivery["root"], delivery["model"]
    core.require(actor["predecessor"] == root["closed_binding"]
                 and {"path": actor["path"], "sha256": actor["sha256"]} == model["checkpoint"]
                 and type(actor["expected_checkpoint_step"]) is int and actor["expected_checkpoint_step"] == model["step"]
                 and actor["architecture"] == model["architecture"], "closed once actor selected checkpoint identity")
    return delivery


def validate_payload(actor: dict[str, Any], payload: dict[str, Any], torch: Any) -> str:
    """Called from the existing collector's one real torch.load interception."""
    from scripts.core_selfplay_union_stored_row_once_proof import fingerprint

    model = validate_actor(actor)["model"]
    keys = {f["tag"] for f in FAMILIES.values()} | {"core_generation_bank_learner_v1",
        "core_generation_bank_policy_anchor_v1", "core_generation_bank_stored_row_once_v1"}
    actual_hash = fingerprint(payload["model"], torch)
    core.require(set(payload) & keys == {model["tag_key"]}
                 and payload[model["tag_key"]] == model["tag"]
                 and actual_hash == model["model_state_sha256"],
                 "closed once actor actual payload tag/typed state differs")
    if actor["lineage"]["family"] in {"policy_anchor", "capped_cycle"}:
        core.require(payload.get("frozen_reference_policy_anchor_v1") == model["tag"]["identity"]["policy_anchor"],
                     "closed once actor actual frozen reference identity")
    return actual_hash
