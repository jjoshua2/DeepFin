"""Strict metadata and owned-phase joins shared by the HOLD scorer/CPU reader.

Only standard-library imports at module scope. All authorities are caller-trusted
hash pins; these routines never issue an acceptance or change candidate status.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import sysconfig
import uuid
from typing import Any, cast

from scripts import core_finite_learner as core
from scripts import core_generation_bank_acceptance as bank
from scripts import core_generation_holdout_acceptance as hold
from scripts import core_generation_holdout_metrics as metrics

REPO = Path(__file__).resolve().parents[1]
SCORE_SCHEMA = "CORE_GENERATION_HOLDOUT_SCORE_V1"
READ_SCHEMA = "CORE_GENERATION_HOLDOUT_READ_V1"
SCORE_PENDING = "CORE_HOLDOUT_OBSERVATIONS_PENDING_ROOT_CLOSURE_V1"
READ_PENDING = "CORE_HOLDOUT_HEALTH_PENDING_ROOT_CLOSURE_V1"
ROSTER_ROOT = "ROOT_ACCEPTS_PROSPECTIVE_WHOLE_GAME_PARENT_ROSTER_V1"
ROSTER_REVIEW = "ACCEPT_PROSPECTIVE_WHOLE_GAME_PARENT_ROSTER_NO_FINDINGS_V1"
DELIVERY_ROOT = "ROOT_ACCEPTS_CLOSED_CORE_BANK_LEARNER_DELIVERY_V1"
DELIVERY_REVIEW = "ACCEPT_CLOSED_CORE_BANK_LEARNER_DELIVERY_NO_FINDINGS_V1"
SCORE_ROOT = "ROOT_ACCEPTS_CLOSED_CORE_HOLDOUT_SCORING_OBSERVATIONS_V1"
SCORE_REVIEW = "ACCEPT_CLOSED_CORE_HOLDOUT_SCORING_OBSERVATIONS_NO_FINDINGS_V1"
ADMISSION = "ROOT_ADMITS_CORE_HOLDOUT_RULER_PHASE_V2"
SOURCE_REVIEW = "ACCEPT_CORE_HOLDOUT_RULER_SOURCE_ONLY_NO_FINDINGS_V1"
ANCHOR_KEY = "core_generation_bank_policy_anchor_v1"
BOUNDARY = "COMPLETED_CORE_GENERATION_BANK_POLICY_ANCHOR_SEGMENT_V1"
GPU_PAYLOAD_SCHEMA = "core_bank_learner_same_gpu_payload_proof_v1"
GPU_PAYLOAD_STATUS = "CORE_BANK_LEARNER_GPU_OWN_LIVE_SAVED_PROOF_PENDING_ROOT_CLOSURE_V1"
GPU_PROOF_ADMISSION = "ROOT_ADMITS_SCIENTIFIC_LEARNER_PASSIVE_PROOF_GPU_V1"
GPU_PROOF_SOURCE_REVIEW = "ACCEPT_SCIENTIFIC_LEARNER_PASSIVE_PROOF_SOURCE_ONLY_NO_FINDINGS_V1"
# Versioned pure-metadata consumer for the independently reviewed finite prover.
GPU_PROOF_HELPER_SHA256 = "7b45088df9ab897dfb0f9cafc6e79bc0f79072c37b1d35dd6e0a8144c9032ba6"
SOURCE_FILES = (
    "scripts/core_generation_holdout_score.py", "scripts/core_generation_holdout_read.py",
    "scripts/core_generation_holdout_ruler.py", "scripts/core_generation_holdout_metrics.py",
    "scripts/core_generation_holdout_acceptance.py", "scripts/core_generation_bank_acceptance.py",
    "scripts/core_finite_learner.py", "scripts/core_finite_admit.py", "scripts/core_finite_collect.py",
    "scripts/check_c_extensions_fresh.py", "chess_anti_engine/ext_build_record.py", "chess_anti_engine/ext_check.py", "chess_anti_engine/mcts/gumbel_c.py", "chess_anti_engine/uci/model_loader.py",
    "chess_anti_engine/model/__init__.py", "chess_anti_engine/model/transformer.py",
    "chess_anti_engine/replay/shard.py", "chess_anti_engine/replay/dataset.py",
    "chess_anti_engine/train/losses.py", "chess_anti_engine/utils/amp.py",
)
require = bank.require


class Metadata(bank._Metadata):
    """Object graph plus actual collector list roster and writable owner terminal."""
    def __init__(self, file_bytes: int, total_bytes: int):
        super().__init__(file_bytes, total_bytes)
        self.special: dict[Path, tuple[str, tuple[int, ...]]] = {}

    def observed(self, ref: dict[str, str], *, roster: bool = False) -> Any:
        path = bank.reference(ref)
        value = path.lstat()
        before = (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)
        require(stat.S_ISREG(value.st_mode) and (not roster or not value.st_mode & 0o222)
                and before[2] <= self.file_bytes and self.used + before[2] <= self.total_bytes, "special metadata bound/type")
        raw = path.read_bytes()
        after = path.lstat()
        require(before == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_mode)
                and len(raw) == before[2] and hashlib.sha256(raw).hexdigest() == ref["sha256"], "special metadata custody differs")
        if path in self.special:
            require(self.special[path] == (ref["sha256"], before), "conflicting special metadata reference")
        body = json.loads(raw, object_pairs_hook=bank.pairs, parse_constant=bank.nonfinite, parse_float=bank.finite_float)
        require(type(body) is (list if roster else dict), "special metadata schema differs")
        self.special[path] = (ref["sha256"], before)
        self.used += len(raw)
        return body

    def finish(self) -> None:
        super().finish()
        for path, (digest, before) in self.special.items():
            value = path.lstat()
            require(path.resolve() == path and before == (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)
                    and core.file_sha(path) == digest, "special metadata changed before return")


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": core.file_sha(path)}


def write(path: Path, value: Any) -> dict[str, str]:
    raw = canonical(value) + b"\n"
    require(len(raw) <= 4 << 20, "aggregate exceeds4MiB")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return reference(path)


def closed_owner(m: Metadata, owner: dict[str, Any], *, resource: str,
                 suffix: list[str], seconds: int | None = None, _same_gpu_proof: bool = False,
                 _ruler_phase: bool = False) -> dict[str, Any]:
    """Join a historical owner without confusing its revision with its child."""
    require(set(owner) == {"terminal", "admission", "command", "source_revision", "cwd", "closure"}, "invalid closed owner schema")
    terminal = m.observed(owner["terminal"])
    admission = m.read(owner["admission"])
    command = owner["command"]
    require(type(command) is list and len(command) > len(suffix) and command[-len(suffix):] == suffix
            and all(type(arg) is str and arg for arg in command), "closed child command differs")
    bank.sha(owner["source_revision"], 40)
    require(type(owner["cwd"]) is str and Path(owner["cwd"]).is_absolute(), "invalid owner cwd")
    require(terminal["status"] == "PASS" and type(terminal["returncode"]) is int and terminal["returncode"] == 0
            and terminal["owned_processes_closed"] is True and terminal["custody"] == "OWNED_PROCESSES_CLOSED"
            and terminal["resource"] == resource and terminal["command"] == command
            and terminal["source_revision"] == owner["source_revision"] and terminal["cwd"] == owner["cwd"], "owner not exact closed PASS")
    cap = bank.integer(terminal["seconds"], "owner cap")
    wall = terminal["wall_seconds"]
    require(type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall <= cap
            and (seconds is None or cap == seconds), "owner cap/wall differs")
    closure = owner["closure"]
    require(set(closure) == {"supported_eof", "exit_code", "supervisor_pid", "supervisor_closed", "owned_births_closed", "known_processes"}
            and closure["supported_eof"] is True and type(closure["exit_code"]) is int and closure["exit_code"] == 0
            and closure["supervisor_closed"] is True and closure["owned_births_closed"] is True
            and type(closure["supervisor_pid"]) is int and closure["supervisor_pid"] == terminal["supervisor_pid"]
            and closure["known_processes"] == terminal["known_processes"] and closure["known_processes"], "closure evidence differs")
    for pid, birth in closure["known_processes"].items():
        require(type(pid) is str and pid.isdecimal() and str(int(pid)) == pid and int(pid) > 0
                and type(birth) is int and birth > 0, "invalid process birth")
    require(not (_same_gpu_proof and _ruler_phase), "conflicting owner admission routes")
    if _same_gpu_proof:
        require(resource == "gpu" and seconds == cap == 900 and admission["status"] == GPU_PROOF_ADMISSION
                and admission["execution_authorized"] is True, "same-GPU proof admission differs")
        for flag, value in (("--admission", owner["admission"]["path"]), ("--admission-sha256", owner["admission"]["sha256"]),
                            ("--observer-plan", admission["observer_plan"]["path"]), ("--observer-plan-sha256", admission["observer_plan"]["sha256"])):
            require(command.count(flag) == 1 and command.index(flag) + 1 < len(command)
                    and command[command.index(flag) + 1] == value, "same-GPU observer command differs")
        normalized = list(command)
        normalized[normalized.index("--admission-sha256") + 1] = "SELF_ADMISSION_SHA256"
        require(admission["normalized_child"] == normalized and admission["resource"] == resource
                and admission["seconds"] == cap and admission["owner"]["sha256"] == terminal["supervisor_sha256"], "same-GPU normalized admission differs")
    elif _ruler_phase:
        require(resource == "gpu" and seconds == cap == 1200, "closed ruler phase cap/resource differs")
        phase_command(admission, admission["protocol"], owner["admission"], command, reader=False)
        require(admission["resource"] == resource and admission["seconds"] == cap
                and admission["owner"]["sha256"] == terminal["supervisor_sha256"], "ruler owner admission differs")
    else:
        require(admission["command"] == command and admission["resource"] == resource
                and admission["seconds"] == cap and admission["owner"]["sha256"] == terminal["supervisor_sha256"], "owner admission differs")
    return terminal


def parent_map(m: Metadata, p: dict[str, Any], view: hold.AcceptedHoldout) -> dict[str, str]:
    root = m.read(p["parents"]["root"])
    review = m.read(p["parents"]["independent"])
    keys = {"status", "study_uuid", "design", "producer_protocol", "producer_roster", "parent_map"}
    require(set(root) == keys and root["status"] == ROSTER_ROOT and root["study_uuid"] == p["study_uuid"]
            and root["design"] == p["design"], "parent authority differs")
    require(set(review) == keys | {"root_acceptance", "findings"} and review["status"] == ROSTER_REVIEW
            and review["root_acceptance"] == p["parents"]["root"] and review["findings"] == []
            and all(review[key] == root[key] for key in keys - {"status"}), "parent independent review differs")
    hroot = m.read(p["holdout"]["root"])
    audit = m.read(hroot["audit_protocol"])
    producer = m.read(root["producer_protocol"])
    require(audit["producer_protocol"] == root["producer_protocol"] and producer["roster"] == root["producer_roster"], "parent/producer provenance differs")
    planned = m.observed(root["producer_roster"], roster=True)
    mapping = m.read(root["parent_map"])
    require(set(mapping) == {"study_uuid", "collection_uuid", "games"} and mapping["study_uuid"] == p["study_uuid"]
            and mapping["collection_uuid"] == view.identity["collection_uuid"] and type(planned) is list
            and len(planned) == len(mapping["games"]) == 64, "incomplete planned64 parent map")
    ids: dict[str, str] = {}
    members: set[str] = set()
    parents: set[str] = set()
    for ordinal, (row, plan) in enumerate(zip(mapping["games"], planned, strict=True)):
        physical = str(view.identity["collection_uuid"]) + ":" + str(ordinal)
        require(set(row) == {"physical_game_id", "parent_id", "parent_member_ids"} and row["physical_game_id"] == plan["physical_game_id"] == physical
                and plan["bank"] == "holdout" and type(row["parent_id"]) is str and row["parent_id"] not in parents
                and row["parent_id"] and type(row["parent_member_ids"]) is list and row["parent_member_ids"], "invalid planned parent/game")
        whole = row["parent_member_ids"]
        require(all(type(member) is str and member for member in whole) and len(set(whole)) == len(whole)
                and not (members & set(whole)), "shared/duplicate complete parent members")
        members.update(whole)
        parents.add(row["parent_id"])
        ids[physical] = row["parent_id"]
    require(all(game["physical_game_id"] in ids for game in view.games), "unplanned HOLD game")
    return ids


def model_delivery(m: Metadata, binding: dict[str, Any], label: str) -> dict[str, Any]:
    """Prospective delivery interface; no payload or historical source reread."""
    require(set(binding) == {"root", "independent"}, "invalid model delivery binding")
    root = m.read(binding["root"])
    review = m.read(binding["independent"])
    keys = {"status", "contract", "stage", "boundary", "checkpoint", "source_revision", "source_checkout", "learner_owner", "payload_audit"}
    require(set(root) == keys and root["status"] == DELIVERY_ROOT, "invalid learner delivery root")
    require(set(review) == {"status", "root_acceptance", "contract", "stage", "boundary", "checkpoint", "payload_audit", "findings"}
            and review["status"] == DELIVERY_REVIEW and review["root_acceptance"] == binding["root"] and review["findings"] == []
            and all(review[key] == root[key] for key in ("contract", "stage", "boundary", "checkpoint", "payload_audit")), "model independent delivery differs")
    c, s, b = (m.read(root[key]) for key in ("contract", "stage", "boundary"))
    require(c["schema"] == ANCHOR_KEY and s["schema"] == "core_generation_bank_policy_anchor_stage_v1"
            and b["status"] == BOUNDARY and s["contract_sha256"] == b["contract_sha256"] == root["contract"]["sha256"]
            and b["stage"] == root["stage"] and b["checkpoint"] == root["checkpoint"]
            and c["source_revision"] == b["source_revision"] == root["source_revision"]
            and type(root["source_checkout"]) is str and Path(root["source_checkout"]).is_absolute(), "model contract/stage/source differs")
    bank.sha(root["source_revision"], 40)
    bank.sha(root["checkpoint"]["sha256"])
    require(b["run_uuid"] == c["run_uuid"] and b["logical_generation"] == c["first_generation"] == 1
            and b["initial_model"] == c["initial_model"] and b["initial_actor_step"] == c["initial_actor_step"], "fresh model lineage differs")
    require(type(s["segment"]) is int and s["segment"] == b["segment"] == 1
            and s["parent"] is None and b["parent"] is None and b["step_start"] == 0
            and type(b["step"]) is int and b["step"] == b["applied_updates"] == sum(c["updates_per_segment"])
            and b["applied_views"] == b["step"] * c["batch_size"] and b["policy_anchor"] == c["policy_anchor"], "incomplete fresh selected model endpoint")
    suffix = ["-m", "scripts.core_generation_bank_policy_anchor", "--contract", root["contract"]["path"], "--contract-sha256", root["contract"]["sha256"],
              "--stage", root["stage"]["path"], "--stage-sha256", root["stage"]["sha256"]]
    audit = root["payload_audit"]
    if set(audit) == {"readout", "owner"}:
        saved = m.read(audit["readout"])
        require(saved["status"] == "CORE_BANK_LEARNER_SAVED_STATE_AUDIT_PENDING_ROOT_CLOSURE_V1", "saved-state audit incomplete")
        closed_owner(m, root["learner_owner"], resource="gpu", suffix=suffix)
        closed_owner(m, audit["owner"], resource="cpu", suffix=saved["command_suffix"])
        require(saved["status"] == "CORE_BANK_LEARNER_SAVED_STATE_AUDIT_PENDING_ROOT_CLOSURE_V1"
                and all(saved[key] == root[key] for key in ("contract", "stage", "boundary", "checkpoint"))
                and saved["source_revision"] == root["source_revision"] and saved["architecture"] == c["architecture"]
                and type(saved["step"]) is int and saved["step"] == b["step"]
                and saved["reconciled"] == dict.fromkeys(("model", "optimizer", "scheduler", "zclip", "counters", "rng", "journal", "membership", "policy_anchor"), True), "saved-state audit incomplete")
    else:
        saved = same_gpu_payload(m, root, c, b, suffix)
    bank.sha(saved["model_state_sha256"])
    membership = b["membership"]
    require(type(membership) is list and membership and all(type(row) is list and len(row) == 3
            and type(row[0]) is int and row[0] == b["logical_generation"] and type(row[2]) is int and row[2] > 0 for row in membership), "invalid full replay membership")
    membership = cast(list[list[Any]], membership)
    for row in membership:
        bank.sha(row[1])
    train_rows = sum(row[2] for row in membership)
    require(train_rows >= 8000 and c["updates_per_segment"] == [min((train_rows + 31) // 32, 1024)]
            and b["drawn_rows"] == b["applied_views"], "selected dose or journal counts differ")
    require(b["accepted_banks"] == s["banks"] and len(b["bank_identities"]) == 1, "model bank horizon differs")
    require(all(set(identity) == {"run_uuid", "collection_uuid", "generation", "source_revision", "architecture", "actor_sha256", "actor_checkpoint_step"}
                for identity in b["bank_identities"]), "HOLD/unknown identity leaked into learner delivery")
    checkpoint = core.external(root["checkpoint"]["path"])
    output = Path(s["output"])
    require(checkpoint == output / "learner.pt" and not output.with_name(output.name + ".failed.json").exists()
            and not (output / "boundary.pending.json").exists(), "failed/pending learner endpoint")
    tag = {key: b[key] for key in ("run_uuid", "source_revision", "segment", "logical_generation", "policy_anchor")}
    tag["contract_sha256"] = root["contract"]["sha256"]
    return {"label": label, "checkpoint": root["checkpoint"], "architecture": c["architecture"], "step": b["step"],
            "tag": tag, "model_state_sha256": saved["model_state_sha256"], "contract": c, "boundary": b, "stage": s}



def same_gpu_payload(m: Metadata, root: dict[str, Any], c: dict[str, Any], b: dict[str, Any], suffix: list[str]) -> dict[str, Any]:
    """Versioned saved proof from the SAME closed learner GPU owner, JSON only."""
    audit = root["payload_audit"]
    require(set(audit) == {"schema", "readout", "owner"} and audit["schema"] == GPU_PAYLOAD_SCHEMA, "invalid saved-state audit binding")
    require(audit["owner"] == root["learner_owner"], "payload proof must share complete learner owner")
    closed_owner(m, audit["owner"], resource="gpu", suffix=suffix, seconds=900, _same_gpu_proof=True)
    saved = m.read(audit["readout"])
    require(saved["status"] == GPU_PAYLOAD_STATUS
            and all(saved[key] == root[key] for key in ("contract", "stage", "boundary", "checkpoint"))
            and saved["source_revision"] == root["source_revision"] and saved["source_checkout"] == root["source_checkout"]
            and saved["architecture"] == c["architecture"] and type(saved["step"]) is int and saved["step"] == b["step"]
            and saved["public_suffix"] == suffix
            and saved["reconciled"] == dict.fromkeys(("model", "optimizer", "scheduler", "zclip", "counters", "rng", "journal", "membership", "policy_anchor"), True), "same-GPU saved-state proof incomplete")
    plan = m.read(saved["observer_plan"])
    admission = m.read(audit["owner"]["admission"])
    require(plan["schema"] == "SCIENTIFIC_ANCHORED_LEARNER_PASSIVE_PROOF_V1"
            and plan["status"] == "BOUND_SCIENTIFIC_LEARNER_PASSIVE_PROOF_NOT_DELIVERY_AUTHORITY_V1"
            and admission["observer_plan"] == saved["observer_plan"]
            and all(saved[key] == plan[key] == admission[key] for key in ("helper", "source_review", "source_pins"))
            and all(plan[key] == root[key] for key in ("contract", "stage")) and plan["public_suffix"] == suffix
            and plan["learner_output"] == str(Path(root["checkpoint"]["path"]).parent)
            and plan["observation"] == audit["readout"]["path"]
            and plan["limits"] == {"outer_seconds": 900, "inner_seconds": 870, "checkpoint_file_bytes": 8 << 30,
                                   "journal_bytes": 16 << 20, "aggregate_bytes": 4 << 20}
            and type(saved["outer_seconds"]) is int and saved["outer_seconds"] == 900
            and type(saved["absolute_inner_seconds"]) is int and saved["absolute_inner_seconds"] == 870,
            "same-GPU plan/admission proof differs")
    helper = saved["helper"]
    require(type(helper) is dict and set(helper) == {"path", "sha256"} and type(helper["path"]) is str
            and Path(helper["path"]).is_absolute() and Path(helper["path"]).suffix == ".py"
            and helper["sha256"] == GPU_PROOF_HELPER_SHA256 and audit["owner"]["command"].count(helper["path"]) == 1,
            "unsupported same-GPU prover")
    review = m.read(saved["source_review"])
    require(review["status"] == GPU_PROOF_SOURCE_REVIEW and review["helper"] == helper
            and review["source_pins"] == saved["source_pins"] and review["findings"] == [], "same-GPU prover source acceptance differs")
    pins = m.read(saved["source_pins"])
    require(pins["source_revision"] == root["source_revision"] and pins["source_checkout"] == root["source_checkout"]
            and type(pins["files"]) is dict and len(pins["files"]) == 219, "same-GPU historical source identity differs")
    require(saved["save_before_equals_after"] is True and saved["reference_immutable"] is True
            and type(saved["additional_final_CPU_checkpoint_decodes"]) is int and saved["additional_final_CPU_checkpoint_decodes"] == 1,
            "same-GPU live/final proof incomplete")
    bank.sha(saved["own_live_state_fingerprint"])
    initial = saved["initial"]
    for key, value in (("state_entries", 494), ("input_planes", 175), ("unique_parameter_numel", 61444448), ("naive_state_numel", 77173088)):
        require(type(initial["model"][key]) is int and initial["model"][key] == value, "same-GPU full model facts differ")
    require(initial["model"]["finite"] is True and saved["initial_model_state_sha256"] == initial["model"]["state_sha256"], "same-GPU initial model differs")
    bank.sha(saved["initial_model_state_sha256"])
    bank.integer(saved["optimizer_named_parameters"], "named optimizer parameters")
    bank.integer(saved["active_optimizer_slots"], "active optimizer slots")
    require(saved["active_optimizer_slots"] <= saved["optimizer_named_parameters"], "same-GPU optimizer ownership differs")
    require(type(b["rng"]) is dict and set(b["rng"]) == {"numpy", "mask", "torch_cpu", "torch_cuda", "python"}
            and type(b["rng"]["torch_cuda"]) is list and len(b["rng"]["torch_cuda"]) == 1, "same-GPU five RNG streams differ")
    require(saved["five_rng_snapshot_sha256"] == hashlib.sha256(canonical(b["rng"])).hexdigest()
            and saved["membership_sha256"] == hashlib.sha256(canonical(b["membership"])).hexdigest()
            and canonical(initial["membership"]) == canonical(b["membership"])
            and saved["actual_backend"] == initial["backend"] == b["backend"], "same-GPU RNG/membership/backend differs")
    membership = b["membership"]
    require(type(membership) is list and membership and all(type(row) is list and len(row) == 3
            and type(row[0]) is int and row[0] == 1 and type(row[2]) is int and row[2] > 0 for row in membership),
            "invalid full replay membership")
    membership = cast(list[list[Any]], membership)
    population = sum(row[2] for row in membership)
    coverage = saved["draw_coverage"]
    require(type(plan["N"]) is int and plan["N"] == population and type(plan["K"]) is int
            and plan["K"] == b["step"] == min((population + 31) // 32, 1024)
            and type(coverage["population_rows"]) is int and coverage["population_rows"] == population
            and type(coverage["batches"]) is int and coverage["batches"] == b["step"]
            and type(coverage["views"]) is int and coverage["views"] == b["drawn_rows"] == b["applied_views"]
            and coverage["journal_sha256"] == b["draws_sha256"]
            and type(coverage["views_per_population_row"]) is float
            and coverage["views_per_population_row"] == coverage["views"] / population, "same-GPU draw totals differ")
    return saved


def same_gpu_game_coverage(m: Metadata, binding: dict[str, Any], model: dict[str, Any], train: bank.AcceptedGeneration) -> None:
    """Join the new aggregate roster to the actual accepted TRAIN, no draws read."""
    root = m.read(binding["root"])
    audit = root["payload_audit"]
    if set(audit) == {"readout", "owner"}:
        return
    require(set(audit) == {"schema", "readout", "owner"} and audit["schema"] == GPU_PAYLOAD_SCHEMA, "invalid saved-state audit binding")
    coverage = m.read(audit["readout"])["draw_coverage"]
    rows = coverage["per_game_aggregate_counts"]
    require(type(rows) is list and len(rows) == len(train.games) and type(coverage["games"]) is int
            and coverage["games"] == len(train.games), "same-GPU coverage game roster differs")
    draws = unique = zero = population = 0
    for row, game in zip(rows, train.games, strict=True):
        require(type(row) is dict and set(row) == {"physical_game_id", "population_rows", "draws", "unique_rows", "unique_fraction"}
                and row["physical_game_id"] == game["physical_game_id"] and type(row["population_rows"]) is int
                and row["population_rows"] == game["rows"], "same-GPU coverage physical game differs")
        count = bank.integer(row["draws"], "game draws", 0)
        seen = bank.integer(row["unique_rows"], "game unique rows", 0)
        require(seen <= min(count, game["rows"]) and (seen > 0) == (count > 0)
                and type(row["unique_fraction"]) is float and row["unique_fraction"] == seen / game["rows"], "same-GPU per-game coverage differs")
        draws += count
        unique += seen
        population += game["rows"]
        zero += count == 0
    require(type(coverage["unique_physical_rows_drawn"]) is int and coverage["unique_physical_rows_drawn"] == unique
            and type(coverage["zero_draw_games"]) is int and coverage["zero_draw_games"] == zero
            and population == coverage["population_rows"] and draws == coverage["views"] == model["boundary"]["drawn_rows"]
            and type(coverage["unique_fraction"]) is float and coverage["unique_fraction"] == unique / population, "same-GPU aggregate coverage differs")


def asset_identity(m: Metadata, donor: dict[str, Any]) -> dict[str, Any]:
    """Join the selected genuine current-H model-only asset authority."""
    require(set(donor) == {"checkpoint", "architecture", "step", "model_state_sha256", "root", "independent", "asset_source"}, "invalid A39 binding")
    asset, independent = m.read(donor["root"]), m.read(donor["independent"])
    readout = m.read(asset["readout"])
    source = donor["asset_source"]
    require(set(source) == {"revision", "checkout", "source_pins"}, "invalid current-H asset source binding")
    bank.sha(source["revision"], 40)
    bank.reference(source["source_pins"])
    require(asset["status"] == "ROOT_ACCEPTS_COLLECTOR_CURRENT_H_CPU_ASSET_READINESS_V1"
            and independent["status"] == "ACCEPT_CLOSED_COLLECTOR_CURRENT_H_CPU_ASSET_READINESS_NO_FINDINGS_V1"
            and independent["root_acceptance"] == donor["root"] and independent["findings"] == []
            and independent["readout"] == asset["readout"] and independent["protocol"] == readout["protocol"] == asset["protocol"]
            and independent["terminal"] == asset["terminal"], "current-H A39 asset acceptance differs")
    require(all(value["source_revision"] == source["revision"] and value["source_checkout"] == source["checkout"]
                for value in (asset, independent, readout))
            and independent["source_pins"] == readout["source_pins"] == source["source_pins"]
            and readout["status"] == "COMPLETE_COLLECTOR_CURRENT_H_CPU_ASSET_QUALIFICATION_CHILD_ROOT_CLOSURE_REQUIRED_V1", "current-H asset source/readout differs")
    observed = readout["donor_model_only"]
    require(asset["donor_model_only"] == observed and observed["actual_CPU_model_only_load"] is True,
            "current-H model-only observation differs")
    require(donor["checkpoint"]["sha256"] == "a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0", "not original A39 donor")
    require(observed["sha256"] == donor["checkpoint"]["sha256"] and observed["path"] == donor["checkpoint"]["path"]
            and observed["embedded_step"] == donor["step"] == 680276 and observed["architecture"] == donor["architecture"]
            and observed["state_sha256"] == donor["model_state_sha256"], "A39 observed identity differs")
    bank.sha(donor["model_state_sha256"])
    return {**donor, "label": "A39", "tag": None}


def training_population(m: Metadata, model: dict[str, Any]) -> Any:
    """Actual TRAIN acceptance API; HOLD can never be a training-bank token."""
    stage = model["stage"]
    require(type(stage["banks"]) is list and len(stage["banks"]) == 1, "selected TRAIN bank horizon differs")
    binding = stage["banks"][0]
    require(type(binding) is dict and set(binding) == {"root_acceptance", "independent_review"}, "invalid TRAIN bank binding")
    binding = cast(dict[str, Any], binding)
    accepted = bank.validate_accepted_generation(binding["root_acceptance"], binding["independent_review"],
        max_file_bytes=m.file_bytes, max_total_bytes=m.total_bytes, max_games=192)
    # Keep a defensive runtime family check at the external acceptance boundary.
    require(isinstance(accepted, bank.AcceptedGeneration), "TRAIN identity changed after training")  # pyright: ignore[reportUnnecessaryIsInstance]
    require(plain(accepted.identity) == model["boundary"]["bank_identities"][0], "TRAIN identity changed after training")
    require(160 <= len(accepted.games) <= 192 and accepted.evidence["eligible_rows"] >= 8000, "selected TRAIN coverage differs")
    membership = [[accepted.identity["generation"], hashlib.sha256(canonical(plain(game["shard"]["files"]))).hexdigest(), game["rows"]] for game in accepted.games]
    require(membership == model["boundary"]["membership"], "TRAIN membership changed after training")
    return accepted


def bindings(m: Metadata, p: dict[str, Any]) -> tuple[hold.AcceptedHoldout, dict[str, str], dict[str, Any]]:
    view = hold.validate_accepted_holdout(p["holdout"]["root"], p["holdout"]["independent"],
        max_file_bytes=p["limits"]["metadata_file_bytes"], max_total_bytes=p["limits"]["metadata_total_bytes"], max_games=64)
    parents = parent_map(m, p, view)
    require(view.identity["bank"] == "holdout" and len(view.games) >= metrics.MIN_PARENTS
            and sum(game["rows"] for game in view.games) >= metrics.MIN_ROWS, "UNREAD insufficient HOLD coverage")
    require(set(p["models"]) == set(metrics.LABELS), "fixed four-model order required")
    models = {label: model_delivery(m, p["models"][label], label) for label in ("A", "B", "C")}
    donor = p["models"]["A39"]
    models["A39"] = asset_identity(m, donor)
    for label in metrics.LABELS:
        require(models[label]["architecture"] == plain(view.identity["architecture"]), "HOLD/model architecture differs")
    for label, weight in (("A", 0), ("B", 0), ("C", 0.1)):
        train = training_population(m, models[label])
        same_gpu_game_coverage(m, p["models"][label], models[label], train)
        require(not ({game["physical_game_id"] for game in train.games} & {game["physical_game_id"] for game in view.games}), "HOLD physical game leaked into training")
        c = models[label]["contract"]
        a = c["policy_anchor"]
        require(c["initial_model"] == donor["checkpoint"] and c["initial_actor_step"] == donor["step"]
                and c["seed"] == 2026111673 and c["batch_size"] == 32 and c["mirror_prob"] == 0.5, "selected model bootstrap/stream differs")
        require(set(a) == {"schema", "direction", "mask", "coefficient", "reference"}
                and a["schema"] == "frozen_reference_policy_anchor_v1" and a["direction"] == "reference_to_student_forward_KL"
                and a["mask"] == "original_network_turn_has_policy_with_explicit_legal_mask_v1"
                and type(a["coefficient"]) in (int, float), "unsupported anchored delivery semantics")
        require(a["coefficient"] == weight and (a["reference"] is None if weight == 0 else
                a["reference"]["checkpoint"] == donor["checkpoint"] and a["reference"]["checkpoint_step"] == donor["step"]), "selected anchor identity differs")
        require(c["lr"] == {"peak": 7.5e-7 if label == "B" else 3e-6, "floor": 7.5e-7 if label == "B" else 3e-6,
                "warmup_start": 7.5e-7 if label == "B" else 3e-6, "warmup_steps": 0, "first_period": 2048}, "selected LR contrast differs")
        require(all(identity["collection_uuid"] != view.identity["collection_uuid"] for identity in models[label]["boundary"]["bank_identities"]), "HOLD collection leaked into training")
    # Frozen recipe identities come from the design-bound delivery contracts;
    # enforce matching data/horizon/seed, preserving the preselected LR contrast.
    for label in ("B", "C"):
        left = {key: value for key, value in models["A"]["contract"].items() if key not in ("run_uuid", "lr", "policy_anchor")}
        right = {key: value for key, value in models[label]["contract"].items() if key not in ("run_uuid", "lr", "policy_anchor")}
        require(left == right and models[label]["boundary"]["membership"] == models["A"]["boundary"]["membership"]
                and models[label]["boundary"]["draws_sha256"] == models["A"]["boundary"]["draws_sha256"]
                and models[label]["boundary"]["rng"] == models["A"]["boundary"]["rng"]
                and models[label]["boundary"]["backend"] == models["A"]["boundary"]["backend"], "arms are not matched")
    return view, parents, models


def import_custody() -> None:
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("scripts.", "chess_anti_engine.")) or name == "chess_anti_engine":
            filename = getattr(module, "__file__", None)
            require(filename is not None and Path(filename).resolve().is_relative_to(REPO), "foreign project import: " + name)


def source_guard(source: dict[str, Any]) -> None:
    require(set(source) == {"checkout", "revision", "files"} and source["checkout"] == str(REPO), "foreign source checkout")
    bank.sha(source["revision"], 40)
    import_custody()
    files = source["files"]
    require(type(files) is dict and set(files) >= set(SOURCE_FILES), "incomplete ruler source pins")
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, env=env, text=True).strip() == source["revision"]
            and not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, env=env), "dirty/wrong source revision")
    for raw_name, expected in files.items():
        require(type(raw_name) is str and not Path(raw_name).is_absolute() and ".." not in Path(raw_name).parts, "invalid source name")
        name = cast(str, raw_name)
        bank.sha(expected)
        require((REPO / name).resolve() == REPO / name and core.file_sha(REPO / name) == expected, "source digest differs")


def native_symbol_abi() -> dict[str, Any]:
    """Invoke the existing in-process required-symbol/numeric-MCTS-ABI gate."""
    from chess_anti_engine import ext_check

    require(Path(ext_check.__file__).resolve() == REPO / "chess_anti_engine/ext_check.py", "foreign native symbol primitive")
    problems = ext_check.extension_problems()
    require(not problems, "native required-symbol/numeric-ABI gate failed: " + "; ".join(problems))
    tree = importlib.import_module(ext_check.MCTS_TREE_EXT)
    import_custody()
    return {"primitive": "chess_anti_engine.ext_check.extension_problems", "problems": problems,
            "required_extensions": list(ext_check.REQUIRED_EXTENSIONS),
            "required_mcts_abi": ext_check.required_abi(), "observed_mcts_abi": int(tree.ABI_VERSION)}


def native_custody(m: Metadata, basis_ref: dict[str, str], files: dict[str, str]) -> dict[str, Any]:
    """Fresh named-file/record/source/ABI bracket, not mapped-byte attestation."""
    from scripts import check_c_extensions_fresh as fresh

    basis = m.read(basis_ref)
    specs = {spec.module: set(spec.dependencies) for spec in fresh.EXTENSION_SPECS}
    union = set().union(*specs.values())
    require(len(specs) == 6 and len(union) == 31 and set(files) >= union, "incomplete native source closure")
    images = basis["images"]
    require(len(images) == 6 and len({item["path"] for item in images}) == 6, "native basis requires six images")
    require({str(path) for path in (REPO / "chess_anti_engine").rglob("*.so")} == {item["path"] for item in images}, "native image roster differs")
    observed = []
    modules: set[str] = set()
    for item in images:
        path = Path(item["path"])
        require(path.is_relative_to(REPO / "chess_anti_engine") and path.resolve() == path, "foreign native image")
        before = bank.stamp(path)
        require(before[2] == item["size"] and core.file_sha(path) == item["sha256"] and bank.stamp(path) == before, "native image differs")
        record_path = bank.reference(item["record"])
        require(record_path == path.with_name(path.name + ".build-record.json") and record_path.stat().st_size <= 4 << 20, "foreign/oversized native record")
        record = m.read(item["record"])
        module = record["module"]
        relative = path.relative_to(REPO)
        derived = ".".join(relative.parts[:-1]) + "." + relative.name.split(".", 1)[0]
        require(module == derived and module not in modules, "native module roster differs")
        modules.add(module)
        require(module in specs and record["version"] == 1 and record["artifact_sha256"] == item["sha256"]
                and set(record["inputs"]) == specs[module] and all(files.get(name) == digest for name, digest in record["inputs"].items()), "native record/source differs")
        require(path.name.endswith(str(sysconfig.get_config_var("EXT_SUFFIX"))), "native interpreter ABI suffix differs")
        loaded = importlib.import_module(module)
        filename = getattr(loaded, "__file__", None)
        require(type(filename) is str and Path(filename).resolve() == path, "native import origin differs")
        observed.append({"path": str(path), "sha256": item["sha256"], "stamp": list(before), "record": item["record"]})
    require(modules == set(specs) and not fresh.check_extensions(REPO), "native roster/freshness failed")
    return {"images": observed, "required_symbols_and_numeric_abi": native_symbol_abi()}


def phase_command(admission: dict[str, Any], protocol_ref: dict[str, str], admission_ref: dict[str, str],
                  command: list[str], *, reader: bool) -> None:
    """Authenticate an actual child against a sealed, non-self-referential V2 admission."""
    module = "scripts.core_generation_holdout_read" if reader else "scripts.core_generation_holdout_score"
    suffix = ["-m", module, "--protocol", protocol_ref["path"], "--protocol-sha256", protocol_ref["sha256"],
              "--admission", admission_ref["path"], "--admission-sha256", admission_ref["sha256"]]
    require(admission["status"] == ADMISSION and admission["protocol"] == protocol_ref and "command" not in admission
            and type(admission.get("normalized_command")) is list, "phase requires normalized V2 admission")
    require(type(command) is list and len(command) > len(suffix)
            and all(type(arg) is str and arg for arg in command)
            and command[-len(suffix):] == suffix, "ruler phase command suffix differs")
    for flag in ("--protocol", "--protocol-sha256", "--admission", "--admission-sha256"):
        require(command.count(flag) == 1, "ruler phase command flags ambiguous")
    normalized = list(command)
    normalized[normalized.index("--admission-sha256") + 1] = "SELF_ADMISSION_SHA256"
    require(admission["normalized_command"] == normalized, "normalized ruler phase command differs")


def phase_start(protocol_ref: dict[str, str], admission_ref: dict[str, str], *, reader: bool = False) -> tuple[dict[str, Any], Metadata, Path]:
    m = Metadata(4 << 20, 256 << 20)
    p, admission = m.read(protocol_ref), m.read(admission_ref)
    schema = READ_SCHEMA if reader else SCORE_SCHEMA
    keys = {"schema", "study_uuid", "design", "holdout", "parents", "models", "source", "runtime", "owner", "stage",
            "outer_seconds", "inner_seconds", "output", "limits", "source_review", "native_basis"}
    if reader:
        keys |= {"scoring"}
    require(set(p) == keys and p["schema"] == schema, "wrong ruler protocol")
    require(str(uuid.UUID(p["study_uuid"])) == p["study_uuid"], "invalid study UUID")
    resource, cap = ("cpu", 900) if reader else ("gpu", 1200)
    require(type(p["outer_seconds"]) is int and p["outer_seconds"] == cap
            and type(p["inner_seconds"]) is int and 0 < p["inner_seconds"] < cap, "unbound ruler caps")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("HOLD ruler inner deadline")))
    signal.alarm(p["inner_seconds"])
    source_guard(p["source"])
    rt = p["runtime"]
    require(set(rt) == {"python", "prefix", "python_version", "managed_site", "torch_version", "numpy_version"}
            and sys.executable == rt["python"] and sys.version == rt["python_version"] and sys.prefix == rt["prefix"]
            and sys.flags.dont_write_bytecode and not sys.flags.optimize, "ruler interpreter differs")
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP", "PYTHONSAFEPATH", "LD_PRELOAD", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG"):
        require(name not in os.environ, "unexpected override: " + name)
    require(os.sched_getaffinity(0) == {20, 22} and os.environ.get("CUDA_VISIBLE_DEVICES") == ("" if reader else "0"), "ruler device/affinity differs")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        require(os.environ.get(name) == "2", "CPU thread environment differs")
    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    pth = site / "managed-dependencies.pth"
    require(list(site.glob("*.pth")) == [pth] and pth.resolve() == pth, "plain-pth roster differs")
    before = pth.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= 4 << 20, "plain-pth size/type")
    stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_mode)
    raw = pth.read_bytes()
    after = pth.lstat()
    require(stamp == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_mode)
            and raw == (rt["managed_site"] + "\n").encode(), "plain-pth content/custody differs")
    require(admission["status"] == ADMISSION and admission["protocol"] == protocol_ref and admission["source"] == p["source"]
            and admission["owner"] == p["owner"] and admission["stage"] == p["stage"] and admission["resource"] == resource
            and admission["seconds"] == cap and admission["runtime"] == rt
            and admission["u3"] and admission["scope"], "phase admission differs")
    issued = datetime.fromisoformat(admission["issued_utc"])
    deadline = datetime.fromisoformat(admission["deadline_utc"])
    now = datetime.now(timezone.utc)
    require(issued.tzinfo is not None and deadline.tzinfo is not None and issued <= now <= deadline, "stale phase admission")
    for key in ("u3", "scope"):
        m.read(admission[key])
    parent = Path("/proc", str(os.getppid()), "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    require(p["owner"]["path"] in parent and core.file_sha(Path(p["owner"]["path"])) == p["owner"]["sha256"], "not selected direct owner")
    for flag, value in (("--name", p["stage"]), ("--seconds", str(cap)), ("--resource", resource)):
        require(parent.count(flag) == 1 and parent[parent.index(flag) + 1] == value, "outer owner phase differs")
    require(parent.count("--") == 1, "outer owner child separator ambiguous")
    phase_command(admission, protocol_ref, admission_ref, parent[parent.index("--") + 1:], reader=reader)
    review = m.read(p["source_review"])
    require(review["status"] == SOURCE_REVIEW and review["source"] == p["source"] and review["findings"] == [], "unreviewed ruler source")
    design = m.read(p["design"])
    require(design["study_uuid"] == p["study_uuid"], "foreign study design")
    limits = p["limits"]
    require(set(limits) == {"metadata_file_bytes", "metadata_total_bytes", "game_decoded_bytes", "total_decoded_bytes", "max_rows", "row_log_bytes", "reference_cache_bytes"}, "wrong ruler limits")
    for name, value in limits.items():
        bank.integer(value, name)
    m.file_bytes = limits["metadata_file_bytes"]
    m.total_bytes = limits["metadata_total_bytes"]
    require(limits["metadata_file_bytes"] <= 128 << 20 and limits["max_rows"] * 1858 * 4 <= limits["reference_cache_bytes"], "invalid metadata/cache cap")
    output = core.external(p["output"])
    require(not output.exists() and not output.with_name(output.name + ".failed.json").exists(), "ruler output already used")
    output.mkdir()
    return p, m, output
