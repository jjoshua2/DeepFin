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
GPU_PROOF_THREAD2_HELPER_SHA256 = "721cf5d141843ef0db61adb5169e9424e4e7fcbca69758b8dadb8c5321a66b26"
D1_SCORE_SCHEMA = "CORE_GENERATION_HOLDOUT_SCORE_D1_V1"
D1_READ_SCHEMA = "CORE_GENERATION_HOLDOUT_READ_D1_V1"
D1_SCORE_PENDING = "CORE_HOLDOUT_D1_OBSERVATIONS_PENDING_ROOT_CLOSURE_V1"
D1_READ_PENDING = "CORE_HOLDOUT_D1_HEALTH_PENDING_ROOT_CLOSURE_V1"
D1_SCORE_ROOT = "ROOT_ACCEPTS_CLOSED_CORE_HOLDOUT_D1_SCORING_OBSERVATIONS_V1"
D1_SCORE_REVIEW = "ACCEPT_CLOSED_CORE_HOLDOUT_D1_SCORING_OBSERVATIONS_NO_FINDINGS_V1"
D1_DESIGN_SHA256 = "98770dcd0ac6dffba73608fcad53a23e5a90c929a8432ed74b6d52d615b1c8dd"
D1_GPU_SCHEMA = "core_bank_learner_same_gpu_payload_proof_D1_v1"
D1_GPU_STATUS = "KL_DEVELOPMENT_FIRST_D1_GPU_OWN_LIVE_SAVED_PROOF_PENDING_ROOT_CLOSURE_V1"
D1_GPU_ADMISSION = "ROOT_ADMITS_KL_DEVELOPMENT_FIRST_D1_LEARNER_GPU_V1"
D1_GPU_SOURCE_REVIEW = "ACCEPT_KL_DEVELOPMENT_FIRST_D1_PASSIVE_PROOF_SOURCE_ONLY_NO_FINDINGS_V1"
D1_GPU_HELPER = "cceb3d1de58762e8557d7df398c50582c4c56519122e227c99057449d664087d"
D1_PROFILE_SHA256 = "ab5cf42f0d7790d564c88af589f0d47f1455032f379347ef9c3038d68de39242"
D1_PROFILE_SELECTION_SHA256 = "e5faa0b1652146b8bbf0185e868d25a64d1d8c64848b68e5f849c6e138cefa9a"
D1_COEFFICIENTS = {"K0": 0, "K001": 0.01, "K01": 0.1, "K1": 1}
D1_MATCH_SETTINGS = {"matched_sims": 64, "temperature": 0.1, "root_noise": True, "compile": "off", "autocast": "bf16",
                     "transport": "float32", "history_plies": 16, "max_additional_plies": 450,
                     "strict_tb_min": 3, "strict_tb_max": 6, "rule50": True}
D1_SELECTION_ROOT = "ROOT_ACCEPTS_CLOSED_POLICY_ANCHOR_D1_DEVELOPMENT_SELECTION_V1"
D1_SELECTION_REVIEW = "ACCEPT_CLOSED_POLICY_ANCHOR_D1_DEVELOPMENT_SELECTION_NO_FINDINGS_V1"
D1_EDGE_ROOT = "ROOT_ACCEPTS_CLOSED_POLICY_ANCHOR_D1_DEVELOPMENT_EDGE_V1"
D1_EDGE_REVIEW = "ACCEPT_CLOSED_POLICY_ANCHOR_D1_DEVELOPMENT_EDGE_NO_FINDINGS_V1"


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
                 _ruler_phase: bool = False, _d1_proof: bool = False, _d1_dev: bool = False, _d1_role: bool = False) -> dict[str, Any]:
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
    require(sum((_same_gpu_proof, _ruler_phase, _d1_proof, _d1_dev, _d1_role)) <= 1, "conflicting owner admission routes")
    if _d1_proof or _d1_dev or _d1_role:
        expected_cap = 3600 if _d1_proof else 900 if resource == "gpu" else 300
        expected_status = D1_GPU_ADMISSION if _d1_proof else "ROOT_ADMITS_KL_DEVELOPMENT_FIRST_D1_PREFIX_CPU_V1" if _d1_role else (
            "ROOT_ADMITS_D1_DEVELOPMENT_GPU_V1" if resource == "gpu" else "ROOT_ADMITS_D1_DEVELOPMENT_READER_CPU_V1")
        require(resource in (("gpu",) if _d1_proof else ("cpu",) if _d1_role else ("gpu", "cpu")) and seconds == cap == expected_cap
                and admission["status"] == expected_status and admission["execution_authorized"] is True, "D1 owner admission differs")
        plan_key, plan_flag = ("observer_plan", "--observer-plan") if _d1_proof else ("protocol", "--protocol") if _d1_role else ("development_plan", "--development-plan")
        for flag, value in (("--admission", owner["admission"]["path"]), ("--admission-sha256", owner["admission"]["sha256"]),
                            (plan_flag, admission[plan_key]["path"]), (plan_flag + "-sha256", admission[plan_key]["sha256"])):
            require(command.count(flag) == 1 and command.index(flag) + 1 < len(command)
                    and command[command.index(flag) + 1] == value, "D1 owner command differs")
        normalized = list(command)
        normalized[normalized.index("--admission-sha256") + 1] = "SELF_ADMISSION_SHA256"
        require(admission["normalized_child"] == normalized and admission["resource"] == resource
                and admission["seconds"] == cap and admission["owner"]["sha256"] == terminal["supervisor_sha256"], "D1 normalized owner differs")
    elif _same_gpu_proof:
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


def model_delivery(m: Metadata, binding: dict[str, Any], label: str, *, _d1: bool = False) -> dict[str, Any]:
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
    require(not _d1 or audit.get("schema") == D1_GPU_SCHEMA, "D1 requires its versioned same-owner proof")
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
        saved = same_gpu_payload(m, root, c, b, suffix, _d1=_d1)
    bank.sha(saved["model_state_sha256"])
    membership = b["membership"]
    require(type(membership) is list and membership and all(type(row) is list and len(row) == 3
            and type(row[0]) is int and row[0] == b["logical_generation"] and type(row[2]) is int and row[2] > 0 for row in membership), "invalid full replay membership")
    membership = cast(list[list[Any]], membership)
    for row in membership:
        bank.sha(row[1])
    train_rows = sum(row[2] for row in membership)
    require(train_rows >= 8000 and c["updates_per_segment"] == ([4096] if _d1 else [min((train_rows + 31) // 32, 1024)])
            and (not _d1 or train_rows == 17568)
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



def same_gpu_payload(m: Metadata, root: dict[str, Any], c: dict[str, Any], b: dict[str, Any], suffix: list[str],
                     *, _d1: bool = False) -> dict[str, Any]:
    """Versioned saved proof from the SAME closed learner GPU owner, JSON only."""
    audit = root["payload_audit"]
    require(set(audit) == {"schema", "readout", "owner"} and audit["schema"] == (D1_GPU_SCHEMA if _d1 else GPU_PAYLOAD_SCHEMA), "invalid saved-state audit binding")
    require(audit["owner"] == root["learner_owner"], "payload proof must share complete learner owner")
    closed_owner(m, audit["owner"], resource="gpu", suffix=suffix, seconds=3600 if _d1 else 900,
                 _same_gpu_proof=not _d1, _d1_proof=_d1)
    saved = m.read(audit["readout"])
    require(saved["status"] == (D1_GPU_STATUS if _d1 else GPU_PAYLOAD_STATUS)
            and all(saved[key] == root[key] for key in ("contract", "stage", "boundary", "checkpoint"))
            and saved["source_revision"] == root["source_revision"] and saved["source_checkout"] == root["source_checkout"]
            and saved["architecture"] == c["architecture"] and type(saved["step"]) is int and saved["step"] == b["step"]
            and saved["public_suffix"] == suffix
            and saved["reconciled"] == dict.fromkeys(("model", "optimizer", "scheduler", "zclip", "counters", "rng", "journal", "membership", "policy_anchor"), True), "same-GPU saved-state proof incomplete")
    plan = m.read(saved["observer_plan"])
    admission = m.read(audit["owner"]["admission"])
    require(plan["schema"] == ("SCIENTIFIC_KL_DEVELOPMENT_FIRST_D1_PASSIVE_PROOF_V1" if _d1 else "SCIENTIFIC_ANCHORED_LEARNER_PASSIVE_PROOF_V1")
            and plan["status"] == ("BOUND_KL_DEVELOPMENT_FIRST_D1_PASSIVE_PROOF_NOT_DELIVERY_AUTHORITY_V1" if _d1 else "BOUND_SCIENTIFIC_LEARNER_PASSIVE_PROOF_NOT_DELIVERY_AUTHORITY_V1")
            and admission["observer_plan"] == saved["observer_plan"]
            and all(saved[key] == plan[key] == admission[key] for key in ("helper", "source_review", "source_pins"))
            and all(plan[key] == root[key] for key in ("contract", "stage")) and plan["public_suffix"] == suffix
            and plan["learner_output"] == str(Path(root["checkpoint"]["path"]).parent)
            and plan["observation"] == audit["readout"]["path"]
            and plan["limits"] == {"outer_seconds": 3600 if _d1 else 900, "inner_seconds": 3570 if _d1 else 870, "checkpoint_file_bytes": 8 << 30,
                                   "journal_bytes": (64 if _d1 else 16) << 20, "aggregate_bytes": 4 << 20}
            and type(saved["outer_seconds"]) is int and saved["outer_seconds"] == (3600 if _d1 else 900)
            and type(saved["absolute_inner_seconds"]) is int and saved["absolute_inner_seconds"] == (3570 if _d1 else 870),
            "same-GPU plan/admission proof differs")
    if _d1:
        bank.reference(plan["design"])
        require(plan["design"]["sha256"] == D1_DESIGN_SHA256 and plan["arm"] in D1_COEFFICIENTS
                and c["policy_anchor"]["coefficient"] == D1_COEFFICIENTS[plan["arm"]], "D1 proof arm/design differs")
    helper = saved["helper"]
    require(type(helper) is dict and set(helper) == {"path", "sha256"} and type(helper["path"]) is str
            and Path(helper["path"]).is_absolute() and Path(helper["path"]).suffix == ".py"
            and helper["sha256"] in ((D1_GPU_HELPER,) if _d1 else (GPU_PROOF_HELPER_SHA256, GPU_PROOF_THREAD2_HELPER_SHA256))
            and audit["owner"]["command"].count(helper["path"]) == 1,
            "unsupported same-GPU prover")
    review = m.read(saved["source_review"])
    require(review["status"] == (D1_GPU_SOURCE_REVIEW if _d1 else GPU_PROOF_SOURCE_REVIEW) and review["helper"] == helper
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
    if _d1 or helper["sha256"] == GPU_PROOF_THREAD2_HELPER_SHA256:
        require(type(saved.get("Torch_intra")) is int and saved["Torch_intra"] == 2
                and type(saved.get("Torch_interop")) is int and saved["Torch_interop"] == 2
                and type(initial["backend"]) is dict
                and type(initial["backend"].get("torch_threads")) is int and initial["backend"]["torch_threads"] == 2
                and type(initial["backend"].get("interop_threads")) is int and initial["backend"]["interop_threads"] == 2,
                "same-GPU thread-2 evidence differs")
    membership = b["membership"]
    require(type(membership) is list and membership and all(type(row) is list and len(row) == 3
            and type(row[0]) is int and row[0] == 1 and type(row[2]) is int and row[2] > 0 for row in membership),
            "invalid full replay membership")
    membership = cast(list[list[Any]], membership)
    population = sum(row[2] for row in membership)
    coverage = saved["draw_coverage"]
    require(type(plan["N"]) is int and plan["N"] == population and type(plan["K"]) is int
            and plan["K"] == b["step"] == (4096 if _d1 else min((population + 31) // 32, 1024))
            and (not _d1 or population == 17568)
            and type(coverage["population_rows"]) is int and coverage["population_rows"] == population
            and type(coverage["batches"]) is int and coverage["batches"] == b["step"]
            and type(coverage["views"]) is int and coverage["views"] == b["drawn_rows"] == b["applied_views"]
            and coverage["journal_sha256"] == b["draws_sha256"]
            and type(coverage["views_per_population_row"]) is float
            and coverage["views_per_population_row"] == coverage["views"] / population, "same-GPU draw totals differ")
    return saved


def same_gpu_game_coverage(m: Metadata, binding: dict[str, Any], model: dict[str, Any], train: bank.AcceptedGeneration, *, _d1: bool = False) -> None:
    """Join the new aggregate roster to the actual accepted TRAIN, no draws read."""
    root = m.read(binding["root"])
    audit = root["payload_audit"]
    if set(audit) == {"readout", "owner"}:
        return
    require(set(audit) == {"schema", "readout", "owner"} and audit["schema"] == (D1_GPU_SCHEMA if _d1 else GPU_PAYLOAD_SCHEMA), "invalid saved-state audit binding")
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
    if "schema" in donor:
        return _current_integrated_asset_identity(m, donor)
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


def _current_integrated_asset_identity(m: Metadata, donor: dict[str, Any]) -> dict[str, Any]:
    """Versioned metadata-only original-A39 identity, never a legacy status alias."""
    require(set(donor) == {"schema", "checkpoint", "architecture", "step", "model_state_sha256", "root",
                           "independent", "asset_source"}
            and donor["schema"] == "core_a39_current_integrated_asset_v1", "invalid current-integrated A39 binding")
    asset, independent = m.read(donor["root"]), m.read(donor["independent"])
    readout = m.read(asset["readout"])
    source = donor["asset_source"]
    require(set(source) == {"revision", "checkout", "source_pins"}, "invalid current-integrated asset source binding")
    bank.sha(source["revision"], 40)
    bank.reference(source["source_pins"])
    require(asset["status"] == "ROOT_ACCEPTS_CURRENT_INTEGRATED_COLLECTOR_CPU_ASSET_READINESS_V1"
            and independent["status"] == "ACCEPT_CLOSED_CURRENT_INTEGRATED_COLLECTOR_CPU_ASSET_READINESS_NO_FINDINGS_V1"
            and independent["root_acceptance"] == donor["root"] and independent["findings"] == []
            and independent["readout"] == asset["readout"]
            and independent["protocol"] == readout["protocol"] == asset["protocol"]
            and independent["terminal"] == asset["terminal"], "current-integrated A39 asset acceptance differs")
    require(all(value["source_revision"] == source["revision"] and value["source_checkout"] == source["checkout"]
                and value["source_pins"] == source["source_pins"] for value in (asset, independent, readout))
            and readout["status"] == "COMPLETE_CURRENT_INTEGRATED_COLLECTOR_CPU_ASSET_QUALIFICATION_CHILD_ROOT_CLOSURE_REQUIRED_V1",
            "current-integrated asset source/readout differs")
    observed = readout["donor_model_only"]
    require(asset["donor_model_only"] == observed and observed["actual_CPU_model_only_load"] is True,
            "current-integrated model-only observation differs")
    require(type(observed["state_tensors"]) is int and observed["state_tensors"] == 494
            and observed["counts"] == {"input_planes": 175, "state_entries": 494,
                                       "unique_parameters": 61444448, "naive_state_elements": 77173088}
            and observed["optimizer_restore_qualified"] is False, "complete current-integrated A39 observation differs")
    require(donor["checkpoint"]["sha256"] == "a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0",
            "not original A39 donor")
    require(observed["sha256"] == donor["checkpoint"]["sha256"] and observed["path"] == donor["checkpoint"]["path"]
            and type(observed["embedded_step"]) is int and type(donor["step"]) is int
            and observed["embedded_step"] == donor["step"] == 680276
            and observed["architecture"] == donor["architecture"]
            and observed["state_sha256"] == donor["model_state_sha256"]
            == "9796079cd73bef73ba2f5d5f2ad0779704498832e34a7c2a880f0d0b7f305229",
            "current-integrated A39 observed identity differs")
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


def is_d1(p: dict[str, Any]) -> bool:
    return p.get("schema") in (D1_SCORE_SCHEMA, D1_READ_SCHEMA)


def protocol_labels(p: dict[str, Any]) -> tuple[str, ...]:
    return metrics.d1_labels(p["selected_arm"]) if is_d1(p) else metrics.LABELS


def _d1_model(m: Metadata, binding: dict[str, Any], label: str, donor: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    require(label in D1_COEFFICIENTS, "invalid D1 arm")
    model = model_delivery(m, binding, label, _d1=True)
    train = training_population(m, model)
    same_gpu_game_coverage(m, binding, model, train, _d1=True)
    expected_bank = design["foundations"]["TRAIN_bank"]
    require(model["stage"]["banks"] == [{"root_acceptance": expected_bank["root"], "independent_review": expected_bank["independent"]}]
            and len(train.games) == 192 and train.evidence["eligible_rows"] == 17568, "D1 exact accepted TRAIN differs")
    c = model["contract"]
    anchor = c["policy_anchor"]
    weight = D1_COEFFICIENTS[label]
    require(c["initial_model"] == donor["checkpoint"] and c["initial_actor_step"] == donor["step"]
            and c["architecture"] == donor["architecture"] and type(c["seed"]) is int and c["seed"] == 2026111673
            and type(c["batch_size"]) is int and c["batch_size"] == 32 and c["mirror_prob"] == 0.5
            and canonical(c["updates_per_segment"]) == canonical([4096])
            and canonical(c["lr"]) == canonical({"peak": 3e-6, "floor": 3e-6, "warmup_start": 3e-6, "warmup_steps": 0, "first_period": 4096}),
            "D1 exact bootstrap/dose/LR differs")
    require(set(anchor) == {"schema", "direction", "mask", "coefficient", "reference"}
            and anchor["schema"] == "frozen_reference_policy_anchor_v1"
            and anchor["direction"] == "reference_to_student_forward_KL"
            and anchor["mask"] == "original_network_turn_has_policy_with_explicit_legal_mask_v1"
            and type(anchor["coefficient"]) in (int, float) and anchor["coefficient"] == weight
            and (anchor["reference"] is None if weight == 0 else
                 anchor["reference"]["checkpoint"] == donor["checkpoint"]
                 and anchor["reference"]["checkpoint_step"] == donor["step"]
                 and anchor["reference"]["architecture"] == donor["architecture"]), "D1 anchor identity differs")
    delivery = m.read(binding["root"])
    saved = m.read(delivery["payload_audit"]["readout"])
    require(c["source_revision"] == "bbb4b725a2d5218ffa4630841dd93e8b2884b58d"
            and saved["initial_model_state_sha256"] == donor["model_state_sha256"], "D1 initial model/source differs")
    return model


def _d1_paired_point(counts: Any) -> float:
    require(type(counts) is list and len(counts) == 5 and all(type(value) is int and value >= 0 for value in counts)
            and sum(counts) == 80, "D1 DEV requires80 complete paired observations")
    return sum(index * value for index, value in enumerate(counts)) / 320


def d1_playing_projection(profile: dict[str, Any]) -> dict[str, Any]:
    """Selected active playing inputs, distinct from collector-only recording knobs.

    Future DEV source must observe the actual two search objects/call arguments
    against this projection; the consumer does not infer observations from flags.
    The constant .75 selfplay schedule is an explicit flat arena override.
    """
    configs = profile["configs"]
    search, game, temp = configs["search"], configs["game"], configs["temp"]
    require(search["gumbel_scale"] == search["gumbel_scale_after"] == 0.75
            and all(temp[key] == 0.1 for key in ("temperature", "after", "endgame", "selfplay_temperature", "selfplay_endgame")),
            "D1 selected constant noise/temperature differs")
    return {"gumbel": {"simulations": search["simulations"], "topk": search["gumbel_topk"], "temperature": 0.1,
                "policy_temp": search["gumbel_policy_temp"], "c_scale": search["gumbel_c_scale"],
                "target_max_visit_cap": search["gumbel_target_max_visit_cap"], "target_untempered_prior": search["gumbel_target_untempered_prior"],
                "add_noise": True, "gumbel_scale": search["gumbel_scale"],
                "input_history_encoding": game["input_history_encoding"], "input_extra_features": game["input_extra_features"],
                "policy_encoding": game["policy_encoding"], "compute_relations": game["record_relations"],
                "volatility_q_scale": search["volatility_q_scale"], "volatility_fpu": search["volatility_fpu"], "volatility_anchor": search["volatility_anchor"]},
            "runner": {"vloss_weight": search["gumbel_vloss_weight"], "target_batch": search["gumbel_target_batch"],
                       "history_rep_fix": game["history_rep_fix"], "tree_reuse": False},
            "game": {key: game[key] for key in ("search_wdl_draw_mode", "syzygy_in_search", "syzygy_adjudicate", "syzygy_adjudicate_fraction",
                       "syzygy_outcome_mode", "syzygy_path")},
            "max_additional_plies": game["max_plies"], "history_plies": 16,
            "serving": {"compile": "off", "autocast": "bf16", "transport": "float32", "explicit_legal_mask": True}}


def d1_collector_only_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Pinned label construction provenance, never observed arena arguments."""
    return {"configs/game/" + key: {"selected_source_value": profile["configs"]["game"][key],
                                  "playing_applicability": "NOT_APPLICABLE_LABEL_ONLY"}
            for key in ("categorical_blend_frac", "categorical_search_blend_frac", "syzygy_rescore_policy")}


def _d1_dev_edge(m: Metadata, binding: dict[str, Any], candidate: str, reference: str,
                 models: dict[str, Any], p: dict[str, Any], profile: dict[str, str], role_use: dict[str, Any],
                 dev_source: dict[str, Any], panels: dict[str, Any]) -> float:
    require(set(binding) == {"root", "independent"}, "invalid D1 DEV binding")
    root, review = m.read(binding["root"]), m.read(binding["independent"])
    keys = {"status", "study_uuid", "design", "candidate", "reference", "models", "profile", "role_use", "source",
            "protocol", "reader_protocol", "readout", "summary", "gpu_owner", "reader_owner"}
    require(set(root) == keys and root["status"] == D1_EDGE_ROOT and root["study_uuid"] == p["study_uuid"]
            and root["design"] == p["design"] and root["candidate"] == candidate and root["reference"] == reference
            and root["models"] == {"candidate": models[candidate]["checkpoint"], "reference": models[reference]["checkpoint"]}
            and root["profile"] == profile and root["role_use"] == role_use and root["source"] == dev_source, "D1 DEV edge identity differs")
    require(set(review) == keys | {"root_acceptance", "findings"} and review["status"] == D1_EDGE_REVIEW
            and review["root_acceptance"] == binding["root"] and review["findings"] == []
            and all(review[key] == root[key] for key in keys - {"status"}), "D1 DEV independent review differs")
    protocol, reader = m.read(root["protocol"]), m.read(root["reader_protocol"])
    readout, summary = m.read(root["readout"]), m.read(root["summary"])
    panel = "vs_A39" if reference == "A39" else "vs_K0"
    seed = 2026111674 if reference == "A39" else 2026111675
    for value in (protocol, readout, summary):
        require(all(value[key] == root[key] for key in ("study_uuid", "design", "candidate", "reference", "models", "profile", "role_use", "source")),
                "D1 DEV protocol/readout projection differs")
    require(protocol["schema"] == "CORE_POLICY_ANCHOR_D1_DEVELOPMENT_EDGE_V1" and protocol["panel"] == panel
            and protocol["seed"] == seed and protocol["opening_book"] == panels[panel]["book"]
            and protocol["outer_seconds"] == 900 and protocol["inner_seconds"] == 870
            and reader["schema"] == "CORE_POLICY_ANCHOR_D1_DEVELOPMENT_READER_V1"
            and reader["protocol"] == root["protocol"] and reader["readout"] == root["readout"]
            and reader["outer_seconds"] == 300 and reader["inner_seconds"] == 270,
            "D1 DEV fixed protocol/caps differs")
    require(m.read(root["gpu_owner"]["admission"])["development_plan"] == root["protocol"]
            and m.read(root["reader_owner"]["admission"])["development_plan"] == root["reader_protocol"], "D1 DEV owner plan differs")
    for owner, phase_protocol, required_status in (
            (root["gpu_owner"], protocol, "ACCEPT_D1_DEVELOPMENT_GPU_SOURCE_ONLY_NO_FINDINGS_V1"),
            (root["reader_owner"], reader, "ACCEPT_D1_DEVELOPMENT_READER_SOURCE_ONLY_NO_FINDINGS_V1")):
        admission = m.read(owner["admission"])
        source_review = m.read(phase_protocol["source_review"])
        require(admission["helper"] == phase_protocol["helper"] and admission["source_review"] == phase_protocol["source_review"]
                and source_review["status"] == required_status and source_review["helper"] == phase_protocol["helper"]
                and source_review["source"] == dev_source and source_review["findings"] == []
                and owner["command"].count(phase_protocol["helper"]["path"]) == 1, "D1 DEV source helper differs")
    closed_owner(m, root["gpu_owner"], resource="gpu", suffix=protocol["public_suffix"], seconds=900, _d1_dev=True)
    closed_owner(m, root["reader_owner"], resource="cpu", suffix=reader["public_suffix"], seconds=300, _d1_dev=True)
    require(readout["status"] == "D1_DEVELOPMENT_EDGE_PENDING_ROOT_CLOSURE_V1"
            and summary["status"] == "D1_DEVELOPMENT_EDGE_RECONCILED_PENDING_ROOT_CLOSURE_V1"
            and summary["protocol"] == root["protocol"] and summary["readout"] == root["readout"]
            and summary["pairs_completed"] == readout["pairs_completed"] == 80
            and summary["games_completed"] == readout["games_completed"] == 160
            and summary["incomplete_pairs"] == readout["incomplete_pairs"] == 0
            and summary["illegal_or_crashed_games"] == readout["illegal_or_crashed_games"] == 0
            and summary["reciprocal_colours_complete"] is readout["reciprocal_colours_complete"] is True
            and summary["pair_order_verified"] is True and summary["all_game_rows_reconciled"] is True,
            "D1 DEV technical evidence incomplete")
    expected_projection = m.read(profile)["active_playing_projection"]
    require(canonical(protocol["active_playing_projection"]) == canonical(expected_projection)
            and canonical(readout["observed_playing_projection"]) == canonical({"candidate": expected_projection, "reference": expected_projection})
            and canonical(summary["observed_playing_projection"]) == canonical(readout["observed_playing_projection"])
            and readout["actual_search_objects_checked"] is True and readout["actual_runner_arguments_checked"] is True,
            "D1 DEV active playing projection differs")
    require(summary["paired_counts"] == readout["paired_counts"] and canonical(summary["effective_settings"]) == canonical(readout["effective_settings"])
            == canonical(D1_MATCH_SETTINGS), "D1 DEV realised profile/counts differ")
    for side, label in (("candidate", candidate), ("reference", reference)):
        observed = readout["served_states"][side]
        require(observed["pre"] == observed["post"] == models[label]["model_state_sha256"], "D1 DEV served state differs")
    require(summary["pair_order_sha256"] == panels[panel]["parent_order_sha256"]
            and summary["history_order_sha256"] == panels[panel]["history_order_sha256"], "D1 DEV exact prefix order differs")
    bank.sha(summary["pair_order_sha256"])
    bulk_reference(summary["game_log"], suffix=".jsonl")
    wdl = summary["wdl_counts"]
    require(wdl == readout["wdl_counts"] and type(wdl) is dict and set(wdl) == {"wins", "draws", "losses"}
            and all(type(value) is int and value >= 0 for value in wdl.values()) and sum(wdl.values()) == 160,
            "D1 DEV complete WDL counts differ")
    wdl = cast(dict[str, int], wdl)
    point = _d1_paired_point(summary["paired_counts"])
    require(type(summary["paired_score"]) in (int, float) and math.isfinite(summary["paired_score"])
            and summary["paired_score"] == readout["paired_score"] == point
            and (wdl["wins"] + 0.5 * wdl["draws"]) / 160 == point, "D1 DEV paired score differs")
    return point


def bulk_reference(value: Any, *, suffix: str) -> Path:
    """Validate a typed bulk descriptor without opening its payload.

    Canonical-path resolution has the same semantics as metadata references.
    Payload bytes, size and readonly custody belong to the accepted producer and
    the later owned consumer; this descriptor check does not attest to them.
    """
    require(type(value) is dict and set(value) == {"path", "sha256"}, "invalid bulk reference")
    bank.sha(value["sha256"])
    require(type(value["path"]) is str, "invalid bulk path")
    path = Path(cast(str, value["path"]))
    require(path.is_absolute() and path.suffix == suffix and path.resolve() == path, "noncanonical bulk path")
    return path


def _d1_role_panels(m: Metadata, binding: dict[str, Any], p: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    """Consume the actual metadata-only prefix producer, never a second role phase."""
    require(set(binding) == {"root", "independent"}, "invalid D1 role-use binding")
    root, review = m.read(binding["root"]), m.read(binding["independent"])
    keys = {"status", "study_uuid", "design", "source_roles", "prefixes", "original_six400_books", "protocol", "readout", "owner"}
    require(set(root) == keys and root["status"] == "ROOT_ACCEPTS_CLOSED_KL_DEVELOPMENT_FIRST_D1_PREFIX_METADATA_V1"
            and root["study_uuid"] == p["study_uuid"] and root["design"] == p["design"]
            and root["source_roles"] == design["roles"], "D1 role-use authority differs")
    require(set(review) == keys | {"root_acceptance", "findings"}
            and review["status"] == "ACCEPT_CLOSED_KL_DEVELOPMENT_FIRST_D1_PREFIX_METADATA_NO_FINDINGS_V1"
            and review["root_acceptance"] == binding["root"] and review["findings"] == []
            and all(review[key] == root[key] for key in keys - {"status"}), "D1 role-use independent review differs")
    protocol, readout = m.read(root["protocol"]), m.read(root["readout"])
    require(protocol["schema"] == "D1_PREFIX_METADATA_CPU_PROTOCOL_V1" and protocol["status"] == "BOUND_D1_PREFIX_METADATA_ONLY_V1"
            and protocol["design"] == p["design"] and protocol["study_uuid"] == p["study_uuid"]
            and {"root": protocol["role_root"], "independent": protocol["role_independent"]} == design["roles"]
            and protocol["limits"]["outer_seconds"] == 300 and protocol["limits"]["inner_seconds"] == 270
            and readout["status"] == "D1_PREFIX_METADATA_CANDIDATE_PENDING_ROOT_CLOSURE_V1" and readout["protocol"] == root["protocol"]
            and readout["design"] == p["design"] and readout["role_root"] == protocol["role_root"]
            and readout["role_independent"] == protocol["role_independent"] and readout["prefixes"] == root["prefixes"]
            and readout["original_six400_books"] == root["original_six400_books"]
            and readout["complete_parent_descriptors_preserved"] is True and readout["controls_passed"] == 12,
            "D1 role-use readout differs")
    domains = readout["all8role_domains_disjoint"]
    require(type(domains) is dict and set(domains) == {"group", "parent", "history", "epd", "history_sha", "input"}
            and all(type(count) is int for count in domains.values()), "D1 actual prefix domain counts differ")
    domains = cast(dict[str, int], domains)
    require(all(domains[key] == 2656 for key in ("group", "history", "epd", "history_sha"))
            and domains["input"] == 5312 and domains["parent"] >= 2656, "D1 actual prefix domain counts differ")
    admission = m.read(root["owner"]["admission"])
    source_review = m.read(protocol["source_review"])
    require(admission["protocol"] == root["protocol"] and admission["helper"] == protocol["helper"]
            and admission["source_review"] == protocol["source_review"] and admission["source_pins"] == protocol["source_pins"]
            and source_review["status"] == "ACCEPT_KL_DEVELOPMENT_FIRST_D1_PREFIX_SOURCE_ONLY_NO_FINDINGS_V1"
            and source_review["helper"] == protocol["helper"] and source_review["controls"] == protocol["controls"]
            and source_review["source_pins"] == protocol["source_pins"] and source_review["findings"] == []
            and root["owner"]["command"].count(protocol["helper"]["path"]) == 1, "D1 role-use source helper differs")
    suffix = [protocol["helper"]["path"], "--protocol", root["protocol"]["path"], "--protocol-sha256", root["protocol"]["sha256"],
              "--admission", root["owner"]["admission"]["path"], "--admission-sha256", root["owner"]["admission"]["sha256"]]
    closed_owner(m, root["owner"], resource="cpu", suffix=suffix, seconds=300, _d1_role=True)
    original, original_ind = m.read(design["roles"]["root"]), m.read(design["roles"]["independent"])
    require(original["status"] == "ROOT_ACCEPTS_CLOSED_PROSPECTIVE_WHOLE_GAME_ROLES_V1"
            and original_ind["status"] == "ACCEPT_CLOSED_PROSPECTIVE_WHOLE_GAME_ROLES_NO_FINDINGS_V1"
            and original_ind["root_acceptance"] == design["roles"]["root"] and original_ind["findings"] == []
            and root["original_six400_books"] == original["books"] and len(original["books"]) == 6, "D1 original role authority differs")
    require(set(root["prefixes"]) == {"SCREEN_C_A", "SCREEN_C_B"}, "D1 requires both complete DEV panels")
    panels = {}
    for panel, selected in (("vs_A39", design["DEV"]["panel_vs_A39"]), ("vs_K0", design["DEV"]["panel_vs_K0"])):
        value = root["prefixes"][selected["source_role"]]
        require(set(value) == {"source_role", "original400_book", "original_full_role", "book", "parent_map", "selection", "prefix_start", "pairs",
                               "complete_parents", "parent_order_sha256", "history_order_sha256", "source_count", "actual_public_loader_count", "no_skips_dedup_shrink"}
                and value["source_role"] == selected["source_role"] and value["original400_book"] == selected["source_book"]
                == original["books"][selected["source_role"]] and value["original_full_role"] == original["roles"][selected["source_role"]]
                and value["selection"] == "exact original ordinals0..79; no rng/subsample/reorder"
                and value["prefix_start"] == 0 and value["pairs"] == value["complete_parents"] == value["actual_public_loader_count"] == 80
                and value["source_count"] == 400 and value["no_skips_dedup_shrink"] is True, "D1 exact prefix materialization differs")
        bulk_reference(value["book"], suffix=".epd")
        bank.reference(value["parent_map"])
        bank.sha(value["parent_order_sha256"])
        bank.sha(value["history_order_sha256"])
        panels[panel] = value
    consumed = readout["public_consumption"]
    expected = [(ref, 400) for ref in original["books"].values()] + [(value["book"], 80) for value in root["prefixes"].values()]
    require(type(consumed) is list and len(consumed) == 8
            and all(sum(row == {"book": ref, "pairs": count, "actual_public_reads": 1, "full16_history_order_EPD_exact": True}
                        for row in consumed) == 1 for ref, count in expected), "D1 eight actual public prefix/book reads differ")
    require(all(root["original_six400_books"][selected["source_role"]] == selected["source_book"]
                for selected in (design["final"]["vs_A39"], design["final"]["vs_K0_if_positive_KL"])), "D1 reserved final books differ")
    return panels

def d1_selection(m: Metadata, p: dict[str, Any], donor: dict[str, Any], design: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    binding = p["d1"]
    require(set(binding) == {"root", "independent"}, "invalid D1 selection binding")
    root, review = m.read(binding["root"]), m.read(binding["independent"])
    keys = {"status", "study_uuid", "design", "arms", "dev_edges", "selected_arm", "profile", "role_use", "dev_source"}
    require(set(root) == keys and root["status"] == D1_SELECTION_ROOT and root["study_uuid"] == p["study_uuid"]
            and root["design"] == p["design"] and set(root["arms"]) == set(D1_COEFFICIENTS), "invalid D1 selected registry")
    require(set(review) == keys | {"root_acceptance", "findings"} and review["status"] == D1_SELECTION_REVIEW
            and review["root_acceptance"] == binding["root"] and review["findings"] == []
            and all(review[key] == root[key] for key in keys - {"status"}), "D1 selection independent review differs")
    models = {label: _d1_model(m, root["arms"][label], label, donor, design) for label in D1_COEFFICIENTS}
    require(len({model["contract"]["run_uuid"] for model in models.values()}) == 4, "D1 arms must have distinct run identities")
    base = models["K0"]
    for label in ("K001", "K01", "K1"):
        model = models[label]
        require({key: value for key, value in model["contract"].items() if key not in ("run_uuid", "policy_anchor")}
                == {key: value for key, value in base["contract"].items() if key not in ("run_uuid", "policy_anchor")}
                and all(model["boundary"][key] == base["boundary"][key] for key in ("membership", "draws_sha256", "rng", "backend")),
                "D1 arms are not matched")
    models["A39"] = donor
    expected = {label + "_vs_A39" for label in D1_COEFFICIENTS} | {label + "_vs_K0" for label in ("K001", "K01", "K1")}
    require(set(root["dev_edges"]) == expected, "D1 all seven DEV edges required")
    profile = m.read(root["profile"])
    require(set(profile) == {"schema", "design", "effective_settings", "full_profile", "selection_authority", "active_playing_projection", "source_only_not_applicable"}
            and profile["schema"] == "D1_MATCH_PROFILE_V1" and profile["design"] == p["design"]
            and canonical(profile["effective_settings"]) == canonical(D1_MATCH_SETTINGS), "D1 fixed match profile differs")
    bank.reference(profile["full_profile"])
    bank.reference(profile["selection_authority"])
    require(profile["full_profile"]["sha256"] == D1_PROFILE_SHA256
            and profile["selection_authority"]["sha256"] == D1_PROFILE_SELECTION_SHA256, "D1 selected profile authority differs")
    selected = m.read(profile["selection_authority"])
    require(selected["status"] == "ROOT_SELECTS_FINAL_PROSPECTIVE_COLLECTOR_PROFILE_AND_ROLES_SOURCE_ONLY_V1"
            and selected["selected_profile"] == profile["full_profile"], "D1 profile selection differs")
    full_profile = m.read(profile["full_profile"])
    require(canonical(profile["active_playing_projection"]) == canonical(d1_playing_projection(full_profile)),
            "D1 active playing projection differs")
    require(canonical(profile["source_only_not_applicable"]) == canonical(d1_collector_only_profile(full_profile)),
            "D1 collector-only label provenance differs")
    require(type(root["dev_source"]) is dict and set(root["dev_source"]) == {"revision", "checkout", "files"}, "invalid D1 DEV source")
    dev_source = cast(dict[str, Any], root["dev_source"])
    bank.sha(dev_source["revision"], 40)
    checkout = dev_source["checkout"]
    require(type(checkout) is str and Path(checkout).is_absolute(), "invalid D1 DEV checkout")
    require(type(dev_source["files"]) is dict and dev_source["files"], "empty D1 DEV source pins")
    source_files = cast(dict[Any, Any], dev_source["files"])
    for name, digest in source_files.items():
        require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts, "invalid D1 DEV source file")
        bank.sha(digest)
    panels = _d1_role_panels(m, root["role_use"], p, design)
    points = {}
    for label in D1_COEFFICIENTS:
        points[label] = _d1_dev_edge(m, root["dev_edges"][label + "_vs_A39"], label, "A39", models, p, root["profile"], root["role_use"], root["dev_source"], panels)
    comparisons = {label: _d1_dev_edge(m, root["dev_edges"][label + "_vs_K0"], label, "K0", models, p, root["profile"], root["role_use"], root["dev_source"], panels)
                   for label in ("K001", "K01", "K1")}
    eligible = [label for label in D1_COEFFICIENTS if points[label] >= 0.55 and (label == "K0" or comparisons[label] >= 0.55)]
    require(eligible, "D1 no eligible DEV arm; HOLD forbidden")
    winner = min(eligible, key=lambda label: (-points[label], D1_COEFFICIENTS[label]))
    require(root["selected_arm"] == p["selected_arm"] == winner, "D1 winner differs from complete DEV selection")
    return winner, models


def d1_bindings(m: Metadata, p: dict[str, Any]) -> tuple[hold.AcceptedHoldout, dict[str, str], dict[str, Any]]:
    bank.reference(p["design"])
    require(p["design"]["sha256"] == D1_DESIGN_SHA256, "wrong D1 design")
    design = m.read(p["design"])
    require(design["study_uuid"] == p["study_uuid"] and design["design_id"] == "D1", "foreign D1 study")
    require(p["holdout"] == design["foundations"]["HOLD_bank"], "D1 exact HOLD bank differs")
    view = hold.validate_accepted_holdout(p["holdout"]["root"], p["holdout"]["independent"],
        max_file_bytes=p["limits"]["metadata_file_bytes"], max_total_bytes=p["limits"]["metadata_total_bytes"], max_games=64)
    parents = parent_map(m, p, view)
    require(view.identity["bank"] == "holdout" and len(view.games) == 64
            and sum(game["rows"] for game in view.games) == 5943, "D1 whole HOLD coverage differs")
    labels = protocol_labels(p)
    require(set(p["models"]) == set(labels), "D1 only selected/control/donor models allowed")
    require(p["models"]["A39"].get("schema") == "core_a39_current_integrated_asset_v1", "D1 requires current-integrated A39 authority")
    donor = asset_identity(m, p["models"]["A39"])
    winner, all_models = d1_selection(m, p, donor, design)
    selected_root = m.read(p["d1"]["root"])
    require(all(p["models"][label] == selected_root["arms"][label] for label in labels if label != "A39"), "D1 selected model binding substituted")
    require(all(all_models[label]["architecture"] == plain(view.identity["architecture"]) for label in labels), "D1 HOLD/model architecture differs")
    require(all(identity["collection_uuid"] != view.identity["collection_uuid"] for label in D1_COEFFICIENTS
                for identity in all_models[label]["boundary"]["bank_identities"]), "D1 HOLD collection leaked into learning")
    require(winner in labels, "missing selected D1 model")
    return view, parents, {label: all_models[label] for label in labels}


def bindings(m: Metadata, p: dict[str, Any]) -> tuple[hold.AcceptedHoldout, dict[str, str], dict[str, Any]]:
    if is_d1(p):
        return d1_bindings(m, p)
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
    d1 = p.get("schema") in (D1_SCORE_SCHEMA, D1_READ_SCHEMA)
    schema = (D1_READ_SCHEMA if reader else D1_SCORE_SCHEMA) if d1 else (READ_SCHEMA if reader else SCORE_SCHEMA)
    keys = {"schema", "study_uuid", "design", "holdout", "parents", "models", "source", "runtime", "owner", "stage",
            "outer_seconds", "inner_seconds", "output", "limits", "source_review", "native_basis"}
    if reader:
        keys |= {"scoring"}
    if d1:
        keys |= {"d1", "selected_arm"}
    require(set(p) == keys and p["schema"] == schema, "wrong ruler protocol")
    if d1:
        require(p["design"]["sha256"] == D1_DESIGN_SHA256 and p["selected_arm"] in D1_COEFFICIENTS, "unselected D1 protocol")
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
