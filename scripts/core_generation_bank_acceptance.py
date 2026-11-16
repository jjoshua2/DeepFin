"""Join caller-trusted postclosure receipts without promoting candidate bytes.

This stdlib-only API reads bounded JSON metadata, never shard/model/source payloads.
Receipt status strings do not authenticate issuers: callers must trust both pins.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, cast
import uuid

ROOT_STATUS = "ROOT_ACCEPTS_CLOSED_CORE_GENERATION_DATA_BANK_V1"
REVIEW_STATUS = "ACCEPT_CLOSED_CORE_GENERATION_DATA_BANK_NO_FINDINGS_V1"
ADMISSION_STATUS = "ROOT_ADMITS_CORE_GENERATION_DATA_AUDIT_V1"
HOLDOUT_ROOT_STATUS = "ROOT_ACCEPTS_CLOSED_CORE_GENERATION_HOLDOUT_V1"
HOLDOUT_REVIEW_STATUS = "ACCEPT_CLOSED_CORE_GENERATION_HOLDOUT_NO_FINDINGS_V1"
HOLDOUT_ADMISSION_STATUS = "ROOT_ADMITS_CORE_GENERATION_HOLDOUT_AUDIT_V1"
SOURCE_FILES = frozenset({
    "scripts/core_finite_learner.py", "scripts/bootstrap_checkpoint_resume.py",
    "scripts/offline_epoch_resume.py", "chess_anti_engine/train/trainer.py",
    "chess_anti_engine/train/losses.py", "chess_anti_engine/config_keys.py",
    "chess_anti_engine/tune/fresh_rl_donor.py", "chess_anti_engine/replay/generation.py",
    "chess_anti_engine/replay/shard.py", "scripts/core_finite_admit.py",
    "scripts/core_generation_data_admit.py",
})
CLASSES = ("NATURAL", "STRICT_TB_ADJUDICATED", "UNRESOLVED")


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: Any, name: str, minimum: int = 1) -> int:
    require(type(value) is int and value >= minimum, "invalid " + name)
    return value


def sha(value: Any, length: int = 64) -> None:
    require(type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value), "invalid digest")


def reference(value: Any) -> Path:
    require(type(value) is dict and set(value) == {"path", "sha256"}, "invalid metadata reference")
    sha(value["sha256"])
    require(type(value["path"]) is str, "invalid metadata path")
    path = Path(cast(str, value["path"]))
    require(path.is_absolute() and path.suffix == ".json" and path.resolve() == path, "noncanonical metadata path")
    return path


def stamp(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not (value.st_mode & 0o222), "metadata must be regular and readonly")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def nonfinite(_value: str) -> Any:
    raise ValueError("nonfinite JSON constant")


def finite_float(value: str) -> float:
    number = float(value)
    require(math.isfinite(number), "nonfinite JSON number")
    return number


class _Metadata:
    def __init__(self, file_bytes: int, total_bytes: int):
        self.file_bytes = integer(file_bytes, "file byte limit")
        require(file_bytes <= (128 << 20), "metadata file limit exceeds128MiB")
        self.total_bytes = integer(total_bytes, "total byte limit")
        self.used = 0
        self.files: dict[Path, tuple[str, tuple[int, ...], dict[str, Any]]] = {}

    def read(self, ref: dict[str, str]) -> dict[str, Any]:
        path = reference(ref)
        if path in self.files:
            require(self.files[path][0] == ref["sha256"], "conflicting metadata reference")
            return self.files[path][2]
        before = stamp(path)
        require(before[2] <= self.file_bytes and self.used + before[2] <= self.total_bytes, "metadata byte limit")
        with path.open("rb") as stream:
            raw = stream.read(before[2] + 1)
        require(len(raw) == before[2] and stamp(path) == before, "metadata changed during read")
        require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "metadata digest differs")
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float)
        require(type(value) is dict, "metadata must be an object")
        self.used += len(raw)
        self.files[path] = (ref["sha256"], before, value)
        return value

    def finish(self) -> None:
        # Re-read only the already bounded metadata graph; atime is not custody.
        for path, (expected, before, _) in self.files.items():
            require(path.resolve() == path and stamp(path) == before, "metadata custody changed")
            with path.open("rb") as stream:
                raw = stream.read(before[2] + 1)
            require(len(raw) == before[2] and hashlib.sha256(raw).hexdigest() == expected
                    and stamp(path) == before, "metadata changed before return")


def frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(frozen(item) for item in value)
    return value


@dataclass(frozen=True)
class AcceptedGeneration:
    """Validated metadata view, not a legacy learner manifest or authority token."""

    identity: Mapping[str, Any]
    games: tuple[Mapping[str, Any], ...]
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class AcceptedHoldout:
    """Evaluation-only metadata view. Never an AcceptedGeneration/learner bank."""

    identity: Mapping[str, Any]
    games: tuple[Mapping[str, Any], ...]
    evidence: Mapping[str, Any]


def _spec(spec: dict[str, Any], *, holdout: bool = False) -> None:
    require(set(spec) == {"schema", "run_uuid", "collection_uuid", "generation", "source_revision", "source_files",
                         "architecture", "actor_sha256", "actor_checkpoint_step", "games_per_generation",
                         "max_games_per_generation"} and spec["schema"] == ("core_generation_holdout_data_spec_v1" if holdout else "core_generation_data_spec_v1"), "invalid data spec")
    for key in ("run_uuid", "collection_uuid"):
        require(type(spec[key]) is str and str(uuid.UUID(spec[key])) == spec[key], "invalid UUID")
    integer(spec["generation"], "generation")
    integer(spec["actor_checkpoint_step"], "actor step", 0)
    sha(spec["source_revision"], 40)
    sha(spec["actor_sha256"])
    require(type(spec["architecture"]) is dict and spec["architecture"], "missing architecture")
    require(integer(spec["games_per_generation"], "minimum games") <=
            integer(spec["max_games_per_generation"], "maximum games"), "invalid game bounds")
    pins = spec["source_files"]
    required = SOURCE_FILES | ({"scripts/core_finite_holdout_collect.py", "scripts/core_generation_holdout_admit.py"} if holdout else set())
    require(type(pins) is dict and pins.keys() >= required, "missing historical source pins")
    for name, digest in pins.items():
        require(type(name) is str and name and not Path(name).is_absolute()
                and str(Path(name)) == name and ".." not in Path(name).parts, "invalid historical source path")
        sha(digest)


def _candidate(value: dict[str, Any], spec: dict[str, Any], protocol: dict[str, str], spec_ref: dict[str, str], *, holdout: bool = False) -> None:
    require(value["audit_protocol"] == protocol and value["data_spec"] == spec_ref
            and value["collection_uuid"] == spec["collection_uuid"] and
            type(value["logical_generation"]) is int and value["logical_generation"] == spec["generation"]
            and type(value["actor_checkpoint_step"]) is int and value["actor_checkpoint_step"] == spec["actor_checkpoint_step"],
            "candidate provenance differs")
    if holdout:
        require(value["evaluation_only"] is True, "HOLDOUT purpose changed")
    require(value["candidate_only"] is True and value["learning_admitted"] is False
            and value["root_postclosure_promotion_required"] is True, "candidate authority changed")


def _owner(m: _Metadata, root: dict[str, Any], p: dict[str, Any], *, holdout: bool = False) -> None:
    owner = root["audit_owner"]
    require(set(owner) == {"terminal", "admission", "command", "source_revision", "cwd", "closure"}, "invalid audit owner binding")
    terminal = m.read(owner["terminal"])
    admission = m.read(owner["admission"])
    runtime = p["runtime"]
    require(set(runtime) == {"owner", "seconds", "inner_seconds", "cpu_affinity", "python"}, "invalid runtime")
    require(type(runtime["owner"]) is dict and set(runtime["owner"]) == {"path", "sha256"}
            and type(runtime["owner"]["path"]) is str and Path(runtime["owner"]["path"]).is_absolute(), "invalid owner code reference")
    sha(runtime["owner"]["sha256"])  # Historical code bytes are NOT opened.
    require(integer(runtime["inner_seconds"], "inner seconds") < integer(runtime["seconds"], "seconds") <= 7200,
            "invalid audit caps")
    require(type(runtime["cpu_affinity"]) is list and len(runtime["cpu_affinity"]) == 2
            and len(set(runtime["cpu_affinity"])) == 2, "invalid CPU affinity")
    for cpu in runtime["cpu_affinity"]:
        integer(cpu, "CPU", 0)
    command = owner["command"]
    suffix = ["-m", "scripts.core_generation_holdout_admit" if holdout else "scripts.core_generation_data_admit", "--protocol", root["audit_protocol"]["path"],
              "--protocol-sha256", root["audit_protocol"]["sha256"]]
    require(type(command) is list and all(type(arg) is str and arg for arg in command)
            and len(command) > len(suffix) and command[-len(suffix):] == suffix
            and command.count("-m") == command.count("--protocol") == command.count("--protocol-sha256") == 1
            and type(runtime["python"]) is str and command.count(runtime["python"]) == 1, "audit command differs")
    sha(owner["source_revision"], 40)
    require(Path(owner["cwd"]).is_absolute() and str(Path(owner["cwd"])) == owner["cwd"], "invalid owner cwd")
    require(terminal["status"] == "PASS" and type(terminal["returncode"]) is int and terminal["returncode"] == 0
            and terminal["owned_processes_closed"] is True and terminal["custody"] == "OWNED_PROCESSES_CLOSED"
            and terminal["resource"] == "cpu" and type(terminal["seconds"]) is int
            and terminal["seconds"] == runtime["seconds"] and terminal["supervisor_sha256"] == runtime["owner"]["sha256"]
            and terminal["command"] == command and terminal["source_revision"] == owner["source_revision"]
            and terminal["cwd"] == owner["cwd"], "audit owner not exact closed CPU PASS")
    wall = terminal["wall_seconds"]
    require(type(wall) in (int, float) and math.isfinite(wall) and 0 < wall <= runtime["seconds"], "invalid audit owner wall")
    births = terminal["known_processes"]
    require(type(births) is dict and births, "missing owned births")
    for pid, birth in births.items():
        require(type(pid) is str and pid.isdecimal() and str(int(pid)) == pid, "invalid owned PID")
        integer(int(cast(str, pid)), "owned PID")
        integer(birth, "owned birth")
    closure = owner["closure"]
    require(set(closure) == {"supported_eof", "exit_code", "supervisor_pid", "supervisor_closed", "owned_births_closed", "known_processes"}
            and closure["supported_eof"] is True and type(closure["exit_code"]) is int and closure["exit_code"] == 0
            and closure["supervisor_closed"] is True and closure["owned_births_closed"] is True
            and type(closure["known_processes"]) is dict and closure["known_processes"] == births
            and all(type(value) is int for value in closure["known_processes"].values()) and type(closure["supervisor_pid"]) is int
            and closure["supervisor_pid"] == integer(terminal["supervisor_pid"], "supervisor PID"), "root closure assertion differs")
    require(admission["status"] == (HOLDOUT_ADMISSION_STATUS if holdout else ADMISSION_STATUS) and admission["audit_protocol"] == root["audit_protocol"]
            and admission["audit_source"] == root["audit_source"] and admission["command"] == command
            and admission["owner"] == runtime["owner"] and admission["resource"] == "cpu"
            and type(admission["seconds"]) is int and admission["seconds"] == runtime["seconds"], "audit admission differs")


def _validate_generation(root_ref: dict[str, str], independent_ref: dict[str, str], *,
                         max_file_bytes: int, max_total_bytes: int, max_games: int, holdout: bool) -> AcceptedGeneration | AcceptedHoldout:
    """Validate trusted receipt pins and return a readonly candidate-bank view.

    Callers authenticate issuers/pins out of band. Bounds count unique JSON bytes,
    then final custody re-reads that same graph once. No current source/payload or
    future consumer identity is opened, inferred, or granted learning authority.
    """
    integer(max_games, "game limit")
    m = _Metadata(max_file_bytes, max_total_bytes)
    root, review = m.read(root_ref), m.read(independent_ref)
    require(set(root) == ({"status", "audit_protocol", "data_spec", "readout", "candidate", "audit_source", "audit_owner", "learning_admitted"} | ({"evaluation_only"} if holdout else set()))
            and root["status"] == (HOLDOUT_ROOT_STATUS if holdout else ROOT_STATUS) and root["learning_admitted"] is False, "invalid root acceptance")
    if holdout:
        require(root["evaluation_only"] is True, "HOLDOUT root purpose changed")
    require(set(review) == ({"status", "root_acceptance", "audit_protocol", "data_spec", "readout", "candidate", "findings"} | ({"evaluation_only"} if holdout else set()))
            and review["status"] == (HOLDOUT_REVIEW_STATUS if holdout else REVIEW_STATUS) and review["root_acceptance"] == root_ref and review["findings"] == []
            and all(review[key] == root[key] for key in ("audit_protocol", "data_spec", "readout", "candidate")), "independent acceptance differs")
    if holdout:
        require(review["evaluation_only"] is True, "HOLDOUT review purpose changed")
    p, spec = m.read(root["audit_protocol"]), m.read(root["data_spec"])
    require(set(p) == {"schema", "data_spec", "producer_protocol", "producer_readout", "producer_inventory", "producer_owner",
                      "gates", "runtime", "output_root", "limits"} and p["schema"] == ("CORE_FINITE_HOLDOUT_DATA_ADMISSION_V1" if holdout else "CORE_FINITE_DATA_ADMISSION_V1")
            and p["data_spec"] == root["data_spec"], "invalid data audit protocol")
    _spec(spec, holdout=holdout)
    source = root["audit_source"]
    require(type(source["checkout"]) is str and Path(source["checkout"]).is_absolute()
            and str(Path(source["checkout"])) == source["checkout"], "invalid historical source checkout")
    require(set(source) == {"revision", "checkout", "source_files"} and source["revision"] == spec["source_revision"]
            and source["source_files"] == spec["source_files"], "audit source differs")
    producer = m.read(p["producer_protocol"])
    require(producer["schema"] == ("CORE_FINITE_HOLDOUT_GENERATION_V1" if holdout else "CORE_FINITE_GENERATION_V1") and producer["source"]["revision"] == source["revision"]
            and producer["source"]["checkout"] == source["checkout"]
            and producer["actor"]["architecture"] == spec["architecture"] and producer["actor"]["sha256"] == spec["actor_sha256"]
            and type(producer["actor"]["expected_checkpoint_step"]) is int
            and producer["actor"]["expected_checkpoint_step"] == spec["actor_checkpoint_step"]
            and type(producer["generation"]) is int
            and all(producer[key] == spec[key] for key in ("run_uuid", "collection_uuid", "generation")), "producer identity differs")
    require(type(producer["attempts"]) is int, "invalid producer attempts")
    for name, digest in producer["source"]["sha256"].items():
        require(type(name) is str and name and not Path(name).is_absolute()
                and str(Path(name)) == name and ".." not in Path(name).parts, "invalid producer source path")
        sha(digest)
        require(spec["source_files"].get(name, digest) == digest, "producer/auditor source pin differs")
    require(producer["source"]["sha256"], "missing producer source pins")
    for key in ("producer_readout", "producer_inventory"):
        reference(p[key])  # Bind historical references without opening bulk input metadata.
    _owner(m, root, p, holdout=holdout)
    output = Path(p["output_root"])
    require(output.is_absolute() and output.resolve() == output, "noncanonical audit output")
    marker = output.with_name(output.name + ".failed.json")
    require(not marker.exists() and not marker.is_symlink(), "failed audit output")
    require(reference(root["readout"]) == output / "readout.json"
            and reference(root["candidate"]) == output / "generation.json", "audit output paths differ")
    readout, candidate = m.read(root["readout"]), m.read(root["candidate"])
    require(readout["status"] == ("AUDITED_CORE_HOLDOUT_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1" if holdout else "AUDITED_CORE_DATA_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1")
            and readout["generation_manifest"] == root["candidate"]
            and candidate["schema"] == ("CORE_HOLDOUT_GENERATION_DATA_CANDIDATE_V1" if holdout else "CORE_TRAIN_GENERATION_DATA_CANDIDATE_V1"), "candidate schema differs")
    for value in (readout, candidate):
        _candidate(value, spec, root["audit_protocol"], root["data_spec"], holdout=holdout)
    for key in ("run_uuid", "source_revision", "generation", "actor_sha256"):
        require(type(candidate[key]) is type(spec[key]) and candidate[key] == spec[key], "generation identity differs")
    serving = m.read(candidate["actor_serving"])
    served = serving["pre_state_sha256"]
    sha(served)
    require(served == serving["post_state_sha256"] == serving["checkpoint_model_state_sha256"]
            and serving["actor_sha256"] == spec["actor_sha256"] and type(serving["checkpoint_step"]) is int
            and serving["checkpoint_step"] == spec["actor_checkpoint_step"]
            and type(serving["logical_generation"]) is int and serving["logical_generation"] == spec["generation"]
            and type(serving["state_tensors"]) is int and serving["state_tensors"] == 494
            and type(serving["model_build_count"]) is int and serving["model_build_count"] == 1
            and integer(serving["forwards"]["calls"], "serving forwards") > 0, "serving evidence differs")
    require(reference(candidate["actor_serving"]) == Path(producer["output_root"]) / "actor-serving.json", "serving path differs")
    gates = p["gates"]
    require(set(gates) == {"attempts", "min_resolved_games", "min_total_rows"}, "invalid gates")
    for key, value in gates.items():
        integer(value, key)
    attempts = integer(readout["attempts"], "attempts")
    require(attempts == gates["attempts"] == spec["max_games_per_generation"] == producer["attempts"]
            and attempts <= max_games and gates["min_resolved_games"] == spec["games_per_generation"], "attempt/gate identity differs")
    audited, games = readout["audited"], candidate["games"]
    require(type(audited) is list and len(audited) == attempts and type(games) is list, "incomplete audit membership")
    eligible: list[dict[str, Any]] = []
    counts = dict.fromkeys(CLASSES, 0)
    for ordinal, game in enumerate(audited):
        require(game["physical_game_id"] == spec["collection_uuid"] + ":" + str(ordinal)
                and type(game["eligible"]) is bool and game["scientific_class"] in CLASSES, "attempt membership differs")
        sha(game["proof_sha256"])
        counts[game["scientific_class"]] += 1
        if game["eligible"]:
            integer(game["rows"], "eligible rows")
            eligible.append(game)
        else:
            require(type(game["rows"]) is int and game["rows"] == 0, "ineligible rows retained")
    require(len(games) == len(eligible) == readout["eligible_games"] and len(eligible) >= gates["min_resolved_games"]
            and type(readout["eligible_games"]) is int and readout["class_counts"] == counts
            and all(type(value) is int for value in readout["class_counts"].values()), "eligible/class counts differ")
    ids: set[int] = set()
    for game, checked in zip(games, eligible, strict=True):
        require(set(game) == {"physical_game_id", "game_id", "bank", "rows", "label_class", "shard", "producer_proof"}, "invalid game entry")
        require(type(game["game_id"]) is int and 0 <= game["game_id"] < 2**63
                and type(checked["game_id"]) is int, "invalid game ID")
        require(all(game[key] == checked[key] for key in ("physical_game_id", "game_id", "bank", "rows", "label_class", "shard"))
                and game["game_id"] not in ids
                and type(game["rows"]) is int and game["bank"] == ("holdout" if holdout else "train") and game["label_class"] ==
                {"NATURAL": "NATURAL", "STRICT_TB_ADJUDICATED": "TB_ADJUDICATED"}.get(checked["scientific_class"]), "game membership differs")
        ids.add(game["game_id"])
        wrapper = m.read(game["producer_proof"])
        require(reference(game["producer_proof"]) == output / ("game-" + game["physical_game_id"].rsplit(":", 1)[1] + ".json"), "wrapper path differs")
        require(wrapper["status"] == ("VERIFIED_RESOLVED_CORE_HOLDOUT_GAME_CANDIDATE_V1" if holdout else "VERIFIED_RESOLVED_CORE_DATA_GAME_CANDIDATE_V1")
                and wrapper["resolved"] is True and wrapper["full_history_verified"] is True
                and wrapper["raw_targets_verified"] is True, "wrapper verification differs")
        _candidate(wrapper, spec, root["audit_protocol"], root["data_spec"], holdout=holdout)
        for key in ("run_uuid", "generation", "source_revision", "actor_sha256", "served_state_sha256", "physical_game_id",
                    "game_id", "bank", "rows", "label_class", "evidence", "shard_inventory_sha256"):
            require(type(wrapper[key]) is type(checked[key]) and wrapper[key] == checked[key], "wrapper membership/provenance differs")
        require(all(checked[key] == spec[key] for key in ("run_uuid", "collection_uuid", "generation", "source_revision", "actor_sha256"))
                and type(checked["generation"]) is int and checked["served_state_sha256"] == served, "audited game identity differs")
        sha(checked["shard_inventory_sha256"])
        require(set(checked["evidence"]) == {"provenance", "validation", "outcome_proof", "raw_metadata"}, "invalid evidence references")
        for ref in checked["evidence"].values():
            reference(ref)  # Original game evidence is bound, not reopened.
        shard = game["shard"]
        require(type(shard) is dict and set(shard) == {"path", "files"}
                and type(shard["path"]) is str and Path(shard["path"]).is_absolute()
                and str(Path(shard["path"])) == shard["path"] and type(shard["files"]) is list
                and shard["files"], "invalid shard inventory descriptor")
        paths: list[str] = []
        for entry in shard["files"]:
            require(set(entry) == {"path", "bytes", "sha256"} and type(entry["path"]) is str
                    and not Path(entry["path"]).is_absolute() and str(Path(entry["path"])) == entry["path"]
                    and ".." not in Path(entry["path"]).parts, "invalid inventory entry")
            integer(entry["bytes"], "inventory bytes", 0)
            sha(entry["sha256"])
            paths.append(entry["path"])
        require(len(set(paths)) == len(paths) and ("." not in paths or paths == ["."]), "duplicate inventory path")
        inventory_digest = hashlib.sha256(json.dumps(shard["files"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        require(inventory_digest == checked["shard_inventory_sha256"], "inventory descriptor digest differs")
    rows = sum(game["rows"] for game in eligible)
    require(type(readout["eligible_rows"]) is int and readout["eligible_rows"] == rows >= gates["min_total_rows"], "row gate differs")
    caps = p["limits"]
    require(set(caps) == {"max_metadata_bytes", "max_inventory_files", "max_inventory_bytes", "max_raw_decoded_bytes_per_game",
                         "max_shard_decoded_bytes_per_game", "max_total_decoded_bytes"}, "invalid audit limits")
    for key, value in caps.items():
        integer(value, key)
    require(caps["max_metadata_bytes"] <= (128 << 20) and
            integer(readout["decoded_bytes"], "decoded bytes", 0) <= caps["max_total_decoded_bytes"], "decoded budget differs")
    require(all(type(readout[key]) is int and readout[key] == 0 for key in ("models_constructed", "forwards", "optimizer_updates")), "audit side effects differ")
    m.finish()
    require(not marker.exists() and not marker.is_symlink(), "failure marker appeared")
    identity = {key: spec[key] for key in ("run_uuid", "collection_uuid", "generation", "source_revision", "architecture",
                                         "actor_sha256", "actor_checkpoint_step")}
    evidence = {"root_acceptance": root_ref, "independent_review": independent_ref,
                "candidate": root["candidate"], "audit_source": source, "audit_owner": root["audit_owner"],
                "eligible_rows": rows, "class_counts": counts, "learning_admitted": False}
    if holdout:
        identity["bank"] = "holdout"
        evidence["evaluation_only"] = True
        return AcceptedHoldout(frozen(identity), tuple(frozen(game) for game in games), frozen(evidence))
    return AcceptedGeneration(frozen(identity), tuple(frozen(game) for game in games), frozen(evidence))


def validate_accepted_generation(root_ref: dict[str, str], independent_ref: dict[str, str], *,
                                 max_file_bytes: int, max_total_bytes: int, max_games: int) -> AcceptedGeneration:
    """Strict existing TRAIN route; HOLD roots/candidates/wrappers remain refused."""
    return cast(AcceptedGeneration, _validate_generation(root_ref, independent_ref,
        max_file_bytes=max_file_bytes, max_total_bytes=max_total_bytes, max_games=max_games, holdout=False))
