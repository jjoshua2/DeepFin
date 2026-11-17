"""Compile caller-trusted, closed self-play metadata into one fixed roster.

No shard/model/book/row data is opened. Legacy accepted-generation pins are
trusted inputs from the Root plan, not authority inferred from a status string.
Recent recipe-independent banks use the existing full metadata acceptance API.
The output is a manifest candidate requiring actual caller admission.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, cast

from chess_anti_engine.replay.fixed_population import (
    SCHEMA, canonical, digest, integer, inventory, origin_map, reference,
    require, sha, validate_manifest,
)

PLAN = "FIXED_SELFPLAY_UNION_PLAN_V1"
MAX_METADATA_FILE = 128 << 20
MAX_METADATA_TOTAL = 4 << 30


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value: str) -> Any:
    raise ValueError("nonfinite JSON: " + value)


def _float(value: str) -> float:
    number = float(value)
    require(math.isfinite(number), "nonfinite JSON float")
    return number


def _stamp(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)


class Metadata:
    """Reserve both first-read and final-recheck bytes before the first I/O."""

    def __init__(self, max_file_bytes: int, max_total_bytes: int):
        self.file_cap = integer(max_file_bytes, "metadata file cap")
        self.total_cap = integer(max_total_bytes, "metadata total cap")
        require(self.file_cap <= MAX_METADATA_FILE and self.total_cap <= MAX_METADATA_TOTAL,
                "metadata caps exceed finite source maximum")
        self.reserved = 0
        self.files: dict[str, tuple[str, tuple[int, ...], dict[str, Any]]] = {}

    def reserve(self, count: int) -> None:
        integer(count, "metadata reservation", 0)
        require(self.reserved + count <= self.total_cap, "metadata aggregate budget before read")
        self.reserved += count

    def read(self, ref: dict[str, str]) -> dict[str, Any]:
        reference(ref)
        path = Path(ref["path"])
        require(path.suffix == ".json" and path.resolve() == path, "noncanonical JSON metadata path")
        cached = self.files.get(str(path))
        if cached is not None:
            require(cached[0] == ref["sha256"], "conflicting metadata pin")
            return cached[2]
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_mode & 0o222 == 0
                and before.st_size <= self.file_cap, "metadata file cap/mode")
        self.reserve(2 * before.st_size)
        raw = self._read(path, _stamp(before))
        require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "metadata hash differs")
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite, parse_float=_float)
        require(type(value) is dict, "metadata object required")
        self.files[str(path)] = (ref["sha256"], _stamp(before), value)
        return value

    @staticmethod
    def _read(path: Path, expected: tuple[int, ...]) -> bytes:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            require(_stamp(os.fstat(fd)) == expected, "metadata fd changed before read")
            parts = []
            remaining = expected[2]
            while remaining:
                block = os.read(fd, min(1 << 20, remaining))
                require(bool(block), "short metadata read")
                parts.append(block)
                remaining -= len(block)
            require(_stamp(os.fstat(fd)) == _stamp(path.lstat()) == expected,
                    "metadata custody changed")
            return b"".join(parts)
        finally:
            os.close(fd)

    def finish(self) -> None:
        for path, (expected, before, _) in self.files.items():
            require(Path(path).resolve() == Path(path), "metadata path changed")
            raw = self._read(Path(path), before)
            require(hashlib.sha256(raw).hexdigest() == expected, "metadata final recheck differs")


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def validate_plan(p: dict[str, Any]) -> None:
    require(set(p) == {"schema", "architecture", "branches", "copies", "expected_sources",
                       "expected_rows", "limits"} and p["schema"] == PLAN, "invalid fixed-union plan")
    require(type(p["architecture"]) is dict and bool(p["architecture"]), "missing selected architecture")
    integer(p["expected_sources"], "expected source count")
    integer(p["expected_rows"], "expected row count")
    require(type(p["branches"]) is list and bool(p["branches"]), "missing pinned branches")
    require(type(p["copies"]) is list, "invalid copy declarations")
    for copy in cast(list[Any], p["copies"]):
        require(type(copy) is dict and set(copy) == {"canonical", "alias"}, "invalid copy proof")
        copy = cast(dict[str, Any], copy)
        reference(copy["canonical"])
        reference(copy["alias"])
    limits = p["limits"]
    require(type(limits) is dict and set(limits) == {
        "metadata_file_bytes", "metadata_total_bytes", "bank_metadata_bytes", "max_games_per_branch",
        "max_inventory_files", "max_inventory_bytes", "max_manifest_bytes"}, "invalid compilation limits")
    limits = cast(dict[str, Any], limits)
    for key, value in limits.items():
        integer(value, key)
    require(limits["metadata_file_bytes"] <= MAX_METADATA_FILE
            and limits["metadata_total_bytes"] <= MAX_METADATA_TOTAL
            and limits["max_manifest_bytes"] <= MAX_METADATA_FILE, "compilation cap too large")
    for branch in cast(list[Any], p["branches"]):
        require(type(branch) is dict and branch.get("kind") in {"legacy", "bank"}, "invalid source family")
        branch = cast(dict[str, Any], branch)
        if branch["kind"] == "legacy":
            require(set(branch) == {"kind", "manifest"}, "invalid trusted legacy binding")
            reference(branch["manifest"])
        else:
            require(set(branch) == {"kind", "root_acceptance", "independent_review"}, "invalid bank binding")
            reference(branch["root_acceptance"])
            reference(branch["independent_review"])


def _equal(actual: dict[str, Any], expected: dict[str, Any], message: str) -> None:
    require(all(type(actual.get(k)) is type(v) and actual.get(k) == v for k, v in expected.items()), message)


def _legacy(m: Metadata, branch: dict[str, Any], architecture: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The core.metadata producing-proof joins, without fictional FIFO identity.

    The exact legacy generation pin is an explicitly trusted Root-plan input.
    This does not replay collection, targets, boards, native code or inference.
    """
    g = m.read(branch["manifest"])
    require(g["schema"] == "ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1"
            and not g.get("evaluation_only", False), "not a caller-trusted legacy TRAIN generation")
    p = m.read(g["audit_protocol"])
    require(p["schema"] == "CORE_FINITE_ADMISSION_V1", "legacy admission family differs")
    producer = m.read(p["producer_protocol"])
    _equal(g, {"run_uuid": producer["run_uuid"], "generation": producer["generation"],
               "source_revision": producer["source"]["revision"], "actor_sha256": producer["actor"]["sha256"]},
           "legacy producer origin mismatch")
    require(producer["actor"]["architecture"] == architecture, "legacy producer architecture differs")
    served = m.read(g["actor_serving"])
    state = served["pre_state_sha256"]
    sha(state)
    require(served["actor_sha256"] == g["actor_sha256"] and
            state == served["post_state_sha256"] == served["checkpoint_model_state_sha256"],
            "legacy actor serving state differs")
    identity = {"run_uuid": g["run_uuid"], "collection_uuid": producer["collection_uuid"],
                "generation": g["generation"], "source_revision": g["source_revision"],
                "actor_sha256": g["actor_sha256"], "actor_checkpoint_step": producer["actor"]["expected_checkpoint_step"],
                "architecture": architecture}
    require(type(g["games"]) is list and bool(g["games"]), "legacy missing games")
    physical: set[str] = set()
    for game in cast(list[dict[str, Any]], g["games"]):
        require(game["bank"] == "train" and game["label_class"] in {"NATURAL", "TB_ADJUDICATED"},
                "protected or unresolved source refused")
        integer(game["rows"], "legacy source rows")
        integer(game["game_id"], "legacy game ID", 0)
        require(type(game["physical_game_id"]) is str and game["physical_game_id"].startswith(
            producer["collection_uuid"] + ":") and game["physical_game_id"] not in physical,
                "legacy physical game namespace differs")
        physical.add(game["physical_game_id"])
        proof = m.read(game["producer_proof"])
        expected = {"status": "VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1", "bank": "train",
                    "physical_game_id": game["physical_game_id"], "game_id": game["game_id"], "rows": game["rows"],
                    "source_revision": g["source_revision"], "actor_sha256": g["actor_sha256"],
                    "resolved": True, "full_history_verified": True, "raw_targets_verified": True,
                    "label_class": game["label_class"], "shard_inventory_sha256": inventory(game["shard"])}
        _equal(proof, expected, "legacy TRAIN proof predicates differ")
        require(proof["audit_protocol"] == g["audit_protocol"], "legacy proof admission differs")
        require({"outcome_proof", "validation", "provenance"} <= set(proof["evidence"]),
                "missing original producing evidence")
        for kind, original_ref in proof["evidence"].items():
            original = m.read(original_ref)
            common = {"source_revision": g["source_revision"], "run_uuid": g["run_uuid"],
                      "generation": g["generation"], "physical_game_id": game["physical_game_id"],
                      "game_id": game["game_id"], "rows": game["rows"],
                      "actor_sha256": g["actor_sha256"], "served_state_sha256": state}
            _equal(original, common, "original producing identity differs: " + kind)
            if kind == "outcome_proof":
                v = original["verification"]
                require(original["schema"] == "CORE_PRODUCING_OUTCOME_PROOF_V1"
                        and original["proof_sha256"] == digest(v), "original outcome proof digest differs")
                _equal(v, {k: x for k, x in common.items() if k not in {"rows", "generation", "run_uuid"}},
                       "original outcome identity differs")
                scientific = "STRICT_TB_ADJUDICATED" if game["label_class"] == "TB_ADJUDICATED" else "NATURAL"
                require(v["resolved"] is True and v["retained_rows"] == game["rows"]
                        and original["scientific_class"] == v["scientific_class"] == scientific,
                        "original outcome resolution differs")
    origin = {"generation_manifest": branch["manifest"], "identity": identity,
              "acceptance": branch, "producer_protocol": p["producer_protocol"],
              "actor_serving": g["actor_serving"], "served_state_sha256": state}
    return origin, cast(list[dict[str, Any]], g["games"])


def _bank(m: Metadata, branch: dict[str, Any], architecture: dict[str, Any], limits: dict[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from scripts import core_generation_bank_acceptance as admission

    # Existing API counts a bounded unique graph, then rereads it once. Reserve
    # BOTH passes globally before invoking it; no per-bank aggregate bypass.
    m.reserve(2 * limits["bank_metadata_bytes"])
    accepted = admission.validate_accepted_generation(branch["root_acceptance"], branch["independent_review"],
        max_file_bytes=limits["metadata_file_bytes"], max_total_bytes=limits["bank_metadata_bytes"],
        max_games=limits["max_games_per_branch"])
    identity, evidence, games = plain(accepted.identity), plain(accepted.evidence), plain(accepted.games)
    require(identity["architecture"] == architecture, "accepted bank architecture differs")
    root = m.read(branch["root_acceptance"])
    p = m.read(root["audit_protocol"])
    candidate = m.read(root["candidate"])
    serving = m.read(candidate["actor_serving"])
    origin = {"generation_manifest": root["candidate"], "identity": identity,
              "acceptance": branch, "producer_protocol": p["producer_protocol"],
              "actor_serving": candidate["actor_serving"], "served_state_sha256": serving["pre_state_sha256"]}
    require(sum(g["rows"] for g in games) == evidence["eligible_rows"], "accepted bank row count differs")
    return origin, games


def compile_union(plan_ref: dict[str, str], *, max_file_bytes: int = MAX_METADATA_FILE,
                  max_total_bytes: int = MAX_METADATA_TOTAL) -> dict[str, Any]:
    """Return a fixed-manifest candidate from the exact caller-trusted Root plan.

    Caller-selected caps bound even the first plan read. The plan may tighten,
    never increase them. Compilation does not grant learning authority.
    """
    m = Metadata(max_file_bytes, max_total_bytes)
    plan = m.read(plan_ref)
    validate_plan(plan)
    limits = plan["limits"]
    require(limits["metadata_file_bytes"] <= m.file_cap and limits["metadata_total_bytes"] <= m.total_cap,
            "plan increases caller metadata limits")
    m.file_cap, m.total_cap = limits["metadata_file_bytes"], limits["metadata_total_bytes"]
    require(m.reserved <= m.total_cap, "plan itself exceeds selected aggregate")
    origins: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    aliases = []
    by_sha: dict[str, dict[str, Any]] = {}
    origin_by_id: dict[str, dict[str, Any]] = {}
    inventory_bytes = inventory_files = 0
    total_rows = 0
    for branch in plan["branches"]:
        origin, games = (_legacy(m, branch, plan["architecture"]) if branch["kind"] == "legacy" else
                         _bank(m, branch, plan["architecture"], limits))
        require(len(games) <= limits["max_games_per_branch"], "branch game cap")
        origin["origin_id"] = digest(origin)
        if origin["origin_id"] not in origin_by_id:
            origins.append(origin)
            origin_by_id[origin["origin_id"]] = origin
        for game in games:
            source_sha = inventory(game["shard"])
            entry: dict[str, Any] = {"slot": len(sources), "generation": origin["identity"]["generation"],
                     "source_sha256": source_sha, "origin_id": origin["origin_id"],
                     **{k: game[k] for k in ("rows", "game_id", "physical_game_id", "bank", "label_class", "shard", "producer_proof")}}
            require(entry["bank"] == "train", "HOLD/final source cannot enter union")
            if source_sha in by_sha:
                old = by_sha[source_sha]
                previous = origin_by_id[old["origin_id"]]
                same_manifest = previous["generation_manifest"] == origin["generation_manifest"]
                declared = {"canonical": previous["generation_manifest"], "alias": origin["generation_manifest"]}
                require(same_manifest or declared in plan["copies"], "unproven copied-source alias")
                require(all(old[k] == entry[k] for k in ("rows", "game_id", "physical_game_id", "bank", "label_class"))
                        and old["shard"]["files"] == entry["shard"]["files"]
                        and previous["identity"] == origin["identity"]
                        and previous["served_state_sha256"] == origin["served_state_sha256"], "copied-source provenance differs")
                aliases.append({"canonical_slot": old["slot"], "origin_id": origin["origin_id"],
                                "shard": entry["shard"], "producer_proof": entry["producer_proof"]})
                continue
            by_sha[source_sha] = entry
            sources.append(entry)
            total_rows += entry["rows"]
            inventory_files += len(entry["shard"]["files"])
            inventory_bytes += sum(f["bytes"] for f in entry["shard"]["files"])
            require(inventory_files <= limits["max_inventory_files"] and inventory_bytes <= limits["max_inventory_bytes"],
                    "union inventory aggregate cap")
            require(len(sources) <= plan["expected_sources"]
                    and total_rows <= plan["expected_rows"], "union roster exceeds fixed totals")
    result = {"schema": SCHEMA, "plan": plan_ref, "architecture": plan["architecture"],
              "expected_sources": plan["expected_sources"], "expected_rows": plan["expected_rows"],
              "sources": sources, "origins": origins, "aliases": aliases,
              "origin_map_sha256": "", "training_authorized": False}
    result["origin_map_sha256"] = digest(origin_map(result))
    validate_manifest(result)
    require(len((canonical(result) + "\n").encode()) <= limits["max_manifest_bytes"], "compiled manifest byte cap")
    m.finish()
    return result


def read_union_manifest(manifest_ref: dict[str, str], expected_plan_ref: dict[str, str], *,
                        max_file_bytes: int, max_total_bytes: int) -> dict[str, Any]:
    """Read trusted compiler bytes and their exact Root-plan identity, no payload."""
    m = Metadata(max_file_bytes, max_total_bytes)
    plan = m.read(expected_plan_ref)
    validate_plan(plan)
    limits = plan["limits"]
    m.total_cap = min(m.total_cap, limits["metadata_total_bytes"])
    m.file_cap = min(m.file_cap, limits["max_manifest_bytes"])
    require(m.reserved <= m.total_cap, "plan exceeds selected aggregate")
    manifest = m.read(manifest_ref)
    validate_manifest(manifest)
    require(manifest["plan"] == expected_plan_ref and manifest["architecture"] == plan["architecture"]
            and all(manifest[k] == plan[k] for k in ("expected_sources", "expected_rows")), "union plan binding differs")
    require(hashlib.sha256((canonical(manifest) + "\n").encode()).hexdigest() == manifest_ref["sha256"],
            "noncanonical compiled manifest bytes")
    require({canonical(o["acceptance"]) for o in manifest["origins"]}
            == {canonical(branch) for branch in plan["branches"]}, "missing or unselected union origin")
    file_count = sum(len(s["shard"]["files"]) for s in manifest["sources"])
    inventory_bytes = sum(f["bytes"] for s in manifest["sources"] for f in s["shard"]["files"])
    require(file_count <= limits["max_inventory_files"] and inventory_bytes <= limits["max_inventory_bytes"],
            "compiled inventory aggregate differs")
    m.finish()
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-file-bytes", required=True, type=int)
    parser.add_argument("--metadata-total-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    value = compile_union({"path": args.plan, "sha256": args.plan_sha256},
        max_file_bytes=args.metadata_file_bytes, max_total_bytes=args.metadata_total_bytes)
    output = Path(args.out)
    require(output.is_absolute() and output.resolve() == output and output.suffix == ".json",
            "canonical output required")
    raw = (canonical(value) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    # A partial output never becomes accepted; no overwrite or automatic retry.
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
