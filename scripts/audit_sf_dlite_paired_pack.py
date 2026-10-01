"""Checkpointed, independent qualification of one native paired training pack.

This is a source/target/transport audit against already qualified D8 labels.
It does not repeat Stockfish searches or turn a builder receipt into admission.
Every large unit is an owned child capped at 30 minutes. Do not run it beside
another heavy-I/O owner; the process takes the shared physical-I/O lease.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

if sys.flags.optimize:
    raise RuntimeError("HOLD: Python -O removes scientific assertions")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts import sf_dlite_pack_audit_core as audit_core
from scripts.sf_dlite_pack_audit_core import (
    audit_shard, require, sha256_file, tree_stamp)


ROWS = 2_500_000
SHARDS = 384
UNIT_SECONDS = 1800
RSS_KIB_MAX = 6 << 20
AVAILABLE_KIB_MIN = 32 << 20
PHYSICAL_BYTES_MAX = 400 << 30
PHYSICAL_SCOPE = (
    "shared cgroup r+w; new device charged from zero; cached reads omitted; "
    "not process logical bytes"
)
OUTPUT_BYTES_MAX = 1 << 30
FREE_BYTES_MIN = 50 << 30
ARTIFACT_ROOT = Path("/home/josh/chess-artifacts")
LEASE = Path("/tmp/chess-physical-heavy-io-exclusive.lock")
LABEL_MONITOR_SOURCE = Path(
    "/tmp/sf_dlite_d8_full07_label_audit_session_supervisor_candidate_20260930.py")
LABEL_FREEZER_SOURCE = Path(
    "/tmp/sf_dlite_d8_full07_audit_freeze_only_supervisor_candidate_20260930.py")
LABEL_AUDITOR_SOURCE = Path(
    "/tmp/sf_dlite_d8_full_label_independent_audit_candidate_v4_20260930.py")
LABEL_AUDIT_ROOT = ARTIFACT_ROOT / "operations/sf-dlite-full07-label-independent-audit-20260930"
LABEL_FREEZE_ROOT = ARTIFACT_ROOT / "operations/sf-dlite-full07-label-independent-freeze01-20260930"
QUALIFICATION_SCHEMA = "sf_dlite_selected_e_paired_native_pack_qualification_v1"
QUALIFICATION_STATUS = "PASS_INDEPENDENT_PAIRED_PACK_QUALIFICATION"
EXPECTED_PINS = {
    "census": "f7880a99dc09a43519aeea884beccc22cc2c0c373cf2574df6e5dd21e611d85b",
    "roster_audit": "159648aa60fa3cdc4f11e79c0ff629a47d2a538316cc8011bee5aa1030865d6a",
    "selected_e_qualification": "2f985f12410e82d6ecc5f4d18012d0f3d030fad475d1d6be8d1c54c13f40222d",
    "injectivity": "2b10f425d98d5f788ec8b06f359094c8cc655749e24d4ef641b9f2b318422b88",
    "roster": "8766e5745a6f36e4ecbed90cc861d012e29f2d21356958fa1477bead4dcc7941",
    "target_scheme": "716ef4da2c928a437ca001f7ea769c8e555091221cb22ca7ee884e5deeaefdfa",
    "builder_source": "79f916c3e0f957c20d79b431192810363890f8094eed291889518f15c88d5bbf",
    "builder_core": "50af80882449d3fc9626ed91fa9a9e99587cc9f7cc1fcb07046b1bdb35adfaf5",
    "label_worker": "31a7730cac1f9b6b39f01b938ccf6772c895d5a2727f89eb5a969aacec24cf83",
    "label_authorization": "b4806340c7f3a5a61c8eedd0ea4156f99928b494274780ae7877fb072c484dc1",
    "label_auditor": "e80ce6c8bbb7585b4a46cfd8720fac21f248df8658a0d2d2045104e90afdceb2",
    "label_monitor": "2bed0f2234bad765359e613bf5f009aaf3a54f7169d77af1705285f0e8be308c",
    "label_freezer": "4ce5bb05137a05dfd116b650cc6a088e37c6d565fa940f94f71af36d9639076a",
}


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path: Path, value: dict[str, Any]) -> str:
    """No-replace, crash-retryable publication; a leftover temp has no credit."""
    require(path.parent.is_dir() and not path.parent.is_symlink(), "unsafe receipt parent")
    require(not os.path.lexists(path), "receipt already exists")
    raw = canonical(value)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}.{secrets.token_hex(8)}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    sync_directory(path.parent)
    temporary.unlink()
    sync_directory(path.parent)
    require(path.read_bytes() == raw, "receipt readback differs")
    return digest(raw)


def read_receipt(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file() and not path.is_symlink(), f"receipt missing: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict) and canonical(value) == raw,
            f"receipt canonical bytes differ: {path}")
    return value, digest(raw)


def pinned(path: Path, expected_sha: str) -> dict[str, Any]:
    require(len(expected_sha) == 64 and expected_sha == expected_sha.lower(),
            "invalid expected SHA-256")
    require(path.is_file() and not path.is_symlink(),
            f"pinned input missing or linked: {path}")
    raw = path.read_bytes()
    require(digest(raw) == expected_sha, f"pinned input changed: {path}")
    value = json.loads(raw)
    require(isinstance(value, dict), "pinned input is not an object")
    return value


def source_provenance() -> dict[str, str]:
    core = Path(__file__).with_name("sf_dlite_pack_audit_core.py").resolve(strict=True)
    require(Path(audit_core.__file__).resolve(strict=True) == core
            and audit_core.audit_shard is audit_shard
            and audit_core.tree_stamp is tree_stamp,
            "imported independent auditor core differs from exact file")
    return {"auditor_source_sha256": sha256_file(Path(__file__)),
            "auditor_core_sha256": sha256_file(core)}


def runtime_provenance() -> dict[str, Any]:
    modules = {"numpy": np, "zarr": audit_core.zarr}
    result: dict[str, Any] = {"python_version": sys.version.split()[0]}
    for name, module in modules.items():
        origin = module.__file__
        if origin is None:
            raise ValueError(f"HOLD: {name} runtime module has no file origin")
        path = Path(origin).resolve(strict=True)
        result[name] = {"version": module.__version__, "origin": str(path),
                        "init_sha256": sha256_file(path)}
    return result


def read_plan(path: Path, expected_sha: str) -> dict[str, Any]:
    plan = pinned(path, expected_sha)
    require(plan.get("schema") == "sf_dlite_independent_paired_pack_audit_plan_v1"
            and plan.get("status") == "REVIEWED_ZERO_CREDIT_PLAN",
            "unreviewed paired pack audit plan")
    for key in ("census", "builder_terminal", "label_terminal", "label_audit",
                "label_audit_session",
                "roster_audit", "selected_e_qualification", "injectivity",
                "target_scheme"):
        ref = plan.get(key)
        require(isinstance(ref, dict) and isinstance(ref.get("path"), str)
                and Path(ref["path"]).is_absolute()
                and isinstance(ref.get("sha256"), str), f"missing {key} pin")
    require(plan.get("auditor_sources") == source_provenance(),
            "independent auditor source bytes differ from plan")
    require(plan.get("auditor_runtime") == runtime_provenance(),
            "independent auditor runtime differs from plan")
    qualification_path = plan.get("qualification_path")
    require(isinstance(qualification_path, str)
            and Path(qualification_path).is_absolute(),
            "trainer-facing qualification path missing")
    return plan


def monitored_label_completion(plan: dict[str, Any],
                               independent: dict[str, Any]) -> dict[str, Any]:
    """Independently require the full07 parent's durable post-child acceptance."""
    path = Path(plan["label_audit_session"]["path"])
    session_root = path.parent
    terminal_path = Path(plan["label_audit"]["path"])
    require(path.name == "COMPLETE.json"
            and session_root.is_dir() and not session_root.is_symlink()
            and session_root.parent == ARTIFACT_ROOT / "operations"
            and session_root.name.startswith(
                "sf-dlite-full07-label-independent-session-")
            and terminal_path == LABEL_AUDIT_ROOT / "TERMINAL.json"
            and not any(os.path.lexists(item / "FAILED.json") for item in
                        (session_root, LABEL_AUDIT_ROOT, LABEL_FREEZE_ROOT)),
            "independent label audit monitor session missing, failed or linked")
    complete = pinned(path, plan["label_audit_session"]["sha256"])
    finish_path = session_root / "FINISH.json"
    finish = pinned(finish_path, complete["finish_monitor_sha256"])
    freeze_path = LABEL_FREEZE_ROOT / "COMPLETE.json"
    freeze = pinned(freeze_path, complete["freeze_monitor_complete_sha256"])
    plan_sha = independent["plan_sha256"]
    terminal_sha = plan["label_audit"]["sha256"]
    require(sha256_file(LABEL_MONITOR_SOURCE) == EXPECTED_PINS["label_monitor"]
            and sha256_file(LABEL_FREEZER_SOURCE) == EXPECTED_PINS["label_freezer"]
            and sha256_file(LABEL_AUDITOR_SOURCE) == EXPECTED_PINS["label_auditor"]
            and sha256_file(terminal_path) == terminal_sha
            and sha256_file(terminal_path.parent / "PLAN.json") == plan_sha,
            "monitored audit source, terminal or frozen plan bytes differ")
    step = finish.get("finish_step")
    require(isinstance(step, dict) and step.get("exit_code") == 0
            and type(step.get("elapsed_seconds")) in (int, float)
            and 0 <= step["elapsed_seconds"] < 2100
            and sha256_file(session_root / "finish.log") == step.get("log_sha256")
            and sha256_file(session_root / "finish.trace.jsonl") ==
            step.get("trace_sha256"),
            "monitored audit finish child trace differs")
    require(complete.get("schema") ==
            "sf_dlite_d8_full07_independent_audit_session_v2"
            and complete.get("status") == "COMPLETE_ALL_LABEL_BLOCKS_ZERO_ADMISSION"
            and complete.get("supervisor_sha256") == EXPECTED_PINS["label_monitor"]
            and complete.get("auditor_sha256") == EXPECTED_PINS["label_auditor"]
            and complete.get("freezer_sha256") == EXPECTED_PINS["label_freezer"]
            and complete.get("freeze_monitor_complete_sha256") ==
            sha256_file(freeze_path)
            and complete.get("plan_path") == str(terminal_path.parent / "PLAN.json")
            and complete.get("plan_sha256") == plan_sha
            and complete.get("terminal_sha256") == terminal_sha
            and complete.get("label_launch_terminal_sha256") ==
            plan["label_terminal"]["sha256"]
            and complete.get("blocks") == independent["blocks"]
            and complete.get("rows") == ROWS
            and complete.get("completed_batches") ==
            (independent["blocks"] + 31) // 32
            and complete.get("credit") == {"target": 0, "training": 0, "elo": 0}
            and freeze.get("schema") ==
            "sf_dlite_d8_full07_independent_freeze_session_v1"
            and freeze.get("status") == "COMPLETE_PLAN_ONLY_NO_AUDIT_CREDIT"
            and freeze.get("supervisor_sha256") == EXPECTED_PINS["label_freezer"]
            and freeze.get("auditor_sha256") == EXPECTED_PINS["label_auditor"]
            and freeze.get("plan_sha256") == plan_sha
            and freeze.get("audit_root") == str(LABEL_AUDIT_ROOT)
            and freeze.get("credit") ==
            {"labels": 0, "targets": 0, "training": 0, "elo": 0}
            and finish.get("schema") ==
            "sf_dlite_d8_full07_independent_audit_finish_monitor_v1"
            and finish.get("status") == "PASS_MONITORED_FINISH_ZERO_ADMISSION"
            and finish.get("supervisor_sha256") == EXPECTED_PINS["label_monitor"]
            and finish.get("auditor_sha256") == EXPECTED_PINS["label_auditor"]
            and finish.get("plan_sha256") == plan_sha
            and finish.get("terminal_sha256") == terminal_sha
            and finish.get("credit") == {"target": 0, "training": 0, "elo": 0}
            and finish.get("finish_step") == step,
            "all-label child TERMINAL lacks exact monitored full07 completion")
    return complete


def sources(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = {key: pinned(Path(plan[key]["path"]), plan[key]["sha256"])
              for key in ("census", "builder_terminal", "label_terminal", "label_audit",
                          "label_audit_session",
                          "roster_audit", "selected_e_qualification", "injectivity",
                          "target_scheme")}
    census = source["census"]
    builder = source["builder_terminal"]
    label = source["label_terminal"]
    independent = source["label_audit"]
    monitored_label_completion(plan, independent)
    expected_source = builder.get("source", {})
    require(expected_source.get("independent_label_audit_session_path") ==
            plan["label_audit_session"]["path"]
            and expected_source.get("independent_label_audit_session_sha256") ==
            plan["label_audit_session"]["sha256"],
            "builder source differs from monitored label-audit completion")
    for key in ("census", "roster_audit", "selected_e_qualification", "injectivity"):
        require(plan[key]["sha256"] == EXPECTED_PINS[key],
                f"{key} differs from frozen Selected-E source")
    require(builder.get("builder_source_sha256") == EXPECTED_PINS["builder_source"]
            and builder.get("core_source_sha256") == EXPECTED_PINS["builder_core"]
            and expected_source.get("roster_sha256") == EXPECTED_PINS["roster"]
            and builder.get("target_scheme", {}).get("sha256") ==
            EXPECTED_PINS["target_scheme"],
            "frozen builder, roster, core or target scheme differs")
    require(census.get("status") ==
            "COMPLETED_UNADMITTED_PENDING_INDEPENDENT_ALL_ROW_AUDIT"
            and census["roster"]["sha256"] == expected_source.get("roster_sha256")
            and census["dedup"]["selected_rows"] == ROWS
            and len(census["shards"]) == SHARDS,
            "census roster or shard contract differs")
    require(source["roster_audit"].get("status") ==
            "PASS_SELECTED_ROSTER_SOURCE_TARGET_AND_GROSS_SELECTION_ZERO_ADMISSION"
            and source["roster_audit"].get("roster_sha256") ==
            expected_source["roster_sha256"], "independent roster audit differs")
    require(source["injectivity"].get("status") == "INJECTIVE_RAW_GAME_ID_OK"
            and source["injectivity"].get("source_qualified_games") == 16412
            and source["injectivity"].get("colliding_original_game_ids") == 0,
            "original game ID injectivity proof differs")
    require(builder.get("schema") == "sf_dlite_selected_e_paired_native_pack_builder_terminal_v1"
            and builder.get("status") == "COMPLETE_UNADMITTED_PENDING_INDEPENDENT_PACK_AUDIT"
            and builder.get("training_route") == {
                "sampling_mode": "game_epoch", "rows": ROWS, "batches": 4883,
                "batch_size": 512, "seed": 121, "shards": SHARDS,
                "game_id_mode": "original_i64_injective"},
            "builder terminal or training route differs")
    for key, field in (("census", "census_terminal_sha256"),
                       ("roster_audit", "independent_roster_audit_sha256"),
                       ("selected_e_qualification", "selected_e_qualification_sha256"),
                       ("injectivity", "game_id_injectivity_sha256"),
                       ("label_terminal", "label_terminal_sha256"),
                       ("label_audit", "independent_label_audit_sha256"),
                       ("label_audit_session", "independent_label_audit_session_sha256")):
        require(plan[key]["sha256"] == expected_source.get(field),
                f"builder source {field} differs from plan")
    require(label.get("status") ==
            "COMPLETE_UNADMITTED_PENDING_INDEPENDENT_LABEL_AUDIT"
            and label.get("workers") == 6 and label.get("worker_exit_codes") == [0] * 6
            and label.get("selection_sha256") == expected_source["roster_sha256"]
            and label.get("operator_sha256") == EXPECTED_PINS["label_worker"]
            and label.get("authorization_sha256") ==
            EXPECTED_PINS["label_authorization"],
            "full D8 label terminal differs")
    require(independent.get("status") ==
            "PASS_ALL_2500000_LABELS_SOURCE_ROSTER_RAW_D8_ZERO_ADMISSION"
            and independent.get("rows") == ROWS
            and independent.get("operator_sha256") == EXPECTED_PINS["label_auditor"]
            and independent.get("roster_sha256") == EXPECTED_PINS["roster"]
            and independent.get("launch_terminal_sha256") ==
            plan["label_terminal"]["sha256"],
            "independent all-label audit differs")
    require(builder.get("target_scheme", {}).get("sha256") ==
            source["target_scheme"].get("sha256"), "target scheme differs")
    require(set(builder.get("arms", {})) == {"control", "candidate"},
            "builder has no exact pair")
    for arm in ("control", "candidate"):
        entry = builder["arms"][arm]
        require(entry["rows"] == ROWS and len(entry["shards"]) == SHARDS
                and entry["kind"] == ("selected_e_control" if arm == "control"
                                      else "sf_dlite_d8_value_only"),
                "builder arm layout differs")
    control_root = Path(builder["arms"]["control"]["root"])
    candidate_root = Path(builder["arms"]["candidate"]["root"])
    pack_root = control_root.parent
    require(candidate_root.parent == pack_root
            and Path(plan["builder_terminal"]["path"]) ==
            pack_root / "BUILDER-TERMINAL.json"
            and Path(plan["qualification_path"]) ==
            pack_root / "PAIRED_PACK_QUALIFICATION.json"
            and pack_root.is_dir() and not pack_root.is_symlink()
            and pack_root == pack_root.resolve(strict=True)
            and pack_root.is_relative_to(ARTIFACT_ROOT.resolve(strict=True)),
            "qualification path or paired pack root differs")
    return census, builder, label


def roster_array(census: dict[str, Any], *, hash_bytes: bool) -> np.memmap:
    path = Path(census["roster"]["path"])
    require(path.is_file() and not path.is_symlink() and
            path.stat().st_size == ROWS * 180, "roster size differs")
    if hash_bytes:
        require(sha256_file(path) == census["roster"]["sha256"],
                "roster bytes changed")
    dtype = np.dtype([tuple(item) for item in census["row_dtype"]], align=False)
    require(dtype.itemsize == 180, "roster dtype differs")
    return np.memmap(path, dtype=dtype, mode="r", shape=(ROWS,))


def qualified_label_blocks(plan: dict[str, Any], label: dict[str, Any]
                           ) -> list[dict[str, Any]]:
    """Use precisely the blocks admitted by the independent raw-label audit."""
    terminal_path = Path(plan["label_audit"]["path"])
    independent = pinned(terminal_path, plan["label_audit"]["sha256"])
    frozen = pinned(terminal_path.parent / "PLAN.json", independent["plan_sha256"])
    require(frozen.get("schema") == "sf_dlite_d8_full_label_independent_plan_v1"
            and frozen.get("status") == "FROZEN_COMPLETED_LABELS_ZERO_ADMISSION"
            and frozen.get("operator_sha256") == EXPECTED_PINS["label_auditor"]
            and frozen.get("launch_terminal_sha256") ==
            plan["label_terminal"]["sha256"]
            and frozen.get("selected_roster_sha256") == EXPECTED_PINS["roster"]
            and frozen.get("label_root") == label["output_root"]
            and frozen.get("rows") == ROWS
            and isinstance(frozen.get("blocks"), list)
            and len(frozen["blocks"]) == independent["blocks"]
            and sum(block["rows"] for block in frozen["blocks"]) == ROWS,
            "frozen independent label block plan differs")
    return frozen["blocks"]


def label_index(root: Path, plan: dict[str, Any], census: dict[str, Any],
                label: dict[str, Any]) -> dict[str, Any]:
    qualified_blocks = qualified_label_blocks(plan, label)
    qualified_blocks.sort(key=lambda block: block["receipt_path"])
    block_by_path = {block["receipt_path"]: block for block in qualified_blocks}
    require(len(block_by_path) == len(qualified_blocks),
            "duplicate qualified label receipt path")
    final = root / "LABEL-INDEX.json"
    receipt: dict[str, Any] | None = None
    if final.exists():
        receipt, _ = read_receipt(final)
        require(receipt["label_terminal_sha256"] == plan["label_terminal"]["sha256"]
                and receipt["label_audit_sha256"] == plan["label_audit"]["sha256"]
                and receipt["roster_sha256"] == census["roster"]["sha256"]
                and receipt["plan_sha256"] == plan["self_sha256"]
                and receipt["auditor_runtime"] == runtime_provenance()
                and all(receipt[key] == value for key, value in
                        source_provenance().items()),
                "staged label source differs")
        for key in ("table", "bitmap"):
            item = receipt[key]
            path = Path(item["path"])
            require(path.is_relative_to(root / "label_stage")
                    and sha256_file(path) == item["sha256"],
                    "staged label file changed")
        require(len(receipt["block_receipts"]) == receipt["blocks"],
                "staged label block receipt count differs")
        require(receipt["block_receipts"] ==
                [{"path": block["receipt_path"], "sha256": block["receipt_sha256"]}
                 for block in qualified_blocks],
                "staged label blocks differ from independent qualification")
        for ref in receipt["block_receipts"]:
            block = pinned(Path(ref["path"]), ref["sha256"])
            data = Path(ref["path"]).parent / block["data_file"]
            require(data.stat().st_size == block["data_bytes"]
                    and sha256_file(data) == block["data_sha256"],
                    "staged label source block changed")
        if require_resource(root, plan["self_sha256"], "label-index"):
            return receipt
        # A proof published before monitor acceptance is not a completed unit.
        # Reconstruct it from qualified bytes before the parent can credit it.
    roster = roster_array(census, hash_bytes=True)
    stage_root = root / "label_stage"
    attempt = stage_root / f"attempt-{os.getpid()}-{secrets.token_hex(8)}"
    attempt.mkdir()
    sync_directory(stage_root)
    table_path = attempt / "labels.f4"
    bitmap_path = attempt / "seen.u1"
    table = np.memmap(table_path, dtype="<f4", mode="w+", shape=(ROWS, 3))
    bitmap = np.memmap(bitmap_path, dtype="u1", mode="w+", shape=(ROWS,))
    bitmap[:] = 0
    label_root = Path(label["output_root"])
    count = 0
    blocks = 0
    block_receipts: list[dict[str, str]] = []
    for worker in range(6):
        folder = label_root / f"worker{worker:02d}"
        complete = json.loads((folder / "COMPLETE.json").read_bytes())
        require(complete["status"] == "COMPLETE_WORKER_UNADMITTED"
                and complete["identity"]["operator_sha256"] ==
                EXPECTED_PINS["label_worker"],
                "worker label completion differs")
        for receipt_path in sorted(folder.glob("s*-b*.receipt.json")):
            expected = block_by_path.get(str(receipt_path))
            if expected is None:
                raise ValueError("HOLD: unqualified label block")
            block = pinned(receipt_path, expected["receipt_sha256"])
            block_receipts.append({"path": str(receipt_path),
                                   "sha256": expected["receipt_sha256"]})
            require(str(folder / block["data_file"]) == expected["data_path"]
                    and block["data_sha256"] == expected["data_sha256"]
                    and block["data_bytes"] == expected["data_bytes"]
                    and block["rows"] == expected["rows"]
                    and block["source_id"] == expected["source_id"]
                    and block["identity"]["worker_id"] == expected["worker_id"],
                    "label block differs from independently qualified block")
            require(block["schema"] == "sf_dlite_legacy_d8_checkpointed_label_v1"
                    and block["identity"]["worker_id"] == worker
                    and block["identity"]["roster_sha256"] ==
                    census["roster"]["sha256"]
                    and block["identity"]["authorization_sha256"] ==
                    EXPECTED_PINS["label_authorization"],
                    "label block identity differs")
            data = folder / block["data_file"]
            require(data.is_file() and not data.is_symlink()
                    and data.stat().st_size == block["data_bytes"],
                    "label block data changed")
            indices: list[int] = []
            consumed = hashlib.sha256()
            consumed_bytes = 0
            with data.open("rb") as stream:
                for line in stream:
                    consumed.update(line)
                    consumed_bytes += len(line)
                    require(consumed_bytes <= block["data_bytes"],
                            "label block data grew during read")
                    row = json.loads(line)
                    index = row["roster_index"]
                    require(type(index) is int and 0 <= index < ROWS and bitmap[index] == 0,
                            "duplicate or missing roster index")
                    source_id = int(roster["source_id"][index])
                    source = census["sources"][source_id]
                    require(row["source_id"] == source_id
                            and row["source_row"] == int(roster["source_row"][index])
                            and row["input_sha256"] ==
                            roster["input_sha256"][index:index + 1].tobytes().hex()
                            and row["context_sha256"] ==
                            roster["context_sha256"][index:index + 1].tobytes().hex()
                            and row["stored_key_hex"] ==
                            roster["stored_key"][index:index + 1].tobytes().hex()
                            and block["source_id"] == source_id
                            and block["source_path"] == source["path"]
                            and block["source_sha256"] == source["sha256"]
                            and row["depth_requested"] == 8
                            and row["wdl_orientation"] == "side_to_move",
                            "label/source-qualified roster join differs")
                    value = np.asarray(row["score"]["d_style_wdl"], dtype="<f4")
                    require(value.shape == (3,) and np.all(np.isfinite(value))
                            and np.all((value >= 0) & (value <= 1))
                            and abs(float(value.sum()) - 1) <= 1e-5,
                            "D8 label WDL invalid")
                    table[index] = value
                    bitmap[index] = 1
                    indices.append(index)
                    count += 1
            require(consumed_bytes == block["data_bytes"]
                    and consumed.hexdigest() == block["data_sha256"],
                    "consumed label block bytes differ from qualified bytes")
            require(len(indices) == block["rows"] and
                    digest(np.asarray(indices, dtype="<i4").tobytes()) ==
                    block["roster_indices_sha256"],
                    "label block roster-index order differs")
            blocks += 1
    require(block_receipts ==
            [{"path": block["receipt_path"], "sha256": block["receipt_sha256"]}
             for block in qualified_blocks],
            "label block set/order differs from independent qualification")
    require(count == ROWS and int(np.count_nonzero(bitmap)) == ROWS,
            "D8 label index incomplete")
    table.flush()
    bitmap.flush()
    del table, bitmap
    for path in (table_path, bitmap_path):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    sync_directory(attempt)
    result: dict[str, Any] = {"schema": "sf_dlite_independent_label_index_v1", "rows": ROWS,
              "blocks": blocks,
              "block_receipts": block_receipts,
              "plan_sha256": plan["self_sha256"],
              **source_provenance(),
              "auditor_runtime": runtime_provenance(),
              "label_terminal_sha256": plan["label_terminal"]["sha256"],
              "label_audit_sha256": plan["label_audit"]["sha256"],
              "roster_sha256": census["roster"]["sha256"],
              "table": {"path": str(table_path), "sha256": sha256_file(table_path)},
              "bitmap": {"path": str(bitmap_path), "sha256": sha256_file(bitmap_path)}}
    if receipt is not None:
        require({key: value for key, value in result.items()
                 if key not in ("table", "bitmap")} ==
                {key: value for key, value in receipt.items()
                 if key not in ("table", "bitmap")}
                and all(result[key]["sha256"] == receipt[key]["sha256"]
                        for key in ("table", "bitmap")),
                "orphan label index differs from independent reconstruction")
        shutil.rmtree(attempt)
        sync_directory(stage_root)
        return receipt
    publish(final, result)
    return result


def physical_counters() -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for line in Path("/sys/fs/cgroup/io.stat").read_text().splitlines():
        device, *fields = line.split()
        values = dict(item.split("=", 1) for item in fields)
        require(device not in result and "rbytes" in values and
                "wbytes" in values, "shared physical-I/O counter invalid")
        read, write = int(values["rbytes"]), int(values["wbytes"])
        require(read >= 0 and write >= 0, "negative shared physical-I/O count")
        result[device] = read, write
    require(bool(result), "shared physical-I/O counters missing")
    return result


def physical_map(counters: dict[str, tuple[int, int]]) -> dict[str, dict[str, int]]:
    return {device: {"rbytes": pair[0], "wbytes": pair[1]}
            for device, pair in sorted(counters.items())}


def parsed_physical_map(value: Any) -> dict[str, tuple[int, int]]:
    require(isinstance(value, dict) and bool(value),
            "physical-I/O device map missing")
    result: dict[str, tuple[int, int]] = {}
    for device, pair in value.items():
        require(isinstance(device, str) and bool(device)
                and isinstance(pair, dict)
                and set(pair) == {"rbytes", "wbytes"}
                and type(pair["rbytes"]) is int and type(pair["wbytes"]) is int
                and pair["rbytes"] >= 0 and pair["wbytes"] >= 0,
                "physical-I/O device map invalid")
        result[device] = pair["rbytes"], pair["wbytes"]
    return result


def physical_delta(before: dict[str, tuple[int, int]],
                   after: dict[str, tuple[int, int]]) -> int:
    require(before.keys() <= after.keys(), "physical-I/O device vanished")
    total = 0
    for device, current in after.items():
        pair = before.get(device, (0, 0))
        require(current[0] >= pair[0] and current[1] >= pair[1],
                "physical-I/O counter reversed")
        total += current[0] - pair[0] + current[1] - pair[1]
    return total


def physical_sample(io_state: dict[str, dict[str, tuple[int, int]]],
                    current: dict[str, tuple[int, int]]) -> int:
    """Reject changes hidden between polls, then charge from attempt start."""
    physical_delta(io_state["last"], current)
    total = physical_delta(io_state["baseline"], current)
    io_state["last"] = current
    return total


def memory_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            require(len(fields) == 3 and fields[2] == "kB" and fields[1].isdigit(),
                    "MemAvailable unit differs")
            return int(fields[1])
    raise ValueError("HOLD: MemAvailable missing")


def output_bytes(root: Path) -> int:
    total = 0
    for current, dirs, files in os.walk(root):
        for name in dirs:
            require(not (Path(current) / name).is_symlink(),
                    "linked auditor output directory")
        for name in files:
            path = Path(current) / name
            if not path.exists() and ".partial." in name:
                continue
            require(path.is_file() and not path.is_symlink(),
                    "linked or special auditor output file")
            total += path.stat().st_size
    return total


def process_rss_kib(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is not None:
        return 0
    try:
        status = Path(f"/proc/{process.pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError):
        require(process.poll() is not None, "live child status missing")
        return 0
    for line in status.splitlines():
        if line.startswith("State:") and line.split()[1].startswith("Z"):
            return 0
        if line.startswith("VmRSS:"):
            fields = line.split()
            require(len(fields) == 3 and fields[2] == "kB" and fields[1].isdigit(),
                    "live child RSS unit differs")
            return int(fields[1])
    raise ValueError("HOLD: live child RSS missing")


def preexec_owned_child(owner_pid: int, seconds: int) -> None:
    """Arm the lease-bearing child before exec, including the startup interval."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        os._exit(127)
    if os.getppid() != owner_pid:
        os._exit(127)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(seconds)


def supervise(command: list[str], root: Path, lease_fd: int,
              plan_sha: str, name: str,
              io_state: dict[str, dict[str, tuple[int, int]]] | None = None) -> None:
    if io_state is None:
        baseline = physical_counters()
        io_state = {"baseline": baseline, "last": baseline}
    started = time.monotonic()
    owner_pid = os.getpid()
    process = subprocess.Popen(
        command, pass_fds=(lease_fd,), start_new_session=True,
        preexec_fn=lambda: preexec_owned_child(owner_pid, UNIT_SECONDS))  # noqa: PLW1509
    maximum_rss = 0
    try:
        with (root / "RESOURCE.jsonl").open("ab") as trace:
            while True:
                code = process.poll()
                elapsed = time.monotonic() - started
                current_io = physical_counters()
                physical = physical_sample(io_state, current_io)
                rss = process_rss_kib(process)
                maximum_rss = max(maximum_rss, rss)
                available = memory_available_kib()
                free = shutil.disk_usage(root).free
                output = output_bytes(root)
                sample = {"unit": name, "wall_seconds": round(elapsed, 3),
                          "child_pid": process.pid, "child_exit_code": code,
                          "child_rss_kib": rss,
                          "host_mem_available_kib": available,
                          "output_bytes": output, "disk_free_bytes": free,
                          "shared_host_physical_io_bytes": physical,
                          "host_io_baseline": physical_map(io_state["baseline"]),
                          "host_io_devices": physical_map(current_io),
                          "physical_scope": PHYSICAL_SCOPE}
                trace.write(canonical(sample))
                trace.flush()
                os.fsync(trace.fileno())
                require(elapsed < UNIT_SECONDS and rss < RSS_KIB_MAX
                        and available >= AVAILABLE_KIB_MIN
                        and physical < PHYSICAL_BYTES_MAX
                        and output < OUTPUT_BYTES_MAX and free >= FREE_BYTES_MIN,
                        "child exceeded wall/RSS/free-memory/physical-I/O/output/disk gate")
                if code is not None:
                    require(code == 0, f"child {name} exited {code}")
                    result_path = unit_result_path(root, name)
                    result_sha = sha256_file(result_path)
                    resource_path = root / "resource_receipts" / f"{name}.json"
                    if resource_path.exists():
                        require(require_resource(root, plan_sha, name),
                                "previous unit resource acceptance differs")
                    else:
                        publish(resource_path, {
                            "schema": "sf_dlite_independent_pack_unit_resource_v1",
                            "status": "PASS_MONITORED_UNIT",
                            "unit": name, "plan_sha256": plan_sha,
                            "result_sha256": result_sha,
                            "wall_seconds": round(elapsed, 3),
                            "sampled_peak_child_rss_kib": maximum_rss,
                            "shared_host_physical_io_bytes": physical,
                            "host_io_baseline": physical_map(io_state["baseline"]),
                            "host_io_devices_at_exit": physical_map(current_io),
                            "output_bytes_at_exit": output,
                            "disk_free_bytes_at_exit": free,
                            "physical_scope": PHYSICAL_SCOPE,
                        })
                    return
                time.sleep(5)
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                require(process.poll() is not None, "child process group vanished")
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    require(process.poll() is not None, "child process group vanished")
                process.wait()


def unit_result_path(root: Path, name: str) -> Path:
    if name == "label-index":
        return root / "LABEL-INDEX.json"
    if name == "final-readback":
        return root / "AUDIT-TERMINAL.json"
    if name.startswith(("audit-", "verify-")):
        phase, raw_id = name.split("-", 1)
        require(len(raw_id) == 6 and raw_id.isdigit() and int(raw_id) < SHARDS,
                "unit shard identity differs")
        directory = "audit_receipts" if phase == "audit" else "verify_receipts"
        return root / directory / f"shard_{raw_id}.json"
    raise ValueError("HOLD: unknown resource unit")


def require_resource(root: Path, plan_sha: str, name: str) -> bool:
    path = root / "resource_receipts" / f"{name}.json"
    if not path.exists():
        return False
    value, _ = read_receipt(path)
    require(value.get("schema") == "sf_dlite_independent_pack_unit_resource_v1"
            and value.get("status") == "PASS_MONITORED_UNIT"
            and value.get("unit") == name
            and value.get("plan_sha256") == plan_sha
            and value.get("result_sha256") == sha256_file(unit_result_path(root, name))
            and 0 <= value.get("wall_seconds", UNIT_SECONDS) < UNIT_SECONDS
            and 0 <= value.get("sampled_peak_child_rss_kib", RSS_KIB_MAX) < RSS_KIB_MAX
            and 0 <= value.get("shared_host_physical_io_bytes", PHYSICAL_BYTES_MAX)
            < PHYSICAL_BYTES_MAX
            and value.get("physical_scope") == PHYSICAL_SCOPE
            and physical_delta(
                parsed_physical_map(value.get("host_io_baseline")),
                parsed_physical_map(value.get("host_io_devices_at_exit"))) ==
            value["shared_host_physical_io_bytes"],
            "unit resource receipt differs")
    return True


def install_owned_child(owner_pid: int) -> None:
    """Bound an orphan child if its watchdog parent dies unexpectedly."""
    require(owner_pid > 1, "invalid child watchdog owner")
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0,
            "could not arm Linux parent-death signal")
    require(os.getppid() == owner_pid, "watchdog parent died before child setup")

    def expired(_signum: int, _frame: object) -> None:
        raise TimeoutError("HOLD: owned child exceeded 30-minute wall cap")

    signal.signal(signal.SIGALRM, expired)
    remaining = signal.getitimer(signal.ITIMER_REAL)[0]
    signal.setitimer(signal.ITIMER_REAL,
                     min(remaining, UNIT_SECONDS) if remaining > 0 else UNIT_SECONDS)


def run_shard(root: Path, plan: dict[str, Any], census: dict[str, Any],
              builder: dict[str, Any], shard_id: int, *, verify: bool) -> None:
    audit_path = root / "audit_receipts" / f"shard_{shard_id:06d}.json"
    verify_path = root / "verify_receipts" / f"shard_{shard_id:06d}.json"
    label_receipt, label_sha = read_receipt(root / "LABEL-INDEX.json")
    roster = roster_array(census, hash_bytes=False)
    labels = np.memmap(label_receipt["table"]["path"], dtype="<f4", mode="r",
                       shape=(ROWS, 3))
    proof = audit_shard(shard_id, census, roster, labels, builder["arms"])
    proof.update({"plan_sha256": plan["self_sha256"],
                  "builder_terminal_sha256": plan["builder_terminal"]["sha256"],
                  "label_index_sha256": label_sha,
                  **source_provenance(),
                  "auditor_runtime": runtime_provenance()})
    if verify:
        original, original_sha = read_receipt(audit_path)
        require(proof == {key: value for key, value in original.items()
                          if key != "previous_audit_sha256"},
                "independent shard proof changed at readback")
        previous = (read_receipt(root / "verify_receipts" /
                    f"shard_{shard_id - 1:06d}.json")[1] if shard_id else "0" * 64)
        expected = {"schema": "sf_dlite_independent_pair_shard_verify_v1",
                    "shard_id": shard_id, "audit_sha256": original_sha,
                    "previous_verify_sha256": previous, "proof": proof}
        if verify_path.exists():
            existing, _ = read_receipt(verify_path)
            require(existing == expected, "orphan verify receipt changed at readback")
        else:
            publish(verify_path, expected)
    else:
        previous = (read_receipt(root / "audit_receipts" /
                    f"shard_{shard_id - 1:06d}.json")[1] if shard_id else "0" * 64)
        proof["previous_audit_sha256"] = previous
        if audit_path.exists():
            existing, _ = read_receipt(audit_path)
            require(existing == proof, "orphan audit receipt changed at readback")
        else:
            publish(audit_path, proof)


def receipt_prefix(root: Path, plan_sha: str,
                   builder_sha: str | None = None) -> tuple[int, int]:
    """Refuse gaps or forged progress before scheduling any new shard work."""
    lengths: list[int] = []
    for directory in ("audit_receipts", "verify_receipts"):
        paths = sorted((root / directory).glob("shard_*.json"))
        require([path.name for path in paths] ==
                [f"shard_{index:06d}.json" for index in range(len(paths))],
                "noncontiguous published shard receipt prefix")
        lengths.append(len(paths))
    audited, verified = lengths
    require(verified <= audited and audited <= SHARDS,
            "verified prefix extends beyond audit prefix")
    prior_audit = "0" * 64
    prior_verify = "0" * 64
    code = source_provenance()
    runtime = runtime_provenance()
    label_sha = (read_receipt(root / "LABEL-INDEX.json")[1]
                 if (root / "LABEL-INDEX.json").exists() else None)
    for shard_id in range(audited):
        audit, audit_sha = read_receipt(root / "audit_receipts" /
                                        f"shard_{shard_id:06d}.json")
        require(audit.get("schema") == "sf_dlite_independent_pair_shard_audit_v1"
                and audit.get("shard_id") == shard_id
                and audit.get("plan_sha256") == plan_sha
                and (builder_sha is None or
                     audit.get("builder_terminal_sha256") == builder_sha)
                and (label_sha is None or audit.get("label_index_sha256") == label_sha)
                and audit.get("auditor_source_sha256") == code["auditor_source_sha256"]
                and audit.get("auditor_core_sha256") == code["auditor_core_sha256"]
                and audit.get("auditor_runtime") == runtime
                and audit.get("previous_audit_sha256") == prior_audit,
                "audit prefix receipt identity or chain differs")
        require_resource(root, plan_sha, f"audit-{shard_id:06d}")
        prior_audit = audit_sha
        if shard_id < verified:
            value, verify_sha = read_receipt(root / "verify_receipts" /
                                              f"shard_{shard_id:06d}.json")
            require(value.get("schema") == "sf_dlite_independent_pair_shard_verify_v1"
                    and value.get("shard_id") == shard_id
                    and value.get("audit_sha256") == audit_sha
                    and value.get("previous_verify_sha256") == prior_verify
                    and value.get("proof") == {key: item for key, item in audit.items()
                                                if key != "previous_audit_sha256"},
                    "verified prefix receipt identity or chain differs")
            require_resource(root, plan_sha, f"verify-{shard_id:06d}")
            prior_verify = verify_sha
    return audited, verified


def verify_final(root: Path, plan: dict[str, Any], census: dict[str, Any],
                 builder: dict[str, Any]) -> dict[str, Any]:
    """Small exact receipt chain/readback and current metadata-stamp gate."""
    require(receipt_prefix(root, plan["self_sha256"],
                           plan["builder_terminal"]["sha256"]) == (SHARDS, SHARDS),
            "independent pack audit prefix incomplete")
    require(require_resource(root, plan["self_sha256"], "label-index"),
            "label index has no durable monitor completion")
    previous_audit = "0" * 64
    previous_verify = "0" * 64
    total = 0
    all_indices: list[np.ndarray] = []
    for shard_id in range(SHARDS):
        audit, audit_sha = read_receipt(root / "audit_receipts" /
                                        f"shard_{shard_id:06d}.json")
        verify, verify_sha = read_receipt(root / "verify_receipts" /
                                          f"shard_{shard_id:06d}.json")
        require(require_resource(root, plan["self_sha256"], f"audit-{shard_id:06d}")
                and require_resource(root, plan["self_sha256"],
                                     f"verify-{shard_id:06d}"),
                "shard proof lacks durable monitor completion")
        require(audit["previous_audit_sha256"] == previous_audit
                and verify["previous_verify_sha256"] == previous_verify
                and verify["audit_sha256"] == audit_sha
                and verify["proof"] == {key: value for key, value in audit.items()
                                         if key != "previous_audit_sha256"}
                and audit["plan_sha256"] == plan["self_sha256"],
                "audited shard receipt chain differs")
        previous_audit, previous_verify = audit_sha, verify_sha
        source = census["shards"][shard_id]
        require(tree_stamp(Path(source["base"])) ==
                audit["source_content"]["base_storage_stamp"]
                and tree_stamp(Path(source["overlay"])) ==
                audit["source_content"]["overlay_tree_stamp"],
                "source changed after verified byte audit")
        for arm in ("control", "candidate"):
            group = builder["arms"][arm]
            arm_root = Path(group["root"])
            require(arm_root.is_dir() and not arm_root.is_symlink()
                    and all(folder.is_dir() and not folder.is_symlink()
                            for folder in (arm_root / "receipts",
                                           arm_root / "roster_index")),
                    "paired arm or receipt directory is linked")
            path = arm_root / f"shard_{shard_id:06d}.zarr"
            require(tree_stamp(path) == audit["arms"][arm]["native_tree_stamp"],
                    "packed files changed after verified byte audit")
            manifest = (arm_root / "receipts" /
                        f"shard_{shard_id:06d}.{arm}.files.json")
            require(sha256_file(manifest) ==
                    audit["arms"][arm]["file_manifest_sha256"],
                    "packed manifest changed after verified byte audit")
            sidecar = arm_root / "roster_index" / f"shard_{shard_id:06d}.i4"
            require(sidecar.is_file() and not sidecar.is_symlink(),
                    "paired sidecar missing or linked")
            sidecar_bytes = sidecar.read_bytes()
            require(digest(sidecar_bytes) == audit["arms"][arm]["roster_index_sha256"],
                    "paired sidecar changed after verified byte audit")
            if arm == "control":
                all_indices.append(np.frombuffer(sidecar_bytes, dtype="<i4"))
        total += audit["rows"]
    require(total == ROWS and
            np.array_equal(np.sort(np.concatenate(all_indices)),
                           np.arange(ROWS, dtype="<i4")),
            "verified paired roster does not cover every row once")
    roster_array(census, hash_bytes=True)
    label_index(root, plan, census, pinned(Path(plan["label_terminal"]["path"]),
                                           plan["label_terminal"]["sha256"]))
    candidate = dict(builder)
    candidate["schema"] = "sf_dlite_independent_pair_qualification_candidate_v1"
    candidate["status"] = "PROOF_READY_UNADMITTED"
    candidate["source"] = dict(builder["source"])
    candidate["source"]["pack_builder_terminal_sha256"] = plan["builder_terminal"]["sha256"]
    candidate["source"]["independent_pack_auditor_source_sha256"] = sha256_file(Path(__file__))
    candidate["source"]["independent_pack_auditor_core_sha256"] = (
        source_provenance()["auditor_core_sha256"])
    candidate["independent_auditor"] = {
        "plan_sha256": plan["self_sha256"], "audit_tail_sha256": previous_audit,
        "verify_tail_sha256": previous_verify, "shard_receipts": SHARDS,
        "rows": ROWS, "auditor_sources": source_provenance(),
        "auditor_runtime": runtime_provenance(),
        "label_index_sha256": read_receipt(root / "LABEL-INDEX.json")[1],
        "scope": "qualified D8 labels to paired native bytes; no raw search rerun"}
    candidate_path = root / "QUALIFICATION-CANDIDATE.json"
    if candidate_path.exists():
        value, _ = read_receipt(candidate_path)
        require(value == candidate, "existing qualification candidate differs")
    else:
        publish(candidate_path, candidate)
    terminal = root / "AUDIT-TERMINAL.json"
    terminal_value = {"schema": "sf_dlite_independent_paired_pack_audit_terminal_v1",
                      "status": "PASS_AUDITOR_PROOF_READY_FOR_QUALIFICATION",
                      "plan_sha256": plan["self_sha256"],
                      "candidate_sha256": digest(canonical(candidate)),
                      "audit_tail_sha256": previous_audit,
                      "verify_tail_sha256": previous_verify}
    if terminal.exists():
        value, _ = read_receipt(terminal)
        require(value == terminal_value, "existing audit terminal differs")
    else:
        publish(terminal, terminal_value)
    return candidate


def publish_qualification(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Only the parent can expose PASS, after durable monitored child success."""
    terminal, _ = read_receipt(root / "AUDIT-TERMINAL.json")
    candidate_path = root / "QUALIFICATION-CANDIDATE.json"
    candidate, candidate_sha = read_receipt(candidate_path)
    require(terminal.get("schema") == "sf_dlite_independent_paired_pack_audit_terminal_v1"
            and terminal.get("status") == "PASS_AUDITOR_PROOF_READY_FOR_QUALIFICATION"
            and terminal.get("plan_sha256") == plan["self_sha256"]
            and terminal.get("candidate_sha256") == candidate_sha
            and candidate.get("schema") == "sf_dlite_independent_pair_qualification_candidate_v1"
            and candidate.get("status") == "PROOF_READY_UNADMITTED"
            and candidate.get("independent_auditor", {}).get("plan_sha256") ==
            plan["self_sha256"]
            and require_resource(root, plan["self_sha256"], "final-readback"),
            "final candidate lacks accepted monitored proof")
    unit_names = ["label-index"]
    for shard_id in range(SHARDS):
        unit_names.extend((f"audit-{shard_id:06d}", f"verify-{shard_id:06d}"))
    unit_names.append("final-readback")
    receipt_hashes = []
    for name in unit_names:
        require(require_resource(root, plan["self_sha256"], name),
                "missing monitored unit completion")
        receipt_hashes.append(read_receipt(root / "resource_receipts" / f"{name}.json")[1])
    qualified = dict(candidate)
    qualified["schema"] = QUALIFICATION_SCHEMA
    qualified["status"] = QUALIFICATION_STATUS
    qualified["independent_auditor"] = dict(candidate["independent_auditor"])
    qualified["independent_auditor"]["resource_receipts_sha256"] = (
        digest(canonical(receipt_hashes)))
    qualified["independent_auditor"]["final_resource_receipt_sha256"] = (
        receipt_hashes[-1])
    final = Path(plan["qualification_path"])
    if final.exists():
        value, _ = read_receipt(final)
        require(value == qualified, "existing qualification differs")
    else:
        publish(final, qualified)
    return qualified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("run", "labels", "shard", "verify", "final"),
                        default="run")
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--inherited-lease-fd", type=int)
    parser.add_argument("--owner-pid", type=int)
    args = parser.parse_args()
    if args.mode != "run":
        require(args.owner_pid is not None, "owned child watchdog PID missing")
        install_owned_child(args.owner_pid)
    artifact = ARTIFACT_ROOT.resolve(strict=True)
    require(args.output.is_absolute() and args.output == args.output.resolve(strict=False)
            and args.output.is_relative_to(artifact) and args.output != artifact
            and args.output.parent.is_dir() and not args.output.parent.is_symlink(),
            "output must be canonical child of artifact root")
    plan = read_plan(args.plan, args.plan_sha256)
    plan["self_sha256"] = args.plan_sha256
    census, builder, label = sources(plan)
    pack_root = Path(builder["arms"]["control"]["root"]).parent
    require(not args.output.is_relative_to(pack_root),
            "auditor checkpoint root must be separate from paired pack")
    if args.mode != "run":
        fd = args.inherited_lease_fd
        require(fd is not None and Path(os.readlink(f"/proc/self/fd/{fd}")) == LEASE,
                "child lacks shared heavy-I/O lease")
        if args.mode == "labels":
            label_index(args.output, plan, census, label)
        elif args.mode in ("shard", "verify"):
            require(args.shard_id is not None and 0 <= args.shard_id < SHARDS,
                    "child shard identity differs")
            if args.shard_id is None:
                raise ValueError("HOLD: child shard ID missing")
            shard_id = args.shard_id
            run_shard(args.output, plan, census, builder, shard_id,
                      verify=args.mode == "verify")
        else:
            verify_final(args.output, plan, census, builder)
        return
    require(args.shard_id is None, "parent cannot take a shard ID")
    with LEASE.open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline_io = physical_counters()
        io_state = {"baseline": baseline_io, "last": baseline_io}
        require(memory_available_kib() >= AVAILABLE_KIB_MIN,
                "host free-memory floor at launch")
        require(shutil.disk_usage(args.output.parent).free >= FREE_BYTES_MIN,
                "output disk free-space floor at launch")
        args.output.mkdir(exist_ok=True)
        sync_directory(args.output.parent)
        for name in ("label_stage", "audit_receipts", "verify_receipts",
                     "resource_receipts"):
            path = args.output / name
            path.mkdir(exist_ok=True)
            require(not path.is_symlink(), "output directory is linked")
            sync_directory(args.output)
        def child(mode: str, shard_id: int | None = None) -> list[str]:
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--plan", str(args.plan), "--plan-sha256", args.plan_sha256,
                       "--output", str(args.output), "--mode", mode,
                       "--inherited-lease-fd", str(lease.fileno()),
                       "--owner-pid", str(os.getpid())]
            if shard_id is not None:
                command.extend(("--shard-id", str(shard_id)))
            return command
        supervise(child("labels"), args.output, lease.fileno(), args.plan_sha256,
                  "label-index", io_state)
        receipt_prefix(args.output, args.plan_sha256,
                       plan["builder_terminal"]["sha256"])
        for shard_id in range(SHARDS):
            if not require_resource(args.output, args.plan_sha256,
                                    f"audit-{shard_id:06d}"):
                supervise(child("shard", shard_id), args.output, lease.fileno(),
                          args.plan_sha256,
                          f"audit-{shard_id:06d}", io_state)
            if not require_resource(args.output, args.plan_sha256,
                                    f"verify-{shard_id:06d}"):
                supervise(child("verify", shard_id), args.output, lease.fileno(),
                          args.plan_sha256,
                          f"verify-{shard_id:06d}", io_state)
        supervise(child("final"), args.output, lease.fileno(), args.plan_sha256,
                  "final-readback", io_state)
        publish_qualification(args.output, plan)


if __name__ == "__main__":
    main()
