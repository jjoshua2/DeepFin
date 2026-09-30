"""Tiny native transport and crash fixtures; no qualified corpus is opened."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import audit_sf_dlite_paired_pack as runner
from scripts import sf_dlite_pack_audit_core as audit


def _write_group(path: Path, arrays: dict[str, np.ndarray]) -> None:
    group = zarr.open_group(str(path), mode="w")
    for name, value in arrays.items():
        group.create_dataset(name, data=value, shape=value.shape)


def _roster() -> np.ndarray:
    dtype = np.dtype([
        ("source_id", "<u4"), ("source_row", "<u4"),
        ("shard_id", "<u2"), ("stored_row", "<u2"),
        ("worker", "<i4"), ("game", "<i8"), ("game_ply", "<i4"),
        ("original_key", "S16"), ("stored_key", "S16"),
        ("input_sha256", "S32"), ("mask_sha256", "S32"),
        ("context_sha256", "S32"), ("rank", "S16"),
        ("pieces", "u1"), ("absolute_ply", "<u4"),
        ("rule50", "<u2"), ("stm", "u1"),
    ], align=False)
    assert dtype.itemsize == 180
    rows = np.zeros(3, dtype=dtype)
    rows["stored_row"] = [0, 2, 3]
    rows["source_row"] = [7, 9, 10]
    rows["worker"] = [2, 2, 2]
    rows["game"] = [991, 991, 992]
    rows["game_ply"] = [4, 6, 10]
    rows["input_sha256"] = [bytes([n]) * 32 for n in (1, 2, 3)]
    return rows


def _arrays() -> dict[str, np.ndarray]:
    count = 4
    arrays = {
        "x": np.zeros((count, 175, 8, 8), dtype="<f2"),
        "policy_target": np.zeros((count, 1858), dtype="<f2"),
        "wdl_target": np.array([0, 1, 2, 1], dtype="i1"),
        "priority": np.ones(count, dtype="<f4"),
        "has_policy": np.ones(count, dtype="u1"),
        "game_id": np.array([991, 999, 991, 992], dtype="<i8"),
        "has_game_id": np.ones(count, dtype="u1"),
        "ply_index": np.array([4, 5, 6, 10], dtype="<i4"),
        "has_ply_index": np.ones(count, dtype="u1"),
        "is_network_turn": np.ones(count, dtype="u1"),
        "has_is_network_turn": np.ones(count, dtype="u1"),
        "is_selfplay": np.zeros(count, dtype="u1"),
        "has_is_selfplay": np.ones(count, dtype="u1"),
        "search_wdl": np.tile(np.array([.25, .5, .25], dtype="<f2"), (count, 1)),
        "has_search_wdl": np.ones(count, dtype="u1"),
        "legal_mask": np.zeros((count, 1858), dtype="u1"),
        "has_legal_mask": np.ones(count, dtype="u1"),
    }
    for row in range(count):
        arrays["x"][row, 0, 0, 0] = row
        arrays["policy_target"][row, row + 1] = 1
        arrays["legal_mask"][row, row + 1] = 1
    return arrays


def _fixture(root: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    base = root / "base.zarr"
    overlay = root / "overlay.zarr"
    base_arrays = _arrays()
    _write_group(base, base_arrays)
    (base / "row_provenance.npz").write_bytes(b"tiny qualified provenance")
    selected = np.array([0, 2, 3], dtype=np.int64)
    search_e = np.tile(np.array([.125, .75, .125], dtype="<f2"), (4, 1))
    policy_e = base_arrays["policy_target"].copy()
    policy_e[:, 0] = np.float16(.125)
    _write_group(overlay, {"policy_target": policy_e, "search_wdl": search_e})
    base_sha = audit.plain_content_sha256(base)
    declaration = {"schema": 2, "kind": "immutable-target-overlay",
                   "base": str(base), "base_seal": {"path": "/synthetic/seal",
                                                    "sha256": "a" * 64},
                   "base_content_sha256": base_sha,
                   "replacements": ["policy_target", "search_wdl"]}
    (overlay / "target_overlay.json").write_text(json.dumps(declaration))
    local_sha = audit.plain_content_sha256(overlay)
    envelope = {"kind": "immutable-policy-overlay-v1", "base": base_sha,
                "base_seal": declaration["base_seal"], "local": local_sha}
    overlay_sha = hashlib.sha256(json.dumps(envelope, sort_keys=True).encode()).hexdigest()
    source = {"base": str(base), "overlay": str(overlay),
              "base_storage_stamp": audit.tree_stamp(base),
              "overlay_tree_stamp": audit.tree_stamp(overlay),
              "provenance_sha256": audit.sha256_file(base / "row_provenance.npz"),
              "overlay_declaration_sha256": audit.sha256_file(
                  overlay / "target_overlay.json"),
              "base_content_sha256": base_sha,
              "qualified_overlay_content_sha256": overlay_sha}
    roster = _roster()
    labels = np.array([[.8, .1, .1], [.1, .2, .7], [.3, .4, .3]], dtype="<f4")
    control = {key: value[selected].copy() for key, value in base_arrays.items()}
    control["policy_target"] = policy_e[selected].copy()
    control["search_wdl"] = search_e[selected].copy()
    candidate = {key: value.copy() for key, value in control.items()}
    candidate["search_wdl"] = ((labels + np.float32(2) *
                                control["search_wdl"].astype("<f4")) /
                               np.float32(3)).astype("<f2")
    identity = hashlib.sha256(audit.IDENTITY_DOMAIN)
    for row in roster:
        identity.update(bytes.fromhex("d" * 64))
        identity.update(struct.pack("<iqIIHH", int(row["worker"]), int(row["game"]),
                                    int(row["source_id"]), int(row["source_row"]),
                                    int(row["shard_id"]), int(row["stored_row"])))
        identity.update(np.asarray(row).tobytes())
    stored_sha = hashlib.sha256(roster["stored_row"].astype("<u2").tobytes()).hexdigest()
    arms: dict[str, Any] = {}
    for name, arrays in (("control", control), ("candidate", candidate)):
        arm_root = root / name
        (arm_root / "receipts").mkdir(parents=True)
        (arm_root / "roster_index").mkdir()
        native = arm_root / "shard_000000.zarr"
        _write_group(native, arrays)
        sidecar = arm_root / "roster_index" / "shard_000000.i4"
        sidecar.write_bytes(np.arange(3, dtype="<i4").tobytes())
        manifest = arm_root / "receipts" / f"shard_000000.{name}.files.json"
        files = audit.native_tree(native)
        manifest.write_bytes(runner.canonical({"schema": "sf_dlite_native_zarr_file_manifest_v1",
                                               "arm": name, "shard_id": 0,
                                               "entries": files}))
        entry = {"name": native.name, "path": str(native), "rows": 3,
                 "source_identity_sha256": identity.hexdigest(),
                 "selected_stored_rows_sha256": stored_sha,
                 "roster_index_sha256": audit.sha256_file(sidecar),
                 "nonmain_arrays_sha256": audit.digest_nonmain(arrays),
                 "search_wdl_sha256": hashlib.sha256(
                     np.ascontiguousarray(arrays["search_wdl"]).tobytes()).hexdigest(),
                 "roster_index_path": str(sidecar),
                 "file_manifest_path": str(manifest),
                 "file_manifest_sha256": audit.sha256_file(manifest),
                 "file_count": len(files),
                 "compressed_bytes": sum(item["bytes"] for item in files)}
        arms[name] = {"root": str(arm_root), "shards": [entry]}
    census = {"shards": [source], "sources": [{"source_namespace": "d" * 64}]}
    return census, roster, labels, arms


def test_tiny_native_pair_full_source_row_and_file_proof(tmp_path: Path) -> None:
    census, roster, labels, arms = _fixture(tmp_path)
    result = audit.audit_shard(0, census, roster, labels, arms)
    assert result["rows"] == 3
    assert result["arms"]["control"]["nonmain_arrays_sha256"] == result["arms"][
        "candidate"]["nonmain_arrays_sha256"]
    assert result["arms"]["control"]["search_wdl_sha256"] != result["arms"][
        "candidate"]["search_wdl_sha256"]


@pytest.mark.parametrize("fault", ["control_nonmain", "candidate_main", "game_id",
                                   "sidecar", "source_stamp", "manifest", "label"])
def test_tiny_native_pair_refuses_byte_or_source_tamper(tmp_path: Path, fault: str) -> None:
    census, roster, labels, arms = _fixture(tmp_path)
    if fault in ("control_nonmain", "candidate_main", "game_id"):
        name = "candidate" if fault == "candidate_main" else "control"
        group = zarr.open_group(str(tmp_path / name / "shard_000000.zarr"), mode="a")
        key = {"control_nonmain": "legal_mask", "candidate_main": "search_wdl",
               "game_id": "game_id"}[fault]
        value = np.asarray(group[key][:])
        value.flat[0] = value.flat[0] + 1
        group[key][:] = value
    elif fault == "sidecar":
        (tmp_path / "control" / "roster_index" / "shard_000000.i4").write_bytes(b"bad")
    elif fault == "source_stamp":
        (tmp_path / "base.zarr" / "row_provenance.npz").write_bytes(b"tampered")
    elif fault == "manifest":
        (tmp_path / "candidate" / "receipts" /
         "shard_000000.candidate.files.json").write_bytes(b"{}\n")
    else:
        labels[0] = np.array([.1, .2, .7], dtype="<f4")
    with pytest.raises(ValueError, match="HOLD"):
        audit.audit_shard(0, census, roster, labels, arms)


def test_receipt_byte_tamper_refused(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    runner.publish(path, {"source": "pinned", "rows": 3})
    path.write_text('{"source":"pinned","rows":3}\n')
    with pytest.raises(ValueError, match="HOLD"):
        runner.read_receipt(path)


def test_pinned_json_parses_the_exact_hashed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "SOURCE.json"
    original = {"source": "qualified"}
    changed = {"source": "replacement"}
    raw = runner.canonical(original)
    path.write_bytes(raw)
    real_hash = runner.sha256_file

    def hash_then_swap(item: Path) -> str:
        value = real_hash(item)
        item.write_bytes(runner.canonical(changed))
        return value

    monkeypatch.setattr(runner, "sha256_file", hash_then_swap)
    assert runner.pinned(path, hashlib.sha256(raw).hexdigest()) == original
    assert path.read_bytes() == raw


def test_independent_three_row_label_index_and_source_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "ROWS", 3)
    roster = _roster()
    monkeypatch.setattr(runner, "roster_array", lambda *_args, **_kwargs: roster)
    (tmp_path / "label_stage").mkdir()
    label_root = tmp_path / "qualified_labels"
    roster_sha = runner.EXPECTED_PINS["roster"]
    source = {"path": "/synthetic/source.jsonl.zst", "sha256": "c" * 64}
    rows = []
    for index, row in enumerate(roster):
        rows.append({"roster_index": index, "source_id": 0,
                     "source_row": int(row["source_row"]),
                     "input_sha256": roster["input_sha256"][index:index + 1].tobytes().hex(),
                     "context_sha256": roster["context_sha256"][index:index + 1].tobytes().hex(),
                     "stored_key_hex": roster["stored_key"][index:index + 1].tobytes().hex(),
                     "depth_requested": 8, "wdl_orientation": "side_to_move",
                     "score": {"d_style_wdl": [.2, .3, .5]}})
    for worker in range(6):
        folder = label_root / f"worker{worker:02d}"
        folder.mkdir(parents=True)
        (folder / "COMPLETE.json").write_text(
            json.dumps({"status": "COMPLETE_WORKER_UNADMITTED",
                        "identity": {"operator_sha256":
                                     runner.EXPECTED_PINS["label_worker"]}}))
    folder = label_root / "worker00"
    data = folder / "s000-b000.jsonl"
    data.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    block = {"schema": "sf_dlite_legacy_d8_checkpointed_label_v1",
             "identity": {"worker_id": 0, "roster_sha256": roster_sha,
                          "authorization_sha256":
                          runner.EXPECTED_PINS["label_authorization"]},
             "data_file": data.name, "data_sha256": audit.sha256_file(data),
             "data_bytes": data.stat().st_size, "source_id": 0,
             "source_path": source["path"], "source_sha256": source["sha256"],
             "rows": 3, "roster_indices_sha256": hashlib.sha256(
                 np.arange(3, dtype="<i4").tobytes()).hexdigest()}
    receipt_path = folder / "s000-b000.receipt.json"
    receipt_path.write_text(json.dumps(block))
    census = {"roster": {"sha256": roster_sha}, "sources": [source]}
    label = {"output_root": str(label_root)}
    audit_root = tmp_path / "qualified_label_audit"
    audit_root.mkdir()
    frozen = {"schema": "sf_dlite_d8_full_label_independent_plan_v1",
              "status": "FROZEN_COMPLETED_LABELS_ZERO_ADMISSION",
              "operator_sha256": runner.EXPECTED_PINS["label_auditor"],
              "launch_terminal_sha256": "f" * 64,
              "selected_roster_sha256": roster_sha,
              "label_root": str(label_root), "rows": 3,
              "blocks": [{"receipt_path": str(receipt_path),
                          "receipt_sha256": audit.sha256_file(receipt_path),
                          "data_path": str(data), "data_sha256": audit.sha256_file(data),
                          "data_bytes": data.stat().st_size, "rows": 3,
                          "source_id": 0, "worker_id": 0}]}
    frozen_path = audit_root / "PLAN.json"
    frozen_path.write_text(json.dumps(frozen))
    audit_terminal = audit_root / "TERMINAL.json"
    audit_terminal.write_text(json.dumps({"plan_sha256": audit.sha256_file(frozen_path),
                                          "blocks": 1}))
    plan = {"self_sha256": "e" * 64,
            "label_terminal": {"sha256": "f" * 64},
            "label_audit": {"path": str(audit_terminal),
                            "sha256": audit.sha256_file(audit_terminal)}}
    result = runner.label_index(tmp_path, plan, census, label)
    assert result["rows"] == 3
    assert result["blocks"] == 1
    assert runner.label_index(tmp_path, plan, census, label) == result
    actual = np.fromfile(result["table"]["path"], dtype="<f4").reshape(3, 3)
    assert np.array_equal(actual, np.tile(np.array([.2, .3, .5], dtype="<f4"), (3, 1)))
    data.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="HOLD"):
        runner.label_index(tmp_path, plan, census, label)
    rows[0]["score"]["d_style_wdl"] = [.1, .4, .5]
    data.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    block["data_sha256"] = audit.sha256_file(data)
    block["data_bytes"] = data.stat().st_size
    receipt_path.write_text(json.dumps(block))
    fresh = tmp_path / "new_auditor_stage"
    fresh.mkdir()
    (fresh / "label_stage").mkdir()
    with pytest.raises(ValueError, match="HOLD"):
        runner.label_index(fresh, plan, census, label)


def test_receipt_prefix_refuses_gap_and_changed_prior(tmp_path: Path) -> None:
    (tmp_path / "audit_receipts").mkdir()
    (tmp_path / "verify_receipts").mkdir()
    plan_sha = "a" * 64
    audit_value = {"schema": "sf_dlite_independent_pair_shard_audit_v1",
                   "shard_id": 0, "plan_sha256": plan_sha,
                   "auditor_source_sha256": audit.sha256_file(Path(runner.__file__)),
                   "auditor_core_sha256": audit.sha256_file(Path(audit.__file__)),
                   "auditor_runtime": runner.runtime_provenance(),
                   "previous_audit_sha256": "0" * 64}
    audit_path = tmp_path / "audit_receipts" / "shard_000000.json"
    audit_sha = runner.publish(audit_path, audit_value)
    assert runner.receipt_prefix(tmp_path, plan_sha) == (1, 0)
    verify_path = tmp_path / "verify_receipts" / "shard_000000.json"
    runner.publish(verify_path, {"schema": "sf_dlite_independent_pair_shard_verify_v1",
                                 "shard_id": 0, "audit_sha256": audit_sha,
                                 "previous_verify_sha256": "0" * 64,
                                 "proof": {key: value for key, value in audit_value.items()
                                           if key != "previous_audit_sha256"}})
    assert runner.receipt_prefix(tmp_path, plan_sha) == (1, 1)
    audit_path.write_bytes(runner.canonical({**audit_value, "rows": 999}))
    with pytest.raises(ValueError, match="HOLD"):
        runner.receipt_prefix(tmp_path, plan_sha)
    audit_path.write_bytes(runner.canonical(audit_value))
    (tmp_path / "audit_receipts" / "shard_000002.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="HOLD"):
        runner.receipt_prefix(tmp_path, plan_sha)


def test_full07_label_terminal_requires_separate_monitored_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    monitor_source = tmp_path / "reviewed-monitor.py"
    monitor_source.write_text("reviewed source\n")
    monitor_sha = audit.sha256_file(monitor_source)
    freezer_source = tmp_path / "reviewed-freezer.py"
    freezer_source.write_text("reviewed freezer\n")
    freezer_sha = audit.sha256_file(freezer_source)
    auditor_source = tmp_path / "reviewed-label-auditor.py"
    auditor_source.write_text("reviewed auditor\n")
    auditor_sha = audit.sha256_file(auditor_source)
    monkeypatch.setattr(runner, "LABEL_MONITOR_SOURCE", monitor_source)
    monkeypatch.setattr(runner, "LABEL_FREEZER_SOURCE", freezer_source)
    monkeypatch.setattr(runner, "LABEL_AUDITOR_SOURCE", auditor_source)
    monkeypatch.setattr(runner, "EXPECTED_PINS",
                        {**runner.EXPECTED_PINS, "label_monitor": monitor_sha,
                         "label_freezer": freezer_sha, "label_auditor": auditor_sha})
    operations = tmp_path / "operations"
    operations.mkdir()
    audit_root = operations / "sf-dlite-full07-label-independent-audit-20260930"
    session_root = operations / "sf-dlite-full07-label-independent-session-test"
    freeze_root = operations / "sf-dlite-full07-label-independent-freeze01-20260930"
    audit_root.mkdir()
    session_root.mkdir()
    freeze_root.mkdir()
    monkeypatch.setattr(runner, "LABEL_AUDIT_ROOT", audit_root)
    monkeypatch.setattr(runner, "LABEL_FREEZE_ROOT", freeze_root)
    frozen_plan = audit_root / "PLAN.json"
    frozen_plan.write_text("frozen plan\n")
    terminal = audit_root / "TERMINAL.json"
    terminal.write_text("audited terminal\n")
    plan_sha = audit.sha256_file(frozen_plan)
    terminal_sha = audit.sha256_file(terminal)
    independent = {"plan_sha256": plan_sha, "blocks": 1}
    log_path = session_root / "finish.log"
    trace_path = session_root / "finish.trace.jsonl"
    log_path.write_text("auditor exited 0\n")
    trace_path.write_text("monitored\n")
    finish = {"schema": "sf_dlite_d8_full07_independent_audit_finish_monitor_v1",
              "status": "PASS_MONITORED_FINISH_ZERO_ADMISSION",
              "supervisor_sha256": monitor_sha,
              "auditor_sha256": auditor_sha,
              "plan_sha256": plan_sha, "terminal_sha256": terminal_sha,
              "finish_step": {"exit_code": 0, "elapsed_seconds": 1,
                              "log_sha256": audit.sha256_file(log_path),
                              "trace_sha256": audit.sha256_file(trace_path)},
              "credit": {"target": 0, "training": 0, "elo": 0}}
    finish_path = session_root / "FINISH.json"
    runner.publish(finish_path, finish)
    freeze = {"schema": "sf_dlite_d8_full07_independent_freeze_session_v1",
              "status": "COMPLETE_PLAN_ONLY_NO_AUDIT_CREDIT",
              "supervisor_sha256": freezer_sha,
              "auditor_sha256": auditor_sha, "plan_sha256": plan_sha,
              "audit_root": str(audit_root),
              "credit": {"labels": 0, "targets": 0, "training": 0, "elo": 0}}
    freeze_path = freeze_root / "COMPLETE.json"
    runner.publish(freeze_path, freeze)
    complete = {"schema": "sf_dlite_d8_full07_independent_audit_session_v2",
                "status": "COMPLETE_ALL_LABEL_BLOCKS_ZERO_ADMISSION",
                "supervisor_sha256": monitor_sha,
                "auditor_sha256": auditor_sha, "freezer_sha256": freezer_sha,
                "freeze_monitor_complete_sha256": audit.sha256_file(freeze_path),
                "plan_path": str(frozen_plan),
                "plan_sha256": plan_sha, "terminal_sha256": terminal_sha,
                "finish_monitor_sha256": audit.sha256_file(finish_path),
                "label_launch_terminal_sha256": "f" * 64,
                "blocks": 1, "rows": 3, "completed_batches": 1,
                "credit": {"target": 0, "training": 0, "elo": 0}}
    complete_path = session_root / "COMPLETE.json"
    runner.publish(complete_path, complete)
    monkeypatch.setattr(runner, "ROWS", 3)
    plan = {"label_audit_session": {"path": str(complete_path),
                                    "sha256": audit.sha256_file(complete_path)},
            "label_audit": {"path": str(terminal), "sha256": terminal_sha},
            "label_terminal": {"sha256": "f" * 64}}
    assert runner.monitored_label_completion(plan, independent) == complete
    failed = session_root / "FAILED.json"
    failed.write_text("failed\n")
    with pytest.raises(ValueError, match="HOLD"):
        runner.monitored_label_completion(plan, independent)
    failed.unlink()
    finish_path.write_text("tampered\n")
    with pytest.raises(ValueError, match="HOLD"):
        runner.monitored_label_completion(plan, independent)


@pytest.mark.parametrize("late_fault", ["sidecar", "manifest"])
def test_tiny_qualification_publishes_only_final_into_pack_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late_fault: str,
) -> None:
    pack_root = tmp_path / "pack"
    pack_root.mkdir()
    census, roster, labels, arms = _fixture(pack_root)
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    for name in ("audit_receipts", "verify_receipts", "resource_receipts"):
        (audit_root / name).mkdir()
    monkeypatch.setattr(runner, "ROWS", 3)
    monkeypatch.setattr(runner, "SHARDS", 1)
    monkeypatch.setattr(runner, "roster_array", lambda *_args, **_kwargs: roster)
    monkeypatch.setattr(runner, "label_index", lambda *_args, **_kwargs: {})
    builder_sha = "f" * 64
    label_terminal = tmp_path / "LABEL-TERMINAL.json"
    label_terminal.write_bytes(runner.canonical({"output_root": "/synthetic/labels"}))
    plan = {"self_sha256": "e" * 64,
            "builder_terminal": {"sha256": builder_sha},
            "label_terminal": {"path": str(label_terminal),
                               "sha256": audit.sha256_file(label_terminal)},
            "qualification_path": str(pack_root / "PAIRED_PACK_QUALIFICATION.json")}
    builder = {"source": {"roster_sha256": "b" * 64},
               "arms": arms, "training_route": {"sampling_mode": "game_epoch"}}
    label_table = audit_root / "labels.f4"
    labels.astype("<f4").tofile(label_table)
    label_sha = runner.publish(audit_root / "LABEL-INDEX.json",
                               {"rows": 3, "table": {"path": str(label_table)}})
    proof = audit.audit_shard(0, census, roster, labels, arms)
    proof.update({"plan_sha256": plan["self_sha256"],
                  "builder_terminal_sha256": builder_sha,
                  "label_index_sha256": label_sha,
                  **runner.source_provenance(),
                  "auditor_runtime": runner.runtime_provenance(),
                  "previous_audit_sha256": "0" * 64})
    audit_path = audit_root / "audit_receipts" / "shard_000000.json"
    audit_sha = runner.publish(audit_path, proof)
    verify_path = audit_root / "verify_receipts" / "shard_000000.json"
    runner.publish(verify_path, {"schema": "sf_dlite_independent_pair_shard_verify_v1",
                                 "shard_id": 0, "audit_sha256": audit_sha,
                                 "previous_verify_sha256": "0" * 64,
                                 "proof": {key: value for key, value in proof.items()
                                           if key != "previous_audit_sha256"}})
    orphan_audit_mtime = audit_path.stat().st_mtime_ns
    orphan_verify_mtime = verify_path.stat().st_mtime_ns
    runner.run_shard(audit_root, plan, census, builder, 0, verify=False)
    runner.run_shard(audit_root, plan, census, builder, 0, verify=True)
    assert audit_path.stat().st_mtime_ns == orphan_audit_mtime
    assert verify_path.stat().st_mtime_ns == orphan_verify_mtime
    def monitored(name: str) -> None:
        runner.publish(audit_root / "resource_receipts" / f"{name}.json",
                       {"schema": "sf_dlite_independent_pack_unit_resource_v1",
                        "status": "PASS_MONITORED_UNIT", "unit": name,
                        "plan_sha256": plan["self_sha256"],
                        "result_sha256": audit.sha256_file(
                            runner.unit_result_path(audit_root, name)),
                        "wall_seconds": 1, "sampled_peak_child_rss_kib": 100,
                        "shared_host_physical_io_bytes": 100})
    for name in ("label-index", "audit-000000", "verify-000000"):
        monitored(name)
    candidate = runner.verify_final(audit_root, plan, census, builder)
    final = pack_root / "PAIRED_PACK_QUALIFICATION.json"
    assert candidate["status"] == "PROOF_READY_UNADMITTED"
    assert not final.exists()
    with pytest.raises(ValueError, match="HOLD"):
        runner.publish_qualification(audit_root, plan)
    monitored("final-readback")
    qualified = runner.publish_qualification(audit_root, plan)
    assert qualified["status"] == runner.QUALIFICATION_STATUS
    assert runner.read_receipt(final)[0] == qualified
    assert (audit_root / "AUDIT-TERMINAL.json").is_file()
    assert not (audit_root / final.name).exists()
    original_mtime = final.stat().st_mtime_ns
    assert runner.verify_final(audit_root, plan, census, builder) == candidate
    assert runner.publish_qualification(audit_root, plan) == qualified
    assert final.stat().st_mtime_ns == original_mtime
    if late_fault == "sidecar":
        (pack_root / "candidate" / "roster_index" / "shard_000000.i4").write_bytes(b"bad")
    else:
        (pack_root / "candidate" / "receipts" /
         "shard_000000.candidate.files.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="HOLD"):
        runner.verify_final(audit_root, plan, census, builder)


@pytest.mark.parametrize("kill_point", ["before", "after"])
def test_subprocess_sigkill_and_new_process_resume(tmp_path: Path,
                                                   kill_point: str) -> None:
    first = tmp_path / "shard_000000.json"
    first_sha = runner.publish(first, {"shard_id": 0})
    original_mtime = first.stat().st_mtime_ns
    second = tmp_path / "shard_000001.json"
    helper = """
import os, signal, sys
from pathlib import Path
from scripts import audit_sf_dlite_paired_pack as a
path = Path(sys.argv[1])
prior = sys.argv[2]
point = sys.argv[3]
if point == 'before':
    def die(*_args, **_kwargs):
        os.kill(os.getpid(), signal.SIGKILL)
    a.os.link = die
elif point == 'after':
    original = a.os.link
    def die(*args, **kwargs):
        original(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGKILL)
    a.os.link = die
if not path.exists():
    a.publish(path, {'shard_id': 1, 'previous_audit_sha256': prior})
assert a.read_receipt(path)[0]['previous_audit_sha256'] == prior
"""
    worktree = Path(__file__).resolve().parents[1]
    killed = subprocess.run([sys.executable, "-c", helper, str(second), first_sha,
                             kill_point], cwd=worktree, check=False)
    assert killed.returncode == -signal.SIGKILL
    assert len(list(tmp_path.glob("shard_000001.json.partial.*"))) == 1
    committed_mtime = second.stat().st_mtime_ns if kill_point == "after" else None
    resumed = subprocess.run([sys.executable, "-c", helper, str(second), first_sha,
                              "resume"], cwd=worktree, check=False)
    assert resumed.returncode == 0
    assert runner.read_receipt(first)[1] == first_sha
    assert first.stat().st_mtime_ns == original_mtime
    assert runner.read_receipt(second)[0]["previous_audit_sha256"] == first_sha
    if committed_mtime is not None:
        assert second.stat().st_mtime_ns == committed_mtime


def test_owned_child_alarm_bounds_orphan_wall() -> None:
    helper = """
import os, time
from scripts import audit_sf_dlite_paired_pack as audit
audit.UNIT_SECONDS = 1
audit.install_owned_child(os.getppid())
time.sleep(5)
"""
    result = subprocess.run([sys.executable, "-c", helper],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, check=False, timeout=5)
    assert result.returncode != 0
    assert b"owned child exceeded 30-minute wall cap" in result.stderr


def test_preexec_guard_caps_child_before_python_watchdog() -> None:
    owner_pid = os.getpid()
    result = subprocess.run(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=Path(__file__).resolve().parents[1], check=False, timeout=5,
        preexec_fn=lambda: runner.preexec_owned_child(owner_pid, 1))
    assert result.returncode == -signal.SIGALRM


def test_preexec_parent_kill_releases_inherited_lease(tmp_path: Path) -> None:
    lease_path = tmp_path / "lease"
    pid_path = tmp_path / "child.pid"
    marker = tmp_path / "child-started"
    owner = """
import fcntl, os, subprocess, sys, time
from pathlib import Path
from scripts import audit_sf_dlite_paired_pack as audit
with Path(sys.argv[1]).open('a+b') as lease:
    fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
    owner_pid = os.getpid()
    child = subprocess.Popen([sys.executable, '-c', sys.argv[4], sys.argv[3]],
                             pass_fds=(lease.fileno(),),
                             preexec_fn=lambda: audit.preexec_owned_child(owner_pid, 30))
    Path(sys.argv[2]).write_text(str(child.pid))
    time.sleep(30)
"""
    child = """
import sys, time
from pathlib import Path
Path(sys.argv[1]).write_text('started')
time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", owner, str(lease_path), str(pid_path),
         str(marker), child], cwd=Path(__file__).resolve().parents[1])
    child_pid = 0
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (pid_path.exists() and
                                                   marker.exists()):
            time.sleep(.02)
        assert marker.read_text() == "started"
        child_pid = int(pid_path.read_text())
        os.kill(process.pid, signal.SIGKILL)
        assert process.wait(timeout=5) == -signal.SIGKILL
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child_pid}/status")
            if not status.exists() or "State:\tZ" in status.read_text():
                break
            time.sleep(.02)
        else:
            pytest.fail("preexec-guarded child survived parent SIGKILL")
        with lease_path.open("a+b") as lease:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_parent_sigkill_terminates_owned_child(tmp_path: Path) -> None:
    marker = tmp_path / "armed"
    child_pid_file = tmp_path / "child.pid"
    child = """
import os, sys, time
from pathlib import Path
from scripts.audit_sf_dlite_paired_pack import install_owned_child
install_owned_child(int(sys.argv[1]))
Path(sys.argv[2]).write_text('armed')
time.sleep(30)
"""
    owner = """
import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', sys.argv[1],
                          str(os.getpid()), sys.argv[2]])
Path(sys.argv[3]).write_text(str(child.pid))
time.sleep(30)
"""
    process = subprocess.Popen([sys.executable, "-c", owner, child,
                                str(marker), str(child_pid_file)],
                               cwd=Path(__file__).resolve().parents[1])
    child_pid = 0
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (marker.exists() and
                                                   child_pid_file.exists()):
            time.sleep(.02)
        assert marker.read_text() == "armed"
        child_pid = int(child_pid_file.read_text())
        os.kill(process.pid, signal.SIGKILL)
        assert process.wait(timeout=5) == -signal.SIGKILL
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child_pid}/status")
            if not status.exists() or "State:\tZ" in status.read_text():
                break
            time.sleep(.02)
        else:
            pytest.fail("owned child survived parent SIGKILL")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_parent_killed_after_unit_proof_restarts_before_monitor_acceptance(
    tmp_path: Path,
) -> None:
    (tmp_path / "resource_receipts").mkdir()
    marker = tmp_path / "proof_written"
    owner = """
import os, sys
from pathlib import Path
from scripts import audit_sf_dlite_paired_pack as audit
with Path(sys.argv[1]).open('a+b') as lease:
    child = [sys.executable, '-c', sys.argv[2], str(os.getpid()),
             sys.argv[3], sys.argv[4], 'hold']
    audit.supervise(child, Path(sys.argv[3]), lease.fileno(), 'a'*64, 'label-index')
"""
    child = """
import os, sys, time
from pathlib import Path
from scripts import audit_sf_dlite_paired_pack as audit
audit.install_owned_child(int(sys.argv[1]))
root = Path(sys.argv[2]); marker = Path(sys.argv[3])
proof = root/'LABEL-INDEX.json'
if not proof.exists():
    audit.publish(proof, {'rows': 3})
marker.write_text('published')
if sys.argv[4] == 'hold':
    time.sleep(30)
"""
    lease_path = tmp_path / "lease"
    worktree = Path(__file__).resolve().parents[1]
    parent = subprocess.Popen([sys.executable, "-c", owner, str(lease_path),
                               child, str(tmp_path), str(marker)], cwd=worktree)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(.02)
        assert marker.read_text() == "published"
        proof = tmp_path / "LABEL-INDEX.json"
        proof_sha = audit.sha256_file(proof)
        proof_mtime = proof.stat().st_mtime_ns
        os.kill(parent.pid, signal.SIGKILL)
        assert parent.wait(timeout=5) == -signal.SIGKILL
        assert not (tmp_path / "resource_receipts" / "label-index.json").exists()
        with lease_path.open("a+b") as lease:
            command = [sys.executable, "-c", child, str(os.getpid()),
                       str(tmp_path), str(marker), "resume"]
            runner.supervise(command, tmp_path, lease.fileno(), "a" * 64,
                             "label-index")
        assert runner.require_resource(tmp_path, "a" * 64, "label-index")
        assert audit.sha256_file(proof) == proof_sha
        assert proof.stat().st_mtime_ns == proof_mtime
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
