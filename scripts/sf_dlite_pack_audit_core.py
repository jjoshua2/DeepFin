"""Independent byte and row checks for a Selected-E / D-lite native pair.

This module deliberately does not import the pack producer or its array helpers.
The caller supplies pinned census, roster, label, and builder-terminal records.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from typing import Any

import numpy as np
import zarr


FIELDS = frozenset({
    "x", "policy_target", "wdl_target", "priority", "has_policy",
    "game_id", "has_game_id", "ply_index", "has_ply_index",
    "is_network_turn", "has_is_network_turn", "is_selfplay",
    "has_is_selfplay", "search_wdl", "has_search_wdl", "legal_mask",
    "has_legal_mask",
})
IDENTITY_DOMAIN = b"sf-dlite/native-pair-source-qualified-identity/v1\0"


def require(condition: object, reason: str) -> None:
    if not bool(condition):
        raise ValueError("HOLD: " + reason)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            length += len(block)
        after = os.fstat(stream.fileno())
    require(length == before.st_size and
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns) ==
            (before.st_size, before.st_mtime_ns, before.st_ctime_ns),
            f"file changed during hash: {path}")
    return digest.hexdigest()


def plain_content_sha256(root: Path) -> str:
    """Independent implementation of the qualified Zarr content framing."""
    require(root.is_dir() and not root.is_symlink(), "source is not a real directory")
    digest = hashlib.sha256()
    count = 0
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for name in dirs:
            child = Path(current) / name
            require(child.is_dir() and not child.is_symlink(), "source has linked directory")
        for name in sorted(files):
            child = Path(current) / name
            require(child.is_file() and not child.is_symlink(), "source has linked or special file")
            relative = child.relative_to(root).as_posix().encode("utf-8", "surrogateescape")
            before = child.stat()
            digest.update(struct.pack("<I", len(relative)))
            digest.update(relative)
            digest.update(struct.pack("<Q", before.st_size))
            length = 0
            with child.open("rb") as stream:
                while block := stream.read(1 << 20):
                    length += len(block)
                    digest.update(block)
                after = os.fstat(stream.fileno())
            require(length == before.st_size and
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns) ==
                    (before.st_size, before.st_mtime_ns, before.st_ctime_ns),
                    "source changed during content hash")
            count += 1
    require(count > 0, "empty source tree")
    return digest.hexdigest()


def tree_stamp(root: Path) -> str:
    """Independently rederive the qualified source membership/identity stamp."""
    require(root.is_dir() and not root.is_symlink(), "source is not a real directory")
    entries: list[tuple[str, int, int, int, int, int, int]] = []
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted([*dirs, *files]):
            child = Path(current) / name
            item = child.lstat()
            require(stat.S_ISREG(item.st_mode) or stat.S_ISDIR(item.st_mode),
                    "source tree has link or special item")
            entries.append((str(child.relative_to(root)), item.st_mode, item.st_dev,
                            item.st_ino, item.st_size, item.st_mtime_ns,
                            item.st_ctime_ns))
    require(bool(entries), "empty source tree")
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def verify_source(shard: dict[str, Any]) -> dict[str, str]:
    base = Path(shard["base"])
    overlay = Path(shard["overlay"])
    require(tree_stamp(base) == shard["base_storage_stamp"], "base storage stamp changed")
    require(tree_stamp(overlay) == shard["overlay_tree_stamp"], "overlay tree stamp changed")
    require(sha256_file(base / "row_provenance.npz") == shard["provenance_sha256"],
            "base provenance changed")
    declaration = overlay / "target_overlay.json"
    require(sha256_file(declaration) == shard["overlay_declaration_sha256"],
            "Selected-E declaration changed")
    base_content = plain_content_sha256(base)
    require(base_content == shard["base_content_sha256"], "base content changed")
    manifest = json.loads(declaration.read_bytes())
    require(manifest["base"] == str(base) and
            manifest["base_content_sha256"] == base_content and
            manifest["replacements"] == ["policy_target", "search_wdl"],
            "Selected-E overlay base or replacement contract changed")
    local_content = plain_content_sha256(overlay)
    envelope = {"kind": "immutable-policy-overlay-v1", "base": base_content,
                "base_seal": manifest["base_seal"], "local": local_content}
    qualified = hashlib.sha256(json.dumps(envelope, sort_keys=True).encode()).hexdigest()
    require(qualified == shard["qualified_overlay_content_sha256"],
            "Selected-E qualified content changed")
    require(tree_stamp(base) == shard["base_storage_stamp"] and
            tree_stamp(overlay) == shard["overlay_tree_stamp"],
            "source moved during verification")
    return {"base_content_sha256": base_content,
            "qualified_overlay_content_sha256": qualified,
            "base_storage_stamp": shard["base_storage_stamp"],
            "overlay_tree_stamp": shard["overlay_tree_stamp"]}


def read_arrays(path: Path, expected: frozenset[str]) -> dict[str, np.ndarray]:
    require(path.is_dir() and not path.is_symlink(), "native array directory absent or linked")
    group = zarr.open_group(str(path), mode="r")
    keys = set(group.array_keys())
    require(keys == set(expected), f"native field set differs: {path}")
    return {key: np.asarray(group[key][:]) for key in sorted(expected)}


def same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return (left.shape == right.shape and left.dtype.str == right.dtype.str
            and np.ascontiguousarray(left).tobytes() ==
            np.ascontiguousarray(right).tobytes())


def digest_nonmain(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(set(arrays) - {"search_wdl"}):
        value = np.ascontiguousarray(arrays[key])
        shape = json.dumps(list(value.shape), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode()
        digest.update(key.encode() + b"\0")
        digest.update(value.dtype.str.encode() + b"\0")
        digest.update(shape + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def source_identity(rows: np.ndarray, sources: list[dict[str, Any]]) -> str:
    require(rows.dtype.itemsize == 180, "roster row layout differs")
    digest = hashlib.sha256(IDENTITY_DOMAIN)
    for row in rows:
        source_id = int(row["source_id"])
        require(0 <= source_id < len(sources), "roster source ID outside census")
        namespace = bytes.fromhex(sources[source_id]["source_namespace"])
        require(len(namespace) == 32, "source namespace length differs")
        digest.update(namespace)
        digest.update(struct.pack("<iqIIHH", int(row["worker"]), int(row["game"]),
                                  source_id, int(row["source_row"]),
                                  int(row["shard_id"]), int(row["stored_row"])))
        digest.update(np.asarray(row).tobytes())
    return digest.hexdigest()


def selected_roster(roster: np.ndarray, shard_id: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(roster["shard_id"] == shard_id).astype("<i4")
    require(len(indices) > 0, "selected shard has no rows")
    indices = indices[np.argsort(roster["stored_row"][indices], kind="stable")]
    rows = roster[indices]
    positions = rows["stored_row"].astype(np.int64)
    require(np.all(np.diff(positions) > 0) and positions[-1] < 65536,
            "selected stored rows out of order")
    return indices, rows


def expected_pair(base: dict[str, np.ndarray], overlay: dict[str, np.ndarray],
                  stored_rows: np.ndarray, labels: np.ndarray
                  ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    require(set(base) == set(FIELDS) and set(overlay) == {"policy_target", "search_wdl"},
            "qualified physical field set differs")
    require(labels.shape == (len(stored_rows), 3) and labels.dtype.str == "<f4",
            "D8 label vector shape/dtype differs")
    require(np.all(np.isfinite(labels)) and np.all((labels >= 0) & (labels <= 1))
            and np.all(np.abs(labels.sum(axis=1) - 1) <= 1e-5),
            "invalid D8 label WDL")
    require(overlay["policy_target"].dtype.str == "<f2" and
            overlay["search_wdl"].dtype.str == "<f2", "Selected-E target dtype differs")
    require(all(value.shape[0] == base["x"].shape[0] for value in base.values()),
            "base array row count differs")
    require(all(overlay[key].shape == base[key].shape for key in overlay),
            "Selected-E overlay row shape differs")
    require(int(stored_rows[-1]) < base["x"].shape[0], "selected row exceeds base")
    control = {key: np.ascontiguousarray(value[stored_rows]) for key, value in base.items()}
    for key in overlay:
        control[key] = np.ascontiguousarray(overlay[key][stored_rows])
    require(np.all(control["has_policy"] == 1) and
            np.all(control["has_search_wdl"] == 1) and
            np.all(control["has_legal_mask"] == 1), "selected target mask inactive")
    old = control["search_wdl"].astype("<f4")
    require(np.all(np.isfinite(old)) and np.all((old >= 0) & (old <= 1))
            and np.all(np.abs(old.sum(axis=1) - 1) <= 0.005),
            "invalid Selected-E WDL")
    candidate = dict(control)
    candidate["search_wdl"] = ((labels + np.float32(2) * old)
                               / np.float32(3)).astype("<f2")
    return control, candidate


def native_tree(path: Path) -> list[dict[str, Any]]:
    require(path.is_dir() and not path.is_symlink(), "native directory absent or linked")
    entries: list[dict[str, Any]] = []
    for current, dirs, files in os.walk(path):
        dirs.sort()
        for name in dirs:
            child = Path(current) / name
            require(child.is_dir() and not child.is_symlink(), "linked native directory")
        for name in sorted(files):
            child = Path(current) / name
            require(child.is_file() and not child.is_symlink(), "linked or special native file")
            entries.append({"path": child.relative_to(path).as_posix(),
                            "bytes": child.stat().st_size, "sha256": sha256_file(child)})
    entries.sort(key=lambda entry: entry["path"])
    require(bool(entries), "empty native directory")
    return entries


def verify_arm(root: Path, shard_id: int, entry: dict[str, Any],
               wanted: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink() and
            root == root.resolve(strict=True), "arm root is absent or linked")
    for folder in (root / "receipts", root / "roster_index"):
        require(folder.is_dir() and not folder.is_symlink(),
                "arm receipt or sidecar directory is linked")
    name = f"shard_{shard_id:06d}.zarr"
    path = root / name
    require(entry["name"] == name and Path(entry["path"]) == path,
            "arm shard path differs")
    before_stamp = tree_stamp(path)
    actual = read_arrays(path, FIELDS)
    require(all(same_array(actual[key], wanted[key]) for key in FIELDS),
            "native array differs from independently derived source rows")
    index_bytes = indices.astype("<i4", copy=False).tobytes()
    sidecar = root / "roster_index" / f"shard_{shard_id:06d}.i4"
    require(Path(entry["roster_index_path"]) == sidecar and
            sidecar.is_file() and not sidecar.is_symlink() and
            sidecar.read_bytes() == index_bytes, "roster index sidecar differs")
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    require(entry["roster_index_sha256"] == index_sha, "roster index digest differs")
    nonmain = digest_nonmain(actual)
    require(entry["nonmain_arrays_sha256"] == nonmain, "non-main digest differs")
    main_sha = hashlib.sha256(np.ascontiguousarray(actual["search_wdl"]).tobytes()).hexdigest()
    require(entry["search_wdl_sha256"] == main_sha, "main WDL digest differs")
    manifest_path = root / "receipts" / f"shard_{shard_id:06d}.{root.name}.files.json"
    require(Path(entry["file_manifest_path"]) == manifest_path,
            "native file manifest path differs")
    manifest_sha = sha256_file(manifest_path)
    require(entry["file_manifest_sha256"] == manifest_sha, "native file manifest digest differs")
    manifest = json.loads(manifest_path.read_bytes())
    entries = native_tree(path)
    require(manifest == {"schema": "sf_dlite_native_zarr_file_manifest_v1",
                         "arm": root.name, "shard_id": shard_id, "entries": entries},
            "native file tree differs from manifest")
    require(entry["file_count"] == len(entries) and
            entry["compressed_bytes"] == sum(item["bytes"] for item in entries),
            "native file tree byte totals differ")
    require(tree_stamp(path) == before_stamp, "native tree changed during audit")
    return {"roster_index_sha256": index_sha, "nonmain_arrays_sha256": nonmain,
            "search_wdl_sha256": main_sha, "file_manifest_sha256": manifest_sha,
            "file_count": len(entries),
            "compressed_bytes": sum(item["bytes"] for item in entries),
            "native_tree_stamp": before_stamp}


def audit_shard(shard_id: int, census: dict[str, Any], roster: np.ndarray,
                labels: np.ndarray, arms: dict[str, Any]) -> dict[str, Any]:
    """Reopen one complete source shard and both packed arms, then prove bytes."""
    shard = census["shards"][shard_id]
    source = verify_source(shard)
    indices, rows = selected_roster(roster, shard_id)
    stored = rows["stored_row"].astype(np.int64)
    base = read_arrays(Path(shard["base"]), FIELDS)
    overlay = read_arrays(Path(shard["overlay"]), frozenset({"policy_target", "search_wdl"}))
    control, candidate = expected_pair(base, overlay, stored, labels[indices])
    require(same_array(control["game_id"], rows["game"]) and
            same_array(control["ply_index"], rows["game_ply"]),
            "original game or ply differs from source-qualified roster")
    identity = source_identity(rows, census["sources"])
    selected_sha = hashlib.sha256(rows["stored_row"].astype("<u2").tobytes()).hexdigest()
    result: dict[str, Any] = {"schema": "sf_dlite_independent_pair_shard_audit_v1",
                              "shard_id": shard_id, "rows": len(indices),
                              "source_identity_sha256": identity,
                              "selected_stored_rows_sha256": selected_sha,
                              "source_content": source, "arms": {}}
    for arm, wanted in (("control", control), ("candidate", candidate)):
        group = arms[arm]
        entry = group["shards"][shard_id]
        require(entry["rows"] == len(indices) and
                entry["source_identity_sha256"] == identity and
                entry["selected_stored_rows_sha256"] == selected_sha,
                "arm source/row identity differs")
        result["arms"][arm] = verify_arm(Path(group["root"]), shard_id, entry,
                                           wanted, indices)
    require(result["arms"]["control"]["nonmain_arrays_sha256"] ==
            result["arms"]["candidate"]["nonmain_arrays_sha256"],
            "paired non-main arrays differ")
    require(tree_stamp(Path(shard["base"])) == source["base_storage_stamp"] and
            tree_stamp(Path(shard["overlay"])) == source["overlay_tree_stamp"],
            "source changed during shard audit")
    return result
