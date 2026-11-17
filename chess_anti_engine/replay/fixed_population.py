"""One immutable, caller-admitted population; no FIFO or sampling policy.

Source identity is the original generation plus globally unique shard-inventory
SHA and local row. Slots are ordering indices, never invented generations.
Metadata validation is not issuer authentication or permission to train.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, cast
import uuid

SCHEMA = "FIXED_SELFPLAY_UNION_MANIFEST_V1"
MAX_DECODED_BYTES = 32 << 30
SOURCE_KEYS = {"slot", "generation", "source_sha256", "rows", "game_id",
               "physical_game_id", "bank", "label_class", "shard", "origin_id", "producer_proof"}
ORIGIN_KEYS = {"origin_id", "generation_manifest", "identity", "acceptance",
               "producer_protocol", "actor_serving", "served_state_sha256"}
IDENTITY_KEYS = {"run_uuid", "collection_uuid", "generation", "source_revision",
                 "actor_sha256", "actor_checkpoint_step", "architecture"}


def require(ok: Any, message: str) -> None:
    if not ok:
        raise ValueError(message)


def integer(value: Any, name: str, minimum: int = 1) -> int:
    require(type(value) is int and value >= minimum, "invalid " + name)
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(value: Any) -> str:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid source digest")
    return value


def reference(value: Any) -> dict[str, str]:
    require(type(value) is dict and set(value) == {"path", "sha256"}, "invalid reference")
    value = cast(dict[str, Any], value)
    require(type(value["path"]) is str, "invalid reference path")
    p = Path(cast(str, value["path"]))
    require(p.is_absolute() and str(p) == value["path"] and ".." not in p.parts,
            "noncanonical absolute reference")
    sha(value["sha256"])
    return value


def inventory(shard: Any) -> str:
    """Validate the recorded descriptor, without statting any payload path."""
    require(type(shard) is dict and set(shard) == {"path", "files"}, "invalid shard descriptor")
    shard = cast(dict[str, Any], shard)
    p = Path(cast(str, shard["path"]))
    require(p.is_absolute() and str(p) == shard["path"] and ".." not in p.parts,
            "noncanonical shard path")
    require(type(shard["files"]) is list and bool(shard["files"]), "empty shard inventory")
    paths = []
    for f in cast(list[Any], shard["files"]):
        require(type(f) is dict and set(f) == {"path", "bytes", "sha256"}, "invalid inventory file")
        f = cast(dict[str, Any], f)
        path = Path(cast(str, f["path"]))
        require(not path.is_absolute() and str(path) == f["path"] and ".." not in path.parts,
                "noncanonical inventory entry")
        integer(f["bytes"], "inventory bytes", 0)
        sha(f["sha256"])
        paths.append(f["path"])
    require(paths == sorted(set(paths)) and ("." not in paths or paths == ["."]),
            "duplicate or unsorted inventory entries")
    return digest(shard["files"])


def origin_map(value: dict[str, Any]) -> dict[str, Any]:
    return {"origins": value["origins"], "aliases": value["aliases"],
            "source_slots": [{k: s[k] for k in ("slot", "generation", "source_sha256", "origin_id")}
                             for s in value["sources"]]}


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    require(type(value) is dict and set(value) == {
        "schema", "plan", "architecture", "expected_sources", "expected_rows", "sources",
        "origins", "aliases", "origin_map_sha256", "training_authorized"}, "invalid union manifest")
    require(value["schema"] == SCHEMA and value["training_authorized"] is False,
            "union metadata is not training authority")
    reference(value["plan"])
    require(type(value["architecture"]) is dict and bool(value["architecture"]), "missing architecture")
    integer(value["expected_sources"], "expected source count")
    integer(value["expected_rows"], "expected row count")
    require(type(value["origins"]) is list and bool(value["origins"]), "missing original origins")
    origins: dict[str, dict[str, Any]] = {}
    for o in cast(list[Any], value["origins"]):
        require(type(o) is dict and set(o) == ORIGIN_KEYS, "invalid origin record")
        o = cast(dict[str, Any], o)
        payload = {k: v for k, v in o.items() if k != "origin_id"}
        require(o["origin_id"] == digest(payload) and o["origin_id"] not in origins, "origin digest/duplicate")
        reference(o["generation_manifest"])
        reference(o["producer_protocol"])
        reference(o["actor_serving"])
        sha(o["served_state_sha256"])
        i = o["identity"]
        require(type(i) is dict and set(i) == IDENTITY_KEYS and cast(dict[str, Any], i)["architecture"] == value["architecture"],
                "origin architecture or identity differs")
        i = cast(dict[str, Any], i)
        integer(i["generation"], "actual generation")
        integer(i["actor_checkpoint_step"], "actor step", 0)
        sha(i["actor_sha256"])
        require(type(i["source_revision"]) is str and re.fullmatch(r"[0-9a-f]{40}", i["source_revision"])
                is not None, "invalid original source revision")
        for key in ("run_uuid", "collection_uuid"):
            require(type(i[key]) is str and str(uuid.UUID(cast(str, i[key]))) == i[key], "invalid original UUID")
        a = o["acceptance"]
        require(type(a) is dict and a.get("kind") in {"legacy", "bank"}, "invalid origin acceptance")
        a = cast(dict[str, Any], a)
        if a["kind"] == "legacy":
            require(set(a) == {"kind", "manifest"} and a["manifest"] == o["generation_manifest"],
                    "legacy caller-trusted pin differs")
        else:
            require(set(a) == {"kind", "root_acceptance", "independent_review"}, "invalid bank acceptance")
            reference(a["root_acceptance"])
            reference(a["independent_review"])
        origins[o["origin_id"]] = o
    require(type(value["sources"]) is list and len(value["sources"]) == value["expected_sources"],
            "source roster length differs")
    seen: set[str] = set()
    rows = 0
    physical: set[tuple[str, str]] = set()
    for slot, s in enumerate(cast(list[Any], value["sources"])):
        require(type(s) is dict and set(s) == SOURCE_KEYS, "invalid canonical source")
        s = cast(dict[str, Any], s)
        require(type(s["slot"]) is int and s["slot"] == slot, "noncanonical union slot")
        sha(s["source_sha256"])
        require(s["source_sha256"] not in seen, "duplicate canonical source SHA")
        seen.add(s["source_sha256"])
        rows += integer(s["rows"], "source rows")
        integer(s["game_id"], "physical game ID", 0)
        o = origins.get(s["origin_id"])
        require(o is not None, "source has no original origin")
        assert o is not None
        require(type(s["generation"]) is int and s["generation"] == o["identity"]["generation"],
                "actual generation differs from origin")
        require(type(s["physical_game_id"]) is str and bool(s["physical_game_id"]), "missing physical game namespace")
        key = (o["origin_id"], s["physical_game_id"])
        require(key not in physical, "duplicate original physical game")
        physical.add(key)
        require(s["bank"] == "train" and s["label_class"] in {"NATURAL", "TB_ADJUDICATED"},
                "protected or unresolved source refused")
        require(inventory(s["shard"]) == s["source_sha256"], "source inventory digest differs")
        reference(s["producer_proof"])
    require(rows == value["expected_rows"], "union row total differs")
    require(type(value["aliases"]) is list, "invalid copied-asset aliases")
    for alias in cast(list[Any], value["aliases"]):
        require(type(alias) is dict and set(alias) == {"canonical_slot", "origin_id", "shard", "producer_proof"},
                "invalid copied-asset alias")
        alias = cast(dict[str, Any], alias)
        slot = integer(alias["canonical_slot"], "alias canonical slot", 0)
        require(slot < len(value["sources"]) and alias["origin_id"] in origins, "unknown alias source/origin")
        s = cast(dict[str, Any], value["sources"][slot])
        require(alias["shard"]["files"] == s["shard"]["files"]
                and inventory(alias["shard"]) == s["source_sha256"], "alias inventory differs")
        reference(alias["producer_proof"])
    require(value["origin_map_sha256"] == digest(origin_map(value)), "origin map digest differs")
    return value


def _header(path: Path, expected_sha: str, cap: int = 1 << 20) -> dict[str, Any]:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_mode & 0o222 == 0 and before.st_size <= cap,
            "invalid readonly shard declaration")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        raw = os.read(fd, before.st_size + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    def stamp(s: os.stat_result) -> tuple[int, ...]:
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)

    require(len(raw) == before.st_size and stamp(before) == stamp(after) == stamp(path.lstat())
            and hashlib.sha256(raw).hexdigest() == expected_sha,
            "shard declaration changed")
    value = json.loads(raw)
    require(type(value) is dict, "invalid shard declaration object")
    return value


def declared_decoded_bound(shard: dict[str, Any], rows: int, planes: int) -> int:
    """Inspect only inventory-named public Zarr declarations before eager decode.

    Charge padded storage chunks as well as logical array sizes. The eager
    loader may attach small identity arrays; reserve explicit extra space.
    Other layouts need a separately reviewed bounded decoder, never a fallback.
    """
    import numpy as np

    root = Path(shard["path"])
    require(any(f["path"] == ".zgroup" for f in shard["files"]), "fixed union requires public Zarr layout")
    total = 65536 + rows * 64
    saw_x = False
    for f in shard["files"]:
        if Path(f["path"]).name != ".zarray":
            continue
        path = root / f["path"]
        d = _header(path, f["sha256"])
        shape, chunks = d["shape"], d["chunks"]
        require(type(shape) is list and all(type(n) is int and n >= 0 for n in shape), "invalid declared shape")
        require(type(chunks) is list and len(chunks) == len(shape)
                and all(type(n) is int and n > 0 for n in chunks), "invalid declared chunks")
        dtype = np.dtype(d["dtype"])
        require(not dtype.hasobject, "object declaration refused")
        logical = padded = 1
        for n, width in zip(shape, chunks, strict=True):
            logical *= n
            padded *= ((n + width - 1) // width) * width
            require(max(logical, padded) * dtype.itemsize <= MAX_DECODED_BYTES, "oversized declaration")
        total += max(logical, padded) * dtype.itemsize
        require(total <= MAX_DECODED_BYTES, "oversized declared source")
        if path.parent.name == "x":
            require(shape == [rows, planes, 8, 8] and dtype == np.dtype("float16"), "declared input shape/precision")
            saw_x = True
    require(saw_x, "missing input declaration")
    return total


def declared_memory_budget(shard: dict[str, Any], rows: int, planes: int) -> dict[str, int]:
    """Pre-decode worst case, independent of optional flags or observed NNZ.

    Zarr returns logical arrays; storage chunk padding is one-source workspace.
    Policy values preserve dtype; columns and lengths are uint16. Legal masks
    become uint16 indices and lengths. Dense sf_p0_regret is not sparsified.
    """
    import numpy as np
    from chess_anti_engine.replay.shard import LEGAL_MASK_FIELDS, POLICY_SPACE_FIELDS

    root = Path(shard["path"])
    require(any(f["path"] == ".zgroup" for f in shard["files"]), "fixed union requires public Zarr layout")
    logical = padded = extra = sparse_elements = 0
    saw_x = False
    for f in shard["files"]:
        if Path(f["path"]).name != ".zarray":
            continue
        relative = Path(f["path"])
        require(len(relative.parts) == 2, "fixed union requires flat public array declarations")
        d = _header(root / relative, f["sha256"])
        shape, chunks = d["shape"], d["chunks"]
        require(type(shape) is list and all(type(n) is int and n >= 0 for n in shape), "invalid declared shape")
        require(type(chunks) is list and len(chunks) == len(shape)
                and all(type(n) is int and n > 0 for n in chunks), "invalid declared chunks")
        dtype = np.dtype(d["dtype"])
        require(not dtype.hasobject, "object declaration refused")
        elements = padded_elements = 1
        for n, width in zip(shape, chunks, strict=True):
            elements *= n
            padded_elements *= ((n + width - 1) // width) * width
            require(max(elements, padded_elements) * dtype.itemsize <= MAX_DECODED_BYTES, "oversized declaration")
        logical += elements * dtype.itemsize
        padded += padded_elements * dtype.itemsize
        require(padded + 65536 + rows * 64 <= MAX_DECODED_BYTES, "oversized declared source")
        name = relative.parts[0]
        if name == "x":
            require(shape == [rows, planes, 8, 8] and dtype == np.dtype("float16"), "declared input shape/precision")
            saw_x = True
        if name in (*POLICY_SPACE_FIELDS, *LEGAL_MASK_FIELDS):
            require(len(shape) == 2 and shape[0] == rows and 0 < shape[1] <= 65535,
                    "invalid sparse declaration shape")
            if name in POLICY_SPACE_FIELDS:
                require(dtype.kind == "f", "invalid sparse policy dtype")
                # The values replace dense values. Full-width uint16 columns
                # add2*elements; uint16 NNZ adds2*rows even for zero policies.
                extra += 2 * elements + 2 * rows
            else:
                require(dtype.kind in "bui", "invalid sparse legal dtype")
                extra += max(0, 2 - dtype.itemsize) * elements + 2 * rows
            sparse_elements += elements
    require(saw_x, "missing input declaration")
    overhead = 65536 + rows * 64  # attached markers, missing priority, small row arrays
    retained = logical + extra + overhead
    encoded = sum(f["bytes"] for f in shard["files"])
    # Eight logical sources cover the eager source, validation copies/casts,
    # sparse output coexistence and transient metadata arrays. The additional
    # 64bytes per sparse element dominates bool masks, int64 nonzero/position
    # vectors, selected values and padded output construction. Charge a whole
    # encoded source and two padded sources for codec/chunk decode workspace.
    workspace = 8 * logical + 64 * sparse_elements + 2 * padded + encoded + 65536 + rows * 1024
    return {"rows": rows, "logical_bytes": logical, "padded_bytes": padded,
            "decoded_bytes": padded + overhead, "retained_bytes": retained,
            "workspace_bytes": workspace,
            "gather_row_bytes": (logical + extra + rows * 64 + rows - 1) // rows}


def population_memory_budget(budgets: list[dict[str, int]], rows: int, manifest_bytes: int) -> dict[str, int]:
    """Single population plus one source workspace; not a process RSS claim.

    Headroom explicitly reserves parsed/canonical metadata, per-source Python
    bookkeeping and row-order objects. Source-local buffers are charged even
    when optional fields will be pruned. No sparsity or allocator reuse assumed.
    """
    require(bool(budgets) and sum(b["rows"] for b in budgets) == rows, "memory budget source rows differ")
    retained = sum(b["retained_bytes"] for b in budgets)
    workspace = max(b["workspace_bytes"] for b in budgets)
    vectors = 32 * rows  # both old/new f4 priority+i1 WDL concatenations and indexing
    gather = 8 * min(32, rows) * max(b["gather_row_bytes"] for b in budgets)
    metadata = 16 * manifest_bytes + 65536 * len(budgets) + 256 * rows
    return {"retained_bytes": retained, "workspace_bytes": workspace, "vectors_bytes": vectors,
            "gather_bytes": gather, "metadata_headroom_bytes": metadata,
            "reserve_bytes": retained + workspace + vectors + gather + metadata}


def resident_array_bytes(arrays: dict[str, Any]) -> int:
    """Charge owning allocations, not just a view's logical nbytes."""
    import numpy as np

    seen: set[int] = set()
    total = 0
    for value in arrays.values():
        require(isinstance(value, np.ndarray) and not value.dtype.hasobject, "non-array/object resident storage")
        owner = value
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        require(owner.base is None, "resident array has unbounded external backing")
        if id(owner) not in seen:
            total += int(owner.nbytes)
            seen.add(id(owner))
    return total


def _checked_arrays(architecture: dict[str, Any], source: dict[str, Any], arrays: dict[str, Any]) -> None:
    """Same core.load_replay shard predicates, without its FIFO assumptions."""
    import numpy as np
    from chess_anti_engine.model import infer_input_planes
    from chess_anti_engine.replay.shard import history_rep_fix_from_arrays

    rows = source["rows"]
    require(arrays["x"].shape == (rows, infer_input_planes(architecture["input_extra_features"]), 8, 8),
            "loaded input shape mismatch")
    require(arrays["x"].dtype == np.float16 and arrays["policy_target"].dtype == np.float16,
            "loaded input/target precision mismatch")
    for key in ("has_policy", "has_legal_mask", "is_network_turn", "has_is_network_turn",
                "is_selfplay", "has_is_selfplay", "has_game_id", "has_ply_index"):
        require(key in arrays and np.all(arrays[key] == 1), "missing active TRAIN identity: " + key)
    require(not np.any(arrays["policy_target"][arrays["legal_mask"] == 0] != 0), "illegal policy target mass")
    require(all(np.issubdtype(arrays[key].dtype, np.integer) for key in ("game_id", "ply_index", "wdl_target")),
            "loaded physical/result indices are not integers")
    require(np.all(arrays["game_id"] == source["game_id"]), "loaded game ID mismatch")
    require(len(np.unique(arrays["ply_index"])) == rows, "duplicate physical rows")
    require(str(np.asarray(arrays["_input_history_encoding"]).item()) == architecture["input_history_encoding"],
            "loaded input history mismatch")
    require(str(np.asarray(arrays["_policy_encoding"]).item()) == architecture["policy_encoding"],
            "loaded policy encoding mismatch")
    require(history_rep_fix_from_arrays(arrays) == architecture["history_rep_fix"], "loaded repetition encoding mismatch")


class FixedPopulation:
    """Read-only identity surface for StoredRowOnceReplay, no replacement API."""

    def __init__(self, buffer: Any, rng: Any, manifest: dict[str, Any], manifest_ref: dict[str, str]):
        self._buffer = buffer
        self.rng = rng
        self.membership = tuple((s["generation"], s["source_sha256"], s["rows"]) for s in manifest["sources"])
        self._origin_json = canonical(origin_map(manifest))
        self._identity_json = canonical(manifest_identity(manifest, manifest_ref))
        require(len(buffer) == manifest["expected_rows"], "fixed assembly lost rows")

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(self._identity_json)

    @property
    def origin_map(self) -> dict[str, Any]:
        return json.loads(self._origin_json)

    def check_identity(self, expected: dict[str, Any]) -> None:
        require(canonical(expected) == self._identity_json, "fixed population checkpoint identity differs")


def manifest_identity(manifest: dict[str, Any], manifest_ref: dict[str, str]) -> dict[str, Any]:
    """Pure closed-consumer/checkpoint binding; no payload/stat/import activity."""
    validate_manifest(manifest)
    reference(manifest_ref)
    require(hashlib.sha256((canonical(manifest) + "\n").encode()).hexdigest() == manifest_ref["sha256"],
            "manifest must retain exact compiler canonical bytes")
    membership = [(s["generation"], s["source_sha256"], s["rows"]) for s in manifest["sources"]]
    return {"manifest": dict(manifest_ref), "origin_map_sha256": manifest["origin_map_sha256"],
            "membership_sha256": digest(membership), "sources": len(membership), "rows": manifest["expected_rows"]}


def load_fixed_population(architecture: dict[str, Any], manifest: dict[str, Any], manifest_ref: dict[str, str], *,
                          rng: Any, max_rows: int, max_decoded_bytes: int) -> FixedPopulation:
    """Decode once in an admitted owner; never authenticate authority by status.

    Caller must bind the hash-authenticated manifest and its plan/acceptance.
    All declared shards are budgeted before the first eager load. The reserve
    covers worst retained sparse storage, one-source decode/validation/sparse
    workspace, vectors, B32 gathers and explicit metadata headroom. It is not
    a process RSS/model/allocator claim. ArrayReplayBuffer is created once, appended once per source, never
    reassembled or evicted. Its small priority vectors are unused by the quota
    adapter; this population exposes no legacy sampling method.
    """
    from chess_anti_engine.model import infer_input_planes
    from chess_anti_engine.replay.buffer import ArrayReplayBuffer
    from chess_anti_engine.replay.shard import load_shard_arrays
    from scripts import core_finite_learner as core

    validate_manifest(manifest)
    manifest_identity(manifest, manifest_ref)
    integer(max_rows, "row limit")
    require(integer(max_decoded_bytes, "decoded limit") <= MAX_DECODED_BYTES, "decoded cap exceeds32GiB")
    require(manifest["architecture"] == architecture and manifest["expected_rows"] <= max_rows,
            "population architecture/row cap differs")
    planes = infer_input_planes(architecture["input_extra_features"])
    budgets = []
    for source in manifest["sources"]:
        path, actual = core.verify_inventory(source["shard"])
        require(actual == source["source_sha256"], "source inventory changed before decode")
        budgets.append(declared_memory_budget(source["shard"], source["rows"], planes))
        require(path == Path(source["shard"]["path"]), "source path differs")
    sizing = population_memory_budget(budgets, manifest["expected_rows"], len(canonical(manifest).encode()) + 1)
    require(sizing["reserve_bytes"] <= max_decoded_bytes, "global decoded reserve exceeded before eager decode")
    buffer = ArrayReplayBuffer(manifest["expected_rows"], rng=rng)
    retained = 0
    retained_limit = 0
    assembled_rows = 0
    assembled_sources = 0
    for source, budget in zip(manifest["sources"], budgets, strict=True):
        path, before = core.verify_inventory(source["shard"])
        require(before == source["source_sha256"], "source changed before eager load")
        arrays, _ = load_shard_arrays(path, lazy=False, validate=True)
        actual_bytes = resident_array_bytes(arrays)
        require(actual_bytes <= budget["decoded_bytes"], "decoded source exceeds pre-reserved declaration")
        _checked_arrays(architecture, source, arrays)
        require(core.verify_inventory(source["shard"])[1] == before, "source changed during eager load")
        for a in arrays.values():
            a.setflags(write=False)
        buffer.add_many_arrays(arrays)
        assembled_sources += 1
        require(len(buffer._chunks) == len(buffer._chunk_sizes) == assembled_sources,
                "buffer chunk bookkeeping differs")
        added = resident_array_bytes(buffer._chunks[-1])
        require(added <= budget["retained_bytes"], "sparse source exceeds pre-reserved retained storage")
        retained += added
        retained_limit += budget["retained_bytes"]
        assembled_rows += source["rows"]
        require(retained <= retained_limit <= sizing["retained_bytes"] and len(buffer) == assembled_rows
                and len(buffer) <= manifest["expected_rows"], "global assembly budget/row cap")
        require(buffer._priority.nbytes + buffer._wdl.nbytes <= sizing["vectors_bytes"],
                "global buffer vectors exceed pre-reserved storage")
        del arrays
    return FixedPopulation(buffer, rng, manifest, manifest_ref)
