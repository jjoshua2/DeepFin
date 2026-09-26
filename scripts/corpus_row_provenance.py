"""Source-qualified row references carried alongside derived replay shards.

The original history key hashes the generator's float32 encoding. It cannot be
recovered from float16 replay storage. The second key binds that quantization to
the actual stored row; neither key is a substitute for the other's identity.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from scripts import gen_sf_rooted_corpus as corpus
from scripts.sidecar_cache import RAW_IDENTITY_DTYPE as RECORD_DTYPE

FILENAME = "row_provenance.npz"
SOURCE_FIELDS = ("source_namespace", "source_dir", "source_config_sha256", "source_shard")
SCHEMA = 1


@lru_cache(maxsize=64)
def _namespace(source_dir: Path, config_sha256: str) -> tuple[str, str]:
    resolved = str(source_dir.resolve())
    digest = hashlib.sha256(json.dumps(
        [resolved, config_sha256], separators=(",", ":"),
    ).encode()).hexdigest()
    return digest, resolved


def reference(row: Mapping[str, Any], source_shard: Path, raw_index: int,
              config_sha256: str, x: np.ndarray) -> dict[str, Any]:
    original_key = row.get("input_key")
    if not isinstance(original_key, str) or original_key != corpus.input_tensor_key(x):
        raise ValueError("row provenance requires a verified original history input_key")
    namespace, source_dir = _namespace(source_shard.parent, config_sha256)
    return {
        "schema": SCHEMA,
        "source_namespace": namespace,
        "source_dir": source_dir,
        "source_config_sha256": config_sha256,
        "source_shard": source_shard.name,
        "source_row": int(raw_index),
        "worker_id": int(row["worker_id"]),
        "game_id": int(row["game_id"]),
        "ply": int(row["ply"]),
        "input_key": original_key,
        "stored_input_key": corpus.input_tensor_key(np.asarray(x, dtype=np.float16)),
    }


def write(path: Path, references: Sequence[Mapping[str, Any]],
          x: np.ndarray, game_id: np.ndarray, ply: np.ndarray) -> dict[str, Any]:
    """Validate row alignment before publishing the sidecar within the shard."""
    if len(references) != len(x) or len(game_id) != len(x) or len(ply) != len(x):
        raise ValueError("row provenance length differs from replay rows")
    seen: set[tuple[str, str, int]] = set()
    for i, ref in enumerate(references):
        identity = (str(ref["source_namespace"]), str(ref["source_shard"]), int(ref["source_row"]))
        if identity in seen:
            raise ValueError("duplicate source-qualified physical row in provenance")
        seen.add(identity)
        if ref["schema"] != SCHEMA or int(ref["source_row"]) < 0:
            raise ValueError("invalid row provenance schema or source row")
        if (ref["stored_input_key"] != corpus.input_tensor_key(np.asarray(x[i], dtype=np.float16))
                or int(ref["game_id"]) != int(game_id[i]) or int(ref["ply"]) != int(ply[i])):
            raise ValueError(f"row provenance does not match stored replay row {i}")
    sources: list[dict[str, str]] = []
    source_ids: dict[tuple[str, ...], int] = {}
    records = np.empty(len(references), dtype=RECORD_DTYPE)
    for i, ref in enumerate(references):
        source = tuple(str(ref[name]) for name in SOURCE_FIELDS)
        if source not in source_ids:
            source_ids[source] = len(sources)
            sources.append(dict(zip(SOURCE_FIELDS, source)))
        records[i]["source"] = source_ids[source]
        for field, key in (("row", "source_row"), ("game_id", "game_id"),
                           ("ply", "ply"), ("worker_id", "worker_id")):
            value = int(ref[key])
            info = np.iinfo(RECORD_DTYPE[field])
            if not info.min <= value <= info.max:
                raise ValueError(f"row provenance {field} outside fixed-width range")
            records[i][field] = value
        for key in ("input_key", "stored_input_key"):
            digest = bytes.fromhex(str(ref[key]))
            if len(digest) != 16:
                raise ValueError(f"row provenance {key} is not a 128-bit digest")
            records[i][key] = np.frombuffer(digest, dtype=np.uint8)
    table = json.dumps({"schema": SCHEMA, "sources": sources}, sort_keys=True,
                       separators=(",", ":")).encode()
    with path.open("xb") as handle:
        np.savez(handle, sources=np.frombuffer(table, dtype=np.uint8), records=records)
    return {"schema": SCHEMA, "path": path.name, "rows": len(references),
            "record_bytes": RECORD_DTYPE.itemsize,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path: Path, *, rows: int) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        records = archive["records"]
        table_array = archive["sources"]
    if (records.dtype != RECORD_DTYPE or records.shape != (rows,)
            or table_array.dtype != np.uint8 or table_array.ndim != 1):
        raise ValueError("row provenance count or format differs from replay rows")
    table = json.loads(table_array.tobytes())
    if table["schema"] != SCHEMA:
        raise ValueError("unknown row provenance schema")
    sources = table["sources"]
    refs: list[dict[str, Any]] = []
    for record in records:
        index = int(record["source"])
        if not 0 <= index < len(sources):
            raise ValueError("row provenance source index outside table")
        refs.append({
            "schema": SCHEMA, **sources[index],
            "source_row": int(record["row"]), "game_id": int(record["game_id"]),
            "ply": int(record["ply"]), "worker_id": int(record["worker_id"]),
            "input_key": record["input_key"].tobytes().hex(),
            "stored_input_key": record["stored_input_key"].tobytes().hex(),
        })
    return refs
