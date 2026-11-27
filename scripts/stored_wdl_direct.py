"""Target-free descriptive producer-row admission and prediction drift.

No target reconstruction, holdout claim, calibration or strength measurement.
Source-game IDs are producer-declared content hashes, not proven independence.
Run extraction only under the reviewed hard-limited process wrapper.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.shard import load_shard_arrays
from scripts.stored_wdl_compare import _identity, _simplex, _nonempty, rows_digest
from scripts.stored_wdl_rows import _history_metadata, _charge_rows, MAX_CHUNK_BYTES

FIELDS = ("x", "game_id", "has_game_id", "is_selfplay", "has_is_selfplay")


def _bounded_bytes(path: Path, cap: int) -> bytes:
    with path.open("rb") as stream:
        value = stream.read(cap + 1)
    if len(value) > cap:
        raise ValueError("oversized source metadata")
    return value


def read_direct_shard(path: Path, *, expected_attrs_sha256: str, expected_seal_sha256: str,
                      expected_rows: int, expected_history_rep_fix: object) -> dict:
    """Take ALL rows of the predeclared <=256-row pure-selfplay sealed shard."""
    if (type(expected_rows) is not int or not 1 <= expected_rows <= 256 or
            not isinstance(expected_history_rep_fix, bool) or path.suffix != ".zarr"):
        raise ValueError("bounded row count, explicit history bool and Zarr directory required")
    _identity(expected_attrs_sha256)
    _identity(expected_seal_sha256)
    attrs_hash = _history_metadata(path, expected_history_rep_fix)
    if attrs_hash != expected_attrs_sha256:
        raise ValueError("source attributes identity mismatch")
    attrs = json.loads(_bounded_bytes(path / ".zattrs", 65536))
    if (attrs.get("positions") != expected_rows or attrs.get("curriculum_games") != 0 or
            type(attrs.get("selfplay_games")) is not int or attrs["selfplay_games"] < 2):
        raise ValueError("predeclared pure-selfplay shard metadata required")
    seal = _bounded_bytes(path / "directory_producer_seal.json", 4_000_000)
    if hashlib.sha256(seal).hexdigest() != expected_seal_sha256:
        raise ValueError("producer seal identity mismatch")
    metadata = {}
    for name in FIELDS:
        raw = _bounded_bytes(path / name / ".zarray", 65536)
        declaration = json.loads(raw)
        shape, chunks = declaration["shape"], declaration["chunks"]
        dtype = np.dtype(declaration["dtype"])
        if (declaration.get("zarr_format") != 2 or dtype.hasobject or
                not 1 <= len(shape) <= 4 or len(shape) != len(chunks) or
                any(type(c) is not int or c < 1 for c in chunks) or
                math.prod(chunks) * dtype.itemsize > MAX_CHUNK_BYTES):
            raise ValueError("bounded numeric chunk declaration required")
        wanted = [expected_rows, 175, 8, 8] if name == "x" else [expected_rows]
        if shape != wanted:
            raise ValueError("exact predeclared row shapes required")
        metadata[name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                          "shape": shape, "chunks": chunks, "dtype": dtype.str}
    # Guarded lazy loader verifies codecs/directory seal before any decode.
    arrays, _ = load_shard_arrays(path, lazy=True)
    for name, declaration in metadata.items():
        arr = arrays[name]
        if (list(arr.shape) != declaration["shape"] or list(arr.chunks) != declaration["chunks"] or
                np.dtype(arr.dtype).str != declaration["dtype"]):
            raise ValueError("opened array differs from pinned declaration")
    cost = sum(_charge_rows(arrays[name], 0, expected_rows) for name in FIELDS)
    if cost > 64 * 1024**2:
        raise ValueError("direct decoded chunk budget exceeded")
    if np.dtype(arrays["x"].dtype) not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("stored float16/float32 planes required")
    if any(np.dtype(arrays[name].dtype) != np.dtype("uint8") for name in FIELDS[2:]):
        raise ValueError("explicit uint8 identity/selfplay flags required")
    values = {name: np.asarray(arrays[name][:]) for name in FIELDS}
    x = values["x"].astype(np.float32)
    games = values["game_id"]
    if (not np.isfinite(x).all() or not np.issubdtype(games.dtype, np.integer) or
            (games < 0).any() or any(not np.all(values[name] == 1) for name in FIELDS[2:])):
        raise ValueError("finite stored rows, explicit game IDs and pure selfplay flags required")
    if len(set(games.tolist())) < 2:
        raise ValueError("at least two declared source-game groups required")
    if _history_metadata(path, expected_history_rep_fix) != attrs_hash:
        raise ValueError("attributes changed during read")
    if hashlib.sha256(_bounded_bytes(path / "directory_producer_seal.json", 4_000_000)).hexdigest() != expected_seal_sha256:
        raise ValueError("producer seal changed during read")
    for name, declaration in metadata.items():
        if hashlib.sha256(_bounded_bytes(path / name / ".zarray", 65536)).hexdigest() != declaration["sha256"]:
            raise ValueError("array metadata changed during read")
    # Single-artifact scope avoids pretending compaction directories are corpus
    # namespaces. Equal IDs coalesce inside this immutable declared artifact.
    row_ids = [f"{expected_seal_sha256}/row/{i}" for i in range(expected_rows)]
    groups = [f"{expected_seal_sha256}/producer-game/{int(g)}" for g in games]
    consumed = {name: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                for name, a in values.items()}
    return {"mode": "descriptive-producer-rows", "x": x, "row_ids": row_ids, "group_ids": groups,
            "rows_sha256": rows_digest(x, row_ids, groups), "source_shard": str(path.resolve()),
            "attrs_sha256": attrs_hash, "producer_seal_sha256": expected_seal_sha256,
            "array_metadata": metadata, "consumed_values_sha256": consumed,
            "expected_history_rep_fix": expected_history_rep_fix,
            "source_rows_scanned": expected_rows, "decoded_chunk_bytes_charged": cost,
            "training_exposure": "training-ingested producer shard; exact sampled-row exposure unknown",
            "group_kind": "single-artifact supplied producer game hashes",
            "holdout_claim": False, "provenance_authenticated": False,
            "limits": {"rows": 256, "decoded_bytes": 64 * 1024**2, "chunk_bytes": MAX_CHUNK_BYTES}}


def prediction_drift(donor: np.ndarray, candidate: np.ndarray, group_ids: list[str], *,
                     input_rows_sha256: str) -> dict:
    """Changes in predictions, not their correctness; W,D,L stored-row STM POV."""
    _identity(input_rows_sha256)
    groups = tuple(group_ids)
    if not 1 <= len(groups) <= 256 or any(not _nonempty(g) for g in groups):
        raise ValueError("bounded explicit groups required")
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("at least two supplied groups required")
    a, b = (_simplex(p, len(groups), "predictions").copy() for p in (donor, candidate))
    delta = b - a
    grouped = np.stack([delta[np.asarray(groups) == g].mean(axis=0) for g in unique])
    def summary(rows):
        mean = rows.mean(axis=0)
        return {"mean_candidate_minus_donor_WDL": mean.tolist(),
                "mean_expected_score_change": float(mean[0] + .5 * mean[1])}
    return {"mode": "target-free-descriptive-prediction-drift", "input_rows_sha256": input_rows_sha256,
            "rows": len(groups), "supplied_groups": len(unique),
            "row_weighted": summary(delta), "equal_supplied_group_weighted": summary(grouped),
            "max_abs_WDL_change": float(np.abs(delta).max()),
            "donor_predictions_sha256": hashlib.sha256(a.astype("<f8").tobytes()).hexdigest(),
            "candidate_predictions_sha256": hashlib.sha256(b.astype("<f8").tobytes()).hexdigest(),
            "controls_performed": [], "pov": "source-declared stored-row STM W,D,L",
            "limits": "No target loss, calibration, heldout, independence or strength claim; training exposure possible."}
