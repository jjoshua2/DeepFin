"""Bounded inspected-prefix reconstruction for descriptive stored-WDL work only.

No corpus scan or output publication occurs on import. Callers must execute this
reader in a separately resource-limited process: its deadline is cooperative
between reads, not an interrupt of a filesystem/codec call. Inputs are an explicit
frozen prefix, not a claim to have enumerated the whole corpus. Only guarded Zarr
v2 directory shards are supported. No pickle, overlay or packed-shard fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np

from scripts import match_audit_rows as match
from chess_anti_engine.replay.shard import load_shard_arrays

MAX_AUDIT_BYTES = 8 * 1024**2
MAX_SHARDS = 32
MAX_SCAN_ROWS = 65536
MAX_DECODED_BYTES = 512 * 1024**2
MAX_CHUNK_BYTES = 16 * 1024**2
MAX_SECONDS = 120.0


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_records(audit_bytes: bytes, expected_sha256: str, count: int = 256) -> list[dict]:
    """Freeze keys independently of labels/predictions; never backfill unmatched keys."""
    if len(audit_bytes) > MAX_AUDIT_BYTES or _sha(audit_bytes) != expected_sha256:
        raise ValueError("audit byte identity/size mismatch")
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("selection count must be in 1..256")
    records = [json.loads(line) for line in audit_bytes.decode().splitlines() if line.strip()]
    keys = [r["key"] for r in records]
    if any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("nonempty unique audit keys required")
    indices = sorted(range(len(keys)), key=lambda i: hashlib.sha256(
        b"stored-wdl-descriptive-v1\0" + keys[i].encode()).digest())[:count]
    selected = []
    for i in indices:
        record = dict(records[i])
        board = chess.Board(record["fen"])
        if board.turn != chess.WHITE or match.position_key(board) != record["key"]:
            raise ValueError("canonical audit key/FEN mismatch")
        target = np.asarray(record["wdl"], dtype=np.float64)
        if (target.shape != (3,) or not np.isfinite(target).all() or
                (target < 0).any() or (target > 1).any() or
                not np.isclose(target.sum(), 1, rtol=0, atol=1e-6)):
            raise ValueError("finite simplex WDL required; no normalization/fallback")
        selected.append({"original_index": i, "record": record})
    return selected


def _metadata(path: Path) -> dict[str, dict]:
    """Bound all declared chunks before guarded lazy loading can decode anything."""
    if path.suffix != ".zarr" or not path.is_dir():
        raise ValueError("only Zarr directory shards supported")
    declarations = {}
    for entry in path.iterdir():
        if entry.name.startswith("."):
            continue
        if len(declarations) >= 128:
            raise ValueError("excessive shard array declarations")
        meta_path = entry / ".zarray"
        if not meta_path.is_file() or meta_path.stat().st_size > 65536:
            raise ValueError("unsupported or oversized array metadata")
        raw = meta_path.read_bytes()
        declaration = json.loads(raw)
        dtype = np.dtype(declaration["dtype"])
        shape, chunks = declaration["shape"], declaration["chunks"]
        if (declaration.get("zarr_format") != 2 or dtype.hasobject or
                not 1 <= len(shape) <= 4 or len(shape) != len(chunks) or
                any(type(x) is not int or x < 0 for x in shape) or
                any(type(x) is not int or x < 1 for x in chunks)):
            raise ValueError("unsupported array declaration")
        if math.prod(chunks) * dtype.itemsize > MAX_CHUNK_BYTES:
            raise ValueError("physical decoded chunk exceeds cap")
        declarations[entry.name] = {"raw_sha256": _sha(raw), "shape": tuple(shape),
                                    "chunks": tuple(chunks), "dtype": dtype.str}
    return declarations


def _charge_rows(arr: Any, start: int, stop: int) -> int:
    """Count full physical chunks for this read; repeated reads are charged again."""
    chunks = arr.chunks
    shape = arr.shape
    count = (stop - 1) // chunks[0] - start // chunks[0] + 1
    for extent, chunk in zip(shape[1:], chunks[1:], strict=True):
        count *= (extent + chunk - 1) // chunk
    return count * math.prod(chunks) * np.dtype(arr.dtype).itemsize


def scan_prefix(audit_bytes: bytes, expected_audit_sha256: str,
                shard_paths: tuple[Path, ...], *, count: int = 256) -> dict:
    """Return evidence for the supplied prefix only, without writing an artifact.

    Metadata/codec guards run before array reads. Any limit, invalid identity,
    interruption or canonicalization mismatch raises without returning success.
    The byte cap counts decoded chunks conservatively, not compressed I/O bytes.
    These partial-prefix components never establish original-game independence.
    """
    paths = tuple(Path(p) for p in shard_paths)
    if (not 1 <= len(paths) <= MAX_SHARDS or
            tuple(sorted(paths, key=str)) != paths or
            len({str(p.resolve()) for p in paths}) != len(paths)):
        raise ValueError("unique sorted explicit bounded shard prefix required")
    selected = select_records(audit_bytes, expected_audit_sha256, count)
    if not selected:
        raise ValueError("empty audit selection")
    records = [s["record"] for s in selected]
    boards = [chess.Board(r["fen"]) for r in records]
    keys = [r["key"] for r in records]
    fingerprints: dict[bytes, list[int]] = {}
    for i, board in enumerate(boards):
        fingerprints.setdefault(match.board_fingerprint(board), []).append(i)
    expected = [match.encode_cboard(
        match.CBoard.from_board(b), input_history_encoding=match.STORED_HISTORY_ENCODING,
        input_extra_features=match.STORED_EXTRA_FEATURES).astype(np.float32) for b in boards]
    found = np.zeros(len(records), dtype=bool)
    missing = np.zeros(len(records), dtype=bool)
    candidates: list[set[match.GameKey]] = [set() for _ in records]
    rows: dict[int, np.ndarray] = {}
    origins: dict[int, dict] = {}
    consumed = []
    decoded_bytes = scanned_rows = 0
    started = time.monotonic()
    for path in paths:
        declarations = _metadata(path)
        arrays, _ = load_shard_arrays(path, lazy=True)
        if match._shard_layout(arrays) != (match.STORED_HISTORY_ENCODING, 175):
            raise ValueError("unsupported stored input layout")
        for name in ("x", "game_id", "has_game_id"):
            if name not in arrays or name not in declarations:
                raise ValueError("explicit array-backed source identities required")
            arr, decl = arrays[name], declarations[name]
            if (tuple(arr.shape) != decl["shape"] or tuple(arr.chunks) != decl["chunks"] or
                    np.dtype(arr.dtype).str != decl["dtype"] or
                    _sha((path / name / ".zarray").read_bytes()) != decl["raw_sha256"]):
                raise ValueError("array metadata changed before read")
        x = arrays["x"]
        if x.shape[1:] != (175, 8, 8) or np.dtype(x.dtype) not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError("expected stored float16/float32 N17588")
        n = x.shape[0]
        for name in ("game_id", "has_game_id"):
            if arrays[name].shape != (n,):
                raise ValueError("identity row shape mismatch")
        input_hash = hashlib.sha256()
        for start in range(0, n, x.chunks[0]):
            stop = min(start + x.chunks[0], n)
            cost = sum(_charge_rows(arrays[name], start, stop)
                       for name in ("x", "game_id", "has_game_id"))
            if (decoded_bytes + cost > MAX_DECODED_BYTES or
                    scanned_rows + stop - start > MAX_SCAN_ROWS or
                    time.monotonic() - started > MAX_SECONDS):
                raise ValueError("prefix scan resource budget exhausted")
            decoded_bytes += cost
            scanned_rows += stop - start
            block = np.asarray(x[start:stop], dtype=np.float32)
            games = np.asarray(arrays["game_id"][start:stop])
            present = np.asarray(arrays["has_game_id"][start:stop])
            if (not np.isfinite(block).all() or not np.issubdtype(games.dtype, np.integer) or
                    not np.isin(present, (0, 1)).all()):
                raise ValueError("invalid stored rows or game identity")
            input_hash.update(block.tobytes())
            input_hash.update(games.tobytes())
            input_hash.update(present.tobytes())
            for offset, fingerprint in enumerate(match.shard_fingerprints(block[:, :12])):
                for i in fingerprints.get(fingerprint, ()):
                    row = block[offset]
                    decoded = match.decode_board_from_planes(
                        row, input_history_encoding=match.STORED_HISTORY_ENCODING)
                    if decoded is None or match.position_key(decoded) != keys[i]:
                        continue
                    if (not np.array_equal(row[:12], expected[i][:12]) or
                            not np.array_equal(row[104:108], expected[i][104:108])):
                        raise ValueError("exact current/castling planes mismatch")
                    if {m.uci() for m in decoded.legal_moves} != {m.uci() for m in boards[i].legal_moves}:
                        raise ValueError("audit legal-move mismatch")
                    if not bool(present[offset]) or int(games[offset]) < 0:
                        missing[i] = True
                    else:
                        candidates[i].add(match.source_game_key(path, int(games[offset])))
                    if not found[i]:
                        rows[i] = row.copy()
                        origins[i] = {"shard": str(path.resolve()), "row": start + offset}
                        found[i] = True
        consumed.append({"path": str(path.resolve()), "declarations": declarations,
                         "consumed_values_sha256": input_hash.hexdigest()})
    if time.monotonic() - started > MAX_SECONDS:
        raise ValueError("prefix scan deadline exceeded")
    component, has_component = match.candidate_game_components(
        candidates, candidate_missing_game_id=missing, found=found)
    if np.any(found & ~has_component):
        raise ValueError("observed candidate source identity incomplete")
    indices = np.flatnonzero(found).tolist()
    if len(set(component[found].tolist())) < 2:
        raise ValueError("fewer than two observed prefix components")
    return {"mode": "descriptive-inspected-prefix", "candidate_universe_complete": False,
            "holdout_claim": False, "group_kind": "inspected_prefix_connected_component",
            "audit_sha256": expected_audit_sha256, "selected": selected,
            "matched_selection_indices": indices, "x": np.stack([rows[i] for i in indices]),
            "targets": np.asarray([records[i]["wdl"] for i in indices], dtype=np.float64),
            "row_ids": [keys[i] for i in indices],
            "group_ids": [str(int(component[i])) for i in indices],
            "observed_candidate_games": [sorted(candidates[i]) for i in indices],
            "origins": [origins[i] for i in indices], "consumed_shards": consumed,
            "decoded_chunk_bytes_charged": decoded_bytes, "source_rows_scanned": scanned_rows,
            "limits": {"shards": MAX_SHARDS, "rows": MAX_SCAN_ROWS,
                       "decoded_bytes": MAX_DECODED_BYTES, "chunk_bytes": MAX_CHUNK_BYTES,
                       "cooperative_seconds": MAX_SECONDS},
            "target_semantics": "position-level teacher STM W,D,L; fallback incidence unknown",
            "provenance_authenticated": False}
