#!/usr/bin/env python3
"""Bank d9 Stockfish rank/gap observations beside a derived replay corpus.

The frozen NNUE-bootstrap corpus stores a deliberately cold d9 policy, which
does not retain the centipawn gaps needed to distinguish true near ties from
float16 probability ties. Sources with row provenance join by physical source
rows and verified full-history keys, independently of output shuffle/repacking.
Legacy sources replay the original derivation's prefix,
drop rule, shard boundaries, and within-shard permutation, then writes the
top-ranked compact move indices and their gaps from the best d9 score.  Every
output shard is checked against the derived row's ``(game_id, ply_index)`` and
legal mask before it is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr
from numcodecs import Blosc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.moves.encode import (
    COMPACT_POLICY_SIZE,
    FULL_TO_COMPACT_POLICY,
    uci_to_policy_index,
)
from chess_anti_engine.replay.shard import iter_shard_paths
from scripts import derive_corpus_targets as derive
from scripts import corpus_row_provenance as provenance
from scripts import gen_sf_rooted_corpus as corpus
from scripts.bt4_policy_dump import file_sha256


SCHEMA = 1
SUMMARY_NAME = "sf_d9_rank_sidecar_summary.json"
DERIVE_SUMMARY = "derive_targets_summary.json"
INDEX_FIELD = "sf_d9_rank_index"
GAP_FIELD = "sf_d9_gap_cp"
COUNT_FIELD = "sf_d9_rank_count"
INVALID_INDEX = np.iinfo(np.uint16).max
_COMPRESSOR = Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE)


@dataclass(frozen=True)
class RankObservation:
    game_id: int
    ply: int
    indices: np.ndarray
    gaps_cp: np.ndarray
    count: int


def _sha_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    writing = path.with_name(f".{path.name}.{os.getpid()}.writing")
    try:
        writing.write_text(
            json.dumps(dict(value), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(writing, path)
    finally:
        writing.unlink(missing_ok=True)


def d9_lines(row: Mapping[str, Any]) -> list[list[Any]]:
    """Return the one complete full-width d9 block from a run03-shaped row."""
    phases = row.get("phases")
    if not isinstance(phases, list):
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has malformed phases",
        )
    if not phases or not isinstance(phases[0], dict):
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has no phase-0 search",
        )
    per_depth = phases[0].get("per_depth")
    if not isinstance(per_depth, list):
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has malformed phase-0 depths",
        )
    if not all(isinstance(block, dict) for block in per_depth):
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has malformed phase-0 blocks",
        )
    blocks = cast(list[dict[str, Any]], per_depth)
    matches = [
        block
        for block in blocks
        if int(block.get("depth", -1)) == 9 and bool(block.get("complete"))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has {len(matches)} "
            "complete d9 blocks, expected exactly one",
        )
    lines = matches[0].get("lines")
    if not isinstance(lines, list) or not lines:
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has an empty d9 block",
        )
    if not all(isinstance(line, list) for line in lines):
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has malformed d9 lines",
        )
    return cast(list[list[Any]], lines)


def rank_observation(row: Mapping[str, Any], *, top_k: int) -> RankObservation:
    """Extract compact indices and best-minus-move effective-cp gaps."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    lines = d9_lines(row)
    if any(len(line) < 3 for line in lines):
        raise ValueError("d9 lines must contain rank, move, and effective-cp score")
    ranked = sorted(lines, key=lambda line: int(line[0]))
    ranks = [int(line[0]) for line in ranked]
    if ranks != list(range(1, len(ranked) + 1)):
        raise ValueError(
            f"row {row.get('game_id')}/{row.get('ply')} has malformed d9 ranks",
        )
    scores = np.asarray([float(line[2]) for line in ranked], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("d9 effective-cp scores must be finite")
    gaps = float(scores[0]) - scores
    if bool(np.any(gaps < -1e-6)):
        raise ValueError("d9 rank order disagrees with effective-cp scores")

    indices = np.full((top_k,), INVALID_INDEX, dtype=np.uint16)
    gaps_cp = np.full((top_k,), np.inf, dtype=np.float32)
    stm = row.get("stm")
    if stm not in {"w", "b"}:
        raise ValueError(f"row side-to-move must be 'w' or 'b', got {stm!r}")
    turn = stm == "w"
    count = min(top_k, len(ranked))
    seen: set[int] = set()
    for offset, line in enumerate(ranked[:count]):
        move = str(line[1])
        full_index = uci_to_policy_index(move, turn)
        compact_index = (
            int(FULL_TO_COMPACT_POLICY[full_index]) if full_index >= 0 else -1
        )
        if not 0 <= compact_index < COMPACT_POLICY_SIZE:
            raise ValueError(f"d9 move {move!r} is not compact-policy encodable")
        if compact_index in seen:
            raise ValueError(f"d9 block repeats compact move index {compact_index}")
        seen.add(compact_index)
        indices[offset] = compact_index
        gaps_cp[offset] = max(0.0, float(gaps[offset]))
    return RankObservation(
        game_id=int(row["game_id"]),
        ply=int(row["ply"]),
        indices=indices,
        gaps_cp=gaps_cp,
        count=count,
    )


def _flush(
    *,
    observations: Sequence[RankObservation],
    order: np.ndarray,
    source_path: Path,
    destination: Path,
    top_k: int,
    source_summary_sha256: str,
    raw_config_sha256: str,
) -> dict[str, Any]:
    rows = len(observations)
    if order.shape != (rows,):
        raise ValueError("permutation length does not match observation count")
    game_ids = np.asarray([row.game_id for row in observations], dtype=np.int64)[order]
    plies = np.asarray([row.ply for row in observations], dtype=np.int32)[order]
    indices = np.stack([row.indices for row in observations], axis=0)[order]
    gaps_cp = np.stack([row.gaps_cp for row in observations], axis=0)[order]
    counts = np.asarray([row.count for row in observations], dtype=np.uint8)[order]

    source: Any = zarr.open_group(str(source_path), mode="r")
    expected_game_ids = np.asarray(source["game_id"][:], dtype=np.int64)
    expected_plies = np.asarray(source["ply_index"][:], dtype=np.int32)
    if not np.array_equal(game_ids, expected_game_ids) or not np.array_equal(
        plies,
        expected_plies,
    ):
        raise ValueError(
            f"{source_path}: replayed raw row identity does not match derived order",
        )
    if indices.shape != (rows, top_k) or gaps_cp.shape != (rows, top_k):
        raise ValueError("rank sidecar arrays have the wrong shape")
    valid = np.arange(top_k)[None, :] < counts[:, None]
    if bool(np.any(counts < 1)) or bool(np.any(counts > top_k)):
        raise ValueError("rank counts must be between one and top-k")
    if bool(np.any(indices[~valid] != INVALID_INDEX)) or not bool(
        np.all(np.isinf(gaps_cp[~valid]))
    ):
        raise ValueError("rank padding is malformed")
    if bool(np.any(indices[valid] >= COMPACT_POLICY_SIZE)) or not np.isfinite(
        gaps_cp[valid]
    ).all():
        raise ValueError("valid rank observations are out of range or non-finite")
    if bool(np.any(gaps_cp[valid] < 0.0)):
        raise ValueError("rank gaps must be nonnegative")
    if bool(np.any(gaps_cp[:, 0] != 0.0)):
        raise ValueError("the first-ranked move must have zero cp gap")
    for row_index, count in enumerate(counts.astype(np.int64)):
        if np.unique(indices[row_index, :count]).size != count:
            raise ValueError("valid rank observations repeat a compact move index")
        if bool(np.any(np.diff(gaps_cp[row_index, :count]) < -1e-6)):
            raise ValueError("rank gaps must be nondecreasing")
    legal = np.asarray(source["legal_mask"][:]) != 0
    row_index = np.repeat(np.arange(rows), counts.astype(np.int64))
    compact_index = indices[valid].astype(np.int64)
    if not bool(np.all(legal[row_index, compact_index])):
        raise ValueError(f"{source_path}: ranked d9 sidecar names an illegal move")
    stored_policy = np.asarray(source["policy_target"][:], dtype=np.float32)
    legal_policy = np.where(legal, stored_policy, -np.inf)
    if not bool(
        np.all(
            legal_policy[np.arange(rows), indices[:, 0].astype(np.int64)]
            == np.max(legal_policy, axis=1)
        )
    ):
        raise ValueError(
            f"{source_path}: d9 rank-1 is not a stored-policy maximum",
        )

    chunk_rows = max(1, min(rows, 8192))
    group: Any = zarr.open_group(str(destination), mode="w")
    group.create_dataset(
        INDEX_FIELD,
        data=indices,
        chunks=(chunk_rows, top_k),
        compressor=_COMPRESSOR,
    )
    group.create_dataset(
        GAP_FIELD,
        data=gaps_cp,
        chunks=(chunk_rows, top_k),
        compressor=_COMPRESSOR,
    )
    group.create_dataset(
        COUNT_FIELD,
        data=counts,
        chunks=(chunk_rows,),
        compressor=_COMPRESSOR,
    )
    identity_sha = _sha_arrays(game_ids, plies)
    payload_sha = _sha_arrays(indices, gaps_cp, counts)
    group.attrs.update(
        {
            "sf_d9_rank_sidecar_schema": SCHEMA,
            "source_shard": source_path.name,
            "source_rows": rows,
            "source_row_identity_sha256": identity_sha,
            "source_derive_summary_sha256": source_summary_sha256,
            "raw_config_sha256": raw_config_sha256,
            "depth": 9,
            "top_k": top_k,
            "index_encoding": "lc0_1858",
            "gap_definition": "rank1_effective_cp-minus-ranked_effective_cp",
            "payload_sha256": payload_sha,
        }
    )
    return {
        "path": destination.name,
        "rows": rows,
        "source_row_identity_sha256": identity_sha,
        "payload_sha256": payload_sha,
    }


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _storage_identity(path: Path) -> str:
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        if any((Path(root) / name).is_symlink() for name in [*dirs, *files]):
            raise ValueError("derived storage contains an untracked symlink")
        for name in [".", *sorted(files)]:
            entry = Path(root) if name == "." else Path(root) / name
            stat = entry.lstat()
            digest.update(repr((str(entry.relative_to(path)), stat.st_mode, stat.st_dev,
                                stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)).encode())
    return digest.hexdigest()


def _policy_support_exclusions(
    summary: Mapping[str, Any], *, source_dir: Path, raw_dir: Path, raw_config: str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], Path | None, str | None]:
    """Bind optional drop evidence to the pinned summary, never just a counter."""
    realized = summary["realized"]
    count = realized.get("rows_dropped_policy_support", 0)
    cap = summary.get("max_policy_support_misses", 0)
    entries = realized.get("policy_support_exclusions", [])
    if (type(count) is not int or type(cap) is not int or not 0 <= count <= cap
            or not isinstance(entries, list) or len(entries) != count):
        raise ValueError("invalid policy support exclusion count or evidence")
    if not count:
        if summary.get("policy_support_misses_file") is not None:
            raise ValueError("policy support evidence file declared without exclusions")
        return {}, None, None
    scheme = summary["scheme"]
    if (summary.get("row_provenance") is None or scheme.get("kind") != "uniform"
            or scheme.get("depth") != 9 or scheme.get("policy_observation") != "phase0"
            or summary.get("policy_support_misses_file") != derive.POLICY_SUPPORT_MISSES_FILE):
        raise ValueError("policy support exclusions require explicit phase0 d9 row provenance")
    path = source_dir / derive.POLICY_SUPPORT_MISSES_FILE
    if path.is_symlink() or not path.is_file():
        raise ValueError("policy support exclusion evidence is not a regular local file")
    payload = path.read_bytes()
    actual = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    if actual != entries:
        raise ValueError("policy support exclusion file differs from pinned summary")
    namespace = hashlib.sha256(json.dumps(
        [str(raw_dir), raw_config], separators=(",", ":"),
    ).encode()).hexdigest()
    exclusions: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in entries:
        if (not isinstance(entry, dict) or type(entry.get("schema")) is not int
                or entry.get("schema") != provenance.SCHEMA
                or entry.get("source_namespace") != namespace
                or entry.get("source_dir") != str(raw_dir)
                or entry.get("source_config_sha256") != raw_config
                or not isinstance(entry.get("source_shard"), str)
                or Path(entry["source_shard"]).name != entry["source_shard"]
                or type(entry.get("source_row")) is not int or entry["source_row"] < 0
                or any(type(entry.get(field)) is not int for field in ("worker_id", "game_id", "ply"))
                or entry.get("reason") != "selected_phase0_policy_support"
                or type(entry.get("policy_depth")) is not int or entry.get("policy_depth") != 9
                or entry.get("full_history_input_key_verified") is not True):
            raise ValueError("invalid source-qualified policy support exclusion")
        key = (entry["source_shard"], entry["source_row"])
        if key in exclusions:
            raise ValueError("duplicate policy support exclusion reference")
        exclusions[key] = entry
    return exclusions, path, hashlib.sha256(payload).hexdigest()


def _verify_policy_support_exclusion(
    row: dict[str, Any], evidence: Mapping[str, Any], *, raw_path: Path,
    offset: int, raw_config: str,
) -> None:
    """Recompute the exceptional support and history before omitting its ranks."""
    if row.get("result") is None:
        raise ValueError("policy support exclusion cannot replace a no-result drop")
    if derive.row_schema_of(row) != derive.ROW_SCHEMA_HISTORY:
        raise ValueError("policy support exclusion requires banked full-history input keys")
    derive.require_row_regime(row)
    board = derive.board_from_row(row)
    if (row["stm"] != ("w" if board.turn else "b")
            or int(row["piece_count"]) != board.occupied.bit_count()):
        raise ValueError("policy support exclusion has inconsistent board metadata")
    legal = {move.uci() for move in board.legal_moves}
    phase = row["phases"][0]
    if (type(phase.get("index")) is not int or phase["index"] != 0
            or phase.get("width_requested") != "all" or phase.get("searchmoves") is not None
            or any(type(phase.get(key)) is not int or phase[key] != len(legal)
                   for key in ("width_realized", "width_streamed"))):
        raise ValueError("policy support exclusion has malformed full-width metadata")
    blocks = [block for block in row["phases"][0]["per_depth"] if int(block["depth"]) == 9]
    if len(blocks) != 1 or blocks[0]["complete"] is not True:
        raise ValueError("policy support exclusion has ambiguous or incomplete d9")
    lines = d9_lines(row)
    if len(lines) != len(legal) or any(len(line) != 4 or type(line[0]) is not int or line[0] != rank
           or not isinstance(line[1], str) or isinstance(line[2], bool)
           or not isinstance(line[2], (int, float)) or not math.isfinite(line[2])
           for rank, line in enumerate(lines, 1)):
        raise ValueError("policy support exclusion has malformed ranks or scores")
    moves = [line[1] for line in lines]
    missing = sorted(legal - set(moves))
    duplicates = sorted(move for move in set(moves) if moves.count(move) > 1)
    if set(moves) - legal or not (missing or duplicates):
        raise ValueError("policy support exclusion does not describe an eligible support defect")
    x = np.asarray(encode_position(
        board, add_features=True, input_history_encoding=derive.INPUT_HISTORY_ENCODING,
        input_extra_features=derive.INPUT_EXTRA_FEATURES,
    ), dtype=np.float32)
    expected = {
        **provenance.reference(row, raw_path, offset, raw_config, x),
        "reason": "selected_phase0_policy_support", "policy_depth": 9,
        "missing_moves": missing, "duplicate_moves": duplicates,
        "full_history_input_key_verified": True,
    }
    if expected != evidence:
        raise ValueError("policy support exclusion differs from raw support or history identity")


def _bank_provenance(
    *, record: derive.CorpusRecord, raw_dir: Path, source_paths: list[Path],
    source_summary: Mapping[str, Any], source_summary_sha: str, writing: Path,
    limit: int, top_k: int, max_cache_bytes: int,
    exclusions: Mapping[tuple[str, int], dict[str, Any]], exclusion_path: Path | None,
) -> tuple[list[dict[str, Any]], int, int, dict[str, Any]]:
    """Read raw rows once; join recorded physical rows even across output revisits."""
    dtype = np.dtype([
        ("game_id", "<i8"), ("ply", "<i4"), ("worker_id", "<i4"),
        ("input_key", "u1", (16,)), ("stored_input_key", "u1", (16,)),
        ("indices", "<u2", (top_k,)), ("gaps", "<f4", (top_k,)),
        ("count", "u1"), ("valid", "u1"),
    ])
    corpus.apply_history_rep_fix()
    counts = derive.shard_row_counts(record)
    required = limit * (dtype.itemsize + 1) + len(counts) * 512
    if max_cache_bytes <= 0 or required > max_cache_bytes:
        raise ValueError(f"rank/history cache needs up to {required} bytes, cap {max_cache_bytes}")
    cache = writing / "._rank_identity_cache"
    cache.mkdir()
    raw_config = str(record.facts["config_sha256"])
    raw_rows = dropped = 0
    index: dict[str, tuple[Path, Path, str, int]] = {}
    identities: dict[Path, tuple[int, int, int, int, int]] = {}
    seen_exclusions: set[tuple[str, int]] = set()
    if exclusion_path is not None:
        identities[exclusion_path] = _file_identity(exclusion_path)
    exclusion_sha = file_sha256(exclusion_path) if exclusion_path is not None else None
    for number, (raw_path, count) in enumerate(zip(record.shards, counts)):
        take = min(int(count), limit - raw_rows)
        if take <= 0:
            break
        if take * dtype.itemsize > 64 * 1024 ** 2:
            raise ValueError("raw shard exceeds the 64MiB rank-cache working-set limit")
        identities[raw_path] = _file_identity(raw_path)
        records = np.zeros(take, dtype=dtype)
        seen = 0
        for offset, row in enumerate(derive.iter_corpus_rows(raw_path)):
            if offset >= take:
                break
            seen += 1
            raw_rows += 1
            derive._check_row_identity(row, raw_config)
            key = (raw_path.name, offset)
            if key in exclusions:
                _verify_policy_support_exclusion(
                    row, exclusions[key], raw_path=raw_path, offset=offset, raw_config=raw_config,
                )
                seen_exclusions.add(key)
                continue
            if row.get("result") is None:
                dropped += 1
                continue
            # A dropped envelope row need not have d9. A selected row must.
            if derive.RowBank(row).full_width_block(9) is None:
                continue
            observation = rank_observation(row, top_k=top_k)
            derive.require_row_regime(row)
            board = derive.board_from_row(row)
            x = np.asarray(encode_position(
                board, add_features=True,
                input_history_encoding=derive.INPUT_HISTORY_ENCODING,
                input_extra_features=derive.INPUT_EXTRA_FEATURES,
            ), dtype=np.float32)
            ref = provenance.reference(row, raw_path, offset, raw_config, x)
            item = records[offset]
            for field in ("game_id", "ply", "worker_id"):
                item[field] = ref[field]
            for field in ("input_key", "stored_input_key"):
                item[field] = np.frombuffer(bytes.fromhex(ref[field]), dtype=np.uint8)
            item["indices"], item["gaps"] = observation.indices, observation.gaps_cp
            item["count"], item["valid"] = observation.count, 1
        if seen != take or _file_identity(raw_path) != identities[raw_path]:
            raise ValueError("raw source count or storage changed while banking ranks")
        target = cache / f"{number:06d}.npy"
        with target.open("xb") as handle:
            np.save(handle, records, allow_pickle=False)
        used_path = cache / f"{number:06d}.used.npy"
        used = np.lib.format.open_memmap(used_path, mode="w+", dtype=np.uint8, shape=(take,))
        used[:] = 0
        used.flush()
        del used
        index[raw_path.name] = (target, used_path, file_sha256(target), take)
    if raw_rows != limit:
        raise ValueError(f"raw prefix has {raw_rows} rows, expected {limit}")
    if seen_exclusions != set(exclusions):
        raise ValueError("policy support exclusion is outside the consumed raw prefix")
    written = []
    derived_stable: dict[Path, str] = {}
    shard_manifest = {entry["path"]: entry for entry in source_summary["shards"]}
    if set(shard_manifest) != {path.name for path in source_paths}:
        raise ValueError("derived summary shard inventory differs from actual shards")
    expected_namespace = hashlib.sha256(json.dumps(
        [str(raw_dir), raw_config], separators=(",", ":"),
    ).encode()).hexdigest()
    for path in source_paths:
        derived_stable[path] = _storage_identity(path)
        source: Any = zarr.open_group(str(path), mode="r")
        x = np.asarray(source["x"][:])
        rows = len(x)
        stamp = dict(source.attrs).get("derive_row_provenance")
        if (not isinstance(stamp, dict) or stamp != shard_manifest[path.name].get("row_provenance")
                or stamp.get("schema") != provenance.SCHEMA or stamp.get("rows") != rows
                or stamp.get("record_bytes") != provenance.RECORD_DTYPE.itemsize
                or stamp.get("path") != provenance.FILENAME):
            raise ValueError("derived row provenance stamp mismatch")
        ref_path = path / provenance.FILENAME
        if file_sha256(ref_path) != stamp["sha256"]:
            raise ValueError("derived row provenance checksum mismatch")
        identities[ref_path] = _file_identity(ref_path)
        identities[path / ".zattrs"] = _file_identity(path / ".zattrs")
        refs = provenance.read(ref_path, rows=rows)
        ordered: list[RankObservation | None] = [None] * rows
        requests: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for offset, ref in enumerate(refs):
            if (ref["source_dir"] != str(raw_dir) or ref["source_config_sha256"] != raw_config
                    or ref["source_shard"] not in index):
                raise ValueError("rank provenance names another raw source")
            requests.setdefault(ref["source_shard"], []).append((offset, ref))
        for raw_name, group in requests.items():
            target, used_path, expected_hash, count = index[raw_name]
            if file_sha256(target) != expected_hash:
                raise ValueError("private rank cache changed")
            records = np.load(target, mmap_mode="r", allow_pickle=False)
            used = np.load(used_path, mmap_mode="r+", allow_pickle=False)
            for offset, ref in group:
                raw_index = int(ref["source_row"])
                if (raw_name, raw_index) in exclusions:
                    raise ValueError("derived row references an excluded policy support row")
                if not 0 <= raw_index < count or used[raw_index]:
                    raise ValueError("duplicate or out-of-prefix physical row reference")
                item = records[raw_index]
                if (not item["valid"] or ref["source_namespace"] != expected_namespace
                        or any(int(ref[field]) != int(item[field]) for field in ("game_id", "ply", "worker_id"))
                        or any(ref[field] != item[field].tobytes().hex() for field in ("input_key", "stored_input_key"))
                        or ref["stored_input_key"] != corpus.input_tensor_key(x[offset])):
                    raise ValueError("raw/derived full-history row identity mismatch")
                used[raw_index] = 1
                ordered[offset] = RankObservation(
                    int(item["game_id"]), int(item["ply"]),
                    np.array(item["indices"]), np.array(item["gaps"]), int(item["count"]),
                )
            used.flush()
            del used, records
        if any(item is None for item in ordered):
            raise ValueError("rank references did not cover every derived row")
        written.append(_flush(
            observations=cast(list[RankObservation], ordered), order=np.arange(rows),
            source_path=path, destination=writing / path.name, top_k=top_k,
            source_summary_sha256=source_summary_sha, raw_config_sha256=raw_config,
        ))
    if (any(_file_identity(path) != identity for path, identity in identities.items())
            or any(_storage_identity(path) != identity for path, identity in derived_stable.items())):
        raise ValueError("raw or derived storage changed during rank publication")
    raw_members = set(record.shards)
    proof = {
        "join": "source-qualified-physical-row-and-full-history-keys-v1",
        "observation": "complete-phase0-d9", "raw_shards_read_once": len(index),
        "cache_record_bytes": dtype.itemsize, "cache_seen_bytes_per_row": 1,
        "cache_budget_bytes": max_cache_bytes,
        "raw_source_metadata": {str(path): list(identity) for path, identity in identities.items()
                                if path in raw_members},
        "policy_observation": source_summary["scheme"].get("policy_observation", "latest-phase"),
        "value_observation": source_summary["scheme"].get("value_observation", "latest-phase"),
        **({"rows_dropped_policy_support": len(seen_exclusions),
            "policy_support_exclusions_sha256": exclusion_sha}
           if exclusion_path is not None else {}),
    }
    return written, raw_rows, dropped, proof


def bank(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw).resolve()
    source_dir = Path(args.shards).resolve()
    out_dir = Path(args.out).resolve()
    limit = int(args.limit)
    top_k = int(args.top_k)
    seed = int(args.seed)
    expected_rows = int(args.expected_rows)
    expected_shards = int(args.expected_shards)
    expected_summary_sha = str(args.expected_source_summary_sha256)
    if limit <= 0 or top_k <= 0 or top_k > 255:
        raise SystemExit("--limit and --top-k must be positive; top-k must be <=255")
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; rank sidecars are immutable")
    writing = out_dir.with_name(out_dir.name + ".writing")
    if writing.exists():
        raise SystemExit(f"stale partial output exists: {writing}")

    source_summary_path = source_dir / DERIVE_SUMMARY
    source_summary_sha = file_sha256(source_summary_path)
    if source_summary_sha != expected_summary_sha:
        raise SystemExit(
            f"source summary SHA-256 mismatch: {source_summary_sha} != "
            f"{expected_summary_sha}",
        )
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    realized = source_summary.get("realized", {})
    expected_contract = {
        "limit_requested": limit,
        "seed": seed,
        "rows_per_shard": int(args.rows_per_shard),
    }
    bad_contract = {
        key: (source_summary.get(key), value)
        for key, value in expected_contract.items()
        if source_summary.get(key) != value
    }
    if int(realized.get("rows_written", -1)) != expected_rows:
        bad_contract["realized.rows_written"] = (
            realized.get("rows_written"),
            expected_rows,
        )
    if bad_contract:
        raise SystemExit(f"derived source contract mismatch: {bad_contract}")
    source_paths = iter_shard_paths(source_dir)
    if len(source_paths) != expected_shards:
        raise SystemExit(
            f"derived source has {len(source_paths)} shards, expected {expected_shards}",
        )

    raw_record_pins = {
        path: file_sha256(path) for path in (raw_dir / "manifest.json", raw_dir / "summary.json")
        if path.is_file()
    }
    record = derive.read_corpus_record(raw_dir)
    selection_path = getattr(args, "source_shards", None)
    if selection_path is not None:
        record = derive.select_corpus_record(raw_dir, record, Path(selection_path))
    if record.source_selection != source_summary.get("source_selection"):
        raise ValueError("raw and derived source selections differ; pass the same --source-shards")
    raw_config_sha = str(record.facts.get("config_sha256", ""))
    source_corpus = source_summary.get("corpus", {})
    if not isinstance(source_corpus, dict) or source_corpus.get(
        "config_sha256"
    ) != raw_config_sha:
        raise SystemExit("raw and derived source config identities differ")

    exclusions, exclusion_path, exclusion_sha = _policy_support_exclusions(
        source_summary, source_dir=source_dir, raw_dir=raw_dir, raw_config=raw_config_sha,
    )
    if exclusion_path is not None:
        raw_record_pins[exclusion_path] = cast(str, exclusion_sha)

    writing.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    pending: list[RankObservation] = []
    written: list[dict[str, Any]] = []
    raw_rows = 0
    dropped_no_result = 0
    started = time.time()
    try:
        provenance_proof: dict[str, Any] = {}
        if source_summary.get("row_provenance") is not None:
            written, raw_rows, dropped_no_result, provenance_proof = _bank_provenance(
                record=record, raw_dir=raw_dir, source_paths=source_paths,
                source_summary=source_summary, source_summary_sha=source_summary_sha,
                writing=writing, limit=limit, top_k=top_k,
                max_cache_bytes=int(getattr(args, "max_provenance_cache_bytes", 8 * 1024 ** 3)),
                exclusions=exclusions, exclusion_path=exclusion_path,
            )
        else:
            for raw_path in record.shards:
                for row in derive.iter_corpus_rows(raw_path):
                    if raw_rows >= limit:
                        break
                    raw_rows += 1
                    if row.get("result") is None:
                        dropped_no_result += 1
                        continue
                    pending.append(rank_observation(row, top_k=top_k))
                    if len(pending) == int(args.rows_per_shard):
                        index = len(written)
                        written.append(
                            _flush(
                                observations=pending,
                                order=rng.permutation(len(pending)),
                                source_path=source_paths[index],
                                destination=writing / source_paths[index].name,
                                top_k=top_k,
                                source_summary_sha256=source_summary_sha,
                                raw_config_sha256=raw_config_sha,
                            )
                        )
                        pending = []
                        elapsed = max(time.time() - started, 1e-9)
                        if len(written) % 16 == 0:
                            print(
                                f"[sf-d9-ranks] {len(written)}/{expected_shards} shards, "
                                f"{sum(item['rows'] for item in written)} rows, "
                                f"{raw_rows / elapsed:.1f} raw rows/s",
                                flush=True,
                            )
                if raw_rows >= limit:
                    break
            if pending:
                index = len(written)
                written.append(
                    _flush(
                        observations=pending,
                        order=rng.permutation(len(pending)),
                        source_path=source_paths[index],
                        destination=writing / source_paths[index].name,
                        top_k=top_k,
                        source_summary_sha256=source_summary_sha,
                        raw_config_sha256=raw_config_sha,
                    )
                )
        rows = sum(int(item["rows"]) for item in written)
        expected_dropped = int(realized.get("rows_dropped_no_result", -1))
        if (
            raw_rows != limit
            or rows != expected_rows
            or len(written) != expected_shards
            or dropped_no_result != expected_dropped
        ):
            raise ValueError(
                "rank sidecar cardinality mismatch: "
                f"raw={raw_rows}/{limit}, rows={rows}/{expected_rows}, "
                f"shards={len(written)}/{expected_shards}, "
                f"drops={dropped_no_result}/{expected_dropped}",
            )
        if provenance_proof:
            if raw_rows - rows - dropped_no_result - len(exclusions) != int(
                realized.get("rows_dropped_envelope", 0)
            ):
                raise ValueError("provenance-selected row count differs from declared drop counters")
            if (file_sha256(source_summary_path) != source_summary_sha
                    or any(file_sha256(path) != digest for path, digest in raw_record_pins.items())):
                raise ValueError("source summary or raw manifest changed during rank publication")
            provenance_proof["raw_record_sha256"] = {str(path): digest for path, digest in raw_record_pins.items()}
            shutil.rmtree(writing / "._rank_identity_cache")
        derive.verify_source_selection(record)
        summary = {
            **({"source_selection": record.source_selection} if record.source_selection is not None else {}),
            "schema": SCHEMA,
            "kind": "sf_d9_rank_gap_sidecar",
            "raw_dir": str(raw_dir),
            "raw_config_sha256": raw_config_sha,
            "raw_limit": limit,
            "raw_rows_read": raw_rows,
            "rows_dropped_no_result": dropped_no_result,
            **({"rows_dropped_policy_support": len(exclusions)}
               if source_summary.get("max_policy_support_misses", 0) else {}),
            "source_dir": str(source_dir),
            "source_derive_summary_sha256": source_summary_sha,
            "rows": rows,
            "shards": len(written),
            "rows_per_shard": int(args.rows_per_shard),
            "seed": seed,
            "depth": 9,
            "top_k": top_k,
            "index_encoding": "lc0_1858",
            "gap_definition": "rank1_effective_cp-minus-ranked_effective_cp",
            "outputs": written,
            **({"row_provenance": provenance_proof} if provenance_proof else {}),
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_json(writing / SUMMARY_NAME, summary)
        os.replace(writing, out_dir)
    except BaseException:
        print(
            f"[sf-d9-ranks] FAILED; preserving partial output at {writing}",
            file=sys.stderr,
        )
        raise
    print(
        f"[sf-d9-ranks] complete: {rows} rows / {len(written)} shards -> {out_dir}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--source-shards", type=Path,
                        help="same source-bound selection used by derivation; --limit cuts its ordered stream")
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-provenance-cache-bytes", type=int, default=8 * 1024 ** 3,
                        help="private temporary rank/history index cap when source has row provenance")
    parser.add_argument("--rows-per-shard", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-source-summary-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return bank(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
