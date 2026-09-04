#!/usr/bin/env python3
"""Stream one-evaluation BT4 priors beside closed raw bootstrap shards.

The raw Stockfish corpus is intentionally append-only while its generators are
live.  This tool snapshots only shards listed by the workers' progress files,
reconstructs and verifies every schema-3 history input, performs one batched
BT4 forward evaluation per row, and publishes one immutable Zarr sidecar per
raw shard.  It never reads the unlisted shard each worker may still be writing.

Run this command repeatedly.  ``--max-shards`` bounds one GPU lease, allowing
arenas or trainers using the same advisory lock to take over between groups.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import sys
import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np
import zarr
from numcodecs import Blosc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.encoding.lc0 import x_to_lc0_planes
from chess_anti_engine.eval.rvg_surgery import FINGERPRINT_BYTES, position_fingerprints
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE, POLICY_ENCODING_LC0_1858
from chess_anti_engine.moves.leela_index import compact_index_for_move
from scripts import derive_corpus_targets as derive
from scripts import gen_sf_rooted_corpus as corpus
from scripts.bt4_policy_dump import (
    DEFAULT_ONNX,
    file_sha256,
    legal_move_policy,
    open_session,
    remap_provenance,
    resolve_policy_output,
)
from scripts.bt4_policy_mix import functional_remap_identity


SCHEMA = 1
PROGRESS_NAME = "bt4_raw_sidecar.progress.jsonl"
STATUS_NAME = "bt4_raw_sidecar.status.json"
VERIFY_NAME = "bt4_raw_sidecar.verify.json"
SOURCE_KEY_FIELD = "source_key"
INPUT_KEY_FIELD = "input_key"
POLICY_FIELD = "bt4_policy"
GAME_ID_FIELD = "game_id"
PLY_FIELD = "ply"
_SOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHARD_SUFFIXES = (".jsonl.zst", ".jsonl.gz")
_COMPRESSOR = Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    corpus_dir: Path
    out_dir: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    inventory: derive.ProgressInventory
    corpus_complete: bool = False


@dataclass(frozen=True)
class PendingShard:
    source: SourceSpec
    path: Path
    claimed_rows: int
    target: Path


def parse_source_arg(raw: str) -> tuple[str, Path]:
    """Parse ``ID=PATH`` without permitting an output-path escape."""
    source_id, sep, path = str(raw).partition("=")
    if not sep or not _SOURCE_RE.fullmatch(source_id):
        raise ValueError(
            f"--source must be ID=PATH with a safe non-empty ID, got {raw!r}",
        )
    if not path:
        raise ValueError(f"--source {source_id!r} has an empty path")
    return source_id, Path(path).resolve()


def sidecar_name(source_name: str) -> str:
    for suffix in _SHARD_SUFFIXES:
        if source_name.endswith(suffix):
            return source_name[: -len(suffix)] + ".bt4.zarr"
    raise ValueError(f"raw source shard has an unsupported name: {source_name!r}")


def source_name_from_sidecar(name: str) -> str | None:
    if not name.endswith(".bt4.zarr"):
        return None
    return name[: -len(".bt4.zarr")] + ".jsonl.zst"


def sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    writing = path.with_name(f".{path.name}.{os.getpid()}.writing")
    try:
        writing.write_text(
            json.dumps(dict(value), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(writing, path)
    finally:
        writing.unlink(missing_ok=True)


def append_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Durably append one compact receipt after its sidecar is published."""
    payload = (json.dumps(dict(receipt), sort_keys=True) + "\n").encode("utf-8")
    trim_torn_progress_tail(path)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError(f"short progress append: {written}/{len(payload)} bytes")
        os.fsync(fd)
    finally:
        os.close(fd)


def trim_torn_progress_tail(path: Path) -> int:
    """Remove only a non-newline-terminated final progress fragment."""
    if not path.exists():
        return 0
    with path.open("rb+") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size == 0:
            return 0
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return 0
        chunk_size = 1 << 16
        position = size
        while position > 0:
            step = min(chunk_size, position)
            position -= step
            handle.seek(position)
            data = handle.read(step)
            newline = data.rfind(b"\n")
            if newline >= 0:
                keep = position + newline + 1
                handle.truncate(keep)
                handle.flush()
                os.fsync(handle.fileno())
                return size - keep
        handle.truncate(0)
        handle.flush()
        os.fsync(handle.fileno())
        return size


def read_receipts(path: Path) -> dict[str, dict[str, Any]]:
    """Read complete progress lines; tolerate only one torn final line."""
    if not path.exists():
        return {}
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    receipts: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(lines):
        is_last = index == len(lines) - 1
        if is_last and not raw.endswith(b"\n"):
            break
        try:
            receipt = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{index + 1}: damaged progress record") from exc
        source_shard = str(receipt.get("source_shard", ""))
        if not source_shard or source_shard in receipts:
            raise ValueError(
                f"{path}:{index + 1}: missing or duplicate source_shard "
                f"{source_shard!r}",
            )
        receipts[source_shard] = dict(receipt)
    return receipts


def load_sources(
    raw_sources: Sequence[str], out_root: Path,
) -> list[SourceSpec]:
    seen: set[str] = set()
    specs: list[SourceSpec] = []
    for raw in raw_sources:
        source_id, corpus_dir = parse_source_arg(raw)
        if source_id in seen:
            raise ValueError(f"duplicate --source ID {source_id!r}")
        seen.add(source_id)
        if not corpus_dir.is_dir():
            raise ValueError(f"source corpus is not a directory: {corpus_dir}")
        out_dir = (out_root / source_id).resolve()
        if corpus_dir == out_dir or corpus_dir in out_dir.parents:
            raise ValueError(f"output for {source_id!r} must be outside its source")
        manifest_path = corpus_dir / corpus.MANIFEST_NAME
        manifest = corpus.read_launch_manifest(corpus_dir)
        if int(manifest.get("row_schema", -1)) != corpus.ROW_SCHEMA:
            raise ValueError(
                f"{source_id}: row_schema {manifest.get('row_schema')!r} is not "
                f"the history-and-keys schema {corpus.ROW_SCHEMA}",
            )
        if manifest.get(corpus.KEY_HISTORY_REP_FIX) is not corpus.HISTORY_REP_FIX:
            raise ValueError(f"{source_id}: history repetition regime mismatch")
        record = derive.read_corpus_record(corpus_dir)
        if str(record.facts.get("config_sha256")) != str(manifest["config_sha256"]):
            raise ValueError(f"{source_id}: manifest/corpus-record config mismatch")
        inventory = derive.ProgressInventory(
            shards=record.shards,
            shard_rows=record.shard_rows,
            rows_claimed=record.rows_claimed,
            progress_files=record.progress_files,
            torn_tail_files=record.torn_tail_files,
            unlisted_on_disk=record.unlisted_on_disk,
        )
        specs.append(
            SourceSpec(
                source_id=source_id,
                corpus_dir=corpus_dir,
                out_dir=out_dir,
                manifest=manifest,
                manifest_sha256=file_sha256(manifest_path),
                inventory=inventory,
                corpus_complete=record.corpus_complete,
            )
        )
    return specs


def expected_existing_attrs(pending: PendingShard) -> dict[str, Any]:
    stat = pending.path.stat()
    return {
        "bt4_raw_sidecar_schema": SCHEMA,
        "source_id": pending.source.source_id,
        "source_dir": str(pending.source.corpus_dir),
        "source_shard": pending.path.name,
        "source_rows_claimed": pending.claimed_rows,
        "source_file_bytes": int(stat.st_size),
        "source_manifest_sha256": pending.source.manifest_sha256,
        "source_config_sha256": str(pending.source.manifest["config_sha256"]),
        "input_history_encoding": derive.INPUT_HISTORY_ENCODING,
        "input_extra_features": derive.INPUT_EXTRA_FEATURES,
        "history_rep_fix": corpus.HISTORY_REP_FIX,
        "policy_encoding": POLICY_ENCODING_LC0_1858,
        "policy_size": COMPACT_POLICY_SIZE,
        "teacher_evaluations_per_position": 1,
        "search_nodes": 0,
        "stored_dtype": "float32",
    }


def validate_existing(
    pending: PendingShard,
    *,
    onnx_sha256: str | None = None,
    policy_output: str | None = None,
    providers: Sequence[str] | None = None,
    functional_remap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not pending.target.is_dir():
        raise ValueError(f"missing sidecar directory {pending.target}")
    group: Any = zarr.open_group(str(pending.target), mode="r")
    attrs = dict(group.attrs)
    expected = expected_existing_attrs(pending)
    if onnx_sha256 is not None:
        expected["onnx_sha256"] = onnx_sha256
    if policy_output is not None:
        expected["policy_output"] = policy_output
    if providers is not None:
        expected["providers"] = list(providers)
    bad = {
        key: (attrs.get(key), value)
        for key, value in expected.items()
        if attrs.get(key) != value
    }
    if bad:
        raise ValueError(f"{pending.target}: provenance mismatch {bad}")
    if functional_remap is not None and functional_remap_identity(
        attrs.get("remap_provenance"),
    ) != dict(functional_remap):
        raise ValueError(f"{pending.target}: functional policy remap mismatch")
    rows = pending.claimed_rows
    layouts = {
        SOURCE_KEY_FIELD: ((rows, FINGERPRINT_BYTES), np.dtype(np.uint8)),
        INPUT_KEY_FIELD: ((rows, FINGERPRINT_BYTES), np.dtype(np.uint8)),
        POLICY_FIELD: ((rows, COMPACT_POLICY_SIZE), np.dtype(np.float32)),
        GAME_ID_FIELD: ((rows,), np.dtype(np.int64)),
        PLY_FIELD: ((rows,), np.dtype(np.int32)),
    }
    for name, (shape, dtype) in layouts.items():
        if name not in group:
            raise ValueError(f"{pending.target}: missing array {name!r}")
        array = group[name]
        if tuple(array.shape) != shape or np.dtype(array.dtype) != dtype:
            raise ValueError(
                f"{pending.target}: {name} is {array.shape}/{array.dtype}, "
                f"expected {shape}/{dtype}",
            )
    return attrs


def pending_shards(
    sources: Sequence[SourceSpec],
    *,
    onnx_sha256: str,
    policy_output: str | None = None,
    providers: Sequence[str] | None = None,
    functional_remap: Mapping[str, Any] | None = None,
) -> tuple[list[PendingShard], dict[str, int]]:
    todo: list[PendingShard] = []
    complete_by_source: dict[str, int] = {}
    for source in sources:
        source.out_dir.mkdir(parents=True, exist_ok=True)
        progress_path = source.out_dir / PROGRESS_NAME
        receipts = read_receipts(progress_path)
        by_name = dict(zip(
            (path.name for path in source.inventory.shards),
            zip(source.inventory.shards, source.inventory.shard_rows, strict=True),
            strict=True,
        ))
        unknown_receipts = sorted(set(receipts) - set(by_name))
        if unknown_receipts:
            raise ValueError(
                f"{progress_path}: receipts name shards outside the current "
                f"closed inventory: {unknown_receipts[:8]}",
            )

        target_dirs = {
            path.name: path
            for path in source.out_dir.iterdir()
            if path.is_dir() and path.name.endswith(".bt4.zarr")
        }
        expected_targets = {sidecar_name(name): name for name in by_name}
        if len(expected_targets) != len(by_name):
            raise ValueError(
                f"{source.source_id}: two source names collapse to one sidecar name",
            )
        foreign_targets = sorted(set(target_dirs) - set(expected_targets))
        if foreign_targets:
            raise ValueError(
                f"{source.out_dir}: sidecars outside the current closed inventory: "
                f"{foreign_targets[:8]}",
            )
        foreign_partials = sorted(
            path.name
            for path in source.out_dir.glob("*.bt4.zarr.writing")
            if path.name[: -len(".writing")] not in expected_targets
        )
        if foreign_partials:
            raise ValueError(
                f"{source.out_dir}: partial sidecars outside the current closed "
                f"inventory: {foreign_partials[:8]}",
            )

        complete = 0
        for source_name, (path, claimed_rows) in by_name.items():
            target = source.out_dir / sidecar_name(source_name)
            item = PendingShard(source, path, int(claimed_rows), target)
            writing = target.with_name(target.name + ".writing")
            if writing.exists():
                raise ValueError(f"stale partial sidecar exists: {writing}")
            receipt = receipts.get(source_name)
            if receipt is not None:
                if not target.exists():
                    raise ValueError(f"{progress_path} lists missing {target.name}")
                attrs = validate_existing(
                    item,
                    onnx_sha256=onnx_sha256,
                    policy_output=policy_output,
                    providers=providers,
                    functional_remap=functional_remap,
                )
                expected_receipt = receipt_from_attrs(attrs, target)
                bad_receipt = {
                    key: (receipt.get(key), value)
                    for key, value in expected_receipt.items()
                    if key != "published_unix" and receipt.get(key) != value
                }
                if bad_receipt:
                    raise ValueError(
                        f"{target}: progress/sidecar receipt mismatch {bad_receipt}",
                    )
                complete += 1
                continue
            if target.exists():
                attrs = validate_existing(
                    item,
                    onnx_sha256=onnx_sha256,
                    policy_output=policy_output,
                    providers=providers,
                    functional_remap=functional_remap,
                )
                append_receipt(progress_path, receipt_from_attrs(attrs, target))
                complete += 1
                continue
            todo.append(item)
        complete_by_source[source.source_id] = complete
    return todo, complete_by_source


def receipt_from_attrs(attrs: Mapping[str, Any], target: Path) -> dict[str, Any]:
    keys = (
        "source_id",
        "source_shard",
        "positions",
        "source_sha256",
        "source_key_sha256",
        "input_key_sha256",
        "bt4_policy_sha256",
        "onnx_sha256",
        "policy_output",
        "providers",
        "remap_provenance",
        "teacher_evaluations_per_position",
    )
    receipt = {key: attrs[key] for key in keys}
    receipt["sidecar"] = target.name
    receipt["published_unix"] = float(attrs["published_unix"])
    return receipt


def encode_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: SourceSpec,
) -> tuple[np.ndarray, list[chess.Board], np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct, encode, and independently key one raw-row batch."""
    planes: list[np.ndarray] = []
    boards: list[chess.Board] = []
    input_keys = np.empty((len(rows), FINGERPRINT_BYTES), dtype=np.uint8)
    game_ids = np.empty((len(rows),), dtype=np.int64)
    plies = np.empty((len(rows),), dtype=np.int32)
    config_sha = str(source.manifest["config_sha256"])
    for offset, row in enumerate(rows):
        derive.require_row_regime(row)
        if derive.row_schema_of(row) != corpus.ROW_SCHEMA:
            raise ValueError(f"{source.source_id}: row is not schema {corpus.ROW_SCHEMA}")
        run = row.get("run")
        if not isinstance(run, Mapping) or str(run.get("config_sha256")) != config_sha:
            raise ValueError(f"{source.source_id}: row config_sha256 mismatch")
        board = derive.board_from_row(row)
        stm = "w" if board.turn == chess.WHITE else "b"
        if stm != str(row.get("stm")):
            raise ValueError(f"{source.source_id}: row stm disagrees with reconstructed board")
        if chess.popcount(board.occupied) != int(row.get("piece_count", -1)):
            raise ValueError(
                f"{source.source_id}: row piece_count disagrees with reconstructed board",
            )
        encoded = np.asarray(
            encode_position(
                board,
                add_features=True,
                input_history_encoding=derive.INPUT_HISTORY_ENCODING,
                input_extra_features=derive.INPUT_EXTRA_FEATURES,
            ),
            dtype=np.float32,
        )
        got_input_key = corpus.input_tensor_key(encoded)
        want_input_key = str(row.get("input_key", ""))
        if got_input_key != want_input_key:
            raise ValueError(
                f"{source.source_id}:{row.get('game_id')}/{row.get('ply')}: "
                f"encoded input key {got_input_key} != banked {want_input_key}",
            )
        try:
            key_bytes = bytes.fromhex(want_input_key)
        except ValueError as exc:
            raise ValueError(f"invalid input_key {want_input_key!r}") from exc
        if len(key_bytes) != FINGERPRINT_BYTES:
            raise ValueError(f"input_key {want_input_key!r} is not 128 bits")
        input_keys[offset] = np.frombuffer(key_bytes, dtype=np.uint8)
        game_ids[offset] = int(row["game_id"])
        plies[offset] = int(row["ply"])
        planes.append(encoded)
        boards.append(board)
    stacked = np.stack(planes) if planes else np.zeros(
        (0, 175, 8, 8), dtype=np.float32,
    )
    return stacked, boards, input_keys, game_ids, plies


def label_shard(
    pending: PendingShard,
    *,
    sess: Any,
    input_name: str,
    input_dtype: np.dtype[Any],
    providers: Sequence[str],
    policy_name: str,
    onnx_path: Path,
    onnx_sha256: str,
    remap_stamp: Mapping[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """Label one closed source shard and atomically publish its sidecar."""
    if pending.target.exists():
        raise FileExistsError(f"refusing to overwrite {pending.target}")
    writing = pending.target.with_name(pending.target.name + ".writing")
    if writing.exists():
        raise FileExistsError(f"stale partial sidecar exists: {writing}")

    rows_expected = int(pending.claimed_rows)
    bt4_policy = np.zeros((rows_expected, COMPACT_POLICY_SIZE), dtype=np.float32)
    source_keys = np.empty((rows_expected, FINGERPRINT_BYTES), dtype=np.uint8)
    input_keys = np.empty_like(source_keys)
    game_ids = np.empty((rows_expected,), dtype=np.int64)
    plies = np.empty((rows_expected,), dtype=np.int32)
    entropy_sum = 0.0
    top1_sum = 0.0
    legal_moves_sum = 0
    cursor = 0
    batch_rows: list[Mapping[str, Any]] = []

    def evaluate_batch(rows: Sequence[Mapping[str, Any]]) -> None:
        nonlocal cursor, entropy_sum, top1_sum, legal_moves_sum
        if not rows:
            return
        stop = cursor + len(rows)
        if stop > rows_expected:
            raise ValueError(
                f"{pending.path}: decoded more than {rows_expected} claimed rows",
            )
        planes, boards, raw_keys, gids, batch_plies = encode_rows(
            rows,
            source=pending.source,
        )
        fingerprints = position_fingerprints(
            planes,
            input_history_encoding=derive.INPUT_HISTORY_ENCODING,
        )
        if len(fingerprints) != len(rows) or any(
            len(key) != FINGERPRINT_BYTES for key in fingerprints
        ):
            raise ValueError(f"{pending.path}: invalid derived-row fingerprints")
        source_keys[cursor:stop] = np.frombuffer(
            b"".join(fingerprints), dtype=np.uint8,
        ).reshape(len(rows), FINGERPRINT_BYTES)
        input_keys[cursor:stop] = raw_keys
        game_ids[cursor:stop] = gids
        plies[cursor:stop] = batch_plies

        feats = x_to_lc0_planes(
            planes,
            input_history_encoding=derive.INPUT_HISTORY_ENCODING,
        ).astype(input_dtype, copy=False)
        output = np.asarray(
            sess.run([policy_name], {input_name: feats})[0],
            dtype=np.float32,
        )
        if output.shape[0] != len(rows):
            raise ValueError(
                f"{pending.path}: BT4 returned {output.shape[0]} rows for {len(rows)} inputs",
            )
        for offset, board in enumerate(boards):
            row_index = cursor + offset
            ucis, probs_raw = legal_move_policy(board, output[offset])
            probs = np.asarray(probs_raw, dtype=np.float32)
            indices = np.asarray(
                [
                    compact_index_for_move(board, chess.Move.from_uci(uci))
                    for uci in ucis
                ],
                dtype=np.int64,
            )
            expected = {
                compact_index_for_move(board, move) for move in board.legal_moves
            }
            if (
                len(indices) != len(expected)
                or len(set(indices.tolist())) != len(indices)
                or set(indices.tolist()) != expected
                or bool(np.any(indices < 0))
                or bool(np.any(indices >= COMPACT_POLICY_SIZE))
            ):
                raise ValueError(f"{pending.path}:{row_index}: legal policy mapping mismatch")
            if (
                probs.shape != (len(indices),)
                or not np.isfinite(probs).all()
                or bool(np.any(probs < 0.0))
                or not np.isclose(float(probs.sum()), 1.0, atol=2e-6)
            ):
                raise ValueError(f"{pending.path}:{row_index}: invalid BT4 legal policy")
            bt4_policy[row_index, indices] = probs
            positive = probs > 0.0
            entropy_sum += float(-np.sum(probs[positive] * np.log(probs[positive])))
            top1_sum += float(probs.max())
            legal_moves_sum += len(indices)
        cursor = stop

    for row in derive.iter_corpus_rows(pending.path):
        batch_rows.append(row)
        if len(batch_rows) >= batch_size:
            evaluate_batch(batch_rows)
            batch_rows = []
    evaluate_batch(batch_rows)
    if cursor != rows_expected:
        raise ValueError(
            f"{pending.path}: decoded {cursor} rows, progress claims {rows_expected}",
        )
    row_sums = bt4_policy.sum(axis=1, dtype=np.float64)
    if not np.isfinite(bt4_policy).all() or bool(np.any(bt4_policy < 0.0)):
        raise ValueError(f"{pending.path}: generated policy is negative or non-finite")
    if not bool(np.allclose(row_sums, 1.0, atol=2e-6)):
        raise ValueError(f"{pending.path}: generated policy is not row-normalized")

    source_stat = pending.path.stat()
    source_sha = file_sha256(pending.path)
    attrs = {
        **expected_existing_attrs(pending),
        "positions": rows_expected,
        "source_sha256": source_sha,
        "source_key_sha256": sha_array(source_keys),
        "input_key_sha256": sha_array(input_keys),
        "game_id_sha256": sha_array(game_ids),
        "ply_sha256": sha_array(plies),
        "bt4_policy_sha256": sha_array(bt4_policy),
        "onnx_path": str(onnx_path.resolve()),
        "onnx_sha256": onnx_sha256,
        "policy_output": policy_name,
        "providers": list(providers),
        "remap_provenance": dict(remap_stamp),
        "bt4_entropy_sum": entropy_sum,
        "bt4_top1_sum": top1_sum,
        "legal_moves_sum": legal_moves_sum,
        "source_file_mtime_ns": int(source_stat.st_mtime_ns),
        "published_unix": time.time(),
    }
    group: Any = zarr.open_group(str(writing), mode="w")
    row_chunk = min(512, max(1, rows_expected))
    group.create_dataset(
        SOURCE_KEY_FIELD,
        data=source_keys,
        chunks=(row_chunk, FINGERPRINT_BYTES),
        compressor=_COMPRESSOR,
    )
    group.create_dataset(
        INPUT_KEY_FIELD,
        data=input_keys,
        chunks=(row_chunk, FINGERPRINT_BYTES),
        compressor=_COMPRESSOR,
    )
    group.create_dataset(
        POLICY_FIELD,
        data=bt4_policy,
        chunks=(row_chunk, COMPACT_POLICY_SIZE),
        compressor=_COMPRESSOR,
    )
    group.create_dataset(
        GAME_ID_FIELD,
        data=game_ids,
        chunks=(row_chunk,),
        compressor=_COMPRESSOR,
    )
    group.create_dataset(
        PLY_FIELD,
        data=plies,
        chunks=(row_chunk,),
        compressor=_COMPRESSOR,
    )
    group.attrs.update(attrs)
    os.replace(writing, pending.target)
    return attrs


def verify_shard(
    pending: PendingShard,
    *,
    onnx_sha256: str,
    expected_policy_output: str,
    expected_providers: Sequence[str],
    expected_remap: Mapping[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """Deeply replay one raw shard and compare every stored sidecar row."""
    attrs = validate_existing(
        pending,
        onnx_sha256=onnx_sha256,
        policy_output=expected_policy_output,
        providers=expected_providers,
    )
    if file_sha256(pending.path) != attrs.get("source_sha256"):
        raise ValueError(f"{pending.target}: compressed source SHA-256 mismatch")
    if functional_remap_identity(attrs.get("remap_provenance")) != dict(expected_remap):
        raise ValueError(f"{pending.target}: functional policy remap mismatch")

    group: Any = zarr.open_group(str(pending.target), mode="r")
    policy_digest = hashlib.sha256()
    source_key_digest = hashlib.sha256()
    input_key_digest = hashlib.sha256()
    game_id_digest = hashlib.sha256()
    ply_digest = hashlib.sha256()
    cursor = 0
    batch_rows: list[Mapping[str, Any]] = []

    def verify_batch(rows: Sequence[Mapping[str, Any]]) -> None:
        nonlocal cursor
        if not rows:
            return
        stop = cursor + len(rows)
        if stop > pending.claimed_rows:
            raise ValueError(f"{pending.path}: more decoded rows than claimed")
        planes, boards, raw_keys, gids, plies = encode_rows(rows, source=pending.source)
        fingerprints = position_fingerprints(
            planes,
            input_history_encoding=derive.INPUT_HISTORY_ENCODING,
        )
        wanted_source_keys = np.frombuffer(
            b"".join(fingerprints), dtype=np.uint8,
        ).reshape(len(rows), FINGERPRINT_BYTES)
        stored_source_keys = np.asarray(group[SOURCE_KEY_FIELD][cursor:stop])
        stored_input_keys = np.asarray(group[INPUT_KEY_FIELD][cursor:stop])
        stored_game_ids = np.asarray(group[GAME_ID_FIELD][cursor:stop])
        stored_plies = np.asarray(group[PLY_FIELD][cursor:stop])
        policy = np.asarray(group[POLICY_FIELD][cursor:stop], dtype=np.float32)
        if not np.array_equal(stored_source_keys, wanted_source_keys):
            raise ValueError(f"{pending.target}:{cursor}: source fingerprint mismatch")
        if not np.array_equal(stored_input_keys, raw_keys):
            raise ValueError(f"{pending.target}:{cursor}: input key mismatch")
        if not np.array_equal(stored_game_ids, gids):
            raise ValueError(f"{pending.target}:{cursor}: game ID mismatch")
        if not np.array_equal(stored_plies, plies):
            raise ValueError(f"{pending.target}:{cursor}: ply mismatch")
        if not np.isfinite(policy).all() or bool(np.any(policy < 0.0)):
            raise ValueError(f"{pending.target}:{cursor}: invalid stored policy values")
        if not bool(np.allclose(policy.sum(axis=1, dtype=np.float64), 1.0, atol=2e-6)):
            raise ValueError(f"{pending.target}:{cursor}: unnormalized stored policy")
        for offset, board in enumerate(boards):
            legal = {
                compact_index_for_move(board, move) for move in board.legal_moves
            }
            nonzero = set(np.flatnonzero(policy[offset] != 0.0).tolist())
            if not nonzero.issubset(legal):
                raise ValueError(
                    f"{pending.target}:{cursor + offset}: policy mass on illegal moves",
                )
        for digest, value in (
            (source_key_digest, stored_source_keys),
            (input_key_digest, stored_input_keys),
            (game_id_digest, stored_game_ids),
            (ply_digest, stored_plies),
            (policy_digest, policy),
        ):
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        cursor = stop

    for row in derive.iter_corpus_rows(pending.path):
        batch_rows.append(row)
        if len(batch_rows) >= batch_size:
            verify_batch(batch_rows)
            batch_rows = []
    verify_batch(batch_rows)
    if cursor != pending.claimed_rows:
        raise ValueError(
            f"{pending.path}: decoded {cursor}, progress claims {pending.claimed_rows}",
        )
    digests = {
        "source_key_sha256": source_key_digest.hexdigest(),
        "input_key_sha256": input_key_digest.hexdigest(),
        "game_id_sha256": game_id_digest.hexdigest(),
        "ply_sha256": ply_digest.hexdigest(),
        "bt4_policy_sha256": policy_digest.hexdigest(),
    }
    bad = {
        key: (attrs.get(key), value)
        for key, value in digests.items()
        if attrs.get(key) != value
    }
    if bad:
        raise ValueError(f"{pending.target}: stored array digest mismatch {bad}")
    return attrs


def verify_all(args: argparse.Namespace, *, out_root: Path, onnx_sha: str) -> int:
    """Deep-verify exact sidecar coverage of the current closed inventories."""
    verify_path = out_root / VERIFY_NAME
    atomic_json(
        verify_path,
        {"schema": SCHEMA, "verdict": "RUNNING", "started_unix": time.time()},
    )
    try:
        sources = load_sources(args.source, out_root)
        total_shards = 0
        total_rows = 0
        common_output: str | None = None
        common_providers: list[str] | None = None
        current_remap = functional_remap_identity(remap_provenance())
        source_results: dict[str, Any] = {}
        for source in sources:
            receipts = read_receipts(source.out_dir / PROGRESS_NAME)
            expected_names = {path.name for path in source.inventory.shards}
            if set(receipts) != expected_names:
                raise ValueError(
                    f"{source.source_id}: receipt inventory does not exactly equal "
                    "the closed source inventory",
                )
            expected_targets = {sidecar_name(name) for name in expected_names}
            found_targets = {
                path.name
                for path in source.out_dir.iterdir()
                if path.is_dir() and path.name.endswith(".bt4.zarr")
            }
            writing = sorted(source.out_dir.glob("*.bt4.zarr.writing"))
            if writing or found_targets != expected_targets:
                raise ValueError(
                    f"{source.source_id}: sidecar inventory mismatch or partial output; "
                    f"writing={[path.name for path in writing]}",
                )
            rows_verified = 0
            for number, (path, claimed_rows) in enumerate(
                zip(source.inventory.shards, source.inventory.shard_rows, strict=True),
                start=1,
            ):
                target = source.out_dir / sidecar_name(path.name)
                group: Any = zarr.open_group(str(target), mode="r")
                attrs = dict(group.attrs)
                output = str(attrs.get("policy_output", ""))
                providers = list(attrs.get("providers", []))
                remap = functional_remap_identity(attrs.get("remap_provenance"))
                if common_output is None:
                    common_output = output
                    common_providers = providers
                if output != common_output or providers != common_providers:
                    raise ValueError(
                        f"{target}: model output/provider differs from bank",
                    )
                if remap != current_remap:
                    raise ValueError(f"{target}: policy remap differs from current code")
                if float(args.gpu_mem_gb) > 0.0 and "CUDAExecutionProvider" not in providers:
                    raise ValueError(f"{target}: GPU labeling requested but CUDA was not used")
                item = PendingShard(source, path, int(claimed_rows), target)
                verify_shard(
                    item,
                    onnx_sha256=onnx_sha,
                    expected_policy_output=common_output,
                    expected_providers=common_providers or [],
                    expected_remap=current_remap,
                    batch_size=int(args.batch_size),
                )
                rows_verified += int(claimed_rows)
                total_shards += 1
                total_rows += int(claimed_rows)
                print(
                    f"[bt4-raw-verify] {source.source_id} {number}/"
                    f"{len(source.inventory.shards)} shards, {rows_verified} rows",
                    flush=True,
                )
            source_results[source.source_id] = {
                "corpus_complete": source.corpus_complete,
                "source_shards": len(source.inventory.shards),
                "source_rows": source.inventory.rows_claimed,
                "verified_shards": len(source.inventory.shards),
                "verified_rows": rows_verified,
                "unlisted_in_flight": list(source.inventory.unlisted_on_disk),
                "manifest_sha256": source.manifest_sha256,
                "config_sha256": str(source.manifest["config_sha256"]),
            }
        snapshot_only = not all(source.corpus_complete for source in sources)
        verdict = "SNAPSHOT_PASS" if snapshot_only else "PASS"
        receipt = {
            "schema": SCHEMA,
            "verdict": verdict,
            "snapshot_only": snapshot_only,
            "onnx_sha256": onnx_sha,
            "policy_output": common_output,
            "providers": common_providers,
            "functional_remap_identity": current_remap,
            "sources": source_results,
            "total_verified_shards": total_shards,
            "total_verified_rows": total_rows,
            "verified_unix": time.time(),
        }
        atomic_json(verify_path, receipt)
        print(
            f"[bt4-raw-verify] {verdict}: {total_rows} rows / {total_shards} shards",
            flush=True,
        )
        return 0
    except BaseException as exc:
        atomic_json(
            verify_path,
            {
                "schema": SCHEMA,
                "verdict": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_unix": time.time(),
            },
        )
        raise


@contextmanager
def gpu_lease(path: Path, *, poll_seconds: float) -> Generator[None, None, None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        announced = False
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not announced:
                    print(f"[bt4-raw] waiting for GPU lease {path}", flush=True)
                    announced = True
                time.sleep(poll_seconds)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_label_group(args: argparse.Namespace) -> int:
    """Process one bounded group; called only inside a parent-held GPU lease."""
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(out_root).free / (1024 ** 3)
    if free_gib < float(args.min_free_gib):
        raise SystemExit(
            f"free disk {free_gib:.1f} GiB is below --min-free-gib "
            f"{float(args.min_free_gib):.1f}",
        )
    onnx_path = Path(args.onnx).resolve()
    onnx_sha = file_sha256(onnx_path)
    sources = load_sources(args.source, out_root)
    todo, complete_before = pending_shards(sources, onnx_sha256=onnx_sha)
    todo.sort(key=lambda item: (item.path.stat().st_mtime_ns, item.source.source_id, item.path.name))
    selected = todo[: int(args.max_shards)]
    if not selected:
        print(
            f"[bt4-raw] caught up: {sum(complete_before.values())} closed shards "
            f"across {len(sources)} sources",
            flush=True,
        )
        return 0

    sess, input_name, input_dtype, providers = open_session(
        str(onnx_path),
        gpu_mem_gb=float(args.gpu_mem_gb),
        threads=int(args.threads),
    )
    if float(args.gpu_mem_gb) > 0.0 and "CUDAExecutionProvider" not in providers:
        raise RuntimeError("GPU labeling requested but CUDAExecutionProvider is unavailable")
    policy_index = resolve_policy_output(sess, args.policy_output)
    policy_name = sess.get_outputs()[policy_index].name
    remap_stamp = remap_provenance()
    remap_identity = functional_remap_identity(remap_stamp)
    # Existing sidecars must agree with the realized session before this
    # invocation can extend the bank.
    sources = load_sources(args.source, out_root)
    todo, complete_before = pending_shards(
        sources,
        onnx_sha256=onnx_sha,
        policy_output=policy_name,
        providers=providers,
        functional_remap=remap_identity,
    )
    todo.sort(
        key=lambda item: (
            item.path.stat().st_mtime_ns,
            item.source.source_id,
            item.path.name,
        )
    )
    selected = todo[: int(args.max_shards)]
    if not selected:
        print("[bt4-raw] caught up after session identity validation", flush=True)
        return 0
    print(
        f"[bt4-raw] lease acquired; providers={providers} policy={policy_name}; "
        f"processing {len(selected)}/{len(todo)} pending closed shards",
        flush=True,
    )
    started = time.time()
    rows_done = 0
    shards_done = 0
    for pending in selected:
        attrs = label_shard(
            pending,
            sess=sess,
            input_name=input_name,
            input_dtype=np.dtype(input_dtype),
            providers=providers,
            policy_name=policy_name,
            onnx_path=onnx_path,
            onnx_sha256=onnx_sha,
            remap_stamp=remap_stamp,
            batch_size=int(args.batch_size),
        )
        append_receipt(
            pending.source.out_dir / PROGRESS_NAME,
            receipt_from_attrs(attrs, pending.target),
        )
        rows_done += int(attrs["positions"])
        shards_done += 1
        elapsed = max(time.time() - started, 1e-9)
        print(
            f"[bt4-raw] {shards_done}/{len(selected)} {pending.source.source_id}/"
            f"{pending.path.name}: {rows_done} rows, {rows_done / elapsed:.1f} pos/s",
            flush=True,
        )

    status: dict[str, Any] = {
        "schema": SCHEMA,
        "onnx_sha256": onnx_sha,
        "last_invocation_shards": shards_done,
        "last_invocation_rows": rows_done,
        "sources": {},
        "updated_unix": time.time(),
    }
    refreshed = load_sources(args.source, out_root)
    _, complete_after = pending_shards(refreshed, onnx_sha256=onnx_sha)
    for source in refreshed:
        status["sources"][source.source_id] = {
            "closed_source_shards": len(source.inventory.shards),
            "closed_source_rows": source.inventory.rows_claimed,
            "complete_sidecars": complete_after[source.source_id],
            "unlisted_in_flight": list(source.inventory.unlisted_on_disk),
        }
    atomic_json(out_root / STATUS_NAME, status)
    return 0


def label_child(args: argparse.Namespace) -> None:
    """Own both the lease and ONNX Runtime until process teardown."""
    with gpu_lease(
        Path(args.gpu_lock).resolve(),
        poll_seconds=float(args.lock_poll_seconds),
    ):
        raise SystemExit(run_label_group(args))


def run(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    onnx_sha = file_sha256(Path(args.onnx).resolve())
    if bool(args.verify_all):
        return verify_all(args, out_root=out_root, onnx_sha=onnx_sha)

    # The disposable child owns both the flock and CUDA. Process teardown drops
    # them together, even if this coordinator dies, so a waiter cannot observe
    # an unlocked lease while an orphaned ORT session is still running.
    child = multiprocessing.get_context("spawn").Process(
        target=label_child,
        args=(args,),
    )
    child.start()
    child.join()
    if child.exitcode != 0:
        raise RuntimeError(f"BT4 label child exited with status {child.exitcode}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="source namespace and live raw-corpus directory; repeatable",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=Path(DEFAULT_ONNX))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-mem-gb", type=float, default=24.0)
    parser.add_argument("--policy-output", default=None)
    parser.add_argument("--max-shards", type=int, default=16)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--lock-poll-seconds", type=float, default=2.0)
    parser.add_argument("--min-free-gib", type=float, default=150.0)
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="deeply verify exact coverage of the current closed inventories; no GPU",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.batch_size) <= 0:
        raise SystemExit("--batch-size must be positive")
    if int(args.threads) < 0:
        raise SystemExit("--threads must be non-negative")
    if int(args.max_shards) <= 0:
        raise SystemExit("--max-shards must be positive")
    if float(args.lock_poll_seconds) <= 0.0:
        raise SystemExit("--lock-poll-seconds must be positive")
    if float(args.min_free_gib) < 0.0:
        raise SystemExit("--min-free-gib must be non-negative")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
