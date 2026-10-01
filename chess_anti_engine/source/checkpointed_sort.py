"""Bounded, checkpointed source metadata and external sort primitives.

This module does not read chess archives, supply native input bytes, resolve
teacher targets, or admit a training pack. Callers must bind those inputs to
the source/config SHA-256 pins and independently verify downstream bytes.
"""

from __future__ import annotations

import hashlib
import heapq
import io
import json
import os
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
import resource
import signal
import shutil
import struct
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Literal, cast


RECORD = struct.Struct("<32s32sQIIIIII4B")
RECORD_BYTES = 100
if RECORD.size != RECORD_BYTES:
    raise RuntimeError("fixed candidate record is not 100 bytes")
MAX_SEGMENT_ROWS = 8192
MAX_TABLE_BYTES = 16 << 20
MAX_FIELD_BYTES = 4096
MAX_RUN_ROWS = 2048
MAX_FANIN = 4
MAX_VERIFIED_RUNS = 128
MAX_VERIFIED_PARTS = 4096
U32 = (1 << 32) - 1
U64 = (1 << 64) - 1
SOURCES = ("BT4", "Ceres", "SF")
TEACHERS = ("BT4", "Ceres")
OUTCOMES = ("1-0", "0-1", "1/2-1/2")
SortKind = Literal["uid", "digest"]
Cursor = tuple[int, int]


class CheckpointError(ValueError):
    """An input, seal, cursor, or exact metadata contract was violated."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise CheckpointError(message)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("ascii")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64 and
            all(char in "0123456789abcdef" for char in value))


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, data: bytes) -> None:
    _need(not path.exists(), f"existing sealed file: {path.name}")
    stage = path.with_name("." + path.name + ".part")
    if stage.exists():
        stage.unlink()  # only this writer's unsealed scratch
    with stage.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stage, path)
    _fsync_dir(path.parent)


def _claim(path: Path, claim: dict[str, Any]) -> str:
    raw = _canonical(claim)
    if path.exists():
        _need(path.read_bytes() == raw, "source/config/code claim changed")
    else:
        _atomic_bytes(path, raw)
    return _sha(raw)


def _uid(value: object) -> tuple[str, str, str, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 5:
        raise CheckpointError("typed source-qualified UID")
    _need(all(type(item) is str and item and
              len(item.encode("utf-8")) <= MAX_FIELD_BYTES
              for item in value[:3]) and
          all(type(item) is int and 0 <= item <= U32 for item in value[3:]),
          "typed source-qualified UID")
    return cast(tuple[str, str, str, int, int], tuple(value))


def _context(value: object) -> tuple[str, str, int, str, str]:
    if not isinstance(value, (tuple, list)) or len(value) != 5:
        raise CheckpointError("exact five-field context")
    _need(all(type(value[i]) is str and value[i] and
              len(value[i].encode("utf-8")) <= MAX_FIELD_BYTES
              for i in (0, 1, 3, 4)) and
          type(value[2]) is int and value[2] >= 0, "exact five-field context")
    return cast(tuple[str, str, int, str, str], tuple(value))


@dataclass(frozen=True)
class CandidateMetadata:
    input_digest_sha256: str
    input_bytes_sha256: str
    ordinal: int
    uid: tuple[str, str, str, int, int]
    context: tuple[str, str, int, str, str]
    game_proof_sha256: str
    provenance_sha256: str
    outcome: str
    source: str
    teacher: str

    def validated(self) -> CandidateMetadata:
        _need(_hex64(self.input_digest_sha256) and
              _hex64(self.input_bytes_sha256) and
              _hex64(self.game_proof_sha256) and
              _hex64(self.provenance_sha256), "candidate SHA-256 field")
        _need(type(self.ordinal) is int and 0 <= self.ordinal <= U64,
              "candidate global ordinal")
        uid = _uid(self.uid)
        _context(self.context)
        _need(self.outcome in OUTCOMES and self.source in SOURCES and
              self.teacher in TEACHERS and uid[1] == self.source,
              "candidate source/outcome/teacher enum")
        return self


def seal_candidate_segment(
    path: Path,
    rows: Iterable[CandidateMetadata],
    *,
    segment: int,
    source_sha256: str,
    config_sha256: str,
    native_source_sha256: str,
) -> dict[str, Any]:
    """Seal <=8,192 metadata rows; native bytes are separately pinned upstream."""
    _need(sys.flags.optimize == 0, "optimized Python mode forbidden")
    _need(type(segment) is int and 0 <= segment <= U32 and
          all(_hex64(value) for value in
              (source_sha256, config_sha256, native_source_sha256)),
          "segment number and source pins")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    claim = {"schema": "candidate_metadata_segment_claim_v1",
             "segment": segment, "source_sha256": source_sha256,
             "config_sha256": config_sha256,
             "native_source_sha256": native_source_sha256,
             "code_sha256": _file_sha(Path(__file__)),
             "record_format": "<32s32sQIIIIII4B",
             "record_bytes": RECORD_BYTES,
             "max_rows": MAX_SEGMENT_ROWS}
    claim_sha = _claim(path / "CLAIM.json", claim)
    receipt_path = path / "RECEIPT.json"
    if receipt_path.exists():
        return verify_candidate_segment(path, source_sha256=source_sha256,
                                        config_sha256=config_sha256,
                                        native_source_sha256=native_source_sha256)
    tables: dict[str, list[Any]] = {"uid_prefixes": [], "contexts": [],
                                    "proofs": [], "provenances": []}
    table_ids: dict[str, dict[bytes, int]] = {name: {} for name in tables}
    encoded = bytearray()
    count = 0
    table_bytes = 0
    seen_uid: set[tuple[str, str, str, int, int]] = set()
    for row in rows:
        row.validated()
        _need(count < MAX_SEGMENT_ROWS and
              row.ordinal == segment * MAX_SEGMENT_ROWS + count,
              "segment row bound and contiguous ordinal")
        uid = _uid(row.uid)
        _need(uid not in seen_uid, "duplicate segment UID")
        seen_uid.add(uid)
        values = {"uid_prefixes": uid[:3], "contexts": _context(row.context),
                  "proofs": row.game_proof_sha256,
                  "provenances": row.provenance_sha256}
        local: dict[str, int] = {}
        for name, value in values.items():
            token = _canonical(value)
            if token not in table_ids[name]:
                table_bytes += len(token) + 2
                _need(table_bytes <= MAX_TABLE_BYTES,
                      "candidate local table byte cap")
                table_ids[name][token] = len(tables[name])
                tables[name].append(value)
            local[name] = table_ids[name][token]
            _need(local[name] <= U32, "local table index overflow")
        encoded.extend(RECORD.pack(
            bytes.fromhex(row.input_digest_sha256),
            bytes.fromhex(row.input_bytes_sha256), row.ordinal,
            local["uid_prefixes"], uid[3], uid[4], local["contexts"],
            local["proofs"], local["provenances"],
            OUTCOMES.index(row.outcome), SOURCES.index(row.source),
            TEACHERS.index(row.teacher), 0))
        count += 1
    _need(count > 0, "empty candidate segment")
    tables_raw = _canonical(tables)
    _need(len(tables_raw) <= MAX_TABLE_BYTES,
          "candidate local table byte cap")
    for name, raw in (("RECORDS.bin", bytes(encoded)),
                      ("TABLES.json", tables_raw)):
        file = path / name
        if file.exists():
            _need(file.read_bytes() == raw, "unsealed segment file mismatch")
        else:
            _atomic_bytes(file, raw)
    receipt = {"schema": "candidate_metadata_segment_receipt_v1",
               "claim_sha256": claim_sha, "segment": segment,
               "start_ordinal": segment * MAX_SEGMENT_ROWS,
               "rows": count, "record_bytes": len(encoded),
               "files": {name: {"sha256": _file_sha(path / name),
                                "bytes": (path / name).stat().st_size}
                         for name in ("RECORDS.bin", "TABLES.json")}}
    _atomic_bytes(receipt_path, _canonical(receipt))
    return verify_candidate_segment(path, source_sha256=source_sha256,
                                    config_sha256=config_sha256,
                                    native_source_sha256=native_source_sha256)


def verify_candidate_segment(
    path: Path, *, source_sha256: str, config_sha256: str,
    native_source_sha256: str,
) -> dict[str, Any]:
    """Validate seal and every fixed record without loading native input bytes."""
    _need(sys.flags.optimize == 0, "optimized Python mode forbidden")
    path = Path(path)
    claim_raw = (path / "CLAIM.json").read_bytes()
    claim = json.loads(claim_raw)
    _need(claim_raw == _canonical(claim) and
          claim["schema"] == "candidate_metadata_segment_claim_v1" and
          claim["source_sha256"] == source_sha256 and
          claim["config_sha256"] == config_sha256 and
          claim["native_source_sha256"] == native_source_sha256 and
          claim["code_sha256"] == _file_sha(Path(__file__)) and
          claim["record_format"] == "<32s32sQIIIIII4B" and
          claim["record_bytes"] == RECORD_BYTES and
          claim["max_rows"] == MAX_SEGMENT_ROWS,
          "candidate source/config/code claim")
    raw = (path / "RECEIPT.json").read_bytes()
    receipt = json.loads(raw)
    count = receipt["rows"]
    _need(raw == _canonical(receipt) and
          receipt["schema"] == "candidate_metadata_segment_receipt_v1" and
          receipt["claim_sha256"] == _sha(claim_raw) and
          receipt["segment"] == claim["segment"] and
          receipt["start_ordinal"] == claim["segment"] * MAX_SEGMENT_ROWS and
          type(count) is int and 0 < count <= MAX_SEGMENT_ROWS and
          receipt["record_bytes"] == count * RECORD_BYTES and
          set(receipt["files"]) == {"RECORDS.bin", "TABLES.json"},
          "candidate segment receipt")
    _need({item.name for item in path.iterdir()} ==
          {"CLAIM.json", "RECEIPT.json", "RECORDS.bin", "TABLES.json"},
          "candidate segment file membership")
    for name, desc in receipt["files"].items():
        _need(_file_sha(path / name) == desc["sha256"] and
              (path / name).stat().st_size == desc["bytes"],
              "candidate segment file hash/size")
    table_raw = (path / "TABLES.json").read_bytes()
    _need(len(table_raw) <= MAX_TABLE_BYTES,
          "candidate local table byte cap")
    tables = json.loads(table_raw)
    _need(table_raw == _canonical(tables) and
          set(tables) == {"uid_prefixes", "contexts", "proofs", "provenances"}
          and all(type(v) is list and len(v) <= count for v in tables.values()),
          "candidate local tables")
    prior_uid: set[tuple[str, str, str, int, int]] = set()
    with (path / "RECORDS.bin").open("rb") as stream:
        for index in range(count):
            record = stream.read(RECORD_BYTES)
            _need(len(record) == RECORD_BYTES, "truncated candidate record")
            (digest, native_sha, ordinal, prefix, game, ply, context, proof,
             provenance, outcome, source, teacher, reserved) = RECORD.unpack(record)
            _need(ordinal == receipt["start_ordinal"] + index and
                  prefix < len(tables["uid_prefixes"]) and
                  context < len(tables["contexts"]) and
                  proof < len(tables["proofs"]) and
                  provenance < len(tables["provenances"]) and
                  outcome < len(OUTCOMES) and source < len(SOURCES) and
                  teacher < len(TEACHERS) and reserved == 0,
                  "candidate fixed record field/reference")
            prefix_value = tables["uid_prefixes"][prefix]
            uid = _uid([*prefix_value, game, ply])
            _context(tables["contexts"][context])
            _need(uid[1] == SOURCES[source] and
                  _hex64(tables["proofs"][proof]) and
                  _hex64(tables["provenances"][provenance]) and
                  uid not in prior_uid and len(digest) == 32 and
                  len(native_sha) == 32,
                  "candidate source/proof/UID semantics")
            prior_uid.add(uid)
        _need(not stream.read(1), "extra candidate record bytes")
    return receipt


def iter_candidate_segment(
    path: Path, *, source_sha256: str, config_sha256: str,
    native_source_sha256: str,
) -> Iterator[CandidateMetadata]:
    """Yield authenticated metadata one record at a time after seal validation."""
    path = Path(path)
    receipt = verify_candidate_segment(
        path, source_sha256=source_sha256,
        config_sha256=config_sha256,
        native_source_sha256=native_source_sha256)
    tables = json.loads((path / "TABLES.json").read_bytes())
    with (path / "RECORDS.bin").open("rb") as stream:
        for _ in range(receipt["rows"]):
            (digest, native_sha, ordinal, prefix, game, ply, context, proof,
             provenance, outcome, source, teacher, _) = RECORD.unpack(
                 stream.read(RECORD_BYTES))
            uid = _uid([*tables["uid_prefixes"][prefix], game, ply])
            yield CandidateMetadata(
                digest.hex(), native_sha.hex(), ordinal, uid,
                _context(tables["contexts"][context]),
                tables["proofs"][proof], tables["provenances"][provenance],
                OUTCOMES[outcome], SOURCES[source], TEACHERS[teacher])


@dataclass(frozen=True)
class SortEntry:
    key: str | tuple[str, str, str, int, int]
    input_index: int
    locator: tuple[int, int]


def _validated_entry(value: SortEntry, kind: SortKind) -> SortEntry:
    _need(type(value) is SortEntry and
          type(value.input_index) is int and 0 <= value.input_index <= U64 and
          type(value.locator) is tuple and len(value.locator) == 2 and
          all(type(item) is int and 0 <= item <= U32
              for item in value.locator), "sort entry/index/locator")
    if kind == "uid":
        _uid(value.key)
    else:
        _need(_hex64(value.key), "digest sort key")
    return value


def _entry_order(entry: SortEntry) -> tuple[Any, int]:
    key = tuple(entry.key) if type(entry.key) in (tuple, list) else entry.key
    return key, entry.input_index


def _order_json(order: tuple[Any, int] | None) -> list[Any] | None:
    if order is None:
        return None
    key, index = order
    return [list(key) if type(key) is tuple else key, index]


def _entry_bytes(entry: SortEntry) -> bytes:
    return _canonical([entry.key, entry.input_index, entry.locator])


def _entry_from_line(raw: bytes, kind: SortKind) -> SortEntry:
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CheckpointError("malformed sort row") from exc
    if not isinstance(value, list) or len(value) != 3 or raw != _canonical(value):
        raise CheckpointError("canonical sort row")
    key, index, loc = value
    parsed_key: str | tuple[str, str, str, int, int]
    if kind == "uid":
        parsed_key = _uid(key)
    else:
        _need(_hex64(key), "digest sort key")
        parsed_key = cast(str, key)
    if not isinstance(loc, list) or len(loc) != 2:
        raise CheckpointError("sort row locator")
    _need(type(index) is int and all(type(x) is int for x in loc),
          "sort row index/locator types")
    return _validated_entry(SortEntry(parsed_key, cast(int, index),
                                      (cast(int, loc[0]), cast(int, loc[1]))),
                            kind)


@dataclass(frozen=True)
class _PartState:
    count: int
    rows: int
    chain_sha256: str
    last_receipt_sha256: str
    end_cursors: tuple[Cursor, ...]
    first: tuple[Any, int] | None
    last: tuple[Any, int] | None
    final_receipt: dict[str, Any] | None


@dataclass(frozen=True)
class _PartProof:
    receipt_sha256: str
    payload_sha256: str
    size: int
    rows: int
    first: tuple[Any, int]
    last: tuple[Any, int]


class _VerificationSession:
    """One invocation's capped proof cache; never persisted across restarts."""

    def __init__(self, owner: CheckpointedSort) -> None:
        self.identity = self._identity(owner)
        self.byte_cap = owner.byte_cap
        self.runs: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self.parts: dict[tuple[Path, int], _PartProof] = {}
        self.stack: set[Path] = set()
        self.run_evictions = 0
        self.payload_reads = 0
        self.payload_bytes_read = 0
        self.cursor_probe_reads = 0

    @staticmethod
    def _identity(owner: CheckpointedSort) -> tuple[object, ...]:
        return (owner.root, owner.source_sha256, owner.config_sha256,
                owner.code_sha256, owner.kind, owner.row_cap,
                owner.byte_cap, owner.fanin)

    def require_owner(self, owner: CheckpointedSort) -> None:
        _need(self.identity == self._identity(owner),
              "verification session store identity changed")

    def record_part(self, run: Path, part: int, proof: _PartProof) -> None:
        key = run, part
        if key not in self.parts:
            _need(len(self.parts) < MAX_VERIFIED_PARTS,
                  "bounded verification session part cap")
        else:
            _need(self.parts[key] == proof, "changed part within verification")
        self.parts[key] = proof

    def record_run(self, run: Path, receipt: dict[str, Any]) -> None:
        if run not in self.runs and len(self.runs) == MAX_VERIFIED_RUNS:
            evict = next((old for old in self.runs
                          if old not in self.stack), None)
            if evict is None:
                raise CheckpointError(
                    "bounded verification session active run cap")
            del self.runs[evict]
            self.run_evictions += 1
        self.runs[run] = receipt
        self.runs.move_to_end(run)

    def read_part(self, path: Path) -> bytes:
        with path.open("rb") as stream:
            payload = stream.read(self.byte_cap + 1)
            extra = stream.read(1) if len(payload) <= self.byte_cap else b""
        self.payload_reads += 1
        self.payload_bytes_read += len(payload) + len(extra)
        _need(len(payload) <= self.byte_cap and not extra,
              "physical sort part exceeds byte cap")
        return payload


def _scan_payload(payload: bytes, kind: SortKind,
                  previous: tuple[Any, int] | None
                  ) -> tuple[int, tuple[Any, int] | None,
                             tuple[Any, int] | None]:
    count = 0
    first = None
    last = previous
    for line in io.BytesIO(payload):
        entry = _entry_from_line(line, kind)
        order = _entry_order(entry)
        _need(last is None or last < order,
              "sort run ordering or repeated input index")
        if first is None:
            first = order
        last = order
        count += 1
    return count, first, last


def _require_run_receipt(run: Path, receipt: dict[str, Any]) -> None:
    """Recheck exact verified metadata, never adopt a later path's hash."""
    _need((run / "RUN.json").read_bytes() == _canonical(receipt),
          "changed input sort run receipt before merge")
    _need(_file_sha(run / "CLAIM.json") == receipt["claim_sha256"],
          "changed input sort claim before merge")


class _RunReader:
    """One-row-at-a-time physical-part reader with restartable byte cursor."""

    def __init__(self, run: Path, kind: SortKind, parts: int,
                 session: _VerificationSession,
                 run_receipt: dict[str, Any],
                 cursor: Cursor = (0, 0)) -> None:
        self.run = run
        self.kind: SortKind = kind
        self.parts = parts
        self.session = session
        self.run_receipt = run_receipt
        self.cursor = cursor
        self.stream: io.BytesIO | None = None
        self.open_part = -1

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self) -> _RunReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def next_entry(self) -> tuple[SortEntry, Cursor] | None:
        while self.cursor[0] < self.parts:
            part, offset = self.cursor
            if self.open_part != part:
                self.close()
                _require_run_receipt(self.run, self.run_receipt)
                proof = self.session.parts.get((self.run, part))
                if proof is None:
                    raise CheckpointError("unverified input sort part")
                _need(_file_sha(self.run / f"part_{part:08d}.receipt.json") ==
                      proof.receipt_sha256,
                      "changed input sort receipt before merge")
                payload = self.session.read_part(
                    self.run / f"part_{part:08d}.jsonl")
                _need(len(payload) == proof.size and
                      _sha(payload) == proof.payload_sha256,
                      "changed input sort part before merge")
                count, first, last = _scan_payload(payload, self.kind, None)
                _need(count == proof.rows and first == proof.first and
                      last == proof.last,
                      "changed input sort part rows before merge")
                self.stream = io.BytesIO(payload)
                self.open_part = part
            stream = self.stream
            if stream is None:
                raise CheckpointError("missing verified input sort buffer")
            size = self.session.parts[(self.run, part)].size
            _need(0 <= offset <= size, "sort cursor byte bounds")
            if offset == size:
                self.cursor = (part + 1, 0)
                continue
            stream.seek(offset)
            if offset:
                stream.seek(offset - 1)
                _need(stream.read(1) == b"\n", "sort cursor line boundary")
                stream.seek(offset)
            raw = stream.readline()
            _need(raw.endswith(b"\n"), "truncated sort row at cursor")
            end = stream.tell()
            self.cursor = (part + 1, 0) if end == size else (part, end)
            return _entry_from_line(raw, self.kind), self.cursor
        _need(self.cursor == (self.parts, 0), "sort end cursor")
        return None


class CheckpointedSort:
    """CPU-only run store with bounded physical parts and resumable merges.

    A logical run can have many small parts. Each completed part records the
    cursor in every input run after its last emitted row. A restarted merge
    validates all existing seals, seeks those input cursors, and emits only the
    unfinished suffix. The caller supplies an owned subprocess watchdog.
    """

    def __init__(self, root: Path, *, source_sha256: str,
                 config_sha256: str, kind: SortKind,
                 row_cap: int = MAX_RUN_ROWS,
                 byte_cap: int = 1 << 20,
                 fanin: int = MAX_FANIN) -> None:
        _need(sys.flags.optimize == 0, "optimized Python mode forbidden")
        _need(_hex64(source_sha256) and _hex64(config_sha256) and
              kind in ("uid", "digest") and
              type(row_cap) is int and 0 < row_cap <= MAX_RUN_ROWS and
              type(byte_cap) is int and 256 <= byte_cap <= (8 << 20) and
              type(fanin) is int and 2 <= fanin <= MAX_FANIN,
              "bounded sort recipe and source pins")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_sha256 = source_sha256
        self.config_sha256 = config_sha256
        self.kind: SortKind = kind
        self.row_cap = row_cap
        self.byte_cap = byte_cap
        self.fanin = fanin
        self.code_sha256 = _file_sha(Path(__file__))

    def _path(self, name: str) -> Path:
        _need(type(name) is str and 0 < len(name) <= 80 and
              all(char in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name),
              "safe run name")
        return self.root / name

    def _claim_value(self, name: str, *, mode: str,
                     expected_rows: int, source_identity: str | None = None,
                     inputs: Sequence[Path] = (),
                     input_receipts: Sequence[dict[str, Any]] = ()
                     ) -> dict[str, Any]:
        _need(type(expected_rows) is int and 0 < expected_rows <= U64,
              "positive expected run count")
        _need(len(inputs) == len(input_receipts), "verified input receipt count")
        if mode == "source":
            _need(_hex64(source_identity) and not inputs and
                  expected_rows <= self.row_cap, "bounded source run input")
        else:
            _need(mode == "merge" and 2 <= len(inputs) <= self.fanin and
                  source_identity is None, "bounded merge input fan-in")
        return {"schema": "checkpointed_sort_claim_v1", "name": name,
                "mode": mode, "kind": self.kind,
                "source_sha256": self.source_sha256,
                "config_sha256": self.config_sha256,
                "code_sha256": self.code_sha256,
                "row_cap": self.row_cap, "byte_cap": self.byte_cap,
                "fanin": self.fanin, "expected_rows": expected_rows,
                "source_identity_sha256": source_identity,
                "inputs": [{"name": path.name,
                            "run_sha256": _sha(_canonical(receipt))}
                           for path, receipt in
                           zip(inputs, input_receipts, strict=True)]}

    def _prepare(self, path: Path, claim: dict[str, Any]) -> str:
        path.mkdir(parents=True, exist_ok=True)
        return _claim(path / "CLAIM.json", claim)

    def _verify_parts(self, path: Path, claim: dict[str, Any],
                      *, require_final: bool,
                      session: _VerificationSession) -> _PartState:
        claim_raw = (path / "CLAIM.json").read_bytes()
        _need(claim_raw == _canonical(claim), "source/config/code run claim")
        claim_sha = _sha(claim_raw)
        raw_inputs = claim["inputs"]
        _need(type(raw_inputs) is list and
              all(type(item) is dict and
                  set(item) == {"name", "run_sha256"} and
                  type(item["name"]) is str and
                  _hex64(item["run_sha256"])
                  for item in raw_inputs), "run input identity schema")
        inputs = cast(list[dict[str, Any]], raw_inputs)
        cursors: tuple[Cursor, ...] = tuple((0, 0) for _ in inputs)
        chain = "0" * 64
        previous = "0" * 64
        count = total = 0
        first: tuple[Any, int] | None = None
        last: tuple[Any, int] | None = None
        while (path / f"part_{count:08d}.receipt.json").exists():
            receipt_path = path / f"part_{count:08d}.receipt.json"
            payload_path = path / f"part_{count:08d}.jsonl"
            raw = receipt_path.read_bytes()
            receipt = json.loads(raw)
            _need(raw == _canonical(receipt) and
                  receipt["schema"] == "checkpointed_sort_part_v1" and
                  receipt["claim_sha256"] == claim_sha and
                  receipt["part"] == count and
                  receipt["previous_receipt_sha256"] == previous and
                  receipt["start_cursors"] == [list(c) for c in cursors],
                  "sort part receipt chain/identity")
            end_raw = receipt["end_cursors"]
            _need(type(end_raw) is list and len(end_raw) == len(inputs) and
                  all(type(c) is list and len(c) == 2 and
                      all(type(x) is int and x >= 0 for x in c)
                      for c in end_raw), "sort input cursor schema")
            end = tuple(tuple(c) for c in end_raw)
            _need(all(a <= b for a, b in zip(cursors, end, strict=True)),
                  "sort input cursor regression")
            for item, cursor in zip(inputs, end, strict=True):
                self._validate_cursor(self._path(item["name"]), cursor,
                                      session=session)
            _need(payload_path.is_file() and
                  0 < receipt["rows"] <= self.row_cap and
                  0 < receipt["bytes"] <= self.byte_cap,
                  "sort part hash/size/bounds")
            payload = session.read_part(payload_path)
            _need(_sha(payload) == receipt["payload_sha256"] and
                  len(payload) == receipt["bytes"],
                  "sort part hash/size/bounds")
            part_count, part_first, part_last = _scan_payload(
                payload, self.kind, last)
            if (part_count != receipt["rows"] or
                    part_first is None or part_last is None):
                raise CheckpointError("sort part row count/truncation")
            _need(receipt["first"] == _order_json(part_first) and
                  receipt["last"] == _order_json(part_last),
                  "sort part key bounds")
            session.record_part(path, count, _PartProof(
                _sha(raw), receipt["payload_sha256"], len(payload),
                part_count, part_first, part_last))
            if first is None:
                first = part_first
            last = part_last
            total += part_count
            previous = _sha(raw)
            chain = _sha(bytes.fromhex(chain) + bytes.fromhex(previous))
            cursors = end
            count += 1
        all_names = {item.name for item in path.iterdir()}
        expected_names = {"CLAIM.json"}
        for index in range(count):
            expected_names.add(f"part_{index:08d}.jsonl")
            expected_names.add(f"part_{index:08d}.receipt.json")
        final = path / "RUN.json"
        if final.exists():
            expected_names.add("RUN.json")
        unsealed = {f"part_{count:08d}.jsonl",
                    f".part_{count:08d}.jsonl.part",
                    f".part_{count:08d}.receipt.json.part"}
        if not require_final:
            if not final.exists():
                unsealed.add(".RUN.json.part")
            for name in all_names & unsealed:
                (path / name).unlink()  # only the next owned unsealed part
            all_names -= unsealed
        _need(all_names == expected_names, "sort run file membership")
        final_receipt = None
        if require_final or final.exists():
            _need(final.exists() and count > 0, "final sort run receipt")
            final_raw = final.read_bytes()
            final_receipt = json.loads(final_raw)
            _need(final_raw == _canonical(final_receipt) and
                  final_receipt == {
                      "schema": "checkpointed_sort_run_v1",
                      "claim_sha256": claim_sha, "parts": count,
                      "rows": total, "receipt_chain_sha256": chain,
                      "first": _order_json(first),
                      "last": _order_json(last)} and
                  total == claim["expected_rows"],
                  "final sort run count/chain")
        return _PartState(count, total, chain, previous, cursors, first, last,
                          final_receipt)

    def _validate_cursor(self, run: Path, cursor: Cursor,
                         *, session: _VerificationSession) -> None:
        run_receipt = json.loads((run / "RUN.json").read_bytes())
        parts = run_receipt["parts"]
        part, offset = cursor
        _need(type(part) is int and type(offset) is int and
              0 <= part <= parts and offset >= 0,
              "input cursor fields")
        if part == parts:
            _need(offset == 0, "end cursor offset")
            return
        payload = run / f"part_{part:08d}.jsonl"
        size = payload.stat().st_size
        _need(offset < size, "unnormalized or oversized input cursor")
        if offset:
            with payload.open("rb") as stream:
                stream.seek(offset - 1)
                byte = stream.read(1)
            session.cursor_probe_reads += 1
            session.payload_bytes_read += len(byte)
            _need(byte == b"\n", "input cursor line boundary")

    def verify_run(self, run: Path, *,
                   _session: _VerificationSession | None = None
                   ) -> dict[str, Any]:
        """Hash and scan all sealed parts with one bounded invocation cache."""
        _need(sys.flags.optimize == 0, "optimized Python mode forbidden")
        run = Path(run)
        _need(run.parent == self.root and not run.is_symlink() and
              self._path(run.name) == run, "run outside store")
        session = _session or _VerificationSession(self)
        session.require_owner(self)
        memo = session.runs
        stack = session.stack
        if run in memo:
            _require_run_receipt(run, memo[run])
            memo.move_to_end(run)
            return memo[run]
        _need(run not in stack, "cyclic run inputs")
        stack.add(run)
        raw = (run / "CLAIM.json").read_bytes()
        claim = json.loads(raw)
        _need(raw == _canonical(claim) and
              claim["schema"] == "checkpointed_sort_claim_v1" and
              claim["name"] == run.name and claim["kind"] == self.kind and
              claim["source_sha256"] == self.source_sha256 and
              claim["config_sha256"] == self.config_sha256 and
              claim["code_sha256"] == self.code_sha256 and
              claim["row_cap"] == self.row_cap and
              claim["byte_cap"] == self.byte_cap and
              claim["fanin"] == self.fanin and
              type(claim["expected_rows"]) is int and
              claim["expected_rows"] > 0,
              "sort source/config/code claim")
        inputs = claim["inputs"]
        child_receipts: dict[str, dict[str, Any]] = {}
        if claim["mode"] == "source":
            _need(_hex64(claim["source_identity_sha256"]) and
                  inputs == [] and claim["expected_rows"] <= self.row_cap,
                  "source run identity")
        else:
            _need(claim["mode"] == "merge" and
                  claim["source_identity_sha256"] is None and
                  2 <= len(inputs) <= self.fanin and
                  len({item["name"] for item in inputs}) == len(inputs),
                  "merge input roster")
            total = 0
            for item in inputs:
                child = self._path(item["name"])
                child_receipt = self.verify_run(child, _session=session)
                child_receipts[item["name"]] = child_receipt
                _need(_file_sha(child / "RUN.json") == item["run_sha256"],
                      "changed merge input receipt")
                total += child_receipt["rows"]
            _need(total == claim["expected_rows"], "merge input row count")
        state = self._verify_parts(run, claim, require_final=True,
                                   session=session)
        if inputs:
            for item, cursor in zip(inputs, state.end_cursors, strict=True):
                child = self._path(item["name"])
                self._validate_cursor(child, cursor, session=session)
                child_run = child_receipts[item["name"]]
                _need(cursor == (child_run["parts"], 0),
                      "final merge did not consume every input")
        receipt = state.final_receipt
        if receipt is None:
            raise CheckpointError("sort final receipt state")
        _require_run_receipt(run, receipt)
        session.record_run(run, receipt)
        stack.remove(run)
        return receipt

    def _seal_part(self, run: Path, *, claim_sha: str, index: int,
                   payload: bytes, rows: int, previous: str,
                   start_cursors: tuple[Cursor, ...],
                   end_cursors: tuple[Cursor, ...],
                   first: tuple[Any, int],
                   last: tuple[Any, int]) -> str:
        _need(0 < rows <= self.row_cap and 0 < len(payload) <= self.byte_cap,
              "physical sort part bound")
        data_file = run / f"part_{index:08d}.jsonl"
        receipt_file = run / f"part_{index:08d}.receipt.json"
        _atomic_bytes(data_file, payload)
        receipt = {"schema": "checkpointed_sort_part_v1",
                   "claim_sha256": claim_sha, "part": index,
                   "previous_receipt_sha256": previous,
                   "payload_sha256": _sha(payload), "bytes": len(payload),
                   "rows": rows, "first": _order_json(first),
                   "last": _order_json(last),
                   "start_cursors": [list(cursor) for cursor in start_cursors],
                   "end_cursors": [list(cursor) for cursor in end_cursors]}
        _atomic_bytes(receipt_file, _canonical(receipt))
        return _file_sha(receipt_file)

    def _finalize(self, run: Path, claim: dict[str, Any],
                  session: _VerificationSession) -> dict[str, Any]:
        state = self._verify_parts(run, claim, require_final=False,
                                   session=session)
        _need(state.rows == claim["expected_rows"] and state.count > 0,
              "sort run incomplete at finalize")
        receipt = {"schema": "checkpointed_sort_run_v1",
                   "claim_sha256": _file_sha(run / "CLAIM.json"),
                   "parts": state.count, "rows": state.rows,
                   "receipt_chain_sha256": state.chain_sha256,
                   "first": _order_json(state.first),
                   "last": _order_json(state.last)}
        _atomic_bytes(run / "RUN.json", _canonical(receipt))
        return self.verify_run(run, _session=session)

    def seal_source_run(self, name: str,
                        entries: Iterable[SortEntry],
                        *, source_identity_sha256: str) -> Path:
        """Sort one independently pinned input slice of at most 2,048 rows."""
        run = self._path(name)
        rows: list[SortEntry] = []
        encoded_bytes = 0
        for entry in entries:
            _validated_entry(entry, self.kind)
            _need(len(rows) < self.row_cap, "source sort run row cap")
            encoded_bytes += len(_entry_bytes(entry))
            _need(encoded_bytes <= self.byte_cap,
                  "source sort run byte cap")
            rows.append(entry)
        _need(bool(rows), "empty source sort run")
        rows.sort(key=_entry_order)
        orders = [_entry_order(entry) for entry in rows]
        _need(all(a < b for a, b in pairwise(orders)),
              "duplicate source sort identity")
        payload = b"".join(_entry_bytes(entry) for entry in rows)
        _need(len(payload) <= self.byte_cap, "source sort run byte cap")
        claim = self._claim_value(name, mode="source",
                                  expected_rows=len(rows),
                                  source_identity=source_identity_sha256)
        session = _VerificationSession(self)
        claim_sha = self._prepare(run, claim)
        if (run / "RUN.json").exists():
            self.verify_run(run, _session=session)
            return run
        state = self._verify_parts(run, claim, require_final=False,
                                   session=session)
        if state.count == 0:
            self._seal_part(run, claim_sha=claim_sha, index=0,
                            payload=payload, rows=len(rows),
                            previous="0" * 64,
                            start_cursors=(), end_cursors=(),
                            first=orders[0], last=orders[-1])
        else:
            _need(state.count == 1 and state.rows == len(rows) and
                  _file_sha(run / "part_00000000.jsonl") == _sha(payload),
                  "changed resumed source sort rows")
        self._finalize(run, claim, session)
        return run

    def merge_run(self, name: str, inputs: Sequence[Path],
                  *, after_part_seal: Callable[[Path, int], None] | None = None,
                  _session: _VerificationSession | None = None
                  ) -> Path:
        """Resume a <=4-way stable merge at a sealed physical-part boundary."""
        _need(2 <= len(inputs) <= self.fanin and
              len(set(inputs)) == len(inputs), "merge input fan-in/uniqueness")
        session = _session or _VerificationSession(self)
        session.require_owner(self)
        input_receipts = [self.verify_run(path, _session=session)
                          for path in inputs]
        claim = self._claim_value(
            name, mode="merge", inputs=inputs, input_receipts=input_receipts,
            expected_rows=sum(item["rows"] for item in input_receipts))
        run = self._path(name)
        claim_sha = self._prepare(run, claim)
        if (run / "RUN.json").exists():
            self.verify_run(run, _session=session)
            return run
        state = self._verify_parts(run, claim, require_final=False,
                                   session=session)
        _need(len(state.end_cursors) == len(inputs),
              "resume input cursor count")
        for path, cursor in zip(inputs, state.end_cursors, strict=True):
            self._validate_cursor(path, cursor, session=session)
        committed = list(state.end_cursors)
        part_start = tuple(committed)
        previous = state.last_receipt_sha256
        part_number = state.count
        buffer = bytearray()
        buffer_rows = 0
        buffer_first: tuple[Any, int] | None = None
        buffer_last: tuple[Any, int] | None = None
        last_order = state.last
        readers = [_RunReader(path, self.kind, receipt["parts"],
                              session, receipt, cursor)
                   for path, receipt, cursor in
                   zip(inputs, input_receipts, committed, strict=True)]
        heap: list[tuple[Any, int, int, SortEntry, Cursor]] = []

        def push(index: int) -> None:
            item = readers[index].next_entry()
            if item is not None:
                entry, after = item
                order = _entry_order(entry)
                heapq.heappush(heap, (order[0], order[1], index,
                                      entry, after))

        def flush() -> None:
            nonlocal buffer, buffer_rows, buffer_first, buffer_last
            nonlocal part_number, part_start, previous
            if not buffer_rows:
                return
            if buffer_first is None or buffer_last is None:
                raise CheckpointError("nonempty physical part order")
            previous = self._seal_part(
                run, claim_sha=claim_sha, index=part_number,
                payload=bytes(buffer), rows=buffer_rows, previous=previous,
                start_cursors=part_start, end_cursors=tuple(committed),
                first=buffer_first, last=buffer_last)
            sealed_number = part_number
            part_number += 1
            part_start = tuple(committed)
            buffer = bytearray()
            buffer_rows = 0
            buffer_first = buffer_last = None
            if after_part_seal is not None:
                after_part_seal(run, sealed_number)

        try:
            for index in range(len(readers)):
                push(index)
            while heap:
                key, origin, index, entry, after = heapq.heappop(heap)
                order = key, origin
                _need(last_order is None or last_order < order,
                      "merge ordering or duplicate sort identity")
                raw = _entry_bytes(entry)
                _need(len(raw) <= self.byte_cap,
                      "one merged row exceeds physical byte cap")
                if buffer_rows and (buffer_rows == self.row_cap or
                                    len(buffer) + len(raw) > self.byte_cap):
                    flush()
                buffer.extend(raw)
                buffer_rows += 1
                if buffer_first is None:
                    buffer_first = order
                buffer_last = last_order = order
                committed[index] = after
                push(index)
            flush()
        finally:
            for reader in readers:
                reader.close()
        self._finalize(run, claim, session)
        return run

    def merge_all(self, inputs: Sequence[Path], *, prefix: str,
                  after_part_seal: Callable[[Path, int], None] | None = None,
                  _session: _VerificationSession | None = None
                  ) -> Path:
        _need(bool(inputs), "no external sort runs")
        session = _session or _VerificationSession(self)
        session.require_owner(self)
        current = list(inputs)
        depth = 0
        while len(current) > 1:
            next_runs: list[Path] = []
            for number, start in enumerate(range(0, len(current), self.fanin)):
                group = current[start:start + self.fanin]
                if len(group) == 1:
                    next_runs.append(group[0])
                else:
                    name = f"{prefix}_{depth:03d}_{number:06d}"
                    next_runs.append(self.merge_run(
                        name, group, after_part_seal=after_part_seal,
                        _session=session))
            current = next_runs
            depth += 1
        return current[0]

    def iter_run(self, run: Path) -> Iterator[SortEntry]:
        session = _VerificationSession(self)
        receipt = self.verify_run(run, _session=session)
        with _RunReader(run, self.kind, receipt["parts"], session,
                        receipt) as reader:
            while (item := reader.next_entry()) is not None:
                yield item[0]


@dataclass(frozen=True)
class DigestMember:
    """Caller-supplied exact native/context facts for one sealed locator."""

    native: bytes
    context: tuple[str, str, int, str, str]
    rank: tuple[bytes, bytes]
    outcome: int


@dataclass(frozen=True)
class DigestGroupResult:
    digest_sha256: str
    winner: SortEntry
    count: int
    outcome_counts: tuple[int, int, int]


def fold_digest_groups(
    sorted_entries: Iterable[SortEntry],
    *, fetch: Callable[[tuple[int, int]], DigestMember],
    on_loser: Callable[[SortEntry], None],
) -> Iterator[DigestGroupResult]:
    """Resolve repeated digests with constant group state and a loser sink.

    The caller's fetch hook must verify the native frame against the candidate
    record's input-byte SHA and digest before returning. This fold also checks
    native/context equality inside each group; it never buffers its members.
    """
    active_key: str | None = None
    first_native: bytes | None = None
    first_context: tuple[str, str, int, str, str] | None = None
    minimum_rank: tuple[bytes, bytes] | None = None
    winner: SortEntry | None = None
    count = 0
    histogram = [0, 0, 0]
    previous_order: tuple[Any, int] | None = None
    for entry in sorted_entries:
        _validated_entry(entry, "digest")
        if type(entry.key) is not str:
            raise CheckpointError("digest stream key")
        order = _entry_order(entry)
        _need(previous_order is None or previous_order < order,
              "digest stream order")
        previous_order = order
        member = fetch(entry.locator)
        _need(type(member) is DigestMember and
              type(member.native) is bytes and
              type(member.rank) is tuple and len(member.rank) == 2 and
              all(type(part) is bytes for part in member.rank) and
              type(member.outcome) is int and
              0 <= member.outcome < len(OUTCOMES), "digest member contract")
        context = _context(member.context)
        if entry.key != active_key:
            if winner is not None:
                if active_key is None:
                    raise CheckpointError("active digest")
                yield DigestGroupResult(active_key, winner, count,
                                        (histogram[0], histogram[1], histogram[2]))
            active_key = entry.key
            first_native = member.native
            first_context = context
            minimum_rank = member.rank
            winner = entry
            count = 1
            histogram = [0, 0, 0]
            histogram[member.outcome] = 1
            continue
        _need(member.native == first_native, "digest group native-byte collision")
        _need(context == first_context, "digest group exact context conflict")
        if winner is None or minimum_rank is None:
            raise CheckpointError("active digest winner")
        count += 1
        histogram[member.outcome] += 1
        if member.rank < minimum_rank:
            on_loser(winner)
            winner = entry
            minimum_rank = member.rank
        else:
            on_loser(entry)
    if winner is not None:
        if active_key is None:
            raise CheckpointError("last digest group")
        yield DigestGroupResult(active_key, winner, count,
                                (histogram[0], histogram[1], histogram[2]))


def run_owned_cpu(
    argv: Sequence[str], *, log_path: Path, timeout_seconds: int = 600,
    address_space_bytes: int = 2 << 30,
) -> int:
    """Supervise one owned CPU unit, killing and reaping on wall timeout."""
    _need(type(timeout_seconds) is int and 0 < timeout_seconds <= 600 and
          type(address_space_bytes) is int and
          64 << 20 <= address_space_bytes <= 4 << 30 and
          len(argv) > 0 and all(type(arg) is str for arg in argv),
          "bounded owned process recipe")
    log_path = Path(log_path)
    _need(not log_path.exists(), "owned process log exists")

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min(address_space_bytes,
              soft if soft >= 0 else address_space_bytes,
              hard if hard >= 0 else address_space_bytes)
    prlimit = shutil.which("prlimit")
    if prlimit is None:
        raise CheckpointError("Linux prlimit unavailable for owned CPU cap")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            [prlimit, f"--as={cap}", "--", *argv],
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=env)
        try:
            code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # child exited at the deadline; still reap it
            process.wait(timeout=10)
            raise TimeoutError("owned CPU unit exceeded wall deadline") from exc
        finally:
            log.flush()
            os.fsync(log.fileno())
    _fsync_dir(log_path.parent)
    return code
