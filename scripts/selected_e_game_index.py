"""Bounded, NO-LAUNCH identity index for authenticated *small* E fixtures.

This module deliberately has no registered-E paths or CLI. A caller must first
authenticate a closed fixture roster and pin decoded scalar-column hashes.
Opening/phase/legal strata and payload qualification belong to a later adapter.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import Blosc
from numcodecs.blosc import cbuffer_sizes


MAX_SHARDS = 128
MAX_ROWS = 50_000
MAX_COLUMN_BYTES = 160_000
MAX_FILE_BYTES = 192 * 1024
MAX_METADATA_BYTES = 4_096
MAX_GAMES = 50_000
MAX_CHUNKS_PER_COLUMN = 512
MAX_TOTAL_CHUNKS = 4_096
MAX_READ_BYTES = 64 * 1024 * 1024
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _hex(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _component(name: str) -> bool:
    return bool(name) and name not in {".", ".."} and "/" not in name and "\0" not in name


@dataclass(frozen=True)
class FixtureShard:
    cohort_index: int
    cohort_manifest_sha256: str
    shard_ordinal: int
    shard_name: str
    shard_rows: int
    resolved_base_parent: str
    game_id_sha256: str
    has_game_id_sha256: str

    def validate(self) -> None:
        require(0 <= self.cohort_index < 35, "cohort index out of range")
        require(self.shard_ordinal >= 0 and self.shard_rows > 0, "invalid shard ordinal/rows")
        require(_component(self.shard_name) and self.shard_name.endswith(".zarr"), "invalid shard name")
        require(Path(self.resolved_base_parent).is_absolute()
                and os.path.normpath(self.resolved_base_parent) == self.resolved_base_parent,
                "base parent must be a canonical absolute name")
        require(all(_hex(h) for h in (self.cohort_manifest_sha256, self.game_id_sha256,
                                      self.has_game_id_sha256)), "invalid SHA-256 pin")

    def as_dict(self) -> dict[str, object]:
        return dict(vars(self))


@dataclass(frozen=True)
class FixtureClosure:
    qualified_receipt_sha256: str
    manifests_roster_sha256: str
    shards: tuple[FixtureShard, ...]
    first_sealed_identity_sha256: str

    def identity_sha256(self) -> str:
        return hashlib.sha256(_json_bytes({
            "schema": "selected-e-identity-fixture-v1",
            "qualified_receipt_sha256": self.qualified_receipt_sha256,
            "manifests_roster_sha256": self.manifests_roster_sha256,
            "shards": [s.as_dict() for s in self.shards],
        })).hexdigest()

    def validate(self) -> None:
        require(_hex(self.qualified_receipt_sha256) and _hex(self.manifests_roster_sha256)
                and _hex(self.first_sealed_identity_sha256), "invalid closure SHA-256 pin")
        require(0 < len(self.shards) <= MAX_SHARDS, "fixture shard cap exceeded")
        require(sum(s.shard_rows for s in self.shards) <= MAX_ROWS, "fixture row cap exceeded")
        seen: set[str] = set()
        ordinals: set[tuple[int, int]] = set()
        manifests: dict[int, str] = {}
        for shard in self.shards:
            shard.validate()
            require(shard.shard_name not in seen, "duplicate fixture shard name")
            ordinal = shard.cohort_index, shard.shard_ordinal
            require(ordinal not in ordinals, "duplicate cohort/shard ordinal")
            require(manifests.get(shard.cohort_index, shard.cohort_manifest_sha256)
                    == shard.cohort_manifest_sha256, "cohort manifest pin differs")
            seen.add(shard.shard_name)
            ordinals.add(ordinal)
            manifests[shard.cohort_index] = shard.cohort_manifest_sha256
        require(self.first_sealed_identity_sha256 == self.identity_sha256(),
                "first-sealed fixture identity differs")


@dataclass
class _ReadBudget:
    chunks: int = 0
    bytes_read: int = 0


def _open_root(path: Path) -> int:
    require(path.is_absolute(), "fixture root must be absolute")
    fd = os.open("/", _DIR_FLAGS)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except OSError as exc:
        os.close(fd)
        raise ValueError("fixture root contains a link or non-directory") from exc


def _open_dir(parent: int, name: str) -> int:
    try:
        fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
    except OSError as exc:
        raise ValueError("fixture directory is absent, linked, or non-directory") from exc
    return fd


def _check_members(fd: int, *, files: set[str], dirs: set[str],
                   required: set[str], cap: int) -> set[str]:
    names: set[str] = set()
    with os.scandir(fd) as entries:
        for entry in entries:
            names.add(entry.name)
            require(len(names) <= cap, "fixture member-count cap exceeded")
    require(required <= names and names <= files | dirs,
            "fixture has missing or unlisted members")
    for name in names:
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if name in files:
            require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_FILE_BYTES,
                    f"fixture member {name!r} is linked, special, or oversized")
        else:
            require(stat.S_ISDIR(info.st_mode), f"fixture member {name!r} is linked or special")
    return names


def _read_file(parent: int, name: str, cap: int, budget: _ReadBudget) -> bytes:
    require(_component(name), "invalid fixture member name")
    try:
        fd = os.open(name, _FILE_FLAGS, dir_fd=parent)
    except OSError as exc:
        raise ValueError(f"fixture member {name!r} is absent or linked") from exc
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode), f"fixture member {name!r} is not a regular file")
        require(info.st_size <= cap, f"fixture member {name!r} exceeds byte cap")
        require(budget.bytes_read + info.st_size <= MAX_READ_BYTES,
                "fixture aggregate read-byte cap exceeded")
        data = os.read(fd, cap + 1)
        require(len(data) == info.st_size and len(data) <= cap,
                f"fixture member {name!r} changed size")
        budget.bytes_read += len(data)
        return data
    finally:
        os.close(fd)


def _read_column(shard_fd: int, column: str, rows: int, dtype: str,
                 expected_sha256: str, budget: _ReadBudget) -> np.ndarray:
    column_fd = _open_dir(shard_fd, column)
    try:
        names = _check_members(column_fd, files={".zarray", ".zattrs"} | {
            str(n) for n in range(rows)
        }, dirs=set(), required={".zarray"}, cap=MAX_CHUNKS_PER_COLUMN + 2)
        raw_meta = _read_file(column_fd, ".zarray", MAX_METADATA_BYTES, budget)
        meta = json.loads(raw_meta)
        require(isinstance(meta, dict) and meta.get("zarr_format") == 2,
                "identity column must be Zarr v2")
        require(set(meta) == {"zarr_format", "shape", "chunks", "dtype", "compressor",
                              "fill_value", "filters", "order"},
                "identity Zarr metadata fields differ")
        require(meta.get("shape") == [rows] and meta.get("dtype") == dtype,
                "identity column shape or dtype differs")
        chunks = meta.get("chunks")
        if not (isinstance(chunks, list) and len(chunks) == 1
                and type(chunks[0]) is int and 0 < chunks[0] <= rows):
            raise ValueError("invalid identity chunk shape")
        chunk_rows: int = chunks[0]
        require(meta.get("order") == "C" and meta.get("filters") is None
                and meta.get("fill_value") in (None, 0, False),
                "unsupported identity Zarr metadata")
        codec = meta.get("compressor")
        if not (isinstance(codec, dict) and codec.get("id") == "blosc"
                and codec.get("cname") == "zstd" and codec.get("shuffle") in (0, 1, 2)
                and set(codec) == {"id", "cname", "clevel", "shuffle", "blocksize"}):
            raise ValueError("unsupported identity codec")
        require(type(codec["clevel"]) is int and 0 <= codec["clevel"] <= 9
                and type(codec["blocksize"]) is int and 0 <= codec["blocksize"] <= MAX_COLUMN_BYTES,
                "invalid identity codec parameters")
        itemsize = np.dtype(dtype).itemsize
        require(rows * itemsize <= MAX_COLUMN_BYTES, "decoded identity column exceeds cap")
        count = (rows + chunk_rows - 1) // chunk_rows
        require(count <= MAX_CHUNKS_PER_COLUMN
                and budget.chunks + count <= MAX_TOTAL_CHUNKS,
                "fixture chunk-count cap exceeded")
        require(names <= {".zarray", ".zattrs"} | {str(n) for n in range(count)},
                "unexpected identity chunk")
        output = np.empty(rows, dtype=np.dtype(dtype))
        decoder = Blosc(cname="zstd", clevel=codec["clevel"], shuffle=codec["shuffle"],
                        blocksize=codec["blocksize"])
        for n in range(count):
            raw = _read_file(column_fd, str(n), MAX_FILE_BYTES, budget)
            budget.chunks += 1
            # Zarr v2 stores the full rectangular last chunk, including padding.
            expected_bytes = chunk_rows * itemsize
            require(len(raw) >= 16 and cbuffer_sizes(raw)[0] == expected_bytes,
                    "identity chunk declared decoded size differs")
            decoded = decoder.decode(raw)
            require(len(decoded) == expected_bytes, "identity chunk decoded size differs")
            start = n * chunk_rows
            take = min(chunk_rows, rows - start)
            output[start:start + take] = np.frombuffer(decoded, dtype=dtype)[:take]
        require(hashlib.sha256(output.tobytes()).hexdigest() == expected_sha256,
                "decoded identity column SHA-256 differs")
        return output
    finally:
        os.close(column_fd)


def index_fixture(root: Path, closure: FixtureClosure) -> dict[tuple[str, int], list[tuple[int, int]]]:
    """Return complete game-to-(shard roster index, row offset) mapping."""
    closure.validate()  # Before any filesystem open.
    root_fd = _open_root(root)
    try:
        _check_members(root_fd, files=set(), dirs={s.shard_name for s in closure.shards},
                       required={s.shard_name for s in closure.shards}, cap=MAX_SHARDS)
        budget = _ReadBudget()
        groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
        for index, shard in enumerate(closure.shards):
            shard_fd = _open_dir(root_fd, shard.shard_name)
            try:
                _check_members(shard_fd, files={".zgroup", ".zattrs"},
                               dirs={"game_id", "has_game_id"},
                               required={"game_id", "has_game_id"}, cap=4)
                ids = _read_column(shard_fd, "game_id", shard.shard_rows, "<i8",
                                   shard.game_id_sha256, budget)
                flags = _read_column(shard_fd, "has_game_id", shard.shard_rows, "|b1",
                                     shard.has_game_id_sha256, budget)
                flag_bytes = flags.view(np.uint8)
                require(bool(np.all((flag_bytes == 0) | (flag_bytes == 1))),
                        "fixture has nonbinary game-identity flags")
                require(bool(np.all(flags)), "fixture has rows without game identity")
                for offset, game_id in enumerate(ids):
                    groups[(shard.resolved_base_parent, int(game_id))].append((index, offset))
            finally:
                os.close(shard_fd)
        require(len(groups) <= MAX_GAMES, "fixture game cap exceeded")
        return dict(groups)
    finally:
        os.close(root_fd)


def select_complete_games(closure: FixtureClosure,
                          groups: dict[tuple[str, int], list[tuple[int, int]]],
                          *, seed_sha256: str, target_rows: int = 12_288,
                          required_cohorts: frozenset[int] = frozenset(range(35))) -> dict[str, Any]:
    """Freeze an exact-size whole-game selection or refuse; never trim a game."""
    closure.validate()
    require(_hex(seed_sha256), "invalid sampling seed SHA-256")
    require(0 < target_rows <= MAX_ROWS, "invalid sample row target")
    require(set(required_cohorts) <= set(range(35)), "invalid required cohort")
    seen_rows: set[tuple[int, int]] = set()
    for key, refs in groups.items():
        require(bool(refs) and len(set(refs)) == len(refs), f"duplicate row in game {key}")
        for index, offset in refs:
            require(0 <= index < len(closure.shards)
                    and 0 <= offset < closure.shards[index].shard_rows,
                    "game row is outside closed fixture")
            require((index, offset) not in seen_rows, "row appears in multiple games")
            seen_rows.add((index, offset))
    require(len(seen_rows) == sum(s.shard_rows for s in closure.shards),
            "game index omits a fixture row")
    available = {s.cohort_index for s in closure.shards}
    require(required_cohorts <= available, "required cohort absent")
    def rank(key: tuple[str, int]) -> bytes:
        return hashlib.sha256(b"selected-e-complete-game-v1\0" + bytes.fromhex(seed_sha256)
                              + _json_bytes([key[0], key[1]])).digest()
    ordered = sorted(groups, key=lambda key: (rank(key), key))
    picked: set[tuple[str, int]] = set()
    covered: set[int] = set()
    used = 0
    for cohort in sorted(required_cohorts):
        if cohort in covered:
            continue
        candidates = (key for key in ordered if key not in picked and any(
            closure.shards[i].cohort_index == cohort for i, _ in groups[key]))
        key = next(candidates, None)
        if key is None:
            raise ValueError("required cohort has no complete game")
        picked.add(key)
        used += len(groups[key])
        covered.update(closure.shards[i].cohort_index for i, _ in groups[key])
    require(used <= target_rows, "cohort anchors exceed exact row target")
    remaining = target_rows - used
    # Bitset subset sum records the first deterministic predecessor per sum.
    reachable = 1
    parents: dict[int, tuple[int, tuple[str, int]]] = {}
    mask = (1 << (remaining + 1)) - 1
    for key in ordered:
        if key in picked:
            continue
        size = len(groups[key])
        if size > remaining:
            continue
        new = ((reachable << size) & mask) & ~reachable
        while new:
            bit = new & -new
            total = bit.bit_length() - 1
            parents[total] = (total - size, key)
            new ^= bit
        reachable |= reachable << size & mask
        if reachable & (1 << remaining):
            break
    require(reachable & (1 << remaining) != 0, "no exact row total under frozen game order")
    cursor = remaining
    while cursor:
        cursor, key = parents[cursor]
        picked.add(key)
    rows = sorted((i, offset, key) for key in picked for i, offset in groups[key])
    require(len(rows) == target_rows, "selected row total differs")
    entries = [{"cohort_index": closure.shards[i].cohort_index,
                "cohort_manifest_sha256": closure.shards[i].cohort_manifest_sha256,
                "shard_ordinal": closure.shards[i].shard_ordinal,
                "shard_name": closure.shards[i].shard_name,
                "shard_rows": closure.shards[i].shard_rows,
                "row_offset": offset,
                "resolved_base_parent": key[0], "game_id": key[1]}
               for i, offset, key in rows]
    result: dict[str, Any] = {
        "schema": "selected-e-complete-game-sample-v1", "status": "NO-LAUNCH",
        "first_sealed_identity_sha256": closure.first_sealed_identity_sha256,
        "qualified_receipt_sha256": closure.qualified_receipt_sha256,
        "manifests_roster_sha256": closure.manifests_roster_sha256,
        "sampling_seed_sha256": seed_sha256, "algorithm": "cohort-anchor-hash-then-exact-subset-v1",
        "rows": entries, "selected_games": len(picked), "target_rows": target_rows,
        "covered_cohorts": sorted({row["cohort_index"] for row in entries}),
    }
    result["sample_sha256"] = hashlib.sha256(_json_bytes(result)).hexdigest()
    return result
