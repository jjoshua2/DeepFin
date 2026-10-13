"""Immutable, ordinary Zarr shards stored as one ZIP_STORED file.

No extraction, overlay resolution, writer, or replay-wide discovery is provided.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from collections.abc import Generator
import zipfile

from zarr.storage import ZipStore


SUFFIX = ".zarr.zip"
_NAME = re.compile(r"shard_(\d+)\.zarr(?:\.zip)?\Z")
# One ordinary shard is a few dozen arrays of at most a few hundred chunks.
# The cap only stops a hostile `.zarray` from turning admission into a loop.
_MAX_DECLARED_CHUNKS = 100_000
_MAX_ZARRAY_BYTES = 65_536
_MAX_ZARRAY_DIMS = 8


def is_packed(path: Path) -> bool:
    return path.name.endswith(SUFFIX) or path.resolve().name.endswith(SUFFIX)


def shard_paths(root: Path) -> list[Path]:
    """Opt-in exact-epoch discovery; preserve ordinary lexical shard order."""
    paths = sorted(
        [*root.glob("shard_*.zarr"), *root.glob("shard_*.zarr.zip")],
        key=lambda p: p.name.removesuffix(".zip"),
    )
    indices: dict[int, Path] = {}
    for path in paths:
        match = _NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid exact-epoch shard name: {path}")
        index = int(match[1])
        if index in indices:
            raise ValueError(f"duplicate shard index: {indices[index]} and {path}")
        indices[index] = path
    return paths


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _validate_members(archive: zipfile.ZipFile) -> None:
    names: set[str] = set()
    for entry in archive.infolist():
        name = entry.filename
        parts = name.split("/")
        mode = entry.external_attr >> 16
        if (
            not name
            or entry.orig_filename != name
            or name in names
            or "\\" in name
            or "\x00" in name
            or PurePosixPath(name).is_absolute()
            or any(p in ("", ".", "..") for p in parts)
            or entry.is_dir()
            or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
            or entry.compress_type != zipfile.ZIP_STORED
            or entry.flag_bits & 1
            or entry.file_size != entry.compress_size
        ):
            raise ValueError(f"unsafe or duplicate packed Zarr member: {name!r}")
        # Ordinary Zarr v2 files plus the root provenance sidecar emitted by
        # derivation. Its opaque bytes stay covered by the archive hash; it is
        # never interpreted as a training array. An overlay or base-binding
        # JSON cannot be silently ignored by directory-based overlay discovery.
        if not (
            parts == ["row_provenance.npz"]
            or parts[-1] in (".zgroup", ".zattrs", ".zarray", ".zmetadata")
            or all(part.isdecimal() for part in parts[-1].split("."))
        ):
            raise ValueError(f"nonordinary packed Zarr member: {name!r}")
        names.add(name)
    if any(
        "/".join(name.split("/")[:i]) in names
        for name in names
        for i in range(1, len(name.split("/")))
    ):
        raise ValueError("packed Zarr member is also a directory prefix")
    if not {".zgroup", ".zattrs"} <= names:
        raise ValueError("packed Zarr root metadata missing")
    _require_declared_chunks(archive, names)


def _json_int(value: object) -> int | None:
    # JSON ``true`` is a bool, and bool is an int subclass. Reject it before
    # the int check so a length cannot be 1 by accident.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _require_declared_chunks(archive: zipfile.ZipFile, names: set[str]) -> None:
    """Reject a chunk grid Zarr would otherwise read back as the fill value.

    The project writer stores every chunk. The archive hash only covers bytes
    that are present, so a dropped or renamed ``x``, ``wdl_target`` or
    ``legal_mask`` chunk stays byte-consistent while those row indices
    materialize as fill (zero boards, class-0 outcomes, or empty masks) and
    the epoch still reports a complete pass.
    """
    for name in sorted(names):
        if PurePosixPath(name).name != ".zarray":
            continue
        if "/" not in name:
            raise ValueError(f"packed Zarr array metadata is not an array member: {name!r}")
        info = archive.getinfo(name)
        if info.file_size > _MAX_ZARRAY_BYTES:
            raise ValueError(f"packed Zarr array metadata is too large: {name!r}")
        try:
            meta = json.loads(archive.read(name).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise ValueError(f"packed Zarr array metadata is unreadable: {name!r}") from exc
        shape = meta.get("shape") if isinstance(meta, dict) else None
        chunks = meta.get("chunks") if isinstance(meta, dict) else None
        if (
            not isinstance(meta, dict)
            or meta.get("zarr_format", 2) != 2
            or not isinstance(shape, list)
            or not isinstance(chunks, list)
            or not 1 <= len(shape) <= _MAX_ZARRAY_DIMS
            or len(shape) != len(chunks)
        ):
            raise ValueError(f"packed Zarr array metadata is malformed: {name!r}")
        sep = meta.get("dimension_separator", ".")
        if sep not in (".", "/"):
            raise ValueError(f"packed Zarr array metadata is malformed: {name!r}")
        declared = 1
        counts: list[int] = []
        for size_raw, chunk_raw in zip(shape, chunks, strict=True):
            size = _json_int(size_raw)
            chunk = _json_int(chunk_raw)
            if size is None or chunk is None or chunk <= 0:
                raise ValueError(f"packed Zarr array metadata is malformed: {name!r}")
            count = 0 if size == 0 else (size + chunk - 1) // chunk
            if count > _MAX_DECLARED_CHUNKS or (
                count > 0 and declared > _MAX_DECLARED_CHUNKS // count
            ):
                raise ValueError(f"packed Zarr array declares too many chunks: {name!r}")
            declared *= count
            counts.append(count)
        array_name = name[: -len(".zarray")].rstrip("/")
        joiner = "." if sep == "." else "/"
        if sep == ".":
            pattern = re.compile(
                rf"{re.escape(array_name)}/\d+(?:\.\d+){{{len(shape) - 1}}}\Z"
            )
        else:
            pattern = re.compile(
                rf"{re.escape(array_name)}/\d+(?:/\d+){{{len(shape) - 1}}}\Z"
            )
        present = {member for member in names if pattern.fullmatch(member)}
        expected: set[str] = set()
        if declared:
            grid: list[tuple[int, ...]] = [()]
            for count in counts:
                grid = [
                    (*prefix, index)
                    for prefix in grid
                    for index in range(count)
                ]
            expected = {
                f"{array_name}/{joiner.join(str(index) for index in coord)}"
                for coord in grid
            }
        if present != expected:
            raise ValueError(
                f"packed Zarr array {array_name!r} stored {len(present)} chunks but "
                f"declares {declared}; a partial archive is not a readable shard"
            )


def content_sha256(path: Path) -> str:
    """Hash the actual archive bytes, refusing a concurrent rewrite/replacement."""
    path = path.resolve(strict=True)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"packed Zarr must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if _identity(os.fstat(handle.fileno())) != _identity(before):
            raise RuntimeError(f"packed Zarr changed before hashing: {path}")
        with zipfile.ZipFile(handle) as archive:
            _validate_members(archive)
        handle.seek(0)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if _identity(os.fstat(handle.fileno())) != _identity(before) or _identity(
            path.stat()
        ) != _identity(before):
            raise RuntimeError(f"packed Zarr changed while hashing: {path}")
    return digest.hexdigest()


@contextmanager
def open_store(path: Path) -> Generator[ZipStore, None, None]:
    """Own the descriptor until all lazy accesses finish, including exceptions."""
    before = path.stat()
    store = ZipStore(str(path), mode="r")
    try:
        _validate_members(store.zf)
        handle = store.zf.fp
        if handle is None:
            raise RuntimeError(f"packed Zarr descriptor closed during admission: {path}")
        if _identity(os.fstat(handle.fileno())) != _identity(before):
            raise RuntimeError(f"packed Zarr changed before reading: {path}")
        yield store
        if _identity(os.fstat(handle.fileno())) != _identity(before) or _identity(
            path.stat()
        ) != _identity(before):
            raise RuntimeError(f"packed Zarr changed while reading: {path}")
    finally:
        store.close()
