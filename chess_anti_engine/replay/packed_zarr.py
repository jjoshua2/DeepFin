"""Immutable, ordinary Zarr shards stored as one ZIP_STORED file.

No extraction, overlay resolution, writer, or replay-wide discovery is provided.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from collections.abc import Generator
import zipfile

from zarr.storage import ZipStore


SUFFIX = ".zarr.zip"
_NAME = re.compile(r"shard_(\d+)\.zarr(?:\.zip)?\Z")


def is_packed(path: Path) -> bool:
    return path.name.endswith(SUFFIX)


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
