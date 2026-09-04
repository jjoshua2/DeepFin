"""Lightweight, recoverable publication for chunk-trajectory evidence pairs.

This module intentionally imports no chess, model, evaluator, or native-extension
code.  The producer can therefore finish an interrupted publication before the
search runtime is available.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
import subprocess
from contextlib import contextmanager
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, NoReturn, Protocol

from scripts.repo_output_guard import repo_controlled_output, reserved_output_path


CHUNK_TRAJECTORY_SCHEMA = "deepfin.chunk_trajectory.v6"


class _ImmutableEvidenceExistsError(RuntimeError):
    """A no-clobber publication destination already exists."""


class _FlushableFile(Protocol):
    def flush(self) -> None: ...

    def fileno(self) -> int: ...


@dataclass(frozen=True)
class _AnchoredFile:
    path: Path
    parent_fd: int
    file_fd: int
    artifact: dict[str, Any]


def _identity(file_stat: os.stat_result) -> tuple[int, int]:
    return int(file_stat.st_dev), int(file_stat.st_ino)


def _raise_path_access_failure(exc: OSError, message: str) -> NoReturn:
    """Separate positive namespace drift from retryable inspection failures."""
    if exc.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
        raise SystemExit(message) from exc
    raise exc


def _open_parent(
    path: Path, *, quarantine_meta_path: Path | None = None,
) -> int:
    parent = path.expanduser().parent
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    descriptor = -1
    try:
        named_before = parent.stat()
        descriptor = os.open(parent, flags)
    except OSError as exc:
        _raise_path_access_failure(
            exc, f"cannot open containing directory for {path}",
        )
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        try:
            opened = os.fstat(descriptor)
            named_after = parent.stat()
        except OSError:
            if quarantine_meta_path is not None:
                _mark_manifest_namespace_invalid(
                    quarantine_meta_path, parent_fd=descriptor,
                )
            os.close(descriptor)
            raise
        if _identity(named_before) != _identity(opened) or _identity(
            named_after
        ) != _identity(opened):
            if quarantine_meta_path is not None:
                _mark_manifest_namespace_invalid(
                    quarantine_meta_path, parent_fd=descriptor,
                )
            os.close(descriptor)
            raise SystemExit(
                f"containing directory changed while being opened: {path.parent}"
            ) from exc
        os.close(descriptor)
        raise
    if not stat.S_ISDIR(opened.st_mode) or _identity(named_before) != _identity(opened):
        if quarantine_meta_path is not None:
            _mark_manifest_namespace_invalid(
                quarantine_meta_path, parent_fd=descriptor,
            )
        os.close(descriptor)
        raise SystemExit(f"containing directory changed while being opened: {path.parent}")
    return descriptor


def _require_parent(path: Path, parent_fd: int) -> None:
    try:
        named = path.expanduser().parent.stat()
    except OSError as exc:
        _raise_path_access_failure(
            exc, f"cannot revalidate containing directory for {path}",
        )
    if _identity(named) != _identity(os.fstat(parent_fd)):
        raise SystemExit(f"containing directory changed during publication: {path.parent}")


def _authenticate_initial_parent(path: Path, parent_fd: int) -> None:
    """Recheck a newly opened parent and classify a transient first failure."""
    opened_before = os.fstat(parent_fd)
    try:
        _require_parent(path, parent_fd)
    except OSError as exc:
        try:
            _require_parent(path, parent_fd)
            opened_after = os.fstat(parent_fd)
        except OSError as authentication_exc:
            raise RuntimeError(
                f"cannot authenticate newly opened evidence parent: {path.parent}"
            ) from authentication_exc
        except (SystemExit, RuntimeError) as integrity_exc:
            raise integrity_exc from exc
        if _identity(opened_before) != _identity(opened_after):
            raise SystemExit(
                f"newly opened evidence parent changed: {path.parent}"
            ) from exc
        raise


def _require_entry(
    path: Path, parent_fd: int, file_fd: int, *, links: int | None,
) -> os.stat_result:
    try:
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        _raise_path_access_failure(
            exc, f"cannot revalidate evidence artifact: {path}",
        )
    opened = os.fstat(file_fd)
    if (
        not stat.S_ISREG(named.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or _identity(named) != _identity(opened)
    ):
        raise SystemExit(f"evidence artifact is not the anchored regular file: {path}")
    if links is not None and int(opened.st_nlink) != links:
        raise SystemExit(f"evidence artifact has unexpected hard links: {path}")
    return opened


def _authenticate_initial_open(
    path: Path,
    parent_fd: int,
    file_fd: int,
    *,
    links: int | None,
) -> None:
    """Bind a returned descriptor to its name despite a transient first check."""
    opened_before = os.fstat(file_fd)
    try:
        _require_entry(path, parent_fd, file_fd, links=links)
        _require_parent(path, parent_fd)
    except OSError as exc:
        try:
            opened_after = _require_entry(
                path, parent_fd, file_fd, links=links,
            )
            _require_parent(path, parent_fd)
        except OSError as authentication_exc:
            raise RuntimeError(
                f"cannot authenticate newly opened evidence: {path}"
            ) from authentication_exc
        except (SystemExit, RuntimeError) as integrity_exc:
            raise integrity_exc from exc
        stable = (
            "st_mode", "st_dev", "st_ino", "st_nlink", "st_size",
            "st_mtime_ns", "st_ctime_ns",
        )
        if any(
            getattr(opened_before, field) != getattr(opened_after, field)
            for field in stable
        ):
            raise SystemExit(
                f"newly opened evidence changed during authentication: {path}"
            ) from exc
        raise


def _artifact_from_fd(file_fd: int, path: Path) -> dict[str, Any]:
    before = os.fstat(file_fd)
    digest = hashlib.sha256()
    offset = 0
    try:
        while block := os.pread(file_fd, 1024 * 1024, offset):
            digest.update(block)
            offset += len(block)
    except OSError as exc:
        try:
            after_failure = os.fstat(file_fd)
        except OSError as authentication_exc:
            raise RuntimeError(
                f"cannot authenticate evidence after failed read: {path}"
            ) from authentication_exc
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(before, field) != getattr(after_failure, field)
            for field in stable
        ):
            raise SystemExit(f"evidence artifact changed during failed read: {path}") from exc
        raise
    after = os.fstat(file_fd)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise SystemExit(f"evidence artifact changed while being read: {path}")
    return {
        "path": str(path.expanduser().resolve()),
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "sha256": digest.hexdigest(),
    }


def _read_stable_bytes_fd(file_fd: int, path: Path) -> bytes:
    """Read exact bytes while rejecting a concurrent inode-content change."""
    before = os.fstat(file_fd)
    chunks: list[bytes] = []
    offset = 0
    try:
        while block := os.pread(file_fd, 1024 * 1024, offset):
            chunks.append(block)
            offset += len(block)
    except OSError as exc:
        try:
            after_failure = os.fstat(file_fd)
        except OSError as authentication_exc:
            raise RuntimeError(
                f"cannot authenticate evidence after failed read: {path}"
            ) from authentication_exc
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(before, field) != getattr(after_failure, field)
            for field in stable
        ):
            raise SystemExit(f"evidence artifact changed during failed read: {path}") from exc
        raise
    after = os.fstat(file_fd)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise SystemExit(f"evidence artifact changed while being read: {path}")
    return b"".join(chunks)


def _require_anchored_artifact_unchanged(
    path: Path,
    *,
    artifact_path: Path | None = None,
    parent_fd: int,
    file_fd: int,
    before: dict[str, Any],
    links: int | tuple[int, ...] | None,
    allow_ctime_change: bool = False,
) -> None:
    """Authenticate an exact name, parent, inode, and content snapshot."""
    opened = _require_entry(
        path,
        parent_fd,
        file_fd,
        links=links if isinstance(links, int) else None,
    )
    if isinstance(links, tuple) and int(opened.st_nlink) not in links:
        raise SystemExit(f"evidence artifact has unexpected hard links: {path}")
    _require_parent(path, parent_fd)
    after = _artifact_from_fd(file_fd, artifact_path or path)
    identity = _publication_artifact_identity if allow_ctime_change else _artifact_identity
    if identity(before) != identity(after):
        raise SystemExit(f"evidence artifact changed during durability barrier: {path}")


def _audit_failed_artifact_mutation(
    path: Path,
    *,
    artifact_path: Path | None = None,
    parent_fd: int,
    file_fd: int,
    before: dict[str, Any],
    links: int | tuple[int, ...] | None,
    failure: BaseException,
    allow_ctime_change: bool = False,
) -> None:
    """Promote proven or unauthenticatable post-mutation drift to integrity."""
    try:
        _require_anchored_artifact_unchanged(
            path,
            artifact_path=artifact_path,
            parent_fd=parent_fd,
            file_fd=file_fd,
            before=before,
            links=links,
            allow_ctime_change=allow_ctime_change,
        )
    except OSError as authentication_exc:
        raise RuntimeError(
            f"cannot authenticate evidence after failed mutation: {path}"
        ) from authentication_exc
    except (SystemExit, RuntimeError) as integrity_exc:
        raise integrity_exc from failure


@contextmanager
def _anchored_file(
    path: Path,
    artifact_path: Path,
    *,
    durable: bool,
    links: int | None,
    parent_fd: int | None = None,
    file_fd: int | None = None,
    exit_links: int | None = None,
) -> Generator[_AnchoredFile, None, None]:
    path = path.expanduser()
    owned_parent = parent_fd is None
    if parent_fd is None:
        parent_fd = _open_parent(path)
    owned_file = file_fd is None
    if file_fd is None:
        file_fd = -1
    try:
        _require_parent(path, parent_fd)
        if owned_file:
            flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
            try:
                file_fd = os.open(path.name, flags, dir_fd=parent_fd)
            except OSError as exc:
                _raise_path_access_failure(
                    exc, f"cannot safely open regular evidence file: {path}",
                )
        _authenticate_initial_open(path, parent_fd, file_fd, links=links)
        try:
            before = _artifact_from_fd(file_fd, artifact_path)
        except BaseException as exc:
            try:
                _require_entry(path, parent_fd, file_fd, links=links)
                _require_parent(path, parent_fd)
            except OSError as authentication_exc:
                raise RuntimeError(
                    f"cannot authenticate evidence after failed initial read: {path}"
                ) from authentication_exc
            except (SystemExit, RuntimeError) as integrity_exc:
                raise integrity_exc from exc
            raise
        if durable:
            try:
                os.fsync(file_fd)
                os.fsync(parent_fd)
                after = _artifact_from_fd(file_fd, artifact_path)
                if _artifact_identity(before) != _artifact_identity(after):
                    raise SystemExit(
                        f"evidence artifact changed while being made durable: {path}"
                    )
                _require_entry(path, parent_fd, file_fd, links=links)
                _require_parent(path, parent_fd)
            except BaseException as exc:
                _audit_failed_artifact_mutation(
                    path,
                    artifact_path=artifact_path,
                    parent_fd=parent_fd,
                    file_fd=file_fd,
                    before=before,
                    links=links,
                    failure=exc,
                )
                raise
        else:
            after = _artifact_from_fd(file_fd, artifact_path)
            if _artifact_identity(before) != _artifact_identity(after):
                raise SystemExit(
                    f"evidence artifact changed while being made durable: {path}"
                )
            _require_entry(path, parent_fd, file_fd, links=links)
            _require_parent(path, parent_fd)
        anchor = _AnchoredFile(path, parent_fd, file_fd, after)
        required_exit_links = links if exit_links is None else exit_links
        try:
            yield anchor
        except BaseException as exc:
            failure_links: int | tuple[int, ...] | None = required_exit_links
            if (
                links is not None
                and required_exit_links is not None
                and links != required_exit_links
            ):
                failure_links = (links, required_exit_links)
            _audit_failed_artifact_mutation(
                path,
                artifact_path=artifact_path,
                parent_fd=parent_fd,
                file_fd=file_fd,
                before=after,
                links=failure_links,
                failure=exc,
                allow_ctime_change=True,
            )
            raise
        _require_anchored_artifact_unchanged(
            path,
            artifact_path=artifact_path,
            parent_fd=parent_fd,
            file_fd=file_fd,
            before=after,
            links=required_exit_links,
            allow_ctime_change=True,
        )
    finally:
        if owned_file and file_fd >= 0:
            os.close(file_fd)
        if owned_parent:
            os.close(parent_fd)


def _flush_fsync_file_and_parent(
    fh: _FlushableFile, path: Path, *, parent_fd: int | None = None,
) -> None:
    """Flush Python buffers, then fsync the file and its containing directory."""
    owned_parent = parent_fd is None
    if parent_fd is None:
        parent_fd = _open_parent(path)
    try:
        _require_parent(path, parent_fd)
        _require_entry(path, parent_fd, fh.fileno(), links=1)
        fh.flush()
        os.fsync(fh.fileno())
        os.fsync(parent_fd)
        _require_entry(path, parent_fd, fh.fileno(), links=1)
        _require_parent(path, parent_fd)
    finally:
        if owned_parent:
            os.close(parent_fd)


def _artifact(path: Path, *, require_file: bool) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if require_file and not resolved.is_file():
        raise SystemExit(f"decision-grade provenance requires a regular file: {resolved}")
    if not resolved.exists():
        raise SystemExit(f"artifact does not exist: {resolved}")
    file_stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size": int(file_stat.st_size),
        "mtime_ns": int(file_stat.st_mtime_ns),
        "ctime_ns": int(file_stat.st_ctime_ns),
        "device": int(file_stat.st_dev),
        "inode": int(file_stat.st_ino),
    }
    if resolved.is_file():
        digest = hashlib.sha256()
        with resolved.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def _artifact_identity(artifact: Any) -> dict[str, Any] | None:
    if not isinstance(artifact, dict):
        return None
    return {
        name: artifact.get(name)
        for name in (
            "path", "size", "mtime_ns", "ctime_ns", "device", "inode", "sha256",
        )
    }


def _publication_artifact_identity(artifact: Any) -> dict[str, Any] | None:
    """Identity preserved when a staged file is atomically renamed.

    Rename updates ctime on Linux, so ctime remains part of generic stability
    comparisons but cannot be a precommitted property of the published name.
    """
    identity = _artifact_identity(artifact)
    if identity is not None:
        identity.pop("ctime_ns")
    return identity


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{secrets.token_hex(16)}")
    with _retained_parent(path) as parent_fd:
        _write_json_staged(tmp_path, payload, parent_fd=parent_fd)
        with _anchored_file(
            tmp_path,
            tmp_path,
            durable=True,
            links=1,
            parent_fd=parent_fd,
            exit_links=2,
        ) as staged:
            if _read_json_fd(staged.file_fd, tmp_path) != payload:
                raise SystemExit("staged JSON differs from requested payload")
            _publish_no_replace(
                tmp_path,
                path,
                source=staged,
                output_parent_fd=parent_fd,
            )
        _require_parent(path, parent_fd)


@contextmanager
def _open_staged_output_file(
    path: Path, *, parent_fd: int | None = None,
) -> Generator[tuple[IO[str], int], None, None]:
    """Create a private regular file relative to a retained parent descriptor."""
    path = path.expanduser()
    owned_parent = parent_fd is None
    if parent_fd is None:
        parent_fd = _open_parent(path)
    descriptor = -1
    try:
        _require_parent(path, parent_fd)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path.name, flags, 0o666, dir_fd=parent_fd)
        except FileExistsError:
            raise
        except OSError as exc:
            try:
                created = _entry_name_exists(path, parent_fd=parent_fd)
            except OSError as authentication_exc:
                raise RuntimeError(
                    f"cannot authenticate failed staged-file creation: {path}"
                ) from authentication_exc
            if created:
                raise SystemExit(
                    f"staged-file creation had an uncertain result: {path}"
                ) from exc
            raise
        _authenticate_initial_open(path, parent_fd, descriptor, links=1)
        with os.fdopen(descriptor, "w+") as fh:
            descriptor = -1
            yield fh, parent_fd
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if owned_parent:
            os.close(parent_fd)


def _write_json_staged(
    path: Path, payload: dict[str, Any], *, parent_fd: int | None = None,
) -> None:
    """Durably stage JSON at a private path before publishing an evidence pair."""
    expected_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with _open_staged_output_file(path, parent_fd=parent_fd) as (fh, parent_fd):
        try:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            _require_entry(path, parent_fd, fh.fileno(), links=1)
            os.fsync(fh.fileno())
            os.fsync(parent_fd)
            _require_entry(path, parent_fd, fh.fileno(), links=1)
            _require_parent(path, parent_fd)
            if _read_stable_bytes_fd(fh.fileno(), path) != expected_bytes:
                raise SystemExit(f"staged JSON differs from requested payload: {path}")
        except BaseException as exc:
            try:
                _require_entry(path, parent_fd, fh.fileno(), links=1)
                _require_parent(path, parent_fd)
                if _read_stable_bytes_fd(fh.fileno(), path) != expected_bytes:
                    raise SystemExit(
                        f"staged JSON differs from requested payload: {path}"
                    )
            except OSError as authentication_exc:
                raise RuntimeError(
                    f"cannot authenticate staged JSON after failed write: {path}"
                ) from authentication_exc
            except (SystemExit, RuntimeError) as integrity_exc:
                raise integrity_exc from exc
            # Never delete a mutable name after a separate identity check.
            # Any partial or uncertain file remains as a fail-closed blocker.
            raise


def _output_lock_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.lock")


def _pending_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.tmp-pending")


def _pending_manifest_path(meta_path: Path) -> Path:
    return meta_path.with_name(f".{meta_path.name}.tmp-pending")


def _invalid_manifest_path(meta_path: Path) -> Path:
    return meta_path.with_name(f".{meta_path.name}.invalid-recovery")


@contextmanager
def _retained_parent(path: Path) -> Generator[int, None, None]:
    """Keep one authenticated containing directory open across an operation."""
    parent_fd = _open_parent(path)
    try:
        _authenticate_initial_parent(path, parent_fd)
        yield parent_fd
    finally:
        os.close(parent_fd)


@contextmanager
def _retained_manifest_parent(meta_path: Path) -> Generator[int, None, None]:
    """Authenticate a manifest parent inside its durable quarantine boundary."""
    parent_fd = _open_parent(meta_path, quarantine_meta_path=meta_path)
    try:
        with _quarantine_manifest_recovery_on_integrity_failure(
            meta_path, parent_fd=parent_fd,
        ):
            _authenticate_initial_parent(meta_path, parent_fd)
            yield parent_fd
    finally:
        os.close(parent_fd)


@contextmanager
def _retained_output_parent(
    output_path: Path, meta_path: Path, *, manifest_parent_fd: int,
) -> Generator[int, None, None]:
    """Reuse one descriptor when both evidence names share a parent."""
    if output_path.expanduser().parent == meta_path.expanduser().parent:
        yield manifest_parent_fd
        return
    with _retained_parent(output_path) as output_parent_fd:
        yield output_parent_fd
        _require_parent(output_path, output_parent_fd)


def _write_manifest_recovery_invalid_once(
    meta_path: Path,
    *,
    parent_fd: int,
    diagnostic: dict[str, Any] | None = None,
) -> None:
    marker = _invalid_manifest_path(meta_path)
    descriptor = -1
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(marker.name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            os.stat(marker.name, dir_fd=parent_fd, follow_symlinks=False)
            os.fsync(parent_fd)
            try:
                os.stat(marker.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise SystemExit("invalid-recovery marker disappeared") from exc
            return
        with os.fdopen(descriptor, "w") as fh:
            descriptor = -1
            marker_payload: dict[str, Any] = {
                "schema": "deepfin.chunk_trajectory.invalid_recovery.v1",
                "invalid": True,
            }
            if diagnostic is not None:
                marker_payload["diagnostic"] = diagnostic
            json.dump(marker_payload, fh, sort_keys=True)
            fh.write("\n")
            fh.flush()
            _require_entry(marker, parent_fd, fh.fileno(), links=1)
            os.fsync(fh.fileno())
            os.fsync(parent_fd)
            _require_entry(marker, parent_fd, fh.fileno(), links=1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _ensure_manifest_recovery_invalid(meta_path: Path, *, parent_fd: int) -> None:
    """Re-establish a blocking name after a failed marker durability barrier."""
    marker = _invalid_manifest_path(meta_path)
    last_failure: BaseException | None = None
    for _attempt in range(3):
        try:
            os.stat(marker.name, dir_fd=parent_fd, follow_symlinks=False)
            os.fsync(parent_fd)
            os.stat(marker.name, dir_fd=parent_fd, follow_symlinks=False)
            return
        except BaseException as exc:
            last_failure = exc
        try:
            _write_manifest_recovery_invalid_once(
                meta_path, parent_fd=parent_fd,
            )
        except BaseException as exc:
            last_failure = exc
    raise RuntimeError("cannot retain invalid-recovery marker") from last_failure


def _mark_manifest_recovery_invalid(
    meta_path: Path,
    *,
    parent_fd: int,
    diagnostic: dict[str, Any] | None = None,
) -> None:
    """Durably quarantine a version whose manifest integrity check failed."""
    try:
        _write_manifest_recovery_invalid_once(
            meta_path, parent_fd=parent_fd, diagnostic=diagnostic,
        )
    except BaseException:
        _ensure_manifest_recovery_invalid(meta_path, parent_fd=parent_fd)
        raise


def _mark_manifest_namespace_invalid(meta_path: Path, *, parent_fd: int) -> None:
    """Block both a retained parent and any concurrently installed alias."""
    _mark_manifest_recovery_invalid(meta_path, parent_fd=parent_fd)
    current_parent_fd = -1
    try:
        current_parent_fd = _open_parent(meta_path)
        if _identity(os.fstat(current_parent_fd)) != _identity(os.fstat(parent_fd)):
            _mark_manifest_recovery_invalid(
                meta_path, parent_fd=current_parent_fd,
            )
    except (OSError, SystemExit, RuntimeError) as alias_exc:
        raise RuntimeError(
            "cannot quarantine the current manifest namespace"
        ) from alias_exc
    finally:
        if current_parent_fd >= 0:
            os.close(current_parent_fd)


@contextmanager
def _quarantine_manifest_recovery_on_integrity_failure(
    meta_path: Path, *, parent_fd: int,
) -> Generator[None, None, None]:
    """Make a detected manifest rejection persist across later retries."""
    try:
        yield
        _require_parent(meta_path, parent_fd)
    except (FileExistsError, SystemExit, RuntimeError):
        _mark_manifest_namespace_invalid(meta_path, parent_fd=parent_fd)
        raise


def _acquire_output_lock(output_path: Path) -> IO[bytes]:
    """Reserve a bank/manifest pair against another producer process."""
    lock_path = _output_lock_path(output_path)
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NONBLOCK
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise SystemExit(f"cannot safely open output lock {lock_path}: {exc}") from exc
    handle = os.fdopen(descriptor, "a+b")
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SystemExit(f"output lock is not a regular file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SystemExit(f"another producer holds the output lock: {lock_path}") from exc
    except BaseException:
        handle.close()
        raise
    return handle


def _acquire_output_locks(
    output_path: Path, meta_path: Path,
) -> tuple[IO[bytes], ...]:
    """Reserve every destination in a bank/manifest pair in canonical order."""
    handles: list[IO[bytes]] = []
    try:
        for target in sorted(
            {output_path.expanduser().resolve(), meta_path.expanduser().resolve()},
            key=str,
        ):
            handles.append(_acquire_output_lock(target))  # noqa: PERF401
    except BaseException:
        for handle in reversed(handles):
            handle.close()
        raise
    return tuple(handles)


def _manifest_recovery_is_invalid(meta_path: Path, *, parent_fd: int) -> bool:
    marker = _invalid_manifest_path(meta_path)
    try:
        os.stat(marker.name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return False
        _ensure_manifest_recovery_invalid(meta_path, parent_fd=parent_fd)
        raise
    return True


def _entry_name_exists(path: Path, *, parent_fd: int) -> bool:
    """Check one name through its retained parent without hiding I/O errors."""
    try:
        os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return False
        raise
    return True


def _require_new_output_pair(
    output_path: Path, meta_path: Path, *, overwrite: bool,
) -> bool:
    """Resume a prepared pair, or require a completely new immutable destination."""
    if overwrite:
        raise SystemExit(
            "--overwrite is disabled for trajectory evidence; choose a new versioned --out"
        )
    with _retained_manifest_parent(meta_path) as manifest_parent_fd:
        if _manifest_recovery_is_invalid(
            meta_path, parent_fd=manifest_parent_fd,
        ):
            raise SystemExit(
                f"manifest recovery was invalidated for {output_path}; "
                "choose a new versioned --out"
            )
        with (
            _quarantine_manifest_recovery_on_integrity_failure(
                meta_path, parent_fd=manifest_parent_fd,
            ),
            _retained_output_parent(
                output_path,
                meta_path,
                manifest_parent_fd=manifest_parent_fd,
            ) as output_parent_fd,
        ):
            return _require_new_output_pair_at_parent(
                output_path,
                meta_path,
                overwrite=overwrite,
                manifest_parent_fd=manifest_parent_fd,
                output_parent_fd=output_parent_fd,
            )


def _require_new_output_pair_at_parent(
    output_path: Path,
    meta_path: Path,
    *,
    overwrite: bool,
    manifest_parent_fd: int,
    output_parent_fd: int,
) -> bool:
    if _manifest_recovery_is_invalid(meta_path, parent_fd=manifest_parent_fd):
        raise SystemExit(
            f"manifest recovery was invalidated for {output_path}; "
            "choose a new versioned --out"
        )
    if overwrite:
        raise SystemExit(
            "--overwrite is disabled for trajectory evidence; choose a new versioned --out"
        )
    with _quarantine_manifest_recovery_on_integrity_failure(
        meta_path, parent_fd=manifest_parent_fd,
    ):
        recovered = _classify_new_output_pair(
            output_path,
            meta_path,
            manifest_parent_fd=manifest_parent_fd,
            output_parent_fd=output_parent_fd,
        )
        _require_parent(output_path, output_parent_fd)
        return recovered


def _require_complete_retained_pair(
    output_path: Path,
    meta_path: Path,
    *,
    manifest_parent_fd: int,
    output_parent_fd: int,
) -> dict[str, Any]:
    """Authenticate all four permanent publication names under retained parents."""
    pending_output = _pending_output_path(output_path)
    pending_meta = _pending_manifest_path(meta_path)
    if _manifest_recovery_is_invalid(meta_path, parent_fd=manifest_parent_fd):
        raise SystemExit(f"manifest recovery was invalidated for {output_path}")
    with (
        _anchored_file(
            meta_path,
            meta_path,
            durable=True,
            links=2,
            parent_fd=manifest_parent_fd,
        ) as final_manifest,
        _anchored_file(
            pending_meta,
            meta_path,
            durable=True,
            links=2,
            parent_fd=manifest_parent_fd,
        ) as pending_manifest,
    ):
        if _identity(os.fstat(pending_manifest.file_fd)) != _identity(
            os.fstat(final_manifest.file_fd)
        ):
            raise SystemExit(
                "published and pending evidence manifests are not the same hard links"
            )
        manifest = _read_pending_manifest_fd(final_manifest.file_fd, meta_path)
        if _read_pending_manifest_fd(pending_manifest.file_fd, pending_meta) != manifest:
            raise SystemExit("published and pending evidence manifests differ")
        with (
            _matching_output(
                output_path,
                output_path,
                manifest,
                expected_links=2,
                parent_fd=output_parent_fd,
            ) as final_bank,
            _matching_output(
                pending_output,
                output_path,
                manifest,
                expected_links=2,
                parent_fd=output_parent_fd,
            ) as pending_bank,
        ):
            if _identity(os.fstat(pending_bank.file_fd)) != _identity(
                os.fstat(final_bank.file_fd)
            ):
                raise SystemExit(
                    "published and pending trajectory banks are not the same hard links"
                )
            _require_publication_guards((
                (final_bank, 2),
                (pending_bank, 2),
                (final_manifest, 2),
                (pending_manifest, 2),
            ))
    if _manifest_recovery_is_invalid(meta_path, parent_fd=manifest_parent_fd):
        raise SystemExit(f"manifest recovery was invalidated for {output_path}")
    return manifest


def _classify_new_output_pair(
    output_path: Path,
    meta_path: Path,
    *,
    manifest_parent_fd: int,
    output_parent_fd: int,
) -> bool:
    pending_output = _pending_output_path(output_path)
    pending_meta = _pending_manifest_path(meta_path)
    output_exists = _entry_name_exists(output_path, parent_fd=output_parent_fd)
    meta_exists = _entry_name_exists(meta_path, parent_fd=manifest_parent_fd)
    pending_output_exists = _entry_name_exists(
        pending_output, parent_fd=output_parent_fd,
    )
    pending_meta_exists = _entry_name_exists(
        pending_meta, parent_fd=manifest_parent_fd,
    )
    if output_exists and meta_exists and pending_output_exists and pending_meta_exists:
        with (
            _quarantine_manifest_recovery_on_integrity_failure(
                meta_path, parent_fd=manifest_parent_fd,
            ),
            _anchored_file(
                meta_path,
                meta_path,
                durable=True,
                links=2,
                parent_fd=manifest_parent_fd,
            ) as final_manifest,
        ):
            try:
                pending_stat = os.stat(
                    pending_meta.name,
                    dir_fd=final_manifest.parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                _raise_path_access_failure(
                    exc,
                    "cannot inspect pending evidence manifest during recovery",
                )
            final_stat = os.fstat(final_manifest.file_fd)
            if (
                not stat.S_ISREG(pending_stat.st_mode)
                or _identity(pending_stat) != _identity(final_stat)
                or int(pending_stat.st_nlink) != 2
                or int(final_stat.st_nlink) != 2
            ):
                raise SystemExit(
                    "published and pending evidence manifests are not the same "
                    "recovery hard links"
                )
            _require_entry(
                meta_path,
                final_manifest.parent_fd,
                final_manifest.file_fd,
                links=2,
            )
            manifest = _read_pending_manifest_fd(final_manifest.file_fd, meta_path)
            _require_entry(
                pending_meta,
                final_manifest.parent_fd,
                final_manifest.file_fd,
                links=2,
            )
            with (
                _matching_output(
                    output_path,
                    output_path,
                    manifest,
                    expected_links=2,
                    parent_fd=output_parent_fd,
                ) as final_bank,
                _matching_output(
                    pending_output,
                    output_path,
                    manifest,
                    expected_links=2,
                    parent_fd=output_parent_fd,
                ) as pending_bank,
            ):
                if _identity(os.fstat(pending_bank.file_fd)) != _identity(
                    os.fstat(final_bank.file_fd)
                ):
                    raise SystemExit(
                        "published and pending trajectory banks are not the same hard links"
                    )
                _require_publication_guards((
                    (final_bank, 2),
                    (pending_bank, 2),
                    (final_manifest, 2),
                ))
            if _read_pending_manifest_fd(final_manifest.file_fd, meta_path) != manifest:
                raise SystemExit("published evidence manifest changed during recovery")
        _require_complete_retained_pair(
            output_path,
            meta_path,
            manifest_parent_fd=manifest_parent_fd,
            output_parent_fd=output_parent_fd,
        )
        return True
    if (
        output_exists
        and not meta_exists
        and pending_output_exists
        and pending_meta_exists
    ):
        with (
            _quarantine_manifest_recovery_on_integrity_failure(
                meta_path, parent_fd=manifest_parent_fd,
            ),
            _durably_read_pending_manifest(
                pending_meta, parent_fd=manifest_parent_fd,
            ) as (manifest, source),
            _matching_output(
                output_path,
                output_path,
                manifest,
                expected_links=2,
                parent_fd=output_parent_fd,
            ) as final_bank,
            _matching_output(
                pending_output,
                output_path,
                manifest,
                expected_links=2,
                parent_fd=output_parent_fd,
            ) as pending_bank,
        ):
            if _identity(os.fstat(pending_bank.file_fd)) != _identity(
                os.fstat(final_bank.file_fd)
            ):
                raise SystemExit(
                    "published and pending trajectory banks are not the same hard links"
                )
            _publish_no_replace(
                pending_meta,
                meta_path,
                source=source,
                guards=((final_bank, 2), (pending_bank, 2)),
                output_parent_fd=manifest_parent_fd,
            )
        _require_complete_retained_pair(
            output_path,
            meta_path,
            manifest_parent_fd=manifest_parent_fd,
            output_parent_fd=output_parent_fd,
        )
        return True
    if (
        not output_exists
        and not meta_exists
        and pending_output_exists
        and pending_meta_exists
    ):
        with (
            _quarantine_manifest_recovery_on_integrity_failure(
                meta_path, parent_fd=manifest_parent_fd,
            ),
            _durably_read_pending_manifest(
                pending_meta, parent_fd=manifest_parent_fd,
            ) as (manifest, source),
        ):
            _require_output_matches_manifest(
                pending_output,
                output_path,
                manifest,
                expected_links=1,
                parent_fd=output_parent_fd,
            )
            published = _publish_output(
                pending_output,
                output_path,
                expected_artifact=manifest.get("output"),
                parent_fd=output_parent_fd,
            )
            if (
                _publication_artifact_identity(published)
                != _publication_artifact_identity(manifest.get("output"))
            ):
                raise RuntimeError("recovered trajectory bank differs from its manifest")
            with (
                _matching_output(
                    output_path,
                    output_path,
                    manifest,
                    expected_links=2,
                    parent_fd=output_parent_fd,
                ) as final_bank,
                _matching_output(
                    pending_output,
                    output_path,
                    manifest,
                    expected_links=2,
                    parent_fd=output_parent_fd,
                ) as pending_bank,
            ):
                if _identity(os.fstat(pending_bank.file_fd)) != _identity(
                    os.fstat(final_bank.file_fd)
                ):
                    raise SystemExit(
                        "published and pending trajectory banks are not the same hard links"
                    )
                _publish_no_replace(
                    pending_meta,
                    meta_path,
                    source=source,
                    guards=((final_bank, 2), (pending_bank, 2)),
                    output_parent_fd=manifest_parent_fd,
                )
        _require_complete_retained_pair(
            output_path,
            meta_path,
            manifest_parent_fd=manifest_parent_fd,
            output_parent_fd=output_parent_fd,
        )
        return True
    if any((output_exists, meta_exists, pending_output_exists, pending_meta_exists)):
        raise SystemExit(
            f"refusing to replace immutable or incomplete evidence for {output_path}; "
            "choose a new versioned --out, or retain a fully prepared pending pair"
        )
    return False


def _link_open_file(file_fd: int, parent_fd: int, name: str) -> None:
    """Hard-link an exact open inode through procfs, never a mutable source name."""
    descriptor_path = Path(f"/proc/self/fd/{file_fd}")
    # Inaccessible procfs is a retryable publication-mechanism failure.  A
    # readable mapping that names the wrong inode remains an integrity failure.
    descriptor_stat = descriptor_path.stat()
    if _identity(descriptor_stat) != _identity(os.fstat(file_fd)):
        raise SystemExit(f"open link source changed: {descriptor_path}")
    os.link(descriptor_path, name, dst_dir_fd=parent_fd, follow_symlinks=True)


def _publish_anchored(
    source: _AnchoredFile,
    output_path: Path,
    guards: tuple[tuple[_AnchoredFile, int], ...],
    *,
    output_parent_fd: int | None = None,
) -> None:
    output_path = output_path.expanduser()
    _require_entry(source.path, source.parent_fd, source.file_fd, links=1)
    _require_parent(source.path, source.parent_fd)
    artifact_path = source.artifact.get("path")
    if not isinstance(artifact_path, str):
        raise RuntimeError("authenticated publication source lacks its artifact path")
    current = _artifact_from_fd(source.file_fd, Path(artifact_path))
    if _publication_artifact_identity(current) != _publication_artifact_identity(
        source.artifact
    ):
        raise SystemExit(f"staged evidence changed before publication: {source.path}")
    owned_output_parent = output_parent_fd is None
    if output_parent_fd is None:
        output_parent_fd = _open_parent(output_path)
    link_attempted = False
    linked = False
    try:
        _require_parent(output_path, output_parent_fd)
        _require_publication_guards(guards)
        try:
            link_attempted = True
            _link_open_file(source.file_fd, output_parent_fd, output_path.name)
        except FileExistsError as exc:
            raise _ImmutableEvidenceExistsError(
                f"refusing to replace immutable evidence at {output_path}"
            ) from exc
        linked = True
        _require_entry(output_path, output_parent_fd, source.file_fd, links=2)
        os.fsync(output_parent_fd)
        _require_publication_guards(guards)
        _require_parent(output_path, output_parent_fd)
        _require_parent(source.path, source.parent_fd)
        _require_entry(source.path, source.parent_fd, source.file_fd, links=2)
        _require_entry(output_path, output_parent_fd, source.file_fd, links=2)
        current = _artifact_from_fd(source.file_fd, Path(artifact_path))
        if _publication_artifact_identity(current) != _publication_artifact_identity(
            source.artifact
        ):
            raise SystemExit(f"staged evidence changed during publication: {source.path}")
        _require_entry(source.path, source.parent_fd, source.file_fd, links=2)
    except OSError as exc:
        if link_attempted and not linked:
            try:
                linked = _entry_name_exists(
                    output_path, parent_fd=output_parent_fd,
                )
            except OSError as authentication_exc:
                raise RuntimeError(
                    "cannot authenticate destination after failed publication link"
                ) from authentication_exc
        if linked:
            try:
                _require_publication_guards(guards)
                _require_parent(source.path, source.parent_fd)
                _require_parent(output_path, output_parent_fd)
                _require_entry(
                    source.path, source.parent_fd, source.file_fd, links=2,
                )
                _require_entry(
                    output_path, output_parent_fd, source.file_fd, links=2,
                )
                current = _artifact_from_fd(source.file_fd, Path(artifact_path))
                if _publication_artifact_identity(
                    current
                ) != _publication_artifact_identity(source.artifact):
                    raise SystemExit(
                        f"staged evidence changed during failed publication: {source.path}"
                    )
            except OSError as authentication_exc:
                raise RuntimeError(
                    "cannot authenticate evidence after failed publication mutation"
                ) from authentication_exc
            except (SystemExit, RuntimeError) as integrity_exc:
                raise integrity_exc from exc
        raise
    finally:
        if owned_output_parent:
            os.close(output_parent_fd)


def _require_publication_guards(
    guards: tuple[tuple[_AnchoredFile, int], ...],
) -> None:
    for guard, links in guards:
        _require_entry(guard.path, guard.parent_fd, guard.file_fd, links=links)
        _require_parent(guard.path, guard.parent_fd)
        artifact_path = guard.artifact.get("path")
        if not isinstance(artifact_path, str):
            raise RuntimeError("publication guard lacks its artifact path")
        current = _artifact_from_fd(guard.file_fd, Path(artifact_path))
        if _publication_artifact_identity(current) != _publication_artifact_identity(
            guard.artifact
        ):
            raise SystemExit(f"guarded evidence changed: {guard.path}")


def _publish_no_replace(
    tmp_path: Path,
    output_path: Path,
    *,
    source: _AnchoredFile | None = None,
    guards: tuple[tuple[_AnchoredFile, int], ...] = (),
    output_parent_fd: int | None = None,
) -> None:
    """Hard-link without replacement and retain the recovery name as a witness.

    The caller must fsync the staged file's bytes before entering this helper.
    The pending hard link is deliberately permanent: deleting a mutable name
    after a separate identity check can delete an attacker-substituted file.
    """
    if source is not None:
        if source.path != tmp_path.expanduser():
            raise RuntimeError("publication source does not match its authenticated path")
        _require_publication_guards(guards)
        _publish_anchored(
            source,
            output_path,
            guards,
            output_parent_fd=output_parent_fd,
        )
        _require_publication_guards(guards)
        return
    with _anchored_file(
        tmp_path, tmp_path, durable=False, links=1, exit_links=2,
    ) as opened_source:
        _require_publication_guards(guards)
        _publish_anchored(
            opened_source,
            output_path,
            guards,
            output_parent_fd=output_parent_fd,
        )
        _require_publication_guards(guards)


def _publish_output(
    tmp_path: Path,
    output_path: Path,
    *,
    expected_artifact: Any = None,
    parent_fd: int | None = None,
) -> dict[str, Any]:
    """Publish exactly the private bytes whose identity the manifest records."""
    with _anchored_file(
        tmp_path,
        output_path,
        durable=True,
        links=1,
        parent_fd=parent_fd,
        exit_links=2,
    ) as source:
        expected = source.artifact
        if (
            expected_artifact is not None
            and _publication_artifact_identity(expected)
            != _publication_artifact_identity(expected_artifact)
        ):
            raise RuntimeError("private trajectory bank differs from its manifest")
        _publish_no_replace(
            tmp_path,
            output_path,
            source=source,
            output_parent_fd=source.parent_fd,
        )
        published = _artifact_from_fd(source.file_fd, output_path)
    if _publication_artifact_identity(published) != _publication_artifact_identity(
        expected
    ):
        raise RuntimeError("published trajectory bank differs from its private output")
    return published


def _prepared_output_artifact(tmp_path: Path, output_path: Path) -> dict[str, Any]:
    private = _artifact(tmp_path, require_file=True)
    return {**private, "path": str(output_path.expanduser().resolve())}


def _durably_prepare_output_artifact(
    fh: _FlushableFile,
    tmp_path: Path,
    output_path: Path,
    *,
    parent_fd: int | None = None,
) -> dict[str, Any]:
    """Make staged bytes and their name durable before snapshotting identity."""
    owned_parent = parent_fd is None
    if parent_fd is None:
        parent_fd = _open_parent(tmp_path)
    reader_fd = -1
    try:
        _flush_fsync_file_and_parent(fh, tmp_path, parent_fd=parent_fd)
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        reader_fd = os.open(tmp_path.name, flags, dir_fd=parent_fd)
        _require_entry(tmp_path, parent_fd, reader_fd, links=1)
        if _identity(os.fstat(reader_fd)) != _identity(os.fstat(fh.fileno())):
            raise SystemExit(f"pending trajectory bank changed: {tmp_path}")
        private = _artifact_from_fd(reader_fd, tmp_path)
        _require_parent(tmp_path, parent_fd)
        return {**private, "path": str(output_path.expanduser().resolve())}
    finally:
        if reader_fd >= 0:
            os.close(reader_fd)
        if owned_parent:
            os.close(parent_fd)


def _durably_prepare_existing_output_artifact(
    tmp_path: Path, output_path: Path, *, expected_links: int | None = None,
) -> dict[str, Any]:
    """Durably snapshot a closed staged output, including recovery inputs."""
    with _anchored_file(
        tmp_path, output_path, durable=True, links=expected_links,
    ) as source:
        return source.artifact


def _durably_prepare_anchored_output_artifact(
    file_fd: int, parent_fd: int, tmp_path: Path, output_path: Path,
) -> dict[str, Any]:
    """Durably snapshot a retained collection file without reopening its path."""
    _require_parent(tmp_path, parent_fd)
    _require_entry(tmp_path, parent_fd, file_fd, links=1)
    os.fsync(file_fd)
    os.fsync(parent_fd)
    artifact = _artifact_from_fd(file_fd, tmp_path)
    _require_entry(tmp_path, parent_fd, file_fd, links=1)
    _require_parent(tmp_path, parent_fd)
    return {**artifact, "path": str(output_path.expanduser().resolve())}


def _write_invalid_recovery_diagnostic(
    pending_output: Path,
    output_path: Path,
    meta_path: Path,
    payload: dict[str, Any],
    *,
    file_fd: int,
    parent_fd: int,
    expected_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve failure evidence without creating a recoverable manifest."""
    if pending_output.expanduser().parent != meta_path.expanduser().parent:
        raise RuntimeError("failure diagnostic does not share the retained bank parent")
    try:
        _require_entry(pending_output, parent_fd, file_fd, links=1)
        os.fsync(file_fd)
        os.fsync(parent_fd)
        artifact = {
            **_artifact_from_fd(file_fd, pending_output),
            "path": str(output_path.expanduser().resolve()),
        }
        _require_entry(pending_output, parent_fd, file_fd, links=1)
        if (
            expected_artifact is not None
            and _publication_artifact_identity(artifact)
            != _publication_artifact_identity(expected_artifact)
        ):
            raise RuntimeError("retained failure bank changed after collection")
        _mark_manifest_recovery_invalid(
            meta_path,
            parent_fd=parent_fd,
            diagnostic={**payload, "output": artifact},
        )
    except BaseException:
        _mark_manifest_recovery_invalid(meta_path, parent_fd=parent_fd)
        raise
    return artifact


def _read_json_fd(file_fd: int, path: Path, *, role: str = "staged evidence") -> Any:
    """Read stable JSON bytes from an exact retained file descriptor."""
    before = _artifact_from_fd(file_fd, path)
    try:
        chunks: list[bytes] = []
        offset = 0
        while block := os.pread(file_fd, 1024 * 1024, offset):
            chunks.append(block)
            offset += len(block)
        payload = json.loads(b"".join(chunks))
    except OSError as exc:
        try:
            after_failure = _artifact_from_fd(file_fd, path)
        except OSError as authentication_exc:
            raise RuntimeError(
                f"cannot authenticate {role} after failed read: {path}"
            ) from authentication_exc
        if _artifact_identity(before) != _artifact_identity(after_failure):
            raise SystemExit(f"{role} changed during failed read: {path}") from exc
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{role} is not valid JSON: {path}") from exc
    after = _artifact_from_fd(file_fd, path)
    if _artifact_identity(before) != _artifact_identity(after):
        raise SystemExit(f"{role} changed while being read: {path}")
    return payload


def _read_pending_manifest_fd(file_fd: int, path: Path) -> dict[str, Any]:
    payload = _read_json_fd(file_fd, path, role="pending evidence manifest")
    if not isinstance(payload, dict) or payload.get("schema") != CHUNK_TRAJECTORY_SCHEMA:
        raise SystemExit(f"pending evidence manifest has the wrong schema: {path}")
    if payload.get("complete") is not True or not isinstance(payload.get("output"), dict):
        raise SystemExit(f"pending evidence manifest is incomplete: {path}")
    return payload


@contextmanager
def _durably_read_pending_manifest(
    path: Path, *, parent_fd: int | None = None,
) -> Generator[tuple[dict[str, Any], _AnchoredFile], None, None]:
    """Keep the exact synced manifest open through its recovery publication."""
    with _anchored_file(
        path,
        path,
        durable=True,
        links=1,
        parent_fd=parent_fd,
        exit_links=2,
    ) as source:
        manifest = _read_pending_manifest_fd(source.file_fd, path)
        after = _artifact_from_fd(source.file_fd, path)
        if _artifact_identity(source.artifact) != _artifact_identity(after):
            raise SystemExit(
                f"pending evidence manifest changed while being made durable: {path}"
            )
        _require_entry(path, source.parent_fd, source.file_fd, links=1)
        _require_parent(path, source.parent_fd)
        yield manifest, source


@contextmanager
def _matching_output(
    candidate_path: Path,
    output_path: Path,
    manifest: dict[str, Any],
    *,
    expected_links: int,
    parent_fd: int | None = None,
) -> Generator[_AnchoredFile, None, None]:
    expected = manifest.get("output")
    if not isinstance(expected, dict):
        raise SystemExit("pending evidence manifest lacks an output artifact")
    with _anchored_file(
        candidate_path,
        output_path,
        durable=True,
        links=expected_links,
        parent_fd=parent_fd,
    ) as source:
        if _publication_artifact_identity(
            source.artifact
        ) != _publication_artifact_identity(expected):
            raise SystemExit("pending trajectory bank does not match its manifest")
        yield source


def _require_output_matches_manifest(
    candidate_path: Path,
    output_path: Path,
    manifest: dict[str, Any],
    *,
    expected_links: int,
    parent_fd: int | None = None,
) -> dict[str, Any]:
    with _matching_output(
        candidate_path,
        output_path,
        manifest,
        expected_links=expected_links,
        parent_fd=parent_fd,
    ) as source:
        return source.artifact


def _publish_evidence_pair(
    pending_output: Path,
    output_path: Path,
    pending_meta: Path,
    meta_path: Path,
    manifest: dict[str, Any],
    *,
    retained_output_fd: int | None = None,
    retained_output_parent_fd: int | None = None,
    retained_output_artifact: dict[str, Any] | None = None,
) -> None:
    """Prepare both artifacts before publishing either; retain recovery state."""
    retained = (
        retained_output_fd,
        retained_output_parent_fd,
        retained_output_artifact,
    )
    if any(item is not None for item in retained) and not all(
        item is not None for item in retained
    ):
        raise RuntimeError("retained publication evidence is incomplete")
    if retained_output_parent_fd is not None:
        if output_path.expanduser().parent != meta_path.expanduser().parent:
            raise RuntimeError(
                "retained bank and manifest publication require one shared parent"
            )
        with _quarantine_manifest_recovery_on_integrity_failure(
            meta_path, parent_fd=retained_output_parent_fd,
        ):
            _publish_evidence_pair_at_parents(
                pending_output,
                output_path,
                pending_meta,
                meta_path,
                manifest,
                manifest_parent_fd=retained_output_parent_fd,
                output_parent_fd=retained_output_parent_fd,
                retained_output_fd=retained_output_fd,
                retained_output_artifact=retained_output_artifact,
            )
            _require_parent(output_path, retained_output_parent_fd)
            _require_parent(meta_path, retained_output_parent_fd)
        return
    with (
        _retained_manifest_parent(meta_path) as manifest_parent_fd,
        _quarantine_manifest_recovery_on_integrity_failure(
            meta_path, parent_fd=manifest_parent_fd,
        ),
        _retained_output_parent(
            output_path,
            meta_path,
            manifest_parent_fd=manifest_parent_fd,
        ) as output_parent_fd,
    ):
        _publish_evidence_pair_at_parents(
            pending_output,
            output_path,
            pending_meta,
            meta_path,
            manifest,
            manifest_parent_fd=manifest_parent_fd,
            output_parent_fd=output_parent_fd,
        )


def _publish_evidence_pair_at_parents(
    pending_output: Path,
    output_path: Path,
    pending_meta: Path,
    meta_path: Path,
    manifest: dict[str, Any],
    *,
    manifest_parent_fd: int,
    output_parent_fd: int,
    retained_output_fd: int | None = None,
    retained_output_artifact: dict[str, Any] | None = None,
) -> None:
    """Publish while keeping the producer's exact collected bank authenticated."""
    with _anchored_file(
        pending_output,
        output_path,
        durable=False,
        links=1,
        parent_fd=output_parent_fd,
        file_fd=retained_output_fd,
        exit_links=2,
    ) as collected_bank:
        if retained_output_artifact is not None and (
            _publication_artifact_identity(collected_bank.artifact)
            != _publication_artifact_identity(retained_output_artifact)
        ):
            raise RuntimeError("retained trajectory bank changed after collection")
        if (
            _publication_artifact_identity(collected_bank.artifact)
            != _publication_artifact_identity(manifest.get("output"))
        ):
            raise RuntimeError("private trajectory bank differs from its manifest")
        _write_json_staged(
            pending_meta, manifest, parent_fd=manifest_parent_fd,
        )
        with _anchored_file(
            pending_output,
            output_path,
            durable=True,
            links=1,
            parent_fd=output_parent_fd,
            file_fd=collected_bank.file_fd,
            exit_links=2,
        ) as pending_bank:
            if (
                _publication_artifact_identity(pending_bank.artifact)
                != _publication_artifact_identity(collected_bank.artifact)
            ):
                raise SystemExit("collected trajectory bank changed before publication")
            with _durably_read_pending_manifest(
                pending_meta, parent_fd=manifest_parent_fd,
            ) as (
                staged_manifest,
                pending_manifest,
            ):
                if staged_manifest != manifest:
                    raise SystemExit(
                        "staged evidence manifest differs from requested manifest"
                    )
                _publish_no_replace(
                    pending_output,
                    output_path,
                    source=pending_bank,
                    output_parent_fd=output_parent_fd,
                )
                published = _artifact_from_fd(pending_bank.file_fd, output_path)
                if (
                    _publication_artifact_identity(published)
                    != _publication_artifact_identity(manifest.get("output"))
                ):
                    raise RuntimeError(
                        "published trajectory bank differs from its manifest"
                    )
                with _matching_output(
                    output_path,
                    output_path,
                    manifest,
                    expected_links=2,
                    parent_fd=output_parent_fd,
                ) as final_bank:
                    if _identity(os.fstat(pending_bank.file_fd)) != _identity(
                        os.fstat(final_bank.file_fd)
                    ):
                        raise SystemExit(
                            "published and pending trajectory banks are not the same hard links"
                        )
                    _publish_no_replace(
                        pending_meta,
                        meta_path,
                        source=pending_manifest,
                        guards=((final_bank, 2), (pending_bank, 2)),
                        output_parent_fd=manifest_parent_fd,
                    )
        _require_complete_retained_pair(
            output_path,
            meta_path,
            manifest_parent_fd=manifest_parent_fd,
            output_parent_fd=output_parent_fd,
        )


def _git_ignored_or_outside(path: Path, repo_root: Path) -> bool:
    """Whether creating a path cannot dirty the producer checkout."""
    root = repo_root.expanduser().resolve()
    for candidate in {path.expanduser().absolute(), path.expanduser().resolve()}:
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        try:
            result = subprocess.run(
                [
                    "git", "-C", str(root), "check-ignore", "-q", "--no-index",
                    "--", str(relative),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return False
        if result.returncode != 0:
            return False
    return True


def _require_safe_output_paths(
    output_path: Path,
    meta_path: Path,
    *,
    protected_files: list[Path],
    protected_directories: list[Path],
    repo_root: Path | None = None,
) -> None:
    """Refuse destructive aliases before any evidence artifact is created."""
    destinations = {
        output_path.expanduser().resolve(),
        meta_path.expanduser().resolve(),
    }
    if any(reserved_output_path(path) for path in (output_path, meta_path)):
        raise SystemExit(
            "--out or its manifest must not use the output lock/staging namespace"
        )
    invalid_manifest = _invalid_manifest_path(meta_path).expanduser()
    outputs = {
        *destinations,
        _output_lock_path(output_path).expanduser().resolve(),
        _output_lock_path(meta_path).expanduser().resolve(),
        _pending_output_path(output_path).expanduser().resolve(),
        _pending_manifest_path(meta_path).expanduser().resolve(),
        invalid_manifest.absolute(),
        invalid_manifest.resolve(),
    }
    root = repo_root or Path(__file__).resolve().parents[1]
    if any(repo_controlled_output(output, root) for output in outputs):
        raise SystemExit(
            "--out or its manifest must not overwrite a tracked or repository-control path"
        )
    if any(not _git_ignored_or_outside(output, root) for output in outputs):
        raise SystemExit(
            "--out, its manifest, locks, and staging files must be Git-ignored "
            "or outside the repository"
        )
    inputs = {path.expanduser().resolve() for path in protected_files}
    if outputs & inputs:
        raise SystemExit("--out or its manifest aliases a consumed input artifact")
    for output in outputs:
        for directory in protected_directories:
            resolved = directory.expanduser().resolve()
            try:
                output.relative_to(resolved)
            except ValueError:
                continue
            raise SystemExit(
                "--out or its manifest must not be inside a Syzygy or authenticated "
                "replay-snapshot directory"
            )


def recover_publication_cli(argv: list[str], *, repo_root: Path) -> None:
    """Finish a staged bank/manifest pair without importing the search runtime."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--recover-publication", action="store_true")
    parser.add_argument("--write-preregistration", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("runs/backtest/chunk_trajectory.jsonl"),
    )
    args = parser.parse_args(argv)
    if not args.recover_publication:
        raise SystemExit("internal recovery dispatch requires --recover-publication")
    if args.write_preregistration is not None:
        raise SystemExit(
            "--recover-publication and --write-preregistration are mutually exclusive"
        )
    meta_path = Path(str(args.out) + ".meta.json")
    _require_safe_output_paths(
        args.out,
        meta_path,
        protected_files=[],
        protected_directories=[],
        repo_root=repo_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    recovery_locks = _acquire_output_locks(args.out, meta_path)
    try:
        recovered = _require_new_output_pair(
            args.out, meta_path, overwrite=bool(args.overwrite),
        )
    finally:
        for recovery_lock in reversed(recovery_locks):
            recovery_lock.close()
    if not recovered:
        raise SystemExit(f"no fully prepared evidence pair to recover for {args.out}")
    print(f"[traj] recovered evidence pair -> {args.out}")
    print(f"[traj] provenance -> {meta_path}")
