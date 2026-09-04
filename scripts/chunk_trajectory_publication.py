"""Lightweight, recoverable publication for chunk-trajectory evidence pairs.

This module intentionally imports no chess, model, evaluator, or native-extension
code.  The producer can therefore finish an interrupted publication before the
search runtime is available.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import IO, Any, Protocol

from scripts.repo_output_guard import repo_controlled_output, reserved_output_path


CHUNK_TRAJECTORY_SCHEMA = "deepfin.chunk_trajectory.v6"


class _ImmutableEvidenceExistsError(RuntimeError):
    """A no-clobber publication destination already exists."""


class _FlushableFile(Protocol):
    def flush(self) -> None: ...

    def fileno(self) -> int: ...


def _fsync_directory(path: Path) -> None:
    """Fsync completed directory-entry transitions; propagate any failure."""
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _flush_fsync_file_and_parent(fh: _FlushableFile, path: Path) -> None:
    """Flush Python buffers, then fsync the file and its containing directory."""
    fh.flush()
    os.fsync(fh.fileno())
    _fsync_directory(path.parent)


def _unlink_durable(path: Path) -> None:
    """Remove one name and fsync its containing directory before returning."""
    path.unlink()
    _fsync_directory(path.parent)


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
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _write_json_staged(tmp_path, payload)
    try:
        _publish_no_replace(tmp_path, path)
    except _ImmutableEvidenceExistsError:
        _unlink_durable(tmp_path)
        raise


def _write_json_staged(path: Path, payload: dict[str, Any]) -> None:
    """Durably stage JSON at a private path before publishing an evidence pair."""
    created = False
    try:
        with path.open("x") as fh:
            created = True
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        if created and path.exists():
            _unlink_durable(path)
        raise
    # Keep the complete staged file if this sync fails: it remains the only
    # recovery source and must not be consumed by failure cleanup.
    _fsync_directory(path.parent)


def _output_lock_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.lock")


def _pending_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.tmp-pending")


def _pending_manifest_path(meta_path: Path) -> Path:
    return meta_path.with_name(f".{meta_path.name}.tmp-pending")


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


def _require_new_output_pair(
    output_path: Path, meta_path: Path, *, overwrite: bool,
) -> bool:
    """Resume a prepared pair, or require a completely new immutable destination."""
    if overwrite:
        raise SystemExit(
            "--overwrite is disabled for trajectory evidence; choose a new versioned --out"
        )
    pending_output = _pending_output_path(output_path)
    pending_meta = _pending_manifest_path(meta_path)
    if (
        output_path.exists()
        and meta_path.exists()
        and not os.path.lexists(pending_output)
        and pending_meta.exists()
    ):
        manifest = _require_same_manifest_hard_link(meta_path, pending_meta)
        _require_output_matches_manifest(output_path, output_path, manifest)
        if _require_same_manifest_hard_link(meta_path, pending_meta) != manifest:
            raise SystemExit("published evidence manifest changed during recovery")
        _durably_prepare_existing_output_artifact(meta_path, meta_path)
        if _require_same_manifest_hard_link(meta_path, pending_meta) != manifest:
            raise SystemExit("published evidence manifest changed during recovery")
        _unlink_durable(pending_meta)
        return True
    if output_path.exists() and not meta_path.exists() and pending_meta.exists():
        manifest = _durably_read_pending_manifest(pending_meta)
        if pending_output.exists():
            _require_output_matches_manifest(pending_output, output_path, manifest)
        _require_output_matches_manifest(output_path, output_path, manifest)
        if pending_output.exists():
            _unlink_durable(pending_output)
        _publish_no_replace(pending_meta, meta_path)
        return True
    if (
        not output_path.exists()
        and not meta_path.exists()
        and pending_output.exists()
        and pending_meta.exists()
    ):
        manifest = _durably_read_pending_manifest(pending_meta)
        _require_output_matches_manifest(pending_output, output_path, manifest)
        published = _publish_output(pending_output, output_path)
        if _publication_artifact_identity(published) != _publication_artifact_identity(
            manifest.get("output")
        ):
            raise RuntimeError("recovered trajectory bank differs from its manifest")
        _publish_no_replace(pending_meta, meta_path)
        return True
    if any(
        path.exists() for path in (output_path, meta_path, pending_output, pending_meta)
    ):
        raise SystemExit(
            f"refusing to replace immutable or incomplete evidence for {output_path}; "
            "choose a new versioned --out, or retain a fully prepared pending pair"
        )
    return False


def _publish_no_replace(tmp_path: Path, output_path: Path) -> None:
    """Hard-link without replacement, then consume the recovery name in order.

    The caller must fsync the staged file's bytes before entering this helper.
    Each directory sync is kept separate even when both names share a parent:
    a crash must not lose the published name after the recovery name is removed.
    """
    try:
        os.link(tmp_path, output_path)
    except FileExistsError as exc:
        raise _ImmutableEvidenceExistsError(
            f"refusing to replace immutable evidence at {output_path}"
        ) from exc
    _fsync_directory(output_path.parent)
    _unlink_durable(tmp_path)


def _publish_output(tmp_path: Path, output_path: Path) -> dict[str, Any]:
    """Publish exactly the private bytes whose identity the manifest records."""
    expected = _durably_prepare_existing_output_artifact(tmp_path, output_path)
    _publish_no_replace(tmp_path, output_path)
    published = _artifact(output_path, require_file=True)
    if _publication_artifact_identity(published) != _publication_artifact_identity(
        expected
    ):
        raise RuntimeError("published trajectory bank differs from its private output")
    return published


def _prepared_output_artifact(tmp_path: Path, output_path: Path) -> dict[str, Any]:
    private = _artifact(tmp_path, require_file=True)
    return {**private, "path": str(output_path.expanduser().resolve())}


def _durably_prepare_output_artifact(
    fh: _FlushableFile, tmp_path: Path, output_path: Path,
) -> dict[str, Any]:
    """Make staged bytes and their name durable before snapshotting identity."""
    _flush_fsync_file_and_parent(fh, tmp_path)
    return _prepared_output_artifact(tmp_path, output_path)


def _durably_prepare_existing_output_artifact(
    tmp_path: Path, output_path: Path,
) -> dict[str, Any]:
    """Durably snapshot a closed staged output, including recovery inputs."""
    with tmp_path.open("rb") as fh:
        return _durably_prepare_output_artifact(fh, tmp_path, output_path)


def _read_pending_manifest(path: Path) -> dict[str, Any]:
    before = _artifact(path, require_file=True)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"pending evidence manifest is not valid JSON: {path}") from exc
    after = _artifact(path, require_file=True)
    if _artifact_identity(before) != _artifact_identity(after):
        raise SystemExit(f"pending evidence manifest changed while being read: {path}")
    if not isinstance(payload, dict) or payload.get("schema") != CHUNK_TRAJECTORY_SCHEMA:
        raise SystemExit(f"pending evidence manifest has the wrong schema: {path}")
    if payload.get("complete") is not True or not isinstance(payload.get("output"), dict):
        raise SystemExit(f"pending evidence manifest is incomplete: {path}")
    return payload


def _durably_read_pending_manifest(path: Path) -> dict[str, Any]:
    """Sync a recovery manifest and then authenticate a stable reread."""
    before = _artifact(path, require_file=True)
    with path.open("rb") as fh:
        _flush_fsync_file_and_parent(fh, path)
    manifest = _read_pending_manifest(path)
    after = _artifact(path, require_file=True)
    if _artifact_identity(before) != _artifact_identity(after):
        raise SystemExit(
            f"pending evidence manifest changed while being made durable: {path}"
        )
    return manifest


def _manifest_name_identity(path: Path, *, role: str) -> tuple[int, int]:
    """Require a manifest name to identify a regular file without a symlink."""
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise SystemExit(f"cannot inspect {role}: {path}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise SystemExit(f"{role} is not a regular file: {path}")
    return int(file_stat.st_dev), int(file_stat.st_ino)


def _require_same_manifest_hard_link(
    meta_path: Path, pending_meta: Path,
) -> dict[str, Any]:
    """Authenticate the final-manifest fsync-failure recovery marker."""
    published_before = _manifest_name_identity(
        meta_path, role="published evidence manifest",
    )
    pending_before = _manifest_name_identity(
        pending_meta, role="pending evidence manifest",
    )
    if published_before != pending_before:
        raise SystemExit(
            "published and pending evidence manifests are not the same recovery hard link"
        )
    pending_manifest = _read_pending_manifest(pending_meta)
    published_manifest = _read_pending_manifest(meta_path)
    published_after = _manifest_name_identity(
        meta_path, role="published evidence manifest",
    )
    pending_after = _manifest_name_identity(
        pending_meta, role="pending evidence manifest",
    )
    if (
        published_after != published_before
        or pending_after != pending_before
        or published_after != pending_after
        or published_manifest != pending_manifest
    ):
        raise SystemExit("published evidence manifest changed during recovery")
    return pending_manifest


def _require_output_matches_manifest(
    candidate_path: Path, output_path: Path, manifest: dict[str, Any],
) -> None:
    expected = manifest.get("output")
    if not isinstance(expected, dict):
        raise SystemExit("pending evidence manifest lacks an output artifact")
    actual = _durably_prepare_existing_output_artifact(candidate_path, output_path)
    if _publication_artifact_identity(actual) != _publication_artifact_identity(
        expected
    ):
        raise SystemExit("pending trajectory bank does not match its manifest")


def _publish_evidence_pair(
    pending_output: Path,
    output_path: Path,
    pending_meta: Path,
    meta_path: Path,
    manifest: dict[str, Any],
) -> None:
    """Prepare both artifacts before publishing either; retain recovery state."""
    _write_json_staged(pending_meta, manifest)
    published = _publish_output(pending_output, output_path)
    if _publication_artifact_identity(published) != _publication_artifact_identity(
        manifest.get("output")
    ):
        raise RuntimeError("published trajectory bank differs from its manifest")
    _publish_no_replace(pending_meta, meta_path)


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
    outputs = {
        *destinations,
        _output_lock_path(output_path).expanduser().resolve(),
        _output_lock_path(meta_path).expanduser().resolve(),
        _pending_output_path(output_path).expanduser().resolve(),
        _pending_manifest_path(meta_path).expanduser().resolve(),
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
