"""Import tracked project Python from authenticated source bytes, never bytecode.

This module is itself compiled directly from bytes authenticated by the entrypoint's
pre-import snapshot.  Keep it stdlib-only: it is the import barrier before any other
project or third-party module executes.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any


SOURCE_ONLY_IMPORT_SCHEMA = "deepfin.source_only_import.v1"
BOOTSTRAP_SOURCE_SHA256 = "filled_by_authenticated_entrypoint"
_PROJECT_ROOTS = ("chess_anti_engine", "scripts")
_ACTIVE_GUARD: SourceOnlyProjectImportGuard | None = None


def _is_project_module(fullname: str) -> bool:
    return any(
        fullname == root or fullname.startswith(root + ".")
        for root in _PROJECT_ROOTS
    )


def _file_bytes(
    path: Path, expected: Mapping[str, Any], *, object_format: str,
) -> bytes:
    """Return exact snapshotted bytes or fail before compiling anything."""
    resolved = path.resolve()
    before = resolved.lstat()
    content = resolved.read_bytes()
    after = resolved.lstat()
    before_identity = (
        int(before.st_mode), int(before.st_size), int(before.st_mtime_ns),
        int(before.st_ctime_ns), int(before.st_dev), int(before.st_ino),
    )
    after_identity = (
        int(after.st_mode), int(after.st_size), int(after.st_mtime_ns),
        int(after.st_ctime_ns), int(after.st_dev), int(after.st_ino),
    )
    blob = f"blob {len(content)}\0".encode("ascii") + content
    observed_oid = hashlib.new(object_format, blob).hexdigest()
    expected_identity = (
        expected.get("git_mode") in ("100644", "100755"),
        expected.get("path") == str(resolved),
        expected.get("size") == len(content),
        expected.get("mtime_ns") == int(after.st_mtime_ns),
        expected.get("ctime_ns") == int(after.st_ctime_ns),
        expected.get("device") == int(after.st_dev),
        expected.get("inode") == int(after.st_ino),
        expected.get("sha256") == hashlib.sha256(content).hexdigest(),
        expected.get("git_blob_oid") == observed_oid,
        expected.get("observed_git_blob_oid") == observed_oid,
        expected.get("stable_read") is True,
        expected.get("matches_git_revision") is True,
    )
    if (
        before_identity != after_identity
        or not stat.S_ISREG(after.st_mode)
        or not all(expected_identity)
    ):
        raise ImportError(
            f"tracked project source changed after the pre-import snapshot: {resolved}"
        )
    return content


class _AuthenticatedSourceLoader(importlib.abc.Loader):
    def __init__(
        self,
        guard: SourceOnlyProjectImportGuard,
        fullname: str,
        source_path: Path,
        relative_path: str,
        expected: Mapping[str, Any],
    ) -> None:
        self._guard = guard
        self._fullname = fullname
        self._source_path = source_path
        self._relative_path = relative_path
        self._expected = expected

    def create_module(self, spec: Any) -> ModuleType | None:
        del spec
        return None

    def exec_module(self, module: ModuleType) -> None:
        try:
            source = _file_bytes(
                self._source_path,
                self._expected,
                object_format=self._guard.object_format,
            )
            code = compile(
                source, str(self._source_path), "exec", dont_inherit=True,
            )
            exec(code, module.__dict__)
        except BaseException as exc:
            self._guard.failures.append({
                "module": self._fullname,
                "repo_relative_path": self._relative_path,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        self._guard.verified_modules[self._fullname] = {
            "repo_relative_path": self._relative_path,
            "sha256": hashlib.sha256(source).hexdigest(),
            "execution": "compiled_authenticated_source_bytes",
            "bytecode_cache_read": False,
        }


class SourceOnlyProjectImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        self.snapshot = snapshot
        self.repo_root = Path(str(snapshot["repo_root"])).resolve()
        self.git_sha = str(snapshot["git_sha"])
        self.object_format = str(snapshot["git_object_format"])
        self.surface_sha256 = str(snapshot["tracked_python_surface_sha256"])
        raw_files = snapshot["files"]
        if not isinstance(raw_files, dict):
            raise ImportError("pre-import Python file inventory is malformed")
        self.files: dict[str, Mapping[str, Any]] = dict(raw_files)
        self.verified_modules: dict[str, dict[str, Any]] = {}
        self.failures: list[dict[str, str]] = []

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> Any:
        if not _is_project_module(fullname):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None:
            return spec
        if not isinstance(spec.origin, str):
            raise ImportError(
                f"project module has no authenticated source origin: {fullname}"
            )
        origin = Path(spec.origin).resolve()
        if isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
            # Native extensions keep the interpreter's normal extension loader.
            return spec
        if origin.suffix != ".py":
            raise ImportError(
                f"project module is not source-backed or native: {fullname}"
            )
        try:
            relative_path = origin.relative_to(self.repo_root).as_posix()
        except ValueError as exc:
            raise ImportError(
                f"project module resolves outside authenticated checkout: {fullname}"
            ) from exc
        expected = self.files.get(relative_path)
        if expected is None:
            raise ImportError(
                f"project module is absent from tracked pre-import snapshot: {fullname}"
            )
        spec.loader = _AuthenticatedSourceLoader(
            self, fullname, origin, relative_path, expected,
        )
        # This loader never consumes or emits a bytecode cache.
        spec.cached = None
        return spec

    def module_verified(self, fullname: str, relative_path: str) -> bool:
        row = self.verified_modules.get(fullname)
        expected = self.files.get(relative_path)
        return bool(
            isinstance(row, dict)
            and isinstance(expected, Mapping)
            and row.get("repo_relative_path") == relative_path
            and row.get("sha256") == expected.get("sha256")
            and row.get("execution") in (
                "compiled_authenticated_source_bytes",
                "compiled_authenticated_bootstrap_source_bytes",
            )
            and row.get("bytecode_cache_read") is False
        )

    def status(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_ONLY_IMPORT_SCHEMA,
            "active": self in sys.meta_path,
            "git_sha": self.git_sha,
            "tracked_python_surface_sha256": self.surface_sha256,
            "project_scope": list(_PROJECT_ROOTS),
            "execution": "compile_authenticated_source_bytes",
            "bytecode_cache_reads": False,
            "native_extension_loading": "unchanged_pathfinder_loader",
            "verified_modules": dict(sorted(self.verified_modules.items())),
            "failures": list(self.failures),
        }


def install(snapshot: Mapping[str, Any]) -> SourceOnlyProjectImportGuard:
    """Install or reuse the process-wide authenticated project-source finder."""
    global _ACTIVE_GUARD
    if (
        snapshot.get("source_tree_matches_revision") is not True
        or snapshot.get("git_sha") != snapshot.get("final_git_sha")
    ):
        raise ImportError("pre-import Python source surface is not revision-authenticated")
    if _ACTIVE_GUARD is not None:
        if (
            _ACTIVE_GUARD.repo_root != Path(str(snapshot.get("repo_root"))).resolve()
            or _ACTIVE_GUARD.git_sha != snapshot.get("git_sha")
            or _ACTIVE_GUARD.surface_sha256
            != snapshot.get("tracked_python_surface_sha256")
        ):
            raise ImportError("process already has a different project source guard")
        return _ACTIVE_GUARD
    guard = SourceOnlyProjectImportGuard(snapshot)
    bootstrap = guard.files.get("scripts/source_only_import.py")
    if not isinstance(bootstrap, Mapping):
        raise ImportError("source-only import guard is absent from its source snapshot")
    guard.verified_modules["scripts.source_only_import"] = {
        "repo_relative_path": "scripts/source_only_import.py",
        "sha256": bootstrap.get("sha256"),
        "execution": "compiled_authenticated_bootstrap_source_bytes",
        "bytecode_cache_read": False,
    }
    sys.meta_path.insert(0, guard)
    _ACTIVE_GUARD = guard
    return guard
