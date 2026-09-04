"""Import tracked project Python from authenticated source bytes, never bytecode.

This module is itself compiled directly from bytes authenticated by the entrypoint's
pre-import snapshot.  Keep it stdlib-only: it is the import barrier before any other
project or third-party module executes.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import errno
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any


SOURCE_ONLY_IMPORT_SCHEMA = "deepfin.source_only_import.v2"
BOOTSTRAP_SOURCE_SHA256 = "filled_by_authenticated_entrypoint"
_PROJECT_ROOTS = ("chess_anti_engine", "scripts")
PERMITTED_NATIVE_MODULES = (
    "chess_anti_engine.encoding._features_ext",
    "chess_anti_engine.encoding._lc0_ext",
    "chess_anti_engine.mcts._mcts_tree",
)
_ACTIVE_GUARD: SourceOnlyProjectImportGuard | None = None


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_mode), int(value.st_size), int(value.st_mtime_ns),
        int(value.st_ctime_ns), int(value.st_dev), int(value.st_ino),
    )


def _stable_regular_bytes(path: Path) -> tuple[bytes, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise OSError(errno.ENOTSUP, "safe reads require O_NOFOLLOW", str(path))
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | nofollow,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        chunks: list[bytes] = []
        offset = 0
        while offset < int(before.st_size):
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, int(before.st_size) - offset),
                offset,
            )
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        if _stable_identity(before) != _stable_identity(after) or len(
            content
        ) != int(after.st_size):
            raise OSError(errno.EIO, "file changed during stable read", str(path))
        return content, after
    finally:
        os.close(descriptor)


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
    content, after = _stable_regular_bytes(resolved)
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
        not stat.S_ISREG(after.st_mode) or not all(expected_identity)
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


def _identity(artifact: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        artifact.get(name)
        for name in (
            "path", "lexical_path", "size", "mtime_ns", "ctime_ns", "device",
            "inode", "sha256", "stable_read",
        )
    )


def _native_file_artifact(path: Path) -> dict[str, Any]:
    """Read one extension as stable bytes before the dynamic loader maps it."""
    resolved = path.resolve()
    try:
        content, after = _stable_regular_bytes(resolved)
    except OSError as exc:
        raise ImportError(
            f"native extension changed while authenticated: {resolved}"
        ) from exc
    return {
        "path": str(resolved),
        "lexical_path": str(resolved),
        "size": len(content),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "sha256": hashlib.sha256(content).hexdigest(),
        "stable_read": True,
    }


class _AuthenticatedExtensionLoader(importlib.abc.Loader):
    """Keep an authorized extension bound to its snapshotted path and bytes."""

    def __init__(
        self,
        guard: SourceOnlyProjectImportGuard,
        fullname: str,
        loader: importlib.machinery.ExtensionFileLoader,
        expected: Mapping[str, Any],
    ) -> None:
        self._guard = guard
        self._fullname = fullname
        self._loader = loader
        self._expected = expected

    def _verify(self) -> dict[str, Any]:
        path = Path(str(self._expected["path"]))
        observed = _native_file_artifact(path)
        if _identity(observed) != _identity(self._expected):
            raise ImportError(
                "authorized native extension changed after its pre-import snapshot: "
                f"{self._fullname}"
            )
        return observed

    def create_module(self, spec: Any) -> ModuleType | None:
        # ExtensionFileLoader maps and initializes the shared object in
        # create_module(), so the last check before that call is essential.
        self._verify()
        module = self._loader.create_module(spec)
        self._verify()
        return module

    def exec_module(self, module: ModuleType) -> None:
        try:
            self._verify()
            self._loader.exec_module(module)
            observed = self._verify()
        except BaseException as exc:
            self._guard.failures.append({
                "module": self._fullname,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        self._guard.verified_native_modules[self._fullname] = {
            **observed,
            "execution": "authenticated_canonical_extension_loader",
            "preimport_artifact_authenticated": True,
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
        self.authorized_native_modules: dict[str, Mapping[str, Any]] = {}
        self.verified_native_modules: dict[str, dict[str, Any]] = {}
        self.failures: list[dict[str, str]] = []

    def authorize_native(self, artifacts: Mapping[str, Any]) -> None:
        """Authorize only the fixed native surface after exact byte snapshots."""
        if set(artifacts) != set(PERMITTED_NATIVE_MODULES):
            raise ImportError(
                "native authorization must cover exactly the permitted module surface"
            )
        authorized: dict[str, Mapping[str, Any]] = {}
        for fullname in PERMITTED_NATIVE_MODULES:
            raw = artifacts.get(fullname)
            if not isinstance(raw, Mapping):
                raise ImportError(f"native pre-import artifact is malformed: {fullname}")
            raw_path = raw.get("path")
            raw_lexical_path = raw.get("lexical_path")
            if not isinstance(raw_path, str) or not isinstance(raw_lexical_path, str):
                raise ImportError(f"native pre-import path is malformed: {fullname}")
            path = Path(raw_path).resolve()
            base = self.repo_root.joinpath(*fullname.split("."))
            canonical_paths = {
                candidate.absolute()
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
                if (
                    not (candidate := base.with_name(base.name + suffix)).is_symlink()
                    and candidate.is_file()
                    and stat.S_ISREG(candidate.lstat().st_mode)
                )
            }
            lexical_path = Path(raw_lexical_path).absolute()
            if (
                lexical_path not in canonical_paths
                or path != lexical_path
                or self.repo_root not in path.parents
            ):
                raise ImportError(
                    f"native extension is not at its canonical project path: {fullname}"
                )
            observed = _native_file_artifact(path)
            if _identity(observed) != _identity(raw):
                raise ImportError(
                    f"native extension differs from its stable pre-import snapshot: {fullname}"
                )
            authorized[fullname] = dict(raw)
        self.authorized_native_modules = authorized

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
            expected = self.authorized_native_modules.get(fullname)
            if expected is None:
                raise ImportError(
                    f"project native extension is not explicitly authorized: {fullname}"
                )
            if origin != Path(str(expected.get("path"))).resolve():
                raise ImportError(
                    f"project native extension resolved outside its authorized path: {fullname}"
                )
            spec.loader = _AuthenticatedExtensionLoader(
                self, fullname, spec.loader, expected,
            )
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

    def loaded_project_module_status(self) -> dict[str, Any]:
        """Account for every loaded project module, including fileless objects."""
        unverified: list[dict[str, str]] = []
        loaded: list[str] = []
        for fullname, module in sorted(sys.modules.items()):
            if not _is_project_module(fullname):
                continue
            loaded.append(fullname)
            module_file = getattr(module, "__file__", None)
            source = self.verified_modules.get(fullname)
            native = self.verified_native_modules.get(fullname)
            if isinstance(source, Mapping):
                expected = self.repo_root / str(source.get("repo_relative_path", ""))
                if (
                    not isinstance(module_file, str)
                    or Path(module_file).resolve() != expected.resolve()
                ):
                    unverified.append({
                        "module": fullname,
                        "reason": "loaded source path differs from authenticated source",
                    })
                continue
            if isinstance(native, Mapping):
                if (
                    not isinstance(module_file, str)
                    or Path(module_file).resolve()
                    != Path(str(native.get("path", ""))).resolve()
                ):
                    unverified.append({
                        "module": fullname,
                        "reason": "loaded native path differs from authenticated extension",
                    })
                continue
            unverified.append({
                "module": fullname,
                "reason": (
                    "loaded project module has no file"
                    if not isinstance(module_file, str)
                    else "loaded project module lacks authenticated execution record"
                ),
            })
        return {
            "passed": not unverified,
            "loaded_modules": loaded,
            "unverified_modules": unverified,
        }

    def status(self) -> dict[str, Any]:
        loaded_status = self.loaded_project_module_status()
        return {
            "schema": SOURCE_ONLY_IMPORT_SCHEMA,
            "active": bool(sys.meta_path and sys.meta_path[0] is self),
            "installed": self in sys.meta_path,
            "first_finder": bool(sys.meta_path and sys.meta_path[0] is self),
            "git_sha": self.git_sha,
            "tracked_python_surface_sha256": self.surface_sha256,
            "project_scope": list(_PROJECT_ROOTS),
            "execution": "compile_authenticated_source_bytes",
            "bytecode_cache_reads": False,
            "native_extension_loading": (
                "default_deny_exact_preimport_artifact_authenticated_loader"
            ),
            "permitted_native_modules": list(PERMITTED_NATIVE_MODULES),
            "authorized_native_modules": sorted(self.authorized_native_modules),
            "authorized_native_artifacts": {
                name: dict(artifact)
                for name, artifact in sorted(self.authorized_native_modules.items())
            },
            "verified_native_modules": dict(sorted(self.verified_native_modules.items())),
            "verified_modules": dict(sorted(self.verified_modules.items())),
            "loaded_project_modules": loaded_status,
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
