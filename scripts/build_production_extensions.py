#!/usr/bin/env python3
"""Build host-optimized in-place C extensions for local production runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from collections.abc import Mapping

# This script is an operational entry point documented as
# ``python3 scripts/build_production_extensions.py``.  Make that invocation
# independent of an inherited PYTHONPATH before importing shared build-input
# definitions from the repository package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_c_extensions_fresh import (
    NATIVE_BUILD_ATTESTED_MODULES,
    extension_spec,
    native_build_attestation,
    require_current_native_build_attestation_schema,
)


_TRANSIENT_BUILD_ENV = (
    "AR",
    "CFLAGS",
    "CPPFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "LDSHARED",
    "RANLIB",
    "CAE_EXT_SANITIZE",
    "CAE_EXT_WERROR",
    "CAE_EXT_BUILD_GIT_SHA",
    "CAE_EXT_BUILD_INPUT_DIGESTS",
)


def select_compiler(*, env: Mapping[str, str], home: Path) -> str:
    """Prefer an explicit compiler, then the validated local GCC 15 install."""
    candidates = (
        env.get("CAE_GCC15_CC", ""),
        str(home / ".local/gcc-15.3/bin/gcc"),
    )
    for candidate in candidates:
        path = Path(candidate).expanduser() if candidate else None
        if path is not None and path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    raise RuntimeError(
        "validated GCC 15 compiler not found; install ~/.local/gcc-15.3/bin/gcc "
        "or set CAE_GCC15_CC (use setup.py directly only for portable builds)",
    )


def build_environment(
    *,
    compiler: str,
    base: Mapping[str, str],
    source_git_sha: str | None = None,
    input_digests: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base)
    for name in _TRANSIENT_BUILD_ENV:
        env.pop(name, None)
    env.update({
        "CC": str(compiler),
        # CPython's sysconfig can hard-code the distro compiler in LDSHARED
        # even when distutils honors CC for compilation. LTO object formats
        # are compiler-version-specific, so force the same validated driver
        # for the shared-library link as well.
        "LDSHARED": f"{compiler} -shared",
        "CAE_EXT_NATIVE": "1",
        "CAE_EXT_LTO": "1",
    })
    if source_git_sha is not None or input_digests is not None:
        if source_git_sha is None or input_digests is None:
            raise ValueError("native build SHA and input digests must be supplied together")
        env["CAE_EXT_BUILD_GIT_SHA"] = source_git_sha
        env["CAE_EXT_BUILD_INPUT_DIGESTS"] = json.dumps(
            dict(input_digests), sort_keys=True, separators=(",", ":"),
        )
    return env


def clean_source_git_sha(root: Path) -> str:
    """Return the exact clean revision whose sources the build will consume."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            [
                "git", "-C", str(root), "status", "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("production native build requires a Git checkout") from exc
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha.lower()):
        raise RuntimeError("production native build could not resolve a full Git commit")
    if status.strip():
        raise RuntimeError(
            "production native build requires a clean tracked/untracked source checkout"
        )
    return sha


def _git_file_at_commit(root: Path, source_git_sha: str, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "show", f"{source_git_sha}:{relative_path}"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"native build input is not tracked at {source_git_sha}: {relative_path}"
        ) from exc


def revision_build_attestations(
    root: Path, source_git_sha: str,
) -> dict[str, dict[str, object]]:
    """Freeze every producer-loaded extension input at one clean revision."""
    require_current_native_build_attestation_schema()
    result: dict[str, dict[str, object]] = {}
    for module in NATIVE_BUILD_ATTESTED_MODULES:
        dependencies: dict[str, bytes] = {}
        for relative_path in extension_spec(module).dependencies:
            current = (root / relative_path).read_bytes()
            committed = _git_file_at_commit(root, source_git_sha, relative_path)
            if current != committed:
                raise RuntimeError(
                    f"native build input differs from {source_git_sha}: {relative_path}"
                )
            dependencies[relative_path] = current
        result[module] = native_build_attestation(
            module, source_git_sha, dependencies,
        )
    return result


def revision_input_identities(root: Path) -> dict[str, dict[str, int]]:
    """Snapshot metadata that exposes an edit restored to the same bytes/mtime."""
    dependencies = sorted({
        relative_path
        for module in NATIVE_BUILD_ATTESTED_MODULES
        for relative_path in extension_spec(module).dependencies
    })
    result: dict[str, dict[str, int]] = {}
    for relative_path in dependencies:
        stat = (root / relative_path).stat()
        result[relative_path] = {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            # Unlike mtime, an unprivileged process cannot restore ctime after
            # rewriting a file.  This closes the alter/restore-mtime race.
            "ctime_ns": stat.st_ctime_ns,
        }
    return result


def verify_embedded_attestations(
    root: Path, attestations: Mapping[str, Mapping[str, object]],
) -> None:
    """Read back the stamps from a fresh interpreter after the forced build."""
    expected = {
        module: {
            "schema": record["schema"],
            "module": record["module"],
            "source_git_sha": record["source_git_sha"],
            "input_sha256": record["input_sha256"],
        }
        for module, record in attestations.items()
    }
    program = "\n".join((
        "import importlib, json, sys",
        f"expected = json.loads({json.dumps(json.dumps(expected, sort_keys=True))})",
        "observed = {}",
        "for name in sorted(expected):",
        "    module = importlib.import_module(name)",
        "    observed[name] = {",
        "        'schema': getattr(module, 'BUILD_ATTESTATION_SCHEMA', None),",
        "        'module': getattr(module, 'BUILD_MODULE_NAME', None),",
        "        'source_git_sha': getattr(module, 'BUILD_SOURCE_GIT_SHA', None),",
        "        'input_sha256': getattr(module, 'BUILD_INPUT_SHA256', None),",
        "    }",
        "if observed != expected:",
        "    raise SystemExit('native build attestation readback mismatch: ' + repr(observed))",
    ))
    subprocess.run(
        [sys.executable, "-c", program], cwd=root, check=True,
        env={**os.environ, "PYTHONPATH": str(root)},
    )


def verify_source_snapshot_unchanged(
    root: Path,
    source_git_sha: str,
    attestations: Mapping[str, Mapping[str, object]],
    input_identities: Mapping[str, Mapping[str, int]],
) -> None:
    """Reject a build if HEAD or any attested input moved during compilation."""
    current_sha = clean_source_git_sha(root)
    if current_sha != source_git_sha:
        raise RuntimeError(
            "production native build source revision changed during compilation"
        )
    current = revision_build_attestations(root, current_sha)
    if current != dict(attestations):
        raise RuntimeError(
            "production native build inputs changed during compilation"
        )
    if revision_input_identities(root) != dict(input_identities):
        raise RuntimeError(
            "production native build input identities changed during compilation"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the build without executing it.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    compiler = select_compiler(env=os.environ, home=Path.home())
    source_git_sha = clean_source_git_sha(root)
    attestations = revision_build_attestations(root, source_git_sha)
    input_identities = revision_input_identities(root)
    input_digests = {
        module: str(record["input_sha256"])
        for module, record in attestations.items()
    }
    # Always force: setuptools does not discover transitive .h dependencies,
    # and a stale portable/sanitized object must never be certified as a
    # production build merely because its top-level .c mtime is unchanged.
    command = [sys.executable, "setup.py", "build_ext", "--inplace", "--force"]
    print(
        f"production extension build: CC={compiler} native=1 lto=1 "
        f"command={shlex.join(command)}",
    )
    if args.dry_run:
        return 0
    subprocess.run(
        command,
        cwd=root,
        env=build_environment(
            compiler=compiler,
            base=os.environ,
            source_git_sha=source_git_sha,
            input_digests=input_digests,
        ),
        check=True,
    )
    # The embedded digest describes the pre-build snapshot.  Recheck the live
    # checkout before accepting that stamp so a concurrent source edit cannot
    # silently produce decision-grade evidence under the old identity.
    verify_source_snapshot_unchanged(
        root, source_git_sha, attestations, input_identities,
    )
    verify_embedded_attestations(root, attestations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
