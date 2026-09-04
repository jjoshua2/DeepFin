"""Shared guards against replacing repository-controlled files with outputs."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _reserved_output_name(name: str) -> bool:
    lock_name = (
        name.startswith(".")
        and name.endswith(".lock")
        and len(name) > len("..lock")
    )
    # Use the final marker: a legitimate hidden output can itself begin
    # ``.tmp-`` and its staging name then contains an earlier marker at index 1.
    staging_marker = name.rfind(".tmp-")
    staging_name = name.startswith(".") and staging_marker > 1
    invalid_recovery_name = (
        name.startswith(".")
        and name.endswith(".invalid-recovery")
        and len(name) > len("..invalid-recovery")
    )
    return lock_name or staging_name or invalid_recovery_name


def reserved_output_path(path: Path) -> bool:
    """Whether a lexical or resolved basename is an internal output name."""
    expanded = path.expanduser()
    names = {expanded.absolute().name, expanded.resolve().name}
    return any(_reserved_output_name(name) for name in names)


def _git_path(repo_root: Path, flag: str) -> Path | None:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--path-format=absolute", flag],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(value).expanduser().resolve() if value else None


def git_control_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return worktree-specific and shared Git control paths."""
    paths = {repo_root.resolve() / ".git"}
    for flag in ("--absolute-git-dir", "--git-common-dir"):
        path = _git_path(repo_root, flag)
        if path is not None:
            paths.add(path)
    return tuple(sorted(paths, key=str))


def _inside_or_equal(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def repo_controlled_output(path: Path, repo_root: Path) -> bool:
    """Whether an output would replace tracked content or Git metadata."""
    root = repo_root.expanduser().resolve()
    candidates = {path.expanduser().absolute(), path.expanduser().resolve()}
    controls = git_control_paths(root)
    if any(
        _inside_or_equal(candidate, control)
        for candidate in candidates
        for control in controls
    ):
        return True

    for candidate in candidates:
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        try:
            result = subprocess.run(
                [
                    "git", "-C", str(root), "ls-files", "--error-unmatch",
                    "--", str(relative),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            # If Git cannot prove that an in-repository destination is safe,
            # fail closed instead of allowing a destructive overwrite.
            return True
        if result.returncode == 0:
            return True
        if result.returncode != 1:
            # Exit 1 is the documented no-match result. Fatal Git failures
            # must not silently turn the destructive-write guard off.
            return True
    return False
