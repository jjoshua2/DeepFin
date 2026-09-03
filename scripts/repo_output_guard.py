"""Shared guards against replacing repository-controlled files with outputs."""

from __future__ import annotations

import subprocess
from pathlib import Path


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
            tracked = subprocess.run(
                [
                    "git", "-C", str(root), "ls-files", "--error-unmatch",
                    "--", str(relative),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
        except OSError:
            # If Git cannot prove that an in-repository destination is safe,
            # fail closed instead of allowing a destructive overwrite.
            tracked = True
        if tracked:
            return True
    return False
