#!/usr/bin/env python3
"""Fail when in-place C extensions are missing or older than their sources."""

from __future__ import annotations

import argparse
import importlib.machinery
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtensionSpec:
    module: str
    dependencies: tuple[str, ...]


EXTENSION_SPECS: tuple[ExtensionSpec, ...] = (
    ExtensionSpec(
        "chess_anti_engine.encoding._features_ext",
        (
            "chess_anti_engine/encoding/_features_ext.c",
            "chess_anti_engine/encoding/_features_impl.h",
        ),
    ),
    ExtensionSpec(
        "chess_anti_engine.encoding._lc0_ext",
        (
            "chess_anti_engine/encoding/_lc0_ext.c",
            "chess_anti_engine/encoding/_cboard_impl.h",
            "chess_anti_engine/encoding/_features_impl.h",
        ),
    ),
    ExtensionSpec(
        "chess_anti_engine.mcts._mcts_tree",
        (
            "chess_anti_engine/mcts/_mcts_tree.c",
            "chess_anti_engine/encoding/_cboard_impl.h",
            "chess_anti_engine/encoding/_features_impl.h",
        ),
    ),
)


def _module_base(root: Path, module: str) -> Path:
    return root.joinpath(*module.split("."))


def _extension_outputs(root: Path, module: str) -> list[Path]:
    base = _module_base(root, module)
    return [base.with_name(base.name + suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES]


def _newest_existing(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime_ns)


def check_extensions(root: Path) -> list[str]:
    """Return actionable freshness problems for in-place C extensions."""
    issues: list[str] = []
    for spec in EXTENSION_SPECS:
        output = _newest_existing(_extension_outputs(root, spec.module))
        if output is None:
            issues.append(f"{spec.module} is missing")
            continue
        output_mtime = output.stat().st_mtime_ns
        for dep_rel in spec.dependencies:
            dep = root / dep_rel
            if not dep.exists():
                issues.append(f"{spec.module} dependency is missing: {dep_rel}")
                continue
            if dep.stat().st_mtime_ns > output_mtime:
                issues.append(f"{spec.module} is older than {dep_rel}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to inspect.")
    parser.add_argument("--quiet", action="store_true", help="Suppress success output.")
    args = parser.parse_args()

    issues = check_extensions(args.root.resolve())
    if issues:
        print("C extension freshness check failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("Run: python3 setup.py build_ext --inplace")
        return 1
    if not args.quiet:
        print("C extensions are present and up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
