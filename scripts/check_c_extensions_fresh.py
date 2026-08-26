#!/usr/bin/env python3
"""Fail when in-place C extensions are missing or older than their sources."""

from __future__ import annotations

import argparse
import importlib.machinery
import re
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
            "chess_anti_engine/encoding/_bitboard_planes_impl.h",
        ),
    ),
    ExtensionSpec(
        "chess_anti_engine.encoding._lc0_ext",
        (
            "chess_anti_engine/encoding/_lc0_ext.c",
            "chess_anti_engine/encoding/_cboard_impl.h",
            "chess_anti_engine/encoding/_features_impl.h",
            "chess_anti_engine/encoding/_bitboard_planes_impl.h",
        ),
    ),
    ExtensionSpec(
        "chess_anti_engine.mcts._mcts_tree",
        (
            "chess_anti_engine/mcts/_mcts_tree.c",
            "chess_anti_engine/mcts/_value_provider.h",
            "chess_anti_engine/mcts/_search_terminal.h",
            "chess_anti_engine/encoding/_cboard_impl.h",
            "chess_anti_engine/encoding/_features_impl.h",
            "chess_anti_engine/encoding/_bitboard_planes_impl.h",
        ),
    ),
    ExtensionSpec(
        "chess_anti_engine.nnue._nnue_ext",
        (
            "chess_anti_engine/nnue/_nnue_ext.c",
            "chess_anti_engine/nnue/_nnue_impl.h",
            "chess_anti_engine/nnue/_nnue_provider.h",
            "chess_anti_engine/nnue/_nnue_state.h",
            "chess_anti_engine/nnue/_arm_providers.h",
            "chess_anti_engine/nnue/_nnue_dag_store.h",
            "chess_anti_engine/nnue/_nnue_dag_api.h",
            "chess_anti_engine/mcts/_position_dag.h",
            "chess_anti_engine/mcts/_value_provider.h",
            "chess_anti_engine/mcts/_check_resolver.h",
            "chess_anti_engine/mcts/_search_terminal.h",
            "chess_anti_engine/encoding/_cboard_impl.h",
            # Reached through _cboard_impl.h, not included directly — the
            # freshness question is what the OBJECT was compiled from, so the
            # graph is transitive and the list has to be too.
            "chess_anti_engine/encoding/_bitboard_planes_impl.h",
        ),
    ),
)


def _module_base(root: Path, module: str) -> Path:
    return root.joinpath(*module.split("."))


def _extension_outputs(root: Path, module: str) -> list[Path]:
    base = _module_base(root, module)
    return [base.with_name(base.name + suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES]


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _gcc_major(path: Path) -> int | None:
    match = re.search(rb"GCC: [^\x00\r\n]*?(\d+)\.\d+", path.read_bytes())
    return int(match.group(1)) if match is not None else None


def _has_production_recipe(path: Path) -> bool:
    data = path.read_bytes()
    # GCC's -frecord-gcc-switches survives the LTO link as a
    # .GCC.command.line string. With native compilation GCC records the
    # resolved host architecture (for example -march=znver3); the LTO
    # translation stage records -fltrans.
    return b"-march=" in data and b"-fltrans" in data


def check_extensions(
    root: Path,
    *,
    min_gcc_major: int | None = None,
    require_production_recipe: bool = False,
) -> list[str]:
    """Return actionable freshness problems for in-place C extensions."""
    issues: list[str] = []
    for spec in EXTENSION_SPECS:
        output = _first_existing(_extension_outputs(root, spec.module))
        if output is None:
            issues.append(f"{spec.module} is missing")
            continue
        if min_gcc_major is not None:
            compiler_major = _gcc_major(output)
            if compiler_major is None:
                issues.append(f"{spec.module} has no detectable GCC compiler identity")
            elif compiler_major < int(min_gcc_major):
                issues.append(
                    f"{spec.module} was built with GCC {compiler_major}; "
                    f"production requires GCC >= {int(min_gcc_major)}",
                )
        if require_production_recipe and not _has_production_recipe(output):
            issues.append(
                f"{spec.module} does not record the native+LTO production recipe",
            )
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
    parser.add_argument(
        "--min-gcc-major", type=int, default=None,
        help="Require every extension's ELF compiler identity to meet this GCC major.",
    )
    parser.add_argument(
        "--require-production-recipe", action="store_true",
        help="Require recorded native architecture and LTO translation flags.",
    )
    args = parser.parse_args()

    issues = check_extensions(
        args.root.resolve(),
        min_gcc_major=args.min_gcc_major,
        require_production_recipe=args.require_production_recipe,
    )
    if issues:
        print("C extension freshness check failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("Run: python3 scripts/build_production_extensions.py")
        return 1
    if not args.quiet:
        print("C extensions are present and up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
