#!/usr/bin/env python3
"""Fail when in-place C extensions are missing or older than their sources."""

from __future__ import annotations

import argparse
import importlib.machinery
import re
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtensionSpec:
    module: str
    dependencies: tuple[str, ...]


# setup.py is a compile input, not merely packaging metadata: it owns extension
# macro sets, native/LTO flags and which translation units receive them. A
# setup-only change can therefore change the binary while leaving every C/header
# mtime untouched. Keep it in every extension's freshness dependency set.
_SETUP_DEP = "setup.py"

EXTENSION_SPECS: tuple[ExtensionSpec, ...] = (
    ExtensionSpec(
        "chess_anti_engine.encoding._features_ext",
        (
            _SETUP_DEP,
            "chess_anti_engine/encoding/_features_ext.c",
            "chess_anti_engine/encoding/_features_impl.h",
            "chess_anti_engine/encoding/_bitboard_planes_impl.h",
            # This extension is built WITHOUT DEEPFIN_FAST_SLIDERS, so the
            # preprocessor never actually opens the slider header. It is listed
            # anyway because this list mirrors the textual include graph (that
            # is what the coverage test walks, and what keeps a missing entry
            # from hiding): the cost of the extra edge is one unnecessary
            # rebuild, and the cost of omitting it would be a silent stale .so
            # the day someone does give this extension the macros.
            "chess_anti_engine/encoding/_slider_attacks_impl.h",
        ),
    ),
    ExtensionSpec(
        "chess_anti_engine.encoding._lc0_ext",
        (
            _SETUP_DEP,
            "chess_anti_engine/encoding/_lc0_ext.c",
            "chess_anti_engine/encoding/_cboard_impl.h",
            "chess_anti_engine/encoding/_features_impl.h",
            "chess_anti_engine/encoding/_bitboard_planes_impl.h",
            "chess_anti_engine/encoding/_slider_attacks_impl.h",
        ),
    ),
    ExtensionSpec(
        "chess_anti_engine.encoding._perft_ext",
        (
            _SETUP_DEP,
            "chess_anti_engine/encoding/_perft_ext.c",
            "chess_anti_engine/encoding/_cboard_impl.h",
            "chess_anti_engine/encoding/_bitboard_planes_impl.h",
            "chess_anti_engine/encoding/_slider_attacks_impl.h",
        ),
    ),
    ExtensionSpec(
        "chess_anti_engine.mcts._mcts_tree",
        (
            _SETUP_DEP,
            "chess_anti_engine/mcts/_mcts_tree.c",
            "chess_anti_engine/mcts/_value_provider.h",
            "chess_anti_engine/mcts/_search_terminal.h",
            "chess_anti_engine/encoding/_cboard_impl.h",
            "chess_anti_engine/encoding/_features_impl.h",
            "chess_anti_engine/encoding/_bitboard_planes_impl.h",
            "chess_anti_engine/encoding/_slider_attacks_impl.h",
        ),
    ),
    ExtensionSpec(
        "chess_anti_engine.nnue._nnue_ext",
        (
            _SETUP_DEP,
            "chess_anti_engine/nnue/_nnue_ext.c",
            "chess_anti_engine/nnue/_nnue_impl.h",
            "chess_anti_engine/nnue/_nnue_provider.h",
            "chess_anti_engine/nnue/_nnue_state.h",
            "chess_anti_engine/nnue/_fastq_see.h",
            "chess_anti_engine/nnue/_fastq_certificate.h",
            "chess_anti_engine/nnue/_fastq_search.h",
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
            "chess_anti_engine/encoding/_slider_attacks_impl.h",
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


class _ElfFormatError(ValueError):
    """The file is not a well-formed ELF image this reader can trust."""


def _elf_section(data: bytes, name: str) -> bytes | None:
    """Return the contents of ELF section ``name``, or None if it is absent.

    A deliberately small reader (section headers + the section-name string
    table only) so the production gate does not need binutils or pyelftools.
    Every offset is bounds-checked against the file before it is used, and any
    malformed header raises ``_ElfFormatError`` -- a corrupt or truncated file
    is a clean rejection, never a traceback out of the startup gate.
    """
    if len(data) < 0x34 or data[:4] != b"\x7fELF":
        raise _ElfFormatError("not an ELF file")
    elf_class, elf_data = data[4], data[5]
    if elf_class not in (1, 2) or elf_data not in (1, 2):
        raise _ElfFormatError(f"unknown ELF class/data bytes {elf_class}/{elf_data}")
    end = "<" if elf_data == 1 else ">"
    try:
        if elf_class == 2:
            if len(data) < 0x40:
                raise _ElfFormatError("truncated ELF header")
            shoff = struct.unpack_from(end + "Q", data, 0x28)[0]
            shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x3A)
            header = end + "IIQQQQIIQQ"
        else:
            shoff = struct.unpack_from(end + "I", data, 0x20)[0]
            shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x2E)
            header = end + "IIIIIIIIII"
        header_size = struct.calcsize(header)
        if shoff == 0 or shnum == 0:
            raise _ElfFormatError("no section header table")
        if shentsize < header_size:
            raise _ElfFormatError(f"section header entry size {shentsize} < {header_size}")
        if shstrndx >= shnum:
            raise _ElfFormatError(f"section-name table index {shstrndx} >= {shnum}")
        if shoff + (shnum - 1) * shentsize + header_size > len(data):
            raise _ElfFormatError("section header table runs past end of file")

        def section_bytes(sec: tuple[int, ...]) -> bytes:
            offset, size = sec[4], sec[5]
            if sec[1] == 8:  # SHT_NOBITS occupies no file bytes
                return b""
            if offset + size > len(data):
                raise _ElfFormatError("section runs past end of file")
            return data[offset:offset + size]

        sections = [
            struct.unpack_from(header, data, shoff + i * shentsize) for i in range(shnum)
        ]
        # Fields: name, type, flags, addr, offset, size, ...
        names = section_bytes(sections[shstrndx])
        wanted = name.encode()
        for sec in sections:
            start = sec[0]
            if start >= len(names):
                continue
            stop = names.find(b"\x00", start)
            if stop < 0:
                stop = len(names)
            if names[start:stop] == wanted:
                return section_bytes(sec)
    except (struct.error, OverflowError) as exc:
        raise _ElfFormatError(f"malformed ELF headers: {exc}") from exc
    return None


# A -march value that names the ISA baseline rather than the build host. The
# portable build pins -march=x86-64 explicitly (setup.py), and distro GCCs may
# default to a -v2/-v3 level; neither is the host tuning production relies on.
_GENERIC_MARCH = frozenset({"x86-64", "x86-64-v2", "x86-64-v3", "x86-64-v4"})


def production_recipe_problem(path: Path) -> str | None:
    """Why ``path`` fails the production recipe gate, or None if it passes.

    What passes is a NON-BASELINE -march RECORDED WITH LTO -- not, strictly,
    "native": an explicit ``-march=haswell`` with recorded switches passes by
    design. The gate exists to catch the portable build (setup.py pins
    ``-march=x86-64``) or a distro ``x86-64-vN`` default reaching production.

    ⚑ The evidence is the ``.GCC.command.line`` SECTION, never a byte search of
    the file. With ``-g`` every object carries DWARF producer strings, and a
    plain portable LTO build's LTO-translation producer reads
    ``GNU GIMPLE ... -march=x86-64 ... -fltrans`` -- so "-march=" and "-fltrans"
    both occur in a build that is neither native nor recorded. (The deployed
    native build's GIMPLE producer ALSO says -march=x86-64: the ltrans stage
    records its own default and carries the real ISA as per-function target
    options, so producer strings cannot discriminate in either direction.)

    The section exists only under -frecord-gcc-switches, which setup.py adds
    only when CAE_EXT_NATIVE and CAE_EXT_LTO are both set. The GCC driver
    expands -march=native before cc1 records it (e.g. ``-march=znver3`` plus
    explicit ``-m`` ISA flags and ``--param=l1-cache-size=...``), and when the
    same command line repeats -march the driver keeps only the last, so the
    LAST recorded -march is the effective one.

    Unknown-CPU fallback (why a newer-than-GCC host is not falsely rejected):
    GCC 15's ``host_detect_local_cpu`` (gcc/config/i386/driver-i386.cc) never
    substitutes an ``x86-64-vN`` level. An unrecognised Intel model is mapped
    to the nearest named core by feature probing (skylake-avx512, alderlake,
    haswell, ...); an unrecognised vendor/family falls to ``core2`` /
    ``nocona`` with every detected ISA flag appended explicitly. Plain
    ``x86-64`` is chosen only for a 64-bit CPU without SSSE3 *and* SSE3 (an
    emulator), where native and baseline are the same code anyway.

    The LTO-translation marker (-fltrans in a GNU GIMPLE producer) is kept as
    proof the link actually ran LTO code generation.
    """
    data = path.read_bytes()
    try:
        section = _elf_section(data, ".GCC.command.line")
    except _ElfFormatError as exc:
        return f"unreadable ELF ({exc})"
    if section is None:
        return "no .GCC.command.line section (not built with -frecord-gcc-switches)"
    lines = [raw.decode("utf-8", "replace") for raw in section.split(b"\x00") if raw.strip()]
    if not lines:
        return "empty .GCC.command.line section"
    for line in lines:
        tokens = line.split()
        marches = [tok.split("=", 1)[1] for tok in tokens if tok.startswith("-march=")]
        if not marches:
            return f"recorded command line has no -march=: {line[:120]!r}"
        if marches[-1] in _GENERIC_MARCH:
            return f"recorded -march={marches[-1]} is a generic ISA baseline, not a host-tuned build"
        if not any(tok == "-flto" or tok.startswith("-flto=") for tok in tokens):
            return "recorded command line has no -flto"
    if not re.search(rb"GNU GIMPLE [^\x00]*-fltrans", data):
        return "no LTO translation (-fltrans) producer record"
    return None


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
        if require_production_recipe:
            problem = production_recipe_problem(output)
            if problem is not None:
                issues.append(
                    f"{spec.module} does not record the native+LTO production recipe: {problem}",
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
        help=(
            "Require a .GCC.command.line record with -flto and a non-baseline "
            "-march (recorded with LTO), plus an LTO translation producer record."
        ),
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
