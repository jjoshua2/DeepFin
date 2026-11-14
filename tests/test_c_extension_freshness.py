from __future__ import annotations

import importlib.machinery
import os
import platform
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from scripts.check_c_extensions_fresh import (
    EXTENSION_SPECS,
    _elf_section,
    check_extensions,
    production_recipe_problem,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# ⚑ Derived from EXTENSION_SPECS, never restated. These lists used to be written
# out by hand in eight places, so adding a fourth extension meant the guard's own
# tests kept passing while saying nothing about it — a test suite that is green
# because it does not know the new thing exists. Deriving them means a module
# registered in the spec is exercised by every test below on the same commit.
ALL_MODULES: tuple[str, ...] = tuple(spec.module for spec in EXTENSION_SPECS)
ALL_SOURCES: tuple[str, ...] = tuple(
    dict.fromkeys(dep for spec in EXTENSION_SPECS for dep in spec.dependencies)
)


def _write(path: Path, *, mtime_ns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _extension_output(root: Path, module: str, suffix: str | None = None) -> Path:
    base = root.joinpath(*module.split("."))
    return base.with_name(base.name + (suffix or importlib.machinery.EXTENSION_SUFFIXES[0]))


def _write_all_sources(root: Path, *, mtime_ns: int = 10) -> None:
    for rel in ALL_SOURCES:
        _write(root / rel, mtime_ns=mtime_ns)


def _write_all_extensions(root: Path, *, mtime_ns: int = 20) -> None:
    for module in ALL_MODULES:
        _write(_extension_output(root, module), mtime_ns=mtime_ns)


def _module_source(module: str) -> Path:
    """The .c a module is built from, e.g. .../mcts/_mcts_tree.c."""
    return REPO_ROOT.joinpath(*module.split(".")).with_suffix(".c")


def _transitive_local_includes(entry: Path) -> set[str]:
    """Every in-repo header reachable from ``entry`` by following #include "...".

    Resolves each include relative to the including file, the way the compiler
    does, and returns repo-relative POSIX paths.
    """
    seen: set[str] = set()
    stack = [entry]
    visited: set[Path] = set()
    while stack:
        current = stack.pop()
        resolved = current.resolve()
        if resolved in visited or not resolved.exists():
            continue
        visited.add(resolved)
        for raw in re.findall(r'^\s*#\s*include\s+"([^"]+)"', current.read_text(), re.M):
            target = (current.parent / raw).resolve()
            if not target.exists() or REPO_ROOT not in target.parents:
                continue
            seen.add(target.relative_to(REPO_ROOT).as_posix())
            stack.append(target)
    return seen


def test_check_extensions_accepts_fresh_outputs(tmp_path: Path):
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)

    assert check_extensions(tmp_path) == []


def test_check_extensions_reports_missing_outputs(tmp_path: Path):
    _write_all_sources(tmp_path)

    issues = check_extensions(tmp_path)

    assert any("chess_anti_engine.encoding._features_ext is missing" in issue for issue in issues)
    assert any("chess_anti_engine.encoding._lc0_ext is missing" in issue for issue in issues)
    assert any("chess_anti_engine.mcts._mcts_tree is missing" in issue for issue in issues)


def test_check_extensions_treats_shared_header_as_dependency(tmp_path: Path):
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    _write(tmp_path / "chess_anti_engine/encoding/_cboard_impl.h", mtime_ns=30)

    issues = check_extensions(tmp_path)

    assert "chess_anti_engine.encoding._lc0_ext is older than chess_anti_engine/encoding/_cboard_impl.h" in issues
    assert "chess_anti_engine.mcts._mcts_tree is older than chess_anti_engine/encoding/_cboard_impl.h" in issues


def test_check_extensions_treats_feature_header_as_dependency(tmp_path: Path):
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    _write(tmp_path / "chess_anti_engine/encoding/_features_impl.h", mtime_ns=30)

    issues = check_extensions(tmp_path)

    assert (
        "chess_anti_engine.encoding._features_ext is older than "
        "chess_anti_engine/encoding/_features_impl.h"
    ) in issues
    assert (
        "chess_anti_engine.encoding._lc0_ext is older than "
        "chess_anti_engine/encoding/_features_impl.h"
    ) in issues
    assert (
        "chess_anti_engine.mcts._mcts_tree is older than "
        "chess_anti_engine/encoding/_features_impl.h"
    ) in issues


def test_check_extensions_treats_bitboard_header_as_dependency(tmp_path: Path):
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    header = "chess_anti_engine/encoding/_bitboard_planes_impl.h"
    _write(tmp_path / header, mtime_ns=30)

    issues = check_extensions(tmp_path)

    dependents = [s.module for s in EXTENSION_SPECS if header in s.dependencies]
    assert len(dependents) >= 3
    for module in dependents:
        assert f"{module} is older than {header}" in issues


def test_check_extensions_uses_python_import_suffix_order(tmp_path: Path):
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    if len(suffixes) < 2:
        pytest.skip("needs at least two extension suffixes to verify suffix-order freshness")
    module = "chess_anti_engine.encoding._features_ext"
    source = "chess_anti_engine/encoding/_features_ext.c"
    _write_all_sources(tmp_path, mtime_ns=20)
    _write_all_extensions(tmp_path, mtime_ns=30)
    _write(_extension_output(tmp_path, module, suffixes[0]), mtime_ns=10)
    _write(_extension_output(tmp_path, module, suffixes[1]), mtime_ns=40)

    issues = check_extensions(tmp_path)

    assert f"{module} is older than {source}" in issues


def test_check_extensions_can_require_production_gcc_major(tmp_path: Path) -> None:
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    for module in ALL_MODULES:
        output = _extension_output(tmp_path, module)
        output.write_bytes(b"binary\x00GCC: (Ubuntu 11.4.0) 11.4.0\x00")
        os.utime(output, ns=(20, 20))

    issues = check_extensions(tmp_path, min_gcc_major=15)

    assert len([i for i in issues if "production requires GCC >= 15" in i]) == len(ALL_MODULES)


def test_check_extensions_accepts_production_gcc_major(tmp_path: Path) -> None:
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    for module in ALL_MODULES:
        output = _extension_output(tmp_path, module)
        output.write_bytes(b"binary\x00GCC: (GNU) 15.3.0\x00")
        os.utime(output, ns=(20, 20))

    assert check_extensions(tmp_path, min_gcc_major=15) == []


def test_check_extensions_rejects_unrecorded_production_recipe(tmp_path: Path) -> None:
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)

    issues = check_extensions(tmp_path, require_production_recipe=True)

    assert len([i for i in issues if "native+LTO production recipe" in i]) == len(ALL_MODULES)


def test_check_extensions_rejects_recipe_strings_outside_the_command_line_section(
    tmp_path: Path,
) -> None:
    """Loose "-march=" / "-fltrans" bytes are what a portable -g build carries.

    This used to be the ACCEPT fixture: the gate grepped the whole file, and a
    portable LTO build's DWARF producer string supplies both substrings.
    """
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    for module in ALL_MODULES:
        output = _extension_output(tmp_path, module)
        output.write_bytes(b"ELF\x00GNU GIMPLE 15.3.0 -march=znver3 -flto -fltrans\x00")
        os.utime(output, ns=(20, 20))

    issues = check_extensions(tmp_path, require_production_recipe=True)

    assert len([i for i in issues if "native+LTO production recipe" in i]) == len(ALL_MODULES)


# ----------------------------------------------------------------------
# Real GCC artifacts: the recipe gate is judged on what GCC actually writes.
# setup.py builds with Python's CFLAGS (which carry -g) plus -O3/-flto, and
# adds -frecord-gcc-switches only when CAE_EXT_NATIVE and CAE_EXT_LTO are both
# set; the portable build pins -march=x86-64 explicitly.
# ----------------------------------------------------------------------

_GCC = shutil.which("gcc")

_RECIPES: dict[str, list[str]] = {
    # build_production_extensions.py: CAE_EXT_NATIVE=1 CAE_EXT_LTO=1
    "native_lto": ["-march=native", "-flto", "-frecord-gcc-switches"],
    # CAE_EXT_LTO=1 alone: the defect — this passed the old byte-grep gate.
    "portable_lto": ["-march=x86-64", "-flto"],
    # Portable LTO where the builder also recorded switches (e.g. via CFLAGS):
    # the section exists but names the generic baseline.
    "portable_lto_recorded": ["-march=x86-64", "-flto", "-frecord-gcc-switches"],
    "portable_v3_lto_recorded": ["-march=x86-64-v3", "-flto", "-frecord-gcc-switches"],
    # Native with recorded switches but no LTO.
    "native_recorded_no_lto": ["-march=native", "-frecord-gcc-switches"],
    # Native, compiled -flto with fat objects but linked WITHOUT LTO: the
    # recorded compile line says -flto, yet no LTO code generation ran.
    "native_fat_lto_linked_without_lto": [
        "-march=native", "-flto", "-ffat-lto-objects", "-frecord-gcc-switches",
    ],
}

_LINK_FLAGS: dict[str, list[str]] = {
    "native_fat_lto_linked_without_lto": ["-fno-lto"],
}


def _build_shared_object(out_dir: Path, recipe: str) -> Path:
    if _GCC is None:
        pytest.skip("gcc is not available")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        pytest.skip("recipe fixtures name x86-64 -march levels")
    src = out_dir / "probe.c"
    src.write_text(
        "int probe_sum(const int *xs, int n) {\n"
        "    int s = 0;\n"
        "    for (int i = 0; i < n; ++i) s += xs[i] * 3;\n"
        "    return s;\n"
        "}\n",
        encoding="utf-8",
    )
    out = out_dir / f"{recipe}.so"
    flags = _RECIPES[recipe]
    lto_link = _LINK_FLAGS.get(recipe, ["-flto"] if "-flto" in flags else [])
    obj = out_dir / f"{recipe}.o"
    subprocess.run(
        [_GCC, "-g", "-O3", "-fPIC", *flags, "-c", str(src), "-o", str(obj)],
        check=True, capture_output=True,
    )
    subprocess.run(
        [_GCC, "-shared", "-g", *lto_link, str(obj), "-o", str(out)],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def recipe_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out_dir = tmp_path_factory.mktemp("recipes")
    return {recipe: _build_shared_object(out_dir, recipe) for recipe in _RECIPES}


def test_native_lto_artifact_passes_the_production_recipe_gate(
    recipe_artifacts: dict[str, Path],
) -> None:
    assert production_recipe_problem(recipe_artifacts["native_lto"]) is None


def test_portable_lto_artifact_carries_the_strings_the_old_gate_grepped(
    recipe_artifacts: dict[str, Path],
) -> None:
    """Pins the fixture's faithfulness: the defect's preconditions really hold."""
    data = recipe_artifacts["portable_lto"].read_bytes()
    assert b"-march=" in data
    assert b"-fltrans" in data


@pytest.mark.parametrize(
    ("recipe", "reason"),
    [
        ("portable_lto", "no .GCC.command.line section"),
        ("portable_lto_recorded", "-march=x86-64 is a generic ISA baseline"),
        ("portable_v3_lto_recorded", "-march=x86-64-v3 is a generic ISA baseline"),
        ("native_recorded_no_lto", "no -flto"),
        ("native_fat_lto_linked_without_lto", "no LTO translation"),
    ],
)
def test_non_production_artifacts_fail_the_production_recipe_gate(
    recipe_artifacts: dict[str, Path], recipe: str, reason: str,
) -> None:
    problem = production_recipe_problem(recipe_artifacts[recipe])
    assert problem is not None
    assert reason in problem


@pytest.mark.parametrize("recipe", ["native_lto", "portable_lto"])
def test_check_extensions_applies_the_recipe_gate_to_real_artifacts(
    tmp_path: Path, recipe_artifacts: dict[str, Path], recipe: str,
) -> None:
    _write_all_sources(tmp_path, mtime_ns=10)
    for module in ALL_MODULES:
        output = _extension_output(tmp_path, module)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(recipe_artifacts[recipe], output)
        os.utime(output, ns=(20, 20))

    issues = check_extensions(tmp_path, require_production_recipe=True)

    if recipe == "native_lto":
        assert issues == []
    else:
        assert len([i for i in issues if "native+LTO production recipe" in i]) == len(ALL_MODULES)


# ----------------------------------------------------------------------
# Synthetic ELF images: exact control over the recorded command line, so the
# branches real GCC cannot produce (GCC's driver keeps only the last -march;
# it never writes an empty record) are still pinned. No compiler needed.
# ----------------------------------------------------------------------

_GIMPLE_LTRANS = b"GNU GIMPLE 15.3.0 -mtune=generic -march=x86-64 -g -O3 -fltrans\x00"


def _elf64(sections: list[tuple[str, bytes]], *, names_trailing_nul: bool = True) -> bytes:
    """A minimal little-endian ELF64 image holding ``sections`` (+ .shstrtab)."""
    names = b"\x00"
    name_offsets: list[int] = []
    for sec_name in [".shstrtab", *(n for n, _ in sections)]:
        name_offsets.append(len(names))
        names += sec_name.encode() + b"\x00"
    if not names_trailing_nul:
        names = names[:-1]
    payloads = [names, *(c for _, c in sections)]
    body = bytearray(64)
    data_offsets: list[int] = []
    for payload in payloads:
        data_offsets.append(len(body))
        body += payload
    shoff = len(body)
    body += b"\x00" * 64  # SHN_UNDEF entry
    for idx, (payload, off) in enumerate(zip(payloads, data_offsets)):
        sh_type = 3 if idx == 0 else 1  # SHT_STRTAB / SHT_PROGBITS
        body += struct.pack("<IIQQQQIIQQ", name_offsets[idx], sh_type, 0, 0, off, len(payload), 0, 0, 1, 0)
    header = b"\x7fELF\x02\x01\x01" + b"\x00" * 9 + struct.pack(
        "<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, len(payloads) + 1, 1,
    )
    body[:64] = header
    return bytes(body)


def _recorded(tmp_path: Path, command_line: bytes, *, name: str = "synthetic.so") -> Path:
    out = tmp_path / name
    out.write_bytes(_elf64([(".GCC.command.line", command_line), (".debug_str", _GIMPLE_LTRANS)]))
    return out


def test_synthetic_native_record_passes(tmp_path: Path) -> None:
    """Pins the builder: a well-formed recorded native+LTO image is accepted."""
    path = _recorded(tmp_path, b"GNU C23 15.3.0 -march=znver3 -mavx2 -O3 -flto\x00")
    assert production_recipe_problem(path) is None


def test_empty_command_line_section_is_rejected(tmp_path: Path) -> None:
    problem = production_recipe_problem(_recorded(tmp_path, b""))
    assert problem is not None
    assert "empty .GCC.command.line" in problem


def test_command_line_without_march_is_rejected_cleanly(tmp_path: Path) -> None:
    problem = production_recipe_problem(_recorded(tmp_path, b"GNU C23 15.3.0 -O3 -flto\x00"))
    assert problem is not None
    assert "no -march=" in problem


@pytest.mark.parametrize(
    ("command_line", "accepted"),
    [
        (b"GNU C23 15.3.0 -march=znver3 -march=x86-64 -O3 -flto\x00", False),
        (b"GNU C23 15.3.0 -march=x86-64 -march=znver3 -O3 -flto\x00", True),
        # The recording switch also reached the link: a second, link-time record.
        (b"GNU C23 15.3.0 -march=znver3 -O3 -flto\x00GNU GIMPLE 15.3.0 -march=znver3 -O3 -fltrans\x00", True),
        (b"GNU C23 15.3.0 -march=znver3 -O3 -flto\x00GNU GIMPLE 15.3.0 -march=x86-64 -O3 -fltrans\x00", False),
        (b"GNU C23 15.3.0 -march=znver3 -O3 -flto\x00GNU C23 15.3.0 -march=znver3 -O3\x00", False),
    ],
)
def test_last_recorded_march_is_the_effective_one(
    tmp_path: Path, command_line: bytes, accepted: bool,
) -> None:
    assert (production_recipe_problem(_recorded(tmp_path, command_line)) is None) is accepted


def test_section_name_without_terminating_nul_is_still_found() -> None:
    image = _elf64([(".GCC.command.line", b"payload")], names_trailing_nul=False)
    assert _elf_section(image, ".GCC.command.line") == b"payload"


def _corrupt(data: bytes, how: str) -> bytes:
    buf = bytearray(data)
    if how == "flipped_endianness":
        buf[5] = 2
    elif how == "huge_shoff":
        struct.pack_into("<Q", buf, 0x28, 0xFFFF_FFFF_FFFF_0000)
    elif how == "shstrndx_out_of_range":
        struct.pack_into("<H", buf, 0x3E, 0xFFFF)
    elif how == "tiny_shentsize":
        struct.pack_into("<H", buf, 0x3A, 4)
    elif how == "truncated_header":
        del buf[40:]
    elif how == "truncated_tail":
        del buf[len(buf) // 2:]
    elif how == "section_past_eof":
        shoff = struct.unpack_from("<Q", buf, 0x28)[0]
        shnum = struct.unpack_from("<H", buf, 0x3C)[0]
        for i in range(1, shnum):
            struct.pack_into("<Q", buf, shoff + i * 64 + 0x18, len(buf) + 4096)
    else:
        raise AssertionError(how)
    return bytes(buf)


_CORRUPTIONS = (
    "flipped_endianness", "huge_shoff", "shstrndx_out_of_range", "tiny_shentsize",
    "truncated_header", "truncated_tail", "section_past_eof",
)


@pytest.mark.parametrize("how", _CORRUPTIONS)
def test_corrupt_elf_is_a_clean_rejection_not_a_traceback(tmp_path: Path, how: str) -> None:
    good = _recorded(tmp_path, b"GNU C23 15.3.0 -march=znver3 -O3 -flto\x00", name="good.so")
    bad = tmp_path / "bad.so"
    bad.write_bytes(_corrupt(good.read_bytes(), how))

    problem = production_recipe_problem(bad)

    assert problem is not None
    assert problem.startswith("unreadable ELF"), problem


@pytest.mark.parametrize("how", _CORRUPTIONS)
def test_corrupt_real_artifact_is_a_clean_rejection(
    tmp_path: Path, recipe_artifacts: dict[str, Path], how: str,
) -> None:
    bad = tmp_path / "bad.so"
    bad.write_bytes(_corrupt(recipe_artifacts["native_lto"].read_bytes(), how))

    problem = production_recipe_problem(bad)

    assert problem is not None
    assert problem.startswith("unreadable ELF"), problem


# ======================================================================
# Every COMPILED extension is registered, and every header it really
# includes is a declared dependency
# ======================================================================


def test_every_built_extension_is_registered_with_the_guard() -> None:
    """setup.py's Extension list and EXTENSION_SPECS must name the same modules.

    ⚑ This is the gap that shipped: a fourth extension was added to setup.py and
    the freshness guard still knew only three, so `scripts/train.sh` and
    `graceful_restart.py` reported "extensions up to date" for a tree whose NNUE
    evaluator was stale or absent. A guard that does not know a module exists
    cannot report it out of date — it reports success, which is the worse of the
    two failures. Deriving the expectation from setup.py means registering the
    next extension is not something anyone has to remember.
    """
    setup_src = (REPO_ROOT / "setup.py").read_text()
    built = set(re.findall(r'Extension\(\s*"([\w.]+)"', setup_src))

    assert built, "could not read any Extension() names out of setup.py"
    assert built == set(ALL_MODULES), (
        "setup.py builds "
        f"{sorted(built)} but the freshness guard tracks {sorted(ALL_MODULES)}; "
        "an untracked extension is reported as up to date when it is stale"
    )


@pytest.mark.parametrize("spec", EXTENSION_SPECS, ids=lambda s: s.module)
def test_declared_dependencies_cover_every_header_actually_included(spec) -> None:
    """The spec must list every in-repo header the module transitively includes.

    Walking the real #include graph rather than trusting the hand-written list is
    what keeps this honest: the failure mode is not a wrong entry, it is a
    MISSING one, and a missing entry is invisible until someone edits that header
    and the binary silently keeps running the old code.
    """
    source = _module_source(spec.module)
    assert source.exists(), f"{source} does not exist"

    actual = _transitive_local_includes(source)
    declared = set(spec.dependencies)
    missing = actual - declared

    assert not missing, (
        f"{spec.module} includes {sorted(missing)} but does not declare "
        "them as freshness dependencies; editing one of those headers would "
        "leave the guard saying the extension is up to date"
    )


@pytest.mark.parametrize(
    "header",
    [
        "chess_anti_engine/nnue/_nnue_impl.h",
        "chess_anti_engine/nnue/_nnue_provider.h",
        "chess_anti_engine/mcts/_value_provider.h",
    ],
)
def test_nnue_extension_is_rebuilt_when_an_evaluator_header_changes(
    tmp_path: Path, header: str
) -> None:
    """Touching any evaluator header must make the NNUE extension read as stale."""
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    _write(tmp_path / header, mtime_ns=30)

    issues = check_extensions(tmp_path)

    assert f"chess_anti_engine.nnue._nnue_ext is older than {header}" in issues


def test_tree_extension_is_rebuilt_when_the_seam_contract_changes(tmp_path: Path) -> None:
    """The tree compiles the seam's contract in, so it depends on that header.

    It does NOT depend on any provider's implementation header — providers reach
    it through a runtime capsule — so _nnue_impl.h is deliberately absent from
    the tree's dependency list rather than forgotten.
    """
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    _write(tmp_path / "chess_anti_engine/mcts/_value_provider.h", mtime_ns=30)

    issues = check_extensions(tmp_path)

    assert (
        "chess_anti_engine.mcts._mcts_tree is older than "
        "chess_anti_engine/mcts/_value_provider.h"
    ) in issues


# ======================================================================
# Optional extensions: missing is allowed, present-but-stale is not
# ======================================================================

_OPTIONAL = [s.module for s in EXTENSION_SPECS if s.optional]


def test_the_tablebase_prober_is_optional_until_production_imports_it() -> None:
    assert _OPTIONAL == ["chess_anti_engine.tb._tb_ext"]
    assert all(not s.optional for s in EXTENSION_SPECS if s.module not in _OPTIONAL)


def test_a_missing_optional_extension_is_not_an_issue(tmp_path: Path) -> None:
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    for module in _OPTIONAL:
        _extension_output(tmp_path, module).unlink()

    assert check_extensions(tmp_path) == []


def test_a_stale_optional_extension_is_still_an_issue(tmp_path: Path) -> None:
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    _write(tmp_path / "chess_anti_engine/tb/fathom/src/tbprobe.c", mtime_ns=30)

    issues = check_extensions(tmp_path)

    assert issues == [
        "chess_anti_engine.tb._tb_ext is older than chess_anti_engine/tb/fathom/src/tbprobe.c"
    ]
