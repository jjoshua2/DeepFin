from __future__ import annotations

import importlib.machinery
import os
import re
from pathlib import Path

import pytest

from scripts import check_c_extensions_fresh as freshness
from scripts.check_c_extensions_fresh import (
    EXTENSION_SPECS,
    ExtensionSpec,
    NATIVE_BUILD_ATTESTED_MODULES,
    check_extensions,
    extension_spec,
    native_build_attestation,
    native_build_dependency_paths,
    require_current_native_build_attestation_schema,
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


@pytest.mark.parametrize("module", NATIVE_BUILD_ATTESTED_MODULES)
def test_native_build_attestation_covers_every_declared_input(module: str) -> None:
    dependencies = {
        path: f"input:{path}".encode()
        for path in extension_spec(module).dependencies
    }
    attestation = native_build_attestation(module, "a" * 40, dependencies)

    assert set(attestation["dependencies"]) == set(dependencies)
    changed = dict(dependencies)
    changed[next(iter(changed))] += b" changed"
    assert (
        native_build_attestation(module, "a" * 40, changed)["input_sha256"]
        != attestation["input_sha256"]
    )
    assert (
        native_build_attestation(module, "b" * 40, dependencies)["input_sha256"]
        != attestation["input_sha256"]
    )


def test_current_extension_specs_match_published_attestation_schema() -> None:
    require_current_native_build_attestation_schema()


def test_published_attestation_schema_rejects_unknown_schema_and_input_set() -> None:
    module = NATIVE_BUILD_ATTESTED_MODULES[0]
    dependencies = {
        path: path.encode()
        for path in native_build_dependency_paths(
            freshness.NATIVE_BUILD_ATTESTATION_SCHEMA, module,
        )
    }

    with pytest.raises(ValueError, match="unknown native build attestation schema"):
        native_build_attestation(
            module, "a" * 40, dependencies, schema="deepfin.native_build.unknown",
        )
    with pytest.raises(ValueError, match="build inputs mismatch"):
        native_build_attestation(
            module, "a" * 40, {**dependencies, "unexpected.h": b"unexpected"},
        )


def test_v1_attestation_digest_semantics_are_pinned() -> None:
    module = "chess_anti_engine.encoding._features_ext"
    schema = "deepfin.native_build.v1"
    dependencies = {
        path: f"known-vector:{path}\n".encode()
        for path in native_build_dependency_paths(schema, module)
    }

    attestation = native_build_attestation(
        module, "0123456789abcdef0123456789abcdef01234567", dependencies,
        schema=schema,
    )

    assert attestation["input_sha256"] == (
        "1e32d91a3a7d04b54f131d4b11cd580b9529c9e34730686c54813f71c49ff42d"
    )


def test_v1_registry_survives_a_current_schema_bump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = "chess_anti_engine.encoding._features_ext"
    v1_paths = native_build_dependency_paths("deepfin.native_build.v1", module)
    monkeypatch.setattr(
        freshness, "NATIVE_BUILD_ATTESTATION_SCHEMA", "deepfin.native_build.v2",
    )

    assert native_build_dependency_paths("deepfin.native_build.v1", module) == v1_paths
    with pytest.raises(RuntimeError, match="unpublished; publish a new schema"):
        require_current_native_build_attestation_schema()


def test_current_spec_growth_requires_a_new_attestation_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = NATIVE_BUILD_ATTESTED_MODULES[0]
    grown = ExtensionSpec(
        module,
        (*extension_spec(module).dependencies, "chess_anti_engine/future_header.h"),
    )
    monkeypatch.setattr(
        freshness,
        "EXTENSION_SPECS",
        tuple(grown if spec.module == module else spec for spec in EXTENSION_SPECS),
    )

    with pytest.raises(RuntimeError, match="publish a new schema"):
        require_current_native_build_attestation_schema()


def test_check_extensions_rejects_unknown_module_filter(tmp_path: Path) -> None:
    assert check_extensions(tmp_path, modules={"misspelled.module"}) == [
        "unknown extension module requested: misspelled.module"
    ]


def test_check_extensions_binds_freshness_to_loaded_binary_path(tmp_path: Path) -> None:
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    module = "chess_anti_engine.mcts._mcts_tree"
    inspected = _extension_output(tmp_path, module).resolve()
    other = (tmp_path / "other-worktree/_mcts_tree.so").resolve()

    issues = check_extensions(
        tmp_path, modules={module}, loaded_paths={module: other},
    )

    assert issues == [
        f"{module} loaded from {other} but freshness inspected {inspected}"
    ]


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


def test_check_extensions_accepts_recorded_production_recipe(tmp_path: Path) -> None:
    _write_all_sources(tmp_path, mtime_ns=10)
    _write_all_extensions(tmp_path, mtime_ns=20)
    for module in ALL_MODULES:
        output = _extension_output(tmp_path, module)
        output.write_bytes(b"ELF\x00-march=znver3\x00-fltrans\x00")
        os.utime(output, ns=(20, 20))

    assert check_extensions(tmp_path, require_production_recipe=True) == []


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
