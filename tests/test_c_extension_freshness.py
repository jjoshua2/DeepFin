from __future__ import annotations

import importlib.machinery
import os
from pathlib import Path

from scripts.check_c_extensions_fresh import check_extensions


def _write(path: Path, *, mtime_ns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _extension_output(root: Path, module: str, suffix: str | None = None) -> Path:
    base = root.joinpath(*module.split("."))
    return base.with_name(base.name + (suffix or importlib.machinery.EXTENSION_SUFFIXES[0]))


def _write_all_sources(root: Path, *, mtime_ns: int = 10) -> None:
    for rel in (
        "chess_anti_engine/encoding/_features_ext.c",
        "chess_anti_engine/encoding/_lc0_ext.c",
        "chess_anti_engine/encoding/_cboard_impl.h",
        "chess_anti_engine/encoding/_features_impl.h",
        "chess_anti_engine/mcts/_mcts_tree.c",
    ):
        _write(root / rel, mtime_ns=mtime_ns)


def _write_all_extensions(root: Path, *, mtime_ns: int = 20) -> None:
    for module in (
        "chess_anti_engine.encoding._features_ext",
        "chess_anti_engine.encoding._lc0_ext",
        "chess_anti_engine.mcts._mcts_tree",
    ):
        _write(_extension_output(root, module), mtime_ns=mtime_ns)


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


def test_check_extensions_uses_python_import_suffix_order(tmp_path: Path):
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    assert len(suffixes) >= 2
    module = "chess_anti_engine.encoding._features_ext"
    source = "chess_anti_engine/encoding/_features_ext.c"
    _write_all_sources(tmp_path, mtime_ns=20)
    _write_all_extensions(tmp_path, mtime_ns=30)
    _write(_extension_output(tmp_path, module, suffixes[0]), mtime_ns=10)
    _write(_extension_output(tmp_path, module, suffixes[1]), mtime_ns=40)

    issues = check_extensions(tmp_path)

    assert f"{module} is older than {source}" in issues
