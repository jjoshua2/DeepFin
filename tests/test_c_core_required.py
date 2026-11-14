"""The compiled C core is required; a stale or partial build refuses, never falls back.

Each test removes ONE symbol from the imported extension module object (or
forces the import-time flag a missing symbol would have produced) and asserts
the entry point refuses with a message naming it. The paired opt-in tests show
the Python reference path still runs inside ``python_reference_paths``.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import chess
import numpy as np
import pytest

import chess_anti_engine.encoding._lc0_ext as lc0_ext
import chess_anti_engine.mcts._mcts_tree as mcts_tree
from chess_anti_engine import ext_check
from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.ext_check import CCoreIncompleteError, python_reference_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "chess_anti_engine"

SELFPLAY_TREE_SYMBOLS = (
    "MCTSTree", "batch_process_ply",
    "batch_encode_146", "batch_encode_146_lc0_root", "batch_encode_146_lc0_root_legacy_meta",
    "batch_encode_146_bf16", "batch_encode_146_lc0_root_bf16",
    "batch_encode_146_lc0_root_legacy_meta_bf16",
    "classify_games", "temperature_resample",
)


# ── the required set itself ───────────────────────────────────────────────


def test_this_build_is_complete() -> None:
    assert ext_check.extension_problems() == []


def _all_required_names() -> set[str]:
    names = {sym for syms in ext_check.REQUIRED_SYMBOLS.values() for sym in syms}
    names |= set(ext_check.REQUIRED_CBOARD_METHODS)
    names |= set(ext_check.REQUIRED_MCTSTREE_METHODS)
    return names


def _extension_object_names() -> set[str]:
    names: set[str] = set()
    for mod_name in ext_check.REQUIRED_EXTENSIONS:
        names |= set(dir(importlib.import_module(mod_name)))
    names |= set(dir(CBoard))
    names |= set(dir(mcts_tree.MCTSTree))
    return names


_EXT_MODULES = tuple(ext_check.REQUIRED_EXTENSIONS)
# Handlers that turn a missing symbol into a quiet alternative path.
_SWALLOWING_HANDLERS = {
    "ImportError", "ModuleNotFoundError", "AttributeError", "Exception", "BaseException",
}


def _package_of(path: Path) -> str:
    """Dotted package a source file's relative imports resolve against."""
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    parts.pop()  # the module itself; for __init__ this leaves the package
    return ".".join(parts)


def _abs_module(node: ast.ImportFrom, pkg: str) -> str | None:
    """``node``'s module as an absolute dotted name (``from ._lc0_ext import x`` too)."""
    if not node.level:
        return node.module
    base = pkg.split(".")
    base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
    return ".".join([*base, node.module] if node.module else base)


def _ext_aliases(tree: ast.AST, pkg: str = "") -> dict[str, str]:
    """Local names bound to a required extension module (``import x as m`` etc.)."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _EXT_MODULES:
                    aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and _abs_module(node, pkg):
            for alias in node.names:
                full = f"{_abs_module(node, pkg)}.{alias.name}"
                if full in _EXT_MODULES:
                    aliases[alias.asname or alias.name] = full
    return aliases


def _swallows(node: ast.Try) -> bool:
    for handler in node.handlers:
        if handler.type is None:  # bare except
            return True
        types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        if {t.id for t in types if isinstance(t, ast.Name)} & _SWALLOWING_HANDLERS:
            return True
    return False


def _ext_reads(node: ast.AST, aliases: dict[str, str], pkg: str = "") -> list[tuple[int, str]]:
    """(line, name) for every extension symbol ``node`` imports or reads.

    ``from <ext> import name`` (absolute or relative, e.g. ``from ._lc0_ext import
    CBoard``), ``<alias>.name`` / ``<dotted ext>.name``, and a whole-module import
    (``import <ext> [as m]`` / ``importlib.import_module("<ext>")``) reported as the
    module's own dotted name.
    """
    out: list[tuple[int, str]] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.ImportFrom):
            mod = _abs_module(sub, pkg)
            if mod in _EXT_MODULES:
                out += [(sub.lineno, alias.name) for alias in sub.names]
            elif mod:
                out += [
                    (sub.lineno, f"{mod}.{alias.name}") for alias in sub.names
                    if f"{mod}.{alias.name}" in _EXT_MODULES
                ]
        elif isinstance(sub, ast.Import):
            out += [(sub.lineno, alias.name) for alias in sub.names if alias.name in _EXT_MODULES]
        elif (
            isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "import_module" and sub.args
            and isinstance(sub.args[0], ast.Constant) and sub.args[0].value in _EXT_MODULES
        ):
            out.append((sub.lineno, str(sub.args[0].value)))
        elif isinstance(sub, ast.Attribute):
            base = ast.unparse(sub.value)
            if base in aliases or base in _EXT_MODULES:
                out.append((sub.lineno, sub.attr))
    return out


def _guarded_probes() -> list[tuple[str, int, str]]:
    """Every (file, line, name) where package code treats a C symbol as optional.

    ``hasattr(x, "name")`` / ``getattr(x, "name", ...)`` whose name exists on an
    extension module, ``CBoard`` or ``MCTSTree``; and any extension import or
    attribute read inside a ``try`` whose handler swallows ImportError,
    ModuleNotFoundError, AttributeError, Exception, BaseException or anything
    (bare ``except``).
    """
    on_ext = _extension_object_names()
    found: list[tuple[str, int, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        pkg = _package_of(path)
        aliases = _ext_aliases(tree, pkg)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("hasattr", "getattr") and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and node.args[1].value in on_ext
            ):
                found.append((rel, node.lineno, node.args[1].value))
            if isinstance(node, ast.Try) and _swallows(node):
                for stmt in node.body:
                    found += [(rel, ln, name) for ln, name in _ext_reads(stmt, aliases, pkg)]
    return found


def _all_ext_references() -> list[tuple[str, int, str]]:
    """Every extension symbol the package imports or reads, guarded or not."""
    found: list[tuple[str, int, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        pkg = _package_of(path)
        found += [(rel, ln, name) for ln, name in _ext_reads(tree, _ext_aliases(tree, pkg), pkg)]
    return found


def _accepted_names() -> set[str]:
    return _all_required_names() | set(ext_check.GUARDED_OPTIONAL_ALLOWLIST) | set(_EXT_MODULES)


def test_every_guarded_c_symbol_is_required_or_allow_listed() -> None:
    """Adding a new ``try: import``/``hasattr`` fallback on a C symbol fails here.

    List the symbol in ``ext_check.REQUIRED_SYMBOLS`` (or the CBoard/MCTSTree
    method tuples) so every entry point refuses a build without it, or add it
    to ``GUARDED_OPTIONAL_ALLOWLIST`` with the reason it is not a C fallback.
    """
    allowed = set(ext_check.GUARDED_OPTIONAL_ALLOWLIST)
    probes = _guarded_probes()
    unlisted = sorted({(f, ln, n) for f, ln, n in probes if n not in _accepted_names()})
    assert not unlisted, (
        f"package code treats these C extension symbols as optional: {unlisted}. "
        "Either add each one to chess_anti_engine/ext_check.py REQUIRED_SYMBOLS "
        "(or REQUIRED_CBOARD_METHODS / REQUIRED_MCTSTREE_METHODS) so every entry point "
        "refuses a build without it, or, if it is genuinely not a C fallback, add it "
        "to GUARDED_OPTIONAL_ALLOWLIST there with a comment giving the reason."
    )
    # The scan is live: it sees the known sites (a scan that finds nothing
    # would pass the assertion above vacuously).
    names = {n for _, _, n in probes}
    for known in ("batch_process_ply", "classify_games", "encode_full", "encode_full_batch",
                  "root_terminal_actions", "set_history_rep_fix", "batch_encode_146_bf16",
                  "legal_move_policy_indices", "encode_piece_planes", "compute_extra_features",
                  "continue_gumbel_sims_legal_bf16", "reset_compact", "batch_descend_puct_tb",
                  "tt_stats", "GSS_HALVING_REV"):
        assert known in names, known
    # Every allow-list entry is still used, so it cannot quietly cover a new site.
    assert allowed <= names | {n for _, _, n in _all_ext_references()}, allowed - names


def test_every_c_symbol_the_package_uses_is_required() -> None:
    """Unguarded imports/reads too: a symbol the code needs must be in the startup check.

    Otherwise a build lacking it passes every startup gate and fails later, at
    import or mid-session.
    """
    refs = _all_ext_references()
    unlisted = sorted({(f, ln, n) for f, ln, n in refs if n not in _accepted_names()})
    assert not unlisted, (
        f"package code uses these C extension symbols but the startup check does not "
        f"require them: {unlisted}. Add each one to chess_anti_engine/ext_check.py "
        "REQUIRED_SYMBOLS (or REQUIRED_CBOARD_METHODS / REQUIRED_MCTSTREE_METHODS) so a "
        "build without it is refused at startup instead of failing mid-session; if it "
        "is genuinely not a C requirement, add it to GUARDED_OPTIONAL_ALLOWLIST there "
        "with a comment giving the reason."
    )
    names = {n for _, _, n in refs}
    # The relative import in encoding/perft.py is seen, not only absolute ones.
    assert ("chess_anti_engine/encoding/perft.py", "CBoard") in {(f, n) for f, _, n in refs}
    for known in ("prepare_sf_multipv", "batch_compute_relations", "SWDL_DRAW_NET_RAW",
                  "SWDL_DRAW_PARAMETRIC_Q", "GSS_MAX_CANDS", "MCTSTree", "CBoard"):
        assert known in names, known


_SYNTHETIC_PATTERNS = {
    "hasattr_probe": (
        "import chess_anti_engine.mcts._mcts_tree as t\n"
        "if hasattr(t, 'slider_selftest'):\n    pass\n",
        {"slider_selftest"},
    ),
    "import_error_from_import": (
        "try:\n    from chess_anti_engine.encoding._lc0_ext import slider_selftest\n"
        "except ImportError:\n    slider_selftest = None\n",
        {"slider_selftest"},
    ),
    "attribute_error_read": (
        "import chess_anti_engine.mcts._mcts_tree as t\n"
        "try:\n    t.slider_selftest()\nexcept AttributeError:\n    pass\n",
        {"slider_selftest"},
    ),
    "bare_except_import": (
        "try:\n    from chess_anti_engine.mcts._mcts_tree import slider_selftest\n"
        "except:\n    pass\n",
        {"slider_selftest"},
    ),
    "base_exception_import": (
        "try:\n    from chess_anti_engine.mcts._mcts_tree import slider_selftest\n"
        "except BaseException:\n    pass\n",
        {"slider_selftest"},
    ),
    "import_as_in_try": (
        "try:\n    import chess_anti_engine.encoding._perft_like as m\n"
        "    import chess_anti_engine.mcts._mcts_tree as t\n    t.slider_selftest\n"
        "except ImportError:\n    t = None\n",
        {"chess_anti_engine.mcts._mcts_tree", "slider_selftest"},
    ),
    "relative_import_in_try": (
        "try:\n    from ._lc0_ext import slider_selftest\n"
        "except ImportError:\n    slider_selftest = None\n",
        {"slider_selftest"},
    ),
    "import_module_in_try": (
        "import importlib\n"
        "try:\n    importlib.import_module('chess_anti_engine.encoding._features_ext')\n"
        "except Exception:\n    pass\n",
        {"chess_anti_engine.encoding._features_ext"},
    ),
}


@pytest.mark.parametrize("pattern", sorted(_SYNTHETIC_PATTERNS))
def test_scan_catches_each_guarded_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pattern: str,
) -> None:
    """The scanner itself, on one synthetic module per pattern it must see."""
    source, want = _SYNTHETIC_PATTERNS[pattern]
    fake_pkg = tmp_path / "chess_anti_engine"
    (fake_pkg / "encoding").mkdir(parents=True)
    # Under encoding/ so the relative pattern resolves to encoding._lc0_ext.
    (fake_pkg / "encoding" / "newmod.py").write_text(source)
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE", fake_pkg)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    names = {n for _, _, n in _guarded_probes()}
    assert names == want


def test_reference_scan_sees_unguarded_imports_and_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pkg = tmp_path / "chess_anti_engine"
    fake_pkg.mkdir()
    (fake_pkg / "newmod.py").write_text(
        "from chess_anti_engine.encoding._lc0_ext import slider_selftest\n"
        "import chess_anti_engine.mcts._mcts_tree as t\n"
        "from chess_anti_engine.mcts import _mcts_tree as t2\n"
        "x = t.SLIDER_BACKEND\ny = t2.tt_stats\n"
        "z = chess_anti_engine.encoding._features_ext.compute_extra_features\n"
    )
    (fake_pkg / "encoding").mkdir()
    (fake_pkg / "encoding" / "rel.py").write_text("from ._lc0_ext import RELATIVE_ONLY\n")
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE", fake_pkg)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    names = {n for _, _, n in _all_ext_references()}
    assert {"slider_selftest", "SLIDER_BACKEND", "tt_stats", "compute_extra_features"} <= names
    assert {n for n in names if n not in _accepted_names()} == {
        "slider_selftest", "SLIDER_BACKEND", "RELATIVE_ONLY",
    }


@pytest.mark.parametrize(
    ("module", "symbol"),
    [(m, s) for m, syms in ext_check.REQUIRED_SYMBOLS.items() for s in syms if s != "ABI_VERSION"],
)
def test_startup_check_names_each_missing_symbol(
    monkeypatch: pytest.MonkeyPatch, module: str, symbol: str,
) -> None:
    mod = importlib.import_module(module)
    monkeypatch.delattr(mod, symbol)
    problems = ext_check.extension_problems(abi_required=lambda: 0)
    assert any(p.startswith(f"{module} lacks ") and symbol in p for p in problems), problems
    with pytest.raises(CCoreIncompleteError, match=symbol):
        ext_check.require_c_core("test", abi_required=lambda: 0)


@pytest.mark.parametrize("method", ext_check.REQUIRED_CBOARD_METHODS)
def test_startup_check_names_missing_cboard_method(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    stand_in = type("CBoard", (), {m: None for m in dir(CBoard) if m != method and not m.startswith("__")})
    monkeypatch.setattr(lc0_ext, "CBoard", stand_in)
    problems = ext_check.extension_problems(abi_required=lambda: 0)
    assert any(f"CBoard.{method}" in p for p in problems), problems


@pytest.mark.parametrize("method", ext_check.REQUIRED_MCTSTREE_METHODS)
def test_startup_check_names_missing_tree_method(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    stand_in = type("MCTSTree", (), {
        m: None for m in dir(mcts_tree.MCTSTree) if m != method and not m.startswith("__")
    })
    monkeypatch.setattr(mcts_tree, "MCTSTree", stand_in)
    problems = ext_check.extension_problems(abi_required=lambda: 0)
    assert any(f"MCTSTree.{method}" in p for p in problems), problems


def test_startup_check_flags_old_abi() -> None:
    problems = ext_check.extension_problems(abi_required=lambda: 10_000)
    assert any("ABI_VERSION=" in p and "< required 10000" in p for p in problems)


def test_uci_gate_uses_the_shared_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.uci import selfcheck as sc

    assert sc.REQUIRED_SYMBOLS is ext_check.REQUIRED_SYMBOLS
    assert sc.REBUILD_HINT == ext_check.REBUILD_HINT
    monkeypatch.delattr(mcts_tree, "batch_process_ply")
    assert any("batch_process_ply" in p for p in sc.startup_problems(check_freshness=False))


def _raise_inside_opt_in(reason: str) -> None:
    with python_reference_paths(reason):
        raise RuntimeError(reason)


def test_opt_in_is_released_when_the_body_raises() -> None:
    assert not ext_check.python_reference_allowed()
    with pytest.raises(RuntimeError, match="single"):
        _raise_inside_opt_in("single")
    assert not ext_check.python_reference_allowed()
    with python_reference_paths("outer"):
        with pytest.raises(RuntimeError, match="inner"):
            _raise_inside_opt_in("inner")
        assert ext_check.python_reference_allowed(), "outer opt-in lost after inner raise"
    assert not ext_check.python_reference_allowed()


def test_opt_in_needs_a_reason_and_is_scoped() -> None:
    with pytest.raises(ValueError, match="non-empty reason"), python_reference_paths("  "):
        pass
    assert not ext_check.python_reference_allowed()
    with python_reference_paths("unit test"):
        assert ext_check.python_reference_allowed()
        with python_reference_paths("nested"):
            assert ext_check.python_reference_allowed()
        assert ext_check.python_reference_allowed()
    assert not ext_check.python_reference_allowed()


# ── selfplay ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("symbol", SELFPLAY_TREE_SYMBOLS)
def test_selfplay_state_create_refuses_each_missing_symbol(
    monkeypatch: pytest.MonkeyPatch, symbol: str,
) -> None:
    from chess_anti_engine.selfplay.state import _probe_c_extensions

    monkeypatch.delattr(mcts_tree, symbol)
    with pytest.raises(CCoreIncompleteError, match=symbol):
        _probe_c_extensions()


@pytest.mark.parametrize("symbol", SELFPLAY_TREE_SYMBOLS)
def test_selfplay_probe_opt_in_takes_the_reference_path(
    monkeypatch: pytest.MonkeyPatch, symbol: str,
) -> None:
    from chess_anti_engine.selfplay.state import _probe_c_extensions

    monkeypatch.delattr(mcts_tree, symbol)
    with python_reference_paths("test of the reference selfplay path"):
        caps = _probe_c_extensions()
    if symbol == "MCTSTree":
        assert caps.mcts_tree is None
    elif symbol in ("classify_games", "temperature_resample"):
        assert not caps.has_classify_c
    else:
        assert not caps.has_c_ply


def _play_batch_kwargs() -> dict[str, Any]:
    from chess_anti_engine.selfplay.config import (
        DiffFocusConfig, GameConfig, SearchConfig, TemperatureConfig,
    )
    from tests.selfplay_helpers import FakeStockfish, UniformPolicyValueModel

    return {
        "device": "cpu", "rng": np.random.default_rng(0),
        "stockfish": FakeStockfish([0.0, 1.0, 0.0]), "games": 2,
        "temp": TemperatureConfig(temperature=1.0),
        "search": SearchConfig(simulations=2, playout_cap_fraction=1.0, fast_simulations=2,
                               mcts_type="gumbel"),
        "diff_focus": DiffFocusConfig(enabled=False),
        "game": GameConfig(max_plies=4, selfplay_fraction=1.0),
        "model": UniformPolicyValueModel().eval(),
    }


def test_play_batch_refuses_a_missing_per_ply_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.selfplay import play_batch

    monkeypatch.delattr(mcts_tree, "batch_process_ply")
    kwargs = _play_batch_kwargs()
    model = kwargs.pop("model")
    with pytest.raises(CCoreIncompleteError, match="batch_process_ply"):
        play_batch(model, **kwargs)


@pytest.mark.parametrize("flag", ["_HAS_GUMBEL_C", "_HAS_C_TREE"])
def test_play_batch_refuses_when_a_compiled_runner_did_not_import(
    monkeypatch: pytest.MonkeyPatch, flag: str,
) -> None:
    from chess_anti_engine.selfplay import network_turn as nt
    from chess_anti_engine.selfplay import play_batch

    monkeypatch.setattr(nt, flag, False)
    kwargs = _play_batch_kwargs()
    model = kwargs.pop("model")
    with pytest.raises(CCoreIncompleteError, match="pure-Python search"):
        play_batch(model, **kwargs)


@pytest.mark.parametrize(("flag", "mcts_type", "sims", "what"), [
    ("_HAS_GUMBEL_C", "gumbel", 2, "pure-Python Gumbel search"),
    ("_HAS_C_TREE", "puct", 80, "pure-Python PUCT search"),
])
def test_network_turn_refuses_the_python_search_at_point_of_use(
    monkeypatch: pytest.MonkeyPatch, flag: str, mcts_type: str, sims: int, what: str,
) -> None:
    """Behind the play_batch check: the dispatch itself refuses (direct callers, tests)."""
    from chess_anti_engine.selfplay import manager
    from chess_anti_engine.selfplay import network_turn as nt
    from chess_anti_engine.selfplay.config import SearchConfig

    monkeypatch.setattr(manager, "_require_compiled_search", lambda: None)
    monkeypatch.setattr(nt, flag, False)
    kwargs = _play_batch_kwargs()
    model = kwargs.pop("model")
    kwargs["search"] = SearchConfig(simulations=sims, playout_cap_fraction=1.0,
                                    fast_simulations=2, mcts_type=mcts_type)
    with pytest.raises(CCoreIncompleteError, match=what):
        manager.play_batch(model, **kwargs)


def test_play_batch_runs_on_a_complete_build() -> None:
    from chess_anti_engine.selfplay import play_batch

    kwargs = _play_batch_kwargs()
    model = kwargs.pop("model")
    samples, stats = play_batch(model, **kwargs)
    assert stats.games == 2
    assert len(samples) > 0


def test_python_per_ply_recorder_refuses_without_opt_in() -> None:
    from chess_anti_engine.selfplay.network_turn import _append_records_via_python

    with pytest.raises(CCoreIncompleteError, match="Python per-ply recorder"):
        _append_records_via_python(
            cast(Any, None), [0], xs_batch=None, pol_logits=np.zeros((1, 1)), wdl_est=np.zeros((1, 3)),
            probs_list=[None], actions=[None], values_list=[None], gumbel_diags=[None],
            masks_list=[None], is_full=np.zeros(1, dtype=bool), sample_weights=[1.0],
            diff_focus=None,
        )


# ── arena ──────────────────────────────────────────────────────────────────


def _arena(monkeypatch: pytest.MonkeyPatch, flag: str, mcts_type: str) -> None:
    import torch

    from chess_anti_engine.moves import POLICY_SIZE
    from chess_anti_engine.selfplay import match as match_mod

    class _Dummy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            bs = int(x.shape[0])
            return {"policy_own": torch.zeros((bs, POLICY_SIZE)), "wdl": torch.zeros((bs, 3))}

    monkeypatch.setattr(match_mod, flag, False)
    m = _Dummy().eval()
    match_mod.play_match_batch(
        m, m, device="cpu", rng=np.random.default_rng(0), games=2, max_plies=2,
        a_plays_white=[True, False], mcts_type=mcts_type, mcts_simulations=2,
    )


@pytest.mark.parametrize(("flag", "mcts_type"), [("_HAS_GUMBEL_C", "gumbel"), ("_HAS_C_TREE", "puct")])
def test_arena_refuses_a_missing_compiled_runner(
    monkeypatch: pytest.MonkeyPatch, flag: str, mcts_type: str,
) -> None:
    with pytest.raises(CCoreIncompleteError, match="arena move selection"):
        _arena(monkeypatch, flag, mcts_type)


@pytest.mark.parametrize(("flag", "mcts_type"), [("_HAS_GUMBEL_C", "gumbel"), ("_HAS_C_TREE", "puct")])
@pytest.mark.usefixtures("python_reference_paths")
def test_arena_reference_search_runs_with_opt_in(
    monkeypatch: pytest.MonkeyPatch, flag: str, mcts_type: str,
) -> None:
    _arena(monkeypatch, flag, mcts_type)


def test_gumbel_c_refuses_missing_bf16_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.mcts import gumbel_c
    from chess_anti_engine.mcts.gumbel import GumbelConfig

    monkeypatch.setattr(gumbel_c, "batch_encode_146_lc0_root_bf16", None)
    with pytest.raises(CCoreIncompleteError, match="bf16"):
        gumbel_c.run_gumbel_root_many_c(
            None, [chess.Board()], device="cpu", rng=np.random.default_rng(0), cfg=GumbelConfig(),
        )


@pytest.mark.parametrize("method", ext_check.REQUIRED_MCTSTREE_METHODS)
def test_selfplay_probe_refuses_a_tree_missing_a_method(
    monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    from chess_anti_engine.selfplay.state import _probe_c_extensions

    real = mcts_tree.MCTSTree
    stand_in = type("MCTSTree", (), {
        m: getattr(real, m) for m in dir(real) if m != method and not m.startswith("__")
    })
    monkeypatch.setattr(mcts_tree, "MCTSTree", stand_in)
    with pytest.raises(CCoreIncompleteError, match=f"MCTSTree.{method}"):
        _probe_c_extensions()


class _LegalBf16Evaluator:
    supports_legal_bf16 = True

    def evaluate_legal_bf16(self, *_a: Any) -> Any:  # never reached
        raise AssertionError("evaluated before the tree check")


class _PlainEvaluator:
    def evaluate_encoded(self, *_a: Any) -> Any:
        raise RuntimeError("plain evaluator reached evaluation")


def test_gumbel_c_refuses_a_tree_without_the_legal_bf16_methods() -> None:
    from chess_anti_engine.mcts import gumbel_c
    from chess_anti_engine.mcts.gumbel import GumbelConfig

    tree = cast(Any, _Without(mcts_tree.MCTSTree(), "continue_gumbel_sims_legal_bf16"))
    kwargs: dict[str, Any] = {
        "device": "cpu", "rng": np.random.default_rng(0), "cfg": GumbelConfig(), "tree": tree,
    }
    with pytest.raises(CCoreIncompleteError, match="continue_gumbel_sims_legal_bf16"):
        gumbel_c.run_gumbel_root_many_c(
            None, [chess.Board()], evaluator=cast(Any, _LegalBf16Evaluator()), **kwargs,
        )
    # An evaluator without the compact transport legitimately runs fp32: the
    # tree check does not fire (it fails later, at evaluation, for this stub).
    with pytest.raises(RuntimeError, match="plain evaluator") as info:
        gumbel_c.run_gumbel_root_many_c(
            None, [chess.Board()], evaluator=cast(Any, _PlainEvaluator()), **kwargs,
        )
    assert not isinstance(info.value, CCoreIncompleteError)


def test_gumbel_c_refuses_missing_halving_rev(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.mcts import gumbel_c

    monkeypatch.setattr(gumbel_c, "_halving_rev_reported", False)
    monkeypatch.delattr(mcts_tree, "GSS_HALVING_REV")
    # Twice: a caught refusal must not latch the report and silence the next call.
    for _ in range(2):
        with pytest.raises(CCoreIncompleteError, match="GSS_HALVING_REV"):
            gumbel_c._report_halving_rev()
    assert gumbel_c._halving_rev_reported is False


def test_gumbel_c_refuses_missing_tt_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.mcts import gumbel_c

    monkeypatch.setattr(gumbel_c, "_tt_health_next_check", 0.0)
    monkeypatch.delattr(mcts_tree, "tt_stats")
    # Twice: a caught refusal must not arm the 60 s throttle and silence the next call.
    for _ in range(2):
        with pytest.raises(CCoreIncompleteError, match="tt_stats"):
            gumbel_c._report_guard_health()
    assert gumbel_c._tt_health_next_check == 0.0


def test_tree_reset_refuses_without_reset_compact() -> None:
    from chess_anti_engine.selfplay.manager import _reset_search_tree

    calls: list[str] = []

    class _OldTree:
        def reset(self) -> None:
            calls.append("reset")

    with pytest.raises(CCoreIncompleteError, match="reset_compact"):
        _reset_search_tree(_OldTree())
    assert calls == []
    with python_reference_paths("reference reset"):
        _reset_search_tree(_OldTree())
    assert calls == ["reset"]


def test_puct_c_refuses_without_cboard(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.mcts import puct_c
    from chess_anti_engine.mcts.puct import MCTSConfig
    from tests.selfplay_helpers import UniformPolicyValueModel

    monkeypatch.setattr(puct_c, "_HAS_CBOARD", False)
    with pytest.raises(CCoreIncompleteError, match="CBoard"):
        puct_c.run_mcts_many_c(
            UniformPolicyValueModel().eval(), [chess.Board()], device="cpu",
            rng=np.random.default_rng(0), cfg=MCTSConfig(simulations=2),
        )


def test_refusals_carry_the_original_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.selfplay import match as match_mod
    from chess_anti_engine.selfplay import network_turn as nt
    from chess_anti_engine.selfplay import play_batch

    monkeypatch.setattr(nt, "_HAS_GUMBEL_C", False)
    monkeypatch.setattr(nt, "_GUMBEL_C_IMPORT_ERROR", "ImportError: cannot import name 'X' (boom)")
    kwargs = _play_batch_kwargs()
    model = kwargs.pop("model")
    with pytest.raises(CCoreIncompleteError, match="boom"):
        play_batch(model, **kwargs)
    monkeypatch.setattr(match_mod, "_GUMBEL_C_IMPORT_ERROR", "ImportError: arena-boom")
    with pytest.raises(CCoreIncompleteError, match="arena-boom"):
        _arena(monkeypatch, "_HAS_GUMBEL_C", "gumbel")


# Run in a fresh interpreter: the consumers capture the ImportError text at
# THEIR import, so the only faithful way to pin it is to make the runner module
# genuinely unimportable before the consumer is first imported.
_IMPORT_TEXT_PROBE = r"""
import sys
import numpy as np
import torch

torch.set_num_threads(2)
blocked, consumer, branch = sys.argv[1], sys.argv[2], sys.argv[3]
sys.modules["chess_anti_engine.mcts." + blocked] = None

from chess_anti_engine.ext_check import CCoreIncompleteError


def run():
    if consumer == "arena":
        from chess_anti_engine.moves import POLICY_SIZE
        from chess_anti_engine.selfplay import match

        class _Dummy(torch.nn.Module):
            def forward(self, x):
                bs = int(x.shape[0])
                return {"policy_own": torch.zeros((bs, POLICY_SIZE)), "wdl": torch.zeros((bs, 3))}

        m = _Dummy().eval()
        match.play_match_batch(
            m, m, device="cpu", rng=np.random.default_rng(0), games=2, max_plies=2,
            a_plays_white=[True, False], mcts_type=branch, mcts_simulations=2,
        )
        return
    from chess_anti_engine.selfplay import manager, play_batch
    from chess_anti_engine.selfplay.config import (
        DiffFocusConfig, GameConfig, SearchConfig, TemperatureConfig,
    )
    from tests.selfplay_helpers import FakeStockfish, UniformPolicyValueModel

    if consumer == "network_turn":
        # Past play_batch's up-front check, so the dispatch site itself refuses.
        manager._require_compiled_search = lambda: None
    sims = 2 if branch == "gumbel" else 80  # PUCT runs only above max(64, fast)
    play_batch(
        UniformPolicyValueModel().eval(), device="cpu", rng=np.random.default_rng(0),
        stockfish=FakeStockfish([0.0, 1.0, 0.0]), games=2,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=sims, playout_cap_fraction=1.0,
                            fast_simulations=2, mcts_type=branch),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=4, selfplay_fraction=1.0),
    )


try:
    run()
except CCoreIncompleteError as exc:
    print("REFUSED::" + str(exc).replace("\n", " "))
else:
    print("NO_REFUSAL")
"""


@pytest.mark.parametrize(("blocked", "consumer", "branch", "site"), [
    ("gumbel_c", "play_batch", "gumbel", "play_batch"),
    ("puct_c", "play_batch", "puct", "play_batch"),
    ("gumbel_c", "network_turn", "gumbel", "selfplay network turn"),
    ("puct_c", "network_turn", "puct", "selfplay network turn"),
    ("gumbel_c", "arena", "gumbel", "arena move selection"),
    ("puct_c", "arena", "puct", "arena move selection"),
])
def test_refusal_carries_the_real_import_error_text(
    blocked: str, consumer: str, branch: str, site: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               PYTHONPATH=os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_TEXT_PROBE, blocked, consumer, branch],
        cwd=root, env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    refused = [ln for ln in proc.stdout.splitlines() if ln.startswith("REFUSED::")]
    assert refused, f"no refusal (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr[-3000:]}"
    message = refused[0]
    assert site in message, message
    # The interpreter's own text for an import blocked via sys.modules[name] = None.
    assert f"import of chess_anti_engine.mcts.{blocked} halted; None in sys.modules" in message, message


# ── worker / training / train.sh gate ──────────────────────────────────────


def _worker_main_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str]) -> list[Any]:
    """Drive worker.main() up to (not into) the session; returns the sessions built."""
    from chess_anti_engine import worker

    monkeypatch.setattr(worker, "_configure_worker_torch_env", lambda: None)
    monkeypatch.setattr(worker, "_setup_worker_logging", lambda args: logging.getLogger("t"))
    built: list[Any] = []

    class _FakeSession:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            built.append(self)

        def run(self) -> None:
            pass

    monkeypatch.setattr(worker, "WorkerSession", _FakeSession)
    monkeypatch.setattr(sys, "argv", [
        "worker", "--work-dir", str(tmp_path), "--username", "u", "--password", "p",
        "--stockfish-path", "/inert/sf", "--no-save-config", *extra,
    ])
    return built


_WORKER_ENTRY_PROBE = r"""
import runpy
import sys

import chess_anti_engine.mcts._mcts_tree as tree

del tree.classify_games  # a partial build: one required symbol missing
sys.argv = ["chess_anti_engine.worker", *sys.argv[1:]]
runpy.run_module("chess_anti_engine.worker", run_name="__main__", alter_sys=True)
"""


def test_worker_entry_point_refusal_lands_once_in_worker_out_and_worker_log(tmp_path: Path) -> None:
    """The real entry point, launched the way the supervisor launches it.

    stdout+stderr go to one file (the supervisor's ``worker.out``, stderr=STDOUT)
    and ``--log-file`` names ``worker.log``: the refusal must be in each exactly once.
    """
    root = Path(__file__).resolve().parents[1]
    worker_out = tmp_path / "worker.out"
    worker_log = tmp_path / "worker.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               PYTHONPATH=os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]))
    env.pop("CAE_SELF_UPDATED", None)
    with worker_out.open("wb") as out:
        proc = subprocess.run(
            [sys.executable, "-c", _WORKER_ENTRY_PROBE,
             "--work-dir", str(tmp_path / "work"), "--username", "u", "--password", "p",
             "--stockfish-path", "/inert/sf", "--no-save-config",
             "--log-file", str(worker_log), "--log-level", "info"],
            cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=300, check=False,
        )
    out_text = worker_out.read_text(encoding="utf-8", errors="replace")
    log_text = worker_log.read_text(encoding="utf-8", errors="replace") if worker_log.exists() else ""
    assert proc.returncode == ext_check.C_CORE_REFUSAL_EXIT_CODE, out_text[-3000:]
    marker = "Self-update was not attempted"
    assert out_text.count(marker) == 1, out_text[-3000:]
    assert log_text.count(marker) == 1, log_text[-3000:]
    assert "classify_games" in out_text
    assert "classify_games" in log_text


def test_supervisor_refusal_names_the_worker_out_and_log_paths(tmp_path: Path) -> None:
    from chess_anti_engine.tune import distributed_runtime

    class _Exited:
        returncode = ext_check.C_CORE_REFUSAL_EXIT_CODE

        def poll(self) -> int:
            return self.returncode

    proc = _Exited()
    setattr(proc, "_cae_worker_log_paths", (tmp_path / "worker.out", tmp_path / "worker.log"))
    with pytest.raises(CCoreIncompleteError) as info:
        distributed_runtime._raise_if_c_core_refusal(
            cast(Any, proc), idx=0, trial_id="t",
        )
    assert str(tmp_path / "worker.out") in str(info.value)
    assert str(tmp_path / "worker.log") in str(info.value)


def test_worker_startup_refuses_without_self_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    from chess_anti_engine import worker

    monkeypatch.delenv("CAE_SELF_UPDATED", raising=False)
    monkeypatch.delattr(mcts_tree, "classify_games")
    built = _worker_main_env(monkeypatch, tmp_path, [])
    with pytest.raises(SystemExit) as info:
        worker.main()
    assert info.value.code == ext_check.C_CORE_REFUSAL_EXIT_CODE
    err = capsys.readouterr().err
    assert "classify_games" in err
    # Printed once (stderr reaches worker.out); a log record as well doubled it there.
    assert err.count("Self-update was not attempted") == 1
    assert not [r for r in caplog.records if "Self-update was not attempted" in r.getMessage()]
    assert built == [], "a session was built on an incomplete C core"


def test_worker_with_self_update_defers_the_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from chess_anti_engine import worker

    monkeypatch.delenv("CAE_SELF_UPDATED", raising=False)
    monkeypatch.delattr(mcts_tree, "classify_games")
    built = _worker_main_env(monkeypatch, tmp_path, ["--self-update"])
    worker.main()
    assert len(built) == 1
    assert built[0]._c_core_check_deferred is True


def test_worker_complete_build_with_self_update_does_not_defer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from chess_anti_engine import worker

    monkeypatch.delenv("CAE_SELF_UPDATED", raising=False)
    built = _worker_main_env(monkeypatch, tmp_path, ["--self-update"])
    worker.main()
    assert built[0]._c_core_check_deferred is False


def test_worker_after_a_self_update_refuses_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from chess_anti_engine import worker

    monkeypatch.setenv("CAE_SELF_UPDATED", "1")
    monkeypatch.delattr(mcts_tree, "classify_games")
    built = _worker_main_env(monkeypatch, tmp_path, ["--self-update"])
    with pytest.raises(SystemExit) as info:
        worker.main()
    assert info.value.code == ext_check.C_CORE_REFUSAL_EXIT_CODE
    assert "already applied" in capsys.readouterr().err
    assert built == []


class _ManifestResponse:
    status_code = 200

    def json(self) -> dict[str, Any]:
        return {"protocol_version": None}


def _poll_session(calls: list[str]) -> Any:
    """A WorkerSession with just enough state for _poll_manifest."""
    from types import SimpleNamespace

    from chess_anti_engine import worker

    sess = cast(Any, object.__new__(worker.WorkerSession))
    sess.inference_client = object()
    sess.fixed_trial_id = "t"
    sess.cfg = {}
    sess.log = logging.getLogger("t")
    sess.args = SimpleNamespace(poll_seconds=0.0)
    sess.trial_api_prefix = "/api"
    sess.worker_id = "w"
    sess.lease_id = None
    sess.manifest_state = "active"
    sess.manifest_state_elapsed_s = None
    sess._manifest_poll_failures = 0
    sess._server_url_for = lambda path: path
    sess._upload_pending_shards = lambda **_kw: None
    sess._upload_pending_arena_results = lambda: None
    sess._requests = SimpleNamespace(get=lambda *a, **kw: _ManifestResponse())
    sess._check_manifest_compat = lambda m: SimpleNamespace(
        protocol_mismatch=False, version_too_old=False, req_proto=None, min_worker_version=None,
    )
    sess._maybe_self_update_from_manifest = lambda m, c: calls.append("self_update")
    sess._check_pause_selfplay = lambda m: (calls.append("pause_check"), False)[1]
    return sess


def test_deferred_refusal_fires_after_the_self_update_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    sess = _poll_session(calls)
    sess._c_core_check_deferred = True
    monkeypatch.delattr(mcts_tree, "classify_games")
    with pytest.raises(SystemExit) as info:
        sess._poll_manifest()
    assert info.value.code == ext_check.C_CORE_REFUSAL_EXIT_CODE
    assert calls == ["self_update"], "refused before trying the update, or ran past it"
    assert "no applicable worker wheel" in capsys.readouterr().err


def test_deferred_check_passes_on_a_complete_build() -> None:
    calls: list[str] = []
    sess = _poll_session(calls)
    sess._c_core_check_deferred = True
    assert sess._poll_manifest() == {"protocol_version": None}
    assert calls == ["self_update", "pause_check"]
    assert sess._c_core_check_deferred is False


def test_run_main_refuses_before_booting_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from chess_anti_engine import run

    monkeypatch.delattr(mcts_tree, "batch_encode_146_bf16")
    harness = ModuleType("chess_anti_engine.tune.harness")

    def forbidden(**_kw: Any) -> None:
        raise AssertionError("run_tune reached on an incomplete C core")

    monkeypatch.setattr(harness, "run_tune", forbidden, raising=False)
    monkeypatch.setitem(sys.modules, "chess_anti_engine.tune.harness", harness)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "chess_anti_engine.run", "--config", str(cfg), "--device", "cpu",
        "--stockfish-path", "/inert/never-start-stockfish", "--work-dir", str(tmp_path / "w"),
    ])
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(run.torch.backends.cuda.matmul, "allow_tf32", run.torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(run.torch.backends.cudnn, "allow_tf32", run.torch.backends.cudnn.allow_tf32)
    with pytest.raises(SystemExit) as info:
        run.main()
    assert info.value.code == ext_check.C_CORE_REFUSAL_EXIT_CODE
    assert "batch_encode_146_bf16" in capsys.readouterr().err


def _load_gate_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "check_c_extensions_fresh.py"
    spec = importlib.util.spec_from_file_location("_gate_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_train_gate_symbol_check_sees_a_partial_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The child interpreter imports the real extensions; a sitecustomize removes one symbol."""
    gate = _load_gate_script()
    assert gate.symbol_problems(REPO_ROOT) == []
    (tmp_path / "sitecustomize.py").write_text(
        "import chess_anti_engine.mcts._mcts_tree as _m\ndel _m.classify_games\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    problems = gate.symbol_problems(REPO_ROOT)
    assert any("classify_games" in p for p in problems), problems


def test_train_gate_inspects_the_root_it_is_given(tmp_path: Path) -> None:
    """--root decides which checkout is imported, not the script's own location."""
    gate = _load_gate_script()
    pkg = tmp_path / "chess_anti_engine"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "ext_check.py").write_text(
        "def extension_problems():\n    return ['stub root: _mcts_tree lacks X']\n"
    )
    assert gate.symbol_problems(tmp_path) == ["stub root: _mcts_tree lacks X"]


def test_train_gate_main_runs_symbol_check_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _load_gate_script()
    calls: list[Path] = []
    monkeypatch.setattr(gate, "check_extensions", lambda root, **_kw: [])

    def fake_symbols(root: Path) -> list[str]:
        calls.append(root)
        return ["chess_anti_engine.mcts._mcts_tree lacks classify_games (built from older source)"]

    monkeypatch.setattr(gate, "symbol_problems", fake_symbols)
    monkeypatch.setattr(sys, "argv", ["gate", "--root", str(REPO_ROOT), "--quiet"])
    assert gate.main() == 0
    assert calls == []
    monkeypatch.setattr(sys, "argv", ["gate", "--root", str(REPO_ROOT), "--quiet", "--check-symbols"])
    assert gate.main() == 1
    assert "classify_games" in capsys.readouterr().out


def test_train_sh_passes_check_symbols() -> None:
    text = (REPO_ROOT / "scripts" / "train.sh").read_text()
    lines = text.splitlines()
    idx = next(i for i, ln in enumerate(lines) if "check_c_extensions_fresh.py" in ln)
    assert "--check-symbols" in " ".join(lines[idx:idx + 2])


# ── encode side ────────────────────────────────────────────────────────────


def test_encode_position_refuses_python_features(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.encoding import encode as enc

    board = chess.Board()
    want = enc.encode_position(board, input_extra_features="v2_threats")
    monkeypatch.setattr(enc, "_HAS_C_EXT", False)
    with pytest.raises(CCoreIncompleteError, match="compute_extra_features"):
        enc.encode_position(board, input_extra_features="v2_threats")
    with python_reference_paths("oracle parity"):
        got = enc.encode_position(board, input_extra_features="v2_threats")
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_encode_position_refuses_python_piece_planes(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.encoding import encode as enc

    board = chess.Board()
    want = enc.encode_position(board, input_extra_features="v2_threats")
    monkeypatch.setattr(enc, "_HAS_LC0_C_EXT", False)
    with pytest.raises(CCoreIncompleteError, match="encode_piece_planes"):
        enc.encode_position(board, input_extra_features="v2_threats")
    with python_reference_paths("oracle parity"):
        got = enc.encode_position(board, input_extra_features="v2_threats")
    np.testing.assert_array_equal(got, want)


def test_recompute_extra_planes_refuses_python_features(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.encoding import plane_decode
    from chess_anti_engine.encoding.encode import encode_position

    rows = np.stack([encode_position(chess.Board(), input_extra_features="v2_threats")])
    want = plane_decode.recompute_extra_planes(rows, version="v2_threats")
    monkeypatch.setattr(plane_decode, "_HAS_C_EXT", False)
    with pytest.raises(CCoreIncompleteError, match="compute_extra_features"):
        plane_decode.recompute_extra_planes(rows, version="v2_threats")
    with python_reference_paths("oracle parity"):
        got = plane_decode.recompute_extra_planes(rows, version="v2_threats")
    np.testing.assert_allclose(got, want, atol=1e-6)


class _Without:
    """A CBoard proxy that lacks one attribute (a CBoard from an older build)."""

    def __init__(self, cb: Any, missing: str) -> None:
        self._cb = cb
        self._missing = missing

    def __getattr__(self, name: str) -> Any:
        if name == self._missing:
            raise AttributeError(name)
        return getattr(self._cb, name)


def test_encode_cboard_refuses_missing_encode_full(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.encoding import cboard_encode as ce

    cb = CBoard.from_board(chess.Board())
    want = ce.encode_cboard(cb, input_extra_features="v2_threats")
    proxy = cast(Any, _Without(cb, "encode_full"))
    with pytest.raises(CCoreIncompleteError, match=r"CBoard\.encode_full"):
        ce.encode_cboard(proxy, input_extra_features="v2_threats")
    with python_reference_paths("oracle parity"):
        got = ce.encode_cboard(proxy, input_extra_features="v2_threats")
        np.testing.assert_array_equal(got, want)
        # Never all-zero feature planes, opt-in or not.
        monkeypatch.setattr(ce, "_HAS_FEATURES_C", False)
        with pytest.raises(CCoreIncompleteError, match="all-zero"):
            ce.encode_cboard(proxy, input_extra_features="v2_threats")


def test_encode_cboard_batch_refuses_missing_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.encoding import cboard_encode as ce

    monkeypatch.delattr(lc0_ext, "encode_full_batch")
    with pytest.raises(CCoreIncompleteError, match="encode_full_batch"):
        ce.encode_cboard_batch([CBoard.from_board(chess.Board())])


@pytest.mark.parametrize("fn_name", ["legal_move_mask", "legal_move_indices"])
def test_legal_moves_refuse_python_generation(monkeypatch: pytest.MonkeyPatch, fn_name: str) -> None:
    from chess_anti_engine.moves import encode as menc

    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    fn = getattr(menc, fn_name)
    want = fn(board)
    monkeypatch.setattr(menc, "_HAS_LC0_C_EXT", False)
    with pytest.raises(CCoreIncompleteError, match="legal_move_policy_indices"):
        fn(board)
    with python_reference_paths("oracle parity"):
        got = fn(board)
    if fn_name == "legal_move_mask":
        np.testing.assert_array_equal(got, want)
    else:
        assert sorted(got.tolist()) == sorted(want.tolist())


def test_root_terminal_scan_refuses_missing_method() -> None:
    from chess_anti_engine.mcts import root_tactics

    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")  # Ra8#
    cb = CBoard.from_board(board)
    legal = cb.legal_move_indices()
    want = root_tactics.immediate_terminal_cboard_policy_or_draws(cb, legal)
    proxy = cast(Any, _Without(cb, "root_terminal_actions"))
    with pytest.raises(CCoreIncompleteError, match="root_terminal_actions"):
        root_tactics.immediate_terminal_cboard_policy_or_draws(proxy, legal)
    with python_reference_paths("oracle parity"):
        got = root_tactics.immediate_terminal_cboard_policy_or_draws(proxy, legal)
    assert want[0] is not None
    assert got[0] is not None
    assert got[0][1] == want[0][1]
    assert got[1] == want[1]


# ── rep_fix ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("module", ["chess_anti_engine.encoding._lc0_ext", "chess_anti_engine.mcts._mcts_tree"])
def test_rep_fix_refuses_a_missing_setter_before_setting_any(
    monkeypatch: pytest.MonkeyPatch, module: str,
) -> None:
    before = rep_fix.current()
    target = not bool(before)
    calls: list[tuple[str, bool]] = []
    for name in ("chess_anti_engine.encoding._lc0_ext", "chess_anti_engine.mcts._mcts_tree"):
        mod = importlib.import_module(name)
        if name == module:
            monkeypatch.delattr(mod, "set_history_rep_fix")
        else:
            monkeypatch.setattr(mod, "set_history_rep_fix",
                                lambda v, _n=name: calls.append((_n, bool(v))))
    with pytest.raises(rep_fix.RepFixUnavailableError, match=f"{module} lacks set_history_rep_fix"):
        rep_fix.apply(target, boards_discarded=True)
    assert calls == [], "the other encoder was flipped before the refusal"
    assert rep_fix.current() is before


def test_rep_fix_refuses_an_unimportable_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    before = rep_fix.current()
    monkeypatch.setitem(sys.modules, "chess_anti_engine.mcts._mcts_tree", None)
    with pytest.raises(rep_fix.RepFixUnavailableError, match="not importable"):
        rep_fix.apply(not bool(before), boards_discarded=True)
    assert rep_fix.current() is before


# ── behaviour-preserving swaps (deliverable 3) ─────────────────────────────


def test_infer_input_planes_matches_the_encoder_for_every_mode() -> None:
    from chess_anti_engine.encoding import encode_position, input_plane_count
    from chess_anti_engine.encoding.features import EXTRA_FEATURE_VERSIONS
    from chess_anti_engine.model import infer_input_planes

    for version in (None, "", *EXTRA_FEATURE_VERSIONS, "V2_THREATS"):
        encoded = encode_position(chess.Board(), add_features=True, input_extra_features=version)
        assert infer_input_planes(version) == input_plane_count(version) == encoded.shape[0]
    for bogus in ("bogus", "off"):
        with pytest.raises(ValueError, match="unknown input_extra_features"):
            encode_position(chess.Board(), add_features=True, input_extra_features=bogus)
        with pytest.raises(ValueError, match="unknown input_extra_features"):
            infer_input_planes(bogus)


def test_warmup_probe_encoder_is_bit_identical_to_python_e3() -> None:
    from chess_anti_engine.encoding import encode_positions_batch
    from chess_anti_engine.encoding.cboard_encode import encode_cboard_batch
    from chess_anti_engine.encoding.features import EXTRA_FEATURE_VERSIONS

    for flag in (False, True):
        rep_fix.apply(flag, boards_discarded=True)
        for hist in ("legacy", "lc0_root", "lc0_root_legacy_meta"):
            for version in EXTRA_FEATURE_VERSIONS:
                board = chess.Board()
                old = encode_positions_batch(
                    [board], input_history_encoding=hist, input_extra_features=version,
                )
                new = encode_cboard_batch(
                    [CBoard.from_board(board)], input_history_encoding=hist,
                    input_extra_features=version,
                )
                assert old.dtype == new.dtype
                assert old.shape == new.shape
                assert old.tobytes() == new.tobytes(), (flag, hist, version)
