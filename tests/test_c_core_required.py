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


_EXT_MODULE_SUFFIXES = ("._lc0_ext", "._features_ext", "._mcts_tree")


def _guarded_probes() -> list[tuple[str, int, str]]:
    """Every (file, line, name) where package code treats a C symbol as optional.

    Two shapes: ``hasattr(x, "name")`` / ``getattr(x, "name", ...)`` whose name
    exists on an extension module, ``CBoard`` or ``MCTSTree``; and a
    ``from <extension> import name`` inside a ``try`` that handles an
    ImportError (or Exception).
    """
    on_ext = _extension_object_names()
    found: list[tuple[str, int, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("hasattr", "getattr") and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and node.args[1].value in on_ext
            ):
                found.append((rel, node.lineno, node.args[1].value))
            if isinstance(node, ast.Try):
                handled = set()
                for handler in node.handlers:
                    types = (
                        handler.type.elts if isinstance(handler.type, ast.Tuple)
                        else [handler.type]
                    )
                    handled |= {t.id for t in types if isinstance(t, ast.Name)}
                if not handled & {"ImportError", "ModuleNotFoundError", "Exception"}:
                    continue
                for stmt in node.body:
                    for sub in ast.walk(stmt):
                        if (
                            isinstance(sub, ast.ImportFrom) and sub.module
                            and sub.module.endswith(_EXT_MODULE_SUFFIXES)
                        ):
                            found += [(rel, sub.lineno, alias.name) for alias in sub.names]
    return found


def test_every_guarded_c_symbol_is_required_or_allow_listed() -> None:
    """Adding a new ``try: import``/``hasattr`` fallback on a C symbol fails here.

    List the symbol in ``ext_check.REQUIRED_SYMBOLS`` (or the CBoard/MCTSTree
    method tuples) so every entry point refuses a build without it, or add it
    to ``GUARDED_OPTIONAL_ALLOWLIST`` with the reason it is not a C fallback.
    """
    required = _all_required_names()
    allowed = set(ext_check.GUARDED_OPTIONAL_ALLOWLIST)
    probes = _guarded_probes()
    unlisted = sorted({(f, ln, n) for f, ln, n in probes if n not in required | allowed})
    assert not unlisted, f"guarded C symbols not in ext_check's required set: {unlisted}"
    # The scan is live: it sees the known sites (a scan that finds nothing
    # would pass the assertion above vacuously).
    names = {n for _, _, n in probes}
    for known in ("batch_process_ply", "classify_games", "encode_full", "encode_full_batch",
                  "root_terminal_actions", "set_history_rep_fix", "batch_encode_146_bf16",
                  "legal_move_policy_indices", "encode_piece_planes", "compute_extra_features",
                  "continue_gumbel_sims_legal_bf16", "reset_compact", "batch_descend_puct_tb"):
        assert known in names, known
    # Every allow-list entry is still used, so it cannot quietly cover a new site.
    assert allowed <= names, allowed - names


def test_scan_catches_a_new_guarded_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation of the scanner's input: a new hasattr fallback is reported."""
    fake_pkg = tmp_path / "chess_anti_engine"
    fake_pkg.mkdir()
    (fake_pkg / "newmod.py").write_text(
        "import chess_anti_engine.mcts._mcts_tree as t\n"
        "if hasattr(t, 'tt_stats'):\n    pass\n"
        "try:\n    from chess_anti_engine.encoding._lc0_ext import prepare_sf_multipv\n"
        "except ImportError:\n    prepare_sf_multipv = None\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE", fake_pkg)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    names = {n for _, _, n in _guarded_probes()}
    assert names == {"tt_stats", "prepare_sf_multipv"}


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


# ── worker / training / train.sh gate ──────────────────────────────────────


def test_worker_startup_refuses_before_reading_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from chess_anti_engine import worker

    monkeypatch.delattr(mcts_tree, "classify_games")
    monkeypatch.setattr(worker, "_configure_worker_torch_env", lambda: None)
    monkeypatch.setattr(worker, "_setup_worker_logging", lambda args: logging.getLogger("t"))

    def forbidden(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("worker read its config before the C core check")

    monkeypatch.setattr(worker, "load_worker_config", forbidden)
    monkeypatch.setattr(sys, "argv", ["worker", "--work-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="classify_games"):
        worker.main()


def test_run_main_refuses_before_booting_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
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
    with pytest.raises(SystemExit, match="batch_encode_146_bf16"):
        run.main()


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
