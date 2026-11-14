"""One definition of "the compiled C core is complete", for every entry point.

Production selfplay, arena, training and the UCI engine run on the compiled
extensions (``_lc0_ext``, ``_features_ext``, ``_mcts_tree``). Most Python call
sites used to probe them with ``try: import`` / ``hasattr`` and, when a symbol
was missing (a stale or partial build), quietly switch to a pure-Python
implementation or a different transport: a slower search, a per-ply recorder
that drops seed history, fp32 instead of bf16 leaves, or all-zero feature
planes. This module owns the list of what must be present and two ways to act
on it:

* :func:`extension_problems` / :func:`require_c_core` — the startup check
  (import + symbols + ``_mcts_tree`` ABI). The UCI startup gate
  (``uci/selfcheck.py``), the worker, ``run.py`` (train/tune) and the
  ``train.sh`` gate (``scripts/check_c_extensions_fresh.py --check-symbols``)
  all call it.
* :func:`refuse_fallback` — the point-of-use refusal. Each former silent
  fallback site calls it instead of switching paths. The Python reference
  implementations stay importable (they are the parity oracles); running
  them in place of a missing C symbol requires the explicit opt-in
  :func:`python_reference_paths`, a context manager that no production module
  enters. There is deliberately no environment-variable switch.

``tests/test_c_core_required.py`` scans the package for guarded imports and
``hasattr``/``getattr`` probes of extension symbols and fails when one is
neither listed here nor in :data:`GUARDED_OPTIONAL_ALLOWLIST`.

Only the standard library is imported at module level (the UCI self-check
reports a broken torch or extension as a FAIL line rather than dying on import).
"""
from __future__ import annotations

import contextlib
import importlib
import threading
from collections.abc import Callable, Generator, Sequence

LC0_EXT = "chess_anti_engine.encoding._lc0_ext"
FEATURES_EXT = "chess_anti_engine.encoding._features_ext"
MCTS_TREE_EXT = "chess_anti_engine.mcts._mcts_tree"

# The extensions selfplay, training, arena and the UCI engine need. setup.py
# also builds `_perft_ext` (diagnostic) and `_nnue_ext` (NNUE research search);
# no production entry point imports either, so neither is required.
REQUIRED_EXTENSIONS: tuple[str, ...] = (LC0_EXT, FEATURES_EXT, MCTS_TREE_EXT)

# Symbols the Python side calls on each extension. Every symbol whose absence
# used to select a different code path is listed (the coverage test enforces
# it). `_mcts_tree` also carries a numeric ABI (`ABI_VERSION` vs
# `gumbel_c._REQUIRED_MCTS_ABI`); the encoders carry no version number, so the
# symbol set is their ABI contract.
REQUIRED_SYMBOLS: dict[str, tuple[str, ...]] = {
    LC0_EXT: (
        "CBoard", "encode_full_batch",
        # moves/encode.py legal_move_mask / legal_move_indices
        "legal_move_policy_indices",
        # encoding/lc0.py -> encoding/encode.py encode_lc0_full_c
        "encode_piece_planes",
        # encoding/rep_fix.py
        "set_history_rep_fix",
        # selfplay/finalize.py (imported unguarded)
        "prepare_sf_multipv",
    ),
    FEATURES_EXT: (
        "compute_extra_features", "compute_relation_matrices",
    ),
    MCTS_TREE_EXT: (
        "MCTSTree", "ABI_VERSION", "batch_encode_146",
        # selfplay per-ply path + gumbel_c root/leaf encoders
        "batch_encode_146_lc0_root", "batch_encode_146_lc0_root_legacy_meta",
        "batch_encode_146_bf16", "batch_encode_146_lc0_root_bf16",
        "batch_encode_146_lc0_root_legacy_meta_bf16",
        "batch_process_ply",
        # selfplay slot scheduling / temperature
        "classify_games", "temperature_resample",
        # encoding/rep_fix.py
        "set_history_rep_fix",
        # gumbel_c reports the root sequential-halving rule it loaded
        "GSS_HALVING_REV",
        # imported or read unguarded by network_turn / gumbel_c
        "batch_compute_relations", "SWDL_DRAW_NET_RAW", "SWDL_DRAW_PARAMETRIC_Q",
        "GSS_MAX_CANDS", "tt_stats",
    ),
}
REQUIRED_CBOARD_METHODS: tuple[str, ...] = (
    "encode_full", "legal_move_indices",
    # mcts/root_tactics.py immediate_terminal_cboard_policy_or_draws
    "root_terminal_actions",
)
# Methods gumbel_c / selfplay probe on the tree object. Without the first two
# the leaf transport falls back from compact-legal bf16 to dense fp32; without
# `reset_compact` the tree reset drops its compaction; `batch_descend_puct_tb`
# is the C TB descent `mcts/puct_tb.py` requires.
REQUIRED_MCTSTREE_METHODS: tuple[str, ...] = (
    "get_pending_legal_indices", "continue_gumbel_sims_legal_bf16",
    "reset_compact", "batch_descend_puct_tb",
)

# Names that the coverage scan finds probed with hasattr/getattr and that also
# exist on an extension object, but are NOT optional C symbols:
GUARDED_OPTIONAL_ALLOWLIST: dict[str, str] = {
    # `cfg.compute_relations` is an MCTSConfig/GumbelConfig field read with a
    # default (mcts/puct.py, mcts/puct_c.py); it only shares its name with the
    # CBoard method.
    "compute_relations": "config field read on a config object, not an extension probe",
    # `module.__file__` is read for provenance (source/bt4_npz_unit_v1.py).
    "__file__": "module provenance, present on every module",
}

# The command that clears a refusal must force a recompile: deleting the .so
# and rebuilding without --force can copy an old object back out of build/.
FORCE_REBUILD_COMMAND = "pip install --no-deps -e ."
# Equivalent where setuptools is installed in the venv (it is not by default on
# Python 3.12+); the editable reinstall gets setuptools from build isolation.
SETUP_FORCE_COMMAND = "python setup.py build_ext --inplace --force"
PRODUCTION_BUILD_COMMAND = "python scripts/build_production_extensions.py"
REBUILD_HINT = (
    "in a source checkout (editable install), recompile the C extensions with "
    f"`{FORCE_REBUILD_COMMAND}` (or `uv pip install --no-deps -e .`), which "
    "compiles every extension from scratch in a temporary build directory; "
    f"`{SETUP_FORCE_COMMAND}` does the same if setuptools is installed; "
    f"production host: `{PRODUCTION_BUILD_COMMAND}`. "
    "A regular install: reinstall it from the checkout or the wheel "
    "(`pip install --force-reinstall --no-deps <checkout or .whl>`)"
)


# Process exit status of a startup refusal (worker, run.py): sysexits'
# EX_CONFIG. Distinct from a crash (1) and from the UCI gate's 2, so a
# supervisor can tell "this build can never run" from a transient failure: tune/distributed_runtime.py treats it as fatal for the
# trial instead of relaunching the worker into the same refusal.
C_CORE_REFUSAL_EXIT_CODE = 78


class CCoreIncompleteError(RuntimeError):
    """The compiled C core lacks something a production path needs."""


def required_abi() -> int:
    """``gumbel_c._REQUIRED_MCTS_ABI`` (gumbel_c owns the C-path ABI number).

    Imported by name at call time: gumbel_c itself imports this module, and
    the check must not pull torch in at import.
    """
    return int(importlib.import_module("chess_anti_engine.mcts.gumbel_c")._REQUIRED_MCTS_ABI)


def extension_problems(
    *, abi_required: Callable[[], int] | None = None,
) -> list[str]:
    """Import + symbol + ABI problems for every required extension."""
    abi_fn = required_abi if abi_required is None else abi_required
    problems: list[str] = []
    for name in REQUIRED_EXTENSIONS:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            problems.append(f"{name} is not importable ({exc})")
            continue
        missing = [sym for sym in REQUIRED_SYMBOLS[name] if not hasattr(module, sym)]
        if name == LC0_EXT and hasattr(module, "CBoard"):
            missing += [
                f"CBoard.{meth}" for meth in REQUIRED_CBOARD_METHODS
                if not hasattr(module.CBoard, meth)
            ]
        if name == MCTS_TREE_EXT and hasattr(module, "MCTSTree"):
            missing += [
                f"MCTSTree.{meth}" for meth in REQUIRED_MCTSTREE_METHODS
                if not hasattr(module.MCTSTree, meth)
            ]
        if missing:
            problems.append(f"{name} lacks {', '.join(missing)} (built from older source)")
        if name == MCTS_TREE_EXT:
            abi = int(getattr(module, "ABI_VERSION", 0))
            try:
                need = abi_fn()
            except ImportError:
                # gumbel_c needs the other extensions; their failure is reported above.
                continue
            if abi < need:
                problems.append(f"{name} ABI_VERSION={abi} < required {need}")
    return problems


def missing_symbols(module_name: str, names: Sequence[str]) -> list[str]:
    """``names`` absent from the (already importable) extension ``module_name``.

    Reads the live module object, so a test can simulate a partial build with
    ``monkeypatch.delattr``. An unimportable module reports every name.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return list(names)
    return [n for n in names if not hasattr(module, n)]


def incomplete_message(context: str, problems: Sequence[str]) -> str:
    lines = [
        f"{context}: the compiled C core is incomplete; refusing to run instead "
        "of falling back to a slower or different Python path:",
    ]
    lines += [f"  - {p}" for p in problems]
    lines.append(f"Rebuild: {REBUILD_HINT}.")
    return "\n".join(lines)


def require_c_core(
    context: str, *, abi_required: Callable[[], int] | None = None,
) -> None:
    """Raise :class:`CCoreIncompleteError` unless every required symbol + ABI is present."""
    problems = extension_problems(abi_required=abi_required)
    if problems:
        raise CCoreIncompleteError(incomplete_message(context, problems))


# ── explicit opt-in to the Python reference paths ────────────────────────

_reference_lock = threading.Lock()
_reference_depth = 0


@contextlib.contextmanager
def python_reference_paths(reason: str) -> Generator[None, None, None]:
    """Allow the Python reference implementations to stand in for missing C symbols.

    For tests and research tools that deliberately exercise the reference
    path (the parity oracles). ``reason`` must say why; it is not optional so
    the opt-in cannot be written by accident. Process-wide (worker threads see
    it too) and re-entrant. No production module enters it.
    """
    global _reference_depth
    if not str(reason).strip():
        raise ValueError("python_reference_paths needs a non-empty reason")
    with _reference_lock:
        _reference_depth += 1
    try:
        yield
    finally:
        with _reference_lock:
            _reference_depth -= 1


def python_reference_allowed() -> bool:
    return _reference_depth > 0


def refuse_fallback(site: str, missing: str | Sequence[str], fallback: str) -> None:
    """Raise unless inside :func:`python_reference_paths`.

    ``site`` names the call site, ``missing`` the C symbol(s) whose absence
    selected the fallback, ``fallback`` what would have run instead.
    """
    if python_reference_allowed():
        return
    names = [missing] if isinstance(missing, str) else list(missing)
    raise CCoreIncompleteError(
        f"{site}: C fast path unavailable (missing: {', '.join(names)}); "
        f"refusing to fall back to {fallback}. A stale or partial extension build "
        f"is the usual cause. Rebuild: {REBUILD_HINT}. (Tests and research tools "
        "that want the Python reference path must enter "
        "chess_anti_engine.ext_check.python_reference_paths(reason).)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m chess_anti_engine.ext_check``: print problems, exit 1 if any."""
    del argv
    problems = extension_problems()
    if problems:
        print(incomplete_message("C core check", problems))
        return 1
    print(f"C core complete: {len(REQUIRED_EXTENSIONS)} extensions, symbols and ABI present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
