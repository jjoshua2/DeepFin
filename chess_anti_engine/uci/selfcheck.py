"""Install self-check for the UCI engine: ``deepfin --selfcheck``.

One line per component (``PASS`` / ``FAIL`` / ``SKIP`` plus a short reason),
then ``SELFCHECK PASS`` or ``SELFCHECK FAIL``; the exit status is 0 only when
nothing failed. ``--json`` prints the same results as one JSON object.

Every component is a behavioural assertion where one is possible, not an
import probe: the C encoder is compared against the pure-Python reference
encoder, a short C Gumbel search must leave its visits in the C tree, and the
engine itself is driven through a UCI exchange in a subprocess.

This module also owns the engine's startup gate (:func:`enforce_startup_gate`),
called from ``uci/__main__.py`` before the first protocol line. It refuses to
start on a missing, ABI-incompatible or stale required extension, or a feature
encoder that yields all-zero planes, instead of letting the engine play on a
silently degraded install. The gate is scoped to the UCI entry point:
selfplay and training share these extensions but never call it.

Only the standard library is imported at module level, so the self-check can
report a broken torch or a missing extension as a FAIL line rather than dying
on import.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import chess
    import numpy as np

Status = Literal["PASS", "FAIL", "SKIP"]

# Tested interpreters: 3.10 (production host venv) and 3.13 (CI, `uv.lock`).
# `requires-python` carries the floor; nothing above 3.13 has been run.
MIN_PYTHON: tuple[int, int, int] = (3, 10, 12)
MAX_PYTHON_EXCLUSIVE: tuple[int, int] = (3, 14)

# The extensions the UCI engine needs to play. setup.py also builds
# `_perft_ext` (diagnostic) and `_nnue_ext` (NNUE research search); the engine
# imports neither, so neither gates startup.
REQUIRED_EXTENSIONS: tuple[str, ...] = (
    "chess_anti_engine.encoding._lc0_ext",
    "chess_anti_engine.encoding._features_ext",
    "chess_anti_engine.mcts._mcts_tree",
)

# Symbols the Python side calls on each extension. `_mcts_tree` also carries a
# numeric ABI (`ABI_VERSION` vs `gumbel_c._REQUIRED_MCTS_ABI`); the encoders
# carry no version number, so the symbol set is their ABI contract, and the
# behavioural encoder check below is what proves the code behind it.
REQUIRED_SYMBOLS: dict[str, tuple[str, ...]] = {
    "chess_anti_engine.encoding._lc0_ext": ("CBoard", "encode_full_batch"),
    "chess_anti_engine.encoding._features_ext": (
        "compute_extra_features", "compute_relation_matrices",
    ),
    "chess_anti_engine.mcts._mcts_tree": ("MCTSTree", "ABI_VERSION", "batch_encode_146"),
}
REQUIRED_CBOARD_METHODS: tuple[str, ...] = ("encode_full", "legal_move_indices")

# Development escape hatch for the startup gate (never for an operator).
ALLOW_UNVERIFIED_ENV = "DEEPFIN_ALLOW_UNVERIFIED_EXTENSIONS"

# The command that clears a refusal must force a recompile: deleting the .so
# and rebuilding without --force can copy an old object back out of build/.
FORCE_REBUILD_COMMAND = "pip install --no-deps -e ."
# Equivalent where setuptools is installed in the venv (it is not by default on
# Python 3.12+); the editable reinstall gets setuptools from build isolation.
SETUP_FORCE_COMMAND = "python setup.py build_ext --inplace --force"
REBUILD_HINT = (
    "in a source checkout (editable install), recompile the C extensions with "
    f"`{FORCE_REBUILD_COMMAND}` (or `uv pip install --no-deps -e .`), which "
    "compiles every extension from scratch in a temporary build directory; "
    f"`{SETUP_FORCE_COMMAND}` does the same if setuptools is installed; "
    "production host: `python scripts/build_production_extensions.py`. "
    "A regular install: reinstall it from the checkout or the wheel "
    "(`pip install --force-reinstall --no-deps <checkout or .whl>`)"
)

# Positions for the encoder parity and move-generation checks: the start
# position, Kiwipete (castling, pins, promotions nearby), an en-passant
# square, and a black-to-move pawn ending.
ENCODER_FENS: tuple[str, ...] = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/2k5/3p4/p2P1p2/P2P1P2/8/3K4/8 b - - 4 40",
)
ENCODER_HISTORY_MOVES: tuple[str, ...] = ("e2e4", "e7e5", "g1f3", "b8c6")
ENCODER_ATOL = 1e-6
FEATURE_VERSIONS: tuple[str, ...] = ("v1", "v2_threats")

# (fen, depth, nodes): published perft values.
PERFT_CASES: tuple[tuple[str, int, int], ...] = (
    (ENCODER_FENS[0], 3, 8902),
    (ENCODER_FENS[1], 2, 2039),
)

# (fen, expected raw Syzygy WDL from the side to move). 3-man tables only, so
# any standard 3-5-man (or larger) set answers them.
SYZYGY_CASES: tuple[tuple[str, int], ...] = (
    ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 2),   # KQvK, white to move: win
    ("4k3/8/8/8/8/8/8/3QK3 b - - 0 1", -2),  # same, black to move: loss
    ("4k3/8/8/8/8/8/8/3BK3 w - - 0 1", 0),   # KBvK: draw
)

# (men, table, fen, expected WDL), probed only when the table is in the path.
SYZYGY_LARGER_CASES: tuple[tuple[int, str, str, int], ...] = (
    (5, "KQQRvK", "8/8/8/7k/8/8/8/QQRK4 w - - 0 1", 2),
    (6, "KQQRRvK", "8/8/8/7k/8/8/8/QQRRK3 w - - 0 1", 2),
)

SEARCH_SIMS = 32
UCI_GO_NODES = 64
UCI_TIMEOUT_FAST_S = 120.0
# A cold torch.compile (max-autotune by default) runs inside the readyok wait.
UCI_TIMEOUT_COMPILED_S = 900.0
# Engine output that means the session is not healthy, whatever move followed.
# Grepped from uci/__main__.py and uci/engine.py: strict startup (#1263), the
# extension gate, and every unsearched-move / degraded-path report (#1261/#1263).
UCI_FAILURE_MARKERS: tuple[str, ...] = (
    "engine startup failed:",
    "refusing to start",
    "search error:",
    "search not started",
    "search did not stop in time",
    "search incomplete reason=",
    "bestmove_fallback_used=",
    "engine unhealthy",
    "DEGRADED",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str


class CheckFailed(Exception):
    """A component's assertion failed; the message is the FAIL reason."""


class CheckSkipped(Exception):
    """A component does not apply to this invocation; the message says why."""


@dataclass(frozen=True)
class SelfcheckOptions:
    checkpoint: str | None = None
    syzygy_path: str | None = None
    devices: tuple[str, ...] = ()
    run_uci: bool = True
    # Engine flags given alongside --selfcheck, passed to the engine unchanged.
    engine_args: tuple[str, ...] = ()
    # --no-compile for the UCI exchange (fast, but not the default play path).
    fast: bool = False
    uci_timeout_s: float | None = None


# ── helpers ────────────────────────────────────────────────────────────────


def _one_line(text: str, limit: int = 400) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name, "PASS", _one_line(fn()))
    except CheckSkipped as exc:
        return CheckResult(name, "SKIP", _one_line(str(exc)))
    except CheckFailed as exc:
        return CheckResult(name, "FAIL", _one_line(str(exc)))
    except Exception as exc:  # any other error inside a check is a failure of that check
        return CheckResult(name, "FAIL", _one_line(f"{type(exc).__name__}: {exc}"))


def source_root() -> Path | None:
    """The source checkout this package runs from, or None for a wheel install.

    An editable / in-place install imports the package from the checkout, so
    the checkout's ``setup.py`` and the freshness tool sit beside it. A regular
    ``pip install .`` copies only the package (and the built ``.so``) into
    site-packages, where there are no sources to compare against.
    """
    root = Path(__file__).resolve().parents[2]
    if (root / "setup.py").is_file() and (root / "scripts" / "check_c_extensions_fresh.py").is_file():
        return root
    return None


def _load_freshness_tool(root: Path) -> ModuleType:
    """Load ``scripts/check_c_extensions_fresh.py`` — the gate train.sh uses."""
    path = root / "scripts" / "check_c_extensions_fresh.py"
    spec = importlib.util.spec_from_file_location("_deepfin_check_c_extensions_fresh", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MTIME_FALLBACK_NOTE = (
    "mtime check: this extension has no build record, so a checkout or a copy that "
    "rewrites timestamps can also trigger this; recompiling with "
    f"`{FORCE_REBUILD_COMMAND}` (production host: "
    "`python scripts/build_production_extensions.py`) clears it and writes a record"
)


def _freshness_rows(
    root: Path, modules: Sequence[str] = REQUIRED_EXTENSIONS,
) -> list[tuple[str, str, str | None]]:
    """(module, method, problem-or-None) per extension.

    Content check against the build record setup.py wrote beside the built
    file (``chess_anti_engine/ext_build_record.py``); for an extension built
    before records existed, the repo's mtime gate is the fallback.
    """
    from chess_anti_engine.ext_build_record import check_record

    tool = _load_freshness_tool(root)
    deps = {spec.module: tuple(spec.dependencies) for spec in tool.EXTENSION_SPECS}
    unknown = sorted(set(modules) - set(deps))
    if unknown:
        raise RuntimeError(f"freshness tool has no dependency list for {unknown}")
    rows: list[tuple[str, str, str | None]] = []
    mtime_issues: list[str] | None = None
    for name in modules:
        # The in-place build under `root` (what a source-checkout import loads).
        ext_file = tool._first_existing(tool._extension_outputs(root, name))
        if ext_file is None:
            rows.append((name, "none", f"{name} is not built under {root}"))
            continue
        res = check_record(ext_file, root=root, inputs=deps[name])
        if res.verdict == "ok":
            rows.append((name, "content", None))
        elif res.verdict != "no_record":
            rows.append((name, "content", f"{name}: {res.detail}"))
        else:
            if mtime_issues is None:
                mtime_issues = list(tool.check_extensions(root))
            issues = [i for i in mtime_issues if i.split(" ", 1)[0] == name]
            rows.append((name, "mtime", "; ".join(issues) if issues else None))
    return rows


def freshness_problems(root: Path, modules: Sequence[str] = REQUIRED_EXTENSIONS) -> list[str]:
    """Stale-build problems for ``modules`` (content check, mtime fallback)."""
    rows = _freshness_rows(root, modules)
    problems = [problem for _, _, problem in rows if problem]
    if any(problem and method == "mtime" for _, method, problem in rows):
        problems.append(MTIME_FALLBACK_NOTE)
    return problems


def required_abi() -> int:
    from chess_anti_engine.mcts.gumbel_c import _REQUIRED_MCTS_ABI

    return int(_REQUIRED_MCTS_ABI)


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
        if name.endswith("._lc0_ext") and hasattr(module, "CBoard"):
            missing += [
                f"CBoard.{meth}" for meth in REQUIRED_CBOARD_METHODS
                if not hasattr(module.CBoard, meth)
            ]
        if missing:
            problems.append(f"{name} lacks {', '.join(missing)} (built from older source)")
        if name.endswith("._mcts_tree"):
            abi = int(getattr(module, "ABI_VERSION", 0))
            try:
                need = abi_fn()
            except ImportError:
                # gumbel_c needs the other extensions; their failure is reported above.
                continue
            if abi < need:
                problems.append(f"{name} ABI_VERSION={abi} < required {need}")
    return problems


def _python_reference_planes(board: chess.Board, version: str) -> np.ndarray:
    import numpy as np

    from chess_anti_engine.encoding.features import extra_feature_planes_fast
    from chess_anti_engine.encoding.lc0 import encode_lc0_full

    return np.concatenate([
        encode_lc0_full(board),
        extra_feature_planes_fast(board, version=version),
    ])


def _encoder_boards() -> list[chess.Board]:
    import chess

    boards = [chess.Board(fen) for fen in ENCODER_FENS]
    with_history = chess.Board()
    for uci in ENCODER_HISTORY_MOVES:
        with_history.push_uci(uci)
    boards.append(with_history)
    return boards


def feature_planes_nonzero_problem() -> str | None:
    """The engine's encoder on the start position must fill feature planes.

    The cheap form of the encoder check, run by the startup gate: a missing or
    broken feature encoder used to yield 112 LC0 planes plus all-zero feature
    planes, which the net then reads as a real (and wrong) position.
    """
    import chess

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.encoding.cboard_encode import encode_cboard

    planes = encode_cboard(CBoard.from_board(chess.Board()), input_extra_features="v2_threats")
    if not planes[112:].any():
        return "feature planes 112+ are all zero on the start position (feature encoder not taken)"
    return None


def startup_problems(root: Path | None = None, *, check_freshness: bool = True) -> list[str]:
    """Everything the engine refuses to start on. Cheap (milliseconds)."""
    problems = extension_problems()
    if problems:
        return problems  # later checks would only re-raise the same import error
    if check_freshness:
        root = source_root() if root is None else root
        if root is not None:
            problems += [
                issue if issue == MTIME_FALLBACK_NOTE else f"{issue} (stale build)"
                for issue in freshness_problems(root)
            ]
    try:
        zero = feature_planes_nonzero_problem()
    except Exception as exc:
        zero = f"encoder raised {type(exc).__name__}: {exc}"
    if zero is not None:
        problems.append(zero)
    return problems


def enforce_startup_gate(
    *,
    stream=None,
    problems_fn: Callable[[], list[str]] = startup_problems,
) -> int | None:
    """Engine-startup refusal. Returns an exit code to stop with, or None to go on.

    ``DEEPFIN_ALLOW_UNVERIFIED_EXTENSIONS=1`` downgrades the refusal to a
    warning for development (e.g. iterating on a .c file between rebuilds).
    """
    out = sys.stderr if stream is None else stream
    problems = problems_fn()
    if not problems:
        return None
    allow = os.environ.get(ALLOW_UNVERIFIED_ENV, "").strip().lower() in {"1", "true", "yes"}
    head = "deepfin: WARNING, starting anyway" if allow else "deepfin: refusing to start"
    print(f"{head}: the C extensions do not match this install:", file=out)
    for problem in problems:
        print(f"  - {problem}", file=out)
    if allow:
        print(f"  ({ALLOW_UNVERIFIED_ENV} is set; results from this engine are not trustworthy)",
              file=out)
        return None
    print(f"Fix: {REBUILD_HINT}. Diagnose with `deepfin --selfcheck`. "
          f"Development override: {ALLOW_UNVERIFIED_ENV}=1.", file=out)
    out.flush()
    return 2


# ── components ─────────────────────────────────────────────────────────────


def check_python(
    version: tuple[int, int, int] | None = None, *, gil_enabled: bool | None = None,
) -> str:
    v = tuple(sys.version_info[:3]) if version is None else version
    shown = ".".join(str(x) for x in v)
    supported = f">={'.'.join(map(str, MIN_PYTHON))},<{'.'.join(map(str, MAX_PYTHON_EXCLUSIVE))}"
    if v < MIN_PYTHON or v[:2] >= MAX_PYTHON_EXCLUSIVE:
        raise CheckFailed(f"Python {shown} is outside the supported range {supported}")
    if gil_enabled is None:
        probe = getattr(sys, "_is_gil_enabled", None)
        gil_enabled = True if probe is None else bool(probe())
    if not gil_enabled:
        raise CheckFailed(
            f"Python {shown} is running free-threaded (GIL disabled); the C extensions "
            "require a standard GIL-enabled interpreter"
        )
    return f"Python {shown} ({sys.implementation.name}) in {supported}"


def _arch_supports(arch_list: Sequence[str], capability: tuple[int, int]) -> bool:
    """Whether a torch build compiled for ``arch_list`` can run on ``capability``.

    ``sm_XY`` cubins run on the same major with minor >= Y; ``compute_XY`` PTX
    JIT-compiles forward to any newer device. Arch-specific suffixes
    (``sm_90a``) are treated like their base.
    """
    major, minor = capability
    for arch in arch_list:
        kind, _, digits = arch.partition("_")
        digits = "".join(ch for ch in digits if ch.isdigit())
        if len(digits) < 2:
            continue
        a_major, a_minor = int(digits[:-1]), int(digits[-1])
        if kind == "sm" and a_major == major and a_minor <= minor:
            return True
        if kind == "compute" and (a_major, a_minor) <= (major, minor):
            return True
    return False


def resolve_devices(requested: Sequence[str], torch_mod: ModuleType | None = None) -> list[str]:
    """``--devices``/``--device`` as the engine would resolve them."""
    devices = [d for d in requested if d and d != "auto"]
    if devices:
        return devices
    torch = importlib.import_module("torch") if torch_mod is None else torch_mod
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def check_torch(requested: Sequence[str], torch_mod: ModuleType | None = None) -> str:
    try:
        torch = importlib.import_module("torch") if torch_mod is None else torch_mod
    except ImportError as exc:
        raise CheckFailed(f"torch is not importable ({exc}); see docs/install.md") from exc
    cuda_build = getattr(torch.version, "cuda", None)
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    parts = [f"torch {torch.__version__}", f"CUDA build {cuda_build or 'none (CPU wheel)'}"]
    gpus = [f"cuda:{i}={torch.cuda.get_device_name(i)}" for i in range(count)]
    parts.append(f"GPUs: {', '.join(gpus) if gpus else 'none visible'}")
    devices = resolve_devices(requested, torch)
    arch_list = list(torch.cuda.get_arch_list()) if available else []
    for dev in devices:
        if dev == "cpu":
            continue
        if not dev.startswith("cuda"):
            raise CheckFailed(f"unsupported device {dev!r} (use cpu or cuda[:N])")
        if not available:
            raise CheckFailed(
                f"device {dev} requested but torch.cuda.is_available() is False "
                f"(CUDA build {cuda_build or 'none: this is the CPU wheel'}); install a CUDA "
                "build of torch that supports this GPU and driver (docs/install.md)"
            )
        index = int(dev.split(":", 1)[1]) if ":" in dev else 0
        if index >= count:
            raise CheckFailed(f"device {dev} requested but only {count} CUDA device(s) are visible")
        cap = tuple(torch.cuda.get_device_capability(index))
        cap_str = f"sm_{cap[0]}{cap[1]}"
        if not _arch_supports(arch_list, (int(cap[0]), int(cap[1]))):
            raise CheckFailed(
                f"{dev} ({torch.cuda.get_device_name(index)}) is {cap_str}, but this torch "
                f"(CUDA {cuda_build}) was built for {' '.join(arch_list) or 'no architectures'}; "
                f"install a torch build whose CUDA version supports {cap_str} "
                "(a newer cuXXX wheel, see docs/install.md)"
            )
        try:
            value = float((torch.ones(8, device=dev) * 2).sum().item())
        except Exception as exc:
            raise CheckFailed(f"tensor op on {dev} failed: {type(exc).__name__}: {exc}") from exc
        if value != 16.0:
            raise CheckFailed(f"tensor op on {dev} returned {value}, expected 16.0")
    parts.append(f"tensor op OK on {', '.join(devices)}")
    return "; ".join(parts)


def check_extensions(*, abi_required: Callable[[], int] | None = None) -> str:
    abi_fn = required_abi if abi_required is None else abi_required
    problems = extension_problems(abi_required=abi_fn)
    if problems:
        raise CheckFailed("; ".join(problems) + f"; fix: {REBUILD_HINT}")
    mcts = importlib.import_module("chess_anti_engine.mcts._mcts_tree")
    return (
        f"{len(REQUIRED_EXTENSIONS)} required extensions import with the expected symbols; "
        f"_mcts_tree ABI_VERSION={mcts.ABI_VERSION} (required {abi_fn()})"
    )


def check_freshness(root: Path | None = None, *, use_source_root: bool = True) -> str:
    if root is None and use_source_root:
        root = source_root()
    if root is None:
        raise CheckSkipped(
            "not a source checkout (regular wheel install): no sources to compare; the "
            "ABI/symbol and behavioural checks still apply"
        )
    rows = _freshness_rows(root)
    problems = freshness_problems(root)
    if problems:
        raise CheckFailed("; ".join(problems))
    by_content = [name.rsplit(".", 1)[-1] for name, method, _ in rows if method == "content"]
    by_mtime = [name.rsplit(".", 1)[-1] for name, method, _ in rows if method == "mtime"]
    detail = []
    if by_content:
        detail.append(f"source content matches the build record for {', '.join(by_content)}")
    if by_mtime:
        detail.append(f"no build record, newer than sources by mtime: {', '.join(by_mtime)}")
    return "; ".join(detail)


def check_encoder(
    *,
    c_encode: Callable[..., np.ndarray] | None = None,
    reference: Callable[[chess.Board, str], np.ndarray] = _python_reference_planes,
) -> str:
    import numpy as np

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.encoding.cboard_encode import encode_cboard
    from chess_anti_engine.mcts._mcts_tree import batch_encode_146

    encode = encode_cboard if c_encode is None else c_encode
    boards = _encoder_boards()
    for board in boards:
        for version in FEATURE_VERSIONS:
            got = np.asarray(encode(CBoard.from_board(board), input_extra_features=version))
            if not got[112:].any():
                raise CheckFailed(
                    f"{version} feature planes are all zero for {board.fen()} "
                    "(C feature encoder not taken)"
                )
            want = np.asarray(reference(board, version))
            if got.shape != want.shape or not np.allclose(got, want, rtol=0.0, atol=ENCODER_ATOL):
                diff = (float(np.abs(got - want).max()) if got.shape == want.shape
                        else f"shape {got.shape} vs {want.shape}")
                raise CheckFailed(
                    f"C encoder differs from the Python reference ({version}, "
                    f"{board.fen()}): max |diff| {diff}"
                )
    # The search tree encodes leaves with its own compiled copy of the encoder.
    out = np.zeros((len(boards), 175, 8, 8), dtype=np.float32)
    batch_encode_146([CBoard.from_board(b) for b in boards], out)
    for i, board in enumerate(boards):
        want = np.asarray(reference(board, "v2_threats"))
        if not np.allclose(out[i], want, rtol=0.0, atol=ENCODER_ATOL):
            raise CheckFailed(
                f"C tree leaf encoder differs from the Python reference for {board.fen()}"
            )
    return (
        f"C encoders (CBoard + tree leaf encoder) match the Python reference on "
        f"{len(boards)} positions x {'/'.join(FEATURE_VERSIONS)}; feature planes non-zero"
    )


def check_movegen() -> str:
    import chess

    from chess_anti_engine.encoding._lc0_ext import CBoard

    for board in _encoder_boards():
        c_count = len(CBoard.from_board(board).legal_move_indices())
        py_count = board.legal_moves.count()
        if c_count != py_count:
            raise CheckFailed(
                f"CBoard has {c_count} legal moves, python-chess {py_count}, for {board.fen()}"
            )
    try:
        perft_mod = importlib.import_module("chess_anti_engine.encoding._perft_ext")
    except ImportError:
        return (f"CBoard legal-move counts match python-chess on {len(ENCODER_FENS) + 1} "
                "positions (perft extension not built; perft skipped)")
    for fen, depth, nodes in PERFT_CASES:
        got = int(perft_mod.perft(CBoard.from_board(chess.Board(fen)), depth))
        if got != nodes:
            raise CheckFailed(f"perft({depth}) = {got}, expected {nodes}, for {fen}")
    return (f"CBoard legal-move counts match python-chess; perft "
            f"{', '.join(f'd{d}={n}' for _, d, n in PERFT_CASES)} correct")


def _tiny_model():
    import torch

    from chess_anti_engine.model import ModelConfig, build_model

    torch.manual_seed(0)
    return build_model(ModelConfig(kind="tiny")).eval()


def check_search(device: str = "cpu") -> str:
    import chess
    import numpy as np

    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.mcts._mcts_tree import MCTSTree
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
    from chess_anti_engine.moves import index_to_move_strict

    board = chess.Board()
    evaluator = LocalModelEvaluator(_tiny_model(), device=device)
    result = run_gumbel_root_many_c(
        None, [board], device=device, rng=np.random.default_rng(0),
        cfg=GumbelConfig(topk=8, add_noise=False, temperature=0.0),
        evaluator=evaluator, per_game_simulations=[SEARCH_SIMS],
    )
    # Typed as object: the runtime check is the point (a Python fallback would
    # hand back its own tree type).
    tree = cast("object", result[4])
    roots = result[5]
    if not isinstance(tree, MCTSTree):
        raise CheckFailed(f"search returned a {type(tree).__name__}, not the C MCTSTree")
    _, visits = tree.get_children_visits(roots[0])
    total = int(np.asarray(visits).sum())
    if total != SEARCH_SIMS:
        raise CheckFailed(f"C tree root holds {total} visits after a {SEARCH_SIMS}-sim search")
    move = index_to_move_strict(int(result[1][0]), board)
    if move not in board.legal_moves:
        raise CheckFailed(f"C search chose {move.uci()}, which is not legal")
    return (f"{SEARCH_SIMS}-sim C Gumbel search on a random tiny net ({device}): "
            f"{total} visits in the C tree, {tree.node_count()} nodes, chose legal {move.uci()}")


def check_checkpoint(
    path: str | None, device: str = "cpu", *, evaluator_factory: Callable[..., Any] | None = None,
) -> str:
    """Load exactly as the engine does, then demand a complete load and a finite forward.

    ``load_model_from_checkpoint`` is the engine's loader; its tolerant
    migrations (torch.compile prefix, fused qkv, v1 -> v2_threats input columns)
    are reported as notes, not failures. FAIL means the loader refused, or a
    model tensor was left at its random initialisation, or the forward pass is
    not finite.
    """
    if not path:
        raise CheckSkipped("no --checkpoint given (or $DEEPFIN_CKPT unset)")
    import chess
    import numpy as np

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.encoding.cboard_encode import encode_cboard
    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.uci.model_loader import _resolve_trainer_pt, load_model_from_checkpoint

    trainer_pt = _resolve_trainer_pt(Path(path))
    loader_log = io.StringIO()
    try:
        with contextlib.redirect_stdout(loader_log):
            model = load_model_from_checkpoint(str(path), device=device)
    except Exception as exc:
        raise CheckFailed(
            f"the engine's checkpoint loader refuses {trainer_pt.name}: {type(exc).__name__}: {exc}"
        ) from exc
    report = getattr(model, "tolerant_load_report", None)
    if report is None:
        raise CheckFailed("the checkpoint loader returned no load report")
    uninitialised = [*report.missing, *report.shape_skipped]
    if uninitialised:
        raise CheckFailed(
            f"{len(uninitialised)} model tensor(s) left at random initialisation by the engine's "
            f"tolerant loader (missing {list(report.missing)[:3]}, shape-skipped "
            f"{list(report.shape_skipped)[:3]}); the engine would play a partly random net"
        )
    notes = [
        line.strip() for line in loader_log.getvalue().splitlines()
        if line.strip().startswith("[uci-load]") and "Tolerant load" not in line
    ]
    if report.compile_prefix_normalized:
        notes.append("torch.compile _orig_mod. key prefix normalised")
    if report.unexpected:
        notes.append(f"{len(report.unexpected)} checkpoint tensor(s) not used by the model")
    history = str(getattr(model, "input_history_encoding", "legacy"))
    features = str(getattr(model, "input_extra_features", "v1"))
    x = encode_cboard(
        CBoard.from_board(chess.Board()),
        input_history_encoding=history, input_extra_features=features,
    )[None].astype(np.float32)
    factory = LocalModelEvaluator if evaluator_factory is None else evaluator_factory
    evaluator = factory(model, device=device)
    policy, wdl = evaluator.evaluate_encoded(x)
    policy, wdl = np.asarray(policy, dtype=np.float64), np.asarray(wdl, dtype=np.float64)
    if not (np.isfinite(policy).all() and np.isfinite(wdl).all()):
        raise CheckFailed("forward pass on the start position produced non-finite outputs")
    # parameters() de-duplicates tied tensors (one Smolgen weight, many keys).
    n_params = sum(p.numel() for p in model.parameters())
    detail = (f"{trainer_pt.name}: {type(model).__name__}, {n_params:,} params, every model "
              f"tensor loaded; input {x.shape[1]} planes ({history}/{features}); forward finite "
              f"on {device}")
    if notes:
        detail += "; loader notes: " + " | ".join(notes)
    return detail


def _probe_wdl_raw(syzygy_path: str, fen: str) -> int:
    import chess

    from chess_anti_engine.tablebase import get_tablebase

    tb = get_tablebase(syzygy_path)
    if tb is None:
        raise CheckFailed(f"no Syzygy directory in {syzygy_path!r} could be opened")
    return int(tb.probe_wdl(chess.Board(fen)))


def check_syzygy(
    path: str | None, *, probe: Callable[[str, str], int] = _probe_wdl_raw,
) -> str:
    if not path:
        raise CheckSkipped("no --syzygy path given")
    dirs = [d.strip() for d in path.split(os.pathsep) if d.strip()]
    absent = [d for d in dirs if not Path(d).is_dir()]
    if not dirs or absent:
        raise CheckFailed(f"Syzygy directories do not exist: {absent or path!r}")
    tables = {f.name for d in dirs for f in Path(d).iterdir() if f.suffix == ".rtbw"}
    if not tables:
        raise CheckFailed(f"no .rtbw (WDL) tables under {dirs}")

    def _probe(fen: str, want: int) -> None:
        try:
            got = probe(path, fen)
        except CheckFailed:
            raise
        except Exception as exc:
            raise CheckFailed(f"probe failed for {fen}: {type(exc).__name__}: {exc}") from exc
        if got != want:
            raise CheckFailed(f"WDL for {fen} = {got}, expected {want}")

    for fen, want in SYZYGY_CASES:
        _probe(fen, want)
    probed = [f"{len(SYZYGY_CASES)} known 3-man positions"]
    skipped = []
    for men, table, fen, want in SYZYGY_LARGER_CASES:
        if f"{table}.rtbw" in tables:
            _probe(fen, want)
            probed.append(f"{men}-man {table}")
        else:
            skipped.append(f"{men}-man ({table}.rtbw not in the path)")
    detail = (f"{len(dirs)} director{'y' if len(dirs) == 1 else 'ies'}, {len(tables)} WDL tables; "
              f"probe correctly: {', '.join(probed)}")
    if skipped:
        detail += f"; not probed: {', '.join(skipped)}"
    return detail


def _write_tiny_checkpoint(directory: Path) -> Path:
    import torch

    ckpt_dir = directory / "checkpoint_000000"
    ckpt_dir.mkdir()
    torch.save({"model": _tiny_model().state_dict(), "step": 0}, ckpt_dir / "trainer.pt")
    (directory / "params.json").write_text(
        json.dumps({"model": "tiny", "input_extra_features": "v2_threats"}), encoding="utf-8",
    )
    return ckpt_dir


def uci_mode(opts: SelfcheckOptions) -> str:
    """Which engine configuration the UCI component exercises, in words."""
    flags = " ".join(opts.engine_args)
    given = f" with engine flags `{flags}`" if flags else ""
    if opts.fast:
        return f"fast: --no-compile{given} (does NOT cover the default compiled play path)"
    if "--no-compile" in opts.engine_args:
        return f"operator flags{given}"
    return f"default play path, torch.compile on{given}"


def engine_command(checkpoint: str, opts: SelfcheckOptions, compile_cache_dir: str) -> list[str]:
    """The engine invocation: the operator's flags, plus a private compile cache.

    ``--compile-cache-dir`` comes first so an operator-supplied one (in
    ``engine_args``) wins; otherwise the selfcheck never writes into a shared
    production cache.
    """
    cmd = [sys.executable, "-u", "-m", "chess_anti_engine.uci", "--checkpoint", checkpoint,
           "--compile-cache-dir", compile_cache_dir]
    if len(opts.devices) > 1:
        cmd += ["--devices", ",".join(opts.devices)]
    elif opts.devices:
        cmd += ["--device", opts.devices[0]]
    cmd += list(opts.engine_args)
    if opts.fast:
        cmd.append("--no-compile")
    return cmd


def _failure_lines(lines: Sequence[str]) -> list[str]:
    return [ln for ln in lines if any(m in ln for m in UCI_FAILURE_MARKERS)]


_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt): ")


def _failure_reason(lines: Sequence[str]) -> str:
    """The engine's own failure line plus the root cause from its traceback.

    Strict startup prints only the outermost exception ("evaluator warmup
    failed at batch=1"); the cause an operator can act on (e.g. a libstdc++
    mismatch inside torch.compile) is the first exception line of the logged
    traceback.
    """
    marked = _failure_lines(lines)
    exc_lines = list(dict.fromkeys(ln.strip() for ln in lines if _EXC_LINE.match(ln.strip())))
    parts = [marked[-1].strip()] if marked else []
    if exc_lines and (not parts or exc_lines[0] not in parts[0]):
        parts.append("cause: " + exc_lines[0])
    if not parts:  # e.g. an argparse error for a flag the engine does not accept
        tail = [ln.strip() for ln in lines if ln.strip()]
        parts = tail[-1:]
    return _one_line("; ".join(parts), 600) if parts else ""


class _Transcript:
    """Pump the engine's merged stdout/stderr, keeping every line."""

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self.lines: list[str] = []
        self._q: queue.Queue[str | None] = queue.Queue()
        self._proc = proc
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self._proc.stdout is not None
        for line in iter(self._proc.stdout.readline, ""):
            self._q.put(line.rstrip("\n"))
        self._q.put(None)

    def read_until(self, token: str, timeout_s: float) -> list[str]:
        got: list[str] = []
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no {token!r} within {timeout_s:g}s")
            try:
                line = self._q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(f"engine exited before {token!r}")
            self.lines.append(line)
            got.append(line)
            if line.split(maxsplit=1)[:1] == [token]:
                return got

    def drain(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                line = self._q.get(timeout=0.2)
            except queue.Empty:
                if self._proc.poll() is not None:
                    return
                continue
            if line is None:
                return
            self.lines.append(line)


def check_uci(
    opts: SelfcheckOptions,
    *, command: Callable[[str, SelfcheckOptions, str], list[str]] = engine_command,
) -> str:
    import chess

    from chess_anti_engine.uci.subprocess_client import send_line

    compiled = not opts.fast and "--no-compile" not in opts.engine_args
    timeout = opts.uci_timeout_s or (UCI_TIMEOUT_COMPILED_S if compiled else UCI_TIMEOUT_FAST_S)
    with tempfile.TemporaryDirectory(prefix="deepfin-selfcheck-") as tmp:
        ckpt = opts.checkpoint or str(_write_tiny_checkpoint(Path(tmp)))
        label = "given checkpoint" if opts.checkpoint else "random-weight tiny net"
        cache = str(Path(tmp) / "engine_cache")
        started = time.monotonic()
        proc = subprocess.Popen(
            command(ckpt, opts, cache), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        out = _Transcript(proc)
        lines: list[str] = []
        try:
            send_line(proc, "uci")
            out.read_until("uciok", timeout)
            if opts.syzygy_path:
                send_line(proc, f"setoption name SyzygyPath value {opts.syzygy_path}")
            send_line(proc, "isready")
            out.read_until("readyok", timeout)
            send_line(proc, "ucinewgame")
            send_line(proc, "position startpos")
            send_line(proc, f"go nodes {UCI_GO_NODES}")
            lines = out.read_until("bestmove", timeout)
            send_line(proc, "quit")
            proc.wait(timeout=30)
            out.drain()
        except (TimeoutError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
            out.drain()
            reason = _failure_reason(out.lines) or str(exc)
            raise CheckFailed(
                f"engine ({uci_mode(opts)}) failed before bestmove, exit {proc.returncode}: {reason}"
            ) from exc
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        bad = _failure_lines(out.lines)
        if bad:
            raise CheckFailed(f"engine ({uci_mode(opts)}) reported: {_failure_reason(out.lines)}")
        token = next(ln for ln in lines if ln.split()[:1] == ["bestmove"]).split()[1]
        try:
            move = chess.Move.from_uci(token)
        except ValueError as exc:
            raise CheckFailed(f"bestmove {token!r} is not a UCI move") from exc
        if move not in chess.Board().legal_moves:
            raise CheckFailed(f"bestmove {token} is not legal in the start position")
        if proc.returncode != 0:
            raise CheckFailed(f"engine exited {proc.returncode} after quit")
        return (f"uci/uciok, isready/readyok, go nodes {UCI_GO_NODES} -> bestmove {token} "
                f"({label}, {uci_mode(opts)}, {time.monotonic() - started:.1f}s, subprocess)")


# ── driver ─────────────────────────────────────────────────────────────────


def run_selfcheck(opts: SelfcheckOptions) -> list[CheckResult]:
    results = [
        _run_check("python", check_python),
        _run_check("torch", lambda: check_torch(opts.devices)),
        _run_check("extensions", check_extensions),
        _run_check("freshness", check_freshness),
        _run_check("encoder", check_encoder),
        _run_check("movegen", check_movegen),
    ]
    torch_ok = results[1].status == "PASS"
    try:
        devices = resolve_devices(opts.devices) if torch_ok else list(opts.devices) or ["cpu"]
    except Exception:
        devices = ["cpu"]
    results += [
        # The search check exercises the C tree, not the accelerator; CPU keeps
        # it independent of the torch/CUDA result above.
        _run_check("search", lambda: check_search("cpu")),
        _run_check("checkpoint", lambda: check_checkpoint(opts.checkpoint, devices[0])),
        _run_check("syzygy", lambda: check_syzygy(opts.syzygy_path)),
    ]
    if opts.run_uci:
        results.append(_run_check("uci", lambda: check_uci(opts)))
    else:
        results.append(CheckResult("uci", "SKIP", "--no-uci given"))
    return results


def overall(results: Sequence[CheckResult]) -> Status:
    return "FAIL" if any(r.status == "FAIL" for r in results) else "PASS"


def skipped(results: Sequence[CheckResult]) -> list[str]:
    return [r.name for r in results if r.status == "SKIP"]


def summary_line(results: Sequence[CheckResult], mode: str | None) -> str:
    parts = []
    if mode is not None:
        parts.append(f"uci: {mode}")
    names = skipped(results)
    if names:
        parts.append(f"{len(names)} skipped: {', '.join(names)}")
    tail = f" ({'; '.join(parts)})" if parts else ""
    return f"SELFCHECK {overall(results)}{tail}"


def format_text(results: Sequence[CheckResult], mode: str | None = None) -> str:
    width = max(len(r.name) for r in results)
    lines = [f"{r.status:<4}  {r.name:<{width}}  {r.detail}" for r in results]
    lines.append(summary_line(results, mode))
    return "\n".join(lines)


def format_json(results: Sequence[CheckResult], mode: str | None = None) -> str:
    return json.dumps(
        {
            "overall": overall(results),
            "skipped": skipped(results),
            "uci_mode": mode,
            "results": [asdict(r) for r in results],
        },
        indent=2,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deepfin --selfcheck",
        allow_abbrev=False,  # an abbreviation must not swallow an engine flag
        description="Verify a DeepFin engine install: interpreter, torch/GPU, C extensions "
                    "(ABI, staleness, behaviour), optional checkpoint and Syzygy, and a UCI "
                    "exchange with the engine's real flags. Any argument not listed here is "
                    "passed to the engine subprocess unchanged (e.g. --walkers 2). "
                    "Exit 0 only if nothing FAILs.",
    )
    p.add_argument("--selfcheck", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true", help="print results as JSON")
    p.add_argument("--checkpoint", default=os.environ.get("DEEPFIN_CKPT"),
                   help="network to verify (trainer.pt or checkpoint dir; default $DEEPFIN_CKPT). "
                        "Without one the checkpoint check is SKIP and the UCI exchange uses a "
                        "random-weight tiny net")
    p.add_argument("--syzygy", dest="syzygy_path", default=None,
                   help="Syzygy directories, os.pathsep-separated; probed directly and given "
                        "to the engine as SyzygyPath")
    p.add_argument("--device", default="auto", help="cpu|cuda|cuda:N (default: auto)")
    p.add_argument("--devices", default=None,
                   help="comma-separated device list for multi-GPU hosts; overrides --device")
    p.add_argument("--fast", action="store_true",
                   help="run the UCI exchange with --no-compile (seconds instead of a cold "
                        "torch.compile); the verdict then does not cover the default play path")
    p.add_argument("--uci-timeout", dest="uci_timeout_s", type=float, default=None,
                   help=f"seconds to wait for each UCI reply (default {UCI_TIMEOUT_COMPILED_S:.0f} "
                        f"compiled, {UCI_TIMEOUT_FAST_S:.0f} with --no-compile)")
    p.add_argument("--no-uci", dest="run_uci", action="store_false",
                   help="skip the subprocess UCI exchange")
    return p


def parse_options(argv: Sequence[str]) -> tuple[SelfcheckOptions, bool]:
    args, engine_args = build_parser().parse_known_args(list(argv))
    if args.devices:
        devices = tuple(d.strip() for d in str(args.devices).split(",") if d.strip())
    elif args.device and args.device != "auto":
        devices = (str(args.device),)
    else:
        devices = ()
    opts = SelfcheckOptions(
        checkpoint=args.checkpoint, syzygy_path=args.syzygy_path,
        devices=devices, run_uci=bool(args.run_uci),
        engine_args=tuple(engine_args), fast=bool(args.fast),
        uci_timeout_s=args.uci_timeout_s,
    )
    return opts, bool(args.json)


def main(argv: Sequence[str] | None = None) -> int:
    opts, as_json = parse_options(sys.argv[1:] if argv is None else list(argv))
    # Library chatter (e.g. the C tree's one-time stderr notice) stays off stdout.
    with contextlib.redirect_stdout(sys.stderr):
        results = run_selfcheck(opts)
    mode = uci_mode(opts) if opts.run_uci else None
    print(format_json(results, mode) if as_json else format_text(results, mode), flush=True)
    return 0 if overall(results) == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
