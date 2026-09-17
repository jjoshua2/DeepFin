from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "native" / "bend_engine" / "probe" / "build_probe.sh"
BEND_VERSION_FILE = ROOT / "native" / "bend_engine" / "BEND_VERSION"

FIXTURE_FENS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "k3r3/8/8/8/8/8/8/4K3 w - - 0 1",
)


def _first_executable(candidates: list[str | None]) -> str | None:
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        found = shutil.which(raw)
        if found:
            return found
    return None


def _bend_bin() -> str | None:
    return _first_executable(
        [
            os.environ.get("BEND_BIN"),
            "bend",
            str(ROOT / "build" / "bend_toolchain" / "bin" / "bend"),
            str(Path.home() / ".bend" / "bin" / "bend"),
        ]
    )


def _cc_bin() -> str | None:
    return _first_executable([os.environ.get("CC"), "clang", "cc"])


_PROBE_TOOLCHAIN_REASON = "Bend/clang native toolchain is not installed"


def _f32(value: float | int | np.floating) -> np.float32:
    return np.float32(value)


def _puct_score(action: int) -> np.float32:
    prior = _f32(_f32((action % 97) + 1) / _f32(100.0))
    q = _f32(_f32(action % 37) / _f32(100.0))
    visits = action % 11
    root = _f32(np.sqrt(_f32(128.0)))
    numerator = _f32(_f32(1.5) * _f32(prior * root))
    explore = _f32(numerator / _f32(visits + 1))
    return _f32(q + explore)


def _checksum(actions: list[int]) -> int:
    acc = 0
    for action in actions:
        acc = ((acc * 2654435761) & 0xFFFFFFFF) ^ ((action + 1) & 0xFFFFFFFF)
    return acc


def _select(actions: list[int]) -> tuple[int, int]:
    assert actions
    best = actions[0]
    best_score = _puct_score(best)
    for action in actions[1:]:
        score = _puct_score(action)
        if bool(score > best_score):
            best = action
            best_score = score
    return best, _checksum(actions)


def _parse_probe(stdout: str) -> dict[int, dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith("fixture="):
            continue
        fields = dict(part.split("=", 1) for part in line.split())
        fixture = int(fields.pop("fixture"))
        rows[fixture] = {key: int(value) for key, value in fields.items()}
    return rows


def _expected(fen: str) -> dict[str, int]:
    board = chess.Board(fen)
    cboard = CBoard.from_board(board)
    legal = [int(x) for x in cboard.legal_move_indices().tolist()]
    best, checksum = _select(legal)

    child = cboard.copy()
    child.push_index(best)
    child_legal = [int(x) for x in child.legal_move_indices().tolist()]
    child_board = chess.Board(child.fen())

    return {
        "legal": len(legal),
        "best": best,
        "checksum": checksum,
        "child_legal": len(child_legal),
        "child_checksum": _checksum(child_legal),
        "child_check": int(child_board.is_check()),
        "child_hash32": int(child.zobrist_hash) & 0xFFFFFFFF,
    }


def _run_checked(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"command failed with exit {result.returncode}: {' '.join(args)}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result


def test_probe_fixtures_cover_claimed_mechanics() -> None:
    """Legal-set claims must hold even when the Bend toolchain is absent."""
    start, castle, ep, promo, check = (chess.Board(fen) for fen in FIXTURE_FENS)
    assert len(list(start.legal_moves)) == 20
    assert chess.Move.from_uci("e1g1") in castle.legal_moves
    assert chess.Move.from_uci("e1c1") in castle.legal_moves
    assert chess.Move.from_uci("e5d6") in ep.legal_moves
    assert {move.uci() for move in promo.legal_moves if move.promotion} == {
        "a7a8q",
        "a7a8r",
        "a7a8b",
        "a7a8n",
    }
    assert check.is_check()
    assert {move.uci() for move in check.legal_moves} == {
        "e1d1",
        "e1d2",
        "e1f1",
        "e1f2",
    }


@pytest.mark.skipif(_bend_bin() is None or _cc_bin() is None, reason=_PROBE_TOOLCHAIN_REASON)
def test_bend_chess_probe_matches_existing_cboard(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["BEND_NO_TELEMETRY"] = "1"
    env["BEND_PROBE_BUILD_DIR"] = str(tmp_path / "build")
    bend = _bend_bin()
    cc = _cc_bin()
    assert bend is not None
    assert cc is not None
    env["BEND_BIN"] = bend
    env["CC"] = cc

    built = _run_checked([str(BUILD_SCRIPT)], env=env)
    binary = Path(built.stdout.strip().splitlines()[-1])
    assert binary.is_file(), built.stdout
    version = BEND_VERSION_FILE.read_text(encoding="utf-8").splitlines()[0].strip()
    assert version in built.stdout, built.stdout

    run = _run_checked([str(binary)])
    observed = _parse_probe(run.stdout)

    assert set(observed) == set(range(len(FIXTURE_FENS))), run.stdout
    for fixture, fen in enumerate(FIXTURE_FENS):
        assert observed[fixture] == _expected(fen)
