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

FIXTURE_FENS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "k3r3/8/8/8/8/8/8/4K3 w - - 0 1",
)

pytestmark = pytest.mark.skipif(
    shutil.which("bend") is None or shutil.which("clang") is None,
    reason="Bend/clang native toolchain is not installed",
)


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


def test_bend_chess_probe_matches_existing_cboard(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["BEND_PROBE_BUILD_DIR"] = str(tmp_path / "build")

    built = subprocess.run(
        [str(BUILD_SCRIPT)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    binary = Path(built.stdout.strip().splitlines()[-1])
    assert binary.is_file(), built.stdout

    run = subprocess.run(
        [str(binary)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    observed = _parse_probe(run.stdout)

    assert set(observed) == set(range(len(FIXTURE_FENS))), run.stdout
    for fixture, fen in enumerate(FIXTURE_FENS):
        assert observed[fixture] == _expected(fen)
