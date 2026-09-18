from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import subprocess

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    _completed_q_transform,
    halving_keep_count,
    halving_visits_per_action,
)


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "native" / "bend_engine" / "search_probe" / "build_probe.sh"

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

_CFG = GumbelConfig(
    c_scale=0.1,
    c_visit=50.0,
    c_visit_root=-1.0,
    c_scale_root=-1.0,
    q_visit_exp_root=99.0,
    halving_div=2,
)
_ROOT_Q = 0.125


def _f32(value: float | int | np.floating) -> np.float32:
    return np.float32(value)


def _prior(action: int) -> float:
    return float(_f32(_f32((action % 29) + 1) / _f32(30.0)))


def _gumbel(action: int) -> float:
    base = _f32(
        _f32(_f32((action * 17) % 101) - _f32(50.0)) / _f32(25.0)
    )
    return float(_f32(base + _f32(_f32(action) / _f32(100000.0))))


def _q(action: int) -> float:
    return float(
        _f32(
            _f32(_f32((action * 13) % 61) - _f32(30.0))
            / _f32(30.0)
        )
    )


def _initial_score(action: int) -> float:
    return float(_f32(_f32(_gumbel(action)) + _f32(math.log(_prior(action)))))


def _seed_xor(actions: list[int]) -> int:
    out = 0
    for action in actions:
        out ^= (action * 2246822519) & 0xFFFFFFFF
    return out


def _visit_xor(legal: list[int], visits: dict[int, int]) -> int:
    out = 0
    for action in legal:
        word = ((action * 2654435761) & 0xFFFFFFFF) ^ visits.get(action, 0)
        out ^= word
    return out


def _expected(fen: str) -> dict[str, int]:
    board = chess.Board(fen)
    cboard = CBoard.from_board(board)
    legal = [int(x) for x in cboard.legal_move_indices().tolist()]
    assert legal

    ranked = sorted(legal, key=_initial_score, reverse=True)
    active = ranked[:8]
    sampled = len(active)
    visits = {action: 0 for action in legal}
    budget = 64

    for _ in range(3):
        if len(active) <= 1:
            break
        vpa = halving_visits_per_action(len(active), budget, 2)
        for action in active:
            visits[action] += vpa
        budget = max(0, budget - vpa * len(active))

        visit_arr = np.asarray([visits[a] for a in legal], dtype=np.float64)
        prior_arr = np.asarray([_prior(a) for a in legal], dtype=np.float64)
        q_arr = np.asarray(
            [_q(a) if visits[a] > 0 else _ROOT_Q for a in legal],
            dtype=np.float64,
        )
        q_logits = _completed_q_transform(
            actions=legal,
            priors=prior_arr,
            visits=visit_arr,
            qvalues=q_arr,
            raw_value=_ROOT_Q,
            cfg=_CFG,
            root=True,
        )
        q_by_action = {
            action: float(q_logits[i])
            for i, action in enumerate(legal)
        }
        active = sorted(
            active,
            key=lambda action: (
                _gumbel(action)
                + math.log(_prior(action))
                + q_by_action[action]
            ),
            reverse=True,
        )[: halving_keep_count(len(active), 2)]

    return {
        "legal": len(legal),
        "sampled": sampled,
        "seed_xor": _seed_xor(ranked[:8]),
        "winner": active[0],
        "active": len(active),
        "budget": budget,
        "total_visits": sum(visits.values()),
        "visit_xor": _visit_xor(legal, visits),
    }


def _parse(stdout: str) -> dict[int, dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith("fixture="):
            continue
        fields = dict(part.split("=", 1) for part in line.split())
        fixture = int(fields.pop("fixture"))
        rows[fixture] = {key: int(value) for key, value in fields.items()}
    return rows


def _run_checked(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
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


def test_bend_gumbel_halving_matches_deepfin_reference(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["BEND_SEARCH_PROBE_BUILD_DIR"] = str(tmp_path / "build")

    built = _run_checked([str(BUILD_SCRIPT)], env=env)
    binary = Path(built.stdout.strip().splitlines()[-1])
    assert binary.is_file(), built.stdout

    run = _run_checked([str(binary)])
    observed = _parse(run.stdout)

    assert set(observed) == set(range(len(FIXTURE_FENS))), run.stdout
    for fixture, fen in enumerate(FIXTURE_FENS):
        assert observed[fixture] == _expected(fen)
