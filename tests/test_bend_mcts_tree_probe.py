from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "native" / "bend_engine" / "tree_probe" / "build_probe.sh"

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

CPUCT = 1.5
FPU_ROOT = 0.25
FPU_TREE = 0.15
SIMS = 24


def _prior(action: int) -> float:
    return ((action % 4) + 1) / 8.0


def _child_value(action: int) -> float:
    return (((action * 5) % 7) - 3) / 4.0


def _grand_value(root_action: int, kid_action: int) -> float:
    return (((root_action * 3 + kid_action) % 7) - 3) / 4.0


def _legal4(board: CBoard) -> list[int]:
    return [int(x) for x in board.legal_move_indices().tolist()[:4]]


def _path_word(actions: list[int]) -> int:
    if len(actions) == 1:
        return ((actions[0] * 2654435761) & 0xFFFFFFFF) ^ 1
    if len(actions) == 2:
        return (
            ((actions[0] * 2654435761) & 0xFFFFFFFF)
            ^ ((actions[1] * 2246822519) & 0xFFFFFFFF)
            ^ 2
        )
    raise AssertionError(f"unexpected path depth {len(actions)}: {actions}")


def _reference(fen: str) -> dict[str, int]:
    root_board = CBoard.from_board(chess.Board(fen))
    root_actions = _legal4(root_board)
    assert root_actions

    tree = MCTSTree()
    root = int(tree.add_root(0, 0.0))
    tree.expand(
        root,
        np.asarray(root_actions, dtype=np.int32),
        np.asarray([_prior(a) for a in root_actions], dtype=np.float64),
    )

    path_mix = 0
    root_w = 0.0

    for _ in range(SIMS):
        selected = tree.select_leaves(
            np.asarray([root], dtype=np.int32),
            CPUCT,
            FPU_ROOT,
            FPU_TREE,
        )[0]
        leaf_id = int(selected[0])
        action_path = [int(x) for x in np.asarray(selected[1]).tolist()]
        node_path = np.asarray(selected[2], dtype=np.int32)
        is_expanded = bool(selected[3])

        path_mix = (
            ((path_mix * 16777619) & 0xFFFFFFFF)
            ^ _path_word(action_path)
        )

        if len(action_path) == 1:
            action = action_path[0]
            value = _child_value(action)
            root_w -= value

            if not is_expanded:
                child_board = root_board.copy()
                child_board.push_index(action)
                kids = _legal4(child_board)
                tree.expand(
                    leaf_id,
                    np.asarray(kids, dtype=np.int32),
                    np.asarray([_prior(a) for a in kids], dtype=np.float64),
                )
        elif len(action_path) == 2:
            action, kid_action = action_path
            value = _grand_value(action, kid_action)
            root_w += value

            if not is_expanded:
                tree.expand(
                    leaf_id,
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.float64),
                )
        else:
            raise AssertionError(f"unexpected selected path: {action_path}")

        tree.backprop(node_path, value)

    actions, visits = tree.get_children_visits(root)
    actions_i = [int(x) for x in actions.tolist()]
    visits_i = [int(x) for x in visits.tolist()]
    assert actions_i == root_actions
    assert sum(visits_i) == SIMS

    best_idx = int(np.argmax(np.asarray(visits_i, dtype=np.int64)))
    expanded = sum(
        int(tree.is_expanded(int(tree.find_child(root, action))))
        for action in root_actions
    )

    visit_xor = 0
    for action, count in zip(actions_i, visits_i, strict=True):
        visit_xor ^= (
            ((action * 2654435761) & 0xFFFFFFFF)
            ^ count
        )

    root_w_from_tree = float(tree.node_q(root)) * SIMS
    assert root_w_from_tree == pytest.approx(root_w, abs=1e-12)

    return {
        "root_moves": len(root_actions),
        "root_n": SIMS,
        "root_w_q4": int(round((root_w + 64.0) * 4.0)),
        "best": actions_i[best_idx],
        "expanded": expanded,
        "visit_xor": visit_xor,
        "path_mix": path_mix,
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


def test_bend_two_ply_mcts_matches_production_c_tree(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["BEND_TREE_PROBE_BUILD_DIR"] = str(tmp_path / "build")

    built = _run_checked([str(BUILD_SCRIPT)], env=env)
    binary = Path(built.stdout.strip().splitlines()[-1])
    assert binary.is_file(), built.stdout

    run = _run_checked([str(binary)])
    observed = _parse(run.stdout)

    assert set(observed) == set(range(len(FIXTURE_FENS))), run.stdout
    for fixture, fen in enumerate(FIXTURE_FENS):
        assert observed[fixture] == _reference(fen)
