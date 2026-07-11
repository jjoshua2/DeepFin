#!/usr/bin/env python3
"""Rebuild and benchmark native extensions under controlled compiler modes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD_MODES: dict[str, dict[str, str]] = {
    "portable": {},
    "native": {"CAE_EXT_NATIVE": "1"},
    "native-lto": {
        "CAE_EXT_NATIVE": "1",
        "CAE_EXT_LTO": "1",
    },
    "native-lto-pgo": {
        "CAE_EXT_NATIVE": "1",
        "CAE_EXT_LTO": "1",
    },
    "gcc11-native-lto": {
        "CC": "/usr/bin/gcc",
        "CAE_EXT_NATIVE": "1",
        "CAE_EXT_LTO": "1",
    },
    "gcc15-native-lto": {
        "CC": os.environ.get(
            "CAE_GCC15_CC",
            str(Path.home() / ".local/gcc-15.3/bin/gcc"),
        ),
        "CAE_EXT_NATIVE": "1",
        "CAE_EXT_LTO": "1",
    },
}


def _median_call_seconds(fn: Callable[[], None], iterations: int, samples: int) -> float:
    fn()
    elapsed: list[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        elapsed.append((time.perf_counter() - start) / iterations)
    return statistics.median(elapsed)


def _measure_current(samples: int, iterations: int, cpu: int | None) -> dict[str, Any]:
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})

    import chess
    import numpy as np

    from chess_anti_engine.encoding._features_ext import compute_extra_features
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.mcts._mcts_tree import (
        MCTSTree,
        batch_encode_146,
        batch_encode_146_bf16,
    )

    rng = np.random.default_rng(20260709)
    boards: list[CBoard] = []
    for index in range(256):
        board = chess.Board()
        for _ in range(index % 48):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(moves[int(rng.integers(len(moves)))])
        boards.append(CBoard.from_board(board))

    out_f32 = np.empty((len(boards), 175, 8, 8), dtype=np.float32)
    out_bf16 = np.empty((len(boards), 175, 8, 8), dtype=np.uint16)
    tree = MCTSTree()
    wdl = np.tile(np.array([[2.0, 0.5, -1.0]], dtype=np.float32), (65_536, 1))

    def encode_f32() -> None:
        batch_encode_146(boards, out_f32)

    def encode_bf16() -> None:
        batch_encode_146_bf16(boards, out_bf16)

    def generate_moves() -> None:
        for board in boards:
            board.legal_move_indices()

    def wdl_to_q() -> None:
        tree.batch_wdl_to_q(wdl)

    # Import and execute the standalone feature extension too: native builds
    # historically missed its AVX2 helper dependency and failed at import.
    pieces = np.zeros(6, dtype=np.uint64)
    compute_extra_features(pieces, pieces, 0, -1, -1, True, -1)

    f32_seconds = _median_call_seconds(encode_f32, iterations, samples)
    bf16_seconds = _median_call_seconds(encode_bf16, iterations, samples)
    movegen_seconds = _median_call_seconds(generate_moves, iterations, samples)
    wdl_seconds = _median_call_seconds(wdl_to_q, iterations, samples)
    encode_f32()
    encode_bf16()
    return {
        "f32_positions_per_second": len(boards) / f32_seconds,
        "bf16_positions_per_second": len(boards) / bf16_seconds,
        "movegen_positions_per_second": len(boards) / movegen_seconds,
        "wdl_rows_per_second": len(wdl) / wdl_seconds,
        "f32_sha256": hashlib.sha256(out_f32.tobytes()).hexdigest(),
        "bf16_sha256": hashlib.sha256(out_bf16.tobytes()).hexdigest(),
    }


def _mode_environment(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "CAE_EXT_NATIVE", "CAE_EXT_LTO", "CAE_EXT_SANITIZE",
        "CC", "CFLAGS", "LDFLAGS",
    ):
        env.pop(name, None)
    env.update(BUILD_MODES[mode])
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = str(ROOT)
    return env


def _build_and_measure(
    mode: str, samples: int, iterations: int, cpu: int | None,
) -> dict[str, Any]:
    env = _mode_environment(mode)
    start = time.perf_counter()
    if mode == "native-lto-pgo":
        subprocess.run(
            [
                sys.executable, "scripts/build_native_pgo.py",
                "--training-iterations", str(iterations),
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            check=True,
        )
    build_seconds = time.perf_counter() - start
    measure_cmd = [
        sys.executable, __file__, "--measure-current",
        "--samples", str(samples), "--iterations", str(iterations),
    ]
    if cpu is not None:
        measure_cmd.extend(("--cpu", str(cpu)))
    raw = subprocess.check_output(
        measure_cmd,
        cwd=ROOT,
        env=env,
        text=True,
    )
    measured = json.loads(raw)
    measured["build_seconds"] = build_seconds
    return measured


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=tuple(BUILD_MODES), default=["native", "native-lto"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--measure-current", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.rounds <= 0 or args.samples <= 0 or args.iterations <= 0:
        raise SystemExit("--rounds, --samples, and --iterations must be positive")
    if args.measure_current:
        print(json.dumps(
            _measure_current(args.samples, args.iterations, args.cpu),
            sort_keys=True,
        ))
        return

    rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in args.modes}
    for round_index in range(args.rounds):
        order = args.modes if round_index % 2 == 0 else list(reversed(args.modes))
        for mode in order:
            rows[mode].append(_build_and_measure(
                mode, args.samples, args.iterations, args.cpu,
            ))

    hashes = {
        key: {str(row[key]) for mode_rows in rows.values() for row in mode_rows}
        for key in ("f32_sha256", "bf16_sha256")
    }
    if any(len(values) != 1 for values in hashes.values()):
        raise SystemExit(f"native build modes produced different outputs: {hashes}")

    summary: dict[str, dict[str, Any]] = {
        mode: {
            "median_build_seconds": _median(mode_rows, "build_seconds"),
            "median_f32_positions_per_second": _median(mode_rows, "f32_positions_per_second"),
            "median_bf16_positions_per_second": _median(mode_rows, "bf16_positions_per_second"),
            "median_movegen_positions_per_second": _median(
                mode_rows, "movegen_positions_per_second",
            ),
            "median_wdl_rows_per_second": _median(mode_rows, "wdl_rows_per_second"),
            "rounds": mode_rows,
        }
        for mode, mode_rows in rows.items()
    }
    if len(args.modes) == 2:
        baseline_name, candidate_name = args.modes
        baseline = summary[baseline_name]
        candidate = summary[candidate_name]

        def ratio(key: str) -> float:
            return float(candidate[key]) / float(baseline[key])

        bf16_ratio = ratio("median_bf16_positions_per_second")
        movegen_ratio = ratio("median_movegen_positions_per_second")
        wdl_ratio = ratio("median_wdl_rows_per_second")
        summary[f"{candidate_name}_vs_{baseline_name}"] = {
            "build_time_ratio": ratio("median_build_seconds"),
            "f32_throughput_ratio": ratio("median_f32_positions_per_second"),
            "bf16_throughput_ratio": bf16_ratio,
            "movegen_throughput_ratio": movegen_ratio,
            "wdl_throughput_ratio": wdl_ratio,
            "production_geomean_ratio": (
                bf16_ratio * movegen_ratio * wdl_ratio
            ) ** (1.0 / 3.0),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
