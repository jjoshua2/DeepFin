"""Benchmark pipelined vs sequential MCTS simulation.

Runs gumbel MCTS on N boards with a real model, comparing:
  1. Sequential (original path): GPU eval → C tree walks → GPU eval → ...
  2. Pipelined (new path): GPU(A) overlaps C(B), GPU(B) overlaps C(A)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.getcwd())

import chess
import numpy as np
import torch

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.model.tiny import TinyNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BOARDS = 256
SIMULATIONS = 64


def make_boards(n: int, rng: np.random.Generator):
    boards = []
    cboards = []
    for _ in range(n):
        b = chess.Board()
        for __ in range(rng.integers(0, 20)):
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(rng.choice(legal))
            if b.is_game_over():
                break
        if b.is_game_over():
            b = chess.Board()
        boards.append(b)
        cboards.append(cboard_from_board_fast(b))
    return boards, cboards


def bench(label: str, model, evaluator, boards, cboards, rng, simulations, pipeline: bool):
    cfg = GumbelConfig(simulations=simulations, temperature=1.0, add_noise=True)

    # Warm up
    run_gumbel_root_many_c(
        model, boards[:16], device=DEVICE, rng=rng, cfg=cfg,
        evaluator=evaluator, cboards=cboards[:16],
    )

    times = []
    for trial in range(5):
        t0 = time.perf_counter()
        run_gumbel_root_many_c(
            model, boards, device=DEVICE, rng=rng, cfg=cfg,
            evaluator=evaluator, cboards=[cboard_from_board_fast(b) for b in boards],
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  {label} trial {trial}: {elapsed:.3f}s")

    med = sorted(times)[len(times) // 2]
    print(f"  {label} median: {med:.3f}s ({N_BOARDS} boards, {simulations} sims)")
    return med


def main():
    rng = np.random.default_rng(42)
    print(f"Device: {DEVICE}, boards: {N_BOARDS}, sims: {SIMULATIONS}")

    model = TinyNet(in_planes=146).to(DEVICE).eval()
    if DEVICE == "cuda":
        model = torch.compile(model)

    evaluator = DirectGPUEvaluator(model, device=DEVICE, max_batch=4096)
    boards, cboards = make_boards(N_BOARDS, rng)

    # The pipeline flag is controlled by _has_async and n_boards >= 16.
    # With DirectGPUEvaluator (has evaluate_encoded_async) and 256 boards,
    # it should use the pipeline path.
    # To test sequential, we'd need to disable async — let's just run both
    # and compare timing.

    print("\n--- Pipeline (default with async evaluator) ---")
    t_pipe = bench("pipeline", model, evaluator, boards, cboards, rng, SIMULATIONS, True)

    # Force sequential by using an evaluator without evaluate_encoded_async
    class SyncEval:
        def evaluate_encoded(self, x):
            return evaluator.evaluate_encoded(x)

    print("\n--- Sequential (sync evaluator) ---")
    t_seq = bench("sequential", model, SyncEval(), boards, cboards, rng, SIMULATIONS, False)

    print(f"\nSpeedup: {t_seq / t_pipe:.2f}x ({t_seq:.3f}s → {t_pipe:.3f}s)")


if __name__ == "__main__":
    main()
