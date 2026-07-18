#!/usr/bin/env python3
"""Compare staged and in-place pinned AOT inputs through Gumbel search."""
from __future__ import annotations

import argparse
import hashlib
import random
import statistics
import time

import chess
import numpy as np
import torch

from chess_anti_engine.inference import AOTEvaluator, model_constant_source
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.model import infer_input_planes
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


class _StagedAOT:
    """Expose AOT's staged contract without its in-place slot methods."""

    supports_input_bf16_bits = True

    def __init__(self, evaluator: AOTEvaluator) -> None:
        self._evaluator = evaluator

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._evaluator.evaluate_encoded(x, relations=relations)

    def evaluate_encoded_async(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        if relations is not None:
            raise NotImplementedError("AOT benchmark does not transport relations")
        return self._evaluator.evaluate_encoded_async(x)


def _boards(count: int) -> list[chess.Board]:
    boards: list[chess.Board] = []
    for seed in range(count):
        rng = random.Random(seed)
        board = chess.Board()
        for _ in range(6):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[rng.randrange(len(moves))])
        boards.append(board)
    return boards


def _checksum(result: tuple) -> str:
    probabilities, actions, values, root_qs, _tree, root_ids = result
    digest = hashlib.sha256()
    for array in probabilities:
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(np.asarray(actions, dtype=np.int32).tobytes())
    digest.update(np.asarray(values, dtype=np.float64).tobytes())
    for array in root_qs:
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(np.asarray(root_ids, dtype=np.int32).tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aot-dir", required=True)
    parser.add_argument("--games", type=int, default=96)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if min(args.games, args.simulations, args.iterations, args.rounds) <= 0:
        raise SystemExit("games, simulations, iterations, and rounds must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    model = load_model_from_checkpoint(args.checkpoint, device="cuda").eval()
    input_planes = infer_input_planes(getattr(model, "input_extra_features", None))
    evaluator = AOTEvaluator(
        args.aot_dir, device="cuda", max_batch=1024, input_planes=input_planes,
    )
    evaluator.load_weights(model_constant_source(model))
    del model
    torch.cuda.empty_cache()
    evaluators = {"staged": _StagedAOT(evaluator), "inplace": evaluator}
    cfg = GumbelConfig(simulations=args.simulations, topk=16, add_noise=False)
    boards = _boards(args.games)

    def run(mode: str, iterations: int) -> tuple:
        result: tuple | None = None
        for _ in range(iterations):
            result = run_gumbel_root_many_c(
                None,
                [board.copy(stack=True) for board in boards],
                device="cuda",
                rng=np.random.default_rng(20260718),
                cfg=cfg,
                evaluator=evaluators[mode],
                target_batch=512,
            )
        assert result is not None
        return result

    expected: dict[str, str] = {}
    for mode in ("staged", "inplace"):
        torch.cuda.synchronize()
        expected[mode] = _checksum(run(mode, args.warmup))

    samples: dict[str, list[float]] = {"staged": [], "inplace": []}
    for round_index in range(args.rounds):
        order = (
            ("staged", "inplace")
            if round_index % 2 == 0 else ("inplace", "staged")
        )
        for mode in order:
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = run(mode, args.iterations)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if _checksum(result) != expected[mode]:
                raise RuntimeError(f"{mode} search output changed between rounds")
            samples[mode].append(elapsed)
            print(f"round={round_index + 1} mode={mode} seconds={elapsed:.6f}")

    if expected["staged"] != expected["inplace"]:
        raise RuntimeError("staged and in-place search outputs differ")
    staged = statistics.median(samples["staged"])
    inplace = statistics.median(samples["inplace"])
    print()
    print(f"staged median:  {staged:.6f}s")
    print(f"inplace median: {inplace:.6f}s")
    print(f"inplace/staged: {inplace / staged:.6f}")
    print(f"speedup:        {(staged / inplace - 1.0) * 100.0:.3f}%")
    print(f"checksum:       {expected['inplace']}")


if __name__ == "__main__":
    main()
