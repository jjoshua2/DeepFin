#!/usr/bin/env python3
"""Compare float32 and BF16-bit AOT inputs through complete Gumbel search."""
from __future__ import annotations

import argparse
import hashlib
import random
import statistics
import time
from typing import Any

import chess
import numpy as np
import torch

from chess_anti_engine.inference import AOTEvaluator, model_constant_source
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.model import infer_input_planes
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


class _InputModeEvaluator:
    def __init__(self, evaluator: AOTEvaluator, *, bf16: bool) -> None:
        self._evaluator = evaluator
        self.supports_input_bf16_bits = bool(bf16)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._evaluator, name)

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
    evaluators = {
        "f32": _InputModeEvaluator(evaluator, bf16=False),
        "bf16": _InputModeEvaluator(evaluator, bf16=True),
    }
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
    for mode in ("f32", "bf16"):
        torch.cuda.synchronize()
        expected[mode] = _checksum(run(mode, args.warmup))

    samples: dict[str, list[float]] = {"f32": [], "bf16": []}
    for round_index in range(args.rounds):
        order = ("f32", "bf16") if round_index % 2 == 0 else ("bf16", "f32")
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

    if expected["f32"] != expected["bf16"]:
        raise RuntimeError("float32 and BF16 search outputs differ")
    f32_median = statistics.median(samples["f32"])
    bf16_median = statistics.median(samples["bf16"])
    print()
    print(f"f32 median:  {f32_median:.6f}s")
    print(f"bf16 median: {bf16_median:.6f}s")
    print(f"bf16/f32:    {bf16_median / f32_median:.6f}")
    print(f"speedup:     {(f32_median / bf16_median - 1.0) * 100.0:.3f}%")
    print(f"checksum:    {expected['bf16']}")


if __name__ == "__main__":
    main()
