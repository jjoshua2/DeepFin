#!/usr/bin/env python3
"""Compare float32 versus bit-packed BF16 AOT evaluator input transport."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import numpy as np
import torch

from chess_anti_engine.inference import (
    AOTEvaluator,
    model_constant_source,
)
from chess_anti_engine.model import infer_input_planes
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


def _checksum(policy: np.ndarray, wdl: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(policy).tobytes())
    digest.update(np.ascontiguousarray(wdl).tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aot-dir", required=True)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch, args.iterations, args.rounds) <= 0:
        raise SystemExit("batch, iterations, and rounds must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    model = load_model_from_checkpoint(args.checkpoint, device="cuda").eval()
    input_planes = infer_input_planes(getattr(model, "input_extra_features", None))
    evaluator = AOTEvaluator(
        args.aot_dir, device="cuda", max_batch=args.batch, input_planes=input_planes,
    )
    evaluator.load_weights(model_constant_source(model))
    del model
    torch.cuda.empty_cache()

    shape = (args.batch, input_planes, 8, 8)
    rng = np.random.default_rng(20260718)
    x_f32 = rng.standard_normal(shape, dtype=np.float32)
    x_bits = torch.from_numpy(x_f32).to(torch.bfloat16).view(torch.uint16).numpy().copy()
    def evaluate_f32() -> tuple[np.ndarray, np.ndarray]:
        return evaluator.evaluate_encoded(x_f32)

    def evaluate_bf16() -> tuple[np.ndarray, np.ndarray]:
        return evaluator.evaluate_encoded(x_bits)

    functions = {"f32": evaluate_f32, "bf16": evaluate_bf16}
    expected: dict[str, str] = {}
    for name, function in functions.items():
        result: tuple[np.ndarray, np.ndarray] | None = None
        for _ in range(args.warmup):
            result = function()
        assert result is not None
        expected[name] = _checksum(*result)

    samples: dict[str, list[float]] = {name: [] for name in functions}
    for round_index in range(args.rounds):
        order = ("f32", "bf16") if round_index % 2 == 0 else ("bf16", "f32")
        for name in order:
            function = functions[name]
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = None
            for _ in range(args.iterations):
                result = function()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert result is not None
            checksum = _checksum(*result)
            if checksum != expected[name]:
                raise RuntimeError(f"{name} output changed between rounds")
            samples[name].append(elapsed)
            print(f"round={round_index + 1} mode={name} seconds={elapsed:.6f}")

    if expected["f32"] != expected["bf16"]:
        raise RuntimeError("float32 and bit-packed BF16 outputs differ")
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
