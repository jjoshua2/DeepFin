#!/usr/bin/env python3
"""Compare pageable versus persistent-pinned AOT evaluator input staging."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import numpy as np
import torch

from chess_anti_engine.inference import AOTEvaluator, model_constant_source
from chess_anti_engine.model import infer_input_planes
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


def _checksum(policy: np.ndarray, wdl: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(policy).tobytes())
    digest.update(np.ascontiguousarray(wdl).tobytes())
    return digest.hexdigest()


def _timed_round(
    evaluator: AOTEvaluator,
    x: np.ndarray,
    *,
    iterations: int,
) -> tuple[float, str]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    policy: np.ndarray | None = None
    wdl: np.ndarray | None = None
    for _ in range(iterations):
        policy, wdl = evaluator.evaluate_encoded(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert policy is not None
    assert wdl is not None
    return elapsed, _checksum(policy, wdl)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aot-dir", required=True)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.batch <= 0 or args.iterations <= 0 or args.rounds <= 0:
        raise SystemExit("batch, iterations, and rounds must be positive")

    model = load_model_from_checkpoint(args.checkpoint, device="cuda").eval()
    input_extra_features = getattr(model, "input_extra_features", None)
    input_planes = infer_input_planes(input_extra_features)
    evaluator = AOTEvaluator(
        args.aot_dir,
        device="cuda",
        max_batch=args.batch,
        input_planes=input_planes,
    )
    evaluator.load_weights(model_constant_source(model))
    del model
    torch.cuda.empty_cache()

    shape = (args.batch, input_planes, 8, 8)
    rng = np.random.default_rng(20260718)
    x = rng.standard_normal(shape, dtype=np.float32)
    pageable = np.empty(shape, dtype=np.float32)
    pinned_tensor = torch.empty(shape, dtype=torch.float32, pin_memory=True)
    pinned = pinned_tensor.numpy()

    # Keep both owners live for the entire benchmark. The evaluator's methods
    # consume only the NumPy view; the tensor owner is what pins that storage.
    original_tensor = evaluator._pinned_input

    def select(mode: str) -> None:
        if mode == "pinned":
            evaluator._pinned_input = pinned_tensor
            evaluator._pinned_input_np = pinned
            evaluator._pinned_inputs[0] = pinned_tensor
            evaluator._pinned_inputs_np[0] = pinned
        else:
            evaluator._pinned_input = original_tensor
            evaluator._pinned_input_np = pageable
            evaluator._pinned_inputs[0] = original_tensor
            evaluator._pinned_inputs_np[0] = pageable

    checksums: dict[str, str] = {}
    for mode in ("pageable", "pinned"):
        select(mode)
        policy: np.ndarray | None = None
        wdl: np.ndarray | None = None
        for _ in range(args.warmup):
            policy, wdl = evaluator.evaluate_encoded(x)
        assert policy is not None
        assert wdl is not None
        checksums[mode] = _checksum(policy, wdl)

    samples: dict[str, list[float]] = {"pageable": [], "pinned": []}
    for round_idx in range(args.rounds):
        order = ("pageable", "pinned") if round_idx % 2 == 0 else ("pinned", "pageable")
        for mode in order:
            select(mode)
            elapsed, checksum = _timed_round(evaluator, x, iterations=args.iterations)
            if checksum != checksums[mode]:
                raise RuntimeError(f"{mode} output changed between rounds")
            samples[mode].append(elapsed)
            print(f"round={round_idx + 1} mode={mode} seconds={elapsed:.6f}")

    if checksums["pageable"] != checksums["pinned"]:
        raise RuntimeError("pageable and pinned outputs differ")
    pageable_median = statistics.median(samples["pageable"])
    pinned_median = statistics.median(samples["pinned"])
    ratio = pinned_median / pageable_median
    print()
    print(f"pageable median: {pageable_median:.6f}s")
    print(f"pinned median:   {pinned_median:.6f}s")
    print(f"pinned/pageable: {ratio:.6f}")
    print(f"speedup:         {(pageable_median / pinned_median - 1.0) * 100.0:.3f}%")
    print(f"checksum:        {checksums['pinned']}")


if __name__ == "__main__":
    main()
