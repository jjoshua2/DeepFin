#!/usr/bin/env python3
"""Benchmark always-on versus final-step-only Aurora UW telemetry on CUDA."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import torch

from chess_anti_engine.train.aurora import AuroraWithAuxAdam


def _make_optimizer(tensors: list[torch.Tensor]) -> tuple[list[torch.nn.Parameter], AuroraWithAuxAdam]:
    params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    optimizer = AuroraWithAuxAdam(
        [{"params": params, "lr": 0.002, "weight_decay": 0.0, "use_aurora": True}],
        aurora_pp_iterations=3,
        aurora_pp_beta=0.25,
        aurora_polar_steps=8,
        aurora_polar_method="polar_express",
        aurora_polar_dtype="fp16",
    )
    return params, optimizer


def _run_path(
    name: str,
    params: list[torch.nn.Parameter],
    optimizer: AuroraWithAuxAdam,
    gradients: list[list[torch.Tensor]],
) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step_index, step_gradients in enumerate(gradients):
        optimizer.set_collect_uw_stats(name == "always" or step_index == len(gradients) - 1)
        for param, gradient in zip(params, step_gradients, strict=True):
            param.grad = gradient
        optimizer.step()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def _state_hash(params: list[torch.nn.Parameter], optimizer: AuroraWithAuxAdam) -> str:
    digest = hashlib.sha256()
    for param in params:
        digest.update(param.detach().cpu().numpy().tobytes())
        digest.update(optimizer.state[param]["momentum_buffer"].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--matrices", type=int, default=8)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=896)
    args = parser.parse_args()
    if min(args.rounds, args.steps, args.matrices, args.rows, args.cols) <= 0:
        raise SystemExit("all arguments must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260712)
    initial = [
        torch.randn(args.rows, args.cols, generator=generator, device=device)
        for _ in range(args.matrices)
    ]
    gradients = [
        [torch.randn_like(tensor, generator=generator) for tensor in initial]
        for _ in range(args.steps)
    ]

    warm_params, warm_optimizer = _make_optimizer(initial[:1])
    _run_path("always", warm_params, warm_optimizer, [[gradients[0][0]]])
    samples: dict[str, list[float]] = {"always": [], "final": []}
    final_hash = ""

    for round_index in range(args.rounds):
        runs = {name: _make_optimizer(initial) for name in samples}
        order = ("always", "final") if round_index % 2 == 0 else ("final", "always")
        row: list[str] = []
        for name in order:
            params, optimizer = runs[name]
            elapsed = _run_path(name, params, optimizer, gradients)
            samples[name].append(elapsed)
            row.append(f"{name}={elapsed:.6f}s")

        always_params, always_optimizer = runs["always"]
        final_params, final_optimizer = runs["final"]
        always_hash = _state_hash(always_params, always_optimizer)
        final_hash = _state_hash(final_params, final_optimizer)
        if final_hash != always_hash:
            raise AssertionError("parameter or momentum state diverged")
        if final_optimizer.last_uw_stats != always_optimizer.last_uw_stats:
            raise AssertionError("final UW statistics diverged")
        print(f"round {round_index + 1}: " + "  ".join(row) + f"  hash={final_hash[:16]}")

    always_median = statistics.median(samples["always"])
    final_median = statistics.median(samples["final"])
    print(f"always median: {always_median:.6f}s")
    print(f"final median:  {final_median:.6f}s")
    print(f"always/final throughput: {always_median / final_median:.6f}x")
    print(f"exact state hash: {final_hash}")
    print("final UW statistics: identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
