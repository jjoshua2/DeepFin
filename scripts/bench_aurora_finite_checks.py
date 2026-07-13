#!/usr/bin/env python3
"""Benchmark per-matrix versus coalesced Aurora finite checks."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import torch

from chess_anti_engine.train.aurora import AuroraWithAuxAdam

_PRODUCTION_WIDTHS = (768, 796, 845, 907, 970, 853, 839, 976, 913, 919, 893, 768)


def _make_optimizer(
    tensors: list[torch.Tensor], *, coalesce: bool,
) -> tuple[list[torch.nn.Parameter], AuroraWithAuxAdam]:
    params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    optimizer = AuroraWithAuxAdam(
        [{"params": params, "lr": 0.002, "weight_decay": 0.0, "use_aurora": True}],
        aurora_pp_iterations=3,
        aurora_pp_beta=0.25,
        aurora_polar_steps=8,
        aurora_polar_method="polar_express",
        aurora_polar_dtype="fp16",
        aurora_cuda_graphs=True,
        aurora_coalesce_finite_checks=coalesce,
    )
    optimizer.set_collect_uw_stats(False)
    return params, optimizer


def _run_steps(
    params: list[torch.nn.Parameter],
    optimizer: AuroraWithAuxAdam,
    gradients: list[list[torch.Tensor]],
) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step_gradients in gradients:
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


def _warm_and_reset(
    params: list[torch.nn.Parameter],
    optimizer: AuroraWithAuxAdam,
    initial: list[torch.Tensor],
    gradients: list[list[torch.Tensor]],
) -> None:
    _run_steps(params, optimizer, gradients[:1])
    for param, tensor in zip(params, initial, strict=True):
        param.data.copy_(tensor)
        param.grad = None
    optimizer.state.clear()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--matrices", type=int, default=8)
    parser.add_argument("--rows", type=int, default=512)
    args = parser.parse_args()
    if min(args.rounds, args.steps, args.matrices, args.rows) <= 0:
        raise SystemExit("all arguments must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(20260712)
    widths = [_PRODUCTION_WIDTHS[index % len(_PRODUCTION_WIDTHS)] for index in range(args.matrices)]
    initial = [
        torch.randn(args.rows, width, generator=generator, device="cuda")
        for width in widths
    ]
    gradients = [
        [torch.randn_like(tensor, generator=generator) for tensor in initial]
        for _ in range(args.steps)
    ]
    samples: dict[str, list[float]] = {"per_matrix": [], "coalesced": []}
    final_hash = ""

    for round_index in range(args.rounds):
        runs = {
            "per_matrix": _make_optimizer(initial, coalesce=False),
            "coalesced": _make_optimizer(initial, coalesce=True),
        }
        for params, optimizer in runs.values():
            _warm_and_reset(params, optimizer, initial, gradients)

        order = (
            ("per_matrix", "coalesced")
            if round_index % 2 == 0 else ("coalesced", "per_matrix")
        )
        row: list[str] = []
        for name in order:
            params, optimizer = runs[name]
            elapsed = _run_steps(params, optimizer, gradients)
            samples[name].append(elapsed)
            row.append(f"{name}={elapsed:.6f}s")

        reference_params, reference_optimizer = runs["per_matrix"]
        reference_hash = _state_hash(reference_params, reference_optimizer)
        candidate_params, candidate_optimizer = runs["coalesced"]
        final_hash = _state_hash(candidate_params, candidate_optimizer)
        if final_hash != reference_hash:
            raise AssertionError("parameter or momentum state diverged")
        print(f"round {round_index + 1}: " + "  ".join(row) + f"  hash={final_hash[:16]}")

    reference_median = statistics.median(samples["per_matrix"])
    candidate_median = statistics.median(samples["coalesced"])
    print(f"per-matrix median: {reference_median:.6f}s")
    print(f"coalesced median:  {candidate_median:.6f}s")
    print(f"per-matrix/coalesced throughput: {reference_median / candidate_median:.6f}x")
    print(f"exact state hash: {final_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
