#!/usr/bin/env python3
"""Benchmark complete eager versus CUDA-graphed Aurora optimizer steps."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import torch

from chess_anti_engine.train.aurora import AuroraWithAuxAdam

_PRODUCTION_WIDTHS = (768, 796, 845, 907, 970, 853, 839, 976, 913, 919, 893, 768)


def _make_optimizer(
    tensors: list[torch.Tensor], *, use_graphs: bool,
) -> tuple[list[torch.nn.Parameter], AuroraWithAuxAdam]:
    params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    optimizer = AuroraWithAuxAdam(
        [{"params": params, "lr": 0.002, "weight_decay": 0.0, "use_aurora": True}],
        aurora_pp_iterations=3,
        aurora_pp_beta=0.25,
        aurora_polar_steps=8,
        aurora_polar_method="polar_express",
        aurora_polar_dtype="fp16",
        aurora_cuda_graphs=use_graphs,
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
    samples: dict[str, list[float]] = {"eager": [], "graph": []}
    final_hash = ""

    for round_index in range(args.rounds):
        runs = {
            "eager": _make_optimizer(initial, use_graphs=False),
            "graph": _make_optimizer(initial, use_graphs=True),
        }
        graph_params, graph_optimizer = runs["graph"]
        warm_gradients = [[gradient.clone() for gradient in gradients[0]]]
        _run_steps(graph_params, graph_optimizer, warm_gradients)
        for param, tensor in zip(graph_params, initial, strict=True):
            param.data.copy_(tensor)
            param.grad = None
        graph_optimizer.state.clear()

        order = ("eager", "graph") if round_index % 2 == 0 else ("graph", "eager")
        row: list[str] = []
        for name in order:
            params, optimizer = runs[name]
            elapsed = _run_steps(params, optimizer, gradients)
            samples[name].append(elapsed)
            row.append(f"{name}={elapsed:.6f}s")

        eager_params, eager_optimizer = runs["eager"]
        eager_hash = _state_hash(eager_params, eager_optimizer)
        final_hash = _state_hash(graph_params, graph_optimizer)
        if final_hash != eager_hash:
            raise AssertionError("parameter or momentum state diverged")
        print(f"round {round_index + 1}: " + "  ".join(row) + f"  hash={final_hash[:16]}")

    eager_median = statistics.median(samples["eager"])
    graph_median = statistics.median(samples["graph"])
    print(f"eager median: {eager_median:.6f}s")
    print(f"graph median: {graph_median:.6f}s")
    print(f"eager/graph throughput: {eager_median / graph_median:.6f}x")
    print(f"exact state hash: {final_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
