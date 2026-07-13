#!/usr/bin/env python3
"""Benchmark eager Aurora Polar Express against exact CUDA-graph replay."""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch import Tensor

from chess_anti_engine.train import aurora
from chess_anti_engine.train.aurora import _polar_express

_PRODUCTION_WIDTHS = (768, 796, 845, 907, 970, 853, 839, 976, 913, 919, 893, 768)


class _CapturedPolar:
    def __init__(
        self,
        sample: Tensor,
        *,
        steps: int,
        eps: float = 1e-7,
        safety: float = 1.01,
        work_dtype: torch.dtype | None = torch.float16,
    ) -> None:
        self.static_input = torch.empty_like(sample)
        self.static_input.copy_(sample)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _polar_express(
                    self.static_input, steps=steps, eps=eps, safety=safety,
                    work_dtype=work_dtype,
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = _polar_express(
                self.static_input, steps=steps, eps=eps, safety=safety,
                work_dtype=work_dtype,
            )

    def __call__(self, matrix: Tensor) -> Tensor:
        self.static_input.copy_(matrix)
        self.graph.replay()
        return self.static_output


def _sync() -> None:
    torch.cuda.synchronize()


@torch.inference_mode()
def _time_eager(matrices: list[Tensor], *, repeats: int, steps: int) -> float:
    _sync()
    started = time.perf_counter()
    for _ in range(repeats):
        for matrix in matrices:
            _polar_express(matrix, steps=steps, work_dtype=torch.float16)
    _sync()
    return repeats * len(matrices) / (time.perf_counter() - started)


@torch.inference_mode()
def _time_graph(
    matrices: list[Tensor], captures: dict[tuple[int, int], _CapturedPolar], *, repeats: int,
) -> float:
    _sync()
    started = time.perf_counter()
    for _ in range(repeats):
        for matrix in matrices:
            captures[(matrix.size(0), matrix.size(1))](matrix)
    _sync()
    return repeats * len(matrices) / (time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--matrices", type=int, default=8)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.matrices, args.rows, args.steps) <= 0:
        raise SystemExit("all arguments must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(20260712)
    widths = [_PRODUCTION_WIDTHS[index % len(_PRODUCTION_WIDTHS)] for index in range(args.matrices)]
    matrices = [
        torch.randn(args.rows, width, generator=generator, device="cuda")
        for width in widths
    ]
    _sync()
    memory_before = torch.cuda.memory_allocated()
    capture_started = time.perf_counter()
    captures = {
        (matrix.size(0), matrix.size(1)): _CapturedPolar(matrix, steps=args.steps)
        for matrix in matrices
    }
    _sync()
    capture_time = time.perf_counter() - capture_started
    capture_bytes = torch.cuda.memory_allocated() - memory_before

    eager_outputs = [
        _polar_express(matrix, steps=args.steps, work_dtype=torch.float16)
        for matrix in matrices
    ]
    graph_outputs = [
        captures[(matrix.size(0), matrix.size(1))](matrix).clone()
        for matrix in matrices
    ]
    _sync()
    for eager, candidate in zip(eager_outputs, graph_outputs, strict=True):
        torch.testing.assert_close(candidate, eager, rtol=0.0, atol=0.0)

    full_updates = [matrix.clone() for matrix in matrices]
    eager_full = [
        aurora._aurora_update(
            update, pp_iterations=3, pp_beta=0.25, polar_steps=args.steps,
            polar_method="polar_express", polar_dtype="fp16",
        )
        for update in full_updates
    ]
    update_captures: dict[tuple[int, int], _CapturedPolar] = {}

    def graphed_polar(
        matrix: Tensor,
        *,
        steps: int,
        eps: float,
        safety: float,
        work_dtype: torch.dtype | None,
    ) -> Tensor:
        key = (matrix.size(0), matrix.size(1))
        captured = update_captures.get(key)
        if captured is None:
            captured = _CapturedPolar(
                matrix, steps=steps, eps=eps, safety=safety, work_dtype=work_dtype,
            )
            update_captures[key] = captured
        return captured(matrix)

    original_polar = aurora._polar_express
    try:
        aurora._polar_express = graphed_polar
        graph_full = [
            aurora._aurora_update(
                update, pp_iterations=3, pp_beta=0.25, polar_steps=args.steps,
                polar_method="polar_express", polar_dtype="fp16",
            ).clone()
            for update in full_updates
        ]
    finally:
        aurora._polar_express = original_polar
    _sync()
    for eager, candidate in zip(eager_full, graph_full, strict=True):
        torch.testing.assert_close(candidate, eager, rtol=0.0, atol=0.0)

    samples: dict[str, list[float]] = {"eager": [], "graph": []}
    for round_index in range(args.rounds):
        order = ("eager", "graph") if round_index % 2 == 0 else ("graph", "eager")
        row: list[str] = []
        for name in order:
            throughput = (
                _time_eager(matrices, repeats=args.repeats, steps=args.steps)
                if name == "eager"
                else _time_graph(matrices, captures, repeats=args.repeats)
            )
            samples[name].append(throughput)
            row.append(f"{name}={throughput:.3f} matrices/s")
        print(f"round {round_index + 1}: " + "  ".join(row))

    eager_median = statistics.median(samples["eager"])
    graph_median = statistics.median(samples["graph"])
    print(f"capture time: {capture_time:.3f}s")
    print(f"capture allocated bytes: {capture_bytes}")
    print(f"unique captured shapes: {len(captures)}")
    print(f"eager median: {eager_median:.3f} matrices/s")
    print(f"graph median: {graph_median:.3f} matrices/s")
    print(f"graph/eager: {graph_median / eager_median:.6f}x")
    print("output parity: bitwise exact")
    print("full Aurora update parity: bitwise exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
