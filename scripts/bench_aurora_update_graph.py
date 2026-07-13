#!/usr/bin/env python3
"""Benchmark Polar-only versus whole-update Aurora CUDA graphs."""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch import Tensor

from chess_anti_engine.train.aurora import _aurora_update, _polar_express

_PRODUCTION_WIDTHS = (768, 796, 845, 907, 970, 853, 839, 976, 913, 919, 893, 768)


class _CapturedPolar:
    def __init__(
        self, sample: Tensor, *, steps: int, eps: float, safety: float,
        work_dtype: torch.dtype | None,
    ) -> None:
        self.static_input = torch.empty_like(sample)
        self.static_input.copy_(sample)
        for _ in range(3):
            _polar_express(
                self.static_input, steps=steps, eps=eps, safety=safety,
                work_dtype=work_dtype,
            )
        torch.cuda.synchronize(sample.device)
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


class _CapturedUpdate:
    def __init__(self, sample: Tensor) -> None:
        self.static_input = torch.empty_like(sample)
        self.static_input.copy_(sample)
        warmup_stream = torch.cuda.Stream(device=sample.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(sample.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._run(self.static_input)
        torch.cuda.current_stream(sample.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(sample.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self._run(self.static_input)

    @staticmethod
    def _run(update: Tensor) -> Tensor:
        return _aurora_update(
            update,
            pp_iterations=3,
            pp_beta=0.25,
            polar_steps=8,
            polar_method="polar_express",
            polar_dtype="fp16",
            check_finite=False,
        )

    def __call__(self, update: Tensor) -> Tensor:
        self.static_input.copy_(update)
        self.graph.replay()
        return self.static_output


def _sync() -> None:
    torch.cuda.synchronize()


def _time_path(function, updates: list[Tensor], *, repeats: int) -> float:
    _sync()
    started = time.perf_counter()
    for _ in range(repeats):
        for update in updates:
            function(update)
    _sync()
    return repeats * len(updates) / (time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--matrices", type=int, default=8)
    parser.add_argument("--rows", type=int, default=512)
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.matrices, args.rows) <= 0:
        raise SystemExit("all arguments must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(20260712)
    widths = [_PRODUCTION_WIDTHS[index % len(_PRODUCTION_WIDTHS)] for index in range(args.matrices)]
    updates = [
        torch.randn(args.rows, width, generator=generator, device="cuda")
        for width in widths
    ]
    polar_captures: dict[tuple[int, int], _CapturedPolar] = {}

    def graphed_polar(
        matrix: Tensor,
        *,
        steps: int,
        eps: float,
        safety: float,
        work_dtype: torch.dtype | None,
    ) -> Tensor:
        key = (matrix.size(0), matrix.size(1))
        captured = polar_captures.get(key)
        if captured is None:
            captured = _CapturedPolar(
                matrix, steps=steps, eps=eps, safety=safety, work_dtype=work_dtype,
            )
            polar_captures[key] = captured
        return captured(matrix)

    def polar_only(update: Tensor) -> Tensor:
        return _aurora_update(
            update,
            pp_iterations=3,
            pp_beta=0.25,
            polar_steps=8,
            polar_method="polar_express",
            polar_dtype="fp16",
            polar_express_fn=graphed_polar,
            check_finite=False,
        )

    polar_outputs = [polar_only(update).clone() for update in updates]
    _sync()
    memory_before = torch.cuda.memory_allocated()
    capture_started = time.perf_counter()
    captures = {
        (update.size(0), update.size(1)): _CapturedUpdate(update)
        for update in updates
    }
    _sync()
    capture_time = time.perf_counter() - capture_started
    capture_bytes = torch.cuda.memory_allocated() - memory_before

    def whole_update(update: Tensor) -> Tensor:
        return captures[(update.size(0), update.size(1))](update)

    whole_outputs = [whole_update(update).clone() for update in updates]
    _sync()
    for reference, candidate in zip(polar_outputs, whole_outputs, strict=True):
        torch.testing.assert_close(candidate, reference, rtol=0.0, atol=0.0)

    samples: dict[str, list[float]] = {"polar": [], "whole": []}
    for round_index in range(args.rounds):
        order = ("polar", "whole") if round_index % 2 == 0 else ("whole", "polar")
        row: list[str] = []
        for name in order:
            function = polar_only if name == "polar" else whole_update
            throughput = _time_path(function, updates, repeats=args.repeats)
            samples[name].append(throughput)
            row.append(f"{name}={throughput:.3f} updates/s")
        print(f"round {round_index + 1}: " + "  ".join(row))

    polar_median = statistics.median(samples["polar"])
    whole_median = statistics.median(samples["whole"])
    print(f"capture time: {capture_time:.3f}s")
    print(f"capture allocated bytes: {capture_bytes}")
    print(f"polar-only median: {polar_median:.3f} updates/s")
    print(f"whole-update median: {whole_median:.3f} updates/s")
    print(f"whole/polar-only: {whole_median / polar_median:.6f}x")
    print("update parity: bitwise exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
