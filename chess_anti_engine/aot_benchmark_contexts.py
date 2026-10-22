"""Exclusive producer ownership for concurrent AOT benchmarks.

Each context owns a separately loaded evaluator (including constants and staging)
and one CUDA stream. This deliberately replicates device memory per producer.
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

_Result = TypeVar("_Result")


class ProducerContext:
    """One producer uses this context for both warmup and measurement."""

    def __init__(self, evaluator: Any, cuda: Any) -> None:
        self.evaluator = evaluator
        self._cuda = cuda
        self._stream = cuda.Stream(device="cuda")
        self._stream.wait_stream(cuda.current_stream())

    def run(self, callback: Callable[[Any], _Result]) -> _Result:
        with self._cuda.stream(self._stream):
            try:
                return callback(self.evaluator)
            finally:
                # Includes exceptional exits: do not dispose staging while queued
                # kernels may still read it. AOT launches use the current stream.
                self._stream.synchronize()
