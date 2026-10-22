from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, local

import pytest

from chess_anti_engine.aot_benchmark_contexts import ProducerContext


class FakeStream:
    def __init__(self, device: str) -> None:
        self.device = device
        self.drains = 0
        self.waited_for = None

    def wait_stream(self, other: object) -> None:
        self.waited_for = other

    def synchronize(self) -> None:
        self.drains += 1


class FakeCuda:
    Stream = FakeStream

    def __init__(self) -> None:
        self.default = object()
        self.active = local()

    def current_stream(self) -> object:
        return getattr(self.active, "stream", self.default)

    @contextmanager
    def stream(self, stream: FakeStream):
        previous = self.current_stream()
        self.active.stream = stream
        try:
            yield
        finally:
            self.active.stream = previous


def test_producers_keep_distinct_staging_and_streams_across_both_phases() -> None:
    cuda = FakeCuda()
    contexts = [ProducerContext({"staging": None}, cuda) for _ in range(2)]
    barrier = Barrier(2, timeout=2)
    observations = []

    def phase(tid: int, offset: int):
        def submit(evaluator):
            evaluator["staging"] = tid + offset
            barrier.wait()
            return evaluator["staging"], cuda.current_stream(), id(evaluator)
        return contexts[tid].run(submit)

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Warmup and measurement use the same owners, in sequence.
        observations = [
            list(pool.map(lambda tid, offset=offset: phase(tid, offset), range(2)))
            for offset in (10, 20)
        ]
    for offset, results in zip((10, 20), observations, strict=True):
        assert [result[0] for result in results] == [offset, offset + 1]
        assert results[0][1] is not results[1][1]
        assert results[0][2] != results[1][2]
    for tid, context in enumerate(contexts):
        assert observations[0][tid][1:] == observations[1][tid][1:]
        assert context._stream.drains == 2
        assert context._stream.waited_for is cuda.default


def test_exception_drains_before_context_can_be_disposed() -> None:
    cuda = FakeCuda()
    context = ProducerContext(object(), cuda)

    def fail(_evaluator):
        assert cuda.current_stream() is context._stream
        raise RuntimeError("producer failed after enqueue")

    with pytest.raises(RuntimeError, match="after enqueue"):
        context.run(fail)
    assert context._stream.drains == 1
    assert cuda.current_stream() is cuda.default
