from __future__ import annotations

import threading
import time
from typing import Any, ClassVar

import pytest

from chess_anti_engine.stockfish import pool as pool_module


class _BlockingEngine:
    instances: ClassVar[list[_BlockingEngine]] = []
    releases: ClassVar[list[threading.Event]] = [
        threading.Event(), threading.Event(),
    ]

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.index = len(self.instances)
        self.started = threading.Event()
        self.closed = False
        self.calls: list[str] = []
        self.instances.append(self)

    def search(self, fen: str, **_kwargs: Any) -> str:
        self.calls.append(fen)
        if fen == "block":
            self.started.set()
            self.releases[self.index].wait(timeout=2.0)
        return f"{self.index}:{fen}"

    def set_nodes(self, _nodes: int) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_pool_rejects_nonpositive_worker_count() -> None:
    with pytest.raises(ValueError, match="num_workers must be positive"):
        pool_module.StockfishPool(path="unused", nodes=1, num_workers=0)


def test_pool_closes_engines_after_partial_initialization(monkeypatch) -> None:
    class _FailsOnSecondEngine(_BlockingEngine):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            if self.instances:
                raise RuntimeError("second engine failed")
            super().__init__()

    _FailsOnSecondEngine.instances = []
    monkeypatch.setattr(pool_module, "StockfishUCI", _FailsOnSecondEngine)
    with pytest.raises(RuntimeError, match="second engine failed"):
        pool_module.StockfishPool(path="unused", nodes=1, num_workers=2)
    assert len(_FailsOnSecondEngine.instances) == 1
    assert _FailsOnSecondEngine.instances[0].closed


def test_pool_does_not_head_of_line_block_an_available_engine(monkeypatch) -> None:
    _BlockingEngine.instances = []
    _BlockingEngine.releases = [threading.Event(), threading.Event()]
    monkeypatch.setattr(pool_module, "StockfishUCI", _BlockingEngine)
    pool = pool_module.StockfishPool(path="unused", nodes=1, num_workers=2)
    try:
        first = pool.submit("block")
        second = pool.submit("block")
        assert all(
            engine.started.wait(timeout=1.0)
            for engine in _BlockingEngine.instances
        )

        queued_busy = pool.submit("queued-busy")  # engine 0
        available = pool.submit("available")  # engine 1
        _BlockingEngine.releases[1].set()

        # Engine 1 must run its own queued request even while engine 0 and its
        # second request remain blocked. A shared executor consumes the freed
        # thread with queued_busy and cannot reach available.
        assert available.result(timeout=0.25) == "1:available"
        assert not queued_busy.done()
        _BlockingEngine.releases[0].set()
        assert first.result(timeout=1.0) == "0:block"
        assert second.result(timeout=1.0) == "1:block"
        assert queued_busy.result(timeout=1.0) == "0:queued-busy"
    finally:
        for release in _BlockingEngine.releases:
            release.set()
        pool.close()
    assert all(engine.closed for engine in _BlockingEngine.instances)


def test_pool_close_cancels_queued_searches(monkeypatch) -> None:
    _BlockingEngine.instances = []
    _BlockingEngine.releases = [threading.Event(), threading.Event()]
    monkeypatch.setattr(pool_module, "StockfishUCI", _BlockingEngine)
    pool = pool_module.StockfishPool(path="unused", nodes=1, num_workers=2)
    running = [pool.submit("block"), pool.submit("block")]
    assert all(
        engine.started.wait(timeout=1.0)
        for engine in _BlockingEngine.instances
    )
    queued = [pool.submit("queued") for _ in range(2)]

    close_thread = threading.Thread(target=pool.close)
    close_thread.start()
    deadline = time.monotonic() + 1.0
    while not all(future.cancelled() for future in queued):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    for release in _BlockingEngine.releases:
        release.set()
    close_thread.join(timeout=1.0)

    assert not close_thread.is_alive()
    assert all(future.done() for future in running)
    assert all(future.cancelled() for future in queued)
    assert all(engine.calls == ["block"] for engine in _BlockingEngine.instances)
    assert all(engine.closed for engine in _BlockingEngine.instances)
