from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import CmdIsReady, CmdSetOption


class _FakePool:
    """Minimal stand-in for MultiGpuPucvPool exposing the budget knobs."""

    def __init__(self, *, n_devices: int, gather: int) -> None:
        self.n_devices = n_devices
        self.gather = gather


class _FakeWorker:
    """Records the ``max_nodes`` warmup_search asks for, without running."""

    def __init__(self, *, chunk_sims: int = 512, pucv_pool: Any | None = None) -> None:
        self._chunk_sims = chunk_sims
        self._pucv_pool = pucv_pool
        self.run_calls: list[dict[str, Any]] = []
        self.reset_calls = 0

    def set_max_tree_mb(self, _n: int) -> None:
        pass

    def run(self, *_args: Any, **kwargs: Any) -> None:
        self.run_calls.append(kwargs)

    def reset_tree(self) -> None:
        self.reset_calls += 1


def test_warmup_single_gpu_uses_one_chunk() -> None:
    worker = _FakeWorker(chunk_sims=512, pucv_pool=None)
    engine = Engine(worker=worker)  # pyright: ignore[reportArgumentType]  # test fake

    engine.warmup_search()

    assert worker.run_calls[-1]["max_nodes"] == 512
    assert worker.reset_calls == 1


def test_warmup_multi_gpu_pucv_covers_every_worker() -> None:
    # 3 devices, gather 256 < chunk 512 → per-device max(512, 256) = 512.
    pool = _FakePool(n_devices=3, gather=256)
    worker = _FakeWorker(chunk_sims=512, pucv_pool=pool)
    engine = Engine(worker=worker)  # pyright: ignore[reportArgumentType]  # test fake

    engine.warmup_search()

    assert worker.run_calls[-1]["max_nodes"] == 512 * 3


def test_warmup_multi_gpu_pucv_gather_exceeds_chunk() -> None:
    # gather 1024 > chunk 512 → per-device share must rise to the gather size.
    pool = _FakePool(n_devices=2, gather=1024)
    worker = _FakeWorker(chunk_sims=512, pucv_pool=pool)
    engine = Engine(worker=worker)  # pyright: ignore[reportArgumentType]  # test fake

    engine.warmup_search()

    assert worker.run_calls[-1]["max_nodes"] == 1024 * 2


def test_reshaping_setoption_marks_warmup_dirty_then_isready_rewarms() -> None:
    worker = MagicMock()
    engine = Engine(worker=worker)
    assert engine._warmup_dirty is False  # noqa: SLF001

    # Cosmetic option must NOT dirty the warmup.
    engine._handle_setoption(CmdSetOption(name="Ponder", value="true"))  # noqa: SLF001
    assert engine._warmup_dirty is False  # noqa: SLF001

    # A search-path reshape must dirty it.
    engine._handle_setoption(CmdSetOption(name="UseVL", value="true"))  # noqa: SLF001
    assert engine._warmup_dirty is True  # noqa: SLF001

    rewarms: list[int] = []
    engine.warmup_search = lambda: rewarms.append(1)  # type: ignore[method-assign]

    engine.dispatch(CmdIsReady())
    assert rewarms == [1]
    assert engine._warmup_dirty is False  # noqa: SLF001

    # Nothing changed since: a second isready must NOT re-warm.
    engine.dispatch(CmdIsReady())
    assert rewarms == [1]
