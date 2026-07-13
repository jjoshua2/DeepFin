from __future__ import annotations

import itertools

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress

from .uci import StockfishResult, StockfishUCI


class StockfishPool:
    """Thread-based pool of independent StockfishUCI processes.

    Each worker owns its own Stockfish process (UCI is not designed for
    concurrent requests over a single process).
    """

    def __init__(
        self,
        *,
        path: str,
        nodes: int,
        num_workers: int,
        multipv: int = 1,
        hash_mb: int | None = None,
        syzygy_path: str | None = None,
        nice: int = 0,
    ):
        self.path = path
        self.nodes = int(nodes)
        self.num_workers = int(num_workers)
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {num_workers!r}")
        self.multipv = int(multipv)
        self.hash_mb = None if hash_mb is None else max(1, int(hash_mb))
        self.syzygy_path = syzygy_path or None
        self.nice = min(19, max(0, int(nice)))

        self._execs: list[ThreadPoolExecutor] = []
        self._engines: list[StockfishUCI] = []
        try:
            for _ in range(self.num_workers):
                self._engines.append(StockfishUCI(
                    self.path,
                    nodes=self.nodes,
                    multipv=self.multipv,
                    hash_mb=self.hash_mb,
                    syzygy_path=self.syzygy_path,
                    nice=self.nice,
                ))
            # Keep each engine's queue on its own thread. Binding requests to
            # an engine before putting them in one shared executor lets a
            # worker thread block on that engine's UCI lock while a different
            # engine is idle (head-of-line blocking under mixed searches).
            self._execs = [
                ThreadPoolExecutor(max_workers=1) for _ in range(self.num_workers)
            ]
        except BaseException:
            for executor in self._execs:
                executor.shutdown(wait=True, cancel_futures=True)
            for engine in self._engines:
                with suppress(Exception):
                    engine.close()
            raise
  # itertools.count: next() is atomic in CPython, so concurrent submit()
  # callers can't both observe the same engine index (a bare int
  # read-modify-write could, skewing the round-robin distribution).
        self._next = itertools.count()

    def close(self) -> None:
        # Stop accepting work and discard requests that have not started. A
        # worker restart must not drain a potentially large async-label queue.
        # Running searches keep their engine lock; close() below waits for
        # those calls before terminating the UCI process, then the final join
        # reaps each executor thread.
        for executor in self._execs:
            executor.shutdown(wait=False, cancel_futures=True)
        for e in self._engines:
            e.close()
        for executor in self._execs:
            executor.shutdown(wait=True, cancel_futures=True)

    def set_nodes(self, nodes: int) -> None:
        self.nodes = int(nodes)
        for e in self._engines:
            e.set_nodes(int(nodes))

    def submit(
        self, fen: str, *, nodes: int | None = None,
        syzygy_path: str | None = None,
        fresh: bool = False,
    ) -> Future[StockfishResult]:
  # Round-robin assignment. `fresh` requests a cold-TT (ucinewgame) search —
  # see StockfishUCI.search; used by the label-escalation re-query. Only
  # forwarded when set, so the default path's engine call stays identical.
        engine_idx = next(self._next) % self.num_workers
        engine = self._engines[engine_idx]
        executor = self._execs[engine_idx]
        if fresh:
            return executor.submit(
                engine.search, fen, nodes=nodes, syzygy_path=syzygy_path, fresh=True,
            )
        return executor.submit(engine.search, fen, nodes=nodes, syzygy_path=syzygy_path)
