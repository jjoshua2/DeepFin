from __future__ import annotations

import queue
import threading

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

        self._exec: ThreadPoolExecutor | None = None
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
            self._available_engines: queue.SimpleQueue[StockfishUCI] = queue.SimpleQueue()
            for engine in self._engines:
                self._available_engines.put(engine)
            self._worker_state = threading.local()
            # Each executor thread owns one engine for its lifetime. Requests
            # stay in the shared FIFO until any engine is free, avoiding both
            # UCI-lock contention and imbalance between fixed engine queues.
            self._exec = ThreadPoolExecutor(
                max_workers=self.num_workers,
                initializer=self._initialize_worker,
            )
        except BaseException:
            if self._exec is not None:
                self._exec.shutdown(wait=True, cancel_futures=True)
            for engine in self._engines:
                with suppress(Exception):
                    engine.close()
            raise

    def _initialize_worker(self) -> None:
        self._worker_state.engine = self._available_engines.get()

    def _search(
        self,
        fen: str,
        nodes: int | None,
        syzygy_path: str | None,
        fresh: bool,
    ) -> StockfishResult:
        engine: StockfishUCI = self._worker_state.engine
        if fresh:
            return engine.search(
                fen, nodes=nodes, syzygy_path=syzygy_path, fresh=True,
            )
        return engine.search(fen, nodes=nodes, syzygy_path=syzygy_path)

    def close(self) -> None:
        # Stop accepting work and discard requests that have not started. A
        # worker restart must not drain a potentially large async-label queue.
        # Running searches keep their engine lock; close() below waits for
        # those calls before terminating the UCI process, then the final join
        # reaps each executor thread.
        assert self._exec is not None
        self._exec.shutdown(wait=False, cancel_futures=True)
        for e in self._engines:
            e.close()
        self._exec.shutdown(wait=True, cancel_futures=True)

    def set_nodes(self, nodes: int) -> None:
        self.nodes = int(nodes)
        for e in self._engines:
            e.set_nodes(int(nodes))

    def submit(
        self, fen: str, *, nodes: int | None = None,
        syzygy_path: str | None = None,
        fresh: bool = False,
    ) -> Future[StockfishResult]:
  # The shared executor gives this request to the first free engine-owning
  # thread. `fresh` requests a cold-TT (ucinewgame) search; it is forwarded
  # only when set so the default engine call stays identical.
        assert self._exec is not None
        return self._exec.submit(
            self._search, fen, nodes, syzygy_path, fresh,
        )
