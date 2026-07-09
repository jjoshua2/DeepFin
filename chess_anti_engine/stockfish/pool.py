from __future__ import annotations

import itertools

from concurrent.futures import Future, ThreadPoolExecutor

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
        self.multipv = int(multipv)
        self.hash_mb = None if hash_mb is None else max(1, int(hash_mb))
        self.syzygy_path = syzygy_path or None
        self.nice = min(19, max(0, int(nice)))

        self._exec = ThreadPoolExecutor(max_workers=self.num_workers)
        self._engines = [
            StockfishUCI(
                self.path,
                nodes=self.nodes,
                multipv=self.multipv,
                hash_mb=self.hash_mb,
                syzygy_path=self.syzygy_path,
                nice=self.nice,
            )
            for _ in range(self.num_workers)
        ]
  # itertools.count: next() is atomic in CPython, so concurrent submit()
  # callers can't both observe the same engine index (a bare int
  # read-modify-write could, skewing the round-robin distribution).
        self._next = itertools.count()

    def close(self) -> None:
        for e in self._engines:
            e.close()
        self._exec.shutdown(wait=True, cancel_futures=False)

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
        engine = self._engines[next(self._next) % self.num_workers]
        if fresh:
            return self._exec.submit(
                engine.search, fen, nodes=nodes, syzygy_path=syzygy_path, fresh=True,
            )
        return self._exec.submit(engine.search, fen, nodes=nodes, syzygy_path=syzygy_path)
