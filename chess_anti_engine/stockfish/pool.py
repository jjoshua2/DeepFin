from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from .uci import StockfishUCI, StockfishResult


class StockfishPool:
    """Thread-based pool of independent StockfishUCI processes.

    Each worker owns its own Stockfish process (UCI is not designed for
    concurrent requests over a single process).
    """

    def __init__(self, *, path: str, nodes: int, num_workers: int, multipv: int = 1, skill_level: int | None = None):
        self.path = path
        self.nodes = int(nodes)
        self.num_workers = int(num_workers)
        self.multipv = int(multipv)
        self.skill_level = skill_level

        self._exec = ThreadPoolExecutor(max_workers=self.num_workers)
        self._engines = [
            StockfishUCI(self.path, nodes=self.nodes, multipv=self.multipv, skill_level=self.skill_level)
            for _ in range(self.num_workers)
        ]
        self._next = 0
        self._next = 0

    def close(self) -> None:
        for e in self._engines:
            e.close()
        self._exec.shutdown(wait=True, cancel_futures=False)

    def set_nodes(self, nodes: int) -> None:
        self.nodes = int(nodes)
        for e in self._engines:
            e.set_nodes(int(nodes))

    def submit(self, fen: str, *, nodes: int | None = None) -> Future[StockfishResult]:
        # Round-robin assignment
        idx = self._next
        self._next = (self._next + 1) % self.num_workers
        engine = self._engines[idx]
        return self._exec.submit(engine.search, fen, nodes=nodes)
