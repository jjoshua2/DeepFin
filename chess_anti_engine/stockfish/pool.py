from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from .uci import StockfishUCI, StockfishResult


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
        skill_level: int | None = None,
        hash_mb: int | None = None,
    ):
        self.path = path
        self.nodes = int(nodes)
        self.num_workers = int(num_workers)
        self.multipv = int(multipv)
        self.skill_level = skill_level
        self.hash_mb = None if hash_mb is None else max(1, int(hash_mb))

        self._exec = ThreadPoolExecutor(max_workers=self.num_workers)
        self._engines = [
            StockfishUCI(
                self.path,
                nodes=self.nodes,
                multipv=self.multipv,
                skill_level=self.skill_level,
                hash_mb=self.hash_mb,
            )
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


def _split_evenly(total: int, parts: int) -> list[int]:
    total = max(1, int(total))
    parts = max(1, int(parts))
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def build_stockfish_clients(
    *,
    path: str,
    nodes: int,
    total_workers: int,
    pipelines: int,
    multipv: int = 1,
    skill_level: int | None = None,
    hash_mb: int | None = None,
) -> list[StockfishUCI | StockfishPool]:
    """Create one or more local Stockfish clients while preserving total workers."""

    total_workers = max(1, int(total_workers))
    pipelines = max(1, min(int(pipelines), total_workers))
    worker_splits = _split_evenly(total_workers, pipelines)

    clients: list[StockfishUCI | StockfishPool] = []
    for worker_count in worker_splits:
        if int(worker_count) > 1:
            clients.append(
                StockfishPool(
                    path=path,
                    nodes=int(nodes),
                    num_workers=int(worker_count),
                    multipv=int(multipv),
                    skill_level=skill_level,
                    hash_mb=hash_mb,
                )
            )
        else:
            clients.append(
                StockfishUCI(
                    path,
                    nodes=int(nodes),
                    multipv=int(multipv),
                    skill_level=skill_level,
                    hash_mb=hash_mb,
                )
            )
    return clients
