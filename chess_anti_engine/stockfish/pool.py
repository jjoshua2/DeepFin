from __future__ import annotations

import logging
import queue
import threading

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress

from .uci import StockfishResult, StockfishUCI


_LOG = logging.getLogger("chess_anti_engine.stockfish")


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
        read_timeout_s: float | None = None,
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
        # Carried into every engine INCLUDING replacements, so a pool built with
        # a custom deadline does not silently revert to the 60s default the
        # moment an engine is swapped.
        self.read_timeout_s = read_timeout_s

        self._exec: ThreadPoolExecutor | None = None
        self._engines: list[StockfishUCI] = []
        # Guards _engines while a worker thread swaps in a replacement process;
        # set_nodes/close iterate the same list from other threads.
        self._engines_lock = threading.Lock()
        self.replacements = 0  # desynced engines swapped out this session
        # Set by close(). Stops _replace_engine spawning a process that would
        # outlive the snapshot close() reaps.
        self._closing = False
        try:
            for _ in range(self.num_workers):
                self._engines.append(self._new_engine())
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

    def _new_engine(self) -> StockfishUCI:
        kwargs = (
            {} if self.read_timeout_s is None
            else {"read_timeout_s": float(self.read_timeout_s)}
        )
        return StockfishUCI(
            self.path,
            nodes=self.nodes,
            multipv=self.multipv,
            hash_mb=self.hash_mb,
            syzygy_path=self.syzygy_path,
            nice=self.nice,
            **kwargs,
        )

    def _initialize_worker(self) -> None:
        self._worker_state.engine = self._available_engines.get()

    def _replace_engine(self, old: StockfishUCI) -> None:
        """Swap a desynced engine for a fresh process, in its owning thread.

        A desynced engine can only raise from here on (see
        ``StockfishDesyncError``), so leaving it in place would zero this
        thread's SF throughput. Replacing it restores throughput WITHOUT ever
        serving a stale result. If building the replacement raises, the poisoned
        engine stays installed: every later call raises, which is loud and
        correct, and the next call retries the replacement.

        No-op once ``close`` has run. A replacement built after ``close`` took
        its snapshot would be an orphaned Stockfish process that nothing reaps.
        """
        with self._engines_lock:
            if self._closing:
                return
        fresh = self._new_engine()
        with self._engines_lock:
            if self._closing:  # closed while the replacement was starting
                with suppress(Exception):
                    fresh.close()
                return
            try:
                self._engines[self._engines.index(old)] = fresh
            except ValueError:  # already swapped out by a previous failure
                self._engines.append(fresh)
            self.replacements += 1
            replacements = self.replacements
        self._worker_state.engine = fresh
        with suppress(Exception):
            old.close()
        # The ONLY notice that the desync repair fired. Everything else about it
        # is silent by design: the raise is swallowed at DEBUG on the label path,
        # and the health counters stay at baseline precisely BECAUSE no stale
        # result was served. Without this line the repair working and the repair
        # never being needed look identical in the log.
        _LOG.warning(
            "stockfish pool: replaced a desynced engine (replacements=%d this "
            "session) — a search was abandoned and its engine was one result "
            "behind; the abandoned query's label is lost, not misfiled",
            replacements,
        )

    def _search(
        self,
        fen: str,
        nodes: int | None,
        syzygy_path: str | None,
        fresh: bool,
        searchmoves: Sequence[str] | None = None,
    ) -> StockfishResult:
        engine: StockfishUCI = self._worker_state.engine
        try:
            # Both optional arguments are forwarded ONLY when set, so the call
            # the production label path makes stays byte-identical to the one
            # it made before either existed.
            extra: dict[str, object] = {}
            if fresh:
                extra["fresh"] = True
            if searchmoves:
                extra["searchmoves"] = searchmoves
            return engine.search(
                fen, nodes=nodes, syzygy_path=syzygy_path, **extra,  # pyright: ignore[reportArgumentType]
            )
        except BaseException:
            # The ORIGINAL raise must reach the caller — it is the one that
            # names what went wrong. `suppress` is load-bearing: _replace_engine
            # spawns a process, and a constructor failure raised from inside
            # this handler would SUPERSEDE the original with an unrelated
            # exception class (FileNotFoundError), which worker.py's
            # (StockfishTimeoutError, StockfishDesyncError) handler does not
            # match and the curriculum-move path does not catch at all.
            # Swallowing it is safe: the poisoned engine simply stays installed,
            # every later call raises StockfishDesyncError, and the next call
            # retries the replacement.
            if engine.desynced:
                with suppress(Exception):
                    self._replace_engine(engine)
            raise

    def close(self) -> None:
        # Stop accepting work and discard requests that have not started. A
        # worker restart must not drain a potentially large async-label queue.
        # Running searches keep their engine lock; close() below waits for
        # those calls before terminating the UCI process, then the final join
        # reaps each executor thread.
        assert self._exec is not None
        self._exec.shutdown(wait=False, cancel_futures=True)
        with self._engines_lock:
            # Under the lock and BEFORE the snapshot: from here on
            # _replace_engine is a no-op, so this list cannot grow behind us and
            # no Stockfish process can survive the join.
            self._closing = True
            engines = list(self._engines)
        for e in engines:
            e.close()
        self._exec.shutdown(wait=True, cancel_futures=True)

    def set_nodes(self, nodes: int) -> None:
        self.nodes = int(nodes)
        with self._engines_lock:
            engines = list(self._engines)
        for e in engines:
            e.set_nodes(int(nodes))

    def submit(
        self, fen: str, *, nodes: int | None = None,
        syzygy_path: str | None = None,
        fresh: bool = False,
        searchmoves: Sequence[str] | None = None,
    ) -> Future[StockfishResult]:
  # The shared executor gives this request to the first free engine-owning
  # thread. `fresh` requests a cold-TT (ucinewgame) search; it is forwarded
  # only when set so the default engine call stays identical.
  #
  # `searchmoves` restricts the ROOT to those moves, which is what makes a
  # TARGETED comparison affordable: the whole node budget is spent separating
  # the listed candidates instead of re-deriving the full move list. It is
  # validated in StockfishUCI.search, so a bad token raises here rather than
  # being silently dropped -- see `_validated_searchmoves`.
        assert self._exec is not None
        return self._exec.submit(
            self._search, fen, nodes, syzygy_path, fresh, searchmoves,
        )
