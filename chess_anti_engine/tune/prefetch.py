"""Background shard prefetcher for distributed selfplay ingest.

Moves zarr disk-decode out of the trainer's iter-boundary ingest into a
daemon thread that runs during the train phase. The trainer's
``_ingest_distributed_selfplay`` drains the queue first, then falls back
to inbox poll for shards that arrived after the prefetcher last scanned.

Deferred-registration design — ``buf.add_many_arrays`` still happens at
iter time on the trainer thread, so the sampling distribution is
unchanged from the pre-prefetch path.

**Ordering invariant (load-bearing):** ``drain()`` must run before any
in-iter inbox poll. ``_ingest_distributed_selfplay`` does this, and the
inbox-poll fallback runs against shards still on disk — those have
already been removed from ``_queue`` by drain, and the trainer's atomic
inbox→processed move at iter time prevents double-registration on the
next scan.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from chess_anti_engine.replay.shard import load_shard_arrays

log = logging.getLogger(__name__)


class BackgroundShardPrefetcher:
    """Polls an inbox directory in a thread and pre-decodes new shards.

    Use lifecycle: ``start()`` once at trial init; call ``drain()`` at
    the start of each iter's ingest to consume all decoded shards;
    ``stop()`` at trial teardown to cleanly join the thread.
    """

    def __init__(
        self,
        inbox_dir: Path,
        *,
        poll_seconds: float = 1.0,
        path_iter: Callable[[Path], list[Path]],
    ) -> None:
        self._inbox_dir = inbox_dir
        self._poll_seconds = max(0.1, float(poll_seconds))
        self._path_iter = path_iter

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = False
  # Single piece of state. A path lives in `_queue` until the trainer
  # drains it; while present it dedupes against re-scans of inbox/.
        self._queue: list[tuple[Path, dict, dict]] = []
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="ShardPrefetch", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=timeout)
        self._thread = None

    def drain(self) -> list[tuple[Path, dict, dict]]:
        """Return all decoded shards and clear the queue."""
        with self._cond:
            items = self._queue
            self._queue = []
        return items

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._stop:
                    return
            try:
                self._scan_once()
            except Exception:  # noqa: BLE001
                log.exception("prefetch scan failed")
            with self._cond:
                if self._stop:
                    return
                self._cond.wait(timeout=self._poll_seconds)

    def _scan_once(self) -> None:
        try:
            shard_paths = self._path_iter(self._inbox_dir)
        except FileNotFoundError:
            return
        with self._lock:
            already_queued = {p for p, _, _ in self._queue}
        for sp in shard_paths:
            if sp in already_queued:
                continue
            try:
                arrs, meta = load_shard_arrays(sp)
            except Exception:  # noqa: BLE001
  # Could be a partial write the worker hasn't atomically renamed
  # yet. Skip — next scan will retry. If genuinely corrupt, the
  # iter-time fallback in _process_shard will quarantine via bad/.
                continue
            with self._lock:
  # Re-check stop under lock; if drained between the queued-set
  # snapshot and now, the next scan will re-pick it up.
                if self._stop:
                    return
                self._queue.append((sp, arrs, meta))
                already_queued.add(sp)
