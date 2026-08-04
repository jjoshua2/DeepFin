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

**Byte budget (audit A19).** The producer rate is "however fast shards
land in the inbox"; the consumer rate is "once per training iteration".
Those are unrelated, and the queue used to bound neither. A 2000-row
shard decodes to ~102 MB resident (measured on live shards, trial
13a9f), so an undrained queue cost ~5.1 GB at 50 shards and ~10.2 GB at
100. The regime that fills it fastest is the cold-start upload burst —
many workers uploading at once while the trainer is still inside a long
first iteration — i.e. the deepest inbox coincides with the longest
drain interval. On a box already holding the training process and the
replay window that is an OOM path, and a silent one until it is fatal.

``max_queued_bytes`` bounds it. When the queue is at or over budget the
scan STOPS DECODING and leaves the remaining shards on disk; it does not
drop them and it does not block. See ``_scan_once`` for why deferring is
the correct backpressure here rather than blocking the producer.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from chess_anti_engine.replay.shard import load_shard_arrays

log = logging.getLogger(__name__)

# ~7.5 shards at the measured 102 MB/shard decode size, chosen against the
# live box: 98 GB total with ~28 GB available beside the training process and
# the replay window. Generous for steady state (the trainer drains every
# iteration, so the queue normally holds the 1-3 shards that landed during
# one train phase) while turning the pathological cold-start case from
# ~10.2 GB at 100 queued shards into a hard ceiling. Deliberately a round
# number of MB rather than an exact multiple of a shard: shard sizes vary
# with row count, which is the whole reason this is a byte budget and not a
# queue-length cap.
DEFAULT_MAX_QUEUED_BYTES = 768 * 1024 * 1024


def _arrays_nbytes(arrs: dict) -> int:
    """Resident size of one decoded shard.

    ``getattr(..., "nbytes", 0)`` rather than assuming every value is an
    ndarray: ``load_shard_arrays`` attaches identity/policy metadata into the
    same dict, and a non-array sneaking in must not take the accounting down —
    an exception here runs on the producer thread and would be swallowed by
    ``_run``'s handler, silently disabling prefetch for the rest of the run.
    """
    return sum(int(getattr(a, "nbytes", 0)) for a in arrs.values())


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
        max_queued_bytes: int = DEFAULT_MAX_QUEUED_BYTES,
    ) -> None:
        self._inbox_dir = inbox_dir
        self._poll_seconds = max(0.1, float(poll_seconds))
        self._path_iter = path_iter
  # A non-positive budget would disable prefetch entirely rather than
  # unbound it -- the opposite of what someone zeroing a "max" expects, and
  # a silent throughput regression. Floor at one shard's worth so the knob
  # cannot be set to a value that means "never prefetch".
        self._max_queued_bytes = max(1, int(max_queued_bytes))

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = False
  # A path lives in `_queue` until the trainer drains it; while present it
  # dedupes against re-scans of inbox/. `_queued_bytes` is the resident
  # size of exactly the items in `_queue`, maintained alongside it under
  # the same lock -- never recomputed from the queue, so the two cannot
  # disagree about what has been counted.
        self._queue: list[tuple[Path, dict, dict]] = []
        self._queued_bytes = 0
        self._deferred_scans = 0
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
        """Return all decoded shards and clear the queue.

        The ``(path, arrays, meta)`` tuple shape is the consumer's contract --
        ``distributed_runtime`` unpacks it as ``for sp, arrs, meta in
        prefetcher.drain()`` and hands ``(arrs, meta)`` on as ``preloaded``.
        The byte budget is tracked in a separate counter precisely so this
        shape did not have to change.
        """
        with self._cond:
            items = self._queue
            self._queue = []
  # Exact, not approximate: the whole queue just left, so the resident
  # total is zero by construction rather than by subtraction.
            self._queued_bytes = 0
  # The producer may be parked in `wait(poll_seconds)` having deferred a
  # scan. Wake it so a freed budget is used now instead of up to a poll
  # interval later -- the drain happens at the START of ingest, so that
  # interval is prime prefetch time.
            self._cond.notify_all()
        return items

    def queued_bytes(self) -> int:
        """Resident bytes currently held by the queue. For tests and logging."""
        with self._lock:
            return self._queued_bytes

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._stop:
                    return
            try:
                self._scan_once()
            except Exception:
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
  # ⚑ Budget checked BEFORE the decode, not after. Decoding and then
  # discarding would already have paid the ~102 MB resident spike, which
  # is the exact cost this bound exists to avoid. The consequence is that
  # the queue can finish OVER budget -- the last shard was admitted while
  # under it -- by strictly less than one shard, so the true ceiling is
  # `max_queued_bytes + largest single shard` (~870 MB at the 768 MB
  # default and the measured 102 MB shard). Bounding it more
  # tightly would require knowing a shard's decoded size before decoding
  # it, which the zarr path cannot answer cheaply.
            with self._lock:
                if self._stop:
                    return
                over_budget = self._queued_bytes >= self._max_queued_bytes
                queued_now, bytes_now = len(self._queue), self._queued_bytes
            if over_budget:
  # DEFER, not drop and not block.
  #
  # Not DROP: nothing is discarded. `sp` is untouched on disk, still in
  # inbox/, and gets picked up either by a later scan once the trainer
  # has drained, or by the iter-time inbox-poll fallback in
  # `_ingest_distributed_selfplay`, which decodes it inline. Prefetch is
  # a latency optimisation over that fallback, so declining to run ahead
  # is a no-op for correctness -- it costs the decode at iter time,
  # which is precisely where it happened before this module existed.
  #
  # Not BLOCK: blocking the producer on a condition variable would add
  # deadlock surface (`stop()` would have to wake a waiter that may be
  # mid-decode) to buy nothing. The backpressure signal here is not
  # in-flight data that would be lost -- it is a file on disk that will
  # still be there next poll. `break` rather than `continue` because
  # every remaining path in this snapshot faces the same full queue.
                self._deferred_scans += 1
                if self._deferred_scans == 1 or self._deferred_scans % 100 == 0:
                    log.warning(
                        "shard prefetch deferred: queue holds %d shard(s) / "
                        "%.1f MB against a %.1f MB budget, so %d newly seen "
                        "shard(s) stay on disk for the iter-time ingest path. "
                        "The trainer is draining slower than shards arrive "
                        "(deferrals this run: %d)",
                        queued_now, bytes_now / 1e6,
                        self._max_queued_bytes / 1e6,
                        len(shard_paths) - queued_now, self._deferred_scans,
                    )
                return
            try:
                arrs, meta = load_shard_arrays(sp)
            except Exception:
  # Could be a partial write the worker hasn't atomically renamed
  # yet. Skip — next scan will retry. If genuinely corrupt, the
  # iter-time fallback in _process_shard will quarantine via bad/.
                continue
            nbytes = _arrays_nbytes(arrs)
            with self._lock:
  # Re-check stop under lock; if drained between the queued-set
  # snapshot and now, the next scan will re-pick it up.
                if self._stop:
                    return
                self._queue.append((sp, arrs, meta))
                self._queued_bytes += nbytes
                already_queued.add(sp)
