"""Flag-gated in-process heap census for worker leak diagnosis (2026-08-07 OOM).

Default OFF. A worker arms it only when ``<work_dir>/heap_census_seconds``
exists (its content is the sampling interval in seconds), so the fleet pays
nothing and a single revived worker can be instrumented by dropping the flag
file into its durable work dir before a graceful restart.

Each pass appends one JSON line to ``<work_dir>/heap_census.jsonl`` with four
independent views of the same heap, chosen so their disagreement localises a
leak that ptrace-based tools cannot reach on this host (WSL2, yama=1):

- ``rss_kb`` / ``anon_kb``: the kernel's view (what the Ray raylet kills on).
- ``malloc``: glibc ``malloc_info`` totals — allocated-vs-free splits a real
  C-side malloc leak from freed-but-unreturned fragmentation.
- ``traced_current_kb`` + ``tm_top``: tracemalloc, which numpy feeds via
  ``PyTraceMalloc_Track``, so a Python/numpy-side leak is named by the exact
  allocating line. A flat traced total under a climbing ``anon_kb`` convicts
  the C extensions instead.
- ``gc_objects`` / ``gc_top``: object counts by type, for retention leaks of
  small objects (records, game states) whose bytes hide inside the above.

Reader contract: a worker killed mid-pass may leave a PARTIAL final JSON line;
consumers must skip unparseable lines. ``tracemalloc_overhead_kb`` is the
trace table itself — it lives in ``anon_kb``/malloc but never in
``traced_current_kb``, so subtract it before comparing those two views.
"""
from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import threading
import time
import tracemalloc
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

FLAG_FILENAME = "heap_census_seconds"
OUTPUT_FILENAME = "heap_census.jsonl"
# Only traceback[0] is ever reported, and the trace table's size (and snapshot
# cost) scales with nframe, so 1 keeps the instrument's own footprint minimal.
_TRACEMALLOC_FRAMES = 1
_GC_TOP = 20
_TM_TOP = 25
_MIN_INTERVAL_SECONDS = 10.0


def _read_interval(work_dir: Path) -> float | None:
    """Interval in seconds, or None when the census is disarmed.

    The first source (env, then flag file) that yields a positive number wins;
    an unparseable source is warned about and skipped rather than silently
    disarming a worker whose flag file is valid.
    """
    def flag_text() -> str | None:
        try:
            return (work_dir / FLAG_FILENAME).read_text()
        except OSError:
            return None

    for raw in (os.environ.get("CAE_HEAP_CENSUS_SECONDS"), flag_text()):
        if raw is None:
            continue
        try:
            interval = float(raw.strip())
        except ValueError:
            logger.warning("heap census setting unreadable, skipping: %r", raw)
            continue
        if interval > 0:
            return max(interval, _MIN_INTERVAL_SECONDS)
    return None


def _proc_mem() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                if line.startswith("Rss:"):
                    out["rss_kb"] = int(line.split()[1])
                elif line.startswith("Anonymous:"):
                    out["anon_kb"] = int(line.split()[1])
    except OSError:
        pass
    return out


def _malloc_info_summary() -> dict[str, int] | None:
    """Whole-process glibc heap totals from ``malloc_info(3)``.

    Root-level ``<total>``/``<system>`` elements are the cross-arena totals;
    the per-arena copies are nested inside ``<heap>`` and skipped by only
    walking direct children. Large allocations that glibc serves via mmap are
    NOT in these numbers — they show up only in ``anon_kb``.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        buf = ctypes.POINTER(ctypes.c_char)()
        size = ctypes.c_size_t()
        libc.open_memstream.restype = ctypes.c_void_p
        fp = libc.open_memstream(ctypes.byref(buf), ctypes.byref(size))
        if not fp:
            return None
        try:
            rc = libc.malloc_info(0, ctypes.c_void_p(fp))
        finally:
            libc.fclose(ctypes.c_void_p(fp))
        try:
            if rc != 0 or not buf:
                return None
            xml_text = ctypes.string_at(buf, size.value)
        finally:
            if buf:
                libc.free(buf)
        root = ET.fromstring(xml_text)
    except (OSError, ET.ParseError, ValueError, AttributeError):
        return None
    free_b = 0
    system_b = 0
    mmap_b = 0
    for child in root:
        if child.tag == "total" and child.get("type") in ("fast", "rest"):
            free_b += int(child.get("size", "0"))
        elif child.tag == "total" and child.get("type") == "mmap":
            # Large blocks glibc serves via mmap: NOT in system/current, and
            # this is where big numpy buffers live.
            mmap_b = int(child.get("size", "0"))
        elif child.tag == "system" and child.get("type") == "current":
            system_b = int(child.get("size", "0"))
    return {
        "system_kb": system_b // 1024,
        "free_kb": free_b // 1024,
        "allocated_kb": max(system_b - free_b, 0) // 1024,
        "mmap_kb": mmap_b // 1024,
    }


def _gc_type_census() -> tuple[int, list[tuple[str, int]]]:
    objs = gc.get_objects()
    counts = Counter(type(o).__name__ for o in objs)
    return len(objs), counts.most_common(_GC_TOP)


def _tracemalloc_top() -> list[tuple[str, int, int]]:
    """Top allocation sites as (file:lineno, kb, count), largest first.

    Deliberately NO ``Snapshot.filter_traces``: it materialises every trace as
    Python objects (~10s and hundreds of MB transient per million live blocks
    — measured in PR #370 review), which would let the census inflate the very
    heap it measures. ``statistics("lineno")`` alone is ~10x cheaper; the few
    uninteresting sites are dropped by string match on the result instead.
    """
    if not tracemalloc.is_tracing():
        return []
    stats = tracemalloc.take_snapshot().statistics("lineno")
    out: list[tuple[str, int, int]] = []
    for stat in stats:
        frame = stat.traceback[0]
        if frame.filename == tracemalloc.__file__:
            continue
        site = "/".join(Path(frame.filename).parts[-3:]) + f":{frame.lineno}"
        out.append((site, stat.size // 1024, stat.count))
        if len(out) >= _TM_TOP:
            break
    return out


def census_once(out_path: Path) -> dict[str, object]:
    """Take one census pass and append it to ``out_path`` as a JSON line."""
    start = time.monotonic()
    record: dict[str, object] = {"ts": time.time()}
    record.update(_proc_mem())
    record["malloc"] = _malloc_info_summary()
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        record["traced_current_kb"] = current // 1024
        record["traced_peak_kb"] = peak // 1024
        # The trace table itself: present in anon/malloc but never in the
        # traced totals. Without this field, the instrument's own growth is
        # indistinguishable from an untraced (C-side) leak.
        record["tracemalloc_overhead_kb"] = tracemalloc.get_tracemalloc_memory() // 1024
    n_objs, top = _gc_type_census()
    record["gc_objects"] = n_objs
    record["gc_top"] = top
    record["tm_top"] = _tracemalloc_top()
    record["census_seconds"] = round(time.monotonic() - start, 3)
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _census_loop(out_path: Path, interval: float, stop: threading.Event) -> None:
    failures = 0
    while not stop.is_set():
        pass_start = time.monotonic()
        try:
            census_once(out_path)
            failures = 0
        except Exception:
            # Diagnostic only: never let the census kill or spam the worker.
            failures += 1
            if failures == 1:
                logger.warning("heap census pass failed", exc_info=True)
            if failures >= 5:
                logger.warning("heap census disabled after repeated failures")
                return
        # Interval measured from PASS START, so a slow pass cannot raise the
        # census duty cycle above interval's design point.
        elapsed = time.monotonic() - pass_start
        if stop.wait(timeout=max(interval - elapsed, 1.0)):
            return


class HeapCensus:
    """Handle for a running census: ``thread`` plus a cooperative ``stop()``."""

    def __init__(self, thread: threading.Thread, stop_event: threading.Event):
        self.thread = thread
        self._stop_event = stop_event

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self.thread.join(timeout=timeout)


def maybe_start_heap_census(work_dir: Path) -> HeapCensus | None:
    """Start the census thread iff this worker's flag file (or env) arms it."""
    interval = _read_interval(work_dir)
    if interval is None:
        return None
    if not tracemalloc.is_tracing():
        tracemalloc.start(_TRACEMALLOC_FRAMES)
    out_path = work_dir / OUTPUT_FILENAME
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_census_loop,
        args=(out_path, interval, stop_event),
        name="heap-census",
        daemon=True,
    )
    thread.start()
    logger.info(
        "heap census armed: interval=%.0fs out=%s tracemalloc_frames=%d",
        interval,
        out_path,
        _TRACEMALLOC_FRAMES,
    )
    return HeapCensus(thread, stop_event)
