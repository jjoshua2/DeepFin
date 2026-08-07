"""heap_census: default-off arming, and one full census pass on this host."""
from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import numpy as np

from chess_anti_engine.heap_census import (
    FLAG_FILENAME,
    OUTPUT_FILENAME,
    _malloc_info_summary,
    _read_interval,
    census_once,
    maybe_start_heap_census,
)


def test_disarmed_without_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CAE_HEAP_CENSUS_SECONDS", raising=False)
    assert maybe_start_heap_census(tmp_path) is None
    assert not (tmp_path / OUTPUT_FILENAME).exists()


def test_interval_from_flag_file_with_floor(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CAE_HEAP_CENSUS_SECONDS", raising=False)
    (tmp_path / FLAG_FILENAME).write_text("1\n")
    assert _read_interval(tmp_path) == 10.0
    (tmp_path / FLAG_FILENAME).write_text("300\n")
    assert _read_interval(tmp_path) == 300.0
    (tmp_path / FLAG_FILENAME).write_text("garbage\n")
    assert _read_interval(tmp_path) is None
    (tmp_path / FLAG_FILENAME).write_text("0\n")
    assert _read_interval(tmp_path) is None


def test_unparseable_env_falls_back_to_flag_file(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / FLAG_FILENAME).write_text("120\n")
    monkeypatch.setenv("CAE_HEAP_CENSUS_SECONDS", "not-a-number")
    assert _read_interval(tmp_path) == 120.0
    monkeypatch.setenv("CAE_HEAP_CENSUS_SECONDS", "60")
    assert _read_interval(tmp_path) == 60.0


def test_census_once_records_all_views(tmp_path: Path) -> None:
    keep_alive = np.zeros((64, 175, 8, 8), dtype=np.float32)
    out_path = tmp_path / OUTPUT_FILENAME
    record = census_once(out_path)
    line = json.loads(out_path.read_text().splitlines()[-1])
    assert line.keys() == {str(k) for k in record}
    assert line["rss_kb"] > 0
    assert line["anon_kb"] > 0
    assert line["gc_objects"] > 1000
    assert line["gc_top"]
    assert isinstance(line["gc_top"][0][0], str)
    assert line["census_seconds"] >= 0
    malloc = line["malloc"]
    assert malloc is not None
    assert malloc["system_kb"] >= malloc["allocated_kb"] >= 0
    assert malloc["mmap_kb"] >= 0
    del keep_alive


def test_malloc_info_sees_new_allocations() -> None:
    before = _malloc_info_summary()
    assert before is not None
    # Many small (non-mmap-threshold) blocks so the glibc heap must grow.
    blocks = [bytearray(8192) for _ in range(4096)]
    after = _malloc_info_summary()
    assert after is not None
    assert after["allocated_kb"] > before["allocated_kb"]
    del blocks


def test_malloc_info_reports_mmap_blocks() -> None:
    big = bytearray(512 * 1024 * 1024)
    summary = _malloc_info_summary()
    assert summary is not None
    assert summary["mmap_kb"] >= 512 * 1024
    del big


def test_thread_starts_writes_and_stops(tmp_path: Path, monkeypatch) -> None:
    was_tracing = tracemalloc.is_tracing()
    monkeypatch.setenv("CAE_HEAP_CENSUS_SECONDS", "60")
    census = maybe_start_heap_census(tmp_path)
    assert census is not None
    try:
        out_path = tmp_path / OUTPUT_FILENAME
        for _ in range(100):
            if out_path.exists() and out_path.read_text().strip():
                break
            census.thread.join(timeout=0.1)
        line = json.loads(out_path.read_text().splitlines()[0])
        # Arming started tracemalloc, so the traced fields must be present.
        assert line["traced_current_kb"] > 0
        assert line["tracemalloc_overhead_kb"] > 0
        assert isinstance(line["tm_top"], list)
    finally:
        census.stop()
        if not was_tracing:
            tracemalloc.stop()
    assert not census.thread.is_alive()
