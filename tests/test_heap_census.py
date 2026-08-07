"""heap_census: default-off arming, and one full census pass on this host."""
from __future__ import annotations

import json
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


def test_census_once_records_all_views(tmp_path: Path) -> None:
    keep_alive = np.zeros((64, 175, 8, 8), dtype=np.float32)
    out_path = tmp_path / OUTPUT_FILENAME
    record = census_once(out_path)
    line = json.loads(out_path.read_text().splitlines()[-1])
    assert line.keys() == {str(k) for k in record}
    assert line["rss_kb"] > 0
    assert line["anon_kb"] > 0
    assert line["gc_objects"] > 1000
    assert line["gc_top"] and isinstance(line["gc_top"][0][0], str)
    malloc = line["malloc"]
    assert malloc is not None
    assert malloc["system_kb"] >= malloc["allocated_kb"] >= 0
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


def test_thread_starts_and_writes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CAE_HEAP_CENSUS_SECONDS", "60")
    thread = maybe_start_heap_census(tmp_path)
    assert thread is not None
    out_path = tmp_path / OUTPUT_FILENAME
    for _ in range(100):
        if out_path.exists() and out_path.read_text().strip():
            break
        thread.join(timeout=0.1)
    line = json.loads(out_path.read_text().splitlines()[0])
    # tracemalloc was started by arming, so the traced fields must be present.
    assert line["traced_current_kb"] > 0
    assert isinstance(line["tm_top"], list)
