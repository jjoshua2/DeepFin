"""Small real-child checks of the paired-pack auditor's exit-aware monitor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import pytest

from scripts import audit_sf_dlite_paired_pack as runner


def _run_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
              child_code: str, *, seconds: int = 1800) -> tuple[float, list[dict]]:
    (tmp_path / "resource_receipts").mkdir()
    monkeypatch.setattr(runner, "UNIT_SECONDS", seconds)
    monkeypatch.setattr(runner, "AVAILABLE_KIB_MIN", 0)
    monkeypatch.setattr(runner, "FREE_BYTES_MIN", 0)
    monkeypatch.setattr(runner, "physical_counters", lambda: {"fixture": (0, 0)})
    monkeypatch.setattr(runner, "memory_available_kib", lambda: 1 << 40)
    baseline = {"fixture": (0, 0)}
    io_state = {"baseline": baseline, "last": baseline}
    lease_fd = os.open("/dev/null", os.O_RDONLY)
    started = time.monotonic()
    try:
        runner.supervise([sys.executable, "-B", "-c", child_code,
                          str(tmp_path / "LABEL-INDEX.json")],
                         tmp_path, lease_fd, "a" * 64, "label-index", io_state)
    finally:
        os.close(lease_fd)
    elapsed = time.monotonic() - started
    samples = [json.loads(line) for line in
               (tmp_path / "RESOURCE.jsonl").read_text().splitlines()]
    assert samples
    assert all(sample["unit"] == "label-index" for sample in samples)
    return elapsed, samples


def _not_running(pid: int) -> bool:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return True
    return any(line.startswith("State:") and line.split()[1].startswith("Z")
               for line in status.read_text().splitlines())


def test_exit_aware_fast_child_has_final_resource_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'{}\\n')"
    elapsed, samples = _run_unit(tmp_path, monkeypatch, child)
    assert elapsed < 2, "prompt child should not wait for a five-second sample tick"
    assert samples[-1]["child_exit_code"] == 0
    assert runner.require_resource(tmp_path, "a" * 64, "label-index")
    assert _not_running(samples[0]["child_pid"])


def test_exit_aware_nonzero_child_has_no_resource_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="child label-index exited 7"):
        _run_unit(tmp_path, monkeypatch, "raise SystemExit(7)")
    samples = [json.loads(line) for line in
               (tmp_path / "RESOURCE.jsonl").read_text().splitlines()]
    assert samples[-1]["child_exit_code"] == 7
    assert not (tmp_path / "resource_receipts/label-index.json").exists()
    assert _not_running(samples[0]["child_pid"])


def test_exit_aware_deadline_reaps_child_without_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = time.monotonic()
    with pytest.raises(ValueError, match="child exceeded wall"):
        _run_unit(tmp_path, monkeypatch, "import time; time.sleep(30)", seconds=1)
    assert time.monotonic() - started < 3
    samples = [json.loads(line) for line in
               (tmp_path / "RESOURCE.jsonl").read_text().splitlines()]
    assert samples[-1]["wall_seconds"] >= 1
    assert not (tmp_path / "resource_receipts/label-index.json").exists()
    assert _not_running(samples[0]["child_pid"])
