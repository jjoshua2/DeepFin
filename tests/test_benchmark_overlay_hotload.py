"""Small synthetic protocol tests; these never open the B shards."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_overlay_hotload.py"
SPEC = importlib.util.spec_from_file_location("benchmark_overlay_hotload", SCRIPT)
assert SPEC
assert SPEC.loader
hotload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hotload)


def fake_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(hotload, "git_head", lambda source: "pinned-commit")
    monkeypatch.setattr(hotload.subprocess, "check_output", lambda *args, **kwargs: "")
    source = tmp_path / "source"
    source.mkdir()
    (source / "target_overlay.py").write_text("original")
    receipt = tmp_path / "B.json"
    shards = [
        {"path": str(tmp_path / f"shard_{i}.zarr"), "rows": 8192,
         "content_sha256": f"{i:064x}"}
        for i in range(8)
    ]
    receipt.write_text(json.dumps({"shards": shards}))
    return {
        "status": "PREPARED_HELD_FOR_UNCONTENDED_CPU_SLOT",
        "runner_sha256": hotload.sha256(SCRIPT),
        "order": list(hotload.ORDER),
        "sources": {
            arm: {"path": str(source), "commit": "pinned-commit",
                  "files_sha256": {"target_overlay.py": hotload.sha256(source / "target_overlay.py")}}
            for arm in ("control", "candidate")
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            **{name: hotload.importlib.metadata.version(name)
               for name in hotload.PACKAGE_NAMES},
        },
        "fixture": {"receipt": str(receipt), "receipt_sha256": hotload.sha256(receipt),
                    "shards": shards, "rows": 65536},
        "bounds": {
            "seconds_per_arm": 180, "cpu_seconds_per_arm": 120,
            "max_arms": 4, "affinity": [16, 17], "nice": 19,
            "threads": 2, "minimum_memory_gib": 40,
            "minimum_disk_gib": 150, "new_output_limit_mib": 4,
            "gpu": False,
        },
    }


def test_exact_pins_and_source_or_receipt_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    hotload.check_pins(plan, runner=SCRIPT)
    source = Path(plan["sources"]["candidate"]["path"]) / "target_overlay.py"
    source.write_text("drift")
    with pytest.raises(hotload.Refused, match="source pin mismatch"):
        hotload.check_pins(plan, runner=SCRIPT)
    source.write_text("original")
    receipt = Path(plan["fixture"]["receipt"])
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises(hotload.Refused, match="qualification receipt pin mismatch"):
        hotload.check_pins(plan, runner=SCRIPT)
    plan["fixture"]["receipt_sha256"] = hotload.sha256(receipt)
    plan["runner_sha256"] = "0" * 64
    with pytest.raises(hotload.Refused, match="runner pin mismatch"):
        hotload.check_pins(plan, runner=SCRIPT)


def test_admission_is_explicit_fresh_and_bound_to_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    plan = {"status": "PREPARED_HELD_FOR_UNCONTENDED_CPU_SLOT"}
    admission_path = tmp_path / "admission.json"
    admission: dict[str, Any] = {
        "status": "ADMITTED", "plan_sha256": hotload.sha256(plan_path),
        "admitted_by": "operator", "quiet_workload_evidence": "quiet slot checked",
        "admitted_unix_seconds": time.time(),
    }
    admission_path.write_text(json.dumps(admission))
    hotload.check_admission(plan, plan_path, admission_path)
    admission["admitted_unix_seconds"] -= 901
    admission_path.write_text(json.dumps(admission))
    with pytest.raises(hotload.Refused, match="fresh"):
        hotload.check_admission(plan, plan_path, admission_path)
    admission["admitted_unix_seconds"] = time.time()
    admission["plan_sha256"] = "0" * 64
    admission_path.write_text(json.dumps(admission))
    with pytest.raises(hotload.Refused, match="matching ADMITTED"):
        hotload.check_admission(plan, plan_path, admission_path)


def test_resource_and_stop_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    # Patch only the resource probes, retaining the real STOP and output checks.
    real_read_text = Path.read_text
    available_kib = [41943040]

    def meminfo(path: Path, *_args: object, **_kwargs: object) -> str:
        if str(path) == "/proc/meminfo":
            return f"MemAvailable: {available_kib[0]} kB\n"
        return real_read_text(path)

    monkeypatch.setattr(Path, "read_text", meminfo)
    monkeypatch.setattr(hotload.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 150 * hotload.GIB})())
    monkeypatch.setattr(hotload.os, "sched_getaffinity", lambda pid: {16, 17})
    hotload.check_resources(plan, tmp_path)
    (tmp_path / "STOP").touch()
    with pytest.raises(hotload.Refused, match="STOP"):
        hotload.check_resources(plan, tmp_path)
    (tmp_path / "STOP").unlink()
    available_kib[0] -= 1
    with pytest.raises(hotload.Refused, match="available-memory"):
        hotload.check_resources(plan, tmp_path)
    available_kib[0] += 1
    monkeypatch.setattr(hotload.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 149 * hotload.GIB})())
    with pytest.raises(hotload.Refused, match="free-disk"):
        hotload.check_resources(plan, tmp_path)


def _receipt(arm: str, index: int, seconds: float) -> dict:
    return {
        "status": "PASS", "arm": arm, "index": index,
        "plan": {"batch_array": [128]}, "record_roster": [{"rows": 8192}],
        "decoded_array_hashes": {"x": {"shape": [1], "dtype": "|u1", "sha256": "a"}},
        "ordered_target_hashes": {"policy_target": {"shape": [1], "sha256": "b"}},
        "hot_load_seconds_per_shard": [seconds / 8] * 8,
    }


def test_exact_parity_controls_decision() -> None:
    receipts = [_receipt(arm, index, seconds) for index, (arm, seconds) in enumerate(
        zip(hotload.ORDER, (10.0, 8.0, 8.2, 10.2), strict=True)
    )]
    result = hotload.compare(receipts)
    assert result["decision"] == "WARRANTS_FUTURE_TRAINER_TEST"
    receipts[1]["decoded_array_hashes"]["x"]["sha256"] = "mutated"
    with pytest.raises(hotload.Refused, match="decoded_array_hashes"):
        hotload.compare(receipts)
    receipts[1]["decoded_array_hashes"]["x"]["sha256"] = "a"
    receipts[2]["plan"]["batch_array"] = [64, 64]
    with pytest.raises(hotload.Refused, match="plan"):
        hotload.compare(receipts)


def test_atomic_output_cap_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    with pytest.raises(hotload.Refused, match="exceeds"):
        hotload.atomic_json(path, {"value": "x" * 100}, 32)
    assert not list(tmp_path.iterdir())
    hotload.atomic_json(path, {"status": "PASS"}, 100)
    with pytest.raises(hotload.Refused, match="already exists"):
        hotload.atomic_json(path, {"status": "FAILED"}, 100)
    assert hotload.load_json(path)["status"] == "PASS"


def test_cleanup_never_signals_reused_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        pid = 12345

        def poll(self) -> None:
            return None

    signaled: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(hotload, "_proc_starttime", lambda pid: 22)
    monkeypatch.setattr(hotload.os, "killpg", lambda pid, sig: signaled.append((pid, sig)))
    assert hotload.cleanup_owned_child(Process(), 21) == "identity_lost_no_signal"
    assert signaled == []


def test_cleanup_escalates_only_owned_child(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        pid = 12345

        def poll(self) -> None:
            return None

        def wait(self, timeout: int) -> None:
            if timeout == 2 and len(signaled) == 1:
                raise subprocess.TimeoutExpired("child", timeout)

    signaled: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(hotload, "_proc_starttime", lambda pid: 21)
    monkeypatch.setattr(hotload.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(hotload.os, "killpg", lambda pid, sig: signaled.append((pid, sig)))
    assert hotload.cleanup_owned_child(Process(), 21) == "killed"
    assert signaled == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


def test_failed_admission_writes_evidence_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(json.dumps({"status": "HELD"}))
    output_dir = tmp_path / "output"

    def unexpected_launch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("child launched without admission")

    monkeypatch.setattr(hotload.subprocess, "Popen", unexpected_launch)
    with pytest.raises(hotload.Refused, match="ADMITTED"):
        hotload.run(plan, plan_path, admission_path, output_dir)
    summary = hotload.load_json(output_dir / "summary.json")
    assert summary["status"] == "FAILED"
    assert summary["completed_arms"] == 0
    assert "ADMITTED" in summary["error"]


def test_deadline_records_owned_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    admission_path = tmp_path / "admission.json"
    admission_path.write_text("{}")
    monkeypatch.setattr(hotload, "check_pins", lambda *args, **kwargs: None)
    monkeypatch.setattr(hotload, "check_admission", lambda *args, **kwargs: None)
    monkeypatch.setattr(hotload, "check_resources", lambda *args, **kwargs: None)
    monkeypatch.setattr(hotload, "_proc_starttime", lambda pid: 21)
    cleaned: list[int] = []

    class TimedOut:
        pid = 12345
        returncode = None

        def wait(self, timeout: int) -> None:
            raise subprocess.TimeoutExpired("child", timeout)

        def poll(self) -> None:
            return None

    monkeypatch.setattr(hotload.subprocess, "Popen", lambda *args, **kwargs: TimedOut())
    monkeypatch.setattr(
        hotload, "cleanup_owned_child",
        lambda process, starttime: cleaned.append(starttime) or "terminated",
    )
    real_supervise = hotload.supervise_child
    ticks = iter((0.0, 181.0))
    monkeypatch.setattr(
        hotload, "supervise_child",
        lambda process, starttime, spec, directory, index: real_supervise(
            process, starttime, spec, directory, index, clock=lambda: next(ticks),
        ),
    )
    output_dir = tmp_path / "output"
    with pytest.raises(hotload.Refused, match="wall deadline"):
        hotload.run(plan, plan_path, admission_path, output_dir)
    assert cleaned
    summary = hotload.load_json(output_dir / "summary.json")
    assert "cleanup=terminated" in summary["error"]


def test_stop_during_arm_triggers_owned_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fake_plan(tmp_path, monkeypatch)
    process = type("Process", (), {
        "wait": lambda self, timeout: (_ for _ in ()).throw(subprocess.TimeoutExpired("child", timeout)),
    })()
    cleaned: list[int] = []
    monkeypatch.setattr(
        hotload, "check_resources",
        lambda spec, directory: (_ for _ in ()).throw(hotload.Refused("STOP requested")),
    )
    monkeypatch.setattr(
        hotload, "cleanup_owned_child",
        lambda child, starttime: cleaned.append(starttime) or "terminated",
    )
    with pytest.raises(hotload.Refused, match=r"STOP requested.*cleanup=terminated"):
        hotload.supervise_child(process, 21, plan, tmp_path, 0, clock=lambda: 0.0)
    assert cleaned == [21]
