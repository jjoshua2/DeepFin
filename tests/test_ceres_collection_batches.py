"""CPU-only stdlib tests for Ceres collection batch orchestration."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from scripts import ceres_collection_batches as tool

REPO = Path(__file__).resolve().parents[1]

WORKER = r"""
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
completion = Path(sys.argv[2])
payload = json.loads(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
completion.parent.mkdir(parents=True, exist_ok=True)
print("chunk-ok", flush=True)
completion.write_text(json.dumps(payload), encoding="utf-8")
"""

WORKER_FAIL = "import sys; print('chunk-fail', flush=True); sys.exit(1)\n"

WORKER_MISSING = r"""
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
print("chunk-no-completion", flush=True)
"""

WORKER_HANG = r"""
import sys
import time
from pathlib import Path
Path(sys.argv[1]).write_text("ready", encoding="utf-8")
print("chunk-hang", flush=True)
time.sleep(600)
"""

WORKER_HANG_KILL = r"""
import signal
import sys
import time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text("ready", encoding="utf-8")
print("chunk-ignore-term", flush=True)
time.sleep(600)
"""

WORKER_MUTATE_PIN = r"""
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
completion = Path(sys.argv[2])
payload = json.loads(sys.argv[3])
pin = Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
completion.parent.mkdir(parents=True, exist_ok=True)
print("chunk-mutate", flush=True)
completion.write_text(json.dumps(payload), encoding="utf-8")
pin.write_text("mutated-pin", encoding="utf-8")
"""

WORKER_LOCK = r"""
import fcntl
import os
import sys
import time
from pathlib import Path
lock = Path(sys.argv[1]).open("a")
fcntl.flock(lock, fcntl.LOCK_EX)
Path(sys.argv[2]).write_text(str(os.getpid()), encoding="utf-8")
print("chunk-lock", flush=True)
time.sleep(600)
"""

WORKER_LEADER_EXITS = r"""
import fcntl
import os
import signal
import sys
import time
from pathlib import Path
lock_path = Path(sys.argv[1])
ready_path = Path(sys.argv[2])
child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    ready_path.write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(600)
    os._exit(0)
time.sleep(600)
"""

FAKE_CERES = r"""
import json
import sys
import time
from pathlib import Path
out = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
mode = sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
root = out / "invocations"

def write_inv(name, *, parent=True, ended=None):
    inv = root / name
    inv.mkdir(parents=True)
    child = {key: value for key, value in payload.items() if key != "ended_unix"}
    (inv / "child_completed.json").write_text(json.dumps(child), encoding="utf-8")
    if parent:
        body = dict(payload)
        body["ended_unix"] = time.time() if ended is None else ended
        (inv / "completed.json").write_text(json.dumps(body), encoding="utf-8")

if mode == "ok":
    write_inv(str(time.time_ns()))
elif mode == "missing_parent":
    write_inv(str(time.time_ns()), parent=False)
elif mode == "ambiguous":
    write_inv("1")
    write_inv("2")
elif mode == "stale":
    write_inv(str(time.time_ns()), ended=1.0)
else:
    raise SystemExit("unknown fake ceres mode")
print("fake-ceres", mode, flush=True)
"""

DRIVER_RUNNER = r"""
import sys
sys.path.insert(0, sys.argv[1])
from scripts import ceres_collection_batches as tool
tool.disk_free_bytes = lambda path: 200 * 2**30
raise SystemExit(tool.main(sys.argv[2:]))
"""


@pytest.fixture
def _plenty_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool, "disk_free_bytes", lambda _path: 200 * 2**30)


def sha(path: Path) -> str:
    return tool.sha256_file(path)


def write_script(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o100)
    return path


def counts(rows: int, padding: int) -> dict[str, int]:
    inputs = rows + padding
    return {
        "real_rows": rows,
        "padding_rows": padding,
        "input_rows": inputs,
        "calls": inputs // 32,
    }


def completion_payload(start: int, shards: int, rows_each: int, padding: int) -> dict[str, Any]:
    rows = shards * rows_each
    accounting = counts(rows, padding)
    return {
        "complete": True,
        "rows": rows,
        "shards": shards,
        "new_shards": shards,
        "selection": [
            {"path": f"shard_{start + index:06d}.zarr", "rows": rows_each}
            for index in range(shards)
        ],
        "collection_counts": accounting,
        "new_collection_counts": accounting,
    }


def chunk_record(
    tmp: Path,
    ident: str,
    start: int,
    shards: int,
    rows_each: int,
    padding: int,
    worker: Path,
    payload: dict[str, Any] | None = None,
    extra_argv: list[str] | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    output = tmp / "outs" / ident
    output.parent.mkdir(parents=True, exist_ok=True)
    completion = output / "completed.json"
    body = payload if payload is not None else completion_payload(
        start, shards, rows_each, padding
    )
    argv = [sys.executable, str(worker), str(output), str(completion), json.dumps(body)]
    if extra_argv:
        argv.extend(extra_argv)
    return {
        "id": ident,
        "start_shard": start,
        "max_shards": shards,
        "expected_rows": shards * rows_each,
        "expected_padding_rows": padding,
        "output_directory": str(output),
        "working_directory": str(tmp),
        "argv": argv,
        "timeout_seconds": timeout_seconds,
        "completion_mode": "fixed_path",
        "expected_completion": str(completion),
    }


def make_plan(
    tmp: Path,
    chunks: list[dict[str, Any]],
    **overrides: Any,
) -> tuple[Path, str, dict[str, Any]]:
    pin = tmp / "operator.json"
    if not pin.exists():
        pin.write_text('{"role":"operator"}', encoding="utf-8")
    plan = {
        "schema": 1,
        "state_directory": str(tmp / "state"),
        "minimum_free_gib": 150,
        "overall_seconds": 120,
        "pause_between_chunks_seconds": 30,
        "pinned_files": {str(pin): sha(pin)},
        "chunks": chunks,
    }
    plan.update(overrides)
    path = tmp / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, sha(path), plan


def two_chunk_plan(tmp: Path, **overrides: Any) -> tuple[Path, str, Path]:
    worker = write_script(tmp / "worker.py", WORKER)
    chunks = [
        chunk_record(tmp, "c16", 16, 2, 32, 0, worker),
        chunk_record(tmp, "c18", 18, 2, 32, 0, worker),
    ]
    path, digest, _plan = make_plan(tmp, chunks, **overrides)
    return path, digest, worker


def ceres_chunk(
    tmp: Path,
    ident: str,
    worker: Path,
    mode: str,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    output = tmp / "outs" / ident
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = completion_payload(16, 2, 32, 0)
    return {
        "id": ident,
        "start_shard": 16,
        "max_shards": 2,
        "expected_rows": 64,
        "expected_padding_rows": 0,
        "output_directory": str(output),
        "working_directory": str(tmp),
        "argv": [sys.executable, str(worker), str(output), json.dumps(payload), mode],
        "timeout_seconds": timeout_seconds,
        "completion_mode": "ceres_invocations",
    }


def lock_is_free(path: Path) -> bool:
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def wait_file(path: Path, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert path.exists(), f"missing {path}"


def test_default_validates_and_does_not_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path, digest, _worker = two_chunk_plan(tmp_path)

    def refuse_popen(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("subprocess started without --execute")

    monkeypatch.setattr(subprocess, "Popen", refuse_popen)
    code = tool.main(["--plan", str(path), "--expected-plan-sha256", digest])
    captured = capsys.readouterr()
    assert code == 0
    printed = json.loads(captured.out)
    assert printed["status"] == "VALID"
    assert printed["plan_sha256"] == digest
    assert [chunk["id"] for chunk in printed["chunks"]] == ["c16", "c18"]
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "outs" / "c16").exists()
    assert captured.err == ""


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda plan: plan.__setitem__("schema", 2), "schema"),
        (lambda plan: plan.__setitem__("schema", True), "integer"),
        (lambda plan: plan["chunks"][0].__setitem__("start_shard", True), "integer"),
        (lambda plan: plan["chunks"][0].__setitem__("start_shard", 1.0), "integer"),
        (lambda plan: plan["chunks"][0].__setitem__("max_shards", 0), "out of range"),
        (lambda plan: plan["chunks"][0].__setitem__("timeout_seconds", 1801), "out of range"),
        (lambda plan: plan["chunks"][0].__setitem__("timeout_seconds", 31), "out of range"),
        (lambda plan: plan.__setitem__("overall_seconds", 108001), "out of range"),
        (lambda plan: plan.__setitem__("pause_between_chunks_seconds", 29), "out of range"),
        (lambda plan: plan.__setitem__("minimum_free_gib", 149), "out of range"),
        (lambda plan: plan["chunks"][1].__setitem__("id", "c16"), "duplicate chunk id"),
        (lambda plan: plan["chunks"][1].__setitem__("start_shard", 17), "overlaps"),
        (lambda plan: plan["chunks"][1].__setitem__(
            "output_directory", plan["chunks"][0]["output_directory"]
        ), "overlaps"),
        (lambda plan: plan["chunks"][0].__setitem__(
            "output_directory", plan["state_directory"]
        ), "overlaps state"),
        (lambda plan: plan["chunks"][0].__setitem__("argv", "echo hi"), "shell string"),
        (lambda plan: plan["chunks"][0].__setitem__(
            "argv", [str(Path("/missing-ceres-batch-exec")), "-c", "pass"]
        ), "absent"),
        (lambda plan: plan["chunks"][0].__setitem__(
            "working_directory", str(Path(plan["state_directory"]) / "missing-cwd")
        ), "absent"),
        (lambda plan: plan.__setitem__("state_directory", "relative-state"), "absolute"),
        (lambda plan: plan["chunks"][0].__setitem__(
            "expected_completion", str(Path(plan["state_directory"]) / "completed.json")
        ), "inside the output directory"),
        (lambda plan: plan.__setitem__("chunks", []), "nonempty"),
        (lambda plan: plan["chunks"][0].__setitem__("id", "bad/id"), "identifier"),
        (lambda plan: plan["chunks"][0].__setitem__("expected_rows", 0), "out of range"),
        (lambda plan: plan.__setitem__("pinned_files", {}), "nonempty"),
    ],
)
def test_malformed_plans_are_rejected(
    tmp_path: Path, mutate: Any, match: str
) -> None:
    path, _digest, _worker = two_chunk_plan(tmp_path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    mutate(plan)
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        tool.load_plan(path, sha(path))


def test_nonfinite_plan_number_is_rejected(tmp_path: Path) -> None:
    path, _digest, _worker = two_chunk_plan(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        '"schema": 1', '"schema": Infinity', 1
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        tool.load_plan(path, sha(path))


@pytest.mark.usefixtures("_plenty_disk")
def test_successful_two_chunks_and_evidence(tmp_path: Path) -> None:
    path, digest, _worker = two_chunk_plan(tmp_path)
    before = time.monotonic()
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    assert code == 0
    state = tmp_path / "state"
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert manifest["plan_sha256"] == digest
    assert [item["id"] for item in manifest["completed_chunks"]] == ["c16", "c18"]
    for ident, start in (("c16", 16), ("c18", 18)):
        output = tmp_path / "outs" / ident
        completion = output / "completed.json"
        receipt = json.loads(
            (state / "chunks" / ident / "receipt.json").read_text(encoding="utf-8")
        )
        log = (state / "chunks" / ident / "stdout.log").read_text(encoding="utf-8")
        assert receipt["complete"] is True
        assert receipt["exit_code"] == 0
        assert receipt["completion_sha256"] == sha(completion)
        assert "chunk-ok" in log
        body = json.loads(completion.read_text(encoding="utf-8"))
        assert body["complete"] is True
        assert body["rows"] == 64
        assert body["selection"][0]["path"] == f"shard_{start:06d}.zarr"
    first = json.loads((state / "chunks" / "c16" / "receipt.json").read_text(encoding="utf-8"))
    second = json.loads((state / "chunks" / "c18" / "receipt.json").read_text(encoding="utf-8"))
    assert second["started_unix"] - first["ended_unix"] >= 29.5
    assert time.monotonic() - before >= 29.5
    assert (state / "actual_start.json").is_file()
    assert not (state / "failed.json").exists()
    assert not (state / "STOP").exists()


@pytest.mark.usefixtures("_plenty_disk")
def test_first_failure_prevents_next_chunk(tmp_path: Path) -> None:
    fail = write_script(tmp_path / "fail.py", WORKER_FAIL)
    ok = write_script(tmp_path / "ok.py", WORKER)
    chunks = [
        chunk_record(tmp_path, "c16", 16, 2, 32, 0, fail),
        chunk_record(tmp_path, "c18", 18, 2, 32, 0, ok),
    ]
    path, digest, _plan = make_plan(tmp_path, chunks)
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    assert code == 1
    state = tmp_path / "state"
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    failed = json.loads((state / "failed.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["completed_chunks"] == []
    assert failed["completed_chunks"] == []
    assert (state / "chunks" / "c16" / "receipt.json").is_file()
    assert "chunk-fail" in (state / "chunks" / "c16" / "stdout.log").read_text(
        encoding="utf-8"
    )
    assert not (state / "chunks" / "c18").exists()
    assert not (tmp_path / "outs" / "c18").exists()


@pytest.mark.parametrize(
    ("payload_mutator", "worker_source", "match"),
    [
        (None, WORKER_MISSING, "missing"),
        (
            lambda body: body.__setitem__("complete", False),
            WORKER,
            "complete=true",
        ),
        (
            lambda body: body.__setitem__("rows", 8),
            WORKER,
            "rows differ",
        ),
        (
            lambda body: body["collection_counts"].__setitem__("padding_rows", 12),
            WORKER,
            "padding_rows",
        ),
        (
            lambda body: body.__setitem__("new_shards", 0),
            WORKER,
            "cached directory",
        ),
        (
            lambda body: body["selection"][0].__setitem__("path", "shard_000099.zarr"),
            WORKER,
            "selection path",
        ),
    ],
)
@pytest.mark.usefixtures("_plenty_disk")
def test_zero_exit_missing_or_mismatched_completion_is_refused(
    tmp_path: Path,
    payload_mutator: Any,
    worker_source: str,
    match: str,
) -> None:
    worker = write_script(tmp_path / "worker.py", worker_source)
    payload = completion_payload(16, 2, 32, 0)
    if payload_mutator is not None:
        payload_mutator(payload)
    ok = write_script(tmp_path / "ok.py", WORKER)
    chunks = [
        chunk_record(tmp_path, "c16", 16, 2, 32, 0, worker, payload=payload),
        chunk_record(tmp_path, "c18", 18, 2, 32, 0, ok),
    ]
    path, digest, _plan = make_plan(tmp_path, chunks)
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    assert code == 1
    state = tmp_path / "state"
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["completed_chunks"] == []
    stderr_receipt = json.loads(
        (state / "chunks" / "c16" / "receipt.json").read_text(encoding="utf-8")
    )
    assert stderr_receipt["complete"] is False
    assert match in stderr_receipt["error"]
    assert not (state / "chunks" / "c18").exists()
    assert not (tmp_path / "outs" / "c18").exists()


@pytest.mark.usefixtures("_plenty_disk")
def test_timeout_kills_hung_process_group_and_skips_rest(
    tmp_path: Path
) -> None:
    hang = write_script(tmp_path / "hang.py", WORKER_HANG_KILL)
    ok = write_script(tmp_path / "ok.py", WORKER)
    ready = tmp_path / "ready"
    hang_chunk = chunk_record(tmp_path, "c16", 16, 2, 32, 0, hang, timeout_seconds=33)
    hang_chunk["argv"] = [sys.executable, str(hang), str(ready)]
    chunks = [hang_chunk, chunk_record(tmp_path, "c18", 18, 2, 32, 0, ok)]
    path, digest, _plan = make_plan(
        tmp_path, chunks, overall_seconds=90
    )
    started = time.monotonic()
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    elapsed = time.monotonic() - started
    assert code == 1
    assert elapsed < 45
    state = tmp_path / "state"
    receipt = json.loads((state / "chunks" / "c16" / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["complete"] is False
    assert "timeout" in receipt["error"]
    assert receipt.get("pid")
    assert not Path(f"/proc/{receipt['pid']}").exists()
    assert not (state / "chunks" / "c18").exists()
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["completed_chunks"] == []


@pytest.mark.usefixtures("_plenty_disk")
def test_stop_kills_running_chunk_and_skips_rest(
    tmp_path: Path
) -> None:
    hang = write_script(tmp_path / "hang.py", WORKER_HANG)
    ok = write_script(tmp_path / "ok.py", WORKER)
    ready = tmp_path / "ready"
    hang_chunk = chunk_record(tmp_path, "c16", 16, 2, 32, 0, hang)
    hang_chunk["argv"] = [sys.executable, str(hang), str(ready)]
    chunks = [hang_chunk, chunk_record(tmp_path, "c18", 18, 2, 32, 0, ok)]
    path, digest, _plan = make_plan(tmp_path, chunks)
    plan = tool.load_plan(path, digest)
    receipt_path = tmp_path / "state" / "chunks" / "c16" / "receipt.json"

    def write_stop() -> None:
        wait_file(ready, 10)
        wait_file(receipt_path, 5)
        (tmp_path / "state" / "STOP").write_text("stop\n", encoding="utf-8")

    stopper = threading.Thread(target=write_stop)
    stopper.start()
    with pytest.raises(ValueError, match="STOP"):
        tool.execute(plan)
    stopper.join(timeout=20)
    assert not stopper.is_alive()
    pid = json.loads(receipt_path.read_text(encoding="utf-8"))["pid"]
    assert not Path(f"/proc/{pid}").exists()
    assert not (tmp_path / "state" / "chunks" / "c18").exists()
    manifest = json.loads((tmp_path / "state" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["completed_chunks"] == []


def test_plan_hash_drift_is_rejected(tmp_path: Path) -> None:
    path, digest, _worker = two_chunk_plan(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="plan hash"):
        tool.load_plan(path, digest)
    assert not (tmp_path / "state").exists()


@pytest.mark.usefixtures("_plenty_disk")
def test_pinned_file_drift_before_execute_is_rejected(
    tmp_path: Path
) -> None:
    path, digest, _worker = two_chunk_plan(tmp_path)
    (tmp_path / "operator.json").write_text("changed", encoding="utf-8")
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    assert code == 1
    assert not (tmp_path / "state" / "actual_start.json").exists()
    assert not (tmp_path / "outs" / "c16").exists()


@pytest.mark.usefixtures("_plenty_disk")
def test_pinned_file_drift_after_first_chunk_stops_queue(
    tmp_path: Path
) -> None:
    mutate = write_script(tmp_path / "mutate.py", WORKER_MUTATE_PIN)
    ok = write_script(tmp_path / "ok.py", WORKER)
    pin = tmp_path / "operator.json"
    pin.write_text('{"role":"operator"}', encoding="utf-8")
    first = chunk_record(tmp_path, "c16", 16, 2, 32, 0, mutate, extra_argv=[str(pin)])
    second = chunk_record(tmp_path, "c18", 18, 2, 32, 0, ok)
    path, digest, _plan = make_plan(tmp_path, [first, second])
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    assert code == 1
    state = tmp_path / "state"
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert [item["id"] for item in manifest["completed_chunks"]] == ["c16"]
    assert manifest["completed_chunks"][0]["completion_sha256"] == sha(
        tmp_path / "outs" / "c16" / "completed.json"
    )
    assert not (state / "chunks" / "c18").exists()
    assert not (tmp_path / "outs" / "c18").exists()


@pytest.mark.usefixtures("_plenty_disk")
def test_execute_refuses_existing_output_as_progress(
    tmp_path: Path
) -> None:
    path, digest, _worker = two_chunk_plan(tmp_path)
    (tmp_path / "outs" / "c16").mkdir(parents=True)
    with pytest.raises(ValueError, match="not fresh"):
        tool.load_plan(path, digest)


@pytest.mark.usefixtures("_plenty_disk")
def test_expected_tail_padding_must_match_completion(
    tmp_path: Path
) -> None:
    worker = write_script(tmp_path / "worker.py", WORKER)
    chunks = [chunk_record(tmp_path, "tail", 2308, 1, 20, 12, worker)]
    path, digest, _plan = make_plan(tmp_path, chunks)
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    assert code == 0
    manifest = json.loads((tmp_path / "state" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    body = json.loads((tmp_path / "outs" / "tail" / "completed.json").read_text(encoding="utf-8"))
    assert body["collection_counts"]["padding_rows"] == 12
    assert body["collection_counts"]["calls"] == 1
    assert body["selection"] == [{"path": "shard_002308.zarr", "rows": 20}]


@pytest.mark.usefixtures("_plenty_disk")
def test_last_chunk_pin_mutation_refuses_complete(
    tmp_path: Path
) -> None:
    mutate = write_script(tmp_path / "mutate.py", WORKER_MUTATE_PIN)
    pin = tmp_path / "operator.json"
    pin.write_text('{"role":"operator"}', encoding="utf-8")
    chunks = [
        chunk_record(tmp_path, "c16", 16, 2, 32, 0, mutate, extra_argv=[str(pin)])
    ]
    path, digest, _plan = make_plan(tmp_path, chunks)
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    assert code == 1
    state = tmp_path / "state"
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    failed = json.loads((state / "failed.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert [item["id"] for item in manifest["completed_chunks"]] == ["c16"]
    assert failed["completed_chunks"] == manifest["completed_chunks"]
    assert manifest["completed_chunks"][0]["completion_sha256"] == sha(
        tmp_path / "outs" / "c16" / "completed.json"
    )


@pytest.mark.usefixtures("_plenty_disk")
def test_second_signal_during_failure_receipt_keeps_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest, _worker = two_chunk_plan(tmp_path)
    real_write = tool.write_exclusive

    def interrupted_chunk(*_args: Any, **_kwargs: Any) -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("first signal must interrupt")

    def interrupted_write(path: Path, value: Any) -> None:
        if path.name == "failed.json":
            os.kill(os.getpid(), signal.SIGTERM)
        real_write(path, value)

    monkeypatch.setattr(tool, "run_chunk", interrupted_chunk)
    monkeypatch.setattr(tool, "write_exclusive", interrupted_write)
    before = signal.getsignal(signal.SIGTERM)
    assert tool.main(["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]) == 1
    assert signal.getsignal(signal.SIGTERM) == before
    for name in ("failed.json", "manifest.json"):
        receipt = json.loads((tmp_path / "state" / name).read_text())
        assert receipt["status"] == "FAILED"
        assert receipt["completed_chunks"] == []


def test_external_sigterm_cleans_owned_child_and_releases_lock(
    tmp_path: Path,
) -> None:
    locker = write_script(tmp_path / "lock.py", WORKER_LOCK)
    runner = write_script(tmp_path / "driver_runner.py", DRIVER_RUNNER)
    lock_path = tmp_path / "dummy.lock"
    ready = tmp_path / "ready"
    hang = chunk_record(tmp_path, "c16", 16, 2, 32, 0, locker, timeout_seconds=1800)
    hang["argv"] = [sys.executable, str(locker), str(lock_path), str(ready)]
    path, digest, _plan = make_plan(tmp_path, [hang], overall_seconds=120)
    driver = subprocess.Popen(
        [sys.executable, str(runner), str(REPO), "--plan", str(path),
         "--expected-plan-sha256", digest, "--execute"],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_file(ready, 10)
        child_pid = int(ready.read_text(encoding="utf-8"))
        os.kill(driver.pid, signal.SIGTERM)
        assert driver.wait(timeout=20) == 1
        assert not Path(f"/proc/{child_pid}").exists()
        assert lock_is_free(lock_path)
        manifest = json.loads(
            (tmp_path / "state" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "FAILED"
        assert manifest["completed_chunks"] == []
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait(timeout=5)


@pytest.mark.usefixtures("_plenty_disk")
def test_stop_kills_same_group_descendant_after_leader_exits(
    tmp_path: Path
) -> None:
    wrapper = write_script(tmp_path / "wrapper.py", WORKER_LEADER_EXITS)
    ok = write_script(tmp_path / "ok.py", WORKER)
    lock_path = tmp_path / "dummy.lock"
    ready = tmp_path / "ready"
    hang = chunk_record(tmp_path, "c16", 16, 2, 32, 0, wrapper, timeout_seconds=33)
    hang["argv"] = [sys.executable, str(wrapper), str(lock_path), str(ready)]
    chunks = [hang, chunk_record(tmp_path, "c18", 18, 2, 32, 0, ok)]
    path, digest, _plan = make_plan(tmp_path, chunks, overall_seconds=90)
    started = time.monotonic()
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    assert code == 1
    assert time.monotonic() - started < 45
    descendant = int(ready.read_text(encoding="utf-8"))
    assert not Path(f"/proc/{descendant}").exists()
    assert lock_is_free(lock_path)
    assert not (tmp_path / "state" / "chunks" / "c18").exists()
    receipt = json.loads(
        (tmp_path / "state" / "chunks" / "c16" / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["complete"] is False
    assert "timeout" in receipt["error"] or "unreapable" in receipt["error"]


def test_unreapable_group_fails_with_pid_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(tool, "process_group_present", lambda _pgid: True)
    monkeypatch.setattr(os, "killpg", lambda _pgid, _sig: None)

    class Fake:
        pid = 4242

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> None:
            if timeout:
                now[0] += timeout
            raise subprocess.TimeoutExpired(["fake"], timeout or 0)

    with pytest.raises(ValueError, match="unreapable") as info:
        tool.stop_process_group(Fake(), 1040.0)  # pyright: ignore[reportArgumentType] -- deliberate process double
    message = str(info.value)
    assert "pgid=4242" in message
    assert "leader_pid=4242" in message
    assert now[0] <= 1040.0


def test_first_external_signal_during_timeout_cleanup_cannot_skip_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1000.0]
    present = [True]
    delivered = []
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(tool, "process_group_present", lambda _pgid: present[0])

    def killpg(_pgid: int, sig: int) -> None:
        delivered.append(sig)
        if sig == signal.SIGTERM:
            os.kill(os.getpid(), signal.SIGTERM)
        if sig == signal.SIGKILL:
            present[0] = False

    monkeypatch.setattr(os, "killpg", killpg)

    class ExitedLeader:
        pid = 4242

        def poll(self) -> int:
            return 0

    before = signal.getsignal(signal.SIGTERM)
    with tool.execute_termination_signals():
        active = signal.getsignal(signal.SIGTERM)
        tool.stop_process_group(ExitedLeader(), 1030.0)  # pyright: ignore[reportArgumentType] -- deliberate process double
        assert signal.getsignal(signal.SIGTERM) == active
    assert signal.getsignal(signal.SIGTERM) == before
    assert delivered == [signal.SIGTERM, signal.SIGKILL]
    assert now[0] < 1030.0


@pytest.mark.usefixtures("_plenty_disk")
def test_successful_leader_cannot_leave_lock_holding_descendant(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "dummy.lock"
    ready = tmp_path / "ready"
    worker_code = WORKER + r'''
import os, signal, time, fcntl
child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    lock = Path(sys.argv[4]).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    Path(sys.argv[5]).write_text(str(os.getpid()))
    time.sleep(600)
    os._exit(0)
while not Path(sys.argv[5]).exists():
    time.sleep(.01)
'''
    worker = write_script(tmp_path / "orphan.py", worker_code)
    ok = write_script(tmp_path / "ok.py", WORKER)
    first = chunk_record(tmp_path, "c16", 16, 2, 32, 0, worker,
                         extra_argv=[str(lock_path), str(ready)])
    second = chunk_record(tmp_path, "c18", 18, 2, 32, 0, ok)
    path, digest, _plan = make_plan(tmp_path, [first, second])
    try:
        assert tool.main(["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]) == 1
        assert lock_is_free(lock_path)
        receipt = json.loads((tmp_path / "state/chunks/c16/receipt.json").read_text())
        assert receipt["complete"] is False
        assert not (tmp_path / "state/chunks/c18").exists()
    finally:
        if ready.exists():
            try:
                os.kill(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.usefixtures("_plenty_disk")
def test_unreapable_cleanup_does_not_queue_next_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_proc: Any, _deadline: float) -> None:
        raise ValueError(
            "owned process group unreapable pgid=9 leader_pid=9 group_present=True"
        )

    monkeypatch.setattr(tool, "stop_process_group", boom)
    hang = write_script(tmp_path / "hang.py", WORKER_HANG)
    ok = write_script(tmp_path / "ok.py", WORKER)
    ready = tmp_path / "ready"
    first = chunk_record(tmp_path, "c16", 16, 2, 32, 0, hang, timeout_seconds=33)
    first["argv"] = [sys.executable, str(hang), str(ready)]
    chunks = [first, chunk_record(tmp_path, "c18", 18, 2, 32, 0, ok)]
    path, digest, _plan = make_plan(tmp_path, chunks, overall_seconds=90)
    try:
        code = tool.main(
            ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
        )
        assert code == 1
        failed = json.loads((tmp_path / "state" / "failed.json").read_text(encoding="utf-8"))
        assert "unreapable" in failed["error"]
        assert failed["completed_chunks"] == []
        assert not (tmp_path / "state" / "chunks" / "c18").exists()
    finally:
        receipt_path = tmp_path / "state" / "chunks" / "c16" / "receipt.json"
        if receipt_path.exists():
            pid = json.loads(receipt_path.read_text(encoding="utf-8")).get("pid")
            if isinstance(pid, int) and Path(f"/proc/{pid}").exists():
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


@pytest.mark.usefixtures("_plenty_disk")
def test_ceres_invocation_discovers_parent_completed_json(
    tmp_path: Path
) -> None:
    worker = write_script(tmp_path / "ceres.py", FAKE_CERES)
    path, digest, _plan = make_plan(tmp_path, [ceres_chunk(tmp_path, "c16", worker, "ok")])
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    assert code == 0
    state = tmp_path / "state"
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    completion = Path(manifest["completed_chunks"][0]["completion_path"])
    assert completion.name == "completed.json"
    assert completion.parent.parent.name == "invocations"
    body = json.loads(completion.read_text(encoding="utf-8"))
    assert body["complete"] is True
    assert isinstance(body["ended_unix"], (int, float))
    receipt = json.loads(
        (state / "chunks" / "c16" / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["completion_sha256"] == sha(completion)


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("missing_parent", "child_completed.json"),
        ("ambiguous", "ambiguous"),
        ("stale", "ended_unix"),
    ],
)
@pytest.mark.usefixtures("_plenty_disk")
def test_ceres_invocation_rejects_missing_ambiguous_or_stale(
    tmp_path: Path, mode: str, match: str
) -> None:
    worker = write_script(tmp_path / "ceres.py", FAKE_CERES)
    ok = write_script(tmp_path / "ok.py", WORKER)
    chunks = [
        ceres_chunk(tmp_path, "c16", worker, mode),
        chunk_record(tmp_path, "c18", 18, 2, 32, 0, ok),
    ]
    path, digest, _plan = make_plan(tmp_path, chunks)
    code = tool.main(
        ["--plan", str(path), "--expected-plan-sha256", digest, "--execute"]
    )
    assert code == 1
    receipt = json.loads(
        (tmp_path / "state" / "chunks" / "c16" / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["complete"] is False
    assert match in receipt["error"]
    assert not (tmp_path / "state" / "chunks" / "c18").exists()
    manifest = json.loads((tmp_path / "state" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["completed_chunks"] == []
