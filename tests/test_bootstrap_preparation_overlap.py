"""Ownership tests use real processes; teacher computation is irrelevant here."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from scripts import bootstrap_preparation_overlap as tool


def config(tmp):
    return {
        "stop_paths": [str(tmp / "STOP")],
        "memory_floor_gib": 0,
        "disk_paths": [str(tmp)],
        "disk_floor_gib": 0,
        "rss_cap_gib": 1,
        "runtime": str(tmp),
        "env": {},
    }


def holder(state, settings, entered, release):
    with tool.stage_lock(
        Path(state), settings, time.monotonic() + 15, queued=False
    ) as fd:
        assert fd is not None
        entered.set()
        assert release.wait(10)


def queued(state, settings, entered):
    with tool.stage_lock(
        Path(state), settings, time.monotonic() + 15, queued=True
    ) as fd:
        assert fd is not None
        entered.set()


def test_real_stage_handoff_and_restart_refused(tmp_path):
    ctx = mp.get_context("fork")
    settings = config(tmp_path)
    active, release, admitted = [ctx.Event() for _ in range(3)]
    a = ctx.Process(target=holder, args=(str(tmp_path), settings, active, release))
    a.start()
    try:
        assert active.wait(5)
        b = ctx.Process(target=queued, args=(str(tmp_path), settings, admitted))
        b.start()
        try:
            until = time.monotonic() + 5
            while (
                not (tmp_path / "handoff.request").exists() and time.monotonic() < until
            ):
                time.sleep(0.01)
            assert (tmp_path / "handoff.request").exists()
            assert not admitted.is_set()
            release.set()
            a.join(5)
            b.join(5)
            assert a.exitcode == b.exitcode == 0
            assert admitted.is_set()
            with tool.stage_lock(
                tmp_path, settings, time.monotonic() + 1, queued=False
            ) as fd:
                assert fd is None
        finally:
            if b.is_alive():
                b.kill()
                b.join()
    finally:
        if a.is_alive():
            a.kill()
            a.join()


def crash_parent(state, settings):
    with tool.stage_lock(
        Path(state), settings, time.monotonic() + 30, queued=False
    ) as fd:
        assert fd is not None
        code = (
            "import time,os; from pathlib import Path; "
            f"Path({str(Path(state) / 'child.pid')!r}).write_text(str(os.getpid())); time.sleep(20)"
        )
        tool.run_child(
            settings,
            [sys.executable, "-c", code],
            time.monotonic() + 25,
            fd,
            Path(state) / "child.log",
        )


def test_killed_sidecar_parent_cannot_release_live_writer_lock(tmp_path):
    import fcntl

    ctx = mp.get_context("fork")
    p = ctx.Process(target=crash_parent, args=(str(tmp_path), config(tmp_path)))
    p.start()
    child = None
    try:
        until = time.monotonic() + 5
        while not (tmp_path / "child.pid").exists() and time.monotonic() < until:
            time.sleep(0.01)
        child = int((tmp_path / "child.pid").read_text())
        p.kill()
        p.join(5)
        with (
            (tmp_path / "ownership.lock").open("a") as f,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child:
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if p.is_alive():
            p.kill()
            p.join()


def test_stop_during_worker_cleans_child(tmp_path):
    settings = config(tmp_path)
    with tool.stage_lock(tmp_path, settings, time.monotonic() + 10, queued=False) as fd:
        assert fd is not None
        code = f"from pathlib import Path; import time; Path({str(tmp_path / 'STOP')!r}).touch(); time.sleep(20)"
        with pytest.raises(RuntimeError, match="STOP"):
            tool.run_child(
                settings,
                [sys.executable, "-c", code],
                time.monotonic() + 10,
                fd,
                tmp_path / "log",
            )


def test_registration_requires_current_exact_wrapper(tmp_path):
    source = str(Path(tool.__file__).resolve())
    cp = tmp_path / "config.json"
    cp.write_text("{}")
    ref = {"path": str(cp), "sha256": tool.sha(cp)}
    desc = {
        "argv": [
            sys.executable,
            source,
            "--mode",
            "queued",
            "--config",
            str(cp),
            "--sha256",
            ref["sha256"],
        ],
        "pins": [ref, {"path": source, "sha256": tool.sha(source)}],
    }
    dp = tmp_path / "command.json"
    dp.write_text(json.dumps(desc))
    q = tmp_path / "queue.json"
    item = {
        "id": "prep",
        "status": "queued",
        "command_file": str(dp),
        "command_sha256": tool.sha(dp),
    }
    q.write_text(json.dumps({"items": [item]}))
    c = {"queue_file": str(q), "queue_id": "prep", "python": sys.executable}
    tool.verify_registration(c, ref)
    item["status"] = "running"
    q.write_text(json.dumps({"items": [item]}))
    with pytest.raises(RuntimeError, match="no longer queued"):
        tool.verify_registration(c, ref)


def test_schedule_only_builds_pinned_teachers_and_smallest_first():
    p = {
        "cohorts": [
            {"rows": 100, "ceres_manifest": {"sha256": "a"}},
            {"rows": 10, "ceres_manifest": {"sha256": "b"}},
            {"rows": 1, "ceres_manifest": {}},
        ]
    }
    assert tool.stages(p) == [
        ("seal", 1),
        ("cohort", 1),
        ("seal", 0),
        ("cohort", 0),
        ("seal", 2),
    ]


def queued_import_parent(state, settings, runner):
    settings = {**settings, "prep_runner": {"path": runner}}
    with tool.stage_lock(
        Path(state), settings, time.monotonic() + 30, queued=True
    ) as fd:
        assert fd is not None
        tool.run_queued(settings, [sys.executable, runner], fd)


def test_queued_new_session_worker_retains_lock_after_parent_sigkill(tmp_path):
    import fcntl

    runner = tmp_path / "prep.py"
    # Models the actual original run_child: a new-session worker followed by wait.
    runner.write_text(
        "import subprocess,sys\nfrom pathlib import Path\n"
        "def main():\n"
        ' p=subprocess.Popen([sys.executable,"-c",'
        + repr(
            "import time,os; from pathlib import Path; Path("
            + repr(str(tmp_path / "queued-child.pid"))
            + ").write_text(str(os.getpid())); time.sleep(20)"
        )
        + "],start_new_session=True)\n p.wait()\n"
    )
    ctx = mp.get_context("fork")
    p = ctx.Process(
        target=queued_import_parent, args=(str(tmp_path), config(tmp_path), str(runner))
    )
    p.start()
    child = None
    try:
        until = time.monotonic() + 5
        while not (tmp_path / "queued-child.pid").exists() and time.monotonic() < until:
            time.sleep(0.01)
        child = int((tmp_path / "queued-child.pid").read_text())
        p.kill()
        p.join(5)
        with (
            (tmp_path / "ownership.lock").open("a") as f,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child:
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if p.is_alive():
            p.kill()
            p.join()


def test_probe_import_does_not_capture_frozen_scripts_namespace(tmp_path):
    import subprocess

    probe = Path(tool.__file__).with_name("bootstrap_preparation_probe.py")
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'scripts' not in sys.modules"
    subprocess.run([sys.executable, "-c", code, str(probe)], cwd=tmp_path, check=True)


def test_build_limit_defers_large_cohort_but_keeps_its_seal():
    plan = {
        "cohorts": [
            {"rows": 100, "ceres_manifest": {"sha256": "a"}},
            {"rows": 10, "ceres_manifest": {"sha256": "b"}},
            {"rows": 5, "ceres_manifest": {}},
        ]
    }
    assert tool.stages(plan, 20) == [
        ("seal", 1),
        ("cohort", 1),
        ("seal", 2),
        ("seal", 0),
    ]


def process_fixture(root: Path, pid: int, parent: int, pages: int) -> None:
    folder = root / str(pid)
    folder.mkdir()
    fields = ["S", str(parent), *(["0"] * 17), str(pid * 100)]
    (folder / "stat").write_text(f"{pid} (worker with spaces) " + " ".join(fields))
    (folder / "statm").write_text(f"100 {pages} 0 0 0 0 0")


def test_procfs_rss_includes_descendants_and_excludes_unrelated(tmp_path):
    process_fixture(tmp_path, 10, 1, 3)
    process_fixture(tmp_path, 11, 10, 5)
    process_fixture(tmp_path, 12, 11, 7)
    process_fixture(tmp_path, 20, 1, 1000)
    assert tool.process_tree_rss_bytes(10, tmp_path) == 15 * os.sysconf("SC_PAGE_SIZE")
    assert tool.process_tree_rss_bytes(99, tmp_path) == 0


def test_procfs_rss_tolerates_exit_but_rejects_malformed_evidence(tmp_path):
    process_fixture(tmp_path, 10, 1, 3)
    process_fixture(tmp_path, 11, 10, 5)
    (tmp_path / "11" / "statm").unlink()
    assert tool.process_tree_rss_bytes(10, tmp_path) == 3 * os.sysconf("SC_PAGE_SIZE")
    (tmp_path / "10" / "statm").write_text("malformed")
    with pytest.raises(RuntimeError, match="malformed process RSS"):
        tool.process_tree_rss_bytes(10, tmp_path)


def test_procfs_rss_rejects_unreadable_evidence(tmp_path, monkeypatch):
    process_fixture(tmp_path, 10, 1, 3)
    original = Path.read_text

    def denied(path, *args, **kwargs):
        if path.name == "statm":
            raise PermissionError("resource read denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError, match="resource read denied"):
        tool.process_tree_rss_bytes(10, tmp_path)


def test_procfs_rss_does_not_charge_reused_pid(tmp_path, monkeypatch):
    process_fixture(tmp_path, 10, 1, 3)
    original = tool._process_identity
    calls = 0

    def identity(pid, root):
        nonlocal calls
        calls += 1
        return (1, 9999) if calls == 3 else original(pid, root)

    monkeypatch.setattr(tool, "_process_identity", identity)
    assert tool.process_tree_rss_bytes(10, tmp_path) == 0


def test_procfs_available_memory_requires_valid_evidence(tmp_path):
    info = tmp_path / "meminfo"
    info.write_text("MemTotal: 2048 kB\nMemAvailable: 1024 kB\n")
    assert tool.available_memory_bytes(tmp_path) == 1024 * 1024
    info.write_text("MemTotal: 2048 kB\n")
    with pytest.raises(RuntimeError, match="MemAvailable missing"):
        tool.available_memory_bytes(tmp_path)
    info.write_text("MemAvailable: 1024 MB\n")
    with pytest.raises(RuntimeError, match="malformed MemAvailable"):
        tool.available_memory_bytes(tmp_path)


def test_guard_enforces_available_memory_floor(tmp_path, monkeypatch):
    settings = {**config(tmp_path), "memory_floor_gib": 1}
    monkeypatch.setattr(tool, "available_memory_bytes", lambda: tool.GIB - 1)
    with pytest.raises(RuntimeError, match="available memory floor"):
        tool.guard(settings, time.monotonic() + 10)


def test_run_child_enforces_rss_cap_and_cleans_worker(tmp_path, monkeypatch):
    settings = config(tmp_path)
    monkeypatch.setattr(tool, "process_tree_rss_bytes", lambda _: 2 * tool.GIB)
    with tool.stage_lock(tmp_path, settings, time.monotonic() + 10, queued=False) as fd:
        assert fd is not None
        with pytest.raises(RuntimeError, match="process RSS cap"):
            tool.run_child(
                settings,
                [sys.executable, "-c", "import time; time.sleep(20)"],
                time.monotonic() + 10,
                fd,
                tmp_path / "rss.log",
            )
    with (tmp_path / "ownership.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
