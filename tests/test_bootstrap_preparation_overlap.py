"""Ownership tests use real processes; teacher computation is irrelevant here."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, cast

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


@pytest.mark.parametrize("outcome", ["return", "raise", "leader_exit"])
def test_queued_return_and_error_reap_owned_writer_groups(tmp_path, outcome):
    import fcntl

    runner = tmp_path / "prep.py"
    ready = tmp_path / "writer.pid"
    child_code = (
        "import os,time; from pathlib import Path; "
        + ("pid=os.fork(); os._exit(0) if pid else None; " if outcome == "leader_exit" else "")
        + f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    runner.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "def main():\n"
        f" p=subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True)\n"
        " end=time.monotonic()+5\n"
        f" while not Path({str(ready)!r}).exists():\n"
        "  assert time.monotonic()<end\n"
        "  time.sleep(.01)\n"
        + (" p.wait(timeout=5)\n" if outcome == "leader_exit" else "")
        + (" raise RuntimeError('coordinator failed')\n" if outcome == "raise" else "")
    )
    settings = {**config(tmp_path), "prep_runner": {"path": str(runner)}}
    try:
        with tool.stage_lock(tmp_path, settings, time.monotonic() + 60, queued=True) as fd:
            assert fd is not None
            if outcome == "raise":
                with pytest.raises(RuntimeError, match="coordinator failed"):
                    tool.run_queued(settings, [sys.executable, str(runner)], fd)
            else:
                tool.run_queued(settings, [sys.executable, str(runner)], fd)
        # A live writer keeps an inherited descriptor, including after its
        # new-session leader exits. Acquiring a distinct description proves
        # cleanup finished before ownership became available.
        with (tmp_path / "ownership.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if ready.exists():
            try:
                os.kill(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cleanup_failure_cannot_unlock_surviving_inherited_writer(tmp_path, monkeypatch):
    import fcntl

    runner = tmp_path / "prep.py"
    ready = tmp_path / "writer.pid"
    runner.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "def main():\n"
        " subprocess.Popen([sys.executable, '-c', "
        + repr(f"import os,time; from pathlib import Path; Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)")
        + "],start_new_session=True)\n"
        " end=time.monotonic()+5\n"
        f" while not Path({str(ready)!r}).exists():\n"
        "  assert time.monotonic()<end\n"
        "  time.sleep(.01)\n"
    )
    settings = {**config(tmp_path), "prep_runner": {"path": str(runner)}}
    children = []
    terminate = tool.terminate

    def failed_cleanup(child):
        children.append(child)
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(tool, "terminate", failed_cleanup)

    def run_under_lock():
        with tool.stage_lock(tmp_path, settings, time.monotonic() + 60, queued=True) as fd:
            assert fd is not None
            tool.run_queued(settings, [sys.executable, str(runner)], fd)

    try:
        with pytest.raises(RuntimeError, match="child cleanup failed"):
            run_under_lock()
        with (
            (tmp_path / "ownership.lock").open("a") as lock,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        for child in children:
            terminate(child)


def valid_bound_inputs(tmp_path) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = {"path": str(tmp_path / "preparation" / "run.py"), "sha256": "a" * 64}
    settings: dict[str, Any] = {
        "python": sys.executable, "prep_runner": runner,
        "rss_cap_gib": 32, "memory_floor_gib": 32, "disk_floor_gib": 80,
        "stage_seconds": 1, "handoff_seconds": 2, "sidecar_seconds": 90,
        "queued_seconds": 102,
        "stop_paths": [str(tmp_path / "preparation" / "STOP"), str(tmp_path / "STOP")],
        "disk_paths": [str(tmp_path)],
    }
    plan: dict[str, Any] = {
        "python": sys.executable, "pins": [runner], "max_seconds": 100,
        "process_memory_cap_gib": 32, "available_memory_floor_gib": 32,
        "disk_floor_gib": 80, "cohorts": [{"output": str(tmp_path / "targets")}],
    }
    return settings, plan


@pytest.mark.parametrize("name", [
    "stage_seconds", "handoff_seconds", "sidecar_seconds", "queued_seconds",
    "rss_cap_gib", "memory_floor_gib", "disk_floor_gib",
])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), -1, True])
def test_config_requires_finite_resource_and_time_bounds(tmp_path, name, value):
    settings, plan = valid_bound_inputs(tmp_path)
    tool.validate_config(settings, plan)
    settings[name] = value
    with pytest.raises(RuntimeError, match=name + " must be finite"):
        tool.validate_config(settings, plan)


def test_infinite_handoff_and_queued_allowance_are_refused(tmp_path):
    settings, plan = valid_bound_inputs(tmp_path)
    settings.update(handoff_seconds=float("inf"), queued_seconds=float("inf"))
    with pytest.raises(RuntimeError, match="handoff_seconds must be finite"):
        tool.validate_config(settings, plan)


@pytest.mark.parametrize("name", [
    "max_seconds", "process_memory_cap_gib", "available_memory_floor_gib", "disk_floor_gib",
])
def test_frozen_plan_requires_finite_original_bounds(tmp_path, name):
    settings, plan = valid_bound_inputs(tmp_path)
    plan[name] = float("inf")
    with pytest.raises(RuntimeError, match="plan." + name + " must be finite"):
        tool.validate_config(settings, plan)


def test_cleanup_never_signals_reused_group_leader(monkeypatch):
    from types import SimpleNamespace

    signals = []
    waits = []
    # Deliberate structural test double: no process is launched by this test.
    child = cast(tool.subprocess.Popen, cast(object, SimpleNamespace(
        pid=12345, _preparation_starttime=111,
        poll=lambda: 0, wait=lambda: waits.append("reaped"),
    )))
    monkeypatch.setattr(tool, "_process_identity", lambda *_: (1, 222))
    monkeypatch.setattr(tool.os, "killpg", lambda *args: signals.append(args))
    tool.terminate(child)
    assert signals == []
    assert waits == ["reaped"]


def test_cleanup_rechecks_identity_before_kill(monkeypatch):
    from types import SimpleNamespace

    signals = []
    identity = [111]
    # Deliberate structural test double: no process is launched by this test.
    child = cast(tool.subprocess.Popen, cast(object, SimpleNamespace(
        pid=12345, _preparation_starttime=111, poll=lambda: 0, wait=lambda: None,
    )))
    monkeypatch.setattr(tool, "_process_identity", lambda *_: (1, identity[0]))

    def signal_group(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGTERM:
            identity[0] = 222

    monkeypatch.setattr(tool.os, "killpg", signal_group)
    tool.terminate(child)
    assert (12345, signal.SIGTERM) in signals
    assert (12345, signal.SIGKILL) not in signals


def test_probe_output_uses_current_operator_shared_root(tmp_path, monkeypatch):
    from scripts import bootstrap_preparation_probe as probe

    home = tmp_path / "operator"
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    probe.require_shared_artifact_output(home / "chess-artifacts" / "probe")
    with pytest.raises(RuntimeError, match="shared artifact output"):
        probe.require_shared_artifact_output(home / "chess-artifacts-other" / "probe")


def test_probe_output_resolves_shared_root_symlink(tmp_path, monkeypatch):
    from scripts import bootstrap_preparation_probe as probe

    home = tmp_path / "operator"
    home.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    (home / "chess-artifacts").symlink_to(shared, target_is_directory=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    probe.require_shared_artifact_output(shared / "probe")
    with pytest.raises(RuntimeError, match="shared artifact output"):
        probe.require_shared_artifact_output(tmp_path / "outside" / "probe")
