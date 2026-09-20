from __future__ import annotations
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SPEC = importlib.util.spec_from_file_location(
    "combined", Path(__file__).parents[1] / "scripts/factorial_prepare_and_train.py"
)
assert SPEC
assert SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def wait_file(path):
    deadline = time.monotonic() + 8
    while not path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.02)


@pytest.mark.parametrize("failed_role", ["preparation", "training"])
def test_either_failure_cleans_new_session_worker(tmp_path, failed_role):
    pid_file = tmp_path / "worker.pid"
    code = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(120)"
    running = subprocess.Popen(
        [sys.executable, "-c", code, str(pid_file)], start_new_session=True
    )
    failed = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    try:
        wait_file(pid_file)
        assert failed.wait() == 7
        roles = {
            failed_role: failed,
            ("training" if failed_role == "preparation" else "preparation"): running,
        }
        module.cleanup(list(roles.values()), grace=0.1)
        p = module.psutil.Process(int(pid_file.read_text()))
        assert not p.is_running() or p.status() == module.psutil.STATUS_ZOMBIE
    except module.psutil.NoSuchProcess:
        pass
    finally:
        module.cleanup([running, failed], grace=0.1)


def test_inherited_ownership_survives_supervisor_sigkill(tmp_path):
    lock = tmp_path / "owner.lock"
    pid_file = tmp_path / "worker.pid"
    code = """import subprocess,sys,fcntl,time
from pathlib import Path
f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX)
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True,pass_fds=(f.fileno(),))
Path(sys.argv[2]).write_text(str(p.pid));time.sleep(120)
"""
    parent = subprocess.Popen([sys.executable, "-c", code, str(lock), str(pid_file)])
    try:
        wait_file(pid_file)
        parent.kill()
        parent.wait()
        with lock.open("a") as f, pytest.raises(BlockingIOError):
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.kill(int(pid_file.read_text()), signal.SIGKILL)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_publish_never_overwrites_existing_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    module.publish(path, {"original": True})
    with pytest.raises(FileExistsError):
        module.publish(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}


def test_real_base_a_runner_executes_and_binds_honest_receipt(tmp_path, monkeypatch):
    # Original training runner dependencies are injected; the real child and
    # file/target/summary admission path remain active.
    import types

    disk = types.ModuleType("disk_pause")

    class DiskGuard:
        used = 0

        def __init__(self, *args, **kwargs):
            pass

        def check(self, _child):
            return 0

        @staticmethod
        def resume_owned_group(child):
            pass

    disk.DiskPauseGuard = DiskGuard
    monkeypatch.setitem(sys.modules, "disk_pause", disk)
    operator = types.ModuleType("bootstrap_experiment_operator")
    operator.terminate_owned_group = lambda child, grace: module.cleanup(
        [child], grace=0.1
    )
    monkeypatch.setitem(sys.modules, "bootstrap_experiment_operator", operator)
    roots = []
    for i in range(35):
        root = tmp_path / f"root{i}"
        root.mkdir()
        roots.append(str(root))
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "rows": 58090688,
                "shards": 7108,
                "arms": {"A": {"roots": roots, "qualification": None}},
            }
        )
    )
    out = tmp_path / "out"
    summary = {
        "seed": 121,
        "batch_size": 512,
        "steps_realized": 1,
        "sampling": {
            "complete": True,
            "rows_planned": 58090688,
            "rows_realized": 58090688,
            "batches_realized": 1,
        },
    }
    code = "from pathlib import Path;import json; p=Path({!r});p.mkdir();(p/'initial_state.json').write_text({!r});(p/'summary.json').write_text({!r});(p/'checkpoint.pt').write_bytes(b'test')".format(
        str(out),
        json.dumps({"seed": 121, "tensor_sha256": "a" * 64}),
        json.dumps(summary),
    )
    plan = {
        "status": "FROZEN_READY",
        "arm": "A",
        "runtime": str(tmp_path),
        "runtime_head": "test",
        "pins": [],
        "out": str(out),
        "dataset_complete": str(targets),
        "base_roots": roots,
        "initial_anchor": str(out / "initial_state.json"),
        "command_prefix": [sys.executable, "-c", code],
        "operator_runtime": str(tmp_path),
        "training_seconds": 60,
        "internal_seconds": 60,
        "pause_seconds": 7200,
        "gpu_lock": str(tmp_path / "gpu.lock"),
        "env": {},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "test")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    prep = tmp_path / "prep.json"
    prep.write_text("{}")
    prep_ref = module.ref(prep)
    targets = tmp_path / "targets.json"
    value = json.loads(targets.read_text())
    value.update(
        status="COMPLETE_FACTORIAL58_BASE_A",
        preparation_plan=prep_ref,
        seals=[prep_ref] * 35,
    )
    targets.write_text(json.dumps(value))
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["preparation_plan"] = prep_ref
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setenv("FACTORIAL_A_CONTROL", str(tmp_path))
    path = Path(__file__).parents[1] / "scripts/factorial_base_a_runner.py"
    spec = importlib.util.spec_from_file_location("base_a", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--plan",
            str(plan_path),
            "--sha256",
            module.sha(plan_path),
            "--execute",
        ],
    )
    with (tmp_path / "coordinator.lock").open("a") as lock:
        monkeypatch.setenv("FACTORIAL_OWNER_FD", str(lock.fileno()))
        assert runner.main() == 0
    receipt = json.loads((tmp_path / "complete.json").read_text())
    assert receipt["status"] == "PASS_FACTORIAL58_ARM"
    assert receipt["bound_inputs"][0] == module.ref(targets)
    command = json.loads((tmp_path / "actual_command.json").read_text())["command"]
    assert command == plan["command_prefix"] + ["--shards", *plan["base_roots"]]


def test_subreaper_cleans_worker_after_child_supervisor_sigkill(tmp_path):
    source = Path(__file__).parents[1] / "scripts/factorial_prepare_and_train.py"
    script = tmp_path / "subreaper.py"
    script.write_text("""import ctypes,importlib.util,subprocess,sys,time,os
from pathlib import Path
spec=importlib.util.spec_from_file_location('coordinator',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
pidfile=Path(sys.argv[2])
code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(120)"
p=subprocess.Popen([sys.executable,'-c',code,str(pidfile)],start_new_session=True)
while not pidfile.exists():time.sleep(.01)
p.kill();p.wait()
time.sleep(.05)
m.cleanup([p],grace=.1,adopted=True)
try:
 worker=m.psutil.Process(int(pidfile.read_text()))
 assert worker.status()==m.psutil.STATUS_ZOMBIE
except m.psutil.NoSuchProcess:pass
""")
    result = subprocess.run(
        [sys.executable, str(script), str(source), str(tmp_path / "pid")],
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
