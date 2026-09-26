"""Run one recorded CPU stage; preserve partial outputs and guard live disk reserve."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

STATE = Path(__file__).resolve().parent
ROOT = STATE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    stage = sys.argv[1]
    if stage not in ("audit", "mix"):
        raise ValueError("choose audit or mix")
    plan_path = STATE / "preparation_plan.json"
    plan = json.loads(plan_path.read_text())
    for path, expected in plan["pins"].items():
        if digest(path) != expected:
            raise ValueError(f"changed pinned input: {path}")
    if (STATE / "STOP").exists():
        raise RuntimeError("preparation STOP exists")
    output = Path(plan["output"])
    if output.exists() or output.with_name(output.name + ".writing").exists():
        raise RuntimeError("output or partial exists; preserve it")
    if shutil.disk_usage(ROOT).free < 300 * 1024**3:
        raise RuntimeError("less than 300 GiB free at admission")
    if stage == "mix":
        receipt = json.loads((STATE / "audit.status.json").read_text())
        if receipt["status"] != "COMPLETE" or receipt["plan_sha256"] != digest(plan_path):
            raise RuntimeError("audit did not complete under this plan")
        if receipt["audit_sha256"] != digest(STATE / "audit_H20.json"):
            raise RuntimeError("audit receipt changed")
    with (STATE / "preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = STATE / f"{stage}.status.json"
        if status_path.exists():
            raise RuntimeError("stage status exists; inspect instead of replacing work")
        record = {"stage": stage, "status": "STARTING", "plan_sha256": digest(plan_path),
                  "supervisor_sha256": digest(__file__), "started_unix": time.time(),
                  "supervisor_pid": os.getpid(), "argv": plan[f"{stage}_argv"]}

        def write():
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(status_path)

        write()
        child = None
        stop_reason = None

        def terminate_owned():
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        def stop(reason):
            nonlocal stop_reason
            stop_reason = stop_reason or reason
            terminate_owned()

        old_term = signal.signal(signal.SIGTERM, lambda *_: stop("supervisor SIGTERM"))
        old_int = signal.signal(signal.SIGINT, lambda *_: stop("supervisor SIGINT"))
        try:
            with (STATE / f"{stage}.log").open("x") as log:
                if stop_reason:
                    raise RuntimeError(stop_reason)
                child = subprocess.Popen(record["argv"], cwd=plan["cwd"], stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                if stop_reason:
                    terminate_owned()
                record.update(status="RUNNING", pid=child.pid)
                write()
                while child.poll() is None:
                    if (STATE / "STOP").exists():
                        stop("preparation STOP marker")
                    if shutil.disk_usage(ROOT).free < 150 * 1024**3:
                        stop("free disk below 150 GiB reserve")
                    if stop_reason:
                        record.update(status="STOP_REQUESTED", stop_reason=stop_reason)
                        write()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                record.update(status="COMPLETE" if child.returncode == 0 and not stop_reason else "FAILED_OR_STOPPED",
                              returncode=child.returncode, completed_unix=time.time())
                if stop_reason:
                    record["stop_reason"] = stop_reason
                if stage == "audit" and record["status"] == "COMPLETE":
                    record["audit_sha256"] = digest(STATE / "audit_H20.json")
                write()
                print(json.dumps(record), flush=True)
                return 0 if record["status"] == "COMPLETE" else 1
        except BaseException as exc:
            record.update(status="FAILED_EXCEPTION", error=repr(exc))
            try:
                write()
            except OSError:
                pass
            raise
        finally:
            terminate_owned()
            if child is not None:
                child.wait()
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)


if __name__ == "__main__":
    raise SystemExit(main())
