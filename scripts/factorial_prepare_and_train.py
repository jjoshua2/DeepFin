"""Overlap immutable target preparation and independently admitted baseline A.

This coordinator owns both jobs until both receipts and actual exit codes pass.
The original preparation wrapper keeps its separate inherited writer lock; the
A successor passes both GPU and coordinator ownership into its training child.
"""

from __future__ import annotations

import argparse
import ctypes
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

import psutil

GIB = 2**30


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin(ref):
    require(sha(ref["path"]) == ref["sha256"], "pin changed: " + ref["path"])
    return read(ref["path"])


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def base_readiness(config):
    """Validate all ordinary immutable bases, never claim changed arms are ready."""
    plan = pin(config["preparation_plan"])
    sys.path.insert(0, plan["runtime"])
    from chess_anti_engine.replay import target_overlay as storage

    seals = []
    rows = shards = 0
    for i, cohort in enumerate(plan["cohorts"]):
        seal = ref(
            Path(config["preparation_control"])
            / "cohorts"
            / f"cohort{i:02d}"
            / "base-seal.json"
        )
        proof = storage.require_base_corpus(seal, Path(cohort["base"]))
        require(
            proof["rows"] == cohort["rows"]
            and len(proof["shards"]) == cohort["shards"],
            "base totals differ",
        )
        pin(cohort["base_summary"])
        rows += proof["rows"]
        shards += len(proof["shards"])
        seals.append(seal)
    require(
        rows == 58090688 and shards == 7108 and len(seals) == 35,
        "exact base58 contract differs",
    )
    publish(
        config["base_ready"],
        {
            "status": "COMPLETE_FACTORIAL58_BASE_A",
            "rows": rows,
            "shards": shards,
            "preparation_plan": config["preparation_plan"],
            "seals": seals,
            "arms": {
                "A": {
                    "roots": [c["base"] for c in plan["cohorts"]],
                    "qualification": None,
                }
            },
        },
    )


def death_signal(parent):
    """Linux child gets TERM if its owning coordinator dies, including SIGKILL."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl parent death signal")
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGTERM)


def descendants(child):
    try:
        return psutil.Process(child.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def cleanup(children, grace=35, adopted=False):
    """Capture separately sessioned workers before terminating their supervisors."""
    owned = {}
    if adopted:
        for process in psutil.Process().children(recursive=True):
            owned[(process.pid, process.create_time())] = process
    for child in children:
        for process in descendants(child):
            owned[(process.pid, process.create_time())] = process
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(c.poll() is None for c in children):
        time.sleep(0.05)
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()
    for process in owned.values():
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(list(owned.values()), timeout=5)


def resources(config, children, started, startup=False):
    require(time.monotonic() - started < config["max_seconds"], "combined deadline")
    require(not any(Path(p).exists() for p in config["stop_paths"]), "STOP requested")
    require(
        psutil.virtual_memory().available >= (40 if startup else 32) * GIB,
        "available RAM reserve",
    )
    require(shutil.disk_usage(config["disk_root"]).free >= 80 * GIB, "disk reserve")
    processes = {}
    for child in children:
        try:
            p = psutil.Process(child.pid)
            for item in [p, *p.children(recursive=True)]:
                processes[item.pid] = item
        except psutil.NoSuchProcess:
            pass
    rss = 0
    for process in processes.values():
        try:
            rss += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    require(rss <= config["combined_rss_gib"] * GIB, "combined RSS cap")


def run(config, config_ref):
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(36, 1, 0, 0, 0) == 0, "cannot enable child subreaper")
    prep_plan = pin(config["preparation_plan"])
    a_plan = pin(config["a_plan"])
    prep_desc = pin(config["preparation_descriptor"])
    a_desc = pin(config["a_descriptor"])
    require(
        a_plan["arm"] == "A"
        and a_plan["preparation_plan"] == config["preparation_plan"],
        "A admission lineage",
    )
    require(a_plan["dataset_complete"] == config["base_ready"], "A readiness path")
    require(
        a_plan["base_roots"] == [c["base"] for c in prep_plan["cohorts"]], "base order"
    )
    for item in config["pins"]:
        require(sha(item["path"]) == item["sha256"], "source pin changed")
    children = []
    streams = []
    start = time.monotonic()
    result = {"status": "INCOMPLETE", "config": config_ref, "started_unix": time.time()}
    control = Path(config["control"])
    require(not (control / "complete.json").exists(), "combined receipt already exists")
    resources(config, children, start, startup=True)
    with (control / "ownership.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def launch(name, command, cwd, env):
            log = (control / f"{name}.log").open("x")
            streams.append(log)
            parent = os.getpid()
            child = subprocess.Popen(
                [sys.executable, __file__, "--owned-exec", str(parent), "--", *command],
                cwd=cwd,
                env={**os.environ, **env, "FACTORIAL_OWNER_FD": str(lock.fileno())},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
            children.append(child)
            return child

        try:
            prep = launch(
                "preparation",
                [
                    "taskset",
                    "-c",
                    "2,3",
                    "nice",
                    "-n",
                    "15",
                    "ionice",
                    "-c",
                    "3",
                    *prep_desc["argv"],
                ],
                prep_desc["cwd"],
                prep_desc.get("env", {}),
            )
            ready = train = None
            seals = [
                Path(config["preparation_control"])
                / "cohorts"
                / f"cohort{i:02d}"
                / "base-seal.json"
                for i in range(35)
            ]
            while True:
                resources(config, children, start)
                require(prep.poll() in (None, 0), "preparation process failed")
                if ready is None and all(p.is_file() for p in seals):
                    ready = launch(
                        "base-readiness",
                        [
                            "taskset",
                            "-c",
                            "2,3",
                            sys.executable,
                            __file__,
                            "--config",
                            config_ref["path"],
                            "--sha256",
                            config_ref["sha256"],
                            "--base-readiness",
                        ],
                        prep_plan["runtime"],
                        {
                            "CUDA_VISIBLE_DEVICES": "",
                            "OMP_NUM_THREADS": "2",
                            "OPENBLAS_NUM_THREADS": "2",
                        },
                    )
                if ready is not None:
                    require(ready.poll() in (None, 0), "base readiness failed")
                    if ready.poll() == 0 and train is None:
                        require(
                            read(config["base_ready"])["status"]
                            == "COMPLETE_FACTORIAL58_BASE_A",
                            "base readiness missing",
                        )
                        resources(config, children, start, startup=True)
                        train = launch(
                            "training-A",
                            a_desc["argv"],
                            a_desc["cwd"],
                            a_desc.get("env", {}),
                        )
                if train is not None:
                    require(train.poll() in (None, 0), "training A process failed")
                    if train.poll() == 0 and prep.poll() == 0:
                        break
                time.sleep(2)
            require(
                read(config["preparation_complete"])["status"]
                == "COMPLETE_FACTORIAL58_TARGETS",
                "preparation receipt failed",
            )
            a_receipt = read(config["a_complete"])
            require(
                a_receipt["status"] == "PASS_FACTORIAL58_ARM"
                and a_receipt["plan_sha256"] == config["a_plan"]["sha256"],
                "A receipt failed",
            )
            publish(
                config["a_terminal"],
                {"returncode": 0, "coordinator": config_ref, "ended_unix": time.time()},
            )
            result.update(
                status="PASS_FACTORIAL58_PREPARATION_AND_A",
                preparation=ref(config["preparation_complete"]),
                training=ref(config["a_complete"]),
                a_terminal=ref(config["a_terminal"]),
            )
        except BaseException as exc:
            result["error"] = repr(exc)
            raise
        finally:
            cleanup(children, adopted=True)
            for stream in streams:
                stream.close()
            result["ended_unix"] = time.time()
            publish(control / "complete.json", result)


def main():
    if len(sys.argv) > 4 and sys.argv[1] == "--owned-exec":
        death_signal(int(sys.argv[2]))
        require(sys.argv[3] == "--", "owned exec separator")
        os.execvpe(sys.argv[4], sys.argv[4:], os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--base-readiness", action="store_true")
    args = parser.parse_args()
    config_ref = {"path": args.config, "sha256": args.sha256}
    config = pin(config_ref)

    def stop(signum, _frame):
        raise InterruptedError(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.base_readiness:
        base_readiness(config)
    else:
        run(config, config_ref)


if __name__ == "__main__":
    main()
