#!/usr/bin/env python3
"""Overlap immutable CPU preparation with labeling; hand off at whole-stage boundaries.

Both actors must use this wrapper before the sidecar is enabled. The sidecar
checks the actual registered queue command, not an operator's assertion. A durable
handoff request prevents any subsequent sidecar stage or restart. The queued
actor retains the same flock through its original, pinned preparation command.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import psutil

GIB = 2**30


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned(ref: dict[str, str]) -> Any:
    require(sha(ref["path"]) == ref["sha256"], "changed pin: " + ref["path"])
    return read(ref["path"])


def append(path: Path, data: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def guard(config: dict[str, Any], deadline: float) -> None:
    require(time.monotonic() < deadline, "wall deadline reached")
    require(not any(Path(p).exists() for p in config["stop_paths"]), "STOP requested")
    require(
        psutil.virtual_memory().available >= config["memory_floor_gib"] * GIB,
        "available memory floor reached",
    )
    for p in config["disk_paths"]:
        require(
            shutil.disk_usage(p).free >= config["disk_floor_gib"] * GIB,
            "disk floor reached: " + p,
        )


def verify_registration(config: dict[str, Any], config_ref: dict[str, str]) -> None:
    """Refuse overlap unless the live queue really uses the handoff wrapper."""
    items = [
        x for x in read(config["queue_file"])["items"] if x["id"] == config["queue_id"]
    ]
    require(
        len(items) == 1 and items[0]["status"] == "queued",
        "preparation no longer queued",
    )
    item = items[0]
    descriptor = pinned(
        {"path": item["command_file"], "sha256": item["command_sha256"]}
    )
    expected = [
        config["python"],
        str(Path(__file__).resolve()),
        "--mode",
        "queued",
        "--config",
        config_ref["path"],
        "--sha256",
        config_ref["sha256"],
    ]
    require(
        descriptor["argv"] == expected,
        "live preparation lacks this exact handoff wrapper",
    )
    require(config_ref in descriptor["pins"], "wrapper config is not registered")
    require(
        {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
        in descriptor["pins"],
        "wrapper source is not registered",
    )


@contextmanager
def stage_lock(state: Path, config: dict[str, Any], deadline: float, *, queued: bool):
    """A request is permanent. Never replace or unlink the lock inode."""
    state.mkdir(parents=True, exist_ok=True)
    request = state / "handoff.request"
    if queued:
        with request.open("a") as f:
            f.flush()
            os.fsync(f.fileno())
        fd = os.open(state, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    with (state / "ownership.lock").open("a") as lock:
        while True:
            guard(config, deadline)
            if not queued and request.exists():
                yield None
                return
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.25)
        try:
            if not queued and request.exists():
                yield None
            else:
                yield lock.fileno()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def terminate(child: subprocess.Popen) -> None:
    """Clean the entire owned group, even if its leader exited first."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    until = time.monotonic() + 20
    while time.monotonic() < until:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait()


def run_child(
    config: dict[str, Any], command: list[str], deadline: float, lock_fd: int, log: Path
) -> None:
    guard(config, deadline)
    with log.open("ab") as output:
        # Inherit ownership into the child: SIGKILL of the wrapper cannot unlock
        # the stage while the producer still writes. Never explicit LOCK_UN until
        # the group has been terminated/reaped.
        child = subprocess.Popen(
            command,
            cwd=config["runtime"],
            env={
                **os.environ,
                **config["env"],
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "BLOSC_NTHREADS": "2",
            },
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lock_fd,),
        )
        try:
            while child.poll() is None:
                guard(config, deadline)
                try:
                    p = psutil.Process(child.pid)
                    rss = sum(
                        x.memory_info().rss for x in [p, *p.children(recursive=True)]
                    )
                    require(
                        rss <= config["rss_cap_gib"] * GIB, "process RSS cap reached"
                    )
                except psutil.NoSuchProcess:
                    pass
                try:
                    child.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            require(
                child.returncode == 0, f"child exited {child.returncode}; see {log}"
            )
        finally:
            terminate(child)


def run_queued(config: dict[str, Any], prefix: list[str], lock_fd: int) -> None:
    """Original coordinator, with ownership inherited by its new-session workers.

    Import instead of spawning an intermediate coordinator: otherwise killing
    that intermediate process can orphan its separately sessioned worker.
    Only this module's subprocess binding is adapted; no runtime file changes.
    """
    spec = importlib.util.spec_from_file_location(
        "_pinned_preparation", config["prep_runner"]["path"]
    )
    require(
        spec is not None and spec.loader is not None, "cannot import preparation runner"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.subprocess

    class LockedSubprocess:
        def __getattr__(self, name):
            return getattr(original, name)

        def Popen(self, *args, **kwargs):
            kwargs["pass_fds"] = tuple({*kwargs.get("pass_fds", ()), lock_fd})
            return original.Popen(*args, **kwargs)

    module.subprocess = LockedSubprocess()
    previous = sys.argv
    try:
        sys.argv = [*prefix[1:], "--execute"]
        module.main()
    finally:
        sys.argv = previous


def stages(plan: dict[str, Any]) -> list[tuple[str, int]]:
    """Small complete cohorts first; finish largest last to bound first measurement."""
    ready = sorted(
        (i for i, c in enumerate(plan["cohorts"]) if c["ceres_manifest"].get("sha256")),
        key=lambda i: plan["cohorts"][i]["rows"],
    )
    rest = sorted(
        (i for i in range(len(plan["cohorts"])) if i not in ready),
        key=lambda i: plan["cohorts"][i]["rows"],
    )
    return [stage for i in ready for stage in [("seal", i), ("cohort", i)]] + [
        ("seal", i) for i in rest
    ]


def validate_config(config: dict[str, Any], plan: dict[str, Any]) -> None:
    require(config["python"] == plan["python"], "interpreter differs")
    require(config["prep_runner"] in plan["pins"], "runner not pinned by frozen plan")
    require(config["rss_cap_gib"] <= plan["process_memory_cap_gib"], "RSS cap relaxed")
    require(
        config["memory_floor_gib"] >= plan["available_memory_floor_gib"],
        "RAM floor relaxed",
    )
    require(config["disk_floor_gib"] >= plan["disk_floor_gib"], "disk floor relaxed")
    require(
        0 < config["stage_seconds"] <= config["handoff_seconds"],
        "stage exceeds handoff allowance",
    )
    require(
        0 < config["sidecar_seconds"] <= plan["max_seconds"],
        "sidecar wall bound relaxed",
    )
    require(
        config["queued_seconds"] == plan["max_seconds"] + config["handoff_seconds"],
        "queued budget must preserve original preparation allowance",
    )
    prep = Path(config["prep_runner"]["path"]).parent
    require(
        {str(prep / "STOP"), str(prep.parent / "STOP")} <= set(config["stop_paths"]),
        "original STOP controls missing",
    )
    required_disks = {Path(c["output"]).parent.resolve() for c in plan["cohorts"]}
    provided_disks = [Path(x).resolve() for x in config["disk_paths"]]
    require(
        all(
            any(d == p or p in d.parents for p in provided_disks)
            for d in required_disks
        ),
        "target output disks not guarded",
    )


def run(config: dict[str, Any], ref: dict[str, str], mode: str) -> None:
    plan = pinned(config["prep_plan"])
    require(
        sha(config["prep_runner"]["path"]) == config["prep_runner"]["sha256"],
        "preparation runner changed",
    )
    require(
        config["runtime"] == plan["runtime"], "runtime differs from frozen preparation"
    )
    validate_config(config, plan)
    state = Path(config["state"])
    state.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))[:2]
    require(bool(cpus), "no available CPUs")
    os.sched_setaffinity(0, cpus)
    os.nice(15)
    # Linux I/O scheduling class idle. No child may inherit high I/O priority.
    subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True)
    wall = time.monotonic()
    deadline = wall + config[mode + "_seconds"]
    prefix = [
        config["python"],
        config["prep_runner"]["path"],
        "--plan",
        config["prep_plan"]["path"],
        "--sha256",
        config["prep_plan"]["sha256"],
    ]
    if mode == "queued":
        with stage_lock(
            state, config, wall + config["handoff_seconds"], queued=True
        ) as fd:
            require(fd is not None, "queued ownership missing")
            run_queued(config, prefix, fd)
        return
    for stage, index in stages(plan):
        with stage_lock(state, config, deadline, queued=False) as fd:
            if fd is None:
                append(
                    state / "events.jsonl",
                    {"status": "HANDED_OFF", "unix": time.time()},
                )
                return
            verify_registration(config, ref)
            start = time.monotonic()
            run_child(
                config,
                [*prefix, "--worker", stage, "--index", str(index)],
                min(deadline, start + config["stage_seconds"]),
                fd,
                state / f"{stage}{index:02d}.log",
            )
            append(
                state / "events.jsonl",
                {
                    "status": "STAGE_COMPLETE",
                    "stage": stage,
                    "index": index,
                    "seconds": time.monotonic() - start,
                    "unix": time.time(),
                    "rows": plan["cohorts"][index]["rows"],
                },
            )
    append(state / "events.jsonl", {"status": "PREWORK_COMPLETE", "unix": time.time()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["sidecar", "queued"], required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--sha256", required=True)
    a = p.parse_args()

    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    ref = {"path": a.config, "sha256": a.sha256}
    run(pinned(ref), ref, a.mode)


if __name__ == "__main__":
    main()
