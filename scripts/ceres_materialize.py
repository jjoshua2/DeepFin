#!/usr/bin/env python3
"""Run one pinned Ceres policy/value materialization, never training or retrying.

Schema1: profile, cwd, commit, python, state, corpus, producer_manifest{path,sha256},
producer_sha256, pins, supervisor_sha256, stop_paths. Optional cpu_affinity and
preparation_lock allocate a reviewed independent lane. Default validates only.
Execution requires --execute and one absolute --deadline shared with an external
GNU timeout (TERM30seconds before deadline, KILL at deadline), maximum8hours.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import ceres_collection_batches as owned

BASE = Path("/home/josh/projects/chess/scratchpad/bt4_joint20/hybrid_endpoint_run01")
LOCK = BASE / "preparation.lock"
PROFILES = {"CeresB50": "ceres_target_mix", "B100CeresV25": "ceres_value_mix"}
MAX_SECONDS = 28800
RESERVE = 150 * 2**30
OUTPUT_CAP = 32 * 2**30
POLL_SECONDS = 2.0
SAMPLE_SECONDS = 300.0
PLAN_KEYS = {
    "schema",
    "profile",
    "cwd",
    "commit",
    "python",
    "state",
    "corpus",
    "producer_manifest",
    "producer_sha256",
    "pins",
    "supervisor_sha256",
    "stop_paths",
}

OPTIONAL_PLAN_KEYS = {"cpu_affinity", "preparation_lock"}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path(value: Any) -> Path:
    require(isinstance(value, str) and bool(value), "path must be a string")
    p = Path(value)
    require(
        p.is_absolute() and p.parent.resolve() == p.parent and not p.is_symlink(),
        "canonical absolute path required",
    )
    return p


def check_pins(p: dict[str, Any]) -> None:
    for filename, expected in p["pins"].items():
        require(sha(filename) == expected, f"changed input pin: {filename}")
    require(
        sha(p["producer_manifest"]["path"]) == p["producer_manifest"]["sha256"],
        "producer manifest changed",
    )
    require(sha(__file__) == p["supervisor_sha256"], "supervisor changed")


def module_path(module: Any) -> Path:
    filename = module.__file__
    require(isinstance(filename, str), "module source path missing")
    return Path(str(filename)).resolve()


def modules(p: dict[str, Any]) -> tuple[Any, Any]:
    sys.path.insert(0, p["cwd"])
    producer = importlib.import_module("scripts." + PROFILES[p["profile"]])
    epoch = importlib.import_module("scripts.bt4_one_epoch_screen")
    require(
        module_path(producer)
        == Path(p["cwd"]) / "scripts" / (PROFILES[p["profile"]] + ".py"),
        "loaded producer is not frozen checkout",
    )
    require(
        module_path(epoch) == Path(p["cwd"]) / "scripts/bt4_one_epoch_screen.py",
        "loaded admission is not frozen checkout",
    )
    return producer, epoch


def verify_inputs(p: dict[str, Any]) -> tuple[Any, Any]:
    check_pins(p)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=p["cwd"], text=True, timeout=15
    ).strip()
    require(head == p["commit"], "frozen checkout commit changed")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=p["cwd"],
        text=True,
        timeout=15,
    )
    require(not dirty, "frozen tracked checkout changed")
    producer, epoch = modules(p)
    for module in (producer, epoch, epoch.arena, owned):
        filename = str(module_path(module))
        require(p["pins"].get(filename) == sha(filename), "missing actual module pin")
    require(
        producer.producer_pins() == p["producer_sha256"],
        "producer dependency freeze changed",
    )
    require(
        all(p["pins"].get(k) == v for k, v in p["producer_sha256"].items()),
        "producer dependencies missing from pins",
    )
    require(
        Path(p["corpus"]) == epoch.CORPORA[p["profile"]], "profile corpus path differs"
    )
    manifest = producer.read_manifest(
        Path(p["producer_manifest"]["path"]), p["producer_manifest"]["sha256"]
    )[0]
    sources = [
        Path(manifest[k]).resolve() for k in ("source", "sf_source") if k in manifest
    ]
    sources += [
        Path(e[k]).resolve() for e in manifest["entries"] for k in ("bt4", "ceres")
    ]
    for out in (Path(p["corpus"]), partial(p), Path(p["state"])):
        require(
            all(not owned.paths_overlap(out, x) for x in sources),
            "output/state overlaps producer input",
        )
    return producer, epoch


def partial(p: dict[str, Any]) -> Path:
    out = Path(p["corpus"])
    return out.with_name(out.name + ".writing")


def execution_settings(p: dict[str, Any]) -> tuple[list[int], Path]:
    affinity = p.get("cpu_affinity", [0, 1])
    require(
        isinstance(affinity, list) and bool(affinity)
        and all(type(cpu) is int and cpu >= 0 for cpu in affinity)
        and len(set(affinity)) == len(affinity),
        "cpu_affinity must contain unique nonnegative integer CPUs",
    )
    if "cpu_affinity" in p:
        require(set(affinity) <= os.sched_getaffinity(0), "requested CPUs unavailable")
    lock = path(p.get("preparation_lock", str(LOCK)))
    require(
        lock in (LOCK, BASE / f"{p['profile']}.preparation.lock")
        and lock.parent.is_dir(),
        "preparation lock must be the shared or fixed profile lock",
    )
    return affinity, lock


def validate(p: dict[str, Any]) -> None:
    require(
        PLAN_KEYS <= set(p) <= PLAN_KEYS | OPTIONAL_PLAN_KEYS
        and type(p["schema"]) is int and p["schema"] == 1,
        "materialization schema/keys differ",
    )
    require(p["profile"] in PROFILES, "unsupported Ceres profile")
    require(
        isinstance(p["commit"], str)
        and len(p["commit"]) == 40
        and all(c in "0123456789abcdef" for c in p["commit"]),
        "invalid commit",
    )
    for key in ("cwd", "state", "corpus"):
        path(p[key])
    # Venv executables can be symlinks; their resolved bytes are explicitly pinned.
    require(
        isinstance(p["python"], str)
        and Path(p["python"]).is_absolute()
        and os.access(p["python"], os.X_OK),
        "absolute executable Python required",
    )
    require(
        Path(p["cwd"]).is_dir()
        and Path(p["state"]).parent.is_dir()
        and Path(p["corpus"]).parent.is_dir(),
        "existing cwd/output parents required",
    )
    require(
        set(p["producer_manifest"]) == {"path", "sha256"},
        "producer manifest pin fields differ",
    )
    path(p["producer_manifest"]["path"])
    for mapping in (p["pins"], p["producer_sha256"]):
        require(isinstance(mapping, dict) and bool(mapping), "nonempty pins required")
        for filename, digest in mapping.items():
            require(
                isinstance(filename, str)
                and Path(filename).is_absolute()
                and isinstance(digest, str)
                and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest),
                "invalid pin",
            )
    require(
        p["pins"].get(p["python"]) == sha(p["python"]), "Python pin missing/different"
    )
    require(
        p["pins"].get(str(Path(__file__).resolve())) == p["supervisor_sha256"],
        "supervisor pin missing",
    )
    require(
        p["pins"].get(p["producer_manifest"]["path"])
        == p["producer_manifest"]["sha256"],
        "manifest pin missing",
    )
    inputs = [Path(x).resolve() for x in p["pins"]] + [Path(p["cwd"])]
    outputs = [Path(p["state"]), Path(p["corpus"]), partial(p)]
    for i, out in enumerate(outputs):
        require(
            all(
                not owned.paths_overlap(out, other)
                for other in inputs + outputs[i + 1 :]
            ),
            "overlapping materialization paths",
        )
    require(
        isinstance(p["stop_paths"], list) and bool(p["stop_paths"]),
        "STOP paths required",
    )
    for stop in p["stop_paths"]:
        path(stop)
    required = {
        str(BASE / "STOP"),
        str(Path(p["state"]) / "STOP"),
        str(Path(p["state"]).parent / "STOP"),
    }
    require(required <= set(p["stop_paths"]), "required STOP paths missing")
    _, lock = execution_settings(p)
    require(all(not owned.paths_overlap(lock, item) for item in inputs + outputs),
            "preparation lock overlaps input/output")
    fresh(p)


def fresh(p: dict[str, Any]) -> None:
    require(
        not any(os.path.lexists(x) for x in (p["state"], p["corpus"], partial(p))),
        "state/output/partial exists; no attempt reuse",
    )


def output_bytes(p: dict[str, Any]) -> int:
    outputs = [x for x in (Path(p["corpus"]), partial(p)) if os.path.lexists(x)]
    require(not any(x.is_symlink() for x in outputs), "output became a symlink")
    if not outputs:
        return 0
    result = subprocess.run(
        ["/usr/bin/du", "-s", "-B1", "--", *(str(x) for x in outputs)],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    return sum(int(line.split()[0]) for line in result.stdout.splitlines())


def guard(p: dict[str, Any], deadline: float) -> None:
    require(time.time() < deadline - 30, "shared deadline cleanup margin reached")
    require(not any(os.path.lexists(x) for x in p["stop_paths"]), "STOP marker")
    require(
        shutil.disk_usage(Path(p["corpus"]).parent).free >= RESERVE,
        "disk below150GiBreserve",
    )


def command(p: dict[str, Any], deadline: float) -> list[str]:
    remaining = deadline - time.time() - 30
    require(remaining > 0, "no producer deadline remains")
    result = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{remaining:.6f}s",
        p["python"],
        str(Path(p["cwd"]) / "scripts" / (PROFILES[p["profile"]] + ".py")),
        "--manifest",
        p["producer_manifest"]["path"],
        "--expected-manifest-sha256",
        p["producer_manifest"]["sha256"],
        "--out",
        p["corpus"],
        "--batch-size",
        "128",
        "--minimum-free-gib",
        "150",
        "--max-seconds",
        f"{remaining:.6f}",
        "--stop",
        str(Path(p["state"]) / "STOP"),
        "--execute",
    ]
    if p["profile"] == "CeresB50":
        result += [
            "--bt4-weight",
            "0.5",
            "--bt4-temperature",
            "0.5",
            "--ceres-temperature",
            "0.5",
        ]
    return result


def verify_publication(p: dict[str, Any], budget: Any) -> dict[str, Any]:
    out = Path(p["corpus"])
    require(
        out.is_dir() and not out.is_symlink() and not os.path.lexists(partial(p)),
        "expected final corpus without partial",
    )
    producer, epoch = verify_inputs(p)
    summary_path = out / producer.SUMMARY
    derived_path = out / "derive_targets_summary.json"
    result = json.loads(summary_path.read_text())
    derived = json.loads(derived_path.read_text())
    require(
        result.get("manifest_sha256") == p["producer_manifest"]["sha256"],
        "publication manifest differs",
    )
    require(
        result.get("producer_sha256") == p["producer_sha256"],
        "publication producer differs",
    )
    admission = {"profile": p["profile"], "ceres_producer_pins": p["producer_sha256"]}
    if p["profile"] == "CeresB50":
        epoch.verify_ceres_recipe(admission, result, derived)
    else:
        epoch.verify_ceres_value_recipe(admission, result, derived)
    outputs = result["outputs"]
    require(
        isinstance(outputs, list) and bool(outputs), "output storage proofs missing"
    )
    for row in outputs:
        budget()
        digest = row.get("output_storage_identity")
        require(
            isinstance(digest, str) and len(digest) == 64,
            "output storage identity missing",
        )
        require(
            producer.wdl.storage_identity(out / row["path"]) == digest
            if p["profile"] == "B100CeresV25"
            else producer.shared.storage_identity(out / row["path"]) == digest,
            "published output storage changed",
        )
    budget()
    require(output_bytes(p) <= OUTPUT_CAP, "published output exceeds32GiBsamplecap")
    return {
        "profile": p["profile"],
        "corpus": str(out),
        "producer_manifest_sha256": p["producer_manifest"]["sha256"],
        "derive_summary_sha256": sha(derived_path),
        "rewrite_summary_sha256": sha(summary_path),
        "producer_sha256": p["producer_sha256"],
    }


def execute(
    p: dict[str, Any],
    plan_sha256: str,
    deadline: float,
    *,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    require(
        math.isfinite(deadline) and 60 < deadline - time.time() <= MAX_SECONDS,
        "deadline must leave60seconds..8hours",
    )

    def budget() -> None:
        guard(p, deadline)
        if plan_path is not None:
            require(sha(plan_path) == plan_sha256, "frozen plan changed")

    validate(p)
    verify_inputs(p)
    require(
        Path(sys.executable).absolute() == Path(p["python"]).absolute(),
        "supervisor Python differs",
    )
    budget()
    _, preparation_lock = execution_settings(p)
    lock_fd = os.open(preparation_lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fresh(p)
        check_pins(p)
        budget()
        state = Path(p["state"])
        state.mkdir()
        record: dict[str, Any] = {
            "schema": 1,
            "status": "STARTING",
            "plan_sha256": plan_sha256,
            "profile": p["profile"],
            "corpus": p["corpus"],
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "preparation_lock": str(preparation_lock),
            "started_unix": time.time(),
            "deadline_unix": deadline,
            "returncode": None,
        }
        owned.write_exclusive(state / "status.json", record)
        child = None
        owned_deadline = time.monotonic() + deadline - time.time()
        with owned.execute_termination_signals():
            try:
                budget()
                env = dict(os.environ)
                env.update(
                    CUDA_VISIBLE_DEVICES="",
                    OMP_NUM_THREADS="2",
                    MKL_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="2",
                    NUMEXPR_NUM_THREADS="2",
                    PYTHONNOUSERSITE="1",
                )
                with (state / "producer.log").open("xb") as log:
                    argv = command(p, deadline)
                    child = subprocess.Popen(
                        argv,
                        cwd=p["cwd"],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(lock_fd,),
                    )
                    record.update(
                        status="RUNNING", argv=argv, pid=child.pid, pgid=child.pid
                    )
                    owned.write_json(state / "status.json", record)
                    sampled_at = time.monotonic()
                    while child.poll() is None:
                        budget()
                        if time.monotonic() - sampled_at >= SAMPLE_SECONDS:
                            size = output_bytes(p)
                            record["sampled_output_allocated_bytes"] = size
                            require(size <= OUTPUT_CAP, "output exceeds32GiBsamplecap")
                            owned.write_json(state / "status.json", record)
                            sampled_at = time.monotonic()
                        try:
                            child.wait(timeout=POLL_SECONDS)
                        except subprocess.TimeoutExpired:
                            pass
                require(child.returncode == 0, f"producer exited{child.returncode}")
                require(
                    not owned.process_group_present(child.pid),
                    "producer left owned descendants",
                )
                budget()
                record.update(verify_publication(p, budget))
                budget()
                record.update(
                    status="COMPLETE", returncode=0, completed_unix=time.time()
                )
                owned.write_json(state / "status.json", record)
            except BaseException as error:
                cleanup_error = None
                if child is not None:
                    try:
                        owned.stop_process_group(child, owned_deadline)
                    except BaseException as failure:
                        cleanup_error = failure
                # Additional signals must not erase the failure/cleanup receipt.
                previous = {
                    sig: signal.signal(sig, signal.SIG_IGN)
                    for sig in (signal.SIGTERM, signal.SIGINT)
                }
                try:
                    record.update(
                        status="FAILED_OR_STOPPED",
                        returncode=None if child is None else child.poll(),
                        error=repr(cleanup_error or error),
                        completed_unix=time.time(),
                        no_automatic_retry=True,
                    )
                    owned.write_json(state / "status.json", record)
                finally:
                    for sig, handler in previous.items():
                        signal.signal(sig, handler)
                if cleanup_error is not None:
                    raise cleanup_error from error
                raise
        return record
    finally:
        os.close(lock_fd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--expected-plan-sha256", required=True)
    ap.add_argument("--deadline", type=float)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    require(sha(a.plan) == a.expected_plan_sha256, "plan pin differs")
    p = json.loads(a.plan.read_text())
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2",
    )
    validate(p)
    affinity, _ = execution_settings(p)
    os.sched_setaffinity(0, set(affinity))
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    subprocess.run(
        ["/usr/bin/ionice", "-c", "3", "-p", str(os.getpid())], check=True, timeout=5
    )
    validate(p)
    if not a.execute:
        verify_inputs(p)
        print(
            json.dumps(
                {
                    "status": "PLAN_VALIDATED_NOT_EXECUTED",
                    "profile": p["profile"],
                    "corpus": p["corpus"],
                }
            )
        )
        return 0
    require(a.deadline is not None, "--execute requires shared --deadline")
    print(
        json.dumps(
            execute(p, a.expected_plan_sha256, a.deadline, plan_path=a.plan.resolve()),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
