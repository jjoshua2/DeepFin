"""Opt-in, exact-package live gather policy from retained CPU service observations.

Python validates provenance and replaces itself with the native live process.
It never owns search state, selects a leaf, schedules a call, or predicts a deadline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

from .profile import BATCHES, cells, digest, integer, object_json, sha, wave_plan

PLAN_ENV = "DEEPFIN_COHORT_DISPATCH"
PACKAGE_ENV = "DEEPFIN_COHORT_PROFILE_PACKAGE_SHA256"


def gather_caps(target: dict[str, Any]) -> list[int]:
    """Largest chunk of each equal-work optimum, NOT its tiny lexicographic head.

    Live roots recur after each forward. Repeatedly taking [1,4]'s head for five
    active roots would repeatedly request one row instead of consuming the plan.
    This table is a receding gather heuristic, not execution of that complete wave.
    """
    return [
        max(wave_plan([target], n)["alternatives"][0]["real_rows_per_call"])
        for n in range(1, 17)
    ]


def policy(
    report: dict[str, Any],
    package_sha: str,
    batch: int,
    channels: int,
    checkpoint_sha: str,
    encoding: dict[str, Any],
    host: dict[str, Any],
) -> dict[str, Any]:
    """Recompute costs from raw samples; summaries cannot silently supply a table."""
    if (
        report.get("schema") != "deepfin.native-service-profile.v1"
        or report.get("status") != "passed"
    ):
        raise ValueError("expected successful native service profile")
    for flag in ("runtime_dispatch_changed", "gpu_qualified", "strength_qualified"):
        if report.get(flag) is not False:
            raise ValueError("unexpected source profile scope")
    if (
        report.get("accepted_neural_rows", "missing") is not None
        or report.get("useful_eps", "missing") is not None
    ):
        raise ValueError("source profile falsely claims accepted search work")
    try:
        integer(batch, 1, 16)
        integer(channels, 146, 175)
    except ValueError as error:
        raise ValueError("unsupported package shape") from error
    if batch not in BATCHES or channels not in (146, 175):
        raise ValueError("unsupported package shape")
    digest(package_sha)
    if (
        digest(report.get("checkpoint_sha256")) != digest(checkpoint_sha)
        or report.get("encoding") != encoding
    ):
        raise ValueError("profile checkpoint/encoding mismatch")
    recorded = report.get("host")
    if not isinstance(recorded, dict):
        raise ValueError("missing measured host")
    # These are coarse compatibility gates, NOT host isolation or fresh calibration.
    # CPU MHz and cpuinfo hash change with time, so do not treat them as identifiers.
    for key in (
        "machine",
        "cpu_model",
        "logical_cpus",
        "torch_version",
        "torch_threads",
        "interop_threads",
    ):
        value = host.get(key)
        if (
            not value
            or type(recorded.get(key)) is not type(value)
            or recorded[key] != value
        ):
            raise ValueError("profile host/runtime mismatch: " + key)
    targets = report.get("targets")
    if not isinstance(targets, list) or not all(isinstance(t, dict) for t in targets):
        raise ValueError("invalid measured targets")
    chosen = [
        t for t in targets if t.get("identity", {}).get("package_sha256") == package_sha
    ]
    if len(chosen) != 1:
        raise ValueError("exact package must have one measured target")
    target = chosen[0]
    identity = target["identity"]
    for key, value in (
        ("batch", batch),
        ("channels", channels),
        ("checkpoint_sha256", checkpoint_sha),
    ):
        if type(identity.get(key)) is not type(value) or identity[key] != value:
            raise ValueError("target identity mismatch: " + key)
    if type(target.get("batch")) is not int or target["batch"] != batch:
        raise ValueError("target batch mismatch")
    runs = target.get("runs")
    if not isinstance(runs, list) or not 2 <= len(runs) <= 8:
        raise ValueError("need two to eight measured processes")
    for run in runs:
        if not isinstance(run, dict) or run.get("identity") != identity:
            raise ValueError("run identity mismatch")
        integer(run.get("samples_per_size"), 20, 256)
        integer(run.get("warmups_per_size"), 1, 32)
    recomputed = cells(runs)  # validates every sample, units, warmup and occupancy
    if recomputed != target.get("cells"):
        raise ValueError("serialized cost cells disagree with raw observations")
    caps = gather_caps({"batch": batch, "cells": recomputed})
    if any(not 1 <= cap <= min(batch, n) for n, cap in enumerate(caps, 1)):
        raise ValueError("unbounded gather plan")
    return {
        "schema": "deepfin.live-dispatch-policy.v1",
        "algorithm": "largest-optimal-chunk-v1",
        "package_sha256": package_sha,
        "checkpoint_sha256": checkpoint_sha,
        "batch": batch,
        "channels": channels,
        "caps_by_active_roots": caps,
        "cost_statistic": "sum of per-call sample p95; not a deadline bound",
        "measured_host": recorded,
        "runtime_host": host,
        "environment": {
            PLAN_ENV: " ".join(map(str, [batch, *caps])),
            PACKAGE_ENV: package_sha,
        },
        "package_switching": False,
        "wait_for_fill": False,
        "end_to_end_speed_qualified": False,
        "deadline_guarantee": False,
    }


def current_host(torch_version: str) -> dict[str, Any]:
    cpu = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    return {
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "cpu_model": next(
            (
                s.split(":", 1)[1].strip()
                for s in cpu.splitlines()
                if s.startswith("model name")
            ),
            "",
        ),
        "torch_threads": 2,
        "interop_threads": 1,
        "torch_version": torch_version,
    }


def launch(
    report_path: Path,
    package: Path,
    binary: Path,
    receipt: Path,
    *,
    simulations: int,
    depth: int,
    evals: int,
    diagnostics: int,
    arena: int,
    prepare_only: bool = False,
) -> None:
    from native.bend_engine.neural_probe.backend import (
        CHECKPOINT_FORMAT,
        package_manifest,
    )

    integer(simulations, 1, 256)
    integer(depth, 1, 32)
    integer(evals, 0, 256)
    integer(diagnostics, 0, 1)
    integer(arena, 1, 65536)
    package, binary = package.resolve(strict=True), binary.resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("native binary must be an executable file")
    manifest, encoding = package_manifest(package)
    if (
        manifest["format"] != CHECKPOINT_FORMAT
        or manifest["device"] != "cpu"
        or manifest["dtype"] != "float32"
    ):
        raise ValueError("requires a trusted CPU-F32 checkpoint package")
    raw = report_path.read_bytes()
    result = policy(
        object_json(raw.decode()),
        sha(package),
        integer(manifest["batch"], 1, 16),
        integer(manifest["channels"], 146, 175),
        digest(manifest["checkpoint_sha256"]),
        asdict(encoding),
        current_host(str(manifest["torch_version"])),
    )
    command = [
        str(binary),
        "--threads",
        "1",
        "--",
        str(simulations),
        str(depth),
        str(evals),
        str(diagnostics),
    ]
    result.update(
        profile_sha256=hashlib.sha256(raw).hexdigest(),
        binary_sha256=sha(binary),
        command=command,
        prepared_only=prepare_only,
    )
    # New file only. This is a launch receipt, not evidence that search completed.
    with receipt.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not prepare_only:
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("DEEPFIN_", "SERVICE_TEST_"))
        }
        env.update(
            result["environment"],
            DEEPFIN_BEND_MODEL_PACKAGE=str(package),
            DEEPFIN_COHORT_ASYNC="1",
            DEEPFIN_COHORT_ARENA_NODES=str(arena),
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
        os.execve(binary, command, env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "package", "binary", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--evals", type=int, default=0)
    parser.add_argument("--diagnostics", type=int, choices=(0, 1), default=0)
    parser.add_argument("--arena", type=int, default=4096)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    launch(
        args.profile,
        args.package,
        args.binary,
        args.receipt,
        simulations=args.simulations,
        depth=args.depth,
        evals=args.evals,
        diagnostics=args.diagnostics,
        arena=args.arena,
        prepare_only=args.prepare_only,
    )


if __name__ == "__main__":
    main()
