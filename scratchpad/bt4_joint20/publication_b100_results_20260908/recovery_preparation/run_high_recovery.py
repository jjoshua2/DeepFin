#!/usr/bin/python3
"""One-shot high-only continuation; original failed attempt remains immutable."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

V3 = Path("/tmp/deepfin-recipe-screen-tools-v3")
sys.path.insert(0, str(V3))
from scripts.bt4_recipe_screen import (reader, owned, inputs, storage, command,
    environment, certify, write, pin, stopped, HARD_SECONDS, OVERLAY_HEAD)


def verify_pins(plan):
    for item in plan["pins"]:
        reader.pinned(item)
    reader.require(sys.version_info[:2] == (3, 10), "producer Python3.10 required")
    reader.require(Path(sys.executable).resolve() == Path("/usr/bin/python3").resolve(), "wrong interpreter")
    reader.require(owned.sha(Path(sys.executable).resolve()) == plan["python_sha256"], "interpreter drift")


def guard(out, plan):
    stopped(out)
    reader.require(not any(p.exists() for p in (Path(plan["original_output"])/"STOP", Path(plan["preparation_stop"]))), "STOP requested")


@contextlib.contextmanager
def recovery_lease(out, plan):
    # Same shared lease as v3; also honor both original and recovery STOP while queued.
    with (owned.ROOT / "scratchpad/gpu0_experiment.lock").open("a") as lease:
        while True:
            guard(out, plan)
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(1)
        try:
            reader.require(not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=10).strip(), "competing GPU process")
            yield lease.fileno()
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def admit(plan):
    verify_pins(plan)
    m = reader.read_json(plan["original_manifest"])
    low = reader.read_json(plan["low_manifest"])
    report = reader.read_json(plan["validated_low_readout"])
    process = reader.read_json(low["process"])
    validation = reader.read_json(plan["low_validation_process"])
    reader.require(report["status"] == "VALID_CELL" and report["mode"] == "low_sprt", "invalid original low")
    reader.require(validation["exit_code"] == 0 and validation["readout_sha256"] == plan["validated_low_readout"]["sha256"] and validation["manifest_sha256"] == plan["low_manifest"]["sha256"], "unbound low recertification")
    reader.require("/usr/bin/python3" in validation["argv"], "low not certified with producer Python")
    reader.require(process["process_complete"] is True and process["exit_code"] == 0, "low process not complete")
    reader.same(process["gpu_seconds"], plan["low_gpu_seconds"], "low charge")
    reader.same(report["evidence"], {k: low[k] for k in ("bank", "result", "process", "launch")}, "low evidence")
    reader.same(m["output"], plan["original_output"], "original output")
    reader.require(not (Path(m["output"])/"high").exists(), "original high was already attempted")
    training = reader.read_json(m["candidate_training"])["training_charge_seconds"]
    reader.same(training, plan["training_gpu_seconds"], "training charge")
    reader.require(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in (training,plan["low_gpu_seconds"])), "invalid charges")
    reader.require(training + plan["low_gpu_seconds"] + HARD_SECONDS <= 27000, "remaining package budget insufficient")
    proof = reader.read_json(m["preparation"])
    rt = reader.read_json(m["runtime"])["runtime"]
    expected = command(m, rt, "high", Path(plan["output"])/"high")
    reader.same(expected, plan["high_command"], "registered high command")
    reader.check_command(expected, proof["observed"]["settings"]["high"], proof["observed"]["execution"]["high"], Path(plan["output"])/"high/arena.games.jsonl", Path(plan["output"])/"high/arena.results.jsonl", False)
    out = Path(plan["output"])
    reader.require(out.is_absolute() and out.parent.is_dir() and not out.exists(), "fresh output required; no resume")
    guard(out, plan)
    return m

def execute_high(m: dict[str, Any], plan: dict[str, Any]) -> None:
    runtime, rt = inputs(m)
    proof = reader.read_json(m["preparation"])
    reader.require(
        proof["status"] == "PASS_RECIPE_SCREEN_PREPARATION", "preparation not complete"
    )
    reader.require(
        set(proof["inputs"])
        == {
            "candidate",
            "reference",
            "candidate_training",
            "reference_training",
            "book",
            "runtime",
            "live_config",
            "preregistration",
        },
        "incomplete preparation inputs",
    )
    for key, value in proof["inputs"].items():
        reader.same(m[key], value, "prepared " + key)
    for key in ("launcher_sha256", "reader_sha256", "supervisor_sha256"):
        reader.same(proof[key], m[key], "prepared " + key)
    reader.opening_panel(proof["opening_panel"])
    reader.require(
        proof["observed"]["prefix_matches"] is True
        and proof["observed"]["cuda_initialized"] is False,
        "history or CPU preparation incomplete",
    )
    reader.same(
        proof["observed"]["runtime"],
        {k: v for k, v in rt.items() if k != "native_extension_sha256"},
        "prepared runtime",
    )
    for settings in proof["observed"]["settings"].values():
        reader.same(
            settings["search_candidate"], owned.qualified_search(), "prepared search"
        )
        reader.same(
            settings["search_reference"],
            owned.qualified_search(),
            "prepared reference search",
        )
    out = Path(plan["output"])
    reader.require(
        out.is_absolute() and out.parent.is_dir() and not out.exists(),
        "output must be new",
    )
    guard(out, plan)
    out.mkdir()
    write(out / "manifest.json", m)
    low_manifest = plan["low_manifest"]
    training_charge = reader.read_json(m["candidate_training"])[
        "training_charge_seconds"
    ]
    reader.require(
        type(training_charge) in (int, float) and 0 < training_charge <= 16200,
        "training charge exceeds cap",
    )
    charges = plan["low_gpu_seconds"]
    try:
        for stage in ("high",):
            guard(out, plan)
            stage_out = out / stage
            cmd = command(m, rt, stage, stage_out)
            settings, execution = (
                proof["observed"]["settings"][stage],
                proof["observed"]["execution"][stage],
            )
            reader.check_command(
                cmd,
                settings,
                execution,
                stage_out / "arena.games.jsonl",
                stage_out / "arena.results.jsonl",
                stage == "low",
            )
            identities = {
                k: {x: m[k][x] for x in ("path", "sha256")}
                for k in (
                    "candidate",
                    "reference",
                    "book",
                    "runtime",
                    "preregistration",
                )
            }
            identities["runtime"]["git_sha"] = OVERLAY_HEAD
            launch = {
                "settings": settings,
                "execution": execution,
                "opening_panel": proof["opening_panel"],
                "candidate_role": "B100",
                "reference_role": "H20",
                "command": cmd,
                "identities": identities,
                "preparation": m["preparation"],
            }
            launch_pin = write(out / f"{stage}.launch.json", launch)
            with recovery_lease(out, plan) as fd:
                # Recheck after any lease wait, before reading weights in the arena.
                inputs(m)
                verify_pins(plan)
                before = storage(m, runtime, rt)
                process = owned.run_owned_stage(
                    cmd,
                    stage_out,
                    HARD_SECONDS,
                    fd,
                    "arena",
                    {},
                    manifest=m,
                    stop_paths=(out / "STOP", Path(plan["original_output"]) / "STOP", Path(plan["preparation_stop"])),
                    cwd=runtime,
                    env=environment(runtime, gpu=True),
                )
            reader.same(storage(m, runtime, rt), before, "arena input stability")
            charges += process["gpu_seconds"]
            reader.require(charges <= 2 * HARD_SECONDS, "arena package cap exceeded")
            reader.require(training_charge + charges <= 27000, "total package cap exceeded")
            cell = {
                "schema": 1,
                "mode": "low_sprt" if stage == "low" else "high_fixed128",
                "bank": pin(stage_out / "arena.games.jsonl"),
                "result": pin(stage_out / "arena.results.jsonl"),
                "process": pin(stage_out / "process.json"),
                "launch": launch_pin,
                "opening_panel": proof["opening_panel"],
                "expected_settings": settings,
                "expected_execution": execution,
            }
            if stage == "high":
                cell["low_manifest"] = low_manifest
            cell_path = out / f"{stage}.reader_manifest.json"
            guard(out, plan)
            report = certify(cell, cell_path, stage_out)
            guard(out, plan)
            reader.require(
                report["status"] == "VALID_CELL", "reader did not certify cell"
            )
            write(
                stage_out / "complete.json",
                {
                    "complete": True,
                    "readout": pin(stage_out / "readout.stdout.json"),
                    "reader_manifest": pin(cell_path),
                    "gpu_seconds": process["gpu_seconds"],
                },
            )
        guard(out, plan)
        write(
            out / "complete.json",
            {
                "complete": True,
                "profile": "B100_H20",
                "gpu_seconds": charges,
                "training_gpu_seconds": training_charge,
                "package_gpu_seconds": training_charge + charges,
                "low": plan["validated_low_readout"],
                "low_reader_manifest": plan["low_manifest"],
                "recovery": "HIGH_ONLY_ORIGINAL_LOW_REUSED",
                "original_failure": plan["original_failure"],
                "high": pin(out / "high/complete.json"),
                "promotion": "NONE; same-seed development screen",
            },
        )
    except BaseException as error:
        write(
            out / "failed.json",
            {
                "complete": False,
                "error": repr(error),
                "gpu_seconds_completed_stages": charges,
            },
        )
        raise

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--expected-plan-sha256",required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    plan=reader.read_json({"path":str(args.plan),"sha256":args.expected_plan_sha256})
    m=admit(plan)
    if not args.execute:
        print(json.dumps({"status":"PASS_PLAN_ONLY_NO_NEW_QUALIFICATION", "command":plan["high_command"], "high_hard_seconds":5400, "training_plus_low_plus_high_cap":plan["training_gpu_seconds"]+plan["low_gpu_seconds"]+5400},indent=2))
        return
    def interrupted(signum,_frame):
        raise InterruptedError(f"signal {signum}")
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    execute_high(m,plan)

if __name__=="__main__":
    main()
