#!/usr/bin/env python3
"""Time a disjoint cohort build using unchanged frozen preparation functions."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

owner_spec = importlib.util.spec_from_file_location(
    "preparation_overlap_owner", Path(__file__).with_name("bootstrap_preparation_overlap.py")
)
assert owner_spec is not None
assert owner_spec.loader is not None
owner = importlib.util.module_from_spec(owner_spec)
owner_spec.loader.exec_module(owner)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--index", required=True, type=int)
    p.add_argument("--seconds", type=int, default=1800)
    p.add_argument("--worker", choices=["seal", "cohort"])
    args = p.parse_args()
    config = owner.pinned({"path": args.config, "sha256": args.sha256})
    plan = owner.pinned(config["prep_plan"])
    for pin in plan["pins"]:
        owner.require(
            owner.sha(pin["path"]) == pin["sha256"], "frozen source/input pin differs"
        )
    owner.validate_config(config, plan)
    source = plan["cohorts"][args.index]
    owner.require(
        bool(source["ceres_manifest"].get("sha256")), "teacher not already pinned"
    )
    out = Path(args.output).resolve()
    owner.require(
        out.is_relative_to(Path("/home/josh/chess-artifacts")),
        "probe requires shared artifact output",
    )
    owner.require(
        all(
            out != Path(c["output"]).resolve()
            and out not in Path(c["output"]).resolve().parents
            and Path(c["output"]).resolve() not in out.parents
            for c in plan["cohorts"]
        ),
        "probe overlaps production outputs",
    )
    owner.require(
        all(
            out != Path(c["base"]).resolve()
            and out not in Path(c["base"]).resolve().parents
            and Path(c["base"]).resolve() not in out.parents
            for c in plan["cohorts"]
        ),
        "probe overlaps immutable inputs",
    )
    if args.worker:
        spec = importlib.util.spec_from_file_location(
            "frozen_probe_preparation", config["prep_runner"]["path"]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.HERE = out / "control"
        probe = {**plan, "cohorts": [{**source, "output": str(out / "targets")}]}
        module.worker(probe, args.worker, 0)
        return
    owner.require(not out.exists(), "probe output must be fresh")
    (out / "control").mkdir(parents=True)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    os.nice(15)
    owner.subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True)

    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    deadline = time.monotonic() + args.seconds
    result = {
        "status": "INCOMPLETE",
        "source_index": args.index,
        "rows": source["rows"],
        "shards": source["shards"],
        "stages": [],
    }
    try:
        with owner.stage_lock(out, config, deadline, queued=False) as fd:
            for stage in ["seal", "cohort"]:
                start = time.monotonic()
                command = [
                    config["python"],
                    str(Path(__file__).resolve()),
                    "--config",
                    args.config,
                    "--sha256",
                    args.sha256,
                    "--output",
                    str(out),
                    "--index",
                    str(args.index),
                    "--worker",
                    stage,
                ]
                owner.run_child(config, command, deadline, fd, out / (stage + ".log"))
                result["stages"].append(
                    {"stage": stage, "seconds": time.monotonic() - start}
                )
        paths = [p for p in (out / "targets").rglob("*") if p.is_file()]
        result.update(
            status="PASS_DISJOINT_COHORT_PROBE",
            target_bytes=sum(p.stat().st_size for p in paths),
            target_allocated_bytes=sum(p.stat().st_blocks * 512 for p in paths),
            target_files=len(paths),
        )
    finally:
        (out / "probe.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
