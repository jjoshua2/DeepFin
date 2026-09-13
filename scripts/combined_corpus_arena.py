#!/usr/bin/env python3
"""Prepare/run only the registered Combined35M V50/SF100 fixed paired arena."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from collections.abc import Generator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bt4_recipe_screen as recipe
from scripts import bt4_package_readout as package
from scripts import combined_corpus_train as combined
from scripts import training_host_memory as memory

owned, reader = recipe.owned, recipe.reader
PROFILE = combined.PROFILE
ROLES = ("Combined35M_V50", "Combined35M_SF100")
STAGE_SECONDS = 7200
WHOLE_SECONDS = 12030


def command(contract: dict[str, Any], executable: str) -> list[str]:
    return [
        executable,
        "scripts/arena_standard.py",
        "--candidate",
        contract["candidate"]["path"],
        "--reference",
        contract["reference"]["path"],
        "--games",
        "512",
        "--mode",
        "matched_sims",
        "--sims",
        "400",
        "--seed",
        "20260913",
        "--openings",
        contract["settings"]["openings"],
        "--opening-plies",
        "16",
        "--max-plies",
        "300",
        "--temperature",
        "0.1",
        "--search-shape",
        "training",
        "--cand-gumbel",
        "policy_temp=1.0",
        "--ref-gumbel",
        "policy_temp=1.0",
        "--compile",
        "on",
        "--device",
        "cuda",
        "--max-concurrent-games",
        "128",
        "--eval-max-batch",
        "4096",
        "--max-seconds",
        "7140.0",
        "--syzygy-max-pieces",
        "0",
        "--games-out",
        contract["bank"]["path"],
        "--out",
        contract["results_path"],
    ]


def contract_for(m: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    out = Path(m["output"])
    return {
        "schema": 1,
        "profile": PROFILE,
        "candidate": m["candidate"],
        "reference": m["reference"],
        "candidate_prior_temperature": 1.0,
        "reference_prior_temperature": 1.0,
        "sims": 400,
        "pairs": 256,
        "seed": 20260913,
        "settings": settings,
        "execution": {
            "loop": "rolling",
            "compile": "on",
            "eval_max_batch": 4096,
            "max_concurrent_games": 128,
            "max_seconds": 7140.0,
        },
        "opening_panel": m["opening_panel"],
        "training": {
            key: m[key] for key in ("candidate_training", "reference_training")
        },
        "bank": {"path": str(out / "arena.games.jsonl")},
        "process": {"path": str(out / "process.json")},
        "results_path": str(out / "arena.results.jsonl"),
    }


@contextmanager
def training_verifier(m: dict[str, Any]) -> Generator[None]:
    # Completed receipts bind the original coordinator's path as well as its bytes.
    ref = m["training_verifier"]
    reader.require(Path(ref["path"]).is_absolute(), "absolute training verifier")
    owned.pin(ref["path"], ref["sha256"])
    reader.same(
        ref["sha256"], owned.sha(combined.__file__), "reviewed combined verifier source"
    )
    for side in ("candidate", "reference"):
        receipt = reader.read_json(m[side + "_training"])
        reader.same(
            receipt["code_pins"].get(ref["path"]),
            ref["sha256"],
            "training verifier binding",
        )
    spec = importlib.util.spec_from_file_location(
        "_combined_arena_training_verifier", ref["path"]
    )
    reader.require(
        spec is not None and spec.loader is not None, "training verifier import"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = package.combined
    package.combined = module
    try:
        module.matched_training_pair(m)
        yield
    finally:
        package.combined = previous


def static(m: dict[str, Any]) -> None:
    reader.require(sys.flags.optimize == 0, "unoptimized assertion-enabled host required")
    reader.same(m["profile"], PROFILE, "combined arena profile")
    reader.same(
        tuple(m[s]["role"] for s in ("candidate", "reference")),
        ROLES,
        "combined direction",
    )
    reader.same(m["opening_panel"]["sha256"], combined.PANEL_SHA, "registered panel")
    reader.same(m["book"]["sha256"], owned.BOOK_SHA, "registered book")
    reader.require(m["sprt_lookahead_pairs"] == 64, "qualified arena overlay")
    for key in ("state", "output"):
        reader.require(Path(m[key]).is_absolute(), "absolute " + key)
    reader.require(m["state"] != m["output"], "separate operator and arena output")
    for path in (
        Path(__file__),
        Path(__file__).with_name("combined_corpus_arena_probe.py"),
        Path(recipe.__file__),
        Path(str(package.__file__)),
        Path(str(owned.__file__)),
        Path(memory.__file__),
        Path(str(reader.__file__)),
        Path(combined.__file__),
        Path(combined.schedule.__file__),
        Path(combined.original.__file__),
    ):
        owned.pin(path, m["code_pins"][str(path.resolve())])
    reader.pinned(m["preregistration"])


def stamps(
    m: dict[str, Any], runtime: Path, rt: dict[str, Any]
) -> dict[str, list[int]]:
    paths = set(m["code_pins"])
    for key in (
        "candidate",
        "reference",
        "candidate_training",
        "reference_training",
        "book",
        "opening_panel",
        "runtime",
        "live_config",
        "preregistration",
        "training_verifier",
    ):
        paths.add(m[key]["path"])
    for side in ("candidate", "reference"):
        receipt = reader.read_json(m[side + "_training"])
        paths.add(str(Path(receipt["run"]) / "summary.json"))
        for key in (
            "prospective",
            "corpus_manifest",
            "runtime_manifest",
            "selected_subset_qualification",
            "training_process",
        ):
            paths.add(receipt[key]["path"])
    paths.update(str(runtime / p) for p in recipe.arena_overlay(m)[1])
    paths.update(rt["native_extension_sha256"])
    return stat_paths(paths)


def stat_paths(paths) -> dict[str, list[int]]:
    result = {}
    for path in paths:
        st = Path(path).stat()
        result[path] = [
            st.st_dev,
            st.st_ino,
            st.st_size,
            st.st_mtime_ns,
            st.st_ctime_ns,
        ]
    return result


def guard(m: dict[str, Any], deadline: float) -> None:
    reader.require(time.time() < deadline, "combined arena deadline")
    reader.require(
        not any(
            Path(p).exists() for p in [*m["stop_paths"], str(Path(m["state"]) / "STOP")]
        ),
        "STOP",
    )
    owned.disk_guard(Path(m["state"]))
    memory.require_available(32)


def environment(runtime: Path, *, gpu: bool) -> dict[str, str]:
    env = recipe.environment(runtime, gpu=gpu)
    for key in ("PYTHONOPTIMIZE", "PYTHONHOME", "LD_PRELOAD"):
        env.pop(key, None)
    return env


def prepare(m: dict[str, Any], manifest_pin: dict[str, str], deadline: float) -> None:
    static(m)
    reader.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only preparation")
    reader.require(0 < deadline - time.time() <= 600, "CPU preparation budget")
    memory.require_available(48)
    guard(m, deadline)
    state = Path(m["state"])
    reader.require(
        not state.exists() and not Path(m["output"]).exists(),
        "fresh preparation/output",
    )
    with training_verifier(m):
        runtime, rt = recipe.qualified_runtime(m)
        owned.pin(m["book"]["path"], m["book"]["sha256"])
        for side in ("candidate", "reference"):
            owned.pin(m[side]["path"], m[side]["sha256"])
        before = stamps(m, runtime, rt)
        state.mkdir()
        request = {
            "packages": {m[s]["role"]: m[s] for s in ("candidate", "reference")},
            "runtime_root": str(runtime),
            "runtime": rt,
            "book": m["book"],
            "panel": m["opening_panel"],
            "arena_seed": 20260913,
            "cells": [
                {
                    "name": "combined_value",
                    "candidate": ROLES[0],
                    "reference": ROLES[1],
                    "priors": [1.0, 1.0],
                }
            ],
        }
        request_pin = recipe.write(state / "request.json", request)
        cmd = [
            rt["executable"],
            str(Path(__file__).with_name("combined_corpus_arena_probe.py")),
            request_pin["path"],
            str(state / "observed.json"),
        ]
        owned.run_owned_stage(
            cmd,
            state / "cpu_process",
            570,
            None,
            "prepare",
            {"runtime": rt},
            manifest=request,
            stop_paths=tuple(Path(p) for p in m["stop_paths"]),
            cwd=runtime,
            env=environment(runtime, gpu=False),
            guard=lambda: guard(m, deadline),
        )
        observed = owned.read(state / "observed.json")
        reader.same(
            observed["status"],
            "PASS_ACTUAL_COMBINED_VALUE_PAIR_CPU_PREPARATION",
            "CPU pair",
        )
        reader.same(
            observed["runtime"],
            {k: v for k, v in rt.items() if k != "native_extension_sha256"},
            "CPU runtime",
        )
        reader.same(
            observed["panel"], reader.read_json(m["opening_panel"]), "actual panel"
        )
        reader.same(observed["cuda_initialized"], False, "CPU-only observation")
        contract = contract_for(m, observed["cells"]["combined_value"]["settings"])
        package.validate(contract)
        cmd = command(contract, rt["executable"])
        package.command_check(cmd, contract)
        reader.same(stamps(m, runtime, rt), before, "inputs changed during preparation")
        guard(m, deadline)
        recipe.write(
            state / "prepared.json",
            {
                "status": "CPU_PREPARED_NOT_LAUNCHED",
                "manifest": manifest_pin,
                "runtime_root": str(runtime),
                "runtime": rt,
                "input_stamps": before,
                "contract_template": contract,
                "command": cmd,
                "observed": recipe.pin(state / "observed.json"),
                "request": request_pin,
            },
        )


def execute(m: dict[str, Any], prepared_pin: dict[str, str], deadline: float) -> None:
    static(m)
    reader.require(0 < deadline - time.time() <= 12000, "whole arena budget")
    memory.require_available(48)
    prepared = reader.read_json(prepared_pin)
    reader.same(prepared["status"], "CPU_PREPARED_NOT_LAUNCHED", "prepared status")
    reader.same(reader.read_json(prepared["manifest"]), m, "prepared manifest")
    reader.read_json(prepared["observed"])
    reader.read_json(prepared["request"])
    with training_verifier(m):
        runtime, rt = recipe.qualified_runtime(m)
        reader.same(str(runtime), prepared["runtime_root"], "prepared runtime")
        reader.same(rt, prepared["runtime"], "prepared runtime identity")
        contract, cmd = prepared["contract_template"], prepared["command"]
        reader.same(
            contract, contract_for(m, contract["settings"]), "prepared contract binding"
        )
        package.validate(contract)
        package.command_check(cmd, contract)
        reader.same(cmd, command(contract, rt["executable"]), "prepared exact command")
        out = Path(m["output"])
        reader.require(not out.exists(), "fresh arena output")
        operator = Path(m["state"]) / "arena_operator"
        operator.mkdir(exist_ok=False)
        recipe.write(
            operator / "started.json",
            {
                "started_unix": time.time(),
                "deadline_unix": deadline,
                "prepared": prepared_pin,
            },
        )

        def check() -> None:
            guard(m, deadline)
            reader.require(not (operator / "STOP").exists(), "operator STOP")
            reader.same(
                stat_paths(prepared["input_stamps"]),
                prepared["input_stamps"],
                "prepared inputs changed",
            )

        try:
            check()
            with (owned.ROOT / "scratchpad/gpu0_experiment.lock").open("a") as lease:
                waiting = time.monotonic()
                while True:
                    check()
                    reader.require(
                        time.monotonic() - waiting < 3600
                        and deadline - time.time() >= 7830,
                        "lease wait or remaining match/readout budget",
                    )
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(1)
                reader.require(
                    not subprocess.check_output(
                        [
                            "/usr/lib/wsl/lib/nvidia-smi",
                            "--query-compute-apps=pid",
                            "--format=csv,noheader",
                        ],
                        text=True,
                        timeout=10,
                    ).strip(),
                    "competing GPU process",
                )
                recipe.write(
                    operator / "lease_acquired.json",
                    {
                        "acquired_unix": time.time(),
                        "lease_wait_seconds": time.monotonic() - waiting,
                    },
                )
                owned.run_owned_stage(
                    cmd,
                    out,
                    STAGE_SECONDS,
                    lease.fileno(),
                    "arena",
                    {"runtime": rt, "prepared": prepared_pin},
                    manifest=prepared,
                    stop_paths=tuple(Path(p) for p in m["stop_paths"]),
                    cwd=runtime,
                    env=environment(runtime, gpu=True),
                    guard=check,
                )
            check()
            actual = dict(
                contract,
                bank=recipe.pin(out / "arena.games.jsonl"),
                process=recipe.pin(out / "process.json"),
            )
            actual_pin = recipe.write(out / "package.contract.json", actual)
            # Existing reader verifies complete bank and actual command; generic launch flag stays false.
            result = package.read_contract(Path(actual_pin["path"]))
            check()
            readout = recipe.write(out / "readout.json", result)
            recipe.write(
                operator / "completed.json",
                {
                    "status": "FIXED_MATCH_COMPLETED_PENDING_REVIEW",
                    "completed_unix": time.time(),
                    "prepared": prepared_pin,
                    "contract": actual_pin,
                    "readout": readout,
                },
            )
        except BaseException as exc:
            recipe.write(
                operator / "failed.json",
                {
                    "status": "FAILED_NO_AUTOMATIC_RETRY",
                    "error": repr(exc),
                    "prepared": prepared_pin,
                },
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-prepared-sha256")
    parser.add_argument("--deadline-unix", type=float)
    args = parser.parse_args()
    ref = {
        "path": str(args.manifest.resolve()),
        "sha256": args.expected_manifest_sha256,
    }
    m = reader.read_json(ref)
    reader.require(not (args.prepare and args.execute), "one operation")
    if not args.prepare and not args.execute:
        static(m)
        print("STATIC_DRAFT_ONLY_NOT_COMPLETION_OR_LAUNCH_ADMISSION")
        return
    reader.require(args.deadline_unix is not None, "absolute owned deadline required")
    if args.prepare:
        prepare(m, ref, args.deadline_unix)
    else:
        reader.require(bool(args.expected_prepared_sha256), "prepared pin required")
        execute(
            m,
            {
                "path": str(Path(m["state"]) / "prepared.json"),
                "sha256": args.expected_prepared_sha256,
            },
            args.deadline_unix,
        )


if __name__ == "__main__":

    def stop(signum, _frame):
        raise InterruptedError("combined arena signal " + str(signum))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main()
