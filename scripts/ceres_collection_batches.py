#!/usr/bin/env python3
"""Orchestrate preprepared Ceres collection chunk commands from a pinned JSON plan.

Consumes an explicit SHA256-pinned plan of chunk argv records. Default action
validates and prints the plan. --execute runs chunks in order with subprocess
and no shell. This does not sandbox argv, invent commands, own a GPU lock,
copy or relabel existing banks, retry, or resume.

Cleanup signals the owned process group (start_new_session leader pgid) even
after that leader has exited. Descendants that deliberately leave the group are
out of scope; the reviewed Ceres collector keeps its child in the inherited
group. A kernel D-state member cannot be reaped from Python.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from typing import Any

SCHEMA = 1
POLL_SECONDS = 2.0
CLEANUP_SECONDS = 30.0
MAX_CHUNK_TIMEOUT_SECONDS = 1800
MAX_OVERALL_SECONDS = 108000
MIN_FREE_GIB = 150
MIN_PAUSE_SECONDS = 30
FIXED32_BATCH = 32
COMPLETION_FIXED = "fixed_path"
COMPLETION_CERES = "ceres_invocations"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
PLAN_REQUIRED = (
    "schema",
    "state_directory",
    "chunks",
    "minimum_free_gib",
    "overall_seconds",
    "pause_between_chunks_seconds",
    "pinned_files",
)
CHUNK_REQUIRED = (
    "id",
    "start_shard",
    "max_shards",
    "expected_rows",
    "expected_padding_rows",
    "output_directory",
    "working_directory",
    "argv",
    "timeout_seconds",
    "completion_mode",
)


def require(ok: Any, message: str) -> None:
    if not ok:
        raise ValueError(message)


def require_int(value: Any, name: str, *, minimum: int | None = None,
                maximum: int | None = None) -> int:
    require(type(value) is int, f"{name} must be an integer")
    if minimum is not None:
        require(value >= minimum, f"{name} out of range")
    if maximum is not None:
        require(value <= maximum, f"{name} out of range")
    return value


def require_finite(value: Any, name: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)),
            f"{name} must be a finite number")
    require(math.isfinite(value), f"{name} must be a finite number")
    return float(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    def reject_nonfinite(token: str) -> Any:
        raise ValueError(f"nonfinite JSON in {path}: {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def dump_json(value: Any) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(dump_json(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.writing")
    try:
        tmp.write_text(dump_json(value), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def disk_free_bytes(path: Path) -> int:
    target = path if path.exists() else path.parent
    require(target.exists(), f"disk check path missing: {path}")
    return shutil.disk_usage(target).free


def paths_overlap(left: Path, right: Path) -> bool:
    a = left.resolve()
    b = right.resolve()
    return a == b or a in b.parents or b in a.parents


def absolute_path(value: Any, name: str) -> Path:
    require(isinstance(value, str) and value, f"{name} must be a nonempty string")
    path = Path(value)
    require(path.is_absolute(), f"{name} must be an absolute path")
    return path


def shard_name(index: int) -> str:
    return f"shard_{index:06d}.zarr"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def verify_plan_hash(plan_path: Path, expected: str) -> str:
    require(SHA256_RE.fullmatch(expected),
            "expected plan SHA256 must be 64 lowercase hex characters")
    digest = sha256_file(plan_path)
    require(digest == expected, "plan hash")
    return digest


def verify_pins(pinned_files: dict[str, str]) -> None:
    for raw_path, digest in pinned_files.items():
        path = Path(raw_path)
        require(path.is_file(), f"pinned file missing: {raw_path}")
        actual = sha256_file(path)
        require(actual == digest, f"pinned file hash differs: {raw_path}")


def check_resources(state: Path, minimum_free_gib: int, extra: list[Path]) -> None:
    require(not (state / "STOP").exists(), "STOP")
    need = minimum_free_gib * 2**30
    seen: set[Path] = set()
    for path in (state, *extra):
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        require(disk_free_bytes(path) >= need, f"disk reserve breached: {path}")


def process_group_present(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _wait_group(proc: subprocess.Popen[bytes], pgid: int, until: float) -> None:
    while process_group_present(pgid) or proc.poll() is None:
        remaining = until - time.monotonic()
        if remaining <= 0:
            return
        if remaining < POLL_SECONDS:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    return
            else:
                time.sleep(remaining)
            return
        if proc.poll() is None:
            try:
                proc.wait(timeout=POLL_SECONDS)
            except subprocess.TimeoutExpired:
                continue
        else:
            time.sleep(POLL_SECONDS)


def stop_process_group(proc: subprocess.Popen[bytes], deadline: float) -> None:
    # Cleanup can start from a timeout/STOP before any external signal arrived.
    # Mask both signals here too, rather than only after the first handler fired.
    previous = {sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        _stop_process_group(proc, deadline)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _stop_process_group(proc: subprocess.Popen[bytes], deadline: float) -> None:
    """TERM then KILL the owned session/group, including after the leader exits.

    Bound includes at most CLEANUP_SECONDS and the caller's deadline. Leader
    poll/wait is not treated as proof that group members have stopped.
    """
    pgid = proc.pid
    started = time.monotonic()
    cleanup_end = min(deadline, started + CLEANUP_SECONDS)
    # Reserve part of the same total cleanup allowance for post-KILL reaping.
    kill_at = started + max(0.0, cleanup_end - started) / 2
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    _wait_group(proc, pgid, kill_at)
    if process_group_present(pgid) or proc.poll() is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _wait_group(proc, pgid, cleanup_end)
    if proc.poll() is None or process_group_present(pgid):
        raise ValueError(
            "owned process group unreapable "
            f"pgid={pgid} leader_pid={proc.pid} leader_exit={proc.poll()} "
            f"group_present={process_group_present(pgid)}"
        )


@contextlib.contextmanager
def execute_termination_signals() -> Generator[None, None, None]:
    handling = False

    def handler(signum: int, _frame: Any) -> None:
        nonlocal handling
        if handling:
            return
        handling = True
        raise InterruptedError(f"termination signal {signum}")

    previous_term = signal.signal(signal.SIGTERM, handler)
    previous_int = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def validate_chunk(chunk: Any, index: int, state: Path, seen_ids: set[str],
                   outputs: list[Path], ranges: list[tuple[int, int]]) -> dict[str, Any]:
    require(isinstance(chunk, dict), f"chunk {index} must be an object")
    missing = [key for key in CHUNK_REQUIRED if key not in chunk]
    require(not missing, f"chunk {index} missing {missing}")
    ident = chunk["id"]
    require(isinstance(ident, str) and CHUNK_ID_RE.fullmatch(ident),
            f"chunk {index} id is not a unique safe identifier")
    require(ident not in seen_ids, f"duplicate chunk id: {ident}")
    seen_ids.add(ident)
    start = require_int(chunk["start_shard"], f"{ident}.start_shard", minimum=0)
    width = require_int(chunk["max_shards"], f"{ident}.max_shards", minimum=1)
    rows = require_int(chunk["expected_rows"], f"{ident}.expected_rows", minimum=1)
    padding = require_int(chunk["expected_padding_rows"],
                          f"{ident}.expected_padding_rows", minimum=0)
    timeout = require_int(chunk["timeout_seconds"], f"{ident}.timeout_seconds",
                          minimum=1, maximum=MAX_CHUNK_TIMEOUT_SECONDS)
    mode = chunk["completion_mode"]
    require(mode in (COMPLETION_FIXED, COMPLETION_CERES),
            f"{ident} completion_mode must be {COMPLETION_FIXED} or {COMPLETION_CERES}")
    output = absolute_path(chunk["output_directory"], f"{ident}.output_directory")
    cwd = absolute_path(chunk["working_directory"], f"{ident}.working_directory")
    completion: Path | None = None
    if mode == COMPLETION_FIXED:
        require("expected_completion" in chunk,
                f"{ident} fixed_path requires expected_completion")
        completion = absolute_path(chunk["expected_completion"],
                                   f"{ident}.expected_completion")
    else:
        require("expected_completion" not in chunk,
                f"{ident} ceres_invocations cannot set expected_completion")
    require(cwd.is_dir(), f"{ident} working directory is absent")
    require(output.parent.is_dir(), f"{ident} output parent is absent")
    require(not output.exists(), f"{ident} output directory is not fresh")
    require(not paths_overlap(output, state), f"{ident} output overlaps state")
    for other in outputs:
        require(not paths_overlap(output, other),
                f"{ident} output overlaps another chunk output")
    outputs.append(output)
    if completion is not None:
        require(completion != output and (completion.parent == output
                                          or output in completion.parents),
                f"{ident} expected completion is not inside the output directory")
    end = start + width
    for other_start, other_end in ranges:
        require(end <= other_start or other_end <= start,
                f"{ident} source range overlaps another chunk")
    ranges.append((start, end))
    argv = chunk["argv"]
    require(isinstance(argv, list) and argv,
            f"{ident} argv must be a nonempty string array, not a shell string")
    require(all(isinstance(item, str) for item in argv),
            f"{ident} argv must contain only strings")
    executable = Path(argv[0])
    require(executable.is_absolute(), f"{ident} argv executable must be absolute")
    require(executable.is_file() and os.access(executable, os.X_OK),
            f"{ident} argv executable is absent")
    return {
        "id": ident,
        "start_shard": start,
        "max_shards": width,
        "expected_rows": rows,
        "expected_padding_rows": padding,
        "output_directory": output,
        "working_directory": cwd,
        "argv": list(argv),
        "timeout_seconds": timeout,
        "completion_mode": mode,
        "expected_completion": completion,
    }


def validate_plan(plan: Any) -> dict[str, Any]:
    require(isinstance(plan, dict), "plan must be a JSON object")
    missing = [key for key in PLAN_REQUIRED if key not in plan]
    require(not missing, f"plan missing {missing}")
    require_int(plan["schema"], "schema", minimum=SCHEMA, maximum=SCHEMA)
    state = absolute_path(plan["state_directory"], "state_directory")
    require(not state.exists() or state.is_dir(), "state_directory is not a directory")
    minimum_free = require_int(plan["minimum_free_gib"], "minimum_free_gib",
                               minimum=MIN_FREE_GIB)
    overall = require_int(plan["overall_seconds"], "overall_seconds",
                          minimum=1, maximum=MAX_OVERALL_SECONDS)
    pause = require_int(plan["pause_between_chunks_seconds"],
                        "pause_between_chunks_seconds", minimum=MIN_PAUSE_SECONDS)
    pinned = plan["pinned_files"]
    require(isinstance(pinned, dict) and pinned,
            "pinned_files must be a nonempty absolute path to SHA256 map")
    bound: dict[str, str] = {}
    for raw_path, digest in pinned.items():
        path = absolute_path(raw_path, "pinned_files path")
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest),
                f"pinned SHA256 is not 64 lowercase hex: {raw_path}")
        bound[str(path)] = digest
    chunks_raw = plan["chunks"]
    require(isinstance(chunks_raw, list) and chunks_raw, "chunks must be a nonempty list")
    seen_ids: set[str] = set()
    outputs: list[Path] = []
    ranges: list[tuple[int, int]] = []
    chunks = [validate_chunk(item, index, state, seen_ids, outputs, ranges)
              for index, item in enumerate(chunks_raw)]
    verify_pins(bound)
    return {
        "schema": SCHEMA,
        "state_directory": state,
        "minimum_free_gib": minimum_free,
        "overall_seconds": overall,
        "pause_between_chunks_seconds": pause,
        "pinned_files": bound,
        "chunks": chunks,
    }


def load_plan(plan_path: Path, expected_sha256: str) -> dict[str, Any]:
    require(plan_path.is_file(), "plan file is absent")
    digest = verify_plan_hash(plan_path, expected_sha256)
    validated = validate_plan(load_json(plan_path))
    validated["plan_path"] = plan_path
    validated["plan_sha256"] = digest
    return validated


def collection_counts_consistent(counts: Any, rows: int, padding: int) -> None:
    require(isinstance(counts, dict), "collection_counts must be an object")
    real = require_int(counts.get("real_rows"), "collection_counts.real_rows", minimum=1)
    pad = require_int(counts.get("padding_rows"), "collection_counts.padding_rows",
                      minimum=0)
    calls = require_int(counts.get("calls"), "collection_counts.calls", minimum=1)
    inputs = require_int(counts.get("input_rows"), "collection_counts.input_rows",
                         minimum=1)
    require(real == rows, "collection real_rows do not match the chunk")
    require(pad == padding, "collection padding_rows do not match the plan")
    require(inputs == real + pad, "collection input_rows are inconsistent")
    require(inputs % FIXED32_BATCH == 0 and calls == inputs // FIXED32_BATCH,
            "collection calls are inconsistent with fixed32 batches")


def discover_ceres_completion(output: Path, ident: str) -> Path:
    root = output / "invocations"
    require(not root.is_symlink(), f"{ident} invocations path is a symlink")
    require(root.is_dir(), f"{ident} expected completion is missing")
    found: list[Path] = []
    child_only = False
    for item in sorted(root.iterdir(), key=lambda path: path.name):
        require(not item.is_symlink(), f"{ident} invocation directory is a symlink")
        if not item.is_dir():
            continue
        completed = item / "completed.json"
        child = item / "child_completed.json"
        require(not completed.is_symlink() and not child.is_symlink(),
                f"{ident} completion receipt is a symlink")
        if completed.is_file():
            found.append(completed)
        elif child.is_file():
            child_only = True
    require(not child_only or found,
            f"{ident} parent completed.json missing; child_completed.json is not accepted")
    require(found, f"{ident} expected completion is missing")
    require(len(found) == 1, f"{ident} completion receipts are ambiguous")
    return found[0]


def validate_completion_body(chunk: dict[str, Any], receipt: dict[str, Any]) -> None:
    ident = chunk["id"]
    require(receipt.get("complete") is True, f"{ident} completion is not complete=true")
    rows = require_int(receipt.get("rows"), f"{ident}.rows", minimum=1)
    require(rows == chunk["expected_rows"], f"{ident} completion rows differ")
    shards = require_int(receipt.get("shards"), f"{ident}.shards", minimum=1)
    new_shards = require_int(receipt.get("new_shards"), f"{ident}.new_shards",
                             minimum=0)
    require(shards == chunk["max_shards"], f"{ident} shard count differs")
    require(new_shards == shards, f"{ident} cached directory is not progress")
    selection = receipt.get("selection")
    if not isinstance(selection, list) or len(selection) != chunk["max_shards"]:
        raise ValueError(f"{ident} selection range differs")
    total = 0
    start = chunk["start_shard"]
    for offset, item in enumerate(selection):
        require(isinstance(item, dict), f"{ident} selection entry is not an object")
        require(item.get("path") == shard_name(start + offset),
                f"{ident} selection path differs")
        item_rows = require_int(item.get("rows"), f"{ident} selection rows", minimum=1)
        total += item_rows
    require(total == chunk["expected_rows"], f"{ident} selection rows differ")
    counts = receipt.get("collection_counts")
    collection_counts_consistent(counts, chunk["expected_rows"],
                                 chunk["expected_padding_rows"])
    if "new_collection_counts" in receipt:
        require(receipt["new_collection_counts"] == counts,
                f"{ident} cached collection counts are not progress")


def validate_completion(chunk: dict[str, Any], started_unix: float,
                        ended_unix: float) -> tuple[Path, str]:
    if chunk["completion_mode"] == COMPLETION_CERES:
        path = discover_ceres_completion(chunk["output_directory"], chunk["id"])
    else:
        path = chunk["expected_completion"]
        require(path.is_file(), f"{chunk['id']} expected completion is missing")
        require(not path.is_symlink(), f"{chunk['id']} completion receipt is a symlink")
    receipt = load_json(path)
    require(isinstance(receipt, dict), f"{chunk['id']} completion is not an object")
    validate_completion_body(chunk, receipt)
    if chunk["completion_mode"] == COMPLETION_CERES:
        parent_ended = require_finite(receipt.get("ended_unix"),
                                      f"{chunk['id']}.ended_unix")
        require(started_unix <= parent_ended <= ended_unix,
                f"{chunk['id']} parent ended_unix is outside this invocation window")
    return path, sha256_file(path)


def interruptible_wait(seconds: float, guard: Callable[[], None]) -> None:
    deadline = time.monotonic() + seconds
    while True:
        guard()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(remaining if remaining < POLL_SECONDS else POLL_SECONDS)


def run_chunk(chunk: dict[str, Any], state: Path, bound: float,
              guard: Callable[[], None]) -> dict[str, Any]:
    require(bound > CLEANUP_SECONDS, f"{chunk['id']} insufficient remaining time")
    require(not chunk["output_directory"].exists(),
            f"{chunk['id']} output directory is not fresh")
    chunk_state = state / "chunks" / chunk["id"]
    log_path = chunk_state / "stdout.log"
    receipt_path = chunk_state / "receipt.json"
    started = time.monotonic()
    deadline = started + bound
    unix_started = time.time()
    receipt: dict[str, Any] = {
        "id": chunk["id"],
        "argv": chunk["argv"],
        "cwd": str(chunk["working_directory"]),
        "timeout_bound_seconds": bound,
        "started_unix": unix_started,
        "complete": False,
    }
    proc: subprocess.Popen[bytes] | None = None
    try:
        chunk_state.mkdir(parents=True, exist_ok=False)
        with log_path.open("xb") as log:
            proc = subprocess.Popen(
                chunk["argv"],
                cwd=str(chunk["working_directory"]),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            receipt["pid"] = proc.pid
            receipt["pgid"] = proc.pid
            write_json(receipt_path, receipt)
            while proc.poll() is None:
                guard()
                remaining_to_term = deadline - CLEANUP_SECONDS - time.monotonic()
                if remaining_to_term < POLL_SECONDS:
                    raise ValueError(f"{chunk['id']} timeout")
                try:
                    proc.wait(timeout=POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    continue
        unix_ended = time.time()
        exit_code = proc.returncode
        elapsed = time.monotonic() - started
        receipt.update(
            ended_unix=unix_ended,
            elapsed_seconds=elapsed,
            exit_code=exit_code,
            stdout_log=str(log_path),
        )
        require(exit_code == 0, f"{chunk['id']} exited {exit_code}")
        require(not process_group_present(proc.pid),
                f"{chunk['id']} leader exited with owned group still present")
        completion_path, completion_sha256 = validate_completion(
            chunk, unix_started, unix_ended
        )
        receipt.update(
            complete=True,
            completion_sha256=completion_sha256,
            completion_path=str(completion_path),
        )
        write_json(receipt_path, receipt)
        return receipt
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        if proc is not None:
            try:
                stop_process_group(proc, deadline)
            except BaseException as stop_exc:
                cleanup_error = stop_exc
        receipt.update(
            complete=False,
            ended_unix=time.time(),
            elapsed_seconds=time.monotonic() - started,
            exit_code=None if proc is None else proc.poll(),
            error=repr(cleanup_error or exc),
            stdout_log=str(log_path) if log_path.exists() else None,
        )
        write_json(receipt_path, receipt)
        if cleanup_error is not None:
            raise cleanup_error from exc
        raise


def execute(plan: dict[str, Any]) -> dict[str, Any]:
    state = plan["state_directory"]
    extra_disk = [chunk["output_directory"].parent for chunk in plan["chunks"]]
    if state.exists():
        require(state.is_dir() and not any(state.iterdir()),
                "state directory is not fresh")
    else:
        state.mkdir(parents=True)
    started_monotonic = time.monotonic()
    overall_deadline = started_monotonic + plan["overall_seconds"]
    write_exclusive(state / "actual_start.json", {
        "pid": os.getpid(),
        "started_unix": time.time(),
        "plan": str(plan["plan_path"]),
        "plan_sha256": plan["plan_sha256"],
        "overall_seconds": plan["overall_seconds"],
        "chunk_ids": [chunk["id"] for chunk in plan["chunks"]],
    })
    completed: list[dict[str, Any]] = []

    def authenticity() -> None:
        verify_plan_hash(plan["plan_path"], plan["plan_sha256"])
        verify_pins(plan["pinned_files"])

    def resources() -> None:
        require(time.monotonic() < overall_deadline - CLEANUP_SECONDS,
                "overall budget exhausted")
        check_resources(state, plan["minimum_free_gib"], extra_disk)

    try:
        with execute_termination_signals():
            authenticity()
            resources()
            for index, chunk in enumerate(plan["chunks"]):
                authenticity()
                resources()
                require(not chunk["output_directory"].exists(),
                        f"{chunk['id']} output directory is not fresh")
                remaining = overall_deadline - time.monotonic()
                bound = min(float(chunk["timeout_seconds"]), remaining)
                receipt = run_chunk(chunk, state, bound, resources)
                completed.append({
                    "id": chunk["id"],
                    "completion_sha256": receipt["completion_sha256"],
                    "completion_path": receipt["completion_path"],
                    "elapsed_seconds": receipt["elapsed_seconds"],
                    "exit_code": receipt["exit_code"],
                    "stdout_log": receipt["stdout_log"],
                })
                authenticity()
                if index < len(plan["chunks"]) - 1:
                    interruptible_wait(plan["pause_between_chunks_seconds"], resources)
            authenticity()
            resources()
            manifest = {
                "schema": SCHEMA,
                "status": "COMPLETE",
                "plan_sha256": plan["plan_sha256"],
                "ended_unix": time.time(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "completed_chunks": completed,
            }
            write_exclusive(state / "manifest.json", manifest)
            return manifest
    except BaseException as exc:
        failed = {
            "schema": SCHEMA,
            "status": "FAILED",
            "plan_sha256": plan["plan_sha256"],
            "ended_unix": time.time(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "error": repr(exc),
            "completed_chunks": completed,
        }
        write_exclusive(state / "failed.json", failed)
        write_exclusive(state / "manifest.json", {
            "schema": SCHEMA,
            "status": "FAILED",
            "plan_sha256": plan["plan_sha256"],
            "ended_unix": failed["ended_unix"],
            "elapsed_seconds": failed["elapsed_seconds"],
            "completed_chunks": completed,
        })
        raise


def public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    chunks = []
    for chunk in plan["chunks"]:
        item = {
            "id": chunk["id"],
            "start_shard": chunk["start_shard"],
            "max_shards": chunk["max_shards"],
            "expected_rows": chunk["expected_rows"],
            "expected_padding_rows": chunk["expected_padding_rows"],
            "output_directory": str(chunk["output_directory"]),
            "working_directory": str(chunk["working_directory"]),
            "argv": chunk["argv"],
            "timeout_seconds": chunk["timeout_seconds"],
            "completion_mode": chunk["completion_mode"],
        }
        if chunk["expected_completion"] is not None:
            item["expected_completion"] = str(chunk["expected_completion"])
        chunks.append(item)
    return {
        "status": "VALID",
        "schema": SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "state_directory": str(plan["state_directory"]),
        "minimum_free_gib": plan["minimum_free_gib"],
        "overall_seconds": plan["overall_seconds"],
        "pause_between_chunks_seconds": plan["pause_between_chunks_seconds"],
        "pinned_files": plan["pinned_files"],
        "chunks": chunks,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = load_plan(Path(args.plan), args.expected_plan_sha256)
        if not args.execute:
            print(json.dumps(public_plan(plan), indent=2, allow_nan=False))
            return 0
        result = execute(plan)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
