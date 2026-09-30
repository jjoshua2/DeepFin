"""Held, read-only CPU diagnostic for schema-2 overlay hot loading.

``check`` verifies cheap pins only. ``run`` requires a separate, fresh admission
receipt and launches at most four bounded children. It never trains a model.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
from typing import Any


MIB = 2**20
GIB = 2**30
ARM_LIMIT = 768 * 1024
SUMMARY_LIMIT = 128 * 1024
OUTPUT_LIMIT = 4 * MIB
ORDER = ("control", "candidate", "candidate", "control")
PACKAGE_NAMES = ("numpy", "torch", "numcodecs", "zarr")


class Refused(RuntimeError):
    """An input, resource, or protocol gate failed closed."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value: Any, limit: int) -> None:
    payload = json_bytes(value)
    if len(payload) > limit:
        raise Refused(f"receipt exceeds {limit} bytes: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise Refused(f"output already exists: {path}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise Refused(f"expected JSON object: {path}")
    return value


def git_head(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
        stderr=subprocess.DEVNULL, timeout=5,
    ).strip()


def check_pins(plan: dict[str, Any], *, runner: Path) -> None:
    if tuple(plan["order"]) != ORDER or plan["fixture"]["rows"] != 65536:
        raise Refused("frozen order or row count changed")
    if len(plan["fixture"]["shards"]) != 8:
        raise Refused("fixture must contain exactly eight shards")
    if sha256(runner) != plan["runner_sha256"]:
        raise Refused("runner pin mismatch")
    if sys.executable != plan["environment"]["python_executable"]:
        raise Refused("Python executable pin mismatch")
    if sys.version.split()[0] != plan["environment"]["python_version"]:
        raise Refused("Python version pin mismatch")
    for package in PACKAGE_NAMES:
        if importlib.metadata.version(package) != plan["environment"][package]:
            raise Refused(f"environment pin mismatch: {package}")
    for arm in ("control", "candidate"):
        spec = plan["sources"][arm]
        source = Path(spec["path"])
        if git_head(source) != spec["commit"]:
            raise Refused(f"{arm} commit pin mismatch")
        changed = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        if changed:
            raise Refused(f"{arm} tracked checkout changed")
        for relative, expected in spec["files_sha256"].items():
            if sha256(source / relative) != expected:
                raise Refused(f"{arm} source pin mismatch: {relative}")
    receipt = Path(plan["fixture"]["receipt"])
    if sha256(receipt) != plan["fixture"]["receipt_sha256"]:
        raise Refused("qualification receipt pin mismatch")
    entries = load_json(receipt)["shards"][:8]
    if sha256(receipt) != plan["fixture"]["receipt_sha256"]:
        raise Refused("qualification receipt changed while reading")
    if entries != plan["fixture"]["shards"]:
        raise Refused("qualified shard roster changed")
    if sum(int(entry["rows"]) for entry in entries) != 65536:
        raise Refused("qualified row count changed")


def check_resources(plan: dict[str, Any], output_dir: Path) -> None:
    bounds = plan["bounds"]
    if (output_dir / "STOP").exists() or (output_dir.parent / "STOP").exists():
        raise Refused("STOP requested")
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < bounds["minimum_memory_gib"] * GIB:
        raise Refused("available-memory reserve failed")
    if shutil.disk_usage(output_dir).free < bounds["minimum_disk_gib"] * GIB:
        raise Refused("free-disk reserve failed")
    if not set(bounds["affinity"]).issubset(os.sched_getaffinity(0)):
        raise Refused("required CPU cores unavailable")
    if bounds != {
        "seconds_per_arm": 180, "cpu_seconds_per_arm": 120,
        "max_arms": 4, "affinity": [16, 17], "nice": 19,
        "threads": 2, "minimum_memory_gib": 40,
        "minimum_disk_gib": 150, "new_output_limit_mib": 4,
        "gpu": False,
    }:
        raise Refused("resource bounds changed")
    total = sum(item.stat().st_size for item in output_dir.iterdir() if item.is_file())
    if total > OUTPUT_LIMIT:
        raise Refused("output budget exhausted")


def check_admission(plan: dict[str, Any], plan_path: Path, admission_path: Path) -> None:
    admission = load_json(admission_path)
    if admission.get("status") != "ADMITTED" or admission.get("plan_sha256") != sha256(plan_path):
        raise Refused("matching ADMITTED plan receipt required")
    if not admission.get("quiet_workload_evidence") or not admission.get("admitted_by"):
        raise Refused("admission lacks workload evidence or owner")
    when = admission.get("admitted_unix_seconds")
    if not isinstance(when, (int, float)) or not 0 <= time.time() - when <= 900:
        raise Refused("admission must be fresh, at most 15 minutes old")
    if plan["status"] != "PREPARED_HELD_FOR_UNCONTENDED_CPU_SLOT":
        raise Refused("unexpected plan status")


def _proc_starttime(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(") ", 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def cleanup_owned_child(process: subprocess.Popen[bytes], starttime: int | None) -> str:
    """Signal only the child process group while its original PID still exists."""
    if process.poll() is not None:
        return "already_exited"
    if starttime is None or _proc_starttime(process.pid) != starttime:
        return "identity_lost_no_signal"
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        return "already_exited"
    if group != process.pid:
        return "group_identity_lost_no_signal"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"
    try:
        process.wait(timeout=2)
        return "terminated"
    except subprocess.TimeoutExpired:
        if _proc_starttime(process.pid) != starttime:
            return "identity_lost_after_term"
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return "already_exited_after_term"
        try:
            process.wait(timeout=2)
            return "killed"
        except subprocess.TimeoutExpired:
            return "kill_unconfirmed"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if not isinstance(value, type) and hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def array_identity(array: Any) -> dict[str, Any]:
    import numpy as np

    value = np.asarray(array)
    return {
        "shape": list(value.shape), "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def child(
    plan: dict[str, Any], plan_path: Path, admission: Path,
    arm: str, output_dir: Path, index: int,
) -> None:
    check_pins(plan, runner=Path(__file__))
    check_admission(plan, plan_path, admission)
    if index not in range(4) or ORDER[index] != arm:
        raise Refused("child arm order mismatch")
    bounds = plan["bounds"]
    os.sched_setaffinity(0, set(bounds["affinity"]))
    os.nice(bounds["nice"])
    resource.setrlimit(resource.RLIMIT_CPU, (119, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MIB, MIB))
    signal.alarm(bounds["seconds_per_arm"])
    for name in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
        os.environ[f"{name}_NUM_THREADS"] = str(bounds["threads"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    check_resources(plan, output_dir)
    source = Path(plan["sources"][arm]["path"])
    sys.path.insert(0, str(source))
    import numpy as np
    import numcodecs.blosc as blosc
    import torch

    blosc.set_nthreads(2)
    torch.set_num_threads(2)
    from chess_anti_engine.replay import game_epoch as epoch
    from chess_anti_engine.replay import target_overlay as storage

    if not Path(epoch.__file__).resolve().is_relative_to(source.resolve()):
        raise Refused("game_epoch imported from wrong source")
    if not Path(storage.__file__).resolve().is_relative_to(source.resolve()):
        raise Refused("target_overlay imported from wrong source")
    entries = plan["fixture"]["shards"]
    paths = [Path(entry["path"]) for entry in entries]
    manifest = load_json(paths[0] / storage.MANIFEST)
    context = getattr(storage, "BaseSeals")([manifest["base_seal"]])
    receipt = Path(plan["fixture"]["receipt"])
    if hasattr(context, "bind_roots"):
        context.bind_roots([paths[0].parent, context.root])
        context.bind_receipt(receipt, storage._receipt_stamp(receipt))
    calls: list[str] = []
    original = getattr(storage, "_open_target_manifest")

    def counted(path: Path, manifest: Any, *, seal: Any = None) -> Any:
        calls.append(str(path))
        return original(path, manifest, seal=seal)

    setattr(storage, "_open_target_manifest", counted)
    stages: dict[str, float] = {}
    result: dict[str, Any] = {"arm": arm, "index": index, "status": "FAILED"}
    started = time.monotonic()
    try:
        check_resources(plan, output_dir)
        t = time.monotonic()
        for path, entry in zip(paths, entries, strict=True):
            if storage.overlay_content_sha256(path, seal=context) != entry["content_sha256"]:
                raise Refused(f"selected shard content drift: {path}")
        stages["selected_shard_identity"] = time.monotonic() - t
        t = time.monotonic()
        records = epoch._scan_shards(paths, 2, allow_target_overlay=True, overlay_seal=context)
        stages["scan_shards"] = time.monotonic() - t
        if len(records) != 8 or sum(record.rows for record in records) != 65536:
            raise Refused("scanned roster differs from frozen fixture")

        def row_count_census(arrays: Any) -> dict[str, float]:
            return {
                "policy": float(np.asarray(arrays["policy_target"]).sum(axis=1).size),
                "value": float(np.asarray(arrays["search_wdl"]).sum(axis=1).size),
            }

        t = time.monotonic()
        records = epoch._attach_objective_mask_weights(
            records, row_count_census, 2, allow_target_overlay=True, overlay_seal=context,
        )
        stages["objective_census"] = time.monotonic() - t
        startup_calls = len(calls)
        t = time.monotonic()
        epoch_plan, ordered = epoch._plan_epoch(
            records, batch_size=128, seed=121, load_workers=1,
            max_working_set_bytes=4 * GIB, mirror_augmentation=False,
        )
        stages["plan"] = time.monotonic() - t
        loader = object.__new__(epoch.GameAwareEpochBuffer)
        loader._allow_target_overlay = True
        loader._overlay_seal = context
        loader._input_planes = None
        loader._objective_mask_counter = row_count_census
        hot_before = len(calls)
        load_times: list[float] = []
        decoded: dict[str, Any] = {}
        for record in ordered:
            check_resources(plan, output_dir)
            t = time.monotonic()
            arrays = loader._load_one(record)
            load_times.append(time.monotonic() - t)
            decoded[str(record.path)] = {
                name: array_identity(array) for name, array in arrays.items()
            }
            del arrays
        stages["hot_load_one"] = sum(load_times)
        hot_calls = len(calls) - hot_before
        t = time.monotonic()
        targets: dict[str, Any] = {}
        for record in ordered:
            check_resources(plan, output_dir)
            arrays, _ = storage.overlay_proxies(
                record.path, ("policy_target", "search_wdl", "game_id"), seal=context,
            )
            targets[str(record.path)] = {
                name: array_identity(array) for name, array in arrays.items()
            }
        stages["ordered_target_hashes"] = time.monotonic() - t
        check_pins(plan, runner=Path(__file__))
        result.update({
            "status": "PASS", "plan": _jsonable(epoch_plan),
            "record_roster": [
                {"path": str(record.path), "rows": record.rows,
                 "content_sha256": record.content_sha256,
                 "objective_mask_weights": _jsonable(record.objective_mask_weights)}
                for record in ordered
            ],
            "decoded_array_hashes": decoded,
            "ordered_target_hashes": targets,
            "hot_load_seconds_per_shard": load_times,
            "startup_semantic_validations": startup_calls,
            "hot_load_semantic_validations": hot_calls,
            "total_semantic_validations": len(calls),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result.update({
        "stages_seconds": stages,
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": sum((resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                            resource.getrusage(resource.RUSAGE_SELF).ru_stime)),
        "scope": "Eight B schema-2 shards; real _load_one with synthetic row-count census; no trainer or GPU.",
    })
    atomic_json(output_dir / f"arm_{index}.json", result, ARM_LIMIT)
    if result["status"] != "PASS":
        raise Refused(result["error"])


def compare(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    if len(receipts) != 4 or [item.get("arm") for item in receipts] != list(ORDER):
        raise Refused("ABBA receipt sequence incomplete")
    if any(item.get("status") != "PASS" for item in receipts):
        raise Refused("one or more arms failed")
    parity_fields = ("plan", "record_roster", "decoded_array_hashes", "ordered_target_hashes")
    for field in parity_fields:
        if any(item[field] != receipts[0][field] for item in receipts[1:]):
            raise Refused(f"exact parity failed: {field}")
    sums = [sum(item["hot_load_seconds_per_shard"]) for item in receipts]
    control = statistics.median((sums[0], sums[3]))
    candidate = statistics.median((sums[1], sums[2]))
    if control <= 0 or candidate < 0:
        raise Refused("invalid hot-load timing")
    gain = 1 - candidate / control
    return {
        "status": "PASS_EXACT_PARITY", "arm_hot_load_seconds": sums,
        "median_control_seconds": control, "median_candidate_seconds": candidate,
        "fraction_reduction": gain,
        "decision": "WARRANTS_FUTURE_TRAINER_TEST" if gain >= 0.10 else "DOES_NOT_MEET_10_PERCENT_GATE",
        "scope": "CPU diagnostic only; no training throughput or deployment conclusion.",
    }


def supervise_child(
    process: subprocess.Popen[bytes], starttime: int | None,
    plan: dict[str, Any], output_dir: Path, index: int,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    deadline = clock() + plan["bounds"]["seconds_per_arm"]
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            cleanup = cleanup_owned_child(process, starttime)
            raise Refused(f"arm {index} exceeded wall deadline; cleanup={cleanup}")
        try:
            process.wait(timeout=min(1, remaining))
            return
        except subprocess.TimeoutExpired:
            try:
                check_resources(plan, output_dir)
            except Exception as exc:
                cleanup = cleanup_owned_child(process, starttime)
                raise Refused(f"arm {index} resource or STOP gate failed: {exc}; cleanup={cleanup}") from exc


def run(plan: dict[str, Any], plan_path: Path, admission: Path, output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise Refused("output directory must be new or empty")
    else:
        output_dir.mkdir(parents=False)
    receipts: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "status": "FAILED", "plan_sha256": sha256(plan_path),
        "admission_sha256": sha256(admission) if admission.is_file() else None,
        "runner_sha256": sha256(Path(__file__)),
        "completed_arms": 0,
    }
    try:
        check_pins(plan, runner=Path(__file__))
        check_admission(plan, plan_path, admission)
        check_resources(plan, output_dir)
        for index, arm in enumerate(ORDER):
            check_pins(plan, runner=Path(__file__))
            check_resources(plan, output_dir)
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "child", str(plan_path),
                 str(admission),
                 arm, str(output_dir), str(index)],
                cwd=output_dir, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            starttime = _proc_starttime(process.pid)
            cleanup = "already_exited"
            try:
                supervise_child(process, starttime, plan, output_dir, index)
            finally:
                if process.poll() is None:
                    cleanup = cleanup_owned_child(process, starttime)
            receipt_path = output_dir / f"arm_{index}.json"
            if process.returncode != 0 or not receipt_path.exists():
                detail = load_json(receipt_path).get("error") if receipt_path.exists() else "no child receipt"
                raise Refused(f"arm {index} exit={process.returncode}; {detail}; cleanup={cleanup}")
            if receipt_path.stat().st_size > ARM_LIMIT:
                raise Refused(f"arm {index} receipt too large")
            receipt = load_json(receipt_path)
            if receipt.get("index") != index or receipt.get("arm") != arm:
                raise Refused(f"arm {index} receipt identity mismatch")
            receipts.append(receipt)
            summary["completed_arms"] = len(receipts)
            if sum(item.stat().st_size for item in output_dir.iterdir() if item.is_file()) > OUTPUT_LIMIT:
                raise Refused("aggregate output budget exhausted")
        check_pins(plan, runner=Path(__file__))
        summary.update(compare(receipts))
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_json(output_dir / "summary.json", summary, SUMMARY_LIMIT)
        total = sum(item.stat().st_size for item in output_dir.iterdir() if item.is_file())
        if total > OUTPUT_LIMIT:
            raise Refused("aggregate output budget exhausted after summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="verify cheap pins; does not read shard content")
    check.add_argument("plan", type=Path)
    execute = sub.add_parser("run", help="requires independent fresh admission receipt")
    execute.add_argument("plan", type=Path)
    execute.add_argument("admission", type=Path)
    execute.add_argument("output_dir", type=Path)
    internal = sub.add_parser("child", help=argparse.SUPPRESS)
    internal.add_argument("plan", type=Path)
    internal.add_argument("admission", type=Path)
    internal.add_argument("arm", choices=("control", "candidate"))
    internal.add_argument("output_dir", type=Path)
    internal.add_argument("index", type=int)
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    if args.command != "check":
        args.admission = args.admission.resolve()
        args.output_dir = args.output_dir.resolve()
    plan = load_json(args.plan)
    try:
        if args.command == "check":
            check_pins(plan, runner=Path(__file__))
            print("Pinned sources, environment, receipt and roster match; plan remains held.")
        elif args.command == "run":
            run(plan, args.plan, args.admission, args.output_dir)
            print((args.output_dir / "summary.json").read_text(), end="")
        else:
            child(plan, args.plan, args.admission, args.arm, args.output_dir, args.index)
    except Exception as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
