#!/usr/bin/env python3
"""Byte-preserving ZIP_STORED pilot; validated decoder, not exact-sampler support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from typing import Any


def digest(arrays: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def pack(source: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), "archive must be fresh")
    require(not source.is_symlink(), "linked shard root")
    before = {}
    for path in sorted(source.rglob("*")):
        require(not path.is_symlink(), "linked shard entry")
        if path.is_file():
            before[path.relative_to(source).as_posix()] = sha(path)
        else:
            require(path.is_dir(), "unsupported shard entry")
    require(".zgroup" in before and ".zattrs" in before, "not a complete Zarr group")
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as z:
        for name in before:
            z.write(source / name, arcname=name, compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(output) as z:
        require(z.namelist() == list(before), "archive membership differs")
        for entry in z.infolist():
            require(
                entry.compress_type == zipfile.ZIP_STORED, "unexpected ZIP compression"
            )
            require(
                hashlib.sha256(z.read(entry)).hexdigest() == before[entry.filename],
                "archive member bytes differ",
            )
    after = {
        p.relative_to(source).as_posix(): sha(p)
        for p in sorted(source.rglob("*"))
        if p.is_file()
    }
    require(after == before, "source changed during packing")
    return {
        "members": len(before),
        "member_sha256": before,
        "archive_sha256": sha(output),
        "archive_bytes": output.stat().st_size,
    }


def owned_decode(path: Path, loader: Any, zarr: Any):
    """One serial eager decoder call owns all stores it implicitly opens."""
    opened = {}
    original = zarr.open_group

    def tracked(*args, **kwargs):
        group = original(*args, **kwargs)
        opened[id(group.store)] = group.store
        return group

    zarr.open_group = tracked
    try:
        return loader(path, lazy=False, validate=True)
    finally:
        zarr.open_group = original
        for store in opened.values():
            store.close()


def publish(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_suffix(".writing")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def abort(output: Path, phase: str, error: BaseException) -> None:
    try:
        (output / f"{phase}-failed.json").write_text(json.dumps({"error": repr(error)}))
    finally:
        os._exit(9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["pack", "measure"])
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--runtime-head")
    p.add_argument("--copies", type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--external-copies", type=Path)
    p.add_argument("--external-output", type=Path)
    p.add_argument("--wait-pid", type=int)
    p.add_argument("--seconds", type=int, default=1800)
    a = p.parse_args()
    require(1 <= a.seconds <= 1800, "wall bound must be <=30min")
    import psutil

    # Do not stat/resolve/list or check free space on any external path until
    # the competing benchmark has ended. PID reuse conservatively blocks.
    if a.phase == "measure":
        require(
            a.wait_pid is not None and not psutil.pid_exists(a.wait_pid),
            "predecessor still exists; no external access allowed",
        )
        require(
            a.external_output is not None and a.external_copies is not None,
            "external paths required for measurement",
        )
        require(not a.external_output.exists(), "external output must be fresh")
        require(a.runtime_head is not None, "expected runtime HEAD required")
        require(
            subprocess.check_output(
                ["git", "-C", str(a.runtime), "rev-parse", "HEAD"], text=True
            ).strip()
            == a.runtime_head,
            "runtime HEAD differs",
        )
    require(sha(a.copies / "results.json") == a.receipt_sha256, "copy receipt changed")
    receipt = json.loads((a.copies / "results.json").read_text())
    copies = receipt["copies"]
    require(
        len(copies) == 16 and len({x["shard"] for x in copies}) == 16,
        "requires16 unique pilot shards",
    )
    require(
        all(
            Path(x["shard"]).name == x["shard"] and x["shard"].endswith(".zarr")
            for x in copies
        ),
        "invalid shard name",
    )
    require(a.copies.resolve() != a.output.resolve(), "output overlaps pilot copies")
    require(
        a.output.resolve().is_relative_to(Path("/home/josh/chess-artifacts")),
        "local artifact root required",
    )
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    os.nice(19)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLOSC_NTHREADS",
    ):
        os.environ[key] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.phase == "pack":
        require(not a.output.exists(), "local output must be fresh")
        a.output.mkdir(parents=True)
    else:
        require((a.output / "pack.json").is_file(), "missing packed receipt")
        require(not (a.output / "measure.json").exists(), "measurement already started")
    disks = [a.output]
    if a.phase == "measure":
        disks += [a.external_output.parent]
    started = time.monotonic()
    done = threading.Event()

    def guard_once():
        require(time.monotonic() - started <= a.seconds, "wall limit")
        require(psutil.Process().memory_info().rss <= 16 * 2**30, "RSS limit")
        require(psutil.virtual_memory().available >= 32 * 2**30, "available RAM floor")
        require(
            all(shutil.disk_usage(x).free >= 80 * 2**30 for x in disks), "disk floor"
        )
        require(not (a.output / "STOP").exists(), "STOP requested")

    guard_once()

    def monitor():
        while not done.wait(0.5):
            try:
                guard_once()
            except BaseException as exc:
                abort(a.output, a.phase, exc)

    threading.Thread(target=monitor, daemon=True).start()
    result = {
        "phase": a.phase,
        "complete": False,
        "scope": "Validated eager decoder only; no sampler integration; uncontrolled caches; fixed-order single pass",
        "copy_receipt_sha256": a.receipt_sha256,
        "runtime": str(a.runtime),
        "runtime_head": a.runtime_head,
        "loader_sha256": sha(a.runtime / "chess_anti_engine/replay/shard.py"),
        "script_sha256": sha(Path(__file__)),
        "records": [],
    }
    try:
        if a.phase == "pack":
            for entry in copies:
                start = time.monotonic()
                proof = pack(
                    a.copies / "zarr" / entry["shard"],
                    a.output / (entry["shard"] + ".zip"),
                )
                result["records"].append(
                    {
                        "shard": entry["shard"],
                        "pack_seconds": time.monotonic() - start,
                        **proof,
                    }
                )
                publish(a.output / "pack.json", result)
        else:
            packed = json.loads((a.output / "pack.json").read_text())
            require(
                packed["complete"]
                and packed["copy_receipt_sha256"] == a.receipt_sha256,
                "invalid pack receipt",
            )
            require(
                [x["shard"] for x in packed["records"]] == [x["shard"] for x in copies],
                "pack roster differs",
            )
            a.external_output.mkdir()
            result["transfers"] = []
            for entry in packed["records"]:
                src = a.output / (entry["shard"] + ".zip")
                dest = a.external_output / src.name
                require(sha(src) == entry["archive_sha256"], "local archive changed")
                start = time.monotonic()
                shutil.copyfile(src, dest)
                seconds = time.monotonic() - start
                require(
                    sha(dest) == entry["archive_sha256"],
                    "external archive bytes differ",
                )
                result["transfers"].append(
                    {
                        "shard": entry["shard"],
                        "copy_seconds": seconds,
                        "bytes": entry["archive_bytes"],
                    }
                )
            sys.path.insert(0, str(a.runtime.resolve()))
            import zarr
            import torch
            from numcodecs.blosc import set_nthreads
            from chess_anti_engine.replay.shard import load_shard_arrays

            torch.set_num_threads(2)
            set_nthreads(2)
            for medium, base, archive in [
                ("nvme", a.copies, a.output),
                ("external", a.external_copies, a.external_output),
            ]:
                for layout in ["zarr", "zip", "npz"]:
                    load_seconds = 0.0
                    rows = 0
                    start = time.monotonic()
                    for entry in copies:
                        name = entry["shard"]
                        path = (
                            archive / (name + ".zip")
                            if layout == "zip"
                            else base / "npz" / (Path(name).stem + ".npz")
                            if layout == "npz"
                            else base / "zarr" / name
                        )
                        tick = time.monotonic()
                        arrays, _ = owned_decode(path, load_shard_arrays, zarr)
                        load_seconds += time.monotonic() - tick
                        require(
                            digest(arrays) == entry["decoded_sha256"],
                            "decoded tensor digest differs: " + str(path),
                        )
                        rows += len(arrays["x"])
                        del arrays
                    result["records"].append(
                        {
                            "medium": medium,
                            "layout": layout,
                            "rows": rows,
                            "validated_decode_seconds": load_seconds,
                            "wall_seconds_including_digest": time.monotonic() - start,
                        }
                    )
                    publish(a.output / "measure.json", result)
        result["complete"] = True
    finally:
        done.set()
        result["elapsed_seconds"] = time.monotonic() - started
        publish(a.output / (a.phase + ".json"), result)


if __name__ == "__main__":
    main()
