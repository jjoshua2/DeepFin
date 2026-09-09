#!/usr/bin/env python3
"""Bank named BT4 WDL outputs from a pinned original derived SF corpus.

No policy, legal-move mapping, raw-history replay or training-target rewrite.
The actual LC0 feed is bound; original float32 input_key is not reconstructed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc
from numcodecs.blosc import set_nthreads, get_nthreads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chess_anti_engine.encoding import lc0
from scripts import bt4_raw_corpus_sidecar as raw, g10_wdl_admission as g10
from scripts.adapt_raw_bt4_sidecars import storage_identity
from scripts.bt4_policy_dump import file_sha256, open_session
from scripts.sf_policy_rewrite import require, shard_contract

SUMMARY = "derive_targets_summary.json"
COLUMNS = ("x", "game_id", "ply_index", "has_game_id", "has_ply_index")
HISTORY = "lc0_root_legacy_meta"
LINEAGE = (
    "Inherited original derived-source history; actual stored x and LC0 feed verified. "
    "No raw-history replay, original float32 input_key reconstruction, or retroactive control qualification."
)


def stored_feed(x: np.ndarray) -> np.ndarray:
    """Reject invalid writer-domain inputs before the shared converter can clip."""
    require(
        x.dtype == np.float16 and x.ndim == 4 and x.shape[1:] == (175, 8, 8),
        "expected float16 [N,175,8,8] source x",
    )
    require(bool(np.isfinite(x).all()), "nonfinite source x")
    require(
        bool(np.all((x[:, :109] == 0) | (x[:, :109] == 1))), "nonbinary LC0 history"
    )
    constants = [*range(12, 104, 13), *range(104, 110)]
    require(
        bool(np.all(x[:, constants] == x[:, constants, :1, :1])),
        "nonconstant LC0 metadata",
    )
    counters = np.asarray(np.arange(101, dtype=np.float32) / 100, dtype=np.float16)
    require(
        bool(np.isin(x[:, 109, 0, 0], counters).all()),
        "rule50 outside stored integer domain",
    )
    return lc0.x_to_lc0_planes(x, input_history_encoding=HISTORY)


def complete_chunks(array: Any) -> None:
    """Zarr silently fills absent chunks; a completed source must have real storage."""
    for coordinates in itertools.product(
        *(
            range(math.ceil(n / c))
            for n, c in zip(array.shape, array.chunks, strict=True)
        )
    ):
        require(
            array._chunk_key(coordinates) in array.chunk_store, "missing stored chunk"
        )


def source_inventory(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(args.source).resolve()
    require(not source.name.endswith(".writing"), "partial source directory")
    path = source / SUMMARY
    require(
        file_sha256(path) == args.expected_source_summary_sha256,
        "source summary pin differs",
    )
    summary = json.loads(path.read_text())
    rows = summary["realized"]["rows_written"]
    qualification = getattr(args, "g10_common_qualification", None)
    qualification_sha = getattr(args, "expected_g10_common_qualification_sha256", None)
    require(bool(qualification) == bool(qualification_sha),
            "G10 qualification requires both path and SHA256")
    if qualification:
        if not isinstance(qualification_sha, str):
            raise ValueError("G10 qualification SHA256 must be a string")
        args.g10_admission = g10.admit(Path(qualification), qualification_sha, source, summary)
    else:
        args.g10_admission = None
    require(
        type(rows) is int and rows > 0 and (args.g10_admission is not None
            or summary["corpus"]["corpus_complete"] is True),
        "source is incomplete",
    )
    require(
        summary.get("policy_target_postprocess") is None, "requires original SF corpus"
    )
    expected = {
        "input_history_encoding": HISTORY,
        "history_rep_fix": True,
        "input_extra_features": "v2_threats",
        "zero_history": False,
        "history_frames_total": 8,
    }
    require(
        all(summary["input"].get(k) == v for k, v in expected.items()),
        "source history regime differs",
    )
    require(
        summary["realized"]["input_key_verified"] == rows
        and summary["realized"]["row_schema_counts"] == {"3": rows},
        "missing inherited history proof",
    )
    specs = summary["shards"]
    require(isinstance(specs, list) and len(specs) > 0, "missing shard inventory")
    for i, spec in enumerate(specs):
        require(
            spec["path"] == f"shard_{i:06d}.zarr"
            and type(spec["rows"]) is int
            and spec["rows"] > 0,
            "invalid canonical shard inventory",
        )
    require(sum(s["rows"] for s in specs) == rows, "source row inventory differs")
    require(
        [p.name for p in sorted(source.glob("shard_*.zarr"))]
        == [s["path"] for s in specs],
        "source shard membership differs",
    )
    require(
        0 <= args.start_shard < len(specs) and args.max_shards > 0,
        "invalid shard selection",
    )
    return summary, specs[args.start_shard : args.start_shard + args.max_shards]


def source_arrays(path: Path, summary: dict[str, Any], rows: int) -> Any:
    group: Any = zarr.open_group(str(path), mode="r")
    shard_contract(dict(group.attrs), summary, rows)
    for name in COLUMNS:
        require(name in group, f"missing source column {name}")
        array = group[name]
        require(
            tuple(array.shape) == ((rows, 175, 8, 8) if name == "x" else (rows,)),
            "source layout differs",
        )
        if name == "x":
            require(array.dtype == np.dtype("float16"), "source x dtype differs")
        else:
            kinds = "iu" if name in ("game_id", "ply_index") else "biu"
            require(array.dtype.kind in kinds, "nonintegral source identity/presence")
        complete_chunks(array)
    return group


def row_digests(feed: np.ndarray) -> np.ndarray:
    return np.asarray(
        [list(hashlib.sha256(row.tobytes(order="C")).digest()) for row in feed],
        dtype=np.uint8,
    )


def g10_binding(args: argparse.Namespace) -> dict[str, Any]:
    admitted = getattr(args, "g10_admission", None)
    return ({"g10_common_admission": admitted,
             "g10_admission_script_sha256": file_sha256(Path(g10.__file__))}
            if admitted is not None else {})


def check_g10_pin(args: argparse.Namespace) -> None:
    admitted = getattr(args, "g10_admission", None)
    if admitted is not None:
        pin = admitted["qualification"]
        require(file_sha256(pin["path"]) == pin["sha256"], "G10 qualification changed")


def binding(
    args: argparse.Namespace, spec: dict[str, Any], state: str
) -> dict[str, Any]:
    return {
        "schema": 1,
        **g10_binding(args),
        "source_dir": str(Path(args.source).resolve()),
        "source_shard": spec["path"],
        "source_summary_sha256": args.expected_source_summary_sha256,
        "source_storage_identity": state,
        "rows": spec["rows"],
        "onnx": str(Path(args.onnx).resolve()),
        "onnx_sha256": args.expected_onnx_sha256,
        "requested_wdl": {"output": args.wdl_output, "kind": args.wdl_output_kind},
        "history_lineage": LINEAGE,
        "producer": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha256(p)
            for p in (
                Path(__file__).resolve(),
                Path(lc0.__file__).resolve(),
                Path(raw.__file__).resolve(),
            )
        },
    }


def verify_cached(
    path: Path, expected: dict[str, Any], batch_size: int
) -> dict[str, Any]:
    before = storage_identity(path)
    group: Any = zarr.open_group(str(path), mode="r")
    attrs = dict(group.attrs)
    require(
        attrs.get("complete") is True and attrs.get("binding") == expected,
        "completed sidecar binding differs",
    )
    n = expected["rows"]
    contract = raw.validate_wdl_metadata(
        group, attrs, n, expected["requested_wdl"], required=True
    )
    assert contract is not None
    require(
        set(attrs.get("source_array_sha256", {})) == set(COLUMNS),
        "missing source content proof",
    )
    require(
        all(
            isinstance(v, str) and len(v) == 64
            for v in attrs["source_array_sha256"].values()
        ),
        "malformed source content proof",
    )
    require(
        attrs["input"]["shape"] == [112, 8, 8]
        and attrs["input"]["encoding"] == HISTORY
        and attrs["input"]["dtype"] in ("float16", "float32"),
        "cached feed contract differs",
    )
    require(
        group["row_index"].dtype == np.dtype("uint64")
        and group["lc0_feed_sha256"].dtype == np.dtype("uint8")
        and group["game_id"].dtype.kind in "iu"
        and group["ply_index"].dtype.kind in "iu",
        "cached identity dtype differs",
    )
    hashes = {
        name: hashlib.sha256()
        for name in (
            "bt4_wdl_raw",
            "row_index",
            "game_id",
            "ply_index",
            "lc0_feed_sha256",
        )
    }
    require(set(group.array_keys()) == set(hashes), "sidecar arrays differ")
    for name in hashes:
        require(
            tuple(group[name].shape)
            == (
                (n, 3)
                if name == "bt4_wdl_raw"
                else (n, 32)
                if name == "lc0_feed_sha256"
                else (n,)
            ),
            "sidecar identity layout differs",
        )
        complete_chunks(group[name])
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        for name, digest in hashes.items():
            values = np.asarray(group[name][start:end])
            if name == "bt4_wdl_raw":
                raw.validate_wdl_values(values, end - start, contract)
            if name == "row_index":
                require(
                    np.array_equal(values, np.arange(start, end)),
                    "sidecar row order differs",
                )
            digest.update(values.tobytes(order="C"))
    require(
        {k: h.hexdigest() for k, h in hashes.items()} == attrs["array_sha256"],
        "sidecar content differs",
    )
    require(
        attrs["array_sha256"]["bt4_wdl_raw"] == contract["sha256"],
        "WDL checksum differs",
    )
    require(storage_identity(path) == before, "sidecar changed during verification")
    return attrs


def label_shard(
    args: argparse.Namespace,
    spec: dict[str, Any],
    summary: dict[str, Any],
    session: Any,
    input_name: str,
    input_dtype: np.dtype,
    contract: dict[str, str],
    guard: Any,
) -> dict[str, Any]:
    source = Path(args.source).resolve() / spec["path"]
    destination = Path(args.out).resolve() / spec["path"]
    writing = destination.with_name(destination.name + ".writing")
    state = storage_identity(source)
    expected = binding(args, spec, state)
    require(not writing.exists(), "partial sidecar exists; preserve and investigate")
    if destination.exists():
        cached = verify_cached(destination, expected, args.batch_size)
        require(
            storage_identity(source) == state,
            "source changed during cache verification",
        )
        return cached
    group = source_arrays(source, summary, spec["rows"])
    writing.mkdir()
    out: Any = zarr.open_group(str(writing), mode="w")
    n = spec["rows"]
    layouts = {
        "bt4_wdl_raw": ((n, 3), contract["dtype"]),
        "row_index": ((n,), "uint64"),
        "game_id": ((n,), group["game_id"].dtype),
        "ply_index": ((n,), group["ply_index"].dtype),
        "lc0_feed_sha256": ((n, 32), "uint8"),
    }
    for name, (shape, dtype) in layouts.items():
        out.create_dataset(
            name,
            shape=shape,
            chunks=(min(n, args.batch_size), *shape[1:]),
            dtype=dtype,
            compressor=Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE),
        )
    hashes = {name: hashlib.sha256() for name in layouts}
    source_hashes = {name: hashlib.sha256() for name in COLUMNS}
    for start in range(0, n, args.batch_size):
        guard()
        end = min(n, start + args.batch_size)
        batch = {name: np.asarray(group[name][start:end]) for name in COLUMNS}
        require(
            all(np.all(batch[name] == 1) for name in ("has_game_id", "has_ply_index")),
            "missing row identity",
        )
        require(
            all(np.all(batch[name] >= 0) for name in ("game_id", "ply_index")),
            "negative row identity",
        )
        feed = stored_feed(batch["x"]).astype(input_dtype, copy=False)
        for name, values in batch.items():
            source_hashes[name].update(values.tobytes(order="C"))
        returned = session.run([contract["output"]], {input_name: feed})
        require(len(returned) == 1, "missing requested WDL output")
        values = np.asarray(returned[0])
        raw.validate_wdl_values(values, end - start, contract)
        arrays = {
            "bt4_wdl_raw": values,
            "row_index": np.arange(start, end, dtype=np.uint64),
            "game_id": batch["game_id"],
            "ply_index": batch["ply_index"],
            "lc0_feed_sha256": row_digests(feed),
        }
        for name, value in arrays.items():
            out[name][start:end] = value
            require(
                np.array_equal(out[name][start:end], value),
                "stored sidecar readback differs",
            )
            hashes[name].update(value.tobytes(order="C"))
    guard()
    require(storage_identity(source) == state, "source changed during inference")
    require(
        file_sha256(Path(args.source) / SUMMARY) == args.expected_source_summary_sha256,
        "source summary changed",
    )
    attrs = {
        "complete": True,
        "binding": expected,
        "source_array_sha256": {k: h.hexdigest() for k, h in source_hashes.items()},
        "array_sha256": {k: h.hexdigest() for k, h in hashes.items()},
        "input": {
            "name": input_name,
            "dtype": str(input_dtype),
            "shape": [112, 8, 8],
            "encoding": HISTORY,
        },
        "providers": session.get_providers(),
        "wdl": {
            **contract,
            "schema": 1,
            "rows": n,
            "order": ["win", "draw", "loss"],
            "pov": "side_to_move",
            "semantic_basis": "explicit_named_output_contract",
            "sha256": hashes["bt4_wdl_raw"].hexdigest(),
        },
    }
    out.attrs.update(attrs)
    guard()
    check_g10_pin(args)
    writing.rename(destination)
    return attrs


def guard_resources(args: argparse.Namespace) -> None:
    if hasattr(args, "deadline"):
        require(time.monotonic() < args.deadline - 30, "wall deadline reached")
    require(
        not (Path(args.out) / "STOP").exists()
        and not (args.stop and Path(args.stop).exists()),
        "STOP requested",
    )
    require(
        shutil.disk_usage(args.out).free >= args.minimum_free_gib * 2**30,
        "disk reserve breached",
    )


def output_size(path: Path) -> int:
    size = 0
    for root, dirs, files in os.walk(path):
        for name in [*dirs, *files]:
            p = Path(root) / name
            try:
                require(not p.is_symlink(), "output symlink refused")
                if p.is_file():
                    size += p.stat().st_size
            except FileNotFoundError:
                # Atomic .writing rename may remove a sampled pathname.
                continue
    return size


def produce(args: argparse.Namespace) -> None:
    """Child owns the ORT session and both leases throughout teardown."""
    with raw.advisory_lease(
        Path(args.out) / ".writer.lock", poll_seconds=1, description="WDL writer"
    ):
        summary, specs = source_inventory(args)
        namespace = Path(args.out) / "g10_common_source.json"
        if getattr(args, "g10_admission", None) is not None:
            expected_namespace = g10_binding(args)
            if namespace.exists():
                require(g10.same(json.loads(namespace.read_text()), expected_namespace),
                        "G10 output namespace differs")
            else:
                require(not list(Path(args.out).glob("shard_*.zarr*")),
                        "G10 output has shards without a namespace receipt")
                raw.atomic_json(namespace, expected_namespace)
        else:
            require(not os.path.lexists(namespace),
                    "G10 output namespace requires matching qualification")
        guard_resources(args)
        model_state = storage_identity(Path(args.onnx))
        require(
            file_sha256(args.onnx) == args.expected_onnx_sha256, "teacher SHA differs"
        )
        require(
            storage_identity(Path(args.onnx)) == model_state,
            "teacher changed during hashing",
        )
        # Completed matching outputs need no session and no GPU lease.
        todo = []
        for spec in specs:
            target = Path(args.out) / spec["path"]
            require(
                not target.with_name(target.name + ".writing").exists(),
                "partial sidecar exists",
            )
            if target.exists():
                source = Path(args.source) / spec["path"]
                state = storage_identity(source)
                verify_cached(target, binding(args, spec, state), args.batch_size)
                require(
                    storage_identity(source) == state,
                    "source changed during cache verification",
                )
            else:
                todo.append(spec)

        def labels() -> None:
            session, input_name, dtype, providers = open_session(
                args.onnx, gpu_mem_gb=args.gpu_mem_gb, threads=args.threads
            )
            require(
                len(session.get_inputs()) == 1,
                "teacher requires unsupported extra inputs",
            )
            inp = session.get_inputs()[0]
            require(
                inp.type in ("tensor(float)", "tensor(float16)")
                and list(inp.shape[1:]) == [112, 8, 8],
                "teacher input layout differs",
            )
            require(
                not args.gpu_mem_gb or "CUDAExecutionProvider" in providers,
                "requested CUDA unavailable",
            )
            contract = raw.resolve_wdl_output(session, raw.requested_wdl(args), "")
            assert contract is not None

            def guard() -> None:
                guard_resources(args)
                require(
                    storage_identity(Path(args.onnx)) == model_state,
                    "teacher changed during inference",
                )

            for spec in todo:
                label_shard(
                    args, spec, summary, session, input_name, dtype, contract, guard
                )
                require(
                    storage_identity(Path(args.onnx)) == model_state,
                    "teacher changed during inference",
                )

        if todo:
            if args.gpu_mem_gb > 0:
                # Disposable child only: deliberately retain this OS descriptor
                # until process exit, including CUDA context teardown. Parent
                # monitors STOP/deadline while flock waits; no early LOCK_UN.
                gpu_path = Path(args.gpu_lock)
                gpu_path.parent.mkdir(parents=True, exist_ok=True)
                gpu_fd = os.open(gpu_path, os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.flock(gpu_fd, fcntl.LOCK_EX)
                guard_resources(args)
                labels()
            else:
                labels()
        guard_resources(args)
        require(
            file_sha256(Path(args.source) / SUMMARY)
            == args.expected_source_summary_sha256,
            "source summary changed",
        )
        require(
            storage_identity(Path(args.onnx)) == model_state,
            "teacher changed before completion",
        )
        for spec in specs:
            target = zarr.open_group(str(Path(args.out) / spec["path"]), mode="r")
            require(
                storage_identity(Path(args.source) / spec["path"])
                == target.attrs["binding"]["source_storage_identity"],
                "source changed before completion",
            )
        if getattr(args, "g10_admission", None) is not None:
            check_g10_pin(args)
            current = g10.admit(Path(args.g10_common_qualification),
                                args.expected_g10_common_qualification_sha256,
                                Path(args.source).resolve(), summary)
            require(g10.same(current, args.g10_admission), "G10 admission changed")
        guard_resources(args)
        raw.atomic_json(
            Path(args.invocation) / "child_completed.json",
            {
                "schema": 1,
                "complete": True,
                "rows": sum(s["rows"] for s in specs),
                "shards": len(specs),
                "new_shards": len(todo),
                **g10_binding(args),
                "source_summary_sha256": args.expected_source_summary_sha256,
                "selection": specs,
                "history_lineage": LINEAGE,
                "scope": "Selected WDL bank only; no training integration",
            },
        )


def child(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    set_nthreads(args.threads)
    raw.atomic_json(
        Path(args.invocation) / "child_started.json",
        {
            "pid": os.getpid(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "blosc_threads": get_nthreads(),
            "ort_threads_requested": args.threads,
            "numpy": np.__version__,
        },
    )
    produce(args)


def run(args: argparse.Namespace) -> int:
    require(
        args.batch_size > 0
        and args.threads > 0
        and args.max_shards > 0
        and args.start_shard >= 0,
        "invalid count bounds",
    )
    require(
        all(
            math.isfinite(v) and v >= 0
            for v in (args.minimum_free_gib, args.gpu_mem_gb)
        )
        and math.isfinite(args.max_output_gib)
        and args.max_output_gib > 0
        and math.isfinite(args.max_seconds)
        and args.max_seconds > 30,
        "invalid resource bounds",
    )
    raw.requested_wdl(args)
    require(
        args.gpu_mem_gb == 0 or (args.gpu_lock and Path(args.gpu_lock).is_absolute()),
        "GPU execution requires explicit absolute shared --gpu-lock",
    )
    args.source, args.out, args.onnx = (
        str(Path(p).resolve()) for p in (args.source, args.out, args.onnx)
    )
    source, out = Path(args.source), Path(args.out)
    require(
        source != out and source not in out.parents and out not in source.parents,
        "output overlaps source",
    )
    out.mkdir(parents=True, exist_ok=True)
    invocation = out / "invocations" / str(time.time_ns())
    invocation.mkdir(parents=True)
    args.invocation = str(invocation)
    raw.atomic_json(
        invocation / "started.json",
        {
            "argv": vars(args),
            "started_unix": time.time(),
            "limits": "Whole invocation deadline includes 30s cleanup; reserve/STOP sampled every 2s, output every 10s, may overshoot transiently.",
        },
    )
    deadline = time.monotonic() + args.max_seconds
    args.deadline = deadline

    def interrupted(_signum: int, _frame: Any) -> None:
        raise InterruptedError("termination requested")

    previous = signal.signal(signal.SIGTERM, interrupted)
    process = multiprocessing.get_context("fork").Process(target=child, args=(args,))
    try:
        guard_resources(args)
        process.start()
        next_size_check = 0.0
        while process.is_alive():
            guard_resources(args)
            if time.monotonic() >= next_size_check:
                require(
                    output_size(out) <= args.max_output_gib * 2**30,
                    "sampled output cap breached",
                )
                next_size_check = time.monotonic() + 10
            require(time.monotonic() < deadline - 30, "wall deadline reached")
            process.join(timeout=2)
        require(process.exitcode == 0, f"WDL child exited {process.exitcode}")
        guard_resources(args)
        require(
            output_size(out) <= args.max_output_gib * 2**30, "final output cap breached"
        )
        receipt = json.loads((invocation / "child_completed.json").read_text())
        receipt["ended_unix"] = time.time()
        raw.atomic_json(invocation / "completed.json", receipt)
        return 0
    except BaseException as exc:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=min(25, max(0, deadline - time.monotonic() - 2)))
            if process.is_alive():
                process.kill()
                process.join()
        raw.atomic_json(
            invocation / "failed.json", {"error": repr(exc), "ended_unix": time.time()}
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "out",
        "onnx",
        "expected-source-summary-sha256",
        "expected-onnx-sha256",
        "wdl-output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--wdl-output-kind", required=True, choices=["logits", "probabilities"]
    )
    parser.add_argument("--g10-common-qualification",
                        help="Pinned completed G10 common-input receipt for this derived batch")
    parser.add_argument("--expected-g10-common-qualification-sha256")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gpu-mem-gb", type=float, default=0)
    parser.add_argument("--gpu-lock")
    parser.add_argument("--minimum-free-gib", type=float, default=150)
    parser.add_argument("--max-output-gib", type=float, default=1)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--stop")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
