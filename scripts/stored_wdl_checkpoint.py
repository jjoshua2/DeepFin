"""Process-isolated checkpoint qualification and explicit bounded stored forwards.

Qualification remains load-only. Forward calls never imply data admission.

The child hashes and loads one immutable snapshot (<=2 GiB) using weights_only.
A successful load authenticates bytes against a supplied hash, not that receipt's
origin or training history. Production holdout qualification remains external.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import signal
import subprocess
import sys

MAX_REQUEST_BYTES = 24 * 1024**2
MAX_CHECKPOINT_BYTES = 2 * 1024**3
# Counts alias-expanded state entries, not unique parameters; compatibility with
# the 61.4M-unique production donor is NOT established by this ceiling.
MAX_STATE_ELEMENTS = 80_000_000


def qualify_checkpoint(path: Path, sha256: str, architecture_json: str) -> dict:
    """Run a fresh CPU process; never change caller Torch/encoding global state."""
    return _run_child({"path": str(path.resolve()), "sha256": sha256,
                       "architecture_json": architecture_json})


def _run_child(payload: dict) -> dict:
    request = json.dumps(payload, allow_nan=False)
    if len(request) > MAX_REQUEST_BYTES:
        raise ValueError("oversized qualification request")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    process = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--child"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, start_new_session=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    try:
        stdout, stderr = process.communicate(request, timeout=90)
    except BaseException as error:
        # Reap our isolated group on cancellation/interruption as well as timeout.
        # No other workload's process or process group is touched.
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.communicate()
        if isinstance(error, subprocess.TimeoutExpired):
            raise ValueError("isolated checkpoint qualification timed out") from None
        raise
    if process.returncode:
        raise ValueError("isolated checkpoint qualification failed: " + stderr[-2000:])
    if len(stdout) > MAX_REQUEST_BYTES:
        raise ValueError("oversized child response")
    return json.loads(stdout)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _qualify(request: dict, *, verified_model_callback=None) -> dict:
    """Internal child seam: caller-supplied Python callback, never request-selected.

    Public serialized qualification/forward routes pass no callback. An internal
    bounded diagnostic may consume the strictly verified model after load; it
    owns additional data admission and must bind callback source independently.
    """
    # Import inside child, after resource limits. No Torch import in caller.
    import contextlib
    import resource
    from dataclasses import asdict

    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "chess_anti_engine"], cwd=repo, check=True)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "chess_anti_engine"],
        cwd=repo, text=True,
    )
    if any(Path(name).suffix in {".py", ".pyc", ".so", ".pyd"} for name in untracked.splitlines()):
        raise ValueError("untracked project import source invalidates source receipt")
    # These checks precede project imports. Dependencies/interpreter are not
    # authenticated by the project revision or the loader-content receipt.
    import torch
    sys.path.insert(0, str(repo))
    from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
    from chess_anti_engine.uci.model_loader import model_config_from_arch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Decode and validate forward bytes before checkpoint access. The explicit
    # operation gate keeps the existing qualification API strictly load-only.
    operation = request.get("operation", "qualify")
    if operation not in {"qualify", "forward"}:
        raise ValueError("unsupported checkpoint operation")
    planes = _decode_planes(request) if operation == "forward" else None
    expected = request["sha256"]
    if (not isinstance(expected, str) or len(expected) != 64 or
            any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("exact lowercase SHA256 required")
    declared = json.loads(request["architecture_json"])
    if not isinstance(declared, dict):
        raise ValueError("full architecture receipt required")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(request["path"], flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_CHECKPOINT_BYTES:
            raise ValueError("bounded regular checkpoint required")
        content = stream.read(MAX_CHECKPOINT_BYTES + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, f) != getattr(after, f) for f in fields):
        raise ValueError("checkpoint changed while snapshotting")
    if len(content) != before.st_size or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("checkpoint byte identity mismatch")
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("arch"), dict):
        raise ValueError("embedded architecture required; no fallback")
    arch = payload["arch"]
    required = {"_schema_version", *asdict(ModelConfig())}
    if (set(arch) != required or type(arch.get("_schema_version")) is not int or
            arch["_schema_version"] != ARCH_SCHEMA_VERSION or _canonical(arch) != _canonical(declared)):
        raise ValueError("exact full current architecture receipt mismatch")
    if (arch["kind"] != "transformer" or
            arch["input_history_encoding"] != "lc0_root_legacy_meta" or
            arch["input_extra_features"] != "v2_threats" or
            type(arch["history_rep_fix"]) is not bool or
            arch["use_dynamic_relations"] is not False or arch["policy_dynamic_relations"] is not False):
        raise ValueError("unsupported stored175 transformer encoding/topology")
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("model tensor mapping required")
    if any(not isinstance(v, torch.Tensor) or v.layout != torch.strided or
           v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError("finite dense float32 state required")
    if sum(v.numel() for v in state.values()) > MAX_STATE_ELEMENTS:
        raise ValueError("checkpoint state exceeds bounded parameter budget")
    # Architecture allocation and global rep_fix changes occur ONLY in this
    # resource-limited disposable process, not in the diagnostic caller.
    with contextlib.redirect_stdout(sys.stderr):
        model = build_model(model_config_from_arch(arch))
    reference = model.state_dict()
    if set(state) != set(reference) or any(state[k].shape != v.shape or state[k].dtype != v.dtype
                                         for k, v in reference.items()):
        raise ValueError("strict complete state shape/dtype mismatch")
    _validate_aliases(model, state)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if not hasattr(model, "value_wdl"):
        raise ValueError("required search WDL head absent")
    report = {"schema": "isolated-checkpoint-load/v1", "sha256": expected,
            "architecture": arch, "bytes": len(content), "state_tensors": len(state),
            "head_declared": "wdl", "pov_declared": "stored row side to move",
            "output_semantics_measured": False, "inference_performed": False,
            "production_qualified": False, "receipt_authenticity_verified": False,
            "source_revision": revision,
            "loader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits": {"address_space_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                       "cpu_seconds": resource.getrlimit(resource.RLIMIT_CPU)[0],
                       "torch_threads": torch.get_num_threads(),
                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}

    if planes is not None:
        import numpy as np
        batch = request["batch_size"]
        probabilities = []
        with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
            for start in range(0, len(planes), batch):
                output = model(torch.from_numpy(planes[start:start + batch].copy()))
                logits = output["wdl"]
                if (logits.dtype != torch.float32 or logits.device.type != "cpu" or
                        logits.shape != (min(batch, len(planes) - start), 3) or
                        not bool(torch.isfinite(logits).all())):
                    raise ValueError("invalid search WDL logits")
                probabilities.append(torch.softmax(logits, dim=1).numpy())
        prediction = np.concatenate(probabilities)
        report.update(schema="isolated-stored-wdl-forward/v1", inference_performed=True,
                      rows_sha256=request["rows_sha256"], rows=len(planes), batch_size=batch,
                      probabilities=prediction.tolist(), wdl_order_declared=["W", "D", "L"],
                      controls_performed=[], controls_required_but_not_run=["donor-donor", "batch-invariance", "POV"])
    if verified_model_callback is not None:
        if operation != "qualify" or not callable(verified_model_callback):
            raise ValueError("internal model consumer requires load-only operation")
        report["diagnostic"] = verified_model_callback(model, dict(report))
    return report


def _decode_planes(request: dict):
    import numpy as np
    from scripts.stored_wdl_compare import rows_digest
    count, batch = request["rows"], request["batch_size"]
    if type(count) is not int or not 1 <= count <= 256 or type(batch) is not int or not 1 <= batch <= 64:
        raise ValueError("bounded row/batch counts required")
    encoded = request["planes_base64"]
    byte_count = count * 175 * 8 * 8 * 4
    if not isinstance(encoded, str) or len(encoded) != 4 * ((byte_count + 2) // 3):
        raise ValueError("stored row byte count mismatch")
    content = base64.b64decode(encoded, validate=True)
    if len(content) != byte_count:
        raise ValueError("stored row byte count mismatch")
    planes = np.frombuffer(content, dtype="<f4").reshape(count, 175, 8, 8)
    if rows_digest(planes, request["row_ids"], request["group_ids"]) != request["rows_sha256"]:
        raise ValueError("stored row identity mismatch")
    return planes


def forward_checkpoint(path: Path, sha256: str, architecture_json: str, planes, *,
                       row_ids, group_ids, expected_rows_sha256: str, batch_size: int = 16) -> dict:
    """Explicit isolated CPU forward, no target scoring or provenance inference.

    Caller-supplied stored planes must already be qualified for this exact
    checkpoint's encoding. WDL order/STM POV are source contracts, not measured
    semantic controls. Real controls must be run and recorded separately.
    """
    import numpy as np
    from scripts.stored_wdl_compare import rows_digest
    row_ids, group_ids = tuple(row_ids), tuple(group_ids)
    if (planes.dtype != np.float32 or len(planes) > 256 or
            type(batch_size) is not int or not 1 <= batch_size <= 64):
        raise ValueError("bounded float32 rows/batch required")
    snapshot = planes.copy()
    if rows_digest(snapshot, row_ids, group_ids) != expected_rows_sha256:
        raise ValueError("stored row identity mismatch")
    return _run_child({"operation": "forward", "path": str(path.resolve()), "sha256": sha256,
                       "architecture_json": architecture_json, "rows": len(snapshot),
                       "batch_size": batch_size, "row_ids": row_ids, "group_ids": group_ids,
                       "rows_sha256": expected_rows_sha256,
                       "planes_base64": base64.b64encode(np.ascontiguousarray(snapshot, dtype="<f4").tobytes()).decode("ascii")})


def _validate_aliases(model, state: dict) -> None:
    import torch
    aliases: dict[int, str] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        previous = aliases.setdefault(id(parameter), name)
        if not torch.equal(state[previous], state[name]):
            raise ValueError("conflicting shared-parameter aliases")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", required=True)
    parser.parse_args()
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    raw = sys.stdin.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("oversized qualification request")
    print(json.dumps(_qualify(json.loads(raw)), allow_nan=False))


if __name__ == "__main__":
    main()
