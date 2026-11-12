"""Hard-limited process wrapper for the descriptive stored-row prefix reader.

No model loading/inference or training. Success is a validated result.json marker
in a unique output directory. Failure retains an incomplete directory for audit,
never replaces an existing artifact or deletes source data.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

MAX_REQUEST = 65536
MAX_RECEIPT = 4 * 1024**2
MAX_FILE = 64 * 1024**2
WALL_SECONDS = 150


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            h.update(chunk)
    return h.hexdigest()


def _sync_directory(path: Path) -> None:
    """Persist file/link creation in the containing directory on supported Linux."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run_reader(audit_path: Path, audit_sha256: str, shard_paths: tuple[Path, ...],
               output_root: Path, *, expected_history_rep_fix: bool, count: int = 256) -> dict:
    """Return the durable success receipt, never launch an unbounded in-process scan."""
    if type(expected_history_rep_fix) is not bool or type(count) is not int or not 1 <= count <= 256:
        raise ValueError("explicit history boolean and bounded selection count required")
    if not 1 <= len(shard_paths) <= 32:
        raise ValueError("bounded explicit shard prefix required")
    output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="stored-wdl-prefix-", dir=output_root))
    _sync_directory(output_root)
    payload = {"audit_path": str(audit_path.resolve()), "audit_sha256": audit_sha256,
               "shard_paths": [str(p.absolute()) for p in shard_paths],
               "output_dir": str(out.resolve()), "count": count,
               "expected_history_rep_fix": expected_history_rep_fix}
    request = json.dumps(payload, allow_nan=False)
    if len(request.encode()) > MAX_REQUEST:
        raise ValueError("oversized reader request")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--child"],
            stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True,
            env=env, cwd=Path(__file__).resolve().parents[1], start_new_session=True,
        )
        try:
            process.communicate(request, timeout=WALL_SECONDS)
        except BaseException as error:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.communicate()
            if isinstance(error, subprocess.TimeoutExpired):
                raise ValueError(f"reader timed out; incomplete artifacts retained at {out}") from None
            raise
    if process.returncode:
        with (out / "stderr.log").open("rb") as stream:
            stream.seek(max(0, (out / "stderr.log").stat().st_size - 2000))
            detail = stream.read(2000).decode(errors="replace")
        raise ValueError(f"reader failed; incomplete artifacts at {out}: {detail}")
    receipt_path = out / "validated-candidate.json"
    if receipt_path.stat().st_size > MAX_RECEIPT:
        raise ValueError("oversized reader receipt")
    receipt = json.loads(receipt_path.read_bytes())
    artifact_path = out / "rows.npz"
    if not 0 < artifact_path.stat().st_size <= MAX_FILE:
        raise ValueError("reader artifact size exceeds cap")
    if (receipt.get("status") != "COMPLETE_DESCRIPTIVE_PREFIX" or
            receipt.get("audit_sha256") != audit_sha256 or
            receipt.get("requested_shard_paths") != payload["shard_paths"] or
            receipt.get("expected_history_rep_fix") is not expected_history_rep_fix or
            receipt.get("artifact_sha256") != _digest(out / "rows.npz")):
        raise ValueError("reader output identity mismatch")
    # An exclusive hard link atomically exposes the fully fsynced candidate.
    # No partial JSON success marker can appear if the parent is interrupted.
    os.link(receipt_path, out / "result.json")
    _sync_directory(out)
    return {"output_dir": str(out), "receipt_sha256": _digest(out / "result.json"), **receipt}


def _source_snapshot(repo: Path) -> dict[str, str]:
    """Hash relevant source and ignored native images; no build or dependency change."""
    names = subprocess.check_output(
        ["git", "ls-files", "--", "chess_anti_engine", "scripts/match_audit_rows.py",
         "scripts/stored_wdl_rows.py"], cwd=repo, text=True).splitlines()
    paths = [repo / name for name in names if Path(name).suffix in {".py", ".c", ".h"}]
    paths += [Path(__file__).resolve()]
    paths += sorted((repo / "chess_anti_engine").rglob("*.so"))
    return {str(p.relative_to(repo)): _digest(p) for p in paths}


def _execute(request: dict) -> dict:
    import importlib.metadata

    repo = Path(__file__).resolve().parents[1]
    source_before = _source_snapshot(repo)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "chess_anti_engine"], cwd=repo, check=True)
    sys.path.insert(0, str(repo))
    import numpy as np
    from scripts.stored_wdl_rows import MAX_AUDIT_BYTES, scan_prefix
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    audit_path = Path(request["audit_path"])
    with audit_path.open("rb") as stream:
        audit = stream.read(MAX_AUDIT_BYTES + 1)
    if len(audit) > MAX_AUDIT_BYTES:
        raise ValueError("audit size exceeds cap")
    result = scan_prefix(
        audit, request["audit_sha256"], tuple(Path(p) for p in request["shard_paths"]),
        expected_history_rep_fix=request["expected_history_rep_fix"], count=request["count"],
    )
    # Snapshot known imported helpers and native dependencies, not merely git HEAD.
    imported = {}
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and (name.startswith(("chess_anti_engine", "scripts.")) or Path(file).suffix == ".so"):
            path = Path(file).resolve()
            if path.is_file():
                imported[name] = {"path": str(path), "sha256": _digest(path)}
    if _source_snapshot(repo) != source_before:
        raise ValueError("project source/native image changed during reader execution")
    out = Path(request["output_dir"])
    x, targets = result.pop("x"), result.pop("targets")
    if x.nbytes + targets.nbytes > 16 * 1024**2:
        raise ValueError("oversized reconstructed arrays")
    artifact = out / "rows.npz"
    with artifact.open("xb") as stream:
        np.savez(stream, x=x, targets=targets)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(out)
    with np.load(artifact, allow_pickle=False) as saved:
        if (set(saved.files) != {"x", "targets"} or
                saved["x"].dtype != np.dtype("float32") or
                not np.array_equal(saved["x"], x) or not np.array_equal(saved["targets"], targets)):
            raise ValueError("reconstructed artifact validation failed")
    import resource
    receipt = {
        **result, "status": "COMPLETE_DESCRIPTIVE_PREFIX",
        "artifact_sha256": _digest(artifact), "artifact_bytes": artifact.stat().st_size,
        "requested_shard_paths": request["shard_paths"],
        "source_revision": revision, "project_source_sha256": source_before,
        "loaded_imports": imported,
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "zarr", "numcodecs", "python-chess")},
        "python": sys.version, "inference_performed": False,
        "limits": {**result["limits"],
                   "address_space_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                   "cpu_seconds": resource.getrlimit(resource.RLIMIT_CPU)[0],
                   "file_bytes": resource.getrlimit(resource.RLIMIT_FSIZE)[0],
                   "wall_seconds": WALL_SECONDS,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                   "torch_threads": torch.get_num_threads()},
        "source_authentication_limit": "observed source/import bytes; not historical corpus authentication",
    }
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > MAX_RECEIPT:
        raise ValueError("oversized success receipt")
    # Candidate only: parent writes success marker after clean exit and verification.
    with (out / "validated-candidate.json").open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(out)
    return receipt


def main() -> None:
    if sys.argv[1:] != ["--child"]:
        raise SystemExit("use run_reader; child-only entry point")
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE, MAX_FILE))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
    if len(raw) > MAX_REQUEST:
        raise ValueError("oversized reader request")
    _execute(json.loads(raw))


if __name__ == "__main__":
    main()
