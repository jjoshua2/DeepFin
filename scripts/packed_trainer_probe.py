"""Observe an actual pinned lc0_control_train run; scheduling belongs to its caller.

Use the same wrapper for both arms. Full tensor parity must be qualified separately.
This records compact, globally remapped game_id/ply order and initial model state.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def observe(driver, receipt, argv):
    """Driver is loaded from the frozen runtime; no training semantics are changed."""
    started = time.monotonic()
    record: dict[str, Any] = {"status": "INCOMPLETE", "argv": argv, "batches": [],
              "first_batch_seconds": None, "observer_seconds": 0.0}
    original_build = driver.build_model
    buffer_class = driver.GameAwareEpochBuffer
    original_sample = buffer_class.sample_batch_arrays

    def build(*args, **kwargs):
        model = original_build(*args, **kwargs)
        tick = time.monotonic()
        record["initial_model_sha256"] = arrays_digest({
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        })
        record["observer_seconds"] += time.monotonic() - tick
        return model

    def sample(self, *args, **kwargs):
        tick = time.monotonic()
        batch = original_sample(self, *args, **kwargs)
        wait = time.monotonic() - tick
        tick = time.monotonic()
        if record["first_batch_seconds"] is None:
            record["first_batch_seconds"] = tick - started
        # game_id has already been remapped to globally source-qualified keys.
        if "game_id" not in batch or "ply" not in batch:
            raise ValueError("actual trainer batches lack qualified game/ply identity")
        record["batches"].append({"rows": len(batch["game_id"]),
            "order_sha256": arrays_digest({k: batch[k] for k in ("game_id", "ply")}),
            "sampler_seconds": wait})
        record["observer_seconds"] += time.monotonic() - tick
        return batch

    driver.build_model = build
    buffer_class.sample_batch_arrays = sample
    try:
        result = driver.main(argv)
        if result not in (None, 0):
            raise RuntimeError(f"trainer returned {result}")
        record["status"] = "TRAINER_RETURNED_SUCCESS"
        return result
    finally:
        driver.build_model = original_build
        buffer_class.sample_batch_arrays = original_sample
        record["wall_seconds"] = time.monotonic() - started
        with receipt.open("x") as stream:
            json.dump(record, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-lease-path", type=Path, required=True)
    parser.add_argument("--gpu-lease-fd", type=int, required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    held, expected = os.fstat(args.gpu_lease_fd), args.gpu_lease_path.stat()
    if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("foreign inherited GPU lease")
    fcntl.flock(args.gpu_lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    runtime = args.runtime.resolve(strict=True)
    path = runtime / "scripts/lc0_control_train.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != args.driver_sha256:
        raise ValueError("frozen trainer driver digest mismatch")
    # The parent must validate the complete runtime roster and environment pins.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location("packed_probe_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned trainer module")
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    argv = args.trainer_args
    if argv and argv[0] == "--":
        argv = argv[1:]
    return observe(driver, args.receipt, argv)


if __name__ == "__main__":
    raise SystemExit(main())
