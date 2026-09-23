"""Observe an actual pinned lc0_control_train run; scheduling belongs to its caller.

Use the same wrapper for both arms. Full tensor parity must be qualified separately.
This records compact, globally remapped game_id/ply_index order and initial model state.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import resource
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


def state_digest(value):
    """Canonical nested tensor/state digest; file serialization is not identity."""
    import torch
    h = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu().contiguous()
            h.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            h.update(f'dict:{len(obj)}'.encode())
            for key in sorted(obj, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            h.update(f'{type(obj).__name__}:{len(obj)}'.encode())
            for item in obj:
                visit(item)
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            h.update(json.dumps([type(obj).__name__, obj], allow_nan=False).encode())
        else:
            raise TypeError(f'unsupported state type: {type(obj)}')
    visit(value)
    return h.hexdigest()


def observe(driver, receipt, argv):
    """Driver is loaded from the frozen runtime; no training semantics are changed."""
    started = time.monotonic()
    record: dict[str, Any] = {"status": "INCOMPLETE", "argv": argv, "batches": [],
              "first_batch_seconds": None, "observer_seconds": 0.0}
    original_build = driver.build_model
    buffer_class = driver.GameAwareEpochBuffer
    original_sample = buffer_class.sample_batch_arrays
    original_init = driver.Trainer.__init__
    original_save = driver.Trainer.save
    original_overlap = driver.Trainer._iter_exact_overlapped_batches
    original_step = driver.Trainer._run_optimizer_step
    record['overlap_batches_consumed'] = 0
    update_loss_tensors = []

    def overlapped(self, *args, **kwargs):
        iterator = original_overlap(self, *args, **kwargs)
        try:
            for batch in iterator:
                record['overlap_batches_consumed'] += 1
                yield batch
        finally:
            iterator.close()
    import torch

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        tick = time.monotonic()
        record['initial_optimizer_sha256'] = state_digest({
            'opt': self.opt.state_dict(), 'scheduler': self._scheduler.state_dict(),
            'zclip': self.zclip_state_dict(), 'peak_lr': float(self._peak_lr)})
        record['observer_seconds'] += time.monotonic() - tick

    def optimizer_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        sums = kwargs['step_sums']
        loss = sums.tensor('loss')
        rows = int(kwargs['step_opt_stats']['samples_seen'])
        if loss is None or rows <= 0:
            raise ValueError('optimizer update has no measured loss or rows')
        # Keep detached device scalars until the epoch ends. One final transfer
        # observes every update without adding a GPU synchronization per step.
        update_loss_tensors.append((loss / rows).detach())
        return result

    def save(self, path):
        original_save(self, path)
        if Path(path).name == 'checkpoint.pt':
            tick = time.monotonic()
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            record['final_model_sha256'] = state_digest(checkpoint['model'])
            record['final_optimizer_sha256'] = state_digest({
                key: checkpoint[key] for key in ('opt', 'scheduler', 'zclip', 'peak_lr')})
            record['final_checkpoint_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            record['final_checkpoint_path'] = str(path)
            record['final_step'] = int(checkpoint['step'])
            record['observer_seconds'] += time.monotonic() - tick


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
        record['host_batch_overlap'] = bool(self.host_batch_overlap)
        wait = time.monotonic() - tick
        tick = time.monotonic()
        if record["first_batch_seconds"] is None:
            record["first_batch_seconds"] = tick - started
        # game_id has already been remapped to globally source-qualified keys.
        if any(key not in batch for key in ("game_id", "ply_index", "has_game_id", "has_ply_index")):
            raise ValueError("actual trainer batches lack qualified game/ply identity")
        if not batch["has_game_id"].all() or not batch["has_ply_index"].all():
            raise ValueError("actual trainer batches contain missing game/ply identity")
        record["batches"].append({"rows": len(batch["game_id"]),
            "order_sha256": arrays_digest({k: batch[k] for k in ("game_id", "ply_index")}),
            "sampler_seconds": wait})
        record["observer_seconds"] += time.monotonic() - tick
        return batch

    driver.Trainer._iter_exact_overlapped_batches = overlapped
    driver.Trainer._run_optimizer_step = optimizer_step
    driver.Trainer.__init__ = initialize
    driver.Trainer.save = save
    driver.build_model = build
    buffer_class.sample_batch_arrays = sample
    try:
        result = driver.main(argv)
        if result not in (None, 0):
            raise RuntimeError(f"trainer returned {result}")
        tick = time.monotonic()
        losses = torch.stack(update_loss_tensors).cpu().tolist() if update_loss_tensors else []
        record['update_losses'] = [v if math.isfinite(v) else str(v) for v in losses]
        record['observer_seconds'] += time.monotonic() - tick
        record["status"] = ("TRAINER_RETURNED_SUCCESS" if all(math.isfinite(v) for v in losses)
                            else "TRAINER_RETURNED_NONFINITE_UPDATE")
        return result
    finally:
        driver.Trainer._iter_exact_overlapped_batches = original_overlap
        driver.Trainer._run_optimizer_step = original_step
        driver.Trainer.__init__ = original_init
        driver.Trainer.save = original_save
        record['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        record['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
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
