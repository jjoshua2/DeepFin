"""Explicitly reviewed CPU-only full-stream qualification; no GPU training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

if __package__:
    from .run_host_overlap_pair import (CONFIG_SHA256, INPUT_ROOT, PREPARATION_SHA256, RUNTIME_ROOT,
                                        RUNTIME_COMMIT, authenticate_sources, qualified_batch_count)
    from .run_packed_trainer_preparation import digest, inventory
    from .packed_trainer_probe import arrays_digest
else:
    from run_host_overlap_pair import (CONFIG_SHA256, INPUT_ROOT, PREPARATION_SHA256, RUNTIME_ROOT,  # pyright: ignore[reportImplicitRelativeImport]
                                       RUNTIME_COMMIT, authenticate_sources, qualified_batch_count)
    from run_packed_trainer_preparation import digest, inventory  # pyright: ignore[reportImplicitRelativeImport]
    from packed_trainer_probe import arrays_digest  # pyright: ignore[reportImplicitRelativeImport]


def write_atomic(path, value):
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def make_host_trainer(driver, cfg, model_cfg):
    """Use frozen host methods and config defaults without building the 61M model."""
    from chess_anti_engine.moves import MODEL_POLICY_SIZE
    from chess_anti_engine.train.trainer import _SfRebuildCoverageAccumulator

    kwargs = driver.trainer_kwargs_from_config(cfg)
    params = inspect.signature(driver.Trainer.__init__).parameters
    host = object.__new__(driver.Trainer)
    host.device = 'cpu'
    host.model = SimpleNamespace(policy_size=MODEL_POLICY_SIZE)
    host._input_history_encoding = model_cfg.input_history_encoding
    host._sf_rebuild_coverage = _SfRebuildCoverageAccumulator()
    for name in ('mirror_prob', 'rebuild_sf_targets', 'sf_policy_sparse_ce',
                 'sf_target_params', 'rebuild_categorical_target',
                 'categorical_target_params', 'soft_policy_min_tv',
                 'policy_target_temp', 'sf_wdl_conf_power', 'sf_wdl_draw_scale',
                 'sf_wdl_temperature'):
        setattr(host, name, kwargs.get(name, params[name].default))
    return host


def measure(buffer, trainer, enabled, guard):
    """Consume the frozen trainer's real OFF/ON host iterator, without updates."""
    raw, prepared, order = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    schedule, producer_threads, preparation_threads = [], set(), set()
    overlap_consumed = 0
    started = time.monotonic()
    plan = buffer.plan.as_dict()
    original_sample = buffer.sample_batch_arrays
    original_prepare = trainer._prepare_host_arrays
    original_overlap = trainer._iter_exact_overlapped_batches

    def sample(*args, **kwargs):
        guard()
        arrays = original_sample(*args, **kwargs)
        producer_threads.add(threading.current_thread().name)
        raw.update(arrays_digest(arrays).encode())
        order.update(arrays_digest({key: arrays[key] for key in ('game_id', 'ply_index')}).encode())
        return arrays

    def prepare(*args, **kwargs):
        arrays = original_prepare(*args, **kwargs)
        preparation_threads.add(threading.current_thread().name)
        prepared.update(arrays_digest(arrays).encode())
        return arrays

    def overlapped(*args, **kwargs):
        nonlocal overlap_consumed
        for batch in original_overlap(*args, **kwargs):
            overlap_consumed += 1
            yield batch

    buffer.sample_batch_arrays = sample
    trainer._prepare_host_arrays = prepare
    trainer._iter_exact_overlapped_batches = overlapped
    try:
        remaining = buffer.num_batches
        while remaining:
            count = min(88, remaining)
            for tensors in trainer._iter_training_batches(
                buffer, batch_size=buffer.plan.batch_size,
                mirror_prob=trainer.mirror_prob, count=count):
                guard()
                schedule.append(int(tensors['x'].shape[0]))
                del tensors
            remaining -= count
        receipt = buffer.receipt()
        return {'enabled': enabled, 'rows': sum(schedule), 'batch_rows': schedule,
                'complete': receipt['complete'], 'same_game_repeats_max': receipt['same_game_repeats_max'],
                'host_overlap_reserve_bytes': plan.get('host_overlap_reserve_bytes', 0),
                'overlap_batches_consumed': overlap_consumed,
                'producer_threads': sorted(producer_threads),
                'preparation_threads': sorted(preparation_threads),
                'plan': plan, 'receipt': receipt,
                'raw_sequence_sha256': raw.hexdigest(),
                'prepared_sequence_sha256': prepared.hexdigest(),
                'order_sequence_sha256': order.hexdigest(),
                'seconds': time.monotonic() - started}
    finally:
        del buffer.sample_batch_arrays
        del trainer._prepare_host_arrays
        del trainer._iter_exact_overlapped_batches
        buffer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.plan_sha256 or digest(args.review) != args.review_sha256:
        raise ValueError('CPU plan/review bytes differ')
    plan = json.loads(args.plan.read_text())
    review = json.loads(args.review.read_text())
    if review['status'] != 'APPROVED' or review['plan_sha256'] != args.plan_sha256:
        raise ValueError('CPU plan not independently reviewed')
    if plan['status'] != 'APPROVED_HOST_OVERLAP_CPU_QUALIFICATION':
        raise ValueError('CPU plan not admitted')
    if digest(__file__) != plan['cpu_producer_sha256']:
        raise ValueError('reviewed CPU producer bytes differ')
    if (plan['runtime_commit'] != RUNTIME_COMMIT or plan['root'] != INPUT_ROOT
            or Path(plan['runtime']).resolve(strict=True) != Path(RUNTIME_ROOT).resolve(strict=True)
            or digest(plan['config']) != CONFIG_SHA256
            or digest(plan['preparation_receipt']) != PREPARATION_SHA256):
        raise ValueError('CPU plan differs from preregistered runtime/bank/config')
    if os.path.abspath(plan['python']) != os.path.abspath(sys.executable):
        raise ValueError('CPU qualification Python interpreter differs from plan')
    seconds = int(plan['cpu_qualification_seconds'])
    if not 1 <= seconds <= 1800:
        raise ValueError('separate CPU qualification cap must be within 30min')
    started = time.monotonic()
    def interrupted(signum, _frame):
        raise RuntimeError(f'CPU qualification signal {signum}')
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    os.sched_setaffinity(0, {14, 15})
    os.nice(19)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '2'
    runtime, out = Path(plan['runtime']), Path(plan['out'])
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    terminal: dict[str, Any] = {'status': 'INCOMPLETE'}
    def guard():
        if (out / 'STOP').exists() or time.monotonic() - started >= seconds:
            raise RuntimeError('CPU qualification STOP/deadline')
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(memory['MemAvailable'].split()[0]) * 1024 < 32 * 1024**3:
            raise RuntimeError('32GiB available RAM floor')
        if shutil.disk_usage(out).free < 150 * 1024**3:
            raise RuntimeError('150GiB disk floor')
    def repin():
        guard()
        for pin in plan['pins']:
            if digest(pin['path']) != pin['sha256']:
                raise ValueError('CPU pin drift')
        if inventory(runtime) != plan['runtime_files']:
            raise ValueError('CPU runtime roster drift')
        head = subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], timeout=10, text=True).strip()
        if head != plan['runtime_commit']:
            raise ValueError('CPU runtime HEAD differs')
    try:
        repin()
        preparation = json.loads(Path(plan['preparation_receipt']).read_text())
        root = Path(plan['root'])
        authenticate_sources(preparation, root, guard)
        sys.path.insert(0, str(runtime))
        spec = importlib.util.spec_from_file_location('overlap_qualification_driver', runtime / 'scripts/lc0_control_train.py')
        if spec is None or spec.loader is None:
            raise RuntimeError('cannot load original driver')
        driver = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = driver
        spec.loader.exec_module(driver)
        from numcodecs.blosc import set_nthreads
        set_nthreads(2)
        import torch
        torch.set_num_threads(2)
        cfg = driver.flatten_run_config_defaults(driver.load_yaml_file(plan['config']))
        model_cfg = driver.model_config_from_flat_config(cfg)
        trainer = make_host_trainer(driver, cfg, model_cfg)
        replay = driver.apply_control_deviations(driver.replay_kwargs_signature(cfg))
        runs = []
        for enabled in (False, True):
            guard()
            buffer = driver.GameAwareEpochBuffer(
                shard_dir=root, batch_size=512, seed=121, input_planes=replay['input_planes'],
                input_history_encoding=model_cfg.input_history_encoding,
                history_rep_fix=bool(model_cfg.history_rep_fix), mirror_augmentation=trainer.mirror_prob > 0,
                plan_workers=2, load_workers=2, max_working_set_bytes=12 * 1024**3,
                objective_mask_counter=trainer.exact_objective_mask_counter, host_batch_overlap=enabled)
            result = measure(buffer, trainer, enabled, guard)
            write_atomic(out / ('ON.json' if enabled else 'OFF.json'), result)
            runs.append(result)
        qualified_batch_count({'runs': runs}, 1963948)
        authenticate_sources(preparation, root, guard)
        repin()
        result = {'status': 'PASS_EXACT_HOST_OVERLAP_CPU_QUALIFICATION', 'runs': runs,
                  'runtime_commit': plan['runtime_commit'], 'root': str(root),
                  'config_sha256': digest(plan['config']),
                  'preparation_sha256': digest(plan['preparation_receipt']),
                  'python_executable': os.path.abspath(sys.executable),
                  'python_version': sys.version,
                  'dependency_distribution_versions': {name: importlib.metadata.version(name)
                    for name in ('numpy', 'torch', 'zarr', 'numcodecs', 'psutil', 'python-chess', 'PyYAML')}}
        write_atomic(out / 'qualification.json', result)
        terminal.update(status='PASS_CPU_QUALIFICATION_NOT_GPU_ADMITTED', qualification_sha256=digest(out / 'qualification.json'))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal['seconds'] = time.monotonic() - started
        write_atomic(out / 'complete.json', terminal)


if __name__ == '__main__':
    main()
