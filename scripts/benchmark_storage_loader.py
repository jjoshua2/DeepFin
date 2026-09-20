"""Bounded CPU-only storage pilot using the production exact-epoch sampler.

Never evicts host caches. Results are cache-affected pilot measurements, not a
cold-disk or GPU-throughput claim. All writes go to fresh explicit artifact roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any


def digest(arrays: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def copy_payload_tree(source: Path, destination: Path) -> None:
    """Copy bytes without POSIX metadata operations unsupported by drvfs."""
    destination.mkdir()
    for item in sorted(source.rglob('*')):
        target = destination / item.relative_to(source)
        if item.is_symlink():
            raise ValueError(f'benchmark source contains a symlink: {item}')
        if item.is_dir():
            target.mkdir()
        elif item.is_file():
            shutil.copyfile(item, target)
        else:
            raise ValueError(f'unsupported source entry: {item}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--external', type=Path, required=True)
    parser.add_argument('--shards', type=int, default=16)
    parser.add_argument('--batch-size', type=int, choices=(128, 256, 512), default=512)
    parser.add_argument('--prepared', action='store_true', help='Read existing complete pilot copies; retain preparation receipt')
    parser.add_argument('--max-seconds', type=int, default=3600)
    args = parser.parse_args()
    if not 1 <= args.shards <= 32 or not 1 <= args.max_seconds <= 7200:
        parser.error('pilot limited to 32 shards and two hours')
    for root in (args.local, args.external):
        if root.exists() != args.prepared:
            parser.error(f'output existence does not match --prepared: {root}')
        if shutil.disk_usage(root.parent).free < 80 * 2**30:
            parser.error(f'insufficient disk reserve: {root.parent}')
    if args.local.resolve() == args.external.resolve():
        parser.error('distinct artifact roots required')
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    os.nice(19)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS'):
        os.environ[key] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(args.runtime.resolve()))
    import numpy as np
    import psutil
    import torch
    from numcodecs.blosc import set_nthreads
    from chess_anti_engine.replay.shard import load_shard_arrays
    from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer

    torch.set_num_threads(2)
    set_nthreads(2)
    if not args.prepared:
        args.local.mkdir()
        args.external.mkdir()
    started = time.monotonic()
    stopped = threading.Event()
    result: dict[str, Any] = {'scope': 'CPU exact-sampler pilot; caches uncontrolled; no GPU measurement',
                              'source': str(args.source), 'runtime': str(args.runtime),
                              'copies': [], 'runs': [], 'complete': False, 'batch_size': args.batch_size}

    result_name = f'results-load-only-v3-batch{args.batch_size}.json' if args.prepared else 'results.json'
    if (args.local / result_name).exists():
        raise ValueError('result already exists')
    if args.prepared:
        previous = json.loads((args.local / 'results.json').read_text())
        if len(previous['copies']) != args.shards or previous['source'] != str(args.source):
            raise ValueError('prepared copy receipt mismatch')
        names = [entry['shard'] for entry in previous['copies']]
        selected = [p.name for p in sorted(args.source.glob('shard_*.zarr'))[:args.shards]]
        if names != selected:
            raise ValueError('prepared source roster changed')
        for root in (args.local, args.external):
            for layout in ('zarr', 'npz'):
                expected = sorted(name if layout == 'zarr' else Path(name).stem + '.npz' for name in names)
                actual = sorted(p.name for p in (root / layout).iterdir())
                if actual != expected:
                    raise ValueError(f'prepared shard roster mismatch: {root / layout}')
        result['copies'] = previous['copies']
        result['reused_prepared_copies'] = True

    def publish() -> None:
        temp = args.local / 'results.tmp'
        temp.write_text(json.dumps(result, indent=2) + '\n')
        temp.replace(args.local / result_name)

    def guard() -> None:
        while not stopped.wait(1):
            try:
                reason = None
                if time.monotonic() - started > args.max_seconds:
                    reason = 'wall limit'
                elif psutil.Process().memory_info().rss > 16 * 2**30:
                    reason = 'RSS limit'
                elif psutil.virtual_memory().available < 32 * 2**30:
                    reason = 'memory reserve'
                elif any(shutil.disk_usage(r).free < 80 * 2**30 for r in (args.local, args.external)):
                    reason = 'disk reserve'
                elif (args.local / 'STOP').exists():
                    reason = 'STOP requested'
            except Exception as exc:
                reason = "resource monitor failed: " + repr(exc)
            if reason:
                try:
                    (args.local / (f'failed-load-only-v3-batch{args.batch_size}.json' if args.prepared else 'failed.json')).write_text(json.dumps({'reason': reason}))
                finally:
                    os._exit(9)

    threading.Thread(target=guard, daemon=True).start()
    try:
        sources = sorted(args.source.glob('shard_*.zarr'))[:args.shards]
        if len(sources) != args.shards:
            raise ValueError('insufficient source shards')
        if not args.prepared:
            for root in (args.local, args.external):
                for layout in ('zarr', 'npz'):
                    (root / layout).mkdir()
            for source in sources:
                t = time.monotonic()
                local = args.local / 'zarr' / source.name
                copy_payload_tree(source, local)
                arrays, meta = load_shard_arrays(local, lazy=False)
                before = digest(arrays)
                packed = args.local / 'npz' / (source.stem + '.npz')
                np.savez_compressed(packed, **arrays, meta_json=np.array(json.dumps(meta, sort_keys=True)))
                restored, _ = load_shard_arrays(packed, lazy=False)
                if digest(restored) != before:
                    raise ValueError('packing changed decoded tensors')
                del arrays, restored
                pack_seconds = time.monotonic() - t
                t = time.monotonic()
                copy_payload_tree(local, args.external / 'zarr' / source.name)
                zarr_copy_seconds = time.monotonic() - t
                t = time.monotonic()
                shutil.copyfile(packed, args.external / 'npz' / packed.name)
                result['copies'].append({'shard': source.name, 'decoded_sha256': before,
                                         'prepare_seconds': pack_seconds,
                                         'zarr_external_copy_seconds': zarr_copy_seconds,
                                         'npz_external_copy_seconds': time.monotonic() - t,
                                         'npz_bytes': packed.stat().st_size,
                                         'zarr_bytes': sum(p.stat().st_size for p in local.rglob('*') if p.is_file())})
                publish()
        # Paired first traversal plus reverse-order repeat; no cold-cache claim.
        variants = [(name, root / layout) for name, root in
                    (('nvme', args.local), ('external', args.external)) for layout in ('zarr',)]
        for trial, ordered in enumerate((variants, list(reversed(variants)))):
            for medium, directory in ordered:
                t = time.monotonic()
                buffer = GameAwareEpochBuffer(shard_dir=directory, batch_size=args.batch_size, seed=121,
                    input_planes=175, input_history_encoding='lc0_root_legacy_meta',
                    history_rep_fix=True, mirror_augmentation=True, plan_workers=1,
                    load_workers=1, max_working_set_bytes=12 * 2**30)
                init_seconds = time.monotonic() - t
                h = hashlib.sha256()
                waits = []
                rows = 0
                digest_seconds = 0.0
                loop_start = time.monotonic()
                try:
                    for _ in range(buffer.num_batches):
                        t = time.monotonic()
                        arrays = buffer.sample_batch_arrays(args.batch_size)
                        waits.append(time.monotonic() - t)
                        rows += len(arrays['x'])
                        digest_start = time.monotonic()
                        h.update(digest(arrays).encode())
                        digest_seconds += time.monotonic() - digest_start
                finally:
                    buffer.close()
                loop_seconds = time.monotonic() - loop_start
                result['runs'].append({'medium': medium, 'layout': directory.name, 'trial': trial,
                    'planning_seconds': init_seconds, 'loader_seconds': sum(waits), 'rows': rows,
                    'consumer_loop_seconds': loop_seconds,
                    'digest_seconds': digest_seconds,
                    'rows_per_consumer_second_including_digest': rows / loop_seconds,
                    'batch_wait_p50': float(np.quantile(waits, .5)),
                    'batch_wait_p95': float(np.quantile(waits, .95)),
                    'batch_wait_max': max(waits), 'sequence_sha256': h.hexdigest()})
                publish()
        # The exact sampler discovers directory Zarr only. Measure NPZ through
        # the actual shard decoder separately; do not claim sampler integration.
        result['packed_decoder_runs'] = []
        for medium, root in (('nvme', args.local), ('external', args.external)):
            begin = time.monotonic()
            load_seconds = 0.0
            rows = 0
            for receipt in result['copies']:
                path = root / 'npz' / (Path(receipt['shard']).stem + '.npz')
                before = time.monotonic()
                arrays, _ = load_shard_arrays(path, lazy=False)
                load_seconds += time.monotonic() - before
                if digest(arrays) != receipt['decoded_sha256']:
                    raise ValueError('packed external decoded tensors differ from source')
                rows += len(arrays['x'])
                del arrays
            result['packed_decoder_runs'].append({'medium': medium, 'rows': rows,
                'decode_seconds': load_seconds, 'wall_seconds_including_digest': time.monotonic() - begin,
                'scope': 'Validated shard decoder only; packed exact-sampler support is not implemented'})
            publish()
        if len({r['sequence_sha256'] for r in result['runs']}) != 1:
            raise ValueError('storage variants produced different batch tensors/order')
        result['complete'] = True
        publish()
    except Exception as exc:
        result['error'] = repr(exc)[:2000]
        publish()
        raise
    finally:
        stopped.set()


if __name__ == '__main__':
    main()
