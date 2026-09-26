"""Opt-in native CPU service curves and offline fixed-package comparisons.

This does not launch search, choose runtime buckets, infer accepted EPS or write
production configuration. Timings cover synchronous model/transport callbacks;
queueing, encoding, batching, backup and process startup are separate costs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

BATCHES = (1, 2, 4, 8, 16)
SCHEMA = 'deepfin.native-service-samples.v1'


def integer(value: Any, lo: int, hi: int) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError('invalid bounded integer')
    return value


def digest(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid SHA256')
    return value


def object_json(text: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('duplicate JSON key')
            obj[key] = value
        return obj

    def invalid(_: str) -> Any:
        raise ValueError('nonfinite JSON number')

    value = json.loads(text, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError('expected JSON object')
    return value


def validate_native(report: dict[str, Any], identity: dict[str, Any],
                    warmups: int, samples: int, offset: int) -> None:
    """No aggregation until the exact planned matrix and physical work reconcile."""
    batch = integer(identity['batch'], 1, 16)
    if batch not in BATCHES:
        raise ValueError('unsupported fixed batch')
    channels = integer(identity['channels'], 146, 175)
    if channels not in (146, 175):
        raise ValueError('unsupported input width')
    integer(warmups, 1, 32)
    integer(samples, 2, 256)
    integer(offset, 0, batch - 1)
    if report.get('schema') != SCHEMA or report.get('status') != 'passed':
        raise ValueError('unsuccessful or unsupported native profile')
    required = {'batch': batch, 'channels': channels, 'output_width': 1861,
                'torch_threads': 2, 'interop_threads': 1, 'warmups_per_size': warmups,
                'samples_per_size': samples, 'order_offset': offset}
    for key, value in required.items():
        if type(report.get(key)) is not int or report[key] != value:
            raise ValueError('native profile configuration mismatch: ' + key)
    for key in ('package_sha256', 'checkpoint_sha256', 'input_sha256', 'reference_sha256'):
        if digest(report.get(key)) != digest(identity[key]):
            raise ValueError('native profile identity mismatch: ' + key)
    if report.get('device') != 'cpu' or report.get('dtype') != 'float32':
        raise ValueError('only CPU-F32 service is qualified')
    for key in ('accepted_neural_rows', 'useful_eps'):
        if key not in report or report[key] is not None:
            raise ValueError('backend execution does not establish accepted search work')
    for key, want in (('atol', 2e-6), ('rtol', 2e-5)):
        if type(report.get(key)) is not float or report[key] != want:
            raise ValueError('numerical tolerance mismatch')
    error = report.get('max_logit_error')
    if error is None or type(error) not in (float, int) or not math.isfinite(error) or error < 0:
        raise ValueError('invalid numerical-error observation')
    integer(report.get('model_open_ns'), 1, (1 << 63) - 1)
    panel = integer(report.get('panel_ns'), 1, (1 << 63) - 1)
    rows = report.get('samples')
    if not isinstance(rows, list) or len(rows) != (warmups + samples) * batch:
        raise ValueError('incomplete service matrix')
    measured_ns = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError('invalid service sample')
        sweep, column = divmod(index, batch)
        real = 1 + (column + sweep + offset) % batch
        if (type(row.get('warmup')) is not bool or row['warmup'] != (sweep < warmups)
                or integer(row.get('sweep'), 0, warmups + samples - 1) != sweep
                or integer(row.get('real_rows'), 1, batch) != real):
            raise ValueError('service sample order/phase mismatch')
        measured_ns += integer(row.get('service_ns'), 1, (1 << 63) - 1)
    if measured_ns > panel:
        raise ValueError('sum of physical callback times exceeds complete panel')


def percentile95(values: list[int]) -> int:
    if not values:
        raise ValueError('no service observations')
    # Nearest-rank sample statistic, not a calibrated deadline bound.
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def cells(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate one compatible target; preserve process-level variation."""
    if not runs:
        raise ValueError('no native runs')
    first = runs[0]['identity']
    samples, warmups = runs[0]['samples_per_size'], runs[0]['warmups_per_size']
    for repeat, run in enumerate(runs):
        if (run['identity'] != first or run['samples_per_size'] != samples
                or run['warmups_per_size'] != warmups):
            raise ValueError('incompatible native run identities')
        validate_native(run['native'], first, warmups, samples, repeat % first['batch'])
    result = []
    for real in range(1, first['batch'] + 1):
        by_process = [[s['service_ns'] for s in run['native']['samples']
                       if not s['warmup'] and s['real_rows'] == real] for run in runs]
        observed = [v for group in by_process for v in group]
        total = sum(observed)
        n = len(observed)
        result.append({'real_rows': real, 'physical_batch': first['batch'],
                       'forward_calls': n, 'executed_real_rows': n * real,
                       'physical_rows': n * first['batch'], 'padded_rows': n * (first['batch'] - real),
                       'service_ns_sum': total, 'median_ns': statistics.median(observed),
                       'p95_ns': percentile95(observed), 'min_ns': min(observed), 'max_ns': max(observed),
                       'process_median_ns': [statistics.median(v) for v in by_process],
                       'executed_rows_per_service_second': n * real * 1e9 / total})
    return result


def wave_plan(targets: list[dict[str, Any]], ready_rows: int) -> dict[str, Any]:
    """Offline equal-work estimate; each alternative uses ONE fixed package.

    Do not mix timings across hosts or estimate unmeasured package-switch costs.
    Sum of sample p95s is a descriptive cost heuristic, not the p95 of the sum.
    No waiting for arrivals, dropping rows or deadline guarantee is implied.
    """
    integer(ready_rows, 1, 64)
    if not targets:
        raise ValueError('no qualified target profiles')
    alternatives = []
    batches: set[int] = set()
    for target in targets:
        batch = integer(target['batch'], 1, 16)
        if batch not in BATCHES or batch in batches:
            raise ValueError('unsupported/duplicate target batch')
        batches.add(batch)
        cost = {}
        for cell in target['cells']:
            real = integer(cell['real_rows'], 1, batch)
            if real in cost or integer(cell['physical_batch'], 1, 16) != batch:
                raise ValueError('duplicate or incompatible cost cell')
            cost[real] = integer(cell['p95_ns'], 1, (1 << 63) - 1)
        if set(cost) != set(range(1, batch + 1)):
            raise ValueError('missing measured occupancy; no interpolation')
        # Dynamic programming permits nonmonotonic service curves, without
        # assuming that either padding less or using a full batch is optimal.
        best: list[tuple[int, list[int]]] = [(0, [])]
        for n in range(1, ready_rows + 1):
            choices = [(best[n - real][0] + cost[real], [*best[n - real][1], real])
                       for real in range(1, min(batch, n) + 1)]
            best.append(min(choices, key=lambda item: (item[0], len(item[1]), item[1])))
        ns, chunks = best[-1]
        alternatives.append({'batch': batch, 'real_rows_per_call': chunks,
                             'forward_calls': len(chunks), 'executed_real_rows': ready_rows,
                             'padded_rows': len(chunks) * batch - ready_rows,
                             'sum_sample_p95_ns': ns})
    alternatives.sort(key=lambda item: (item['sum_sample_p95_ns'], item['forward_calls'], item['batch']))
    return {'ready_rows': ready_rows, 'alternatives': alternatives,
            'lowest_estimated_cost_batch': alternatives[0]['batch'],
            'applied_to_runtime': False, 'deadline_guarantee': False,
            'estimate': 'sum of per-call sample p95; not measured wave latency or a percentile bound'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(package: Path) -> dict[str, str]:
    return {**{k: v for k, v in os.environ.items() if not k.startswith('DEEPFIN_')},
            'DEEPFIN_BEND_MODEL_PACKAGE': str(package.resolve()),
            'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}


def measure(targets: list[tuple[Path, Path]], checkpoint: Path, out: Path,
            warmups: int, samples: int, repeats: int) -> dict[str, Any]:
    import numpy as np
    import torch
    from chess_anti_engine.encoding import rep_fix
    from chess_anti_engine.encoding.cboard_encode import encode_cboard
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.inference import _policy_output
    from native.bend_engine.neural_probe.backend import CHECKPOINT_FORMAT, package_manifest
    from native.bend_engine.neural_probe.checkpoint import load_checkpoint
    from native.bend_engine.standalone.verify_neural import fixtures
    from native.bend_engine.standalone.verify_rules import position

    integer(warmups, 1, 32)
    integer(samples, 20, 256)
    integer(repeats, 2, 8)
    if not 1 <= len(targets) <= 5:
        raise ValueError('supply 1..5 unique prebuilt CPU batch targets')
    torch.set_num_threads(2)
    manifests = [package_manifest(package) for _, package in targets]
    shared_keys = ('checkpoint_sha256', 'weights_key', 'torch_version', 'arch', 'resolved_model_config')
    baseline, encoding = manifests[0]
    if baseline['device'] != 'cpu' or baseline['dtype'] != 'float32':
        raise ValueError('CPU-only measurement; CUDA is not qualified')
    loaded = load_checkpoint(checkpoint, weights_key=str(baseline['weights_key']))
    if loaded.encoding != encoding or loaded.identity['checkpoint_sha256'] != baseline['checkpoint_sha256']:
        raise ValueError('trusted checkpoint/encoding does not match target')
    batches = set()
    for m, e in manifests:
        if (m['format'] != CHECKPOINT_FORMAT or m['device'] != 'cpu' or m['dtype'] != 'float32'
                or e != encoding or m['batch'] in batches
                or any(m[k] != baseline[k] for k in shared_keys)):
            raise ValueError('incompatible target packages or duplicate batch')
        batches.add(m['batch'])
    rep_fix.apply(encoding.history_rep_fix)
    positions = fixtures()
    # Fixed repeated public root tensors, not claimed to represent a search
    # arrival distribution. Repeat the first two fixtures to cover batch sixteen.
    positions = (positions + positions[:2])[:16]
    inputs = np.ascontiguousarray(np.stack([encode_cboard(CBoard.from_board(b),
        input_history_encoding=encoding.input_history_encoding,
        input_extra_features=encoding.input_extra_features) for b in positions]), dtype=np.float32)
    with torch.inference_mode():
        expected = np.concatenate([np.concatenate((_policy_output(y).float().numpy(),
            y['wdl'].float().numpy()), axis=1) for y in
            [loaded.model(torch.from_numpy(inputs[i:i + 1].copy())) for i in range(16)]], axis=0)
    target_records = []
    for (binary, package), (m, _) in zip(targets, manifests, strict=True):
        batch = integer(m['batch'], 1, 16)
        directory = out / f'batch-{batch}'
        directory.mkdir()
        input_path, ref_path = directory / 'inputs.f32', directory / 'reference.f32'
        inputs[:batch].astype('<f4').tofile(input_path)
        expected[:batch].astype('<f4').tofile(ref_path)
        identity = {'batch': batch, 'channels': encoding.channels, 'package_sha256': sha(package),
                    'checkpoint_sha256': baseline['checkpoint_sha256'], 'input_sha256': sha(input_path),
                    'reference_sha256': sha(ref_path), 'binary_sha256': sha(binary)}
        target_records.append({'batch': batch, 'identity': identity, 'runs': []})
    launch_order = []
    for repeat in range(repeats):
        # Counterbalance whole-package order too; do not benchmark all of A then B.
        for j in range(len(targets)):
            i = (j + repeat) % len(targets)
            binary, package = targets[i]
            record = target_records[i]
            batch = record['batch']
            directory = out / f'batch-{batch}'
            command = [str(binary.resolve()), str(directory / 'inputs.f32'), str(directory / 'reference.f32'),
                       str(warmups), str(samples), str(repeat % batch)]
            if sha(binary) != record['identity']['binary_sha256'] or sha(package) != record['identity']['package_sha256']:
                raise ValueError('executable/package changed during measurement')
            start = time.perf_counter_ns()
            child = subprocess.run(command, check=False, env=environment(package), capture_output=True, text=True, timeout=180)
            external_ns = time.perf_counter_ns() - start
            (directory / f'run-{repeat}.stdout').write_text(child.stdout)
            (directory / f'run-{repeat}.stderr').write_text(child.stderr)
            if child.returncode or child.stderr:
                raise ValueError(f'native profile failed: batch {batch}: {child.returncode}: {child.stderr[:500]}')
            native = object_json(child.stdout)
            validate_native(native, record['identity'], warmups, samples, repeat % batch)
            if native['model_open_ns'] + native['panel_ns'] > external_ns:
                raise ValueError('native intervals exceed external process lifetime')
            record['runs'].append({'identity': record['identity'], 'native': native,
                                   'samples_per_size': samples, 'warmups_per_size': warmups,
                                   'process_wall_ns': external_ns})
            launch_order.append({'repeat': repeat, 'batch': batch})
    for record in target_records:
        record['cells'] = cells(record['runs'])
    cpu = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else platform.processor()
    host = {'platform': platform.platform(), 'machine': platform.machine(), 'cpuinfo_sha256': hashlib.sha256(cpu.encode()).hexdigest(),
            'cpu_model': next((line.split(':', 1)[1].strip() for line in cpu.splitlines() if line.startswith('model name')), ''),
            'logical_cpus': os.cpu_count(), 'torch_threads': 2, 'interop_threads': 1,
            'torch_version': str(torch.__version__), 'python_version': platform.python_version()}
    return {'schema': 'deepfin.native-service-profile.v1', 'status': 'passed',
            'recorded_at': datetime.now(timezone.utc).isoformat(), 'host': host,
            'checkpoint_sha256': baseline['checkpoint_sha256'], 'weights_key': baseline['weights_key'],
            'encoding': asdict(encoding), 'parameter_count': loaded.identity['parameter_count'],
            'input_positions': [position(b) for b in positions], 'launch_order': launch_order,
            'targets': target_records,
            'offline_fixed_package_plans': [wave_plan(target_records, n) for n in (1, 2, 4, 8, 16)],
            'accepted_neural_rows': None, 'useful_eps': None, 'runtime_dispatch_changed': False,
            'gpu_qualified': False, 'strength_qualified': False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', nargs=2, type=Path, action='append', required=True, metavar=('BINARY', 'PACKAGE'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--warmups', type=int, default=4)
    p.add_argument('--samples', type=int, default=24)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        report = measure(args.target, args.checkpoint, args.out, args.warmups, args.samples, args.repeats)
    except Exception as error:
        (args.out / 'report.json').write_text(json.dumps({'status': 'failed',
            'runtime_dispatch_changed': False, 'error': f'{type(error).__name__}: {error}'}, indent=2) + '\n')
        raise
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'passed', 'report': str(args.out / 'report.json')}))


if __name__ == '__main__':
    main()
