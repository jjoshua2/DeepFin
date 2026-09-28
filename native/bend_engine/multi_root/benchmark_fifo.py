"""Paired full-runner callback screen; never infer GPU throughput from these timings."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

from .verify import environment, parse
from .verify_fifo import compare, semantic_view

PAIRS = 6
BATCH = 4
SIMS = 256
DEPTH = 8
MAX_REPEATS = 8
TARGET_SECONDS = 0.2
OPENINGS = ('startpos', 'startpos moves e2e4 e7e5',
            'startpos moves d2d4 d7d5', 'startpos moves g1f3 g8f6 b1c3 b8c6')
TERMINAL = 'fen k7/1Q6/2K5/8/8/8/8/8 b - - 150 1'
CASES = {
    'single': OPENINGS[:1],
    'openings-16': tuple(OPENINGS[i % 4] for i in range(16)),
    'mixed-16': tuple(TERMINAL if i % 4 == 0 else OPENINGS[i % 4] for i in range(16)),
}
ARMS = ('list', 'fifo')


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact(result: dict[str, Any]) -> dict[str, Any]:
    view = semantic_view(result)
    return {k: v for k, v in view.items() if k not in ('events', 'nodes')}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require a complete balanced, equal-work panel before reporting any ratios."""
    keys = [(r['case'], r['async'], r['pair'], r['arm']) for r in rows]
    wanted = {(case, mode, pair, arm) for case in CASES for mode in (False, True)
              for pair in range(PAIRS) for arm in ARMS}
    if len(keys) != len(wanted) or set(keys) != wanted:
        raise ValueError('incomplete or duplicate benchmark panel')
    result = {}
    for case in CASES:
        for mode in (False, True):
            panel = [r for r in rows if (r['case'], r['async']) == (case, mode)]
            if len({r['repeats'] for r in panel}) != 1 or len({r['work_sha256'] for r in panel}) != 1:
                raise ValueError('unequal work in benchmark panel')
            reps = panel[0]['repeats']
            if type(reps) is not int or not 1 <= reps <= MAX_REPEATS:
                raise ValueError('invalid benchmark repetition count')
            for row in panel:
                if row['order'] != (row['pair'] + ARMS.index(row['arm'])) % 2:
                    raise ValueError('unbalanced benchmark order')
                for field in ('process_seconds', 'coordinator_seconds'):
                    v = row[field]
                    if type(v) not in (float, int) or not math.isfinite(v) or v < 0:
                        raise ValueError('invalid benchmark time')
                if row['process_seconds'] <= 0:
                    raise ValueError('invalid process duration')
                if row['coordinator_seconds'] > row['process_seconds'] + reps * 0.002:
                    raise ValueError('coordinator duration exceeds process duration')
            item = {'repeats': reps, 'work_sha256': panel[0]['work_sha256']}
            for field in ('coordinator_seconds', 'process_seconds'):
                # Aggregating short runs does not remove their millisecond quantization.
                floor = max(TARGET_SECONDS, reps * 0.05) if field == 'coordinator_seconds' else TARGET_SECONDS
                reliable = all(r[field] >= floor for r in panel)
                ratios = []
                for pair in range(PAIRS):
                    values = {r['arm']: r[field] for r in panel if r['pair'] == pair}
                    ratios.append(values['list'] / values['fifo'] if values['fifo'] else None)
                if not reliable:
                    decision = 'below_measurement_floor'
                elif all(ratio is not None and ratio > 1.05 for ratio in ratios):
                    decision = 'consistent_over_5pct_improvement'
                elif all(ratio is not None and ratio < 1 / 1.05 for ratio in ratios):
                    decision = 'consistent_over_5pct_regression'
                else:
                    decision = 'inconclusive_at_5pct'
                item[field] = {'reliable': reliable, 'decision': decision,
                               'median_list_over_fifo': statistics.median(ratios) if reliable else None,
                               'paired_ratios': ratios if reliable else None,
                               'median_list': statistics.median(r[field] for r in panel if r['arm'] == 'list'),
                               'median_fifo': statistics.median(r[field] for r in panel if r['arm'] == 'fifo')}
            result[f'{case}/{"async" if mode else "sync"}'] = item
    return result


def execute(binary: Path, positions: tuple[str, ...], asynchronous: bool,
            diagnostics: bool) -> tuple[dict[str, Any], float, str]:
    env = {**environment(), 'DEEPFIN_COHORT_ASYNC': str(int(asynchronous))}
    cmd = [str(binary.resolve()), '--threads', '1', '--', str(SIMS), str(DEPTH), '0',
           str(int(diagnostics)), *positions]
    start = time.perf_counter_ns()
    run = subprocess.run(cmd, env=env, input='', capture_output=True, text=True, timeout=60, check=False)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    if run.returncode or run.stderr:
        raise RuntimeError(f'benchmark executable failed: {run.returncode}: {run.stderr}')
    parsed = parse(run.stdout, len(positions), BATCH, SIMS, 0, diagnostics, asynchronous=asynchronous)
    return parsed, elapsed, hashlib.sha256(run.stdout.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('report already exists; preserve prior observations')
    binaries = {'list': args.reference, 'fifo': args.candidate}
    hashes = {arm: hashlib.sha256(p.read_bytes()).hexdigest() for arm, p in binaries.items()}
    if hashes['list'] == hashes['fifo']:
        parser.error('reference and candidate binaries must differ')
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {'status': 'failed', 'scope': 'full coordinator with deterministic callback; no real model/GPU',
                              'platform': platform.platform(), 'python': platform.python_version(),
                              'binaries': hashes, 'batch': BATCH, 'channels': 146, 'sims': SIMS, 'depth': DEPTH,
                              'pairs': PAIRS, 'max_repeats': MAX_REPEATS, 'target_seconds': TARGET_SECONDS,
                              'diagnostic_checks': [], 'calibration': [], 'samples': rows}
    deadline = time.monotonic() + 240
    try:
        for case, positions in CASES.items():
            for asynchronous in (False, True):
                if time.monotonic() > deadline:
                    raise TimeoutError('benchmark total execution budget exhausted')
                reference, _, _ = execute(args.reference, positions, asynchronous, True)
                candidate, _, _ = execute(args.candidate, positions, asynchronous, True)
                full_hash = compare(reference, candidate)
                expected = compact(reference)
                work_hash = digest(expected)
                report['diagnostic_checks'].append({'case': case, 'async': asynchronous,
                    'complete_tree_and_event_sha256': full_hash, 'work_sha256': work_hash,
                    'accepted_neural_rows': reference['work']['accepted_neural_rows'],
                    'completed_simulations': reference['work']['completed_simulations'],
                    'forward_calls': reference['work']['forward_calls']})
                del reference, candidate
                warmup = []
                for arm in ARMS:
                    parsed, wall, _ = execute(binaries[arm], positions, asynchronous, False)
                    if compact(parsed) != expected:
                        raise ValueError('diagnostics-off execution changed semantic work')
                    warmup.append(wall)
                    report['calibration'].append({'case': case, 'async': asynchronous, 'arm': arm,
                        'process_seconds': wall, 'coordinator_seconds': parsed['work']['wall_seconds']})
                repeats = min(MAX_REPEATS, max(1, math.ceil(2 * TARGET_SECONDS / min(warmup))))
                for pair in range(PAIRS):
                    order = ARMS if pair % 2 == 0 else ARMS[::-1]
                    for ordinal, arm in enumerate(order):
                        observations = []
                        for _ in range(repeats):
                            if time.monotonic() > deadline:
                                raise TimeoutError('benchmark total execution budget exhausted')
                            parsed, wall, output_hash = execute(binaries[arm], positions, asynchronous, False)
                            if compact(parsed) != expected:
                                raise ValueError('timed execution changed semantic work')
                            observations.append({'process_seconds': wall, 'coordinator_seconds': parsed['work']['wall_seconds'],
                                                 'stdout_sha256': output_hash})
                        rows.append({'case': case, 'async': asynchronous, 'pair': pair, 'order': ordinal,
                            'arm': arm, 'repeats': repeats, 'work_sha256': work_hash,
                            'process_seconds': sum(r['process_seconds'] for r in observations),
                            'coordinator_seconds': sum(r['coordinator_seconds'] for r in observations),
                            'observations': observations})
                print(f'Completed {case} async={asynchronous} repeats={repeats}', flush=True)
        report['summary'] = summarize(rows)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
