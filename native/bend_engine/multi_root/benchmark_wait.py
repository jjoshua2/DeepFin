"""Matched sleep/notification runner timings, with semantic and child-CPU checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time
from typing import Any

from .verify import environment, parse
from .verify_fifo import compare, semantic_view

ARMS = ('sleep', 'notify')
PAIRS = 6
MAX_REPEATS = 16
MIN_SECONDS = 0.2
BATCH = 4
SIMS = 256
DEPTH = 8
OPENINGS = ('startpos', 'startpos moves e2e4 e7e5', 'startpos moves d2d4 d7d5',
            'startpos moves g1f3 g8f6 b1c3 b8c6')
TERMINAL = 'fen k7/1Q6/2K5/8/8/8/8/8 b - - 150 1'
CASES = {'single': OPENINGS[:1],
         'openings-16': tuple(OPENINGS[i % 4] for i in range(16)),
         'mixed-16': tuple(TERMINAL if i % 4 == 0 else OPENINGS[i % 4] for i in range(16))}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in semantic_view(result).items() if k not in ('nodes', 'events')}


def execute(binary: Path, positions: tuple[str, ...], asynchronous: bool,
            diagnostics: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    env = {**environment(), 'DEEPFIN_COHORT_ASYNC': str(int(asynchronous))}
    cmd = [str(binary.resolve()), '--threads', '1', '--', str(SIMS), str(DEPTH), '0',
           str(int(diagnostics)), *positions]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter_ns()
    run = subprocess.run(cmd, env=env, input='', capture_output=True, text=True, timeout=60, check=False)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if run.returncode or run.stderr:
        raise RuntimeError(f'completion-wait executable failed: {run.returncode}: {run.stderr}')
    result = parse(run.stdout, len(positions), BATCH, SIMS, 0, diagnostics, asynchronous=asynchronous)
    return result, {'wall_seconds': elapsed,
                    'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                    'coordinator_seconds': result['work']['wall_seconds'],
                    'stdout_sha256': hashlib.sha256(run.stdout.encode()).hexdigest()}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wanted = {(case, mode, pair, arm) for case in CASES for mode in (False, True)
              for pair in range(PAIRS) for arm in ARMS}
    keys = [(r['case'], r['async'], r['pair'], r['arm']) for r in rows]
    if len(keys) != len(wanted) or set(keys) != wanted:
        raise ValueError('incomplete or duplicate wait panel')
    for row in rows:
        if type(row['async']) is not bool or type(row['pair']) is not int:
            raise ValueError('invalid wait panel identity')
        if row['order'] != (row['pair'] + ARMS.index(row['arm'])) % 2:
            raise ValueError('unbalanced wait panel order')
        reps = row['repeats']
        if type(reps) is not int or not 1 <= reps <= MAX_REPEATS or len(row['observations']) != reps:
            raise ValueError('invalid wait repetition count')
        for obs in row['observations']:
            for field in ('wall_seconds', 'cpu_seconds', 'coordinator_seconds'):
                value = obs[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('invalid wait sample duration')
            if not 0 < obs['wall_seconds'] <= 61:
                raise ValueError('invalid wait sample wall duration')
            if obs['coordinator_seconds'] > obs['wall_seconds'] + 0.002:
                raise ValueError('internal wait timer exceeds child wall time')
        for field in ('wall_seconds', 'cpu_seconds', 'coordinator_seconds'):
            # Recompute from raw observations; do not trust independently edited totals.
            total = sum(obs[field] for obs in row['observations'])
            if not math.isclose(total, row[field], rel_tol=0, abs_tol=1e-9):
                raise ValueError('wait aggregate differs from raw observations')
    result = {}
    for case in CASES:
        for mode in (False, True):
            panel = [r for r in rows if (r['case'], r['async']) == (case, mode)]
            if len({r['repeats'] for r in panel}) != 1 or len({r['work_sha256'] for r in panel}) != 1:
                raise ValueError('unequal work in wait panel')
            reliable = all(r['wall_seconds'] >= MIN_SECONDS for r in panel)
            ratios = []
            for pair in range(PAIRS):
                arms = {r['arm']: r for r in panel if r['pair'] == pair}
                ratios.append(arms['sleep']['wall_seconds'] / arms['notify']['wall_seconds'])
            decision = 'below_measurement_floor'
            if reliable:
                if all(v > 1.05 for v in ratios):
                    decision = 'consistent_over_5pct_improvement'
                elif all(v < 1 / 1.05 for v in ratios):
                    decision = 'consistent_over_5pct_regression'
                else:
                    decision = 'inconclusive_at_5pct'
            reps = panel[0]['repeats']
            medians = {arm: {field: statistics.median(r[field] / reps for r in panel if r['arm'] == arm)
                             for field in ('wall_seconds', 'cpu_seconds', 'coordinator_seconds')} for arm in ARMS}
            cpu = medians['sleep']['cpu_seconds']
            result[f'{case}/{"async" if mode else "sync"}'] = {
                'repeats': reps, 'work_sha256': panel[0]['work_sha256'], 'reliable': reliable,
                'decision': decision, 'median_sleep_over_notify': statistics.median(ratios) if reliable else None,
                'paired_ratios': ratios if reliable else None, 'medians_per_invocation': medians,
                'notify_over_sleep_cpu': medians['notify']['cpu_seconds'] / cpu if cpu > 0 else None,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('report already exists; preserve prior observations')
    binaries = {'sleep': args.reference, 'notify': args.candidate}
    hashes = {arm: hashlib.sha256(path.read_bytes()).hexdigest() for arm, path in binaries.items()}
    if hashes['sleep'] == hashes['notify']:
        parser.error('reference and candidate binaries must differ')
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {'status': 'failed', 'scope': 'full callback coordinator, no model/GPU',
                              'platform': platform.platform(), 'binaries': hashes, 'batch': BATCH,
                              'channels': 146, 'sims': SIMS, 'depth': DEPTH, 'samples': rows,
                              'diagnostics': [], 'warmups': [], 'minimum_group_seconds': MIN_SECONDS,
                              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    deadline = time.monotonic() + 240
    try:
        for case, positions in CASES.items():
            for mode in (False, True):
                left, _ = execute(args.reference, positions, mode, True)
                right, _ = execute(args.candidate, positions, mode, True)
                full_hash = compare(left, right)
                expected = compact(left)
                work_hash = digest(expected)
                report['diagnostics'].append({'case': case, 'async': mode, 'semantic_sha256': full_hash,
                    'work_sha256': work_hash, 'work': expected['work']})
                del left, right
                warmups = []
                for arm in ARMS:
                    parsed, observation = execute(binaries[arm], positions, mode, False)
                    if compact(parsed) != expected:
                        raise ValueError('warmup changed semantic work')
                    warmups.append(observation['wall_seconds'])
                    report['warmups'].append({'case': case, 'async': mode, 'arm': arm, **observation})
                repeats = min(MAX_REPEATS, max(1, math.ceil(2 * MIN_SECONDS / min(warmups))))
                for pair in range(PAIRS):
                    order = ARMS if pair % 2 == 0 else ARMS[::-1]
                    for ordinal, arm in enumerate(order):
                        observations = []
                        for _ in range(repeats):
                            if time.monotonic() > deadline:
                                raise TimeoutError('completion-wait panel budget exhausted')
                            parsed, observation = execute(binaries[arm], positions, mode, False)
                            if compact(parsed) != expected:
                                raise ValueError('timed wait execution changed semantic work')
                            observations.append(observation)
                        rows.append({'case': case, 'async': mode, 'pair': pair, 'order': ordinal,
                            'arm': arm, 'repeats': repeats, 'work_sha256': work_hash,
                            'observations': observations,
                            **{field: sum(o[field] for o in observations)
                               for field in ('wall_seconds', 'cpu_seconds', 'coordinator_seconds')}})
                print(f'Completed {case} async={mode} repeats={repeats}', flush=True)
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
