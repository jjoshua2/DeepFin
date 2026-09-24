"""Opt-in equal-work arena footprint screen; per-child peak RSS, not live memory advice."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import time
from typing import Any

from . import verify
from .verify_fifo import semantic_view

CAPACITIES = (4096, 4097, 8192, 16384, 65536)
ROOT_COUNTS = (1, 16)
SIMULATIONS = (1, 64)
REPEATS = 5
BATCH = 4
CHANNELS = 146
DEPTH = 8
ENV = 'DEEPFIN_COHORT_ARENA_NODES'
OPENINGS = ('startpos', 'startpos moves e2e4 e7e5', 'startpos moves d2d4 d7d5',
            'startpos moves g1f3 g8f6 b1c3 b8c6')


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def physical_capacity(capacity: int) -> int:
    verify.integer(capacity, 1, 65536)
    return max(4096, 1 << (capacity - 1).bit_length())


def rss_result(text: str) -> int:
    """GNU time reports THIS child's peak resident memory in KiB and exit status."""
    fields = text.split()
    if len(fields) != 2 or any(not v.isascii() or not v.isdecimal() for v in fields):
        raise ValueError('invalid per-child RSS record')
    rss, code = map(int, fields)
    if not 0 < rss <= (1 << 40) or code != 0:
        raise ValueError('failed child or invalid peak RSS')
    return rss


def checked_work(parsed: dict[str, Any], roots: int, sims: int, *, diagnostics: bool) -> dict[str, Any]:
    """Capacity exhaustion is not a faster or smaller equal-work observation."""
    if set(parsed['roots']) != set(range(1, roots + 1)):
        raise ValueError('arena screen root identities differ')
    if any(r['completed_simulations'] != sims or r['stop_code'] != 0 for r in parsed['roots'].values()):
        raise ValueError('arena screen did not complete equal requested work')
    if parsed['work']['completed_simulations'] != roots * sims:
        raise ValueError('arena screen completed-work total differs')
    view = semantic_view(parsed)
    return view if diagnostics else {k: v for k, v in view.items() if k not in ('nodes', 'events')}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {(roots, sims, cap, sample) for roots in ROOT_COUNTS for sims in SIMULATIONS
              for cap in CAPACITIES for sample in range(REPEATS)}
    keys = [(r['roots'], r['simulations'], r['capacity'], r['sample']) for r in rows]
    if len(keys) != len(wanted) or set(keys) != wanted:
        raise ValueError('incomplete or duplicate arena measurement panel')
    for row in rows:
        for name in ('roots', 'simulations', 'capacity', 'sample', 'order', 'physical_capacity', 'peak_rss_kib'):
            if type(row[name]) is not int:
                raise ValueError('invalid arena measurement identity or RSS type')
        if row['physical_capacity'] != physical_capacity(row['capacity']):
            raise ValueError('arena physical allocation differs')
        if row['order'] != (CAPACITIES.index(row['capacity']) - row['sample']) % len(CAPACITIES):
            raise ValueError('unbalanced arena measurement order')
        if not 0 < row['peak_rss_kib'] <= (1 << 40):
            raise ValueError('invalid arena peak RSS')
        for name in ('process_seconds', 'coordinator_seconds'):
            v = row[name]
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError('invalid arena duration')
        if not 0 < row['process_seconds'] <= 61 or row['coordinator_seconds'] > row['process_seconds'] + 0.002:
            raise ValueError('invalid arena timer relationship')
        for name in ('work_sha256', 'stdout_sha256'):
            v = row[name]
            if not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
                raise ValueError('invalid arena observation digest')
    results = []
    for roots in ROOT_COUNTS:
        for sims in SIMULATIONS:
            matched = [r for r in rows if (r['roots'], r['simulations']) == (roots, sims)]
            if len({r['work_sha256'] for r in matched}) != 1:
                raise ValueError('unequal realized work across arena capacities')
            for capacity in CAPACITIES:
                selected = [r for r in matched if r['capacity'] == capacity]
                rss = [r['peak_rss_kib'] for r in selected]
                results.append({'roots': roots, 'simulations': sims, 'capacity': capacity,
                                'physical_capacity': physical_capacity(capacity), 'samples': len(selected),
                                'peak_rss_kib_median': statistics.median(rss),
                                'peak_rss_kib_min': min(rss), 'peak_rss_kib_max': max(rss),
                                'process_seconds_median': statistics.median(r['process_seconds'] for r in selected),
                                'coordinator_seconds_median': statistics.median(r['coordinator_seconds'] for r in selected),
                                'work_sha256': selected[0]['work_sha256']})
    return results


def execute(binary: Path, timer: Path, folder: Path, roots: int, sims: int, capacity: int,
            *, diagnostics: bool, trace: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    folder.mkdir()  # Every child has a fresh RSS file; never reuse a prior maximum.
    env = {**verify.environment(), ENV: str(capacity), 'DEEPFIN_COHORT_ASYNC': '0', 'LC_ALL': 'C'}
    if trace is not None:
        env['DEEPFIN_BEND_MODEL_TRACE'] = str(trace.resolve())
    args = [str(timer.resolve()), '-f', '%M %x', '-o', str((folder / 'rss.txt').resolve()), '--',
            str(binary.resolve()), '--threads', '1', '--', str(sims), str(DEPTH), '0', str(int(diagnostics)),
            *(OPENINGS[i % len(OPENINGS)] for i in range(roots))]
    started = time.perf_counter_ns()
    with subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as child:
        try:
            stdout, stderr = child.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            stdout, stderr = child.communicate()
            (folder / 'stdout.txt').write_text(stdout)
            (folder / 'stderr.txt').write_text(stderr)
            raise
        elapsed = (time.perf_counter_ns() - started) / 1e9
    (folder / 'stdout.txt').write_text(stdout)
    (folder / 'stderr.txt').write_text(stderr)
    if child.returncode != 0 or stderr:
        raise RuntimeError(f'arena measurement child failed: {child.returncode}: {stderr}')
    rss = rss_result((folder / 'rss.txt').read_text())
    parsed = verify.parse(stdout, roots, BATCH, sims, 0, diagnostics, arena_nodes=capacity)
    work = checked_work(parsed, roots, sims, diagnostics=False)
    return parsed, {'peak_rss_kib': rss, 'process_seconds': elapsed,
                    'coordinator_seconds': parsed['work']['wall_seconds'],
                    'stdout_sha256': hashlib.sha256(stdout.encode()).hexdigest(), 'work_sha256': digest(work)}


def main() -> None:
    import chess
    import torch
    from chess_anti_engine.encoding import rep_fix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--time', type=Path, default=Path('/usr/bin/time'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('this peak-RSS screen is qualified on Linux only')
    output = args.output.resolve()
    if output.exists():
        parser.error('output exists; preserve old evidence and choose a fresh directory')
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {'status': 'failed', 'scope': 'equal-work callback runner peak RSS and whole-process duration',
                              'platform': platform.platform(), 'samples': rows, 'warmups': [], 'diagnostics': [],
                              'batch': BATCH, 'channels': CHANNELS, 'asynchronous': False,
                              'capacities': CAPACITIES, 'root_counts': ROOT_COUNTS, 'simulations': SIMULATIONS,
                              'repeats': REPEATS, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        torch.set_num_threads(2)
        rep_fix.apply(True)
        report['reference_history_rep_fix'] = rep_fix.current()
        report['binary_sha256'] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
        report['oracle_sha256'] = hashlib.sha256(args.oracle.read_bytes()).hexdigest()
        timer_version = subprocess.run([str(args.time), '--version'], capture_output=True, text=True, check=True, timeout=10)
        if timer_version.stderr or 'GNU' not in timer_version.stdout:
            raise ValueError('GNU time is required for KiB RSS units')
        report['time_version'] = timer_version.stdout
        starts = []
        for position in OPENINGS:
            b = chess.Board()
            for move in position.partition(' moves ')[2].split():
                b.push_uci(move)
            starts.append(b)
        deadline = time.monotonic() + 240
        for roots in ROOT_COUNTS:
            for sims in SIMULATIONS:
                boards = [starts[i % len(starts)] for i in range(roots)]
                expected = None
                for capacity in CAPACITIES:
                    if time.monotonic() > deadline:
                        raise TimeoutError('arena screen budget exhausted')
                    case = f'r{roots}-s{sims}-c{capacity}'
                    trace = output / (case + '.trace') if capacity == 4096 else None
                    parsed, _ = execute(args.binary, args.time, output / ('diagnostic-' + case), roots, sims,
                                        capacity, diagnostics=True, trace=trace)
                    view = checked_work(parsed, roots, sims, diagnostics=True)
                    independent = None
                    if capacity == 4096:
                        assert trace is not None
                        independent = verify.oracle_check(parsed, boards, trace, args.oracle, CHANNELS,
                                                          BATCH, sims, DEPTH, 0, arena_nodes=capacity)
                        trace.unlink()
                        expected = view
                    elif view != expected:
                        raise ValueError('capacity changed complete trees, events or non-time work')
                    report['diagnostics'].append({'roots': roots, 'simulations': sims, 'capacity': capacity,
                                                  'semantic_sha256': digest(view),
                                                  'independent_oracle': independent if capacity == 4096 else None})
                    plain, observation = execute(args.binary, args.time, output / ('warmup-' + case), roots, sims,
                                                 capacity, diagnostics=False)
                    if checked_work(plain, roots, sims, diagnostics=False) != {k: v for k, v in view.items() if k not in ('nodes', 'events')}:
                        raise ValueError('quiet arena execution changed work')
                    report['warmups'].append({'roots': roots, 'simulations': sims, 'capacity': capacity, **observation})
                assert expected is not None
                expected_work = digest({k: v for k, v in expected.items() if k not in ('nodes', 'events')})
                for sample in range(REPEATS):
                    order = CAPACITIES[sample:] + CAPACITIES[:sample]
                    for ordinal, capacity in enumerate(order):
                        if time.monotonic() > deadline:
                            raise TimeoutError('arena screen budget exhausted')
                        _, observation = execute(args.binary, args.time,
                            output / f'measure-r{roots}-s{sims}-p{sample}-c{capacity}', roots, sims, capacity, diagnostics=False)
                        if observation['work_sha256'] != expected_work:
                            raise ValueError('measured arena execution changed work')
                        rows.append({'roots': roots, 'simulations': sims, 'capacity': capacity,
                                     'physical_capacity': physical_capacity(capacity), 'sample': sample,
                                     'order': ordinal, **observation})
                print(f'Completed roots={roots}, simulations={sims}', flush=True)
        report['summary'] = summarize(rows)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
