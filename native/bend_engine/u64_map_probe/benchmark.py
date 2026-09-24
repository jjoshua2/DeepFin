"""Matched numeric-map operation screen; no cache/search integration or speed gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time
from typing import Any

HERE = Path(__file__).resolve().parent
MASK = (1 << 32) - 1
MULT = 1664525
ARMS = ('hash', 'scan')
PAIRS = 6
MIN_MS = 50
MAX_ROUNDS = 16384


@dataclass(frozen=True)
class Workload:
    name: str
    bits: int
    initial: tuple[tuple[int, int], ...]
    ops: tuple[tuple[int, int, int], ...]  # get=0, put=1, remove=2


def bucket(key: int, mask: int) -> int:
    # Fixture generation only, not the dictionary value oracle.
    fold = (key & MASK) ^ (((key >> 32) * 2654435761) & MASK)
    mixed = ((fold ^ (fold >> 16)) * 2246822519) & MASK
    return (mixed ^ (mixed >> 13)) & mask


def workloads() -> list[Workload]:
    result = []
    for bits, distribution in [(4, 'random'), (7, 'random'), (9, 'random'),
                                (7, 'high-half'), (7, 'cluster')]:
        rng = random.Random(20260924 + bits)
        count = 1 << (bits - 1)
        keys: list[int] = []
        while len(keys) < 2 * count:
            k = rng.getrandbits(64) if distribution == 'random' else rng.getrandbits(32) << 32
            if k in keys:
                continue
            if distribution == 'cluster' and bucket(k, (1 << bits) - 1) != (1 << bits) - 2:
                continue
            keys.append(k)
        initial = tuple((k, rng.getrandbits(32)) for k in keys[:count])
        for kind in ('hits', 'misses', 'churn'):
            ops = []
            if kind in ('hits', 'misses'):
                domain = keys[:count] if kind == 'hits' else keys[count:]
                ops = [(0, rng.choice(domain), 0) for _ in range(256)]
            else:
                for _ in range(32):
                    k, old = rng.choice(initial)
                    changed = old ^ MASK
                    # Includes full rejection, replacement, delete/miss, reinsertion.
                    ops.extend([(1, rng.choice(keys[count:]), 0), (1, k, changed),
                                (0, k, 0), (1, k, old), (2, k, 0), (0, k, 0),
                                (1, k, old), (0, k, 0)])
            result.append(Workload(f'{distribution}-{count}-{kind}', bits, initial, tuple(ops)))
    return result


def model(case: Workload) -> tuple[list[int], dict[int, int]]:
    table = dict(case.initial)
    if len(table) != len(case.initial) or len(table) > 1 << (case.bits - 1):
        raise ValueError('invalid initial dictionary')
    tokens = []
    for kind, key, value in case.ops:
        if kind == 1:
            if key in table:
                tag, previous = 2, table[key]
                table[key] = value
            elif len(table) == 1 << (case.bits - 1):
                tag, previous = 3, 0
            else:
                tag, previous = 1, 0
                table[key] = value
        elif kind in (0, 2):
            tag, previous = ((4 if kind == 0 else 6), table[key]) if key in table else ((5 if kind == 0 else 7), 0)
            if kind == 2:
                table.pop(key, None)
        else:
            raise ValueError('invalid operation')
        tokens.append((7 * previous + tag) & MASK)
    return tokens, table


def checksum(tokens: list[int], rounds: int) -> int:
    """Repeat one state-preserving cycle using an affine fold, independently of the map."""
    a, b = 1, 0
    for token in tokens:
        a, b = a * MULT & MASK, (b * MULT + token) & MASK
    total = 0
    while rounds:
        if rounds & 1:
            total = (a * total + b) & MASK
        a, b = a * a & MASK, b * (a + 1) & MASK
        rounds >>= 1
    return total


def encode(case: Workload, rounds: int) -> str:
    if type(rounds) is not int or not 0 <= rounds <= MAX_ROUNDS:
        raise ValueError('invalid repeat count')
    def words(kind: int, key: int, value: int) -> str:
        if type(key) is not int or not 0 <= key < 1 << 64 or not 0 <= value <= MASK or kind not in (0, 1, 2):
            raise ValueError('invalid operation words')
        return f'{kind} {key >> 32} {key & MASK} {value}'
    initial = ';'.join(words(1, k, v) for k, v in case.initial)
    ops = ';'.join(words(*op) for op in case.ops)
    encoded = f'{case.bits}|{rounds}|{initial}|{ops}'
    if not initial or not ops or len(encoded) > 100000:
        raise ValueError('invalid benchmark transport size')
    return encoded


def parse(text: str, case: Workload, rounds: int) -> int:
    tokens, expected = model(case)
    if expected != dict(case.initial):
        raise ValueError('cycle must preserve dictionary state')
    lines = text.splitlines()
    if len(lines) != len(expected) + 2 or lines[-1] != 'end':
        raise ValueError('incomplete or extra map output')
    def fields(line: str, label: str, count: int) -> list[int]:
        parts = line.split()
        if len(parts) != count + 1 or parts[0] != label or any(not v.isascii() or not v.isdecimal() for v in parts[1:]):
            raise ValueError('malformed map output')
        return list(map(int, parts[1:]))
    r, ms, checked, length = fields(lines[0], 'sample', 4)
    if (r, checked, length) != (rounds, checksum(tokens, rounds), len(expected)) or ms > 60000:
        raise ValueError('map identity, checksum, size or timer mismatch')
    actual: dict[int, int] = {}
    for line in lines[1:-1]:
        high, low, value = fields(line, 'entry', 3)
        if max(high, low, value) > MASK:
            raise ValueError('out-of-range map entry')
        key = high << 32 | low
        if key in actual:
            raise ValueError('duplicate map entry')
        actual[key] = value
    if actual != expected:
        raise ValueError('final map differs from dictionary oracle')
    return ms


def summarize(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    selected = [r for r in rows if r['phase'] == 'measurement']
    wanted = {(name, pair, arm) for name in names for pair in range(PAIRS) for arm in ARMS}
    keys = [(r['case'], r['pair'], r['arm']) for r in selected]
    if len(keys) != len(wanted) or set(keys) != wanted:
        raise ValueError('incomplete or duplicate measurement panel')
    result = {}
    for name in names:
        panel = [r for r in selected if r['case'] == name]
        if len({r['rounds'] for r in panel}) != 1 or len({r['input_sha256'] for r in panel}) != 1:
            raise ValueError('unequal benchmark work')
        for row in panel:
            if type(row['pair']) is not int or type(row['rounds']) is not int or not 1 <= row['rounds'] <= MAX_ROUNDS:
                raise ValueError('invalid measurement identity')
            if row['order'] != (row['pair'] + ARMS.index(row['arm'])) % 2:
                raise ValueError('unbalanced measurement order')
            if type(row['milliseconds']) is not int or not 0 <= row['milliseconds'] <= 60000:
                raise ValueError('invalid measurement duration')
        reliable = min(r['milliseconds'] for r in panel) >= MIN_MS
        ratios = []
        for pair in range(PAIRS):
            values = {r['arm']: r['milliseconds'] for r in panel if r['pair'] == pair}
            ratios.append(values['scan'] / values['hash'] if values['hash'] else 0.0)
        decision = 'below_measurement_floor'
        if reliable:
            decision = ('hash_over_5pct_faster' if all(r > 1.05 for r in ratios) else
                        'scan_over_5pct_faster' if all(r < 1 / 1.05 for r in ratios) else 'inconclusive_at_5pct')
        result[name] = {'rounds': panel[0]['rounds'], 'reliable': reliable, 'decision': decision,
                        'median_ms': {a: statistics.median(r['milliseconds'] for r in panel if r['arm'] == a) for a in ARMS},
                        'median_scan_over_hash': statistics.median(ratios) if reliable else None,
                        'paired_ratios': ratios if reliable else None}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measure', action='store_true')
    args = parser.parse_args()
    if not args.bun or not args.cc or args.output.exists():
        parser.error('Bun, Clang and a fresh output directory are required')
    output = args.output.resolve()
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {'status': 'failed', 'scope': 'hash versus dense scan, shared slots and entry budget',
                             'samples': rows, 'cases': [asdict(c) for c in workloads()], 'measured': args.measure}
    def command(argv: list[str], label: str, env: dict[str, str] | None = None) -> str:
        run = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=60, check=False)
        (output / f'{label}.stdout').write_text(run.stdout)
        (output / f'{label}.stderr').write_text(run.stderr)
        if run.returncode or run.stderr:
            raise RuntimeError(f'{label}: exit {run.returncode}; captured diagnostics retained')
        return run.stdout
    try:
        env = dict(os.environ, BEND_NO_TELEMETRY='1')
        env.pop('BEND_U64_CORRUPT', None)
        report['compiler'] = command([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'), str(args.compiler_root.resolve())], 'compiler', env)
        report['cc'] = command([args.cc, '--version'], 'cc')
        report['source_sha256'] = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                                   for n in ('U64Map.bend', 'ScanMap.bend', 'benchmark.bend', 'benchmark.py')}
        cases = workloads()
        for c in cases:
            if model(c)[1] != dict(c.initial):
                raise ValueError('non-neutral workload')
        binaries = {}
        for arm in ARMS:
            folder = output / arm
            folder.mkdir()
            for name in ('U64Map.bend', 'ScanMap.bend'):
                shutil.copyfile(HERE / name, folder / name)
            source = (HERE / 'benchmark.bend').read_text()
            if arm == 'scan':
                if source.count('import ./U64Map.bend as Impl') != 1:
                    raise ValueError('implementation import changed')
                source = source.replace('import ./U64Map.bend as Impl', 'import ./ScanMap.bend as Impl')
            (folder / 'benchmark.bend').write_text(source)
            generated = folder / 'bench.c'
            command([args.bun, str(args.compiler_root.resolve() / 'bend2/main.ts'), str(folder / 'benchmark.bend'), '-o', str(generated)], 'generate-' + arm, env)
            for mode, flags in [('native', ['-march=native']), ('ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all'])]:
                binary = folder / mode
                command([args.cc, '-std=c11', '-O3', '-ffp-contract=off', *flags, str(generated), '-pthread', '-lm', '-o', str(binary)], f'build-{arm}-{mode}')
                binaries[arm, mode] = binary
        from . import run_probe as reference
        reference_cases = reference.fixtures()
        trace_counts = []
        for arm in ARMS:
            folder = output / arm
            trace_source = (HERE / 'main.bend').read_text()
            trace_source = trace_source.replace('import ./U64Map.bend as M',
                'import ./U64Map.bend as M\nimport ./' + ('U64Map' if arm == 'hash' else 'ScanMap') + '.bend as Impl')
            for operation in ('get', 'put', 'remove'):
                trace_source = trace_source.replace('M.' + operation + '(', 'Impl.' + operation + '(')
            (folder / 'trace.bend').write_text(trace_source)
            generated = folder / 'trace.c'
            command([args.bun, str(args.compiler_root.resolve() / 'bend2/main.ts'), str(folder / 'trace.bend'), '-o', str(generated)], 'trace-generate-' + arm, env)
            for mode, flags in [('native', ['-march=native']), ('ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all'])]:
                binary = folder / ('trace-' + mode)
                command([args.cc, '-std=c11', '-O3', '-ffp-contract=off', *flags, str(generated), '-pthread', '-lm', '-o', str(binary)], 'trace-build-' + arm + '-' + mode)
                for case in reference_cases:
                    text = command([str(binary), '--threads', '1'], 'trace-' + arm + '-' + mode + '-' + case.name,
                                   dict(env, DEEPFIN_U64_MAP_TRACE=reference.encode(case)))
                    reference.verify(text, case)
                trace_counts.append({'arm': arm, 'mode': mode, 'cases': len(reference_cases),
                                     'operations': sum(len(c.ops) for c in reference_cases)})
        report['reference_checks'] = trace_counts

        def sample(case: Workload, arm: str, rounds: int, phase: str, pair: int, order: int, mode: str = 'native') -> int:
            encoded = encode(case, rounds)
            label = f'{len(rows):04}-{case.name}-{arm}-{phase}'
            text = command([str(binaries[arm, mode]), '--threads', '1'], label, dict(env, DEEPFIN_MAP_BENCH=encoded))
            ms = parse(text, case, rounds)
            rows.append({'case': case.name, 'arm': arm, 'rounds': rounds, 'phase': phase, 'pair': pair,
                         'order': order, 'mode': mode, 'milliseconds': ms,
                         'input_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
                         'stdout_sha256': hashlib.sha256(text.encode()).hexdigest()})
            return ms
        for mode in ('native', 'ubsan'):
            for case in cases:
                for arm in ARMS:
                    for rounds in (0, 1, 3):
                        sample(case, arm, rounds, 'contract', 0, 0, mode)
        report['contract_executions'] = len(rows)
        if args.measure:
            deadline = time.monotonic() + 240
            for case in cases:
                rounds = 32
                for attempt in range(10):
                    times = [sample(case, a, rounds, 'calibration', attempt, i) for i, a in enumerate(ARMS)]
                    if min(times) >= 75 or max(times) >= 1500 or rounds == MAX_ROUNDS:
                        break
                    rounds = min(MAX_ROUNDS, rounds * 2)
                for pair in range(PAIRS):
                    for order, arm in enumerate(ARMS if pair % 2 == 0 else ARMS[::-1]):
                        if time.monotonic() > deadline:
                            raise TimeoutError('measurement budget exhausted')
                        sample(case, arm, rounds, 'measurement', pair, order)
                print('Measured', case.name, flush=True)
            report['summary'] = summarize(rows, [c.name for c in cases])
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report.get('summary', {'status': report['status']}), indent=2))


if __name__ == '__main__':
    main()
