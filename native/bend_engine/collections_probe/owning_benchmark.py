"""Owning collection contracts and calibrated native timings; no scheduler adoption."""
from __future__ import annotations

import argparse
from collections import deque
from functools import cache
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
MASK = (1 << 32) - 1
CAPACITIES = (0, 1, 2, 3, 7, 16, 17, 4096, 4097, MASK)
ARMS = ('list', 'fifo', 'ring')
SIZES = (1, 16, 64)
MAX_STEPS = 20_000_000
MIN_MS = 50


def expected_ring_trace() -> str:
    lines: list[str] = []
    for capacity in CAPACITIES:
        if capacity > 4096:
            lines.append(f'invalid {capacity}')
            continue
        q: deque[int] = deque()
        lines.append(f'capacity {capacity}')

        def push(x: int) -> None:
            x &= MASK
            if len(q) == capacity:
                lines.append(f'reject {x ^ 2779096485} {x}')
            else:
                q.append(x)
                lines.append('accept')
            lines.append(f'size {len(q)}')

        def pop() -> None:
            if q:
                x = q.popleft()
                lines.append(f'value {x ^ 2779096485} {x}')
            else:
                lines.append('empty')
            lines.append(f'size {len(q)}')

        for _ in range(2):
            pop()
        for x in range(4294967280, 4294967280 + capacity + 2):
            push(x)
        for _ in range(capacity // 2 + 1):
            pop()
        for x in range(100, 100 + capacity + 2):
            push(x)
        x = 305419896 + capacity
        for _ in range(512):
            if (x >> 16) & 7 < 4:
                push(x)
            else:
                pop()
            x = (x * 1664525 + 1013904223) & MASK
        for _ in range(capacity + 2):
            pop()
        push(42)
        push(43)
        for _ in range(3):
            pop()
    return '\n'.join(lines) + '\n'


def verify_ring_trace(text: str) -> None:
    if text != expected_ring_trace():
        raise ValueError('bounded ring trace differs from deque oracle')


@cache
def checksum(size: int, steps: int) -> int:
    """Independent round-robin visit arithmetic; no native collection implementation."""
    result = 0
    for step in range(steps):
        visits, root = divmod(step, size)
        result = ((result * 1664525 + 1013904223) & MASK) ^ root ^ visits
    return result


def expected_roots(size: int, steps: int) -> list[str]:
    full, extra = divmod(steps, size)
    rows = []
    for offset in range(size):
        root = (extra + offset) % size
        visits = full + int(root < extra)
        rows.append(f'root {root} 1 4096 {visits} 4096 {root + 1} 1 0 20 {visits} {root ^ 2779096485} {root}')
    return rows


def parse_sample(text: str, arm: int, size: int, steps: int) -> int:
    lines = text.splitlines()
    if len(lines) != size + 2:
        raise ValueError('owning sample missing or extra rows')
    fields = lines[0].split()
    if len(fields) != 6 or fields[0] != 'sample' or any(not s.isascii() or not s.isdecimal() for s in fields[1:]):
        raise ValueError('owning sample malformed header')
    a, n, rounds, ms, observed = map(int, fields[1:])
    if (a, n, rounds) != (arm, size, steps) or observed != checksum(size, steps):
        raise ValueError('owning sample identity or checksum mismatch')
    if lines[1:-1] != expected_roots(size, steps) or lines[-1] != 'length 0':
        raise ValueError('owning sample root order, visits, identity, history or length mismatch')
    return ms


def timing_summary(rows: list[dict[str, int | str]]) -> dict[str, object]:
    """Never report a ratio from calibration or below-resolution measurements."""
    result: dict[str, object] = {}
    for size in SIZES:
        selected = [row for row in rows if row['phase'] == 'measurement' and row['size'] == size]
        if not selected:
            continue
        times = {arm: [int(row['milliseconds']) for row in selected if row['arm'] == arm] for arm in ARMS}
        reliable = all(len(v) == 6 and min(v) >= MIN_MS for v in times.values())
        medians = {arm: statistics.median(v) for arm, v in times.items() if v}
        result[str(size)] = {'reliable': reliable, 'median_ms': medians,
                             'list_over_fifo': medians['list'] / medians['fifo'] if reliable else None,
                             'fifo_over_ring': medians['fifo'] / medians['ring'] if reliable else None}
    return result


def main() -> None:
    # Import compiler/build helpers only for native execution; parsers remain portable.
    from native.bend_engine.collections_probe.run_probe import MODES, build, check_compiler, command

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not args.bun or not args.cc:
        parser.error('Bun and Clang are required')
    rows: list[dict[str, int | str]] = []
    report: dict[str, object] = {'status': 'failed', 'scope': 'owning collection microbenchmark, not scheduler adoption',
                                'machine': platform.platform(), 'samples': rows, 'minimum_sample_ms': MIN_MS}
    try:
        check_compiler(args.compiler_root)
        report['compiler_root'] = str(args.compiler_root)
        report['cc'] = command([args.cc, '--version'])
        report['bun'] = command([args.bun, '--version'])
        report['sources'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (HERE / name for name in ('Ring.bend', 'Queue.bend', 'ring_trace.bend', 'owning.bend', 'owning_benchmark.py'))}
        with tempfile.TemporaryDirectory(prefix='owning-collections-') as tmp:
            directory = Path(tmp)
            hashes: dict[str, str] = {}
            for mode, flags in MODES.items():
                binary = directory / ('ring-' + mode)
                build(HERE / 'ring_trace.bend', binary, args.compiler_root, args.bun, args.cc, flags)
                text = command([str(binary), '--threads', '1'])
                verify_ring_trace(text)
                hashes[mode] = hashlib.sha256(text.encode()).hexdigest()
            report['ring_trace_sha256'] = hashes
            report['ring_rows_per_mode'] = len(expected_ring_trace().splitlines())
            mutant = directory / 'mutant'
            mutant.mkdir()
            source = (HERE / 'Ring.bend').read_text()
            assert source.count('next(h, c)') == 1
            (mutant / 'Ring.bend').write_text(source.replace('next(h, c)', 'h'))
            shutil.copyfile(HERE / 'ring_trace.bend', mutant / 'ring_trace.bend')
            binary = mutant / 'wrong-head'
            build(mutant / 'ring_trace.bend', binary, args.compiler_root, args.bun, args.cc, [])
            text = command([str(binary), '--threads', '1'])
            try:
                verify_ring_trace(text)
            except ValueError:
                report['wrong_head_mutation_rejected'] = True
            else:
                raise AssertionError('wrong-head mutation survived')

            def sample(binary: Path, arm: int, size: int, steps: int, phase: str, index: int) -> int:
                env = dict(os.environ, DEEPFIN_COLLECTION_BENCH=f'{arm} {size} {steps}')
                run = subprocess.run([str(binary), '--threads', '1'], env=env, capture_output=True,
                                     text=True, timeout=120, check=False)
                if run.returncode or run.stderr:
                    raise RuntimeError(f'owning executable failed: {run.returncode}: {run.stderr}')
                ms = parse_sample(run.stdout, arm, size, steps)
                rows.append({'phase': phase, 'index': index, 'arm': ARMS[arm], 'size': size,
                             'steps': steps, 'milliseconds': ms, 'stdout_sha256': hashlib.sha256(run.stdout.encode()).hexdigest()})
                return ms

            # Test full owning payloads, including zero work and incomplete rounds.
            for mode in ('generic', 'ubsan', 'native'):
                binary = directory / ('owning-' + mode)
                build(HERE / 'owning.bend', binary, args.compiler_root, args.bun, args.cc, MODES[mode])
                for arm in range(3):
                    for size in (1, 3, 16):
                        for steps in (0, 1, 2 * size + 1):
                            sample(binary, arm, size, steps, 'contract-' + mode, 0)
            if args.benchmark:
                binary = directory / 'owning-native'
                for size in SIZES:
                    steps = 32768
                    for attempt in range(11):
                        times = [sample(binary, arm, size, steps, 'calibration', attempt) for arm in range(3)]
                        if min(times) >= 2 * MIN_MS or steps == MAX_STEPS:
                            break
                        steps = min(MAX_STEPS, steps * 2)
                    for index, order in enumerate(permutations(range(3))):
                        for arm in order:
                            sample(binary, arm, size, steps, 'measurement', index)
                report['timing_summary'] = timing_summary(rows)
            report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
