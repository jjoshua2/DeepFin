#!/usr/bin/env python3
"""Opt-in owning FIFO native checks and non-production ID queue timing screen."""
from __future__ import annotations

import argparse
from collections import deque
from functools import cache
import hashlib
import json
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import tempfile

from native.bend_engine.bitboard_probe.run_probe import compiler_digest

HERE = Path(__file__).resolve().parent
MASK = (1 << 32) - 1
MODES: dict[str, list[str]] = {
    'generic': [],
    'native': ['-march=native'],
    'ubsan': ['-fsanitize=undefined', '-fno-sanitize-recover=all'],
}


def check_compiler(source: Path) -> None:
    """Keep the qualified 2.0.21+U64 pin; do not alter legacy probe manifests."""
    paths = [source / 'bend2' / name for name in ('base.bend', 'bend.ts', 'comp.ts', 'main.ts')]
    paths.extend(p for p in (source / 'bend2/effs').rglob('*') if p.is_file())
    if source.is_symlink() or (source / 'bend2').is_symlink() or (source / 'bend2/effs').is_symlink():
        raise ValueError('compiler root must contain regular sources')
    if len(paths) != 84 or any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('compiler input inventory differs from the 84-file pin')
    if compiler_digest(source) != 'd9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4':
        raise ValueError('compiler sources differ from the qualified 2.0.21+U64 pin')


# Existing session transport entry points and their transitive foreign callers.
# This is an explicit I/O trust boundary, never a source-proof acceptance rule.
SESSION_FOREIGN_DEFS = (
    'Job.load', 'Command.read', 'Reply.read', 'step_path_checked', 'step_traced',
    'step_read', 'step_prepared', 'step', 'run', 'begin_valid', 'begin', 'command',
    'dispatch', 'command_dispatch', 'serve', 'validated', 'start', 'main',
)


def session_notice(returncode: int, stdout: str, stderr: str) -> str:
    """Accept only the unchanged session driver's explicit foreign-I/O notice."""
    expected = 'All terms check, but 18 defs rely on unsafe or foreign code:\n' + '\n'.join('- ' + name for name in SESSION_FOREIGN_DEFS)
    if returncode != 0 or stdout.strip() or stderr.strip() != expected:
        raise ValueError(f'unexpected session compiler diagnostic: exit={returncode}\n{stdout}\n{stderr}')
    return stderr


def expected_trace() -> str:
    """Python deque oracle; every emitted value includes the full U64 payload."""
    q: deque[int] = deque()
    lines: list[str] = []

    def pop() -> None:
        if q:
            x = q.popleft()
            lines.append(f'value {x ^ 2779096485} {x}')
        else:
            lines.append('empty')

    for _ in range(3):
        pop()
    q.extend(range(4294967280, 1 << 32))
    for _ in range(18):
        pop()
    x = 305419896
    for _ in range(4096):
        if (x >> 16) & 7 < 4:
            q.append(x)
        else:
            pop()
        x = (x * 1664525 + 1013904223) & MASK
    for _ in range(4098):
        pop()
    lines.append(f'length {len(q)}')
    return '\n'.join(lines) + '\n'


def verify_trace(text: str) -> None:
    expected = expected_trace()
    if text != expected:
        got, want = text.splitlines(), expected.splitlines()
        mismatch = next((i for i, pair in enumerate(zip(got, want)) if pair[0] != pair[1]), min(len(got), len(want)))
        raise ValueError(f'owning FIFO trace differs at row {mismatch}; rows={len(got)}, expected={len(want)}')


@cache
def expected_checksum(size: int, steps: int = 20000) -> int:
    q = deque(range(size))
    result = 0
    for _ in range(steps):
        x = q.popleft()
        result = ((result * 1664525 + 1013904223) & MASK) ^ x
        q.append(x ^ 2654435769)
    return result


def parse_benchmark(text: str) -> list[dict[str, int | str | bool]]:
    rows: list[dict[str, int | str | bool]] = []
    expected_order = [(arm, size, sample) for size in (16, 256, 4096) for sample in range(6)
                      for arm in (('list', 'queue') if sample % 2 == 0 else ('queue', 'list'))]
    lines = text.splitlines()
    if len(lines) != len(expected_order):
        raise ValueError('missing or extra benchmark samples')
    for line, (expected_arm, expected_size, expected_sample) in zip(lines, expected_order):
        words = line.split()
        if len(words) != 6 or any(not x.isascii() or not x.isdecimal() for x in words[1:]):
            raise ValueError('malformed benchmark row')
        arm = words[0]
        size, sample, ms, checksum, length = map(int, words[1:])
        if (arm, size, sample) != (expected_arm, expected_size, expected_sample):
            raise ValueError('wrong benchmark order or identity')
        if length != size or checksum != expected_checksum(size):
            raise ValueError('benchmark result differs from Python deque')
        rows.append({'arm': arm, 'size': size, 'sample': sample, 'milliseconds': ms,
                     'checksum': checksum, 'warmup': sample == 0, 'below_20ms': ms < 20})
    return rows


def expected_traversal() -> str:
    lines: list[str] = []
    for depth in range(8):
        for leaf, fuel in ((0, 0), (0, 1), (0, 3), (0, 33), (2, 33), (3, 33)):
            lines.append(f'select {depth} {leaf} {fuel} {min(depth, fuel)}')
        for fuel, active in ((0, True), (1, True), (3, True), (33, True), (33, False)):
            for i in range(8):
                visited = active and i <= depth and depth - i < fuel
                n = 3 if visited else 2
                w = 0.25 + (0.5 if (depth - i) % 2 == 0 else -0.5) if visited else 0.25
                bits = int.from_bytes(struct.pack('<f', w), 'little')
                lines.append(f'backup {depth} {fuel} {int(active)} {i} {n} {bits}')
    return '\n'.join(lines) + '\n'


def verify_traversal(text: str) -> None:
    if text != expected_traversal():
        raise ValueError('search traversal differs from independent path/count/sign expectations')


def command(argv: list[str], timeout: int = 300) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode or result.stderr:
        raise RuntimeError(f'{argv}: exit={result.returncode}\n{result.stdout}\n{result.stderr}')
    return result.stdout


def build(source: Path, destination: Path, compiler: Path, bun: str, cc: str, flags: list[str]) -> None:
    generated = destination.with_suffix('.c')
    command([bun, str(compiler / 'bend2/main.ts'), str(source), '-o', str(generated)])
    command([cc, '-std=c11', '-O3', '-ffp-contract=off', *flags, str(generated), '-pthread', '-lm', '-o', str(destination)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not args.bun or not args.cc:
        parser.error('Bun and Clang are required')
    report: dict[str, object] = {'status': 'failed', 'scope': 'FIFO arrays and ID rotation, not scheduler adoption',
                                'machine': platform.platform(), 'compiler_root': str(args.compiler_root)}
    try:
        check_compiler(args.compiler_root)
        with tempfile.TemporaryDirectory(prefix='bend-collections-') as tmp:
            directory = Path(tmp)
            traces: dict[str, str] = {}
            for mode, flags in MODES.items():
                binary = directory / mode
                build(HERE / 'main.bend', binary, args.compiler_root, args.bun, args.cc, flags)
                output = command([str(binary), '--threads', '1'])
                verify_trace(output)
                traces[mode] = hashlib.sha256(output.encode()).hexdigest()
            traversal: dict[str, str] = {}
            for mode in ('generic', 'ubsan'):
                binary = directory / f'traversal-{mode}'
                build(HERE / 'traversal.bend', binary, args.compiler_root, args.bun, args.cc, MODES[mode])
                output = command([str(binary), '--threads', '1'])
                verify_traversal(output)
                traversal[mode] = hashlib.sha256(output.encode()).hexdigest()
            report['traversal_sha256'] = traversal
            report['traversal_rows_per_mode'] = len(expected_traversal().splitlines())
            # Real executable mutation control: remove FIFO reversal, preserving types.
            mutant = directory / 'mutant'
            mutant.mkdir()
            shutil.copyfile(HERE / 'main.bend', mutant / 'main.bend')
            original = (HERE / 'Queue.bend').read_text()
            needle = 'List.reverse(a, T, rear)'
            if original.count(needle) != 1:
                raise ValueError('mutation target changed; review the control')
            (mutant / 'Queue.bend').write_text(original.replace(needle, 'rear'))
            binary = directory / 'lifo-mutant'
            build(mutant / 'main.bend', binary, args.compiler_root, args.bun, args.cc, [])
            output = command([str(binary), '--threads', '1'])
            try:
                verify_trace(output)
            except ValueError:
                report['lifo_mutation_rejected'] = True
            else:
                raise AssertionError('oracle accepted a LIFO queue')
            report['trace_sha256'] = traces
            report['trace_rows_per_mode'] = len(expected_trace().splitlines())
            if args.benchmark:
                binary = directory / 'benchmark'
                build(HERE / 'benchmark.bend', binary, args.compiler_root, args.bun, args.cc, ['-march=native'])
                report['samples'] = parse_benchmark(command([str(binary), '--threads', '1'], timeout=300))
            report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
