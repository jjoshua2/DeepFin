"""Full advertised-capacity map/registry checks; not a performance experiment."""
from __future__ import annotations

import argparse
from collections.abc import Iterator
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any

from . import cpu_target
from .run_probe import MASK, MODES

HERE = Path(__file__).resolve().parent
BITS = (2, 10, 16)
DEPENDENCIES = ('U64Map.bend', 'InsertOnce.bend', 'IdRegistry.bend')


def entry_limit(bits: int) -> int:
    if type(bits) is not int or not 2 <= bits <= 16:
        raise ValueError('invalid scale exponent')
    return 1 << (bits - 1)


def key(index: int) -> int:
    # Low limb is injective throughout these workloads; high limb also varies.
    return ((index * 2654435761 & MASK) << 32) | (index ^ 2779096485)


def map_operations(n: int) -> Iterator[tuple[str, int, int]]:
    yield from (('p', i, 0) for i in range(n))
    yield 'p', n, 0
    yield 'g', n, 0
    yield from (('p', i, MASK) for i in range(n))
    yield from (('e', i, 0) for i in range(n))
    yield 'e', n, 0
    yield from (('g', i, 0) for i in range(n))
    yield from (('d', i, 0) for i in range(0, n, 2))
    yield from (('g', i, 0) for i in range(n))
    yield from (('p', i, 0) for i in range(n, n + n // 2))
    yield from (('g', i, 0) for i in range(n + n // 2))
    yield from (('d', i, 0) for i in range(1, n, 2))
    yield from (('d', i, 0) for i in range(n, n + n // 2))
    yield from (('g', i, 0) for i in range(n + n // 2))
    yield from (('p', 0, 7), ('e', 0, 0), ('g', 0, 0), ('d', 0, 0))


def registry_operations(n: int) -> Iterator[tuple[str, int]]:
    yield from (('r', i) for i in range(n - 1))
    yield from (('a', n - 1), ('g', n - 1), ('c', n), ('g', n - 1), ('r', n - 1))
    yield from (('c', i) for i in range(n - 1))
    yield from (('g', i) for i in range(n - 1))
    yield from (('r', n), ('g', n), ('c', n + 1))


def expected_lines(bits: int) -> Iterator[str]:
    """Python dict plus an unbounded ID counter; no probing/layout logic mirrored."""
    n = entry_limit(bits)
    table: dict[int, int] = {}
    yield f'map {bits} {n}'
    for action, index, salt in map_operations(n):
        k, v = key(index), ((index * 2246822519) & MASK) ^ salt
        if action in ('p', 'e'):
            if k in table:
                outcome = f'{"replaced" if action == "p" else "existing"} {table[k]}'
                if action == 'p':
                    table[k] = v
            elif len(table) == n:
                outcome = 'full'
            else:
                table[k] = v
                outcome = 'added' if action == 'p' else f'inserted {v}'
        else:
            outcome = f'value {table[k]}' if k in table else 'missing'
            if action == 'd':
                table.pop(k, None)
        yield f'{action} {index} {outcome} {len(table)}'
    yield f'map-end {len(table)}'
    for first in (0, MASK - n + 1):
        table = {}
        next_id = first
        yield f'registry {first}'
        for action, index in registry_operations(n):
            k = key(index)
            if k in table:
                outcome = f'{"value" if action == "g" else "known"} {table[k]}'
            elif action == 'g':
                outcome = 'missing'
            elif next_id > MASK:
                outcome = 'exhausted'
            elif len(table) == n:
                outcome = 'full'
            elif action == 'a':
                outcome = f'reserved {next_id} aborted'
            else:
                prefix = f'reserved {next_id} ' if action == 'c' else ''
                outcome = prefix + f'assigned {next_id}'
                table[k] = next_id
                next_id += 1
            next_text = str(next_id) if next_id <= MASK else 'none'
            yield f'{action} {index} {outcome} {len(table)} {next_text}'
        yield f'registry-end {len(table)} {next_id if next_id <= MASK else "none"}'


def verify(text: str, bits: int) -> int:
    """Check the whole trace, including its length and final newline, not a checksum."""
    if not text.endswith('\n'):
        raise ValueError('scale output has no final newline')
    rows = 0
    for rows, (actual, expected) in enumerate(zip_longest(text.splitlines(), expected_lines(bits)), 1):
        if actual != expected:
            raise ValueError(f'scale trace mismatch at row {rows}: {actual!r} != {expected!r}')
    return rows


class Commands:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.records: list[dict[str, Any]] = []

    def run(self, argv: list[str], name: str, bits: int | None = None) -> str:
        env = dict(os.environ, BEND_NO_TELEMETRY='1')
        env.pop('BEND_U64_CORRUPT', None)
        env.pop('DEEPFIN_MAP_SCALE_BITS', None)
        if bits is not None:
            entry_limit(bits)
            env['DEEPFIN_MAP_SCALE_BITS'] = str(bits)
        timed_out = False
        with subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True) as child:
            try:
                stdout, stderr = child.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                stdout, stderr = child.communicate()
            code = child.returncode
        (self.output / f'{name}.stdout').write_text(stdout)
        (self.output / f'{name}.stderr').write_text(stderr)
        self.records.append({'stage': name, 'exit': code, 'timed_out': timed_out, 'argv': argv})
        (self.output / 'commands.json').write_text(json.dumps(self.records, indent=2) + '\n')
        if timed_out or code != 0 or stderr:
            raise RuntimeError(f'{name}: exit={code}, timed_out={timed_out}; see saved diagnostics')
        return stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.bun or not args.cc or args.output.exists():
        parser.error('Bun, Clang and a fresh output directory are required')
    output = args.output.resolve()
    output.mkdir(parents=True)
    commands = Commands(output)
    report: dict[str, Any] = {'status': 'failed', 'scope': __doc__, 'observations': [], 'mutations_rejected': []}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        compiler = [args.bun, str(args.compiler_root.resolve() / 'bend2/main.ts')]
        report['compiler'] = commands.run([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'),
                                          str(args.compiler_root.resolve())], 'compiler')
        report['cc'] = commands.run([args.cc, '--version'], 'cc')
        report['cpu_target'] = cpu_target.qualify(args.cc, output, commands.run)
        report['source_sha256'] = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                                   for n in (*DEPENDENCIES, 'scale.bend', 'scale.py', 'cpu_target.py', 'run_probe.py')}
        generated = output / 'scale.c'
        commands.run([*compiler, str(HERE / 'scale.bend'), '-o', str(generated)], 'generate')
        report['generated_c_sha256'] = hashlib.sha256(generated.read_bytes()).hexdigest()
        for mode, flags in MODES.items():
            binary = output / mode
            commands.run([args.cc, '-std=c11', '-O2', '-ffp-contract=off', *flags,
                          str(generated), '-pthread', '-lm', '-o', str(binary)], 'build-' + mode)
            for bits in BITS:
                text = commands.run([str(binary), '--threads', '1'], f'{mode}-{bits}', bits)
                rows = verify(text, bits)
                report['observations'].append({'mode': mode, 'flags': flags, 'bits': bits,
                    'buckets': 1 << bits, 'entries': entry_limit(bits), 'rows': rows,
                    'operations': rows - 6, 'stdout_sha256': hashlib.sha256(text.encode()).hexdigest()})
        # High-bucket writes must not alias low buckets; a shortened population
        # must not silently turn the advertised maximum check into a smoke test.
        mutants = {
            'fold-upper-buckets': ('U64Map.bend', 'Array.set(Slot, slots, index, Entry{key, value})',
                                  'Array.set(Slot, slots, U32.and(index, 32767), Entry{key, value})'),
            'shortened-population': ('scale.bend', 'map_loop(U32.to_nat(n), 0, 0, 1, 0, t)',
                                    'map_loop(U32.to_nat(U32.shr(n)), 0, 0, 1, 0, t)'),
        }
        for name, (filename, old, new) in mutants.items():
            folder = output / name
            folder.mkdir()
            for dependency in (*DEPENDENCIES, 'scale.bend'):
                shutil.copyfile(HERE / dependency, folder / dependency)
            path = folder / filename
            source = path.read_text()
            if source.count(old) != 1:
                raise ValueError(f'scale mutation site changed: {name}')
            path.write_text(source.replace(old, new))
            generated = folder / 'scale.c'
            commands.run([*compiler, str(folder / 'scale.bend'), '-o', str(generated)], 'generate-' + name)
            binary = folder / 'probe'
            commands.run([args.cc, '-std=c11', '-O2', str(generated), '-pthread', '-lm', '-o', str(binary)], 'build-' + name)
            text = commands.run([str(binary), '--threads', '1'], 'mutation-' + name, 16)
            try:
                verify(text, 16)
            except ValueError as error:
                report['mutations_rejected'].append({'name': name, 'diagnostic': str(error)})
            else:
                raise AssertionError(f'scale mutation survived: {name}')
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
