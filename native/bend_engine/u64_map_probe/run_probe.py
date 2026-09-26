"""Opt-in native U64 map qualification against Python dict; no engine integration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from typing import NamedTuple

from . import cpu_target

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MASK = (1 << 32) - 1
MODES = {'generic': [], 'portable': ['-DBEND_U64_PORTABLE'], cpu_target.TARGET_NAME: list(cpu_target.TARGET_FLAGS),
         'ubsan': ['-fsanitize=undefined', '-fno-sanitize-recover=all']}


class Op(NamedTuple):
    action: str
    key: int
    value: int = 0


class Case(NamedTuple):
    name: str
    bits: int
    ops: list[Op]


def bucket(key: int, mask: int) -> int:
    """Fixture construction only: choose collisions. The value oracle uses dict."""
    folded = (key & MASK) ^ (((key >> 32) * 2654435761) & MASK)
    mixed = ((folded ^ (folded >> 16)) * 2246822519) & MASK
    return (mixed ^ (mixed >> 13)) & mask


def colliders(bits: int, home: int, count: int, *, high_only: bool = False) -> list[int]:
    found = []
    for number in range(2_000_000):
        key = number << 32 if high_only else number
        if bucket(key, (1 << bits) - 1) == home:
            found.append(key)
            if len(found) == count:
                return found
    raise ValueError('collision fixture search exhausted its bound')


def fixtures() -> list[Case]:
    result = [Case(f'constructor-{n}', n, []) for n in (*range(17), 17, MASK)]
    for bits in range(1, 7):
        limit = 1 << (bits - 1)
        keys = [(i * 0x9e3779b97f4a7c15) & ((1 << 64) - 1) for i in range(limit)]
        ops = [Op('p', key, (MASK - i)) for i, key in enumerate(keys)]
        ops += [Op('p', (1 << 64) - 1, 7), Op('p', keys[-1], 0)]
        ops += [Op('g', key) for key in keys]
        ops += [Op('d', keys[0]), Op('d', keys[0]), Op('p', (1 << 64) - 1, MASK)]
        ops += [Op('g', key) for key in [*keys, (1 << 64) - 1]]
        ops += [Op('d', key) for key in keys]
        ops += [Op('d', (1 << 64) - 1), Op('p', keys[0], 19), Op('g', keys[0])]
        result.append(Case(f'limit-{bits}', bits, ops))
    # All low halves are zero and bucket positions collide: dropping high bits
    # cannot hide behind distinct home buckets. Cluster crosses bucket 15 -> 0.
    keys = colliders(4, 14, 8, high_only=True)
    ops = [Op('p', k, i) for i, k in enumerate(keys)]
    for key in (keys[0], keys[4], keys[-1], keys[2]):
        ops += [Op('d', key), *[Op('g', k) for k in keys]]
    ops += [Op('p', k, MASK - i) for i, k in enumerate(keys)]
    ops += [Op('g', k) for k in keys]
    result.append(Case('high-half-wrapped-cluster', 4, ops))
    # Different homes interleave; deletion must sometimes skip and sometimes move.
    keys = colliders(4, 14, 3) + colliders(4, 15, 2) + colliders(4, 0, 2)
    ops = [Op('p', k, i + 11) for i, k in enumerate(keys)]
    for key in keys:
        ops += [Op('d', key), *[Op('g', k) for k in keys]]
    result.append(Case('mixed-home-wrapped-cluster', 4, ops))
    for seed in range(4):
        rng = random.Random(20260924 + seed)
        domain = [0, (1 << 64) - 1, 1 << 63, 1 << 32, MASK]
        domain += [rng.getrandbits(64) for _ in range(80)]
        ops = []
        for step in range(800):
            ops.append(Op(rng.choice(('p', 'p', 'g', 'd')), rng.choice(domain), rng.getrandbits(32)))
            if step % 100 == 99:
                ops.extend(Op('g', key) for key in domain)
        ops.extend(Op('d', key) for key in domain)
        ops.extend(Op('g', key) for key in domain)
        result.append(Case(f'churn-{seed}', 6, ops))
    # High physical bucket, zero/MAX keys and values, but not a full 32K-entry test.
    keys = [*colliders(16, 65535, 3), 0, (1 << 64) - 1, 1 << 63]
    ops = [Op('p', key, i) for i, key in enumerate(keys)]
    ops += [Op('d', keys[0]), *[Op('g', key) for key in keys]]
    result.append(Case('maximum-address-smoke', 16, ops))
    return result


def encode(case: Case) -> str:
    lines = [str(case.bits)]
    for op in case.ops:
        if op.action not in ('p', 'g', 'd') or not 0 <= op.key < 1 << 64 or not 0 <= op.value <= MASK:
            raise ValueError('invalid fixture operation')
        words = f'{op.action} {op.key >> 32} {op.key & MASK}'
        lines.append(words + (f' {op.value}' if op.action == 'p' else ''))
    text = ';'.join(lines)
    if len(text) > 100_000:
        raise ValueError('probe input exceeds bounded environment transport')
    return text


def expected(case: Case) -> str:
    """Independent semantics, no open-addressing or deletion algorithm mirrored."""
    if not 1 <= case.bits <= 16:
        return f'invalid {case.bits}\n'
    table: dict[int, int] = {}
    lines = [f'begin {case.bits}']
    limit = 1 << (case.bits - 1)
    for op in case.ops:
        prefix = f'{op.action} {op.key >> 32} {op.key & MASK}'
        if op.action == 'p':
            if op.key in table:
                disposition = f'replaced {table[op.key]}'
                table[op.key] = op.value
            elif len(table) == limit:
                disposition = 'full'
            else:
                disposition = 'added'
                table[op.key] = op.value
        elif op.action in ('g', 'd'):
            disposition = f'value {table[op.key]}' if op.key in table else 'missing'
            if op.action == 'd':
                table.pop(op.key, None)
        else:
            raise ValueError('invalid fixture operation')
        lines.append(f'{prefix} {disposition} {len(table)}')
    return '\n'.join([*lines, f'end {len(table)}']) + '\n'


def verify(text: str, case: Case) -> None:
    if text != expected(case):
        raise ValueError(f'map trace differs from dict oracle: {case.name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.bun or not args.cc:
        parser.error('Bun and Clang are required')
    output = args.output.resolve()
    if output.exists():
        parser.error('output exists; preserve old evidence')
    output.mkdir(parents=True)
    report: dict[str, object] = {'status': 'failed', 'scope': 'standalone exact map; no search/cache integration'}
    clean_env = dict(os.environ, BEND_NO_TELEMETRY='1')
    clean_env.pop('BEND_U64_CORRUPT', None)

    def command(argv: list[str], name: str, env: dict[str, str] | None = None) -> str:
        run = subprocess.run(argv, env=clean_env if env is None else env, capture_output=True,
                             text=True, timeout=120, check=False)
        (output / f'{name}.stdout').write_text(run.stdout)
        (output / f'{name}.stderr').write_text(run.stderr)
        if run.returncode or run.stderr:
            raise RuntimeError(f'{name}: exit={run.returncode}; see captured diagnostics')
        return run.stdout

    try:
        report['compiler'] = command([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'),
                                      str(args.compiler_root.resolve())], 'compiler')
        report['cc'] = command([args.cc, '--version'], 'cc')
        report['cpu_target'] = cpu_target.qualify(args.cc, output, command)
        report['mode_flags'] = MODES
        report['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (HERE / n for n in ('U64Map.bend', 'main.bend', 'run_probe.py', 'cpu_target.py'))}
        cases = fixtures()
        report['cases'] = [{'name': c.name, 'bits': c.bits, 'operations': len(c.ops),
                            'input_sha256': hashlib.sha256(encode(c).encode()).hexdigest(),
                            'expected_sha256': hashlib.sha256(expected(c).encode()).hexdigest()} for c in cases]
        compiler = str(args.compiler_root.resolve() / 'bend2/main.ts')
        generated = output / 'map.c'
        command([args.bun, compiler, str(HERE / 'main.bend'), '-o', str(generated)], 'generate')
        results = []
        report['modes'] = results
        for mode, flags in MODES.items():
            binary = output / mode
            command([args.cc, '-std=c11', '-O2', '-ffp-contract=off', *flags, str(generated),
                     '-pthread', '-lm', '-o', str(binary)], 'build-' + mode)
            for case in cases:
                text = command([str(binary), '--threads', '1'], mode + '-' + case.name,
                               dict(clean_env, DEEPFIN_U64_MAP_TRACE=encode(case)))
                verify(text, case)
            results.append({'mode': mode, 'cases': len(cases),
                            'operations': sum(len(c.ops) for c in cases), 'exact': True})
        source = (HERE / 'U64Map.bend').read_text()
        mutants = {
            'low-half-only': ('U64.is_eq(key, stored)', 'U32.is_eq(U64.low(key), U64.low(stored))'),
            'no-backshift': ('U32.is_lt(U32.and(U32.sub(hole, home), mask), U32.and(U32.sub(cursor, home), mask))', 'False{}'),
            'ignore-entry-limit': ('U32.is_ge(size, U32.shr(U32.inc(mask)))', 'False{}'),
        }
        rejected: list[str] = []
        report['mutations_rejected'] = rejected
        for name, (old, new) in mutants.items():
            if source.count(old) != 1:
                raise ValueError(f'mutation site changed: {name}')
            folder = output / name
            folder.mkdir()
            (folder / 'U64Map.bend').write_text(source.replace(old, new))
            shutil.copyfile(HERE / 'main.bend', folder / 'main.bend')
            command([args.bun, compiler, str(folder / 'main.bend'), '-o', str(folder / 'map.c')], 'generate-' + name)
            command([args.cc, '-std=c11', '-O2', str(folder / 'map.c'), '-pthread', '-lm',
                     '-o', str(folder / 'map')], 'build-' + name)
            case = next(c for c in cases if c.name == ('limit-1' if name == 'ignore-entry-limit' else 'high-half-wrapped-cluster'))
            text = command([str(folder / 'map'), '--threads', '1'], 'mutant-' + name,
                           dict(clean_env, DEEPFIN_U64_MAP_TRACE=encode(case)))
            try:
                verify(text, case)
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError(f'mutation survived: {name}')
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
