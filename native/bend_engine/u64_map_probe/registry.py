"""Bounded ID-registry contracts; no node allocation, search integration or timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any, NamedTuple

from . import cpu_target
from .run_probe import MASK, MODES, colliders

HERE = Path(__file__).resolve().parent


class Case(NamedTuple):
    name: str
    bits: int
    first: int
    ops: list[tuple[str, int]]


def encode(case: Case) -> str:
    if any(type(v) is not int or not 0 <= v <= MASK for v in (case.bits, case.first)):
        raise ValueError('invalid registry configuration')
    lines = [f'{case.bits} {case.first}']
    for action, key in case.ops:
        if action not in ('r', 'g') or type(key) is not int or not 0 <= key < 1 << 64:
            raise ValueError('invalid registry operation')
        lines.append(f'{action} {key >> 32} {key & MASK}')
    text = ';'.join(lines)
    if len(text) > 100_000:
        raise ValueError('registry transport exceeds budget')
    return text


def expected(case: Case) -> str:
    """Dictionary plus an unbounded Python integer; never mirror a U32 wrap."""
    encode(case)
    if not 1 <= case.bits <= 16:
        return f'invalid {case.bits} {case.first}\n'
    table: dict[int, int] = {}
    next_id = case.first
    limit = 1 << (case.bits - 1)
    lines = [f'begin {case.bits} {case.first}']

    def state() -> str:
        available = str(next_id) if next_id <= MASK else 'none'
        return f'size {len(table)} next {available}'

    for action, key in case.ops:
        if key in table:
            result = f'{"known" if action == "r" else "value"} {table[key]}'
        elif action == 'g':
            result = 'missing'
        elif next_id > MASK:
            result = 'exhausted'
        elif len(table) == limit:
            result = 'full'
        else:
            table[key] = next_id
            result = f'assigned {next_id}'
            next_id += 1
        lines.append(f'{action} {key >> 32} {key & MASK} {result} {state()}')
    return '\n'.join([*lines, f'end {state()}']) + '\n'


def verify(text: str, case: Case) -> None:
    if text != expected(case):
        raise ValueError(f'registry trace differs from dictionary/counter oracle: {case.name}')


def encounter_case(name: str, keys: list[int]) -> Case:
    if not keys or len(keys) > 520:
        raise ValueError('registry encounter stream exceeds corpus budget')
    # IDs are allocated by native code, not assigned by the fixture producer.
    ops = [op for key in keys for op in (('r', key), ('g', key))]
    ops.extend(('g', key) for key in dict.fromkeys(keys))
    case = Case('chess-' + name, 11, 0, ops)
    encode(case)
    return case


def fixtures() -> list[Case]:
    cases = [Case(f'constructor-{b}', b, 0, []) for b in (*range(18), MASK)]
    for bits in range(1, 7):
        keys = [i << 32 for i in range(1 << (bits - 1))]
        ops = [('g', 0), *[('r', key) for key in keys]]
        ops += [('r', keys[0]), ('g', keys[-1]), ('r', MASK), ('g', MASK),
                ('r', MASK), ('r', keys[-1]), ('g', keys[0])]
        cases.append(Case(f'full-{bits}', bits, 100, ops))
    for count in (1, 2, 4):
        keys = [0, 1 << 63, MASK, (1 << 64) - 1, 1 << 32]
        ops = [('r', key) for key in keys]
        ops += [op for key in keys for op in (('r', key), ('g', key))]
        cases.append(Case(f'last-{count}-ids', 4, MASK - count + 1, ops))
    cases.append(Case('both-limits', 1, MASK,
                      [('r', 0), ('r', 1), ('g', 1), ('r', 0), ('g', 0)]))
    keys = colliders(6, 62, 32, high_only=True)
    ops = [('r', key) for key in keys] + [('r', key) for key in reversed(keys)]
    ops += [('g', key) for key in keys]
    cases.append(Case('wrapped-high-half', 6, 0, ops))
    for seed in range(4):
        rng = random.Random(20260924 + seed)
        keys = [0, MASK, 1 << 32, 1 << 63, (1 << 64) - 1]
        keys += [rng.getrandbits(64) for _ in range(43)]
        ops = [(rng.choice(('r', 'r', 'g')), rng.choice(keys)) for _ in range(512)]
        ops += [('g', key) for key in keys]
        cases.append(Case(f'mixed-{seed}', 6, 0 if seed < 2 else MASK - 15, ops))
    keys = colliders(16, 65535, 3)
    cases.append(Case('maximum-address', 16, 0,
                      [('r', key) for key in keys] + [('r', key) for key in keys] + [('g', key) for key in keys]))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chess', action='store_true')
    args = parser.parse_args()
    if not args.bun or not args.cc or args.output.exists():
        parser.error('Bun, Clang and a fresh output directory are required')
    output = args.output.resolve()
    output.mkdir(parents=True)
    env = dict(os.environ, BEND_NO_TELEMETRY='1')
    env.pop('BEND_U64_CORRUPT', None)
    report: dict[str, Any] = {'status': 'failed', 'scope': __doc__, 'modes': [], 'mutations_rejected': []}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')

    def command(argv: list[str], name: str, case: Case | None = None) -> str:
        child_env = env if case is None else dict(env, DEEPFIN_ID_REGISTRY=encode(case))
        try:
            run = subprocess.run(argv, env=child_env, capture_output=True, text=True, timeout=120, check=False)
        except subprocess.TimeoutExpired as error:
            for suffix, data in (('stdout', error.stdout), ('stderr', error.stderr)):
                text = data.decode(errors='replace') if isinstance(data, bytes) else data or ''
                (output / f'{name}.{suffix}').write_text(text)
            raise
        (output / f'{name}.stdout').write_text(run.stdout)
        (output / f'{name}.stderr').write_text(run.stderr)
        if run.returncode or run.stderr:
            raise RuntimeError(f'{name}: exit={run.returncode}; see retained diagnostics')
        return run.stdout

    try:
        report['compiler'] = command([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'),
                                      str(args.compiler_root.resolve())], 'compiler')
        report['cc'] = command([args.cc, '--version'], 'cc')
        report['cpu_target'] = cpu_target.qualify(args.cc, output, command)
        cases = fixtures()
        if args.chess:
            from .chess_workloads import collect
            _, _, corpus = collect()
            cases += [encounter_case(g['name'], [r['key'] for r in g['records']]) for g in corpus['corpora']]
            report['chess_corpus'] = {**corpus, 'corpora': [
                {k: v for k, v in group.items() if k != 'records'} for group in corpus['corpora']]}
        report['sources'] = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in
                             ('U64Map.bend', 'InsertOnce.bend', 'IdRegistry.bend', 'registry.bend', 'registry.py', 'cpu_target.py')}
        report['cases'] = [{'name': c.name, 'bits': c.bits, 'first_id': c.first, 'operations': len(c.ops),
                            'input_sha256': hashlib.sha256(encode(c).encode()).hexdigest(),
                            'expected_sha256': hashlib.sha256(expected(c).encode()).hexdigest()} for c in cases]
        compiler = str(args.compiler_root.resolve() / 'bend2/main.ts')
        generated = output / 'registry.c'
        command([args.bun, compiler, str(HERE / 'registry.bend'), '-o', str(generated)], 'generate')
        report['generated_c_sha256'] = hashlib.sha256(generated.read_bytes()).hexdigest()
        for mode, flags in MODES.items():
            binary = output / mode
            command([args.cc, '-std=c11', '-O2', '-ffp-contract=off', *flags, str(generated),
                     '-pthread', '-lm', '-o', str(binary)], 'build-' + mode)
            for case in cases:
                verify(command([str(binary), '--threads', '1'], mode + '-' + case.name, case), case)
            report['modes'].append({'mode': mode, 'cases': len(cases), 'operations': sum(len(c.ops) for c in cases),
                                    'rows': sum(len(expected(c).splitlines()) for c in cases), 'exact': True, 'flags': flags})
        source = (HERE / 'IdRegistry.bend').read_text()
        mutants = {
            'advance-known': ('case I.Existing{id}: (Registry{t, Some{candidate}}, Known{id})',
                              'case I.Existing{id}: (Registry{t, after_id(candidate, U32.is_eq(candidate, 4294967295))}, Known{id})', 'full-1'),
            'advance-full': ('case I.Saturated{}: (Registry{t, Some{candidate}}, TableFull{})',
                             'case I.Saturated{}: (Registry{t, after_id(candidate, U32.is_eq(candidate, 4294967295))}, TableFull{})', 'full-1'),
            'wrap-ids': ('case True{}: None{}', 'case True{}: Some{0}', 'last-1-ids'),
            'reject-known-exhausted': ('case Some{id}: (Registry{t, None{}}, Known{id})',
                                       'case Some{id}: (Registry{t, None{}}, IdsExhausted{})', 'last-1-ids'),
        }
        for name, (old, new, fixture) in mutants.items():
            if source.count(old) != 1:
                raise ValueError(f'registry mutation site changed: {name}')
            folder = output / name
            folder.mkdir()
            (folder / 'IdRegistry.bend').write_text(source.replace(old, new))
            for filename in ('U64Map.bend', 'InsertOnce.bend', 'registry.bend'):
                shutil.copyfile(HERE / filename, folder / filename)
            command([args.bun, compiler, str(folder / 'registry.bend'), '-o', str(folder / 'probe.c')], 'generate-' + name)
            command([args.cc, '-std=c11', '-O2', str(folder / 'probe.c'), '-pthread', '-lm', '-o', str(folder / 'probe')], 'build-' + name)
            case = next(c for c in cases if c.name == fixture)
            text = command([str(folder / 'probe'), '--threads', '1'], 'mutation-' + name, case)
            try:
                verify(text, case)
            except ValueError:
                report['mutations_rejected'].append(name)
            else:
                raise AssertionError(f'registry mutation survived: {name}')
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
