"""Native insert-once contracts; no timing, ID allocator, or neural-cache integration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any

from . import cpu_target
from .run_probe import Case, MASK, MODES, Op, colliders

HERE = Path(__file__).resolve().parent


def encode(case: Case) -> str:
    if type(case.bits) is not int or not 0 <= case.bits <= MASK:
        raise ValueError('invalid insert-once capacity')
    lines = [str(case.bits)]
    for op in case.ops:
        if (op.action not in ('e', 'p', 'g', 'd') or type(op.key) is not int
                or not 0 <= op.key < 1 << 64 or type(op.value) is not int or not 0 <= op.value <= MASK):
            raise ValueError('invalid insert-once operation')
        line = f'{op.action} {op.key >> 32} {op.key & MASK}'
        lines.append(line + (f' {op.value}' if op.action in ('e', 'p') else ''))
    text = ';'.join(lines)
    if len(text) > 100_000:
        raise ValueError('insert-once transport exceeds budget')
    return text


def expected(case: Case) -> str:
    """Python dictionary membership/setdefault, not a mirrored probing algorithm."""
    encode(case)
    if not 1 <= case.bits <= 16:
        return f'invalid {case.bits}\n'
    table: dict[int, int] = {}
    lines = [f'begin {case.bits}']
    limit = 1 << (case.bits - 1)
    for action, key, value in case.ops:
        if action == 'e':
            if key in table:
                result = f'existing {table[key]}'
            elif len(table) == limit:
                result = 'full'
            else:
                result = f'inserted {table.setdefault(key, value)}'
        elif action == 'p':
            if key in table:
                result = f'replaced {table[key]}'
                table[key] = value
            elif len(table) == limit:
                result = 'full'
            else:
                table[key] = value
                result = 'added'
        else:
            result = f'value {table[key]}' if key in table else 'missing'
            if action == 'd':
                table.pop(key, None)
        lines.append(f'{action} {key >> 32} {key & MASK} {result} {len(table)}')
    return '\n'.join([*lines, f'end {len(table)}']) + '\n'


def verify(text: str, case: Case) -> None:
    if text != expected(case):
        raise ValueError(f'insert-once trace differs from dictionary: {case.name}')


def from_encounters(name: str, keys: list[int]) -> Case:
    # Candidate IDs deliberately advance even on revisits. The first value must
    # survive; a later proposal is not authority to replace an earlier binding.
    if not keys or len(keys) > 520:
        raise ValueError('encounter stream exceeds fixed corpus budget')
    ops = [op for index, key in enumerate(keys)
           for op in (Op('e', key, index), Op('g', key))]
    ops.extend(Op('g', key) for key in dict.fromkeys(keys))
    case = Case('insert-once-' + name, 11, ops)
    encode(case)
    return case


def fixtures() -> list[Case]:
    cases = [Case(f'constructor-{bits}', bits, []) for bits in (0, 17, MASK)]
    for bits in range(1, 7):
        keys = [(i + 1) << 32 for i in range(1 << (bits - 1))]
        ops = [Op('e', key, i) for i, key in enumerate(keys)]
        ops += [Op('e', keys[0], MASK), Op('g', keys[0]), Op('e', 0, 7), Op('g', 0)]
        ops += [Op('d', keys[-1]), Op('e', 0, MASK), Op('e', 0, 0), Op('g', 0)]
        ops += [Op('p', keys[0], 31), Op('e', keys[0], 63), Op('g', keys[0])]
        ops += [Op('g', key) for key in keys]
        cases.append(Case(f'limit-{bits}', bits, ops))
    cases.append(Case('zero-max', 3, [
        Op('e', 0, 0), Op('e', (1 << 64) - 1, MASK), Op('e', 0, MASK),
        Op('e', (1 << 64) - 1, 0), Op('g', 0), Op('g', (1 << 64) - 1),
        Op('d', 0), Op('e', 0, 19), Op('e', 0, 20), Op('g', 0),
    ]))
    keys = colliders(4, 14, 8, high_only=True)
    ops = [Op('e', key, i) for i, key in enumerate(keys)]
    ops += [Op('d', keys[0]), Op('d', keys[4])]
    ops += [Op('e', key, MASK - i) for i, key in enumerate(keys)]
    ops += [Op('g', key) for key in keys]
    cases.append(Case('wrapped-high-half', 4, ops))
    for seed in range(4):
        rng = random.Random(20260924 + seed)
        keys = [0, MASK, 1 << 32, 1 << 63, (1 << 64) - 1]
        keys += [rng.getrandbits(64) for _ in range(43)]
        ops = [Op(rng.choice(('e', 'e', 'p', 'g', 'd')), rng.choice(keys), rng.getrandbits(32))
               for _ in range(512)]
        ops += [Op('g', key) for key in keys]
        cases.append(Case(f'mixed-{seed}', 6, ops))
    keys = colliders(16, 65535, 3)
    ops = [Op('e', key, i) for i, key in enumerate(keys)]
    ops += [Op('d', keys[0]), *[Op('e', key, MASK) for key in keys], *[Op('g', key) for key in keys]]
    cases.append(Case('maximum-address', 16, ops))
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
        child_env = env if case is None else dict(env, DEEPFIN_MAP_INSERT_ONCE=encode(case))
        run = subprocess.run(argv, env=child_env, capture_output=True, text=True, timeout=120, check=False)
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
            for group in corpus['corpora']:
                cases.append(from_encounters(group['name'], [row['key'] for row in group['records']]))
            report['chess_corpus'] = {**corpus, 'corpora': [
                {k: v for k, v in group.items() if k != 'records'} for group in corpus['corpora']]}
        report['sources'] = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                             for n in ('U64Map.bend', 'InsertOnce.bend', 'insert_once.bend', 'insert_once.py', 'cpu_target.py')}
        report['cases'] = [{'name': c.name, 'operations': len(c.ops),
                            'input_sha256': hashlib.sha256(encode(c).encode()).hexdigest(),
                            'expected_sha256': hashlib.sha256(expected(c).encode()).hexdigest()} for c in cases]
        compiler = str(args.compiler_root.resolve() / 'bend2/main.ts')
        generated = output / 'insert_once.c'
        command([args.bun, compiler, str(HERE / 'insert_once.bend'), '-o', str(generated)], 'generate')
        report['generated_c_sha256'] = hashlib.sha256(generated.read_bytes()).hexdigest()
        for mode, flags in MODES.items():
            binary = output / mode
            command([args.cc, '-std=c11', '-O2', '-ffp-contract=off', *flags, str(generated),
                     '-pthread', '-lm', '-o', str(binary)], 'build-' + mode)
            for case in cases:
                verify(command([str(binary), '--threads', '1'], mode + '-' + case.name, case), case)
            report['modes'].append({'mode': mode, 'cases': len(cases), 'operations': sum(len(c.ops) for c in cases),
                                    'exact': True, 'flags': flags})
        source = (HERE / 'InsertOnce.bend').read_text()
        old = 'case M.Present{index, previous}: (t, Existing{previous})'
        mutants = {
            'overwrite-existing': 'case M.Present{index, previous}:\n      stored(candidate, M.put_at(key, candidate, (t, M.Present{index, previous})))',
            'return-proposal': 'case M.Present{index, previous}: (t, Existing{candidate})',
            'reject-existing': 'case M.Present{index, previous}: (t, Saturated{})',
        }
        if source.count(old) != 1:
            raise ValueError('insert-once mutation site changed')
        for name, replacement in mutants.items():
            folder = output / name
            folder.mkdir()
            (folder / 'InsertOnce.bend').write_text(source.replace(old, replacement))
            for filename in ('U64Map.bend', 'insert_once.bend'):
                shutil.copyfile(HERE / filename, folder / filename)
            command([args.bun, compiler, str(folder / 'insert_once.bend'), '-o', str(folder / 'probe.c')], 'generate-' + name)
            command([args.cc, '-std=c11', '-O2', str(folder / 'probe.c'), '-pthread', '-lm', '-o', str(folder / 'probe')], 'build-' + name)
            case = next(c for c in cases if c.name == 'limit-1')
            text = command([str(folder / 'probe'), '--threads', '1'], 'mutation-' + name, case)
            try:
                verify(text, case)
            except ValueError:
                report['mutations_rejected'].append(name)
            else:
                raise AssertionError(f'insert-once mutation survived: {name}')
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
