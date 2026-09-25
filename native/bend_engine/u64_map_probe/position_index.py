"""Canonical-field interning with forced hash collisions; no engine/cache integration."""
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
from .run_probe import MASK, MODES

HERE = Path(__file__).resolve().parent
Identity = tuple[int, ...]


class Op(NamedTuple):
    action: str
    key: int
    identity: Identity


class Case(NamedTuple):
    name: str
    bits: int
    ops: list[Op]


def encode(case: Case) -> str:
    if type(case.bits) is not int or not 0 <= case.bits <= MASK:
        raise ValueError('invalid position-index capacity')
    lines = [str(case.bits)]
    for action, key, identity in case.ops:
        if action not in ('r', 'g') or type(key) is not int or not 0 <= key < 1 << 64:
            raise ValueError('invalid position-index operation')
        if (len(identity) != 11 or any(type(v) is not int or not 0 <= v < 1 << 64 for v in identity[:8])
                or any(type(v) is not int or not 0 <= v <= MASK for v in identity[8:])):
            raise ValueError('invalid position-index identity')
        values = [key >> 32, key & MASK]
        for value in identity[:8]:
            values.extend((value >> 32, value & MASK))
        values.extend(identity[8:])
        lines.append(action + ' ' + ' '.join(map(str, values)))
    text = ';'.join(lines)
    if len(text) > 100_000:
        raise ValueError('position-index transport exceeds budget')
    return text


def expected(case: Case) -> str:
    """A dict indexed by the complete (hash, fields) pair, not linked-list logic."""
    encode(case)
    if not 1 <= case.bits <= 16:
        return f'invalid {case.bits}\n'
    table: dict[tuple[int, Identity], int] = {}
    capacity = 1 << (case.bits - 1)
    lines = [f'begin {case.bits}']
    for ordinal, op in enumerate(case.ops):
        identity = (op.key, op.identity)
        if identity in table:
            result = f'{"known" if op.action == "r" else "value"} {table[identity]}'
        elif op.action == 'g':
            result = 'missing'
        elif len(table) == capacity:
            result = 'full'
        else:
            table[identity] = len(table)
            result = f'assigned {table[identity]}'
        lines.append(f'{op.action} {ordinal} {result} size {len(table)}')
    return '\n'.join([*lines, f'end {len(table)}']) + '\n'


def verify(text: str, case: Case) -> None:
    if text != expected(case):
        raise ValueError('position-index differs from complete-identity oracle: ' + case.name)


def replay(name: str, records: list[tuple[int, Identity]], bits: int = 7) -> Case:
    ops = [Op(action, key, identity) for key, identity in records for action in ('g', 'r', 'r', 'g')]
    ops.extend(Op('g', key, identity) for key, identity in reversed(records))
    case = Case(name, bits, ops)
    encode(case)
    return case


def fixtures() -> list[Case]:
    cases = [Case(f'constructor-{n}', n, []) for n in (*range(18), MASK)]
    zero: Identity = (0,) * 8 + (1, 0, 64)
    # These are field-sensitivity vectors, not claims that each is a legal board.
    changed: list[Identity] = [zero]
    for field in range(11):
        item: list[int] = list(zero)
        item[field] ^= (1 << 63) if field < 8 else 1
        changed.append(tuple(item))
    cases.append(replay('every-field-same-hash', [(0, p) for p in changed]))
    for bits in (1, 2, 5, 7):
        count = 1 << (bits - 1)
        # Collision records, not distinct hash keys, must consume the entry budget.
        identities = [(*zero[:7], i, *zero[8:]) for i in range(count + 1)]
        ops = [Op('r', MASK, p) for p in identities]
        ops += [Op('r', MASK, p) for p in reversed(identities)]
        ops += [Op('g', MASK, p) for p in identities]
        cases.append(Case(f'collision-full-{bits}', bits, ops))
    rng = random.Random(20260924)
    domain = [(rng.choice((0, MASK, (1 << 64) - 1, 1 << 63)),
               (*(rng.getrandbits(64) for _ in range(8)), 1, 15, 64)) for _ in range(70)]
    cases.append(replay('mixed-hash-chains', [rng.choice(domain) for _ in range(60)], 7))
    cases.append(replay('maximum-allocation-smoke', domain[:3], 16))
    return cases


def chess_cases(corpus: dict[str, Any]) -> list[Case]:
    import chess
    from .chess_workloads import structural

    cases = []
    for group in corpus['corpora']:
        records = []
        for row in group['records']:
            identity = structural(chess.Board(row['fen']))
            # The native Identity uses 64 for absent EP, rather than signed -1.
            records.append((row['key'], (*identity[:-1], 64 if identity[-1] == -1 else identity[-1])))
        for start in range(0, len(records), 32):
            part = records[start:start + 32]
            name = f"chess-{group['name']}-{start // 32}"
            cases.append(replay(name + '-native-hash', part))
            cases.append(replay(name + '-forced-collision', [(0, identity) for _, identity in part]))
    for row in corpus['identity_checks']:
        records = []
        for side in ('left', 'right'):
            identity = structural(chess.Board(row[side + '_fen']))
            records.append((0, (*identity[:-1], 64 if identity[-1] == -1 else identity[-1])))
        cases.append(replay('boundary-' + row['name'], records))
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
    report: dict[str, Any] = {'status': 'failed', 'scope': __doc__, 'modes': [], 'mutations_rejected': []}
    env = dict(os.environ, BEND_NO_TELEMETRY='1')
    env.pop('BEND_U64_CORRUPT', None)
    env.pop('DEEPFIN_POSITION_INDEX', None)

    def command(argv: list[str], name: str, case: Case | None = None) -> str:
        child_env = env if case is None else dict(env, DEEPFIN_POSITION_INDEX=encode(case))
        try:
            result = subprocess.run(argv, env=child_env, capture_output=True, text=True, timeout=120, check=False)
        except subprocess.TimeoutExpired as error:
            for suffix, data in (('stdout', error.stdout), ('stderr', error.stderr)):
                text = data.decode(errors='replace') if isinstance(data, bytes) else data or ''
                (output / f'{name}.{suffix}').write_text(text)
            raise
        (output / f'{name}.stdout').write_text(result.stdout)
        (output / f'{name}.stderr').write_text(result.stderr)
        if result.returncode or result.stderr:
            raise RuntimeError(f'{name}: exit={result.returncode}; see retained diagnostics')
        return result.stdout

    try:
        report['compiler'] = command([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'),
                                      str(args.compiler_root.resolve())], 'compiler')
        report['cc'] = command([args.cc, '--version'], 'cc')
        report['cpu_target'] = cpu_target.qualify(args.cc, output, command)
        cases = fixtures()
        if args.chess:
            from .chess_workloads import collect
            _, _, corpus = collect()
            cases.extend(chess_cases(corpus))
            report['chess_corpus'] = {**corpus, 'corpora': [
                {k: v for k, v in group.items() if k != 'records'} for group in corpus['corpora']]}
        report['sources'] = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                             for n in ('U64Map.bend', 'PositionIndex.bend', 'position_index.bend', 'position_index.py', 'cpu_target.py')}
        report['cases'] = [{'name': c.name, 'bits': c.bits, 'operations': len(c.ops),
                            'input_sha256': hashlib.sha256(encode(c).encode()).hexdigest(),
                            'expected_sha256': hashlib.sha256(expected(c).encode()).hexdigest()} for c in cases]
        compiler = [args.bun, str(args.compiler_root.resolve() / 'bend2/main.ts')]
        generated = output / 'index.c'
        command([*compiler, str(HERE / 'position_index.bend'), '-o', str(generated)], 'generate')
        report['generated_c_sha256'] = hashlib.sha256(generated.read_bytes()).hexdigest()
        for mode, flags in MODES.items():
            binary = output / mode
            command([args.cc, '-std=c11', '-O2', '-ffp-contract=off', *flags,
                     str(generated), '-pthread', '-lm', '-o', str(binary)], 'build-' + mode)
            for case in cases:
                verify(command([str(binary), '--threads', '1'], mode + '-' + case.name, case), case)
            report['modes'].append({'mode': mode, 'cases': len(cases), 'operations': sum(len(c.ops) for c in cases),
                                   'rows': sum(len(expected(c).splitlines()) for c in cases), 'flags': flags})
        source = (HERE / 'PositionIndex.bend').read_text()
        mutants = {
            'hash-only-identity': ('same(identity, stored)', 'True{}', 'every-field-same-hash'),
            'lost-collision-chain': ('Stored{identity, next}), U32.inc(size)', 'Stored{identity, None{}}), U32.inc(size)', 'every-field-same-hash'),
            'reject-known-at-capacity': ('(Index{heads, cells, size, capacity}, Known{id})',
                                         '(Index{heads, cells, size, capacity}, Full{})', 'collision-full-1'),
        }
        for name, (old, new, fixture) in mutants.items():
            if source.count(old) != 1:
                raise ValueError('position-index mutation site changed: ' + name)
            folder = output / name
            folder.mkdir()
            (folder / 'PositionIndex.bend').write_text(source.replace(old, new))
            for filename in ('U64Map.bend', 'position_index.bend'):
                shutil.copyfile(HERE / filename, folder / filename)
            generated = folder / 'index.c'
            command([*compiler, str(folder / 'position_index.bend'), '-o', str(generated)], 'generate-' + name)
            binary = folder / 'index'
            command([args.cc, '-std=c11', '-O2', str(generated), '-pthread', '-lm', '-o', str(binary)], 'build-' + name)
            case = next(c for c in cases if c.name == fixture)
            text = command([str(binary), '--threads', '1'], 'mutation-' + name, case)
            try:
                verify(text, case)
            except ValueError:
                report['mutations_rejected'].append(name)
            else:
                raise AssertionError('position-index mutation survived: ' + name)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
