"""Full PositionIndex record populations and collision chains; no performance claim."""
from __future__ import annotations

import argparse
from collections.abc import Iterator
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, NamedTuple

HERE = Path(__file__).resolve().parent
MASK = (1 << 32) - 1
SCENARIOS = (*((bits, width) for bits in (2, 10, 16) for width in (1, 8)), (10, 512))
INVALID = ('16 32768', '17 1', '2 0', '10 2', '2 x', '2 1 extra')


class Op(NamedTuple):
    action: str
    record: int
    changed: int
    group: int


def population(bits: int, width: int) -> int:
    if type(bits) is not int or type(width) is not int or (bits, width) not in SCENARIOS:
        raise ValueError('invalid position scale configuration')
    return 1 << (bits - 1)


def key(group: int) -> int:
    return ((group * 2654435761 & MASK) << 32) | (group ^ 2779096485)


def identity(index: int, changed: int) -> tuple[int, ...]:
    fields = [(((index + field) ^ (0 if field == changed else 1 << 31)) << 32)
              | (((index * 2654435761) & MASK) ^ ((field * 2246822519) & MASK))
              for field in range(8)]
    fields.extend(((index & mask) ^ int(changed == field))
                  for field, mask in ((8, 1), (9, 15), (10, 63)))
    return tuple(fields)


def operations(bits: int, width: int) -> Iterator[Op]:
    n = population(bits, width)
    yield Op('g', 0, 11, 0)
    yield from (Op('r', i, 11, i // width) for i in range(n))
    yield from (Op('g', i, 11, i // width) for i in reversed(range(n)))
    yield from (Op('r', i, 11, i // width) for i in range(n))
    # Missing identity sharing an existing hash; then a genuinely missing hash.
    yield from (Op(action, n, 11, 0) for action in ('r', 'g'))
    yield from (Op(action, n + 1, 11, n // width + 1) for action in ('r', 'g'))
    for i in (0, n - 1):
        for field in range(11):
            yield from (Op(action, i, field, i // width) for action in ('g', 'r'))
    # 73 is odd, so multiplication permutes all indices of this power-of-two size.
    for ordinal in range(n):
        i = ordinal * 73 & (n - 1)
        yield Op('g', i, 11, i // width)


def expected_lines(bits: int, width: int) -> Iterator[str]:
    """Full-key dictionary semantics; no probe or collision-chain algorithm copied."""
    n = population(bits, width)
    table: dict[tuple[int, tuple[int, ...]], int] = {}
    yield f'position-scale {bits} {width} {n}'
    for action, index, changed, group in operations(bits, width):
        binding = key(group), identity(index, changed)
        if binding in table:
            outcome = f'{"known" if action == "r" else "value"} {table[binding]}'
        elif action == 'g':
            outcome = 'missing'
        elif len(table) == n:
            outcome = 'full'
        else:
            table[binding] = len(table)
            outcome = f'assigned {table[binding]}'
        yield f'{action} {index} {changed} {group} {outcome} size {len(table)}'
    yield f'end {len(table)}'


def verify(text: str, bits: int, width: int) -> int:
    if not text.endswith('\n'):
        raise ValueError('position scale trace lacks final newline')
    rows = 0
    for rows, (actual, expected) in enumerate(zip_longest(text.splitlines(), expected_lines(bits, width)), 1):
        if actual != expected:
            raise ValueError(f'position scale mismatch at row {rows}: {actual!r} != {expected!r}')
    return rows


def check_invalid(result: subprocess.CompletedProcess[str]) -> None:
    if (result.returncode != 2 or result.stdout
            or result.stderr != 'invalid position scale configuration\n'):
        raise ValueError('invalid scale configuration was not rejected as intended')


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
    report: dict[str, Any] = {'status': 'failed', 'scope': __doc__, 'observations': [],
                              'mutations_rejected': [], 'invalid_configurations': []}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        from . import cpu_target
        from .reservation_ownership import Commands
        from .run_probe import MODES

        commands = Commands(output)
        compiler = [args.bun, str(args.compiler_root.resolve() / 'bend2/main.ts')]
        report['compiler'] = commands.clean([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'),
                                             str(args.compiler_root.resolve())], 'compiler')
        report['cc'] = commands.clean([args.cc, '--version'], 'cc')
        report['cpu_target'] = cpu_target.qualify(args.cc, output, commands.clean)
        dependencies = ('U64Map.bend', 'PositionIndex.bend')
        report['source_sha256'] = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
            for n in (*dependencies, 'position_scale.bend', 'position_scale.py',
                      'reservation_ownership.py', 'cpu_target.py', 'run_probe.py')}
        generated = output / 'scale.c'
        commands.clean([*compiler, str(HERE / 'position_scale.bend'), '-o', str(generated)], 'generate')
        report['generated_c_sha256'] = hashlib.sha256(generated.read_bytes()).hexdigest()
        for mode, flags in MODES.items():
            binary = output / mode
            commands.clean([args.cc, '-std=c11', '-O2', '-ffp-contract=off', *flags,
                            str(generated), '-pthread', '-lm', '-o', str(binary)], 'build-' + mode)
            for bits, width in SCENARIOS:
                # env replaces the driver variable even if the parent inherited it.
                argv = ['env', f'DEEPFIN_POSITION_SCALE={bits} {width}', str(binary), '--threads', '1']
                text = commands.clean(argv, f'{mode}-{bits}-{width}')
                rows = verify(text, bits, width)
                n = population(bits, width)
                report['observations'].append({'mode': mode, 'flags': flags, 'bits': bits,
                    'records': n, 'hash_groups': (n + width - 1) // width,
                    'maximum_chain': min(n, width), 'width': width, 'rows': rows,
                    'operations': rows - 2, 'stdout_sha256': hashlib.sha256(text.encode()).hexdigest()})
            for ordinal, config in enumerate(INVALID):
                result = commands.run(['env', 'DEEPFIN_POSITION_SCALE=' + config,
                                       str(binary), '--threads', '1'], f'invalid-{mode}-{ordinal}')
                check_invalid(result)
                report['invalid_configurations'].append({'mode': mode, 'config': config})
        mutants = {
            'alias-upper-records': ('PositionIndex.bend',
                'Array.set(Cell, cells, size, Stored{identity, next})',
                'Array.set(Cell, cells, U32.and(size, 16383), Stored{identity, next})'),
            'shortened-population': ('position_scale.bend',
                'loop(U32.to_nat(n), 0, 0, 1, width, index)\n    index : P.Index <- loop(U32.to_nat(n), 1,',
                'loop(U32.to_nat(U32.shr(n)), 0, 0, 1, width, index)\n    index : P.Index <- loop(U32.to_nat(n), 1,'),
        }
        for name, (filename, old, new) in mutants.items():
            folder = output / name
            folder.mkdir()
            for dependency in (*dependencies, 'position_scale.bend'):
                shutil.copyfile(HERE / dependency, folder / dependency)
            path = folder / filename
            source = path.read_text()
            if source.count(old) != 1:
                raise ValueError('position scale mutation site changed: ' + name)
            path.write_text(source.replace(old, new))
            generated = folder / 'mutant.c'
            commands.clean([*compiler, str(folder / 'position_scale.bend'), '-o', str(generated)], 'generate-' + name)
            binary = folder / 'mutant'
            commands.clean([args.cc, '-std=c11', '-O2', str(generated), '-pthread', '-lm', '-o', str(binary)], 'build-' + name)
            text = commands.clean(['env', 'DEEPFIN_POSITION_SCALE=16 8', str(binary), '--threads', '1'], 'mutation-' + name)
            try:
                verify(text, 16, 8)
            except ValueError as error:
                report['mutations_rejected'].append({'name': name, 'diagnostic': str(error)})
            else:
                raise AssertionError('position scale mutation survived: ' + name)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
