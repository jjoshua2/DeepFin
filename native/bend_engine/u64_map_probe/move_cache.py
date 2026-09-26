"""Legal-move cache versus uncached Bend ordering and the independent CBoard oracle."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from ..legal_probe import run_probe as legal
from . import board_index, cpu_target
from .board_index import Case
from .reservation_ownership import Commands
from .run_probe import MODES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


@dataclass(frozen=True)
class Reference:
    position: legal.Position
    halfmove: int
    fullmove: int
    history: int
    moves: dict[legal.Move, legal.Position]


def canonical(position: legal.Position) -> tuple[int, ...]:
    # Independent square geometry, not the adapter's shifted attack masks.
    ep, side = position[10], position[8]
    pawns = position[0] & position[6 if side else 7]
    usable = ep < 64 and any(pawns & (1 << square)
        and ep // 8 - square // 8 == (1 if side else -1)
        and abs(ep % 8 - square % 8) == 1 for square in range(64))
    return (*position[:10], ep if usable else 64)


def accesses(references: list[Reference], bits: int) -> list[str]:
    if type(bits) is not int or not 1 <= bits <= 16:
        raise ValueError('invalid cache reference capacity')
    stored: set[tuple[int, ...]] = set()
    result = []
    for item in references:
        key = canonical(item.position)
        if key in stored:
            result.append('hit')
        elif len(stored) < 1 << (bits - 1):
            stored.add(key)
            result.append('filled')
        else:
            result.append('bypassed')
    return result


class Oracle:
    def __init__(self, binary: Path, output: Path) -> None:
        self.binary, self.output = binary, output
        self.known: dict[legal.Position, dict[legal.Move, legal.Position]] = {}

    def moves(self, position: legal.Position) -> dict[legal.Move, legal.Position]:
        if position not in self.known:
            request = legal.request(position, mode=1)
            run = subprocess.run([str(self.binary)], input=request, text=True,
                                 capture_output=True, timeout=30, check=False)
            stem = 'oracle-' + hashlib.sha256(request.encode()).hexdigest()
            (self.output / (stem + '.stdout')).write_text(run.stdout)
            (self.output / (stem + '.stderr')).write_text(run.stderr)
            if run.returncode or run.stderr:
                raise ValueError('independent legal oracle failed: ' + stem)
            parsed, _ = legal.parse_moves(run.stdout)
            assert isinstance(parsed, dict)
            self.known[position] = parsed
        return self.known[position]

    def reference(self, command: str) -> Reference:
        words = command.split()
        if words and words[0] == 'startpos':
            fen, remaining = legal.START, words[1:]
        elif len(words) >= 7 and words[0] == 'fen':
            fen, remaining = ' '.join(words[1:7]), words[7:]
        else:
            raise ValueError('invalid reference position')
        position = legal.fen_position(fen)
        half, full = map(int, fen.split()[-2:])
        if remaining and remaining[0] != 'moves':
            raise ValueError('invalid reference move list')
        moves = remaining[1:]
        for uci in moves:
            candidates = self.moves(position)
            def notation(move: legal.Move) -> str:
                src, dst, promotion, _ = move
                return (chr(97 + src % 8) + str(1 + src // 8)
                        + chr(97 + dst % 8) + str(1 + dst // 8)
                        + (' nbrq'[promotion] if promotion else ''))
            chosen = [m for m in candidates if notation(m) == uci]
            if len(chosen) != 1:
                raise ValueError('illegal reference move')
            move = chosen[0]
            pawn = position[0] & (1 << move[0])
            captured = position[7 if position[8] else 6] & (1 << move[1])
            half = 0 if pawn or captured or move[3] == 1 else half + 1
            full += int(position[8] == 0)
            position = candidates[move]
        return Reference(position, half, full, len(moves), self.moves(position))


def verify(text: str, references: list[Reference], bits: int) -> dict[str, int]:
    if not text.endswith('\n'):
        raise ValueError('incomplete move-cache trace')
    lines = iter(text.splitlines())
    routes = accesses(references, bits)
    counts = {name: routes.count(name) for name in ('hit', 'filled', 'bypassed')}
    try:
        for number, (ref, route) in enumerate(zip(references, routes, strict=True)):
            native_order = []
            for _ in ref.moves:
                row = next(lines).split()
                if row[:2] != ['reference', str(number)]:
                    raise ValueError('invalid uncached reference row')
                native_order.append(legal._move(legal._numbers(row[2:])))
            if next(lines) != f'reference-end {number}':
                raise ValueError('incomplete uncached move list')
            if next(lines) != f'request {number} {route} {len(ref.moves)}':
                raise ValueError('cache access or move count differs')
            if next(lines) != f'context {number} {ref.halfmove} {ref.fullmove} {ref.history}':
                raise ValueError('history context changed')
            actual = []
            for _ in ref.moves:
                row = next(lines).split()
                if len(row) != 25 or row[:2] != ['move', str(number)]:
                    raise ValueError('invalid cached move row')
                values = legal._numbers(row[2:])
                actual.append((legal._move(values[:4]), legal._position(values[4:])))
            if len(dict(actual)) != len(actual) or dict(actual) != ref.moves:
                raise ValueError('cached moves/children differ from CBoard')
            if [move for move, _ in actual] != native_order:
                raise ValueError('cache changed native move ordering')
            if next(lines) != f'end {number}':
                raise ValueError('incomplete request')
        if next(lines) != f'done {len(references)}' or next(lines, None) is not None:
            raise ValueError('extra or missing move-cache requests')
    except StopIteration as error:
        raise ValueError('incomplete move-cache trace') from error
    return counts


def fixtures() -> list[Case]:
    cases = board_index.fixtures()
    # Repeated lists include empty results; a cached stalemate is not a cache miss.
    for name, fen in [*((n, f) for n, f, _ in legal.CANONICAL), *legal.EDGES]:
        cases.append(Case('legal-' + name, ('fen ' + fen, 'fen ' + fen)))
    cases += [
        Case('interleaved', ('startpos', 'startpos moves e2e4', 'startpos moves d2d4',
                            'startpos moves d2d4', 'startpos', 'startpos moves e2e4'), 3),
        Case('saturated', ('startpos', 'startpos moves e2e4', 'startpos moves e2e4',
                          'startpos', 'fen 7k/6Q1/5K2/8/8/8/8/8 b - - 0 1', 'startpos'), 1),
        Case('maximum-allocation', ('startpos', 'startpos'), 16),
    ]
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
    commands = Commands(output)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        compiler = [args.bun, str(args.compiler_root.resolve() / 'bend2/main.ts')]
        report['compiler'] = commands.clean([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'),
                                             str(args.compiler_root.resolve())], 'compiler')
        report['cc'] = commands.clean([args.cc, '--version'], 'cc')
        report['cpu_target'] = cpu_target.qualify(args.cc, output, commands.clean)
        oracle_binary = output / 'oracle'
        commands.clean([args.cc, '-std=c11', '-O2', '-DLEGAL_ORACLE', '-I', str(ROOT),
                        str(HERE.parent / 'legal_probe/support.c'), '-pthread', '-lm', '-o', str(oracle_binary)], 'oracle-build')
        oracle = Oracle(oracle_binary, output)
        cases = fixtures()
        if args.chess:
            from .chess_workloads import collect
            _, _, corpus = collect()
            cases += board_index.corpus_cases(corpus)
            report['chess_corpus'] = {**corpus, 'corpora': [
                {k: v for k, v in group.items() if k != 'records'} for group in corpus['corpora']]}
        references = {case.name: [oracle.reference(p) for p in case.positions] for case in cases}
        report['cases'] = [{'name': c.name, 'requests': len(c.positions), 'bits': c.bits,
                            'input_sha256': hashlib.sha256(board_index.encode(c).encode()).hexdigest(),
                            'accesses': accesses(references[c.name], c.bits)} for c in cases]
        dependencies = ('U64Map.bend', 'PositionIndex.bend', 'BoardIndex.bend', 'MoveCache.bend', 'move_cache.bend')
        paths = [*(HERE / n for n in (*dependencies, 'move_cache.py', 'board_index.py', 'cpu_target.py', 'reservation_ownership.py')),
                 *(HERE.parent / 'legal_probe').glob('*.[ch]'), *(ROOT / 'chess_anti_engine/encoding').glob('*.h'),
                 *(HERE.parent / 'standalone').glob('*.bend'), HERE.parent / 'legal_probe/Chess.bend']
        report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        generated = output / 'cache.c'
        commands.clean([*compiler, str(HERE / 'move_cache.bend'), '-o', str(generated)], 'generate')
        report['generated_c_sha256'] = hashlib.sha256(generated.read_bytes()).hexdigest()

        def exercise(binary: Path, mode: str) -> None:
            observations = []
            for case in cases:
                text = commands.clean(['env', 'DEEPFIN_MOVE_CACHE=' + board_index.encode(case), str(binary), '--threads', '1'], mode + '-' + case.name)
                counts = verify(text, references[case.name], case.bits)
                observations.append({'case': case.name, **counts, 'stdout_sha256': hashlib.sha256(text.encode()).hexdigest()})
            report['modes'].append({'mode': mode, 'observations': observations})
            print(f'{mode}: {len(observations)} cache cases passed', flush=True)

        for mode, flags in MODES.items():
            binary = output / mode
            commands.clean([args.cc, '-std=c11', '-O1', '-ffp-contract=off', *flags,
                            str(generated), '-pthread', '-lm', '-o', str(binary)], 'build-' + mode)
            exercise(binary, mode)
        invalid = (('0;startpos', 'invalid move-cache capacity'),
                   ('17;startpos', 'invalid move-cache capacity'),
                   ('7;startpos moves e2e5', 'invalid move-cache position'),
                   ('7;' + ';'.join(['startpos'] * 129), 'move-cache input exceeds limit'))
        for number, (text, error) in enumerate(invalid):
            result = commands.run(['env', 'DEEPFIN_MOVE_CACHE=' + text,
                                   str(output / 'generic'), '--threads', '1'], f'invalid-{number}')
            if result.returncode != 2 or result.stdout or result.stderr != error + '\n':
                raise ValueError('invalid cache input was not rejected as intended')
        report['invalid_inputs_rejected'] = len(invalid)
        # Preserve relative imports through a disposable sibling of the source module.
        with tempfile.TemporaryDirectory(prefix='move-cache-check-', dir=HERE.parent) as temp:
            folder = Path(temp)
            for name in dependencies:
                shutil.copyfile(HERE / name, folder / name)
            adapter = (HERE / 'BoardIndex.bend').read_text()
            if adapter.count('hash(fields)') != 2:
                raise ValueError('constant-hash control site changed')
            (folder / 'BoardIndex.bend').write_text(adapter.replace('hash(fields)', 'U64.zero()'))
            forced = output / 'constant.c'
            commands.clean([*compiler, str(folder / 'move_cache.bend'), '-o', str(forced)], 'constant-generate')
            binary = output / 'constant-ubsan'
            commands.clean([args.cc, '-std=c11', '-O1', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                            str(forced), '-pthread', '-lm', '-o', str(binary)], 'constant-build')
            exercise(binary, 'constant-hash-ubsan')
            (folder / 'BoardIndex.bend').write_text(adapter)
            source = (HERE / 'MoveCache.bend').read_text()
            mutations = {
                'lost-cached-list': ('Array.set(Entry, entries, id, Entry{moves})', 'Array.set(Entry, entries, id, Entry{Nil{}})', 'interleaved'),
                'full-loses-legal-moves': ('Cache{index, entries}, moves, Bypassed{}', 'Cache{index, entries}, Nil{}, Bypassed{}', 'saturated'),
                'wrong-cached-slot': ('Array.get(Entry, entries, id)', 'Array.get(Entry, entries, 0)', 'interleaved'),
            }
            for name, (old, new, fixture) in mutations.items():
                if source.count(old) != 1:
                    raise ValueError('cache mutation site changed: ' + name)
                (folder / 'MoveCache.bend').write_text(source.replace(old, new))
                generated = output / (name + '.c')
                commands.clean([*compiler, str(folder / 'move_cache.bend'), '-o', str(generated)], 'generate-' + name)
                binary = output / name
                commands.clean([args.cc, '-std=c11', '-O1', str(generated), '-pthread', '-lm', '-o', str(binary)], 'build-' + name)
                case = next(c for c in cases if c.name == fixture)
                text = commands.clean(['env', 'DEEPFIN_MOVE_CACHE=' + board_index.encode(case), str(binary), '--threads', '1'], 'mutant-' + name)
                try:
                    verify(text, references[fixture], case.bits)
                except ValueError:
                    report['mutations_rejected'].append(name)
                else:
                    raise AssertionError('cache mutation survived: ' + name)
        report['oracle_positions'] = len(oracle.known)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'modes', 'cases', 'chess_corpus')}, indent=2))


if __name__ == '__main__':
    main()
