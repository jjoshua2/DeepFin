"""Native Board-to-PositionIndex adapter qualification; no neural cache or timing."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
ROOT = HERE.parents[2]
U64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class Case:
    name: str
    positions: tuple[str, ...]
    bits: int = 7


@dataclass(frozen=True)
class Snapshot:
    identity: tuple[int, ...]
    raw_ep: int
    halfmove: int
    fullmove: int
    history: int


def encode(case: Case) -> str:
    if type(case.bits) is not int or not 1 <= case.bits <= 16:
        raise ValueError('invalid board-index capacity')
    if not 1 <= len(case.positions) <= 128:
        raise ValueError('invalid board-index position count')
    if any(not p or not p.isascii() or any(c in p for c in ';\n\r\x00') for p in case.positions):
        raise ValueError('invalid board-index position transport')
    text = ';'.join((str(case.bits), *case.positions))
    if len(text) > 100_000:
        raise ValueError('board-index transport exceeds budget')
    return text


def fingerprint(fields: tuple[int, ...]) -> int:
    if len(fields) != 11 or any(type(v) is not int or not 0 <= v <= U64_MASK for v in fields):
        raise ValueError('invalid board identity words')
    value = 0xCBF29CE484222325
    for word in fields:
        value = ((value ^ word) * 0x100000001B3) & U64_MASK
    return value


def expected(snapshots: list[Snapshot], bits: int) -> str:
    # IDs depend only on the complete identity, never the hash/mixer model.
    if type(bits) is not int or not 1 <= bits <= 16:
        raise ValueError('invalid reference capacity')
    ids: dict[tuple[int, ...], int] = {}
    lines = []
    for ordinal, snap in enumerate(snapshots):
        fields = snap.identity
        hashed = fingerprint(fields)
        words = [limb for v in fields[:8] for limb in (v >> 32, v & MASK)]
        words.extend(fields[8:])
        lines.append(f'identity {ordinal} ' + ' '.join(map(str, words)))
        lines.append(f'hash {ordinal} {hashed >> 32} {hashed & MASK}')
        lines.append(f'context {ordinal} {snap.raw_ep} {snap.halfmove} {snap.fullmove} {snap.history}')
        if fields in ids:
            outcome = f'known {ids[fields]}'
        elif len(ids) == 1 << (bits - 1):
            outcome = 'full'
        else:
            ids[fields] = len(ids)
            outcome = f'assigned {ids[fields]}'
        lines.append(f'intern {ordinal} {outcome} size {len(ids)}')
        lookup = f'value {ids[fields]}' if fields in ids else 'missing'
        lines.append(f'get {ordinal} {lookup} size {len(ids)}')
    return '\n'.join([*lines, f'end {len(ids)}']) + '\n'


def verify(text: str, reference_text: str) -> None:
    if text != reference_text:
        raise ValueError('native board-index output differs from reference')



def reference(case: Case) -> list[Snapshot]:
    import chess

    encode(case)
    result = []
    for command in case.positions:
        tokens = command.split()
        if tokens[0] == 'startpos':
            board, rest = chess.Board(), tokens[1:]
        elif tokens[0] == 'fen' and len(tokens) >= 7:
            board, rest = chess.Board(' '.join(tokens[1:7])), tokens[7:]
        else:
            raise ValueError('invalid reference position command')
        if not board.is_valid() or (rest and rest[0] != 'moves'):
            raise ValueError('invalid reference position or move list')
        for move in rest[1:]:
            board.push_uci(move)  # Rejects illegal moves, not merely malformed UCI.
        rights = (int(board.has_kingside_castling_rights(chess.WHITE))
                  | int(board.has_queenside_castling_rights(chess.WHITE)) << 1
                  | int(board.has_kingside_castling_rights(chess.BLACK)) << 2
                  | int(board.has_queenside_castling_rights(chess.BLACK)) << 3)
        raw_ep = board.ep_square if board.ep_square is not None else 64
        fields = (board.pawns, board.knights, board.bishops, board.rooks, board.queens,
                  board.kings, board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
                  int(board.turn), rights, raw_ep if board.has_pseudo_legal_en_passant() else 64)
        result.append(Snapshot(fields, raw_ep, board.halfmove_clock, board.fullmove_number, len(board.move_stack)))
    return result


def fixtures() -> list[Case]:
    start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'
    pushed = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq'
    cases = [Case('history-and-unused-ep', (
        'startpos', 'startpos moves g1f3 g8f6 f3g1 f6g8', f'fen {start} 99 42',
        'startpos moves e2e4', f'fen {pushed} - 0 1', f'fen {pushed} e3 0 1',
        'startpos moves d2d4', 'startpos moves g1f3',
        'startpos moves g1f3 g8f6', 'startpos moves g2g3 g7g6 g1f3',
        'startpos moves g1f3 g7g6 g2g3')),
        Case('full-wrapper', ('startpos', 'startpos moves e2e4', 'startpos'), 1)]
    ep_positions = (
        '4k3/8/8/3pP3/8/8/8/4K3 w - d6',
        '4k3/8/8/8/3Pp3/8/8/4K3 b - d3',
        '4r1k1/8/8/3pP3/8/8/8/4K3 w - d6',
        '4k3/8/8/8/3Pp3/8/8/4R1K1 b - d3',
        '4k3/8/8/pP6/8/8/8/4K3 w - a6',
        '4k3/8/8/6Pp/8/8/8/4K3 w - h6',
        '4k3/8/8/8/Pp6/8/8/4K3 b - a3',
        '4k3/8/8/8/6pP/8/8/4K3 b - h3',
    )
    for number, fen in enumerate(ep_positions):
        without = fen.rsplit(' ', 1)[0] + ' -'
        cases.append(Case(f'ep-boundary-{number}', (f'fen {fen} 0 1', f'fen {without} 0 1', f'fen {fen} 0 1')))
    cases.append(Case('native-special-moves', (
        'fen 4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1 moves e5d6',
        'fen 4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1 moves e4d3',
        'fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1g1',
        'fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1c1',
        'fen r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1 moves e8g8',
        'fen r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1 moves e8c8',
        *(f'fen 7k/P7/8/8/8/8/8/7K w - - 0 1 moves a7a8{piece}' for piece in 'qrbn'),
        *(f'fen 7k/8/8/8/8/8/p7/7K b - - 0 1 moves a2a1{piece}' for piece in 'qrbn'),
        'fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
        'fen r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1',
    )))
    return cases


def corpus_cases(corpus: dict[str, Any]) -> list[Case]:
    # Every stored FEN is reconstructed by Bend. Repeat first/last within each
    # bounded lifetime; chunk-local IDs are not claimed to be global identifiers.
    cases = []
    for group in corpus['corpora']:
        records = group['records']
        for start in range(0, len(records), 64):
            positions = tuple('fen ' + row['fen'] for row in records[start:start + 64])
            cases.append(Case(f"corpus-{group['name']}-{start // 64}", (*positions, positions[0], positions[-1])))
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
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    records = []

    def command(argv: list[str], name: str, text: str | None = None, error: str | None = None) -> str:
        env = dict(os.environ, BEND_NO_TELEMETRY='1')
        env.pop('BEND_U64_CORRUPT', None)
        env.pop('DEEPFIN_BOARD_INDEX', None)
        if text is not None:
            env['DEEPFIN_BOARD_INDEX'] = text
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
        (output / f'{name}.stdout').write_text(stdout)
        (output / f'{name}.stderr').write_text(stderr)
        records.append({'name': name, 'exit': code, 'timed_out': timed_out, 'argv': argv})
        (output / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        if timed_out or code != (2 if error else 0) or (stderr != (error + '\n' if error else '')) or (error and stdout):
            raise RuntimeError(f'{name}: exit={code}, timed_out={timed_out}; see saved diagnostics')
        return stdout

    try:
        report['compiler'] = command([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'),
                                      str(args.compiler_root.resolve())], 'compiler')
        report['cc'] = command([args.cc, '--version'], 'cc')
        report['cpu_target'] = cpu_target.qualify(args.cc, output, command)
        cases = fixtures()
        if args.chess:
            from .chess_workloads import collect
            _, _, corpus = collect()
            cases.extend(corpus_cases(corpus))
            report['corpus'] = {**corpus, 'corpora': [
                {k: v for k, v in group.items() if k != 'records'} for group in corpus['corpora']]}
        answers = {c.name: expected(reference(c), c.bits) for c in cases}
        report['cases'] = [{'name': c.name, 'bits': c.bits, 'positions': len(c.positions),
                            'input_sha256': hashlib.sha256(encode(c).encode()).hexdigest(),
                            'expected_sha256': hashlib.sha256(answers[c.name].encode()).hexdigest()} for c in cases]
        report['sources'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (HERE / 'BoardIndex.bend', HERE / 'board_index.bend', HERE / 'board_index.py',
                              HERE / 'PositionIndex.bend', HERE / 'U64Map.bend', HERE / 'cpu_target.py',
                              HERE.parent / 'legal_probe/Chess.bend', HERE.parent / 'standalone/Position.bend',
                              HERE.parent / 'standalone/Protocol.bend', HERE.parent / 'standalone/Tables.bend')}
        compiler = [args.bun, str(args.compiler_root.resolve() / 'bend2/main.ts')]
        generated = output / 'board.c'
        command([*compiler, str(HERE / 'board_index.bend'), '-o', str(generated)], 'generate')
        report['generated_c_sha256'] = hashlib.sha256(generated.read_bytes()).hexdigest()
        for mode, flags in MODES.items():
            binary = output / mode
            command([args.cc, '-std=c11', '-O1', '-ffp-contract=off', *flags, str(generated),
                     '-pthread', '-lm', '-o', str(binary)], 'build-' + mode)
            for case in cases:
                verify(command([str(binary), '--threads', '1'], mode + '-' + case.name, encode(case)), answers[case.name])
            report['modes'].append({'mode': mode, 'cases': len(cases), 'positions': sum(len(c.positions) for c in cases),
                                    'rows': sum(len(answers[c.name].splitlines()) for c in cases), 'flags': flags})
        invalid = (
            ('0;startpos', 'invalid native board-index capacity'),
            ('17;startpos', 'invalid native board-index capacity'),
            ('7;startpos moves e2e5', 'invalid native board-index position'),
            ('7;fen nonsense', 'invalid native board-index position'),
            ('7;fen 4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1 moves e5d6', 'invalid native board-index position'),
            ('7;' + ';'.join(['startpos'] * 129), 'native board-index input exceeds limit'),
        )
        for n, (text, error) in enumerate(invalid):
            command([str(output / 'generic'), '--threads', '1'], f'invalid-{n}', text, error)
        report['invalid_inputs_rejected'] = len(invalid)
        source = (HERE / 'BoardIndex.bend').read_text()
        mutants = {
            'raw-ep': ('ep_checked(b, ep, U32.is_lt(ep, 64))', 'ep', 'history-and-unused-ep'),
            'erase-ep': ('ep_checked(b, ep, U32.is_lt(ep, 64))', '64', 'ep-boundary-0'),
        }
        # A disposable sibling preserves the native module's relative imports.
        for name, (old, new, fixture) in mutants.items():
            import tempfile
            with tempfile.TemporaryDirectory(prefix='board-mutant-', dir=HERE.parent) as temp:
                folder = Path(temp)
                for p in ('U64Map.bend', 'PositionIndex.bend', 'board_index.bend'):
                    shutil.copyfile(HERE / p, folder / p)
                if source.count(old) != 1:
                    raise ValueError('board adapter mutation site changed')
                (folder / 'BoardIndex.bend').write_text(source.replace(old, new))
                generated = output / f'{name}.c'
                command([*compiler, str(folder / 'board_index.bend'), '-o', str(generated)], 'generate-' + name)
                binary = output / name
                command([args.cc, '-std=c11', '-O1', str(generated), '-pthread', '-lm', '-o', str(binary)], 'build-' + name)
                case = next(c for c in cases if c.name == fixture)
                text = command([str(binary), '--threads', '1'], 'mutation-' + name, encode(case))
                try:
                    verify(text, answers[case.name])
                except ValueError:
                    report['mutations_rejected'].append(name)
                else:
                    raise AssertionError('native board adapter mutation survived: ' + name)
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
