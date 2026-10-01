"""Actual castling producer/filter sequence versus independent forward attacks.

Inputs are raw Boards, side-of-castling and arbitrary predefined list tails, never
expected decisions. Candidate Tables/Chess operations thread one real array through
queries; two unused-slot reads per request observe (not prove) retained state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
REFERENCE = SUITE.parent / 'castle_kings/verify_native.py'
_spec = importlib.util.spec_from_file_location('castling_coordinate_reference', REFERENCE)
if _spec is None or _spec.loader is None:
    raise ImportError(str(REFERENCE))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)
Ref = K.Ref
MARKER = [826366246, 1398314899]
TAILS = [[], [(17, 25, 0, 0)], [(4, 6, 0, 2), (17, 25, 0, 0)], [(60, 58, 0, 2), (4, 6, 0, 2)]]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(args: list[str], timeout: int = 240) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False,
                            env={**os.environ, 'BEND_NO_TELEMETRY': '1', 'TERM': 'dumb'})
    if result.returncode != 0 or result.stderr:
        raise AssertionError((args[:3], result.returncode, result.stderr[-2000:], result.stdout[-1000:]))
    return result


class WrongValue(AssertionError):
    """Successful execution produced an incorrect list, Board, or retained marker."""


def fixtures() -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    rng = random.Random(0x5E0C2026)

    def add(category: str, board: list, side: int, wing: int, mode: int, rights: int = 15, ep: int = 64) -> None:
        meta = (side, rights, ep)
        src, mid, dst = ((4, 5 if wing else 3, 6 if wing else 2) if side else
                         (60, 61 if wing else 59, 62 if wing else 58))
        # Every baseline observation of in_check is inside its singleton-side domain.
        assert K.consistent(board) and K.selected(board, side) == [src]
        key = tuple([wing, mode, *Ref.encode(board, meta)])
        if key in seen:
            return
        seen.add(key)
        path_guard = K.guard(board, meta, wing)
        transit, tm = K.move(board, meta, src, mid, False)
        final, fm = K.move(board, meta, src, dst, True)
        checks = [int(Ref.attacked(b, sq, 1 - side)) for b, sq in [(board, src), (transit, mid), (final, dst)]]
        if path_guard:
            assert K.selected(transit, side) == [mid] and K.selected(final, side) == [dst]
        move = (src, dst, 0, 2)
        producer = ([move] if path_guard and checks[:2] == [0, 0] else []) + TAILS[mode]
        filtered = [move] if path_guard and checks == [0, 0, 0] else []
        expected: list[str] = []
        for label, entries in [('producer-state', producer), ('filtered-state', filtered)]:
            expected.append('begin ' + str(len(entries)))
            for s, d, p, f in entries:
                child, cm = K.move(board, meta, s, d, f == 2)
                expected.append('move ' + ' '.join(map(str, [s, d, p, f, *Ref.encode(child, cm)])))
            expected += ['end', label + ' ' + ' '.join(map(str, MARKER))]
        out.append({'category': category, 'input': list(key), 'guard': path_guard,
                    'geometric_checks': checks, 'emitted': len(producer) - len(TAILS[mode]),
                    'accepted': len(filtered), 'tail_length': len(TAILS[mode]),
                    'board_records': len(producer) + len(filtered), 'expected': expected})

    for side in (0, 1):
        home, src = (0, 4) if side else (56, 60)
        for wing in (0, 1):
            rook = home + (7 if wing else 0)
            for mode in range(4):
                b = Ref.empty()
                Ref.put(b, src, 5, side)
                Ref.put(b, rook, 3, side)
                add('safe_tail_shapes', b, side, wing, mode)
                add('false_rights_tail_preserved', b, side, wing, mode, rights=0)
            for sq in range(64):
                if sq in (src, rook):
                    continue
                for kind in range(6):
                    b = Ref.empty()
                    Ref.put(b, src, 5, side)
                    Ref.put(b, rook, 3, side)
                    Ref.put(b, sq, kind, 1 - side)
                    add('enemy_piece_geometry', b, side, wing, (sq + kind) % 4)
            for i in range(64):
                b = Ref.empty()
                for sq in range(64):
                    if sq not in (src, rook) and rng.randrange(5) == 0:
                        Ref.put(b, sq, rng.randrange(5), rng.randrange(2))
                Ref.put(b, src, 5, side)
                Ref.put(b, rook, 3, side)
                add('mixed_board_metadata', b, side, wing, i % 4, 15 | (rng.getrandbits(28) << 4), rng.getrandbits(32))
    assert len(out) == len(seen)
    # Require genuinely different safe/start/transit/destination outcomes.
    for checks in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]):
        assert any(c['guard'] and c['geometric_checks'] == checks for c in out), checks
    assert any(not c['guard'] and c['tail_length'] for c in out)
    return out


def observe(binary: Path, cases: list[dict], allow_mismatch: bool = False) -> dict:
    texts: list[str] = []
    wrong: list[dict] = []
    for start in range(0, len(cases), 64):
        batch = cases[start:start + 64]
        text = run([str(binary), *(str(v) for case in batch for v in case['input'])], 120).stdout
        lines = text.splitlines()
        want = [line for c in batch for line in c['expected']]
        if lines != want:
            at = next((i for i, pair in enumerate(zip(lines, want)) if pair[0] != pair[1]), min(len(lines), len(want)))
            item = {'batch_start': start, 'line': at,
                    'observed': lines[at] if at < len(lines) else '<missing>',
                    'expected': want[at] if at < len(want) else '<extra>',
                    'only_markers_differ': len(lines) == len(want) and all(
                        a == b or (a.split()[0] in {'producer-state', 'filtered-state'} and a.split()[0] == b.split()[0])
                        for a, b in zip(lines, want))}
            if not allow_mismatch:
                raise WrongValue(json.dumps(item))
            wrong.append(item)
        texts.append(text)
    return {'output_sha256': digest(''.join(texts).encode()), 'mismatches': wrong}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    bun, cc = os.environ.get('BUN', 'bun'), os.environ.get('CC', 'clang')
    verify = [bun, str(ENGINE / 'standalone/verify_compiler.js'), str(compiler)]
    identity = run(verify).stdout
    paths = [p for p in SUITE.iterdir() if p.is_file()] + [REFERENCE, SUITE.parent / 'attack_witness/verify_native.py']
    paths += [ENGINE / p for p in ['legal_probe/Chess.bend', 'standalone/Tables.bend', 'standalone/Text.bend',
                                   'bitboard_probe/Sliders.bend', 'standalone/toolchain.json', 'standalone/verify_compiler.js']]
    hashes = {str(p.relative_to(ENGINE)): digest(p.read_bytes()) for p in paths}
    cases = fixtures()
    base = list(map(str, cases[0]['input']))
    malformed = []
    for index, value in [(0, '2'), (1, '4'), (2, '-1'), (2, '4294967296'), (2, 'x')]:
        row = base.copy()
        row[index] = value
        malformed.append(row)
    malformed += [base[:-1], base + ['0'], base * 65]
    with tempfile.TemporaryDirectory(prefix='castle-sequence-native-') as directory:
        work = Path(directory)
        generated = work / 'sequence.c'
        run([bun, str(compiler / 'bend2/main.ts'), str(SUITE / 'probe.bend'), '-o', str(generated)])
        modes = []
        for mode, flags in Ref.MODES.items():
            binary = work / mode
            run([cc, '-std=c11', '-O1', '-ffp-contract=off', '-Werror=shift-count-overflow', *flags,
                 str(generated), '-pthread', '-lm', '-o', str(binary)])
            result = observe(binary, cases)
            for row in malformed:
                invalid = subprocess.run([str(binary), *row], capture_output=True, text=True, timeout=90, check=False)
                assert invalid.returncode == 2 and 'invalid sequence' in invalid.stderr, (invalid.returncode, invalid.stderr)
            modes.append({'mode': mode, 'requests': len(cases), 'output_sha256': result['output_sha256'],
                          'invalid_rejections': len(malformed), 'marker_reads': 2 * len(cases)})
            print(f'PASS {mode}: {len(cases)} sequential requests and {2 * len(cases)} marker reads', flush=True)
        mutations = []
        copied = work / 'engine'
        shutil.copytree(ENGINE, copied, symlinks=True)
        chess = copied / 'legal_probe/Chess.bend'
        pristine = chess.read_text()
        edits = [
            ('ignore-start-check', 'retain_move(Bool.or(start_check, transit_check), m, acc)', 'retain_move(transit_check, m, acc)'),
            ('ignore-transit-check', 'retain_move(Bool.or(start_check, transit_check), m, acc)', 'retain_move(start_check, m, acc)'),
            ('erase-only-returned-marker', '(table, retain_move(Bool.or(start_check, transit_check), m, acc))',
             '(Array.set(U64,table,131071,U64.zero()), retain_move(Bool.or(start_check, transit_check), m, acc))'),
        ]
        for name, old, new in edits:
            assert pristine.count(old) == 1, (name, pristine.count(old))
            chess.write_text(pristine.replace(old, new))
            mutated_c = work / (name + '.c')
            run([bun, str(compiler / 'bend2/main.ts'), str(copied / 'standalone/proofs/castle_sequence/probe.bend'), '-o', str(mutated_c)])
            binary = work / name
            run([cc, '-std=c11', '-O1', '-ffp-contract=off', '-Werror=shift-count-overflow', str(mutated_c), '-pthread', '-lm', '-o', str(binary)])
            result = observe(binary, cases, allow_mismatch=True)
            assert result['mismatches'], name
            if name == 'erase-only-returned-marker':
                assert all(m['only_markers_differ'] for m in result['mismatches']), 'query answers must remain correct'
            mutations.append({'name': name, 'compiled_and_executed': True, 'rejected': True,
                              'first_mismatch': result['mismatches'][0], 'affected_batches': len(result['mismatches']),
                              'all_move_lists_and_board_values_match': name == 'erase-only-returned-marker'})
            print(f'PASS actual mutation {name}: rejected on values after compilation and execution', flush=True)
        chess.write_text(pristine)
    assert {p: digest((ENGINE / p).read_bytes()) for p in hashes} == hashes
    assert run(verify).stdout == identity
    guard_cases = [c for c in cases if c['guard']]
    report = {'native_gate': 'PASS', 'compiler_identity': identity, 'cc': run([cc, '--version']).stdout,
              'requests_per_mode': len(cases), 'distinct_inputs': len({tuple(c['input']) for c in cases}),
              'producer_emissions': sum(c['emitted'] for c in cases), 'filtered_acceptances': sum(c['accepted'] for c in cases),
              'complete_child_boards_per_mode': sum(c['board_records'] for c in cases),
              'board_fields_per_mode': 19 * sum(c['board_records'] for c in cases),
              'guard_true_cases': len(guard_cases), 'guarded_check_patterns': dict(Counter(''.join(map(str, c['geometric_checks'])) for c in guard_cases)),
              'categories': dict(Counter(c['category'] for c in cases)),
              'fixture_sha256': digest(json.dumps(cases, sort_keys=True, separators=(',', ':')).encode()),
              'modes': modes, 'mutations': mutations, 'source_sha256s': hashes,
              'scope': 'One actual castle_side and actual filter_legal of its initially empty candidate list; not full/optimized legal_moves, forward/reverse source geometry proof or native full-buffer/lifetime correctness. Four modes repeat fixtures; mutations generic only.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
