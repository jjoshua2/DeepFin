"""Independent castling-subset and destination-safety checks of actual legal_moves.

The candidate receives raw Boards, never oracle-generated decisions. Complete
noncastling move-set validation and source-array/native lifetime equivalence are
outside this opt-in test. All clean modes repeat the same input contexts.
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
MODES = {'generic': [], 'portable': ['-DBEND_U64_PORTABLE'], 'native': ['-march=native'],
         'ubsan': ['-fsanitize=undefined', '-fno-sanitize-recover=all']}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


K = load('accepted_castle_reference', SUITE.parent / 'castle_kings/verify_native.py')
V = load('accepted_castle_validation', SUITE.parent / 'table_preservation/_validation.py')
Ref = K.Ref


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False,
                            env={**os.environ, 'BEND_NO_TELEMETRY': '1', 'TERM': 'dumb'})
    V.require(result.returncode == 0 and not result.stderr,
              repr((cmd[:3], result.returncode, result.stderr[-2000:], result.stdout[-1000:])))
    return result


def fixtures() -> list[dict]:
    out = []
    seen: set[tuple[int, ...]] = set()
    rng = random.Random(0xACCE9726)

    def add(category: str, board: list, side: int, rights: int = 15, ep: int = 64) -> None:
        src = 4 if side else 60
        V.require(K.consistent(board) and K.selected(board, side) == [src], 'fixture king premise')
        meta = (side, rights, ep)
        encoded = Ref.encode(board, meta)
        key = tuple(encoded)
        if key in seen:
            return
        seen.add(key)
        entries, stages = [], []
        for wing in (1, 0):
            mid, dst = ((5 if wing else 3, 6 if wing else 2) if side else
                        (61 if wing else 59, 62 if wing else 58))
            transit, _ = K.move(board, meta, src, mid, False)
            child, cm = K.move(board, meta, src, dst, True)
            values = [int(Ref.attacked(b, sq, 1 - side)) for b, sq in
                      ((board, src), (transit, mid), (child, dst))]
            guard = bool(K.guard(board, meta, wing))
            stages.append({'wing': wing, 'guard': guard, 'checks': values})
            if guard and values == [0, 0, 0]:
                V.require(K.selected(child, side) == [dst], 'fixture final singleton')
                entries.append([src, dst, 0, 2, 0, *Ref.encode(child, cm)])
        expected = ['begin ' + str(len(entries)),
                    *['castle ' + ' '.join(map(str, row)) for row in entries], 'end']
        out.append({'input': encoded, 'category': category, 'stage_checks': stages,
                    'castles': len(entries), 'expected': expected})

    for side in (0, 1):
        home, src = (0, 4) if side else (56, 60)
        def empty_route():
            b = Ref.empty()
            Ref.put(b, src, 5, side)
            Ref.put(b, home, 3, side)
            Ref.put(b, home + 7, 3, side)
            return b
        for rights in range(16):
            add('rights_and_both_routes', empty_route(), side, rights)
        for sq in range(64):
            if sq in (home, src, home + 7):
                continue
            for kind in range(6):
                b = empty_route()
                Ref.put(b, sq, kind, 1 - side)
                add('enemy_origin_kind', b, side)
        for square in (1, 2, 3, 5, 6):
            for kind in range(5):
                b = empty_route()
                Ref.put(b, home + square, kind, side)
                add('own_path_blocker', b, side)
        for _ in range(64):
            b = empty_route()
            for sq in range(64):
                if sq not in (home, src, home + 7) and rng.randrange(6) == 0:
                    Ref.put(b, sq, rng.randrange(5), rng.randrange(2))
            add('mixed_consistent_raw_metadata', b, side, 15 | (rng.getrandbits(28) << 4), rng.getrandbits(32))
    V.require(any(c['castles'] == 2 for c in out), 'missing two-castle cases')
    V.require(any(s['guard'] and s['checks'] == [0, 0, 1] for c in out for s in c['stage_checks']),
              'missing destination-only attack cases')
    return out


class WrongValue(AssertionError):
    """Successful compiled execution disagrees with independent expected output."""


def observe(binary: Path, cases: list[dict], context: int) -> dict:
    texts = []
    for start in range(0, len(cases), 64):
        batch = cases[start:start + 64]
        text = run([str(binary), str(context), *(str(x) for c in batch for x in c['input'])], 120).stdout
        lines = text.splitlines()
        marker = [0, 0] if context == 0 else [305419896, 2596069104]
        expected = [line for c in batch for line in c['expected']] + ['marker ' + ' '.join(map(str, marker))]
        if lines != expected:
            at = next((i for i, (a, b) in enumerate(zip(lines, expected)) if a != b), min(len(lines), len(expected)))
            raise WrongValue(json.dumps({'batch_start': start, 'line': at,
                'expected': expected[at] if at < len(expected) else '<extra>',
                'observed': lines[at] if at < len(lines) else '<missing>'}))
        texts.append(text)
    return {'context': context, 'requests': len(cases), 'castles': sum(c['castles'] for c in cases),
            'check_answers': sum(c['castles'] for c in cases), 'marker_reads': len(texts),
            'output_sha256': digest(''.join(texts).encode())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    V.begin_report(args.report, 'native_gate')
    compiler = args.compiler.resolve()
    bun, cc = os.environ.get('BUN', 'bun'), os.environ.get('CC', 'clang')
    verify = [bun, str(ENGINE / 'standalone/verify_compiler.js'), str(compiler)]
    identity = run(verify).stdout
    cases = fixtures()
    dependencies = [SUITE / 'probe.bend', Path(__file__), SUITE.parent / 'castle_kings/verify_native.py',
                    SUITE.parent / 'attack_witness/verify_native.py', SUITE.parent / 'table_preservation/_validation.py']
    dependencies += [ENGINE / p for p in ['legal_probe/Chess.bend', 'standalone/Tables.bend',
        'standalone/Text.bend', 'bitboard_probe/Sliders.bend', 'standalone/toolchain.json', 'standalone/verify_compiler.js']]
    before = {str(p.relative_to(ENGINE)): digest(p.read_bytes()) for p in dependencies}
    modes, mutations = [], []
    valid = list(map(str, cases[0]['input']))
    invalids = [[], ['bad'], ['2'], ['0', '1'], ['0', *valid[:-1]],
                ['0', *valid[:-3], '2', *valid[-2:]], ['0', '4294967296', *valid[1:]], ['0', *(valid * 65)]]
    with tempfile.TemporaryDirectory(prefix='accepted-castle-native-') as directory:
        work = Path(directory)
        def compile_source(root: Path, name: str) -> Path:
            c = work / (name + '.c')
            run([bun, str(compiler / 'bend2/main.ts'), str(root / 'standalone/proofs/accepted_castle/probe.bend'), '-o', str(c)])
            return c
        def build(c: Path, name: str, flags: list[str]) -> Path:
            binary = work / name
            run([cc, '-std=c11', '-O1', '-ffp-contract=off', '-Werror=shift-count-overflow',
                 *flags, str(c), '-pthread', '-lm', '-o', str(binary)])
            return binary
        clean = compile_source(ENGINE, 'candidate')
        for mode, flags in MODES.items():
            binary = build(clean, mode, flags)
            results = [observe(binary, cases, ctx) for ctx in (0, 1)]
            for argv in invalids:
                bad = subprocess.run([str(binary), *argv], capture_output=True, text=True, timeout=120, check=False)
                V.require(bad.returncode == 2 and 'invalid accepted-castle' in bad.stderr,
                          repr((argv[:3], bad.returncode, bad.stderr)))
            modes.append({'mode': mode, 'observations': results, 'invalid_rejections': len(invalids)})
            print('PASS ' + mode, flush=True)
        edits = [
            ('bypass-destination-rejection', '(table, retain_move(check, m, acc))', '(table, retain_move(False{}, m, acc))'),
            ('omit-kings-from-sensitive-mask', 'sensitive = U64.and(own, U64.or(rays, get_kings(b)))',
             'sensitive = U64.and(own, rays)'),
            ('drop-all-castling', 'retain_move(Bool.or(start_check, transit_check), m, acc)',
             'retain_move(True{}, m, acc)'),
        ]
        copied = work / 'engine'
        shutil.copytree(ENGINE, copied, symlinks=True)
        chess = copied / 'legal_probe/Chess.bend'
        pristine = chess.read_text()
        for name, old, new in edits:
            V.require(pristine.count(old) == 1, 'nonunique mutation ' + name)
            chess.write_text(pristine.replace(old, new))
            binary = build(compile_source(copied, name), name, [])
            try:
                observe(binary, cases, 0)
            except WrongValue as error:
                mutations.append({'name': name, 'compiled_and_executed': True, 'rejected': True,
                                  'build_mode': 'generic', 'diagnostic': str(error)})
            else:
                raise AssertionError('mutation accepted: ' + name)
            print('PASS mutation ' + name, flush=True)
        chess.write_text(pristine)
    V.require(before == {str(p.relative_to(ENGINE)): digest(p.read_bytes()) for p in dependencies}, 'source drift')
    V.require(run(verify).stdout == identity, 'compiler drift')
    selected = sum(c['castles'] for c in cases)
    report = {'native_gate': 'PASS', 'base_requests': len(cases), 'initialization_contexts': 2,
              'requests_per_mode': len(cases) * 2, 'accepted_castles_per_mode': selected * 2,
              'complete_child_board_fields_per_mode': selected * 2 * 19,
              'independent_destination_checks_per_mode': selected * 2,
              'destination_only_rejected_routes': sum(s['guard'] and s['checks'] == [0, 0, 1] for c in cases for s in c['stage_checks']),
              'two_castle_inputs': sum(c['castles'] == 2 for c in cases),
              'categories': dict(Counter(c['category'] for c in cases)),
              'fixture_sha256': digest(json.dumps(cases, sort_keys=True, separators=(',', ':')).encode()),
              'modes': modes, 'mutations': mutations, 'source_sha256s': before,
              'compiler_identity': identity, 'cc': run([cc, '--version']).stdout.splitlines()[0],
              'scope': 'Independent ordered castling-subset and destination check after actual full generation, plus complete castling child Boards and sampled returned-table state. Not independent noncastling legality, full-buffer or lifetime correctness.'}
    V.write_report(args.report, report)


if __name__ == '__main__':
    main()
