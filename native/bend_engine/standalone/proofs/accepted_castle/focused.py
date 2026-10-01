"""Opt-in complete accepted-castle consumer and classified rejection controls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
NAMES = ['prepared_castle_check_returns_false', 'generated_castle_check_returns_false',
         'initialized_generated_castle_destination_is_unattacked']


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Base = load('accepted_castle_host_support', SUITE.parent / 'table_preservation/focused.py')
V = load('accepted_castle_validation', SUITE.parent / 'table_preservation/_validation.py')


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def manifest(suite: Path) -> None:
    laws = re.findall(r'^law (\w+):', (suite / 'LAWS.bend').read_text(), re.MULTILINE)
    proof = (suite / 'PROOF.bend').read_text()
    bodies = re.findall(r'^def Laws\.(\w+)\(', proof, re.MULTILINE)
    if laws != NAMES or bodies != NAMES:
        raise ValueError('law/proof inventory')
    if 'import ./LAWS.bend as Laws' not in proof:
        raise ValueError('missing law import')
    if 'import ./PROOF.bend as Proof' not in (suite / 'consumer.bend').read_text():
        raise ValueError('missing proof import')


def invoke(bun: str, compiler: Path, entry: Path, timeout: int = 180) -> dict:
    started = time.monotonic()
    result = subprocess.run([bun, '--smol', str(compiler / 'bend2/main.ts'), str(entry)],
        capture_output=True, timeout=timeout, check=False,
        env={**os.environ, 'TERM': 'dumb', 'BEND_NO_TELEMETRY': '1'})
    raw = result.stdout + result.stderr
    return {'exit_code': result.returncode, 'raw_text': raw.decode(), 'sha256': digest(raw),
            'seconds': time.monotonic() - started}


def safe(result: dict) -> None:
    V.require(result['exit_code'] == 0 and result['raw_text'].strip() == 'All terms check.',
              repr(result)[-5000:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--controls-only', action='store_true', help='Do not claim a law/consumer pass')
    args = parser.parse_args()
    V.begin_report(args.report, 'focused_gate')
    compiler = args.compiler.resolve()
    bun = os.environ.get('BUN', 'bun')
    pin = [bun, str(ENGINE / 'standalone/verify_compiler.js'), str(compiler)]
    identity = subprocess.run(pin, capture_output=True, text=True, check=True)
    V.require(not identity.stderr, 'compiler identity warning')
    manifest(SUITE)
    closure = Base.closure(SUITE / 'consumer.bend', ENGINE)
    paths = [ENGINE / p for p in closure] + [p for p in SUITE.iterdir() if p.is_file()]
    paths += [SUITE.parent / p for p in ('table_preservation/focused.py', 'table_preservation/_validation.py',
                                        'castle_kings/verify_native.py', 'attack_witness/verify_native.py')]
    paths += [ENGINE / p for p in ('standalone/toolchain.json', 'standalone/verify_compiler.js')]
    before = {str(p.relative_to(ENGINE)): digest(p.read_bytes()) for p in paths}
    consumer = None
    if not args.controls_only:
        consumer = invoke(bun, compiler, SUITE / 'consumer.bend', 1800)
        safe(consumer)
        print('PASS complete consumer', flush=True)
    edits = [
        ('destination-rejection-bypassed', 'legal_probe/Chess.bend', 'Filter.bend', 'after',
         '(table, retain_move(check, m, acc))', '(table, retain_move(False{}, m, acc))'),
        ('wrong-checked-side', 'legal_probe/Chess.bend', 'Filter.bend', 'step',
         'filter_after(m, acc, in_check(table, make_move(b, m), get_turn(b)))',
         'filter_after(m, acc, in_check(table, make_move(b, m), U32.xor(get_turn(b),1)))'),
        ('required-step-skipped', 'legal_probe/Chess.bend', 'Filter.bend', 'fast_step',
         'case True{}: filter_step(b, m, r)', 'case True{}: r'),
        ('unchecked-accumulator-injected', 'legal_probe/Chess.bend', 'Filter.bend', 'blockers',
         'filter_fast(moves, b, sensitive, (table, Nil{}))',
         'filter_fast(moves, b, sensitive, (table, Con{Ply{4,6,0,2},Nil{}}))'),
        ('required-premise-omitted', 'standalone/proofs/accepted_castle/Spec.bend', 'Accept.bend', 'fast_insert',
         '{Chess.filter_requires(sensitive,m) == True{} : Bool} -> safe(c,b,m)',
         '{True{} == True{} : Bool} -> safe(c,b,m)'),
        ('generated-membership-omitted', 'standalone/proofs/accepted_castle/Public.bend', 'Public.bend', 'legal',
         '  here: E.member(m,E.moves(Chess.legal_moves(a,b))),tag: {C.flag(m) == 2 : U32}) ->',
         '  here: Unit,tag: {C.flag(m) == 2 : U32}) ->'),
        ('opponent-check-substituted', 'standalone/proofs/accepted_castle/Destination.bend', 'Destination.bend', 'decide',
         '{S.answer(Chess.in_check(a,K.final(b,white,ks),Bool.to_u32(white))) == False{} : Bool}:',
         '{S.answer(Chess.in_check(a,K.final(b,white,ks),Bool.to_u32(Bool.not(white)))) == False{} : Bool}:'),
        ('false-geometric-conclusion', 'standalone/proofs/accepted_castle/Finish.bend', 'Finish.bend', 'stage',
         '{G.attacked(K.final(b,white,ks),U32.to_nat(K.dst(white,ks)),Bool.not(white)) == False{} : Bool}:',
         '{G.attacked(K.final(b,white,ks),U32.to_nat(K.dst(white,ks)),Bool.not(white)) == True{} : Bool}:'),
    ]
    controls = []
    for name, target, entry, location, old, new in edits:
        with tempfile.TemporaryDirectory(prefix='accepted-castle-rejection-') as directory:
            copied = Path(directory) / 'engine'
            shutil.copytree(ENGINE, copied, symlinks=True)
            Base.replace(copied / target, old, new)
            result = invoke(bun, compiler, copied / 'standalone/proofs/accepted_castle' / entry)
            output = result['raw_text']
            forbidden = ['no such file', 'RangeError', 'Maximum call stack', 'more than once',
                         'a decreasing self-call', 'a defined name', 'a parameter or field scrutinee']
            V.require(result['exit_code'] == 1 and 'expected' in output and 'observed' in output and
                      bool(re.search(r'Location:\s*(?:[\w./]+\.)*' + location + r'\b', output)) and
                      not any(s in output for s in forbidden),
                      'Not an intended semantic rejection ' + name + ': ' + repr(result)[-5000:])
            controls.append({'name': name, 'kind': 'source semantic/refinement', 'rejected': True,
                             'entry': entry, 'diagnostic_sha256': result['sha256'], 'excerpt': output[-2200:]})
            print('PASS semantic ' + name, flush=True)
    policies = [
        ('missing-law', lambda s: Base.replace(s / 'LAWS.bend', 'law generated_castle_check_returns_false:', 'def missing:')),
        ('missing-proof', lambda s: Base.replace(s / 'PROOF.bend', 'def Laws.generated_castle_check_returns_false(', 'def missing(')),
        ('missing-law-import', lambda s: Base.replace(s / 'PROOF.bend', 'import ./LAWS.bend as Laws\n', '')),
        ('missing-proof-import', lambda s: Base.replace(s / 'consumer.bend', 'import ./PROOF.bend as Proof\n', '')),
        ('proof-hole', lambda s: (s / 'Generated.bend').write_text((s / 'Generated.bend').read_text() + '\n?missing\n')),
        ('unsafe', lambda s: (s / 'Generated.bend').write_text('@unsafe\n' + (s / 'Generated.bend').read_text())),
        ('foreign', lambda s: (s / 'Generated.bend').write_text((s / 'Generated.bend').read_text() + '\nimport "oracle.c"\n')),
        ('symlink', lambda s: ((s / 'Spec.bend').rename(s / 'Spec.original'), (s / 'Spec.bend').symlink_to('Spec.original'))),
    ]
    for name, edit in policies:
        with tempfile.TemporaryDirectory(prefix='accepted-castle-policy-') as directory:
            copied = Path(directory) / 'engine'
            shutil.copytree(ENGINE, copied, symlinks=True)
            suite = copied / 'standalone/proofs/accepted_castle'
            edit(suite)
            try:
                manifest(suite)
                Base.closure(suite / 'consumer.bend', copied)
            except (ValueError, OSError):
                controls.append({'name': name, 'kind': 'manifest/import policy', 'rejected': True})
            else:
                raise AssertionError('policy accepted ' + name)
    try:
        safe({'exit_code': 0, 'raw_text': 'All terms check.\nWARNING: unsafe dependency'})
    except AssertionError:
        controls.append({'name': 'warning-on-zero-exit', 'kind': 'synthetic output-wrapper unit', 'rejected': True})
    else:
        raise AssertionError('warning accepted')
    V.require(before == {str(p.relative_to(ENGINE)): digest(p.read_bytes()) for p in paths}, 'source drift')
    final_identity = subprocess.run(pin, capture_output=True, text=True, check=True)
    V.require(final_identity.stdout == identity.stdout and not final_identity.stderr, 'compiler drift')
    V.require(len(controls) == 17, 'control inventory')
    report = {'focused_gate': 'CONTROLS_ONLY_PASS' if args.controls_only else 'PASS',
              'consumer': consumer, 'new_law_count': 0 if args.controls_only else len(NAMES),
              'new_laws': NAMES, 'new_control_count': len(controls), 'negative_controls': controls,
              'control_categories': dict(Counter(c['kind'] for c in controls)),
              'source_sha256s': before, 'compiler_identity': identity.stdout,
              'scope': 'Actual full-generator castling destination rejection and initialized target-centred geometric safety. Not full all-stage/noncastling legality or completeness.'}
    V.write_report(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {'consumer', 'negative_controls', 'source_sha256s'}}, indent=2))


if __name__ == '__main__':
    main()
