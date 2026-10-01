"""Opt-in source qualification of initialized castling control flow and array threading.

Public proofs derive all query certificates from the retained initialized-stage law.
Short source mutants target the new implementation-linked continuation proofs, not
another execution of the large initialization proof per mutant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

LAWS = ['initialized_castle_producer_matches_geometry',
        'initialized_filtered_castle_matches_three_stage_geometry']
SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code(path: Path) -> str:
    return '\n'.join(line.split('#', 1)[0] for line in path.read_text().splitlines())


def source_closure(entry: Path, engine: Path) -> dict[str, str]:
    """Normalize parent components without resolving away a symlink before checking it."""
    seen: dict[str, str] = {}
    engine = engine.resolve(strict=True)

    def visit(path: Path) -> None:
        lexical = Path(os.path.abspath(path))
        resolved = lexical.resolve(strict=True)
        if lexical != resolved or lexical.is_symlink() or not lexical.is_file():
            raise ValueError(f'Symlinked or nonregular proof input: {lexical}')
        try:
            rel = lexical.relative_to(engine).as_posix()
        except ValueError as exc:
            raise ValueError(f'Escaped engine: {lexical}') from exc
        if not (rel.startswith('standalone/') or rel in {'legal_probe/Chess.bend', 'bitboard_probe/Sliders.bend'}):
            raise ValueError(f'Out-of-scope input: {rel}')
        if rel in seen:
            return
        raw = lexical.read_bytes()
        seen[rel] = digest(raw)
        text = code(lexical)
        if '@unsafe' in text or '?' in text:
            raise ValueError(f'Unsafe dependency or proof hole: {rel}')
        for name in re.findall(r'^\s*import\s+(\S+)', text, re.MULTILINE):
            if name == 'Base':
                continue
            if not re.fullmatch(r'\.{1,2}/[A-Za-z0-9_/.]+\.bend', name):
                raise ValueError(f'Foreign import: {name}')
            visit(lexical.parent / name)

    visit(entry)
    return dict(sorted(seen.items()))


def manifest(suite: Path) -> None:
    laws = re.findall(r'^law (\w+):', code(suite / 'LAWS.bend'), re.MULTILINE)
    proof = code(suite / 'PROOF.bend')
    bodies = re.findall(r'^def Laws\.(\w+)\(', proof, re.MULTILINE)
    if laws != LAWS or bodies != LAWS:
        raise ValueError('Missing, duplicate or unexpected public law/proof')
    if 'import ./LAWS.bend as Laws' not in proof or 'import ../castle_stage_geometry/PROOF.bend as StageProof' not in proof:
        raise ValueError('Missing declared law or initialized producer body import')
    if 'import ./PROOF.bend as Proof' not in code(suite / 'consumer.bend'):
        raise ValueError('Consumer omits public proof bodies')


def invoke(bun: str, compiler: Path, entry: Path, timeout: int, logfile: Path) -> dict:
    start = time.monotonic()
    command = [bun, *(['--smol'] if os.environ.get('CASTLE_PROOF_SMOL') == '1' else []), str(compiler / 'bend2/main.ts'), str(entry)]
    with logfile.open('wb') as output:
        try:
            proc = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False,
                                  env={**os.environ, 'TERM': 'dumb', 'BEND_NO_TELEMETRY': '1'})
            status: int | str = proc.returncode
        except subprocess.TimeoutExpired:
            status = 'TIMEOUT'
    raw = logfile.read_bytes()
    return {'command': command, 'exit_code': status, 'seconds': time.monotonic() - start,
            'sha256': digest(raw), 'raw_text': raw.decode()}


def safe(result: dict) -> None:
    if result['exit_code'] != 0 or result['raw_text'].strip() != 'All terms check.':
        raise AssertionError(str(result)[-6000:])


def replace(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if source.count(old) != 1:
        raise AssertionError(f'Mutation site not unique: {path}: {old}')
    path.write_text(source.replace(old, new))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--consumer-timeout', type=int, default=1800)
    parser.add_argument('--smol', action='store_true', help='Run the unchanged compiler with Bun smaller-heap mode')
    args = parser.parse_args()
    if args.consumer_timeout < 1:
        parser.error('consumer timeout must be positive')
    if args.smol:
        os.environ['CASTLE_PROOF_SMOL'] = '1'
    compiler = args.compiler.resolve(strict=True)
    bun = os.environ.get('BUN', 'bun')
    pin_command = [bun, str(ENGINE / 'standalone/verify_compiler.js'), str(compiler)]
    pin = subprocess.run(pin_command, capture_output=True, text=True, timeout=60, check=True)
    if pin.stderr:
        raise AssertionError(pin.stderr)
    manifest(SUITE)
    before = source_closure(SUITE / 'consumer.bend', ENGINE)
    # Cover the executable adapter and actual native dependencies as well.
    before.update(source_closure(SUITE / 'probe.bend', ENGINE))
    for path in [SUITE / 'focused.py', SUITE / 'verify_native.py', SUITE / 'README.md',
                 SUITE.parent / 'castle_kings/verify_native.py', SUITE.parent / 'attack_witness/verify_native.py',
                 ENGINE / 'standalone/toolchain.json', ENGINE / 'standalone/verify_compiler.js']:
        before[path.relative_to(ENGINE).as_posix()] = digest(path.read_bytes())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    logs = args.report.parent / (args.report.stem + '-logs')
    logs.mkdir(exist_ok=False)
    controls: list[dict] = []
    consumer = invoke(bun, compiler, SUITE / 'consumer.bend', args.consumer_timeout, logs / 'consumer.log')
    safe(consumer)
    print('PASS complete initialized consumer', flush=True)
    mutations = [
        ('ignore-start-check', 'legal_probe/Chess.bend',
         'retain_move(Bool.or(start_check, transit_check), m, acc)', 'retain_move(transit_check, m, acc)', 'finish'),
        ('ignore-transit-check', 'legal_probe/Chess.bend',
         'retain_move(Bool.or(start_check, transit_check), m, acc)', 'retain_move(start_check, m, acc)', 'finish'),
        ('check-unmoved-transit-board', 'legal_probe/Chess.bend',
         'in_check(table, make_move(b, Ply{src, transit, 0, 0}), get_turn(b))', 'in_check(table, b, get_turn(b))', 'producer'),
        ('transit-checks-flipped-side', 'legal_probe/Chess.bend',
         'in_check(table, make_move(b, Ply{src, transit, 0, 0}), get_turn(b))',
         'in_check(table, make_move(b, Ply{src, transit, 0, 0}), U32.xor(get_turn(b), 1))', 'producer'),
        ('final-checks-flipped-side', 'legal_probe/Chess.bend',
         'filter_after(m, acc, in_check(table, make_move(b, m), get_turn(b)))',
         'filter_after(m, acc, in_check(table, make_move(b, m), U32.xor(get_turn(b), 1)))', 'filter_one'),
        ('final-ignores-check-result', 'legal_probe/Chess.bend',
         '(table, retain_move(check, m, acc))', '(table, retain_move(False{}, m, acc))', 'after'),
        ('erase-returned-table-slot', 'legal_probe/Chess.bend',
         '(table, retain_move(Bool.or(start_check, transit_check), m, acc))',
         '(Array.set(U64,table,131071,U64.zero()), retain_move(Bool.or(start_check, transit_check), m, acc))', 'finish'),
        ('adapter-bypasses-final-filter', 'standalone/proofs/castle_sequence/Runtime.bend',
         'Chess.filter_legal(moves,b,(table,Nil{}))', '(table,moves)', 'filter_one'),
    ]
    for name, target, old, new, location in mutations:
        with tempfile.TemporaryDirectory(prefix='castle-sequence-control-') as directory:
            copied = Path(directory) / 'engine'
            shutil.copytree(ENGINE, copied, symlinks=True)
            replace(copied / target, old, new)
            entry = copied / 'standalone/proofs/castle_sequence/Wire.bend'
            result = invoke(bun, compiler, entry, 180, logs / (name + '.log'))
            output = result['raw_text']
            forbidden = ['no such file', 'RangeError', 'Maximum call stack', 'more than once',
                         'a decreasing self-call', 'a defined name', 'a pattern (a binder']
            if not (result['exit_code'] == 1 and re.search(r'expected[\s\S]*observed', output)
                    and re.search(r'Location:\s*' + re.escape(location) + r'\b', output)
                    and not any(s in output for s in forbidden)):
                raise AssertionError(f'Not intended refinement rejection {name}: {result}')
            controls.append({'name': name, 'kind': 'source semantic/refinement', 'rejected': True,
                             'entry': 'Wire.bend', 'mutation_target': target, **result})
            print('PASS semantic control ' + name, flush=True)
    policies = [
        ('missing-law', 'LAWS.bend', 'law ' + LAWS[0] + ':', 'def missing:'),
        ('missing-proof', 'PROOF.bend', 'def Laws.' + LAWS[0] + '(', 'def missing('),
        ('missing-law-import', 'PROOF.bend', 'import ./LAWS.bend as Laws', ''),
        ('missing-consumer-body-import', 'consumer.bend', 'import ./PROOF.bend as Proof', ''),
        ('proof-hole', 'Wire.bend', None, '\n?missing\n'),
        ('unsafe-proof', 'Wire.bend', None, '\n@unsafe\n'),
        ('foreign-proof', 'Wire.bend', None, '\nimport "./oracle.c"\n'),
        ('symlink-proof', 'Wire.bend', None, None),
    ]
    for name, filename, old, new in policies:
        with tempfile.TemporaryDirectory(prefix='castle-sequence-policy-') as directory:
            copied = Path(directory) / 'engine'
            shutil.copytree(ENGINE, copied, symlinks=True)
            suite = copied / 'standalone/proofs/castle_sequence'
            target = suite / filename
            if name == 'symlink-proof':
                target.rename(suite / 'Wire.orig')
                target.symlink_to('Wire.orig')
            elif old is not None:
                replace(target, old, new)
            else:
                target.write_text(target.read_text() + new)
            try:
                manifest(suite)
                source_closure(suite / 'consumer.bend', copied)
            except (ValueError, FileNotFoundError) as error:
                controls.append({'name': name, 'kind': 'manifest/import policy', 'rejected': True, 'message': str(error)})
            else:
                raise AssertionError('Policy corruption accepted: ' + name)
    try:
        safe({'exit_code': 0, 'raw_text': 'All terms check.\nWARNING: unsafe dependency'})
    except AssertionError:
        controls.append({'name': 'warning-on-zero-exit', 'kind': 'synthetic output-wrapper unit', 'rejected': True})
    else:
        raise AssertionError('Unsafe success output accepted')
    assert len(controls) == 17
    for name, wanted in before.items():
        if digest((ENGINE / name).read_bytes()) != wanted:
            raise AssertionError('Source drift: ' + name)
    if subprocess.run(pin_command, capture_output=True, text=True, timeout=60, check=True).stdout != pin.stdout:
        raise AssertionError('Compiler drift')
    report = {'focused_gate': 'PASS', 'consumer': consumer, 'new_law_count': 2, 'new_laws': LAWS,
              'new_control_count': 17, 'negative_controls': controls,
              'compiler_identity': pin.stdout, 'source_sha256s': dict(sorted(before.items())),
              'scope': 'Guarded initialized actual single-producer and its full-filter sequential behavior; not full optimized legal_moves or universal forward/reverse attack semantics.'}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS 2 public laws and 17 classified controls', flush=True)


if __name__ == '__main__':
    main()
