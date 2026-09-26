"""Compiler ownership regressions and an owning-payload caller; no engine integration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any, NamedTuple

HERE = Path(__file__).resolve().parent
MAX_ID = (1 << 32) - 1
FIRST_IDS = (0, 7, MAX_ID - 1, MAX_ID)
DEPENDENCIES = ('U64Map.bend', 'InsertOnce.bend', 'IdRegistry.bend')


class Misuse(NamedTuple):
    name: str
    diagnostic: str
    valid: str
    old: str
    new: str


def consumed(binder: str) -> str:
    return (f'Error:\n- expected : {binder}\n'
            f'- observed : {binder} (consumed more than once)\nLocation: exercise\n')


TYPE_ERROR = 'Error:\n- expected : Data\n- observed : Type\nLocation: exercise\n'

# Every invalid program differs from a checked positive control at one use site.
MISUSES = (
    Misuse(
        'double-commit', consumed('q'),
        'def exercise(q: R.Reservation, other: R.Reservation) -> (R.Registry & R.Outcome) & (R.Registry & R.Outcome):\n'
        '  (R.commit(q), R.commit(other))',
        'R.commit(other)', 'R.commit(q)',
    ),
    Misuse(
        'commit-abort', consumed('q'),
        'def exercise(q: R.Reservation, other: R.Registry) -> (R.Registry & R.Outcome) & R.Registry:\n'
        '  (R.commit(q), other)',
        'other)', 'R.abort(q))',
    ),
    Misuse(
        'abort-commit', consumed('q'),
        'def exercise(q: R.Reservation, other: R.Reservation) -> R.Registry & (R.Registry & R.Outcome):\n'
        '  (R.abort(q), R.commit(other))',
        'R.commit(other)', 'R.commit(q)',
    ),
    Misuse(
        'candidate-alias', consumed('q'),
        'def exercise(q: R.Reservation, other: R.Registry) -> (R.Reservation & U32) & R.Registry:\n'
        '  (R.candidate(q), other)',
        'other)', 'R.abort(q))',
    ),
    Misuse(
        'registry-alias', consumed('r'),
        'def exercise(r: R.Registry, other: R.Registry) -> R.Preparation & R.Registry:\n'
        '  (R.reserve(r, U64.zero()), other)',
        'other)', 'r)',
    ),
    Misuse(
        'two-reservations', consumed('r'),
        'def exercise(r: R.Registry, other: R.Registry) -> R.Preparation & R.Preparation:\n'
        '  (R.reserve(r, U64.zero()), R.reserve(other, U64{1, 0}))',
        'R.reserve(other,', 'R.reserve(r,',
    ),
    Misuse(
        'owned-stage-alias', consumed('s'),
        'type Stage is Type:\n'
        '  Stage{reservation: R.Reservation, payload: Array<U64>}\n'
        '\n'
        'def exercise(s: Stage, other: Stage) -> Stage & Stage:\n'
        '  (s, other)',
        '(s, other)', '(s, s)',
    ),
    Misuse(
        'captured-reservation', consumed('q'),
        'def exercise(q: R.Reservation, other: R.Registry) -> (Unit -> R.Registry) & R.Registry:\n'
        '  (ignored => R.abort(q), other)',
        'other)', 'R.abort(q))',
    ),
    Misuse(
        'copy-annotation', TYPE_ERROR,
        'def exercise(q: R.Reservation) -> R.Registry & R.Outcome:\n'
        '  R.commit(q)',
        'exercise(q:', 'exercise(+q:',
    ),
)

VALID_ONLY = {
    'branch-exclusive': (
        'def exercise(q: R.Reservation, b: Bool) -> R.Registry:\n'
        '  match b:\n'
        '    case True{}: R.abort(q)\n'
        '    case False{}: R.abort(q)'
    ),
    'candidate-returned-owner': (
        'def returned(r: R.Reservation & U32) -> R.Registry:\n'
        '  (q, id) = r\n'
        '  R.abort(q)\n'
        '\n'
        'def exercise(q: R.Reservation) -> R.Registry:\n'
        '  returned(R.candidate(q))'
    ),
    'affine-drop': (
        'def exercise(q: R.Reservation) -> U32:\n'
        '  0'
    ),
}


def source(body: str) -> str:
    return 'import Base\nimport ./IdRegistry.bend as R\n\n' + body + '\n\ndef main() -> U32:\n  0\n'


def program(case: Misuse, *, invalid: bool) -> str:
    if case.valid.count(case.old) != 1:
        raise ValueError('ownership use site changed')
    return source(case.valid.replace(case.old, case.new) if invalid else case.valid)


def check_compilation(result: subprocess.CompletedProcess[str], expected: str | None) -> None:
    if expected is None:
        if result.returncode != 0 or result.stdout != 'All terms check.\n' or result.stderr:
            raise ValueError('valid ownership program did not check cleanly')
        return
    if (result.returncode != 1 or result.stdout or not result.stderr.startswith(expected)
            or result.stderr.count('Error:') != 1):
        raise ValueError('misuse did not fail with the expected ownership diagnostic')


def expected_payload() -> str:
    lines = []
    for first in FIRST_IDS:
        next_id = str(first + 1) if first < MAX_ID else 'none'
        lines.extend((f'case {first}', f'reserved {first}', f'aborted 0 {first}',
                      'unpublished missing', f'assigned {first}', f'committed 1 {next_id}',
                      'old-key missing', f'new-key value {first}', f'known {first}',
                      f'final 1 {next_id}', f'payload-first {first ^ 2779096485} {first}',
                      'payload-last 4294967295 305419896', 'end'))
    return '\n'.join(lines) + '\n'


def check_payload(text: str) -> None:
    if text != expected_payload():
        raise ValueError('owned-payload transaction differs from expected public observations')


class Commands:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.records: list[dict[str, Any]] = []

    def run(self, argv: list[str], name: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ, BEND_NO_TELEMETRY='1')
        env.pop('BEND_U64_CORRUPT', None)
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
        (self.output / f'{name}.stdout').write_text(stdout)
        (self.output / f'{name}.stderr').write_text(stderr)
        self.records.append({'stage': name, 'argv': argv, 'exit': code, 'timed_out': timed_out})
        (self.output / 'commands.json').write_text(json.dumps(self.records, indent=2) + '\n')
        if timed_out:
            raise TimeoutError(f'{name}: timed out; not an accepted compiler rejection')
        return subprocess.CompletedProcess(argv, code, stdout, stderr)

    def clean(self, argv: list[str], name: str) -> str:
        result = self.run(argv, name)
        if result.returncode or result.stderr:
            raise RuntimeError(f'{name}: exit={result.returncode}; see retained diagnostics')
        return result.stdout


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
    report: dict[str, Any] = {'status': 'failed', 'scope': __doc__, 'compiler_cases': [], 'payload_modes': []}
    commands = Commands(output)
    try:
        report['compiler'] = commands.clean([args.bun, str(HERE.parent / 'standalone/verify_compiler.js'),
                                             str(args.compiler_root.resolve())], 'compiler')
        report['cc'] = commands.clean([args.cc, '--version'], 'cc')
        names = (*DEPENDENCIES, 'reservation_payload.bend', 'reservation_ownership.py')
        report['source_sha256'] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
        checks = output / 'checks'
        checks.mkdir()
        for name in DEPENDENCIES:
            shutil.copyfile(HERE / name, checks / name)
        compiler = [args.bun, str(args.compiler_root.resolve() / 'bend2/main.ts')]
        for case in MISUSES:
            for invalid in (False, True):
                name = case.name + ('-invalid' if invalid else '-valid')
                code = program(case, invalid=invalid)
                path = checks / (name + '.bend')
                path.write_text(code)
                result = commands.run([*compiler, str(path), '--check-only'], name)
                check_compilation(result, case.diagnostic if invalid else None)
                report['compiler_cases'].append({'name': name, 'expected_rejection': invalid,
                    'source_sha256': hashlib.sha256(code.encode()).hexdigest(), 'exit': result.returncode})
        for name, body in VALID_ONLY.items():
            code = source(body)
            path = checks / (name + '.bend')
            path.write_text(code)
            result = commands.run([*compiler, str(path), '--check-only'], name)
            check_compilation(result, None)
            report['compiler_cases'].append({'name': name, 'expected_rejection': False,
                'source_sha256': hashlib.sha256(code.encode()).hexdigest(), 'exit': result.returncode})
        generated = output / 'payload.c'
        commands.clean([*compiler, str(HERE / 'reservation_payload.bend'), '-o', str(generated)], 'payload-generate')
        report['payload_generated_c_sha256'] = hashlib.sha256(generated.read_bytes()).hexdigest()
        for mode, flags in (('generic', []), ('ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all'])):
            binary = output / ('payload-' + mode)
            commands.clean([args.cc, '-std=c11', '-O2', '-ffp-contract=off', *flags, str(generated),
                            '-pthread', '-lm', '-o', str(binary)], 'payload-build-' + mode)
            text = commands.clean([str(binary), '--threads', '1'], 'payload-' + mode)
            check_payload(text)
            report['payload_modes'].append({'mode': mode, 'cases': len(FIRST_IDS), 'rows': len(text.splitlines()),
                'stdout_sha256': hashlib.sha256(text.encode()).hexdigest()})
        # Execute a validly compiled but wrong payload update, rather than count a
        # missing import, compiler failure or crash as observing the payload bytes.
        old = 'Array.set(U64, payload, 3, U64{305419896, 4294967295})'
        code = (HERE / 'reservation_payload.bend').read_text()
        if code.count(old) != 1:
            raise ValueError('payload mutation site changed')
        mutant = checks / 'lost-payload-update.bend'
        mutant.write_text(code.replace(old, 'payload'))
        generated = checks / 'mutant.c'
        commands.clean([*compiler, str(mutant), '-o', str(generated)], 'mutant-generate')
        binary = checks / 'mutant'
        commands.clean([args.cc, '-std=c11', '-O2', str(generated), '-pthread', '-lm', '-o', str(binary)], 'mutant-build')
        text = commands.clean([str(binary), '--threads', '1'], 'mutant')
        try:
            check_payload(text)
        except ValueError:
            report['lost_payload_update_rejected'] = True
        else:
            raise AssertionError('lost payload update was not detected')
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
