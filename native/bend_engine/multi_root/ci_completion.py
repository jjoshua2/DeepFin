"""Source-only completion regression gates. See CI.md; no benchmark or artifact input."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BACKEND = HERE.parent / 'batch_backend'
NATIVE_RESULTS = {
    'worker': {'status': 'passed', 'assertions': 1456, 'configurations': 10, 'reused_calls': 160},
    'wait': {'status': 'passed', 'cases': 12, 'assertions': 326},
}


def native_result(text: str, kind: str) -> dict[str, Any]:
    """A clean exit without the complete expected report is not a test pass."""
    report = json.loads(text)
    expected = NATIVE_RESULTS[kind]
    if not isinstance(report, dict) or set(report) != set(expected):
        raise ValueError('native test report fields differ')
    if any(type(report[k]) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError('native test report result or coverage differs')
    return report


def matrix_reports(directory: Path) -> dict[str, str]:
    """Require every expected independent-verifier output, not a possibly empty glob."""
    names = ['matrix/matrix.json', 'matrix/worker-normal.json', 'matrix/worker-sanitized.json']
    for channels in (146, 175):
        names.extend(f'matrix/c{channels}-b{batch}/{mode}.json'
                     for batch in (1, 2, 4, 8, 16) for mode in ('sync', 'async'))
        names.extend(f'matrix/c{channels}-b4/{name}.json'
                     for name in ('ubsan', 'control', 'control-ubsan'))
        names.extend(f'deadlines/c{channels}-{mode}.json' for mode in ('normal', 'ubsan'))
    hashes = {}
    for name in names:
        raw = (directory / name).read_bytes()
        report = json.loads(raw)
        if not isinstance(report, dict) or report.get('status') != 'passed':
            raise ValueError(f'cohort verifier failed: {name}')
        if name == 'matrix/matrix.json' and report != {
            'status': 'passed', 'normal_configurations': 10, 'modes_each': 2,
            'ubsan_configurations': 2, 'control_configurations': 4,
            'all_final_tree_bits_match_serial': True,
        }:
            raise ValueError('cohort matrix coverage differs')
        if name.startswith('matrix/worker-'):
            native_result(raw.decode(), 'worker')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


class Commands:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.records: list[dict[str, Any]] = []

    def run(self, stage: str, argv: list[str], timeout: int = 120, *,
            expected_exit: int = 0, expected_stderr: str | None = None) -> str:
        """Capture evidence even on failure; kill shell children as well on timeout."""
        started = time.monotonic()
        timed_out = False
        with subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True) as child:
            try:
                stdout, stderr = child.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                stdout, stderr = child.communicate()
            code = child.returncode
        (self.output / f'{stage}.stdout').write_text(stdout)
        (self.output / f'{stage}.stderr').write_text(stderr)
        self.records.append({'stage': stage, 'argv': argv, 'exit': code, 'timed_out': timed_out,
                             'seconds': time.monotonic() - started})
        (self.output / 'commands.json').write_text(json.dumps(self.records, indent=2) + '\n')
        if timed_out or code != expected_exit:
            raise RuntimeError(f'{stage}: exit={code}, timed_out={timed_out}; see captured logs')
        if expected_stderr is not None and stderr != expected_stderr:
            raise ValueError(f'{stage}: unexpected native diagnostic: {stderr!r}')
        return stdout


def native(commands: Commands, cxx: str) -> dict[str, Any]:
    reports = {}
    commands.run('cxx-version', [cxx, '--version'])
    for mode in ('normal', 'sanitized'):
        flags = [] if mode == 'normal' else ['-fsanitize=address,undefined', '-fno-sanitize-recover=all']
        for kind, filename in (('worker', 'async_batch_test.cpp'), ('wait', 'async_wait_test.cpp')):
            binary = commands.output / f'{kind}-{mode}'
            commands.run(f'build-{kind}-{mode}', [cxx, '-std=c++20', '-O1', '-pthread',
                         '-Wall', '-Wextra', '-Werror', *flags, str(BACKEND / filename), '-o', str(binary)])
            text = commands.run(f'{kind}-{mode}', [str(binary)], timeout=30, expected_stderr='')
            reports[f'{kind}-{mode}'] = native_result(text, kind)
            if kind == 'wait':
                text = commands.run(f'unwind-{mode}', [str(binary), '--abort-held'], timeout=10,
                                    expected_exit=1, expected_stderr='intentional held-callback assertion\n')
                if text:
                    raise ValueError('intentional assertion produced a passing report')
    mutant = commands.output / 'no-notify'
    mutant.mkdir()
    source = (BACKEND / 'async_batch.h').read_text()
    if source.count('ready_.notify_one();') != 1:
        raise ValueError('notification mutation site differs; review the negative control')
    (mutant / 'async_batch.h').write_text(source.replace('ready_.notify_one();', '(void)0;'))
    shutil.copyfile(BACKEND / 'async_wait_test.cpp', mutant / 'async_wait_test.cpp')
    binary = mutant / 'test'
    commands.run('build-no-notify', [cxx, '-std=c++20', '-O1', '-pthread', '-Wall', '-Wextra',
                 '-Werror', str(mutant / 'async_wait_test.cpp'), '-o', str(binary)])
    text = commands.run('no-notify', [str(binary)], timeout=30, expected_exit=1,
                        expected_stderr='completion notification did not wake waiter\n')
    if text:
        raise ValueError('missing notification produced a passing report')
    return {'native_reports': reports, 'no_notify_rejected': True, 'held_assertion_unwinds': True}


def cohort(commands: Commands, compiler: Path, bun: str, cc: str) -> dict[str, Any]:
    # No stale generated C, previous runs, network artifacts, or performance thresholds.
    commands.run('verify-compiler', [bun, str(HERE.parent / 'standalone/verify_compiler.js'), str(compiler)])
    generated = commands.output / 'runner.c'
    commands.run('generate', [bun, str(compiler / 'bend2/main.ts'), str(HERE / 'main.bend'),
                             '-o', str(generated)], timeout=600)
    if not generated.is_file() or generated.stat().st_size == 0:
        raise ValueError('compiler produced no coordinator C')
    oracle = commands.output / 'oracle'
    commands.run('oracle', [cc, '-std=c11', '-O3', '-DLEGAL_ORACLE', '-I', str(ROOT),
                 str(HERE.parent / 'legal_probe/support.c'), '-pthread', '-lm', '-o', str(oracle)])
    commands.run('matrix', ['bash', str(HERE / 'qualify_async.sh'), str(generated), str(oracle),
                           str(commands.output / 'matrix')], timeout=900)
    commands.run('deadlines', ['bash', str(HERE / 'qualify_deadlines.sh'), str(compiler),
                              str(commands.output / 'matrix'), str(commands.output / 'deadlines')], timeout=180)
    return {'verifier_reports': matrix_reports(commands.output),
            'generated_c_sha256': hashlib.sha256(generated.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope', choices=('native', 'cohort'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler-root', type=Path)
    args = parser.parse_args()
    if args.scope == 'cohort' and args.compiler_root is None:
        parser.error('cohort scope requires --compiler-root')
    output = args.output.resolve()
    if output.exists():
        parser.error('output exists; preserve old evidence and use a fresh directory')
    output.mkdir(parents=True)
    report: dict[str, Any] = {'status': 'failed', 'scope': args.scope, 'no_performance_measurement': True,
                              'checkout_sha': os.environ.get('GITHUB_SHA')}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        files = sorted(p for p in HERE.parent.rglob('*') if p.is_file() and
                       p.suffix in ('.bend', '.py', '.js', '.json', '.c', '.cpp', '.h', '.sh'))
        report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        commands = Commands(output)
        if args.scope == 'native':
            report.update(native(commands, os.environ.get('CXX', 'clang++')))
        else:
            assert args.compiler_root is not None
            report.update(cohort(commands, args.compiler_root.resolve(), os.environ.get('BUN', 'bun'),
                                 os.environ.get('CC', 'clang')))
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
