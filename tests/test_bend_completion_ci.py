"""CI evidence admission and process cleanup; no compiler required by pytest."""
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pytest

from native.bend_engine.multi_root.ci_completion import (
    Commands, NATIVE_RESULTS, matrix_reports, native_result,
)


@pytest.mark.parametrize('kind', ['wait', 'worker'])
def test_complete_native_report(kind: str) -> None:
    assert native_result(json.dumps(NATIVE_RESULTS[kind]), kind) == NATIVE_RESULTS[kind]


@pytest.mark.parametrize('fault', ['missing', 'extra', 'failed', 'short', 'wrong_type', 'list'])
def test_invalid_native_coverage(fault: str) -> None:
    report: dict[str, Any] = dict(NATIVE_RESULTS['wait'])
    if fault == 'missing':
        del report['cases']
    elif fault == 'extra':
        report['skipped'] = 1
    elif fault == 'failed':
        report['status'] = 'failed'
    elif fault == 'short':
        report['assertions'] = 325
    elif fault == 'wrong_type':
        report['cases'] = 12.0
    text = '[]' if fault == 'list' else json.dumps(report)
    with pytest.raises(ValueError, match='native test report'):
        native_result(text, 'wait')


def write_matrix(tmp_path: Path) -> Path:
    for channels in (146, 175):
        for batch in (1, 2, 4, 8, 16):
            for mode in ('sync', 'async'):
                p = tmp_path / f'matrix/c{channels}-b{batch}/{mode}.json'
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('{"status":"passed"}')
        for name in ('ubsan', 'control', 'control-ubsan'):
            (tmp_path / f'matrix/c{channels}-b4/{name}.json').write_text('{"status":"passed"}')
        for mode in ('normal', 'ubsan'):
            p = tmp_path / f'deadlines/c{channels}-{mode}.json'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('{"status":"passed"}')
    for mode in ('normal', 'sanitized'):
        (tmp_path / f'matrix/worker-{mode}.json').write_text(json.dumps(NATIVE_RESULTS['worker']))
    (tmp_path / 'matrix/matrix.json').write_text(json.dumps({
        'status': 'passed', 'normal_configurations': 10, 'modes_each': 2,
        'ubsan_configurations': 2, 'control_configurations': 4,
        'all_final_tree_bits_match_serial': True,
    }))
    return tmp_path


def test_all_33_reports_required(tmp_path: Path) -> None:
    reports = matrix_reports(write_matrix(tmp_path))
    assert len(reports) == 33
    assert all(len(digest) == 64 for digest in reports.values())


def test_empty_matrix_does_not_pass(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        matrix_reports(tmp_path)


@pytest.mark.parametrize('filename', ['matrix/c175-b16/async.json', 'deadlines/c146-ubsan.json',
                                      'matrix/c146-b4/control.json'])
def test_missing_single_gate_rejected(tmp_path: Path, filename: str) -> None:
    write_matrix(tmp_path)
    (tmp_path / filename).unlink()
    with pytest.raises(FileNotFoundError):
        matrix_reports(tmp_path)


@pytest.mark.parametrize('fault', ['failed', 'no_status', 'short_matrix', 'bad_worker', 'bad_json'])
def test_bad_gate_rejected(tmp_path: Path, fault: str) -> None:
    write_matrix(tmp_path)
    name = 'deadlines/c175-normal.json'
    content = '{"status":"failed"}'
    if fault == 'no_status':
        content = '{}'
    elif fault == 'short_matrix':
        name = 'matrix/matrix.json'
        content = (tmp_path / name).read_text().replace('10', '9')
    elif fault == 'bad_worker':
        name = 'matrix/worker-sanitized.json'
        content = (tmp_path / name).read_text().replace('1456', '1455')
    elif fault == 'bad_json':
        content = '{'
    (tmp_path / name).write_text(content)
    with pytest.raises(ValueError, match=r'cohort|native|Expecting'):
        matrix_reports(tmp_path)


def test_command_keeps_expected_negative_evidence(tmp_path: Path) -> None:
    command = Commands(tmp_path)
    output = command.run('negative', [sys.executable, '-c',
                         'import sys; print("expected failure", file=sys.stderr); sys.exit(1)'],
                         expected_exit=1, expected_stderr='expected failure\n')
    assert output == ''
    assert command.records[0]['exit'] == 1
    assert (tmp_path / 'negative.stderr').read_text() == 'expected failure\n'
    assert json.loads((tmp_path / 'commands.json').read_text()) == command.records


@pytest.mark.parametrize('fault', ['wrong_exit', 'unexpected_stderr'])
def test_command_rejects_wrong_failure(tmp_path: Path, fault: str) -> None:
    command = Commands(tmp_path)
    script = 'pass' if fault == 'wrong_exit' else 'import sys; print("other", file=sys.stderr); sys.exit(1)'
    error = RuntimeError if fault == 'wrong_exit' else ValueError
    with pytest.raises(error, match=r'exit=|unexpected'):
        command.run('wrong', [sys.executable, '-c', script], expected_exit=1,
                    expected_stderr='expected failure\n')
    assert (tmp_path / 'wrong.stderr').exists()
    assert len(command.records) == 1


def test_timeout_kills_spawned_child_group(tmp_path: Path) -> None:
    # Both child and grandchild deliberately outlive the command budget. Killing
    # only the shell would leave its grandchild running and holding the pipes.
    pid_file = tmp_path / 'grandchild.pid'
    script = ('import pathlib, subprocess, sys, time; '
              'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
              f'pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); '
              'print("started",flush=True); time.sleep(30)')
    command = Commands(tmp_path)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='timed_out=True'):
        command.run('timeout', [sys.executable, '-c', script], timeout=1)
    assert time.monotonic() - start < 10
    assert command.records[0]['timed_out'] is True
    assert (tmp_path / 'timeout.stdout').read_text() == 'started\n'
    pid = int(pid_file.read_text())
    proc = Path(f'/proc/{pid}/stat')
    # Linux may briefly retain an already-dead orphan as a zombie before reaping.
    deadline = time.monotonic() + 2
    while True:
        try:
            if proc.read_text().split()[2] in ('Z', 'X'):
                break
        except FileNotFoundError:
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert pid != os.getpid()
