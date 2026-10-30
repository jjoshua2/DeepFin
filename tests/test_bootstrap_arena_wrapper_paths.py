"""Run generated arena wrappers ONLY after substituting an inert stdlib argv.

No arena, model, GPU probe, process discovery or queue command is executed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from tests.script_loading import load_script_module


def _generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
              runtime_name: str = 'runtime', out_name: str = 'out',
              relative_out: bool = False, budget: int = 2) -> tuple[Path, Path, dict]:
    module = load_script_module('bootstrap_experiment_operator.py', 'arena_wrapper_paths')
    monkeypatch.chdir(tmp_path)
    runtime = tmp_path / runtime_name
    runtime.mkdir()
    out = tmp_path / out_name
    calls: list[str] = []

    def gpu_apps() -> str:
        calls.append('inert-gpu-probe')
        return ''

    def start_wrapper(*_args: Any) -> SimpleNamespace:
        calls.append('mocked-launch')
        return SimpleNamespace(pid=12345)

    def no_external(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('wrapper generation must not execute any command')

    monkeypatch.setattr(module, 'RUNTIME', runtime)
    monkeypatch.setattr(module, 'gpu_apps', gpu_apps)
    monkeypatch.setattr(module, 'mem_avail_gib', lambda: 64)
    monkeypatch.setattr(module, 'disk_free_gib', lambda _path: 200)
    monkeypatch.setattr(module, 'launch_budget', lambda _item: budget)
    monkeypatch.setattr(module, 'start_wrapper', start_wrapper)
    monkeypatch.setattr(module, 'log', lambda *_args, **_kwargs: None)
    item = {'id': 'inert', 'out': out_name if relative_out else str(out),
            'games': 2, 'sims': 1, 'seed': 1,
            'candidate': 'NEVER-OPENED-candidate.pt', 'reference': 'NEVER-OPENED-reference.pt'}
    # Guard the generation boundary as well as mocking its known call sites.
    with monkeypatch.context() as guard:
        guard.setattr(module.subprocess, 'Popen', no_external)
        guard.setattr(module.subprocess, 'check_output', no_external)
        module.launch_arena(item)
    assert calls == ['inert-gpu-probe', 'inert-gpu-probe', 'mocked-launch']
    descriptor = out / 'command.json'
    original = json.loads(descriptor.read_text())
    assert original['command'][:3] == ['/usr/bin/python3', '-m', 'scripts.arena_standard']
    assert item['out'] == (out_name if relative_out else str(out))
    return runtime, out, original


def _replace_command(out: Path, command: Any) -> None:
    descriptor = out / 'command.json'
    data = json.loads(descriptor.read_text())
    data['command'] = command
    descriptor.write_text(json.dumps(data))
    assert json.loads(descriptor.read_text())['command'] == command


def _run_inert_wrapper(out: Path, *, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    # This function is called only after the original command descriptor was
    # explicitly replaced. No production launcher or queue process is started.
    command = json.loads((out / 'command.json').read_text())['command']
    assert not isinstance(command, list) or '-m' not in command
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(out.parent), 'CUDA_VISIBLE_DEVICES': ''}
    env.update(env_extra or {})
    wrapper = out / 'launch_parent.sh'
    syntax = subprocess.run(['/bin/bash', '-n', str(wrapper)], env=env,
                            capture_output=True, text=True, check=False, timeout=3)
    assert syntax.returncode == 0, syntax.stderr
    proc = subprocess.Popen(['/bin/bash', str(wrapper)], cwd=out.parent, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=5)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=3)


PROBE = (
    "import json,os,sys; print(json.dumps({'argv':sys.argv[1:], 'cwd':os.getcwd(), "
    "'pythonpath':os.environ.get('PYTHONPATH'), 'threads':os.environ.get('OMP_NUM_THREADS')}))"
)
SPECIAL_NAMES = ['plain', 'with space', "single'quote", 'double"quote', '$HOME',
                 '$(printf NO_EXPANSION)', '`printf NO_EXPANSION`', '*?[glob]',
                 'semicolon;literal', 'line\nbreak', 'back\\slash',
                 'line\nARGV_PY\nTERMINAL_PY']


@pytest.mark.parametrize('name', SPECIAL_NAMES)
@pytest.mark.parametrize('which', ['runtime', 'out'])
def test_arena_paths_remain_literal_across_shell_and_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, which: str,
) -> None:
    runtime, out, _ = _generate(tmp_path, monkeypatch,
                                runtime_name=name if which == 'runtime' else 'runtime',
                                out_name=name if which == 'out' else 'out')
    arguments = ['space value', "quote'\"", '$HOME', '$(printf NO_EXPANSION)',
                 '`printf NO_EXPANSION`', 'line\nbreak', '', 'trailing\n', '\u2603']
    _replace_command(out, ['/usr/bin/python3', '-c', PROBE, *arguments])
    result = _run_inert_wrapper(out, env_extra={'PYTHONPATH': 'inherited literal'})
    assert result.returncode == 0, result.stderr
    actual = json.loads((out / 'arena.log').read_text())
    assert actual == {'argv': arguments, 'cwd': str(runtime),
                      'pythonpath': str(runtime) + ':inherited literal', 'threads': '2'}
    terminal = json.loads((out / 'parent_outer_terminal.json').read_text())
    assert terminal['returncode'] == 0
    assert terminal['elapsed_seconds'] >= 0
    assert terminal['ended_unix'] >= json.loads((out / 'parent_outer_started.json').read_text())['started_unix']
    # Literal metacharacter names never expand into alternate artifacts.
    assert {p.name for p in out.iterdir()} == {
        'command.json', 'launch_parent.sh', 'parent_outer_started.json',
        'parent_outer.pid', 'arena.log', 'parent_outer_terminal.json',
    }


def test_relative_output_is_bound_before_runtime_chdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, out, original = _generate(tmp_path, monkeypatch, out_name='relative out', relative_out=True)
    assert original['command'][original['command'].index('--out') + 1] == str(out / 'arena.results.jsonl')
    _replace_command(out, ['/usr/bin/python3', '-c', PROBE, 'one'])
    result = _run_inert_wrapper(out)
    assert result.returncode == 0, result.stderr
    assert json.loads((out / 'arena.log').read_text())['cwd'] == str(runtime)
    assert json.loads((out / 'parent_outer_terminal.json').read_text())['returncode'] == 0
    assert not (runtime / 'relative out').exists()


def test_nonzero_inert_child_retains_exact_terminal_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime, out, _ = _generate(tmp_path, monkeypatch, out_name="out 'quoted'")
    _replace_command(out, ['/usr/bin/python3', '-c', 'raise SystemExit(7)'])
    result = _run_inert_wrapper(out)
    assert result.returncode == 7
    assert json.loads((out / 'parent_outer_terminal.json').read_text())['returncode'] == 7


def test_timeout_of_owned_inert_child_has_terminal_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime, out, _ = _generate(tmp_path, monkeypatch, budget=1)
    _replace_command(out, ['/usr/bin/python3', '-c', 'import time; time.sleep(2)'])
    result = _run_inert_wrapper(out)
    assert result.returncode == 124
    assert json.loads((out / 'parent_outer_terminal.json').read_text())['returncode'] == 124


@pytest.mark.parametrize('invalid', [[], 'not an argv', [None], ['echo', 'bad\0arg']])
def test_invalid_argv_cannot_execute_a_partial_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: Any,
) -> None:
    _runtime, out, _ = _generate(tmp_path, monkeypatch)
    _replace_command(out, invalid)
    result = _run_inert_wrapper(out)
    assert result.returncode != 0
    assert json.loads((out / 'parent_outer_terminal.json').read_text())['returncode'] != 0
    assert 'command must be a nonempty list of NUL-free strings' in result.stderr
