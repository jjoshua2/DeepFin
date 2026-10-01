"""Host admission checks; these are not native clock qualification evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('flag', ['-O', '-OO'])
@pytest.mark.parametrize('verifier', ['standalone.verify_time_control', 'standalone.verify_session', 'async_probe.verify_search'])
def test_optimized_verifier_fails_closed_before_engine_start(
        tmp_path: Path, flag: str, verifier: str) -> None:
    report = tmp_path / 'report.json'
    report.write_text('{"qualified": true}\n')
    extra = ['--accounting-output', str(tmp_path / 'missing.txt')] if verifier.startswith('async') else []
    result = subprocess.run(
        [sys.executable, flag, '-m', 'native.bend_engine.' + verifier,
         '--report', str(report), *extra, '--command', '/deliberately/missing/engine'],
        capture_output=True, text=True, timeout=20, check=False,
        env={**os.environ, 'PYTHONOPTIMIZE': ''},
    )
    assert result.returncode != 0
    assert 'qualification requires assertions' in result.stderr
    assert 'FileNotFoundError' not in result.stderr
    assert json.loads(report.read_text())['qualified'] is False


@pytest.mark.parametrize('verifier', ['standalone.verify_time_control', 'standalone.verify_session', 'async_probe.verify_search'])
def test_normal_interpreter_allows_the_assertion_guard(verifier: str) -> None:
    result = subprocess.run(
        [sys.executable, '-c',
         f'from native.bend_engine.{verifier} import require_assertions; '
         'require_assertions(); print("guard passed")'],
        capture_output=True, text=True, timeout=20, check=False,
        env={**os.environ, 'PYTHONOPTIMIZE': ''},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'guard passed'
