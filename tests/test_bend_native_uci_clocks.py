"""Host admission checks; these are not native clock qualification evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('flag', ['-O', '-OO'])
def test_optimized_verifier_fails_closed_before_engine_start(tmp_path: Path, flag: str) -> None:
    report = tmp_path / 'report.json'
    report.write_text('{"qualified": true}\n')
    result = subprocess.run(
        [sys.executable, flag, '-m', 'native.bend_engine.standalone.verify_time_control',
         '--report', str(report), '--command', '/deliberately/missing/engine'],
        capture_output=True, text=True, timeout=20, check=False,
        env={**os.environ, 'PYTHONOPTIMIZE': ''},
    )
    assert result.returncode != 0
    assert 'qualification requires assertions' in result.stderr
    assert 'FileNotFoundError' not in result.stderr
    assert json.loads(report.read_text())['qualified'] is False


def test_normal_interpreter_allows_the_assertion_guard() -> None:
    result = subprocess.run(
        [sys.executable, '-c',
         'from native.bend_engine.standalone.verify_time_control import require_assertions; '
         'require_assertions(); print("guard passed")'],
        capture_output=True, text=True, timeout=20, check=False,
        env={**os.environ, 'PYTHONOPTIMIZE': ''},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'guard passed'
