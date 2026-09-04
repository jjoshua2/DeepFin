"""Grok implementation orchestration preserves source and exposes failures."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

_WRAPPER = Path(__file__).resolve().parents[1] / "scripts/grok_fix.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.mark.parametrize("failure", ["none", "grok", "lint", "test"])
def test_isolated_fix_and_exit_status(tmp_path: Path, failure: str) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@invalid")
    (repo / "scripts").mkdir()
    lint = repo / "scripts/lint.sh"
    lint.write_text('#!/bin/sh\nexit "${FAKE_LINT_EXIT:-0}"\n')
    lint.chmod(0o755)
    (repo / "value.py").write_text("value = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    head = _git(repo, "rev-parse", "HEAD")
    bins = tmp_path / "bin"
    bins.mkdir()
    grok = bins / "grok"
    grok.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args=sys.argv[1:]
wt=Path(args[args.index('--cwd')+1])
(wt/'value.py').write_text('value = 2\\n')
print('actual stub output')
sys.exit(int(os.environ.get('FAKE_GROK_EXIT', '0')))
''')
    grok.chmod(0o755)
    spec = tmp_path / "spec.md"
    spec.write_text("Update the value.")
    env = dict(os.environ)
    env.update(
        PATH=f"{bins}:{env['PATH']}", GROK_FIX_OUT=str(tmp_path / "results"),
        FAKE_GROK_EXIT="7" if failure == "grok" else "0",
        FAKE_LINT_EXIT="8" if failure == "lint" else "0",
    )
    result = subprocess.run(
        ["bash", str(_WRAPPER), "--spec", str(spec), "--branch", "fix/test",
         "--base", "HEAD", "--test", "exit 9" if failure == "test" else "exit 0"],
        cwd=repo, env=env, capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == (0 if failure == "none" else 70), result.stderr
    assert _git(repo, "rev-parse", "HEAD") == head
    assert (repo / "value.py").read_text() == "value = 1\n"
    wt = next((tmp_path / "results").glob("*/worktree"))
    assert (wt / "value.py").read_text() == "value = 2\n"
    assert _git(wt, "diff", "--name-only") == "value.py"
    raw = wt.parent / "raw.txt"
    assert raw.read_text() == "actual stub output\n"
    assert raw.stat().st_mode & 0o077 == 0
    assert head in (wt.parent / "prompt.md").read_text()
