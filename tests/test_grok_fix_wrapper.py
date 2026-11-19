"""Grok implementation orchestration preserves source and exposes failures."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import subprocess
import time

import pytest

_WRAPPER = Path(__file__).resolve().parents[1] / "scripts/grok_fix.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _setup(tmp_path: Path) -> tuple[Path, dict[str, str], str, Path]:
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
import json, os, signal, subprocess, sys, time
from pathlib import Path
args=sys.argv[1:]
Path(os.environ['ARGV_FILE']).write_text(json.dumps(args))
wt=Path(args[args.index('--cwd')+1])
(wt/'value.py').write_text('value = 2\\n')
print('actual stub output', flush=True)
if os.environ.get('RESIST_TERM'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    Path(os.environ['CHILD_PID']).write_text(str(child.pid))
    time.sleep(60)
sys.exit(int(os.environ.get('FAKE_GROK_EXIT', '0')))
''')
    grok.chmod(0o755)
    spec = tmp_path / "spec.md"
    spec.write_text("Update the value.")
    env = dict(os.environ)
    # The caller's shell must not leak a budget override into the default test.
    env.pop("GROK_FIX_MAX_TURNS", None)
    env.pop("GROK_FIX_TIMEOUT_SECONDS", None)
    env.update(
        PATH=f"{bins}:{env['PATH']}", GROK_FIX_OUT=str(tmp_path / "results"),
        ARGV_FILE=str(tmp_path / "argv.json"),
    )
    return repo, env, head, spec


def _run(repo: Path, env: dict[str, str], spec: Path,
         testcmd: str = "exit 0") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_WRAPPER), "--spec", str(spec), "--branch", "fix/test",
         "--base", "HEAD", "--test", testcmd],
        cwd=repo, env=env, capture_output=True, text=True, timeout=30, check=False,
    )


def _max_turns(env: dict[str, str]) -> str:
    argv = json.loads(Path(env["ARGV_FILE"]).read_text())
    assert argv.count("--max-turns") == 1, argv
    return argv[argv.index("--max-turns") + 1]


def _assert_reaped(pid: int) -> None:
    status = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 2
    while True:
        try:
            state = status.read_text().split(") ", 1)[1]
        except FileNotFoundError:
            return
        # A terminated orphan can briefly remain as a zombie pending its reaper.
        if state.startswith("Z"):
            return
        assert time.monotonic() < deadline, f"owned pid {pid} survived cleanup"
        time.sleep(0.01)


@pytest.mark.parametrize("failure", ["none", "grok", "lint", "test"])
def test_isolated_fix_and_exit_status(tmp_path: Path, failure: str) -> None:
    repo, env, head, spec = _setup(tmp_path)
    env.update(
        FAKE_GROK_EXIT="7" if failure == "grok" else "0",
        FAKE_LINT_EXIT="8" if failure == "lint" else "0",
    )
    result = _run(repo, env, spec, "exit 9" if failure == "test" else "exit 0")
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


def test_default_turn_budget_reaches_grok(tmp_path: Path) -> None:
    repo, env, _, spec = _setup(tmp_path)
    result = _run(repo, env, spec)
    assert result.returncode == 0, result.stderr
    assert _max_turns(env) == "300"


def test_turn_budget_override_reaches_grok(tmp_path: Path) -> None:
    repo, env, _, spec = _setup(tmp_path)
    env["GROK_FIX_MAX_TURNS"] = "17"
    result = _run(repo, env, spec)
    assert result.returncode == 0, result.stderr
    assert _max_turns(env) == "17"


@pytest.mark.parametrize("turns", ["abc", "0", "00", "-5", "3 00"])
def test_invalid_turn_budget_fails_before_any_side_effect(tmp_path: Path, turns: str) -> None:
    repo, env, head, spec = _setup(tmp_path)
    env["GROK_FIX_MAX_TURNS"] = turns
    result = _run(repo, env, spec)
    assert result.returncode == 64, result.stderr
    assert "GROK_FIX_MAX_TURNS" in result.stderr
    assert not Path(env["ARGV_FILE"]).exists()
    assert not (tmp_path / "results").exists()
    assert _git(repo, "branch", "--list", "fix/test") == ""
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert _git(repo, "rev-parse", "HEAD") == head


@pytest.mark.parametrize("duration", ["0", "abc", "-1"])
def test_invalid_timeout_fails_before_any_side_effect(tmp_path: Path, duration: str) -> None:
    repo, env, _, spec = _setup(tmp_path)
    env["GROK_FIX_TIMEOUT_SECONDS"] = duration
    assert _run(repo, env, spec).returncode == 64
    assert not Path(env["ARGV_FILE"]).exists()
    assert not (tmp_path / "results").exists()
    assert _git(repo, "branch", "--list", "fix/test") == ""


def _recording_timeout(bins: Path, record: Path) -> None:
    """Install a ``timeout`` that records its argv, then runs the real one."""
    real = shutil.which("timeout")
    assert real is not None
    shim = bins / "timeout"
    shim.write_text(f"""#!/usr/bin/env bash
printf '%s\\n' "$@" > {str(record)!r}
exec {real!r} "$@"
""")
    shim.chmod(0o755)


@pytest.mark.parametrize(("override", "expected"), [(None, "10800s"), ("77", "77s")])
def test_wall_clock_budget_reaches_timeout(
    tmp_path: Path, override: str | None, expected: str,
) -> None:
    repo, env, _, spec = _setup(tmp_path)
    if override is not None:
        env["GROK_FIX_TIMEOUT_SECONDS"] = override
    record = tmp_path / "timeout-argv"
    _recording_timeout(tmp_path / "bin", record)
    result = _run(repo, env, spec)
    assert result.returncode == 0, result.stderr
    argv = record.read_text().splitlines()
    assert argv[:2] == ["--kill-after=5s", expected], argv


def test_timeout_kills_term_resistant_process_group(tmp_path: Path) -> None:
    repo, env, _, spec = _setup(tmp_path)
    pid_file = tmp_path / "child.pid"
    env.update(RESIST_TERM="1", CHILD_PID=str(pid_file), GROK_FIX_TIMEOUT_SECONDS="1")
    result = _run(repo, env, spec)
    assert result.returncode == 70, result.stderr
    _assert_reaped(int(pid_file.read_text()))
    # The worktree is retained for the caller even when Grok is killed.
    assert next((tmp_path / "results").glob("*/worktree")).is_dir()


def test_wrapper_kills_descendants_a_child_only_timeout_leaves_behind(tmp_path: Path) -> None:
    """The wrapper, not ``timeout``, owns the kill of Grok's process group.

    Whether ``timeout`` signals the whole group is implementation-specific, so
    this substitutes one that kills only its direct child.
    """
    repo, env, _, spec = _setup(tmp_path)
    fake_timeout = tmp_path / "bin" / "timeout"
    fake_timeout.write_text('''#!/usr/bin/env python3
import subprocess, sys
args = sys.argv[1:]
while args[0].startswith("--"):
    args.pop(0)
child = subprocess.Popen(args[1:])
try:
    raise SystemExit(child.wait(timeout=float(args[0].rstrip("s"))))
except subprocess.TimeoutExpired:
    child.kill()
    child.wait()
    raise SystemExit(124)
''')
    fake_timeout.chmod(0o755)
    pid_file = tmp_path / "child.pid"
    env.update(RESIST_TERM="1", CHILD_PID=str(pid_file), GROK_FIX_TIMEOUT_SECONDS="3")
    result = _run(repo, env, spec)
    assert result.returncode == 70, result.stderr
    _assert_reaped(int(pid_file.read_text()))


def test_live_pgid_inspection_failure_cleans_up_owned_descendant(tmp_path: Path) -> None:
    """Inspection errors fail closed and the EXIT trap still kills the group."""
    repo, env, _, spec = _setup(tmp_path)
    child_file = tmp_path / "child.pid"
    leader_file = tmp_path / "leader.pid"
    env.update(RESIST_TERM="1", CHILD_PID=str(child_file), PS_LEADER_FILE=str(leader_file))
    proxy = tmp_path / "bin" / "ps"
    proxy.write_text("""#!/usr/bin/env bash
printf '%s\\n' "${4}" > "$PS_LEADER_FILE"
for _ in $(seq 1 500); do
    [ -s "$CHILD_PID" ] && break
    sleep 0.01
done
[ -s "$CHILD_PID" ] || exit 3
exit 2
""")
    proxy.chmod(0o755)
    result = _run(repo, env, spec)
    assert result.returncode == 70, result.stderr
    assert "could not inspect the Grok process group" in result.stderr
    for pid_file in (leader_file, child_file):
        _assert_reaped(int(pid_file.read_text()))
