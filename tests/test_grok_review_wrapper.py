"""Exercise review snapshots and verdict handling without calling an external model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

_WRAPPER = Path(__file__).resolve().parents[1] / "scripts" / "grok_review.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _setup(tmp_path: Path) -> tuple[Path, dict[str, str], str, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@invalid")
    (repo / "value.txt").write_text("base")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "value.txt").write_text("candidate")
    _git(repo, "commit", "-qam", "candidate")
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", base)
    bins = tmp_path / "bin"
    bins.mkdir()
    grok = bins / "grok"
    grok.write_text('''#!/usr/bin/env python3
import json, os, signal, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
snap = Path(args[args.index('--cwd') + 1])
prompt = Path(args[args.index('--prompt-file') + 1]).read_text()
Path(os.environ['OBSERVED']).write_text(json.dumps({
    'snapshot': str(snap), 'value': (snap/'value.txt').read_text(),
    'extra': (snap/'extra.txt').read_text() if (snap/'extra.txt').exists() else None,
    'prompt': prompt,
}))
(snap/'value.txt').write_text('model edit in disposable snapshot')
if os.environ.get('CHANGE_SOURCE'):
    Path(os.environ['CHANGE_SOURCE']).write_text('concurrent user edit')
if os.environ.get('RESIST_TERM'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    Path(os.environ['CHILD_PID']).write_text(str(child.pid))
    time.sleep(60)
if os.environ.get('SLEEP'):
    time.sleep(5)
print(os.environ.get('FAKE_OUTPUT', 'BEGIN_FINDINGS\\nNONE\\nEND_FINDINGS'), flush=True)
sys.exit(int(os.environ.get('FAKE_EXIT', '0')))
''')
    grok.chmod(0o755)
    gh = bins / "gh"
    gh.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
if args[:2] == ['repo', 'view']:
    print('owner/repo')
elif args[:2] == ['pr', 'view']:
    print('# PR 1')
elif args[:2] == ['pr', 'diff']:
    print('diff from stable PR head')
    if os.environ.get('MOVE_PR'):
        Path(os.environ['MOVED']).touch()
elif args[0] == 'api':
    head = '0'*40 if Path(os.environ['MOVED']).exists() else os.environ['PR_HEAD']
    print(os.environ['PR_BASE'], head)
else:
    sys.exit(1)
''')
    gh.chmod(0o755)
    env = dict(os.environ)
    env.update(
        PATH=f"{bins}:{env['PATH']}",
        GROK_REVIEW_OUT=str(tmp_path / "results"),
        GROK_REVIEW_TIMEOUT_SECONDS="10",
        OBSERVED=str(tmp_path / "observed.json"),
        MOVED=str(tmp_path / "moved"),
        PR_BASE=base,
        PR_HEAD=head,
    )
    return repo, env, base, head


def _run(repo: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_WRAPPER), *args], cwd=repo, env=env,
        capture_output=True, text=True, timeout=15, check=False,
    )


def test_range_uses_candidate_not_callers_head(tmp_path: Path) -> None:
    repo, env, base, head = _setup(tmp_path)
    result = _run(repo, env, "--diff", f"{base}...{head}")
    assert result.returncode == 0, result.stderr
    observed = json.loads(Path(env["OBSERVED"]).read_text())
    assert observed["value"] == "candidate"
    assert f"Resolved base: {base}; snapshot head: {head}" in observed["prompt"]
    assert not Path(observed["snapshot"]).exists()
    assert (repo / "value.txt").read_text() == "base"
    assert "BEGIN_FINDINGS\nNONE\nEND_FINDINGS" in result.stdout
    raw = next(Path(env["GROK_REVIEW_OUT"]).glob("*/raw.txt"))
    assert raw.read_text() == "BEGIN_FINDINGS\nNONE\nEND_FINDINGS\n"
    assert raw.stat().st_mode & 0o077 == 0


def test_worktree_includes_edits_and_untracked(tmp_path: Path) -> None:
    repo, env, _, _ = _setup(tmp_path)
    (repo / "value.txt").write_text("edited")
    (repo / "extra.txt").write_text("untracked")
    result = _run(repo, env, "--worktree", str(repo))
    assert result.returncode == 0, result.stderr
    observed = json.loads(Path(env["OBSERVED"]).read_text())
    assert observed["value"] == "edited"
    assert observed["extra"] == "untracked"
    assert _git(repo, "rev-parse", "HEAD") in observed["prompt"]
    assert "Source fingerprint" in observed["prompt"]
    assert (repo / "value.txt").read_text() == "edited"


@pytest.mark.parametrize("moved", [False, True])
def test_pr_snapshot_and_revision_consistency(tmp_path: Path, moved: bool) -> None:
    repo, env, _, head = _setup(tmp_path)
    if moved:
        env["MOVE_PR"] = "1"
    result = _run(repo, env, "--pr", "1")
    if moved:
        assert result.returncode == 65
        assert not Path(env["OBSERVED"]).exists()
    else:
        assert result.returncode == 0, result.stderr
        observed = json.loads(Path(env["OBSERVED"]).read_text())
        assert observed["value"] == "candidate"
        assert head in observed["prompt"]


@pytest.mark.parametrize("output", [
    "", "looks good", "BEGIN_FINDINGS\nNONE", "BEGIN_FINDINGS\ngarbage\nEND_FINDINGS",
    "BEGIN_FINDINGS\nNONE\nEND_FINDINGS\nBEGIN_FINDINGS\nNONE\nEND_FINDINGS",
])
def test_malformed_output_is_not_a_clean_review(tmp_path: Path, output: str) -> None:
    repo, env, base, head = _setup(tmp_path)
    env["FAKE_OUTPUT"] = output
    result = _run(repo, env, "--diff", f"{base}...{head}")
    assert result.returncode == 70
    assert "unparseable" in result.stderr
    assert next(Path(env["GROK_REVIEW_OUT"]).glob("*/raw.txt")).read_text() == output + "\n"


def test_findings_are_preserved(tmp_path: Path) -> None:
    repo, env, base, head = _setup(tmp_path)
    output = "BEGIN_FINDINGS\n1 | value.txt:1 | MAJOR | claim | input -> wrong | check\nEND_FINDINGS"
    env["FAKE_OUTPUT"] = output
    result = _run(repo, env, "--diff", f"{base}...{head}")
    assert result.returncode == 0, result.stderr
    assert output in result.stdout


@pytest.mark.parametrize("mode", ["failure", "timeout", "concurrent_change"])
def test_execution_failures_never_report_clean(tmp_path: Path, mode: str) -> None:
    repo, env, base, head = _setup(tmp_path)
    if mode == "failure":
        env["FAKE_EXIT"] = "1"
    elif mode == "timeout":
        env.update(SLEEP="1", GROK_REVIEW_TIMEOUT_SECONDS="1")
    else:
        env["CHANGE_SOURCE"] = str(repo / "value.txt")
    result = _run(repo, env, "--diff", f"{base}...{head}")
    assert result.returncode == (71 if mode == "concurrent_change" else 70)
    assert "raw_output:" in result.stdout
    if mode == "concurrent_change":
        assert (repo / "value.txt").read_text() == "concurrent user edit"


@pytest.mark.parametrize("args", [("--pr",), ("--pr", "1", "--diff", "a...b")])
def test_bad_arguments_fail_without_invoking_grok(tmp_path: Path, args: tuple[str, ...]) -> None:
    repo, env, _, _ = _setup(tmp_path)
    assert _run(repo, env, *args).returncode == 64
    assert not Path(env["OBSERVED"]).exists()


@pytest.mark.parametrize("duration", ["0", "00", "000", "-1", "nope"])
def test_invalid_timeout_cannot_disable_budget(tmp_path: Path, duration: str) -> None:
    repo, env, base, head = _setup(tmp_path)
    env["GROK_REVIEW_TIMEOUT_SECONDS"] = duration
    assert _run(repo, env, "--diff", f"{base}...{head}").returncode == 64
    assert not Path(env["OBSERVED"]).exists()


def test_timeout_kills_term_resistant_process_group(tmp_path: Path) -> None:
    repo, env, base, head = _setup(tmp_path)
    pid_file = tmp_path / "child.pid"
    env.update(RESIST_TERM="1", CHILD_PID=str(pid_file), GROK_REVIEW_TIMEOUT_SECONDS="1")
    result = _run(repo, env, "--diff", f"{base}...{head}")
    assert result.returncode == 70, result.stderr
    child = int(pid_file.read_text())
    status = Path(f"/proc/{child}/stat")
    # A terminated orphan can briefly remain as a zombie pending its parent's reaper.
    assert not status.exists() or status.read_text().split(") ", 1)[1].startswith("Z")
