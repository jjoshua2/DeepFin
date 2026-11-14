"""scripts/sync_bend_with_main.sh against a throwaway git repo.

The script is the only thing between the `bend` branch and a silent wipe when
main (which does not carry Bend) is merged into it, so every rule it applies to
a Bend path is pinned here: main's deletions are restored, main's own changes
and additions are kept, changes on both sides are left for the operator, and it
refuses to run on a detached HEAD, a branch it was not told about, or a dirty
tree.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_bend_with_main.sh"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo, env=_ENV, check=True,
                         capture_output=True, text=True)
    return out.stdout


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _commit(repo: Path, msg: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)


def _sync(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(repo / "scripts" / "sync_bend_with_main.sh"), *args],
                          cwd=repo, env=_ENV, capture_output=True, text=True, check=False)


def _merging(repo: Path) -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
                          cwd=repo, env=_ENV, capture_output=True, check=False).returncode == 0


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Base commit with shared + Bend files; `main` then drops Bend, as #1282 did."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _write(r, "shared.txt", "shared v1\n")
    _write(r, "native/bend_engine/branch_paths.txt", "native/bend_engine/\ntests/test_bend_*.py\nscripts/sync_bend_with_main.sh\n")
    _write(r, "native/bend_engine/engine.bend", "engine v1\n")
    _write(r, "native/bend_engine/install.sh", "install v1\n")
    _write(r, "tests/test_bend_a.py", "a v1\n")
    _write(r, "scripts/sync_bend_with_main.sh", SCRIPT.read_text())
    _commit(r, "base")
    _git(r, "branch", "bend")
    _git(r, "rm", "-q", "-r", "native", "tests/test_bend_a.py", "scripts/sync_bend_with_main.sh")
    _commit(r, "main: Bend leaves main")
    _git(r, "switch", "-q", "bend")
    return r


def _first_sync(repo: Path) -> None:
    res = _sync(repo, "main")
    assert res.returncode == 0, res.stdout + res.stderr


def test_main_deletion_is_restored_and_second_run_is_a_noop(repo: Path) -> None:
    _write(repo, "native/bend_engine/install.sh", "install bend-side\n")   # modify/delete too
    _commit(repo, "bend: edit install")
    res = _sync(repo, "main")
    assert res.returncode == 0, res.stdout + res.stderr
    assert not _merging(repo)
    assert (repo / "native/bend_engine/engine.bend").read_text() == "engine v1\n"
    assert (repo / "native/bend_engine/install.sh").read_text() == "install bend-side\n"
    assert (repo / "tests/test_bend_a.py").read_text() == "a v1\n"
    assert "restored from bend  (main deleted)     native/bend_engine/engine.bend" in res.stdout
    again = _sync(repo, "main")
    assert again.returncode == 0
    assert "already up to date" in again.stdout


def test_later_sync_takes_main_changes_and_keeps_bend(repo: Path) -> None:
    _first_sync(repo)
    _git(repo, "switch", "-q", "main")
    _write(repo, "shared.txt", "shared v2\n")
    _commit(repo, "main: shared change")
    _git(repo, "switch", "-q", "bend")
    _write(repo, "native/bend_engine/install.sh", "install bend-side\n")
    _commit(repo, "bend: Bend edit")
    res = _sync(repo, "main")
    assert res.returncode == 0, res.stdout + res.stderr
    assert (repo / "shared.txt").read_text() == "shared v2\n"
    assert (repo / "native/bend_engine/install.sh").read_text() == "install bend-side\n"
    assert (repo / "native/bend_engine/engine.bend").read_text() == "engine v1\n"
    assert "Bend paths that main changed relative to the merge base: 0" in res.stdout


def test_main_modification_of_an_untouched_bend_file_is_taken(repo: Path) -> None:
    # main still carries the Bend file here (no removal): an ordinary modify.
    _git(repo, "switch", "-q", "main")
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    _write(repo, "tests/test_bend_a.py", "a main-side\n")
    _commit(repo, "main: edit Bend test")
    _git(repo, "switch", "-q", "bend")
    _write(repo, "native/bend_engine/install.sh", "install bend-side\n")
    _commit(repo, "bend: edit install only")
    res = _sync(repo, "main")
    assert res.returncode == 0, res.stdout + res.stderr
    assert (repo / "tests/test_bend_a.py").read_text() == "a main-side\n"
    assert (repo / "native/bend_engine/install.sh").read_text() == "install bend-side\n"
    assert "took main's version (main changed)     tests/test_bend_a.py" in res.stdout


def test_change_on_both_sides_is_left_for_the_operator(repo: Path) -> None:
    _git(repo, "switch", "-q", "main")
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    _write(repo, "native/bend_engine/install.sh", "install main-side\n")
    _commit(repo, "main: edit install")
    _git(repo, "switch", "-q", "bend")
    _write(repo, "native/bend_engine/install.sh", "install bend-side\n")
    _commit(repo, "bend: edit install")
    before = _git(repo, "rev-parse", "HEAD")
    res = _sync(repo, "main")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "native/bend_engine/install.sh" in res.stderr
    assert "LEFT FOR OPERATOR" in res.stdout
    assert _merging(repo)
    assert _git(repo, "rev-parse", "HEAD") == before


def test_clean_textual_merge_on_both_sides_is_still_left_for_the_operator(repo: Path) -> None:
    # Different hunks of one Bend file: git merges them without a conflict, but
    # the script must not commit main's and bend's edits together unreviewed.
    lines = [f"line {i}\n" for i in range(12)]
    _git(repo, "switch", "-q", "main")
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    _write(repo, "native/bend_engine/engine.bend", "".join(lines))
    _commit(repo, "main+bend base: multi-line Bend file")
    _git(repo, "switch", "-q", "bend")
    _git(repo, "reset", "-q", "--hard", "main")
    _git(repo, "switch", "-q", "main")
    _write(repo, "native/bend_engine/engine.bend", "".join(["main top\n", *lines[1:]]))
    _commit(repo, "main: edit top hunk")
    _git(repo, "switch", "-q", "bend")
    _write(repo, "native/bend_engine/engine.bend", "".join([*lines[:-1], "bend bottom\n"]))
    _commit(repo, "bend: edit bottom hunk")
    before = _git(repo, "rev-parse", "HEAD")
    res = _sync(repo, "main")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "native/bend_engine/engine.bend" in res.stderr
    assert _merging(repo)
    assert _git(repo, "rev-parse", "HEAD") == before
    assert not _git(repo, "diff", "--name-only", "--diff-filter=U")   # git merged it cleanly


def test_merge_in_progress_is_reported_as_such(repo: Path) -> None:
    _write(repo, "native/bend_engine/install.sh", "install bend-side\n")
    _commit(repo, "bend: edit install")
    _git(repo, "switch", "-q", "main")
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    _write(repo, "native/bend_engine/install.sh", "install main-side\n")
    _commit(repo, "main: edit install")
    _git(repo, "switch", "-q", "bend")
    assert _sync(repo, "main").returncode == 2
    res = _sync(repo, "main")
    assert res.returncode == 1
    assert "merge is already in progress" in res.stderr


def test_add_add_on_a_bend_path_is_left_for_the_operator(repo: Path) -> None:
    _first_sync(repo)
    _git(repo, "switch", "-q", "main")
    _write(repo, "native/bend_engine/NEW.md", "main\n")
    _commit(repo, "main: add")
    _git(repo, "switch", "-q", "bend")
    _write(repo, "native/bend_engine/NEW.md", "bend\n")
    _commit(repo, "bend: add")
    res = _sync(repo, "main")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "native/bend_engine/NEW.md" in res.stderr


def test_main_added_file_under_a_bend_path_is_kept(repo: Path) -> None:
    _first_sync(repo)
    _git(repo, "switch", "-q", "main")
    _write(repo, "tests/test_bend_new.py", "new\n")
    _commit(repo, "main: add Bend-path file")
    _git(repo, "switch", "-q", "bend")
    res = _sync(repo, "main")
    assert res.returncode == 0, res.stdout + res.stderr
    assert (repo / "tests/test_bend_new.py").read_text() == "new\n"
    assert "took main's version (main added)       tests/test_bend_new.py" in res.stdout


def test_refuses_detached_head(repo: Path) -> None:
    _git(repo, "switch", "-q", "--detach", "bend")
    res = _sync(repo, "main")
    assert res.returncode == 1
    assert "detached" in res.stderr
    assert not _merging(repo)


def test_refuses_an_unconfirmed_branch_and_accepts_a_confirmed_one(repo: Path) -> None:
    _git(repo, "switch", "-q", "-c", "sync/main-test", "bend")
    res = _sync(repo, "main")
    assert res.returncode == 1
    assert "--branch sync/main-test" in res.stderr
    wrong = _sync(repo, "--branch", "sync/other", "main")
    assert wrong.returncode == 1
    assert not _merging(repo)
    ok = _sync(repo, "--branch", "sync/main-test", "main")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert (repo / "native/bend_engine/engine.bend").exists()


def test_refuses_a_dirty_tree(repo: Path) -> None:
    _write(repo, "shared.txt", "uncommitted\n")
    res = _sync(repo, "main")
    assert res.returncode == 1
    assert "uncommitted" in res.stderr
    assert not _merging(repo)
