"""Change-scoped selection of script-only test files (validate.py + conftest).

Any doubt must run everything: these tests pin the gate, the reference closure
and every fallback, against a small synthetic tree unless stated otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts import validate
from tests import conftest

# The package name is mentioned here on purpose: it keeps this file out of the
# script-only set, so the scoping tests always run.
_PACKAGE = "chess_anti_engine"
_CI_PR = {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "pull_request"}


@pytest.fixture(autouse=True)
def _no_ambient_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GITHUB_ACTIONS", "GITHUB_EVENT_NAME", "GITHUB_BASE_REF", "GITHUB_SHA", validate.DESELECT_LIST_ENV):
        monkeypatch.delenv(name, raising=False)


def _tree(root: Path) -> Path:
    files = {
        "scripts/__init__.py": "",
        "scripts/a.py": "def f():\n    return 1\n",
        "scripts/b.py": "from scripts.a import f\n",
        "scripts/c.sh": "python scripts/b.py \"$@\"\n",
        "scripts/d.py": "def g():\n    return 2\n",
        "scripts/e.py": "def h():\n    return 3\n",
        # A dynamic import names e only as a quoted string.
        "scripts/loader.py": "import importlib\nNAMES = ['e']\nMODS = [importlib.import_module('scripts.' + n) for n in NAMES]\n",
        "scripts/validate.py": "RULE = 1\n",
        "scripts/sub/x.sh": "true\n",
        "scripts/data.json": "{}\n",
        "tests/test_a.py": "from scripts import a\n",
        "tests/test_c.py": "import subprocess\nSCRIPT = 'scripts/c.sh'\n",
        "tests/test_d.py": "from scripts.d import g\n\ndef helper():\n    return g()\n",
        "tests/test_uses_d.py": "from tests.test_d import helper\nfrom scripts import a\n",
        "tests/test_imp.py": "from tests import test_d\nfrom scripts import a\n",
        "tests/test_ad.py": "from scripts import a, d\n",
        "tests/test_loader.py": "from scripts import loader\n",
        "tests/test_pkg.py": f"from scripts import d\nimport {_PACKAGE}\n",
        "tests/test_docs.py": "from scripts import d\nfiles = (ROOT / 'docs').glob('*.md')\n",
        "tests/test_readme.py": "from scripts import d\ntext = (ROOT / 'README.md').read_text()\n",
        "tests/test_scan.py": "from scripts import d\nnames = (ROOT / 'scripts').glob('*.sh')\n",
        "tests/test_unrelated.py": "def test_x():\n    pass\n",
        "docs/guide.md": "text\n",
        "README.md": "text\n",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


ALL_SCRIPT_ONLY = {f"tests/test_{name}.py" for name in ("a", "ad", "c", "d", "uses_d", "imp", "loader")}


def test_only_unaffected_script_only_files_are_deselected_for_a_docs_change(tmp_path: Path) -> None:
    deselected, _ = validate.script_test_deselection(["docs/guide.md"], _tree(tmp_path))
    # Never script-only: test_pkg imports the package, test_docs reads docs/,
    # test_readme reads a top-level md, test_scan globs scripts/, and
    # test_unrelated references no script.
    assert deselected == ALL_SCRIPT_ONLY


@pytest.mark.parametrize("unsafe", [
    f"{_PACKAGE}/config.py", "tests/conftest.py", "tests/helpers.py", "tests/data/x.json",
    "pyproject.toml", "uv.lock", "setup.py", ".github/workflows/ci.yml", "configs/pbt2_small.yaml",
    "docs.md/x", "csrc/board.c", "scripts/validate.py",
])
def test_a_change_outside_the_safe_classes_disables_scoping(tmp_path: Path, unsafe: str) -> None:
    deselected, reason = validate.script_test_deselection(["scripts/d.py", unsafe], _tree(tmp_path))
    assert deselected is None
    assert unsafe in reason


def test_a_script_change_reaches_tests_through_script_references(tmp_path: Path) -> None:
    deselected, _ = validate.script_test_deselection(["scripts/a.py"], _tree(tmp_path))
    # test_c runs c.sh, which runs b.py, which imports a; the rest import a.
    assert deselected == {"tests/test_d.py", "tests/test_loader.py"}


def test_a_quoted_script_name_is_a_dependency(tmp_path: Path) -> None:
    deselected, _ = validate.script_test_deselection(["scripts/e.py"], _tree(tmp_path))
    assert deselected == ALL_SCRIPT_ONLY - {"tests/test_loader.py"}


def test_a_deleted_script_still_selects_its_tests(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "scripts/a.py").unlink()
    deselected, _ = validate.script_test_deselection(["scripts/a.py"], root)
    # test_ad still names an existing script (d), so it stays script-only, and
    # its reference to the deleted a must still select it.
    assert deselected == {"tests/test_d.py", "tests/test_loader.py"}


def test_a_changed_test_file_selects_itself_and_the_tests_that_import_it(tmp_path: Path) -> None:
    deselected, _ = validate.script_test_deselection(["tests/test_d.py"], _tree(tmp_path))
    # test_uses_d uses `from tests.test_d import`, test_imp `from tests import test_d`.
    assert deselected == {"tests/test_a.py", "tests/test_ad.py", "tests/test_c.py", "tests/test_loader.py"}


def test_a_deleted_test_file_still_selects_the_tests_that_import_it(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "tests/test_d.py").unlink()
    deselected, _ = validate.script_test_deselection(["tests/test_d.py"], root)
    assert deselected == {"tests/test_a.py", "tests/test_ad.py", "tests/test_c.py", "tests/test_loader.py"}


def test_a_deleted_top_level_md_keeps_its_reader_running(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "README.md").unlink()
    deselected, _ = validate.script_test_deselection(["README.md"], root)
    assert deselected == ALL_SCRIPT_ONLY


@pytest.mark.parametrize("path", ["scripts/sub/x.sh", "scripts/data.json", "scripts/__init__.py", "scripts/deepfin"])
def test_an_unmappable_scripts_path_disables_scoping(tmp_path: Path, path: str) -> None:
    deselected, reason = validate.script_test_deselection([path], _tree(tmp_path))
    assert deselected is None
    assert "cannot map" in reason


def test_an_empty_change_set_disables_scoping(tmp_path: Path) -> None:
    assert validate.script_test_deselection(["", "  "], _tree(tmp_path))[0] is None


def _items(*rels: str) -> list[pytest.Item]:
    return cast("list[pytest.Item]", [SimpleNamespace(path=validate.ROOT / rel) for rel in rels])


SCRIPT_ONLY_PROBE = "tests/test_blindspot_retire.py"


def test_conftest_deselects_only_a_readable_listed_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    items = _items(SCRIPT_ONLY_PROBE, "tests/test_validation_runner.py")
    assert conftest.change_scoped_deselection(items) == []
    listing = tmp_path / "list.txt"
    monkeypatch.setenv(validate.DESELECT_LIST_ENV, str(listing))
    assert conftest.change_scoped_deselection(items) == []  # missing file: nothing
    listing.write_text(f"{SCRIPT_ONLY_PROBE}\n")
    assert conftest.change_scoped_deselection(items) == [items[0]]


def test_real_pytest_reports_the_listed_deselection(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != validate.DESELECT_LIST_ENV}
    env["PYTHONPATH"] = str(validate.ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider", SCRIPT_ONLY_PROBE]
    plain = subprocess.run(command, cwd=validate.ROOT, env=env, capture_output=True, text=True, timeout=300,
                           check=False)
    listing = tmp_path / "list.txt"
    listing.write_text(f"{SCRIPT_ONLY_PROBE}\n")
    env[validate.DESELECT_LIST_ENV] = str(listing)
    scoped = subprocess.run(command, cwd=validate.ROOT, env=env, capture_output=True, text=True, timeout=300,
                            check=False)
    assert plain.returncode == 0, plain.stdout + plain.stderr
    assert "deselected" not in plain.stdout
    assert scoped.returncode == 5, scoped.stdout + scoped.stderr
    assert "deselected" in scoped.stdout


def _fake_git(monkeypatch: pytest.MonkeyPatch, outputs: dict[str, list[str]] | Exception) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def fake(*args: str) -> list[str]:
        calls.append(args)
        if isinstance(outputs, Exception):
            raise outputs
        return outputs.get(args[0], [])

    monkeypatch.setattr(validate, "_git", fake)
    return calls


_MERGE = {"rev-parse": ["m" * 40], "rev-list": ["m p1 p2"]}


def test_local_opt_in_adds_uncommitted_and_untracked_files(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_git(monkeypatch, {"diff": ["scripts/a.py"], "ls-files": ["tests/test_new.py"]})
    files, _ = validate.changed_files("origin/main", {})
    assert files == ["scripts/a.py", "tests/test_new.py"]
    assert ("diff", "--name-only", "--no-renames", "origin/main...HEAD") in calls
    assert ("diff", "--name-only", "--no-renames", "HEAD") in calls


@pytest.mark.parametrize("env", [
    {}, {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "push"}, {"GITHUB_EVENT_NAME": "pull_request"},
])
def test_no_pull_request_means_no_scoping(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    calls = _fake_git(monkeypatch, {**_MERGE, "diff": ["docs/x.md"]})
    assert validate.changed_files(None, env)[0] is None
    assert not calls


# Real repositories: an upstream with a PR merged onto main, after which main
# gained the PR's scripts/a.py change on its own, then a depth-1 clone of the
# merge commit as actions/checkout makes it.
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
}


def _run_git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                   env={**os.environ, **_GIT_ENV})


def _commit(repo: Path, files: dict[str, str], message: str) -> None:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _run_git(repo, "add", "-A")
    _run_git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name, value in _GIT_ENV.items():
        monkeypatch.setenv(name, value)
    repo = tmp_path / "upstream"
    repo.mkdir()
    _run_git(repo, "init", "-q", "-b", "main")
    _run_git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    _commit(repo, {"scripts/a.py": "1\n", "README.md": "r\n"}, "base")
    _run_git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, {"scripts/a.py": "2\n", "docs/x.md": "x\n"}, "pr")
    _run_git(repo, "checkout", "-q", "main")
    _commit(repo, {"README.md": "r2\n"}, "base moves")
    _run_git(repo, "checkout", "-q", "-b", "merge")
    _run_git(repo, "merge", "-q", "--no-ff", "-m", "merge", "feature")
    _run_git(repo, "checkout", "-q", "main")
    _commit(repo, {"scripts/a.py": "2\n"}, "base repeats the PR's change")
    return repo


def _ci_pr(clone: Path) -> dict[str, str]:
    """The environment GitHub gives a pull_request job checked out at ``clone``."""
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    return {**_CI_PR, "GITHUB_SHA": head}


def _clone(upstream: Path, branch: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    clone = upstream.parent / f"clone-{branch}"
    _run_git(upstream.parent, "clone", "-q", "--depth=1", "--branch", branch, upstream.as_uri(), clone.name)
    monkeypatch.setattr(validate, "ROOT", clone)
    return clone


def test_pull_request_change_set_is_the_merge_commit_against_its_first_parent(
    upstream: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = _clone(upstream, "merge", monkeypatch)
    # Against main's current tip scripts/a.py would vanish; it is the PR's change.
    assert validate.changed_files(None, _ci_pr(clone)) == (["docs/x.md", "scripts/a.py"], "ok")


def test_a_merge_head_that_is_not_the_test_merge_disables_scoping(
    upstream: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = _clone(upstream, "merge", monkeypatch)
    # A PR head that merged main in is also a two-parent commit; its first-parent
    # diff is what main brought in, not the PR's change.
    files, reason = validate.changed_files(None, {**_CI_PR, "GITHUB_SHA": "0" * 40})
    assert files is None
    assert "merge commit" in reason
    files, reason = validate.changed_files(None, _CI_PR)
    assert files is None
    assert _ci_pr(clone)["GITHUB_SHA"] != ""


def test_a_non_merge_head_disables_scoping(upstream: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clone = _clone(upstream, "main", monkeypatch)
    files, reason = validate.changed_files(None, _ci_pr(clone))
    assert files is None
    assert "merge" in reason


def test_a_failed_parent_fetch_disables_scoping(upstream: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clone = _clone(upstream, "merge", monkeypatch)
    _run_git(clone, "remote", "set-url", "origin", (upstream.parent / "missing").as_uri())
    files, reason = validate.changed_files(None, _ci_pr(clone))
    assert files is None
    assert "change detection failed" in reason


def _fake_pytest_runs(monkeypatch: pytest.MonkeyPatch, codes: tuple[int, ...]) -> list[str | None]:
    """Record, per pytest phase, the deselection list it was handed (None if none)."""
    seen: list[str | None] = []
    returns = iter(codes)

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[1:3] != ["-m", "pytest"]:  # platform.platform() may shell out
            return subprocess.CompletedProcess(command, 0, stdout="")
        listing = kwargs["env"].get(validate.DESELECT_LIST_ENV)
        seen.append(None if listing is None else Path(listing).read_text())
        return subprocess.CompletedProcess(command, next(returns))

    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    return seen


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(128, ["git", "fetch"]), OSError("no git"), {"diff": []},
    {"rev-list": ["m p1"], "diff": ["docs/x.md"]},          # not a two-parent merge
    {"rev-parse": ["f" * 40], "diff": ["docs/x.md"]},      # not the PR's test merge
])
def test_change_detection_failure_runs_everything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
    failure: Exception | dict[str, list[str]],
) -> None:
    for name, value in {**_CI_PR, "GITHUB_SHA": _MERGE["rev-parse"][0]}.items():
        monkeypatch.setenv(name, value)
    inherited = tmp_path / "inherited.txt"
    inherited.write_text(f"{SCRIPT_ONLY_PROBE}\n")
    monkeypatch.setenv(validate.DESELECT_LIST_ENV, str(inherited))  # must not leak
    _fake_git(monkeypatch, failure if isinstance(failure, Exception) else {**_MERGE, **failure})
    seen = _fake_pytest_runs(monkeypatch, (0, 0))
    assert validate.main(["cpu", "--threads", "auto", "--report-dir", str(tmp_path)]) == 0
    assert seen == [None, None]
    assert "change scoping: off" in capsys.readouterr().out


def test_scoped_run_hands_both_phases_one_list_and_still_fails_an_empty_phase(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    for name, value in {**_CI_PR, "GITHUB_SHA": _MERGE["rev-parse"][0]}.items():
        monkeypatch.setenv(name, value)
    _fake_git(monkeypatch, {**_MERGE, "diff": ["docs/x.md"]})
    seen = _fake_pytest_runs(monkeypatch, (0, 5))
    # Script-only files are a minority of both phases, so an empty phase is a
    # real fault even when scoping is on.
    assert validate.main(["cpu", "--threads", "auto", "--report-dir", str(tmp_path)]) == 5
    expected, _ = validate.script_test_deselection(["docs/x.md"], validate.ROOT)
    assert expected
    assert seen == ["".join(f"{path}\n" for path in sorted(expected))] * 2
    out = capsys.readouterr().out
    assert "change scoping: on" in out
    assert f"deselecting {len(expected)} " in out


def test_changed_vs_is_cpu_only() -> None:
    with pytest.raises(SystemExit):
        validate.main(["capped", "--changed-vs", "main"])
