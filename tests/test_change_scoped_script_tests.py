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
from typing import cast

import pytest

from scripts import validate
from tests import conftest

# The package name is mentioned here on purpose: it keeps this file out of the
# script-only set, so the scoping tests always run.
_PACKAGE = "chess_anti_engine"


@pytest.fixture(autouse=True)
def _no_ambient_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GITHUB_EVENT_NAME", "GITHUB_BASE_REF", validate.CHANGED_FILES_ENV):
        monkeypatch.delenv(name, raising=False)


def _tree(root: Path) -> Path:
    files = {
        "scripts/__init__.py": "",
        "scripts/a.py": "def f():\n    return 1\n",
        "scripts/b.py": "from scripts.a import f\n",
        "scripts/c.sh": "python scripts/b.py \"$@\"\n",
        "scripts/d.py": "def g():\n    return 2\n",
        "scripts/sub/x.sh": "true\n",
        "scripts/data.json": "{}\n",
        "tests/test_a.py": "from scripts import a\n",
        "tests/test_c.py": "import subprocess\nSCRIPT = 'scripts/c.sh'\n",
        "tests/test_d.py": "from scripts.d import g\n\ndef helper():\n    return g()\n",
        "tests/test_uses_d.py": "from tests.test_d import helper\nfrom scripts import a\n",
        "tests/test_ad.py": "from scripts import a, d\n",
        "tests/test_pkg.py": f"from scripts import d\nimport {_PACKAGE}\n",
        "tests/test_docs.py": "from scripts import d\nfiles = (ROOT / 'docs').glob('*.md')\n",
        "tests/test_unrelated.py": "def test_x():\n    pass\n",
        "docs/guide.md": "text\n",
        "README.md": "text\n",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


ALL_SCRIPT_ONLY = {
    "tests/test_a.py", "tests/test_ad.py", "tests/test_c.py", "tests/test_d.py", "tests/test_uses_d.py",
}


def test_only_unaffected_script_only_files_are_deselected_for_a_docs_change(tmp_path: Path) -> None:
    deselected, _ = validate.script_test_deselection(["docs/guide.md", "README.md"], _tree(tmp_path))
    # test_pkg imports the package, test_docs reads docs/, test_unrelated
    # references no script: none of them may ever be deselected.
    assert deselected == ALL_SCRIPT_ONLY


@pytest.mark.parametrize("unsafe", [
    f"{_PACKAGE}/config.py", "tests/conftest.py", "tests/helpers.py", "tests/data/x.json",
    "pyproject.toml", "uv.lock", "setup.py", ".github/workflows/ci.yml", "configs/pbt2_small.yaml",
    "docs.md/x", "csrc/board.c",
])
def test_a_change_outside_the_safe_classes_disables_scoping(tmp_path: Path, unsafe: str) -> None:
    deselected, reason = validate.script_test_deselection(["scripts/d.py", unsafe], _tree(tmp_path))
    assert deselected is None
    assert unsafe in reason


def test_a_script_change_reaches_tests_through_script_references(tmp_path: Path) -> None:
    deselected, _ = validate.script_test_deselection(["scripts/a.py"], _tree(tmp_path))
    # test_a imports a; test_c runs c.sh, which runs b.py, which imports a;
    # test_uses_d imports a too. Only test_d is untouched.
    assert deselected == {"tests/test_d.py"}


def test_a_deleted_script_still_selects_its_tests(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "scripts/a.py").unlink()
    deselected, _ = validate.script_test_deselection(["scripts/a.py"], root)
    # test_ad still names an existing script (d), so it stays script-only, and
    # its reference to the deleted a must still select it.
    assert deselected == {"tests/test_d.py"}


def test_a_changed_test_file_selects_itself_and_the_tests_that_import_it(tmp_path: Path) -> None:
    deselected, _ = validate.script_test_deselection(["tests/test_d.py"], _tree(tmp_path))
    assert deselected == {"tests/test_a.py", "tests/test_ad.py", "tests/test_c.py"}


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


def test_conftest_deselects_nothing_unless_the_change_set_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    deselected, _ = validate.script_test_deselection(["docs/x.md"], validate.ROOT)
    assert deselected is not None
    assert SCRIPT_ONLY_PROBE in deselected
    items = _items(SCRIPT_ONLY_PROBE, "tests/test_validation_runner.py")
    assert conftest.change_scoped_deselection(items) == []
    monkeypatch.setenv(validate.CHANGED_FILES_ENV, "docs/x.md")
    assert conftest.change_scoped_deselection(items) == [items[0]]
    monkeypatch.setenv(validate.CHANGED_FILES_ENV, f"docs/x.md\n{_PACKAGE}/config.py")
    assert conftest.change_scoped_deselection(items) == []


def test_real_pytest_reports_the_scoped_deselection() -> None:
    env = {k: v for k, v in os.environ.items() if k != validate.CHANGED_FILES_ENV}
    env["PYTHONPATH"] = str(validate.ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider", SCRIPT_ONLY_PROBE]
    plain = subprocess.run(command, cwd=validate.ROOT, env=env, capture_output=True, text=True, timeout=300,
                           check=False)
    env[validate.CHANGED_FILES_ENV] = "docs/x.md"
    scoped = subprocess.run(command, cwd=validate.ROOT, env=env, capture_output=True, text=True, timeout=300,
                           check=False)
    assert plain.returncode == 0, plain.stdout + plain.stderr
    assert "deselected" not in plain.stdout
    # Every test of the file is deselected and reported as such.
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


def test_pull_request_change_set_is_the_diff_against_the_fetched_base(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_git(monkeypatch, {"diff": ["scripts/a.py", "docs/x.md"]})
    env = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_BASE_REF": "main"}
    assert validate.changed_files(None, env) == (["docs/x.md", "scripts/a.py"], "ok")
    assert calls[0][0] == "fetch"
    assert calls[0][-2:] == ("origin", "main")
    assert calls[1] == ("diff", "--name-only", "--no-renames", "FETCH_HEAD", "HEAD")


def test_local_opt_in_adds_uncommitted_and_untracked_files(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_git(monkeypatch, {"diff": ["scripts/a.py"], "ls-files": ["tests/test_new.py"]})
    files, _ = validate.changed_files("origin/main", {})
    assert files == ["scripts/a.py", "tests/test_new.py"]
    assert ("diff", "--name-only", "--no-renames", "origin/main...HEAD") in calls
    assert ("diff", "--name-only", "--no-renames", "HEAD") in calls


@pytest.mark.parametrize("env", [
    {}, {"GITHUB_EVENT_NAME": "push", "GITHUB_BASE_REF": "main"}, {"GITHUB_EVENT_NAME": "pull_request"},
])
def test_no_pull_request_means_no_scoping(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    calls = _fake_git(monkeypatch, {"diff": ["docs/x.md"]})
    assert validate.changed_files(None, env)[0] is None
    assert not calls


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(128, ["git", "fetch"]), OSError("no git"), {"diff": []},
])
def test_change_detection_failure_runs_everything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
    failure: Exception | dict[str, list[str]],
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_BASE_REF", "main")
    monkeypatch.setenv(validate.CHANGED_FILES_ENV, "docs/x.md")  # inherited; must not leak
    _fake_git(monkeypatch, failure)
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    seen: list[dict[str, str]] = []

    def fake_run(command: list[str], **kwargs: dict[str, str]) -> subprocess.CompletedProcess[str]:
        if command[1:3] != ["-m", "pytest"]:  # platform.platform() may shell out
            return subprocess.CompletedProcess(command, 0, stdout="")
        seen.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    assert validate.main(["cpu", "--threads", "auto", "--report-dir", str(tmp_path)]) == 0
    assert len(seen) == 2
    assert all(validate.CHANGED_FILES_ENV not in env for env in seen)
    assert "change scoping: off" in capsys.readouterr().out


@pytest.mark.parametrize("scoped", [True, False])
def test_an_emptied_phase_passes_only_when_scoping_emptied_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, scoped: bool,
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_BASE_REF", "main")
    _fake_git(monkeypatch, {"diff": ["docs/x.md" if scoped else f"{_PACKAGE}/config.py"]})
    monkeypatch.setattr(validate, "preflight", lambda *_args: None)
    seen: list[dict[str, str]] = []
    codes = iter((0, 5))

    def fake_run(command: list[str], **kwargs: dict[str, str]) -> subprocess.CompletedProcess[str]:
        if command[1:3] != ["-m", "pytest"]:  # platform.platform() may shell out
            return subprocess.CompletedProcess(command, 0, stdout="")
        seen.append(kwargs["env"])
        return subprocess.CompletedProcess(command, next(codes))

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    assert validate.main(["cpu", "--threads", "auto", "--report-dir", str(tmp_path)]) == (0 if scoped else 5)
    out = capsys.readouterr().out
    if scoped:
        assert all(env[validate.CHANGED_FILES_ENV] == "docs/x.md" for env in seen)
        assert "change scoping: on" in out
        assert "deselecting" in out
    else:
        assert all(validate.CHANGED_FILES_ENV not in env for env in seen)
        assert "change scoping: off" in out


def test_changed_vs_is_cpu_only() -> None:
    with pytest.raises(SystemExit):
        validate.main(["capped", "--changed-vs", "main"])
