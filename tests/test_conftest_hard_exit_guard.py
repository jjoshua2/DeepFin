"""The conftest's guard on ``os._exit`` inside the pytest process.

The UCI ``quit`` after a device fault leaves with ``os._exit`` (no interpreter
teardown). Called from a test in-process, that would end the whole pytest run
with the status the engine chose -- ``0`` for a clean quit -- so the run reads
green with every later test unrun. The autouse fixture stubs ``_hard_exit``
only on a ``uci.__main__`` module already imported when the test starts; these
tests pin the backstop for everything else.

Every probe that can reach a real ``os._exit`` runs in a child pytest: run
in-process, a missing guard would end THIS run with status 0 and no output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import conftest

_REPO = Path(__file__).resolve().parents[1]


_PROBE = textwrap.dedent('''
    import os

    def test_a_exits_in_process():
        os._exit(0)

    def test_b_swallows_it():
        try:
            os._exit(0)
        except BaseException:
            pass

    def test_c_a_later_test_still_runs():
        assert True
''')

_LAZY_PROBE = textwrap.dedent('''
    import importlib
    import sys

    def test_a_lazy_uci_main_hard_exit():
        # A fresh module object, imported inside the test (after the autouse
        # fixture ran): exactly the case the fixture's stub cannot see.
        sys.modules.pop("chess_anti_engine.uci.__main__", None)
        fresh = importlib.import_module("chess_anti_engine.uci.__main__")
        fresh._hard_exit(0)

    def test_b_a_later_test_still_runs():
        assert True
''')


def _run_probe(tmp_path: Path, source: str, *extra: str) -> tuple[int, str]:
    probe = tmp_path / "test_probe_exit.py"
    probe.write_text(source)
    inherited = os.environ.get("PYTHONPATH", "")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(p for p in [str(_REPO / "tests"), str(_REPO), inherited] if p))
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "conftest", "-p", "no:cacheprovider",
         "-o", "addopts=", "-rA", "--rootdir", str(tmp_path), *extra, str(probe)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=300, check=False,
    )
    return done.returncode, done.stdout + done.stderr


def test_a_lazily_imported_uci_main_hard_exit_fails_the_test_instead_of_ending_pytest(tmp_path):
    rc, text = _run_probe(tmp_path, _LAZY_PROBE)
    assert rc != 0, text  # unguarded: the run ends here with status 0 and no output
    assert "FAILED test_probe_exit.py::test_a_lazy_uci_main_hard_exit" in text, text
    assert "InProcessHardExit" in text, text
    assert "PASSED test_probe_exit.py::test_b_a_later_test_still_runs" in text, text


_XFAIL_AND_SKIP_PROBE = textwrap.dedent('''
    import os
    import pytest

    @pytest.mark.xfail(reason="probe")
    def test_a_swallows_then_fails_as_xfail():
        try:
            os._exit(0)
        except BaseException:
            pass
        raise AssertionError("expected failure")

    def test_b_swallows_then_skips():
        try:
            os._exit(0)
        except BaseException:
            pass
        pytest.skip("skipped after the exit")

    def test_c_a_later_test_still_runs():
        assert True
''')


@pytest.mark.parametrize("xdist", [False, True])
def test_a_swallowed_os_exit_fails_an_xfail_or_skipped_test_too(tmp_path, xdist):
    extra = ("-n", "2") if xdist else ()
    if xdist:
        pytest.importorskip("xdist")
    rc, text = _run_probe(tmp_path, _XFAIL_AND_SKIP_PROBE, *extra)
    assert "FAILED test_probe_exit.py::test_a_swallows_then_fails_as_xfail" in text, text
    assert "FAILED test_probe_exit.py::test_b_swallows_then_skips" in text, text
    assert "PASSED test_probe_exit.py::test_c_a_later_test_still_runs" in text, text
    assert rc != 0, text


_SWALLOW_ONLY = _PROBE.split("def test_a_exits_in_process")[0] + _PROBE[_PROBE.index("def test_b_swallows_it"):]


def test_a_swallowed_in_process_os_exit_fails_that_test_and_the_run(tmp_path):
    # The code under test catches the guard's exception and the test body
    # passes: the per-test report check (and the sessionfinish check) make it red.
    rc, text = _run_probe(tmp_path, _SWALLOW_ONLY)
    assert "FAILED test_probe_exit.py::test_b_swallows_it" in text, text
    assert "PASSED test_probe_exit.py::test_c_a_later_test_still_runs" in text, text
    assert rc != 0, text
    assert "conftest: intercepted in-process os._exit(0)" in text, text


def test_a_swallowed_in_process_os_exit_fails_under_xdist_too(tmp_path):
    # CI's main phase runs under xdist, where a worker's session exit status
    # never reaches the controller: only the test's own failed report does.
    pytest.importorskip("xdist")
    rc, text = _run_probe(tmp_path, _SWALLOW_ONLY, "-n", "2")
    assert "FAILED test_probe_exit.py::test_b_swallows_it" in text, text
    assert "PASSED test_probe_exit.py::test_c_a_later_test_still_runs" in text, text
    assert rc != 0, text


def test_an_in_process_os_exit_fails_the_run_and_later_tests_still_run(tmp_path):
    rc, text = _run_probe(tmp_path, _PROBE)
    assert rc != 0, text  # unguarded: os._exit(0) ends the run with 0
    assert "PASSED test_probe_exit.py::test_c_a_later_test_still_runs" in text, text
    assert "FAILED test_probe_exit.py::test_a_exits_in_process" in text, text
    assert "FAILED test_probe_exit.py::test_b_swallows_it" in text, text
    # Both interceptions are also listed at sessionfinish.
    assert text.count("conftest: intercepted in-process os._exit(0) in test_probe_exit.py") >= 2, text


def _child_exits_7() -> None:
    os._exit(7)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_still_gets_the_real_os_exit():
    # The child inherits the wrapper with its own pid: the guard must hand it
    # the real exit, or it raises InProcessHardExit and dies with status 1.
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child
        try:
            os._exit(7)
        finally:
            conftest._REAL_OS_EXIT(99)  # the guard raised instead of exiting
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status), status
    assert os.WEXITSTATUS(status) == 7, status


def test_a_spawned_child_that_imports_the_guard_still_gets_the_real_os_exit():
    # A spawned child unpickles its target by importing this module, which
    # imports ``tests.conftest`` afresh in the child: the guard must stay off
    # there (the child is not a pytest process).
    import multiprocessing

    proc = multiprocessing.get_context("spawn").Process(target=_child_exits_7)
    proc.start()
    proc.join(120)
    assert proc.exitcode == 7, proc.exitcode
