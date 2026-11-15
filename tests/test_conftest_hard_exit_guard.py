"""The conftest's guard on ``os._exit`` inside the pytest process.

The UCI ``quit`` after a device fault leaves with ``os._exit`` (no interpreter
teardown). Called from a test in-process, that would end the whole pytest run
with the status the engine chose -- ``0`` for a clean quit -- so the run reads
green with every later test unrun. The autouse fixture stubs ``_hard_exit``
only on a ``uci.__main__`` module already imported when the test starts; these
tests pin the backstop for everything else.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import conftest

_REPO = Path(__file__).resolve().parents[1]


def test_a_lazily_imported_uci_main_hard_exit_raises_instead_of_ending_pytest(monkeypatch):
    # A fresh module object, imported inside the test (after the autouse
    # fixture ran): exactly the case the fixture's stub cannot see.
    monkeypatch.delitem(sys.modules, "chess_anti_engine.uci.__main__", raising=False)
    fresh = importlib.import_module("chess_anti_engine.uci.__main__")
    before = len(conftest._INTERCEPTED_HARD_EXITS)
    with pytest.raises(conftest.InProcessHardExit, match=r"os\._exit\(0\)"):
        fresh._hard_exit(0)
    # The record would fail this session at sessionfinish: drop ours.
    assert len(conftest._INTERCEPTED_HARD_EXITS) == before + 1
    del conftest._INTERCEPTED_HARD_EXITS[before:]


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


def _run_probe(tmp_path: Path, source: str) -> tuple[int, str]:
    probe = tmp_path / "test_probe_exit.py"
    probe.write_text(source)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(_REPO / "tests"), str(_REPO)]),
               CUDA_VISIBLE_DEVICES="")
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("PYTEST_ADDOPTS", None)
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "conftest", "-p", "no:cacheprovider",
         "-o", "addopts=", "-rA", "--rootdir", str(tmp_path), str(probe)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=300, check=False,
    )
    return done.returncode, done.stdout + done.stderr


def test_a_swallowed_in_process_os_exit_still_fails_the_run(tmp_path):
    # The code under test catches the guard's exception and every test
    # passes: only the sessionfinish check can make this run red.
    source = _PROBE.split("def test_a_exits_in_process")[0] + _PROBE[_PROBE.index("def test_b_swallows_it"):]
    rc, text = _run_probe(tmp_path, source)
    assert "PASSED test_probe_exit.py::test_b_swallows_it" in text, text
    assert rc != 0, text
    assert "conftest: intercepted in-process os._exit(0)" in text, text


def test_an_in_process_os_exit_fails_the_run_and_later_tests_still_run(tmp_path):
    rc, text = _run_probe(tmp_path, _PROBE)
    assert rc != 0, text  # unguarded: os._exit(0) ends the run with 0
    assert "PASSED test_probe_exit.py::test_c_a_later_test_still_runs" in text, text
    assert "FAILED test_probe_exit.py::test_a_exits_in_process" in text, text
    # Swallowed by the code under test, still counted at sessionfinish.
    assert text.count("conftest: intercepted in-process os._exit(0)") == 2, text
