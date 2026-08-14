"""The suite's torch thread cap must actually take effect.

⚑ A cap that is configured and then ignored is worse than no cap: it produces a
header line claiming the box is protected while the run takes every core anyway.
That is this codebase's signature defect (a value accepted and silently
dropped), so the cap gets a test that reads the REALIZED thread count rather
than the configured one.

⚑ And a test that only asserts ``get_num_threads() == cap`` can pass on a
2-core machine where torch would have chosen 2 unaided — the assertion would be
true and would prove nothing. So binding-ness is checked first
(``check_the_resource_is_binding``): when the machine has no more cores than the
cap, the test SKIPS with the reason stated instead of banking a free pass.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from tests.conftest import classify_thread_cap, resolve_thread_cap, thread_cap_regime_line


def test_the_configured_cap_is_the_realized_cap(thread_cap: int | None) -> None:
    if thread_cap is None:
        pytest.skip("CAE_TEST_THREADS asks for no cap; nothing to assert")
    cores = os.cpu_count() or 1
    if cores <= thread_cap:
        pytest.skip(
            f"cap {thread_cap} is not binding on a {cores}-core machine — "
            "asserting it here would pass without testing anything"
        )
    assert torch.get_num_threads() == thread_cap


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 2),          # the default, and the whole point of the file
        ("auto", None),
        ("off", None),
        ("0", None),
        # ⚑ NOT uncapped: YAML `CAE_TEST_THREADS:` with no value (or `~`/`null`)
        # arrives as "". Nobody types that meaning "take every core".
        ("", 2),
        # ⚑ YAML 1.1 turns a bare `off` into the boolean False, which reaches the
        # process as "false". CI sets this variable from yaml, so rejecting these
        # would cap CI silently.
        ("false", None),
        ("no", None),
        ("n", None),
        ("AUTO", None),     # case-folded before lookup
        ("8", 8),
        ("1", 1),
        ("-4", 1),          # clamped, never zero
        ("banana", 2),      # ⚑ unparseable falls back to the CAP, not to uncapped
    ],
)
def test_resolve_thread_cap(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: int | None,
) -> None:
    if value is None:
        monkeypatch.delenv("CAE_TEST_THREADS", raising=False)
    else:
        monkeypatch.setenv("CAE_TEST_THREADS", value)
    assert resolve_thread_cap() == expected


def test_an_uncapped_session_really_is_uncapped() -> None:
    """The NEGATIVE CONTROL: prove the cap is what constrains torch, not the box.

    Without this, `test_the_configured_cap_is_the_realized_cap` is consistent
    with a conftest that does nothing on a machine that happens to default to 2.
    A child process with the cap lifted must report MORE threads than the cap.

    ⚑ What actually does the lifting here is the env SCRUB below, not the
    ``CAE_TEST_THREADS="auto"`` we hand the child: the child is bare `python -c`,
    never loads this conftest, and so never reads that variable. It is set only
    so the child's environment states the intent it is running under. Removing
    the scrub makes this test FAIL rather than silently pass, which is the
    property that makes it a control at all.
    """
    cap = resolve_thread_cap()
    cores = os.cpu_count() or 1
    if cap is None or cores <= cap:
        pytest.skip("no cap configured, or the cap is not binding on this machine")
    env = dict(os.environ, CAE_TEST_THREADS="auto")
    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        env.pop(var, None)
    out = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.get_num_threads())"],
        capture_output=True, text=True, env=env, timeout=300, check=True,
    )
    assert int(out.stdout.strip()) > cap, (
        "an uncapped torch chose no more threads than the cap, so this machine "
        "cannot distinguish a working cap from a broken one"
    )


def test_a_child_process_inherits_the_cap() -> None:
    """The env-var half of the cap is the ONLY thing that reaches subprocesses.

    ⚑ This exists because both halves of the cap independently suffice for THIS
    process, so a mutation deleting either one leaves the rest of the suite
    green. The halves are not interchangeable: `torch.set_num_threads` dies with
    this interpreter, while `OMP_NUM_THREADS` is what a spawned worker reads at
    its own torch import. The e2e tests boot real selfplay workers, so losing
    this half silently un-caps the processes that do the most work.
    """
    cap = resolve_thread_cap()
    cores = os.cpu_count() or 1
    if cap is None or cores <= cap:
        pytest.skip("no cap configured, or the cap is not binding on this machine")
    assert os.environ.get("OMP_NUM_THREADS") == str(cap), (
        "conftest did not export OMP_NUM_THREADS; child processes will not be capped"
    )
    out = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.get_num_threads())"],
        capture_output=True, text=True, timeout=300, check=True,
    )
    assert int(out.stdout.strip()) <= cap, (
        f"a child process chose {out.stdout.strip()} threads despite the cap"
    )


def test_an_unparseable_value_is_announced_not_just_absorbed() -> None:
    """Failing closed is right; failing closed SILENTLY is not.

    `CAE_TEST_THREADS=atuo` caps at the default forever and, without this,
    produces output byte-identical to a deliberate cap.
    """
    assert classify_thread_cap("banana") == (2, True)
    assert classify_thread_cap("auto") == (None, False)
    assert classify_thread_cap("8") == (8, False)
    assert classify_thread_cap(None) == (2, False)


def test_the_regime_line_names_the_escape_hatch() -> None:
    """The line must carry the override spelling, since it is the only place a
    reader is told the cap is liftable."""
    line = thread_cap_regime_line()
    assert "CAE_TEST_THREADS" in line
    if resolve_thread_cap() is not None:
        assert "auto" in line


@pytest.mark.parametrize(
    ("env_value", "must_contain"),
    [
        (None,   "capped at 2 (realized 2)"),
        ("atuo", "is not a number"),
        ("auto", "UNCAPPED"),
    ],
    ids=["default", "unparseable", "lifted"],
)
def test_the_regime_line_reaches_the_terminal_with_its_payload(
    env_value: str | None, must_contain: str,
) -> None:
    """⚑ THE OBSERVABLE AND ITS CONTENT — neither alone is enough.

    Asserting what `thread_cap_regime_line()` RETURNS proves nothing about whether
    a human sees it: three plausible emit sites printed nothing here (`-q` is in
    this repo's `addopts` and suppresses `pytest_report_header`; and
    `pluginmanager.get_plugin("terminalreporter")` is None inside a conftest's
    `pytest_configure`, a branch that cannot fire).

    But asserting only that the PREFIX `"torch threads:"` appeared is not enough
    either, and that gap was real: an emit site hardcoded to print
    `"torch threads: UNCAPPED (32) ..."` during a CAPPED run passed the whole
    suite — green while the terminal actively lied about the regime, which is the
    exact failure this line exists to prevent, inverted. Deleting the
    unparseable-value annotation also survived, leaving F3's fix with no observable
    test at all.

    So each regime asserts a payload only a truthful line can carry, through a
    real subprocess pytest run. `-q` is passed EXPLICITLY rather than inherited
    from `addopts`, so this keeps testing the `-q` case if `addopts` ever changes.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    if env_value is None:
        env.pop("CAE_TEST_THREADS", None)
    else:
        env["CAE_TEST_THREADS"] = env_value
    # The child re-reads conftest from scratch; the parent's exported OMP_* would
    # otherwise pre-empt the regime under test.
    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        env.pop(var, None)
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_pytest_thread_cap.py",
         "-k", "resolve_thread_cap", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=repo_root, env=env, timeout=600, check=False,
    )
    # ⚑ pytest_terminal_summary fires even when zero tests are collected, so a
    # child that died during collection would still print the line and pass a
    # content check. Pin that it actually ran something.
    assert out.returncode == 0, (
        f"the child pytest run did not succeed (rc={out.returncode}); the regime "
        f"line below proves nothing about a working run.\n"
        f"--- stdout ---\n{out.stdout}\n--- stderr ---\n{out.stderr}"
    )
    assert must_contain in out.stdout, (
        f"regime line missing its payload {must_contain!r} on the "
        f"CAE_TEST_THREADS={env_value!r} run — the terminal is either silent about "
        f"the regime or misreporting it.\n--- stdout ---\n{out.stdout}"
    )
