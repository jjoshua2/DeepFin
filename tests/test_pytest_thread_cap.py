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

from tests.conftest import resolve_thread_cap


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
        ("", None),
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
