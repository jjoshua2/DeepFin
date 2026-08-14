"""Session-wide pytest configuration.

⚑⚑ THE SUITE SHARES A BOX WITH LIVE TRAINING, AND BY DEFAULT IT DOES NOT SHARE
POLITELY. torch's intra-op pool sizes itself to the machine, so a single
``python -m pytest`` on tiny unit-test tensors was measured at **848% CPU** on
2026-08-14 while a training arm was mid-iteration — 8.5 cores taken from
selfplay, on a box whose selfplay workers deliberately run at ``nice 15``. It
does not corrupt anything; it costs the live run wall clock, silently, and the
person who started the tests never sees the bill.

So the cap is applied HERE rather than left to whoever remembers the environment
variable. ``conftest.py`` is loaded by pytest on every invocation — bare,
``-k``-filtered, from an agent, from a script, from an IDE — which is the only
placement that does not depend on memory.

To lift it, when training is paused or on a machine with nothing else running::

    CAE_TEST_THREADS=auto python -m pytest        # torch decides (pre-2026-08-14)
    CAE_TEST_THREADS=8    python -m pytest        # an explicit number

The regime is printed on EVERY run, so a capped session is never mistaken for a
slow one. ⚑ See ``pytest_terminal_summary`` below for why it is emitted there
and not from ``pytest_report_header``.
"""
from __future__ import annotations

import os

import pytest

_ENV = "CAE_TEST_THREADS"
_DEFAULT_THREADS = 2
# Values that mean "do not cap". `auto` is the documented spelling; the rest are
# what people actually type when they mean it.
#
# ⚑ "false"/"no"/"n" are here for a YAML reason, not a typing one: this variable
# is set in `.github/workflows/ci.yml`, and YAML 1.1 parses a bare `off` as the
# BOOLEAN False, which reaches the process as the string "false". Documenting
# `off` while rejecting "false" would cap CI silently. The same trap is recorded
# in this repo's memory as `yaml_off_parses_as_boolean_false`.
_UNCAPPED = frozenset({"auto", "off", "none", "no", "n", "false", "0", ""})


def classify_thread_cap(raw: str | None) -> tuple[int | None, bool]:
    """Resolve ``raw`` to ``(cap, fell_back_from_garbage)``.

    The second element exists so the regime line can SAY that it ignored what you
    typed. A silent fallback and a deliberate cap are indistinguishable at the
    prompt otherwise, which is how ``CAE_TEST_THREADS=atuo`` caps forever.
    """
    if raw is None:
        return _DEFAULT_THREADS, False
    cleaned = raw.strip().lower()
    if cleaned in _UNCAPPED:
        return None, False
    try:
        return max(1, int(cleaned)), False
    except ValueError:
        # An unparseable value must not silently fall back to "no cap" — that is
        # the failure mode this whole file exists to prevent. Failing CLOSED is
        # deliberate: failing open steals ~8.5 cores from a live run and bills
        # someone else, failing closed only makes your own tests slow.
        return _DEFAULT_THREADS, True


def resolve_thread_cap() -> int | None:
    """The configured cap, or None when the caller asked for no cap.

    Exposed rather than inlined so ``tests/test_pytest_thread_cap.py`` can assert
    the REALIZED thread count against the same rule that set it. A cap nothing
    checks is exactly the kind of knob this codebase accepts and then ignores.
    """
    cap, _ = classify_thread_cap(os.environ.get(_ENV))
    return cap


_RAW = os.environ.get(_ENV)
_CAP, _CAP_IS_FALLBACK = classify_thread_cap(_RAW)

if _CAP is not None:
    # Set the OpenMP/BLAS variables FIRST: several of them are read once, at
    # library load, so setting them after `import torch` is a no-op. `setdefault`
    # keeps an explicitly-exported value winning over ours.
    #
    # ⚑ This half is NOT redundant with the torch calls below, and a measurement
    # says so: deleting it leaves this process capped (torch's own API carries
    # the CPU bill) but un-caps every CHILD process, because subprocesses inherit
    # these variables and never load this file. The e2e tests boot real workers.
    for _var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_var, str(_CAP))

    import torch

    # ⚑ And ALSO through torch's API, because the environment variables above are
    # already too late if any plugin imported torch before this file was loaded.
    # Belt and braces here is not decoration: which of the two takes effect
    # depends on plugin import order, which is not ours to control.
    torch.set_num_threads(_CAP)
    try:
        torch.set_num_interop_threads(_CAP)
    except RuntimeError:
        # Raises once the inter-op pool has been started. Intra-op is where the
        # 848% came from, and that call above always lands, so this is a genuine
        # no-op rather than a swallowed failure.
        pass


def thread_cap_regime_line() -> str:
    """One line describing the REALIZED regime, not the configured one."""
    import torch

    if _CAP is None:
        return f"torch threads: UNCAPPED ({torch.get_num_threads()}) — {_ENV} is set to no-cap"
    line = (
        f"torch threads: capped at {_CAP} (realized {torch.get_num_threads()}) "
        f"to leave CPU for live training — lift with {_ENV}=auto"
    )
    if _CAP_IS_FALLBACK:
        line += f" [⚑ {_ENV}={_RAW!r} is not a number — ignored, capped anyway]"
    return line


def pytest_report_header() -> str:
    """The regime line for normal/verbose runs."""
    return thread_cap_regime_line()


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """And again at the end, which is the only place ``-q`` cannot suppress it.

    ⚑ MEASURED, not assumed. Two earlier emit sites each looked right and printed
    NOTHING: ``pytest_report_header`` alone is skipped entirely at ``-q``'s
    verbosity of -1, and this repo's ``pyproject.toml`` puts ``-q`` in
    ``addopts`` — so the documented default invocation was the broken case. And
    ``config.pluginmanager.get_plugin("terminalreporter")`` returns **None**
    inside a conftest's ``pytest_configure``, making a write through it a branch
    that cannot fire. The end of the run is also where the reader actually is
    when they wonder why the suite felt slow.
    """
    terminalreporter.write_line(thread_cap_regime_line())


@pytest.fixture(scope="session")
def thread_cap() -> int | None:
    """The session's configured cap, for tests that need to reason about it."""
    return _CAP
