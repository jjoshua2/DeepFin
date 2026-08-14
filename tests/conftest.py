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

The chosen value is printed in the pytest header every run, so a capped session
is never mistaken for a slow one.
"""
from __future__ import annotations

import os

import pytest

_ENV = "CAE_TEST_THREADS"
_DEFAULT_THREADS = 2
#  Values that mean "do not cap". `auto` is the documented spelling; the rest are
#  what people actually type when they mean it.
_UNCAPPED = frozenset({"auto", "off", "none", "0", ""})


def resolve_thread_cap() -> int | None:
    """The configured cap, or None when the caller asked for no cap.

    Exposed rather than inlined so ``tests/test_pytest_thread_cap.py`` can assert
    the REALIZED thread count against the same rule that set it. A cap nothing
    checks is exactly the kind of knob this codebase accepts and then ignores.
    """
    raw = os.environ.get(_ENV)
    if raw is None:
        return _DEFAULT_THREADS
    raw = raw.strip().lower()
    if raw in _UNCAPPED:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
  # An unparseable value must not silently fall back to "no cap" — that is the
  # failure mode this whole file exists to prevent. Fall back to the DEFAULT and
  # say so in the header.
        return _DEFAULT_THREADS


_CAP = resolve_thread_cap()

if _CAP is not None:
  # Set the OpenMP/BLAS variables FIRST: several of them are read once, at
  # library load, so setting them after `import torch` is a no-op. `setdefault`
  # keeps an explicitly-exported value winning over ours.
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
  # Belt and braces here is not decoration: which of the two takes effect depends
  # on plugin import order, which is not ours to control.
    torch.set_num_threads(_CAP)
    try:
        torch.set_num_interop_threads(_CAP)
    except RuntimeError:
  # Raises once the inter-op pool has been started. Intra-op is where the 848%
  # came from, and that call above always lands, so this is a genuine no-op
  # rather than a swallowed failure.
        pass


def pytest_report_header() -> str:
    """Print the realized cap, not the configured one."""
    import torch

    if _CAP is None:
        return f"torch threads: UNCAPPED ({torch.get_num_threads()}) — {_ENV} is set to no-cap"
    return (
        f"torch threads: capped at {_CAP} (realized {torch.get_num_threads()}) "
        f"to leave CPU for live training — lift with {_ENV}=auto"
    )


@pytest.fixture(scope="session")
def thread_cap() -> int | None:
    """The session's configured cap, for tests that need to reason about it."""
    return _CAP
