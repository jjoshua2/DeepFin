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

The other resident here is the ``pytest_runtest_protocol`` hookwrapper at the
bottom of the file — a guard against a different kind of shared state, the
process-global repetition-fix encoder flag. Same placement argument: a conftest
is the only site that does not depend on the next test author remembering.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import weakref
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    # Type-only: the runtime lookup goes through `sys.modules` so that this
    # conftest never drags numpy and zarr into a run that has nothing to do
    # with replay. See `_reap_replay_prefetch_threads`.
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

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
#
# ⚑ The EMPTY string is deliberately NOT here. It is the one spelling nobody types
# on purpose: YAML `CAE_TEST_THREADS:` with no value, or `~`/`null`, arrives as ""
# — and an author who writes that means "leave it at the default", not "give this
# run every core on the box". Unset (`None`) and set-but-empty are distinguished
# below, so "" takes the announced fail-closed path like any other garbage.
_UNCAPPED = frozenset({"auto", "off", "none", "no", "n", "false", "0"})


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


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Emit the regime line — one site, at the end, at every verbosity.

    ⚑ There is deliberately NO ``pytest_report_header`` companion. It was here and
    was removed: ``terminal_summary`` already fires at every verbosity, so the
    header only duplicated the line under ``-v`` while being dead under ``-q`` —
    and a mutation that made it return ``""`` was not detectable by any test,
    which is the signature of code carrying no weight.

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


PREFETCH_THREAD_NAME_PREFIX = "replay-prefetch-"

_REAPED_PREFETCH_BUFFERS = 0


def reaped_prefetch_buffer_count() -> int:
    """How many prefetch-thread-bearing buffers this session has reaped so far.

    Exposed for the same reason ``resolve_thread_cap`` is: so a test can assert
    against the REALIZED number rather than infer it. "No prefetch thread is
    alive" is the outcome that matters, but on its own it does not say WHO
    achieved it — and a fixture that has quietly stopped reaping looks identical
    to one that reaps, right up until the day something else stops cleaning up
    too. ``tests/test_replay_prefetch_thread_lifecycle.py`` pins this above zero
    on a run known to have built buffers, so the day the reaping moves elsewhere
    (say ``close()`` becomes reachable from ``__del__``) it is reported instead
    of silently inherited.
    """
    return _REAPED_PREFETCH_BUFFERS


@pytest.fixture(autouse=True)
def _reap_replay_prefetch_threads() -> Iterator[None]:
    """Close every ``DiskReplayBuffer`` whose prefetch thread this test started.

    ``DiskReplayBuffer`` starts a daemon prefetch thread the first time a sample
    triggers a shuffle refresh, and ``close()`` is what stops it. Tests that
    never call ``close()`` therefore hand the thread to the REST OF THE SESSION,
    and it keeps polling on a 0.1s tick forever. Measured on ``main``
    (2026-08-24): ``pytest tests/test_replay_disk_buffer.py`` alone finishes with
    **17** live ``replay-prefetch-*`` threads, and every later test in that
    process runs alongside all of them.

    ⚑⚑ THE LEAK IS NOT MERELY UNTIDY — IT WRITES INTO OTHER TESTS' STDOUT. The
    orphaned thread still holds the shard paths it was constructed with, and
    once pytest removes the owning ``tmp_path`` those paths are still TRACKED in
    ``_shard_paths``. ``_note_shard_load_failure`` classifies a tracked-but-
    unreadable shard as a REAL fault rather than the benign trim race, so each
    orphan prints ``[disk_buf] WARNING: shuffle refresh failed to load a TRACKED
    shard ...`` — into whichever unrelated test happens to be running. The
    throttle is per buffer (``_SHARD_LOAD_WARN_INTERVAL_S``, 60s), so N orphans
    give N lines a minute, not one.

    ⚑ ``__del__`` IS NOT THE SAFETY NET IT LOOKS LIKE, AND CANNOT BE. It calls
    ``close()``, but a running ``threading.Thread`` is an external GC root and
    the thread's ``target`` is the BOUND METHOD ``self._prefetch_loop`` — so a
    buffer with a live prefetch thread is strongly reachable from
    ``threading._active`` and is never collected. Measured directly: drop the
    last reference, ``gc.collect()`` three times, and the thread is still alive.
    The finaliser works only in the case that does not matter (no thread ever
    started), which is why the reaping has to be explicit here.

    Mechanism: wrap ``_ensure_prefetch_thread`` — the single site that creates
    the thread — so each buffer that actually starts one is recorded, then
    ``close()`` them at teardown. Recording at the CREATION site rather than at
    ``__init__`` keeps the reap set exact: buffers built with
    ``deterministic_refresh`` or ``refresh_interval <= 0`` never start a thread
    and are never touched.

    ⚑ A function-scoped autouse fixture is right HERE even though the
    ``pytest_runtest_protocol`` hookwrapper below argues the opposite for the
    encoder flag, and the difference is snapshot-vs-reap. That hook must observe
    state that EXISTS BEFORE any fixture runs, so a function-scoped fixture is
    too late. This one only reaps objects the test itself created during the
    call phase, so being set up last costs nothing — and being function-scoped
    is what makes it safe, since a module-scoped fixture's buffer would
    otherwise be closed out from under the module's second test. Checked when
    this was written: no fixture in this suite constructs a ``DiskReplayBuffer``
    at all, every construction is inside a test body.

    ⚑ ``sys.modules`` is consulted rather than importing ``disk_buffer`` here.
    This fixture is autouse for the WHOLE suite and a conftest import would pull
    numpy and zarr into every run that has nothing to do with replay. Test
    modules are imported during collection, which completes before the first
    test executes, so the module is already present whenever any collected test
    could build a buffer.
    """
    disk_buffer = sys.modules.get("chess_anti_engine.replay.disk_buffer")
    if disk_buffer is None:
        yield
        return

    cls = disk_buffer.DiskReplayBuffer
    # ⚑ WEAK references, and the choice is load-bearing rather than tidy. A
    # strong list here would make the FIXTURE the thing keeping every buffer
    # alive, which silently invalidates any test that reasons about the
    # buffer's own reachability — `test_prefetch_thread_keeps_its_buffer_alive_
    # so_del_cannot_close_it` would pass no matter what the thread did. Weak is
    # also sufficient: a buffer with a running prefetch thread is reachable from
    # `threading._active` and cannot be collected, so the referent is still
    # there at teardown precisely when there is a thread to stop. If one HAS
    # been collected, `__del__` already ran `close()` and there is nothing left
    # to do.
    started: weakref.WeakSet[DiskReplayBuffer] = weakref.WeakSet()
    # The THREADS are tracked separately, also weakly, because `close()` is not
    # proof of shutdown: it joins for only 1.0s and then drops
    # `_prefetch_thread` even if the thread is still alive (a slow or stalled
    # shard read), which would hand the next test a live thread this fixture
    # believed it had reaped. A weakref to a Thread does not keep the BUFFER
    # alive (the reachability argument above is undisturbed); a running thread
    # is pinned by `threading._active`, and one that has finished may be
    # collected — then there is nothing left to join.
    threads: weakref.WeakSet[threading.Thread] = weakref.WeakSet()
    real = cls._ensure_prefetch_thread

    def _tracking_ensure_prefetch_thread(self: DiskReplayBuffer) -> None:
        real(self)
        # Record only on success: `_ensure_prefetch_thread` returns without
        # starting anything for the suppressed configurations above.
        if self._prefetch_thread is not None:
            started.add(self)
            threads.add(self._prefetch_thread)

    cls._ensure_prefetch_thread = _tracking_ensure_prefetch_thread
    try:
        yield
    finally:
        global _REAPED_PREFETCH_BUFFERS
        cls._ensure_prefetch_thread = real
        # Materialise before closing: `close()` can drop the last strong
        # reference and mutating a WeakSet while iterating it raises.
        for buf in list(started):
            # A test that deliberately leaves a buffer in a broken state must not
            # turn teardown into an error; the thread is what we are here for.
            with contextlib.suppress(Exception):
                buf.close()
            _REAPED_PREFETCH_BUFFERS += 1
        started.clear()
        # `close()` returning is NOT proof the thread stopped (1.0s join, then
        # the reference is dropped) — wait out any straggler ourselves. This
        # also covers a thread that outlived a test's OWN `close()` call, which
        # the buffer no longer remembers but this WeakSet does. The stop event
        # is already set, so a live thread here is finishing one shard read;
        # 30s is a shared ceiling, not a per-thread hang budget.
        deadline = time.monotonic() + 30.0
        for t in list(threads):
            t.join(timeout=max(0.0, deadline - time.monotonic()))
            if t.is_alive():
                # Loud, not fatal: the next tests may see its output, and that
                # must be attributable to THIS test, not mistaken for theirs.
                print(
                    f"[conftest] WARNING: prefetch thread {t.name} survived "
                    "close() plus a 30s join and may pollute later tests",
                    file=sys.stderr,
                )
        threads.clear()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol() -> Iterator[None]:
    """Snapshot/restore the process-global ``history_rep_fix`` encoder flag.

    ``rep_fix._current`` and the globals it pushes into ``_lc0_ext`` and
    ``_mcts_tree`` are PROCESS state, not per-test state: whichever test touched
    them last decides how every later test in the same pytest process encodes
    its repetition planes. A test does not have to mention the flag to change
    it — ``pick_moves_for_boards`` applies ``model.history_rep_fix`` on the way
    in and ``build_model`` applies the checkpoint's value, so any test that
    builds a model or runs a search sets it as a side effect. That makes the
    leak a cross-FILE ordering flake: a test passes or fails on a value some
    other file left behind, and the two files never mention each other.

    ⚑⚑ A HOOKWRAPPER, NOT AN AUTOUSE FIXTURE, AND THE DIFFERENCE IS THE WHOLE
    GUARANTEE. This was a function-scoped autouse fixture first, and a
    function-scoped fixture is set up AFTER every broader-scoped one — so a
    module- or session-scoped fixture that changes the flag has already changed
    it by the time the snapshot is taken, and the snapshot records the mutated
    value as if it were the baseline. MEASURED on
    ``pytest tests/test_param_count.py tests/test_action_decode_strict.py``,
    whose module-scoped ``production_model`` builds the production config
    (``history_rep_fix: true``): with the fixture, 31/31 tests ended holding
    ``True`` and the session exited holding ``True`` — the leak the guard claims
    to close, intact. ``pytest_runtest_protocol`` wraps the whole
    setup/call/teardown protocol for one item, so the snapshot precedes every
    fixture of every scope, and cross-file isolation stops depending on
    collection order.

    ⚑ The consequence, stated rather than discovered later: a broader-scoped
    fixture's flag mutation no longer survives into the NEXT test of its own
    module either — the fixture VALUE is still cached, only the process flag is
    rewound. Checked before choosing this granularity: no test in this suite
    relies on such persistence. The only higher-scoped fixture that turns the
    flag ON is ``test_param_count.py``'s ``production_model``, whose three
    consumers read ``state_dict``/``named_parameters`` and never encode a board;
    the other module-scoped model builders (``test_mcts_c_tree``,
    ``test_threaded_selfplay``, ``test_widen_ffn_aligned``, ``test_e2e_smoke``)
    all build with the flag OFF, which is what a rewind gives them anyway.
    ``test_rep_fix_autouse_restore.py`` pins both halves.

    ⚑ It is SENTINEL-BLIND: it observes ``rep_fix.current()``, so a direct poke
    at an extension's ``set_history_rep_fix`` bypasses it entirely — the C
    global is write-only from Python, so there is nothing to read back. Every
    such poke today lives in ``tests/test_history_rep_fix.py``, which repairs
    itself in ``_force_flag_off``; a new direct-poke author is outside this
    guard and must do the same.

    ⚑ Restoring passes ``boards_discarded=True`` because ``apply`` refuses a
    flip without it, and the keyword is TRUE here rather than a rubber stamp.
    Per ``rep_fix``'s module docstring, per-slot repetition flags are recorded
    at push time and never recomputed, so a ``CBoard`` carried across a flip
    encodes planes matching NEITHER regime (audit E3 measured
    ``[1,0,1,0,1,0,1,0]`` against ``[1,1,1,1,1,1,1,0]`` under either clean
    regime). This hook only acts when the finished test CHANGED the flag — and a
    board that outlived that test was already mis-encoded in the direction the
    test left it, before this hook ran. Restoring does not create that hazard;
    it replaces an order-dependent suite baseline with a deterministic one. The
    E3 guard still fires inside the test, where the live board is.

    ⚑ ``apply`` cannot express "never set" — it takes a ``bool``. So the
    never-set snapshot restores in two steps: put the extensions on their
    documented default (off), which is exactly the state a never-set flag leaves
    them in, then restore the module's own ``None`` sentinel so ``current()``
    reports the truth rather than a value nobody chose.

    ``rep_fix`` is imported lazily: a conftest is loaded on EVERY pytest
    invocation and this package pulls in numpy and chess, and an environment
    without the compiled encoders should reach ``apply`` (which skips missing
    setters with a warning) rather than fail at collection. On the common path,
    where the test left the flag alone, nothing is called at all. The hook takes
    no parameters because pluggy passes an implementation only the hookspec
    arguments it actually declares, and this one needs none.

    ⚑ ``hookwrapper=True`` (old style) rather than pytest 8's ``wrapper=True``,
    deliberately: ``pyproject.toml`` declares ``pytest>=7.0`` and the newer
    spelling does not exist there. Do not "modernize" this without raising the
    floor. The old style also gives the behaviour wanted here for free — a
    failing test does not raise at the ``yield``, so the restore runs whatever
    the test did.
    """
    from chess_anti_engine.encoding import rep_fix

    before = rep_fix.current()
    yield
    if rep_fix.current() is before:
        return
    rep_fix.apply(bool(before), boards_discarded=True)
    if before is None:
        rep_fix._current = None
