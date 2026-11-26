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

import atexit
import contextlib
import importlib.abc
import importlib.machinery
import os
import sys
import threading
import time
import types
import weakref
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.serial_phase import runs_in_serial_phase

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


# ⚑⚑ `CUDA_VISIBLE_DEVICES=` (empty) does NOT keep a process off the GPU on
# WSL2. `torch.cuda.is_available()` defaults to `cudaGetDeviceCount`, which runs
# `cuInit`, and on WSL2 the driver opens the adapter (/dev/dxg) BEFORE it
# applies the visibility mask -- and holds that fd for the life of the process.
# Measured 2026-11-22 with torch 2.10.0+cu128 and 2.11.0+cu128: with the
# variable empty, `is_available()` returns False and the process holds
# /dev/dxg; `device_count()` (the NVML path, which returns 0 for an empty mask
# before it even loads NVML) opens nothing. The suite reached the real device
# eight times from `tests/test_uci_selfcheck.py` alone (a real `check_torch`,
# and engine children started with `--device auto`). A test must never
# initialise the real GPU: on this box that has crashed training before.
#
# `PYTORCH_NVML_BASED_CUDA_CHECK=1` routes `is_available()` through that NVML
# path. torch reads it on EVERY `is_available()` call (`os.getenv` in
# `torch.cuda._nvml_based_avail`), not at import, so a plugin that imported
# torch before this line does not make it late. Children inherit it, like the
# thread caps below; a child started with an `env` built from scratch does not.
# Caveat: if NVML discovery/initialisation fails, torch falls back to
# `cudaGetDeviceCount` (and so to `cuInit`). With a GPU visible (the variable
# unset), NVML counts it and `is_available()` is still True. `setdefault`: an
# explicitly exported value (e.g. `0`) still wins.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
# ⚑ What this does NOT cover (accepted, 2026-11-22): on WSL2 the first CPU
# `backward()` makes the C++ autograd engine count CUDA devices
# (`cudaGetDeviceCount` -> `cuInit`), which opens /dev/dxg for the process
# lifetime even with `CUDA_VISIBLE_DEVICES=` empty and the NVML switch on;
# tests that run a backward pass therefore still touch the adapter; nothing is
# allocated (no visible device, no context). Reproduction:
#     x = torch.ones(3, requires_grad=True)
#     (x * 2).sum()            # no /dev/dxg fd yet
#     (x * 2).sum().backward() # /dev/dxg fd now held

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


_REPO_ROOT = Path(__file__).resolve().parents[1]


def change_scoped_deselection(items: list[pytest.Item]) -> list[pytest.Item]:
    """Items of the test files ``scripts/validate.py`` listed for deselection.

    The runner decides once (see ``validate.script_test_deselection``) and
    passes a file of repo-relative test paths in ``CAE_DESELECT_LIST``; xdist
    workers inherit it. Unset, missing or unreadable deselects nothing, so a
    direct or local run is unchanged.
    """
    listing = os.environ.get("CAE_DESELECT_LIST")
    if not listing:
        return []
    try:
        with open(listing, encoding="utf-8") as handle:
            deselected = {line.strip() for line in handle if line.strip()}
    except (OSError, ValueError):  # ValueError: undecodable bytes
        return []
    paths = {path: Path(path).resolve() for path in {item.path for item in items}}
    selected = {path for path, real in paths.items()
                if real.is_relative_to(_REPO_ROOT) and real.relative_to(_REPO_ROOT).as_posix() in deselected}
    return [item for item in items if item.path in selected]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    drop = change_scoped_deselection(items)
    if drop:
        dropped = set(map(id, drop))
        items[:] = [item for item in items if id(item) not in dropped]
        config.hook.pytest_deselected(items=drop)
    for item in items:
        if runs_in_serial_phase(item.path):
            item.add_marker(pytest.mark.serial)


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
    without the compiled encoders should fail in the test that needs them
    (``apply`` raises ``RepFixUnavailableError`` on a missing setter) rather
    than at collection. On the common path,
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


@pytest.fixture
def python_reference_paths() -> Iterator[None]:
    """Opt this test in to the Python reference paths (``ext_check``).

    Production refuses to substitute a Python search, per-ply recorder or
    encoder for a missing C symbol. Tests that deliberately exercise those
    reference implementations (by forcing ``has_c_ply=False``, patching
    ``_HAS_GUMBEL_C``/``_HAS_C_TREE`` off, or deleting an extension symbol to
    test the reference composition) request this fixture.
    """
    from chess_anti_engine.ext_check import python_reference_paths as _opt_in

    with _opt_in("test exercises the Python reference path"):
        yield



# ── Test runs must not write into the checkout ───────────────────────────────
#
# Two production defaults are relative to the checkout and are right for
# production but wrong for a test process:
#
# * ``Trainer`` writes ``SummaryWriter(log_dir=str(log_dir or "tb"))``: with no
#   ``log_dir`` the events go to ``./tb`` in the CURRENT DIRECTORY, i.e. the
#   checkout pytest runs from (127 event files per full-suite run). Production
#   always passes ``log_dir`` (``tune/trainable.py`` -> ``<work_dir>/tb``).
# * ``scripts/arena_standard.py`` banks every game under
#   ``DEFAULT_GAME_LOG_DIR = REPO_ROOT / "runs" / "arena_games"``; a test that
#   drives ``run_arena`` without patching it appends to ``runs/`` in the checkout.
#
# The defaults are left alone; only the TEST process redirects them to a
# session temp dir. An explicit ``log_dir`` passes through untouched, and a test
# that monkeypatches ``DEFAULT_GAME_LOG_DIR`` itself still wins.
#
# Each patch is applied when its module is imported (a meta-path finder wraps
# the loader), or at once if it already is, so a test that imports the module
# lazily is covered and a test that never touches it pays nothing.

_TRAINER_MODULE = "chess_anti_engine.train.trainer"
_ARENA_MODULE = "scripts.arena_standard"
_TB_DEFAULT = "tb"
_checkout_redirect_root: Path | None = None


def _redirect_trainer_default_tb(module: types.ModuleType) -> None:
    real = getattr(module, "SummaryWriter")
    if getattr(real, "_cae_default_tb_redirect", False):
        return

    def _summary_writer(*args: object, **kwargs: object) -> object:
        root = _checkout_redirect_root
        if root is not None and kwargs.get("log_dir") == _TB_DEFAULT and not args:
            kwargs["log_dir"] = str(root / "trainer_default_tb")
        return real(*args, **kwargs)

    setattr(_summary_writer, "_cae_default_tb_redirect", True)
    setattr(module, "SummaryWriter", _summary_writer)


def _redirect_arena_game_log_dir(module: types.ModuleType) -> None:
    if _checkout_redirect_root is not None:
        setattr(module, "DEFAULT_GAME_LOG_DIR", _checkout_redirect_root / "arena_games")


_CHECKOUT_REDIRECTS = {
    _TRAINER_MODULE: _redirect_trainer_default_tb,
    _ARENA_MODULE: _redirect_arena_game_log_dir,
}


class _CheckoutRedirectImportHook(importlib.abc.MetaPathFinder):
    """Meta-path finder: patch a redirected module right after it executes."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del target
        patch = _CHECKOUT_REDIRECTS.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        original_exec = loader.exec_module

        def _exec_module(module: types.ModuleType) -> None:
            original_exec(module)
            patch(module)

        setattr(loader, "exec_module", _exec_module)
        return spec


@pytest.fixture(scope="session", autouse=True)
def _test_writes_stay_outside_the_checkout(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    global _checkout_redirect_root
    _checkout_redirect_root = tmp_path_factory.mktemp("checkout_redirects")
    hook = _CheckoutRedirectImportHook()
    sys.meta_path.insert(0, hook)
    for name, patch in _CHECKOUT_REDIRECTS.items():
        if name in sys.modules:
            patch(sys.modules[name])
    try:
        yield
    finally:
        # Untested on purpose: it runs after the last test, and a missed removal
        # is harmless. It only matters for an in-process re-run (pytest.main
        # twice), where a second hook would stack in front of the first. The
        # import system stops at the first finder that returns a spec, so only
        # one hook ever wraps a loader, and both patches are idempotent: the
        # trainer patch checks its `_cae_default_tb_redirect` marker, and the
        # arena patch assigns a constant.
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(hook)


class InProcessHardExit(BaseException):
    """``os._exit`` was called from the pytest process itself.

    A ``BaseException`` so ``except Exception`` in the code under test cannot
    swallow it. Raised instead of exiting: a real ``os._exit`` in this process
    ends the whole run with the status the caller chose -- ``0`` for a clean
    UCI ``quit`` -- so the run would look green with every later test unrun."""


# The pid of the pytest process (an xdist worker is its own), set by
# ``pytest_configure``, which only a pytest process runs. Not taken at import:
# a ``spawn`` multiprocessing child that imports a test module imports this
# conftest afresh, and must not mistake itself for the pytest process.
_PYTEST_PID: int | None = None
_REAL_OS_EXIT = os._exit
_INTERCEPTED_HARD_EXITS: list[str] = []
_SESSION_STATE = {"started": False, "finished": False}


def _guarded_os_exit(status: int) -> None:
    # Forked children (multiprocessing, os.fork in tests) inherit this wrapper
    # but have their own pid, and spawned children never ran pytest_configure:
    # both get the real exit.
    if _PYTEST_PID is None or os.getpid() != _PYTEST_PID:
        _REAL_OS_EXIT(status)
    where = os.environ.get("PYTEST_CURRENT_TEST", "<outside any test>")
    _INTERCEPTED_HARD_EXITS.append(f"os._exit({status}) in {where}")
    raise InProcessHardExit(
        f"os._exit({status}) called inside the pytest process ({where}); it would "
        "have ended the run silently. Stub the caller (see _reset_uci_wedge_latch)."
    )


os._exit = _guarded_os_exit


def pytest_configure() -> None:
    global _PYTEST_PID
    _PYTEST_PID = os.getpid()


def pytest_sessionstart() -> None:
    _SESSION_STATE["started"] = True


_EXIT_MARK = pytest.StashKey[int]()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    item.stash[_EXIT_MARK] = len(_INTERCEPTED_HARD_EXITS)


@pytest.hookimpl(wrapper=True, tryfirst=True)  # outermost: sees the final xfail/skip outcome
def pytest_runtest_makereport(item: pytest.Item):
    """Fail the offending test's own report when an ``os._exit`` was
    intercepted during it (setup, call or teardown), even if the code under
    test swallowed the ``InProcessHardExit``. The session-level check below
    cannot do this under xdist: a worker's ``session.exitstatus`` never reaches
    the controller, so a swallowed exit there read as a green run."""
    report = yield
    new = _INTERCEPTED_HARD_EXITS[item.stash.get(_EXIT_MARK, len(_INTERCEPTED_HARD_EXITS)):]
    item.stash[_EXIT_MARK] = len(_INTERCEPTED_HARD_EXITS)
    # Anything but an existing failure: a pass, a skip raised after the
    # swallowed exit, or an xfail-marked test that then failed (XFAIL reads green).
    if new and not report.failed:
        report.outcome = "failed"
        # pytest counts a failure only without ``wasxfail`` (_pytest/main.py):
        # left in place, a flipped xfail reads FAILED yet the run exits 0.
        if hasattr(report, "wasxfail"):
            del report.wasxfail
        report.longrepr = (
            "conftest: intercepted in-process " + "; ".join(new)
            + " (the code under test swallowed InProcessHardExit; outside pytest this "
            "would have ended the process)"
        )
    return report


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Fail the run if any test reached ``os._exit`` in-process, even when the
    code under test caught the ``InProcessHardExit`` it raised."""
    _SESSION_STATE["finished"] = True
    if _INTERCEPTED_HARD_EXITS and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    for line in _INTERCEPTED_HARD_EXITS:
        sys.stderr.write(f"conftest: intercepted in-process {line}\n")


def _check_session_reached_its_end() -> None:
    # A run that started but is exiting without ``pytest_sessionfinish`` was
    # ended from inside (an exit path that bypassed pytest): never let it
    # leave with a status that reads as success.
    if (_SESSION_STATE["started"] and not _SESSION_STATE["finished"]
            and os.getpid() == _PYTEST_PID):
        sys.stderr.write("conftest: the pytest session ended before pytest_sessionfinish; "
                         "exiting 3\n")
        sys.stderr.flush()
        _REAL_OS_EXIT(3)


atexit.register(_check_session_reached_its_end)


@pytest.fixture(autouse=True)
def _reset_uci_wedge_latch():
    """Each test is a fresh engine session for the multi-GPU wedge latch
    (``chess_anti_engine.uci.wedge``): a device wedged or failed in one test
    must not be left out of, or make workers park in, the next. Imports no torch.

    In-process ``uci.__main__.main()`` calls: after a device fault the real
    ``quit`` leaves with ``os._exit`` (no interpreter teardown), which would end
    the pytest process. While the module is loaded, ``_hard_exit`` becomes a
    recorder that returns, so ``main()`` falls through to its ordinary close
    and returns its status. The real exit is covered by the subprocess tests
    (``tests/test_uci_degrade_gaps_subprocess.py``). A module imported only
    inside a test escapes this stub; the module-level ``os._exit`` guard above
    then raises ``InProcessHardExit`` and fails the run instead of ending it."""
    import sys

    from chess_anti_engine.uci import wedge

    # Patched by hand, not via ``monkeypatch``: an autouse fixture requesting
    # ``monkeypatch`` reorders its teardown against other fixtures (it undid a
    # test's ``signal.signal`` patch only after that test's handler restore).
    uci_main = sys.modules.get("chess_anti_engine.uci.__main__")
    real_hard_exit = getattr(uci_main, "_hard_exit", None) if uci_main is not None else None
    if real_hard_exit is not None:
        setattr(uci_main, "_hard_exit", lambda rc: None)
    wedge.reset_session_for_tests()
    try:
        yield
    finally:
        wedge.reset_session_for_tests()
        if real_hard_exit is not None:
            setattr(uci_main, "_hard_exit", real_hard_exit)
