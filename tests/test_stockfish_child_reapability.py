"""Audit R2: a Stockfish engine must not survive the death of its worker.

Before this, an engine was spawned with no process group, carried nothing that
identified it, and was closed only through a `finally` that `os._exit` (both
hang watchdogs) and SIGKILL both skip. Its cmdline is the bare binary path, so
`terminate_matching_processes` could not match it whatever `reap_terms` it was
given, and once orphaned its ancestry was gone too. The result survived until
reboot holding ~2.6 GB RSS, eight per worker.

⚑ NO CUDA AND NO REAL STOCKFISH. Every process test drives a tiny `sleep`-style
stub through the real spawn options, so the mechanisms — process group,
PDEATHSIG, env marker — are exercised for real without the engine binary or a
GPU.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from chess_anti_engine.stockfish.uci import (
    CAE_ENGINE_MARKER_ENV,
    StockfishUCI,
    _child_env,
)
from chess_anti_engine.tune.process_cleanup import (
    list_pids_with_env,
    terminate_engines_owned_by,
)

# ⚑ MODULE-WIDE, not per-test. Every mechanism here is Linux-only -- prctl for
# PDEATHSIG, procfs for the env marker and the zombie state -- and even the
# AST-only tests are about Linux-only code. Two tests carried this decorator
# individually and three later ones quietly did not, which is how a non-Linux
# contributor gets failures where the convention promised skips. One marker
# means a new test cannot forget it.
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="prctl/procfs are Linux-only",
)

_STUB = "import time; time.sleep(600)"


def _spawn_stub(env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    """A stand-in engine, spawned exactly as `uci.py` spawns the real one."""
    return subprocess.Popen(
        [sys.executable, "-c", _STUB], start_new_session=True, env=env,
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A zombie is not alive for our purposes; it has released its memory.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _wait_gone(pid: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _assert_is_zombie(pid: int) -> None:
    """Require ``pid`` to be a real unreaped corpse, state ``Z``.

    ⚑ `_wait_gone` returns True for BOTH "zombie" and "fully gone", so a test
    that only waits and then asserts "not alive" would pass having exercised
    nothing: the whole difference between the two `_pid_exists` implementations
    exists only while the entry is in state `Z`. Any test whose subject is the
    zombie path has to say so out loud rather than rely on an unstated fact
    about its own runner (that pytest is the parent and has not waited).
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError as exc:  # the entry is gone: reaped, or never existed
        raise AssertionError(
            f"pid {pid} has no /proc entry, so it was fully reaped rather than "
            f"left as a zombie -- the zombie path was never exercised ({exc})",
        ) from exc
    close = stat.rfind(b")")
    assert stat[close + 2 : close + 3] == b"Z", (
        f"pid {pid} is not an unreaped zombie, so the zombie path was never "
        f"exercised: {stat[:80]!r}"
    )


# ── process group ────────────────────────────────────────────────────────────


def test_start_new_session_gives_the_engine_its_own_group() -> None:
    """The property `close()`'s group-kill depends on. Without it the engine
    shares the worker's group and killing that group would kill the worker."""
    proc = _spawn_stub()
    try:
        assert os.getpgid(proc.pid) == proc.pid, "engine is not its own group leader"
        assert os.getpgid(proc.pid) != os.getpgid(os.getpid())
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)


def test_killing_the_group_kills_the_engine_and_its_own_children() -> None:
    """Why the kill targets the group and not the pid: it is then correct by
    construction rather than by the assumption that Stockfish never forks."""
    code = (
        "import subprocess, sys, time;"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']);"
        "print(c.pid, flush=True); time.sleep(600)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    grandchild = int(proc.stdout.readline().strip())
    try:
        assert _alive(grandchild)
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        assert _wait_gone(proc.pid), "engine survived the group kill"
        assert _wait_gone(grandchild), "grandchild survived the group kill"
    finally:
        for pid in (proc.pid, grandchild):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        proc.wait(timeout=10)


def test_a_group_kill_cannot_reach_the_parent() -> None:
    """The regression this must not introduce. `start_new_session` exists so the
    kill is precise; if the engine were still in our group, `close()` would take
    the worker down with it."""
    proc = _spawn_stub()
    try:
        assert os.getpgid(proc.pid) != os.getpid()
        assert os.getpgid(proc.pid) != os.getpgid(os.getpid())
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)


# ── PDEATHSIG ────────────────────────────────────────────────────────────────


def test_pdeathsig_kills_the_engine_when_its_parent_is_SIGKILLed() -> None:
    """THE headline scenario, end to end.

    A worker is SIGKILLed (or `os._exit`s from a hang watchdog) — no `finally`
    runs, so nothing closes the engine. Without PDEATHSIG the engine is
    reparented to init and lives forever. Driven with a real intermediate
    process so the death is the kernel's doing, not the test's.
    """
    parent_code = (
        "import subprocess, sys, time;"
        "import os, functools;"
        "from chess_anti_engine.stockfish.uci import _child_reap_guard;"
        "g = functools.partial(_child_reap_guard, os.getpid());"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],"
        "                     start_new_session=True, preexec_fn=g);"
        "print(c.pid, flush=True); time.sleep(600)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdout=subprocess.PIPE, env=env,
    )
    assert parent.stdout is not None
    engine_pid = int(parent.stdout.readline().strip())
    try:
        assert _alive(engine_pid)
        parent.kill()  # SIGKILL: no finally, no close(), no cleanup
        parent.wait(timeout=10)
        assert _wait_gone(engine_pid), (
            "the engine outlived its SIGKILLed parent — PDEATHSIG did not fire, "
            "and this is exactly the orphan that survives until reboot"
        )
    finally:
        with contextlib.suppress(OSError):
            os.kill(engine_pid, signal.SIGKILL)


def test_without_the_guard_the_child_is_orphaned() -> None:
    """Negative control for the test above. Same harness, no `preexec_fn` — the
    child MUST survive, or the previous test proves nothing about PDEATHSIG."""
    parent_code = (
        "import subprocess, sys, time;"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],"
        "                     start_new_session=True);"
        "print(c.pid, flush=True); time.sleep(600)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdout=subprocess.PIPE,
    )
    assert parent.stdout is not None
    engine_pid = int(parent.stdout.readline().strip())
    try:
        parent.kill()
        parent.wait(timeout=10)
        time.sleep(1.0)
        assert _alive(engine_pid), "control failed: the child died without PDEATHSIG"
    finally:
        with contextlib.suppress(OSError):
            os.kill(engine_pid, signal.SIGKILL)


def test_the_guard_does_no_loader_work_and_never_raises() -> None:
    """The guard must be a bare call through an ALREADY-RESOLVED pointer.

    Two properties in one measurement, because they have the same cause. It runs
    post-fork, where only async-signal-safe work is legal: `dlopen` AND `dlsym`
    take the loader lock, and a lock held by another thread at the instant of
    the fork is inherited HELD by a child that can never release it — the spawn
    then hangs forever. Loading the library at import is only half of that,
    because `ctypes.CDLL.__getattr__` resolves a symbol lazily on first access
    and caches it ON THE INSTANCE: a lookup inside the guard happens in the
    child, so the parent's cache never warms and it recurs on EVERY spawn.

    So this instruments `CDLL.__getattr__` and requires the guard to trigger it
    ZERO times — the reviewer's own measurement, turned into a test.

    ⚑ Runs in a subprocess deliberately. Calling the guard in-process arms
    `PR_SET_PDEATHSIG=SIGKILL` on the pytest runner itself and never disarms it,
    so any later parent-exit would SIGKILL the suite mid-run.
    """
    probe = (
        "import ctypes, os, sys;"
        "import chess_anti_engine.stockfish.uci as u;"
        "assert u._PRCTL is not None, 'libc did not load on a Linux host';"
        "calls = [];"
        "orig = ctypes.CDLL.__getattr__;"
        "ctypes.CDLL.__getattr__ = lambda self, name: (calls.append(name), orig(self, name))[1];"
        "u._child_reap_guard(os.getppid());"  # our real parent: the guard must not self-exit
        "print('LOOKUPS', calls)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env,
        timeout=60, check=False,
    )
    assert done.returncode == 0, done.stderr  # the guard must never raise
    assert "LOOKUPS []" in done.stdout, (
        f"the guard did loader work after fork: {done.stdout.strip()} {done.stderr}"
    )


def test_the_guard_calls_prctl_through_the_module_level_pointer() -> None:
    """Structural companion: nothing in the guard may touch ctypes at all. The
    behavioural test above would not notice a NEW `CDLL(...)` load whose symbol
    happened to be pre-cached, and this is the cheaper regression tripwire."""
    import ast

    from chess_anti_engine.stockfish import uci as uci_mod

    tree = ast.parse(Path(uci_mod.__file__).read_text())
    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_child_reap_guard"
    )
    touches = [
        ast.unparse(node)
        for node in ast.walk(guard)
        if isinstance(node, ast.Name | ast.Attribute) and "ctypes" in ast.unparse(node)
    ]
    assert not touches, f"the guard does loader work after fork: {touches}"


def test_the_guard_is_a_no_op_when_prctl_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-glibc/non-Linux: no prctl, but the spawn must still succeed — the env
    marker alone still makes the child findable. (Safe in-process: with `_PRCTL`
    None the guard returns before arming anything on the runner.)"""
    from chess_anti_engine.stockfish import uci as uci_mod

    monkeypatch.setattr(uci_mod, "_PRCTL", None)
    uci_mod._child_reap_guard(os.getppid())

    # And prove it armed nothing on the RUNNER. An in-process guard call that
    # reached prctl would leave PDEATHSIG=SIGKILL set on pytest forever, so any
    # later parent-exit SIGKILLs the suite mid-run.
    import ctypes

    out = ctypes.c_int(0)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(2, ctypes.byref(out), 0, 0, 0)  # 2 == PR_GET_PDEATHSIG
    assert out.value == 0, f"pytest is left armed with PDEATHSIG={out.value}"


def test_the_guard_exits_when_the_parent_is_already_gone() -> None:
    """H4: the re-check must compare against the pid captured BEFORE the fork.

    Comparing against a `getppid()` read inside the child closes only half the
    race — if the parent died first, that read already returns 1, the comparison
    can never differ, and the child arms PDEATHSIG against init and lives
    forever. Simulated by passing an `expected_parent` we are demonstrably not
    the child of, which is exactly the post-reparent state.
    """
    probe = (
        "import os, sys;"
        "import chess_anti_engine.stockfish.uci as u;"
        "u._child_reap_guard(999999);"
        "print('SURVIVED')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env,
        timeout=60, check=False,
    )
    assert "SURVIVED" not in done.stdout, (
        "the guard armed PDEATHSIG against a parent it does not have and "
        "carried on — this is the orphan it exists to prevent"
    )
    assert done.returncode == 0, done.stderr  # os._exit(0), not a crash


# ── the env marker, and why it is the reaper's key ───────────────────────────


def test_the_marker_names_the_owning_process() -> None:
    """Per-process, not per-trial: a box runs several workers, and reaping after
    one dead worker must not touch a live worker's engines."""
    env = _child_env()
    assert env[CAE_ENGINE_MARKER_ENV] == str(os.getpid())
    assert env["PATH"] == os.environ["PATH"], "inherited environment was dropped"


def test_an_orphan_is_findable_by_marker_when_ancestry_is_gone() -> None:
    """THE reason the key is the environment.

    An orphan's ppid is 1 and its pgid is its own, so ancestry carries no
    information — and its cmdline is the engine binary's, which is why
    `terminate_matching_processes` could never match it. `/proc/<pid>/environ`
    is fixed at exec and survives being orphaned.
    """
    owner = 4242424
    env = dict(os.environ)
    env[CAE_ENGINE_MARKER_ENV] = str(owner)
    proc = _spawn_stub(env)
    try:
        found = list_pids_with_env(CAE_ENGINE_MARKER_ENV, str(owner))
        assert proc.pid in found, found
        # And it is not confusable with an engine owned by a live worker.
        assert list_pids_with_env(CAE_ENGINE_MARKER_ENV, "9999999") == []
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)


def test_terminate_engines_owned_by_kills_only_the_matching_owner() -> None:
    """The isolation property the reaper turns on. Killing indiscriminately
    would take out a healthy concurrent worker's engines."""
    mine, theirs = 4242001, 4242002
    procs = {}
    for owner in (mine, theirs):
        env = dict(os.environ)
        env[CAE_ENGINE_MARKER_ENV] = str(owner)
        procs[owner] = _spawn_stub(env)
    try:
        killed = terminate_engines_owned_by(mine)
        assert killed == [procs[mine].pid], killed
        assert _wait_gone(procs[mine].pid)
        assert _alive(procs[theirs].pid), "reaped another owner's engine"
    finally:
        for proc in procs.values():
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)


def test_a_reaped_engine_is_reported_killed_even_as_an_unwaited_zombie() -> None:
    """`os.kill(pid, 0)` succeeds against a zombie, so the reaper's liveness
    check had to learn to read the process state: a corpse has released its
    ~2.6 GB and the reaper's entire purpose is that RSS. Without this it burns
    its full timeout and then reports the kill as FAILED — and the caller logs
    a warning about engines it did in fact reap.
    """
    owner = 4242003
    env = dict(os.environ)
    env[CAE_ENGINE_MARKER_ENV] = str(owner)
    proc = _spawn_stub(env)
    try:
        started = time.monotonic()
        killed = terminate_engines_owned_by(owner, timeout_s=5.0)
        elapsed = time.monotonic() - started
        # This test IS the parent, so nothing has reaped the zombie yet.
        assert killed == [proc.pid], killed
        assert elapsed < 2.0, f"waited {elapsed:.1f}s on a corpse"
        # Prove the zombie path was the one exercised: we are the parent and
        # have not waited, so the entry must still be there in state Z. (Reads
        # /proc rather than proc.poll(), which would reap it.)
        _assert_is_zombie(proc.pid)
    finally:
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)


def test_graceful_restarts_pid_probe_agrees_about_zombies() -> None:
    """`scripts/graceful_restart.py` carries its own same-named `_pid_exists` —
    it is documented as a no-PYTHONPATH invocation, so it must stay stdlib-only
    and cannot import the shared helper. Two same-named helpers with opposite
    semantics is how the next reader gets it wrong, so pin that they agree: on a
    zombie tuner the old answer burns the full 30s wait and then reports "did
    not exit after SIGTERM" about a process that did exit.
    """
    import importlib.util

    from chess_anti_engine.tune.process_cleanup import _pid_exists as shared

    spec = importlib.util.spec_from_file_location(
        "_gr_probe", Path(__file__).resolve().parents[1] / "scripts" / "graceful_restart.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    proc = _spawn_stub()
    proc.kill()
    _wait_gone(proc.pid)  # dead but unwaited: a zombie, because we are its parent
    try:
        # ⚑ Say it out loud. `_wait_gone` is true for "zombie" AND for "fully
        # gone", and on a vanished entry BOTH implementations return False --
        # so without this the assertions below would pass having tested
        # nothing. The divergence being pinned exists only in state Z.
        _assert_is_zombie(proc.pid)
        assert shared(proc.pid) is False
        assert module._pid_exists(proc.pid) is False, (
            "graceful_restart still reports a zombie as alive"
        )
        assert shared(os.getpid()) is True
        assert module._pid_exists(os.getpid()) is True
    finally:
        proc.wait(timeout=10)


def test_reaping_an_owner_with_no_engines_is_a_no_op() -> None:
    assert terminate_engines_owned_by(999_999_9) == []


def test_unmarked_processes_are_never_matched() -> None:
    """Negative control on the scan itself: an empty or absent marker must not
    match everything, which would make the reaper a machine-wide kill."""
    assert list_pids_with_env(CAE_ENGINE_MARKER_ENV, "") == []
    assert list_pids_with_env("", "1234") == []


# ── the spawn site really passes these options ───────────────────────────────


def test_the_engine_spawn_uses_all_three_mechanisms() -> None:
    """Structural guard. Every test above exercises the mechanisms directly, so
    all of them would still pass if `Popen` in `uci.py` had been left untouched
    — the "gate that cannot fail" shape.
    """
    import ast

    from chess_anti_engine.stockfish import uci as uci_mod

    tree = ast.parse(Path(uci_mod.__file__).read_text())
    spawns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "subprocess.Popen"
    ]
    assert len(spawns) == 1, f"expected one engine spawn, found {len(spawns)}"
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in spawns[0].keywords}
    assert kwargs.get("start_new_session") == "True", kwargs
    assert kwargs.get("preexec_fn", "").startswith("partial("), kwargs
    assert "_child_reap_guard" in kwargs.get("preexec_fn", ""), kwargs
    assert kwargs.get("env") == "_child_env()", kwargs


# ── close() is re-entrant, and must not signal a reaped pid ──────────────────


class _ClosableEngine:
    """The REAL `StockfishUCI.close`, bound to a stub with a real child process.

    Constructing a `StockfishUCI` needs the Stockfish binary and a UCI
    handshake; this exercises the shipped `close` body itself against a sleep
    stub, so the test is about production code rather than a re-implementation.
    """

    close = StockfishUCI.close

    def __init__(self) -> None:
        self.proc = _spawn_stub()
        self._pgid = self.proc.pid  # what __init__ records, for the same reason
        self._lock = threading.Lock()
        self._tty_fd = os.open(os.devnull, os.O_RDONLY)
        self.sent: list[str] = []

    def _send(self, line: str) -> None:
        self.sent.append(line)


def test_close_kills_the_engine_and_is_safe_to_call_twice() -> None:
    """Double-close is reachable by construction: `_replace_engine`'s
    already-swapped branch closes an engine a previous call already closed.
    `Popen.kill()` — what the group kill replaced — was unconditionally safe on
    a reaped process, and that property must not be lost."""
    engine = _ClosableEngine()
    pid = engine.proc.pid
    engine.close()
    assert _wait_gone(pid), "close() did not kill the engine"
    engine.close()  # must not raise, and must not signal anything
    engine.close()


def test_close_does_not_signal_a_pid_it_has_already_reaped() -> None:
    """THE regression H1 names. After `wait()` the kernel may recycle the pid,
    so a second `close()` that signals it can SIGKILL a LIVE STRANGER's whole
    process group as our uid — the trainer and the server are in range. The
    guard is `poll()`, exactly what `Popen.send_signal` does.
    """
    engine = _ClosableEngine()
    engine.close()
    assert engine.proc.poll() is not None, "the child was not reaped by close()"

    signalled: list[tuple[int, int]] = []
    real_killpg = os.killpg
    try:
        os.killpg = lambda pgid, sig: signalled.append((pgid, sig))  # type: ignore[assignment]
        engine.close()
    finally:
        os.killpg = real_killpg
    assert signalled == [], f"close() signalled a reaped pid: {signalled}"


def test_the_recorded_pgid_is_the_child_pid_and_is_never_looked_up() -> None:
    """`start_new_session=True` makes the child a group leader, so its pgid IS
    its pid. Recording it at spawn is what removes the `os.getpgid` round-trip
    on a pid that may no longer be ours."""
    engine = _ClosableEngine()
    try:
        assert engine._pgid == engine.proc.pid
        assert os.getpgid(engine.proc.pid) == engine._pgid
    finally:
        engine.close()


def test_the_spawn_records_the_pgid_it_will_later_kill() -> None:
    """Structural: the tests above build `_pgid` themselves, so they would all
    pass if `__init__` never recorded it."""
    import ast

    from chess_anti_engine.stockfish import uci as uci_mod

    tree = ast.parse(Path(uci_mod.__file__).read_text())
    init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        and any("subprocess.Popen" in ast.unparse(c) for c in ast.walk(node))
    )
    assigns = [
        ast.unparse(node)
        for node in ast.walk(init)
        if isinstance(node, ast.Assign) and "self._pgid" in ast.unparse(node.targets[0])
    ]
    assert assigns == ["self._pgid = self.proc.pid"], assigns


def test_the_pgid_is_recorded_immediately_after_the_spawn() -> None:
    """The POSITION is load-bearing, not just the assignment.

    The UCI handshake later in `__init__` ends in
    `except BaseException: self.close()`, and `close()` reads `self._pgid`. Any
    statement that can raise between the spawn and this assignment turns a
    handshake failure into an `AttributeError` raised from inside the failure
    handler -- the original error lost, and the engine leaked. Pinned so a
    reorder fails here instead of at 3am.
    """
    import ast

    from chess_anti_engine.stockfish import uci as uci_mod

    tree = ast.parse(Path(uci_mod.__file__).read_text())
    init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        and any("subprocess.Popen" in ast.unparse(c) for c in ast.walk(node))
    )
    body = [ast.unparse(stmt) for stmt in init.body]
    spawn = next(i for i, stmt in enumerate(body) if "subprocess.Popen(" in stmt)
    assert body[spawn + 1] == "self._pgid = self.proc.pid", (
        "the pgid must be recorded in the statement immediately after the "
        f"spawn; found {body[spawn + 1]!r}"
    )


def _calls_in(path: str, func_name: str) -> list[str]:
    """Every call expression inside a named function, as source text.

    ⚑ AST, not substring. A `"…" in <source slice>` assertion passes on a
    COMMENTED-OUT line — `# os.killpg(` contains `os.killpg(` — so it cannot
    tell "the code runs" from "the text is present". Parsing means only real
    call nodes count.
    """
    import ast

    tree = ast.parse(Path(path).read_text())
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )
    return [
        ast.unparse(node.func)
        for node in ast.walk(target)
        if isinstance(node, ast.Call)
    ]


def test_close_kills_the_process_group() -> None:
    """`close()` must target the group; killing the pid alone leaves anything
    the engine spawned behind, which is the whole reason for the new session."""
    from chess_anti_engine.stockfish import uci as uci_mod

    calls = _calls_in(uci_mod.__file__, "close")
    assert "os.killpg" in calls, f"close() does not kill the process group: {calls}"
    assert "os.getpgid" not in calls, (
        "close() looks up the pgid at close time — on a reaped-and-recycled pid "
        "that is a stranger's group. Use the pgid recorded at spawn."
    )


def test_the_worker_stop_path_reaps_orphaned_engines() -> None:
    """PDEATHSIG is thread-scoped and Linux-only, so the supervisor keeps a
    belt: after stopping a worker, anything still carrying its pid is an orphan."""
    from chess_anti_engine.tune import distributed_runtime as dr

    calls = _calls_in(dr.__file__, "_stop_worker_processes")
    assert "terminate_engines_owned_by" in calls, (
        f"_stop_worker_processes does not reap the stopped worker's engines: {calls}"
    )
