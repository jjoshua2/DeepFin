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
import time
from pathlib import Path

import pytest

from chess_anti_engine.stockfish.uci import (
    CAE_ENGINE_MARKER_ENV,
    _child_env,
    _child_reap_guard,
)
from chess_anti_engine.tune.process_cleanup import (
    list_pids_with_env,
    terminate_engines_owned_by,
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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is Linux-only")
def test_pdeathsig_kills_the_engine_when_its_parent_is_SIGKILLed() -> None:
    """THE headline scenario, end to end.

    A worker is SIGKILLed (or `os._exit`s from a hang watchdog) — no `finally`
    runs, so nothing closes the engine. Without PDEATHSIG the engine is
    reparented to init and lives forever. Driven with a real intermediate
    process so the death is the kernel's doing, not the test's.
    """
    parent_code = (
        "import subprocess, sys, time;"
        "from chess_anti_engine.stockfish.uci import _child_reap_guard;"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],"
        "                     start_new_session=True, preexec_fn=_child_reap_guard);"
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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is Linux-only")
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


def test_the_guard_never_raises() -> None:
    """It runs between fork and exec in the child; an exception there fails the
    spawn. It must degrade to a no-op on a platform without prctl."""
    _child_reap_guard()  # in-process, parent is alive; must simply return


def test_libc_is_resolved_before_the_fork_not_inside_the_child() -> None:
    """The guard runs post-fork, where only async-signal-safe work is legal, and
    `dlopen` is not: a loader lock held by another thread at the instant of the
    fork is inherited HELD by a child that can never release it, and the spawn
    hangs forever. So the handle must already exist at import.
    """
    import ast

    from chess_anti_engine.stockfish import uci as uci_mod

    assert uci_mod._LIBC is not None, "libc did not load on a Linux test host"

    tree = ast.parse(Path(uci_mod.__file__).read_text())
    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_child_reap_guard"
    )
    loads = [
        node
        for node in ast.walk(guard)
        if isinstance(node, ast.Call) and "CDLL" in ast.unparse(node.func)
    ]
    assert not loads, "the guard dlopens after fork — that can deadlock the spawn"


def test_the_guard_is_a_no_op_when_libc_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-glibc/non-Linux: no prctl, but the spawn must still succeed — the env
    marker alone still makes the child findable."""
    from chess_anti_engine.stockfish import uci as uci_mod

    monkeypatch.setattr(uci_mod, "_LIBC", None)
    uci_mod._child_reap_guard()


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
        # have not waited, so the entry must still be there in state Z. (Read
        # /proc rather than proc.poll(), which would reap it.)
        stat = Path(f"/proc/{proc.pid}/stat").read_bytes()
        close = stat.rfind(b")")
        assert stat[close + 2 : close + 3] == b"Z", stat[:80]
    finally:
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
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
    assert kwargs.get("preexec_fn") == "_child_reap_guard", kwargs
    assert kwargs.get("env") == "_child_env()", kwargs


def test_close_kills_the_process_group() -> None:
    """`close()` must target the group; killing the pid alone leaves anything
    the engine spawned behind, which is the whole reason for the new session."""
    from chess_anti_engine.stockfish import uci as uci_mod

    src = Path(uci_mod.__file__).read_text()
    close_at = src.index("    def close(self) -> None:")
    body = src[close_at: close_at + 1400]
    assert "os.killpg(" in body, "close() does not kill the process group"


def test_the_worker_stop_path_reaps_orphaned_engines() -> None:
    """PDEATHSIG is thread-scoped and Linux-only, so the supervisor keeps a
    belt: after stopping a worker, anything still carrying its pid is an orphan."""
    from chess_anti_engine.tune import distributed_runtime as dr

    src = Path(dr.__file__).read_text()
    stop_at = src.index("def _stop_worker_processes(")
    body = src[stop_at: stop_at + 1200]
    assert "terminate_engines_owned_by(" in body, (
        "_stop_worker_processes does not reap the stopped worker's engines"
    )
