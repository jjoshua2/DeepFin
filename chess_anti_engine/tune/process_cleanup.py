from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path

from chess_anti_engine.stockfish.uci import CAE_ENGINE_MARKER_ENV

_PROCFS = Path("/proc")


def _list_matching_pids(
    *,
    module: str,
    required_terms: Iterable[str],
    ps_output: str | None = None,
    exclude_pids: Iterable[int] = (),
) -> list[int]:
    terms = [str(term) for term in required_terms if str(term)]
    excluded = {int(pid) for pid in exclude_pids}
    excluded.add(int(os.getpid()))

    if ps_output is None:
        try:
            ps_output = subprocess.check_output(
                ["ps", "-eo", "pid=,args="],
                text=True,
            )
        except (subprocess.CalledProcessError, OSError):
  # ps unavailable or failed — nothing we can match against
            return []

    matches: list[int] = []
    for raw_line in str(ps_output).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_raw, _, cmd = line.partition(" ")
        try:
            pid = int(pid_raw)
        except ValueError:
  # header row or malformed ps output — skip
            continue
        if pid in excluded:
            continue
        if module not in cmd:
            continue
        if not all(term in cmd for term in terms):
            continue
        matches.append(pid)
    return matches


def _pid_env(pid: int) -> dict[str, str]:
    """Read ``/proc/<pid>/environ``. Empty on any failure."""
    try:
        raw = (_PROCFS / str(int(pid)) / "environ").read_bytes()
    except OSError:
        return {}  # gone, not ours, or no procfs
    out: dict[str, str] = {}
    for chunk in raw.split(b"\0"):
        if not chunk:
            continue
        key, sep, value = chunk.partition(b"=")
        if sep:
            out[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return out


def list_pids_with_env(name: str, value: str) -> list[int]:
    """Pids whose environment has ``name == value``, excluding this process.

    ⚑ WHY ENV AND NOT CMDLINE OR ANCESTRY (audit R2). The processes this exists
    to find are Stockfish engines orphaned by a worker that died without running
    a `finally`. Their cmdline is the bare engine binary — no module name, no
    `--trial-id` — so `_list_matching_pids` structurally cannot match them
    whatever `required_terms` it is given. And ancestry is destroyed by exactly
    the event we are reacting to: an orphan is reparented to init, so ppid and
    pgid carry no information. `/proc/<pid>/environ` is fixed at exec and
    survives both.

    Failure modes, since they bound what this can promise:
    * only readable for processes owned by the same uid — fine here (the worker
      spawns its engines as itself), useless for cross-user cleanup;
    * it is the environment at EXEC time, so a later `setenv` in the child is
      invisible. That is what makes it reliable for a marker stamped at spawn;
    * Linux-only (procfs). Returns nothing elsewhere rather than raising;
    * O(number of pids) — a cleanup-path cost, not a hot-path one.
    """
    if not str(name) or not str(value):
        return []
    me = os.getpid()
    out: list[int] = []
    try:
        entries = os.listdir(_PROCFS)
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == me:
            continue
        if _pid_env(pid).get(str(name)) == str(value):
            out.append(pid)
    return out


def terminate_engines_owned_by(owner_pid: int, *, timeout_s: float = 5.0) -> list[int]:
    """Kill every Stockfish engine stamped as owned by ``owner_pid``.

    Called after a worker process is known to be gone; anything still carrying
    its marker is by definition an orphan.
    """
    return [
        pid
        for pid in list_pids_with_env(CAE_ENGINE_MARKER_ENV, str(int(owner_pid)))
        if _terminate_pid(pid, timeout_s=timeout_s)
    ]


def _pid_exists(pid: int) -> bool:
    """Is ``pid`` a live process — deliberately counting a zombie as gone.

    ``os.kill(pid, 0)`` succeeds against a zombie, so on its own it reports a
    process that has already released every byte of its memory as still
    running. Every caller here is asking "did the termination take", and for
    that question a zombie is a yes: the answer this reaper exists to give is
    about reclaimed RSS, not about whether the parent has got round to
    ``wait()``. Left as ``os.kill`` alone, ``_terminate_pid`` burns its whole
    timeout on a corpse and then reports failure — a status that does not mean
    what its name says.
    """
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's process; alive as far as we can tell
    try:
        stat = (_PROCFS / str(int(pid)) / "stat").read_bytes()
    except OSError:
        # Entry gone under a real procfs means the process is gone. With no
        # procfs at all (non-Linux) keep the pre-existing os.kill answer rather
        # than silently reporting every live process as dead.
        return not _PROCFS.is_dir()
    # "<pid> (<comm>) <state> ..." — comm is arbitrary bytes and may itself
    # contain spaces and parentheses, so the state is what follows the LAST
    # ')'. Anything unparseable is reported as alive, the conservative answer.
    close = stat.rfind(b")")
    if close < 0:
        return True
    return stat[close + 2 : close + 3] != b"Z"


def _terminate_pid(pid: int, *, timeout_s: float = 5.0) -> bool:
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
  # not our process or kernel refused — treat as failure
        return False

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if not _pid_exists(int(pid)):
            return True
        time.sleep(0.1)

    try:
        os.kill(int(pid), signal.SIGKILL)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
  # not our process or kernel refused — treat as failure
        return False
    return not _pid_exists(int(pid))


def terminate_matching_processes(
    *,
    module: str,
    required_terms: Iterable[str],
    exclude_pids: Iterable[int] = (),
    timeout_s: float = 5.0,
) -> list[int]:
    return [
        int(pid)
        for pid in _list_matching_pids(
            module=module,
            required_terms=required_terms,
            exclude_pids=exclude_pids,
        )
        if _terminate_pid(pid, timeout_s=float(timeout_s))
    ]

