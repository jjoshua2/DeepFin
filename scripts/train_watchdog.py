#!/usr/bin/env python3
"""Training-health watchdog: one detect-and-report check per invocation.

Detects CRASHED / STOPPED / PAUSED-HELD / PAUSE-ABANDONED / STALLED / OK for
the live train.sh-managed run. Never signals, deletes, restarts, or resumes —
read-only report only.

Usage (cron / loop):
  PYTHONPATH=. python3 scripts/train_watchdog.py
  PYTHONPATH=. python3 scripts/train_watchdog.py --stall-minutes 90
  PYTHONPATH=. python3 scripts/train_watchdog.py --notify-cmd 'notify-send training'

Exit codes: 0=OK, 1=STOPPED, 2=PAUSED-HELD, 3=STALLED, 4=ERROR, 5=CRASHED,
6=PAUSE-ABANDONED.
STOPPED vs CRASHED both mean 'no live PID' and differ only by whether
train.sh's intentional-stop marker is present (see --stop-marker).
Stdout is always exactly one ``watchdog: ...`` line (or ERROR on failure).

⚑ WHY PAUSE-ABANDONED EXISTS (2026-08-09, the ack-gated pause window).
``decide()`` returns PAUSED-HELD whenever the loop is flat and a ``pause.txt``
is present, and STALLED requires ``pause_txt is None``; ``watchdog_loop.sh``
recovers only on STALLED. So while ANY marker is held, auto-recovery is
structurally unable to fire and ``recover_stall.sh``'s ``rm -f ... pause.txt``
is unreachable. That was tolerable while a marker meant "an operator is doing
something"; ``scripts/pause_window.sh`` makes a held marker a NIGHTLY event, so
the same verdict now means both "production is parked" and "the ratchet is
running", and a wrapper that dies holding the marker parks production until a
human notices.

The marker itself carries the discriminator: ``pause_window.sh`` writes
``pid=<n> started=<iso>``, and ``graceful_restart.py`` writes prose with no
``pid=``. So ONLY a self-identifying marker can be judged abandoned — an
operator's marker is never touched, whatever its age — and the criterion is
"the owning process is gone", alone. Age is never sufficient: see below.

⚑ ``--pause-max-minutes`` NO LONGER CLEARS ANYTHING (2026-08-20). It used to
be a second abandonment criterion, sized to the 120-min ratchet window (ack 30
+ BUDGET_MIN 90, 180 = 50% headroom), and the paragraph that sat here warned
that raising the bound past ~150 would make the age branch fire on a HEALTHY
window. The failure arrived from the other side: the WINDOW grew. The
lc0-control pause held its marker for a planned ~10 h, and at minute 185 the
age branch declared abandonment on a verdict whose own details read
``pause_owner_alive=1``; ``watchdog_loop.sh`` deleted the live window's marker
and ten minutes later force-restarted production beside the control train it
was parked for (2026-08-20 16:02/16:12, ``scratchpad/watchdog_alerts.log``).
How long an operator may hold a pause is a policy question, not a machine's
call. The flag now only sets when a held marker's log line gains a
``pause_overheld`` annotation; CLEARING requires a dead owner.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Match scripts/train.sh defaults. TRAIN_PIDFILE is the same seam
# ratchet_loop.sh already reads for the same file; without it nothing can drive
# watchdog_loop.sh against a sandbox, because the loop does not pass --pidfile
# and the check would silently read the LIVE run's pid.
DEFAULT_PIDFILE = Path(os.environ.get("TRAIN_PIDFILE", "/tmp/chess_training.pid"))
DEFAULT_LOG = Path("/tmp/chess_training.log")
DEFAULT_WORK_DIR = Path(os.environ.get("TRAIN_WORK_DIR", "runs/pbt2_small"))
DEFAULT_STALL_MINUTES = 90.0
DEFAULT_PAUSE_MAX_MINUTES = 180.0
# Touched by `train.sh stop` BEFORE it kills, removed by `train.sh start`. Its
# presence is the ONLY thing separating a deliberate stop from a crash: both
# leave no PID. Before 2026-08-08 both reported STOPPED, so a crash that killed
# the process was indistinguishable from the operator stopping for the evening
# — the 4h32m outage that day went unnoticed for exactly that reason.
DEFAULT_STOP_MARKER = Path("/tmp/chess_training.intentional_stop")

EXIT_OK = 0
EXIT_STOPPED = 1
EXIT_PAUSED_HELD = 2
EXIT_STALLED = 3
EXIT_ERROR = 4
# New in 2026-08-08. Appended, never renumbered: watchdog_loop.sh switches on
# EXIT_STALLED by value for auto-recovery, and any consumer treating "non-zero"
# as "needs attention" keeps working unchanged.
EXIT_CRASHED = 5
# ⚑ 6, NOT 5. This branch was written against a main that stopped at 4 and it
# claimed 5, which #371 had already taken for CRASHED. Two states sharing an
# exit code is the same defect one level up from the one this state exists to
# fix: watchdog_loop.sh would have cleared a pause marker on a CRASHED verdict.
# Caught by reading origin/main, not by any test here -- a merge conflict would
# have shown the collision only if both sides had touched the same LINE, and
# they did not.
EXIT_PAUSE_ABANDONED = 6

STATE_OK = "OK"
STATE_STOPPED = "STOPPED"
STATE_PAUSED_HELD = "PAUSED-HELD"
STATE_PAUSE_ABANDONED = "PAUSE-ABANDONED"
STATE_STALLED = "STALLED"
STATE_ERROR = "ERROR"
STATE_CRASHED = "CRASHED"

# `pid=<n>` anywhere in the marker, on a word boundary. Written by
# scripts/pause_window.sh; graceful_restart.py's marker deliberately has none.
_PAUSE_OWNER_RE = re.compile(r"(?:^|\s)pid=(\d+)\b")


@dataclass(frozen=True)
class ProgressSnapshot:
    """Observed progress for one check (injectable in tests)."""

    pid: int | None
    pid_alive: bool
    pause_txt: str | None
    rows: int
    rows_prev: int | None
    minutes_flat: float
    trial_dir: str | None = None
    progress_file: str | None = None
    # True when train.sh's intentional-stop marker exists. Defaults False so a
    # caller that forgets to supply it reports CRASHED rather than silently
    # downgrading a real crash to a benign stop: the safe default is the loud one.
    intentional_stop: bool = False
    # Present only for a SELF-IDENTIFYING pause marker (one carrying `pid=`). A
    # marker without an owner line is an operator's and is never judged
    # abandoned, so both stay None and the verdict is PAUSED-HELD as before.
    pause_owner_pid: int | None = None
    pause_owner_alive: bool | None = None
    pause_age_minutes: float | None = None


@dataclass(frozen=True)
class Verdict:
    """Decision output: state name, exit code, and detail kv pairs."""

    state: str
    exit_code: int
    details: dict[str, Any]

    def format_line(self) -> str:
        parts = [f"watchdog: {self.state}"]
        for key, value in self.details.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return " ".join(parts)


@dataclass(frozen=True)
class PersistedState:
    """Cross-invocation flatness memory (row count + wall time of last growth)."""

    rows: int
    wall_time: float
    trial: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"rows": self.rows, "wall_time": self.wall_time}
        if self.trial is not None:
            d["trial"] = self.trial
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistedState:
        return cls(
            rows=int(data["rows"]),
            wall_time=float(data["wall_time"]),
            trial=(str(data["trial"]) if data.get("trial") is not None else None),
        )


def decide(
    snap: ProgressSnapshot,
    stall_minutes: float,
    pause_max_minutes: float = DEFAULT_PAUSE_MAX_MINUTES,
) -> Verdict:
    """Pure state machine: snapshot in, verdict out. No I/O.

    Check order: CRASHED/STOPPED → PAUSE-ABANDONED → PAUSED-HELD → STALLED → OK.
    Flatness is ``rows`` not greater than ``rows_prev`` (or no prior rows).
    PAUSED-HELD requires pause.txt AND flat progress (the boundary-hold case
    appends no rows; mtime is untrustworthy because Ray metadata syncs).
    PAUSE-ABANDONED is the same state narrowed to a marker that NAMES ITS
    OWNER and whose owner is GONE — the only pause a machine may safely
    clear. A live owner's marker stays PAUSED-HELD whatever its age; past
    ``pause_max_minutes`` its line carries a ``pause_overheld`` annotation.
    STALLED is flat without pause for longer than ``stall_minutes``.
    Within the stall window (or after growth), the result is OK.

    A dead/absent PID splits on ``intentional_stop``: with the marker it is a
    deliberate STOPPED (the operator ran `train.sh stop`), without it the
    process died on its own and the verdict is CRASHED. Both are non-zero, so
    every existing "non-OK" consumer is unaffected; only the label and the
    exit code differ, which is what makes a crash legible in the log after
    the fact instead of looking identical to an evening off.
    """
    details: dict[str, Any] = {
        "pid": snap.pid if snap.pid is not None else "none",
        "rows": snap.rows,
        "rows_prev": snap.rows_prev if snap.rows_prev is not None else "none",
        "minutes_flat": f"{snap.minutes_flat:.1f}",
    }
    if snap.trial_dir is not None:
        details["trial"] = snap.trial_dir
    if snap.progress_file is not None:
        details["progress"] = snap.progress_file
    if snap.pause_txt is not None:
        details["pause_txt"] = snap.pause_txt
    if snap.pause_owner_pid is not None:
        details["pause_owner"] = snap.pause_owner_pid
        details["pause_owner_alive"] = int(bool(snap.pause_owner_alive))
    if snap.pause_age_minutes is not None:
        details["pause_age_min"] = f"{snap.pause_age_minutes:.1f}"

    if snap.pid is None or not snap.pid_alive:
        if snap.intentional_stop:
            return Verdict(STATE_STOPPED, EXIT_STOPPED, details)
        return Verdict(STATE_CRASHED, EXIT_CRASHED, details)

    flat = snap.rows_prev is not None and snap.rows <= snap.rows_prev
    # First observation (no prior state): not flat — arm the clock, report OK.
    if snap.rows_prev is None:
        flat = False

    if flat and snap.pause_txt is not None:
        reason = _abandoned_reason(snap)
        if reason is not None:
            details["pause_abandoned"] = reason
            return Verdict(STATE_PAUSE_ABANDONED, EXIT_PAUSE_ABANDONED, details)
        if (
            snap.pause_age_minutes is not None
            and snap.pause_age_minutes > pause_max_minutes
        ):
            # Held past the bound but not clearable (owner alive, or unowned):
            # say so on the log line. Deliberately NOT a new state — the
            # transition alerter already paged on entry into PAUSED-HELD, and
            # the operator-side bound on a hung window is the pause lease.
            details["pause_overheld"] = (
                f"held_{snap.pause_age_minutes:.0f}min"
                f"_over_{pause_max_minutes:.0f}"
            )
        return Verdict(STATE_PAUSED_HELD, EXIT_PAUSED_HELD, details)

    if flat and snap.pause_txt is None and snap.minutes_flat > stall_minutes:
        return Verdict(STATE_STALLED, EXIT_STALLED, details)

    return Verdict(STATE_OK, EXIT_OK, details)


def _abandoned_reason(snap: ProgressSnapshot) -> str | None:
    """Why this held marker is recoverable, or None if it must be left alone.

    ⚑ THE GATE IS ``pause_owner_pid is not None``, not the age. A marker with
    no ``pid=`` line is an operator's ``graceful_restart.py`` pause, and an
    operator's pause is allowed to outlast any bound we could pick — clearing
    it would resume the run they deliberately parked. So an unowned marker can
    never reach the criterion below.

    ⚑ A LIVE OWNER IS NEVER ABANDONED, WHATEVER THE AGE (2026-08-20). Age used
    to be a second criterion here, and it cleared a marker whose own verdict
    line read ``pause_owner_alive=1`` — see the module docstring for the
    incident. An unknown liveness (``None``) is also never abandoned: a
    machine must not delete on a read it could not complete.
    """
    if snap.pause_owner_pid is None:
        return None
    if snap.pause_owner_alive is False:
        return f"owner_pid_{snap.pause_owner_pid}_is_gone"
    return None


def parse_pause_marker(
    text: str,
    *,
    mtime: float | None,
    now: float,
) -> tuple[int | None, float | None]:
    """``(owner_pid, age_minutes)`` for a pause marker's contents.

    ``pause_window.sh`` writes ``pause_window.sh pid=<n> started=<iso>``. The
    age comes from ``started=`` when it parses and from the file's mtime
    otherwise; a marker with no ``pid=`` yields ``(None, None)`` regardless,
    because nothing downstream may act on an unowned marker's age.
    """
    m = _PAUSE_OWNER_RE.search(text)
    if m is None:
        return None, None
    pid = int(m.group(1))

    started = re.search(r"(?:^|\s)started=(\S+)", text)
    if started is not None:
        try:
            begin = dt.datetime.fromisoformat(started.group(1)).timestamp()
        except (ValueError, OSError, OverflowError):
            pass
        else:
            return pid, max(0.0, (now - begin) / 60.0)
    if mtime is not None:
        return pid, max(0.0, (now - mtime) / 60.0)
    return pid, None


def compute_flatness(
    *,
    rows: int,
    trial: str | None,
    now: float,
    prev: PersistedState | None,
) -> tuple[int | None, float, PersistedState, bool]:
    """Compare current rows to persisted state.

    Returns ``(rows_prev, minutes_flat, new_state, grew)``.
    Growth (or a trial switch / first sight) resets the flatness clock.
    A row *decrease* also resets — new trial dirs or rotated progress.csv.
    """
    if prev is None:
        return None, 0.0, PersistedState(rows=rows, wall_time=now, trial=trial), False

    trial_changed = prev.trial is not None and trial is not None and prev.trial != trial
    grew = rows > prev.rows or trial_changed or rows < prev.rows
    if grew:
        # rows < prev also resets: treat as a new baseline, not continued flatness.
        return prev.rows, 0.0, PersistedState(rows=rows, wall_time=now, trial=trial), True

    minutes_flat = max(0.0, (now - prev.wall_time) / 60.0)
    # Keep the original wall_time so minutes_flat accumulates across invocations.
    return prev.rows, minutes_flat, prev, False


def read_pid(pidfile: Path) -> int | None:
    """Read the train.sh PID file. Missing/empty/garbage → None."""
    try:
        if not pidfile.is_file():
            return None
        text = pidfile.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return int(text.split()[0])
    except (OSError, ValueError):
        return None


def pid_is_alive(pid: int) -> bool:
    """True if ``kill(pid, 0)`` succeeds (same signal used by train.sh status)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def find_pause_txt(tune_dir: Path) -> Path | None:
    """Any pause.txt under the tune dir (root or per-trial markers)."""
    if not tune_dir.is_dir():
        return None
    root = tune_dir / "pause.txt"
    if root.is_file():
        return root
    try:
        for p in sorted(tune_dir.rglob("pause.txt")):
            if p.is_file():
                return p
    except OSError:
        return None
    return None


def _progress_csv_rows(csv: Path) -> int | None:
    """Data rows in progress.csv (header excluded). None if unusable."""
    try:
        if not csv.is_file() or csv.stat().st_size < 2:
            return None
        with csv.open(encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except OSError:
        return None


def _result_json_rows(path: Path) -> int | None:
    """Non-empty line count of result.json. None if missing/unreadable."""
    try:
        if not path.is_file():
            return None
        with path.open(encoding="utf-8", errors="replace") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def newest_trial_dir(tune_dir: Path) -> Path | None:
    """Newest ``train_trial_*`` under tune (populated trials preferred).

    Mirrors scripts/trial_paths._trial_sort_key: a dir with progress/result
    outranks an empty one; then progress mtime; then name.
    """
    if not tune_dir.is_dir():
        return None
    try:
        trials = [p for p in tune_dir.glob("train_trial_*") if p.is_dir()]
        if not trials:
            trials = [p for p in tune_dir.rglob("train_trial_*") if p.is_dir()]
    except OSError:
        return None
    if not trials:
        return None

    def sort_key(trial: Path) -> tuple[int, float, str]:
        for name in ("progress.csv", "result.json"):
            candidate = trial / name
            try:
                if candidate.is_file():
                    return (1, candidate.stat().st_mtime, trial.name)
            except OSError:
                continue
        try:
            return (0, trial.stat().st_mtime, trial.name)
        except OSError:
            return (0, 0.0, trial.name)

    return max(trials, key=sort_key)


def count_progress_rows(trial_dir: Path | None) -> tuple[int, str | None]:
    """Row count for the trial: progress.csv preferred, else result.json lines."""
    if trial_dir is None:
        return 0, None
    csv = trial_dir / "progress.csv"
    n = _progress_csv_rows(csv)
    if n is not None:
        return n, str(csv)
    result = trial_dir / "result.json"
    n = _result_json_rows(result)
    if n is not None:
        return n, str(result)
    return 0, None


def load_state(path: Path) -> PersistedState | None:
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "rows" not in data or "wall_time" not in data:
            return None
        return PersistedState.from_dict(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def save_state(path: Path, state: PersistedState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def default_state_path(log_path: Path = DEFAULT_LOG) -> Path:
    """State file lives next to the training log (train.sh default)."""
    return log_path.with_suffix(".watchdog.json")


def build_snapshot(
    *,
    pidfile: Path,
    work_dir: Path,
    state_path: Path,
    stop_marker: Path,
    now: float | None = None,
    pid_alive_fn: Callable[[int], bool] = pid_is_alive,
) -> tuple[ProgressSnapshot, PersistedState]:
    """Gather filesystem observations + compute flatness vs persisted state.

    ``stop_marker`` is REQUIRED rather than defaulted: a caller that silently
    got the production ``/tmp`` path would make tests depend on whether the
    operator happens to have training stopped right now.
    """
    now = time.time() if now is None else now
    pid = read_pid(pidfile)
    alive = bool(pid is not None and pid_alive_fn(pid))
    intentional = stop_marker.exists()

    tune_dir = work_dir / "tune"
    pause = find_pause_txt(tune_dir)
    trial = newest_trial_dir(tune_dir)
    rows, progress_file = count_progress_rows(trial)
    trial_name = trial.name if trial is not None else None

    owner_pid: int | None = None
    owner_alive: bool | None = None
    pause_age: float | None = None
    if pause is not None:
        # Fail-soft: an unreadable marker is reported as an ordinary
        # PAUSED-HELD, never as one a machine may clear.
        try:
            text = pause.read_text(encoding="utf-8", errors="replace")
            mtime = pause.stat().st_mtime
        except OSError:
            text, mtime = "", None
        owner_pid, pause_age = parse_pause_marker(text, mtime=mtime, now=now)
        if owner_pid is not None:
            owner_alive = pid_alive_fn(owner_pid)

    prev = load_state(state_path)
    rows_prev, minutes_flat, new_state, _grew = compute_flatness(
        rows=rows, trial=trial_name, now=now, prev=prev,
    )

    snap = ProgressSnapshot(
        pid=pid,
        pid_alive=alive,
        pause_txt=str(pause) if pause is not None else None,
        rows=rows,
        rows_prev=rows_prev,
        minutes_flat=minutes_flat,
        trial_dir=trial_name,
        progress_file=progress_file,
        intentional_stop=intentional,
        pause_owner_pid=owner_pid,
        pause_owner_alive=owner_alive,
        pause_age_minutes=pause_age,
    )
    return snap, new_state


def maybe_notify(notify_cmd: str | None, line: str, exit_code: int) -> None:
    """Fail-soft: run notify_cmd with the status line as one extra argument."""
    if not notify_cmd or exit_code == EXIT_OK:
        return
    try:
        if _looks_like_shell(notify_cmd):
            subprocess.run(
                f"{notify_cmd} {_shell_quote(line)}",
                shell=True,
                check=False,
                timeout=60,
                capture_output=True,
            )
        else:
            subprocess.run(
                [notify_cmd, line],
                shell=False,
                check=False,
                timeout=60,
                capture_output=True,
            )
    except Exception:
        pass


def _looks_like_shell(cmd: str) -> bool:
    # Multi-word commands / pipelines need the shell; a bare executable path does not.
    return any(ch in cmd for ch in (" ", "|", ";", "&", ">", "<", "`", "$"))


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pidfile",
        type=Path,
        default=DEFAULT_PIDFILE,
        help=f"train.sh PID file (default: {DEFAULT_PIDFILE})",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=f"Training work dir; tune lives at WORK_DIR/tune (default: {DEFAULT_WORK_DIR})",
    )
    ap.add_argument(
        "--state",
        type=Path,
        default=None,
        help="JSON state path for row-count flatness (default: alongside training log)",
    )
    ap.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Training log path; used only to place default --state (default: {DEFAULT_LOG})",
    )
    ap.add_argument(
        "--stall-minutes",
        type=float,
        default=DEFAULT_STALL_MINUTES,
        help=f"Minutes of flat progress before STALLED (default: {DEFAULT_STALL_MINUTES:g})",
    )
    ap.add_argument(
        "--stop-marker",
        type=Path,
        default=DEFAULT_STOP_MARKER,
        help=(
            "train.sh intentional-stop marker. Present + dead PID = STOPPED "
            f"(deliberate); absent + dead PID = CRASHED (default: {DEFAULT_STOP_MARKER})"
        ),
    )
    ap.add_argument(
        "--pause-max-minutes",
        type=float,
        default=DEFAULT_PAUSE_MAX_MINUTES,
        help=(
            "A held marker older than this gains a pause_overheld annotation "
            f"on its PAUSED-HELD line (default: {DEFAULT_PAUSE_MAX_MINUTES:g} "
            "min). It no longer clears anything: only a marker whose pid= "
            "owner is GONE is PAUSE-ABANDONED (2026-08-20 incident)."
        ),
    )
    ap.add_argument(
        "--notify-cmd",
        default=None,
        help="On non-OK, run this command with the status line as one argument (fail-soft)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.stall_minutes <= 0:
            raise ValueError("--stall-minutes must be > 0")
        if args.pause_max_minutes <= 0:
            raise ValueError("--pause-max-minutes must be > 0")

        state_path = args.state if args.state is not None else default_state_path(args.log)
        work_dir = args.work_dir
        if not work_dir.is_absolute():
            # Resolve relative to repo root (parent of scripts/), matching graceful_restart.
            repo = Path(__file__).resolve().parent.parent
            candidate = repo / work_dir
            if candidate.is_dir() or not work_dir.is_dir():
                work_dir = candidate

        snap, new_state = build_snapshot(
            pidfile=args.pidfile,
            work_dir=work_dir,
            state_path=state_path,
            stop_marker=args.stop_marker,
        )
        verdict = decide(
            snap,
            stall_minutes=float(args.stall_minutes),
            pause_max_minutes=float(args.pause_max_minutes),
        )
        # Always persist the flatness baseline so the next invocation can compare.
        try:
            save_state(state_path, new_state)
        except OSError as exc:
            line = f"watchdog: ERROR (failed to write state {state_path}: {exc})"
            print(line)
            return EXIT_ERROR

        line = verdict.format_line()
        print(line)
        maybe_notify(args.notify_cmd, line, verdict.exit_code)
        return verdict.exit_code
    except Exception as exc:
        # CLI must never raise; any internal failure is ERROR exit 4.
        print(f"watchdog: ERROR ({exc})")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
