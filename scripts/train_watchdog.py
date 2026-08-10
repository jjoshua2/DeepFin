#!/usr/bin/env python3
"""Training-health watchdog: one detect-and-report check per invocation.

Detects CRASHED / STOPPED / PAUSED-HELD / STALLED / OK for the live train.sh-managed
run. Never signals, deletes, restarts, or resumes — read-only report only.

Usage (cron / loop):
  PYTHONPATH=. python3 scripts/train_watchdog.py
  PYTHONPATH=. python3 scripts/train_watchdog.py --stall-minutes 90
  PYTHONPATH=. python3 scripts/train_watchdog.py --notify-cmd 'notify-send training'

Exit codes: 0=OK, 1=STOPPED, 2=PAUSED-HELD, 3=STALLED, 4=ERROR, 5=CRASHED.
STOPPED vs CRASHED both mean 'no live PID' and differ only by whether
train.sh's intentional-stop marker is present (see --stop-marker).
Stdout is always exactly one ``watchdog: ...`` line (or ERROR on failure).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Match scripts/train.sh defaults.
DEFAULT_PIDFILE = Path("/tmp/chess_training.pid")
DEFAULT_LOG = Path("/tmp/chess_training.log")
DEFAULT_WORK_DIR = Path(os.environ.get("TRAIN_WORK_DIR", "runs/pbt2_small"))
DEFAULT_STALL_MINUTES = 90.0
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

STATE_OK = "OK"
STATE_STOPPED = "STOPPED"
STATE_PAUSED_HELD = "PAUSED-HELD"
STATE_STALLED = "STALLED"
STATE_ERROR = "ERROR"
STATE_CRASHED = "CRASHED"


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


def decide(snap: ProgressSnapshot, stall_minutes: float) -> Verdict:
    """Pure state machine: snapshot in, verdict out. No I/O.

    Check order: CRASHED/STOPPED → PAUSED-HELD → STALLED → OK.
    Flatness is ``rows`` not greater than ``rows_prev`` (or no prior rows).
    PAUSED-HELD requires pause.txt AND flat progress (the boundary-hold case
    appends no rows; mtime is untrustworthy because Ray metadata syncs).
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

    if snap.pid is None or not snap.pid_alive:
        if snap.intentional_stop:
            return Verdict(STATE_STOPPED, EXIT_STOPPED, details)
        return Verdict(STATE_CRASHED, EXIT_CRASHED, details)

    flat = snap.rows_prev is not None and snap.rows <= snap.rows_prev
    # First observation (no prior state): not flat — arm the clock, report OK.
    if snap.rows_prev is None:
        flat = False

    if flat and snap.pause_txt is not None:
        return Verdict(STATE_PAUSED_HELD, EXIT_PAUSED_HELD, details)

    if flat and snap.pause_txt is None and snap.minutes_flat > stall_minutes:
        return Verdict(STATE_STALLED, EXIT_STALLED, details)

    return Verdict(STATE_OK, EXIT_OK, details)


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
        verdict = decide(snap, stall_minutes=float(args.stall_minutes))
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
