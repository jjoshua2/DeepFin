#!/usr/bin/env python3
"""Gracefully pause a running PBT experiment and wait for N trials to reach a
clean iteration boundary, then kill the tuner and restart automatically.

Usage:
    python3 scripts/graceful_restart.py                  # pause, wait, restart (default)
    python3 scripts/graceful_restart.py --no-auto-kill   # pause and print status only
    python3 scripts/graceful_restart.py --wait 3         # wait for 3 trials to be idle
    python3 scripts/graceful_restart.py --tune-dir runs/pbt2_small/tune
    python3 scripts/graceful_restart.py --resume-cmd "custom restart command"

What it does:
  1. Creates pause.txt in the tune dir  → trials finish their current iteration
     then hold at the start of the next one (the existing _wait_if_paused hook).
  2. Snapshots progress.csv row count at pause.  A trial is considered "paused"
     when (a) it appended >= 1 new row after pause and the row count then
     stayed flat for one poll cycle, OR (b) row count is still at the snapshot
     and a --grace-secs grace window has elapsed (handles the edge case where
     pause.txt was created exactly at an iter boundary, so the next iter
     blocks before writing any post-pause row). Row count is the right
     signal — Ray Tune touches progress.csv via metadata sync independent of
     iter completion, so mtime polling falsely reports "active" forever.
  3. Once --wait trials are paused, sends SIGTERM to the tuner, removes
     pause.txt, and runs the resume command.  Pass --no-auto-kill to skip
     this and just print status instead.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _active_trials(tune_dir: Path) -> list[Path]:
    """Return progress.csv paths for all trials that have at least one row."""
    csvs = []
    for d in sorted(tune_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("train_trial_"):
            continue
        csv = d / "progress.csv"
        if csv.exists() and csv.stat().st_size > 200:
            csvs.append(csv)
    return csvs


def _row_count(csv: Path) -> int:
    """Count data rows in progress.csv (excludes header)."""
    try:
        with csv.open() as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _find_tuner_pid() -> int | None:
    """Best-effort lookup of the top-level Tune driver process.

    Intentionally ignore Ray trial actors and worker processes so --auto-kill
    only targets the main `chess_anti_engine.run --mode tune` driver.
    """
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
        candidates: list[int] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pid_s, args = line.split(None, 1)
            except ValueError:
                continue
            if "python3 -m chess_anti_engine.run" not in args:
                continue
            if "--mode tune" not in args:
                continue
            if "ray::ImplicitFunc.train" in args or "chess_anti_engine.worker" in args:
                continue
            if pid_s.isdigit():
                candidates.append(int(pid_s))
        if len(candidates) == 1:
            return candidates[0]
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune-dir", default="runs/pbt2_small/tune",
                    help="Path to the Ray Tune experiment directory")
    ap.add_argument("--wait", type=int, default=1,
                    help="Number of trials that must be idle before declaring safe")
    ap.add_argument("--grace-secs", type=int, default=90,
                    help="Grace window for the boundary-edge case (no post-pause row yet — "
                         "treat as paused if row count has been at snapshot for this long)")
    ap.add_argument("--poll", type=int, default=15,
                    help="Polling interval in seconds")
    ap.add_argument("--no-auto-kill", dest="auto_kill", action="store_false",
                    help="Print status and exit without killing or restarting")
    ap.add_argument("--resume-cmd", default="./scripts/train.sh restart",
                    help="Shell command to run after killing (default: ./scripts/train.sh restart)")
    args = ap.parse_args()

    tune_dir = Path(args.tune_dir)
    if not tune_dir.is_dir():
        # Try relative to repo root
        repo = Path(__file__).parent.parent
        tune_dir = repo / args.tune_dir
    if not tune_dir.is_dir():
        print(f"ERROR: tune dir not found: {args.tune_dir}", file=sys.stderr)
        sys.exit(1)

    pause_file = tune_dir / "pause.txt"
    auto_kill = args.auto_kill

    # Step 1: create the pause marker(s). Drop one in tune_dir AND one in
    # every active trial dir — the trial checks both. Without the per-trial
    # markers a previous run pause-no-op'd silently when the actor's view of
    # tune_dir/pause.txt didn't fire its exists() check (root cause never
    # diagnosed; the per-trial marker is the belt-and-suspenders fix).
    pause_targets: list[Path] = [pause_file]
    for csv in _active_trials(tune_dir):
        pause_targets.append(csv.parent / "pause.txt")

    for target in pause_targets:
        if target.exists():
            print(f"[graceful_restart] pause.txt already exists at {target}")
        else:
            target.write_text("graceful restart in progress\n")
            print(f"[graceful_restart] Created {target}")
    print("[graceful_restart] Trials will pause after their current iteration.")

    print(f"[graceful_restart] Waiting for {args.wait} of the active trials to "
          f"stop appending rows to progress.csv...")
    print()

    pause_created_ts = pause_file.stat().st_mtime if pause_file.exists() else time.time()
    start = time.time()
    # Snapshot row count at pause time per trial. A trial counts as paused
    # once we observe (rows_now > rows_at_pause) AND rows_now stayed flat for
    # one full poll cycle — i.e. exactly one post-pause iter completed and
    # the next one is now blocked at _wait_if_paused.
    snapshot_rows: dict[Path, int] = {c: _row_count(c) for c in _active_trials(tune_dir)}
    prev_rows: dict[Path, int] = dict(snapshot_rows)
    while True:
        csvs = _active_trials(tune_dir)
        if not csvs:
            print("[graceful_restart] No active trials found yet — waiting...")
            time.sleep(args.poll)
            continue

        idle_trials: list[tuple[Path, str]] = []
        observations: list[tuple[Path, int, int, str]] = []
        for csv in csvs:
            rc = _row_count(csv)
            snap = snapshot_rows.setdefault(csv, rc)  # late-arriving trial: anchor at first sight
            prev = prev_rows.get(csv, snap)
            if rc > snap and rc == prev:
                state = "PAUSED"
                idle_trials.append((csv, state))
            elif rc == snap and time.time() - pause_created_ts >= args.grace_secs:
                state = "PAUSED-AT-BOUNDARY"
                idle_trials.append((csv, state))
            else:
                state = f"running rows {snap}->{rc}"
            observations.append((csv, snap, rc, state))
            prev_rows[csv] = rc

        # Print status
        elapsed = int(time.time() - start)
        print(f"[{elapsed:4d}s] {len(idle_trials)}/{len(csvs)} trials paused "
              f"(need {args.wait}):")
        for csv, snap, rc, state in observations:
            trial = csv.parent.name.split("_")[2] + "_" + csv.parent.name.split("_")[3]
            print(f"         {trial}  rows@pause={snap} rows_now={rc}  {state}")

        if len(idle_trials) >= args.wait:
            print()
            print(f"[graceful_restart] {len(idle_trials)} trials are at a clean stopping point.")

            if auto_kill:
                pid = _find_tuner_pid()
                if pid:
                    print(f"[graceful_restart] Sending SIGTERM to tuner PID {pid}...")
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(3)
                else:
                    print("[graceful_restart] Could not find tuner PID — kill it manually.")

                for target in pause_targets:
                    if target.exists():
                        print(f"[graceful_restart] Removing {target}")
                        target.unlink(missing_ok=True)

                if args.resume_cmd:
                    print(f"[graceful_restart] Running: {args.resume_cmd}")
                    time.sleep(5)  # let Ray finish shutting down
                    subprocess.run(args.resume_cmd, shell=True)
            else:
                print()
                print("  Next steps:")
                print("  1. Kill the tuner process (Ctrl+C or kill the run command)")
                print(f"  2. rm {pause_file}")
                print("  3. Restart with --resume")
                print()
                print(f"[graceful_restart] Watching until you kill — remove {pause_file} "
                      f"to let trials continue if you change your mind.")
                # Keep watching so user can see if trials start moving again
                try:
                    while True:
                        time.sleep(args.poll)
                        csvs2 = _active_trials(tune_dir)
                        still_idle = 0
                        for c in csvs2:
                            rc = _row_count(c)
                            snap = snapshot_rows.get(c, rc)
                            prev = prev_rows.get(c, snap)
                            if rc == prev and (rc > snap or time.time() - pause_created_ts >= args.grace_secs):
                                still_idle += 1
                            prev_rows[c] = rc
                        elapsed2 = int(time.time() - start)
                        print(f"[{elapsed2:4d}s] still {still_idle}/{len(csvs2)} paused — "
                              f"safe to kill and restart with --resume")
                except KeyboardInterrupt:
                    print("\n[graceful_restart] Interrupted. "
                          f"Remember to rm {pause_file} if you are not restarting.")
            return

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
