#!/usr/bin/env python3
"""Gracefully pause a running PBT experiment and wait for N trials to reach a
clean iteration boundary before signalling that it's safe to kill and restart.

Usage:
    python3 scripts/graceful_restart.py                  # pause all, wait for 2
    python3 scripts/graceful_restart.py --wait 3         # wait for 3 trials
    python3 scripts/graceful_restart.py --tune-dir runs/pbt2_small/tune
    python3 scripts/graceful_restart.py --resume-cmd "python3 -m chess_anti_engine.run --config configs/pbt2_small.yaml --mode tune --resume"

What it does:
  1. Creates pause.txt in the tune dir  → trials finish their current iteration
     then hold at the start of the next one (the existing _wait_if_paused hook).
  2. Polls progress.csv mtime for each trial.  A trial is considered "paused"
     when its progress.csv hasn't been written for --idle-secs seconds (default 90).
  3. Once --wait trials are paused, prints a ready message.  If --auto-kill is
     set it also sends SIGTERM to the tuner process and removes pause.txt.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _find_tune_dir(work_dir: Path) -> Path:
    p = work_dir / "tune"
    if p.is_dir():
        return p
    raise FileNotFoundError(f"Could not find tune dir at {p}")


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


def _idle_seconds(csv: Path) -> float:
    """Seconds since the progress.csv was last modified."""
    return time.time() - csv.stat().st_mtime


def _find_tuner_pid(tune_dir: Path) -> int | None:
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
    ap.add_argument("--wait", type=int, default=2,
                    help="Number of trials that must be idle before declaring safe")
    ap.add_argument("--idle-secs", type=int, default=90,
                    help="Seconds without a progress.csv write to call a trial idle/paused")
    ap.add_argument("--poll", type=int, default=15,
                    help="Polling interval in seconds")
    ap.add_argument("--auto-kill", action="store_true",
                    help="Automatically send SIGTERM to the tuner process when ready")
    ap.add_argument("--resume-cmd", default="",
                    help="Shell command to run after killing (implies --auto-kill)")
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
    auto_kill = args.auto_kill or bool(args.resume_cmd)

    # Step 1: create the pause marker
    if pause_file.exists():
        print(f"[graceful_restart] pause.txt already exists at {pause_file}")
    else:
        pause_file.write_text("graceful restart in progress\n")
        print(f"[graceful_restart] Created {pause_file}")
        print("[graceful_restart] Trials will pause after their current iteration.")

    print(f"[graceful_restart] Waiting for {args.wait} of the active trials to go idle "
          f"(>{args.idle_secs}s without a progress.csv update)...")
    print()

    start = time.time()
    while True:
        csvs = _active_trials(tune_dir)
        if not csvs:
            print("[graceful_restart] No active trials found yet — waiting...")
            time.sleep(args.poll)
            continue

        idle = [(csv, _idle_seconds(csv)) for csv in csvs]
        idle_trials = [(csv, age) for csv, age in idle if age >= args.idle_secs]

        # Print status
        elapsed = int(time.time() - start)
        print(f"[{elapsed:4d}s] {len(idle_trials)}/{len(csvs)} trials paused "
              f"(need {args.wait}):")
        for csv, age in sorted(idle, key=lambda x: -x[1]):
            trial = csv.parent.name.split("_")[2] + "_" + csv.parent.name.split("_")[3]
            status = "PAUSED" if age >= args.idle_secs else "running"
            print(f"         {trial}  {status}  ({age:.0f}s idle)")

        if len(idle_trials) >= args.wait:
            print()
            print(f"[graceful_restart] {len(idle_trials)} trials are at a clean stopping point.")

            if auto_kill:
                pid = _find_tuner_pid(tune_dir)
                if pid:
                    print(f"[graceful_restart] Sending SIGTERM to tuner PID {pid}...")
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(3)
                else:
                    print("[graceful_restart] Could not find tuner PID — kill it manually.")

                print(f"[graceful_restart] Removing {pause_file}")
                pause_file.unlink(missing_ok=True)

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
                        still_idle = sum(
                            1 for c in csvs2 if _idle_seconds(c) >= args.idle_secs
                        )
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
