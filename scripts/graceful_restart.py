#!/usr/bin/env python3
"""Gracefully pause a running PBT experiment and wait for active trials to reach
a clean iteration boundary, then kill the tuner and restart automatically.

Usage:
    python3 scripts/graceful_restart.py                  # pause, wait, restart (default)
    python3 scripts/graceful_restart.py --no-auto-kill   # pause and print status only
    python3 scripts/graceful_restart.py --timeout-secs 1800
    python3 scripts/graceful_restart.py --wait 3         # wait for 3 active trials
    python3 scripts/graceful_restart.py --wait-all       # wait for every active trial
    python3 scripts/graceful_restart.py --tune-dir runs/pbt2_small/tune
    python3 scripts/graceful_restart.py --resume-cmd "custom restart command"

What it does:
  1. Creates pause.txt in the tune dir  → trials finish their current iteration
     then hold at the start of the next one (the existing _wait_if_paused hook).
  2. Detects "paused" per trial. PRIMARY signal: the trial's _wait_if_paused
    writes a `.paused_<trial_id>.ack` next to each pause marker (incl. the
    persistent tune root) the instant it holds, and we poll for that — so a
    trial that holds at the boundary BEFORE logging any new row is detected
    immediately (the old row-count heuristic could not see this and would poll
    until timeout). FALLBACK (pre-ack trials): row count appended >= 1 after
    pause then flat for one cycle — Ray Tune touches progress.csv via metadata
    sync, so mtime polling falsely reports "active" forever; row count is the
    right heuristic. --grace-secs is a last-resort manual override (a long
    in-flight iteration also has no post-pause row); the ack makes it unneeded
    for trials on current code.
  3. Once enough active trials are paused, sends SIGTERM to the tuner, removes
     all pause markers, and runs the resume command. By default this keeps old
     automation semantics by requiring one active trial; pass --wait-all to
     require every active trial. Pass
     --no-auto-kill to skip this and just print status instead.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


_LIVE_TRIAL_STATES = {"PENDING", "RUNNING"}


def _progress_csv_for_trial_dir(trial_dir: Path) -> Path | None:
    csv = trial_dir / "progress.csv"
    if csv.exists() and csv.stat().st_size > 200:
        return csv
    return None


def _trial_record_from_state_item(item) -> dict | None:  # skylos: ignore[ANN001]
    try:
        raw = item[0] if isinstance(item, list) and item else item
        if isinstance(raw, str):
            raw = json.loads(raw)
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _active_trials_from_experiment_state(tune_dir: Path) -> list[Path]:
    states = sorted(tune_dir.glob("experiment_state-*.json"), key=lambda p: p.stat().st_mtime)
    if not states:
        return []
    try:
        payload = json.loads(states[-1].read_text())
    except Exception:
        return []

    csvs: list[Path] = []
    for item in payload.get("trial_data", []):
        trial = _trial_record_from_state_item(item)
        if trial is None:
            continue
        if str(trial.get("status", "")).upper() not in _LIVE_TRIAL_STATES:
            continue
        rel = str(trial.get("relative_logdir", "") or "").strip()
        trial_id = str(trial.get("trial_id", "") or "").strip()
        candidates: list[Path] = []
        if rel:
            candidates.append(tune_dir / rel)
        if trial_id:
            candidates.extend(sorted(tune_dir.glob(f"train_trial_{trial_id}*")))
        for trial_dir in candidates:
            csv = _progress_csv_for_trial_dir(trial_dir)
            if csv is not None and csv not in csvs:
                csvs.append(csv)
                break
    return sorted(csvs)


def _active_trials(tune_dir: Path) -> list[Path]:
    """Return progress.csv paths for trials Ray still considers live."""
    from_state = _active_trials_from_experiment_state(tune_dir)
    if from_state:
        return from_state

    csvs: list[Path] = []
    for d in sorted(tune_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("train_trial_"):
            continue
        csv = _progress_csv_for_trial_dir(d)
        if csv is not None:
            csvs.append(csv)
    return csvs


def _row_count(csv: Path) -> int:
    """Count data rows in progress.csv (excludes header)."""
    try:
        with csv.open() as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _pause_ack_files(tune_dir: Path, csvs: list[Path], since_ts: float) -> list[Path]:
    """Per-trial pause-ack files written by the trial's `_wait_if_paused`.

    The trial drops ``.paused_<trial_id>.ack`` next to each pause marker it
    holds on — including the persistent tune root (== this ``tune_dir``) — the
    instant it reaches the boundary. This is the deterministic pause signal the
    progress.csv row-growth heuristic misses when a trial holds *before*
    appending a post-pause row (it would otherwise poll until timeout). Only
    acks touched at/after the pause request count (``since_ts``), so a stale ack
    from a crashed prior run is ignored.
    """
    dirs: set[Path] = {tune_dir} | {c.parent for c in csvs}
    found: dict[str, Path] = {}
    for d in dirs:
        try:
            acks = sorted(d.glob(".paused_*.ack"))
        except OSError:
            continue
        for ack in acks:
            try:
                # +1s slack for coarse filesystem mtime granularity.
                if ack.stat().st_mtime + 1.0 >= since_ts:
                    found.setdefault(ack.name, ack)
            except OSError:
                continue
    return list(found.values())


def _ack_trial_id(ack: Path) -> str:
    """Extract the trial id embedded in a ``.paused_<id>.ack`` filename."""
    return ack.name[len(".paused_"):-len(".ack")]


def _trial_is_acked(csv: Path, acks: list[Path]) -> bool:
    """True if some ack belongs to this trial. The trial dir is
    ``train_trial_<trial_id>_<num>_...``, so anchor the id at the start of the
    post-prefix stem (followed by ``_`` or end) rather than a loose substring.
    A loose ``in`` match would (a) let a degenerate ``trial_id="trial"`` fallback
    match EVERY ``train_trial_*`` dir, and (b) let a sibling id that is a prefix
    of another (non-zero-padded) collide. Anchoring removes both."""
    name = csv.parent.name
    stem = name[len("train_trial_"):] if name.startswith("train_trial_") else name
    for a in acks:
        tid = _ack_trial_id(a)
        if stem == tid or stem.startswith(f"{tid}_"):
            return True
    return False


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


_PROCFS = Path("/proc")


def _pid_exists(pid: int) -> bool:
    """Is ``pid`` a live process — counting a zombie as gone.

    ⚑ DELIBERATE DUPLICATE of ``chess_anti_engine.tune.process_cleanup._pid_exists``,
    kept in sync by hand. This script is documented (README, AGENTS.md) as
    ``python3 scripts/graceful_restart.py`` with no ``PYTHONPATH=.``, so it must
    stay stdlib-only; importing the shared helper would break the documented
    invocation. If you change one, change the other.

    The zombie case is the reason both read the process state instead of
    trusting ``os.kill(pid, 0)``, which succeeds against a corpse. The caller
    below asks "did the SIGTERM take": on a zombie tuner the bare kill-probe
    says "still running", so it burns the full 30 s and then reports
    "did not exit after SIGTERM; not resuming" about a process that did exit.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's process; alive as far as we can tell
    try:
        stat = (_PROCFS / str(int(pid)) / "stat").read_bytes()
    except OSError:
        # Entry gone under a real procfs means the process is gone. With no
        # procfs at all, keep the os.kill answer rather than reporting every
        # live process dead.
        return not _PROCFS.is_dir()
    # "<pid> (<comm>) <state> ..." — comm is arbitrary bytes and may contain
    # spaces and parentheses, so the state follows the LAST ')'. Anything
    # unparseable reads as alive, the conservative direction.
    close = stat.rfind(b")")
    if close < 0:
        return True
    return stat[close + 2 : close + 3] != b"Z"


def _required_paused_count(active_count: int, wait_arg: int) -> int:
    active = max(0, int(active_count))
    wait = int(wait_arg)
    if wait > 0:
        return min(wait, active)
    return active


_REPO_ROOT = Path(__file__).resolve().parent.parent

# Must stay argv-identical to train.sh's own C-extension gate. A preflight
# STRICTER than train.sh blocks restarts that would have succeeded, which is
# worse than the bug this guards against; LOOSER, and it fails to catch it.
# test_preflight_argv_matches_train_sh pins the two together.
_C_EXT_CHECK_ARGV = (
    "scripts/check_c_extensions_fresh.py", "--quiet",
    "--min-gcc-major", "15", "--require-production-recipe",
)

_DEFAULT_RESUME_CMD = "./scripts/train.sh restart"


def _resume_preflight() -> list[str]:
    """Return reasons ``./scripts/train.sh restart`` would fail, empty if clean.

    Mirrors train.sh's own gate (same flags) so preflight and the real start
    cannot drift into disagreeing.
    """
    if os.environ.get("TRAIN_SKIP_C_EXT_CHECK") == "1":
        return []
    proc = subprocess.run(
        # "python3" (not sys.executable) so the interpreter matches the one
        # train.sh will use — a venv on a different minor version would look
        # for differently-tagged .so files and block a restart that works.
        ["python3", *_C_EXT_CHECK_ARGV],
        cwd=_REPO_ROOT, env={**os.environ, "PYTHONPATH": "."},
        capture_output=True, text=True, check=False,
    )
    if proc.returncode == 0:
        return []
    detail = (proc.stdout + proc.stderr).strip()
    return [line.strip() for line in detail.splitlines() if line.strip()] or [
        f"check_c_extensions_fresh.py exited {proc.returncode}"
    ]


def _tuner_is_running() -> bool:
    """True when a tune driver process is alive, independent of any PID file."""
    return _find_tuner_pid() is not None


def _run_resume(resume_cmd: str, *, pause_targets: list[Path]) -> None:
    """Run the resume command, retrying once, and never exit quietly on failure.

    A failed resume means training is DOWN right now, so the operator needs
    the recovery steps on screen rather than a traceback.
    """
    for attempt in (1, 2):
        if attempt == 2 and _tuner_is_running():
            # train.sh's stop/start keys entirely off its PID file. If attempt 1
            # spawned the trainer but died before writing that file, a second
            # start would not find it to stop and would race a second trainer
            # onto the same tune dir and GPU.
            print("[graceful_restart] A trainer is already running — not retrying.")
            return
        print(f"[graceful_restart] Running: {resume_cmd} (attempt {attempt}/2)")
        proc = subprocess.run(resume_cmd, shell=True, cwd=_REPO_ROOT, check=False)
        if proc.returncode == 0:
            # train.sh start returns as soon as it writes the PID file, so a 0
            # exit does not mean the trainer survived startup (bad config, CUDA
            # init, stale-worker wedge). Confirm it is still alive, since
            # "exited 0 but training is down" is the failure this script exists
            # to prevent.
            time.sleep(30)
            if _tuner_is_running():
                return
            print("[graceful_restart] Resume exited 0 but no trainer is running.")
        else:
            print(f"[graceful_restart] Resume FAILED with exit {proc.returncode}.")
        if attempt == 1:
            time.sleep(10)
    print()
    print("!" * 72)
    print("[graceful_restart] TRAINING IS DOWN — resume failed twice.")
    print("  Recover manually:")
    for target in pause_targets:
        print(f"    rm -f {target}")
    print("    python3 scripts/build_production_extensions.py   # if C extensions are stale")
    print("    setsid ./scripts/train.sh start")
    print("!" * 72)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune-dir", default="runs/pbt2_small/tune",
                    help="Path to the Ray Tune experiment directory")
    ap.add_argument("--wait", type=int, default=1,
                    help="Require N active trials paused. Default 1 preserves old no-arg behavior.")
    ap.add_argument("--wait-all", action="store_true",
                    help="Require all active trials to pause before restarting.")
    ap.add_argument("--timeout-secs", type=int, default=0,
                    help="Maximum seconds to wait before failing. Default 0 waits indefinitely.")
    ap.add_argument("--grace-secs", type=int, default=-1,
                    help="Optional grace window for the boundary-edge case. Disabled by "
                         "default because an in-flight iteration also has no post-pause row. "
                         "Use only after manually verifying the trial is already paused.")
    ap.add_argument("--poll", type=int, default=15,
                    help="Polling interval in seconds")
    ap.add_argument("--no-auto-kill", dest="auto_kill", action="store_false",
                    help="Print status and exit without killing or restarting")
    ap.add_argument("--resume-cmd", default=_DEFAULT_RESUME_CMD,
                    help="Shell command to run after killing (default: ./scripts/train.sh restart)")
    args = ap.parse_args()

    if args.wait < 0:
        raise SystemExit("--wait must be >= 0")
    if args.timeout_secs < 0:
        raise SystemExit("--timeout-secs must be >= 0")
    if args.grace_secs < -1:
        raise SystemExit("--grace-secs must be >= -1")
    if args.poll <= 0:
        raise SystemExit("--poll must be > 0")

    tune_dir = Path(args.tune_dir)
    if not tune_dir.is_dir():
        # Try relative to repo root
        repo = Path(__file__).parent.parent
        tune_dir = repo / args.tune_dir
    if not tune_dir.is_dir():
        print(f"ERROR: tune dir not found: {args.tune_dir}", file=sys.stderr)
        sys.exit(1)
    if args.wait_all and args.wait != 1:
        raise SystemExit("--wait-all cannot be combined with --wait")

    pause_file = tune_dir / "pause.txt"
    auto_kill = args.auto_kill

    # Step 0: verify the resume can actually succeed BEFORE touching the live
    # trial. train.sh refuses to start on stale in-place C extensions, and we
    # used to discover that only after the tuner was already dead — leaving
    # training DOWN (2026-07-24, ~18 min of lost selfplay after a merge that
    # carried .c changes). Preflight is skipped for a custom --resume-cmd,
    # whose prerequisites we cannot know.
    if auto_kill and args.resume_cmd == _DEFAULT_RESUME_CMD:
        blockers = _resume_preflight()
        if blockers:
            print("[graceful_restart] Resume preflight FAILED — not touching the live run:")
            for blocker in blockers:
                print(f"  - {blocker}")
            sys.exit(1)
        print("[graceful_restart] Resume preflight OK.")

    # Step 1: create the pause marker(s). Drop one in tune_dir AND one in
    # every active trial dir — the trial checks both. Without the per-trial
    # markers a previous run pause-no-op'd silently when the actor's view of
    # tune_dir/pause.txt didn't fire its exists() check (root cause never
    # diagnosed; the per-trial marker is the belt-and-suspenders fix).
    pause_targets: list[Path] = [pause_file]
    pause_targets.extend(csv.parent / "pause.txt" for csv in _active_trials(tune_dir))

    for target in pause_targets:
        if target.exists():
            print(f"[graceful_restart] pause.txt already exists at {target}")
        else:
            target.write_text("graceful restart in progress\n")
            print(f"[graceful_restart] Created {target}")
    print("[graceful_restart] Trials will pause after their current iteration.")

    wait_desc = "all active trials" if args.wait_all else f"{int(args.wait)} active trial(s)"
    print(f"[graceful_restart] Waiting for {wait_desc} to stop appending rows to progress.csv...")
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
            if args.timeout_secs and time.time() - start >= args.timeout_secs:
                raise SystemExit("[graceful_restart] Timed out waiting for active trials.")
            time.sleep(args.poll)
            continue

        # Deterministic signal first: the trial writes a .paused_<id>.ack the
        # moment it holds, so a boundary-hold is caught even with no post-pause
        # progress row (the heuristic's blind spot). Row growth is the fallback
        # for trials running pre-ack code.
        acks = _pause_ack_files(tune_dir, csvs, pause_created_ts)
        idle_trials: list[tuple[Path, str]] = []
        observations: list[tuple[Path, int, int, str]] = []
        for csv in csvs:
            rc = _row_count(csv)
            snap = snapshot_rows.setdefault(csv, rc)  # late-arriving trial: anchor at first sight
            prev = prev_rows.get(csv, snap)
            if _trial_is_acked(csv, acks):
                state = "PAUSED (ack)"
                idle_trials.append((csv, state))
            elif rc > snap and rc == prev:
                state = "PAUSED"
                idle_trials.append((csv, state))
            elif (
                args.grace_secs >= 0
                and rc == snap
                and time.time() - pause_created_ts >= args.grace_secs
            ):
                state = "PAUSED-AT-BOUNDARY"
                idle_trials.append((csv, state))
            else:
                state = f"running rows {snap}->{rc}"
            observations.append((csv, snap, rc, state))
            prev_rows[csv] = rc

        # Print status
        elapsed = int(time.time() - start)
        required_paused = len(csvs) if args.wait_all else _required_paused_count(len(csvs), int(args.wait))
        need_label = "all" if args.wait_all else str(required_paused)
        print(f"[{elapsed:4d}s] {len(idle_trials)}/{len(csvs)} trials paused "
              f"(need {need_label}):")
        for csv, snap, rc, state in observations:
            trial = csv.parent.name.split("_")[2] + "_" + csv.parent.name.split("_")[3]
            print(f"         {trial}  rows@pause={snap} rows_now={rc}  {state}")

        if len(idle_trials) >= required_paused:
            print()
            print(f"[graceful_restart] {len(idle_trials)} trials are at a clean stopping point.")

            if auto_kill:
                pid = _find_tuner_pid()
                if pid:
                    print(f"[graceful_restart] Sending SIGTERM to tuner PID {pid}...")
                    os.kill(pid, signal.SIGTERM)
                    deadline = time.time() + 30.0
                    while _pid_exists(pid) and time.time() < deadline:
                        time.sleep(1)
                    if _pid_exists(pid):
                        print(f"[graceful_restart] Tuner PID {pid} did not exit after SIGTERM; not resuming.")
                        return
                else:
                    print("[graceful_restart] Could not find tuner PID — kill it manually.")
                    print("[graceful_restart] Leaving pause markers in place and not running resume.")
                    return

                for target in pause_targets:
                    if target.exists():
                        print(f"[graceful_restart] Removing {target}")
                        target.unlink(missing_ok=True)

                if args.resume_cmd:
                    time.sleep(5)  # let Ray finish shutting down
                    _run_resume(args.resume_cmd, pause_targets=pause_targets)
            else:
                print()
                print("  Next steps:")
                print("  1. Kill the tuner process (Ctrl+C or kill the run command)")
                print("  2. Remove all pause markers:")
                for target in pause_targets:
                    print(f"     rm {target}")
                print("  3. Restart with --resume")
                print()
                print(f"[graceful_restart] Watching until you kill — remove {pause_file} "
                      f"to let trials continue if you change your mind.")
                # Keep watching so user can see if trials start moving again
                try:
                    while True:
                        time.sleep(args.poll)
                        csvs2 = _active_trials(tune_dir)
                        acks2 = _pause_ack_files(tune_dir, csvs2, pause_created_ts)
                        still_idle = 0
                        for c in csvs2:
                            rc = _row_count(c)
                            snap = snapshot_rows.get(c, rc)
                            prev = prev_rows.get(c, snap)
                            # Ack is authoritative (matches the main loop); row
                            # growth + grace remain the fallback for pre-ack code.
                            if _trial_is_acked(c, acks2) or (
                                rc == prev
                                and (
                                    rc > snap
                                    or (
                                        args.grace_secs >= 0
                                        and time.time() - pause_created_ts >= args.grace_secs
                                    )
                                )
                            ):
                                still_idle += 1
                            prev_rows[c] = rc
                        elapsed2 = int(time.time() - start)
                        print(f"[{elapsed2:4d}s] still {still_idle}/{len(csvs2)} paused — "
                              f"safe to kill and restart with --resume")
                except KeyboardInterrupt:
                    print("\n[graceful_restart] Interrupted. "
                          f"Remember to rm {pause_file} if you are not restarting.")
            return

        if args.timeout_secs and time.time() - start >= args.timeout_secs:
            raise SystemExit("[graceful_restart] Timed out before all trials paused.")

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
