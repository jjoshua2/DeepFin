"""Show current training run status.

Usage:
    python3 scripts/status.py
    python3 scripts/status.py --run runs/pbt2_fresh_run9
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

def _find_active_ray_session(active_pid: str | None = None) -> Path | None:
    """Return the Ray session dir for the active process.

    Prefers the session whose directory name ends with the active PID (Ray
    embeds the head-node PID in the session name). Falls back to the most
    recently modified session that has tune artifacts.
    """
    sessions = sorted(
        Path("/tmp/ray").glob("session_*/"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Primary: session name ends with the active PID.
    if active_pid:
        for s in sessions:
            if s.name.endswith(f"_{active_pid}"):
                return s
    # Fallback: most recently modified session that has tune artifacts.
    for s in sessions:
        artifacts = s / "artifacts"
        if not artifacts.exists():
            continue
        if any(True for _ in artifacts.rglob("progress.csv")):
            return s
    return sessions[0] if sessions else None


def _find_trial_result_jsons(run_dir: Path, active_pid: str | None = None) -> list[Path]:
    """Find result.json files for the active session only, fall back to run_dir."""
    session = _find_active_ray_session(active_pid=active_pid)
    results = []
    if session:
        # Only search the most recently modified experiment subdir so we don't
        # mix artifacts from a previous run that reused the same session.
        artifacts = session / "artifacts"
        if artifacts.exists():
            exp_dirs = sorted(artifacts.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for exp_dir in exp_dirs:
                if exp_dir.is_dir():
                    results = sorted(exp_dir.rglob("result.json"), key=lambda p: p.stat().st_mtime)
                    if results:
                        break
    if not results:
        results = sorted(run_dir.rglob("result.json"), key=lambda p: p.stat().st_mtime)
    return results


def _latest_worker_log_line(run_dir: Path, trial_id: str, active_pid: str | None = None) -> str:
    """Get last 'batch done' line from any worker for the given trial."""
    pattern = str(run_dir / "**" / f"*{trial_id}*" / "**" / "worker.log")
    logs = glob.glob(pattern, recursive=True)
    # Also search in active Ray session
    session = _find_active_ray_session(active_pid=active_pid)
    if session:
        pattern2 = str(session / "**" / f"*{trial_id}*" / "**" / "worker.log")
        logs += glob.glob(pattern2, recursive=True)
    best_line = ""
    best_time = 0.0
    for log in logs:
        try:
            mtime = os.path.getmtime(log)
            if mtime < best_time:
                continue
            with open(log, "r", errors="replace") as f:
                for line in f:
                    if "batch done" in line:
                        best_line = line.strip()
            best_time = mtime
        except Exception:
            pass
    return best_line


def _running_pids() -> dict[str, str]:
    """Map config path substring → PID for running chess_anti_engine processes."""
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
    except Exception:
        return {}
    result: dict[str, str] = {}
    for line in out.splitlines():
        if "chess_anti_engine.run" in line and "grep" not in line:
            parts = line.split()
            pid = parts[1]
            cmd = " ".join(parts[10:])
            result[cmd] = pid
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="Run directory (default: auto-detect from active process)")
    ap.add_argument("--all", action="store_true", help="Show all trials, not just most recent per experiment")
    args = ap.parse_args()

    # Find active run
    pids = _running_pids()
    active_config = ""
    active_pid = ""
    for cmd, pid in pids.items():
        if "--config" in cmd:
            cfg_idx = cmd.split().index("--config") + 1
            active_config = cmd.split()[cfg_idx]
            active_pid = pid
            break

    if args.run:
        run_dir = Path(args.run)
    elif active_config:
        # Infer run_dir from config work_dir or config name
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(Path(active_config).read_text())
            run_dir = Path(cfg.get("work_dir", f"runs/{Path(active_config).stem}"))
        except Exception:
            run_dir = Path(f"runs/{Path(active_config).stem}")
    else:
        print("No active run found. Use --run to specify.")
        return

    run_dir = run_dir.expanduser()

    print(f"\n{'═'*72}")
    if active_pid:
        print(f"  Run:    {run_dir.name}   (PID {active_pid})")
        print(f"  Config: {active_config}")
    else:
        print(f"  Run:    {run_dir.name}   (not currently running)")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*72}\n")

    # Load all result.json files
    result_files = _find_trial_result_jsons(run_dir, active_pid=active_pid)
    if not result_files:
        print("  No result.json files found.")
        return

    # Group by trial_id, keep only latest result per trial
    by_trial: dict[str, dict] = {}
    for rf in result_files:
        try:
            lines = rf.read_text().splitlines()
            if not lines:
                continue
            r = json.loads(lines[-1])
            tid = str(r.get("trial_id", rf.parent.name[:12]))
            existing = by_trial.get(tid)
            if existing is None or int(r.get("iter", 0)) >= int(existing.get("iter", 0)):
                by_trial[tid] = r
        except Exception:
            pass

    if not by_trial:
        print("  No valid results found.")
        return

    # Sort by EMA opponent_strength (falls back to raw) descending
    trials = sorted(by_trial.values(), key=lambda r: float(r.get("opponent_strength_ema", 0) or r.get("opponent_strength", 0)), reverse=True)

    # Header
    print(f"  {'trial':<14} {'iter':>4}  {'opp':>6}  {'ema':>6}  {'skill':>5}  {'ingest':>7}  {'train':>6}  {'steps':>5}  {'replay':>8}  {'loss':>7}  {'stale':>5}")
    print(f"  {'-'*14} {'-'*4}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*5}")

    for r in trials:
        tid     = str(r.get("trial_id", "?"))[:14]
        it      = r.get("iter", "?")
        opp     = float(r.get("opponent_strength", 0))
        opp_ema = float(r.get("opponent_strength_ema", 0))
        skill   = r.get("skill_level", "?")
        ingest  = float(r.get("ingest_ms", 0)) / 1000
        train   = float(r.get("train_ms", 0)) / 1000
        steps   = r.get("train_steps_used", "?")
        replay  = int(r.get("replay", 0))
        loss    = r.get("train_loss", r.get("loss", "?"))
        stale   = r.get("distributed_stale_games", "?")
        loss_str = f"{float(loss):.3f}" if isinstance(loss, (int, float)) else "?"
        ema_str = f"{opp_ema:.1f}" if opp_ema > 0 else "   -"
        print(f"  {tid:<14} {it:>4}  {opp:>6.1f}  {ema_str:>6}  {skill:>5}  {ingest:>6.0f}s  {train:>5.0f}s  {steps:>5}  {replay:>8,}  {loss_str:>7}  {stale:>5}")

    # Per-source loss split — how selfplay vs curriculum data is training the model.
    # Only meaningful once shards tagged with is_selfplay have cycled into the replay
    # window (frac_tagged_batch should climb toward 1.0 over ~2 days).
    has_split = any(
        any(k in r for k in ("policy_loss_selfplay", "policy_loss_curriculum", "frac_tagged_batch"))
        for r in trials
    )
    if has_split:
        print()
        print("  Per-source / per-phase loss split (train batch averages):")
        print(f"  {'trial':<14}  {'tagged%':>7}  {'sp%':>6}  {'pol_sp':>7}  {'pol_cur':>7}  {'pol_open':>8}  {'pol_mid':>7}  {'pol_end':>7}")
        print(f"  {'-'*14}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}")
        for r in trials:
            tid      = str(r.get("trial_id", "?"))[:14]
            tagged   = float(r.get("frac_tagged_batch", 0.0))
            frac_sp  = float(r.get("frac_is_selfplay_batch", 0.0))
            pol_sp   = float(r.get("policy_loss_selfplay", 0.0))
            pol_cur  = float(r.get("policy_loss_curriculum", 0.0))
            pol_o    = float(r.get("policy_loss_open", 0.0))
            pol_m    = float(r.get("policy_loss_mid", 0.0))
            pol_e    = float(r.get("policy_loss_end", 0.0))
            # Zero means "no tagged samples in this batch yet" — show as dash.
            def _f(v: float, width: int = 7) -> str:
                return f"{v:>{width}.3f}" if v > 0 else ("-" * (width - 3)).rjust(width)
            print(f"  {tid:<14}  {tagged:>7.1%}  {frac_sp:>6.1%}  {_f(pol_sp)}  {_f(pol_cur)}  {_f(pol_o, 8)}  {_f(pol_m)}  {_f(pol_e)}")

    # WDL calibration (holdout only) + realized-selfplay-rate in ingested shards.
    has_cal = any(
        any(k in r for k in ("test_wdl_brier", "test_wdl_ece", "ingest_frac_selfplay"))
        for r in trials
    )
    if has_cal:
        print()
        print("  WDL calibration (holdout) + shard ingest:")
        print(f"  {'trial':<14}  {'brier':>6}  {'ece':>6}  {'ingest_sp%':>10}  {'tagged_pos':>10}")
        print(f"  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")
        for r in trials:
            tid      = str(r.get("trial_id", "?"))[:14]
            brier    = r.get("test_wdl_brier")
            ece      = r.get("test_wdl_ece")
            ing_sp   = float(r.get("ingest_frac_selfplay", 0.0))
            tagged_n = int(r.get("ingest_is_selfplay_tagged", 0))
            brier_s  = f"{float(brier):.3f}" if isinstance(brier, (int, float)) and brier == brier else "   -"
            ece_s    = f"{float(ece):.3f}" if isinstance(ece, (int, float)) and ece == ece else "   -"
            print(f"  {tid:<14}  {brier_s:>6}  {ece_s:>6}  {ing_sp:>10.1%}  {tagged_n:>10,}")

    print()

    # Worker throughput from last batch log line
    print("  Worker throughput (latest batch per trial):")
    shown: set[str] = set()
    for r in trials:
        tid = str(r.get("trial_id", ""))
        if tid in shown:
            continue
        shown.add(tid)
        line = _latest_worker_log_line(run_dir, tid, active_pid=active_pid)
        if line:
            # Extract timestamp and key stats
            parts = line.split(" ", 3)
            ts = f"{parts[0]} {parts[1]}" if len(parts) > 2 else ""
            info = parts[-1] if parts else line
            # Shorten to fit
            info = info.replace("batch done: ", "").replace("rand=", "r=").replace("timeouts=", "to=")
            print(f"    {tid:<14}  [{ts}]  {info[:60]}")

    # Disk usage
    print()
    try:
        out = subprocess.check_output(["du", "-sh", str(run_dir)], text=True, stderr=subprocess.DEVNULL)
        print(f"  Disk: {out.strip()}")
    except Exception:
        pass

    # Server stats if users.json exists
    server_root = run_dir.parent / f"{run_dir.name}_server"
    users_json = server_root / "users.json"
    if not users_json.exists():
        users_json = run_dir / "server" / "users.json"
    if users_json.exists():
        try:
            users = json.loads(users_json.read_text())
            print(f"\n  Contributors ({users_json}):")
            for uname, u in sorted(users.items()):
                total_pos = int(u.get("total_positions", 0))
                uploads = int(u.get("uploads", 0))
                print(f"    {uname:<20}  uploads={uploads:<6}  positions={total_pos:>12,}")
                for machine, m in sorted(u.get("machines", {}).items()):
                    mpos = int(m.get("positions", 0))
                    mup  = int(m.get("uploads", 0))
                    print(f"      {machine:<18}  uploads={mup:<6}  positions={mpos:>12,}")
        except Exception:
            pass

    print()


if __name__ == "__main__":
    main()
