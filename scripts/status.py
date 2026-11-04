"""Show current training run status.

Usage:
    python3 scripts/status.py
    python3 scripts/status.py --run runs/pbt2_small

Run/process association is observed metadata, not immutable runtime identity.
Linux proc argv/cwd and readable supplied YAML enable automatic selection;
ambiguous or unavailable observations require --run. External Ray trials need a
matching params.json work_dir; relative metadata additionally needs one observed
process cwd. Durable files below an explicit run remain available without Ray.
The selected result directory also owns the displayed worker log: no global
session or trial-ID fallback is used for throughput.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

# ── helpers ──────────────────────────────────────────────────────────────────

class _ObservedRun(NamedTuple):
    pid: str
    config: str
    run_dir: Path
    cwd: Path


def _running_pids() -> list[str]:
    """Discover candidates only; ps text is not argv and cannot prove identity."""
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
    except Exception:
        return []
    return sorted({
        parts[1] for line in out.splitlines()
        if "chess_anti_engine.run" in line and len(parts := line.split()) > 10
        and parts[1].isdigit()
    })


def _observe_run(pid: str) -> _ObservedRun | None:
    """Read Linux process metadata without guessing quoted/relative paths.

    This is an observation, not an immutable copy of startup configuration:
    YAML and cwd can change. Unsupported or unreadable metadata stays unknown.
    Only these two root YAML keys affect run identity; run.py filters nulls and
    applies CLI overrides after YAML, with work_dir="runs" and mode="train".
    """
    try:
        proc = Path("/proc") / pid
        argv = proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        cwd = proc.joinpath("cwd").resolve(strict=True)
        module_index = argv.index("-m")
        if (not Path(argv[0]).name.startswith("python")
                or argv[module_index + 1] != "chess_anti_engine.run"
                or any(x not in {"-u", "-B", "-O", "-OO", "-E", "-s", "-S", "-I"}
                       for x in argv[1:module_index])):
            return None
        options: dict[str, str] = {}
        arguments = iter(argv[module_index + 2:])
        names = {"--config", "--work-dir", "--mode"}
        for arg in arguments:
            name, sep, value = arg.partition("=")
            if name in names:
                options[name] = value if sep else next(arguments)
                if not options[name] or (not sep and options[name].startswith("-")):
                    return None
            elif name == "--" or (name.startswith("--") and any(n.startswith(name) for n in names)):
                # The real parser accepts abbreviations. Do not guess how an
                # abbreviated option competes with its many other arguments.
                return None
        config = ""
        defaults: dict[str, object] = {}
        if "--config" in options:
            import yaml
            config_path = (cwd / options["--config"]).resolve()
            loaded = yaml.safe_load(config_path.read_text())
            if loaded is not None and not isinstance(loaded, dict):
                return None
            defaults = loaded or {}
            config = str(config_path)
        mode: object = options.get("--mode", defaults.get("mode", "train"))
        if mode is None:
            mode = "train"
        work_dir: object = options.get("--work-dir", defaults.get("work_dir", "runs"))
        if work_dir is None:
            work_dir = "runs"
        if mode not in ("train", "tune") or not isinstance(work_dir, str) or not work_dir:
            return None
        # run.py does not expand '~' in YAML/argv; resolve exactly like Path.
        return _ObservedRun(pid, config, (cwd / work_dir).resolve(), cwd)
    except (OSError, ValueError, IndexError, StopIteration, RuntimeError):
        return None
    except Exception:
        # Includes malformed YAML; never print config content or full argv.
        return None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return float("-inf")


def _ray_result_matches(path: Path, run_dir: Path, process_cwd: Path | None) -> bool:
    """Ray writes params.json beside result.json from the actual trial config."""
    try:
        params = json.loads(path.with_name("params.json").read_text())
        work_dir = params.get("work_dir")
        if not isinstance(work_dir, str) or not work_dir:
            return False
        root = Path(work_dir)
        if not root.is_absolute():
            if process_cwd is None:
                return False
            root = process_cwd / root
        return root.resolve() == (run_dir / "tune").resolve()
    except (OSError, ValueError, AttributeError, RuntimeError):
        return False


def _find_trial_result_jsons(run_dir: Path, process_cwd: Path | None = None) -> list[Path]:
    """Snapshot one matching Ray experiment, otherwise the selected durable root.

    Session PID names and recency only order candidates: each external trial
    must bind to the selected root via params.json. Relative metadata requires
    an unambiguous observed driver cwd. Archives under --run remain usable even
    without a process or Ray metadata (including moved durable run directories).
    """
    for session in sorted(Path("/tmp/ray").glob("session_*/"), key=_mtime, reverse=True):
        artifacts = session / "artifacts"
        try:
            experiments = sorted(artifacts.iterdir(), key=_mtime, reverse=True)
        except OSError:
            continue
        for experiment in experiments:
            results = [p for p in experiment.rglob("result.json")
                       if _ray_result_matches(p, run_dir, process_cwd)]
            if results:
                return sorted(results, key=_mtime)
    return sorted(run_dir.rglob("result.json"), key=_mtime)


def _trial_id_from_result_path(path: Path) -> str:
    for parent in [path.parent, *path.parents]:
        name = parent.name
        if not name.startswith("train_trial_"):
            continue
        parts = name.split("_")
        if len(parts) >= 4:
            return f"{parts[2]}_{parts[3]}"
        return name.removeprefix("train_trial_")
    return path.parent.name


def _latest_worker_log_line(trial_dir: Path) -> str:
    """Read only logs belonging to the exact selected result's trial directory."""
    for log in sorted((trial_dir / "distributed_workers").rglob("worker.log"),
                      key=_mtime, reverse=True):
        try:
            latest = ""
            with log.open(errors="replace") as stream:
                for line in stream:
                    if "batch done" in line:
                        latest = line.strip()
            if latest:
                return latest
        except OSError:
            pass
    return ""


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="Run directory (default: auto-detect from active process)")
    ap.add_argument("--all", action="store_true", help="Show all trials, not just most recent per experiment")
    args = ap.parse_args()

    pids = _running_pids()
    observed = [run for pid in pids if (run := _observe_run(pid)) is not None]
    if args.run:
        run_dir = Path(args.run).expanduser().resolve()
    elif len({run.run_dir for run in observed}) == 1 and len(observed) == len(pids):
        run_dir = observed[0].run_dir
    else:
        print("No unambiguous observed run. Use --run to specify.")
        return

    matching = [run for run in observed if run.run_dir == run_dir]
    cwds = {run.cwd for run in matching}
    process_cwd = next(iter(cwds)) if len(cwds) == 1 else None
    print(f"\n{'═'*72}")
    if len(matching) == 1:
        print(f"  Run:    {run_dir.name}   (PID {matching[0].pid}; observed metadata)")
        if matching[0].config:
            print(f"  Config: {matching[0].config}")
    elif matching:
        print(f"  Run:    {run_dir.name}   (multiple matching drivers; observed metadata)")
    else:
        print(f"  Run:    {run_dir.name}   (no matching driver observed)")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*72}\n")

    # Load all result.json files
    result_files = _find_trial_result_jsons(run_dir, process_cwd=process_cwd)
    if not result_files:
        print("  No result.json files found.")
        return

    # Group by trial_id, keep only latest result per trial
    by_trial: dict[str, dict] = {}
    trial_dirs: dict[str, Path] = {}
    for rf in result_files:
        try:
            lines = rf.read_text().splitlines()
            if not lines:
                continue
            r = json.loads(lines[-1])
            tid = str(r.get("trial_id") or _trial_id_from_result_path(rf))
            existing = by_trial.get(tid)
            if existing is None or int(r.get("iter", 0)) >= int(existing.get("iter", 0)):
                r["trial_id"] = tid
                by_trial[tid] = r
                trial_dirs[tid] = rf.parent
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
            pol_o    = float(r.get("policy_loss_phase_open", 0.0))
            pol_m    = float(r.get("policy_loss_phase_mid", 0.0))
            pol_e    = float(r.get("policy_loss_phase_end", 0.0))
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
        line = _latest_worker_log_line(trial_dirs[tid])
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
            # Counters moved out of users.json into a sibling stats file. Fall
            # back to the users file for a DB that has not been migrated yet,
            # so this keeps working either side of the split.
            stats_json = users_json.with_name(f"{users_json.stem}.stats.json")
            stats = json.loads(stats_json.read_text()) if stats_json.exists() else {}
            print(f"\n  Contributors ({users_json}):")
            for uname, u in sorted(users.items()):
                from_stats = stats.get(uname)
                counters = from_stats if isinstance(from_stats, dict) else u
                total_pos = int(counters.get("total_positions", 0))
                uploads = int(counters.get("uploads", 0))
                print(f"    {uname:<20}  uploads={uploads:<6}  positions={total_pos:>12,}")
                for machine, m in sorted(counters.get("machines", {}).items()):
                    mpos = int(m.get("positions", 0))
                    mup  = int(m.get("uploads", 0))
                    print(f"      {machine:<18}  uploads={mup:<6}  positions={mpos:>12,}")
        except Exception:
            pass

    print()


if __name__ == "__main__":
    main()
