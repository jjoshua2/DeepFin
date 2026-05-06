"""Profile distributed tune configs systematically.

Tests combinations of workers_per_trial, sf_workers, and selfplay_batch
by running a short tune session and measuring GPU utilization, CPU load,
and positions generated.

Usage:
    python3 scripts/profile_distributed.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "pbt2_small.yaml"
WORK_DIR = REPO_ROOT / "runs" / "profile_distributed"
SERVER_DIR = WORK_DIR / "server"
REPLAY_DIR = WORK_DIR / "replay"

# Combinations to test (workers_per_trial, sf_workers, selfplay_batch)
CONFIGS = [
    # (workers, sf_workers, batch)
    (1, 1, 64),
    (1, 2, 64),
    (1, 3, 64),
    (2, 1, 32),
    (2, 2, 32),
    (2, 2, 48),
    (2, 2, 64),
    (2, 3, 32),
    (3, 2, 32),
    (3, 2, 24),
    (4, 2, 16),
    (4, 1, 32),
]

# How long to let each config run (seconds) — enough for selfplay to start
WARMUP_SECONDS = 45
GPU_SAMPLES = 6
GPU_SAMPLE_INTERVAL = 5


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _reset_dir(path: Path) -> None:
    """Remove and recreate a benchmark-owned directory."""
    resolved = _resolve(path)
    if resolved in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"refusing to reset unsafe directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved, ignore_errors=True)
    resolved.mkdir(parents=True, exist_ok=True)


def cleanup(
    *,
    work_dir: Path | None = None,
    server_dir: Path | None = None,
    replay_dir: Path | None = None,
    stop_ray: bool = True,
) -> None:
    """Reset benchmark-owned dirs and optionally stop the Ray cluster."""
    work_dir = WORK_DIR if work_dir is None else work_dir
    server_dir = SERVER_DIR if server_dir is None else server_dir
    replay_dir = REPLAY_DIR if replay_dir is None else replay_dir

    time.sleep(2)
    if stop_ray:
        try:
            subprocess.run(["ray", "stop", "--force"], capture_output=True)
        except FileNotFoundError:
            pass
    time.sleep(3)
    for d in [server_dir, replay_dir]:
        _reset_dir(d)
    tune_dir = work_dir / "tune"
    if tune_dir.exists():
        shutil.rmtree(tune_dir, ignore_errors=True)


def make_config(
    workers: int,
    sf_workers: int,
    batch: int,
    *,
    base_config: Path | None = None,
    work_dir: Path | None = None,
    server_dir: Path | None = None,
    replay_dir: Path | None = None,
) -> Path:
    """Patch the base config with the given parameters."""
    import yaml

    base_config = BASE_CONFIG if base_config is None else base_config
    work_dir = WORK_DIR if work_dir is None else work_dir
    server_dir = SERVER_DIR if server_dir is None else server_dir
    replay_dir = REPLAY_DIR if replay_dir is None else replay_dir

    with open(base_config) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("selfplay", {})
    cfg.setdefault("tune", {})

    cfg["selfplay"]["selfplay_batch"] = batch
    # Use only 1 concurrent trial for profiling to isolate per-trial perf
    cfg["tune"]["distributed_workers_per_trial"] = workers
    cfg["tune"]["distributed_worker_sf_workers"] = sf_workers
    cfg["tune"]["max_concurrent_trials"] = 4
    cfg["tune"]["num_samples"] = 4
    # Fast iteration: few games
    cfg["selfplay"]["games_per_iter"] = 150
    cfg["selfplay"]["games_per_iter_start"] = 32
    # Override dirs
    cfg["work_dir"] = str(work_dir / "tune")
    cfg["tune"]["distributed_server_root_override"] = str(server_dir)
    cfg["tune"]["tune_replay_root_override"] = str(replay_dir)

    out = work_dir / "profile_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return out


def sample_gpu(n_samples: int = GPU_SAMPLES, interval: float = GPU_SAMPLE_INTERVAL) -> list[int]:
    """Sample GPU utilization n times."""
    samples = []
    for _ in range(n_samples):
        time.sleep(interval)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        try:
            val = int(result.stdout.strip().replace(" %", "").replace("%", ""))
            samples.append(val)
        except ValueError:
            pass
    return samples


def sample_cpu() -> dict:
    """Get CPU load and idle percentage."""
    result = subprocess.run(
        ["top", "-bn1"], capture_output=True, text=True,
    )
    lines = result.stdout.strip().split("\n")
    info = {}
    for line in lines[:5]:
        if "load average" in line:
            m = re.search(r"load average:\s*([\d.]+)", line)
            if m:
                info["load_avg"] = float(m.group(1))
        if "%Cpu" in line:
            m = re.search(r"([\d.]+)\s*id", line)
            if m:
                info["cpu_idle"] = float(m.group(1))
            m = re.search(r"([\d.]+)\s*ni", line)
            if m:
                info["cpu_nice"] = float(m.group(1))
    return info


def count_processes() -> dict:
    """Count stockfish and worker processes."""
    sf = subprocess.run(
        "ps aux | grep stockfish | grep -v grep | wc -l",
        shell=True, capture_output=True, text=True,
    )
    workers = subprocess.run(
        "ps aux | grep 'chess_anti_engine.worker' | grep -v grep | wc -l",
        shell=True, capture_output=True, text=True,
    )
    return {
        "sf_procs": int(sf.stdout.strip()),
        "worker_procs": int(workers.stdout.strip()),
    }


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_config(
    workers: int,
    sf_workers: int,
    batch: int,
    *,
    base_config: Path | None = None,
    work_dir: Path | None = None,
    server_dir: Path | None = None,
    replay_dir: Path | None = None,
    warmup_seconds: int = WARMUP_SECONDS,
    gpu_samples: int = GPU_SAMPLES,
    gpu_sample_interval: float = GPU_SAMPLE_INTERVAL,
    stop_ray: bool = True,
) -> dict:
    """Run a single config and measure performance."""
    label = f"{workers}w×{sf_workers}sf×{batch}b"
    print(f"\n{'='*60}")
    print(f"  Testing: {label}")
    print(f"{'='*60}")

    work_dir = WORK_DIR if work_dir is None else work_dir
    server_dir = SERVER_DIR if server_dir is None else server_dir
    replay_dir = REPLAY_DIR if replay_dir is None else replay_dir

    cleanup(work_dir=work_dir, server_dir=server_dir, replay_dir=replay_dir, stop_ray=stop_ray)
    cfg_path = make_config(
        workers,
        sf_workers,
        batch,
        base_config=base_config,
        work_dir=work_dir,
        server_dir=server_dir,
        replay_dir=replay_dir,
    )

    # Launch
    proc = subprocess.Popen(  # pylint: disable=consider-using-with  # long-lived benchmark subprocess, terminated later
        [sys.executable, "-m", "chess_anti_engine.run",
         "--config", str(cfg_path), "--mode", "tune"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
        start_new_session=True,
    )

    # Wait for warmup
    print(f"  Warming up {warmup_seconds}s...", flush=True)
    time.sleep(warmup_seconds)

    # Check it's still running
    if proc.poll() is not None:
        print(f"  ERROR: Process exited with code {proc.returncode}")
        return {"label": label, "error": True}

    # Sample GPU
    print(f"  Sampling GPU ({gpu_samples}x @ {gpu_sample_interval}s)...", flush=True)
    gpu_values = sample_gpu(gpu_samples, gpu_sample_interval)
    cpu_info = sample_cpu()
    procs = count_processes()

    # Kill
    _terminate_process_tree(proc)

    result = {
        "label": label,
        "workers": workers,
        "sf_workers": sf_workers,
        "batch": batch,
        "gpu_samples": gpu_values,
        "gpu_avg": sum(gpu_values) / len(gpu_values) if gpu_values else 0,
        "gpu_min": min(gpu_values) if gpu_values else 0,
        "gpu_max": max(gpu_values) if gpu_values else 0,
        **cpu_info,
        **procs,
        "error": False,
    }

    print(f"  GPU: {result['gpu_avg']:.0f}% avg ({result['gpu_min']}-{result['gpu_max']}%)")
    print(f"  CPU: load={result.get('load_avg', '?')}, idle={result.get('cpu_idle', '?')}%")
    print(f"  Procs: {result['sf_procs']} SF, {result['worker_procs']} workers")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--server-dir", type=Path, default=None)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--warmup-seconds", type=int, default=WARMUP_SECONDS)
    parser.add_argument("--gpu-samples", type=int, default=GPU_SAMPLES)
    parser.add_argument("--gpu-sample-interval", type=float, default=GPU_SAMPLE_INTERVAL)
    parser.add_argument("--no-ray-stop", action="store_true", help="do not run ray stop --force during cleanup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir
    server_dir = args.server_dir or work_dir / "server"
    replay_dir = args.replay_dir or work_dir / "replay"
    stop_ray = not args.no_ray_stop

    print("Distributed Profile - Testing worker/SF/batch combinations")
    print(f"Base config: {args.base_config}")
    print(f"Work dir: {work_dir}")
    print(f"Server dir: {server_dir}")
    print(f"Replay dir: {replay_dir}")
    print(f"Warmup: {args.warmup_seconds}s, Sample: {args.gpu_samples}x @ {args.gpu_sample_interval}s")
    print(f"Configs to test: {len(CONFIGS)}")
    est_minutes = len(CONFIGS) * (args.warmup_seconds + args.gpu_samples * args.gpu_sample_interval + 15) / 60
    print(f"Estimated time: {est_minutes:.0f} minutes")
    print()

    results = []
    for workers, sf_workers, batch in CONFIGS:
        r = run_config(
            workers,
            sf_workers,
            batch,
            base_config=args.base_config,
            work_dir=work_dir,
            server_dir=server_dir,
            replay_dir=replay_dir,
            warmup_seconds=args.warmup_seconds,
            gpu_samples=args.gpu_samples,
            gpu_sample_interval=args.gpu_sample_interval,
            stop_ray=stop_ray,
        )
        results.append(r)

    cleanup(work_dir=work_dir, server_dir=server_dir, replay_dir=replay_dir, stop_ray=stop_ray)

    # Summary table
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<18} {'GPU avg':>8} {'GPU range':>12} {'CPU load':>9} {'CPU idle':>9} {'SF':>4} {'Workers':>8}")
    print("-" * 80)

    # Sort by GPU avg descending
    for r in sorted(results, key=lambda x: x.get("gpu_avg", 0), reverse=True):
        if r.get("error"):
            print(f"{r['label']:<18} {'ERROR':>8}")
            continue
        gpu_range = f"{r['gpu_min']}-{r['gpu_max']}%"
        print(
            f"{r['label']:<18} {r['gpu_avg']:>7.0f}% {gpu_range:>12} "
            f"{r.get('load_avg', 0):>8.1f} {r.get('cpu_idle', 0):>8.1f}% "
            f"{r['sf_procs']:>4} {r['worker_procs']:>8}"
        )

    # Save results
    out_path = work_dir / "profile_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
