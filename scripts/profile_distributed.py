"""Profile distributed tune configs systematically.

Tests combinations of workers_per_trial, sf_workers, and selfplay_batch
by running a short tune session and measuring GPU utilization, CPU load,
and positions generated.

Usage:
    python3 scripts/profile_distributed.py
"""
from __future__ import annotations

import subprocess
import time
import json
import re
import sys
import shutil
from pathlib import Path

BASE_CONFIG = Path("configs/pbt2_fresh_run9.yaml")
WORK_DIR = Path("runs/profile_distributed")
SERVER_DIR = Path("/mnt/c/chess_active/profile_dist_server")
REPLAY_DIR = Path("/mnt/c/chess_active/profile_dist_replay")

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


def cleanup():
    """Kill any running chess processes and ray."""
    subprocess.run(
        "ps aux | grep 'chess_anti_engine' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null",
        shell=True, capture_output=True,
    )
    time.sleep(2)
    subprocess.run("ray stop --force 2>/dev/null", shell=True, capture_output=True)
    time.sleep(3)
    # Clean dirs
    for d in [SERVER_DIR, REPLAY_DIR]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    tune_dir = WORK_DIR / "tune"
    if tune_dir.exists():
        shutil.rmtree(tune_dir, ignore_errors=True)


def make_config(workers: int, sf_workers: int, batch: int) -> Path:
    """Patch the base config with the given parameters."""
    import yaml

    with open(BASE_CONFIG) as f:
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
    cfg["work_dir"] = str(WORK_DIR / "tune")
    cfg["tune"]["distributed_server_root_override"] = str(SERVER_DIR)
    cfg["tune"]["tune_replay_root_override"] = str(REPLAY_DIR)

    out = WORK_DIR / "profile_config.yaml"
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


def run_config(workers: int, sf_workers: int, batch: int) -> dict:
    """Run a single config and measure performance."""
    label = f"{workers}w×{sf_workers}sf×{batch}b"
    print(f"\n{'='*60}")
    print(f"  Testing: {label}")
    print(f"{'='*60}")

    cleanup()
    cfg_path = make_config(workers, sf_workers, batch)

    # Launch
    proc = subprocess.Popen(
        [sys.executable, "-m", "chess_anti_engine.run",
         "--config", str(cfg_path), "--mode", "tune"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(Path.cwd()),
    )

    # Wait for warmup
    print(f"  Warming up {WARMUP_SECONDS}s...", flush=True)
    time.sleep(WARMUP_SECONDS)

    # Check it's still running
    if proc.poll() is not None:
        print(f"  ERROR: Process exited with code {proc.returncode}")
        return {"label": label, "error": True}

    # Sample GPU
    print(f"  Sampling GPU ({GPU_SAMPLES}x @ {GPU_SAMPLE_INTERVAL}s)...", flush=True)
    gpu_samples = sample_gpu()
    cpu_info = sample_cpu()
    procs = count_processes()

    # Kill
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    result = {
        "label": label,
        "workers": workers,
        "sf_workers": sf_workers,
        "batch": batch,
        "gpu_samples": gpu_samples,
        "gpu_avg": sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0,
        "gpu_min": min(gpu_samples) if gpu_samples else 0,
        "gpu_max": max(gpu_samples) if gpu_samples else 0,
        **cpu_info,
        **procs,
        "error": False,
    }

    print(f"  GPU: {result['gpu_avg']:.0f}% avg ({result['gpu_min']}-{result['gpu_max']}%)")
    print(f"  CPU: load={result.get('load_avg', '?')}, idle={result.get('cpu_idle', '?')}%")
    print(f"  Procs: {result['sf_procs']} SF, {result['worker_procs']} workers")

    return result


def main():
    print("Distributed Profile - Testing worker/SF/batch combinations")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Warmup: {WARMUP_SECONDS}s, Sample: {GPU_SAMPLES}x @ {GPU_SAMPLE_INTERVAL}s")
    print(f"Configs to test: {len(CONFIGS)}")
    est_minutes = len(CONFIGS) * (WARMUP_SECONDS + GPU_SAMPLES * GPU_SAMPLE_INTERVAL + 15) / 60
    print(f"Estimated time: {est_minutes:.0f} minutes")
    print()

    results = []
    for workers, sf_workers, batch in CONFIGS:
        r = run_config(workers, sf_workers, batch)
        results.append(r)

    cleanup()

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
    out_path = WORK_DIR / "profile_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
