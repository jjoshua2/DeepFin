from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def build_worker_command(*, worker_args: list[str], worker_dir: Path, shared_cache_dir: Path) -> list[str]:
    if any(arg == "--work-dir" or arg.startswith("--work-dir=") for arg in worker_args):
        raise ValueError("worker_pool manages per-worker work dirs; do not pass --work-dir to child workers")
    if any(arg == "--shared-cache-dir" or arg.startswith("--shared-cache-dir=") for arg in worker_args):
        raise ValueError("worker_pool manages the shared cache dir; do not pass --shared-cache-dir to child workers")
    return [
        sys.executable,
        "-m",
        "chess_anti_engine.worker",
        *worker_args,
        "--work-dir",
        str(worker_dir),
        "--shared-cache-dir",
        str(shared_cache_dir),
    ]


def _terminate_children(children: list[subprocess.Popen[bytes]], *, timeout_s: float = 5.0) -> None:
    for proc in children:
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    deadline = time.time() + float(timeout_s)
    for proc in children:
        if proc.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    for proc in children:
        if proc.poll() is None:
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Spawn multiple chess_anti_engine.worker processes on one machine.",
    )
    ap.add_argument("--workers", type=int, default=1, help="Number of worker processes to launch.")
    ap.add_argument(
        "--pool-work-dir",
        type=str,
        default="worker_pool",
        help="Base work dir for the worker pool; each child gets pool_work_dir/worker_XX.",
    )
    ap.add_argument(
        "--stagger-seconds",
        type=float,
        default=1.0,
        help="Delay between worker launches to avoid stampeding startup.",
    )
    ap.add_argument(
        "--respawn",
        action="store_true",
        help="Respawn workers that exit unexpectedly.",
    )
    ap.add_argument(
        "--restart-delay-seconds",
        type=float,
        default=3.0,
        help="Delay before respawning a dead worker when --respawn is enabled.",
    )
    args, worker_args = ap.parse_known_args()

    num_workers = int(args.workers)
    if num_workers <= 0:
        raise SystemExit("--workers must be >= 1")

    pool_dir = Path(args.pool_work_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    shared_cache_dir = pool_dir / "shared_cache"
    shared_cache_dir.mkdir(parents=True, exist_ok=True)

    stop = False

    def _handle_stop(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    children: list[subprocess.Popen[bytes]] = []
    commands: list[list[str]] = []
    for idx in range(num_workers):
        worker_dir = pool_dir / f"worker_{idx:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_worker_command(
            worker_args=list(worker_args),
            worker_dir=worker_dir,
            shared_cache_dir=shared_cache_dir,
        )
        commands.append(cmd)
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        children.append(proc)
        if idx + 1 < num_workers and float(args.stagger_seconds) > 0.0:
            time.sleep(float(args.stagger_seconds))

    try:
        while not stop:
            time.sleep(1.0)
            if not bool(args.respawn):
                if any(proc.poll() is not None for proc in children):
                    stop = True
                continue
            for idx, proc in enumerate(children):
                if proc.poll() is None:
                    continue
                if stop:
                    break
                time.sleep(float(args.restart_delay_seconds))
                if stop:
                    break
                children[idx] = subprocess.Popen(commands[idx], cwd=str(Path(__file__).resolve().parents[1]))
    finally:
        _terminate_children(children)


if __name__ == "__main__":
    main()
