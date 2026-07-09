"""Run offline_replay_epoch.py under a hard per-process GPU memory cap.

The 512x16 bootstrap trains CONCURRENT with the live RL run, and
offline_replay_epoch.py has no --gpu-mem-fraction flag (retarget_retrain.py
does). Without a cap, an allocation spike here could OOM the LIVE trainer —
the same failure mode as the 2026-06-18 256-sim-arena crash. The cap makes
THIS process the one that dies on overflow, never the live run.

Usage:
  python3 scripts/bootstrap_memcap_wrapper.py --mem-fraction 0.30 -- <offline_replay_epoch args...>
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch


def main() -> None:
    argv = sys.argv[1:]
    frac = 0.30
    if argv and argv[0] == "--mem-fraction":
        frac = float(argv[1])
        argv = argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"[bootstrap-memcap] GPU memory capped at fraction {frac} on cuda:0", flush=True)
    # Prefer repo-root scripts/ path (cwd is repo root for the bootstrap driver).
    script = Path("scripts/offline_replay_epoch.py")
    if not script.is_file():
        script = Path(__file__).resolve().parent / "offline_replay_epoch.py"
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
