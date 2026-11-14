#!/usr/bin/env python3
"""Tournament prewarm: populate torch.compile disk cache + search-path graphs.

TCEC-style ops budget:
  * Offline prewarm (once per host/checkpoint/flag set): up to ~5–10 minutes.
  * Per-game process launch: isready should return within ~30 s after the
    disk cache is warm.

This script drives the real UCI path (same factories, same multi-GPU pool)
so the cache matches what cutechess will load.

Examples:
  # single GPU (default max-autotune)
  PYTHONPATH=. python3 scripts/uci_prewarm.py --checkpoint best_model.pt

  # multi-GPU PUCV (auto-downgrades cudagraph compile modes)
  PYTHONPATH=. python3 scripts/uci_prewarm.py --checkpoint best_model.pt \\
      --devices cuda:0,cuda:1 --multi-gpu-pucv

Engine flags left out here are left out of the engine command too, so the
engine runs on its own defaults (they cannot drift apart). Pass exactly the
engine flags you will register.

  # re-check warm-process isready latency after the cold pass
  PYTHONPATH=. python3 scripts/uci_prewarm.py --checkpoint best_model.pt \\
      --devices cuda:0,cuda:1 --multi-gpu-pucv --repeat 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _engine_cmd(
    *,
    checkpoint: str,
    device: str | None,
    devices: str | None,
    multi_gpu_pucv: bool,
    search_parallel: str | None,
    compile_mode: str | None,
    no_compile: bool,
    vl_gather: int | None,
    max_batch: int | None,
    chunk_sims: int | None,
    walkers: int | None,
    compile_cache_dir: str | None = None,
) -> list[str]:
    """The engine command line: an engine flag is passed only when given."""
    cmd = [
        sys.executable, "-u", "-m", "chess_anti_engine.uci",
        "--checkpoint", checkpoint,
        "--log-level", "INFO",
    ]
    for flag, value in (
        ("--chunk-sims", chunk_sims),
        ("--max-batch", max_batch),
        ("--walkers", walkers),
        ("--vl-gather", vl_gather),
        ("--search-parallel", search_parallel),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    if devices:
        cmd.extend(["--devices", devices])
    elif device is not None:
        cmd.extend(["--device", device])
    if compile_cache_dir is not None:
        cmd.extend(["--compile-cache-dir", compile_cache_dir])
    if multi_gpu_pucv:
        cmd.append("--multi-gpu-pucv")
    if no_compile:
        cmd.append("--no-compile")
    elif compile_mode is not None:
        cmd.extend(["--compile-mode", compile_mode])
    return cmd


def _run_once(
    *,
    cmd: list[str],
    nodes: int,
    ready_timeout_s: float,
    search_timeout_s: float,
) -> tuple[float, float, str]:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    stdin = proc.stdin  # local binding keeps the narrowed type inside closures

    def send(line: str) -> None:
        stdin.write(line + "\n")
        stdin.flush()

    stderr_tail: list[str] = []

    def drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_tail.append(line.rstrip())
            if len(stderr_tail) > 200:
                del stderr_tail[:100]

    import threading
    threading.Thread(target=drain_stderr, daemon=True).start()

    try:
        send("uci")
        t_uci = time.perf_counter()
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("EOF before uciok")
            if "uciok" in line:
                break
            if time.perf_counter() - t_uci > ready_timeout_s:
                raise TimeoutError("timed out waiting for uciok")

        t0 = time.perf_counter()
        send("isready")
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("EOF before readyok — engine load failed?")
            if "engine load failed" in line:
                raise RuntimeError(line.strip())
            if "readyok" in line:
                break
            if time.perf_counter() - t0 > ready_timeout_s:
                raise TimeoutError(
                    f"timed out waiting for readyok after {ready_timeout_s:.0f}s; "
                    f"last stderr:\n" + "\n".join(stderr_tail[-30:])
                )
        ready_s = time.perf_counter() - t0

        send("position startpos")
        t1 = time.perf_counter()
        send(f"go nodes {nodes}")
        best = ""
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("bestmove"):
                best = line.strip()
                break
            if "search error" in line:
                raise RuntimeError(line.strip())
            if time.perf_counter() - t1 > search_timeout_s:
                raise TimeoutError("timed out waiting for bestmove")
        search_s = time.perf_counter() - t1
        send("quit")
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        return ready_s, search_s, best
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    # Engine flags default to None = not passed: the engine's own default runs
    # (`--device auto` picks "cuda" when a GPU is visible). The engine inherits
    # this process's environment, DEEPFIN_COMPILE_CACHE included.
    p.add_argument("--device", default=None)
    p.add_argument("--devices", default=None,
                   help="comma-separated multi-GPU list, e.g. cuda:0,cuda:1")
    p.add_argument("--multi-gpu-pucv", action="store_true",
                   help="force multi-GPU PUCV pool (auto-on for >1 devices)")
    # Engine flags default to None = not passed: the engine's own default runs.
    p.add_argument("--search-parallel", choices=["pucv", "gumbel"], default=None)
    p.add_argument("--compile-mode", default=None,
                   help="requested compile mode (default: the engine's); multi-GPU may "
                        "rewrite cudagraph modes")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-cache-dir", default=None,
                   help="the engine's --compile-cache-dir; pass the one you register")
    p.add_argument("--vl-gather", type=int, default=None)
    p.add_argument("--max-batch", type=int, default=None)
    p.add_argument("--chunk-sims", type=int, default=None)
    p.add_argument("--walkers", type=int, default=None)
    p.add_argument("--nodes", type=int, default=2048,
                   help="fixed-node search after isready to exercise the pool")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeat full process launches (2+ measures warm-cache isready)")
    p.add_argument("--ready-timeout-s", type=float, default=900.0)
    p.add_argument("--search-timeout-s", type=float, default=120.0)
    return p


def main() -> int:
    args = build_parser().parse_args()

    multi = bool(args.multi_gpu_pucv or (args.devices and "," in args.devices))
    print(
        f"# prewarm checkpoint={args.checkpoint} "
        f"devices={args.devices or args.device or 'engine default'} "
        f"multi_gpu={multi} compile_mode={args.compile_mode or 'engine default'!r} "
        f"repeat={args.repeat}",
        flush=True,
    )
    for i in range(max(1, int(args.repeat))):
        print(f"\n## launch {i + 1}/{args.repeat}", flush=True)
        ready_s, search_s, best = _run_once(
            cmd=_engine_cmd(
                checkpoint=args.checkpoint,
                device=args.device,
                devices=args.devices,
                multi_gpu_pucv=bool(args.multi_gpu_pucv),
                search_parallel=args.search_parallel,
                compile_mode=args.compile_mode,
                no_compile=bool(args.no_compile),
                vl_gather=args.vl_gather,
                max_batch=args.max_batch,
                chunk_sims=args.chunk_sims,
                walkers=args.walkers,
                compile_cache_dir=args.compile_cache_dir,
            ),
            nodes=int(args.nodes),
            ready_timeout_s=float(args.ready_timeout_s),
            search_timeout_s=float(args.search_timeout_s),
        )
        print(
            f"  isready={ready_s:.1f}s  go_nodes={args.nodes} wall={search_s:.2f}s  {best}",
            flush=True,
        )
        if i == 0 and ready_s > 600:
            print(
                "  note: cold compile exceeded 10 min — check DEEPFIN_COMPILE_CACHE "
                "disk and free GPU memory; warm launches should still be fast",
                flush=True,
            )
        if i > 0 and ready_s > 45:
            print(
                "  WARNING: warm isready >45s — above the ~30s tournament target; "
                "inspect compile cache reuse and Hash/model load cost",
                flush=True,
            )
    print("\n# done — disk cache lives under DEEPFIN_COMPILE_CACHE "
          "(default ~/.cache/deepfin/worker_cache)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
