"""Benchmark the UCI engine on a set of positions.

Use: run during a training stop-and-resume window to measure current search
throughput without training contention on the GPU. Loads a checkpoint via
the real UCI path so we're benchmarking what actually ships, not a synthetic
harness.

Reports per-position sims/sec + wall-clock for a fixed node budget, so we can
track how throughput changes when we tune topk / chunk_sims / evaluator
settings. Runs on whatever device the machine has (cpu/cuda); pass --device
explicitly to pin.

Example:
  PYTHONPATH=. python3 scripts/bench_uci_engine.py \\
      --checkpoint runs/pbt2_small/tune/train_trial_*/checkpoint_NNN \\
      --nodes 1024 --repeats 3
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

from chess_anti_engine.uci.subprocess_client import LineReader as _LineReader
from chess_anti_engine.uci.subprocess_client import send_line as _send

# Matches the DEBUG profile line from gumbel_c.py. Parses out fields we care
# about for the bench summary: GPU call count, total positions fed to GPU,
# and wall-time split between tree/GPU/glue.
_PROFILE_RE = re.compile(
    r"gumbel profile \(n_boards=(?P<n_boards>\d+)\): "
    r"total=(?P<total>[\d.]+)s "
    r"init=(?P<init>[\d.]+) prep=(?P<prep>[\d.]+) "
    r"gpu=(?P<gpu>[\d.]+)\((?P<gpu_calls>\d+)calls,(?P<gpu_pos>\d+)pos,avg=(?P<avg_batch>[\d.]+)\) "
    r"finish=(?P<finish>[\d.]+) score=(?P<score>[\d.]+) policy=(?P<policy>[\d.]+) glue=(?P<glue>[-\d.]+)"
)


# Representative positions covering the main phases + one tactical spike.
# UCI spec: every position is a full FEN, startpos handled specially.
_POSITIONS: list[tuple[str, str]] = [
    ("startpos", "startpos"),
    ("mid_game_open", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 6"),
    ("mid_game_closed", "r2q1rk1/pppnbppp/3pbn2/4p3/4P3/1NNPB3/PPPQBPPP/R4RK1 w - - 0 9"),
    ("tactical", "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 5"),
    ("endgame_kp", "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"),
]


def _float_from_result(result: dict[str, object], key: str, default: float = 0.0) -> float:
    value = result.get(key, default)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _spawn(checkpoint: str, device: str, *,
           chunk_sims: int, topk: int, max_batch: int,
           walkers: int = 1, coalesce: bool = True,
           log_level: str = "WARNING") -> subprocess.Popen[str]:
    cmd = [sys.executable, "-u", "-m", "chess_anti_engine.uci",
           "--checkpoint", checkpoint, "--device", device,
           "--chunk-sims", str(chunk_sims),
           "--topk", str(topk),
           "--max-batch", str(max_batch),
           "--walkers", str(walkers),
           "--log-level", log_level]
    if not coalesce:
        cmd.append("--no-coalesce")
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )


def _run_one(proc: subprocess.Popen[str], reader: _LineReader, *,
             fen: str, nodes: int, timeout_s: float) -> dict[str, object]:
    _send(proc, f"position {fen}" if fen == "startpos" else f"position fen {fen}")
    _send(proc, f"go nodes {nodes}")
    t0 = time.monotonic()
    lines = reader.read_until("bestmove", timeout_s=timeout_s)
    elapsed = time.monotonic() - t0

    # Parse last info line for nps, nodes, depth, etc.
    last_info = next((l for l in reversed(lines) if l.startswith("info ")), "")
    tokens = last_info.split()
    info: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("depth", "nodes", "nps", "time", "score", "seldepth", "hashfull"):
            if tok == "score":
                info["score_kind"] = tokens[i + 1]
                info["score_val"] = tokens[i + 2]
                i += 3
            else:
                info[tok] = tokens[i + 1]
                i += 2
        else:
            i += 1

    # Aggregate any gumbel DEBUG profile lines this search emitted.
    profiles = [_PROFILE_RE.search(l) for l in lines]
    profiles = [p for p in profiles if p is not None]
    prof_agg: dict[str, float] = {}
    if profiles:
        prof_agg["n_searches"] = float(len(profiles))
        prof_agg["gpu_calls"] = sum(float(p["gpu_calls"]) for p in profiles)
        prof_agg["gpu_pos"] = sum(float(p["gpu_pos"]) for p in profiles)
        prof_agg["total_s"] = sum(float(p["total"]) for p in profiles)
        prof_agg["gpu_s"] = sum(float(p["gpu"]) for p in profiles)
        prof_agg["prep_s"] = sum(float(p["prep"]) for p in profiles)
        prof_agg["finish_s"] = sum(float(p["finish"]) for p in profiles)
        prof_agg["avg_batch"] = prof_agg["gpu_pos"] / max(1.0, prof_agg["gpu_calls"])

    return {
        "wall_s": round(elapsed, 3),
        "sims_per_s": round(nodes / elapsed, 1) if elapsed > 0 else 0,
        "info_nodes": int(info.get("nodes", 0) or 0),
        "info_nps": int(info.get("nps", 0) or 0),
        "info_time_ms": int(info.get("time", 0) or 0),
        "info_depth": int(info.get("depth", 0) or 0),
        "bestmove": next((l.split()[1] for l in lines if l.startswith("bestmove ")), ""),
        "profile": prof_agg,
    }


def _run_config(
    checkpoint: str, device: str, *,
    nodes: int, repeats: int, timeout_s: float,
    chunk_sims: int, topk: int, max_batch: int,
    label: str,
    walkers: int = 1, coalesce: bool = True,
    log_level: str = "WARNING",
) -> None:
    proc = _spawn(checkpoint, device,
                  chunk_sims=chunk_sims, topk=topk, max_batch=max_batch,
                  walkers=walkers, coalesce=coalesce,
                  log_level=log_level)
    reader = _LineReader(proc)
    _send(proc, "uci")
    reader.read_until("uciok", timeout_s=60.0)
    _send(proc, "isready")
    reader.read_until("readyok", timeout_s=60.0)

    coal_str = "" if walkers == 1 else f"  walkers={walkers}  coalesce={coalesce}"
    print(f"\n## {label}  chunk_sims={chunk_sims}  topk={topk}  max_batch={max_batch}{coal_str}")
    try:
        position_stats: dict[str, list[float]] = {}
        profile_runs: list[dict[str, float]] = []
        for pos_label, fen in _POSITIONS:
            _send(proc, "ucinewgame")
            for _ in range(repeats):
                result = _run_one(proc, reader, fen=fen, nodes=nodes, timeout_s=timeout_s)
                position_stats.setdefault(pos_label, []).append(_float_from_result(result, "sims_per_s"))
                prof = result.get("profile") or {}
                if isinstance(prof, dict) and prof:
                    profile_runs.append(prof)
        for pos_label, vals in position_stats.items():
            mean = sum(vals) / len(vals)
            print(f"  {pos_label:<20} sims/s avg={mean:>7.1f}  runs={vals}")
        if profile_runs:
            agg_calls = sum(p.get("gpu_calls", 0.0) for p in profile_runs)
            agg_pos = sum(p.get("gpu_pos", 0.0) for p in profile_runs)
            agg_total = sum(p.get("total_s", 0.0) for p in profile_runs)
            agg_gpu = sum(p.get("gpu_s", 0.0) for p in profile_runs)
            agg_prep = sum(p.get("prep_s", 0.0) for p in profile_runs)
            agg_finish = sum(p.get("finish_s", 0.0) for p in profile_runs)
            avg_batch = agg_pos / max(1.0, agg_calls)
            gpu_pct = 100.0 * agg_gpu / max(1e-9, agg_total)
            prep_pct = 100.0 * agg_prep / max(1e-9, agg_total)
            finish_pct = 100.0 * agg_finish / max(1e-9, agg_total)
            print(
                f"  profile: gpu_calls={int(agg_calls)}  gpu_pos={int(agg_pos)}  "
                f"avg_batch={avg_batch:.1f}  gpu={gpu_pct:.1f}%  "
                f"tree_prep={prep_pct:.1f}%  finish={finish_pct:.1f}%"
            )
    finally:
        _send(proc, "quit")
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--nodes", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--sweep", action="store_true",
                   help="sweep predefined (chunk_sims, topk, max_batch) configs")
    p.add_argument("--walker-sweep", action="store_true",
                   help="sweep --walkers {1,2,4,8} at the production chunk=512/topk=32/mb=1024 config. "
                        "Single-walker uses classic Gumbel; >1 switches to PUCT walker pool.")
    p.add_argument("--chunk-sims", type=int, default=32)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--walkers", type=int, default=1,
                   help="PUCT walker threads for a single --walkers config run (default 1 = Gumbel).")
    p.add_argument("--no-coalesce", dest="coalesce", action="store_false",
                   help="disable walker-call coalescing (only meaningful with --walkers > 1)")
    p.set_defaults(coalesce=True)
    p.add_argument("--log-level", default="WARNING",
                   help="DEBUG to see per-search gumbel profile (GPU calls, avg batch, time breakdown).")
    args = p.parse_args()

    print(f"# checkpoint={args.checkpoint}  device={args.device}  nodes={args.nodes}  repeats={args.repeats}")

    if args.sweep:
        # One variable at a time; first row is baseline so we can A/B against it.
        configs = [
            ("baseline",                      32,   16,  32),
            ("chunk_sims=128",                128,  16,  32),
            ("chunk_sims=512",                512,  16,  32),
            ("chunk_sims=nodes",              args.nodes, 16, args.nodes),  # one shot
            ("topk=8",                        32,    8,  32),
            ("topk=32",                       32,   32,  64),   # max_batch must accommodate topk
            ("max_batch=128",                 32,   16, 128),
            ("max_batch=512",                 32,   16, 512),
            ("chunk=nodes + topk=32 + mb=512", args.nodes, 32, max(512, args.nodes)),
        ]
        for label, cs, tk, mb in configs:
            _run_config(
                args.checkpoint, args.device,
                nodes=args.nodes, repeats=args.repeats, timeout_s=args.timeout_s,
                chunk_sims=cs, topk=tk, max_batch=mb, label=label,
                log_level=args.log_level,
            )
    elif args.walker_sweep:
        # Production chunk/topk/mb (2026-04-21 sweep winners). Vary walkers only.
        # walkers=1 runs the classic Gumbel path; >1 switches to PUCT walker
        # pool with virtual loss + batch coalescing.
        walker_configs = [
            ("walkers=1 (Gumbel)",         1,  True),
            ("walkers=2 + coalesce",       2,  True),
            ("walkers=4 + coalesce",       4,  True),
            ("walkers=8 + coalesce",       8,  True),
            ("walkers=4 no-coalesce",      4,  False),
        ]
        for label, w, coal in walker_configs:
            _run_config(
                args.checkpoint, args.device,
                nodes=args.nodes, repeats=args.repeats, timeout_s=args.timeout_s,
                chunk_sims=512, topk=32, max_batch=1024,
                walkers=w, coalesce=coal, label=label,
                log_level=args.log_level,
            )
    else:
        _run_config(
            args.checkpoint, args.device,
            nodes=args.nodes, repeats=args.repeats, timeout_s=args.timeout_s,
            chunk_sims=args.chunk_sims, topk=args.topk, max_batch=args.max_batch,
            walkers=args.walkers, coalesce=args.coalesce,
            label="single",
            log_level=args.log_level,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
