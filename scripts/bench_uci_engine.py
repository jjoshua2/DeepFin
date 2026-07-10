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
_PUCV_PROFILE_RE = re.compile(
    r"multi_gpu_pucv target=(?P<target>\d+) leaves=(?P<leaves>\d+) "
    r"batches=(?P<batches>\d+) avg_batch=(?P<avg_batch>[\d.]+) "
    r"wall=(?P<wall>[\d.]+)s worker_leaves=\[(?P<worker_leaves>[^\]]*)\]"
)
_PUCV_INFO_RE = re.compile(
    r"pucv(?: leaves=(?P<leaves>\d+) batches=(?P<batches>\d+) "
    r"avg_batch=(?P<avg_batch>[\d.]+) max_batch=(?P<max_batch>\d+) "
    r"workers=(?P<workers>[^ ]+))? pending=(?P<pending>[-\w]+)"
    r"(?: cache=(?P<cache_hits>\d+)/(?P<cache_requests>\d+)"
    r"\((?P<cache_rate>[\d.]+)%\))?"
)
_GIL_PROFILE_RE = re.compile(
    r"gil_profile .*?threads=(?P<threads>\d+) devices=(?P<devices>\d+) .*?"
    r"samples=(?P<samples>\d+) rate=(?P<rate>[\d.]+)/s "
    r"delay_mean=(?P<mean>[\d.]+)ms .*?p95<=(?P<p95>[\d.]+)ms "
    r"p99<=(?P<p99>[\d.]+)ms max=(?P<max>[\d.]+)ms "
    r"over1ms=(?P<over1>[\d.]+)% over5ms=(?P<over5>[\d.]+)%"
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
           devices: str | None = None,
           chunk_sims: int, topk: int, max_batch: int,
           eval_cache_entries: int = 0,
           walkers: int = 1, coalesce: bool = True,
           multi_gpu_pucv: bool = False, vl_gather: int = 512,
           pucv_pending_mode: str = "legacy",
           compile_model: bool = True,
           compile_mode: str = "max-autotune",
           compile_cache_dir: str | None = None,
           gil_profile: bool = False,
           log_level: str = "WARNING") -> subprocess.Popen[str]:
    cmd = [sys.executable, "-u", "-m", "chess_anti_engine.uci",
           "--checkpoint", checkpoint,
           "--chunk-sims", str(chunk_sims),
           "--topk", str(topk),
           "--max-batch", str(max_batch),
           "--eval-cache-entries", str(max(0, int(eval_cache_entries))),
           "--walkers", str(walkers),
           "--vl-gather", str(vl_gather),
           "--pucv-pending-mode", pucv_pending_mode,
           "--log-level", log_level]
    if devices:
        cmd.extend(["--devices", devices])
    else:
        cmd.extend(["--device", device])
    if not coalesce:
        cmd.append("--no-coalesce")
    if multi_gpu_pucv:
        cmd.append("--multi-gpu-pucv")
    if compile_model:
        cmd.extend(["--compile-mode", compile_mode])
    else:
        cmd.append("--no-compile")
    if compile_cache_dir:
        cmd.extend(["--compile-cache-dir", compile_cache_dir])
    if gil_profile:
        cmd.append("--gil-profile")
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
    last_info = next(
        (line for line in reversed(lines) if line.startswith("info ") and " nodes " in line),
        "",
    )
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
    pucv_profiles = [_PUCV_PROFILE_RE.search(l) for l in lines]
    pucv_profiles = [p for p in pucv_profiles if p is not None]
    if pucv_profiles:
        prof_agg["pucv_searches"] = float(len(pucv_profiles))
        prof_agg["pucv_target"] = sum(float(p["target"]) for p in pucv_profiles)
        prof_agg["pucv_leaves"] = sum(float(p["leaves"]) for p in pucv_profiles)
        prof_agg["pucv_batches"] = sum(float(p["batches"]) for p in pucv_profiles)
        prof_agg["pucv_wall_s"] = sum(float(p["wall"]) for p in pucv_profiles)
        prof_agg["pucv_avg_batch"] = (
            prof_agg["pucv_leaves"] / max(1.0, prof_agg["pucv_batches"])
        )
    pucv_infos = [_PUCV_INFO_RE.search(l) for l in lines]
    pucv_infos = [p for p in pucv_infos if p is not None]
    if pucv_infos:
        prof_agg["pucv_info_lines"] = float(len(pucv_infos))
        prof_agg["pucv_cache_hits"] = sum(float(p["cache_hits"] or 0) for p in pucv_infos)
        prof_agg["pucv_cache_requests"] = sum(float(p["cache_requests"] or 0) for p in pucv_infos)
        info_leaves = sum(float(p["leaves"] or 0) for p in pucv_infos)
        info_batches = sum(float(p["batches"] or 0) for p in pucv_infos)
        if info_leaves > 0:
            prof_agg["pucv_leaves"] = prof_agg.get("pucv_leaves", 0.0) + info_leaves
        if info_batches > 0:
            prof_agg["pucv_batches"] = prof_agg.get("pucv_batches", 0.0) + info_batches
    gil_profiles = [_GIL_PROFILE_RE.search(line) for line in lines]
    gil_profiles = [profile for profile in gil_profiles if profile is not None]
    if gil_profiles:
        samples = sum(float(profile["samples"]) for profile in gil_profiles)
        prof_agg["gil_samples"] = samples
        prof_agg["gil_mean_weighted_ms"] = sum(
            float(profile["mean"]) * float(profile["samples"])
            for profile in gil_profiles
        )
        prof_agg["gil_over1_weighted"] = sum(
            float(profile["over1"]) * float(profile["samples"])
            for profile in gil_profiles
        )
        prof_agg["gil_over5_weighted"] = sum(
            float(profile["over5"]) * float(profile["samples"])
            for profile in gil_profiles
        )
        prof_agg["gil_p95_max_ms"] = max(float(profile["p95"]) for profile in gil_profiles)
        prof_agg["gil_p99_max_ms"] = max(float(profile["p99"]) for profile in gil_profiles)
        prof_agg["gil_max_ms"] = max(float(profile["max"]) for profile in gil_profiles)

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
    devices: str | None = None,
    nodes: int, repeats: int, timeout_s: float,
    chunk_sims: int, topk: int, max_batch: int,
    eval_cache_entries: int = 0,
    label: str,
    walkers: int = 1, coalesce: bool = True,
    multi_gpu_pucv: bool = False, vl_gather: int = 512,
    pucv_pending_mode: str = "legacy",
    use_vl: bool = False,
    compile_model: bool = True,
    compile_mode: str = "max-autotune",
    compile_cache_dir: str | None = None,
    gil_profile: bool = False,
    log_level: str = "WARNING",
) -> None:
    proc = _spawn(checkpoint, device, devices=devices,
                  chunk_sims=chunk_sims, topk=topk, max_batch=max_batch,
                  eval_cache_entries=eval_cache_entries,
                  walkers=walkers, coalesce=coalesce,
                  multi_gpu_pucv=multi_gpu_pucv, vl_gather=vl_gather,
                  pucv_pending_mode=pucv_pending_mode,
                  compile_model=compile_model, compile_mode=compile_mode,
                  compile_cache_dir=compile_cache_dir,
                  gil_profile=gil_profile,
                  log_level=log_level)
    reader = _LineReader(proc)
    _send(proc, "uci")
    startup_timeout = max(60.0, float(timeout_s))
    reader.read_until("uciok", timeout_s=startup_timeout)
    if use_vl:
        _send(proc, "setoption name UseVL value true")
        _send(proc, f"setoption name VLGather value {vl_gather}")
        _send(proc, f"setoption name PUCVPendingMode value {pucv_pending_mode}")
    _send(proc, "isready")
    reader.read_until("readyok", timeout_s=startup_timeout)

    coal_str = "" if walkers == 1 else f"  walkers={walkers}  coalesce={coalesce}"
    device_str = f"devices={devices}" if devices else f"device={device}"
    pucv_str = ""
    if devices:
        pucv_str = (
            f"  multi_gpu_pucv={multi_gpu_pucv}  vl_gather={vl_gather} "
            f"pending={pucv_pending_mode}"
        )
    elif use_vl:
        pucv_str = f"  use_vl=True  vl_gather={vl_gather} pending={pucv_pending_mode}"
    print(f"\n## {label}  {device_str}  chunk_sims={chunk_sims}  topk={topk}  max_batch={max_batch}{coal_str}{pucv_str}")
    if eval_cache_entries > 0:
        print(f"  eval_cache_entries={eval_cache_entries}")
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
        gumbel_runs = [profile for profile in profile_runs if "gpu_calls" in profile]
        if gumbel_runs:
            agg_calls = sum(p.get("gpu_calls", 0.0) for p in gumbel_runs)
            agg_pos = sum(p.get("gpu_pos", 0.0) for p in gumbel_runs)
            agg_total = sum(p.get("total_s", 0.0) for p in gumbel_runs)
            agg_gpu = sum(p.get("gpu_s", 0.0) for p in gumbel_runs)
            agg_prep = sum(p.get("prep_s", 0.0) for p in gumbel_runs)
            agg_finish = sum(p.get("finish_s", 0.0) for p in gumbel_runs)
            avg_batch = agg_pos / max(1.0, agg_calls)
            gpu_pct = 100.0 * agg_gpu / max(1e-9, agg_total)
            prep_pct = 100.0 * agg_prep / max(1e-9, agg_total)
            finish_pct = 100.0 * agg_finish / max(1e-9, agg_total)
            print(
                f"  profile: gpu_calls={int(agg_calls)}  gpu_pos={int(agg_pos)}  "
                f"avg_batch={avg_batch:.1f}  gpu={gpu_pct:.1f}%  "
                f"tree_prep={prep_pct:.1f}%  finish={finish_pct:.1f}%"
            )
        pucv_runs = [p for p in profile_runs if "pucv_batches" in p]
        if pucv_runs:
            leaves = sum(p.get("pucv_leaves", 0.0) for p in pucv_runs)
            batches = sum(p.get("pucv_batches", 0.0) for p in pucv_runs)
            wall = sum(p.get("pucv_wall_s", 0.0) for p in pucv_runs)
            cache_hits = sum(p.get("pucv_cache_hits", 0.0) for p in profile_runs)
            cache_requests = sum(p.get("pucv_cache_requests", 0.0) for p in profile_runs)
            cache = ""
            if cache_requests > 0:
                cache = (
                    f"  cache={int(cache_hits)}/{int(cache_requests)}"
                    f"({100.0 * cache_hits / cache_requests:.1f}%)"
                )
            print(
                f"  pucv profile: leaves={int(leaves)}  batches={int(batches)}  "
                f"avg_batch={leaves / max(1.0, batches):.1f}  wall={wall:.3f}s{cache}"
            )
        gil_runs = [profile for profile in profile_runs if "gil_samples" in profile]
        if gil_runs:
            samples = sum(profile.get("gil_samples", 0.0) for profile in gil_runs)
            mean_ms = sum(
                profile.get("gil_mean_weighted_ms", 0.0) for profile in gil_runs
            ) / max(1.0, samples)
            over1 = sum(
                profile.get("gil_over1_weighted", 0.0) for profile in gil_runs
            ) / max(1.0, samples)
            over5 = sum(
                profile.get("gil_over5_weighted", 0.0) for profile in gil_runs
            ) / max(1.0, samples)
            p95_max = max(profile.get("gil_p95_max_ms", 0.0) for profile in gil_runs)
            p99_max = max(profile.get("gil_p99_max_ms", 0.0) for profile in gil_runs)
            max_ms = max(profile.get("gil_max_ms", 0.0) for profile in gil_runs)
            print(
                f"  gil-delay upper bound: samples={int(samples)} mean={mean_ms:.3f}ms "
                f"worst-search-p95<={p95_max:.2f}ms p99<={p99_max:.2f}ms "
                f"max={max_ms:.2f}ms over1ms={over1:.1f}% over5ms={over5:.1f}%"
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
    p.add_argument("--devices", default=None,
                   help="comma-separated device list for multi-GPU UCI, e.g. cuda:0,cuda:1")
    p.add_argument("--nodes", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--sweep", action="store_true",
                   help="sweep predefined (chunk_sims, topk, max_batch) configs")
    p.add_argument("--walker-sweep", action="store_true",
                   help="sweep --walkers {1,2,4,8} at the production chunk=512/topk=32/mb=1024 config. "
                        "Single-walker uses classic Gumbel; >1 switches to PUCT walker pool.")
    p.add_argument("--pucv-sweep", action="store_true",
                   help="sweep multi-GPU PUCV gather and pending modes; requires --devices")
    p.add_argument("--single-pucv-sweep", action="store_true",
                   help="sweep single-GPU UseVL gather and pending modes")
    p.add_argument("--use-vl", action="store_true",
                   help="enable UCI UseVL for a single non-sweep config")
    p.add_argument("--chunk-sims", type=int, default=32)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--eval-cache-entries", type=int, default=0,
                   help="EvalCacheEntries passed to UCI (0 disables)")
    p.add_argument("--cache-sweep", action="store_true",
                   help="for PUCV sweeps, run cache off and on")
    p.add_argument("--walkers", type=int, default=1,
                   help="PUCT walker threads for a single --walkers config run (default 1 = Gumbel).")
    p.add_argument("--vl-gather", type=int, default=512,
                   help="VLGather / multi-GPU PUCV gather size")
    p.add_argument("--multi-gpu-pucv", action="store_true",
                   help="launch UCI with --multi-gpu-pucv when --devices is set")
    p.add_argument("--pucv-pending-mode", choices=["legacy", "virtual-mean"],
                   default="legacy",
                   help="pending accounting for batched PUCV paths")
    p.add_argument("--no-coalesce", dest="coalesce", action="store_false",
                   help="disable walker-call coalescing (only meaningful with --walkers > 1)")
    p.set_defaults(coalesce=True)
    p.add_argument("--log-level", default="WARNING",
                   help="DEBUG to see per-search gumbel profile (GPU calls, avg batch, time breakdown).")
    p.add_argument(
        "--gil-profile", action="store_true",
        help="collect and summarize per-search delayed-GIL-reacquisition telemetry",
    )
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="pass --no-compile to the UCI engine")
    p.add_argument("--compile-mode", default="max-autotune",
                   help="UCI torch.compile mode when compile is enabled")
    p.add_argument("--compile-cache-dir", default=None,
                   help="UCI TorchInductor/Triton cache root")
    p.set_defaults(compile=True)
    args = p.parse_args()

    def cache_values() -> tuple[int, ...]:
        if not args.cache_sweep:
            return (max(0, int(args.eval_cache_entries)),)
        enabled = max(1, int(args.eval_cache_entries)) if args.eval_cache_entries > 0 else 131072
        return (0, enabled)

    dev_label = f"devices={args.devices}" if args.devices else f"device={args.device}"
    print(f"# checkpoint={args.checkpoint}  {dev_label}  nodes={args.nodes}  repeats={args.repeats}")

    if args.single_pucv_sweep:
        for pending_mode in ("legacy", "virtual-mean"):
            for gather in (128, 256, 384, 512, 768, 1024):
                for cache_entries in cache_values():
                    _run_config(
                        args.checkpoint, args.device, devices=args.devices,
                        nodes=args.nodes, repeats=args.repeats, timeout_s=args.timeout_s,
                        chunk_sims=512, topk=32, max_batch=max(args.max_batch, gather),
                        eval_cache_entries=cache_entries,
                        walkers=1, coalesce=args.coalesce,
                        multi_gpu_pucv=False, vl_gather=gather,
                        pucv_pending_mode=pending_mode,
                        label=(
                            f"single-gpu-pucv gather={gather} pending={pending_mode} "
                            f"cache={cache_entries}"
                        ),
                        log_level=args.log_level,
                        use_vl=True,
                        compile_model=args.compile,
                        compile_mode=args.compile_mode,
                        compile_cache_dir=args.compile_cache_dir,
                        gil_profile=args.gil_profile,
                    )
    elif args.pucv_sweep:
        if not args.devices:
            p.error("--pucv-sweep requires --devices")
        for pending_mode in ("legacy", "virtual-mean"):
            for gather in (128, 256, 384, 512, 768, 1024):
                for cache_entries in cache_values():
                    _run_config(
                        args.checkpoint, args.device, devices=args.devices,
                        nodes=args.nodes, repeats=args.repeats, timeout_s=args.timeout_s,
                        chunk_sims=512, topk=32, max_batch=max(args.max_batch, gather),
                        eval_cache_entries=cache_entries,
                        walkers=args.walkers, coalesce=args.coalesce,
                        multi_gpu_pucv=True, vl_gather=gather,
                        pucv_pending_mode=pending_mode,
                        label=(
                            f"multi-gpu-pucv gather={gather} pending={pending_mode} "
                            f"cache={cache_entries}"
                        ),
                        log_level=args.log_level,
                        compile_model=args.compile,
                        compile_mode=args.compile_mode,
                        compile_cache_dir=args.compile_cache_dir,
                        gil_profile=args.gil_profile,
                    )
    elif args.sweep:
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
                args.checkpoint, args.device, devices=args.devices,
                nodes=args.nodes, repeats=args.repeats, timeout_s=args.timeout_s,
                chunk_sims=cs, topk=tk, max_batch=mb, label=label,
                eval_cache_entries=args.eval_cache_entries,
                multi_gpu_pucv=args.multi_gpu_pucv, vl_gather=args.vl_gather,
                pucv_pending_mode=args.pucv_pending_mode,
                compile_model=args.compile,
                compile_mode=args.compile_mode,
                compile_cache_dir=args.compile_cache_dir,
                gil_profile=args.gil_profile,
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
                args.checkpoint, args.device, devices=args.devices,
                nodes=args.nodes, repeats=args.repeats, timeout_s=args.timeout_s,
                chunk_sims=512, topk=32, max_batch=1024,
                eval_cache_entries=args.eval_cache_entries,
                walkers=w, coalesce=coal, label=label,
                multi_gpu_pucv=args.multi_gpu_pucv, vl_gather=args.vl_gather,
                pucv_pending_mode=args.pucv_pending_mode,
                compile_model=args.compile,
                compile_mode=args.compile_mode,
                compile_cache_dir=args.compile_cache_dir,
                gil_profile=args.gil_profile,
                log_level=args.log_level,
            )
    else:
        _run_config(
            args.checkpoint, args.device, devices=args.devices,
            nodes=args.nodes, repeats=args.repeats, timeout_s=args.timeout_s,
            chunk_sims=args.chunk_sims, topk=args.topk, max_batch=args.max_batch,
            eval_cache_entries=args.eval_cache_entries,
            walkers=args.walkers, coalesce=args.coalesce,
            multi_gpu_pucv=args.multi_gpu_pucv, vl_gather=args.vl_gather,
            pucv_pending_mode=args.pucv_pending_mode,
            use_vl=args.use_vl,
            compile_model=args.compile,
            compile_mode=args.compile_mode,
            compile_cache_dir=args.compile_cache_dir,
            gil_profile=args.gil_profile,
            label="single",
            log_level=args.log_level,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
