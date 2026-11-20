#!/usr/bin/env python3
"""Throughput + identity bench for in-process ``play_batch`` (collector shape).

Runs ``play_batch`` the way ``scripts/core_finite_collect.py`` does: one
``DirectGPUEvaluator``, the collector's six config groups, a FEN dole queue,
and a ``StockfishPool`` under the strict-TB contract. It records per-phase
timings, every evaluator forward (rows, wall), GPU utilisation, peak torch
memory, the CUDA-graph counters, and a DIGEST of every game's provenance and
every stored ``ReplaySample`` array.

Two code revisions are identity-equivalent on a workload when the same
command gives the same ``digest`` on both. Run it from each revision's tree:

    PYTHONPATH=<tree> python <this script> --model NET.pt --sf-path SF \\
        --syzygy "WDL_DIR:DTZ_DIR" --configs groups.json --starts starts.epd \\
        --games 192 --slots 128 --sims 64 [--cuda-graphs] --out run.json

``--configs`` is a JSON object with the groups ``game``, ``search``, ``temp``,
``opening``, ``opponent`` and ``diff_focus`` (the collector protocol's config
values). ``--starts`` has one FEN per line. The serving flags default to
dense fp32 transport (``legal_bf16`` off), the only transport the graphs
serve; ``--legal-bf16`` selects compact transport (and cannot be combined
with ``--cuda-graphs``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _gpu_log(path: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-lms", "200"],
        stdout=path.open("w"),
    )


def _util_samples(path: Path) -> list[int]:
    out: list[int] = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[1].isdigit():
                out.append(int(parts[1]))
    return out


def _sample_digest(sample: Any) -> str:
    h = hashlib.sha256()
    fields = vars(sample) if hasattr(sample, "__dict__") else asdict(sample)
    for k, v in sorted(fields.items()):
        if isinstance(v, np.ndarray):
            h.update(k.encode())
            h.update(str(v.dtype).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, (int, float, str, bool, type(None))):
            h.update(k.encode())
            h.update(repr(v).encode())
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--sf-path", required=True)
    p.add_argument("--syzygy", required=True, help="colon-separated WDL/DTZ directories")
    p.add_argument("--configs", required=True)
    p.add_argument("--starts", required=True)
    p.add_argument("--games", type=int, default=192)
    p.add_argument("--slots", type=int, default=128)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=512)
    p.add_argument("--legal-bf16", action="store_true")
    p.add_argument("--cuda-graphs", action="store_true")
    p.add_argument("--cuda-graph-max-rows", type=int, default=512)
    p.add_argument("--inference-only", action="store_true")
    p.add_argument("--max-plies", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.manual_seed(args.seed)

    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.selfplay.config import (
        DiffFocusConfig,
        GameConfig,
        OpponentConfig,
        SearchConfig,
        TemperatureConfig,
    )
    from chess_anti_engine.selfplay.opening import OpeningConfig
    from chess_anti_engine.stockfish.pool import StockfishPool
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    values = json.loads(Path(args.configs).read_text())
    values["opening"]["opening_fen_list_path"] = args.starts
    values["game"]["syzygy_path"] = args.syzygy
    values["game"]["stockfish_syzygy_path"] = args.syzygy
    if args.max_plies:
        values["game"]["max_plies"] = args.max_plies
    values["search"]["simulations"] = args.sims
    values["search"]["fast_simulations"] = args.sims
    cfg: dict[str, Any] = {
        "game": GameConfig(**values["game"]), "search": SearchConfig(**values["search"]),
        "temp": TemperatureConfig(**values["temp"]), "opening": OpeningConfig(**values["opening"]),
        "opponent": OpponentConfig(**values["opponent"]),
        "diff_focus": DiffFocusConfig(**values["diff_focus"]),
    }

    model = load_model_from_checkpoint(args.model, device=args.device).eval()
    if args.inference_only:
        setattr(model, "_inference_only", True)  # policy_own + wdl only (bitwise equal heads)
    forwards: list[tuple[int, float]] = []

    class TimedEvaluator(DirectGPUEvaluator):
        def _run_forward(self, bsz: int, **kw: Any) -> Any:
            t0 = time.perf_counter()
            out = super()._run_forward(bsz, **kw)
            forwards.append((int(bsz), time.perf_counter() - t0))
            return out

        def evaluate_inplace_async(self, bsz: int, **kw: Any) -> Any:
            # play_batch never pipelines (persistent tree): every caller syncs
            # at once, so syncing here only moves the wait into the timer.
            t0 = time.perf_counter()
            pol, wdl, ev = super().evaluate_inplace_async(bsz, **kw)
            if ev is not None:
                ev.synchronize()
            forwards.append((int(bsz), time.perf_counter() - t0))
            return pol, wdl, ev

        def evaluate_legal_bf16(self, x: Any, legal_flat: Any, legal_counts: Any, *a: Any, **kw: Any) -> Any:
            t0 = time.perf_counter()
            out = super().evaluate_legal_bf16(x, legal_flat, legal_counts, *a, **kw)
            forwards.append((int(x.shape[0]), time.perf_counter() - t0))
            return out

    graph_kw: dict[str, Any] = (
        {"cuda_graphs": True, "cuda_graph_max_rows": args.cuda_graph_max_rows} if args.cuda_graphs else {}
    )
    evaluator = TimedEvaluator(
        model, device=args.device, max_batch=args.max_batch, use_amp=True, amp_dtype="bfloat16",
        n_slots=1, input_bf16=False, legal_bf16=args.legal_bf16, pad_to_compiled_buckets=False,
        **graph_kw,
    )

    timings: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    def on_timing(key: str, dt: float) -> None:
        timings[key] += float(dt)
        counts[key] += 1

    game_digests: dict[int, str] = {}
    game_moves: dict[int, str] = {}
    totals = {"rows": 0, "plies": 0}

    def on_samples(batch: Any) -> None:
        for s in batch.samples:
            gid = int(getattr(s, "game_id", -1))
            game_digests[gid] = hashlib.sha256(
                (game_digests.get(gid, "") + _sample_digest(s)).encode()).hexdigest()
            totals["rows"] += 1

    def on_game(audit: Any) -> None:
        a = asdict(audit)
        moves = a.get("moves_uci") or a.get("played_uci") or a.get("moves") or []
        game_moves[int(audit.game_id)] = json.dumps(
            {k: a[k] for k in sorted(a) if k != "start"}, sort_keys=True, default=str)
        totals["plies"] += int(a.get("total_plies") or len(moves) or 0)

    queue = Path(args.starts).read_text().splitlines()[: args.games]
    sf = StockfishPool(path=args.sf_path, nodes=64, multipv=1, hash_mb=16, num_workers=1,
                       syzygy_path=args.syzygy, syzygy_50_move_rule=True, syzygy_probe_limit=6,
                       read_timeout_s=20)
    gpu_log = Path(args.out).with_suffix(".nvsmi")
    smi = _gpu_log(gpu_log)
    t0 = time.perf_counter()
    try:
        _, stats = play_batch(
            model, device=args.device, rng=np.random.default_rng(args.seed), stockfish=sf,
            evaluator=evaluator, games=args.slots, target_games=len(queue),
            on_game_complete=on_samples, on_game_provenance=on_game, fen_dole_queue=queue,
            on_timing=on_timing, **cfg)
    finally:
        wall = time.perf_counter() - t0
        smi.terminate()
        smi.wait()
        sf.close()
    util = _util_samples(gpu_log)
    rows = np.array([r for r, _ in forwards]) if forwards else np.zeros(1)
    fwd_s = sum(t for _, t in forwards)
    digest = hashlib.sha256()
    for gid in sorted(game_moves):
        digest.update(game_moves[gid].encode())
        digest.update(game_digests.get(gid, "").encode())
    res = {
        "args": vars(args), "wall_s": wall, "games": stats.games, "rows": totals["rows"],
        "positions": stats.positions, "plies_total": totals["plies"],
        "games_per_hour": stats.games / wall * 3600, "plies_per_s": totals["plies"] / wall,
        "forwards": len(forwards), "leaf_rows": int(rows.sum()), "leaf_rows_per_s": float(rows.sum()) / wall,
        "fwd_wall_s": fwd_s, "fwd_share": fwd_s / wall,
        "batch_mean": float(rows.mean()),
        "batch_p10_50_90": [float(x) for x in np.percentile(rows, [10, 50, 90])],
        "gpu_util_mean": float(np.mean(util)) if util else None, "gpu_util_samples": len(util),
        "cuda_graph_shapes": list(getattr(evaluator, "cuda_graph_shapes", ())),
        "cuda_graph_stats": getattr(evaluator, "cuda_graph_stats", None),
        "torch_max_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "torch_max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "timings": dict(timings), "timing_counts": dict(counts),
        "digest": digest.hexdigest(),
        "per_game": {str(k): hashlib.sha256((game_moves[k] + game_digests.get(k, "")).encode()).hexdigest()
                     for k in sorted(game_moves)},
    }
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_game", "args")}, indent=1))


if __name__ == "__main__":
    main()
