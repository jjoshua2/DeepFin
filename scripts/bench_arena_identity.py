#!/usr/bin/env python3
"""Throughput + identity bench for the matched_sims rolling arena (net vs itself).

Calls ``scripts.arena_standard.play_paired_games_matched_sims_rolling`` the way
``run_arena`` does: one hoisted ``DirectGPUEvaluator`` (built by
``build_arena_evaluator``), strict 3-6-man Syzygy, the training search shape,
no noise, temperature 0. The result has a DIGEST of every game's moves,
result and termination, so two revisions can be compared on the same command:

    PYTHONPATH=<tree> python <this script> --model NET.pt --syzygy "WDL:DTZ" \\
        --starts starts.txt --pairs 64 --pool 128 --out run.json

``--starts`` lines are ``<root FEN> | <uci moves...>``. Also recorded: GPU
utilisation, peak torch memory and the evaluator's CUDA-graph counters. The
evaluator replays CUDA graphs of its eager forward on CUDA, as the arena does
(its one fast path); off CUDA it runs eager.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--syzygy", required=True, help="colon-separated WDL/DTZ directories")
    p.add_argument("--starts", required=True)
    p.add_argument("--pairs", type=int, default=64)
    p.add_argument("--pool", type=int, default=128)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--max-batch", type=int, default=4096)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--inference-only", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.manual_seed(args.seed)

    from chess_anti_engine.tablebase import SyzygyProbe, open_strict_match_tablebase
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
    from scripts.arena_standard import (
        build_arena_evaluator,
        play_paired_games_matched_sims_rolling,
        resolve_search_shape,
    )

    model = load_model_from_checkpoint(args.model, device=args.device).eval()
    if args.inference_only:
        setattr(model, "_inference_only", True)
    openings: list[chess.Board] = []
    for line in Path(args.starts).read_text().splitlines()[: args.pairs]:
        root, moves = line.split(" | ")
        board = chess.Board(root)
        for uci in moves.split():
            board.push_uci(uci)
        openings.append(board)

    side = resolve_search_shape("training")
    ev = build_arena_evaluator(model, device=args.device, max_batch=args.max_batch,
                               cuda_graphs=str(args.device).startswith("cuda"))
    forwards: list[int] = []
    for name in ("evaluate_inplace_async", "evaluate_encoded_async", "evaluate_encoded"):
        orig = getattr(ev, name, None)
        if orig is None:
            continue

        def wrap(*a: Any, _o: Any = orig, _n: str = name, **kw: Any) -> Any:
            forwards.append(int(a[0]) if _n == "evaluate_inplace_async" else int(a[0].shape[0]))
            return _o(*a, **kw)
        setattr(ev, name, wrap)

    tb = open_strict_match_tablebase(args.syzygy, max_pieces=6)
    probe = SyzygyProbe(args.syzygy, max_pieces=6, rule50_aware=True, tablebase=tb)
    games: dict[tuple[int, int], str] = {}
    plies = [0]

    def sink(**kw: Any) -> None:
        games[(kw["pair_id"], kw["half"])] = json.dumps(
            {"m": [m.uci() for m in kw["moves"]], "r": kw["result"], "t": kw["termination"]})
        plies[0] += int(kw["plies"])

    gpu_log = Path(args.out).with_suffix(".nvsmi")
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-lms", "200"], stdout=gpu_log.open("w"))
    t0 = time.perf_counter()
    try:
        scores = play_paired_games_matched_sims_rolling(
            model, model, openings, device=args.device, rng=np.random.default_rng(args.seed),
            sims_candidate=args.sims, sims_reference=args.sims, max_plies=args.max_plies,
            temperature=0.0, gumbel_add_noise=False, search_candidate=side, search_reference=side,
            syzygy_tablebase=tb, tb_probe=probe, tb_max_pieces=6, pool_size=args.pool,
            report_every=10**9, pgn_sink=sink, evaluator_candidate=ev, evaluator_reference=ev,
            free_cached_vram=True,
        )
    finally:
        wall = time.perf_counter() - t0
        smi.terminate()
        smi.wait()
    util = [int(parts[1]) for parts in (
        [x.strip() for x in line.split(",")] for line in gpu_log.read_text().splitlines())
        if len(parts) >= 2 and parts[1].isdigit()]
    rows = np.array(forwards) if forwards else np.zeros(1)
    d = hashlib.sha256()
    for k in sorted(games):
        d.update(repr(k).encode())
        d.update(games[k].encode())
    res = {
        "args": vars(args), "wall_s": wall, "games": len(games), "plies": plies[0],
        "games_per_hour": len(games) / wall * 3600, "plies_per_s": plies[0] / wall,
        "forwards": len(forwards), "rows_padded": int(rows.sum()), "rows_per_s": float(rows.sum()) / wall,
        "batch_mean": float(rows.mean()),
        "batch_p10_50_90": [float(x) for x in np.percentile(rows, [10, 50, 90])],
        "gpu_util_mean": float(np.mean(util)) if util else None,
        "score": float(np.mean(scores)) if scores else None, "digest": d.hexdigest(),
        "cuda_graph_shapes": list(getattr(ev, "cuda_graph_shapes", ())),
        "cuda_graph_stats": getattr(ev, "cuda_graph_stats", None),
        "torch_max_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "torch_max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
    }
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
