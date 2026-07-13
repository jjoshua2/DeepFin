#!/usr/bin/env python3
"""Benchmark legacy versus final-line Stockfish MultiPV parsing."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections.abc import Callable

import numpy as np

from chess_anti_engine.stockfish.uci import (
    StockfishPV,
    StockfishResult,
    _SearchInfoAccumulator,
)


def _int_after(parts: list[str], token: str) -> int | None:
    try:
        return int(parts[parts.index(token) + 1])
    except (ValueError, IndexError):
        return None


def _parse_score(parts: list[str]) -> tuple[int | None, int | None]:
    try:
        idx = parts.index("score")
        kind = parts[idx + 1]
        value = int(parts[idx + 2])
    except (ValueError, IndexError):
        return None, None
    if kind == "cp":
        return value, None
    if kind == "mate":
        return None, value
    return None, None


def _parse_wdl(parts: list[str]) -> np.ndarray | None:
    try:
        idx = parts.index("wdl")
        vec = np.asarray(
            [int(parts[idx + 1]), int(parts[idx + 2]), int(parts[idx + 3])],
            dtype=np.float32,
        )
    except (ValueError, IndexError):
        return None
    total = float(vec.sum())
    return vec / total if total > 0.0 else None


def _parse_pv_move(parts: list[str]) -> str | None:
    try:
        idx = parts.index("pv")
    except ValueError:
        return None
    return parts[idx + 1] if idx + 1 < len(parts) else None


def _legacy_parse(lines: list[str]) -> StockfishResult:
    wdl_pv1 = None
    cp_pv1 = mate_pv1 = nodes_seen = depth_seen = None
    pvs: dict[int, StockfishPV] = {}
    for line in lines:
        parts = line.split()
        mpv = _int_after(parts, "multipv") or 1
        nodes = _int_after(parts, "nodes")
        if nodes is not None:
            nodes_seen = nodes
        depth = _int_after(parts, "depth")
        if depth is not None:
            depth_seen = max(depth_seen or 0, depth)
        cp, mate = _parse_score(parts)
        wdl = _parse_wdl(parts)
        pv_move = _parse_pv_move(parts)
        if mpv == 1:
            if wdl is not None:
                wdl_pv1 = wdl
            if cp is not None:
                cp_pv1, mate_pv1 = cp, None
            if mate is not None:
                mate_pv1, cp_pv1 = mate, None
        if pv_move is not None:
            pvs[mpv] = StockfishPV(pv_move, wdl, cp=cp, mate=mate)
    return StockfishResult(
        bestmove_uci="e2e4",
        wdl=wdl_pv1,
        pvs=[pvs[k] for k in sorted(pvs)],
        cp=cp_pv1,
        mate=mate_pv1,
        nodes=nodes_seen,
        depth=depth_seen,
    )


def _candidate_parse(lines: list[str]) -> StockfishResult:
    info = _SearchInfoAccumulator()
    for line in lines:
        info.consume(line.split())
    return info.result("e2e4")


def _make_lines(multipv: int, depths: int) -> list[str]:
    moves = (
        "e2e4", "d2d4", "g1f3", "c2c4", "b1c3", "g2g3", "e2e3", "b2b3",
    )
    lines: list[str] = []
    for depth in range(1, depths + 1):
        for rank in range(1, multipv + 1):
            cp = 180 - rank * 7 + depth
            win = max(0, min(1000, 500 + cp))
            loss = max(0, min(1000 - win, 300 - cp // 2))
            draw = 1000 - win - loss
            lines.append(
                f"info depth {depth} seldepth {depth + 3} multipv {rank} "
                f"score cp {cp} wdl {win} {draw} {loss} nodes {depth * 50000 + rank} "
                f"nps 1234567 hashfull {min(999, depth * 20)} tbhits 0 time {depth * 17} "
                f"pv {moves[(rank - 1) % len(moves)]} e7e5 g1f3"
            )
    return lines


def _hash_result(result: StockfishResult) -> str:
    payload = {
        "bestmove": result.bestmove_uci,
        "wdl": None if result.wdl is None else result.wdl.tolist(),
        "cp": result.cp,
        "mate": result.mate,
        "nodes": result.nodes,
        "depth": result.depth,
        "pvs": [
            {
                "move": pv.move_uci,
                "wdl": None if pv.wdl is None else pv.wdl.tolist(),
                "cp": pv.cp,
                "mate": pv.mate,
            }
            for pv in result.pvs
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _time_parser(fn: Callable[[list[str]], StockfishResult], lines: list[str], iterations: int) -> float:
    gc.collect()
    gc.disable()
    start = time.perf_counter()
    try:
        for _ in range(iterations):
            fn(lines)
    finally:
        elapsed = time.perf_counter() - start
        gc.enable()
    return float(iterations) / elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multipv", type=int, default=40)
    ap.add_argument("--depths", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--iterations", type=int, default=250)
    args = ap.parse_args()
    if min(args.multipv, args.depths, args.rounds, args.iterations) <= 0:
        raise SystemExit("all numeric arguments must be positive")

    lines = _make_lines(args.multipv, args.depths)
    reference_hash = _hash_result(_legacy_parse(lines))
    candidate_hash = _hash_result(_candidate_parse(lines))
    if candidate_hash != reference_hash:
        raise SystemExit(
            f"parser result mismatch: reference={reference_hash} candidate={candidate_hash}"
        )

    reference_rates: list[float] = []
    candidate_rates: list[float] = []
    for round_idx in range(args.rounds):
        order = (
            (("reference", _legacy_parse), ("candidate", _candidate_parse))
            if round_idx % 2 == 0
            else (("candidate", _candidate_parse), ("reference", _legacy_parse))
        )
        for name, fn in order:
            rate = _time_parser(fn, lines, args.iterations)
            (reference_rates if name == "reference" else candidate_rates).append(rate)

    ref_median = float(np.median(np.asarray(reference_rates)))
    cand_median = float(np.median(np.asarray(candidate_rates)))
    result = {
        "multipv": args.multipv,
        "depths": args.depths,
        "lines_per_stream": len(lines),
        "rounds": args.rounds,
        "iterations": args.iterations,
        "reference_streams_per_s": ref_median,
        "candidate_streams_per_s": cand_median,
        "speedup": cand_median / ref_median,
        "result_hash": reference_hash,
        "hash_match": True,
        "reference_rates": reference_rates,
        "candidate_rates": candidate_rates,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
