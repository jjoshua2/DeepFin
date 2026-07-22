#!/usr/bin/env python3
"""Paired match: 1-GPU classic Gumbel vs multi-GPU search (PUCV or RPG).

Purpose: after multi-GPU throughput work, check that higher nps still
*plays* (no crashes, no time forfeits) and estimate Elo at fixed clock.
This is NOT a substitute for the frozen deep-SF audit — it answers
"does the parallel path cash the NPS into games?"

Examples:
  PYTHONPATH=. python3 scripts/match_multi_gpu.py \\
      --checkpoint best_model.pt --devices cuda:0,cuda:1 \\
      --games 20 --movetime 1000

  # root-parallel Gumbel on 2 GPUs vs 1-GPU Gumbel
  PYTHONPATH=. python3 scripts/match_multi_gpu.py \\
      --checkpoint best_model.pt --devices cuda:0,cuda:1 \\
      --multi-mode gumbel --games 20 --movetime 2000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _engine_cmd(
    *,
    checkpoint: str,
    devices: str | None,
    device: str | None,
    multi_mode: str | None,
    compile_mode: str,
    no_compile: bool,
    vl_gather: int,
    max_batch: int,
    chunk_sims: int,
) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "chess_anti_engine.uci",
        "--checkpoint", checkpoint,
        "--chunk-sims", str(chunk_sims),
        "--max-batch", str(max_batch),
        "--vl-gather", str(vl_gather),
        "--log-level", "WARNING",
    ]
    if no_compile:
        cmd.append("--no-compile")
    else:
        cmd.extend(["--compile-mode", compile_mode])
    if devices:
        cmd.extend(["--devices", devices])
        if multi_mode == "gumbel":
            cmd.extend(["--search-parallel", "gumbel", "--walkers", "1"])
        else:
            cmd.extend(["--multi-gpu-pucv", "--walkers", "1"])
    else:
        cmd.extend(["--device", device or "cuda:0", "--walkers", "1"])
    return cmd


def _play_game(
    white_cmd: list[str],
    black_cmd: list[str],
    *,
    movetime_ms: int,
    max_plies: int,
    ready_timeout_s: float,
) -> dict[str, object]:
    """Minimal UCI self-play (no cutechess). Returns result from white POV."""
    import chess

    def spawn(cmd: list[str]) -> subprocess.Popen[str]:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def send(p: subprocess.Popen[str], line: str) -> None:
        assert p.stdin is not None
        p.stdin.write(line + "\n")
        p.stdin.flush()

    def wait_token(p: subprocess.Popen[str], token: str, timeout_s: float) -> None:
        assert p.stdout is not None
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"EOF waiting for {token}")
            if token in line:
                return
            if "engine load failed" in line or "search error" in line:
                raise RuntimeError(line.strip())
        raise TimeoutError(f"timeout waiting for {token}")

    def go_bestmove(p: subprocess.Popen[str], fen: str, movetime: int) -> str:
        assert p.stdout is not None
        if fen == chess.STARTING_FEN:
            send(p, "position startpos")
        else:
            send(p, f"position fen {fen}")
        send(p, f"go movetime {movetime}")
        deadline = time.perf_counter() + max(30.0, movetime / 1000.0 + 60.0)
        while time.perf_counter() < deadline:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("EOF waiting for bestmove")
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) < 2:
                    raise RuntimeError(f"bad bestmove line: {line!r}")
                return parts[1]
            if "search error" in line:
                raise RuntimeError(line.strip())
        raise TimeoutError("timeout waiting for bestmove")

    w = spawn(white_cmd)
    b = spawn(black_cmd)
    try:
        for p in (w, b):
            send(p, "uci")
            wait_token(p, "uciok", ready_timeout_s)
            send(p, "isready")
            wait_token(p, "readyok", ready_timeout_s)

        board = chess.Board()
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < max_plies:
            engine = w if board.turn == chess.WHITE else b
            mv_uci = go_bestmove(engine, board.fen(), movetime_ms)
            if mv_uci in ("0000", "(none)"):
                break
            try:
                move = chess.Move.from_uci(mv_uci)
            except ValueError as exc:
                raise RuntimeError(f"illegal uci {mv_uci!r}") from exc
            if move not in board.legal_moves:
                raise RuntimeError(f"illegal move {mv_uci} in {board.fen()}")
            board.push(move)
            plies += 1

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            result = "1/2-1/2"
        elif outcome.winner == chess.WHITE:
            result = "1-0"
        else:
            result = "0-1"
        return {
            "result": result,
            "plies": plies,
            "fen": board.fen(),
            "termination": None if outcome is None else str(outcome.termination),
        }
    finally:
        for p in (w, b):
            try:
                send(p, "quit")
                p.wait(timeout=15)
            except Exception:
                p.kill()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--devices", default="cuda:0,cuda:1",
                   help="multi-GPU device list for engine B")
    p.add_argument("--device-a", default="cuda:0",
                   help="single-GPU device for engine A (classic Gumbel)")
    p.add_argument("--multi-mode", choices=["pucv", "gumbel"], default="pucv",
                   help="multi-GPU search mode for engine B")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--movetime", type=int, default=1000, help="ms per move")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--compile-mode", default="reduce-overhead")
    p.add_argument(
        "--no-compile", action="store_true",
        help="eager path (recommended for short match ladders; avoids "
             "per-game cold compile when engines respawn)",
    )
    p.add_argument("--vl-gather", type=int, default=256)
    p.add_argument("--max-batch", type=int, default=1024)
    p.add_argument("--chunk-sims", type=int, default=512)
    p.add_argument("--ready-timeout-s", type=float, default=600.0)
    p.add_argument("--out", default=None, help="optional JSONL results path")
    args = p.parse_args()

    cmd_a = _engine_cmd(
        checkpoint=args.checkpoint,
        devices=None,
        device=args.device_a,
        multi_mode=None,
        compile_mode=args.compile_mode,
        no_compile=bool(args.no_compile),
        vl_gather=args.vl_gather,
        max_batch=args.max_batch,
        chunk_sims=args.chunk_sims,
    )
    cmd_b = _engine_cmd(
        checkpoint=args.checkpoint,
        devices=args.devices,
        device=None,
        multi_mode=args.multi_mode,
        compile_mode=args.compile_mode,
        no_compile=bool(args.no_compile),
        vl_gather=args.vl_gather,
        max_batch=args.max_batch,
        chunk_sims=args.chunk_sims,
    )
    print(
        f"# A=1gpu-gumbel({args.device_a})  B=multi-{args.multi_mode}({args.devices})  "
        f"games={args.games} movetime={args.movetime}ms",
        flush=True,
    )

    out_path = Path(args.out) if args.out else None
    out_fh = out_path.open("w") if out_path else None
    score_b = 0.0  # from B's perspective across both colors
    crashes = 0
    try:
        for g in range(args.games):
            # Alternate colors so B is white on even games.
            if g % 2 == 0:
                white, black, b_is_white = cmd_b, cmd_a, True
            else:
                white, black, b_is_white = cmd_a, cmd_b, False
            t0 = time.perf_counter()
            try:
                res = _play_game(
                    white, black,
                    movetime_ms=args.movetime,
                    max_plies=args.max_plies,
                    ready_timeout_s=args.ready_timeout_s,
                )
            except Exception as exc:
                crashes += 1
                res = {
                    "result": "crash",
                    "error": repr(exc),
                    "plies": 0,
                }
                print(f"game {g}: CRASH {exc!r}", flush=True)
            else:
                r = str(res["result"])
                if r == "1-0":
                    score_b += 1.0 if b_is_white else 0.0
                elif r == "0-1":
                    score_b += 0.0 if b_is_white else 1.0
                else:
                    score_b += 0.5
                print(
                    f"game {g}: {r} plies={res['plies']} "
                    f"B_white={b_is_white} wall={time.perf_counter() - t0:.1f}s "
                    f"B_score={score_b:.1f}/{g + 1}",
                    flush=True,
                )
            row = {
                "game": g,
                "b_is_white": b_is_white,
                "multi_mode": args.multi_mode,
                **res,
            }
            if out_fh is not None:
                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
    finally:
        if out_fh is not None:
            out_fh.close()

    played = args.games - crashes
    print(
        f"\n# done  B_score={score_b:.1f}/{played}  crashes={crashes}  "
        f"mode={args.multi_mode}",
        flush=True,
    )
    if crashes:
        print("# FAIL: crashes observed — do not promote this multi-GPU path", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
