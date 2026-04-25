#!/usr/bin/env python3
"""UCI-vs-UCI match driver.

Pits two UCI engines against each other. Used to benchmark our model's UCI
engine against external references like rofChade/Stockfish/lc0 to get an
absolute Elo number.

Each side plays White and Black equally. Per-move time budget is fixed
(``--time-ms``). Standard chess rules apply (5-fold repetition, 50-move,
insufficient material). Adjudicates after ``--max-plies``.

Result tabulation prints:
- A wins / draws / A losses
- A score (winrate including half-credit for draws)
- Crude Elo estimate from winrate (saturates near 0/1)

Usage::

    python scripts/match_vs_uci.py \\
        --engine-a "python -m chess_anti_engine.uci --checkpoint <path> --device cpu" \\
        --engine-b /path/to/rofChadeAVX2 \\
        --games 50 --time-ms 200
"""
from __future__ import annotations

import argparse
import math
import shlex
import time
from pathlib import Path

import chess
import chess.engine


def _open_engine(spec: str, cwd: str | None = None) -> chess.engine.SimpleEngine:
    """``spec`` is either a single binary path or a shell-style command.

    For single-binary specs, defaults to spawning with cwd set to the
    binary's directory — many engines (rofChade, Stockfish with NNUE
    sidecar) look for net/data files relative to cwd.
    """
    parts = shlex.split(spec)
    if not parts:
        raise ValueError(f"empty engine spec: {spec!r}")
    if len(parts) == 1 and Path(parts[0]).is_file():
        binary = Path(parts[0]).resolve()
        return chess.engine.SimpleEngine.popen_uci(
            str(binary), cwd=cwd or str(binary.parent),
        )
    return chess.engine.SimpleEngine.popen_uci(parts, cwd=cwd)


def _set_options(eng: chess.engine.SimpleEngine, opts: dict[str, str]) -> None:
    for k, v in opts.items():
        if k in eng.options:
            try:
                option = eng.options[k]
                if option.type == "spin":
                    eng.configure({k: int(v)})
                elif option.type == "check":
                    eng.configure({k: str(v).lower() in ("1", "true", "yes")})
                else:
                    eng.configure({k: v})
            except (chess.engine.EngineError, ValueError) as exc:
                print(f"[warn] could not set {k}={v}: {exc}")


def play_one_game(
    eng_w: chess.engine.SimpleEngine,
    eng_b: chess.engine.SimpleEngine,
    *,
    time_ms: int,
    max_plies: int,
) -> tuple[str, int]:
    """Play one game. Returns (result_str, plies). result_str in {"1-0","0-1","1/2-1/2"}."""
    board = chess.Board()
    limit = chess.engine.Limit(time=time_ms / 1000.0)
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        eng = eng_w if board.turn == chess.WHITE else eng_b
        result = eng.play(board, limit)
        if result.move is None:
            break
        board.push(result.move)
        plies += 1
    if board.is_game_over(claim_draw=True):
        return board.result(claim_draw=True), plies
    return "1/2-1/2", plies  # adjudicated draw at max_plies


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-a", required=True, help="UCI engine A (binary path or shell command)")
    p.add_argument("--engine-b", required=True, help="UCI engine B (binary path or shell command)")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--time-ms", type=int, default=200, help="per-move time budget (ms)")
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--option-a", action="append", default=[], help="Set UCI option on A (Name=Value)")
    p.add_argument("--option-b", action="append", default=[], help="Set UCI option on B (Name=Value)")
    args = p.parse_args()

    def _parse_opts(specs: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for spec in specs:
            if "=" not in spec:
                raise SystemExit(f"--option must be Name=Value, got {spec!r}")
            k, v = spec.split("=", 1)
            out[k.strip()] = v.strip()
        return out

    opts_a = _parse_opts(args.option_a)
    opts_b = _parse_opts(args.option_b)

    print(f"[match] opening A: {args.engine_a}")
    eng_a = _open_engine(args.engine_a)
    _set_options(eng_a, opts_a)
    print(f"[match]   id={eng_a.id.get('name', '?')!r}")

    print(f"[match] opening B: {args.engine_b}")
    eng_b = _open_engine(args.engine_b)
    _set_options(eng_b, opts_b)
    print(f"[match]   id={eng_b.id.get('name', '?')!r}")

    a_wins = a_draws = a_losses = 0
    a_white_count = 0
    a_black_count = 0
    total_plies = 0
    t0 = time.time()
    print(f"[match] playing {args.games} games at time={args.time_ms}ms/move, max_plies={args.max_plies}")

    try:
        for i in range(args.games):
            a_white = (i % 2 == 0)
            if a_white:
                a_white_count += 1
                result, plies = play_one_game(eng_a, eng_b, time_ms=args.time_ms, max_plies=args.max_plies)
                if result == "1-0":
                    a_wins += 1
                elif result == "0-1":
                    a_losses += 1
                else:
                    a_draws += 1
            else:
                a_black_count += 1
                result, plies = play_one_game(eng_b, eng_a, time_ms=args.time_ms, max_plies=args.max_plies)
                if result == "0-1":
                    a_wins += 1
                elif result == "1-0":
                    a_losses += 1
                else:
                    a_draws += 1
            total_plies += plies
            elapsed = time.time() - t0
            print(
                f"[game {i+1:>3}/{args.games}] {args.label_a}={'W' if a_white else 'B'} "
                f"result={result} plies={plies}  "
                f"running A: {a_wins}W-{a_draws}D-{a_losses}L  ({elapsed:.0f}s)",
                flush=True,
            )
    finally:
        eng_a.quit()
        eng_b.quit()

    total = a_wins + a_draws + a_losses
    if total == 0:
        print("[match] no games completed.")
        return
    score = (a_wins + 0.5 * a_draws) / total
    dt = time.time() - t0
    print()
    print(f"[match] {total} games in {dt:.0f}s ({dt/total:.1f}s/game, avg {total_plies/total:.0f} plies/game)")
    print(f"[match] {args.label_a} (A) vs {args.label_b} (B):")
    print(f"  A wins   : {a_wins}")
    print(f"  draws    : {a_draws}")
    print(f"  A losses : {a_losses}")
    print(f"  A score  : {score:.3f}  (incl. half for draws)")
    print(f"  A as W/B : {a_white_count}/{a_black_count}")
    if 0.005 < score < 0.995:
        elo = -400.0 * math.log10(1.0 / score - 1.0)
        print(f"  Elo (A - B) ≈ {elo:+.0f}")


if __name__ == "__main__":
    main()
