#!/usr/bin/env python3
"""Generate bootstrap data: 100k random games saved as NPZ shards.

Both sides play random legal moves with a 1-ply checkmate check
(if any move delivers checkmate, play it). No network or SF needed.

Each worker processes a small batch (~100 games), writes a shard to disk,
then frees memory. This keeps peak memory under control.

Usage:
    PYTHONPATH=. python3 scripts/generate_bootstrap.py --games 100000 --out data/bootstrap
    PYTHONPATH=. python3 scripts/generate_bootstrap.py --games 100000 --out data/bootstrap --workers 8
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.moves.encode import POLICY_SIZE, legal_move_mask, move_to_index
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import ShardMeta, save_npz


MAX_PLIES = 300  # Hard cap per game
GAMES_PER_BATCH = 100  # Small batches to limit per-worker memory


def play_one_random_game(seed: int) -> list[ReplaySample]:
    """Play a single random game, return list of ReplaySamples."""
    rng = np.random.default_rng(seed)
    board = chess.Board()

    # Records: (encoded_x, policy_uniform, legal_mask, side_to_move_is_white, ply)
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]] = []

    for ply in range(MAX_PLIES):
        if board.is_game_over(claim_draw=True):
            break

        legal = list(board.legal_moves)
        if not legal:
            break

        # Encode position before the move
        x = encode_position(board, add_features=True, feature_dropout_p=0.0)

        # Uniform policy over legal moves
        lm = legal_move_mask(board)
        n_legal = max(int(lm.sum()), 1)
        policy = lm.astype(np.float32) / n_legal

        records.append((x, policy, lm.astype(np.uint8), board.turn == chess.WHITE, ply))

        # 1-ply checkmate check: if any move is checkmate, play it
        chosen = None
        for m in legal:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                chosen = m
                break
            board.pop()

        if chosen is None:
            chosen = legal[int(rng.integers(len(legal)))]

        board.push(chosen)

    # Determine result
    result = board.result(claim_draw=True)
    if result == "1-0":
        white_wdl = 0  # white won
    elif result == "0-1":
        white_wdl = 2  # white lost
    else:
        white_wdl = 1  # draw

    # Build samples — WDL is side-to-move relative (same convention as training)
    samples: list[ReplaySample] = []
    total_plies = len(records)
    for x, policy, lm, is_white_turn, ply in records:
        # Convert white_wdl to side-to-move-relative wdl
        if is_white_turn:
            wdl = white_wdl  # 0=stm won, 2=stm lost
        else:
            # Flip: white_wdl 0 (white won) -> 2 (black lost from black's POV)
            wdl = 2 - white_wdl if white_wdl != 1 else 1

        s = ReplaySample(
            x=x,
            policy_target=policy,
            wdl_target=wdl,
            priority=1.0,
            has_policy=False,  # Random moves, not searched
            legal_mask=lm,
            moves_left=float(total_plies - ply) / MAX_PLIES if total_plies > 0 else 0.0,
            is_network_turn=True,  # Both sides are "network" in bootstrap
        )
        samples.append(s)

    return samples


def _worker_batch(args: tuple[int, int, str]) -> tuple[str, int, int, int, int, int]:
    """Play a batch of games, write shard to disk, return (path, n_positions, wins, draws, losses, n_games)."""
    start_seed, count, out_path = args
    all_samples: list[ReplaySample] = []
    wins = draws = losses = 0
    for i in range(count):
        samples = play_one_random_game(start_seed + i)
        if samples:
            wdl_first = samples[0].wdl_target  # From white's perspective (first move is white)
            if wdl_first == 0:
                wins += 1
            elif wdl_first == 1:
                draws += 1
            else:
                losses += 1
        all_samples.extend(samples)

    # Write shard to disk immediately (free memory in caller)
    meta = ShardMeta(
        games=count,
        positions=len(all_samples),
        wins=wins,
        draws=draws,
        losses=losses,
    )
    save_npz(out_path, samples=all_samples, meta=meta)
    return out_path, len(all_samples), wins, draws, losses, count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random bootstrap games")
    parser.add_argument("--games", type=int, default=100_000, help="Number of games to generate")
    parser.add_argument("--out", type=str, default="data/bootstrap", help="Output directory")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto)")
    parser.add_argument("--batch-games", type=int, default=GAMES_PER_BATCH,
                        help="Games per worker batch/shard (controls peak memory)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers or min(cpu_count(), 8)  # Cap at 8 to limit memory
    games = args.games
    batch_games = args.batch_games

    # Split games into small batches (each becomes one shard on disk)
    batches: list[tuple[int, int, str]] = []
    remaining = games
    seed = args.seed
    shard_idx = 0
    while remaining > 0:
        n = min(batch_games, remaining)
        shard_path = str(out_dir / f"bootstrap_{shard_idx:04d}.npz")
        batches.append((seed, n, shard_path))
        seed += n
        remaining -= n
        shard_idx += 1

    print(f"Generating {games} random games → {len(batches)} shards with {workers} workers...")
    t0 = time.time()

    total_positions = 0
    total_games_done = 0
    total_wins = total_draws = total_losses = 0

    if workers == 1:
        for batch in batches:
            path, n_pos, w, d, l, n_games = _worker_batch(batch)
            total_positions += n_pos
            total_games_done += n_games
            total_wins += w
            total_draws += d
            total_losses += l
            elapsed = time.time() - t0
            rate = total_games_done / elapsed if elapsed > 0 else 0
            print(f"  {total_games_done}/{games} games ({total_positions} positions) "
                  f"[{rate:.0f} games/s] → {path}")
    else:
        with Pool(workers) as pool:
            for path, n_pos, w, d, l, n_games in pool.imap_unordered(_worker_batch, batches):
                total_positions += n_pos
                total_games_done += n_games
                total_wins += w
                total_draws += d
                total_losses += l
                elapsed = time.time() - t0
                rate = total_games_done / elapsed if elapsed > 0 else 0
                print(f"  {total_games_done}/{games} games ({total_positions} positions) "
                      f"[{rate:.0f} games/s] → {Path(path).name}")

    elapsed = time.time() - t0
    print(f"\nDone: {games} games, {total_positions} positions in {elapsed:.1f}s")
    print(f"  W/D/L: {total_wins}/{total_draws}/{total_losses}")
    print(f"  Avg positions/game: {total_positions / max(games, 1):.1f}")
    print(f"  {len(batches)} shard(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
