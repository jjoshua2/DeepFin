#!/usr/bin/env python3
"""Generate bootstrap data: 100k random games saved as NPZ shards.

Both sides play random legal moves with a 1-ply checkmate check
(if any move delivers checkmate, play it). No network or SF needed.

Each worker processes a small batch (~100 games), writes a shard to disk,
then frees memory. This keeps peak memory under control.

Positions are encoded in the PRODUCTION input encoding by default (read from
``--config``), not the encoder's legacy v1 default. Bootstrap shards are the
one live producer of legacy-encoded replay rows, and a legacy row entering an
``lc0_root_legacy_meta`` window is exactly the case
``select_input_history_arrays`` now refuses (docs/rl_loop_audit.md M11/M12).

Usage:
    PYTHONPATH=. python3 scripts/generate_bootstrap.py --games 100000 --out data/bootstrap
    PYTHONPATH=. python3 scripts/generate_bootstrap.py --games 100000 --out data/bootstrap --workers 8
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding
from chess_anti_engine.moves.encode import legal_move_mask
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import ShardMeta, save_npz
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

MAX_PLIES = 300  # Hard cap per game
GAMES_PER_BATCH = 100  # Small batches to limit per-worker memory
DEFAULT_CONFIG = "configs/pbt2_small.yaml"


@dataclass(frozen=True)
class EncodingSpec:
    """The input encoding the generated shards are written in."""

    input_history_encoding: str
    input_extra_features: str
    history_rep_fix: bool


def play_one_random_game(seed: int, enc: EncodingSpec) -> list[ReplaySample]:
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
        x = encode_position(
            board,
            add_features=True,
            feature_dropout_p=0.0,
            input_history_encoding=enc.input_history_encoding,
            input_extra_features=enc.input_extra_features,
        )

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
        if is_white_turn:  # noqa: SIM108 — nested ternary would bury the POV flip
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
            input_history_encoding=enc.input_history_encoding,
        )
        samples.append(s)

    return samples


def _worker_batch(args: tuple[int, int, str, EncodingSpec]) -> tuple[str, int, int, int, int, int]:
    """Play a batch of games, write shard to disk, return (path, n_positions, wins, draws, losses, n_games)."""
    start_seed, count, out_path, enc = args
    # Process-global encoder switch; each Pool worker is its own process.
    rep_fix.apply(enc.history_rep_fix)
    all_samples: list[ReplaySample] = []
    wins = draws = losses = 0
    for i in range(count):
        samples = play_one_random_game(start_seed + i, enc)
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
        input_history_encoding=enc.input_history_encoding,
        history_rep_fix=enc.history_rep_fix,
    )
    save_npz(out_path, samples=all_samples, meta=meta)
    return out_path, len(all_samples), wins, draws, losses, count


def _encoding_spec(args: argparse.Namespace) -> EncodingSpec:
    """Resolve the shard encoding: config values unless explicitly overridden.

    Defaulting to the production config (rather than to the encoder's legacy
    v1 fallback) is the point: bootstrap shards must be readable by the model
    that will train on them.
    """
    cfg = flatten_run_config_defaults(load_yaml_file(args.config))
    history = args.input_history_encoding or cfg.get("input_history_encoding")
    extra = args.input_extra_features or cfg.get("input_extra_features")
    if not history or not extra:
        raise SystemExit(
            f"{args.config} does not define input_history_encoding / "
            "input_extra_features; pass them explicitly"
        )
    return EncodingSpec(
        input_history_encoding=normalize_lc0_history_encoding(str(history)),
        input_extra_features=str(extra),
        history_rep_fix=bool(cfg.get("history_rep_fix", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random bootstrap games")
    parser.add_argument("--games", type=int, default=100_000, help="Number of games to generate")
    parser.add_argument("--out", type=str, default="data/bootstrap", help="Output directory")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto)")
    parser.add_argument("--batch-games", type=int, default=GAMES_PER_BATCH,
                        help="Games per worker batch/shard (controls peak memory)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing bootstrap_*.npz shards in --out")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Config the input encoding defaults are read from")
    parser.add_argument("--input-history-encoding", type=str, default=None,
                        help="Override the config's input_history_encoding")
    parser.add_argument("--input-extra-features", type=str, default=None,
                        help="Override the config's input_extra_features")
    args = parser.parse_args()

    if args.games <= 0:
        raise SystemExit("--games must be > 0")
    if args.workers < 0:
        raise SystemExit("--workers must be >= 0")
    if args.batch_games <= 0:
        raise SystemExit("--batch-games must be > 0")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("bootstrap_*.npz"))
    if existing and not args.overwrite:
        raise SystemExit(
            f"{out_dir} already contains {len(existing)} bootstrap shard(s); "
            "pass --overwrite to replace them"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    enc = _encoding_spec(args)
    rep_fix.apply(enc.history_rep_fix)
    print(
        f"Encoding: history={enc.input_history_encoding} "
        f"extra_features={enc.input_extra_features} rep_fix={enc.history_rep_fix}"
    )

    workers = args.workers or min(cpu_count(), 8)  # Cap at 8 to limit memory
    games = args.games
    batch_games = args.batch_games

    # Split games into small batches (each becomes one shard on disk)
    batches: list[tuple[int, int, str, EncodingSpec]] = []
    remaining = games
    seed = args.seed
    shard_idx = 0
    while remaining > 0:
        n = min(batch_games, remaining)
        shard_path = str(out_dir / f"bootstrap_{shard_idx:04d}.npz")
        batches.append((seed, n, shard_path, enc))
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
