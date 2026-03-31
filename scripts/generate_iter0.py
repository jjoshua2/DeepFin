#!/usr/bin/env python3
"""Generate shared iteration-0 selfplay data from the bootstrap net.

All PB2 trials start from the same bootstrap checkpoint and identical SF
settings, so their first iteration of selfplay is redundant. This script plays
those games once and saves the samples as NPZ shards that each trial can copy
into its replay directory on startup.

Usage:
    PYTHONPATH=. python3 scripts/generate_iter0.py --config configs/pbt2_small.yaml

Output:
    data/iter0_shared/shard_000000.npz ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.replay.shard import ShardMeta, save_npz
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish import StockfishUCI
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def _load_model_from_checkpoint(*, ckpt_path: Path, cfg: ModelConfig, device: str) -> torch.nn.Module:
    model = build_model(cfg)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(str(device))
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate shared iter-0 selfplay data")
    ap.add_argument("--config", type=str, default="configs/pbt2_small.yaml")
    ap.add_argument("--output-dir", type=str, default="data/iter0_shared")
    ap.add_argument("--games", type=int, default=None, help="Override games_per_iter")
    ap.add_argument("--batch", type=int, default=10, help="Mini-batch size for selfplay")
    ap.add_argument("--shard-size", type=int, default=1000, help="Samples per shard")
    ap.add_argument("--overwrite", action="store_true", help="Delete any existing shard_*.npz in output-dir")
    args = ap.parse_args()

    cfg = load_yaml_file(args.config)
    flat = flatten_run_config_defaults(cfg)

    device = str(flat.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(flat.get("seed", 0))
    rng = np.random.default_rng(seed)

    bootstrap_ckpt = flat.get("bootstrap_checkpoint")
    if not bootstrap_ckpt:
        raise ValueError("No bootstrap_checkpoint in config")
    bp = Path(str(bootstrap_ckpt))
    if not bp.exists():
        raise FileNotFoundError(f"Bootstrap checkpoint not found: {bp}")

    model_cfg = ModelConfig(
        kind=str(flat.get("model", "transformer")),
        embed_dim=int(flat.get("embed_dim", 128)),
        num_layers=int(flat.get("num_layers", 4)),
        num_heads=int(flat.get("num_heads", 4)),
        ffn_mult=int(flat.get("ffn_mult", 2)),
        use_smolgen=not bool(flat.get("no_smolgen", False)),
        use_nla=bool(flat.get("use_nla", False)),
        use_qk_rmsnorm=bool(flat.get("use_qk_rmsnorm", False)),
        use_gradient_checkpointing=bool(flat.get("gradient_checkpointing", False)),
    )

    print(f"Loading bootstrap checkpoint: {bp}")
    model = _load_model_from_checkpoint(ckpt_path=bp, cfg=model_cfg, device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("shard_*.npz"))
    if existing:
        if not bool(args.overwrite):
            raise SystemExit(
                f"Refusing to write into non-empty {out_dir} (found {len(existing)} shard_*.npz). "
                "Pass --overwrite to delete them first."
            )
        for p in existing:
            p.unlink(missing_ok=True)

    # Stockfish.
    sf_path = str(flat.get("stockfish_path", "stockfish"))
    sf_nodes = int(flat.get("sf_nodes", 250))
    sf_multipv = int(flat.get("sf_multipv", 3))
    sf = StockfishUCI(sf_path, nodes=sf_nodes, multipv=sf_multipv)

    total_games = int(args.games) if args.games is not None else int(flat.get("games_per_iter", 1000))
    batch_size = max(1, int(args.batch))
    shard_size = max(1, int(args.shard_size))

    opponent_cfg = OpponentConfig(
        random_move_prob=float(flat.get("sf_pid_random_move_prob_start", 1.0)),
    )
    temp_cfg = TemperatureConfig(
        temperature=float(flat.get("temperature", 1.0)),
        drop_plies=int(flat.get("temperature_drop_plies", 0)),
        after=float(flat.get("temperature_after", 0.0)),
        decay_start_move=int(flat.get("temperature_decay_start_move", 20)),
        decay_moves=int(flat.get("temperature_decay_moves", 60)),
        endgame=float(flat.get("temperature_endgame", 0.6)),
    )
    search_cfg = SearchConfig(
        simulations=int(flat.get("mcts_simulations", 64)),
        mcts_type=str(flat.get("mcts", "puct")),
        playout_cap_fraction=float(flat.get("playout_cap_fraction", 0.25)),
        fast_simulations=int(flat.get("fast_simulations", 8)),
        fpu_reduction=float(flat.get("fpu_reduction", 1.2)),
        fpu_at_root=float(flat.get("fpu_at_root", 1.0)),
    )
    opening_cfg = OpeningConfig(
        opening_book_path=flat.get("opening_book_path"),
        opening_book_max_plies=int(flat.get("opening_book_max_plies", 4)),
        opening_book_max_games=int(flat.get("opening_book_max_games", 200_000)),
        opening_book_prob=float(flat.get("opening_book_prob", 1.0)),
        random_start_plies=int(flat.get("random_start_plies", 0)),
    )
    diff_focus_cfg = DiffFocusConfig(
        enabled=bool(flat.get("diff_focus_enabled", True)),
        q_weight=float(flat.get("diff_focus_q_weight", 6.0)),
        pol_scale=float(flat.get("diff_focus_pol_scale", 3.5)),
        slope=float(flat.get("diff_focus_slope", 3.0)),
        min_keep=float(flat.get("diff_focus_min", 0.025)),
    )
    game_cfg = GameConfig(
        max_plies=int(flat.get("max_plies", 120)),
        sf_policy_temp=float(flat.get("sf_policy_temp", 0.25)),
        sf_policy_label_smooth=float(flat.get("sf_policy_label_smooth", 0.05)),
        timeout_adjudication_threshold=float(flat.get("timeout_adjudication_threshold", 0.90)),
        volatility_source=str(flat.get("volatility_source", "raw")),
        syzygy_path=flat.get("syzygy_path"),
        syzygy_policy=bool(flat.get("syzygy_policy", False)),
        categorical_bins=int(flat.get("categorical_bins", 32)),
        hlgauss_sigma=float(flat.get("hlgauss_sigma", 0.04)),
    )
    selfplay_kwargs = dict(
        device=device, rng=rng, stockfish=sf,
        opponent=opponent_cfg, temp=temp_cfg, search=search_cfg,
        opening=opening_cfg, diff_focus=diff_focus_cfg, game=game_cfg,
    )

    run_id = f"iter0_shared_seed{seed}_{int(time.time())}"

    games_remaining = int(total_games)
    total_positions = 0
    total_w = total_d = total_l = 0
    write_buf: list = []
    shard_idx = 0
    t0 = time.time()

    print(
        f"Playing {total_games} games (batch={batch_size}, sf_nodes={sf_nodes}, sims={search_cfg.simulations}, "
        f"random_move_prob={opponent_cfg.random_move_prob:.2f})..."
    )

    try:
        while games_remaining > 0:
            chunk = min(batch_size, games_remaining)
            samples, stats = play_batch(model, games=chunk, **selfplay_kwargs)
            games_remaining -= chunk

            total_positions += int(stats.positions)
            total_w += int(stats.w)
            total_d += int(stats.d)
            total_l += int(stats.l)

            write_buf.extend(samples)
            del samples

            while len(write_buf) >= shard_size:
                shard_samples = write_buf[:shard_size]
                write_buf = write_buf[shard_size:]

                shard_path = out_dir / f"shard_{shard_idx:06d}.npz"
                meta = ShardMeta(
                    run_id=run_id,
                    generated_at_unix=int(time.time()),
                    positions=int(len(shard_samples)),
                )
                save_npz(shard_path, samples=shard_samples, meta=meta)
                shard_idx += 1

            elapsed = time.time() - t0
            games_done = total_games - games_remaining
            rate = games_done / elapsed if elapsed > 0 else 0.0
            eta = (games_remaining / rate) if rate > 0 else 0.0
            print(
                f"  {games_done}/{total_games} games  {total_positions} pos  "
                f"W/D/L={total_w}/{total_d}/{total_l}  "
                f"{rate:.1f} games/s  ETA {eta:.0f}s"
            )

        if write_buf:
            shard_path = out_dir / f"shard_{shard_idx:06d}.npz"
            meta = ShardMeta(
                run_id=run_id,
                generated_at_unix=int(time.time()),
                positions=int(len(write_buf)),
            )
            save_npz(shard_path, samples=write_buf, meta=meta)
            shard_idx += 1

    finally:
        sf.close()

    elapsed = time.time() - t0
    print(f"\nDone: {total_games} games, {total_positions} positions, {shard_idx} shards")
    print(f"W/D/L = {total_w}/{total_d}/{total_l}")
    print(f"Saved to {out_dir}/")
    print(f"Time: {elapsed:.0f}s ({total_games / max(1e-9, elapsed):.1f} games/s)")


if __name__ == "__main__":
    main()
