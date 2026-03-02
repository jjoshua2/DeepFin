from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.config import RunConfig, StockfishConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.replay import DiskReplayBuffer, ReplayBuffer, balance_wdl
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.budget import progressive_mcts_simulations
from chess_anti_engine.stockfish import DifficultyPID, StockfishPool, StockfishUCI
from chess_anti_engine.train import Trainer
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def _run_single(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = RunConfig(
        iterations=int(args.iterations),
        replay_capacity=int(args.replay_capacity),
    )
    cfg.work_dir = Path(args.work_dir)
    cfg.seed = int(args.seed)
    cfg.train.device = device
    cfg.train.lr = float(args.lr)
    cfg.train.batch_size = int(args.batch_size)
    cfg.train.train_steps_per_iter = int(args.train_steps)
    cfg.selfplay.games_per_iter = int(args.games_per_iter)
    cfg.selfplay.temperature = float(args.temperature)
    cfg.selfplay.temperature_drop_plies = int(args.temperature_drop_plies)
    cfg.selfplay.temperature_after = float(args.temperature_after)
    cfg.selfplay.temperature_decay_start_move = int(args.temperature_decay_start_move)
    cfg.selfplay.temperature_decay_moves = int(args.temperature_decay_moves)
    cfg.selfplay.temperature_endgame = float(args.temperature_endgame)
    cfg.selfplay.max_plies = int(args.max_plies)
    cfg.selfplay.opening_book_path = args.opening_book_path
    cfg.selfplay.opening_book_max_plies = int(args.opening_book_max_plies)
    cfg.selfplay.opening_book_max_games = int(args.opening_book_max_games)
    cfg.selfplay.opening_book_prob = float(args.opening_book_prob)
    cfg.selfplay.random_start_plies = int(args.random_start_plies)
    cfg.selfplay.sf_policy_temp = float(args.sf_policy_temp)
    cfg.selfplay.sf_policy_label_smooth = float(args.sf_policy_label_smooth)
    cfg.selfplay.timeout_adjudication_threshold = float(
        getattr(args, "timeout_adjudication_threshold", cfg.selfplay.timeout_adjudication_threshold)
    )
    cfg.stockfish = StockfishConfig(path=args.stockfish_path, nodes=int(args.sf_nodes), multipv=int(args.sf_multipv))
    cfg.stockfish.pid_enabled = bool(args.sf_pid_enabled)
    cfg.stockfish.pid_target_winrate = float(args.sf_pid_target_winrate)
    cfg.stockfish.pid_ema_alpha = float(args.sf_pid_ema_alpha)
    cfg.stockfish.pid_deadzone = float(args.sf_pid_deadzone)
    cfg.stockfish.pid_rate_limit = float(args.sf_pid_rate_limit)
    cfg.stockfish.pid_min_games_between_adjust = int(args.sf_pid_min_games_between_adjust)
    cfg.stockfish.pid_kp = float(args.sf_pid_kp)
    cfg.stockfish.pid_ki = float(args.sf_pid_ki)
    cfg.stockfish.pid_kd = float(args.sf_pid_kd)
    cfg.stockfish.pid_integral_clamp = float(args.sf_pid_integral_clamp)
    cfg.stockfish.pid_min_nodes = int(args.sf_pid_min_nodes)
    cfg.stockfish.pid_max_nodes = int(args.sf_pid_max_nodes)

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.seed))

    model = build_model(
        ModelConfig(
            kind=str(args.model),
            embed_dim=int(args.embed_dim),
            num_layers=int(args.num_layers),
            num_heads=int(args.num_heads),
            ffn_mult=int(args.ffn_mult),
            use_smolgen=not bool(args.no_smolgen),
            use_nla=bool(args.use_nla),
            use_gradient_checkpointing=bool(args.gradient_checkpointing),
        )
    )
    trainer = Trainer(
        model,
        device=cfg.train.device,
        lr=cfg.train.lr,
        grad_clip=float(args.grad_clip),
        log_dir=cfg.work_dir / "tb",
        use_amp=not bool(args.no_amp),
        feature_dropout_p=float(args.feature_dropout_p),
        w_volatility=float(args.w_volatility),
        accum_steps=int(args.accum_steps),
        warmup_steps=int(args.warmup_steps),
        lr_eta_min=float(args.lr_eta_min),
        lr_T0=int(args.lr_T0),
        lr_T_mult=int(args.lr_T_mult),
        use_compile=bool(args.use_compile),
        optimizer=str(args.optimizer),
        swa_start=int(args.swa_start),
        swa_freq=int(args.swa_freq),
        w_policy=float(getattr(args, "w_policy", 1.0)),
        w_soft=float(getattr(args, "w_soft", 0.5)),
        w_future=float(getattr(args, "w_future", 0.15)),
        w_wdl=float(getattr(args, "w_wdl", 1.0)),
        w_sf_move=float(getattr(args, "w_sf_move", 0.15)),
        w_sf_eval=float(getattr(args, "w_sf_eval", 0.15)),
        w_categorical=float(getattr(args, "w_categorical", 0.10)),
        w_sf_volatility=float(getattr(args, "w_sf_volatility", getattr(args, "w_volatility", 0.05))),
        w_moves_left=float(getattr(args, "w_moves_left", 0.02)),
        w_sf_wdl=float(getattr(args, "w_sf_wdl", 1.0)),
    )

    # Growing sliding window
    window_start = int(getattr(args, "replay_window_start", 100_000))
    window_max = int(getattr(args, "replay_window_max", cfg.replay_capacity))
    window_growth = int(getattr(args, "replay_window_growth", 10_000))
    current_window = window_start
    buf = DiskReplayBuffer(
        current_window,
        shard_dir=cfg.work_dir / "replay_shards",
        rng=rng,
        shuffle_cap=int(getattr(args, "shuffle_buffer_size", 20_000)),
        shard_size=int(getattr(args, "shard_size", 1000)),
    )

    # On resume, DiskReplayBuffer discovers existing shards on disk. If the on-disk
    # sample count already exceeds replay_window_start, keep the effective capacity
    # large enough to avoid pruning old shards just because we restarted.
    current_window = max(int(current_window), int(len(buf)))
    buf.capacity = int(current_window)

    # Load pre-trained bootstrap checkpoint (trained offline via scripts/train_bootstrap.py).
    # Only load MODEL WEIGHTS — see trainable.py comment for why we skip optimizer/scheduler.
    bootstrap_ckpt = getattr(args, "bootstrap_checkpoint", None)
    if bootstrap_ckpt:
        bp = Path(bootstrap_ckpt)
        if bp.exists():
            print(f"Loading pre-trained bootstrap model weights: {bp}")
            ckpt_data = torch.load(str(bp), map_location=args.device)
            trainer.model.load_state_dict(ckpt_data["model"])
            del ckpt_data
        else:
            print(f"WARNING: bootstrap checkpoint not found: {bp}")

    if int(args.sf_workers) > 1:
        sf = StockfishPool(
            path=cfg.stockfish.path,
            nodes=cfg.stockfish.nodes,
            num_workers=int(args.sf_workers),
            multipv=int(getattr(cfg.stockfish, "multipv", 1)),
        )
    else:
        sf = StockfishUCI(cfg.stockfish.path, nodes=cfg.stockfish.nodes, multipv=int(getattr(cfg.stockfish, "multipv", 1)))

    ckpt_path = cfg.work_dir / "ckpt.pt"
    if ckpt_path.exists():
        trainer.load(ckpt_path)

    pid_state_path = cfg.work_dir / "pid_state.json"
    pid_state = None
    if pid_state_path.exists():
        try:
            pid_state = json.loads(pid_state_path.read_text(encoding="utf-8"))
        except Exception:
            pid_state = None

    # Best-model tracking
    best_state_path = cfg.work_dir / "best.json"
    best_ckpt_path = cfg.work_dir / "best_ckpt.pt"
    best_model_path = cfg.work_dir / "best_model.pt"

    best_loss = float("inf")
    if best_state_path.exists():
        try:
            d = json.loads(best_state_path.read_text(encoding="utf-8"))
            best_loss = float(d.get("best_loss", d.get("loss", best_loss)))
        except Exception:
            pass

    # Puzzle evaluation suite (optional overspecialization canary).
    puzzle_suite = None
    puzzle_interval = int(getattr(args, "puzzle_interval", 1))
    puzzle_sims = int(getattr(args, "puzzle_simulations", 200))
    if getattr(args, "puzzle_epd", None) and puzzle_interval > 0:
        from chess_anti_engine.eval import load_epd
        puzzle_suite = load_epd(args.puzzle_epd)

    pid = None
    if bool(getattr(cfg.stockfish, "pid_enabled", False)):
        pid = DifficultyPID(
            initial_nodes=int(cfg.stockfish.nodes),
            target_winrate=float(cfg.stockfish.pid_target_winrate),
            ema_alpha=float(cfg.stockfish.pid_ema_alpha),
            deadzone=float(cfg.stockfish.pid_deadzone),
            rate_limit=float(cfg.stockfish.pid_rate_limit),
            min_games_between_adjust=int(cfg.stockfish.pid_min_games_between_adjust),
            kp=float(cfg.stockfish.pid_kp),
            ki=float(cfg.stockfish.pid_ki),
            kd=float(cfg.stockfish.pid_kd),
            integral_clamp=float(cfg.stockfish.pid_integral_clamp),
            min_nodes=int(cfg.stockfish.pid_min_nodes),
            max_nodes=int(cfg.stockfish.pid_max_nodes),
            # Optional opponent random-move schedule (may be provided via YAML defaults
            # even if not exposed as explicit CLI flags).
            initial_random_move_prob=float(getattr(args, "sf_pid_random_move_prob_start", 0.0)),
            random_move_prob_min=float(getattr(args, "sf_pid_random_move_prob_min", 0.0)),
            random_move_prob_max=float(getattr(args, "sf_pid_random_move_prob_max", 1.0)),
            random_move_stage_end=float(getattr(args, "sf_pid_random_move_stage_end", 0.5)),
            max_rand_step=float(getattr(args, "sf_pid_max_rand_step", 0.01)),
        )
        if pid_state is not None:
            try:
                pid.load_state_dict(pid_state)
                if hasattr(sf, "set_nodes"):
                    sf.set_nodes(int(pid.nodes))
                else:
                    setattr(sf, "nodes", int(pid.nodes))
            except Exception:
                pass

    try:
        for it in range(cfg.iterations):
            base_sims = int(args.mcts_simulations)
            sims = base_sims
            if bool(getattr(args, "progressive_mcts", True)):
                sims = progressive_mcts_simulations(
                    int(getattr(trainer, "step", 0)),
                    start=int(getattr(args, "mcts_start_simulations", 50)),
                    max_sims=base_sims,
                    ramp_steps=int(getattr(args, "mcts_ramp_steps", 10_000)),
                    exponent=float(getattr(args, "mcts_ramp_exponent", 2.0)),
                )

            # Mini-batch selfplay: play games in small batches to limit memory
            selfplay_batch = int(getattr(args, "selfplay_batch", 10))
            games_remaining = cfg.selfplay.games_per_iter
            total_positions = 0
            total_w, total_d, total_l = 0, 0, 0
            total_sf_d6 = 0.0
            current_rand = float(pid.random_move_prob) if pid is not None else 0.0

            selfplay_kwargs = dict(
                device=cfg.train.device,
                rng=rng,
                stockfish=sf,
                opponent_random_move_prob=current_rand,
                temperature=cfg.selfplay.temperature,
                temperature_drop_plies=int(cfg.selfplay.temperature_drop_plies),
                temperature_after=float(cfg.selfplay.temperature_after),
                temperature_decay_start_move=int(cfg.selfplay.temperature_decay_start_move),
                temperature_decay_moves=int(cfg.selfplay.temperature_decay_moves),
                temperature_endgame=float(cfg.selfplay.temperature_endgame),
                max_plies=cfg.selfplay.max_plies,
                mcts_simulations=int(sims),
                mcts_type=str(args.mcts),
                playout_cap_fraction=float(args.playout_cap_fraction),
                fast_simulations=int(args.fast_simulations),
                sf_policy_temp=float(cfg.selfplay.sf_policy_temp),
                sf_policy_label_smooth=float(cfg.selfplay.sf_policy_label_smooth),
                timeout_adjudication_threshold=float(cfg.selfplay.timeout_adjudication_threshold),
                volatility_source=str(getattr(args, "volatility_source", "raw")),
                opening_book_path=cfg.selfplay.opening_book_path,
                opening_book_max_plies=int(cfg.selfplay.opening_book_max_plies),
                opening_book_max_games=int(cfg.selfplay.opening_book_max_games),
                opening_book_prob=float(cfg.selfplay.opening_book_prob),
                random_start_plies=int(cfg.selfplay.random_start_plies),
                syzygy_path=args.syzygy_path,
                syzygy_policy=bool(args.syzygy_policy),
                diff_focus_enabled=bool(getattr(args, "diff_focus_enabled", True)),
                diff_focus_q_weight=float(getattr(args, "diff_focus_q_weight", 6.0)),
                diff_focus_pol_scale=float(getattr(args, "diff_focus_pol_scale", 3.5)),
                diff_focus_slope=float(getattr(args, "diff_focus_slope", 3.0)),
                diff_focus_min=float(getattr(args, "diff_focus_min", 0.025)),
                categorical_bins=int(getattr(args, "categorical_bins", 32)),
                hlgauss_sigma=float(getattr(args, "hlgauss_sigma", 0.04)),
                fpu_reduction=float(getattr(args, "fpu_reduction", 1.2)),
                fpu_at_root=float(getattr(args, "fpu_at_root", 1.0)),
            )
            while games_remaining > 0:
                chunk = min(selfplay_batch, games_remaining)
                samples, sp_stats = play_batch(
                    trainer.model, games=chunk, **selfplay_kwargs,
                )
                games_remaining -= chunk
                total_positions += sp_stats.positions
                total_w += sp_stats.w
                total_d += sp_stats.d
                total_l += sp_stats.l
                total_sf_d6 += sp_stats.sf_eval_delta6 * chunk
                buf.add_many(samples)
                del samples

            buf.flush()

            # KataGo-style sliding window: train on full buffer, steps = new_positions / batch.
            steps = max(1, total_positions // cfg.train.batch_size)
            metrics = trainer.train_steps(
                buf,
                batch_size=cfg.train.batch_size,
                steps=steps,
            )

            # Update PID once per iteration (after training) so difficulty changes align
            # to net updates rather than intra-iteration selfplay noise.
            pid_ema = None
            rand_next = None
            sf_nodes_next = None
            if pid is not None and (total_w + total_d + total_l) > 0:
                upd = pid.observe(wins=total_w, draws=total_d, losses=total_l)
                pid_ema = float(upd.ema_winrate)
                rand_next = float(pid.random_move_prob)
                sf_nodes_next = int(pid.nodes)
                if hasattr(sf, "set_nodes"):
                    sf.set_nodes(int(pid.nodes))
                else:
                    setattr(sf, "nodes", int(pid.nodes))

                # Persist PID state so restarts resume at the same difficulty.
                try:
                    pid_state_path.write_text(
                        json.dumps(pid.state_dict(), sort_keys=True, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

            # Best model (by train loss)
            cur_loss = float(metrics.loss)
            if cur_loss < best_loss - 1e-12:
                best_loss = cur_loss
                trainer.save(best_ckpt_path)
                trainer.export_swa(best_model_path)
                best_state_path.write_text(
                    json.dumps(
                        {
                            "best_loss": float(best_loss),
                            "iter": int(it),
                            "trainer_step": int(getattr(trainer, "step", 0)),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

            # Puzzle evaluation (periodic overspecialization canary).
            puzzle_str = ""
            if puzzle_suite is not None and puzzle_interval > 0 and (it % puzzle_interval == 0):
                from chess_anti_engine.eval import run_puzzle_eval
                pr = run_puzzle_eval(
                    trainer.model, puzzle_suite,
                    device=cfg.train.device, mcts_simulations=puzzle_sims, rng=rng,
                )
                puzzle_str = f" puzzle={pr.correct}/{pr.total}({pr.accuracy:.1%})"
                trainer.writer.add_scalar("eval/puzzle_accuracy", pr.accuracy, trainer.step)

            # quick eval: reuse selfplay stats at current sf nodes
            print(
                f"iter={it} step={getattr(trainer, 'step', 0)} sims={int(sims)} replay={len(buf)} pos_added={total_positions} "
                f"W/D/L={total_w}/{total_d}/{total_l} "
                f"loss={metrics.loss:.4f} best={best_loss:.4f} pol={metrics.policy_loss:.4f} soft={metrics.soft_policy_loss:.4f} "
                f"fut={metrics.future_policy_loss:.4f} wdl={metrics.wdl_loss:.4f} "
                f"sf_move={metrics.sf_move_loss:.4f} sf_acc={metrics.sf_move_acc:.3f} sf_eval={metrics.sf_eval_loss:.4f} sf_d6={total_sf_d6 / max(1, cfg.selfplay.games_per_iter):.4f} "
                f"cat={metrics.categorical_loss:.4f} vol={metrics.volatility_loss:.4f} sf_vol={metrics.sf_volatility_loss:.4f} ml={metrics.moves_left_loss:.4f} "
                f"rand_prob={float(current_rand):.3f} rand_next={rand_next} sf_nodes_next={sf_nodes_next} pid_wr={pid_ema}"
                + puzzle_str
            )

            trainer.save(ckpt_path)

    finally:
        sf.close()


def main() -> None:
    # Two-pass parse so a YAML config can provide defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    pre_args, _ = pre.parse_known_args()

    cfg_defaults: dict[str, object] = {}
    if pre_args.config:
        cfg = load_yaml_file(pre_args.config)
        cfg_defaults = flatten_run_config_defaults(cfg)

    ap = argparse.ArgumentParser(parents=[pre])

    # Mandatory harness: default to Ray Tune.
    ap.add_argument("--mode", type=str, default="tune", choices=["tune", "single"])
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous Ray Tune run (restore errored and unfinished trials)",
    )

    # Disk usage controls
    ap.add_argument(
        "--tune-num-to-keep",
        type=int,
        default=2,
        help="Ray Tune: number of checkpoints to keep per trial (older ones are deleted).",
    )
    ap.add_argument(
        "--tune-keep-last-experiments",
        type=int,
        default=2,
        help="When starting a fresh (non-resume) run, keep only the most recent N Tune experiments in work_dir/tune (0 disables).",
    )

    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--max-concurrent-trials", type=int, default=10)
    ap.add_argument("--cpus-per-trial", type=int, default=2)
    ap.add_argument(
        "--gpus-per-trial", type=float, default=0.1,
        help="GPU fraction per trial. Use <1.0 to pack multiple trials on one GPU "
             "(e.g. 0.1 for 10 trials on a single 5090).",
    )
    ap.add_argument(
        "--tune-scheduler", type=str, default="pb2", choices=["pb2", "asha"],
        help="Tune scheduler: pb2 (PB2, recommended for RL) or asha (legacy, for architecture ablations).",
    )
    ap.add_argument(
        "--pb2-perturbation-interval", type=int, default=10,
        help="PB2: iterations between exploit+explore steps.",
    )
    ap.add_argument("--search-smolgen", action="store_true",
                    help="PB2/ASHA: include smolgen on/off as a binary search dimension.")
    ap.add_argument("--search-nla", action="store_true",
                    help="PB2/ASHA: include NLA on/off as a binary search dimension.")
    ap.add_argument("--search-optimizer", action="store_true",
                    help="PB2/ASHA: include optimizer (nadamw vs soap) as a binary search dimension.")

    ap.add_argument("--tune-metric", type=str, default="train_loss")
    ap.add_argument("--tune-mode", type=str, default=None, choices=["min", "max"])
    ap.add_argument(
        "--search-feature-dropout-p",
        action="store_true",
        help="If set, Tune searches feature_dropout_p; otherwise it is pinned to YAML/CLI value.",
    )
    ap.add_argument(
        "--search-w-volatility",
        action="store_true",
        help="If set, Tune searches w_volatility (0/0.05/0.10); otherwise it is pinned to YAML/CLI value.",
    )
    ap.add_argument(
        "--search-volatility-source",
        action="store_true",
        help="If set, Tune searches volatility_source (raw vs search); otherwise it is pinned to YAML/CLI value.",
    )
    ap.add_argument("--eval-games", type=int, default=0)
    ap.add_argument("--eval-sf-nodes", type=int, default=None)
    ap.add_argument("--eval-mcts-simulations", type=int, default=None)
    ap.add_argument("--holdout-fraction", type=float, default=0.02)
    ap.add_argument("--holdout-capacity", type=int, default=50_000)
    ap.add_argument("--test-steps", type=int, default=10)
    ap.add_argument("--freeze-holdout-at", type=int, default=0)
    ap.add_argument("--reset-holdout-on-drift", action="store_true")
    ap.add_argument("--drift-threshold", type=float, default=0.0)
    ap.add_argument("--drift-sample-size", type=int, default=256)

    # Bootstrap and gating
    ap.add_argument("--bootstrap-dir", type=str, default=None, help="Directory with bootstrap NPZ shards")
    ap.add_argument("--bootstrap-checkpoint", type=str, default=None, help="Path to pre-trained bootstrap checkpoint (from scripts/train_bootstrap.py)")
    ap.add_argument("--bootstrap-max-positions", type=int, default=0, help="Max bootstrap positions to load (0=unlimited)")
    ap.add_argument("--bootstrap-train-steps", type=int, default=0, help="Pre-training steps on bootstrap data (0=disabled)")
    ap.add_argument("--gate-games", type=int, default=0, help="Net gating: games to play after training (0=disabled)")
    ap.add_argument("--gate-threshold", type=float, default=0.50, help="Net gating: reject if winrate below this")
    ap.add_argument("--gate-interval", type=int, default=1, help="Net gating: check every N iterations")
    ap.add_argument("--gate-mcts-sims", type=int, default=1, help="MCTS sims for gate games (1=raw policy)")
    ap.add_argument("--shuffle-buffer-size", type=int, default=20_000, help="Disk replay: in-memory shuffle buffer size")
    ap.add_argument("--shard-size", type=int, default=1000, help="Disk replay: samples per on-disk shard")
    ap.add_argument("--replay-window-start", type=int, default=100_000, help="Initial sliding window size")
    ap.add_argument("--replay-window-max", type=int, default=1_000_000, help="Max sliding window size")
    ap.add_argument("--replay-window-growth", type=int, default=10_000, help="Window growth per iteration")
    ap.add_argument("--w-sf-wdl", type=float, default=1.0, help="SF WDL bootstrap weight for main value head")
    ap.add_argument("--sf-wdl-floor", type=float, default=0.1, help="SF WDL weight floor")
    ap.add_argument("--sf-wdl-floor-at", type=float, default=0.1, help="random_move_prob at which SF WDL weight reaches floor")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--model", type=str, default="transformer", choices=["tiny", "transformer"])
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--ffn-mult", type=int, default=2)
    ap.add_argument("--no-smolgen", action="store_true")
    ap.add_argument("--use-nla", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Enable gradient checkpointing to reduce VRAM (~50%% less activation memory)")
    ap.add_argument("--stockfish-path", type=str, default=None)
    ap.add_argument("--sf-nodes", type=int, default=2000)
    ap.add_argument("--sf-multipv", type=int, default=5)
    ap.add_argument("--sf-workers", type=int, default=1)
    ap.add_argument("--sf-policy-temp", type=float, default=0.25)
    ap.add_argument("--sf-policy-label-smooth", type=float, default=0.05)

    ap.add_argument("--opening-book-path", type=str, default=None, help="Path to opening book (.bin polyglot, .pgn, or .pgn.zip)")
    ap.add_argument("--opening-book-max-plies", type=int, default=4)
    ap.add_argument("--opening-book-max-games", type=int, default=200000)
    ap.add_argument("--opening-book-prob", type=float, default=1.0)
    ap.add_argument("--random-start-plies", type=int, default=0)

    pid_group = ap.add_mutually_exclusive_group()
    pid_group.add_argument("--sf-pid", dest="sf_pid_enabled", action="store_true", help="Enable adaptive SF PID")
    pid_group.add_argument("--no-sf-pid", dest="sf_pid_enabled", action="store_false", help="Disable adaptive SF PID")
    ap.set_defaults(sf_pid_enabled=True)

    ap.add_argument("--sf-pid-target-winrate", type=float, default=0.52)
    ap.add_argument("--sf-pid-ema-alpha", type=float, default=0.03)
    ap.add_argument("--sf-pid-deadzone", type=float, default=0.05)
    ap.add_argument("--sf-pid-rate-limit", type=float, default=0.10)
    ap.add_argument("--sf-pid-min-games-between-adjust", type=int, default=30)
    ap.add_argument("--sf-pid-kp", type=float, default=1.5)
    ap.add_argument("--sf-pid-ki", type=float, default=0.10)
    ap.add_argument("--sf-pid-kd", type=float, default=0.0)
    ap.add_argument("--sf-pid-integral-clamp", type=float, default=1.0)
    ap.add_argument("--sf-pid-min-nodes", type=int, default=250)
    ap.add_argument("--sf-pid-max-nodes", type=int, default=1000000)
    ap.add_argument("--games-per-iter", type=int, default=10)
    ap.add_argument("--selfplay-batch", type=int, default=10, help="Play games in mini-batches of this size to limit memory")
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--playout-cap-fraction", type=float, default=0.25)
    ap.add_argument("--fast-simulations", type=int, default=8)
    ap.add_argument("--fpu-reduction", type=float, default=1.2, help="FPU reduction for non-root MCTS nodes (LC0 default: 1.2)")
    ap.add_argument("--fpu-at-root", type=float, default=1.0, help="FPU reduction for root node (LC0 default: 1.0)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--temperature-drop-plies",
        type=int,
        default=0,
        help="Step schedule: drop temperature after N (full) moves (0 disables)",
    )
    ap.add_argument(
        "--temperature-after",
        type=float,
        default=0.0,
        help="Step schedule: temperature once move >= temperature_drop_plies",
    )
    ap.add_argument(
        "--temperature-decay-start-move",
        type=int,
        default=20,
        help="Linear schedule: move number to start decaying temperature (1-based)",
    )
    ap.add_argument(
        "--temperature-decay-moves",
        type=int,
        default=60,
        help="Linear schedule: number of moves to decay over (<=0 disables linear schedule)",
    )
    ap.add_argument(
        "--temperature-endgame",
        type=float,
        default=0.6,
        help="Linear schedule: endgame/floor temperature",
    )
    ap.add_argument("--max-plies", type=int, default=200)

    # Progressive simulation budget: start low (fast) and ramp up as training improves.
    mcts_prog = ap.add_mutually_exclusive_group()
    mcts_prog.add_argument("--progressive-mcts", dest="progressive_mcts", action="store_true")
    mcts_prog.add_argument("--no-progressive-mcts", dest="progressive_mcts", action="store_false")
    ap.set_defaults(progressive_mcts=True)

    ap.add_argument("--mcts-start-simulations", type=int, default=50)
    ap.add_argument("--mcts-ramp-steps", type=int, default=10_000)
    ap.add_argument("--mcts-ramp-exponent", type=float, default=2.0)

    ap.add_argument("--mcts-simulations", type=int, default=800)
    ap.add_argument("--mcts", type=str, default="puct", choices=["puct", "gumbel"])
    ap.add_argument("--replay-capacity", type=int, default=200000)
    ap.add_argument("--min-replay-size", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-volatility", type=float, default=0.05)
    ap.add_argument("--volatility-source", type=str, default="raw", choices=["raw", "search"],
                    help="Volatility target source: raw (network raw WDL) or search (search-adjusted WDL).")
    ap.add_argument("--feature-dropout-p", type=float, default=0.3)
    ap.add_argument("--work-dir", type=str, default="runs")
    ap.add_argument("--optimizer", type=str, default="nadamw", choices=["nadamw", "adamw"],
                    help="Optimizer: nadamw (spec default) or adamw")
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP (BF16 autocast on CUDA)")
    ap.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation micro-batches")
    ap.add_argument("--warmup-steps", type=int, default=1500, help="Linear LR warmup steps")
    ap.add_argument("--lr-eta-min", type=float, default=1e-5, help="Cosine schedule min LR")
    ap.add_argument("--lr-T0", type=int, default=5000, help="Cosine schedule T_0")
    ap.add_argument("--lr-T-mult", type=int, default=2, help="Cosine schedule T_mult")
    ap.add_argument("--grad-clip", type=float, default=10.0, help="Gradient clipping max norm")
    ap.add_argument("--use-compile", action="store_true", help="Enable torch.compile for training")
    ap.add_argument("--swa-start", type=int, default=0, help="Step to start SWA averaging (0=disabled)")
    ap.add_argument("--swa-freq", type=int, default=50, help="SWA update frequency in steps")
    ap.add_argument("--syzygy-path", type=str, default=None, help="Path to Syzygy tablebase directory")
    ap.add_argument("--syzygy-policy", action="store_true",
                    help="Also rescore policy targets with DTZ-optimal best move in TB positions")
    ap.add_argument(
        "--timeout-adjudication-threshold",
        dest="timeout_adjudication_threshold",
        type=float,
        default=0.90,
        help="Stockfish WDL confidence required to label max_plies timeouts as decisive",
    )

    # Puzzle evaluation (overspecialization canary)
    ap.add_argument("--puzzle-epd", type=str, default=None,
                    help="Path to EPD puzzle file for periodic evaluation (e.g. data/wac.epd)")
    ap.add_argument("--puzzle-interval", type=int, default=1,
                    help="Run puzzle eval every N iterations (0 to disable)")
    ap.add_argument("--puzzle-simulations", type=int, default=200,
                    help="MCTS simulations per puzzle position")

    ap.set_defaults(**cfg_defaults)
    args = ap.parse_args()

    if args.stockfish_path is None:
        raise SystemExit("--stockfish-path is required (or set stockfish.path in --config)")

    if str(args.mode) == "single":
        _run_single(args)
        return

    # Tune mode (default)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = {
        "seed": int(args.seed),
        "device": device,
        "model": str(args.model),
        "iterations": int(args.iterations),
        "replay_capacity": int(args.replay_capacity),
        "min_replay_size": int(args.min_replay_size),
        "stockfish_path": str(args.stockfish_path),
        "volatility_source": str(args.volatility_source),
        "search_volatility_source": bool(args.search_volatility_source),
        "sf_nodes": int(args.sf_nodes),
        "sf_multipv": int(args.sf_multipv),
        "sf_policy_temp": float(args.sf_policy_temp),
        "sf_policy_label_smooth": float(args.sf_policy_label_smooth),
        "timeout_adjudication_threshold": float(getattr(args, "timeout_adjudication_threshold", 0.90)),
        "sf_pid_enabled": bool(args.sf_pid_enabled),
        "sf_pid_target_winrate": float(args.sf_pid_target_winrate),
        "sf_pid_ema_alpha": float(args.sf_pid_ema_alpha),
        "sf_pid_deadzone": float(args.sf_pid_deadzone),
        "sf_pid_rate_limit": float(args.sf_pid_rate_limit),
        "sf_pid_min_games_between_adjust": int(args.sf_pid_min_games_between_adjust),
        "sf_pid_kp": float(args.sf_pid_kp),
        "sf_pid_ki": float(args.sf_pid_ki),
        "sf_pid_kd": float(args.sf_pid_kd),
        "sf_pid_integral_clamp": float(args.sf_pid_integral_clamp),
        "sf_pid_min_nodes": int(args.sf_pid_min_nodes),
        "sf_pid_max_nodes": int(args.sf_pid_max_nodes),
        "opening_book_path": args.opening_book_path,
        "opening_book_max_plies": int(args.opening_book_max_plies),
        "opening_book_max_games": int(args.opening_book_max_games),
        "opening_book_prob": float(args.opening_book_prob),
        "random_start_plies": int(args.random_start_plies),
        "eval_games": int(args.eval_games),
        "eval_sf_nodes": int(args.eval_sf_nodes) if args.eval_sf_nodes is not None else int(args.sf_nodes),
        "eval_mcts_simulations": (
            int(args.eval_mcts_simulations)
            if args.eval_mcts_simulations is not None
            else (
                int(args.mcts_start_simulations) if bool(args.progressive_mcts) else int(args.mcts_simulations)
            )
        ),
        "holdout_fraction": float(args.holdout_fraction),
        "holdout_capacity": int(args.holdout_capacity),
        "test_steps": int(args.test_steps),
        "freeze_holdout_at": int(args.freeze_holdout_at),
        "search_feature_dropout_p": bool(args.search_feature_dropout_p),
        "search_w_volatility": bool(args.search_w_volatility),
        "reset_holdout_on_drift": bool(args.reset_holdout_on_drift),
        "drift_threshold": float(args.drift_threshold),
        "drift_sample_size": int(args.drift_sample_size),
        "sf_workers": int(args.sf_workers),
        "games_per_iter": int(args.games_per_iter),
        "selfplay_batch": int(args.selfplay_batch),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "feature_dropout_p": float(args.feature_dropout_p),
        "w_volatility": float(args.w_volatility),
        "max_plies": int(args.max_plies),
        "use_amp": not bool(args.no_amp),
        "accum_steps": int(args.accum_steps),
        "warmup_steps": int(args.warmup_steps),
        "lr_eta_min": float(args.lr_eta_min),
        "lr_T0": int(args.lr_T0),
        "lr_T_mult": int(args.lr_T_mult),
        "grad_clip": float(args.grad_clip),
        "use_compile": bool(args.use_compile),
        "optimizer": str(args.optimizer),
        "embed_dim": int(args.embed_dim),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "ffn_mult": int(args.ffn_mult),
        "use_smolgen": not bool(args.no_smolgen),
        "use_nla": bool(args.use_nla),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "mcts": str(args.mcts),
        "mcts_simulations": int(args.mcts_simulations),
        "progressive_mcts": bool(args.progressive_mcts),
        "mcts_start_simulations": int(args.mcts_start_simulations),
        "mcts_ramp_steps": int(args.mcts_ramp_steps),
        "mcts_ramp_exponent": float(args.mcts_ramp_exponent),
        "fast_simulations": int(args.fast_simulations),
        "playout_cap_fraction": float(args.playout_cap_fraction),
        "fpu_reduction": float(args.fpu_reduction),
        "fpu_at_root": float(args.fpu_at_root),
        "temperature": float(args.temperature),
        "temperature_drop_plies": int(args.temperature_drop_plies),
        "temperature_after": float(args.temperature_after),
        "temperature_decay_start_move": int(args.temperature_decay_start_move),
        "temperature_decay_moves": int(args.temperature_decay_moves),
        "temperature_endgame": float(args.temperature_endgame),
        "swa_start": int(args.swa_start),
        "swa_freq": int(args.swa_freq),
        "syzygy_path": args.syzygy_path,
        "syzygy_policy": bool(args.syzygy_policy),
        "puzzle_epd": args.puzzle_epd,
        "puzzle_interval": int(args.puzzle_interval),
        "puzzle_simulations": int(args.puzzle_simulations),
        "bootstrap_dir": getattr(args, "bootstrap_dir", None),
        "bootstrap_checkpoint": getattr(args, "bootstrap_checkpoint", None),
        "bootstrap_max_positions": int(getattr(args, "bootstrap_max_positions", 0)),
        "bootstrap_train_steps": int(getattr(args, "bootstrap_train_steps", 0)),
        "shared_shards_dir": getattr(args, "shared_shards_dir", None),
        "shuffle_buffer_size": int(getattr(args, "shuffle_buffer_size", 20_000)),
        "shard_size": int(getattr(args, "shard_size", 1000)),
        "replay_window_start": int(getattr(args, "replay_window_start", 100_000)),
        "replay_window_max": int(getattr(args, "replay_window_max", 1_000_000)),
        "replay_window_growth": int(getattr(args, "replay_window_growth", 10_000)),
        "gate_games": int(getattr(args, "gate_games", 0)),
        "gate_threshold": float(getattr(args, "gate_threshold", 0.50)),
        "gate_interval": int(getattr(args, "gate_interval", 1)),
        "gate_mcts_sims": int(getattr(args, "gate_mcts_sims", 1)),
        "w_sf_wdl": float(getattr(args, "w_sf_wdl", 1.0)),
        "sf_wdl_floor": float(getattr(args, "sf_wdl_floor", 0.1)),
        "sf_wdl_floor_at": float(getattr(args, "sf_wdl_floor_at", 0.1)),
        "work_dir": str(Path(args.work_dir) / "tune"),
        "tune_num_to_keep": int(args.tune_num_to_keep),
        "tune_keep_last_experiments": int(args.tune_keep_last_experiments),
        "max_concurrent_trials": int(args.max_concurrent_trials),
        "cpus_per_trial": int(args.cpus_per_trial),
        "gpus_per_trial": float(args.gpus_per_trial),
        "tune_scheduler": str(args.tune_scheduler),
        "pb2_perturbation_interval": int(args.pb2_perturbation_interval),
        "search_smolgen": bool(args.search_smolgen),
        "search_nla": bool(args.search_nla),
        "search_optimizer": bool(args.search_optimizer),
    }
    # Forward pb2_bounds_* keys from config to base dict for PB2 scheduler.
    for k, v in vars(args).items():
        if k.startswith("pb2_bounds_"):
            base[k] = v

    from chess_anti_engine.tune.harness import run_tune

    metric = str(args.tune_metric)
    mode = str(args.tune_mode) if args.tune_mode is not None else ("max" if metric.endswith("winrate") else "min")

    run_tune(
        base_config=base,
        work_dir=Path(args.work_dir),
        num_samples=int(args.num_samples),
        metric=metric,
        mode=mode,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
