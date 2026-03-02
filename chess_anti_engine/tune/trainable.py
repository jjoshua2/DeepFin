from __future__ import annotations

# Optional dependency module (Ray Tune). Kept import-light so the core package
# works without installing `.[tune]`.

from pathlib import Path
import json

import numpy as np
import torch

import math

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.replay import DiskReplayBuffer, ReplayBuffer
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.budget import progressive_mcts_simulations
from chess_anti_engine.stockfish import DifficultyPID, StockfishPool, StockfishUCI
from chess_anti_engine.train import Trainer


def _opponent_strength(
    *,
    random_move_prob: float,
    sf_nodes: int,
    skill_level: int,
    ema_winrate: float,
    min_nodes: int,
    max_nodes: int,
) -> float:
    """Composite metric capturing full difficulty progression.

    Returns a single scalar (higher = harder opponent = better model):
      Stage 1 (score   0-100): random_move_prob 1.0 → 0.0
      Stage 2 (score 100-200): sf_nodes min → max (log-scaled)
      Stage 3 (score 200-400): skill_level 0 → 20

    The PID controller holds winrate at ~0.53, so the only differentiator
    between trials is how hard an opponent they can maintain that against.
    """
    rand_prob = float(random_move_prob)
    nodes = int(sf_nodes)
    skill = int(skill_level)

    min_nodes = int(min_nodes)
    max_nodes = int(max_nodes)

    # Stage 1: random_move_prob 1.0→0.0 maps to score 0→100
    stage1 = (1.0 - rand_prob) * 100.0

    # Stage 2: sf_nodes on log scale, maps to score 100→200
    if min_nodes < max_nodes and nodes > 0:
        log_frac = (math.log(max(nodes, min_nodes)) - math.log(max(1, min_nodes))) / (
            math.log(max(1, max_nodes)) - math.log(max(1, min_nodes))
        )
        log_frac = max(0.0, min(1.0, log_frac))
    else:
        log_frac = 0.0
    stage2 = log_frac * 100.0

    # Stage 3: skill_level 0→20 maps to score 200→400
    stage3 = (float(skill) / 20.0) * 200.0

    # Winrate tiebreaker: when all trials are at the same difficulty
    # (e.g. everyone stuck at random_move_prob=1.0), the winrate term
    # differentiates who is closest to breaking through the PID threshold.
    # Range 0-10, negligible once opponent_strength starts climbing (0-400).
    winrate_bonus = float(ema_winrate) * 10.0

    return stage1 + stage2 + stage3 + winrate_bonus



def _gate_check(
    model: torch.nn.Module,
    *,
    device: str,
    rng: np.random.Generator,
    sf: object,
    gate_games: int,
    opponent_random_move_prob: float,
    config: dict,
) -> tuple[float, int, int, int]:
    """Play gate games to measure winrate. Returns (winrate, W, D, L)."""
    _gate_samples, gate_stats = play_batch(
        model,
        device=device,
        rng=rng,
        stockfish=sf,
        games=gate_games,
        opponent_random_move_prob=opponent_random_move_prob,
        temperature=0.3,  # Low temperature for gating (exploit, don't explore)
        temperature_drop_plies=0,
        temperature_after=0.0,
        temperature_decay_start_move=10,
        temperature_decay_moves=30,
        temperature_endgame=0.1,
        max_plies=int(config.get("max_plies", 120)),
        mcts_simulations=int(config.get("gate_mcts_sims", 1)),  # 1 = raw policy + value
        mcts_type=str(config.get("mcts", "puct")),
        playout_cap_fraction=1.0,
        fast_simulations=0,
        sf_policy_temp=float(config.get("sf_policy_temp", 0.25)),
        sf_policy_label_smooth=float(config.get("sf_policy_label_smooth", 0.05)),
        timeout_adjudication_threshold=float(config.get("timeout_adjudication_threshold", 0.90)),
        volatility_source=str(config.get("volatility_source", "raw")),
        opening_book_path=config.get("opening_book_path"),
        opening_book_max_plies=int(config.get("opening_book_max_plies", 4)),
        opening_book_max_games=int(config.get("opening_book_max_games", 200_000)),
        opening_book_prob=float(config.get("opening_book_prob", 1.0)),
        random_start_plies=int(config.get("random_start_plies", 0)),
        fpu_reduction=float(config.get("fpu_reduction", 1.2)),
        fpu_at_root=float(config.get("fpu_at_root", 1.0)),
    )
    w, d, l = gate_stats.w, gate_stats.d, gate_stats.l
    total = max(1, w + d + l)
    winrate = (w + 0.5 * d) / total
    return winrate, w, d, l


def _prune_trial_checkpoints(*, trial_dir: Path, keep_last: int) -> None:
    """Best-effort deletion of old checkpoint_* dirs inside a Tune trial.

    This complements Ray's `CheckpointConfig(num_to_keep=...)`.
    In particular, it helps when resuming an older experiment whose RunConfig did
    not have checkpoint retention enabled.
    """

    import shutil

    keep_last = int(keep_last)
    if keep_last <= 0:
        return

    ckpts = sorted(
        [p for p in trial_dir.glob("checkpoint_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if len(ckpts) <= keep_last:
        return

    for p in ckpts[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)


def train_trial(config: dict):
    """Ray Tune trainable.

    Reports metrics per outer-loop iteration. Supports Ray AIR checkpoint restore.
    """

    from ray.air import session
    from ray.train import Checkpoint

    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)

    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(
        ModelConfig(
            kind=str(config.get("model", "transformer")),
            embed_dim=int(config.get("embed_dim", 256)),
            num_layers=int(config.get("num_layers", 6)),
            num_heads=int(config.get("num_heads", 8)),
            ffn_mult=int(config.get("ffn_mult", 2)),
            use_smolgen=bool(config.get("use_smolgen", True)),
            use_nla=bool(config.get("use_nla", False)),
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        )
    )

    # Use Ray-provided trial directory for ALL per-trial state (checkpoints,
    # replay shards, gate state, best model, TensorBoard logs).
    # IMPORTANT: Do NOT use config["work_dir"] here — it points to the shared
    # runs/pbt2_small/ directory. Using it caused all 10 trials to write
    # checkpoints to the same directory, making PB2 unable to clone checkpoints
    # ("no checkpoint for trial X. Skip exploit.").
    trial_dir = Path(session.get_trial_dir())
    work_dir = trial_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # Gate match counter (so we can log gate scalars against match #, not iteration).
    gate_state_path = work_dir / "gate_state.json"
    gate_match_idx = 0
    if gate_state_path.exists():
        try:
            d = json.loads(gate_state_path.read_text(encoding="utf-8"))
            gate_match_idx = int(d.get("matches", 0))
        except Exception:
            gate_match_idx = 0

    # Best-model tracking (per trial)
    best_state_path = work_dir / "best.json"
    best_dir = work_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    if best_state_path.exists():
        try:
            d = json.loads(best_state_path.read_text(encoding="utf-8"))
            best_loss = float(d.get("best_loss", d.get("loss", best_loss)))
        except Exception:
            pass

    trainer = Trainer(
        model,
        device=device,
        lr=float(config.get("lr", 3e-4)),
        zclip_z_thresh=float(config.get("zclip_z_thresh", 2.5)),
        zclip_alpha=float(config.get("zclip_alpha", 0.97)),
        zclip_max_norm=float(config.get("zclip_max_norm", 1.0)),
        log_dir=work_dir / "tb",
        use_amp=bool(config.get("use_amp", True)),
        feature_dropout_p=float(config.get("feature_dropout_p", 0.3)),
        w_volatility=float(config.get("w_volatility", 0.05)),
        accum_steps=int(config.get("accum_steps", 1)),
        warmup_steps=int(config.get("warmup_steps", 1500)),
        lr_eta_min=float(config.get("lr_eta_min", 1e-5)),
        lr_T0=int(config.get("lr_T0", 5000)),
        lr_T_mult=int(config.get("lr_T_mult", 2)),
        use_compile=bool(config.get("use_compile", False)),
        optimizer=str(config.get("optimizer", "nadamw")),
        swa_start=int(config.get("swa_start", 0)),
        swa_freq=int(config.get("swa_freq", 50)),
        # Tunable loss weights (Ray Tune ablations)
        w_policy=float(config.get("w_policy", 1.0)),
        w_soft=float(config.get("w_soft", 0.5)),
        w_future=float(config.get("w_future", 0.15)),
        w_wdl=float(config.get("w_wdl", 1.0)),
        w_sf_move=float(config.get("w_sf_move", 0.15)),
        w_sf_eval=float(config.get("w_sf_eval", 0.15)),
        w_categorical=float(config.get("w_categorical", 0.10)),
        w_sf_volatility=float(config.get("w_sf_volatility", config.get("w_volatility", 0.05))),
        w_moves_left=float(config.get("w_moves_left", 0.02)),
        w_sf_wdl=float(config.get("w_sf_wdl", 1.0)),
    )

    # Dynamic sf_wdl weight schedule: start at w_sf_wdl (default 1.0, equal to w_wdl)
    # and decline linearly as random_move_prob drops.  When random_move_prob reaches
    # sf_wdl_floor_at (default 0.1), the weight is sf_wdl_floor (default 0.1).
    # This bootstraps the value head from SF evaluations early on, then fades to
    # game-outcome WDL once the model is strong enough to generate meaningful games.
    sf_wdl_start = float(config.get("w_sf_wdl", 1.0))
    sf_wdl_floor = float(config.get("sf_wdl_floor", 0.1))
    sf_wdl_floor_at = float(config.get("sf_wdl_floor_at", 0.1))  # random_move_prob at which we hit the floor

    # Restore from checkpoint if provided by Ray.
    # NOTE: we restore PID state later (after PID is constructed).
    restored_pid_state = None
    ckpt = session.get_checkpoint()
    if ckpt is not None:
        ckpt_dir = Path(ckpt.to_directory())
        maybe = ckpt_dir / "trainer.pt"
        if maybe.exists():
            trainer.load(maybe)
        pid_path = ckpt_dir / "pid_state.json"
        if pid_path.exists():
            try:
                restored_pid_state = json.loads(pid_path.read_text(encoding="utf-8"))
            except Exception:
                restored_pid_state = None

    # Growing sliding window: start small, grow as the net matures.
    window_start = int(config.get("replay_window_start", 100_000))
    window_max = int(config.get("replay_window_max", int(config.get("replay_capacity", 1_000_000))))
    window_growth = int(config.get("replay_window_growth", 10_000))
    current_window = window_start

    shuffle_cap = int(config.get("shuffle_buffer_size", 20_000))
    shard_size = int(config.get("shard_size", 1000))
    replay_shard_dir = work_dir / "replay_shards"

    # Seed replay buffer with shared iter-0 data (played once from bootstrap net).
    # Only copy if this is a fresh trial (no existing shards in replay_shard_dir).
    shared_shards_dir = config.get("shared_shards_dir")
    if shared_shards_dir and not any(replay_shard_dir.glob("shard_*.npz")):
        src = Path(shared_shards_dir)
        if src.is_dir():
            replay_shard_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            copied = 0
            for sp in sorted(src.glob("shard_*.npz")):
                shutil.copy2(str(sp), str(replay_shard_dir / sp.name))
                copied += 1
            if copied:
                print(f"[trial] Copied {copied} shared iter-0 shards from {src}")

    buf = DiskReplayBuffer(
        current_window,
        shard_dir=replay_shard_dir,
        rng=rng,
        shuffle_cap=shuffle_cap,
        shard_size=shard_size,
    )

    # On resume, DiskReplayBuffer discovers existing shards on disk. If the on-disk
    # sample count already exceeds replay_window_start, keep the effective capacity
    # large enough to avoid pruning old shards just because we restarted.
    current_window = max(int(current_window), int(len(buf)))
    buf.capacity = int(current_window)
    holdout_buf = ReplayBuffer(int(config.get("holdout_capacity", 50_000)), rng=rng)
    holdout_frac = float(config.get("holdout_fraction", 0.02))

    # Load pre-trained bootstrap checkpoint (trained offline via scripts/train_bootstrap.py).
    # This gives the value head a working signal so first MCTS searches are better than random.
    # IMPORTANT: Only load MODEL WEIGHTS — do NOT restore optimizer/scheduler/step.
    # The bootstrap was trained for ~13k steps with its own LR schedule; carrying that
    # state into the trainable causes: (1) step=13323 skips warmup entirely,
    # (2) scheduler resumes mid-cosine-cycle with near-zero LR then spikes on restart,
    # (3) optimizer momentum buffers from bootstrap's data distribution cause wrong
    # gradient directions on selfplay data, (4) PB2's lr perturbation has no effect
    # because scheduler's base_lr is locked to bootstrap's lr (0.0003).
    bootstrap_ckpt = config.get("bootstrap_checkpoint")
    if bootstrap_ckpt and ckpt is None:
        # Only load bootstrap if Ray didn't restore a trial checkpoint (i.e. fresh start).
        bp = Path(bootstrap_ckpt)
        if bp.exists():
            print(f"[trial] Loading pre-trained bootstrap model weights: {bp}")
            ckpt_data = torch.load(str(bp), map_location=device)
            trainer.model.load_state_dict(ckpt_data["model"])
            # Deliberately skip: optimizer, scheduler, step — start fresh.
            del ckpt_data
        else:
            print(f"[trial] WARNING: bootstrap checkpoint not found: {bp}")

    # Net gating config
    gate_games = int(config.get("gate_games", 0))  # 0 = disabled
    gate_threshold = float(config.get("gate_threshold", 0.50))
    gate_interval = int(config.get("gate_interval", 1))  # gate every N iters

    # Holdout management: optionally freeze once it reaches a target size, and optionally
    # reset it if the training distribution drifts too far.
    freeze_holdout_at = int(config.get("freeze_holdout_at", 0))
    reset_holdout_on_drift = bool(config.get("reset_holdout_on_drift", False))
    drift_threshold = float(config.get("drift_threshold", 0.0))
    drift_sample_size = int(config.get("drift_sample_size", 256))
    holdout_frozen = False
    holdout_generation = 0

    sf_workers = int(config.get("sf_workers", 1))
    sf_multipv = int(config.get("sf_multipv", 1))
    if sf_workers > 1:
        sf = StockfishPool(
            path=str(config["stockfish_path"]),
            nodes=int(config.get("sf_nodes", 500)),
            num_workers=sf_workers,
            multipv=sf_multipv,
        )
    else:
        sf = StockfishUCI(
            str(config["stockfish_path"]),
            nodes=int(config.get("sf_nodes", 500)),
            multipv=sf_multipv,
        )

    eval_games = int(config.get("eval_games", 0))
    eval_sf_nodes = int(config.get("eval_sf_nodes", config.get("sf_nodes", 500)))
    eval_mcts_sims = int(config.get("eval_mcts_simulations", config.get("mcts_simulations", 50)))

    eval_sf = None
    if eval_games > 0:
        # For fixed-strength evaluation, use a dedicated engine instance with its own node limit.
        eval_sf = StockfishUCI(str(config["stockfish_path"]), nodes=eval_sf_nodes, multipv=1)

    pid = None
    if bool(config.get("sf_pid_enabled", True)):
        pid = DifficultyPID(
            initial_nodes=int(config.get("sf_nodes", 500)),
            target_winrate=float(config.get("sf_pid_target_winrate", 0.52)),
            ema_alpha=float(config.get("sf_pid_ema_alpha", 0.03)),
            deadzone=float(config.get("sf_pid_deadzone", 0.05)),
            rate_limit=float(config.get("sf_pid_rate_limit", 0.10)),
            min_games_between_adjust=int(config.get("sf_pid_min_games_between_adjust", 30)),
            kp=float(config.get("sf_pid_kp", 1.5)),
            ki=float(config.get("sf_pid_ki", 0.10)),
            kd=float(config.get("sf_pid_kd", 0.0)),
            integral_clamp=float(config.get("sf_pid_integral_clamp", 1.0)),
            min_nodes=int(config.get("sf_pid_min_nodes", 250)),
            max_nodes=int(config.get("sf_pid_max_nodes", 1000000)),
            initial_random_move_prob=float(config.get("sf_pid_random_move_prob_start", 1.0)),
            random_move_prob_min=float(config.get("sf_pid_random_move_prob_min", 0.0)),
            random_move_prob_max=float(config.get("sf_pid_random_move_prob_max", 1.0)),
            random_move_stage_end=float(config.get("sf_pid_random_move_stage_end", 0.5)),
            max_rand_step=float(config.get("sf_pid_max_rand_step", 0.01)),
        )
        if restored_pid_state is not None:
            try:
                pid.load_state_dict(restored_pid_state)
            except Exception:
                pass

    # Optional puzzle evaluation suite.
    puzzle_suite = None
    puzzle_interval = int(config.get("puzzle_interval", 1))
    puzzle_sims = int(config.get("puzzle_simulations", 200))
    _puzzle_epd = config.get("puzzle_epd")
    if _puzzle_epd and puzzle_interval > 0:
        from chess_anti_engine.eval import load_epd
        try:
            puzzle_suite = load_epd(_puzzle_epd)
        except FileNotFoundError:
            puzzle_suite = None

    try:
        iterations = int(config.get("iterations", 10))
        for it in range(iterations):
            # Difficulty knobs used for this iteration's selfplay (kept fixed across
            # selfplay chunks). PID is updated once per iteration AFTER training so
            # changes align to net updates rather than chunk noise.
            current_rand = float(pid.random_move_prob) if pid is not None else float(config.get("sf_pid_random_move_prob_start", 0.0))
            sf_nodes_used = int(getattr(sf, "nodes", 0) or 0)
            skill_level_used = int(getattr(pid, "skill_level", 0) or 0) if pid is not None else 0

            base_sims = int(config.get("mcts_simulations", 50))
            sims = base_sims
            if bool(config.get("progressive_mcts", True)):
                sims = progressive_mcts_simulations(
                    int(getattr(trainer, "step", 0)),
                    start=int(config.get("mcts_start_simulations", 50)),
                    max_sims=base_sims,
                    ramp_steps=int(config.get("mcts_ramp_steps", 10_000)),
                    exponent=float(config.get("mcts_ramp_exponent", 2.0)),
                )

            # Skip selfplay on iter 0 if shared shards were pre-loaded (the data
            # is already in the replay buffer — no need to play redundant games).
            skip_selfplay = (it == 0 and len(buf) > 0 and shared_shards_dir is not None)

            # Play games in mini-batches to keep memory low (each mini-batch
            # frees its MCTS trees / board objects before the next starts).
            total_games = int(config.get("games_per_iter", 4))
            selfplay_batch = int(config.get("selfplay_batch", 10))
            games_remaining = 0 if skip_selfplay else total_games

            # Accumulators for stats across mini-batches.
            all_train_samples: list = []
            total_w = total_d = total_l = 0
            total_positions = 0
            total_sf_d6 = 0.0
            total_sf_d6_n = 0
            last_stats = None

            if skip_selfplay:
                total_positions = len(buf)
                print(f"[trial] Skipping selfplay for iter 0 — using {total_positions} pre-loaded positions")

            selfplay_kwargs = dict(
                device=device,
                rng=rng,
                stockfish=sf,
                opponent_random_move_prob=current_rand,
                temperature=float(config.get("temperature", 1.0)),
                temperature_drop_plies=int(config.get("temperature_drop_plies", 0)),
                temperature_after=float(config.get("temperature_after", 0.0)),
                temperature_decay_start_move=int(config.get("temperature_decay_start_move", 20)),
                temperature_decay_moves=int(config.get("temperature_decay_moves", 60)),
                temperature_endgame=float(config.get("temperature_endgame", 0.6)),
                max_plies=int(config.get("max_plies", 120)),
                mcts_simulations=int(sims),
                mcts_type=str(config.get("mcts", "puct")),
                playout_cap_fraction=float(config.get("playout_cap_fraction", 0.25)),
                fast_simulations=int(config.get("fast_simulations", 8)),
                sf_policy_temp=float(config.get("sf_policy_temp", 0.25)),
                sf_policy_label_smooth=float(config.get("sf_policy_label_smooth", 0.05)),
                timeout_adjudication_threshold=float(config.get("timeout_adjudication_threshold", 0.90)),
                volatility_source=str(config.get("volatility_source", "raw")),
                opening_book_path=config.get("opening_book_path"),
                opening_book_max_plies=int(config.get("opening_book_max_plies", 4)),
                opening_book_max_games=int(config.get("opening_book_max_games", 200_000)),
                opening_book_prob=float(config.get("opening_book_prob", 1.0)),
                random_start_plies=int(config.get("random_start_plies", 0)),
                syzygy_path=config.get("syzygy_path"),
                syzygy_policy=bool(config.get("syzygy_policy", False)),
                diff_focus_enabled=bool(config.get("diff_focus_enabled", True)),
                diff_focus_q_weight=float(config.get("diff_focus_q_weight", 6.0)),
                diff_focus_pol_scale=float(config.get("diff_focus_pol_scale", 3.5)),
                diff_focus_slope=float(config.get("diff_focus_slope", 3.0)),
                diff_focus_min=float(config.get("diff_focus_min", 0.025)),
                categorical_bins=int(config.get("categorical_bins", 32)),
                hlgauss_sigma=float(config.get("hlgauss_sigma", 0.04)),
                fpu_reduction=float(config.get("fpu_reduction", 1.2)),
                fpu_at_root=float(config.get("fpu_at_root", 1.0)),
                soft_policy_temp=float(config.get("soft_policy_temp", 2.0)),
            )

            while games_remaining > 0:
                chunk = min(selfplay_batch, games_remaining)
                samples, stats = play_batch(trainer.model, games=chunk, **selfplay_kwargs)
                games_remaining -= chunk
                last_stats = stats

                # Accumulate stats.
                total_w += stats.w
                total_d += stats.d
                total_l += stats.l
                total_positions += stats.positions
                total_sf_d6 += float(getattr(stats, "sf_eval_delta6", 0.0)) * int(getattr(stats, "sf_eval_delta6_n", 0))
                total_sf_d6_n += int(getattr(stats, "sf_eval_delta6_n", 0))

                # Split into train vs holdout and flush to disk immediately.
                for s in samples:
                    if holdout_frac > 0.0 and (not holdout_frozen) and (rng.random() < holdout_frac):
                        holdout_buf.add_many([s])
                    else:
                        all_train_samples.append(s)
                buf.add_many(all_train_samples)
                all_train_samples = []
                del samples  # Free memory from this mini-batch.

            # Aggregate stats for reporting.
            stats = last_stats  # Use last batch's stats for PID/misc fields.

            # Growing window: expand buffer capacity each iteration.
            if current_window < window_max:
                current_window = min(current_window + window_growth, window_max)
                if buf.capacity < current_window:
                    buf.capacity = current_window

            train_samples = []  # Already flushed to buf above.

            if (not holdout_frozen) and freeze_holdout_at > 0 and len(holdout_buf) >= freeze_holdout_at:
                holdout_frozen = True

            # Drift estimates (cheap heuristics for monitoring when data distribution moves).
            drift_input_l2 = None
            drift_wdl_js = None
            drift_policy_entropy_diff = None
            drift_policy_entropy_train = None
            drift_policy_entropy_holdout = None

            if len(buf) >= drift_sample_size and len(holdout_buf) >= drift_sample_size:
                train_batch = buf.sample_batch(drift_sample_size)
                hold_batch = holdout_buf.sample_batch(drift_sample_size)

                # (1) Input drift: L2 distance between mean input plane tensors.
                train_x = np.stack([s.x for s in train_batch], axis=0).astype(np.float32, copy=False)
                hold_x = np.stack([s.x for s in hold_batch], axis=0).astype(np.float32, copy=False)
                drift_input_l2 = float(np.linalg.norm(train_x.mean(axis=0) - hold_x.mean(axis=0)))

                # (2) Target drift: WDL label distribution (JS divergence).
                def _wdl_hist(ss: list) -> np.ndarray:
                    h = np.zeros((3,), dtype=np.float64)
                    for s in ss:
                        t = int(getattr(s, "wdl_target", 1))
                        if 0 <= t <= 2:
                            h[t] += 1.0
                    h /= max(1.0, float(h.sum()))
                    return h

                p = _wdl_hist(train_batch)
                q = _wdl_hist(hold_batch)
                m = 0.5 * (p + q)
                eps = 1e-12
                drift_wdl_js = float(
                    0.5 * np.sum(p * (np.log(p + eps) - np.log(m + eps)))
                    + 0.5 * np.sum(q * (np.log(q + eps) - np.log(m + eps)))
                )

                # (3) Policy drift: mean entropy of stored policy targets.
                def _mean_entropy(ss: list) -> float:
                    ent = 0.0
                    n = 0
                    for s in ss:
                        pt = getattr(s, "policy_target", None)
                        if pt is None:
                            continue
                        p = np.asarray(pt, dtype=np.float64)
                        ps = float(p.sum())
                        if ps <= 0:
                            continue
                        p = p / ps
                        ent += float(-np.sum(p * np.log(p + eps)))
                        n += 1
                    return float(ent / max(1, n))

                drift_policy_entropy_train = _mean_entropy(train_batch)
                drift_policy_entropy_holdout = _mean_entropy(hold_batch)
                drift_policy_entropy_diff = float(drift_policy_entropy_train - drift_policy_entropy_holdout)

                # Optional holdout reset based on input drift threshold.
                if (
                    reset_holdout_on_drift
                    and (drift_threshold > 0.0)
                    and (drift_input_l2 is not None)
                    and (drift_input_l2 > drift_threshold)
                ):
                    holdout_buf.clear()
                    holdout_frozen = False
                    holdout_generation += 1

            # Re-read loss weights from config each iteration so PB2 perturbations
            # take effect immediately (PB2 mutates the config dict in-place).
            for wk in ("w_soft", "w_future", "w_sf_move", "w_sf_eval",
                        "w_categorical", "w_volatility", "w_sf_wdl"):
                if wk in config:
                    setattr(trainer, wk, float(config[wk]))

            # Also update sf_wdl_start for the dynamic schedule below.
            sf_wdl_start = float(config.get("w_sf_wdl", sf_wdl_start))

            # Dynamic sf_wdl weight: interpolate between sf_wdl_start and sf_wdl_floor
            # based on how far random_move_prob has dropped from 1.0.
            # At random_move_prob=1.0: full SF bootstrapping (sf_wdl_start).
            # At random_move_prob=sf_wdl_floor_at: SF weight reaches sf_wdl_floor.
            if sf_wdl_start > 0:
                rp = current_rand  # 1.0 → 0.0 as model improves
                # Linear interp: rp=1.0 → sf_wdl_start, rp=sf_wdl_floor_at → sf_wdl_floor
                if rp >= 1.0:
                    cur_sf_wdl = sf_wdl_start
                elif rp <= sf_wdl_floor_at:
                    cur_sf_wdl = sf_wdl_floor
                else:
                    t = (rp - sf_wdl_floor_at) / (1.0 - sf_wdl_floor_at)
                    cur_sf_wdl = sf_wdl_floor + t * (sf_wdl_start - sf_wdl_floor)
                trainer.w_sf_wdl = cur_sf_wdl

            batch_size = int(config.get("batch_size", 128))
            skip_train = len(buf) < batch_size
            if skip_train:
                metrics = None
                gate_passed = True
            else:
                # Training steps: new positions / batch_size (each new sample seen ~1x),
                # capped by train_steps config to prevent overfitting on large data loads
                # (e.g., iter 0 loads 99k shared shards → 386 uncapped steps would overfit).
                max_steps = int(config.get("train_steps", 25))
                steps = min(max(1, total_positions // batch_size), max_steps)

                # Save model state for potential rollback (net gating).
                gate_passed = True
                pre_train_state = None
                if gate_games > 0 and (it % gate_interval == 0):
                    pre_train_state = {
                        k: v.clone() for k, v in trainer.model.state_dict().items()
                    }

                metrics = trainer.train_steps(
                    buf,
                    batch_size=batch_size,
                    steps=steps,
                )

                # Net gating: play gate_games and reject if winrate < threshold.
                if pre_train_state is not None and gate_games > 0:
                    gate_wr, gate_w, gate_d, gate_l = _gate_check(
                        trainer.model,
                        device=device,
                        rng=rng,
                        sf=sf,
                        gate_games=gate_games,
                        opponent_random_move_prob=current_rand,
                        config=config,
                    )

                    # Gate match index is separate from iteration because gates are rare.
                    gate_match_idx += 1
                    try:
                        gate_state_path.write_text(
                            json.dumps(
                                {"matches": int(gate_match_idx)},
                                indent=2,
                                sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

                    # TensorBoard logging for gating (x-axis = gate match #).
                    # Useful when gate checks happen infrequently.
                    try:
                        trainer.writer.add_scalar("gate/winrate", float(gate_wr), int(gate_match_idx))
                        trainer.writer.add_scalar("gate/win", float(gate_w), int(gate_match_idx))
                        trainer.writer.add_scalar("gate/draw", float(gate_d), int(gate_match_idx))
                        trainer.writer.add_scalar("gate/loss", float(gate_l), int(gate_match_idx))
                        trainer.writer.add_scalar("gate/passed", float(1.0 if gate_wr >= gate_threshold else 0.0), int(gate_match_idx))
                    except Exception:
                        pass

                    if gate_wr < gate_threshold:
                        # Revert model — training made it worse.
                        trainer.model.load_state_dict(pre_train_state)
                        gate_passed = False

            test_metrics = None
            if len(holdout_buf) >= int(config.get("batch_size", 128)):
                test_metrics = trainer.eval_steps(
                    holdout_buf,
                    batch_size=int(config.get("batch_size", 128)),
                    steps=int(config.get("test_steps", 10)),
                )

            eval_dict = {}
            if eval_games > 0 and eval_sf is not None:
                # Evaluation: fixed-strength games (no training data generated, only W/D/L).
                _eval_samples, eval_stats = play_batch(
                    trainer.model,
                    device=device,
                    rng=rng,
                    stockfish=eval_sf,
                    games=eval_games,
                    temperature=float(config.get("eval_temperature", 0.25)),
                    max_plies=int(config.get("eval_max_plies", config.get("max_plies", 120))),
                    mcts_simulations=eval_mcts_sims,
                    mcts_type=str(config.get("mcts", "puct")),
                    playout_cap_fraction=1.0,
                    fast_simulations=0,
                    sf_policy_temp=float(config.get("sf_policy_temp", 0.25)),
                    sf_policy_label_smooth=float(config.get("sf_policy_label_smooth", 0.05)),
                    timeout_adjudication_threshold=float(config.get("timeout_adjudication_threshold", 0.90)),
                    volatility_source=str(config.get("volatility_source", "raw")),
                )
                denom = float(max(1, eval_stats.w + eval_stats.d + eval_stats.l))
                eval_dict = {
                    "eval_win": eval_stats.w,
                    "eval_draw": eval_stats.d,
                    "eval_loss": eval_stats.l,
                    "eval_winrate": (float(eval_stats.w) + 0.5 * float(eval_stats.d)) / denom,
                }

            # Flush any remaining samples to disk before checkpointing.
            buf.flush()

            # Save a lightweight checkpoint (model+optimizer+step + PID state).
            ckpt_dir = work_dir / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "trainer.pt"
            trainer.save(ckpt_path)
            if pid is not None:
                try:
                    (ckpt_dir / "pid_state.json").write_text(
                        json.dumps(pid.state_dict(), sort_keys=True, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            checkpoint = Checkpoint.from_directory(str(ckpt_dir))

            test_dict = {
                "holdout_frozen": int(1 if holdout_frozen else 0),
                "holdout_generation": int(holdout_generation),
            }
            if drift_input_l2 is not None:
                test_dict["data_drift_input_l2"] = float(drift_input_l2)
            if drift_wdl_js is not None:
                test_dict["data_drift_wdl_js"] = float(drift_wdl_js)
            if drift_policy_entropy_diff is not None:
                test_dict["data_drift_policy_entropy_diff"] = float(drift_policy_entropy_diff)
            if drift_policy_entropy_train is not None:
                test_dict["data_drift_policy_entropy_train"] = float(drift_policy_entropy_train)
            if drift_policy_entropy_holdout is not None:
                test_dict["data_drift_policy_entropy_holdout"] = float(drift_policy_entropy_holdout)

            if test_metrics is not None:
                test_dict.update(
                    {
                        "test_size": len(holdout_buf),
                        "test_loss": test_metrics.loss,
                        "test_policy_loss": test_metrics.policy_loss,
                        "test_soft_policy_loss": test_metrics.soft_policy_loss,
                        "test_future_policy_loss": test_metrics.future_policy_loss,
                        "test_wdl_loss": test_metrics.wdl_loss,
                        "test_sf_move_loss": test_metrics.sf_move_loss,
                        "test_sf_move_acc": test_metrics.sf_move_acc,
                        "test_sf_eval_loss": test_metrics.sf_eval_loss,
                        "test_categorical_loss": test_metrics.categorical_loss,
                        "test_volatility_loss": test_metrics.volatility_loss,
                        "test_sf_volatility_loss": test_metrics.sf_volatility_loss,
                        "test_moves_left_loss": test_metrics.moves_left_loss,
                    }
                )

            # Best-model tracking: prefer holdout loss when available, skip if no training yet.
            cur_loss = float(test_metrics.loss) if test_metrics is not None else (float(metrics.loss) if metrics is not None else float("inf"))
            if cur_loss < best_loss - 1e-12:
                best_loss = cur_loss
                trainer.save(best_dir / "trainer.pt")
                trainer.export_swa(best_dir / "best_model.pt")
                best_state_path.write_text(
                    json.dumps(
                        {
                            "best_loss": float(best_loss),
                            "iter": int(it),
                            "trainer_step": int(getattr(trainer, "step", 0)),
                            "source": "test_loss" if test_metrics is not None else "train_loss",
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

            # Update PID ONCE per iteration (after training) so difficulty changes
            # line up with net updates rather than intra-iteration selfplay noise.
            pid_update = None
            pid_ema_wr = 0.0
            sf_nodes_next = int(sf_nodes_used)
            random_move_prob_next = float(current_rand)
            skill_level_next = int(skill_level_used)
            if pid is not None and (total_w + total_d + total_l) > 0:
                pid_update = pid.observe(wins=total_w, draws=total_d, losses=total_l)
                pid_ema_wr = float(pid_update.ema_winrate)
                sf_nodes_next = int(pid.nodes)
                random_move_prob_next = float(pid.random_move_prob)
                skill_level_next = int(pid.skill_level)
                if hasattr(sf, "set_nodes"):
                    sf.set_nodes(int(sf_nodes_next))
                else:
                    setattr(sf, "nodes", int(sf_nodes_next))

            opp_strength = _opponent_strength(
                random_move_prob=float(current_rand),
                sf_nodes=int(sf_nodes_used),
                skill_level=int(skill_level_used),
                ema_winrate=float(pid_ema_wr),
                min_nodes=int(getattr(pid, "min_nodes", 50)) if pid is not None else 50,
                max_nodes=int(getattr(pid, "max_nodes", 50000)) if pid is not None else 50000,
            )

            # Puzzle evaluation (overspecialization canary).
            puzzle_dict = {}
            if puzzle_suite is not None and puzzle_interval > 0 and (it % puzzle_interval == 0):
                from chess_anti_engine.eval import run_puzzle_eval
                pr = run_puzzle_eval(
                    trainer.model, puzzle_suite,
                    device=device, mcts_simulations=puzzle_sims, rng=rng,
                )
                puzzle_dict = {
                    "puzzle_accuracy": pr.accuracy,
                    "puzzle_correct": pr.correct,
                    "puzzle_total": pr.total,
                }

            session.report(
                {
                    "iter": it,
                    "replay": len(buf),
                    "test_replay": len(holdout_buf),
                    "positions_added": total_positions,
                    "win": total_w,
                    "draw": total_d,
                    "loss": total_l,
                    "sf_eval_delta6": float(total_sf_d6 / max(1, total_sf_d6_n)) if total_sf_d6_n > 0 else 0.0,
                    "sf_eval_delta6_n": total_sf_d6_n,
                    "sf_nodes": int(sf_nodes_used),
                    "sf_nodes_next": int(sf_nodes_next),
                    "pid_ema_winrate": float(pid_ema_wr),
                    "random_move_prob": float(current_rand),
                    "random_move_prob_next": float(random_move_prob_next),
                    "skill_level": int(skill_level_used),
                    "skill_level_next": int(skill_level_next),
                    "opponent_strength": float(opp_strength),
                    "w_sf_wdl": float(trainer.w_sf_wdl),
                    "train_loss": float(metrics.loss) if metrics is not None else 999.0,
                    "best_loss": float(best_loss),
                    "policy_loss": float(metrics.policy_loss) if metrics is not None else 0.0,
                    "soft_policy_loss": float(metrics.soft_policy_loss) if metrics is not None else 0.0,
                    "future_policy_loss": float(metrics.future_policy_loss) if metrics is not None else 0.0,
                    "wdl_loss": float(metrics.wdl_loss) if metrics is not None else 0.0,
                    "sf_move_loss": float(metrics.sf_move_loss) if metrics is not None else 0.0,
                    "sf_move_acc": float(metrics.sf_move_acc) if metrics is not None else 0.0,
                    "sf_eval_loss": float(metrics.sf_eval_loss) if metrics is not None else 0.0,
                    "categorical_loss": float(metrics.categorical_loss) if metrics is not None else 0.0,
                    "volatility_loss": float(metrics.volatility_loss) if metrics is not None else 0.0,
                    "sf_volatility_loss": float(metrics.sf_volatility_loss) if metrics is not None else 0.0,
                    "moves_left_loss": float(metrics.moves_left_loss) if metrics is not None else 0.0,
                    "gate_passed": int(1 if gate_passed else 0),
                    **eval_dict,
                    **test_dict,
                    **puzzle_dict,
                },
                checkpoint=checkpoint,
            )

            # Best-effort: keep disk usage bounded even when resuming an older
            # experiment that did not have checkpoint retention configured.
            _prune_trial_checkpoints(
                trial_dir=trial_dir,
                keep_last=int(config.get("tune_num_to_keep", 2)),
            )
    finally:
        sf.close()
        if eval_sf is not None:
            eval_sf.close()
