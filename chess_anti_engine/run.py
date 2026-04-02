from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.config import RunConfig, StockfishConfig

from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
    reinit_volatility_head_parameters_,
    zero_policy_head_parameters_,
)
from chess_anti_engine.replay import DiskReplayBuffer, ReplayBuffer, balance_wdl
from chess_anti_engine.replay.shard import iter_shard_paths, samples_to_arrays
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.budget import progressive_mcts_simulations
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish import DifficultyPID, StockfishPool, StockfishUCI, pid_from_config
from chess_anti_engine.train import Trainer, trainer_kwargs_from_config
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS
from chess_anti_engine.tune.replay_exchange import _read_jsonl_rows
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def _trial_run_id_from_name(name: str) -> str | None:
    m = re.match(r"^train_trial_(?P<rid>[^_]+)_", name)
    if not m:
        return None
    return str(m.group("rid"))


def _latest_tune_run_id(tune_dir: Path) -> str | None:
    # Prefer latest PB2 policy log naming (pbt_policy_<runid>_<trial>.txt).
    policy_files = sorted(
        tune_dir.glob("pbt_policy_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in policy_files:
        m = re.match(r"^pbt_policy_(?P<rid>[^_]+)_\d+\.txt$", p.name)
        if m:
            return str(m.group("rid"))

    # Fallback: latest trial directory by mtime.
    best: tuple[float, str] | None = None
    for d in tune_dir.glob("train_trial_*"):
        if not d.is_dir():
            continue
        rid = _trial_run_id_from_name(d.name)
        if not rid:
            continue
        mt = float(d.stat().st_mtime)
        if best is None or mt > best[0]:
            best = (mt, rid)
    return best[1] if best is not None else None


def _run_salvage(args: argparse.Namespace) -> None:
    """Export top-N trial seeds (checkpoint + replay shards) for fresh restart."""
    tune_dir = Path(args.work_dir) / "tune"
    if not tune_dir.exists():
        raise SystemExit(f"Tune directory not found: {tune_dir}")

    run_id = str(args.salvage_source_run_id) if args.salvage_source_run_id else _latest_tune_run_id(tune_dir)
    if not run_id:
        raise SystemExit(f"Could not infer run id from {tune_dir}")

    metric_key = str(getattr(args, "salvage_metric", "opponent_strength"))
    replay_root_override = str(getattr(args, "tune_replay_root_override", "") or "").strip()
    top_n = int(getattr(args, "salvage_top_n", 0))
    if top_n <= 0:
        top_n = int(getattr(args, "num_samples", 1))
    top_n = max(1, top_n)

    trial_dirs = sorted([d for d in tune_dir.glob(f"train_trial_{run_id}_*") if d.is_dir()])
    scored: list[tuple[float, int, Path, dict]] = []
    for td in trial_dirs:
        rows = _read_jsonl_rows(td / "result.json")
        if not rows:
            continue

        best_any: tuple[float, int, dict] | None = None
        best_with_ckpt: tuple[float, int, dict] | None = None
        for row in rows:
            mv = row.get(metric_key)
            if not isinstance(mv, (int, float)):
                continue
            metric = float(mv)
            if not np.isfinite(metric):
                continue
            itv = row.get("training_iteration", row.get("iter", -1))
            it = int(itv) if isinstance(itv, (int, float)) else -1

            cand = (metric, it, row)
            if best_any is None or (metric, it) > (best_any[0], best_any[1]):
                best_any = cand

            ckname = row.get("checkpoint_dir_name")
            has_ckpt = False
            if isinstance(ckname, str) and ckname.strip():
                has_ckpt = (td / ckname / "trainer.pt").exists()
            if has_ckpt and (best_with_ckpt is None or (metric, it) > (best_with_ckpt[0], best_with_ckpt[1])):
                best_with_ckpt = cand

        picked = best_with_ckpt or best_any
        if picked is None:
            continue
        metric, it, row = picked
        scored.append((float(metric), int(it), td, row))

    if not scored:
        raise SystemExit(
            f"No trials with metric '{metric_key}' found under run_id={run_id} in {tune_dir}"
        )

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = scored[: min(top_n, len(scored))]

    out_dir_raw = getattr(args, "salvage_out_dir", None)
    if out_dir_raw:
        out_dir = Path(str(out_dir_raw))
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.work_dir) / "salvage" / f"{run_id}_{stamp}"
    seeds_dir = out_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    copy_replay = bool(getattr(args, "salvage_copy_replay", True))
    entries: list[dict] = []
    for slot, (metric, it, td, row) in enumerate(selected):
        seed_dir = seeds_dir / f"slot_{slot:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        row_ckpt_name = row.get("checkpoint_dir_name") if isinstance(row, dict) else None
        row_ckpt_dir = td / str(row_ckpt_name) if isinstance(row_ckpt_name, str) and row_ckpt_name.strip() else None
        ckpt_dir = row_ckpt_dir if (row_ckpt_dir is not None and (row_ckpt_dir / "trainer.pt").exists()) else (td / "ckpt")
        ckpt_source = "result_row_checkpoint" if ckpt_dir is row_ckpt_dir else "mutable_ckpt_fallback"
        if ckpt_source != "result_row_checkpoint":
            print(
                f"[salvage] WARNING: using fallback ckpt for {td.name} "
                f"(row checkpoint missing: {row_ckpt_name})"
            )
        for fn in ("trainer.pt", "pid_state.json", "trial_meta.json", "rng_state.json"):
            src = ckpt_dir / fn
            if src.exists():
                shutil.copy2(str(src), str(seed_dir / fn))

        # Align salvaged PID state with the same metrics row used for ranking.
        # This avoids stale difficulty carryover when ckpt writes and result rows
        # are out of phase (e.g. salvaging while trials are still running).
        pid_state_overrides: list[str] = []
        pid_seed_path = seed_dir / "pid_state.json"
        if pid_seed_path.exists() and isinstance(row, dict):
            try:
                pid_obj = json.loads(pid_seed_path.read_text(encoding="utf-8"))
            except Exception:
                pid_obj = None
            if isinstance(pid_obj, dict):
                def _set_pid(dst_key: str, src_keys: tuple[str, ...], *, as_int: bool = False) -> None:
                    for sk in src_keys:
                        v = row.get(sk)
                        if not isinstance(v, (int, float)):
                            continue
                        fv = float(v)
                        if not np.isfinite(fv):
                            continue
                        pid_obj[dst_key] = int(fv) if as_int else fv
                        pid_state_overrides.append(f"{dst_key}<-{sk}")
                        return

                _set_pid("random_move_prob", ("random_move_prob_next", "random_move_prob"))
                _set_pid("nodes", ("sf_nodes_next", "sf_nodes"), as_int=True)
                _set_pid("skill_level", ("skill_level_next", "skill_level"), as_int=True)
                _set_pid("ema_winrate", ("pid_ema_winrate",))

                if pid_state_overrides:
                    try:
                        pid_seed_path.write_text(
                            json.dumps(pid_obj, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

        copied_shards = 0
        if copy_replay:
            src_replay = td / "replay_shards"
            if (not src_replay.is_dir()) and replay_root_override:
                src_replay = Path(replay_root_override).expanduser() / td.name / "replay_shards"
            if src_replay.is_dir():
                dst_replay = seed_dir / "replay_shards"
                dst_replay.mkdir(parents=True, exist_ok=True)
                for sp in iter_shard_paths(src_replay):
                    dst = dst_replay / sp.name
                    if sp.is_dir():
                        shutil.copytree(str(sp), str(dst))
                    else:
                        shutil.copy2(str(sp), str(dst))
                    copied_shards += 1

        entries.append(
            {
                "slot": int(slot),
                "metric": float(metric),
                "training_iteration": int(it),
                "source_trial_dir": str(td.resolve()),
                "checkpoint_source": str(ckpt_source),
                "checkpoint_dir_name": str(row_ckpt_name) if isinstance(row_ckpt_name, str) else "",
                "seed_dir": str(seed_dir.relative_to(out_dir)),
                "copied_replay_shards": int(copied_shards),
                "pid_state_overrides": list(pid_state_overrides),
                "result_row": row,
            }
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_tune_dir": str(tune_dir.resolve()),
        "source_run_id": str(run_id),
        "metric": metric_key,
        "top_n": int(len(entries)),
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        f"[salvage] wrote {len(entries)} seeds from run_id={run_id} "
        f"metric={metric_key} to {out_dir}"
    )
    for e in entries:
        print(
            f"[salvage] slot={e['slot']:02d} metric={e['metric']:.3f} "
            f"iter={e['training_iteration']} shards={e['copied_replay_shards']} "
            f"src={Path(e['source_trial_dir']).name}"
        )


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
    cfg.selfplay.selfplay_fraction = float(getattr(args, "selfplay_fraction", 0.0))
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
    cfg.selfplay.opening_book_path_2 = getattr(args, "opening_book_path_2", None)
    cfg.selfplay.opening_book_max_plies_2 = int(getattr(args, "opening_book_max_plies_2", 16))
    cfg.selfplay.opening_book_max_games_2 = int(getattr(args, "opening_book_max_games_2", 200_000))
    cfg.selfplay.opening_book_mix_prob_2 = float(getattr(args, "opening_book_mix_prob_2", 0.0))
    cfg.selfplay.random_start_plies = int(args.random_start_plies)
    cfg.selfplay.sf_policy_temp = float(args.sf_policy_temp)
    cfg.selfplay.sf_policy_label_smooth = float(args.sf_policy_label_smooth)
    cfg.selfplay.timeout_adjudication_threshold = float(
        getattr(args, "timeout_adjudication_threshold", cfg.selfplay.timeout_adjudication_threshold)
    )
    cfg.stockfish = StockfishConfig(
        path=args.stockfish_path,
        nodes=int(args.sf_nodes),
        multipv=int(args.sf_multipv),
        hash_mb=int(getattr(args, "sf_hash_mb", 16)),
    )
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
            ffn_mult=float(args.ffn_mult),
            use_smolgen=not bool(args.no_smolgen),
            use_nla=bool(args.use_nla),
            use_gradient_checkpointing=bool(args.gradient_checkpointing),
        )
    )
    trainer_ctor = trainer_kwargs_from_config(
        vars(args) | {"device": cfg.train.device, "use_amp": not bool(args.no_amp)},
        log_dir=cfg.work_dir / "tb",
    )
    trainer = Trainer(model, **trainer_ctor)

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
        refresh_interval=int(getattr(args, "shuffle_refresh_interval", 5)),
        refresh_shards=int(getattr(args, "shuffle_refresh_shards", 3)),
        draw_cap_frac=float(getattr(args, "shuffle_draw_cap_frac", 0.90)),
        wl_max_ratio=float(getattr(args, "shuffle_wl_max_ratio", 1.5)),
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
            load_state_dict_tolerant(trainer.model, ckpt_data["model"], label="bootstrap")
            if bool(getattr(args, "bootstrap_zero_policy_heads", False)):
                zeroed = zero_policy_head_parameters_(trainer.model)
                if zeroed:
                    print(f"Zeroed bootstrap policy heads: {', '.join(zeroed)}")
            if bool(getattr(args, "bootstrap_reinit_volatility_heads", False)):
                reinit = reinit_volatility_head_parameters_(trainer.model)
                if reinit:
                    print(f"Reinitialized bootstrap volatility heads: {', '.join(reinit)}")
            del ckpt_data
        else:
            print(f"WARNING: bootstrap checkpoint not found: {bp}")

    if int(args.sf_workers) > 1:
        sf = StockfishPool(
            path=cfg.stockfish.path,
            nodes=cfg.stockfish.nodes,
            num_workers=int(args.sf_workers),
            multipv=int(getattr(cfg.stockfish, "multipv", 1)),
            hash_mb=int(getattr(cfg.stockfish, "hash_mb", 16)),
        )
    else:
        sf = StockfishUCI(
            cfg.stockfish.path,
            nodes=cfg.stockfish.nodes,
            multipv=int(getattr(cfg.stockfish, "multipv", 1)),
            hash_mb=int(getattr(cfg.stockfish, "hash_mb", 16)),
        )

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
        pid = pid_from_config(vars(args))
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
            total_positions = 0
            total_w, total_d, total_l = 0, 0, 0
            total_sf_d6 = 0.0
            current_rand = float(pid.random_move_prob) if pid is not None else 0.0

            opponent = OpponentConfig(
                random_move_prob=current_rand,
                topk_stage_end=float(getattr(args, "sf_pid_topk_stage_end", getattr(args, "sf_pid_random_move_stage_end", 0.5))),
                topk_min=int(getattr(args, "sf_pid_topk_min", 1)),
                suboptimal_wdl_regret_max=float(getattr(args, "sf_pid_suboptimal_wdl_regret_max", -1.0)),
                suboptimal_wdl_regret_min=float(getattr(args, "sf_pid_suboptimal_wdl_regret_min", -1.0)),
                random_move_prob_start=float(getattr(args, "sf_pid_random_move_prob_start", 1.0)),
                random_move_prob_min=float(getattr(args, "sf_pid_random_move_prob_min", 0.0)),
            )
            temp_cfg = TemperatureConfig(
                temperature=cfg.selfplay.temperature,
                drop_plies=int(cfg.selfplay.temperature_drop_plies),
                after=float(cfg.selfplay.temperature_after),
                decay_start_move=int(cfg.selfplay.temperature_decay_start_move),
                decay_moves=int(cfg.selfplay.temperature_decay_moves),
                endgame=float(cfg.selfplay.temperature_endgame),
            )
            search_cfg = SearchConfig(
                simulations=int(sims),
                mcts_type=str(args.mcts),
                playout_cap_fraction=float(args.playout_cap_fraction),
                fast_simulations=int(args.fast_simulations),
                fpu_reduction=float(getattr(args, "fpu_reduction", 1.2)),
                fpu_at_root=float(getattr(args, "fpu_at_root", 1.0)),
            )
            opening_cfg = OpeningConfig(
                opening_book_path=cfg.selfplay.opening_book_path,
                opening_book_max_plies=int(cfg.selfplay.opening_book_max_plies),
                opening_book_max_games=int(cfg.selfplay.opening_book_max_games),
                opening_book_prob=float(cfg.selfplay.opening_book_prob),
                opening_book_path_2=getattr(cfg.selfplay, "opening_book_path_2", None),
                opening_book_max_plies_2=int(getattr(cfg.selfplay, "opening_book_max_plies_2", 16)),
                opening_book_max_games_2=int(getattr(cfg.selfplay, "opening_book_max_games_2", 200_000)),
                opening_book_mix_prob_2=float(getattr(cfg.selfplay, "opening_book_mix_prob_2", 0.0)),
                random_start_plies=int(cfg.selfplay.random_start_plies),
            )
            diff_focus_cfg = DiffFocusConfig(
                enabled=bool(getattr(args, "diff_focus_enabled", True)),
                q_weight=float(getattr(args, "diff_focus_q_weight", 6.0)),
                pol_scale=float(getattr(args, "diff_focus_pol_scale", 3.5)),
                slope=float(getattr(args, "diff_focus_slope", 3.0)),
                min_keep=float(getattr(args, "diff_focus_min", 0.025)),
            )
            game_cfg = GameConfig(
                max_plies=cfg.selfplay.max_plies,
                selfplay_fraction=float(getattr(cfg.selfplay, "selfplay_fraction", 0.0)),
                sf_policy_temp=float(cfg.selfplay.sf_policy_temp),
                sf_policy_label_smooth=float(cfg.selfplay.sf_policy_label_smooth),
                timeout_adjudication_threshold=float(cfg.selfplay.timeout_adjudication_threshold),
                volatility_source=str(getattr(args, "volatility_source", "raw")),
                syzygy_path=args.syzygy_path,
                syzygy_policy=bool(args.syzygy_policy),
                categorical_bins=int(getattr(args, "categorical_bins", DEFAULT_CATEGORICAL_BINS)),
                hlgauss_sigma=float(getattr(args, "hlgauss_sigma", 0.04)),
            )
            samples, sp_stats = play_batch(
                trainer.model,
                device=cfg.train.device,
                rng=rng,
                stockfish=sf,
                games=selfplay_batch,
                target_games=cfg.selfplay.games_per_iter,
                opponent=opponent,
                temp=temp_cfg,
                search=search_cfg,
                opening=opening_cfg,
                diff_focus=diff_focus_cfg,
                game=game_cfg,
            )
            total_positions += sp_stats.positions
            total_w += sp_stats.w
            total_d += sp_stats.d
            total_l += sp_stats.l
            total_sf_d6 += sp_stats.sf_eval_delta6 * sp_stats.games
            if samples:
                buf.add_many_arrays(samples_to_arrays(samples))
            del samples

            buf.flush()

            if len(buf) == 0:
                print(
                    f"iter={it} step={getattr(trainer, 'step', 0)} sims={int(sims)} replay=0 pos_added=0 "
                    "skipping train step because selfplay produced no policy samples"
                )
                continue

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
                upd = pid.observe(wins=total_w, draws=total_d, losses=total_l, force=True)
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


# Keys that are internal to the harness or need special handling -- NOT passed
# through to Ray Tune per-trial config.
_TUNE_CONFIG_DENYLIST = frozenset({
    "config",           # YAML path, not a tunable
    "resume",           # harness control
    "mode",             # harness control
    "num_samples",      # tune.run parameter, not per-trial config
    "tune_metric",      # harness-level metric selection
    "tune_mode",        # harness-level mode (min/max)
    # Salvage-mode CLI args (only used by _run_salvage, not per-trial config).
    "salvage_source_run_id",
    "salvage_top_n",
    "salvage_out_dir",
    "salvage_metric",
    "salvage_copy_replay",
    # search_optimizer_choices needs list conversion, handled explicitly below.
    "search_optimizer_choices",
})


def _build_tune_config_dict(args: argparse.Namespace) -> dict:
    """Convert parsed CLI args into the flat config dict consumed by Ray Tune.

    Uses ``vars(args)`` as the base so any new argparse key is automatically
    forwarded.  A small denylist excludes harness-internal keys that should not
    be per-trial config.
    """
    base = {k: v for k, v in vars(args).items() if k not in _TUNE_CONFIG_DENYLIST}

    # Derived values that differ from the raw argparse value.
    base["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base["use_amp"] = not bool(args.no_amp)
    base["use_smolgen"] = not bool(args.no_smolgen)
    base["work_dir"] = str(Path(args.work_dir) / "tune")

    # eval_mcts_simulations: default to start sims if progressive, else base sims.
    if args.eval_mcts_simulations is None:
        base["eval_mcts_simulations"] = (
            int(args.mcts_start_simulations) if bool(args.progressive_mcts) else int(args.mcts_simulations)
        )

    # eval_sf_nodes: default to sf_nodes if not explicitly set.
    if args.eval_sf_nodes is None:
        base["eval_sf_nodes"] = int(args.sf_nodes)

    # search_optimizer_choices: convert to list.
    choices = args.search_optimizer_choices
    base["search_optimizer_choices"] = list(choices) if choices else None

    return base


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
    ap.add_argument("--mode", type=str, default="tune", choices=["tune", "single", "salvage"])
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous Ray Tune run (restore errored and unfinished trials)",
    )
    ap.add_argument("--salvage-source-run-id", type=str, default=None,
                    help="Salvage mode: source Tune run id (e.g. 7773d). Default: auto-detect latest.")
    ap.add_argument("--salvage-top-n", type=int, default=0,
                    help="Salvage mode: number of top trials to export (<=0 uses num_samples).")
    ap.add_argument("--salvage-out-dir", type=str, default=None,
                    help="Salvage mode: output directory for exported seeds (default under work_dir/salvage/).")
    ap.add_argument("--salvage-metric", type=str, default="opponent_strength",
                    help="Salvage mode: metric key to rank trials by.")
    copy_group = ap.add_mutually_exclusive_group()
    copy_group.add_argument("--salvage-copy-replay", dest="salvage_copy_replay", action="store_true",
                            help="Salvage mode: copy replay shard windows into exported seeds (default on).")
    copy_group.add_argument("--no-salvage-copy-replay", dest="salvage_copy_replay", action="store_false",
                            help="Salvage mode: export checkpoints only (no replay shards).")
    ap.set_defaults(salvage_copy_replay=True)

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
        "--distributed-workers-per-trial",
        type=int,
        default=0,
        help="If >0, Tune trials launch this many real worker subprocesses against the central server instead of local in-process selfplay.",
    )
    ap.add_argument(
        "--distributed-worker-sf-workers",
        type=int,
        default=1,
        help="Stockfish subprocesses per distributed worker process.",
    )
    ap.add_argument(
        "--distributed-worker-poll-seconds",
        type=float,
        default=1.0,
        help="Manifest polling interval for distributed Tune workers.",
    )
    ap.add_argument(
        "--distributed-worker-device",
        type=str,
        default=None,
        help="Device for distributed Tune workers (defaults to the trial device).",
    )
    ap.add_argument(
        "--distributed-worker-use-compile",
        action="store_true",
        help="Enable torch.compile for distributed worker inference/search models on CUDA.",
    )
    ap.add_argument(
        "--distributed-worker-auto-tune",
        action="store_true",
        help="Enable worker-side games_per_batch auto-tuning for distributed Tune workers.",
    )
    ap.add_argument(
        "--distributed-worker-target-batch-seconds",
        type=float,
        default=30.0,
        help="Worker auto-tune target wall-clock seconds per batch.",
    )
    ap.add_argument(
        "--distributed-worker-min-games-per-batch",
        type=int,
        default=1,
        help="Worker auto-tune lower bound for games_per_batch.",
    )
    ap.add_argument(
        "--distributed-worker-max-games-per-batch",
        type=int,
        default=64,
        help="Worker auto-tune upper bound for games_per_batch.",
    )
    ap.add_argument(
        "--distributed-worker-upload-target-positions",
        type=int,
        default=500,
        help="Flush a small durable worker upload once this many positions are buffered.",
    )
    ap.add_argument(
        "--distributed-worker-upload-flush-seconds",
        type=float,
        default=60.0,
        help="Flush at the next completed-game boundary once this many seconds have elapsed since the last successful upload.",
    )
    ap.add_argument(
        "--distributed-worker-shared-cache-dir",
        type=str,
        default="",
        help="Optional shared cache directory for Tune-managed local worker processes.",
    )
    ap.add_argument(
        "--distributed-worker-username",
        type=str,
        default="",
        help="Username for distributed worker server authentication.",
    )
    ap.add_argument(
        "--distributed-worker-password",
        type=str,
        default="",
        help="Password for distributed worker server authentication.",
    )
    ap.add_argument(
        "--distributed-min-workers-per-trial",
        type=int,
        default=1,
        help="Minimum worker floor to maintain per active trial before floating workers are reassigned.",
    )
    ap.add_argument(
        "--distributed-max-worker-delta-per-rebalance",
        type=int,
        default=1,
        help="Maximum worker-count lead one trial can gain over the least-served trial in one rebalance step.",
    )
    ap.add_argument(
        "--distributed-wait-timeout-seconds",
        type=float,
        default=900.0,
        help="How long a Tune trial waits for enough distributed selfplay shards for the current model.",
    )
    ap.add_argument(
        "--distributed-server-port",
        type=int,
        default=0,
        help="Central distributed Tune server port (0 = auto-pick a free local port).",
    )
    ap.add_argument(
        "--distributed-server-host",
        type=str,
        default="127.0.0.1",
        help="Host/interface for the central distributed Tune server to bind (use 0.0.0.0 for LAN workers).",
    )
    ap.add_argument(
        "--distributed-server-public-url",
        type=str,
        default="",
        help="Optional externally reachable base URL for remote workers (defaults to http://<host>:<port>).",
    )
    ap.add_argument(
        "--distributed-server-root-override",
        type=str,
        default="",
        help=(
            "Optional filesystem root for the Tune-managed distributed server state. "
            "If unset, defaults to <work_dir>/server."
        ),
    )
    ap.add_argument(
        "--tune-replay-root-override",
        type=str,
        default="",
        help=(
            "Optional filesystem root for per-trial replay shard storage. "
            "If unset, replay stays under each Ray trial directory."
        ),
    )
    ap.add_argument(
        "--distributed-upload-compact-shard-size",
        type=int,
        default=2000,
        help="Server-side compaction target positions for learner-facing distributed selfplay shards.",
    )
    ap.add_argument(
        "--distributed-upload-compact-max-age-seconds",
        type=float,
        default=90.0,
        help="Age-based partial flush threshold for server-side selfplay upload compaction.",
    )
    ap.add_argument(
        "--distributed-inference-broker-enabled",
        action="store_true",
        help="Launch one per-trial local inference broker and let local workers submit shared-memory eval batches to it.",
    )
    ap.add_argument(
        "--distributed-inference-batch-wait-ms",
        type=float,
        default=3.0,
        help="How long the per-trial inference broker waits to aggregate local worker eval requests.",
    )
    ap.add_argument(
        "--distributed-inference-max-batch-per-slot",
        "--distributed-inference-max-batch-positions",
        type=int,
        default=512,
        help="Maximum encoded positions a single worker can submit in one shared-memory inference request.",
    )
    ap.add_argument(
        "--gpus-per-trial", type=float, default=0.1,
        help="GPU fraction per trial. Use <1.0 to pack multiple trials on one GPU "
             "(e.g. 0.1 for 10 trials on a single 5090).",
    )
    ap.add_argument(
        "--tune-scheduler", type=str, default="pb2", choices=["pb2", "pbt", "gpbt_pl", "asha"],
        help=(
            "Tune scheduler: pb2 (default, sample-efficient for RL), "
            "pbt (vanilla population-based training), gpbt_pl (pairwise-learning "
            "PBT variant), or asha (legacy, "
            "for architecture ablations)."
        ),
    )
    ap.add_argument(
        "--pb2-perturbation-interval", type=int, default=10,
        help="PB2/PBT: iterations between exploit+explore steps.",
    )
    ap.add_argument(
        "--pbt-synch", action="store_true",
        help="PBT: perturb synchronously across trials instead of asynchronously.",
    )
    ap.add_argument(
        "--gpbt-inertia-weight", type=float, default=1.0,
        help="GPBT-PL: inertia weight for per-trial velocity (multiplied by r1~U[0,1]).",
    )
    ap.add_argument(
        "--gpbt-winner-weight", type=float, default=1.0,
        help="GPBT-PL: winner weight for attraction toward donor (multiplied by r2~U[0,1]).",
    )
    # Legacy aliases kept for backwards compatibility with old configs
    ap.add_argument("--gpbt-pairwise-lr", type=float, default=0.35, help=argparse.SUPPRESS)
    ap.add_argument("--gpbt-pairwise-momentum", type=float, default=0.5, help=argparse.SUPPRESS)
    ap.add_argument(
        "--gpbt-quantile-fraction", type=float, default=0.25,
        help="GPBT-PL: bottom/top population fraction used for exploit pairing.",
    )
    ap.add_argument(
        "--gpbt-resample-probability", type=float, default=0.05,
        help="GPBT-PL: resample probability for categorical hyperparameters.",
    )
    ap.add_argument("--search-smolgen", action="store_true",
                    help="PB2/ASHA: include smolgen on/off as a binary search dimension.")
    ap.add_argument("--search-nla", action="store_true",
                    help="PB2/ASHA: include NLA on/off as a binary search dimension.")
    ap.add_argument("--search-optimizer", action="store_true",
                    help="Include optimizer family as a Tune search dimension.")
    ap.add_argument(
        "--search-optimizer-choices",
        type=str,
        nargs="+",
        default=None,
        help="Restrict optimizer search/mutation to this subset, e.g. --search-optimizer-choices adamw cosmos_fast.",
    )
    ap.add_argument("--asha-optimizer-only", action="store_true",
                    help="ASHA: isolate optimizer as the only search dimension and use a deterministic optimizer grid.")
    ap.add_argument("--asha-optimizer-repeats", type=int, default=1,
                    help="ASHA optimizer-only mode: number of seed repeats per optimizer.")

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
    ap.add_argument("--bootstrap-zero-policy-heads", action="store_true",
                    help="After loading bootstrap weights, zero policy heads so priors start near-uniform.")
    ap.add_argument("--worker-wheel-path", type=str, default=None,
                    help="Optional wheel file to publish for worker self-update in Tune manifests.")
    ap.add_argument("--bootstrap-max-positions", type=int, default=0, help="Max bootstrap positions to load (0=unlimited)")
    ap.add_argument("--bootstrap-train-steps", type=int, default=0, help="Pre-training steps on bootstrap data (0=disabled)")
    ap.add_argument("--gate-games", type=int, default=0, help="Net gating: games to play after training (0=disabled)")
    ap.add_argument("--gate-threshold", type=float, default=0.50, help="Net gating: reject if winrate below this")
    ap.add_argument("--gate-interval", type=int, default=1, help="Net gating: check every N iterations")
    ap.add_argument("--gate-mcts-sims", type=int, default=1, help="MCTS sims for gate games (1=raw policy)")
    ap.add_argument("--shuffle-buffer-size", type=int, default=20_000, help="Disk replay: in-memory shuffle buffer size")
    ap.add_argument(
        "--shuffle-draw-cap-frac",
        type=float,
        default=0.90,
        help="Disk replay: cap draws to this fraction of each balanced batch.",
    )
    ap.add_argument(
        "--shuffle-wl-max-ratio",
        type=float,
        default=1.5,
        help="Disk replay: max allowed win/loss imbalance ratio in balanced batches.",
    )
    ap.add_argument("--shard-size", type=int, default=1000, help="Disk replay: samples per on-disk shard")
    ap.add_argument("--exploit-replay-refresh-enabled", action="store_true",
                    help="On PB2 exploit restore, refresh recipient replay shards (prune oldest + inject donor recent).")
    ap.add_argument("--exploit-replay-keep-fraction", type=float, default=0.60,
                    help="Fraction of newest local replay shards to keep on exploit refresh.")
    ap.add_argument("--exploit-replay-donor-shards", type=int, default=6,
                    help="Number of donor recent replay shards to copy into recipient on exploit refresh.")
    ap.add_argument("--exploit-replay-skip-newest", type=int, default=1,
                    help="Skip this many newest donor shards when injecting (avoid in-flight write races).")
    ap.add_argument("--exploit-replay-share-top-enabled", action="store_true",
                    help="On exploit restore, also inject recent replay shards from current top sibling trials.")
    ap.add_argument("--exploit-replay-top-k-trials", type=int, default=0,
                    help="How many top sibling trials to source from on exploit refresh (0 or negative means all).")
    ap.add_argument("--exploit-replay-top-within-best-frac", type=float, default=0.10,
                    help="Only source from trials with metric >= best*(1-frac), e.g. 0.10 keeps within 10%% of best.")
    ap.add_argument("--exploit-replay-top-shards-per-source", type=int, default=0,
                    help="Deprecated/ignored: latest-generation shards are imported automatically.")
    ap.add_argument("--exploit-replay-top-min-metric", type=float, default=-1e9,
                    help="Minimum latest trial metric required for top-trial replay sharing.")
    ap.add_argument("--exploit-replay-local-keep-recent-fraction", type=float, default=0.20,
                    help="Fraction of recipient's most recent local generation to keep on exploit refresh.")
    ap.add_argument("--exploit-replay-local-keep-older-fraction", type=float, default=0.65,
                    help="Fraction of recipient's older local replay to keep on exploit refresh.")
    ap.add_argument("--pause-file", type=str, default=None,
                    help="If this file exists, each trial pauses at iteration boundaries until it is removed.")
    ap.add_argument("--pause-poll-seconds", type=int, default=60,
                    help="Pause gate polling interval in seconds.")
    ap.add_argument("--salvage-seed-pool-dir", type=str, default=None,
                    help="Optional seed pool dir (from --mode salvage) to warm-start fresh trials.")
    ap.add_argument("--salvage-restore-pid-state", action="store_true",
                    help="When warm-starting from a salvage seed, also restore the donor PID difficulty state.")
    ap.add_argument("--replay-window-start", type=int, default=100_000, help="Initial sliding window size")
    ap.add_argument("--replay-window-max", type=int, default=1_000_000, help="Max sliding window size")
    ap.add_argument("--replay-window-growth", type=int, default=10_000, help="Window growth per iteration")
    ap.add_argument("--w-sf-wdl", type=float, default=1.0, help="SF WDL bootstrap weight for main value head")
    ap.add_argument(
        "--sf-wdl-conf-power",
        type=float,
        default=0.0,
        help="Optional confidence damping for SF-WDL loss: weight by (1-draw_prob)^power (0=disabled).",
    )
    ap.add_argument(
        "--sf-wdl-draw-scale",
        type=float,
        default=1.0,
        help="Optional extra multiplier for SF-WDL on draw outcomes (1=disabled, <1 damps draws).",
    )
    ap.add_argument("--sf-wdl-floor", type=float, default=0.1, help="SF WDL weight floor")
    ap.add_argument("--sf-wdl-floor-at", type=float, default=0.1, help="random_move_prob at which SF WDL weight reaches floor")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--model", type=str, default="transformer", choices=["tiny", "transformer"])
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--ffn-mult", type=float, default=2)
    ap.add_argument("--no-smolgen", action="store_true")
    ap.add_argument("--use-nla", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Enable gradient checkpointing to reduce VRAM (~50%% less activation memory)")
    ap.add_argument("--stockfish-path", type=str, default=None)
    ap.add_argument("--sf-nodes", type=int, default=2000)
    ap.add_argument("--sf-multipv", type=int, default=5)
    ap.add_argument("--sf-hash-mb", type=int, default=16)
    ap.add_argument("--sf-workers", type=int, default=1)
    ap.add_argument("--sf-policy-temp", type=float, default=0.25)
    ap.add_argument("--sf-policy-label-smooth", type=float, default=0.05)

    ap.add_argument("--opening-book-path", type=str, default=None, help="Path to opening book (.bin polyglot, .pgn, or .pgn.zip)")
    ap.add_argument("--opening-book-max-plies", type=int, default=4)
    ap.add_argument("--opening-book-max-games", type=int, default=200000)
    ap.add_argument("--opening-book-prob", type=float, default=1.0)
    ap.add_argument("--opening-book-path-2", type=str, default=None, help="Path to second opening book")
    ap.add_argument("--opening-book-max-plies-2", type=int, default=16)
    ap.add_argument("--opening-book-max-games-2", type=int, default=200000)
    ap.add_argument("--opening-book-mix-prob-2", type=float, default=0.0, help="Fraction of book games using book 2")
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
    ap.add_argument("--sf-pid-initial-skill-level", type=int, default=0)
    ap.add_argument("--sf-pid-skill-min", type=int, default=0)
    ap.add_argument("--sf-pid-skill-max", type=int, default=20)
    ap.add_argument("--sf-pid-skill-promote-nodes", type=int, default=200)
    ap.add_argument("--sf-pid-skill-demote-nodes", type=int, default=100)
    ap.add_argument("--sf-pid-skill-nodes-on-promote", type=int, default=100)
    ap.add_argument("--sf-pid-skill-nodes-on-demote", type=int, default=150)
    ap.add_argument("--sf-pid-random-move-prob-start", type=float, default=1.0)
    ap.add_argument("--sf-pid-random-move-prob-min", type=float, default=0.0)
    ap.add_argument("--sf-pid-random-move-prob-max", type=float, default=1.0)
    ap.add_argument("--sf-pid-random-move-stage-end", type=float, default=0.5)
    ap.add_argument("--sf-pid-topk-stage-end", type=float, default=0.5)
    ap.add_argument("--sf-pid-topk-min", type=int, default=1)
    ap.add_argument(
        "--sf-pid-suboptimal-wdl-regret-max",
        type=float,
        default=-1.0,
        help="If >=0, treat opponent_random_move_prob as chance of choosing a non-best move within this WDL regret band at easy stage start.",
    )
    ap.add_argument(
        "--sf-pid-suboptimal-wdl-regret-min",
        type=float,
        default=-1.0,
        help="If >=0, WDL regret floor reached when random_move_prob hits its configured minimum.",
    )
    ap.add_argument("--sf-pid-max-rand-step", type=float, default=0.01)
    ap.add_argument("--games-per-iter", type=int, default=10)
    ap.add_argument("--games-per-iter-start", type=int, default=0,
                    help="Optional starting games_per_iter for Tune ramping (0 disables).")
    ap.add_argument("--games-per-iter-ramp-iters", type=int, default=0,
                    help="Iterations to ramp games_per_iter_start up to games_per_iter.")
    ap.add_argument("--selfplay-batch", type=int, default=10, help="Play games in mini-batches of this size to limit memory")
    ap.add_argument("--selfplay-fraction", type=float, default=0.0, help="Fraction of games that are true net-vs-net self-play")
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
    ap.add_argument("--max-plies", type=int, default=240)

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
    ap.add_argument("--optimizer", type=str, default="nadamw", choices=["nadamw", "adamw", "muon", "cosmos", "cosmos_fast", "soap"],
                    help="Optimizer: nadamw, adamw, muon, cosmos, cosmos_fast, or soap")
    ap.add_argument("--cosmos-rank", type=int, default=64, help="COSMOS/COSMOSFast low-rank subspace rank")
    ap.add_argument("--cosmos-gamma", type=float, default=0.2, help="COSMOS/COSMOSFast residual branch weight")
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP (BF16 autocast on CUDA)")
    ap.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation micro-batches")
    ap.add_argument("--warmup-steps", type=int, default=1500, help="Linear LR warmup steps")
    ap.add_argument("--warmup-lr-start", type=float, default=None,
                    help="Optional warmup starting LR override (defaults to --lr-eta-min).")
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

    if str(args.mode) == "salvage":
        _run_salvage(args)
        return

    if args.stockfish_path is None:
        raise SystemExit("--stockfish-path is required (or set stockfish.path in --config)")

    if str(args.mode) == "single":
        _run_single(args)
        return

    # Tune mode (default)
    base = _build_tune_config_dict(args)

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
