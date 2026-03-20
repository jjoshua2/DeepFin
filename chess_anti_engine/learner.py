from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay import ArrayReplayBuffer
from chess_anti_engine.replay.shard import load_npz_arrays
from chess_anti_engine.selfplay.budget import progressive_mcts_simulations
from chess_anti_engine.stockfish.pid import DifficultyPID
from chess_anti_engine.train import Trainer
from chess_anti_engine.version import PACKAGE_VERSION, PROTOCOL_VERSION


log = logging.getLogger("chess_anti_engine.learner")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write a text file atomically (temp + os.replace).

    Important for files served over HTTP while the learner is running.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _atomic_copy2(src: Path, dst: Path) -> None:
    """shutil.copy2, but atomically replace dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        shutil.copy2(src, tmp)
        os.replace(str(tmp), str(dst))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _iter_shards(inbox_dir: Path) -> list[Path]:
    # Depth: inbox/<user>/<sha>.npz
    return sorted(inbox_dir.glob("*/*.npz"))


def _iter_arena_results(arena_inbox_dir: Path) -> list[Path]:
    # Depth: arena_inbox/<user>/<sha>.json
    return sorted(arena_inbox_dir.glob("*/*.json"))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="Learner (trainer) for distributed selfplay")

    ap.add_argument("--server-root", type=str, default="server")
    ap.add_argument("--trial-id", type=str, default=None)
    ap.add_argument("--publish-dir", type=str, default="publish")
    ap.add_argument("--inbox-dir", type=str, default="inbox")
    ap.add_argument("--processed-dir", type=str, default="processed")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)

    # Model config (must match what workers will load)
    ap.add_argument("--model", type=str, default="transformer", choices=["tiny", "transformer"])
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--ffn-mult", type=int, default=2)
    ap.add_argument("--no-smolgen", action="store_true")
    ap.add_argument("--use-nla", action="store_true")
    ap.add_argument(
        "--use-qk-rmsnorm",
        action="store_true",
        help="Enable QK RMSNorm (normalize Q/K per-head before attention dot product)",
    )
    ap.add_argument("--gradient-checkpointing", action="store_true")

    # Training knobs
    ap.add_argument("--replay-capacity", type=int, default=200_000)
    ap.add_argument("--min-replay-size", type=int, default=2048)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--zclip-z-thresh", type=float, default=2.5)
    ap.add_argument("--zclip-alpha", type=float, default=0.97)
    ap.add_argument("--zclip-max-norm", type=float, default=1.0)
    ap.add_argument("--work-dir", type=str, default="server/work")

    ap.add_argument("--feature-dropout-p", type=float, default=0.3)
    ap.add_argument("--w-volatility", type=float, default=0.05)
    ap.add_argument("--w-soft", type=float, default=0.0,
                    help="Weight for MCTS temperature-2 soft policy target. 0 = disabled (default). "
                         "Only useful once the network plays real chess.")
    ap.add_argument("--w-sf-wdl", type=float, default=0.0,
                    help="Initial weight for soft WDL loss using SF eval as target on the main WDL head.")
    ap.add_argument("--w-sf-wdl-end", type=float, default=None,
                    help="Final w_sf_wdl after ramp (when opponent_random_move_prob reaches 0). "
                         "Defaults to w_sf_wdl (no ramp).")


    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--warmup-steps", type=int, default=1500)
    ap.add_argument("--lr-eta-min", type=float, default=1e-5)
    ap.add_argument("--lr-T0", type=int, default=5000)
    ap.add_argument("--lr-T-mult", type=int, default=2)
    ap.add_argument("--optimizer", type=str, default="nadamw", choices=["nadamw", "adamw", "muon", "cosmos", "cosmos_fast", "soap"])
    ap.add_argument("--cosmos-rank", type=int, default=64)
    ap.add_argument("--cosmos-gamma", type=float, default=0.2)

    # Loop control
    ap.add_argument("--sleep-seconds", type=float, default=2.0)
    ap.add_argument("--max-shards-per-iter", type=int, default=32)

    # Optional: advertise an opening book in the manifest
    ap.add_argument("--opening-book-path", type=str, default=None)

    # Optional: publish additional artifacts for workers to download.
    # - stockfish: optional server-distributed Stockfish binary (or archive)
    # - worker-wheel: optional wheel file for worker self-update
    ap.add_argument("--stockfish-binary-path", type=str, default=None)
    ap.add_argument("--worker-wheel-path", type=str, default=None)

    # Arena (latest vs best) scheduling
    arena_group = ap.add_mutually_exclusive_group()
    arena_group.add_argument(
        "--arena",
        dest="arena_enabled",
        action="store_true",
        help="Enable arena matches (latest vs best) between training rounds",
    )
    arena_group.add_argument(
        "--no-arena",
        dest="arena_enabled",
        action="store_false",
        help="Disable arena matches",
    )
    # Default to match-gated promotion (best model = arena champion), not train-loss.
    ap.set_defaults(arena_enabled=True)

    ap.add_argument("--arena-games-target", type=int, default=100)
    ap.add_argument("--arena-every-n-steps", type=int, default=1000,
                    help="Trigger arena once per this many trainer steps (default 1000 ≈ every ~25 min).")
    ap.add_argument("--arena-batch-games", type=int, default=8, help="Games per worker upload batch during arena")
    ap.add_argument("--arena-max-plies", type=int, default=200)
    ap.add_argument("--arena-mcts", type=str, default="puct", choices=["puct", "gumbel"])
    ap.add_argument("--arena-mcts-simulations", type=int, default=200)
    ap.add_argument("--arena-c-puct", type=float, default=2.5)
    ap.add_argument("--arena-swap-sides", action="store_true", help="Alternate colors in arena batches")
    ap.set_defaults(arena_swap_sides=True)
    ap.add_argument("--arena-temperature", type=float, default=0.1,
                    help="Sampling temperature for arena moves (default 0.1; near-greedy but avoids identical games)")
    ap.add_argument("--arena-random-start-plies", type=int, default=2,
                    help="Random opening moves per game to diversify arena positions (default 2)")

    # Default: accept challenger if it scores at least 50% vs champion (promote on tie).
    ap.add_argument("--arena-accept-winrate", type=float, default=0.50)

    # Protocol / compatibility
    ap.add_argument(
        "--min-worker-version",
        type=str,
        default=str(PACKAGE_VERSION),
        help="Minimum worker package version allowed to participate (download manifest, upload shards).",
    )

    # Recommended worker knobs (manifest-only; workers can still override locally)
    ap.add_argument("--recommended-games-per-batch", type=int, default=8)
    ap.add_argument("--recommended-max-plies", type=int, default=200)
    ap.add_argument("--max-plies-start", type=int, default=None,
                    help="Initial max_plies for progressive ramp. Defaults to --recommended-max-plies.")
    ap.add_argument("--max-plies-ramp-steps", type=int, default=0,
                    help="Linearly ramp max_plies from --max-plies-start to --recommended-max-plies "
                         "over this many trainer steps. 0 = no ramp (fixed at recommended-max-plies).")
    ap.add_argument("--recommended-mcts", type=str, default="puct", choices=["puct", "gumbel"])
    ap.add_argument(
        "--recommended-opponent-random-move-prob",
        type=float,
        default=0.0,
        help="Probability the opponent plays a random legal move instead of Stockfish's best move (still records SF targets).",
    )
    # Progressive simulation budget for workers: start low and ramp to a higher cap.
    ap.add_argument("--recommended-mcts-simulations", type=int, default=800, help="Max (cap) simulations once ramped")

    reco_prog = ap.add_mutually_exclusive_group()
    reco_prog.add_argument("--recommended-progressive-mcts", dest="recommended_progressive_mcts", action="store_true")
    reco_prog.add_argument("--no-recommended-progressive-mcts", dest="recommended_progressive_mcts", action="store_false")
    ap.set_defaults(recommended_progressive_mcts=True)

    ap.add_argument("--recommended-mcts-start-simulations", type=int, default=50)
    ap.add_argument("--recommended-mcts-ramp-steps", type=int, default=10_000)
    ap.add_argument("--recommended-mcts-ramp-exponent", type=float, default=2.0)
    ap.add_argument("--recommended-playout-cap-fraction", type=float, default=0.25)
    ap.add_argument("--recommended-fast-simulations", type=int, default=8)

    # Stockfish settings to keep data distribution consistent across workers
    ap.add_argument("--recommended-sf-nodes", type=int, default=2000)
    ap.add_argument("--recommended-sf-multipv", type=int, default=5)

    # PID adaptive difficulty: adjust sf_nodes to maintain target network win rate
    pid_group = ap.add_mutually_exclusive_group()
    pid_group.add_argument("--pid", dest="pid_enabled", action="store_true",
                           help="Enable PID adaptive difficulty (default on)")
    pid_group.add_argument("--no-pid", dest="pid_enabled", action="store_false",
                           help="Disable PID; sf_nodes stays fixed at --recommended-sf-nodes")
    ap.set_defaults(pid_enabled=True)
    ap.add_argument("--pid-target-winrate", type=float, default=0.53,
                    help="Network win rate the PID targets (default 0.53)")
    ap.add_argument("--pid-ema-alpha", type=float, default=0.05,
                    help="EMA smoothing factor for win rate (higher = more reactive)")
    ap.add_argument("--pid-min-nodes", type=int, default=250)
    ap.add_argument("--pid-max-nodes", type=int, default=500_000)
    ap.add_argument("--pid-random-move-prob-start", type=float, default=None,
                    help="Initial opponent random-move probability for PID (defaults to --recommended-opponent-random-move-prob).")
    ap.add_argument("--pid-random-move-prob-min", type=float, default=0.0)
    ap.add_argument("--pid-random-move-prob-max", type=float, default=1.0)
    ap.add_argument("--pid-max-rand-step", type=float, default=0.01,
                    help="Max absolute change to random_move_prob per PID adjustment (default 0.01 = 1%%)")
    ap.add_argument("--pid-random-move-stage-end", type=float, default=0.5,
                    help="While random_move_prob > this threshold, PID adjusts only random_move_prob (nodes/skill frozen).")
    ap.add_argument("--pid-min-games-between-adjust", type=int, default=20,
                    help="Minimum game positions ingested before each PID adjustment")
    ap.add_argument("--recommended-sf-policy-temp", type=float, default=0.25)
    ap.add_argument("--recommended-sf-policy-label-smooth", type=float, default=0.05)
    ap.add_argument("--recommended-sf-skill-level", type=int, default=None,
                    help="Stockfish Skill Level 0-20 (None = max strength). Lower = weaker/faster.")

    # Temperature schedule knobs (optional; workers can still be told to pin these)
    ap.add_argument("--recommended-temperature", type=float, default=1.0)
    ap.add_argument("--recommended-temperature-decay-start-move", type=int, default=20)
    ap.add_argument("--recommended-temperature-decay-moves", type=int, default=60)
    ap.add_argument("--recommended-temperature-endgame", type=float, default=0.6)

    ap.add_argument("--recommended-opening-book-max-plies", type=int, default=4)
    ap.add_argument("--recommended-opening-book-max-games", type=int, default=200000)
    ap.add_argument("--recommended-opening-book-prob", type=float, default=1.0)
    ap.add_argument("--recommended-random-start-plies", type=int, default=0)

    args = ap.parse_args()

    # Progressive max_plies: start low and ramp up to recommended_max_plies.
    _max_plies_start: int = int(args.max_plies_start) if args.max_plies_start is not None else int(args.recommended_max_plies)
    _max_plies_end: int = int(args.recommended_max_plies)
    _max_plies_ramp_steps: int = int(args.max_plies_ramp_steps)

    def _compute_max_plies(step: int) -> int:
        if _max_plies_ramp_steps <= 0 or _max_plies_start >= _max_plies_end:
            return _max_plies_end
        frac = min(1.0, step / _max_plies_ramp_steps)
        return int(_max_plies_start + frac * (_max_plies_end - _max_plies_start))

    # SF-bootstrap loss weight ramp: as the opponent gets harder (random_move_prob → 0),
    # gradually reduce reliance on SF eval (wdl + sf_move) as training targets.
    # NOTE: w_soft is the MCTS temperature-2 policy target (not SF) and is NOT ramped.
    _w_sf_wdl_start: float = float(args.w_sf_wdl)
    _w_sf_wdl_end: float = float(args.w_sf_wdl_end) if args.w_sf_wdl_end is not None else _w_sf_wdl_start
    _rand_prob_ramp_start: float = float(args.pid_random_move_prob_start) if args.pid_random_move_prob_start is not None else float(args.recommended_opponent_random_move_prob)
    # Ramp completes (SF weights reach end value) once random_move_prob falls to this threshold.
    _rand_prob_ramp_end_threshold: float = 0.10

    def _sf_bootstrap_ramp_frac(rand_prob: float) -> float:
        """Returns 0.0 at ramp start (full SF bootstrap) → 1.0 when rand_prob ≤ 10% random."""
        span = _rand_prob_ramp_start - _rand_prob_ramp_end_threshold
        if span <= 0.0:
            return 1.0
        return max(0.0, min(1.0, (_rand_prob_ramp_start - rand_prob) / span))

    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    # Mutable difficulty knobs. PID may update these over time.
    current_sf_nodes: int = int(args.recommended_sf_nodes)
    current_sf_skill_level: int | None = (
        None if args.recommended_sf_skill_level is None else int(args.recommended_sf_skill_level)
    )
    current_opponent_random_move_prob: float = float(args.recommended_opponent_random_move_prob)

    pid: DifficultyPID | None = None
    if bool(args.pid_enabled):
        rm_start = args.pid_random_move_prob_start
        if rm_start is None:
            rm_start = float(args.recommended_opponent_random_move_prob)

        pid = DifficultyPID(
            initial_nodes=current_sf_nodes,
            target_winrate=float(args.pid_target_winrate),
            ema_alpha=float(args.pid_ema_alpha),
            min_nodes=int(args.pid_min_nodes),
            max_nodes=int(args.pid_max_nodes),
            min_games_between_adjust=int(args.pid_min_games_between_adjust),
            initial_skill_level=int(current_sf_skill_level) if current_sf_skill_level is not None else 1,
            skill_min=0,  # Level 0 is weakest (still plays legal moves), 20 is strongest
            initial_random_move_prob=float(rm_start),
            random_move_prob_min=float(args.pid_random_move_prob_min),
            random_move_prob_max=float(args.pid_random_move_prob_max),
            random_move_stage_end=float(args.pid_random_move_stage_end),
            max_rand_step=float(args.pid_max_rand_step),
        )
        current_opponent_random_move_prob = float(pid.random_move_prob)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(int(args.seed))

    server_root = Path(args.server_root)
    trial_id = str(args.trial_id).strip() if args.trial_id is not None else ""
    trial_root = server_root / "trials" / trial_id if trial_id else server_root
    api_prefix = f"/v1/trials/{trial_id}" if trial_id else "/v1"

    def _resolve_trial_subdir(path_str: str, *, fallback_name: str | None = None) -> Path:
        raw = Path(path_str)
        if raw.is_absolute():
            return raw
        if trial_id:
            if fallback_name is not None and raw == Path("server") / fallback_name:
                return trial_root / fallback_name
            return trial_root / raw
        return server_root / raw if raw.parts[:1] != (server_root.name,) else raw

    publish_dir = _resolve_trial_subdir(str(args.publish_dir))
    inbox_dir = _resolve_trial_subdir(str(args.inbox_dir))
    processed_dir = _resolve_trial_subdir(str(args.processed_dir))
    arena_inbox_dir = trial_root / "arena_inbox"
    arena_processed_dir = trial_root / "arena_processed"
    work_dir = _resolve_trial_subdir(str(args.work_dir), fallback_name="work")

    publish_dir.mkdir(parents=True, exist_ok=True)
    inbox_dir.mkdir(parents=True, exist_ok=True)

    # Optional published artifacts (copied into publish/ for HTTP serving).
    published_stockfish_path: Path | None = None
    published_worker_wheel_path: Path | None = None

    if args.stockfish_binary_path:
        src = Path(args.stockfish_binary_path)
        if src.exists() and src.is_file():
            dst = publish_dir / ("stockfish" + src.suffix)
            try:
                _atomic_copy2(src, dst)
                published_stockfish_path = dst
            except Exception:
                published_stockfish_path = None

    if args.worker_wheel_path:
        src = Path(args.worker_wheel_path)
        if src.exists() and src.is_file():
            # Keep a stable filename so the endpoint doesn't need to change.
            dst = publish_dir / "worker.whl"
            try:
                _atomic_copy2(src, dst)
                published_worker_wheel_path = dst
            except Exception:
                published_worker_wheel_path = None
    processed_dir.mkdir(parents=True, exist_ok=True)
    arena_inbox_dir.mkdir(parents=True, exist_ok=True)
    arena_processed_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Restore PID state across restarts.
    pid_state_path = work_dir / "pid_state.json"
    if pid is not None and pid_state_path.exists():
        try:
            pd = json.loads(pid_state_path.read_text(encoding="utf-8"))
            pid.nodes = int(pd.get("nodes", current_sf_nodes))
            raw_ema = pd.get("ema_winrate")
            pid.ema_winrate = float(raw_ema) if raw_ema is not None else None
            pid._integral = float(pd.get("integral", 0.0))
            raw_skill = pd.get("skill_level")
            if raw_skill is not None:
                pid.skill_level = max(pid.skill_min, min(pid.skill_max, int(raw_skill)))
            raw_rm = pd.get("random_move_prob")
            if raw_rm is not None:
                pid.random_move_prob = float(raw_rm)

            current_sf_nodes = pid.nodes
            current_sf_skill_level = pid.skill_level
            current_opponent_random_move_prob = float(pid.random_move_prob)
        except Exception:
            pass

    model_cfg = ModelConfig(
        kind=str(args.model),
        embed_dim=int(args.embed_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        ffn_mult=int(args.ffn_mult),
        use_smolgen=not bool(args.no_smolgen),
        use_nla=bool(args.use_nla),
        use_qk_rmsnorm=bool(args.use_qk_rmsnorm),
        use_gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    model = build_model(model_cfg)

    trainer = Trainer(
        model,
        device=str(device),
        lr=float(args.lr),
        zclip_z_thresh=float(args.zclip_z_thresh),
        zclip_alpha=float(args.zclip_alpha),
        zclip_max_norm=float(args.zclip_max_norm),
        log_dir=work_dir / "tb",
        use_amp=bool(str(device).startswith("cuda")),
        feature_dropout_p=float(args.feature_dropout_p),
        w_volatility=float(args.w_volatility),
        w_soft=float(args.w_soft),
        w_sf_wdl=_w_sf_wdl_start,
        accum_steps=int(args.accum_steps),
        warmup_steps=int(args.warmup_steps),
        warmup_lr_start=getattr(args, "warmup_lr_start", None),
        lr_eta_min=float(args.lr_eta_min),
        lr_T0=int(args.lr_T0),
        lr_T_mult=int(args.lr_T_mult),
        optimizer=str(args.optimizer),
        cosmos_rank=int(getattr(args, "cosmos_rank", 64)),
        cosmos_gamma=float(getattr(args, "cosmos_gamma", 0.2)),
    )

    ckpt_path = work_dir / "trainer.pt"
    if ckpt_path.exists():
        trainer.load(ckpt_path)

    # Best-by-loss tracking (local diagnostics only; does NOT control arena opponent).
    best_loss_state_path = work_dir / "best_loss.json"
    best_loss_ckpt_path = work_dir / "best_loss_trainer.pt"
    best_loss_model_path = work_dir / "best_loss_model.pt"

    best_loss = float("inf")
    best_loss_trainer_step = -1

    if best_loss_state_path.exists():
        try:
            d = json.loads(best_loss_state_path.read_text(encoding="utf-8"))
            best_loss = float(d.get("best_loss", d.get("loss", best_loss)))
            best_loss_trainer_step = int(d.get("trainer_step", d.get("best_trainer_step", best_loss_trainer_step)))
        except Exception:
            pass

    # Arena incumbent/champion model (published) + trainer state (used for rejection rollback).
    best_model_path = publish_dir / "best_model.pt"
    champion_ckpt_path = work_dir / "champion_trainer.pt"
    best_model_sha = _sha256_file(best_model_path) if best_model_path.exists() else ""

    # Last-known training metrics (used for manifest while arena is running)
    last_train_path = work_dir / "last_train.json"
    last_train = {
        "loss": None,
        "policy_loss": None,
        "wdl_loss": None,
        "updated_at_unix": None,
    }
    if last_train_path.exists():
        try:
            last_train.update(json.loads(last_train_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    # Arena state
    arena_state_path = work_dir / "arena_state.json"
    arena_active = False
    arena_latest_sha = ""
    arena_best_sha = ""
    arena_games_done = 0
    arena_a_win = 0
    arena_a_draw = 0
    arena_a_loss = 0
    arena_last_winrate = None
    arena_last_accepted = None

    if arena_state_path.exists():
        try:
            s = json.loads(arena_state_path.read_text(encoding="utf-8"))
            arena_active = bool(s.get("active", False))
            arena_latest_sha = str(s.get("latest_sha", ""))
            arena_best_sha = str(s.get("best_sha", ""))
            arena_games_done = int(s.get("games_done", 0))
            arena_a_win = int(s.get("a_win", 0))
            arena_a_draw = int(s.get("a_draw", 0))
            arena_a_loss = int(s.get("a_loss", 0))
            arena_last_winrate = s.get("last_winrate")
            arena_last_accepted = s.get("last_accepted")
        except Exception:
            pass

    buf = ArrayReplayBuffer(int(args.replay_capacity), rng=rng)

    # Bootstrap publish: write an initial manifest + initial model weights so workers can
    # start generating shards even before the learner has enough data to train.
    try:
        model_path = publish_dir / "latest_model.pt"
        trainer.export_swa(model_path)
        model_sha = _sha256_file(model_path) if model_path.exists() else ""

        if not best_model_path.exists() and model_path.exists():
            _atomic_copy2(model_path, best_model_path)
        best_model_sha = _sha256_file(best_model_path) if best_model_path.exists() else ""

        trainer_step = int(getattr(trainer, "step", 0))
        max_sims = int(args.recommended_mcts_simulations)
        reco_sims = max_sims
        if bool(getattr(args, "recommended_progressive_mcts", True)):
            reco_sims = progressive_mcts_simulations(
                trainer_step,
                start=int(args.recommended_mcts_start_simulations),
                max_sims=max_sims,
                ramp_steps=int(args.recommended_mcts_ramp_steps),
                exponent=float(args.recommended_mcts_ramp_exponent),
            )

        recommended_worker = {
            "games_per_batch": int(args.recommended_games_per_batch),
            "max_plies": _compute_max_plies(trainer_step),
            "mcts": str(args.recommended_mcts),
            "mcts_simulations": int(reco_sims),
            "playout_cap_fraction": float(args.recommended_playout_cap_fraction),
            "fast_simulations": int(args.recommended_fast_simulations),
            "opening_book_max_plies": int(args.recommended_opening_book_max_plies),
            "opening_book_max_games": int(args.recommended_opening_book_max_games),
            "opening_book_prob": float(args.recommended_opening_book_prob),
            "random_start_plies": int(args.recommended_random_start_plies),
            "sf_nodes": int(current_sf_nodes),
            "sf_multipv": int(args.recommended_sf_multipv),
            "sf_policy_temp": float(args.recommended_sf_policy_temp),
            "sf_policy_label_smooth": float(args.recommended_sf_policy_label_smooth),
            "sf_skill_level": int(current_sf_skill_level) if current_sf_skill_level is not None else None,
            "opponent_random_move_prob": float(current_opponent_random_move_prob),
            "temperature": float(args.recommended_temperature),
            "temperature_decay_start_move": int(args.recommended_temperature_decay_start_move),
            "temperature_decay_moves": int(args.recommended_temperature_decay_moves),
            "temperature_endgame": float(args.recommended_temperature_endgame),
        }

        manifest = {
            "server_time_unix": int(time.time()),
            "protocol_version": int(PROTOCOL_VERSION),
            "server_version": str(PACKAGE_VERSION),
            "min_worker_version": str(args.min_worker_version),
            "trial_id": trial_id or None,
            "trainer_step": trainer_step,
            "task": {"type": "selfplay"},
            "recommended_worker": recommended_worker,
            "encoding": {
                "input_planes": 146,
                "policy_size": int(POLICY_SIZE),
                "policy_encoding": "lc0_4672",
            },
            "model": {
                "sha256": str(model_sha),
                "endpoint": api_prefix + "/model",
                "filename": "latest_model.pt",
                "format": "torch_state_dict",
            },
            "model_config": {
                "kind": str(args.model),
                "embed_dim": int(args.embed_dim),
                "num_layers": int(args.num_layers),
                "num_heads": int(args.num_heads),
                "ffn_mult": int(args.ffn_mult),
                "use_smolgen": not bool(args.no_smolgen),
                "use_nla": bool(args.use_nla),
                "use_qk_rmsnorm": bool(args.use_qk_rmsnorm),
                "gradient_checkpointing": bool(args.gradient_checkpointing),
            },
            "data": {
                "replay_size": int(len(buf)),
                "shards_ingested": 0,
                "positions_ingested": 0,
            },
            "train": {
                "loss": None,
                "policy_loss": None,
                "wdl_loss": None,
            },
        }

        if best_model_sha:
            manifest["best_model"] = {
                "sha256": str(best_model_sha),
                "endpoint": api_prefix + "/best_model",
                "filename": "best_model.pt",
                "format": "torch_state_dict",
            }

        # Optional: stockfish distribution
        if published_stockfish_path is not None and published_stockfish_path.exists():
            manifest["stockfish"] = {
                "endpoint": api_prefix + "/stockfish",
                "filename": published_stockfish_path.name,
                "sha256": _sha256_file(published_stockfish_path),
            }

        # Optional: worker self-update wheel
        if published_worker_wheel_path is not None and published_worker_wheel_path.exists():
            manifest["worker_wheel"] = {
                "endpoint": api_prefix + "/worker_wheel",
                "filename": published_worker_wheel_path.name,
                "sha256": _sha256_file(published_worker_wheel_path),
                "version": str(PACKAGE_VERSION),
            }

        _atomic_write_text(
            publish_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        # Best-effort; the learner will overwrite manifest on the first successful iteration.
        pass

    # Main loop
    while True:
        # Ingest shards
        shards = _iter_shards(inbox_dir)
        ingested = 0
        positions = 0
        iter_wins = 0
        iter_draws = 0
        iter_losses = 0
        for sp in shards[: int(args.max_shards_per_iter)]:
            try:
                shard_arrs, meta = load_npz_arrays(sp)
            except Exception:
                # Bad shard; quarantine
                qdir = processed_dir / "bad"
                qdir.mkdir(parents=True, exist_ok=True)
                sp.replace(qdir / sp.name)
                continue

            shard_positions = int(np.asarray(shard_arrs["x"]).shape[0])
            if shard_positions > 0:
                buf.add_many_arrays(shard_arrs)
                positions += shard_positions
                # Accumulate game-level win/draw/loss for PID from shard metadata.
                # Using game counts (not position counts) keeps min_games_between_adjust
                # meaningful and prevents long draw games from inflating the win-rate EMA.
                _mw = meta.get("wins")
                _md = meta.get("draws")
                _ml = meta.get("losses")
                if _mw is not None and _md is not None and _ml is not None:
                    iter_wins += int(_mw)
                    iter_draws += int(_md)
                    iter_losses += int(_ml)
                else:
                    # Fallback for old shards without game-level metadata: count unique
                    # game outcomes by tracking wdl transitions across positions.
                    wdl = np.asarray(shard_arrs["wdl_target"], dtype=np.int8)
                    if wdl.size > 0:
                        run_starts = np.empty(wdl.shape[0], dtype=bool)
                        run_starts[0] = True
                        if wdl.shape[0] > 1:
                            run_starts[1:] = wdl[1:] != wdl[:-1]
                        run_values = wdl[run_starts]
                        iter_wins += int(np.count_nonzero(run_values == 0))
                        iter_draws += int(np.count_nonzero(run_values == 1))
                        iter_losses += int(np.count_nonzero(run_values == 2))

            # Move to processed/<user>/
            rel = sp.relative_to(inbox_dir)
            out = processed_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            sp.replace(out)
            ingested += 1

        # Run PID on positions ingested this iteration.
        if pid is not None and (iter_wins + iter_draws + iter_losses) > 0:
            pid_update = pid.observe(wins=iter_wins, draws=iter_draws, losses=iter_losses)
            current_sf_nodes = pid.nodes
            current_sf_skill_level = pid.skill_level
            current_opponent_random_move_prob = float(pid.random_move_prob)
            trainer_step_now = int(getattr(trainer, "step", 0))
            trainer.writer.add_scalar("difficulty/sf_nodes", float(current_sf_nodes), trainer_step_now)
            trainer.writer.add_scalar("difficulty/pid_ema_winrate", float(pid_update.ema_winrate), trainer_step_now)
            trainer.writer.add_scalar("difficulty/skill_level", float(current_sf_skill_level), trainer_step_now)
            trainer.writer.add_scalar(
                "difficulty/opponent_random_move_prob",
                float(current_opponent_random_move_prob),
                trainer_step_now,
            )

            # Ramp down SF bootstrap weights as the opponent gets harder.
            # w_soft (MCTS temperature-2 policy) is NOT ramped — independent of opponent strength.
            _ramp = _sf_bootstrap_ramp_frac(current_opponent_random_move_prob)
            trainer.w_sf_wdl = _lerp(_w_sf_wdl_start, _w_sf_wdl_end, _ramp)
            trainer.writer.add_scalar("loss_weights/w_sf_wdl", trainer.w_sf_wdl, trainer_step_now)

            # Persist so restarts resume from same difficulty.
            try:
                pid_state_path.write_text(
                    json.dumps(
                        {
                            "nodes": int(pid.nodes),
                            "skill_level": int(pid.skill_level),
                            "random_move_prob": float(pid.random_move_prob),
                            "ema_winrate": pid.ema_winrate,
                            "integral": float(pid._integral),
                            "updated_at_unix": int(time.time()),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        if len(buf) < int(args.min_replay_size):
            time.sleep(float(args.sleep_seconds))
            continue

        in_arena = bool(args.arena_enabled) and bool(arena_active)

        # Defaults for manifest train metrics (updated if we actually train this loop)
        train_loss = last_train.get("loss")
        train_policy_loss = last_train.get("policy_loss")
        train_wdl_loss = last_train.get("wdl_loss")

        model_path = publish_dir / "latest_model.pt"

        if in_arena:
            # Ingest arena results; do NOT train or touch model weights on disk.
            results = _iter_arena_results(arena_inbox_dir)
            for rp in results[: int(args.max_shards_per_iter)]:
                try:
                    payload = json.loads(rp.read_text(encoding="utf-8"))
                except Exception:
                    qdir = arena_processed_dir / "bad"
                    qdir.mkdir(parents=True, exist_ok=True)
                    rp.replace(qdir / rp.name)
                    continue

                try:
                    a_sha = str(payload.get("a_sha256") or "")
                    b_sha = str(payload.get("b_sha256") or "")
                    games = int(payload.get("games") or 0)
                    a_win = int(payload.get("a_win") or 0)
                    a_draw = int(payload.get("a_draw") or 0)
                    a_loss = int(payload.get("a_loss") or 0)
                except Exception:
                    a_sha = b_sha = ""
                    games = 0
                    a_win = a_draw = a_loss = 0

                if a_sha == arena_latest_sha and b_sha == arena_best_sha and games > 0 and (a_win + a_draw + a_loss == games):
                    arena_games_done += int(games)
                    arena_a_win += int(a_win)
                    arena_a_draw += int(a_draw)
                    arena_a_loss += int(a_loss)

                    # Log each uploaded batch (makes match progress visible in server logs).
                    denom = float(max(1, arena_games_done))
                    winrate = (float(arena_a_win) + 0.5 * float(arena_a_draw)) / denom
                    log.info(
                        "arena batch: games_done=%d/%d W/D/L=%d/%d/%d winrate=%.3f latest=%s best=%s",
                        int(arena_games_done),
                        int(args.arena_games_target),
                        int(arena_a_win),
                        int(arena_a_draw),
                        int(arena_a_loss),
                        float(winrate),
                        str(arena_latest_sha)[:8],
                        str(arena_best_sha)[:8],
                    )

                try:
                    rel = rp.relative_to(arena_inbox_dir)
                    out = arena_processed_dir / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    rp.replace(out)
                except Exception:
                    rp.unlink(missing_ok=True)

            # Completion / gating
            denom = float(max(1, arena_games_done))
            winrate = (float(arena_a_win) + 0.5 * float(arena_a_draw)) / denom

            if arena_games_done >= int(args.arena_games_target):
                accepted = bool(winrate >= float(args.arena_accept_winrate))
                arena_last_winrate = float(winrate)
                arena_last_accepted = bool(accepted)

                log.info(
                    "arena decision: %s (accept_winrate=%.3f) final W/D/L=%d/%d/%d winrate=%.3f latest=%s best=%s",
                    "ACCEPT" if accepted else "REJECT",
                    float(args.arena_accept_winrate),
                    int(arena_a_win),
                    int(arena_a_draw),
                    int(arena_a_loss),
                    float(winrate),
                    str(arena_latest_sha)[:8],
                    str(arena_best_sha)[:8],
                )

                if accepted:
                    # Promote challenger -> champion
                    if model_path.exists():
                        _atomic_copy2(model_path, best_model_path)
                        best_model_sha = _sha256_file(best_model_path)
                        # Save champion trainer state for potential rollback.
                        trainer.save(champion_ckpt_path)
                else:
                    # Revert to champion if available (discard challenger).
                    if champion_ckpt_path.exists():
                        trainer.load(champion_ckpt_path)
                        trainer.save(ckpt_path)
                        trainer.export_swa(model_path)

                # End arena
                arena_active = False
                arena_latest_sha = ""
                arena_best_sha = ""
                arena_games_done = 0
                arena_a_win = arena_a_draw = arena_a_loss = 0

            # Persist arena state
            arena_state_path.write_text(
                json.dumps(
                    {
                        "active": bool(arena_active),
                        "latest_sha": str(arena_latest_sha),
                        "best_sha": str(arena_best_sha),
                        "games_done": int(arena_games_done),
                        "a_win": int(arena_a_win),
                        "a_draw": int(arena_a_draw),
                        "a_loss": int(arena_a_loss),
                        "last_winrate": arena_last_winrate,
                        "last_accepted": arena_last_accepted,
                        "updated_at_unix": int(time.time()),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            # Update shas for manifest
            model_sha = _sha256_file(model_path) if model_path.exists() else ""
            best_model_sha = _sha256_file(best_model_path) if best_model_path.exists() else ""

        else:
            # Train a bit
            metrics = trainer.train_steps(buf, batch_size=int(args.batch_size), steps=int(args.train_steps))
            trainer.save(ckpt_path)

            train_loss = float(metrics.loss)
            train_policy_loss = float(metrics.policy_loss)
            train_wdl_loss = float(metrics.wdl_loss)
            last_train_path.write_text(
                json.dumps(
                    {
                        "loss": float(train_loss),
                        "policy_loss": float(train_policy_loss),
                        "wdl_loss": float(train_wdl_loss),
                        "updated_at_unix": int(time.time()),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            last_train.update({"loss": train_loss, "policy_loss": train_policy_loss, "wdl_loss": train_wdl_loss})

            # Best-by-loss tracking (local only)
            cur_loss = float(metrics.loss)
            if cur_loss < best_loss - 1e-12:
                best_loss = cur_loss
                best_loss_trainer_step = int(getattr(trainer, "step", 0))
                trainer.save(best_loss_ckpt_path)
                trainer.export_swa(best_loss_model_path)
                best_loss_state_path.write_text(
                    json.dumps(
                        {
                            "best_loss": float(best_loss),
                            "trainer_step": int(best_loss_trainer_step),
                            "updated_at_unix": int(time.time()),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

            # Publish latest model weights
            trainer.export_swa(model_path)
            model_sha = _sha256_file(model_path)

            # Initialize champion model if missing.
            if not best_model_path.exists():
                shutil.copy2(model_path, best_model_path)
                best_model_sha = _sha256_file(best_model_path)
                trainer.save(champion_ckpt_path)
            else:
                best_model_sha = _sha256_file(best_model_path)

            # Start arena for this challenger (once), throttled by --arena-every-n-steps.
            _arena_step = int(getattr(trainer, "step", 0))
            _arena_due = (_arena_step % int(args.arena_every_n_steps)) < int(args.train_steps)
            if bool(args.arena_enabled) and (not arena_active) and best_model_sha and model_sha and best_model_sha != model_sha and _arena_due:
                arena_active = True
                arena_latest_sha = str(model_sha)
                arena_best_sha = str(best_model_sha)
                arena_games_done = 0
                arena_a_win = arena_a_draw = arena_a_loss = 0

                log.info(
                    "arena start: games_target=%d accept_winrate=%.3f latest=%s best=%s",
                    int(args.arena_games_target),
                    float(args.arena_accept_winrate),
                    str(arena_latest_sha)[:8],
                    str(arena_best_sha)[:8],
                )
                arena_state_path.write_text(
                    json.dumps(
                        {
                            "active": True,
                            "latest_sha": str(arena_latest_sha),
                            "best_sha": str(arena_best_sha),
                            "games_done": 0,
                            "a_win": 0,
                            "a_draw": 0,
                            "a_loss": 0,
                            "updated_at_unix": int(time.time()),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

        trainer_step = int(getattr(trainer, "step", 0))
        max_sims = int(args.recommended_mcts_simulations)
        reco_sims = max_sims
        if bool(getattr(args, "recommended_progressive_mcts", True)):
            reco_sims = progressive_mcts_simulations(
                trainer_step,
                start=int(args.recommended_mcts_start_simulations),
                max_sims=max_sims,
                ramp_steps=int(args.recommended_mcts_ramp_steps),
                exponent=float(args.recommended_mcts_ramp_exponent),
            )

        recommended_worker = {
            "games_per_batch": int(args.recommended_games_per_batch),
            "max_plies": _compute_max_plies(trainer_step),
            "mcts": str(args.recommended_mcts),
            "mcts_simulations": int(reco_sims),
            "playout_cap_fraction": float(args.recommended_playout_cap_fraction),
            "fast_simulations": int(args.recommended_fast_simulations),
            "opponent_random_move_prob": float(current_opponent_random_move_prob),
            "opening_book_max_plies": int(args.recommended_opening_book_max_plies),
            "opening_book_max_games": int(args.recommended_opening_book_max_games),
            "opening_book_prob": float(args.recommended_opening_book_prob),
            "random_start_plies": int(args.recommended_random_start_plies),
            "sf_nodes": int(current_sf_nodes),
            "sf_multipv": int(args.recommended_sf_multipv),
            "sf_policy_temp": float(args.recommended_sf_policy_temp),
            "sf_policy_label_smooth": float(args.recommended_sf_policy_label_smooth),
            "sf_skill_level": int(current_sf_skill_level) if current_sf_skill_level is not None else None,
            "temperature": float(args.recommended_temperature),
            "temperature_decay_start_move": int(args.recommended_temperature_decay_start_move),
            "temperature_decay_moves": int(args.recommended_temperature_decay_moves),
            "temperature_endgame": float(args.recommended_temperature_endgame),
        }

        task: dict[str, object]
        if bool(args.arena_enabled) and arena_active and arena_latest_sha and arena_best_sha:
            task = {
                "type": "arena",
                "arena": {
                    "batch_games": int(args.arena_batch_games),
                    "max_plies": int(args.arena_max_plies),
                    "mcts": str(args.arena_mcts),
                    "mcts_simulations": int(args.arena_mcts_simulations),
                    "c_puct": float(args.arena_c_puct),
                    "swap_sides": bool(args.arena_swap_sides),
                    "temperature": float(args.arena_temperature),
                    "random_start_plies": int(args.arena_random_start_plies),
                    "a_sha256": str(arena_latest_sha),
                    "b_sha256": str(arena_best_sha),
                    "games_target": int(args.arena_games_target),
                },
            }
        else:
            task = {"type": "selfplay"}

        manifest = {
            "server_time_unix": int(time.time()),
            "protocol_version": int(PROTOCOL_VERSION),
            "server_version": str(PACKAGE_VERSION),
            "min_worker_version": str(args.min_worker_version),
            "trial_id": trial_id or None,
            "trainer_step": trainer_step,
            "task": task,
            "recommended_worker": recommended_worker,
            "encoding": {
                "input_planes": 146,
                "policy_size": int(POLICY_SIZE),
                "policy_encoding": "lc0_4672",
            },
            "model": {
                "sha256": model_sha,
                "endpoint": api_prefix + "/model",
                "filename": "latest_model.pt",
                "format": "torch_state_dict",
            },
            "model_config": {
                "kind": str(args.model),
                "embed_dim": int(args.embed_dim),
                "num_layers": int(args.num_layers),
                "num_heads": int(args.num_heads),
                "ffn_mult": int(args.ffn_mult),
                "use_smolgen": not bool(args.no_smolgen),
                "use_nla": bool(args.use_nla),
                "use_qk_rmsnorm": bool(args.use_qk_rmsnorm),
                "gradient_checkpointing": bool(args.gradient_checkpointing),
            },
            "data": {
                "replay_size": int(len(buf)),
                "shards_ingested": int(ingested),
                "positions_ingested": int(positions),
            },
            "train": {
                "loss": None if train_loss is None else float(train_loss),
                "policy_loss": None if train_policy_loss is None else float(train_policy_loss),
                "wdl_loss": None if train_wdl_loss is None else float(train_wdl_loss),
            },
        }

        if best_loss < float("inf") and best_loss_trainer_step >= 0:
            manifest["best_loss"] = {
                "loss": float(best_loss),
                "trainer_step": int(best_loss_trainer_step),
            }

        if best_model_sha:
            manifest["best_model"] = {
                "sha256": str(best_model_sha),
                "endpoint": api_prefix + "/best_model",
                "filename": "best_model.pt",
                "format": "torch_state_dict",
            }

        if bool(args.arena_enabled):
            manifest["arena"] = {
                "active": bool(arena_active),
                "latest_sha256": str(arena_latest_sha),
                "best_sha256": str(arena_best_sha),
                "games_done": int(arena_games_done),
                "games_target": int(args.arena_games_target),
                "a_win": int(arena_a_win),
                "a_draw": int(arena_a_draw),
                "a_loss": int(arena_a_loss),
                "last_winrate": arena_last_winrate,
                "last_accepted": arena_last_accepted,
                "accept_winrate": float(args.arena_accept_winrate),
            }

        if args.opening_book_path:
            p = Path(args.opening_book_path)
            if p.exists():
                manifest["opening_book"] = {
                    "endpoint": "/v1/opening_book",
                    "filename": p.name,
                    "sha256": _sha256_file(p),
                }

        # Optional: stockfish distribution
        if published_stockfish_path is not None and published_stockfish_path.exists():
            manifest["stockfish"] = {
                "endpoint": api_prefix + "/stockfish",
                "filename": published_stockfish_path.name,
                "sha256": _sha256_file(published_stockfish_path),
            }

        # Optional: worker self-update wheel
        if published_worker_wheel_path is not None and published_worker_wheel_path.exists():
            manifest["worker_wheel"] = {
                "endpoint": api_prefix + "/worker_wheel",
                "filename": published_worker_wheel_path.name,
                "sha256": _sha256_file(published_worker_wheel_path),
                "version": str(PACKAGE_VERSION),
            }

        _atomic_write_text(
            publish_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
