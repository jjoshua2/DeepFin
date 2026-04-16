from __future__ import annotations

# Optional dependency module (Ray Tune). Kept import-light so the core package
# works without installing `.[tune]`.

from pathlib import Path
import csv
import dataclasses
import json
import logging
import math
import shutil
import subprocess
import time

log = logging.getLogger(__name__)

import numpy as np
import torch

from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
    reinit_volatility_head_parameters_,
    zero_policy_head_parameters_,
)
from chess_anti_engine.replay import ArrayReplayBuffer, DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    copy_or_link_shard,
    iter_shard_paths,
    load_shard_arrays,
    local_iter_shard_path,
    save_local_shard_arrays,
    samples_to_arrays,
)
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.manager import _effective_curriculum_topk
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.budget import progressive_mcts_simulations
from chess_anti_engine.stockfish import DifficultyPID, StockfishPool, StockfishUCI, pid_from_config
from chess_anti_engine.train import Trainer, trainer_kwargs_from_config
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS
from chess_anti_engine.tune.trial_config import DriftMetrics, PidResult, RestoreResult, SelfplayResult, TrainingResult, TrialConfig
from chess_anti_engine.tune._utils import (
    concat_array_batches as _concat_array_batches,
    resolve_local_override_root as _resolve_local_override_root,
    stable_seed_u32 as _stable_seed_u32,
)
from chess_anti_engine.tune.distributed_runtime import (
    _ensure_distributed_workers,
    _ensure_inference_broker,
    _ingest_distributed_selfplay,
    _prune_processed_shards,
    _publish_distributed_trial_state,
    _quarantine_inbox_shards,
    _set_active_run_prefix,
    _stop_process,
    _stop_worker_processes,
    _trial_server_dirs,
)
from chess_anti_engine.tune.recovery import (
    _load_salvage_manifest_entry,
    _merge_pid_state_from_result_row,
    _select_salvage_seed_slot,
)
from chess_anti_engine.tune.replay_exchange import (
    _latest_trial_result_row,
    _refresh_replay_shards_on_exploit,
    _share_top_replay_each_iteration,
    _trial_replay_shard_dir,
)



def _count_jsonl_rows(path: Path) -> int:
    """Count non-empty rows in a JSONL file (best effort)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0


def _iteration_pause_metrics(
    *,
    iteration_started_at: float,
    iteration_finished_at: float,
    pause_started_at: float | None,
    pause_active: bool,
) -> dict[str, float]:
    """Compute how much of an iteration was spent with selfplay paused."""
    elapsed_s = max(0.0, float(iteration_finished_at) - float(iteration_started_at))
    paused_s = 0.0
    if pause_active and pause_started_at is not None:
        paused_s = max(0.0, float(iteration_finished_at) - max(float(iteration_started_at), float(pause_started_at)))
    paused_fraction = 0.0
    if elapsed_s > 0.0:
        paused_fraction = min(1.0, max(0.0, paused_s / elapsed_s))
    return {
        "iteration_elapsed_s": float(elapsed_s),
        "paused_seconds": float(paused_s),
        "paused_fraction": float(paused_fraction),
        "paused_percent": float(paused_fraction * 100.0),
    }


def _compute_train_step_budget(
    *,
    positions_added: int,
    imported_samples: int,
    replay_size: int,
    batch_size: int,
    accum_steps: int,
    base_max_steps: int,
    train_window_fraction: float,
) -> dict[str, int]:
    effective_batch_size = max(1, int(batch_size) * max(1, int(accum_steps)))
    window_target_samples = int(math.ceil(float(train_window_fraction) * max(0, int(replay_size))))
    target_sample_budget = max(
        int(positions_added) + int(imported_samples),
        int(window_target_samples),
    )
    target_steps = max(1, int(math.ceil(float(target_sample_budget) / float(effective_batch_size))))
    if int(imported_samples) > 0:
        steps = int(target_steps)
    else:
        steps = min(int(target_steps), max(1, int(base_max_steps)))
    return {
        "steps": int(steps),
        "target_sample_budget": int(target_sample_budget),
        "window_target_samples": int(window_target_samples),
    }

def _resolve_pause_marker_path(*, tc: TrialConfig, trial_dir: Path) -> Path:
    tune_root = trial_dir.parent
    raw_work_dir = tc.work_dir
    if raw_work_dir and raw_work_dir.strip():
        tune_root = Path(raw_work_dir.strip()).expanduser()
        if not tune_root.is_absolute():
            tune_root = Path.cwd() / tune_root
    raw = tc.pause_file
    if raw and raw.strip():
        p = Path(raw.strip())
        if not p.is_absolute():
            p = tune_root / p
        return p
    return tune_root / "pause.txt"


def _wait_if_paused(
    *,
    pause_marker_path: Path,
    poll_seconds: int,
    trial_id: str,
    iteration: int,
) -> None:
    poll_s = max(1, int(poll_seconds))
    announced = False
    while pause_marker_path.exists():
        if not announced:
            print(
                f"[trial] pause marker detected: {pause_marker_path} "
                f"(trial={trial_id}, next_iter={iteration})"
            )
            announced = True
        time.sleep(float(poll_s))
    if announced:
        print(
            f"[trial] pause marker cleared: {pause_marker_path} "
            f"(trial={trial_id}, resuming_iter={iteration})"
        )

# Default weights for regret-only mode (rmp/topk pinned).
# When regret is disabled, rmp/topk weights are restored automatically.
_W_RMP = 0.0
_W_TOPK = 0.0
_W_REGRET = 350.0
_W_NODES = 100.0
_W_SKILL = 50.0


def _opponent_strength(
    *,
    random_move_prob: float,
    sf_nodes: int,
    skill_level: int,
    ema_winrate: float,
    min_nodes: int,
    max_nodes: int,
    pid_target_winrate: float = 0.60,
    wdl_regret: float = -1.0,
    wdl_regret_max: float = 1.0,
    topk: int = -1,
    topk_max: int = 12,
    topk_min: int = 2,
) -> float:
    """Composite metric: weighted sum of normalised difficulty factors.

    Each factor maps to 0.0 (easiest) → 1.0 (hardest), then scaled by its
    weight.  All factors contribute independently — no sequential stages.

    Winrate scaling with floor: penalises below-target winrate by at most 50%
    to avoid death spirals after exploit, while still ranking trials that are
    losing worse than those that are winning at equal difficulty.

    Factors & weights (total max ≈ 500):
      wdl_regret        ×350   — primary difficulty lever (regret-only mode)
      sf_nodes          ×100   — search depth (log-scaled, secondary)
      skill_level        ×50   — usually pinned at 20
      random_move_prob    ×0   — pinned at 0.01, not a lever
      topk                ×0   — pinned at multipv, not a lever

    Final score multiplied by min(1, ema_winrate / target) to penalise
    PID overshoot without rewarding above-target winrate.
    """
    rand_prob = float(random_move_prob)
    nodes = int(sf_nodes)
    skill = int(skill_level)
    min_nodes = int(min_nodes)
    max_nodes = int(max_nodes)

    # rmp: 1.0→0.0 maps to 0→1
    rmp_score = 1.0 - max(0.0, min(1.0, rand_prob))

    # topk: max→min maps to 0→1
    k = int(topk)
    k_max = max(1, int(topk_max))
    k_min = max(1, int(topk_min))
    if k < 0 or k_max <= k_min:
        topk_score = 1.0  # not available — assume full difficulty
    else:
        k = max(k_min, min(k_max, k))
        topk_score = 1.0 - (k - k_min) / max(1, k_max - k_min)

    # regret: max→0 maps to 0→1 (negative = disabled = no contribution)
    regret_enabled = float(wdl_regret) >= 0.0
    if not regret_enabled:
        regret_score = 0.0
    else:
        r_max = max(0.001, float(wdl_regret_max))
        regret_score = 1.0 - max(0.0, min(1.0, float(wdl_regret) / r_max))

    # nodes: log-scaled, min→max maps to 0→1
    if min_nodes < max_nodes and nodes > 0:
        log_frac = (math.log(max(nodes, min_nodes)) - math.log(max(1, min_nodes))) / (
            math.log(max(1, max_nodes)) - math.log(max(1, min_nodes))
        )
        nodes_score = max(0.0, min(1.0, log_frac))
    else:
        nodes_score = 0.0

    # skill: 0→20 maps to 0→1
    skill_score = max(0.0, min(1.0, float(skill) / 20.0))

    # When regret is disabled, restore rmp/topk weights so the old
    # multi-stage curriculum is reflected in opponent_strength.
    w_rmp = 200.0 if not regret_enabled else _W_RMP
    w_topk = 150.0 if not regret_enabled else _W_TOPK
    w_regret = _W_REGRET if regret_enabled else 0.0

    difficulty = (
        w_rmp * rmp_score
        + w_topk * topk_score
        + w_regret * regret_score
        + _W_NODES * nodes_score
        + _W_SKILL * skill_score
    )

    # Winrate scaling with floor: cap penalty at 50% so a single bad batch
    # can't crater the metric (old death spiral), but still penalise trials
    # that are consistently losing at their difficulty level.
    target = max(0.01, float(pid_target_winrate))
    winrate_factor = max(0.5, min(1.0, max(0.0, float(ema_winrate)) / target))

    return difficulty * winrate_factor


def _should_retry_distributed_iteration_without_games(
    *,
    use_distributed_selfplay: bool,
    total_games_generated: int,
) -> bool:
    """Return True when a distributed iteration should wait for fresh selfplay."""
    return bool(use_distributed_selfplay) and int(total_games_generated) <= 0


def _play_batch_kwargs(tc: TrialConfig) -> dict:
    """Extract all config-driven play_batch kwargs as dataclass instances.

    Callers (selfplay, gate, eval) use dataclasses.replace() for per-site overrides.
    This is the single source of truth for config → play_batch mapping.
    """
    return dict(
        opponent=OpponentConfig(
            topk_stage_end=tc.sf_pid_topk_stage_end,
            topk_min=tc.sf_pid_topk_min,
            suboptimal_wdl_regret_max=tc.sf_pid_suboptimal_wdl_regret_max,
            suboptimal_wdl_regret_min=tc.sf_pid_suboptimal_wdl_regret_min,
            random_move_prob_start=tc.sf_pid_random_move_prob_start,
            random_move_prob_min=tc.sf_pid_random_move_prob_min,
        ),
        temp=TemperatureConfig(
            temperature=tc.temperature,
            drop_plies=tc.temperature_drop_plies,
            after=tc.temperature_after,
            decay_start_move=tc.temperature_decay_start_move,
            decay_moves=tc.temperature_decay_moves,
            endgame=tc.temperature_endgame,
        ),
        search=SearchConfig(
            mcts_type=tc.mcts,
            playout_cap_fraction=tc.playout_cap_fraction,
            fast_simulations=tc.fast_simulations,
            fpu_reduction=tc.fpu_reduction,
            fpu_at_root=tc.fpu_at_root,
        ),
        opening=OpeningConfig(
            opening_book_path=tc.opening_book_path,
            opening_book_max_plies=tc.opening_book_max_plies,
            opening_book_max_games=tc.opening_book_max_games,
            opening_book_prob=tc.opening_book_prob,
            opening_book_path_2=tc.opening_book_path_2,
            opening_book_max_plies_2=tc.opening_book_max_plies_2,
            opening_book_max_games_2=tc.opening_book_max_games_2,
            opening_book_mix_prob_2=tc.opening_book_mix_prob_2,
            random_start_plies=tc.random_start_plies,
        ),
        diff_focus=DiffFocusConfig(
            enabled=tc.diff_focus_enabled,
            q_weight=tc.diff_focus_q_weight,
            pol_scale=tc.diff_focus_pol_scale,
            slope=tc.diff_focus_slope,
            min_keep=tc.diff_focus_min,
        ),
        game=GameConfig(
            max_plies=tc.max_plies,
            selfplay_fraction=tc.selfplay_fraction,
            sf_policy_temp=tc.sf_policy_temp,
            sf_policy_label_smooth=tc.sf_policy_label_smooth,
            soft_policy_temp=tc.soft_policy_temp,
            timeout_adjudication_threshold=tc.timeout_adjudication_threshold,
            volatility_source=tc.volatility_source,
            syzygy_path=tc.syzygy_path,
            syzygy_policy=tc.syzygy_policy,
            categorical_bins=tc.categorical_bins,
            hlgauss_sigma=tc.hlgauss_sigma,
        ),
    )


def _blended_winrate_raw_or_none(
    *,
    wins: int,
    draws: int,
    losses: int,
) -> float | None:
    total_games_played = int(wins) + int(draws) + int(losses)
    if total_games_played <= 0:
        return None
    return (float(wins) + 0.5 * float(draws)) / float(total_games_played)



def _gate_check(
    model: torch.nn.Module,
    *,
    device: str,
    rng: np.random.Generator,
    sf: object,
    gate_games: int,
    opponent_random_move_prob: float,
    tc: TrialConfig,
) -> tuple[float, int, int, int]:
    """Play gate games to measure winrate. Returns (winrate, W, D, L)."""
    kw = _play_batch_kwargs(tc)
    # Gate: exploit mode — low temperature, no playout cap, minimal search.
    kw["opponent"] = dataclasses.replace(kw["opponent"], random_move_prob=opponent_random_move_prob)
    kw["temp"] = TemperatureConfig(temperature=0.3, drop_plies=0, after=0.0, decay_start_move=10, decay_moves=30, endgame=0.1)
    kw["search"] = dataclasses.replace(kw["search"], simulations=tc.gate_mcts_sims, playout_cap_fraction=1.0, fast_simulations=0)
    kw["diff_focus"] = dataclasses.replace(kw["diff_focus"], enabled=False)
    kw["game"] = dataclasses.replace(kw["game"], selfplay_fraction=0.0)
    _gate_samples, gate_stats = play_batch(model, device=device, rng=rng, stockfish=sf, games=gate_games, **kw)
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

def _games_per_iter_for_iteration(tc: TrialConfig, iteration_idx: int) -> int:
    target = max(1, tc.games_per_iter)
    start = tc.games_per_iter_start
    ramp_iters = max(0, tc.games_per_iter_ramp_iters)

    if ramp_iters <= 0 or iteration_idx >= ramp_iters:
        return int(target)

    frac = float(max(0, iteration_idx - 1)) / float(ramp_iters)
    value = float(start) + (float(target) - float(start)) * frac
    return max(1, int(round(value)))


def _run_puzzle_eval_if_due(
    model: torch.nn.Module,
    puzzle_suite,
    *,
    tc: TrialConfig,
    device: str,
    rng,
    iteration_zero_based: int,
) -> dict:
    """Run puzzle evaluation if due this iteration. Returns metrics dict."""
    _defaults = {"puzzle_accuracy": float("nan"), "puzzle_correct": 0, "puzzle_total": 0}
    if puzzle_suite is None or tc.puzzle_interval <= 0:
        return _defaults
    if iteration_zero_based % tc.puzzle_interval != 0:
        return _defaults
    from chess_anti_engine.eval import run_puzzle_eval
    pr = run_puzzle_eval(model, puzzle_suite, device=device, mcts_simulations=tc.puzzle_simulations, rng=rng)
    return {
        "puzzle_accuracy": pr.accuracy,
        "puzzle_correct": pr.correct,
        "puzzle_total": pr.total,
    }


def _write_status_csv_row(
    path: Path,
    *,
    iteration_idx: int,
    opp_strength: float,
    opp_strength_ema: float,
    skill_level: int,
    sf_nodes: int,
    current_rand: float,
    wdl_regret: float,
    ingest_ms: float,
    train_ms: float,
    total_iter_ms: float,
    steps: int,
    replay_size: int,
    positions_ingested: int,
    stale_games: int,
    train_loss: float | None,
    best_loss: float,
    total_w: int,
    total_d: int,
    total_l: int,
    opt_lr: float,
    startup_source: str,
) -> None:
    """Append a compact status CSV row (best-effort)."""
    try:
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([
                int(iteration_idx),
                int(iteration_idx),
                f"{float(opp_strength):.1f}",
                f"{float(opp_strength_ema):.1f}",
                int(skill_level),
                int(sf_nodes),
                f"{float(current_rand):.3f}",
                f"{float(wdl_regret):.4f}",
                f"{float(ingest_ms)/1000:.1f}",
                f"{float(train_ms)/1000:.1f}",
                f"{float(total_iter_ms)/1000:.1f}",
                int(steps),
                int(replay_size),
                int(positions_ingested),
                int(stale_games),
                f"{float(train_loss):.4f}" if train_loss is not None else "",
                f"{float(best_loss):.4f}",
                int(total_w),
                int(total_d),
                int(total_l),
                f"{float(opt_lr):.2e}",
                str(startup_source),
            ])
    except Exception:
        pass


def _save_trial_checkpoint(
    *,
    trainer,
    buf,
    ckpt_dir: Path,
    rng,
    trial_id: str,
    trial_dir: Path,
    config: dict,
    base_seed: int,
    restore: RestoreResult,
    iteration_idx: int,
    Checkpoint,
):
    """Flush replay buffer and save a lightweight checkpoint."""
    buf.flush()
    trainer.save(ckpt_dir / "trainer.pt")
    try:
        (ckpt_dir / "rng_state.json").write_text(
            json.dumps(rng.bit_generator.state, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        pass
    try:
        (ckpt_dir / "trial_meta.json").write_text(
            json.dumps({
                "owner_trial_id": str(trial_id),
                "owner_trial_dir": str(trial_dir.resolve()),
                "optimizer": str(config.get("optimizer", "nadamw")).lower(),
                "base_seed": int(base_seed),
                "active_seed": int(restore.active_seed),
                "startup_source": str(restore.startup_source),
                "salvage_origin_used": bool(restore.salvage_origin_used),
                "salvage_origin_slot": int(restore.salvage_origin_slot),
                "salvage_origin_slots_total": int(restore.salvage_origin_slots_total),
                "salvage_origin_dir": str(restore.salvage_origin_dir),
                "global_iter": int(iteration_idx),
            }, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return Checkpoint.from_directory(str(ckpt_dir))


def _update_best_model(
    *,
    trainer,
    test_metrics,
    train_metrics,
    best_loss: float,
    best_dir: Path,
    best_state_path: Path,
    iteration_idx: int,
    opp_strength_ema: float,
) -> float:
    """Update best model if current loss improved. Returns updated best_loss."""
    cur_loss = (
        float(test_metrics.loss) if test_metrics is not None
        else (float(train_metrics.loss) if train_metrics is not None else float("inf"))
    )
    if cur_loss < best_loss - 1e-12:
        best_loss = cur_loss
        trainer.save(best_dir / "trainer.pt")
        trainer.export_swa(best_dir / "best_model.pt")
        best_state_path.write_text(
            json.dumps({
                "best_loss": float(best_loss),
                "iter": int(iteration_idx),
                "trainer_step": int(getattr(trainer, "step", 0)),
                "source": "test_loss" if test_metrics is not None else "train_loss",
                "opp_strength_ema": float(opp_strength_ema),
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return best_loss


_BEST_REGRET_KEEP = 3  # Keep top-N checkpoints by lowest regret


def _update_best_regret_checkpoints(
    *,
    trainer,
    pid,
    best_regret_dir: Path,
    iteration_idx: int,
    opp_strength_ema: float,
    best_loss: float,
) -> None:
    """Save checkpoint if current regret is in the top-N lowest seen."""
    if pid is None:
        return
    regret = float(pid.wdl_regret)
    if regret < 0:
        return
    ema_wr = float(pid.ema_winrate)
    step = int(getattr(trainer, "step", 0))

    best_regret_dir.mkdir(parents=True, exist_ok=True)

    # Read existing entries
    index_path = best_regret_dir / "index.json"
    entries: list[dict] = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []

    # Check if this regret qualifies
    if len(entries) >= _BEST_REGRET_KEEP:
        worst = max(entries, key=lambda e: e["regret"])
        if regret >= worst["regret"]:
            return  # Not in top-N

    # Save checkpoint
    tag = f"regret_{regret:.4f}_step{step}_iter{iteration_idx}"
    slot_dir = best_regret_dir / tag
    slot_dir.mkdir(parents=True, exist_ok=True)
    try:
        trainer.save(slot_dir / "trainer.pt")
        pid_state = pid.state_dict()
        (slot_dir / "pid_state.json").write_text(
            json.dumps(pid_state, sort_keys=True, indent=2), encoding="utf-8",
        )
        (slot_dir / "meta.json").write_text(
            json.dumps({
                "regret": regret, "step": step, "iter": iteration_idx,
                "ema_winrate": ema_wr, "best_loss": best_loss,
                "opp_strength_ema": opp_strength_ema,
            }, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return

    entries.append({"regret": regret, "step": step, "iter": iteration_idx, "tag": tag})

    # Prune to top-N (lowest regret)
    entries.sort(key=lambda e: e["regret"])
    evicted = entries[_BEST_REGRET_KEEP:]
    entries = entries[:_BEST_REGRET_KEEP]

    for ev in evicted:
        ev_dir = best_regret_dir / ev["tag"]
        if ev_dir.exists():
            try:
                import shutil
                shutil.rmtree(ev_dir)
            except Exception:
                pass

    index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


# Keys that affect broker/worker topology — changing these mid-run requires
# a restart because the broker's shared-memory layout and worker processes
# are configured at launch time.
_TOPOLOGY_KEYS = frozenset({
    # Worker-level keys (workers_per_trial, use_compile, sf_workers, threaded,
    # selfplay_threads) removed — _ensure_distributed_workers spawns new workers
    # with updated config each iteration.
    "distributed_inference_max_batch_per_slot",
    "distributed_inference_batch_wait_ms",
    "distributed_inference_use_compile",
    "distributed_inference_broker_enabled",
    "distributed_inference_shared_broker",
    "num_samples",
    "max_concurrent_trials",
    "gpus_per_trial",
})


def _reload_yaml_into_config(config: dict, yaml_path: str | None) -> None:
    """Overlay YAML values into *config*, preserving PB2-searched keys.

    Topology keys that require a broker/worker restart are detected and
    logged as warnings instead of being silently applied.

    PB2-searched keys are determined from the *existing* config (which has
    the baked-in bounds from trial creation), not from the YAML being loaded.
    This prevents YAML edits from accidentally overriding tuned hyperparams.
    """
    if not yaml_path:
        return
    try:
        from chess_anti_engine.utils import load_yaml_file, flatten_run_config_defaults
        fresh = flatten_run_config_defaults(load_yaml_file(yaml_path))
        # Derive searched keys from the config's own bounds (stable), not YAML.
        searched = {
            k.removeprefix("pb2_bounds_")
            for k in config if k.startswith("pb2_bounds_")
        }
        for k, v in fresh.items():
            if k in searched or k.startswith("pb2_bounds_"):
                continue
            if k in _TOPOLOGY_KEYS and k in config and config[k] != v:
                log.warning(
                    "YAML reload: %s changed (%s -> %s) but requires restart — skipping",
                    k, config[k], v,
                )
                continue
            config[k] = v
    except Exception as exc:
        log.warning("YAML reload failed (%s): %s", yaml_path, exc)


def _sample_drift_arrays(src_buf: object, n: int) -> dict[str, np.ndarray]:
    """Sample x, wdl_target, policy_target arrays for drift computation."""
    if hasattr(src_buf, "sample_batch_arrays"):
        arrs = getattr(src_buf, "sample_batch_arrays")(n, wdl_balance=False)
        return {
            "x": np.asarray(arrs["x"], dtype=np.float32),
            "wdl_target": np.asarray(arrs["wdl_target"], dtype=np.int64),
            "policy_target": np.asarray(arrs["policy_target"], dtype=np.float32),
        }
    samples = getattr(src_buf, "sample_batch")(n, wdl_balance=False)
    return {
        "x": np.stack([s.x for s in samples], axis=0).astype(np.float32, copy=False),
        "wdl_target": np.array([int(getattr(s, "wdl_target", 1)) for s in samples], dtype=np.int64),
        "policy_target": np.stack([s.policy_target for s in samples], axis=0).astype(np.float32, copy=False),
    }


def _mean_entropy(arrs: dict[str, np.ndarray], eps: float = 1e-12) -> float:
    """Mean per-sample policy entropy across the batch."""
    p = np.asarray(arrs["policy_target"], dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return 0.0
    ps = p.sum(axis=1, keepdims=True)
    valid = ps[:, 0] > 0.0
    if not np.any(valid):
        return 0.0
    p = p[valid] / ps[valid]
    ent = -np.sum(p * np.log(p + eps), axis=1)
    return float(np.mean(ent))


def _wdl_hist(arrs: dict[str, np.ndarray]) -> np.ndarray:
    """Normalised WDL histogram from sample arrays."""
    arr = np.asarray(arrs["wdl_target"], dtype=np.int64)
    valid = arr[(arr >= 0) & (arr <= 2)]
    hst = np.bincount(valid, minlength=3).astype(np.float64)
    hst /= max(1.0, float(hst.sum()))
    return hst


def _dynamic_sf_wdl_weight(
    *,
    sf_wdl_start: float,
    sf_wdl_floor: float,
    sf_wdl_floor_at_regret: float,
    sf_wdl_floor_at_rmp: float,
    regret_max: float,
    wdl_regret_used: float,
    current_rand: float,
) -> float | None:
    """Compute the dynamic sf_wdl weight based on difficulty proxy.

    Returns the interpolated weight, or None if sf_wdl_start <= 0 (disabled).
    """
    if sf_wdl_start <= 0:
        return None
    if float(wdl_regret_used) >= 0.0:
        regret = float(wdl_regret_used)
        if regret >= regret_max:
            return sf_wdl_start
        elif regret <= sf_wdl_floor_at_regret:
            return sf_wdl_floor
        else:
            t = (regret - sf_wdl_floor_at_regret) / (regret_max - sf_wdl_floor_at_regret)
            return sf_wdl_floor + t * (sf_wdl_start - sf_wdl_floor)
    else:
        rmp = float(current_rand)
        if rmp >= 1.0:
            return sf_wdl_start
        elif rmp <= sf_wdl_floor_at_rmp:
            return sf_wdl_floor
        else:
            t = (rmp - sf_wdl_floor_at_rmp) / (1.0 - sf_wdl_floor_at_rmp)
            return sf_wdl_floor + t * (sf_wdl_start - sf_wdl_floor)


def _sync_trainer_weights(
    trainer: object,
    config: dict,
    tc: TrialConfig,
    wdl_regret_used: float,
    current_rand: float,
) -> None:
    """Re-read loss weights and LR from config into trainer.

    Called each iteration so PB2 perturbations and live YAML changes
    take effect immediately.
    """
    if "lr" in config:
        trainer.set_peak_lr(float(config["lr"]), rescale_current=True)
    if "cosmos_gamma" in config and hasattr(trainer.opt, "gamma"):
        trainer.opt.gamma = float(config["cosmos_gamma"])
    for wk in ("w_soft", "w_future", "w_wdl", "w_sf_move", "w_sf_eval",
                "w_categorical", "w_volatility", "w_sf_wdl",
                "sf_wdl_conf_power", "sf_wdl_draw_scale"):
        if wk in config:
            setattr(trainer, wk, float(config[wk]))

    cur_sf_wdl = _dynamic_sf_wdl_weight(
        sf_wdl_start=tc.w_sf_wdl,
        sf_wdl_floor=tc.sf_wdl_floor,
        sf_wdl_floor_at_regret=tc.sf_wdl_floor_at_regret,
        sf_wdl_floor_at_rmp=tc.sf_wdl_floor_at,
        regret_max=tc.sf_pid_wdl_regret_max,
        wdl_regret_used=wdl_regret_used,
        current_rand=current_rand,
    )
    if cur_sf_wdl is not None:
        trainer.w_sf_wdl = cur_sf_wdl


def _log_iteration_scalars(
    *,
    writer: object,
    pid_result: PidResult,
    current_rand: float,
    wdl_regret_used: float,
    pause_metrics: dict,
    restore: RestoreResult,
    iteration_step: int,
) -> None:
    """Write per-iteration TensorBoard scalars (best-effort)."""
    try:
        pr = pid_result
        writer.add_scalar("difficulty/opponent_strength", float(pr.opp_strength), iteration_step)
        writer.add_scalar("difficulty/opponent_strength_ema", float(pr.opp_strength_ema), iteration_step)
        writer.add_scalar("difficulty/random_move_prob", float(current_rand), iteration_step)
        writer.add_scalar("difficulty/random_move_prob_next", float(pr.random_move_prob_next), iteration_step)
        writer.add_scalar("difficulty/opponent_topk", float(pr.curr_topk), iteration_step)
        writer.add_scalar("difficulty/pid_ema_winrate", float(pr.pid_ema_wr), iteration_step)
        writer.add_scalar("difficulty/wdl_regret", float(wdl_regret_used), iteration_step)
        writer.add_scalar("difficulty/wdl_regret_next", float(pr.wdl_regret_next), iteration_step)
        if pr.blended_winrate_raw is not None:
            writer.add_scalar("difficulty/blended_winrate_raw", float(pr.blended_winrate_raw), iteration_step)
            writer.add_scalar("selfplay/avg_game_plies", float(pr.avg_game_plies), iteration_step)
            if pr.avg_plies_win > 0:
                writer.add_scalar("selfplay/avg_plies_win", float(pr.avg_plies_win), iteration_step)
            if pr.avg_plies_draw > 0:
                writer.add_scalar("selfplay/avg_plies_draw", float(pr.avg_plies_draw), iteration_step)
            if pr.avg_plies_loss > 0:
                writer.add_scalar("selfplay/avg_plies_loss", float(pr.avg_plies_loss), iteration_step)
            writer.add_scalar("selfplay/adjudication_rate", float(pr.adjudication_rate), iteration_step)
            writer.add_scalar("selfplay/draw_rate", float(pr.draw_rate), iteration_step)
            writer.add_scalar("selfplay/selfplay_adjudication_rate", float(pr.selfplay_adjudication_rate), iteration_step)
            writer.add_scalar("selfplay/selfplay_draw_rate", float(pr.selfplay_draw_rate), iteration_step)
        writer.add_scalar("selfplay/curriculum_adjudication_rate", float(pr.curriculum_adjudication_rate), iteration_step)
        writer.add_scalar("selfplay/curriculum_draw_rate", float(pr.curriculum_draw_rate), iteration_step)
        writer.add_scalar("backpressure/paused_seconds", float(pause_metrics["paused_seconds"]), iteration_step)
        writer.add_scalar("backpressure/paused_fraction", float(pause_metrics["paused_fraction"]), iteration_step)
        writer.add_scalar("backpressure/paused_percent", float(pause_metrics["paused_percent"]), iteration_step)
        writer.add_scalar("meta/salvage_warmstart_used", float(1 if restore.seed_warmstart_used else 0), iteration_step)
        writer.add_scalar("meta/salvage_warmstart_slot", float(restore.seed_warmstart_slot), iteration_step)
    except Exception:
        pass


def _compute_drift_metrics(
    *,
    buf: object,
    holdout_buf: object,
    drift_sample_size: int,
) -> DriftMetrics:
    """Compute drift and data diversity metrics from training and holdout buffers."""
    dm = DriftMetrics()
    eps = 1e-12

    if len(buf) < drift_sample_size:
        return dm, False

    train_batch = _sample_drift_arrays(buf, drift_sample_size)

    dm.data_policy_entropy = _mean_entropy(train_batch)

    # Unique positions (approximate via row-hash).
    train_x = train_batch["x"]
    x_flat = train_x.reshape(train_x.shape[0], -1)
    row_sums = x_flat.view(np.uint32).reshape(x_flat.shape[0], -1).sum(axis=1)
    dm.data_unique_positions = float(np.unique(row_sums).shape[0]) / float(max(1, train_x.shape[0]))

    # WDL balance: entropy of WDL distribution.
    wdl_arr = np.asarray(train_batch["wdl_target"], dtype=np.int64)
    wdl_valid = wdl_arr[(wdl_arr >= 0) & (wdl_arr <= 2)]
    if wdl_valid.size > 0:
        h = np.bincount(wdl_valid, minlength=3).astype(np.float64)
        h /= max(1.0, float(h.sum()))
        dm.data_wdl_balance = float(-np.sum(h * np.log(h + eps)))

    # Drift metrics (train vs holdout).
    if len(holdout_buf) >= drift_sample_size:
        hold_batch = _sample_drift_arrays(holdout_buf, drift_sample_size)
        hold_x = hold_batch["x"]
        dm.drift_input_l2 = float(np.linalg.norm(train_x.mean(axis=0) - hold_x.mean(axis=0)))

        p = _wdl_hist(train_batch)
        q = _wdl_hist(hold_batch)
        m = 0.5 * (p + q)
        dm.drift_wdl_js = float(
            0.5 * np.sum(p * (np.log(p + eps) - np.log(m + eps)))
            + 0.5 * np.sum(q * (np.log(q + eps) - np.log(m + eps)))
        )

        dm.drift_policy_entropy_train = dm.data_policy_entropy
        dm.drift_policy_entropy_holdout = _mean_entropy(hold_batch)
        dm.drift_policy_entropy_diff = float(dm.drift_policy_entropy_train - dm.drift_policy_entropy_holdout)

    return dm


def _run_eval_games(
    *,
    tc: TrialConfig,
    trainer: object,
    device: str,
    rng: np.random.Generator,
    eval_sf: object | None,
) -> dict:
    """Run fixed-strength evaluation games (no training data generated)."""
    eval_dict: dict = {
        "eval_win": 0, "eval_draw": 0, "eval_loss": 0, "eval_winrate": 0.0,
    }
    if tc.eval_games <= 0 or eval_sf is None:
        return eval_dict
    kw = _play_batch_kwargs(tc)
    kw["temp"] = TemperatureConfig(temperature=tc.eval_temperature)
    kw["search"] = dataclasses.replace(
        kw["search"],
        simulations=tc.eval_mcts_simulations,
        playout_cap_fraction=1.0,
        fast_simulations=0,
    )
    kw["game"] = dataclasses.replace(
        kw["game"],
        max_plies=tc.eval_max_plies,
        selfplay_fraction=0.0,
    )
    kw["diff_focus"] = dataclasses.replace(kw["diff_focus"], enabled=False)
    kw["opening"] = OpeningConfig()
    _eval_samples, eval_stats = play_batch(
        trainer.model, device=device, rng=rng, stockfish=eval_sf,
        games=tc.eval_games, **kw,
    )
    denom = float(max(1, eval_stats.w + eval_stats.d + eval_stats.l))
    eval_dict.update({
        "eval_win": eval_stats.w,
        "eval_draw": eval_stats.d,
        "eval_loss": eval_stats.l,
        "eval_winrate": (float(eval_stats.w) + 0.5 * float(eval_stats.d)) / denom,
    })
    return eval_dict


def _run_training_and_gating(
    *,
    tc: TrialConfig,
    trainer,
    buf,
    holdout_buf,
    config: dict,
    model_cfg,
    device: str,
    rng,
    pid,
    sf,
    current_rand: float,
    skill_level_used: int,
    sims: int,
    wdl_regret_used: float,
    total_positions: int,
    imported_samples_this_iter: int,
    gate_match_idx: int,
    gate_state_path: Path,
    use_distributed_selfplay: bool,
    distributed_server_root: str | None,
    distributed_dirs: dict | None,
    iteration_idx: int,
    iteration_zero_based: int,
    trial_id: str,
    startup_source: str,
    salvage_origin_used: bool,
) -> TrainingResult:
    """Compute step budget, run training, net gating, and holdout eval."""
    train_t0 = time.monotonic()
    batch_size = tc.batch_size
    accum_steps = max(1, tc.accum_steps)
    skip_train = len(buf) < batch_size
    steps = 0
    target_sample_budget = 0
    window_target_samples = 0
    gate_passed = True
    metrics = None

    if not skip_train:
        train_budget = _compute_train_step_budget(
            positions_added=int(total_positions),
            imported_samples=int(imported_samples_this_iter),
            replay_size=int(len(buf)),
            batch_size=int(batch_size),
            accum_steps=int(accum_steps),
            base_max_steps=int(tc.train_steps),
            train_window_fraction=float(max(0.0, tc.train_window_fraction)),
        )
        steps = int(train_budget["steps"])
        target_sample_budget = int(train_budget["target_sample_budget"])
        window_target_samples = int(train_budget["window_target_samples"])

        if startup_source == "salvage" and bool(salvage_origin_used):
            if int(iteration_zero_based) < tc.salvage_startup_no_share_iters and tc.salvage_startup_max_train_steps > 0:
                steps = min(steps, tc.salvage_startup_max_train_steps)
            elif (
                tc.salvage_startup_post_share_ramp_iters > 0
                and int(iteration_zero_based) < (tc.salvage_startup_no_share_iters + tc.salvage_startup_post_share_ramp_iters)
                and tc.salvage_startup_post_share_max_train_steps > 0
            ):
                steps = min(steps, tc.salvage_startup_post_share_max_train_steps)

        # Save model state for potential rollback (net gating).
        pre_train_state = None
        if tc.gate_games > 0 and (iteration_zero_based % tc.gate_interval == 0):
            pre_train_state = {
                k: v.clone() for k, v in trainer.model.state_dict().items()
            }

        if tc.distributed_pause_selfplay_during_training and use_distributed_selfplay and distributed_dirs is not None:
            _publish_distributed_trial_state(
                trainer=trainer, config=config, model_cfg=model_cfg,
                server_root=distributed_server_root, trial_id=trial_id,
                training_iteration=int(iteration_idx),
                trainer_step=int(getattr(trainer, "step", 0)),
                sf_nodes=int(pid.nodes) if pid is not None else tc.sf_nodes,
                random_move_prob=float(current_rand),
                skill_level=int(skill_level_used),
                mcts_simulations=int(sims),
                wdl_regret=float(pid.wdl_regret) if pid is not None else -1.0,
                pause_selfplay=True,
                pause_reason="training",
                export_model=False,
            )

        metrics = trainer.train_steps(
            buf,
            batch_size=batch_size,
            steps=steps,
        )

        # Net gating: play gate_games and reject if winrate < threshold.
        if pre_train_state is not None and tc.gate_games > 0:
            gate_wr, gate_w, gate_d, gate_l = _gate_check(
                trainer.model,
                device=device,
                rng=rng,
                sf=sf,
                gate_games=tc.gate_games,
                opponent_random_move_prob=current_rand,
                tc=tc,
            )

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

            try:
                trainer.writer.add_scalar("gate/winrate", float(gate_wr), int(gate_match_idx))
                trainer.writer.add_scalar("gate/win", float(gate_w), int(gate_match_idx))
                trainer.writer.add_scalar("gate/draw", float(gate_d), int(gate_match_idx))
                trainer.writer.add_scalar("gate/loss", float(gate_l), int(gate_match_idx))
                trainer.writer.add_scalar("gate/passed", float(1.0 if gate_wr >= tc.gate_threshold else 0.0), int(gate_match_idx))
            except Exception:
                pass

            if gate_wr < tc.gate_threshold:
                trainer.model.load_state_dict(pre_train_state)
                gate_passed = False

    test_metrics = None
    if len(holdout_buf) >= tc.batch_size:
        test_metrics = trainer.eval_steps(
            holdout_buf,
            batch_size=tc.batch_size,
            steps=tc.test_steps,
        )

    train_ms = (time.monotonic() - train_t0) * 1000.0

    return TrainingResult(
        metrics=metrics,
        test_metrics=test_metrics,
        gate_passed=gate_passed,
        steps=steps,
        target_sample_budget=target_sample_budget,
        window_target_samples=window_target_samples,
        train_ms=train_ms,
        gate_match_idx=gate_match_idx,
    )


def _run_pid_and_eval(
    *,
    tc: TrialConfig,
    pid: DifficultyPID | None,
    sf: object | None,
    sp_result: SelfplayResult,
    iteration_zero_based: int,
    opp_strength_ema: float,
    opp_ema_alpha: float,
    current_rand: float,
    sf_nodes_used: int,
    skill_level_used: int,
    wdl_regret_used: float,
) -> PidResult:
    """Update PID, compute opponent strength and derived game stats.

    Mutates *pid* (observe + param refresh) and *sf* (set_nodes) in place.
    """
    total_w = sp_result.total_w
    total_d = sp_result.total_d
    total_l = sp_result.total_l

    # --- Derived game stats ---
    gen = float(max(1, int(sp_result.total_games_generated)))
    sp = float(max(1, int(sp_result.total_selfplay_games)))
    cur = float(max(1, int(sp_result.total_curriculum_games)))

    blended_winrate_raw = _blended_winrate_raw_or_none(
        wins=total_w, draws=total_d, losses=total_l,
    )

    # --- PID update ---
    pid_update = None
    pid_ema_wr = float(pid.ema_winrate) if pid is not None else 0.0
    sf_nodes_next = int(sf_nodes_used)
    random_move_prob_next = float(current_rand)
    skill_level_next = int(skill_level_used)

    if pid is not None and (total_w + total_d + total_l) > 0:
        _step_start = tc.sf_pid_max_rand_step_start
        _step_ramp = tc.sf_pid_max_rand_step_ramp_iters
        if _step_start > 0 and _step_ramp > 0 and iteration_zero_based < _step_ramp:
            pid.max_rand_step = _step_start
        else:
            pid.max_rand_step = tc.sf_pid_max_rand_step
        pid.max_regret_step = tc.sf_pid_max_regret_step
        pid.max_regret_ease_step = tc.sf_pid_max_regret_ease_step
        pid.target = tc.sf_pid_target_winrate
        pid.kp = tc.sf_pid_kp
        pid.ki = tc.sf_pid_ki
        pid.alpha = tc.sf_pid_ema_alpha
        pid_update = pid.observe(wins=total_w, draws=total_d, losses=total_l, force=True)
        pid_ema_wr = float(pid_update.ema_winrate)
        sf_nodes_next = int(pid.nodes)
        random_move_prob_next = float(pid.random_move_prob)
        skill_level_next = int(pid.skill_level)
        if sf is not None:
            if hasattr(sf, "set_nodes"):
                sf.set_nodes(int(sf_nodes_next))
            else:
                setattr(sf, "nodes", int(sf_nodes_next))

    wdl_regret_next = float(pid.wdl_regret) if pid is not None else -1.0

    # --- Opponent strength ---
    _curr_topk = _effective_curriculum_topk(
        random_move_prob=current_rand,
        stage_end=tc.sf_pid_topk_stage_end,
        topk_max=tc.sf_multipv,
        topk_min=tc.sf_pid_topk_min,
    )
    opp_strength = _opponent_strength(
        random_move_prob=float(current_rand),
        sf_nodes=int(sf_nodes_used),
        skill_level=int(skill_level_used),
        ema_winrate=float(pid_ema_wr),
        min_nodes=int(getattr(pid, "min_nodes", 50)) if pid is not None else 50,
        max_nodes=int(getattr(pid, "max_nodes", 50000)) if pid is not None else 50000,
        pid_target_winrate=tc.sf_pid_target_winrate,
        wdl_regret=float(wdl_regret_used),
        wdl_regret_max=tc.sf_pid_wdl_regret_max,
        topk=int(_curr_topk),
        topk_max=tc.sf_multipv,
        topk_min=tc.sf_pid_topk_min,
    )
    if opp_strength_ema == 0.0:
        opp_strength_ema = float(opp_strength)
    else:
        opp_strength_ema = (
            opp_ema_alpha * float(opp_strength)
            + (1.0 - opp_ema_alpha) * opp_strength_ema
        )

    return PidResult(
        sf_nodes_next=sf_nodes_next,
        random_move_prob_next=random_move_prob_next,
        skill_level_next=skill_level_next,
        wdl_regret_next=wdl_regret_next,
        pid_ema_wr=pid_ema_wr,
        pid_update=pid_update,
        blended_winrate_raw=blended_winrate_raw,
        avg_game_plies=float(sp_result.total_game_plies) / gen,
        adjudication_rate=float(sp_result.total_adjudicated_games) / gen,
        draw_rate=float(sp_result.total_draw_games) / gen,
        selfplay_adjudication_rate=float(sp_result.total_selfplay_adjudicated_games) / sp,
        selfplay_draw_rate=float(sp_result.total_selfplay_draw_games) / sp,
        curriculum_adjudication_rate=float(sp_result.total_curriculum_adjudicated_games) / cur,
        curriculum_draw_rate=float(sp_result.total_curriculum_draw_games) / cur,
        checkmate_rate=float(sp_result.total_checkmate_games) / gen,
        stalemate_rate=float(sp_result.total_stalemate_games) / gen,
        avg_plies_win=float(sp_result.total_plies_win) / float(max(1, total_w)) if total_w > 0 else 0.0,
        avg_plies_draw=float(sp_result.total_plies_draw) / float(max(1, total_d)) if total_d > 0 else 0.0,
        avg_plies_loss=float(sp_result.total_plies_loss) / float(max(1, total_l)) if total_l > 0 else 0.0,
        opp_strength=opp_strength,
        opp_strength_ema=opp_strength_ema,
        curr_topk=int(_curr_topk),
    )


def _run_selfplay_phase(
    *,
    tc: TrialConfig,
    config: dict,
    trainer,
    model_cfg,
    buf,
    holdout_buf,
    holdout_frozen: bool,
    device: str,
    rng,
    sf,
    pid,
    use_distributed_selfplay: bool,
    distributed_dirs: dict | None,
    distributed_server_root: str | None,
    distributed_worker_procs: list,
    distributed_inference_broker_proc,
    prev_published_model_sha: str,
    current_rand: float,
    skill_level_used: int,
    sims: int,
    iteration_idx: int,
    iteration_zero_based: int,
    trial_id: str,
    trial_dir: Path,
    selfplay_shards_dir: Path,
    replay_shard_dir: Path,
    current_window: int,
    in_salvage_startup_grace: bool,
) -> tuple[SelfplayResult, str, int, object | None]:
    """Run selfplay, ingest, export shards, grow window, cross-trial share.

    Mutates *buf*, *holdout_buf*, *distributed_worker_procs* in place.
    Returns ``(sp_result, prev_published_model_sha, current_window,
    distributed_inference_broker_proc)``.
    """
    total_games = _games_per_iter_for_iteration(tc, iteration_idx)

    # --- Play games (distributed or local) ---
    ingest_t0 = time.monotonic()
    _new_selfplay_shards: list = []
    _new_selfplay_batches: list[dict[str, np.ndarray]] = []

    # Accumulators (populated by either branch, bundled into SelfplayResult at end).
    total_w = total_d = total_l = 0
    total_games_generated = 0
    total_game_plies = 0
    total_adjudicated_games = 0
    total_draw_games = 0
    total_selfplay_games = 0
    total_selfplay_adjudicated_games = 0
    total_selfplay_draw_games = 0
    total_curriculum_games = 0
    total_curriculum_adjudicated_games = 0
    total_curriculum_draw_games = 0
    total_checkmate_games = 0
    total_stalemate_games = 0
    total_plies_win = 0
    total_plies_draw = 0
    total_plies_loss = 0
    total_positions = 0
    replay_positions_ingested = 0
    total_sf_d6 = 0.0
    total_sf_d6_n = 0
    distributed_stale_positions = 0
    distributed_stale_games = 0

    if use_distributed_selfplay:
        distributed_worker_procs[:] = _ensure_distributed_workers(
            config=config,
            trial_dir=trial_dir,
            trial_id=trial_id,
            procs=distributed_worker_procs,
        )
        published_model_sha = _publish_distributed_trial_state(
            trainer=trainer,
            config=config,
            model_cfg=model_cfg,
            server_root=distributed_server_root,
            trial_id=trial_id,
            training_iteration=int(iteration_idx),
            trainer_step=int(getattr(trainer, "step", 0)),
            sf_nodes=int(pid.nodes) if pid is not None else tc.sf_nodes,
            random_move_prob=float(current_rand),
            skill_level=int(skill_level_used),
            mcts_simulations=int(sims),
            wdl_regret=float(pid.wdl_regret) if pid is not None else -1.0,
            pause_selfplay=False,
            pause_reason="",
        )
        distributed_inference_broker_proc = _ensure_inference_broker(
            config=config,
            trial_id=trial_id,
            trial_dir=trial_dir,
            publish_dir=distributed_dirs["publish_dir"],
            proc=distributed_inference_broker_proc,
        )
        _shards_before_ingest = set(buf._shard_paths)

        ingest_summary = _ingest_distributed_selfplay(
            buf=buf,
            holdout_buf=holdout_buf,
            holdout_frac=tc.holdout_fraction,
            holdout_frozen=bool(holdout_frozen),
            inbox_dir=distributed_dirs["inbox_dir"],
            processed_dir=distributed_dirs["processed_dir"],
            target_games=int(total_games),
            accepted_model_shas={str(published_model_sha)} | ({str(prev_published_model_sha)} if prev_published_model_sha else set()),
            prev_model_sha=str(prev_published_model_sha) if prev_published_model_sha else None,
            prev_model_max_fraction=tc.distributed_prev_model_max_fraction,
            wait_timeout_s=tc.distributed_wait_timeout_seconds,
            poll_seconds=tc.distributed_worker_poll_seconds,
            rng=rng,
            min_games_fraction=tc.distributed_min_games_fraction,
        )

        buf.flush()
        _new_selfplay_shards = [p for p in buf._shard_paths if p not in _shards_before_ingest]
        total_w = int(ingest_summary["matching_w"])
        total_d = int(ingest_summary["matching_d"])
        total_l = int(ingest_summary["matching_l"])
        total_games_generated = int(ingest_summary["matching_games"])
        total_game_plies = int(ingest_summary["matching_total_game_plies"])
        total_adjudicated_games = int(ingest_summary["matching_adjudicated_games"])
        total_draw_games = int(ingest_summary["matching_total_draw_games"])
        total_selfplay_games = int(ingest_summary["matching_selfplay_games"])
        total_selfplay_adjudicated_games = int(ingest_summary["matching_selfplay_adjudicated_games"])
        total_selfplay_draw_games = int(ingest_summary["matching_selfplay_draw_games"])
        total_curriculum_games = int(ingest_summary["matching_curriculum_games"])
        total_curriculum_adjudicated_games = int(ingest_summary["matching_curriculum_adjudicated_games"])
        total_curriculum_draw_games = int(ingest_summary["matching_curriculum_draw_games"])
        total_positions = int(ingest_summary["matching_positions"])
        total_plies_win = int(ingest_summary.get("matching_plies_win", 0))
        total_plies_draw = int(ingest_summary.get("matching_plies_draw", 0))
        total_plies_loss = int(ingest_summary.get("matching_plies_loss", 0))
        total_checkmate_games = int(ingest_summary.get("matching_checkmate_games", 0))
        total_stalemate_games = int(ingest_summary.get("matching_stalemate_games", 0))
        distributed_stale_positions = int(ingest_summary["stale_positions"])
        distributed_stale_games = int(ingest_summary["stale_games"])
        replay_positions_ingested = int(ingest_summary["positions_replay_added"])
        prev_published_model_sha = str(published_model_sha)

        if iteration_zero_based % 10 == 0:
            _prune_processed_shards(
                processed_dir=distributed_dirs["processed_dir"],
                max_age_seconds=tc.processed_max_age_seconds,
            )
    else:
        kw = _play_batch_kwargs(tc)
        kw["opponent"] = dataclasses.replace(kw["opponent"], random_move_prob=current_rand)
        kw["search"] = dataclasses.replace(kw["search"], simulations=int(sims))

        samples, stats = play_batch(
            trainer.model,
            device=device, rng=rng, stockfish=sf,
            games=tc.selfplay_batch,
            target_games=total_games,
            **kw,
        )

        total_games_generated += int(stats.games)
        total_w += stats.w
        total_d += stats.d
        total_l += stats.l
        total_game_plies += int(getattr(stats, "total_game_plies", 0))
        total_adjudicated_games += int(getattr(stats, "adjudicated_games", getattr(stats, "timeout_games", 0)))
        total_draw_games += int(getattr(stats, "total_draw_games", 0))
        total_selfplay_games += int(getattr(stats, "selfplay_games", 0))
        total_selfplay_adjudicated_games += int(getattr(stats, "selfplay_adjudicated_games", 0))
        total_selfplay_draw_games += int(getattr(stats, "selfplay_draw_games", 0))
        total_curriculum_games += int(getattr(stats, "curriculum_games", 0))
        total_curriculum_adjudicated_games += int(getattr(stats, "curriculum_adjudicated_games", 0))
        total_curriculum_draw_games += int(getattr(stats, "curriculum_draw_games", 0))
        total_checkmate_games += int(getattr(stats, "checkmate_games", 0))
        total_stalemate_games += int(getattr(stats, "stalemate_games", 0))
        total_plies_win += int(getattr(stats, "plies_win", 0))
        total_plies_draw += int(getattr(stats, "plies_draw", 0))
        total_plies_loss += int(getattr(stats, "plies_loss", 0))
        total_positions += stats.positions
        total_sf_d6 += float(getattr(stats, "sf_eval_delta6", 0.0)) * int(getattr(stats, "sf_eval_delta6_n", 0))
        total_sf_d6_n += int(getattr(stats, "sf_eval_delta6_n", 0))
        replay_positions_ingested = int(total_positions)

        holdout_samples: list = []
        train_samples: list = []
        for s in samples:
            if tc.holdout_fraction > 0.0 and (not holdout_frozen) and (rng.random() < tc.holdout_fraction):
                holdout_samples.append(s)
            else:
                train_samples.append(s)
        if holdout_samples:
            holdout_buf.add_many_arrays(samples_to_arrays(holdout_samples))
        if train_samples:
            train_arrs = samples_to_arrays(train_samples)
            buf.add_many_arrays(train_arrs)
            _new_selfplay_batches.append(train_arrs)
        del samples

    ingest_ms = (time.monotonic() - ingest_t0) * 1000.0

    # --- Retry if distributed returned no games ---
    if _should_retry_distributed_iteration_without_games(
        use_distributed_selfplay=use_distributed_selfplay,
        total_games_generated=total_games_generated,
    ):
        print(
            "[trial] distributed iteration waiting for fresh selfplay: "
            f"trial={trial_id} iter={iteration_idx} replay={len(buf)} "
            f"timeout_s={tc.distributed_wait_timeout_seconds:.1f}"
        )
        time.sleep(max(0.5, tc.distributed_worker_poll_seconds))
        return (
            SelfplayResult(should_retry=True, ingest_ms=ingest_ms),
            prev_published_model_sha,
            current_window,
            distributed_inference_broker_proc,
        )

    # --- Export selfplay shards for sibling trials ---
    _selfplay_export_path = local_iter_shard_path(selfplay_shards_dir, iteration_idx)
    if use_distributed_selfplay:
        if _new_selfplay_shards:
            _export_batches: list[dict[str, np.ndarray]] = []
            for _sp_path in _new_selfplay_shards:
                try:
                    _arrs, _ = load_shard_arrays(_sp_path, lazy=True)
                    if int(np.asarray(_arrs["x"]).shape[0]) > 0:
                        _export_batches.append(_arrs)
                except Exception:
                    pass
            if _export_batches:
                save_local_shard_arrays(_selfplay_export_path, arrs=_concat_array_batches(_export_batches))
    else:
        if _new_selfplay_batches:
            save_local_shard_arrays(_selfplay_export_path, arrs=_concat_array_batches(_new_selfplay_batches))
    _keep_selfplay_iters = tc.exploit_replay_max_unseen_iters_per_source + 2
    _all_selfplay_exports = iter_shard_paths(selfplay_shards_dir)
    for _old in _all_selfplay_exports[:-_keep_selfplay_iters]:
        try:
            shutil.rmtree(_old) if _old.is_dir() else _old.unlink(missing_ok=True)
        except Exception:
            pass

    # --- Growing window ---
    if current_window < tc.replay_window_max:
        current_window = min(current_window + tc.replay_window_growth, tc.replay_window_max)
        if buf.capacity < current_window:
            buf.capacity = current_window

    # --- Cross-trial replay sharing ---
    shared_summary: dict = {
        "source_trials_selected": 0,
        "source_trials_ingested": 0,
        "source_trials_skipped_repeat": 0,
        "source_shards_loaded": 0,
        "source_samples_ingested": 0,
    }
    if tc.exploit_replay_share_top_enabled and (not in_salvage_startup_grace):
        shared_summary = _share_top_replay_each_iteration(
            config=config,
            recipient_trial_dir=trial_dir,
            replay_shard_dir=replay_shard_dir,
            buf=buf,
            top_k_trials=tc.exploit_replay_top_k_trials,
            within_best_frac=tc.exploit_replay_top_within_best_frac,
            min_metric=tc.exploit_replay_top_min_metric,
            source_skip_newest=tc.exploit_replay_skip_newest,
            shard_size=tc.shard_size,
            holdout_fraction=tc.holdout_fraction,
            max_unseen_iters_per_source=tc.exploit_replay_max_unseen_iters_per_source,
            max_shards_per_source=tc.exploit_replay_top_shards_per_source,
            share_fraction=tc.exploit_replay_share_fraction,
        )
        if shared_summary["source_samples_ingested"] > 0:
            print(
                "[trial] replay share this iter: "
                f"sources_selected={shared_summary['source_trials_selected']} "
                f"sources_ingested={shared_summary['source_trials_ingested']} "
                f"sources_skipped_repeat={shared_summary['source_trials_skipped_repeat']} "
                f"source_shards_loaded={shared_summary['source_shards_loaded']} "
                f"source_samples_ingested={shared_summary['source_samples_ingested']}"
            )
    elif tc.exploit_replay_share_top_enabled and in_salvage_startup_grace:
        print(
            "[trial] replay share skipped during salvage startup grace: "
            f"iter={iteration_zero_based} grace_iters={tc.salvage_startup_no_share_iters}"
        )

    imported_samples_this_iter = int(shared_summary.get("source_samples_ingested", 0))

    sp = SelfplayResult(
        total_w=total_w, total_d=total_d, total_l=total_l,
        total_games_generated=total_games_generated,
        total_game_plies=total_game_plies,
        total_adjudicated_games=total_adjudicated_games,
        total_draw_games=total_draw_games,
        total_selfplay_games=total_selfplay_games,
        total_selfplay_adjudicated_games=total_selfplay_adjudicated_games,
        total_selfplay_draw_games=total_selfplay_draw_games,
        total_curriculum_games=total_curriculum_games,
        total_curriculum_adjudicated_games=total_curriculum_adjudicated_games,
        total_curriculum_draw_games=total_curriculum_draw_games,
        total_checkmate_games=total_checkmate_games,
        total_stalemate_games=total_stalemate_games,
        total_plies_win=total_plies_win,
        total_plies_draw=total_plies_draw,
        total_plies_loss=total_plies_loss,
        total_positions=total_positions,
        replay_positions_ingested=replay_positions_ingested,
        total_sf_d6=total_sf_d6,
        total_sf_d6_n=total_sf_d6_n,
        distributed_stale_positions=distributed_stale_positions,
        distributed_stale_games=distributed_stale_games,
        shared_summary=shared_summary,
        imported_samples_this_iter=imported_samples_this_iter,
        ingest_ms=ingest_ms,
    )
    return sp, prev_published_model_sha, current_window, distributed_inference_broker_proc


def _finalize_iteration(
    *,
    tc: TrialConfig,
    trainer,
    pid,
    sp: SelfplayResult,
    tr: TrainingResult,
    drift: DriftMetrics,
    pid_result: PidResult,
    eval_dict: dict,
    checkpoint,
    best_loss: float,
    ckpt_dir: Path,
    work_dir: Path,
    trial_dir: Path,
    status_csv_path: Path,
    tune_report_fn,
    puzzle_suite,
    current_rand: float,
    wdl_regret_used: float,
    sf_nodes_used: int,
    skill_level_used: int,
    use_distributed_selfplay: bool,
    distributed_pause_started_at: float | None,
    distributed_pause_active: bool,
    restore: RestoreResult,
    holdout_frozen: bool,
    holdout_generation: int,
    buf_size: int,
    holdout_buf_size: int,
    iter_t0: float,
    iteration_idx: int,
    iteration_zero_based: int,
    completed_iterations: int,
    device: str,
    rng,
) -> None:
    """End-of-iteration bookkeeping: PID persist, reporting, CSV, prune."""
    # Persist PID state AFTER observe() so checkpoints carry the
    # post-iteration difficulty that produced random_move_prob_next.
    if pid is not None:
        try:
            (ckpt_dir / "pid_state.json").write_text(
                json.dumps(pid.state_dict(), sort_keys=True, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # Save top-3 checkpoints by lowest regret (best-effort).
    try:
        _update_best_regret_checkpoints(
            trainer=trainer, pid=pid,
            best_regret_dir=work_dir / "best_regret",
            iteration_idx=iteration_idx,
            opp_strength_ema=pid_result.opp_strength_ema,
            best_loss=best_loss,
        )
    except Exception:
        pass

    # Puzzle evaluation (overspecialization canary).
    puzzle_dict = _run_puzzle_eval_if_due(
        trainer.model, puzzle_suite,
        tc=tc, device=device, rng=rng,
        iteration_zero_based=iteration_zero_based,
    )

    pause_metrics = _iteration_pause_metrics(
        iteration_started_at=iter_t0,
        iteration_finished_at=time.monotonic(),
        pause_started_at=distributed_pause_started_at,
        pause_active=distributed_pause_active,
    )
    _log_iteration_scalars(
        writer=trainer.writer,
        pid_result=pid_result,
        current_rand=current_rand,
        wdl_regret_used=wdl_regret_used,
        pause_metrics=pause_metrics,
        restore=restore,
        iteration_step=int(iteration_idx),
    )

    report_dict = _build_report_dict(
        tc=tc, trainer=trainer,
        pr=pid_result, sp=sp, tr=tr, drift=drift,
        eval_dict=eval_dict, puzzle_dict=puzzle_dict,
        current_rand=current_rand,
        wdl_regret_used=wdl_regret_used,
        sf_nodes_used=sf_nodes_used,
        skill_level_used=skill_level_used,
        use_distributed_selfplay=use_distributed_selfplay,
        pause_metrics=pause_metrics,
        restore=restore,
        best_loss=best_loss,
        iter_t0=iter_t0,
        iteration_idx=iteration_idx,
        buf_size=buf_size,
        holdout_buf_size=holdout_buf_size,
        holdout_frozen=holdout_frozen,
        holdout_generation=holdout_generation,
    )

    tune_report_fn(report_dict, checkpoint=checkpoint)

    # Write compact status row (best-effort — never crash the trial).
    _write_status_csv_row(
        status_csv_path,
        iteration_idx=iteration_idx,
        opp_strength=pid_result.opp_strength,
        opp_strength_ema=pid_result.opp_strength_ema,
        skill_level=skill_level_used,
        sf_nodes=sf_nodes_used,
        current_rand=current_rand,
        wdl_regret=wdl_regret_used,
        ingest_ms=sp.ingest_ms,
        train_ms=tr.train_ms,
        total_iter_ms=report_dict["total_iter_ms"],
        steps=tr.steps,
        replay_size=buf_size,
        positions_ingested=sp.replay_positions_ingested,
        stale_games=sp.distributed_stale_games,
        train_loss=float(tr.metrics.loss) if tr.metrics is not None else None,
        best_loss=best_loss,
        total_w=sp.total_w,
        total_d=sp.total_d,
        total_l=sp.total_l,
        opt_lr=float(trainer.opt.param_groups[0]["lr"]),
        startup_source=restore.startup_source,
    )

    # Best-effort: keep disk usage bounded.
    if (completed_iterations + 1) % 5 == 0:
        _prune_trial_checkpoints(
            trial_dir=trial_dir,
            keep_last=tc.tune_num_to_keep,
        )


def _build_report_dict(
    *,
    tc: TrialConfig,
    trainer,
    pr: PidResult,
    sp: SelfplayResult,
    tr: TrainingResult,
    drift: DriftMetrics,
    eval_dict: dict,
    puzzle_dict: dict,
    # Iteration context
    current_rand: float,
    wdl_regret_used: float,
    sf_nodes_used: int,
    skill_level_used: int,
    use_distributed_selfplay: bool,
    pause_metrics: dict,
    restore: RestoreResult,
    best_loss: float,
    iter_t0: float,
    iteration_idx: int,
    buf_size: int,
    holdout_buf_size: int,
    holdout_frozen: bool,
    holdout_generation: int,
) -> dict:
    """Assemble the per-iteration report dict for Ray Tune."""
    metrics = tr.metrics

    # --- Test / holdout metrics ---
    test_dict: dict = {
        "holdout_frozen": int(1 if holdout_frozen else 0),
        "holdout_generation": int(holdout_generation),
        "data_drift_input_l2": float(drift.drift_input_l2),
        "data_drift_wdl_js": float(drift.drift_wdl_js),
        "data_drift_policy_entropy_diff": float(drift.drift_policy_entropy_diff),
        "data_drift_policy_entropy_train": float(drift.drift_policy_entropy_train),
        "data_drift_policy_entropy_holdout": float(drift.drift_policy_entropy_holdout),
        "data_policy_entropy": float(drift.data_policy_entropy),
        "data_unique_positions": float(drift.data_unique_positions),
        "data_wdl_balance": float(drift.data_wdl_balance),
        "test_size": 0,
        "test_loss": float("nan"),
        "test_policy_loss": float("nan"),
        "test_soft_policy_loss": float("nan"),
        "test_future_policy_loss": float("nan"),
        "test_wdl_loss": float("nan"),
        "test_sf_move_loss": float("nan"),
        "test_sf_move_acc": float("nan"),
        "test_sf_eval_loss": float("nan"),
        "test_categorical_loss": float("nan"),
        "test_volatility_loss": float("nan"),
        "test_sf_volatility_loss": float("nan"),
        "test_moves_left_loss": float("nan"),
    }
    if tr.test_metrics is not None:
        tm = tr.test_metrics
        test_dict.update({
            "test_size": int(holdout_buf_size),
            "test_loss": tm.loss,
            "test_policy_loss": tm.policy_loss,
            "test_soft_policy_loss": tm.soft_policy_loss,
            "test_future_policy_loss": tm.future_policy_loss,
            "test_wdl_loss": tm.wdl_loss,
            "test_sf_move_loss": tm.sf_move_loss,
            "test_sf_move_acc": tm.sf_move_acc,
            "test_sf_eval_loss": tm.sf_eval_loss,
            "test_categorical_loss": tm.categorical_loss,
            "test_volatility_loss": tm.volatility_loss,
            "test_sf_volatility_loss": tm.sf_volatility_loss,
            "test_moves_left_loss": tm.moves_left_loss,
        })

    return {
        "opponent_random_move_prob": float(current_rand),
        "opponent_random_move_prob_next": float(pr.random_move_prob_next),
        "opponent_sf_nodes": int(sf_nodes_used),
        "opponent_sf_nodes_next": int(pr.sf_nodes_next),
        "opponent_wdl_regret_limit": float(wdl_regret_used),
        "opponent_wdl_regret_limit_next": float(pr.wdl_regret_next),
        "opponent_topk": int(pr.curr_topk),
        "iter": int(iteration_idx),
        "global_iter": int(iteration_idx),
        "replay": int(buf_size),
        "test_replay": int(holdout_buf_size),
        "positions_added": sp.total_positions,
        "replay_positions_ingested": int(sp.replay_positions_ingested),
        "games_generated": int(sp.total_games_generated),
        "avg_game_plies": float(pr.avg_game_plies),
        "adjudication_rate": float(pr.adjudication_rate),
        "draw_rate": float(pr.draw_rate),
        "selfplay_adjudication_rate": float(pr.selfplay_adjudication_rate),
        "selfplay_draw_rate": float(pr.selfplay_draw_rate),
        "curriculum_adjudication_rate": float(pr.curriculum_adjudication_rate),
        "curriculum_draw_rate": float(pr.curriculum_draw_rate),
        "checkmate_rate": float(pr.checkmate_rate),
        "stalemate_rate": float(pr.stalemate_rate),
        "avg_plies_win": float(pr.avg_plies_win),
        "avg_plies_draw": float(pr.avg_plies_draw),
        "avg_plies_loss": float(pr.avg_plies_loss),
        "shared_samples_ingested": int(sp.imported_samples_this_iter),
        "shared_trials_selected": int(sp.shared_summary.get("source_trials_selected", 0)),
        "shared_trials_ingested": int(sp.shared_summary.get("source_trials_ingested", 0)),
        "shared_trials_skipped_repeat": int(sp.shared_summary.get("source_trials_skipped_repeat", 0)),
        "shared_shards_loaded": int(sp.shared_summary.get("source_shards_loaded", 0)),
        "distributed_selfplay": int(1 if use_distributed_selfplay else 0),
        "distributed_workers_per_trial": int(tc.distributed_workers_per_trial),
        "distributed_stale_games": int(sp.distributed_stale_games),
        "distributed_stale_positions": int(sp.distributed_stale_positions),
        "backpressure_paused_seconds": float(pause_metrics["paused_seconds"]),
        "backpressure_paused_fraction": float(pause_metrics["paused_fraction"]),
        "backpressure_paused_percent": float(pause_metrics["paused_percent"]),
        "startup_source": str(restore.startup_source),
        "salvage_warmstart_used": int(1 if restore.seed_warmstart_used else 0),
        "salvage_warmstart_slot": int(restore.seed_warmstart_slot),
        "salvage_warmstart_slots_total": int(restore.seed_warmstart_slots_total),
        "salvage_origin_used": int(1 if restore.salvage_origin_used else 0),
        "salvage_origin_slot": int(restore.salvage_origin_slot),
        "salvage_origin_slots_total": int(restore.salvage_origin_slots_total),
        "train_steps_used": int(tr.steps),
        "train_target_samples": int(tr.target_sample_budget),
        "train_window_target_samples": int(tr.window_target_samples),
        "win": sp.total_w,
        "draw": sp.total_d,
        "loss": sp.total_l,
        "sf_eval_delta6": float(sp.total_sf_d6 / max(1, sp.total_sf_d6_n)) if sp.total_sf_d6_n > 0 else 0.0,
        "sf_eval_delta6_n": sp.total_sf_d6_n,
        "sf_nodes": int(sf_nodes_used),
        "sf_nodes_next": int(pr.sf_nodes_next),
        "pid_ema_winrate": float(pr.pid_ema_wr),
        "pid_curriculum_w": int(sp.total_w),
        "pid_curriculum_d": int(sp.total_d),
        "pid_curriculum_l": int(sp.total_l),
        "selfplay_games": int(sp.total_selfplay_games),
        "selfplay_draw_games": int(sp.total_selfplay_draw_games),
        "random_move_prob": float(current_rand),
        "random_move_prob_next": float(pr.random_move_prob_next),
        "wdl_regret": float(wdl_regret_used),
        "wdl_regret_next": float(pr.wdl_regret_next),
        "skill_level": int(skill_level_used),
        "skill_level_next": int(pr.skill_level_next),
        "opponent_strength": float(pr.opp_strength),
        "opponent_strength_ema": float(pr.opp_strength_ema),
        "opt_lr": float(trainer.opt.param_groups[0]["lr"]),
        "peak_lr": float(getattr(trainer, "_peak_lr", 0.0)),
        "w_wdl": float(trainer.w_wdl),
        "w_soft": float(trainer.w_soft),
        "w_categorical": float(trainer.w_categorical),
        "w_sf_move": float(trainer.w_sf_move),
        "w_sf_wdl": float(trainer.w_sf_wdl),
        "diff_focus_q_weight": tc.diff_focus_q_weight,
        "feature_dropout_p": tc.feature_dropout_p,
        "fdp_king_safety": tc.fdp_king_safety,
        "fdp_pins": tc.fdp_pins,
        "fdp_pawns": tc.fdp_pawns,
        "fdp_mobility": tc.fdp_mobility,
        "fdp_outposts": tc.fdp_outposts,
        "selfplay_fraction": tc.selfplay_fraction,
        "optimizer_name": tc.optimizer,
        "sf_wdl_conf_power": float(trainer.sf_wdl_conf_power),
        "sf_wdl_draw_scale": float(trainer.sf_wdl_draw_scale),
        "train_loss": float(metrics.loss) if metrics is not None else 999.0,
        "train_time_s": float(metrics.train_time_s) if metrics is not None else 0.0,
        "optimizer_step_time_s": float(metrics.opt_step_time_s) if metrics is not None else 0.0,
        "trainer_steps_done": int(metrics.train_steps_done) if metrics is not None else 0,
        "train_samples_seen": int(metrics.train_samples_seen) if metrics is not None else 0,
        "trainer_steps_per_s": float(
            metrics.train_steps_done / max(metrics.train_time_s, 1e-9)
        ) if metrics is not None and metrics.train_time_s > 0.0 else 0.0,
        "trainer_samples_per_s": float(
            metrics.train_samples_seen / max(metrics.train_time_s, 1e-9)
        ) if metrics is not None and metrics.train_time_s > 0.0 else 0.0,
        "optimizer_steps_per_s": float(
            metrics.train_steps_done / max(metrics.opt_step_time_s, 1e-9)
        ) if metrics is not None and metrics.opt_step_time_s > 0.0 else 0.0,
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
        "gate_passed": int(1 if tr.gate_passed else 0),
        "ingest_ms": float(sp.ingest_ms),
        "train_ms": float(tr.train_ms),
        "total_iter_ms": float((time.monotonic() - iter_t0) * 1000.0),
        **eval_dict,
        **test_dict,
        **puzzle_dict,
        "blended_winrate_raw": float(pr.blended_winrate_raw) if pr.blended_winrate_raw is not None else 0.0,
    }


def _restore_checkpoint_or_salvage(
    *,
    config: dict,
    trainer,
    device: str,
    trial_id: str,
    trial_dir: Path,
    base_seed: int,
    active_seed: int,
    rng,
    ckpt,
    Checkpoint,
) -> tuple[RestoreResult, object]:
    """Restore from Ray checkpoint, salvage seed pool, or start fresh.

    Mutates *trainer* (model/optimizer load), *config* (donor config overlay),
    and may reseed *rng* on exploit clone.  Returns ``(restore_result, rng)``.
    """
    salvage_restore_donor_config = bool(config.get("salvage_restore_donor_config", False))
    salvage_restore_pid_state = bool(config.get("salvage_restore_pid_state", False))
    salvage_restore_full_trainer_state = bool(config.get("salvage_restore_full_trainer_state", False))

    rr = RestoreResult(active_seed=active_seed)

    restored_rng_state = None
    restored_trial_meta = None
    restored_owner_optimizer = ""

    if ckpt is not None:
        ckpt_dir = Path(ckpt.to_directory())
        maybe = ckpt_dir / "trainer.pt"
        pid_path = ckpt_dir / "pid_state.json"
        if pid_path.exists():
            try:
                rr.restored_pid_state = json.loads(pid_path.read_text(encoding="utf-8"))
            except Exception:
                rr.restored_pid_state = None
        rng_path = ckpt_dir / "rng_state.json"
        if rng_path.exists():
            try:
                restored_rng_state = json.loads(rng_path.read_text(encoding="utf-8"))
            except Exception:
                restored_rng_state = None
        trial_meta_path = ckpt_dir / "trial_meta.json"
        if trial_meta_path.exists():
            try:
                restored_trial_meta = json.loads(trial_meta_path.read_text(encoding="utf-8"))
            except Exception:
                restored_trial_meta = None
        if isinstance(restored_trial_meta, dict):
            restored_owner_optimizer = str(restored_trial_meta.get("optimizer", "") or "")
        current_optimizer = str(config.get("optimizer", "nadamw")).lower()
        if maybe.exists():
            model_only_restore = False
            if isinstance(restored_trial_meta, dict):
                owner_trial_id = str(restored_trial_meta.get("owner_trial_id", ""))
                if owner_trial_id and owner_trial_id != trial_id:
                    if restored_owner_optimizer:
                        model_only_restore = restored_owner_optimizer.lower() != current_optimizer
                    elif bool(config.get("search_optimizer", False)):
                        model_only_restore = True
            if model_only_restore:
                ckpt_data = torch.load(str(maybe), map_location=device)
                load_state_dict_tolerant(
                    trainer.model, ckpt_data["model"],
                    label="checkpoint_model_only",
                )
                del ckpt_data
                trainer._init_swa()
                rr.startup_source = "checkpoint_model_only"
            else:
                trainer.load(maybe)
                rr.startup_source = "checkpoint"
            if "lr" in config:
                trainer.set_peak_lr(float(config["lr"]), rescale_current=False)
    elif isinstance(config.get("salvage_seed_pool_dir"), str) and str(config.get("salvage_seed_pool_dir", "")).strip():
        seed_pool_dir = Path(str(config.get("salvage_seed_pool_dir"))).expanduser()
        if not seed_pool_dir.is_dir():
            raise RuntimeError(f"salvage_seed_pool_dir not found: {seed_pool_dir}")

        picked_dir, picked_slot, num_slots = _select_salvage_seed_slot(
            seed_pool_dir=seed_pool_dir,
            trial_dir=trial_dir,
            trial_id=trial_id,
        )
        if picked_dir is None:
            raise RuntimeError(
                f"salvage requested but no seed slot could be selected for "
                f"trial_id={trial_id} from {seed_pool_dir}"
            )

        maybe = Path(picked_dir) / "trainer.pt"
        if not maybe.exists():
            raise RuntimeError(f"salvage seed missing trainer.pt: {maybe}")

        rr.startup_source = "salvage"
        rr.seed_warmstart_used = True
        rr.seed_warmstart_slot = int(picked_slot)
        rr.seed_warmstart_slots_total = int(num_slots)
        rr.seed_warmstart_dir = Path(picked_dir)
        rr.seed_warmstart_replay_dir = rr.seed_warmstart_dir / "replay_shards"
        rr.salvage_origin_used = True
        rr.salvage_origin_slot = int(rr.seed_warmstart_slot)
        rr.salvage_origin_slots_total = int(rr.seed_warmstart_slots_total)
        rr.salvage_origin_dir = str(rr.seed_warmstart_dir.resolve())
        seed_entry = _load_salvage_manifest_entry(
            seed_pool_dir=seed_pool_dir,
            slot=rr.seed_warmstart_slot,
        )
        seed_warmstart_manifest_row: dict | None = None
        if isinstance(seed_entry, dict):
            result_row = seed_entry.get("result_row")
            if isinstance(result_row, dict):
                seed_warmstart_manifest_row = result_row
                if salvage_restore_donor_config:
                    donor_cfg = result_row.get("config")
                    if isinstance(donor_cfg, dict):
                        for k in (
                            "lr",
                            "cosmos_gamma",
                            "w_soft",
                            "w_future",
                            "w_wdl",
                            "w_sf_move",
                            "w_sf_eval",
                            "w_categorical",
                            "w_volatility",
                            "w_sf_wdl",
                            "sf_wdl_conf_power",
                            "sf_wdl_draw_scale",
                        ):
                            if k in donor_cfg:
                                config[k] = donor_cfg[k]
                        if "lr" in config:
                            trainer.set_peak_lr(float(config["lr"]), rescale_current=False)
                        if "cosmos_gamma" in config and hasattr(trainer.opt, "gamma"):
                            trainer.opt.gamma = float(config["cosmos_gamma"])
                        for wk in (
                            "w_soft",
                            "w_future",
                            "w_wdl",
                            "w_sf_move",
                            "w_sf_eval",
                            "w_categorical",
                            "w_volatility",
                            "w_sf_wdl",
                            "sf_wdl_conf_power",
                            "sf_wdl_draw_scale",
                        ):
                            if wk in config:
                                setattr(trainer, wk, float(config[wk]))

        if salvage_restore_full_trainer_state:
            trainer.load(maybe)
        else:
            salvage_ckpt = torch.load(str(maybe), map_location=device)
            load_state_dict_tolerant(
                trainer.model, salvage_ckpt["model"],
                label="salvage",
            )
            del salvage_ckpt
        if bool(config.get("salvage_reinit_volatility_heads", False)):
            reinit = reinit_volatility_head_parameters_(trainer.model)
            if reinit:
                print(f"[trial] Reinitialized salvage volatility heads: {', '.join(reinit)}")
        print(
            f"[trial] salvage warmstart loaded slot={rr.seed_warmstart_slot} "
            f"of {rr.seed_warmstart_slots_total} from {rr.seed_warmstart_dir}"
        )
        if salvage_restore_pid_state:
            pid_path = rr.seed_warmstart_dir / "pid_state.json"
            if pid_path.exists():
                try:
                    rr.restored_pid_state = json.loads(pid_path.read_text(encoding="utf-8"))
                except Exception:
                    rr.restored_pid_state = None
            rr.restored_pid_state, pid_manifest_overrides = _merge_pid_state_from_result_row(
                pid_state=rr.restored_pid_state,
                result_row=seed_warmstart_manifest_row,
            )
            if pid_manifest_overrides:
                print(
                    "[trial] salvage PID overrides from manifest row: "
                    + ", ".join(pid_manifest_overrides)
                )

    # Post-restore metadata fixups.
    restored_owner_trial_id = ""
    if isinstance(restored_trial_meta, dict):
        restored_owner_trial_id = str(restored_trial_meta.get("owner_trial_id", ""))
        rr.restored_owner_trial_dir = str(restored_trial_meta.get("owner_trial_dir", ""))
        restored_owner_optimizer = str(restored_trial_meta.get("optimizer", restored_owner_optimizer))
        rr.salvage_origin_used = bool(restored_trial_meta.get("salvage_origin_used", rr.salvage_origin_used))
        rr.salvage_origin_slot = int(restored_trial_meta.get("salvage_origin_slot", rr.salvage_origin_slot))
        rr.salvage_origin_slots_total = int(
            restored_trial_meta.get("salvage_origin_slots_total", rr.salvage_origin_slots_total)
        )
        rr.salvage_origin_dir = str(restored_trial_meta.get("salvage_origin_dir", rr.salvage_origin_dir or ""))
        rr.global_iter = int(restored_trial_meta.get("global_iter", 0))
    rr.cross_trial_restore = bool(
        ckpt is not None and restored_owner_trial_id and restored_owner_trial_id != trial_id
    )
    if rr.cross_trial_restore and rr.startup_source == "checkpoint":
        rr.startup_source = "exploit_restore"
    elif rr.cross_trial_restore and rr.startup_source == "checkpoint_model_only":
        rr.startup_source = "exploit_restore_model_only"

    if restored_rng_state is not None and not rr.cross_trial_restore:
        try:
            rng.bit_generator.state = restored_rng_state
        except Exception:
            pass

    if ckpt is not None and rr.cross_trial_restore:
        # PB2 exploit clone: fork RNG stream so recipients do not replay donor opening sequences.
        restore_rows = _count_jsonl_rows(trial_dir / "result.json")
        fork_seed = _stable_seed_u32(
            base_seed,
            trial_id,
            "exploit",
            restore_rows,
            int(getattr(trainer, "step", 0)),
        )
        rr.active_seed = int(fork_seed)
        rng = np.random.default_rng(rr.active_seed)
        torch.manual_seed(rr.active_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rr.active_seed)
        print(
            f"[trial] PB2 exploit restore detected: owner={restored_owner_trial_id} "
            f"recipient={trial_id} fork_seed={rr.active_seed} "
            f"owner_optimizer={restored_owner_optimizer or 'unknown'} "
            f"recipient_optimizer={str(config.get('optimizer', 'nadamw')).lower()} "
            f"restore_mode={rr.startup_source}"
        )
        # Inherit donor's EMA so the recipient isn't ranked on stale
        # pre-exploit history.
        if rr.restored_owner_trial_dir:
            _donor_row = _latest_trial_result_row(Path(rr.restored_owner_trial_dir))
            if _donor_row is not None:
                _donor_ema = _donor_row.get(
                    "opponent_strength_ema",
                    _donor_row.get("opponent_strength", 0.0),
                )
                rr.opp_strength_ema = float(_donor_ema)
                print(
                    f"[trial] exploit: inherited donor opp_strength_ema={rr.opp_strength_ema:.1f}"
                )

    return rr, rng


def train_trial(config: dict):
    """Ray Tune trainable.

    Reports metrics per outer-loop iteration. Supports checkpoint restore.
    """

    from ray.tune import report as _tune_report, get_checkpoint as _tune_get_checkpoint
    from ray.tune import Checkpoint, get_context as _tune_get_context

    # Re-read YAML and overlay all keys EXCEPT those PB2 is actively searching.
    # This lets --resume pick up config changes without clobbering tuned hyperparams.
    _reload_yaml_into_config(config, config.get("_yaml_config_path"))
    if "device" not in config:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    tc = TrialConfig.from_dict(config)
    _yaml_path = tc._yaml_config_path

    _ctx = _tune_get_context()
    base_seed = tc.seed
    trial_id = str(_ctx.get_trial_id() or "trial")
    trial_seed = _stable_seed_u32(base_seed, trial_id)
    active_seed = int(trial_seed)
    rng = np.random.default_rng(active_seed)
    torch.manual_seed(active_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(active_seed)

    device = tc.device

    model_cfg = ModelConfig(
        kind=tc.model,
        embed_dim=tc.embed_dim,
        num_layers=tc.num_layers,
        num_heads=tc.num_heads,
        ffn_mult=tc.ffn_mult,
        use_smolgen=tc.use_smolgen,
        use_nla=tc.use_nla,
        use_gradient_checkpointing=tc.gradient_checkpointing,
    )
    model = build_model(model_cfg)

    # Use Ray-provided trial directory for ALL per-trial state (checkpoints,
    # replay shards, gate state, best model, TensorBoard logs).
    # IMPORTANT: Do NOT use config["work_dir"] here — it points to the shared
    # runs/pbt2_small/ directory. Using it caused all 10 trials to write
    # checkpoints to the same directory, making PB2 unable to clone checkpoints
    # ("no checkpoint for trial X. Skip exploit.").
    trial_dir = Path(_ctx.get_trial_dir())
    work_dir = trial_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # Compact status CSV — reset on each process start so checkpoint-restore rows don't accumulate.
    _STATUS_CSV_PATH = trial_dir / "status.csv"
    _STATUS_COLS = [
        "iter", "global_iter", "opp", "opp_ema", "skill", "sf_nodes", "rmp", "regret",
        "ingest_s", "train_s", "iter_s", "steps", "replay", "pos_added",
        "stale", "train_loss", "best_loss", "win", "draw", "loss", "lr", "startup",
    ]
    with _STATUS_CSV_PATH.open("w", newline="") as _f:
        csv.writer(_f).writerow(_STATUS_COLS)

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
    opp_strength_ema = 0.0
    _OPP_EMA_ALPHA = 0.3  # smoothing factor — higher = more responsive
    if best_state_path.exists():
        try:
            d = json.loads(best_state_path.read_text(encoding="utf-8"))
            best_loss = float(d.get("best_loss", d.get("loss", best_loss)))
            opp_strength_ema = float(d.get("opp_strength_ema", 0.0))
        except Exception:
            pass

    trainer_ctor = trainer_kwargs_from_config(
        config | {"device": device}, log_dir=work_dir / "tb",
    )
    trainer = Trainer(model, **trainer_ctor)

    ckpt = _tune_get_checkpoint()
    restore, rng = _restore_checkpoint_or_salvage(
        config=config, trainer=trainer, device=device,
        trial_id=trial_id, trial_dir=trial_dir,
        base_seed=base_seed, active_seed=active_seed,
        rng=rng, ckpt=ckpt, Checkpoint=Checkpoint,
    )
    restored_pid_state = restore.restored_pid_state
    global_iter = restore.global_iter
    opp_strength_ema = restore.opp_strength_ema

    # Rebuild tc — _restore_checkpoint_or_salvage may overlay donor config.
    tc = TrialConfig.from_dict(config)

    # Growing sliding window: start small, grow as the net matures.
    current_window = tc.replay_window_start
    replay_shard_dir = _trial_replay_shard_dir(config=config, trial_dir=trial_dir)
    selfplay_shards_dir = work_dir / "selfplay_shards"
    selfplay_shards_dir.mkdir(parents=True, exist_ok=True)

    # Optional warmstart replay from salvage seed slot (fresh trials only).
    if (
        restore.seed_warmstart_used
        and (not restore.cross_trial_restore)
        and restore.seed_warmstart_replay_dir is not None
        and restore.seed_warmstart_replay_dir.is_dir()
        and (not iter_shard_paths(replay_shard_dir))
    ):
        replay_shard_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for sp in iter_shard_paths(restore.seed_warmstart_replay_dir):
            copy_or_link_shard(sp, replay_shard_dir / sp.name)
            copied += 1
        if copied:
            print(
                f"[trial] Seeded {copied} replay shards from salvage slot "
                f"{restore.seed_warmstart_slot} ({restore.seed_warmstart_replay_dir})"
            )

    shared_summary = {
        "source_trials_selected": 0,
        "source_trials_ingested": 0,
        "source_trials_skipped_repeat": 0,
        "source_shards_loaded": 0,
        "source_samples_ingested": 0,
    }

    # Seed replay buffer with shared iter-0 data (played once from bootstrap net).
    # Only copy if this is a fresh trial (no existing shards in replay_shard_dir).
    if tc.shared_shards_dir and not iter_shard_paths(replay_shard_dir) and (not restore.cross_trial_restore):
        src = Path(tc.shared_shards_dir)
        if src.is_dir():
            replay_shard_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for sp in iter_shard_paths(src):
                copy_or_link_shard(sp, replay_shard_dir / sp.name)
                copied += 1
            if copied:
                shared_summary["source_shards_loaded"] = int(copied)
                print(f"[trial] Seeded {copied} shared iter-0 shards from {src}")

    if restore.cross_trial_restore and tc.exploit_replay_refresh_enabled:
        donor_trial_dir = Path(restore.restored_owner_trial_dir).expanduser() if restore.restored_owner_trial_dir else None
        refresh_summary = _refresh_replay_shards_on_exploit(
            config=config,
            replay_shard_dir=replay_shard_dir,
            recipient_trial_dir=trial_dir,
            donor_trial_dir=donor_trial_dir,
            keep_recent_fraction=tc.exploit_replay_local_keep_recent_fraction,
            keep_older_fraction=tc.exploit_replay_local_keep_older_fraction,
            donor_shards=tc.exploit_replay_donor_shards,
            donor_skip_newest=tc.exploit_replay_skip_newest,
            shard_size=tc.shard_size,
            holdout_fraction=tc.holdout_fraction,
        )
        print(
            "[trial] replay refresh after exploit: "
            f"local_before={refresh_summary['local_before']} "
            f"deleted={refresh_summary['local_deleted']} "
            f"local_recent_deleted={refresh_summary['local_recent_deleted']} "
            f"local_older_deleted={refresh_summary['local_older_deleted']} "
            f"after_keep={refresh_summary['local_after_keep']} "
            f"donor_available={refresh_summary['donor_available']} "
            f"donor_selected={refresh_summary['donor_selected']} "
            f"donor_copied={refresh_summary['donor_copied']} "
            f"final={refresh_summary['local_final']}"
        )
        # Set current_window based on shards actually kept on disk after refresh,
        # so the DiskReplayBuffer capacity is correct from construction and
        # _enforce_window doesn't aggressively evict on first add_many.
        kept_after_refresh = iter_shard_paths(replay_shard_dir)
        if kept_after_refresh:
            current_window = max(int(current_window), len(kept_after_refresh) * tc.shard_size)
            print(
                f"[trial] exploit restore: pre-set window={current_window} "
                f"for {len(kept_after_refresh)} kept shards"
            )

    buf = DiskReplayBuffer(
        current_window,
        shard_dir=replay_shard_dir,
        rng=rng,
        shuffle_cap=tc.shuffle_buffer_size,
        shard_size=tc.shard_size,
        refresh_interval=tc.shuffle_refresh_interval,
        refresh_shards=tc.shuffle_refresh_shards,
        draw_cap_frac=tc.shuffle_draw_cap_frac,
        wl_max_ratio=tc.shuffle_wl_max_ratio,
    )

    # Preserve intentionally seeded replay (resume, salvage warmstart, shared-shard
    # bootstrap), but keep plain fresh starts at replay_window_start so easy early
    # games evict promptly instead of inheriting stale local shards.
    seeded_replay_start = bool(ckpt is not None or restore.seed_warmstart_used or shared_summary["source_shards_loaded"] > 0)
    if seeded_replay_start:
        current_window = max(int(current_window), int(len(buf)))
    buf.capacity = int(current_window)
    print(
        f"[trial] buffer init: startup_source={restore.startup_source} "
        f"seeded={seeded_replay_start} cross_trial={restore.cross_trial_restore} "
        f"len(buf)={len(buf)} capacity={buf.capacity} "
        f"tracked_shards={len(buf._shard_paths)} total_pos={buf._total_positions}"
    )
    holdout_buf = ArrayReplayBuffer(tc.holdout_capacity, rng=rng)

    # Load pre-trained bootstrap checkpoint (trained offline via scripts/train_bootstrap.py).
    # This gives the value head a working signal so first MCTS searches are better than random.
    # IMPORTANT: Only load MODEL WEIGHTS — do NOT restore optimizer/scheduler/step.
    # The bootstrap was trained for ~13k steps with its own LR schedule; carrying that
    # state into the trainable causes: (1) step=13323 skips warmup entirely,
    # (2) scheduler resumes mid-cosine-cycle with near-zero LR then spikes on restart,
    # (3) optimizer momentum buffers from bootstrap's data distribution cause wrong
    # gradient directions on selfplay data, (4) PB2's lr perturbation has no effect
    # because scheduler's base_lr is locked to bootstrap's lr (0.0003).
    if tc.bootstrap_checkpoint and ckpt is None and (not restore.seed_warmstart_used):
        # Only load bootstrap if Ray didn't restore a trial checkpoint (i.e. fresh start).
        bp = Path(tc.bootstrap_checkpoint)
        if bp.exists():
            print(f"[trial] Loading pre-trained bootstrap model weights: {bp}")
            ckpt_data = torch.load(str(bp), map_location=device)
            model_sd = ckpt_data.get("model") or ckpt_data.get("model_state_dict") or ckpt_data
            load_state_dict_tolerant(trainer.model, model_sd, label="bootstrap")
            if tc.bootstrap_zero_policy_heads:
                zeroed = zero_policy_head_parameters_(trainer.model)
                if zeroed:
                    print(f"[trial] Zeroed bootstrap policy heads: {', '.join(zeroed)}")
            if tc.bootstrap_reinit_volatility_heads:
                reinit = reinit_volatility_head_parameters_(trainer.model)
                if reinit:
                    print(f"[trial] Reinitialized bootstrap volatility heads: {', '.join(reinit)}")
            # Re-sync SWA with the newly loaded weights (AveragedModel deep-copies at init).
            trainer._init_swa()
            # Deliberately skip: optimizer, scheduler, step — start fresh.
            del ckpt_data
        else:
            print(f"[trial] WARNING: bootstrap checkpoint not found: {bp}")

    holdout_frozen = False
    holdout_generation = 0

    use_distributed_selfplay = (
        tc.distributed_workers_per_trial > 0
        and bool((tc.distributed_server_root or "").strip())
        and bool((tc.distributed_server_url or "").strip())
    )

    need_local_sf = (not use_distributed_selfplay) or (tc.gate_games > 0)
    sf = None
    if need_local_sf:
        if tc.sf_workers > 1:
            sf = StockfishPool(
                path=tc.stockfish_path,
                nodes=tc.sf_nodes,
                num_workers=tc.sf_workers,
                multipv=tc.sf_multipv,
                hash_mb=tc.sf_hash_mb,
            )
        else:
            sf = StockfishUCI(
                tc.stockfish_path,
                nodes=tc.sf_nodes,
                multipv=tc.sf_multipv,
                hash_mb=tc.sf_hash_mb,
            )

    distributed_server_root = (
        _resolve_local_override_root(
            raw_root=str(tc.distributed_server_root),
            tune_work_dir=tc.work_dir or str(work_dir),
            suffix="server",
        )
        if use_distributed_selfplay
        else None
    )
    if distributed_server_root is not None:
        _set_active_run_prefix(server_root=distributed_server_root, trial_id=trial_id)
    distributed_dirs = (
        _trial_server_dirs(server_root=distributed_server_root, trial_id=trial_id)
        if distributed_server_root is not None
        else None
    )
    distributed_worker_procs: list[subprocess.Popen[bytes]] = []
    distributed_inference_broker_proc: subprocess.Popen[bytes] | None = None

    if use_distributed_selfplay and ckpt is not None and distributed_dirs is not None:
        quarantined = _quarantine_inbox_shards(
            inbox_dir=distributed_dirs["inbox_dir"],
            processed_dir=distributed_dirs["processed_dir"],
            reason=f"{restore.startup_source}_resume",
        )
        if int(quarantined["moved_shards"]) > 0:
            print(
                "[trial] quarantined preexisting inbox shards on resume: "
                f"trial={trial_id} moved={quarantined['moved_shards']} "
                f"dst={quarantined['quarantine_root']}"
            )

    eval_sf = None
    if tc.eval_games > 0:
        # For fixed-strength evaluation, use a dedicated engine instance with its own node limit.
        eval_sf = StockfishUCI(
            tc.stockfish_path,
            nodes=tc.eval_sf_nodes,
            multipv=1,
            hash_mb=tc.sf_hash_mb,
        )

    pid = None
    if tc.sf_pid_enabled:
        pid = pid_from_config(config)
        if restored_pid_state is not None:
            try:
                pid.load_state_dict(restored_pid_state)
            except Exception:
                pass

    # Optional puzzle evaluation suite.
    puzzle_suite = None
    if tc.puzzle_epd and tc.puzzle_interval > 0:
        from chess_anti_engine.eval import load_epd
        try:
            puzzle_suite = load_epd(tc.puzzle_epd)
        except FileNotFoundError:
            puzzle_suite = None

    pause_marker_path = _resolve_pause_marker_path(tc=tc, trial_dir=trial_dir)

    if use_distributed_selfplay:
        current_rand_init = (
            float(pid.random_move_prob)
            if pid is not None
            else tc.sf_pid_random_move_prob_start
        )
        current_nodes_init = int(pid.nodes) if pid is not None else tc.sf_nodes
        current_skill_init = int(pid.skill_level) if pid is not None else 0
        sims_init = tc.mcts_simulations
        if tc.progressive_mcts:
            sims_init = progressive_mcts_simulations(
                int(getattr(trainer, "step", 0)),
                start=tc.mcts_start_simulations,
                max_sims=tc.mcts_simulations,
                ramp_steps=tc.mcts_ramp_steps,
                exponent=tc.mcts_ramp_exponent,
            )
        prev_published_model_sha = _publish_distributed_trial_state(
            trainer=trainer,
            config=config,
            model_cfg=model_cfg,
            server_root=distributed_server_root,
            trial_id=trial_id,
            training_iteration=0,
            trainer_step=int(getattr(trainer, "step", 0)),
            sf_nodes=current_nodes_init,
            random_move_prob=current_rand_init,
            skill_level=current_skill_init,
            mcts_simulations=int(sims_init),
            wdl_regret=float(pid.wdl_regret) if pid is not None else -1.0,
            pause_selfplay=False,
            pause_reason="",
        )
        distributed_inference_broker_proc = _ensure_inference_broker(
            config=config,
            trial_id=trial_id,
            trial_dir=trial_dir,
            publish_dir=distributed_dirs["publish_dir"],
            proc=distributed_inference_broker_proc,
        )
        distributed_worker_procs = _ensure_distributed_workers(
            config=config,
            trial_dir=trial_dir,
            trial_id=trial_id,
            procs=distributed_worker_procs,
        )
    distributed_pause_active = False
    distributed_pause_started_at: float | None = None
    prev_published_model_sha: str = ""

    ckpt_dir = work_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        iterations = tc.iterations
        completed_iterations = 0
        while completed_iterations < iterations:
            iteration_zero_based = int(global_iter)
            iteration_idx = iteration_zero_based + 1
            iter_t0 = time.monotonic()

            # Live-reload YAML config each iteration so changes apply without restart.
            _reload_yaml_into_config(config, _yaml_path)
            tc = TrialConfig.from_dict(config)

            in_salvage_startup_grace = (
                restore.startup_source == "salvage"
                and bool(restore.salvage_origin_used)
                and int(iteration_zero_based) < tc.salvage_startup_no_share_iters
            )
            _wait_if_paused(
                pause_marker_path=pause_marker_path,
                poll_seconds=tc.pause_poll_seconds,
                trial_id=trial_id,
                iteration=iteration_idx,
            )

            # Difficulty knobs used for this iteration's selfplay (kept fixed across
            # selfplay chunks). PID is updated once per iteration AFTER training so
            # changes align to net updates rather than chunk noise.
            current_rand = float(pid.random_move_prob) if pid is not None else tc.sf_pid_random_move_prob_start
            wdl_regret_used = float(pid.wdl_regret) if pid is not None else -1.0
            sf_nodes_used = (
                int(getattr(sf, "nodes", 0) or 0)
                if sf is not None
                else (int(pid.nodes) if pid is not None else tc.sf_nodes)
            )
            skill_level_used = int(getattr(pid, "skill_level", 0) or 0) if pid is not None else 0

            base_sims = tc.mcts_simulations
            sims = base_sims
            if tc.progressive_mcts:
                sims = progressive_mcts_simulations(
                    int(getattr(trainer, "step", 0)),
                    start=tc.mcts_start_simulations,
                    max_sims=base_sims,
                    ramp_steps=tc.mcts_ramp_steps,
                    exponent=tc.mcts_ramp_exponent,
                )

            sp, prev_published_model_sha, current_window, distributed_inference_broker_proc = _run_selfplay_phase(
                tc=tc, config=config, trainer=trainer, model_cfg=model_cfg,
                buf=buf, holdout_buf=holdout_buf,
                holdout_frozen=holdout_frozen,
                device=device, rng=rng, sf=sf, pid=pid,
                use_distributed_selfplay=use_distributed_selfplay,
                distributed_dirs=distributed_dirs,
                distributed_server_root=distributed_server_root,
                distributed_worker_procs=distributed_worker_procs,
                distributed_inference_broker_proc=distributed_inference_broker_proc,
                prev_published_model_sha=prev_published_model_sha,
                current_rand=current_rand,
                skill_level_used=skill_level_used,
                sims=sims,
                iteration_idx=iteration_idx,
                iteration_zero_based=iteration_zero_based,
                trial_id=trial_id,
                trial_dir=trial_dir,
                selfplay_shards_dir=selfplay_shards_dir,
                replay_shard_dir=replay_shard_dir,
                current_window=current_window,
                in_salvage_startup_grace=in_salvage_startup_grace,
            )
            distributed_pause_active = False
            distributed_pause_started_at = None
            if sp.should_retry:
                continue

            if (not holdout_frozen) and tc.freeze_holdout_at > 0 and len(holdout_buf) >= tc.freeze_holdout_at:
                holdout_frozen = True

            drift = _compute_drift_metrics(
                buf=buf, holdout_buf=holdout_buf,
                drift_sample_size=tc.drift_sample_size,
            )

            # Optional holdout reset based on input drift threshold.
            if (
                tc.reset_holdout_on_drift
                and (tc.drift_threshold > 0.0)
                and (drift.drift_input_l2 > tc.drift_threshold)
            ):
                holdout_buf.clear()
                holdout_frozen = False
                holdout_generation += 1

            _sync_trainer_weights(trainer, config, tc, wdl_regret_used, current_rand)

            tr = _run_training_and_gating(
                tc=tc, trainer=trainer, buf=buf, holdout_buf=holdout_buf,
                config=config, model_cfg=model_cfg,
                device=device, rng=rng, pid=pid, sf=sf,
                current_rand=current_rand,
                skill_level_used=skill_level_used,
                sims=sims,
                wdl_regret_used=wdl_regret_used,
                total_positions=sp.total_positions,
                imported_samples_this_iter=sp.imported_samples_this_iter,
                gate_match_idx=gate_match_idx,
                gate_state_path=gate_state_path,
                use_distributed_selfplay=use_distributed_selfplay,
                distributed_server_root=distributed_server_root,
                distributed_dirs=distributed_dirs,
                iteration_idx=iteration_idx,
                iteration_zero_based=iteration_zero_based,
                trial_id=trial_id,
                startup_source=restore.startup_source,
                salvage_origin_used=restore.salvage_origin_used,
            )
            gate_match_idx = tr.gate_match_idx

            eval_dict = _run_eval_games(
                tc=tc, trainer=trainer, device=device, rng=rng,
                eval_sf=eval_sf,
            )

            # Flush replay + save checkpoint (model+optimizer+step + PID state).
            checkpoint = _save_trial_checkpoint(
                trainer=trainer,
                buf=buf,
                ckpt_dir=ckpt_dir,
                rng=rng,
                trial_id=trial_id,
                trial_dir=trial_dir,
                config=config,
                base_seed=base_seed,
                restore=restore,
                iteration_idx=iteration_idx,
                Checkpoint=Checkpoint,
            )

            # Best-model tracking: prefer holdout loss when available, skip if no training yet.
            best_loss = _update_best_model(
                trainer=trainer, test_metrics=tr.test_metrics, train_metrics=tr.metrics,
                best_loss=best_loss, best_dir=best_dir, best_state_path=best_state_path,
                iteration_idx=iteration_idx, opp_strength_ema=opp_strength_ema,
            )

            pid_result = _run_pid_and_eval(
                tc=tc, pid=pid, sf=sf,
                sp_result=sp,
                iteration_zero_based=iteration_zero_based,
                opp_strength_ema=opp_strength_ema,
                opp_ema_alpha=_OPP_EMA_ALPHA,
                current_rand=current_rand,
                sf_nodes_used=sf_nodes_used,
                skill_level_used=skill_level_used,
                wdl_regret_used=wdl_regret_used,
            )
            opp_strength_ema = pid_result.opp_strength_ema

            _finalize_iteration(
                tc=tc, trainer=trainer, pid=pid,
                sp=sp, tr=tr, drift=drift,
                pid_result=pid_result,
                eval_dict=eval_dict,
                checkpoint=checkpoint,
                best_loss=best_loss,
                ckpt_dir=ckpt_dir,
                work_dir=work_dir,
                trial_dir=trial_dir,
                status_csv_path=_STATUS_CSV_PATH,
                tune_report_fn=_tune_report,
                puzzle_suite=puzzle_suite,
                current_rand=current_rand,
                wdl_regret_used=wdl_regret_used,
                sf_nodes_used=sf_nodes_used,
                skill_level_used=skill_level_used,
                use_distributed_selfplay=use_distributed_selfplay,
                distributed_pause_started_at=distributed_pause_started_at,
                distributed_pause_active=distributed_pause_active,
                restore=restore,
                holdout_frozen=holdout_frozen,
                holdout_generation=holdout_generation,
                buf_size=len(buf),
                holdout_buf_size=len(holdout_buf),
                iter_t0=iter_t0,
                iteration_idx=iteration_idx,
                iteration_zero_based=iteration_zero_based,
                completed_iterations=completed_iterations,
                device=device,
                rng=rng,
            )

            global_iter = int(iteration_idx)
            completed_iterations += 1
    finally:
        _stop_worker_processes(distributed_worker_procs)
        _stop_process(distributed_inference_broker_proc)
        if sf is not None:
            sf.close()
        if eval_sf is not None:
            eval_sf.close()
