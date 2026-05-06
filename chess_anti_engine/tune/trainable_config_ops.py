"""Config / lifecycle helpers for the Ray Tune trainable.

Converts TrialConfig → play_batch kwargs, overlays YAML onto the config
dict, syncs runtime-mutable loss weights onto the trainer, and provides
the pause-marker primitives used by the outer loop.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from chess_anti_engine.config_keys import TRAINER_WEIGHT_KEYS
from chess_anti_engine.selfplay.budget import progressive_mcts_simulations
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.train import Trainer
from chess_anti_engine.tune.trainable_metrics import _dynamic_sf_wdl_weight
from chess_anti_engine.tune.trial_config import DifficultyState, TrialConfig

log = logging.getLogger(__name__)


_TRAINER_WEIGHT_KEYS = TRAINER_WEIGHT_KEYS


def _resolve_sims(tc: TrialConfig, trainer, *, max_sims: int) -> int:
    """Resolve MCTS simulation count, honouring progressive ramp if enabled."""
    if not tc.progressive_mcts:
        return int(max_sims)
    return progressive_mcts_simulations(
        int(getattr(trainer, "step", 0)),
        start=tc.mcts_start_simulations,
        max_sims=int(max_sims),
        ramp_steps=tc.mcts_ramp_steps,
        exponent=tc.mcts_ramp_exponent,
    )


def _resolve_pause_marker_paths(*, tc: TrialConfig, trial_dir: Path) -> list[Path]:
    """All paths the trial considers a "pause" marker; ANY existing one pauses.

    The trial sees TWO different "tune dirs": the ephemeral Ray session
    artifacts dir under ``/tmp/ray/session_*/artifacts/.../tune/driver_artifacts/``
    (what ``_ctx.get_trial_dir().parent`` returns), and the persistent
    ``<work_dir>/tune/`` (where graceful_restart writes). Earlier "fixes"
    only checked the ephemeral path, so pause.txt at the persistent path was
    silently invisible. Now we check both, plus a custom ``tc.pause_file``.
    """
    tune_root_ephemeral = trial_dir.parent
    candidates: list[Path] = []
    raw = tc.pause_file
    if raw and raw.strip():
        p = Path(raw.strip())
        if not p.is_absolute():
            p = tune_root_ephemeral / p
        candidates.append(p)
    candidates.append(trial_dir / "pause.txt")
    candidates.append(tune_root_ephemeral / "pause.txt")
    if tc.work_dir:
        # work_dir is relative in YAML (e.g. "runs/pbt2_small"); inside the Ray
        # actor cwd is /tmp/ray/.../working_dirs/..., NOT the repo root, so a
        # bare Path(work_dir) resolves to nowhere. Anchor against the yaml
        # config's grandparent dir (yaml lives in <repo>/configs/X.yaml).
        wd = Path(tc.work_dir)
        if not wd.is_absolute() and tc._yaml_config_path:
            wd = Path(tc._yaml_config_path).resolve().parent.parent / wd
        candidates.append(wd / "tune" / "pause.txt")
    # Dedupe while preserving order (a custom pause_file may equal one of the defaults).
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _wait_if_paused(
    *,
    pause_marker_paths: list[Path],
    poll_seconds: int,
    trial_id: str,
    iteration: int,
) -> None:
    poll_s = max(1, int(poll_seconds))
    announced = False

    def _existing_markers() -> list[Path]:
        return [p for p in pause_marker_paths if p.exists()]

    while True:
        present = _existing_markers()
        if not present:
            break
        if not announced:
            # flush=True so the message reaches the log even though the
            # subsequent sleep would otherwise hold it in stdio buffers.
            print(
                f"[trial] pause marker(s) detected: {[str(p) for p in present]} "
                f"(trial={trial_id}, next_iter={iteration})",
                flush=True,
            )
            announced = True
        time.sleep(float(poll_s))
    if announced:
        print(
            f"[trial] pause marker cleared (trial={trial_id}, resuming_iter={iteration})",
            flush=True,
        )


def _play_batch_kwargs(tc: TrialConfig, ds: DifficultyState | None = None) -> dict:
    """Extract all config-driven play_batch kwargs as dataclass instances.

    Callers (selfplay, gate, eval) use dataclasses.replace() for per-site overrides.
    This is the single source of truth for config → play_batch mapping.

    When ``ds`` is provided and carries a non-sentinel regret value (>=0), the
    PID-controlled ``wdl_regret_limit`` is threaded into ``OpponentConfig`` so
    that gate matches and local selfplay train against the same opponent the
    distributed workers see (which read the same value off the server manifest).
    ``ds=None`` is for fixed-strength eval, where PID regret deliberately
    does not apply.
    """
    wdl_regret_limit: float | None = None
    if ds is not None and float(ds.wdl_regret) >= 0.0:
        wdl_regret_limit = float(ds.wdl_regret)
    return dict(
        opponent=OpponentConfig(wdl_regret_limit=wdl_regret_limit),
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
            sf_wdl_use_cp_logistic=tc.sf_wdl_use_cp_logistic,
            sf_wdl_cp_slope=tc.sf_wdl_cp_slope,
            sf_wdl_cp_draw_width=tc.sf_wdl_cp_draw_width,
            soft_policy_temp=tc.soft_policy_temp,
            timeout_adjudication_threshold=tc.timeout_adjudication_threshold,
            volatility_source=tc.volatility_source,
            syzygy_path=tc.syzygy_path,
            syzygy_rescore_policy=tc.syzygy_rescore_policy,
            syzygy_adjudicate=tc.syzygy_adjudicate,
            syzygy_adjudicate_fraction=tc.syzygy_adjudicate_fraction,
            syzygy_in_search=tc.syzygy_in_search,
            categorical_bins=tc.categorical_bins,
            hlgauss_sigma=tc.hlgauss_sigma,
        ),
    )


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
        from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
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


def _apply_lr_gamma_weights(trainer: Trainer, config: dict, *, rescale_current_lr: bool) -> None:
    """Push lr / cosmos_gamma / loss-weight keys from config into trainer.

    ``rescale_current_lr=True`` is the iter-loop call (PB2 perturbations
    take effect immediately). ``False`` is the one-shot init call from
    salvage donor-config overlay (we don't want to scale a freshly-built
    schedule).
    """
    if "lr" in config:
        trainer.set_peak_lr(float(config["lr"]), rescale_current=rescale_current_lr)
    if "cosmos_gamma" in config and hasattr(trainer.opt, "gamma"):
        trainer.opt.gamma = float(config["cosmos_gamma"])
    for wk in _TRAINER_WEIGHT_KEYS:
        if wk in config:
            setattr(trainer, wk, float(config[wk]))
    if "w_sf_volatility" not in config and "w_volatility" in config:
        trainer.w_sf_volatility = float(config["w_volatility"])


def _sync_trainer_weights(
    trainer: Trainer,
    config: dict,
    tc: TrialConfig,
    ds: DifficultyState,
) -> None:
    """Re-read loss weights and LR from config into trainer.

    Called each iteration so PB2 perturbations and live YAML changes
    take effect immediately.
    """
    _apply_lr_gamma_weights(trainer, config, rescale_current_lr=True)

    cur_sf_frac = _dynamic_sf_wdl_weight(
        sf_wdl_start=tc.sf_wdl_frac,
        sf_wdl_floor=tc.sf_wdl_frac_floor,
        sf_wdl_floor_at_regret=tc.sf_wdl_floor_at_regret,
        regret_max=tc.sf_pid_wdl_regret_max,
        wdl_regret_used=ds.wdl_regret,
    )
    if cur_sf_frac is not None:
        trainer.sf_wdl_frac = cur_sf_frac
