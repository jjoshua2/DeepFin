"""Config / lifecycle helpers for the Ray Tune trainable.

Converts TrialConfig → play_batch kwargs, overlays YAML onto the config
dict, syncs runtime-mutable loss weights onto the trainer, and provides
the pause-marker primitives used by the outer loop.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path

from chess_anti_engine.config_keys import TRAINER_WEIGHT_KEYS
from chess_anti_engine.model import ModelConfig
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


def _pause_ack_name(trial_id: str) -> str:
    """Filename of the per-trial pause acknowledgement marker."""
    return f".paused_{trial_id}.ack"


def _write_pause_acks(markers: list[Path], *, trial_id: str, iteration: int) -> list[Path]:
    """Drop a deterministic per-trial ack next to each present pause marker.

    graceful_restart.py otherwise infers "paused" from progress.csv row growth,
    which structurally can't fire when a trial holds at the iteration boundary
    *before* appending a post-pause row (it just sits idle until timeout). The
    ack is an explicit "I am now holding" signal the harness can poll directly.
    Best-effort: an ack write must never block or break the pause itself.
    """
    name = _pause_ack_name(trial_id)
    written: list[Path] = []
    seen: set[Path] = set()
    for m in markers:
        ack = m.parent / name
        if ack in seen:
            continue
        seen.add(ack)
        try:
            ack.write_text(f"trial={trial_id} next_iter={iteration}\n")
            written.append(ack)
        except OSError:
            pass  # ack is advisory; never let it interfere with pausing
    return written


def _clear_pause_acks(ack_paths: list[Path]) -> None:
    """Remove the acks written by _write_pause_acks (called on resume) so a
    later graceful restart doesn't see a stale 'paused' signal."""
    for ack in ack_paths:
        try:
            ack.unlink(missing_ok=True)
        except OSError:
            pass


def _wait_if_paused(
    *,
    pause_marker_paths: list[Path],
    poll_seconds: int,
    trial_id: str,
    iteration: int,
) -> None:
    poll_s = max(1, int(poll_seconds))
    announced = False
    ack_paths: list[Path] = []

    def _existing_markers() -> list[Path]:
        return [p for p in pause_marker_paths if p.exists()]

    try:
        while True:
            present = _existing_markers()
            if not present:
                break
            if not announced:
                # Write the ack BEFORE announcing so the marker is on disk the
                # instant the log line appears. flush=True so the message reaches
                # the log even though the subsequent sleep would otherwise hold
                # it in stdio buffers.
                ack_paths = _write_pause_acks(present, trial_id=trial_id, iteration=iteration)
                print(
                    f"[trial] pause marker(s) detected: {[str(p) for p in present]} "
                    f"(trial={trial_id}, next_iter={iteration}, "
                    f"ack={[str(p) for p in ack_paths]})",
                    flush=True,
                )
                announced = True
            time.sleep(float(poll_s))
    finally:
        _clear_pause_acks(ack_paths)
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
            selfplay_temperature=tc.selfplay_temperature,
            selfplay_decay_start_move=tc.selfplay_temperature_decay_start_move,
            selfplay_decay_moves=tc.selfplay_temperature_decay_moves,
            selfplay_endgame=tc.selfplay_temperature_endgame,
        ),
        search=SearchConfig(
            mcts_type=tc.mcts,
            playout_cap_fraction=tc.playout_cap_fraction,
            full_ply_pair_fraction=tc.full_ply_pair_fraction,
            fast_simulations=tc.fast_simulations,
            fpu_reduction=tc.fpu_reduction,
            fpu_at_root=tc.fpu_at_root,
            gumbel_topk=tc.gumbel_topk,
            gumbel_c_scale=tc.gumbel_c_scale,
            gumbel_scale=tc.gumbel_scale,
            gumbel_scale_after=tc.gumbel_scale_after,
            gumbel_scale_decay_start_move=tc.gumbel_scale_decay_start_move,
            gumbel_scale_decay_moves=tc.gumbel_scale_decay_moves,
            curriculum_gumbel_scale=tc.curriculum_gumbel_scale,
            curriculum_gumbel_scale_after=tc.curriculum_gumbel_scale_after,
            curriculum_gumbel_scale_decay_start_move=tc.curriculum_gumbel_scale_decay_start_move,
            curriculum_gumbel_scale_decay_moves=tc.curriculum_gumbel_scale_decay_moves,
            volatility_q_scale=tc.volatility_q_scale,
            volatility_fpu=tc.volatility_fpu,
            volatility_anchor=tc.volatility_anchor,
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
            sf_move_nodes=tc.sf_move_nodes,
            sf_fast_ply_node_scale=tc.sf_fast_ply_node_scale,
            sf_label_nodes_cap=tc.sf_label_nodes_cap,
            sf_policy_temp=tc.sf_policy_temp,
            sf_policy_label_smooth=tc.sf_policy_label_smooth,
            sf_wdl_use_cp_logistic=tc.sf_wdl_use_cp_logistic,
            sf_wdl_cp_slope=tc.sf_wdl_cp_slope,
            sf_wdl_cp_draw_width=tc.sf_wdl_cp_draw_width,
            soft_policy_temp=tc.soft_policy_temp,
            timeout_adjudication_threshold=tc.timeout_adjudication_threshold,
            volatility_source=tc.volatility_source,
            syzygy_path=tc.syzygy_path,
            stockfish_syzygy_path=tc.stockfish_syzygy_path,
            syzygy_rescore_policy=tc.syzygy_rescore_policy,
            syzygy_adjudicate=tc.syzygy_adjudicate,
            syzygy_adjudicate_fraction=tc.syzygy_adjudicate_fraction,
            syzygy_in_search=tc.syzygy_in_search,
            categorical_bins=tc.categorical_bins,
            hlgauss_sigma=tc.hlgauss_sigma,
            categorical_blend_frac=tc.categorical_blend_frac,
            categorical_search_blend_frac=tc.categorical_search_blend_frac,
            policy_encoding=tc.policy_encoding,
            input_history_encoding=tc.input_history_encoding,
            input_extra_features=tc.input_extra_features,
            record_lc0_root_input=tc.record_lc0_root_input,
            history_rep_fix=tc.history_rep_fix,
            record_dense_sf_policy=tc.record_dense_sf_policy,
            record_sf_p0_policy=tc.record_sf_p0_policy,
            record_sf_p0_regret=tc.record_sf_p0_regret,
            record_fast_ply_value=tc.record_fast_ply_value,
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
    "distributed_inference_compile_mode",
    "distributed_inference_broker_enabled",
    "distributed_inference_shared_broker",
    "distributed_inference_slots_per_worker",
    "num_samples",
    "max_concurrent_trials",
    "gpus_per_trial",
    "optimizer",
    "matrix_optimizer_scope",
    "weight_decay_mode",
    "soda_scope",
    "soda_start_step",
    "input_pos_encoding",
    "qkv_projection",
    "use_deepnorm",
    "policy_encoding",
    "input_history_encoding",
    "history_rep_fix",
    "input_extra_features",
    "use_dynamic_relations",
    "dynamic_relation_count",
    "policy_dynamic_relations",
    "input_global_embedding",
    "input_global_embedding_channels",
    "input_square_embedding",
    "smolgen_mode",
    "smolgen_pooling",
    "smolgen_hidden_channels",
    "smolgen_hidden_sz",
    "smolgen_gen_sz",
    "smolgen_bias_scale",
    "smolgen_bias_norm",
    "arc_attention_bias",
    "smolgen_relation_basis",
    "smolgen_relation_norm",
    "smolgen_relation_coeff_norm",
    "smolgen_relation_scale",
    "phase_output_adapter",
    "phase_output_adapter_dim",
    "phase_smolgen",
    "phase_piece_thresholds",
    "num_layers",
    "embed_dim_by_layer",
    "ffn_mult_by_layer",
})

# Every key that feeds the model build = ModelConfig's own fields. On resume
# these come from the checkpoint arch (resume_model_config_from_arch) or, on a
# no-arch / salvage warm-start, directly from config (trainable.py builds the
# model from config_model_cfg when there is no arch). NOTE these are a distinct
# set from _TOPOLOGY_KEYS: core shape fields (embed_dim, num_heads, ffn_mult,
# use_smolgen, ...) are ModelConfig fields but are NOT in _TOPOLOGY_KEYS, so the
# gate below keys off _MODEL_BUILD_KEYS too — otherwise they would fall through
# to the unconditional overlay.
_MODEL_BUILD_KEYS = frozenset(f.name for f in dataclasses.fields(ModelConfig))
# Model keys that must NOT be startup-auto-filled from YAML: injecting one that
# is absent from an older restored config would rebuild the model at a layout
# the checkpoint tensors don't match — shape schedule (num_layers, embed_dim,
# ffn_mult, smolgen sizes, qkv_projection) or encoding identity
# (policy/input/history) — and the tolerant loader then zero-inits the mismatch,
# crashing the optimizer at step() (the variable-width-FFN resume corruption).
# An encoding/shape migration must be deliberate (donor/salvage config).
# history_rep_fix is the sole exception: a ModelConfig flag that changes NO
# tensor shape (only the selfplay rep-plane encoding) and is exactly the
# worker/selfplay flag this fill exists to propagate. Derived from ModelConfig,
# so every current AND future model field is covered automatically.

# Optimizer-construction keys: the Trainer (and its optimizer) is built from
# config BEFORE the checkpoint is restored (trainable.py: Trainer(...) precedes
# _restore_checkpoint_or_salvage). These are the ones that change the optimizer
# STRUCTURE — which optimizer, how params are grouped, or moment-tensor shapes —
# so force-applying one on resume builds an optimizer the saved state can't load
# into; Trainer.load() then rejects/reinitializes the moments + scheduler. They
# must come from the checkpoint's own config, not a resume-time YAML overlay.
#
# NOT included (deliberately): the per-group VALUE knobs read alongside these in
# trainer_kwargs_from_config — matrix_lr_multiplier, matrix_weight_decay,
# aux_weight_decay, global_board_*_lr_multiplier/weight_decay. Those only set the
# lr/weight-decay of an already-fixed param group (the grouping is decided by
# matrix_optimizer_scope), so changing one on resume is as safe as changing `lr`
# — no state-shape mismatch. test_optimizer_construction_keys_cover_structural
# locks this split so a newly-added STRUCTURE knob can't silently slip back in.
_OPTIMIZER_CONSTRUCTION_KEYS = frozenset({
    "optimizer", "matrix_optimizer_scope", "weight_decay_mode",
    "soda_scope", "soda_start_step", "cosmos_rank",
})

# Construction-bound keys: changing one on resume rebuilds the model OR the
# optimizer away from the checkpoint (state then reinits/crashes). Neither the
# startup fill below NOR the experiment-state resume overlay
# (_patch_experiment_state_for_resume) may propagate these from YAML — only a
# deliberate migration (donor/salvage config) changes them. history_rep_fix is
# excluded (shape-neutral selfplay flag, safe + intended to propagate).
_RESUME_CONSTRUCTION_BOUND_KEYS = frozenset(
    k for k in (_MODEL_BUILD_KEYS | _OPTIMIZER_CONSTRUCTION_KEYS) if k != "history_rep_fix"
)


def _reload_yaml_into_config(config: dict, yaml_path: str | None, *, live_reload: bool = False) -> None:
    """Overlay YAML values into *config*, preserving PB2-searched keys.

    Topology keys that require a broker/worker restart are detected and
    logged as warnings instead of being silently applied.
    ``lr_schedule`` is restart-only for a live trainer, but may be updated
    during trial startup/resume before the trainer is constructed.

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
        missing = object()
        for k, v in fresh.items():
            if k in searched or k.startswith("pb2_bounds_"):
                continue
            if live_reload and k == "lr_schedule" and k in config and config[k] != v:
                log.warning(
                    "YAML reload: %s changed (%s -> %s) but requires restart — skipping",
                    k, config[k], v,
                )
                continue
            if k in _TOPOLOGY_KEYS or k in _RESUME_CONSTRUCTION_BOUND_KEYS:
                current = config.get(k, missing)
                if current is missing:
                    if not live_reload and k not in _RESUME_CONSTRUCTION_BOUND_KEYS:
                        # Startup/resume: the broker + workers are (re)built
                        # from this config *after* the overlay, so a worker/
                        # infra/selfplay topology key absent from an older
                        # restored config (introduced after that checkpoint was
                        # saved) must be applied from yaml — otherwise it
                        # silently defaults off. Safe here because no component
                        # is running yet. Construction-bound keys are EXCLUDED
                        # (_RESUME_CONSTRUCTION_BOUND_KEYS): auto-filling a model
                        # shape/encoding key OR an optimizer-construction key
                        # would rebuild the model/optimizer away from the
                        # checkpoint it must match (see that set's note). Log so
                        # a silent startup change leaves a trail.
                        log.info(
                            "YAML startup reload: applying %s=%r (absent from restored config)",
                            k, v,
                        )
                        config[k] = v
                        continue
                    log.warning(
                        "YAML reload: %s is absent from restored config but requires restart — skipping",
                        k,
                    )
                    continue
                if current != v:
                    log.warning(
                        "YAML reload: %s changed (%s -> %s) but requires restart — skipping",
                        k, current, v,
                    )
                    continue
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
    release_keys = {
        "lr_release_cycle_steps",
        "lr_release_start_frac",
        "lr_release_min_scale",
        "lr_release_shape",
    }
    if release_keys.intersection(config) and hasattr(trainer, "set_lr_release_config"):
        trainer.set_lr_release_config(
            cycle_steps=config.get("lr_release_cycle_steps"),
            release_start_frac=config.get("lr_release_start_frac"),
            min_scale=config.get("lr_release_min_scale"),
            release_shape=config.get("lr_release_shape"),
        )
    if "cosmos_gamma" in config and hasattr(trainer.opt, "gamma"):
        trainer.opt.gamma = float(config["cosmos_gamma"])
    aurora_group_updates: dict[str, object] = {}
    if "aurora_pp_iterations" in config:
        aurora_group_updates["aurora_pp_iterations"] = int(config["aurora_pp_iterations"])
    if "aurora_pp_beta" in config:
        aurora_group_updates["aurora_pp_beta"] = float(config["aurora_pp_beta"])
    if "aurora_polar_steps" in config:
        aurora_group_updates["aurora_polar_steps"] = int(config["aurora_polar_steps"])
    if "aurora_polar_method" in config:
        aurora_group_updates["aurora_polar_method"] = str(config["aurora_polar_method"])
    if "aurora_polar_dtype" in config:
        aurora_group_updates["aurora_polar_dtype"] = str(config["aurora_polar_dtype"])
    if "aurora_polar_safety" in config:
        aurora_group_updates["aurora_polar_safety"] = float(config["aurora_polar_safety"])
    if aurora_group_updates:
        for group in trainer.opt.param_groups:
            if bool(group.get("use_aurora", False)):
                group.update(aurora_group_updates)
    for wk in _TRAINER_WEIGHT_KEYS:
        if wk in config:
            setattr(trainer, wk, float(config[wk]))
    if "w_sf_volatility" not in config and "w_volatility" in config:
        trainer.w_sf_volatility = float(config["w_volatility"])
    if "use_adjusted_wdl_target" in config:
        trainer.use_adjusted_wdl_target = bool(config["use_adjusted_wdl_target"])
    if "adjusted_wdl_regret_source" in config:
        trainer.adjusted_wdl_regret_source = str(config["adjusted_wdl_regret_source"])
    if "adjusted_wdl_regret_scale" in config:
        trainer.adjusted_wdl_regret_scale = float(config["adjusted_wdl_regret_scale"])
    if "adjusted_wdl_regret_cap" in config:
        trainer.adjusted_wdl_regret_cap = float(config["adjusted_wdl_regret_cap"])
    if "resid_channel_dropout" in config:
        drop = max(0.0, min(0.95, float(config["resid_channel_dropout"])))
        trainer.resid_channel_dropout = drop
        if hasattr(trainer.model, "resid_channel_dropout"):
            setattr(trainer.model, "resid_channel_dropout", drop)
    if "resid_channel_balance_weight" in config:
        weight = max(0.0, float(config["resid_channel_balance_weight"]))
        trainer.resid_channel_balance_weight = weight
        if hasattr(trainer.model, "resid_channel_balance_weight"):
            setattr(trainer.model, "resid_channel_balance_weight", weight)


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
