"""Config / lifecycle helpers for the Ray Tune trainable.

Converts TrialConfig → play_batch kwargs, overlays YAML onto the config
dict, syncs runtime-mutable loss weights onto the trainer, and provides
the pause-marker primitives used by the outer loop.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

from chess_anti_engine.config_keys import is_inert_dead_config_value, TRAINER_WEIGHT_KEYS
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
from chess_anti_engine.train.trainer import resolve_sf_target_params
from chess_anti_engine.tune.trainable_metrics import _dynamic_sf_wdl_weight
from chess_anti_engine.tune.trial_config import DifficultyState, TrialConfig
import contextlib

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
        with contextlib.suppress(OSError):
            ack.unlink(missing_ok=True)


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
    return {
        "opponent": OpponentConfig(wdl_regret_limit=wdl_regret_limit),
        "temp": TemperatureConfig(
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
        "search": SearchConfig(
            mcts_type=tc.mcts,
            playout_cap_fraction=tc.playout_cap_fraction,
            full_ply_pair_fraction=tc.full_ply_pair_fraction,
            fast_simulations=tc.fast_simulations,
            fpu_reduction=tc.fpu_reduction,
            fpu_at_root=tc.fpu_at_root,
            gumbel_topk=tc.gumbel_topk,
            gumbel_policy_temp=tc.gumbel_policy_temp,
            gumbel_target_batch=tc.gumbel_target_batch,
            gumbel_vloss_weight=tc.gumbel_vloss_weight,
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
        "opening": OpeningConfig(
            opening_book_path=tc.opening_book_path,
            opening_book_max_plies=tc.opening_book_max_plies,
            opening_book_max_games=tc.opening_book_max_games,
            opening_book_prob=tc.opening_book_prob,
            opening_book_path_2=tc.opening_book_path_2,
            opening_book_max_plies_2=tc.opening_book_max_plies_2,
            opening_book_max_games_2=tc.opening_book_max_games_2,
            opening_book_mix_prob_2=tc.opening_book_mix_prob_2,
            random_start_plies=tc.random_start_plies,
            opening_fen_list_path=tc.opening_fen_list_path,
            opening_fen_prob=tc.opening_fen_prob,
            opening_fen_net_side_to_move=tc.opening_fen_net_side_to_move,
            opening_fen_selfplay_only=tc.opening_fen_selfplay_only,
            opening_fen_dole_per_iter=tc.opening_fen_dole_per_iter,
        ),
        "diff_focus": DiffFocusConfig(
            enabled=tc.diff_focus_enabled,
            q_weight=tc.diff_focus_q_weight,
            pol_scale=tc.diff_focus_pol_scale,
            slope=tc.diff_focus_slope,
            min_keep=tc.diff_focus_min,
        ),
        "game": GameConfig(
            max_plies=tc.max_plies,
            selfplay_fraction=tc.selfplay_fraction,
            sf_move_nodes=tc.sf_move_nodes,
            sf_fast_ply_node_scale=tc.sf_fast_ply_node_scale,
            sf_label_nodes_cap=tc.sf_label_nodes_cap,
            sf_label_nodes_floor=tc.sf_label_nodes_floor,
            sf_label_escalate_q_gap=tc.sf_label_escalate_q_gap,
            sf_label_escalate_nodes=tc.sf_label_escalate_nodes,
            sf_label_escalate_max_per_game=tc.sf_label_escalate_max_per_game,
            sf_policy_temp=tc.sf_policy_temp,
            sf_policy_score_mode=tc.sf_policy_score_mode,
            sf_policy_cp_temp=tc.sf_policy_cp_temp,
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
            blindspot_harvest_out_path=tc.blindspot_harvest_out_path,
        ),
    }


# Keys that affect broker/worker topology — changing these mid-run requires
# a restart because the broker's shared-memory layout and worker processes
# are configured at launch time.
_TOPOLOGY_KEYS = frozenset({
  # Worker-level keys (workers_per_trial, use_compile, sf_workers, threaded,
  # selfplay_threads) removed — _ensure_distributed_workers spawns new workers
  # with updated config each iteration.
    "distributed_inference_max_batch_per_slot",
    "distributed_inference_batch_wait_ms",
    "distributed_inference_adaptive_idle_ms",
    "distributed_inference_use_compile",
    "distributed_inference_compile_mode",
    "distributed_inference_aot_dir",
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
    "soda_scope", "soda_start_step",
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

# Opening-BOOK file paths served by the distributed server, which captures
# them at process launch (create_app / --opening-book-path*). The manifest
# publisher re-reads config each iteration, so a live change to either of
# these would advertise a new SHA while the server still serves the
# launch-time bytes. Frozen against live reload (restart to change); see the
# reload guard. opening_fen_list_path is NOT in this set — the manifest
# publisher copies it into publish_dir under a fixed name each iteration
# (chess_anti_engine/tune/distributed_runtime.py) and the server serves that
# copy, so a yaml path change there takes effect on the next manifest publish
# with no restart needed.
_LAUNCH_FIXED_ASSET_PATH_KEYS = frozenset({
    "opening_book_path", "opening_book_path_2",
})

# Keys whose ONLY consumer is a constructor argument in
# ``trainable_init._init_replay_buffers``, which runs once per trial process.
# A live yaml edit to one of these lands in ``config`` — and therefore in the
# result-row ``config`` block and in every tool that reads it — while the
# object that was going to use it was built at startup and never re-reads it.
# That is strictly worse than being ignored: the running trial reports the new
# value back as though it had taken, so `audit_realized_config.py` printed
# "the running value is correct" about a `shuffle_buffer_size` the
# `DiskReplayBuffer` was not running (found 2026-08-01, this file's own class
# of defect). Listing them turns that silent overlay into a WARNING plus a
# PENDING-RESTART finding, exactly as for `gate_games` below.
#
# Membership requires the key to have NO other consumer — a key read again on
# the per-iteration path takes effect live and must NOT be frozen here. The
# four sampling knobs passed to the same constructor
# (`replay_sf_gap_priority_weight`, `replay_fast_low_surprise_priority`,
# `diff_focus_pol_scale`, `diff_focus_q_weight`) are re-pushed onto the buffer
# every iteration by ``trainable.py``, and `shard_size` is re-read per
# iteration by the replay-share ingest, so all five stay live-reloadable.
# ``tests/test_construction_only_config_keys.py`` re-derives that split from
# the source and fails on any newly-added constructor argument that joins this
# class without being classified.
_CONSTRUCTION_ONLY_REPLAY_KEYS = frozenset({
    # DiskReplayBuffer(...) — trainable_init.py:625
    "shuffle_buffer_size",        # -> shuffle_cap, disk_buffer.py:249
    "shuffle_refresh_interval",   # -> refresh_interval, disk_buffer.py:251
    "shuffle_refresh_shards",     # -> refresh_shards, disk_buffer.py:252
    # -> shard_recency_exponent. Same class as the two above and for a stronger
    # reason than "the constructor is the only reader": it is the SHAPE of the
    # shuffle-refresh shard draw, so a mid-run edit would split one replay
    # window across two sampling regimes -- the same argument that froze
    # shuffle_draw_cap_frac / shuffle_wl_max_ratio below.
    "replay_shard_recency_exponent",
    # ⚑ THESE TWO WERE A REAL CHOICE, not a classification. Unlike the three
    # above (consumed inside the buffer's own construction), draw_cap_frac and
    # wl_max_ratio are plain public attributes read PER BATCH DRAW
    # (disk_buffer.py:1456, :1469) — structurally identical to
    # diff_focus_pol_scale, which trainable.py:774 pushes live every iteration.
    # Two lines in that push loop would make them live-editable.
    # DECIDED: freeze them. Both change the SAMPLING DISTRIBUTION the window is
    # drawn with, so a mid-window edit splits one replay window across two
    # sampling regimes and silently confounds whatever readout spans it — the
    # experiment protocol's one-data-affecting-change-per-window rule cannot be
    # honoured for a knob that can move underneath a running window. The
    # diff_focus_* knobs are pushed live because they were already being tuned
    # against a same-day readout; these have no such caller. Revisit only with a
    # ledger entry that names the readout wanting them live.
    "shuffle_draw_cap_frac",      # -> draw_cap_frac, disk_buffer.py:344
    "shuffle_wl_max_ratio",       # -> wl_max_ratio, disk_buffer.py:345
    "replay_upgrade_v1_planes",   # -> upgrade_v1_planes, disk_buffer.py:235
    # The initial sliding-window size, read once into a local at
    # trainable_init.py:605; every later window move goes through
    # replay_window_max / replay_window_growth, which ARE live.
    "replay_window_start",
    # ArrayReplayBuffer(...) — trainable_init.py:675. `.capacity` is never
    # re-assigned from config afterwards.
    "holdout_capacity",
})

# The era-forgetting probes' SET IDENTITY. `_init_era_probes` loads both frozen
# npz files once per trial process (trainable_init.py) and the loop then scores
# the loaded arrays; nothing re-reads these three keys afterwards, so a live
# edit would land in `config`, echo back correct in the result row, and leave
# the probe scoring the launch-time set.
#
# The classification is stronger than "only a constructor reads it", and the
# reason is the standing rule that a ruler change must invalidate its records:
# `probe_era_policy_eregret` is only comparable across iterations because the
# rows behind it did not move. A mid-run repoint would silently splice two
# rulers into one column of progress.csv, whose header is fixed from row 1 and
# whose segments append across restarts — the reader has no way to see the
# seam. A restart is the honest way to change a ruler, and the loader prints
# the new set's digest when it happens.
#
# `era_probe_interval` and `era_probe_batch_size` are deliberately NOT here:
# both are read off the freshly reloaded TrialConfig on the iteration they act
# on (trainable_phases._run_era_probes_if_due), so a live edit DOES take effect
# and freezing them would break a working throttle. Neither changes what is
# measured, only how often and in what chunk size.
_CONSTRUCTION_ONLY_PROBE_KEYS = frozenset({
    "era_probe_path",
    "era_probe_inwindow_path",
    "era_probe_rows",
})

# The same class as the replay/probe sets above, found one file further out
# (audit T2): keys whose only consumer runs in ``train_trial`` BEFORE the
# iteration loop. The classification had been finished for the replay buffer
# and the era probes and never swept over the rest of the startup block, so
# ``scripts/audit_realized_config.py`` reached its live-reloadable branch for
# these — the branch that reports "the running value is correct".
#
# ``tests/test_startup_only_config_keys.py`` DERIVES this class from the source
# instead of restating it: reachability from the startup region of
# ``train_trial``, minus everything the ``while`` loop can reach. A key added to
# the schema tomorrow whose only reader is a startup helper fails that test
# until it is classified here.
#
#   iterations      -- the `while` bound is read into a local ONCE
#                      (trainable.py). Also, and separately worth knowing when
#                      an operator edits it: it is a PER-PROCESS budget, not a
#                      per-run one -- `completed_iterations` starts at 0 in
#                      every process.
#   puzzle_epd      -- `_load_puzzle_suite` loads the EPD file once. The
#                      `puzzle_interval` / `puzzle_simulations` throttles ARE
#                      live and deliberately stay live.
#   eval_sf_nodes   -- the eval engine's node limit, fixed when that engine is
#                      constructed (or not constructed at all).
#   sf_pid_enabled  -- `_init_pid` returns None when it is false at launch, and
#                      `pid.refresh_live_params(config)` cannot resurrect a
#                      controller that does not exist.
#   pause_file      -- `_resolve_pause_marker_paths` resolves the marker list
#                      once; the loop polls the resolved paths.
#
# NOT here, decided rather than overlooked:
#   eval_games      -- ⚑ IT WAS HERE, AND THAT WAS WRONG (review of PR F, B1).
#                      The first version of this set froze it on the reasoning
#                      that `_init_eval_stockfish` returns None for <= 0 at
#                      startup and `eval_sf` is never rebuilt, so a live edit
#                      "can only ever DISABLE". That is true of the ENABLE
#                      direction and false of the key: `_run_eval_games` also
#                      reads it as a MAGNITUDE (`games=tc.eval_games`,
#                      trainable_phases.py), off a `tc` the loop rebuilds every
#                      iteration, so for a trial launched with eval_games > 0
#                      an edit to the COUNT takes effect live -- measured on
#                      main. Freezing it removed a working knob, which is the
#                      same objection that (correctly) keeps `log_level` out of
#                      this set. What remains true and is NOT fixed here: with
#                      eval_games 0 at launch, turning eval on live is still a
#                      silent no-op, because nothing rebuilds `eval_sf`. That is
#                      a wiring gap in `_init_eval_stockfish`, not a reload
#                      class, and freezing the key papered over it while
#                      breaking the direction that worked.
#   log_level       -- startup-only for the TRIAL's own logger, but it is in
#                      `_WORKER_LAUNCH_CONFIG_KEYS`, so a live edit changes the
#                      worker launch signature and the next relaunch honours
#                      it. Freezing it would break a knob that works.
#   the sf_pid_* lever gains -- genuinely construction-only inside
#                      `pid_from_config` (only ~10 of the family are re-read by
#                      `refresh_live_params`), and the derivation test flags
#                      every one of them. Freezing ~30 knobs of the live
#                      difficulty controller is a behaviour change to a running
#                      experiment's control surface, not an observability fix,
#                      so it belongs to its own ledger entry.
#   tune_scheduler / pb2_perturbation_interval / tune_keep_last_experiments --
#                      DRIVER-side keys (harness.py); the trial actor never
#                      reads them. Calling them "restart required" here would
#                      also be false for the first two: on `--resume` they come
#                      back from `tuner.pkl`, so not even a restart applies them
#                      (audit T6). They need the driver-side fix, not this set.
_STARTUP_ONLY_TRIAL_KEYS = frozenset({
    "iterations",
    "puzzle_epd",
    "eval_sf_nodes",
    "sf_pid_enabled",
    "pause_file",
})

# `lr_schedule` is skipped by the live reload alone (the trainer's scheduler is
# already built), but IS applied during startup/resume before the trainer
# exists — so it is restart-required for a running trial like the rest.
#
# The gate_* keys are the same shape: `PromotionGate` is constructed ONCE from
# the launch config (trainable.py), so a live edit to any of them would sit in
# the config dict doing nothing until the next restart. `gate_games` is the
# dangerous one — it raises at startup, so an operator editing it live gets
# neither the old behaviour nor the error, just silence. Listing them here
# turns that silence into a WARNING on the iteration the edit lands, which is
# the whole difference between "restart required" and "quietly ignored".
# Window shape is deliberately construction-time as well: changing
# window_iters mid-window redefines what a half-filled window means.
#
# `gate_threshold` / `gate_interval` / `gate_mcts_sims` are NOT here any more
# (audit G3-11): they are DEAD, not construction-only, and the two classes are
# disjoint by design -- a restart applies a construction-only key and REFUSES a
# dead one, so listing a dead key here would tell an operator to restart into a
# hard error. `gate_games` stays: it is not in the dead set, because it raises
# with its own message from `gate_config_from_dict`.
_LIVE_RELOAD_SKIPPED_KEYS = frozenset({
    "lr_schedule",
    "gate_games",
    "gate_mode", "gate_window_iters", "gate_min_iters",
    "gate_min_games_per_side", "gate_demote_delta_elo", "gate_demote_step_elo",
    "gate_alpha", "gate_max_hold_iters",
}) | _CONSTRUCTION_ONLY_REPLAY_KEYS | _CONSTRUCTION_ONLY_PROBE_KEYS | _STARTUP_ONLY_TRIAL_KEYS


# Yaml keys the validator still ACCEPTS and `TrialConfig` still parses, but
# which no production code path consumes. This is not the construction-only
# class: those reach a constructor and are merely frozen after it. These reach
# NOTHING, so their value is read back correctly from the trial's own config
# while the object that would use it never sees the number at all.
#
# `replay_sf_gap_priority_signed` (backlog #106): allow-listed at
# `utils/config_yaml.py:277`, parsed at `trial_config.py:727`, supported by
# `DiskReplayBuffer.sf_gap_priority_signed` — and never passed. The production
# construction site (`trainable_init.py`, `DiskReplayBuffer(...)`) passes only
# `sf_gap_priority_weight`, and the per-iteration live push
# (`trainable.py`) pushes only the same four live knobs. Its real consumers are
# both OFFLINE — `scripts/retarget_retrain.py` and
# `scripts/holdout_policy_screen.py` — which is why the key is refused here
# rather than deleted from `TrialConfig`.
#
# Each key maps to the ONE value that is inert, i.e. the value the production
# object actually runs at. That value is tolerated: deleting a key from a live
# yaml is itself a reload risk (the validator is all-or-nothing), so an
# operator must be able to leave it in place. Any other value is refused
# loudly, on the precedent of `promotion_gate.gate_config_from_dict`'s
# `gate_games` check — silently accepting it is the failure this repo keeps
# paying for.
# `gate_threshold` / `gate_interval` / `gate_mcts_sims` (audit G3-11): the
# corpses of the removed 1-sim-vs-Stockfish gate. `grep` finds each of them in
# exactly three places -- the yaml allowlist, the live-reload skip set, and
# `TrialConfig`, which parses them and never reads them. `gate_config_from_dict`
# builds the anchored gate from the eight `gate_mode`-family keys and looks at
# none of these. `gate_games` is the only one that refuses, and it refuses
# because a non-zero value asks for BEHAVIOUR that was deleted; these three
# were left "tolerated at any value" on the grounds that they are inert
# scalars, which is true of the value and false of the operator's belief: a
# `gate_interval: 5` starts a run, reads back correctly from the trial's own
# config, and does nothing at all.
#
# Their inert values are the ones the production yaml ships (pbt2_small.yaml
# :1077-1079), so this refusal cannot fire on the live config as it stands.
_DEAD_CONFIG_KEY_INERT_VALUES: dict[str, object] = {
    "replay_sf_gap_priority_signed": False,
    "gate_threshold": 0.50,
    "gate_interval": 1,
    "gate_mcts_sims": 1,
}

# Why each dead key is dead, in the words the refusal needs. Keyed by the same
# names, and ``test_every_dead_config_key_explains_itself`` requires one per
# key so a key cannot be added to the set above without saying what happened
# to it.
_DEAD_CONFIG_KEY_REASONS: dict[str, str] = {
    "replay_sf_gap_priority_signed": (
        "the trial builds its DiskReplayBuffer without it (trainable_init.py) "
        "and never pushes it on live reload (trainable.py). Its only consumers "
        "are the offline scripts/retarget_retrain.py and "
        "scripts/holdout_policy_screen.py, and the gap-priority family was "
        "KILLED by experiment #104 (docs/experiment_ledger.md), so re-enabling "
        "it needs a ledger entry and a deliberate wiring change, not a yaml line"
    ),
    "gate_threshold": (
        "it was the win-rate line of the REMOVED 1-sim vs-Stockfish gate. "
        "Nothing reads it: the anchored gate demotes on gate_demote_delta_elo "
        "against the previous MODEL, never on a score against a PID-controlled "
        "opponent whose setpoint equalled this number"
    ),
    "gate_interval": (
        "it was how often the REMOVED 1-sim vs-Stockfish gate ran. The "
        "anchored gate judges EVERY iteration -- there is no interval to set, "
        "and TrialConfig.gate_interval is parsed and read by nothing"
    ),
    "gate_mcts_sims": (
        "it was the search width of the REMOVED 1-sim vs-Stockfish gate, and 1 "
        "simulation is raw policy argmax rather than the quantity production "
        "plays. The anchored gate plays no games at all: it splits the games "
        "the loop already played by publishing model"
    ),
}


def dead_config_key_inert_values() -> dict[str, object]:
    """The dead keys with the ONE value each that is inert, for reporting tools.

    ``dead_config_keys()`` answers "is this key dead"; this answers "is this
    VALUE the harmless one", which a provenance report needs in order to stay
    quiet on a yaml that merely leaves the key in place at its realized default.
    """
    return dict(_DEAD_CONFIG_KEY_INERT_VALUES)


def dead_config_key_violations(
    config: Mapping[str, object],
) -> list[tuple[str, object, object]]:
    """Dead keys in *config* set to something other than their inert value.

    ⚑ THE SINGLE TABLE, walked with the SINGLE PREDICATE
    (``config_keys.is_inert_dead_config_value``, which lives in the
    dependency-free key module so the reporting script can import it at module
    scope rather than inside its per-key loop). ``reject_dead_config_keys``
    raises at startup on exactly this list; ``scripts/audit_realized_config.py``
    reports on exactly this predicate. Two implementations that agree today
    drift the moment a dead key is added whose inert value is not falsey, and
    then the audit would call safe a value the trial refuses to start on — a
    guard disagreeing with the criterion it guards.

    Returns ``(key, value, inert)`` triples so a caller can name all three.
    """
    return [
        (key, config[key], inert)
        for key, inert in _DEAD_CONFIG_KEY_INERT_VALUES.items()
        if key in config and not is_inert_dead_config_value(config[key], inert)
    ]




def reject_dead_config_keys(config: Mapping[str, object]) -> None:
    """Raise on a yaml key that is parsed but reaches no production consumer.

    Called from the production replay-buffer construction path, so the refusal
    happens at STARTUP, before any training compute. The live-reload path
    cannot reach here — it warns and declines to apply instead (see
    ``_reload_yaml_into_config``), because crashing a running trial on a yaml
    edit would crash-loop under ``train.sh``'s auto-resume.
    """
    for key, value, inert in dead_config_key_violations(config):
        raise ValueError(
            f"{key}={value!r} is parsed but reaches no production code path: "
            f"{_DEAD_CONFIG_KEY_REASONS.get(key, 'nothing reads it')}. The "
            f"only value this run can realize is {inert!r}. Refusing rather "
            f"than accepting a knob that reads back correctly and does nothing."
        )


def dead_config_keys() -> frozenset[str]:
    """Yaml keys that reach NO production consumer at any value.

    A FOURTH reload class, distinct from all three in
    ``restart_required_config_keys()``. Those are declined live because
    applying them would be unsafe or could not act until a restart; a dead key
    is declined live and then REFUSED at the restart
    (``reject_dead_config_keys``). So "not overlaid" is true of both, and
    "a restart will apply it" is true only of the others.

    Exported so a provenance tool can say which one it is looking at. Without
    it, a set-but-declined dead key falls through every branch and is reported
    as a healthy live-reloadable key whose yaml and realized values agree —
    the exact verdict shape PR #303 removed for construction-only keys.
    """
    return frozenset(_DEAD_CONFIG_KEY_INERT_VALUES)


def construction_only_config_keys() -> frozenset[str]:
    """Yaml keys that reach ONLY a constructor, so a live edit cannot act.

    A strict subset of ``restart_required_config_keys()``, and the reason the
    two are not the same question: the rest of that set is frozen because
    applying it live would be UNSAFE (rebuild the model away from its
    checkpoint, advertise an asset the server cannot serve). These are frozen
    because applying it live would be a LIE — the value would read back correct
    from the trial's own config while the object using it kept the launch
    value.

    Exported for tooling that reports on config provenance: a checker may say
    "the running value is correct" only about a key some live consumer
    re-reads, and this set is the one it must never say it about.

    ``_STARTUP_ONLY_TRIAL_KEYS`` joined it in the audit-T2 sweep: same class,
    same lie, one file further out (``train_trial``'s startup block rather than
    a buffer constructor).
    """
    return (
        _CONSTRUCTION_ONLY_REPLAY_KEYS
        | _CONSTRUCTION_ONLY_PROBE_KEYS
        | _STARTUP_ONLY_TRIAL_KEYS
    )


# ---------------------------------------------------------------------------
# Driver-side keys (audit angle C, tranche 12: C2/C3/C4/C5).
#
# ⚑ EVERY SET ABOVE IS DERIVED BY RUNNING THE TRIAL-ACTOR RELOADER, so it
# structurally cannot see a key whose only consumer lives in the DRIVER
# process. `run.py` builds one `base_config` dict, `harness.run_tune` reads it
# once to construct the Tuner and to spawn the uvicorn server and the inference
# broker, and the trial actor never sees those reads at all. A live yaml edit is
# accepted, overlaid into the trial's config, reads back correct -- and cannot
# move a bound socket or a command line that was baked at launch.
#
# `restart_required_config_keys()`'s docstring states the consequence exactly:
# "for every key NOT in here, the live yaml is re-read each iteration and wins".
# That is the wrong story for these, and the gap was already half-known --
# `num_samples`, `max_concurrent_trials` and `gpus_per_trial` are in
# `_TOPOLOGY_KEYS`, while `cpus_per_trial`, read from the same dict in the same
# `tune.Tuner(...)` call, was in nothing. The prose at `_STARTUP_ONLY_TRIAL_KEYS`
# named three more (`tune_scheduler`, `pb2_perturbation_interval`,
# `tune_keep_last_experiments`) and stopped at prose.
#
# So: declare the class, and let `tests/test_driver_config_coverage.py` re-derive
# the membership by walking harness.py/run.py for `base_config.get("...")`. A key
# added to the driver without a classification FAILS that test. That is the same
# self-invalidating shape as `tests/test_reco_coverage.py`, which is what has
# kept the worker's published key set honest.

# Read ONCE by the driver, before or during Tuner/server/broker construction.
# A live edit cannot act, and -- unlike a restart-required key -- a restart of
# the TRIAL cannot act either: only re-running `run.py` can.
_DRIVER_LAUNCH_FIXED_KEYS = frozenset({
  # uvicorn command line, harness.py `_launch_distributed_server` (called once).
    "distributed_server_host",
    "distributed_server_port",
    "distributed_server_public_url",
    "distributed_server_root_override",
    "distributed_min_workers_per_trial",
    "distributed_max_worker_delta_per_rebalance",
    "distributed_upload_compact_shard_size",
    "distributed_upload_compact_max_age_seconds",
    "worker_self_register",
  # Read through a LOOP VARIABLE (`base_config.get(cfg_key)`), which is why the
  # coverage test has to account for non-literal reads rather than skip them.
  # Both are also `restart_required` on the TRIAL axis, and that is not a
  # contradiction: the driver bakes the path into the uvicorn command line at
  # launch, and `/v1/opening_book` keeps serving that file for the life of the
  # server, while the trial declines a live edit. Two independent reasons the
  # same edit does nothing, on two different clocks.
    "opening_book_path",
    "opening_book_path_2",
  # ⚑ FIRST-PROVISIONING-ONLY, which is stronger than launch-fixed and is why
  # these are called out rather than lumped in. `_prepare_distributed_worker_auth`
  # gates BOTH the `upsert_user` and the `.password` file write on
  # `not user_existed`, so against an existing server_root -- every run after the
  # first -- the yaml value reaches no consumer at any value. That is deliberate
  # (an admin rotation via manage_users.py must not be clobbered by yaml drift,
  # and the file stays symmetric with the user so workers can never be handed a
  # secret the server has not got). Rotate with manage_users.py; editing the yaml
  # alone is accepted and changes nothing, forever.
  #
  # ⚑ `distributed_worker_username` IS NOT ONE OF THESE -- see
  # `_DRIVER_DUAL_CLOCK_KEYS`. Round 1 of this PR put it here and wrote that
  # claim into the production yaml; it is false and the failure mode is a fleet
  # outage. The password keys really are inert; the username is not.
    "distributed_worker_password",
    "distributed_worker_password_env",
  # Ray driver: resources, scheduler choice and search space, all consumed
  # inside the single `tune.Tuner(...)` construction.
    "cpus_per_trial",
    "tune_metric",
    "tune_mode",
    "tune_scheduler",
    "tune_keep_last_experiments",
    "pb2_perturbation_interval",
    "pbt_synch",
    "asha_optimizer_only",
    "asha_optimizer_repeats",
    "gpbt_inertia_weight",
    "gpbt_winner_weight",
    "gpbt_quantile_fraction",
    "gpbt_resample_probability",
  # `search_optimizer` is NOT here: trainable_init.py:197 reads it per restore
  # to decide `model_only_restore`. See `_DRIVER_DUAL_CLOCK_KEYS`.
    "search_optimizer_choices",
    "search_smolgen",
    "search_nla",
    "search_feature_dropout_p",
    "search_w_volatility",
    "search_volatility_source",
})

# Read on BOTH clocks: once by the driver AND again per-iteration by the trial.
# ⚑ NO SINGLE CLASSIFICATION OF THESE IS CORRECT, which is the whole reason the
# set exists. A live edit moves the trial-side consumer and leaves the driver-side
# one on the launch value, so the two silently disagree; calling them "live" hides
# half the truth and calling them "launch-fixed" hides the other half.
#
#   tune_num_to_keep  -- harness.py `RunConfig` checkpoint retention (launch)
#                        vs `_prune_trial_checkpoints(keep_last=tc.tune_num_to_keep)`
#                        in trainable_phases.py (per-iteration).
#   shard_size        -- the server's `--upload-compact-shard-size` DEFAULT
#                        (launch, and it is the live default because
#                        distributed_upload_compact_shard_size is unset) vs the
#                        trainer's replay writer in trainable_init.py.
#   distributed_workers_per_trial -- the boot floor in run.py/harness.py (launch)
#                        vs `_ensure_distributed_workers`, which respawns with
#                        updated config each iteration (live).
#   device / seed / iterations -- driver-level resource and search-space
#                        decisions; the trial reads them too.
_DRIVER_DUAL_CLOCK_KEYS = frozenset({
    "tune_num_to_keep",
    "shard_size",
    "distributed_workers_per_trial",
    "device",
    "seed",
  # ⚑ THE DANGEROUS ONE, and round 1 of this PR got it wrong in the direction
  # that costs the most. `distributed_worker_username` is in
  # `_WORKER_LAUNCH_CONFIG_KEYS` (distributed_runtime.py), which
  # `_ensure_distributed_workers` hashes EVERY ITERATION: a live edit therefore
  # relaunches the whole fleet -- with a username the server has never heard of,
  # because `_prepare_distributed_worker_auth` provisions the user only on the
  # first run against a given server root. So the live edit is not free and is
  # not inert; it is a fleet-wide 401. The two clocks here disagree in the worst
  # possible way: the worker side applies it immediately and the server side
  # never does.
    "distributed_worker_username",
  # Driver: one grid_search decision inside the single `tune.Tuner(...)` build.
  # Trial: `trainable_init.py:197` reads it on every restore path to decide
  # `model_only_restore` when a checkpoint carries another trial's optimizer.
    "search_optimizer",
})

# ⚑ `iterations` IS DELIBERATELY NOT IN THE SET ABOVE. Round 1 put it there and
# the justification did not survive contact: dual-clock asserts that a live edit
# MOVES the trial-side consumer while the driver keeps the launch value, and for
# `iterations` the reloader REFUSES the edit (it is in
# `_STARTUP_ONLY_TRIAL_KEYS`), so neither clock moves and the two cannot
# disagree. `restart_required_config_keys()` is the whole story for it. The
# general rule this cost us: dual-clock and restart-required are CONTRADICTORY,
# and `test_the_classification_sets_do_not_overlap` now asserts that pair
# directly. Launch-fixed and restart-required are NOT contradictory -- they are
# orthogonal axes (the driver reads at launch; the trial refuses live), which is
# the true shape of the two opening-book paths.

# Written INTO base_config by the driver itself -- the `base_config.update({...})`
# at the end of `_launch_distributed_server` -- and never read from the yaml.
# Classifying them as config keys at all would be wrong. That update writes five
# keys; the other two (`distributed_server_public_url`,
# `distributed_worker_username`) are NOT here, because the yaml does supply them
# and the driver merely normalises them in place.
_DRIVER_DERIVED_KEYS = frozenset({
    "distributed_server_root",
    "distributed_server_url",
    "distributed_worker_password_file",
})


def driver_launch_fixed_config_keys() -> frozenset[str]:
    """Keys the DRIVER reads once; no trial-side consumer re-reads them.

    Neither a live reload nor a trial restart applies these -- only re-running
    `run.py` does. Reporting one as live-reloadable (the current default for
    "not in restart_required_config_keys()") tells an operator their edit is in
    effect when it is not, and reporting it as restart-required tells them the
    wrong restart.

    Deliberately NOT folded into `restart_required_config_keys()`: that set's
    contract is "a live reload refuses to apply this to a RUNNING TRIAL", and it
    is consumed by `_reload_yaml_into_config`'s branching. These keys never reach
    the reloader at all, so putting them there would make that function's
    derivation test claim a branch that does not exist.
    """
    return _DRIVER_LAUNCH_FIXED_KEYS


def driver_dual_clock_config_keys() -> frozenset[str]:
    """Keys read once by the driver AND per-iteration by the trial.

    A live edit moves the trial-side consumer only, so the two disagree until
    someone re-runs `run.py`. Callers that must name one authority for a key
    should refuse on this set rather than pick a side.
    """
    return _DRIVER_DUAL_CLOCK_KEYS


def driver_derived_config_keys() -> frozenset[str]:
    """Keys the driver WRITES into its own config dict; not yaml surface."""
    return _DRIVER_DERIVED_KEYS


def restart_required_config_keys() -> frozenset[str]:
    """Yaml keys a live reload refuses to apply to a running trial.

    The other half of this answer is the one that keeps biting: for every key
    NOT in here, the live yaml is re-read each iteration and wins, so the
    trial's ``params.json`` — which Ray writes from the launch config — is a
    historical snapshot wearing the name of current state. Ray rewrites the
    file on every checkpoint, so its mtime tracks the run while its contents do
    not: on 2026-07-26 it read ``opening_fen_dole_per_iter = 1`` and
    ``...retire_307.txt`` against a live 0 / ``retire_56`` (rl_loop_audit J5).
    For keys IN here the reverse holds — the yaml may have moved and the trial
    is still running the launch value until someone restarts it.

    Composed from the exact sets ``_reload_yaml_into_config`` branches on, so a
    key added to any of them is reclassified here in the same commit.
    ``test_restart_required_keys_match_the_reloader`` re-derives the answer by
    running the reloader instead of restating these sets, so the two cannot
    agree merely by construction.

    PB2-searched keys are also preserved across a reload, but that is a
    property of the trial's own ``pb2_bounds_*`` entries rather than of the key,
    so callers must exclude them separately.

    "Not in here" means the reloader overlays the yaml value into ``config``
    **unless the key is dead** — see ``dead_config_keys()``. A dead key is
    declined on live reload like a restart-required one, but it is deliberately
    NOT in this set, because a restart does not apply it either: it makes the
    trial REFUSE to start. Reporting it as restart-required would tell an
    operator to do the one thing that turns a silent no-op into a crash.
    Callers that branch on this set must handle that class separately;
    ``scripts/audit_realized_config.py`` takes it as its own injected argument.

    Being overlaid still does not mean a running component re-read it — that is
    a per-key fact about consumers, and ``construction_only_config_keys()`` is
    the part of it already proven. A tool that reads "overlaid" as "in effect"
    will report a knob as healthy precisely when it is inert.
    """
    return (
        _TOPOLOGY_KEYS
        | _RESUME_CONSTRUCTION_BOUND_KEYS
        | _LAUNCH_FIXED_ASSET_PATH_KEYS
        | _LIVE_RELOAD_SKIPPED_KEYS
    )


# The flat key set the last SUCCESSFUL load of each yaml path produced. Keyed
# by path because one trial process reloads exactly one yaml while the tests
# drive several, and only written after `flatten_run_config_defaults` returns —
# a rejected reload must not move the baseline, or the deletion it could not
# see would be reported against the wrong iteration.
_LAST_RELOAD_YAML_KEYS: dict[str, frozenset[str]] = {}


def reset_yaml_reload_key_tracking() -> None:
    """Forget every remembered key set. A fresh trial process starts empty."""
    _LAST_RELOAD_YAML_KEYS.clear()


def last_reload_yaml_keys(yaml_path: str | None) -> frozenset[str]:
    """The key set of the last successful load of *yaml_path*, for persisting.

    Empty when nothing has been loaded yet. The checkpoint writer banks this so
    the NEXT process can compare against the yaml this one was running -- see
    ``seed_yaml_reload_baseline``.
    """
    if not yaml_path:
        return frozenset()
    return _LAST_RELOAD_YAML_KEYS.get(yaml_path, frozenset())


def seed_yaml_reload_baseline(
    *, yaml_path: str | None, keys: Iterable[str], config: Mapping[str, object],
) -> None:
    """Adopt a PREVIOUS PROCESS's yaml key set and report what has gone missing.

    Closes the restart-shaped hole in the delete warning (review B2). The
    in-process baseline is seeded by this process's own startup reload, i.e.
    from the yaml as it is NOW, so a key deleted while the trial was down was
    invisible: `--resume` restores the saved trial config (which still carries
    the value, because neither the experiment-state overlay nor the reloader
    ever removes a key) and the first live reload compares the new yaml against
    a baseline that also lacks the key. Pause -> edit -> restart is the common
    operator path, and it was the one path the warning could not see.

    Called once at startup with the key set banked in the restored checkpoint's
    trial_meta.json. The comparison is the same one the live path makes, so the
    operator gets the same sentence.

    Silent, and leaves the baseline alone, when there is nothing to adopt: a
    fresh start, or a checkpoint written before this was banked.
    """
    banked = frozenset(keys)
    if not (yaml_path and banked):
        return
    current = _LAST_RELOAD_YAML_KEYS.get(yaml_path)
    if current is None:
        return
    _warn_on_keys_dropped_from_yaml(
        yaml_path=yaml_path, fresh=current, config=config, previous=banked,
        since="the checkpoint this process resumed from",
    )


def _warn_on_keys_dropped_from_yaml(
    *,
    yaml_path: str,
    fresh: Mapping[str, object] | Iterable[str],
    config: Mapping[str, object],
    previous: frozenset[str] | None = None,
    since: str = "the previous reload",
) -> None:
    """Name every key the yaml stopped setting while the trial still runs it.

    The reload is ADD/UPDATE-only: it iterates the fresh yaml and writes
    ``config[k] = v``, so there is no pass that could remove anything. Deleting
    a line, and writing ``key: null`` (``flatten_run_config_defaults`` drops
    None before the reloader sees it), are therefore both no-ops — and they are
    no-ops the three loud branches below cannot report, because every one of
    them keys off a key the fresh yaml still names (audit T4). An operator who
    removes a line to turn an experiment off gets exactly the silence the
    memory rule "a yaml edit may NOT be in effect" was written about.

    This is OBSERVABILITY ONLY. Reverting the value would be a behaviour change
    — the trial has been running on it, and the revert target (an argparse
    default, or a value from a config that no longer exists) is not knowable
    from the yaml — so it needs its own pre-registered entry. What is knowable,
    and what this says, is that the deletion did nothing.

    ⚑ IT FIRES ONCE PER DELETION, not once per iteration. The baseline moves to
    the new key set immediately afterwards, so the second reload has nothing to
    compare (review N5). That is the same noise trade the other three loud
    branches make, and it is a real limit of the observable: one WARNING in a
    665 s iteration is easy to lose in a multi-hour stdout stream. The durable
    form would be a report column, which rotates progress.csv and belongs with
    whoever next changes the report schema.

    ⚑ Two baselines feed this, and the restart one is why ``previous`` is an
    argument rather than a module lookup: it defaults to the in-process record
    of the last successful load, and ``seed_yaml_reload_baseline`` passes the
    set banked in the checkpoint so a deletion made while the trial was DOWN is
    reported too (review B2). Still invisible: a deletion made while down when
    the resumed checkpoint predates the banking, a salvage restart (a pool's key
    set describes another experiment and is deliberately not adopted), and a
    deletion of a key the trial does not carry at all.
    """
    if previous is None:
        previous = _LAST_RELOAD_YAML_KEYS.get(yaml_path)
    if previous is None:
        return
    missing = object()
    for key in sorted(previous - frozenset(fresh)):
        current = config.get(key, missing)
        if current is missing:
  # The yaml stopped naming a key the trial does not carry either: nothing
  # is in effect, so nothing was silently kept.
            continue
        log.warning(
            "YAML reload: %s is no longer set by %s (it was, as of %s) but the "
            "trial is STILL RUNNING %r — delete does not revert; write the old "
            "value explicitly. (`%s: null` is dropped before the reloader sees "
            "it, so it reads as a delete too.)",
            key, yaml_path, since, current, key,
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
  # Before the overlay, so `config` still holds the values this reload is
  # about to leave in place, and after the parse, so a rejected reload leaves
  # the baseline where it was.
        if live_reload:
            _warn_on_keys_dropped_from_yaml(yaml_path=yaml_path, fresh=fresh, config=config)
        _LAST_RELOAD_YAML_KEYS[yaml_path] = frozenset(fresh)
  # Derive searched keys from the config's own bounds (stable), not YAML.
        searched = {
            k.removeprefix("pb2_bounds_")
            for k in config if k.startswith("pb2_bounds_")
        }
        missing = object()
        for k, v in fresh.items():
            if k in searched or k.startswith("pb2_bounds_"):
                continue
            # A dead key gets its own message, NOT the "requires restart" one
            # below: restarting does not make it work, it makes the trial
            # refuse to start (``reject_dead_config_keys``). Telling an
            # operator to restart into a hard error would be the same lie in a
            # different place. Declining to apply also keeps the poison out of
            # `config`, so the trial keeps running on the value it has actually
            # been running on.
            #
            # FIRST, ahead of the skipped-keys branch, and it stays first even
            # though this diff removed the three gate_* keys from
            # `_LIVE_RELOAD_SKIPPED_KEYS`: the ordering is what makes
            # `dead_config_keys() & restart_required_config_keys()` free to be
            # empty. Were a future dead key also declared restart-required,
            # this branch is the one that must win -- the other tells an
            # operator to do the one thing that turns a silent no-op into a
            # crash. `tests/test_dead_config_keys.py` pins both halves.
            if (
                live_reload
                and k in _DEAD_CONFIG_KEY_INERT_VALUES
                and not is_inert_dead_config_value(v, _DEAD_CONFIG_KEY_INERT_VALUES[k])
            ):
                log.warning(
                    "YAML reload: %s=%r reaches no production code path and is "
                    "NOT applied — a restart will REFUSE to start with this "
                    "value, not honour it. Set it back to %r. See "
                    "reject_dead_config_keys.",
                    k, v, _DEAD_CONFIG_KEY_INERT_VALUES[k],
                )
                continue
            if (
                live_reload
                and k in _LIVE_RELOAD_SKIPPED_KEYS
  # `config.get(k)`, NOT `k in config and config[k] != v`. The old form
  # could not warn about a live ADD, and an add is the ONLY way these
  # keys ever change in practice: `configs/pbt2_small.yaml` ships no
  # `gate_*` key at all, so "just turn the gate on" is `gate_mode:
  # shadow` appearing where there was nothing. Under the old form that
  # took the silent-overlay path -- the value landed in `config`, the
  # `PromotionGate` had been constructed from the launch config
  # iterations earlier, and the operator got a knob that reads back
  # correctly and does nothing. Same argument for `lr_schedule`: the
  # trainer's scheduler is already built, so a live add is a no-op that
  # must announce itself.
                and config.get(k) != v
            ):
                log.warning(
                    "YAML reload: %s changed (%s -> %s) but requires restart — skipping",
                    k, config.get(k), v,
                )
                continue
            # Opening-asset PATHS are captured by the server process at launch
            # (create_app / --opening-*-path) and served from that snapshot,
            # while the manifest publisher re-reads config each iteration. A
            # live path change would advertise the new file's SHA while the
            # server still serves the old bytes -> worker download-verify
            # failures (Codex review). Freeze the paths to launch; a value
            # change needs a restart (which relaunches the server).
            if (
                live_reload
                and k in _LAUNCH_FIXED_ASSET_PATH_KEYS
                and config.get(k) != v
            ):
                # config.get(k) (not `k in config`) so a fresh live-ADD of the
                # path (absent -> a path) is caught too: the server was launched
                # without the flag, so advertising the new asset would 404 every
                # worker. Both add and change require a restart (Codex review).
                log.warning(
                    "YAML reload: %s changed (%s -> %s) but is server-launch-fixed"
                    " — skipping (restart to add/change the seed/book file)",
                    k, config.get(k), v,
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
    """Push lr / loss-weight keys from config into trainer.

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
    # Terminal-proximal outcome share. Pushed here so a live yaml edit reaches
    # the trained target at the next iteration instead of waiting for a
    # restart; `moves_left_max_plies` deliberately does NOT ride along (see the
    # Trainer constructor -- it re-interprets rows already in the window).
    if "wdl_terminal_outcome_plies" in config:
        trainer.wdl_terminal_outcome_plies = int(config["wdl_terminal_outcome_plies"])
    if "wdl_terminal_outcome_full_plies" in config:
        trainer.wdl_terminal_outcome_full_plies = int(
            config["wdl_terminal_outcome_full_plies"]
        )
    if "wdl_terminal_outcome_sf_frac" in config:
        trainer.wdl_terminal_outcome_sf_frac = float(config["wdl_terminal_outcome_sf_frac"])
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

    # SF target rebuild: construction-time on the Trainer, so it needs an
    # explicit push or a live yaml edit silently waits for the next restart.
    # This is what lets an SfTargetParams change hit the SF-labelled rows
    # already in the replay window at the next iteration, instead of only newly
    # generated data. All of them on healthy data -- see target_builder.
    want_rebuild = bool(config.get("rebuild_sf_targets", False))
    was_rebuild = trainer.rebuild_sf_targets
    want_params = resolve_sf_target_params(config)
    # `set_sf_target_rebuild` returns True only when something a live consumer
    # reads actually moved -- the flag itself, or the params while the rebuild
    # or sf_policy_sparse_ce is on. With both consumers off, `sf_policy_temp`
    # edits retune the worker's capture params and touch nothing here.
    if trainer.set_sf_target_rebuild(enabled=want_rebuild, params=want_params):
        log.warning(
            "SF target rebuild: enabled=%s (was %s) params=%s. While ON, SF "
            "targets are rebuilt from sf_multipv_raw for every SF-LABELLED "
            "replay row -- un-labelled rows have no SF target to rebuild. "
            "sf_p0_policy_target / sf_volatility_target are MASKED (their "
            "sources live on other shard rows) so has_sf_p0_frac reads 0. The "
            "frozen full-pass holdout is NOT rebuilt. Proof of effect is "
            "sf_rebuild_policy_frac in progress.csv, not this line; that "
            "column sitting BELOW sf_rebuild_wdl_frac means desynced Stockfish "
            "labels in the window, not rows the rebuild could not reach.",
            want_rebuild, was_rebuild, want_params,
        )

    cur_sf_frac = _dynamic_sf_wdl_weight(
        sf_wdl_start=tc.sf_wdl_frac,
        sf_wdl_floor=tc.sf_wdl_frac_floor,
        sf_wdl_floor_at_regret=tc.sf_wdl_floor_at_regret,
        regret_max=tc.sf_pid_wdl_regret_max,
        wdl_regret_used=ds.wdl_regret,
    )
    if cur_sf_frac is not None:
        trainer.sf_wdl_frac = cur_sf_frac
