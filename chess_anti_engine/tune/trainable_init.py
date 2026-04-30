"""Restore / bootstrap / replay-buffer initialization for the Ray Tune trainable.

Three phases that run once at trial startup (before the main loop):
  * _restore_checkpoint_or_salvage — Ray ckpt or salvage seed pool or fresh
  * _maybe_load_bootstrap          — pre-trained weights for fresh starts
  * _init_replay_buffers           — replay + holdout buffers, seeded
                                     from warmstart / shared / exploit
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model import (
    load_state_dict_tolerant,
    reinit_volatility_head_parameters_,
    zero_policy_head_parameters_,
)
from chess_anti_engine.replay import ArrayReplayBuffer, DiskReplayBuffer
from chess_anti_engine.replay.shard import copy_or_link_shard, iter_shard_paths
from chess_anti_engine.tune._utils import (
    SIDECAR_PID_STATE,
    SIDECAR_RNG_STATE,
    SIDECAR_TRIAL_META,
    load_optional_json,
    stable_seed_u32 as _stable_seed_u32,
)
from chess_anti_engine.tune.recovery import (
    _load_salvage_manifest_entry,
    _merge_pid_state_from_result_row,
    _select_salvage_seed_slot,
)
from chess_anti_engine.tune.replay_exchange import (
    _latest_trial_result_row,
    _refresh_replay_shards_on_exploit,
    _trial_replay_shard_dir,
)
from chess_anti_engine.tune.trainable_config_ops import (
    _TRAINER_WEIGHT_KEYS,
    _apply_lr_gamma_weights,
)
from chess_anti_engine.tune.trainable_metrics import _count_jsonl_rows
from chess_anti_engine.tune.trial_config import RestoreResult, TrialConfig

log = logging.getLogger(__name__)


def _load_model_only(maybe: Path, trainer, *, device: str, label: str) -> None:
    """Deserialize trainer.pt and load only the model state_dict.

    ``mmap=True`` keeps unread tensors (the optimizer state, ~2x model
    size for Adam-family) on disk so we don't materialize ~120MB of
    moments on the GPU just to ``del`` them.
    """
    ckpt = torch.load(str(maybe), map_location=device, mmap=True)
    load_state_dict_tolerant(trainer.model, ckpt["model"], label=label)
    del ckpt


def _restore_from_ray_checkpoint(
    *, ckpt, trainer, config: dict, device: str, trial_id: str, rr: RestoreResult,
) -> tuple[dict | None, dict | None]:
    """Load model/optimizer from a Ray checkpoint dir.

    Returns (restored_trial_meta, restored_rng_state). Caller derives
    ``owner_optimizer`` from trial_meta. Falls back to model-only restore
    if optimizer differs across trials (PB2 exploit clone donor used a
    different optimizer family).
    """
    ckpt_dir = Path(ckpt.to_directory())
    maybe = ckpt_dir / "trainer.pt"
    rr.restored_pid_state = load_optional_json(ckpt_dir / SIDECAR_PID_STATE)
    restored_rng_state = load_optional_json(ckpt_dir / SIDECAR_RNG_STATE)
    restored_trial_meta = load_optional_json(ckpt_dir / SIDECAR_TRIAL_META)
    if not maybe.exists():
        return restored_trial_meta, restored_rng_state

    owner_optimizer = (
        str(restored_trial_meta.get("optimizer", "") or "")
        if isinstance(restored_trial_meta, dict) else ""
    )
    current_optimizer = str(config.get("optimizer", "nadamw")).lower()
    model_only_restore = False
    if isinstance(restored_trial_meta, dict):
        owner_trial_id = str(restored_trial_meta.get("owner_trial_id", ""))
        if owner_trial_id and owner_trial_id != trial_id:
            if owner_optimizer:
                model_only_restore = owner_optimizer.lower() != current_optimizer
            elif bool(config.get("search_optimizer", False)):
                model_only_restore = True
    if model_only_restore:
        _load_model_only(maybe, trainer, device=device, label="checkpoint_model_only")
        trainer._init_swa()  # noqa: SLF001
        rr.startup_source = "checkpoint_model_only"
    else:
        trainer.load(maybe)
        rr.startup_source = "checkpoint"
    if "lr" in config:
        trainer.set_peak_lr(float(config["lr"]), rescale_current=False)
    return restored_trial_meta, restored_rng_state


def _restore_from_salvage_pool(
    *, config: dict, trainer, device: str, trial_id: str, trial_dir: Path,
    rr: RestoreResult,
) -> None:
    """Load weights + optional optimizer/PID from a salvage seed slot."""
    seed_pool_dir = Path(str(config.get("salvage_seed_pool_dir"))).expanduser()
    if not seed_pool_dir.is_dir():
        raise RuntimeError(f"salvage_seed_pool_dir not found: {seed_pool_dir}")

    picked_dir, picked_slot, num_slots = _select_salvage_seed_slot(
        seed_pool_dir=seed_pool_dir, trial_dir=trial_dir, trial_id=trial_id,
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
        seed_pool_dir=seed_pool_dir, slot=rr.seed_warmstart_slot,
    )
    manifest_row: dict | None = (
        seed_entry.get("result_row") if isinstance(seed_entry, dict) else None
    )
    if not isinstance(manifest_row, dict):
        manifest_row = None
    donor_cfg = manifest_row.get("config") if manifest_row else None
    if (
        bool(config.get("salvage_restore_donor_config", False))
        and isinstance(donor_cfg, dict)
    ):
        _apply_donor_config_overlay(config, donor_cfg, trainer)

    if bool(config.get("salvage_restore_full_trainer_state", False)):
        trainer.load(maybe)
    else:
        _load_model_only(maybe, trainer, device=device, label="salvage")
    if config.get("salvage_reinit_volatility_heads", False):
        reinit = reinit_volatility_head_parameters_(trainer.model)
        if reinit:
            print(f"[trial] Reinitialized salvage volatility heads: {', '.join(reinit)}")
    print(
        f"[trial] salvage warmstart loaded slot={rr.seed_warmstart_slot} "
        f"of {rr.seed_warmstart_slots_total} from {rr.seed_warmstart_dir}"
    )
    if bool(config.get("salvage_restore_pid_state", False)):
        rr.restored_pid_state = load_optional_json(rr.seed_warmstart_dir / SIDECAR_PID_STATE)
        rr.restored_pid_state, pid_manifest_overrides = _merge_pid_state_from_result_row(
            pid_state=rr.restored_pid_state, result_row=manifest_row,
        )
        if pid_manifest_overrides:
            print("[trial] salvage PID overrides from manifest row: " + ", ".join(pid_manifest_overrides))


def _apply_donor_config_overlay(config: dict, donor_cfg: dict, trainer) -> None:
    """Copy lr/cosmos_gamma/loss-weights from donor manifest row into config + trainer.

    Dual-write pattern: ``config[k] = donor_cfg[k]`` so live YAML reload
    preserves the donor's values across iter boundaries; ``setattr`` (via
    ``_apply_lr_gamma_weights``) makes them take effect before the first
    iteration. ``rescale_current_lr=False`` because the trainer was just
    built — no scheduler progress to rescale against.
    """
    for k in ("lr", "cosmos_gamma", *_TRAINER_WEIGHT_KEYS):
        if k in donor_cfg:
            config[k] = donor_cfg[k]
    _apply_lr_gamma_weights(trainer, config, rescale_current_lr=False)


def _apply_restored_trial_meta(rr: RestoreResult, restored_trial_meta: dict | None) -> tuple[str, str]:
    """Pull owner/salvage/global-iter fields from checkpoint metadata into ``rr``.
    Returns (owner_trial_id, owner_optimizer)."""
    if not isinstance(restored_trial_meta, dict):
        return "", ""
    rr.restored_owner_trial_dir = str(restored_trial_meta.get("owner_trial_dir", ""))
    rr.salvage_origin_used = bool(restored_trial_meta.get("salvage_origin_used", rr.salvage_origin_used))
    rr.salvage_origin_slot = int(restored_trial_meta.get("salvage_origin_slot", rr.salvage_origin_slot))
    rr.salvage_origin_slots_total = int(
        restored_trial_meta.get("salvage_origin_slots_total", rr.salvage_origin_slots_total)
    )
    rr.salvage_origin_dir = str(restored_trial_meta.get("salvage_origin_dir", rr.salvage_origin_dir or ""))
    rr.global_iter = int(restored_trial_meta.get("global_iter", 0))
    return (
        str(restored_trial_meta.get("owner_trial_id", "")),
        str(restored_trial_meta.get("optimizer", "") or ""),
    )


def _fork_rng_for_exploit_clone(
    *,
    config: dict, trainer, base_seed: int, trial_id: str, trial_dir: Path,
    rr: RestoreResult, owner_trial_id: str, owner_optimizer: str,
) -> np.random.Generator:
    """PB2 exploit clone: fork RNG/torch seeds so recipients don't replay donor
    openings, plus inherit donor's opp_strength_ema. Returns the new RNG."""
    restore_rows = _count_jsonl_rows(trial_dir / "result.json")
    fork_seed = _stable_seed_u32(
        base_seed, trial_id, "exploit", restore_rows, int(getattr(trainer, "step", 0)),
    )
    rr.active_seed = int(fork_seed)
    rng = np.random.default_rng(rr.active_seed)
    torch.manual_seed(rr.active_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rr.active_seed)
    print(
        f"[trial] PB2 exploit restore detected: owner={owner_trial_id} "
        f"recipient={trial_id} fork_seed={rr.active_seed} "
        f"owner_optimizer={owner_optimizer or 'unknown'} "
        f"recipient_optimizer={str(config.get('optimizer', 'nadamw')).lower()} "
        f"restore_mode={rr.startup_source}"
    )
    if rr.restored_owner_trial_dir:
        donor_row = _latest_trial_result_row(Path(rr.restored_owner_trial_dir))
        if donor_row is not None:
            rr.opp_strength_ema = float(donor_row.get(
                "opponent_strength_ema", donor_row.get("opponent_strength", 0.0),
            ))
            print(f"[trial] exploit: inherited donor opp_strength_ema={rr.opp_strength_ema:.1f}")
    return rng


def _restore_checkpoint_or_salvage(
    *,
    config: dict,
    trainer,
    device: str,
    trial_id: str,
    trial_dir: Path,
    base_seed: int,
    active_seed: int,
    rng: np.random.Generator,
    ckpt,
) -> tuple[RestoreResult, np.random.Generator]:
    """Restore from Ray checkpoint, salvage seed pool, or start fresh.

    Mutates *trainer* (model/optimizer load), *config* (donor config overlay),
    and may reseed *rng* on exploit clone.  Returns ``(restore_result, rng)``.
    """
    rr = RestoreResult(active_seed=active_seed)
    restored_rng_state: dict | None = None
    restored_trial_meta: dict | None = None

    if ckpt is not None:
        restored_trial_meta, restored_rng_state = _restore_from_ray_checkpoint(
            ckpt=ckpt, trainer=trainer, config=config, device=device,
            trial_id=trial_id, rr=rr,
        )
    elif isinstance(config.get("salvage_seed_pool_dir"), str) and str(config.get("salvage_seed_pool_dir", "")).strip():
        _restore_from_salvage_pool(
            config=config, trainer=trainer, device=device,
            trial_id=trial_id, trial_dir=trial_dir, rr=rr,
        )

    owner_trial_id, owner_optimizer = _apply_restored_trial_meta(rr, restored_trial_meta)
    rr.cross_trial_restore = bool(
        ckpt is not None and owner_trial_id and owner_trial_id != trial_id
    )
    if rr.cross_trial_restore:
        if rr.startup_source == "checkpoint":
            rr.startup_source = "exploit_restore"
        elif rr.startup_source == "checkpoint_model_only":
            rr.startup_source = "exploit_restore_model_only"

    if restored_rng_state is not None and not rr.cross_trial_restore:
        try:
            rng.bit_generator.state = restored_rng_state
        except (ValueError, TypeError, KeyError) as exc:
            log.warning(
                "[trial] failed to restore RNG state from checkpoint (%s); continuing with current rng",
                exc,
            )

    if ckpt is not None and rr.cross_trial_restore:
        rng = _fork_rng_for_exploit_clone(
            config=config, trainer=trainer, base_seed=base_seed,
            trial_id=trial_id, trial_dir=trial_dir, rr=rr,
            owner_trial_id=owner_trial_id, owner_optimizer=owner_optimizer,
        )

    return rr, rng


def _maybe_load_bootstrap(
    *,
    tc: TrialConfig,
    trainer,
    device: str,
    ckpt,
    restore: RestoreResult,
) -> None:
    """Load pre-trained bootstrap model weights on fresh start (no checkpoint/warmstart).

    IMPORTANT: Only loads MODEL WEIGHTS — skips optimizer/scheduler/step.
    The bootstrap was trained for ~13k steps with its own LR schedule; carrying that
    state causes: (1) step=13323 skips warmup entirely, (2) scheduler resumes
    mid-cosine-cycle with near-zero LR then spikes on restart, (3) optimizer momentum
    buffers from bootstrap data cause wrong gradient directions on selfplay data,
    (4) PB2's lr perturbation has no effect because scheduler's base_lr is locked.
    """
    if not (tc.bootstrap_checkpoint and ckpt is None and (not restore.seed_warmstart_used)):
        return
    bp = Path(tc.bootstrap_checkpoint)
    if not bp.exists():
        print(f"[trial] WARNING: bootstrap checkpoint not found: {bp}")
        return
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
    trainer._init_swa()  # noqa: SLF001
    del ckpt_data


def _seed_replay_from_warmstart(
    *, restore: RestoreResult, replay_shard_dir: Path,
) -> None:
    """Salvage-seed warmstart: only when no shards exist locally and not cross-trial restore."""
    if not (
        restore.seed_warmstart_used
        and not restore.cross_trial_restore
        and restore.seed_warmstart_replay_dir is not None
        and restore.seed_warmstart_replay_dir.is_dir()
        and not iter_shard_paths(replay_shard_dir)
    ):
        return
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


def _seed_replay_from_shared_shards(
    *, tc: TrialConfig, restore: RestoreResult, replay_shard_dir: Path,
) -> int:
    """Iter-0 shared-bootstrap seed: only on fresh start (no shards + not cross-trial)."""
    if (
        not tc.shared_shards_dir
        or iter_shard_paths(replay_shard_dir)
        or restore.cross_trial_restore
    ):
        return 0
    src = Path(tc.shared_shards_dir)
    if not src.is_dir():
        return 0
    replay_shard_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for sp in iter_shard_paths(src):
        copy_or_link_shard(sp, replay_shard_dir / sp.name)
        copied += 1
    if copied:
        print(f"[trial] Seeded {copied} shared iter-0 shards from {src}")
    return int(copied)


def _refresh_replay_after_exploit(
    *, tc: TrialConfig, config: dict, restore: RestoreResult,
    replay_shard_dir: Path, trial_dir: Path, current_window: int,
) -> int:
    """Run exploit-replay refresh + bump current_window to fit the kept shards.

    The window pre-bump prevents DiskReplayBuffer's _enforce_window from
    aggressively evicting on the first add_many post-restore.
    """
    if not (restore.cross_trial_restore and tc.exploit_replay_refresh_enabled):
        return current_window
    donor_trial_dir = (
        Path(restore.restored_owner_trial_dir).expanduser()
        if restore.restored_owner_trial_dir else None
    )
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
    kept = iter_shard_paths(replay_shard_dir)
    if kept:
        new_window = max(int(current_window), len(kept) * tc.shard_size)
        print(
            f"[trial] exploit restore: pre-set window={new_window} "
            f"for {len(kept)} kept shards"
        )
        return new_window
    return current_window


def _init_replay_buffers(
    *,
    tc: TrialConfig,
    config: dict,
    restore: RestoreResult,
    trial_dir: Path,
    work_dir: Path,
    rng,
    ckpt,
) -> tuple:
    """Set up replay + holdout buffers, seeding shards from warmstart/shared/exploit.

    Returns ``(buf, holdout_buf, current_window, replay_shard_dir,
    selfplay_shards_dir)``.
    """
    current_window = tc.replay_window_start
    replay_shard_dir = _trial_replay_shard_dir(config=config, trial_dir=trial_dir)
    selfplay_shards_dir = work_dir / "selfplay_shards"
    selfplay_shards_dir.mkdir(parents=True, exist_ok=True)

    _seed_replay_from_warmstart(restore=restore, replay_shard_dir=replay_shard_dir)
    shared_shards_loaded = _seed_replay_from_shared_shards(
        tc=tc, restore=restore, replay_shard_dir=replay_shard_dir,
    )
    current_window = _refresh_replay_after_exploit(
        tc=tc, config=config, restore=restore,
        replay_shard_dir=replay_shard_dir, trial_dir=trial_dir,
        current_window=current_window,
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
    seeded_replay_start = bool(ckpt is not None or restore.seed_warmstart_used or shared_shards_loaded > 0)
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

    return buf, holdout_buf, current_window, replay_shard_dir, selfplay_shards_dir
