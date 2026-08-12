from __future__ import annotations

import logging
import os
import subprocess
import time

# Optional dependency module (Ray Tune). Kept import-light so the core package
# works without installing `.[tune]`.
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model, resume_model_config_from_arch
from chess_anti_engine.stockfish import StockfishPool, StockfishUCI, pid_from_config
from chess_anti_engine.train import (
    Trainer,
    resolve_zclip_max_norm,
    trainer_kwargs_from_config,
)
from chess_anti_engine.tune._utils import (
    DURABLE_BEST_STATE,
    DURABLE_GATE_STATE,
    load_optional_json,
    resolve_local_override_root as _resolve_local_override_root,
)
from chess_anti_engine.tune._utils import (
    stable_seed_u32 as _stable_seed_u32,
)
from chess_anti_engine.tune.distributed_runtime import (
    _ensure_distributed_workers,
    _ensure_inference_broker,
    _publish_distributed_trial_state,
    _quarantine_inbox_shards,
    _resolve_shared_cache_root,
    _set_active_run_prefix,
    _stop_process,
    _stop_worker_processes,
    _trial_server_dirs,
)
from chess_anti_engine.tune.trainable_config_ops import (
    _reload_yaml_into_config,
    _resolve_pause_marker_paths,
    _resolve_sims,
    _sync_trainer_weights,
    _wait_if_paused,
    last_reload_yaml_keys,
    seed_yaml_reload_baseline,
)
from chess_anti_engine.tune.trainable_init import (
    _init_era_probes,
    _init_replay_buffers,
    _maybe_load_bootstrap,
    _restore_checkpoint_or_salvage,
    peek_checkpoint_arch,
)
from chess_anti_engine.tune.promotion_gate import (
    GateHoldController,
    PromotionGate,
    gate_config_from_dict,
)
from chess_anti_engine.tune.trainable_metrics import _compute_drift_metrics
from chess_anti_engine.tune.trainable_phases import (
    _finalize_iteration,
    _run_eval_games,
    _run_pid_and_eval,
    _run_selfplay_phase,
    _run_training_and_gating,
)
from chess_anti_engine.tune.trainable_report import (
    _guard_checkpoint_index,
    _init_status_csv,
    _reset_best_on_cross_trial_restore,
    _save_trial_checkpoint,
    _update_best_model,
)
from chess_anti_engine.tune.trial_config import DifficultyState, RestoreResult, TrialConfig

import re as _re
import contextlib

_AVG_BATCH_RE = _re.compile(rb"avg=([0-9.]+)")

  # Smoothing factor for opp_strength EMA — higher = more responsive.
_OPP_EMA_ALPHA = 0.3


def _set_log_level(config: dict) -> None:
    """Set chess_anti_engine logger level from env or yaml; log if DEBUG."""
    lvl_name = (os.environ.get("CHESS_LOG_LEVEL") or config.get("log_level") or "INFO").upper()
    lvl = getattr(logging, lvl_name, logging.INFO)
    logging.getLogger("chess_anti_engine").setLevel(lvl)
    if lvl <= logging.DEBUG:
        logging.getLogger().info("chess_anti_engine logging at DEBUG (gumbel batch-size profiling enabled)")


def _durable_trial_dir(staging_trial_dir: Path) -> Path:
    """The trial directory whose contents survive a process restart.

    `ray.train.get_context().get_trial_dir()` does NOT return one.  It returns
    the per-Ray-session driver staging directory,
    ``/tmp/ray/session_<id>/artifacts/<ts>/tune/driver_artifacts/<trial>``.
    Ray syncs that UP to persistent storage; it never syncs it back DOWN, and
    every restart opens a new Ray session, so the staging dir is EMPTY at
    process start.  Anything the trial reads back from it during startup
    therefore always reads as missing -- silently, because "sidecar absent" is
    also the legitimate first-run case.

    Measured live 2026-07-25: 68 session staging dirs for one trial, each
    holding its own `best.json` with only that session's minimum, and 33 of 33
    post-restart rows in `result.json` re-seeding `best_loss` from that
    iteration's training loss.

    The persistent location is `StorageContext.trial_fs_path` -- the same
    directory the synced copies already land in, so nothing moves.  Ray's
    artifact sync copies staging files up without deleting anything else
    there, so a file only we write is left alone.  Fall back to the staging
    dir when storage is not a local filesystem: writing a torch checkpoint
    through a pyarrow filesystem is a different problem, and today's behaviour
    is the honest default until someone needs it.

    `_resolve_pause_marker_paths` documents this same hazard for pause
    markers; it was never carried across to the state the trial reloads.
    """
    try:
        from pyarrow.fs import LocalFileSystem
        from ray.train._internal.session import get_session

        storage = get_session().storage  # pyright: ignore[reportOptionalMemberAccess]
        if not isinstance(storage.storage_filesystem, LocalFileSystem):
            print(
                f"[trial] storage is {type(storage.storage_filesystem).__name__}, "
                f"not a local filesystem; keeping per-trial state in the "
                f"per-session staging dir (it will NOT survive a restart): "
                f"{staging_trial_dir}",
                flush=True,
            )
            return staging_trial_dir
        durable = Path(storage.trial_fs_path)
        durable.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(
            f"[trial] could not resolve the durable trial dir ({exc}); keeping "
            f"per-trial state in the per-session staging dir (it will NOT "
            f"survive a restart): {staging_trial_dir}",
            flush=True,
        )
        return staging_trial_dir
    return durable


def _load_best_state(best_state_path: Path) -> tuple[float, float, str]:
    """Return (best_loss, opp_strength_ema, best_source) from best.json.

    ``best_source`` says which ruler produced ``best_loss`` -- the holdout
    (``test_loss``) or the training batches (``train_loss``). They are not
    comparable, so the caller needs it to avoid pitting one against the other.
    An older best.json without the field is treated as ``train_loss``, the
    conservative reading: it lets the first holdout evaluation take over the
    ruler rather than being locked out by a number it cannot beat.

    ``opp_strength_ema`` is the FALLBACK source for the opponent-strength EMA:
    the checkpoint's trial_meta.json is preferred because it is rewritten every
    iteration while this file is rewritten only when a best-model candidate is
    accepted. It stops being decorative here (audit T1) -- until then it was
    read and then overwritten by the restore result one line later.
    """
    d = load_optional_json(best_state_path) or {}
    source = str(d.get("source", "train_loss"))
    return (
        float(d.get("best_loss", d.get("loss", float("inf")))),
        float(d.get("opp_strength_ema", 0.0)),
        source if source in ("test_loss", "train_loss") else "train_loss",
    )


def _startup_opp_strength_ema(*, restore: RestoreResult, best_json_ema: float) -> float:
    """Decide which opponent-strength EMA this process continues from (T1).

    The restore wins ONLY when it carries one -- the checkpoint's
    trial_meta.json, or a PB2 donor's result row. Otherwise ``best.json``'s
    copy stands, and only when neither exists does the loop fall back to its
    ``== 0.0`` re-seed from the first iteration's raw ``opponent_strength``.

    ``train_trial`` used to assign ``restore.opp_strength_ema``
    UNCONDITIONALLY, one line after loading best.json's value, and that field
    was 0.0 on every path except the exploit clone -- which ``num_samples: 1``
    makes unreachable. So the EMA was re-seeded at every process start, on the
    single row the post-restart winrate truncation makes least trustworthy
    (6/6 live process segments). It is the tune scheduler's own objective
    (``GPBTPairwiseScheduler._metric``), the ``--salvage-metric`` family, and
    the sibling-share ranking metric, so the discontinuity sat in the number
    the run is steered by.

    Prints its source because "the EMA is continuous now" is not observable
    from the value alone: a re-seed and a restore look identical on a row where
    the EMA happens to be near the raw strength.
    """
    if restore.opp_strength_ema is not None:
        resolved, source = float(restore.opp_strength_ema), f"restore/{restore.startup_source}"
    elif best_json_ema != 0.0:
        resolved, source = float(best_json_ema), "best.json"
    else:
        resolved, source = float(best_json_ema), "none (the loop will re-seed from iteration 1)"
    print(
        f"[trial] opponent-strength EMA at startup: {resolved:.4f} source={source}",
        flush=True,
    )
    return resolved


def _startup_gate_state(*, gate_state_path: Path, restore: RestoreResult) -> dict:
    """The promotion gate's state for this process, own file first (T10).

    A trial with its own ``gate_state.json`` uses it and nothing else: an
    ordinary ``--resume`` must never be handed a window from somewhere else.
    Only when there is none -- the first process of a salvage-restarted
    experiment -- does the pool's exported copy seed the gate, which is what
    stops a salvage restart from silently starting with an empty window while a
    resume keeps its own.

    Safe without the anchor, which pools deliberately do not carry:
    ``GateHoldController.create`` RELEASES a restored hold whose
    ``gate_promoted_model.pt`` is absent rather than serving something else.

    A PRESENT-BUT-UNREADABLE own file is treated as own state, not as absence
    (review N3). ``load_optional_json`` returns None for a missing path and for
    a JSONDecodeError alike, so keying the pool fallback off its result would
    let a truncated own file -- a crash mid-write is exactly how one is produced
    -- hand this trial another lineage's window. Empty is the safe reading: the
    gate refills its window, which is what a corrupt file already meant.
    """
    own = load_optional_json(gate_state_path)
    if own is not None:
        return own
    if gate_state_path.exists():
        print(
            f"[trial] promotion-gate state at {gate_state_path} exists but did "
            f"not parse; starting from an EMPTY window rather than adopting "
            f"anything else",
            flush=True,
        )
        return {}
    if restore.restored_gate_state is None:
        return {}
    seeded = restore.restored_gate_state
    print(
        f"[trial] promotion-gate state seeded from the salvage pool: "
        f"matches={seeded.get('matches', 0)} holds={seeded.get('holds', 0)}. "
        f"The anchor is not exported with a pool, so a restored hold releases "
        f"rather than serving a model this pool does not carry.",
        flush=True,
    )
    return seeded


def _assert_distributed_configured(tc: TrialConfig) -> None:
    if not (
        tc.distributed_workers_per_trial > 0
        and bool((tc.distributed_server_root or "").strip())
        and bool((tc.distributed_server_url or "").strip())
    ):
        raise RuntimeError(
            "Trainable requires distributed selfplay to be configured: "
            "distributed_workers_per_trial>0 plus server_root/server_url. "
            "run.py auto-enables this for --mode train; if you see this from "
            "a direct trainable invocation, call run.py --mode train (or tune) "
            "or populate the distributed_* keys."
        )


def _make_stockfish_uci(tc: TrialConfig, *, nodes: int, multipv: int) -> StockfishUCI:
    """Single-engine factory used by both gate and eval SF init."""
    return StockfishUCI(
        tc.stockfish_path, nodes=nodes, multipv=multipv,
        hash_mb=tc.sf_hash_mb,
        syzygy_path=tc.stockfish_syzygy_path or tc.syzygy_path,
        nice=tc.sf_nice,
    )


def _init_eval_stockfish(tc: TrialConfig) -> StockfishUCI | None:
    """Dedicated fixed-strength engine for eval games (separate node limit)."""
    if tc.eval_games <= 0:
        return None
    return _make_stockfish_uci(tc, nodes=tc.eval_sf_nodes, multipv=1)


def _init_pid(tc: TrialConfig, config: dict, restored_pid_state, sf):
    """Build PID from config + optional restored state. Sync local SF nodes if any."""
    if not tc.sf_pid_enabled:
        return None
    pid = pid_from_config(config)
    if restored_pid_state is not None:
        try:
            pid.load_state_dict(restored_pid_state)
        except (KeyError, ValueError, TypeError) as exc:
  # Silent fallback would lose regret/winrate history; surface it.
            logging.getLogger(__name__).warning("PID state restore failed (using defaults): %s", exc)
  # Keep gate-game SF in lock-step with restored PID nodes so gate plays
  # at the same difficulty distributed workers do. Cosmetic — fail quietly.
    if sf is not None:
        with contextlib.suppress(OSError, ValueError, AttributeError):
            sf.set_nodes(int(pid.nodes))
    return pid


def _load_puzzle_suite(tc: TrialConfig):
    if not (tc.puzzle_epd and tc.puzzle_interval > 0):
        return None
    from chess_anti_engine.eval import load_epd
    try:
        return load_epd(tc.puzzle_epd)
    except FileNotFoundError:
        return None


# Keys already warned about, so the line is one per TRANSITION rather than one
# per iteration -- a per-iteration flood is how a warning stops being read.
# Cleared when the key goes back to true, so a flip-flop is fully reported.
#
# ⚑ PER PROCESS, NOT PER TRIAL. Keyed by config key alone, so a second trial in
# the same actor process would find the flag already set and stay silent. Inert
# in production (`max_concurrent_trials: 1`, one trial per actor) and left that
# way deliberately: keying by trial id would mean threading one through a helper
# that has no other use for it, to fix a case the production config cannot
# reach. If concurrent trials per actor ever ship, key this by
# `(trial_id, key)` -- it is the only change needed.
_DECLINED_OFF_WARNED: set[str] = set()


def _warn_declined_off(key: str, enabled: bool, obj: object, iteration_idx: int) -> None:
    """Say so when a live yaml edit turned a one-way key off and was ignored."""
    if enabled or obj is None:
        _DECLINED_OFF_WARNED.discard(key)
        return
    if key in _DECLINED_OFF_WARNED:
        return
    _DECLINED_OFF_WARNED.add(key)
    logging.getLogger("chess_anti_engine.iter").warning(
        "[trial]%s is false in the live config but its object is already "
        "running (iter %d): this key is live ONLY in the on direction, so the "
        "edit is ACCEPTED AND IGNORED until the trial restarts. Restart the "
        "trial to apply it.",
        key, iteration_idx,
    )


def _lazy_construct_iter_helpers(
    *, shard_prefetcher, async_test_eval, tc: TrialConfig,
    distributed_dirs: dict, iteration_idx: int,
):
    """First-iter activation gates for prefetcher + async_test_eval.

    Lazily constructed inside the iter loop so live YAML reload can flip
    them on; constructing at trial-setup time misses --resume strips of
    these keys from tuner.pkl param_space.

    ⚑ THESE TWO KEYS ARE LIVE IN ONE DIRECTION ONLY (audit angle C, C1). The
    `is None` guard is the whole gate: once the object exists, `tc.distributed_*`
    is never consulted again -- the iter loop passes the object on
    unconditionally -- so `true -> false` in the live yaml is accepted, reads
    back correct from the trial's own config, and does nothing until a restart.
    Neither key is in `restart_required_config_keys()`, so a provenance report
    reads yaml == realized as healthy precisely when the edit is inert.

    That is not academic: the mitigation an operator reaches for under memory
    pressure is `distributed_prefetch_shards: false` (the queue holds ~102 MB
    per shard -- audit A19), and it cannot take effect on a run whose restarts
    cost a day. Tearing the objects down mid-iteration is NOT the fix here --
    the prefetcher owns decoded shards the ingest phase is about to consume and
    `AsyncTestEval` owns an in-flight snapshot eval, so a live stop would have
    to drain both, which is a behaviour change to a running trial rather than an
    observability one. So: make the declined OFF AUDIBLE, once per transition.
    `_warn_declined_off` is the single observation that distinguishes "the edit
    took" from "the edit was accepted and ignored" -- there was none before.
    """
    if shard_prefetcher is None and tc.distributed_prefetch_shards:
        from chess_anti_engine.tune.distributed_runtime import _iter_shard_paths_nested
        from chess_anti_engine.tune.prefetch import BackgroundShardPrefetcher
        max_queued_bytes = int(tc.distributed_prefetch_max_queued_mb) * 1024 * 1024
        shard_prefetcher = BackgroundShardPrefetcher(
            inbox_dir=distributed_dirs["inbox_dir"],
            path_iter=_iter_shard_paths_nested,
            max_queued_bytes=max_queued_bytes,
        )
        shard_prefetcher.start()
  # REALIZED budget, read back off the constructed object rather than
  # printed from the config value handed in. A key that is accepted and
  # then not applied is this codebase's signature defect; printing the
  # input would report success either way, so the log would be evidence of
  # nothing. `_max_queued_bytes` is what `_scan_once` actually compares.
        logging.getLogger("chess_anti_engine.iter").info(
            "[trial]BackgroundShardPrefetcher started (iter %d) "
            "max_queued=%.0f MB (distributed_prefetch_max_queued_mb=%d)",
            iteration_idx,
            shard_prefetcher._max_queued_bytes / (1024 * 1024),
            int(tc.distributed_prefetch_max_queued_mb),
        )
    if async_test_eval is None and tc.distributed_async_test_eval:
        from chess_anti_engine.train.async_eval import AsyncTestEval
        async_test_eval = AsyncTestEval()
        logging.getLogger("chess_anti_engine.iter").info("[trial]AsyncTestEval enabled (iter %d)", iteration_idx)
    _warn_declined_off(
        "distributed_prefetch_shards", tc.distributed_prefetch_shards,
        shard_prefetcher, iteration_idx,
    )
    _warn_declined_off(
        "distributed_async_test_eval", tc.distributed_async_test_eval,
        async_test_eval, iteration_idx,
    )
    return shard_prefetcher, async_test_eval


def _log_iter_phase_split(
    *, trainer, iteration_idx: int, distributed_dirs: dict,
    worker_log_offsets: dict[str, int],
    selfplay_secs: float, train_secs: float, iter_so_far: float,
) -> None:
    other_secs = max(0.0, iter_so_far - selfplay_secs - train_secs)
    denom = max(iter_so_far, 1e-9)
    logging.getLogger("chess_anti_engine.iter").info(
        "iter %d phase split: selfplay=%.1fs (%.0f%%) train=%.1fs (%.0f%%) other=%.1fs (%.0f%%) total_so_far=%.1fs",
        iteration_idx,
        selfplay_secs, 100.0 * selfplay_secs / denom,
        train_secs, 100.0 * train_secs / denom,
        other_secs, 100.0 * other_secs / denom,
        iter_so_far,
    )
    try:
        trainer.writer.add_scalar("iter/selfplay_secs", float(selfplay_secs), iteration_idx)
        trainer.writer.add_scalar("iter/train_secs", float(train_secs), iteration_idx)
        trainer.writer.add_scalar("iter/other_secs", float(other_secs), iteration_idx)
        avg_batch = _avg_batch_from_worker_logs(distributed_dirs, worker_log_offsets)
        if avg_batch > 0:
            trainer.writer.add_scalar("selfplay/avg_batch_size", float(avg_batch), iteration_idx)
    except Exception:
        pass  # TB scalar emission must never break training


def _cleanup_trial_resources(
    *, shard_prefetcher, async_test_eval,
    distributed_worker_procs, distributed_inference_broker_proc,
    sf, eval_sf,
) -> None:
    if shard_prefetcher is not None:
        shard_prefetcher.stop()
    if async_test_eval is not None:
  # Drain in-flight eval, then signal the long-lived worker thread to
  # exit cleanly (avoids the daemon thread getting torn down mid-CUDA-
  # call on interpreter exit).
        async_test_eval.collect(timeout=10.0)
        async_test_eval.shutdown(timeout=5.0)
    _stop_worker_processes(distributed_worker_procs)
    _stop_process(distributed_inference_broker_proc)
    if sf is not None:
        sf.close()
    if eval_sf is not None:
        eval_sf.close()


def _avg_batch_from_worker_logs(distributed_dirs, offsets: dict[str, int]) -> float:
    """Tail each worker log since last call and return mean of `avg=N` values
    in any new `gumbel profile (...)` line. Returns 0 if no new samples.
    Mutates ``offsets`` in place to record post-read positions per file.
    """
  # distributed_dirs is a dict from _trial_server_dirs (key "workers_root").
  # Pre-2026-04-29 this used getattr() and silently returned 0 every iter,
  # killing the selfplay/avg_batch_size TB scalar.
    workers_root = distributed_dirs.get("workers_root")
    if workers_root is None:
        return 0.0
    samples: list[float] = []
    for log_path in Path(workers_root).glob("worker_*/worker.log"):
        key = str(log_path)
        try:
            sz = log_path.stat().st_size
        except OSError:
            continue
        last = int(offsets.get(key, 0))
        # Truncated/rotated → reset and skip the now-stale tail.
        if sz < last:
            offsets[key] = sz
            continue
        if sz == last:
            continue
        try:
            with log_path.open("rb") as fh:
                fh.seek(last)
                chunk = fh.read(sz - last)
        except OSError:
            continue
        offsets[key] = sz
        for m in _AVG_BATCH_RE.finditer(chunk):
            try:
                samples.append(float(m.group(1)))
            except ValueError:
                continue
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def _maybe_reset_holdout_on_drift(
    *, holdout_buf, drift, tc: TrialConfig, holdout_frozen: bool, holdout_generation: int,
    async_test_eval,
) -> tuple[bool, int]:
    """Clear holdout when input-drift L2 crosses threshold; bumps generation
    counter to mark which holdout the test_loss in this row reflects.

    THE ONLY LOOP-TIME HOLDOUT MUTATOR THE START-SIDE GUARD CANNOT COVER (audit
    L2 residual). ``_ingest_train_arrays`` is the one that guard keys on, and
    ``holdout_state.load_holdout_rows`` needs no ordering at all because it runs
    during buffer construction; this one is neither, and it runs at the worst
    possible point in the iteration -- after the previous
    iteration's ``AsyncTestEval.start()`` and BEFORE the ``collect()`` that
    ends it -- so the eval thread can be walking the very buffer being cleared.
    The start-side guard cannot cover this one: whether a reset fires depends
    on drift metrics computed a full iteration AFTER the start it would have to
    veto, so the only place the two can be ordered is here, at the mutator.

    ``async_test_eval`` is REQUIRED and unconditional rather than defaulted,
    because both failure modes are invisible from this frame:

    * Concurrent clear. ``clear()`` REBINDS ``_chunks`` instead of mutating it,
      so the eval thread's iterator over the old deque survives and there is no
      ``RuntimeError: deque mutated during iteration`` -- ``rows_slice_arrays``
      instead reads the new ``_size`` of 0 and ``_gather_rows`` raises
      ``ValueError: ArrayReplayBuffer is empty`` (measured, mid-pass and
      before the first batch alike). Loud, but it costs the row.
    * The SILENT one, and it is not even a race: an eval that finished cleanly
      against generation G is collected AFTER this bump, so ``_finalize_iteration``
      stamps a generation-G measurement with ``holdout_generation`` G+1. The
      counter exists precisely to say which holdout a ``test_loss`` came from,
      and on every reset iteration it said the wrong one.

    Draining first fixes both WHEN THE DRAIN SUCCEEDS: the reader is gone
    before the writer runs, and the pre-reset result is DISCARDED rather than
    relabelled. The reset row therefore carries no ``test_*`` at all, which is
    the honest reading -- the set it would describe no longer exists. One row,
    on a path production has switched off (``reset_holdout_on_drift: false``,
    and ``drift_threshold`` 0.0 independently disables it).

    The drain is BEST-EFFORT, and the exception is stated rather than implied:
    ``AsyncTestEval.collect`` returns ``(None, -1)`` from BEFORE its
    ``with self._lock`` block when the wait times out, so ``_inflight_iter``
    stays set, ``has_inflight()`` stays True, and the clear below proceeds with
    the eval thread possibly still walking the buffer -- i.e. both failure
    modes above are reopened on that one path. Deliberately not handled by
    skipping the reset: a wedged eval thread would then suppress every drift
    reset indefinitely, trading a bounded one-row loss for a gate that silently
    stops firing, which is the worse failure here. It is made LOUD instead --
    the post-drain ``has_inflight()`` re-check below is the observation that
    says the guarantee did not hold this time.
    """
    if not (
        tc.reset_holdout_on_drift
        and tc.drift_threshold > 0.0
        and drift.drift_input_l2 > tc.drift_threshold
    ):
        return holdout_frozen, holdout_generation
  # Drain BEFORE the clear, never after: after is the race. Guarded by
  # has_inflight() so a reset with no eval outstanding does not pay the full
  # collect timeout waiting on an event nobody will set.
    if async_test_eval is not None and async_test_eval.has_inflight():
        print(
            "[trial] holdout drift reset: draining and DISCARDING the in-flight "
            "async holdout eval before clearing the set -- it measured "
            f"generation {int(holdout_generation)}, which is about to stop "
            f"existing. test_* is absent for this row by design (audit L2)",
            flush=True,
        )
        logging.getLogger(__name__).warning(
            "holdout drift reset: discarding the in-flight async eval "
            "(generation %d); test_* absent for this row (audit L2)",
            int(holdout_generation),
        )
        async_test_eval.collect(timeout=tc.distributed_async_test_eval_timeout_s)
  # collect() returns on timeout BEFORE it clears _inflight_iter, so this
  # re-check is not redundant: True here means the drain did not take and the
  # clear below runs under a live reader after all. Say so rather than
  # proceeding silently -- it is the only frame that can tell.
        if async_test_eval.has_inflight():
            print(
                "[trial] holdout drift reset: the drain TIMED OUT after "
                f"{float(tc.distributed_async_test_eval_timeout_s):.1f}s -- the eval "
                "thread may still be reading the holdout, so the clear below is "
                "unordered against it. Expect either a ValueError from the eval "
                "thread or a generation-mislabelled test_* row (audit L2)",
                flush=True,
            )
            logging.getLogger(__name__).error(
                "holdout drift reset: drain timed out after %.1fs; clearing the "
                "holdout under a possibly-live eval reader (audit L2)",
                float(tc.distributed_async_test_eval_timeout_s),
            )
    holdout_buf.clear()
    return False, holdout_generation + 1


def _maybe_bump_generation_on_ruler_change(
    *, observed_ruler: str, known_ruler: str, holdout_generation: int,
) -> tuple[str, int]:
    """Bump the generation when the MEASUREMENT changed, not just the set.

    `holdout_generation` is the ruler identity the best-model comparison keys
    on, but until now only the holdout SET could move it: a drift reset, or a
    restart that could not restore the rows. The measurement applied to the
    set was outside the counter entirely, so PR #277's swap from 2560
    priority-weighted draws to a deterministic 2000-row pass kept generation
    1 on both sides, `_update_best_model` took its same-ruler branch, and
    4.85326 was promoted over 4.90535 across two different instruments -- a
    -0.156-nat step that was -5.70 sd on policy and flat on WDL, which is what
    dropping priority weighting looks like, not what learning looks like.

    ``observed_ruler`` is `TrainMetrics.eval_ruler`: the identity of the eval
    that produced THIS iteration's `test_loss`, carried on the metrics from
    the code that ran them. On the async path those metrics were computed one
    iteration ago, and that is the point -- the number and its ruler stay
    together no matter how late the number arrives.

    Three rules, all about never claiming to know more than we do:

    * an empty ``observed_ruler`` changes nothing. No holdout eval ran this
      iteration (the G5 refill window, an async cold start, a timed-out
      collect), or the source digest was unavailable; either way there is no
      evidence of a new ruler and inventing a bump would hand the best model
      over for nothing.
    * an empty ``known_ruler`` adopts without bumping. This is the migration
      path: every checkpoint written before this existed carries no ruler, and
      the record it defends was measured on the ruler that is running right
      now. Bumping there would invalidate a valid best model on the first
      restart, and -- because it happens once per unknown, not once per change
      -- would teach the operator to ignore the handover line.
    * a different non-empty ruler bumps by one and says so. Monotone, so a
      revert to a previous measurement reads as a third generation rather
      than aliasing back onto the first; the cost is one extra handover and
      the alternative is a comparison that can silently become valid again.
    """
    if not observed_ruler or observed_ruler == known_ruler:
        return known_ruler or observed_ruler, int(holdout_generation)
    if not known_ruler:
        return observed_ruler, int(holdout_generation)
    print(
        f"[trial] holdout RULER changed: the measurement was {known_ruler} and "
        f"is now {observed_ruler}; generation {holdout_generation} -> "
        f"{holdout_generation + 1} so the best-model record is handed over "
        f"instead of compared across instruments",
        flush=True,
    )
    return observed_ruler, int(holdout_generation) + 1


def _maybe_freeze_holdout(
    *, holdout_buf, tc: TrialConfig, holdout_frozen: bool,
) -> bool:
    """Freeze the holdout once it holds `freeze_holdout_at` rows.

    Freezing is one-way within a process, and now across processes too: the
    flag rides in the checkpoint. A holdout restored at or above the threshold
    therefore freezes on its first iteration back, which is the point --
    before the holdout was persisted this test could only ever fire against a
    fresh post-restart sample, so `freeze_holdout_at` re-froze a DIFFERENT
    2000 rows after every restart and the ruler moved anyway.
    """
    if holdout_frozen or tc.freeze_holdout_at <= 0:
        return holdout_frozen
    return len(holdout_buf) >= tc.freeze_holdout_at


def _log_pid_iter_summary(
    *, iteration_idx: int, pid_result, sp,
) -> None:
    """One-line iter summary: PID-relevant winrate, ema, regret, nodes, game counts."""
    cur_wr = pid_result.curriculum_winrate_raw
    logging.getLogger("chess_anti_engine.iter").info(
        "iter %d wr: cur_raw=%s ema=%.3f → regret=%.3f nodes=%d (sp_games=%d cur_games=%d)",
        iteration_idx,
        f"{cur_wr:.3f}" if cur_wr is not None else "n/a",
        pid_result.pid_ema_wr,
        pid_result.wdl_regret_next, pid_result.sf_nodes_next,
        int(sp.total_selfplay_games), int(sp.total_curriculum_games),
    )


def _build_trial_model_config(tc: TrialConfig) -> ModelConfig:
    """Build the ModelConfig from a TrialConfig — pure mapping."""
    return ModelConfig(
        kind=tc.model,
        embed_dim=tc.embed_dim,
        num_layers=tc.num_layers,
        num_heads=tc.num_heads,
        embed_dim_by_layer=tc.embed_dim_by_layer,
        ffn_mult=tc.ffn_mult,
        ffn_mult_by_layer=tc.ffn_mult_by_layer,
        use_smolgen=tc.use_smolgen,
        use_nla=tc.use_nla,
        use_qk_rmsnorm=tc.use_qk_rmsnorm,
        use_gradient_checkpointing=tc.gradient_checkpointing,
        input_pos_encoding=tc.input_pos_encoding,
        qkv_projection=tc.qkv_projection,
        use_deepnorm=tc.use_deepnorm,
        policy_encoding=tc.policy_encoding,
        input_history_encoding=tc.input_history_encoding,
        history_rep_fix=tc.history_rep_fix,
        input_extra_features=tc.input_extra_features,
        use_dynamic_relations=tc.use_dynamic_relations,
        dynamic_relation_count=tc.dynamic_relation_count,
        policy_dynamic_relations=tc.policy_dynamic_relations,
        input_global_embedding=tc.input_global_embedding,
        input_global_embedding_channels=tc.input_global_embedding_channels,
        input_square_embedding=tc.input_square_embedding,
        smolgen_mode=tc.smolgen_mode,
        smolgen_pooling=tc.smolgen_pooling,
        smolgen_hidden_channels=tc.smolgen_hidden_channels,
        smolgen_hidden_sz=tc.smolgen_hidden_sz,
        smolgen_gen_sz=tc.smolgen_gen_sz,
        smolgen_bias_scale=tc.smolgen_bias_scale,
        smolgen_bias_norm=tc.smolgen_bias_norm,
        arc_attention_bias=tc.arc_attention_bias,
        smolgen_relation_basis=tc.smolgen_relation_basis,
        smolgen_relation_norm=tc.smolgen_relation_norm,
        smolgen_relation_coeff_norm=tc.smolgen_relation_coeff_norm,
        smolgen_relation_scale=tc.smolgen_relation_scale,
        phase_output_adapter=tc.phase_output_adapter,
        phase_output_adapter_dim=tc.phase_output_adapter_dim,
        phase_smolgen=tc.phase_smolgen,
        phase_piece_thresholds=tc.phase_piece_thresholds,
        categorical_head_coupled=tc.categorical_head_coupled,
    )


def _quarantine_inbox_on_resume(
    *, distributed_dirs: dict, restore, trial_id: str,
) -> None:
    """On resume, move any preexisting inbox shards out of the way so they
    don't get ingested as if produced by the resuming model generation."""
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


def train_trial(config: dict):
    """Ray Tune trainable.

    Reports metrics per outer-loop iteration. Supports checkpoint restore.
    """

    from ray.tune import Checkpoint
    from ray.tune import get_checkpoint as _tune_get_checkpoint
    from ray.tune import get_context as _tune_get_context
    from ray.tune import report as _tune_report

  # Re-read YAML and overlay all keys EXCEPT those PB2 is actively searching.
  # This lets --resume pick up config changes without clobbering tuned hyperparams.
    _reload_yaml_into_config(config, config.get("_yaml_config_path"))
    if "device" not in config:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    tc = TrialConfig.from_dict(config)
    _set_log_level(config)
    _yaml_path = tc._yaml_config_path

    _ctx = _tune_get_context()
    base_seed = tc.seed
    trial_id = str(_ctx.get_trial_id() or "trial")
    active_seed = int(_stable_seed_u32(base_seed, trial_id))
    rng = np.random.default_rng(active_seed)
    torch.manual_seed(active_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(active_seed)

    device = tc.device
    config_model_cfg = _build_trial_model_config(tc)
  # On resume, rebuild the model at the CHECKPOINT's topology, not the trial
  # config's. The trial config can drift from the saved model (e.g. a
  # variable-width FFN schedule that the config no longer carries); building
  # from config then loading tolerantly silently drops the mismatched blocks
  # and crashes the optimizer at the first step(). Topology follows the saved
  # arch; the four encoding keys (input_extra_features / policy_encoding /
  # input_history_encoding / history_rep_fix) stay config-driven so a
  # deliberate warm-start migration (v1 -> v2_threats) is still honored. No
  # checkpoint / older checkpoint without arch / salvage -> config-driven.
    ckpt = _tune_get_checkpoint()
    ckpt_arch = peek_checkpoint_arch(ckpt)
    if ckpt_arch is not None:
        model_cfg = resume_model_config_from_arch(ckpt_arch, config_model_cfg)
        if model_cfg != config_model_cfg:
            print(
                "[trial] resume: rebuilding model from checkpoint arch "
                "(topology from checkpoint, encoding from config)"
            )
    else:
        model_cfg = config_model_cfg
    model = build_model(model_cfg)

  # Ray-provided trial directory for per-trial WORKING files (checkpoint
  # staging, replay shards, TensorBoard logs). It is per-Ray-session and does
  # not survive a restart, so anything this trial reads back at startup uses
  # `durable_dir` below instead.
  # IMPORTANT: Do NOT use config["work_dir"] here — it points to the shared
  # runs/pbt2_small/ directory. Using it caused all 10 trials to write
  # checkpoints to the same directory, making PB2 unable to clone checkpoints
  # ("no checkpoint for trial X. Skip exploit.").
    trial_dir = Path(_ctx.get_trial_dir())
    work_dir = trial_dir
    work_dir.mkdir(parents=True, exist_ok=True)

  # Per-trial state the NEXT process has to read back does not belong in
  # `trial_dir` — that is a per-Ray-session staging dir, empty at every start.
    durable_dir = _durable_trial_dir(trial_dir)
    if durable_dir != trial_dir:
        print(f"[trial] per-trial state that outlives this process: {durable_dir}", flush=True)

  # Before the first report+checkpoint: a stale index seeded from the driver
  # would otherwise overwrite existing checkpoint dirs, including the one this
  # process just restored from. `checkpoint_NNNNNN/` dirs are written to
  # PERSISTENT storage, so the guard has to scan there — pointed at the staging
  # dir it would find nothing and silently never fire.
    _guard_checkpoint_index(trial_dir=durable_dir)

  # Compact status CSV — reset on each process start so checkpoint-restore rows don't accumulate.
    _STATUS_CSV_PATH = _init_status_csv(trial_dir)

    gate_state_path = durable_dir / DURABLE_GATE_STATE
    best_state_path = durable_dir / DURABLE_BEST_STATE
    best_dir = durable_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_loss, opp_strength_ema, best_source = _load_best_state(best_state_path)

    trainer_ctor = trainer_kwargs_from_config(
        config | {"device": device}, log_dir=work_dir / "tb",
    )
    trainer = Trainer(model, model_config=model_cfg, **trainer_ctor)

    restore, rng = _restore_checkpoint_or_salvage(
        config=config, trainer=trainer, device=device,
        trial_id=trial_id, trial_dir=trial_dir,
        base_seed=base_seed, active_seed=active_seed,
        rng=rng, ckpt=ckpt,
    )
    restored_pid_state = restore.restored_pid_state
    global_iter = restore.global_iter
    opp_strength_ema = _startup_opp_strength_ema(
        restore=restore, best_json_ema=opp_strength_ema,
    )

  # The delete-observable's restart leg (review B2). The in-process baseline was
  # seeded by THIS process's startup reload, i.e. from the yaml as it is now, so
  # a key removed while the trial was down matched nothing. The checkpoint banks
  # the key set the previous process was running; comparing against it here is
  # what makes `pause -> edit the yaml -> restart` -- the operator path that
  # actually happens -- report the same sentence a live deletion does.
    seed_yaml_reload_baseline(
        yaml_path=_yaml_path, keys=restore.restored_yaml_keys, config=config,
    )

    best_loss, best_source = _reset_best_on_cross_trial_restore(
        restore=restore, best_loss=best_loss, best_source=best_source,
        best_dir=best_dir, best_state_path=best_state_path,
    )

  # Rebuild tc — _restore_checkpoint_or_salvage may overlay donor config.
    tc = TrialConfig.from_dict(config)

  # Resolved AFTER the restore, which is the only way a salvage restart can
  # find a gate window at all (audit T10). The donor-config overlay the restore
  # may apply touches lr/gamma/loss weights only, so the gate is still built
  # from the LAUNCH config here: window shape and mode are construction-time,
  # so a live yaml edit does not silently change what a half-filled window
  # means (see the live-reload rules in CLAUDE.md). ``gate_config_from_dict``
  # also raises on a non-zero ``gate_games``, so the removed 1-sim
  # vs-Stockfish gate cannot be turned back on by editing the yaml.
    _gate_state = _startup_gate_state(gate_state_path=gate_state_path, restore=restore)
    gate_match_idx = int(_gate_state.get("matches", 0))
    gate = PromotionGate(cfg=gate_config_from_dict(config))
    gate.load_state_dict(_gate_state)

  # Every piece of gate-side mutable loop state lives in this object, not in
  # locals: the anchor path (None while the gate is off, so a disabled feature
  # copies nothing), the hold path, the restart restore, the retry ageing and
  # the sign-validity history. Six loose locals here were individually mutable
  # and individually untested.
  #
  # Constructed AFTER the restore, because it needs the resumed iteration: the
  # anchor on disk is admitted only if its stamp says it was written within
  # gate_max_hold_iters of where this process is starting. Without that, an
  # off -> on cycle re-arms whatever gate_promoted_model.pt an earlier era left
  # in durable_dir and a hold publishes it to the whole fleet (audit G3-5). Its
  # counters come from the same gate_state.json the window does, so a restart
  # no longer resets anchor_refresh_failures to zero.
    _hold_state = _gate_state.get("hold")
    gate_hold = GateHoldController.create(
        gate, durable_dir=durable_dir,
        state=_hold_state if isinstance(_hold_state, dict) else None,
        current_iteration=int(global_iter),
    )

    buf, holdout_buf, current_window, replay_shard_dir, selfplay_shards_dir = _init_replay_buffers(
        tc=tc, config=config, restore=restore,
        trial_dir=trial_dir, work_dir=work_dir, rng=rng, ckpt=ckpt,
    )

    _maybe_load_bootstrap(tc=tc, trainer=trainer, device=device, ckpt=ckpt, restore=restore)

  # Both settled by _init_replay_buffers against what the holdout sidecar
  # actually restored — NOT reset per process, which is what made every
  # restart silently swap the ruler `test_loss` is read against.
    holdout_frozen = bool(restore.holdout_frozen)
    holdout_generation = int(restore.holdout_generation)
  # The measurement half of the ruler's identity, restored alongside the set's
  # (see _maybe_bump_generation_on_ruler_change). Empty on a checkpoint written
  # before the ruler was identified at all, and on a fresh start.
    holdout_ruler = str(restore.holdout_ruler)

    _assert_distributed_configured(tc)
  # No in-process Stockfish. Its only consumer was the removed gate-check
  # (``gate_games`` vs-SF games at 1 sim); distributed workers run their own SF
  # for selfplay, and the anchored gate plays no games at all. Kept as an
  # explicit None rather than dropped, because ``sf`` still travels to the PID
  # (``set_nodes``) and to ``DifficultyState.from_pid``, both None-tolerant.
    sf: StockfishUCI | StockfishPool | None = None

    distributed_server_root = _resolve_local_override_root(
        raw_root=str(tc.distributed_server_root),
        tune_work_dir=tc.work_dir or str(work_dir),
        suffix="server",
    )
    _set_active_run_prefix(server_root=distributed_server_root, trial_id=trial_id)
    distributed_dirs = _trial_server_dirs(server_root=distributed_server_root, trial_id=trial_id)
  # Share the trainer's torch.compile cache with selfplay workers so FX
  # graphs + autotuned Triton kernels survive restarts (otherwise the
  # trainer writes to /tmp and pays a fresh ~7min compile each restart).
    from chess_anti_engine.worker import _configure_shared_compile_cache
    _configure_shared_compile_cache(
        cache_dir=_resolve_shared_cache_root(config, distributed_server_root)
    )
    distributed_worker_procs: list[subprocess.Popen[bytes]] = []
  # Boxed, not a bare local: _run_selfplay_phase can relaunch a dead broker
  # mid-iteration, and a returned handle would leave this scope holding a
  # corpse if the ingest raised after the relaunch (orphaning a broker that
  # holds VRAM and shm slots). The box makes every relaunch visible here
  # immediately, including on the exception path into the finally below.
    _broker_box: list[subprocess.Popen[bytes] | None] = [None]

    if ckpt is not None:
        _quarantine_inbox_on_resume(
            distributed_dirs=distributed_dirs, restore=restore, trial_id=trial_id,
        )

    eval_sf = _init_eval_stockfish(tc)
    pid = _init_pid(tc, config, restored_pid_state, sf)
    puzzle_suite = _load_puzzle_suite(tc)
  # Loaded ONCE, here, before the first iteration: the whole point of the
  # instrument is that a forgetting hinge shows up in days, which needs the
  # column populated from row 1 rather than from whenever someone remembers.
  # `_init_era_probes` prints each set's digest and row count.
    era_probes = _init_era_probes(tc=tc)

    pause_marker_paths = _resolve_pause_marker_paths(tc=tc, trial_dir=trial_dir)
    # One-shot startup log so we can compare against the path graceful_restart
    # writes if the pause hook ever fails to fire again. flush=True because
    # this prints once and the actor's stdout buffer is large.
    _present = [p for p in pause_marker_paths if p.exists()]
    print(
        f"[trial] graceful pause marker paths: "
        f"{[str(p) for p in pause_marker_paths]} "
        f"(present at startup: {[str(p) for p in _present]})",
        flush=True,
    )

    current_nodes_init = int(pid.nodes) if pid is not None else tc.sf_nodes
    sims_init = _resolve_sims(tc, trainer, max_sims=tc.mcts_simulations)
    prev_published_model_sha = _publish_distributed_trial_state(
        trainer=trainer,
        config=config,
        model_cfg=model_cfg,
        server_root=distributed_server_root,
        trial_id=trial_id,
        training_iteration=0,
        trainer_step=int(getattr(trainer, "step", 0)),
        sf_nodes=current_nodes_init,
        mcts_simulations=int(sims_init),
        wdl_regret=float(pid.wdl_regret) if pid is not None else -1.0,
        pause_selfplay=False,
        pause_reason="",
        reuse_existing_model_for_same_step=(ckpt is not None),
  # A resume mid-hold must not lift the brake for one publish. This is the
  # only publish outside _publish_iteration_model, so it takes the raw path
  # rather than the controller; note_published() is deliberately NOT called --
  # the loop's first real publish is what starts the sign-validity history.
        override_model_path=gate_hold.hold_path,
    )
    _broker_box[0] = _ensure_inference_broker(
        config=config,
        trial_id=trial_id,
        trial_dir=trial_dir,
        publish_dir=distributed_dirs["publish_dir"],
        proc=_broker_box[0],
    )
    distributed_worker_procs = _ensure_distributed_workers(
        config=config,
        trial_dir=trial_dir,
        trial_id=trial_id,
        procs=distributed_worker_procs,
    )
    distributed_pause_active = False
    distributed_pause_started_at: float | None = None

    ckpt_dir = work_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

  # Lazily-constructed: build on first iter where the live YAML reload says
  # they're enabled. Constructing at trial-setup time (before the per-iter
  # YAML reload) misses cases where --resume strips these keys from
  # tuner.pkl's param_space — the YAML still has them, but the initial `tc`
  # was built from the stripped config.
  #
  # shard_prefetcher polls the inbox during train phase and decodes zarr in
  # a daemon thread; iter-boundary ingest then just does in-memory append.
  # async_test_eval snapshots the post-train model and evals on a long-lived
  # daemon thread so the ~30-50s holdout pass overlaps next iter's selfplay
  # (test_metrics in row N measures iter N-1's model — see test_metrics_source_iter).
    shard_prefetcher = None
    async_test_eval = None

    try:
        iterations = tc.iterations
        completed_iterations = 0
        _worker_log_offsets: dict[str, int] = {}
        while completed_iterations < iterations:
            iteration_zero_based = int(global_iter)
            iteration_idx = iteration_zero_based + 1
            iter_t0 = time.monotonic()

  # Live-reload YAML config each iteration so changes apply without restart.
            _reload_yaml_into_config(config, _yaml_path, live_reload=True)
            tc = TrialConfig.from_dict(config)

  # Surprise-priority shaping knobs apply at shuffle append, so pushing
  # them before this iteration's ingest makes yaml edits live.
            buf.sf_gap_priority_weight = tc.replay_sf_gap_priority_weight
            buf.fast_low_surprise_priority = tc.replay_fast_low_surprise_priority
            buf.diff_focus_pol_scale = tc.diff_focus_pol_scale
            buf.diff_focus_q_weight = tc.diff_focus_q_weight

  # zclip's fixed hard cap is read once in ZClip.__init__, so without this
  # push a yaml edit to zclip_max_norm silently does nothing to a running
  # trainer (rl_loop_audit I11/I13). Log only on transition.
  # Resolve through the SAME helper the constructor uses -- a bare
  # config.get() would read an absent key as "cap disabled".
            _old_cap = trainer.grad_clip_max_norm
            if trainer.set_grad_clip_max_norm(resolve_zclip_max_norm(config)):
                print(
                    f"[trial] zclip_max_norm {_old_cap} -> {trainer.grad_clip_max_norm} "
                    f"(live reload, iteration {iteration_idx})",
                    flush=True,
                )

            shard_prefetcher, async_test_eval = _lazy_construct_iter_helpers(
                shard_prefetcher=shard_prefetcher,
                async_test_eval=async_test_eval,
                tc=tc,
                distributed_dirs=distributed_dirs,
                iteration_idx=iteration_idx,
            )

            in_salvage_startup_grace = (
                restore.startup_source == "salvage"
                and bool(restore.salvage_origin_used)
                and int(iteration_zero_based) < tc.salvage_startup_no_share_iters
            )
            _wait_if_paused(
                pause_marker_paths=pause_marker_paths,
                poll_seconds=tc.pause_poll_seconds,
                trial_id=trial_id,
                iteration=iteration_idx,
            )

  # Difficulty knobs used for this iteration's selfplay (kept fixed across
  # selfplay chunks). PID is updated once per iteration AFTER training so
  # changes align to net updates rather than chunk noise.
            ds = DifficultyState.from_pid(pid, sf, tc)

            base_sims = tc.mcts_simulations
            sims = _resolve_sims(tc, trainer, max_sims=base_sims)

            t_selfplay_start = time.monotonic()
            sp, prev_published_model_sha, current_window = _run_selfplay_phase(
                tc=tc, config=config, trainer=trainer, model_cfg=model_cfg,
                buf=buf, holdout_buf=holdout_buf,
                holdout_frozen=holdout_frozen,
                rng=rng,
                distributed_dirs=distributed_dirs,
                distributed_server_root=distributed_server_root,
                distributed_worker_procs=distributed_worker_procs,
                broker_proc_box=_broker_box,
                prev_published_model_sha=prev_published_model_sha,
                ds=ds,
                sims=sims,
                iteration_idx=iteration_idx,
                iteration_zero_based=iteration_zero_based,
                trial_id=trial_id,
                trial_dir=trial_dir,
                selfplay_shards_dir=selfplay_shards_dir,
                replay_shard_dir=replay_shard_dir,
                current_window=current_window,
                in_salvage_startup_grace=in_salvage_startup_grace,
                prefetcher=shard_prefetcher,
                reuse_existing_model_for_same_step=(ckpt is not None),
                hold=gate_hold,
            )
            t_selfplay_secs = time.monotonic() - t_selfplay_start
            distributed_pause_active = False
            distributed_pause_started_at = None
            if sp.should_retry:
  # No games ingested, so the gate never observes or decides this
  # iteration -- but the fleet is still held. Age the hold anyway, or
  # a run stuck in retry during a hold holds forever with the release
  # counter frozen.
                gate_hold.on_aborted_iteration()
                continue

            holdout_frozen = _maybe_freeze_holdout(
                holdout_buf=holdout_buf, tc=tc, holdout_frozen=holdout_frozen,
            )

            drift = _compute_drift_metrics(
                buf=buf, holdout_buf=holdout_buf,
                drift_sample_size=tc.drift_sample_size,
            )
            holdout_frozen, holdout_generation = _maybe_reset_holdout_on_drift(
                holdout_buf=holdout_buf, drift=drift, tc=tc,
                holdout_frozen=holdout_frozen, holdout_generation=holdout_generation,
  # The eval this reset would corrupt is the one started at the END of the
  # previous iteration and not yet collected; the reset has to order itself
  # against it (audit L2 residual).
                async_test_eval=async_test_eval,
            )

            _sync_trainer_weights(trainer, config, tc, ds)

            t_train_start = time.monotonic()
            tr = _run_training_and_gating(
                tc=tc, trainer=trainer, buf=buf, holdout_buf=holdout_buf,
  # POST-freeze-check value, and it has to be: the async eval started at the
  # end of THIS iteration is read by the eval thread during the NEXT
  # iteration's ingest, and this is the flag that ingest will use to decide
  # whether to append to the holdout (audit L2).
                holdout_frozen=holdout_frozen,
                config=config, model_cfg=model_cfg,
                device=device,
                ds=ds,
                sims=sims,
                sp=sp,
                positions_ingested=sp.replay_positions_ingested,
                imported_samples_this_iter=sp.imported_samples_this_iter,
                gate=gate,
                gate_match_idx=gate_match_idx,
                gate_state_path=gate_state_path,
                gate_hold=gate_hold,
                distributed_server_root=distributed_server_root,
                iteration_idx=iteration_idx,
                iteration_zero_based=iteration_zero_based,
                trial_id=trial_id,
                restore=restore,
                async_test_eval=async_test_eval,
            )
            t_train_secs = time.monotonic() - t_train_start
            gate_match_idx = tr.gate_match_idx
  # The verdict acts on the NEXT publish, not on the weights just trained.
  # The return value carries the anchor's refresh health back into the
  # decision so `gate_anchor_refresh_failures` reaches progress.csv; a
  # `hold` that never reaches `on_decision` is the mutation
  # `test_the_loop_body_calls_on_decision_with_the_verdict_and_keeps_it`
  # exists to catch.
            tr.gate_decision = gate_hold.on_decision(
                tr.gate_decision, iteration=int(iteration_idx),
            )
  # Before the checkpoint is written and before the best-model comparison
  # reads the generation: the number that arrived in `tr.test_metrics` has to
  # be judged against the generation of the ruler that produced IT.
            holdout_ruler, holdout_generation = _maybe_bump_generation_on_ruler_change(
                observed_ruler=(
                    tr.test_metrics.eval_ruler if tr.test_metrics is not None else ""
                ),
                known_ruler=holdout_ruler,
                holdout_generation=holdout_generation,
            )
            _log_iter_phase_split(
                trainer=trainer, iteration_idx=iteration_idx,
                distributed_dirs=distributed_dirs,
                worker_log_offsets=_worker_log_offsets,
                selfplay_secs=t_selfplay_secs, train_secs=t_train_secs,
                iter_so_far=time.monotonic() - iter_t0,
            )

            eval_dict = _run_eval_games(
                tc=tc, trainer=trainer, device=device, rng=rng,
                eval_sf=eval_sf,
            )

  # Flush replay + save checkpoint (model+optimizer+step + PID state + holdout).
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
                current_window=current_window,
                holdout_buf=holdout_buf,
                holdout_frozen=holdout_frozen,
                holdout_generation=holdout_generation,
                holdout_ruler=holdout_ruler,
                opp_strength_ema=opp_strength_ema,
                yaml_keys=last_reload_yaml_keys(_yaml_path),
                Checkpoint=Checkpoint,
            )

  # Best-model tracking: the holdout is the ruler; training loss is only a
  # stand-in until the holdout exists, and the two are never compared.
            best_loss, best_source = _update_best_model(
                trainer=trainer, test_metrics=tr.test_metrics, train_metrics=tr.metrics,
                test_metrics_source_iter=tr.test_metrics_source_iter,
                holdout_generation=holdout_generation,
                holdout_ruler=holdout_ruler,
                best_loss=best_loss, best_source=best_source,
                best_dir=best_dir, best_state_path=best_state_path,
                iteration_idx=iteration_idx, opp_strength_ema=opp_strength_ema,
            )

            pid_result = _run_pid_and_eval(
                tc=tc, config=config, pid=pid, sf=sf,
                sp_result=sp,
                opp_strength_ema=opp_strength_ema,
                opp_ema_alpha=_OPP_EMA_ALPHA,
                ds=ds,
            )
            opp_strength_ema = pid_result.opp_strength_ema
            # PID-relevant winrate (curriculum_winrate_raw) is what the controller
            # observes — selfplay-vs-SF only, not the games-generated blend that
            # mixes in selfplay-vs-self draws/losses.
            _log_pid_iter_summary(iteration_idx=iteration_idx, pid_result=pid_result, sp=sp)

            _finalize_iteration(
                replay_priority_stats=buf.pop_priority_mass_stats(),
                tc=tc, trainer=trainer, pid=pid,
                sp=sp, tr=tr, drift=drift,
                pid_result=pid_result,
                eval_dict=eval_dict,
                checkpoint=checkpoint,
                best_loss=best_loss,
                ckpt_dir=ckpt_dir,
                durable_dir=durable_dir,
                status_csv_path=_STATUS_CSV_PATH,
                tune_report_fn=_tune_report,
                puzzle_suite=puzzle_suite,
                era_probes=era_probes,
                ds=ds,
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
                global_iter=global_iter,
                completed_iterations=completed_iterations,
                device=device,
                rng=rng,
            )

            global_iter = int(iteration_idx)
            completed_iterations += 1
    finally:
        _cleanup_trial_resources(
            shard_prefetcher=shard_prefetcher,
            async_test_eval=async_test_eval,
            distributed_worker_procs=distributed_worker_procs,
            distributed_inference_broker_proc=_broker_box[0],
            sf=sf, eval_sf=eval_sf,
        )
