from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.inference import trial_slot_prefix
from chess_anti_engine.mcts.gumbel import SELFPLAY_GUMBEL_C_SCALE
from chess_anti_engine.model import ModelConfig, model_config_to_manifest_dict
from chess_anti_engine.moves import policy_size_for_encoding
from chess_anti_engine.replay import ArrayReplayBuffer, DiskReplayBuffer
from chess_anti_engine.train.target_builder import SfTargetParams
from chess_anti_engine.replay.shard import (
    IN_FLIGHT_DIR_NAME,
    LEGACY_SHARD_SUFFIX,
    LOCAL_SHARD_SUFFIX,
    PENDING_DIR_NAME,
    delete_shard_path,
    is_tmp_shard_name,
    load_shard_arrays,
)
from chess_anti_engine.train import Trainer
from chess_anti_engine.tune._utils import (
    resolve_local_override_root,
    slice_array_batch,
    stable_seed_u32,
)
from chess_anti_engine.tune._utils import (
    terminate_process as _stop_process,
)
from chess_anti_engine.tune.process_cleanup import (
    terminate_engines_owned_by,
    terminate_matching_processes,
)
from chess_anti_engine.tune.trainable_metrics import _games_per_iter_for_iteration
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils import sha256_file
from chess_anti_engine.utils.atomic import atomic_copy2, atomic_write_text
from chess_anti_engine.version import PACKAGE_VERSION, PROTOCOL_VERSION

log = logging.getLogger(__name__)

# Repo root used as cwd for spawned workers/brokers (so relative imports of
# the chess_anti_engine package work). Resolved once at import time — fs walk
# would otherwise repeat for every spawn.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Single home of the SF target-construction defaults (see
# trainer.resolve_sf_target_params, which derives from the same dataclass).
_SF_TARGET_DEFAULTS = SfTargetParams()

# Cache for SHA256 of static files (opening books, worker wheel) that don't
# change during a run.  Keyed by (path_str, file_size).
_static_sha_cache: dict[tuple[str, int, int], str] = {}


def _sha256_cached(p: Path) -> str:
    """SHA256 with caching, keyed on (path, size, mtime) so an edited file at
    the same path is always rehashed (mtime resolution is coarser than a
    single write, but a real edit always bumps size or mtime — this is NOT
    "files that don't change", it just avoids rehashing large static assets
    like the worker wheel/opening books on every manifest publish)."""
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    cached = _static_sha_cache.get(key)
    if cached is not None:
        return cached
    h = sha256_file(p)
    _static_sha_cache[key] = h
    return h


def _resolve_distributed_worker_auth(
    *,
    config: dict,
    server_root: Path,
) -> tuple[str, Path]:
    username = str(config.get("distributed_worker_username", "") or "").strip()
    password_file_raw = str(config.get("distributed_worker_password_file", "") or "").strip()
    password_file = Path(password_file_raw).expanduser() if password_file_raw else (server_root / f"{username}.password")
    if password_file.as_posix().startswith("/mnt/c/chess_active/"):
        password_file = server_root / password_file.name
    if username and password_file.exists():
        return username, password_file

    candidates = sorted(
        server_root.glob("tune_worker_*.password"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        password_file = candidates[0]
        username = password_file.stem
    return username, password_file


def _set_active_run_prefix(*, server_root: Path, trial_id: str) -> None:
    prefix = str(trial_id).split("_", 1)[0].strip()
    if not prefix:
        return
    atomic_write_text(server_root / "active_run_prefix.txt", prefix + "\n")


def _trial_server_dirs(*, server_root: Path, trial_id: str) -> dict[str, Path]:
    trial_root = Path(server_root) / "trials" / str(trial_id)
    return {
        "trial_root": trial_root,
        "publish_dir": trial_root / "publish",
        "inbox_dir": trial_root / "inbox",
        "processed_dir": trial_root / "processed",
        "workers_root": trial_root / "workers",
    }


# Re-export under the historical _-prefixed names used in this module's
# private callers (private to the module, not the wire protocol).
_is_tmp_shard_name = is_tmp_shard_name
_PENDING_DIR_NAME = PENDING_DIR_NAME
_IN_FLIGHT_DIR_NAME = IN_FLIGHT_DIR_NAME
# Server's post-flush archive dir (``inbox_root / "_compacted"`` in app.py).
# Unlike per-user upload dirs it is continuously appended while retaining old
# shards, so its own mtime is always fresh — the mtime-gate fast path below
# must NOT short-circuit it or its old shards never age out.
_COMPACTED_DIR_NAME = "_compacted"


def _iter_shard_paths_nested(root: Path) -> list[Path]:
    """List shard paths under a two-level inbox/processed layout
    (``root/<user>/<shard>``).

    Returns both current ``.zarr`` and legacy ``.npz`` entries. The archival
    ``.npz`` support is load-bearing for ``_prune_processed_shards``, which
    ages out old uploads from pre-zarr runs in ``processed/_compacted``.

    Skips in-progress temp directories (``tmp_*`` / ``._tmp_*``) at both
    levels: these are mid-upload .zarr dirs the server will atomically
    rename to their final names. Descending into one with ``glob("*/*.npz")``
    would scandir its internals and race with that rename, raising
    FileNotFoundError. Any dir that vanishes mid-iteration is also skipped;
    its contents will be picked up on a later call once the rename lands.

    Also skips the server's ``_pending`` and ``_in_flight`` staging dirs.
    Pending shards already contributed their samples to an in-memory
    accumulator; in-flight shards are being compacted. Both land in replay via
    ``_compacted`` once the server flushes, so ingesting them here would
    double-count them.
    """
    paths: list[Path] = []
    try:
        user_dirs = list(root.iterdir())
    except FileNotFoundError:
        return paths
    for user_dir in user_dirs:
        if not user_dir.is_dir() or _is_tmp_shard_name(user_dir.name):
            continue
        if user_dir.name in {_PENDING_DIR_NAME, _IN_FLIGHT_DIR_NAME}:
            continue
        try:
            for entry in user_dir.iterdir():
                name = entry.name
                if _is_tmp_shard_name(name):
                    continue
                if name.endswith((LOCAL_SHARD_SUFFIX, LEGACY_SHARD_SUFFIX)):
                    paths.append(entry)
        except FileNotFoundError:
            continue
    return sorted(paths)


def _quarantine_inbox_shards(
    *,
    inbox_dir: Path,
    processed_dir: Path,
    reason: str,
) -> dict[str, int | str]:
    """Move preexisting inbox shards out of the active intake path."""
    inbox_dir = Path(inbox_dir)
    processed_dir = Path(processed_dir)
    reason_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(reason).strip() or "resume")
    quarantine_root = processed_dir / "_quarantine" / f"{reason_slug}_{int(time.time())}"
    moved = 0
    for sp in _iter_shard_paths_nested(inbox_dir):
        rel = sp.relative_to(inbox_dir)
        dst = quarantine_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            sp.replace(dst)
        except Exception:
            try:
                shutil.copy2(str(sp), str(dst))
                sp.unlink(missing_ok=True)
            except Exception:
                continue
        moved += 1
    return {
        "moved_shards": int(moved),
        "quarantine_root": str(quarantine_root),
    }


def _prune_processed_shards(
    *,
    processed_dir: Path,
    max_age_seconds: float = 86400.0,
) -> int:
    """Delete processed shards older than *max_age_seconds*.

    Returns the number of files deleted. Per-user-dir mtime gate skips
    walking subtrees whose newest entry is younger than the cutoff —
    O(n_users) stat calls instead of O(n_total_shards) on long runs.
    """
    if max_age_seconds <= 0 or not processed_dir.is_dir():
        return 0
    cutoff = time.time() - float(max_age_seconds)
    deleted = 0
    try:
        user_dirs = list(processed_dir.iterdir())
    except FileNotFoundError:
        return 0
    for user_dir in user_dirs:
        if not user_dir.is_dir() or _is_tmp_shard_name(user_dir.name):
            continue
        if user_dir.name == _PENDING_DIR_NAME:
            continue
        try:
  # If the user-dir's own mtime is fresh, every shard inside is
  # also fresh (mtime is updated on add). Skip the per-shard scan.
  # The ``_compacted`` archive is exempt: it is continuously appended
  # while retaining old shards, so its mtime is always fresh and this
  # short-circuit would leak its aged-out shards forever.
            if user_dir.name != _COMPACTED_DIR_NAME and user_dir.stat().st_mtime >= cutoff:
                continue
            entries = list(user_dir.iterdir())
        except FileNotFoundError:
            continue
        for entry in entries:
            name = entry.name
            if _is_tmp_shard_name(name):
                continue
            if not (name.endswith((LOCAL_SHARD_SUFFIX, LEGACY_SHARD_SUFFIX))):
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    delete_shard_path(entry)
                    deleted += 1
            except FileNotFoundError:
                continue
  # Remove empty subdirectories.
    for d in processed_dir.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
            except OSError:
                pass  # raced with another writer or permission issue — skip
    return deleted


def build_recommended_worker(
    *,
    config: dict,
    model_cfg: ModelConfig,
    sf_nodes: int,
    mcts_simulations: int,
    wdl_regret: float = -1.0,
    pause_selfplay: bool = False,
    pause_reason: str = "",
) -> dict[str, object]:
    """Build the ``recommended_worker`` block of the worker manifest.

    This is the ONLY channel by which a yaml selfplay knob reaches a
    distributed worker: a key that is not built here is not published, the
    worker never sees it, and the corresponding selfplay dataclass silently
    keeps its own default. That failure mode is invisible to a diff over the
    keys the manifest and the yaml *share* — it was how ``soft_policy_temp``
    ran at 2.0 against a configured 3.0 for five months (rl_loop_audit E13).
    ``scripts/audit_realized_config.py --reco-diff`` now diffs over the UNION
    of the two key sets; keep it green when adding a knob here.

    Split out of ``_publish_distributed_trial_state`` so that the published
    key set is computable from a config alone, without exporting a model.
    """
    return {
        "games_per_batch": int(config.get("selfplay_batch", 4)),
        "max_plies": int(config.get("max_plies", 240)),
        "mcts": str(config.get("mcts", "puct")),
        "mcts_simulations": int(mcts_simulations),
        "playout_cap_fraction": float(config.get("playout_cap_fraction", 0.25)),
        "full_ply_pair_fraction": float(config.get("full_ply_pair_fraction", 0.0)),
        "fast_simulations": int(config.get("fast_simulations", 8)),
        "gumbel_topk": int(config.get("gumbel_topk", 16)),
        "gumbel_target_batch": int(config.get("gumbel_target_batch", 0)),
        "gumbel_vloss_weight": int(config.get("gumbel_vloss_weight", 0)),
  # volatility_q_scale / volatility_fpu / volatility_anchor are DELIBERATELY not
  # published. The worker resolves all three out of the reco and the live-yaml
  # validator accepts them, so publishing looks like it would close a
  # silent-ignore gap -- but a non-zero volatility_q_scale/volatility_fpu makes
  # volatility_search_enabled() true, which drops network_turn.py off the C path
  # onto run_gumbel_root_many, which RAISES ValueError unless the evaluator
  # exposes evaluate_encoded_with_volatility. None of the four evaluators a
  # distributed worker can hold does: MultiSlotInferenceClient, ThreadedDispatcher,
  # SlotInferenceClient, AOTEvaluator (only LocalModelEvaluator/DirectGPUEvaluator,
  # which are in-trainer). The ValueError is caught nowhere, so the worker process
  # exits -- and every worker reads the same manifest, so they die together.
  # Publishing would therefore turn an inert yaml key into a fleet crash switch.
  # config_yaml._check_volatility_search_unsupported now rejects a non-zero value
  # at load time instead, so the knob is loud rather than either silently ignored
  # or fatal. Pinned by tests/test_selfplay_gumbel_batching_plumbing.py.
        "gumbel_c_scale": float(config.get("gumbel_c_scale", SELFPLAY_GUMBEL_C_SCALE)),
        "gumbel_scale": float(config.get("gumbel_scale", 1.0)),
        "gumbel_scale_after": float(config.get("gumbel_scale_after", 0.0)),
        "gumbel_scale_decay_start_move": int(config.get("gumbel_scale_decay_start_move", 0)),
        "gumbel_scale_decay_moves": int(config.get("gumbel_scale_decay_moves", 0)),
        "curriculum_gumbel_scale": float(config.get("curriculum_gumbel_scale", 0.0)),
        "curriculum_gumbel_scale_after": float(config.get("curriculum_gumbel_scale_after", 0.0)),
        "curriculum_gumbel_scale_decay_start_move": int(config.get("curriculum_gumbel_scale_decay_start_move", 0)),
        "curriculum_gumbel_scale_decay_moves": int(config.get("curriculum_gumbel_scale_decay_moves", 0)),
        "opening_book_max_plies": int(config.get("opening_book_max_plies", 4)),
        "opening_book_max_games": int(config.get("opening_book_max_games", 200_000)),
        "opening_book_prob": float(config.get("opening_book_prob", 1.0)),
        "opening_book_path_2": config.get("opening_book_path_2"),
        "opening_book_max_plies_2": int(config.get("opening_book_max_plies_2", 16)),
        "opening_book_max_games_2": int(config.get("opening_book_max_games_2", 200_000)),
        "opening_book_mix_prob_2": float(config.get("opening_book_mix_prob_2", 0.0)),
        "random_start_plies": int(config.get("random_start_plies", 0)),
        "opening_fen_prob": float(config.get("opening_fen_prob", 0.0)),
        "opening_fen_net_side_to_move": bool(
            config.get("opening_fen_net_side_to_move", True)
        ),
        "opening_fen_selfplay_only": bool(config.get("opening_fen_selfplay_only", False)),
        "opening_fen_dole_per_iter": int(config.get("opening_fen_dole_per_iter", 0)),
        # Absolute per-iteration seeded-game budget, resolved here (the worker
        # does not know games_per_iter). 0 = uncapped.
        "opening_fen_dole_max_games": math.ceil(
            max(0.0, float(config.get("opening_fen_dole_max_fraction", 0.0)))
            * float(config.get("games_per_iter", 0) or 0)
        ),
        "opening_fen_sf_refute_frac": float(config.get("opening_fen_sf_refute_frac", 0.0)),
        "opening_fen_sf_refute_plies": int(config.get("opening_fen_sf_refute_plies", 5)),
        "sf_refute_full_node_moves": bool(config.get("sf_refute_full_node_moves", False)),
        "sf_refute_record_opp_rows": bool(config.get("sf_refute_record_opp_rows", False)),
        "sf_refute_opp_policy_net_blend": float(
            config.get("sf_refute_opp_policy_net_blend", 0.0)
        ),
        "selfplay_fraction": float(config.get("selfplay_fraction", 0.0)),
        "slot_oversubscribe": float(config.get("slot_oversubscribe", 1.0)),
        "sf_nodes": int(sf_nodes),
        "sf_move_nodes": int(config.get("sf_move_nodes", 0)),
        "sf_fast_ply_node_scale": float(config.get("sf_fast_ply_node_scale", 0.25)),
        "sf_label_nodes_cap": int(config.get("sf_label_nodes_cap", 0)),
        "sf_label_escalate_q_gap": float(config.get("sf_label_escalate_q_gap", 0.0)),
        "sf_label_escalate_nodes": int(config.get("sf_label_escalate_nodes", 3_000_000)),
        "sf_label_escalate_max_per_game": int(
            config.get("sf_label_escalate_max_per_game", 2)
        ),
        "sf_multipv": int(config.get("sf_multipv", 1)),
        "sf_hash_mb": int(config.get("sf_hash_mb", 16)),
  # Defaults for the five SF target-construction keys come from
  # SfTargetParams — the trainer-side rebuild (target_builder.py) resolves the
  # SAME yaml keys through the same dataclass, so a default that drifted here
  # would make capture-time and rebuilt targets silently disagree on any
  # config that omits the key.
        "sf_policy_temp": float(
            config.get("sf_policy_temp", _SF_TARGET_DEFAULTS.sf_policy_temp)
        ),
        "sf_policy_label_smooth": float(
            config.get(
                "sf_policy_label_smooth", _SF_TARGET_DEFAULTS.sf_policy_label_smooth
            )
        ),
  # Exponent of the policy_soft target (p^(1/T), selfplay/finalize.py). The
  # default MUST stay at GameConfig.soft_policy_temp: this key was absent
  # from the manifest until 2026-07-26, so every worker built the target at
  # the dataclass default while the yaml claimed otherwise (audit E13).
        "soft_policy_temp": float(config.get("soft_policy_temp", 2.0)),
        "policy_encoding": str(model_cfg.policy_encoding),
        "input_history_encoding": str(config.get("input_history_encoding", "legacy")),
        "input_extra_features": str(model_cfg.input_extra_features),
        "use_dynamic_relations": bool(model_cfg.use_dynamic_relations),
        "record_relations": bool(
            config.get("record_relations", model_cfg.use_dynamic_relations)
        ),
        "record_lc0_root_input": bool(config.get("record_lc0_root_input", False)),
        "history_rep_fix": bool(config.get("history_rep_fix", False)),
        "record_dense_sf_policy": bool(config.get("record_dense_sf_policy", True)),
        "record_sf_p0_policy": bool(config.get("record_sf_p0_policy", False)),
        "record_sf_p0_regret": bool(config.get("record_sf_p0_regret", False)),
        "record_fast_ply_value": bool(config.get("record_fast_ply_value", False)),
        "blindspot_harvest_out_path": str(config.get("blindspot_harvest_out_path", "")),
        "selfplay_resume_inflight_games": bool(
            config.get("selfplay_resume_inflight_games", False)
        ),
        "categorical_blend_frac": float(config.get("categorical_blend_frac", 0.0)),
        "categorical_search_blend_frac": float(
            config.get("categorical_search_blend_frac", 0.0)
        ),
        "sf_wdl_use_cp_logistic": bool(
            config.get("sf_wdl_use_cp_logistic", _SF_TARGET_DEFAULTS.sf_wdl_use_cp_logistic)
        ),
        "sf_wdl_cp_slope": float(
            config.get("sf_wdl_cp_slope", _SF_TARGET_DEFAULTS.sf_wdl_cp_slope)
        ),
        "sf_wdl_cp_draw_width": float(
            config.get("sf_wdl_cp_draw_width", _SF_TARGET_DEFAULTS.sf_wdl_cp_draw_width)
        ),
        "opponent_wdl_regret_limit": float(wdl_regret) if float(wdl_regret) >= 0.0 else None,
        "temperature": float(config.get("temperature", 1.0)),
        "temperature_decay_start_move": int(config.get("temperature_decay_start_move", 20)),
        "temperature_decay_moves": int(config.get("temperature_decay_moves", 60)),
        "temperature_endgame": float(config.get("temperature_endgame", 0.6)),
        "selfplay_temperature": config.get("selfplay_temperature"),
        "selfplay_temperature_decay_start_move": config.get("selfplay_temperature_decay_start_move"),
        "selfplay_temperature_decay_moves": config.get("selfplay_temperature_decay_moves"),
        "selfplay_temperature_endgame": config.get("selfplay_temperature_endgame"),
        "timeout_adjudication_threshold": float(config.get("timeout_adjudication_threshold", 0.90)),
  # Syzygy. `path` must be a filesystem location visible to workers
  # (same layout on all nodes in a multi-node deployment). Server
  # operators can edit these directly in publish/manifest.json to
  # change endgame adjudication behavior without restarting anyone.
        "syzygy_path": config.get("syzygy_path") or None,
        "stockfish_syzygy_path": config.get("stockfish_syzygy_path") or None,
        "syzygy_rescore_policy": bool(config.get("syzygy_rescore_policy", False)),
        "syzygy_adjudicate": bool(config.get("syzygy_adjudicate", False)),
        "syzygy_adjudicate_fraction": float(config.get("syzygy_adjudicate_fraction", 1.0)),
        "syzygy_in_search": bool(config.get("syzygy_in_search", False)),
        "pause_selfplay": bool(pause_selfplay),
        "pause_reason": str(pause_reason),
    }


def _first_iteration_games_need(config: dict) -> int:
    """Matching games iteration 1's ingest wait will hold out for.

    Reproduces the trainer's chain WHOLE -- ``TrialConfig.from_dict(config)``
    then ``_games_per_iter_for_iteration(tc, 1)`` -- which is exactly how
    ``trainable_phases.py`` derives ``total_games``. The guard and the
    criterion it must clear therefore share one instrument end to end, key
    defaulting included.

    Reading the dict directly is what an earlier revision of this function
    did, and it was a silent no-op: ``TrialConfig.from_dict`` defaults
    ``games_per_iter_start`` to ``games_per_iter`` (``trial_config.py:540``),
    while a ``config.get(..., 0)`` defaults it to 0. Under a ramp with the
    key absent -- a shape no test covered -- the trainer waited for 440 games
    and the floor computed a need of 1, so ``max(264, 1)`` left the deadlock
    fully in place. Sharing the arithmetic is not enough; the guard must
    share the defaulting too.
    """
    return _games_per_iter_for_iteration(TrialConfig.from_dict(config), 1)


def _seed_dole_gate_claims_iteration(
    *, server_root: Path, trial_id: str, training_iteration: int,
) -> bool:
    """True when ``seed_dole_gate.json`` already records this iteration claimed.

    Split out of ``_publish_distributed_trial_state`` so the read can be
    ORDERED before the manifest write while its use stays with the rearm block
    (audit L5) -- and so the ordering is testable without driving a publish.

    The server writes the gate with tmp+rename, so this unlocked cross-process
    read can never see a torn file; what it CAN see is a claim that landed
    after the caller made this iteration visible, which is what the ordering
    prevents. Suffix matching on the trial key and the fail-closed ``except``
    are carried over verbatim from the inline version.
    """
    gate_path = Path(server_root) / "seed_dole_gate.json"
    try:
        if not gate_path.exists():
            return False
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        tid = str(trial_id or "").strip()
        last = gate.get(tid)
        if last is None and tid:
            for k, v in gate.items():
                ks = str(k)
                if ks == tid or ks.endswith(tid) or tid.endswith(ks):
                    last = v
                    break
        return int(last if last is not None else -1) == int(training_iteration)
    except Exception:
        return False


def _publish_distributed_trial_state(
    *,
    trainer: Trainer,
    config: dict,
    model_cfg: ModelConfig,
    server_root: Path,
    trial_id: str,
    training_iteration: int,
    trainer_step: int,
    sf_nodes: int,
    mcts_simulations: int,
    wdl_regret: float = -1.0,
    pause_selfplay: bool = False,
    pause_reason: str = "",
    backpressure: dict[str, object] | None = None,
    export_model: bool = True,
    reuse_existing_model_for_same_step: bool = False,
    override_model_path: Path | None = None,
) -> str:
    dirs = _trial_server_dirs(server_root=server_root, trial_id=trial_id)
    publish_dir = dirs["publish_dir"]
    publish_dir.mkdir(parents=True, exist_ok=True)

    model_path = publish_dir / "latest_model.pt"
  # Promotion-gate hold: publish a previously promoted export instead of the
  # trainer's current weights. Training is untouched -- only what the selfplay
  # fleet plays with is held back. Copied through a temp + atomic rename so a
  # worker polling mid-write can never fetch a truncated model.
    if override_model_path is not None:
        src = Path(override_model_path)
        if not src.is_file():
            raise FileNotFoundError(
                f"promotion-gate hold requested but {src} does not exist; "
                "refusing to publish an unheld model under a hold decision"
            )
        tmp = model_path.with_suffix(model_path.suffix + ".gatehold.tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(model_path)
        export_model = False
    if export_model and reuse_existing_model_for_same_step and model_path.exists():
        manifest_path = publish_dir / "manifest.json"
        try:
            prev_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prev_step = int(prev_manifest.get("trainer_step") or -1)
            prev_sha = str((prev_manifest.get("model") or {}).get("sha256") or "")
            if prev_step == int(trainer_step) and prev_sha and sha256_file(model_path) == prev_sha:
                export_model = False
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    if export_model or (not model_path.exists()):
        trainer.export_swa(model_path)
    model_sha = sha256_file(model_path)
    api_prefix = f"/v1/trials/{trial_id}"
    published_worker_wheel_path: Path | None = None
    stale_pause_target_games = int(config.get("distributed_stale_pause_target_games", -1))
    if stale_pause_target_games < 0:
        stale_pause_target_games = math.ceil(
            float(config.get("games_per_iter", 0))
            * max(0.0, float(config.get("distributed_prev_model_max_fraction", 0.0)))
        )
  # COLD-START FLOOR (latent bug; did NOT fire on 2026-08-04 -- see the
  # ledger CORRECTION: production sets this key explicitly to 1870, so the
  # formula above never ran and the cap sat ABOVE the need, not below it).
  #
  # For a config that DOES rely on the default, the frac-based value is below
  # iteration 1's need by construction whenever the fraction is < 1:
  # ceil(440 * 0.6) = 264 against a 440-game wait. From cold there is exactly
  # one published sha, so the server pauses the fleet at 264 while the
  # trainer waits for 440, and only a NEW sha -- which needs the iteration to
  # finish -- can release it.
  #
  # ``trainer_step <= 0`` IS the cold-start test, and the scope matters:
  # no training step has been taken, so no second sha can have been published
  # and the trainer's accepted set has exactly one member. It covers the
  # bootstrap publish and iteration 1's republish of the same weights, then
  # self-clears the moment training advances the step. A RESUME is excluded
  # for free (its step is restored non-zero), which is correct -- prev-sha
  # games count as matching there, and that is what makes the whole class
  # cold-start-only.
  #
  # ⚑ Gating on the step is not a refinement, it is the fix being correct.
  # An unconditional floor also raises the target in STEADY state, where no
  # deadlock is possible, which silently moves when backpressure engages --
  # caught by test_distributed_selfplay_backpressure's steady-state cases
  # (trainer_step=123: 500 must stay 500, not become 1000) after an earlier
  # revision of this block claimed "steady state is unchanged" without
  # enforcing it.
  #
  # The floor also only ever RAISES, and only on the auto path: an explicit
  # target is a recorded operator decision (production's 1870 is ledger'd)
  # and is published verbatim.
        if int(trainer_step) <= 0:
            stale_pause_target_games = max(
                stale_pause_target_games, _first_iteration_games_need(config),
            )

    worker_wheel_raw = str(config.get("worker_wheel_path", "")).strip()
    if worker_wheel_raw:
        worker_wheel_src = Path(worker_wheel_raw)
        if worker_wheel_src.exists() and worker_wheel_src.is_file():
            dst = publish_dir / "worker.whl"
  # Skip copy if dst already matches src (wheel is static during a run).
            src_size = worker_wheel_src.stat().st_size
            needs_copy = (not dst.exists()) or dst.stat().st_size != src_size
            try:
                if needs_copy:
                    shutil.copy2(str(worker_wheel_src), str(dst))
                published_worker_wheel_path = dst
            except Exception:
                published_worker_wheel_path = None

    recommended_worker = build_recommended_worker(
        config=config,
        model_cfg=model_cfg,
        sf_nodes=sf_nodes,
        mcts_simulations=mcts_simulations,
        wdl_regret=wdl_regret,
        pause_selfplay=pause_selfplay,
        pause_reason=pause_reason,
    )

    manifest: dict[str, object] = {
        "server_time_unix": int(time.time()),
        "protocol_version": int(PROTOCOL_VERSION),
        "server_version": str(PACKAGE_VERSION),
        "min_worker_version": str(PACKAGE_VERSION),
        "trial_id": str(trial_id),
        "training_iteration": int(training_iteration),
        "trainer_step": int(trainer_step),
        "task": {"type": "selfplay"},
        "backpressure": {
            **(dict(backpressure) if isinstance(backpressure, dict) else {}),
            "pause_selfplay": bool(pause_selfplay),
            "pause_reason": str(pause_reason),
            "stale_pause_model_sha": str(model_sha),
            "stale_pause_target_games": max(0, int(stale_pause_target_games)),
        },
        "recommended_worker": recommended_worker,
        "encoding": {
            "input_planes": int(input_plane_count(model_cfg.input_extra_features)),
            "policy_size": int(policy_size_for_encoding(model_cfg.policy_encoding)),
            "policy_encoding": str(model_cfg.policy_encoding),
            "input_history_encoding": str(config.get("input_history_encoding", "legacy")),
            "input_extra_features": str(model_cfg.input_extra_features),
            "use_dynamic_relations": bool(model_cfg.use_dynamic_relations),
        },
        "model": {
            "sha256": str(model_sha),
            "endpoint": api_prefix + "/model",
            "filename": "latest_model.pt",
            "format": "torch_state_dict",
        },
        "model_config": model_config_to_manifest_dict(model_cfg),
    }

    for cfg_key, manifest_key, endpoint in (
        ("opening_book_path", "opening_book", "/v1/opening_book"),
        ("opening_book_path_2", "opening_book_2", "/v1/opening_book_2"),
    ):
        raw = config.get(cfg_key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        p = Path(raw.strip())
        if not p.exists():
            continue
        manifest[manifest_key] = {
            "endpoint": endpoint,
            "filename": p.name,
            "sha256": _sha256_cached(p),
        }

    # opening_fen_list_path is LIVE-reloadable (unlike the two book paths
    # above, which the server still captures once at launch): copy the
    # CURRENT source file into publish_dir under a FIXED name (mirrors
    # latest_model.pt — same name every iteration, content/sha differentiate
    # versions) so a yaml path change takes effect on the NEXT manifest
    # publish instead of requiring a restart to relaunch the server with a
    # new launch-time path. atomic_copy2 so a worker mid-GET never reads a
    # torn write. Always copy (unconditional) and hash fresh (sha256_file,
    # NOT _sha256_cached): the file is tiny (a few KB-100KB) so both are
    # free, and this is the one asset here that's actually EXPECTED to
    # change often — a cache keyed on mtime could tie across two edits
    # within one filesystem-clock tick (same size, same coarse mtime) and
    # silently keep advertising a stale hash. Not a concern for the cached
    # book paths / worker wheel above: those are large and effectively
    # static for a run's duration, which is exactly what that cache assumes.
    fen_raw = config.get("opening_fen_list_path")
    if isinstance(fen_raw, str) and fen_raw.strip():
        fen_src = Path(fen_raw.strip())
        if not fen_src.exists():
            # Live-reloadable => a yaml typo can now dangle mid-run; without
            # this the manifest entry vanishes silently, workers set their
            # list to None, and ALL FEN seeding no-ops with nothing in any log.
            log.warning(
                "opening_fen_list_path %s does not exist; omitting from manifest "
                "(FEN seeding disabled until the path resolves)",
                fen_src,
            )
        else:
            fen_dst = publish_dir / "opening_fen_list_live.txt"
            atomic_copy2(fen_src, fen_dst)
            manifest["opening_fen_list"] = {
                # Trial-scoped (api_prefix), unlike the two hardcoded
                # /v1/opening_book* endpoints above: the artifact only lives
                # under THIS trial's publish dir (see the route's docstring
                # in server/app.py), so a non-scoped URL would be ambiguous
                # for any server managing more than the single default trial.
                "endpoint": api_prefix + "/opening_fen_list",
                "filename": fen_dst.name,
                "sha256": sha256_file(fen_dst),
            }

    if published_worker_wheel_path is not None and published_worker_wheel_path.exists():
        manifest["worker_wheel"] = {
            "endpoint": api_prefix + "/worker_wheel",
            "filename": published_worker_wheel_path.name,
            "sha256": _sha256_cached(published_worker_wheel_path),
            "version": str(PACKAGE_VERSION),
        }

  # READ THE GATE BEFORE THE MANIFEST IS WRITTEN (audit L5). The rearm block
  # below asks "was this iteration already claimed", and the manifest write on
  # the next statement is the ONLY way a worker learns this iteration exists.
  # Read afterwards, that predicate silently became "claimed at any point,
  # including 400us ago BECAUSE OF THIS VERY PUBLISH": a worker polls the fresh
  # manifest, wins claim(N), the server persists the gate, and the read then
  # arms a rearm that a SECOND worker consumes -- handing the whole seed list
  # out twice for one iteration, the exact double-dole the block exists to
  # prevent. The server consumes the rearm under ``_SeedDoleGate._lock`` and
  # this producer never held it, so no care on the consumer side can close it.
  #
  # Hoisting is the fix rather than locking, because before the manifest exists
  # any claim recorded in the gate is NECESSARILY from an earlier publish -- a
  # previous process (mid-iteration resume) or an earlier attempt at this same
  # iteration (retry republish). That is exactly the set of cases that should
  # arm, so ordering buys the discriminator for free. THE ADJACENCY IS
  # LOAD-BEARING: anything inserted between this read and the write below
  # reopens the window it closes.
    seed_dole_gate_claimed = _seed_dole_gate_claims_iteration(
        server_root=Path(server_root),
        trial_id=str(trial_id),
        training_iteration=int(training_iteration),
    )

    atomic_write_text(
        publish_dir / "manifest.json",
        json.dumps(manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    # One-shot dole rearm ONLY for true mid-iter resume: durable gate already
    # shows this training_iteration claimed (workers died; claim would otherwise
    # stay burned). Do NOT arm on every normal selfplay open — that races with
    # multi-worker first-claim and double-doles (PR #209 review). Consumed under
    # the gate lock in server claim().
    #
    # The gate read this decision rests on is taken ABOVE, BEFORE the manifest
    # write -- see the comment there for why the ordering is the whole fix
    # (audit L5). Everything else about the predicate is unchanged.
    rearm_path = publish_dir / "seed_dole_rearm.json"
    dole_n = int(config.get("opening_fen_dole_per_iter", 0) or 0)
    arm_rearm = False
    if (not pause_selfplay) and dole_n > 0 and manifest.get("opening_fen_list"):
        arm_rearm = seed_dole_gate_claimed
    if arm_rearm:
        atomic_write_text(
            rearm_path,
            json.dumps({"training_iteration": int(training_iteration)}, sort_keys=True),
            encoding="utf-8",
        )
    else:
        with contextlib.suppress(OSError):
            rearm_path.unlink(missing_ok=True)
    return model_sha


_WORKER_LAUNCH_CONFIG_KEYS: tuple[str, ...] = (
    "distributed_server_url",
    "distributed_worker_username",
    "distributed_worker_password_file",
    "distributed_server_root",
    "distributed_worker_shared_cache_dir",
    "stockfish_path",
    "sf_nice",
    "distributed_worker_device",
    "device",
    "distributed_worker_aot_dir",
    "distributed_worker_use_compile",
    "distributed_worker_compile_mode",
    "distributed_worker_inference_fp8",
    "distributed_worker_threaded",
    "distributed_worker_selfplay_threads",
    "distributed_worker_dispatcher_batch_wait_ms",
    "distributed_worker_dispatcher_max_batch",
    "distributed_worker_dispatcher_target_batch",
    "distributed_worker_sf_workers",
    "distributed_worker_poll_seconds",
    "distributed_worker_upload_target_positions",
    "distributed_worker_upload_flush_seconds",
    "seed",
    "log_level",
    "distributed_inference_broker_enabled",
    "distributed_inference_max_batch_per_slot",
    "distributed_inference_max_batch_positions",
    "distributed_inference_slots_per_worker",
    "distributed_worker_auto_tune",
    "distributed_worker_target_batch_seconds",
    "distributed_worker_min_games_per_batch",
    "distributed_worker_max_games_per_batch",
)


def _worker_launch_signature(
    *, config: dict, trial_id: str, worker_index: int,
) -> tuple[object, ...]:
    """Stable signature of worker process-level settings.

    Selfplay parameters delivered through the manifest do not belong here; only
    values baked into the worker's command line or auth/cache resolution should
    restart an already-running worker.
    """
    return (
        str(trial_id),
        int(worker_index),
        tuple((k, config.get(k)) for k in _WORKER_LAUNCH_CONFIG_KEYS),
    )


  # How many previous worker-log generations to keep beside the live file.
  # 2 because one revive is not the interesting case: the 2026-08-04 stall
  # began ~00:23 and was revived at 01:47, so a SECOND revive before anyone
  # read the logs would have taken the original evidence with it.
_WORKER_LOG_GENERATIONS = 2


def _rotate_worker_logs(*paths: Path) -> None:
    """Move existing worker logs aside so a (re)launch cannot overwrite them.

    ⚑ Banked from the 2026-08-04 cold start, where this cost us the diagnosis.
    The 01:47:22 revive left ``worker_00/worker.log`` holding a single
    ``worker starting version=`` line at 01:47:22 and nothing from the
    00:19-01:47 incident window -- the 00:34:22 pause line and ~4,264
    upload-buffer drop lines from 00:37:20 were read live and can never be read
    again. The primary failure (the upload path wedging at ~00:23) is still
    UNDIAGNOSED for exactly that reason.

    The truncating writer was NOT identified, and this fix deliberately does
    not depend on identifying it: ``logging.FileHandler`` defaults to append
    (``worker.py``), ``_spawn_with_reap`` opens the .out with ``"ab"``,
    ``os.execv`` on self-update reuses the same argv, and ``scripts/train.sh``
    documents the append behaviour as the reason its drain must truncate its
    READING at ``worker starting version=`` -- yet the file was truncated in
    place (the artifact directory's mtime never moved, so nothing was
    recreated). Renaming the previous generation to a different filename
    BEFORE the replacement process can open anything makes whatever truncates
    ``worker.log`` afterwards truncate a fresh, empty file instead.

    Never raises: the revive exists to bring a dead worker back, and a guard
    that can take down the thing it guards is worse than the gap it closes.
    """
    for path in paths:
        try:
  # An empty file carries no evidence, and rotating it would push a real
  # generation out of the window for nothing.
            if not path.exists() or path.stat().st_size <= 0:
                continue
            for gen in range(_WORKER_LOG_GENERATIONS - 1, 0, -1):
                older = path.with_name(f"{path.name}.{gen}")
                if older.exists():
                    older.replace(path.with_name(f"{path.name}.{gen + 1}"))
            path.replace(path.with_name(f"{path.name}.1"))
        except OSError:
            log.warning(
                "could not rotate worker log %s; relaunching anyway (the "
                "previous generation may be overwritten)", path, exc_info=True,
            )


def _launch_distributed_worker(
    *,
    config: dict,
    trial_dir: Path,
    trial_id: str,
    worker_index: int,
) -> subprocess.Popen[bytes]:
    worker_artifact_root = trial_dir / "distributed_workers" / f"worker_{worker_index:02d}"
    worker_artifact_root.mkdir(parents=True, exist_ok=True)
    worker_log = worker_artifact_root / "worker.log"
    worker_out = worker_artifact_root / "worker.out"
  # BEFORE the spawn, not after: rotating afterwards would race the
  # replacement's first writes into the rotated file and leave the live log
  # holding a fragment -- the same unreadable interleaving, one step later.
    _rotate_worker_logs(worker_log, worker_out)

    server_root_raw = str(config.get("distributed_server_root") or "").strip()
    if server_root_raw:
        server_root = resolve_local_override_root(
            raw_root=server_root_raw,
            tune_work_dir=config.get("work_dir", trial_dir),
            suffix="server",
        )
        server_dirs = _trial_server_dirs(server_root=server_root, trial_id=trial_id)
        worker_root = server_dirs["workers_root"] / f"worker_{worker_index:02d}"
    else:
  # Fallback for non-standard local setups: keep previous behavior.
        worker_root = worker_artifact_root
    worker_root.mkdir(parents=True, exist_ok=True)

    cmd = _build_distributed_worker_cmd(
        config=config,
        trial_root=worker_root,
        trial_id=trial_id,
        worker_index=worker_index,
        worker_log=worker_log,
    )
    proc = _spawn_with_reap(
        cmd=cmd,
        log_path=worker_out,
        reap_module="chess_anti_engine.worker",
        reap_terms=["--trial-id", str(trial_id), "--work-dir", str(worker_root)],
        reap_label=f"distributed workers (trial={trial_id} idx={worker_index:02d})",
    )
    setattr(
        proc,
        "_cae_worker_launch_signature",
        _worker_launch_signature(
            config=config, trial_id=trial_id, worker_index=worker_index,
        ),
    )
    return proc


def _build_distributed_worker_cmd(
    *,
    config: dict,
    trial_root: Path,
    trial_id: str,
    worker_index: int,
    worker_log: Path,
) -> list[str]:
    device = str(config.get("distributed_worker_device") or config.get("device", "cpu"))
    server_root = resolve_local_override_root(
        raw_root=config.get("distributed_server_root", ""),
        tune_work_dir=config.get("work_dir", trial_root),
        suffix="server",
    )
    worker_username, worker_password_file = _resolve_distributed_worker_auth(
        config=config,
        server_root=server_root,
    )
    shared_cache_raw = str(config.get("distributed_worker_shared_cache_dir") or "").strip()
    if shared_cache_raw:
        shared_cache_root = Path(shared_cache_raw).expanduser()
    else:
        shared_cache_root = server_root / "worker_cache"
    shared_cache_root.mkdir(parents=True, exist_ok=True)

    aot_dir = str(config.get("distributed_worker_aot_dir", "")).strip()
    threaded = bool(config.get("distributed_worker_threaded", False))
    if aot_dir and threaded:
  # The ThreadedDispatcher branch of worker._build_evaluator compiles
  # on its own thread and ignores --aot-dir entirely; passing it
  # INSTEAD of --compile-inference silently ran the dispatcher eager
  # (multi-x throughput regression). Fail towards compile.
        log.warning(
            "distributed_worker_aot_dir=%r is incompatible with "
            "distributed_worker_threaded=true (the threaded dispatcher "
            "compiles on its own thread); ignoring aot_dir and emitting "
            "--compile-inference as configured",
            aot_dir,
        )
        aot_dir = ""

    cmd = [
        sys.executable,
        "-m",
        "chess_anti_engine.worker",
        "--server-url",
        str(config["distributed_server_url"]),
        "--trial-id",
        str(trial_id),
        "--username",
        str(worker_username),
        "--password-file",
        str(worker_password_file),
        "--stockfish-path",
        str(config["stockfish_path"]),
        "--work-dir",
        str(trial_root),
        "--shared-cache-dir",
        str(shared_cache_root),
        "--device",
        device,
        *(
            ["--aot-dir", aot_dir]
            if aot_dir
            else (
                ["--compile-inference", "--compile-mode", str(config.get("distributed_worker_compile_mode", "reduce-overhead"))]
                if bool(config.get("distributed_worker_use_compile", False))
                else []
            )
        ),
        *(["--inference-fp8"] if bool(config.get("distributed_worker_inference_fp8", False)) else []),
        *(
            [
                "--threaded-selfplay",
                "--selfplay-threads",
                str(int(config.get("distributed_worker_selfplay_threads", 16))),
                "--dispatcher-batch-wait-ms",
                str(float(config.get("distributed_worker_dispatcher_batch_wait_ms", 0.0))),
                "--dispatcher-max-batch",
                str(int(config.get("distributed_worker_dispatcher_max_batch", 4096))),
                "--dispatcher-target-batch",
                str(int(config.get("distributed_worker_dispatcher_target_batch", 0))),
            ]
            if bool(config.get("distributed_worker_threaded", False))
            else []
        ),
        "--sf-workers",
        str(int(config.get("distributed_worker_sf_workers", 1))),
        "--sf-nice",
        str(int(config.get("sf_nice", 0))),
        "--poll-seconds",
        str(float(config.get("distributed_worker_poll_seconds", 1.0))),
        "--upload-target-positions",
        str(int(config.get("distributed_worker_upload_target_positions", 500))),
        "--upload-flush-seconds",
        str(float(config.get("distributed_worker_upload_flush_seconds", 60.0))),
        "--seed",
        str(stable_seed_u32("dist-worker", trial_id, worker_index, config.get("seed", 0))),
        "--log-file",
        str(worker_log),
        "--log-level",
        str(config.get("log_level", "info")).lower(),
    ]

    if config.get("distributed_inference_broker_enabled", False):
        slot_prefix = _trial_slot_prefix(trial_id=trial_id)
        slots_per_worker = _resolve_slots_per_worker(config)
        first_slot = int(worker_index) * slots_per_worker
        slot_name = ",".join(
            f"{slot_prefix}-{slot_idx}"
            for slot_idx in range(first_slot, first_slot + slots_per_worker)
        )
        max_batch = int(
            config.get(
                "distributed_inference_max_batch_per_slot",
                config.get("distributed_inference_max_batch_positions", 256),
            )
        )
        cmd.extend(
            [
                "--inference-slot-name",
                str(slot_name),
                "--inference-slot-max-batch",
                str(max_batch),
                "--inference-slot-input-planes",
                str(input_plane_count(config.get("input_extra_features"))),
            ]
        )

    if config.get("distributed_worker_auto_tune", False):
        cmd.extend(
            [
                "--auto-tune",
                "--target-batch-seconds",
                str(float(config.get("distributed_worker_target_batch_seconds", 30.0))),
                "--min-games-per-batch",
                str(int(config.get("distributed_worker_min_games_per_batch", 1))),
                "--max-games-per-batch",
                str(int(config.get("distributed_worker_max_games_per_batch", 64))),
            ]
        )
    return cmd


def _trial_slot_prefix(*, trial_id: str) -> str:
    """Deterministic shared-memory slot prefix for a trial's inference broker.

    Carries the slot-protocol version, so a wire-format change also changes the
    NAME. Without that, the name is stable across restarts for a given trial_id
    and an old worker — one that survived a restart, or was launched from a
    checkout that had not pulled (see the `stale_worker_ingest_wedge_on_resume`
    history) — reconnects by name to a NEW broker's segment on its next
    disconnect/connect cycle. The version guards in the header cannot save it:
    the old client runs OLD code and has no validation to fail, and because the
    v2 header grew at the FRONT while state@0 / mode@1 / batch_size@4 kept their
    offsets, the state machine still completes with the two sides 16 bytes out
    of phase — measured to return an all-zero policy and an all-zero WDL with no
    exception, which is exactly the poisoning this protocol version exists to
    delete. Versioning the name turns that into a FileNotFoundError -> TimeoutError.

    Delegates to ``inference.trial_slot_prefix``: the broker creates the
    segments this names, so the two must never be able to drift.
    """
    return trial_slot_prefix(trial_id=trial_id)


def _resolve_shared_cache_root(config: dict, server_root: Path) -> Path:
    """Resolve and create the broker's torch.compile/triton cache dir.

    Caller-supplied (``distributed_worker_shared_cache_dir``) wins; otherwise
    falls back to ``<server_root>/worker_cache``. Cache is per-machine, so a
    single location shared across trials is correct.
    """
    raw = str(config.get("distributed_worker_shared_cache_dir") or "").strip()
    root = Path(raw).expanduser() if raw else (server_root / "worker_cache")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_compile_inference(config: dict) -> bool:
    """Compile-inference toggle. ``CAE_INFERENCE_COMPILE`` env var overrides
    config so ``--resume`` picks up changes without re-baking the tuner config.
    """
    env = os.environ.get("CAE_INFERENCE_COMPILE")
    if env is not None:
        return env == "1"
    return bool(config.get("distributed_inference_use_compile", False))


def _resolve_inference_compile_mode(config: dict) -> str:
    return str(
        config.get(
            "distributed_inference_compile_mode",
            config.get("distributed_worker_compile_mode", "reduce-overhead"),
        )
    )


def _resolve_max_batch_per_slot(config: dict) -> int:
    return int(
        config.get(
            "distributed_inference_max_batch_per_slot",
            config.get("distributed_inference_max_batch_positions", 256),
        )
    )


def _resolve_slots_per_worker(config: dict) -> int:
    return max(1, int(config.get("distributed_inference_slots_per_worker", 1)))


def _spawn_with_reap(
    *,
    cmd: list[str],
    log_path: Path,
    reap_module: str,
    reap_terms: list[str],
    reap_label: str,
) -> subprocess.Popen[bytes]:
    """Reap stale instances matching ``reap_module``+``reap_terms``, then
    spawn ``cmd`` with stdout/stderr appended to ``log_path``.
    """
    out_fh = log_path.open("ab")
    try:
        stale_pids = terminate_matching_processes(
            module=reap_module, required_terms=reap_terms,
        )
        if stale_pids:
            print(f"[trial] reaped stale {reap_label}: pids={stale_pids}")
        return subprocess.Popen(
            cmd,
            cwd=str(_REPO_ROOT),
            stdout=out_fh,
            stderr=subprocess.STDOUT,
        )
    finally:
        out_fh.close()


def check_dynamic_relations_transport(config: dict) -> None:
    """Fail loud when dynamic relations are enabled on a transport that
    can't carry them. The model tolerates absent relations (zero bias), but
    a silently bias-less selfplay stream would invalidate the experiment.

    Supported: worker-local DirectGPUEvaluator (with the thread-safe /
    coalescing / multi-GPU dispatchers and the eval cache) and the
    in-process gumbel-C path. Not yet wired: the shared-memory slot broker
    and the threaded worker dispatcher.
    """
    if not (
        bool(config.get("use_dynamic_relations", False))
        or bool(config.get("record_relations", False))
    ):
        return
    offenders = [
        key for key in (
            "distributed_inference_broker_enabled",
            "distributed_inference_shared_broker",
            "distributed_worker_threaded",
        )
        if bool(config.get(key, False))
    ]
    if str(config.get("distributed_worker_aot_dir") or "").strip():
        offenders.append("distributed_worker_aot_dir")
    if str(config.get("mcts", "puct")).lower() != "gumbel":
        offenders.append("mcts != gumbel (PUCT paths don't transport relations)")
    if offenders:
        raise ValueError(
            "dynamic relations require worker-local direct inference on the "
            f"gumbel path; offending config: {offenders} "
            "(relations are not transported on those paths yet)"
        )


def _launch_inference_broker(
    *,
    config: dict,
    trial_id: str,
    publish_dir: Path,
    trial_dir: Path,
) -> subprocess.Popen[bytes]:
    broker_artifact_root = trial_dir / "distributed_inference"
    broker_artifact_root.mkdir(parents=True, exist_ok=True)
    slot_prefix = _trial_slot_prefix(trial_id=trial_id)
    server_root = Path(str(config["distributed_server_root"]))
    aot_dir = str(config.get("distributed_inference_aot_dir", "") or "").strip()
    cmd = [
        sys.executable, "-m", "chess_anti_engine.inference",
        "--publish-dir", str(publish_dir),
        "--slot-prefix", str(slot_prefix),
        "--num-slots", str(
            int(config.get("distributed_workers_per_trial", 2))
            * _resolve_slots_per_worker(config)
        ),
        "--max-batch-per-slot", str(_resolve_max_batch_per_slot(config)),
        "--input-planes", str(input_plane_count(config.get("input_extra_features"))),
        "--device", str(config.get("distributed_worker_device") or config.get("device", "cpu")),
        "--batch-wait-ms", str(float(config.get("distributed_inference_batch_wait_ms", 5.0))),
        "--adaptive-idle-ms", str(float(config.get("distributed_inference_adaptive_idle_ms", 0.0))),
        "--compile-mode", _resolve_inference_compile_mode(config),
        "--shared-cache-dir", str(_resolve_shared_cache_root(config, server_root)),
        *(["--compile-inference"] if _resolve_compile_inference(config) else []),
        *(["--aot-dir", aot_dir] if aot_dir else []),
    ]
    return _spawn_with_reap(
        cmd=cmd,
        log_path=broker_artifact_root / "broker.out",
        reap_module="chess_anti_engine.inference",
        reap_terms=["--publish-dir", str(publish_dir), "--slot-prefix", str(slot_prefix)],
        reap_label=f"inference brokers (trial={trial_id})",
    )


def _ensure_inference_broker(
    *,
    config: dict,
    trial_id: str,
    trial_dir: Path,
    publish_dir: Path,
    proc: subprocess.Popen[bytes] | None,
) -> subprocess.Popen[bytes] | None:
    if not config.get("distributed_inference_broker_enabled", False):
        _stop_process(proc)
        return None
  # Per-trial broker is mutually exclusive with the shared broker.
    if config.get("distributed_inference_shared_broker", False):
        _stop_process(proc)
        return None
    if proc is not None and proc.poll() is None:
        return proc
    return _launch_inference_broker(
        config=config,
        trial_id=trial_id,
        publish_dir=publish_dir,
        trial_dir=trial_dir,
    )


def launch_shared_inference_broker(
    *,
    config: dict,
    server_root: Path,
) -> subprocess.Popen[bytes] | None:
    """Launch a single shared inference broker for all trials."""
    if not bool(config.get("distributed_inference_broker_enabled", False)):
        return None
    if not bool(config.get("distributed_inference_shared_broker", False)):
        return None
    cmd = [
        sys.executable, "-m", "chess_anti_engine.inference",
        "shared",
        "--server-root", str(server_root),
        "--slots-per-trial", str(
            int(config.get("distributed_workers_per_trial", 2))
            * _resolve_slots_per_worker(config)
        ),
        "--max-batch-per-slot", str(_resolve_max_batch_per_slot(config)),
        "--input-planes", str(input_plane_count(config.get("input_extra_features"))),
        "--device", str(config.get("distributed_worker_device") or config.get("device", "cpu")),
        "--batch-wait-ms", str(float(config.get("distributed_inference_batch_wait_ms", 0.0))),
        "--compile-mode", _resolve_inference_compile_mode(config),
        "--shared-cache-dir", str(_resolve_shared_cache_root(config, server_root)),
        *(["--compile-inference"] if _resolve_compile_inference(config) else []),
    ]
    proc = _spawn_with_reap(
        cmd=cmd,
        log_path=server_root / "shared_broker.out",
        reap_module="chess_anti_engine.inference",
        reap_terms=["shared", "--server-root", str(server_root)],
        reap_label="shared inference broker",
    )
    print(f"[tune] launched shared inference broker: pid={proc.pid}")
    return proc


def _stop_worker_processes(procs: list[subprocess.Popen[bytes]]) -> None:
    for proc in procs:
        worker_pid = int(proc.pid)
        _stop_process(proc)
        # ⚑ AUDIT R2. `_stop_process` escalates to SIGKILL on the WORKER only,
        # which leaves its Stockfish engines running: they are in their own
        # process group now, their cmdline is unmatchable by `reap_terms`, and
        # SIGKILL runs no `finally`. PDEATHSIG covers the common case, but it is
        # thread-scoped and Linux-only, so this is the belt to its braces --
        # anything still stamped with the dead worker's pid is by definition an
        # orphan. Cheap: it only scans /proc when a worker is being stopped.
        #
        # The marker is a pid, and `_stop_process` has already reaped this one,
        # so in principle the kernel could recycle it before this scan. For a
        # mis-reap the recycled pid would have to belong to a NEW worker that
        # had already stamped and spawned engines inside that window --
        # effectively unreachable, and noted rather than fixed because the
        # alternative keys (ancestry, cmdline) are the ones R2 ruled out.
        orphans = terminate_engines_owned_by(worker_pid)
        if orphans:
            log.warning(
                "reaped %d orphaned Stockfish engine(s) %s left by worker pid %d "
                "(PDEATHSIG did not fire)", len(orphans), orphans, worker_pid,
            )


def _ensure_distributed_workers(
    *,
    config: dict,
    trial_dir: Path,
    trial_id: str,
    procs: list[subprocess.Popen[bytes]],
) -> list[subprocess.Popen[bytes]]:
    want = max(0, int(config.get("distributed_workers_per_trial", 0)))
    out = list(procs)
    for idx in range(want):
        desired_signature = _worker_launch_signature(
            config=config, trial_id=trial_id, worker_index=idx,
        )
        if idx < len(out) and out[idx].poll() is None:
            current_signature = getattr(out[idx], "_cae_worker_launch_signature", None)
            if current_signature == desired_signature:
                continue
            print(
                f"[trial] restarting distributed worker idx={idx} "
                f"config_changed=true trial={trial_id}"
            )
            _stop_process(out[idx])
            out[idx] = _launch_distributed_worker(
                config=config,
                trial_dir=trial_dir,
                trial_id=trial_id,
                worker_index=idx,
            )
            continue
        if idx < len(out) and out[idx].poll() is not None:
            print(
                f"[trial] restarting distributed worker idx={idx} "
                f"exit_code={out[idx].returncode} trial={trial_id}"
            )
            out[idx] = _launch_distributed_worker(
                config=config,
                trial_dir=trial_dir,
                trial_id=trial_id,
                worker_index=idx,
            )
        elif idx >= len(out):
            out.append(
                _launch_distributed_worker(
                    config=config,
                    trial_dir=trial_dir,
                    trial_id=trial_id,
                    worker_index=idx,
                )
            )
  # Kill excess workers if count decreased
    for p in out[want:]:
        if p.poll() is None:
            print(f"[trial] stopping excess worker pid={p.pid} trial={trial_id}")
            p.terminate()
    return out[:want]


def revive_dead_selfplay_processes(
    *,
    config: dict,
    trial_id: str,
    trial_dir: Path,
    publish_dir: Path,
    broker_proc_box: list[subprocess.Popen[bytes] | None],
    worker_procs: list[subprocess.Popen[bytes]],
) -> bool:
    """Relaunch the broker and any workers that have EXITED. Returns True if any were.

    Called from inside the ingest wait loop, so it must be safe to run
    hundreds of times per iteration and must never disturb a healthy fleet.
    Both halves therefore key strictly off ``poll() is not None`` — a process
    that is alive but wedged is left alone, because relaunching next to it
    would leave two brokers bound to the same slots.

    Deliberately NOT ``_ensure_distributed_workers``: that function also
    restarts workers whose launch signature no longer matches the config, and
    the config is re-read from the live yaml every iteration. Calling it mid
    wait-loop would let an unrelated yaml edit tear down workers with games in
    flight — the exact waste PR #224 removed. Reviving a corpse is
    unambiguous; re-signing a live worker is a policy decision that belongs at
    the phase boundary where it already happens.
    """
    revived = False

    broker = broker_proc_box[0]
    if broker is not None and broker.poll() is not None:
      # print AND log. This is a Ray trial actor: logging traffic often only
      # reaches the Ray session logs, while trial stdout is what train.sh
      # captures into /tmp/chess_training.log -- which is exactly where the
      # pre-registered ledger yardstick greps for this string. Logging alone
      # could let a real revive fire and still leave the yardstick reading
      # zero forever. The rest of the fleet path already prints "[trial] ...".
        print(
            f"[trial] inference broker exited (code={broker.returncode}) "
            "mid-iteration; relaunching - workers have had no inference "
            "since it died",
            flush=True,
        )
        log.warning(
            "inference broker exited (code=%s) mid-iteration; relaunching — "
            "workers have had no inference since it died",
            broker.returncode,
        )
        broker_proc_box[0] = _ensure_inference_broker(
            config=config,
            trial_id=trial_id,
            trial_dir=trial_dir,
            publish_dir=publish_dir,
            proc=None,
        )
        revived = True

    for idx, proc in enumerate(worker_procs):
        if proc.poll() is None:
            continue
        print(
            f"[trial] distributed worker idx={idx} exited "
            f"(code={proc.returncode}) mid-iteration; relaunching",
            flush=True,
        )
        log.warning(
            "distributed worker idx=%d exited (code=%s) mid-iteration; relaunching",
            idx, proc.returncode,
        )
        worker_procs[idx] = _launch_distributed_worker(
            config=config,
            trial_dir=trial_dir,
            trial_id=trial_id,
            worker_index=idx,
        )
        revived = True

    return revived


def _empty_ingest_summary() -> dict[str, Any]:
    return {
        "matching_games": 0,
        "matching_positions": 0,
        "matching_w": 0,
        "matching_d": 0,
        "matching_l": 0,
        "matching_total_game_plies": 0,
        "matching_adjudicated_games": 0,
        "matching_tb_adjudicated_games": 0,
        "matching_total_draw_games": 0,
        "matching_selfplay_games": 0,
        "matching_selfplay_adjudicated_games": 0,
        "matching_selfplay_draw_games": 0,
        "matching_curriculum_games": 0,
        "matching_curriculum_adjudicated_games": 0,
        "matching_curriculum_draw_games": 0,
        "matching_plies_win": 0,
        "matching_plies_draw": 0,
        "matching_plies_loss": 0,
        "matching_checkmate_games": 0,
        "matching_stalemate_games": 0,
        "matching_sf_d6_sum": 0.0,
        "matching_sf_d6_n": 0,
        "positions_replay_added": 0,
        "stale_games": 0,
        "stale_positions": 0,
        "matching_shards": 0,
        "stale_shards": 0,
  # Anchored promotion-gate split of the SAME accepted games. ``matching_w/d/l``
  # pools the current and previous published models; the gate needs them apart,
  # because both faced the same handicapped Stockfish and their difference is a
  # free A/B of one training iteration (chess_anti_engine/tune/promotion_gate).
  # These are vs-SF (curriculum) outcomes only -- selfplay games are
  # model-vs-itself and already excluded from w/d/l upstream.
        "gate_cur_w": 0,
        "gate_cur_d": 0,
        "gate_cur_l": 0,
        "gate_prev_w": 0,
        "gate_prev_d": 0,
        "gate_prev_l": 0,
  # Games-weighted opponent difficulty per arm, and the games that carried a
  # difficulty at all. The ratio is the arm's mean ``wdl_regret``; a zero
  # denominator means "the shards did not say", which is reported as NaN and
  # never as 0.0.
        "gate_cur_regret_weighted": 0.0,
        "gate_cur_regret_games": 0,
        "gate_prev_regret_weighted": 0.0,
        "gate_prev_regret_games": 0,
  # Per-sample source counters (is_selfplay tag): sum of tagged samples
  # and the selfplay-true subset, across ingested shards.  Used to
  # compute ingest_frac_selfplay = selfplay / tagged.
        "ingest_is_selfplay_tagged": 0,
        "ingest_is_selfplay_true": 0,
        "matching_diff_focus_records": 0,
        "matching_diff_focus_kept": 0,
        "matching_diff_focus_keep_prob_sum": 0.0,
        "matching_diff_focus_keep_limited": 0,
        "matching_diff_focus_sample_weight_sum": 0.0,
        "matching_diff_focus_sample_weight_limited": 0,
        "matching_diff_focus_priority_sum": 0.0,
        "matching_diff_focus_priority_sq_sum": 0.0,
        "matching_diff_focus_priority_min": 0.0,
        "matching_diff_focus_priority_max": 0.0,
        "matching_gumbel_policy_diag_n": 0,
        "matching_gumbel_policy_top_prob_sum": 0.0,
        "matching_gumbel_policy_action_prob_sum": 0.0,
        "matching_gumbel_policy_entropy_sum": 0.0,
        "matching_gumbel_policy_eff_moves_sum": 0.0,
        "matching_gumbel_policy_candidate_mass_sum": 0.0,
        "matching_gumbel_policy_non_candidate_top_prob_sum": 0.0,
        "matching_gumbel_policy_argmax_is_candidate_sum": 0,
        "matching_gumbel_policy_argmax_is_action_sum": 0,
        "matching_gumbel_policy_legal_count_sum": 0,
        "matching_gumbel_policy_candidate_count_sum": 0,
        "matching_outcome_stats": {},
        "replay_priority_n": 0,
        "replay_priority_sum": 0.0,
        "replay_priority_sq_sum": 0.0,
        "replay_priority_min": 0.0,
        "replay_priority_max": 0.0,
        "replay_has_policy_n": 0,
        "replay_has_policy_sum": 0,
        "replay_has_sf_wdl_n": 0,
        "replay_has_sf_wdl_sum": 0,
        "replay_has_search_wdl_n": 0,
        "replay_has_search_wdl_sum": 0,
        "replay_wdl_0": 0,
        "replay_wdl_1": 0,
        "replay_wdl_2": 0,
    }


  # (meta_key, summary_suffix). int unless suffix is "sf_d6_sum" (float).
_SHARD_META_FIELDS: tuple[tuple[str, str], ...] = (
    ("total_game_plies", "total_game_plies"),
    ("tb_adjudicated_games", "tb_adjudicated_games"),
    ("selfplay_games", "selfplay_games"),
    ("selfplay_adjudicated_games", "selfplay_adjudicated_games"),
    ("selfplay_draw_games", "selfplay_draw_games"),
    ("curriculum_games", "curriculum_games"),
    ("curriculum_adjudicated_games", "curriculum_adjudicated_games"),
    ("curriculum_draw_games", "curriculum_draw_games"),
    ("plies_win", "plies_win"),
    ("plies_draw", "plies_draw"),
    ("plies_loss", "plies_loss"),
    ("checkmate_games", "checkmate_games"),
    ("stalemate_games", "stalemate_games"),
    ("sf_d6_n", "sf_d6_n"),
    ("diff_focus_records", "diff_focus_records"),
    ("diff_focus_kept", "diff_focus_kept"),
    ("diff_focus_keep_limited", "diff_focus_keep_limited"),
    ("diff_focus_sample_weight_limited", "diff_focus_sample_weight_limited"),
    ("gumbel_policy_diag_n", "gumbel_policy_diag_n"),
    ("gumbel_policy_argmax_is_candidate_sum", "gumbel_policy_argmax_is_candidate_sum"),
    ("gumbel_policy_argmax_is_action_sum", "gumbel_policy_argmax_is_action_sum"),
    ("gumbel_policy_legal_count_sum", "gumbel_policy_legal_count_sum"),
    ("gumbel_policy_candidate_count_sum", "gumbel_policy_candidate_count_sum"),
)

_SHARD_META_FLOAT_FIELDS: tuple[tuple[str, str], ...] = (
    ("diff_focus_keep_prob_sum", "diff_focus_keep_prob_sum"),
    ("diff_focus_sample_weight_sum", "diff_focus_sample_weight_sum"),
    ("diff_focus_priority_sum", "diff_focus_priority_sum"),
    ("diff_focus_priority_sq_sum", "diff_focus_priority_sq_sum"),
    ("diff_focus_priority_min", "diff_focus_priority_min"),
    ("diff_focus_priority_max", "diff_focus_priority_max"),
    ("gumbel_policy_top_prob_sum", "gumbel_policy_top_prob_sum"),
    ("gumbel_policy_action_prob_sum", "gumbel_policy_action_prob_sum"),
    ("gumbel_policy_entropy_sum", "gumbel_policy_entropy_sum"),
    ("gumbel_policy_eff_moves_sum", "gumbel_policy_eff_moves_sum"),
    ("gumbel_policy_candidate_mass_sum", "gumbel_policy_candidate_mass_sum"),
    ("gumbel_policy_non_candidate_top_prob_sum", "gumbel_policy_non_candidate_top_prob_sum"),
)


def _sanitize_outcome_stats(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, val in raw.items():
        key_s = str(key)
        if not re.fullmatch(r"[a-z0-9_]{1,96}", key_s):
            continue
        out[key_s] = int(out.get(key_s, 0)) + int(val or 0)
    return out


def _merge_outcome_stats(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, val in src.items():
        dst[key] = int(dst.get(key, 0)) + int(val)


def _extract_shard_metrics(meta: dict, shard_n: int) -> dict[str, Any]:
    """Pull all per-shard counts/sums from the meta dict in one place.

    Output keys map directly onto summary["matching_*"] suffixes (no prefix);
    callers add the matching/stale prefix at update time.
    """
    wins = int(meta.get("wins", 0) or 0)
    draws = int(meta.get("draws", 0) or 0)
    losses = int(meta.get("losses", 0) or 0)
    out: dict[str, Any] = {
        "w": wins,
        "d": draws,
        "l": losses,
        "games": int(meta.get("games", wins + draws + losses) or 0),
        "positions": int(meta.get("positions", shard_n) or shard_n),
  # adjudicated_games legacy-aliases timeout_games for old shards
        "adjudicated_games": int(meta.get("adjudicated_games", meta.get("timeout_games", 0)) or 0),
        "total_draw_games": int(meta.get("total_draw_games", draws) or draws),
        "sf_d6_sum": float(meta.get("sf_d6_sum", 0.0) or 0.0),
    }
    for src_key, out_key in _SHARD_META_FIELDS:
        out[out_key] = int(meta.get(src_key, 0) or 0)
    for src_key, out_key in _SHARD_META_FLOAT_FIELDS:
        out[out_key] = float(meta.get(src_key, 0.0) or 0.0)
    out["outcome_stats"] = _sanitize_outcome_stats(meta.get("outcome_stats"))
    return out


def _merge_minmax_summary(
    summary: dict[str, Any],
    *,
    n_key: str,
    min_key: str,
    max_key: str,
    incoming_n: int,
    incoming_min: float,
    incoming_max: float,
) -> None:
    if incoming_n <= 0:
        return
    old_n = max(0, int(summary.get(n_key, 0) or 0) - int(incoming_n))
    if old_n <= 0:
        summary[min_key] = float(incoming_min)
        summary[max_key] = float(incoming_max)
    else:
        summary[min_key] = min(float(summary.get(min_key, incoming_min)), float(incoming_min))
        summary[max_key] = max(float(summary.get(max_key, incoming_max)), float(incoming_max))


def _record_replay_array_metrics(
    shard_arrs: dict,
    train_mask: np.ndarray,
    summary: dict[str, Any],
) -> None:
    train_n = int(np.count_nonzero(train_mask))
    if train_n <= 0:
        return

    priorities = np.asarray(
        shard_arrs.get("priority", np.ones((train_mask.shape[0],), dtype=np.float32)),
        dtype=np.float32,
    )[train_mask]
    if priorities.size > 0:
        incoming_n = int(priorities.size)
        priorities64 = priorities.astype(np.float64, copy=False)
        summary["replay_priority_n"] += incoming_n
        summary["replay_priority_sum"] += float(priorities64.sum(dtype=np.float64))
        summary["replay_priority_sq_sum"] += float((priorities64 * priorities64).sum(dtype=np.float64))
        _merge_minmax_summary(
            summary,
            n_key="replay_priority_n",
            min_key="replay_priority_min",
            max_key="replay_priority_max",
            incoming_n=incoming_n,
            incoming_min=float(priorities.min()),
            incoming_max=float(priorities.max()),
        )

    for flag_name, n_key, sum_key in (
        ("has_policy", "replay_has_policy_n", "replay_has_policy_sum"),
        ("has_sf_wdl", "replay_has_sf_wdl_n", "replay_has_sf_wdl_sum"),
        ("has_search_wdl", "replay_has_search_wdl_n", "replay_has_search_wdl_sum"),
    ):
        if flag_name not in shard_arrs:
            continue
        flags = np.asarray(shard_arrs[flag_name], dtype=np.uint8)[train_mask]
        summary[n_key] += int(flags.size)
        summary[sum_key] += int(flags.sum(dtype=np.int64))

    if "wdl_target" in shard_arrs:
        wdl = np.asarray(shard_arrs["wdl_target"], dtype=np.int64)[train_mask]
        counts = np.bincount(np.clip(wdl, 0, 2), minlength=3)
        summary["replay_wdl_0"] += int(counts[0])
        summary["replay_wdl_1"] += int(counts[1])
        summary["replay_wdl_2"] += int(counts[2])


def _ingest_train_arrays(
    shard_arrs: dict,
    shard_n: int,
    *,
    buf: DiskReplayBuffer,
    holdout_buf: ArrayReplayBuffer,
    holdout_frac: float,
    holdout_frozen: bool,
    rng: np.random.Generator,
    summary: dict[str, Any],
) -> None:
    """Split shard rows into holdout vs train, push to buffers, count is_selfplay tags."""
    if shard_n <= 0:
        return
    holdout_mask = np.zeros((shard_n,), dtype=bool)
    if holdout_frac > 0.0 and (not holdout_frozen):
        holdout_mask = rng.random(shard_n) < holdout_frac
        if np.any(holdout_mask):
            holdout_buf.add_many_arrays(
                slice_array_batch(shard_arrs, np.flatnonzero(holdout_mask))
            )
    train_mask = ~holdout_mask
    if not np.any(train_mask):
        return
    _record_replay_array_metrics(shard_arrs, train_mask, summary)
    buf.add_many_arrays(slice_array_batch(shard_arrs, np.flatnonzero(train_mask)))
  # Per-sample is_selfplay tag count, training rows only. Shards written
  # before this field existed won't carry it — silently skip.
    if "has_is_selfplay" not in shard_arrs:
        return
    has_sp = np.asarray(shard_arrs["has_is_selfplay"], dtype=np.uint8)[train_mask]
    tagged = int(has_sp.sum())
    if tagged <= 0:
        return
    is_sp = np.asarray(shard_arrs.get("is_selfplay", np.zeros_like(has_sp)), dtype=np.uint8)[train_mask]
    summary["ingest_is_selfplay_tagged"] += tagged
    summary["ingest_is_selfplay_true"] += int((is_sp & has_sp).sum())


def _process_shard(
    sp: Path,
    *,
    inbox_dir: Path,
    processed_dir: Path,
    buf: DiskReplayBuffer,
    holdout_buf: ArrayReplayBuffer,
    holdout_frac: float,
    holdout_frozen: bool,
    accepted_model_shas: set[str],
    rng: np.random.Generator,
    summary: dict[str, Any],
    preloaded: tuple[dict, dict] | None = None,
    prev_model_sha: str | None = None,
) -> str:
    """Load one shard from inbox, ingest into replay buffer, update summary.

    Returns the shard's model_sha256 (empty string if unknown).

    If ``preloaded`` is provided, skip the disk read — the background
    prefetcher already has ``(shard_arrs, meta)`` in memory. The atomic
    move of the original ``sp`` file from inbox→processed still happens
    here, so the prefetcher must NOT touch sp other than reading it.
    """
    rel = sp.relative_to(inbox_dir)
    out = processed_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if preloaded is not None:
        if not sp.exists():
            return ""
        shard_arrs, meta = preloaded
    else:
        try:
            shard_arrs, meta = load_shard_arrays(sp)
        except Exception:
            bad = processed_dir / "bad" / rel.name
            bad.parent.mkdir(parents=True, exist_ok=True)
            try:
                sp.replace(bad)
            except Exception:
                delete_shard_path(sp)
            return ""

    model_sha = str(meta.get("model_sha256") or "")
    shard_n = int(np.asarray(shard_arrs["x"]).shape[0])
    m = _extract_shard_metrics(meta, shard_n)

    _ingest_train_arrays(
        shard_arrs, shard_n,
        buf=buf, holdout_buf=holdout_buf,
        holdout_frac=holdout_frac, holdout_frozen=holdout_frozen,
        rng=rng, summary=summary,
    )
    summary["positions_replay_added"] += m["positions"]

    if model_sha in accepted_model_shas:
        for key, val in m.items():
            if key in {"diff_focus_priority_min", "diff_focus_priority_max", "outcome_stats"}:
                continue
            summary[f"matching_{key}"] += val
        _merge_outcome_stats(
            summary["matching_outcome_stats"],
            m.get("outcome_stats", {}),
        )
        _merge_minmax_summary(
            summary,
            n_key="matching_diff_focus_records",
            min_key="matching_diff_focus_priority_min",
            max_key="matching_diff_focus_priority_max",
            incoming_n=int(m.get("diff_focus_records", 0) or 0),
            incoming_min=float(m.get("diff_focus_priority_min", 0.0) or 0.0),
            incoming_max=float(m.get("diff_focus_priority_max", 0.0) or 0.0),
        )
        summary["matching_shards"] += 1
  # Anchored gate split. ``prev_model_sha`` is only ever the sha the trainer
  # published LAST iteration, so "not prev" is unambiguously the current one:
  # any other sha would not be in ``accepted_model_shas`` at all and would have
  # taken the stale branch below.
        side = "prev" if (prev_model_sha and model_sha == prev_model_sha) else "cur"
        summary[f"gate_{side}_w"] += m["w"]
        summary[f"gate_{side}_d"] += m["d"]
        summary[f"gate_{side}_l"] += m["l"]
  # ...and the DIFFICULTY that arm played at, games-weighted. Model and
  # difficulty ship in one manifest, so the prev arm is always one PID step
  # behind: without this the anchored delta carries a controller term nothing
  # can measure (see promotion_gate's "THE PID LAG DOES NOT CANCEL"). Shards
  # written before ShardMeta carried the field contribute NOTHING to either
  # sum or denominator -- absent is UNKNOWN, and a 0.0 regret would read as
  # "unhandicapped Stockfish", the opposite end of the range.
  # From the raw shard META, not from ``m``: ``_extract_shard_metrics``
  # yields the counters the ``matching_`` loop above SUMS, and a difficulty
  # is not summable.
        shard_regret = meta.get("opponent_wdl_regret_limit")
        if shard_regret is not None:
            shard_games = int(m["w"]) + int(m["d"]) + int(m["l"])
            summary[f"gate_{side}_regret_weighted"] += (
                float(shard_regret) * shard_games
            )
            summary[f"gate_{side}_regret_games"] += shard_games
    else:
        summary["stale_games"] += m["games"]
        summary["stale_positions"] += m["positions"]
        summary["stale_shards"] += 1

    try:
        sp.replace(out)
    except Exception:
        delete_shard_path(sp)
    return model_sha


def _process_shard_with_prev_cap(
    sp: Path,
    *,
    inbox_dir: Path,
    processed_dir: Path,
    buf: DiskReplayBuffer,
    holdout_buf: ArrayReplayBuffer,
    holdout_frac: float,
    holdout_frozen: bool,
    effective_accepted: set[str],
    rng: np.random.Generator,
    summary: dict[str, Any],
    cap_prev: bool,
    prev_model_sha: str | None,
    prev_max_games: int,
    prev_matching_games_box: list[int],
    preloaded: tuple | None = None,
) -> None:
    """Ingest one shard and apply the prev-model SHA cap. Mutates ``summary`` and
    ``effective_accepted`` (drops prev SHA once its quota is reached).

    ``prev_matching_games_box`` is a single-element list used as a mutable counter
    carried across the prefetcher-drain and the poll loop. Both of those run on
    the TRAINER THREAD, which is the only writer of this counter, of ``summary``
    and of ``effective_accepted`` — the box is a closure cell, NOT a cross-thread
    handoff, and none of these three unguarded read-modify-writes is safe against
    one (audit L6). The prefetcher thread only decodes; registration stays here.
    """
    games_before = summary["matching_games"]
    shard_sha = _process_shard(
        sp,
        inbox_dir=inbox_dir,
        processed_dir=processed_dir,
        buf=buf,
        holdout_buf=holdout_buf,
        holdout_frac=holdout_frac,
        holdout_frozen=holdout_frozen,
        accepted_model_shas=effective_accepted,
        rng=rng,
        summary=summary,
        preloaded=preloaded,
        prev_model_sha=prev_model_sha,
    )
    if not cap_prev or prev_model_sha not in effective_accepted or shard_sha != prev_model_sha:
        return
    games_added = int(summary["matching_games"]) - int(games_before)
    if games_added <= 0:
        return
    prev_matching_games_box[0] += games_added
    if prev_matching_games_box[0] >= prev_max_games:
        effective_accepted.discard(prev_model_sha)


def _ingest_distributed_selfplay(
    *,
    buf: DiskReplayBuffer,
    holdout_buf: ArrayReplayBuffer,
    holdout_frac: float,
    holdout_frozen: bool,
    inbox_dir: Path,
    processed_dir: Path,
    target_games: int,
    accepted_model_shas: set[str],
    wait_timeout_s: float,
    poll_seconds: float,
    rng: np.random.Generator,
    min_games_fraction: float = 0.5,
    prev_model_sha: str | None = None,
    prev_model_max_fraction: float = 1.0,
    prefetcher=None,
  # REQUIRED, deliberately without a default. This wait loop can run for
  # wait_timeout_s * 3 (8100s in production), and whether anything checks that
  # the fleet is still alive during it is the difference between a retry and
  # the 2026-07-24 50-minute outage. A default of None would let the single
  # production call site drop the callback in a refactor and still run, just
  # silently without recovery. Pass None explicitly to opt out.
    on_poll: Callable[[], None] | None,
    on_poll_interval_s: float = 60.0,
) -> dict[str, Any]:
    """Poll inbox until enough games arrive, then return.

    Shards whose ``model_sha256`` is in *accepted_model_shas* count as
    ``matching`` and contribute toward *target_games*.  Typically this
    includes both the current model SHA and the previous one so that
    the one-generation-stale batch workers finish after a model update
    counts toward the target instead of creating a permanent lag.

    The timeout only fires once at least ``min_games_fraction`` of
    *target_games* have been collected.  This prevents pathologically
    thin iterations that destabilise training.

    *on_poll* is invoked at most once per *on_poll_interval_s* while waiting,
    and is where the caller revives dead selfplay processes.  It runs inside
    the loop rather than at the phase boundary because a broker death here is
    otherwise invisible until the whole wait times out: with zero matching
    games the soft deadline's ``min_games`` guard can never fire, so the fleet
    sits idle for the FULL hard ceiling of ``wait_timeout_s * 3``.  On
    2026-07-24 that was 50 minutes of zero games from one dead process.
    Exceptions from *on_poll* are logged and swallowed — a revive that fails
    must not abort an ingest that is otherwise progressing.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    target_games = max(1, int(target_games))
    min_games = max(1, math.ceil(float(min_games_fraction) * target_games))
    _now = time.time()
    deadline = _now + float(wait_timeout_s)
  # Hard ceiling: if matching_games never reaches min_games (all workers stale
  # or dead), the soft deadline's matching_games guard never fires.  3× gives
  # workers time to restart while bounding the worst-case hang.
    hard_deadline = _now + float(wait_timeout_s) * 3.0
    summary = _empty_ingest_summary()

  # Cap prev-model games at a fraction of target.  Once reached, demote
  # the prev SHA so further prev-model shards count as stale.
  # Skip if prev == current (discarding would remove the only accepted SHA).
    cap_prev = bool(prev_model_sha) and len(accepted_model_shas) > 1
    prev_max_games = math.ceil(float(prev_model_max_fraction) * target_games) if cap_prev else 0
    prev_matching_games_box = [0]
    effective_accepted = set(accepted_model_shas)

    def _ingest(sp: Path, *, preloaded: tuple | None = None) -> None:
        _process_shard_with_prev_cap(
            sp, inbox_dir=inbox_dir, processed_dir=processed_dir,
            buf=buf, holdout_buf=holdout_buf,
            holdout_frac=holdout_frac, holdout_frozen=holdout_frozen,
            effective_accepted=effective_accepted, rng=rng,
            summary=summary, cap_prev=cap_prev,
            prev_model_sha=prev_model_sha, prev_max_games=prev_max_games,
            prev_matching_games_box=prev_matching_games_box,
            preloaded=preloaded,
        )

  # Drain prefetcher first so the inbox-poll fallback below only sees
  # shards that arrived after the last background scan (the trainer's
  # atomic inbox→processed move inside _process_shard prevents the
  # next scan from re-picking the same path).
    if prefetcher is not None:
        for sp, arrs, meta in prefetcher.drain():
            _ingest(sp, preloaded=(arrs, meta))

  # Due immediately, NOT one interval from now. The fleet is ensured at the
  # phase boundary just above, but a broker can die during its own launch or
  # in the gap before the first shard scan -- and with zero matching games the
  # soft deadline's min_games guard can never fire, so that death costs the
  # full hard ceiling. Waiting 60s to ask the first question is 60s of an
  # outage we already know how to end.
    next_poll_cb = _now

    def _maybe_on_poll() -> None:
        nonlocal next_poll_cb
        if on_poll is None:
            return
        now = time.time()
        if now < next_poll_cb:
            return
        next_poll_cb = now + float(on_poll_interval_s)
        try:
            on_poll()
        except Exception:
            log.exception("ingest on_poll callback failed; continuing to wait")

    while summary["matching_games"] < target_games:
        _maybe_on_poll()
        _now = time.time()
        if _now >= deadline and summary["matching_games"] >= min_games:
            break
        if _now >= hard_deadline:
            log.warning(
                "ingest hard timeout (%.0fs): %d/%d matching games — all workers likely stale",
                wait_timeout_s * 3, summary["matching_games"], target_games,
            )
            break
        shard_paths = _iter_shard_paths_nested(inbox_dir)
        if not shard_paths:
            time.sleep(float(poll_seconds))
            continue

        for sp in shard_paths:
          # Also polled here, not just at the top of the while: shard_paths is a
          # snapshot, so a several-hundred-shard backlog at ~0.2-2s each would
          # otherwise starve the revive for minutes. The call is a single
          # time.time() compare when not yet due, so it costs nothing per shard.
            _maybe_on_poll()
            _ingest(sp)
            if summary["matching_games"] >= target_games:
                break
            _now = time.time()
            if _now >= deadline and summary["matching_games"] >= min_games:
                break
            if _now >= hard_deadline:
                break

    return summary
