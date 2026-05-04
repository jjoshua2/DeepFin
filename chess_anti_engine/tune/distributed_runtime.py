from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from chess_anti_engine.model import ModelConfig, model_config_to_manifest_dict
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay import ArrayReplayBuffer, DiskReplayBuffer
from chess_anti_engine.replay.shard import (
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
from chess_anti_engine.tune.process_cleanup import terminate_matching_processes
from chess_anti_engine.utils import sha256_file
from chess_anti_engine.utils.atomic import atomic_write_text
from chess_anti_engine.version import PACKAGE_VERSION, PROTOCOL_VERSION

log = logging.getLogger(__name__)

# Repo root used as cwd for spawned workers/brokers (so relative imports of
# the chess_anti_engine package work). Resolved once at import time — fs walk
# would otherwise repeat for every spawn.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Cache for SHA256 of static files (opening books, worker wheel) that don't
# change during a run.  Keyed by (path_str, file_size).
_static_sha_cache: dict[tuple[str, int], str] = {}


def _sha256_cached(p: Path) -> str:
    """SHA256 with caching for files that don't change during a run."""
    key = (str(p), p.stat().st_size)
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

    Also skips the server's ``_pending`` staging dir. Pending shards already
    contributed their samples to an in-memory accumulator and will land in
    replay via ``_compacted`` once the server flushes; ingesting them here
    would double-count them.
    """
    paths: list[Path] = []
    try:
        user_dirs = list(root.iterdir())
    except FileNotFoundError:
        return paths
    for user_dir in user_dirs:
        if not user_dir.is_dir() or _is_tmp_shard_name(user_dir.name):
            continue
        if user_dir.name == _PENDING_DIR_NAME:
            continue
        try:
            for entry in user_dir.iterdir():
                name = entry.name
                if _is_tmp_shard_name(name):
                    continue
                if name.endswith(LOCAL_SHARD_SUFFIX) or name.endswith(LEGACY_SHARD_SUFFIX):
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
            if user_dir.stat().st_mtime >= cutoff:
                continue
            entries = list(user_dir.iterdir())
        except FileNotFoundError:
            continue
        for entry in entries:
            name = entry.name
            if _is_tmp_shard_name(name):
                continue
            if not (name.endswith(LOCAL_SHARD_SUFFIX) or name.endswith(LEGACY_SHARD_SUFFIX)):
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
) -> str:
    dirs = _trial_server_dirs(server_root=server_root, trial_id=trial_id)
    publish_dir = dirs["publish_dir"]
    publish_dir.mkdir(parents=True, exist_ok=True)

    model_path = publish_dir / "latest_model.pt"
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

    recommended_worker = {
        "games_per_batch": int(config.get("selfplay_batch", 4)),
        "max_plies": int(config.get("max_plies", 240)),
        "mcts": str(config.get("mcts", "puct")),
        "mcts_simulations": int(mcts_simulations),
        "playout_cap_fraction": float(config.get("playout_cap_fraction", 0.25)),
        "fast_simulations": int(config.get("fast_simulations", 8)),
        "opening_book_max_plies": int(config.get("opening_book_max_plies", 4)),
        "opening_book_max_games": int(config.get("opening_book_max_games", 200_000)),
        "opening_book_prob": float(config.get("opening_book_prob", 1.0)),
        "opening_book_path_2": config.get("opening_book_path_2"),
        "opening_book_max_plies_2": int(config.get("opening_book_max_plies_2", 16)),
        "opening_book_max_games_2": int(config.get("opening_book_max_games_2", 200_000)),
        "opening_book_mix_prob_2": float(config.get("opening_book_mix_prob_2", 0.0)),
        "random_start_plies": int(config.get("random_start_plies", 0)),
        "selfplay_fraction": float(config.get("selfplay_fraction", 0.0)),
        "sf_nodes": int(sf_nodes),
        "sf_multipv": int(config.get("sf_multipv", 1)),
        "sf_policy_temp": float(config.get("sf_policy_temp", 0.25)),
        "sf_policy_label_smooth": float(config.get("sf_policy_label_smooth", 0.05)),
        "opponent_wdl_regret_limit": float(wdl_regret) if float(wdl_regret) >= 0.0 else None,
        "temperature": float(config.get("temperature", 1.0)),
        "temperature_decay_start_move": int(config.get("temperature_decay_start_move", 20)),
        "temperature_decay_moves": int(config.get("temperature_decay_moves", 60)),
        "temperature_endgame": float(config.get("temperature_endgame", 0.6)),
        "timeout_adjudication_threshold": float(config.get("timeout_adjudication_threshold", 0.90)),
  # Syzygy. `path` must be a filesystem location visible to workers
  # (same layout on all nodes in a multi-node deployment). Server
  # operators can edit these directly in publish/manifest.json to
  # change endgame adjudication behavior without restarting anyone.
        "syzygy_path": config.get("syzygy_path") or None,
        "syzygy_rescore_policy": bool(config.get("syzygy_rescore_policy", False)),
        "syzygy_adjudicate": bool(config.get("syzygy_adjudicate", False)),
        "syzygy_adjudicate_fraction": float(config.get("syzygy_adjudicate_fraction", 1.0)),
        "syzygy_in_search": bool(config.get("syzygy_in_search", False)),
        "pause_selfplay": bool(pause_selfplay),
        "pause_reason": str(pause_reason),
    }

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
        },
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

    if published_worker_wheel_path is not None and published_worker_wheel_path.exists():
        manifest["worker_wheel"] = {
            "endpoint": api_prefix + "/worker_wheel",
            "filename": published_worker_wheel_path.name,
            "sha256": _sha256_cached(published_worker_wheel_path),
            "version": str(PACKAGE_VERSION),
        }

    atomic_write_text(
        publish_dir / "manifest.json",
        json.dumps(manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return model_sha


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
    return _spawn_with_reap(
        cmd=cmd,
        log_path=worker_out,
        reap_module="chess_anti_engine.worker",
        reap_terms=["--trial-id", str(trial_id), "--work-dir", str(worker_root)],
        reap_label=f"distributed workers (trial={trial_id} idx={worker_index:02d})",
    )


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
            ["--aot-dir", str(config.get("distributed_worker_aot_dir", ""))]
            if str(config.get("distributed_worker_aot_dir", "")).strip()
            else (
                ["--compile-inference", "--compile-mode", str(config.get("distributed_worker_compile_mode", "reduce-overhead"))]
                if bool(config.get("distributed_worker_use_compile", False))
                else []
            )
        ),
        *(["--inference-fp8"] if bool(config.get("distributed_worker_inference_fp8", False)) else []),
        *(
            ["--threaded-selfplay", "--selfplay-threads",
             str(int(config.get("distributed_worker_selfplay_threads", 16)))]
            + (["--threaded-dispatcher",
                "--dispatcher-batch-wait-ms",
                str(float(config.get("distributed_worker_dispatcher_batch_wait_ms", 1.0)))]
               if bool(config.get("distributed_worker_threaded_dispatcher", False)) else [])
            if bool(config.get("distributed_worker_threaded", False))
            else []
        ),
        "--sf-workers",
        str(int(config.get("distributed_worker_sf_workers", 1))),
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
        slot_name = f"{slot_prefix}-{worker_index}"
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
    """Deterministic shared-memory slot prefix for a trial's inference broker."""
    h = stable_seed_u32("slot-prefix", trial_id)
    return f"cae-{h:08x}"


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


def _resolve_max_batch_per_slot(config: dict) -> int:
    return int(
        config.get(
            "distributed_inference_max_batch_per_slot",
            config.get("distributed_inference_max_batch_positions", 256),
        )
    )


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
    cmd = [
        sys.executable, "-m", "chess_anti_engine.inference",
        "--publish-dir", str(publish_dir),
        "--slot-prefix", str(slot_prefix),
        "--num-slots", str(int(config.get("distributed_workers_per_trial", 2))),
        "--max-batch-per-slot", str(_resolve_max_batch_per_slot(config)),
        "--device", str(config.get("distributed_worker_device") or config.get("device", "cpu")),
        "--batch-wait-ms", str(float(config.get("distributed_inference_batch_wait_ms", 5.0))),
        "--shared-cache-dir", str(_resolve_shared_cache_root(config, server_root)),
        *(["--compile-inference"] if _resolve_compile_inference(config) else []),
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
        "--slots-per-trial", str(int(config.get("distributed_workers_per_trial", 2))),
        "--max-batch-per-slot", str(_resolve_max_batch_per_slot(config)),
        "--device", str(config.get("distributed_worker_device") or config.get("device", "cpu")),
        "--batch-wait-ms", str(float(config.get("distributed_inference_batch_wait_ms", 0.0))),
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
        _stop_process(proc)


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
        if idx < len(out) and out[idx].poll() is None:
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


def _empty_ingest_summary() -> dict[str, int | float]:
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
  # Per-sample source counters (is_selfplay tag): sum of tagged samples
  # and the selfplay-true subset, across ingested shards.  Used to
  # compute ingest_frac_selfplay = selfplay / tagged.
        "ingest_is_selfplay_tagged": 0,
        "ingest_is_selfplay_true": 0,
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
)


def _extract_shard_metrics(meta: dict, shard_n: int) -> dict[str, int | float]:
    """Pull all per-shard counts/sums from the meta dict in one place.

    Output keys map directly onto summary["matching_*"] suffixes (no prefix);
    callers add the matching/stale prefix at update time.
    """
    wins = int(meta.get("wins", 0) or 0)
    draws = int(meta.get("draws", 0) or 0)
    losses = int(meta.get("losses", 0) or 0)
    out: dict[str, int | float] = {
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
    return out


def _ingest_train_arrays(
    shard_arrs: dict,
    shard_n: int,
    *,
    buf: DiskReplayBuffer,
    holdout_buf: ArrayReplayBuffer,
    holdout_frac: float,
    holdout_frozen: bool,
    rng: np.random.Generator,
    summary: dict[str, int | float],
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
    summary: dict[str, int | float],
    preloaded: tuple[dict, dict] | None = None,
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
            summary[f"matching_{key}"] += val
        summary["matching_shards"] += 1
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
    summary: dict[str, int | float],
    cap_prev: bool,
    prev_model_sha: str | None,
    prev_max_games: int,
    prev_matching_games_box: list[int],
    preloaded: tuple | None = None,
) -> None:
    """Ingest one shard and apply the prev-model SHA cap. Mutates ``summary`` and
    ``effective_accepted`` (drops prev SHA once its quota is reached).

    ``prev_matching_games_box`` is a single-element list used as a mutable counter
    shared between the prefetcher-drain and the poll loop.
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
) -> dict[str, int | float]:
    """Poll inbox until enough games arrive, then return.

    Shards whose ``model_sha256`` is in *accepted_model_shas* count as
    ``matching`` and contribute toward *target_games*.  Typically this
    includes both the current model SHA and the previous one so that
    the one-generation-stale batch workers finish after a model update
    counts toward the target instead of creating a permanent lag.

    The timeout only fires once at least ``min_games_fraction`` of
    *target_games* have been collected.  This prevents pathologically
    thin iterations that destabilise training.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    target_games = max(1, int(target_games))
    min_games = max(1, int(math.ceil(float(min_games_fraction) * target_games)))
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
    prev_max_games = int(math.ceil(float(prev_model_max_fraction) * target_games)) if cap_prev else 0
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

    while summary["matching_games"] < target_games:
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
            _ingest(sp)
            if summary["matching_games"] >= target_games:
                break
            _now = time.time()
            if _now >= deadline and summary["matching_games"] >= min_games:
                break
            if _now >= hard_deadline:
                break

    return summary
