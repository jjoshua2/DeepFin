from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from chess_anti_engine.model import ModelConfig
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay import ArrayReplayBuffer, DiskReplayBuffer
from chess_anti_engine.replay.shard import load_npz_arrays
from chess_anti_engine.train import Trainer
from chess_anti_engine.tune._utils import (
    atomic_write_text,
    resolve_local_override_root,
    sha256_file,
    slice_array_batch,
    stable_seed_u32,
    terminate_process as _stop_process,
)
from chess_anti_engine.tune.process_cleanup import terminate_matching_processes
from chess_anti_engine.version import PACKAGE_VERSION, PROTOCOL_VERSION

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
    for sp in sorted(inbox_dir.glob("*/*.npz")):
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
    random_move_prob: float,
    skill_level: int,
    mcts_simulations: int,
    wdl_regret: float = -1.0,
    pause_selfplay: bool = False,
    pause_reason: str = "",
    backpressure: dict[str, object] | None = None,
    export_model: bool = True,
) -> str:
    dirs = _trial_server_dirs(server_root=server_root, trial_id=trial_id)
    publish_dir = dirs["publish_dir"]
    publish_dir.mkdir(parents=True, exist_ok=True)

    model_path = publish_dir / "latest_model.pt"
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
        "sf_skill_level": int(skill_level),
        "opponent_random_move_prob": float(random_move_prob),
        "opponent_topk_stage_end": float(config.get("sf_pid_topk_stage_end", config.get("sf_pid_random_move_stage_end", 0.5))),
        "opponent_topk_min": int(config.get("sf_pid_topk_min", 1)),
        "opponent_suboptimal_wdl_regret_max": float(config.get("sf_pid_suboptimal_wdl_regret_max", -1.0)),
        "opponent_suboptimal_wdl_regret_min": float(config.get("sf_pid_suboptimal_wdl_regret_min", -1.0)),
        "opponent_random_move_prob_start": float(config.get("sf_pid_random_move_prob_start", 1.0)),
        "opponent_random_move_prob_min": float(config.get("sf_pid_random_move_prob_min", 0.0)),
        "opponent_wdl_regret_limit": float(wdl_regret) if float(wdl_regret) >= 0.0 else None,
        "temperature": float(config.get("temperature", 1.0)),
        "temperature_decay_start_move": int(config.get("temperature_decay_start_move", 20)),
        "temperature_decay_moves": int(config.get("temperature_decay_moves", 60)),
        "temperature_endgame": float(config.get("temperature_endgame", 0.6)),
        "timeout_adjudication_threshold": float(config.get("timeout_adjudication_threshold", 0.90)),
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
        "model_config": {
            "kind": str(model_cfg.kind),
            "embed_dim": int(model_cfg.embed_dim),
            "num_layers": int(model_cfg.num_layers),
            "num_heads": int(model_cfg.num_heads),
            "ffn_mult": float(model_cfg.ffn_mult),
            "use_smolgen": bool(model_cfg.use_smolgen),
            "use_nla": bool(model_cfg.use_nla),
            "use_qk_rmsnorm": bool(getattr(model_cfg, "use_qk_rmsnorm", False)),
            "gradient_checkpointing": bool(model_cfg.use_gradient_checkpointing),
        },
    }

    opening_book_path = config.get("opening_book_path")
    if isinstance(opening_book_path, str) and opening_book_path.strip():
        p = Path(opening_book_path.strip())
        if p.exists():
            manifest["opening_book"] = {
                "endpoint": "/v1/opening_book",
                "filename": p.name,
                "sha256": _sha256_cached(p),
            }

    opening_book_path_2 = config.get("opening_book_path_2")
    if isinstance(opening_book_path_2, str) and opening_book_path_2.strip():
        p2 = Path(opening_book_path_2.strip())
        if p2.exists():
            manifest["opening_book_2"] = {
                "endpoint": "/v1/opening_book_2",
                "filename": p2.name,
                "sha256": _sha256_cached(p2),
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

    out_fh = worker_out.open("ab")
    try:
        stale_worker_pids = terminate_matching_processes(
            module="chess_anti_engine.worker",
            required_terms=["--trial-id", str(trial_id), "--work-dir", str(worker_root)],
        )
        if stale_worker_pids:
            print(
                f"[trial] reaped stale distributed workers: "
                f"trial={trial_id} worker_index={worker_index} pids={stale_worker_pids}"
            )
        return subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=out_fh,
            stderr=subprocess.STDOUT,
        )
    finally:
        out_fh.close()


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
            ["--compile-inference"]
            if bool(config.get("distributed_worker_use_compile", False))
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
        "info",
    ]

    if bool(config.get("distributed_inference_broker_enabled", False)):
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

    if bool(config.get("distributed_worker_auto_tune", False)):
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


def _trial_inference_endpoint(*, trial_id: str) -> str:
    port = 46000 + (stable_seed_u32("inference-broker", trial_id) % 10000)
    return f"127.0.0.1:{int(port)}"


def _trial_slot_prefix(*, trial_id: str) -> str:
    """Deterministic shared-memory slot prefix for a trial's inference broker."""
    h = stable_seed_u32("slot-prefix", trial_id)
    return f"cae-{h:08x}"


def _launch_inference_broker(
    *,
    config: dict,
    trial_id: str,
    publish_dir: Path,
    trial_dir: Path,
) -> subprocess.Popen[bytes]:
    broker_artifact_root = trial_dir / "distributed_inference"
    broker_artifact_root.mkdir(parents=True, exist_ok=True)
    broker_out = broker_artifact_root / "broker.out"
    slot_prefix = _trial_slot_prefix(trial_id=trial_id)
    num_workers = int(config.get("distributed_workers_per_trial", 2))
    max_batch = int(
        config.get(
            "distributed_inference_max_batch_per_slot",
            config.get("distributed_inference_max_batch_positions", 256),
        )
    )
    shared_cache_raw = str(config.get("distributed_worker_shared_cache_dir") or "").strip()
    if shared_cache_raw:
        shared_cache_root = Path(shared_cache_raw).expanduser()
    else:
        shared_cache_root = Path(str(config["distributed_server_root"])) / "worker_cache"
    shared_cache_root.mkdir(parents=True, exist_ok=True)
    compile_inference = bool(config.get("distributed_inference_use_compile", False))
    cmd = [
        sys.executable,
        "-m",
        "chess_anti_engine.inference",
        "--publish-dir",
        str(publish_dir),
        "--slot-prefix",
        str(slot_prefix),
        "--num-slots",
        str(num_workers),
        "--max-batch-per-slot",
        str(max_batch),
        "--device",
        str(config.get("distributed_worker_device") or config.get("device", "cpu")),
        "--batch-wait-ms",
        str(float(config.get("distributed_inference_batch_wait_ms", 5.0))),
        "--shared-cache-dir",
        str(shared_cache_root),
        *(["--compile-inference"] if compile_inference else []),
    ]
    out_fh = broker_out.open("ab")
    try:
        stale_broker_pids = terminate_matching_processes(
            module="chess_anti_engine.inference",
            required_terms=["--publish-dir", str(publish_dir), "--slot-prefix", str(slot_prefix)],
        )
        if stale_broker_pids:
            print(f"[trial] reaped stale inference brokers: trial={trial_id} pids={stale_broker_pids}")
        return subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=out_fh,
            stderr=subprocess.STDOUT,
        )
    finally:
        out_fh.close()


def _ensure_inference_broker(
    *,
    config: dict,
    trial_id: str,
    trial_dir: Path,
    publish_dir: Path,
    proc: subprocess.Popen[bytes] | None,
) -> subprocess.Popen[bytes] | None:
    if not bool(config.get("distributed_inference_broker_enabled", False)):
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
    return out[:want]


def _empty_ingest_summary() -> dict[str, int]:
    return {
        "matching_games": 0,
        "matching_positions": 0,
        "matching_w": 0,
        "matching_d": 0,
        "matching_l": 0,
        "matching_total_game_plies": 0,
        "matching_adjudicated_games": 0,
        "matching_total_draw_games": 0,
        "matching_selfplay_games": 0,
        "matching_selfplay_adjudicated_games": 0,
        "matching_selfplay_draw_games": 0,
        "matching_curriculum_games": 0,
        "matching_curriculum_adjudicated_games": 0,
        "matching_curriculum_draw_games": 0,
        "positions_replay_added": 0,
        "stale_games": 0,
        "stale_positions": 0,
        "matching_shards": 0,
        "stale_shards": 0,
    }


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
    summary: dict[str, int],
) -> None:
    """Load one shard from inbox, ingest into replay buffer, update summary."""
    rel = sp.relative_to(inbox_dir)
    out = processed_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        shard_arrs, meta = load_npz_arrays(sp)
    except Exception:
        bad = processed_dir / "bad" / rel.name
        bad.parent.mkdir(parents=True, exist_ok=True)
        try:
            sp.replace(bad)
        except Exception:
            sp.unlink(missing_ok=True)
        return

    model_sha = str(meta.get("model_sha256") or "")
    wins = int(meta.get("wins", 0) or 0)
    draws = int(meta.get("draws", 0) or 0)
    losses = int(meta.get("losses", 0) or 0)
    games = int(meta.get("games", wins + draws + losses) or 0)
    shard_n = int(np.asarray(shard_arrs["x"]).shape[0])
    positions = int(meta.get("positions", shard_n) or shard_n)
    total_game_plies = int(meta.get("total_game_plies", 0) or 0)
    adjudicated_games = int(meta.get("adjudicated_games", meta.get("timeout_games", 0)) or 0)
    total_draw_games = int(meta.get("total_draw_games", draws) or draws)
    selfplay_games = int(meta.get("selfplay_games", 0) or 0)
    selfplay_adjudicated_games = int(meta.get("selfplay_adjudicated_games", 0) or 0)
    selfplay_draw_games = int(meta.get("selfplay_draw_games", 0) or 0)
    curriculum_games = int(meta.get("curriculum_games", 0) or 0)
    curriculum_adjudicated_games = int(meta.get("curriculum_adjudicated_games", 0) or 0)
    curriculum_draw_games = int(meta.get("curriculum_draw_games", 0) or 0)

    if shard_n > 0:
        holdout_mask = np.zeros((shard_n,), dtype=bool)
        if holdout_frac > 0.0 and (not holdout_frozen):
            holdout_mask = rng.random(shard_n) < holdout_frac
            if np.any(holdout_mask):
                holdout_buf.add_many_arrays(
                    slice_array_batch(shard_arrs, np.flatnonzero(holdout_mask))
                )

        train_mask = ~holdout_mask
        if np.any(train_mask):
            train_arrs = slice_array_batch(shard_arrs, np.flatnonzero(train_mask))
            buf.add_many_arrays(train_arrs)
    summary["positions_replay_added"] += positions

    if model_sha in accepted_model_shas:
        summary["matching_games"] += games
        summary["matching_positions"] += positions
        summary["matching_w"] += wins
        summary["matching_d"] += draws
        summary["matching_l"] += losses
        summary["matching_total_game_plies"] += total_game_plies
        summary["matching_adjudicated_games"] += adjudicated_games
        summary["matching_total_draw_games"] += total_draw_games
        summary["matching_selfplay_games"] += selfplay_games
        summary["matching_selfplay_adjudicated_games"] += selfplay_adjudicated_games
        summary["matching_selfplay_draw_games"] += selfplay_draw_games
        summary["matching_curriculum_games"] += curriculum_games
        summary["matching_curriculum_adjudicated_games"] += curriculum_adjudicated_games
        summary["matching_curriculum_draw_games"] += curriculum_draw_games
        summary["matching_shards"] += 1
    else:
        summary["stale_games"] += games
        summary["stale_positions"] += positions
        summary["stale_shards"] += 1

    try:
        sp.replace(out)
    except Exception:
        sp.unlink(missing_ok=True)


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
) -> dict[str, int]:
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
    deadline = time.time() + float(wait_timeout_s)
    summary = _empty_ingest_summary()
    _shard_kw = dict(
        inbox_dir=inbox_dir, processed_dir=processed_dir,
        buf=buf, holdout_buf=holdout_buf,
        holdout_frac=holdout_frac, holdout_frozen=holdout_frozen,
        accepted_model_shas=accepted_model_shas, rng=rng, summary=summary,
    )

    while summary["matching_games"] < target_games:
        shard_paths = sorted(inbox_dir.glob("*/*.npz"))
        if not shard_paths:
            if time.time() >= deadline and summary["matching_games"] >= min_games:
                break
            time.sleep(float(poll_seconds))
            continue

        processed_any = False
        for sp in shard_paths:
            processed_any = True
            _process_shard(sp, **_shard_kw)

            if summary["matching_games"] >= target_games:
                break

        if not processed_any:
            if time.time() >= deadline and summary["matching_games"] >= min_games:
                break
            time.sleep(float(poll_seconds))

    return summary


def _ingest_available_shards(
    *,
    buf: DiskReplayBuffer,
    holdout_buf: ArrayReplayBuffer,
    holdout_frac: float,
    holdout_frozen: bool,
    inbox_dir: Path,
    processed_dir: Path,
    accept_model_shas: set[str],
    rng: np.random.Generator,
) -> dict[str, int]:
    """Non-blocking ingest: scan inbox once, process all available shards, return immediately.

    Unlike ``_ingest_distributed_selfplay`` this never polls or waits.
    Shards whose ``model_sha256`` is in *accept_model_shas* count as
    ``matching``; all others count as ``stale`` (but are still ingested
    into the replay buffer — one-step-stale data is fine).
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary = _empty_ingest_summary()

    for sp in sorted(inbox_dir.glob("*/*.npz")):
        _process_shard(
            sp,
            inbox_dir=inbox_dir, processed_dir=processed_dir,
            buf=buf, holdout_buf=holdout_buf,
            holdout_frac=holdout_frac, holdout_frozen=holdout_frozen,
            accepted_model_shas=accept_model_shas, rng=rng, summary=summary,
        )

    return summary
