from __future__ import annotations

import json
import logging
import os
import re
import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    UPLOAD_TAR_SUFFIX,
    ShardMeta,
    arrays_to_samples,
    delete_shard_path,
    extract_uploaded_shard_tar,
    samples_to_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.utils.atomic import atomic_write_text
from chess_anti_engine.utils.versioning import version_lt


@dataclass
class _BufferedUploadAccumulator:
    trial_id: str | None
    model_sha256: str
    created_at_unix: float
    last_update_unix: float
    samples: list[ReplaySample] = field(default_factory=list)
    games: int = 0
    positions: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    total_game_plies: int = 0
    adjudicated_games: int = 0
    tb_adjudicated_games: int = 0
    total_draw_games: int = 0
    selfplay_games: int = 0
    selfplay_adjudicated_games: int = 0
    selfplay_draw_games: int = 0
    curriculum_games: int = 0
    curriculum_adjudicated_games: int = 0
    curriculum_draw_games: int = 0
    plies_win: int = 0
    plies_draw: int = 0
    plies_loss: int = 0
    checkmate_games: int = 0
    stalemate_games: int = 0
    model_step: int | None = None

    def add_upload(
        self,
        *,
        samples: list[ReplaySample],
        meta: dict[str, Any],
        now_unix: float,
    ) -> None:
        self.samples.extend(samples)
        self.positions += len(samples)
        self.games += int(meta.get("games") or 0)
        self.wins += int(meta.get("wins") or 0)
        self.draws += int(meta.get("draws") or 0)
        self.losses += int(meta.get("losses") or 0)
        self.total_game_plies += int(meta.get("total_game_plies") or 0)
        self.adjudicated_games += int(meta.get("adjudicated_games") or 0)
        self.tb_adjudicated_games += int(meta.get("tb_adjudicated_games") or 0)
        self.total_draw_games += int(meta.get("total_draw_games") or 0)
        self.selfplay_games += int(meta.get("selfplay_games") or 0)
        self.selfplay_adjudicated_games += int(meta.get("selfplay_adjudicated_games") or 0)
        self.selfplay_draw_games += int(meta.get("selfplay_draw_games") or 0)
        self.curriculum_games += int(meta.get("curriculum_games") or 0)
        self.curriculum_adjudicated_games += int(meta.get("curriculum_adjudicated_games") or 0)
        self.curriculum_draw_games += int(meta.get("curriculum_draw_games") or 0)
        self.plies_win += int(meta.get("plies_win") or 0)
        self.plies_draw += int(meta.get("plies_draw") or 0)
        self.plies_loss += int(meta.get("plies_loss") or 0)
        self.checkmate_games += int(meta.get("checkmate_games") or 0)
        self.stalemate_games += int(meta.get("stalemate_games") or 0)
        _step_raw = meta.get("model_step")
        if _step_raw is not None:
            self.model_step = int(_step_raw)
        self.last_update_unix = float(now_unix)


def _buffered_upload_ready(
    *,
    acc: _BufferedUploadAccumulator,
    now_unix: float,
    target_positions: int,
    max_age_s: float,
) -> bool:
    if acc.positions <= 0 or not acc.samples:
        return False
    if int(target_positions) > 0 and int(acc.positions) >= int(target_positions):
        return True
    if float(max_age_s) > 0.0 and (float(now_unix) - float(acc.created_at_unix)) >= float(max_age_s):
        return True
    return False


def _flush_buffered_upload_to_inbox(
    *,
    inbox_root: Path,
    acc: _BufferedUploadAccumulator,
    now_unix: float,
) -> Path | None:
    if acc.positions <= 0 or not acc.samples:
        return None
    compacted_dir = inbox_root / "_compacted"
    compacted_dir.mkdir(parents=True, exist_ok=True)
    samples = list(acc.samples)
    meta = ShardMeta(
        username="server_compactor",
        generated_at_unix=int(now_unix),
        model_sha256=str(acc.model_sha256) or None,
        model_step=acc.model_step,
        games=int(acc.games),
        positions=int(acc.positions),
        wins=int(acc.wins),
        draws=int(acc.draws),
        losses=int(acc.losses),
        total_game_plies=int(acc.total_game_plies),
        adjudicated_games=int(acc.adjudicated_games),
        total_draw_games=int(acc.total_draw_games),
        selfplay_games=int(acc.selfplay_games),
        selfplay_adjudicated_games=int(acc.selfplay_adjudicated_games),
        selfplay_draw_games=int(acc.selfplay_draw_games),
        curriculum_games=int(acc.curriculum_games),
        curriculum_adjudicated_games=int(acc.curriculum_adjudicated_games),
        curriculum_draw_games=int(acc.curriculum_draw_games),
        plies_win=int(acc.plies_win),
        plies_draw=int(acc.plies_draw),
        plies_loss=int(acc.plies_loss),
        checkmate_games=int(acc.checkmate_games),
        stalemate_games=int(acc.stalemate_games),
    )
    final = compacted_dir / (
        f"{int(now_unix)}_{str(acc.model_sha256)[:8]}_{int(acc.games)}g_{int(acc.positions)}p_"
        f"{secrets.token_hex(8)}{LOCAL_SHARD_SUFFIX}"
    )
    compacted_dir.mkdir(parents=True, exist_ok=True)
    arrs = samples_to_arrays(samples)
    save_local_shard_arrays(final, arrs=arrs, meta=meta)
    return final


def create_app(
    *,
    server_root: str | Path = "server",
    publish_dir: str = "publish",
    inbox_dir: str = "inbox",
    quarantine_dir: str = "quarantine",
    users_db: str = "users.json",
    opening_book_path: str | None = None,
    opening_book_path_2: str | None = None,
    max_upload_mb: int = 256,
    min_workers_per_trial: int = 1,
    max_worker_delta_per_rebalance: int = 1,
    upload_compact_shard_size: int = 2000,
    upload_compact_max_age_seconds: float = 90.0,
):
    """Create the HTTP server.

    Layout under server_root:
    - publish/manifest.json
    - publish/latest_model.pt
    - inbox/<username>/<sha256>.zarr
    - users.json
    """

    try:
        from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, UploadFile
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.security import HTTPBasic, HTTPBasicCredentials

        # Important: this module uses `from __future__ import annotations`, so FastAPI/Pydantic
        # will resolve annotations (e.g. UploadFile) via the *module* globals, not function locals.
        # Export these types into globals() so file upload endpoints work under Pydantic v2.
        globals()["UploadFile"] = UploadFile
        globals()["HTTPBasicCredentials"] = HTTPBasicCredentials
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FastAPI server requires optional dependencies. Install with: pip install -e '.[server]'"
        ) from e

    from chess_anti_engine.replay.shard import load_shard_arrays

    from .auth import load_users, record_upload, save_users, verify_password
    from .lease import (
        assign_trial_lease,
        available_trial_ids,
        load_lease,
        normalize_trial_id,
    )

    root = Path(server_root)
    pub = root / publish_dir
    inbox = root / inbox_dir
    quarantine = root / quarantine_dir
    arena_inbox = root / "arena_inbox"
    users_path = root / users_db

    inbox.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)
    arena_inbox.mkdir(parents=True, exist_ok=True)
    pub.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("chess_anti_engine.server")
    leases_root = root / "leases"
    stats_path = root / "worker_throughput_by_gpu.json"
    trial_stats_path = root / "trial_throughput_by_trial.json"
    leases_root.mkdir(parents=True, exist_ok=True)
    compact_target_positions = max(1, int(upload_compact_shard_size))
    compact_max_age_seconds = max(1.0, float(upload_compact_max_age_seconds))
    upload_accumulators: dict[tuple[str | None, str], _BufferedUploadAccumulator] = {}
    recent_upload_shas: dict[tuple[str | None, str], float] = {}
    upload_lock = threading.Lock()

    app = FastAPI(title="chess-anti-engine server", version="0.1")

    _trial_id_re = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

    def _normalize_trial_id(trial_id: str | None) -> str | None:
        tid = normalize_trial_id(trial_id)
        if tid is None:
            return None
        if not _trial_id_re.fullmatch(tid):
            raise HTTPException(status_code=400, detail="invalid trial_id")
        return tid

    def _trial_root(trial_id: str | None) -> Path:
        tid = _normalize_trial_id(trial_id)
        return root if tid is None else (root / "trials" / tid)

    def _publish_root(trial_id: str | None) -> Path:
        tid = _normalize_trial_id(trial_id)
        return pub if tid is None else (_trial_root(tid) / publish_dir)

    def _inbox_root(trial_id: str | None) -> Path:
        tid = _normalize_trial_id(trial_id)
        return inbox if tid is None else (_trial_root(tid) / inbox_dir)

    def _quarantine_root(trial_id: str | None) -> Path:
        tid = _normalize_trial_id(trial_id)
        return quarantine if tid is None else (_trial_root(tid) / quarantine_dir)

    def _arena_inbox_root(trial_id: str | None) -> Path:
        tid = _normalize_trial_id(trial_id)
        return arena_inbox if tid is None else (_trial_root(tid) / "arena_inbox")

    def _try_flush_and_pop(acc_key, *, inbox_root: Path, acc, now_unix: float) -> bool:
        """Flush an accumulator to disk; on success pop it, on failure leave it.

        Transactional: dedup state has already recorded the uploaded shas, so
        dropping the accumulator on a transient write error would silently lose
        replay samples the worker can't resend. Returning False keeps the acc
        in memory for the next upload or periodic tick to retry.
        """
        try:
            _flush_buffered_upload_to_inbox(
                inbox_root=inbox_root,
                acc=acc,
                now_unix=now_unix,
            )
        except Exception:
            log.exception(
                "compaction flush failed for key=%r; keeping accumulator (%d samples) for retry",
                acc_key, int(getattr(acc, "positions", 0)),
            )
            return False
        upload_accumulators.pop(acc_key, None)
        return True

    def _flush_ready_upload_accumulators(
        *,
        trial_id: str | None = None,
        force_age: bool = True,
        force_all: bool = False,
    ) -> int:
        now_unix = time.time()
        flushed = 0
        normalized_trial_id = _normalize_trial_id(trial_id)
        with upload_lock:
            stale_seen = [
                key
                for key, seen_at in recent_upload_shas.items()
                if (now_unix - float(seen_at)) > 6.0 * 3600.0
            ]
            for key in stale_seen:
                recent_upload_shas.pop(key, None)

            ready_keys: list[tuple[str | None, str]] = []
            for key, acc in upload_accumulators.items():
                trial_key, _model_sha = key
                if normalized_trial_id is not None and trial_key != normalized_trial_id:
                    continue
                if force_all:
                    ready_keys.append(key)
                    continue
                if not force_age:
                    continue
                if _buffered_upload_ready(
                    acc=acc,
                    now_unix=now_unix,
                    target_positions=compact_target_positions + 1,
                    max_age_s=compact_max_age_seconds,
                ):
                    ready_keys.append(key)
            for key in ready_keys:
                acc = upload_accumulators.get(key)
                if acc is None:
                    continue
                if _try_flush_and_pop(key, inbox_root=_inbox_root(acc.trial_id), acc=acc, now_unix=now_unix):
                    flushed += 1
        return flushed

    def _load_manifest(trial_id: str | None = None) -> dict[str, Any] | None:
        mf = _publish_root(trial_id) / "manifest.json"
        if not mf.exists():
            return None
        try:
            return dict(json.loads(mf.read_text(encoding="utf-8")))
        except Exception:
            return None

    class _LeaseAssignLock:
        def __init__(self, path: Path, *, timeout_s: float = 10.0) -> None:
            self.path = path
            self.timeout_s = float(timeout_s)
            self._held = False

        def __enter__(self) -> _LeaseAssignLock:
            deadline = time.time() + float(self.timeout_s)
            while True:
                try:
                    fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(f"{os.getpid()}\n")
                    self._held = True
                    return self
                except FileExistsError:
                    if time.time() >= deadline:
                        try:
                            self.path.unlink(missing_ok=True)
                        except Exception:
                            pass
                    time.sleep(0.05)

        def __exit__(self, exc_type, exc, tb) -> None:
            if self._held:
                try:
                    self.path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _load_json_stats(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _primary_gpu_model(*, lease: dict[str, Any] | None) -> str:
        if not isinstance(lease, dict):
            return "cpu"
        worker_info = lease.get("worker_info")
        if not isinstance(worker_info, dict):
            return "cpu"
        gpu_models = worker_info.get("gpu_models")
        if isinstance(gpu_models, list):
            for item in gpu_models:
                model = str(item).strip()
                if model:
                    return model
        device = str(worker_info.get("device") or "").strip().lower()
        return "cpu" if not device else device

    def _record_gpu_throughput(
        *,
        lease: dict[str, Any] | None,
        trial_id: str | None,
        positions: int,
        games: int,
        elapsed_s: float | None,
    ) -> None:
        if elapsed_s is None or elapsed_s <= 0.0:
            return
        gpu_model = _primary_gpu_model(lease=lease)
        now_unix = int(time.time())
        stats = _load_json_stats(stats_path)
        entry = stats.get(gpu_model)
        if not isinstance(entry, dict):
            entry = {}
        entry["gpu_model"] = gpu_model
        entry["samples"] = int(entry.get("samples", 0)) + 1
        entry["total_positions"] = int(entry.get("total_positions", 0)) + int(positions)
        entry["total_games"] = int(entry.get("total_games", 0)) + int(games)
        entry["total_elapsed_s"] = float(entry.get("total_elapsed_s", 0.0)) + float(elapsed_s)
        total_elapsed_s = max(1e-9, float(entry["total_elapsed_s"]))
        entry["avg_positions_per_s"] = float(entry["total_positions"]) / total_elapsed_s
        entry["avg_games_per_s"] = float(entry["total_games"]) / total_elapsed_s
        entry["last_trial_id"] = _normalize_trial_id(trial_id)
        if isinstance(lease, dict):
            worker_info = lease.get("worker_info")
            if isinstance(worker_info, dict):
                hostname = str(worker_info.get("hostname") or "").strip()
                if hostname:
                    entry["last_hostname"] = hostname
                cpu_count = worker_info.get("cpu_count")
                if cpu_count is not None:
                    try:
                        entry["last_cpu_count"] = int(cpu_count)
                    except Exception:
                        pass
        entry["last_updated_unix"] = now_unix
        stats[gpu_model] = entry
        atomic_write_text(stats_path, json.dumps(stats, indent=2, sort_keys=True))

    def _record_trial_throughput(
        *,
        trial_id: str | None,
        positions: int,
        games: int,
        elapsed_s: float | None,
    ) -> None:
        tid = _normalize_trial_id(trial_id)
        if tid is None or elapsed_s is None or elapsed_s <= 0.0:
            return
        stats = _load_json_stats(trial_stats_path)
        entry = stats.get(tid)
        if not isinstance(entry, dict):
            entry = {}
        now_unix = int(time.time())
        entry["trial_id"] = tid
        entry["samples"] = int(entry.get("samples", 0)) + 1
        entry["total_positions"] = int(entry.get("total_positions", 0)) + int(positions)
        entry["total_games"] = int(entry.get("total_games", 0)) + int(games)
        entry["total_elapsed_s"] = float(entry.get("total_elapsed_s", 0.0)) + float(elapsed_s)
        total_elapsed_s = max(1e-9, float(entry["total_elapsed_s"]))
        batch_positions_per_s = float(positions) / max(1e-9, float(elapsed_s))
        batch_games_per_s = float(games) / max(1e-9, float(elapsed_s))
        alpha = 0.30
        prev_pos = float(entry.get("ema_positions_per_s", batch_positions_per_s) or batch_positions_per_s)
        prev_games = float(entry.get("ema_games_per_s", batch_games_per_s) or batch_games_per_s)
        entry["ema_positions_per_s"] = (1.0 - alpha) * prev_pos + alpha * batch_positions_per_s
        entry["ema_games_per_s"] = (1.0 - alpha) * prev_games + alpha * batch_games_per_s
        entry["avg_positions_per_s"] = float(entry["total_positions"]) / total_elapsed_s
        entry["avg_games_per_s"] = float(entry["total_games"]) / total_elapsed_s
        entry["last_updated_unix"] = now_unix
        stats[tid] = entry
        atomic_write_text(trial_stats_path, json.dumps(stats, indent=2, sort_keys=True))

    def _check_worker_compat(
        *,
        trial_id: str | None = None,
        worker_version: str | None,
        worker_protocol: str | None,
    ) -> tuple[bool, str]:
        """Check whether a worker is allowed to participate.

        This is intentionally driven by the learner-published manifest so the learner
        can upgrade protocol requirements without server CLI changes.
        """
        mf = _load_manifest(trial_id)
        if mf is None:
            return True, ""

        min_v = mf.get("min_worker_version")
        req_proto = mf.get("protocol_version")

        # Backward-compat: if fields are missing, don't enforce.
        if min_v is None and req_proto is None:
            return True, ""

        wv = str(worker_version or "0.0.0")
        wp = str(worker_protocol or "0")

        if req_proto is not None:
            try:
                req_p = int(req_proto)
                got_p = int(wp)
            except Exception:
                return False, f"bad protocol version header (got {wp!r})"
            if got_p != req_p:
                return False, f"protocol mismatch: worker={got_p} required={req_p}"

        if min_v is not None:
            try:
                if version_lt(wv, str(min_v)):
                    return False, f"worker too old: worker={wv} min_required={min_v}"
            except Exception:
                return False, f"bad worker version header (got {wv!r})"

        return True, ""

    basic = HTTPBasic()

    def _auth_user(creds: HTTPBasicCredentials = Depends(basic)) -> str:
        users = load_users(users_path)
        rec = users.get(str(creds.username))
        if rec is None:
            raise HTTPException(status_code=401, detail="unknown user")
        if bool(rec.disabled):
            raise HTTPException(status_code=403, detail="user disabled")
        if not verify_password(str(creds.password), rec):
            raise HTTPException(status_code=401, detail="bad password")
        return str(creds.username)

    def _get_manifest_impl(
        trial_id: str | None,
        *,
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        _flush_ready_upload_accumulators(trial_id=trial_id, force_age=True, force_all=False)
        mf = _publish_root(trial_id) / "manifest.json"
        if not mf.exists():
            raise HTTPException(status_code=404, detail="manifest not published yet")

        ok, reason = _check_worker_compat(
            trial_id=trial_id,
            worker_version=x_cae_worker_version,
            worker_protocol=x_cae_protocol_version,
        )
        if not ok:
            # 426 (Upgrade Required) communicates "update your client".
            raise HTTPException(status_code=426, detail=reason)

        return JSONResponse(content=json.loads(mf.read_text(encoding="utf-8")))

    @app.get("/v1/manifest")
    def get_manifest(
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        return _get_manifest_impl(
            None,
            x_cae_worker_version=x_cae_worker_version,
            x_cae_protocol_version=x_cae_protocol_version,
        )

    @app.get("/v1/trials/{trial_id}/manifest")
    def get_trial_manifest(
        trial_id: str,
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        return _get_manifest_impl(
            trial_id,
            x_cae_worker_version=x_cae_worker_version,
            x_cae_protocol_version=x_cae_protocol_version,
        )

    def _get_model_impl(trial_id: str | None) -> Any:
        mp = _publish_root(trial_id) / "latest_model.pt"
        if not mp.exists():
            raise HTTPException(status_code=404, detail="model not published yet")
        return FileResponse(str(mp), media_type="application/octet-stream", filename="latest_model.pt")

    @app.get("/v1/model")
    def get_model() -> Any:
        return _get_model_impl(None)

    @app.get("/v1/trials/{trial_id}/model")
    def get_trial_model(trial_id: str) -> Any:
        return _get_model_impl(trial_id)

    def _get_best_model_impl(trial_id: str | None) -> Any:
        mp = _publish_root(trial_id) / "best_model.pt"
        if not mp.exists():
            raise HTTPException(status_code=404, detail="best model not published yet")
        return FileResponse(str(mp), media_type="application/octet-stream", filename="best_model.pt")

    @app.get("/v1/best_model")
    def get_best_model() -> Any:
        return _get_best_model_impl(None)

    @app.get("/v1/trials/{trial_id}/best_model")
    def get_trial_best_model(trial_id: str) -> Any:
        return _get_best_model_impl(trial_id)

    @app.get("/v1/opening_book")
    def get_opening_book() -> Any:
        if opening_book_path is None:
            raise HTTPException(status_code=404, detail="no opening book configured")
        p = Path(opening_book_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail="opening book not found")
        return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

    @app.get("/v1/opening_book_2")
    def get_opening_book_2() -> Any:
        if opening_book_path_2 is None:
            raise HTTPException(status_code=404, detail="no opening book 2 configured")
        p = Path(opening_book_path_2)
        if not p.exists():
            raise HTTPException(status_code=404, detail="opening book 2 not found")
        return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

    def _artifact_from_publish(key: str, *, default_name: str, trial_id: str | None = None) -> Path | None:
        mf = _load_manifest(trial_id) or {}
        rec = mf.get(key)
        if isinstance(rec, dict) and rec.get("filename"):
            name = str(rec.get("filename"))
        else:
            name = str(default_name)
        p = _publish_root(trial_id) / name
        if p.exists() and p.is_file():
            return p
        return None

    @app.get("/v1/stockfish")
    def get_stockfish() -> Any:
        p = _artifact_from_publish("stockfish", default_name="stockfish")
        if p is None:
            raise HTTPException(status_code=404, detail="stockfish not published")
        return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

    @app.get("/v1/trials/{trial_id}/stockfish")
    def get_trial_stockfish(trial_id: str) -> Any:
        p = _artifact_from_publish("stockfish", default_name="stockfish", trial_id=trial_id)
        if p is None:
            raise HTTPException(status_code=404, detail="stockfish not published")
        return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

    def _get_update_info_impl(trial_id: str | None) -> Any:
        """Minimal compatibility/update metadata.

        This endpoint intentionally does NOT enforce worker compatibility so an out-of-date
        worker can still learn how to update itself.
        """
        mf = _load_manifest(trial_id) or {}
        out: dict[str, Any] = {
            "server_version": mf.get("server_version"),
            "protocol_version": mf.get("protocol_version"),
            "min_worker_version": mf.get("min_worker_version"),
        }
        if isinstance(mf.get("worker_wheel"), dict):
            out["worker_wheel"] = mf.get("worker_wheel")
        return JSONResponse(content=out)

    @app.get("/v1/update_info")
    def get_update_info() -> Any:
        return _get_update_info_impl(None)

    @app.get("/v1/trials/{trial_id}/update_info")
    def get_trial_update_info(trial_id: str) -> Any:
        return _get_update_info_impl(trial_id)

    @app.get("/v1/worker_throughput")
    def get_worker_throughput() -> Any:
        return JSONResponse(content=_load_json_stats(stats_path))

    @app.get("/v1/trial_throughput")
    def get_trial_throughput() -> Any:
        return JSONResponse(content=_load_json_stats(trial_stats_path))

    def _get_worker_wheel_impl(trial_id: str | None) -> Any:
        p = _artifact_from_publish("worker_wheel", default_name="worker.whl", trial_id=trial_id)
        if p is None:
            raise HTTPException(status_code=404, detail="worker wheel not published")
        return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

    @app.get("/v1/worker_wheel")
    def get_worker_wheel() -> Any:
        return _get_worker_wheel_impl(None)

    @app.get("/v1/trials/{trial_id}/worker_wheel")
    def get_trial_worker_wheel(trial_id: str) -> Any:
        return _get_worker_wheel_impl(trial_id)

    @app.post("/v1/lease_trial")
    def lease_trial(
        payload: dict[str, Any] = Body(default_factory=dict),
        username: str = Depends(_auth_user),
    ) -> Any:
        with _LeaseAssignLock(leases_root / ".assign.lock"):
            lease_seconds = 3600
            requested_lease_id = str(payload.get("lease_id") or "").strip()
            requested_trial_id = _normalize_trial_id(payload.get("trial_id"))
            worker_info = payload.get("worker_info")
            if not isinstance(worker_info, dict):
                worker_info = {}
            available_trials = available_trial_ids(server_root=root, publish_dir=publish_dir)
            if not available_trials and _load_manifest(None) is None:
                raise HTTPException(status_code=503, detail="no published trials available")
            lease = assign_trial_lease(
                leases_root=leases_root,
                username=str(username),
                worker_info=worker_info,
                available_trials=available_trials,
                manifest_loader=_load_manifest,
                trial_throughput_loader=lambda tid: _load_json_stats(trial_stats_path).get(str(tid), {}),
                requested_lease_id=requested_lease_id,
                requested_trial_id=requested_trial_id,
                lease_seconds=lease_seconds,
                min_workers_per_trial=int(min_workers_per_trial),
                max_worker_delta_per_rebalance=int(max_worker_delta_per_rebalance),
            )
            return {
                "lease_id": str(lease.get("lease_id")),
                "trial_id": lease.get("trial_id"),
                "api_prefix": str(lease.get("api_prefix") or "/v1"),
                "lease_seconds": lease_seconds,
                "expires_at_unix": int(lease.get("expires_at_unix") or 0),
            }

    async def _upload_shard_impl(
        trial_id: str | None,
        *,
        file: UploadFile = File(...),
        username: str = Depends(_auth_user),
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
        x_cae_worker_lease_id: str | None = Header(None, alias="X-CAE-Worker-Lease-ID"),
        x_cae_batch_elapsed_s: str | None = Header(None, alias="X-CAE-Batch-Elapsed-S"),
        x_cae_machine_id: str | None = Header(None, alias="X-CAE-Machine-ID"),
    ) -> Any:
        ok, reason = _check_worker_compat(
            trial_id=trial_id,
            worker_version=x_cae_worker_version,
            worker_protocol=x_cae_protocol_version,
        )
        if not ok:
            log.warning("rejecting shard upload from user=%s: %s", username, reason)
            return {"stored": False, "rejected": True, "reason": reason}
        # Size guard: FastAPI doesn't enforce this automatically.
        max_bytes = int(max_upload_mb) * 1024 * 1024
        inbox_root = _inbox_root(trial_id)
        quarantine_root = _quarantine_root(trial_id)
        inbox_root.mkdir(parents=True, exist_ok=True)
        quarantine_root.mkdir(parents=True, exist_ok=True)

        upload_name = str(file.filename or "")
        if not upload_name.endswith(UPLOAD_TAR_SUFFIX):
            # Workers only produce zarr-tar uploads; reject anything else so
            # stale clients fail fast rather than writing a partial file.
            return {
                "stored": False,
                "rejected": True,
                "reason": f"unsupported upload suffix: expected {UPLOAD_TAR_SUFFIX}",
            }
        tmp = inbox_root / f"tmp_{os.getpid()}_{secrets.token_hex(8)}{UPLOAD_TAR_SUFFIX}"
        # Stream upload to disk and hash it in the same pass; rehashing the
        # file after validation would re-read the full shard from disk.
        h = hashlib.sha256()
        n = 0
        with tmp.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                n += len(chunk)
                if n > max_bytes:
                    tmp.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="upload too large")
                h.update(chunk)
                f.write(chunk)
        sha = h.hexdigest()

        tmp_zarr: Path | None = None
        try:
            tmp_zarr = inbox_root / f"tmp_{os.getpid()}_{secrets.token_hex(8)}{LOCAL_SHARD_SUFFIX}"
            zarr_root = extract_uploaded_shard_tar(tmp, tmp_zarr)
            shard_arrs, meta = load_shard_arrays(zarr_root)
        except Exception as e:
            # Quarantine so we can inspect bad uploads without causing worker retry storms.
            qdir = quarantine_root / "invalid"
            qdir.mkdir(parents=True, exist_ok=True)
            qpath = qdir / tmp.name
            try:
                tmp.replace(qpath)
                (qpath.with_suffix(qpath.suffix + ".reason.txt")).write_text(
                    f"{type(e).__name__}: {e}", encoding="utf-8"
                )
            except Exception:
                tmp.unlink(missing_ok=True)
            if tmp_zarr is not None:
                delete_shard_path(tmp_zarr)

            return {
                "stored": False,
                "rejected": True,
                "reason": f"invalid shard: {type(e).__name__}: {e}",
            }

        positions = int(shard_arrs["x"].shape[0])
        tmp.unlink(missing_ok=True)
        if tmp_zarr is not None:
            delete_shard_path(tmp_zarr)
        trial_key = _normalize_trial_id(trial_id)
        upload_seen_key = (trial_key, sha)
        now_unix = time.time()
        stored = False
        with upload_lock:
            if upload_seen_key not in recent_upload_shas:
                model_sha = str(meta.get("model_sha256") or sha)
                acc_key = (trial_key, model_sha)
                acc = upload_accumulators.get(acc_key)
                if acc is None:
                    acc = _BufferedUploadAccumulator(
                        trial_id=trial_key,
                        model_sha256=model_sha,
                        created_at_unix=now_unix,
                        last_update_unix=now_unix,
                    )
                    upload_accumulators[acc_key] = acc
                acc.add_upload(
                    samples=arrays_to_samples(shard_arrs),
                    meta=meta,
                    now_unix=now_unix,
                )
                recent_upload_shas[upload_seen_key] = now_unix
                stored = True
                if _buffered_upload_ready(
                    acc=acc,
                    now_unix=now_unix,
                    target_positions=compact_target_positions,
                    max_age_s=compact_max_age_seconds,
                ):
                    _try_flush_and_pop(acc_key, inbox_root=inbox_root, acc=acc, now_unix=now_unix)

        lease = None
        if x_cae_worker_lease_id is not None:
            lease = load_lease(leases_root=leases_root, lease_id=str(x_cae_worker_lease_id).strip())
        batch_elapsed_s: float | None = None
        if x_cae_batch_elapsed_s is not None:
            try:
                batch_elapsed_s = float(x_cae_batch_elapsed_s)
            except Exception:
                batch_elapsed_s = None

        _record_gpu_throughput(
            lease=lease,
            trial_id=trial_id,
            positions=int(positions),
            games=int(meta.get("games") or 0),
            elapsed_s=batch_elapsed_s,
        )
        _record_trial_throughput(
            trial_id=trial_id,
            positions=int(positions),
            games=int(meta.get("games") or 0),
            elapsed_s=batch_elapsed_s,
        )

        # Update user stats.
        try:
            users = load_users(users_path)
            machine_id = str(x_cae_machine_id).strip() if x_cae_machine_id else None
            record_upload(users, username=username, bytes_uploaded=int(n), positions=positions, machine_id=machine_id)
            save_users(users_path, users)
        except Exception:
            # Stats failure should not fail the upload.
            pass

        return {
            "stored": bool(stored),
            "trial_id": _normalize_trial_id(trial_id),
            "sha256": sha,
            "bytes": int(n),
            "positions": int(positions),
            "meta": meta,
        }

    @app.post("/v1/upload_shard")
    async def upload_shard(
        file: UploadFile = File(...),
        username: str = Depends(_auth_user),
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
        x_cae_worker_lease_id: str | None = Header(None, alias="X-CAE-Worker-Lease-ID"),
        x_cae_batch_elapsed_s: str | None = Header(None, alias="X-CAE-Batch-Elapsed-S"),
        x_cae_machine_id: str | None = Header(None, alias="X-CAE-Machine-ID"),
    ) -> Any:
        return await _upload_shard_impl(
            None,
            file=file,
            username=username,
            x_cae_worker_version=x_cae_worker_version,
            x_cae_protocol_version=x_cae_protocol_version,
            x_cae_worker_lease_id=x_cae_worker_lease_id,
            x_cae_batch_elapsed_s=x_cae_batch_elapsed_s,
            x_cae_machine_id=x_cae_machine_id,
        )

    @app.post("/v1/trials/{trial_id}/upload_shard")
    async def upload_trial_shard(
        trial_id: str,
        file: UploadFile = File(...),
        username: str = Depends(_auth_user),
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
        x_cae_worker_lease_id: str | None = Header(None, alias="X-CAE-Worker-Lease-ID"),
        x_cae_batch_elapsed_s: str | None = Header(None, alias="X-CAE-Batch-Elapsed-S"),
        x_cae_machine_id: str | None = Header(None, alias="X-CAE-Machine-ID"),
    ) -> Any:
        return await _upload_shard_impl(
            trial_id,
            file=file,
            username=username,
            x_cae_worker_version=x_cae_worker_version,
            x_cae_protocol_version=x_cae_protocol_version,
            x_cae_worker_lease_id=x_cae_worker_lease_id,
            x_cae_batch_elapsed_s=x_cae_batch_elapsed_s,
            x_cae_machine_id=x_cae_machine_id,
        )

    async def _upload_arena_result_impl(
        trial_id: str | None,
        *,
        payload: dict[str, Any] = Body(...),
        username: str = Depends(_auth_user),
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        ok, reason = _check_worker_compat(
            trial_id=trial_id,
            worker_version=x_cae_worker_version,
            worker_protocol=x_cae_protocol_version,
        )
        if not ok:
            log.warning("rejecting arena upload from user=%s: %s", username, reason)
            return {"stored": False, "rejected": True, "reason": reason}
        # Basic schema validation
        def _req_int(k: str) -> int:
            if k not in payload:
                raise HTTPException(status_code=400, detail=f"missing field {k}")
            try:
                return int(payload[k])
            except Exception:
                raise HTTPException(status_code=400, detail=f"bad int field {k}")

        def _req_str(k: str) -> str:
            if k not in payload:
                raise HTTPException(status_code=400, detail=f"missing field {k}")
            v = str(payload[k])
            if not v:
                raise HTTPException(status_code=400, detail=f"empty field {k}")
            return v

        games = _req_int("games")
        a_win = _req_int("a_win")
        a_draw = _req_int("a_draw")
        a_loss = _req_int("a_loss")
        if a_win + a_draw + a_loss != games:
            raise HTTPException(status_code=400, detail="W/D/L must sum to games")

        a_sha = _req_str("a_sha256")
        b_sha = _req_str("b_sha256")
        ts = int(payload.get("generated_at_unix") or 0)

        # Store under arena_inbox/<username>/
        arena_root = _arena_inbox_root(trial_id)
        user_dir = arena_root / username
        user_dir.mkdir(parents=True, exist_ok=True)

        body = json.dumps(payload, sort_keys=True).encode("utf-8")

        sha = hashlib.sha256(body).hexdigest()
        out = user_dir / f"{sha}.json"
        if not out.exists():
            out.write_bytes(body)

        return {
            "stored": True,
            "trial_id": _normalize_trial_id(trial_id),
            "sha256": sha,
            "username": username,
            "games": int(games),
            "a_sha256": a_sha,
            "b_sha256": b_sha,
            "generated_at_unix": int(ts),
        }

    @app.post("/v1/upload_arena_result")
    async def upload_arena_result(
        payload: dict[str, Any] = Body(...),
        username: str = Depends(_auth_user),
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        return await _upload_arena_result_impl(
            None,
            payload=payload,
            username=username,
            x_cae_worker_version=x_cae_worker_version,
            x_cae_protocol_version=x_cae_protocol_version,
        )

    @app.post("/v1/trials/{trial_id}/upload_arena_result")
    async def upload_trial_arena_result(
        trial_id: str,
        payload: dict[str, Any] = Body(...),
        username: str = Depends(_auth_user),
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        return await _upload_arena_result_impl(
            trial_id,
            payload=payload,
            username=username,
            x_cae_worker_version=x_cae_worker_version,
            x_cae_protocol_version=x_cae_protocol_version,
        )

    @app.on_event("shutdown")
    def _flush_upload_accumulators_on_shutdown() -> None:
        _flush_ready_upload_accumulators(force_age=False, force_all=True)

    return app
