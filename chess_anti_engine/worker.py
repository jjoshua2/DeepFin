from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.inference import SlotInferenceClient, ShmBatchInferenceClient
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import ShardMeta, save_npz
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.match import play_match_batch
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish import StockfishPool, StockfishUCI
from chess_anti_engine.utils.versioning import version_lt
from chess_anti_engine.version import PACKAGE_NAME, PACKAGE_VERSION, PROTOCOL_VERSION
from chess_anti_engine.worker_config import load_worker_config, save_worker_config


def tune_games_per_batch(
    *,
    current: int,
    elapsed_s: float,
    target_s: float,
    min_games: int,
    max_games: int,
    max_change_factor: float = 1.5,
) -> int:
    """Heuristic auto-tune for worker throughput.

    Adjusts games_per_batch so batches trend toward `target_s` wall-clock time.

    This keeps slow clients from taking excessively long batches, and allows fast
    clients to scale up without manual flags.
    """
    cur = max(1, int(current))
    if elapsed_s <= 0:
        return cur

    tgt = max(1e-6, float(target_s))
    ratio = tgt / float(elapsed_s)
    # Clamp adjustment magnitude to avoid oscillations.
    ratio = max(1.0 / float(max_change_factor), min(float(max_change_factor), float(ratio)))

    nxt = int(round(float(cur) * float(ratio)))
    nxt = max(int(min_games), min(int(max_games), int(nxt)))
    return max(1, nxt)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class _BufferedUpload:
    samples: list[ReplaySample] = field(default_factory=list)
    model_sha: str | None = None
    model_step: int | None = None
    games: int = 0
    positions: int = 0
    w: int = 0
    d: int = 0
    l: int = 0
    total_game_plies: int = 0
    adjudicated_games: int = 0
    total_draw_games: int = 0
    selfplay_games: int = 0
    selfplay_adjudicated_games: int = 0
    selfplay_draw_games: int = 0
    curriculum_games: int = 0
    curriculum_adjudicated_games: int = 0
    curriculum_draw_games: int = 0
    first_buffered_at_s: float | None = None

    def reset(self) -> None:
        self.samples.clear()
        self.model_sha = None
        self.model_step = None
        self.games = 0
        self.positions = 0
        self.w = 0
        self.d = 0
        self.l = 0
        self.total_game_plies = 0
        self.adjudicated_games = 0
        self.total_draw_games = 0
        self.selfplay_games = 0
        self.selfplay_adjudicated_games = 0
        self.selfplay_draw_games = 0
        self.curriculum_games = 0
        self.curriculum_adjudicated_games = 0
        self.curriculum_draw_games = 0
        self.first_buffered_at_s = None


def _buffer_elapsed_s(*, buf: _BufferedUpload, now_s: float) -> float:
    if buf.first_buffered_at_s is None:
        return 0.0
    return max(0.0, float(now_s) - float(buf.first_buffered_at_s))


def _buffer_should_flush(
    *,
    buf: _BufferedUpload,
    now_s: float,
    last_send_s: float,
    target_positions: int,
    flush_seconds: float,
) -> bool:
    if buf.positions <= 0:
        return False
    if int(target_positions) > 0 and int(buf.positions) >= int(target_positions):
        return True
    if float(flush_seconds) > 0.0 and (float(now_s) - float(last_send_s)) >= float(flush_seconds):
        return True
    return False


def _buffer_add_completed_game(
    *,
    buf: _BufferedUpload,
    game_batch,
    now_s: float,
    model_sha: str,
    model_step: int,
) -> None:
    if getattr(game_batch, "positions", 0) <= 0 or not getattr(game_batch, "samples", None):
        return
    if buf.positions > 0:
        if str(buf.model_sha or "") != str(model_sha) or int(buf.model_step or 0) != int(model_step):
            raise ValueError("buffered upload model metadata mismatch")
    else:
        buf.model_sha = str(model_sha)
        buf.model_step = int(model_step)
    if buf.first_buffered_at_s is None:
        buf.first_buffered_at_s = float(now_s)
    buf.samples.extend(list(game_batch.samples))
    buf.games += int(getattr(game_batch, "games", 0))
    buf.positions += int(getattr(game_batch, "positions", 0))
    buf.w += int(getattr(game_batch, "w", 0))
    buf.d += int(getattr(game_batch, "d", 0))
    buf.l += int(getattr(game_batch, "l", 0))
    buf.total_game_plies += int(getattr(game_batch, "total_game_plies", 0))
    buf.adjudicated_games += int(getattr(game_batch, "adjudicated_games", 0))
    buf.total_draw_games += int(getattr(game_batch, "total_draw_games", 0))
    buf.selfplay_games += int(getattr(game_batch, "selfplay_games", 0))
    buf.selfplay_adjudicated_games += int(getattr(game_batch, "selfplay_adjudicated_games", 0))
    buf.selfplay_draw_games += int(getattr(game_batch, "selfplay_draw_games", 0))
    buf.curriculum_games += int(getattr(game_batch, "curriculum_games", 0))
    buf.curriculum_adjudicated_games += int(getattr(game_batch, "curriculum_adjudicated_games", 0))
    buf.curriculum_draw_games += int(getattr(game_batch, "curriculum_draw_games", 0))


def _pending_elapsed_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(shard_path.suffix + ".elapsed_s")


def _flush_upload_buffer_to_pending(
    *,
    pending_dir: Path,
    username: str,
    buf: _BufferedUpload,
    now_s: float,
) -> tuple[Path | None, float]:
    if buf.positions <= 0 or not buf.samples:
        return None, 0.0
    model_sha = str(buf.model_sha or "")
    if not model_sha:
        raise ValueError("buffered upload missing model sha")
    if buf.model_step is None:
        raise ValueError("buffered upload missing model step")
    ts = int(now_s)
    shard_path = pending_dir / f"{ts}_{model_sha[:8]}_{buf.games}g_{buf.positions}p.npz"
    elapsed_s = _buffer_elapsed_s(buf=buf, now_s=now_s)
    meta = ShardMeta(
        username=str(username),
        generated_at_unix=ts,
        model_sha256=str(model_sha),
        model_step=int(buf.model_step),
        games=int(buf.games),
        positions=int(buf.positions),
        wins=int(buf.w),
        draws=int(buf.d),
        losses=int(buf.l),
        total_game_plies=int(buf.total_game_plies),
        adjudicated_games=int(buf.adjudicated_games),
        total_draw_games=int(buf.total_draw_games),
        selfplay_games=int(buf.selfplay_games),
        selfplay_adjudicated_games=int(buf.selfplay_adjudicated_games),
        selfplay_draw_games=int(buf.selfplay_draw_games),
        curriculum_games=int(buf.curriculum_games),
        curriculum_adjudicated_games=int(buf.curriculum_adjudicated_games),
        curriculum_draw_games=int(buf.curriculum_draw_games),
    )
    tmp_path = pending_dir / f"_tmp_{shard_path.name}"
    save_npz(tmp_path, samples=list(buf.samples), meta=meta, compress=False)
    tmp_path.replace(shard_path)
    _pending_elapsed_path(shard_path).write_text(f"{float(elapsed_s):.6f}\n", encoding="utf-8")
    buf.reset()
    return shard_path, float(elapsed_s)


def _maybe_flush_upload_buffer(
    *,
    pending_dir: Path,
    username: str,
    buf: _BufferedUpload,
    now_s: float,
    last_send_s: float,
    target_positions: int,
    flush_seconds: float,
    force: bool = False,
) -> tuple[Path | None, float]:
    if not force and not _buffer_should_flush(
        buf=buf,
        now_s=now_s,
        last_send_s=last_send_s,
        target_positions=target_positions,
        flush_seconds=flush_seconds,
    ):
        return None, 0.0
    return _flush_upload_buffer_to_pending(
        pending_dir=pending_dir,
        username=username,
        buf=buf,
        now_s=now_s,
    )


def _should_write_fallback_batch(
    *,
    shard_path: Path | None,
    samples: list[ReplaySample],
    saw_completed_game: bool,
) -> bool:
    """Return True only when batch-level fallback is actually needed.

    If incremental callbacks already observed completed games, an empty upload
    buffer at batch end usually means those samples were already flushed and
    uploaded earlier. Writing the full batch again would duplicate data.
    """
    return shard_path is None and bool(samples) and not saw_completed_game


def _worker_headers(*, machine_id: str | None = None) -> dict[str, str]:
    headers = {
        "X-CAE-Worker-Version": str(PACKAGE_VERSION),
        "X-CAE-Protocol-Version": str(PROTOCOL_VERSION),
    }
    if machine_id:
        headers["X-CAE-Machine-ID"] = machine_id
    return headers


def _manifest_poll_headers(
    *,
    worker_id: str,
    lease_id: str = "",
    state: str | None = None,
    elapsed_s: float | None = None,
) -> dict[str, str]:
    headers = dict(_worker_headers())
    headers["X-CAE-Worker-ID"] = str(worker_id)
    if str(lease_id).strip():
        headers["X-CAE-Worker-Lease-ID"] = str(lease_id)
    state_text = str(state or "").strip()
    if state_text:
        headers["X-CAE-Worker-State"] = state_text
    if elapsed_s is not None and float(elapsed_s) > 0.0:
        headers["X-CAE-Worker-State-Elapsed-S"] = str(float(elapsed_s))
    return headers


def _collect_worker_info(*, device: str) -> dict[str, object]:
    out: dict[str, object] = {
        "hostname": str(socket.gethostname()),
        "device": str(device),
    }

    try:
        out["cpu_count"] = int(os.cpu_count() or 1)
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            gpu_models: list[str] = []
            for idx in range(int(torch.cuda.device_count())):
                props = torch.cuda.get_device_properties(idx)
                gpu_models.append(str(props.name))
            out["gpu_models"] = gpu_models
        except Exception:
            pass

    return out


def _ensure_executable(path: Path) -> None:
    """Best-effort chmod +x for POSIX systems."""
    try:
        if os.name != "nt":
            st = os.stat(path)
            os.chmod(path, st.st_mode | 0o111)
    except Exception:
        pass


def _pip_install_wheel(wheel_path: Path) -> None:
    # Use a PEP508 direct reference so we can request extras (installs worker deps like requests).
    uri = wheel_path.resolve().as_uri()
    req = f"{PACKAGE_NAME}[worker] @ {uri}"
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", req]
    subprocess.check_call(cmd)


def _restart_process() -> None:
    # Avoid infinite update loops.
    os.environ["CAE_SELF_UPDATED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _download(
    url: str,
    *,
    out_path: Path,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> None:
    try:
        import requests  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("worker requires requests; install with pip install -e '.[worker]' ") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
    tmp.replace(out_path)


def _download_and_verify(
    url: str,
    *,
    out_path: Path,
    expected_sha256: str | None,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> None:
    """Download a file and verify its sha256 if provided.

    If verification fails, we delete and retry once.
    """
    exp = str(expected_sha256 or "")

    def _once() -> None:
        _download(url, out_path=out_path, timeout=timeout, headers=headers)
        if exp:
            got = _sha256_file(out_path)
            if got != exp:
                raise RuntimeError(f"sha256 mismatch for {out_path.name}: got={got} expected={exp}")

    try:
        _once()
    except Exception:
        out_path.unlink(missing_ok=True)
        _once()


def _download_and_verify_shared(
    url: str,
    *,
    out_path: Path,
    expected_sha256: str | None,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    lock_timeout_s: float = 600.0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        if not expected_sha256 or _sha256_file(out_path) == str(expected_sha256):
            return
        out_path.unlink(missing_ok=True)

    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    deadline = time.time() + float(lock_timeout_s)
    have_lock = False

    while not have_lock:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
            have_lock = True
        except FileExistsError:
            if out_path.exists():
                if not expected_sha256 or _sha256_file(out_path) == str(expected_sha256):
                    return
                out_path.unlink(missing_ok=True)
            if time.time() >= deadline:
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:
                    pass
            time.sleep(0.25)

    try:
        if out_path.exists():
            if not expected_sha256 or _sha256_file(out_path) == str(expected_sha256):
                return
            out_path.unlink(missing_ok=True)
        _download_and_verify(
            url,
            out_path=out_path,
            expected_sha256=expected_sha256,
            timeout=timeout,
            headers=headers,
        )
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def _prune_cached_models(*, cache_dir: Path, keep_shas: set[str]) -> None:
    """Delete cached model checkpoints not in keep_shas.

    Files:
    - model_<sha>.pt (downloaded from /v1/model)
    - best_<sha>.pt (downloaded from /v1/best_model)

    This keeps worker disk usage bounded as best/latest advance over time.
    """
    keep = {str(s) for s in keep_shas if str(s)}

    for p in cache_dir.glob("model_*.pt"):
        name = p.name
        if not name.startswith("model_") or not name.endswith(".pt"):
            continue
        sha = name[len("model_") : -len(".pt")]
        if sha and sha not in keep:
            p.unlink(missing_ok=True)

    for p in cache_dir.glob("best_*.pt"):
        name = p.name
        if not name.startswith("best_") or not name.endswith(".pt"):
            continue
        sha = name[len("best_") : -len(".pt")]
        if sha and sha not in keep:
            p.unlink(missing_ok=True)


def _maybe_compile_inference_model(model: torch.nn.Module, *, device: str) -> torch.nn.Module:
    if not str(device).startswith("cuda"):
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        return model


def _configure_shared_compile_cache(*, cache_dir: Path) -> None:
    """Point TorchInductor/Triton caches at a shared worker cache root."""
    compile_cache_root = cache_dir / "compile_cache"
    inductor_dir = compile_cache_root / "torchinductor"
    triton_dir = compile_cache_root / "triton"
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_dir))


def main() -> None:
    ap = argparse.ArgumentParser(description="Distributed selfplay worker")

    ap.add_argument("--server-url", type=str, default=None)
    ap.add_argument("--trial-id", type=str, default=None)
    ap.add_argument("--username", type=str, default=None)
    ap.add_argument("--password", type=str, default=None, help="If omitted, prompt (or loaded from config)")
    ap.add_argument(
        "--password-file",
        type=str,
        default=None,
        help="Optional path to read the password from (first line). Safer than --password.",
    )

    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Worker YAML config path. Default: <work_dir>/worker.yaml",
    )
    # Default behavior: save config + password to make volunteer setup one command.
    # Users can opt out with --no-save-config / --no-save-password.
    ap.set_defaults(save_config=True)
    ap.set_defaults(save_password=True)
    ap.add_argument(
        "--no-save-config",
        dest="save_config",
        action="store_false",
        help="Do not write <work_dir>/worker.yaml.",
    )
    ap.add_argument(
        "--no-save-password",
        dest="save_password",
        action="store_false",
        help="Do not store password in worker.yaml (will prompt each run unless provided via CLI).",
    )

    ap.add_argument(
        "--allow-overrides",
        action="store_true",
        help="Allow overriding server-managed selfplay strength knobs via CLI (debug only).",
    )

    ap.add_argument(
        "--self-update",
        "--update",
        action="store_true",
        help="Allow this worker to download and install a newer worker wheel from the server when required.",
    )

    ap.add_argument(
        "--stockfish-from-server",
        "--binaries",
        action="store_true",
        help="Download Stockfish from the server if it is published in the manifest (instead of requiring --stockfish-path).",
    )

    ap.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write worker debug logs (disabled by default).",
    )
    ap.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level for --log-file.",
    )

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--inference-broker-socket", type=str, default=None)
    ap.add_argument("--inference-slot-name", type=str, default=None,
                    help="Shared-memory slot name for slot-based inference broker.")
    ap.add_argument("--inference-slot-max-batch", type=int, default=256,
                    help="Max batch size for inference slot (must match broker).")
    ap.add_argument(
        "--compile-inference",
        action="store_true",
        help="Enable torch.compile for worker-side inference/search models on CUDA.",
    )

    ap.add_argument("--stockfish-path", type=str, default=None)

    # Server-managed strength knobs (defaults come from manifest recommended_worker).
    ap.add_argument("--sf-nodes", type=int, default=None)
    ap.add_argument("--sf-multipv", type=int, default=None)
    ap.add_argument("--sf-policy-temp", type=float, default=None)
    ap.add_argument("--sf-policy-label-smooth", type=float, default=None)

    # Local performance knob
    ap.add_argument("--sf-workers", type=int, default=None)

    # selfplay: if omitted, defaults come from server manifest `recommended_worker`.
    ap.add_argument("--games-per-batch", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=None)
    ap.add_argument("--mcts", type=str, default=None, choices=["puct", "gumbel"])
    ap.add_argument("--mcts-simulations", type=int, default=None)
    ap.add_argument("--playout-cap-fraction", type=float, default=None)
    ap.add_argument("--fast-simulations", type=int, default=None)

    ap.add_argument(
        "--auto-tune",
        action="store_true",
        help="Auto-tune games_per_batch to target a given batch wall-clock time.",
    )
    ap.add_argument("--target-batch-seconds", type=float, default=30.0)
    ap.add_argument("--min-games-per-batch", type=int, default=1)
    ap.add_argument("--max-games-per-batch", type=int, default=64)
    ap.add_argument(
        "--upload-target-positions",
        type=int,
        default=500,
        help="Flush a completed-game upload batch once at least this many positions are buffered locally.",
    )
    ap.add_argument(
        "--upload-flush-seconds",
        type=float,
        default=60.0,
        help="Flush/upload at the next completed game boundary once this many seconds have elapsed since the last successful send.",
    )

    # Server-managed exploration knobs
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--temperature-decay-start-move", type=int, default=None)
    ap.add_argument("--temperature-decay-moves", type=int, default=None)
    ap.add_argument("--temperature-endgame", type=float, default=None)

    # Opening diversification: if omitted, defaults come from server manifest `recommended_worker`.
    ap.add_argument("--opening-book-prob", type=float, default=None)
    ap.add_argument("--opening-book-max-plies", type=int, default=None)
    ap.add_argument("--opening-book-max-games", type=int, default=None)
    ap.add_argument("--random-start-plies", type=int, default=None)

    ap.add_argument("--work-dir", type=str, default="worker")
    ap.add_argument(
        "--shared-cache-dir",
        type=str,
        default=None,
        help="Optional shared cache dir for models/books/binaries across worker processes.",
    )
    ap.add_argument("--poll-seconds", type=float, default=10.0)

    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Run auto-tune for a few batches, save games_per_batch to config, then exit.",
    )
    ap.add_argument("--calibrate-batches", type=int, default=5)

    args = ap.parse_args()

    # Optional debug logging (off by default).
    log = logging.getLogger("chess_anti_engine.worker")
    if args.log_file:
        level = str(args.log_level).lower()
        lvl = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }.get(level, logging.INFO)

        log.setLevel(lvl)
        fh = logging.FileHandler(str(args.log_file))
        fh.setLevel(lvl)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)

    log.info("worker starting version=%s protocol=%s", str(PACKAGE_VERSION), int(PROTOCOL_VERSION))

    # Load config (if present) and merge defaults.
    work_dir = Path(args.work_dir)
    cfg_path = Path(args.config) if args.config is not None else (work_dir / "worker.yaml")
    cfg = load_worker_config(cfg_path)

    # Remember which knobs were explicitly pinned on the CLI.
    pinned_games_per_batch_cli = args.games_per_batch is not None

    # Merge: CLI wins; config provides defaults.
    args.server_url = args.server_url or cfg.get("server_url") or "http://127.0.0.1:45453"
    args.trial_id = args.trial_id or cfg.get("trial_id")
    args.username = args.username or cfg.get("username")

    # password-file: CLI wins; config can provide a default.
    if args.password_file is None and cfg.get("password_file"):
        args.password_file = str(cfg.get("password_file"))

    # Optional persisted flags in worker.yaml
    if (not bool(args.self_update)) and bool(cfg.get("self_update", False)):
        args.self_update = True
    if (not bool(args.stockfish_from_server)) and bool(cfg.get("stockfish_from_server", False)):
        args.stockfish_from_server = True

    args.stockfish_path = args.stockfish_path or cfg.get("stockfish_path")
    args.shared_cache_dir = args.shared_cache_dir or cfg.get("shared_cache_dir")
    if args.sf_workers is None:
        args.sf_workers = int(cfg.get("sf_workers", 1))
    if args.games_per_batch is None and "games_per_batch" in cfg:
        args.games_per_batch = int(cfg.get("games_per_batch"))
    if "upload_target_positions" in cfg:
        args.upload_target_positions = int(cfg.get("upload_target_positions"))
    if "upload_flush_seconds" in cfg:
        args.upload_flush_seconds = float(cfg.get("upload_flush_seconds"))

    # --password-file (CLI) always wins over the saved password in worker.yaml
    # so that a fresh session with a rotated password loads correctly.
    if args.password_file:
        try:
            p = Path(str(args.password_file))
            args.password = p.read_text(encoding="utf-8").splitlines()[0].strip()
        except Exception as e:
            raise SystemExit(f"Failed to read --password-file {args.password_file!r}: {e}")
    elif args.password is None:
        args.password = cfg.get("password")

    if args.username is None:
        raise SystemExit("--username is required (or set username in worker config)")
    if args.stockfish_path is None and not bool(args.stockfish_from_server):
        raise SystemExit(
            "--stockfish-path is required (or set stockfish_path in worker config), unless --stockfish-from-server is enabled"
        )

    # If calibrate is requested, force auto-tune on.
    if bool(args.calibrate):
        args.auto_tune = True
        args.save_config = True

    # Hardening: prevent accidental/drive-by messing with server-managed knobs.
    # To use --allow-overrides, require an explicit env var on top of the CLI flag.
    if bool(getattr(args, "allow_overrides", False)) and os.environ.get("CHESS_ANTI_ENGINE_DEBUG_OVERRIDES") != "1":
        raise SystemExit(
            "--allow-overrides is disabled by default. Set CHESS_ANTI_ENGINE_DEBUG_OVERRIDES=1 to enable."
        )

    try:
        import requests  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("worker requires requests; install with pip install -e '.[worker]' ") from e

    if args.password is None:
        args.password = getpass.getpass("Password: ")

    # If user allows it, persist password in config.
    if bool(args.save_config) and bool(args.save_password):
        cfg["password"] = str(args.password)

    server = str(args.server_url).rstrip("/")
    trial_id = str(args.trial_id).strip() if args.trial_id is not None else ""
    fixed_trial_id = str(trial_id)
    leased_trial_id = str(trial_id)
    trial_api_prefix = f"/v1/trials/{leased_trial_id}" if leased_trial_id else "/v1"
    lease_id = ""

    def _server_url_for(endpoint: str) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if endpoint.startswith("/"):
            return server + endpoint
        return server + "/" + endpoint

    cache_dir = Path(str(args.shared_cache_dir)) if args.shared_cache_dir is not None else (work_dir / "cache")
    shared_cache_enabled = args.shared_cache_dir is not None
    _configure_shared_compile_cache(cache_dir=cache_dir)
    shard_dir = work_dir / "shards"
    pending_dir = shard_dir / "pending"
    uploaded_dir = shard_dir / "uploaded"

    arena_dir = work_dir / "arena"
    arena_pending_dir = arena_dir / "pending"
    arena_uploaded_dir = arena_dir / "uploaded"

    pending_dir.mkdir(parents=True, exist_ok=True)
    uploaded_dir.mkdir(parents=True, exist_ok=True)
    arena_pending_dir.mkdir(parents=True, exist_ok=True)
    arena_uploaded_dir.mkdir(parents=True, exist_ok=True)

    # Local throughput override that can be tuned and persisted.
    games_per_batch_local = int(args.games_per_batch) if args.games_per_batch is not None else None
    worker_id = str(cfg.get("worker_id") or "").strip()
    if not worker_id:
        worker_id = uuid.uuid4().hex
        cfg["worker_id"] = worker_id

    machine_id = str(cfg.get("machine_name") or "").strip() or socket.gethostname()

    def _persist_cfg() -> None:
        if not bool(args.save_config):
            return
        cfg["server_url"] = str(args.server_url)
        if trial_id:
            cfg["trial_id"] = trial_id
        else:
            cfg.pop("trial_id", None)
        cfg["username"] = str(args.username)
        cfg["self_update"] = bool(args.self_update)
        cfg["stockfish_from_server"] = bool(args.stockfish_from_server)
        if args.password_file:
            cfg["password_file"] = str(args.password_file)
        else:
            cfg.pop("password_file", None)
        if not bool(args.stockfish_from_server):
            cfg["stockfish_path"] = str(args.stockfish_path)
        else:
            cfg.pop("stockfish_path", None)
        if args.shared_cache_dir:
            cfg["shared_cache_dir"] = str(args.shared_cache_dir)
        else:
            cfg.pop("shared_cache_dir", None)
        cfg["sf_workers"] = int(args.sf_workers)
        if games_per_batch_local is not None:
            cfg["games_per_batch"] = int(games_per_batch_local)
        cfg["upload_target_positions"] = int(args.upload_target_positions)
        cfg["upload_flush_seconds"] = float(args.upload_flush_seconds)
        if not bool(args.save_password):
            cfg.pop("password", None)
        save_worker_config(cfg_path, cfg)

    # Save initial config (best effort).
    _persist_cfg()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    worker_info = _collect_worker_info(
        device=str(device),
    )
    worker_info["worker_id"] = str(worker_id)

    rng = np.random.default_rng(int(args.seed))

    # Engine: initialize with placeholder settings; we will align nodes from manifest each loop.
    # MultiPV and Skill Level are set at init time; reinitialize when either changes.
    sf = None
    sf_multipv_active = None
    sf_skill_level_active: int | None = None

    last_model_sha = None
    model_cfg_active: ModelConfig | None = None
    model = None
    def _make_inference_client():
        if str(args.inference_slot_name or "").strip():
            return SlotInferenceClient(
                slot_name=str(args.inference_slot_name),
                max_batch=int(args.inference_slot_max_batch),
            )
        if str(args.inference_broker_socket or "").strip():
            return ShmBatchInferenceClient(
                endpoint=str(args.inference_broker_socket),
            )
        return None

    inference_client = _make_inference_client()
    pause_selfplay_active = False
    manifest_state = "active"
    manifest_state_elapsed_s: float | None = None
    upload_buf = _BufferedUpload()
    last_successful_send_s = time.time()

    last_best_sha = None
    best_model = None

    def _upload_pending_shards(*, default_elapsed_s: float | None = None) -> float | None:
        nonlocal last_successful_send_s
        last_uploaded_at: float | None = None
        for sp in sorted(p for p in pending_dir.glob("*.npz") if not p.name.startswith("_tmp_")):
            elapsed_s = default_elapsed_s
            elapsed_path = _pending_elapsed_path(sp)
            if elapsed_path.exists():
                try:
                    elapsed_s = float(elapsed_path.read_text(encoding="utf-8").strip())
                except Exception:
                    elapsed_s = default_elapsed_s
            with sp.open("rb") as f:
                files = {"file": (sp.name, f, "application/octet-stream")}
                r = requests.post(
                    _server_url_for(trial_api_prefix + "/upload_shard"),
                    files=files,
                    auth=(str(args.username), str(args.password)),
                    headers={
                        **_worker_headers(machine_id=machine_id),
                        **(
                            {"X-CAE-Worker-Lease-ID": str(lease_id)}
                            if str(lease_id).strip()
                            else {}
                        ),
                        **(
                            {"X-CAE-Batch-Elapsed-S": str(float(elapsed_s))}
                            if elapsed_s is not None and float(elapsed_s) > 0.0
                            else {}
                        ),
                    },
                    timeout=60.0,
                )
            if r.status_code == 200:
                try:
                    sp.unlink(missing_ok=True)
                except OSError:
                    pass
                elapsed_path.unlink(missing_ok=True)
                last_successful_send_s = time.time()
                last_uploaded_at = float(last_successful_send_s)
            else:
                break
        return last_uploaded_at

    try:
        while True:
            if inference_client is None:
                inference_client = _make_inference_client()

            if not fixed_trial_id:
                body: dict[str, object] = {"worker_info": worker_info}
                if lease_id:
                    body["lease_id"] = str(lease_id)
                if leased_trial_id:
                    body["trial_id"] = str(leased_trial_id)
                r_lease = requests.post(
                    _server_url_for("/v1/lease_trial"),
                    json=body,
                    auth=(str(args.username), str(args.password)),
                    headers=_worker_headers(),
                    timeout=30.0,
                )
                if r_lease.status_code != 200:
                    time.sleep(float(args.poll_seconds))
                    continue
                lease = r_lease.json()
                new_trial_id = str(lease.get("trial_id") or "").strip()
                new_api_prefix = str(lease.get("api_prefix") or "/v1").strip() or "/v1"
                new_lease_id = str(lease.get("lease_id") or "").strip()
                if new_trial_id != leased_trial_id:
                    log.info("leased trial assignment changed: %s -> %s", leased_trial_id or "<root>", new_trial_id or "<root>")
                leased_trial_id = new_trial_id
                trial_api_prefix = new_api_prefix
                lease_id = new_lease_id

            # Upload any pending shards first (skip in-progress temp files).
            _upload_pending_shards(default_elapsed_s=float(cfg.get("_last_batch_elapsed_s", 0.0) or 0.0))

            # Upload any pending arena results.
            for jp in sorted(arena_pending_dir.glob("*.json")):
                try:
                    payload = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    # bad local file; quarantine to uploaded to avoid retry storms
                    jp.replace(arena_uploaded_dir / jp.name)
                    continue

                r = requests.post(
                    _server_url_for(trial_api_prefix + "/upload_arena_result"),
                    json=payload,
                    auth=(str(args.username), str(args.password)),
                    headers=_worker_headers(),
                    timeout=60.0,
                )
                if r.status_code == 200:
                    jp.unlink(missing_ok=True)
                else:
                    break

            # Poll manifest
            r = requests.get(
                _server_url_for(trial_api_prefix + "/manifest"),
                timeout=30.0,
                headers=_manifest_poll_headers(
                    worker_id=worker_id,
                    lease_id=lease_id,
                    state=manifest_state,
                    elapsed_s=manifest_state_elapsed_s,
                ),
            )
            manifest_state = "active"
            manifest_state_elapsed_s = None
            if r.status_code == 426:
                # Server says "upgrade required".
                if bool(args.self_update) and os.environ.get("CAE_SELF_UPDATED") != "1":
                    # Ask the server for minimal update info (does not require compatibility).
                    r2 = requests.get(_server_url_for(trial_api_prefix + "/update_info"), timeout=30.0)
                    if r2.status_code != 200:
                        raise SystemExit(f"Upgrade required but could not fetch update info for self-update: {r2.text}")
                    update_info = r2.json()

                    ww = update_info.get("worker_wheel")
                    if not (isinstance(ww, dict) and ww.get("endpoint") and ww.get("sha256")):
                        try:
                            detail = r.json().get("detail")
                        except Exception:
                            detail = None
                        raise SystemExit(
                            f"Worker is not compatible with server and no worker_wheel was published for self-update: {detail or r.text}"
                        )

                    wheel_version = str(ww.get("version") or update_info.get("server_version") or "0.0.0")
                    sha = str(ww.get("sha256"))
                    endpoint = str(ww.get("endpoint"))
                    wheel_path = cache_dir / f"worker_{sha}.whl"
                    log.warning(
                        "self-update: installing worker wheel version=%s sha=%s",
                        wheel_version,
                        sha,
                    )
                    _download_and_verify_shared(
                        _server_url_for(endpoint),
                        out_path=wheel_path,
                        expected_sha256=sha,
                        headers=_worker_headers(),
                    )
                    _pip_install_wheel(wheel_path)
                    _restart_process()

                try:
                    detail = r.json().get("detail")
                except Exception:
                    detail = None
                raise SystemExit(f"Worker is not compatible with server: {detail or r.text}")

            if r.status_code != 200:
                time.sleep(float(args.poll_seconds))
                continue

            manifest = r.json()

            # Compatibility guardrails (manifest-driven).
            req_proto = manifest.get("protocol_version")
            protocol_mismatch = False
            if req_proto is not None:
                try:
                    protocol_mismatch = int(req_proto) != int(PROTOCOL_VERSION)
                except Exception:
                    raise SystemExit(f"Bad protocol_version in manifest: {req_proto!r}")

            min_v = manifest.get("min_worker_version")
            version_too_old = bool(min_v is not None and version_lt(str(PACKAGE_VERSION), str(min_v)))

            enc = manifest.get("encoding") or {}
            if "policy_size" in enc and int(enc.get("policy_size") or 0) != int(POLICY_SIZE):
                raise SystemExit(f"policy_size mismatch: worker={POLICY_SIZE} server={enc.get('policy_size')}")
            if "input_planes" in enc and int(enc.get("input_planes") or 0) != 146:
                raise SystemExit(f"input_planes mismatch: worker expects 146, server={enc.get('input_planes')}")

            # Optional self-update (manifest-driven).
            if bool(args.self_update) and os.environ.get("CAE_SELF_UPDATED") != "1":
                ww = manifest.get("worker_wheel")
                if isinstance(ww, dict) and ww.get("endpoint") and ww.get("sha256"):
                    wheel_version = str(ww.get("version") or manifest.get("server_version") or "0.0.0")
                    need = False
                    if protocol_mismatch or version_too_old:
                        need = True
                    elif version_lt(str(PACKAGE_VERSION), wheel_version):
                        need = True

                    if need:
                        sha = str(ww.get("sha256"))
                        endpoint = str(ww.get("endpoint"))
                        wheel_path = cache_dir / f"worker_{sha}.whl"
                        log.warning(
                            "self-update: installing worker wheel version=%s sha=%s",
                            wheel_version,
                            sha,
                        )
                        _download_and_verify_shared(
                            _server_url_for(endpoint),
                            out_path=wheel_path,
                            expected_sha256=sha,
                            headers=_worker_headers(),
                        )
                        _pip_install_wheel(wheel_path)
                        _restart_process()

            # Enforce after any self-update opportunity.
            if protocol_mismatch:
                raise SystemExit(f"Protocol mismatch: worker={PROTOCOL_VERSION} server_required={req_proto}")
            if version_too_old:
                raise SystemExit(f"Worker too old: worker={PACKAGE_VERSION} min_required={min_v}")

            reco = manifest.get("recommended_worker") or {}
            backpressure = manifest.get("backpressure") or {}
            task = manifest.get("task") or {"type": "selfplay"}
            task_type = str(task.get("type", "selfplay")).lower()
            pause_selfplay = False
            pause_reason = ""
            if task_type == "selfplay":
                pause_selfplay = bool(reco.get("pause_selfplay", False))
                pause_reason = str(reco.get("pause_reason") or "")
                if (not pause_selfplay) and isinstance(backpressure, dict):
                    pause_selfplay = bool(backpressure.get("pause_selfplay", False))
                    pause_reason = str(backpressure.get("pause_reason") or pause_reason)
            if pause_selfplay:
                if not pause_selfplay_active:
                    log.info(
                        "selfplay paused by server%s",
                        f": {pause_reason}" if pause_reason else "",
                    )
                    pause_selfplay_active = True
                sleep_s = max(0.1, float(args.poll_seconds))
                time.sleep(sleep_s)
                manifest_state = "paused_selfplay"
                manifest_state_elapsed_s = sleep_s
                continue
            if pause_selfplay_active:
                log.info("selfplay pause cleared by server")
                pause_selfplay_active = False
            need_local_model = inference_client is None or task_type == "arena"

            model_info = manifest.get("model") or {}
            model_sha = str(model_info.get("sha256") or "")
            if not model_sha:
                time.sleep(float(args.poll_seconds))
                continue
            model_step = int(manifest.get("trainer_step") or 0)

            if (
                upload_buf.positions > 0
                and (
                    str(upload_buf.model_sha or "") != str(model_sha)
                    or int(upload_buf.model_step or 0) != int(model_step)
                )
            ):
                shard_path, elapsed_s = _flush_upload_buffer_to_pending(
                    pending_dir=pending_dir,
                    username=str(args.username),
                    buf=upload_buf,
                    now_s=time.time(),
                )
                if shard_path is not None:
                    uploaded_at = _upload_pending_shards(default_elapsed_s=float(elapsed_s))
                    if uploaded_at is not None:
                        last_successful_send_s = float(uploaded_at)

            if need_local_model and model_sha != last_model_sha:
                log.info("switching to latest model sha=%s", model_sha)
                model_path = cache_dir / f"model_{model_sha}.pt"
                # Download (or re-download) and verify sha256.
                # Note: the server serves a stable endpoint (/v1/model) whose contents can change
                # whenever the learner publishes a new model. If we fetched a slightly stale
                # manifest, the expected sha may not match what /v1/model returns.
                #
                # In that case, do NOT crash-loop; just re-poll the manifest next iteration.
                if (not model_path.exists()) or (_sha256_file(model_path) != model_sha):
                    try:
                        _download_and_verify_shared(
                            _server_url_for(str(model_info.get("endpoint") or (trial_api_prefix + "/model"))),
                            out_path=model_path,
                            expected_sha256=model_sha,
                            headers=_worker_headers(),
                        )
                    except Exception as e:
                        log.warning(
                            "model download failed (likely mid-publish race). Will re-poll manifest: %s",
                            e,
                        )
                        model_path.unlink(missing_ok=True)
                        time.sleep(float(args.poll_seconds))
                        continue

                mc = manifest.get("model_config") or {}
                model_cfg = ModelConfig(
                    kind=str(mc.get("kind", "transformer")),
                    embed_dim=int(mc.get("embed_dim", 256)),
                    num_layers=int(mc.get("num_layers", 6)),
                    num_heads=int(mc.get("num_heads", 8)),
                    ffn_mult=int(mc.get("ffn_mult", 2)),
                    use_smolgen=bool(mc.get("use_smolgen", True)),
                    use_nla=bool(mc.get("use_nla", False)),
                    use_qk_rmsnorm=bool(mc.get("use_qk_rmsnorm", False)),
                    use_gradient_checkpointing=bool(mc.get("gradient_checkpointing", False)),
                )
                model_cfg_active = model_cfg

                model = build_model(model_cfg)
                ckpt = torch.load(str(model_path), map_location="cpu")
                sd = ckpt.get("model", ckpt)
                model.load_state_dict(sd)
                model.to(device)
                model.eval()
                if bool(args.compile_inference):
                    compile_t0 = time.time()
                    log.info("compile starting model_sha=%s", str(model_sha)[:8])
                    model = _maybe_compile_inference_model(model, device=str(device))
                    log.info(
                        "compile finished model_sha=%s elapsed_s=%.2f",
                        str(model_sha)[:8],
                        float(time.time() - compile_t0),
                    )

                if last_model_sha is not None and not fixed_trial_id:
                    # Reconsider assignment at natural model-boundary checkpoints.
                    lease_id = ""
                    leased_trial_id = ""
                    trial_api_prefix = "/v1"
                last_model_sha = model_sha

            # Opening book (optional) — downloaded once, used by both selfplay and arena.
            opening_book_path = None
            if "opening_book" in manifest:
                ob = manifest.get("opening_book") or {}
                filename = str(ob.get("filename") or "opening_book")
                sha = str(ob.get("sha256") or "")

                if sha:
                    ob_path = cache_dir / f"opening_{sha}_{filename}"
                else:
                    ob_path = cache_dir / filename

                if sha:
                    if (not ob_path.exists()) or (_sha256_file(ob_path) != sha):
                        log.info("downloading opening book sha=%s filename=%s", sha, filename)
                        _download_and_verify_shared(
                            _server_url_for(str(ob.get("endpoint") or "/v1/opening_book")),
                            out_path=ob_path,
                            expected_sha256=sha,
                            headers=_worker_headers(),
                        )
                else:
                    if not ob_path.exists():
                        _download(
                            _server_url_for(str(ob.get("endpoint") or "/v1/opening_book")),
                            out_path=ob_path,
                            headers=_worker_headers(),
                        )
                opening_book_path = str(ob_path)

            # Opening book 2 (optional).
            opening_book_path_2 = None
            if "opening_book_2" in manifest:
                ob2 = manifest.get("opening_book_2") or {}
                filename2 = str(ob2.get("filename") or "opening_book_2")
                sha2 = str(ob2.get("sha256") or "")

                if sha2:
                    ob2_path = cache_dir / f"opening2_{sha2}_{filename2}"
                else:
                    ob2_path = cache_dir / f"book2_{filename2}"

                if sha2:
                    if (not ob2_path.exists()) or (_sha256_file(ob2_path) != sha2):
                        log.info("downloading opening book 2 sha=%s filename=%s", sha2, filename2)
                        _download_and_verify_shared(
                            _server_url_for(str(ob2.get("endpoint") or "/v1/opening_book_2")),
                            out_path=ob2_path,
                            expected_sha256=sha2,
                            headers=_worker_headers(),
                        )
                else:
                    if not ob2_path.exists():
                        _download(
                            _server_url_for(str(ob2.get("endpoint") or "/v1/opening_book_2")),
                            out_path=ob2_path,
                            headers=_worker_headers(),
                        )
                opening_book_path_2 = str(ob2_path)

            # Resolve effective settings.
            # By default we keep strength knobs server-managed for consistency.
            server_managed = [
                "max_plies",
                "mcts",
                "mcts_simulations",
                "playout_cap_fraction",
                "fast_simulations",
                "opponent_random_move_prob",
                "opening_book_prob",
                "opening_book_max_plies",
                "opening_book_max_games",
                "random_start_plies",
                "sf_nodes",
                "sf_multipv",
                "sf_policy_temp",
                "sf_policy_label_smooth",
                "timeout_adjudication_threshold",
                "temperature",
                "temperature_decay_start_move",
                "temperature_decay_moves",
                "temperature_endgame",
            ]
            if not bool(args.allow_overrides):
                if (
                    args.max_plies is not None
                    or args.mcts is not None
                    or args.mcts_simulations is not None
                    or args.playout_cap_fraction is not None
                    or args.fast_simulations is not None
                    or args.opening_book_prob is not None
                    or args.opening_book_max_plies is not None
                    or args.opening_book_max_games is not None
                    or args.random_start_plies is not None
                    or args.sf_nodes is not None
                    or args.sf_multipv is not None
                    or args.sf_policy_temp is not None
                    or args.sf_policy_label_smooth is not None
                    or args.temperature is not None
                    or args.temperature_decay_start_move is not None
                    or args.temperature_decay_moves is not None
                    or args.temperature_endgame is not None
                ):
                    raise SystemExit(
                        "This worker is configured for server-managed strength knobs. "
                        "Remove selfplay/strength flags (nodes/sims/etc), or pass --allow-overrides for debugging."
                    )

            games_per_batch = (
                int(games_per_batch_local)
                if games_per_batch_local is not None
                else int(reco.get("games_per_batch", 8))
            )

            max_plies = int(args.max_plies) if args.max_plies is not None else int(reco.get("max_plies", 240))
            mcts_type = str(args.mcts) if args.mcts is not None else str(reco.get("mcts", "puct"))
            mcts_sims = int(args.mcts_simulations) if args.mcts_simulations is not None else int(reco.get("mcts_simulations", 50))
            playout_cap_fraction = (
                float(args.playout_cap_fraction) if args.playout_cap_fraction is not None else float(reco.get("playout_cap_fraction", 0.25))
            )
            fast_sims = int(args.fast_simulations) if args.fast_simulations is not None else int(reco.get("fast_simulations", 8))
            opponent_random_move_prob = float(reco.get("opponent_random_move_prob", 0.0))
            opponent_topk_min = int(reco.get("opponent_topk_min", 1))
            opponent_suboptimal_wdl_regret_max = float(reco.get("opponent_suboptimal_wdl_regret_max", -1.0))
            opponent_suboptimal_wdl_regret_min = float(reco.get("opponent_suboptimal_wdl_regret_min", -1.0))
            opponent_random_move_prob_start = float(reco.get("opponent_random_move_prob_start", 1.0))
            opponent_random_move_prob_min = float(reco.get("opponent_random_move_prob_min", 0.0))
            opponent_wdl_regret_limit_raw = reco.get("opponent_wdl_regret_limit", None)
            opponent_wdl_regret_limit = float(opponent_wdl_regret_limit_raw) if opponent_wdl_regret_limit_raw is not None else None
            selfplay_fraction = float(reco.get("selfplay_fraction", 0.0))
            timeout_adjudication_threshold = float(reco.get("timeout_adjudication_threshold", 0.90))

            opening_book_prob = (
                float(args.opening_book_prob) if args.opening_book_prob is not None else float(reco.get("opening_book_prob", 1.0))
            )
            opening_book_max_plies = (
                int(args.opening_book_max_plies) if args.opening_book_max_plies is not None else int(reco.get("opening_book_max_plies", 4))
            )
            opening_book_max_games = (
                int(args.opening_book_max_games) if args.opening_book_max_games is not None else int(reco.get("opening_book_max_games", 200000))
            )
            opening_book_max_plies_2 = int(reco.get("opening_book_max_plies_2", 16))
            opening_book_max_games_2 = int(reco.get("opening_book_max_games_2", 200000))
            opening_book_mix_prob_2 = float(reco.get("opening_book_mix_prob_2", 0.0))
            random_start_plies = (
                int(args.random_start_plies) if args.random_start_plies is not None else int(reco.get("random_start_plies", 0))
            )

            sf_nodes = int(args.sf_nodes) if args.sf_nodes is not None else int(reco.get("sf_nodes", 2000))
            sf_multipv = int(args.sf_multipv) if args.sf_multipv is not None else int(reco.get("sf_multipv", 5))
            _reco_skill = reco.get("sf_skill_level")
            sf_skill_level: int | None = None if _reco_skill is None else int(_reco_skill)
            sf_policy_temp = (
                float(args.sf_policy_temp) if args.sf_policy_temp is not None else float(reco.get("sf_policy_temp", 0.25))
            )
            sf_policy_label_smooth = (
                float(args.sf_policy_label_smooth)
                if args.sf_policy_label_smooth is not None
                else float(reco.get("sf_policy_label_smooth", 0.05))
            )

            temperature = float(args.temperature) if args.temperature is not None else float(reco.get("temperature", 1.0))
            t_start = (
                int(args.temperature_decay_start_move)
                if args.temperature_decay_start_move is not None
                else int(reco.get("temperature_decay_start_move", 20))
            )
            t_moves = (
                int(args.temperature_decay_moves)
                if args.temperature_decay_moves is not None
                else int(reco.get("temperature_decay_moves", 60))
            )
            t_end = (
                float(args.temperature_endgame)
                if args.temperature_endgame is not None
                else float(reco.get("temperature_endgame", 0.6))
            )

            # If server requests arena matches, run those instead of selfplay.
            if task_type == "arena":
                best_info = manifest.get("best_model") or {}
                best_sha = str(best_info.get("sha256") or "")
                if not best_sha:
                    time.sleep(float(args.poll_seconds))
                    continue
                if model is None or model_cfg_active is None:
                    time.sleep(float(args.poll_seconds))
                    continue

                if best_sha != last_best_sha:
                    log.info("arena task: loading best model sha=%s", best_sha)
                    best_path = cache_dir / f"best_{best_sha}.pt"
                    endpoint = str(best_info.get("endpoint") or "/v1/best_model")
                    if (not best_path.exists()) or (_sha256_file(best_path) != best_sha):
                        try:
                            _download_and_verify_shared(
                                _server_url_for(endpoint),
                                out_path=best_path,
                                expected_sha256=best_sha,
                                headers=_worker_headers(),
                            )
                        except Exception as e:
                            log.warning(
                                "best model download failed (likely mid-publish race). Will retry: %s",
                                e,
                            )
                            best_path.unlink(missing_ok=True)
                            time.sleep(float(args.poll_seconds))
                            continue

                    best_model = build_model(model_cfg_active)
                    ckpt = torch.load(str(best_path), map_location="cpu")
                    sd = ckpt.get("model", ckpt)
                    best_model.load_state_dict(sd)
                    best_model.to(device)
                    best_model.eval()
                    if bool(args.compile_inference):
                        compile_t0 = time.time()
                        log.info("compile starting best_model_sha=%s", str(best_sha)[:8])
                        best_model = _maybe_compile_inference_model(best_model, device=str(device))
                        log.info(
                            "compile finished best_model_sha=%s elapsed_s=%.2f",
                            str(best_sha)[:8],
                            float(time.time() - compile_t0),
                        )
                    last_best_sha = best_sha

                if best_model is None:
                    time.sleep(float(args.poll_seconds))
                    continue

                arena_cfg = (task.get("arena") or {}) if isinstance(task, dict) else {}
                batch_games = int(arena_cfg.get("batch_games", 8))
                max_plies_arena = int(arena_cfg.get("max_plies", 240))
                mcts_type_arena = str(arena_cfg.get("mcts", "puct"))
                mcts_sims_arena = int(arena_cfg.get("mcts_simulations", 200))
                c_puct_arena = float(arena_cfg.get("c_puct", 2.5))
                swap_sides = bool(arena_cfg.get("swap_sides", True))
                temperature_arena = float(arena_cfg.get("temperature", 0.1))
                random_start_plies_arena = int(arena_cfg.get("random_start_plies", 2))
                opening_book_prob_arena = float(arena_cfg.get("opening_book_prob", 1.0))
                opening_book_max_plies_arena = int(arena_cfg.get("opening_book_max_plies", 4))
                opening_book_max_games_arena = int(arena_cfg.get("opening_book_max_games", 200000))
                arena_opening_cfg = OpeningConfig(
                    opening_book_path=opening_book_path,
                    opening_book_prob=opening_book_prob_arena if opening_book_path else 0.0,
                    opening_book_max_plies=opening_book_max_plies_arena,
                    opening_book_max_games=opening_book_max_games_arena,
                    random_start_plies=random_start_plies_arena,
                )

                g = max(1, batch_games)
                a_plays_white = [bool(i % 2 == 0) for i in range(g)] if swap_sides else [True] * g

                stats = play_match_batch(
                    model,
                    best_model,
                    device=str(device),
                    rng=rng,
                    games=g,
                    max_plies=max_plies_arena,
                    a_plays_white=a_plays_white,
                    mcts_type=mcts_type_arena,
                    mcts_simulations=mcts_sims_arena,
                    temperature=temperature_arena,
                    c_puct=c_puct_arena,
                    opening_cfg=arena_opening_cfg,
                )

                ts = int(time.time())
                payload = {
                    "generated_at_unix": ts,
                    "worker_username": str(args.username),
                    "a_sha256": str(model_sha),
                    "b_sha256": str(best_sha),
                    "games": int(stats.games),
                    "a_win": int(stats.a_win),
                    "a_draw": int(stats.a_draw),
                    "a_loss": int(stats.a_loss),
                    "a_as_white": int(stats.a_as_white),
                    "a_as_black": int(stats.a_as_black),
                    "max_plies": int(stats.max_plies),
                    "mcts": str(mcts_type_arena),
                    "mcts_simulations": int(mcts_sims_arena),
                    "c_puct": float(c_puct_arena),
                    "swap_sides": bool(swap_sides),
                }

                out = arena_pending_dir / f"{ts}_{model_sha[:8]}_vs_{best_sha[:8]}_{stats.games}g.json"
                out.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

                if not shared_cache_enabled:
                    _prune_cached_models(cache_dir=cache_dir, keep_shas={model_sha, best_sha})

                time.sleep(0.1)
                continue

            # (opening_book_path already resolved above, shared by arena and selfplay)

            if need_local_model and model is None:
                time.sleep(float(args.poll_seconds))
                continue

            # Resolve stockfish binary.
            stockfish_path = str(args.stockfish_path) if args.stockfish_path is not None else ""
            if bool(args.stockfish_from_server):
                sf_rec = manifest.get("stockfish")
                if not isinstance(sf_rec, dict) or not sf_rec.get("endpoint") or not sf_rec.get("sha256"):
                    raise SystemExit("--stockfish-from-server enabled but server did not publish stockfish")
                sf_sha = str(sf_rec.get("sha256"))
                sf_endpoint = str(sf_rec.get("endpoint"))
                sf_filename = str(sf_rec.get("filename") or "stockfish")
                sf_cached = cache_dir / f"stockfish_{sf_sha}_{sf_filename}"
                if (not sf_cached.exists()) or (_sha256_file(sf_cached) != sf_sha):
                    log.info("downloading stockfish sha=%s filename=%s", sf_sha, sf_filename)
                    _download_and_verify_shared(
                        _server_url_for(sf_endpoint),
                        out_path=sf_cached,
                        expected_sha256=sf_sha,
                        headers=_worker_headers(),
                    )
                    _ensure_executable(sf_cached)
                stockfish_path = str(sf_cached)

            # (Re)initialize engine if multipv or skill_level changed (must be set at init time)
            skill_changed = sf_skill_level_active != sf_skill_level
            multipv_changed = sf_multipv_active is None or int(sf_multipv_active) != int(sf_multipv)
            if sf is None or multipv_changed or skill_changed:
                if sf is not None:
                    try:
                        sf.close()
                    except Exception:
                        pass

                if int(args.sf_workers) > 1:
                    sf = StockfishPool(
                        path=str(stockfish_path),
                        nodes=int(sf_nodes),
                        num_workers=int(args.sf_workers),
                        multipv=int(sf_multipv),
                        skill_level=sf_skill_level,
                    )
                else:
                    sf = StockfishUCI(str(stockfish_path), nodes=int(sf_nodes), multipv=int(sf_multipv),
                                      skill_level=sf_skill_level)
                sf_multipv_active = int(sf_multipv)
                sf_skill_level_active = sf_skill_level
            else:
                # update nodes dynamically
                if hasattr(sf, "set_nodes"):
                    sf.set_nodes(int(sf_nodes))

            # Generate a shard
            t0 = time.time()
            saw_completed_game = False

            def _on_completed_game(game_batch) -> None:
                nonlocal last_successful_send_s, saw_completed_game
                saw_completed_game = True
                now_s = time.time()
                _buffer_add_completed_game(
                    buf=upload_buf,
                    game_batch=game_batch,
                    now_s=now_s,
                    model_sha=model_sha,
                    model_step=model_step,
                )
                shard_path, elapsed_s = _maybe_flush_upload_buffer(
                    pending_dir=pending_dir,
                    username=str(args.username),
                    buf=upload_buf,
                    now_s=now_s,
                    last_send_s=last_successful_send_s,
                    target_positions=int(args.upload_target_positions),
                    flush_seconds=float(args.upload_flush_seconds),
                    force=False,
                )
                if shard_path is not None:
                    uploaded_at = _upload_pending_shards(default_elapsed_s=float(elapsed_s))
                    if uploaded_at is not None:
                        last_successful_send_s = float(uploaded_at)

            try:
                samples, stats = play_batch(
                    model if need_local_model else None,
                    device=str(device),
                    rng=rng,
                    stockfish=sf,
                    evaluator=inference_client,
                    games=int(games_per_batch),
                    temperature=float(temperature),
                    temperature_decay_start_move=int(t_start),
                    temperature_decay_moves=int(t_moves),
                    temperature_endgame=float(t_end),
                    max_plies=int(max_plies),
                    mcts_simulations=int(mcts_sims),
                    mcts_type=str(mcts_type),
                    playout_cap_fraction=float(playout_cap_fraction),
                    fast_simulations=int(fast_sims),
                    sf_policy_temp=float(sf_policy_temp),
                    sf_policy_label_smooth=float(sf_policy_label_smooth),
                    timeout_adjudication_threshold=float(timeout_adjudication_threshold),
                    opponent_random_move_prob=float(opponent_random_move_prob),
                    opponent_topk_stage_end=float(reco.get("opponent_topk_stage_end", 0.5)),
                    opponent_topk_min=int(opponent_topk_min),
                    opponent_suboptimal_wdl_regret_max=float(opponent_suboptimal_wdl_regret_max),
                    opponent_suboptimal_wdl_regret_min=float(opponent_suboptimal_wdl_regret_min),
                    opponent_random_move_prob_start=float(opponent_random_move_prob_start),
                    opponent_random_move_prob_min=float(opponent_random_move_prob_min),
                    opponent_wdl_regret_limit=opponent_wdl_regret_limit,
                    selfplay_fraction=float(selfplay_fraction),
                    opening_book_path=opening_book_path,
                    opening_book_max_plies=int(opening_book_max_plies),
                    opening_book_max_games=int(opening_book_max_games),
                    opening_book_prob=float(opening_book_prob),
                    opening_book_path_2=opening_book_path_2,
                    opening_book_max_plies_2=int(opening_book_max_plies_2),
                    opening_book_max_games_2=int(opening_book_max_games_2),
                    opening_book_mix_prob_2=float(opening_book_mix_prob_2),
                    random_start_plies=int(random_start_plies),
                    on_game_complete=_on_completed_game,
                )
            except TimeoutError as exc:
                if inference_client is None:
                    raise
                log.warning("inference broker timed out; resetting client: %s", exc)
                try:
                    inference_client.close()
                except Exception:
                    pass
                inference_client = None
                time.sleep(float(args.poll_seconds))
                continue
            except RuntimeError as exc:
                err = str(exc).lower()
                if inference_client is None or not any(tok in err for tok in ("inference", "broker", "slot")):
                    raise
                log.warning("inference broker error; resetting client: %s", exc)
                try:
                    inference_client.close()
                except Exception:
                    pass
                inference_client = None
                time.sleep(float(args.poll_seconds))
                continue
            t1 = time.time()

            # Log batch outcome (only visible if --log-file is enabled).
            log.info(
                "batch done: games=%d positions=%d W/D/L=%d/%d/%d draws=%d timeouts=%d rand=%.2f sf_nodes=%s ppg=%.1f elapsed_s=%.2f",
                int(stats.games),
                int(stats.positions),
                int(stats.w),
                int(stats.d),
                int(stats.l),
                int(getattr(stats, "total_draw_games", 0)),
                int(getattr(stats, "timeout_games", 0)),
                float(opponent_random_move_prob),
                str(stats.sf_nodes),
                float(stats.positions) / max(1, int(stats.games)),
                float(t1 - t0),
            )

            if bool(args.auto_tune) and not bool(pinned_games_per_batch_cli):
                tuned = tune_games_per_batch(
                    current=int(games_per_batch),
                    elapsed_s=float(t1 - t0),
                    target_s=float(args.target_batch_seconds),
                    min_games=int(args.min_games_per_batch),
                    max_games=int(args.max_games_per_batch),
                )
                games_per_batch_local = int(tuned)

                # Persist the calibrated value if requested.
                _persist_cfg()

            if bool(args.calibrate):
                # Count completed batches and exit once we have calibrated.
                cfg["_calibrate_batches_done"] = int(cfg.get("_calibrate_batches_done", 0)) + 1
                if int(cfg["_calibrate_batches_done"]) >= int(args.calibrate_batches):
                    cfg.pop("_calibrate_batches_done", None)
                    _persist_cfg()
                    raise SystemExit(0)

            cfg["_last_batch_elapsed_s"] = float(t1 - t0)
            now_s = time.time()
            shard_path, elapsed_s = _maybe_flush_upload_buffer(
                pending_dir=pending_dir,
                username=str(args.username),
                buf=upload_buf,
                now_s=now_s,
                last_send_s=last_successful_send_s,
                target_positions=int(args.upload_target_positions),
                flush_seconds=float(args.upload_flush_seconds),
                force=True,
            )
            if _should_write_fallback_batch(
                shard_path=shard_path,
                samples=samples,
                saw_completed_game=saw_completed_game,
            ):
                log.warning(
                    "batch returned %d samples but upload buffer was empty; writing fallback batch shard",
                    int(len(samples)),
                )
                ts = int(now_s)
                shard_path = pending_dir / f"{ts}_{model_sha[:8]}_{stats.games}g_{stats.positions}p.npz"
                meta = ShardMeta(
                    username=str(args.username),
                    generated_at_unix=ts,
                    model_sha256=str(model_sha),
                    model_step=int(model_step),
                    games=int(stats.games),
                    positions=int(stats.positions),
                    wins=int(stats.w),
                    draws=int(stats.d),
                    losses=int(stats.l),
                    total_game_plies=int(getattr(stats, "total_game_plies", 0)),
                    adjudicated_games=int(getattr(stats, "adjudicated_games", 0)),
                    total_draw_games=int(getattr(stats, "total_draw_games", 0)),
                    selfplay_games=int(getattr(stats, "selfplay_games", 0)),
                    selfplay_adjudicated_games=int(getattr(stats, "selfplay_adjudicated_games", 0)),
                    selfplay_draw_games=int(getattr(stats, "selfplay_draw_games", 0)),
                    curriculum_games=int(getattr(stats, "curriculum_games", 0)),
                    curriculum_adjudicated_games=int(getattr(stats, "curriculum_adjudicated_games", 0)),
                    curriculum_draw_games=int(getattr(stats, "curriculum_draw_games", 0)),
                )
                tmp_path = pending_dir / f"_tmp_{shard_path.name}"
                save_npz(tmp_path, samples=samples, meta=meta, compress=False)
                tmp_path.replace(shard_path)
                elapsed_path = _pending_elapsed_path(shard_path)
                elapsed_s = float(t1 - t0)
                elapsed_path.write_text(f"{float(elapsed_s):.6f}\n", encoding="utf-8")
            elif shard_path is None:
                log.info("batch produced no policy samples; skipping shard write")

            # Prune old cached models opportunistically.
            best_info = manifest.get("best_model") or {}
            best_sha = str(best_info.get("sha256") or "")
            if not shared_cache_enabled:
                _prune_cached_models(cache_dir=cache_dir, keep_shas={model_sha, best_sha})

            if shard_path is not None:
                uploaded_at = _upload_pending_shards(default_elapsed_s=float(elapsed_s))
                if uploaded_at is not None:
                    last_successful_send_s = float(uploaded_at)

            # Try uploading immediately next loop.
            time.sleep(0.1)

    finally:
        if inference_client is not None and hasattr(inference_client, "close"):
            try:
                inference_client.close()
            except Exception:
                pass
        if sf is not None:
            sf.close()


if __name__ == "__main__":
    main()
