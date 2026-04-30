from __future__ import annotations

import argparse
import dataclasses
import getpass
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, TypeVar, cast, overload

import numpy as np
import torch

if TYPE_CHECKING:
    import requests

from chess_anti_engine.inference import (
    AOTEvaluator,
    DirectGPUEvaluator,
    SlotInferenceClient,
    ThreadedBatchEvaluator,
)
from chess_anti_engine.inference_threaded import ThreadedDispatcher
from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
    model_config_from_manifest_dict,
)
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    delete_shard_path,
    pack_shard_for_upload,
)
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.manager import BatchStats
from chess_anti_engine.selfplay.match import play_match_batch
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish import StockfishPool, StockfishUCI
from chess_anti_engine.utils import sha256_file as _sha256_file
from chess_anti_engine.utils.versioning import version_lt
from chess_anti_engine.version import PACKAGE_NAME, PACKAGE_VERSION, PROTOCOL_VERSION
from chess_anti_engine.worker_assets import (
    _cached_sha_asset_needs_refresh,
    _download_and_verify_shared,
    _download_opening_book,
    _ensure_executable,
    _prune_cached_models,
)
from chess_anti_engine.worker_buffer import (
    _buffer_add_completed_game,
    _BufferedUpload,
    _flush_upload_buffer_to_pending,
    _maybe_flush_upload_buffer,
    _pending_elapsed_path,
)
from chess_anti_engine.worker_config import load_worker_config, save_worker_config

_ResolveT = TypeVar("_ResolveT")

  # Wire-protocol values sent in X-CAE-Worker-State; tests assert on them
  # (tests/test_distributed_selfplay_backpressure.py).
_MANIFEST_STATE_ACTIVE = "active"
_MANIFEST_STATE_PAUSED = "paused_selfplay"


def _extract_worker_wheel(payload: dict) -> dict | None:
    """Return the worker_wheel dict iff it has both endpoint and sha256, else None."""
    ww = payload.get("worker_wheel")
    if isinstance(ww, dict) and ww.get("endpoint") and ww.get("sha256"):
        return ww
    return None


class _ManifestCompat(NamedTuple):
    protocol_mismatch: bool
    version_too_old: bool
    req_proto: object
    min_worker_version: object


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


def _maybe_apply_fp8_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Quantize eligible Linear layers to FP8 (dynamic activation, FP8 weight).

    Skips layers where FP8 hurts accuracy or speed: Smolgen (small batch dim),
    non-16-aligned dims (tensor core requirement), tiny output heads.
    Requires torch.compile afterwards for actual speedup.
    """
    try:
        from torchao.quantization import (  # optional dep
            Float8DynamicActivationFloat8WeightConfig,
            PerRow,
            quantize_,
        )
    except ImportError:
        return model

    def _fp8_filter(mod: torch.nn.Module, fqn: str) -> bool:
        if not isinstance(mod, torch.nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if "smolgen" in fqn:
            return False
        if fqn.endswith(".net.2") and mod.out_features <= 32:
            return False
        return True

    try:
        quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), filter_fn=_fp8_filter)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("FP8 quantization failed, continuing with BF16: %s", exc)
    return model


def _maybe_compile_inference_model(
    model: torch.nn.Module, *, device: str, mode: str = "reduce-overhead",
    use_fp8: bool = False,
) -> torch.nn.Module:
    if not str(device).startswith("cuda"):
        return model
    if use_fp8:
        model = _maybe_apply_fp8_inference(model)
    try:
        return cast("torch.nn.Module", torch.compile(model, mode=mode))
    except Exception:
        return model


def _sync_evaluator_to_model(
    evaluator: DirectGPUEvaluator | ThreadedBatchEvaluator | ThreadedDispatcher | AOTEvaluator,
    model: torch.nn.Module,
) -> None:
    if isinstance(evaluator, (ThreadedBatchEvaluator, ThreadedDispatcher)):
        evaluator.update_model(model)
    elif isinstance(evaluator, AOTEvaluator):
        evaluator.load_weights(model.state_dict())
    else:
        evaluator.model = model


def _configure_shared_compile_cache(*, cache_dir: Path) -> None:
    """Point TorchInductor/Triton caches at a shared worker cache root.

    Uses forced env var assignment (not setdefault) so it overrides the
    system default /tmp location.  All workers sharing the same cache_dir
    reuse autotuned kernels and FX graph compilations.
    """
    compile_cache_root = cache_dir / "compile_cache"
    inductor_dir = compile_cache_root / "torchinductor"
    triton_dir = compile_cache_root / "triton"
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)
  # Enable FX graph cache so compiled graphs persist across restarts.
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"


def _configure_worker_torch_env() -> None:
    """Set PyTorch / CUDA env tuning that must precede any imports of torch ops.

    - PYTORCH_CUDA_ALLOC_CONF: hand unused VRAM back to the driver so multiple
      compiled workers can share one GPU.
    - TORCH_COMPILE_THREADS: cap inductor's compile-worker fan-out (each
      subprocess can accumulate >1GB of compiled kernels).
    - TF32: enable for fp32 ops outside autocast BF16 scope.
    """
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_COMPILE_THREADS", "1")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _setup_worker_logging(args) -> logging.Logger:
    """Attach a file handler to the parent ``chess_anti_engine`` logger.

    We attach to the parent (not ``chess_anti_engine.worker``) so DEBUG lines
    from sibling submodules (mcts.gumbel_c, inference, etc.) propagate to the
    same log file — gumbel profile lines are emitted by mcts and would
    otherwise be invisible.
    """
    log = logging.getLogger("chess_anti_engine.worker")
    if not args.log_file:
        return log
    lvl = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }.get(str(args.log_level).lower(), logging.INFO)
    parent = logging.getLogger("chess_anti_engine")
    parent.setLevel(lvl)
    fh = logging.FileHandler(str(args.log_file))
    fh.setLevel(lvl)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    parent.addHandler(fh)
    log.setLevel(lvl)
    return log


# Args whose CLI value (when None/falsy) falls back to cfg.get(<key>).
# Order is the original assignment order so server_url retains its hardcoded fallback.
_OR_FALLBACK_FIELDS: tuple[str, ...] = (
    "trial_id", "username", "stockfish_path", "shared_cache_dir",
)

# Args whose YAML value should override the CLI default whenever the YAML key
# is present. Each entry is (cli_field, yaml_key, cast).
_TYPED_OVERRIDE_FIELDS: tuple[tuple[str, str, type], ...] = (
    ("upload_target_positions", "upload_target_positions", int),
    ("upload_flush_seconds", "upload_flush_seconds", float),
)


def _merge_cli_with_yaml_defaults(args, cfg: dict) -> None:
    """Fill missing CLI fields from worker.yaml. CLI always wins."""
    args.server_url = (
        args.server_url or cfg.get("server_url") or "http://127.0.0.1:45453"
    )
    for field_name in _OR_FALLBACK_FIELDS:
        setattr(args, field_name, getattr(args, field_name) or cfg.get(field_name))

    if args.password_file is None and cfg.get("password_file"):
        args.password_file = str(cfg.get("password_file"))

    for flag in ("self_update", "stockfish_from_server"):
        if not bool(getattr(args, flag)) and bool(cfg.get(flag, False)):
            setattr(args, flag, True)

    if args.sf_workers is None:
        args.sf_workers = int(cfg.get("sf_workers", 1))
    if args.games_per_batch is None and "games_per_batch" in cfg:
        args.games_per_batch = int(cfg["games_per_batch"])

    for cli_field, yaml_key, caster in _TYPED_OVERRIDE_FIELDS:
        if yaml_key in cfg:
            setattr(args, cli_field, caster(cfg[yaml_key]))


def _resolve_worker_password(args, cfg: dict) -> str:
    """CLI > --password-file > config password > prompt.

    --password-file (CLI) wins over saved cfg password so that a fresh session
    with a rotated password loads correctly.
    """
    if args.password_file:
        try:
            return Path(str(args.password_file)).read_text(encoding="utf-8").splitlines()[0].strip()
        except Exception as e:
            raise SystemExit(f"Failed to read --password-file {args.password_file!r}: {e}") from e
    if args.password is not None:
        return str(args.password)
    cfg_pw = cfg.get("password")
    if cfg_pw is not None:
        return str(cfg_pw)
    return getpass.getpass("Password: ")


def main() -> None:
    _configure_worker_torch_env()
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
    ap.add_argument("--inference-slot-name", type=str, default=None,
                    help="Shared-memory slot name for slot-based inference broker.")
    ap.add_argument("--inference-slot-max-batch", type=int, default=256,
                    help="Max batch size for inference slot (must match broker).")
    ap.add_argument(
        "--compile-inference",
        action="store_true",
        help="Enable torch.compile for worker-side inference/search models on CUDA.",
    )
    ap.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode (reduce-overhead, max-autotune, default).",
    )
    ap.add_argument(
        "--inference-fp8",
        action="store_true",
        help="Quantize transformer block linears to FP8 before compile (requires torchao).",
    )

    ap.add_argument(
        "--aot-dir",
        type=str,
        default=None,
        help="Directory with pre-compiled AOTInductor .pt2 packages. "
             "If set, uses AOTEvaluator instead of torch.compile.",
    )

    ap.add_argument(
        "--threaded-selfplay",
        action="store_true",
        help="Use multi-threaded selfplay with shared GPU.",
    )
    ap.add_argument(
        "--selfplay-threads",
        type=int,
        default=16,
        help="Number of selfplay threads (with --threaded-selfplay).",
    )
    ap.add_argument(
        "--threaded-dispatcher",
        action="store_true",
        help="With --threaded-selfplay: use single-consumer ThreadedDispatcher "
             "(cudagraph-safe) instead of ThreadedBatchEvaluator.",
    )
    ap.add_argument(
        "--dispatcher-batch-wait-ms",
        type=float,
        default=1.0,
        help="ThreadedDispatcher batching window in ms (latency vs batch fill).",
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
    ap.add_argument(
        "--upload-max-buffered-positions",
        type=int,
        default=5000,
        help="Hard cap on in-memory buffered positions. New game batches are dropped with a warning once this is exceeded — a backstop for when disk flush keeps failing.",
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

    log = _setup_worker_logging(args)
    log.info("worker starting version=%s protocol=%s", str(PACKAGE_VERSION), int(PROTOCOL_VERSION))

    work_dir = Path(args.work_dir)
    cfg_path = Path(args.config) if args.config is not None else (work_dir / "worker.yaml")
    cfg = load_worker_config(cfg_path)

  # Remember which knobs were explicitly pinned on the CLI.
    pinned_games_per_batch_cli = args.games_per_batch is not None

    _merge_cli_with_yaml_defaults(args, cfg)

    if args.username is None:
        raise SystemExit("--username is required (or set username in worker config)")
    if args.stockfish_path is None and not bool(args.stockfish_from_server):
        raise SystemExit(
            "--stockfish-path is required (or set stockfish_path in worker config), unless --stockfish-from-server is enabled"
        )

    if args.calibrate:
        args.auto_tune = True
        args.save_config = True

  # --allow-overrides is gated by an env var on top of the CLI flag so
  # drive-by usage can't override server-managed strength knobs.
    if bool(getattr(args, "allow_overrides", False)) and os.environ.get("CHESS_ANTI_ENGINE_DEBUG_OVERRIDES") != "1":
        raise SystemExit(
            "--allow-overrides is disabled by default. Set CHESS_ANTI_ENGINE_DEBUG_OVERRIDES=1 to enable."
        )

    try:
        import requests
    except Exception as e:  # pragma: no cover
        raise RuntimeError("worker requires requests; install with pip install -e '.[worker]' ") from e

    args.password = _resolve_worker_password(args, cfg)
    if bool(args.save_config) and bool(args.save_password):
        cfg["password"] = str(args.password)

    session = WorkerSession(
        args,
        cfg=cfg,
        cfg_path=cfg_path,
        log=log,
        pinned_games_per_batch_cli=pinned_games_per_batch_cli,
        requests_mod=requests,
    )
    session.run()


class WorkerSession:
    """Manages a worker's lifecycle: poll manifest -> sync assets -> play -> upload."""

    def __init__(
        self,
        args,
        *,
        cfg: dict,
        cfg_path: Path,
        log: logging.Logger,
        pinned_games_per_batch_cli: bool,
        requests_mod,
    ) -> None:
        self.args = args
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.log = log
        self.pinned_games_per_batch_cli = pinned_games_per_batch_cli
        self._requests = requests_mod
        self._auth: tuple[str, str] = (str(args.username), str(args.password))

        self.server = str(args.server_url).rstrip("/")
        trial_id = str(args.trial_id).strip() if args.trial_id is not None else ""
        self.fixed_trial_id = str(trial_id)
        self.leased_trial_id = str(trial_id)
        self.trial_api_prefix = f"/v1/trials/{self.leased_trial_id}" if self.leased_trial_id else "/v1"
        self.lease_id = ""

        work_dir = Path(args.work_dir)
        self.cache_dir = Path(str(args.shared_cache_dir)) if args.shared_cache_dir is not None else (work_dir / "cache")
        self.shared_cache_enabled = args.shared_cache_dir is not None
        _configure_shared_compile_cache(cache_dir=self.cache_dir)
        shard_dir = work_dir / "shards"
        self.pending_dir = shard_dir / "pending"
        self.uploaded_dir = shard_dir / "uploaded"

        arena_dir = work_dir / "arena"
        self.arena_pending_dir = arena_dir / "pending"
        self.arena_uploaded_dir = arena_dir / "uploaded"

        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.uploaded_dir.mkdir(parents=True, exist_ok=True)
        self.arena_pending_dir.mkdir(parents=True, exist_ok=True)
        self.arena_uploaded_dir.mkdir(parents=True, exist_ok=True)

  # Local throughput override that can be tuned and persisted.
        self.games_per_batch_local = int(args.games_per_batch) if args.games_per_batch is not None else None
        self.worker_id = str(cfg.get("worker_id") or "").strip()
        if not self.worker_id:
            self.worker_id = uuid.uuid4().hex
            cfg["worker_id"] = self.worker_id

        self.machine_id = str(cfg.get("machine_name") or "").strip() or socket.gethostname()

  # Save initial config (best effort).
        self._persist_cfg()

        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.worker_info = _collect_worker_info(
            device=str(self.device),
        )
        self.worker_info["worker_id"] = str(self.worker_id)

        self.rng = np.random.default_rng(int(args.seed))

  # Validate server-managed overrides once at startup (args never change).
        if not bool(args.allow_overrides):
            _server_managed_keys = [
                "max_plies", "mcts", "mcts_simulations", "playout_cap_fraction",
                "fast_simulations", "opening_book_prob", "opening_book_max_plies",
                "opening_book_max_games", "random_start_plies", "sf_nodes",
                "sf_multipv", "sf_policy_temp", "sf_policy_label_smooth",
                "timeout_adjudication_threshold", "temperature",
                "temperature_decay_start_move", "temperature_decay_moves",
                "temperature_endgame",
                "syzygy_path", "syzygy_rescore_policy", "syzygy_adjudicate",
                "syzygy_adjudicate_fraction", "syzygy_in_search",
            ]
            overridden = [k for k in _server_managed_keys if getattr(args, k, None) is not None]
            if overridden:
                raise SystemExit(
                    "This worker is configured for server-managed strength knobs. "
                    f"Remove selfplay/strength flags ({', '.join(overridden)}), or pass --allow-overrides for debugging."
                )

  # Engine: initialize with placeholder settings; we will align nodes from manifest each loop.
  # MultiPV and SyzygyPath are set at init time; reinitialize when they change.
        self.sf: StockfishPool | StockfishUCI | None = None
        self.sf_multipv_active: int | None = None
        self.sf_syzygy_path_active: str | None = None

        self.last_model_sha = None
        self.last_ob_sha: str | None = None
        self.last_ob2_sha: str | None = None
        self.last_sf_sha: str | None = None
        self.model_cfg_active: ModelConfig | None = None
        self.model = None
        self._direct_evaluator: (
            DirectGPUEvaluator | ThreadedBatchEvaluator | ThreadedDispatcher | AOTEvaluator | None
        ) = None
        self.inference_client = self._make_inference_client()
        self.pause_selfplay_active = False
        self.manifest_state = _MANIFEST_STATE_ACTIVE
        self.manifest_state_elapsed_s: float | None = None
        self.upload_buf = _BufferedUpload()
        self.last_successful_send_s = time.time()

        self.last_best_sha = None
        self.best_model = None

  # Opening book paths (set by _sync_opening_books each iteration).
        self.opening_book_path: str | None = None
        self.opening_book_path_2: str | None = None

  # Per-iteration state (set during each loop iteration).
        self.model_sha = ""
        self.model_step = 0
        self._saw_completed_game = False
        self._stop_selfplay = False
        self._upload_buf_lock: threading.Lock | None = None  # set when threaded
        self._last_manifest_poll_s: float = 0.0
  # Manifest-watch state (lazy-init in _check_model_update via getattr fallback).
        self._manifest_path: Path | None = None
        self._manifest_mtime: float | None = None
        self._active_reco: dict | None = None
        self._evaluator_model_id: int | None = None

    def _build_evaluator(
        self, model: torch.nn.Module,
    ) -> DirectGPUEvaluator | ThreadedBatchEvaluator | ThreadedDispatcher | AOTEvaluator:
        device = str(self.device)
        if self.args.threaded_selfplay and self.args.threaded_dispatcher:
            disp_compile = (
                str(self.args.compile_mode) if self.args.compile_inference else None
            )
            return ThreadedDispatcher(
                model, device=device, max_batch=4096,
                batch_wait_ms=float(self.args.dispatcher_batch_wait_ms),
                compile_mode=disp_compile,
            )
        if self.args.threaded_selfplay:
            return ThreadedBatchEvaluator(model, device=device, max_batch=4096, min_batch=256)
        if self.args.aot_dir:
            aot = AOTEvaluator(self.args.aot_dir, device=device, max_batch=4096)
            aot.load_weights(model.state_dict())
            return aot
        return DirectGPUEvaluator(model, device=device, max_batch=4096, n_slots=2)

    def _stop_fn(self) -> bool:
        """Called every ply by play_batch.  Return True to exit continuous selfplay."""
        return self._stop_selfplay

    def run(self) -> None:
        """Main loop."""
        try:
            while True:
                manifest = self._poll_manifest()
                if manifest is None:
                    continue
                self._sync_assets(manifest)
                if not self.model_sha:
  # Model download failed (mid-publish race); retry next poll.
                    time.sleep(float(self.args.poll_seconds))
                    continue
                task = manifest.get("task") or {"type": "selfplay"}
                task_type = str(task.get("type", "selfplay")).lower()
                if task_type == "arena":
                    self._run_arena(manifest, task)
                else:
                    self._run_selfplay(manifest)
        finally:
            self._cleanup()

  # -- Helper methods -------------------------------------------------------

    def _server_url_for(self, endpoint: str) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if endpoint.startswith("/"):
            return self.server + endpoint
        return self.server + "/" + endpoint

    def _persist_cfg(self) -> None:
        if not bool(self.args.save_config):
            return
        self.cfg["server_url"] = str(self.args.server_url)
        if self.fixed_trial_id:
            self.cfg["trial_id"] = self.fixed_trial_id
        else:
            self.cfg.pop("trial_id", None)
        self.cfg["username"] = str(self.args.username)
        self.cfg["self_update"] = bool(self.args.self_update)
        self.cfg["stockfish_from_server"] = bool(self.args.stockfish_from_server)
        if self.args.password_file:
            self.cfg["password_file"] = str(self.args.password_file)
        else:
            self.cfg.pop("password_file", None)
        if not bool(self.args.stockfish_from_server):
            self.cfg["stockfish_path"] = str(self.args.stockfish_path)
        else:
            self.cfg.pop("stockfish_path", None)
        if self.args.shared_cache_dir:
            self.cfg["shared_cache_dir"] = str(self.args.shared_cache_dir)
        else:
            self.cfg.pop("shared_cache_dir", None)
        self.cfg["sf_workers"] = int(self.args.sf_workers)
        if self.games_per_batch_local is not None:
            self.cfg["games_per_batch"] = int(self.games_per_batch_local)
        self.cfg["upload_target_positions"] = int(self.args.upload_target_positions)
        self.cfg["upload_flush_seconds"] = float(self.args.upload_flush_seconds)
        if not bool(self.args.save_password):
            self.cfg.pop("password", None)
        save_worker_config(self.cfg_path, self.cfg)

    def _make_inference_client(self):
        if str(self.args.inference_slot_name or "").strip():
            return SlotInferenceClient(
                slot_name=str(self.args.inference_slot_name),
                max_batch=int(self.args.inference_slot_max_batch),
            )
        return None

    def _upload_pending_shards(self, *, default_elapsed_s: float | None = None) -> float | None:
        last_uploaded_at: float | None = None
        pending: list[Path] = [
            p for p in self.pending_dir.iterdir()
            if not p.name.startswith("_tmp_")
            and not p.name.startswith("._")
            and p.suffix == LOCAL_SHARD_SUFFIX
        ]
        for sp in sorted(pending):
            elapsed_s = default_elapsed_s
            elapsed_path = _pending_elapsed_path(sp)
            if elapsed_path.exists():
                try:
                    elapsed_s = float(elapsed_path.read_text(encoding="utf-8").strip())
                except Exception:
                    elapsed_s = default_elapsed_s
            upload_name, payload = pack_shard_for_upload(sp)
            files = {"file": (upload_name, payload, "application/x-tar")}
            try:
                r = self._requests.post(
                    self._server_url_for(self.trial_api_prefix + "/upload_shard"),
                    files=files,
                    auth=(str(self.args.username), str(self.args.password)),
                    headers={
                        **_worker_headers(machine_id=self.machine_id),
                        **(
                            {"X-CAE-Worker-Lease-ID": str(self.lease_id)}
                            if str(self.lease_id).strip()
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
            finally:
                payload.close()
            if r.status_code == 200:
                delete_shard_path(sp)
                elapsed_path.unlink(missing_ok=True)
                self.last_successful_send_s = time.time()
                last_uploaded_at = float(self.last_successful_send_s)
            else:
                break
        return last_uploaded_at

    def _load_and_compile_model(self, path: Path, cfg: ModelConfig, *, label: str, sha_short: str) -> torch.nn.Module:
        """Build model, load checkpoint, optionally compile."""
        model = build_model(cfg)
        ckpt = torch.load(str(path), map_location="cpu")
        sd = ckpt.get("model", ckpt)
        load_state_dict_tolerant(model, sd, label=label)
        model.to(self.device)
        model.eval()
  # Selfplay only needs policy_own + wdl; skip 8 unused heads.
        if hasattr(model, "_inference_only"):
            setattr(model, "_inference_only", True)
        # ThreadedDispatcher compiles on its own thread so cudagraph TLS lives
        # where the forwards happen. Pre-compiling here would cause the
        # dispatcher's first forward to fall back to eager.
        if self.args.compile_inference and not getattr(self.args, "threaded_dispatcher", False):
            compile_t0 = time.time()
            self.log.info("compile starting %s sha=%s", label, sha_short)
            _compile_mode = str(self.args.compile_mode)
            model = _maybe_compile_inference_model(model, device=str(self.device), mode=_compile_mode, use_fp8=bool(getattr(self.args, "inference_fp8", False)))
            self.log.info("compile finished %s sha=%s elapsed_s=%.2f", label, sha_short, float(time.time() - compile_t0))
        return model

    @overload
    def _resolve_reco(self, reco: dict, key: str, default: Any) -> float: ...
    @overload
    def _resolve_reco(self, reco: dict, key: str, default: Any, cast: Callable[[Any], _ResolveT]) -> _ResolveT: ...
    def _resolve_reco(self, reco: dict, key: str, default: Any, cast: Callable[[Any], Any] = float) -> Any:
        """CLI overrides server recommendation; fall back to reco then default."""
        v = getattr(self.args, key, None)
        return cast(v) if v is not None else cast(reco.get(key, default))

    def _reco_changed(self, manifest: dict, *, source_tag: str) -> bool:
        """Return True (and request session restart) if reco knobs differ from active."""
        new_reco = manifest.get("recommended_worker") or {}
        active = getattr(self, "_active_reco", None)
        if active is None:
            return False
        new_snap = {k: new_reco.get(k) for k in self._RECO_RESTART_KEYS}
        if new_snap == active:
            return False
        self.log.info("recommended_worker changed (%s), restarting selfplay session", source_tag)
        self._stop_selfplay = True
        return True

    def _periodic_manifest_poll(self) -> None:
        """Tier 0: every 30s, re-poll for task/pause/trial-reassign/reco changes."""
        try:
            old_tid = self.leased_trial_id
            manifest = self._poll_manifest()
            if manifest is None:
  # _poll_manifest returns None on pause or transient failure.
  # If paused, stop continuous selfplay so the outer loop can
  # re-poll properly.  Transient failures are retried next cycle.
                if self.pause_selfplay_active:
                    self._stop_selfplay = True
                return
            task_type = str((manifest.get("task") or {}).get("type", "selfplay")).lower()
            if task_type != "selfplay":
                self._stop_selfplay = True
                return
  # Trial reassignment: stop selfplay so the outer loop can re-lease.
            if self.leased_trial_id != old_tid:
                self._stop_selfplay = True
                return
            if self._reco_changed(manifest, source_tag="poll"):
                return
            self._sync_assets(manifest)
        except Exception:
            pass

    def _resolve_local_manifest_path(self) -> Path | None:
        """Lazily compute the on-disk manifest path; cached on self."""
        manifest_path = getattr(self, "_manifest_path", None)
        if manifest_path is not None:
            return manifest_path
        try:
  # work_dir is .../trials/{trial_id}/workers/worker_XX
            trials_dir = Path(self.args.work_dir).parent.parent.parent
            tid = self.leased_trial_id or self.fixed_trial_id or ""
            if not tid:
                return None
            manifest_path = trials_dir / tid / "publish" / "manifest.json"
            if not manifest_path.parent.exists():
                return None  # Not a local worker — between-batch poll handles it
            self._manifest_path = manifest_path
            self._manifest_mtime = None
            return manifest_path
        except Exception:
            return None

    def _swap_model_from_manifest(self, manifest: dict) -> None:
        """Tier 2 inner: download new SHA if changed, hot-swap model, flush old shards."""
        new_sha = str(manifest.get("model", {}).get("sha256", ""))
        if not new_sha or new_sha == self.model_sha:
            return
        model_info = manifest.get("model") or {}
        model_path = self.cache_dir / f"model_{new_sha}.pt"
        if (not model_path.exists()) or (_sha256_file(model_path) != new_sha):
            _download_and_verify_shared(
                self._server_url_for(str(model_info.get("endpoint") or (self.trial_api_prefix + "/model"))),
                out_path=model_path,
                expected_sha256=new_sha,
                headers=_worker_headers(),
            )
        mc = manifest.get("model_config") or self.model_cfg_active
        _cfg = mc if isinstance(mc, ModelConfig) else self.model_cfg_active
        if _cfg is None:
            raise RuntimeError("no model_config available for worker-model load")
        self.model = self._load_and_compile_model(
            model_path, _cfg,
            label="worker-model", sha_short=str(new_sha)[:8],
        )
        if self._direct_evaluator is not None:
            _sync_evaluator_to_model(self._direct_evaluator, self.model)
            self._evaluator_model_id = id(self.model)
  # Flush upload buffer tagged with old SHA. Lock (when set by
  # threaded selfplay) protects against concurrent _on_completed_game
  # calls; in single-threaded mode it's None and nullcontext is a no-op.
        buf_lock = self._upload_buf_lock
        now = time.time()
        with buf_lock if buf_lock is not None else nullcontext():
            if self.upload_buf.positions > 0:
                _flush_upload_buffer_to_pending(
                    pending_dir=self.pending_dir, username=str(self.args.username),
                    buf=self.upload_buf, now_s=now,
                )
                self._upload_pending_shards(default_elapsed_s=0.0)
            self.model_sha = new_sha
            self.model_step = int(manifest.get("trainer_step") or 0)
            self.last_model_sha = new_sha
        self.log.info("mid-batch model switch sha=%s", str(new_sha)[:8])

    def _check_model_update(self) -> None:
        """Check for new model between moves (called every ply by play_batch).

        Three-tier check:
        1. Every 30s: re-poll manifest for task/pause changes (sets _stop_selfplay)
        2. Every call: stat() the local manifest file mtime (~1µs)
        3. Only if mtime changed: read manifest, download model, swap
        """
        _now = time.time()
        if _now - self._last_manifest_poll_s > 30.0:
            self._last_manifest_poll_s = _now
            self._periodic_manifest_poll()
            if self._stop_selfplay:
                return

        manifest_path = self._resolve_local_manifest_path()
        if manifest_path is None:
            return
        try:
            mtime = manifest_path.stat().st_mtime
        except OSError:
            return
        if mtime == self._manifest_mtime:
            return
        self._manifest_mtime = mtime

  # Tier 2: mtime changed — read and potentially swap model / reco
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if self._reco_changed(manifest, source_tag="mtime"):
                return
            self._swap_model_from_manifest(manifest)
        except Exception as _exc:
            self.log.debug("mid-batch model check failed: %s", _exc)

    def _on_completed_game(self, game_batch) -> None:
        now_s = time.time()
        self._saw_completed_game = True
        _buffer_add_completed_game(
            buf=self.upload_buf,
            game_batch=game_batch,
            now_s=now_s,
            model_sha=self.model_sha,
            model_step=self.model_step,
            max_positions=int(self.args.upload_max_buffered_positions),
        )
        shard_path, elapsed_s = _maybe_flush_upload_buffer(
            pending_dir=self.pending_dir,
            username=str(self.args.username),
            buf=self.upload_buf,
            now_s=now_s,
            last_send_s=self.last_successful_send_s,
            target_positions=int(self.args.upload_target_positions),
            flush_seconds=float(self.args.upload_flush_seconds),
            force=False,
        )
        if shard_path is not None:
            uploaded_at = self._upload_pending_shards(default_elapsed_s=float(elapsed_s))
            if uploaded_at is not None:
                self.last_successful_send_s = float(uploaded_at)

  # -- Lifecycle methods ----------------------------------------------------

    def _negotiate_lease(self) -> bool:
        """POST /v1/lease_trial; update self.lease_id/api_prefix/leased_trial_id.

        Returns True on success, False to signal 'sleep and retry'.
        """
        body: dict[str, object] = {"worker_info": self.worker_info}
        if self.lease_id:
            body["lease_id"] = str(self.lease_id)
        if self.leased_trial_id:
            body["trial_id"] = str(self.leased_trial_id)
        r_lease = self._requests.post(
            self._server_url_for("/v1/lease_trial"),
            json=body,
            auth=self._auth,
            headers=_worker_headers(),
            timeout=30.0,
        )
        if r_lease.status_code != 200:
            time.sleep(float(self.args.poll_seconds))
            return False
        lease = r_lease.json()
        new_trial_id = str(lease.get("trial_id") or "").strip()
        if new_trial_id != self.leased_trial_id:
            self.log.info("leased trial assignment changed: %s -> %s", self.leased_trial_id or "<root>", new_trial_id or "<root>")
        self.leased_trial_id = new_trial_id
        self.trial_api_prefix = str(lease.get("api_prefix") or "/v1").strip() or "/v1"
        self.lease_id = str(lease.get("lease_id") or "").strip()
        return True

    def _upload_pending_arena_results(self) -> None:
        """Drain arena_pending_dir to the server; quarantine bad files."""
        upload_url = self._server_url_for(self.trial_api_prefix + "/upload_arena_result")
        for jp in sorted(self.arena_pending_dir.glob("*.json")):
            try:
                payload = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
  # bad local file; quarantine to uploaded to avoid retry storms
                jp.replace(self.arena_uploaded_dir / jp.name)
                continue
            r = self._requests.post(
                upload_url, json=payload, auth=self._auth,
                headers=_worker_headers(), timeout=60.0,
            )
            if r.status_code == 200:
                jp.unlink(missing_ok=True)
            else:
                break

    def _install_worker_wheel(self, ww: dict, version_source: dict) -> None:
        """Download + verify + pip install + exec-restart. Does not return."""
        wheel_version = str(ww.get("version") or version_source.get("server_version") or "0.0.0")
        sha = str(ww.get("sha256"))
        endpoint = str(ww.get("endpoint"))
        wheel_path = self.cache_dir / f"worker_{sha}.whl"
        self.log.warning("self-update: installing worker wheel version=%s sha=%s", wheel_version, sha)
        _download_and_verify_shared(
            self._server_url_for(endpoint),
            out_path=wheel_path,
            expected_sha256=sha,
            headers=_worker_headers(),
        )
        _pip_install_wheel(wheel_path)
        _restart_process()

    def _self_update_enabled(self) -> bool:
        return bool(self.args.self_update) and os.environ.get("CAE_SELF_UPDATED") != "1"

    def _handle_upgrade_required(self, r: requests.Response) -> None:
        """Server returned 426. Try self-update (no return) or raise SystemExit."""
        if self._self_update_enabled():
            r2 = self._requests.get(self._server_url_for(self.trial_api_prefix + "/update_info"), timeout=30.0)
            if r2.status_code != 200:
                raise SystemExit(f"Upgrade required but could not fetch update info for self-update: {r2.text}")
            update_info = r2.json()
            ww = _extract_worker_wheel(update_info)
            if ww is not None:
                self._install_worker_wheel(ww, update_info)
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = None
        raise SystemExit(f"Worker is not compatible with server: {detail or r.text}")

    def _check_manifest_compat(self, manifest: dict) -> _ManifestCompat:
        """Validate manifest; raise SystemExit on encoding mismatch.

        Returns the compat struct (mismatches + raw values) so the caller
        can attempt self-update before deciding whether to SystemExit on
        protocol/version. Encoding mismatches are unrecoverable so they
        raise here.
        """
        req_proto = manifest.get("protocol_version")
        protocol_mismatch = False
        if req_proto is not None:
            try:
                protocol_mismatch = int(req_proto) != int(PROTOCOL_VERSION)
            except Exception:
                raise SystemExit(f"Bad protocol_version in manifest: {req_proto!r}")
        min_v = manifest.get("min_worker_version")
        version_too_old = bool(min_v is not None and version_lt(PACKAGE_VERSION, str(min_v)))
        enc = manifest.get("encoding") or {}
        if "policy_size" in enc and int(enc.get("policy_size") or 0) != int(POLICY_SIZE):
            raise SystemExit(f"policy_size mismatch: worker={POLICY_SIZE} server={enc.get('policy_size')}")
        if "input_planes" in enc and int(enc.get("input_planes") or 0) != 146:
            raise SystemExit(f"input_planes mismatch: worker expects 146, server={enc.get('input_planes')}")
        return _ManifestCompat(
            protocol_mismatch=protocol_mismatch, version_too_old=version_too_old,
            req_proto=req_proto, min_worker_version=min_v,
        )

    def _maybe_self_update_from_manifest(self, manifest: dict, compat: _ManifestCompat) -> None:
        """If manifest carries a worker_wheel and we're behind, install it (no return)."""
        if not self._self_update_enabled():
            return
        ww = _extract_worker_wheel(manifest)
        if ww is None:
            return
        wheel_version = str(ww.get("version") or manifest.get("server_version") or "0.0.0")
        if compat.protocol_mismatch or compat.version_too_old or version_lt(PACKAGE_VERSION, wheel_version):
            self._install_worker_wheel(ww, manifest)

    def _check_pause_selfplay(self, manifest: dict) -> bool:
        """Return True (and sleep) when server has paused selfplay."""
        reco = manifest.get("recommended_worker") or {}
        backpressure = manifest.get("backpressure") or {}
        task = manifest.get("task") or {"type": "selfplay"}
        pause_selfplay = False
        pause_reason = ""
        if str(task.get("type", "selfplay")).lower() == "selfplay":
            pause_selfplay = bool(reco.get("pause_selfplay", False))
            pause_reason = str(reco.get("pause_reason") or "")
            if (not pause_selfplay) and isinstance(backpressure, dict):
                pause_selfplay = bool(backpressure.get("pause_selfplay", False))
                pause_reason = str(backpressure.get("pause_reason") or pause_reason)
        if not pause_selfplay:
            if self.pause_selfplay_active:
                self.log.info("selfplay pause cleared by server")
                self.pause_selfplay_active = False
            return False
        if not self.pause_selfplay_active:
            self.log.info("selfplay paused by server%s", f": {pause_reason}" if pause_reason else "")
            self.pause_selfplay_active = True
        sleep_s = max(0.1, float(self.args.poll_seconds))
        time.sleep(sleep_s)
        self.manifest_state = _MANIFEST_STATE_PAUSED
        self.manifest_state_elapsed_s = sleep_s
        return True

    def _poll_manifest(self) -> dict | None:
        """Lease negotiation + manifest fetch + version checks + self-update.

        Returns the manifest dict, or None to signal 'sleep and retry'.
        """
        if self.inference_client is None:
            self.inference_client = self._make_inference_client()

        if not self.fixed_trial_id and not self._negotiate_lease():
            return None

        self._upload_pending_shards(default_elapsed_s=float(self.cfg.get("_last_batch_elapsed_s", 0.0) or 0.0))
        self._upload_pending_arena_results()

        r = self._requests.get(
            self._server_url_for(self.trial_api_prefix + "/manifest"),
            timeout=30.0,
            headers=_manifest_poll_headers(
                worker_id=self.worker_id,
                lease_id=self.lease_id,
                state=self.manifest_state,
                elapsed_s=self.manifest_state_elapsed_s,
            ),
        )
        self.manifest_state = _MANIFEST_STATE_ACTIVE
        self.manifest_state_elapsed_s = None
        if r.status_code == 426:
            self._handle_upgrade_required(r)
        if r.status_code != 200:
            time.sleep(float(self.args.poll_seconds))
            return None

        manifest = r.json()
        compat = self._check_manifest_compat(manifest)
        self._maybe_self_update_from_manifest(manifest, compat)
  # Self-update may not have returned; if we're still here and behind, exit.
        if compat.protocol_mismatch:
            raise SystemExit(f"Protocol mismatch: worker={PROTOCOL_VERSION} server_required={compat.req_proto}")
        if compat.version_too_old:
            raise SystemExit(f"Worker too old: worker={PACKAGE_VERSION} min_required={compat.min_worker_version}")

        if self._check_pause_selfplay(manifest):
            return None
        return manifest

    def _sync_assets(self, manifest: dict) -> None:
        """Sync model, opening books, and stockfish from manifest."""
        self._sync_model(manifest)
        self._sync_opening_books(manifest)

    def _flush_pre_swap_buffer_if_stale(self, *, model_sha: str, model_step: int) -> None:
        """If buffered samples were generated under a different model_sha/step,
        flush them to pending before we swap the active model."""
        if self.upload_buf.positions <= 0:
            return
        if (
            str(self.upload_buf.model_sha or "") == str(model_sha)
            and int(self.upload_buf.model_step or 0) == int(model_step)
        ):
            return
        shard_path, elapsed_s = _flush_upload_buffer_to_pending(
            pending_dir=self.pending_dir,
            username=str(self.args.username),
            buf=self.upload_buf,
            now_s=time.time(),
        )
        if shard_path is None:
            return
        uploaded_at = self._upload_pending_shards(default_elapsed_s=float(elapsed_s))
        if uploaded_at is not None:
            self.last_successful_send_s = float(uploaded_at)

    def _ensure_local_model_at_sha(self, *, model_info: dict, model_sha: str) -> Path | None:
        """Ensure cache_dir has model_{sha}.pt with matching sha256. Returns the
        path on success, None on download race (caller should re-poll). The
        server serves a stable /v1/model endpoint whose contents can change
        whenever the learner publishes a new model — a slightly stale manifest
        can name a different sha than what's currently published."""
        model_path = self.cache_dir / f"model_{model_sha}.pt"
        if model_path.exists() and _sha256_file(model_path) == model_sha:
            return model_path
        try:
            _download_and_verify_shared(
                self._server_url_for(str(model_info.get("endpoint") or (self.trial_api_prefix + "/model"))),
                out_path=model_path,
                expected_sha256=model_sha,
                headers=_worker_headers(),
            )
        except Exception as e:
            self.log.warning(
                "model download failed (likely mid-publish race). Will re-poll manifest: %s",
                e,
            )
            model_path.unlink(missing_ok=True)
            return None
        return model_path

    def _sync_model(self, manifest: dict) -> None:
        """Download + build + load + compile model if SHA changed."""
        task_type = str((manifest.get("task") or {"type": "selfplay"}).get("type", "selfplay")).lower()
        need_local_model = self.inference_client is None or task_type == "arena"

        model_info = manifest.get("model") or {}
        model_sha = str(model_info.get("sha256") or "")
        if not model_sha:
            return
        model_step = int(manifest.get("trainer_step") or 0)

  # Store for use by other methods this iteration.
        self.model_sha = model_sha
        self.model_step = model_step

        self._flush_pre_swap_buffer_if_stale(model_sha=model_sha, model_step=model_step)

        if not (need_local_model and model_sha != self.last_model_sha):
            return

        self.log.info("switching to latest model sha=%s", model_sha)
        model_path = self._ensure_local_model_at_sha(model_info=model_info, model_sha=model_sha)
        if model_path is None:
  # Signal caller to sleep and retry by clearing model_sha.
            self.model_sha = ""
            return

        # Force-off gradient checkpointing: workers run inference only, and
        # grad-ckpt hooks would diverge the compiled graph from the broker
        # paths that hardcode False.
        model_cfg = dataclasses.replace(
            model_config_from_manifest_dict(manifest.get("model_config") or {}),
            use_gradient_checkpointing=False,
        )
        self.model_cfg_active = model_cfg
        self.model = self._load_and_compile_model(
            model_path, model_cfg, label="worker-model", sha_short=str(model_sha)[:8],
        )

        if self.last_model_sha is not None and not self.fixed_trial_id:
  # Reconsider assignment at natural model-boundary checkpoints.
            self.lease_id = ""
            self.leased_trial_id = ""
            self.trial_api_prefix = "/v1"
        self.last_model_sha = model_sha

    def _sync_opening_books(self, manifest: dict) -> None:
        """Download opening books if SHA changed."""
        hdrs = _worker_headers()
        self.opening_book_path, self.last_ob_sha = _download_opening_book(
            manifest, "opening_book", self.cache_dir,
            cache_prefix="opening", default_endpoint="/v1/opening_book",
            server_url_fn=self._server_url_for, headers=hdrs, log=self.log,
            last_sha=self.last_ob_sha,
        )
        self.opening_book_path_2, self.last_ob2_sha = _download_opening_book(
            manifest, "opening_book_2", self.cache_dir,
            cache_prefix="opening2", default_endpoint="/v1/opening_book_2",
            server_url_fn=self._server_url_for, headers=hdrs, log=self.log,
            last_sha=self.last_ob2_sha,
        )

    def _sync_stockfish(
        self,
        manifest: dict,
        sf_nodes: int,
        sf_multipv: int,
        syzygy_path: str | None = None,
    ) -> str:
        """Download SF binary + (re)init engine if multipv or syzygy path changed.

        Returns the resolved stockfish_path.
        """
        stockfish_path = str(self.args.stockfish_path) if self.args.stockfish_path is not None else ""
        if self.args.stockfish_from_server:
            sf_rec = manifest.get("stockfish")
            if not isinstance(sf_rec, dict) or not sf_rec.get("endpoint") or not sf_rec.get("sha256"):
                raise SystemExit("--stockfish-from-server enabled but server did not publish stockfish")
            sf_sha = str(sf_rec.get("sha256"))
            sf_endpoint = str(sf_rec.get("endpoint"))
            sf_filename = str(sf_rec.get("filename") or "stockfish")
            sf_cached = self.cache_dir / f"stockfish_{sf_sha}_{sf_filename}"
            if _cached_sha_asset_needs_refresh(path=sf_cached, sha256=sf_sha, last_sha256=self.last_sf_sha):
                self.log.info("downloading stockfish sha=%s filename=%s", sf_sha, sf_filename)
                _download_and_verify_shared(
                    self._server_url_for(sf_endpoint),
                    out_path=sf_cached,
                    expected_sha256=sf_sha,
                    headers=_worker_headers(),
                )
                _ensure_executable(sf_cached)
            self.last_sf_sha = sf_sha
            stockfish_path = str(sf_cached)

  # (Re)initialize engine if multipv or syzygy path changed (both must be set at init time)
        sz_path = syzygy_path or None
        multipv_changed = self.sf_multipv_active is None or int(self.sf_multipv_active) != int(sf_multipv)
        syzygy_changed = self.sf_syzygy_path_active != sz_path
        if self.sf is None or multipv_changed or syzygy_changed:
            if self.sf is not None:
                try:
                    self.sf.close()
                except Exception:
                    pass

            if int(self.args.sf_workers) > 1:
                self.sf = StockfishPool(
                    path=str(stockfish_path),
                    nodes=int(sf_nodes),
                    num_workers=int(self.args.sf_workers),
                    multipv=int(sf_multipv),
                    syzygy_path=sz_path,
                )
            else:
                self.sf = StockfishUCI(
                    str(stockfish_path),
                    nodes=int(sf_nodes),
                    multipv=int(sf_multipv),
                    syzygy_path=sz_path,
                )
            self.sf_multipv_active = int(sf_multipv)
            self.sf_syzygy_path_active = sz_path
        else:
  # update nodes dynamically
            if hasattr(self.sf, "set_nodes"):
                self.sf.set_nodes(int(sf_nodes))

        return stockfish_path

    def _run_arena(self, manifest: dict, task: dict) -> None:
        """Arena match logic."""
        model_sha = self.model_sha

        best_info = manifest.get("best_model") or {}
        best_sha = str(best_info.get("sha256") or "")
        if not best_sha:
            time.sleep(float(self.args.poll_seconds))
            return
        if self.model is None or self.model_cfg_active is None:
            time.sleep(float(self.args.poll_seconds))
            return

        if best_sha != self.last_best_sha:
            self.log.info("arena task: loading best model sha=%s", best_sha)
            best_path = self.cache_dir / f"best_{best_sha}.pt"
            endpoint = str(best_info.get("endpoint") or "/v1/best_model")
            if (not best_path.exists()) or (_sha256_file(best_path) != best_sha):
                try:
                    _download_and_verify_shared(
                        self._server_url_for(endpoint),
                        out_path=best_path,
                        expected_sha256=best_sha,
                        headers=_worker_headers(),
                    )
                except Exception as e:
                    self.log.warning(
                        "best model download failed (likely mid-publish race). Will retry: %s",
                        e,
                    )
                    best_path.unlink(missing_ok=True)
                    time.sleep(float(self.args.poll_seconds))
                    return

            self.best_model = self._load_and_compile_model(
                best_path, self.model_cfg_active, label="worker-best", sha_short=str(best_sha)[:8],
            )
            self.last_best_sha = best_sha

        if self.best_model is None:
            time.sleep(float(self.args.poll_seconds))
            return

        arena_cfg = task.get("arena") or {}
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
            opening_book_path=self.opening_book_path,
            opening_book_prob=opening_book_prob_arena if self.opening_book_path else 0.0,
            opening_book_max_plies=opening_book_max_plies_arena,
            opening_book_max_games=opening_book_max_games_arena,
            random_start_plies=random_start_plies_arena,
        )

        g = max(1, batch_games)
        a_plays_white = [bool(i % 2 == 0) for i in range(g)] if swap_sides else [True] * g

        stats = play_match_batch(
            self.model,
            self.best_model,
            device=str(self.device),
            rng=self.rng,
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
            "worker_username": str(self.args.username),
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

        out = self.arena_pending_dir / f"{ts}_{model_sha[:8]}_vs_{best_sha[:8]}_{stats.games}g.json"
        out.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

        if not self.shared_cache_enabled:
            _prune_cached_models(cache_dir=self.cache_dir, keep_shas={model_sha, best_sha})

        time.sleep(0.1)

  # Fields in recommended_worker that affect gameplay and should trigger
  # a session restart when the trainer updates them between iterations.
    _RECO_RESTART_KEYS = (
        "sf_nodes",
        "opponent_wdl_regret_limit", "mcts_simulations", "fast_simulations",
        "selfplay_fraction",
        # Syzygy knobs affect adjudication + in-search overrides — without a
        # restart, workers keep producing shards under stale TB settings until
        # an unrelated key changes. Flagged by Codex adversarial review.
        "syzygy_path", "syzygy_rescore_policy", "syzygy_adjudicate",
        "syzygy_adjudicate_fraction", "syzygy_in_search",
    )

    def _build_selfplay_configs(self, reco: dict) -> tuple[dict, tuple]:
        """Unpack manifest.recommended_worker into the 5 frozen config dataclasses.

        Returns ``(cfgs, sf_args)`` where ``cfgs`` is a dict keyed by
        ``play_batch`` kwarg names (``opponent``/``temp``/``search``/``opening``/
        ``game``), ready for ``play_batch(..., **cfgs)``; ``sf_args`` is
        ``(sf_nodes, sf_multipv, syzygy_path)`` for the Stockfish bring-up.
        """
        regret_raw = reco.get("opponent_wdl_regret_limit", None)
  # None must propagate (means "no regret limit"); only cast real values.
        regret_limit = float(regret_raw) if regret_raw is not None else None
  # Syzygy is fully manifest-driven so the server operator can tune
  # adjudication behavior live by editing publish/manifest.json. None
  # for path = disabled; fraction 1.0 = always adjudicate when on.
        syzygy_path = reco.get("syzygy_path") or None
        cfgs = {
            "opponent": OpponentConfig(wdl_regret_limit=regret_limit),
            "temp": TemperatureConfig(
                temperature=self._resolve_reco(reco, "temperature", 1.0),
                decay_start_move=self._resolve_reco(reco, "temperature_decay_start_move", 20, int),
                decay_moves=self._resolve_reco(reco, "temperature_decay_moves", 60, int),
                endgame=self._resolve_reco(reco, "temperature_endgame", 0.6),
            ),
            "search": SearchConfig(
                simulations=self._resolve_reco(reco, "mcts_simulations", 50, int),
                mcts_type=self._resolve_reco(reco, "mcts", "puct", str),
                playout_cap_fraction=self._resolve_reco(reco, "playout_cap_fraction", 0.25),
                fast_simulations=self._resolve_reco(reco, "fast_simulations", 8, int),
            ),
            "opening": OpeningConfig(
                opening_book_path=self.opening_book_path,
                opening_book_max_plies=self._resolve_reco(reco, "opening_book_max_plies", 4, int),
                opening_book_max_games=self._resolve_reco(reco, "opening_book_max_games", 200000, int),
                opening_book_prob=self._resolve_reco(reco, "opening_book_prob", 1.0),
                opening_book_path_2=self.opening_book_path_2,
                opening_book_max_plies_2=int(reco.get("opening_book_max_plies_2", 16)),
                opening_book_max_games_2=int(reco.get("opening_book_max_games_2", 200000)),
                opening_book_mix_prob_2=float(reco.get("opening_book_mix_prob_2", 0.0)),
                random_start_plies=self._resolve_reco(reco, "random_start_plies", 0, int),
            ),
            "game": GameConfig(
                max_plies=self._resolve_reco(reco, "max_plies", 240, int),
                selfplay_fraction=float(reco.get("selfplay_fraction", 0.0)),
                sf_policy_temp=self._resolve_reco(reco, "sf_policy_temp", 0.25),
                sf_policy_label_smooth=self._resolve_reco(reco, "sf_policy_label_smooth", 0.05),
                timeout_adjudication_threshold=float(reco.get("timeout_adjudication_threshold", 0.90)),
                syzygy_path=syzygy_path,
                syzygy_rescore_policy=bool(reco.get("syzygy_rescore_policy", False)),
                syzygy_adjudicate=bool(reco.get("syzygy_adjudicate", False)),
                syzygy_adjudicate_fraction=float(reco.get("syzygy_adjudicate_fraction", 1.0)),
                syzygy_in_search=bool(reco.get("syzygy_in_search", False)),
            ),
        }
        sf_args = (
            self._resolve_reco(reco, "sf_nodes", 2000, int),
            self._resolve_reco(reco, "sf_multipv", 5, int),
            syzygy_path,
        )
        return cfgs, sf_args

    def _reset_inference_client(self, reason: str, exc: BaseException) -> None:
        """Close + null the broker client and back off; caller `return`s after."""
        self.log.warning("%s; resetting client: %s", reason, exc)
        try:
            assert self.inference_client is not None  # caller verified
            self.inference_client.close()
        except Exception:
            pass
        self.inference_client = None
        time.sleep(float(self.args.poll_seconds))

    @staticmethod
    def _aggregate_thread_stats(all_stats: list[BatchStats]) -> BatchStats:
        """Per-field reduce: sum ints, average floats, take thread-0 for the rest."""
        agg: dict = {}
        for fld in dataclasses.fields(all_stats[0]):
            vals = [getattr(st, fld.name) for st in all_stats]
            if all(isinstance(v, int) for v in vals):
                agg[fld.name] = sum(vals)
            elif all(isinstance(v, float) for v in vals):
                agg[fld.name] = sum(vals) / len(vals)
            else:
                agg[fld.name] = vals[0]
        return BatchStats(**agg)

    def _run_selfplay_threaded(self, *, games_per_batch: int, sf, eval_, cfgs: dict) -> BatchStats:
        """Multi-threaded selfplay: N threads share one GPU evaluator."""
        n_threads = min(int(self.args.selfplay_threads), games_per_batch)
        base_games, remainder = divmod(games_per_batch, n_threads)
        thread_games = [base_games + (1 if i < remainder else 0) for i in range(n_threads)]
        lock = threading.Lock()
        self._upload_buf_lock = lock  # shared with _check_model_update

        def _on_game_thread_safe(game_batch):
            with lock:
                self._on_completed_game(game_batch)

        seeds = [int(self.rng.integers(2**63)) for _ in range(n_threads)]

        def _run_one_thread(tid):
            return play_batch(
                None, device=str(self.device), rng=np.random.default_rng(seeds[tid]),
                stockfish=sf, evaluator=eval_,
                games=thread_games[tid],
                on_game_complete=_on_game_thread_safe,
                on_step=self._check_model_update if tid == 0 else None,
                stop_fn=self._stop_fn,
                **cfgs,
            )

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(_run_one_thread, i) for i in range(n_threads)]
            all_stats = [f.result()[1] for f in futures]
        return self._aggregate_thread_stats(all_stats)

    def _dispatch_selfplay_one_shard(
        self, *, games_per_batch: int, cfgs: dict, need_local_model: bool,
    ) -> "BatchStats | None":
        """Run one continuous-selfplay session. Returns None on broker errors that
        we recover from via reset (caller exits the iteration); otherwise BatchStats."""
        assert self.sf is not None  # caller (_run_selfplay) calls _sync_stockfish first
        _eval = self.inference_client or self._direct_evaluator
        try:
            if self.args.threaded_selfplay:
                return self._run_selfplay_threaded(
                    games_per_batch=int(games_per_batch),
                    sf=self.sf, eval_=_eval, cfgs=cfgs,
                )
  # Continuous selfplay: 256 slots always full, games recycled on completion.
  # Runs until _stop_selfplay is set (task change, pause, or shutdown).
  # Samples flow via _on_completed_game.
            _samples, stats = play_batch(
                self.model if (need_local_model and _eval is None) else None,
                device=str(self.device), rng=self.rng,
                stockfish=self.sf, evaluator=_eval,
                games=int(games_per_batch),
                on_game_complete=self._on_completed_game,
                on_step=self._check_model_update,
                stop_fn=self._stop_fn,
                **cfgs,
            )
            return stats
        except TimeoutError as exc:
            if self.inference_client is None:
                raise
            self._reset_inference_client("inference broker timed out", exc)
            return None
        except RuntimeError as exc:
            err = str(exc).lower()
            if self.inference_client is None or not any(tok in err for tok in ("inference", "broker", "slot")):
                raise
            self._reset_inference_client("inference broker error", exc)
            return None

    def _flush_and_upload_after_shard(self, manifest: dict, model_sha: str) -> None:
        """Flush any buffered samples + upload pending shards + prune unused cached models."""
        if self.upload_buf.positions > 0:
            _flush_upload_buffer_to_pending(
                pending_dir=self.pending_dir, username=str(self.args.username),
                buf=self.upload_buf, now_s=time.time(),
            )
            self._upload_pending_shards(default_elapsed_s=0.0)

        best_sha = str((manifest.get("best_model") or {}).get("sha256") or "")
        if not self.shared_cache_enabled:
            _prune_cached_models(cache_dir=self.cache_dir, keep_shas={model_sha, best_sha})
        self._upload_pending_shards(default_elapsed_s=0.0)

    def _run_selfplay(self, manifest: dict) -> None:
        """Continuous selfplay — runs until stop signal (task change/pause/shutdown)."""
        self._stop_selfplay = False
        self._last_manifest_poll_s = time.time()
        reco = manifest.get("recommended_worker") or {}
        self._active_reco = {k: reco.get(k) for k in self._RECO_RESTART_KEYS}
        model_sha = self.model_sha

        need_local_model = self.inference_client is None

        if not model_sha or (need_local_model and self.model is None):
            time.sleep(float(self.args.poll_seconds))
            return

        if need_local_model:
            assert self.model is not None
            if self._direct_evaluator is None:
                self._direct_evaluator = self._build_evaluator(self.model)
            if self._evaluator_model_id != id(self.model):
                _sync_evaluator_to_model(self._direct_evaluator, self.model)
                self._evaluator_model_id = id(self.model)

        games_per_batch = (
            int(self.games_per_batch_local)
            if self.games_per_batch_local is not None
            else int(reco.get("games_per_batch", 8))
        )

        cfgs, (sf_nodes, sf_multipv, syzygy_path) = self._build_selfplay_configs(reco)
        self._sync_stockfish(manifest, sf_nodes, sf_multipv, syzygy_path=syzygy_path)
        assert self.sf is not None  # _sync_stockfish always assigns

        t0 = time.time()
        self._saw_completed_game = False
        stats = self._dispatch_selfplay_one_shard(
            games_per_batch=games_per_batch, cfgs=cfgs, need_local_model=need_local_model,
        )
        if stats is None:
            return
        t1 = time.time()

        self.log.info(
            "selfplay stopped: games=%d positions=%d W/D/L=%d/%d/%d elapsed_s=%.1f",
            int(stats.games), int(stats.positions),
            int(stats.w), int(stats.d), int(stats.l),
            float(t1 - t0),
        )
        if isinstance(self._direct_evaluator, ThreadedDispatcher):
            ds = self._direct_evaluator.stats
            self.log.info(
                "dispatcher stats: batches=%d avg_batch=%.1f avg_forward_ms=%.2f full_drains=%d",
                ds["lifetime_batches"], ds["avg_batch_size"],
                ds["avg_forward_ms"], ds["lifetime_full_drains"],
            )

        self._flush_and_upload_after_shard(manifest, model_sha)

    def _cleanup(self) -> None:
        if self.inference_client is not None and hasattr(self.inference_client, "close"):
            try:
                self.inference_client.close()
            except Exception:
                pass
        if self.sf is not None:
            try:
                self.sf.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
