from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import getpass
import hashlib
import ipaddress
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, overload
from collections.abc import Callable

import chess
import numpy as np
import torch

if TYPE_CHECKING:
    import requests

from chess_anti_engine.heap_census import maybe_start_heap_census
from chess_anti_engine.broker_hang import (
    WORKER_HANG_ABORT_EXIT_CODE,
    BrokerHangWatchdog,
    pin_nvml_cuda_check,
    resolve_worker_boot_hang_abort_seconds,
)
from chess_anti_engine.mcts.gumbel import (
    DEFAULT_VOLATILITY_ANCHOR,
    SELFPLAY_GUMBEL_C_SCALE,
    policy_temp_active,
)
from chess_anti_engine.encoding import encode_positions_batch, input_plane_count
from chess_anti_engine.moves.encode import legal_move_indices
from chess_anti_engine.inference import (
    AOTEvaluator,
    DirectGPUEvaluator,
    MultiSlotInferenceClient,
    SlotInferenceClient,
    model_constant_source,
    require_model_planes,
)
from chess_anti_engine.inference_threaded import ThreadedDispatcher
from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    POLICY_ENCODING_LC0_1858,
    POLICY_SIZE,
    normalize_policy_encoding,
)
from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
    model_config_from_manifest_dict,
)
from chess_anti_engine.utils.config_yaml import strict_config_bool
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    delete_shard_path,
    load_shard_arrays,
    pack_shard_for_upload,
)
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import DiffFocusConfig
from chess_anti_engine.selfplay.config import (
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.diff_focus_norm import DiffFocusNormalizer
from chess_anti_engine.selfplay.manager import BatchStats
from chess_anti_engine.selfplay.state import build_diff_focus_normalizer
from chess_anti_engine.train.target_builder import SfTargetParams
from chess_anti_engine.selfplay.match import play_match_batch
from chess_anti_engine.selfplay.resume import (
    resume_inflight_games,
    suspend_inflight_games,
    sweep_orphan_state_files,
)
from chess_anti_engine.selfplay.opening import (
    OpeningConfig,
    _load_fen_list,
    cap_dole_batch,
    expand_dole_seeds,
    sample_sf_refute_batch,
    warm_opening_book_cache,
)
from chess_anti_engine.stockfish import StockfishPool, StockfishUCI
from chess_anti_engine.stockfish.uci import (
    StockfishDesyncError,
    StockfishTimeoutError,
)
from chess_anti_engine.utils import sha256_file as _sha256_file
from chess_anti_engine.utils.gil_probe import GilContentionProbe
from chess_anti_engine.utils.versioning import version_lt
from chess_anti_engine.version import (
    PACKAGE_NAME,
    PACKAGE_VERSION,
    PROTOCOL_VERSION,
    UPLOAD_CONTENT_SHA256_HEADER,
)
from chess_anti_engine.worker_assets import (
    _cached_sha_asset_needs_refresh,
    _download_and_verify_shared,
    _download_opening_book,
    _ensure_executable,
    _prune_cached_models,
    _safe_manifest_filename,
)
from chess_anti_engine.worker_buffer import (
    _buffer_add_completed_game,
    _buffer_should_flush,
    _BufferedUpload,
    _flush_upload_buffer_to_pending,
    _pending_elapsed_path,
)
from chess_anti_engine.worker_config import load_worker_config, save_worker_config
from chess_anti_engine.worker_inference import (
    maybe_compile_inference_model as _maybe_compile_inference_model,
    sync_evaluator_to_model as _sync_evaluator_to_model,
)

_ResolveT = TypeVar("_ResolveT")

# Single home of the SF target-construction defaults (see
# trainer.resolve_sf_target_params, which derives from the same dataclass).
_SF_TARGET_DEFAULTS = SfTargetParams()


class _MissingRequiredReco(RuntimeError):
    """A safety-critical worker config (e.g. sf_nodes) was set neither on the
    CLI nor in the server ``recommended_worker`` manifest.

    Raised instead of falling back to a guessed default that would run
    Stockfish at the wrong strength and silently poison training data (this is
    what produced the ~100% curriculum-winrate batches). The main loop catches
    it, skips the session, and re-polls for a complete manifest.
    """

  # Wire-protocol values sent in X-CAE-Worker-State; tests assert on them
  # (tests/test_distributed_selfplay_backpressure.py).
_MANIFEST_STATE_ACTIVE = "active"
_MANIFEST_STATE_PAUSED = "paused_selfplay"

# How long our shard tag may trail the published model before we alarm. Must
# clear a normal swap (poll + download + compile of a ~250MB checkpoint) so a
# healthy publish never trips it, while staying far under one training
# iteration — the window in which stale uploads still cost a whole iteration.
_MODEL_STALE_ALARM_S = 180.0


def _require_compact_policy_encoding(raw: object) -> str:
    """Reject az_4672 as a recording encoding (stale/legacy server manifest).

    Models only emit compact 1858 and finalize writes compact train targets
    unconditionally, but GameConfig-driven widths (e.g. the TB one-hot
    override in selfplay/finalize.py) still honor the declared encoding — an
    accepted az_4672 would silently write compact indices into 4672-wide
    vectors instead of failing loudly here.
    """
    enc = normalize_policy_encoding(str(raw) if raw is not None else None)
    if enc != POLICY_ENCODING_LC0_1858:
        raise SystemExit(
            f"policy_encoding {raw!r} unsupported: recording is compact "
            f"{POLICY_ENCODING_LC0_1858} only (stale server manifest?)",
        )
    return enc


def _bounded_poll_detail(response: Any) -> str:
    """A short, safe rendering of a failed poll's body for one log line."""
    try:
        body = response.text
    except Exception:
        return "<unreadable body>"
    text = " ".join(str(body).split())
    return text[:200] if text else "<empty body>"


def _upload_response_allows_pending_delete(response: Any) -> bool:
    """True when a worker may delete its local pending shard after upload.

    The server returns HTTP 200 for both accepted uploads and explicit protocol
    rejections. Only accepted/deduped uploads are safe to drop locally.
    """
    if int(getattr(response, "status_code", 0)) != 200:
        return False
    try:
        body = response.json()
    except Exception:
        return False
    if not isinstance(body, dict):
        return False
    if bool(body.get("rejected", False)):
        return False
    if body.get("stored") is True:
        return True
    # Duplicate uploads return stored=false after the server has already
    # accumulated or recovered the shard; with no rejected flag, local retry
    # would only resend data the server intentionally deduped.
    return body.get("stored") is False


def _upload_response_rejection_reason(response: Any) -> str | None:
    """Return the terminal server rejection reason for a shard upload, if any.

    ⚑ CARRIES `reason_code` THROUGH, which is what gives that field a consumer.
    The server added it so a monitor could switch on a stable token while the
    human prose stays free to change, and at the time nothing in the repo read
    it -- one refactor from being a field nobody reads, and the way to notice
    its absence is to put it somewhere a person looks. The string returned here
    is what lands in the `.reason.txt` beside the quarantined shard and in the
    worker's warning line, so the code now reaches both.

    Appended rather than substituted: the prose is the part an operator reads
    first, and replacing it with `6` would be a worse artifact than no code.
    """
    if int(getattr(response, "status_code", 0)) != 200:
        return None
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict) or not bool(body.get("rejected", False)):
        return None
    reason = body.get("reason")
    text = str(reason) if reason is not None else "server rejected shard"
    code = body.get("reason_code")
    return f"{text} (reason_code={code})" if code is not None else text


def _quarantine_rejected_pending_shard(shard_path: Path, reason: str) -> Path:
    """Move a terminally rejected local shard out of the retry queue."""
    corrupt_dir = shard_path.parent.parent / "corrupt"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    dest = corrupt_dir / shard_path.name
    if dest.exists():
        dest = corrupt_dir / f"{shard_path.name}.rejected.{int(time.time())}"
    shard_path.replace(dest)
    (dest.with_suffix(dest.suffix + ".reason.txt")).write_text(reason, encoding="utf-8")

    elapsed_path = _pending_elapsed_path(shard_path)
    if elapsed_path.exists():
        elapsed_dest = dest.with_suffix(dest.suffix + ".elapsed_s")
        if elapsed_dest.exists():
            elapsed_dest = dest.with_suffix(dest.suffix + f".elapsed_s.{int(time.time())}")
        elapsed_path.replace(elapsed_dest)
    return dest


def _bad_shard_report_payload(*, shard_path: Path, quarantine_path: Path, reason: str, kind: str) -> dict[str, object]:
    files = 0
    bytes_total = 0
    nonzero_files = 0
    if quarantine_path.is_dir():
        try:
            for fp in quarantine_path.rglob("*"):
                if not fp.is_file():
                    continue
                files += 1
                size = int(fp.stat().st_size)
                bytes_total += size
                if size > 0:
                    nonzero_files += 1
        except OSError:
            pass
    return {
        "kind": str(kind),
        "shard_name": shard_path.name,
        "reason": str(reason),
        "quarantine_name": quarantine_path.name,
        "files": int(files),
        "bytes": int(bytes_total),
        "nonzero_files": int(nonzero_files),
        "created_mtime": float(quarantine_path.stat().st_mtime) if quarantine_path.exists() else 0.0,
    }


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

    with suppress(Exception):
        out["cpu_count"] = int(os.cpu_count() or 1)

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
    os.execv(sys.executable, [sys.executable, *sys.argv])


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
    - PYTORCH_NVML_BASED_CUDA_CHECK: keep `torch.cuda.is_available()` off the
      driver-init path (audit R3).
    """
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_COMPILE_THREADS", "1")
    # Route the CUDA availability probe around `cuInit`, which never returns on
    # a wedged dxg bridge. Load-bearing and previously inherited from the
    # launching shell only -- see `pin_nvml_cuda_check` for the full why.
    pin_nvml_cuda_check()
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


_LOOPBACK_HOST_NAMES = frozenset({"localhost", "0.0.0.0"})


def _is_loopback_host(host: str) -> bool:
    """True when traffic to `host` provably never leaves this machine.

    ⚑ Not a string match against `127.0.0.1`. The whole of `127.0.0.0/8` is
    loopback, and running local services on distinct addresses inside it
    (`127.0.0.2`, ...) is a normal thing to do -- refusing those would push a
    user to set the cleartext override on a host where nothing is at risk,
    which trains them to set it where something is. `ipaddress` also gets
    `::1` and IPv4-mapped forms right, which a literal set does not.

    `0.0.0.0` is not loopback but is the address a locally-run server is
    typically BOUND to and then dialled by, so it stays exempt by name.
    """
    h = host.strip().lower()
    if not h:
        return False
    if h in _LOOPBACK_HOST_NAMES:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
    # Not a literal IP -- a DNS name, which may resolve anywhere. Resolving it
    # here would make the refusal depend on whatever the resolver says at
    # startup, so treat it as remote.
        return False


def cleartext_transport_refusal(server_url: str, *, allow: bool) -> str | None:
    """Return a refusal message for an unencrypted non-loopback server URL.

    The worker sends HTTP Basic credentials -- a reusable username/password,
    not a scoped token -- on EVERY authenticated request, and the same cleartext
    channel carries the manifest whose sha256 values are the only integrity
    check on the model checkpoint and the optional self-update wheel. So an
    on-path observer gets a credential that works on every route, and an on-path
    *modifier* can rewrite an artifact and its expected digest together.

    Loopback is exempt because there is no path to observe. `0.0.0.0` is in the
    exempt set as the address a locally-run server is typically BOUND to and
    then dialled by; it is not a routable destination.

    Pure and module level so the rule is testable without constructing a worker.
    """
    if allow:
        return None
    url = str(server_url or "").strip()
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme != "http":
        return None
    host = (parsed.hostname or "").strip().lower()
    if _is_loopback_host(host):
        return None
    return (
        f"refusing to send credentials in cleartext to {host or url!r} over http://. "
        "HTTP Basic auth and the model manifest both travel this connection, so an "
        "on-path attacker can steal a reusable credential and swap the checkpoint "
        "it authenticates. Use https:// (terminate TLS at a reverse proxy if the "
        "server itself does not), or pass --allow-cleartext-http to override on a "
        "network you control."
    )


def _merge_cli_with_yaml_defaults(args, cfg: dict) -> None:
    """Fill missing CLI fields from worker.yaml. CLI always wins."""
    args.server_url = (
        args.server_url or cfg.get("server_url") or "http://127.0.0.1:45453"
    )
  # Checked HERE rather than at first request: this runs before any credential
  # is read or sent, so a refusal costs nothing and cannot half-leak. Raising
  # `SystemExit` rather than warning, because a warning on a volunteer's
  # terminal is a warning nobody reads while the password goes out anyway.
  # ⚑ The yaml key is read HERE, not in the flag loop further down: the refusal
  # must see it, and the refusal has to happen before any credential is read.
  # A `worker.yaml` override that the guard could not see would be a documented
  # escape hatch that does nothing -- an accepted-and-ignored value.
  # ⚑ strict_config_bool, not truthiness: `allow_cleartext_http: "false"` is a
  # non-empty STRING and would ENABLE the escape hatch, sending reusable Basic
  # credentials over the wire -- the failure this guard exists to prevent,
  # reached by typing the word that means "don't".
    if (
        not getattr(args, "allow_cleartext_http", False)
        and "allow_cleartext_http" in cfg
        and strict_config_bool(
            cfg.get("allow_cleartext_http"),
            key="allow_cleartext_http", source="worker.yaml",
        )
    ):
        args.allow_cleartext_http = True
    refusal = cleartext_transport_refusal(
        args.server_url, allow=bool(getattr(args, "allow_cleartext_http", False)),
    )
    if refusal is not None:
        raise SystemExit(refusal)
    for field_name in _OR_FALLBACK_FIELDS:
        setattr(args, field_name, getattr(args, field_name) or cfg.get(field_name))

    if args.password_file is None and cfg.get("password_file"):
        args.password_file = str(cfg.get("password_file"))

    for flag in ("self_update", "stockfish_from_server"):
        if not bool(getattr(args, flag)) and bool(cfg.get(flag, False)):
            setattr(args, flag, True)

    if args.sf_workers is None:
        args.sf_workers = int(cfg.get("sf_workers", 1))
    if args.sf_nice is None:
        args.sf_nice = int(cfg.get("sf_nice", 0))
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
    ap.add_argument(
        "--allow-cleartext-http",
        action="store_true",
        help=(
            "Permit http:// to a non-loopback server. Sends your password in "
            "cleartext on every request; only for a network you control."
        ),
    )
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
    ap.add_argument("--inference-slot-input-planes", type=int, default=input_plane_count(),
                    help="Input channel count for inference slot (must match broker; "
                    "146 for v1, 175 for v2_threats).")
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
    # Compatibility for workers spawned by an already-running pre-upgrade Tune
    # controller. The fast dispatcher is unconditional; this flag is a no-op.
    ap.add_argument("--threaded-dispatcher", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--dispatcher-batch-wait-ms",
        type=float,
        default=0.0,
        help="ThreadedDispatcher batching window in ms (default: 0; no fixed wait).",
    )
    ap.add_argument(
        "--dispatcher-max-batch",
        type=int,
        default=4096,
        help="Maximum rows per ThreadedDispatcher forward.",
    )
    ap.add_argument(
        "--dispatcher-target-batch",
        type=int,
        default=0,
        help="Preferred ThreadedDispatcher drain size; 0 uses --dispatcher-max-batch.",
    )

    ap.add_argument("--stockfish-path", type=str, default=None)

  # Server-managed strength knobs (defaults come from manifest recommended_worker).
    ap.add_argument("--sf-nodes", type=int, default=None)
    ap.add_argument("--sf-multipv", type=int, default=None)
    ap.add_argument("--sf-policy-temp", type=float, default=None)
    ap.add_argument("--sf-policy-label-smooth", type=float, default=None)

  # Local performance knob
    ap.add_argument("--sf-workers", type=int, default=None)
    ap.add_argument(
        "--sf-nice",
        type=int,
        default=None,
        help="Absolute POSIX nice target for Stockfish subprocesses, clamped to 0..19; never raises priority above the parent.",
    )

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
    ap.add_argument("--selfplay-temperature", type=float, default=None)
    ap.add_argument("--selfplay-temperature-decay-start-move", type=int, default=None)
    ap.add_argument("--selfplay-temperature-decay-moves", type=int, default=None)
    ap.add_argument("--selfplay-temperature-endgame", type=float, default=None)
    ap.add_argument("--gumbel-scale", type=float, default=None)

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

  # In-flight selfplay resume, off unless a manifest turns it on. Class-level
  # so the OFF path is the default for any session — including the ones tests
  # build with object.__new__ — rather than an AttributeError at the hook.
  # _begin_resume_session sets all three per session; session-fixed by design.
    _resume_inflight_enabled: bool = False
    _resume_compat_fingerprint_active: str = ""
  # Trial this session's games belong to, SNAPSHOT at session start. Not read
  # live: _negotiate_lease assigns self.leased_trial_id BEFORE the caller
  # notices the change and sets _stop_selfplay, so by the time the suspend hook
  # runs the live value is already the trial we were reassigned TO. Stamping
  # that on games played entirely under the old trial would let them sail
  # through the trial guard and be uploaded as the NEW trial's data — which is
  # the one thing the guard exists to stop, and config_mismatch cannot catch it
  # (sibling GPBT trials differ on LR and loss weights, not on the 11
  # record-shaping keys).
    _resume_trial_id_active: str = ""

  # Shutdown latch. NOT the same reasoning as the three above: those are
  # class-only (nothing in __init__ touches them; _begin_resume_session assigns
  # all three, :3216), whereas __init__ ALSO sets this pair (:922-923) — the one
  # place in this class with two sources of truth for the same initial value.
  # The class default therefore only ever serves a session that never ran
  # __init__, i.e. the ones tests build with object.__new__. Such a session
  # reaches _run_selfplay's entry guard (`if self._shutdown_requested: return`,
  # :3778) BEFORE any per-ply hook, and without a default it raises
  # AttributeError there rather than taking the not-shutting-down path — that
  # guard, not run()'s loop head, is the line the three
  # tests/test_selfplay_resume.py cases hit. (run()'s loop head, :1093, and
  # _stop_fn, :1036, read the flag too; neither is reached first, and neither is
  # the on_suspend hook — that is _suspend_inflight_games, :3259, which does not
  # read this flag at all.)
  # False is fail-OPEN with respect to shutting down: it means "no shutdown
  # requested, keep working". That is right because it is the state a session
  # that never ran is in — not because refusing to stop is a safe default.
  # ⚠ Applied inconsistently across this class: _stop_selfplay (:919) and
  # _hold_selfplay (:928), the other two flags the per-ply hooks read, have NO
  # class defaults, so the object.__new__ invariant holds for some of these
  # flags and not others. Left alone deliberately — no test needs them today,
  # and adding defaults would mint two more two-source-of-truth pairs.
    _shutdown_requested: bool = False
    _shutdown_signal: int = 0

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

  # In-flight selfplay state (selfplay/resume.py). Deliberately NOT the Ray
  # session/trial dir: state left there is already known to be lost at a
  # restart, which is the exact event this survives. work_dir is the worker's
  # own durable root — the same place pending shards wait out a restart — so a
  # suspended game outlives the process that wrote it.
        self.resume_dir = work_dir / "selfplay_resume"

        arena_dir = work_dir / "arena"
        self.arena_pending_dir = arena_dir / "pending"
        self.arena_uploaded_dir = arena_dir / "uploaded"
  # ⚑ Separate from `uploaded`, which means "this result reached the server".
  # Unparseable pending files used to be moved into `uploaded` to stop a retry
  # storm -- correct motivation, wrong destination: a file that reached NOBODY
  # then sat there indistinguishable from a genuine upload, so any tooling
  # that counts uploads by listing the directory over-reported. Arena results
  # feed Elo readings and promotion decisions, and this project has been
  # burned before by rulers that silently lost rows.
        self.arena_rejected_dir = arena_dir / "rejected"

        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.uploaded_dir.mkdir(parents=True, exist_ok=True)
        self.resume_dir.mkdir(parents=True, exist_ok=True)
  # Leak-diagnosis census (2026-08-07 worker OOM): default off. Armed per
  # worker by a flag file in this durable work dir, so one revived worker can
  # be instrumented without touching the rest of the fleet or the driver.
        maybe_start_heap_census(work_dir)
        self.arena_pending_dir.mkdir(parents=True, exist_ok=True)
        self.arena_uploaded_dir.mkdir(parents=True, exist_ok=True)
        self.arena_rejected_dir.mkdir(parents=True, exist_ok=True)

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
                "fast_simulations", "gumbel_topk", "gumbel_target_max_visit_cap",
                "gumbel_target_untempered_prior",
                "gumbel_c_scale", "gumbel_scale", "gumbel_scale_after",
                "gumbel_scale_decay_start_move", "gumbel_scale_decay_moves",
                "curriculum_gumbel_scale", "curriculum_gumbel_scale_after",
                "curriculum_gumbel_scale_decay_start_move", "curriculum_gumbel_scale_decay_moves",
                "opening_book_prob", "opening_book_max_plies",
                "opening_book_max_games", "random_start_plies", "sf_nodes",
                "sf_multipv", "sf_policy_temp", "sf_policy_label_smooth",
                "sf_policy_score_mode", "sf_policy_cp_temp",
                "timeout_adjudication_threshold", "temperature",
                "temperature_decay_start_move", "temperature_decay_moves",
                "temperature_endgame",
                "selfplay_temperature", "selfplay_temperature_decay_start_move",
                "selfplay_temperature_decay_moves", "selfplay_temperature_endgame",
                "syzygy_path", "stockfish_syzygy_path", "syzygy_rescore_policy",
                "syzygy_adjudicate", "syzygy_adjudicate_fraction", "syzygy_in_search",
            ]
            overridden = [k for k in _server_managed_keys if getattr(args, k, None) is not None]
            if overridden:
                raise SystemExit(
                    "This worker is configured for server-managed strength knobs. "
                    f"Remove selfplay/strength flags ({', '.join(overridden)}), or pass --allow-overrides for debugging."
                )

  # Engine: initialize with placeholder settings; we will align nodes from manifest each loop.
  # MultiPV, hash size, and SyzygyPath are process-level; rebuild when they change.
  # ⚑ AND THIS LINE MUST STAY ABOVE THE BOOT WATCHDOG BLOCK BELOW. That block
  # installs `_boot_hang_exit` as the watchdog's `exit_fn`, and `_boot_hang_exit`
  # reads `self.sf` to kill the Stockfish children before `os._exit`. The
  # watchdog thread is live from `start()`, so on a first-session wedge the
  # handler can run before anything else assigns `self.sf` -- it is this `None`
  # that makes that safe, not the handler's `suppress(Exception)`, which would
  # silently swallow the AttributeError and leak the engines it exists to kill.
        self.sf: StockfishPool | StockfishUCI | None = None
        self.sf_multipv_active: int | None = None
        self.sf_hash_mb_active: int | None = None
        self.sf_syzygy_path_active: str | None = None
  # Resolved binary path the live engine was built with; re-init on change so a
  # hot-swapped Stockfish (new SHA -> new cached path) is actually loaded.
        self.sf_path_active: str | None = None

        self.last_model_sha = None
        self.last_ob_sha: str | None = None
        self.last_ob2_sha: str | None = None
        self.last_fenlist_sha: str | None = None
        self.last_sf_sha: str | None = None
        self.model_cfg_active: ModelConfig | None = None
        self.model = None
        self._direct_evaluator: (
            DirectGPUEvaluator | ThreadedDispatcher | AOTEvaluator | None
        ) = None
        self.inference_client = self._make_inference_client()
        # One-shot warn latch for a manifest that names no encoding, so the
        # slot-width check below cannot skip itself silently. See
        # _require_slot_planes_match_manifest.
        self._slot_planes_unknown_warned = False
        self.pause_selfplay_active = False
        self.manifest_state = _MANIFEST_STATE_ACTIVE
        self._manifest_poll_failures = 0
        self.manifest_state_elapsed_s: float | None = None
        self.upload_buf = _BufferedUpload()
        self.last_successful_send_s = time.time()
        self._last_upload_flush_s = self.last_successful_send_s

        self.last_best_sha = None
        self.best_model = None

  # Opening book paths (set by _sync_opening_books each iteration).
        self.opening_book_path: str | None = None
        self.opening_book_path_2: str | None = None
        self.opening_fen_list_path: str | None = None
  # Blind-spot FEN seeds doled by the server (the server hands the whole list to
  # exactly one worker-poll per iteration). Two-stage delivery:
  #  * _pending_fen_dole — stash for the NEXT session start (set when no session
  #    is live; survives model-not-ready / dispatch-failure retries).
  #  * _live_dole_queue — the shared FIFO the CURRENTLY-running session drains
  #    (the same object handed to every play_batch, incl. all threads of the
  #    threaded path). None between sessions. A mid-session dole refills it in
  #    place so the running selfplay picks up the batch without a restart.
  # Parallel SF-refute channel (_pending/_live_sf_refute_queue) holds a
  # round-robin slice for short SF-as-opponent openings (then handoff to selfplay).
  # _dole_lock guards refill vs the session-boundary hand-off (drain itself is
  # lock-free — resolve_slot_opening pops exception-safely, atomic under the GIL).
        self._pending_fen_dole: list[str] = []
        self._live_dole_queue: list[str] | None = None
        self._pending_sf_refute: list[str] = []
        self._live_sf_refute_queue: list[str] | None = None
        self._dole_lock = threading.Lock()

  # Per-iteration state (set during each loop iteration).
        self.model_sha = ""
        self.model_step = 0
        self._saw_completed_game = False
        self._stop_selfplay = False
  # Set from the SIGTERM/SIGINT handler; read by run()'s loop head so the worker
  # exits only AFTER the normal suspend path has banked in-flight games.
        self._shutdown_requested = False
        self._shutdown_signal = 0
  # Train-phase pause: HOLD in-flight games (play_batch pause_fn) instead of
  # tearing the session down — teardown discards every partial game (~all
  # slots) each iteration and censors games longer than one selfplay window
  # out of replay. Env CAE_WORKER_STOP_ON_PAUSE=1 restores the old teardown.
        self._hold_selfplay = False
        self._hold_on_pause = os.environ.get("CAE_WORKER_STOP_ON_PAUSE", "") != "1"
        self._upload_buf_lock: threading.Lock | None = None  # set when threaded
        self._pending_upload_lock = threading.Lock()
        self._pending_buffer_flushes: deque[
            tuple[_BufferedUpload, float, str | None, int]
        ] = deque()
        self._pending_buffer_positions = 0
        self._last_manifest_poll_s: float = 0.0
        self._last_dispatcher_stats_log_s: float = time.time()
        self._last_dispatcher_stats_snapshot: tuple[int, int, int, float, float, float, float, float] = (
            0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        self._last_broker_client_stats_log_s: float = time.time()
        self._last_broker_client_stats_snapshot: dict[str, Any] = {}
  # In-flight resume counters (the flag + fingerprint are class-level defaults;
  # see _resume_inflight_enabled).
        self._resume_counts_lock = threading.Lock()
        self._resume_counts: dict[str, int] = {
            "suspended": 0, "suspend_skipped": 0, "resumed": 0, "discarded": 0,
        }
        self._resume_skip_reasons: dict[str, int] = {}
        self._completion_telemetry_lock = threading.Lock()
        self._completion_games = 0
        self._completion_positions = 0
        self._completion_callback_s = 0.0
        self._completion_upload_s = 0.0
        self._completion_by_thread: dict[int, int] = {}
        self._last_completion_stats_snapshot: tuple[int, int, float, float, dict[int, int]] = (
            0, 0, 0.0, 0.0, {},
        )
        self._phase_timing_lock = threading.Lock()
        self._phase_timing_s: dict[str, float] = {}
        self._last_phase_timing_snapshot: dict[str, float] = {}
        # Selfplay session liveness: bumped on phase stats / completed games.
        # Watchdog exits the process if we stall after session start so the
        # launcher restarts a hung worker (e.g. slot-acquire deadlock) instead
        # of waiting forever. Override with CAE_WORKER_STALL_TIMEOUT_S (0=off).
        self._last_selfplay_progress_s: float = time.time()
        self._selfplay_session_active = False
        self._stall_watchdog_started = False
        self._gil_probe = GilContentionProbe(interval_s=0.010)
        # ⚑ AUDIT R1: no watchdog covered the CUDA-init calls the worker makes
        # while holding a server lease. `_start_selfplay_stall_watchdog` arms
        # only once `_selfplay_session_active` is True, which happens AFTER
        # `model.to(self.device)` and `_build_evaluator` -- the two calls that
        # block forever on a wedged dxg bridge. A worker that wedged there sat
        # silently holding a 1h server lease, its shm slots and its Stockfish
        # children, with no log line and no exit.
        #
        # ⚑ WHAT THIS BOUNDS, EXACTLY. The detector fires only while a `stage()`
        # is open, so its coverage is the three stages and nothing else:
        # `model_to_device`, `compile_inference_model`, `build_evaluator`.
        # It is NOT a bound on the lease-held span. These run inside the lease
        # and outside any stage, and are deliberately left uncovered:
        # `_upload_pending_shards`, `_upload_pending_arena_results`,
        # `_sync_assets` (model download + torch.load), `_begin_resume_session`,
        # `_build_selfplay_configs`, `warm_opening_book_cache`,
        # `_sync_stockfish`. None of them is a CUDA driver call, which is the
        # failure R1 is about; they are network and disk work whose legitimate
        # duration varies with link speed, so a 1800s bound on them would turn a
        # slow model download into a crash loop. If one of THOSE hangs the
        # symptom is a stalled fleet, not a wedged device, and it belongs to a
        # different instrument.
        #
        # Non-obvious consequence worth stating: THE ARENA PATH IS COVERED FOR
        # TWO OF THE THREE STAGES. `model_to_device` and
        # `compile_inference_model` live in `_load_and_compile_model`, which
        # `_run_arena` also calls, so an arena task's model load is watched.
        # Only `build_evaluator` is selfplay-only (`_run_selfplay`).
        #
        # Same detector as the broker (one implementation, two `component`
        # values). Live from construction, so it also covers a wedge that
        # happens before the first poll.
        self._boot_hang_watchdog = BrokerHangWatchdog(
            threshold_s=resolve_worker_boot_hang_abort_seconds(),
            component="worker",
            exit_code=WORKER_HANG_ABORT_EXIT_CODE,
            exit_fn=self._boot_hang_exit,
        )
        self._boot_hang_watchdog.start()
  # Model-freshness watch: a dedicated daemon thread owns manifest polling so
  # model freshness never depends on any selfplay thread staying alive (see
  # _start_model_watch_thread). The lock keeps it from running concurrently
  # with the per-ply on_step call; _model_stale_since_s drives the alarm.
        self._model_watch_started = False
        self._model_watch_lock = threading.Lock()
        self._model_stale_since_s: float | None = None
        self._model_stale_alarmed = False
  # Manifest-watch state (lazy-init in _check_model_update via getattr fallback).
        self._manifest_path: Path | None = None
        self._manifest_mtime: float | None = None
        self._active_reco: dict | None = None
  # Counters for the swallow paths below, so "this fired, N times" is a
  # number somebody can read rather than a guess reconstructed offline.
  # ⚑ Deliberately IN-PROCESS and surfaced through the warning lines that
  # increment them, not pushed to the server. The worker has no heartbeat or
  # stats channel today -- its only upward posts are shard/arena uploads, the
  # lease, and `/report_bad_shard` -- so a server-visible counter would mean a
  # new endpoint plus server-side handling, which is a larger change than the
  # defect warrants and would need its own review. The greppable log line is
  # what converts this class from "invisible" to "one grep", which was the
  # point; wiring telemetry can follow separately if it earns it.
        self._reco_corrupt_count = 0
        self._arena_rejected_count = 0
  # Last corrupt (regret, nodes) pair already warned about. `_active_difficulty`
  # runs once per COMPLETED GAME, so an unconditional warning there would be a
  # per-game flood on a stuck-corrupt manifest; this rate-limits to one line
  # per distinct offending value, which is what makes it greppable.
        self._reco_corrupt_last: tuple[Any, Any] | None = None
  # Session-start asset SHAs (SF binary, opening books) — a mid-run change to
  # these can't be live-applied, so it forces a restart. Set in _run_selfplay.
        self._active_assets: tuple | None = None
  # Live SelfplayState registry for the running continuous session, used to
  # apply live-safe reco changes (selfplay_fraction / regret / SF nodes) in
  # place without bouncing the session. The single path registers its one
  # state; the threaded path registers one per thread (a live apply must reach
  # ALL of them — updating only one would leave the rest on stale config,
  # which is why the threaded path historically fell back to a full restart;
  # post pause-hold/oversubscribe that restart abandons ~2x games_per_batch
  # in-flight games, so live apply now matters). Emptied between sessions.
        self._live_states: list[Any] = []
        self._live_states_lock = threading.Lock()
  # (new_game, new_opp, sf_nodes) from the latest in-session live apply;
  # transplanted onto any state that registers after that apply. None when no
  # apply has run this session.
        self._pending_live_override: tuple[Any, Any, int] | None = None
        self._evaluator_model_id: int | None = None

    def _build_evaluator(
        self, model: torch.nn.Module,
    ) -> DirectGPUEvaluator | ThreadedDispatcher | AOTEvaluator:
        device = str(self.device)
        if self.args.threaded_selfplay:
            if self.args.aot_dir:
                raise ValueError(
                    "--aot-dir is incompatible with --threaded-selfplay: the "
                    "ThreadedDispatcher compiles on its own thread and ignores "
                    "AOT artifacts, so it would silently run eager. Use "
                    "--compile-inference instead (or drop --threaded-selfplay)."
                )
            disp_compile = (
                str(self.args.compile_mode) if self.args.compile_inference else None
            )
            return ThreadedDispatcher(
                model, device=device, max_batch=int(self.args.dispatcher_max_batch),
                target_batch=int(self.args.dispatcher_target_batch),
                batch_wait_ms=float(self.args.dispatcher_batch_wait_ms),
                compile_mode=disp_compile,
            )
        if self.args.aot_dir:
  # AOT .pt2 artifacts are compiled for a fixed input width; size the
  # pinned buffers from the model so a v2_threats net fails loud at
  # weight load instead of corrupting a 146-channel buffer.
            aot = AOTEvaluator(
                self.args.aot_dir, device=device, max_batch=4096,
                input_planes=input_plane_count(
                    getattr(model, "input_extra_features", None),
                ),
            )
            aot.load_weights(model_constant_source(model))
            return aot
        return DirectGPUEvaluator(model, device=device, max_batch=4096, n_slots=2)

    def _stop_fn(self) -> bool:
        """Called every ply by play_batch.  Return True to exit continuous selfplay.

        `_shutdown_requested` is checked SEPARATELY rather than relying on the
        `_stop_selfplay` the handler also sets: `_run_selfplay` clears
        `_stop_selfplay` at session start, so a SIGTERM landing between `run()`'s
        loop-head check and that line was erased, and a full 32-thread session
        started with nothing left to stop it. A shutdown is terminal — once
        requested it can never be un-requested — so it belongs here, not in a
        flag that session setup resets.
        """
        return self._stop_selfplay or self._shutdown_requested

    def _pause_fn(self) -> bool:
        """play_batch hold gate: True while the server's train-phase pause is
        active. A pending stop always wins so teardown is never delayed."""
        return self._hold_selfplay and not self._stop_fn()

    def _install_shutdown_handlers(self) -> None:
        """Turn SIGTERM/SIGINT into the SAME graceful teardown a reco restart-key
        change already performs, so a shutdown suspends in-flight games instead
        of discarding them.

        Without this the worker had no signal handler at all, so
        `_suspend_inflight_games` was reachable ONLY via `_stop_selfplay` — and
        an operator pause therefore killed every in-flight game. Measured
        2026-07-31: `resumed games=0` on all four workers after a pause, versus
        `resumed games=24 records=731` across a graceful session restart. The
        discarded games are what makes the next window length-truncated (only
        short decisive games have finished), which spikes winrate and drives the
        PID to tighten regret ~-0.008 against a phantom.

        The handler only sets flags. The per-ply `_stop_fn` poll does the real
        work, which keeps this async-signal-safe and reuses the suspend path that
        is already exercised in production. Handlers run on the MAIN thread only;
        selfplay threads observe the flag through `_stop_fn`, so threaded
        selfplay is covered.
        """
        def _handler(signum: int, _frame: object) -> None:
  # Signal-handler context: set flags, do not log or take locks.
            self._shutdown_signal = int(signum)
            self._shutdown_requested = True
            self._stop_selfplay = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
  # Not the main thread, or the platform refuses — the worker must still
  # run; it just falls back to the old discard-on-kill behaviour.
                self.log.warning("could not install %s handler", sig)

    def run(self) -> None:
        """Main loop."""
        self._install_shutdown_handlers()
        try:
            while True:
                if self._shutdown_requested:
  # Set by SIGTERM/SIGINT. Any in-flight session has already been torn
  # down through the normal suspend path by the time control returns
  # here, so this is the clean exit point.
                    self.log.info(
                        "shutdown requested (signal %d); exiting after suspend",
                        self._shutdown_signal,
                    )
                    return
                manifest = self._poll_manifest()
                if manifest is None:
                    continue
                self._sync_assets(manifest)
  # Ingest a won dole HERE — after assets sync (FEN path current), before the
  # model-not-ready retry and the arena branch below — so a session-start dole
  # survives those early-continues. Between sessions there is no live state, so
  # it stashes into _pending_fen_dole for the next _run_selfplay; the stash is
  # only cleared once a session actually consumes it. (The server won't claim
  # for a paused/arena/non-selfplay poll, so this is a selfplay dole.)
                self._maybe_ingest_dole_flag(manifest)
                if not self.model_sha:
  # Model download failed (mid-publish race); retry next poll.
                    time.sleep(float(self.args.poll_seconds))
                    continue
                task = manifest.get("task") or {"type": "selfplay"}
                task_type = str(task.get("type", "selfplay")).lower()
                if task_type == "arena":
                    self._run_arena(manifest, task)
                else:
                    try:
                        self._run_selfplay(manifest)
                    except _MissingRequiredReco as exc:
                        # Fail closed: never generate games with a guessed SF
                        # strength. Warn loudly + re-poll for a complete manifest.
                        self.log.warning(
                            "skipping selfplay session: %s; re-polling for a "
                            "complete recommended_worker manifest", exc,
                        )
                        time.sleep(float(self.args.poll_seconds))
                        continue
        finally:
            self._cleanup()

  # -- Helper methods -------------------------------------------------------

    def _server_url_for(self, endpoint: str) -> str:
        if endpoint.startswith(("http://", "https://")):
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
        self.cfg["sf_nice"] = int(self.args.sf_nice)
        if self.games_per_batch_local is not None:
            self.cfg["games_per_batch"] = int(self.games_per_batch_local)
        self.cfg["upload_target_positions"] = int(self.args.upload_target_positions)
        self.cfg["upload_flush_seconds"] = float(self.args.upload_flush_seconds)
        if not bool(self.args.save_password):
            self.cfg.pop("password", None)
        save_worker_config(self.cfg_path, self.cfg)

    def _make_inference_client(self):
        if str(self.args.inference_slot_name or "").strip():
            slot_names = [
                s.strip()
                for s in str(self.args.inference_slot_name).split(",")
                if s.strip()
            ]
            if len(slot_names) == 1:
                return SlotInferenceClient(
                    slot_name=slot_names[0],
                    max_batch=int(self.args.inference_slot_max_batch),
                    input_planes=int(self.args.inference_slot_input_planes),
                )
            return MultiSlotInferenceClient(
                slot_names=slot_names,
                max_batch=int(self.args.inference_slot_max_batch),
                input_planes=int(self.args.inference_slot_input_planes),
            )
        return None

    def _queue_upload_buffer_locked(self, *, now_s: float) -> bool:
        """Detach the active buffer in O(1); caller holds ``_upload_buf_lock``."""
        positions = int(self.upload_buf.positions)
        if positions <= 0 or not self.upload_buf.samples:
            return False
        queued = self.upload_buf
        self.upload_buf = _BufferedUpload()
        trial_id = self.leased_trial_id or self.fixed_trial_id or None
        pending = getattr(self, "_pending_buffer_flushes", None)
        if pending is None:
            pending = deque()
            self._pending_buffer_flushes = pending
        pending.append((queued, float(now_s), trial_id, positions))
        self._pending_buffer_positions = int(
            getattr(self, "_pending_buffer_positions", 0),
        ) + positions
        return True

    def _materialize_queued_buffers_locked(self) -> float | None:
        """Write detached buffers to durable pending shards outside the buffer lock.

        The caller holds ``_pending_upload_lock``, so exactly one thread owns
        materialization.  A failed write leaves the untouched buffer at the
        queue head for retry; successful writes reset and remove it.
        """
        latest_elapsed_s: float | None = None
        while True:
            buf_lock = getattr(self, "_upload_buf_lock", None)
            with buf_lock if buf_lock is not None else nullcontext():
                pending = getattr(self, "_pending_buffer_flushes", None)
                if not pending:
                    break
                queued, queued_at_s, trial_id, positions = pending[0]
            shard_path, elapsed_s = _flush_upload_buffer_to_pending(
                pending_dir=self.pending_dir,
                username=str(self.args.username),
                buf=queued,
                now_s=queued_at_s,
                trial_id=trial_id,
            )
            if shard_path is None:
                raise RuntimeError("queued upload buffer unexpectedly empty")
            with buf_lock if buf_lock is not None else nullcontext():
                head = pending.popleft()
                if head[0] is not queued:
                    raise RuntimeError("queued upload buffer order changed")
                self._pending_buffer_positions = max(
                    0, int(self._pending_buffer_positions) - int(positions),
                )
            latest_elapsed_s = float(elapsed_s)
        return latest_elapsed_s

    def _upload_pending_shards(self, *, default_elapsed_s: float | None = None) -> float | None:
        with self._pending_upload_lock:
            queued_elapsed_s = self._materialize_queued_buffers_locked()
            return self._upload_pending_shards_locked(
                default_elapsed_s=(
                    queued_elapsed_s if queued_elapsed_s is not None else default_elapsed_s
                ),
            )

    def _try_upload_pending_shards(self, *, default_elapsed_s: float | None = None) -> float | None:
        """Drain pending shards unless another callback is already doing so.

        Shards are durable before this is called.  A busy uploader is therefore
        a reason to defer, not to stall a selfplay completion thread; the next
        flush or the session-end blocking drain retries every pending shard.
        """
        if not self._pending_upload_lock.acquire(blocking=False):
            return None
        try:
            queued_elapsed_s = self._materialize_queued_buffers_locked()
            return self._upload_pending_shards_locked(
                default_elapsed_s=(
                    queued_elapsed_s if queued_elapsed_s is not None else default_elapsed_s
                ),
            )
        finally:
            self._pending_upload_lock.release()

    def _report_bad_pending_shard(self, payload: dict[str, object]) -> None:
        try:
            self._requests.post(
                self._server_url_for(self.trial_api_prefix + "/report_bad_shard"),
                json=payload,
                auth=(str(self.args.username), str(self.args.password)),
                headers={
                    **_worker_headers(machine_id=self.machine_id),
                    **(
                        {"X-CAE-Worker-Lease-ID": str(self.lease_id)}
                        if str(self.lease_id).strip()
                        else {}
                    ),
                },
                timeout=15.0,
            )
        except Exception as exc:
            self.log.warning("failed to report bad local shard to server: %s", exc)

    def _upload_pending_shards_locked(self, *, default_elapsed_s: float | None = None) -> float | None:
        last_uploaded_at: float | None = None
        current_trial_id = self.leased_trial_id or self.fixed_trial_id or ""
        pending: list[Path] = [
            p for p in self.pending_dir.iterdir()
            if not p.name.startswith("_tmp_")
            and not p.name.startswith("._")
            and p.suffix == LOCAL_SHARD_SUFFIX
        ]
        for sp in sorted(pending):
            try:
                _arrs, meta = load_shard_arrays(sp, lazy=True)
                shard_trial_id = str(meta.get("run_id") or "").strip()
            except Exception as exc:
                self.log.warning(
                    "local pending shard %s is invalid before upload; sending to server for quarantine: %s",
                    sp, exc,
                )
                shard_trial_id = ""
            if shard_trial_id and shard_trial_id != current_trial_id:
                continue
            elapsed_s = default_elapsed_s
            elapsed_path = _pending_elapsed_path(sp)
            if elapsed_path.exists():
                try:
                    elapsed_s = float(elapsed_path.read_text(encoding="utf-8").strip())
                except Exception:
                    elapsed_s = default_elapsed_s
            try:
                upload_name, payload = pack_shard_for_upload(sp)
            except Exception as exc:
                reason = f"local pack failed: {type(exc).__name__}: {exc}"
                qpath = _quarantine_rejected_pending_shard(sp, reason) if sp.exists() else sp
                self.log.warning(
                    "local pending shard %s could not be packed; quarantined at %s: %s",
                    sp, qpath, exc,
                )
                self._report_bad_pending_shard(
                    _bad_shard_report_payload(
                        shard_path=sp, quarantine_path=qpath, reason=reason, kind="local_pack_failed",
                    ),
                )
                elapsed_path.unlink(missing_ok=True)
                continue
            files = {"file": (upload_name, payload, "application/x-tar")}
  # End-to-end integrity: the server hashes what it RECEIVES, so without a
  # digest computed here there is nothing to compare it against and a
  # corrupted transfer is indistinguishable from an honest upload. Plain
  # HTTP on the LAN means the only guard today is TCP's 16-bit checksum;
  # the shard ARRAYS are incidentally protected because Blosc/zstd fails to
  # decompress corrupt blocks, but `.zattrs` is uncompressed JSON where a
  # flipped digit stays valid JSON with a wrong number.
            payload_sha = hashlib.sha256(payload.getbuffer()).hexdigest()
            try:
                r = self._requests.post(
                    self._server_url_for(self.trial_api_prefix + "/upload_shard"),
                    files=files,
                    auth=(str(self.args.username), str(self.args.password)),
                    headers={
                        **_worker_headers(machine_id=self.machine_id),
                        UPLOAD_CONTENT_SHA256_HEADER: payload_sha,
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
            if _upload_response_allows_pending_delete(r):
                delete_shard_path(sp)
                elapsed_path.unlink(missing_ok=True)
                self.last_successful_send_s = time.time()
                last_uploaded_at = float(self.last_successful_send_s)
            else:
                rejection_reason = _upload_response_rejection_reason(r)
                if rejection_reason is not None:
                    qpath = _quarantine_rejected_pending_shard(sp, rejection_reason)
                    elapsed_path.unlink(missing_ok=True)
                    self.log.warning(
                        "server rejected pending shard %s; quarantined at %s: %s",
                        sp, qpath, rejection_reason,
                    )
                    continue
                if r.status_code == 200:
                    try:
                        body = r.json()
                    except Exception:
                        body = {}
                    self.log.warning(
                        "server did not accept pending shard %s; keeping for retry: %s",
                        sp, body,
                    )
                else:
      # ⚑ Previously a BARE `break` for every non-200: both helpers above
      # return "not accepted / not terminal", the 200-only branch is skipped,
      # and the worker stopped uploading with NO client-side log line at all.
      # The only signal anywhere was a server-side WARNING. Silent upload
      # wedges are this project's recurring outage class, and the lease
      # authorization added alongside this makes 403 the first DESIGNED
      # non-200 on this route -- undiagnosable in the field without this.
                    detail = ""
                    try:
                        detail = str(r.json().get("detail", ""))
                    except Exception:
                        detail = (r.text or "")[:200]
                    self.log.warning(
                        "server returned HTTP %s for pending shard %s; keeping for retry: %s",
                        r.status_code, sp, detail,
                    )
                break
        return last_uploaded_at

    def _load_and_compile_model(self, path: Path, cfg: ModelConfig, *, label: str, sha_short: str) -> torch.nn.Module:
        """Build model, load checkpoint, optionally compile."""
        model = build_model(cfg)
        # Slot width vs the model actually built here. NOT the production
        # selfplay check: a worker with a broker client does not reach this
        # function at all (see _require_slot_planes_match_manifest, which runs on
        # the manifest path every selfplay worker takes). This covers the case
        # where a local model IS built alongside a client — arena tasks — where
        # the built model is the more specific source of truth than the manifest.
        client = self.inference_client
        if client is not None:
            require_model_planes(
                model, int(client.input_planes),
                where="worker inference slot (local model)",
            )
  # weights_only=True blocks arbitrary pickle execution on a file that came
  # off the network. The manifest sha is verified before we get here, but the
  # sha arrives over the SAME channel as the file, so it does not authenticate
  # anything an on-path attacker could not also rewrite. Our published export
  # is `{"model": state_dict, "arch": {...primitives...}}` -- measured: a full
  # trainer.pt (model + opt + scheduler + zclip) loads clean under
  # weights_only=True, and this payload is a strict subset of that.
  # Same rationale as uci/model_loader.py.
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        sd = ckpt.get("model", ckpt)
        load_state_dict_tolerant(model, sd, label=label)
        # `.to(device)` is the first driver-touching call on this path and the
        # one observed to wedge (audit R1). It runs while a server lease is
        # already held, so an unbounded block here is a silently stalled worker.
        with self._boot_hang_watchdog.stage("model_to_device"):
            model.to(self.device)
        model.eval()
  # Selfplay only needs policy_own + wdl; skip 8 unused heads.
        if hasattr(model, "_inference_only"):
            setattr(model, "_inference_only", True)
        # ThreadedDispatcher compiles on its own thread so cudagraph TLS lives
        # where the forwards happen. Pre-compiling here would cause the
        # dispatcher's first forward to fall back to eager.
        if self.args.compile_inference and not self.args.threaded_selfplay:
            compile_t0 = time.time()
            self.log.info("compile starting %s sha=%s", label, sha_short)
            _compile_mode = str(self.args.compile_mode)
            with self._boot_hang_watchdog.stage("compile_inference_model"):
                model = _maybe_compile_inference_model(
                    model,
                    device=str(self.device),
                    mode=_compile_mode,
                    use_fp8=bool(getattr(self.args, "inference_fp8", False)),
                )
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

    def _require_reco(self, reco: dict, key: str, cast: Callable[[Any], Any] = float) -> Any:
        """Like ``_resolve_reco`` but with NO default — raise ``_MissingRequiredReco``
        if ``key`` is set neither on the CLI nor in the reco.

        For safety-critical knobs (``sf_nodes``) where a silent fallback would
        run the opponent at the wrong strength rather than refuse to play.
        """
        v = getattr(self.args, key, None)
        if v is None:
            v = reco.get(key)
        if v is None:
            raise _MissingRequiredReco(
                f"'{key}' is set on neither the CLI nor the recommended_worker manifest",
            )
        return cast(v)

    def _snapshot_reco(self, reco: dict) -> dict:
        """Snapshot every watched reco key (restart + live) for change detection."""
        return {k: reco.get(k) for k in self._RECO_WATCH_KEYS}

    def _active_difficulty(self) -> tuple[float | None, int | None]:
        """The opponent difficulty currently applied, for the shard's metadata.

        Read off ``_active_reco``, which is the snapshot the worker actually
        applied -- set at session start (``_run_selfplay``) and re-set on every
        live apply (``_maybe_restart_for_reco``). Reading the manifest instead
        would record what the SERVER published rather than what these games
        were played at, which is the whole distinction the field exists to make
        (a worker mid-poll is one manifest behind, and that lag is exactly what
        the promotion gate's anchored delta confounds on).

        ``(None, None)`` before any reco has been applied: absent means
        UNKNOWN, never 0.0.

        ⚑ CORRUPT is not ABSENT, and until this warning existed the two were
        the same observation. A non-numeric ``opponent_wdl_regret_limit`` or
        ``sf_nodes`` -- a string, a nested dict, a ``"None"`` literal, a float
        in the field this ``int()``s -- returned ``(None, None)``, byte
        identical to the legitimate "the server did not send these keys",
        with no log and no counter. The return value is still ``(None, None)``,
        because for METADATA purposes unknown is the honest answer; what
        changes is that it is now audible.

        ⚑ Scope, because the audit that raised this rated it HIGH on the
        premise that it silently reverts the worker to default DIFFICULTY, and
        that premise does not survive reading the call site. This method has
        two production callers, and NEITHER sets the difficulty: the
        per-game one (``_on_completed_game``) stamps the pair onto the
        SHARD'S METADATA, and the session-start log line added alongside this
        docstring reads it to print what the session adopted. The difficulty
        the games are
        actually played at comes from ``_build_selfplay_configs``, which reads
        the same two keys independently and casts them UNGUARDED -- a corrupt
        value raises there and takes the session down loudly. So the blast
        radius is provenance, not curriculum: shards recorded "difficulty
        unknown" while the server had in fact published a value. That still
        matters (the promotion gate's anchored current-vs-previous delta is
        checked against this field, and this project has been burned by rulers
        that silently lost rows), but it is a metrics defect, not a
        training-data one.
        """
        reco = getattr(self, "_active_reco", None)
        if not isinstance(reco, dict):
            return None, None
        raw_regret = reco.get("opponent_wdl_regret_limit")
        raw_nodes = reco.get("sf_nodes")
        try:
            regret = float(raw_regret) if raw_regret is not None else None
            nodes = int(raw_nodes) if raw_nodes is not None else None
        except (TypeError, ValueError) as exc:
            self._reco_corrupt_count += 1
            offending = (raw_regret, raw_nodes)
            if offending != self._reco_corrupt_last:
                self._reco_corrupt_last = offending
                self.log.warning(
                    "reco difficulty is CORRUPT, not absent: "
                    "opponent_wdl_regret_limit=%r sf_nodes=%r (%s). Shards "
                    "from these games will record difficulty as UNKNOWN, which "
                    "reads identically to a server that published no reco -- "
                    "the promotion gate's anchored delta cannot be checked "
                    "against them. Occurrences this session: %d",
                    raw_regret, raw_nodes, exc, self._reco_corrupt_count,
                )
            return None, None
        return regret, nodes

    def _asset_fingerprint(self, manifest: dict) -> tuple:
        """Published SHAs of the session-start assets that a live reco-apply
        cannot swap on a running session (SF binary — only when served — and the
        opening books). Compared against the session-start snapshot so an asset
        change forces a restart instead of being silently ignored."""
        def _sha(key: str) -> str | None:
            rec = manifest.get(key)
            return str(rec.get("sha256")) if isinstance(rec, dict) and rec.get("sha256") else None
        from_server = bool(getattr(self.args, "stockfish_from_server", False))
        sf_sha = _sha("stockfish") if from_server else None
  # The FEN list is a session-start asset ONLY for the opening_fen_prob sampling
  # path, which bakes it into OpeningConfig. In dole mode (prob 0) seeds arrive
  # through the live queue instead: a mid-session poll re-downloads the asset via
  # _sync_assets and _maybe_ingest_dole_flag refills _live_dole_queue in place, so
  # a new list needs NO restart. Fingerprinting it unconditionally routed every
  # list change into the restart branch and abandoned all in-flight games. With
  # the retire step rewriting the list every iteration (monitor RETIRE_EVERY=1)
  # that cost 58% of ALL started games and ~89% of curriculum games specifically
  # -- they are long enough to essentially never survive to a session boundary,
  # which is why the realized curriculum share was 16.5% against a configured
  # 65%. opening_fen_prob is itself a restart key, so a 0 -> positive transition
  # still rebuilds the session and rebakes the list.
        fen_sha = (
            _sha("opening_fen_list")
            if float((manifest.get("recommended_worker") or {}).get("opening_fen_prob", 0.0) or 0.0) > 0.0
            else None
        )
        return (
            sf_sha,
            _sha("opening_book"),
            _sha("opening_book_2"),
            fen_sha,
        )

    def _reco_changed(self, manifest: dict, *, source_tag: str) -> bool:
        """Return True (and request session restart) if reco knobs differ from active.

        Restart keys force a full session rebuild. Live keys (selfplay_fraction,
        opponent regret, SF node budgets) are applied to the running session in
        place when possible — every consumer reads them fresh per step/move/
        recycle, so a restart (which abandons the 256 in-flight games and
        collapses curriculum throughput for ~2 iters) is unnecessary.
        """
        new_reco = manifest.get("recommended_worker") or {}
        active = getattr(self, "_active_reco", None)
        if active is None:
            return False
  # A CLI-pinned games_per_batch wins over reco (see _run_selfplay), so a
  # server-side change to it would otherwise force a no-op restart. Drop it from
  # restart detection when pinned.
        restart_keys = self._RECO_RESTART_KEYS
        if getattr(self, "games_per_batch_local", None) is not None:
            restart_keys = tuple(k for k in restart_keys if k != "games_per_batch")
        changed_restart_keys = [
            k for k in restart_keys if new_reco.get(k) != active.get(k)
        ]
        restart_changed = bool(changed_restart_keys)
        live_changed = any(
            new_reco.get(k) != active.get(k) for k in self._RECO_LIVE_KEYS
        )
  # Session-start assets (SF binary, opening books) are applied only when a new
  # session starts — _sync_stockfish / the OpeningConfig bake them in. If they
  # changed alongside only live reco keys, the live path would keep the stale
  # engine/book, so treat an asset change as a restart trigger too.
        assets_changed = self._asset_fingerprint(manifest) != getattr(self, "_active_assets", None)
        if not restart_changed and not live_changed and not assets_changed:
            return False
  # Live-only change: apply in place if a session is running, no restart.
        if not restart_changed and not assets_changed and self._apply_live_reco(new_reco):
            self._active_reco = self._snapshot_reco(new_reco)
            self.log.info("recommended_worker live-applied (%s), no session restart", source_tag)
            return False
  # Name the trigger: a restart abandons every in-flight game, so a recurring
  # one must be attributable without a code change to find it.
        self.log.info(
            "recommended_worker changed (%s), restarting selfplay session "
            "[restart_keys=%s assets_changed=%s]",
            source_tag,
            ",".join(changed_restart_keys) or "-",
            assets_changed,
        )
        self._stop_selfplay = True
        return True

    def _register_live_state(self, state: Any) -> None:
        """play_batch on_state_ready hook: register a live state for live reco.

        Called once per play_batch session start — once total on the single
        path, once per thread on the threaded path. If a live reco apply
        already ran this session (threads register over ms-s while publishes
        land any time), the stashed override is transplanted onto the late
        registrant here, under the same lock — otherwise it would run the
        whole continuous session on session-start config (stale regret) while
        _active_reco already claims the new values."""
        with self._live_states_lock:
            self._live_states.append(state)
            if self._pending_live_override is not None:
                self._transplant_live_fields(state, *self._pending_live_override)
        # Every slot is a fresh game at this point, so each is a valid target
        # for a persisted one. Files are claimed by rename, so the threaded
        # path's N registrations divide the backlog instead of racing for it.
        if self._resume_inflight_enabled:
            self._resume_inflight_games(state)

    def _clear_live_states(self) -> None:
        """Drop all registered states (and the session's pending live override)
        so a between-session reco poll restarts cleanly rather than mutating
        dead states."""
        with self._live_states_lock:
            self._live_states.clear()
            self._pending_live_override = None

    @staticmethod
    def _transplant_live_fields(
        state: Any, new_game: Any, new_opp: Any, sf_nodes: int,
    ) -> None:
        """Transplant ONLY the live-safe fields onto one running state's config
        (never the whole rebuilt config — session-fixed fields must not change
        mid-session). Must stay in sync with _RECO_LIVE_KEYS (asserted by
        test_every_live_key_is_transplanted)."""
        game = dataclasses.replace(
            state.game,
            selfplay_fraction=new_game.selfplay_fraction,
            sf_fast_ply_node_scale=new_game.sf_fast_ply_node_scale,
            sf_label_nodes_cap=new_game.sf_label_nodes_cap,
            sf_label_nodes_floor=new_game.sf_label_nodes_floor,
            sf_label_escalate_q_gap=new_game.sf_label_escalate_q_gap,
            sf_label_escalate_nodes=new_game.sf_label_escalate_nodes,
            sf_label_escalate_max_per_game=new_game.sf_label_escalate_max_per_game,
        )
        opponent = dataclasses.replace(
            state.opponent, wdl_regret_limit=new_opp.wdl_regret_limit,
        )
        state.apply_live_overrides(game=game, opponent=opponent, base_nodes=sf_nodes)

    def _maybe_ingest_dole_flag(self, manifest: dict) -> None:
        """If the server doled THIS iteration's seed batch to our poll, load the
        whole FEN list and hand it to selfplay.

        The server sets ``dole_fen_seeds: true`` on exactly one worker-poll per
        iteration (see the server's seed-dole gate), so total seeded games ==
        N*len(list) per iteration regardless of client count. We play them as
        selfplay seeds. Delivery depends on whether a session is running:
          * session live (``_live_dole_queue`` set): refill that shared queue in
            place — the running selfplay (single OR every thread of the threaded
            path) drains it, so a fresh iteration's dole supersedes any undrained
            backlog without a restart;
          * no session: stash into ``_pending_fen_dole`` for the next session
            start (survives model-not-ready / dispatch-failure retries — the
            stash is only cleared once a session actually consumes it).
        No-op unless dole mode is on (opening_fen_dole_per_iter>0) and a FEN list
        is available.

        Ingestion rides the HTTP manifest poll only (the mtime model-swap fast
        path reads the local manifest file, which never carries the per-request
        ``dole_fen_seeds`` flag). That is fine while the training iteration is
        longer than the poll interval — production iters are minutes vs a ~30s
        poll, so exactly one poll per iteration wins the claim — but a smoke
        config with sub-poll-interval iterations can advance past an unpolled
        iteration and skip its dole."""
        if not bool(manifest.get("dole_fen_seeds")):
            return
        reco = manifest.get("recommended_worker") or {}
        n = int(reco.get("opening_fen_dole_per_iter", 0) or 0)
        if n <= 0 or not self.opening_fen_list_path:
            return
        try:
            seeds = list(_load_fen_list(str(self.opening_fen_list_path)))
        except Exception as exc:
            self.log.warning(
                "dole: failed to load FEN list %s: %s", self.opening_fen_list_path, exc,
            )
            return
        # Per-seed weight= markers multiply exposure inside one dose (and
        # weight=0 soft-retires a seed from the dole); the global n scales
        # the whole batch.
        queue = expand_dole_seeds(seeds, str(self.opening_fen_list_path)) * n
        # SF-refute channel: round-robin slice of the UNEXPANDED list (one
        # pass per seed in the window; weights already shape the selfplay dole).
        sf_frac = float(reco.get("opening_fen_sf_refute_frac", 0.0) or 0.0)
        sf_plies = int(reco.get("opening_fen_sf_refute_plies", 5) or 0)
        train_iter = int(manifest.get("training_iteration", 0) or 0)
        sf_queue: list[str] = []
        if sf_frac > 0.0 and sf_plies > 0:
            sf_queue = sample_sf_refute_batch(
                seeds,
                frac=sf_frac,
                training_iteration=train_iter,
                path=str(self.opening_fen_list_path),
            )
        # Bound total seeded games/iter. Without this the seeded share of
        # selfplay is just pool_size/capacity: on 2026-07-24 a growing seed pool
        # (141 -> 341) pushed it 81.8% -> 100%, leaving ZERO normal-opening
        # selfplay games. Rotates both channels so no seed is starved.
        max_games = int(reco.get("opening_fen_dole_max_games", 0) or 0)
        if max_games > 0:
            before = (len(queue), len(sf_queue))
            queue, sf_queue = cap_dole_batch(
                queue, sf_queue, max_games=max_games, training_iteration=train_iter,
            )
            if (len(queue), len(sf_queue)) != before:
                self.log.info(
                    "dole: capped seeded games %d+%d -> %d+%d (max %d)",
                    before[0], before[1], len(queue), len(sf_queue), max_games,
                )
        with self._dole_lock:
            live = self._live_dole_queue
            if live is not None:
  # Refill in place so the drainers (which hold this same object) see the batch.
                live.clear()
                live.extend(queue)
                dest = "live session"
            else:
                self._pending_fen_dole = queue
                dest = "next session"
            live_sf = self._live_sf_refute_queue
            if live_sf is not None:
                live_sf.clear()
                live_sf.extend(sf_queue)
            else:
                self._pending_sf_refute = list(sf_queue)
        self.log.info(
            "dole: received %d seed(s) x%d -> %s; sf_refute=%d (frac=%.2f plies=%d iter=%d)",
            len(seeds), n, dest, len(sf_queue), sf_frac, sf_plies, train_iter,
        )

    def _apply_live_reco(self, reco: dict) -> bool:
        """Swap live-safe config onto every registered SelfplayState. Returns
        False if no live session is active (caller falls back to a restart)."""
        with self._live_states_lock:
            states = list(self._live_states)
        if not states:
            return False
        try:
            cfgs, (sf_nodes, _sf_multipv, _sf_hash_mb, _sz) = self._build_selfplay_configs(reco)
        except _MissingRequiredReco:
  # Incomplete reco (e.g. sf_nodes dropped) — restart will re-poll cleanly.
            return False
        except Exception as exc:
  # A malformed reco field can make config construction raise. Don't let the
  # caller's broad except swallow it silently (which would pin the worker at
  # the old config with no restart): log and fall back to a restart, whose
  # bring-up re-parses the reco and surfaces/handles the error as before.
            self.log.warning(
                "live reco apply failed to build configs (%s); falling back to restart",
                exc, exc_info=True,
            )
            return False
  # Transplant ONLY the live-safe fields onto the running config — never the
  # whole rebuilt config. Other GameConfig fields (max_plies, sf_policy_temp,
  # ...) are built from the same reco but are session-fixed; swapping the whole
  # object would let an untracked field change take effect mid-session purely
  # because a live key changed in the same publish. The live-field set here must
  # stay in sync with _RECO_LIVE_KEYS (asserted by
  # test_every_live_key_is_transplanted). recycle_slot / the stockfish turn read
  # these refs fresh, so the swap takes effect immediately.
        new_game, new_opp = cfgs["game"], cfgs["opponent"]
  # Apply under the registry lock and stash the override inside the same
  # critical section: a thread registering concurrently either lands before
  # the apply (gets the loop) or after (gets the stash in
  # _register_live_state) — no ordering can leave it on stale config.
        with self._live_states_lock:
            states = list(self._live_states)
            for state in states:
                self._transplant_live_fields(state, new_game, new_opp, sf_nodes)
            self._pending_live_override = (new_game, new_opp, int(sf_nodes))
        self._set_sf_nodes(sf_nodes)
        last = states[-1]
        self.log.info(
            "live reco (%d state(s)): selfplay_fraction=%.3f regret=%s sf_nodes=%d"
            " label_floor=%d",
            len(states),
            float(last.game.selfplay_fraction),
            (f"{last.opponent.wdl_regret_limit:.4f}")
            if last.opponent.wdl_regret_limit is not None else "none",
            int(sf_nodes),
            int(getattr(last.game, "sf_label_nodes_floor", 0) or 0),
        )
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
                    if self._hold_on_pause:
                        self._hold_selfplay = True
                    else:
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
            reco_changed = self._reco_changed(manifest, source_tag="poll")
            if reco_changed:
  # A boundary poll can both win the dole and trigger a restart (restart-key or
  # session-start asset change). Ingest the dole BEFORE the (fallible) model swap
  # so a mid-publish download/compile error there — swallowed by the outer except
  # — can't drop the batch: refresh the FEN asset first (the restart path skips
  # _sync_assets below, so a changed opening_fen_list would otherwise load stale),
  # then refill the live queue (undrained seeds ride into the next session as
  # leftover). Both best-effort; the model swap still runs.
                if bool(manifest.get("dole_fen_seeds")):
                    with suppress(Exception):
                        self._sync_opening_books(manifest)
                    with suppress(Exception):
                        self._maybe_ingest_dole_flag(manifest)
                self._swap_model_from_manifest(manifest)
                return
            self._sync_assets(manifest)
  # Mid-session: if the server doled this iteration's seed batch to this poll,
  # push it onto the live SelfplayState (recycled selfplay slots drain it).
            self._maybe_ingest_dole_flag(manifest)
        except Exception as _exc:
            self.log.warning(
                "mid-session manifest poll failed (dole/live-reco/assets may lag): %s",
                _exc,
                exc_info=True,
            )

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

    def _resync_evaluator_to_model(self) -> None:
        """Point the DirectGPU evaluator at the current self.model if it drifted.
        Single source of truth for the post-model-swap sync (the swap path, the
        poll/_sync_model path, and session start all call this). Idempotent and a
        no-op before the evaluator exists. Callers that mutate shared inference
        state concurrently (threaded selfplay) must hold the upload-buffer lock —
        the swap path does; the non-production threaded poll path is the only one
        that doesn't, and threaded selfplay is not used in production."""
        if (
            self._direct_evaluator is not None
            and self.model is not None
            and self._evaluator_model_id != id(self.model)
        ):
            _sync_evaluator_to_model(self._direct_evaluator, self.model)
            self._evaluator_model_id = id(self.model)

    def _swap_model_from_manifest(self, manifest: dict) -> None:
        """Tier 2 inner: download new SHA if changed, hot-swap model, flush old shards."""
        new_sha = str(manifest.get("model", {}).get("sha256", ""))
        if not new_sha or new_sha == self.model_sha:
            return
        # The mid-batch tier-2 path also returns early for client-only workers
        # (it only re-tags), so check the width here as well.
        self._require_slot_planes_match_manifest(manifest)
        model_info = manifest.get("model") or {}
        model_step = int(manifest.get("trainer_step") or 0)
        if self.inference_client is not None:
            self._flush_pre_swap_buffer_if_stale(model_sha=new_sha, model_step=model_step)
            self.model_sha = new_sha
            self.model_step = model_step
            self.last_model_sha = new_sha
            self.log.info("mid-batch model tag switch sha=%s", str(new_sha)[:8])
            return
        model_path = self.cache_dir / f"model_{new_sha}.pt"
        if (not model_path.exists()) or (_sha256_file(model_path) != new_sha):
            _download_and_verify_shared(
                self._server_url_for(str(model_info.get("endpoint") or (self.trial_api_prefix + "/model"))),
                out_path=model_path,
                expected_sha256=new_sha,
                headers=_worker_headers(),
            )
        mc = manifest.get("model_config")
        if isinstance(mc, ModelConfig):
            _cfg = mc
        elif isinstance(mc, dict) and mc:
            _cfg = model_config_from_manifest_dict(mc)
        else:
            _cfg = self.model_cfg_active
        if _cfg is None:
            raise RuntimeError("no model_config available for worker-model load")
        _cfg = dataclasses.replace(_cfg, use_gradient_checkpointing=False)
        new_model = self._load_and_compile_model(
            model_path, _cfg,
            label="worker-model", sha_short=str(new_sha)[:8],
        )
        # Flush the old-model upload buffer and swap evaluator/metadata as one
        # critical section. In threaded selfplay, completed-game callbacks use
        # this same lock before reading self.model_sha for shard metadata.
        buf_lock = self._upload_buf_lock
        now = time.time()
        flushed_elapsed_s: float | None = None
        with buf_lock if buf_lock is not None else nullcontext():
            if self.upload_buf.positions > 0:
                _shard_path, elapsed_s = _flush_upload_buffer_to_pending(
                    pending_dir=self.pending_dir, username=str(self.args.username),
                    buf=self.upload_buf, now_s=now,
                    trial_id=self.leased_trial_id or self.fixed_trial_id or None,
                )
                flushed_elapsed_s = float(elapsed_s)
            self.model = new_model
            self._resync_evaluator_to_model()
            self.model_sha = new_sha
            self.model_step = model_step
            self.last_model_sha = new_sha
        if flushed_elapsed_s is not None:
            self._upload_pending_shards(default_elapsed_s=flushed_elapsed_s)
        self.log.info("mid-batch model switch sha=%s", str(new_sha)[:8])

    def _check_model_update(self) -> None:
        """Check for new model between moves (called every ply by play_batch).

        Three-tier check:
        1. Every 30s: re-poll manifest for task/pause changes (sets _stop_selfplay)
        2. Every call: stat() the local manifest file mtime (~1µs)
        3. Only if mtime changed: read manifest, download model, swap

        Runs from two callers — play_batch's per-ply ``on_step`` (sub-second
        swap latency) and the model-watch thread (liveness floor). Neither may
        run while the other is mid-swap, so a contended call is SKIPPED rather
        than queued: whichever caller holds the lock is already doing this
        exact work, and blocking here would park a selfplay thread behind a
        network download.
        """
        if not self._model_watch_lock.acquire(blocking=False):
            return
        try:
            self._check_model_update_locked()
        finally:
            self._model_watch_lock.release()

    def _check_model_update_locked(self) -> None:
        """Body of _check_model_update; caller holds _model_watch_lock."""
        _now = time.time()
        if _now - self._last_manifest_poll_s > 30.0:
            self._last_manifest_poll_s = _now
            self._maybe_log_dispatcher_stats(_now)
            self._maybe_log_broker_client_stats(_now)
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
            if self._hold_selfplay:
  # Held: the freshly published manifest is the unpause signal —
  # _check_pause_selfplay clears pause_selfplay_active/_hold_selfplay
  # when the pause flag dropped (and just short-sleeps if still paused).
                self._check_pause_selfplay(manifest)
            reco_changed = self._reco_changed(manifest, source_tag="mtime")
            self._swap_model_from_manifest(manifest)
            if reco_changed:
                return
        except Exception as _exc:
            self.log.warning("mid-batch model check failed (worker stuck on old model): %s", _exc, exc_info=True)

    def _maybe_log_dispatcher_stats(self, now_s: float) -> None:
        if not isinstance(self._direct_evaluator, ThreadedDispatcher):
            return
        elapsed_s = float(now_s - self._last_dispatcher_stats_log_s)
        if elapsed_s < 60.0:
            return
        ds = self._direct_evaluator.stats
        batches = int(ds["lifetime_batches"])
        positions = int(ds["lifetime_positions"])
        forward_rows = int(ds.get("lifetime_forward_rows", positions))
        legal_bf16_positions = int(ds.get("lifetime_legal_bf16_positions", 0))
        forward_s = float(ds["lifetime_forward_s"])
        pack_s = float(ds.get("lifetime_pack_s", 0.0))
        wait_s = float(ds.get("lifetime_wait_s", 0.0))
        scatter_s = float(ds.get("lifetime_scatter_s", 0.0))
        drain_s = float(ds.get("lifetime_drain_s", 0.0))
        (
            prev_batches, prev_positions, prev_forward_rows, prev_forward_s, prev_wait_s,
            prev_pack_s, prev_scatter_s, prev_drain_s,
        ) = self._last_dispatcher_stats_snapshot
        delta_batches = max(0, batches - int(prev_batches))
        delta_positions = max(0, positions - int(prev_positions))
        delta_forward_rows = max(0, forward_rows - int(prev_forward_rows))
        delta_forward_s = max(0.0, forward_s - float(prev_forward_s))
        delta_wait_s = max(0.0, wait_s - prev_wait_s)
        delta_pack_s = max(0.0, pack_s - prev_pack_s)
        delta_scatter_s = max(0.0, scatter_s - prev_scatter_s)
        delta_drain_s = max(0.0, drain_s - prev_drain_s)
        self._last_dispatcher_stats_log_s = float(now_s)
        self._last_dispatcher_stats_snapshot = (
            batches, positions, forward_rows, forward_s, wait_s, pack_s, scatter_s, drain_s,
        )
        if delta_batches <= 0:
            return
        self.log.info(
            "dispatcher live stats: batches=%d(+%d) avg_batch=%.1f "
            "req_per_batch=%.1f rows_per_req=%.1f q_avg=%.1f q_max=%d "
            "avg_drain_ms=%.2f avg_pack_ms=%.2f avg_submit_ms=%.2f "
            "avg_wait_ms=%.2f avg_scatter_ms=%.2f leaf_per_s=%.0f "
            "fwd_rows_per_s=%.0f pad_ratio=%.2f "
            "forward_busy=%.1f%% pack_busy=%.1f%% wait_busy=%.1f%% "
            "scatter_busy=%.1f%% drain_busy=%.1f%% legal_bf16=%.1f%% "
            "complete_gps=%.2f complete_pos_s=%.0f callback_busy=%.1f%% "
            "upload_busy=%.1f%% active_threads=%d thread_game_skew=%d "
            "full_drains=%d",
            batches, delta_batches,
            float(ds["avg_batch_size"]),
            float(ds.get("avg_requests_per_batch", 0.0)),
            float(ds.get("avg_rows_per_request", 0.0)),
            float(ds.get("avg_queue_depth", 0.0)),
            int(ds.get("queue_depth_max", 0)),
            float(ds.get("avg_drain_ms", 0.0)),
            float(ds.get("avg_pack_ms", 0.0)),
            float(ds.get("avg_submit_ms", ds["avg_forward_ms"])),
            float(ds.get("avg_wait_ms", 0.0)),
            float(ds.get("avg_scatter_ms", 0.0)),
            float(delta_positions / max(1e-6, elapsed_s)),
            float(delta_forward_rows / max(1e-6, elapsed_s)),
            float(delta_forward_rows / max(1, delta_positions)),
            float(100.0 * delta_forward_s / max(1e-6, elapsed_s)),
            float(100.0 * delta_pack_s / max(1e-6, elapsed_s)),
            float(100.0 * delta_wait_s / max(1e-6, elapsed_s)),
            float(100.0 * delta_scatter_s / max(1e-6, elapsed_s)),
            float(100.0 * delta_drain_s / max(1e-6, elapsed_s)),
            float(100.0 * legal_bf16_positions / max(1, positions)),
            *self._completion_telemetry_delta(elapsed_s),
            int(ds["lifetime_full_drains"]),
        )
        self._maybe_log_selfplay_phase_stats(elapsed_s)

    def _maybe_log_broker_client_stats(self, now_s: float) -> None:
        if not isinstance(self.inference_client, MultiSlotInferenceClient):
            return
        elapsed_s = float(now_s - self._last_broker_client_stats_log_s)
        if elapsed_s < 60.0:
            return
        ds = self.inference_client.stats
        prev = self._last_broker_client_stats_snapshot
        requests = int(ds["lifetime_requests"])
        positions = int(ds["lifetime_positions"])
        legal_requests = int(ds.get("lifetime_legal_requests", 0))
        legal_positions = int(ds.get("lifetime_legal_positions", 0))
        wait_s = float(ds.get("lifetime_wait_s", 0.0))
        roundtrip_s = float(ds.get("lifetime_roundtrip_s", 0.0))
        served_requests = int(ds.get("lifetime_served_requests", requests))
        failed_requests = int(ds.get("lifetime_failed_requests", 0))
        # SHARED_BROKER_AUDIT B6: a wedged slot is INVISIBLE in slot_requests
        # (attempts are uniform across live and dead slots -- measured
        # [3, 3, 3, 3] with slot 3 serving nothing). slot_served separates
        # them, and slot_roundtrip_s -- computed since forever and read by
        # nothing until now -- carries the 100x separation that identifies
        # WHICH slot is stalling.
        slot_requests = [int(x) for x in ds.get("slot_requests", [])]
        slot_served = [int(x) for x in ds.get("slot_served", slot_requests)]
        slot_roundtrip_s = [float(x) for x in ds.get("slot_roundtrip_s", [])]

        prev_requests = int(prev.get("requests", 0))
        prev_positions = int(prev.get("positions", 0))
        prev_legal_requests = int(prev.get("legal_requests", 0))
        prev_legal_positions = int(prev.get("legal_positions", 0))
        prev_wait_s = float(prev.get("wait_s", 0.0))
        prev_roundtrip_s = float(prev.get("roundtrip_s", 0.0))
        prev_served_requests = int(prev.get("served_requests", 0))
        prev_failed_requests = int(prev.get("failed_requests", 0))
        prev_slot_requests = [int(x) for x in prev.get("slot_requests", [0] * len(slot_requests))]
        prev_slot_served = [int(x) for x in prev.get("slot_served", [0] * len(slot_served))]
        prev_slot_roundtrip_s = [
            float(x) for x in prev.get("slot_roundtrip_s", [0.0] * len(slot_roundtrip_s))
        ]

        delta_requests = max(0, requests - prev_requests)
        delta_positions = max(0, positions - prev_positions)
        delta_legal_requests = max(0, legal_requests - prev_legal_requests)
        delta_legal_positions = max(0, legal_positions - prev_legal_positions)
        delta_wait_s = max(0.0, wait_s - prev_wait_s)
        delta_roundtrip_s = max(0.0, roundtrip_s - prev_roundtrip_s)
        delta_served = max(0, served_requests - prev_served_requests)
        delta_failed = max(0, failed_requests - prev_failed_requests)
        slot_attempt_deltas = [
            max(0, count - (prev_slot_requests[idx] if idx < len(prev_slot_requests) else 0))
            for idx, count in enumerate(slot_requests)
        ]
        slot_deltas = [
            max(0, count - (prev_slot_served[idx] if idx < len(prev_slot_served) else 0))
            for idx, count in enumerate(slot_served)
        ]
        active_slot_deltas = [count for count in slot_deltas if count > 0]
        active_slots = len(active_slot_deltas)
        slot_skew = max(active_slot_deltas) - min(active_slot_deltas) if active_slot_deltas else 0
        # Worst per-slot mean roundtrip in this window. A wedged slot reads
        # ~request_timeout_s here while healthy ones read single-digit ms.
        slot_rt_ms = [
            1000.0
            * max(0.0, rt - (prev_slot_roundtrip_s[idx] if idx < len(prev_slot_roundtrip_s) else 0.0))
            / max(1, slot_attempt_deltas[idx] if idx < len(slot_attempt_deltas) else 0)
            for idx, rt in enumerate(slot_roundtrip_s)
        ]
        slot_rt_ms_max = max(slot_rt_ms) if slot_rt_ms else 0.0

        self._last_broker_client_stats_log_s = float(now_s)
        self._last_broker_client_stats_snapshot = {
            "requests": requests,
            "positions": positions,
            "legal_requests": legal_requests,
            "legal_positions": legal_positions,
            "wait_s": wait_s,
            "roundtrip_s": roundtrip_s,
            "served_requests": served_requests,
            "failed_requests": failed_requests,
            "slot_requests": slot_requests,
            "slot_served": slot_served,
            "slot_roundtrip_s": slot_roundtrip_s,
        }
        # delta_requests counts ATTEMPTS, so a window in which every request
        # FAILED still has delta_requests > 0 and reaches the log -- the gate
        # below is unchanged from main and no `or delta_failed` clause is
        # needed to make that true. It is stated (and pinned by
        # test_a_window_where_every_request_failed_still_logs) because the
        # obvious refactor once served/failed exist is to key this on
        # delta_served, which WOULD go silent for exactly the stall this line
        # exists to show.
        if delta_requests <= 0:
            return

        completion_stats = self._completion_telemetry_delta(elapsed_s)
        self.log.info(
            "broker client stats: requests=%d(+%d) rows_per_req=%.1f "
            "req_s=%.1f pos_s=%.0f wait_ms=%.2f roundtrip_ms=%.2f "
            "wait_thread_busy=%.1f%% roundtrip_thread_busy=%.1f%% "
            "legal_req=%.1f%% legal_pos=%.1f%% slots_active=%d/%d "
            "slot_req_skew=%d slot_rt_ms_max=%.0f max_inflight=%d available=%d "
            "complete_gps=%.2f complete_pos_s=%.0f callback_busy=%.1f%% "
            "upload_busy=%.1f%% active_threads=%d thread_game_skew=%d"
            # Only when non-zero, so the healthy line is unchanged and any
            # non-zero value is a grep hit rather than something to eyeball.
            # Counts responses the client refused because the broker echoed a
            # different request id -- i.e. the timeout/re-submit race (audit
            # I1) actually occurred and was caught rather than silently
            # feeding another position's policy into MCTS.
            + (
                f" stale_responses_rejected={int(ds.get('stale_responses_rejected', 0))}"
                if int(ds.get("stale_responses_rejected", 0)) else ""
            )
            # Same rule for the B6 counters: absent on a healthy run, and a
            # grep hit the moment a slot starts failing. failed_req is the
            # number rows_per_req/pos_s above deliberately EXCLUDE.
            + (
                f" failed_req=+{int(delta_failed)}"
                f" failed_req_total={int(failed_requests)}"
                if delta_failed or failed_requests else ""
            )
            + (
                f" slots_quarantined={int(ds.get('slots_quarantined', 0))}"
                if int(ds.get("slots_quarantined", 0)) else ""
            ),
            requests, delta_requests,
            float(delta_positions / max(1, delta_served)),
            float(delta_requests / max(1e-6, elapsed_s)),
            float(delta_positions / max(1e-6, elapsed_s)),
            float(1000.0 * delta_wait_s / max(1, delta_requests)),
            float(1000.0 * delta_roundtrip_s / max(1, delta_requests)),
            float(100.0 * delta_wait_s / max(1e-6, elapsed_s)),
            float(100.0 * delta_roundtrip_s / max(1e-6, elapsed_s)),
            float(100.0 * delta_legal_requests / max(1, delta_served)),
            float(100.0 * delta_legal_positions / max(1, delta_positions)),
            active_slots,
            int(ds.get("slots", 0)),
            int(slot_skew),
            float(slot_rt_ms_max),
            int(ds.get("max_inflight", 0)),
            int(ds.get("available_slots", 0)),
            *completion_stats,
        )
        self._maybe_log_selfplay_phase_stats(elapsed_s)

    def _completion_telemetry_delta(self, elapsed_s: float) -> tuple[float, float, float, float, int, int]:
        with self._completion_telemetry_lock:
            games = int(self._completion_games)
            positions = int(self._completion_positions)
            callback_s = float(self._completion_callback_s)
            upload_s = float(self._completion_upload_s)
            by_thread = dict(self._completion_by_thread)
            (
                prev_games, prev_positions, prev_callback_s, prev_upload_s, prev_by_thread,
            ) = self._last_completion_stats_snapshot
            self._last_completion_stats_snapshot = (
                games, positions, callback_s, upload_s, by_thread,
            )

        delta_games = max(0, games - int(prev_games))
        delta_positions = max(0, positions - int(prev_positions))
        delta_callback_s = max(0.0, callback_s - float(prev_callback_s))
        delta_upload_s = max(0.0, upload_s - float(prev_upload_s))
        thread_deltas = [
            max(0, int(count) - int(prev_by_thread.get(tid, 0)))
            for tid, count in by_thread.items()
        ]
        active = [count for count in thread_deltas if count > 0]
        skew = (max(active) - min(active)) if active else 0
        return (
            float(delta_games / max(1e-6, elapsed_s)),
            float(delta_positions / max(1e-6, elapsed_s)),
            float(100.0 * delta_callback_s / max(1e-6, elapsed_s)),
            float(100.0 * delta_upload_s / max(1e-6, elapsed_s)),
            len(active),
            int(skew),
        )

    def _record_selfplay_phase_timing(self, phase: str, seconds: float) -> None:
        if seconds <= 0.0:
            return
        with self._phase_timing_lock:
            self._phase_timing_s[phase] = float(self._phase_timing_s.get(phase, 0.0)) + float(seconds)

    def _note_selfplay_progress(self) -> None:
        self._last_selfplay_progress_s = time.time()

    def _selfplay_stalled(self, now: float, timeout_s: float) -> bool:
        """True iff the active selfplay session has hung past ``timeout_s``.

        A pause-hold (``distributed_pause_selfplay_during_training`` or a
        graceful-restart hold) is intentionally idle and may last hours, so it
        refreshes the progress timer rather than counting as a stall — the
        watchdog must never hard-exit a healthy paused worker.
        """
        if not self._selfplay_session_active:
            return False
        if self._hold_selfplay:
            self._last_selfplay_progress_s = now
            return False
        return (now - float(self._last_selfplay_progress_s)) >= timeout_s

    def _start_selfplay_stall_watchdog(self) -> None:
        """Exit the process if a selfplay session produces no progress for too long.

        Covers hangs that per-request timeouts miss (e.g. all slots held and
        threads parked in an unbounded wait). Prefer fixing the hang; this is
        the automatic recovery the launcher already expects when a worker dies.
        """
        if self._stall_watchdog_started:
            return
        raw = os.environ.get("CAE_WORKER_STALL_TIMEOUT_S", "300").strip()
        try:
            timeout_s = float(raw)
        except ValueError:
            timeout_s = 300.0
            self.log.warning(
                "CAE_WORKER_STALL_TIMEOUT_S=%r is not a number; using the "
                "default %.1fs instead. The value you exported is NOT in "
                "effect", raw, timeout_s,
            )
        if timeout_s <= 0.0:
            self.log.info("stall watchdog DISABLED (realized timeout_s=%.1f)", timeout_s)
            return
        self._stall_watchdog_started = True
        poll_s = min(30.0, max(5.0, timeout_s / 10.0))
  # ⚑ Print the REALIZED values, always -- not only on the rejection path.
  # This is a safety device: running it at 300s when the operator exported
  # 600s means it fires on healthy work, and until this line existed there was
  # no observation anywhere that distinguished "600 in force" from "600
  # rejected, using 300". Same family as this repo's standing "a yaml edit +
  # restart may NOT be in effect" / "dead config keys pin to realized" rules,
  # reproduced in the env-var path.
        self.log.info(
            "stall watchdog started: timeout_s=%.1f poll_s=%.1f "
            "(CAE_WORKER_STALL_TIMEOUT_S=%r)", timeout_s, poll_s, raw,
        )

        def _loop() -> None:
            while True:
                time.sleep(poll_s)
                if not self._selfplay_stalled(time.time(), timeout_s):
                    continue
                idle_s = time.time() - float(self._last_selfplay_progress_s)
                self.log.error(
                    "selfplay stalled for %.0fs (limit=%.0fs) with no phase-stats/"
                    "completions; exiting so the launcher restarts this worker "
                    "(set CAE_WORKER_STALL_TIMEOUT_S=0 to disable)",
                    idle_s, timeout_s,
                )
                # Hard exit: daemon threads / stuck GIL holders will not join.
                os._exit(2)

        threading.Thread(
            target=_loop, name="selfplay-stall-watchdog", daemon=True,
        ).start()

    def _start_model_watch_thread(self) -> None:
        """Own manifest polling and model swaps on a dedicated daemon thread.

        Model freshness is a worker-GLOBAL invariant, but it used to be driven
        only by play_batch's ``on_step`` hook, which the threaded selfplay path
        wires to selfplay thread 0 alone. Threads are submitted to a pool and
        joined via ``future.result()`` only at session end — which in continuous
        mode never arrives — so if that one thread died or parked, the hook went
        with it, silently: the other 31 threads kept playing and uploading, every
        shard tagged with a frozen ``model_sha``, and the trainer counted all of
        it stale. Observed 2026-07-24: all four workers frozen on one sha for 80+
        minutes, ~75% of selfplay excluded from the iteration for days before
        that (one worker's per-iteration session restart was accidentally
        masking it).

        This thread does the same work on a fixed cadence and cannot be taken
        down by a selfplay thread, so the worst case is a slow swap, never a
        permanently stale one.
        """
        if self._model_watch_started:
            return
        raw = os.environ.get("CAE_WORKER_MODEL_WATCH_S", "5").strip()
        try:
            poll_s = float(raw)
        except ValueError:
            poll_s = 5.0
            self.log.warning(
                "CAE_WORKER_MODEL_WATCH_S=%r is not a number; using the "
                "default %.1fs instead. The value you exported is NOT in "
                "effect", raw, poll_s,
            )
        if poll_s <= 0.0:
            self.log.info("model watch DISABLED (realized poll_s=%.1f)", poll_s)
            return
        self._model_watch_started = True
  # Realized value, on every start -- see the stall watchdog above.
        self.log.info(
            "model watch started: poll_s=%.1f (CAE_WORKER_MODEL_WATCH_S=%r)",
            poll_s, raw,
        )

        def _loop() -> None:
            while True:
                time.sleep(poll_s)
                if not self._selfplay_session_active:
                    # Between sessions the outer run() loop owns the manifest;
                    # a concurrent swap here would race its session setup.
                    continue
                try:
                    self._check_model_update()
                    self._check_model_freshness()
                except Exception as exc:
                    # Never let a transient failure kill the watch thread —
                    # that would recreate the exact single-point-of-failure
                    # this thread exists to remove.
                    self.log.warning(
                        "model watch iteration failed (retrying in %.0fs): %s",
                        poll_s, exc, exc_info=True,
                    )

        threading.Thread(
            target=_loop, name="model-watch", daemon=True,
        ).start()

    def _check_model_freshness(self) -> None:
        """Alarm when our shard tag has lagged the published model too long.

        The swap paths are best-effort and swallow their errors, so "we are
        still on the old sha" is the only symptom that survives every one of
        them. Workers tag shards with ``self.model_sha``; if that trails the
        published manifest the trainer counts everything we upload as stale,
        which is invisible from here without this check.

        LOCAL WORKERS ONLY, deliberately: reading the published manifest off
        disk is a source independent of every swap path, which is the whole
        point — an HTTP re-poll would go through the same code whose failure we
        are trying to detect. Remote workers have no local manifest and get
        nothing here; the equivalent for them is "no successful poll in N
        seconds", a different check worth adding when we actually run remote
        workers (production launches all four locally off the trial dir).
        """
        manifest_path = self._resolve_local_manifest_path()
        if manifest_path is None:
            return
        try:
            published = str(
                json.loads(manifest_path.read_text(encoding="utf-8"))
                .get("model", {}).get("sha256", "")
            )
        except Exception:
            return
        if not published or published == self.model_sha:
            self._model_stale_since_s = None
            self._model_stale_alarmed = False
            return
        now = time.time()
        if self._model_stale_since_s is None:
            self._model_stale_since_s = now
            return
        stale_s = now - float(self._model_stale_since_s)
        if stale_s < _MODEL_STALE_ALARM_S or self._model_stale_alarmed:
            return
        # Latched: one loud line per stale episode, not one per poll.
        self._model_stale_alarmed = True
        self.log.error(
            "model tag STALE for %.0fs: uploading shards as sha=%s while the "
            "server publishes sha=%s — the trainer is counting our games as "
            "stale. Model swap is wedged; restart this worker if it persists.",
            stale_s, str(self.model_sha)[:8], published[:8],
        )

    def _maybe_log_selfplay_phase_stats(self, elapsed_s: float) -> None:
        phases = (
            "check_model", "tb_adjudicate", "classify", "network",
            "sf_submit_curriculum", "sf_submit_label",
            "sf_finish_curriculum", "sf_block_starved",
            "sf_finish_label", "sf_label_poll",
            "finalize", "tree_reset",
        )
        with self._phase_timing_lock:
            current = dict(self._phase_timing_s)
            previous = dict(self._last_phase_timing_snapshot)
            self._last_phase_timing_snapshot = current
        deltas = {
            phase: max(0.0, float(current.get(phase, 0.0)) - float(previous.get(phase, 0.0)))
            for phase in phases
        }
        total = sum(deltas.values())
        if total <= 0.0:
            return
        self._note_selfplay_progress()

        def pct(phase: str) -> float:
            return 100.0 * float(deltas[phase]) / max(1e-6, float(elapsed_s))

        observations = max(
            1.0,
            float(current.get("slot_observations", 0.0))
            - float(previous.get("slot_observations", 0.0)),
        )

        def observed_mean(metric: str) -> float:
            delta = max(
                0.0,
                float(current.get(metric, 0.0)) - float(previous.get(metric, 0.0)),
            )
            return delta / observations

        pending_overlap_pct = 100.0 * observed_mean("sf_pending_with_runnable_steps")

        self.log.info(
            "selfplay phase stats: check=%.1f%% tb=%.1f%% classify=%.1f%% "
            "network=%.1f%% sf_submit_cur=%.1f%% sf_submit_label=%.1f%% "
            "sf_finish_cur=%.1f%% sf_block_starved=%.1f%% "
            "sf_finish_label=%.1f%% sf_label_poll=%.1f%% "
            "finalize=%.1f%% tree_reset=%.1f%% total_thread=%.1f%% "
            "runnable_avg=%.1f/%.1f/%.1f pending_excluded_avg=%.1f "
            "pending_with_runnable=%.1f%% in_flight_avg=%.1f",
            pct("check_model"),
            pct("tb_adjudicate"),
            pct("classify"),
            pct("network"),
            pct("sf_submit_curriculum"),
            pct("sf_submit_label"),
            pct("sf_finish_curriculum"),
            pct("sf_block_starved"),
            pct("sf_finish_label"),
            pct("sf_label_poll"),
            pct("finalize"),
            pct("tree_reset"),
            100.0 * total / max(1e-6, float(elapsed_s)),
            observed_mean("runnable_net_sum"),
            observed_mean("runnable_sp_opp_sum"),
            observed_mean("runnable_cur_opp_sum"),
            observed_mean("sf_pending_excluded_sum"),
            pending_overlap_pct,
            observed_mean("games_in_flight_sum"),
        )
        gil = self._gil_probe.read_and_reset()
        self.log.info(
            "gil delay stats: samples=%d rate=%.1f/s mean=%.3fms "
            "p50<=%.2fms p95<=%.2fms p99<=%.2fms max=%.2fms "
            "over1ms=%.1f%% over5ms=%.1f%% "
            "(upper bound: GIL reacquire + OS scheduling)",
            gil.samples, gil.sample_rate, gil.mean_ms,
            gil.p50_ms, gil.p95_ms, gil.p99_ms, gil.max_ms,
            gil.over_1ms_pct, gil.over_5ms_pct,
        )

    def _on_completed_game(self, game_batch) -> None:
        callback_t0 = time.perf_counter()
        upload_s = 0.0
        now_s = time.time()
        self._saw_completed_game = True
        self._note_selfplay_progress()
        queued_for_flush = False
        buf_lock = getattr(self, "_upload_buf_lock", None)
  # Difficulty travels with the shard so the promotion gate's anchored
  # current-vs-previous delta can be checked against the difficulty each arm
  # played at. Read once per callback, from the reco the worker APPLIED.
        cur_regret, cur_sf_nodes = self._active_difficulty()
        with buf_lock if buf_lock is not None else nullcontext():
            try:
                _buffer_add_completed_game(
                    buf=self.upload_buf,
                    game_batch=game_batch,
                    now_s=now_s,
                    model_sha=self.model_sha,
                    model_step=self.model_step,
                    opponent_wdl_regret_limit=cur_regret,
                    sf_nodes=cur_sf_nodes,
                    max_positions=int(self.args.upload_max_buffered_positions),
                    buffered_positions_offset=int(getattr(self, "_pending_buffer_positions", 0)),
                )
            except ValueError as exc:
                if str(exc) != "buffered upload model metadata mismatch":
                    raise
                old_sha = str(self.upload_buf.model_sha or "")
                old_step = int(self.upload_buf.model_step or 0)
                self.log.warning(
                    "upload buffer model metadata changed old_sha=%s old_step=%d "
                    "new_sha=%s new_step=%d old_regret=%s new_regret=%s "
                    "old_sf_nodes=%s new_sf_nodes=%s; flushing buffered shard "
                    "before retry",
                    old_sha[:8],
                    old_step,
                    str(self.model_sha)[:8],
                    int(self.model_step),
                    self.upload_buf.opponent_wdl_regret_limit,
                    cur_regret,
                    self.upload_buf.sf_nodes,
                    cur_sf_nodes,
                )
                if self._queue_upload_buffer_locked(now_s=now_s):
                    queued_for_flush = True
                    self._last_upload_flush_s = float(now_s)
                _buffer_add_completed_game(
                    buf=self.upload_buf,
                    game_batch=game_batch,
                    now_s=now_s,
                    model_sha=self.model_sha,
                    model_step=self.model_step,
                    opponent_wdl_regret_limit=cur_regret,
                    sf_nodes=cur_sf_nodes,
                    max_positions=int(self.args.upload_max_buffered_positions),
                    buffered_positions_offset=int(getattr(self, "_pending_buffer_positions", 0)),
                )
            if _buffer_should_flush(
                buf=self.upload_buf,
                now_s=now_s,
                last_send_s=max(
                    float(self.last_successful_send_s),
                    float(getattr(self, "_last_upload_flush_s", 0.0)),
                ),
                target_positions=int(self.args.upload_target_positions),
                flush_seconds=float(self.args.upload_flush_seconds),
            ) and self._queue_upload_buffer_locked(now_s=now_s):
                queued_for_flush = True
                self._last_upload_flush_s = float(now_s)

        # One callback owns detached-buffer materialization + upload. Other
        # callbacks continue immediately on a fresh active buffer.
        if queued_for_flush:
            upload_t0 = time.perf_counter()
            uploaded_at = self._try_upload_pending_shards()
            upload_s += time.perf_counter() - upload_t0
            if uploaded_at is not None:
                self.last_successful_send_s = float(uploaded_at)
        callback_s = time.perf_counter() - callback_t0
        tid = threading.get_ident()
        with self._completion_telemetry_lock:
            self._completion_games += int(getattr(game_batch, "games", 1) or 1)
            self._completion_positions += int(
                getattr(game_batch, "positions", 0) or len(getattr(game_batch, "samples", ()))
            )
            self._completion_callback_s += float(callback_s)
            self._completion_upload_s += float(upload_s)
            self._completion_by_thread[tid] = int(self._completion_by_thread.get(tid, 0)) + 1

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
        current_trial_id = self.leased_trial_id or self.fixed_trial_id or ""
        upload_url = self._server_url_for(self.trial_api_prefix + "/upload_arena_result")
        for jp in sorted(self.arena_pending_dir.glob("*.json")):
            try:
                payload = json.loads(jp.read_text(encoding="utf-8"))
            except Exception as exc:
  # Bad local file. Still moved out of `pending` in one rename, which is
  # what keeps the retry storm away -- only the DESTINATION changes.
  # `uploaded` means "reached the server"; this one did not reach anyone,
  # and filing it there made the directory's name a lie for that entry
  # and inflated any upload count taken by listing it.
                self._arena_rejected_count += 1
                self.log.warning(
                    "arena result %s is unreadable (%s); moving to %s "
                    "(total rejected this session: %d). It was NOT uploaded "
                    "and will not be retried",
                    jp.name, exc, self.arena_rejected_dir, self._arena_rejected_count,
                )
                jp.replace(self.arena_rejected_dir / jp.name)
                continue
            payload_trial_id = str(payload.get("trial_id") or "").strip()
            if payload_trial_id and payload_trial_id != current_trial_id:
                continue
            r = self._requests.post(
                upload_url, json=payload, auth=self._auth,
                headers=_worker_headers(), timeout=60.0,
            )
            if _upload_response_allows_pending_delete(r):
                jp.unlink(missing_ok=True)
                continue
            if r.status_code == 200:
                try:
                    body = r.json()
                except Exception:
                    body = {}
  # ⚑ TERMINAL vs TRANSIENT, and this loop had only one of them. Gated on
  # `terminal`, NOT on `rejected`: `_compat_rejection` also answers
  # `rejected: True` and that case is transient -- the worker self-updates and
  # the same file uploads fine -- so quarantining on `rejected` would discard
  # arena results merely because a worker was out of date. An existing test
  # (`test_arena_upload_keeps_rejected_200_response`, reason "protocol
  # mismatch") encodes exactly that, and it caught this. Files are
  # drained in sorted (timestamp) order and a non-accepted result was ALWAYS
  # kept and then `break`ed on -- so a result the server will never accept sits
  # at the head of the queue and blocks every later one, permanently. The shard
  # loop has had a terminal channel for this since it was written; this one did
  # not, which only stayed invisible because no permanent rejection existed on
  # the arena route until the body-size cap added one.
                if isinstance(body, dict) and bool(body.get("terminal", False)):
                    self._arena_rejected_count += 1
                    self.log.warning(
                        "arena result %s was REJECTED by the server (%s); moving to %s "
                        "(total rejected this session: %d). It will not be retried",
                        jp.name, body.get("reason"), self.arena_rejected_dir,
                        self._arena_rejected_count,
                    )
                    jp.replace(self.arena_rejected_dir / jp.name)
                    continue
                self.log.warning(
                    "server did not accept pending arena result %s; keeping for retry: %s",
                    jp, body,
                )
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
                raise SystemExit(f"Bad protocol_version in manifest: {req_proto!r}") from None
        min_v = manifest.get("min_worker_version")
        version_too_old = bool(min_v is not None and version_lt(PACKAGE_VERSION, str(min_v)))
        enc = manifest.get("encoding") or {}
        if "policy_size" in enc:
            server_policy_size = int(enc.get("policy_size") or 0)
            if server_policy_size not in (int(POLICY_SIZE), int(COMPACT_POLICY_SIZE)):
                raise SystemExit(
                    f"policy_size mismatch: worker supports {POLICY_SIZE}/{COMPACT_POLICY_SIZE} "
                    f"server={enc.get('policy_size')}",
                )
            if server_policy_size == int(COMPACT_POLICY_SIZE):
                policy_encoding = normalize_policy_encoding(enc.get("policy_encoding", "lc0_1858"))
                if policy_encoding != POLICY_ENCODING_LC0_1858:
                    raise SystemExit(
                        f"policy_encoding mismatch: compact size requires lc0_1858, "
                        f"server={enc.get('policy_encoding')}",
                    )
        if "input_planes" in enc:
            expected = input_plane_count(enc.get("input_extra_features"))
            if int(enc.get("input_planes") or 0) != expected:
                raise SystemExit(
                    f"input_planes mismatch: worker expects {expected} "
                    f"(input_extra_features={enc.get('input_extra_features', 'v1')!r}), "
                    f"server={enc.get('input_planes')}"
                )
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
                self._hold_selfplay = False
            return False
        if not self.pause_selfplay_active:
            self.log.info("selfplay paused by server%s", f": {pause_reason}" if pause_reason else "")
            self.pause_selfplay_active = True
  # While hold-on-pause is active the play_batch hold loop paces retries and
  # the tier-2 mtime check must keep running to catch the unpause manifest
  # promptly — a full poll_seconds sleep here would starve it and idle the
  # GPU for up to ~30s after every training phase. The old teardown path
  # (CAE_WORKER_STOP_ON_PAUSE=1) keeps the long sleep: its outer loop has no
  # other pacing.
        sleep_s = 1.0 if self._hold_on_pause else max(0.1, float(self.args.poll_seconds))
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
  # A worker that cannot poll is a worker doing nothing, and this branch
  # used to sleep in complete silence. Paired with the server's 503 for an
  # undecidable compat gate, that made "the fleet is stalled" invisible
  # from BOTH ends. Same cadence idiom as the server side: first, then
  # every 100th, so a routine blip is one line and a stuck fleet keeps
  # saying so.
            self._manifest_poll_failures += 1
            if self._manifest_poll_failures == 1 or self._manifest_poll_failures % 100 == 0:
                self.log.warning(
                    "manifest poll returned HTTP %d (%s); sleeping %.1fs and "
                    "retrying — no selfplay runs until it succeeds "
                    "(consecutive failures: %d)",
                    r.status_code, _bounded_poll_detail(r),
                    float(self.args.poll_seconds), self._manifest_poll_failures,
                )
            time.sleep(float(self.args.poll_seconds))
            return None
        if self._manifest_poll_failures:
            self.log.info(
                "manifest poll recovered after %d consecutive failure(s)",
                self._manifest_poll_failures,
            )
            self._manifest_poll_failures = 0

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
        buf_lock = self._upload_buf_lock
        with buf_lock if buf_lock is not None else nullcontext():
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
                trial_id=self.leased_trial_id or self.fixed_trial_id or None,
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

    def _require_slot_planes_match_manifest(self, manifest: dict) -> None:
        """Broker slot width vs the encoding the manifest declares for the model.

        This runs on the path a PRODUCTION SELFPLAY worker actually takes.
        ``_load_and_compile_model`` does not: with a broker client, ``_sync_model``
        returns before loading (``need_local_model`` is False) and
        ``_swap_model_from_manifest`` returns after only re-tagging, so a check
        placed there fires on arena tasks and nowhere else (PR #321 review F2).

        The hazard is ``INFERENCE_AUDIT`` I2: ``--inference-slot-input-planes``
        defaults to the v1 146 while production is 175, a narrower client fits
        inside the broker's larger segment, the protocol completes, and the
        client reads policy/WDL out of the middle of the broker's INPUT region —
        all-zero, silently. The manifest is the first thing a client-only worker
        sees that knows the model's real width, so compare here.

        THREE numbers must agree, not two (WORKER_AUDIT K5). The CLI arg
        ``--inference-slot-input-planes`` is a third source of truth: the
        manifest's width comes from ``model_cfg.input_extra_features``
        (``distributed_runtime.py:384``) while the launcher's CLI value comes
        from ``config["input_extra_features"]`` (``:890``) — two keys nothing
        compared. In the launched fleet they happen to be equal, so this leg is
        a cross-check that fires the moment that stops being true; the client
        leg is the one that fires for a hand-launched worker.
        ``_check_manifest_compat`` validates only the manifest against ITSELF
        and never sees either.

        Client leg first: when both disagree it is the more informative error
        (it names the width the shared-memory segment was actually built with).
        """
        self._require_client_slot_planes_match_manifest(manifest)
        self._require_cli_slot_planes_match_manifest(manifest)

    def _require_client_slot_planes_match_manifest(self, manifest: dict) -> None:
        """The realized client's slot width vs the manifest. See the caller."""
        client = self.inference_client
        if client is None:
            return
        declared = self._manifest_declared_extra_features(manifest)
        if declared is None:
            # Production manifests always carry it (model_config_to_manifest_dict).
            # A manifest that does not is the one case this check cannot make, so
            # say so once rather than skipping in silence.
            if not self._slot_planes_unknown_warned:
                self._slot_planes_unknown_warned = True
                self.log.warning(
                    "manifest model_config declares no input_extra_features; "
                    "cannot verify inference slot width (%d planes) against the "
                    "model. A skew here returns all-zero policy/WDL silently.",
                    int(client.input_planes),
                )
            return
        want = int(input_plane_count(str(declared)))
        got = int(client.input_planes)
        if got != want:
            raise ValueError(
                f"inference slot is {got} input planes but the manifest declares "
                f"input_extra_features={declared!r} = {want} planes. The broker "
                "and this worker must be started with the same width "
                "(--input-planes / --inference-slot-input-planes; the default is "
                "the v1 146). A narrower client fits inside the broker's segment "
                "and reads policy/WDL out of its input region — all-zero, with no "
                "error (INFERENCE_AUDIT I2)."
            )

    @staticmethod
    def _manifest_declared_extra_features(manifest: dict) -> str | None:
        """The manifest's declared encoding version, or None if it carries none.

        Both shapes reach here: the poll path gets the manifest dict, and
        ``_swap_model_from_manifest`` accepts a ``ModelConfig`` directly.
        Reading only the dict form would silently downgrade the object form to
        the "cannot check" branch.
        """
        mc = manifest.get("model_config") or {}
        if isinstance(mc, ModelConfig):
            return str(mc.input_extra_features)
        if isinstance(mc, dict):
            value = mc.get("input_extra_features")
            return None if value is None else str(value)
        return None

    def _require_cli_slot_planes_match_manifest(self, manifest: dict) -> None:
        """``--inference-slot-input-planes`` vs the manifest (WORKER_AUDIT K5).

        The third leg. Only meaningful when a slot name was configured — with
        no broker the flag sizes nothing, and a local-inference worker that
        left it at the v1 default must not be killed for it.
        """
        if not str(getattr(self.args, "inference_slot_name", "") or "").strip():
            return
        declared = self._manifest_declared_extra_features(manifest)
        if declared is None:
            return  # already warned once by the client leg
        want = int(input_plane_count(declared))
        got = int(self.args.inference_slot_input_planes)
        if got != want:
            raise ValueError(
                f"--inference-slot-input-planes is {got} but the manifest declares "
                f"input_extra_features={declared!r} = {want} planes. The launcher's "
                "CLI value and the manifest's model_config are two different "
                "sources of truth (config['input_extra_features'] vs "
                "model_cfg.input_extra_features) and nothing compared them "
                "(WORKER_AUDIT K5); a skew reads policy/WDL out of the broker's "
                "input region — all-zero, with no error (INFERENCE_AUDIT I2)."
            )

    def _sync_model(self, manifest: dict) -> None:
        """Download + build + load + compile model if SHA changed."""
        task_type = str((manifest.get("task") or {"type": "selfplay"}).get("type", "selfplay")).lower()
        need_local_model = self.inference_client is None or task_type == "arena"

        model_info = manifest.get("model") or {}
        model_sha = str(model_info.get("sha256") or "")
        if not model_sha:
            return
        # Before the need_local_model early return below: a client-only selfplay
        # worker never reaches the model load, and this is the check it needs.
        self._require_slot_planes_match_manifest(manifest)
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
  # Keep the running evaluator pointed at the new model. Without this a mid-
  # session model swap on the poll path (remote workers, or a model change that
  # rode the live-reco path with no restart) would leave play_batch using the
  # old model while shards are tagged with the new SHA. No-op before the
  # evaluator exists (session start syncs it itself).
        self._resync_evaluator_to_model()

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
        self.opening_fen_list_path, self.last_fenlist_sha = _download_opening_book(
            manifest, "opening_fen_list", self.cache_dir,
            # Trial-scoped (unlike the two opening_book endpoints above) —
            # the manifest always carries its own "endpoint" in practice, this
            # is only the defensive fallback, matching the model download's
            # same trial_api_prefix-based pattern (see model_info handling).
            cache_prefix="fenlist", default_endpoint=self.trial_api_prefix + "/opening_fen_list",
            server_url_fn=self._server_url_for, headers=hdrs, log=self.log,
            last_sha=self.last_fenlist_sha,
        )

    def _sync_stockfish(
        self,
        manifest: dict,
        sf_nodes: int,
        sf_multipv: int,
        sf_hash_mb: int,
        syzygy_path: str | None = None,
    ) -> str:
        """Download SF and reinitialize when its process-level options change.

        Returns the resolved stockfish_path.
        """
        stockfish_path = str(self.args.stockfish_path) if self.args.stockfish_path is not None else ""
        if self.args.stockfish_from_server:
            sf_rec = manifest.get("stockfish")
            if not isinstance(sf_rec, dict) or not sf_rec.get("endpoint") or not sf_rec.get("sha256"):
                raise SystemExit("--stockfish-from-server enabled but server did not publish stockfish")
            sf_sha = str(sf_rec.get("sha256"))
            sf_endpoint = str(sf_rec.get("endpoint"))
            sf_filename = _safe_manifest_filename(sf_rec.get("filename"), default="stockfish")
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

  # (Re)initialize engine if the binary path/SHA, multipv, hash, or syzygy path
  # changed (all must be set at init time). The binary check is what makes the
  # asset-fingerprint restart actually swap a hot-swapped Stockfish: without it
  # a new SHA downloads to a new path but the running engine keeps the old one.
        sz_path = syzygy_path or None
        multipv_changed = self.sf_multipv_active is None or int(self.sf_multipv_active) != int(sf_multipv)
        hash_changed = self.sf_hash_mb_active is None or int(self.sf_hash_mb_active) != int(sf_hash_mb)
        syzygy_changed = self.sf_syzygy_path_active != sz_path
        binary_changed = self.sf_path_active != stockfish_path
        if self.sf is None or multipv_changed or hash_changed or syzygy_changed or binary_changed:
            if self.sf is not None:
                with suppress(Exception):
                    self.sf.close()

            if int(self.args.sf_workers) > 1:
                self.sf = StockfishPool(
                    path=str(stockfish_path),
                    nodes=int(sf_nodes),
                    num_workers=int(self.args.sf_workers),
                    multipv=int(sf_multipv),
                    hash_mb=int(sf_hash_mb),
                    syzygy_path=sz_path,
                    nice=int(self.args.sf_nice),
                )
            else:
                self.sf = StockfishUCI(
                    str(stockfish_path),
                    nodes=int(sf_nodes),
                    multipv=int(sf_multipv),
                    hash_mb=int(sf_hash_mb),
                    syzygy_path=sz_path,
                    nice=int(self.args.sf_nice),
                )
            self.sf_multipv_active = int(sf_multipv)
            self.sf_hash_mb_active = int(sf_hash_mb)
            self.sf_syzygy_path_active = sz_path
            self.sf_path_active = stockfish_path
        else:
  # update nodes dynamically (same live-update used by _apply_live_reco)
            self._set_sf_nodes(sf_nodes)

        return stockfish_path

    def _set_sf_nodes(self, sf_nodes: int) -> None:
        """Push a new SF node budget to the live engine, if it supports it.
        Shared by _sync_stockfish (restart path) and _apply_live_reco (live)."""
        if self.sf is not None and hasattr(self.sf, "set_nodes"):
            self.sf.set_nodes(int(sf_nodes))

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
            "trial_id": self.leased_trial_id or self.fixed_trial_id or "",
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

  # Live-tunable fields: the trainer/PID changes these between iterations during
  # a normal run (difficulty levers + the selfplay/curriculum mix). Every
  # consumer reads them fresh per step/move/recycle, so they are applied to the
  # running session in place (see _apply_live_reco) instead of restarting it.
  # Restarting bounces SF, abandons the 256 in-flight games, and collapses
  # curriculum throughput for ~2 iters (small-sample winrate spike).
    _RECO_LIVE_KEYS = (
        "selfplay_fraction", "opponent_wdl_regret_limit",
        "sf_nodes", "sf_fast_ply_node_scale", "sf_label_nodes_cap",
        "sf_label_nodes_floor",
  # Label-escalation knobs are read fresh at label-attach time
  # (stockfish_turn._maybe_submit_label_escalation), so they apply live like
  # sf_label_nodes_cap — the experiment can be toggled without a restart.
        "sf_label_escalate_q_gap", "sf_label_escalate_nodes",
        "sf_label_escalate_max_per_game",
    )

  # Fields in recommended_worker that affect gameplay and should trigger
  # a session restart when the trainer updates them between iterations.
    _RECO_RESTART_KEYS = (
  # sf_multipv is applied only at engine (re)init in _sync_stockfish (it can't
  # be set live like sf_nodes), so a change must restart — otherwise the worker
  # keeps producing labels at the old PV count until an unrelated restart.
        "sf_multipv",
        "sf_hash_mb",
  # games_per_batch is consumed at session start in _run_selfplay (play_batch
  # slot count) and cannot be resized on a running SelfplayState, so a change
  # must restart.
        "games_per_batch",
        "slot_oversubscribe",
  # Both Gumbel batching knobs are read into SearchConfig by
  # _build_selfplay_configs, which runs once at session start — a running
  # SelfplayState keeps its old frozen SearchConfig, so a mid-flight change
  # would be silently ignored and the ledger would carry a verdict for an
  # experiment that never ran.
        "gumbel_target_batch",
        "gumbel_vloss_weight",
  # Same reason: _build_selfplay_configs freezes it into SearchConfig at
  # session start, so a mid-flight change would be accepted and never reach
  # a search -- the exact shape of a verdict for an experiment that never ran.
        "gumbel_target_max_visit_cap",
        "gumbel_target_untempered_prior",
  # Same reason again, plus a second one: the diff-focus norm group is baked
  # into a frozen DiffFocusConfig at session start, and `norm_enabled`
  # additionally decides whether SelfplayState.create builds a normalizer at
  # all. A restart is also the CORRECT semantics rather than merely the
  # achievable one -- changing the window, quantile or slope invalidates the
  # quantile window accumulated under the old settings, and a restart
  # discards it.
        "diff_focus_norm_enabled",
        "diff_focus_norm_window",
        "diff_focus_norm_warmup",
        "diff_focus_norm_quantile",
        "diff_focus_norm_slope",
        "diff_focus_norm_clip",
  # And this one decides how MANY estimators exist, which is fixed the moment
  # the session's threads are launched.
        "diff_focus_norm_shared",
  # sf_move_nodes gates the curriculum SF query path: lowering it to 0 mid-flight
  # would make pending move-futures (submitted at the old positive budget) get
  # reused as full-strength label futures, writing low-node SF targets to replay.
  # It's a static knob, so make it restart-only rather than drain futures live.
        "sf_move_nodes",
        "mcts_simulations", "fast_simulations",
  # Same reason as the two batching knobs above: _build_selfplay_configs reads
  # it into the frozen SearchConfig once per session, so a mid-flight change
  # would be accepted by the manifest and never reach a search.
        "gumbel_policy_temp",
        "gumbel_topk", "gumbel_c_scale", "gumbel_scale", "gumbel_scale_after",
        "gumbel_scale_decay_start_move", "gumbel_scale_decay_moves",
        "curriculum_gumbel_scale", "curriculum_gumbel_scale_after",
        "curriculum_gumbel_scale_decay_start_move", "curriculum_gumbel_scale_decay_moves",
        "temperature", "temperature_decay_start_move", "temperature_decay_moves",
        "temperature_endgame", "selfplay_temperature",
        "selfplay_temperature_decay_start_move", "selfplay_temperature_decay_moves",
        "selfplay_temperature_endgame",
        "sf_wdl_use_cp_logistic", "sf_wdl_cp_slope", "sf_wdl_cp_draw_width",
        "input_history_encoding", "record_lc0_root_input", "history_rep_fix",
        "record_dense_sf_policy", "record_sf_p0_policy", "record_sf_p0_regret",
        "record_fast_ply_value", "blindspot_harvest_out_path",
        "categorical_blend_frac",
        "categorical_search_blend_frac",
        # Syzygy knobs affect adjudication + in-search overrides — without a
        # restart, workers keep producing shards under stale TB settings until
        # an unrelated key changes. Flagged by Codex adversarial review.
        "syzygy_path", "stockfish_syzygy_path", "syzygy_rescore_policy",
        "syzygy_adjudicate", "syzygy_adjudicate_fraction", "syzygy_in_search",
        "policy_encoding", "input_extra_features",
        "use_dynamic_relations", "record_relations",
  # Remaining session-fixed fields built from reco. They used to propagate only
  # incidentally — when the PID happened to move a (then-restart-keyed) sf_nodes
  # /regret lever that iteration, the rebuild flushed them. Now those levers are
  # live (no restart), so without listing these a mid-run change to a label knob
  # (sf_policy_temp / sf_policy_label_smooth), gameplay structure (max_plies,
  # mcts, playout_cap_fraction, random_start_plies, timeout_adjudication_threshold),
  # opening sampling, or the volatility-search knobs would silently never reach
  # workers. test_every_reco_field_is_watched guards completeness.
        "sf_policy_temp", "sf_policy_label_smooth", "soft_policy_temp",
        "sf_policy_score_mode", "sf_policy_cp_temp",
        "timeout_adjudication_threshold",
        "max_plies", "mcts", "playout_cap_fraction", "full_ply_pair_fraction",
        "random_start_plies",
        "opening_book_prob", "opening_book_max_plies", "opening_book_max_games",
        "opening_book_max_plies_2", "opening_book_max_games_2", "opening_book_mix_prob_2",
        "opening_fen_prob", "opening_fen_net_side_to_move", "opening_fen_selfplay_only",
        "opening_fen_dole_per_iter",
        "opening_fen_sf_refute_frac", "opening_fen_sf_refute_plies",
        # SF-refute opp-row recording (OpeningConfig, session-fixed → restart).
        "sf_refute_full_node_moves", "sf_refute_record_opp_rows",
        "sf_refute_opp_policy_net_blend",
        "volatility_q_scale", "volatility_fpu", "volatility_anchor",
  # Read once at session start (_run_selfplay) to decide whether this session
  # persists its in-flight games at teardown and resumes persisted ones at
  # start. Session-fixed, hence restart-keyed; toggling it costs exactly one
  # more abandonment, after which restarts stop abandoning.
        "selfplay_resume_inflight_games",
    )

  # Every reco key worth tracking for change detection (restart + live).
    _RECO_WATCH_KEYS = _RECO_LIVE_KEYS + _RECO_RESTART_KEYS

  # ── In-flight resume compatibility ───────────────────────────────────────
  # A resumed game carries plies recorded by the OLD session into a shard
  # written by the NEW one. That is only sound while the knobs that decide what
  # a stored ply MEANS are unchanged; a mismatch discards the game (counted,
  # never silently mixed). Only two classes qualify:
  #   * the input encoding the net read when it produced those targets, and
  #   * label knobs stamped onto the record AT LABEL TIME (stockfish_turn),
  #     which a later finalize cannot re-derive.
  # Knobs applied uniformly at FINALIZE time (soft_policy_temp, categorical_*,
  # syzygy_rescore_policy, record_dense_sf_policy, record_sf_p0_*, policy
  # encoding) are exempt: finalize applies the new value to every row of the
  # game, resumed or not. So are the recomputed inputs (record_relations,
  # record_lc0_root_input) — resume re-encodes every restored ply under the new
  # config, so the game stays internally consistent either way.
    _RESUME_COMPAT_KEYS = (
        "input_history_encoding", "input_extra_features", "history_rep_fix",
        "sf_multipv", "sf_policy_temp", "sf_policy_label_smooth",
        "sf_policy_score_mode", "sf_policy_cp_temp",
        "sf_wdl_use_cp_logistic", "sf_wdl_cp_slope", "sf_wdl_cp_draw_width",
        "sf_refute_record_opp_rows", "sf_refute_opp_policy_net_blend",
  # RECORD-SHAPING, not plumbing: `priority` and `keep_prob` are stored columns,
  # and the norm group decides the UNITS they are stored in (raw difficulty vs
  # difficulty / reference-quantile, capped). A game whose early plies were
  # recorded under one setting and whose later plies were recorded under another
  # would carry two unit systems into one shard, and the replay sampler draws on
  # that column without any way to tell them apart. Discard such games on resume.
        "diff_focus_norm_enabled", "diff_focus_norm_window",
        "diff_focus_norm_warmup", "diff_focus_norm_quantile",
        "diff_focus_norm_slope", "diff_focus_norm_clip",
    )

  # Restart keys deliberately NOT part of the resume fingerprint. Listed rather
  # than defaulted so a new restart key must be classified into one bucket or
  # the other (test_every_restart_key_is_resume_classified) — an unclassified
  # record-shaping key would silently mix two schemas inside one game.
    _RESUME_COMPAT_EXEMPT_KEYS = (
  # Engine/session plumbing — no bearing on a stored ply.
        "sf_hash_mb", "games_per_batch", "slot_oversubscribe",
        "selfplay_resume_inflight_games",
  # Deliberately NOT with the other diff_focus_norm_* keys above, which are
  # resume-INCOMPATIBLE because they change the UNITS of the stored `priority`.
  # This one does not: both settings run the identical unarmed→armed branches
  # and the identical arithmetic on each. It changes only how many estimators
  # exist, hence how fast one arms. A game that straddles the arm point already
  # mixes raw and normalized rows under either setting, so gating resume on this
  # key would guard something the codebase already accepts.
        "diff_focus_norm_shared",
  # NOT plumbing — stockfish_turn branches on it (>0 buys separate node-capped
  # label queries; 0 reuses the full-strength opponent-move future), so a flip
  # across a resume mixes label node budgets within one game. Exempt anyway,
  # accepted rather than overlooked: sf_nodes — the dominant label-budget knob
  # (108k-698k under the PID) — is a LIVE key applied to running games, so
  # mixed-depth labels within one game are already normal operation, and
  # discarding suspended games on this knob while accepting sf_nodes swings
  # would be a gate that guards nothing.
        "sf_move_nodes",
  # Search shape. Changes the target's provenance the same way a mid-game model
  # swap does (which already happens every iteration); each ply keeps a valid
  # target for its own position.
        "gumbel_target_batch", "gumbel_vloss_weight",
        "gumbel_target_max_visit_cap", "gumbel_target_untempered_prior",
        "mcts_simulations", "fast_simulations", "mcts", "playout_cap_fraction",
        "full_ply_pair_fraction",
        "gumbel_topk", "gumbel_policy_temp",
        "gumbel_c_scale", "gumbel_scale", "gumbel_scale_after",
        "gumbel_scale_decay_start_move", "gumbel_scale_decay_moves",
        "curriculum_gumbel_scale", "curriculum_gumbel_scale_after",
        "curriculum_gumbel_scale_decay_start_move", "curriculum_gumbel_scale_decay_moves",
        "volatility_q_scale", "volatility_fpu", "volatility_anchor",
        "temperature", "temperature_decay_start_move", "temperature_decay_moves",
        "temperature_endgame", "selfplay_temperature",
        "selfplay_temperature_decay_start_move", "selfplay_temperature_decay_moves",
        "selfplay_temperature_endgame",
  # Applied at finalize to every row of the game at once.
        "soft_policy_temp", "record_dense_sf_policy", "record_sf_p0_policy",
        "record_sf_p0_regret", "record_fast_ply_value", "policy_encoding",
        "categorical_blend_frac", "categorical_search_blend_frac",
        "syzygy_path", "stockfish_syzygy_path", "syzygy_rescore_policy",
        "syzygy_adjudicate", "syzygy_adjudicate_fraction", "syzygy_in_search",
        "timeout_adjudication_threshold", "max_plies",
        "blindspot_harvest_out_path",
  # Recomputed from the board at resume, so consistent under either value.
        "record_relations", "use_dynamic_relations", "record_lc0_root_input",
  # Opening selection — decided when a game STARTS; a resumed game already has
  # its opening and a fresh one uses the new setting.
        "random_start_plies",
        "opening_book_prob", "opening_book_max_plies", "opening_book_max_games",
        "opening_book_max_plies_2", "opening_book_max_games_2", "opening_book_mix_prob_2",
        "opening_fen_prob", "opening_fen_net_side_to_move", "opening_fen_selfplay_only",
        "opening_fen_dole_per_iter",
        "opening_fen_sf_refute_frac", "opening_fen_sf_refute_plies",
        "sf_refute_full_node_moves",
    )

    def _resume_compat_fingerprint(self, reco: dict) -> str:
        """Stable digest of the reco knobs a resumed game must be matched on."""
        payload = json.dumps(
            {k: reco.get(k) for k in sorted(self._RESUME_COMPAT_KEYS)},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _begin_resume_session(self, reco: dict) -> None:
        """Fix this session's in-flight-resume state, and sweep the state dir.

        Called once by ``_run_selfplay`` before the session starts. All three
        values are session-FIXED by design: a session either does in-flight
        resume for its whole life or not at all, so the suspend hook and the
        restore hook can never disagree about the flag, the compat fingerprint,
        or the trial.
        """
  # Session-fixed: toggling mid-session would let the suspend and restore hooks
  # disagree about whether this session's games are resumable.
        self._resume_inflight_enabled = bool(
            reco.get("selfplay_resume_inflight_games", False),
        )
        self._resume_compat_fingerprint_active = self._resume_compat_fingerprint(reco)
  # A SNAPSHOT, not a live read — see _resume_trial_id for why reading it at
  # suspend time stamps the wrong trial on the games it is guarding.
        self._resume_trial_id_active = self._current_trial_id()
        if self._resume_inflight_enabled and not self._resume_trial_id_active:
  # A leased worker between model swaps has no trial id yet
  # (_check_model_update clears leased_trial_id on a sha change). Nothing can
  # be persisted or resumed this session — existing files are preserved on
  # disk, not destroyed — but without this line `resumed_inflight_games: 0`
  # is indistinguishable from the feature simply not working.
            self.log.warning(
                "selfplay resume: enabled but no trial id at session start; "
                "suspended games stay on disk unresumed and this session's "
                "games cannot be persisted until a lease assigns a trial",
            )
  # Unconditional, flag or no flag. Turning the flag OFF is the documented
  # revert, and it disables the drain (resume_inflight_games runs only under the
  # flag) along with the suspend — so sweeping only under the flag would leave
  # the last teardown's 400-1024 files in selfplay_resume/ forever, in exactly
  # the case the backstop is named for. Never fatal: this is housekeeping.
        try:
            sweep_orphan_state_files(self.resume_dir)
        except Exception:
            self.log.exception("selfplay resume: orphan sweep failed (ignored)")

    def _resume_trial_id(self) -> str:
        """Trial a suspended game belongs to, AS OF SESSION START.

        A trial reassignment also tears the session down, and a game played
        under another trial's hyperparameters must not be finished and uploaded
        under this one. That guard only works on a snapshot: the reassignment
        that triggers the teardown mutates ``leased_trial_id`` first
        (``_negotiate_lease``) and sets ``_stop_selfplay`` second, so reading it
        live inside the suspend hook returns the NEW trial's id and stamps it on
        the OLD trial's games. ``_run_selfplay`` captures
        ``_resume_trial_id_active`` before the session starts; both the suspend
        and the resume side read that.
        """
        return str(self._resume_trial_id_active or "")

    def _current_trial_id(self) -> str:
        """The trial this worker is leased to RIGHT NOW. Only
        ``_begin_resume_session`` may read this for resume purposes; every other
        resume caller must go through the ``_resume_trial_id`` snapshot."""
        return str(self.leased_trial_id or self.fixed_trial_id or "")

    def _suspend_inflight_games(self, state: Any) -> None:
        """play_batch on_suspend hook: persist this session's in-flight games."""
        # _PendingSfLabel.slot; a label still in flight here dies with the
        # session, so its ply resumes UNLABELLED. Counted, not hidden.
        pending = [int(getattr(p, "slot", -1)) for p in state.pending_sf_labels]
        try:
            report = suspend_inflight_games(
                state,
                out_dir=self.resume_dir,
                compat_fingerprint=str(self._resume_compat_fingerprint_active),
                model_sha=str(self.model_sha),
                model_step=int(self.model_step),
                trial_id=self._resume_trial_id(),
                pending_label_slots=[i for i in pending if i >= 0],
            )
        except Exception:
            # A failed suspend must never take the worker down: the old
            # behaviour (abandon everything) is exactly the fallback.
            self.log.exception("selfplay resume: suspend failed; games abandoned")
            return
        with self._resume_counts_lock:
            self._resume_counts["suspended"] += int(report.persisted)
            # report.skipped no longer counts freshly-recycled empty slots (see
            # SuspendReport), so a nonzero value here is always a real loss.
            self._resume_counts["suspend_skipped"] += int(report.skipped)
            for reason, n in report.reasons.items():
                if reason == "empty_slot":
                    continue
                self._resume_skip_reasons[reason] = (
                    self._resume_skip_reasons.get(reason, 0) + int(n)
                )

    def _resume_inflight_games(self, state: Any) -> None:
        """on_state_ready hook: refill fresh slots with persisted games."""
        try:
            report = resume_inflight_games(
                state,
                in_dir=self.resume_dir,
                compat_fingerprint=str(self._resume_compat_fingerprint_active),
                model_sha=str(self.model_sha),
                model_step=int(self.model_step),
                trial_id=self._resume_trial_id(),
            )
        except Exception:
            self.log.exception("selfplay resume: restore failed; starting fresh games")
            return
        with self._resume_counts_lock:
            self._resume_counts["resumed"] += int(report.resumed)
            self._resume_counts["discarded"] += int(report.discarded)
        if report.resumed or report.discarded:
            with self._resume_counts_lock:
                skipped = int(self._resume_counts["suspend_skipped"])
                skip_reasons = ",".join(
                    f"{k}={v}" for k, v in sorted(self._resume_skip_reasons.items())
                ) or "-"
            # suspend_skipped is a LOSS count (games we meant to persist and
            # could not); printing it next to the successes is the only place
            # it is ever read.
            self.log.info(
                "selfplay resume totals: suspended=%d resumed=%d discarded=%d "
                "suspend_skipped=%d [%s]",
                self._resume_counts["suspended"],
                self._resume_counts["resumed"],
                self._resume_counts["discarded"],
                skipped, skip_reasons,
            )

    def _build_selfplay_configs(self, reco: dict) -> tuple[dict, tuple]:
        """Unpack manifest.recommended_worker into the 5 frozen config dataclasses.

        Returns ``(cfgs, sf_args)`` where ``cfgs`` is a dict keyed by
        ``play_batch`` kwarg names (``opponent``/``temp``/``search``/``opening``/
        ``game``), ready for ``play_batch(..., **cfgs)``; ``sf_args`` is
        ``(sf_nodes, sf_multipv, sf_hash_mb, syzygy_path)`` for the Stockfish
        bring-up.
        """
        regret_raw = reco.get("opponent_wdl_regret_limit")
  # None must propagate (means "no regret limit"); only cast real values.
        regret_limit = float(regret_raw) if regret_raw is not None else None
  # Syzygy is fully manifest-driven so the server operator can tune
  # adjudication behavior live by editing publish/manifest.json. None
  # for path = disabled; fraction 1.0 = always adjudicate when on.
        syzygy_path = reco.get("syzygy_path") or None
        stockfish_syzygy_path = reco.get("stockfish_syzygy_path") or None
        def _optional_reco(key: str, default: Any, cast: Callable[[Any], Any] = float) -> Any | None:
            if reco.get(key) is None and getattr(self.args, key, None) is None:
                return None
            return self._resolve_reco(reco, key, default, cast)

        cfgs = {
            "slot_oversubscribe": self._resolve_reco(
                reco, "slot_oversubscribe", 1.0,
            ),
            "opponent": OpponentConfig(wdl_regret_limit=regret_limit),
            "temp": TemperatureConfig(
                temperature=self._resolve_reco(reco, "temperature", 1.0),
                decay_start_move=self._resolve_reco(reco, "temperature_decay_start_move", 20, int),
                decay_moves=self._resolve_reco(reco, "temperature_decay_moves", 60, int),
                endgame=self._resolve_reco(reco, "temperature_endgame", 0.6),
                selfplay_temperature=_optional_reco("selfplay_temperature", 1.0),
                selfplay_decay_start_move=_optional_reco(
                    "selfplay_temperature_decay_start_move", 20, int,
                ),
                selfplay_decay_moves=_optional_reco(
                    "selfplay_temperature_decay_moves", 60, int,
                ),
                selfplay_endgame=_optional_reco("selfplay_temperature_endgame", 0.6),
            ),
            "search": SearchConfig(
                simulations=self._resolve_reco(reco, "mcts_simulations", 50, int),
                mcts_type=self._resolve_reco(reco, "mcts", "puct", str),
                playout_cap_fraction=self._resolve_reco(reco, "playout_cap_fraction", 0.25),
                full_ply_pair_fraction=self._resolve_reco(
                    reco, "full_ply_pair_fraction", 0.0,
                ),
                fast_simulations=self._resolve_reco(reco, "fast_simulations", 8, int),
                gumbel_topk=self._resolve_reco(reco, "gumbel_topk", 16, int),
                gumbel_policy_temp=self._resolve_reco(
                    reco, "gumbel_policy_temp", 1.0,
                ),
                gumbel_target_batch=self._resolve_reco(
                    reco, "gumbel_target_batch", 0, int,
                ),
                gumbel_vloss_weight=self._resolve_reco(
                    reco, "gumbel_vloss_weight", 0, int,
                ),
                gumbel_target_max_visit_cap=self._resolve_reco(
                    reco, "gumbel_target_max_visit_cap", 0, int,
                ),
                gumbel_target_untempered_prior=self._resolve_reco(
                    reco, "gumbel_target_untempered_prior", False, bool,
                ),
                volatility_q_scale=self._resolve_reco(reco, "volatility_q_scale", 0.0),
                volatility_fpu=self._resolve_reco(reco, "volatility_fpu", 0.0),
                volatility_anchor=self._resolve_reco(reco, "volatility_anchor", DEFAULT_VOLATILITY_ANCHOR),
                gumbel_c_scale=self._resolve_reco(
                    reco, "gumbel_c_scale", SELFPLAY_GUMBEL_C_SCALE,
                ),
                gumbel_scale=self._resolve_reco(reco, "gumbel_scale", 1.0),
                gumbel_scale_after=self._resolve_reco(reco, "gumbel_scale_after", 0.0),
                gumbel_scale_decay_start_move=self._resolve_reco(
                    reco, "gumbel_scale_decay_start_move", 0, int,
                ),
                gumbel_scale_decay_moves=self._resolve_reco(
                    reco, "gumbel_scale_decay_moves", 0, int,
                ),
                curriculum_gumbel_scale=self._resolve_reco(reco, "curriculum_gumbel_scale", 0.0),
                curriculum_gumbel_scale_after=self._resolve_reco(
                    reco, "curriculum_gumbel_scale_after", 0.0,
                ),
                curriculum_gumbel_scale_decay_start_move=self._resolve_reco(
                    reco, "curriculum_gumbel_scale_decay_start_move", 0, int,
                ),
                curriculum_gumbel_scale_decay_moves=self._resolve_reco(
                    reco, "curriculum_gumbel_scale_decay_moves", 0, int,
                ),
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
                opening_fen_list_path=self.opening_fen_list_path,
                opening_fen_prob=(
                    float(reco.get("opening_fen_prob", 0.0))
                    if self.opening_fen_list_path else 0.0
                ),
                opening_fen_net_side_to_move=bool(
                    reco.get("opening_fen_net_side_to_move", True)
                ),
                opening_fen_selfplay_only=bool(reco.get("opening_fen_selfplay_only", False)),
                opening_fen_dole_per_iter=(
                    int(reco.get("opening_fen_dole_per_iter", 0))
                    if self.opening_fen_list_path else 0
                ),
                opening_fen_sf_refute_frac=(
                    float(reco.get("opening_fen_sf_refute_frac", 0.0) or 0.0)
                    if self.opening_fen_list_path else 0.0
                ),
                opening_fen_sf_refute_plies=int(
                    reco.get("opening_fen_sf_refute_plies", 5) or 0
                ),
                sf_refute_full_node_moves=bool(
                    reco.get("sf_refute_full_node_moves", False)
                ),
                sf_refute_record_opp_rows=bool(
                    reco.get("sf_refute_record_opp_rows", False)
                ),
                sf_refute_opp_policy_net_blend=float(
                    reco.get("sf_refute_opp_policy_net_blend", 0.0) or 0.0
                ),
            ),
  # Only the norm group is read here; the other five DiffFocusConfig fields
  # stay at their dataclass defaults, which is exactly what this worker ran
  # before (play_batch's `diff_focus` default). See build_recommended_worker.
            "diff_focus": DiffFocusConfig(
                norm_enabled=bool(reco.get("diff_focus_norm_enabled", False)),
                norm_window=self._resolve_reco(
                    reco, "diff_focus_norm_window", 8192, int,
                ),
                norm_warmup=self._resolve_reco(
                    reco, "diff_focus_norm_warmup", 1024, int,
                ),
                norm_quantile=self._resolve_reco(
                    reco, "diff_focus_norm_quantile", 0.5,
                ),
                norm_slope=self._resolve_reco(reco, "diff_focus_norm_slope", 1.62),
                norm_clip=self._resolve_reco(reco, "diff_focus_norm_clip", 8.0),
                norm_shared=bool(reco.get("diff_focus_norm_shared", False)),
            ),
            "game": GameConfig(
                max_plies=self._resolve_reco(reco, "max_plies", 240, int),
                selfplay_fraction=float(reco.get("selfplay_fraction", 0.0)),
                sf_move_nodes=self._resolve_reco(reco, "sf_move_nodes", 0, int),
                sf_fast_ply_node_scale=self._resolve_reco(
                    reco, "sf_fast_ply_node_scale", 0.25,
                ),
                sf_label_nodes_cap=self._resolve_reco(reco, "sf_label_nodes_cap", 0, int),
                sf_label_nodes_floor=self._resolve_reco(
                    reco, "sf_label_nodes_floor", 0, int,
                ),
                sf_label_escalate_q_gap=self._resolve_reco(
                    reco, "sf_label_escalate_q_gap", 0.0,
                ),
                sf_label_escalate_nodes=self._resolve_reco(
                    reco, "sf_label_escalate_nodes", 3_000_000, int,
                ),
                sf_label_escalate_max_per_game=self._resolve_reco(
                    reco, "sf_label_escalate_max_per_game", 2, int,
                ),
  # Defaults for the seven SF target-construction keys derive from
  # SfTargetParams — the trainer-side rebuild resolves the SAME keys through
  # that dataclass (trainer.resolve_sf_target_params), so a default drifting
  # here would silently split capture-time and rebuilt targets whenever a
  # manifest omits the key.
                sf_policy_temp=self._resolve_reco(
                    reco, "sf_policy_temp", _SF_TARGET_DEFAULTS.sf_policy_temp,
                ),
                sf_policy_score_mode=str(
                    reco.get(
                        "sf_policy_score_mode",
                        _SF_TARGET_DEFAULTS.sf_policy_score_mode,
                    )
                ),
                sf_policy_cp_temp=self._resolve_reco(
                    reco, "sf_policy_cp_temp", _SF_TARGET_DEFAULTS.sf_policy_cp_temp,
                ),
                sf_policy_label_smooth=self._resolve_reco(
                    reco,
                    "sf_policy_label_smooth",
                    _SF_TARGET_DEFAULTS.sf_policy_label_smooth,
                ),
                sf_wdl_use_cp_logistic=bool(
                    reco.get(
                        "sf_wdl_use_cp_logistic",
                        _SF_TARGET_DEFAULTS.sf_wdl_use_cp_logistic,
                    )
                ),
                sf_wdl_cp_slope=float(
                    reco.get("sf_wdl_cp_slope", _SF_TARGET_DEFAULTS.sf_wdl_cp_slope)
                ),
                sf_wdl_cp_draw_width=float(
                    reco.get(
                        "sf_wdl_cp_draw_width", _SF_TARGET_DEFAULTS.sf_wdl_cp_draw_width,
                    )
                ),
  # No CLI counterpart on purpose (hence reco.get, not _resolve_reco): this
  # is a TRAINING TARGET exponent, and a per-worker override would silently
  # mix two target sharpnesses inside one replay window.
                soft_policy_temp=float(reco.get("soft_policy_temp", 2.0)),
                timeout_adjudication_threshold=float(reco.get("timeout_adjudication_threshold", 0.90)),
                syzygy_path=syzygy_path,
                stockfish_syzygy_path=stockfish_syzygy_path,
                syzygy_rescore_policy=bool(reco.get("syzygy_rescore_policy", False)),
                syzygy_adjudicate=bool(reco.get("syzygy_adjudicate", False)),
                syzygy_adjudicate_fraction=float(reco.get("syzygy_adjudicate_fraction", 1.0)),
                syzygy_in_search=bool(reco.get("syzygy_in_search", False)),
                policy_encoding=_require_compact_policy_encoding(reco.get("policy_encoding")),
                input_history_encoding=str(reco.get("input_history_encoding", "legacy")),
                input_extra_features=str(reco.get("input_extra_features", "v1")),
                record_lc0_root_input=bool(reco.get("record_lc0_root_input", False)),
                history_rep_fix=bool(reco.get("history_rep_fix", False)),
                record_relations=bool(
                    reco.get("record_relations", reco.get("use_dynamic_relations", False))
                ),
                record_dense_sf_policy=bool(reco.get("record_dense_sf_policy", True)),
                record_sf_p0_policy=bool(reco.get("record_sf_p0_policy", False)),
                record_sf_p0_regret=bool(reco.get("record_sf_p0_regret", False)),
                record_fast_ply_value=bool(reco.get("record_fast_ply_value", False)),
                blindspot_harvest_out_path=str(reco.get("blindspot_harvest_out_path", "")),
                categorical_blend_frac=float(reco.get("categorical_blend_frac", 0.0)),
                categorical_search_blend_frac=float(
                    reco.get("categorical_search_blend_frac", 0.0)
                ),
            ),
        }
        # sf_nodes is required, never defaulted: a fail-weak fallback (the old
        # 2000) silently runs Stockfish far below the configured strength and
        # poisons curriculum games. Missing => skip the session and re-poll.
        sf_nodes = self._require_reco(reco, "sf_nodes", int)
        if sf_nodes <= 0:
            raise _MissingRequiredReco(f"sf_nodes must be a positive budget, got {sf_nodes}")
        sf_args = (
            sf_nodes,
            self._resolve_reco(reco, "sf_multipv", 5, int),
            self._resolve_reco(reco, "sf_hash_mb", 16, int),
            stockfish_syzygy_path or syzygy_path,
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
        """Per-field reduce for threaded selfplay summaries."""
        agg: dict = {}
        for fld in dataclasses.fields(all_stats[0]):
            vals = [getattr(st, fld.name) for st in all_stats]
            if all(isinstance(v, int) for v in vals):
                agg[fld.name] = sum(vals)
            elif all(isinstance(v, float) for v in vals):
                if fld.name == "sf_eval_delta6":
                    total_n = sum(int(st.sf_eval_delta6_n) for st in all_stats)
                    agg[fld.name] = (
                        sum(float(st.sf_eval_delta6) * int(st.sf_eval_delta6_n) for st in all_stats) / total_n
                        if total_n > 0 else 0.0
                    )
                elif fld.name == "diff_focus_priority_min":
                    active_vals = [
                        float(st.diff_focus_priority_min)
                        for st in all_stats
                        if int(st.diff_focus_records) > 0
                    ]
                    agg[fld.name] = min(active_vals) if active_vals else 0.0
                elif fld.name == "diff_focus_priority_max":
                    active_vals = [
                        float(st.diff_focus_priority_max)
                        for st in all_stats
                        if int(st.diff_focus_records) > 0
                    ]
                    agg[fld.name] = max(active_vals) if active_vals else 0.0
                elif fld.name.endswith("_sum"):
                    agg[fld.name] = sum(vals)
                else:
                    agg[fld.name] = sum(vals) / len(vals)
            elif all(isinstance(v, dict) for v in vals):
                merged: dict[str, int] = {}
                for stats_dict in vals:
                    for key, val in stats_dict.items():
                        merged[str(key)] = int(merged.get(str(key), 0)) + int(val or 0)
                agg[fld.name] = merged
            else:
                agg[fld.name] = vals[0]
        return BatchStats(**agg)

    def _run_selfplay_threaded(
        self, *, games_per_batch: int, sf, eval_, cfgs: dict,
        fen_dole_queue: list[str] | None = None,
        fen_sf_refute_queue: list[str] | None = None,
        diff_focus_norm: DiffFocusNormalizer | None = None,
    ) -> BatchStats:
        """Multi-threaded selfplay: N threads share one GPU evaluator.

        Every thread registers its SelfplayState via on_state_ready, and
        _apply_live_reco updates ALL registered states — so live keys (PID
        regret/nodes, selfplay_fraction, label knobs) apply in place with no
        session restart. This path used to skip registration and fall back to
        a restart; post pause-hold/oversubscribe that restart abandons ~2x
        games_per_batch in-flight games every time the PID moves a lever, so
        the fallback stopped being cheap. apply_live_overrides is a ref swap
        (consumers read the config fresh per step), so cross-thread mutation
        is safe without per-state locks.

        ``fen_dole_queue`` (opt-in dole mode) IS shared across all N threads — each
        thread's SelfplayState drains the same list, so the server-doled batch is
        played once in total (not once per thread). The drain is lock-free and
        thread-safe: resolve_slot_opening pops exception-safely and list.pop(0) is
        atomic under the GIL, so concurrent threads get distinct seeds and an empty
        queue simply falls back to normal openings.

        ``diff_focus_norm`` is likewise shared across all N threads when
        ``diff_focus_norm_shared`` is on: this is the ONE call site where the
        per-thread multiplication of ``diff_focus_norm_warmup`` happens, because
        it is the site that turns one session into N ``play_batch`` calls. None
        (the default) leaves each thread building its own, unchanged.
        """
        n_threads = min(int(self.args.selfplay_threads), games_per_batch)
        base_games, remainder = divmod(games_per_batch, n_threads)
        thread_games = [base_games + (1 if i < remainder else 0) for i in range(n_threads)]
        lock = threading.Lock()
        self._upload_buf_lock = lock  # shared with _check_model_update

        seeds = [int(self.rng.integers(2**63)) for _ in range(n_threads)]

        def _run_one_thread(tid):
            try:
                return _play_one_thread(tid)
            except BaseException:
                # Futures are joined only at session end, which in continuous
                # mode never arrives — so without this, a thread's death is
                # invisible for the life of the session and the batch quietly
                # runs short-handed. Log at the point of failure instead.
                self.log.exception("selfplay thread %d died", tid)
                raise

        def _play_one_thread(tid):
            return play_batch(
                None, device=str(self.device), rng=np.random.default_rng(seeds[tid]),
                stockfish=sf, evaluator=eval_,
                games=thread_games[tid],
                on_game_complete=self._on_completed_game,
                on_timing=self._record_selfplay_phase_timing,
                # Sub-second swap latency while this thread lives; the
                # model-watch thread is the liveness floor if it does not.
                on_step=self._check_model_update if tid == 0 else None,
                on_state_ready=self._register_live_state,
                on_suspend=(
                    self._suspend_inflight_games
                    if self._resume_inflight_enabled else None
                ),
                stop_fn=self._stop_fn,
                pause_fn=self._pause_fn,
                fen_dole_queue=fen_dole_queue,  # shared across threads (drained once total)
                fen_sf_refute_queue=fen_sf_refute_queue,
                diff_focus_norm=diff_focus_norm,  # shared across threads, or None
                **cfgs,
            )

        try:
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                futures = [pool.submit(_run_one_thread, i) for i in range(n_threads)]
                all_stats = [f.result()[1] for f in futures]
        finally:
            self._clear_live_states()
        return self._aggregate_thread_stats(all_stats)

    def _build_shared_diff_focus_norm(
        self, cfgs: dict, games_per_batch: int,
    ) -> DiffFocusNormalizer | None:
        """The ONE difficulty-scale estimator this session shares, or None.

        None means "every SelfplayState builds its own", which is what this
        worker did before ``diff_focus_norm_shared`` existed and what it still
        does with the knob off — so the default path is unchanged, not merely
        equivalent.

        The knob is worth its own line because the cost it removes is invisible
        from the yaml: ``diff_focus_norm_warmup`` is per ESTIMATOR, and this
        worker runs one ``play_batch`` per ``--selfplay-threads`` (32 live), so
        the realized warm-up is 32x the configured number of plies per worker and
        takes 32x the wall clock, every restart. The log line names the realized
        count so the multiplication is readable in worker.log rather than
        inferable from two files.
        """
        # Subscript and attribute access, not `.get`/`getattr` with defaults: a
        # renamed key or field must fail loudly here rather than resolve to
        # "sharing off", which is indistinguishable from the knob working.
        df: DiffFocusConfig = cfgs["diff_focus"]
        if not bool(df.norm_shared):
            return None
        norm = build_diff_focus_normalizer(df)
        if norm is None:
            # norm_shared with the group off: nothing to share, and
            # SelfplayState.create would refuse a supplied estimator anyway.
            self.log.warning(
                "diff_focus_norm_shared is set but diff_focus_norm_enabled is "
                "False; no estimator is built and the knob does nothing",
            )
            return None
        n_states = (
            max(1, min(int(self.args.selfplay_threads), int(games_per_batch)))
            if self.args.threaded_selfplay else 1
        )
        self.log.info(
            "diff_focus norm estimator SHARED across %d selfplay state(s): "
            "warmup=%d plies total for this worker (unshared would be %d)",
            n_states, int(df.norm_warmup), n_states * int(df.norm_warmup),
        )
        return norm

    def _dispatch_selfplay_one_shard(
        self, *, games_per_batch: int, cfgs: dict, need_local_model: bool,
        fen_dole_queue: list[str] | None = None,
        fen_sf_refute_queue: list[str] | None = None,
    ) -> BatchStats | None:
        """Run one continuous-selfplay session. Returns None on broker errors that
        we recover from via reset (caller exits the iteration); otherwise BatchStats.

        ``fen_dole_queue`` (opt-in dole mode) is the shared server-doled seed
        batch; its selfplay slots start from these FENs. Wired into BOTH paths:
        the single path's one state and every thread of the threaded path drain
        the SAME list, so total seeded games == its length (drain is exception-safe
        and atomic under the GIL — see resolve_slot_opening)."""
        assert self.sf is not None  # caller (_run_selfplay) calls _sync_stockfish first
        _eval = self.inference_client or self._direct_evaluator
        shared_norm = self._build_shared_diff_focus_norm(cfgs, int(games_per_batch))
        try:
            if self.args.threaded_selfplay:
                return self._run_selfplay_threaded(
                    games_per_batch=int(games_per_batch),
                    sf=self.sf, eval_=_eval, cfgs=cfgs,
                    fen_dole_queue=fen_dole_queue,
                    fen_sf_refute_queue=fen_sf_refute_queue,
                    diff_focus_norm=shared_norm,
                )
  # Continuous selfplay: 256 slots always full, games recycled on completion.
  # Runs until _stop_selfplay is set (task change, pause, or shutdown).
  # Samples flow via _on_completed_game.
            try:
                _samples, stats = play_batch(
                    self.model if (need_local_model and _eval is None) else None,
                    device=str(self.device), rng=self.rng,
                    stockfish=self.sf, evaluator=_eval,
                    games=int(games_per_batch),
                    on_game_complete=self._on_completed_game,
                    on_timing=self._record_selfplay_phase_timing,
                    on_step=self._check_model_update,
                    on_state_ready=self._register_live_state,
                    on_suspend=(
                        self._suspend_inflight_games
                        if self._resume_inflight_enabled else None
                    ),
                    stop_fn=self._stop_fn,
                    pause_fn=self._pause_fn,
                    fen_dole_queue=fen_dole_queue,
                    fen_sf_refute_queue=fen_sf_refute_queue,
  # A no-op on this branch (one state per session already), passed so the
  # knob cannot mean two different things depending on --threaded-selfplay.
                    diff_focus_norm=shared_norm,
                    **cfgs,
                )
            finally:
                self._clear_live_states()
            return stats
        except (StockfishTimeoutError, StockfishDesyncError) as exc:
  # Stockfish went silent (DTZ load latency, GPU pressure, etc.), or an
  # abandoned search left an engine one result behind and the pool could
  # not build its replacement.  Kill the process and let _sync_stockfish
  # restart it on the next shard.  Both must land here: a desynced engine
  # that reaches a caller is refusing to serve, never serving stale data,
  # so a fresh process is the whole repair.
            self.log.warning("Stockfish unusable, restarting SF process: %s", exc)
            with suppress(Exception):
                self.sf.close()
            self.sf = None
            return None
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
                trial_id=self.leased_trial_id or self.fixed_trial_id or None,
            )
            self._upload_pending_shards(default_elapsed_s=0.0)

        best_sha = str((manifest.get("best_model") or {}).get("sha256") or "")
        if not self.shared_cache_enabled:
            _prune_cached_models(cache_dir=self.cache_dir, keep_shas={model_sha, best_sha})
        self._upload_pending_shards(default_elapsed_s=0.0)

    def _promote_pending_dole(self) -> tuple[list[str], list[str]]:
        """Promote stashed session-start dole (+ SF-refute) into shared live queues.

        Returns the live queue OBJECTS (possibly empty) that the session must
        hold for its whole lifetime: with pause-hold sessions running for hours,
        the first dole usually arrives MID-session, and _maybe_ingest_dole_flag
        refills these same lists in place. Returning None on empty (the old
        ``or None``) orphaned the queue — the session had nothing to drain and
        every subsequent dole refilled a list nobody held (seed injection
        silently off for the whole session; found 2026-07-12, seeds dead since
        the 07-11 swap restart).
        """
        with self._dole_lock:
            self._live_dole_queue = list(self._pending_fen_dole)
            self._pending_fen_dole = []
            self._live_sf_refute_queue = list(self._pending_sf_refute)
            self._pending_sf_refute = []
            return self._live_dole_queue, self._live_sf_refute_queue

    def _await_broker_ready(self, cfgs: dict) -> None:
        """Block session start until the inference broker answers one probe.

        Broker mode only. A freshly launched broker ``torch.compile``s the
        model on its FIRST forward (``SlotBroker._process_batch`` logs it as
        "first inference (includes kernel compile)"; there is no broker-side
        warmup forward), which takes ~1-2 min under reduce-overhead — longer
        than the client's 30s request timeout. Selfplay threads spawned into
        that window die on their first ``evaluate_legal_bf16`` call and are
        never respawned (futures are joined only at session end), which cost
        4/32 threads per worker on live trial 379f6 (-12.5% selfplay capacity
        for the whole session, 2026-08-06).

        The probe TRIGGERS the compile rather than waiting out an independent
        one, and one probe is enough because the compiled model is broker-
        process-level (one ``self._model`` behind every slot), not per-slot.
        It is sent through the production transport (same
        ``evaluate_legal_bf16`` mode as ``run_gumbel_many_c``) so it compiles
        the same legal-policy forward the session will use, with the session's
        own encoding config. Caveat: on an AOT-covered broker (``aot_dir``
        with the probe's padded bucket in the package set) the probe may be
        served by a pre-compiled AOT model without triggering torch.compile
        at all — then the gate degrades to a cheap readiness check. The
        "broker ready after N.Ns" wait time is the observable that
        disambiguates: ~compile-scale N means the probe did the warmup,
        sub-second N means the broker was already fast to serve.

        The per-attempt timeout stays the client's 30s — it is a deliberate
        wedge detector — and only the OVERALL budget is new:
        ``CAE_WORKER_BROKER_WARMUP_TIMEOUT_S``, default 240, deliberately
        below the 300s selfplay stall watchdog. On exhaustion the session
        proceeds with today's behavior — a warmup gate must never turn a
        slow-but-recovering broker into a full worker outage.
        """
        client = self.inference_client
        if client is None:
            return
        raw = os.environ.get("CAE_WORKER_BROKER_WARMUP_TIMEOUT_S", "240").strip()
        try:
            timeout_s = float(raw)
        except ValueError:
            timeout_s = 240.0
  # ⚑ Same rule as the stall watchdog: print the REALIZED value, so "240 in
  # force" and "exported value rejected, using 240" stay distinguishable.
            self.log.warning(
                "CAE_WORKER_BROKER_WARMUP_TIMEOUT_S=%r is not a number; using "
                "the default %.1fs instead. The value you exported is NOT in "
                "effect", raw, timeout_s,
            )
        if timeout_s <= 0.0:
            self.log.info(
                "broker warmup gate DISABLED (realized timeout_s=%.1f)", timeout_s,
            )
            return

  # Probe input: the standard starting position under the SESSION's encoding
  # config, with legal-move metadata built the way the production caller
  # builds it (full 4672 action ids; see gumbel_c.py's evaluate_legal_bf16
  # call). The bf16-bits transport matches _MODE_LEGAL_BF16. Construction is
  # guarded: a cfgs shape this gate does not expect (a changed key, a test
  # stub) must degrade to a loud skip, never a crash -- the gate protecting
  # session start must not be able to become the thing that breaks it.
        try:
            game_cfg = cfgs["game"]
            board = chess.Board()
            x = encode_positions_batch(
                [board],
                input_history_encoding=str(game_cfg.input_history_encoding),
                input_extra_features=str(game_cfg.input_extra_features),
            )
            xb = torch.from_numpy(x).to(torch.bfloat16).view(torch.uint16).numpy()
            legal_flat = np.ascontiguousarray(
                legal_move_indices(board), dtype=np.int32,
            )
            legal_counts = np.array([int(legal_flat.shape[0])], dtype=np.int32)
        except Exception as exc:
            self.log.warning(
                "broker warmup gate SKIPPED: probe construction failed (%s); "
                "threads spawn without warmup", exc,
            )
            return

        t0 = time.monotonic()
        deadline = t0 + timeout_s
        attempts = 0
        while True:
            if self._stop_fn():
                return  # shutdown/task change — the session itself will bail
            attempts += 1
            try:
                client.evaluate_legal_bf16(xb, legal_flat, legal_counts)
            except TimeoutError:
  # Legitimate warmup, not a stall: keep the session watchdog fed even
  # though the gate runs before _selfplay_session_active is set — the
  # progress note costs nothing and survives a future reordering.
                self._note_selfplay_progress()
                if time.monotonic() >= deadline:
                    self.log.error(
                        "broker not ready after %.0fs (%d probe attempt(s), "
                        "CAE_WORKER_BROKER_WARMUP_TIMEOUT_S=%r) -- starting "
                        "selfplay threads anyway; expect thread deaths",
                        time.monotonic() - t0, attempts, raw,
                    )
                    return
                continue
            except Exception as exc:
  # Anything but a timeout is not "still compiling". Proceed and let the
  # session-level machinery (_dispatch_selfplay_one_shard's reset/retry)
  # see the same failure unchanged, rather than teaching this gate a
  # second error protocol it would apply before the session exists.
                self.log.warning(
                    "broker warmup probe failed with a non-timeout error after "
                    "%d attempt(s): %s -- proceeding to session start", attempts, exc,
                )
                return
  # The deploy proof future sessions grep for.
            self.log.info(
                "broker ready after %.1fs (%d probe attempt(s))",
                time.monotonic() - t0, attempts,
            )
            return

    def _run_selfplay(self, manifest: dict) -> None:
        """Continuous selfplay — runs until stop signal (task change/pause/shutdown)."""
  # Do not start a session we have already been told to shut down. `run()`'s
  # loop-head check cannot cover the gap between itself and here — a SIGTERM
  # landing while the worker is between sessions (blocked in `_poll_manifest`'s
  # 30s GET, syncing assets, resuming banked games) would be erased by the reset
  # below. That window is NARROW — sessions restart 1-2x in 11h on the live
  # run — but it is silent, and the session it starts would also un-bank games
  # an earlier teardown had saved.
        if self._shutdown_requested:
            return
        self._stop_selfplay = False
        self._hold_selfplay = False
        self._last_manifest_poll_s = time.time()
        reco = manifest.get("recommended_worker") or {}
        self._active_reco = self._snapshot_reco(reco)
        self._active_assets = self._asset_fingerprint(manifest)
  # The realized difficulty this session will stamp on its shards. The LIVE
  # apply path already prints its pair ("live reco (N state(s)): ... regret=
  # ... sf_nodes=..."); session start did not, so a worker that started on a
  # corrupt reco had no line anywhere naming what it adopted. Reads the same
  # `_active_difficulty` the shard metadata does, so the log and the recorded
  # value cannot disagree -- printing the raw reco here instead would show
  # what the SERVER published rather than what these games are played at,
  # which is the whole distinction that field exists to make.
        _start_regret, _start_nodes = self._active_difficulty()
        self.log.info(
            "session-start reco applied: regret=%s sf_nodes=%s label_floor=%d"
            " score_mode=%s cp_temp=%.2f",
            f"{_start_regret:.4f}" if _start_regret is not None else "unknown",
            _start_nodes if _start_nodes is not None else "unknown",
  # The same _resolve_reco expression the GameConfig build uses, so this line
  # and the budget the labels actually get cannot disagree. The floor exists
  # because an 11x teacher cut happened with no log line anywhere; a deploy is
  # verified by label_floor>0 here plus sf_label_meta[:,0]>=floor in the shards.
            self._resolve_reco(reco, "sf_label_nodes_floor", 0, int),
  # Same rule for the policy-target score mode: its deploy is verified by
  # score_mode=cp here plus won-bucket sf_policy_target entropy dropping
  # toward ~1.0 nats within one window turnover. Same expressions as the
  # GameConfig build above, so log and targets cannot disagree.
            str(
                reco.get(
                    "sf_policy_score_mode", _SF_TARGET_DEFAULTS.sf_policy_score_mode
                )
            ),
            self._resolve_reco(
                reco, "sf_policy_cp_temp", _SF_TARGET_DEFAULTS.sf_policy_cp_temp,
            ),
        )
        model_sha = self.model_sha

        need_local_model = self.inference_client is None

        if not model_sha or (need_local_model and self.model is None):
            time.sleep(float(self.args.poll_seconds))
            return

        if need_local_model:
            assert self.model is not None
            if self._direct_evaluator is None:
                self.log.info(
                    "building local evaluator (threaded=%s compile=%s aot=%s)",
                    bool(self.args.threaded_selfplay),
                    bool(self.args.compile_inference),
                    bool(self.args.aot_dir),
                )
                # AOT package load / dispatcher warmup / pinned allocation all
                # happen in here, every one of them a driver call.
                with self._boot_hang_watchdog.stage("build_evaluator"):
                    self._direct_evaluator = self._build_evaluator(self.model)
                self.log.info("local evaluator ready type=%s", type(self._direct_evaluator).__name__)
            self._resync_evaluator_to_model()

        games_per_batch = (
            int(self.games_per_batch_local)
            if self.games_per_batch_local is not None
            else int(reco.get("games_per_batch", 8))
        )

        self._begin_resume_session(reco)

        cfgs, (sf_nodes, sf_multipv, sf_hash_mb, syzygy_path) = self._build_selfplay_configs(reco)
        warm_opening_book_cache(cfgs["opening"])
        self._sync_stockfish(
            manifest, sf_nodes, sf_multipv, sf_hash_mb, syzygy_path=syzygy_path,
        )
        assert self.sf is not None  # _sync_stockfish always assigns

  # Promote the stashed session-start dole into the shared live queue the session
  # drains (the outer loop / a prior poll stashed it — survives model-not-ready).
  # The SAME object is handed to play_batch (single path) or every thread (threaded
  # path), and mid-session doles refill it in place via _periodic_manifest_poll.
        fen_dole_queue, fen_sf_refute_queue = self._promote_pending_dole()

  # Broker path only: gate BEFORE any selfplay thread is spawned (both the
  # threaded and single dispatch paths race the broker's first-forward
  # compile), and BEFORE _selfplay_session_active is set, so the stall
  # watchdog cannot count a legitimate warmup as a stalled session.
        if self.inference_client is not None:
            self._await_broker_ready(cfgs)

        t0 = time.time()
        self._saw_completed_game = False
        self._note_selfplay_progress()
        self._selfplay_session_active = True
        self._start_selfplay_stall_watchdog()
        self._start_model_watch_thread()
        n_dole = len(fen_dole_queue)
        n_refute = len(fen_sf_refute_queue)
        search_cfg = cfgs["search"]
  # policy_temp is printed from the SearchConfig the session is about to run
  # with, and `tempered` comes from mcts.gumbel.policy_temp_active -- the same
  # predicate the search itself gates on. A yaml value that failed to reach the
  # worker therefore shows up here as 1.0/False on the very first session, in
  # the worker's own log, before a single shard is written.
        self.log.info(
            "selfplay session starting: games_per_batch=%d threaded=%s "
            "broker=%s dole=%d sf_refute=%d sims=%d gumbel_topk=%d "
            "gumbel_policy_temp=%.4g tempered=%s",
            int(games_per_batch),
            bool(self.args.threaded_selfplay),
            self.inference_client is not None,
            n_dole, n_refute,
            int(search_cfg.simulations),
            int(search_cfg.gumbel_topk),
            float(search_cfg.gumbel_policy_temp),
            policy_temp_active(float(search_cfg.gumbel_policy_temp)),
        )
        try:
            stats = self._dispatch_selfplay_one_shard(
                games_per_batch=games_per_batch, cfgs=cfgs, need_local_model=need_local_model,
                fen_dole_queue=fen_dole_queue,
                fen_sf_refute_queue=fen_sf_refute_queue,
            )
        finally:
            self._selfplay_session_active = False
  # Session over: whatever the session did NOT drain is preserved for the next
  # session start — a normally-stopped-but-empty session (games=0) keeps its full
  # batch, a recoverable dispatch failure (stats is None) keeps the remainder,
  # and a fully-drained one clears. Only draining consumes the dole.
        with self._dole_lock:
            leftover = list(self._live_dole_queue or [])
            self._live_dole_queue = None
            self._pending_fen_dole = leftover
            leftover_sf = list(self._live_sf_refute_queue or [])
            self._live_sf_refute_queue = None
            self._pending_sf_refute = leftover_sf
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
                "dispatcher stats: batches=%d avg_batch=%.1f req_per_batch=%.1f "
                "rows_per_req=%.1f q_avg=%.1f q_max=%d avg_drain_ms=%.2f "
                "avg_pack_ms=%.2f avg_submit_ms=%.2f avg_wait_ms=%.2f "
                "avg_scatter_ms=%.2f full_drains=%d",
                ds["lifetime_batches"], ds["avg_batch_size"],
                ds.get("avg_requests_per_batch", 0.0),
                ds.get("avg_rows_per_request", 0.0),
                ds.get("avg_queue_depth", 0.0),
                ds.get("queue_depth_max", 0),
                ds.get("avg_drain_ms", 0.0),
                ds.get("avg_pack_ms", 0.0),
                ds.get("avg_submit_ms", ds["avg_forward_ms"]),
                ds.get("avg_wait_ms", 0.0),
                ds.get("avg_scatter_ms", 0.0),
                ds["lifetime_full_drains"],
            )

        self._flush_and_upload_after_shard(manifest, model_sha)

    def _boot_hang_exit(self, code: int) -> None:
        """Hard-exit from the hang watchdog, killing Stockfish children first.

        **Why `os._exit` and not something gentler.** The main thread is blocked
        inside a CUDA driver call that never returns. `sys.exit` from this
        (watchdog) thread only unwinds this thread; a self-SIGTERM lands in
        `_install_shutdown_handlers`, which by design only sets flags that the
        blocked main thread will never read. `os._exit` is the only mechanism
        that reliably ends the process, and it is what the broker already uses.

        **What that costs, and what is done about it.** `os._exit` skips every
        `finally`, so `run()`'s `self._cleanup()` does NOT run. Taking the
        resources one at a time:

        * **Stockfish children — would leak, so they are killed here.** This is
          audit R2: they are spawned without a process group and their bare
          cmdline is unmatchable by `terminate_matching_processes`, so an
          orphan survives until reboot holding ~2.6 GB RSS each. SIGKILL by pid
          is used rather than `self.sf.close()` because close() sends `quit`
          and waits on per-engine locks -- a watchdog must not be able to block.
        * **Shared-memory slots — no residue.** The worker only ever ATTACHES
          (`create=False`) and `_detach_attached_shm_from_resource_tracker`
          removes it from the resource tracker, so the worker neither owns nor
          unlinks a segment. Process death drops the mapping; the broker keeps
          the segment for the replacement worker.
        * **Server lease — residue, and accepted.** There is no release
          endpoint (`grep release_lease` finds none), and a lease taken by a
          process whose GPU context is dead cannot be handed back cleanly
          anyway. It expires by TTL (`lease_seconds`, 3600s) and
          `prune_expired_leases` reclaims it. That is strictly better than the
          status quo, where the wedged worker held it indefinitely because it
          never exited at all.
        * **The log file** is opened by the launcher, which owns closing it.

        """
        with suppress(Exception):
            self._kill_stockfish_children()
        os._exit(int(code))

    def _kill_stockfish_children(self) -> int:
        """SIGKILL every live Stockfish child by pid. Returns how many were signalled.

        Deliberately lock-free and wait-free: it runs from the hang watchdog,
        where blocking would defeat the whole mechanism.
        """
        sf = self.sf
        if sf is None:
            return 0
        engines = getattr(sf, "_engines", None)
        candidates = list(engines) if isinstance(engines, list) else [sf]
        killed = 0
        for engine in candidates:
            proc = getattr(engine, "proc", None)
            pid = getattr(proc, "pid", None)
            if proc is None or not pid:
                continue
            if proc.poll() is not None:
                continue  # already exited; counting it would overstate the kill
            try:
                os.kill(int(pid), signal.SIGKILL)
                killed += 1
            except (OSError, ProcessLookupError):
                continue
        if killed:
            self.log.critical(
                "hang abort: SIGKILLed %d Stockfish child process(es) that "
                "os._exit would otherwise have orphaned", killed,
            )
        return killed

    def _cleanup(self) -> None:
        self._boot_hang_watchdog.stop()
        self._gil_probe.close()
        if self.inference_client is not None and hasattr(self.inference_client, "close"):
            with suppress(Exception):
                self.inference_client.close()
        if self.sf is not None:
            with suppress(Exception):
                self.sf.close()


if __name__ == "__main__":
    main()
