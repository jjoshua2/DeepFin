from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    DEFAULT_MAX_SHARD_POSITIONS,
    DEFAULT_MAX_SHARD_UNCOMPRESSED_BYTES,
    IN_FLIGHT_DIR_NAME,
    LOCAL_SHARD_SUFFIX,
    PENDING_DIR_NAME,
    UPLOAD_TAR_SUFFIX,
    ShardMeta,
    arrays_to_samples,
    delete_shard_path,
    extract_uploaded_shard_tar,
    is_tmp_shard_name,
    samples_to_arrays,
    save_local_shard_arrays,
    validate_array_declarations,
)
from chess_anti_engine.utils.atomic import atomic_write_text
from chess_anti_engine.utils.versioning import version_lt
import contextlib

# Pending-upload staging dir name (server side of the wire protocol — see
# ``replay.shard.PENDING_DIR_NAME`` for the canonical constant). Aliased
# here purely to keep the local docstring at the call sites.
_PENDING_DIR_NAME = PENDING_DIR_NAME
_IN_FLIGHT_DIR_NAME = IN_FLIGHT_DIR_NAME


def _compacted_token_suffix(flush_token: str) -> str:
    """Trailing filename segment linking a compacted shard to its flush token.

    Single source of truth for the commit witness: the flush path names the
    compacted shard with this suffix, and startup recovery matches on it to
    decide whether an ``_in_flight/<token>/`` group already committed. A
    silent drift between the two sides would make recovery re-seed committed
    samples (duplicates in replay).
    """
    return f"_{flush_token}{LOCAL_SHARD_SUFFIX}"


# One-shot rearm request written by the trainable when it opens a selfplay
# window (``pause_selfplay=False``). Consumed on the first *eligible* dole
# resolve so a mid-iteration trial restart can re-hand seeds for the still-
# open iteration without re-handing on every subsequent poll (or on a bare
# server restart — see ``test_seed_dole_gate_persists_across_restart``).
SEED_DOLE_REARM_FILENAME = "seed_dole_rearm.json"


class _SeedDoleGate:
    """Per-iteration blind-spot FEN-seed doling gate.

    ``opening_fen_dole_per_iter`` seeding hands the WHOLE seed list to exactly
    ONE worker-poll per training iteration (per trial): that worker plays every
    seed once (×N) as selfplay games while every other worker plays normal
    games, giving deterministic, variance-free, client-count-independent
    coverage. This gate is the "exactly one" arbiter: ``claim`` returns True for
    the first caller of a given (trial, iteration) and False thereafter, resetting
    when the iteration advances. The asyncio lock makes the check-and-set atomic
    so concurrent boundary polls can't both win.

    The last-claimed iteration per trial is persisted to ``state_path`` (atomic
    rename) and reloaded on construction, so a server restart mid-iteration does
    not re-hand the same iteration's dole (which would double the batch). Best
    effort: a load/save failure degrades to in-memory only.

    Mid-iteration *trial* restarts (cheese pause, crash, ``train.sh stop``)
    are different: the incomplete iteration stays claimed on disk while
    workers are gone, so the whole first selfplay window after resume would
    get zero seeds. The trainable writes a one-shot ``seed_dole_rearm.json``
    when opening each selfplay window; ``claim(..., allow_rearm=True)`` rolls
    that iteration back once so a single poll can win again. Concurrent polls
    still serialize — only one winner after rearm.
    """

    def __init__(self, state_path: Path | None = None) -> None:
        self._state_path = state_path
        self._lock = asyncio.Lock()
        self._last_iter: dict[str, int] = {}
        if state_path is not None and state_path.exists():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
                self._last_iter = {str(k): int(v) for k, v in dict(loaded).items()}
            except Exception:
                self._last_iter = {}

    def _persist(self) -> None:
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._last_iter), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception:
            pass  # in-memory state still holds; durability is best-effort

    async def claim(
        self,
        trial_key: str,
        training_iteration: int,
        *,
        allow_rearm: bool = False,
    ) -> bool:
        """Claim the dole for ``(trial_key, training_iteration)``.

        When ``allow_rearm`` is True and this exact iteration was already
        claimed, roll the gate back one step first so a single new winner can
        take the batch (used after a trial restart on an incomplete iter).
        """
        async with self._lock:
            last = int(self._last_iter.get(trial_key, -1))
            if allow_rearm and last == int(training_iteration):
                last = int(training_iteration) - 1
                self._last_iter[trial_key] = last
            if int(training_iteration) > last:
                self._last_iter[trial_key] = int(training_iteration)
                self._persist()
                return True
            return False


def consume_seed_dole_rearm(publish_dir: Path, training_iteration: int) -> bool:
    """Atomically consume a one-shot rearm request for ``training_iteration``.

    Returns True only when a rearm file was present, targeted this iteration,
    and this caller won the rename race (so concurrent polls only rearm once).
    """
    path = Path(publish_dir) / SEED_DOLE_REARM_FILENAME
    tmp = path.with_suffix(path.suffix + ".consuming")
    try:
        path.rename(tmp)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    try:
        raw = tmp.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        req_iter = int(data.get("training_iteration", -1))
        return req_iter == int(training_iteration)
    except Exception:
        return False
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def resolve_publish_artifact_path(publish_root: Path, filename: str) -> Path | None:
    """Return a publish-root-contained artifact path, or None on escape."""
    root = Path(publish_root).resolve()
    path = (root / str(filename)).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


def resolve_arena_user_dir(arena_root: Path, username: str) -> Path | None:
    """Return a single-directory arena user path, or None on unsafe names."""
    name = str(username or "").strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        return None
    root = Path(arena_root).resolve()
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


def shard_run_id_matches_upload_trial(upload_trial_id: str | None, shard_run_id: object) -> bool:
    """Return whether a tagged shard belongs on the upload route's trial."""
    run_id = str(shard_run_id or "").strip()
    if not run_id:
        return True
    route_trial = str(upload_trial_id or "").strip()
    return run_id == route_trial


# Aggregate counter fields shared between the upload-meta dict (input) and
# accumulator attribute (output). Each entry is both the meta-dict key and the
# accumulator field name. ``positions`` is NOT in here — it tracks
# ``len(samples)`` directly, not a meta-dict counter.
_AGGREGATE_COUNTER_FIELDS: tuple[str, ...] = (
    "games", "wins", "draws", "losses",
    "total_game_plies",
    "adjudicated_games", "tb_adjudicated_games", "total_draw_games",
    "selfplay_games", "selfplay_adjudicated_games", "selfplay_draw_games",
    "curriculum_games", "curriculum_adjudicated_games", "curriculum_draw_games",
    "plies_win", "plies_draw", "plies_loss",
    "checkmate_games", "stalemate_games",
    "diff_focus_records", "diff_focus_kept",
    "diff_focus_keep_limited", "diff_focus_sample_weight_limited",
    "gumbel_policy_diag_n",
    "gumbel_policy_argmax_is_candidate_sum", "gumbel_policy_argmax_is_action_sum",
    "gumbel_policy_legal_count_sum", "gumbel_policy_candidate_count_sum",
)

_AGGREGATE_FLOAT_FIELDS: tuple[str, ...] = (
    "diff_focus_keep_prob_sum",
    "diff_focus_sample_weight_sum",
    "diff_focus_priority_sum",
    "diff_focus_priority_sq_sum",
    "gumbel_policy_top_prob_sum",
    "gumbel_policy_action_prob_sum",
    "gumbel_policy_entropy_sum",
    "gumbel_policy_eff_moves_sum",
    "gumbel_policy_candidate_mass_sum",
    "gumbel_policy_non_candidate_top_prob_sum",
)

_MAX_OUTCOME_STAT_KEYS = 128
_QUEUED_GAMES_CACHE_TTL_S = 2.0
_MAX_BAD_SHARD_REPORT_FIELD_CHARS = 512


def _merge_outcome_stats(dst: dict[str, int], src: Any) -> None:
    if not isinstance(src, dict):
        return
    for key, val in src.items():
        key_s = str(key)
        if not re.fullmatch(r"[a-z0-9_]{1,96}", key_s):
            continue
        if key_s not in dst and len(dst) >= _MAX_OUTCOME_STAT_KEYS:
            continue
        try:
            val_i = int(val or 0)
        except (TypeError, ValueError):
            continue
        dst[key_s] = int(dst.get(key_s, 0)) + val_i


def _bounded_report_field(value: Any) -> str:
    text = str(value or "")
    if len(text) > _MAX_BAD_SHARD_REPORT_FIELD_CHARS:
        return text[:_MAX_BAD_SHARD_REPORT_FIELD_CHARS] + "...<truncated>"
    return text


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
    diff_focus_records: int = 0
    diff_focus_kept: int = 0
    diff_focus_keep_prob_sum: float = 0.0
    diff_focus_keep_limited: int = 0
    diff_focus_sample_weight_sum: float = 0.0
    diff_focus_sample_weight_limited: int = 0
    diff_focus_priority_sum: float = 0.0
    diff_focus_priority_sq_sum: float = 0.0
    diff_focus_priority_min: float = float("inf")
    diff_focus_priority_max: float = -float("inf")
    gumbel_policy_diag_n: int = 0
    gumbel_policy_top_prob_sum: float = 0.0
    gumbel_policy_action_prob_sum: float = 0.0
    gumbel_policy_entropy_sum: float = 0.0
    gumbel_policy_eff_moves_sum: float = 0.0
    gumbel_policy_candidate_mass_sum: float = 0.0
    gumbel_policy_non_candidate_top_prob_sum: float = 0.0
    gumbel_policy_argmax_is_candidate_sum: int = 0
    gumbel_policy_argmax_is_action_sum: int = 0
    gumbel_policy_legal_count_sum: int = 0
    gumbel_policy_candidate_count_sum: int = 0
    outcome_stats: dict[str, int] = field(default_factory=dict)
    model_step: int | None = None
    input_history_encoding: str | None = None
    input_history_unknown_seen: bool = False
    # None until the first upload; absent in shard meta means the flag
    # predates the field, which provably means off.
    history_rep_fix: bool | None = None
    # Disk-resident extracted shards that contributed to this accumulator and
    # have NOT yet been folded into a compacted shard. Deleted only after the
    # compacted shard has been written to disk so a crash mid-flush leaves the
    # pending shards in place to be replayed at restart.
    pending_paths: list[Path] = field(default_factory=list)

    def add_upload(
        self,
        *,
        samples: list[ReplaySample],
        meta: dict[str, Any],
        now_unix: float,
    ) -> None:
        self.samples.extend(samples)
        self.positions += len(samples)
        for field_name in _AGGREGATE_COUNTER_FIELDS:
            setattr(
                self, field_name,
                getattr(self, field_name) + int(meta.get(field_name) or 0),
            )
        for field_name in _AGGREGATE_FLOAT_FIELDS:
            setattr(
                self, field_name,
                getattr(self, field_name) + float(meta.get(field_name) or 0.0),
            )
        incoming_diff_records = int(meta.get("diff_focus_records") or 0)
        if incoming_diff_records > 0:
            incoming_min = float(meta.get("diff_focus_priority_min") or 0.0)
            incoming_max = float(meta.get("diff_focus_priority_max") or 0.0)
            if int(self.diff_focus_records) == incoming_diff_records:
                self.diff_focus_priority_min = incoming_min
                self.diff_focus_priority_max = incoming_max
            else:
                self.diff_focus_priority_min = min(float(self.diff_focus_priority_min), incoming_min)
                self.diff_focus_priority_max = max(float(self.diff_focus_priority_max), incoming_max)
        step_raw = meta.get("model_step")
        if step_raw is not None:
            self.model_step = int(step_raw)
        history_raw = meta.get("input_history_encoding")
        if history_raw is None:
            if self.input_history_encoding is not None:
                raise ValueError(
                    "cannot compact uploads with missing input_history_encoding "
                    f"and {self.input_history_encoding!r}"
                )
            self.input_history_unknown_seen = True
        else:
            history = str(history_raw)
            if self.input_history_unknown_seen:
                raise ValueError(
                    "cannot compact uploads with missing input_history_encoding "
                    f"and {history!r}"
                )
            if self.input_history_encoding is None:
                self.input_history_encoding = history
            elif self.input_history_encoding != history:
                raise ValueError(
                    "cannot compact uploads with mixed input_history_encoding "
                    f"{self.input_history_encoding!r} and {history!r}"
                )
        rep_fix = bool(meta.get("history_rep_fix") or False)
        if self.history_rep_fix is None:
            self.history_rep_fix = rep_fix
        elif bool(self.history_rep_fix) != rep_fix:
            raise ValueError(
                "cannot compact uploads with mixed history_rep_fix "
                f"{bool(self.history_rep_fix)} and {rep_fix}"
            )
        _merge_outcome_stats(self.outcome_stats, meta.get("outcome_stats"))
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
    return bool(float(max_age_s) > 0.0 and float(now_unix) - float(acc.created_at_unix) >= float(max_age_s))


def _flush_buffered_upload_to_inbox(
    *,
    inbox_root: Path,
    acc: _BufferedUploadAccumulator,
    now_unix: float,
    flush_token: str,
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
        input_history_encoding=acc.input_history_encoding,
        history_rep_fix=bool(acc.history_rep_fix or False),
        games=int(acc.games),
        positions=int(acc.positions),
        wins=int(acc.wins),
        draws=int(acc.draws),
        losses=int(acc.losses),
        total_game_plies=int(acc.total_game_plies),
        adjudicated_games=int(acc.adjudicated_games),
        tb_adjudicated_games=int(acc.tb_adjudicated_games),
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
        diff_focus_records=int(acc.diff_focus_records),
        diff_focus_kept=int(acc.diff_focus_kept),
        diff_focus_keep_prob_sum=float(acc.diff_focus_keep_prob_sum),
        diff_focus_keep_limited=int(acc.diff_focus_keep_limited),
        diff_focus_sample_weight_sum=float(acc.diff_focus_sample_weight_sum),
        diff_focus_sample_weight_limited=int(acc.diff_focus_sample_weight_limited),
        diff_focus_priority_sum=float(acc.diff_focus_priority_sum),
        diff_focus_priority_sq_sum=float(acc.diff_focus_priority_sq_sum),
        diff_focus_priority_min=(
            float(acc.diff_focus_priority_min) if int(acc.diff_focus_records) > 0 else 0.0
        ),
        diff_focus_priority_max=(
            float(acc.diff_focus_priority_max) if int(acc.diff_focus_records) > 0 else 0.0
        ),
        gumbel_policy_diag_n=int(acc.gumbel_policy_diag_n),
        gumbel_policy_top_prob_sum=float(acc.gumbel_policy_top_prob_sum),
        gumbel_policy_action_prob_sum=float(acc.gumbel_policy_action_prob_sum),
        gumbel_policy_entropy_sum=float(acc.gumbel_policy_entropy_sum),
        gumbel_policy_eff_moves_sum=float(acc.gumbel_policy_eff_moves_sum),
        gumbel_policy_candidate_mass_sum=float(acc.gumbel_policy_candidate_mass_sum),
        gumbel_policy_non_candidate_top_prob_sum=float(
            acc.gumbel_policy_non_candidate_top_prob_sum,
        ),
        gumbel_policy_argmax_is_candidate_sum=int(acc.gumbel_policy_argmax_is_candidate_sum),
        gumbel_policy_argmax_is_action_sum=int(acc.gumbel_policy_argmax_is_action_sum),
        gumbel_policy_legal_count_sum=int(acc.gumbel_policy_legal_count_sum),
        gumbel_policy_candidate_count_sum=int(acc.gumbel_policy_candidate_count_sum),
        outcome_stats=dict(acc.outcome_stats),
    )
    # ``flush_token`` doubles as the compacted shard's uniqueness suffix and
    # the link back to ``_in_flight/<flush_token>/``. Recovery globs for this
    # token in ``_compacted/*`` to decide whether the in-flight group has
    # already committed.
    final = compacted_dir / (
        f"{int(now_unix)}_{str(acc.model_sha256)[:8]}_{int(acc.games)}g_{int(acc.positions)}p"
        f"{_compacted_token_suffix(flush_token)}"
    )
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
    max_upload_positions: int = DEFAULT_MAX_SHARD_POSITIONS,
    max_upload_uncompressed_bytes: int = DEFAULT_MAX_SHARD_UNCOMPRESSED_BYTES,
):
    """Create the HTTP server.

    Layout under server_root:
    - publish/manifest.json
    - publish/latest_model.pt
    - inbox/<username>/<sha256>.zarr
    - users.json
    """

    try:
        from fastapi import (
            Body,
            Depends,
            FastAPI,
            File,
            Header,
            HTTPException,
            UploadFile,
        )
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
    upload_accumulators: dict[tuple[str | None, str, str], _BufferedUploadAccumulator] = {}
    recent_upload_shas: dict[tuple[str | None, str], float] = {}
    upload_lock = threading.Lock()
    queued_games_cache: dict[str | None, tuple[float, dict[str, int]]] = {}
    queued_games_cache_lock = threading.Lock()
    seed_dole_gate = _SeedDoleGate(state_path=root / "seed_dole_gate.json")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(_app):  # skylos: ignore (FastAPI lifespan signature requires app arg)
        try:
            yield
        finally:
            _flush_ready_upload_accumulators(force_age=False, force_all=True)

    app = FastAPI(title="chess-anti-engine server", version="0.1", lifespan=_lifespan)

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

    def _upload_history_acc_key(meta: dict[str, Any]) -> str:
        raw = meta.get("input_history_encoding")
        history = "<missing>" if raw is None else str(raw)
        # history_rep_fix changes the encoded planes under the same history
        # mode, so it is part of compaction identity (absent means off).
        return f"{history}|repfix={bool(meta.get('history_rep_fix') or False)}"

    def _invalidate_queued_games_cache(trial_id: str | None) -> None:
        with queued_games_cache_lock:
            queued_games_cache.pop(_normalize_trial_id(trial_id), None)

    def _try_flush_and_pop(acc_key, *, inbox_root: Path, acc, now_unix: float) -> bool:
        """Flush an accumulator to disk; on success pop it, on failure leave it.

        Crash-safety contract:
        - Move pending zarrs into ``_in_flight/<flush_token>/`` BEFORE writing
          the compacted shard. The compacted shard's filename also embeds
          ``<flush_token>``, so its presence on disk is a durable witness
          that this group of pending zarrs is now committed.
        - Recovery (``_recover_in_flight_dirs``) handles every crash window:
          before-rename → pending intact, re-seeded; mid-rename → some in
          flight, no compacted → moved back; post-rename, no compacted →
          moved back; post-compacted → in-flight deleted as committed.

        Returning False on a flush error leaves the acc in memory and
        moves the in-flight zarrs back to ``_pending`` so the next attempt
        starts from the same on-disk state. If any rename-back fails, the
        in-flight group is left in place instead (never deleted) with
        ``pending_paths`` pointing at the surviving locations; recovery
        re-seeds such orphaned groups on the next startup.
        """
        flush_token = secrets.token_hex(8)
        in_flight_dir = inbox_root / _IN_FLIGHT_DIR_NAME / flush_token
        try:
            in_flight_dir.mkdir(parents=True, exist_ok=False)
        except Exception:
            log.exception(
                "failed to create in-flight dir %s; keeping accumulator for retry",
                in_flight_dir,
            )
            return False
        moved: list[tuple[Path, Path]] = []  # (original_pending, in_flight_path)
        try:
            for pending in list(acc.pending_paths):
                target = in_flight_dir / pending.name
                pending.replace(target)
                moved.append((pending, target))
            acc.pending_paths = [target for _, target in moved]
        except Exception:
            log.exception(
                "failed to stage pending shards into %s; rolling back",
                in_flight_dir,
            )
            rolled_back: list[Path] = []
            rollback_failed = False
            for original, target in moved:
                try:
                    target.replace(original)
                    rolled_back.append(original)
                except Exception:
                    log.exception("rollback rename failed for %s -> %s", target, original)
                    rollback_failed = True
                    rolled_back.append(target)  # data is still at the in-flight path
            moved_originals = {orig for orig, _ in moved}
            acc.pending_paths = rolled_back + [
                p for p in acc.pending_paths if p not in moved_originals
            ]
            if rollback_failed:
                # Do NOT delete the in-flight dir: it still holds zarrs whose
                # rollback rename failed. Startup recovery re-seeds orphaned
                # in-flight groups (no matching compacted shard) to _pending,
                # so leaving the dir is safe; deleting it would lose samples.
                log.error(
                    "leaving %s in place (partial rollback) for startup recovery",
                    in_flight_dir,
                )
            else:
                delete_shard_path(in_flight_dir)
            return False

        try:
            compacted_path = _flush_buffered_upload_to_inbox(
                inbox_root=inbox_root,
                acc=acc,
                now_unix=now_unix,
                flush_token=flush_token,
            )
        except Exception:
            log.exception(
                "compaction flush failed for key=%r; keeping accumulator (%d samples) for retry",
                acc_key, int(getattr(acc, "positions", 0)),
            )
            # Compacted shard never landed; restore pending so the retry
            # path (and any later recovery) operates on the same state as
            # before the flush attempt.
            restored: list[Path] = []
            restore_failed = False
            for original, target in moved:
                try:
                    target.replace(original)
                    restored.append(original)
                except Exception:
                    log.exception("restore rename failed for %s -> %s", target, original)
                    restore_failed = True
                    restored.append(target)  # data is still at the in-flight path
            acc.pending_paths = restored
            if restore_failed:
                # Same rationale as the staging rollback above: the in-flight
                # dir still holds zarrs that pending_paths now references —
                # deleting it here would destroy them. Startup recovery
                # handles orphaned in-flight groups.
                log.error(
                    "leaving %s in place (partial restore) for startup recovery",
                    in_flight_dir,
                )
            else:
                delete_shard_path(in_flight_dir)
            return False

        if compacted_path is not None:
            # Compacted shard exists with ``flush_token`` in its name — this is
            # the commit point. From here, the in-flight group is safe to
            # delete; if we crash before this delete completes, recovery
            # token-matches and deletes the leftover.
            delete_shard_path(in_flight_dir)
            acc.pending_paths.clear()
        upload_accumulators.pop(acc_key, None)
        _invalidate_queued_games_cache(acc.trial_id)
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

            ready_keys: list[tuple[str | None, str, str]] = []
            for key, acc in upload_accumulators.items():
                trial_key, _model_sha, _history_key = key
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
        except (OSError, json.JSONDecodeError):
            return None  # manifest mid-write or missing

    def _iter_visible_inbox_shards(inbox_root: Path) -> list[Path]:
        """List upload shards visible to learner ingest, excluding staging dirs."""
        paths: list[Path] = []
        try:
            user_dirs = list(inbox_root.iterdir())
        except FileNotFoundError:
            return paths
        for user_dir in user_dirs:
            if not user_dir.is_dir() or is_tmp_shard_name(user_dir.name):
                continue
            if user_dir.name in {_PENDING_DIR_NAME, _IN_FLIGHT_DIR_NAME}:
                continue
            try:
                for entry in user_dir.iterdir():
                    if is_tmp_shard_name(entry.name):
                        continue
                    if entry.name.endswith(LOCAL_SHARD_SUFFIX):
                        paths.append(entry)
            except FileNotFoundError:
                continue
        return sorted(paths)

    def _queued_games_by_model(*, trial_id: str | None) -> dict[str, int]:
        """Count un-ingested uploaded games by model SHA for one trial."""
        tid = _normalize_trial_id(trial_id)
        now_unix = time.time()
        with queued_games_cache_lock:
            cached = queued_games_cache.get(tid)
            if cached is not None:
                cached_at, cached_totals = cached
                if (now_unix - float(cached_at)) <= _QUEUED_GAMES_CACHE_TTL_S:
                    return dict(cached_totals)
        totals: dict[str, int] = {}
        with upload_lock:
            for (acc_tid, acc_sha, _history_key), acc in upload_accumulators.items():
                if acc_tid != tid:
                    continue
                sha = str(acc_sha or "")
                if not sha:
                    continue
                totals[sha] = int(totals.get(sha, 0)) + int(acc.games)
        for shard_path in _iter_visible_inbox_shards(_inbox_root(tid)):
            try:
                _arrs, meta = load_shard_arrays(shard_path, lazy=True)
            except Exception:
                continue
            sha = str(meta.get("model_sha256") or "")
            if not sha:
                continue
            totals[sha] = int(totals.get(sha, 0)) + int(meta.get("games") or 0)
        with queued_games_cache_lock:
            queued_games_cache[tid] = (now_unix, dict(totals))
        return totals

    def _apply_dynamic_stale_pause(trial_id: str | None, manifest: dict[str, Any]) -> dict[str, Any]:
        """Pause workers once enough old-model backlog exists for next ingest."""
        backpressure = manifest.get("backpressure")
        if not isinstance(backpressure, dict):
            return manifest
        target_games = int(backpressure.get("stale_pause_target_games") or 0)
        if target_games <= 0:
            return manifest
        reco = manifest.get("recommended_worker")
        if not isinstance(reco, dict):
            return manifest
        if bool(reco.get("pause_selfplay")) or bool(backpressure.get("pause_selfplay")):
            return manifest
        model_sha = str(backpressure.get("stale_pause_model_sha") or "")
        if not model_sha:
            model_sha = str((manifest.get("model") or {}).get("sha256") or "")
        queued_by_model = _queued_games_by_model(trial_id=trial_id)
        queued_games = int(queued_by_model.get(model_sha, 0))
        stale_queued_games = int(
            sum(games for sha, games in queued_by_model.items() if str(sha) != model_sha)
        )
        total_queued_games = int(sum(queued_by_model.values()))
        if queued_games < target_games and stale_queued_games < target_games:
            return manifest

        out = dict(manifest)
        out_reco = dict(reco)
        out_backpressure = dict(backpressure)
        if stale_queued_games >= target_games:
            pause_games = stale_queued_games
            reason = (
                f"stale backlog target reached: {stale_queued_games}/{target_games} "
                f"queued old-model games"
            )
        else:
            pause_games = queued_games
            reason = (
                f"stale backlog target reached: {queued_games}/{target_games} "
                f"games for model {model_sha[:8]}"
            )
        out_reco["pause_selfplay"] = True
        out_reco["pause_reason"] = reason
        out_backpressure["pause_selfplay"] = True
        out_backpressure["pause_reason"] = reason
        out_backpressure["stale_pause_queued_games"] = int(pause_games)
        out_backpressure["stale_pause_current_queued_games"] = int(queued_games)
        out_backpressure["stale_pause_stale_queued_games"] = int(stale_queued_games)
        out_backpressure["stale_pause_total_queued_games"] = int(total_queued_games)
        out["recommended_worker"] = out_reco
        out["backpressure"] = out_backpressure
        return out

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
                        with contextlib.suppress(Exception):
                            self.path.unlink(missing_ok=True)
                    time.sleep(0.05)

        def __exit__(self, _exc_type, _exc, _tb) -> None:
            if self._held:
                with contextlib.suppress(Exception):
                    self.path.unlink(missing_ok=True)

    def _load_json_stats(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
  # file mid-write or corrupted — caller retries next poll
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
        return device if device else "cpu"

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
                    with contextlib.suppress(Exception):
                        entry["last_cpu_count"] = int(cpu_count)
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
        if rec.disabled:
            raise HTTPException(status_code=403, detail="user disabled")
        if not verify_password(str(creds.password), rec):
            raise HTTPException(status_code=401, detail="bad password")
        return str(creds.username)

    def _record_bad_shard_report(
        trial_id: str | None,
        *,
        username: str,
        payload: dict[str, Any],
        x_cae_worker_version: str | None,
        x_cae_protocol_version: str | None,
        x_cae_worker_lease_id: str | None,
        x_cae_machine_id: str | None,
    ) -> dict[str, Any]:
        ok, reason = _check_worker_compat(
            trial_id=trial_id,
            worker_version=x_cae_worker_version,
            worker_protocol=x_cae_protocol_version,
        )
        if not ok:
            log.warning("rejecting bad-shard report from user=%s: %s", username, reason)
            return {"stored": False, "rejected": True, "reason": reason}
        qdir = _quarantine_root(trial_id) / "client_reports"
        qdir.mkdir(parents=True, exist_ok=True)
        now_unix = time.time()
        payload_summary = {
            "shard_name": _bounded_report_field(payload.get("shard_name")),
            "reason": _bounded_report_field(payload.get("reason")),
        }
        report = {
            "reported_at_unix": now_unix,
            "username": str(username),
            "trial_id": _normalize_trial_id(trial_id),
            "worker_version": x_cae_worker_version,
            "protocol_version": x_cae_protocol_version,
            "lease_id": x_cae_worker_lease_id,
            "machine_id": x_cae_machine_id,
            "payload": payload_summary,
        }
        out = qdir / f"{int(now_unix)}_{secrets.token_hex(8)}.json"
        atomic_write_text(out, json.dumps(report, indent=2, sort_keys=True))
        log.warning(
            "worker reported bad shard trial=%s user=%s machine=%s shard=%s reason=%s",
            _normalize_trial_id(trial_id),
            username,
            x_cae_machine_id,
            payload_summary["shard_name"],
            payload_summary["reason"],
        )
        return {"stored": True, "path": str(out)}

    def _get_manifest_impl(
        trial_id: str | None,
        *,
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> dict[str, Any]:
        """Build the manifest served to a worker poll (post stale-pause).

        Returns the manifest DICT (not a Response) so the async handler can add
        the per-request ``dole_fen_seeds`` flag under the seed-dole lock before
        serializing. Raises HTTPException on 404 (unpublished) / 426 (too old)."""
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

        manifest = json.loads(mf.read_text(encoding="utf-8"))
        return _apply_dynamic_stale_pause(trial_id, manifest)

    async def _resolve_dole_fen_seeds(trial_id: str | None, manifest: dict[str, Any]) -> bool:
        """Whether THIS poll should receive the doled seed batch this iteration.

        True only when dole mode is on (recommended_worker.opening_fen_dole_per_iter
        > 0), a FEN list is actually published (top-level ``opening_fen_list``
        asset present), the task is selfplay, selfplay is NOT paused, and this poll
        is the first for the current ``training_iteration`` (arbitrated by
        ``seed_dole_gate``). Always resolved (True/False) so the worker sees an
        explicit field."""
        reco = manifest.get("recommended_worker")
        if not isinstance(reco, dict):
            return False
        if int(reco.get("opening_fen_dole_per_iter", 0) or 0) <= 0:
            return False
        if not isinstance(manifest.get("opening_fen_list"), dict):
            return False
  # Only a selfplay task can play the seeds. An arena (or other) task would take
  # the worker's non-selfplay path and never ingest, silently burning the single
  # per-iteration claim; leave it unclaimed for a selfplay poll instead.
        task = manifest.get("task") or {"type": "selfplay"}
        if str((task if isinstance(task, dict) else {}).get("type", "selfplay")).lower() != "selfplay":
            return False
  # Don't burn the single per-iteration claim on a paused poll: the worker drops
  # a paused manifest (returns None from _poll_manifest) before it can ingest the
  # seeds, so claiming here would consume the dole without playing any games. The
  # gate stays unclaimed so a later non-paused poll this iteration can win it.
        backpressure = manifest.get("backpressure")
        if bool(reco.get("pause_selfplay")) or (
            isinstance(backpressure, dict) and bool(backpressure.get("pause_selfplay"))
        ):
            return False
        trial_key = str(_normalize_trial_id(trial_id) or "")
        training_iteration = int(manifest.get("training_iteration", 0) or 0)
        # One-shot rearm from the trainable's selfplay-window open: only
        # consumed on an eligible (selfplay, unpaused, dole-on) poll so a
        # paused/arena poll cannot burn it. Claim+rearm share the gate lock
        # so concurrent eligible polls still yield exactly one winner.
        allow_rearm = consume_seed_dole_rearm(
            _publish_root(trial_id), training_iteration,
        )
        return await seed_dole_gate.claim(
            trial_key, training_iteration, allow_rearm=allow_rearm,
        )

    async def _serve_manifest(
        trial_id: str | None,
        *,
        x_cae_worker_version: str | None,
        x_cae_protocol_version: str | None,
    ) -> Any:
        manifest = _get_manifest_impl(
            trial_id,
            x_cae_worker_version=x_cae_worker_version,
            x_cae_protocol_version=x_cae_protocol_version,
        )
        manifest["dole_fen_seeds"] = await _resolve_dole_fen_seeds(trial_id, manifest)
        return JSONResponse(content=manifest)

    @app.get("/v1/manifest")
    async def get_manifest(
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        return await _serve_manifest(
            None,
            x_cae_worker_version=x_cae_worker_version,
            x_cae_protocol_version=x_cae_protocol_version,
        )

    @app.get("/v1/trials/{trial_id}/manifest")
    async def get_trial_manifest(
        trial_id: str,
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        return await _serve_manifest(
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

    @app.get("/v1/trials/{trial_id}/opening_fen_list")
    def get_trial_opening_fen_list(trial_id: str) -> Any:
        # Unlike the two opening-book routes above (server-launch-fixed), this
        # is served from the manifest-tracked publish_dir copy so a yaml
        # opening_fen_list_path change takes effect on the next manifest
        # publish, no restart needed (see _LAUNCH_FIXED_ASSET_PATH_KEYS).
        # Trial-scoped (unlike opening_book): the manifest that advertises it
        # is built per-trial (_publish_distributed_trial_state), so the
        # matching artifact only ever lives under that trial's publish dir —
        # an un-scoped route defaulting to trial_id=None would read the
        # wrong (empty) directory whenever the server manages more than the
        # single default trial.
        p = _artifact_from_publish("opening_fen_list", default_name="opening_fen_list_live.txt", trial_id=trial_id)
        if p is None:
            raise HTTPException(status_code=404, detail="no opening FEN list configured")
        return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

    def _artifact_from_publish(key: str, *, default_name: str, trial_id: str | None = None) -> Path | None:
        mf = _load_manifest(trial_id) or {}
        rec = mf.get(key)
        if isinstance(rec, dict) and rec.get("filename"):
            name = str(rec.get("filename"))
        else:
            name = str(default_name)
        p = resolve_publish_artifact_path(_publish_root(trial_id), name)
        if p is None:
            log.warning(
                "rejecting manifest artifact path outside publish root: key=%s filename=%r",
                key, name,
            )
            return None
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
            zarr_root = extract_uploaded_shard_tar(
                tmp,
                tmp_zarr,
                max_extract_bytes=int(max_upload_uncompressed_bytes),
            )
            shard_arrs_lazy, meta = load_shard_arrays(zarr_root, lazy=True)
            validate_array_declarations(
                shard_arrs_lazy,
                max_positions=int(max_upload_positions),
                max_uncompressed_bytes=int(max_upload_uncompressed_bytes),
            )
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

        trial_key = _normalize_trial_id(trial_id)
        if not shard_run_id_matches_upload_trial(trial_key, meta.get("run_id")):
            tmp.unlink(missing_ok=True)
            delete_shard_path(tmp_zarr)
            return {
                "stored": False,
                "rejected": True,
                "reason": "shard run_id does not match upload trial",
            }

        positions = int(shard_arrs["x"].shape[0])
        tmp.unlink(missing_ok=True)
        upload_seen_key = (trial_key, sha)
        now_unix = time.time()
        stored = False
        # Atomically promote the extracted zarr group to the pending dir
        # before acknowledging the upload. ``Path.replace`` is atomic on the
        # same filesystem (the staging path and pending_dir share
        # inbox_root). If we fail to promote, drop the temp shard and
        # reject the upload — workers retry on non-stored responses.
        #
        # ``extract_uploaded_shard_tar`` returns the actual zarr group root,
        # which may be ``tmp_zarr`` itself or a single nested child dir
        # holding ``.zgroup``. Promote that exact directory so the pending
        # path is a directly loadable shard.
        pending_dir = inbox_root / _PENDING_DIR_NAME
        pending_dir.mkdir(parents=True, exist_ok=True)
        # Full sha (not sha[:8]) so recovery can repopulate
        # ``recent_upload_shas`` from the filename alone — without that, a
        # worker retry after a crash would not be deduped against the
        # already-recovered samples.
        pending_path = pending_dir / (
            f"{int(now_unix)}_{sha}_{secrets.token_hex(8)}{LOCAL_SHARD_SUFFIX}"
        )
        try:
            zarr_root.replace(pending_path)
        except Exception as exc:
            delete_shard_path(tmp_zarr)
            log.exception("failed to promote extracted shard to pending dir")
            return {
                "stored": False,
                "rejected": True,
                "reason": f"pending stage failed: {type(exc).__name__}: {exc}",
            }
        # Clean up the (now empty) wrapper dir if extraction nested a child.
        if zarr_root != tmp_zarr:
            delete_shard_path(tmp_zarr)
        tmp_zarr = None
        with upload_lock:
            if upload_seen_key not in recent_upload_shas:
                model_sha = str(meta.get("model_sha256") or sha)
                acc_key = (trial_key, model_sha, _upload_history_acc_key(meta))
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
                acc.pending_paths.append(pending_path)
                recent_upload_shas[upload_seen_key] = now_unix
                stored = True
                _invalidate_queued_games_cache(trial_key)
                if _buffered_upload_ready(
                    acc=acc,
                    now_unix=now_unix,
                    target_positions=compact_target_positions,
                    max_age_s=compact_max_age_seconds,
                ):
                    _try_flush_and_pop(acc_key, inbox_root=inbox_root, acc=acc, now_unix=now_unix)
            else:
                # Duplicate upload (already accumulated). The pending shard we
                # just promoted is redundant — drop it so it isn't re-seeded
                # on restart.
                delete_shard_path(pending_path)
                _invalidate_queued_games_cache(trial_key)

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

    @app.post("/v1/report_bad_shard")
    def report_bad_shard(
        payload: dict[str, Any] = Body(...),
        username: str = Depends(_auth_user),
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
        x_cae_worker_lease_id: str | None = Header(None, alias="X-CAE-Worker-Lease-ID"),
        x_cae_machine_id: str | None = Header(None, alias="X-CAE-Machine-ID"),
    ) -> Any:
        return _record_bad_shard_report(
            None,
            username=username,
            payload=payload,
            x_cae_worker_version=x_cae_worker_version,
            x_cae_protocol_version=x_cae_protocol_version,
            x_cae_worker_lease_id=x_cae_worker_lease_id,
            x_cae_machine_id=x_cae_machine_id,
        )

    @app.post("/v1/trials/{trial_id}/report_bad_shard")
    def report_trial_bad_shard(
        trial_id: str,
        payload: dict[str, Any] = Body(...),
        username: str = Depends(_auth_user),
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
        x_cae_worker_lease_id: str | None = Header(None, alias="X-CAE-Worker-Lease-ID"),
        x_cae_machine_id: str | None = Header(None, alias="X-CAE-Machine-ID"),
    ) -> Any:
        return _record_bad_shard_report(
            trial_id,
            username=username,
            payload=payload,
            x_cae_worker_version=x_cae_worker_version,
            x_cae_protocol_version=x_cae_protocol_version,
            x_cae_worker_lease_id=x_cae_worker_lease_id,
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
                raise HTTPException(status_code=400, detail=f"bad int field {k}") from None

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
        user_dir = resolve_arena_user_dir(arena_root, username)
        if user_dir is None:
            raise HTTPException(status_code=400, detail="invalid username")
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

    # Crash-recovery, two phases:
    #
    # Phase 1 — ``_recover_in_flight_dirs``: settle ``_in_flight/<token>/``
    # groups left over from a flush that crashed mid-cleanup. The compacted
    # shard's filename embeds ``<token>``, so:
    #   - if ``_compacted/*<token>*`` exists, the flush committed; the
    #     in-flight group is just leftover state to delete.
    #   - if no compacted match, the flush never committed; move each
    #     in-flight zarr back to ``_pending`` so phase 2 re-seeds it.
    #
    # Phase 2 — ``_scan_pending_dir``: re-seed accumulators from any pending
    # shards left on disk (uploads accepted but never flushed). Filename
    # encodes the full upload sha, so we also backfill ``recent_upload_shas``
    # — without that, a worker retry after the crash would not be deduped
    # against the recovered samples and would double-count. If the same sha
    # appears twice (e.g. a duplicate upload whose ``delete_shard_path``
    # silently failed), only the first pending file is re-seeded; the rest
    # are orphaned duplicates that get deleted.
    #
    # Pending shards are NOT picked up by the trainable's inbox scan
    # (filtered in ``_iter_shard_paths_nested``), so this is the only path
    # that turns them back into compacted shards.

    def _parse_pending_sha(name: str) -> str | None:
        """Extract the full upload sha from a pending shard filename.

        Format: ``<int_now>_<sha64>_<token>.zarr``. Returns ``None`` if the
        middle field doesn't look like a sha256 hex digest.
        """
        stem = name[: -len(LOCAL_SHARD_SUFFIX)] if name.endswith(LOCAL_SHARD_SUFFIX) else name
        parts = stem.split("_")
        if len(parts) < 3:
            return None
        candidate = parts[1]
        if len(candidate) != 64 or any(c not in "0123456789abcdef" for c in candidate):
            return None
        return candidate

    def _recover_in_flight_dirs(
        *,
        in_flight_root: Path,
        compacted_dir: Path,
        trial_key: str | None,
    ) -> None:
        if not in_flight_root.is_dir():
            return
        for token_dir in sorted(in_flight_root.iterdir()):
            if not token_dir.is_dir():
                continue
            token = token_dir.name
            committed = compacted_dir.is_dir() and any(
                p.name.endswith(_compacted_token_suffix(token))
                for p in compacted_dir.iterdir()
            )
            if committed:
                # Compacted shard exists for this token → samples already
                # durable. Backfill upload-sha dedupe keys before cleanup so
                # a worker retry after restart cannot be accepted again.
                for entry in sorted(token_dir.iterdir()):
                    if not entry.name.endswith(LOCAL_SHARD_SUFFIX):
                        continue
                    upload_sha = _parse_pending_sha(entry.name)
                    if upload_sha is None:
                        continue
                    try:
                        mtime = float(entry.stat().st_mtime)
                    except OSError:
                        mtime = float(time.time())
                    recent_upload_shas[(trial_key, upload_sha)] = mtime
                delete_shard_path(token_dir)
                continue
            # No matching compacted shard → flush never committed. Move
            # every shard back to ``_pending`` for phase 2 to re-seed.
            pending_dir = in_flight_root.parent / _PENDING_DIR_NAME
            pending_dir.mkdir(parents=True, exist_ok=True)
            restore_failed = False
            for entry in sorted(token_dir.iterdir()):
                if not entry.name.endswith(LOCAL_SHARD_SUFFIX):
                    continue
                try:
                    entry.replace(pending_dir / entry.name)
                except Exception:
                    log.exception(
                        "failed to restore in-flight shard %s to pending; leaving in place",
                        entry,
                    )
                    restore_failed = True
            if restore_failed:
                # Same contract as the live flush paths: the token dir still
                # holds shards whose restore rename failed — deleting it would
                # destroy them. Leave it for the next startup recovery pass.
                log.error(
                    "leaving %s in place (partial restore) for a later recovery",
                    token_dir,
                )
            else:
                delete_shard_path(token_dir)

    def _scan_pending_dir(*, pending_dir: Path, trial_key: str | None) -> int:
        if not pending_dir.is_dir():
            return 0
        recovered = 0
        for entry in sorted(pending_dir.iterdir()):
            name = entry.name
            if is_tmp_shard_name(name):
                continue
            if not name.endswith(LOCAL_SHARD_SUFFIX):
                continue
            upload_sha = _parse_pending_sha(name)
            # Orphaned duplicate: an earlier pending with the same upload
            # sha was already re-seeded this scan, so this one's samples
            # are already in the accumulator. Treat as a leftover from a
            # silent ``delete_shard_path`` failure on the duplicate-upload
            # path and drop it.
            if upload_sha is not None and (trial_key, upload_sha) in recent_upload_shas:
                try:
                    delete_shard_path(entry)
                except Exception:
                    log.exception("failed to drop orphaned duplicate pending %s", entry)
                continue
            try:
                arrs, meta_dict = load_shard_arrays(entry)
            except Exception:
                log.exception("failed to load pending shard %s; skipping", entry)
                continue
            try:
                samples = arrays_to_samples(arrs)
            except Exception:
                log.exception("failed to materialize pending shard %s; skipping", entry)
                continue
            model_sha = str(meta_dict.get("model_sha256") or "")
            if not model_sha:
                # Without a model_sha we cannot key the accumulator the same
                # way as the live upload path. Fall back to the shard's
                # filename prefix so re-seeding is still deterministic.
                model_sha = entry.stem
            try:
                mtime = float(entry.stat().st_mtime)
            except OSError:
                mtime = float(time.time())
            acc_key = (trial_key, model_sha, _upload_history_acc_key(meta_dict))
            acc = upload_accumulators.get(acc_key)
            if acc is None:
                acc = _BufferedUploadAccumulator(
                    trial_id=trial_key,
                    model_sha256=model_sha,
                    created_at_unix=mtime,
                    last_update_unix=mtime,
                )
                upload_accumulators[acc_key] = acc
            acc.add_upload(samples=samples, meta=meta_dict, now_unix=mtime)
            acc.pending_paths.append(entry)
            if upload_sha is not None:
                recent_upload_shas[(trial_key, upload_sha)] = mtime
            recovered += 1
        return recovered

    def _recover_pending_uploads() -> None:
        # Default (no-trial) inbox.
        try:
            _recover_in_flight_dirs(
                in_flight_root=inbox / _IN_FLIGHT_DIR_NAME,
                compacted_dir=inbox / "_compacted",
                trial_key=None,
            )
        except Exception:
            log.exception("in-flight recovery (default inbox) failed")
        try:
            _scan_pending_dir(pending_dir=inbox / _PENDING_DIR_NAME, trial_key=None)
        except Exception:
            log.exception("pending-upload recovery (default inbox) failed")
        # Per-trial inboxes.
        trials_dir = root / "trials"
        if trials_dir.is_dir():
            for trial_dir in sorted(trials_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                try:
                    trial_key = _normalize_trial_id(trial_dir.name)
                except HTTPException:
                    continue
                if trial_key is None:
                    continue
                trial_inbox = trial_dir / inbox_dir
                try:
                    _recover_in_flight_dirs(
                        in_flight_root=trial_inbox / _IN_FLIGHT_DIR_NAME,
                        compacted_dir=trial_inbox / "_compacted",
                        trial_key=trial_key,
                    )
                except Exception:
                    log.exception("in-flight recovery for trial %s failed", trial_key)
                try:
                    _scan_pending_dir(
                        pending_dir=trial_inbox / _PENDING_DIR_NAME,
                        trial_key=trial_key,
                    )
                except Exception:
                    log.exception("pending-upload recovery for trial %s failed", trial_key)

    _recover_pending_uploads()

    return app
