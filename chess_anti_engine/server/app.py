from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import logging
import os
import re
import secrets
import socket
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


# One-shot rearm request written by the trainable only when opening a selfplay
# window for an iteration that the durable gate has ALREADY claimed (true
# mid-iter resume after workers died). Consumed inside the gate lock with claim
# so concurrent multi-worker polls cannot double-dole (see PR #209 review).
SEED_DOLE_REARM_FILENAME = "seed_dole_rearm.json"


_LOG = logging.getLogger("chess_anti_engine.server")
_LOCK_HOST = socket.gethostname()


class _LeaseAssignBusy(RuntimeError):
    """The lease-assign lock is held by a live holder and did not free up.

    Raised instead of stealing. The route turns it into a 503 + Retry-After:
    a worker retries its next poll for free, whereas two workers assigned the
    same trial slot is not recoverable at all. Failing CLOSED is the whole
    point of audit A17.
    """


class _LeaseAssignLock:
    """Cross-process mutual exclusion for lease assignment, with a STALENESS test.

    ⚑ THE OLD BODY DID NOT EXCLUDE, AND FAILED OPEN (audit A17). It was
    `while True` with no failure exit: past the deadline it unlinked the
    lock file every 50ms *unconditionally*, so a holder that was merely
    SLOW had its lock deleted out from under it. The next `O_EXCL` create
    then succeeded, both holders had `_held = True` and both ran the
    critical section; each unlinked on exit, so a third caller walked
    straight in. Reproduced deterministically -- see
    `tests/test_server_lease_assign_lock.py`.

    Three changes, and the first is the whole fix:

    * **A steal requires EVIDENCE.** The lock file carries the holder's pid
      and creation time. We take it only when the holder process is gone
      (`os.kill(pid, 0)` raises `ProcessLookupError`) or the file is older
      than `stale_after_s`. A slow-but-alive holder is never stolen from; a
      crashed one does not wedge the server until someone restarts it.
    * **The deadline FAILS the acquisition** instead of forcing it. Without
      staleness evidence we raise `_LeaseAssignBusy`, which the route turns
      into a 503 with `Retry-After` -- a worker retries a poll happily, and
      a wrong lease assignment is not recoverable at all.
    * **Release checks ownership.** `__exit__` unlinks only if the file
      still carries OUR token, so a holder that was legitimately stolen
      from (it really had crashed... or it was paused past
      `stale_after_s`) cannot delete its successor's lock on the way out.
      This is the half that turns a single stolen lock into a cascade.

    `stale_after_s` defaults to 10x `timeout_s`, i.e. a generous multiple of
    the expected section time: assignment is a few small file reads and a
    write. Every steal is logged with the evidence that justified it,
    because a steal is either a crash recovery or a bug and the log line is
    how anyone tells which.

    A18: this is a file lock used from SYNC routes (`lease_trial` is a
    `def`, run in Starlette's threadpool). Nothing here goes near the event
    loop, and nothing here should be made `async` -- the blocking retry
    sleep is correct precisely because it is not on the loop.
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout_s: float = 10.0,
        stale_after_s: float | None = None,
    ) -> None:
        self.path = path
        self.timeout_s = float(timeout_s)
        self.stale_after_s = (
            float(stale_after_s) if stale_after_s is not None else 10.0 * float(timeout_s)
        )
        self._held = False
        # Identifies THIS acquisition, not this process: two acquisitions in
        # one process must not be able to release each other.
        self._token = secrets.token_hex(8)

    def _read_holder(self) -> dict[str, Any]:
        """The lock file's contents, or {} if it has none we can use.

        A legacy pre-fix lock file held `f"{pid}\n"`, which is valid JSON and
        decodes to an int, so it lands in the `{}` branch below and is judged
        on file age. There is deliberately no `JSONDecodeError` special case
        for it: an earlier revision of this method had one and it was DEAD --
        the decode it claimed to rescue never fails.
        """
        try:
            raw = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _lock_age(self, holder: dict[str, Any], now: float) -> tuple[float, str] | None:
        """Seconds since the lock was taken, and where the clock came from.

        ⚑ THE AGE TEST IS THE ONLY BACKSTOP THE STALENESS CHECK HAS, so it must
        ALWAYS have a clock source. An earlier revision consulted `st_mtime`
        only when the holder dict was EMPTY, which left a hole: a NON-empty
        holder whose `created_at_unix` was missing, non-numeric or in the
        future had no usable age, so with a live-looking pid (a recycled one,
        say) it was never stealable AT ALL -- the reviewer aged such a file to
        10,000s with `stale_after_s = 1.0` and it stayed busy permanently.
        That is a total outage of the lease path, on precisely the crashed-
        holder case this class exists to recover from, and each blocked poll
        holds a threadpool token for the full timeout while it happens. The
        future-timestamp variant is reachable on WSL2 clock jumps.

        So: trust `created_at_unix` only when it is a finite number that is not
        in the future, and fall back to `st_mtime` in every other case,
        including a non-empty holder. Age is clamped non-negative -- a clock
        that ran backwards must read as "brand new", never as a negative age
        that some future comparison treats as enormous.
        """
        raw = holder.get("created_at_unix")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            stamp = float(raw)
            if math.isfinite(stamp) and stamp <= now:
                return max(0.0, now - stamp), "created_at_unix"
        try:
            return max(0.0, now - self.path.stat().st_mtime), "file mtime"
        except OSError:
            return None

    def _staleness_reason(self, now: float) -> str | None:
        """Why the existing lock may be taken, or None if it may not be."""
        holder = self._read_holder()
        aged = self._lock_age(holder, now)
        age_txt = "unknown" if aged is None else f"{aged[0]:.1f}s"
        pid = holder.get("pid")
        if isinstance(pid, int) and pid > 0 and holder.get("host") in (None, _LOCK_HOST):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return f"holder pid {pid} is gone (age {age_txt})"
            except PermissionError:
                pass  # alive, owned by another user
            except OSError:
                pass
        if aged is not None and aged[0] > self.stale_after_s:
            held_by = f"holder pid {pid}" if pid else "an unreadable holder"
            return (
                f"{held_by} did not release: lock age {aged[0]:.1f}s "
                f"(by {aged[1]}) > {self.stale_after_s:.1f}s"
            )
        return None

    def __enter__(self) -> _LeaseAssignLock:
        deadline = time.time() + self.timeout_s
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "pid": os.getpid(),
                        "host": _LOCK_HOST,
                        "token": self._token,
                        "created_at_unix": time.time(),
                    }))
                self._held = True
                return self
            except FileExistsError as exists_exc:
                now = time.time()
                reason = self._staleness_reason(now)
                if reason is not None:
                    _LOG.warning(
                        "stealing stale lease-assign lock %s: %s", self.path, reason,
                    )
                    with contextlib.suppress(OSError):
                        self.path.unlink(missing_ok=True)
                    continue
                if now >= deadline:
                    raise _LeaseAssignBusy(
                        f"lease assignment is busy: {self.path} held by "
                        f"{self._read_holder() or 'an unreadable holder'} and not "
                        f"stale after {self.timeout_s:.1f}s"
                    ) from exists_exc
                time.sleep(0.05)

    def close(self) -> None:
        """Release. For callers that entered manually and use `contextlib.closing`
        because they need to catch `_LeaseAssignBusy` around the acquisition."""
        self.__exit__(None, None, None)

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if not self._held:
            return
        self._held = False
        # Only OUR lock. If we were stolen from, the file now belongs to a
        # live successor and unlinking it would restart the cascade this
        # class exists to stop.
        if self._read_holder().get("token") != self._token:
            _LOG.warning(
                "lease-assign lock %s was taken from us before release; "
                "leaving the current holder's file alone", self.path,
            )
            return
        with contextlib.suppress(OSError):
            self.path.unlink(missing_ok=True)


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
    leave the incomplete iteration claimed on disk while workers are gone. The
    trainable writes ``seed_dole_rearm.json`` only when the gate already shows
    that iteration claimed; ``claim`` consumes that file **under the same lock**
    as the claim decision so concurrent polls still yield exactly one winner.
    """

    def __init__(self, state_path: Path | None = None) -> None:
        self._state_path = state_path
        # ⚑ CREATED LAZILY, IN THE LOOP THAT FIRST USES IT. An `asyncio.Lock`
        # binds to a running loop, and constructing it here binds it to
        # whatever loop happened to be running when the gate was made -- or to
        # none. That was survivable while `claim` never awaited inside the
        # `async with`, because an uncontended acquire returns without touching
        # the loop; adding the threadpool hops (A9) made the binding real and
        # broke every test that drives one gate from two loops
        # (`asyncio.Lock bound to a different event loop`). Rebuilding when the
        # running loop changes keeps the critical section exactly as wide as it
        # was: one loop's concurrent polls still contend on one lock, which is
        # the only case that has to be mutually exclusive.
        self._lock: asyncio.Lock | None = None
        self._lock_loop: Any = None
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

    def _consume_rearm_unlocked(
        self, publish_dir: Path | None, training_iteration: int,
    ) -> bool:
        """Consume rearm file; caller MUST hold ``self._lock``."""
        if publish_dir is None:
            return False
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
            return int(data.get("training_iteration", -1)) == int(training_iteration)
        except Exception:
            return False
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def _loop_lock(self) -> asyncio.Lock:
        """The lock, bound to the loop currently running.

        Rebuilt if the running loop changed since it was made. Two different
        loops never share a gate in production -- there is exactly one server
        loop -- so this only rebuilds at first use and in tests, and a rebuild
        cannot drop a concurrent holder, because a holder can only exist on the
        loop it was acquired in.
        """
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def claim(
        self,
        trial_key: str,
        training_iteration: int,
        *,
        publish_dir: Path | None = None,
        allow_rearm: bool = False,
    ) -> bool:
        """Claim the dole for ``(trial_key, training_iteration)``.

        When a matching rearm file is present in ``publish_dir`` (or
        ``allow_rearm=True`` for tests) AND this exact iteration is already
        claimed, roll the gate back one step so a single new winner can take
        the batch. Rearm consumption and the claim decision share one lock so
        concurrent multi-worker polls cannot double-dole.
        """
        # A9: this runs on `/v1/lease_trial`, which every worker polls, and
        # the body does rename + read_text + unlink + write_text + replace.
        # Those are blocking FS calls, and they were on the event-loop thread
        # inside an `async def`: the busiest route on the server stalled the
        # loop for a filesystem round-trip on every poll.
        #
        # ⚑ THE LOCK STAYS AN `asyncio.Lock`, AND IT STAYS OUT HERE. Both FS
        # steps are pushed into the threadpool INDIVIDUALLY, from inside the
        # `async with`, rather than moving the whole body into one thread
        # function. Moving the body would mean acquiring an `asyncio.Lock`
        # from a worker thread, which is not merely awkward -- `asyncio.Lock`
        # is not thread-safe and has no blocking acquire, so it cannot be
        # held across a thread boundary at all. Awaiting inside the `async
        # with` keeps the critical section exactly as wide as it was: the
        # rearm consumption and the claim decision are still one section, so
        # concurrent multi-worker polls still cannot double-dole. The only
        # change is which thread does the syscalls.
        # Imported here, not at module scope: this module guards its
        # fastapi imports inside `create_app` so it stays importable in the
        # lite `[worker]` install, and starlette arrives with fastapi. A
        # module-level import would silently take that property away.
        from starlette.concurrency import run_in_threadpool

        async with self._loop_lock():
            rearm = bool(allow_rearm)
            if publish_dir is not None:
                # File wins over the kwarg when both are set; under the lock so
                # rename + claim decision is one critical section.
                rearm = await run_in_threadpool(
                    self._consume_rearm_unlocked, publish_dir, training_iteration,
                )
            last = int(self._last_iter.get(trial_key, -1))
            if rearm and last == int(training_iteration):
                last = int(training_iteration) - 1
                self._last_iter[trial_key] = last
            if int(training_iteration) > last:
                self._last_iter[trial_key] = int(training_iteration)
                await run_in_threadpool(self._persist)
                return True
            return False


def consume_seed_dole_rearm(publish_dir: Path, training_iteration: int) -> bool:
    """Test helper: consume a rearm file outside the gate (prefer claim+publish_dir)."""
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
        return int(data.get("training_iteration", -1)) == int(training_iteration)
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
    "sf_d6_n",
    "diff_focus_records", "diff_focus_kept",
    "diff_focus_keep_limited", "diff_focus_sample_weight_limited",
    "gumbel_policy_diag_n",
    "gumbel_policy_argmax_is_candidate_sum", "gumbel_policy_argmax_is_action_sum",
    "gumbel_policy_legal_count_sum", "gumbel_policy_candidate_count_sum",
)

_AGGREGATE_FLOAT_FIELDS: tuple[str, ...] = (
    "sf_d6_sum",
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

# Per-shard IDENTITY values: not aggregable, and NOT safe to take from an
# arbitrary member of the merge group. Every one of these is part of the
# accumulator key (``_upload_identity_acc_key``), so a group is uniform in all
# of them by construction and the compacted shard can carry the value verbatim.
#
# ``opponent_wdl_regret_limit`` / ``sf_nodes`` are the opponent difficulty the
# games were played at. The worker already treats the pair as buffer identity
# (``worker_buffer._difficulty_matches`` flushes on a change) so a raw shard's
# value is exact rather than an average; the server now matches that, which is
# what makes the promotion gate's PID-confound leg readable at all.
#
# ``input_history_encoding`` / ``history_rep_fix`` are handled separately
# below: they predate this table and carry bespoke "absent means unknown" /
# "absent means off" semantics that the generic path would flatten.
_COMPACTION_IDENTITY_FIELDS: tuple[str, ...] = (
    "policy_encoding",
    "policy_size",
    "opponent_wdl_regret_limit",
    "sf_nodes",
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


# How the compactor treats each ``ShardMeta`` field when it merges N uploaded
# shards into one. EVERY field must appear exactly once; the module-level
# assertion below refuses to import otherwise.
#
# This table exists because the previous compactor hand-wrote 48 of the 56
# ShardMeta kwargs and silently defaulted the other 8. Two of the eight
# (``opponent_wdl_regret_limit``, ``sf_nodes``) are the promotion gate's only
# instrument for its own dominant confound, and because compaction is
# unconditional they were ``None`` on 100% of production shards — the gate's
# pre-registered KILL leg read an empty set for 68 consecutive iterations. A
# hand-written kwarg list cannot fail when a field is ABSENT from it, so the
# list is gone: the summed classes below drive the kwargs directly, and this
# table makes an unclassified field a hard import error rather than a silent
# default.
_SHARD_META_FIELD_KINDS: dict[str, str] = {
    # Summed across the merged shards.
    **dict.fromkeys(_AGGREGATE_COUNTER_FIELDS, "sum_int"),
    **dict.fromkeys(_AGGREGATE_FLOAT_FIELDS, "sum_float"),
    # Per-shard identity: in the accumulator key, carried verbatim.
    **dict.fromkeys(_COMPACTION_IDENTITY_FIELDS, "identity"),
    "input_history_encoding": "identity",
    "history_rep_fix": "identity",
    "model_sha256": "identity",
    # Identity too — in the key, so a merge group cannot mix policy widths —
    # but the value PERSISTED is canonicalized by the shard writer, which
    # derives encoding and width from the policy arrays themselves and raises
    # if the meta disagrees with them. Passing the accumulator's value in is
    # therefore a cross-check of the uploads' declaration against the merged
    # arrays, not the source of what lands on disk.
    "policy_encoding": "identity_writer_canonicalized",
    "policy_size": "identity_writer_canonicalized",
    # Extremum over the merged shards (only meaningful where records > 0).
    "diff_focus_priority_min": "extremum",
    "diff_focus_priority_max": "extremum",
    # Key-collapsed counters merged key-by-key.
    "outcome_stats": "merged_dict",
    # Determined by ``model_sha256``, which is in the key, so "last wins" is a
    # no-op for any group a worker can actually produce. Deliberately NOT an
    # identity assert: republishing identical weights under a new step is a
    # bookkeeping oddity, not data corruption, and must not reject an upload.
    "model_step": "last_wins",
    # Owned by the writer, not carried from the inputs. The compacted shard is
    # a NEW shard: it is serialized by this process, at this time, from this
    # many samples, on behalf of the trial it is being written for. The
    # contributing shards' values are either wrong for it (``username`` — a
    # compacted shard can merge several workers, so no single uploader owns it)
    # or would be a stale copy (``version``, ``generated_at_unix``,
    # ``positions``, ``run_id``).
    "version": "writer_owned",
    "username": "writer_owned",
    "generated_at_unix": "writer_owned",
    "positions": "writer_owned",
    "run_id": "writer_owned",
}


def _assert_shard_meta_fields_classified() -> None:
    """Fail loudly if ``ShardMeta`` grew a field the compactor does not handle.

    Import-time on purpose. The alternative — discovering it from a column of
    ``None`` in a downstream metric months later — is exactly how the
    promotion gate's confound instrument came to be dead for 68 iterations.
    """
    declared = {f.name for f in dataclasses.fields(ShardMeta)}
    classified = set(_SHARD_META_FIELD_KINDS)
    missing = sorted(declared - classified)
    extra = sorted(classified - declared)
    if missing or extra:
        raise RuntimeError(
            "server upload compactor is out of sync with ShardMeta: "
            f"unclassified fields={missing} stale entries={extra}. Add each new "
            "field to _SHARD_META_FIELD_KINDS with its aggregation semantics."
        )


_assert_shard_meta_fields_classified()


def _coerce_identity_value(name: str, raw: Any) -> Any:
    """Normalize one identity value out of an upload's meta dict.

    Shared by the accumulator key and the accumulator's uniformity assert so
    the guard cannot disagree with the criterion it is guarding. ``None``
    (field absent, e.g. a shard written before the field existed) is a value in
    its own right and is preserved: "no regret limit recorded" and "regret
    limit 0.0" are different opponents, not the same one written two ways.

    Raises ``ValueError``/``TypeError`` on an uncoercible value; callers on the
    upload path reject such a shard rather than letting it through untyped.
    """
    if raw is None:
        return None
    if name in ("policy_size", "sf_nodes"):
        return int(raw)
    if name == "opponent_wdl_regret_limit":
        return float(raw)
    return str(raw)


def _compaction_identity_error(meta: dict[str, Any]) -> str | None:
    """Reason to reject an upload whose identity fields cannot be typed."""
    for name in _COMPACTION_IDENTITY_FIELDS:
        try:
            _coerce_identity_value(name, meta.get(name))
        except (TypeError, ValueError):
            return (
                f"shard meta field {name} is not usable as compaction identity: "
                f"{_bounded_report_field(repr(meta.get(name)))}"
            )
    return None


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
    sf_d6_sum: float = 0.0
    sf_d6_n: int = 0
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
    # Per-shard identity values (``_COMPACTION_IDENTITY_FIELDS``), captured
    # from the first upload and asserted equal on every later one. ``None`` is
    # a legitimate captured value, so ``identity_captured`` — not a None check
    # — is what distinguishes "nothing absorbed yet" from "absorbed a None".
    identity_captured: bool = False
    policy_encoding: str | None = None
    policy_size: int | None = None
    opponent_wdl_regret_limit: float | None = None
    sf_nodes: int | None = None
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
        self._absorb_identity(meta)
        _merge_outcome_stats(self.outcome_stats, meta.get("outcome_stats"))
        self.last_update_unix = float(now_unix)

    def _absorb_identity(self, meta: dict[str, Any]) -> None:
        """Capture (first upload) or verify (later uploads) the identity fields.

        The mismatch raise is unreachable from the upload path: every field
        here is in ``_upload_identity_acc_key``, so two shards that disagree
        land in different accumulators. It is kept as the structural backstop
        for a future edit that adds a field to one side only — the failure it
        prevents is a compacted shard whose recorded difficulty belongs to
        only some of the games inside it, which is worse than a rejected
        upload because nothing downstream can detect it.
        """
        incoming = {
            name: _coerce_identity_value(name, meta.get(name))
            for name in _COMPACTION_IDENTITY_FIELDS
        }
        if not self.identity_captured:
            for name, value in incoming.items():
                setattr(self, name, value)
            self.identity_captured = True
            return
        for name, value in incoming.items():
            current = getattr(self, name)
            if current != value:
                raise ValueError(
                    f"cannot compact uploads with mixed {name} "
                    f"{current!r} and {value!r}"
                )


def _upload_identity_acc_key(meta: dict[str, Any]) -> str:
    """Third element of the compaction accumulator key: per-shard identity.

    Module level, not an app closure, so the invariant it encodes can be
    asserted directly by a test rather than inferred from the shards that come
    out the far end.

    ``history_rep_fix`` changes the encoded planes under the same history mode,
    so it is part of compaction identity (absent means off), and the rest of
    ``_COMPACTION_IDENTITY_FIELDS`` joins it here. Difficulty in particular
    MUST be in the key: the worker flushes its own upload buffer when
    difficulty changes, so a raw shard's value is exact rather than an average,
    and a server that merged two difficulties under one model sha would hand
    the promotion gate a number that is true of neither half of the games.
    Coerced through the same helper the accumulator's uniformity assert uses,
    so the key and the guard agree about whether two values are the same value
    — for every value a worker can send. The agreement is not total: this key
    compares by ``repr()`` while ``_absorb_identity`` compares by ``!=``, and
    the two disagree on NaN (same key, unequal values, so the assert would
    raise on the second upload) and on ``0.0`` vs ``-0.0`` (different keys,
    equal values, so the group splits harmlessly). Both are unreachable in
    production: ``distributed_runtime.py`` maps a non-finite or negative
    ``wdl_regret`` to ``None`` before it can reach a worker, so no shard can
    carry a NaN regret limit. Do not close that gap by relaxing the assert to
    a repr comparison — a NaN difficulty is a broken reading, and rejecting
    the upload is the correct outcome if one ever gets that far.
    """
    raw = meta.get("input_history_encoding")
    history = "<missing>" if raw is None else str(raw)
    identity = tuple(
        _coerce_identity_value(name, meta.get(name))
        for name in _COMPACTION_IDENTITY_FIELDS
    )
    return f"{history}|repfix={bool(meta.get('history_rep_fix') or False)}|{identity!r}"


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
    # Built from ``_SHARD_META_FIELD_KINDS``, not a hand-written kwarg list: a
    # ShardMeta field the compactor forgets is an import error, not a column of
    # None in production. See the table for the per-class semantics.
    fields_out: dict[str, Any] = {}
    for name in _AGGREGATE_COUNTER_FIELDS:
        fields_out[name] = int(getattr(acc, name))
    for name in _AGGREGATE_FLOAT_FIELDS:
        fields_out[name] = float(getattr(acc, name))
    for name in _COMPACTION_IDENTITY_FIELDS:
        fields_out[name] = getattr(acc, name)
    has_diff_focus = int(acc.diff_focus_records) > 0
    meta = ShardMeta(
        # writer_owned. ``version`` is deliberately left at the ShardMeta
        # default: the arrays below are re-serialized by THIS writer, so the
        # compacted shard's format version is the current one regardless of
        # what the contributing uploads were written by.
        username="server_compactor",
        run_id=acc.trial_id,
        generated_at_unix=int(now_unix),
        positions=int(acc.positions),
        # identity (bespoke absent-value semantics; see add_upload)
        model_sha256=str(acc.model_sha256) or None,
        input_history_encoding=acc.input_history_encoding,
        history_rep_fix=bool(acc.history_rep_fix or False),
        # last_wins
        model_step=acc.model_step,
        # extremum — undefined with no contributing records, and 0.0 rather
        # than +/-inf because inf does not survive the JSON meta round-trip.
        diff_focus_priority_min=float(acc.diff_focus_priority_min) if has_diff_focus else 0.0,
        diff_focus_priority_max=float(acc.diff_focus_priority_max) if has_diff_focus else 0.0,
        # merged_dict
        outcome_stats=dict(acc.outcome_stats),
        # sum_int / sum_float / identity
        **fields_out,
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
        from starlette.concurrency import run_in_threadpool
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
    # ⚑ THE EVENT LOOP WAS SERIALISING THESE FOR FREE, AND A5 TOOK THAT AWAY.
    # Three unlocked read-modify-write cycles run on the upload path -- the two
    # throughput stats files and `load_users`/`record_upload`/`save_users`.
    # While `_upload_shard_impl` ran them on the loop thread they could not
    # interleave: one thread, no `await` between the read and the write. A5
    # moved the tail into the threadpool, so concurrent uploads now read the
    # same file, mutate their own copy, and write it back over each other.
    # `atomic_write_text` makes each WRITE atomic, so nothing tears -- the file
    # just quietly ends up with fewer increments than there were uploads.
    # Measured by review on 8 concurrent uploads: base 8/8 every run, this
    # branch without this lock [5, 4, 8, 7, 6].
    #
    # It nests with nothing: the stats tail runs after `_accumulate_under_lock`
    # has returned, so `upload_lock` is already released and there is no
    # ordering question between the two.
    #
    # ⚑ METHOD NOTE, worth more than the fix: A18 records single-PROCESS as the
    # precondition for these cycles being safe. The real precondition is single
    # EXECUTION CONTEXT, and the loop was providing it invisibly. This change
    # kept the process count at one and still broke it. Anything that moves
    # work off the loop must first enumerate what the loop was accidentally
    # serialising.
    stats_write_lock = threading.Lock()
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
                trial_key, _model_sha, _identity_key = key
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
            for (acc_tid, acc_sha, _identity_key), acc in upload_accumulators.items():
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
        # Serialised against the other RMW cycles on this path; see
        # `stats_write_lock`.
        with stats_write_lock:
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
        # Serialised against the other RMW cycles on this path; see
        # `stats_write_lock`.
        with stats_write_lock:
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
        # Rearm file (if any) is consumed inside claim under the gate lock so
        # concurrent multi-worker polls cannot double-dole. Paused/arena/dole-off
        # polls return above and never reach claim — they cannot burn rearm.
        return await seed_dole_gate.claim(
            trial_key,
            training_iteration,
            publish_dir=_publish_root(trial_id),
        )

    async def _serve_manifest(
        trial_id: str | None,
        *,
        x_cae_worker_version: str | None,
        x_cae_protocol_version: str | None,
    ) -> Any:
        # A16: `/v1/manifest` is the most frequently polled route on the
        # server -- every worker hits it on every poll -- and
        # `_get_manifest_impl` reads the manifest json (and stats it) from
        # disk. On an `async def` that read is on the loop thread, so the
        # busiest route is also a per-poll stall. The file-serving GETs next
        # to it are already correct (`def` + Starlette's threadpool); this
        # brings the manifest to the same footing without changing the
        # response.
        manifest = await run_in_threadpool(
            _get_manifest_impl,
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
        try:
            lease_lock = _LeaseAssignLock(leases_root / ".assign.lock")
            lease_lock.__enter__()
        except _LeaseAssignBusy as exc:
            # 503 + Retry-After, not a 500: the server is healthy and the
            # worker should poll again shortly. A traceback here would read as
            # a server fault and, worse, the pre-A17 code would have silently
            # double-assigned instead of saying anything at all.
            log.warning("lease assignment busy, asking the worker to retry: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="lease assignment is busy; retry shortly",
                headers={"Retry-After": "1"},
            ) from exc
        with contextlib.closing(lease_lock):
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

        def _ensure_upload_dirs() -> None:
            inbox_root.mkdir(parents=True, exist_ok=True)
            quarantine_root.mkdir(parents=True, exist_ok=True)

        # The only blocking work that has to happen BEFORE the drain -- the
        # temp file lives under `inbox_root` -- so it gets its own hop rather
        # than being left on the loop as "cheap". `mkdir(exist_ok=True)` is
        # cheap on a warm local dir and is not cheap on a cold or networked
        # one, and the yardstick below has no room for either.
        await run_in_threadpool(_ensure_upload_dirs)

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

        def _finish_upload() -> Any:
            """Everything after the drain, on ONE thread (audit A5).

            ⚑ THE DRAIN IS THE ONLY PART THAT CAN STAY. `_upload_shard_impl`
            must `await file.read(...)`, so the streaming hash-and-write loop
            above is on the loop by necessity -- and it is the one blocking
            stretch that already yields, once per chunk. Everything AFTER it is
            straight-line blocking work: untar, validate, two `load_shard_arrays`
            passes, `zarr_root.replace`, six recursive `delete_shard_path`
            calls, the accumulate/flush section, `load_users`/`save_users`.
            #335 moved only the locked block and left ~20 such sites here,
            worth a measured 0.110s of loop stall per upload on a 1500-position
            shard (0.287s before #335) -- per upload, with no contention needed,
            scaling with shard size.

            One thread, not several: each `run_in_threadpool` hop is a
            round-trip through the loop and a chance to interleave, and the
            steps here are strictly sequential over the same temp files. The
            handoff is also the natural error boundary -- every `return
            {"stored": False, ...}` rejection and the extract/validate
            `try/except` live INSIDE this function, so the quarantine paths and
            the exception behaviour are byte-identical to the inline version;
            `run_in_threadpool` re-raises anything else in the coroutine
            exactly where it was raised before.

            `_accumulate_under_lock` is called DIRECTLY rather than via another
            `run_in_threadpool`: we are already off the loop, and hopping again
            would put the lock back under a second threadpool slot for no
            reason. The `threading.Lock` contention semantics merged in #335 are
            unchanged -- same lock, same block, still held on a worker thread.

            Nothing here touches the request object. `file` is fully drained by
            the time this runs (the assertion #335's review made explicitly),
            and the only request-derived values used are the plain strings and
            ints captured above.
            """

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
            # Identity values are keyed and asserted below; a value that cannot be
            # typed would either crash the keying (500 -> worker retry storm) or
            # have to be silently dropped, which is the defect this whole path
            # exists to make impossible. Reject the shard instead.
            identity_reason = _compaction_identity_error(meta)
            if identity_reason is not None:
                tmp.unlink(missing_ok=True)
                delete_shard_path(tmp_zarr)
                return {"stored": False, "rejected": True, "reason": identity_reason}

            positions = int(shard_arrs["x"].shape[0])
            tmp.unlink(missing_ok=True)
            upload_seen_key = (trial_key, sha)
            now_unix = time.time()
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

            def _accumulate_under_lock() -> bool:
                """The upload critical section. Runs in a THREAD, never on the loop.

                ⚑ ``upload_lock`` is a ``threading.Lock`` and this used to be taken
                inline in ``_upload_shard_impl``, which is an ``async def``. A
                blocking acquire in a coroutine does not yield -- it holds the one
                event-loop thread -- and the body below is not cheap: it calls
                ``arrays_to_samples`` on the whole shard and, on the flush path,
                ``_try_flush_and_pop``, which writes a compacted shard to disk.
                So for the length of every compaction the server was not merely
                slow, it was serving NOTHING: no lease, no health, no manifest, no
                publish, no arena upload. It presents as wedged.

                The lock stays a ``threading.Lock`` on purpose. The other two
                acquisitions (the stale-flush sweep and the queued-games count) are
                plain ``def``s that Starlette already runs in its threadpool, and a
                ``def`` cannot ``await`` an ``asyncio.Lock``; converting the lock
                would silently drop mutual exclusion between this path and those
                two. Running this block through ``run_in_threadpool`` instead puts
                the coroutine's acquisition on exactly the same footing as theirs.

                Nothing in here touches the event loop or request-scope state:
                ``file`` has already been fully drained to disk above, and the body
                works on plain locals (``meta``, ``shard_arrs``, ``pending_path``)
                plus the module-level dicts the lock exists to protect. The one
                nested acquisition, ``_invalidate_queued_games_cache`` ->
                ``queued_games_cache_lock``, keeps its existing order.

                Returns whether the upload was newly accumulated.
                """
                with upload_lock:
                    return _accumulate_locked()

            def _accumulate_locked() -> bool:
                stored_local = False
                if upload_seen_key not in recent_upload_shas:
                    model_sha = str(meta.get("model_sha256") or sha)
                    acc_key = (trial_key, model_sha, _upload_identity_acc_key(meta))
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
                    stored_local = True
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
                return stored_local

            stored = _accumulate_under_lock()

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
                with stats_write_lock:
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

        return await run_in_threadpool(_finish_upload)

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
            # A6: same defect class as A5 -- a blocking write on the loop
            # thread. Arena results are small, but the route is an
            # `async def` and the write is unbounded by anything except the
            # size guard above, so it stalls every other route for its
            # duration.
            await run_in_threadpool(out.write_bytes, body)

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

    def _quarantine_unloadable_pending(
        *, entry: Path, trial_key: str | None, exc: BaseException
    ) -> None:
        """Move a permanently-unloadable pending shard to ``quarantine/unloadable``.

        Best-effort: if the move itself fails we leave the shard in place and
        accept the retry, which is no worse than the old behavior. Writes the
        same ``.reason.txt`` sidecar as the upload-time quarantine path — these
        shards surface days later, so the reason must not depend on log
        retention.
        """
        qdir = _quarantine_root(trial_key) / "unloadable"
        try:
            qdir.mkdir(parents=True, exist_ok=True)
            dest = qdir / entry.name
            if dest.exists():
                dest = qdir / f"{entry.stem}_{secrets.token_hex(4)}{LOCAL_SHARD_SUFFIX}"
            entry.replace(dest)
        except Exception:
            log.exception("failed to quarantine unloadable pending shard %s", entry)
            return
        with contextlib.suppress(Exception):
            (dest.parent / f"{dest.name}.reason.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )

    def _is_transient_load_error(exc: BaseException) -> bool:
        """True when a pending-shard load failure may succeed on a later startup.

        Quarantining is only correct for failures that are a deterministic
        property of the shard's bytes. System-level errors (fd exhaustion at
        startup, EIO, a full disk, MemoryError while the trainer holds the box)
        say nothing about the shard, so retry-forever stays the right answer
        there — it costs one failed load per startup and loses no data.
        """
        return isinstance(exc, (OSError, MemoryError))

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
            except Exception as exc:
                # A shard that fails to load for a reason intrinsic to its bytes
                # will fail identically on every future startup — "skipping"
                # left corrupt shards (truncated zarr metadata from an
                # interrupted write) retried forever, 7 of them for up to 5 days
                # as of 2026-07-24. Move those out of the recovery path so the
                # failure is bounded, keeping the bytes for post-mortem rather
                # than deleting them. Transient system errors still just retry.
                if _is_transient_load_error(exc):
                    log.exception("failed to load pending shard %s; skipping", entry)
                    continue
                log.exception("failed to load pending shard %s; quarantining", entry)
                _quarantine_unloadable_pending(entry=entry, trial_key=trial_key, exc=exc)
                continue
            try:
                samples = arrays_to_samples(arrs)
            except Exception as exc:
                # Same argument as the load branch, and strictly more
                # deterministic: this runs on arrays that already loaded and
                # passed validate_arrays, so the environment is not in play.
                if _is_transient_load_error(exc):
                    log.exception("failed to materialize pending shard %s; skipping", entry)
                    continue
                log.exception("failed to materialize pending shard %s; quarantining", entry)
                _quarantine_unloadable_pending(entry=entry, trial_key=trial_key, exc=exc)
                continue
            identity_reason = _compaction_identity_error(meta_dict)
            if identity_reason is not None:
                # Same rule as the upload path: an untypable identity value is
                # a bad shard, not something to key around. Quarantine rather
                # than skip so it cannot be rescanned every restart.
                log.warning("pending shard %s has bad identity meta; quarantining: %s",
                            entry, identity_reason)
                _quarantine_unloadable_pending(
                    entry=entry, trial_key=trial_key, exc=ValueError(identity_reason),
                )
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
            acc_key = (trial_key, model_sha, _upload_identity_acc_key(meta_dict))
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
