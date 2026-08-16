"""Persist in-flight selfplay games at a session teardown and resume them.

A ``recommended_worker`` change on a ``_RECO_RESTART_KEYS`` key sets
``_stop_selfplay``; ``play_batch`` returns and every in-flight game is dropped
on the floor. That costs the SF labels already bought for those plies (~698k
nodes EACH) and, worse, biases the completed-game sample: only SHORT games have
finished, so ``avg_plies_win``/``avg_plies_draw`` ramp back up from ~0 over ~9
iterations, draws (the longest games) arrive last, and the PID reads a winrate
that is +0.110 too high and tightens regret spuriously (audit C14/C14b).

What is persisted, and what is NOT
----------------------------------
The encoded planes are **not** persisted: ``x`` alone is 175x8x8 float32 =
44.8 KB per ply, ~665 MB for a full slot table. Everything that is a pure
function of the board and its history — ``x``, ``x_lc0_root``, ``relations``,
``legal_mask`` — is REPLAYED instead: we store the game's root FEN, the
opening's own move stack, and the full list of played action ids, and rebuild
the exact ``CBoard`` sequence the live session built.

That reconstruction is byte-exact because it is the *same* construction:
``CBoard.from_board(opening_board)`` followed by one ``push_index`` per played
action, which is literally what the live path does. The history planes are the
hazard here — this repo has twice produced bogus findings from boards that
LOOKED right but had no move stack (``Board.mirror()`` and ``copy(stack=False)``
both drop it, leaving 117/175 planes zero). Replaying real moves gives a real
stack; ``tests/test_selfplay_resume.py`` asserts ``np.array_equal`` on every
ply's ``x``/``x_lc0_root``/``relations``/``legal_mask`` rather than trusting it.

Everything that is NOT recomputable is persisted per ply: the MCTS visit
distribution, the WDL estimates, the diff-focus priorities, and above all the
``sf_*`` label fields, which are the expensive ones and can never be recomputed.

Fail-safe, never fail-silent: a missing, unreadable, version-mismatched or
inconsistent state file is logged with its reason and the slot starts a fresh
game. ``ResumeReport``/``SuspendReport`` carry the counts so the effect is
observable rather than assumed.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chess
import numpy as np

from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.moves import POLICY_SIZE, index_to_move
from chess_anti_engine.selfplay.state import SelfplayState, _NetRecord

if TYPE_CHECKING:
    from chess_anti_engine.encoding._lc0_ext import CBoard

_LOG = logging.getLogger("chess_anti_engine.selfplay.resume")

# Bump whenever the on-disk layout or the meaning of a stored field changes.
# A mismatch is a discard-with-reason, never a best-effort partial decode.
RESUME_FORMAT_VERSION = 1

# Suffix for a durable per-game state file. One file per game keeps a partial
# write (worker killed mid-suspend) contained to that single game: the npz
# CRC check fails and only that game is discarded.
RESUME_FILE_SUFFIX = ".game.npz"
# A file is renamed to this before being read, so two selfplay threads (or two
# worker processes sharing a work dir) can never resume the same game twice.
_CLAIM_SUFFIX = ".claimed"
_TMP_SUFFIX = ".tmp"

# A suspended game older than this is not resumed: it comes from a session
# boundary the worker never came back from, so its opponent difficulty and its
# net are from another era of the run. It also bounds the directory's size —
# nothing else deletes a file that is never claimed.
DEFAULT_MAX_AGE_S: float = 6 * 3600.0


class ResumeStateError(ValueError):
    """A persisted game failed validation and must not be resumed.

    ``reason`` is a short stable token for the discard counter; the message
    carries the human detail for the log line.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = str(reason)


@dataclass
class SuspendReport:
    """Outcome of one ``suspend_inflight_games`` call."""

    persisted: int = 0
    # Games we MEANT to persist and could not. Deliberately excludes empty
    # slots: `_finalize_completed_slots` recycles every finished slot just
    # before teardown, so at a real suspend most of the table is freshly-empty
    # and counting those here buries the two readings that matter
    # (serialize_failed / write_failed) under hundreds of `empty_slot`.
    skipped: int = 0
    # Empty slots, counted separately so the noise stays visible but apart.
    empty_slots: int = 0
    records: int = 0
    # Games whose most recent record still had an SF label query in flight. The
    # future dies with the session. On a CURRICULUM slot the label is re-bought:
    # the board has not advanced, so the next step's opponent-move future is
    # reused for the label (submit_async_sf_labels_from_curriculum_moves, gated
    # on the same "latest record is unlabelled" test). On a SELFPLAY slot it is
    # not — label queries are submitted for the record created in that same
    # step — so that one ply stays unlabelled. Counted either way; it is the
    # bound on what suspension costs in labels.
    games_with_label_refetch: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def note_empty(self) -> None:
        self.empty_slots += 1
        self.reasons["empty_slot"] = self.reasons.get("empty_slot", 0) + 1


@dataclass
class ResumeReport:
    """Outcome of one ``resume_inflight_games`` call."""

    resumed: int = 0
    discarded: int = 0
    # Files refused for a reason that is about OUR session's state, not a
    # defect of the file (see _PRESERVE_FILE_REASONS). They are put back on
    # disk for a later, matching session — refused, but not LOST, so they are
    # deliberately kept out of `discarded` (and out of the
    # `resume_discarded_games` counter it feeds).
    preserved: int = 0
    records: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.discarded += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def note_preserved(self, reason: str) -> None:
        self.preserved += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def _reason_str(reasons: dict[str, int]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "-"


# ── sparse / ragged packing ─────────────────────────────────────────────────
# Selection is by RAW BYTE PATTERN, not by ``!= 0``: a -0.0 has all-zero value
# but a nonzero sign bit, so a value-based filter would round-trip it to +0.0
# and break byte-exactness. Viewing as uint8 makes the criterion dtype-agnostic
# and exactly inverse to filling a zeroed buffer.
def _nonzero_rows(arr: np.ndarray) -> np.ndarray:
    """Indices of entries in a 1-D array whose raw bytes are not all zero."""
    raw = arr.reshape(arr.shape[0], -1).view(np.uint8)
    return np.flatnonzero(raw.any(axis=1))


def _pack_sparse(
    rows: Sequence[np.ndarray | None], *, width: int, dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Pack optional fixed-width 1-D arrays as (present, counts, idx, values)."""
    present = np.zeros(len(rows), dtype=np.uint8)
    counts = np.zeros(len(rows), dtype=np.int32)
    idx_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for r, row in enumerate(rows):
        if row is None:
            continue
        flat = np.asarray(row).reshape(-1)
        if flat.shape[0] != width or flat.dtype != dtype:
            raise ResumeStateError(
                "bad_sparse_row",
                f"row {r}: expected ({width},) {dtype}, got {flat.shape} {flat.dtype}",
            )
        nz = _nonzero_rows(flat)
        present[r] = 1
        counts[r] = nz.shape[0]
        idx_parts.append(nz.astype(np.int32, copy=False))
        val_parts.append(flat[nz])
    return {
        "present": present,
        "counts": counts,
        "idx": (
            np.concatenate(idx_parts) if idx_parts
            else np.zeros(0, dtype=np.int32)
        ),
        "values": (
            np.concatenate(val_parts) if val_parts else np.zeros(0, dtype=dtype)
        ),
    }


def _unpack_sparse(
    blocks: dict[str, np.ndarray], *, n: int, width: int, dtype: np.dtype,
) -> list[np.ndarray | None]:
    present = blocks["present"]
    counts = blocks["counts"]
    idx = blocks["idx"]
    values = blocks["values"]
    if present.shape[0] != n or counts.shape[0] != n:
        raise ResumeStateError(
            "record_count_mismatch", "sparse block length != record count",
        )
    if idx.shape[0] != values.shape[0] or int(counts.sum()) != idx.shape[0]:
        raise ResumeStateError(
            "sparse_payload_mismatch", "sparse counts != payload length",
        )
    out: list[np.ndarray | None] = []
    pos = 0
    for r in range(n):
        k = int(counts[r])
        if not present[r]:
            if k:
                raise ResumeStateError(
                    "sparse_payload_mismatch", "payload for an absent row",
                )
            out.append(None)
            continue
        row = np.zeros(width, dtype=dtype)
        sel = idx[pos:pos + k]
        if k and (int(sel.min()) < 0 or int(sel.max()) >= width):
            raise ResumeStateError("sparse_index_out_of_range", f"row {r}")
        row[sel] = values[pos:pos + k]
        pos += k
        out.append(row)
    return out


def _pack_ragged2d(
    rows: Sequence[np.ndarray | None], *, ncols: int, dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Pack optional (K, ncols) arrays with a per-row K."""
    present = np.zeros(len(rows), dtype=np.uint8)
    counts = np.zeros(len(rows), dtype=np.int32)
    parts: list[np.ndarray] = []
    for r, row in enumerate(rows):
        if row is None:
            continue
        arr = np.asarray(row)
        if arr.ndim != 2 or arr.shape[1] != ncols or arr.dtype != dtype:
            raise ResumeStateError(
                "bad_ragged_row",
                f"row {r}: expected (K, {ncols}) {dtype}, got {arr.shape} {arr.dtype}",
            )
        present[r] = 1
        counts[r] = arr.shape[0]
        parts.append(arr)
    return {
        "present": present,
        "counts": counts,
        "values": (
            np.concatenate(parts, axis=0) if parts
            else np.zeros((0, ncols), dtype=dtype)
        ),
    }


def _unpack_ragged2d(
    blocks: dict[str, np.ndarray], *, n: int, ncols: int, dtype: np.dtype,
) -> list[np.ndarray | None]:
    present = blocks["present"]
    counts = blocks["counts"]
    values = blocks["values"]
    if present.shape[0] != n or counts.shape[0] != n:
        raise ResumeStateError(
            "record_count_mismatch", "ragged block length != record count",
        )
    if values.ndim != 2 or values.shape[1] != ncols:
        raise ResumeStateError("bad_ragged_row", "wrong column count")
    if int(counts.sum()) != values.shape[0]:
        raise ResumeStateError(
            "ragged_payload_mismatch", "ragged counts != payload length",
        )
    out: list[np.ndarray | None] = []
    pos = 0
    for r in range(n):
        k = int(counts[r])
        if not present[r]:
            if k:
                raise ResumeStateError(
                    "ragged_payload_mismatch", "payload for an absent row",
                )
            out.append(None)
            continue
        out.append(np.array(values[pos:pos + k], dtype=dtype, copy=True))
        pos += k
    return out


def _opt_float(rows: Sequence[float | None]) -> tuple[np.ndarray, np.ndarray]:
    present = np.array([0 if v is None else 1 for v in rows], dtype=np.uint8)
    vals = np.array([0.0 if v is None else float(v) for v in rows], dtype=np.float64)
    return present, vals


def _opt_int(rows: Sequence[int | None]) -> tuple[np.ndarray, np.ndarray]:
    present = np.array([0 if v is None else 1 for v in rows], dtype=np.uint8)
    vals = np.array([0 if v is None else int(v) for v in rows], dtype=np.int64)
    return present, vals


def _opt_fixed(
    rows: Sequence[np.ndarray | None], *, width: int, dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    present = np.zeros(len(rows), dtype=np.uint8)
    vals = np.zeros((len(rows), width), dtype=dtype)
    for r, row in enumerate(rows):
        if row is None:
            continue
        arr = np.asarray(row).reshape(-1)
        if arr.shape[0] != width or arr.dtype != dtype:
            raise ResumeStateError(
                "bad_fixed_row",
                f"row {r}: expected ({width},) {dtype}, got {arr.shape} {arr.dtype}",
            )
        present[r] = 1
        vals[r] = arr
    return present, vals


# ── suspend ─────────────────────────────────────────────────────────────────

_SF_MULTIPV_COLS = 5
_SF_LABEL_META_LEN = 6


def _game_payload(state: SelfplayState, i: int) -> dict[str, Any]:
    """Serialize slot ``i``'s in-flight game into npz-ready arrays + meta."""
    records: list[_NetRecord] = list(state.samples_per_game[i])
    n = len(records)
    offsets = [getattr(rec, "move_offset", -1) for rec in records]
    if any(int(o) < 0 for o in offsets):
        raise ResumeStateError("no_move_offset", "a record carries no move_offset")
    pos_hashes = [int(getattr(rec, "pos_hash", 0)) for rec in records]
    if any(h == 0 for h in pos_hashes):
        raise ResumeStateError("no_pos_hash", "a record carries no position hash")

    starting_boards = state.starting_boards
    if starting_boards is None:
        raise ResumeStateError("no_starting_boards", "state has no starting_boards")
    start = starting_boards[i]

    meta: dict[str, Any] = {
        "format_version": RESUME_FORMAT_VERSION,
        "root_fen": start.root().fen(),
        "opening_moves": [m.uci() for m in start.move_stack],
        "moves": [int(m) for m in state.move_idx_history[i]],
        "n_records": n,
        "net_color": int(state.net_color_arr[i]),
        "is_selfplay": int(state.selfplay_arr[i]),
        "opening_source": str(state.opening_source_arr[i]),
        "sf_refute_opp_plies_left": (
            int(state.sf_refute_opp_plies_left[i])
            if i < len(state.sf_refute_opp_plies_left) else 0
        ),
        "tb_adj_roll": int(state.tb_adj_roll_arr[i]),
        "consecutive_low_winrate": int(state.consecutive_low_winrate[i]),
        "last_net_full": bool(state.last_net_full[i]),
        "force_full_next": bool(state.force_full_next[i]),
        "sf_label_escalations": (
            int(state.sf_label_escalations[i])
            if i < len(state.sf_label_escalations) else 0
        ),
        "gumbel_policy_diag": [rec.gumbel_policy_diag for rec in records],
        # Position identity of the LIVE board after the last stored move. The
        # per-record hashes below cover the plies a record sits on; this one
        # covers the tail after the last record, so a move list truncated or
        # extended at the end cannot pass.
        "final_pos_hash": int(state.cboards[i].zobrist_hash),
        # On the C play path a record's ply_index IS the CBoard ply, so the
        # replay can check it. The Python fallback counts move_stack entries
        # instead, which starts from the opening's own stack — not comparable.
        "has_c_ply": bool(state.has_c_ply),
    }

    arrays: dict[str, np.ndarray] = {
        "ply_index": np.array([int(r.ply_index) for r in records], dtype=np.int32),
        "move_offset": np.array([int(o) for o in offsets], dtype=np.int32),
        # uint64: CBoard's zobrist is a full 64-bit value.
        "pos_hash": np.array(pos_hashes, dtype=np.uint64),
        "pov_color": np.array([1 if r.pov_color else 0 for r in records], dtype=np.uint8),
        "has_policy": np.array([1 if r.has_policy else 0 for r in records], dtype=np.uint8),
        "is_sf_refute_opp": np.array(
            [1 if r.is_sf_refute_opp else 0 for r in records], dtype=np.uint8,
        ),
        "priority": np.array([float(r.priority) for r in records], dtype=np.float64),
        "sample_weight": np.array(
            [float(r.sample_weight) for r in records], dtype=np.float64,
        ),
        "keep_prob": np.array([float(r.keep_prob) for r in records], dtype=np.float64),
    }

    pres, vals = _opt_float([r.priority_policy_kl for r in records])
    arrays["priority_policy_kl_present"], arrays["priority_policy_kl"] = pres, vals
    pres, vals = _opt_float([r.priority_q_delta for r in records])
    arrays["priority_q_delta_present"], arrays["priority_q_delta"] = pres, vals
    pres, vals = _opt_float([r.sf_played_regret for r in records])
    arrays["sf_played_regret_present"], arrays["sf_played_regret"] = pres, vals
    pres, vals = _opt_int([r.sf_move_index for r in records])
    arrays["sf_move_index_present"], arrays["sf_move_index"] = pres, vals
    pres, vals = _opt_int([r.sf_played_move_index for r in records])
    arrays["sf_played_move_index_present"], arrays["sf_played_move_index"] = pres, vals
    pres, vals = _opt_int([r.sf_played_rank for r in records])
    arrays["sf_played_rank_present"], arrays["sf_played_rank"] = pres, vals

    f32 = np.dtype(np.float32)
    for name, rows in (
        ("net_wdl_est", [r.net_wdl_est for r in records]),
        ("search_wdl_est", [r.search_wdl_est for r in records]),
        ("sf_wdl", [r.sf_wdl for r in records]),
        ("sf_wdl_original", [r.sf_wdl_original for r in records]),
    ):
        pres, vals = _opt_fixed(rows, width=3, dtype=f32)
        arrays[f"{name}_present"], arrays[name] = pres, vals

    pres, vals = _opt_fixed(
        [r.sf_label_meta for r in records],
        width=_SF_LABEL_META_LEN, dtype=np.dtype(np.int32),
    )
    arrays["sf_label_meta_present"], arrays["sf_label_meta"] = pres, vals

    for name, rows, dtype in (
        ("policy_probs", [r.policy_probs for r in records], f32),
        ("sf_policy_target", [r.sf_policy_target for r in records], f32),
        ("sf_legal_mask", [r.sf_legal_mask for r in records], np.dtype(np.uint8)),
    ):
        for key, arr in _pack_sparse(rows, width=POLICY_SIZE, dtype=dtype).items():
            arrays[f"{name}_{key}"] = arr

    for key, arr in _pack_ragged2d(
        [r.sf_multipv_raw for r in records],
        ncols=_SF_MULTIPV_COLS, dtype=np.dtype(np.int16),
    ).items():
        arrays[f"sf_multipv_raw_{key}"] = arr

    return {"meta": meta, "arrays": arrays}


def _write_game(path: Path, payload: dict[str, Any], *, header: dict[str, Any]) -> None:
    """Atomically write one game file (tmp + rename, so a reader never sees a
    partial archive)."""
    meta = dict(payload["meta"])
    meta.update(header)
    arrays = dict(payload["arrays"])
    arrays["meta_json"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez(fh, **arrays)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def suspend_inflight_games(
    state: SelfplayState,
    *,
    out_dir: Path,
    compat_fingerprint: str,
    model_sha: str,
    model_step: int,
    trial_id: str = "",
    pending_label_slots: Sequence[int] = (),
) -> SuspendReport:
    """Write every in-flight game of ``state`` to ``out_dir``.

    A game is in flight when its slot is neither done nor finalized. Slots that
    have played no move and hold no record are indistinguishable from the fresh
    game ``SelfplayState.create`` would build, so they are skipped.

    ``model_sha``/``model_step`` are recorded but never gate the write: games
    already span model swaps in normal operation (``_check_model_update`` runs
    as ``play_batch``'s per-ply ``on_step`` and a completed game is stamped at
    COMPLETION), so a same-net guard would buy no correctness. The resume side
    logs when they differ.
    """
    report = SuspendReport()
    # A game persisted without a trial id can never come back:
    # should_resume_game refuses an empty id on BOTH sides. Writing it anyway
    # would produce hundreds of files per teardown that only the sweep ever
    # removes, and a feature that looks armed while doing nothing. Refuse
    # loudly instead.
    if not str(trial_id or ""):
        _LOG.warning(
            "selfplay resume: no trial id at session start; in-flight games are "
            "NOT persisted (a game with no trial id could never be resumed)",
        )
        for i in range(state.batch_size):
            if state.finalized_arr[i] or state.done_arr[i]:
                continue
            if not state.move_idx_history[i] and not state.samples_per_game[i]:
                report.note_empty()
                continue
            report.note("no_trial_id")
        return report
    out_dir.mkdir(parents=True, exist_ok=True)
    awaiting = {int(i) for i in pending_label_slots}
    header = {
        "model_sha": str(model_sha),
        "model_step": int(model_step),
        "compat_fingerprint": str(compat_fingerprint),
        "trial_id": str(trial_id or ""),
        "saved_at": float(time.time()),
    }
    for i in range(state.batch_size):
        if state.finalized_arr[i] or state.done_arr[i]:
            continue
        if not state.move_idx_history[i] and not state.samples_per_game[i]:
            report.note_empty()
            continue
        try:
            payload = _game_payload(state, i)
        except (ResumeStateError, ValueError, IndexError, AttributeError) as exc:
            _LOG.warning("selfplay resume: slot %d not persisted: %s", i, exc)
            report.note("serialize_failed")
            continue
        path = out_dir / f"{uuid.uuid4().hex}{RESUME_FILE_SUFFIX}"
        try:
            _write_game(path, payload, header=header)
        except Exception as exc:
            # Deliberately broad, and the module docstring's "a partial write is
            # contained to that single game" depends on it. `json.dumps(meta)`
            # has no `default=`, so a non-JSON value anywhere in the header (a
            # gumbel_policy_diag that stops being plain floats, say) raises
            # TypeError; np.savez can raise ValueError. Either would escape a
            # narrow `except OSError`, hit the caller's blanket handler and
            # abandon EVERY remaining in-flight game — including the ones
            # already serialized fine. One bad game must cost one game.
            _LOG.warning("selfplay resume: slot %d write failed: %r", i, exc)
            report.note("write_failed")
            continue
        report.persisted += 1
        report.records += int(payload["meta"]["n_records"])
        if i in awaiting:
            report.games_with_label_refetch += 1
    _LOG.info(
        "selfplay resume: suspended games=%d records=%d skipped=%d empty_slots=%d "
        "[%s] label_refetch=%d dir=%s",
        report.persisted, report.records, report.skipped, report.empty_slots,
        _reason_str(report.reasons), report.games_with_label_refetch, out_dir,
    )
    return report


# ── resume ──────────────────────────────────────────────────────────────────


def should_resume_game(
    meta: dict[str, Any],
    *,
    compat_fingerprint: str,
    trial_id: str = "",
    now_s: float | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    has_c_ply: bool | None = None,
) -> str | None:
    """THE resume-or-discard decision. Returns a discard reason, or None to resume.

    Every path that could keep a persisted game funnels through here, so the
    policy lives in one place: version, record-affecting config identity, trial
    identity, age, and structural sanity of the header. Decoding failures
    downstream are discards too, but they are failures of the FILE, not of
    policy.

    The record-affecting config must match exactly. Half a game's plies encoded
    under one ``input_history_encoding`` (or one label-shape flag) and half
    under another is a corrupt game, not a resumed one — and a restart-key
    change is precisely when that happens.

    The TRIAL must match too. A trial reassignment also tears the session down,
    and a game whose plies were produced under another trial's hyperparameters
    would be uploaded under this one's id. Model swaps within a trial are fine
    (games already span those); a different trial is not the same experiment.
    """
    version = meta.get("format_version")
    if version != RESUME_FORMAT_VERSION:
        return "version_mismatch"
    if str(meta.get("compat_fingerprint") or "") != str(compat_fingerprint):
        return "config_mismatch"
    # `ply_index` means CBoard.ply on the C play path but len(move_stack) on
    # the Python fallback — different origins for a seeded opening. A session
    # on the OTHER convention cannot validate the stored values (the replay's
    # ply check is keyed off the file's own flag), and finalize keys its sf_p0
    # one-ply shift off ply_index, so reinterpreting the index would silently
    # shift a teacher. Refuse instead. ``None`` skips the check (header-only
    # callers that have no session).
    if has_c_ply is not None and bool(meta.get("has_c_ply", False)) != bool(has_c_ply):
        return "ply_convention_mismatch"
    # An EMPTY id must refuse, not match. `"" == ""` would make the trial guard
    # inert exactly when we cannot tell whose data this is, and "no evidence of
    # a mismatch" is not "evidence of a match". The caller passes a snapshot
    # taken at SESSION START (see WorkerSession._resume_trial_id_active) —
    # reading it at suspend time returns the id of the trial we were just
    # reassigned TO, so games played under the old trial would sail through.
    saved_trial = str(meta.get("trial_id") or "")
    want_trial = str(trial_id or "")
    if not saved_trial or not want_trial:
        return "no_trial_id"
    if saved_trial != want_trial:
        return "trial_mismatch"
    if not isinstance(meta.get("moves"), list) or not isinstance(
        meta.get("opening_moves"), list,
    ):
        return "malformed_meta"
    if not isinstance(meta.get("root_fen"), str) or not meta.get("root_fen"):
        return "malformed_meta"
    # Written by every writer of this format version; its absence means the
    # header was tampered with or truncated, and it is the ONLY check that can
    # see a move list corrupted after the last record (see _replay_and_reencode).
    if not int(meta.get("final_pos_hash") or 0):
        return "malformed_meta"
    age_s = float(now_s if now_s is not None else time.time()) - float(
        meta.get("saved_at") or 0.0,
    )
    if max_age_s > 0 and age_s > max_age_s:
        # A game older than this is from a session boundary we already gave up
        # on (worker down, flag toggled off and back); its opponent difficulty
        # and net are from another era of the run.
        return "stale"
    return None


def _decode_meta(npz: Any) -> dict[str, Any]:
    raw = npz["meta_json"]
    meta = json.loads(bytes(np.asarray(raw, dtype=np.uint8)).decode("utf-8"))
    if not isinstance(meta, dict):
        raise ResumeStateError("malformed_meta", "meta is not an object")
    return meta


def _rebuild_opening_board(meta: dict[str, Any]) -> chess.Board:
    """Root FEN + the opening's own moves, replayed so the board carries a REAL
    move stack (the history planes are wrong without one)."""
    board = chess.Board(str(meta["root_fen"]))
    for tok in meta["opening_moves"]:
        move = chess.Move.from_uci(str(tok))
        if move not in board.legal_moves:
            raise ResumeStateError("illegal_opening_move", f"move {tok!r}")
        board.push(move)
    return board


@dataclass
class _DecodedGame:
    """One validated game, ready to be transplanted onto a slot."""

    meta: dict[str, Any]
    opening_board: chess.Board
    cboard: CBoard
    board: chess.Board
    records: list[_NetRecord]


def _record_fields(npz: Any, n: int) -> list[dict[str, Any]]:
    """Per-record field dicts, with every length/shape checked against ``n``."""
    def col(name: str, dtype: np.dtype) -> np.ndarray:
        arr = np.asarray(npz[name])
        if arr.shape[0] != n:
            raise ResumeStateError(
                "record_count_mismatch",
                f"{name} has {arr.shape[0]} rows, expected {n}",
            )
        return arr.astype(dtype, copy=False)

    f32 = np.dtype(np.float32)
    ply_index = col("ply_index", np.dtype(np.int32))
    move_offset = col("move_offset", np.dtype(np.int32))
    pos_hash = col("pos_hash", np.dtype(np.uint64))
    pov_color = col("pov_color", np.dtype(np.uint8))
    has_policy = col("has_policy", np.dtype(np.uint8))
    is_refute = col("is_sf_refute_opp", np.dtype(np.uint8))
    priority = col("priority", np.dtype(np.float64))
    sample_weight = col("sample_weight", np.dtype(np.float64))
    keep_prob = col("keep_prob", np.dtype(np.float64))

    def opt_scalar(name: str, dtype: np.dtype) -> list[Any]:
        pres = col(f"{name}_present", np.dtype(np.uint8))
        vals = col(name, dtype)
        return [None if not pres[r] else vals[r].item() for r in range(n)]

    def opt_fixed(name: str, width: int, dtype: np.dtype) -> list[np.ndarray | None]:
        pres = col(f"{name}_present", np.dtype(np.uint8))
        vals = np.asarray(npz[name])
        if vals.shape != (n, width):
            raise ResumeStateError(
                "record_count_mismatch",
                f"{name} has shape {vals.shape}, expected {(n, width)}",
            )
        vals = vals.astype(dtype, copy=False)
        return [
            None if not pres[r] else np.array(vals[r], dtype=dtype, copy=True)
            for r in range(n)
        ]

    def sparse(name: str, dtype: np.dtype) -> list[np.ndarray | None]:
        blocks = {
            key: np.asarray(npz[f"{name}_{key}"])
            for key in ("present", "counts", "idx", "values")
        }
        return _unpack_sparse(blocks, n=n, width=POLICY_SIZE, dtype=dtype)

    policy_probs = sparse("policy_probs", f32)
    sf_policy_target = sparse("sf_policy_target", f32)
    sf_legal_mask = sparse("sf_legal_mask", np.dtype(np.uint8))
    sf_multipv_raw = _unpack_ragged2d(
        {
            key: np.asarray(npz[f"sf_multipv_raw_{key}"])
            for key in ("present", "counts", "values")
        },
        n=n, ncols=_SF_MULTIPV_COLS, dtype=np.dtype(np.int16),
    )
    net_wdl = opt_fixed("net_wdl_est", 3, f32)
    search_wdl = opt_fixed("search_wdl_est", 3, f32)
    sf_wdl = opt_fixed("sf_wdl", 3, f32)
    sf_wdl_original = opt_fixed("sf_wdl_original", 3, f32)
    sf_label_meta = opt_fixed("sf_label_meta", _SF_LABEL_META_LEN, np.dtype(np.int32))
    priority_kl = opt_scalar("priority_policy_kl", np.dtype(np.float64))
    priority_qd = opt_scalar("priority_q_delta", np.dtype(np.float64))
    sf_played_regret = opt_scalar("sf_played_regret", np.dtype(np.float64))
    sf_move_index = opt_scalar("sf_move_index", np.dtype(np.int64))
    sf_played_move_index = opt_scalar("sf_played_move_index", np.dtype(np.int64))
    sf_played_rank = opt_scalar("sf_played_rank", np.dtype(np.int64))

    out: list[dict[str, Any]] = []
    for r in range(n):
        if policy_probs[r] is None or net_wdl[r] is None or search_wdl[r] is None:
            raise ResumeStateError("missing_required_field", f"record {r}")
        out.append({
            "ply_index": int(ply_index[r]),
            "move_offset": int(move_offset[r]),
            "pos_hash": int(pos_hash[r]),
            "pov_color": bool(pov_color[r]),
            "has_policy": bool(has_policy[r]),
            "is_sf_refute_opp": bool(is_refute[r]),
            "priority": float(priority[r]),
            "sample_weight": float(sample_weight[r]),
            "keep_prob": float(keep_prob[r]),
            "policy_probs": policy_probs[r],
            "net_wdl_est": net_wdl[r],
            "search_wdl_est": search_wdl[r],
            "sf_policy_target": sf_policy_target[r],
            "sf_legal_mask": sf_legal_mask[r],
            "sf_multipv_raw": sf_multipv_raw[r],
            "sf_wdl": sf_wdl[r],
            "sf_wdl_original": sf_wdl_original[r],
            "sf_label_meta": sf_label_meta[r],
            "priority_policy_kl": priority_kl[r],
            "priority_q_delta": priority_qd[r],
            "sf_played_regret": sf_played_regret[r],
            "sf_move_index": sf_move_index[r],
            "sf_played_move_index": sf_played_move_index[r],
            "sf_played_rank": sf_played_rank[r],
        })
    return out


def _validate_net_color(meta: dict[str, Any], fields: list[dict[str, Any]]) -> None:
    """Check the header's ``net_color`` against what the RECORDS imply.

    ``net_color`` is not derivable from the move list — but it IS derivable from
    the persisted records. On a CURRICULUM game (``is_selfplay == 0``) the net
    plays exactly one seat: every record that is not an SF-refute opponent row
    was taken on a net turn, so its ``pov_color`` (the side to move, already
    pinned to the replayed board by the pov check) IS the net's colour.

    This is not cosmetic. ``net_color`` decides which side ``state.net_idxs``
    lets the net move for the rest of the game, and finalize scores the game
    from that seat — a flipped value hands the PID a win it lost and writes the
    game's outcome keys from the wrong side, with no crash and no metric move.
    Corruption is caught by the zip CRC; this catches a header that is
    internally consistent but wrong.

    A SELFPLAY game is exempt: the net plays both seats, ``pov_color``
    alternates, and nothing in the records constrains the field.
    """
    if int(meta.get("is_selfplay", 0)):
        return
    net_turns = {
        bool(f["pov_color"]) for f in fields if not bool(f["is_sf_refute_opp"])
    }
    if not net_turns:
        return  # nothing to derive from (all rows are refute-opp rows)
    raw = meta.get("net_color")
    if raw is None:
        raise ResumeStateError("malformed_meta", "net_color missing")
    net_color = bool(int(raw))
    if len(net_turns) > 1:
        raise ResumeStateError(
            "net_color_mismatch",
            "a curriculum game's net records span BOTH colours",
        )
    if net_turns != {net_color}:
        raise ResumeStateError(
            "net_color_mismatch",
            f"header says net_color={int(net_color)} but every net record was "
            f"taken with {int(next(iter(net_turns)))} to move",
        )


def _legal_mask_for(cb: CBoard) -> np.ndarray:
    mask = np.zeros((POLICY_SIZE,), dtype=np.uint8)
    mask[np.asarray(cb.legal_move_indices(), dtype=np.int64)] = 1
    return mask


def _replay_and_reencode(
    state: SelfplayState, meta: dict[str, Any], fields: list[dict[str, Any]],
) -> _DecodedGame:
    """Rebuild the CBoard sequence and re-encode every record's inputs.

    Walks the exact construction the live session used —
    ``CBoard.from_board(opening)`` then one ``push_index`` per action — and
    encodes at each record's move offset. Raises ``ResumeStateError`` on any
    inconsistency: an out-of-range or non-monotone offset, an illegal action,
    or a side-to-move that disagrees with the stored record POV (which is what
    a dropped or extra move in the persisted list looks like).
    """
    from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard

    opening_board = _rebuild_opening_board(meta)
    moves = [int(m) for m in meta["moves"]]
    offsets = [int(f["move_offset"]) for f in fields]
    if any(b <= a for a, b in itertools.pairwise(offsets)):
        raise ResumeStateError(
            "offsets_not_increasing", "record move offsets are not strictly increasing",
        )
    if offsets and (offsets[0] < 0 or offsets[-1] > len(moves)):
        raise ResumeStateError(
            "offset_out_of_range",
            f"record offset {offsets[-1]} outside a {len(moves)}-move game",
        )
    # Needs only the records, but it belongs with the other per-record header
    # invariants (ply_index below) rather than in should_resume_game, which
    # decides POLICY from the header alone and never sees a record.
    _validate_net_color(meta, fields)

    game = state.game
    want_lc0_root_alt = bool(game.record_lc0_root_input) and not _uses_lc0_root(game)
    want_relations = bool(game.record_relations)
    check_ply = bool(meta.get("has_c_ply"))

    board = opening_board.copy()
    cb = _CBoard.from_board(opening_board)
    records: list[_NetRecord] = []
    by_offset = {off: idx for idx, off in enumerate(offsets)}

    for k in range(len(moves) + 1):
        r = by_offset.get(k)
        if r is not None:
            f = fields[r]
            # The decisive gate. Colour alone cannot catch a move list that
            # lost or gained a move (the side to move at an even offset is the
            # same either way); the position identity can.
            if int(cb.zobrist_hash) != int(f["pos_hash"]):
                raise ResumeStateError(
                    "position_mismatch",
                    f"record {r} at offset {k}: the replayed position is not the "
                    "one this record was taken at",
                )
            if check_ply and int(cb.ply) != int(f["ply_index"]):
                # ply_index is not derivable from the moves — finalize keys its
                # sf_p0 one-ply shift off it, so a wrong value silently shifts
                # a teacher rather than failing.
                raise ResumeStateError(
                    "ply_index_mismatch",
                    f"record {r}: stored ply {f['ply_index']} != replayed {cb.ply}",
                )
            if bool(cb.turn) != bool(f["pov_color"]):
                raise ResumeStateError(
                    "pov_mismatch",
                    f"record {r} at offset {k}: replayed side to move disagrees "
                    "with the stored POV — the persisted move list is not this game's",
                )
            records.append(_rebuild_record(state, cb, f,
                                           want_lc0_root_alt=want_lc0_root_alt,
                                           want_relations=want_relations))
        if k >= len(moves):
            break
        move = index_to_move(moves[k], board)
        if move not in board.legal_moves:
            raise ResumeStateError(
                "illegal_action", f"action {moves[k]} at move {k}",
            )
        board.push(move)
        cb.push_index(moves[k])

    if len(records) != len(fields):
        raise ResumeStateError(
            "unmatched_record", "not every record was matched to a position",
        )
    # Mandatory: should_resume_game has already rejected a header without it,
    # so this cannot be silently skipped.
    final_hash = int(meta.get("final_pos_hash") or 0)
    if int(cb.zobrist_hash) != final_hash:
        raise ResumeStateError(
            "final_position_mismatch",
            "replaying the move list does not reach the suspended position",
        )
    return _DecodedGame(
        meta=meta, opening_board=opening_board, cboard=cb,
        board=board, records=records,
    )


def _uses_lc0_root(game: Any) -> bool:
    from chess_anti_engine.encoding.lc0 import uses_lc0_root_history

    return bool(uses_lc0_root_history(game.input_history_encoding))


def _rebuild_record(
    state: SelfplayState,
    cb: CBoard,
    f: dict[str, Any],
    *,
    want_lc0_root_alt: bool,
    want_relations: bool,
) -> _NetRecord:
    """One ``_NetRecord`` with re-encoded inputs and restored stored fields."""
    from chess_anti_engine.encoding.lc0 import LC0_HISTORY_ROOT

    game = state.game
    # An SF-refute OPPONENT row is built by _emit_sf_refute_opp_record with
    # x_lc0_root=None and relations=None ALWAYS, whatever record_lc0_root_input
    # / record_relations say. Re-encoding them here would give a resumed refute
    # row columns its non-resumed twin does not have — the same row shape
    # differing by whether a restart happened. Match the live path exactly.
    is_refute_opp = bool(f["is_sf_refute_opp"])
    x = encode_cboard(
        cb,
        input_history_encoding=game.input_history_encoding,
        input_extra_features=game.input_extra_features,
    )
    x_lc0_root = (
        encode_cboard(
            cb,
            input_history_encoding=LC0_HISTORY_ROOT,
            input_extra_features=game.input_extra_features,
        )
        if (want_lc0_root_alt and not is_refute_opp) else None
    )
    rec = _NetRecord(
        x=x,
        policy_probs=f["policy_probs"],
        net_wdl_est=f["net_wdl_est"],
        search_wdl_est=f["search_wdl_est"],
        pov_color=chess.WHITE if f["pov_color"] else chess.BLACK,
        ply_index=int(f["ply_index"]),
        has_policy=bool(f["has_policy"]),
        priority=float(f["priority"]),
        sample_weight=float(f["sample_weight"]),
        keep_prob=float(f["keep_prob"]),
        legal_mask=_legal_mask_for(cb),
        sf_policy_target=f["sf_policy_target"],
        sf_move_index=f["sf_move_index"],
        sf_wdl=f["sf_wdl"],
        x_lc0_root=x_lc0_root,
        relations=(
            cb.compute_relations()
            if (want_relations and not is_refute_opp) else None
        ),
        priority_policy_kl=f["priority_policy_kl"],
        priority_q_delta=f["priority_q_delta"],
        gumbel_policy_diag=f.get("gumbel_policy_diag"),
        move_offset=int(f["move_offset"]),
        pos_hash=int(f["pos_hash"]),
    )
    rec.sf_wdl_original = f["sf_wdl_original"]
    rec.sf_multipv_raw = f["sf_multipv_raw"]
    rec.sf_label_meta = f["sf_label_meta"]
    rec.sf_played_move_index = f["sf_played_move_index"]
    rec.sf_played_rank = f["sf_played_rank"]
    rec.sf_played_regret = f["sf_played_regret"]
    rec.sf_legal_mask = f["sf_legal_mask"]
    rec.is_sf_refute_opp = bool(f["is_sf_refute_opp"])
    return rec


def decode_game(
    state: SelfplayState,
    path: Path,
    *,
    compat_fingerprint: str,
    trial_id: str = "",
    now_s: float | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> _DecodedGame:
    """Load + fully validate one persisted game. Raises ``ResumeStateError``."""
    with np.load(path, allow_pickle=False) as npz:
        meta = _decode_meta(npz)
        reason = should_resume_game(
            meta, compat_fingerprint=compat_fingerprint, trial_id=trial_id,
            now_s=now_s, max_age_s=max_age_s, has_c_ply=bool(state.has_c_ply),
        )
        if reason is not None:
            raise ResumeStateError(reason, f"file {path.name}")
        n = int(meta.get("n_records", -1))
        if n < 0:
            raise ResumeStateError("malformed_meta", "n_records missing")
        fields = _record_fields(npz, n)
    diags = meta.get("gumbel_policy_diag") or [None] * n
    if len(diags) != n:
        raise ResumeStateError(
            "record_count_mismatch", "gumbel diag count != record count",
        )
    for f, diag in zip(fields, diags, strict=True):
        f["gumbel_policy_diag"] = diag
    return _replay_and_reencode(state, meta, fields)


def _restore_into_slot(state: SelfplayState, i: int, game: _DecodedGame) -> None:
    """Transplant a decoded game onto slot ``i``, replacing its fresh game."""
    meta = game.meta
    state.boards[i] = (
        game.opening_board.copy() if state.has_c_ply else game.board
    )
    state.cboards[i] = game.cboard
    if state.starting_boards is not None:
        state.starting_boards[i] = game.opening_board.copy()
    state.starting_ply_arr[i] = game.opening_board.ply()
    state.move_idx_history[i] = [int(m) for m in meta["moves"]]
    state.samples_per_game[i] = list(game.records)
    state.opening_source_arr[i] = str(meta.get("opening_source", "resumed"))
    state.net_color_arr[i] = 1 if int(meta.get("net_color", 1)) else 0
    state.selfplay_arr[i] = 1 if int(meta.get("is_selfplay", 0)) else 0
    if i < len(state.sf_refute_opp_plies_left):
        state.sf_refute_opp_plies_left[i] = int(meta.get("sf_refute_opp_plies_left", 0))
    state.tb_adj_roll_arr[i] = 1 if int(meta.get("tb_adj_roll", 0)) else 0
    state.tb_result_arr[i] = None
    state.consecutive_low_winrate[i] = int(meta.get("consecutive_low_winrate", 0))
    state.last_net_full[i] = bool(meta.get("last_net_full", True))
    state.force_full_next[i] = bool(meta.get("force_full_next", False))
    if i < len(state.sf_label_escalations):
        state.sf_label_escalations[i] = int(meta.get("sf_label_escalations", 0))
    state.done_arr[i] = 0
    state.finalized_arr[i] = 0
    state.root_ids[i] = -1  # No tree carryover across a session.
    state.pending_sf_moves.pop(i, None)
    state.resumed_from_disk[i] = True


def _claim(path: Path) -> Path | None:
    """Atomically take ownership of a state file (rename), or None if lost."""
    claimed = path.with_name(path.name + _CLAIM_SUFFIX)
    try:
        path.rename(claimed)
    except OSError:
        return None
    return claimed


def _resumable_slots(state: SelfplayState) -> list[int]:
    """Slots a resumed game may replace, in preference order.

    A fresh game on a fenlist slot has ALREADY consumed a doled blind-spot seed
    (``SelfplayState.create`` drains ``fen_dole_queue`` at construction), and
    the seed line cannot be pushed back from here. Overwriting such a slot
    would silently destroy the seed — the exact "delivered but never played"
    failure the dole feed has hit before. Non-seeded slots cost nothing to
    replace, so they are the only targets; when they run out, the remaining
    state files stay on disk for the next session rather than being claimed
    and dropped.
    """
    return [
        i for i in range(state.batch_size)
        if not str(state.opening_source_arr[i]).startswith("fenlist")
    ]


# Refusal reasons that are about OUR session's state rather than a defect of
# the FILE. A leased worker between model swaps has an empty trial id
# (`_check_model_update` clears `leased_trial_id` on a sha change, so
# `_current_trial_id()` returns ""), a reassigned worker carries another
# trial's id, and a reverted restart-key change carries another fingerprint —
# in every case the game itself is intact and a later, matching session may
# still resume it. Unlinking here would make the feature DESTROY the previous
# session's games on exactly the leased workers it was built for, so these
# files are put back instead. Expiry stays bounded: `stale` (counted) and the
# sweep's 4x backstop remain the deleting paths.
_PRESERVE_FILE_REASONS = frozenset(
    {"no_trial_id", "trial_mismatch", "config_mismatch"},
)


def initial_resume_counts() -> dict[str, int]:
    """The worker's cross-session resume tally, zeroed.

    One definition instead of three literals. The worker built this dict inline
    and two tests built their own copies of it, so adding ``preserved`` broke
    both doubles with a ``KeyError`` -- a drift the type checker cannot see,
    because they are plain ``dict[str, int]``. A test double that silently
    lacks a key the production code writes is the same class of defect this
    module's counters exist to catch.
    """
    return {
        "suspended": 0, "suspend_skipped": 0, "resumed": 0, "discarded": 0,
        "preserved": 0,
    }

# A state file this far past ``max_age_s`` is garbage-collected without being
# read. The multiple matters: an expired GAME is meant to be discarded through
# should_resume_game (so it is counted with a reason), and only a directory
# nobody is draining should reach the backstop.
_SWEEP_AGE_MULTIPLE = 4.0


def count_unclaimed_resume_files(in_dir: Path) -> int:
    """How many suspended games are still sitting in ``in_dir``, unclaimed.

    ⚑ THE LOSS NO COUNTER IN THIS MODULE CAN SEE. ``resume_inflight_games``
    walks ``sorted(glob(...))`` and breaks at ``report.resumed >= len(slots)``:
    it is DEMAND-driven, bounded by the slots the restarting threads present.
    When suspend wrote more games than the new session ever asks for, the
    surplus is never claimed, never decoded, and therefore never reaches
    ``ResumeReport.note`` or ``note_preserved``. ``discarded`` counts files the
    resume EXAMINED and rejected; ``suspend_skipped`` counts games suspend
    failed to write. A file that nothing looked at is outside both, so both
    report a truthful zero.

    The stranded files are NOT picked up later. ``resume_inflight_games`` runs
    from ``on_state_ready``, which fires once per selfplay SESSION -- and a
    session is the worker's whole continuous run, not one shard. (Verified on
    the live arm-B worker: one ``_dispatch_selfplay_one_shard`` session, 32
    resume calls, all inside three seconds of worker start, then nothing for the
    next four hours while ~50 iterations of shards went out.) So the next resume
    attempt is the next worker START. By then the files are past
    ``DEFAULT_MAX_AGE_S`` (6h) and ``should_resume_game`` rejects them as stale;
    ``sweep_orphan_state_files`` deletes them at
    ``DEFAULT_MAX_AGE_S * _SWEEP_AGE_MULTIPLE`` (24h). Between 6h and 24h they
    are examined and counted ``stale`` rather than silently dropped.

    MEASURED (2026-08-14 arm-B pause/resume): suspend 3046 games across four
    workers, resume restored 3017, and worker_02's directory held exactly 29
    ``*.game.npz`` afterwards -- the whole gap, in one worker, with
    ``suspend_skipped=0`` and ``discarded=0`` everywhere. Still all 29 there
    3h50m later.

    ``.claimed`` and ``.tmp`` are excluded by the glob suffix: those belong to
    a resume or suspend that was interrupted, which is
    ``sweep_orphan_state_files``' business, not this counter's.

    ⚑ TWO THINGS THIS NUMBER IS NOT.

    It is not stranded-only: files rejected for a reason in
    ``_PRESERVE_FILE_REASONS`` are deliberately renamed BACK into ``in_dir`` and
    match this glob. A worker that preserved 3 and stranded 0 counts 3 here.
    Callers must report ``ResumeReport.preserved`` alongside it -- the count is
    only interpretable as a pair.

    It is not a settled reading. Callers run it per selfplay thread against a
    shared directory, and the sample is separated from its emission by a
    contended lock, so ANY single reading -- including the last one logged --
    can be stale in either direction. Treat a nonzero value as "go look at the
    directory", not as a measurement.
    """
    if not in_dir.is_dir():
        return 0
    return sum(1 for _ in in_dir.glob(f"*{RESUME_FILE_SUFFIX}"))


def sweep_orphan_state_files(
    in_dir: Path, *, now_s: float | None = None, max_age_s: float = DEFAULT_MAX_AGE_S,
) -> int:
    """Delete the debris of a killed suspend/resume, and long-dead state files.

    Nothing else removes a ``.tmp`` from a suspend killed mid-write or a
    ``.claimed`` from a resume killed between the rename and the unlink. Whole
    game files are swept only well past ``max_age_s``, as a bound on a
    directory nobody drains — the ordinary expiry path is
    ``should_resume_game`` returning ``stale``, which is counted.

    **Call this unconditionally at session start, NOT only when the resume flag
    is on.** The one case this backstop names — a directory nobody drains — is
    exactly the flag-off case: turning the flag off is the documented revert,
    and it disables the drain along with the suspend, so the last teardown's
    400-1024 files (~30-78 MB) would otherwise sit in ``selfplay_resume/``
    forever. Bounded residue rather than a growth leak, since flag-off also
    stops anything new being written, but a revert should not leave litter.
    """
    if max_age_s <= 0 or not in_dir.is_dir():
        return 0
    now_s = time.time() if now_s is None else float(now_s)
    removed = 0
    limits = (
        (f"*{RESUME_FILE_SUFFIX}{_TMP_SUFFIX}", max_age_s),
        (f"*{RESUME_FILE_SUFFIX}{_CLAIM_SUFFIX}", max_age_s),
        (f"*{RESUME_FILE_SUFFIX}", max_age_s * _SWEEP_AGE_MULTIPLE),
    )
    for pattern, limit in limits:
        for path in in_dir.glob(pattern):
            try:
                if now_s - path.stat().st_mtime <= limit:
                    continue
                path.unlink(missing_ok=True)
            except OSError:
                continue
            removed += 1
    if removed:
        _LOG.info("selfplay resume: swept %d expired/orphaned state files", removed)
    return removed


def resume_inflight_games(
    state: SelfplayState,
    *,
    in_dir: Path,
    compat_fingerprint: str,
    model_sha: str = "",
    model_step: int = 0,
    trial_id: str = "",
    max_games: int | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> ResumeReport:
    """Restore persisted games into ``state``'s fresh slots.

    Called from ``play_batch``'s ``on_state_ready`` hook, i.e. after
    ``SelfplayState.create`` has built a full table of fresh games, so every
    slot holds a replaceable game. Each file is claimed by rename before it is
    read, so concurrent selfplay threads never resume the same game twice.
    A resumed or genuinely defective file is deleted — a file that cannot be
    decoded once will not decode later either, and leaving it would retry the
    same failure at every restart. A file refused for a reason in
    ``_PRESERVE_FILE_REASONS`` (a fact about THIS session, not the file) is
    put back for a later, matching session instead.
    """
    report = ResumeReport()
    if not in_dir.is_dir():
        return report
    now_s = time.time()
    sweep_orphan_state_files(in_dir, now_s=now_s, max_age_s=max_age_s)
    slots = _resumable_slots(state)
    if max_games is not None:
        slots = slots[: max(0, int(max_games))]
    for path in sorted(in_dir.glob(f"*{RESUME_FILE_SUFFIX}")):
        if report.resumed >= len(slots):
            break
        claimed = _claim(path)
        if claimed is None:
            continue  # another thread took it
        keep_file = False
        try:
            game = decode_game(
                state, claimed, compat_fingerprint=compat_fingerprint,
                trial_id=trial_id, now_s=now_s, max_age_s=max_age_s,
            )
        except ResumeStateError as exc:
            if exc.reason in _PRESERVE_FILE_REASONS:
                # OUR state refused it; the file is fine. Put it back.
                keep_file = True
                _LOG.warning(
                    "selfplay resume: leaving %s for a matching session: %s",
                    path.name, exc,
                )
                report.note_preserved(exc.reason)
            else:
                _LOG.warning("selfplay resume: discarding %s: %s", path.name, exc)
                report.note(exc.reason)
            continue
        except Exception as exc:
            # Deliberately broad: a state file that cannot be decoded — bad
            # zip CRC from a half-written archive, a missing array, a numpy
            # version quirk — must cost exactly that one game. Abandoning it
            # is the pre-existing behaviour, so the fallback is never worse
            # than not having the feature.
            _LOG.warning("selfplay resume: unreadable %s: %r", path.name, exc)
            report.note("unreadable")
            continue
        finally:
            if keep_file:
                try:
                    claimed.rename(path)
                except OSError:
                    # Cannot un-claim (dir vanished, permissions): better to
                    # drop this one file than to leak a .claimed the sweep
                    # only collects a day later.
                    claimed.unlink(missing_ok=True)
            else:
                claimed.unlink(missing_ok=True)
        saved_sha = str(game.meta.get("model_sha") or "")
        if model_sha and saved_sha and saved_sha != model_sha:
            _LOG.info(
                "selfplay resume: game suspended under model %s step %s, "
                "resuming under %s step %d (games already span model swaps)",
                saved_sha[:8], game.meta.get("model_step"), model_sha[:8],
                int(model_step),
            )
        slot = slots[report.resumed]
        try:
            _restore_into_slot(state, slot, game)
        except Exception as exc:
            # A half-transplanted slot is the one outcome worse than an
            # abandoned game: it would play a resumed board against the fresh
            # game's records. Put the slot back to a clean fresh game.
            _LOG.warning(
                "selfplay resume: restore into slot %d failed (%r); recycling it",
                slot, exc,
            )
            state.recycle_slot(slot)
            report.note("restore_failed")
            continue
        report.resumed += 1
        report.records += len(game.records)
    # Park the failure where the success already goes. `resumed_inflight_games`
    # reaches result.json via the per-game counters; a discard has no game of
    # its own, so the next finalize drains this onto the same path (see
    # finalize._update_aggregate_stats). Without it the only evidence that a
    # restart LOST games is a worker log line nobody diffs.
    if report.discarded:
        pending = getattr(state, "pending_outcome_stats", None)
        if pending is not None:
            pending["resume_discarded_games"] = (
                int(pending.get("resume_discarded_games", 0)) + int(report.discarded)
            )
    _LOG.info(
        "selfplay resume: resumed games=%d records=%d discarded=%d preserved=%d "
        "[%s] target_slots=%d dir=%s",
        report.resumed, report.records, report.discarded, report.preserved,
        _reason_str(report.reasons), len(slots), in_dir,
    )
    return report


__all__ = [
    "RESUME_FILE_SUFFIX",
    "RESUME_FORMAT_VERSION",
    "ResumeReport",
    "ResumeStateError",
    "SuspendReport",
    "count_unclaimed_resume_files",
    "decode_game",
    "initial_resume_counts",
    "resume_inflight_games",
    "should_resume_game",
    "suspend_inflight_games",
    "sweep_orphan_state_files",
]
