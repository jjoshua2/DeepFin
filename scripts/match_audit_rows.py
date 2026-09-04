#!/usr/bin/env python3
"""Recover the ORIGINAL stored 175-plane input row for every frozen audit position.

This builds the index the `--input-encoding stored` ruler mode reads
(`scripts/value_regret.py`, `scripts/audit_targets.py`, `scripts/score_audit_v2.py`).

The frozen audit set stores only `key`/`fen`/`phase`/`source` per position — no
shard path, no row index, no move stack. But `data/audit_set_v1_README.md`
records the replay snapshot it was sampled from, that snapshot still exists on
disk, and the stored `x` rows in its v2_threats twin carry the REAL lc0 8-frame
history that the FEN-only audit scorer zero-fills.

So the join is by POSITION rather than by provenance: for every audit row, find
a stored shard row whose reconstructed side-to-move-canonical board has the same
`position_key`. That yields the exact production input, history included, for
the SAME frozen positions — a RULER change, not a set change.

Two stages so the ~1500-shard scan stays cheap:

  1. vectorised fingerprint over the 12 current-frame piece planes (packbits ->
     96 bytes), which is a superset of the key match;
  2. full `decode_board_from_planes` + `position_key` comparison on the
     fingerprint hits only.

The index is a BUILD ARTIFACT and is not checked in, exactly like the audit set
itself and the shallow-SF cache. It re-derives in ~1 minute from the named
snapshot; the report it writes alongside is the evidence that it did.

Verification written into `<out>.report.json` and printed. The 2026-08-02 build
over 1463 shards, 4000/4000 matched in 24 s:

  - `canonicalisation_current_frame_identical` / `..._castling_104_107_identical`
    — the recovered row's current-frame piece planes and castling planes are
    bit-identical to the FEN-only encoding, so side-to-move canonicalisation is
    preserved exactly. Both true; either being false is an exit code.
  - `legal_move_sets_identical` — true on 4000/4000. ⚑ READ THIS PRECISELY.
    The join already accepts a row only when `position_key(decode(row))` equals
    the audit row's `key`, and `position_key` (placement/turn/castling/ep)
    determines legal moves — so for a row that passed the join, this check is
    IMPLIED and cannot fail on the join's account. What it independently proves
    is that the audit set's own `key` agrees with its own `fen`, since the
    legal moves are compared against `chess.Board(fen)` while the join matched
    on `key`. It is a guard on AUDIT-SET SELF-CONSISTENCY — worth having,
    because a set whose `key` and `fen` disagreed would silently attach the
    wrong per-UCI labels — but it is NOT independent validation of the join.
    The join's own evidence is the canonicalisation pair below plus the
    duplicate-multiplicity and match counts.
  - the planes that are EXPECTED to differ, each reported as a row fraction
    rather than folded into one boolean: the colour flag (108, 0.508 — this IS
    audit-v2), current-frame repetition (12, 0.006), rule50 (109, 0.002), EP
    (110, 0.030, where the FEN round-trip dropped an EP square python-chess
    deems invalid for want of a capturer) and the v2_threats block
    (112..174, 0.013, downstream of those).
  - history-plane occupancy before vs after the fill (0.0 -> 0.713)
  - duplicate multiplicity (the same position can occur in several games;
    first match in shard-name order wins, deterministically — mean 1.010, max 3)

Usage:

    PYTHONPATH=. python3 scripts/match_audit_rows.py \\
        --audit-set data/audit_set_v1.jsonl \\
        --snapshot runs/parallel_candidate_replay_snapshots/\\
current_live_20260602_202037_v2threats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding
from chess_anti_engine.eval.audit import decode_board_from_planes, position_key
from chess_anti_engine.eval.audit_history import (
    STORED_EXTRA_FEATURES,
    STORED_HISTORY_ENCODING,
    STORED_PLANES,
    default_matched_rows_path,
)
from chess_anti_engine.replay.shard import load_shard_arrays

PIECE_PLANES = 12
_PIECE_SLOTS = ((0, chess.WHITE), (6, chess.BLACK))
# The 7 non-current lc0 history frames: 0% occupied FEN-only, ~71% occupied
# from a stored row. This block IS audit-v2.
_HISTORY_PLANES = slice(13, 104)
MATCHED_ROWS_REPORT_SCHEMA = "deepfin.matched_audit_rows.v1"


def default_matched_rows_report_path(matched_rows: str | Path) -> Path:
    """Provenance sidecar written for one matched-row NPZ."""
    path = Path(matched_rows)
    return path.with_suffix(path.suffix + ".report.json")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_artifact(path: Path) -> dict[str, Any]:
    """Content and filesystem identity of one immutable input/output file."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"matched-row provenance requires a regular file: {resolved}")
    file_stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(file_stat.st_size),
        "mtime_ns": int(file_stat.st_mtime_ns),
        "ctime_ns": int(file_stat.st_ctime_ns),
        "device": int(file_stat.st_dev),
        "inode": int(file_stat.st_ino),
        "sha256": _sha256_bytes(resolved.read_bytes()),
    }


def _artifact_content_identity(value: object) -> tuple[object, object, object]:
    if not isinstance(value, dict):
        return None, None, None
    return value.get("path"), value.get("size"), value.get("sha256")


def _git_file_at_commit(commit: str, relative_path: str) -> bytes | None:
    relative = PurePosixPath(relative_path)
    if (
        len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit.lower())
        or not relative_path
        or relative.is_absolute()
        or ":" in relative_path
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        return None
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative.as_posix()}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _builder_git_provenance() -> dict[str, Any]:
    """Bind the index builder to exact tracked bytes and a clean revision."""
    repo_root = Path(__file__).resolve().parents[1]
    relative_path = Path(__file__).resolve().relative_to(repo_root).as_posix()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    artifact = _file_artifact(Path(__file__))
    committed = _git_file_at_commit(revision, relative_path)
    return {
        "git_sha": revision,
        "git_dirty": dirty,
        "script": {
            **artifact,
            "repo_relative_path": relative_path,
            "matches_git_revision": bool(
                committed is not None
                and artifact["size"] == len(committed)
                and artifact["sha256"] == _sha256_bytes(committed)
            ),
        },
    }


def snapshot_inventory(snapshot: str | Path) -> dict[str, Any]:
    """Deterministic identity inventory of every replay shard and its files.

    The inventory intentionally hashes filesystem identities, not an additional
    full copy of every replay byte.  The producer separately reads and verifies
    every candidate row needed to prove the selected positions' origin and
    duplicate-game identity.  Two inventory passes make concurrent replacement
    visible without turning a minute-long row scan into a multi-hundred-GB hash.
    """
    root = Path(snapshot).expanduser().resolve()
    shards = sorted(root.glob("*.zarr"), key=lambda path: path.name)
    if not shards:
        raise SystemExit(f"no *.zarr shards under {root}")
    shard_rows: list[dict[str, Any]] = []
    for shard in shards:
        shard_stat = shard.stat()
        entries: list[list[object]] = []
        total_bytes = 0
        for child in sorted(
            (path for path in shard.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(shard).as_posix(),
        ):
            child_stat = child.stat()
            total_bytes += int(child_stat.st_size)
            entries.append([
                child.relative_to(shard).as_posix(),
                int(child_stat.st_size),
                int(child_stat.st_mtime_ns),
                int(child_stat.st_ctime_ns),
                int(child_stat.st_dev),
                int(child_stat.st_ino),
            ])
        entries_document = json.dumps(
            entries, ensure_ascii=True, separators=(",", ":"),
        ).encode("utf-8")
        shard_rows.append({
            "name": shard.name,
            "device": int(shard_stat.st_dev),
            "inode": int(shard_stat.st_ino),
            "mtime_ns": int(shard_stat.st_mtime_ns),
            "ctime_ns": int(shard_stat.st_ctime_ns),
            "file_count": len(entries),
            "total_bytes": total_bytes,
            "entries_identity_sha256": _sha256_bytes(entries_document),
        })
    document = json.dumps(
        shard_rows, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "path": str(root),
        "shard_count": len(shard_rows),
        "shards": shard_rows,
        "inventory_sha256": _sha256_bytes(document),
    }


def board_fingerprint(board: chess.Board) -> bytes:
    """96-byte packbits fingerprint of the 12 current-frame piece planes.

    Encoder layout, us/them order. Audit boards are side-to-move canonical
    (always white to move, `decode_board_from_planes`), so "us" is white and no
    POV flip is applied here — `require_canonical` enforces that precondition
    rather than leaving it implicit.
    """
    planes = np.zeros((PIECE_PLANES, 64), dtype=np.uint8)
    for slot, color in _PIECE_SLOTS:
        for i, pt in enumerate(chess.PIECE_TYPES):
            for sq in board.pieces(pt, color):
                planes[slot + i, chess.square_rank(sq) * 8 + chess.square_file(sq)] = 1
    return np.packbits(planes.reshape(-1)).tobytes()


def shard_fingerprints(x12: np.ndarray) -> list[bytes]:
    """The same fingerprint for a whole shard; `x12` is (n, 12, 8, 8)."""
    bits = (x12 > 0.5).astype(np.uint8).reshape(x12.shape[0], -1)
    return [row.tobytes() for row in np.packbits(bits, axis=1)]


def require_canonical(boards: list[chess.Board]) -> None:
    """The fingerprint join assumes white-to-move canonical audit boards.

    A black-to-move audit row would fingerprint under the un-flipped layout and
    simply never match, which would show up as a silent shortfall in the match
    count rather than as an error. Refuse instead.
    """
    bad = [i for i, b in enumerate(boards) if b.turn != chess.WHITE]
    if bad:
        raise SystemExit(
            f"{len(bad)} audit positions are not side-to-move canonical "
            f"(first at index {bad[0]}); the position-key join assumes the "
            "white-to-move canonical form build_audit_set.py writes"
        )


def _shard_layout(group: dict[str, Any]) -> tuple[str, int] | None:
    """(history encoding, plane count) of a shard, or None when it has no `x`."""
    if "x" not in group:
        return None
    arr = group["x"]
    hist = "legacy"
    if "_input_history_encoding" in group:
        raw = np.asarray(group["_input_history_encoding"])
        hist = normalize_lc0_history_encoding(str(raw.reshape(-1)[0]))
    return hist, int(arr.shape[1])


def _game_ids_and_presence(
    group: dict[str, Any], *, source: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return source game ids and the rows on which they are meaningful.

    Replay's optional-field loader can expose a dense, zero-filled ``game_id``
    array while ``has_game_id`` is false. The value array alone therefore
    cannot establish provenance. Legacy shards that predate presence masks are
    different: they only stored ``game_id`` when that column was real, so keep
    that layout readable and treat its rows as present.
    """
    if "game_id" not in group:
        return None, None
    game_ids = np.asarray(group["game_id"][:]).reshape(-1)
    if "has_game_id" not in group:
        return game_ids, np.ones(game_ids.shape, dtype=bool)
    raw_presence = np.asarray(group["has_game_id"][:]).reshape(-1)
    if raw_presence.shape != game_ids.shape:
        raise ValueError(
            f"{source}: has_game_id shape {raw_presence.shape} does not match "
            f"game_id shape {game_ids.shape}"
        )
    if np.any((raw_presence != 0) & (raw_presence != 1)):
        raise ValueError(f"{source}: has_game_id is not binary")
    return game_ids, raw_presence.astype(bool, copy=False)


def _source_game_cluster_is_ambiguous(
    *,
    existing_has_game: bool,
    existing_game: int,
    candidate_has_game: bool,
    candidate_game: int,
) -> bool:
    """Whether two copies of one position disagree on source-game identity.

    Shard identity is deliberately absent: fixed-size replay flushing can split
    one completed game's rows across adjacent shards.
    """
    return (
        not existing_has_game
        or not candidate_has_game
        or candidate_game != existing_game
    )


def _strict_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one immutable report and reject duplicate JSON keys."""
    before = _file_artifact(path)

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SystemExit(f"matched-row report has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read matched-row report {path}: {exc}") from exc
    after = _file_artifact(path)
    if before != after:
        raise SystemExit(f"matched-row report changed while being read: {path}")
    if not isinstance(payload, dict):
        raise SystemExit(f"matched-row report is not a JSON object: {path}")
    return payload, before


def _report_builder_is_bound(report: Mapping[str, Any]) -> bool:
    builder = report.get("builder")
    if not isinstance(builder, dict) or builder.get("git_dirty") is not False:
        return False
    revision = builder.get("git_sha")
    script = builder.get("script")
    if (
        not isinstance(revision, str)
        or not isinstance(script, dict)
        or script.get("repo_relative_path") != "scripts/match_audit_rows.py"
        or script.get("matches_git_revision") is not True
    ):
        return False
    committed = _git_file_at_commit(revision, "scripts/match_audit_rows.py")
    return bool(
        committed is not None
        and script.get("size") == len(committed)
        and script.get("sha256") == _sha256_bytes(committed)
    )


def _matched_array(data: Any, name: str, n: int) -> np.ndarray:
    if name not in data:
        raise SystemExit(f"matched-row index lacks decision-grade field {name!r}")
    value = np.asarray(data[name])
    if value.ndim != 1 or value.shape[0] != n:
        raise SystemExit(
            f"matched-row field {name!r} has shape {value.shape}; expected ({n},)"
        )
    return value


def _row_sha256(row: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(row, dtype=np.float16))
    return _sha256_bytes(value.tobytes())


def verify_selected_row_origins(
    *,
    audit_set: str | Path,
    matched_rows: str | Path,
    report_path: str | Path,
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate selected cluster identities by replay-snapshot readback.

    The report and NPZ are build artifacts, not signatures.  Their claims only
    become decision-grade after this function independently rescans the exact
    snapshot, reads every selected source row, and recomputes all occurrences'
    game identities.  That full scan is what prevents a fabricated set of
    unique NPZ game ids from narrowing a clustered interval.
    """
    audit_path = Path(audit_set).expanduser().resolve()
    matched_path = Path(matched_rows).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    report, report_artifact = _strict_report(report_file)
    if report.get("schema") != MATCHED_ROWS_REPORT_SCHEMA:
        raise SystemExit(
            f"matched-row report has schema {report.get('schema')!r}; rebuild it"
        )
    if report.get("complete") is not True or report.get("verification_passed") is not True:
        raise SystemExit("matched-row report is incomplete or failed its builder checks")
    actual_audit = _file_artifact(audit_path)
    actual_matched = _file_artifact(matched_path)
    if _artifact_content_identity(report.get("audit_set")) != (
        actual_audit["path"], actual_audit["size"], actual_audit["sha256"],
    ):
        raise SystemExit("matched-row report does not bind the consumed audit set")
    if _artifact_content_identity(report.get("output")) != (
        actual_matched["path"], actual_matched["size"], actual_matched["sha256"],
    ):
        raise SystemExit("matched-row report does not bind the consumed NPZ")
    if not _report_builder_is_bound(report):
        raise SystemExit("matched-row report builder is dirty or not bound to its Git revision")
    report_origins = report.get("row_origins")
    if not isinstance(report_origins, list) or not all(
        isinstance(row, dict) for row in report_origins
    ):
        raise SystemExit("matched-row report lacks raw row-origin evidence")
    report_origins_document = json.dumps(
        report_origins, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    if (
        report.get("row_origin_count") != len(report_origins)
        or report.get("row_origins_sha256") != _sha256_bytes(report_origins_document)
    ):
        raise SystemExit("matched-row report row-origin inventory is inconsistent")
    report_origins_by_key = {
        str(row.get("key")): row for row in report_origins
    }
    if len(report_origins_by_key) != len(report_origins):
        raise SystemExit("matched-row report repeats a row-origin key")

    reported_snapshot = report.get("snapshot_inventory")
    if not isinstance(reported_snapshot, dict):
        raise SystemExit("matched-row report lacks a replay snapshot inventory")
    snapshot_path = Path(str(reported_snapshot.get("path", ""))).expanduser().resolve()
    before_inventory = snapshot_inventory(snapshot_path)
    if before_inventory != reported_snapshot:
        raise SystemExit("replay snapshot identity differs from the matched-row report")

    selected_by_key: dict[str, chess.Board] = {}
    for record in selected:
        key = record.get("key")
        fen = record.get("fen")
        if not isinstance(key, str) or not key or not isinstance(fen, str):
            raise SystemExit("selected audit origin request lacks key/FEN")
        if key in selected_by_key:
            raise SystemExit(f"selected audit origin request repeats key {key!r}")
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise SystemExit(f"selected audit key {key!r} has invalid FEN") from exc
        if not board.turn or position_key(board) != key:
            raise SystemExit(f"selected audit key {key!r} disagrees with its canonical FEN")
        selected_by_key[key] = board

    with np.load(matched_path, allow_pickle=False) as data:
        if "key" not in data or "found" not in data or "x_stored" not in data:
            raise SystemExit("matched-row index lacks current decision-grade row fields")
        raw_keys = np.asarray(data["key"])
        if raw_keys.ndim != 1 or raw_keys.dtype.kind not in ("S", "U"):
            raise SystemExit("matched-row key field is not a one-dimensional string array")
        all_keys = [str(value) for value in raw_keys]
        n = len(all_keys)
        found = np.asarray(data["found"])
        if found.ndim != 1 or found.shape[0] != n or found.dtype != np.dtype(bool):
            raise SystemExit("matched-row found mask has the wrong shape")
        if len(set(all_keys)) != n:
            raise SystemExit("matched-row index contains duplicate audit keys")
        key_to_index = {key: index for index, key in enumerate(all_keys)}
        x_stored = np.asarray(data["x_stored"])
        if (
            x_stored.shape != (n, STORED_PLANES, 8, 8)
            or x_stored.dtype != np.dtype(np.float16)
        ):
            raise SystemExit("matched-row x_stored has the wrong shape")
        raw_src_shard = _matched_array(data, "src_shard", n)
        raw_src_row = _matched_array(data, "src_row", n)
        raw_game_id = _matched_array(data, "game_id", n)
        has_game_id = _matched_array(data, "has_game_id", n)
        ambiguity = _matched_array(data, "source_cluster_ambiguous", n)
        raw_dup_count = _matched_array(data, "dup_count", n)
        if raw_src_shard.dtype.kind not in ("S", "U"):
            raise SystemExit("matched-row src_shard is not a string array")
        if any(
            value.dtype != np.dtype(np.int64)
            for value in (raw_src_row, raw_game_id, raw_dup_count)
        ):
            raise SystemExit("matched-row row/game/duplicate claims are not int64")
        if any(
            value.dtype != np.dtype(np.uint8)
            for value in (has_game_id, ambiguity)
        ):
            raise SystemExit("matched-row presence/ambiguity claims are not uint8")
        src_shard = raw_src_shard.astype(str)
        src_row = raw_src_row.astype(np.int64, copy=False)
        game_id = raw_game_id.astype(np.int64, copy=False)
        dup_count = raw_dup_count.astype(np.int64, copy=False)
        for name, value in (
            ("has_game_id", has_game_id),
            ("source_cluster_ambiguous", ambiguity),
        ):
            if np.any((value != 0) & (value != 1)):
                raise SystemExit(f"matched-row {name} is not binary")
        def scalar_string(name: str) -> str:
            if name not in data:
                raise SystemExit(f"matched-row NPZ lacks {name!r}")
            raw = np.asarray(data[name])
            if raw.size != 1 or raw.dtype.kind not in ("S", "U"):
                raise SystemExit(f"matched-row NPZ {name!r} is not one string")
            return str(raw.reshape(-1)[0])

        matched_snapshot = scalar_string("snapshot")
        if Path(matched_snapshot).expanduser().resolve() != snapshot_path:
            raise SystemExit("matched-row NPZ and report name different replay snapshots")
        matched_audit = scalar_string("audit_set")
        if Path(matched_audit).expanduser().resolve() != audit_path:
            raise SystemExit("matched-row NPZ and report name different audit sets")
        if (
            scalar_string("input_history_encoding") != STORED_HISTORY_ENCODING
            or scalar_string("input_extra_features") != STORED_EXTRA_FEATURES
        ):
            raise SystemExit("matched-row NPZ does not declare the production stored layout")

        selected_indices: dict[str, int] = {}
        for key in selected_by_key:
            index = key_to_index.get(key)
            if index is None or not bool(found[index]):
                raise SystemExit(f"matched-row index has no recovered row for {key!r}")
            selected_indices[key] = index

        fps: dict[bytes, list[str]] = {}
        for key, board in selected_by_key.items():
            fps.setdefault(board_fingerprint(board), []).append(key)
        occurrences: dict[str, list[dict[str, Any]]] = {
            key: [] for key in selected_by_key
        }
        for shard in sorted(snapshot_path.glob("*.zarr"), key=lambda path: path.name):
            group, _meta = load_shard_arrays(shard, lazy=True)
            layout = _shard_layout(group)
            if layout != (STORED_HISTORY_ENCODING, STORED_PLANES):
                continue
            rows = group["x"]
            x12 = np.asarray(rows[:, :PIECE_PLANES], dtype=np.float32)
            candidates: list[tuple[int, str]] = []
            for row_index, fingerprint in enumerate(shard_fingerprints(x12)):
                candidates.extend(
                    (row_index, key) for key in fps.get(fingerprint, ())
                )
            if not candidates:
                continue
            game_values, game_presence = _game_ids_and_presence(group, source=shard)
            for row_index, key in candidates:
                stored = np.asarray(rows[row_index], dtype=np.float16)
                board = decode_board_from_planes(
                    np.asarray(stored, dtype=np.float32),
                    input_history_encoding=STORED_HISTORY_ENCODING,
                )
                if board is None or position_key(board) != key:
                    continue
                has_game = bool(
                    game_presence is not None and game_presence[row_index]
                )
                source_game = (
                    int(game_values[row_index])
                    if has_game and game_values is not None else -1
                )
                occurrences[key].append({
                    "shard": shard.name,
                    "row": int(row_index),
                    "position_key": key,
                    "stored_x_sha256": _row_sha256(stored),
                    "has_game_id": has_game,
                    "game_id": source_game if has_game else None,
                })

        proofs: list[dict[str, Any]] = []
        for key in sorted(selected_by_key):
            index = selected_indices[key]
            copies = occurrences[key]
            if not copies:
                raise SystemExit(f"replay snapshot no longer contains selected key {key!r}")
            first = copies[0]
            first_has_game = bool(first["has_game_id"])
            first_game = int(first["game_id"]) if first_has_game else -1
            recomputed_ambiguous = any(
                _source_game_cluster_is_ambiguous(
                    existing_has_game=first_has_game,
                    existing_game=first_game,
                    candidate_has_game=bool(copy["has_game_id"]),
                    candidate_game=(
                        int(copy["game_id"]) if copy["has_game_id"] else -1
                    ),
                )
                for copy in copies[1:]
            )
            mismatches: list[str] = []
            if str(src_shard[index]) != first["shard"]:
                mismatches.append("src_shard")
            if int(src_row[index]) != first["row"]:
                mismatches.append("src_row")
            if _row_sha256(x_stored[index]) != first["stored_x_sha256"]:
                mismatches.append("x_stored")
            if bool(has_game_id[index]) != first_has_game:
                mismatches.append("has_game_id")
            if first_has_game and int(game_id[index]) != first_game:
                mismatches.append("game_id")
            if bool(ambiguity[index]) != recomputed_ambiguous:
                mismatches.append("source_cluster_ambiguous")
            if int(dup_count[index]) != len(copies):
                mismatches.append("dup_count")
            expected_report_origin = {
                "key": key,
                "shard": str(src_shard[index]),
                "row": int(src_row[index]),
                "stored_x_sha256": _row_sha256(x_stored[index]),
                "has_game_id": bool(has_game_id[index]),
                "game_id": int(game_id[index]) if bool(has_game_id[index]) else None,
                "source_cluster_ambiguous": bool(ambiguity[index]),
                "duplicate_count": int(dup_count[index]),
            }
            if report_origins_by_key.get(key) != expected_report_origin:
                mismatches.append("report_row_origin")
            if mismatches:
                raise SystemExit(
                    f"matched-row claims disagree with replay readback for {key!r}: "
                    + ", ".join(mismatches)
                )
            cluster_unique = bool(
                first_has_game and first_game >= 0 and not recomputed_ambiguous
            )
            if not cluster_unique:
                raise SystemExit(
                    f"selected key {key!r} has no provably unique source-game cluster"
                )
            proofs.append({
                "key": key,
                "source_dir": str(snapshot_path),
                "selected_origin": first,
                "duplicate_count": len(copies),
                "source_cluster_ambiguous": recomputed_ambiguous,
                "source_cluster_unique": cluster_unique,
                "occurrences": copies,
            })

    after_inventory = snapshot_inventory(snapshot_path)
    if before_inventory != after_inventory:
        raise SystemExit("replay snapshot changed while selected origins were verified")
    return {
        "schema": MATCHED_ROWS_REPORT_SCHEMA,
        "passed": True,
        "report": report_artifact,
        "report_builder": report["builder"],
        "report_audit_set": report["audit_set"],
        "report_output": report["output"],
        "report_input_stability": report.get("input_stability"),
        "snapshot_inventory": before_inventory,
        "selected_position_count": len(proofs),
        "selected_position_keys_sha256": _sha256_bytes(json.dumps(
            sorted(selected_by_key), ensure_ascii=True, separators=(",", ":"),
        ).encode("utf-8")),
        "rows": proofs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="replay snapshot the audit set was sampled from (the v2_threats twin)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="output .npz (default: <audit-set>.matched_rows.npz, next to the set)",
    )
    args = ap.parse_args()

    args.audit_set = args.audit_set.expanduser().resolve()
    args.snapshot = args.snapshot.expanduser().resolve()
    out_path = (args.out or default_matched_rows_path(args.audit_set)).expanduser().resolve()
    report_path = default_matched_rows_report_path(out_path)
    if out_path.suffix != ".npz":
        raise SystemExit("--out must end in .npz")
    if (
        out_path == args.audit_set
        or report_path == args.audit_set
        or out_path.is_relative_to(args.snapshot)
        or report_path.is_relative_to(args.snapshot)
    ):
        raise SystemExit("matched-row outputs cannot replace an input or live inside the snapshot")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    initial_audit_artifact = _file_artifact(args.audit_set)
    initial_snapshot_inventory = snapshot_inventory(args.snapshot)
    builder = _builder_git_provenance()

    audit = [
        json.loads(line)
        for line in args.audit_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [str(rec["key"]) for rec in audit]
    boards = [chess.Board(str(rec["fen"])) for rec in audit]
    require_canonical(boards)

    fps: dict[bytes, list[int]] = {}
    for i, board in enumerate(boards):
        fps.setdefault(board_fingerprint(board), []).append(i)
    print(f"[audit] {len(audit)} rows, {len(fps)} distinct piece-plane fingerprints")

    # Deterministic scan order (shard name ascending); first exact key match wins.
    shards = sorted(args.snapshot.glob("*.zarr"))
    if not shards:
        raise SystemExit(f"no *.zarr shards under {args.snapshot}")
    print(f"[scan] {len(shards)} shards under {args.snapshot}")

    n = len(audit)
    found = np.zeros(n, dtype=bool)
    x_stored = np.zeros((n, STORED_PLANES, 8, 8), dtype=np.float16)
    src_shard = ["" for _ in range(n)]
    src_row = np.full(n, -1, dtype=np.int64)
    src_game = np.full(n, -1, dtype=np.int64)
    src_has_game = np.zeros(n, dtype=np.uint8)
    src_cluster_ambiguous = np.zeros(n, dtype=np.uint8)
    src_ply = np.full(n, -1, dtype=np.int64)
    src_selfplay = np.full(n, -1, dtype=np.int64)
    dup_count = np.zeros(n, dtype=np.int64)

    t0 = time.time()
    fp_hits = 0
    skipped_layout: dict[str, int] = {}
    for si, path in enumerate(shards):
        group, _meta = load_shard_arrays(path, lazy=True)
        layout = _shard_layout(group)
        if layout is None:
            skipped_layout["no-x"] = skipped_layout.get("no-x", 0) + 1
            continue
        hist, planes = layout
        # A shard written under a different history layout would decode and
        # splice into completely different planes; taking its rows would put
        # silently wrong bytes in the index. Skip and COUNT, so a snapshot that
        # is not uniformly production layout shows up in the report.
        if hist != STORED_HISTORY_ENCODING or planes != STORED_PLANES:
            tag = f"{hist}/{planes}p"
            skipped_layout[tag] = skipped_layout.get(tag, 0) + 1
            continue
        arr = group["x"]
        x12 = np.asarray(arr[:, :PIECE_PLANES], dtype=np.float32)
        cand: list[tuple[int, int]] = []
        for r, fp in enumerate(shard_fingerprints(x12)):
            cand.extend((r, ai) for ai in fps.get(fp, ()))
        if cand:
            fp_hits += len(cand)
            xfull = np.asarray(arr[:], dtype=np.float16)
            game_values, game_presence = _game_ids_and_presence(group, source=path)
            cols = {
                name: (
                    game_values if name == "game_id" else (
                        np.asarray(group[name][:]).reshape(-1)
                        if name in group else None
                    )
                )
                for name in ("game_id", "ply_index", "is_selfplay")
            }
            decoded: dict[int, str | None] = {}
            for r in sorted({r for r, _ in cand}):
                board = decode_board_from_planes(
                    np.asarray(xfull[r], dtype=np.float32),
                    input_history_encoding=hist,
                )
                decoded[r] = None if board is None else position_key(board)
            for r, ai in cand:
                if decoded[r] != keys[ai]:
                    continue
                dup_count[ai] += 1
                if found[ai]:
                    candidate_has_game = bool(
                        game_presence is not None and game_presence[r]
                    )
                    candidate_game = (
                        int(game_values[r])
                        if candidate_has_game and game_values is not None else -1
                    )
                    if _source_game_cluster_is_ambiguous(
                        existing_has_game=bool(src_has_game[ai]),
                        existing_game=int(src_game[ai]),
                        candidate_has_game=candidate_has_game,
                        candidate_game=candidate_game,
                    ):
                        src_cluster_ambiguous[ai] = 1
                    continue
                found[ai] = True
                x_stored[ai] = xfull[r]
                src_shard[ai] = path.name
                src_row[ai] = r
                for name, target in (
                    ("game_id", src_game),
                    ("ply_index", src_ply),
                    ("is_selfplay", src_selfplay),
                ):
                    col = cols[name]
                    if col is not None:
                        if name == "game_id" and (
                            game_presence is None or not bool(game_presence[r])
                        ):
                            continue
                        target[ai] = int(col[r])
                        if name == "game_id":
                            src_has_game[ai] = 1
        if (si + 1) % 200 == 0 or si + 1 == len(shards):
            print(
                f"[scan] {si + 1}/{len(shards)} shards, matched "
                f"{int(found.sum())}/{n} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    scan_seconds = time.time() - t0

    # ---- verification: canonicalisation preserved, history actually filled ----
    x_fen_only = np.stack([
        encode_cboard(
            CBoard.from_board(board),
            input_history_encoding=STORED_HISTORY_ENCODING,
            input_extra_features=STORED_EXTRA_FEATURES,
        ).astype(np.float16)
        for board in boards
    ])

    def occupancy(x: np.ndarray, planes: slice) -> float:
        nz = np.abs(np.asarray(x[found], dtype=np.float32)).max(axis=(2, 3)) > 1e-9
        return float(nz[:, planes].mean())

    def rows_differing(planes: slice) -> float:
        a = np.asarray(x_fen_only[found][:, planes], dtype=np.float32)
        b = np.asarray(x_stored[found][:, planes], dtype=np.float32)
        return float((a != b).any(axis=(1, 2, 3)).mean())

    # Guard on AUDIT-SET SELF-CONSISTENCY, not on the join. The join already
    # required position_key(decode(row)) == audit["key"], and position_key
    # determines legal moves, so this can only fire when the audit set's own
    # `key` and `fen` disagree — in which case the per-UCI labels would be
    # attached to a different position than the one scored. Cheap, and the
    # thing it catches is unrecoverable downstream. Two blocks are
    # EXPECTED to disagree and their disagreement is the point of audit-v2:
    # the colour flag (plane 108, wrong on ~51% of FEN-only rows) and the
    # repetition/history planes. A handful of rows also differ on EP (110):
    # decode_board_from_planes drops an EP square python-chess deems invalid
    # (no capturer), so the FEN round-trip lost a flag that cannot change move
    # generation. `legal_move_sets_identical` is what proves that.
    legal_mismatch: list[int] = []
    for i in np.flatnonzero(found):
        recovered = decode_board_from_planes(
            np.asarray(x_stored[i], dtype=np.float32),
            input_history_encoding=STORED_HISTORY_ENCODING,
        )
        if recovered is None or {m.uci() for m in recovered.legal_moves} != {
            m.uci() for m in boards[i].legal_moves
        }:
            legal_mismatch.append(int(i))

    per_plane_fen = (
        np.abs(np.asarray(x_fen_only[found], dtype=np.float32)).max(axis=(2, 3)) > 1e-9
    ).mean(0)
    per_plane_stored = (
        np.abs(np.asarray(x_stored[found], dtype=np.float32)).max(axis=(2, 3)) > 1e-9
    ).mean(0)
    row_origins = [
        {
            "key": keys[index],
            "shard": src_shard[index],
            "row": int(src_row[index]),
            "stored_x_sha256": _row_sha256(x_stored[index]),
            "has_game_id": bool(src_has_game[index]),
            "game_id": (
                int(src_game[index]) if bool(src_has_game[index]) else None
            ),
            "source_cluster_ambiguous": bool(src_cluster_ambiguous[index]),
            "duplicate_count": int(dup_count[index]),
        }
        for index in np.flatnonzero(found)
    ]
    row_origins_document = json.dumps(
        row_origins, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")

    report: dict[str, Any] = {
        "schema": MATCHED_ROWS_REPORT_SCHEMA,
        "complete": True,
        "builder": builder,
        "audit_set": initial_audit_artifact,
        "snapshot_inventory": initial_snapshot_inventory,
        "input_history_encoding": STORED_HISTORY_ENCODING,
        "input_extra_features": STORED_EXTRA_FEATURES,
        "audit_rows": n,
        "matched": int(found.sum()),
        "unmatched": int((~found).sum()),
        "shards_scanned": len(shards),
        "shards_skipped_wrong_layout": skipped_layout,
        "fingerprint_candidate_pairs": fp_hits,
        "duplicate_multiplicity_mean": float(dup_count[found].mean()) if found.any() else 0.0,
        "duplicate_multiplicity_max": int(dup_count.max()),
        "row_origin_count": len(row_origins),
        "row_origins_sha256": _sha256_bytes(row_origins_document),
        "row_origins": row_origins,
        # Each of these names exactly what it proves. The predecessor of this
        # block was a single `meta_planes_104_111_identical` boolean that read
        # FALSE on a sound join — because plane 108 is *supposed* to differ —
        # and so could neither pass nor fail anything.
        "canonicalisation_current_frame_identical": rows_differing(
            slice(0, PIECE_PLANES)) == 0.0,
        "canonicalisation_castling_104_107_identical": rows_differing(
            slice(104, 108)) == 0.0,
        "legal_move_sets_identical": not legal_mismatch,
        "legal_move_set_mismatch_rows": legal_mismatch[:16],
        "rows_differing_stm_flag_108": rows_differing(slice(108, 109)),
        "rows_differing_current_repetition_12": rows_differing(slice(12, 13)),
        "rows_differing_rule50_109": rows_differing(slice(109, 110)),
        "rows_differing_ep_110": rows_differing(slice(110, 111)),
        "rows_differing_extra_features_112_174": rows_differing(
            slice(112, STORED_PLANES)),
        "history_plane_nonzero_frac_fen_only": occupancy(x_fen_only, _HISTORY_PLANES),
        "history_plane_nonzero_frac_stored": occupancy(x_stored, _HISTORY_PLANES),
        "all_plane_nonzero_frac_fen_only": occupancy(x_fen_only, slice(None)),
        "all_plane_nonzero_frac_stored": occupancy(x_stored, slice(None)),
        "planes_zero_in_fen_only_nonzero_in_stored": [
            int(p) for p in range(STORED_PLANES)
            if per_plane_fen[p] < 1e-9 and per_plane_stored[p] > 1e-9
        ],
        "scan_seconds": scan_seconds,
    }

    np.savez_compressed(
        out_path,
        x_stored=x_stored, x_fen_only=x_fen_only, found=found,
        key=np.array(keys), phase=np.array([int(r["phase"]) for r in audit]),
        source=np.array([int(r["source"]) for r in audit]),
        src_shard=np.array(src_shard), src_row=src_row,
        game_id=src_game, has_game_id=src_has_game,
        source_cluster_ambiguous=src_cluster_ambiguous,
        ply_index=src_ply, is_selfplay=src_selfplay,
        dup_count=dup_count,
        per_plane_nonzero_fen_only=per_plane_fen,
        per_plane_nonzero_stored=per_plane_stored,
        input_history_encoding=np.array([STORED_HISTORY_ENCODING]),
        input_extra_features=np.array([STORED_EXTRA_FEATURES]),
        snapshot=np.array([str(args.snapshot)]),
        audit_set=np.array([str(args.audit_set)]),
    )
    final_audit_artifact = _file_artifact(args.audit_set)
    final_snapshot_inventory = snapshot_inventory(args.snapshot)
    final_builder = _builder_git_provenance()
    report.update({
        "output": _file_artifact(out_path),
        "input_stability": {
            "audit_set_unchanged": initial_audit_artifact == final_audit_artifact,
            "snapshot_unchanged": (
                initial_snapshot_inventory == final_snapshot_inventory
            ),
            "builder_checkout_unchanged": builder == final_builder,
        },
    })
    report["verification_passed"] = bool(
        report["canonicalisation_current_frame_identical"]
        and report["canonicalisation_castling_104_107_identical"]
        and report["legal_move_sets_identical"]
        and report["input_stability"]["audit_set_unchanged"]
        and report["input_stability"]["snapshot_unchanged"]
        and report["input_stability"]["builder_checkout_unchanged"]
        and not skipped_layout
        and builder["git_dirty"] is False
        and builder["script"]["matches_git_revision"] is True
    )
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"[match] index -> {out_path}\n[match] report -> {report_path}")
    # An audit set whose `key` and `fen` disagree would attach every per-UCI
    # label to the wrong position while still looking like a successful build,
    # so this is an exit code, not a printed warning.
    if legal_mismatch:
        raise SystemExit(
            f"{len(legal_mismatch)} matched rows generate a different legal-move "
            "set than their audit FEN — the audit set's own 'key' and 'fen' "
            "disagree on those rows, so the per-UCI deep-SF labels belong to a "
            "different position than the one being scored"
        )
    if not report["canonicalisation_current_frame_identical"]:
        raise SystemExit("recovered rows are not side-to-move canonical")


if __name__ == "__main__":
    main()
