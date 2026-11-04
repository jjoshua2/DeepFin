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

Validation precedes publication. Rejected builds write only a uniquely named
JSON diagnostic (with captured-input hashes and at most four exact offending
rows), never a consumer-loadable rejected NPZ. Valid builds use atomic NPZ
replacement; a unique validated-candidate report is saved before that commit.
`<out>.report.json` is a compatibility view updated afterwards, NOT an atomic
pair with the NPZ. Its candidate hash/build ID detects a stale view. Failures
after replacement may leave a valid new NPZ visible; no blind rollback occurs.
Old invalid indexes produced before this guard need an explicit rebuild.

Hashing binds decision inputs: all current-frame scans, ordered candidate indices
and their full rows, plus identity columns. It does not hash unused full rows or
claim whole-shard/file provenance. It reads the prior output/report and compressed
staged output once, without rereading shard files or freezing a changing corpus.
Rejected diagnostics reobserve final-output identities. All unique reports are
retained privately (0600), with no automatic cleanup; raw rejected rows are not
printed. Existing final mode bits are preserved from staging-file creation, but
atomic replacement does not preserve inode/owner/ACL/xattrs/hard-link identity.
Terminal output/report symlinks are explicitly refused, not followed/replaced;
use a trusted output directory without concurrent permission/link mutation.

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
    first match in shard-name order supplies the stored input, deterministically
    — mean 1.010, max 3)

The first matching row is NOT used as the bootstrap cluster identity. The
original manifest discarded row provenance, so for each audit position the
index records every source game in which it could have originated and takes
connected components of the audit-position/source-game bipartite graph. Two
rows that could be siblings therefore always share ``game_cluster_id``. This
is conservative when a position occurs in multiple games, but it cannot split
dependent rows merely because the deterministic history join chose different
first matches.

Usage:

    PYTHONPATH=. python3 scripts/match_audit_rows.py \\
        --audit-set data/audit_set_v1.jsonl \\
        --snapshot runs/parallel_candidate_replay_snapshots/\\
current_live_20260602_202037_v2threats
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import stat
import time
import uuid
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding
from chess_anti_engine.eval.audit import decode_board_from_planes, position_key
from chess_anti_engine.eval.audit_history import (
    GAME_CLUSTER_KIND,
    STORED_EXTRA_FEATURES,
    STORED_HISTORY_ENCODING,
    STORED_PLANES,
    default_matched_rows_path,
)
from chess_anti_engine.replay.shard import load_shard_arrays
from chess_anti_engine.utils import sha256_file
from chess_anti_engine.utils.atomic import atomic_write, atomic_write_text

PIECE_PLANES = 12
_PIECE_SLOTS = ((0, chess.WHITE), (6, chess.BLACK))
# The 7 non-current lc0 history frames: 0% occupied FEN-only, ~71% occupied
# from a stored row. This block IS audit-v2.
_HISTORY_PLANES = slice(13, 104)


GameKey = tuple[str, int]


def source_game_key(shard_path: Path, game_id: int) -> GameKey:
    """Corpus-wide identity for a conversion-local game id.

    Snapshot directories may be flat symlink farms over several independent
    ``lc0_data_to_rows`` outputs, each of which starts ``game_id`` at zero.
    Resolve the shard first so equal integers from different conversion source
    directories cannot create a false bootstrap-cluster edge.
    """
    return str(shard_path.resolve().parent), int(game_id)


def candidate_game_components(
    candidate_games: list[set[GameKey]],
    *,
    candidate_missing_game_id: np.ndarray,
    found: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservative cluster ids for an audit-position/source-game graph.

    A position can match several stored games. Unioning audit rows through
    every candidate game guarantees that two positions sampled from the same
    source game cannot be assigned to different bootstrap clusters. If any
    exact match lacks an authoritative game id, no reported component is
    usable: that unidentified game could also contain a different audit row
    and therefore hide an edge anywhere in the graph. Global invalidation is
    intentionally conservative and remains safe after downstream row filters.
    """
    n = len(candidate_games)
    found_arr = np.asarray(found, dtype=bool)
    missing_arr = np.asarray(candidate_missing_game_id, dtype=bool)
    if found_arr.shape != (n,) or missing_arr.shape != (n,):
        raise ValueError("candidate-game masks must match the audit row count")

    parent = list(range(n))

    def root(row: int) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return row

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    first_row_for_game: dict[GameKey, int] = {}
    for row, games in enumerate(candidate_games):
        if not found_arr[row]:
            continue
        for game in sorted(games):
            previous = first_row_for_game.setdefault(game, row)
            union(row, previous)

    cluster_id = np.full(n, -1, dtype=np.int64)
    for row, games in enumerate(candidate_games):
        if found_arr[row] and games:
            cluster_id[row] = root(row)
    graph_identity_complete = not bool(np.any(found_arr & missing_arr))
    has_cluster = np.asarray([
        bool(graph_identity_complete and found_arr[row] and games)
        for row, games in enumerate(candidate_games)
    ], dtype=bool)
    return cluster_id, has_cluster


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


def require_complete_cluster_scan(skipped_layout: dict[str, int]) -> None:
    """Refuse cluster ids when any shard could hide a dependency edge."""
    if not skipped_layout:
        return
    detail = ", ".join(
        f"{layout}={count}" for layout, count in sorted(skipped_layout.items())
    )
    raise SystemExit(
        "cannot produce complete game_cluster_id values after skipping replay "
        f"shards with unscannable/wrong layouts ({detail}); use a uniform "
        "snapshot or extend the scanner to recover candidate-game edges from "
        "every layout",
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


def _array_identity(array: np.ndarray) -> dict[str, object]:
    """Hash the consumed C-order values with bounded scratch, not shard rereads."""
    header = {"dtype": array.dtype.str, "shape": list(array.shape)}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode() + b"\0")
    # external_loop buffers only a bounded chunk even for non-contiguous views.
    with np.nditer([array], flags=["external_loop", "buffered", "zerosize_ok"],
                   op_flags=[["readonly"]], order="C", buffersize=131072) as chunks:
        for chunk in chunks:
            digest.update(np.asarray(chunk).tobytes())
    return {**header, "sha256": digest.hexdigest(), "nbytes": int(array.nbytes)}


def _existing_file_identity(path: Path) -> dict[str, object]:
    try:
        return {"path": str(path), "sha256": sha256_file(path)}
    except FileNotFoundError:
        return {"path": str(path), "sha256": None}


def _candidate_rows_identity(array: np.ndarray, indices: list[int]) -> dict[str, object]:
    """Bind ordered decode candidates without copying/hashing unused full rows."""
    header = {
        "scheme": "ordered-index-and-row-digests-v1", "dtype": array.dtype.str,
        "source_shape": list(array.shape), "selected_rows": len(indices),
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode() + b"\0")
    for index in indices:
        digest.update(int(index).to_bytes(8, "little", signed=True))
        digest.update(bytes.fromhex(str(_array_identity(array[index])["sha256"])))
    return {**header, "sha256": digest.hexdigest()}


def _output_mode(path: Path) -> int | None:
    """Observe ordinary mode bits; refuse terminal links before publication.

    Atomic replacement creates a new inode: owner/ACL/xattrs/hard-link identity
    are not preserved. This is not protection against hostile directory or
    concurrent permission changes; callers must use a trusted output directory.
    """
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"refusing non-regular or symlink output: {path}")
    return stat.S_IMODE(info.st_mode)


def _validation_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report["matched"] == 0:
        return ["no audit rows matched the replay snapshot"]
    if not report["legal_move_sets_identical"]:
        errors.append(
            "matched rows generate a different legal-move set than their audit FEN; "
            "the audit set's key and fen disagree, so per-UCI labels belong to a "
            "different position than the one being scored"
        )
    if not report["canonicalisation_current_frame_identical"]:
        errors.append("recovered rows are not side-to-move canonical")
    if not report["canonicalisation_castling_104_107_identical"]:
        errors.append("recovered rows do not preserve castling-plane identity")
    return errors


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

    requested_out = args.out or default_matched_rows_path(args.audit_set)
    # Preserve NumPy's historical auto-appended .npz destination and the
    # existing report name for callers passing --out without that suffix.
    report_path = requested_out.with_suffix(requested_out.suffix + ".report.json")
    out_path = requested_out if str(requested_out).endswith(".npz") else Path(str(requested_out) + ".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_mode, report_mode = (_output_mode(path) for path in (out_path, report_path))
    prior_outputs = [_existing_file_identity(path) for path in (out_path, report_path)]
    audit_bytes = args.audit_set.read_bytes()
    audit = [
        json.loads(line)
        for line in audit_bytes.decode("utf-8").splitlines()
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
    src_has_game = np.zeros(n, dtype=bool)
    src_ply = np.full(n, -1, dtype=np.int64)
    src_selfplay = np.full(n, -1, dtype=np.int64)
    dup_count = np.zeros(n, dtype=np.int64)
    candidate_games: list[set[GameKey]] = [set() for _ in range(n)]
    candidate_missing_game_id = np.zeros(n, dtype=bool)

    t0 = time.time()
    fp_hits = 0
    skipped_layout: dict[str, int] = {}
    consumed_shards: list[dict[str, Any]] = []
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
        consumed: dict[str, Any] = {
            "path": str(path), "resolved_path": str(path.resolve()),
            "layout": {"history_encoding": hist, "planes": planes},
            "current_frame_scan": _array_identity(x12),
        }
        consumed_shards.append(consumed)
        cand: list[tuple[int, int]] = []
        for r, fp in enumerate(shard_fingerprints(x12)):
            cand.extend((r, ai) for ai in fps.get(fp, ()))
        if cand:
            fp_hits += len(cand)
            xfull = np.asarray(arr[:], dtype=np.float16)
            cols = {
                name: (
                    np.asarray(group[name][:]).reshape(-1)
                    if name in group else None
                )
                for name in (
                    "game_id", "has_game_id", "ply_index", "is_selfplay",
                )
            }
            candidate_rows = sorted({r for r, _ in cand})
            consumed["candidate_row_decision_inputs"] = _candidate_rows_identity(xfull, candidate_rows)
            consumed["identity_columns"] = {
                name: None if value is None else _array_identity(value)
                for name, value in cols.items()
            }
            decoded: dict[int, str | None] = {}
            for r in candidate_rows:
                board = decode_board_from_planes(
                    np.asarray(xfull[r], dtype=np.float32),
                    input_history_encoding=hist,
                )
                decoded[r] = None if board is None else position_key(board)
            for r, ai in cand:
                if decoded[r] != keys[ai]:
                    continue
                dup_count[ai] += 1
                game = cols["game_id"]
                has_game = cols["has_game_id"]
                if (
                    game is not None
                    and has_game is not None
                    and bool(has_game[r])
                ):
                    candidate_games[ai].add(source_game_key(path, int(game[r])))
                else:
                    candidate_missing_game_id[ai] = True
                if found[ai]:
                    continue
                found[ai] = True
                x_stored[ai] = xfull[r]
                src_shard[ai] = path.name
                src_row[ai] = r
                if (
                    game is not None
                    and has_game is not None
                    and bool(has_game[r])
                ):
                    src_game[ai] = int(game[r])
                    src_has_game[ai] = True
                for name, target in (
                    ("ply_index", src_ply),
                    ("is_selfplay", src_selfplay),
                ):
                    col = cols[name]
                    if col is not None:
                        target[ai] = int(col[r])
        if (si + 1) % 200 == 0 or si + 1 == len(shards):
            print(
                f"[scan] {si + 1}/{len(shards)} shards, matched "
                f"{int(found.sum())}/{n} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    scan_seconds = time.time() - t0
    # The selected history rows must use the frozen production layout, but the
    # bootstrap component graph needs every possible audit-row/source-game
    # edge. A skipped shard can duplicate one audit position and connect its
    # unseen game to another audit row, so successful matches in the accepted
    # subset do not make cluster ids complete. Fail before emitting an artifact
    # that downstream code could mistake for statistically independent data.
    require_complete_cluster_scan(skipped_layout)
    game_cluster_id, has_game_cluster_id = candidate_game_components(
        candidate_games,
        candidate_missing_game_id=candidate_missing_game_id,
        found=found,
    )

    # ---- verification: canonicalisation preserved, history actually filled ----
    x_fen_only = np.stack([
        encode_cboard(
            CBoard.from_board(board),
            input_history_encoding=STORED_HISTORY_ENCODING,
            input_extra_features=STORED_EXTRA_FEATURES,
        ).astype(np.float16)
        for board in boards
    ])

    def occupancy(x: np.ndarray, planes: slice) -> float | None:
        if not found.any():
            return None
        nz = np.abs(np.asarray(x[found], dtype=np.float32)).max(axis=(2, 3)) > 1e-9
        return float(nz[:, planes].mean())

    def rows_differing(planes: slice) -> float | None:
        if not found.any():
            return None
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
    ).mean(0) if found.any() else np.zeros(STORED_PLANES)
    per_plane_stored = (
        np.abs(np.asarray(x_stored[found], dtype=np.float32)).max(axis=(2, 3)) > 1e-9
    ).mean(0) if found.any() else np.zeros(STORED_PLANES)

    report: dict[str, Any] = {
        "audit_set": str(args.audit_set),
        "snapshot": str(args.snapshot),
        "input_history_encoding": STORED_HISTORY_ENCODING,
        "input_extra_features": STORED_EXTRA_FEATURES,
        "audit_rows": n,
        "matched": int(found.sum()),
        "unmatched": int((~found).sum()),
        "matched_rows_with_selected_source_game_id": int(
            np.count_nonzero(found & src_has_game)
        ),
        "matched_rows_with_game_cluster_id": int(
            np.count_nonzero(found & has_game_cluster_id)
        ),
        "game_cluster_kind": GAME_CLUSTER_KIND,
        "game_clusters": int(np.unique(
            game_cluster_id[has_game_cluster_id]
        ).shape[0]),
        "positions_with_multiple_candidate_games": int(sum(
            len(games) > 1 for games in candidate_games
        )),
        "positions_with_unidentified_candidate_game": int(
            np.count_nonzero(found & candidate_missing_game_id)
        ),
        "shards_scanned": len(shards),
        "shards_skipped_wrong_layout": skipped_layout,
        "fingerprint_candidate_pairs": fp_hits,
        "duplicate_multiplicity_mean": float(dup_count[found].mean()) if found.any() else 0.0,
        "duplicate_multiplicity_max": int(dup_count.max()),
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

    errors = _validation_errors(report)
    build_id = uuid.uuid4().hex
    status = "rejected" if errors else "validated"
    attempt_path = out_path.with_name(f"{out_path.name}.{status}.{build_id}.json")
    # Bind the actual input snapshots, not a later reread of a mutable corpus.
    report.update({
        "build_id": build_id, "validation_status": status,
        "validation_errors": errors, "attempt_report": str(attempt_path),
        "audit_set_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "consumed_shards": consumed_shards,
        "consumed_hash_scope": "decision inputs: captured audit bytes, entire current-frame scans, ordered candidate indices/full rows and identity columns; not whole shard/full-file hashes",
        "prior_outputs": prior_outputs,
        "prior_outputs_scope": "observed before input capture; not necessarily the immediate predecessor under concurrency",
        "requested_output": str(requested_out),
        "working_directory": str(Path.cwd()),
        "output_entry_path": str(out_path.parent.resolve() / out_path.name),
        "report_entry_path": str(report_path.parent.resolve() / report_path.name),
        "candidate_output": {"path": str(out_path), "sha256": None},
        "publication": "not attempted" if errors else "validated candidate; publication not confirmed",
    })
    if errors:
        report["observed_final_outputs"] = [
            _existing_file_identity(path) for path in (out_path, report_path)
        ]
        offending = [int(i) for i in np.flatnonzero(found) if (
            int(i) in legal_mismatch
            or not np.array_equal(x_stored[i, :PIECE_PLANES], x_fen_only[i, :PIECE_PLANES])
            or not np.array_equal(x_stored[i, 104:108], x_fen_only[i, 104:108])
        )]
        report["offending_rows_total"] = len(offending)
        report["offending_rows_limit"] = 4
        report["offending_rows"] = [{
            "audit_row": i, "key": keys[i], "fen": str(audit[i]["fen"]),
            "phase": int(audit[i]["phase"]), "source": int(audit[i]["source"]),
            "src_shard": src_shard[i], "src_row": int(src_row[i]),
            "arrays": {name: {
                **_array_identity(array),
                "base64_c_order": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
            } for name, array in (("stored", x_stored[i]), ("fen_only", x_fen_only[i]))},
        } for i in offending[:4]]
        # Evidence failure must never fall through to publishing invalid rows.
        atomic_write_text(attempt_path, json.dumps(report, indent=1, allow_nan=False), mode=0o600)
        # The bounded raw rows belong only in the private diagnostic, not logs.
        print(f"[match] rejected: {'; '.join(errors)}; evidence -> {attempt_path}")
        raise SystemExit("; ".join(errors) + f"; rejected evidence -> {attempt_path}")

    def write_candidate(path: Path) -> None:
        # The final destination above already preserves NumPy's suffix rule.
        # A file handle binds serialization to the exact atomic staging file.
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                build_id=np.asarray([build_id]),
                x_stored=x_stored, x_fen_only=x_fen_only, found=found,
                key=np.array(keys), phase=np.array([int(r["phase"]) for r in audit]),
                source=np.array([int(r["source"]) for r in audit]),
                src_shard=np.array(src_shard), src_row=src_row,
                # The selected source row identifies the history-bearing x_stored row.
                # It is not safe as a bootstrap cluster when a position has several
                # source matches, so the conservative component is stored separately.
                game_id=src_game, has_game_id=src_has_game,
                game_cluster_id=game_cluster_id,
                has_game_cluster_id=has_game_cluster_id,
                game_cluster_kind=np.array([GAME_CLUSTER_KIND]),
                ply_index=src_ply, is_selfplay=src_selfplay,
                dup_count=dup_count,
                per_plane_nonzero_fen_only=per_plane_fen,
                per_plane_nonzero_stored=per_plane_stored,
                input_history_encoding=np.array([STORED_HISTORY_ENCODING]),
                input_extra_features=np.array([STORED_EXTRA_FEATURES]),
                snapshot=np.array([str(args.snapshot)]),
                audit_set=np.array([str(args.audit_set)]),
            )
        report["candidate_output"] = {"path": str(out_path), "sha256": sha256_file(path)}
        # Save the attempt before the sole consumer-visible commit point. If this
        # fails, atomic_write preserves the old NPZ and normal report.
        atomic_write_text(attempt_path, json.dumps(report, indent=1, allow_nan=False), mode=0o600)
        # Catch terminal links introduced during the scan/serialization as well.
        # This check is not a lock against hostile output-directory mutation.
        for target in (out_path, report_path):
            _output_mode(target)

    try:
        atomic_write(out_path, write_candidate, preserve_suffix=True, mode=out_mode)
        report["publication"] = "NPZ published; compatibility report is not an atomic pair"
        _output_mode(report_path)
        atomic_write_text(report_path, json.dumps(report, indent=1, allow_nan=False), mode=report_mode)
    except BaseException:
        # replace may have succeeded before directory fsync or report failure.
        # Never restore an old artifact over a concurrent successful publisher.
        print(
            f"[match] publication did not confirm completion; inspect {out_path} "
            f"and attempt {attempt_path}; a valid NPZ may already be visible",
            flush=True,
        )
        raise
    print(json.dumps(report, indent=1, allow_nan=False))
    print(f"[match] index -> {out_path}\n[match] report -> {report_path}")


if __name__ == "__main__":
    main()
