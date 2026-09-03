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
import json
import time
from pathlib import Path
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

    out_path = args.out or default_matched_rows_path(args.audit_set)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    report: dict[str, object] = {
        "audit_set": str(args.audit_set),
        "snapshot": str(args.snapshot),
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
    report_path = out_path.with_suffix(out_path.suffix + ".report.json")
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
