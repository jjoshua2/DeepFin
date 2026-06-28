"""Frozen deep-SF audit set: shared helpers for build + scoring scripts.

The audit pipeline (scripts/build_audit_set.py + scripts/audit_targets.py)
evaluates training-target candidates DIRECTLY against deep unhandicapped
Stockfish labels, before any training compute is spent. This module holds
the pieces both scripts and the tests share:

- board reconstruction from stored shard input planes (shards do not store
  FENs; the current-position piece planes + metadata planes are invertible
  up to the fullmove number, which does not affect evaluation)
- the per-position JSONL record schema
- distribution-vs-deep-labels scoring: expected/top-1 regret in cp, and
  WDL Brier / expected calibration error

Phase buckets use the same piece-count thresholds as the soft-policy probe
and the model's phase adapter (model/transformer.py): endgame <= 13 pieces
< middlegame <= 22 < opening.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.encoding.lc0 import (
    normalize_lc0_history_encoding,
    uses_lc0_root_history,
)
from chess_anti_engine.stockfish.wdl import mate_to_effective_cp

PHASE_THRESHOLDS = (13, 22)
PHASE_NAMES = ("endgame", "middlegame", "opening")
SOURCE_NAMES = ("selfplay", "curriculum")


def phase_bucket(piece_count: int) -> int:
    """0=endgame, 1=middlegame, 2=opening — same thresholds as the probe."""
    low, high = PHASE_THRESHOLDS
    if piece_count <= low:
        return 0
    if piece_count > high:
        return 2
    return 1


def position_key(board: chess.Board) -> str:
    """Dedup key: FEN without move counters (placement/turn/castling/ep)."""
    parts = board.fen().split(" ")
    return " ".join(parts[:4])


# ---------------------------------------------------------------------------
# Shard-plane -> board reconstruction (side-to-move canonical)
# ---------------------------------------------------------------------------
#
# The input encoding is POV-normalized: planes 0..11 are us/them pieces in
# the side-to-move's orientation (plane[rank][file] for white to move,
# plane[7-rank][file] for black), and the legacy layout does not store the
# absolute color AT ALL (verified empirically -- plane 101 is constant 1.0
# for both sides). Chess is exactly symmetric under color-mirror, so the
# audit reconstructs every position in CANONICAL form: side to move = white,
# "us" = white. For black-to-move originals this yields the color-mirrored
# twin -- identical eval, move quality, and legality structure from the
# side-to-move's point of view, and it dedupes mirrored duplicates for free.
#
# Metadata planes (verified empirically against encode_cboard):
#   legacy:               castling 96=us-K 97=us-Q 98=them-K 99=them-Q,
#                         ep file 100, rule50/100 102
#   lc0_root[_legacy_meta]: castling 104=us-Q 105=us-K 106=them-Q 107=them-K,
#                         rule50 109 (/100 for legacy_meta, raw count for
#                         plain lc0_root), ep file 110 (legacy_meta only --
#                         plain lc0_root has no EP plane, EP is dropped)


def decode_board_from_planes(
    x: np.ndarray, *, input_history_encoding: str | None,
) -> chess.Board | None:
    """Reconstruct the stored position in side-to-move-canonical form.

    Returns a board with WHITE to move whose white pieces are the original
    side to move (the color-mirrored twin when black was to move). Returns
    None for corrupt/illegal rows. The fullmove number is not stored and is
    set to 1 -- it does not affect evaluation.
    """
    hist = normalize_lc0_history_encoding(input_history_encoding)
    root = uses_lc0_root_history(hist)
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (8, 8) or arr.shape[0] < 112:
        return None

    if root:
        castle_planes = (105, 104, 107, 106)  # us-K, us-Q, them-K, them-Q
        rule50_raw = float(arr[109, 0, 0])
        rule50 = int(round(rule50_raw * 100.0)) if hist == "lc0_root_legacy_meta" else int(round(rule50_raw))
        ep_plane = arr[110] if hist == "lc0_root_legacy_meta" else None
    else:
        castle_planes = (96, 97, 98, 99)
        rule50 = int(round(float(arr[102, 0, 0]) * 100.0))
        ep_plane = arr[100]

    board = chess.Board(fen=None)
    board.clear()
    board.turn = chess.WHITE
    for slot, color in ((0, chess.WHITE), (6, chess.BLACK)):
        for i, pt in enumerate(chess.PIECE_TYPES):
            for row, col in np.argwhere(arr[slot + i] > 0.5):
                sq = chess.square(int(col), int(row))
                if board.piece_at(sq) is not None:
                    return None  # overlapping piece planes -> corrupt row
                board.set_piece_at(sq, chess.Piece(pt, color))

    flags = [bool(arr[p].max() > 0.5) for p in castle_planes]
    castling = "".join(
        ch for has, ch in zip(flags, ("K", "Q", "k", "q"), strict=True) if has
    )
    try:
        board.set_castling_fen(castling or "-")
    except ValueError:
        return None

    if ep_plane is not None and float(ep_plane.max()) > 0.5:
        ep_file = int(np.argmax(ep_plane.max(axis=0)))
        board.ep_square = chess.square(ep_file, 5)

    board.halfmove_clock = max(0, min(150, rule50))
    board.fullmove_number = 1

    status = board.status()
    # EP squares python-chess deems invalid (no capturer) are dropped rather
    # than failing the whole position.
    if status & chess.STATUS_INVALID_EP_SQUARE:
        board.ep_square = None
        status = board.status()
    if status != chess.STATUS_VALID:
        return None
    return board


# ---------------------------------------------------------------------------
# Audit record schema + IO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditPosition:
    """One deep-labeled audit position (a parsed JSONL row)."""

    key: str
    fen: str
    phase: int           # 0/1/2 per phase_bucket
    source: int          # 0=selfplay, 1=curriculum
    # Deep SF labels (unhandicapped, >=1M nodes):
    move_cp: dict[str, float]   # uci -> cp from the side to move (mate folded in)
    best_cp: float
    deep_wdl: tuple[float, float, float]   # native SF WDL, side-to-move POV, sums to 1
    sf_nodes: int
    sf_depth: int
    # Optional full-strength game outcome (0=W/1=D/2=L from stm POV); None
    # when the source game continued under a handicap (the usual case).
    outcome: int | None = None


def parse_audit_record(line: str) -> AuditPosition:
    d = json.loads(line)
    move_cp: dict[str, float] = {}
    for row in d["multipv"]:
        cp = row.get("cp")
        mate = row.get("mate")
        if mate:
            eff = float(mate_to_effective_cp(int(mate)))
        elif cp is not None:
            eff = float(cp)
        else:
            continue  # unscoreable line (no cp, no mate) — skip, don't crash
        # First (best-ranked) listing of a move wins on duplicates.
        move_cp.setdefault(str(row["move"]), eff)
    if not move_cp:
        raise ValueError(f"audit record without scored lines: {d.get('fen')}")
    wdl = d["wdl"]
    total = max(1e-9, float(wdl[0]) + float(wdl[1]) + float(wdl[2]))
    return AuditPosition(
        key=str(d["key"]),
        fen=str(d["fen"]),
        phase=int(d["phase"]),
        source=int(d["source"]),
        move_cp=move_cp,
        best_cp=max(move_cp.values()),
        deep_wdl=(float(wdl[0]) / total, float(wdl[1]) / total, float(wdl[2]) / total),
        sf_nodes=int(d["nodes"]),
        sf_depth=int(d["depth"]),
        outcome=d.get("outcome"),
    )


def load_audit_set(path: str | Path) -> list[AuditPosition]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(parse_audit_record(line))
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def legal_full_indices(board: chess.Board) -> tuple[list[str], np.ndarray]:
    """Legal moves as (uci list, full-4672 policy indices).

    Audit boards are side-to-move canonical (white to move), so the
    encoding turn flag is always True. Lives here, next to the schema, so
    scorers and tests share one definition.
    """
    from chess_anti_engine.moves.encode import uci_to_policy_index

    ucis: list[str] = []
    idxs: list[int] = []
    for mv in board.legal_moves:
        uci = mv.uci()
        a = uci_to_policy_index(uci, True)
        if a >= 0:
            ucis.append(uci)
            idxs.append(a)
    return ucis, np.asarray(idxs, dtype=np.int64)


def move_regrets(
    pos: AuditPosition, legal_ucis: list[str],
) -> np.ndarray:
    """Per-legal-move deep-SF regret in cp (best_cp - cp(move), >= 0).

    Moves the deep MultiPV did not list get the regret of the WORST listed
    line as a floor — a lower bound on their true regret, biased optimistic
    for bad distributions. With MultiPV >= 10 at 1M nodes the unlisted mass
    of any sane candidate is tiny; the bound is reported, not hidden.
    """
    worst_listed = min(pos.move_cp.values())
    floor = pos.best_cp - worst_listed
    out = np.empty(len(legal_ucis), dtype=np.float64)
    for i, uci in enumerate(legal_ucis):
        cp = pos.move_cp.get(uci)
        out[i] = (pos.best_cp - cp) if cp is not None else floor
    return np.maximum(out, 0.0)


def expected_and_top1_regret(
    probs: np.ndarray, regrets: np.ndarray,
) -> tuple[float, float]:
    """(expected regret of sampling from probs, regret of argmax move)."""
    p = np.asarray(probs, dtype=np.float64)
    total = float(p.sum())
    if total <= 0.0:
        # Degenerate distribution: treat as uniform over legal moves.
        p = np.full_like(p, 1.0 / max(1, p.size))
    else:
        p = p / total
    return float((p * regrets).sum()), float(regrets[int(np.argmax(p))])


# Criticality = deep-SF cp gap between the best and 2nd-best listed move. Shared
# by audit_targets / bt4_audit / audit_compare_buckets so a joined comparison
# can't put the same position in different buckets.
CRITICALITY_GAP_EDGES: tuple[float, ...] = (20.0, 50.0, 100.0)
CRITICALITY_BUCKET_NAMES: tuple[str, ...] = (
    "quiet(<20)", "soft(20-50)", "sharp(50-100)", "decisive(>=100)",
)


def criticality_gap(move_cp: dict[str, float]) -> float:
    """cp gap between the best and 2nd-best listed move (inf if <2 moves)."""
    listed = sorted(move_cp.values(), reverse=True)
    return float(listed[0] - listed[1]) if len(listed) >= 2 else float("inf")


def criticality_bucket(gap: float) -> int:
    """Index into CRITICALITY_BUCKET_NAMES for a best-vs-2nd cp gap."""
    for i, edge in enumerate(CRITICALITY_GAP_EDGES):
        if gap < edge:
            return i
    return len(CRITICALITY_GAP_EDGES)


def wdl_brier(pred: np.ndarray, target: np.ndarray) -> float:
    """Squared error between two (3,) WDL distributions."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(((pred - target) ** 2).sum())


def wdl_ece(
    preds: np.ndarray, targets: np.ndarray, *, n_bins: int = 10,
) -> float:
    """Expected calibration error of (N,3) WDL predictions vs targets.

    Confidence is the max predicted class probability; "correct" means the
    prediction's argmax matches the target distribution's argmax. Same
    binning convention as train/losses.py::wdl_calibration_stats.
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    conf = preds.max(axis=1)
    correct = (preds.argmax(axis=1) == targets.argmax(axis=1)).astype(np.float64)
    edges = np.linspace(1.0 / n_bins, 1.0 - 1.0 / n_bins, n_bins - 1)
    bins = np.clip(np.digitize(conf, edges), 0, n_bins - 1)
    ece = 0.0
    n = preds.shape[0]
    for b in range(n_bins):
        sel = bins == b
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(float(conf[sel].mean()) - float(correct[sel].mean()))
    return float(ece)
