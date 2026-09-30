"""Reconstruct a Ceres-selected label from a proved saved-game source row.

The caller must obtain ``selection`` from an independently verified route,
source schedule and physical-call ledger. A saved-game archive cannot itself
authenticate its source manifest SHA, schedule root ID or observed model-call
feed. This module performs no teacher inference and does not admit a row to a
corpus.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import chess.syzygy
import numpy as np

from chess_anti_engine.source.ceres_saved_game import (
    SavedCeresGame,
    load_saved_game,
    read_saved_game_archive,
)

POLICY_WIDTH = 1858
ROUTE_DOMAIN = b"mixed500m/selected-teacher/v1\0"
ROUTE_SEED = "c8308d72aa70a137c34679f076959d9183e58dbf4cae3fef879b60fde7a2850b"
INPUT_DOMAIN = (
    b"mixed500m/student-x/v1;history=lc0_root_legacy_meta;"
    b"extras=v2_threats;repfix=1;shape=175x8x8;pack=<f2,C;model=<f4,C\n"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _need(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def selected_teacher(uid: tuple[str, str, str, int, int]) -> str:
    """The pinned equal-route selector for the mixed500m chosen target."""
    _need(type(uid) is tuple and len(uid) == 5, "source-qualified UID")
    payload = json.dumps([ROUTE_SEED, *uid], separators=(",", ":"),
                         ensure_ascii=True).encode("ascii") + b"\n"
    return ("BT4" if hashlib.sha256(ROUTE_DOMAIN + payload).digest()[0] & 1 == 0
            else "Ceres")


@dataclass(frozen=True)
class CeresOwnerSelection:
    """Pins from the verified source index, fair route and call ledger."""

    uid: tuple[str, str, str, int, int]
    source_archive_sha256: str
    model_sha256: str
    input_digest: str
    feed_sha256: str
    history_stack_sha256: str
    legal_context_sha256: str
    teacher_query_sha256: str
    teacher: str


@dataclass(frozen=True, eq=False)
class CeresOwnerTarget:
    uid: tuple[str, str, str, int, int]
    policy: np.ndarray  # float16[1858], compact policy space
    wdl: np.ndarray     # float16[3], calibrated dual-head WDL
    policy_sha256: str
    wdl_sha256: str
    input_digest: str
    feed_sha256: str
    model_sha256: str


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    _need(data.shape == (3,) and bool(np.isfinite(data).all()),
          "Ceres value head shape/finite differs")
    scaled = (data - data.max()) / temperature
    weights = np.exp(scaled)
    return weights / weights.sum()


def reconstruct_ceres_owner_target(
    game: SavedCeresGame, selection: CeresOwnerSelection, *,
    source_manifest_sha256: str, root_id: str,
) -> CeresOwnerTarget:
    """Build the chosen Ceres target from one verified saved-game row.

    ``game`` should come from ``read_saved_game_archive`` (or the equivalent
    strict Syzygy reader).  The caller owns the independently pinned schedule
    and selection receipt.  Raw source logits are in compact-index order.
    """
    uid = selection.uid
    _need(selection.teacher == "Ceres" and selected_teacher(uid) == "Ceres"
          and uid[0] == source_manifest_sha256 and uid[2] == root_id
          and uid[1] == game.game["source_namespace"]
          and type(uid[3]) is int and uid[3] == game.game["game_id"]
          and type(uid[4]) is int and 0 <= uid[4] < len(game.rows),
          "Ceres route/UID/schedule binding differs")
    _need(game.source_archive_sha256 == selection.source_archive_sha256
          and game.model_sha256 == selection.model_sha256,
          "Ceres physical archive/model pin differs")
    index = uid[4]
    row = game.rows[index]
    x = game.arrays["x"][index]
    _need(row["ply_index"] == index and row["game_id"] == uid[3]
          and row["source_namespace"] == uid[1]
          and x.shape == (175, 8, 8) and x.dtype == np.float16,
          "Ceres selected row identity/input shape differs")
    consumed = np.ascontiguousarray(x, dtype="<f4").tobytes(order="C")
    _need(len(consumed) == 44800
          and selection.input_digest == _sha(INPUT_DOMAIN + consumed)
          and selection.feed_sha256 == row["feed_sha256"]
          and selection.history_stack_sha256 == row["history_stack_sha256"],
          "Ceres selected input/feed/history provenance differs")
    legal_sha = _sha(_canonical(sorted(row["legal_moves_uci_sorted"])))
    query_sha = _sha(_canonical([
        "lc0_root_legacy_meta", "v2_threats", True, row["root_fen"],
        selection.history_stack_sha256, legal_sha,
    ]))
    _need(selection.legal_context_sha256 == legal_sha
          and selection.teacher_query_sha256 == query_sha,
          "Ceres selected legal/query provenance differs")
    offsets = game.arrays["legal_offsets"]
    lo, hi = int(offsets[index]), int(offsets[index + 1])
    compact = game.arrays["legal_compact"][lo:hi]
    logits = game.arrays["legal_logits"][lo:hi]
    _need(0 < len(compact) == len(logits) <= POLICY_WIDTH
          and len(np.unique(compact)) == len(compact)
          and bool(np.all(compact < POLICY_WIDTH))
          and logits.dtype == np.float16 and bool(np.isfinite(logits).all()),
          "Ceres compact legal head differs")
    dense = np.zeros(POLICY_WIDTH, dtype=np.float16)
    dense[compact] = logits
    mask = np.zeros(POLICY_WIDTH, dtype=bool)
    mask[compact] = True
    scaled = np.where(mask, dense.astype(np.float64), -np.inf)
    scaled = (scaled - scaled.max()) / .5
    weights = np.exp(scaled)
    policy = (weights / weights.sum()).astype(np.float16)
    primary = game.arrays["value"][index]
    secondary = game.arrays["value2"][index]
    wdl = (.6 * _softmax(primary, .55)
           + .4 * _softmax(secondary, 1.5)).astype(np.float16)
    return CeresOwnerTarget(
        uid, policy, wdl, _sha(policy.tobytes()), _sha(wdl.tobytes()),
        selection.input_digest, selection.feed_sha256, selection.model_sha256,
    )


def reconstruct_ceres_owner_targets(
    game: SavedCeresGame, selections: tuple[CeresOwnerSelection, ...], *,
    source_manifest_sha256: str, root_id: str,
) -> tuple[CeresOwnerTarget, ...]:
    """Process selected rows of one proved game without reopening its archive."""
    _need(bool(selections) and len(selections) <= len(game.rows)
          and len({selection.uid for selection in selections}) == len(selections),
          "Ceres owner selection roster empty, duplicated or oversized")
    return tuple(reconstruct_ceres_owner_target(
        game, selection, source_manifest_sha256=source_manifest_sha256,
        root_id=root_id) for selection in selections)


def read_ceres_owner_target(
    archive: Path, selection: CeresOwnerSelection, *,
    source_manifest_sha256: str, root_id: str,
    syzygy_path: str | None = None,
    match_tablebase: chess.syzygy.Tablebase | None = None,
) -> CeresOwnerTarget:
    """Public physical/semantic reader followed by selected-owner reconstruction."""
    game = read_saved_game_archive(
        archive, syzygy_path=syzygy_path, match_tablebase=match_tablebase)
    return reconstruct_ceres_owner_target(
        game, selection, source_manifest_sha256=source_manifest_sha256,
        root_id=root_id,
    )


def load_source_ceres_owner_targets(
    source_archive: Path, selections: tuple[CeresOwnerSelection, ...], *,
    source_manifest_sha256: str, root_id: str,
    syzygy_path: str | None = None,
    match_tablebase: chess.syzygy.Tablebase | None = None,
) -> tuple[CeresOwnerTarget, ...]:
    """Physically verify a source game once, then reconstruct its chosen rows."""
    _need(bool(selections) and len({selection.uid for selection in selections}) == len(selections)
          and all(selection.uid[3] == selections[0].uid[3]
                  and selection.source_archive_sha256 == selections[0].source_archive_sha256
                  for selection in selections),
          "Ceres owner selections must share one pinned game/archive")
    game = load_saved_game(
        source_archive, selections[0].source_archive_sha256, selections[0].uid[3],
        syzygy_path=syzygy_path, match_tablebase=match_tablebase)
    return reconstruct_ceres_owner_targets(
        game, selections, source_manifest_sha256=source_manifest_sha256,
        root_id=root_id,
    )
