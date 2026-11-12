"""Caller-owned PUCT leaf snapshots and post-cache Syzygy overrides.

A FEN cannot establish repetition history. These routes conservatively probe
only reset-clock leaves, even when a legacy probe would accept a positive clock.
This is NN fallback on missing legacy tables, not strict match adjudication.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts.vloss import MAX_PATH
from chess_anti_engine.tablebase import SyzygyProbe

LeafSnapshots = tuple[np.ndarray, list[CBoard]]


def descend_batch(
    tree: Any,
    root_id: int,
    root_cb: CBoard,
    n: int,
    c_puct: float,
    fpu_root: float,
    fpu_red: float,
    vloss: int,
    enc: np.ndarray,
    bufs: dict[str, np.ndarray],
    vloss_mode: int,
    cache_keys: np.ndarray | None,
    relations: np.ndarray | None,
    history_args: tuple[int, ...],
    probe: SyzygyProbe | None,
) -> LeafSnapshots | None:
    """Preserve the old ABI/off path; TB batches return private sparse boards."""
    args = (
        c_puct,
        fpu_root,
        fpu_red,
        vloss,
        enc,
        bufs["leaf_ids"],
        bufs["path_buf"],
        bufs["path_lens"],
        bufs["legal_buf"],
        bufs["legal_lens"],
        bufs["term_qs"],
        bufs["is_term"],
        vloss_mode,
        cache_keys,
        relations,
        *history_args,
    )
    if probe is not None and probe.max_pieces > 0:
        descend = getattr(tree, "batch_descend_puct_tb", None)
        if descend is None:
            raise RuntimeError(
                "compiled MCTSTree lacks batch_descend_puct_tb; rebuild the native extension"
            )
        return descend(root_id, root_cb, n, probe.max_pieces, *args)
    tree.batch_descend_puct(root_id, root_cb, n, *args)
    return None


def override_leaf_values(
    probe: SyzygyProbe | None,
    snapshots: LeafSnapshots | None,
    wdl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Copy only on eligible rows, after raw cache materialization/storage."""
    if probe is None or snapshots is None:
        return wdl, None
    rows, boards = snapshots
    eligible = [
        j
        for j, cb in enumerate(boards)
        if int(cb.halfmove_clock) == 0
        and int(cb.castling) == 0
        and (int(cb.occ_white) | int(cb.occ_black)).bit_count() <= probe.max_pieces
    ]
    if not eligible:
        return wdl, None
    indices = np.asarray(rows[eligible], dtype=np.int32)
    result = np.array(wdl, dtype=np.float32, copy=True)
    statuses = np.zeros(len(eligible), dtype=np.int8)
    probe.apply(
        [boards[j] for j in eligible], result, indices=indices, solved_out=statuses
    )
    solved = np.zeros(len(wdl), dtype=np.int8)
    solved[indices] = statuses
    return result, solved


def mark_batch_solved(
    tree: Any,
    bufs: dict[str, np.ndarray],
    solved: np.ndarray | None,
    *,
    path_prefix: tuple[int, ...] = (),
) -> None:
    """Called after integration/VL release; never release those paths twice."""
    if solved is None:
        return
    for row in np.flatnonzero(solved):
        start = int(row) * MAX_PATH
        path = bufs["path_buf"][start : start + int(bufs["path_lens"][row])]
        if path_prefix:
            path = np.concatenate((np.asarray(path_prefix, dtype=np.int32), path))
        tree.mark_solved_path(path, int(solved[row]))
