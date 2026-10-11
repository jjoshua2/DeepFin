"""Unqualified, CPU-testable C3 own-root inference contract.

This is an adapter, not a generator or a G10-derived sidecar. A future worker
must create the ORT session with the pinned collector's ``open_teacher`` and
retain its first-call provider proof before using any result as a source row.
The move actor and final shard writer deliberately live outside this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np


SOURCE_SCHEMA = "ceres_own_source_v0_unqualified"
POLICY_WIDTH = 1858
FEED_SHAPE = (64, 137)
OUTPUT_NAMES = ("policy", "value", "value2")


@dataclass(frozen=True)
class Root:
    slot_id: int
    board: Any
    fen: str
    move_stack: tuple[Any, ...]
    legal_moves: tuple[Any, ...]
    compact_indices: tuple[int, ...]
    independent_leela_indices: tuple[int, ...]
    feed: np.ndarray


@dataclass(frozen=True)
class RawRootOutput:
    source_schema: str
    slot_id: int
    fen: str
    compact_indices: np.ndarray
    leela_indices: np.ndarray
    feed: np.ndarray
    feed_sha256: str
    policy_logits: np.ndarray  # Full, untransformed C3 Leela-1858 FP16 output.
    legal_policy_logits: np.ndarray  # Raw legal subset, still logits.
    value_logits: np.ndarray  # Primary WDL, win/draw/loss, side to move.
    value2_logits: np.ndarray  # Secondary WDL, same raw native dtype/order.


@dataclass(frozen=True)
class BatchReceipt:
    real_rows: int
    padding_rows: int
    physical_rows: int
    calls: int
    remainder: str
    output_names: tuple[str, str, str]


def _frozen_copy(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


def infer_raw_roots(
    roots: Sequence[Root],
    *,
    session: Any,
    physical_batch: int = 32,
    gather_context: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    gather_indices: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[tuple[RawRootOutput, ...], BatchReceipt]:
    """Use the collector's exact parameterized isolated repeat-last feed and raw C3 outputs.

    The session must come from pinned ``ceres_derived_sidecar.open_teacher``;
    that method's provider proof and runtime/source pins are separate gates.
    No actor softmax/temperature or G10 qualification is implied here.
    """
    if type(physical_batch) is not int or physical_batch not in (32,256,512):
        raise ValueError("isolated physical batch must be32/256/512")
    n = len(roots)
    if n < 1 or n > 8192:
        raise ValueError("C3 parameterized isolated adapter needs 1..8192 real roots")
    if len({root.slot_id for root in roots}) != n:
        raise ValueError("duplicate C3 root slot")
    for root in roots:
        indices = root.compact_indices + root.independent_leela_indices
        if (type(root.slot_id) is not int or root.slot_id < 0
                or not root.legal_moves or len(root.compact_indices) != len(root.legal_moves)
                or len(root.independent_leela_indices) != len(root.legal_moves)
                or any(type(index) is not int or not 0 <= index < POLICY_WIDTH for index in indices)
                or len(set(root.compact_indices)) != len(root.legal_moves)
                or len(set(root.independent_leela_indices)) != len(root.legal_moves)):
            raise ValueError("C3 root legal indices are invalid or duplicated")
    output: list[RawRootOutput] = []
    padding = 0
    calls = 0
    for start in range(0, n, physical_batch):
        chunk = roots[start:start + physical_batch]
        real = len(chunk)
        for root in chunk:
            if root.board.fen() != root.fen or tuple(root.board.move_stack) != root.move_stack:
                raise ValueError("prepared C3 board changed before inference")
        feed = np.stack([root.feed for root in chunk])
        physical = feed if real == physical_batch else np.concatenate(
            (feed, np.repeat(feed[-1:], physical_batch - real, axis=0)), axis=0)
        padding += physical_batch - real
        if physical.dtype != np.uint8 or physical.shape != (physical_batch, *FEED_SHAPE):
            raise ValueError("physical C3 feed differs from parameterized isolated uint8 contract")
        # Refuse a changed/ambiguous legal map before spending a model call.
        pawn_mask, castling = gather_context(feed)
        gather: Any = gather_indices(pawn_mask, castling)
        if not isinstance(gather, np.ndarray) or gather.shape != (real, POLICY_WIDTH):
            raise ValueError("C3 Leela gather matrix shape differs")
        mapped_rows: list[np.ndarray] = []
        for local, root in enumerate(chunk):
            compact = np.asarray(root.compact_indices, dtype=np.int64)
            mapped = np.asarray(gather[local, compact], dtype=np.int64)
            if (len(set(mapped.tolist())) != len(mapped)
                    or not np.array_equal(mapped, root.independent_leela_indices)):
                raise ValueError("C3 legal Leela map disagrees with independent move oracle")
            mapped_rows.append(mapped)
        fetched = session.run(list(OUTPUT_NAMES), {"squares_byte": physical})
        calls += 1
        if not isinstance(fetched, (list, tuple)) or len(fetched) != 3:
            raise ValueError("C3 session returned missing raw heads")
        for value, width in zip(fetched, (POLICY_WIDTH, 3, 3), strict=True):
            if (not isinstance(value, np.ndarray) or value.dtype != np.dtype("float16")
                    or value.shape != (physical_batch, width) or not np.isfinite(value).all()):
                raise ValueError("C3 raw output dtype, shape, or finiteness differs")
        for local, root in enumerate(chunk):
            compact = np.asarray(root.compact_indices, dtype=np.int64)
            mapped = mapped_rows[local]
            policy = _frozen_copy(fetched[0][local])
            raw_feed = _frozen_copy(root.feed)
            output.append(RawRootOutput(
                SOURCE_SCHEMA, root.slot_id, root.fen,
                _frozen_copy(compact.astype(np.uint16)),
                _frozen_copy(mapped.astype(np.uint16)),
                raw_feed, sha256(raw_feed.tobytes(order="C")).hexdigest(),
                policy, _frozen_copy(policy[mapped]),
                _frozen_copy(fetched[1][local]), _frozen_copy(fetched[2][local]),
            ))
    receipt = BatchReceipt(n, padding, n + padding, calls, "repeat_last_real", OUTPUT_NAMES)
    if receipt.physical_rows != receipt.calls * physical_batch:
        raise AssertionError("C3 physical batch accounting differs")
    return tuple(output), receipt
