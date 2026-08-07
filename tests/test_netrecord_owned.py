"""_NetRecord must own its array fields (2026-08-07 worker OOM).

Records built from rows of (N, ...) batch arrays (``c_x[j]``/``xs_batch[j]``
in selfplay/network_turn.py) used to store bare views, pinning the whole
parent batch until the record's game finalized — tens of GB per worker with
~16 slots x hundreds of in-flight plies. The record now takes ownership at
construction; these tests fail on the pre-fix code.
"""
from __future__ import annotations

import chess
import numpy as np

from chess_anti_engine.selfplay.state import _NetRecord


def _record_from_batch_rows(j: int, batch_x, batch_probs, batch_wdl, batch_mask):
    return _NetRecord(
        batch_x[j], batch_probs[j], batch_wdl[j], batch_wdl[j],
        chess.WHITE, 10, True, 1.0, 1.0, 1.0,
        legal_mask=batch_mask[j],
        sf_wdl=batch_wdl[j],
        sf_policy_target=batch_probs[j],
        x_lc0_root=batch_x[j],
        relations=batch_x[j, :2],
    )


def test_record_does_not_pin_parent_batch():
    batch_x = np.zeros((8, 175, 8, 8), dtype=np.float32)
    batch_probs = np.zeros((8, 1858), dtype=np.float32)
    batch_wdl = np.zeros((8, 3), dtype=np.float32)
    batch_mask = np.zeros((8, 1858), dtype=np.uint8)
    rec = _record_from_batch_rows(3, batch_x, batch_probs, batch_wdl, batch_mask)
    for name in ("x", "policy_probs", "net_wdl_est", "search_wdl_est",
                 "legal_mask", "sf_wdl", "sf_policy_target", "x_lc0_root",
                 "relations"):
        arr = getattr(rec, name)
        assert arr is not None
        assert arr.base is None, f"{name} is a view — pins its parent batch"
        for parent in (batch_x, batch_probs, batch_wdl, batch_mask):
            assert not np.shares_memory(arr, parent), (
                f"{name} shares memory with a batch array"
            )


def test_record_values_survive_ownership_copy():
    batch_x = np.arange(8 * 175 * 8 * 8, dtype=np.float32).reshape(8, 175, 8, 8)
    batch_probs = np.arange(8 * 1858, dtype=np.float32).reshape(8, 1858)
    batch_wdl = np.arange(24, dtype=np.float32).reshape(8, 3)
    batch_mask = (np.arange(8 * 1858) % 2).astype(np.uint8).reshape(8, 1858)
    rec = _record_from_batch_rows(5, batch_x, batch_probs, batch_wdl, batch_mask)
    np.testing.assert_array_equal(rec.x, batch_x[5])
    np.testing.assert_array_equal(rec.policy_probs, batch_probs[5])
    np.testing.assert_array_equal(rec.net_wdl_est, batch_wdl[5])
    assert rec.legal_mask is not None
    np.testing.assert_array_equal(rec.legal_mask, batch_mask[5])


def test_owned_arrays_are_not_copied():
    x = np.zeros((175, 8, 8), dtype=np.float32)
    probs = np.zeros((1858,), dtype=np.float32)
    wdl = np.zeros((3,), dtype=np.float32)
    rec = _NetRecord(x, probs, wdl, wdl.copy(), chess.WHITE, 0, True,
                     1.0, 1.0, 1.0)
    assert rec.x is x
    assert rec.policy_probs is probs
    assert rec.net_wdl_est is wdl
