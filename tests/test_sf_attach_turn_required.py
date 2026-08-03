"""The SF-label attach cannot be called without `turn`.

`_attach_sf_target_to_record` used to take `turn: bool | None = None` and skip
the sparse-MultiPV stamp when it was None. Every call site passed it, so the
skip was unreachable -- but a future caller that omitted it would have produced
a fully-labelled row with `has_sf_multipv_raw = 0`, which is precisely the
per-row fingerprint `sf_multipv_presence_counts` reports as SF desync
contamination. A guard whose only reachable effect is a false alarm is worse
than no guard; `turn` is required now.

The near-duplicate `_attach_sf_target_to_last_record` (same 15 lines, different
record lookup) delegates rather than repeating them, so the two cannot drift.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.state import _NetRecord
from chess_anti_engine.selfplay.stockfish_turn import (
    _attach_sf_target_to_last_record,
    _attach_sf_target_to_record,
)


def _rec() -> _NetRecord:
    return _NetRecord(
        x=np.zeros((1,), dtype=np.float32),
        policy_probs=np.zeros((1,), dtype=np.float32),
        net_wdl_est=np.zeros((3,), dtype=np.float32),
        search_wdl_est=np.zeros((3,), dtype=np.float32),
        pov_color=True,
        ply_index=0,
        has_policy=True,
        priority=1.0,
        sample_weight=1.0,
        keep_prob=1.0,
    )


def _res() -> SimpleNamespace:
    pv = SimpleNamespace(move_uci="e2e4", cp=25, mate=None, wdl=[0.5, 0.3, 0.2])
    return SimpleNamespace(
        bestmove_uci="e2e4", cp=25, mate=None, wdl=[0.5, 0.3, 0.2], pvs=[pv],
        depth=12, seldepth=18, nodes=1000, multipv=1,
    )


def _legal() -> np.ndarray:
    return np.array([uci_to_policy_index("e2e4", True)], dtype=np.int64)


def _p_sf() -> np.ndarray:
    p = np.zeros((POLICY_SIZE,), dtype=np.float32)
    p[int(_legal()[0])] = 1.0
    return p


def test_attach_requires_turn() -> None:
    with pytest.raises(TypeError):
        _attach_sf_target_to_record(  # pyright: ignore[reportCallIssue]
            _rec(), p_sf=_p_sf(), a_idx=int(_legal()[0]), res=_res(), legal_indices=_legal(),
        )


def test_attach_stamps_the_sparse_multipv_row() -> None:
    """The label and its provenance row are written together, never one alone."""
    rec = _rec()
    _attach_sf_target_to_record(
        rec, p_sf=_p_sf(), a_idx=int(_legal()[0]), res=_res(), legal_indices=_legal(),
        turn=True,
    )
    assert rec.sf_policy_target is not None
    assert rec.sf_multipv_raw is not None, "labelled row with no sf_multipv_raw reads as desync"


def test_last_record_wrapper_matches_the_direct_attach() -> None:
    direct = _rec()
    _attach_sf_target_to_record(
        direct, p_sf=_p_sf(), a_idx=int(_legal()[0]), res=_res(), legal_indices=_legal(),
        turn=True,
    )

    via_state = _rec()
    state = SimpleNamespace(samples_per_game=[[via_state]])
    _attach_sf_target_to_last_record(
        state, 0, p_sf=_p_sf(), a_idx=int(_legal()[0]), res=_res(),  # pyright: ignore[reportArgumentType]
        legal_indices=_legal(), turn=True,
    )

    np.testing.assert_array_equal(via_state.sf_policy_target, direct.sf_policy_target)
    np.testing.assert_array_equal(via_state.sf_legal_mask, direct.sf_legal_mask)
    np.testing.assert_array_equal(via_state.sf_wdl, direct.sf_wdl)
    assert via_state.sf_move_index == direct.sf_move_index
    np.testing.assert_array_equal(via_state.sf_multipv_raw, direct.sf_multipv_raw)


def test_last_record_wrapper_is_a_noop_without_records() -> None:
    state = SimpleNamespace(samples_per_game=[[]])
    _attach_sf_target_to_last_record(
        state, 0, p_sf=_p_sf(), a_idx=int(_legal()[0]), res=_res(),  # pyright: ignore[reportArgumentType]
        legal_indices=_legal(), turn=True,
    )
