from __future__ import annotations

import chess
import numpy as np

from chess_anti_engine.selfplay.finalize import _stable_game_id, _suffix_sf_regret_features
from chess_anti_engine.selfplay.state import _NetRecord


def _record(regret: float | None, ply_index: int) -> _NetRecord:
    rec = _NetRecord(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_probs=np.ones((1,), dtype=np.float32),
        net_wdl_est=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        search_wdl_est=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        pov_color=chess.WHITE,
        ply_index=ply_index,
        has_policy=True,
        priority=1.0,
        sample_weight=1.0,
        keep_prob=1.0,
    )
    rec.sf_played_regret = regret
    return rec


def test_suffix_sf_regret_features_summarize_future_opponent_loss():
    records = [_record(0.01, 0), _record(None, 2), _record(0.02, 4)]

    features = _suffix_sf_regret_features(records, is_selfplay=False)
    assert np.allclose([f.total for f in features], [0.03, 0.02, 0.02])
    assert np.allclose([f.d95 for f in features], [0.01 + 0.95**2 * 0.02, 0.95 * 0.02, 0.02])
    assert np.allclose([f.d98 for f in features], [0.01 + 0.98**2 * 0.02, 0.98 * 0.02, 0.02])
    assert np.allclose([f.max_single for f in features], [0.02, 0.02, 0.02])
    assert np.allclose([f.h4 for f in features], [0.01, 0.02, 0.02])
    assert np.allclose([f.h6 for f in features], [0.03, 0.02, 0.02])
    assert np.allclose([f.h12 for f in features], [0.03, 0.02, 0.02])
    assert [f.count for f in features] == [2, 1, 1]

    selfplay = _suffix_sf_regret_features(records, is_selfplay=True)
    assert np.allclose([f.total for f in selfplay], [0.0, 0.0, 0.0])
    assert np.allclose([f.h12 for f in selfplay], [0.0, 0.0, 0.0])


def test_stable_game_id_includes_opening_identity():
    base = _stable_game_id(
        start_fen="fen-a",
        opening_source="book1",
        move_trace="12,34,56",
        result="1-0",
        total_plies_played=42,
    )

    assert base != _stable_game_id(
        start_fen="fen-b",
        opening_source="book1",
        move_trace="12,34,56",
        result="1-0",
        total_plies_played=42,
    )
    assert base != _stable_game_id(
        start_fen="fen-a",
        opening_source="book2",
        move_trace="12,34,56",
        result="1-0",
        total_plies_played=42,
    )
