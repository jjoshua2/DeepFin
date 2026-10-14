"""Opt-in refute rows carry the actual board relation input through replay."""
import dataclasses

import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.replay.shard import arrays_to_samples, samples_to_arrays
from chess_anti_engine.selfplay.resume import _rebuild_record
from tests.test_sf_refute_opp_rows import (
    _build_sf_refute_opp_sample,
    _fake_black_res,
    _process_sf_results,
    _refute_state,
)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("history", ["legacy", "lc0_root"])
def test_refute_live_resume_and_replay_use_same_position_relations(enabled, history):
    state = _refute_state(record_opp_rows=True)
    state.game = dataclasses.replace(
        state.game, record_relations=enabled, input_history_encoding=history,
    )
    before = CBoard.from_board(state.boards[0])
    expected = before.compute_relations()
    assert np.count_nonzero(expected) > 0
    _process_sf_results(
        state, [0], results={0: _fake_black_res()},
        play_curriculum_moves=True, attach_labels=False,
    )
    rec = state.samples_per_game[0][0]
    restored = _rebuild_record(
        state, before, {name: getattr(rec, name) for name in rec.__slots__},
        want_lc0_root_alt=False, want_relations=enabled,
    )
    sample = _build_sf_refute_opp_sample(
        state, rec, result="1-0", game_id=7, is_selfplay_slot=True,
    )
    arrays = samples_to_arrays([sample])
    roundtrip = arrays_to_samples(arrays)[0]
    if enabled:
        for relations in (rec.relations, restored.relations, sample.relations, roundtrip.relations):
            np.testing.assert_array_equal(relations, expected)
        assert arrays["has_relations"].tolist() == [1]
        batch = collate_arrays(arrays, device="cpu")
        np.testing.assert_array_equal(batch["relations"][0].numpy(), expected)
    else:
        assert rec.relations is restored.relations is sample.relations is roundtrip.relations is None
        assert arrays["has_relations"].tolist() == [0]
