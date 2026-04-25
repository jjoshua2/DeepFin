"""Unit tests for the pure helpers in ``chess_anti_engine.selfplay.finalize``.

Full-path finalization is covered by ``test_play_batch_continuous.py``
end-to-end.  This file pins the small, easy-to-test units:

* ``_sf_terminal_result`` — POV flip + adjudication threshold dispatch
* ``_compute_volatility_and_sf_delta`` — single-pass volatility targets
  and log-only SF delta6 metric
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from chess_anti_engine.selfplay.finalize import (
    _compute_volatility_and_sf_delta,
    _sf_terminal_result,
)
from chess_anti_engine.selfplay.state import _NetRecord, _StatsAcc


def _sf_res(wdl: list[float]) -> Mock:
    m = Mock()
    m.wdl = wdl
    return m


class TestSfTerminalResult:
    """SF's wdl is reported from side-to-move POV; the helper must flip it
    to white-POV before thresholding."""

    def test_no_wdl_returns_draw(self):
        assert (
            _sf_terminal_result(
                turn_is_white=True, sf_res=None, adjudication_threshold=0.9,
            )
            == "1/2-1/2"
        )
        no_wdl = Mock()
        no_wdl.wdl = None
        assert (
            _sf_terminal_result(
                turn_is_white=False, sf_res=no_wdl, adjudication_threshold=0.9,
            )
            == "1/2-1/2"
        )

    def test_white_to_move_white_winning(self):
        # wdl from white POV (STM=white): strong white win.
        result = _sf_terminal_result(
            turn_is_white=True,
            sf_res=_sf_res([0.95, 0.04, 0.01]),
            adjudication_threshold=0.9,
        )
        assert result == "1-0"

    def test_black_to_move_white_winning_requires_flip(self):
        # wdl from black POV: L=0.95 means white wins. Flip required.
        result = _sf_terminal_result(
            turn_is_white=False,
            sf_res=_sf_res([0.01, 0.04, 0.95]),
            adjudication_threshold=0.9,
        )
        assert result == "1-0"

    def test_white_to_move_black_winning(self):
        result = _sf_terminal_result(
            turn_is_white=True,
            sf_res=_sf_res([0.01, 0.04, 0.95]),
            adjudication_threshold=0.9,
        )
        assert result == "0-1"

    def test_below_threshold_is_draw(self):
        result = _sf_terminal_result(
            turn_is_white=True,
            sf_res=_sf_res([0.7, 0.25, 0.05]),  # high W but below 0.9
            adjudication_threshold=0.9,
        )
        assert result == "1/2-1/2"

    def test_threshold_is_strict_less_than(self):
        """The bound is strict: exactly at threshold -> draw."""
        result = _sf_terminal_result(
            turn_is_white=True,
            sf_res=_sf_res([0.9, 0.09, 0.01]),
            adjudication_threshold=0.9,
        )
        assert result == "1/2-1/2"


def _record(
    ply: int,
    net_wdl: list[float],
    search_wdl: list[float] | None = None,
    sf_wdl: list[float] | None = None,
) -> _NetRecord:
    return _NetRecord(
        x=np.zeros((1,), dtype=np.float32),
        policy_probs=np.zeros((1,), dtype=np.float32),
        net_wdl_est=np.asarray(net_wdl, dtype=np.float32),
        search_wdl_est=np.asarray(
            search_wdl if search_wdl is not None else net_wdl, dtype=np.float32,
        ),
        pov_color=True,
        ply_index=ply,
        has_policy=True,
        priority=1.0,
        sample_weight=1.0,
        keep_prob=1.0,
        sf_wdl=(np.asarray(sf_wdl, dtype=np.float32) if sf_wdl is not None else None),
    )


class TestComputeVolatilityAndSfDelta:
    def _mk_state(self, volatility_source: str = "raw"):
        """Lightweight state surrogate exposing just the fields the helper
        reads.  A full SelfplayState is overkill for this pure function."""
        state = Mock()
        state.volatility_source = volatility_source
        state.stats = _StatsAcc()
        return state

    def test_no_pairing_returns_none_targets(self):
        state = self._mk_state()
        # Only one record → no pair at ply+6 → all targets are None.
        records = [_record(0, [0.5, 0.4, 0.1])]
        vol, sf_vol = _compute_volatility_and_sf_delta(state, records)
        assert vol == [None]
        assert sf_vol == [None]
        assert state.stats.sf_d6_n == 0

    def test_raw_volatility_uses_net_wdl(self):
        state = self._mk_state(volatility_source="raw")
        records = [
            _record(0, [0.6, 0.3, 0.1]),
            _record(6, [0.2, 0.3, 0.5]),
        ]
        vol, sf_vol = _compute_volatility_and_sf_delta(state, records)
        # |[0.6,0.3,0.1] - [0.2,0.3,0.5]| = [0.4, 0.0, 0.4]
        assert vol[0] == pytest.approx([0.4, 0.0, 0.4])
        assert vol[1] is None
        assert sf_vol == [None, None]

    def test_search_volatility_uses_search_wdl(self):
        state = self._mk_state(volatility_source="search")
        records = [
            _record(0, [0.5, 0.4, 0.1], search_wdl=[0.8, 0.1, 0.1]),
            _record(6, [0.5, 0.4, 0.1], search_wdl=[0.2, 0.2, 0.6]),
        ]
        vol, _sf_vol = _compute_volatility_and_sf_delta(state, records)
        # The helper reads search_wdl_est when volatility_source == "search",
        # so the diff should come from those values (not net_wdl).
        assert vol[0] == pytest.approx([0.6, 0.1, 0.5])

    def test_sf_delta6_sums_absolute_winrate_deltas(self):
        state = self._mk_state()
        # Pair at ply=0 with ply=6. SF winrate-like = W + 0.5 * D.
        # r0: 0.7 + 0.5*0.2 = 0.8
        # r6: 0.1 + 0.5*0.3 = 0.25
        # delta = 0.55
        records = [
            _record(0, [0.5, 0.4, 0.1], sf_wdl=[0.7, 0.2, 0.1]),
            _record(6, [0.5, 0.4, 0.1], sf_wdl=[0.1, 0.3, 0.6]),
        ]
        _compute_volatility_and_sf_delta(state, records)
        assert state.stats.sf_d6_n == 1
        assert state.stats.sf_d6_sum == pytest.approx(0.55)

    def test_sf_delta6_skipped_when_either_sf_wdl_missing(self):
        state = self._mk_state()
        records = [
            _record(0, [0.5, 0.4, 0.1], sf_wdl=[0.7, 0.2, 0.1]),
            _record(6, [0.5, 0.4, 0.1], sf_wdl=None),
        ]
        _compute_volatility_and_sf_delta(state, records)
        assert state.stats.sf_d6_n == 0
