"""Unit tests for the pure helpers in ``chess_anti_engine.selfplay.finalize``.

Full-path finalization is covered by ``test_play_batch_continuous.py``
end-to-end.  This file pins the small, easy-to-test units:

* ``_sf_terminal_result`` — POV flip + adjudication threshold dispatch
* ``_compute_volatility_and_sf_delta`` — single-pass volatility targets
  and log-only SF delta6 metric
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import chess
import numpy as np
import pytest

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.selfplay.finalize import (
    _compute_diff_focus_game_stats,
    _compute_volatility_and_sf_delta,
    _build_replay_samples,
    _finalization_hlgauss_target,
    _sf_terminal_result,
)
from chess_anti_engine.selfplay.state import _NetRecord, _StatsAcc


def _sf_res(wdl: list[float]) -> Mock:
    m = Mock()
    m.wdl = wdl
    return m


def test_cached_ternary_hlgauss_returns_independent_writable_rows() -> None:
    first = _finalization_hlgauss_target(1.0, num_bins=32, sigma=0.04)
    second = _finalization_hlgauss_target(1.0, num_bins=32, sigma=0.04)
    expected = second.copy()

    assert first.flags.writeable
    assert second.flags.writeable
    assert not np.shares_memory(first, second)
    first[0] = 42.0
    np.testing.assert_array_equal(second, expected)


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
        vol, sf_vol = _compute_volatility_and_sf_delta(state, records, {int(rec.ply_index): idx for idx, rec in enumerate(records)})
        assert vol == [None]
        assert sf_vol == [None]
        assert state.stats.sf_d6_n == 0

    def test_raw_volatility_uses_net_wdl(self):
        state = self._mk_state(volatility_source="raw")
        records = [
            _record(0, [0.6, 0.3, 0.1]),
            _record(6, [0.2, 0.3, 0.5]),
        ]
        vol, sf_vol = _compute_volatility_and_sf_delta(state, records, {int(rec.ply_index): idx for idx, rec in enumerate(records)})
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
        vol, _sf_vol = _compute_volatility_and_sf_delta(state, records, {int(rec.ply_index): idx for idx, rec in enumerate(records)})
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
        _compute_volatility_and_sf_delta(state, records, {int(rec.ply_index): idx for idx, rec in enumerate(records)})
        assert state.stats.sf_d6_n == 1
        assert state.stats.sf_d6_sum == pytest.approx(0.55)

    def test_sf_delta6_skipped_when_either_sf_wdl_missing(self):
        state = self._mk_state()
        records = [
            _record(0, [0.5, 0.4, 0.1], sf_wdl=[0.7, 0.2, 0.1]),
            _record(6, [0.5, 0.4, 0.1], sf_wdl=None),
        ]
        _compute_volatility_and_sf_delta(state, records, {int(rec.ply_index): idx for idx, rec in enumerate(records)})
        assert state.stats.sf_d6_n == 0


def test_build_replay_samples_records_producer_input_history_encoding() -> None:
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    policy[0] = 1.0
    record = _NetRecord(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_probs=policy,
        net_wdl_est=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        search_wdl_est=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        pov_color=chess.WHITE,
        ply_index=0,
        has_policy=True,
        priority=1.0,
        sample_weight=1.0,
        keep_prob=1.0,
    )
    state = cast(Any, SimpleNamespace(
        selfplay_arr=[False],
        starting_boards=[chess.Board()],
        opening_source_arr=["unit"],
        move_idx_history=[[]],
        rng=np.random.default_rng(1),
        game=SimpleNamespace(
            categorical_bins=32,
            hlgauss_sigma=0.04,
            max_plies=240,
            policy_encoding="az_4672",
            soft_policy_temp=1.0,
            input_history_encoding="lc0_root_legacy_meta",
            history_rep_fix=False,
            # `_build_replay_samples` re-reads this at the shard boundary (the
            # kill switch has to bind on the bytes, not only on the capture), so
            # a double that omits it stops exercising that branch. Production's
            # default, which is what these fixtures otherwise assume.
            record_prior_top1=True,
        ),
    ))

    samples = _build_replay_samples(
        state,
        0,
        [record],
        result="1-0",
        tb_policy_overrides={},
        vol_targets=[None],
        sf_vol_targets=[None],
        total_plies_played=1,
        ply_to_index={0: 0},
    )

    assert len(samples) == 1
    assert samples[0].input_history_encoding == "lc0_root_legacy_meta"


def _fastply_record(
    *,
    ply_index: int,
    has_policy: bool,
    sf_policy_target: np.ndarray | None = None,
    sf_multipv_raw: np.ndarray | None = None,
    keep_prob: float = 1.0,
    priority: float = 1.0,
) -> _NetRecord:
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    policy[0] = 1.0
    rec = _NetRecord(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_probs=policy,
        net_wdl_est=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        search_wdl_est=np.array([0.6, 0.3, 0.1], dtype=np.float32),
        pov_color=chess.WHITE,
        ply_index=ply_index,
        has_policy=has_policy,
        priority=priority,
        sample_weight=1.0,
        keep_prob=keep_prob,
        sf_policy_target=sf_policy_target,
    )
    rec.sf_multipv_raw = sf_multipv_raw
    return rec


def _fastply_state(
    *,
    record_fast_ply_value: bool,
    record_sf_p0_policy: bool = False,
    record_sf_p0_regret: bool = False,
) -> Any:
    return cast(Any, SimpleNamespace(
        selfplay_arr=[True],
        starting_boards=[chess.Board()],
        opening_source_arr=["unit"],
        move_idx_history=[[]],
        rng=np.random.default_rng(1),
        game=SimpleNamespace(
            categorical_bins=32,
            hlgauss_sigma=0.04,
            max_plies=240,
            policy_encoding="az_4672",
            soft_policy_temp=1.0,
            input_history_encoding="legacy",
            history_rep_fix=False,
            record_fast_ply_value=record_fast_ply_value,
            record_sf_p0_policy=record_sf_p0_policy,
            record_sf_p0_regret=record_sf_p0_regret,
            record_dense_sf_policy=True,
            # `_build_replay_samples` re-reads this at the shard boundary (the
            # kill switch has to bind on the bytes, not only on the capture), so
            # a double that omits it stops exercising that branch. Production's
            # default, which is what these fixtures otherwise assume.
            record_prior_top1=True,
        ),
    ))


def test_build_replay_samples_drops_fast_ply_records_by_default() -> None:
    records = [
        _fastply_record(ply_index=0, has_policy=True),
        _fastply_record(ply_index=1, has_policy=False),
    ]
    samples = _build_replay_samples(
        _fastply_state(record_fast_ply_value=False), 0, records,
        result="1-0", tb_policy_overrides={},
        vol_targets=[None, None], sf_vol_targets=[None, None],
        total_plies_played=2,
        ply_to_index={0: 0, 1: 1},
    )
    assert len(samples) == 1
    assert samples[0].has_policy is True


def test_build_replay_samples_keeps_fast_ply_records_as_value_rows() -> None:
    records = [
        _fastply_record(ply_index=0, has_policy=True),
        _fastply_record(ply_index=1, has_policy=False),
    ]
    samples = _build_replay_samples(
        _fastply_state(record_fast_ply_value=True), 0, records,
        result="1-0", tb_policy_overrides={},
        vol_targets=[None, None], sf_vol_targets=[None, None],
        total_plies_played=2,
        ply_to_index={0: 0, 1: 1},
    )
    assert len(samples) == 2
    fast = samples[1]
    assert fast.has_policy is False
    # Value-side targets are fully populated on the fast row: outcome WDL is
    # from the fast ply's own POV (white won, black to move at ply 1 -> loss),
    # search-root WDL is carried, and no SF fields are fabricated.
    assert fast.wdl_target == 0  # pov_color WHITE in this fixture, result 1-0
    assert fast.search_wdl is not None
    assert fast.sf_wdl is None
    assert fast.moves_left is not None
    # Value-only contract: no aux policy-head targets on the fast row.
    assert fast.policy_soft_target is None


def test_build_replay_samples_fast_ply_rows_carry_no_aux_policy_targets() -> None:
    """Value-only fast rows must not train aux policy heads: losses.py masks
    soft/future/sf_p0 by their own presence flags (has_policy_soft/has_future/
    has_sf_p0*), not by has_policy, so finalize must clear those targets."""
    sf_pol = np.zeros((POLICY_SIZE,), dtype=np.float32)
    sf_pol[0] = 1.0
    sf_raw = np.array(
        [[0, 50, 0, 500, 300], [1, 10, 0, 450, 300]], dtype=np.int32,
    )
    records = [
        _fastply_record(
            ply_index=0, has_policy=True,
            sf_policy_target=sf_pol, sf_multipv_raw=sf_raw,
        ),
        _fastply_record(
            ply_index=1, has_policy=True,
            sf_policy_target=sf_pol, sf_multipv_raw=sf_raw,
        ),
        _fastply_record(ply_index=2, has_policy=False),
        _fastply_record(ply_index=3, has_policy=True),
        _fastply_record(ply_index=4, has_policy=True),
    ]
    samples = _build_replay_samples(
        _fastply_state(
            record_fast_ply_value=True,
            record_sf_p0_policy=True,
            record_sf_p0_regret=True,
        ),
        0, records,
        result="1-0", tb_policy_overrides={},
        vol_targets=[None] * 5, sf_vol_targets=[None] * 5,
        total_plies_played=5,
        ply_to_index={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    )
    assert len(samples) == 5

    full = samples[1]  # full row, prev ply is a labeled full record
    assert full.has_policy is True
    assert full.policy_soft_target is not None
    assert full.future_policy_target is not None  # from ply 3
    assert full.has_future is True
    assert full.sf_p0_policy_target is not None  # prev ply 0 is labeled
    assert full.has_sf_p0 is True
    assert full.sf_p0_regret is not None
    assert full.has_sf_p0_regret is True

    fast = samples[2]  # fast row: same neighbors, all aux targets gated off
    assert fast.has_policy is False
    assert fast.policy_soft_target is None
    assert fast.future_policy_target is None  # ply 4 exists but row is gated
    assert fast.has_future is False
    assert fast.sf_p0_policy_target is None  # prev ply 1 labeled but gated
    assert fast.has_sf_p0 is False
    assert fast.sf_p0_regret is None
    assert fast.has_sf_p0_regret is False


def test_diff_focus_kept_excludes_fast_ply_value_rows() -> None:
    """diff_focus_records only counts policy-bearing records, so the kept
    count must exclude value-only fast rows too — otherwise games that keep
    fast rows report keep rates > 1.0 in the replay-filter telemetry."""
    records = [
        _fastply_record(ply_index=0, has_policy=True),
        _fastply_record(ply_index=1, has_policy=False),
        _fastply_record(ply_index=2, has_policy=True),
        _fastply_record(ply_index=3, has_policy=False),
    ]
    samples = _build_replay_samples(
        _fastply_state(record_fast_ply_value=True), 0, records,
        result="1-0", tb_policy_overrides={},
        vol_targets=[None] * 4, sf_vol_targets=[None] * 4,
        total_plies_played=4,
        ply_to_index={0: 0, 1: 1, 2: 2, 3: 3},
    )
    assert len(samples) == 4  # 2 policy rows + 2 value-only fast rows

    stats = _compute_diff_focus_game_stats(records, samples)
    assert stats.records == 2  # policy-bearing records only
    assert stats.kept == 2  # NOT len(samples) == 4: keep rate stays <= 1.0
    assert stats.kept <= stats.records


def test_fast_ply_value_rows_bypass_diff_focus_keep_prob() -> None:
    """keep_prob is a POLICY-difficulty subsample; value-only rows must not be
    filtered (or biased) by it — keep_prob=0.0 would drop every policy row but
    a fast value row must survive."""
    records = [
        _fastply_record(ply_index=0, has_policy=True, keep_prob=0.0),
        _fastply_record(ply_index=1, has_policy=False, keep_prob=0.0),
    ]
    samples = _build_replay_samples(
        _fastply_state(record_fast_ply_value=True), 0, records,
        result="1-0", tb_policy_overrides={},
        vol_targets=[None, None], sf_vol_targets=[None, None],
        total_plies_played=2,
        ply_to_index={0: 0, 1: 1},
    )
    assert len(samples) == 1
    assert samples[0].has_policy is False  # policy row dropped, value row kept


def test_fast_ply_value_rows_get_neutral_priority() -> None:
    """Fast-row difficulty scores come from the playout-capped search and are
    not calibrated against full-ply priorities; value-only rows enter the
    surprise-weighted sampler at a neutral 1.0."""
    records = [
        _fastply_record(ply_index=0, has_policy=True, priority=7.5),
        _fastply_record(ply_index=1, has_policy=False, priority=7.5),
    ]
    samples = _build_replay_samples(
        _fastply_state(record_fast_ply_value=True), 0, records,
        result="1-0", tb_policy_overrides={},
        vol_targets=[None, None], sf_vol_targets=[None, None],
        total_plies_played=2,
        ply_to_index={0: 0, 1: 1},
    )
    assert len(samples) == 2
    assert samples[0].priority == 7.5   # policy row keeps its difficulty score
    assert samples[1].priority == 1.0   # value-only row is neutral
