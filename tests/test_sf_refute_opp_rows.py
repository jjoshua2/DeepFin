"""Flag-gated SF-refute opp-row recording + full-node refute moves (staged).

All three flags default off; these tests cover the default no-op, the
full-node move override, opp-row emission (policy target / masks / sf_wdl /
POV), finalize outcome-WDL POV, and the net-blend config rejection.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.finalize import _build_sf_refute_opp_sample
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.selfplay.stockfish_turn import _eff_sf_nodes, _process_sf_results
from chess_anti_engine.tune.trial_config import (
    TrialConfig,
    _validate_sf_refute_net_blend,
)


def _pv(uci: str, wdl: list[float]) -> SimpleNamespace:
    return SimpleNamespace(move_uci=uci, cp=None, mate=None, wdl=wdl)


def _fake_black_res() -> SimpleNamespace:
    """SF result for black to move after 1.e4 (e7e5 best)."""
    return SimpleNamespace(
        bestmove_uci="e7e5",
        cp=None,
        mate=None,
        wdl=[0.55, 0.25, 0.2],
        pvs=[_pv("e7e5", [0.55, 0.25, 0.2]), _pv("c7c5", [0.45, 0.25, 0.3])],
    )


def _refute_state(
    *,
    record_opp_rows: bool,
    plies_left: int = 3,
    net_blend: float = 0.0,
) -> SelfplayState:
    """SelfplayState with one selfplay-tagged SF-refute slot, black = SF seat."""
    n = 1
    board = chess.Board()
    board.push_uci("e2e4")  # white (net) moved; black (SF opp) to move
    opening = OpeningConfig(
        opening_fen_sf_refute_frac=0.5,
        opening_fen_sf_refute_plies=plies_left,
        sf_refute_record_opp_rows=record_opp_rows,
        sf_refute_opp_policy_net_blend=net_blend,
    )
    return SelfplayState(
        device="cpu",
        rng=np.random.default_rng(0),
        stockfish=MagicMock(),
        evaluator=MagicMock(),
        model=None,
        opponent=OpponentConfig(),
        temp=TemperatureConfig(),
        search=SearchConfig(),
        opening=opening,
        diff_focus=DiffFocusConfig(),
        diff_focus_norm=None,
        game=GameConfig(),
        batch_size=n,
        continuous=False,
        target=1,
        volatility_source="raw",
        base_nodes=1000,
        terminal_eval_nodes=1000,
        done_arr=np.zeros(n, dtype=np.int8),
        finalized_arr=np.zeros(n, dtype=np.int8),
        net_color_arr=np.ones(n, dtype=np.int8),  # net = white
        selfplay_arr=np.ones(n, dtype=np.int8),   # selfplay-tagged
        opening_source_arr=["fenlist_sf_refute"],
        boards=[board],
        cboards=[CBoard.from_board(board)],
        starting_boards=None,
        starting_ply_arr=np.zeros(n, dtype=np.int32),
        move_idx_history=[[] for _ in range(n)],
        samples_per_game=[[] for _ in range(n)],
        consecutive_low_winrate=[0] * n,
        last_net_full=[True] * n,
        force_full_next=[False] * n,
        root_ids=[-1] * n,
        pending_sf_labels=[],
        pending_sf_moves={},
        sf_label_escalations=[0] * n,
        tb_probe=None,
        tb_result_arr=[None] * n,
        tb_adj_roll_arr=np.zeros(n, dtype=np.int8),
        fen_dole_queue=None,
        fen_sf_refute_queue=None,
        sf_refute_opp_plies_left=np.array([plies_left], dtype=np.int32),
        games_started=1,
        games_completed=0,
        mcts_tree=None,
    )


# ── (b) full_node_moves flips _eff_sf_nodes for in-refute slots only ─────────

def _eff_nodes_state(*, full_node_moves: bool, plies_left: int) -> Any:
    return SimpleNamespace(
        base_nodes=1000,
        last_net_full=[False],  # fast ply → would normally scale 0.25×
        game=GameConfig(),      # sf_fast_ply_node_scale=0.25, sf_move_nodes=0
        opening=OpeningConfig(sf_refute_full_node_moves=full_node_moves),
        sf_refute_opp_plies_left=np.array([plies_left], dtype=np.int32),
    )


def test_full_node_moves_off_keeps_fast_scale() -> None:
    st = _eff_nodes_state(full_node_moves=False, plies_left=3)
    # Flag off: fast-ply move query stays scaled (0.25 × 1000).
    assert _eff_sf_nodes(st, 0, for_move=True) == 250


def test_full_node_moves_on_uses_full_nodes_in_refute() -> None:
    st = _eff_nodes_state(full_node_moves=True, plies_left=3)
    assert _eff_sf_nodes(st, 0, for_move=True) == 1000


def test_full_node_moves_on_but_not_in_refute_keeps_scale() -> None:
    st = _eff_nodes_state(full_node_moves=True, plies_left=0)  # not in refute
    assert _eff_sf_nodes(st, 0, for_move=True) == 250


def test_full_node_moves_does_not_touch_label_queries() -> None:
    st = _eff_nodes_state(full_node_moves=True, plies_left=3)
    # Label queries are never move queries → still scaled even in refute.
    assert _eff_sf_nodes(st, 0, for_label=True) == 250


# ── (a) default-off no-op: refute step emits exactly today's rows ────────────

def test_default_off_emits_no_opp_row() -> None:
    st = _refute_state(record_opp_rows=False)
    _process_sf_results(
        st, [0], results={0: _fake_black_res()},
        play_curriculum_moves=True, attach_labels=False,
    )
    # No training row is created for the SF-played ply (today's behaviour).
    assert st.samples_per_game[0] == []
    # The SF move was still pushed and the countdown decremented.
    assert int(st.sf_refute_opp_plies_left[0]) == 2


# ── (c) opp-row emission: policy target / masks / sf_wdl / POV ───────────────

def test_opp_row_emitted_with_correct_targets_and_pov() -> None:
    st = _refute_state(record_opp_rows=True)
    res = _fake_black_res()
    _process_sf_results(
        st, [0], results={0: res},
        play_curriculum_moves=True, attach_labels=False,
    )
    recs = st.samples_per_game[0]
    assert len(recs) == 1
    rec = recs[0]
    assert bool(rec.is_sf_refute_opp) is True
    assert bool(rec.has_policy) is True
    # POV is the SF seat (black to move here) → no wdl flip on the label.
    assert rec.pov_color == chess.BLACK
    np.testing.assert_allclose(rec.sf_wdl, np.array([0.55, 0.25, 0.2]), atol=1e-6)
    # MAIN policy target: soft SF MultiPV distribution, argmax at e7e5.
    probs = np.asarray(rec.policy_probs, dtype=np.float64)
    assert probs.shape == (POLICY_SIZE,)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    e7e5 = uci_to_policy_index("e7e5", False)
    assert int(np.argmax(probs)) == e7e5
    # Aux-head record fields left absent (masked downstream).
    assert rec.sf_policy_target is None
    assert rec.sf_move_index is None
    # Countdown still advanced.
    assert int(st.sf_refute_opp_plies_left[0]) == 2


def test_opp_row_finalize_masks_and_outcome_pov() -> None:
    st = _refute_state(record_opp_rows=True)
    _process_sf_results(
        st, [0], results={0: _fake_black_res()},
        play_curriculum_moves=True, attach_labels=False,
    )
    rec = st.samples_per_game[0][0]
    fake: Any = SimpleNamespace(game=GameConfig())

    # (d) POV: black seat, white wins → the row is a LOSS for the SF seat.
    s_win = _build_sf_refute_opp_sample(
        fake, rec, result="1-0", game_id=7, is_selfplay_slot=True,
    )
    assert s_win.wdl_target == 2  # loss from black POV
    # Black wins → win for the seat.
    s_loss = _build_sf_refute_opp_sample(
        fake, rec, result="0-1", game_id=7, is_selfplay_slot=True,
    )
    assert s_loss.wdl_target == 0

    # Trained heads present; every aux head masked.
    assert s_win.has_policy is True
    assert s_win.is_network_turn is True  # so MAIN policy CE + wdl loss fire
    assert s_win.is_selfplay is True      # stays out of PID via fenlist source
    assert s_win.sf_wdl is not None
    assert s_win.categorical_target is not None
    assert s_win.legal_mask is not None
    assert s_win.policy_soft_target is None
    assert s_win.future_policy_target is None
    assert s_win.sf_policy_target is None
    assert s_win.volatility_target is None
    assert s_win.sf_volatility_target is None
    assert s_win.moves_left is None
    assert s_win.search_wdl is None


# ── (e) blend > 0 rejected at config validation ──────────────────────────────

def test_net_blend_zero_ok() -> None:
    assert _validate_sf_refute_net_blend(0.0) == 0.0


def test_net_blend_positive_rejected() -> None:
    with pytest.raises(ValueError, match="not yet wired"):
        _validate_sf_refute_net_blend(0.25)


def test_net_blend_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="must be in"):
        _validate_sf_refute_net_blend(1.5)


def test_trial_config_rejects_positive_blend() -> None:
    with pytest.raises(ValueError, match="not yet wired"):
        TrialConfig.from_dict({"sf_refute_opp_policy_net_blend": 0.5})


def test_trial_config_default_blend_ok() -> None:
    cfg = TrialConfig.from_dict({})
    assert cfg.sf_refute_opp_policy_net_blend == 0.0
    assert cfg.sf_refute_full_node_moves is False
    assert cfg.sf_refute_record_opp_rows is False
