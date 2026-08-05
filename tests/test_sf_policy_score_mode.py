"""sf_policy_score_mode="cp": the non-saturating SF policy target.

The 2026-07-31 screen (ledger "policy-target saturation", sustained by the
2026-08-04 conjugacy audit) showed the production w+0.5*d score saturates in
decisive positions — every candidate scores ~1.0, the softmax cannot rank
them, and the net learns to shuffle instead of convert. cp mode scores raw
effective centipawns (mate folded) at a centipawn-units temperature.

These tests pin the three ways the deploy could silently not happen:
  * the scorer itself (wrong units, wrong temp, wrong mate precedence),
  * the rebuild path drifting from the live path (capture/rebuild parity),
  * the config route (worker restart/resume classification, log line).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from chess_anti_engine.replay.shard import SF_CP_SENTINEL
from chess_anti_engine.stockfish.wdl import mate_to_effective_cp
from chess_anti_engine.train.target_builder import (
    SfTargetParams,
    rebuild_sf_policy_target,
    rebuild_sf_policy_targets_batch,
)


def _pv(move_uci: str, cp: int | None, mate: int | None) -> SimpleNamespace:
    return SimpleNamespace(move_uci=move_uci, cp=cp, mate=mate, wdl=None)


def test_pv_cp_score_units_and_mate_precedence() -> None:
    from chess_anti_engine.selfplay.stockfish_turn import _pv_cp_score

    assert _pv_cp_score(_pv("a", 150, None)) == 150.0
    assert _pv_cp_score(_pv("a", -40, None)) == -40.0
    # mate wins over cp, folded exactly like cp_to_wdl's precedence
    assert _pv_cp_score(_pv("a", 150, 4)) == mate_to_effective_cp(4)
    assert _pv_cp_score(_pv("a", None, -3)) == mate_to_effective_cp(-3)
    # nothing to score
    assert _pv_cp_score(_pv("a", None, None)) is None


def test_cp_mode_ranks_won_positions_where_wdl_saturates() -> None:
    """The reason the mode exists: +2500 vs +1500 cp (a queen of winning
    margin) is invisible to the saturating score at production logistic
    params, and decisive to cp."""
    from chess_anti_engine.selfplay.stockfish_turn import (
        _build_sf_policy_target,
        _pv_cp_score,
        _pv_wdl_score,
    )

    pvs = [(10, _pv("a", 2500, None)), (20, _pv("b", 1500, None))]
    legal = np.array([10, 20], dtype=np.int64)

    # production wdl params (pbt2_small.yaml): logistic slope 0.006, draw 120
    wdl_scores = [
        _pv_wdl_score(
            pv, sf_wdl_use_cp_logistic=True,
            sf_wdl_cp_slope=0.006, sf_wdl_cp_draw_width=120.0,
        )
        for _a, pv in pvs
    ]
    cp_scores = [_pv_cp_score(pv) for _a, pv in pvs]
    assert None not in wdl_scores
    assert None not in cp_scores

    p_wdl = _build_sf_policy_target(
        [a for a, _pv_ in pvs], [float(s) for s in wdl_scores if s is not None],
        legal_indices=legal,
        sf_policy_temp=0.012, sf_policy_label_smooth=0.0,
    )
    p_cp = _build_sf_policy_target(
        [a for a, _pv_ in pvs], [float(s) for s in cp_scores if s is not None],
        legal_indices=legal,
        sf_policy_temp=16.2, sf_policy_label_smooth=0.0,
    )
    # saturating score: both moves ~1.0 -> near-uniform target
    assert float(p_wdl[10]) < 0.75, "wdl score unexpectedly ranks a won position"
    # cp score: 1000cp gap at temp 16.2 -> essentially one-hot
    assert float(p_cp[10]) > 0.99


def test_score_params_resolver_picks_temp_by_mode() -> None:
    from chess_anti_engine.selfplay.stockfish_turn import _sf_policy_score_params

    game_wdl = SimpleNamespace(
        sf_policy_score_mode="wdl", sf_policy_temp=0.012, sf_policy_cp_temp=16.2,
    )
    game_cp = SimpleNamespace(
        sf_policy_score_mode="cp", sf_policy_temp=0.012, sf_policy_cp_temp=16.2,
    )
    assert _sf_policy_score_params(game_wdl) == ("wdl", 0.012)
    assert _sf_policy_score_params(game_cp) == ("cp", 16.2)
    # absent attribute (old GameConfig pickles) -> wdl, never a crash
    legacy = SimpleNamespace(sf_policy_temp=0.012)
    assert _sf_policy_score_params(legacy) == ("wdl", 0.012)


def test_move_selection_call_site_stays_in_wdl_units() -> None:
    """The curriculum MOVE-selection candidates feed the PID's wdl_regret
    band, which is defined in win-fraction units — cp mode must never reach
    that call. Pin the call site itself: the first _collect_sf_pv_candidates
    call in _process_sf_results (move selection) takes no score-mode kwarg,
    while the label branch's does."""
    import inspect

    from chess_anti_engine.selfplay import stockfish_turn

    src = inspect.getsource(stockfish_turn._process_sf_results)
    calls = src.split("_collect_sf_pv_candidates(")[1:]
    assert len(calls) == 2, "call-site count changed; re-audit score-mode routing"
    move_call, label_call = calls[0], calls[1]
    assert "sf_policy_score_mode" not in move_call.split(")")[0], (
        "move-selection candidates must stay w+0.5d: the PID wdl_regret band "
        "is win-fraction units"
    )
    assert "sf_policy_score_mode=score_mode" in label_call.split(")")[0]


@pytest.mark.parametrize("with_mate", [False, True])
def test_rebuild_matches_live_construction_cp_mode(with_mate: bool) -> None:
    """Capture/rebuild parity in cp mode, the same invariant the wdl modes
    already pin (test_sparse_multipv_labels)."""
    from chess_anti_engine.selfplay.stockfish_turn import (
        _build_sf_policy_target,
        _pv_cp_score,
    )

    rows = np.full((4, 5), -1, dtype=np.int16)
    rows[0] = (100, 50, 0, 600, 300)
    rows[1] = (200, -20, 0, 450, 320)
    rows[2] = (300, SF_CP_SENTINEL, 4 if with_mate else 0, 990, 10)
    legal = np.array([100, 200, 300, 400], dtype=np.int64)
    params = SfTargetParams(
        sf_policy_temp=0.012, sf_policy_label_smooth=0.01,
        sf_policy_score_mode="cp", sf_policy_cp_temp=16.2,
    )

    cand_idxs, cand_scores = [], []
    for move_idx, cp, mate, _w, _d in rows[rows[:, 0] >= 0].tolist():
        score = _pv_cp_score(
            _pv("0000", None if cp == SF_CP_SENTINEL else int(cp),
                None if mate == 0 else int(mate))
        )
        if score is None:
            continue
        cand_idxs.append(int(move_idx))
        cand_scores.append(score)
    # the mate-less sentinel row scores nothing in cp mode
    assert len(cand_idxs) == (3 if with_mate else 2)

    live = _build_sf_policy_target(
        cand_idxs, cand_scores, legal_indices=legal,
        sf_policy_temp=16.2, sf_policy_label_smooth=0.01,
    )
    rebuilt = rebuild_sf_policy_target(
        rows, legal_indices=legal, policy_size=4672, params=params,
    )
    assert rebuilt is not None
    np.testing.assert_allclose(rebuilt, live, atol=1e-5)

    batch, ok = rebuild_sf_policy_targets_batch(
        rows[None, ...], legal_dense=None, policy_size=4672, params=params,
    )
    assert bool(ok[0])
    # legal_dense=None disables smoothing; compare against an unsmoothed scalar
    unsmoothed = rebuild_sf_policy_target(
        rows, legal_indices=np.zeros((0,), dtype=np.int64), policy_size=4672,
        params=params,
    )
    assert unsmoothed is not None
    np.testing.assert_allclose(batch[0], unsmoothed, atol=1e-6)


def test_cp_mode_ignores_native_wdl_only_rows() -> None:
    """A row with native WDL but no cp/mate is unscoreable in cp mode — it
    must drop out, not score 0 cp (0 would rank it as 'equal')."""
    rows = np.full((2, 5), -1, dtype=np.int16)
    rows[0] = (100, 120, 0, 600, 300)
    rows[1] = (200, SF_CP_SENTINEL, 0, 450, 320)   # wdl-only row
    params = SfTargetParams(sf_policy_score_mode="cp", sf_policy_cp_temp=16.2)
    rebuilt = rebuild_sf_policy_target(
        rows, legal_indices=np.array([100, 200], dtype=np.int64),
        policy_size=4672, params=params,
    )
    assert rebuilt is not None
    assert float(rebuilt[100]) > 0.9
    # only smoothing mass may remain on the unscoreable move
    assert float(rebuilt[200]) < 0.05


def test_sparse_ce_matches_dense_soft_ce_in_cp_mode() -> None:
    """The train-time sparse-CE path (sf_policy_sparse_ce, default OFF) builds
    the target from sf_multipv_raw with its own torch scorer. If it ignored
    the score mode, flipping sparse CE on under cp mode would silently train
    against a DIFFERENT target than the stored one — the capture/loss
    divergence class the dense-parity tests exist for."""
    import torch

    from chess_anti_engine.train.losses import soft_cross_entropy
    from chess_anti_engine.train.sparse_sf_ce import sparse_sf_policy_ce

    width = 4672
    params = SfTargetParams(
        sf_policy_label_smooth=0.01,
        sf_policy_score_mode="cp", sf_policy_cp_temp=16.2,
    )
    raw = np.full((2, 8, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[0, 0] = (100, 250, 0, 600, 300)
    raw[0, 1] = (200, -20, 0, 450, 320)
    raw[0, 2] = (300, SF_CP_SENTINEL, 4, 990, 10)   # mate-in-4
    raw[0, 3] = (400, SF_CP_SENTINEL, 0, 500, 300)  # wdl-only: unscoreable in cp
    raw[1, 0] = (150, 60, 0, 700, 200)
    legal_idx = np.array([100, 150, 200, 300, 400], dtype=np.int64)
    legal = np.zeros((2, width), np.float32)
    legal[:, legal_idx] = 1.0

    dense = np.zeros((2, width), np.float32)
    for i in range(2):
        rebuilt = rebuild_sf_policy_target(
            raw[i], legal_indices=legal_idx, policy_size=width, params=params,
        )
        assert rebuilt is not None
        dense[i] = rebuilt

    torch.manual_seed(0)
    logits = torch.randn(2, width)
    batch = {
        "sf_multipv_raw": torch.from_numpy(raw.astype(np.int32)),
        "has_sf_multipv_raw": torch.ones(2),
        "sf_legal_mask": torch.from_numpy(legal),
        "has_sf_legal_mask": torch.ones(2),
        "sf_move_index": torch.tensor([100, 150]),
        "has_sf_move": torch.ones(2),
    }
    dense_ce = soft_cross_entropy(logits, torch.from_numpy(dense))
    sparse_ce, ok = sparse_sf_policy_ce(
        logits, batch, params=params, legal_aligned=batch["sf_legal_mask"],
    )
    assert ok.tolist() == [1.0, 1.0]
    torch.testing.assert_close(sparse_ce, dense_ce, atol=1e-5, rtol=1e-5)


def test_score_mode_is_validated_everywhere() -> None:
    from chess_anti_engine.selfplay.config import GameConfig

    with pytest.raises(ValueError, match="sf_policy_score_mode"):
        GameConfig(sf_policy_score_mode="cp_rank")
    with pytest.raises(ValueError, match="sf_policy_cp_temp"):
        GameConfig(sf_policy_score_mode="cp", sf_policy_cp_temp=0.0)
    with pytest.raises(ValueError, match="sf_policy_score_mode"):
        SfTargetParams(sf_policy_score_mode="cpgap")


def test_worker_classifies_the_new_keys() -> None:
    """Restart-keyed (target semantics are stamped at label time, not live)
    AND resume-fingerprinted (a resumed game's old plies were labelled under
    the other mode — mixing them in one shard is the defect class
    _RESUME_COMPAT_KEYS exists for)."""
    from chess_anti_engine.worker import WorkerSession

    for key in ("sf_policy_score_mode", "sf_policy_cp_temp"):
        assert key in WorkerSession._RECO_RESTART_KEYS
        assert key in WorkerSession._RESUME_COMPAT_KEYS


def test_session_start_log_line_names_the_score_mode() -> None:
    """PR #354's R2 rule applied to this knob: the session-start reco line is
    the deploy-verification instrument. The format string lives inline in
    _run_selfplay, so pin the source — dropping score_mode= from it must fail
    a test."""
    import inspect

    from chess_anti_engine import worker

    src = inspect.getsource(worker)
    assert "session-start reco applied" in src
    start = src.index("session-start reco applied")
    fmt_region = src[start:start + 400]
    msg = (
        "the session-start reco line no longer reports the policy-target "
        "score mode — the deploy proof for sf_policy_score_mode"
    )
    assert "score_mode=%s" in fmt_region, msg
    assert "cp_temp=%.2f" in fmt_region, msg


def test_trial_config_and_publisher_carry_the_keys() -> None:
    """The E13/H2 defect class: a key that parses but is never published, or
    publishes but never reaches GameConfig. Walk yaml -> TrialConfig ->
    GameConfig sync -> reco publisher dict."""
    from chess_anti_engine.tune.trial_config import TrialConfig

    tc = TrialConfig.from_dict(
        {"sf_policy_score_mode": "cp", "sf_policy_cp_temp": 20.0}
    )
    assert tc.sf_policy_score_mode == "cp"
    assert tc.sf_policy_cp_temp == 20.0

    from chess_anti_engine.utils.config_yaml import SELFPLAY_CONFIG_KEYS

    assert "sf_policy_score_mode" in SELFPLAY_CONFIG_KEYS
    assert "sf_policy_cp_temp" in SELFPLAY_CONFIG_KEYS
