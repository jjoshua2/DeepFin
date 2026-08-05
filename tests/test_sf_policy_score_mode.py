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


def test_score_mode_reaches_every_label_site_and_never_move_selection() -> None:
    """Module-wide call-site accounting (PR #355 review F2: one-line kwarg
    deletions at individual sites survived a narrower version of this test).

    Every _collect_sf_pv_candidates call that feeds a LABEL build must pass
    the score mode; the single move-selection call must NOT (the PID's
    wdl_regret band is win-fraction units). Every _build_sf_policy_target
    label call must take the mode-selected temp, never the raw wdl temp.
    """
    import inspect

    from chess_anti_engine.selfplay import stockfish_turn

    src = inspect.getsource(stockfish_turn)

    def _call_args(text: str) -> str:
        """Text of one call's arguments, balanced-paren aware (the calls
        contain nested bool()/float() casts)."""
        depth = 1
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[:i]
        raise AssertionError("unbalanced parens in scanned source")

    collect_calls = src.split("_collect_sf_pv_candidates(")
    # element 0 is pre-text, element 1 follows the def; the rest are calls
    collect_args = [_call_args(c) for c in collect_calls[2:]]
    assert len(collect_args) == 4, (
        f"expected 4 _collect_sf_pv_candidates call sites, found "
        f"{len(collect_args)} — re-audit score-mode routing at the new site"
    )
    with_mode = [a for a in collect_args if "sf_policy_score_mode" in a]
    without_mode = [a for a in collect_args if "sf_policy_score_mode" not in a]
    assert len(with_mode) == 3, "a label site dropped the score-mode kwarg"
    assert len(without_mode) == 1, (
        "move-selection candidates must stay w+0.5d: the PID wdl_regret band "
        "is win-fraction units"
    )

    build_calls = src.split("_build_sf_policy_target(")
    build_args = [_call_args(c) for c in build_calls[2:]]
    assert len(build_args) == 3, (
        f"expected 3 _build_sf_policy_target call sites, found "
        f"{len(build_args)} — thread the mode-selected temp at the new site"
    )
    for a in build_args:
        assert "sf_policy_temp=score_temp" in a, (
            "a label build reverted to a raw temp instead of the "
            "mode-selected score_temp"
        )


def _label_state_cp(legal: np.ndarray):
    """Real-path state stub (borrowed from test_stockfish_label_gating) with
    cp scoring configured at production-adjacent params."""
    from chess_anti_engine.selfplay.config import GameConfig, OpponentConfig
    from tests.test_stockfish_label_gating import _state

    state = _state(
        has_policy=True,
        legal_indices=legal,
        game=GameConfig(
            sf_policy_score_mode="cp",
            sf_policy_cp_temp=16.2,
            sf_policy_temp=0.012,
            sf_policy_label_smooth=0.0,
        ),
        opponent=OpponentConfig(wdl_regret_limit=0.0),
    )
    state.opening = SimpleNamespace(sf_refute_opp_policy_net_blend=0.0)
    return state


# softmax([250, 180]/16.2) — the analytic cp-mode target. This single number
# kills both one-line reverts at a site: mode deletion scores the EQUAL wdl
# arrays below (target ~0.5), temp reversion divides cp by 0.012 (target ~1.0).
_P_BEST_250_180 = 1.0 / (1.0 + np.exp(-(250.0 - 180.0) / 16.2))


def _res_250_180(best: str, other: str):
    from chess_anti_engine.stockfish.uci import StockfishPV, StockfishResult

    same_wdl = np.array([0.90, 0.05, 0.05], dtype=np.float32)
    return StockfishResult(
        bestmove_uci=best,
        wdl=same_wdl,
        pvs=[
            StockfishPV(move_uci=best, wdl=same_wdl, cp=250),
            StockfishPV(move_uci=other, wdl=same_wdl, cp=180),
        ],
    )


def test_cp_mode_governs_the_curriculum_gate_label() -> None:
    """End-to-end through the REAL _process_sf_results label branch."""
    from chess_anti_engine.moves.encode import uci_to_policy_index
    from chess_anti_engine.selfplay.stockfish_turn import _process_sf_results

    best, other = "a2a3", "a2a4"
    b, o = uci_to_policy_index(best, True), uci_to_policy_index(other, True)
    legal = np.array([b, o], dtype=np.int64)
    state = _label_state_cp(legal)
    rec = state.samples_per_game[0][0]

    _process_sf_results(
        state, [0], results={0: _res_250_180(best, other)},
        play_curriculum_moves=True, attach_labels=True,
    )

    assert rec.sf_policy_target is not None
    np.testing.assert_allclose(
        float(rec.sf_policy_target[b]), _P_BEST_250_180, atol=1e-3,
    )


def test_cp_mode_governs_the_async_attach_label() -> None:
    """Same invariant through _process_sf_label_result_for_record — the
    primary production label path (async selfplay labels + curriculum
    reuse). Review F2's most important surviving deletion was here."""
    from chess_anti_engine.moves.encode import uci_to_policy_index
    from chess_anti_engine.selfplay.stockfish_turn import (
        _process_sf_label_result_for_record,
    )

    best, other = "a2a3", "a2a4"
    b, o = uci_to_policy_index(best, True), uci_to_policy_index(other, True)
    legal = np.array([b, o], dtype=np.int64)
    state = _label_state_cp(legal)
    rec = state.samples_per_game[0][0]

    _process_sf_label_result_for_record(
        state, rec=rec, res=_res_250_180(best, other), turn=True,
        legal_indices=legal,
    )

    assert rec.sf_policy_target is not None
    np.testing.assert_allclose(
        float(rec.sf_policy_target[b]), _P_BEST_250_180, atol=1e-3,
    )


def test_cp_mode_governs_the_refute_opp_target_temp() -> None:
    """_sf_refute_opp_policy_target selects the temp inside the helper —
    a revert to state.game.sf_policy_temp makes the target one-hot."""
    from chess_anti_engine.selfplay.stockfish_turn import (
        _sf_refute_opp_policy_target,
    )

    legal = np.array([10, 20], dtype=np.int64)
    state = _label_state_cp(legal)
    p = _sf_refute_opp_policy_target(
        state, 0, cand_idxs=[10, 20], cand_scores=[250.0, 180.0],
        legal_indices=legal,
    )
    np.testing.assert_allclose(float(p[10]), _P_BEST_250_180, atol=1e-3)


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
    # Review F3: the first fixture was degenerate — a 2420cp mate row gave an
    # identical (one-hot) target at cp_temp 16.2 AND at the wdl temp, so
    # deleting the mode-selected temp stayed green. Row 0's 250-vs-180 pair is
    # temperature-SENSITIVE (p=0.987 at 16.2, 1.0 at 0.25/0.012); row 1
    # carries a BOTH-cp-and-mate row so a mate-precedence flip in either the
    # torch scorer or the numpy rebuild breaks their parity.
    raw = np.full((2, 8, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[0, 0] = (100, 250, 0, 600, 300)
    raw[0, 1] = (200, 180, 0, 450, 320)
    raw[0, 2] = (400, SF_CP_SENTINEL, 0, 500, 300)  # wdl-only: unscoreable in cp
    raw[1, 0] = (150, 60, 3, 700, 200)              # cp AND mate: mate wins
    raw[1, 1] = (160, 40, 0, 650, 250)
    legal_idx = np.array([100, 150, 160, 200, 400], dtype=np.int64)
    legal = np.zeros((2, width), np.float32)
    legal[:, legal_idx] = 1.0

    dense = np.zeros((2, width), np.float32)
    for i in range(2):
        rebuilt = rebuild_sf_policy_target(
            raw[i], legal_indices=legal_idx, policy_size=width, params=params,
        )
        assert rebuilt is not None
        dense[i] = rebuilt

    # Analytic pin on the rebuild itself (temp 16.2 HARDCODED here): softmax
    # over [250, 180] then the uncovered-legal smoothing. A temp revert inside
    # target_builder cannot hide behind torch/numpy agreeing on the mutation.
    z = np.exp(np.array([250.0, 180.0]) / 16.2)
    p_best = z[0] / z.sum()
    smooth = 0.01
    expect_best = p_best * (1.0 - smooth) + smooth / float(legal_idx.size)
    np.testing.assert_allclose(float(dense[0, 100]), expect_best, atol=1e-4)

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


def test_audit_ruler_scores_cp_mode_at_the_cp_temp() -> None:
    """Review residual on F1 (PR #355): the audit-first ruler's cp branch had
    zero coverage — reverting the scorer, the temp selection, or the whole
    mode branch in `_sf_soft_distribution` stayed GREEN, which is exactly the
    F1 defect (the ruler silently audits the OTHER candidate under this
    name). The fixture's two rows carry EQUAL native wdl, so a scorer revert
    to `_pv_wdl_score` reads 0.5/0.5; `sf_policy_temp` is set to the
    production 0.012, so a temp-selection revert saturates to ~1.0. Both
    fail the 0.98686 analytic pin at atol 1e-3."""
    from chess_anti_engine.moves.encode import uci_to_policy_index
    from scripts.audit_targets import _SfSoftParams, _sf_soft_distribution

    a_best = uci_to_policy_index("e2e4", True)
    a_other = uci_to_policy_index("d2d4", True)
    assert a_best >= 0
    assert a_other >= 0
    rec = {
        "pvs": [
            {"move": "e2e4", "cp": 250, "mate": None, "wdl": [500, 300, 200]},
            {"move": "d2d4", "cp": 180, "mate": None, "wdl": [500, 300, 200]},
        ]
    }
    legal = np.array([a_best, a_other], dtype=np.int64)
    params = _SfSoftParams(
        sf_policy_temp=0.012,
        sf_policy_label_smooth=0.01,
        sf_wdl_use_cp_logistic=False,
        sf_wdl_cp_slope=0.006,
        sf_wdl_cp_draw_width=120.0,
        sf_policy_score_mode="cp",
        sf_policy_cp_temp=16.2,
    )
    dist = _sf_soft_distribution(rec, legal, params=params)
    # candidates cover every legal move, so no smoothing: pure softmax over
    # [250, 180] cp at temp 16.2 (constant shared with the production-path
    # tests above)
    p_best = 1.0 / (1.0 + np.exp(-(250.0 - 180.0) / 16.2))
    np.testing.assert_allclose(float(dist[0]), p_best, atol=1e-3)
    np.testing.assert_allclose(float(dist.sum()), 1.0, atol=1e-6)


def test_audit_ruler_params_derive_from_the_flat_config() -> None:
    """The yaml-read leg of the same residual: `main()` built `_SfSoftParams`
    inline, so deleting the `sf_policy_score_mode` read was invisible to
    every test. The construction now has one home; drive it."""
    from scripts.audit_targets import _sf_soft_params_from_flat

    got = _sf_soft_params_from_flat(
        {"sf_policy_score_mode": "cp", "sf_policy_cp_temp": 7.5}
    )
    assert got.sf_policy_score_mode == "cp"
    assert got.sf_policy_cp_temp == 7.5
    # absent keys keep the production-shipped defaults (wdl mode: the ruler
    # must not silently switch modes on a config predating the key)
    dflt = _sf_soft_params_from_flat({})
    assert dflt.sf_policy_score_mode == "wdl"
    assert dflt.sf_policy_cp_temp == 16.2


def test_sparse_ce_wdl_only_block_is_unscoreable_in_cp_mode() -> None:
    """Review residual on the sparse-CE scoreable mask: an all-wdl-only
    MultiPV block in cp mode must be UNSCOREABLE end to end — the dense
    rebuild returns None (caller keeps the stored target) and the sparse path
    takes the bestmove-fallback branch. Dropping `has_cp | has_mate` from the
    cp branch of `_row_scores` would instead softmax two equal SENTINEL
    scores into a uniform 0.5/0.5 target; the analytic fallback pin below
    goes RED on that mutation."""
    import torch

    from chess_anti_engine.train.sparse_sf_ce import sparse_sf_policy_ce

    width = 4672
    params = SfTargetParams(
        sf_policy_label_smooth=0.01,
        sf_policy_score_mode="cp", sf_policy_cp_temp=16.2,
    )
    raw = np.full((1, 8, 5), -1, np.int16)
    raw[:, :, 1] = SF_CP_SENTINEL
    raw[0, 0] = (100, SF_CP_SENTINEL, 0, 600, 300)   # wdl-only
    raw[0, 1] = (200, SF_CP_SENTINEL, 0, 450, 320)   # wdl-only
    legal_idx = np.array([100, 200, 300], dtype=np.int64)

    # dense semantics: no scoreable rows -> None, stored target kept
    assert rebuild_sf_policy_target(
        raw[0], legal_indices=legal_idx, policy_size=width, params=params,
    ) is None
    # the same block IS scoreable in wdl mode — the mask is mode-driven
    assert rebuild_sf_policy_target(
        raw[0], legal_indices=legal_idx, policy_size=width,
        params=SfTargetParams(sf_policy_label_smooth=0.01),
    ) is not None

    legal = np.zeros((1, width), np.float32)
    legal[:, legal_idx] = 1.0
    torch.manual_seed(0)
    logits = torch.randn(1, width)
    batch = {
        "sf_multipv_raw": torch.from_numpy(raw.astype(np.int32)),
        "has_sf_multipv_raw": torch.ones(1),
        "sf_legal_mask": torch.from_numpy(legal),
        "has_sf_legal_mask": torch.ones(1),
        "sf_move_index": torch.tensor([100]),
        "has_sf_move": torch.ones(1),
    }
    sparse_ce, ok = sparse_sf_policy_ce(
        logits, batch, params=params, legal_aligned=batch["sf_legal_mask"],
    )
    assert ok.tolist() == [1.0]
    # analytic fallback: one-hot at sf_move_index, smoothed over the 3-move
    # legal set (fallback covers 1 of 3 legal -> smoothing applies)
    lsm = torch.log_softmax(logits.float(), dim=-1)[0]
    smooth = 0.01
    expect = -(1.0 - smooth) * lsm[100] - smooth * lsm[legal_idx].mean()
    torch.testing.assert_close(sparse_ce[0], expect, atol=1e-5, rtol=1e-5)
