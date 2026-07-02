"""Tests for the inverse-fit lever logic in DifficultyPID (regret stage)."""
from __future__ import annotations

import math
import random

from chess_anti_engine.stockfish.pid import (
    DifficultyPID,
    _fit_inverse_lever,
    _observation_se,
)


# Test-local alias matching the regret-side slope convention so legacy tests
# keep their signature (``_fit_inverse_regret(history, target_wr=...)``).
def _fit_inverse_regret(history, *, target_wr, recency_half_life=0.0):
    return _fit_inverse_lever(
        history,
        target_wr=target_wr,
        expected_slope_sign=+1,
        recency_half_life=recency_half_life,
    )


def _mk_pid(**overrides) -> DifficultyPID:
    defaults: dict = {
        "initial_nodes": 5000,
        "target_winrate": 0.57,
        "ema_alpha": 0.5,
        "min_nodes": 5000,
        "max_nodes": 25000,
        "initial_wdl_regret": 0.15,
        "wdl_regret_min": 0.02,
        "wdl_regret_max": 1.0,
        "regret_window": 10,
        "regret_max_step": 0.01,
        "regret_safety_floor": 0.50,
        "regret_emergency_ease_step": 0.01,
    }
    defaults.update(overrides)
    return DifficultyPID(**defaults)


def _feed_observations(pid: DifficultyPID, pairs: list[tuple[float, float]], n_games: int = 800) -> None:
    """Feed a sequence of (regret_to_set, winrate) iters. Appends to history."""
    for target_regret, wr in pairs:
        pid.wdl_regret = float(target_regret)
        w = int(round(wr * n_games))
        d = 0
        losses = n_games - w - d
        pid.observe(wins=w, draws=d, losses=losses, force=True)


def test_min_max_nodes_live_reload_ratchets_node_count():
    # 2026-06-20: node bounds are now live so the operator can ratchet the
    # stage-1-frozen node count up without a restart (regret-lever regime).
    pid = _mk_pid(
        initial_nodes=500_000, min_nodes=500_000, max_nodes=1_000_000,
        initial_wdl_regret=0.08, wdl_regret_min=0.0075, wdl_regret_max=0.7,
    )
    assert pid.nodes == 500_000
    # Raise the floor live → node count drags up to it (the "bump nodes" workflow).
    pid.refresh_live_params({"sf_pid_min_nodes": 650_000})
    assert pid.min_nodes == 650_000
    assert pid.nodes == 650_000
    # Raise the ceiling live → bounds update, node count unchanged (within range).
    pid.refresh_live_params({"sf_pid_max_nodes": 800_000})
    assert pid.max_nodes == 800_000
    assert pid.nodes == 650_000

    # Lowering the ceiling below the current count (still >= floor) clamps it down.
    pid2 = _mk_pid(
        initial_nodes=900_000, min_nodes=500_000, max_nodes=1_000_000,
        initial_wdl_regret=0.08, wdl_regret_min=0.0075, wdl_regret_max=0.7,
    )
    assert pid2.nodes == 900_000
    pid2.refresh_live_params({"sf_pid_max_nodes": 600_000})
    assert pid2.max_nodes == 600_000
    assert pid2.nodes == 600_000


def test_regret_airbag_mult_defaults_to_legacy():
    # Codex review #1: the emergency-ease multiplier is a NODES feature; the
    # regret lever must default to the legacy fixed airbag (mult=0) so configs
    # with a nonzero regret ease_step_frac keep their tuned airbag size.
    pid = _mk_pid()
    assert pid.regret_lever.emergency_ease_mult == 0.0
    assert pid.nodes_lever.emergency_ease_mult == 2.0


def test_inverted_live_node_bounds_normalized():
    # Codex review #3: a live edit dropping max below the current min must not
    # strand nodes above max — the floor yields to the ceiling.
    pid = _mk_pid(
        initial_nodes=900_000, min_nodes=500_000, max_nodes=1_000_000,
        initial_wdl_regret=0.08, wdl_regret_min=0.0075, wdl_regret_max=0.7,
    )
    pid.refresh_live_params({"sf_pid_max_nodes": 400_000})  # below current min 500k
    assert pid.max_nodes == 400_000
    assert pid.min_nodes == 400_000          # floor clamped down to the ceiling
    assert pid.nodes == 400_000              # not stranded above max
    assert pid.min_nodes <= pid.max_nodes


def test_metric_node_bounds_frozen_across_ratchet():
    # Codex review #2: ratcheting the live node floor must NOT move the
    # opponent_strength metric normalizer (else the higher node count scores as
    # the new minimum, zeroing the search-depth credit).
    pid = _mk_pid(
        initial_nodes=500_000, min_nodes=500_000, max_nodes=1_000_000,
        initial_wdl_regret=0.08, wdl_regret_min=0.0075, wdl_regret_max=0.7,
    )
    assert pid.metric_min_nodes == 500_000
    assert pid.metric_max_nodes == 1_000_000
    pid.refresh_live_params({"sf_pid_min_nodes": 650_000})
    assert pid.min_nodes == 650_000           # live clamp bound moves
    assert pid.metric_min_nodes == 500_000    # metric baseline stays frozen
    assert pid.metric_max_nodes == 1_000_000


def test_metric_node_bounds_survive_ratchet_and_resume():
    # Codex review #2 follow-up: a live node-floor ratchet then a checkpoint/
    # resume (the PID is rebuilt from the ratcheted yaml before load_state_dict)
    # must keep the ORIGINAL metric bounds, else opponent_strength drifts.
    pid = _mk_pid(
        initial_nodes=500_000, min_nodes=500_000, max_nodes=1_000_000,
        initial_wdl_regret=0.08, wdl_regret_min=0.0075, wdl_regret_max=0.7,
    )
    pid.refresh_live_params({"sf_pid_min_nodes": 650_000})  # ratchet the floor up
    state = pid.state_dict()
    # Resume: rebuild from the now-ratcheted yaml (min_nodes=650k), then restore.
    resumed = _mk_pid(
        initial_nodes=650_000, min_nodes=650_000, max_nodes=1_000_000,
        initial_wdl_regret=0.08, wdl_regret_min=0.0075, wdl_regret_max=0.7,
    )
    assert resumed.metric_min_nodes == 650_000   # construction picks up ratcheted value
    resumed.load_state_dict(state)
    assert resumed.metric_min_nodes == 500_000   # restored to the original frozen baseline
    assert resumed.metric_max_nodes == 1_000_000
    # Legacy checkpoint (no metric keys) falls back to the construction value.
    legacy = _mk_pid(initial_nodes=650_000, min_nodes=650_000, max_nodes=1_000_000)
    legacy.load_state_dict({"nodes": 650_000})
    assert legacy.metric_min_nodes == 650_000


def test_unchanged_node_bounds_preserve_lever_float():
    # Codex review: the flattened live config always carries sf_pid_min/max_nodes,
    # so a no-op refresh (bounds unchanged) must NOT re-clamp — that would rewrite
    # the nodes lever's fractional accumulator through the ceil-based property and
    # defeat the sub-integer accumulation the nodes PID relies on.
    pid = _mk_pid(initial_nodes=5000, min_nodes=1, max_nodes=25000)
    pid.nodes_lever.value = 1.4  # sub-integer accumulation
    pid.refresh_live_params({"sf_pid_min_nodes": 1, "sf_pid_max_nodes": 25000})  # unchanged
    assert pid.nodes_lever.value == 1.4  # preserved, not rewritten to ceil(1.4)=2.0


def test_observation_se_matches_known_distribution():
    # Unanimous batch at modest n: observed variance is 0, so the SAMPLE-SIZE floor
    # (0.5/sqrt n, the widest binomial SE) dominates — SE is wide, not the old flat
    # 0.01. (Codex review #2 + the curriculum-starvation airbag misfire on tiny
    # batches: a small all-same-result sample has large TRUE uncertainty.)
    se_pure = _observation_se(10, 0, 0)
    assert abs(se_pure - 0.5 / math.sqrt(10)) < 1e-9  # = 0.158, not the 0.01 floor
    # Small-sample airbag protection: 2 straight losses must NOT read as a confident
    # 0% that trips the airbag (which fires on raw_wr + 1.5*se < safety_floor=0.50).
    # se = 0.5/sqrt(2) = 0.354, so 0 + 1.5*0.354 = 0.53 > 0.50 → cannot fire on n=2.
    assert 1.5 * _observation_se(0, 0, 2) >= 0.50
    # 50/50 binary at large n: the observed Bernoulli SE (≈0.0177 at n=800) just
    # clears the sample-size floor, so the real observed SE is used.
    se = _observation_se(400, 0, 400)
    assert abs(se - math.sqrt(0.25 / 799)) < 1e-4


def test_fit_recovers_linear_relationship():
    # Synthetic linear data: winrate = 0.4 + 2.0 * regret
    history = [
        (0.10, 0.60, 0.018),
        (0.12, 0.64, 0.018),
        (0.14, 0.68, 0.018),
        (0.16, 0.72, 0.018),
        (0.18, 0.76, 0.018),
    ]
    predicted_r = _fit_inverse_regret(history, target_wr=0.70)
    assert predicted_r is not None
    # Target 0.70 should map to regret 0.15 (slope 2, intercept 0.4)
    assert abs(predicted_r - 0.15) < 0.005


def test_fit_rejects_insufficient_span():
    # All regret values identical → span=0, fit should reject
    history = [(0.15, 0.60, 0.018)] * 5
    fit = _fit_inverse_regret(history, target_wr=0.60)
    assert fit is None


def test_fit_rejects_wrong_sign_slope():
    # Non-physical: higher regret maps to LOWER winrate (e.g. model collapsing)
    history = [
        (0.10, 0.75, 0.018),
        (0.15, 0.65, 0.018),
        (0.20, 0.55, 0.018),
    ]
    fit = _fit_inverse_regret(history, target_wr=0.60)
    assert fit is None


def test_safety_floor_forces_ease_below_threshold():
    pid = _mk_pid(initial_wdl_regret=0.15)
    n_games = 800
    # Winrate 0.40 → well below safety_floor 0.50 → emergency ease
    wins = int(0.40 * n_games)
    draws = 0
    losses = n_games - wins - draws
    pid.observe(wins=wins, draws=draws, losses=losses, force=True)
    # Should ease toward higher regret (easier SF) by exactly max_step (cap)
    assert pid.wdl_regret > 0.15  # did ease
    # Emergency ease is capped at inverse_regret_max_step = 0.01
    # (even if computed urgency suggests much larger step).
    assert abs(pid.wdl_regret - 0.15 - 0.01) < 1e-9


def test_emergency_ease_never_exceeds_max_step():
    # Codex HIGH finding #1: one 0.40 batch must not jump regret by +0.13.
    pid = _mk_pid(
        initial_wdl_regret=0.15,
        regret_max_step=0.01,
        regret_emergency_ease_step=0.01,
        regret_safety_floor=0.50,
    )
    # Even an extreme batch (zero wins, zero draws) must not exceed max_step.
    pid.observe(wins=0, draws=0, losses=800, force=True)
    assert pid.wdl_regret <= 0.15 + 0.01 + 1e-9
    # And a moderate emergency (40% wr) also respects cap.
    pid.wdl_regret = 0.15  # reset
    pid.observe(wins=320, draws=0, losses=480, force=True)
    assert pid.wdl_regret <= 0.15 + 0.01 + 1e-9


def test_saturated_batches_do_not_dominate_fit():
    # Codex HIGH finding #2: all-win/all-draw batches should NOT get unbounded
    # weight. The SE floor caps their influence.
    # History has one normal batch at r=0.10 and three saturated (all-win)
    # batches at r=0.12. Without SE floor, saturated batches would dominate
    # and drag the fit; with floor, they contribute finite weight.
    n = 800
    se_normal = _observation_se(int(0.55 * n), 0, int(0.45 * n))
    se_saturated = _observation_se(n, 0, 0)  # all wins → SE should be floored
    # Saturated SE must not be zero (that was the bug).
    assert se_saturated > 0.0
    # SE floor caps how much more weight a saturated batch gets vs a normal one.
    # (1/se_saturated^2) should be close to (1/se_normal^2), not orders larger.
    assert (1.0 / se_saturated ** 2) / (1.0 / se_normal ** 2) < 10.0


# Removed test_sigma_pred_reflects_residual_variance — sigma_pred has been
# dropped from _fit_inverse_regret. Backtest on 168 iters showed it had
# ~0 correlation with actual prediction error (Pearson 0.009), so the
# step-size policy ignores it; the cov00/cov01/cov11 + sqrt block was
# dead compute. See pid.py:_fit_inverse_regret for the rationale.


def test_regret_history_round_trips_via_state_dict():
    # Codex MED finding #3: history must persist across save/load. Use
    # observations far from target (raw_wr 0.40, target 0.57) so they
    # bypass the dual z-gate deadband and definitely append.
    pid1 = _mk_pid(initial_wdl_regret=0.15)
    # Construction seeds 1 anchor entry; 5 observations append → 6 total.
    _feed_observations(pid1, [
        (0.10, 0.40),
        (0.12, 0.42),
        (0.14, 0.44),
        (0.16, 0.46),
        (0.18, 0.48),
    ], n_games=800)
    state = pid1.state_dict()
    assert "regret_history" in state
    saved_len = len(state["regret_history"])
    assert saved_len == 6

    # New PID, restore state — history should come back identically.
    pid2 = _mk_pid(initial_wdl_regret=0.15)
    assert len(pid2.regret_lever.history) == 1  # construction seed
    pid2.load_state_dict(state)
    assert len(pid2.regret_lever.history) == saved_len
    for (r1, w1, s1), (r2, w2, s2) in zip(pid1.regret_lever.history, pid2.regret_lever.history):
        assert abs(r1 - r2) < 1e-12
        assert abs(w1 - w2) < 1e-12
        assert abs(s1 - s2) < 1e-12


def test_deadband_holds_skip_history_append():
    # Constant-x history collapses the WLS slope (det → 0 → degenerate fit),
    # forcing a blind exploration step the next time something disturbs the
    # system. Skip the append in deadband so the deque retains older
    # varied-x points.
    pid = _mk_pid(
        initial_wdl_regret=0.15,
        target_winrate=0.57,
        ema_alpha=0.5,
        regret_safety_floor=0.30,  # below test wr so airbag never fires
        regret_deadband_sigma=2.0,  # wide gate so test wr lands in it
    )
    # Pre-seed history with one entry so we can verify it's not overwritten.
    pid.regret_lever.history.append((0.10, 0.55, 0.018))
    pid.ema_winrate = 0.57  # neutralise EMA so ema_err = 0
    initial_len = len(pid.regret_lever.history)
    # raw_wr = 0.57 → err = 0; clearly in deadband.
    pid.observe(wins=456, draws=0, losses=344, force=True)  # 0.57 wr exact
    assert len(pid.regret_lever.history) == initial_len, (
        "in-deadband observe must not append; otherwise constant-x entries "
        "fill the deque and break future fits"
    )


def test_legacy_checkpoint_without_regret_history_loads_gracefully():
    # Older state dicts (pre-inverse-model or from disabled runs) have no
    # "inverse_history" key. Loading should not crash.
    pid = _mk_pid(initial_wdl_regret=0.15)
    legacy_state = {
        "nodes": 5000, "skill_level": 20, "random_move_prob": 0.0,
        "wdl_regret": 0.15, "ema_winrate": 0.60,
        "integral": 0.0, "prev_err": None, "games_since_adjust": 0,
        "random_stage_complete": True, "regret_stage_complete": False,
    }
    pid.load_state_dict(legacy_state)
    # Legacy state has no regret_history → _restore_history is a no-op and
    # the construction-time seed entry stays. Crash-free is the contract.
    assert len(pid.regret_lever.history) == 1


def test_holds_with_insufficient_history():
    pid = _mk_pid(initial_wdl_regret=0.15)
    # Feed single observation → history len 1, not enough for fit
    pid.observe(wins=456, draws=0, losses=344, force=True)  # wr = 0.57
    # Regret should stay put
    assert abs(pid.wdl_regret - 0.15) < 1e-9


def test_moves_toward_predicted_target_regret():
    pid = _mk_pid(initial_wdl_regret=0.15, target_winrate=0.57)
    # Build linear history via observations: wr ≈ 0.5 + 1.0*regret
    # At 0.10 → 0.60, 0.12 → 0.62, 0.14 → 0.64, 0.16 → 0.66
    # Target 0.57 → predicted regret = (0.57 - 0.5) / 1.0 = 0.07
    # But that's below the observed range [0.10, 0.16] → clamped to 0.10
    _feed_observations(pid, [
        (0.10, 0.60),
        (0.12, 0.62),
        (0.14, 0.64),
        (0.16, 0.66),
        (0.15, 0.65),
    ], n_games=800)
    # Regret should have moved toward lower values (we're winning more than target)
    assert pid.wdl_regret < 0.15
    # Should stay within max_step (0.01) per iter
    # (can't test exact trajectory here; fit depends on recorded history)


def test_never_extrapolates_beyond_observed_range():
    # Scenario: slope fit predicts regret WAY above observed range
    pid = _mk_pid(initial_wdl_regret=0.10, target_winrate=0.90)
    # History with moderate slope; predicted r* for wr=0.90 extrapolates high
    _feed_observations(pid, [
        (0.08, 0.55),
        (0.10, 0.57),
        (0.12, 0.59),
        (0.09, 0.56),
        (0.11, 0.58),
    ], n_games=800)
    # r_max_obs ≈ 0.12; predicted r* would be >> 0.12. Should clamp to 0.12.
    # Per-step cap is 0.01 so from 0.10 it can only reach 0.11.
    assert pid.wdl_regret <= 0.12 + 1e-9
    assert pid.wdl_regret > 0.10  # did move up


def test_step_size_respects_max_step_cap():
    pid = _mk_pid(initial_wdl_regret=0.15, regret_max_step=0.003)
    _feed_observations(pid, [
        (0.05, 0.45),
        (0.10, 0.55),
        (0.15, 0.65),
        (0.20, 0.75),
        (0.25, 0.85),
    ], n_games=800)
    # No single step should exceed max_step=0.003
    # (history was fed via multiple observe() calls, each capped at 0.003)
    # Final regret may have drifted but each step was ≤ 0.003.
    # We check that cumulative movement is ≤ 5 × 0.003 = 0.015 from initial 0.15:
    # from initial 0.15 through 4 capped steps (initial pre-obs sets to 0.05, 0.10...)
    # Actually: _feed_observations sets regret directly then observes, so each
    # observe() call starts from the set regret. Final value is last observation
    # set (0.25) with one observe-step applied.
    # So we just verify the post-call regret ≤ 0.25 + 0.003 and ≥ 0.25 - 0.003.
    assert abs(pid.wdl_regret - 0.25) <= 0.003 + 1e-9


def test_raw_sign_disagreement_steps_half_max_in_raw_direction():
    # When the fit's predicted delta disagrees with the live raw signal's
    # direction, the fit is provably stale (averaged over the recent window
    # while raw is the fresh single-iter measurement). Step in raw's
    # direction at 0.5 × abs_max_step — neither full step (would defeat the
    # fit's role of magnitude calibration on agreement) nor zero (the prior
    # behavior, which held steady through several iters of underperformance
    # because the fit hadn't caught up).
    pid = _mk_pid(
        initial_wdl_regret=0.15,
        target_winrate=0.60,
        regret_max_step=0.01,
        regret_safety_floor=0.50,  # stay above so the airbag doesn't fire
        ema_alpha=0.5,
    )
    # Pre-populate history with a steep positive-slope relationship at LOW
    # regret values, so even after the new low-raw observation is appended,
    # the fit still predicts r* far below current regret 0.20 (i.e. wants
    # to tighten). Using direct insertion bypasses the auto-step that
    # ``_feed_observations`` triggers.
    pid.regret_lever.history.extend([
        (0.04, 0.30, 0.018),
        (0.06, 0.40, 0.018),
        (0.08, 0.50, 0.018),
        (0.10, 0.60, 0.018),
        (0.12, 0.70, 0.018),
    ])
    pid.wdl_regret = 0.20  # well above the fit's r* ≈ 0.10
    regret_before = pid.wdl_regret
    pid.ema_winrate = 0.60  # neutral, won't dominate the deadband
    n = 800
    wins = int(0.55 * n)  # raw 0.55 < target 0.60 → wants ease
    pid.observe(wins=wins, draws=0, losses=n - wins, force=True)
    # Fit: tighten. Raw: ease. Disagreement → +0.5 × max_step = +0.005.
    delta = pid.wdl_regret - regret_before
    assert delta > 0, f"expected ease when raw < target, got delta={delta}"
    assert abs(delta - 0.005) < 1e-9, f"expected +0.005, got {delta}"


def test_raw_sign_agreement_lets_fit_drive_magnitude():
    # Sanity for the symmetric path: when fit and raw signs AGREE, the fit's
    # delta governs magnitude (subject to abs_max_step cap). The half-step
    # override only applies on disagreement.
    pid = _mk_pid(
        initial_wdl_regret=0.15,
        target_winrate=0.60,
        regret_max_step=0.01,
        regret_safety_floor=0.50,
        ema_alpha=0.5,
    )
    # Seed: lower regret → lower winrate, so fit predicts r* above current
    # for target 0.60 (wants to ease).
    _feed_observations(pid, [
        (0.10, 0.50),
        (0.12, 0.54),
        (0.14, 0.58),
        (0.16, 0.62),
        (0.18, 0.66),
    ], n_games=800)
    # Set regret low so fit definitely predicts ease, then feed below-target
    # raw. Fit (ease) and raw (below target → wants ease) AGREE.
    pid.wdl_regret = 0.10
    regret_before = pid.wdl_regret
    n = 800
    wins = int(0.55 * n)  # raw 0.55 < target 0.60
    pid.observe(wins=wins, draws=0, losses=n - wins, force=True)
    delta = pid.wdl_regret - regret_before
    assert delta > 0, f"expected ease when both signal ease, got {delta}"
    # Magnitude is fit-driven, capped at max_step=0.01. Should NOT be the
    # half-step 0.005 from the disagreement branch.
    assert delta <= 0.01 + 1e-9


def test_tighten_gain_damps_regret_hardening():
    pid = _mk_pid(
        initial_wdl_regret=0.20,
        target_winrate=0.60,
        regret_max_step=0.10,
        regret_max_step_frac=0.03,
        regret_ease_step_frac=0.06,
        regret_tighten_gain=0.5,
        regret_safety_floor=0.30,
        ema_alpha=0.5,
    )
    pid.regret_lever.history.clear()
    pid.regret_lever.history.extend([
        (0.10, 0.55, 0.018),
        (0.12, 0.58, 0.018),
        (0.14, 0.61, 0.018),
        (0.16, 0.64, 0.018),
    ])
    pid.ema_winrate = 0.60
    regret_before = pid.wdl_regret

    pid.observe(wins=600, draws=0, losses=200, force=True)

    delta = pid.wdl_regret - regret_before
    assert delta < 0
    assert abs(delta + 0.003) < 1e-9


def test_tighten_streak_gain_damps_only_after_repeated_hardening():
    pid = _mk_pid(
        initial_wdl_regret=0.20,
        target_winrate=0.60,
        regret_max_step=0.10,
        regret_max_step_frac=0.03,
        regret_ease_step_frac=0.06,
        regret_tighten_streak_after=2,
        regret_tighten_streak_gain=0.5,
        regret_safety_floor=0.30,
        ema_alpha=0.5,
    )
    pid.regret_lever.history.clear()
    pid.regret_lever.history.extend([
        (0.18, 0.64, 0.018),
        (0.17, 0.64, 0.018),
        (0.16, 0.64, 0.018),
        (0.15, 0.64, 0.018),
    ])
    pid.ema_winrate = 0.60
    regret_before = pid.wdl_regret

    pid.observe(wins=600, draws=0, losses=200, force=True)

    delta = pid.wdl_regret - regret_before
    assert delta < 0
    assert abs(delta + 0.003) < 1e-9

    pid.regret_lever.history.clear()
    pid.regret_lever.history.extend([
        (0.18, 0.64, 0.018),
        (0.19, 0.64, 0.018),
        (0.20, 0.64, 0.018),
        (0.19, 0.64, 0.018),
    ])
    pid.wdl_regret = 0.20
    regret_before = pid.wdl_regret

    pid.observe(wins=600, draws=0, losses=200, force=True)

    delta = pid.wdl_regret - regret_before
    assert delta < 0
    assert abs(delta + 0.006) < 1e-9


def test_crash_ease_minimum_overrides_timid_fit():
    pid = _mk_pid(
        initial_wdl_regret=0.24,
        target_winrate=0.58,
        regret_max_step=0.10,
        regret_max_step_frac=0.03,
        regret_ease_step_frac=0.06,
        regret_safety_floor=0.30,
        regret_crash_ease_z=2.0,
        regret_crash_ease_min_frac=0.5,
        ema_alpha=0.5,
    )
    pid.regret_lever.history.clear()
    pid.regret_lever.history.extend([
        (0.20, 0.52, 0.018),
        (0.22, 0.56, 0.018),
        (0.24, 0.60, 0.018),
        (0.26, 0.64, 0.018),
    ])
    pid.ema_winrate = 0.58
    regret_before = pid.wdl_regret

    pid.observe(wins=400, draws=0, losses=400, force=True)

    delta = pid.wdl_regret - regret_before
    assert abs(delta - 0.0072) < 1e-9


def test_inverse_model_handles_noisy_data():
    # Add measurement noise; fit should still give reasonable predictions
    random.seed(42)
    true_slope = 2.0
    true_intercept = 0.40
    history = []
    for regret in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]:
        true_wr = true_intercept + true_slope * regret
        noisy_wr = true_wr + random.gauss(0, 0.018)
        history.append((regret, noisy_wr, 0.018))
    predicted_r = _fit_inverse_regret(history, target_wr=0.70)
    assert predicted_r is not None
    # Target 0.70: true r* = 0.15. Noise should allow fit within ±0.02
    assert abs(predicted_r - 0.15) < 0.03


# --- Nodes lever (direction=+1) tests ---


def _mk_pid_nodes_only(**overrides) -> DifficultyPID:
    """PID with regret disabled so the nodes lever runs every iter."""
    defaults: dict = {
        "initial_nodes": 10000,
        "target_winrate": 0.57,
        "ema_alpha": 0.5,
        "min_nodes": 2000,
        "max_nodes": 200_000,
        "initial_wdl_regret": -1.0,  # disable regret stage; gate stays complete
        "nodes_window": 10,
        "nodes_max_step_frac": 0.10,
        "nodes_safety_floor": 0.0,
        "nodes_emergency_ease_step": 0.0,
        "nodes_deadband_sigma": 0.0,
    }
    defaults.update(overrides)
    return DifficultyPID(**defaults)


def test_fit_inverse_lever_recovers_negative_slope_for_nodes():
    # Synthetic linear data with NEGATIVE slope: more nodes → fewer wins.
    # winrate = 0.80 - 0.000004 * nodes
    history: list[tuple[float, float, float]] = [
        (5_000.0, 0.78, 0.018),
        (10_000.0, 0.76, 0.018),
        (20_000.0, 0.72, 0.018),
        (40_000.0, 0.64, 0.018),
        (80_000.0, 0.48, 0.018),
    ]
    predicted = _fit_inverse_lever(
        history, target_wr=0.57, expected_slope_sign=-1
    )
    assert predicted is not None
    # target 0.57 → nodes ≈ (0.80-0.57)/0.000004 ≈ 57500. Allow ±10%.
    assert 50_000 < predicted < 65_000


def test_fit_inverse_lever_rejects_positive_slope_for_nodes():
    # Nodes-side fit must reject positive slope (would mean more nodes →
    # more wins, which is non-physical for a stronger opponent).
    history: list[tuple[float, float, float]] = [
        (5_000.0, 0.50, 0.018),
        (10_000.0, 0.55, 0.018),
        (20_000.0, 0.60, 0.018),
    ]
    fit = _fit_inverse_lever(history, target_wr=0.57, expected_slope_sign=-1)
    assert fit is None


def test_nodes_lever_decreases_when_winning_too_much():
    # Underperforming target → easier SF → fewer nodes (direction=+1, ease=down).
    pid = _mk_pid_nodes_only(initial_nodes=10_000)
    n = 800
    # Win 40% (well below target 0.57) — controller should ease, dropping nodes.
    wins = int(0.40 * n)
    initial = pid.nodes
    pid.observe(wins=wins, draws=0, losses=n - wins, force=True)
    assert pid.nodes < initial, f"expected nodes to drop, got {initial}→{pid.nodes}"


def test_nodes_lever_increases_when_winning_too_little_for_opponent():
    # Outperforming target → harder SF → more nodes.
    pid = _mk_pid_nodes_only(initial_nodes=10_000)
    n = 800
    # Win 75% (well above target 0.57) — controller tightens, raising nodes.
    wins = int(0.75 * n)
    initial = pid.nodes
    pid.observe(wins=wins, draws=0, losses=n - wins, force=True)
    assert pid.nodes > initial, f"expected nodes to rise, got {initial}→{pid.nodes}"


def test_nodes_step_capped_by_max_step_frac():
    # ±10%/iter cap from a single observation, even when raw is extreme.
    pid = _mk_pid_nodes_only(initial_nodes=10_000, nodes_max_step_frac=0.10)
    pid.observe(wins=800, draws=0, losses=0, force=True)  # 100% wr → tighten hard
    # 10% cap → max nodes = 11_000
    assert pid.nodes <= 11_000 + 1
    pid.nodes = 10_000
    pid.observe(wins=0, draws=0, losses=800, force=True)  # 0% wr → ease hard
    assert pid.nodes >= 9_000 - 1


def test_nodes_history_round_trips_via_state_dict():
    pid1 = _mk_pid_nodes_only(initial_nodes=10_000)
    # Construction seeds 1 anchor entry; 4 observations append → 5 total.
    for _ in range(4):
        pid1.observe(wins=400, draws=0, losses=400, force=True)
    state = pid1.state_dict()
    assert "nodes_history" in state
    assert len(state["nodes_history"]) == 5

    pid2 = _mk_pid_nodes_only(initial_nodes=10_000)
    pid2.load_state_dict(state)
    assert len(pid2.nodes_lever.history) == 5


def test_regret_lever_freezes_after_stage_2_entered():
    # Once stage 2 is entered (regret reached stage_end), regret lever must
    # NEVER move again — including its airbag. This is the "single knob per
    # regime" invariant: stage 2 changes are nodes-only so an effect can be
    # attributed to one cause.
    pid = DifficultyPID(
        initial_nodes=10_000,
        target_winrate=0.57,
        ema_alpha=0.5,
        min_nodes=2_000,
        max_nodes=200_000,
        initial_wdl_regret=0.01,           # already at floor
        wdl_regret_min=0.01,
        wdl_regret_max=1.0,
        wdl_regret_stage_end=0.01,         # stage 2 active from start
        regret_window=10,
        regret_max_step=0.10,
        regret_safety_floor=0.99,           # absurdly high → would airbag every iter if active
        regret_emergency_ease_step=0.10,
        nodes_window=10,
        nodes_max_step_frac=0.10,
    )
    assert pid._regret_stage_complete  # precondition
    initial_regret = pid.wdl_regret
    # Catastrophic crash sequence — would normally trip regret airbag every iter.
    for _ in range(5):
        pid.observe(wins=0, draws=0, losses=800, force=True)
    assert pid.wdl_regret == initial_regret, (
        f"regret moved during stage 2 (was {initial_regret}, now {pid.wdl_regret})"
    )
    # Stage 2 stays entered even if regret somehow drifted up.
    assert pid._regret_stage_complete


def test_load_state_dict_reseeds_history_at_restored_value():
    # Regression: prior bug had construction-time seed at config initial_nodes
    # (e.g., 5000) but load_state_dict restored nodes=6354 from checkpoint
    # without updating the seed. The fit then anchored at a value the
    # controller wasn't actually at. Now load_state_dict re-seeds at the
    # restored value when saved history is empty.
    pid1 = _mk_pid_nodes_only(initial_nodes=5_000)
    # Seed entry should be at construction value 5000.
    assert list(pid1.nodes_lever.history) == [(5000.0, pid1.target, 0.01)]

    # Simulate a legacy / pre-seed checkpoint: state_dict with nodes=6354
    # but no nodes_history (matches the empty-history path in old saves).
    legacy_state = {
        "nodes": 6354,
        "wdl_regret": -1.0,
        "ema_winrate": 0.59,
        "regret_history": [],
        "nodes_history": [],
        "regret_stage_complete": True,
        "games_since_adjust": 0,
    }
    pid2 = _mk_pid_nodes_only(initial_nodes=5_000)
    pid2.load_state_dict(legacy_state)
    # Nodes should be restored to 6354 AND the seed entry should now anchor
    # at 6354 (current value), not the stale construction value 5000.
    assert pid2.nodes_lever.value == 6354
    assert len(pid2.nodes_lever.history) == 1
    seed_value, seed_wr, seed_se = pid2.nodes_lever.history[0]
    assert seed_value == 6354.0, (
        f"expected re-seeded anchor at restored value 6354, got {seed_value}"
    )
    assert seed_wr == pid2.target
    assert seed_se == 0.01

    # When saved history IS non-empty, _restore_history takes precedence and
    # the seed is replaced by the saved data (no re-seed).
    full_state = {
        "nodes": 6354,
        "wdl_regret": -1.0,
        "ema_winrate": 0.59,
        "nodes_history": [[5500.0, 0.55, 0.018], [6000.0, 0.58, 0.018]],
        "regret_history": [],
        "regret_stage_complete": True,
        "games_since_adjust": 0,
    }
    pid3 = _mk_pid_nodes_only(initial_nodes=5_000)
    pid3.load_state_dict(full_state)
    assert len(pid3.nodes_lever.history) == 2
    assert pid3.nodes_lever.history[0] == (5500.0, 0.55, 0.018)


def test_nodes_airbag_recovers_at_min_nodes():
    # Stage 2 entered, nodes already at the floor, winrate crashes.
    # Codex finding: at min_nodes=5000 the airbag had no escape (clamp
    # blocked further easing, regret was frozen). The shipped fix lowers
    # min_nodes to 1, so the airbag can keep dropping nodes; once winrate
    # recovers, the inverse fit ramps nodes back up.
    pid = DifficultyPID(
        initial_nodes=5_000,
        target_winrate=0.57,
        ema_alpha=0.5,
        min_nodes=1,
        max_nodes=1_000_000,
        initial_wdl_regret=0.005,        # below stage_end → stage 2 entered
        wdl_regret_min=0.001,
        wdl_regret_max=1.0,
        wdl_regret_stage_end=0.01,
        regret_window=10,
        regret_safety_floor=0.50,
        nodes_window=10,
        nodes_max_step_frac=0.10,
        nodes_safety_floor=0.50,
        nodes_emergency_ease_step=1000,
    )
    assert pid._regret_stage_complete, "stage 2 should be entered at construction"
    # 0/800 → raw_wr=0 fires the nodes airbag.
    pid.observe(wins=0, draws=0, losses=800, force=True)
    assert pid.nodes < 5_000, f"airbag should drop nodes from 5000, got {pid.nodes}"
    # Drive nodes to the floor.
    for _ in range(10):
        pid.observe(wins=0, draws=0, losses=800, force=True)
    assert pid.nodes == 1, f"expected nodes pinned at floor=1, got {pid.nodes}"
    # Floor reached → no further easing, but the lever isn't dead. Once
    # winrate recovers above target, the controller ramps nodes back up.
    # Ramp is slow at floor=1 because the frac cap (10% of 1 = 0.1) gates
    # per-iter movement, but the lever's float value accumulates and
    # eventually clears integer rounding.
    floor_value = pid.nodes_lever.value
    for _ in range(20):
        pid.observe(wins=720, draws=0, losses=80, force=True)  # 90% wr
    assert pid.nodes_lever.value > floor_value, (
        f"nodes lever should ramp from floor when winning, got "
        f"{floor_value}→{pid.nodes_lever.value}"
    )


def test_nodes_airbag_scales_with_node_count():
    # The airbag eases by emergency_ease_mult x the normal ease cap
    # (max_step_frac x nodes), so it auto-scales with node count instead of a
    # fixed absolute step. At 100k nodes that is 2 x 0.10 x 100k = 20k, not the
    # legacy ~1000-node fixed drop (which was weaker than a routine step at
    # high node counts — backwards for an emergency).
    pid = _mk_pid_nodes_only(
        initial_nodes=100_000,
        nodes_safety_floor=0.50,
        nodes_emergency_ease_step=0.0,   # no absolute floor → purely fractional
        nodes_emergency_ease_mult=2.0,
        nodes_max_step_frac=0.10,
    )
    before = pid.nodes
    pid.observe(wins=0, draws=0, losses=800, force=True)  # raw_wr=0 → airbag
    drop = before - pid.nodes
    assert 19_000 < drop < 21_000, (
        f"expected ~20k (2x10% of 100k) fractional airbag drop, got {drop}"
    )
    # The emergency_ease_step floor still governs when it exceeds the fractional
    # step (small node counts): 2x10% of 3000 = 600 < 1000 floor → drop 1000.
    pid2 = _mk_pid_nodes_only(
        initial_nodes=3_000,
        min_nodes=1,
        nodes_safety_floor=0.50,
        nodes_emergency_ease_step=1000.0,
        nodes_emergency_ease_mult=2.0,
        nodes_max_step_frac=0.10,
    )
    b2 = pid2.nodes
    pid2.observe(wins=0, draws=0, losses=800, force=True)
    assert b2 - pid2.nodes == 1000, f"floor should govern small N, got {b2 - pid2.nodes}"


def test_nodes_lever_max_step_unset_uses_frac_only():
    # Codex finding: shipping both an absolute and frac cap was a footgun
    # (frac silently dominated above ~50k nodes). Default behavior with
    # nodes_max_step left unset = no absolute cap; frac is the only bound.
    # nodes_degen_step_frac=1.0 so degenerate fallback uses the full frac cap;
    # this isolates the absolute-vs-frac question from the degen-step-size question.
    pid = _mk_pid_nodes_only(
        initial_nodes=200_000,
        max_nodes=2_000_000,
        nodes_max_step_frac=0.10,
        nodes_degen_step_frac=1.0,
        # nodes_max_step intentionally unset → defaults to ~unbounded
    )
    # 75% wr drives nodes up; cap should be 10% of value (= 20_000), not a
    # spurious 5000-style absolute.
    pid.observe(wins=600, draws=0, losses=200, force=True)
    delta = pid.nodes - 200_000
    assert 15_000 <= delta <= 22_000, (
        f"expected ~10%/iter step (~20k), got Δ={delta}"
    )


def test_nodes_lever_paused_when_regret_stage_incomplete():
    # Regret enabled with stage_end below current regret → stage incomplete →
    # nodes must not move regardless of winrate.
    pid = DifficultyPID(
        initial_nodes=10_000,
        target_winrate=0.57,
        ema_alpha=0.5,
        min_nodes=2_000,
        max_nodes=200_000,
        initial_wdl_regret=0.10,
        wdl_regret_min=0.01,
        wdl_regret_max=1.0,
        wdl_regret_stage_end=0.01,  # stage gate active; current regret 0.10 > 0.01
        regret_window=10,
        regret_max_step=0.005,
        regret_safety_floor=0.50,
        nodes_window=10,
        nodes_max_step_frac=0.10,
    )
    initial = pid.nodes
    pid.observe(wins=600, draws=0, losses=200, force=True)  # 75% wr
    assert pid.nodes == initial, "nodes must stay pinned while regret stage incomplete"


def test_nodes_diagnostics_expose_fit_prediction():
    pid = _mk_pid_nodes_only(
        initial_nodes=50_000,
        target_winrate=0.57,
        nodes_max_step_frac=0.10,
        nodes_deadband_sigma=0.0,
    )
    pid.nodes_lever.history.clear()
    pid.nodes_lever.history.extend([
        (20_000.0, 0.72, 0.018),
        (40_000.0, 0.64, 0.018),
        (60_000.0, 0.56, 0.018),
        (80_000.0, 0.48, 0.018),
    ])
    pid.ema_winrate = pid.target

    update = pid.observe(wins=480, draws=0, losses=320, force=True)  # raw_wr=0.60

    diag = update.nodes_diag
    assert update.nodes_active
    assert update.regret_diag is None
    assert diag is not None
    assert diag.name == "nodes"
    assert diag.predicted_value is not None
    assert diag.fit_slope is not None and diag.fit_slope < 0.0
    assert diag.reason in {"fit", "fit_capped"}
    assert diag.value_before == 50_000.0
    assert diag.value_after == pid.nodes_lever.value
    assert diag.applied_delta > 0.0
    assert diag.cap == 5_000.0
    assert diag.history_len == 5


def test_deadband_diagnostics_report_hold_without_history_append():
    pid = _mk_pid_nodes_only(
        initial_nodes=10_000,
        target_winrate=0.57,
        ema_alpha=0.5,
        nodes_deadband_sigma=2.0,
    )
    pid.ema_winrate = 0.57
    initial_len = len(pid.nodes_lever.history)

    update = pid.observe(wins=456, draws=0, losses=344, force=True)

    diag = update.nodes_diag
    assert diag is not None
    assert diag.reason == "deadband"
    assert not diag.changed
    assert diag.value_before == diag.value_after == 10_000.0
    assert diag.history_len == initial_len
    assert len(pid.nodes_lever.history) == initial_len


def test_min_games_for_adjust_holds_levers_below_sample_floor():
    """Curriculum-starvation fix: a tiny finished-game sample must NOT step the
    levers — not even on the forced iteration-boundary adjust — and the sample
    must accumulate across iters until it clears the floor. A slow SF opponent
    can finish only ~1-7 curriculum games/iter; a 2-game raw 0% was tripping the
    airbag and blowing regret around. The EMA still updates every iter."""
    pid = _mk_pid(
        min_games_for_adjust=30,
        initial_wdl_regret=0.15,
        regret_safety_floor=0.50,
    )
    start_regret = pid.wdl_regret

    # 2 finished games at raw 0% — below the 30-game floor: hold + accumulate.
    pid.observe(wins=0, draws=0, losses=2, force=True)
    assert pid.wdl_regret == start_regret          # lever held (no airbag)
    assert pid._games_since_adjust == 2            # counter NOT reset; accumulating

    # More tiny iters keep accumulating without stepping.
    pid.observe(wins=0, draws=0, losses=3, force=True)
    assert pid.wdl_regret == start_regret
    assert pid._games_since_adjust == 5

    # Once the accumulated sample clears the floor the forced adjust fires; a
    # genuine 0% crash eases regret upward (easier SF) via the airbag.
    pid.observe(wins=0, draws=0, losses=30, force=True)  # total 35 >= 30
    assert pid.wdl_regret > start_regret
    assert pid._games_since_adjust == 0            # reset after a real adjust


def test_min_games_for_adjust_zero_is_legacy_behavior():
    """Default 0 disables the floor: the forced boundary adjust steps on any
    sample, exactly as before the fix."""
    pid = _mk_pid(min_games_for_adjust=0, initial_wdl_regret=0.15,
                  regret_safety_floor=0.50)
    start_regret = pid.wdl_regret
    pid.observe(wins=0, draws=0, losses=2, force=True)  # 2 games, raw 0%
    assert pid.wdl_regret > start_regret           # airbag fires immediately


def test_min_games_for_adjust_live_reloadable():
    """The floor is a live-reload key so the operator can set it without a
    restart (registered in config_yaml's stockfish key set)."""
    pid = _mk_pid(min_games_for_adjust=0)
    pid.refresh_live_params({"sf_pid_min_games_for_adjust": 30})
    assert pid.min_games_for_adjust == 30


def test_min_games_for_adjust_pools_held_sample_into_the_step():
    """Review #83: the held iterations' W/D/L must POOL into the step's
    winrate/se, not just the count. A healthy accumulated sample whose boundary
    iter happens to be a tiny 0% sweep must NOT score the levers (or the airbag,
    which reads raw_wr, not the EMA) on that lone 0% iter. The count-only floor
    left this open — the crossing iteration's raw_wr alone drove the step."""
    pid = _mk_pid(min_games_for_adjust=30, initial_wdl_regret=0.15,
                  regret_safety_floor=0.50)

    # 4 held iters of 6 games at ~67% (4W/2L) = 24 games, all below the 30 floor.
    for _ in range(4):
        u = pid.observe(wins=4, draws=0, losses=2, force=True)
        assert not u.adjusted                          # held, accumulating

    # Boundary iter: 6 games at a 0% sweep pushes the count to 30 and steps.
    u = pid.observe(wins=0, draws=0, losses=6, force=True)
    # The step scored the POOLED 16W/14L = 0.533 sample, not the 0% boundary iter.
    assert abs(u.raw_winrate - 16 / 30) < 1e-9
    assert u.regret_diag is not None
    # 0.533 + 1.5*se is well above the 0.50 floor → the airbag did NOT fire.
    # (On the count-only floor the 0% boundary iter would give reason="airbag".)
    assert u.regret_diag.reason != "airbag"
    assert pid._games_since_adjust == 0
    assert pid._held_wins == 0 and pid._held_losses == 0    # pooled sample reset


def test_held_sample_survives_state_dict_roundtrip():
    """A restart mid-hold must not drop the accumulated sample: held W/D/L round-
    trip through state_dict so the post-restart step still scores the full pool."""
    pid = _mk_pid(min_games_for_adjust=30)
    pid.observe(wins=4, draws=0, losses=2, force=True)   # held; pool = 4/0/2
    pid.observe(wins=4, draws=0, losses=2, force=True)   # held; pool = 8/0/4
    state = pid.state_dict()
    assert state["held_wins"] == 8 and state["held_losses"] == 4

    pid2 = _mk_pid(min_games_for_adjust=30)
    pid2.load_state_dict(state)
    assert pid2._held_wins == 8 and pid2._held_losses == 4
    assert pid2._games_since_adjust == 12
