"""Tests for the inverse-model regret controller in DifficultyPID."""
from __future__ import annotations

import math
import random

from chess_anti_engine.stockfish.pid import (
    DifficultyPID,
    _fit_inverse_regret,
    _observation_se,
)


def _mk_pid(**overrides) -> DifficultyPID:
    defaults = dict(
        initial_nodes=5000,
        target_winrate=0.57,
        ema_alpha=0.5,
        min_nodes=5000,
        max_nodes=25000,
        initial_wdl_regret=0.15,
        wdl_regret_min=0.02,
        wdl_regret_max=1.0,
        inverse_regret_window=10,
        inverse_regret_sigma_tolerance=0.02,
        inverse_regret_max_step=0.01,
        inverse_regret_safety_floor=0.50,
        inverse_regret_safety_band=0.05,
        inverse_regret_emergency_ease_step=0.01,
    )
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


def test_observation_se_matches_known_distribution():
    # Pure win: raw variance 0 but SE is floored to prevent infinite weight.
    # (See Codex adversarial-review finding #2.)
    se_pure = _observation_se(10, 0, 0)
    assert se_pure > 0.0  # floor applied, no longer 0
    # 50/50 binary: Bernoulli variance is 0.25, SE = sqrt(0.25 / n).
    # At n=800 this is ≈ 0.0177, well above the floor of 0.01.
    se = _observation_se(400, 0, 400)
    assert abs(se - math.sqrt(0.25 / 800 * (800 / 799))) < 1e-6
    # Draw-heavy should have lower raw variance per game than 50/50 wins/losses,
    # but both exceed the floor at n=800.
    se_draws = _observation_se(0, 800, 0)  # pure draws → variance 0, floored
    se_mixed = _observation_se(400, 0, 400)
    # Floored all-draw SE equals the floor; mixed stays above the floor.
    assert se_mixed > se_draws


def test_fit_recovers_linear_relationship():
    # Synthetic linear data: winrate = 0.4 + 2.0 * regret
    history = [
        (0.10, 0.60, 0.018),
        (0.12, 0.64, 0.018),
        (0.14, 0.68, 0.018),
        (0.16, 0.72, 0.018),
        (0.18, 0.76, 0.018),
    ]
    fit = _fit_inverse_regret(history, target_wr=0.70)
    assert fit is not None
    predicted_r, sigma_pred, slope = fit
    # Target 0.70 should map to regret 0.15 (slope 2, intercept 0.4)
    assert abs(predicted_r - 0.15) < 0.005
    assert abs(slope - 2.0) < 0.1
    assert sigma_pred < 0.05  # tight fit


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
        inverse_regret_max_step=0.01,
        inverse_regret_emergency_ease_step=0.01,
        inverse_regret_safety_floor=0.50,
        inverse_regret_safety_band=0.05,
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


def test_sigma_pred_reflects_residual_variance():
    # Codex HIGH finding #2 (continued): residual variance must flow into
    # sigma_pred. Two fits with identical regret spread but different residuals
    # should give different predicted-winrate uncertainty.
    tight_history = [
        (0.10, 0.60, 0.018),
        (0.12, 0.64, 0.018),
        (0.14, 0.68, 0.018),
        (0.16, 0.72, 0.018),
    ]
    noisy_history = [
        (0.10, 0.70, 0.018),   # residual far from linear trend
        (0.12, 0.50, 0.018),
        (0.14, 0.80, 0.018),
        (0.16, 0.55, 0.018),
    ]
    tight = _fit_inverse_regret(tight_history, target_wr=0.66)
    noisy = _fit_inverse_regret(noisy_history, target_wr=0.66)
    assert tight is not None
    # Noisy fit may return None (sign-check reject) or very high sigma_pred.
    # If it returned a fit, sigma_pred should exceed the tight fit's.
    if noisy is not None:
        assert noisy[1] > tight[1]
    assert tight[1] < 0.05  # tight residuals → low prediction uncertainty


def test_inverse_history_round_trips_via_state_dict():
    # Codex MED finding #3: history must persist across save/load.
    pid1 = _mk_pid(initial_wdl_regret=0.15)
    _feed_observations(pid1, [
        (0.10, 0.55),
        (0.12, 0.58),
        (0.14, 0.62),
        (0.16, 0.66),
        (0.18, 0.70),
    ], n_games=800)
    state = pid1.state_dict()
    assert "inverse_history" in state
    assert len(state["inverse_history"]) == 5

    # New PID, restore state — history should come back.
    pid2 = _mk_pid(initial_wdl_regret=0.15)
    assert len(pid2._inverse_history) == 0
    pid2.load_state_dict(state)
    assert len(pid2._inverse_history) == 5
    # Entries should match (within float precision)
    for (r1, w1, s1), (r2, w2, s2) in zip(pid1._inverse_history, pid2._inverse_history):
        assert abs(r1 - r2) < 1e-12
        assert abs(w1 - w2) < 1e-12
        assert abs(s1 - s2) < 1e-12


def test_legacy_checkpoint_without_inverse_history_loads_gracefully():
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
    assert len(pid._inverse_history) == 0  # remains empty, no crash


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
    pid = _mk_pid(initial_wdl_regret=0.15, inverse_regret_max_step=0.003)
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
    fit = _fit_inverse_regret(history, target_wr=0.70)
    assert fit is not None
    predicted_r, sigma_pred, slope = fit
    # Target 0.70: true r* = 0.15. Noise should allow fit within ±0.02
    assert abs(predicted_r - 0.15) < 0.03
    assert abs(slope - true_slope) < 0.5
