"""The SF-approved-move probability floor: its SET, its wiring, and its inertness.

The eight semantic cases below are ports of the hand-built checks that pinned
the offline reference implementation. They are written against the SET the term
selects, not against a golden loss value, because the set is the design: every
one of them fails for a different single-clause mutation of

    top1 = argmin over LEGAL moves of regret
    F    = {top1} u {m : regret_m <= delta_cp/CAP AND regret_m < regret_ours}

⚑ A NEW TEST FILE PLUS A GREEN LINT HAS CERTIFIED A CRASH HERE BEFORE. These
cases were mutation-tested clause by clause -- drop the unconditional `top1`,
relax `<` to `<=`, drop the `better_than_ours` gate, drop the cp window, divide
by 1 instead of the cap, drop the `have` mask, drop the legality restriction on
the argmin, floor `top1` at `tau` instead of `tau_top1` -- and each mutant is
killed by at least one assertion here. The mutant table is in the PR.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from chess_anti_engine.selfplay import finalize
from chess_anti_engine.train.constants import (
    DEFAULT_GUMBEL_TOPK,
    SF_OWN_REGRET_CAP_CP,
    normalize_gumbel_topk,
)
from chess_anti_engine.train.losses import (
    SF_POLICY_FLOOR_TAU_DEFAULT,
    SfPolicyFloorParams,
    apply_policy_mask_to_logits,
    compute_loss,
    policy_legal_bool,
    search_inclusion_guarantee_tau,
    sf_policy_floor_deficit,
)

CAP = SF_OWN_REGRET_CAP_CP


def R(cp: float) -> float:
    """A cp regret in the units `sf_p0_regret` is stored in."""
    return cp / CAP


def floor_loss(
    regret: torch.Tensor,
    logits: torch.Tensor,
    *,
    delta_cp: float = 20.0,
    tau: float = 0.15,
    tau_top1: float | None = None,
    tau_played: float | None = 0.0,
    played_target: torch.Tensor | None = None,
    legal: torch.Tensor | None = None,
    have: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor, float]:
    """(loss over covered rows, grad wrt logits, binding rate) -- the rig's shape.

    Mirrors `compute_loss`'s own reduction: mean over covered rows of the
    per-row deficit, with the row mask applied to the per-row tensors rather
    than to the selection.

    ``tau_played`` defaults to 0.0 here -- COLLAR OFF -- so the eight semantic
    cases below exercise the SF set alone; the collar has its own cases, and
    folding it into every fixture would make each of them test two things.
    """
    lg = logits.clone().requires_grad_(True)
    probs = torch.softmax(lg if legal is None else lg.masked_fill(~legal, -1e9), dim=-1)
    params = SfPolicyFloorParams.resolve(
        delta_cp=delta_cp, tau=tau, tau_top1=tau_top1, tau_played=tau_played,
    )
    deficit, binds = sf_policy_floor_deficit(
        probs, regret, legal, played_target, params=params,
    )
    mask = torch.ones(regret.shape[0]) if have is None else have.to(torch.float32)
    denom = mask.sum().clamp_min(1.0)
    loss = (deficit * mask).sum() / denom
    (grad,) = torch.autograd.grad(loss, [lg])
    return float(loss.detach()), grad, float(((binds * mask).sum() / denom).detach())


# --------------------------------------------------------------------------
# The eight semantic cases.
# --------------------------------------------------------------------------

# regret layout shared by several cases: move 0 is SF's best, 1 is 10cp worse,
# 2 is 30cp worse (OUTSIDE a 20cp window), 3 is 500cp worse, 4/5 are unsurfaced.
_REG = torch.tensor([[R(0), R(10), R(30), R(500), 0.55, 0.55]])


def test_case1_our_argmax_is_sfs_best_and_above_tau_is_exactly_silent() -> None:
    """Our pick IS SF's best and clears tau -> loss EXACTLY 0, grad EXACTLY 0.

    Not `< 1e-6`: the term is one-sided, so on a correct confident row it must
    contribute no gradient AT ALL. An `approx` here would pass for a term that
    quietly nudges every row.
    """
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    loss, grad, binds = floor_loss(_REG, logits)
    assert loss == 0.0
    assert float(grad.abs().sum()) == 0.0
    assert binds == 0.0


def test_case2_set_is_top1_union_within_window_and_better() -> None:
    """Our argmax is move 2 (30cp, outside the window). F = {0, 1}, exactly."""
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    loss, _, binds = floor_loss(_REG, logits)
    p = torch.softmax(logits, dim=-1)[0]
    expected = float(torch.relu(0.15 - p[0]) + torch.relu(0.15 - p[1]))
    assert loss == pytest.approx(expected, abs=1e-6)
    # Move 3 (500cp) is better than ours and would enter without the window.
    assert float(torch.relu(0.15 - p[3])) > 0.0
    assert binds == 1.0


def test_case3_better_than_ours_gate_excludes_our_own_pick() -> None:
    """Our argmax is move 1 (10cp, inside the window). F = {top1} only.

    Move 1 is not strictly better than itself, so the adaptive set is empty and
    the term reduces to the unconditional top-1 floor.
    """
    logits = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]])
    loss, _, _ = floor_loss(_REG, logits)
    p = torch.softmax(logits, dim=-1)[0]
    assert loss == pytest.approx(float(torch.relu(0.15 - p[0])), abs=1e-6)


def test_case4_default_regret_moves_can_never_enter_the_set() -> None:
    """Unsurfaced / illegal entries carry `(worst + 1)/2 >= 0.5` and stay out.

    This is the property that lets the cp window alone do the exclusion, so it
    is asserted against `_build_sf_p0_regret_vector`'s actual fill rule below
    (`test_the_default_regret_fill_is_above_every_realistic_window`) rather than
    only against a hand-written 0.5.
    """
    regret = torch.tensor([[R(0), 0.5, 0.5, 0.5, 0.5, 0.5]])
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 5.0]])  # ours is unsurfaced
    loss, _, _ = floor_loss(regret, logits)
    p = torch.softmax(logits, dim=-1)[0]
    assert loss == pytest.approx(float(torch.relu(0.15 - p[0])), abs=1e-6)


def test_case5_uncovered_rows_contribute_nothing() -> None:
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    loss, grad, binds = floor_loss(_REG, logits, have=torch.zeros(1, dtype=torch.bool))
    assert loss == 0.0
    assert float(grad.abs().sum()) == 0.0
    assert binds == 0.0


def test_case6_top1_is_unconditional_and_binds_onto_our_own_correct_move() -> None:
    """Our argmax IS top1 but sits BELOW tau -> the floor binds onto it.

    This is the case that proves `top1` enters F unconditionally: without the
    scatter, F is empty here and the loss is 0. Correct behaviour is to push
    mass ONTO the move we already got right, never off it.
    """
    logits = torch.tensor([[0.30, 0.0, 0.0, 0.0, 0.0, 0.0]])
    loss, _, binds = floor_loss(_REG, logits, tau=0.25)
    p = torch.softmax(logits, dim=-1)[0]
    assert int(p.argmax()) == 0, "setup: our argmax must be SF's best"
    expected = float(torch.relu(0.25 - p[0]))
    assert expected > 0.0, "setup: the floor must actually bind"
    assert loss == pytest.approx(expected, abs=1e-6)
    assert binds == 1.0


def test_case7_a_move_tied_with_our_pick_inside_the_window_is_excluded() -> None:
    """`<`, not `<=`. cp scores are integer-quantised, so ties are common.

    ⚑ THE TIE MUST SIT INSIDE THE cp WINDOW. Put it outside and the window
    rejects it first, the tie clause is never exercised, and the test is
    vacuous -- which is exactly how `<=` survived a first round of review.
    """
    regret = torch.tensor([[R(0), R(10), R(10), R(900), 0.95, 0.95]])
    logits = torch.tensor([[0.0, 3.0, -3.0, 0.0, 0.0, 0.0]])  # ours = 1, move 2 tied
    loss, _, _ = floor_loss(regret, logits)
    p = torch.softmax(logits, dim=-1)[0]
    assert float(torch.relu(0.15 - p[2])) > 0.0, "setup: the tied move must be below tau"
    assert regret[0, 2] <= 20.0 / CAP, "setup: the tied move must be INSIDE the window"
    assert loss == pytest.approx(float(torch.relu(0.15 - p[0])), abs=1e-6)


def test_case8_the_window_is_in_cp_over_the_cap_not_raw_cp() -> None:
    """A move 100cp better than ours is still outside a 20cp window."""
    regret = torch.tensor([[R(0), R(100), R(200), R(900), 0.95, 0.95]])
    logits = torch.tensor([[0.0, -3.0, 3.0, 0.0, 0.0, 0.0]])  # ours = move 2 (200cp)
    loss, _, _ = floor_loss(regret, logits)
    p = torch.softmax(logits, dim=-1)[0]
    assert float(torch.relu(0.15 - p[1])) > 0.0, "setup: the far move must be below tau"
    assert loss == pytest.approx(float(torch.relu(0.15 - p[0])), abs=1e-6)


# --------------------------------------------------------------------------
# The properties the eight cases rest on.
# --------------------------------------------------------------------------


def test_the_default_regret_fill_is_above_every_realistic_window() -> None:
    """Case 4's premise, read off the WRITER instead of assumed.

    `_build_sf_p0_regret_vector` fills unsurfaced and illegal indices with
    `(worst_surfaced + 1) / 2`, so the floor of that fill is 0.5 -- 25x the
    `delta_cp/CAP = 0.02` window at the default 20cp. If that rule ever changes,
    the cp window stops excluding illegal moves on its own and this test says so
    before the loss does.
    """
    import numpy as np

    # One surfaced move at regret 0 -> the smallest possible default fill.
    rows = np.array([[0, 0, 0, 0, 0]], dtype=np.int16)
    vec = finalize._build_sf_p0_regret_vector(rows, policy_encoding="lc0_1858")
    assert vec is not None
    assert float(vec.min()) == 0.0
    unsurfaced = float(np.sort(np.unique(vec))[1])
    assert unsurfaced >= 0.5
    assert unsurfaced > 20.0 / CAP


def test_illegal_moves_are_excluded_even_at_an_absurd_delta() -> None:
    """The legality clause, isolated with the cp window switched OFF.

    At `delta_cp = CAP` the window admits everything, so this is the only test
    in which the `& legal` term in `within` is load-bearing -- and it is also
    the only one that can see the argmin landing on an illegal move.
    """
    regret = torch.tensor([[R(300), R(0), R(10), R(20), R(30), R(40)]])
    legal = torch.tensor([[True, False, False, True, True, True]])
    logits = torch.zeros(1, 6)
    loss, _, _ = floor_loss(regret, logits, delta_cp=CAP, legal=legal, tau=0.5)
    # Moves 1 and 2 carry the two LOWEST regrets in the row (0cp and 10cp) and
    # are illegal, so they are the ones an unguarded argmin or an unguarded
    # window would seize on -- and they sit at p == 0, so admitting either shows
    # up as a whole extra `tau` of deficit.
    p = torch.softmax(logits.masked_fill(~legal, -1e9), dim=-1)[0]
    assert float(p[1]) == 0.0
    assert float(p[2]) == 0.0
    # legal = {0, 3, 4, 5} at p = 0.25 each; ours = 0 (300cp); top1 = 3 (20cp);
    # the window is wide open, so F = {3, 4, 5} -- all three better than ours.
    assert loss == pytest.approx(3 * float(torch.relu(0.5 - p[3])), abs=1e-6)
    assert loss == pytest.approx(0.75, abs=1e-6)


def test_top1_can_carry_its_own_floor() -> None:
    """`tau_top1` is independently settable and applies to SF's best move only."""
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    p = torch.softmax(logits, dim=-1)[0]
    loss, _, _ = floor_loss(_REG, logits, tau=0.15, tau_top1=0.40)
    expected = float(torch.relu(0.40 - p[0]) + torch.relu(0.15 - p[1]))
    assert loss == pytest.approx(expected, abs=1e-6)


def test_tau_top1_below_tau_floors_sfs_best_LOWER_when_we_already_picked_it() -> None:
    """⚑ F5. The MAX rule's asymmetry, executed rather than argued.

    SF's top-1 enters `adaptive` only when our argmax is something ELSE (the
    strict `regret < our_r` clause excludes a move from beating itself). So with
    `tau_top1` BELOW `tau`, SF's best move is floored:

    * at `tau_top1` alone when our argmax IS it -- the rows invariant 1 calls
      the whole mechanism;
    * at `max(tau, tau_top1)` when our argmax is wrong.

    An earlier docstring claimed `tau_top1 < tau` "cannot pull SF's best move
    under the floor its F-membership already earns it", which is circular and
    backwards. Inert at the shipped default (`tau_top1: null -> tau`), so this
    pins a documented property, not a behaviour anyone currently runs.
    """
    tau, tau_top1 = 0.50, 0.10

    # (a) our argmax IS SF's best (index 0), and sits below both thresholds.
    ours_right = torch.tensor([[0.30, 0.0, 0.0, 0.0, 0.0, 0.0]])
    p_right = torch.softmax(ours_right, dim=-1)[0]
    assert int(p_right.argmax()) == 0
    loss_right, _, _ = floor_loss(_REG, ours_right, tau=tau, tau_top1=tau_top1)
    # Index 0's threshold is tau_top1 alone; move 1 (10cp) is NOT better than
    # ours (0cp), so `adaptive` is empty and contributes nothing.
    assert loss_right == pytest.approx(float(torch.relu(tau_top1 - p_right[0])), abs=1e-6)

    # (b) our argmax is move 2 (30cp, outside the window) -> index 0 joins the
    # adaptive set and its threshold rises to max(tau, tau_top1).
    ours_wrong = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    p_wrong = torch.softmax(ours_wrong, dim=-1)[0]
    loss_wrong, _, _ = floor_loss(_REG, ours_wrong, tau=tau, tau_top1=tau_top1)
    expected = float(torch.relu(tau - p_wrong[0]) + torch.relu(tau - p_wrong[1]))
    assert loss_wrong == pytest.approx(expected, abs=1e-6)
    # The asymmetry itself, as one comparison: SF's best is floored HIGHER on
    # the row where we got it wrong.
    assert max(tau, tau_top1) > tau_top1


def test_binding_rate_is_a_row_rate_not_a_move_rate() -> None:
    """Two covered rows, one binding -> 0.5, whatever the per-row move count."""
    regret = _REG.repeat(2, 1)
    logits = torch.tensor([
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # ours is SF's best and clears tau
        [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],   # two moves below tau
    ])
    _, _, binds = floor_loss(regret, logits)
    assert binds == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The COLLAR: the played move keeps its root-candidate slot.
# --------------------------------------------------------------------------


def _one_hot(index: int, width: int = 6) -> torch.Tensor:
    out = torch.zeros(1, width)
    out[0, index] = 1.0
    return out


def test_case9_the_collar_floors_the_move_search_actually_played() -> None:
    """The played move is protected even though it is in neither F nor our argmax.

    Move 4 is unsurfaced (default regret), so the SF set can never contain it;
    without the collar the term is free to squeeze it, and squeezing it below
    `1/topk` drops it out of the root candidate set on 2.67% of production rows.
    """
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, -2.0, 0.0]])
    p = torch.softmax(logits, dim=-1)[0]
    assert float(p[4]) < 0.0625, "setup: the played move must be under the collar"
    without, _, _ = floor_loss(_REG, logits)
    with_collar, _, _ = floor_loss(
        _REG, logits, tau_played=0.0625, played_target=_one_hot(4),
    )
    assert with_collar == pytest.approx(
        without + float(torch.relu(0.0625 - p[4])), abs=1e-6,
    )


def test_case10_tau_played_zero_disables_the_collar_exactly() -> None:
    """The ablation arm is a clean no-op, not a special case."""
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, -2.0, 0.0]])
    without, grad_without, _ = floor_loss(_REG, logits)
    ablated, grad_ablated, _ = floor_loss(
        _REG, logits, tau_played=0.0, played_target=_one_hot(4),
    )
    assert ablated == without
    assert torch.equal(grad_ablated, grad_without)


def test_case11_a_played_move_that_is_also_in_f_is_floored_once_at_the_max() -> None:
    """⚑ NO DOUBLE COUNT. One relu per move, at the MAX of its thresholds.

    The played move here is SF's top-1 AND our own argmax is elsewhere, so the
    move carries the F threshold (tau) and the collar threshold (tau_played) at
    once. Summing them would floor it at 0.2125; letting the first one win would
    floor it at whichever the implementation happened to apply last. The exact
    expected value below distinguishes all three.
    """
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])  # ours = move 2
    p = torch.softmax(logits, dim=-1)[0]
    tau, tau_played = 0.15, 0.0625
    loss, _, _ = floor_loss(
        _REG, logits, tau=tau, tau_played=tau_played, played_target=_one_hot(0),
    )
    # F = {0, 1} as in case 2; move 0 is ALSO the played move.
    expected = float(torch.relu(max(tau, tau_played) - p[0]) + torch.relu(tau - p[1]))
    assert loss == pytest.approx(expected, abs=1e-6)
    # The two mutants this exists to kill, stated as numbers:
    summed = float(torch.relu(tau + tau_played - p[0]) + torch.relu(tau - p[1]))
    collar_wins = float(torch.relu(tau_played - p[0]) + torch.relu(tau - p[1]))
    assert loss != pytest.approx(summed, abs=1e-6)
    assert loss != pytest.approx(collar_wins, abs=1e-6)


def test_case12_the_collar_is_silent_above_its_threshold() -> None:
    """A comfortably-ranked played move adds nothing."""
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    p = torch.softmax(logits, dim=-1)[0]
    assert float(p[2]) > 0.0625, "setup: the played move must clear the collar"
    without, _, _ = floor_loss(_REG, logits)
    collared, _, _ = floor_loss(
        _REG, logits, tau_played=0.0625, played_target=_one_hot(2),
    )
    assert collared == without


def test_a_row_with_no_policy_target_gets_no_collar() -> None:
    """An absent or masked-out target argmaxes to index 0; it must not be collared.

    This is the difference between "the played move" and "index 0 of a zero
    vector", and without the mass check they are the same tensor.
    """
    logits = torch.tensor([[-4.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    p = torch.softmax(logits, dim=-1)[0]
    assert float(p[0]) < 0.0625
    without, _, _ = floor_loss(_REG, logits)
    zeroed, _, _ = floor_loss(
        _REG, logits, tau_played=0.5, played_target=torch.zeros(1, 6),
    )
    assert zeroed == without


def test_the_collar_reads_the_policy_target_not_the_nets_argmax() -> None:
    """⚑ THE MEMBER IS THE PLAYED MOVE. The net's argmax cannot be squeezed out.

    Same row, two different `played_target`s; if the implementation quietly used
    the net's own argmax instead, both calls would return the same number.
    """
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, -2.0, -2.0]])
    a, _, _ = floor_loss(_REG, logits, tau_played=0.0625, played_target=_one_hot(4))
    b, _, _ = floor_loss(_REG, logits, tau_played=0.0625, played_target=_one_hot(2))
    assert a > b


# --------------------------------------------------------------------------
# tau's derivation from the search width.
# --------------------------------------------------------------------------


def test_tau_defaults_to_the_ranking_calibrated_value() -> None:
    """0.15, above the guarantee -- tau's second role, and it is topk-independent."""
    assert SF_POLICY_FLOOR_TAU_DEFAULT == 0.15
    for topk in (8, 16, 32):
        params = SfPolicyFloorParams.resolve(gumbel_topk=topk)
        assert params.tau == pytest.approx(SF_POLICY_FLOOR_TAU_DEFAULT)
        assert params.tau_top1 == pytest.approx(SF_POLICY_FLOOR_TAU_DEFAULT)


def test_an_explicit_tau_overrides_the_default() -> None:
    params = SfPolicyFloorParams.resolve(tau=0.2)
    assert params.tau == pytest.approx(0.2)
    # `tau_top1: null` follows the RESOLVED tau, not the dataclass default.
    assert params.tau_top1 == pytest.approx(0.2)
    assert SfPolicyFloorParams.resolve(tau=0.2, tau_top1=0.5).tau_top1 == pytest.approx(0.5)


def test_the_search_inclusion_guarantee_is_one_over_topk() -> None:
    """The rank argument, exercised as arithmetic rather than asserted.

    If `p_i >= 1/topk` then fewer than `topk` moves can strictly exceed it (they
    would be disjoint and sum past 1), so move `i` is inside the top-`topk` by
    prior. That is what makes it a GUARANTEE rather than a heuristic.
    """
    for topk in (4, 8, 16):
        tau = search_inclusion_guarantee_tau(topk)
        assert tau == pytest.approx(1.0 / topk)
        assert math.floor(1.0 / tau) - 1 <= topk - 1
    # Normalized exactly like the search's own width, including the >= 1 clamp.
    assert search_inclusion_guarantee_tau(0) == pytest.approx(1.0)
    assert search_inclusion_guarantee_tau(DEFAULT_GUMBEL_TOPK) == pytest.approx(0.0625)


def test_the_guard_warns_when_tau_falls_below_the_inclusion_guarantee() -> None:
    """⚑ THE GUARD MUST BE REACHABLE. A guard that cannot fire is worse than none.

    Reached here by NARROWING `gumbel_topk` (which RAISES the guarantee
    `1/topk`) rather than by lowering tau, because that is the way it fires
    WITHOUT anyone editing the floor's own keys -- a search-width change
    silently revoking the guarantee is the whole failure this warns about.
    topk 4 -> guarantee 0.25, above the default tau of 0.15.
    """
    with pytest.warns(RuntimeWarning, match="BELOW the root-search inclusion guarantee"):
        SfPolicyFloorParams.resolve(gumbel_topk=4)
    # And the direct route: an explicit sub-guarantee tau at the production width.
    with pytest.warns(RuntimeWarning, match="gumbel_topk=16"):
        SfPolicyFloorParams.resolve(tau=0.02, gumbel_topk=16)


def test_the_guard_fires_on_tau_played_the_parameter_it_exists_for() -> None:
    """⚑ F2. The first version of this guard checked `tau` ONLY.

    `tau` is the RANKING knob at 0.15 -- 2.4x the production guarantee, so it
    essentially never trips -- while `tau_played`, whose documented sole job IS
    the inclusion guarantee, was never looked at. `resolve(tau=0.15,
    tau_played=0.001, gumbel_topk=16)` returned in silence: a guard aimed at the
    parameter that cannot lose the property and blind on the one that can.

    The realistic trigger is the collar-strength arm: someone sets
    `sf_policy_floor_tau_played: 0.03` and the collar quietly stops guaranteeing
    the played move a root-candidate slot.
    """
    with pytest.warns(RuntimeWarning, match="sf_policy_floor_tau_played"):
        SfPolicyFloorParams.resolve(w=1.0, tau=0.15, tau_played=0.001, gumbel_topk=16)
    with pytest.warns(RuntimeWarning, match="sf_policy_floor_tau_played"):
        _round_trip({"sf_policy_floor_tau_played": 0.03})


def test_the_collar_ablation_is_silent_because_zero_is_an_off_switch() -> None:
    """`tau_played: 0.0` is the documented ablation arm, not a broken floor.

    ⚑ THE NEGATIVE CONTROL THAT SHAPES THE PREDICATE. Without it the natural
    `tau_played < guarantee` check would warn on every run of the ablation arm,
    and a guard that cries on its own control gets filtered out -- which is how
    a guard stops being read at all. Hence `0 < tau_played < guarantee`.
    """
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        got = SfPolicyFloorParams.resolve(w=1.0, tau=0.15, tau_played=0.0, gumbel_topk=16)
    assert got.tau_played == 0.0


def test_the_guard_is_silent_while_the_default_tau_still_guarantees_inclusion() -> None:
    """The negative control on the guard: topk 8 and 16 must NOT warn at tau 0.15."""
    import warnings as _warnings

    for topk in (8, 16):
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", RuntimeWarning)
            got = SfPolicyFloorParams.resolve(gumbel_topk=topk)
        assert got.tau == 0.15
        # ...and the DERIVED collar sits exactly at the guarantee, so it is
        # never the thing that trips its own guard.
        assert got.tau_played == pytest.approx(1.0 / topk)


# --------------------------------------------------------------------------
# Validation: every one of the four keys is range-checked.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "key"),
    [
        ({"w": -1.0}, "w_sf_policy_floor"),
        ({"w": float("nan")}, "w_sf_policy_floor"),
        ({"delta_cp": -1.0}, "sf_policy_floor_delta_cp"),
        ({"delta_cp": float("inf")}, "sf_policy_floor_delta_cp"),
        ({"tau": -0.01}, "sf_policy_floor_tau"),
        ({"tau": 1.01}, "sf_policy_floor_tau"),
        ({"tau": float("nan")}, "sf_policy_floor_tau"),
        ({"tau_top1": 2.0}, "sf_policy_floor_tau_top1"),
        ({"tau_top1": -0.5}, "sf_policy_floor_tau_top1"),
        ({"tau_played": 1.5}, "sf_policy_floor_tau_played"),
        ({"tau_played": -0.1}, "sf_policy_floor_tau_played"),
        ({"tau_played": float("nan")}, "sf_policy_floor_tau_played"),
    ],
)
def test_out_of_range_values_are_rejected_by_name(kwargs: dict, key: str) -> None:
    with pytest.raises(ValueError, match=key):
        SfPolicyFloorParams.resolve(**kwargs)


def test_the_range_check_accepts_the_endpoints() -> None:
    """A band that rejected 0.0 or 1.0 would be a different band than documented."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)  # tau 0.0 trips the guard
        assert SfPolicyFloorParams.resolve(
            w=0.0, delta_cp=0.0, tau=0.0, tau_top1=1.0, tau_played=0.0,
        )


def test_a_live_weight_push_is_validated_by_replace() -> None:
    """The live path re-runs the validator, so a bad `w` cannot arrive raw.

    `_apply_lr_gamma_weights` pushes `w_sf_policy_floor` by `setattr`, which no
    validator sees; `_loss_kwargs` re-stamps it with `dataclasses.replace`,
    which does re-run `__post_init__`. If that ever stops being true this fails.
    """
    from dataclasses import replace

    params = SfPolicyFloorParams.resolve()
    with pytest.raises(ValueError, match="w_sf_policy_floor"):
        replace(params, w=-1.0)


# --------------------------------------------------------------------------
# Wiring: the config round-trip, and inertness at the default.
# --------------------------------------------------------------------------


def _flat(overrides: dict, *, section: str = "train") -> dict:
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    return flatten_run_config_defaults({section: overrides})


def _round_trip(overrides: dict, *, section: str = "train") -> object:
    from chess_anti_engine.tune.trial_config import TrialConfig

    return TrialConfig.from_dict(_flat(overrides, section=section))


class _TinyModel(torch.nn.Module):
    """Smallest model a `Trainer` will build an optimizer over.

    Same shape as `tests/test_wdl_terminal_outcome.py`'s, which is the in-repo
    template for a config -> Trainer -> `_loss_kwargs` wiring test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32),
        }


# The core keys every `Trainer` needs, kept OUT of the section under test so a
# `selfplay:` override cannot smuggle `lr` into the selfplay allowlist.
_TRAINER_CORE = {"device": "cpu", "lr": 1e-3, "no_amp": True}


def _trainer_flat(overrides: dict, *, section: str = "train") -> dict:
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    cfg: dict = {"train": dict(_TRAINER_CORE)}
    cfg.setdefault(section, {}).update(overrides)
    return flatten_run_config_defaults(cfg)


def _trainer(overrides: dict, tmp_path, *, section: str = "train"):
    from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config

    ctor = trainer_kwargs_from_config(
        _trainer_flat(overrides, section=section), log_dir=tmp_path,
    )
    return Trainer(_TinyModel(), prefetch_batches=False, **ctor)


# ⚑ THE TEST THIS SECTION REPLACED WAS THE HAZARD ITS OWN SUBJECT WARNS ABOUT.
# It stopped at `trainer_kwargs_from_config` and then called
# `SfPolicyFloorParams.resolve(...)` in the test body -- "resolve, then re-derive
# at the call site", reproduced inside the test written to prevent it. Its name
# said "reach the trainer" and it never constructed one, so an independent
# reviewer's mutants that made `Trainer.__init__` ignore the configured `tau`,
# `delta_cp` or `tau_played`, that deleted the `_loss_kwargs` key outright, and
# that swallowed the live weight push ALL SURVIVED. Everything below asserts on
# the object `compute_loss` is actually handed, against LITERAL expected values.


def test_the_yaml_keys_reach_the_loss_call(tmp_path) -> None:
    """config -> Trainer -> `_loss_kwargs` -> `compute_loss`, end to end.

    `_loss_kwargs` is the dict the training step splats into `compute_loss`
    (both call sites), so the object under that key IS what the loss receives.
    Compared against a literal `SfPolicyFloorParams`, never against a second
    call to `resolve`: a re-derivation in the test body agrees with a trainer
    that ignored the config entirely.
    """
    trainer = _trainer({
        "w_sf_policy_floor": 0.41,
        "sf_policy_floor_delta_cp": 35.0,
        "sf_policy_floor_tau": 0.31,
        "sf_policy_floor_tau_top1": 0.22,
        "sf_policy_floor_tau_played": 0.09,
    }, tmp_path)

    assert trainer._loss_kwargs["sf_policy_floor"] == SfPolicyFloorParams(
        w=0.41, delta_cp=35.0, tau=0.31, tau_top1=0.22, tau_played=0.09,
    )

    # And the dict as a whole is accepted by the function it is splatted into
    # AND the configured SHAPE reaches the arithmetic. Presence of the key would
    # be satisfied by a trainer that forwarded an all-defaults object, which is
    # exactly reviewer mutant R4; the inequality below is not.
    outputs, batch = _tiny_batch()
    got = compute_loss(outputs, batch, **trainer._loss_kwargs)
    # Built as a plain copy rather than a `{**kw, "k": v}` literal: the literal
    # widens every value's inferred type to a union and basedpyright then
    # rejects the splat against `compute_loss`'s real signature.
    default_kwargs = dict(trainer._loss_kwargs)
    default_kwargs["sf_policy_floor"] = SfPolicyFloorParams()
    defaults = compute_loss(outputs, batch, **default_kwargs)
    assert float(got["sf_policy_floor_sum"].detach()) != float(
        defaults["sf_policy_floor_sum"].detach()
    )
    assert float(got["total"].detach()) != float(defaults["total"].detach())


def test_the_defaults_reach_the_loss_call_and_are_off(tmp_path) -> None:
    """An empty config must not enable the floor on the production path."""
    floor = _trainer({}, tmp_path)._loss_kwargs["sf_policy_floor"]
    assert floor == SfPolicyFloorParams(
        w=0.0,
        delta_cp=20.0,
        tau=SF_POLICY_FLOOR_TAU_DEFAULT,
        tau_top1=SF_POLICY_FLOOR_TAU_DEFAULT,
        tau_played=search_inclusion_guarantee_tau(DEFAULT_GUMBEL_TOPK),
    )


def test_the_collar_default_follows_the_trials_own_gumbel_topk(tmp_path) -> None:
    """`tau_played: null` resolves against the SEARCH's width, through the Trainer."""
    floor = _trainer({"gumbel_topk": 8}, tmp_path, section="selfplay")._loss_kwargs[
        "sf_policy_floor"
    ]
    assert floor.tau_played == pytest.approx(0.125)


def test_a_live_weight_push_reaches_the_loss_kwargs(tmp_path) -> None:
    """`w_sf_policy_floor` rides `TRAINER_WEIGHT_KEYS`; the SHAPE must not move.

    `_apply_lr_gamma_weights` pushes by `setattr`, and `_loss_kwargs` re-stamps
    the attribute onto the frozen params with `replace`. Asserted at
    `_loss_kwargs`, not at the attribute: a trainer that accepts the push and
    never forwards it is exactly the defect this PR is about, and reading back
    `trainer.w_sf_policy_floor` cannot see it.
    """
    from chess_anti_engine.tune.trainable_config_ops import _apply_lr_gamma_weights

    trainer = _trainer({"sf_policy_floor_tau": 0.31}, tmp_path)
    assert trainer._loss_kwargs["sf_policy_floor"].w == 0.0

    _apply_lr_gamma_weights(trainer, {"w_sf_policy_floor": 0.77}, rescale_current_lr=True)
    pushed = trainer._loss_kwargs["sf_policy_floor"]
    assert pushed.w == 0.77
    # The shape is startup-only and must survive the weight push untouched.
    assert pushed.tau == 0.31


def test_a_live_weight_push_is_range_validated(tmp_path) -> None:
    """The `setattr` path has no validator; `replace` in `_loss_kwargs` is it."""
    from chess_anti_engine.tune.trainable_config_ops import _apply_lr_gamma_weights

    trainer = _trainer({}, tmp_path)
    _apply_lr_gamma_weights(trainer, {"w_sf_policy_floor": -1.0}, rescale_current_lr=True)
    with pytest.raises(ValueError, match="w_sf_policy_floor"):
        _ = trainer._loss_kwargs


def test_a_live_gumbel_topk_edit_re_points_the_derived_collar(tmp_path) -> None:
    """F3: `gumbel_topk` is LIVE and `tau_played = 1/topk` was frozen at launch.

    The derived collar must follow the search width, or it goes on guaranteeing
    inclusion in a top-k that no longer exists -- silently, because the key is
    genuinely live for selfplay and the startup-only machinery cannot flag it.
    """
    trainer = _trainer({"gumbel_topk": 16}, tmp_path, section="selfplay")
    assert trainer._loss_kwargs["sf_policy_floor"].tau_played == pytest.approx(0.0625)

    # Narrowing to 4 also puts the (undERIVED) ranking tau under the new
    # guarantee, so the guard fires on THAT key -- asserted, not leaked.
    with pytest.warns(RuntimeWarning, match="sf_policy_floor_tau="):
        trainer.sync_search_width(4)
    assert trainer._loss_kwargs["sf_policy_floor"].tau_played == pytest.approx(0.25)
    # The ranking knob is NOT derived and must not move with the width.
    assert trainer._loss_kwargs["sf_policy_floor"].tau == pytest.approx(
        SF_POLICY_FLOOR_TAU_DEFAULT,
    )


def test_a_pinned_tau_played_is_re_checked_not_re_pointed(tmp_path) -> None:
    """A number an operator typed is theirs to keep -- but it gets a warning.

    Narrowing the search raises the guarantee past a pinned collar. Moving the
    operator's value would be a config edit nobody made; staying silent would be
    a floor that has quietly stopped guaranteeing anything. So: keep, and warn.
    """
    trainer = _trainer({
        "sf_policy_floor_tau_played": 0.07,
    }, tmp_path)
    assert trainer._loss_kwargs["sf_policy_floor"].tau_played == pytest.approx(0.07)

    # topk 8 -> guarantee 0.125, which is ABOVE the pinned 0.07 and BELOW the
    # default tau of 0.15, so exactly one key trips and the assertion cannot be
    # satisfied by the wrong warning.
    with pytest.warns(RuntimeWarning, match="sf_policy_floor_tau_played") as caught:
        trainer.sync_search_width(8)
    assert len(caught) == 1
    assert trainer._loss_kwargs["sf_policy_floor"].tau_played == pytest.approx(0.07)


def test_the_live_sync_calls_the_width_hook(tmp_path) -> None:
    """⚑ THE HOOK MUST BE ON THE PER-ITERATION PATH, not merely callable.

    `sync_search_width` existing and being right proves nothing if nothing calls
    it every iteration -- the shape of every dead knob in this codebase. This
    drives `_sync_trainer_weights`, the real per-iteration entry point.
    """
    from chess_anti_engine.tune.trainable_config_ops import _sync_trainer_weights
    from chess_anti_engine.tune.trial_config import DifficultyState, TrialConfig

    trainer = _trainer({"gumbel_topk": 16}, tmp_path, section="selfplay")
    assert trainer._loss_kwargs["sf_policy_floor"].tau_played == pytest.approx(0.0625)

    flat = _trainer_flat({"gumbel_topk": 8}, section="selfplay")
    _sync_trainer_weights(
        trainer, flat, TrialConfig.from_dict(flat),
        DifficultyState(wdl_regret=0.05, sf_nodes=75000),
    )
    assert trainer._loss_kwargs["sf_policy_floor"].tau_played == pytest.approx(0.125)


_REQUIRED_METRICS = dict.fromkeys(
    ("loss", "policy_loss", "soft_policy_loss", "future_policy_loss", "wdl_loss",
     "sf_move_loss", "sf_move_acc", "sf_eval_loss", "categorical_loss",
     "volatility_loss", "sf_volatility_loss", "moves_left_loss"), 0.0,
)


def test_the_two_columns_reach_train_metrics() -> None:
    """The declared take-effect observation must survive to `TrainMetrics`.

    ⚑ RATIOS OF SUMS, so they travel by a different path from the per-batch
    means and can be half-wired without any loss test noticing. The numerator
    and denominator here are DIFFERENT numbers (3, 2, 4) so a mutant that
    divides by the wrong count, or that pairs the wrong sum with the wrong
    column, cannot land on the right answer by symmetry.
    """
    from chess_anti_engine.train.trainer import Trainer

    # `sf_move_acc` is computed by `_build_metrics` from `acc_sums`, not read
    # from `sums`, so passing it here would collide with its own keyword.
    loss_sums = {k: v for k, v in _REQUIRED_METRICS.items() if k != "sf_move_acc"}
    metrics = Trainer._build_metrics(
        {**loss_sums,
         "sf_policy_floor_sum": 3.0,
         "sf_policy_floor_binds_sum": 2.0,
         "sf_own_regret_rows": 4.0},
        {}, 1.0,
    )
    assert metrics.m_sf_policy_floor == pytest.approx(0.75)
    assert metrics.sf_policy_floor_binds_frac == pytest.approx(0.5)


def test_the_two_columns_reach_the_result_row() -> None:
    """...and out of `TrainMetrics` into the row an operator actually reads."""
    from dataclasses import replace as _replace

    from chess_anti_engine.train.trainer import TrainMetrics
    from chess_anti_engine.tune.trainable_report import _train_metrics_dict

    row = _train_metrics_dict(_replace(
        TrainMetrics(**_REQUIRED_METRICS),
        m_sf_policy_floor=0.75, sf_policy_floor_binds_frac=0.5,
    ))
    assert row["m_sf_policy_floor"] == pytest.approx(0.75)
    assert row["sf_policy_floor_binds_frac"] == pytest.approx(0.5)


def test_the_round_trip_checks_the_guarantee_against_the_trials_own_topk() -> None:
    """The guard fires from `from_dict`, on the width the trial actually runs.

    This is the only test that proves the guard is REACHABLE from a yaml, which
    is the path it exists for -- `resolve` being able to warn says nothing about
    whether the config loader passes it the trial's real search width.
    """
    from chess_anti_engine.tune.trial_config import TrialConfig

    flat = _flat({"gumbel_topk": 4}, section="selfplay")
    assert normalize_gumbel_topk(flat["gumbel_topk"]) == 4
    with pytest.warns(RuntimeWarning, match="gumbel_topk=4"):
        TrialConfig.from_dict(flat)


def test_an_out_of_range_live_value_is_rejected_at_config_load() -> None:
    """CLAUDE.md category (b): the trial dies loudly, naming the key.

    ⚑ THIS IS WHY `from_dict` VALIDATES A KEY IT DOES NOT STORE. The loop
    rebuilds `tc` every iteration from the reloaded yaml, so a typo in a live
    edit raises here instead of reaching the trainer as a raw float.
    """
    with pytest.raises(ValueError, match="sf_policy_floor_tau"):
        _round_trip({"sf_policy_floor_tau": 5.0})
    with pytest.raises(ValueError, match="sf_policy_floor_tau_played"):
        _round_trip({"sf_policy_floor_tau_played": -1.0})
    with pytest.raises(ValueError, match="w_sf_policy_floor"):
        _round_trip({"w_sf_policy_floor": -0.5})


# --------------------------------------------------------------------------
# `policy_legal_bool` must agree with the logit masker, not with a re-reading
# of the same rule.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("with_has", [True, False])
def test_policy_legal_bool_matches_the_logit_maskers_support(with_has: bool) -> None:
    torch.manual_seed(0)
    width = 12
    mask = (torch.rand(3, width) < 0.4).to(torch.float32)
    mask[:, 0] = 1.0
    batch: dict[str, torch.Tensor] = {"x": torch.zeros(3, 1), "legal_mask": mask}
    if with_has:
        batch["has_legal_mask"] = torch.tensor([1.0, 0.0, 1.0])
    logits = torch.randn(3, width)
    masked = apply_policy_mask_to_logits(logits, batch, "legal_mask", "has_legal_mask")
    # A move is in the softmax's support iff the masker left its logit alone.
    support = masked == logits
    legal = policy_legal_bool(batch, width=width)
    assert legal is not None
    assert torch.equal(legal, support)


def test_policy_legal_bool_is_none_without_a_mask() -> None:
    assert policy_legal_bool({"x": torch.zeros(2, 1)}, width=4) is None


# --------------------------------------------------------------------------
# INERTNESS, by execution: at the default weight `total` is bit-identical.
# --------------------------------------------------------------------------


def _tiny_batch(width: int = 32, rows: int = 8) -> tuple[dict, dict]:
    torch.manual_seed(20260817)
    legal = (torch.rand(rows, width) < 0.5).to(torch.float32)
    legal[:, 0] = 1.0
    target = torch.softmax(torch.randn(rows, width), dim=-1) * legal
    target = target / target.sum(-1, keepdim=True)
    outputs = {
        "policy": torch.randn(rows, width, requires_grad=True),
        "wdl": torch.randn(rows, 3),
    }
    batch = {
        "x": torch.zeros(rows, 175, 8, 8),
        "legal_mask": legal,
        "has_legal_mask": torch.ones(rows),
        "policy_t": target,
        "wdl_t": torch.randint(0, 3, (rows,)),
        "sf_p0_regret_t": torch.rand(rows, width),
        "has_sf_p0_regret": (torch.rand(rows) < 0.5).to(torch.float32),
    }
    return outputs, batch


def test_total_is_bit_identical_at_the_default_weight() -> None:
    """Not `approx`: `0.0 * x` is only zero for finite x, so the term is added
    to `total` under an `if` rather than multiplied by its weight. This asserts
    the consequence -- an exact float equality against the same call with the
    term's parameter absent altogether."""
    outputs, batch = _tiny_batch()
    without = compute_loss(outputs, batch)
    with_default = compute_loss(outputs, batch, sf_policy_floor=SfPolicyFloorParams())
    a = float(with_default["total"].detach().item())
    b = float(without["total"].detach().item())
    assert a == b
    assert a.hex() == b.hex()


def test_a_nan_in_the_term_cannot_reach_total_at_weight_zero() -> None:
    """The reason inertness is an `if` and not a `* 0.0`.

    A NaN reaching `total` through a weight that means "off" is the shape of
    "a clamp is not a validator": the guard reads healthy while the loss is NaN.
    The NaN is injected by defeating the validator with `object.__setattr__`
    (nothing a config can reach) precisely so the COMPOSITION rule is what is
    under test here, not the range check.
    """
    outputs, batch = _tiny_batch()
    poisoned = SfPolicyFloorParams()
    object.__setattr__(poisoned, "tau", float("nan"))
    losses = compute_loss(outputs, batch, sf_policy_floor=poisoned)
    assert math.isnan(float(losses["sf_policy_floor"].detach())), "setup: term NaN"
    assert not math.isnan(float(losses["total"].detach()))


def test_a_positive_weight_moves_total_and_the_columns_report_it() -> None:
    """The other half of inertness: the term is not inert when switched on.

    An inertness test alone passes for a term that is wired to nothing at all.
    """
    outputs, batch = _tiny_batch()
    off = compute_loss(
        outputs, batch,
        sf_policy_floor=SfPolicyFloorParams.resolve(w=0.0, tau=0.5, delta_cp=20.0),
    )
    on = compute_loss(
        outputs, batch,
        sf_policy_floor=SfPolicyFloorParams.resolve(w=1.0, tau=0.5, delta_cp=20.0),
    )
    assert float(on["total"].detach()) > float(off["total"].detach())
    assert float(on["sf_policy_floor_binds_sum"]) > 0.0
    assert float(on["sf_policy_floor_sum"].detach()) > 0.0
    # SAME SHAPE, ONLY THE WEIGHT DIFFERING: the diagnostic columns are live at
    # weight zero, which is what makes the binding rate readable BEFORE the
    # weight is ever raised.
    assert float(off["sf_policy_floor_binds_sum"]) == float(on["sf_policy_floor_binds_sum"])
    assert float(off["sf_policy_floor_sum"].detach()) == float(
        on["sf_policy_floor_sum"].detach()
    )


def test_the_floor_gradient_only_ever_pushes_floored_moves_up() -> None:
    """One-sided, on a fixture whose set F is known exactly.

    Case 2's row: ours is move 2 (30cp, outside the window), F = {0, 1}. Descent
    must RAISE both members' logits (negative gradient) and take the mass off
    our own wrong pick (positive gradient on move 2). A term that dragged mass
    off a floored move -- the failure the `better_than_ours` clause exists for --
    would show the opposite sign on 0 or 1.
    """
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    _, grad, _ = floor_loss(_REG, logits)
    assert float(grad[0, 0]) < 0.0
    assert float(grad[0, 1]) < 0.0
    assert float(grad[0, 2]) > 0.0
    # Non-members outside F are pushed down too -- the mass has to come from
    # somewhere -- but never a member.
    assert float(grad[0, 3]) > 0.0


# --------------------------------------------------------------------------
# M13: the collar's `has_policy` guard at the `compute_loss` call site.
# --------------------------------------------------------------------------


def test_a_row_without_a_policy_target_is_not_collared_through_compute_loss() -> None:
    """⚑ MUTANT M13. `aligned_pol_target * has_policy.unsqueeze(-1)` -> bare.

    `sf_policy_floor_deficit` decides "this row has a played move to protect"
    by asking whether the target it was handed carries any MASS. That is the
    right question only if the caller has already zeroed the rows whose policy
    target is not real. `has_policy` is 0 on exactly those rows -- most
    concretely the fast plies `selfplay.record_fast_ply_value` records, whose
    `policy_t` is written but is not a search distribution -- and dropping the
    multiply hands the deficit a target the row is not entitled to, so the
    collar floors a move that was never played and the term pays gradient for
    it.

    Unreachable in production TODAY only because `record_fast_ply_value` is OFF
    (CLAUDE.md: tried and REVERTED for trunk dilution). "Off in production" is
    not a guard, so the mutant is killed here rather than argued away.

    The row is built so the two answers cannot coincide: move 5 carries the
    whole (bogus) target and sits far below `tau_played`, while `has_policy`
    says the row has no policy target at all.
    """
    width = 6
    legal = torch.ones(1, width)
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, -6.0]])
    # `has_policy = 0`, but `policy_t` is non-empty -- the exact shape a fast
    # ply produces, and the shape a `.sum() > 0` test alone cannot tell from a
    # real search distribution.
    batch = {
        "x": torch.zeros(1, 175, 8, 8),
        "legal_mask": legal,
        "has_legal_mask": torch.ones(1),
        "policy_t": _one_hot(5, width),
        "has_policy": torch.zeros(1),
        "wdl_t": torch.zeros(1, dtype=torch.long),
        "sf_p0_regret_t": _REG.clone(),
        "has_sf_p0_regret": torch.ones(1),
    }
    outputs = {"policy": logits, "wdl": torch.zeros(1, 3)}
    params = SfPolicyFloorParams.resolve(w=1.0, tau=0.15, tau_played=0.5)

    p = torch.softmax(logits, dim=-1)[0]
    assert float(p[5]) < 0.5, "setup: the bogus played move must be under the collar"

    got = compute_loss(outputs, batch, sf_policy_floor=params)
    # The same row with the target genuinely absent -- what a correctly guarded
    # call site is equivalent to.
    honest = dict(batch)
    honest["policy_t"] = torch.zeros(1, width)
    expect = compute_loss(outputs, honest, sf_policy_floor=params)

    assert float(got["sf_policy_floor_sum"].detach()) == pytest.approx(
        float(expect["sf_policy_floor_sum"].detach()), abs=1e-6,
    )
    # ...and the mutant's answer is a DIFFERENT number, so the equality above is
    # not satisfied by both branches collapsing to zero.
    collared = compute_loss(
        outputs, {**batch, "has_policy": torch.ones(1)}, sf_policy_floor=params,
    )
    assert float(collared["sf_policy_floor_sum"].detach()) > float(
        got["sf_policy_floor_sum"].detach()
    ) + 1e-6


def test_the_collar_still_fires_on_a_row_that_does_have_a_policy_target() -> None:
    """The negative control for the test above.

    A guard test that passes because the collar never fires at all is vacuous.
    Same fixture, `has_policy = 1`, and the collar must ADD the deficit.
    """
    width = 6
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, -6.0]])
    batch = {
        "x": torch.zeros(1, 175, 8, 8),
        "legal_mask": torch.ones(1, width),
        "has_legal_mask": torch.ones(1),
        "policy_t": _one_hot(5, width),
        "has_policy": torch.ones(1),
        "wdl_t": torch.zeros(1, dtype=torch.long),
        "sf_p0_regret_t": _REG.clone(),
        "has_sf_p0_regret": torch.ones(1),
    }
    outputs = {"policy": logits, "wdl": torch.zeros(1, 3)}
    p = torch.softmax(logits, dim=-1)[0]

    on = compute_loss(
        outputs, batch,
        sf_policy_floor=SfPolicyFloorParams.resolve(w=1.0, tau=0.15, tau_played=0.5),
    )
    off = compute_loss(
        outputs, batch,
        sf_policy_floor=SfPolicyFloorParams.resolve(w=1.0, tau=0.15, tau_played=0.0),
    )

    assert float(on["sf_policy_floor_sum"].detach()) == pytest.approx(
        float(off["sf_policy_floor_sum"].detach()) + float(torch.relu(0.5 - p[5])),
        abs=1e-6,
    )


# --------------------------------------------------------------------------
# The reported inclusion claim must cover every threshold it claims about.
# --------------------------------------------------------------------------


def test_the_guard_fires_on_tau_top1_as_well() -> None:
    """⚑ `tau_top1` WAS THE THIRD BLIND SPOT, one level below `tau_played`'s.

    SF's top-1 gets `tau_top1` ALONE on the rows where our argmax already is
    that move -- `adaptive` is empty there, so the running max has nothing else
    to take. Those are the rows invariant 1 calls the whole mechanism, and a
    sub-guarantee `tau_top1` revokes their root slot in silence.
    """
    with pytest.warns(RuntimeWarning, match="sf_policy_floor_tau_top1"):
        SfPolicyFloorParams.resolve(w=1.0, tau=0.15, tau_top1=0.001, gumbel_topk=16)


def test_the_reported_inclusion_claim_covers_tau_top1(tmp_path, capsys) -> None:
    """⚑ THE BOOLEAN WAS LYING, NOT THE FLOOR.

    The startup line printed `guarantees_inclusion=True` beside
    `tau_top1=0.001`, because the min it took ran over `tau` and `tau_played`
    only. The floor itself is fine -- with `tau_top1 < tau` SF's top-1 joins
    the adaptive set on the rows where our pick is wrong and still gets
    `max(tau, tau_top1)`, and the measured deficit is identical at 0.15 and
    0.001 -- so what needed fixing was the claim.

    Asserted on the PRINTED LINE, which is the artefact an operator reads, not
    on a re-derivation of the same min in the test body.
    """
    with pytest.warns(RuntimeWarning, match="sf_policy_floor_tau_top1"):
        _trainer({"sf_policy_floor_tau_top1": 0.001}, tmp_path)
    line = [
        ln for ln in capsys.readouterr().out.splitlines()
        if ln.startswith("[trainer] sf_policy_floor w=")
    ]
    assert len(line) == 1
    assert "tau_top1=0.001" in line[0]
    assert "guarantees_inclusion=False" in line[0]


def test_the_reported_inclusion_claim_is_true_at_the_shipped_defaults(
    tmp_path, capsys,
) -> None:
    """The negative control. A claim that is always False is not a claim."""
    _trainer({}, tmp_path)
    line = [
        ln for ln in capsys.readouterr().out.splitlines()
        if ln.startswith("[trainer] sf_policy_floor w=")
    ]
    assert len(line) == 1
    assert "guarantees_inclusion=True" in line[0]


def test_the_inclusion_guarantee_holds_at_the_production_fast_ply_budget() -> None:
    """⚑ CORRECTING A FALSE REVIEW FINDING, and pinning why it is false.

    The claim under review was that `tau = 1/topk` stops guaranteeing inclusion
    at `fast_simulations: 8` and at `n_legal < 16`. Both are wrong:

    * the search keeps the top `m = max(2, min(topk, n_legal, max(2, (sims+1)//2)))`,
      so the sim budget can only bind when `(sims+1)//2 < topk`, i.e.
      `sims < 31` at `gumbel_topk: 16`. Production runs `mcts_simulations: 256`
      (ramped to 100, still >= 31) and `fast_simulations: 32`, which gives
      `m = 16` on fast plies exactly as on full ones;
    * when `n_legal < topk` the top-`m` IS every legal move, so inclusion is
      trivially guaranteed and tau is irrelevant. The narrow case is the safe
      one, and the reviewer's `1/m > tau` arithmetic is inverted for it.

    Read off the production yaml and the search's own selector rather than
    restated, so a config change to a binding budget fails here.
    """
    import yaml as _yaml

    from chess_anti_engine.mcts.gumbel import _select_top_m_with_gumbel

    cfg = _yaml.safe_load(
        Path("configs/pbt2_small.yaml").read_text(encoding="utf-8"),
    )
    topk = int(cfg["selfplay"]["gumbel_topk"])
    budgets = (int(cfg["selfplay"]["mcts_simulations"]), int(cfg["selfplay"]["fast_simulations"]))
    assert topk == 16
    assert budgets == (256, 32)

    import numpy as np

    rng = np.random.default_rng(0)
    for sims in (*budgets, 100):  # 100 = the progressive_mcts ramp floor
        for n_legal in (4, 16, 30):
            legal = np.arange(n_legal)
            pri = np.full(n_legal, 1.0 / n_legal)
            cands, _ = _select_top_m_with_gumbel(
                legal=legal, pri=pri, sim_budget=sims, topk=topk,
                add_noise=False, gumbel_scale=0.0, rng=rng,
            )
            # Every move at or above the guarantee is inside the candidate set:
            # either the whole legal set is kept (n_legal <= m) or m == topk.
            assert len(cands) == min(topk, n_legal), (sims, n_legal, len(cands))
    assert search_inclusion_guarantee_tau(topk) == pytest.approx(1.0 / topk)


def test_inclusion_under_production_noise_is_not_a_guarantee() -> None:
    """`1/topk` is a RANK threshold, not a probability of 1 under real noise.

    Codex review of PR #448, F2: the neighbouring inclusion test pins the
    candidate-set CARDINALITY with `add_noise=False, gumbel_scale=0.0`, which is
    true but cannot see the claim that matters. Production ranks
    `gumbel_scale * Gumbel + log(prior)` with `gumbel_scale: 1.0`, and Gumbel
    noise has unbounded support, so a move at exactly `p = 1/topk` can always be
    displaced.

    This test exists to FALSIFY the word "guarantee", so it asserts the drop is
    real rather than asserting it away. The peaked-tail column is included as
    the control: it is why the original 3000-row production measurement read
    1.0000 and why that reading was about the prior distribution, not the
    sampler.
    """
    import numpy as np

    from chess_anti_engine.mcts.gumbel import _select_top_m_with_gumbel

    topk = DEFAULT_GUMBEL_TOPK
    tau = search_inclusion_guarantee_tau(topk)

    def inclusion(n_legal: int, scale: float, *, peaked: bool) -> float:
        rng = np.random.default_rng(12345)
        hits = 0
        trials = 2000
        for _ in range(trials):
            pri = np.empty(n_legal, dtype=np.float64)
            pri[0] = tau
            if peaked:
                k = min(topk - 1, n_legal - 1)
                pri[1:1 + k] = tau - 1e-6
                rest = n_legal - 1 - k
                if rest > 0:
                    pri[1 + k:] = max(1e-9, 1.0 - tau - k * (tau - 1e-6)) / rest
            else:
                pri[1:] = (1.0 - tau) / (n_legal - 1)
            pri = pri / pri.sum()
            cands, _ = _select_top_m_with_gumbel(
                legal=np.arange(n_legal), pri=pri, sim_budget=256, topk=topk,
                add_noise=True, gumbel_scale=scale, rng=rng,
            )
            hits += int(0 in cands)
        return hits / trials

    # THE CLAIM UNDER TEST: with the tail spread evenly, the "guaranteed" move
    # is dropped a large fraction of the time at the production noise scale.
    flat_40 = inclusion(40, 1.0, peaked=False)
    assert flat_40 < 0.85, flat_40
    assert 0.60 < flat_40 < 0.85, flat_40

    # Monotone in the noise scale: halving it (production does, from move 12)
    # recovers most of the loss. A non-monotone result would mean the rig, not
    # the sampler, is producing the number.
    assert inclusion(40, 0.5, peaked=False) > flat_40

    # CONTROL: a peaked tail -- what real production priors look like -- keeps
    # every draw, which is the left column of the docstring table.
    assert inclusion(40, 1.0, peaked=True) == 1.0
