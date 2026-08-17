"""The fabricated-tail gate on the ``sf_own_regret`` loss term.

``sf_p0_regret`` is a CONSTANT-TAIL construction: up to ``sf_multipv`` real
normalized cp-regrets, then ONE fitted value repeated over every remaining legal
move. The term ``sum_m p_own(m) * regret(m)`` therefore puts IDENTICAL gradient on
every unsurfaced move -- it carries no information about their relative merit and
weights them by a number Stockfish never produced.

⚑⚑ THE MUTANT THESE TESTS EXIST TO KILL. ``selfplay/finalize.py`` says absent
moves "default to 1.0", so ``reg < 1.0`` reads as the obvious way to find the
surfaced set. It is WRONG on production data: the fill is the row's fitted
constant (measured 0.5259 on a live 28-legal-move row) and only 3.75% of live
legal entries are exactly 1.0, so ``reg < 1.0`` selects ~every legal move and the
gate would silently never fire. The surfaced set is ``reg < row_max``.
``test_the_surfaced_mask_is_not_the_reg_below_one_shortcut`` fails under that
mutant.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from chess_anti_engine.config_keys import TRAINER_WEIGHT_KEYS
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.train.losses import (
    compute_loss,
    retemper_main_policy_target,
    sf_regret_gate_scale,
    sf_regret_surfaced_mask,
)

# The fitted tail value measured on a live shard row. Deliberately NOT 1.0 and
# deliberately larger than every real regret below it -- that is the shape the
# production data has, verified as a plateau on 2350/2350 live rows.
ALPHA = 0.52587890625
REAL_REGRETS = (0.0, 0.012, 0.017, 0.033, 0.046, 0.052)
N_LEGAL = 28


def _constant_tail_row(*, n_legal: int = N_LEGAL, alpha: float = ALPHA) -> torch.Tensor:
    """One regret vector shaped exactly like production: 6 real values then alpha."""
    reg = torch.zeros((POLICY_SIZE,), dtype=torch.float32)
    for i, v in enumerate(REAL_REGRETS):
        reg[i] = v
    reg[len(REAL_REGRETS) : n_legal] = alpha
    return reg


def _legal(n_legal: int = N_LEGAL) -> torch.Tensor:
    m = torch.zeros((POLICY_SIZE,), dtype=torch.float32)
    m[:n_legal] = 1.0
    return m


def _onehot(idx: int) -> torch.Tensor:
    p = torch.zeros((POLICY_SIZE,), dtype=torch.float32)
    p[idx] = 1.0
    return p


def _peaked_tail_target(idx: int, *, n_legal: int = N_LEGAL) -> torch.Tensor:
    """A target peaked on tail move ``idx`` but NOT one-hot.

    ⚑ WHY THIS EXISTS. `retemper_main_policy_target` is a power transform, so it is
    EXACTLY the identity on a one-hot (0**x == 0, and the row max is scaled to 1.0).
    A one-hot fixture therefore makes `pol_target` bit-identical to `policy_t` and
    `test_the_gate_reads_the_STORED_target_not_the_retempered_one` cannot fail under
    its own mutant -- measured: the mutant survived with 17/17 green. Mass is placed
    so listed mass is 0.04 at temp 1.0 (below a 0.1 threshold) and ~0.23 once
    flattened at temp 4.0 (above it), which is what makes the two branches separable.
    """
    p = torch.zeros((POLICY_SIZE,), dtype=torch.float32)
    n_surfaced = len(REAL_REGRETS)
    p[:n_surfaced] = 0.04 / n_surfaced
    p[n_surfaced:n_legal] = 0.06 / (n_legal - n_surfaced - 1)
    p[idx] = 0.90
    return p


# ── the surfaced-set mask ────────────────────────────────────────────


def test_the_surfaced_mask_is_not_the_reg_below_one_shortcut() -> None:
    """⚑ THE CENTRAL REGRESSION. With alpha=0.5259 the `reg < 1.0` mutant marks
    every legal move surfaced, so the gate can never fire. The correct mask marks
    exactly the 6 real MultiPV entries."""
    reg = _constant_tail_row().unsqueeze(0)
    legal = _legal().unsqueeze(0)

    surfaced = sf_regret_surfaced_mask(reg, legal)
    assert int(surfaced.sum()) == len(REAL_REGRETS)
    assert surfaced[0, : len(REAL_REGRETS)].all()
    assert not surfaced[0, len(REAL_REGRETS) : N_LEGAL].any()

    # The mutant, spelled out, so the contrast is in the test rather than only in
    # the docstring: it would have marked all 28.
    mutant = (legal.bool() & (reg < 1.0))
    assert int(mutant.sum()) == N_LEGAL
    assert int(mutant.sum()) != int(surfaced.sum())


def test_an_illegal_entry_larger_than_alpha_cannot_set_the_row_max() -> None:
    """Mutant: drop the legality clamp from the `amax`. Absent indices densify to
    0.0 today, but a legacy/garbage row must not be able to redefine the tail."""
    reg = _constant_tail_row()
    reg[N_LEGAL + 3] = 9.0  # ILLEGAL index, larger than alpha
    surfaced = sf_regret_surfaced_mask(reg.unsqueeze(0), _legal().unsqueeze(0))
    assert int(surfaced.sum()) == len(REAL_REGRETS)


def test_a_row_with_no_legal_moves_surfaces_nothing_and_stays_finite() -> None:
    reg = _constant_tail_row().unsqueeze(0)
    surfaced = sf_regret_surfaced_mask(reg, torch.zeros((1, POLICY_SIZE)))
    assert int(surfaced.sum()) == 0
    scale, gated = sf_regret_gate_scale(
        reg, _onehot(0).unsqueeze(0), torch.zeros((1, POLICY_SIZE)),
        listed_mass_min=0.5, unlisted_scale=0.0,
    )
    assert torch.isfinite(scale).all()
    assert torch.isfinite(gated).all()


def test_a_row_whose_regrets_are_all_equal_surfaces_nothing() -> None:
    """Degenerate but reachable: if SF returned one line, every legal move shares
    the fill and there is no relative information at all. The mask must be empty
    rather than arbitrarily electing a 'best'."""
    reg = torch.full((1, POLICY_SIZE), ALPHA)
    assert int(sf_regret_surfaced_mask(reg, _legal().unsqueeze(0)).sum()) == 0


# ── the per-row gate scale ───────────────────────────────────────────


def test_the_gate_scale_is_all_ones_at_the_defaults() -> None:
    reg = _constant_tail_row().unsqueeze(0)
    scale, gated = sf_regret_gate_scale(
        reg, _onehot(20).unsqueeze(0), _legal().unsqueeze(0),
        listed_mass_min=0.0, unlisted_scale=1.0,
    )
    assert torch.equal(scale, torch.ones_like(scale))
    assert float(gated.sum()) == 0.0


@pytest.mark.parametrize(
    ("target_index", "expect_gated"),
    [(0, False), (3, False), (20, True), (N_LEGAL - 1, True)],
)
def test_the_gate_fires_exactly_on_targets_living_in_the_tail(
    target_index: int, expect_gated: bool,
) -> None:
    scale, gated = sf_regret_gate_scale(
        _constant_tail_row().unsqueeze(0),
        _onehot(target_index).unsqueeze(0),
        _legal().unsqueeze(0),
        listed_mass_min=0.5, unlisted_scale=0.25,
    )
    assert bool(gated[0] > 0.0) is expect_gated
    assert float(scale[0]) == pytest.approx(0.25 if expect_gated else 1.0)


def test_the_scale_interpolates_rather_than_hard_zeroing() -> None:
    """`unlisted_scale` is a DOWNWEIGHT, not a drop: within the |dQ|>=0.10 tail it
    is SF's move that is worse 30.2% of the time, so a hard zero throws away rows
    that are measurably better by a non-SF judge."""
    for s in (0.0, 0.25, 0.5, 1.0):
        scale, _ = sf_regret_gate_scale(
            _constant_tail_row().unsqueeze(0), _onehot(20).unsqueeze(0),
            _legal().unsqueeze(0), listed_mass_min=0.5, unlisted_scale=s,
        )
        assert float(scale[0]) == pytest.approx(s)


# ── end to end through compute_loss ──────────────────────────────────


def _outputs(b: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "policy_own": torch.randn((b, POLICY_SIZE)),
        "policy_soft": torch.randn((b, POLICY_SIZE)),
        "policy_sf": torch.randn((b, POLICY_SIZE)),
        "policy_future": torch.randn((b, POLICY_SIZE)),
        "wdl": torch.randn((b, 3)),
        "sf_eval": torch.randn((b, 3)),
        "categorical": torch.randn((b, 32)),
        "volatility": torch.rand((b, 3)),
        "sf_volatility": torch.rand((b, 3)),
        "moves_left": torch.rand((b, 1)),
    }


def _batch(
    target_indices: tuple[int, ...],
    *,
    targets: list[torch.Tensor] | None = None,
    has_sf_p0_regret: tuple[float, ...] | None = None,
) -> dict[str, torch.Tensor]:
    b = len(target_indices)
    rows = targets if targets is not None else [_onehot(i) for i in target_indices]
    elig = (
        torch.tensor(has_sf_p0_regret, dtype=torch.float32)
        if has_sf_p0_regret is not None
        else torch.ones((b,))
    )
    return {
        "x": torch.zeros((b, 146, 8, 8)),
        "policy_t": torch.stack(rows),
        "wdl_t": torch.zeros((b,), dtype=torch.long),
        "has_policy": torch.ones((b,)),
        "is_network_turn": torch.ones((b,)),
        "legal_mask": torch.stack([_legal() for _ in range(b)]),
        "has_legal_mask": torch.ones((b,)),
        "sf_p0_regret_t": torch.stack([_constant_tail_row() for _ in range(b)]),
        "has_sf_p0_regret": elig,
        "is_selfplay": torch.ones((b,)),
        "has_is_selfplay": torch.ones((b,)),
    }


def test_the_gate_is_a_BIT_EXACT_identity_at_the_defaults() -> None:
    """Not a tolerance. If arming the code path perturbs the loss at all, an
    unrelated regression could hide inside 'it changed a little'."""
    out, batch = _outputs(4), _batch((0, 3, 20, 27))
    base = compute_loss(out, batch, w_sf_own_regret=0.7)
    gated = compute_loss(
        out, batch, w_sf_own_regret=0.7,
        sf_own_regret_listed_mass_min=0.0, sf_own_regret_unlisted_scale=1.0,
    )
    assert set(base) == set(gated)
    for k in base:
        assert torch.equal(base[k], gated[k]), f"{k} moved at the identity defaults"


def test_arming_the_gate_lowers_the_term_and_the_total() -> None:
    out, batch = _outputs(4), _batch((0, 3, 20, 27))
    base = compute_loss(out, batch, w_sf_own_regret=0.7)
    armed = compute_loss(
        out, batch, w_sf_own_regret=0.7,
        sf_own_regret_listed_mass_min=0.5, sf_own_regret_unlisted_scale=0.0,
    )
    assert float(armed["sf_own_regret"]) < float(base["sf_own_regret"])
    assert float(armed["total"]) < float(base["total"])


def test_the_gate_cannot_move_the_total_when_its_weight_is_zero() -> None:
    """The live yaml runs `w_sf_own_regret: 0.0`. Arming the gate there must be a
    no-op on `total` -- otherwise the gate has found a second path into the loss."""
    out, batch = _outputs(4), _batch((0, 3, 20, 27))
    off = compute_loss(out, batch, w_sf_own_regret=0.0)
    armed = compute_loss(
        out, batch, w_sf_own_regret=0.0,
        sf_own_regret_listed_mass_min=0.5, sf_own_regret_unlisted_scale=0.0,
    )
    assert torch.equal(off["total"], armed["total"])


def test_the_gated_row_COUNT_is_reported_and_reads_zero_at_the_defaults() -> None:
    """⚑ The observation that proves the gate reached the production path. It is a
    COUNT, not a per-batch rate: `_RATIO_METRIC_FIELDS` divides summed numerator by
    summed denominator, and a mean of per-batch rates is the wrong estimator for
    the sf_p0 terms, whose eligible count swings batch to batch."""
    out, batch = _outputs(4), _batch((0, 3, 20, 27))
    base = compute_loss(out, batch, w_sf_own_regret=0.7)
    assert "sf_own_regret_gated_rows" in base
    assert float(base["sf_own_regret_gated_rows"]) == 0.0

    armed = compute_loss(
        out, batch, w_sf_own_regret=0.7,
        sf_own_regret_listed_mass_min=0.5, sf_own_regret_unlisted_scale=0.0,
    )
    # Two of the four targets (20, 27) live in the tail.
    assert float(armed["sf_own_regret_gated_rows"]) == 2.0


def test_the_gate_reads_the_STORED_target_not_the_retempered_one() -> None:
    """Mutant: gate on `pol_target`, which has already been through
    `retemper_main_policy_target`. The gate must describe the DATA, so a training
    knob must not be able to move which rows it selects. At temp 4.0 the target is
    flattened enough that a tail row's listed mass crosses the threshold -- the
    mutant would ungate it."""
    targets = [_peaked_tail_target(20), _peaked_tail_target(27)]
    out, batch = _outputs(2), _batch((20, 27), targets=targets)
  # The fixture must make the two branches SEPARABLE, or the test cannot fail
  # under its own mutant. Assert that directly rather than trusting it.
    pol_t = batch["policy_t"]
    surfaced = sf_regret_surfaced_mask(batch["sf_p0_regret_t"], batch["legal_mask"])
    cold_mass = float((pol_t * surfaced)[0].sum())
    hot_mass = float(
        (retemper_main_policy_target(pol_t, temp=4.0) * surfaced)[0].sum()
    )
    assert cold_mass < 0.1 < hot_mass, (
        f"fixture is degenerate: listed mass {cold_mass:.4f} -> {hot_mass:.4f} does "
        "not straddle the 0.1 threshold, so the mutant would survive"
    )

    cold = compute_loss(
        out, batch, policy_target_temp=1.0, w_sf_own_regret=0.7,
        sf_own_regret_listed_mass_min=0.1, sf_own_regret_unlisted_scale=0.0,
    )
    hot = compute_loss(
        out, batch, policy_target_temp=4.0, w_sf_own_regret=0.7,
        sf_own_regret_listed_mass_min=0.1, sf_own_regret_unlisted_scale=0.0,
    )
    assert float(cold["sf_own_regret_gated_rows"]) == 2.0
    assert float(hot["sf_own_regret_gated_rows"]) == 2.0, (
        "the retemper moved the gate -- it is reading pol_target, not policy_t"
    )


def test_the_gated_row_COUNT_excludes_rows_with_no_sf_p0_regret() -> None:
    """Mutant: drop the `sf_p0_regret_base` weighting from the count.

    ⚑ The numerator and denominator of `sf_own_regret_gated_frac` MUST share the
    eligibility mask. A row with no stored regret contributes 0 to the denominator
    `sf_own_regret_rows`, so counting it in the numerator makes the reported fraction
    exceed 1.0 -- the same impossible-coverage signature that exposed the P0
    alignment defect. Row 2 here is gate-eligible by its target but has
    `has_sf_p0_regret = 0`, so the count must read 2.0, not 3.0.
    """
    out = _outputs(3)
    batch = _batch((20, 27, 21), has_sf_p0_regret=(1.0, 1.0, 0.0))
    armed = compute_loss(
        out, batch, w_sf_own_regret=0.7,
        sf_own_regret_listed_mass_min=0.5, sf_own_regret_unlisted_scale=0.0,
    )
    assert float(armed["sf_own_regret_gated_rows"]) == 2.0
    assert float(armed["sf_own_regret_gated_rows"]) <= float(
        armed["sf_own_regret_rows"]
    ), "gated rows exceed the term's own eligible rows -- the fraction can exceed 1.0"


def test_the_metric_counts_rows_SCALED_not_rows_MATCHING_the_predicate() -> None:
    """⚑ The converse of the break the PR asked reviewers to attempt, and it was
    REAL: with `listed_mass_min: 0.5` and `unlisted_scale` left at its default 1.0,
    every returned tensor is bit-identical to ungated -- the gate provably did
    nothing -- while the metric read 0.5. A counter is not the mechanism behind it,
    so `gated` is the rows actually downweighted."""
    out, batch = _outputs(4), _batch((0, 3, 20, 27))
    base = compute_loss(out, batch, w_sf_own_regret=0.7)
    matched_but_unscaled = compute_loss(
        out, batch, w_sf_own_regret=0.7,
        sf_own_regret_listed_mass_min=0.5, sf_own_regret_unlisted_scale=1.0,
    )
    for k in base:
        assert torch.equal(base[k], matched_but_unscaled[k]), f"{k} moved"
    assert float(matched_but_unscaled["sf_own_regret_gated_rows"]) == 0.0, (
        "reported gated rows on a run where nothing was scaled"
    )


@pytest.mark.parametrize("n_legal", [1, 3, len(REAL_REGRETS)])
def test_a_row_with_NO_fabricated_tail_is_never_gated(n_legal: int) -> None:
    """⚑ The gate's justification is that the tail's magnitudes are INVENTED. Where
    SF surfaced every legal move there is no tail, nothing is invented, and scaling
    the term down would discard real supervision.

    These shapes were mis-scored by the first version, and the 2350-row plateau
    validation could not have caught them: it required >= 8 legal moves, so it
    excluded fully-covered and forced-move rows BY CONSTRUCTION. Measuring on a
    population that excludes the failure case is not evidence about it.
    """
    reg = torch.zeros((POLICY_SIZE,), dtype=torch.float32)
    for i in range(n_legal):
        reg[i] = REAL_REGRETS[i]
    legal = _legal(n_legal)
    # Target sits on the LAST legal move, i.e. the row max -- the shape that looks
    # like tail exposure and is not.
    target = _onehot(n_legal - 1).unsqueeze(0)
    scale, gated = sf_regret_gate_scale(
        reg.unsqueeze(0), target, legal.unsqueeze(0),
        listed_mass_min=0.5, unlisted_scale=0.0,
    )
    assert float(scale[0]) == 1.0, "gated a row with no fabricated tail"
    assert float(gated[0]) == 0.0


@pytest.mark.parametrize(
    ("regrets", "why"),
    [
        # The reviewer's exact reproduction: cps 10,5,0,-30,-30,-30 -> three real
        # regrets TIED at the worst. Plateau multiplicity 3, max 0.04, no cap.
        ((0.0, 0.005, 0.01, 0.04, 0.04, 0.04), "three real regrets tied at the max"),
        # Two legal moves, both equal -- the surfaced set is EMPTY, so listed_mass is
        # 0.0 and the row looks maximally tail-exposed while having no tail at all.
        ((0.0, 0.0), "every legal move equal, so the surfaced set is empty"),
        # ⚑ The residual the `>= 0.5` range test CANNOT catch, pinned so the docstring's
        # "~1 row in 2,931" claim is a tested statement and not a hope: a fully-covered
        # row whose tied real max is itself >= 0.5 DOES still gate. Asserted as the
        # known limit below, not as a pass.
        ((0.0, 0.2, 0.62, 0.62), "tied real max at/above the fill's floor -- KNOWN LIMIT"),
    ],
)
def test_TIED_REAL_REGRETS_on_a_fully_covered_row(
    regrets: tuple[float, ...], why: str,
) -> None:
    """⚑⚑ THE PLATEAU TEST ALONE DOES NOT EXCLUDE A FULLY-COVERED ROW, and an earlier
    revision's `⚑⚑ ... IS NEVER GATED` docstring was FALSE on **22.2% of the gate's own
    firings** because of it.

    Real regrets are integer cp / 1000, so two of them tying at the row max is routine
    -- and then multiplicity >= 2 fires on a row where SF surfaced EVERY legal move and
    nothing is invented. Gating ENRICHES for these rows, because `reg < row_max`
    excludes the tied max and drives `listed_mass` down.

    ⚑ My own guard test could not see this BY CONSTRUCTION: `REAL_REGRETS` are all
    DISTINCT and `n_legal` was only ever {1, 3, 6}. Same population-exclusion error as
    the >= 8-legal-move validation, one level down. That is why this case is
    parametrized over the shapes rather than over a count.

    The discriminator is arithmetic: the builder's fill is `(worst_surfaced + 1) / 2`
    with `worst_surfaced` in [0, 1], so **the fill is always >= 0.5** and a row whose max
    is below that provably carries none. Measured on 2,931 live plateau rows, that test
    rejects 149 of the ~150 non-filled rows.
    """
    n_legal = len(regrets)
    reg = torch.zeros((POLICY_SIZE,), dtype=torch.float32)
    for i, r in enumerate(regrets):
        reg[i] = r
    legal = _legal(n_legal)
    target = _onehot(n_legal - 1).unsqueeze(0)

    row_max = max(regrets)
    # Non-degeneracy: this row MUST look like a plateau, or the test proves nothing
    # about the discriminator that follows it.
    assert sum(1 for r in regrets if r == row_max) >= 2, "fixture is not a plateau"
    assert row_max < 1.0, "fixture hits the cap, which a different guard handles"

    scale, gated = sf_regret_gate_scale(
        reg.unsqueeze(0), target, legal.unsqueeze(0),
        listed_mass_min=0.5, unlisted_scale=0.0,
    )
    if row_max >= 0.5:
        # KNOWN LIMIT, asserted so it cannot regress silently into a claim of "never".
        assert float(scale[0]) == 0.0, (
            "the known limit changed -- if this row is now ungated the docstring's "
            "'~1 row in 2,931' residual is stale and must be re-measured"
        )
    else:
        assert float(scale[0]) == 1.0, f"gated a fully-covered row: {why}"
        assert float(gated[0]) == 0.0


def test_a_negative_unlisted_scale_cannot_INVERT_the_term() -> None:
    """⚑ Neither key is range-validated by `TrialConfig` -- CLAUDE.md category (c),
    so the schema accepts it, nothing range-checks it, and the consumer gets it raw.
    `unlisted_scale: -5` is a plausible typo and would make the optimizer MAXIMISE
    SF regret on exactly the rows the gate exists to protect: silent wrongness, the
    slowest failure of the three to notice. Clamped at the consumer."""
    reg = _constant_tail_row().unsqueeze(0)
    for bad in (-5.0, -1e-9, 7.0):
        scale, _ = sf_regret_gate_scale(
            reg, _onehot(20).unsqueeze(0), _legal().unsqueeze(0),
            listed_mass_min=0.5, unlisted_scale=bad,
        )
        assert 0.0 <= float(scale[0]) <= 1.0, f"unlisted_scale={bad} escaped the clamp"


def test_the_gate_does_not_fire_when_legality_is_absent() -> None:
    """⚑ `legal_mask` is OPTIONAL in `compute_loss` — every other consumer reaches
    it through `_get_mask`/`apply_policy_mask_to_logits`. Subscripting it raised
    `KeyError` on 5 tests in `test_sf_p0_teacher_metrics.py` and 3 in
    `test_phase_loss_buckets.py`. Without legality the surfaced set is not
    identifiable, so the gate must SKIP rather than guess.

    ⚑ `ones_like` is the WRONG fallback: it marks all 1858 padding slots legal, so
    the row max becomes the padding fill and the gate silently never fires on any
    row — a knob accepted and then ignored, this repo's signature defect.

    ⚑ `policy_t` is deliberately NOT tested here: `losses.py:942` already
    subscripts it on the BASE branch to build `pol_target`, so it is required
    upstream of this gate and claiming otherwise would be a false guarantee. The
    `.get` on it is defensive only.
    """
    out = _outputs(2)
    batch = _batch((20, 27))
    del batch["legal_mask"]
    armed = compute_loss(
        out, batch, w_sf_own_regret=0.7,
        sf_own_regret_listed_mass_min=0.5, sf_own_regret_unlisted_scale=0.0,
    )
    assert torch.isfinite(armed["total"])
    assert float(armed["sf_own_regret_gated_rows"]) == 0.0


def test_listed_mass_is_normalized_over_the_LEGAL_support() -> None:
    """`policy_t` is stored fp16 and is not guaranteed to sum to 1.0 after
    alignment, so an unnormalised sum makes `listed_mass_min` mean subtly different
    things on different rows.

    ⚑ The fixture must straddle the threshold under exactly one of the two
    readings, or the test cannot fail under its own mutant -- my first attempt
    scaled a full distribution, which keeps both readings on the SAME side and let
    the mutant survive with 76/76 green. Here total mass is 0.20 with 0.03 on the
    surfaced set: normalised that is 0.15 (ABOVE a 0.1 threshold, so not gated),
    unnormalised it is 0.03 (BELOW it, so gated).
    """
    reg = _constant_tail_row().unsqueeze(0)
    legal = _legal().unsqueeze(0)
    n_surfaced = len(REAL_REGRETS)
    target = torch.zeros((POLICY_SIZE,), dtype=torch.float32)
    target[:n_surfaced] = 0.03 / n_surfaced
    target[n_surfaced:N_LEGAL] = 0.17 / (N_LEGAL - n_surfaced)
    target = target.unsqueeze(0)

    assert float(target.sum()) == pytest.approx(0.20)
    assert float(target[0, :n_surfaced].sum()) == pytest.approx(0.03)

    scale, gated = sf_regret_gate_scale(
        reg, target, legal, listed_mass_min=0.1, unlisted_scale=0.0)
    assert float(scale[0]) == 1.0, (
        "gated a row whose NORMALIZED listed mass (0.15) is above the threshold -- "
        "listed_mass is being read as a raw sum"
    )
    assert float(gated[0]) == 0.0


def test_a_row_whose_max_sits_AT_THE_CAP_is_never_gated() -> None:
    """⚑ Normalized regret is capped at 1.0, so a REAL regret that hit the cap is
    numerically INDISTINGUISHABLE from the fill, and `reg < row_max` then counts
    those capped real moves as tail. An independent review MEASURED the cost: a
    cap-plateau row discarded 0.1771 of gradient, **2x the gate's intended-target
    row (0.0880)**. Ties at the max are STRUCTURAL here, not rare -- the earlier
    docstring called them rare and that was wrong."""
    reg = torch.zeros((POLICY_SIZE,), dtype=torch.float32)
    reg[0], reg[1] = 0.0, 0.2
    reg[2:N_LEGAL] = 1.0          # two capped REAL moves + the fill, all at 1.0
    scale, gated = sf_regret_gate_scale(
        reg.unsqueeze(0), _onehot(3).unsqueeze(0), _legal().unsqueeze(0),
        listed_mass_min=0.5, unlisted_scale=0.0,
    )
    assert float(scale[0]) == 1.0, "gated a row whose max is the cap"
    assert float(gated[0]) == 0.0


def test_the_plateau_boundary_is_TWO_not_three() -> None:
    """Kills the boundary mutant `>= 2` -> `>= 3`, which the previous reviewer's
    own mutation run found SURVIVING. At `sf_multipv: 6` an 8-legal row has a
    two-move tail, so a 3-multiplicity bar would silently stop gating it."""
    reg = torch.zeros((POLICY_SIZE,), dtype=torch.float32)
    for i, v in enumerate(REAL_REGRETS):
        reg[i] = v
    reg[6], reg[7] = ALPHA, ALPHA   # exactly TWO tail moves
    scale, gated = sf_regret_gate_scale(
        reg.unsqueeze(0), _onehot(6).unsqueeze(0), _legal(8).unsqueeze(0),
        listed_mass_min=0.5, unlisted_scale=0.0,
    )
    assert float(scale[0]) == 0.0, "a two-move tail must still be gated"
    assert float(gated[0]) == 1.0


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 10.0, 1e9])
def test_a_NON_FINITE_or_out_of_range_key_cannot_poison_the_loss(bad: float) -> None:
    """⚑⚑ NaN is handled EXPLICITLY, not by the clamp: Python's `min`/`max`
    PROPAGATE NaN (every comparison with NaN is False, so the first argument wins).
    A NaN scale took `total` to NaN **even at `w_sf_own_regret: 0.0`** while
    `sf_own_regret_gated_frac` read 0.0 -- production dies and the instrument says
    nothing happened. Both keys are category (c): schema-accepted, unvalidated,
    delivered raw."""
  # ⚑ Inlined rather than splatted from a dict: pyright widens a mixed dict to
  # `dict[str, float]` and then reports every unrelated `compute_loss` parameter as
  # a type error. Cost me a red lint run once already.
    out, batch = _outputs(4), _batch((0, 3, 20, 27))
    for weight in (0.0, 0.7):
        bad_min = compute_loss(
            out, batch, w_sf_own_regret=weight,
            sf_own_regret_listed_mass_min=bad, sf_own_regret_unlisted_scale=0.0,
        )
        bad_scale = compute_loss(
            out, batch, w_sf_own_regret=weight,
            sf_own_regret_listed_mass_min=0.5, sf_own_regret_unlisted_scale=bad,
        )
        for got, which in ((bad_min, "listed_mass_min"), (bad_scale, "unlisted_scale")):
            assert torch.isfinite(got["total"]), f"total NaN: {which}={bad} w={weight}"
            assert torch.isfinite(got["sf_own_regret"]), f"{which}={bad} w={weight}"
            assert torch.isfinite(got["sf_own_regret_gated_rows"])


def test_an_out_of_range_listed_mass_min_cannot_gate_EVERY_row() -> None:
    """⚑ The finiteness test above cannot see this one, and that is the point of
    having both: an unclamped `listed_mass_min` is not a NaN hazard, it is a
    BEHAVIOURAL one. `listed_mass` is a fraction in [0, 1], so `listed_mass_min: 10`
    makes `mass < 10` true on EVERY row -- the gate stops selecting the fabricated
    tail and downweights the whole term, including rows whose target sits entirely
    inside SF's surfaced set. Clamping to <= 1.0 with a strict `<` makes a
    fully-listed row (mass exactly 1.0) unreachable.

    Verified by mutation: deleting the clamp left the finiteness test green.
    """
    reg = _constant_tail_row().unsqueeze(0)
    legal = _legal().unsqueeze(0)
    fully_listed = _onehot(0).unsqueeze(0)   # all mass on a SURFACED move
    for absurd in (10.0, 1e9, 2.0):
        scale, gated = sf_regret_gate_scale(
            reg, fully_listed, legal,
            listed_mass_min=absurd, unlisted_scale=0.0,
        )
        assert float(scale[0]) == 1.0, (
            f"listed_mass_min={absurd} gated a fully-listed row -- the clamp is gone"
        )
        assert float(gated[0]) == 0.0


def test_the_gate_is_PINNED_OFF_on_the_eval_path() -> None:
    """⚑⚑ THE RULER MUST NOT MOVE WITH THE ARM. The gate LOOKS like a weight
    (`sf_own_regret * scale`) but applies PER ROW on a data-dependent predicate, so
    it changes WHICH rows the column is measured over -- it REDEFINES the column
    rather than scaling it. A reviewer measured an unchanged model reading
    `sf_own_regret` 0.4174 -> 0.2112, a 2x move from a training knob. Unpinned,
    arming the arm would read as the eval loss improving with zero model change.

    This is also why `eval_ruler_id`'s source-digest blind spot on
    `sf_regret_gate_scale` does not matter: the helper cannot reach the eval
    measurement at all.
    """
    from chess_anti_engine.train.trainer import Trainer

    getter = Trainer._eval_loss_kwargs.fget
    assert getter is not None
    src = inspect.getsource(getter)
    for key, off in (
        ("sf_own_regret_listed_mass_min", "0.0"),
        ("sf_own_regret_unlisted_scale", "1.0"),
    ):
        assert f'"{key}": {off}' in src, f"{key} is not pinned off in _eval_loss_kwargs"


def test_both_keys_reach_the_TRAINERS_OWN_loss_kwargs() -> None:
    """⚑⚑ THE MUTANT THAT SURVIVED THE PREVIOUS REVIEW AT 295 TESTS GREEN: deleting
    both keys from `Trainer._loss_kwargs` broke nothing, because nothing in this PR
    constructed a `Trainer` or called `train_steps`. Every other test here calls
    `compute_loss` DIRECTLY, so the whole config -> trainer -> loss leg was
    unverified -- a knob accepted and then never delivered, this repo's signature
    defect, sitting inside the change meant to guard against it.

    Reading the property's source rather than building a Trainer keeps this a unit
    test (a real `Trainer` needs a model, an optimizer and a device), and it is the
    argument-level check the `_loss_kwargs` -> `compute_loss` seam actually needs.
    """
    from chess_anti_engine.train.trainer import Trainer

    getter = Trainer._loss_kwargs.fget
    assert getter is not None
    src = inspect.getsource(getter)
    for key in GATE_KEYS:
        assert f'"{key}"' in src, f"{key} never reaches Trainer._loss_kwargs"
        assert f"self.{key}" in src, f"{key} is not read from the trainer's own state"


@pytest.mark.parametrize(
    ("mass_min", "scale"), [(0.8, 0.25), (0.5, 0.0), (1.0, 0.9)],
)
def test_a_YAML_VALUE_SURVIVES_THE_WHOLE_CONFIG_TO_TRAINER_KWARGS_SEAM(
    mass_min: float, scale: float,
) -> None:
    """⚑⚑ THE MUTANT THAT SURVIVED THE THIRD REVIEW AT 446 TESTS GREEN: deleting both
    keys from `trainer_kwargs_from_config` broke nothing.

    The test above covers `Trainer._loss_kwargs -> compute_loss`. It does NOT cover
    `config -> trainer_kwargs_from_config -> Trainer`, which is one leg further
    upstream and is where an operator's yaml value actually enters. With those two
    lines absent the failure is fully silent: the schema still accepts the key,
    `flatten_run_config_defaults` still carries it, `params.json` and the
    realized-config report still echo the operator's value -- and the `Trainer` is
    built with the CODE DEFAULT. A day-plus paired arena would run with the gate OFF
    while every artifact says it was on, and `sf_own_regret_gated_frac` reading 0.0
    would be indistinguishable from "armed but nothing matched".

    ⚑ BEHAVIOURAL ON PURPOSE. A source-text assertion (`inspect.getsource`) would not
    have killed that mutant either -- it would only have moved the vacuity one level.
    This drives the real function with a real flattened config and reads the value out.
    """
    from chess_anti_engine.train.trainer import trainer_kwargs_from_config
    from chess_anti_engine.utils import flatten_run_config_defaults

    cfg = flatten_run_config_defaults({
        "train": {
            "sf_own_regret_listed_mass_min": mass_min,
            "sf_own_regret_unlisted_scale": scale,
        },
    })
    kwargs = trainer_kwargs_from_config(cfg)
    assert kwargs["sf_own_regret_listed_mass_min"] == pytest.approx(mass_min)
    assert kwargs["sf_own_regret_unlisted_scale"] == pytest.approx(scale)
    # ⚑ And the DEFAULT must be the identity, not merely "some float": an omitted key
    # has to leave the gate off, or merging arms it for everyone.
    bare = trainer_kwargs_from_config(flatten_run_config_defaults({}))
    assert bare["sf_own_regret_listed_mass_min"] == 0.0
    assert bare["sf_own_regret_unlisted_scale"] == 1.0


@pytest.mark.parametrize("key", [
    "sf_own_regret_listed_mass_min", "sf_own_regret_unlisted_scale",
])
def test_the_startup_only_KEYS_ARE_READ_LITERALLY_SO_THE_DERIVER_CAN_SEE_THEM(
    key: str,
) -> None:
    """⚑⚑ Read via the `_f(...)` helper these keys are INVISIBLE to two instruments,
    and the comment forbidding that sat two lines above where they were added.

    `tests/test_startup_only_config_keys.py` derives the startup-only set from LITERAL
    config reads; `tests/test_construction_only_config_keys.py` matches only
    `.get("k"`, `["k"]`, `tc.k`. Under `_f` the first went RED (the key is in neither
    read-set, so it reported a declared-startup-only key with a live consumer) and the
    second went GREEN **VACUOUSLY** -- a reviewer's mutant repointing the declaration at
    `worker.py`, which never reads the key, still passed. One spelling closed both.

    Pinned here so a future refactor to `_f` for tidiness fails loudly rather than
    quietly hollowing out a guard.
    """
    import inspect

    from chess_anti_engine.train.trainer import trainer_kwargs_from_config

    src = inspect.getsource(trainer_kwargs_from_config)
    assert f'config.get("{key}"' in src, (
        f"{key} must be read as a literal config.get -- via _f(...) the startup-only "
        "deriver cannot see it and the construction-only guard passes vacuously"
    )
    assert f'_f("{key}"' not in src, f"{key} is read through the _f helper again"


# ── plumbing: the knobs must actually be reachable ───────────────────


GATE_KEYS = ("sf_own_regret_listed_mass_min", "sf_own_regret_unlisted_scale")


def test_both_keys_are_schema_accepted_but_STARTUP_ONLY() -> None:
    """Schema acceptance and live-pushability are DIFFERENT properties, and these
    keys need the first without the second.

    Schema acceptance is mandatory: `flatten_run_config_defaults` runs outside any
    `try` in `run.py`, so a live-yaml key the running schema does not know prevents
    the process from BOOTING -- it is not a silent revert.

    Live-pushability is the part they must NOT have. `TRAINER_WEIGHT_KEYS` is the
    every-iteration live-push set AND the salvage-donor overlay, and by that
    tuple's own documented criterion it is for loss WEIGHTS. These select WHICH
    ROWS the term applies to, so a mid-run edit re-shapes the term's population
    with no restart boundary to mark where a readout's data changed -- the same
    reason `policy_target_temp` is excluded. The arm's readout is a day-plus
    paired arena, i.e. exactly the window that must not move under it.
    """
    from chess_anti_engine.tune.trainable_config_ops import _STARTUP_ONLY_TRIAL_KEYS
    from chess_anti_engine.utils.config_yaml import _TRAIN_KEYS

    for key in GATE_KEYS:
        assert key in _TRAIN_KEYS, f"{key} would be fatal at launch"
        assert key not in TRAINER_WEIGHT_KEYS, f"{key} must not be live-pushable"
        assert key in _STARTUP_ONLY_TRIAL_KEYS, f"{key} must warn on a mid-run edit"


def test_the_gated_frac_reaches_the_RAY_RESULT_ROW_not_only_tensorboard() -> None:
    """⚑ The metric is the ONLY observation that proves the gate fired, so it has to
    land in the Ray result row. `_train_metrics_dict` enumerates columns BY NAME
    (there is no `asdict` pass-through) and TensorBoard event files rotate per Ray
    session -- a field added to `TrainMetrics` and not here is computed every step
    and read by nobody. That is this repo's signature defect applied to the very
    instrument meant to rule it out."""
    import dataclasses

    from chess_anti_engine.train.trainer import TrainMetrics
    from chess_anti_engine.tune.trainable_report import (
        _TRAIN_METRIC_DEFAULTS,
        _train_metrics_dict,
    )

    names = {f.name for f in dataclasses.fields(TrainMetrics)}
    assert "sf_own_regret_gated_frac" in names, "a ratio entry with no field CRASHES"
    assert "sf_own_regret_gated_frac" in _TRAIN_METRIC_DEFAULTS
    metrics = TrainMetrics(
        loss=0.0, policy_loss=0.0, soft_policy_loss=0.0, future_policy_loss=0.0,
        wdl_loss=0.0, sf_move_loss=0.0, sf_move_acc=0.0, sf_eval_loss=0.0,
        categorical_loss=0.0, volatility_loss=0.0, sf_volatility_loss=0.0,
        moves_left_loss=0.0, sf_own_regret_gated_frac=0.25,
    )
    row = _train_metrics_dict(metrics)
    # ⚑ Not a membership check: a column wired to a literal 0.0 passes any
    # `in` test and reads as a healthy run. Assert the VALUE arrives.
    assert row["sf_own_regret_gated_frac"] == 0.25


def test_every_ratio_metric_has_a_TrainMetrics_field() -> None:
    """The general form of the crash above. `_ratio_metric_kwargs` emits its keys
    UNFILTERED, unlike the loss-key loop which filters on `_TRAIN_METRICS_FIELDS`,
    so `_build_metrics` splats them straight into `TrainMetrics(**...)`. One entry
    without a field raises `TypeError` on iteration 1 inside a `try:` that has a
    `finally:` and zero `except` -- the trial dies mid-iteration. Guard the whole
    table, not just the key this PR added."""
    import dataclasses

    from chess_anti_engine.train.trainer import _RATIO_METRIC_FIELDS, TrainMetrics

    names = {f.name for f in dataclasses.fields(TrainMetrics)}
    missing = sorted(set(_RATIO_METRIC_FIELDS) - names)
    assert not missing, f"ratio metrics with no TrainMetrics field (will crash): {missing}"


def test_the_ratio_table_registers_the_gated_frac_against_the_terms_own_rows() -> None:
    from chess_anti_engine.train.trainer import _RATIO_METRIC_FIELDS

    assert _RATIO_METRIC_FIELDS["sf_own_regret_gated_frac"] == (
        "sf_own_regret_gated_rows", "sf_own_regret_rows",
    )
    # Same denominator as the term itself, so numerator and denominator cannot
    # disagree about how many rows were eligible.
    assert (
        _RATIO_METRIC_FIELDS["sf_own_regret_gated_frac"][1]
        == _RATIO_METRIC_FIELDS["m_sf_own_regret"][1]
    )
