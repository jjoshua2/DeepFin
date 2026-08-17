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


# ── plumbing: the knobs must actually be reachable ───────────────────


def test_both_keys_are_live_pushable_and_therefore_schema_accepted() -> None:
    """`TRAINER_WEIGHT_KEYS` is BOTH the every-iteration live-push set and the YAML
    allowlist, so membership is what makes a live `sf_own_regret_*` key legal. A key
    the schema rejects is FATAL at launch, not a silent revert."""
    assert "sf_own_regret_listed_mass_min" in TRAINER_WEIGHT_KEYS
    assert "sf_own_regret_unlisted_scale" in TRAINER_WEIGHT_KEYS


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
