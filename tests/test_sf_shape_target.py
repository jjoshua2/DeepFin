"""The SF-shape conditional-KL term: its SET, its gradient, its wiring, its inertness.

Three things are under test here and they fail for different reasons:

1. **THE SET.** ``sf_surfaced_move_mask`` recovers the moves Stockfish actually
   scored from ``sf_p0_regret`` alone, because no surfaced-mask field reaches the
   training batch (see the function's docstring). At ``sf_multipv: 6`` about 79%
   of the move list carries a FABRICATED ``default_regret``, so a recovery that
   admits one invented entry trains the net on invented data across most of the
   move list. The cases below are written against the WRITER
   (`selfplay/finalize._build_sf_p0_regret_vector`), not against a hand-copied
   fill rule, so a change to the writer breaks them.

2. **THE GRADIENT.** The term is a KL between two CONDITIONAL distributions over
   that set, which makes "SF asserts nothing about the moves it never scored" a
   property of the arithmetic: no logit outside the set gets any gradient, and
   the gradients inside it sum to zero. ``test_the_gradient_cannot_move_mass_in_
   or_out_of_the_surfaced_set`` pins exactly that, and the mutant that kills it
   is the full-length cross-entropy formulation this term was nearly built as.

3. **THE WIRING.** config -> Trainer -> ``_loss_kwargs`` -> ``compute_loss``,
   asserted against a LITERAL expected object. Never against a second call to
   ``SfShapeParams.resolve``: a re-derivation in the test body agrees with a
   trainer that ignored the config entirely.

⚑ Every guard here was mutation-tested; the mutant table is in the PR.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from chess_anti_engine.config_keys import TRAINER_WEIGHT_KEYS
from chess_anti_engine.selfplay import finalize
from chess_anti_engine.train.constants import SF_OWN_REGRET_CAP_CP
from chess_anti_engine.train.losses import (
    SF_SHAPE_TEMP_CP_DEFAULT,
    SfShapeParams,
    compute_loss,
    matched_support_entropy_stats,
    row_entropy,
    sf_shape_conditional_kl,
    sf_surfaced_move_mask,
)

CAP = SF_OWN_REGRET_CAP_CP


def R(cp: float) -> float:
    """A cp regret in the units `sf_p0_regret` is stored in."""
    return cp / CAP


# --------------------------------------------------------------------------
# 1. THE SET: what `sf_surfaced_move_mask` can and cannot admit.
# --------------------------------------------------------------------------


def _writer_row(scored: list[tuple[int, float]], width: int) -> torch.Tensor:
    """One `sf_p0_regret` row built by the PRODUCTION WRITER.

    `scored` is (move index, SF score in cp, higher = better). Going through
    `_build_sf_p0_regret_vector` rather than filling the fabricated value by hand
    is the point: the recovery under test is a claim ABOUT THE WRITER, and a
    hand-written fill would keep passing after the writer changed its rule.
    """
    rows = np.array(
        [[idx, int(cp), 0, 0, 0] for idx, cp in scored], dtype=np.int32,
    )
    full = finalize._build_sf_p0_regret_vector(rows, policy_encoding="full")
    assert full is not None
    return torch.from_numpy(np.asarray(full)[:width]).to(torch.float32).unsqueeze(0)


def test_the_recovered_set_is_exactly_what_the_writer_scored() -> None:
    """Three surfaced moves out of six, recovered through the real writer."""
    reg = _writer_row([(0, 0.0), (1, -10.0), (2, -50.0)], width=6)
    legal = torch.ones(1, 6, dtype=torch.bool)
    surfaced, count = sf_surfaced_move_mask(reg, legal)
    assert surfaced.tolist() == [[True, True, True, False, False, False]]
    assert float(count) == 3.0


def test_a_fabricated_entry_can_never_enter_the_set() -> None:
    """⚑ THE CRUX. 79% of the move list carries `(worst_surfaced + 1) / 2`.

    Asserted as a PROPERTY over the writer's whole output, not on one fixture
    row: every index the writer did not cover must be excluded, at every
    coverage width. A recovery keyed on a hardcoded constant (0.5, or a
    `> 0.5` threshold) passes the fixture above and fails here at width 1,
    where the fabricated value is 0.5 exactly.
    """
    for k in (1, 2, 3, 5):
        scored = [(i, -100.0 * i) for i in range(k)]
        reg = _writer_row(scored, width=8)
        surfaced, count = sf_surfaced_move_mask(reg, torch.ones(1, 8, dtype=torch.bool))
        assert float(count) == float(k), f"k={k}"
        assert surfaced[0, :k].all(), f"k={k}"
        assert not surfaced[0, k:].any(), f"k={k}"


def test_the_fill_ties_the_worst_move_only_when_it_hit_the_cap() -> None:
    """The ONE case the recovery is lossy in, stated as a test rather than hoped.

    `d == worst` iff `worst == 1.0`, i.e. SF surfaced a move at or beyond the
    1000 cp cap. Then the capped move ties the fill and drops out. The error is
    one-sided -- the set is a SUBSET of the truth, never a superset -- which is
    the property that matters, and it is what the assertion pins.
    """
    reg = _writer_row([(0, 0.0), (1, -20.0), (2, -5000.0)], width=6)
    assert float(reg[0, 2]) == 1.0, "setup: move 2 is at the cap"
    surfaced, count = sf_surfaced_move_mask(reg, torch.ones(1, 6, dtype=torch.bool))
    # The capped move is dropped, and nothing invented is admitted in its place.
    assert surfaced.tolist() == [[True, True, False, False, False, False]]
    assert float(count) == 2.0


def test_illegal_moves_are_excluded_from_the_set() -> None:
    """The fill covers ILLEGAL indices too; only the legal mask separates them."""
    reg = _writer_row([(0, 0.0), (1, -10.0), (2, -50.0)], width=6)
    legal = torch.tensor([[True, False, True, True, True, True]])
    surfaced, count = sf_surfaced_move_mask(reg, legal)
    assert surfaced.tolist() == [[True, False, True, False, False, False]]
    assert float(count) == 2.0


def test_a_fully_covered_row_keeps_its_worst_legal_move_in_the_set() -> None:
    """⚑ WHY THE FILL IS THE FULL-ROW MAX AND NOT THE MAX OVER LEGAL ENTRIES.

    In an endgame every legal move can be surfaced. A legal-only max would then
    be `worst_surfaced` and would drop SF's worst legal move -- on exactly the
    rows where coverage is PERFECT, which is the last place a recovery should get
    less accurate. The writer's fill covers illegal indices too, so the full-row
    max is still the fabricated value there.
    """
    reg = _writer_row([(0, 0.0), (1, -10.0), (2, -50.0)], width=6)
    legal = torch.tensor([[True, True, True, False, False, False]])
    surfaced, count = sf_surfaced_move_mask(reg, legal)
    assert float(count) == 3.0
    assert surfaced.tolist() == [[True, True, True, False, False, False]]


def test_an_absent_legal_mask_leaves_the_regret_rule_alone() -> None:
    """`legal=None` is the batch-has-no-mask case (`policy_legal_bool`)."""
    reg = _writer_row([(0, 0.0), (1, -10.0)], width=6)
    surfaced, count = sf_surfaced_move_mask(reg, None)
    assert float(count) == 2.0
    assert surfaced.tolist() == [[True, True, False, False, False, False]]


def test_a_nan_regret_row_yields_an_empty_set_rather_than_a_nan() -> None:
    reg = torch.tensor([[0.0, float("nan"), 0.5, 0.5]])
    surfaced, count = sf_surfaced_move_mask(reg, None)
    assert float(count) == 0.0
    assert not surfaced.any()


# --------------------------------------------------------------------------
# 2. THE GRADIENT: the conditioning, as a property of the arithmetic.
# --------------------------------------------------------------------------


def _kl_and_grad(
    logits: torch.Tensor,
    reg: torch.Tensor,
    legal: torch.Tensor | None = None,
    *,
    temp_cp: float = SF_SHAPE_TEMP_CP_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = logits.clone().requires_grad_(True)
    probs = torch.softmax(z if legal is None else z.masked_fill(~legal, -1e9), dim=-1)
    out = sf_shape_conditional_kl(
        z, probs, reg, legal, params=SfShapeParams(temp_cp=temp_cp),
    )
    (grad,) = torch.autograd.grad(out.kl.sum(), [z])
    return out.kl.detach(), grad


def test_the_gradient_cannot_move_mass_in_or_out_of_the_surfaced_set() -> None:
    """⚑⚑ THE PROPERTY THE WHOLE FORMULATION EXISTS FOR.

    Both sides of the KL are conditional on S, so:

      * every logit OUTSIDE S receives EXACTLY zero gradient (not "small"), and
      * the gradients INSIDE S sum to zero, so descent cannot raise or lower the
        total probability mass on S -- it can only redistribute within it.

    ⚑ MUTANT M1 (the formulation originally specified) KILLS THIS TEST. Replace
    the conditional KL with a full-length soft cross-entropy against a
    mass-preserving target -- target = M * q_SF on S, base off S, then
    `soft_cross_entropy(masked_logits, target)` -- and the off-S gradients become
    non-zero and the on-S sum stops being zero, because the full-length softmax
    couples every logit to every other. The two formulations are NOT equivalent
    and this is the assertion that says so.
    """
    reg = _writer_row([(0, 0.0), (1, -10.0), (2, -50.0)], width=6)
    legal = torch.ones(1, 6, dtype=torch.bool)
    logits = torch.tensor([[0.0, 2.0, -1.0, 0.5, 0.3, -2.0]])
    kl, grad = _kl_and_grad(logits, reg, legal)

    assert float(kl) > 0.0, "setup: the term must be doing something on this row"
    # Outside S: exactly zero, asserted as an exact float equality.
    assert grad[0, 3:].abs().max().item() == 0.0
    # Inside S: sums to zero, i.e. no net push on the mass of S.
    assert float(grad[0, :3].sum()) == pytest.approx(0.0, abs=1e-6)
    # And it is not zero move-by-move -- otherwise the sum above is trivial.
    assert grad[0, :3].abs().max().item() > 1e-3


def test_the_gradient_pushes_toward_stockfishs_ordering_inside_the_set() -> None:
    """Direction, on a row where our ranking inside S is exactly backwards.

    SF's best is move 0; we put our mass on move 2, the worst of the three.
    Descent must RAISE move 0 (negative gradient) and LOWER move 2.
    """
    reg = _writer_row([(0, 0.0), (1, -10.0), (2, -300.0)], width=6)
    legal = torch.ones(1, 6, dtype=torch.bool)
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    _, grad = _kl_and_grad(logits, reg, legal)
    assert float(grad[0, 0]) < 0.0
    assert float(grad[0, 2]) > 0.0


def test_the_pull_on_sfs_best_move_does_not_fade_as_its_prior_goes_to_zero() -> None:
    """⚑ THE FAILURE `sum_m p_m * r_m` CANNOT FIX, ASSERTED AS A CONTRAST.

    That term's gradient carries a factor `p_i`, so its pull on SF's best move
    collapses as we starve that move. The conditional KL's gradient on move `i`
    is `p_S_i - q_i`, which SATURATES toward `-q_i` instead. Both are measured
    here on the same two rows, and the assertion is on the RATIO between the
    starved row and the healthy one -- an absolute threshold would be a
    restatement of the arithmetic rather than a comparison of the two terms.
    """
    reg = _writer_row([(0, 0.0), (1, -10.0), (2, -50.0)], width=6)
    legal = torch.ones(1, 6, dtype=torch.bool)
    healthy = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    starved = torch.tensor([[-6.0, 1.0, 1.0, 0.0, 0.0, 0.0]])

    _, g_healthy = _kl_and_grad(healthy, reg, legal)
    _, g_starved = _kl_and_grad(starved, reg, legal)
    kl_ratio = abs(float(g_starved[0, 0])) / abs(float(g_healthy[0, 0]))

    def regret_grad(logits: torch.Tensor) -> float:
        z = logits.clone().requires_grad_(True)
        p = torch.softmax(z, dim=-1)
        loss = (p * reg).sum()
        (g,) = torch.autograd.grad(loss, [z])
        return abs(float(g[0, 0]))

    regret_ratio = regret_grad(starved) / regret_grad(healthy)
    # Starving the move must not cost the KL its pull the way it costs `p * r`.
    assert kl_ratio > 1.0
    assert regret_ratio < 0.2
    assert kl_ratio > 5.0 * regret_ratio


def test_conditional_expected_regret_is_under_our_conditional_not_our_full_policy() -> None:
    """⚑ E_{m ~ p_S}[r(m)], in CP -- the column entropy cannot substitute for.

    Two distributions with IDENTICAL entropy can order the surfaced moves
    oppositely, and this is the number that sees it. Asserted against the value
    computed by hand from the row's own conditional, and shown to DIFFER from the
    same expectation taken under the full-width policy -- the mutant that reads
    `probs` instead of `p_S` scales every value by M_S and stays plausible.
    """
    reg = _writer_row([(0, 0.0), (1, -10.0), (2, -300.0)], width=6)
    legal = torch.ones(1, 6, dtype=torch.bool)
    logits = torch.tensor([[0.0, 0.0, 0.0, 3.0, 3.0, 3.0]])
    probs = torch.softmax(logits, dim=-1)
    out = sf_shape_conditional_kl(logits, probs, reg, legal, params=SfShapeParams())

    # Uniform over the three surfaced moves: (0 + 10 + 300) / 3.
    assert float(out.regret_cp_given_s) == pytest.approx(310.0 / 3.0, abs=1e-3)
    # Under the FULL policy the same expectation is dominated by the fabricated
    # tail and is a completely different number.
    full = float((probs * reg).sum() * CAP)
    assert abs(full - float(out.regret_cp_given_s)) > 50.0


def test_entropy_can_match_while_the_ranking_is_backwards() -> None:
    """⚑⚑ WHY ENTROPY ALONE MUST NOT GATE THE TERM.

    Our conditional and SF's are permutations of each other, so the entropy gap
    is EXACTLY zero -- and the KL is large and the expected regret is terrible.
    A gate that read only `sf_shape_entropy_gap` would veto the term on this row,
    which is precisely the row it is most useful on.
    """
    reg = _writer_row([(0, 0.0), (1, -100.0), (2, -200.0)], width=6)
    legal = torch.ones(1, 6, dtype=torch.bool)
    # Logits mirroring the teacher's ordering, reversed: temp_cp 100 makes the
    # teacher's surfaced scores 0, -1, -2 in nats, so these are its exact reverse.
    logits = torch.tensor([[-2.0, -1.0, 0.0, -9.0, -9.0, -9.0]])
    probs = torch.softmax(logits.masked_fill(~legal, -1e9), dim=-1)
    out = sf_shape_conditional_kl(logits, probs, reg, legal, params=SfShapeParams())

    assert float(out.h_ours_given_s) == pytest.approx(float(out.h_sf_given_s), abs=1e-5)
    assert float(out.kl) > 0.5
    assert float(out.regret_cp_given_s) > 100.0


@pytest.mark.parametrize(
    ("scored", "label"),
    [
        ([(0, 0.0)], "single surfaced move"),
        ([(0, 0.0), (1, 0.0), (2, 0.0)], "all-equal scores"),
    ],
)
def test_adversarial_rows_are_finite(scored: list[tuple[int, float]], label: str) -> None:
    reg = _writer_row(scored, width=6)
    kl, grad = _kl_and_grad(torch.tensor([[0.0, 2.0, -1.0, 0.5, 0.3, -2.0]]), reg)
    assert math.isfinite(float(kl)), label
    assert torch.isfinite(grad).all(), label


def test_a_single_surfaced_move_is_exactly_zero_at_every_weight() -> None:
    """|S| == 1: `q` and `p_S` are the same one-hot, so the KL is 0 by arithmetic.

    Exact equality, not `approx`: this is what lets the term skip a branch, and
    `sf_shape_active_frac` is the column that reports these rows rather than
    letting them dilute the mean silently.
    """
    reg = _writer_row([(0, 0.0)], width=6)
    kl, grad = _kl_and_grad(torch.tensor([[0.0, 2.0, -1.0, 0.5, 0.3, -2.0]]), reg)
    assert float(kl) == 0.0
    assert grad.abs().max().item() == 0.0


def test_an_empty_surfaced_set_is_exactly_zero() -> None:
    """No legal move surfaced -- including the degenerate all-illegal row."""
    reg = _writer_row([(0, 0.0), (1, -10.0)], width=6)
    legal = torch.zeros(1, 6, dtype=torch.bool)
    kl, grad = _kl_and_grad(torch.tensor([[0.0, 2.0, -1.0, 0.5, 0.3, -2.0]]), reg, legal)
    assert float(kl) == 0.0
    assert grad.abs().max().item() == 0.0


def test_an_absent_regret_vector_leaves_every_column_at_zero() -> None:
    """The whole family must be silent when the shard carries no regret field."""
    outputs, batch = _tiny_batch()
    del batch["sf_p0_regret_t"]
    losses = compute_loss(outputs, batch, sf_shape=SfShapeParams(w=1.0))
    for key, value in losses.items():
        if key.startswith("sf_shape"):
            assert float(value.detach()) == 0.0, key
    assert math.isfinite(float(losses["total"].detach()))


def test_a_lower_temperature_sharpens_the_teacher() -> None:
    """The knob is monotone in the direction its name claims.

    A knob whose effect is not monotone in the intervention cannot be calibrated,
    and calibrating this one against a reference conditional entropy is the
    explicitly-deferred follow-up.
    """
    reg = _writer_row([(0, 0.0), (1, -10.0), (2, -50.0)], width=6)
    legal = torch.ones(1, 6, dtype=torch.bool)
    logits = torch.zeros(1, 6)
    probs = torch.softmax(logits, dim=-1)
    entropies = [
        float(
            sf_shape_conditional_kl(
                logits, probs, reg, legal, params=SfShapeParams(temp_cp=t),
            ).h_sf_given_s
        )
        for t in (10.0, 50.0, 200.0, 1000.0)
    ]
    assert entropies == sorted(entropies), entropies
    # The flattest end approaches uniform over the three surfaced moves.
    assert entropies[-1] == pytest.approx(math.log(3.0), abs=0.05)


def test_a_zero_temperature_is_rejected_because_it_is_a_divisor() -> None:
    with pytest.raises(ValueError, match="sf_shape_temp_cp"):
        SfShapeParams(temp_cp=0.0)
    with pytest.raises(ValueError, match="sf_shape_temp_cp"):
        SfShapeParams.resolve(temp_cp=float("nan"))
    with pytest.raises(ValueError, match="w_sf_shape"):
        SfShapeParams(w=-1.0)


def test_row_entropy_is_finite_at_exact_zeros() -> None:
    h = row_entropy(torch.tensor([[0.5, 0.5, 0.0, 0.0]]))
    assert float(h) == pytest.approx(math.log(2.0))
    assert float(row_entropy(torch.zeros(1, 4))) == 0.0


# --------------------------------------------------------------------------
# 3. THE INSTRUMENT: the columns that must be live at weight zero.
# --------------------------------------------------------------------------


def _tiny_batch(width: int = 32, rows: int = 8) -> tuple[dict, dict]:
    """A batch whose regret rows come from the WRITER, so the mask is realistic.

    ⚑ A batch of `torch.rand` regret vectors -- which is what the floor's own
    fixture uses, correctly, because the floor only reads a cp window -- would
    make EVERY entry "surfaced" here (a random row's max is unique, so all the
    other 31 entries fall strictly below it). The mask under test would then be
    exercised in the one regime it can never see in production, and the
    unsurfaced-mass column would read ~0 for a reason that has nothing to do
    with Stockfish.
    """
    torch.manual_seed(20260818)
    legal = (torch.rand(rows, width) < 0.5).to(torch.float32)
    legal[:, :6] = 1.0
    target = torch.softmax(torch.randn(rows, width), dim=-1) * legal
    target = target / target.sum(-1, keepdim=True)
    reg = torch.cat([
        _writer_row([(0, 0.0), (2, -25.0), (4, -180.0)], width=width)
        for _ in range(rows)
    ])
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
        "sf_p0_regret_t": reg,
        "has_sf_p0_regret": torch.ones(rows),
    }
    return outputs, batch


def test_total_is_bit_identical_at_the_default_weight() -> None:
    """Not `approx`: `0.0 * x` is only zero for finite x, so the term is added to
    `total` under an `if` rather than multiplied by its weight."""
    outputs, batch = _tiny_batch()
    without = compute_loss(outputs, batch)
    with_default = compute_loss(outputs, batch, sf_shape=SfShapeParams())
    a = float(with_default["total"].detach().item())
    b = float(without["total"].detach().item())
    assert a == b
    assert a.hex() == b.hex()


def test_a_nan_in_the_term_cannot_reach_total_at_weight_zero() -> None:
    """The reason inertness is an `if` and not a `* 0.0`.

    The NaN is injected by defeating the validator with `object.__setattr__`
    (nothing a config can reach) precisely so the COMPOSITION rule is under test
    here, not the range check.
    """
    outputs, batch = _tiny_batch()
    poisoned = SfShapeParams()
    object.__setattr__(poisoned, "temp_cp", float("nan"))
    losses = compute_loss(outputs, batch, sf_shape=poisoned)
    assert math.isnan(float(losses["sf_shape_ce_sum"].detach())), "setup: term NaN"
    assert not math.isnan(float(losses["total"].detach()))


def test_the_instrument_is_live_and_invariant_at_weight_zero() -> None:
    """⚑ THE WHOLE POINT OF THE CHANGE.

    Every diagnostic column is computed BEFORE the weight is applied, so the
    entropy comparison is readable before anyone raises `w_sf_shape` -- and it is
    bit-equal at w=0.0 and w=1.0 on the same batch, which is what makes it an
    instrument rather than a function of the arm.
    """
    outputs, batch = _tiny_batch()
    off = compute_loss(outputs, batch, sf_shape=SfShapeParams(w=0.0))
    on = compute_loss(outputs, batch, sf_shape=SfShapeParams(w=1.0))

    columns = [k for k in off if k.startswith("sf_shape")]
    assert len(columns) == 12, columns
    for key in columns:
        assert float(off[key].detach()) == float(on[key].detach()), key

    assert float(off["sf_shape_ce_sum"].detach()) > 0.0
    assert float(on["total"].detach()) > float(off["total"].detach())


def test_the_surfaced_count_column_reports_the_real_mask_not_the_legal_count() -> None:
    """The mask-health column, on a batch whose true coverage is known: 3 of ~19.

    ⚑ THIS IS THE COLUMN THAT WOULD CATCH THE RECOVERY SILENTLY COLLAPSING. A
    mutant that returns `legal` as the surfaced set (i.e. treats the fabricated
    79% as real SF opinion, the defect this term is designed around) leaves every
    other column plausible and moves this one to the legal count.
    """
    outputs, batch = _tiny_batch()
    losses = compute_loss(outputs, batch)
    rows = float(losses["sf_own_regret_rows"].detach())
    assert rows == 8.0
    assert float(losses["sf_shape_surfaced_sum"].detach()) / rows == 3.0
    legal_per_row = float(batch["legal_mask"].sum()) / rows
    assert legal_per_row > 10.0, "setup: the legal count must differ from |S|"


def test_the_surfaced_mass_column_is_a_real_share_not_a_structural_one() -> None:
    """⚑⚑ M_S: THE COLUMN THAT AN OFFLINE MEASUREMENT COULD NOT GET, AND THE ONE
    THAT DECIDES WHETHER THIS LOSS SHOULD EVER BE ENABLED.

    On banked wide-era shards SF's labels cover 26.63 of 26.82 legal moves, so
    the surfaced set is not restricted and M_S reads ~1 for a reason that has
    nothing to do with the policy. Here the set is genuinely 3 of ~19, so the
    number is a real share -- and this assertion FAILS against a fixture that
    reproduced the vacuous case, which is the only thing that makes it a test of
    the measurement rather than of the arithmetic.
    """
    outputs, batch = _tiny_batch()
    losses = compute_loss(outputs, batch)
    rows = float(losses["sf_own_regret_rows"].detach())
    mass = float(losses["sf_shape_surfaced_mass_sum"].detach()) / rows
    assert 0.0 < mass < 0.7, mass


def test_p_on_sfs_best_move_is_absolute_and_names_the_floors_move() -> None:
    """It must be p_own on the argmin of regret, NOT the conditional probability.

    Absolute, so it moves with M_S -- that is what makes it comparable to
    `sf_policy_floor_tau`, which is a floor on this same quantity. Asserted
    against the probability read directly off the batch's own logits at the index
    the writer made SF's best, never against a second derivation of the argmin.
    """
    outputs, batch = _tiny_batch()
    losses = compute_loss(outputs, batch)
    rows = float(losses["sf_own_regret_rows"].detach())
    got = float(losses["sf_shape_p_sf_best_sum"].detach()) / rows

    masked = outputs["policy"].detach().masked_fill(batch["legal_mask"] < 0.5, -1e9)
    probs = torch.softmax(masked, dim=-1)
    # Move 0 is the one `_tiny_batch`'s writer row scored at regret 0.0.
    assert got == pytest.approx(float(probs[:, 0].mean()), abs=1e-6)
    # And it is strictly below the conditional probability of the same move,
    # because M_S < 1 -- the distinction the column exists to carry.
    conditional = float((probs[:, 0] / probs[:, [0, 2, 4]].sum(-1)).mean())
    assert got < conditional


def test_the_entropy_gap_is_the_teacher_minus_us_on_one_support() -> None:
    """Sign convention, pinned: POSITIVE means WE are the sharp one.

    Also pins that the gap column is the difference of the two entropy columns
    rather than an independently-derived third number, and that the row rate
    agrees with the sign of the gap on a batch whose rows are identical.
    """
    outputs, batch = _tiny_batch()
    losses = compute_loss(outputs, batch)
    rows = float(losses["sf_own_regret_rows"].detach())
    teacher = float(losses["sf_shape_h_sf_given_s_sum"].detach()) / rows
    ours = float(losses["sf_shape_h_ours_given_s_sum"].detach()) / rows
    gap = float(losses["sf_shape_entropy_gap_sum"].detach()) / rows
    sharper = float(losses["sf_shape_sharper_sum"].detach()) / rows
    assert gap == pytest.approx(teacher - ours, abs=1e-6)
    # A ROW RATE, not a restatement of the mean's sign: this batch's rows differ,
    # so it must land strictly inside (0, 1). An assertion that only checked the
    # mean's sign would pass for a column hardwired to `gap > 0`.
    assert 0.0 < sharper < 1.0, sharper

    # ... and it reads exactly 1.0 on a batch built so every row IS sharper than
    # its teacher: a near-one-hot policy against a 3-move surfaced set.
    sharp_outputs, sharp_batch = _tiny_batch()
    sharp_outputs["policy"] = sharp_outputs["policy"].detach() * 0.0
    sharp_outputs["policy"][:, 0] = 20.0
    sharp = compute_loss(sharp_outputs, sharp_batch)
    assert float(sharp["sf_shape_sharper_sum"].detach()) / rows == 1.0
    assert float(sharp["sf_shape_entropy_gap_sum"].detach()) > 0.0
    # The FULL-support policy entropy is a different, larger quantity than the
    # conditional one; publishing only one of them is how a number gets quoted
    # against the wrong support.
    policy_h = float(losses["sf_shape_h_ours_full_legal_sum"].detach()) / rows
    assert policy_h > ours


def test_rows_below_two_surfaced_moves_are_reported_not_hidden() -> None:
    """`sf_shape_active_frac`'s numerator, on a batch that is half degenerate."""
    outputs, batch = _tiny_batch()
    batch["sf_p0_regret_t"][:4] = _writer_row([(0, 0.0)], width=32).expand(4, -1)
    losses = compute_loss(outputs, batch)
    assert float(losses["sf_shape_active_sum"].detach()) == 4.0
    assert float(losses["sf_own_regret_rows"].detach()) == 8.0


# --------------------------------------------------------------------------
# 3b. MATCHED SUPPORT: the confound made impossible rather than avoided.
# --------------------------------------------------------------------------


def test_matched_support_entropies_ignore_a_tail_the_target_never_covers() -> None:
    """⚑⚑ TWO ROWS WITH IDENTICAL CONDITIONAL ENTROPY AND DIFFERENT FULL ENTROPY.

    The target's support is {0, 1} and it is (0.5, 0.5) there. Our policy is
    (0.4, 0.4) on that support -- conditionally IDENTICAL, ln 2 -- plus a long
    thin tail of 0.2 spread over four moves the target never covers. So:

      * the matched-support entropies must be EQUAL,
      * the tail must be reported by `tail_mass_ours` and NOWHERE ELSE, and
      * the FULL-support entropies must differ, which is what makes the first
        assertion non-trivial.

    ⚑ MUTANT M2. Compute the two entropies on unmatched FULL support --
    `row_entropy(probs)` against `row_entropy(target)` -- and the first assertion
    fails by 0.5 nats. That is the exact shape of the `audit_targets.py` reading
    (raw policy 0.8827 over ~27 legal against target 0.6255 over ~16 candidates)
    which this helper exists to make unrepeatable.
    """
    probs = torch.tensor([[0.4, 0.4, 0.05, 0.05, 0.05, 0.05]])
    target = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])
    stats = matched_support_entropy_stats(probs, target, None)

    assert float(stats.h_ours) == pytest.approx(math.log(2.0), abs=1e-6)
    assert float(stats.h_target) == pytest.approx(math.log(2.0), abs=1e-6)
    assert float(stats.h_ours) == pytest.approx(float(stats.h_target), abs=1e-6)
    assert float(stats.support_size) == 2.0
    assert float(stats.tail_mass_ours) == pytest.approx(0.2, abs=1e-6)

    # The unmatched comparison the helper refuses to make, shown to be different:
    # without this the equality above could hold for a trivial reason.
    assert float(row_entropy(probs)) - float(row_entropy(target)) > 0.4


def test_matched_support_drops_illegal_moves_from_the_shared_support() -> None:
    """`T = (target > 0) AND legal`, so an illegal target entry cannot join it."""
    probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    target = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    legal = torch.tensor([[True, False, True, True]])
    stats = matched_support_entropy_stats(probs, target, legal)
    assert float(stats.support_size) == 1.0
    assert float(stats.h_ours) == 0.0
    assert float(stats.tail_mass_ours) == pytest.approx(0.75, abs=1e-6)


def test_matched_support_is_finite_on_a_row_with_no_target() -> None:
    stats = matched_support_entropy_stats(
        torch.tensor([[0.25, 0.25, 0.25, 0.25]]), torch.zeros(1, 4), None,
    )
    assert float(stats.support_size) == 0.0
    assert float(stats.h_ours) == 0.0
    assert float(stats.h_target) == 0.0
    assert float(stats.tail_mass_ours) == pytest.approx(1.0)


def test_the_matched_support_columns_reach_compute_loss() -> None:
    """The family is published over its OWN row count, on every batch."""
    outputs, batch = _tiny_batch()
    losses = compute_loss(outputs, batch)
    rows = float(losses["policy_target_rows"].detach())
    assert rows == 8.0
    size = float(losses["policy_support_size_sum"].detach()) / rows
    tail = float(losses["policy_tail_mass_sum"].detach()) / rows
    gap = float(losses["policy_support_gap_sum"].detach()) / rows
    h_ours = float(losses["policy_support_h_ours_sum"].detach()) / rows
    h_target = float(losses["policy_support_h_target_sum"].detach()) / rows
    assert 0.0 < size < float(batch["legal_mask"].shape[-1])
    assert 0.0 <= tail < 1.0
    assert gap == pytest.approx(h_target - h_ours, abs=1e-6)


# --------------------------------------------------------------------------
# 4. THE WIRING: config -> Trainer -> `_loss_kwargs` -> `compute_loss`.
# --------------------------------------------------------------------------

_TRAINER_CORE = {"device": "cpu", "lr": 1e-3, "no_amp": True}

# The `TrainMetrics` fields `_train_metrics_dict` requires before it will build a
# row at all; everything this file asserts on rides on top of them.
_REQUIRED_METRICS = dict.fromkeys(
    ("loss", "policy_loss", "soft_policy_loss", "future_policy_loss", "wdl_loss",
     "sf_move_loss", "sf_move_acc", "sf_eval_loss", "categorical_loss",
     "volatility_loss", "sf_volatility_loss", "moves_left_loss"), 0.0,
)


class _TinyModel(torch.nn.Module):
    """Smallest model a `Trainer` will build an optimizer over."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32),
        }


def _trainer(overrides: dict, tmp_path):
    from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    cfg: dict = {"train": dict(_TRAINER_CORE)}
    cfg["train"].update(overrides)
    ctor = trainer_kwargs_from_config(
        flatten_run_config_defaults(cfg), log_dir=tmp_path,
    )
    return Trainer(_TinyModel(), prefetch_batches=False, **ctor)


def test_the_yaml_keys_reach_the_loss_call(tmp_path) -> None:
    """config -> Trainer -> `_loss_kwargs` -> `compute_loss`, end to end.

    Compared against a LITERAL `SfShapeParams`, never against a second call to
    `resolve`: a re-derivation in the test body agrees with a trainer that
    ignored the config entirely.
    """
    trainer = _trainer({"w_sf_shape": 0.37, "sf_shape_temp_cp": 42.0}, tmp_path)
    assert trainer._loss_kwargs["sf_shape"] == SfShapeParams(w=0.37, temp_cp=42.0)

    # And the configured SHAPE reaches the ARITHMETIC. Presence of the key would
    # be satisfied by a trainer that forwarded an all-defaults object.
    outputs, batch = _tiny_batch()
    got = compute_loss(outputs, batch, **trainer._loss_kwargs)
    default_kwargs = dict(trainer._loss_kwargs)
    default_kwargs["sf_shape"] = SfShapeParams()
    defaults = compute_loss(outputs, batch, **default_kwargs)
    assert float(got["sf_shape_h_sf_given_s_sum"].detach()) != float(
        defaults["sf_shape_h_sf_given_s_sum"].detach()
    )
    assert float(got["total"].detach()) != float(defaults["total"].detach())


def test_the_defaults_reach_the_loss_call_and_are_off(tmp_path) -> None:
    """An empty config must not enable the term on the production path."""
    assert _trainer({}, tmp_path)._loss_kwargs["sf_shape"] == SfShapeParams(
        w=0.0, temp_cp=SF_SHAPE_TEMP_CP_DEFAULT,
    )


def test_a_live_weight_push_reaches_the_loss_kwargs(tmp_path) -> None:
    """`w_sf_shape` rides `TRAINER_WEIGHT_KEYS`; the SHAPE must not move.

    Asserted at `_loss_kwargs`, not at the attribute: a trainer that accepts the
    push and never forwards it is exactly the defect this repo keeps producing,
    and reading back `trainer.w_sf_shape` cannot see it.
    """
    from chess_anti_engine.tune.trainable_config_ops import _apply_lr_gamma_weights

    trainer = _trainer({"sf_shape_temp_cp": 42.0}, tmp_path)
    assert trainer._loss_kwargs["sf_shape"].w == 0.0

    _apply_lr_gamma_weights(trainer, {"w_sf_shape": 0.77}, rescale_current_lr=True)
    pushed = trainer._loss_kwargs["sf_shape"]
    assert pushed.w == 0.77
    assert pushed.temp_cp == 42.0


def test_a_live_weight_push_is_range_validated(tmp_path) -> None:
    """`replace` re-runs `__post_init__`, so a live negative weight raises."""
    from chess_anti_engine.tune.trainable_config_ops import _apply_lr_gamma_weights

    trainer = _trainer({}, tmp_path)
    _apply_lr_gamma_weights(trainer, {"w_sf_shape": -1.0}, rescale_current_lr=True)
    with pytest.raises(ValueError, match="w_sf_shape"):
        _ = trainer._loss_kwargs["sf_shape"]


def test_an_out_of_range_temperature_is_rejected_at_config_load() -> None:
    """CLAUDE.md category (b): the trial dies once, loudly, naming the key."""
    from chess_anti_engine.tune.trial_config import TrialConfig
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    flat = flatten_run_config_defaults({"train": {"sf_shape_temp_cp": 0.0}})
    with pytest.raises(ValueError, match="sf_shape_temp_cp"):
        TrialConfig.from_dict(flat)


def test_the_weight_is_covered_by_the_holdout_ruler() -> None:
    """A membership flip must MOVE the ruler, not freeze the best-model record.

    `eval_ruler.active_loss_terms` hashes the SET of non-zero weights, read off
    `TRAINER_WEIGHT_KEYS`, and `compute_loss` adds this term under an `if` -- so
    switching it on steps `total` strictly up (the KL is non-negative) with no
    way back down. Membership in that tuple is what hands the record over.
    """
    from chess_anti_engine.train.eval_ruler import active_loss_terms

    assert "w_sf_shape" in TRAINER_WEIGHT_KEYS
    weights = dict.fromkeys(TRAINER_WEIGHT_KEYS, 0.0)
    assert "w_sf_shape" not in active_loss_terms(weights)
    weights["w_sf_shape"] = 0.5
    assert "w_sf_shape" in active_loss_terms(weights)


def test_the_temperature_is_classified_restart_required() -> None:
    """It is folded into a frozen object at construction, so a live edit no-ops.

    Listing it turns that silence into a restart-required WARNING on the
    iteration the edit lands.
    """
    from chess_anti_engine.tune.trainable_config_ops import restart_required_config_keys

    assert "sf_shape_temp_cp" in restart_required_config_keys()
    assert "w_sf_shape" not in restart_required_config_keys()


def test_the_columns_reach_train_metrics() -> None:
    """`compute_loss` sums -> `TrainMetrics`, through the real pooling path."""
    from chess_anti_engine.train.trainer import _loss_sums_to_metric_kwargs

    out = _loss_sums_to_metric_kwargs(
        {
            "sf_shape_ce_sum": 4.0,
            "sf_shape_active_sum": 3.0,
            "sf_shape_h_sf_given_s_sum": 8.0,
            "sf_shape_h_ours_given_s_sum": 2.0,
            "sf_shape_entropy_gap_sum": 6.0,
            "sf_shape_sharper_sum": 4.0,
            "sf_shape_h_ours_full_legal_sum": 12.0,
            "sf_shape_surfaced_sum": 20.0,
            "sf_shape_surfaced_mass_sum": 6.0,
            "sf_shape_p_sf_best_sum": 2.0,
            "sf_shape_regret_cp_sum": 100.0,
            "sf_own_regret_rows": 4.0,
            "policy_support_h_ours_sum": 10.0,
            "policy_support_h_target_sum": 8.0,
            "policy_support_gap_sum": -2.0,
            "policy_support_size_sum": 160.0,
            "policy_tail_mass_sum": 0.4,
            "policy_target_rows": 10.0,
        },
        n=2.0,
    )
    assert out["m_sf_shape"] == 1.0
    assert out["sf_shape_active_frac"] == 0.75
    assert out["sf_shape_h_sf_given_s"] == 2.0
    assert out["sf_shape_h_ours_given_s"] == 0.5
    assert out["sf_shape_entropy_gap"] == 1.5
    assert out["sf_shape_sharper_frac"] == 1.0
    assert out["sf_shape_h_ours_full_legal"] == 3.0
    assert out["sf_shape_surfaced_moves"] == 5.0
    assert out["sf_shape_surfaced_mass"] == 1.5
    assert out["sf_shape_p_sf_best"] == 0.5
    assert out["sf_shape_regret_cp_given_s"] == 25.0
    # ⚑ THE MATCHED-SUPPORT FAMILY DIVIDES BY ITS OWN ROW COUNT (10), not by the
    # SF family's (4). Borrowing `sf_own_regret_rows` would silently restrict a
    # whole-batch measurement to the ~21% of rows carrying SF regret, and every
    # number would stay plausible.
    assert out["policy_support_h_ours"] == 1.0
    assert out["policy_support_h_target"] == 0.8
    assert out["policy_support_gap"] == pytest.approx(-0.2)
    assert out["policy_support_size"] == 16.0
    assert out["policy_tail_mass_ours"] == pytest.approx(0.04)


def test_the_columns_reach_the_result_row() -> None:
    """`TrainMetrics` -> the per-iteration result row an operator actually reads."""
    from dataclasses import replace as _replace

    from chess_anti_engine.train.trainer import TrainMetrics
    from chess_anti_engine.tune.trainable_report import _train_metrics_dict

    row = _train_metrics_dict(_replace(
        TrainMetrics(**_REQUIRED_METRICS),
        m_sf_shape=0.5,
        sf_shape_active_frac=0.9,
        sf_shape_h_sf_given_s=1.2,
        sf_shape_h_ours_given_s=0.7,
        sf_shape_entropy_gap=0.5,
        sf_shape_sharper_frac=0.64,
        sf_shape_h_ours_full_legal=0.68,
        sf_shape_surfaced_moves=5.57,
        sf_shape_surfaced_mass=0.69,
        sf_shape_p_sf_best=0.22,
        sf_shape_regret_cp_given_s=57.7,
        policy_support_h_ours=0.61,
        policy_support_h_target=0.55,
        policy_support_gap=-0.06,
        policy_support_size=15.9,
        policy_tail_mass_ours=0.048,
    ))
    assert row["m_sf_shape"] == 0.5
    assert row["sf_shape_entropy_gap"] == 0.5
    assert row["sf_shape_sharper_frac"] == 0.64
    assert row["sf_shape_surfaced_moves"] == pytest.approx(5.57)
    assert row["sf_shape_h_ours_given_s"] == pytest.approx(0.7)
    assert row["sf_shape_h_sf_given_s"] == pytest.approx(1.2)
    assert row["sf_shape_surfaced_mass"] == pytest.approx(0.69)
    assert row["sf_shape_p_sf_best"] == pytest.approx(0.22)
    assert row["sf_shape_regret_cp_given_s"] == pytest.approx(57.7)
    assert row["policy_support_h_ours"] == pytest.approx(0.61)
    assert row["policy_support_h_target"] == pytest.approx(0.55)
    assert row["policy_support_gap"] == pytest.approx(-0.06)
    assert row["policy_support_size"] == pytest.approx(15.9)
    assert row["policy_tail_mass_ours"] == pytest.approx(0.048)


# ─────────────────────────────────────────────────────────────────────────────
# Review round 2 (PR #479): two independent reviewers, complementary findings.
# Every guard below is a REGRESSION guard -- each one fails against the version
# of the code that shipped to review, and the mutant is named in the docstring.
# ─────────────────────────────────────────────────────────────────────────────


def test_a_nan_regret_row_leaves_every_sf_shape_column_finite() -> None:
    """One NaN must not take out the readout for the whole iteration.

    MUTANT: restore `* surfaced_f` in `regret_cp_given_s` -- `0.0 * NaN == NaN`,
    so masking by multiplication does not mask. The reported value is an
    ITERATION-WIDE SUM, so a single bad row poisoned the column for every row
    behind it. `total` was never at risk; the claim that was false is
    "NaN-tolerant by construction" applied to the READOUT.
    """
    outputs, batch = _tiny_batch()
    reg = batch["sf_p0_regret_t"].clone()
    reg[0, 3] = float("nan")
    batch = {**batch, "sf_p0_regret_t": reg}

    for w in (0.0, 0.7):
        got = compute_loss(outputs, batch, sf_shape=SfShapeParams(w=w))
        bad = {
            k: float(v.detach())
            for k, v in got.items()
            if k.startswith("sf_shape") and not math.isfinite(float(v.detach()))
        }
        assert not bad, f"non-finite sf_shape columns at w={w}: {bad}"
        assert math.isfinite(float(got["total"].detach()))


def test_p_sf_best_is_zero_on_an_empty_surfaced_set() -> None:
    """MUTANT: delete `* (count > 0)` in `p_sf_best`.

    Survived the shipped suite (M17). With the guard gone, a row whose set is
    empty -- a NaN row, or one fully at the 1000 cp cap -- reports `p_own` at an
    arbitrary FABRICATED index as though it were SF's best move. The existing
    empty-set test asserts only `kl` and `grad`, so the column was unpinned.
    """
    reg = torch.full((1, 6), 500.0)          # every entry ties -> nothing surfaced
    legal = torch.ones((1, 6), dtype=torch.bool)
    logits = torch.zeros((1, 6), requires_grad=True)
    out = sf_shape_conditional_kl(logits, torch.softmax(logits, -1), reg, legal,
                                  params=SfShapeParams(w=0.7))
    assert float(out.surfaced_count[0].detach()) == 0.0
    assert float(out.p_sf_best[0].detach()) == 0.0
    assert float(out.kl[0].detach()) == 0.0


def test_the_two_surfaced_set_rules_diverge_only_on_a_fully_covered_row() -> None:
    """⚑ TWO recoveries of S exist in this module and they DISAGREE. Pinned, not fixed.

    `sf_surfaced_move_mask` (this feature) takes the FULL-ROW max;
    `_sf_regret_surfaced_and_row_max` (#447, consumed by the LIVE
    `sf_policy_floor` at `w = 0.8`) takes the LEGAL-ONLY max. The writer puts the
    fill `d` on uncovered LEGAL indices, so on a PARTIALLY covered row both rules
    agree. On a FULLY covered row no legal index carries `d`, the legal-only max
    IS the worst real regret, and `reg < row_max` drops SF's worst legal move
    from the set.

    ⚑ DELIBERATELY NOT UNIFIED HERE, and the reason is scope, not taste.
    Unifying downward would regress this term to a rule that is wrong on fully
    covered rows. Unifying upward would change which moves the LIVE floor term
    trains on -- a training-affecting change that needs its own ledger entry,
    prereg and readout, not a drive-by inside a default-off PR. This test exists
    so the divergence is discoverable and cannot widen silently.
    """
    from chess_anti_engine.train.losses import _sf_regret_surfaced_and_row_max

    legal = torch.tensor([[True, True, True, True, False, False]])

    partial = torch.tensor([[10.0, 25.0, 40.0, 60.0, 60.0, 60.0]])
    new_p, _ = sf_surfaced_move_mask(partial, legal)
    old_p, _ = _sf_regret_surfaced_and_row_max(partial, legal)
    assert new_p.tolist() == old_p.tolist(), "partially covered rows must agree"

    full = torch.tensor([[10.0, 25.0, 40.0, 55.0, 90.0, 90.0]])
    new_f, _ = sf_surfaced_move_mask(full, legal)
    old_f, _ = _sf_regret_surfaced_and_row_max(full, legal)
    assert new_f[0].tolist() == [True, True, True, True, False, False]
    assert old_f[0].tolist() == [True, True, True, False, False, False]
    assert new_f.tolist() != old_f.tolist()


def test_a_nonzero_weight_moves_the_policy_gradient_only_inside_the_set() -> None:
    """The take-effect observation, through `total.backward()` -- not the kernel.

    The shipped suite asserted the gradient property on `sf_shape_conditional_kl`
    in isolation and the weight property on the loss VALUE, but never drove
    `compute_loss` -> `total.backward()`. So "the term reaches the optimizer"
    rested on reading the composition rather than on running it. MUTANT: hard-wire
    the weighted term to 0.0, or drop it from `weighted_terms` -- both leave the
    isolated-kernel tests green.
    """
    outputs, batch = _tiny_batch()
    logits = outputs["policy"]
    surfaced, _ = sf_surfaced_move_mask(
        batch["sf_p0_regret_t"], batch["legal_mask"].to(torch.bool),
    )

    def grad_at(w: float) -> torch.Tensor:
        lg = logits.detach().clone().requires_grad_(True)
        loss = compute_loss({**outputs, "policy": lg}, batch,
                            sf_shape=SfShapeParams(w=w))
        loss["total"].backward()
        assert lg.grad is not None
        return lg.grad.detach().clone()

    off, on = grad_at(0.0), grad_at(0.7)
    outside = ~surfaced
    assert torch.equal(off[outside], on[outside]), "gradient moved OUTSIDE the set"
    assert not torch.equal(off[surfaced], on[surfaced]), "no gradient reached the set"


def test_the_temperature_band_rejects_the_silent_absurdity() -> None:
    """MUTANT: delete the band check -- `1e-9` is then accepted in silence.

    CLAUDE.md category (c): in-schema, in-sign, never range-checked, and the
    slowest failure to notice. `sf_shape_temp_cp` is a DIVISOR whose own
    docstring says it must be swept, so hand-typed values are expected. Outside
    the band `q_S` collapses to a delta or flattens to uniform and
    `sf_shape_h_sf_given_s` -- the column the calibration is read off -- stops
    meaning anything, WITHOUT looking wrong.
    """
    from chess_anti_engine.train.losses import (
        SF_SHAPE_TEMP_CP_MAX,
        SF_SHAPE_TEMP_CP_MIN,
    )

    SfShapeParams(w=0.0, temp_cp=SF_SHAPE_TEMP_CP_MIN)   # endpoints inclusive
    SfShapeParams(w=0.0, temp_cp=SF_SHAPE_TEMP_CP_MAX)
    SfShapeParams(w=0.0, temp_cp=SF_SHAPE_TEMP_CP_DEFAULT)

    for bad in (1e-9, 0.5, 1e5, 1e12):
        with pytest.raises(ValueError, match="sf_shape_temp_cp"):
            SfShapeParams(w=0.0, temp_cp=bad)

    # The pre-existing sign/finiteness guard must still fire on its own terms.
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sf_shape_temp_cp"):
            SfShapeParams(w=0.0, temp_cp=bad)


def test_the_trainer_announces_sf_shape_from_the_consumers_object(tmp_path, capsys) -> None:
    """MUTANT: rebuild the line from `self.w_sf_shape` instead of `_loss_kwargs`.

    That mutant still prints a plausible line while the object `compute_loss`
    actually receives is whatever the trainer forwarded -- the exact "announce
    from the consumer's own parameter" trap. Printing unconditionally (including
    at the shipped `w=0.0`) is deliberate: "present and inert" and "never wired"
    are the two states this line exists to separate.
    """
    _trainer({"w_sf_shape": 0.37, "sf_shape_temp_cp": 42.0}, tmp_path)
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("[trainer] sf_shape ")]
    assert len(line) == 1, "expected exactly one sf_shape announcement"
    assert "w=0.37" in line[0]
    assert "temp_cp=42.0" in line[0]
    assert "active=True" in line[0]
    # The knob is under the ruler when it is on -- the operator-visible half of
    # the fix an independent review of #479 found missing.
    assert "in_ruler_shape=True" in line[0]
