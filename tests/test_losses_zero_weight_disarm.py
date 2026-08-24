"""A ZERO loss weight must DISARM its term, because ``0.0 * nan`` is ``nan``.

⚑ THE HAZARD IS NOT HYPOTHETICAL AND `masked_mean` IS NOT A DEFENCE. Its
DENOMINATOR is ``clamp_min(1.0)``, so an empty mask cannot divide by zero — but
its NUMERATOR is ``(x * mask).sum()``, and ``0.0 * nan`` is ``nan`` there too.
So a term whose per-sample loss is NaN over an EMPTY mask reports NaN, not 0.0,
and the flat ``w * term`` assembly then carried that NaN into ``total`` — and
into every gradient built from it — through a weight that is supposed to mean
"off".

That regime is the EXPECTED one for the AZ-purity arm rather than an edge case:
it zeroes several loss weights, and gen-0 shards carry no SF fields at all, so a
zero-weighted SF term over an empty denominator is what a normal iteration of
that arm looks like.

⚑ EVERY CASE HERE IS PAIRED WITH ITS OWN NON-VACUITY CHECK. The same poisoned
batch is also run with the weight at 0.5 and REQUIRED to produce a NaN
``total``: a disarm assertion over a poison that does not actually poison passes
for the wrong reason, which is this repo's signature defect wearing a test's
clothes. The source-level mutant table (one guard reverted at a time) is in the
PR.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from chess_anti_engine.train.losses import SfPolicyFloorParams, compute_loss

Outputs = dict[str, torch.Tensor]
Batch = dict[str, torch.Tensor]
Kwargs = dict[str, object]

_B = 2
_ACTIONS = 3


#: The assembly, in `compute_loss`'s own order: (weight kwarg, returned component
#: key). The fold helpers below re-implement `total` from this table, so a guard
#: that skips the WRONG term is caught by an arithmetic mismatch and not only by
#: a NaN. `sf_policy_floor` carries its weight inside its params object, which is
#: why the weight is read through `_weight_of` rather than indexed directly.
_ASSEMBLY: tuple[tuple[str, str], ...] = (
    ("w_policy", "policy_ce"),
    ("w_soft", "soft_policy_ce"),
    ("w_future", "future_policy_ce"),
    ("w_sf_own", "sf_own_ce"),
    ("w_sf_own_regret", "sf_own_regret"),
    ("w_wdl", "wdl_ce"),
    ("w_sf_move", "sf_move_ce"),
    ("w_sf_eval", "sf_eval_ce"),
    ("w_categorical", "categorical_ce"),
    ("w_volatility", "volatility"),
    ("w_sf_volatility", "sf_volatility"),
    ("w_moves_left", "moves_left"),
    ("sf_policy_floor", "sf_policy_floor"),
)

#: Every term ON, with values distinct enough that a mis-ordered fold does not
#: coincidentally agree. `w_sf_own` / `w_sf_own_regret` default to 0.0 in
#: production and are switched on here precisely so their guards are exercised.
_ALL_ON: dict[str, float] = {
    "w_policy": 1.0,
    "w_soft": 0.5,
    "w_future": 0.15,
    "w_sf_own": 0.2,
    "w_sf_own_regret": 0.3,
    "w_wdl": 1.0,
    "w_sf_move": 0.15,
    "w_sf_eval": 0.11,
    "w_categorical": 0.10,
    "w_volatility": 0.05,
    "w_sf_volatility": 0.07,
    "w_moves_left": 0.02,
}

_FLOOR_ON = 0.25


def _outputs() -> Outputs:
    return {
        "policy": torch.tensor(
            [[2.0, -1.0, 0.0], [0.5, 1.5, -1.0]], requires_grad=True,
        ),
        "policy_soft": torch.tensor(
            [[0.5, 1.0, -0.5], [1.0, 0.0, 0.5]], requires_grad=True,
        ),
        "policy_future": torch.tensor(
            [[0.0, -1.0, 1.5], [0.2, 0.4, -0.6]], requires_grad=True,
        ),
        "policy_sf": torch.tensor(
            [[1.0, -1.0, 0.5], [-0.5, 0.5, 1.5]], requires_grad=True,
        ),
        "wdl": torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.0, -0.1]], requires_grad=True),
        "sf_eval": torch.tensor(
            [[0.2, 0.1, -0.4], [-0.2, 0.3, 0.1]], requires_grad=True,
        ),
        "categorical": torch.tensor(
            [[0.3, -0.1, 0.2, 0.0], [0.1, 0.2, -0.2, 0.4]], requires_grad=True,
        ),
        "volatility": torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]], requires_grad=True,
        ),
        "sf_volatility": torch.tensor(
            [[0.3, 0.2, 0.1], [0.0, 0.1, 0.2]], requires_grad=True,
        ),
        "moves_left": torch.tensor([[0.5], [0.25]], requires_grad=True),
    }


def _batch() -> Batch:
    """Every head live, every target present, every row mask full."""
    ones = torch.ones((_B,), dtype=torch.float32)
    return {
        "x": torch.zeros((_B, 1, 1, 1), dtype=torch.float32),
        "policy_t": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "wdl_t": torch.tensor([0, 2], dtype=torch.long),
        "has_policy": ones.clone(),
        "is_network_turn": ones.clone(),
        "policy_soft_t": torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]]),
        "has_policy_soft": ones.clone(),
        "future_policy_t": torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]),
        "has_future": ones.clone(),
        "sf_policy_t": torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]),
        "has_sf_policy": ones.clone(),
        "has_sf_move": ones.clone(),
        "sf_p0_policy_t": torch.tensor([[0.1, 0.8, 0.1], [0.3, 0.3, 0.4]]),
        "has_sf_p0": ones.clone(),
        # ⚑ SHAPED SO THE FLOOR ACTUALLY BINDS. SF's best move (regret 0.0) is
        # the one the net likes LEAST in both rows, and each row carries a
        # second move inside the `delta_cp` window, so `F` has an ADAPTIVE
        # member as well as the unconditional top-1 — which is the only way
        # `tau` (as opposed to `tau_top1`) reaches the arithmetic at all. A
        # fixture where the floor is structurally 0.0 would make the
        # `sf_policy_floor` case of every test below vacuous.
        "sf_p0_regret_t": torch.tensor([[0.9, 0.02, 0.0], [0.03, 0.7, 0.0]]),
        "has_sf_p0_regret": ones.clone(),
        "sf_wdl": torch.tensor([[0.2, 0.5, 0.3], [0.6, 0.2, 0.2]]),
        "has_sf_wdl": ones.clone(),
        "categorical_t": torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]],
        ),
        "has_categorical": ones.clone(),
        "volatility_t": torch.tensor([[0.2, 0.1, 0.0], [0.0, 0.2, 0.1]]),
        "has_volatility": ones.clone(),
        "sf_volatility_t": torch.tensor([[0.0, 0.1, 0.2], [0.2, 0.1, 0.0]]),
        "has_sf_volatility": ones.clone(),
        "moves_left": torch.tensor([0.25, 0.5]),
        "has_moves_left": ones.clone(),
        "legal_mask": torch.ones((_B, _ACTIONS)),
        "has_legal_mask": ones.clone(),
    }


def _nan_rows(width: int) -> torch.Tensor:
    return torch.full((_B, width), float("nan"), dtype=torch.float32)


def _zeros() -> torch.Tensor:
    return torch.zeros((_B,), dtype=torch.float32)


@dataclass
class _Fix:
    """One evaluation's mutable inputs, handed to a poison as a single object.

    A poison touches one or two of the three; passing them as three parameters
    would leave the others unread in almost every function here.
    """

    outputs: Outputs
    batch: Batch
    kwargs: Kwargs


def _poison_policy(fix: _Fix) -> None:
    fix.batch["policy_t"] = _nan_rows(_ACTIONS)
    fix.batch["has_policy"] = _zeros()


def _poison_soft(fix: _Fix) -> None:
    fix.batch["policy_soft_t"] = _nan_rows(_ACTIONS)
    fix.batch["has_policy_soft"] = _zeros()


def _poison_future(fix: _Fix) -> None:
    fix.batch["future_policy_t"] = _nan_rows(_ACTIONS)
    fix.batch["has_future"] = _zeros()


def _poison_sf_own(fix: _Fix) -> None:
    fix.batch["sf_p0_policy_t"] = _nan_rows(_ACTIONS)
    fix.batch["has_sf_p0"] = _zeros()


def _poison_sf_own_regret(fix: _Fix) -> None:
    fix.batch["sf_p0_regret_t"] = _nan_rows(_ACTIONS)
    fix.batch["has_sf_p0_regret"] = _zeros()


def _poison_wdl(fix: _Fix) -> None:
    # ⚑ THE VALUE HEAD'S ONLY ROW MASK IS `net_mask`, which every other head is
    # also masked by, so emptying it would empty the whole batch and make the
    # arithmetic comparison below trivial. The empty-denominator route for this
    # head gets its own test
    # (`test_an_empty_net_mask_over_a_nan_value_head_is_disarmed_too`); here the
    # NaN arrives through the head's logits with the mask left full, so the other
    # twelve terms keep real, distinct values.
    fix.outputs["wdl"] = torch.full((_B, 3), float("nan"), requires_grad=True)


def _poison_sf_move(fix: _Fix) -> None:
    fix.batch["sf_policy_t"] = _nan_rows(_ACTIONS)
    fix.batch["has_sf_policy"] = _zeros()
    fix.batch["has_sf_move"] = _zeros()


def _poison_sf_eval(fix: _Fix) -> None:
    fix.outputs["sf_eval"] = torch.full((_B, 3), float("nan"), requires_grad=True)
    fix.batch["has_sf_wdl"] = _zeros()


def _poison_categorical(fix: _Fix) -> None:
    fix.batch["categorical_t"] = _nan_rows(4)
    fix.batch["has_categorical"] = _zeros()


def _poison_volatility(fix: _Fix) -> None:
    fix.batch["volatility_t"] = _nan_rows(3)
    fix.batch["has_volatility"] = _zeros()


def _poison_sf_volatility(fix: _Fix) -> None:
    fix.batch["sf_volatility_t"] = _nan_rows(3)
    fix.batch["has_sf_volatility"] = _zeros()


def _poison_moves_left(fix: _Fix) -> None:
    fix.batch["moves_left"] = torch.full((_B,), float("nan"), dtype=torch.float32)
    fix.batch["has_moves_left"] = _zeros()


def _poison_floor(fix: _Fix) -> None:
    # `__post_init__` refuses a non-finite tau, so the NaN is installed past the
    # validator on purpose: what is under test is the COMPOSITION rule, not the
    # range check. Same technique as
    # `tests/test_sf_policy_floor.py::test_a_nan_in_the_term_cannot_reach_total_at_weight_zero`.
    object.__setattr__(fix.kwargs["sf_policy_floor"], "tau", float("nan"))


@dataclass(frozen=True)
class _Case:
    """One term of the assembly, plus how to make its component NaN."""

    weight: str
    key: str
    poison: Callable[[_Fix], None]


#: ⚑ EVERY OTHER TERM STAYS WEIGHTED ON IN EVERY CASE, so a poison that leaked
#: into a second component would fail the disarm assertion loudly instead of
#: being pre-arranged out of the way. `policy_t` and `sf_p0_regret_t` are both
#: read by the SF floor as well, and were checked rather than assumed: NaN
#: reaches its `argmin`/comparisons, where every comparison is False and the
#: adaptive set is simply empty, so `sf_policy_floor` stays finite.
_CASES: tuple[_Case, ...] = (
    _Case("w_policy", "policy_ce", _poison_policy),
    _Case("w_soft", "soft_policy_ce", _poison_soft),
    _Case("w_future", "future_policy_ce", _poison_future),
    _Case("w_sf_own", "sf_own_ce", _poison_sf_own),
    _Case("w_sf_own_regret", "sf_own_regret", _poison_sf_own_regret),
    _Case("w_wdl", "wdl_ce", _poison_wdl),
    _Case("w_sf_move", "sf_move_ce", _poison_sf_move),
    _Case("w_sf_eval", "sf_eval_ce", _poison_sf_eval),
    _Case("w_categorical", "categorical_ce", _poison_categorical),
    _Case("w_volatility", "volatility", _poison_volatility),
    _Case("w_sf_volatility", "sf_volatility", _poison_sf_volatility),
    _Case("w_moves_left", "moves_left", _poison_moves_left),
    _Case("sf_policy_floor", "sf_policy_floor", _poison_floor),
)


def _kwargs(**overrides: float) -> Kwargs:
    """`_ALL_ON` with named weights overridden; `sf_policy_floor` by its `w`."""
    floor_w = overrides.pop("sf_policy_floor", _FLOOR_ON)
    weights: dict[str, object] = {**_ALL_ON, **overrides}
    weights["sf_policy_floor"] = SfPolicyFloorParams(w=float(floor_w))
    return weights


def _weight_of(kwargs: Kwargs, name: str) -> float:
    if name == "sf_policy_floor":
        floor = kwargs["sf_policy_floor"]
        assert isinstance(floor, SfPolicyFloorParams)
        return float(floor.w)
    return float(kwargs[name])  # pyright: ignore[reportArgumentType]


def _fold(
    losses: dict[str, torch.Tensor], kwargs: Kwargs, *, skip_zero: bool,
) -> torch.Tensor:
    """`total`, re-implemented from the returned components.

    ``skip_zero=False`` is the PARENT expression this PR replaced — every term
    multiplied in, nothing skipped — so a parity assertion against it is a
    comparison with the objective as it was before the guard existed. The fold
    is left-associated in `_ASSEMBLY` order and stays in the components' own
    float32, which makes the comparison EXACT rather than approximate.
    """
    total: torch.Tensor | None = None
    for name, key in _ASSEMBLY:
        w = _weight_of(kwargs, name)
        if skip_zero and w == 0.0:
            continue
        term = w * losses[key].detach()
        total = term if total is None else total + term
    assert total is not None, "the fold found no term at all"
    return total


def _case_id(case: _Case) -> str:
    return case.weight


def _run(
    case: _Case, weight: float,
) -> tuple[dict[str, torch.Tensor], Kwargs, Outputs]:
    """Poison `case`'s component, set its weight, and evaluate the loss.

    The `outputs` dict is returned as well because it owns the tensors the graph
    is built on — a fresh fixture would have no gradients to inspect.
    """
    fix = _Fix(_outputs(), _batch(), _kwargs(**{case.weight: weight}))
    case.poison(fix)
    losses = compute_loss(fix.outputs, fix.batch, **fix.kwargs)  # pyright: ignore[reportArgumentType]
    return losses, fix.kwargs, fix.outputs


# ── the fixture itself ───────────────────────────────────────────────


def test_the_unpoisoned_fixture_makes_every_term_live_and_finite() -> None:
    """NON-VACUITY OF THE WHOLE FILE.

    Every assertion below is about one term of a thirteen-term sum. A fixture
    that left a term structurally zero would make its case pass while proving
    nothing, so the components are required to be finite AND distinct from each
    other here, before any of them is poisoned.
    """
    kwargs = _kwargs()
    losses = compute_loss(_outputs(), _batch(), **kwargs)  # pyright: ignore[reportArgumentType]
    keys = [key for _, key in _ASSEMBLY]
    values = [float(losses[key].detach()) for key in keys]
    labelled = dict(zip(keys, values, strict=True))
    assert all(math.isfinite(v) for v in values), labelled
    assert all(v != 0.0 for v in values), "a structurally zero term proves nothing"
    # ...and PAIRWISE DISTINCT, which the two lines above do not give: thirteen
    # terms that happened to share a value would let a fold that added the wrong
    # one agree with the right answer by arithmetic accident.
    assert len(set(values)) == len(values), labelled
    assert float(losses["total"].detach()) == float(_fold(losses, kwargs, skip_zero=False))


# ── (1) a zero weight disarms a NaN term ─────────────────────────────


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_a_zero_weight_disarms_a_nan_term(case: _Case) -> None:
    """The fix: at weight 0.0 the NaN component is not in `total` at all."""
    losses, kwargs, _ = _run(case, 0.0)

    assert math.isnan(float(losses[case.key].detach())), (
        f"setup: {case.key} was expected to be NaN and is not"
    )
    total = float(losses["total"].detach())
    assert math.isfinite(total), f"{case.weight}=0.0 still admitted a NaN into total"
    # ...and it is the total of the OTHER terms, not merely some finite number:
    # a guard that skipped the wrong term would land here.
    assert total == float(_fold(losses, kwargs, skip_zero=True))


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_the_poison_is_real_at_a_non_zero_weight(case: _Case) -> None:
    """⚑ NON-VACUITY, PER CASE. The same poison at weight 0.5 MUST reach `total`.

    Without this, a poison that quietly failed to poison — a mask that silences
    the term for an unrelated reason, a target the loss never reads — would make
    the disarm assertion above pass for exactly the wrong reason.
    """
    losses, _, _ = _run(case, 0.5)

    assert math.isnan(float(losses[case.key].detach())), "setup: term is not NaN"
    assert math.isnan(float(losses["total"].detach())), (
        f"{case.weight}=0.5 did not carry its NaN term into total, so the "
        "zero-weight case proves nothing"
    )


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_a_disarmed_nan_term_reaches_no_gradient(case: _Case) -> None:
    """A finite `total` is not the claim — `total` is what every gradient
    comes from, so the NaN must be absent from the BACKWARD pass as well.

    A term dropped from the sum is dropped from the graph, so no head can
    receive a NaN through it. Checked on the heads, which is where a poisoned
    term would surface.
    """
    losses, _, outputs = _run(case, 0.0)
    losses["total"].backward()

    graded = [
        name for name, t in outputs.items() if t.grad is not None
    ]
    assert graded, "setup: backward populated no gradient at all"
    for name in graded:
        grad = outputs[name].grad
        assert grad is not None
        assert torch.isfinite(grad).all(), f"{name} received a non-finite gradient"


# ── (2) non-zero weights keep the parent's arithmetic ────────────────


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_a_non_zero_weight_matches_the_unguarded_parent_sum(case: _Case) -> None:
    """Bit-exact parity with `w * term` summed unconditionally.

    The guard may only remove terms whose weight is exactly zero; for every
    other value the objective has to be the one the flat expression produced,
    down to the last bit of float32.
    """
    kwargs = _kwargs(**{case.weight: 0.5})
    losses = compute_loss(_outputs(), _batch(), **kwargs)  # pyright: ignore[reportArgumentType]

    assert math.isfinite(float(losses[case.key].detach())), "setup: term is NaN"
    assert float(losses["total"].detach()) == float(
        _fold(losses, kwargs, skip_zero=False)
    )


def test_the_default_production_weights_match_the_unguarded_parent_sum() -> None:
    """The same parity on `compute_loss`'s own defaults, where several are 0.0.

    `w_sf_own`, `w_sf_own_regret` and the floor default to 0.0, so this is the
    case where guarded and unguarded arithmetic could actually diverge — and
    they may not, because `0.0 * finite` is `0.0`.
    """
    outputs, batch = _outputs(), _batch()
    losses = compute_loss(outputs, batch)
    defaults: Kwargs = {
        "w_policy": 1.0, "w_soft": 0.5, "w_future": 0.15, "w_sf_own": 0.0,
        "w_sf_own_regret": 0.0, "w_wdl": 1.0, "w_sf_move": 0.15,
        "w_sf_eval": 0.15, "w_categorical": 0.10, "w_volatility": 0.05,
        "w_sf_volatility": 0.05, "w_moves_left": 0.02,
        "sf_policy_floor": SfPolicyFloorParams(),
    }
    assert float(losses["total"].detach()) == float(
        _fold(losses, defaults, skip_zero=False)
    )


def test_a_negative_weight_is_still_a_term() -> None:
    """⚑ THE GUARD IS `!= 0.0`, NOT `> 0.0`.

    `eval_ruler.active_loss_terms` reports a negative plain multiplier as an
    ACTIVE term ("sign-flipped, but present"). If the assembly dropped it, the
    holdout ruler would hash a term set the objective does not have.
    """
    kwargs = _kwargs(w_categorical=-0.25)
    losses = compute_loss(_outputs(), _batch(), **kwargs)  # pyright: ignore[reportArgumentType]

    without = _fold(losses, {**kwargs, "w_categorical": 0.0}, skip_zero=True)
    assert float(losses["total"].detach()) != float(without)
    assert float(losses["total"].detach()) == float(
        _fold(losses, kwargs, skip_zero=False)
    )


# ── regimes the flat expression could not survive ────────────────────


def test_an_empty_net_mask_over_a_nan_value_head_is_disarmed_too() -> None:
    """The value head's empty-denominator route, which `_poison_wdl` cannot take.

    `net_mask` is the value head's row mask and every other head's as well, so
    this is the whole-batch version: no row is a network turn, every masked mean
    divides by the `clamp_min(1.0)` floor, and the NaN survives the empty mask
    through `0.0 * nan` in the NUMERATOR.
    """
    outputs, batch = _outputs(), _batch()
    outputs["wdl"] = torch.full((_B, 3), float("nan"), requires_grad=True)
    batch["is_network_turn"] = _zeros()

    kwargs = _kwargs(w_wdl=0.0)
    losses = compute_loss(outputs, batch, **kwargs)  # pyright: ignore[reportArgumentType]
    assert math.isnan(float(losses["wdl_ce"].detach())), "setup: value term is not NaN"
    assert math.isfinite(float(losses["total"].detach()))


def test_a_gen0_shaped_batch_with_every_sf_weight_off_is_finite() -> None:
    """The motivating regime, end to end.

    Gen-0 shards carry NO SF fields, and the AZ-purity arm zeroes every SF loss
    weight. Here the fields are present but NaN and every mask is empty, which
    is the harsher version of the same shape: nothing SF-flavoured may reach
    `total`, and what remains is exactly the AZ objective.
    """
    outputs, batch = _outputs(), _batch()
    for key, width in (
        ("sf_policy_t", _ACTIONS), ("sf_p0_policy_t", _ACTIONS),
        ("sf_p0_regret_t", _ACTIONS), ("sf_volatility_t", 3),
    ):
        batch[key] = _nan_rows(width)
    outputs["sf_eval"] = torch.full((_B, 3), float("nan"), requires_grad=True)
    for flag in (
        "has_sf_policy", "has_sf_move", "has_sf_p0", "has_sf_p0_regret",
        "has_sf_wdl", "has_sf_volatility",
    ):
        batch[flag] = _zeros()

    kwargs = _kwargs(
        w_sf_own=0.0, w_sf_own_regret=0.0, w_sf_move=0.0, w_sf_eval=0.0,
        w_sf_volatility=0.0, sf_policy_floor=0.0,
    )
    losses = compute_loss(outputs, batch, **kwargs)  # pyright: ignore[reportArgumentType]

    for key in (
        "sf_own_ce", "sf_own_regret", "sf_move_ce", "sf_eval_ce", "sf_volatility",
    ):
        assert math.isnan(float(losses[key].detach())), f"setup: {key} is not NaN"
    total = losses["total"]
    assert math.isfinite(float(total.detach()))
    assert float(total.detach()) == float(_fold(losses, kwargs, skip_zero=True))

    # And the surviving objective still trains: the NaN reaches no gradient.
    total.backward()
    for name in ("policy", "policy_soft", "policy_future", "wdl"):
        grad = outputs[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name


def test_absent_sf_fields_leave_the_value_blend_finite_at_a_live_frac() -> None:
    """⚑ THE CLAIM THE AZ-PURITY LANE ACTUALLY RESTS ON, PINNED.

    The value BLEND is the one weighted sum in `compute_loss` that this PR's
    guard does not cover, and deliberately so: `sf_wdl_frac * sf_component`
    builds the TARGET (which is `.detach()`ed before the CE), not `total`. What
    protects it in the gen-0 regime is not the weight — it is FIELD ABSENCE.
    With `sf_wdl` missing, `sf_component` IS `blend_fallback_target`, so the
    target stays finite whatever `sf_wdl_frac` happens to be.

    ⚑ NO OTHER TEST IN THIS FILE REACHES THE BLEND AT A NON-ZERO FRAC. The
    sibling gen-0 test leaves `sf_wdl_frac` at its 0.0 default, so it would pass
    even if the fallback did not exist. This one runs the blend live: `sf_wdl` /
    `search_wdl` both weighted in, `w_wdl` on, every SF LOSS weight left on too,
    and every SF field genuinely popped out of the batch the way a gen-0 shard
    delivers it.

    ⚑ IT PINS ABSENCE, AND ONLY ABSENCE. A `sf_wdl` tensor that is PRESENT and
    NaN still poisons `target` at any frac, because the per-row mask arithmetic
    (`sf_effective_b * sf_wdl_probs`) is `0.0 * nan` all over again. That is a
    separate, still-unguarded hazard on the target-construction path and is
    tracked as a follow-up; nothing here claims otherwise.
    """
    outputs, batch = _outputs(), _batch()
    for key in (
        "sf_wdl", "has_sf_wdl", "sf_policy_t", "has_sf_policy", "has_sf_move",
        "sf_p0_policy_t", "has_sf_p0", "sf_p0_regret_t", "has_sf_p0_regret",
        "sf_volatility_t", "has_sf_volatility",
    ):
        del batch[key]
    # The search half of the blend stays present, so `search_wdl_frac` is a live
    # component rather than a second silent fallback.
    batch["search_wdl"] = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])
    batch["has_search_wdl"] = torch.ones((_B,), dtype=torch.float32)

    kwargs = _kwargs()
    losses = compute_loss(
        outputs, batch, sf_wdl_frac=0.5, search_wdl_frac=0.2,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )

    # The SF-fed terms are absent-target zeros, not NaN...
    for key in (
        "sf_own_ce", "sf_own_regret", "sf_move_ce", "sf_eval_ce",
        "sf_volatility", "sf_policy_floor",
    ):
        assert float(losses[key].detach()) == 0.0, key
    # ...and the VALUE head, which is where a poisoned blend would surface,
    # is finite and genuinely trained.
    assert math.isfinite(float(losses["wdl_ce"].detach()))
    total = losses["total"]
    assert math.isfinite(float(total.detach()))
    assert float(total.detach()) == float(_fold(losses, kwargs, skip_zero=True))

    total.backward()
    grad = outputs["wdl"].grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() > 0, "the blend produced no value gradient"


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_negative_zero_is_off_as_well(case: _Case) -> None:
    """⚑ `-0.0` COMPARES EQUAL TO `0.0`, so it must disarm the term too.

    Both the guard's comment and the PR claim this; a claim nothing executes is
    the shape of defect this file exists to catch. `math.copysign` re-reads the
    sign off the weight the loss was actually handed, so the case cannot decay
    into a duplicate of the `0.0` one if a `float()` somewhere normalises it.
    """
    losses, kwargs, _ = _run(case, -0.0)

    w = _weight_of(kwargs, case.weight)
    assert math.copysign(1.0, w) == -1.0, "setup: the weight is no longer -0.0"
    assert math.isnan(float(losses[case.key].detach())), "setup: term is not NaN"
    assert math.isfinite(float(losses["total"].detach()))
    assert float(losses["total"].detach()) == float(_fold(losses, kwargs, skip_zero=True))


def test_every_weight_zero_gives_a_finite_graph_free_total() -> None:
    """The empty objective is a finite constant that carries NO gradient path.

    ⚑ AND `backward()` ON IT RAISES AT THE TRAINER, WHICH IS THE INTENDED
    BEHAVIOUR RATHER THAN AN OVERSIGHT. `torch.zeros_like` has no `grad_fn`, so
    the one production caller — `losses["total"] / accum_steps` then
    `loss.backward()` in `Trainer._run_optimizer_step` — fails LOUDLY on the
    first step instead of quietly stepping the optimizer on all-zero gradients,
    which is what the flat expression did: `0.0 * m_policy` kept a graph the
    objective no longer had. Fail-loud is the right direction for "this config
    asks for no objective at all", and it is unreachable from any real config —
    production runs `w_policy: 1.0` and `w_wdl: 1.0`.

    The exception is asserted here rather than described, so the trade is a
    pinned fact for whoever reads it next and not a claim in a comment.
    """
    kwargs = _kwargs(sf_policy_floor=0.0, **dict.fromkeys(_ALL_ON, 0.0))
    losses = compute_loss(_outputs(), _batch(), **kwargs)  # pyright: ignore[reportArgumentType]
    total = losses["total"]
    assert float(total.detach()) == 0.0
    assert total.shape == ()
    assert total.requires_grad is False
    with pytest.raises(RuntimeError, match="does not require grad"):
        total.backward()


def test_the_floor_params_default_is_the_same_objective_as_none() -> None:
    """`sf_policy_floor=None` and the all-default object stay interchangeable.

    Documented in `compute_loss`'s docstring and unchanged by the guard: both
    resolve to weight 0.0, so neither puts the term in `total`.
    """
    a = compute_loss(_outputs(), _batch(), **_kwargs(sf_policy_floor=0.0))  # pyright: ignore[reportArgumentType]
    b = compute_loss(
        _outputs(), _batch(), sf_policy_floor=None,
        **_ALL_ON,  # pyright: ignore[reportArgumentType]
    )
    assert float(a["total"].detach()) == float(b["total"].detach())
