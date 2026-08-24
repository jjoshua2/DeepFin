"""A PRESENT-BUT-NaN value label must not poison the target through a zero mask.

⚑ FIELD ABSENCE WAS GUARDED; FIELD PRESENCE WITH A NaN ROW WAS NOT. `compute_loss`
substitutes `blend_fallback_target` wholesale when `sf_wdl` / `search_wdl` are
MISSING (the gen-0 shape), which is what
`tests/test_losses_zero_weight_disarm.py::test_absent_sf_fields_leave_the_value_blend_finite_at_a_live_frac`
pins. It did nothing at all for a tensor that is PRESENT and carries a NaN row:
the per-row arithmetic

    sf_effective_b * sf_wdl_probs + (1.0 - sf_effective_b) * blend_fallback_target

is ``0.0 * nan`` for a row whose own ``has_sf_wdl`` is 0, so a row the blend does
not want contributes NaN anyway -- at ANY frac, including 0.0, and through
``w_wdl = 1.0``, which no zero weight could ever disarm.
`_normalize_sf_wdl_probs` is not a defence: `clamp_min` and the renormalise both
PROPAGATE NaN (a clamp is not a validator), so the sanitiser here is applied to
the NORMALIZED tensor the blend actually consumes.

⚑ THE TWO REGIMES ARE NOT THE SAME EVENT.
  - mask 0: the row does not claim the label, so its content is irrelevant and
    it must take the fallback EXACTLY. Pinned by an A/B where the only
    difference is what those unclaimed rows contain -- NaN vs an arbitrary
    finite distribution -- and the value loss must be bit-identical.
  - mask non-zero: the row CLAIMS the label and the label is NaN. Dirty shard
    data, not a training regime; `compute_loss` raises. Silently substituting
    the fallback there would train the value head on the game outcome while the
    shard asserts an SF opinion -- an accepted-then-ignored value, which is this
    repo's signature defect.

⚑ NON-VACUITY IS CHECKED IN-FILE. Every disarm assertion is paired with the same
batch run through an UNGUARDED `_finite_blend_component` (the parent expression,
reproduced verbatim in `_unguarded_blend_component`), which must produce a NaN.
A poison that quietly failed to poison would otherwise pass for the wrong
reason. The source-level mutant table is in the PR.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from chess_anti_engine.train import losses as losses_mod
from chess_anti_engine.train import trainer as trainer_mod
from chess_anti_engine.train.losses import compute_loss
from chess_anti_engine.train.trainer import Trainer

Outputs = dict[str, torch.Tensor]
Batch = dict[str, torch.Tensor]

_B = 4
_ACTIONS = 3

#: Rows 0-1 CLAIM their SF/search label; rows 2-3 do not. Every blend test below
#: turns on this split, so "the mask is 0" and "the row is NaN" are separable.
_CLAIMED = (1.0, 1.0, 0.0, 0.0)

_NAN = float("nan")

#: What an unclaimed row carries. The A/B is the whole point: with the guard in
#: place these two must be indistinguishable in the value loss, because an
#: unclaimed row's label is not part of the objective either way.
_UNCLAIMED_NAN = [_NAN, _NAN, _NAN]
_UNCLAIMED_FINITE = [0.9, 0.05, 0.05]

#: Production weights for the two knobs this file leans on
#: (`configs/pbt2_small.yaml`, 2026-08-23): the aux `sf_eval` head is OFF, and
#: the confidence damping that carries `sf_wdl` into that head's ROW MASK is on.
#: Both are stated rather than defaulted, because the defaults differ
#: (`w_sf_eval=0.15`, `sf_wdl_conf_power=0.0`) and the difference decides which
#: term a NaN reaches.
_PROD_W_SF_EVAL = 0.0
_PROD_SF_WDL_CONF_POWER = 1.0

#: ⚑ `main`'s yaml is NOT the live production yaml -- they differ on this exact
#: key (`configs/pbt2_small.yaml:863` on `main` vs the running trial's tree,
#: which reads 0.0). Both regimes are exercised, and the difference decides
#: whether a NaN aux term is disarmed-and-counted or reaches `total` and trips
#: the non-finite-gradient guard. Neither is silent, which is the point.
_MAIN_YAML_W_SF_EVAL = 0.10

#: A live blend, in production proportions.
_SF_FRAC = 0.69
_SEARCH_FRAC = 0.31


def _outputs() -> Outputs:
    return {
        "policy": torch.tensor(
            [[2.0, -1.0, 0.0], [0.5, 1.5, -1.0], [0.0, 0.3, 1.0], [-0.4, 0.2, 0.6]],
            requires_grad=True,
        ),
        "wdl": torch.tensor(
            [[0.1, -0.2, 0.3], [0.4, 0.0, -0.1], [-0.3, 0.2, 0.1], [0.2, 0.2, -0.4]],
            requires_grad=True,
        ),
        "categorical": torch.tensor(
            [
                [0.3, -0.1, 0.2, 0.0], [0.1, 0.2, -0.2, 0.4],
                [0.0, 0.1, 0.3, -0.1], [0.2, -0.3, 0.1, 0.2],
            ],
            requires_grad=True,
        ),
    }


def _base_batch() -> Batch:
    """Policy + value live, no SF/search value fields yet."""
    ones = torch.ones((_B,), dtype=torch.float32)
    return {
        "x": torch.zeros((_B, 1, 1, 1), dtype=torch.float32),
        "policy_t": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
        ),
        "wdl_t": torch.tensor([0, 2, 1, 0], dtype=torch.long),
        "has_policy": ones.clone(),
        "is_network_turn": ones.clone(),
        "legal_mask": torch.ones((_B, _ACTIONS)),
        "has_legal_mask": ones.clone(),
    }


def _claimed_mask(claimed: tuple[float, ...] = _CLAIMED) -> torch.Tensor:
    return torch.tensor(claimed, dtype=torch.float32)


def _labels(unclaimed_row: list[float]) -> torch.Tensor:
    """Finite distributions on rows 0-1, ``unclaimed_row`` on rows 2-3."""
    return torch.tensor(
        [[0.2, 0.5, 0.3], [0.6, 0.2, 0.2], list(unclaimed_row), list(unclaimed_row)],
    )


def _sf_batch(
    unclaimed_row: list[float], *, claimed: tuple[float, ...] = _CLAIMED,
) -> Batch:
    batch = _base_batch()
    batch["sf_wdl"] = _labels(unclaimed_row)
    batch["has_sf_wdl"] = _claimed_mask(claimed)
    return batch


def _search_batch(
    unclaimed_row: list[float], *, claimed: tuple[float, ...] = _CLAIMED,
) -> Batch:
    batch = _base_batch()
    batch["search_wdl"] = _labels(unclaimed_row)
    batch["has_search_wdl"] = _claimed_mask(claimed)
    return batch


#: ⚑ WITHOUT DAMPENING THE BLEND WEIGHT IS ONLY EVER 0.0 OR 1.0, so
#: ``w * probs + (1 - w) * fallback`` collapses to one operand and any
#: re-association of it is trivially exact. `sf_search_dampen_sf_*` is the only
#: thing that puts a FRACTION in `sf_effective`, and a fraction is where a
#: rewritten blend can drift. Every parity claim in this file is therefore made
#: over `_dampened_batch`, not over a plain 0/1 batch.
_DAMPEN_SF_LOW = 0.3
_DAMPEN_SF_HIGH = 0.1


def _dampened_batch() -> Batch:
    """Both halves present and CLAIMED, with a directional disagreement per row.

    Row 0 is `sf_low` (SF says STM losing, search says winning) and row 1 is
    `sf_high`; rows 2-3 agree. With the dampen knobs on, `sf_effective` reads
    (0.7, 0.9, 1.0, 1.0) -- three distinct fractional weights in one batch.
    """
    batch = _base_batch()
    batch["sf_wdl"] = torch.tensor(
        [[0.05, 0.05, 0.9], [0.9, 0.05, 0.05], [0.6, 0.2, 0.2], [0.2, 0.3, 0.5]],
    )
    batch["has_sf_wdl"] = _claimed_mask((1.0, 1.0, 1.0, 1.0))
    batch["search_wdl"] = torch.tensor(
        [[0.9, 0.05, 0.05], [0.05, 0.05, 0.9], [0.5, 0.3, 0.2], [0.1, 0.3, 0.6]],
    )
    batch["has_search_wdl"] = _claimed_mask((1.0, 1.0, 0.0, 1.0))
    return batch


def _kwargs(**overrides: Any) -> dict[str, Any]:
    """A live blend at production weights, with the SF frac genuinely non-zero."""
    kwargs: dict[str, Any] = {
        "sf_wdl_frac": _SF_FRAC,
        "search_wdl_frac": _SEARCH_FRAC,
        "w_sf_eval": _PROD_W_SF_EVAL,
    }
    kwargs.update(overrides)
    return kwargs


def _run(batch: Batch, **overrides: Any) -> dict[str, torch.Tensor]:
    return compute_loss(_outputs(), batch, **_kwargs(**overrides))


def _f(losses: dict[str, torch.Tensor], key: str) -> float:
    return float(losses[key].detach())


def _unguarded_blend_component(
    probs: torch.Tensor | None,
    *,
    raw: torch.Tensor | None = None,
    weight: torch.Tensor,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """`main`'s blend component, character for character, in the new signature.

    ⚑ THIS IS THE PARENT EXPRESSION, NOT A PARAPHRASE. It is what the guarded
    version must agree with BIT-FOR-BIT on every finite batch, and what must
    still produce a NaN on a poisoned one -- otherwise the poison is not a
    poison and every disarm assertion in this file passes for the wrong reason.
    """
    del raw
    if probs is None:
        return fallback, None, None
    return weight * probs + (1.0 - weight) * fallback, None, None


@pytest.fixture
def unguarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run `compute_loss` with the guard removed, as `main` had it."""
    monkeypatch.setattr(
        losses_mod, "_finite_blend_component", _unguarded_blend_component,
    )


# --------------------------------------------------------------------------
# 1. An UNCLAIMED row's label content is irrelevant -- including when it is NaN.
# --------------------------------------------------------------------------


def test_an_unclaimed_nan_sf_label_leaves_the_value_loss_bit_identical() -> None:
    """The A/B that IS the fix: rows 2-3 have ``has_sf_wdl == 0``, so what those
    rows carry cannot move the objective. NaN vs an arbitrary finite
    distribution must give the same value loss to the last bit."""
    nan_side = _run(_sf_batch(_UNCLAIMED_NAN))
    finite_side = _run(_sf_batch(_UNCLAIMED_FINITE))

    assert math.isfinite(_f(nan_side, "wdl_ce"))
    assert math.isfinite(_f(nan_side, "total"))
    assert _f(nan_side, "wdl_ce") == _f(finite_side, "wdl_ce")
    assert _f(nan_side, "total") == _f(finite_side, "total")


def test_an_unclaimed_nan_search_label_leaves_the_value_loss_bit_identical() -> None:
    """The search half of the blend, same rule, its own mask."""
    nan_side = _run(_search_batch(_UNCLAIMED_NAN))
    finite_side = _run(_search_batch(_UNCLAIMED_FINITE))

    assert math.isfinite(_f(nan_side, "wdl_ce"))
    assert math.isfinite(_f(nan_side, "total"))
    assert _f(nan_side, "wdl_ce") == _f(finite_side, "wdl_ce")
    assert _f(nan_side, "total") == _f(finite_side, "total")


def test_an_all_unclaimed_nan_field_is_exactly_field_absence() -> None:
    """⚑ "USES THE FALLBACK EXACTLY", stated as an equality rather than as a
    finiteness check.

    With every row unclaimed, the SF component must be `blend_fallback_target`
    on every row -- which is precisely what the field being ABSENT produces. So
    the present-and-all-NaN batch and the field-absent batch have to agree bit
    for bit, not merely both be finite.

    ⚑ WHAT THIS TEST CANNOT SEE, MEASURED RATHER THAN ASSUMED. It does NOT
    distinguish "the fallback" from "zeros": `soft_cross_entropy` RENORMALISES
    the target row, and with `sf_wdl`/`search_wdl` both absent every surviving
    contribution to this batch's target is proportional to the same `game_oh`,
    so dropping one of them only RESCALES the row and the CE is unchanged. A
    `nan_to_num` substitution survives here (mutant M3, run). The test that
    separates them is
    `test_the_unclaimed_row_takes_the_fallback_and_not_zeros`, which puts a
    CLAIMED search label on the row so the two substitutions differ in MIX and
    not only in scale.
    """
    present = _run(_sf_batch(_UNCLAIMED_NAN, claimed=(0.0, 0.0, 0.0, 0.0)))
    absent = _run(_base_batch())

    assert _f(present, "wdl_ce") == _f(absent, "wdl_ce")
    assert _f(present, "total") == _f(absent, "total")
    assert math.isfinite(_f(present, "total"))


def _mixed_batch(unclaimed_row: list[float]) -> Batch:
    """An unclaimed SF row whose SEARCH label is claimed and points elsewhere.

    Rows 2-3 have ``has_sf_wdl == 0`` but ``has_search_wdl == 1``, and the search
    distributions are deliberately NOT proportional to their one-hot outcome. So
    what the SF component contributes on those rows changes the target's MIX,
    not just its scale, and survives `soft_cross_entropy`'s renormalise.
    """
    batch = _sf_batch(unclaimed_row)
    batch["search_wdl"] = torch.tensor(
        [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.15, 0.7, 0.15], [0.25, 0.35, 0.4]],
    )
    batch["has_search_wdl"] = _claimed_mask((1.0, 1.0, 1.0, 1.0))
    return batch


def test_the_unclaimed_row_takes_the_fallback_and_not_zeros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ THE SUBSTITUTION HAS TO BE THE FALLBACK, NOT MERELY SOMETHING FINITE.

    `nan_to_num` also removes the NaN, and on a batch where the fallback is the
    row's only other contributor it is indistinguishable (see the test above).
    Here the unclaimed rows carry a CLAIMED search label pointing away from the
    game outcome, so zeroing the SF share shifts the target's MIX and the CE
    moves. Both facts are asserted: the guard is invisible (NaN vs finite
    content agree bit for bit) AND a zeros-substitution is visible.
    """
    nan_side = _run(_mixed_batch(_UNCLAIMED_NAN))
    finite_side = _run(_mixed_batch(_UNCLAIMED_FINITE))
    assert math.isfinite(_f(nan_side, "wdl_ce"))
    assert _f(nan_side, "wdl_ce") == _f(finite_side, "wdl_ce")

    def _zeros_instead(
        probs: torch.Tensor | None, *, raw: torch.Tensor | None = None,
        weight: torch.Tensor, fallback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        del raw
        if probs is None:
            return fallback, None, None
        blended = weight * probs + (1.0 - weight) * fallback
        return torch.nan_to_num(blended), torch.zeros(()), torch.zeros(())

    monkeypatch.setattr(losses_mod, "_finite_blend_component", _zeros_instead)
    zeroed = _run(_mixed_batch(_UNCLAIMED_NAN))
    assert math.isfinite(_f(zeroed, "wdl_ce")), "nan_to_num is finite -- that is the trap"
    assert _f(zeroed, "wdl_ce") != _f(nan_side, "wdl_ce")


def test_the_guard_holds_at_frac_zero() -> None:
    """⚑ ``0.0 * nan`` IS ``nan``. A zero blend frac never protected the target,
    which is why the hazard reaches `total` through ``w_wdl = 1.0`` at every
    frac. Run at the frac where the term contributes nothing at all."""
    losses = _run(
        _sf_batch(_UNCLAIMED_NAN), sf_wdl_frac=0.0, search_wdl_frac=0.0,
    )
    assert math.isfinite(_f(losses, "wdl_ce"))
    assert math.isfinite(_f(losses, "total"))


def test_the_poison_is_real_without_the_guard(unguarded: None) -> None:
    """NON-VACUITY. The same unclaimed-NaN batch, through the parent
    expression, must be NaN -- at a live frac AND at frac 0.0."""
    del unguarded
    live = _run(_sf_batch(_UNCLAIMED_NAN))
    assert math.isnan(_f(live, "wdl_ce"))
    assert math.isnan(_f(live, "total"))

    at_zero = _run(
        _sf_batch(_UNCLAIMED_NAN), sf_wdl_frac=0.0, search_wdl_frac=0.0,
    )
    assert math.isnan(_f(at_zero, "wdl_ce")), "0.0 * nan must still be nan"
    assert math.isnan(_f(at_zero, "total"))


def test_the_search_poison_is_real_without_the_guard(unguarded: None) -> None:
    """NON-VACUITY for the search half."""
    del unguarded
    losses = _run(_search_batch(_UNCLAIMED_NAN))
    assert math.isnan(_f(losses, "wdl_ce"))
    assert math.isnan(_f(losses, "total"))


def test_the_value_gradient_survives_an_unclaimed_nan_row() -> None:
    """`total` being finite is not the claim -- every gradient built from it is.
    The blend feeds a `.detach()`ed target, so a NaN there reaches the weights
    only through the CE, which is exactly the path checked here."""
    outputs = _outputs()
    losses = compute_loss(outputs, _sf_batch(_UNCLAIMED_NAN), **_kwargs())
    losses["total"].backward()

    grad = outputs["wdl"].grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() > 0, "the blend produced no value gradient"


# --------------------------------------------------------------------------
# 2. A CLAIMED NaN row is dirty data and must fail loudly.
# --------------------------------------------------------------------------


def test_a_claimed_nan_sf_label_raises() -> None:
    """mask 1 + NaN = the shard asserts an SF opinion it does not have. The
    message has to name the FIELD and the ROW COUNT, or the operator is left
    diffing shards to find out which of the two labels broke."""
    batch = _sf_batch(_UNCLAIMED_FINITE, claimed=(1.0, 1.0, 1.0, 0.0))
    batch["sf_wdl"] = torch.tensor(
        [[0.2, 0.5, 0.3], [_NAN, _NAN, _NAN], [_NAN, 0.4, 0.6], [0.1, 0.1, 0.8]],
    )
    with pytest.raises(ValueError, match=r"sf_wdl: 2 row\(s\)"):
        _run(batch)


def test_a_claimed_nan_search_label_raises() -> None:
    batch = _search_batch(_UNCLAIMED_FINITE, claimed=(1.0, 0.0, 0.0, 0.0))
    batch["search_wdl"] = torch.tensor(
        [[_NAN, _NAN, _NAN], [0.3, 0.3, 0.4], [0.1, 0.1, 0.8], [0.2, 0.2, 0.6]],
    )
    with pytest.raises(ValueError, match=r"search_wdl: 1 row\(s\)"):
        _run(batch)


def test_the_raise_names_every_offending_field() -> None:
    """Both halves dirty in one batch. One field named and the other silently
    dropped would send the operator to the wrong shard column."""
    batch = _sf_batch(_UNCLAIMED_FINITE, claimed=(1.0, 1.0, 1.0, 1.0))
    batch["sf_wdl"] = torch.tensor(
        [[_NAN, _NAN, _NAN], [0.6, 0.2, 0.2], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]],
    )
    batch["search_wdl"] = torch.tensor(
        [[0.3, 0.3, 0.4], [_NAN, _NAN, _NAN], [_NAN, _NAN, _NAN], [0.2, 0.2, 0.6]],
    )
    batch["has_search_wdl"] = _claimed_mask((1.0, 1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match=r"sf_wdl: 1 row\(s\)") as excinfo:
        _run(batch)
    message = str(excinfo.value)
    assert "sf_wdl: 1 row(s)" in message
    assert "search_wdl: 2 row(s)" in message


def test_the_raise_fires_at_frac_zero_too() -> None:
    """The label is dirty whatever the blend does with it. Keying the check on
    the frac would make a shard defect appear and disappear with a PID
    excursion -- `sf_wdl_frac` is recomputed every iteration."""
    batch = _sf_batch(_UNCLAIMED_FINITE, claimed=(1.0, 0.0, 0.0, 0.0))
    batch["sf_wdl"] = torch.tensor(
        [[_NAN, _NAN, _NAN], [0.6, 0.2, 0.2], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]],
    )
    with pytest.raises(ValueError, match=r"sf_wdl: 1 row\(s\)"):
        _run(batch, sf_wdl_frac=0.0, search_wdl_frac=0.0)


def test_a_dampened_row_that_reaches_weight_zero_is_not_an_error() -> None:
    """⚑ THE PREDICATE IS THE ROW'S OWN EFFECTIVE MULTIPLIER, NOT `has_sf_wdl`.

    `sf_effective = sf_available * keep`, and a fully dampened disagreement row
    reaches 0.0 with `has_sf_wdl == 1`. Its label is multiplied by zero, so it
    is unclaimed by the only definition that matters to the arithmetic, and the
    fallback is the honest substitute rather than a crash.

    ⚑ WHAT THIS DOES *NOT* SHOW, and an earlier version of this docstring
    implied that it did: a NON-FINITE row can never BE the dampened one. Every
    comparison with NaN is False, so a NaN row's `dis_sf_low`/`dis_sf_high` are
    0 and its `keep` is exactly 1.0 -- "a dampened-to-zero NaN row" is
    unreachable, not merely tolerated. See
    `test_a_nan_row_can_never_be_dampened_below_a_full_weight`, which pins that
    property directly. What this test shows is the reverse pair: the corrupt row
    is NOT dampened and still raises, and a FINITE row that IS dampened to zero
    does not.
    """
    batch = _base_batch()
    batch["sf_wdl"] = torch.tensor(
        [[0.9, 0.05, 0.05], [_NAN, _NAN, _NAN], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]],
    )
    batch["has_sf_wdl"] = _claimed_mask((1.0, 1.0, 1.0, 1.0))
    batch["search_wdl"] = torch.tensor(
        [[0.05, 0.05, 0.9], [0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]],
    )
    batch["has_search_wdl"] = _claimed_mask((1.0, 1.0, 1.0, 1.0))

    # Row 1 has SF NaN, so its direction comparisons are all False and it is NOT
    # dampened -- the raise must still fire. Checked FIRST so the negative case
    # below cannot be mistaken for the guard simply never firing.
    with pytest.raises(ValueError, match=r"sf_wdl: 1 row\(s\)"):
        _run(batch, sf_search_dampen_sf_low=1.0, sf_search_dampen_sf_high=1.0)

    # Now make row 1's SF label finite and DISAGREEING, dampened all the way to
    # zero. Nothing is dirty any more, and the batch must run.
    batch["sf_wdl"] = torch.tensor(
        [[0.9, 0.05, 0.05], [0.05, 0.05, 0.9], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]],
    )
    losses = _run(batch, sf_search_dampen_sf_low=1.0, sf_search_dampen_sf_high=1.0)
    assert math.isfinite(_f(losses, "total"))


# --------------------------------------------------------------------------
# 2b. ±inf must not be laundered into a valid-looking label.
# --------------------------------------------------------------------------

_POS_INF = float("inf")
_NEG_INF = float("-inf")

#: ⚑ `-inf` IS THE DANGEROUS ONE AND IT IS THE ONE THAT LOOKED FINE.
#: `_normalize_sf_wdl_probs` opens with `clamp_min(0.0)`, so `[-inf, 0.5, 0.5]`
#: normalises to `[0.0, 0.5, 0.5]` -- an ordinary, plausible distribution. A
#: finiteness test taken on the NORMALIZED tensor sees nothing and the row
#: trains. `+inf` survives the clamp and dies at `inf / inf`, so before the raw
#: test the two infinities behaved OPPOSITELY, decided by a sign.
_NON_FINITE_LABELS: tuple[tuple[str, list[float]], ...] = (
    ("nan", [_NAN, _NAN, _NAN]),
    ("pos_inf", [_POS_INF, 0.5, 0.5]),
    ("neg_inf", [_NEG_INF, 0.5, 0.5]),
)


def _label_id(case: tuple[str, list[float]]) -> str:
    return case[0]


@pytest.mark.parametrize("case", _NON_FINITE_LABELS, ids=_label_id)
def test_every_non_finite_label_behaves_the_same_when_unclaimed(
    case: tuple[str, list[float]],
) -> None:
    """NaN, +inf and -inf are ONE case: unclaimed ⇒ fallback, and COUNTED.

    The counted half is not decoration. An unclaimed corrupt row is tolerated,
    and a guard that silently repairs its input is the accepted-then-ignored
    shape this repo keeps regrowing -- so the substitution has to leave a
    reading behind. `-inf` is the row that would otherwise arrive as
    `[0.0, 0.5, 0.5]` and be counted as nothing at all.
    """
    _, row = case
    corrupt = _run(_sf_batch(row))
    clean = _run(_sf_batch(_UNCLAIMED_FINITE))

    assert math.isfinite(_f(corrupt, "wdl_ce"))
    assert math.isfinite(_f(corrupt, "total"))
    assert _f(corrupt, "wdl_ce") == _f(clean, "wdl_ce")
    # Rows 2 and 3 are the unclaimed ones.
    assert _f(corrupt, "blend_unclaimed_nonfinite_rows") == 2.0
    assert _f(clean, "blend_unclaimed_nonfinite_rows") == 0.0


@pytest.mark.parametrize("case", _NON_FINITE_LABELS, ids=_label_id)
def test_every_non_finite_label_raises_when_claimed(
    case: tuple[str, list[float]],
) -> None:
    """The other half of "one case": claimed ⇒ ValueError, whichever it is."""
    _, row = case
    batch = _sf_batch(_UNCLAIMED_FINITE, claimed=(1.0, 0.0, 0.0, 0.0))
    batch["sf_wdl"] = torch.tensor([list(row), [0.6, 0.2, 0.2], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]])
    with pytest.raises(ValueError, match=r"sf_wdl: 1 row\(s\)"):
        _run(batch)


@pytest.mark.parametrize("case", _NON_FINITE_LABELS, ids=_label_id)
def test_every_non_finite_search_label_raises_when_claimed(
    case: tuple[str, list[float]],
) -> None:
    """The search half carries the same clamp, so it carries the same hole."""
    _, row = case
    batch = _search_batch(_UNCLAIMED_FINITE, claimed=(0.0, 1.0, 0.0, 0.0))
    batch["search_wdl"] = torch.tensor(
        [[0.3, 0.3, 0.4], list(row), [0.1, 0.1, 0.8], [0.2, 0.2, 0.6]],
    )
    with pytest.raises(ValueError, match=r"search_wdl: 1 row\(s\)"):
        _run(batch)


def test_the_clamp_hides_negative_infinity_from_the_normalized_tensor() -> None:
    """⚑ THE MECHANISM, MEASURED RATHER THAN ASSERTED IN A COMMENT.

    This is the fact the raw-tensor test exists for, and it is checked against
    `_normalize_sf_wdl_probs` itself so it cannot drift from the real clamp: a
    `-inf` entry comes out FINITE and normalized, while `+inf` comes out NaN. If
    this ever stops being true the guard's `raw` argument is load-bearing for a
    reason that no longer exists, and this test says so first.
    """
    normalized = losses_mod._normalize_sf_wdl_probs(
        torch.tensor([[_NEG_INF, 0.5, 0.5], [_POS_INF, 0.5, 0.5]]),
    )
    assert normalized is not None
    assert torch.isfinite(normalized[0]).all(), "clamp_min laundered -inf into a valid row"
    assert torch.equal(normalized[0], torch.tensor([0.0, 0.5, 0.5]))
    assert not torch.isfinite(normalized[1]).all(), "+inf was expected to die at inf/inf"


def test_the_guard_refuses_a_present_probs_with_a_missing_raw() -> None:
    """⚑ THE COUPLING IS ENFORCED, NOT ASSUMED — AND IT WAS UNTESTED.

    `raw` is where the `-inf` test has to happen, so a future call site that
    passes only the normalized tensor would silently reopen the clamp hole while
    every existing test stayed green. The function refuses instead of degrading.

    Without this test the refusal is itself the shape this PR exists to remove: a
    guard that cannot fail. Measured — the mutant that replaces the raise with
    `raw = probs` SURVIVED the whole file until this case was written, because no
    call site or test ever reached the branch.
    """
    probs = torch.tensor([[0.2, 0.5, 0.3]])
    with pytest.raises(ValueError, match="clamp_min"):
        losses_mod._finite_blend_component(
            probs, raw=None, weight=torch.tensor([[1.0]]), fallback=probs,
        )


def test_a_finite_negative_label_is_still_clamped_not_rejected() -> None:
    """⚑ THE GUARD MUST NOT WIDEN INTO A RANGE CHECK. `clamp_min(0.0)` on a
    small negative is the DOCUMENTED intent of `_normalize_sf_wdl_probs`, not a
    defect, so a claimed row of `[-0.01, 0.5, 0.5]` must still train. Only
    NON-FINITE raw values are the defect."""
    batch = _sf_batch(_UNCLAIMED_FINITE, claimed=(1.0, 1.0, 1.0, 1.0))
    batch["sf_wdl"] = torch.tensor(
        [[-0.01, 0.5, 0.5], [0.6, 0.2, 0.2], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]],
    )
    losses = _run(batch)
    assert math.isfinite(_f(losses, "total"))
    assert _f(losses, "blend_unclaimed_nonfinite_rows") == 0.0


def test_a_claimed_nan_label_raises_even_with_the_value_head_off() -> None:
    """⚑ STATED POLICY, PINNED AS A TEST: the raise is NOT gated on `w_wdl`.

    A corrupt CLAIMED value label is a data defect, and the shard is just as
    broken on an arm that has the value head weighted off. Gating on the weight
    would let the same shard pass undetected on the arm that is not looking and
    then fire later on the arm that is -- with the bad rows already banked in
    the replay window. `w_wdl = 0.0` is a real configuration (the AZ-purity
    lane), which is why this is a test and not a sentence.
    """
    batch = _sf_batch(_UNCLAIMED_FINITE, claimed=(1.0, 1.0, 0.0, 0.0))
    batch["sf_wdl"] = torch.tensor(
        [[0.2, 0.5, 0.3], [_NAN, _NAN, _NAN], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]],
    )
    with pytest.raises(ValueError, match=r"sf_wdl: 1 row\(s\)"):
        _run(batch, w_wdl=0.0)


def test_a_nan_row_can_never_be_dampened_below_a_full_weight() -> None:
    """⚑ IEEE, AND IT IS LOAD-BEARING FOR WHY A CLAIMED NaN REACHES THE RAISE.

    `keep = 1 - (dampen_low * dis_sf_low + dampen_high * dis_sf_high)`, and both
    `dis_` terms are built from `sf_sig < 0` / `sf_sig > 0`. Every comparison
    with NaN is False, so a NaN row's `dis_` terms are 0 and its `keep` is
    EXACTLY 1.0 -- at any dampening, including 1.0/1.0. "A NaN row dampened to
    zero weight" is therefore UNREACHABLE, and an earlier docstring here implied
    it was merely tolerated. The expression below is `compute_loss`'s own, and
    the end-to-end consequence is asserted underneath it so the two cannot drift
    apart silently.
    """
    sf_sig = torch.tensor([_NAN])
    sr_sig = torch.tensor([1.0])
    dis_sf_low = ((sf_sig < 0) & (sr_sig > 0)).float()
    dis_sf_high = ((sf_sig > 0) & (sr_sig < 0)).float()
    assert float(dis_sf_low) == 0.0
    assert float(dis_sf_high) == 0.0
    keep = 1.0 - (1.0 * dis_sf_low + 1.0 * dis_sf_high)
    assert float(keep) == 1.0

    # The consequence: a CLAIMED NaN row still reaches the raise at maximum
    # dampening, because the dampening cannot touch it.
    batch = _base_batch()
    batch["sf_wdl"] = torch.tensor(
        [[0.9, 0.05, 0.05], [_NAN, _NAN, _NAN], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]],
    )
    batch["has_sf_wdl"] = _claimed_mask((1.0, 1.0, 1.0, 1.0))
    batch["search_wdl"] = torch.tensor(
        [[0.05, 0.05, 0.9], [0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]],
    )
    batch["has_search_wdl"] = _claimed_mask((1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match=r"sf_wdl: 1 row\(s\)"):
        _run(batch, sf_search_dampen_sf_low=1.0, sf_search_dampen_sf_high=1.0)


# --------------------------------------------------------------------------
# 3. Bit-identity on every path that works today.
# --------------------------------------------------------------------------


#: A component input with THREE fractional weights and both endpoints, so the
#: parity comparison below is made over the case where a rewritten blend could
#: actually drift rather than over 0/1 rows where it collapses to one operand.
def _component_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.tensor([
        [0.21, 0.34, 0.45], [0.62, 0.19, 0.19], [0.08, 0.47, 0.45],
        [0.55, 0.30, 0.15], [0.33, 0.33, 0.34], [0.11, 0.24, 0.65],
    ])
    fallback = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 2, 0, 1, 2]), 3,
    ).float()
    weight = torch.tensor([[0.7], [0.9], [1.0], [0.0], [0.31], [0.69]])
    return probs, weight, fallback


def test_a_row_with_one_non_finite_entry_is_a_non_finite_row() -> None:
    """⚑ THE CONTRACT IS `.all`, AND IT CANNOT BE CHECKED END TO END.

    `_normalize_sf_wdl_probs` divides by the row SUM, and a sum containing one
    NaN is NaN, so by the time the blend sees a row it is either wholly finite
    or wholly NaN -- `.any` and `.all` are indistinguishable downstream (mutant
    M4 survives the whole end-to-end file). That equivalence is a property of
    the CALLER, not of this function, so the contract is pinned HERE: one
    non-finite entry makes the row non-finite, for the substitution and for the
    count alike.
    """
    probs = torch.tensor([[0.2, 0.5, 0.3], [_NAN, 0.4, 0.6]])
    fallback = torch.nn.functional.one_hot(torch.tensor([0, 2]), 3).float()

    unclaimed, bad, _ = losses_mod._finite_blend_component(
        probs, raw=probs, weight=torch.tensor([[0.0], [0.0]]), fallback=fallback,
    )
    assert torch.equal(unclaimed[1], fallback[1]), "the partial-NaN row must take the fallback"
    assert bad is not None
    assert float(bad) == 0.0

    _, claimed_bad, _ = losses_mod._finite_blend_component(
        probs, raw=probs, weight=torch.tensor([[1.0], [1.0]]), fallback=fallback,
    )
    assert claimed_bad is not None
    assert float(claimed_bad) == 1.0


def test_the_component_is_bit_identical_to_the_unguarded_expression() -> None:
    """⚑ THE BIT-IDENTITY CLAIM, MADE AT FULL RESOLUTION.

    `torch.equal` on the (rows x 3) component, not `==` on a reduced scalar: a
    sub-ULP difference in the target is invisible after the CE's sum and mean
    (measured -- a 1-ULP `nextafter` on the whole target leaves `wdl_ce`
    unchanged), so the end-to-end check below cannot be the load-bearing one.
    `torch.where` SELECTS a value rather than recomputing one, which is why
    every finite row is the identical float32 it was before this PR.
    """
    probs, weight, fallback = _component_inputs()
    guarded, bad, _ = losses_mod._finite_blend_component(
        probs, raw=probs, weight=weight, fallback=fallback,
    )
    parent, _, _ = _unguarded_blend_component(probs, weight=weight, fallback=fallback)

    assert torch.equal(guarded, parent)
    assert bad is not None
    assert float(bad) == 0.0


@pytest.mark.parametrize("rewrite", ["reassociated", "float64", "one_ulp"])
def test_the_component_comparison_has_bit_resolution(rewrite: str) -> None:
    """NON-VACUITY for the equality above. Each rewrite is algebraically the
    same blend (or a single ULP away from it) and must still be caught, or
    `torch.equal` is comparing a value with itself and proves nothing.

    Measured max |difference|: 5.96e-08 for both arithmetic rewrites -- far
    below anything `pytest.approx` would notice, which is the point.
    """
    probs, weight, fallback = _component_inputs()
    guarded, _, _ = losses_mod._finite_blend_component(
        probs, raw=probs, weight=weight, fallback=fallback,
    )
    parent, _, _ = _unguarded_blend_component(probs, weight=weight, fallback=fallback)
    if rewrite == "reassociated":
        other = fallback + weight * (probs - fallback)
    elif rewrite == "float64":
        other = (
            weight.double() * probs.double()
            + (1.0 - weight.double()) * fallback.double()
        ).float()
    else:
        other = torch.nextafter(parent, torch.full_like(parent, float("inf")))

    assert not torch.equal(guarded, other)
    assert torch.allclose(guarded, other, atol=1e-6)


def test_a_clean_batch_is_bit_identical_to_the_unguarded_blend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same claim END TO END: on a batch with nothing wrong in it, EVERY
    scalar `compute_loss` returns is `==` to what the parent expression
    produced -- not `approx`, and not just `total`.

    ⚑ ITS RESOLUTION IS THE CE's, NOT THE TARGET's. A sub-ULP target difference
    does not survive the reduction, so this test rules out a VISIBLE change and
    the component-level test above rules out an invisible one. Both are needed;
    neither replaces the other.
    """
    batch = _dampened_batch()
    damped = {
        "sf_search_dampen_sf_low": _DAMPEN_SF_LOW,
        "sf_search_dampen_sf_high": _DAMPEN_SF_HIGH,
    }

    guarded = _run(batch, **damped)
    monkeypatch.setattr(
        losses_mod, "_finite_blend_component", _unguarded_blend_component,
    )
    parent = _run(batch, **damped)

    assert set(guarded) - {"disarmed_nonfinite_terms"} == set(parent) - {
        "disarmed_nonfinite_terms",
    }
    for key in guarded:
        if key == "disarmed_nonfinite_terms":
            continue
        assert _f(guarded, key) == _f(parent, key), key
    # The comparison is only worth anything over a blend that actually ran.
    assert _f(guarded, "wdl_ce") > 0.0


def test_the_end_to_end_parity_check_can_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NON-VACUITY for the end-to-end parity: a blend that DOES move the value
    loss must be caught, or that test is comparing one code path with itself."""
    def _leans_on_the_fallback(
        probs: torch.Tensor | None, *, raw: torch.Tensor | None = None,
        weight: torch.Tensor, fallback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        del raw
        if probs is None:
            return fallback, None, None
        return (0.5 * weight) * probs + (1.0 - 0.5 * weight) * fallback, None, None

    batch = _dampened_batch()
    damped = {
        "sf_search_dampen_sf_low": _DAMPEN_SF_LOW,
        "sf_search_dampen_sf_high": _DAMPEN_SF_HIGH,
    }
    guarded = _run(batch, **damped)
    monkeypatch.setattr(losses_mod, "_finite_blend_component", _leans_on_the_fallback)
    moved = _run(batch, **damped)

    assert _f(guarded, "wdl_ce") != _f(moved, "wdl_ce")


def test_absent_fields_are_untouched() -> None:
    """The gen-0 shape, still the field-absence path and still finite. The full
    version of this claim lives in
    `test_losses_zero_weight_disarm.py::test_absent_sf_fields_leave_the_value_blend_finite_at_a_live_frac`;
    this only pins that the new code path does not change it."""
    losses = _run(_base_batch())
    assert math.isfinite(_f(losses, "wdl_ce"))
    assert math.isfinite(_f(losses, "total"))
    assert _f(losses, "disarmed_nonfinite_terms") == 0.0


# --------------------------------------------------------------------------
# 4. `disarmed_nonfinite_terms` -- what the zero-weight guard now swallows.
# --------------------------------------------------------------------------


def _poisoned_categorical_batch() -> Batch:
    """`categorical_t` NaN over an EMPTY `has_categorical` -- the empty-mask
    route, which `masked_mean` reports as NaN rather than 0.0 because its
    numerator is ``(x * mask).sum()``."""
    batch = _base_batch()
    batch["categorical_t"] = torch.full((_B, 4), _NAN)
    batch["has_categorical"] = torch.zeros((_B,), dtype=torch.float32)
    return batch


def test_a_poisoned_zero_weighted_term_is_counted() -> None:
    losses = _run(_poisoned_categorical_batch(), w_categorical=0.0)
    assert _f(losses, "disarmed_nonfinite_terms") == 1.0
    assert math.isfinite(_f(losses, "total")), "the #458 guard should still hold"


def test_a_clean_batch_reports_no_disarmed_terms() -> None:
    """Zero-weighted terms EXIST in this call (`w_sf_eval` is 0.0, production's
    value), so a counter that simply never incremented would also read 0.0 --
    which is why the positive case above is the load-bearing one."""
    losses = _run(_sf_batch(_UNCLAIMED_FINITE))
    assert _f(losses, "disarmed_nonfinite_terms") == 0.0


def test_a_nonfinite_term_at_a_live_weight_is_not_counted() -> None:
    """⚑ "DISARMED" MEANS THE WEIGHT IS ZERO. At a live weight the NaN reaches
    `total` and the existing non-finite-gradient guard sees it, so counting it
    here would double-report the loud case and dilute the silent one."""
    losses = _run(_poisoned_categorical_batch(), w_categorical=0.5)
    assert _f(losses, "disarmed_nonfinite_terms") == 0.0
    assert math.isnan(_f(losses, "total")), "the poison must actually poison"


def test_the_production_shape_reaches_both_halves() -> None:
    """⚑ ONE BATCH, BOTH ITEMS, AT THE LIVE YAML'S KNOB VALUES.

    An unclaimed NaN `sf_wdl` row under `sf_wdl_conf_power: 1.0` reaches TWO
    places. `_compute_sf_wdl_mask` multiplies the row mask by
    ``(1 - p_draw)^power``, and ``0.0 * nan`` makes the aux `sf_eval` term NaN
    -- deliberately NOT sanitised (see the sibling test for the other weight).
    At `w_sf_eval: 0.0` the term is disarmed, and the two counts below are then
    the only things that say the shard is dirty. The value TARGET is the other
    place, and it has no such escape: `w_wdl` is 1.0.
    """
    losses = _run(
        _sf_batch(_UNCLAIMED_NAN), sf_wdl_conf_power=_PROD_SF_WDL_CONF_POWER,
    )
    assert math.isnan(_f(losses, "sf_eval_ce")), "the aux term is the dirty-data signal"
    assert _f(losses, "disarmed_nonfinite_terms") == 1.0
    assert _f(losses, "blend_unclaimed_nonfinite_rows") == 2.0
    assert math.isfinite(_f(losses, "wdl_ce")), "the value target must be clean"
    assert math.isfinite(_f(losses, "total"))


def test_the_aux_head_is_loud_at_the_main_yaml_weight() -> None:
    """⚑⚑ THE SAME ROW AT `w_sf_eval: 0.10`, WHICH IS WHAT `main`'s YAML SETS.

    The two yamls disagree, and the disagreement is the whole reason this test
    exists as a sibling rather than a replacement:

      * live production (`configs/pbt2_small.yaml` on the running trial's tree)
        runs `w_sf_eval: 0.0` -- the term is disarmed and COUNTED;
      * `main`'s `configs/pbt2_small.yaml:863` runs `w_sf_eval: 0.10` -- the
        term is IN `total`.

    Deliberately left un-sanitised in BOTH regimes, and this is the branch that
    justifies it: at any non-zero weight the NaN reaches `total`, the
    non-finite-gradient guard in `_run_optimizer_step` trips, the step is
    SKIPPED and a warning is logged. That is loud. Sanitising the aux head's
    input would instead fabricate a finite CE out of a corrupt label and train
    on it quietly -- strictly worse than either regime.

    ⚑ The VALUE TARGET is finite in both, which is this PR's claim and is
    independent of `w_sf_eval` entirely.
    """
    losses = _run(
        _sf_batch(_UNCLAIMED_NAN),
        sf_wdl_conf_power=_PROD_SF_WDL_CONF_POWER,
        w_sf_eval=_MAIN_YAML_W_SF_EVAL,
    )
    assert math.isnan(_f(losses, "sf_eval_ce"))
    assert math.isnan(_f(losses, "total")), "a live weight must reach `total` -- that is LOUD"
    # Not counted as disarmed: at a live weight the gradient guard is the
    # instrument, and double-reporting it would dilute the silent case.
    assert _f(losses, "disarmed_nonfinite_terms") == 0.0
    # The blend's own row count is weight-independent, so the shard defect is
    # still reported at either weight.
    assert _f(losses, "blend_unclaimed_nonfinite_rows") == 2.0
    assert math.isfinite(_f(losses, "wdl_ce")), "the value target must be clean either way"


# --------------------------------------------------------------------------
# 5. The CONSUMER announces it -- once per iteration, off its own accumulator.
# --------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Two parameter groups so `Trainer`'s aurora/adamw split has something to
    do; the forward is a stub because `compute_loss` is patched out."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32, device=self.head.weight.device),
        }


def _trainer(tmp_path: Path) -> Trainer:
    return Trainer(
        _TinyModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )


def _drive(
    trainer: Trainer, monkeypatch: pytest.MonkeyPatch, *, disarmed: float, steps: int,
) -> None:
    head = cast(torch.Tensor, trainer.model.head.weight)  # pyright: ignore[reportAttributeAccessIssue]

    def fake_compute_loss(out: Any, batch: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del out, batch, kwargs
        zero = torch.zeros(())
        return {
            "total": (head * head).sum(),
            "disarmed_nonfinite_terms": torch.tensor(disarmed),
            **dict.fromkeys(trainer_mod._LOSS_KEY_TO_METRIC_FIELD, zero),
        }

    monkeypatch.setattr(trainer_mod, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})
    monkeypatch.setattr(
        trainer,
        "_iter_prefetched_batches",
        lambda *_args, **_kwargs: iter([{"x": torch.zeros((1, 4, 8, 8))}] * 64),
    )
    trainer.train_steps(cast(Any, None), batch_size=1, steps=steps)


def _disarm_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and "zero-weighted loss component" in record.getMessage()
    ]


def test_the_trainer_warns_once_per_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑ ANNOUNCED FROM THE CONSUMER'S OWN ACCUMULATOR. `train_steps` reads the
    value out of the `sums` IT built over the iteration's microbatches, not out
    of a number handed down by `compute_loss` -- and it emits ONE line for the
    whole iteration however many steps carried the NaN."""
    caplog.set_level(logging.WARNING)
    _drive(_trainer(tmp_path), monkeypatch, disarmed=1.0, steps=4)

    warnings = _disarm_warnings(caplog)
    assert len(warnings) == 1, warnings
    # The accumulated count, not the per-microbatch one: 4 steps x 1 term.
    assert "4 zero-weighted loss component" in warnings[0]


def test_the_trainer_is_silent_on_a_clean_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """NON-VACUITY for the test above: a warning that fired unconditionally
    would pass it."""
    caplog.set_level(logging.WARNING)
    _drive(_trainer(tmp_path), monkeypatch, disarmed=0.0, steps=4)

    assert _disarm_warnings(caplog) == []


# --------------------------------------------------------------------------
# 6. The SAME chain with NOTHING stubbed between the two ends.
# --------------------------------------------------------------------------


class _RealHeadModel(nn.Module):
    """Emits the heads `compute_loss` reads, from real parameters.

    A stub `compute_loss` cannot catch a key rename, because the test writes
    both sides of the name. This model exists so the REAL `compute_loss` runs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.head = nn.Linear(1, _ACTIONS)
        self.wdl_head = nn.Linear(1, 3)
        self.cat_head = nn.Linear(1, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        driver = x.reshape(x.shape[0], -1)[:, :1]
        return {
            "policy": self.head(driver),
            "wdl": self.wdl_head(driver),
            "categorical": self.cat_head(driver),
        }


def _real_loss_batch() -> Batch:
    """A poisoned batch for the REAL `compute_loss`, on the empty-mask route."""
    batch = _poisoned_categorical_batch()
    batch["x"] = torch.ones((_B, 1, 1, 1), dtype=torch.float32)
    return batch


def _drive_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, batch: Batch, steps: int,
) -> None:
    trainer = Trainer(
        _RealHeadModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )
  # ⚑ SET THE WEIGHT ATTRIBUTES, NOT `_loss_kwargs`. That is a PROPERTY built
  # from them, and it is the production path -- substituting the dict would stub
  # out the very assembly this test exists to leave real. `w_categorical = 0.0`
  # is the disarm under test; `w_policy`/`w_wdl` stay at their defaults so
  # `total` has a real gradient and the optimizer step is a real one.
    monkeypatch.setattr(trainer, "w_categorical", 0.0)
    monkeypatch.setattr(trainer, "w_sf_eval", 0.0)
    monkeypatch.setattr(trainer, "sf_wdl_frac", _SF_FRAC)
    monkeypatch.setattr(trainer, "search_wdl_frac", _SEARCH_FRAC)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, b: {})
    monkeypatch.setattr(
        trainer, "_iter_prefetched_batches",
        lambda *_a, **_k: iter([{k: v.clone() for k, v in batch.items()} for _ in range(64)]),
    )
    trainer.train_steps(cast(Any, None), batch_size=_B, steps=steps)


def test_the_whole_chain_carries_the_key_with_nothing_stubbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑⚑ THE KEY IS WRITTEN BY `compute_loss` AND READ BY `train_steps`, AND
    NOTHING BETWEEN THEM IS STUBBED HERE.

    The sibling test above monkeypatches `compute_loss`, so it writes BOTH ends
    of the name: rename the dict key on either side and it still passes while the
    warning is dead. That is this repo's signature defect wearing a test's
    clothes. This one runs the real `compute_loss`, the real
    `_extract_loss_scalars`, the real per-microbatch accumulation into `sums`
    and the real `train_steps` read -- so the string only appears once in the
    production path, and a rename at EITHER end fails here.
    """
    caplog.set_level(logging.WARNING)
    _drive_real(tmp_path, monkeypatch, batch=_real_loss_batch(), steps=3)

    warnings = _disarm_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert "3 zero-weighted loss component" in warnings[0]


def test_the_whole_chain_is_silent_on_a_clean_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """NON-VACUITY for the integration test: the same real chain, clean data."""
    caplog.set_level(logging.WARNING)
    clean = _real_loss_batch()
    clean["categorical_t"] = torch.full((_B, 4), 0.25)
    _drive_real(tmp_path, monkeypatch, batch=clean, steps=3)

    assert _disarm_warnings(caplog) == []


def _blend_row_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and "value-label row reading" in record.getMessage()
    ]


@pytest.mark.parametrize("case", _NON_FINITE_LABELS, ids=_label_id)
def test_the_whole_chain_announces_unclaimed_non_finite_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    case: tuple[str, list[float]],
) -> None:
    """The row counter, end to end, for all three corrupt encodings. `-inf` is
    the one that reaches this line only because the finiteness test is taken on
    the RAW tensor."""
    _, row = case
    caplog.set_level(logging.WARNING)
    batch = _real_loss_batch()
    batch["categorical_t"] = torch.full((_B, 4), 0.25)
    batch["sf_wdl"] = _labels(row)
    batch["has_sf_wdl"] = _claimed_mask()
    _drive_real(tmp_path, monkeypatch, batch=batch, steps=2)

    warnings = _blend_row_warnings(caplog)
    assert len(warnings) == 1, warnings
    # 2 unclaimed rows per microbatch x 2 steps.
    assert "4 value-label row reading" in warnings[0]


@pytest.mark.parametrize(
    "key", ["disarmed_nonfinite_terms", "blend_unclaimed_nonfinite_rows"],
)
def test_a_count_key_never_reaches_train_metrics(key: str) -> None:
    """⚑ UNITS PIN for the eval-path trap named at `_compute_metrics`'s
    accumulation site.

    Neither counter is in `_RAW_SUM_LOSS_KEYS`, so `_compute_metrics` scales it
    by `n_rows` and it stops being a count. That is inert TODAY only because
    `_loss_sums_to_metric_kwargs` drops every key that is not a `TrainMetrics`
    field -- the scaled value is computed and thrown away. Give either key a
    field WITHOUT also giving it `_RAW_COUNT_METRIC_FIELDS` membership and the
    published column silently becomes rows-times-count. Asserting the absence is
    what turns that comment into something enforceable; the assertion message is
    the instruction for whoever trips it.

    ⚑ NOT a request to leave them unpublished forever. It is a request to
    publish them through `_RAW_COUNT_METRIC_FIELDS`, which is the path that
    carries the right units -- and which puts them in `_RAW_SUM_LOSS_KEYS`, so
    the first assertion below stops being true at the same moment as the second.
    """
    fields = {f.name for f in dataclasses.fields(trainer_mod.TrainMetrics)}
    published = trainer_mod._loss_sums_to_metric_kwargs({key: 7.0}, 1.0)
    assert key not in fields, (
        f"{key} gained a TrainMetrics field. Add it to _RAW_COUNT_METRIC_FIELDS "
        "in the SAME change, or `_compute_metrics` publishes rows-times-count."
    )
    assert key not in trainer_mod._RAW_SUM_LOSS_KEYS
    assert key not in published
