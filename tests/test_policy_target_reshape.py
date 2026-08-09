"""``policy_target_temp`` / ``policy_target_sf_blend``: reshaping the MAIN policy target.

Both knobs default to an EXACT no-op, and that is the property most worth testing:
they ship default-off into a live training loop, so anything other than bit-identity
at the defaults is a silent production change.

The second load-bearing property is the SF-blend gate. ``sf_policy_t`` is absent on
many rows and an absent target is an ALL-ZERO vector, not a missing key. Blending that
in ungated would scale the row's real target toward zero -- a corrupted target that no
loss curve would reveal, because the loss would still go down. This repo's signature
defect is a value accepted and then silently ignored; this is the same shape with the
sign flipped, so the gate gets a test with a worked counterexample.
"""
from __future__ import annotations

import pytest
import torch

from chess_anti_engine.train.losses import reshape_main_policy_target

WIDTH = 8


def _batch(sf: torch.Tensor | None, has_sf: torch.Tensor | None) -> dict[str, torch.Tensor]:
    b: dict[str, torch.Tensor] = {}
    if sf is not None:
        b["sf_policy_t"] = sf
    if has_sf is not None:
        b["has_sf_policy"] = has_sf
    return b


def _target(rows: list[list[float]]) -> torch.Tensor:
    t = torch.tensor(rows, dtype=torch.float32)
    return t / t.sum(dim=-1, keepdim=True)


def test_defaults_are_bit_identical() -> None:
    """THE CONTROL: shipping default-off must not perturb a single bit."""
    pol = _target([[5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sf = _target([[0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0]])
    out = reshape_main_policy_target(
        pol, batch=_batch(sf, torch.tensor([1.0])), width=WIDTH, temp=1.0, sf_blend=0.0,
    )
    assert out is pol, "the default path must not even allocate a new tensor"


def test_temperature_flattens_without_resurrecting_zeros() -> None:
    """T>1 must move mass toward the middle of the SUPPORT, not onto zeros.

    The moves this target zeroes were measured to lose a median 538cp, so a
    reshape that lifted them off zero would be importing known-bad moves.
    """
    pol = _target([[8.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = reshape_main_policy_target(
        pol, batch={}, width=WIDTH, temp=1.30, sf_blend=0.0,
    )
    assert torch.allclose(out.sum(-1), torch.ones(1), atol=1e-6)
    assert (out[0, 3:] == 0.0).all(), "temperature lifted a zeroed move off zero"
    ent = lambda p: float(-(p * torch.log(p.clamp_min(1e-30))).sum())  # noqa: E731
    assert ent(out[0]) > ent(pol[0]), "T>1 must RAISE entropy"
    assert int(out[0].argmax()) == int(pol[0].argmax()), "ordering must be preserved"


def test_temperature_below_one_sharpens() -> None:
    """Negative control on direction: the knob is a temperature, not a magnitude."""
    pol = _target([[8.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = reshape_main_policy_target(pol, batch={}, width=WIDTH, temp=0.70, sf_blend=0.0)
    ent = lambda p: float(-(p * torch.log(p.clamp_min(1e-30))).sum())  # noqa: E731
    assert ent(out[0]) < ent(pol[0])


def test_sf_blend_moves_a_labelled_row_toward_the_sf_target() -> None:
    pol = _target([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sf = _target([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = reshape_main_policy_target(
        pol, batch=_batch(sf, torch.tensor([1.0])), width=WIDTH, temp=1.0, sf_blend=0.30,
    )
    assert out[0, 1].item() == pytest.approx(0.30, abs=1e-6)
    assert out[0, 0].item() == pytest.approx(0.70, abs=1e-6)


def test_an_unlabelled_row_keeps_its_target_exactly() -> None:
    """THE GATE. Worked counterexample for the ungated version.

    Row 1 has NO SF label, so `sf_policy_t` is all zeros and `has_sf_policy` is 0.
    Ungated, `0.7*pol + 0.3*0` renormalises back to `pol` -- which looks harmless
    until you notice the row is then trained on a target that had 30% of its mass
    deleted and restored by normalisation, i.e. it survives only by accident of the
    renormalise. Any variant that does not renormalise, or that blends AFTER the
    temperature, corrupts it outright. Pin the invariant instead of the arithmetic.
    """
    pol = _target([[4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sf = torch.zeros_like(pol)
    sf[0] = _target([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])[0]
    out = reshape_main_policy_target(
        pol, batch=_batch(sf, torch.tensor([1.0, 0.0])), width=WIDTH,
        temp=1.0, sf_blend=0.30,
    )
    assert torch.allclose(out[1], pol[1], atol=0, rtol=0), (
        "an SF-unlabelled row was altered by the blend; its stored sf_policy_t is "
        "all zeros, so this is the target being scaled toward nothing"
    )
    assert out[0, 3].item() == pytest.approx(0.30, abs=1e-6), "labelled row must blend"


def test_a_missing_sf_key_is_not_an_all_zero_blend() -> None:
    """`sf_policy_t` absent from the batch entirely must also be a no-op."""
    pol = _target([[4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = reshape_main_policy_target(
        pol, batch={}, width=WIDTH, temp=1.0, sf_blend=0.30,
    )
    assert torch.allclose(out, pol, atol=0, rtol=0)


def test_an_integer_presence_flag_does_not_truncate_the_mix() -> None:
    """`has_*` flags are stored as uint8. A gate left in an int dtype would floor
    the blend weight to 0 and turn the whole arm into a silent no-op -- the exact
    class of defect this repo keeps finding."""
    pol = _target([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sf = _target([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = reshape_main_policy_target(
        pol, batch=_batch(sf, torch.ones(1, dtype=torch.uint8)),
        width=WIDTH, temp=1.0, sf_blend=0.30,
    )
    assert out[0, 1].item() == pytest.approx(0.30, abs=1e-6)


def test_rows_stay_normalised_under_both_knobs_together() -> None:
    pol = _target([[6.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    sf = _target([[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    out = reshape_main_policy_target(
        pol, batch=_batch(sf, torch.tensor([1.0, 1.0])), width=WIDTH,
        temp=1.30, sf_blend=0.30,
    )
    assert torch.allclose(out.sum(-1), torch.ones(2), atol=1e-6)
    assert (out >= 0).all()
