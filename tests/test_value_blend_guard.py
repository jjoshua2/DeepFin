"""The SF-to-outcome fallback, measured, and the guard that refuses it.

⚑ The first test here does NOT test the guard. It tests the CLAIM the guard
exists for, straight through the production `compute_loss`, by intercepting the
value target the loss actually builds. If that test ever stops failing under
mutation, the guard is defending a defect that no longer exists — and a guard
whose premise has evaporated is worse than no guard, because it reads as
protection.
"""
from __future__ import annotations

import re
from typing import Any

import pytest
import torch

from chess_anti_engine.train import losses as losses_module
from chess_anti_engine.train.losses import compute_loss, normalize_value_blend_fracs
from chess_anti_engine.train.value_blend_guard import (
    ValueBlendMisconfigured,
    assert_no_silent_outcome_fallback,
    assert_pid_cannot_reassert_sf_wdl,
    value_blend_readout,
)
from chess_anti_engine.tune.trainable_metrics import _dynamic_sf_wdl_weight

BATCH = 8
POLICY = 32


def _batch(*, with_sf_wdl: bool) -> dict[str, torch.Tensor]:
    """A minimal batch shaped like an lc0-derived row set.

    lc0 rows carry `search_wdl` and `has_search_wdl` and NO `sf_wdl` —
    `with_sf_wdl=True` adds the production-shaped columns so the same test can
    show the leak closing when every row IS labelled.
    """
    torch.manual_seed(0)
    batch: dict[str, torch.Tensor] = {
        "x": torch.zeros(BATCH, 175, 8, 8),
        "policy_t": torch.full((BATCH, POLICY), 1.0 / POLICY),
        "wdl_t": torch.zeros(BATCH, dtype=torch.int64),
        "has_policy": torch.ones(BATCH),
  # Deliberately not a one-hot and not degenerate, so the search component is
  # distinguishable from the outcome component in the built target.
        "search_wdl": torch.tensor([[0.5, 0.3, 0.2]]).repeat(BATCH, 1),
        "has_search_wdl": torch.ones(BATCH),
    }
    if with_sf_wdl:
        batch["sf_wdl"] = torch.tensor([[0.2, 0.5, 0.3]]).repeat(BATCH, 1)
        batch["has_sf_wdl"] = torch.ones(BATCH)
    return batch


def _outputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "policy": torch.randn(BATCH, POLICY),
        "wdl": torch.randn(BATCH, 3),
    }


def _captured_wdl_target(
    monkeypatch: pytest.MonkeyPatch,
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    **kwargs: Any,
) -> torch.Tensor:
    """The value target `compute_loss` built, taken from the real call.

    `compute_loss` never returns the blended target, so it is intercepted at
    `soft_cross_entropy` and selected by the IDENTITY of the logits argument —
    the wdl head's — rather than by call order, which changes with which heads
    are present.
    """
    seen: list[torch.Tensor] = []
    original = losses_module.soft_cross_entropy

    def spy(logits: torch.Tensor, target: torch.Tensor, *args: Any, **kw: Any) -> torch.Tensor:
        if logits is outputs["wdl"]:
            seen.append(target.detach().clone())
        return original(logits, target, *args, **kw)

    monkeypatch.setattr(losses_module, "soft_cross_entropy", spy)
    compute_loss(outputs, batch, **kwargs)
    assert len(seen) == 1, f"expected exactly one wdl target, saw {len(seen)}"
    return seen[0]


def _decompose(target: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    """Solve the built target back into its component SHARES.

    ⚑ The obvious shortcut — take the search share off one coordinate and call
    the rest `1 - search` — is WRONG, and it was in this file until a mutation
    that deleted the whole fallback branch failed to fail the test. `1 - x`
    silently assumes the components still sum to 1, which is exactly the
    property a broken blend stops having. So the coefficients are solved for
    directly (least squares over the three WDL coordinates against the actual
    component vectors) and `total` is returned alongside, un-normalised, so a
    blend that loses mass reads as lost mass instead of as an outcome share.
    """
    import numpy as np

    one_hot = np.eye(3)[int(batch["wdl_t"][0].item())]
    names = ["outcome"]
    columns = [one_hot]
    if "sf_wdl" in batch:
        names.append("sf")
        columns.append(batch["sf_wdl"][0].numpy().astype(np.float64))
    names.append("search")
    columns.append(batch["search_wdl"][0].numpy().astype(np.float64))
    basis = np.stack(columns, axis=1)
    coeffs, *_ = np.linalg.lstsq(basis, target[0].numpy().astype(np.float64), rcond=None)
    shares = dict(zip(names, (float(c) for c in coeffs), strict=True))
    residual = float(np.abs(basis @ coeffs - target[0].numpy()).max())
    assert residual < 1e-5, f"target is not a combination of its components ({residual})"
    shares["total"] = float(target[0].sum().item())
    return shares


def test_missing_sf_wdl_silently_moves_the_sf_share_onto_the_game_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ THE DEFECT ITSELF, measured through the production loss."""
    batch, outputs = _batch(with_sf_wdl=False), _outputs()
    kwargs = {"sf_wdl_frac": 0.50, "search_wdl_frac": 0.20}
    target = _captured_wdl_target(monkeypatch, batch, outputs, **kwargs)

  # game_frac is 1 - 0.5 - 0.2 = 0.30. The realized outcome mass is 0.80,
  # because the entire 0.50 SF share fell back onto the same one-hot.
    shares = _decompose(target, batch)
    assert shares["outcome"] == pytest.approx(0.80, abs=1e-5)
    assert shares["search"] == pytest.approx(0.20, abs=1e-5)
  # The blend is still a distribution — the SF share MOVED, it did not vanish.
  # Without this the "0.80 outcome" reading cannot be told from a target that
  # simply dropped 0.50 of its mass.
    assert shares["total"] == pytest.approx(1.0, abs=1e-5)
    _sf, _search, game_frac = normalize_value_blend_fracs(0.50, 0.20)
    assert game_frac == pytest.approx(0.30)


def test_the_override_puts_the_outcome_mass_back_where_the_config_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lc0 config: sf_wdl_frac 0 and the share handed to search."""
    batch, outputs = _batch(with_sf_wdl=False), _outputs()
    target = _captured_wdl_target(
        monkeypatch, batch, outputs, sf_wdl_frac=0.0, search_wdl_frac=0.70,
    )
    shares = _decompose(target, batch)
    assert shares["outcome"] == pytest.approx(0.30, abs=1e-5)
    assert shares["search"] == pytest.approx(0.70, abs=1e-5)
    assert shares["total"] == pytest.approx(1.0, abs=1e-5)


def test_labelled_rows_do_not_leak_so_production_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same weights, rows that DO carry sf_wdl: outcome mass stays at 0.30.

    This is what stops the guard being a change to production behaviour: the
    leak is a property of the DATA, not of the weights.
    """
    batch, outputs = _batch(with_sf_wdl=True), _outputs()
    target = _captured_wdl_target(
        monkeypatch, batch, outputs, sf_wdl_frac=0.50, search_wdl_frac=0.20,
    )
    shares = _decompose(target, batch)
    assert shares["outcome"] == pytest.approx(0.30, abs=1e-5)
    assert shares["sf"] == pytest.approx(0.50, abs=1e-5)
    assert shares["search"] == pytest.approx(0.20, abs=1e-5)
    assert shares["total"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    ("sf", "search", "expected"),
    [
        (0.5, 0.2, (0.5, 0.2, 0.3)),
        (0.0, 0.7, (0.0, 0.7, 0.3)),
        (-1.0, 0.4, (0.0, 0.4, 0.6)),
  # Over-subscribed: renormalised, and the outcome share is squeezed to zero.
        (0.8, 0.4, (0.8 / 1.2, 0.4 / 1.2, 0.0)),
    ],
)
def test_normalize_value_blend_fracs(
    sf: float, search: float, expected: tuple[float, float, float],
) -> None:
    assert normalize_value_blend_fracs(sf, search) == pytest.approx(expected)


def test_normalization_is_the_one_compute_loss_uses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard's arithmetic IS the loss's, on the renormalising branch too.

    A guard that keeps its own copy of the criterion can pass while the two
    drift. Rather than assert they are the same function, this drives the
    over-subscribed branch through `compute_loss` and checks the target agrees
    with `normalize_value_blend_fracs`.
    """
    batch, outputs = _batch(with_sf_wdl=False), _outputs()
    target = _captured_wdl_target(
        monkeypatch, batch, outputs, sf_wdl_frac=0.8, search_wdl_frac=0.4,
    )
    sf_frac, search_frac, game_frac = normalize_value_blend_fracs(0.8, 0.4)
    assert game_frac == 0.0
    shares = _decompose(target, batch)
  # The renormalised SF share still lands on the outcome (nothing labels it),
  # so the outcome coefficient is the renormalised sf_frac and not game_frac.
    assert shares["outcome"] == pytest.approx(sf_frac, abs=1e-5)
    assert shares["search"] == pytest.approx(search_frac, abs=1e-5)
    assert shares["total"] == pytest.approx(1.0, abs=1e-5)


def test_guard_raises_on_the_leak_and_names_the_number() -> None:
    readout = value_blend_readout(
        sf_wdl_frac=0.50, search_wdl_frac=0.20, sf_wdl_rows=0.0, batch_rows=512.0,
    )
    assert readout.leaked_to_outcome == pytest.approx(0.50)
    assert readout.outcome_borne_frac == pytest.approx(0.80)
    with pytest.raises(ValueBlendMisconfigured, match=re.escape("0.5000 of the WDL")):
        assert_no_silent_outcome_fallback(readout)


def test_guard_passes_on_the_override() -> None:
    readout = value_blend_readout(
        sf_wdl_frac=0.0, search_wdl_frac=0.70, sf_wdl_rows=0.0, batch_rows=512.0,
    )
    assert readout.leaked_to_outcome == 0.0
    assert readout.outcome_borne_frac == pytest.approx(0.30)
    assert_no_silent_outcome_fallback(readout)


def test_guard_passes_when_every_row_carries_a_label() -> None:
    """Production's own shape must not trip the guard."""
    readout = value_blend_readout(
        sf_wdl_frac=0.50, search_wdl_frac=0.20, sf_wdl_rows=512.0, batch_rows=512.0,
    )
    assert readout.leaked_to_outcome == 0.0
    assert_no_silent_outcome_fallback(readout)


def test_guard_catches_a_partial_leak() -> None:
    """Half the rows labelled leaks half the SF share, and is not tolerated."""
    readout = value_blend_readout(
        sf_wdl_frac=0.50, search_wdl_frac=0.20, sf_wdl_rows=256.0, batch_rows=512.0,
    )
    assert readout.leaked_to_outcome == pytest.approx(0.25)
    with pytest.raises(ValueBlendMisconfigured):
        assert_no_silent_outcome_fallback(readout)


def test_pid_ramp_guard_matches_the_controller_it_describes() -> None:
    """⚑ VERIFIED AGAINST `_dynamic_sf_wdl_weight`, not against its docstring.

    The claim the lc0 config rests on is "sf_wdl_frac: 0.0 means the PID cannot
    re-raise the SF share". That is only true if the controller returns None —
    the value `trainable_config_ops` treats as "leave the trainer alone" — at
    EVERY regret it can reach, not merely at the one someone tried. Swept
    across the band, including the endpoints and a regret below the floor.
    """
    for regret in (0.0, 0.0075, 0.05, 0.10, 0.35, 0.70, 1.0, 5.0):
        assert _dynamic_sf_wdl_weight(
            sf_wdl_start=0.0,
            sf_wdl_floor=0.45,
            sf_wdl_floor_at_regret=0.10,
            regret_max=0.70,
            wdl_regret_used=regret,
        ) is None, f"the ramp produced a weight at regret={regret}"

  # ...and it DOES produce one when the start is non-zero, so the test above
  # is not passing because the function is inert.
    assert _dynamic_sf_wdl_weight(
        sf_wdl_start=0.50,
        sf_wdl_floor=0.45,
        sf_wdl_floor_at_regret=0.10,
        regret_max=0.70,
        wdl_regret_used=0.70,
    ) == pytest.approx(0.50)


def test_pid_ramp_assertion_rejects_a_non_zero_start_or_floor() -> None:
    assert_pid_cannot_reassert_sf_wdl(sf_wdl_frac=0.0, sf_wdl_frac_floor=0.0)
    with pytest.raises(ValueBlendMisconfigured, match=re.escape("sf_wdl_frac=0.5")):
        assert_pid_cannot_reassert_sf_wdl(sf_wdl_frac=0.5, sf_wdl_frac_floor=0.0)
    with pytest.raises(ValueBlendMisconfigured, match=re.escape("sf_wdl_frac_floor=0.45")):
        assert_pid_cannot_reassert_sf_wdl(sf_wdl_frac=0.0, sf_wdl_frac_floor=0.45)
